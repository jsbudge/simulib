#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scene_tracer import build_tracer, trace_reflector_scene
from simulib.simulib.simulation_functions import db, upsamplePulse, enu2llh, llh2enu, azelToVec, getElevationMap, genChirp
from simulib.simulib.sim_objects import AESA
from simulib.simulib.utils import c0, DTR, _float, _complex_float, getRadarAndEnvironment
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
from sdrparse.SDRV2Parsing import load
from backproject_utils import backprojectPulseStream
import trimesh as tri
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import plotly.io as pio
import plotly.express as px
import requests
from itertools import product, repeat
from io import BytesIO
import mmap
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import array
import time
pio.renderers.default = 'browser'

GECOEFF = 156543.03392
TILE_SIZE = 256


def renderBlock(a_ptimes, a_rp, a_tracer, a_chirps, a_bw_az, a_bw_el, a_points,
                a_point_power, a_nsam, a_fc, a_fs, a_near_range_s, far_range, a_nray_sqrt, transmit_power, rx_gain,
                tx_gain, rec_gain, noise_figure, operating_temp, a_fft_len, add_noise, has_aesa, near_range_cutoff):
    if has_aesa:
        bore = azelToVec(a_rp.tx.az_aesa_iner(a_ptimes), a_rp.tx.el_aesa_iner(a_ptimes))
    else:
        bore = a_rp.tx.boresight(a_ptimes)
    # aesa_bore = azelToVec(a_rp.tx.az_iner(a_ptimes), a_rp.tx.el_iner(a_ptimes))
    txposes = a_rp.txpos(a_ptimes).swapaxes(0, 1).swapaxes(1, 2)
    rxposes = a_rp.rxpos(a_ptimes).swapaxes(0, 1).swapaxes(1, 2)
    a_bd = trace_reflector_scene(a_tracer, a_chirps, txposes, rxposes, a_bw_az, a_bw_el, a_rp.tx.boresight(a_ptimes[0]),
                       bore, a_rp.att(a_ptimes)[:, 1], a_points, a_point_power, a_ptimes, a_nsam,
                       a_fc, a_fs, a_near_range_s, far_range, a_nray_sqrt, a_nray_sqrt, transmit_power, rx_gain, tx_gain,
                       rec_gain, noise_figure, operating_temp, a_fft_len, add_noise=add_noise, add_chirp=True)[0, 0]
    a_bd = np.fft.fft(np.fft.ifft(a_bd, axis=1)[:, near_range_cutoff:a_nsam], a_fft_len, axis=1)
    return a_bd


def saveToFile(a_bd, a_nsam, a_atts, a_copy_path, a_frames, a_scale):
    save_data = np.fft.ifft(a_bd, axis=1, norm='ortho')[:, :a_nsam]
    # save_int_data = np.zeros((*save_data.shape, 2)).astype(np.int16)
    ndata = (save_data.astype(np.complex64).view('(2,)float32') / 10 ** (
            a_atts[:, None, None] / 20))
    ndata = (-1 + (ndata - -a_scale) * 1 / a_scale) * 32768
    save_int_data = ndata.astype(np.int16)
    # save_data /= old_tmax * 32768
    with open(a_copy_path, 'r+b') as fin:
        with mmap.mmap(fin.fileno(), 0) as mm:
            for fr_idx, fr in enumerate(a_frames):
                # ndata = (save_data[fr_idx].astype(np.complex64).view('(2,)float32') / 10**(atts[fr] / 20)).astype(np.int16)
                tmp_data = array.array('h', save_int_data[fr_idx].astype(np.int16).flatten())
                tmp_data.byteswap()
                mm.seek(ref_pts[fr])
                mm.write(bytes(tmp_data))
    return True


def calcGoogleCoords(a_origin: tuple[float, float], a_lats: float | np.ndarray, a_lons: float | np.ndarray,
                     mpp: float):
    zoomlevel = int(np.round(np.log2(GECOEFF * np.cos(a_origin[0] * DTR) / mpp)))

    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    m_scale = 1 << zoomlevel

    pix_coords = np.zeros((len(a_lats), 2))
    sy = np.sin(a_lats * DTR)
    pix_coords[:, 0] = (TILE_SIZE / 2. + a_lons * TILE_SIZE / 360) * m_scale
    pix_coords[:, 1] = ((TILE_SIZE / 2) + .5 * np.log((1 + sy) / (1 - sy)) * -(TILE_SIZE / (2 * np.pi))) * m_scale

    tile_coords = pix_coords // TILE_SIZE

    start_x = int(tile_coords[:, 0].min())
    start_y = int(tile_coords[:, 1].min())

    # Get interpolated values from image for lat/lons
    im_x = pix_coords[:, 0] - start_x * TILE_SIZE
    im_y = pix_coords[:, 1] - start_y * TILE_SIZE
    return tile_coords, zoomlevel, im_x, im_y


def resampleGoogleMap(a_lats: float | np.ndarray, a_lons: float | np.ndarray, mpp: float):
    tile_coords, zoomlevel, im_x, im_y = calcGoogleCoords(((a_lats.max() + a_lats.min()) / 2,
                                                           (a_lons.max() + a_lons.min()) / 2), a_lats, a_lons, mpp)

    start_x = int(tile_coords[:, 0].min())
    start_y = int(tile_coords[:, 1].min())

    tile_width = int(tile_coords[:, 0].max() - tile_coords[:, 0].min()) + 1
    tile_height = int(tile_coords[:, 1].max() - tile_coords[:, 1].min()) + 1

    # Determine the size of the image in pixels
    width, height = TILE_SIZE * tile_width, TILE_SIZE * tile_height

    # Create a new image of the size require
    map_img = Image.new('RGB', (width, height))

    for x, y in tqdm(product(range(tile_width), range(tile_height))):
        url = f'https://mt1.google.com/vt?lyrs=s&x={start_x + x}&y={start_y + y}&z={zoomlevel}'
        response = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0'})
        im_google = Image.open(BytesIO(response.content))
        map_img.paste(im_google, (x * TILE_SIZE, y * TILE_SIZE))

    return map_img, im_x, im_y



if __name__ == "__main__":
    cfig = load_yaml_config('/home/jeff/repo/optix_radartracer/simulib/scripts/anduril_params.yaml')
    npulses = 512
    upsample = 1
    near_range_ppp = .3
    show_plots = True
    save_file = True
    randomize_pts = True

    max_pix = 512
    pix_res: float = 2.
    nray_sqrt = 3000
    nrays = nray_sqrt * nray_sqrt

    start_time = time.time()

    # fnme = '/home/jeff/SDR_DATA/RAW/10282025/SAR_10282025_112741.sar'
    # fnme = '/home/jeff/SDR_DATA/RAW/12112025/SAR_12112025_144929.sar'
    # fnme = '/home/jeff/SDR_DATA/RAW/12192025/SAR_12192025_110646.sar'
    fnme = '/home/jeff/SDR_DATA/RAW/11112025/SAR_11112025_145023.sar'

    # Copy/paste another file into the same directory
    if save_file:
        source_path = Path(fnme)
        source_xml_path = f'{source_path.parent}/{source_path.stem}.xml'
        copy_path = f'{source_path.parent}/{source_path.stem}_copy.sar'
        copy_xml_path = f'{source_path.parent}/{source_path.stem}_copy.xml'

        try:
            shutil.copy2(source_path, copy_path)
            print(f"File copied successfully to {copy_path}")
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except IsADirectoryError:
            print("Destination is a directory.")
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            shutil.copy2(source_xml_path, copy_xml_path)
            print(f"File copied successfully to {copy_xml_path}")
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except IsADirectoryError:
            print("Destination is a directory.")
        except Exception as e:
            print(f"An error occurred: {e}")

    sdr = load(fnme, progress_tracker=True)
    sdr_dataset = 0
    try:
        ival = sdr.intervals[sdr[sdr_dataset].interval]
        path = sdr[sdr_dataset].path
        ref_pts = ival.packet_points[path]
        atts = ival.atts[path]
    except AttributeError:
        ref_pts = sdr[sdr_dataset].packet_points
        atts = sdr[sdr_dataset].atts

    wavelength = c0 / sdr[sdr_dataset].fc
    fs = sdr[sdr_dataset].fs
    fc = sdr[sdr_dataset].fc

    # Design the antenna
    '''az_elements = 100
    el_elements = 100
    ant = AESA(fc, az_elements, el_elements, 1, 1)
    bw_az, bw_el = ant.calc_beamwidth(0., 0.)
    tx_num = sdr[0].trans_num if not sdr.is_v2 else sdr[0].tx_num
    while bw_az > sdr.ant[sdr.port[tx_num].assoc_ant].az_bw * 2:
        az_elements += 1
        ant = AESA(fc, az_elements, 1, 1, 1)
        bw_az, bw_el = ant.calc_beamwidth(0., 0.)
    while bw_el > sdr.ant[sdr.port[tx_num].assoc_ant].el_bw * 2:
        el_elements += 1
        ant = AESA(fc, az_elements, el_elements, 1, 1)
        bw_az, bw_el = ant.calc_beamwidth(0., 0.)'''

    # origin = ()
    if sdr.has_aesa:
        ant = AESA(fc, 1, 1, 1, 1)
        bg, rp = getRadarAndEnvironment(sdr, platform_args=dict(tx_offset=ant.phase_center_offsets, rx_offset=ant.phase_center_offsets))
    else:
        bg, rp = getRadarAndEnvironment(sdr)
    rp.fs = fs
    origin = bg.origin
    bw_az = rp.az_half_bw * 2
    bw_el = rp.el_half_bw * 2

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(0., 0.0, near_range_ppp, 0., upsample)
    aug_nsam = nsam + int(nr * near_range_ppp)

    # This is for backprojection inside of script and shouldn't affect the final file
    grid_ranges = rp.calcRanges(0, 0, 0., 0.)
    # ((gx, gy, gz), bg_transform), grid_width_m, grid_height_m, origin = bg.gridFromSwath(*grid_ranges, beamwidth=rp.az_half_bw)
    grid_height_m = np.sqrt(grid_ranges[1]**2 - rp.pos(rp.gpst)[:, 2].mean()**2) - np.sqrt(grid_ranges[0]**2 - rp.pos(rp.gpst)[:, 2].mean()**2)
    grid_width_m = np.linalg.norm(rp.pos(rp.gpst[0]) - rp.pos(rp.gpst[-1]))
    # near_range_s -= 1e-6
    sample_height_m = grid_height_m * 3.
    sample_width_m = np.linalg.norm(rp.pos(rp.gpst[0]) - rp.pos(rp.gpst[-1])) * 1.1

    chirps = sdr[sdr_dataset].cal_chirp.reshape((1, -1)).astype(_complex_float)
    # chirps = genChirp(nr, fs, fc, sdr[0].bw).reshape((1, -1)).astype(_complex_float)
    mf_chirps = sdr.genReciprocalRipple(0, 0, 0).reshape((1, -1)).astype(_complex_float)
    # mf_chirps = np.fft.fft(chirps, fft_len, axis=1).conj()

    pulse_times = sdr[sdr_dataset].pulse_time


    if show_plots:
        pix_width = min(max_pix, int(sample_width_m / pix_res))
        pix_height = min(max_pix, int(sample_height_m / pix_res))
        (gx, gy, gz), bg_transform = bg.getGrid(origin, along_track_m=grid_width_m, cross_track_m=grid_height_m,
                                                nrows=pix_height, ncols=pix_width, cross_track_angle=bg.cross_track_angle)

    bpj_grid = np.zeros(gx.shape, dtype=_complex_float)


    point_grid_sz = (int(sample_width_m / pix_res), int(sample_height_m / pix_res))
    (bgx, bgy, bgz), bbg_transform = bg.getGrid(origin, along_track_m=sample_width_m, cross_track_m=sample_height_m,
                                            nrows=point_grid_sz[0],
                                            ncols=point_grid_sz[1], cross_track_angle=bg.cross_track_angle)

    # ((gx, gy, gz), bg_transform), grid_width_m, grid_height_m, origin = bg.gridFromSwath()

    # gx, gy, gz = bg.getGrid(origin, along_track_m=grid_width_m + 20., cross_track_m=grid_height_m + 20., nrows=pix_height,
    #                         ncols=pix_width, cross_track_angle=bg.cross_track_angle)
    # gz[:] = gz.mean()

    print('Getting Google Maps grid...')
    res_mpp = max((bgx.max() - bgx.min()) / point_grid_sz[1], (bgy.max() - bgy.min()) / point_grid_sz[0])
    lats, lons, alts = enu2llh(bgx.flatten(), bgy.flatten(), bgz.flatten(), bg.ref)
    im, imx, imy = resampleGoogleMap(lats, lons, res_mpp)
    # Google image is transposed compared to what we expect in SDREnvironment
    im = np.array(im).sum(axis=2).T
    im = ((im - im.min()) * .95 / (im.max() - im.min()))
    npts = nrays

    grid_pts = np.array([bgx[::10, ::10].flatten(), bgy[::10, ::10].flatten(), bgz[::10, ::10].flatten()]).T
    tris_2d = Delaunay(grid_pts[:, :2])
    elevation_mesh = tri.Trimesh(vertices=grid_pts, faces=tris_2d.simplices)
    el_mats = np.zeros((elevation_mesh.triangles.shape[0], 2))
    el_mats[:, 0] = 1e6
    el_mats[:, 1] = .017
    grid_int = RegularGridInterpolator((np.arange(im.shape[0]), np.arange(im.shape[1])), im)



    # Build a trimesh of the elevation map
    # Set points for rays
    if randomize_pts:
        # Rotate uniform points to lay on top of grid
        points = (np.random.rand(npts, 3) * np.array([point_grid_sz[0] - .1, point_grid_sz[1] - .1, 0.]) -
                  np.array([point_grid_sz[0] / 2, point_grid_sz[1] / 2, -1.]))
        points = points @ bbg_transform.T
        plats, plons, _ = enu2llh(*points.T, bg.ref)
        points[:, 2] = getElevationMap(plats, plons) - bg.ref[2]
    else:
        points = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T

    print('Sampling Google Map...')
    plats, plons, _ = enu2llh(*points.T, bg.ref)

    _, _, iimx, iimy = calcGoogleCoords(((lats.max() + lats.min()) / 2, (lons.max() + lons.min()) / 2), plats, plons, res_mpp)
    point_power = grid_int((iimx, iimy)).astype(_float)

    tracer = build_tracer([BaseMesh(elevation_mesh, el_mats, motion_keys=None)], 'anduril_trace.cu', pulse_times=pulse_times)

    print(f'Launching {npts} rays.')
    ptimes = [pulse_times[frame[0]:frame[0] + npulses] for frame in list(zip(*(iter(range(0, len(pulse_times), npulses)),)))]
    '''results = [renderBlock(p, rp, tracer, chirps, bw_az, bw_el, points, point_power, nsam, fc, fs, near_range_s, ranges[-1], nray_sqrt,
                           cfig.ant_params.transmit_power, cfig.ant_params.rx_gain, cfig.ant_params.tx_gain, cfig.ant_params.rec_gain,
                           cfig.ant_params.noise_figure, cfig.ant_params.operating_temperature, fft_len, True) for p in ptimes]'''

    with ThreadPoolExecutor(max_workers=15) as executor:
        ex_map = executor.map(renderBlock, ptimes, repeat(rp), repeat(tracer), repeat(chirps), repeat(bw_az),
                              repeat(bw_el), repeat(points), repeat(point_power), repeat(aug_nsam), repeat(fc), repeat(fs),
                              repeat(near_range_s), repeat(ranges[-1]), repeat(nray_sqrt),
                              repeat(cfig.ant_params.transmit_power), repeat(cfig.ant_params.rx_gain),
                              repeat(cfig.ant_params.tx_gain), repeat(cfig.ant_params.rec_gain),
                              repeat(cfig.ant_params.noise_figure), repeat(cfig.ant_params.operating_temperature),
                              repeat(fft_len), repeat(False), repeat(sdr.has_aesa), repeat(int(nr * near_range_ppp)))

        results = list(ex_map)

    if save_file:
        scale = max([abs(d).max() for d in results])
        print(f'Saving file as {copy_path}...')
        frames = [np.arange(frame[0], min(frame[0] + npulses, len(pulse_times))) for frame in list(zip(*(iter(range(0, len(pulse_times), npulses)),)))]
        att_list = [atts[frame[0]:frame[0] + npulses] for frame in list(zip(*(iter(range(0, len(pulse_times), npulses)),)))]
        with ProcessPoolExecutor(max_workers=15) as executor:
            ex_map = executor.map(saveToFile, results, repeat(nsam), att_list, repeat(copy_path), frames, repeat(scale))
            saves = list(ex_map)

    grids = []
    for bd, frame in tqdm(zip(results, list(zip(*(iter(range(0, len(pulse_times), npulses)),))))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]

        if show_plots:
            '''if save_file:
                new_data = np.zeros((len(ptimes), nsam), dtype=np.complex128)
                # tmp_data = bytes(save_int_data[fr_idx].flatten())
                for fr_idx, fr in enumerate(np.arange(frame[0], frame[0] + len(ptimes))):
                    ndata = bytes(save_int_data[fr_idx].flatten())
                    # ndata = tmp_data[fr_idx:fr_idx + nsam * 4]
                    re_data = array.array('h', ndata)
                    # re_data.byteswap()
                    re_data = np.array(re_data)
                    new_data[fr_idx] = (re_data[:nsam * 2:2] + 1j * re_data[1:nsam * 2:2]) * 10 ** (atts[fr] / 20)
                bd = np.fft.fft(new_data, fft_len, axis=1)'''
            # bd = np.fft.fft((save_int_data[..., 0] + 1j * save_int_data[..., 1]) * 10 ** (31 / 20), fft_len, axis=1)

            rpi_data = upsamplePulse(bd * mf_chirps, fft_len, upsample, is_freq=True,
                                     time_len=nsam).astype(_complex_float)
            grids.append(backprojectPulseStream([rpi_data], [rp.tx.az_aesa_iner(ptimes) if sdr.has_aesa else rp.tx.az_iner(ptimes)],
                                                   [rp.rxpos(ptimes)[:, 0, 0]],
                                                   [rp.txpos(ptimes)[:, 0, 0]], gz, _float(wavelength),
                                                   _float(near_range_s), _float(fs * upsample), _float(rp.az_half_bw),
                                                   gx=gx, gy=gy))
    bpj_grid = sum(grids)

    end_time = time.time()

    print(f"Final timing is {end_time - start_time}s for a {sdr[0].pulse_time[-1] - sdr[0].pulse_time[0]}s collect "
          f"({(end_time - start_time) / (sdr[0].pulse_time[-1] - sdr[0].pulse_time[0])}x).")

    if save_file:
        print(f'File saved as {copy_path}.')

    if show_plots:
        plt.figure('Block Data')
        plt.subplot(2, 1, 1)
        plt.title('Range-Doppler')
        plt.imshow(db(np.fft.fft(rpi_data, axis=0)), extent=(ranges[0], ranges[-1], 0, rpi_data.shape[0]))
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.title('Range-Sample')
        plt.imshow(db(rpi_data), extent=(ranges[0], ranges[-1], 0, rpi_data.shape[0]))
        plt.axis('tight')

        scaled_bpj = abs(bpj_grid)
        scaled_bpj = (scaled_bpj - scaled_bpj.min()) * 30 / scaled_bpj.mean() + 1
        plt.figure('Backprojection vs. Map')
        plt.subplot(2, 1, 1)
        plt.title('Backprojection')
        plt.imshow(db(scaled_bpj), cmap='gray')
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.title('Map')
        plt.imshow(grid_int((imx, imy)).reshape(bgx.shape), cmap='gray')
        # plt.imshow(im, cmap='gray')
        plt.axis('tight')
        plt.show()

        base_pos = rp.pos(pulse_times[::100])

        fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),)
        fig.add_scatter3d(x=base_pos[:, 0], y=base_pos[:, 1], z=base_pos[:, 2])
        fig.add_scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker=dict(color=point_power))
        fig.show()

        from scipy.spatial import delaunay_plot_2d

        # 2. Use SciPy's dedicated plotting utility
        fig, ax = plt.subplots()
        delaunay_plot_2d(tris_2d, ax=ax)

        plt.figure('Chirps')
        plt.plot(db(np.fft.fft(chirps[0], fft_len)))
        plt.plot(db(mf_chirps[0]))

        plt.figure('Data')
        plt.imshow(db(np.fft.ifft(results[0] * mf_chirps, axis=1)[:, :nsam]))
        plt.axis('tight')

