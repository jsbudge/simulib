#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scene_tracer import build_tracer, trace_reflector_scene
from simulib.simulib.simulation_functions import db, upsamplePulse, enu2llh, llh2enu, azelToVec, getElevationMap, genChirp, getRadarCoeff
from simulib.simulib.utils import c0, DTR, _float, _complex_float, getRadarAndEnvironment
from scipy.interpolate import RegularGridInterpolator
from sdrparse.SDRParsing import load
from backproject_utils import backprojectPulseStream, simulatePulseStream
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
import warp as wp
import cupy as cp
from warp_kernels import simple_simulation, runBackproject
import array
import time
import matplotlib.transforms as mtrans
pio.renderers.default = 'browser'

GECOEFF = 156543.03392
TILE_SIZE = 256


def renderBlock(a_ptimes, a_rp, a_tracer, a_chirps, a_bw_az, a_bw_el, a_gridheight,
                a_point_power, a_grid_transform, a_nsam, a_fc, a_fs, a_near_range_s, far_range, a_nray_width, a_nray_height, transmit_power, rx_gain,
                tx_gain, rec_gain, noise_figure, operating_temp, a_fft_len, add_noise, near_range_cutoff):
    """
    Render a block of pulses inside the simulation.
    :param a_ptimes: Pulse times.
    :param a_rp: Instance of SDRPlatform that provides boresight, attitude and position information.
    :param a_tracer: Instance of a raytracing program for CUDA.
    :param a_chirps: Basebanded chirp used for final convolution. Time domain data.
    :param a_bw_az: Half-bandwidth for azimuth in radians.
    :param a_bw_el: Half-bandwidth for elevation in radians.
    :param a_gridheight: (X, Y, Z) positions of points to simulate.
    :param a_point_power: Power values of points to simulate.
    :param a_nsam: Number of samples in the range gate.
    :param a_fc: Center frequency of the chirp in Hz.
    :param a_fs: Sampling frequency of the system in Hz.
    :param a_near_range_s: This is the timing of the near range gate in seconds.
    :param far_range:
    :param a_nray_sqrt: Square root of the number of rays to send out.
    :param transmit_power: Transmit power of antenna in watts.
    :param rx_gain: Receiver gain in dB.
    :param tx_gain: Transmitter gain in dB.
    :param rec_gain: Internal gains from power amplifiers in dB.
    :param noise_figure: Noise figure of the radar.
    :param operating_temp: Operating temperature in K.
    :param a_fft_len: FFT length for convolution with chirp.
    :param add_noise: If True, adds thermal noise to final results.
    :param near_range_cutoff: Amount of data to cut off the beginning of a pulse, in bins. Used to simulate partial pulse
        returns from the beginning of a range gate.
    :return: Frequency spectrum data of simulated radar returns.
    """
    # Swap position axes around to get data into a format the tracer expects
    txposes = a_rp.txpos(a_ptimes).swapaxes(0, 1).swapaxes(1, 2)
    rxposes = a_rp.rxpos(a_ptimes).swapaxes(0, 1).swapaxes(1, 2)
    a_bd = trace_reflector_scene(a_tracer, a_chirps, txposes, rxposes, a_bw_az, a_bw_el,
                                 a_rp.tx.boresight(a_ptimes), a_rp.att(a_ptimes)[:, 1], a_gridheight, a_point_power, a_grid_transform, a_ptimes, a_nsam,
                                 a_fc, a_fs, a_near_range_s, far_range, a_nray_width, a_nray_height, transmit_power, rx_gain, tx_gain,
                                 rec_gain, noise_figure, operating_temp, a_fft_len, add_noise=add_noise, add_chirp=True)[0, 0]
    a_bd = np.fft.fft(np.fft.ifft(a_bd, axis=1)[:, near_range_cutoff:a_nsam], a_fft_len, axis=1)
    return a_bd


def renderWarp(a_rcs, grid_transform, tx, rx, a_az, a_el,
               bw_az, bw_el, near_range_s, fs, wavelength, radar_coeff,
               aug_nsam, near_range_cutoff, chirp_gpu):
    txpos = wp.array(tx, dtype=wp.vec3f, device='cuda:0')
    rxpos = wp.array(rx, dtype=wp.vec3f, device='cuda:0')
    az = wp.array(a_az, dtype=wp.float32, device='cuda:0')
    el = wp.array(a_el, dtype=wp.float32, device='cuda:0')
    idata = wp.zeros((len(a_az), len(chirp_gpu)), dtype=wp.vec2f, device='cuda:0')
    # qdata = wp.zeros((len(p), aug_nsam), dtype=wp.float32, device='cuda:0')
    wp.launch(
        kernel=simple_simulation,
        dim=(a_rcs.shape[0] - 1, a_rcs.shape[1] - 1),
        inputs=[a_rcs, grid_transform, txpos, rxpos, az, el, bw_az, bw_el, near_range_s, fs, 2 * np.pi / wavelength,
                radar_coeff,
                aug_nsam, idata],
    )
    wp.synchronize()
    return cp.asnumpy(
        cp.fft.fft(cp.fft.ifft(cp.fft.fft(cp.array(idata.view(wp.float64)).view(_complex_float), axis=-1) * chirp_gpu,
                               axis=-1)[:, near_range_cutoff:aug_nsam], len(chirp_gpu), axis=-1))



def saveToFile(a_bd, a_nsam, a_atts, a_copy_path, a_frames, a_scale, a_ref_pts):
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
                mm.seek(a_ref_pts[fr])
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


def do_plot(ax, Z, extents, trans_mat):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=extents, clip_on=True)

    transform = mtrans.Affine2D()
    transform.set_matrix(trans_mat)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)


def runSimulation(cfig, fnme):
    npulses = cfig.tracer_params.block_size
    upsample = cfig.tracer_params.upsample
    near_range_ppp = cfig.sim_params.near_range_ppp
    show_plots = cfig.sim_params.show_plots
    save_file = cfig.sim_params.save_file
    max_pix = cfig.sim_params.max_pixels
    pix_res: float = cfig.sim_params.pix_resolution

    start_time = time.time()

    # Copy/paste another file into the same directory
    if save_file:
        # Create a copy that we mess with, so the original data is untouched.
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
    # If working with V2, try this (except case is for V1 data)
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

    # Grab Platform and Environment
    bg, rp = getRadarAndEnvironment(sdr)
    rp.fs = fs
    origin = bg.origin
    # SDR uses 3db beamwidth and simulator uses null-to-null, hence the doubling
    bw_az = rp.az_half_bw * 2
    bw_el = rp.el_half_bw * 2

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(0., 0.0,
                                                                                                     near_range_ppp, 0.,
                                                                                                     upsample)
    near_range_cutoff = int(nr * near_range_ppp)
    aug_nsam = nsam + near_range_cutoff


    # This is for backprojection inside of script and shouldn't affect the final file
    grid_ranges = rp.calcRanges(0, 0, 0., 0.)
    # ((gx, gy, gz), bg_transform), grid_width_m, grid_height_m, origin = bg.gridFromSwath(*grid_ranges, beamwidth=rp.az_half_bw)
    grid_height_m = (np.sqrt(grid_ranges[1] ** 2 - rp.pos(rp.gpst)[:, 2].mean() ** 2) - np.sqrt(
        grid_ranges[0] ** 2 - rp.pos(rp.gpst)[:, 2].mean() ** 2))
    grid_width_m = np.linalg.norm(rp.pos(rp.gpst[0]) - rp.pos(rp.gpst[-1]))

    # Set sampling to be just bigger than expected swath
    sample_height_m = grid_height_m * 1.5
    sample_width_m = np.linalg.norm(rp.pos(rp.gpst[0]) - rp.pos(rp.gpst[-1])) * 1.5

    chirps = sdr[sdr_dataset].cal_chirp.reshape((1, -1)).astype(_complex_float)
    fft_chirp = np.fft.fft(chirps, fft_len, axis=-1)
    pulse_times = sdr[sdr_dataset].pulse_time

    if show_plots:
        _, _, _, _, bpj_near_range_s, _, _, _ = rp.getRadarParams(0., 0.0, 0., 0., upsample)
        # Smaller grid for backprojection inside of script
        pix_width = min(max_pix, int(sample_width_m / pix_res))
        pix_height = min(max_pix, int(sample_height_m / pix_res))
        (gx, gy, gz), bg_transform = bg.getGrid(origin, along_track_m=grid_width_m, cross_track_m=grid_height_m,
                                                nrows=pix_height, ncols=pix_width,
                                                cross_track_angle=bg.cross_track_angle, use_elevation=False)
        # gz[:] = 0.

        bpj_grid = np.zeros(gx.shape, dtype=_complex_float)

    # This is the grid we use to sample points for simulation
    point_grid_sz = min(int(sample_width_m / pix_res), int(sample_height_m / pix_res))
    (bgx, bgy, bgz), bbg_transform = bg.getGrid(origin, along_track_m=sample_width_m, cross_track_m=sample_height_m,
                                                nrows=point_grid_sz, ncols=point_grid_sz,
                                                cross_track_angle=bg.cross_track_angle, use_elevation=False)
    bgz[:] = 0.
    points = np.array([bgx.flatten(), bgy.flatten(), bgz.flatten()])

    print('Getting Google Maps grid...')
    res_mpp = max((bgx.max() - bgx.min()) / point_grid_sz, (bgy.max() - bgy.min()) / point_grid_sz)
    lats, lons, alts = enu2llh(*points, bg.ref)
    im, imx, imy = resampleGoogleMap(lats, lons, res_mpp)
    # Google image is transposed compared to what we expect in SDREnvironment
    im = np.array(im).sum(axis=2).T
    im = ((im - im.min()) * .95 / (im.max() - im.min()))
    grid_int = RegularGridInterpolator((np.arange(im.shape[0]), np.arange(im.shape[1])), im, bounds_error=False, fill_value=0.)

    print('Sampling Google Map...')
    plats, plons, _ = enu2llh(*points, bg.ref)

    # Get power values for random points by using interpolation of Google Maps data
    _, _, iimx, iimy = calcGoogleCoords(((lats.max() + lats.min()) / 2, (lons.max() + lons.min()) / 2), plats, plons,
                                        res_mpp)
    point_power = grid_int((iimx, iimy)).astype(_float).reshape(bgz.shape).T
    # point_power[:] = 0
    # point_power[233, 233] = 10

    # im = grid
    npts = len(point_power)

    print(f'Launching {npts} rays.')
    ptimes = [pulse_times[frame[0]:frame[0] + npulses] for frame in
              list(zip(*(iter(range(0, len(pulse_times), npulses)),)))]

    render_time = time.time()

    rcs = wp.array(point_power, dtype=wp.float32, device='cuda:0')
    radar_coeff = getRadarCoeff(fc, cfig.ant_params.transmit_power, cfig.ant_params.rx_gain, cfig.ant_params.tx_gain,
                                cfig.ant_params.rec_gain)
    idata = wp.zeros((npulses, fft_len), dtype=wp.vec2f, device='cuda:0')
    chirp_gpu = cp.array(fft_chirp, dtype=_complex_float)
    grid_transform = wp.mat33f(bbg_transform)
    results = []

    '''txes = [rp.txpos(p)[:, 0, 0, :] for p in ptimes]
    rxes = [rp.rxpos(p)[:, 0, 0, :] for p in ptimes]
    azes = [rp.az_iner(p) for p in ptimes]
    eles = [rp.el_iner(p) for p in ptimes]

    with ThreadPoolExecutor(max_workers=15) as executor:
        ex_map = executor.map(renderWarp, repeat(rcs), repeat(grid_transform), txes, rxes,
                              azes, eles, repeat(bw_az), repeat(bw_el),
                              repeat(near_range_s), repeat(fs), repeat(wavelength), repeat(radar_coeff), repeat(aug_nsam),
                              repeat(near_range_cutoff),
                              repeat(chirp_gpu))
        results = list(ex_map)'''
    for p in tqdm(ptimes):
        txpos = wp.array(rp.txpos(p)[:, 0, 0, :], dtype=wp.vec3f, device='cuda:0')
        rxpos = wp.array(rp.rxpos(p)[:, 0, 0, :], dtype=wp.vec3f, device='cuda:0')
        az = wp.array(rp.az_iner(p), dtype=wp.float32, device='cuda:0')
        el = wp.array(rp.el_iner(p), dtype=wp.float32, device='cuda:0')
        idata.zero_()
        # qdata = wp.zeros((len(p), aug_nsam), dtype=wp.float32, device='cuda:0')
        wp.launch(
            kernel=simple_simulation,
            dim=(bgx.shape[0] - 1, bgx.shape[1] - 1, 2),
            inputs=[rcs, grid_transform, txpos, rxpos, az, el, bw_az, bw_el, near_range_s, fs, 2 * np.pi / wavelength, radar_coeff,
                    aug_nsam, idata],
        )
        # wp.synchronize()
        results.append(cp.asnumpy(cp.fft.fft(cp.fft.ifft(cp.fft.fft(cp.array(idata.view(wp.float64)).view(_complex_float), axis=-1) * chirp_gpu,
                                              axis=-1)[:len(p), near_range_cutoff:aug_nsam], fft_len, axis=-1)))

    print(f'Rendered simulation in {time.time() - render_time} seconds.')

    if save_file:
        scale = max([abs(d).max() for d in results])
        print(f'Saving file as {copy_path}...')
        frames = [np.arange(frame[0], min(frame[0] + npulses, len(pulse_times))) for frame in
                  list(zip(*(iter(range(0, len(pulse_times), npulses)),)))]
        att_list = [atts[frame[0]:frame[0] + npulses] for frame in
                    list(zip(*(iter(range(0, len(pulse_times), npulses)),)))]
        with ProcessPoolExecutor(max_workers=15) as executor:
            ex_map = executor.map(saveToFile, results, repeat(nsam), att_list, repeat(copy_path), frames, repeat(scale), repeat(ref_pts))
            saves = list(ex_map)

    if show_plots:
        print('Running backprojection of data...')
        mf_chirps = sdr.genMatchedFilter(0, fft_len=fft_len).reshape((1, -1)).astype(
            _complex_float)  # np.fft.fft(chirps, fft_len, axis=1).conj()
        grids = []
        pass_grid = np.stack([bgx, bgy, bgz], axis=-1)
        for bd, frame in tqdm(zip(results, list(zip(*(iter(range(0, len(pulse_times), npulses)),))))):
            ptimes = pulse_times[frame[0]:frame[0] + npulses]
            rpi_data = upsamplePulse(bd * mf_chirps, fft_len, upsample, is_freq=True,
                                     time_len=nsam).astype(_complex_float)
            txpos = rp.txpos(ptimes)[:, 0, 0, :]
            rxpos = rp.rxpos(ptimes)[:, 0, 0, :]
            az = rp.az_iner(ptimes)
            grids.append(runBackproject(rpi_data, pass_grid, txpos, rxpos, az, bw_az, bw_el, near_range_s, fs * upsample, 2 * np.pi / wavelength))
            '''grids.append(backprojectPulseStream([rpi_data], [rp.tx.az_iner(ptimes)],
                                                [rp.rxpos(ptimes)[:, 0, 0]],
                                                [rp.txpos(ptimes)[:, 0, 0]], gz, _float(wavelength),
                                                _float(bpj_near_range_s), _float(fs * upsample), _float(rp.az_half_bw * 2),
                                                gx=gx, gy=gy))'''
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
        glats, glons, _ = enu2llh(*np.array([gx.flatten(), gy.flatten(), gz.flatten()]), bg.ref)

        # Get power values for random points by using interpolation of Google Maps data
        res_mpp = max((gx.max() - gx.min()) / point_grid_sz, (gy.max() - gy.min()) / point_grid_sz)
        _, _, gimx, gimy = calcGoogleCoords(((lats.max() + lats.min()) / 2, (lons.max() + lons.min()) / 2), glats,
                                            glons, res_mpp)
        gpoint_power = grid_int((gimx, gimy)).astype(_float).reshape(gx.shape).T

        disp_transform = bg_transform.T

        fig, ((ax0, ax1)) = plt.subplots(1, 2)
        fig.suptitle('BPJ cs. MAP')
        do_plot(ax0, db(scaled_bpj), [0, gx.shape[0] - 1, 0, gx.shape[1] - 1],
                disp_transform)
        do_plot(ax1, gpoint_power, [0, gx.shape[0] - 1, 0, gx.shape[1] - 1],
                disp_transform)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(im)
        plt.subplot(2, 1, 2)
        plt.scatter(plons, plats, c=point_power)

        base_pos = rp.pos(pulse_times[::100])

        if points.shape[1] < 512 ** 2:
            fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten(), )
            fig.add_scatter3d(x=base_pos[:, 0], y=base_pos[:, 1], z=base_pos[:, 2])
            fig.add_scatter3d(x=points[0, :], y=points[1, :], z=points[2, :], mode='markers',
                              marker=dict(color=point_power))
            fig.show()

        plt.figure('Chirps')
        plt.plot(db(np.fft.fft(chirps[0], fft_len)))
        plt.plot(db(mf_chirps[0]))

        plt.figure('Data')
        plt.imshow(db(np.fft.ifft(results[0] * mf_chirps, axis=1)[:, :nsam]))
        plt.axis('tight')

        plt.show()

    return True



if __name__ == "__main__":
    cfig = load_yaml_config('/home/jeff/repo/optix_radartracer/simulib/scripts/anduril_params.yaml')

    test_files = ['/home/jeff/SDR_DATA/RAW/04292025/SAR_04292025_111051.sar',
                  '/home/jeff/SDR_DATA/RAW/06032025/SAR_06032025_124843.sar',
                  '/home/jeff/SDR_DATA/RAW/05072025/SAR_05072025_140502.sar',
                  '/home/jeff/SDR_DATA/RAW/05072025/SAR_05072025_141901.sar',
                  '/home/jeff/SDR_DATA/RAW/05072025/SAR_05072025_144041.sar',
                  '/home/jeff/SDR_DATA/RAW/07232025/SAR_07232025_144305.sar',
                  '/home/jeff/SDR_DATA/RAW/12172024/SAR_12172024_112906.sar']
    test_files = ['/home/jeff/SDR_DATA/RAW/06032025/SAR_06032025_124843.sar']
    test_files = ['/home/jeff/SDR_DATA/RAW/04292025/SAR_04292025_111051.sar']

    success_files = []

    for fnme in test_files:
        try:
            runSimulation(cfig, fnme)
            success_files.append(fnme)
        except Exception as e:
            print(e)