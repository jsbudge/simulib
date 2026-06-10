#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scene_tracer import build_tracer, trace_scene
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec, upsamplePulse, enu2llh, llh2enu
from simulib.simulib.platform_helper import RadarPlatform, SDRPlatform
from simulib.simulib.utils import c0, DTR, _float, _complex_float, getRadarAndEnvironment
from simulib.simulib.sim_objects import AESA
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
from sdrparse.SDRV2Parsing import load
from simulib.simulib.grid_helper import SDREnvironment
from backproject_utils import backprojectPulseStream
import trimesh as tri
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import plotly.io as pio
import plotly.express as px
import pickle
import requests
from urllib.request import Request, urlopen, urlretrieve
from itertools import product
from io import BytesIO
import pandas as pd
import mmap
import shutil
from pathlib import Path
import array
pio.renderers.default = 'browser'

GECOEFF = 156543.03392
TILE_SIZE = 256


def calcGoogleCoords(a_origin, lats, lons, mpp):
    center_lat = a_origin[0]
    center_lon = a_origin[1]
    zoomlevel = int(np.round(np.log2(GECOEFF * np.cos(center_lat * np.pi / 180.) / mpp)))
    gmpp = GECOEFF * np.cos(center_lat * np.pi / 180.) / (2 ** zoomlevel)

    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    scale = 1 << zoomlevel

    box_coords = np.array([lats, lons]).T
    pix_coords = np.zeros_like(box_coords)
    sy = np.sin(box_coords[:, 0] * np.pi / 180.0)
    pix_coords[:, 0] = (TILE_SIZE / 2. + box_coords[
        :, 1] * TILE_SIZE / 360) * scale  # // TILE_SIZE # np.floor(TILE_SIZE * (.5 + box_coords[:, 1] / 360.) * scale)
    pix_coords[:, 1] = ((TILE_SIZE / 2) + .5 * np.log((1 + sy) / (1 - sy)) * -(
                TILE_SIZE / (2 * np.pi))) * scale  # // TILE_SIZE

    tile_coords = pix_coords // TILE_SIZE

    start_x = int(tile_coords[:, 0].min())
    start_y = int(tile_coords[:, 1].min())

    # Get interpolated values from image for lat/lons
    im_x = pix_coords[:, 0] - start_x * TILE_SIZE
    im_y = pix_coords[:, 1] - start_y * TILE_SIZE
    return tile_coords, zoomlevel, im_x, im_y


def resampleGoogleMap(lats, lons, mpp):
    tile_coords, zoomlevel, im_x, im_y = calcGoogleCoords(((lats.max() + lats.min()) / 2, (lons.max() + lons.min()) / 2), lats, lons, mpp)

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
        im = Image.open(BytesIO(response.content))
        map_img.paste(im, (x * TILE_SIZE, y * TILE_SIZE))

    return map_img, im_x, im_y



if __name__ == "__main__":
    cfig = load_yaml_config('/home/jeff/repo/optix_radartracer/simulib/scripts/anduril_params.yaml')
    npulses = 512
    upsample = 1
    show_plots = True
    origin = (40.098902, -111.659862, 1399.)
    save_file = False
    randomize_pts = True
    add_target = False

    pix_width = 512
    pix_height = 512
    nray_sqrt = 1000
    nrays = nray_sqrt * nray_sqrt

    fnme = '/home/jeff/SDR_DATA/RAW/10282025/SAR_10282025_112741.sar'

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
    except:
        ref_pts = sdr[sdr_dataset].packet_points
        atts = sdr[sdr_dataset].atts

    # origin = ()
    bg, rp = getRadarAndEnvironment(sdr)

    origin = bg.origin

    wavelength = c0 / sdr[sdr_dataset].fc
    fs = sdr[sdr_dataset].fs
    rp.fs = fs
    fc = rp.fc

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(0., 0., upsample)
    grid_height_m = (granges[-1] - granges[0]) / 2
    grid_width_m = np.linalg.norm(rp.pos(rp.gpst[0]) - rp.pos(rp.gpst[-1])) - ranges[-1] * np.tan(rp.az_half_bw)

    chirps = sdr[sdr_dataset].cal_chirp.reshape((1, -1)).astype(_complex_float)
    mf_chirps = sdr.genReciprocalRipple(0, 0, 0).reshape((1, -1)).astype(_complex_float)

    # Design the antenna
    ant = AESA(rp.fc, 10, 2, 1, 1)
    bw_az, bw_el = ant.calc_beamwidth(0., 0.)

    pulse_times = sdr[sdr_dataset].pulse_time

    # gx, gy, gz = bg.gridFromSwath(0, 0, bw_az, 1500e6)

    gx, gy, gz = bg.getGrid(origin, width=grid_height_m, height=grid_width_m, nrows=pix_width, ncols=pix_height, az=bg.heading)
    gz[:] = gz.mean()

    print('Getting Google Maps grid...')
    res_mpp = max((gx.max() - gx.min()) / pix_height, (gy.max() - gy.min()) / pix_height)
    lats, lons, alts = enu2llh(gx.flatten(), gy.flatten(), gz.flatten(), bg.ref)
    im, imx, imy = resampleGoogleMap(lats, lons, res_mpp)
    # Google image is transposed compared to what we expect in SDREnvironment
    im = np.array(im).sum(axis=2).T
    im = ((im - im.min()) * 1 / (im.max() - im.min())) + .001
    npts = nrays

    grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
    tris_2d = Delaunay(grid_pts[:, :2])
    elevation_mesh = tri.Trimesh(vertices=grid_pts, faces=tris_2d.simplices)
    el_mats = np.zeros((elevation_mesh.triangles.shape[0], 2))
    el_mats[:, 0] = 1e6
    el_mats[:, 1] = .017
    grid_int = RegularGridInterpolator((np.arange(im.shape[0]), np.arange(im.shape[1])), im)

    if add_target:
        b2 = tri.load('/home/jeff/Documents/target_meshes/piper_pa18.obj', force='mesh')
        b2.apply_transform(
            tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))

        # Move the target to be on top of the scene mesh in the random location
        pos_in_scene = np.array(llh2enu(*origin, bg.ref))
        pos_in_scene = pos_in_scene + (pos_in_scene - np.array(llh2enu(40.098447, -111.659208, 1399., bg.ref)))
        pos_in_scene[2] = gz.mean() + 1.
        b2.apply_translation(-b2.bounding_box.bounds.mean(axis=0) + pos_in_scene)
        b2_mats = np.zeros((b2.triangles.shape[0], 2))
        b2_mats[:, 0] = 1e6
        b2_mats[:, 1] = .017
        b2_mesh = BaseMesh(b2, b2_mats, motion_keys=None)

    bpj_grid = np.zeros(gx.shape, dtype=_complex_float)

    # Build a trimesh of the elevation map
    # Set points for rays
    if randomize_pts:
        # Rejection sampling, since it's easier
        points = np.random.rand(npts, 3) * np.array([grid_width_m - .1, grid_height_m - .1, 0]) - np.array([grid_width_m / 2, grid_height_m / 2, -1.])
        bg_transform = bg.transforms + 0.
        bg_transform[:2, 2] = llh2enu(*origin, bg.ref)[:2]
        points = points @ bg_transform.T
    else:
        points = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T

    print('Sampling Google Map...')
    plats, plons, _ = enu2llh(*points.T, bg.ref)

    _, _, iimx, iimy = calcGoogleCoords(((lats.max() + lats.min()) / 2, (lons.max() + lons.min()) / 2), plats, plons, res_mpp)
    point_power = grid_int((iimx, iimy)).astype(_float)
    # point_power = im.flatten().astype(_float)

    # Get the location to place it randomly
    query_points = points[:, :2]

    # Create vertical rays starting from above the mesh
    # Start high up (z=max_z + margin) and shoot down (-z)
    ray_origins = np.column_stack([query_points, np.full(len(query_points), elevation_mesh.bounds[1][2] + 1500.0)])
    ray_directions = np.tile([0, 0, -1], (len(query_points), 1))
    locations, index_ray, index_tri = elevation_mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
    )

    points[index_ray, 2] = locations[:, 2]

    if add_target:
        targlocations, targindex_ray, targindex_tri = b2.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False,
        )
        points[targindex_ray, 2] = targlocations[:, 2]
        point_power[targindex_ray] = 1.

    meshes = [BaseMesh(elevation_mesh, el_mats, motion_keys=None), b2_mesh] if add_target else [BaseMesh(elevation_mesh, el_mats, motion_keys=None)]

    tracer = build_tracer(meshes, 'sar_trace.cu', pulse_times=pulse_times)

    print(f'Launching {npts} rays.')

    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]

        aesa_bore = azelToVec(rp.tx.az_aesa_iner(ptimes), rp.tx.el_aesa_iner(ptimes))
        # Compute the AESA phi and theta angles for commanding it
        aesa_phi_r, aesa_theta_r = rp.tx.aesa_frame_phi_theta(ptimes[0])
        # Get the element weights for the designed AESA phi and theta
        weights_tx = ant.get_weights(aesa_phi_r, aesa_theta_r)
        weights_rx = ant.get_weights(aesa_phi_r, aesa_theta_r)
        txposes = rp.txpos(ptimes).swapaxes(0, 1).swapaxes(1, 2)
        rxposes = rp.rxpos(ptimes).swapaxes(0, 1).swapaxes(1, 2)
        block_data = trace_scene(tracer, chirps, txposes, rxposes, weights_tx, weights_rx,
                               rp.tx.boresight(ptimes[0]), aesa_bore, points, point_power,
                               ptimes, nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el,
                               nray_sqrt, nray_sqrt,
                               cfig.ant_params.transmit_power, cfig.ant_params.rx_gain,
                               cfig.ant_params.tx_gain,
                               cfig.ant_params.rec_gain, cfig.ant_params.noise_figure,
                               cfig.ant_params.operating_temperature, fft_len, add_noise=True, add_chirp=True)[0, 0]

        block_data[np.isnan(block_data)] = 0.

        if save_file:
            save_data = np.fft.ifft(block_data, axis=1, norm='ortho')[:, :nsam]
            # save_int_data = np.zeros((*save_data.shape, 2)).astype(np.int16)
            ndata = (save_data.astype(np.complex64).view('(2,)float32') / 10 ** (atts[frame[0]:frame[0] + len(ptimes)][:, None, None] / 20))
            ndata = (-1 + (ndata - ndata.min()) * 2 / (ndata.max() - ndata.min())) * 32768
            save_int_data = ndata.astype(np.int16)
            # save_data /= old_tmax * 32768
            with open(copy_path, 'r+b') as fin:
                with mmap.mmap(fin.fileno(), 0) as mm:
                    for fr_idx, fr in enumerate(np.arange(frame[0], frame[0] + len(ptimes))):
                        # ndata = (save_data[fr_idx].astype(np.complex64).view('(2,)float32') / 10**(atts[fr] / 20)).astype(np.int16)
                        tmp_data = array.array('h', save_int_data[fr_idx].astype(np.int16).flatten())
                        tmp_data.byteswap()
                        mm.seek(ref_pts[fr])
                        mm.write(bytes(tmp_data))


        if show_plots:
            if save_file:
                new_data = np.zeros((len(ptimes), nsam), dtype=np.complex128)
                # tmp_data = bytes(save_int_data[fr_idx].flatten())
                for fr_idx, fr in enumerate(np.arange(frame[0], frame[0] + len(ptimes))):
                    ndata = bytes(save_int_data[fr_idx].flatten())
                    # ndata = tmp_data[fr_idx:fr_idx + nsam * 4]
                    re_data = array.array('h', ndata)
                    # re_data.byteswap()
                    re_data = np.array(re_data)
                    new_data[fr_idx] = (re_data[:nsam * 2:2] + 1j * re_data[1:nsam * 2:2]) * 10 ** (atts[fr] / 20)
                block_data = np.fft.fft(new_data, fft_len, axis=1)
            # block_data = np.fft.fft((save_int_data[..., 0] + 1j * save_int_data[..., 1]) * 10 ** (31 / 20), fft_len, axis=1)

            rpi_data = upsamplePulse(block_data * mf_chirps, fft_len, upsample, is_freq=True,
                                     time_len=nsam).astype(_complex_float)
            bpj_grid += backprojectPulseStream([rpi_data], [rp.tx.az_aesa_iner(ptimes)],
                                                   [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                                   [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                                   _float(near_range_s), _float(fs * upsample), _float(rp.az_half_bw),
                                                   gx=gx, gy=gy)

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
        plt.imshow(db(scaled_bpj), extent=(lats.min(), lats.max(), lons.min(), lons.max()), cmap='gray')
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.title('Map')
        plt.imshow(grid_int((imx, imy)).reshape(gx.shape), cmap='gray')
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



