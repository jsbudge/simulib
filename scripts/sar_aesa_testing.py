#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scene_tracer import build_tracer, trace_scene
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec, upsamplePulse
from simulib.simulib.platform_helper import RadarPlatform
from simulib.simulib.utils import c0, DTR, _float, _complex_float
from simulib.simulib.sim_objects import AESA
from simulib.simulib.grid_helper import MapEnvironment
from backproject_utils import backprojectPulseStream
import trimesh as tri
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import plotly.io as pio
import pickle
import pandas as pd

def aliasFrequency(f, a_fs):
    return f - int(f / (a_fs / 2)) * a_fs / 2


def calcDopplerPRF(v, fc, bandwidth, half_az_bw):
    return 4 * v * np.sin(half_az_bw) * (fc + bandwidth / 2) / c0
pio.renderers.default = 'browser'



if __name__ == "__main__":
    cfig = load_yaml_config('/home/jeff/repo/optix_radartracer/simulib/scripts/aesa_test_params.yaml')
    npulses = 256
    npts = cfig.tracer_params.pix_height * cfig.tracer_params.pix_width
    fc = 16e9
    fs = 2e9
    bandwidth = 600e6
    upsample = 2
    show_plots = False

    wavelength = c0 / fc
    velocity = 80.

    # Position of plane/heli
    n_total_gps = int(cfig.sim_params.time * 100)
    d_traveled = cfig.sim_params.time * velocity
    gps_times = np.arange(n_total_gps) / 100
    e = np.zeros(n_total_gps) + np.linspace(-d_traveled, d_traveled, n_total_gps)
    n = np.zeros(n_total_gps) - 1400
    u = np.zeros(n_total_gps) + 550
    r = np.zeros(n_total_gps)
    p = np.zeros(n_total_gps)
    y = np.zeros(n_total_gps) + np.pi / 2

    pts = np.array([e, n, u]).T
    vels = np.gradient(pts, axis=0)

    # Position of gimbal
    gim_pan = np.zeros_like(
        gps_times)  # sawtooth(gps_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T
    dep_ang = 35.

    # Design the antenna
    ant = AESA(fc, 2, 2, 1, 1)
    aesa = np.zeros((len(gps_times), 2))
    gimbal_rotations = np.array([180. * DTR, dep_ang * DTR, -np.pi / 2])
    gimbal_offsets = np.array([.0753, 1.6053, -.7873])
    aesa_pointing = np.array([0, -1800, 1524])
    bw_az, bw_el = ant.calc_beamwidth(0., 0.)

    prf = calcDopplerPRF(velocity, fc, bandwidth, bw_az / 2) * 1.5

    pulse_times = np.arange(int(gps_times[-1] * prf)) / prf
    goffsets = np.array([0, np.pi / 2, 0.])

    rps = RadarPlatform(e, n, u, r, p, y, gps_times, ant.phase_center_offsets, ant.phase_center_offsets,
                        gimbal=gimbal,
                        gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                        az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa)

    bg = MapEnvironment(origin=(0, 0, 0), extent=(300, 300), pixels_per_meter=(1., 1.))

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rps.getRadarParams(u.mean(), .2)

    gx, gy, gz = bg.getGrid((0, 0, 0), 300, 300, 300, 300)
    # gx, gy, gz = bg.getGrid(origin, 200, 200, 400, 400, rp.heading(rp.gpst).mean() - np.pi / 2)
    bpj_grid = np.zeros(gx.shape, dtype=_complex_float)

    villa = tri.load(f'/home/jeff/Documents/roman_facade/scene.gltf', force='mesh')
    villa.apply_transform(
        tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
    villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0))
    villa_mats = np.zeros((villa.triangles.shape[0], 2))
    villa_mats[:, 0] = 1e6
    villa_mats[:, 1] = .017
    villa_mesh = BaseMesh(villa, villa_mats, motion_keys=None)
    points = np.array(tri.sample.sample_surface(villa, npts)[0])
    point_power = np.ones(npts)

    # Get the location to place it randomly
    query_points = np.array([gx.flatten(), gy.flatten()]).T

    # 3. Create vertical rays starting from above the mesh
    # Start high up (z=max_z + margin) and shoot down (-z)
    '''ray_origins = np.column_stack([query_points, np.full(len(query_points), villa.bounds[1][2] + 1.0)])
    ray_directions = np.tile([0, 0, -1], (len(query_points), 1))
    locations, index_ray, index_tri = villa.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        multiple_hits=False,
    )

    regz = gz.flatten()
    regz[index_ray] = locations[:, 2]
    gz = regz.reshape(gx.shape)'''

    chirps = genChirp(nr, fs, fc, bandwidth).reshape((1, -1))
    fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)
    mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps

    print('Launching...')

    tracer = build_tracer([villa_mesh], 'sar_trace.cu', pulse_times=pulse_times)

    print(f'Launching {npts} rays.')

    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]

        aesa_bore = azelToVec(rps.tx.az_aesa_iner(ptimes), rps.tx.el_aesa_iner(ptimes))
        # Compute the AESA phi and theta angles for commanding it
        aesa_phi_r, aesa_theta_r = rps.tx.aesa_frame_phi_theta(ptimes[0])
        # Get the element weights for the designed AESA phi and theta
        weights_tx = ant.get_weights(aesa_phi_r, aesa_theta_r)
        weights_rx = ant.get_weights(aesa_phi_r, aesa_theta_r)
        txposes = rps.txpos(ptimes).swapaxes(0, 1).swapaxes(1, 2)
        rxposes = rps.rxpos(ptimes).swapaxes(0, 1).swapaxes(1, 2)
        block_data = trace_scene(tracer, chirps, txposes, rxposes, weights_tx, weights_rx,
                               rps.tx.boresight(ptimes[0]), aesa_bore, points, point_power,
                               ptimes, nsam, fc, fs, near_range_s, ranges[-1], bw_az, bw_el,
                               cfig.tracer_params.pix_width, cfig.tracer_params.pix_height,
                               cfig.ant_params.transmit_power, cfig.ant_params.rx_gain,
                               cfig.ant_params.tx_gain,
                               cfig.ant_params.rec_gain, cfig.ant_params.noise_figure,
                               cfig.ant_params.operating_temperature, fft_len, add_noise=False, add_chirp=True)[0, 0]

        base_pos = rps.pos(ptimes)

        rpi_data = upsamplePulse(block_data * mf_chirps, fft_len, upsample, is_freq=True,
                                 time_len=nsam).astype(_complex_float)

        bpj_grid += backprojectPulseStream([rpi_data], [rps.tx.az_iner(ptimes)],
                                               [rps.rxpos(ptimes).mean(axis=(1, 2))],
                                               [rps.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                               _float(near_range_s), _float(fs * upsample), rps.az_half_bw,
                                               gx=gx, gy=gy)

        if np.any(np.isnan(block_data)):
            break

    plt.figure('Block Data')
    plt.imshow(db(block_data))
    plt.axis('tight')

    plt.figure('Backprojection')
    plt.imshow(db(bpj_grid))
    plt.axis('tight')
    plt.show()





