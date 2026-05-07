#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scene_tracer import build_tracer, trace_scene
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec
from simulib.simulib.platform_helper import RadarPlatform
from simulib.simulib.utils import c0, DTR
from simulib.simulib.sim_objects import AESA
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
    cfig = load_yaml_config('./sikorsky_params.yaml')
    npulses = 256
    npts = cfig.tracer_params.pix_height * cfig.tracer_params.pix_width
    fc = 16e9
    fs = 2e9
    save_file = True
    show_plots = False

    wavelength = c0 / fc

    # Generate circle points for target
    base_e = np.concatenate([np.cos(2 * np.pi * np.linspace(0, 1, 16)) for _ in range(16)])
    base_n = np.concatenate([np.sin(2 * np.pi * np.linspace(0, 1, 16)) for _ in range(16)])
    base_u = np.concatenate([np.ones(16) * l for l in np.linspace(0, .7, 16)])
    base_pts = np.array([base_e, base_n, base_u])
    base_pts = base_pts / np.linalg.norm(base_pts, axis=0)


    # Generate a platform
    print('Generating platform...', end='')

    bandwidth = fs * .35  # float(np.random.rand() * 300e6 + 200e6)
    # Position of plane/heli
    n_total_gps = base_pts.shape[1]
    gps_times = np.arange(n_total_gps)

    # Position of gimbal
    gim_pan = np.zeros_like(gps_times)
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T
    dep_ang = 35.

    # Design the antenna
    ant = AESA(fc, 2, 2, 1, 1)
    aesa = np.zeros((len(gps_times), 2))
    gimbal_rotations = np.array([180. * DTR, dep_ang * DTR, -np.pi / 2])
    gimbal_offsets = np.array([.0753, 1.6053, -.7873])
    bw_az, bw_el = ant.calc_beamwidth(0., 0.)

    prf = 7

    pulse_times = gps_times / 5.
    goffsets = np.array([0, np.pi / 2, 0.])


    # nr = 1000





    # scenes = ['rock_ring', 'winter_trees']
    # targets = ['hangar.gltf', 'b2spirit.gltf', 'helic.obj', 'piper_pa18.obj', 'Porsche_911_GT2.obj']

    targets = ['hangar.gltf', 'b2spirit.gltf', 'helic.obj', 'piper_pa18.obj', 'Porsche_911_GT2.obj',
               'farmhouse_obj.obj', 'cessna-172-obj.obj', 'Humvee.obj', 'Tiger.obj']

    scalings = pd.read_csv('/home/jeff/repo/apache/data/target_info.csv')

    for target_idx, target in enumerate(targets):

        b2 = tri.load(f'/home/jeff/Documents/target_meshes/{target}', force='mesh')
        b2.apply_transform(
            tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
        if target in scalings['filename'].values.astype(str):
            b2.apply_scale(1 / scalings.loc[scalings['filename'] == target, 'scaling'].values[0])

        # Move the target to be on top of the scene mesh in the random location
        b2.apply_translation(-b2.bounding_box.bounds.mean(axis=0))
        b2_mats = np.zeros((b2.triangles.shape[0], 2))
        b2_mats[:, 0] = 1e6
        b2_mats[:, 1] = .01
        b2_mesh = BaseMesh(b2, b2_mats, motion_keys=None)
        points = np.array(tri.sample.sample_surface(b2, npts)[0])
        point_power = np.ones(npts)

        just_targ = []
        print(f'Now running {target}.')
        for run in range(15):
            rng = np.random.rand() * 1000 + 1200
            e, n, u = base_pts * rng
            r = np.zeros(n_total_gps)
            p = np.zeros(n_total_gps)
            y = np.arctan2(e, n)
            rps = RadarPlatform(e, n, u, r, p, y, gps_times, ant.phase_center_offsets, ant.phase_center_offsets,
                                gimbal=gimbal,
                                gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                                az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth,
                                aesa=aesa)
            nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rps.getRadarParams(
                u.mean(),
                .2,
                a_ranges=[
                    rng - 500,
                    rng + 500])
            chirps = genChirp(nr, fs, fc, bandwidth).reshape((1, -1))
            fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)
            mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps
            build_params = {'fc': fc, 'fs': fs, 'bandwidth': bandwidth, 'dep_ang': rps.dep_ang,
                            'standoff': ranges[0],
                            'prf': prf, 'rotation': 0., 'nsam': nsam, 'nr': nr}

            print('Launching...')

            if show_plots:
                fig = plt.figure(constrained_layout=True)
                gs = GridSpec(2, 3, figure=fig)
                sumax = fig.add_subplot(gs[0, 0])
                elax = fig.add_subplot(gs[0, 1])
                azax = fig.add_subplot(gs[1, 0])
                scatax = fig.add_subplot(gs[:, 2])
                cbax = fig.add_subplot(gs[1, 1])

            tracer = build_tracer([b2_mesh], 'sar_trace.cu', pulse_times=pulse_times)

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

                if show_plots:
                    neg_beamvec = np.array([np.sin(rps.az_iner(ptimes).mean() - bw_az / 2),
                                            np.cos(rps.az_iner(ptimes).mean() - bw_az / 2)])
                    pos_beamvec = np.array([np.sin(rps.az_iner(ptimes).mean() + bw_az / 2),
                                            np.cos(rps.az_iner(ptimes).mean() + bw_az / 2)])
                    beamx = np.array([neg_beamvec[0] * granges[0], neg_beamvec[0] * granges[-1], pos_beamvec[0] * granges[-1],
                              pos_beamvec[0] * granges[0], neg_beamvec[0] * granges[0]]) + base_pos[0, 0]
                    beamy = np.array([neg_beamvec[1] * granges[0], neg_beamvec[1] * granges[-1], pos_beamvec[1] * granges[-1],
                                      pos_beamvec[1] * granges[0], neg_beamvec[1] * granges[0]]) + base_pos[0, 1]
                    # plt.cla()

                    plot_data = np.fft.fft(block_data, axis=0)


                    sumax.cla()
                    elax.cla()
                    azax.cla()
                    scatax.cla()
                    sumax.set_title('BlockData')
                    sumax.imshow(db(block_data).T, origin='lower', aspect='auto',
                                 extent=(0, npulses, ranges[0], ranges[-1]))
                    sumax.set_ylabel('Range (m)')
                    elax.set_title('IFFT')
                    elax.imshow(db(plot_data).T, origin='lower', aspect='auto',
                                 extent=(0, npulses, ranges[0], ranges[-1]))
                    elax.set_ylabel('Range (m)')
                    if abs(rps.az_iner(ptimes).mean() - np.arctan2(base_pos[0, 0], base_pos[0, 1])) < bw_az:
                        sumax.scatter([0], [np.linalg.norm(rps.txpos(ptimes).mean(axis=0))], c='red')
                    scatax.scatter(base_pos[:, 0], base_pos[:, 1], c='blue')
                    for h in tracer.hulls:
                        scatax.scatter(h.vertices[:, 0], h.vertices[:, 1])
                    scatax.plot(beamx, beamy, c='blue')
                    plt.draw()
                    plt.pause(.01)

                block_data = np.fft.ifft(block_data * mf_chirps)[:, :nsam] * ((ranges - ranges[0]) / ranges[-1] + 1.)**2
                # block_data = block_data[0]

                if np.any(np.isnan(block_data)):
                    break

                # Split into segments for saving
                if save_file:
                    check_data = db(block_data)
                    thresh = check_data > check_data.mean() + 2 * check_data.std()
                    thresh_min = int(np.where(thresh)[1].mean() - 2000)
                    thresh_max = int(np.where(thresh)[1].mean() + 2000)
                    just_targ.append(block_data[:, thresh_min:thresh_max].astype(np.complex64).view('(2,)float32').swapaxes(-1, -3))

        plt.figure()
        plt.imshow(db(block_data))
        plt.axis('tight')
        plt.show()

        check_data = db(block_data)
        thresh = check_data > check_data.mean() + 3 * check_data.std()
        plt.figure()
        plt.imshow(thresh)
        plt.axis('tight')
        plt.show()

        target_id = np.zeros(len(targets))
        target_id[target_idx] = 1

        if save_file:
            # Save out target data to file
            with open(f'/home/jeff/repo/apache/data/target_new/{target.split('.')[0].replace('-', '_')}-embedding.pic', 'wb') as f:
                pickle.dump({'target': just_targ, 'build': build_params, 'name': target, 'target_id': target_id}, f)





