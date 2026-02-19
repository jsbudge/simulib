#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scene_tracer import build_tracer, trace_cpi
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow
from simulib.simulib.platform_helper import RadarPlatform
from simulib.simulib.utils import c0, DTR
import trimesh as tri
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import plotly.io as pio
import pickle

def aliasFrequency(f, a_fs):
    return f - int(f / (a_fs / 2)) * a_fs / 2


def calcDopplerPRF(v, fc, bandwidth, half_az_bw):
    return 4 * v * np.sin(half_az_bw) * (fc + bandwidth / 2) / c0
pio.renderers.default = 'browser'



if __name__ == "__main__":
    cfig = load_yaml_config('./sikorsky_params.yaml')
    npulses = 64
    nper_seq = 32
    fc = 16e9
    fs = 1e9
    bw_az = 4. * DTR
    bw_el = 3. * DTR
    save_file = True
    show_plots = False

    velocity = 77.1667
    wavelength = c0 / fc

    # Generate a platform
    print('Generating platform...', end='')
    for sample_run in range(10):

        bandwidth = np.random.rand() * 300e6 + 200e6

        # Position of plane/heli
        n_total_gps = int(cfig.sim_params.time * 100)
        d_traveled = cfig.sim_params.time * velocity
        gps_times = np.arange(n_total_gps) / 100
        e = np.zeros(n_total_gps) + np.linspace(-d_traveled, d_traveled, n_total_gps)
        n = np.zeros(n_total_gps) - 1400
        u = np.zeros(n_total_gps) + np.random.rand() * 300 + 300
        r = np.zeros(n_total_gps)
        p = np.zeros(n_total_gps)
        y = np.zeros(n_total_gps) + np.pi / 2

        pts = np.array([e, n, u]).T
        vels = np.gradient(pts, axis=0)

        # Position of gimbal
        gim_pan = -np.arctan2(vels[:, 0], vels[:, 1])
        gim_el = np.arcsin(-u / np.linalg.norm(pts, axis=1))
        gimbal = np.array([gim_pan, gim_el]).T

        prf = calcDopplerPRF(velocity, fc, bandwidth, bw_az / 2) * 1.5

        pulse_times = np.arange(int(gps_times[-1] * prf)) / prf
        goffsets = np.array([0, np.pi / 2, 0.])

        rps = RadarPlatform(e, n, u, r, p, y, gps_times, gimbal=gimbal, dep_angle=-(gim_el.mean() - bw_el / 4) / DTR,
                               gimbal_offset=np.array([0., 1., 0.]), gimbal_rotations=goffsets,
                               az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth)

        nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rps.getRadarParams(u.mean(), .25)

        chirp = genChirp(nr, fs, fc, bandwidth) * 1000.
        fft_chirp = np.fft.fft(chirp, fft_len)
        mf_chirp = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirp

        build_params = {'fc': fc, 'fs': fs, 'bandwidth': bandwidth, 'dep_ang': rps.dep_ang, 'standoff': ranges[0], 'prf': prf, 'rotation': 0., 'nsam': nsam, 'nr': nr}

        # Steering vector for overall receive beam
        # k_wave = azelToVec(gim_az, gim_el) * 2 * np.pi / (c0 / fc)
        # v_k = np.exp(-1j * np.dot(k_wave[None, :], rx_offsets.T)).flatten()

        scenes = ['rock_ring', 'winter_trees']
        targets = ['hangar.gltf', 'b2spirit.gltf', 'helic.obj', 'piper_pa18.obj', 'Porsche_911_GT2.obj']

        for n in range(5):
            rotation = np.random.rand() * np.pi
            build_params['rotation'] = rotation
            for scene in scenes:
                for target in targets:
                    non_targ = []
                    yes_targ = []
                    targ_p = []
                    spheres = []
                    mats = []
                    print(f'Now running {scene} and {target}.')
                    for run in range(2):
                        use_target = run == 1
                        if not use_target:
                            villa = tri.load(f'/home/jeff/Documents/{scene}/scene.gltf', force='mesh')
                            villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
                            villa.apply_transform(
                                tri.transformations.rotation_matrix(rotation, np.array([0., 0., 1.]), np.array([0, 0, 0.])))
                            villa.apply_scale(40.)
                            villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0))
                            villa_mats = np.zeros((villa.triangles.shape[0], 2))
                            villa_mats[:, 0] = 1.1
                            villa_mats[:, 1] = .017
                            spheres.append(BaseMesh(villa, 1, villa_mats, do_sample=False, motion_keys=None))
                            mats.append([1e6, .017])


                        if use_target:
                            b2 = tri.load(f'/home/jeff/Documents/target_meshes/{target}', force='mesh')
                            b2.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
                            b2.apply_transform(
                                tri.transformations.rotation_matrix(rotation, np.array([0., 0., 1.]), np.array([0, 0, 0.])))
                            # Get the location to place it randomly
                            query_points = np.random.rand(2).reshape((1, 2)) * 10

                            # 3. Create vertical rays starting from above the mesh
                            # Start high up (z=max_z + margin) and shoot down (-z)
                            ray_origins = np.column_stack([query_points, np.full(len(query_points), villa.bounds[1][2] + 1.0)])
                            ray_directions = np.tile([0, 0, -1], (len(query_points), 1))
                            locations, index_ray, index_tri = villa.ray.intersects_location(
                                ray_origins=ray_origins,
                                ray_directions=ray_directions
                            )
                            b2.apply_translation(-b2.bounding_box.bounds.mean(axis=0) + locations[0])
                            villa_mats = np.zeros((b2.triangles.shape[0], 2))
                            villa_mats[:, 0] = 1e6
                            villa_mats[:, 1] = .017
                            spheres.append(BaseMesh(b2, 1, villa_mats, do_sample=False, motion_keys=None))
                            mats.append([1e6, .017])


                        materials = np.stack(mats)

                        print('Launching...')

                        if show_plots:
                            fig = plt.figure(constrained_layout=True)
                            gs = GridSpec(2, 3, figure=fig)
                            sumax = fig.add_subplot(gs[0, 0])
                            elax = fig.add_subplot(gs[0, 1])
                            azax = fig.add_subplot(gs[1, 0])
                            scatax = fig.add_subplot(gs[:, 2])
                            cbax = fig.add_subplot(gs[1, 1])

                        tracer = build_tracer(spheres, pulse_times=pulse_times)

                        print(f'Launching {cfig.tracer_params.pix_height * cfig.tracer_params.pix_width} rays.')

                        for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
                            ptimes = pulse_times[frame[0]:frame[0] + npulses]
                            if len(ptimes) < npulses:
                                break
                            txposes = rps.txpos(ptimes)[:, :3].reshape((1, npulses, 3))
                            rxposes = rps.rxpos(ptimes)[:, :3].reshape((1, npulses, 3))
                            block_data = trace_cpi(tracer, chirp,
                                                npulses, txposes, rxposes,
                                                   rps._tx.boresight(ptimes).T, materials, ptimes,
                                                nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el / 2,
                                                cfig.tracer_params.pix_width, cfig.tracer_params.pix_height, cfig.ant_params.transmit_power,
                                                cfig.ant_params.rx_gain, cfig.ant_params.tx_gain, cfig.ant_params.rec_gain,
                                                cfig.ant_params.noise_figure, cfig.ant_params.operating_temperature, fft_len, add_chirp=False)

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


                                sumax.cla()
                                elax.cla()
                                azax.cla()
                                scatax.cla()
                                sumax.set_title('Sum Beam')
                                sumax.imshow(db(block_data[0]).T, origin='lower', aspect='auto', extent=(0, npulses, ranges[0], ranges[-1]))
                                sumax.set_ylabel('Range (m)')
                                if abs(rps.az_iner(ptimes).mean() - np.arctan2(base_pos[0, 0], base_pos[0, 1])) < bw_az:
                                    sumax.scatter([0], [np.linalg.norm(rps.txpos(ptimes).mean(axis=0))], c='red')
                                scatax.scatter(base_pos[:, 0], base_pos[:, 1], c='blue')
                                for h in tracer.hulls:
                                    scatax.scatter(h.vertices[:, 0], h.vertices[:, 1])
                                scatax.plot(beamx, beamy, c='blue')
                                plt.draw()
                                plt.pause(.01)

                            block_data = np.fft.fft(block_data, 8192, axis=2)[0]
                            # block_data = block_data[0]

                            if np.any(np.isnan(block_data)):
                                break

                            # Split into segments for saving
                            if use_target:
                                yes_targ.append(np.stack([block_data[n:n + nper_seq] for n in range(0, len(block_data) - nper_seq)]))
                                targ_pos = np.linalg.norm(txposes[0] - b2.bounding_box.bounds.mean(axis=0), axis=1)
                                targ_p.append(np.array([int((targ_pos[n:n + nper_seq].mean() * 2 / c0 - 2 * near_range_s) * fs) for n in range(0, len(targ_pos) - nper_seq)]))
                            else:
                                non_targ.append(np.stack([block_data[n:n + nper_seq] for n in range(0, len(block_data) - nper_seq)]))

                    # Save out target data to file
                    yes_targ = np.concatenate(yes_targ).view('(2,)float').astype(np.float32).swapaxes(-1, -2)
                    non_targ = np.concatenate(non_targ).view('(2,)float').astype(np.float32).swapaxes(-1, -2)
                    targ_p = np.stack(targ_p)
                    if save_file:
                        with open(f'/home/jeff/repo/apache/data/target_new/{scene}_{target.split('.')[0]}_{sample_run}_{n}.pic', 'wb') as f:
                            pickle.dump({'target': yes_targ, 'clutter': non_targ, 't_idx': targ_p, 'build': build_params}, f)




    '''ani = anim.ArtistAnimation(fig, ims, interval=150, blit=True)
    test = anim.FFMpegWriter(fps=5)
    ani.save('data.gif', writer=test)'''

    # plt.figure('Minimum Range Distance')
    # plt.imshow(range_grid)

    plt.figure('Chirps')
    plt.plot(db(fft_chirp))
    plt.plot(db(mf_chirp))

    plt.figure('Autocorrelation')
    plt.plot(np.fft.fftshift(db(np.fft.ifft(fft_chirp * mf_chirp))))

    plt.figure('Data')
    yes_data = db(np.fft.ifft(yes_targ[2, 0, 0] + 1j * yes_targ[2, 0, 1]))
    plt.plot(yes_data)
    plt.plot(db(np.fft.ifft(non_targ[2, 0, 0] + 1j * non_targ[2, 0, 1])))
    plt.vlines([targ_p[2, 0]], yes_data.min(), yes_data.max(), color='black')
    plt.show()





