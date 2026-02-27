#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh, OceanMesh
from scipy.signal import fftconvolve, sawtooth
from scipy.interpolate import interpn, make_interp_spline
from scipy.signal.windows import taylor
from scene_tracer import build_tracer, trace_cpi
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec, \
    getDopplerLine, genPulse
from simulib.simulib.platform_helper import RadarPlatform, getRotationOffsetMatrix
from plotly_utils import drawAntennaBox
from simulib.simulib.utils import c0, DTR, _complex_float, design_element_positions, get_element_phases
from simulib.simulib.rotation_functions import get_aesa_pointing_from_phi_theta
import trimesh as tri
import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from scipy.ndimage import generic_filter, label
from writer_utils import writePacket, writeDataParams

def aliasFrequency(f, a_fs):
    return f - int(f / (a_fs / 2)) * a_fs / 2


def calcDopplerPRF(v, fc, bandwidth, half_az_bw):
    return 4 * v * np.sin(half_az_bw) * (fc + bandwidth / 2) / c0


def to_vec(phi, theta):
    return np.array([np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta),
              np.cos(theta)])


if __name__ == "__main__":
    cfig = load_yaml_config('./sikorsky_params.yaml')
    plp = cfig.radar_params.plp
    dopp_upsample = cfig.tracer_params.dopp_upsample
    init_ranges = cfig.sim_params.init_ranges
    upsample = cfig.tracer_params.upsample
    npulses = cfig.tracer_params.block_size
    prf_broadening_factor = cfig.radar_params.broadening_factor
    fc = cfig.radar_params.fc
    fs = cfig.radar_params.fs
    bw_az = cfig.ant_params.az_bw * DTR
    bw_el = cfig.ant_params.el_bw * DTR
    prf = cfig.radar_params.prf * prf_broadening_factor
    bandwidth = cfig.radar_params.bandwidth
    sim_time = cfig.sim_params.time
    wavelength = c0 / fc

    # Generate a platform
    print('Generating platform...', end='')

    # Position of plane/heli
    n_total_gps = int(sim_time * 100)
    gps_times = np.arange(n_total_gps) / 100
    e = np.zeros(n_total_gps) - 25. * sim_time + np.linspace(0, 50 * sim_time, n_total_gps)
    n = np.zeros(n_total_gps) - 1400
    u = np.zeros(n_total_gps) + 1524
    r = np.zeros(n_total_gps)
    p = np.zeros(n_total_gps)
    y = np.arctan2(np.gradient(e), np.gradient(n))

    # Position of gimbal
    gim_pan = np.zeros_like(gps_times) # sawtooth(gps_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T

    aesa_elem_pos_m, aesa_width_m, aesa_height_m, element_patch_size_m = \
        design_element_positions(fc, 5, 4, 4, 2)
    bw_az = 115 / (aesa_width_m / wavelength) * DTR / 2
    bw_el = 115 / (aesa_height_m / wavelength) * DTR / 2
    pointing_az = sawtooth(gps_times * cfig.radar_params.scan_rate / cfig.radar_params.scan_limit * 2 * np.pi,
                           width=.5) * cfig.radar_params.scan_limit * DTR / 2
    aesa = get_aesa_pointing_from_phi_theta(np.zeros_like(pointing_az), pointing_az).T
    # aesa = get_aesa_pointing_from_phi_theta(np.zeros_like(pointing_az), np.zeros_like(pointing_az) + np.pi / 4).T
    # aesa_elem_pos_m, aesa_width_m, aesa_height_m, element_patch_size_m = \
    #    design_element_positions(fc, 2, 2, 1, 1)
    tx_el_offsets = [aesa_elem_pos_m.swapaxes(2, 4).swapaxes(2, 3).reshape((-1, 3))]
    rx_el_offsets = [aesa_elem_pos_m.swapaxes(2, 4).swapaxes(2, 3).reshape((-1, 3))]
    # rx_el_offsets = [x for xs in [[aesa_elem_pos_m[i, j].swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 3) for j in range(aesa_elem_pos_m.shape[1])] for i in range(aesa_elem_pos_m.shape[0])] for x in xs]

    gimbal_rotations = np.array([180. * DTR, 60 * DTR, -np.pi / 2])
    gimbal_offsets = np.array([.0753, 1.6053, -.7873])
    aesa_pointing = np.array([0, -1800, 1524])

    dep_ang = 30.
    # rxel_array = np.stack([rxe.dot(getRotationOffsetMatrix(*goffsets)) for rxe in rx_el_offsets], axis=0)
    rxel_array = np.stack([rxe for rxe in rx_el_offsets], axis=0)
    rx_array = rxel_array.mean(axis=1)
    txel_array = np.stack([txe for txe in tx_el_offsets], axis=0)
    tx_array = txel_array.mean(axis=1)
    vx_array = np.concatenate([rx_array + tx for tx in tx_array])

    prf = calcDopplerPRF(np.linalg.norm(np.gradient(np.array([e, n, u]).T, axis=0), axis=1).mean() * 100., fc,
                         bandwidth, bw_az / 2 + cfig.radar_params.scan_limit * DTR / 2)# * prf_broadening_factor
    pulse_times = np.arange(int(gps_times[-1] * prf)) / prf

    rp_rx = [RadarPlatform(e, n, u, r, p, y, gps_times, np.array([0, 0, 0.]), rxo, gimbal=gimbal,
                           gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                           az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa) for rxo in rx_el_offsets]
    rp_tx = RadarPlatform(e, n, u, r, p, y, gps_times, tx_el_offsets[0], np.array([0, 0, 0.]), gimbal=gimbal,
                          gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                          az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa)

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp_tx.getRadarParams(u.mean(), .25)

    chirps = genChirp(nr, fs, fc, bandwidth).reshape((1, -1)) * 1000.
    fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)
    mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps
    doppwin = taylor(npulses * dopp_upsample, nbar=11, sll=90)

    dopp_freq = np.fft.fftshift(np.fft.fftfreq(len(doppwin), 1. / prf))

    villa = tri.load(f'/home/jeff/Documents/roman_facade/scene.gltf')
    spheres = []
    materials = []
    for s in range(1):
        # villa = tri.creation.box(extents=np.array([10000, 10000, 1]))
        villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
        villa.apply_scale(7.)
        villa.apply_transform(tri.transformations.translation_matrix(-villa.bounding_box.bounds.mean(axis=0) + np.array([0, 2000, 1])))
        # mats = np.random.rand(len(villa.geometry), 2) * np.array([1e3, .1])
        spheres.append(BaseMesh(villa, motion_keys=None))

    villa = tri.creation.box(extents=np.array([10000, 10000, 1]))
    villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) + np.array([0, 2000, -1]))
    spheres.append(BaseMesh(villa, motion_keys=None))

    '''for saucer in range(5):
        boat_motion_keys = []

        boat_motion_vector = azelToVec((np.random.rand() * 360) * DTR, 0.) * (np.random.rand() * 20. + 5.) * np.diff(pulse_times[::100])[1]

        for idx, o in enumerate(pulse_times[::100]):
            boat_motion_keys += [
                1.0, 0.0, 0.0, idx * boat_motion_vector[0],
                0.0, 1.0, 0.0, idx * boat_motion_vector[1],
                0.0, 0.0, 1.0, idx * boat_motion_vector[2],
            ]
        villa = tri.creation.icosphere(3, np.random.rand() * 20. + 4.)
        villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) + np.array([0, 1600 - np.random.rand() * 100., 150 * np.random.rand() - 75]))
        villa_mats = np.zeros((villa.triangles.shape[0], 2))
        villa_mats[:, 0] = 1e6
        villa_mats[:, 1] = .017
        spheres.append(BaseMesh(villa, 1, villa_mats, do_sample=False, motion_keys=boat_motion_keys))
        materials.append([1e6, .017])'''

    boat_motion_keys = []

    boat_motion_vector = azelToVec(-45 * DTR, 0.) * 10.2889 * np.diff(pulse_times[::100])[1]

    for idx, o in enumerate(pulse_times[::100]):
        boat_motion_keys += [
            1.0, 0.0, 0.0, idx * boat_motion_vector[0],
            0.0, 1.0, 0.0, idx * boat_motion_vector[1],
            0.0, 0.0, 1.0, idx * boat_motion_vector[2],
        ]
    villa = tri.load(f'/home/jeff/Documents/target_meshes/helic.obj')
    villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
    villa.apply_scale(10.)
    villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) + np.array([0, 1600, 150]))
    mats = np.random.rand(len(villa.geometry), 2) * np.array([1e3, .1])
    spheres.append(BaseMesh(villa, materials=mats, motion_keys=boat_motion_keys))

    print('Launching...')

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig)
    sumax = fig.add_subplot(gs[0, 0])
    scatax = fig.add_subplot(gs[0, 1])
    # fig, ax = plt.subplots()
    # ims = []

    tracer = build_tracer(spheres, pulse_times=pulse_times, debug=False)

    print(f'Launching {cfig.tracer_params.pix_height * cfig.tracer_params.pix_width} rays.')
    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
        if len(ptimes) < npulses:
            break
        aesa_bore = azelToVec(rp_tx._tx.az_aesa_iner(ptimes), rp_tx._tx.el_aesa_iner(ptimes)).T
        # Compute the AESA phi and theta angles for commanding it
        aesa_phi_r, aesa_theta_r = rp_tx._tx.aesa_frame_phi_theta(ptimes[0])
        # Get the element weights for the designed AESA phi and theta
        elem_phases_r, elem_weights = get_element_phases(aesa_elem_pos_m, aesa_theta_r, aesa_phi_r, wavelength)
        weights_tx = np.expand_dims(elem_weights.flatten(), axis=0)
        # weights_rx = np.stack([x for xs in [[elem_weights[i, j].flatten() for j in range(elem_weights.shape[1])] for i in range(elem_weights.shape[0])] for x in xs], axis=0)
        weights_rx = np.expand_dims(elem_weights.flatten(), axis=0)
        txposes = np.expand_dims(rp_tx.txpos(ptimes)[..., :3].swapaxes(0, 1), axis=0)
        rxposes = np.stack([rp.rxpos(ptimes)[..., :3].swapaxes(0, 1) for rp in rp_rx], axis=0)
        block_data = trace_cpi(tracer, chirps, npulses, txposes, rxposes, weights_tx, weights_rx, rp_tx._tx.boresight(ptimes[0]), aesa_bore,
                               ptimes, nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el / 2,
                                cfig.tracer_params.pix_width, cfig.tracer_params.pix_height,
                                cfig.ant_params.transmit_power, cfig.ant_params.rx_gain, cfig.ant_params.tx_gain,
                               cfig.ant_params.rec_gain, cfig.ant_params.noise_figure,
                               cfig.ant_params.operating_temperature, fft_len, add_noise=True)
        block_data = np.sum(block_data, axis=0)

        # rp_data.append(upsamplePulse(block_data * mf_chirp, fft_len, cfig.tracer_params.upsample, is_freq=True,
        #                         time_len=nsam).astype(_complex_float))

        # Matched Filter
        rp_data = np.fft.ifft(block_data * mf_chirps, fft_len, axis=-1)[..., :nsam].astype(_complex_float)
        base_pos = rp_tx.pos(ptimes)
        '''dopp_correction = getDopplerLine(rp_tx.az_iner(ptimes).mean(), ranges,
                                         rp_tx.vel(ptimes).mean(axis=0),
                                         base_pos.mean(axis=0),
                                         rp_tx.el_iner(ptimes).mean() + bw_el / 2,
                                         bw_az / 2, prf, c0 / fc)
        dp_f, dp_times = np.meshgrid(dopp_correction[0], ptimes)
        dopp = np.exp(-1j * 2 * np.pi * dp_f * dp_times)'''
        # dopp_corr_data = np.fft.fftshift(np.fft.fft(rp_data * dopp * doppwin[:, None], axis=1), axes=1)
        dopp_corr_data = np.fft.fftshift(np.fft.fft(rp_data * doppwin[:, None], axis=1), axes=1)
        # sum_data = dopp_corr_data.sum(axis=0)
        sum_data = dopp_corr_data[0]
        '''az_del_data = (dopp_corr_data[0] + dopp_corr_data[1]) - (dopp_corr_data[2] + dopp_corr_data[3])
        el_del_data = (dopp_corr_data[0] + dopp_corr_data[2]) - (dopp_corr_data[1] + dopp_corr_data[3])
        det_sum = abs(sum_data)'''


        # print(f'{aesa_bore} - {boresight_mbs}')
        locations, index_ray, index_tri = spheres[0].intersects(
            ray_origins=np.repeat(txposes[0, 0, 0].reshape(1, -1), 2, axis=0),
            ray_directions=np.stack([aesa_bore[0], rp_tx._tx.boresight(ptimes[0])], axis=0)
        )
        aesa_az = np.arctan2(aesa_bore[0, 0], aesa_bore[0, 1])
        aesa_el = -np.arcsin(aesa_bore[0, 2])
        aesa_near_range = ranges[0] * np.cos(aesa_el)
        aesa_far_range = ranges[-1] * np.cos(aesa_el)

        neg_beamvec = np.array([np.sin(aesa_az - bw_az / 2),
                                np.cos(aesa_az - bw_az / 2)])
        pos_beamvec = np.array([np.sin(aesa_az + bw_az / 2),
                                np.cos(aesa_az + bw_az / 2)])
        beamx = np.array([neg_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_far_range, pos_beamvec[0] * aesa_far_range,
                  pos_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_near_range]) + base_pos[0, 0]
        beamy = np.array([neg_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_far_range, pos_beamvec[1] * aesa_far_range,
                          pos_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_near_range]) + base_pos[0, 1]

        neg_beamvec = np.array([np.sin(rp_tx.az_iner(ptimes).mean() - bw_az / 2),
                                np.cos(rp_tx.az_iner(ptimes).mean() - bw_az / 2)])
        pos_beamvec = np.array([np.sin(rp_tx.az_iner(ptimes).mean() + bw_az / 2),
                                np.cos(rp_tx.az_iner(ptimes).mean() + bw_az / 2)])
        aesa_near_range = ranges[0] * np.cos(rp_tx.el_iner(ptimes).mean())
        aesa_far_range = ranges[-1] * np.cos(rp_tx.el_iner(ptimes).mean())
        gimx = np.array(
            [neg_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_far_range, pos_beamvec[0] * aesa_far_range,
             pos_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_near_range]) + base_pos[0, 0]
        gimy = np.array(
            [neg_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_far_range, pos_beamvec[1] * aesa_far_range,
             pos_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_near_range]) + base_pos[0, 1]
        # plt.cla()


        sumax.cla()
        scatax.cla()
        sumax.set_title('Sum Beam')
        im = sumax.imshow(db(sum_data).T, origin='lower', cmap='jet', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]), aspect='auto')
        im.set_clim([-30, 15])
        sumax.set_ylabel('Range (m)')
        scatax.scatter(base_pos[:, 0], base_pos[:, 1], c='blue')
        for h in tracer.hulls:
            scatax.scatter(h.vertices[:, 0], h.vertices[:, 1], c='red')
        scatax.plot(beamx, beamy, c='blue')
        scatax.plot(gimx, gimy, c='red')
        scatax.scatter(locations[:, 0], locations[:, 1], c='orange')

        plt.draw()
        plt.pause(.01)

        '''im = ax.imshow(db(sum_data).T, origin='lower', cmap='jet',
                       extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]), aspect='auto', animated=True)
        im.set_clim([-30, 15])
        ims.append([im])'''

    import matplotlib.animation as anim
    ani = anim.ArtistAnimation(fig, ims, interval=150, blit=True)
    test = anim.FFMpegWriter(fps=5)
    ani.save('data.gif', writer=test)

    # plt.figure('Minimum Range Distance')
    # plt.imshow(range_grid)

    plt.figure('Chirps')
    plt.plot(db(fft_chirps[0]))
    plt.plot(db(mf_chirps[0]))

    cloud_fig = px.scatter_3d(x=e, y=n, z=u)
    for h in tracer.hulls:
        cloud_fig.add_mesh3d(x=h.vertices[:, 0],
                       y=h.vertices[:, 1],
                       z=h.vertices[:, 2],
                       # i, j and k give the vertices of triangles
                       i=h.faces[:, 0],
                       j=h.faces[:, 1],
                       k=h.faces[:, 2])
    cloud_fig.update_layout(
        scene=dict(xaxis=dict(range=[-100, 100]), yaxis=dict(range=[-1400, 100]), zaxis=dict(range=[-1, 1530])))
    cloud_fig.show()

    # rx_array = np.stack([rp_tx.txpos(rp_tx.gpst[0]) for rp in rps])
    maxes = txel_array[0].max()
    mins = txel_array[0].min()
    fig = px.scatter_3d(x=txel_array[0, :, 0], y=txel_array[0, :, 1], z=txel_array[0, :, 2])
    fig.add_scatter3d(x=tx_array[:, 0], y=tx_array[:, 1], z=tx_array[:, 2], mode='markers')
    fig.add_scatter3d(x=rx_array[:, 0], y=rx_array[:, 1], z=rx_array[:, 2], mode='markers')
    fig.update_layout(scene=dict(xaxis=dict(range=[mins, maxes]), yaxis=dict(range=[mins, maxes]), zaxis=dict(range=[mins, maxes])))
    fig.show()

    txa = rp_tx.txpos(ptimes[0])
    rxa = rp_rx[0].rxpos(ptimes[0])
    fig = px.scatter_3d(x=txa[:, 0], y=txa[:, 1], z=txa[:, 2])
    fig.add_scatter3d(x=rxa[:, 0], y=rxa[:, 1], z=rxa[:, 2], mode='markers')
    fig.show()

    for sph in spheres:
        fig = go.Figure(data=[go.Mesh3d(x=sph.mesh.vertices[:, 0], y=sph.mesh.vertices[:, 1], z=sph.mesh.vertices[:, 2],
                                        i=sph.mesh.faces[:, 0], j=sph.mesh.faces[:, 1], k=sph.mesh.faces[:, 2])])
        fig.update_layout(scene=dict(zaxis=dict(range=[sph.mesh.vertices[:, 2].min(), sph.mesh.vertices.max()])))
        fig.show()





