#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from config import load_yaml_config
from mesh_utils import BaseMesh
from scipy.signal import sawtooth
from scene_tracer import build_tracer, trace_beampattern
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec
from simulib.simulib.platform_helper import RadarPlatform
from simulib.simulib.utils import c0, DTR, _complex_float, design_element_positions, get_element_phases
from simulib.simulib.rotation_functions import get_aesa_pointing_from_phi_theta
import trimesh as tri
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px

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
    wavelength = c0 / fc

    # Generate a platform
    print('Generating platform...', end='')

    # Position of plane/heli
    n_total_gps = int(cfig.sim_params.time * 100)
    gps_times = np.arange(n_total_gps) / 100
    e = np.zeros(n_total_gps)
    n = np.zeros(n_total_gps) - 1400
    u = np.zeros(n_total_gps) + 1524
    r = np.zeros(n_total_gps)
    p = np.zeros(n_total_gps)
    y = np.zeros(n_total_gps) + np.pi / 2 # np.arctan2(np.gradient(e), np.gradient(n))

    # Position of gimbal
    gim_pan = np.zeros_like(gps_times) # sawtooth(gps_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T

    # prf = calcDopplerPRF(np.linalg.norm(np.gradient(np.array([e, n, u]).T, axis=0), axis = 1).mean() * 100., fc, bandwidth, bw_az)

    pulse_times = np.arange(int(gps_times[-1] * prf)) / prf

    pointing_az = sawtooth(gps_times * cfig.radar_params.scan_rate / cfig.radar_params.scan_limit * 2 * np.pi, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    aesa = get_aesa_pointing_from_phi_theta(np.zeros_like(pointing_az), pointing_az).T
    # aesa = get_aesa_pointing_from_phi_theta(np.zeros_like(pointing_az), np.zeros_like(pointing_az)).T
    aesa_elem_pos_m, aesa_width_m, aesa_height_m, element_patch_size_m = \
        design_element_positions(fc, 5, 4, 4, 2)
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

    rp_rx = [RadarPlatform(e, n, u, r, p, y, gps_times, np.array([0, 0, 0.]), rxo, gimbal=gimbal,
                           gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                           az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa) for rxo in rx_el_offsets]
    rp_tx = RadarPlatform(e, n, u, r, p, y, gps_times, tx_el_offsets[0], np.array([0, 0, 0.]), gimbal=gimbal,
                          gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                          az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa)

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp_tx.getRadarParams(u.mean(), .25)

    chirps = genChirp(nr, fs, fc, bandwidth).reshape((1, -1)) * 100.
    fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)
    mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps

    # villa = tri.load(f'/home/jeff/Documents/taiwan_hills/scene.gltf', force='mesh')
    spheres = []
    materials = []
    for s in range(1):
        villa = tri.creation.box(extents=np.array([10000, 10000, 1]))
        # villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
        # villa.apply_scale(100.)
        villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) + np.array([0, 2000, 1]))
        villa_mats = np.zeros((villa.triangles.shape[0], 2))
        villa_mats[:, 0] = 1e1
        villa_mats[:, 1] = .0017
        spheres.append(BaseMesh(villa, 1, villa_mats, do_sample=False, motion_keys=None))
        materials.append([1e1, .00017])

    gx, gy = np.meshgrid(np.linspace(-5000, 5000, cfig.tracer_params.pix_width), np.linspace(-3000, 7000, cfig.tracer_params.pix_height))
    gz = np.zeros_like(gx) + 1.5
    gxyz = np.stack([gx, gy, gz], axis=0)


    materials = np.stack(materials)

    print('Launching...')

    fig, ax = plt.subplots()
    ims = []

    tracer = build_tracer(spheres, cu_file='beampattern.cu', pulse_times=pulse_times)

    print(f'Launching {cfig.tracer_params.pix_height * cfig.tracer_params.pix_width} rays.')
    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
        if len(ptimes) < npulses:
            break
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
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
        gpower, cam_u, cam_v, cam_w = trace_beampattern(tracer, chirps, gxyz, npulses, txposes, rxposes, weights_tx, weights_rx, rp_tx._tx.boresight(ptimes[0]), aesa_bore, materials,
                               ptimes, nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el / 2,
                                cfig.tracer_params.pix_width, cfig.tracer_params.pix_height,
                                cfig.ant_params.transmit_power, cfig.ant_params.rx_gain, cfig.ant_params.tx_gain,
                               cfig.ant_params.rec_gain, cfig.ant_params.noise_figure,
                               cfig.ant_params.operating_temperature, fft_len, add_noise=False)

        locations, index_ray, index_tri = villa.ray.intersects_location(
            ray_origins=np.repeat(txposes[0, 0, 0].reshape(1, -1), 2, axis=0),
            ray_directions=np.stack([aesa_bore[0], rp_tx._tx.boresight(ptimes[0])], axis=0)
        )

        plt.cla()
        ax.imshow(gpower, origin = 'lower', extent=[gx.min(), gx.max(), gy.min(), gy.max()], cmap='jet')
        ax.scatter([locations[:, 0]], [locations[:, 1]])
        plt.draw()
        plt.pause(0.01)

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

    grid_vecs = np.array([gx.flatten() - e[0], gy.flatten() - n[0], gz.flatten() - u[0]]).T
    rot_mat = np.stack([cam_u[0], cam_v[0], cam_w[0]])
    grid_vecs = grid_vecs / np.linalg.norm(grid_vecs, axis=1)[:, None]
    grid_mbs = grid_vecs @ rot_mat
    grid_redux = grid_mbs @ rot_mat.T

    plt.figure('UV')
    plt.scatter(grid_mbs[:, 0], grid_mbs[:, 1], c=gpower.flatten())
    plt.show()

    plt.figure('Inertial')
    plt.scatter(grid_redux[:, 0], grid_redux[:, 1], c=gpower.flatten())
    plt.show()





