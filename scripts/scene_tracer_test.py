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
from backproject_utils import backprojectPulseStream
import trimesh as tri
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

GPS_UPDATE_HZ = 100

def aliasFrequency(f, a_fs):
    return f - int(f / (a_fs / 2)) * a_fs / 2


def calcDopplerPRF(v, fc, bandwidth, half_az_bw):
    return 4 * v * np.sin(half_az_bw) * (fc + bandwidth / 2) / c0


if __name__ == "__main__":
    cfig = load_yaml_config('/home/jeff/repo/optix_radartracer/simulib/scripts/aesa_test_params.yaml')
    plp = cfig.radar_params.plp
    upsample = cfig.tracer_params.upsample
    npulses = 16384
    prf_broadening_factor = cfig.radar_params.broadening_factor
    fc = 9.6e9
    fs = 2e9
    bandwidth = 600e6
    sim_time = 3.
    wavelength = c0 / fc

    # Generate a platform
    print('Generating platform...', end='')

    # Position of plane/heli
    n_total_gps = int(sim_time * GPS_UPDATE_HZ)
    gps_times = np.arange(n_total_gps) / GPS_UPDATE_HZ
    e = np.zeros(n_total_gps) - 200. + np.linspace(0, 400, n_total_gps)
    n = np.zeros(n_total_gps) - 1400
    u = np.zeros(n_total_gps) + 1524
    r = np.zeros(n_total_gps)
    p = np.zeros(n_total_gps)
    y = np.arctan2(np.gradient(e), np.gradient(n))

    # Position of gimbal
    gim_pan = np.zeros_like(gps_times) # sawtooth(gps_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T

    # Design the antenna
    ant = AESA(fc, 20, 20, 1, 1)
    aesa = np.zeros((len(gps_times), 2))
    gimbal_rotations = np.array([180. * DTR, 40 * DTR, -np.pi / 2])
    gimbal_offsets = np.array([.0753, 1.6053, -.7873])
    aesa_pointing = np.array([0, -1800, 1524])
    bw_az, bw_el = ant.calc_beamwidth(0., 0.)

    dep_ang = 50.

    prf = calcDopplerPRF(np.linalg.norm(np.gradient(np.array([e, n, u]).T, axis=0), axis=1).mean() * GPS_UPDATE_HZ, fc,
                         bandwidth, bw_az) * prf_broadening_factor
    pulse_times = np.arange(int(gps_times[-1] * prf)) / prf

    rp = RadarPlatform(e, n, u, r, p, y, gps_times, ant.phase_center_offsets, ant.phase_center_offsets, gimbal=gimbal,
                           gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                           az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa)

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(u.mean(), plp, upsample)

    chirps = genChirp(nr, fs, fc, bandwidth).reshape((1, -1))
    fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)
    mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps

    villa = tri.load(f'/home/jeff/Documents/roman_facade/scene.gltf').to_mesh()
    spheres = []

    # villa = tri.creation.box(extents=np.array([10000, 10000, 1]))
    villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
    # villa.apply_transform(tri.transformations.rotation_matrix(np.pi, np.array([0., 0., 1.]), np.array([0, 0, 0.])))
    # villa.apply_scale(100.)
    villa.apply_transform(tri.transformations.translation_matrix(-villa.bounding_box.bounds.mean(axis=0)))
    stretch_matrix = np.array([
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # villa.apply_transform(stretch_matrix)
    npts = 800

    gx, gy = np.meshgrid(np.linspace(-200, 200, npts), np.linspace(-200, 200, npts))
    bpj_grid = np.zeros(gx.shape, dtype=_complex_float)
    gx = gx.flatten()
    gy = gy.flatten()

    grid_pts = np.stack((gx, gy, np.zeros_like(gx)), axis=-1)
    grid_pts[:, 2] = 1524.
    locations, index_ray, index_tri = villa.ray.intersects_location(
        ray_origins=grid_pts,
        ray_directions=np.zeros_like(grid_pts) + np.array([0, 0, -1]),
    )

    grid_pts[:, 2] = locations.mean(axis=0)[2]
    gx = grid_pts[:, 0].reshape((npts, npts))
    gy = grid_pts[:, 1].reshape((npts, npts))
    gz = grid_pts[:, 2].reshape((npts, npts))

    grid_power = np.ones_like(grid_pts)

    spheres.append(BaseMesh(villa, motion_keys=None))

    print('Launching...')

    # tracer = build_tracer(spheres, pulse_times=pulse_times, debug=False)
    tracer = build_tracer(spheres, debug=False)

    print(f'Launching {cfig.tracer_params.pix_height * cfig.tracer_params.pix_width} rays.')
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
        block_data = trace_scene(tracer, chirps, txposes, rxposes, weights_tx, weights_rx, rp.tx.boresight(ptimes[0]), aesa_bore,
                               grid_pts, grid_power, ptimes, nsam, fc, fs, near_range_s, ranges[-1], bw_az, bw_el,
                               cfig.tracer_params.pix_width, cfig.tracer_params.pix_height,
                               cfig.ant_params.transmit_power, cfig.ant_params.rx_gain, cfig.ant_params.tx_gain,
                               cfig.ant_params.rec_gain, cfig.ant_params.noise_figure,
                               cfig.ant_params.operating_temperature, fft_len, add_noise=False)
        block_data = np.sum(block_data, axis=0)

        # rp_data.append(upsamplePulse(block_data * mf_chirp, fft_len, cfig.tracer_params.upsample, is_freq=True,
        #                         time_len=nsam).astype(_complex_float))

        # Matched Filter
        rp_data = upsamplePulse(block_data * mf_chirps, fft_len, upsample, is_freq=True,
                                time_len=nsam).astype(_complex_float)
        # rp_data = np.fft.ifft(block_data * mf_chirps, fft_len, axis=-1)[..., :nsam].astype(_complex_float)
        base_pos = rp.pos(ptimes)

        bpj_grid += backprojectPulseStream([rp_data[0]], [rp.tx.az_aesa_iner(ptimes)], [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                           [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength), _float(near_range_s), _float(fs * upsample), bw_az,
                                           gx=gx, gy=gy)

        aesa_az = np.arctan2(aesa_bore[0, 0], aesa_bore[0, 1])
        aesa_el = -np.arcsin(aesa_bore[0, 2])
        aesa_near_range = ranges[0] * np.cos(aesa_el)
        aesa_far_range = ranges[-1] * np.cos(aesa_el)

        neg_beamvec = np.array([np.sin(aesa_az - bw_az),
                                np.cos(aesa_az - bw_az)])
        pos_beamvec = np.array([np.sin(aesa_az + bw_az),
                                np.cos(aesa_az + bw_az)])
        beamx = np.array([neg_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_far_range, pos_beamvec[0] * aesa_far_range,
                  pos_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_near_range]) + base_pos[0, 0]
        beamy = np.array([neg_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_far_range, pos_beamvec[1] * aesa_far_range,
                          pos_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_near_range]) + base_pos[0, 1]

        neg_beamvec = np.array([np.sin(rp.az_iner(ptimes).mean() - bw_az),
                                np.cos(rp.az_iner(ptimes).mean() - bw_az)])
        pos_beamvec = np.array([np.sin(rp.az_iner(ptimes).mean() + bw_az),
                                np.cos(rp.az_iner(ptimes).mean() + bw_az)])
        aesa_near_range = ranges[0] * np.cos(rp.el_iner(ptimes).mean())
        aesa_far_range = ranges[-1] * np.cos(rp.el_iner(ptimes).mean())
        gimx = np.array(
            [neg_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_far_range, pos_beamvec[0] * aesa_far_range,
             pos_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_near_range]) + base_pos[0, 0]
        gimy = np.array(
            [neg_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_far_range, pos_beamvec[1] * aesa_far_range,
             pos_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_near_range]) + base_pos[0, 1]
        # plt.cla()


        '''sumax.cla()
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
        plt.pause(.01)'''

    # plt.figure('Minimum Range Distance')
    # plt.imshow(range_grid)

    plt.figure('Chirps')
    plt.plot(db(fft_chirps[0]))
    plt.plot(db(mf_chirps[0]))

    plt.figure('Backprojection')
    plt.imshow(db(bpj_grid), extent=[-200, 200, -100, 100], origin='lower', cmap='gray')
    # plt.clim([-5, 40])

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
        scene=dict(xaxis=dict(range=[-100, 100]), yaxis=dict(range=[-1400, 300]), zaxis=dict(range=[-1, 1530])))
    cloud_fig.show()

    # rx_array = np.stack([rp.txpos(rp.gpst[0]) for rp in rps])
    maxes = ant.phase_center_offsets[0].max()
    mins = ant.phase_center_offsets[0].min()
    fig = px.scatter_3d(x=ant.phase_center_offsets[0, :, 0], y=ant.phase_center_offsets[0, :, 1], z=ant.phase_center_offsets[0, :, 2])
    fig.add_scatter3d(x=ant.phase_centers[:, 0], y=ant.phase_centers[:, 1], z=ant.phase_centers[:, 2], mode='markers')
    fig.add_scatter3d(x=ant.phase_centers[:, 0], y=ant.phase_centers[:, 1], z=ant.phase_centers[:, 2], mode='markers')
    fig.update_layout(scene=dict(xaxis=dict(range=[mins, maxes]), yaxis=dict(range=[mins, maxes]), zaxis=dict(range=[mins, maxes])))
    fig.show()

    for sph in spheres:
        if isinstance(sph.mesh, tri.Scene):
            fig = go.Figure(data=[go.Mesh3d(x=g.vertices[:, 0], y=g.vertices[:, 1], z=g.vertices[:, 2],
                                            i=g.faces[:, 0], j=g.faces[:, 1], k=g.faces[:, 2]) for g in sph.mesh.geometry.values()])
        else:
            fig = go.Figure(data=[go.Mesh3d(x=sph.mesh.vertices[:, 0], y=sph.mesh.vertices[:, 1], z=sph.mesh.vertices[:, 2],
                                            i=sph.mesh.faces[:, 0], j=sph.mesh.faces[:, 1], k=sph.mesh.faces[:, 2])])
        # fig.update_layout(scene=dict(zaxis=dict(range=[sph.mesh.vertices[:, 2].min(), sph.mesh.vertices.max()])))
        fig.show()





