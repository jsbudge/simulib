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
from simulib.simulib.utils import c0, DTR, _complex_float, design_element_positions, get_element_phases, get_aesa_phi_theta
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


def std_filter(arr):
    """Function to be applied by generic_filter to calculate std dev."""
    return np.std(arr)
pio.renderers.default = 'browser'

def getRParams(a_fs, a_fc, a_plp, a_upsample, near_rng, far_rng, a_npulses, a_dopp_upsample, bandwidth, is_dechirp: bool = False):
    max_nr = int(near_rng * 2 / c0 * a_fs)
    nr = min(int(max_nr * a_plp), 4096)
    chirp_rate = bandwidth / (nr / a_fs)
    nsam = int(2 * (far_rng - near_rng) * chirp_rate * nr / (c0 * a_fs)) + nr if is_dechirp else int((far_rng - near_rng) / (c0 / 2) * a_fs)
    # nr = min(int(2 * (far_rng - near_rng) * bandwidth / (a_fs * c0) * a_fs) + 1, int(near_rng * 2 / c0 * a_plp * a_fs)
    dechirp_bandwidth = chirp_rate * nsam / a_fs
    fft_len = 2 ** int(np.ceil(np.log2(nsam + nr)))
    chirp = genChirp(nr, a_fs, a_fc, bandwidth)
    fft_chirp = np.fft.fft(chirp, fft_len)
    MPP = c0 * a_fs / (2 * nr * chirp_rate) if is_dechirp else c0 * a_upsample / (a_fs * 2)
    dechirp = genChirp(nsam, a_fs, a_fc, dechirp_bandwidth) if is_dechirp else None
    # near_rng -= MPP * (dechirp_bandwidth - bandwidth) / 2 / chirp_rate * a_fs if is_dechirp else 0.
    if is_dechirp:
        mwing = .55 * (bandwidth if bandwidth * 2 < a_fs / 2 else a_fs / 2 - (dechirp_bandwidth - bandwidth) / 2)
        near_rng = (near_rng / c0 - mwing / chirp_rate) * c0
    ranges = near_rng + np.arange(nsam) * MPP + c0 / a_fs
    granges = np.sqrt(ranges ** 2 - launch_height ** 2)
    near_range_s = near_rng / c0
    mf_chirp = None if is_dechirp else genTaylorWindow((a_fc % a_fs), bandwidth / 2, a_fs, fft_len) / fft_chirp
    doppwin = taylor(a_npulses * a_dopp_upsample, nbar=11, sll=90)
    mod_fs = a_fs if is_dechirp else 1 / ((ranges[1] - ranges[0]) * 2 / c0)
    return nsam, nr, ranges, granges, near_range_s, fft_len, chirp, fft_chirp, mf_chirp, np.fft.fftshift(doppwin), dechirp, mod_fs, chirp_rate

def point_weights(az, el, txoffsets, rxoffsets, wavelengt):
    weights_tx = np.stack([np.exp(-2j * np.pi * (txo @ azelToVec(az, el)) / wavelengt) for txo in txoffsets], axis=0)
    # weights_tx = np.ones((1, 200))
    # weights_rx = np.ones((8, 25))
    weights_rx = np.stack([np.exp(-2j * np.pi * (rxo @ azelToVec(az, el)) / wavelengt) for rxo in rxoffsets], axis=0)
    return weights_tx, weights_rx



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
    n_eigs = 1
    angle_npts = (64, 64)
    wavelength = c0 / fc

    # Generate a platform
    print('Generating platform...', end='')

    # Position of plane/heli
    n_total_gps = int(cfig.sim_params.time * 100)
    gps_times = np.arange(n_total_gps) / 100
    e = np.zeros(n_total_gps) - 50. + np.linspace(0, 100, n_total_gps)
    n = np.zeros(n_total_gps) - 1400
    u = np.zeros(n_total_gps) + 1524
    r = np.zeros(n_total_gps) + np.pi / 2
    p = np.zeros(n_total_gps)
    y = np.zeros(n_total_gps)

    # Position of gimbal
    gim_pan = np.zeros_like(gps_times)  # sawtooth(gps_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T

    dep_ang = (-np.arcsin(u[0] / np.sqrt(u[0]**2 + n[0]**2 + e[0]**2)) - bw_el * 2) / DTR

    pulse_times = np.arange(int(gps_times[-1] * prf)) / prf

    pointing_az = sawtooth(pulse_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    aesa_elem_pos_m, aesa_width_m, aesa_height_m, element_patch_size_m = \
        design_element_positions(fc, 5, 4, 4, 2)
    tx_el_offsets = [aesa_elem_pos_m.swapaxes(2, 4).reshape((-1, 3))]
    rx_el_offsets = [x for xs in [[aesa_elem_pos_m[i, j].swapaxes(0, 2).reshape(-1, 3) for j in range(aesa_elem_pos_m.shape[1])] for i in range(aesa_elem_pos_m.shape[0])] for x in xs]

    weights_tx, weights_rx = point_weights(0, 0, tx_el_offsets, rx_el_offsets, wavelength)
    goffsets = np.array([0, np.pi / 2, 0.])
    # rxel_array = np.stack([rxe.dot(getRotationOffsetMatrix(*goffsets)) for rxe in rx_el_offsets], axis=0)
    rxel_array = np.stack([rxe for rxe in rx_el_offsets], axis=0)
    rx_array = rxel_array.mean(axis=1)
    txel_array = np.stack([txe for txe in tx_el_offsets], axis=0)
    tx_array = txel_array.mean(axis=1)
    vx_array = np.concatenate([rx_array + tx for tx in tx_array])

    rp_rx = [RadarPlatform(e, n, u, r, p, y, gps_times, np.array([0, 0, 0.]), rxo, gimbal=gimbal,
                  gimbal_offset=np.array([0., 1., 0.]), gimbal_rotations=goffsets, dep_angle=dep_ang,
                  az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth) for rxo in rx_el_offsets]
    rp_tx = RadarPlatform(e, n, u, r, p, y, gps_times, tx_el_offsets[0], np.array([0, 0, 0.]), gimbal=gimbal,
                  gimbal_offset=np.array([0., 1., 0.]), gimbal_rotations=goffsets, dep_angle=dep_ang,
                  az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth)

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp_tx.getRadarParams(u.mean(), .5)

    chirps = genChirp(nr, fs, fc, bandwidth).reshape((1, -1)) * 1000.
    fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)  # [np.fft.fft(chirp, fft_len) for chirp in chirps]
    mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps  # [genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirp for fft_chirp in fft_chirps]
    doppwin = taylor(npulses * dopp_upsample, nbar=11, sll=90)

    dopp_freq = np.fft.fftshift(np.fft.fftfreq(len(doppwin), 1. / prf))

    # Steering vector for overall receive beam
    # k_wave = azelToVec(gim_az, gim_el) * 2 * np.pi / (c0 / fc)
    # v_k = np.exp(-1j * np.dot(k_wave[None, :], rx_offsets.T)).flatten()

    # villa = tri.load(f'/home/jeff/Documents/rock_ring/scene.gltf', force='mesh')
    spheres = []
    materials = []
    for s in range(1):
        villa = tri.creation.box(extents=np.array([10000, 10000, 1]))
        # villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
        # villa.apply_scale(10.)
        villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) - np.array([0, 0, 1]))
        villa_mats = np.zeros((villa.triangles.shape[0], 2))
        villa_mats[:, 0] = 1e6
        villa_mats[:, 1] = .017
        spheres.append(BaseMesh(villa, 1, villa_mats, do_sample=False, motion_keys=None))
        materials.append([1e6, .0017])

    boat_motion_keys = []

    boat_motion_vector = azelToVec(-45 * DTR, 0.) * 10.2889 * np.diff(pulse_times[::100])[1]

    for idx, o in enumerate(pulse_times[::100]):
        boat_motion_keys += [
            1.0, 0.0, 0.0, idx * boat_motion_vector[0],
            0.0, 1.0, 0.0, idx * boat_motion_vector[1],
            0.0, 0.0, 1.0, idx * boat_motion_vector[2],
        ]
    villa = tri.load(f'/home/jeff/Documents/target_meshes/helic.obj', force='mesh')
    villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
    villa.apply_scale(10.)
    villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) + np.array([0, 2000, 15]))
    villa_mats = np.zeros((villa.triangles.shape[0], 2))
    villa_mats[:, 0] = 1e6
    villa_mats[:, 1] = .017
    spheres.append(BaseMesh(villa, 1, villa_mats, do_sample=False, motion_keys=boat_motion_keys))
    materials.append([1e6, .0017])


    materials = np.stack(materials)

    print('Launching...')

    '''fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig)
    sumax = fig.add_subplot(gs[0, 0])
    scatax = fig.add_subplot(gs[0, 1])'''
    fig, ax = plt.subplots()
    ims = []

    tracer = build_tracer(spheres, pulse_times=pulse_times)

    scan_kernel = np.ones((45, 45))
    scan_kernel[8:37, 8:37] = 0
    scan_kernel = scan_kernel / np.sum(scan_kernel)
    clutter_window = 1 - taylor(12)

    # Vectors for angle detection
    az_mesh = np.linspace(-bw_az, bw_az, angle_npts[0])
    el_mesh = np.linspace(-bw_el, bw_el, angle_npts[1])
    azes, eles = np.meshgrid(az_mesh, el_mesh)
    fine_az_mesh = azes.flatten()
    fine_el_mesh = eles.flatten()
    ublock = azelToVec(fine_az_mesh, fine_el_mesh)
    fine_ucavec = np.exp(-2j * np.pi * fc / c0 * vx_array.dot(ublock)).T
    fine_ucavec /= np.linalg.norm(fine_ucavec, axis=1)[:, None]

    print(f'Launching {cfig.tracer_params.pix_height * cfig.tracer_params.pix_width} rays.')
    pcloud0 = []
    pcloud1 = []

    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
        if len(ptimes) < npulses:
            break
        boresight_mbs = azelToVec(-np.pi / 2, pointing_az[frame[0]] - np.pi/ 2)
        boresight_mbs /= np.linalg.norm(boresight_mbs)
        # Compute the AESA phi and theta angles for commanding it
        aesa_phi_r, aesa_theta_r = get_aesa_phi_theta(boresight_mbs)
        # Get the element weights for the designed AESA phi and theta
        elem_phases_r, elem_weights = get_element_phases(
            aesa_elem_pos_m, aesa_theta_r, aesa_phi_r, wavelength)
        weights_tx = np.expand_dims(elem_weights.flatten(), axis=0)
        weights_rx = np.stack([x for xs in [[elem_weights[i, j].flatten() for j in range(elem_weights.shape[1])] for i in range(elem_weights.shape[0])] for x in xs], axis=0)
        txposes = np.expand_dims(rp_tx.txpos(ptimes)[..., :3].swapaxes(0, 1), axis=0)
        rxposes = np.stack([rp.rxpos(ptimes)[..., :3].swapaxes(0, 1) for rp in rp_rx], axis=0)
        block_data = trace_cpi(tracer, chirps,
                                npulses, txposes, rxposes, weights_tx, weights_rx,
                                rp_tx._tx.boresight(ptimes).T, materials, ptimes,
                                nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el / 2,
                                cfig.tracer_params.pix_width, cfig.tracer_params.pix_height,
                                cfig.ant_params.transmit_power,
                                cfig.ant_params.rx_gain, cfig.ant_params.tx_gain, cfig.ant_params.rec_gain,
                                cfig.ant_params.noise_figure, cfig.ant_params.operating_temperature, fft_len,
                                add_noise=True)
        '''block_data = [trace_cpi(tracer, chirp,
                                npulses, txpos.reshape((1, 1, *txpos.shape)), rxposes, weights,
                                rp_tx._tx.boresight(ptimes).T, materials, ptimes,
                                nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el / 2,
                                cfig.tracer_params.pix_width, cfig.tracer_params.pix_height, cfig.ant_params.transmit_power,
                                cfig.ant_params.rx_gain, cfig.ant_params.tx_gain, cfig.ant_params.rec_gain,
                                cfig.ant_params.noise_figure, cfig.ant_params.operating_temperature, fft_len,
                                add_noise=True) for chirp, txpos in zip(chirps, txposes)]'''
        block_data = np.sum(block_data, axis=0)

        # rp_data.append(upsamplePulse(block_data * mf_chirp, fft_len, cfig.tracer_params.upsample, is_freq=True,
        #                         time_len=nsam).astype(_complex_float))

        # Matched Filter
        rp_data = np.fft.ifft(block_data * mf_chirps, fft_len, axis=-1)[..., :nsam].astype(_complex_float)
        base_pos = rp_tx.pos(ptimes)
        dopp_correction = getDopplerLine(rp_tx.az_iner(ptimes).mean(), ranges,
                                         rp_tx.vel(ptimes).mean(axis=0),
                                         base_pos.mean(axis=0),
                                         rp_tx.el_iner(ptimes).mean() + bw_el / 2,
                                         bw_az / 2, prf, c0 / fc)
        dp_f, dp_times = np.meshgrid(dopp_correction[0], ptimes)
        dopp = np.exp(-1j * 2 * np.pi * dp_f * dp_times)
        # dopp_corr_data = np.fft.fftshift(np.fft.fft(rp_data * dopp * doppwin[:, None], axis=1), axes=1)
        dopp_corr_data = np.fft.fftshift(np.fft.fft(rp_data, axis=1), axes=1)
        sum_data = dopp_corr_data.sum(axis=0)
        '''az_del_data = (dopp_corr_data[0] + dopp_corr_data[1]) - (dopp_corr_data[2] + dopp_corr_data[3])
        el_del_data = (dopp_corr_data[0] + dopp_corr_data[2]) - (dopp_corr_data[1] + dopp_corr_data[3])
        det_sum = abs(sum_data)'''

        neg_beamvec = np.array([np.sin(rp_tx.az_iner(ptimes).mean() - bw_az / 2),
                                np.cos(rp_tx.az_iner(ptimes).mean() - bw_az / 2)])
        pos_beamvec = np.array([np.sin(rp_tx.az_iner(ptimes).mean() + bw_az / 2),
                                np.cos(rp_tx.az_iner(ptimes).mean() + bw_az / 2)])
        beamx = np.array([neg_beamvec[0] * granges[0], neg_beamvec[0] * granges[-1], pos_beamvec[0] * granges[-1],
                  pos_beamvec[0] * granges[0], neg_beamvec[0] * granges[0]]) + base_pos[0, 0]
        beamy = np.array([neg_beamvec[1] * granges[0], neg_beamvec[1] * granges[-1], pos_beamvec[1] * granges[-1],
                          pos_beamvec[1] * granges[0], neg_beamvec[1] * granges[0]]) + base_pos[0, 1]
        # plt.cla()


        '''sumax.cla()
        scatax.cla()
        sumax.set_title('Sum Beam')
        sumax.imshow(db(sum_data).T, origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]), aspect='auto')
        sumax.set_ylabel('Range (m)')
        scatax.scatter(base_pos[:, 0], base_pos[:, 1], c='blue')
        for h in tracer.hulls:
            scatax.scatter(h.vertices[:, 0], h.vertices[:, 1], c='red')
        scatax.plot(beamx, beamy, c='blue')
        plt.draw()'''
        im = ax.imshow(db(sum_data).T, origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]),
                     aspect='auto', animated=True)
        ims.append([im])
        # plt.pause(.01)


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





