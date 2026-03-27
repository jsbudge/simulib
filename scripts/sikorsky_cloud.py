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
import trimesh as tri
import matplotlib.pyplot as plt
import matplotlib as mplib
from simulib.simulib.rotation_functions import get_aesa_pointing_from_phi_theta
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
    prf = cfig.radar_params.prf
    bandwidth = cfig.radar_params.bandwidth
    n_eigs = 1
    angle_npts = (64, 64)
    wavelength = c0 / fc

    # Generate a platform
    print('Generating platform...', end='')

    # Position of plane/heli
    n_total_gps = int(cfig.sim_params.time * 100)
    gps_times = np.arange(n_total_gps) / 100
    e = np.zeros(n_total_gps) + 150
    n = np.zeros(n_total_gps) - 1400 + np.linspace(0, 100, n_total_gps)
    u = np.zeros(n_total_gps)
    r = np.zeros(n_total_gps)
    p = np.zeros(n_total_gps)
    y = np.arctan2(np.gradient(e), np.gradient(n))

    # Position of gimbal
    gim_pan = np.zeros_like(
        gps_times)  # sawtooth(gps_times / cfig.radar_params.scan_rate, width=.5) * cfig.radar_params.scan_limit * DTR / 2
    gim_el = np.zeros_like(gps_times)
    gimbal = np.array([gim_pan, gim_el]).T
    pointing_az = sawtooth(gps_times * cfig.radar_params.scan_rate / cfig.radar_params.scan_limit * 2 * np.pi,
                           width=.5) * cfig.radar_params.scan_limit * DTR / 2
    aesa = get_aesa_pointing_from_phi_theta(np.zeros_like(pointing_az), pointing_az).T

    # Transmit array
    aesa_elem_tx, aesa_width_m, aesa_height_m, element_patch_size_m = \
        design_element_positions(fc, 10, 16, 2, 1)
    tx_el_offsets = [x for xs in [
        [aesa_elem_tx[i, j].swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 3) for j in range(aesa_elem_tx.shape[1])]
        for i in range(aesa_elem_tx.shape[0])] for x in xs]

    # Receive array
    aesa_elem_rx, aesa_width_m, aesa_height_m, element_patch_size_m = \
        design_element_positions(fc, 10, 8, 2, 2)
    rx_el_offsets = [x for xs in [
        [aesa_elem_rx[i, j].swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 3) for j in range(aesa_elem_rx.shape[1])]
        for i in range(aesa_elem_rx.shape[0])] for x in xs]
    bw_az = 115 / (aesa_width_m / wavelength) * DTR / 2
    bw_el = 115 / (aesa_height_m / wavelength) * DTR / 2

    pulse_times = np.arange(int(gps_times[-1] * prf)) / prf

    dep_ang = 0.

    gimbal_rotations = np.array([180. * DTR, np.pi / 2, 0.0])
    gimbal_offsets = np.array([0.0, 1.6053, 1.2])
    rxel_array = np.stack([rxe for rxe in rx_el_offsets], axis=0)
    rx_array = rxel_array.mean(axis=1)
    txel_array = np.stack([txe for txe in tx_el_offsets], axis=0)
    tx_array = txel_array.mean(axis=1)
    vx_array = np.concatenate([rx_array + tx for tx in tx_array])

    rp_rx = [RadarPlatform(e, n, u, r, p, y, gps_times, np.array([0, 0, 0.]), rxo, gimbal=gimbal,
                           gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                           az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa) for
             rxo in rx_el_offsets]
    rp_tx = [RadarPlatform(e, n, u, r, p, y, gps_times, txo, np.array([0, 0, 0.]), gimbal=gimbal,
                          gimbal_offset=gimbal_offsets, gimbal_rotations=gimbal_rotations, dep_angle=dep_ang,
                          az_bw=bw_az / DTR, el_bw=bw_el / DTR, fs=fs, fc=fc, prf=prf, bwidth=bandwidth, aesa=aesa) for
             txo in tx_el_offsets]

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp_tx[0].getRadarParams(u.mean(), .5, a_ranges=init_ranges)

    chirps = np.array([genChirp(nr, fs, fc, bandwidth) * 1000.,
              genPulse(np.linspace(0, 1, 10), np.linspace(1, 0, 10), nr, fs, fc, bandwidth) * 1000.])
    fft_chirps = np.fft.fft(chirps, fft_len, axis=-1)  # [np.fft.fft(chirp, fft_len) for chirp in chirps]
    mf_chirps = genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirps  # [genTaylorWindow((fc % fs), bandwidth / 2, fs, fft_len) / fft_chirp for fft_chirp in fft_chirps]
    doppwin = taylor(npulses * dopp_upsample, nbar=11, sll=90)

    dopp_freq = np.fft.fftshift(np.fft.fftfreq(len(doppwin), 1. / prf))

    # Steering vector for overall receive beam
    # k_wave = azelToVec(gim_az, gim_el) * 2 * np.pi / (c0 / fc)
    # v_k = np.exp(-1j * np.dot(k_wave[None, :], rx_offsets.T)).flatten()

    # villa = tri.load('/home/jeff/Documents/roman_facade/scene.gltf', force='mesh')
    spheres = []
    materials = []
    for s in range(4):
        villa = tri.creation.icosphere(3, radius=10)
        villa.apply_transform(tri.transformations.rotation_matrix(np.pi / 2, np.array([1., 0., 0]), np.array([0, 0, 0.])))
        # villa.apply_scale(10.)
        villa.apply_translation(-villa.bounding_box.bounds.mean(axis=0) + np.array([s * 100, s * 60, 0]))
        spheres.append(BaseMesh(villa, motion_keys=None))

    print('Launching...')

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    sumax = fig.add_subplot(gs[0, 0])
    elax = fig.add_subplot(gs[0, 1])
    azax = fig.add_subplot(gs[1, 0])
    scatax = fig.add_subplot(gs[:, 2])
    cbax = fig.add_subplot(gs[1, 1])
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
        aesa_bore = azelToVec(rp_tx[0].tx.az_aesa_iner(ptimes), rp_tx[0].tx.el_aesa_iner(ptimes)).T
        # Compute the AESA phi and theta angles for commanding it
        aesa_phi_r, aesa_theta_r = rp_tx[0].tx.aesa_frame_phi_theta(ptimes[0])
        # Get the element weights for the designed AESA phi and theta
        _, elem_weights = get_element_phases(aesa_elem_tx, aesa_theta_r, aesa_phi_r, wavelength)
        weights_tx = np.stack([x for xs in [[elem_weights[i, j].flatten() for j in range(elem_weights.shape[1])] for i in range(elem_weights.shape[0])] for x in xs], axis=0)
        _, elem_weights = get_element_phases(aesa_elem_tx, aesa_theta_r, aesa_phi_r, wavelength)
        weights_rx = np.stack([x for xs in [[elem_weights[i, j].flatten() for j in range(elem_weights.shape[1])] for i in range(elem_weights.shape[0])] for x in xs], axis=0)
        txposes = np.stack([rp.txpos(ptimes)[..., :3].swapaxes(0, 1) for rp in rp_tx], axis=0)
        rxposes = np.stack([rp.rxpos(ptimes)[..., :3].swapaxes(0, 1) for rp in rp_rx], axis=0)
        block_data = trace_cpi(tracer, chirps, npulses, txposes, rxposes, weights_tx, weights_rx,
                               rp_tx[0].tx.boresight(ptimes[0]), aesa_bore,
                               ptimes, nsam, fc, fs, near_range_s, ranges[-1], bw_az / 2, bw_el / 2,
                               cfig.tracer_params.pix_width, cfig.tracer_params.pix_height,
                               cfig.ant_params.transmit_power, cfig.ant_params.rx_gain, cfig.ant_params.tx_gain,
                               cfig.ant_params.rec_gain, cfig.ant_params.noise_figure,
                               cfig.ant_params.operating_temperature, fft_len, add_chirp=False, add_noise=True)
        # block_data = np.sum(block_data, axis=0)
        block_data = np.concatenate([np.fft.fft(b, fft_len, axis=-1) * ff[None, None, None, :]
                                     for b, ff in zip(block_data, fft_chirps)], axis=0)
        block_data = np.sum(block_data, axis=0)

        # rp_data.append(upsamplePulse(block_data * mf_chirp, fft_len, cfig.tracer_params.upsample, is_freq=True,
        #                         time_len=nsam).astype(_complex_float))

        # Matched Filter
        rp_data = np.concatenate((np.fft.ifft(block_data * mf_chirps[0], fft_len)[..., :nsam].astype(_complex_float),
                                  np.fft.ifft(block_data * mf_chirps[1], fft_len)[..., :nsam].astype(_complex_float)), axis=0)
        base_pos = rp_tx[0].pos(ptimes)
        dopp_correction = getDopplerLine(rp_tx[0].az_iner(ptimes).mean(), ranges,
                                         rp_tx[0].vel(ptimes).mean(axis=0),
                                         base_pos.mean(axis=0),
                                         rp_tx[0].el_iner(ptimes).mean() + bw_el / 2,
                                         bw_az / 2, prf, c0 / fc)
        dp_f, dp_times = np.meshgrid(dopp_correction[0], ptimes)
        dopp = np.exp(-1j * 2 * np.pi * dp_f * dp_times)
        dopp_corr_data = np.fft.fftshift(np.fft.fft(rp_data * dopp * doppwin[:, None], axis=1), axes=1)
        sum_data = dopp_corr_data.sum(axis=0)
        el_del_data = (dopp_corr_data[0] + dopp_corr_data[1]) - (dopp_corr_data[2] + dopp_corr_data[3])
        az_del_data = (dopp_corr_data[0] + dopp_corr_data[2]) - (dopp_corr_data[1] + dopp_corr_data[3])
        det_sum = abs(sum_data)
        # det_sum[32 - clutter_window.shape[0] // 2:32 + clutter_window.shape[0] // 2] *= clutter_window[:, None]

        # Detect a target
        poss_dect = fftconvolve(det_sum, scan_kernel, mode='same') + det_sum.std() * cfig.radar_params.cfar_pfa
        poss_targ = np.where(det_sum - poss_dect > 0)
        blobs, num_blobs = label(det_sum - poss_dect > 0)

        if num_blobs > 0:
            for b in range(num_blobs):
                blob_x, blob_y = np.where(blobs == b + 1)
                blob_i = int(np.mean(blob_y))
                blob_ext = blob_y.max() - blob_y.min() + 2
                blob_idx = np.arange(blob_i - blob_ext, blob_i + blob_ext)
                centered_data = rp_data[:, :, blob_idx].swapaxes(0, 1) - rp_data[:, :, blob_idx].swapaxes(0, 1).mean(
                    axis=2, keepdims=True)
                Rhat = np.einsum('aij,akj->aik', centered_data, centered_data) / (centered_data.shape[2] - 1)
                evs, Uh = np.linalg.eig(Rhat)

                # Outer product of the eigenvectors
                Re = np.einsum('ali,ajk->aik', Uh[:, n_eigs:, :], Uh[:, n_eigs:, :].conj())
                music_ = abs(1. / np.einsum('bj,ajk,bk->ab', fine_ucavec.conj(), Re, fine_ucavec))
                music_ /= np.linalg.norm(music_, axis=1)[:, None]

                best_pts = np.argmax(music_, axis=1)

                dirs = azelToVec(fine_az_mesh[best_pts] + rp_tx[0].tx.az_aesa_iner(ptimes).mean(),
                                 fine_el_mesh[best_pts] + rp_tx[0].tx.el_aesa_iner(ptimes).mean()).T
                poss_pts = rp_tx[0].pos(ptimes).mean(axis=0) + dirs.mean(axis=0) * ranges[blob_idx].mean()
                pcloud0.append(poss_pts)
                apr = az_del_data / sum_data
                epr = el_del_data / sum_data
                target_angles = np.array([[rp_tx[0].tx.az_aesa_iner(ptimes).mean(),
                                           rp_tx[0].tx.el_aesa_iner(ptimes).mean()] for _ in blob_x])
                target_angles[:, 0] += np.arcsin(
                    (c0 / fc) / (np.pi * np.linalg.norm(rx_array[0] - rx_array[1])) * np.arctan(apr.imag[blob_x, blob_y]))
                target_angles[:, 1] += np.arcsin(
                    (c0 / fc) / (np.pi * np.linalg.norm(rx_array[0] - rx_array[2])) * np.arctan(epr.imag[blob_x, blob_y]))
                target_ranges = ranges[blob_y]
                final_angle = np.sum(target_angles * det_sum[blob_x, blob_y][:, None], axis=0) / sum(det_sum[blob_x, blob_y])
                pcloud1.append(rp_tx[0].pos(ptimes).mean(axis=0) + target_ranges.mean() * azelToVec(*final_angle).T)

        aesa_az = np.arctan2(aesa_bore[0, 0], aesa_bore[0, 1])
        aesa_el = -np.arcsin(aesa_bore[0, 2])
        aesa_near_range = ranges[0] * np.cos(aesa_el)
        aesa_far_range = ranges[-1] * np.cos(aesa_el)

        neg_beamvec = np.array([np.sin(aesa_az - bw_az / 2),
                                np.cos(aesa_az - bw_az / 2)])
        pos_beamvec = np.array([np.sin(aesa_az + bw_az / 2),
                                np.cos(aesa_az + bw_az / 2)])
        beamx = np.array(
            [neg_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_far_range, pos_beamvec[0] * aesa_far_range,
             pos_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_near_range]) + base_pos[0, 0]
        beamy = np.array(
            [neg_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_far_range, pos_beamvec[1] * aesa_far_range,
             pos_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_near_range]) + base_pos[0, 1]

        neg_beamvec = np.array([np.sin(rp_tx[0].az_iner(ptimes).mean() - bw_az / 2),
                                np.cos(rp_tx[0].az_iner(ptimes).mean() - bw_az / 2)])
        pos_beamvec = np.array([np.sin(rp_tx[0].az_iner(ptimes).mean() + bw_az / 2),
                                np.cos(rp_tx[0].az_iner(ptimes).mean() + bw_az / 2)])
        aesa_near_range = ranges[0] * np.cos(rp_tx[0].el_iner(ptimes).mean())
        aesa_far_range = ranges[-1] * np.cos(rp_tx[0].el_iner(ptimes).mean())
        gimx = np.array(
            [neg_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_far_range, pos_beamvec[0] * aesa_far_range,
             pos_beamvec[0] * aesa_near_range, neg_beamvec[0] * aesa_near_range]) + base_pos[0, 0]
        gimy = np.array(
            [neg_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_far_range, pos_beamvec[1] * aesa_far_range,
             pos_beamvec[1] * aesa_near_range, neg_beamvec[1] * aesa_near_range]) + base_pos[0, 1]


        sumax.cla()
        elax.cla()
        azax.cla()
        scatax.cla()
        sumax.set_title('Sum Beam')
        sumax.imshow(db(sum_data).T, origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]), aspect='auto')
        sumax.set_ylabel('Range (m)')
        azax.set_title('Az. Del Beam')
        azim = azax.imshow(db(rp_data[0]).T, origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]),
                     aspect='auto')
        fig.colorbar(azim, cax=cbax)
        ll, bb, ww, hh = cbax.get_position().bounds
        cbax.set_position([ll, bb, ww*.25, hh])
        elax.set_title('El. Del Beam')
        elax.imshow(db(db(rp_data[1])).T, origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]),
                     aspect='auto')
        elax.set_xlabel('Doppler (Hz)')
        if len(poss_targ[0]) > 0:
            sumax.scatter(dopp_freq[poss_targ[0]], ranges[poss_targ[1]], c='orange')
        if abs(rp_tx[0].az_iner(ptimes).mean() - np.arctan2(base_pos[0, 0], base_pos[0, 1])) < bw_az:
            sumax.scatter([0], [np.linalg.norm(rp_tx[0].txpos(ptimes).mean(axis=0))], c='red')
        scatax.scatter(base_pos[:, 0], base_pos[:, 1], c='blue')
        for h in tracer.hulls:
            scatax.scatter(h.vertices[:, 0], h.vertices[:, 1], c='red')
        scatax.plot(beamx, beamy, c='blue')
        scatax.plot(gimx, gimy, c='red')
        plt.draw()
        plt.pause(.01)

    '''ani = anim.ArtistAnimation(fig, ims, interval=150, blit=True)
    test = anim.FFMpegWriter(fps=5)
    ani.save('data.gif', writer=test)'''

    # plt.figure('Minimum Range Distance')
    # plt.imshow(range_grid)

    plt.figure('Chirps')
    plt.plot(db(fft_chirps[0]))
    plt.plot(db(mf_chirps[0]))

    plt.figure('Autocorrelation')
    plt.plot(np.fft.fftshift(db(np.fft.ifft(fft_chirps[0] * mf_chirps[0]))))
    plt.plot(np.fft.fftshift(db(np.fft.ifft(fft_chirps[1] * mf_chirps[0]))))
    plt.plot(np.fft.fftshift(db(np.fft.ifft(fft_chirps[0] * mf_chirps[1]))))

    points0 = np.stack(pcloud0)
    points1 = np.stack(pcloud1)

    cloud_fig = px.scatter_3d(x=points0[:, 0], y=points0[:, 1], z=points0[:, 2])
    cloud_fig.add_scatter3d(x=points1[:, 0], y=points1[:, 1], z=points1[:, 2], mode='markers')
    for h in tracer.hulls:
        cloud_fig.add_mesh3d(x=h.vertices[:, 0],
                       y=h.vertices[:, 1],
                       z=h.vertices[:, 2],
                       # i, j and k give the vertices of triangles
                       i=h.faces[:, 0],
                       j=h.faces[:, 1],
                       k=h.faces[:, 2])
    cloud_fig.show()

    # rx_array = np.stack([rp_tx[0].txpos(rp_tx[0].gpst[0]) for rp in rps])
    fig = px.scatter_3d(x=rx_array[:, 0], y=rx_array[:, 1], z=rx_array[:, 2])
    fig.add_scatter3d(x=tx_array[:, 0], y=tx_array[:, 1], z=tx_array[:, 2], mode='markers')
    fig.add_scatter3d(x=vx_array[:, 0], y=vx_array[:, 1], z=vx_array[:, 2], mode='markers')
    fig.show()





