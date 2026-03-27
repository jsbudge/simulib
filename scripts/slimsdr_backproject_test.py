#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from scipy.signal import sawtooth, stft, istft
from scipy.signal.windows import taylor
from sdrparse.SARParsing import SARParse
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec, upsamplePulse, llh2enu, enu2llh, getElevationTIFF
from simulib.simulib.platform_helper import SDRPlatform, SARPlatform
from simulib.simulib.grid_helper import SAREnvironment
from sdrparse.SDRV2Parsing import load
from simulib.simulib.utils import c0, DTR, _float, _complex_float, getRadarAndEnvironment
from backproject_utils import backprojectPulseStream
import matplotlib.pyplot as plt
import matplotlib as mplib
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

GPS_UPDATE_HZ = 100

def aliasFrequency(f, a_fs):
    return f - int(f / (a_fs / 2)) * a_fs / 2


def calcDopplerPRF(v, fc, bandwidth, half_az_bw):
    return 4 * v * np.sin(half_az_bw) * (fc + bandwidth / 2) / c0


if __name__ == "__main__":
    npulses = 128
    upsample = 4
    if False:
        origin = np.array([40.116744, -111.626010, 1420])

        # Generate a platform
        print('Generating platform...', end='')
        sdr = load('/home/jeff/SDR_DATA/RAW/11112025/SAR_11112025_145023.sar', progress_tracker=True)
        bg, rp = getRadarAndEnvironment(sdr)

        wavelength = c0 / sdr[0].fc
        fs = sdr[0].fs

        nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(0., 0., upsample)

        mfilt = sdr.genReciprocalRipple(0, 0, 0, fft_len=fft_len)  # np.fft.fft(sdr[0].cal_chirp, fft_len).conj()
        nbpj_pts = [400, 400]

        gx, gy, gz = bg.getGrid(origin, 200, 200, nrows=nbpj_pts[0], ncols=nbpj_pts[1])
        bpj_grid = np.zeros(gx.shape, dtype=_complex_float)

        for frame in tqdm(list(zip(*(iter(sdr[0].frames[::npulses]),)))):
            fnums = sdr[0].frames[frame[0]:frame[0] + npulses]
            ptimes = sdr[0].pulse_time[fnums]

            block_data = np.fft.fft(sdr.getPulses(fnums, 0)[1][:, 0, :], fft_len)

            # Matched Filter
            rp_data = upsamplePulse(block_data * mfilt, fft_len, upsample, is_freq=True, time_len=nsam).astype(_complex_float)
            # rp_data = np.fft.ifft(block_data * mf_chirps, fft_len, axis=-1)[..., :nsam].astype(_complex_float)
            base_pos = rp.pos(ptimes)

            bpj_grid += backprojectPulseStream([rp_data], [rp.tx.az_aesa_iner(ptimes)],
                                               [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                               [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                               _float(near_range_s), _float(fs * upsample), rp.az_half_bw, gx=gx, gy=gy)

        scaled_bpj = abs(bpj_grid)
        scaled_bpj = (scaled_bpj - scaled_bpj.min()) * 30 / scaled_bpj.mean() + 1
        plt.figure('SDRV2 Backprojection')
        plt.imshow(scaled_bpj, extent=[gx.min(), gx.max(), gy.min(), gy.max()], origin='lower', cmap='gray')
        plt.clim([0, 255])

    '''
    -----------------------SLIMSAR---------------------------------
    '''

    if True:
        npulses = 32
        upsample = 4
        # test_name = '/data1/SAR_DATA/2018/10052018/SAR_10052018_133856.sar'
        test_name = '/home/jeff/SDR_DATA/RAW/02272025/SAR_02272025_132339.sar'
        # origin = np.array([ 40.138044, -111.660027, 1380.])
        origin = np.array([64.676301, -148.285144, 119])

        # Generate a platform
        print('Generating platform...', end='')
        sar = SARParse(test_name)
        bg = SAREnvironment(sar, origin=origin, local_height=119)
        # bg = SAREnvironment(sar, local_grid=np.zeros((400, 400)), origin=origin)
        rp = SARPlatform(sar, origin=bg.ref, gimbal_offset=np.array([.1960, .1257, .0059]), gimbal_rotations=np.array([0, 0, 0]))
        # bg, rp = getRadarAndEnvironment(sar, is_sdr = False)

        wavelength = c0 / sar[0].fc
        fs = sar[0].fs
        mean_az = 2.7227

        nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(0, 0., upsample)

        mfilt = sar.genMatchedFilter(0)  # np.fft.fft(sdr[0].cal_chirp, fft_len).conj()
        bounds = np.where(abs(mfilt != 0))[0]
        lower_bound = bounds.min()
        upper_bound = bounds.max() + 1

        gx, gy, gz = bg.getGrid(origin, 1600, 800, 3200, 1600, rp.heading(rp.gpst).mean() - np.pi / 2)
        # gx, gy, gz = bg.getGrid(origin, 200, 200, 400, 400, rp.heading(rp.gpst).mean() - np.pi / 2)
        bpj_grid = np.zeros(gx.shape, dtype=_complex_float)
        bpj_rfi_grid = np.zeros(gx.shape, dtype=_complex_float)
        kurtosi = []
        test_kurtosi = []

        for frame in tqdm(range(0, sar[0].nframes, npulses)):
            fnums = np.arange(frame, min(frame + npulses, sar[0].nframes))
            ptimes = sar[0].pulse_time[fnums]

            block_data = np.fft.fft(sar.getPulses(fnums, 0).T, fft_len)# * mfilt
            rfi_data = block_data * mfilt

            # Matched Filter
            rp_data = upsamplePulse(rfi_data, fft_len, upsample, is_freq=True,
                                    time_len=nsam).astype(_complex_float)

            tmp_grid = backprojectPulseStream([rp_data], [mean_az * np.ones(len(fnums))],
                                              [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                              [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                              _float(near_range_s), _float(fs * upsample), rp.az_half_bw,
                                              gx=gx, gy=gy)

            bpj_grid += tmp_grid

            # Check for RFI
            rfi_check = abs(block_data[:, lower_bound:upper_bound])
            mu = np.mean(rfi_check / 1e13, axis=1)
            kurt = rfi_check.shape[1] * np.sum((rfi_check / 1e13 - mu[:, None])**4, axis=1) / np.sum((rfi_check / 1e13 - mu[:, None])**2, axis=1)**2 - 3
            kurtosi.append(kurt)
            # cfar = np.convolve(kurt, np.array([.25, .25, 0, 0, 0, .25, .25]), mode='same') + kurt.std() * .5

            pulses_to_rfi = np.where(kurt > 6.)[0]

            if len(pulses_to_rfi) > 0:
                for idx, dt in zip(pulses_to_rfi, rfi_data[pulses_to_rfi, :]):
                    '''---ZERO-BIN---'''
                    # abs_dt = abs(dt)
                    # dt_cfar = np.convolve(abs_dt, np.array([.25, .25, 0, 0, 0, .25, .25]), mode='same') + abs_dt.std() * 2.
                    # block_data[idx, abs_dt > dt_cfar] = 0.
                    '''---ZERO-PULSE---'''
                    rfi_data[idx, :] = 0.
                    '''---SVD---'''
                    # stdt = stft(dt, return_onesided=False)[2]
                    # U, s, Vh = np.linalg.svd(stdt, full_matrices=False)
                    # s[:2] = 0.
                    # block_data[idx] = istft(U.dot(np.diag(s)).dot(Vh), nperseg=256)[1]

                rpi_data = upsamplePulse(rfi_data, fft_len, upsample, is_freq=True,
                                         time_len=nsam).astype(_complex_float)

                bpj_rfi_grid += backprojectPulseStream([rpi_data], [rp.tx.az_iner(ptimes)],
                                                       [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                                       [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                                       _float(near_range_s), _float(fs * upsample), rp.az_half_bw,
                                                       gx=gx, gy=gy)
            else:
                bpj_rfi_grid += tmp_grid

        import matplotlib.transforms as mtransforms



        fig, (ax0, ax1) = plt.subplots(1, 2)
        scaled_bpj = abs(bpj_grid)
        scaled_rfi_bpj = (scaled_bpj - scaled_bpj.min()) * 30 / scaled_bpj.mean() + 1
        im = ax0.imshow(scaled_rfi_bpj, origin='lower', cmap='gray', interpolation='none',
                        clip_on=True)
        trans_data = mtransforms.Affine2D().translate(-gx.shape[1] // 2, -gx.shape[0] // 2).scale(bg.cps, bg.rps).rotate(-rp.heading(rp.gpst).mean() + np.pi / 2).translate(gx.mean(), gy.mean()) + ax0.transData
        im.set_transform(trans_data)
        im.set_clim(0, 255)
        ax0.plot(rp.txpos(rp.gpst)[::10, 0, 0, 0], rp.txpos(rp.gpst)[::10, 0, 0, 1])
        # ax0.scatter(gx.flatten(), gy.flatten())
        # plt.clim([0, 255])
        scaled_bpj = abs(bpj_rfi_grid)
        scaled_bpj = (scaled_bpj - scaled_bpj.min()) * 30 / scaled_bpj.mean() + 1
        im1 = ax1.imshow(scaled_bpj, origin='lower', cmap='gray', interpolation='none',
                        clip_on=True)
        im1.set_clim(0, 255)
        trans_data = mtransforms.Affine2D().translate(-gx.shape[1] // 2, -gx.shape[0] // 2).scale(bg.cps, bg.rps).rotate(-rp.heading(rp.gpst).mean() + np.pi / 2).translate(gx.mean(), gy.mean()) + ax1.transData
        im1.set_transform(trans_data)
        ax1.plot(rp.txpos(rp.gpst)[::10, 0, 0, 0], rp.txpos(rp.gpst)[::10, 0, 0, 1])
        # plt.clim([200, 230])

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('Unchanged')
        plt.imshow(scaled_rfi_bpj.T, cmap='gray')
        plt.axis('off')
        plt.clim([0, 100])
        plt.subplot(2, 1, 2)
        plt.title('Mitigated')
        plt.imshow(scaled_bpj.T, cmap='gray')
        plt.axis('off')
        plt.clim([0, 100])

        plt.figure()
        plt.plot(rp.tx.az_iner(rp.gpst))

        plt.figure('Kurtoses')
        plt.plot(np.concatenate(kurtosi))
        plt.plot(np.concatenate(test_kurtosi))
        plt.legend(['Scipy', 'MyPy'])

