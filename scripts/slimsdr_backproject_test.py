#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np  # Packing of structures in C-compatible format
from scipy.signal import sawtooth
from scipy.signal.windows import taylor
from sdrparse.SARParsing import SARParse
from simulib.simulib.simulation_functions import db, genChirp, genTaylorWindow, azelToVec, upsamplePulse, llh2enu, enu2llh, getElevationMap
from simulib.simulib.platform_helper import SDRPlatform, SARPlatform
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
    if True:
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

            bpj_grid += backprojectPulseStream([rp_data], [rp._tx.az_aesa_iner(ptimes)],
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

    if False:
        npulses = 512
        upsample = 8
        origin = np.array([64.543218, -147.799325, 140])

        # Generate a platform
        print('Generating platform...', end='')
        sar = SARParse('/home/jeff/SDR_DATA/RAW/11092025/SAR_11072025_111357.sar')
        rp = SARPlatform(sar, origin=origin)
        # bg, rp = getRadarAndEnvironment(sar, is_sdr = False)

        wavelength = c0 / sar[0].fc
        fs = sar[0].fs

        nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = rp.getRadarParams(0, 0., upsample)

        mfilt = sar.genMatchedFilter(0)  # np.fft.fft(sdr[0].cal_chirp, fft_len).conj()
        npts = 1600

        gx, gy = np.meshgrid(np.linspace(-400, 400, npts), np.linspace(-400, 400, npts))
        bpj_grid = np.zeros(gx.shape, dtype=_complex_float)
        bpj_rfi_grid = bpj_grid + 0.
        try:
            lats, lons, alts = enu2llh(gx.flatten(), gy.flatten(), np.zeros_like(gx).flatten(), origin)
            gz = getElevationMap(lats, lons).reshape(gx.shape)
        except:
            gz = np.zeros(gx.shape) + origin[2]

        for frame in tqdm(list(zip(*(iter(sar[0].frame_num[::npulses]),)))):
            fnums = sar[0].frame_num[frame[0]:frame[0] + npulses]
            try:
                ptimes = sar[0].pulse_time[fnums]
            except:
                break

            block_data = np.fft.fft(sar.getPulses(fnums, 0).T, fft_len) * mfilt
            rfi_data = block_data + 0.0

            # Matched Filter
            rp_data = upsamplePulse(block_data, fft_len, upsample, is_freq=True,
                                    time_len=nsam).astype(_complex_float)

            tmp_grid = backprojectPulseStream([rp_data], [rp._tx.az_iner(ptimes)],
                                               [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                               [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                               _float(near_range_s), _float(fs * upsample), rp.az_half_bw,
                                               gx=gx, gy=gy)

            bpj_grid += tmp_grid

            # Check for RFI
            from scipy.stats import kurtosis
            rfi_check = abs(block_data[:, :fft_len // 2])
            kurt = kurtosis(rfi_check, axis=1)
            cfar = np.convolve(kurt, np.array([.25, .25, 0, 0, 0, .25, .25]), mode='same') + kurt.std() * 1.

            pulses_to_rfi = np.where(kurt > cfar)[0]

            if len(pulses_to_rfi) > 0:
                for idx, dt in zip(pulses_to_rfi, block_data[pulses_to_rfi, :]):
                    abs_dt = abs(dt)
                    dt_cfar = np.convolve(abs_dt, np.array([.25, .25, 0, 0, 0, .25, .25]), mode='same') + abs_dt.std() * 3.
                    block_data[idx, abs_dt > dt_cfar] = 0.

                rpi_data = upsamplePulse(rfi_data, fft_len, upsample, is_freq=True,
                                         time_len=nsam).astype(_complex_float)

                bpj_rfi_grid += backprojectPulseStream([rpi_data], [rp._tx.az_iner(ptimes)],
                                                       [rp.rxpos(ptimes).mean(axis=(1, 2))],
                                                       [rp.txpos(ptimes).mean(axis=(1, 2))], gz, _float(wavelength),
                                                       _float(near_range_s), _float(fs * upsample), rp.az_half_bw,
                                                       gx=gx, gy=gy)
            else:
                bpj_rfi_grid += tmp_grid





        plt.figure('Backprojection')
        plt.subplot(1, 2, 1)
        plt.title('RFI mitigated')
        scaled_bpj = abs(bpj_grid)
        scaled_bpj = (scaled_bpj - scaled_bpj.min()) * 30 / scaled_bpj.mean() + 1
        plt.imshow(scaled_bpj, extent=[gx.min(), gx.max(), gy.min(), gy.max()], origin='lower', cmap='gray')
        plt.clim([0, 255])
        # plt.clim([200, 230])
        plt.subplot(1, 2, 2)
        plt.title('RFI')
        scaled_bpj = abs(bpj_rfi_grid)
        scaled_bpj = (scaled_bpj - scaled_bpj.min()) * 30 / scaled_bpj.mean() + 1
        plt.imshow(scaled_bpj, extent=[gx.min(), gx.max(), gy.min(), gy.max()], origin='lower', cmap='gray')
        plt.clim([0, 255])
        # plt.clim([200, 230])






