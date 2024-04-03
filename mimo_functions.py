import numpy as np
from simulib.platform_helper import RadarPlatform
import cupy as cupy
from scipy.signal.windows import taylor

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


def genChannels(n_tx, n_rx, tx_pos, rx_pos, plat_e, plat_n, plat_u, plat_r, plat_p, plat_y,
                gpst, gimbal, goff, grot, dep_ang, az_half_bw, el_half_bw, fs):
    rps = []
    rx_array = []
    vx_perm = [(n, q) for q in range(n_tx) for n in range(n_rx)]
    for tx, rx in vx_perm:
        txpos = np.array(tx_pos[tx])
        rxpos = np.array(rx_pos[rx])
        vx_pos = rxpos + txpos
        if not np.any([sum(vx_pos - r) == 0 for r in rx_array]):
            rps.append(
                RadarPlatform(plat_e, plat_n, plat_u, plat_r, plat_p, plat_y, gpst, txpos, rxpos, gimbal, goff,
                              grot, dep_ang, 0., az_half_bw * 2 / DTR, el_half_bw * 2 / DTR,
                              fs, tx_num=tx, rx_num=rx))
            rx_array.append(rxpos + txpos)
    rpref = RadarPlatform(plat_e, plat_n, plat_u, plat_r, plat_p, plat_y, gpst, np.array([0., 0., 0.]),
                          np.array([0., 0., 0.]), gimbal, goff, grot, dep_ang, 0.,
                          az_half_bw * 2 / DTR, el_half_bw * 2 / DTR, fs)
    return rpref, rps, np.array(rx_array)


def genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, fft_len, cpi_len):
    # Get Taylor window of appropriate length and shift it to the aliased frequency of fc
    taywin = int(bwidth / fs * fft_len)
    taywin = taywin + 1 if taywin % 2 != 0 else taywin
    taytay = np.zeros(fft_len, dtype=np.complex128)
    twin_tmp = taylor(taywin, nbar=10, sll=60)
    taytay[:taywin // 2] = twin_tmp[taywin // 2:]
    taytay[-taywin // 2:] = twin_tmp[:taywin // 2]
    alias_shift = int(fft_len + (fc % (fs / 2) - fs / 2) / fs * fft_len)
    taytay = np.roll(taytay, alias_shift).astype(np.complex128)

    # Chirps and Mfilts for each channel
    chirps = []
    mfilt = []
    for rp in rps:
        mf = waves[rp.tx_num].conj() * taytay
        chirps.append(cupy.array(waves[rp.tx_num], dtype=np.complex128))
        mfilt.append(cupy.array(mf, dtype=np.complex128))
    return taytay, chirps, mfilt


def applyRangeRolloff(bpj_truedata):
    mag_data = np.sqrt(abs(bpj_truedata))
    brightness_raw = np.median(np.sqrt(abs(bpj_truedata)), axis=1)
    brightness_curve = np.polyval(np.polyfit(np.arange(bpj_truedata.shape[0]), brightness_raw, 4),
                                  np.arange(bpj_truedata.shape[1]))
    brightness_curve /= brightness_curve.max()
    brightness_curve = 1. / brightness_curve
    mag_data *= np.outer(np.ones(mag_data.shape[0]), brightness_curve)
    return mag_data