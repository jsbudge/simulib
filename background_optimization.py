import numpy as np
from SDRParsing import load, loadASIFile, loadASHFile
from cuda_kernels import getMaxThreads, cpudiff, applyRadiationPatternCPU
from grid_helper import SDREnvironment, mesh
from platform_helper import SDRPlatform, RadarPlatform
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.io as pio
pio.renderers.default = 'browser'

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180

fnme = '/data6/SAR_DATA/2024/07082024/SAR_07082024_112333.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / 9.6e9
ant_gain = 25
transmit_power = 100
pixel_to_m = .25
upsample = 4

ref_pt = [543, 424]

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(10., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)

gx, gy, gz = bg.getGrid()
asi_data = bg.getRefGrid()
pt_pos = np.array([gx[*ref_pt], gy[*ref_pt], gz[*ref_pt]])

rngs = np.zeros(sdr[0].nframes)
pvecs = np.zeros((sdr[0].nframes, 3))
pmods = np.ones(sdr[0].nframes)
rhos = np.zeros(sdr[0].nframes)
norm = np.array([0., 0., 1.])
scat = 1.
nper = 256

for n in tqdm(range(0, sdr[0].nframes - 1, nper)):
    tn = np.arange(n, n + nper)
    try:
        vec = pt_pos - rp.txpos(sdr[0].pulse_time[tn])
        tmp_rngs = np.linalg.norm(vec, axis=1)
        rng_bin = np.round((tmp_rngs / c0 - 2 * near_range_s) * fs * upsample).astype(int)
        if np.any(rng_bin < 0) or np.any(rng_bin >= nsam * upsample):
            continue
        _, pdata = sdr.getPulses(sdr[0].frame_num[tn], 0)
        mfdata = np.fft.fft(pdata, fft_len, axis=0) * mfilt[:, None]
        updata = np.zeros((up_fft_len, len(tn)), dtype=np.complex128)
        updata[:fft_len // 2, :] = mfdata[:fft_len // 2, :]
        updata[-fft_len // 2:, :] = mfdata[-fft_len // 2:, :]
        updata = np.fft.ifft(updata, axis=0)[:nsam * upsample, :].T
        dmag = abs(updata[np.arange(nper), rng_bin])
        if np.all(dmag == 0):
            continue
        rngs[tn] = tmp_rngs
        pvecs[tn, :] = vec / tmp_rngs[:, None]
        rhos[tn] = dmag
        azes = np.arctan2(pvecs[tn, 0], pvecs[tn, 1])
        eles = -np.arcsin(pvecs[tn, 2])
        pmods[tn] = [applyRadiationPatternCPU(eles[i], azes[i], rp.pan(sdr[0].pulse_time[m]),
                                              rp.tilt(sdr[0].pulse_time[m]), rp.pan(sdr[0].pulse_time[m]),
                                              rp.tilt(sdr[0].pulse_time[m]), rp.az_half_bw, rp.el_half_bw)
                     for i, m in enumerate(tn)]
    except IndexError:
        continue

valids = np.logical_and(rhos > 0, pmods > 1e-2)
pvecs = pvecs[valids]
rngs = rngs[valids]
rhos = rhos[valids]
pmods = pmods[valids]

coeff = pmods / rngs**4
coeff = coeff / coeff.max()
rhos_scaled = rhos / rhos.max() * coeff.max()

def minfunc(x):
    mnorm = x[1:] / np.linalg.norm(x[1:])
    x_hat = coeff * np.sinc(x[0] / np.pi * np.arccos(np.sum(pvecs * (pvecs - 2 * np.outer(np.sum(pvecs * mnorm, axis=1), mnorm)), axis=1)))**2
    return np.linalg.norm(rhos_scaled - x_hat)

opt_x = minimize(minfunc, np.array([scat, *norm]), bounds=[(1e-9, np.inf), (-1, 1), (-1, 1), (0, 1)])

opt_norm = opt_x['x'][1:] / np.linalg.norm(opt_x['x'][1:])
opt_scat = opt_x['x'][0]

check = coeff * np.sinc(opt_scat / np.pi * np.arccos(np.sum(pvecs * (pvecs - 2 * np.outer(np.sum(pvecs * opt_norm, axis=1), opt_norm)), axis=1)))**2

plt.figure()
plt.plot(rhos_scaled)
plt.plot(check)
plt.show()