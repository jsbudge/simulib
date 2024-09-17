import numpy as np
from SDRParsing import load, loadASIFile, loadASHFile
from cuda_kernels import getMaxThreads, cpudiff, applyRadiationPatternCPU
from grid_helper import SDREnvironment, mesh
from platform_helper import SDRPlatform, RadarPlatform
from scipy.optimize import minimize, basinhopping
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from simulation_functions import azelToVec
from tqdm import tqdm
import plotly.io as pio
pio.renderers.default = 'browser'

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180

fnme = '/home/jeff/SDR_DATA/ARCHIVE/07082024/SAR_07082024_112333.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / 9.6e9
ant_gain = 25
transmit_power = 100
pixel_to_m = .25
upsample = 4

ref_pt = [50, 40]

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(2., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)

gx, gy, gz = bg.getGrid()
asi_data = bg.getRefGrid()
pt_pos = np.array([gx[*ref_pt], gy[*ref_pt], gz[*ref_pt]])

rngs = np.zeros(sdr[0].nframes)
pvecs = np.zeros((sdr[0].nframes, 3))
pmods = np.ones(sdr[0].nframes)
rhos = np.zeros(sdr[0].nframes)
norm = np.array([0., 0.])
scat = 1.
nper = 256

access_pts = []
for t in tqdm(sdr[0].pulse_time):
    vecs = np.array([gx - rp.txpos(t)[0], gy - rp.txpos(t)[1],
                     gz - rp.txpos(t)[2]])
    bins = np.round((np.linalg.norm(vecs, axis=0) / c0 - 2 * near_range_s) * fs * upsample).astype(int)
    access_pts.append([(a, b) for a, b in zip(*np.where(bins == bins[*ref_pt]))])
all_pts = np.array(list(set([x for xs in access_pts for x in xs])))


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

valids = np.logical_and(rhos > 0, pmods > 1e-3)
pvecs = pvecs[valids]
rngs = rngs[valids]
rhos = medfilt(rhos[valids], 15)
pmods = pmods[valids]

coeff = 1 / rngs**4
coeff = coeff / coeff.max()
rhos_scaled = rhos / rhos.max() * coeff.max()

def minfunc(x):
    mnorm = azelToVec(x[1], x[2])
    xr = np.sum(pvecs * (pvecs - 2 * np.outer(np.sum(pvecs * mnorm, axis=1), mnorm)), axis=1)
    xr[xr < 0] = 0
    x_hat = coeff * xr / x[0]**2 * np.exp(-xr**2 / (2 * x[0]**2))
    return np.linalg.norm(rhos_scaled - x_hat)

opt_x = basinhopping(minfunc, np.array([4, *norm]))

opt_norm = azelToVec(opt_x['x'][1], opt_x['x'][2])
opt_scat = opt_x['x'][0]

check = coeff * np.sinc(opt_scat / np.pi * np.arccos(np.sum(pvecs * (pvecs - 2 * np.outer(np.sum(pvecs * opt_norm, axis=1), opt_norm)), axis=1)))**2

plt.figure()
plt.plot(rhos_scaled)
plt.plot(check)
plt.show()