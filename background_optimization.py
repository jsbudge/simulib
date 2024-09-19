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
norm = np.array([0., 0.])
scat = 1.
nper = 16

access_pts = []
for t in tqdm(sdr[0].pulse_time):
    vecs = np.array([gx - rp.txpos(t)[0], gy - rp.txpos(t)[1],
                     gz - rp.txpos(t)[2]])
    bins = np.round((np.linalg.norm(vecs, axis=0) / c0 - 2 * near_range_s) * fs * upsample).astype(int)
    access_pts.append([(a, b) for a, b in zip(*np.where(bins == bins[*ref_pt]))])
all_pts = np.array(list(set([x for xs in access_pts for x in xs])))

rho_matrix = np.zeros((all_pts.shape[0], nper))
rngs = np.zeros_like(rho_matrix)
pmods = np.zeros_like(rho_matrix)
pvecs = np.zeros((*rho_matrix.shape, 3))

for n in tqdm(range(nper)):
    vec = np.array([gx[all_pts[:, 0], all_pts[:, 1]] - rp.txpos(sdr[0].pulse_time[n])[0],
                    gy[all_pts[:, 0], all_pts[:, 1]] - rp.txpos(sdr[0].pulse_time[n])[1],
                    gz[all_pts[:, 0], all_pts[:, 1]] - rp.txpos(sdr[0].pulse_time[n])[2]]).T
    tmp_rngs = np.linalg.norm(vec, axis=1)
    rng_bin = np.round((tmp_rngs / c0 - 2 * near_range_s) * fs * upsample).astype(int)
    _, pdata = sdr.getPulses([sdr[0].frame_num[n]], 0)
    mfdata = np.fft.fft(pdata, fft_len, axis=0) * mfilt[:, None]
    updata = np.zeros((up_fft_len, 1), dtype=np.complex128)
    updata[:fft_len // 2, :] = mfdata[:fft_len // 2, :]
    updata[-fft_len // 2:, :] = mfdata[-fft_len // 2:, :]
    updata = np.fft.ifft(updata, axis=0)[:nsam * upsample, :].T
    dmag = abs(updata[0, rng_bin])
    rho_matrix[:, n] = dmag
    rngs[:, n] = tmp_rngs
    pvecs[:, n, :] = vec / tmp_rngs[:, None]
    azes = np.arctan2(pvecs[:, n, 0], pvecs[:, n, 1])
    eles = -np.arcsin(pvecs[:, n, 2])
    pmods[:, n] = [applyRadiationPatternCPU(eles[i], azes[i], rp.pan(sdr[0].pulse_time[n]),
                                            rp.tilt(sdr[0].pulse_time[n]), rp.pan(sdr[0].pulse_time[n]),
                                            rp.tilt(sdr[0].pulse_time[n]), rp.az_half_bw, rp.el_half_bw)
                   for i in range(rho_matrix.shape[0])]

coeff = 1 / rngs ** 4
coeff = coeff / coeff.max()
rhos_scaled = rho_matrix / rho_matrix.max() * coeff.max()

x0 = np.ones(rho_matrix.shape[0] + 2 * rho_matrix.shape[0])


def minfunc(x):
    mnorm = azelToVec(x[1::3], x[2::3]).T
    xr = np.sum(pvecs * (pvecs - 2 * np.einsum('ji,jk->jik',
                                               np.sum(pvecs * mnorm[:, None, :], axis=2), mnorm)), axis=2)
    xr[xr < 0] = 0
    x_hat = coeff * xr / x[0::3][:, None] ** 2 * np.exp(-xr ** 2 / (2 * x[0::3][:, None] ** 2))
    return np.linalg.norm(rhos_scaled - x_hat)


opt_x = minimize(minfunc, x0)

opt_norm = azelToVec(opt_x['x'][1], opt_x['x'][2])
opt_scat = opt_x['x'][0]

xr = np.sum(pvecs * (pvecs - 2 * np.outer(np.sum(pvecs * opt_norm, axis=1), opt_norm)), axis=1)
xr[xr < 0] = 0
check = coeff * xr / opt_scat ** 2 * np.exp(-xr ** 2 / (2 * opt_scat ** 2))

plt.figure()
plt.plot(rhos_scaled)
plt.plot(check)
plt.show()
