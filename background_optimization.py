import cupy
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from triangle.plot import vertices
from scipy.spatial import Delaunay
from SDRParsing import load
from cuda_kernels import calcRangeProfile, getMaxThreads, calcRangeProfileScattering
from grid_helper import SDREnvironment
from models import ImageSegmenter
from platform_helper import SDRPlatform
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.measure import label, find_contours
from skimage.morphology import medial_axis
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import open3d as o3d
from simulation_functions import db, llh2enu, getElevationMap, azelToVec, enu2llh
from pywt import wavedec2
from trimesh.creation import extrude_polygon
import plotly.io as pio
pio.renderers.default = 'browser'


def solvey(x1, y1, z1, x0, y0, z0, R1, R0, z):
    y = (4 * y1 ** 3 - 4 * y0 * y1 ** 2 + 4 * x1 ** 2 * y1 + 4 * z1 ** 2 * y1 + 4 * x0 ** 2 * y1 - 4 * y0 ** 2 * y1 - 4 * z0 ** 2 * y1 -
            4 * R1 ** 2 * y1 + 4 * R0 ** 2 * y1 - 8 * x1 * x0 * y1 - 8 * z1 * z * y1 + 8 * z0 * z * y1 + 4 * y0 ** 3 + 4 * y0 * z0 ** 2 +
            4 * y0 * R1 ** 2 - 4 * y0 * R0 ** 2 + 4 * x1 ** 2 * y0 - 4 * z1 ** 2 * y0 + 4 * x0 ** 2 * y0 - 8 * x1 * x0 * y0 + 8 * z1 * y0 * z -
            8 * y0 * z0 * z - np.sqrt((-4 * y1 ** 3 + 4 * y0 * y1 ** 2 - 4 * x1 ** 2 * y1 - 4 * z1 ** 2 * y1 - 4 * x0 ** 2 * y1 +
                                       4 * y0 ** 2 * y1 + 4 * z0 ** 2 * y1 + 4 * R1 ** 2 * y1 - 4 * R0 ** 2 * y1 + 8 * x1 * x0 * y1 + 8 * z1 * z * y1 -
                                       8 * z0 * z * y1 - 4 * y0 ** 3 - 4 * y0 * z0 ** 2 - 4 * y0 * R1 ** 2 + 4 * y0 * R0 ** 2 - 4 * x1 ** 2 * y0 + 4 * z1 ** 2 * y0 -
                                       4 * x0 ** 2 * y0 + 8 * x1 * x0 * y0 - 8 * z1 * y0 * z + 8 * y0 * z0 * z) ** 2 -
                                      4 * (4 * x1 ** 2 - 8 * x0 * x1 + 4 * y1 ** 2 + 4 * x0 ** 2 + 4 * y0 ** 2 - 8 * y1 * y0) *
                                      (x1 ** 4 - 4 * x0 * x1 ** 3 + 2 * y1 ** 2 * x1 ** 2 + 2 * z1 ** 2 * x1 ** 2 + 6 * x0 ** 2 * x1 ** 2 +
                                       2 * y0 ** 2 * x1 ** 2 + 2 * z0 ** 2 * x1 ** 2 - 2 * R1 ** 2 * x1 ** 2 - 2 * R0 ** 2 * x1 ** 2 + 4 * z ** 2 * x1 ** 2 -
                                       4 * z1 * z * x1 ** 2 - 4 * z0 * z * x1 ** 2 - 4 * x0 ** 3 * x1 - 4 * x0 * y0 ** 2 * x1 - 4 * x0 * z0 ** 2 * x1 +
                                       4 * x0 * R1 ** 2 * x1 + 4 * x0 * R0 ** 2 * x1 - 8 * x0 * z ** 2 * x1 - 4 * y1 ** 2 * x0 * x1 - 4 * z1 ** 2 * x0 * x1 +
                                       8 * z1 * x0 * z * x1 + 8 * x0 * z0 * z * x1 + y1 ** 4 + z1 ** 4 + x0 ** 4 + y0 ** 4 + z0 ** 4 + R1 ** 4 + R0 ** 4 +
                                       2 * y1 ** 2 * z1 ** 2 + 2 * y1 ** 2 * x0 ** 2 + 2 * z1 ** 2 * x0 ** 2 - 2 * y1 ** 2 * y0 ** 2 - 2 * z1 ** 2 * y0 ** 2 +
                                       2 * x0 ** 2 * y0 ** 2 - 2 * y1 ** 2 * z0 ** 2 - 2 * z1 ** 2 * z0 ** 2 + 2 * x0 ** 2 * z0 ** 2 + 2 * y0 ** 2 * z0 ** 2 -
                                       2 * y1 ** 2 * R1 ** 2 - 2 * z1 ** 2 * R1 ** 2 - 2 * x0 ** 2 * R1 ** 2 + 2 * y0 ** 2 * R1 ** 2 + 2 * z0 ** 2 * R1 ** 2 +
                                       2 * y1 ** 2 * R0 ** 2 + 2 * z1 ** 2 * R0 ** 2 - 2 * x0 ** 2 * R0 ** 2 - 2 * y0 ** 2 * R0 ** 2 - 2 * z0 ** 2 * R0 ** 2 -
                                       2 * R1 ** 2 * R0 ** 2 + 4 * z1 ** 2 * z ** 2 + 4 * x0 ** 2 * z ** 2 + 4 * z0 ** 2 * z ** 2 - 8 * z1 * z0 * z ** 2 -
                                       4 * z1 ** 3 * z - 4 * z0 ** 3 * z - 4 * z1 * x0 ** 2 * z + 4 * z1 * y0 ** 2 * z + 4 * z1 * z0 ** 2 * z +
                                       4 * z1 * R1 ** 2 * z - 4 * z0 * R1 ** 2 * z - 4 * z1 * R0 ** 2 * z + 4 * z0 * R0 ** 2 * z - 4 * y1 ** 2 * z1 * z +
                                       4 * y1 ** 2 * z0 * z + 4 * z1 ** 2 * z0 * z - 4 * x0 ** 2 * z0 * z - 4 * y0 ** 2 * z0 * z))) / (
        2 * (4 * x1 ** 2 - 8 * x0 * x1 + 4 * y1 ** 2 + 4 * x0 ** 2 + 4 * y0 ** 2 - 8 * y1 * y0))
    x = x1 + np.sqrt(-y1**2 + 2 * y1 * y - z1**2 + 2 * z1 * z + R0**2 - y**2 -z**2)
    return np.array([x, y]).T

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180
ROAD_ID = 0
BUILDING_ID = 1
TREE_ID = 2
FIELD_ID = 3
UNKNOWN_ID = 4

fnme = '/home/jeff/SDR_DATA/RAW/08072024/SAR_08072024_111617.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / sdr[0].fc
ant_gain = 55
transmit_power = 100
eff_aperture = 10. * 10.
upsample = 1
pixel_to_m = .25

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
    rp.getRadarParams(2., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
chirp_filt = np.fft.fft(sdr[0].cal_chirp, fft_len) * mfilt

threads_per_block = getMaxThreads()

# This is all the constants in the radar equation for this simulation
radar_coeff = transmit_power * 10**(ant_gain / 10) * eff_aperture / (4 * np.pi)**2
chip_shape = (768, 768)

gx, gy, gz = bg.getGrid(bg.origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape)
pix_data = bg.getRefGrid(bg.origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape) * 1e11

points = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
beta = np.zeros(points.shape[0]) + 10.
sigma = pix_data.flatten()
az = np.zeros(points.shape[0])
el = np.zeros(points.shape[0]) - np.pi / 2

boresight = rp.boresight(sdr[0].pulse_time).mean(axis=0)
pointing_az = np.arctan2(boresight[0], boresight[1])
pointing_el = -np.arcsin(boresight[2] / np.linalg.norm(boresight))

plat_pos = rp.txpos(rp.gpst.mean())
vecs = np.array([points[:, 0] - plat_pos[0], points[:, 1] - plat_pos[1],
                         points[:, 2] - plat_pos[2]])

rng_bins = ((np.linalg.norm(vecs, axis=0) * 2 / c0 - 2 * near_range_s) * fs).astype(int)

rng_pts, opt_pts = np.unique(rng_bins, return_index=True)

tmp_pts = points[opt_pts]
tmp_betas = np.ones(len(opt_pts))
tmp_sigmas = np.ones(len(opt_pts)) * 100.
tmp_az = az[opt_pts]
tmp_el = el[opt_pts]
for tt in tqdm(range(0, sdr[0].nframes, 16)):
    times = np.arange(tt, tt + 16)
    use_pts = np.zeros(points.shape[0]).astype(bool)
    permission = False
    for tim in sdr[0].pulse_time[times]:
        plat_pos = rp.txpos(tim)
        vecs = np.array([tmp_pts[:, 0] - plat_pos[0], tmp_pts[:, 1] - plat_pos[1],
                         tmp_pts[:, 2] - plat_pos[2]])
        pt_az = np.arctan2(vecs[0, :], vecs[1, :])
        pt_el = -np.arcsin(vecs[2, :] / np.linalg.norm(vecs, axis=0))
        if sum(np.logical_and(abs(pt_az - pointing_az) < rp.az_half_bw * 2,
                                    abs(pt_el - pointing_el) < rp.el_half_bw * 2)) >= tmp_pts.shape[0] - 2:
            permission = True
            break
    if not permission:
        continue
    for tim in sdr[0].pulse_time[times]:
        plat_pos = rp.txpos(tim)
        vecs = np.array([points[:, 0] - plat_pos[0], points[:, 1] - plat_pos[1],
                         points[:, 2] - plat_pos[2]])
        pt_az = np.arctan2(vecs[0, :], vecs[1, :])
        pt_el = -np.arcsin(vecs[2, :] / np.linalg.norm(vecs, axis=0))
        use_pts[np.logical_and(abs(pt_az - pointing_az) < rp.az_half_bw * 2,
                                    abs(pt_el - pointing_el) < rp.el_half_bw * 2)] = True
    use_pts[opt_pts] = False
    if sum(use_pts) < len(opt_pts):
        continue

    _, pdata = sdr.getPulses(sdr[0].frame_num[times], 0)
    mfdata = np.fft.fft(pdata, fft_len, axis=0) * mfilt[:, None]
    updata = np.zeros((up_fft_len, mfdata.shape[1]), dtype=np.complex128)
    updata[:fft_len // 2, :] = mfdata[:fft_len // 2, :]
    updata[-fft_len // 2:, :] = mfdata[-fft_len // 2:, :]
    updata = np.fft.ifft(updata, axis=0)[:nsam * upsample, :].T

    unopt_pts_gpu = cupy.array(points[use_pts], dtype=np.float32)
    unopt_sigma_gpu = cupy.array(sigma[use_pts], dtype=np.float32)
    source_gpu = cupy.array(rp.txpos(sdr[0].pulse_time[times]), dtype=np.float32)
    pan_gpu = cupy.array(rp.pan(sdr[0].pulse_time[times]), dtype=np.float32)
    tilt_gpu = cupy.array(rp.tilt(sdr[0].pulse_time[times]), dtype=np.float32)
    pd_r = cupy.zeros((len(times), nsam), dtype=np.float64)
    pd_i = cupy.zeros((len(times), nsam), dtype=np.float64)

    bprun = (max(1, 16 // threads_per_block[0] + 1),
             unopt_pts_gpu.shape[0] // threads_per_block[1] + 1)

    calcRangeProfile[bprun, threads_per_block](unopt_pts_gpu, source_gpu, pan_gpu, tilt_gpu, pd_r, pd_i, near_range_s,
                                               fs, rp.az_half_bw, rp.el_half_bw, unopt_sigma_gpu,
                                               2 * np.pi / wavelength, radar_coeff)

    del unopt_sigma_gpu
    del unopt_pts_gpu

    base_pulses = pd_r.get() + pd_i.get() * 1j

    x0 = []
    bounds = []
    for n in range(tmp_pts.shape[0]):
        x0 = np.concatenate((x0, np.array([tmp_pts[n, 2], tmp_betas[n], tmp_sigmas[n], tmp_az[n], tmp_el[n]])))
        bounds += [(tmp_pts[n, 2] - 1., tmp_pts[n, 2] + 10.), (1e-9, 15), (1e-9, 1000.), (1e-9, 2 * np.pi), (-np.pi / 2, np.pi / 2)]

    bprun = (max(1, 16 // threads_per_block[0] + 1),
             tmp_pts.shape[0] // threads_per_block[1] + 1)
    solve_rbins = np.array([np.linalg.norm(np.array([tmp_pts[:, 0] - rp.txpos(sdr[0].pulse_time[0])[0], tmp_pts[:, 1] - rp.txpos(sdr[0].pulse_time[0])[1],
                     tmp_pts[:, 2] - rp.txpos(sdr[0].pulse_time[-1])[2]]), axis=0),
                            np.linalg.norm(np.array([tmp_pts[:, 0] - rp.txpos(sdr[0].pulse_time[-1])[0],
                                                     tmp_pts[:, 1] - rp.txpos(sdr[0].pulse_time[-1])[1],
                                                     tmp_pts[:, 2] - rp.txpos(sdr[0].pulse_time[-1])[2]]), axis=0)])
    solve_ppts = np.array([rp.txpos(sdr[0].pulse_time[0]), rp.txpos(sdr[0].pulse_time[-1])])
    def minfunc(x):
        tmp_pts[:, 2] = x[::5]
        tmp_pts[:, :2] = solvey(*solve_ppts[0], *solve_ppts[1], solve_rbins[0], solve_rbins[1], tmp_pts[:, 2])
        opt_pts_gpu = cupy.array(tmp_pts, dtype=np.float32)
        opt_beta_gpu = cupy.array(x[1::5], dtype=np.float32)
        opt_sigma_gpu = cupy.array(x[2::5], dtype=np.float32)
        opt_normal_gpu = cupy.array(azelToVec(x[3::5], x[4::5]).T, dtype=np.float32)
        pd_r = cupy.zeros((len(times), nsam), dtype=np.float64)
        pd_i = cupy.zeros((len(times), nsam), dtype=np.float64)

        calcRangeProfileScattering[bprun, threads_per_block](opt_pts_gpu, source_gpu, pan_gpu, tilt_gpu, pd_r, pd_i,
                                                   near_range_s, fs, rp.az_half_bw, rp.el_half_bw, opt_sigma_gpu,
                                                             opt_beta_gpu, opt_normal_gpu, 2 * np.pi / wavelength,
                                                             radar_coeff)

        pulses = base_pulses + (pd_r.get() + pd_i.get() * 1j)
        x_hat = np.fft.fft(pulses, fft_len, axis=1) * chirp_filt
        upx = np.zeros((len(times), up_fft_len), dtype=np.complex128)
        upx[:, :fft_len // 2] = x_hat[:, :fft_len // 2]
        upx[:, -fft_len // 2:] = x_hat[:, -fft_len // 2:]
        upx = np.fft.ifft(upx, axis=1)[:, :nsam * upsample]

        return np.linalg.norm(updata[:, rng_pts] - upx[:, rng_pts])


    opt_x = minimize(minfunc, x0, bounds=bounds)

    sigma[opt_pts] = opt_x['x'][2::5]
    beta[opt_pts] = opt_x['x'][1::5]
    points[opt_pts, 2] = opt_x['x'][::5]
    az[opt_pts] = opt_x[3::5]
    el[opt_pts] = opt_x[4::5]



