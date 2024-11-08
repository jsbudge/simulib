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
from cuda_kernels import calcRangeProfile, getMaxThreads, calcRangeProfileScattering, applyRadiationPatternCPU
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

def solvey(x1, y1, z1, x0, y0, z0, R1, R0, z):
    y = (
                    4 * y1 ** 3 - 4 * y0 * y1 ** 2 + 4 * x1 ** 2 * y1 + 4 * z1 ** 2 * y1 + 4 * x0 ** 2 * y1 - 4 * y0 ** 2 * y1 - 4 * z0 ** 2 * y1 -
                    4 * R1 ** 2 * y1 + 4 * R0 ** 2 * y1 - 8 * x1 * x0 * y1 - 8 * z1 * z * y1 + 8 * z0 * z * y1 + 4 * y0 ** 3 + 4 * y0 * z0 ** 2 +
                    4 * y0 * R1 ** 2 - 4 * y0 * R0 ** 2 + 4 * x1 ** 2 * y0 - 4 * z1 ** 2 * y0 + 4 * x0 ** 2 * y0 - 8 * x1 * x0 * y0 + 8 * z1 * y0 * z -
                    8 * y0 * z0 * z - np.sqrt(
                (-4 * y1 ** 3 + 4 * y0 * y1 ** 2 - 4 * x1 ** 2 * y1 - 4 * z1 ** 2 * y1 - 4 * x0 ** 2 * y1 +
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
    x = x1 + np.sqrt(-y1 ** 2 + 2 * y1 * y - z1 ** 2 + 2 * z1 * z + R0 ** 2 - y ** 2 - z ** 2)
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
pt_to_check = [15, 15]

# This is all the constants in the radar equation for this simulation
radar_coeff = transmit_power * 10 ** (ant_gain / 10) * eff_aperture / (4 * np.pi) ** 2 * 1e5
chip_shape = (768, 768)

gx, gy, gz = bg.getGrid(bg.origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape)
pix_data = bg.getRefGrid(bg.origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape) * 1e11
pointing_vec = rp.boresight(rp.gpst).mean(axis=0)
point_az = np.arctan2(pointing_vec[0], pointing_vec[1])
point_el = -np.arcsin(pointing_vec[2])

pt = np.array([gx[*pt_to_check], gy[*pt_to_check], gz[*pt_to_check]])
pt_norm = -pointing_vec # azelToVec(np.pi / 2, np.pi / 4)
pt_beta = 1.


flight_pos = rp.txpos(sdr[0].pulse_time)

pt_vec = pt - flight_pos
pt_rng = np.linalg.norm(pt_vec, axis=1)
pt_dir = pt_vec / pt_rng[:, None]
pt_az = np.arctan2(pt_vec[:, 0], pt_vec[:, 1])
pt_el = -np.arcsin(pt_vec[:, 2] / pt_rng)
rng_bin = ((pt_rng * 2 / c0 - 2 * near_range_s) * fs).astype(int)

frames = sdr[0].frame_num[(abs(pt_az - point_az) - rp.az_half_bw) <= 0]
nframes = len(frames)

real_data = np.zeros(nframes, dtype=np.complex128)
calc_data = np.zeros_like(real_data)
for idx in tqdm(range(0, nframes, 64)):
    idx_rng = range(idx, min(idx + 64, nframes))
    val_rng = range(frames[idx], frames[idx] + len(idx_rng))
    _, data = sdr.getPulses(sdr[0].frame_num[val_rng], 0)
    data = data.T
    real_data[idx_rng] = data[np.arange(data.shape[0]), rng_bin[val_rng]]
    tmp_data = np.zeros_like(data)

    # Bounce
    bounce_dot = np.sum(pt_dir[val_rng] * pt_norm, axis=1) * 2.
    bounce = pt_dir[val_rng] - np.outer(bounce_dot, pt_norm)
    bounce = bounce / np.linalg.norm(bounce, axis=1)[:, None]

    # Attenuation
    delta_phi = np.sum(bounce * pt_norm, axis=1)
    delta_phi[delta_phi <= 0] = 1e-9
    att = 1 - np.exp(-delta_phi**2 / (2 * pt_beta**2))
    nrho = radar_coeff * att / pt_rng[val_rng]**4
    nrho *= np.array([applyRadiationPatternCPU(point_az, point_el, pt_a, pt_e, pt_a, pt_e, rp.az_half_bw, rp.el_half_bw) for pt_a, pt_e in zip(pt_az[val_rng], pt_el[val_rng])])
    tmp_data[:, rng_bin[val_rng]] += np.exp(1j * 2 * np.pi * sdr[0].fc / c0 * pt_rng[val_rng] * 2) * nrho
    tmp_data = np.fft.ifft(np.fft.fft(tmp_data, fft_len, axis=1) * chirp_filt)[:, :nsam]

    calc_data[idx_rng] = tmp_data[np.arange(data.shape[0]), rng_bin[val_rng]] * 1e8

plt.figure(); plt.plot(calc_data.real); plt.plot(real_data.real); plt.show()