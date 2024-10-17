import cupy
import numpy as np
import cma
from scipy.spatial import Delaunay
from SDRParsing import load
from backproject_functions import backprojectPulseSet
from cuda_kernels import calcRangeProfile, getMaxThreads
from cuda_mesh_kernels import getRangeProfileFromMesh, getBoxesSamplesFromMesh
from grid_helper import SDREnvironment
from platform_helper import SDRPlatform
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
from simulation_functions import db, upsamplePulse
import plotly.io as pio

pio.renderers.default = 'browser'


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

grid_origin = (40.133830, -111.665722, 1385.)
fnme = '/home/jeff/SDR_DATA/RAW/08072024/SAR_08072024_111617.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / sdr[0].fc
ant_gain = 55
transmit_power = 100
eff_aperture = 10. * 10.
upsample = 1
pixel_to_m = 1.
nboxes = 4
points_to_sample = 256
npulses = 1024
num_bounces = 0
nbounce_rays = 5

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
    rp.getRadarParams(10., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
chirp_filt = np.fft.fft(sdr[0].cal_chirp, fft_len) * mfilt

threads_per_block = getMaxThreads()

# This is all the constants in the radar equation for this simulation
radar_coeff = transmit_power * 10 ** (ant_gain / 10) * eff_aperture / (4 * np.pi) ** 2
chip_shape = (16, 16)

gx, gy, gz = bg.getGrid(grid_origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape)
pix_data = bg.getRefGrid(grid_origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape) * 1e11

points = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T

boresight = rp.boresight(sdr[0].pulse_time).mean(axis=0)
pointing_az = np.arctan2(boresight[0], boresight[1])

# Locate the extrema to speed up the optimization
flight_path = rp.txpos(sdr[0].pulse_time)
pmax = points.max(axis=0)
vecs = np.array([pmax[0] - flight_path[:, 0], pmax[1] - flight_path[:, 1],
                 pmax[2] - flight_path[:, 2]]).T
pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
max_pts = sdr[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw]
pmin = points.min(axis=0)
vecs = np.array([pmin[0] - flight_path[:, 0], pmin[1] - flight_path[:, 1],
                 pmin[2] - flight_path[:, 2]]).T
pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
min_pts = sdr[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw]
pulse_lims = [min(min(max_pts), min(min_pts)), max(max(max_pts), max(min_pts))]

tri_ = Delaunay(points[:, :2])
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(points)
mesh.triangles = o3d.utility.Vector3iVector(tri_.simplices)
mesh.compute_vertex_normals()
mesh.compute_triangle_normals()
mesh.normalize_normals()

mesh.triangle_material_ids = o3d.utility.IntVector([i for i in range(len(mesh.triangles))])
msigmas = [1. for _ in range(len(mesh.triangles))]
pdata = db(pix_data)

x0 = np.concatenate((msigmas, np.zeros(points.shape[0])))

def minfunc(x):
    tmp_p = points + 0.0
    tmp_p[:, 2] += x[len(mesh.triangles):]
    mesh.vertices = o3d.utility.Vector3dVector(tmp_p)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    # Load in boxes and meshes for speedup of ray tracing
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample,
                                                      material_sigmas=x[:len(mesh.triangles)])
    bpj_grid = np.zeros_like(gx).astype(np.complex128)

    # MAIN LOOP
    for frame in range(pulse_lims[0], pulse_lims[1], npulses):
        dt = sdr[0].pulse_time[frame:frame + npulses]
        trp = getRangeProfileFromMesh(*box_tree, sample_points, rp.txpos(dt), rp.boresight(dt),
                                      radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, sdr[0].fc, near_range_s, num_bounces=num_bounces,
                                      bounce_rays=nbounce_rays)
        clean_pulse = np.fft.fft(trp, fft_len) * chirp_filt
        mf_pulse = upsamplePulse(clean_pulse + (np.random.normal(0, 1e-3, size=clean_pulse.shape) + 1j * np.random.normal(0, 1e-3, size=clean_pulse.shape))
        , fft_len, upsample,
            is_freq=True, time_len=nsam)
        bpj_grid += backprojectPulseSet(mf_pulse.T, rp.pan(dt), rp.tilt(dt), rp.rxpos(dt), rp.txpos(dt), gx, gy, gz,
                                       c0 / sdr[0].fc, near_range_s, fs * upsample, rp.az_half_bw, rp.el_half_bw)
    return np.linalg.norm(db(bpj_grid) - pdata)

es = cma.CMAEvolutionStrategy(x0, 3.3, {'bounds': [0, 10]})
es.optimize(minfunc)
