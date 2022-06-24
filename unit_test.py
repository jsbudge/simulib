import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
from SDRParsing import SDRParse

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 1e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

bg_file = '/data5/SAR_DATA/2022/03032022/SAR_03032022_130706.sar'
upsample = 1
cpi_len = 128
plp = .5
pts_per_tri = 1
debug = True
nbpj_pts = 700

print('Loading SDR file...')
sdr = SDRParse(bg_file)

# Generate the background for simulation
print('Generating environment...', end='')
# bg = MapEnvironment((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'], sdr.ash['geo']['hRef']), extent=(120, 120))
bg = SDREnvironment(sdr, num_vertices=20000)

# Grab vertices and such
vertices = bg.vertices
triangles = bg.triangles
normals = bg.normals
# isCircle = np.linalg.norm(vertices[:, :2], axis=1) == np.linalg.norm(vertices[:, :2], axis=1).min()
# bg.setReflectivityCoeffs(isCircle * 10000, 10000)
print('Done.')

# Generate a platform
print('Generating platform...', end='')
rp = SDRPlatform(sdr, bg.origin)

# Get reference data
flight = rp.pos(rp.gpst)
fc = sdr[0].fc
bwidth = sdr[0].bw
pan = CubicSpline(rp.gpst, rp.heading(rp.gpst) - np.pi / 2)
el = lambda x: np.zeros(len(x)) + rp.dep_ang
print('Done.')

# Generate a backprojected image
print('Calculating grid parameters...')
# General calculations for slant ranges, etc.
# plat_height = rp.pos(rp.gpst)[2, :].mean()
plat_height = 0
nr = rp.calcPulseLength(plat_height, plp, use_tac=True)
nsam = rp.calcNumSamples(plat_height, plp)
ranges = rp.calcRangeBins(plat_height, upsample, plp)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample

# Chirp and matched filter calculations
taywin = int(sdr_f[0].bw / fs * up_fft_len)
taytay = taylor(taywin)
tayd = np.fft.fftshift(taylor(cpi_len))
taydopp = np.fft.fftshift(np.ones((nsam * upsample, 1)).dot(tayd.reshape(1, -1)), axes=1)
chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, rp.fs, fc,
                            bwidth) * 1e4, up_fft_len)
mfilt = chirp.conj()
mfilt[:taywin // 2] *= taytay[taywin // 2 + 1:]
mfilt[-taywin // 2:] *= taytay[:taywin // 2 + 1]
mfilt[taywin // 2:-taywin // 2] = 0
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)

# Generate a test strip of data
tri_vert_indices = cupy.array(triangles, dtype=np.int32)
vert_xyz = cupy.array(vertices, dtype=np.float64)
vert_norms = cupy.array(normals, dtype=np.float64)
scattering_coef = cupy.array(bg.scat_coefs, dtype=np.float64)
ref_coef_gpu = cupy.array(bg.ref_coefs, dtype=np.float64)
rbins_gpu = cupy.array(ranges, dtype=np.float64)

# Calculate out points on the ground
gx, gy = np.meshgrid(np.linspace(-150, 150, nbpj_pts) + 328.28, np.linspace(-150, 150, nbpj_pts) + 110.37)
latg, long, altg = enu2llh(gx.flatten(), gy.flatten(), np.zeros(gx.flatten().shape[0]), bg.origin)
gz = (getElevationMap(latg, long) - bg.origin[2]).reshape(gx.shape)
gx_gpu = cupy.array(gx, dtype=np.float64)
gy_gpu = cupy.array(gy, dtype=np.float64)
gz_gpu = cupy.array(gz, dtype=np.float64)

if debug:
    pts_debug = cupy.zeros((triangles.shape[0], 3), dtype=np.float64)
    angs_debug = cupy.zeros((triangles.shape[0], 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_ranges = (max(1, triangles.shape[0] // threads_per_block[0] + 1), cpi_len // threads_per_block[1] + 1)
bpg_bpj = (max(1, nbpj_pts // threads_per_block[0] + 1), nbpj_pts // threads_per_block[1] + 1)
# rng_states = create_xoroshiro128p_states(triangles.shape[0], seed=10)

# Data blocks for imaging
bpj_res = np.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

# Run through loop to get data simulated
data_t = np.interp(np.linspace(0, len(rp.gpst),
                               int((rp.gpst[-1] - rp.gpst[0]) * rp._sdr[0].nframes / (rp.gpst[-1] - rp.gpst[0]))),
                   np.arange(len(rp.gpst)), rp.gpst)
print('Simulating...')
pulse_pos = 0
for ts in tqdm([data_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)]):
    tmp_len = len(ts)
    # att = rp.att(ts)
    panrx_gpu = cupy.array(pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(el(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.pos(ts), dtype=np.float64)
    data_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    data_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    bpj_grid = cupy.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)
    genRangeWithoutIntersection[bpg_ranges, threads_per_block](tri_vert_indices, vert_xyz, vert_norms,
                                                                    scattering_coef, ref_coef_gpu,
                                                                    posrx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                                    panrx_gpu,
                                                                    elrx_gpu, data_r, data_i, pts_debug, angs_debug,
                                                                    c0 / fc, rp.calcRanges(plat_height)[0] / c0,
                                                                    rp.fs, rp.az_half_bw, rp.el_half_bw, pts_per_tri,
                                                                    debug)

    cupy.cuda.Device().synchronize()
    rtdata = cupy.fft.fft(data_r + 1j * data_i, fft_len, axis=0)
    upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.fft.ifft(upsample_data * chirp_gpu[:, :tmp_len] *
                           mfilt_gpu[:, :tmp_len], axis=0)[:nsam * upsample, :]
    cupy.cuda.Device().synchronize()
    # rtdata = cupy.array(sdr.getPulses(np.arange(pulse_pos, pulse_pos + len(ts)), 0), dtype=np.complex128)
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp = rp.pos(ts[-1])
        test = rtdata.get()
        angd = angs_debug.get()
        locd = pts_debug.get()

    backproject[bpg_bpj, threads_per_block](posrx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu, elrx_gpu,
                                            panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                            c0 / fc, rp.calcRanges(plat_height)[0] / c0,
                                            rp.fs, bwidth, rp.az_half_bw, rp.el_half_bw, 0)
    cupy.cuda.Device().synchronize()

    bpj_res += bpj_grid.get()

    del panrx_gpu
    del elrx_gpu
    del data_r
    del data_i
    del rtdata
    del upsample_data
    del bpj_grid

del rbins_gpu
del gx_gpu
del gy_gpu
del gz_gpu

dfig = go.Figure(data=[go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                 vertexcolor=db(bg.ref_coefs))])
dfig.add_scatter3d(x=flight[0, :], y=flight[1, :], z=flight[2, :])
dfig.add_scatter3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
dfig.show()

bfig = px.imshow(db(bpj_res))
bfig.show()

tfig = px.imshow(db(np.fft.fft(test, axis=1)), aspect='auto')
tfig.show()

v1 = vertices[triangles[:, 0], :]
v2 = vertices[triangles[:, 1], :]
v3 = vertices[triangles[:, 2], :]

tri_pts = v1 * .33 + v2 * .33 + v3 * .33
dists = np.linalg.norm(tri_pts - np.array([328.28, 110.37, 23]), axis=1)
closest = np.where(dists == dists.min())[0][0]
