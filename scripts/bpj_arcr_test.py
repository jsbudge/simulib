import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject, cpudiff
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, basinhopping
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
from scipy.signal import medfilt, stft
import plotly.express as px
from scipy.spatial.transform import Rotation as Rot
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm
from SDRParsing import SDRParse, load, findAllFilenames
from SARParsing import SARParse
from celluloid import Camera
from sarpy.geometry import point_projection
from sarpy.io.complex.sicd import SICDReader
from sarpy.processing.sicd.subaperture import subaperture_processing_array
import vg
from pytransform3d.rotations import matrix_from_axis_angle

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def spline(x, k0, y0, x0, x1, x2):
    # Piecewise spline for smooth transitioning between CPIs
    y1 = x[0]
    y2 = x[1]
    k1 = x[2]
    k2 = x[3]
    tx0 = (np.arange(x0, x1) - x0) / (x1 - x0)
    q0 = (1 - tx0) * y0 + tx0 * y1 + tx0 * (1 - tx0) * ((1 - tx0) * (k0 * (x1 - x0) - (y1 - y0)) +
                                                        tx0 * (-k1 * (x1 - x0) + (y1 - y0)))
    tx1 = (np.arange(x1, x2) - x1) / (x2 - x1)
    q1 = (1 - tx1) * y1 + tx1 * y2 + tx1 * (1 - tx1) * ((1 - tx1) * (k1 * (x2 - x1) - (y2 - y1)) +
                                                        tx1 * (-k2 * (x2 - x1) + (y2 - y1)))
    return np.concatenate((q0, q1))


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

# This is the file used to backproject data
# bg_file = '/data5/SAR_DATA/2021/05052021/SAR_05052021_112647.sar'
# bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
# bg_file = '/data5/SAR_DATA/2022/Redstone/SAR_08122022_170753.sar'
bg_file = '/data5/SAR_DATA/2022/09262022/SAR_09262022_143509.sar'
# bg_file = '/data6/Tower_Redo_Again/tower_redo_SAR_03292023_120731.sar'
# bg_file = '/data5/SAR_DATA/2022/09272022/SAR_09272022_103053.sar'
# bg_file = '/data5/SAR_DATA/2019/08072019/SAR_08072019_100120.sar'
upsample = 4
poly_num = 1
use_rcorr = False
use_aps_debug = True
rotate_grid = True
use_ecef = False
cpi_len = 32
plp = 0
partial_pulse_percent = .2
debug = True
pulse_dec_factor = 2
pts_per_m = .25
grid_width = 1000
grid_height = 1000
channel = 0

files = findAllFilenames(bg_file, use_debug=True, exact_matches=False)

print('Loading SDR file...')
is_sar = False
try:
    sdr = load(bg_file)
except KeyError:
    is_sar = True
    print('Using SlimSAR parser instead.')
    sdr = SARParse(bg_file)
    sdr.ash = sdr.loadASH('/data5/SAR_DATA/2019/08072019/SAR_08072019_100120RVV_950000_95.ash')
try:
    origin = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
              getElevation((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'])))
    heading = sdr.gim.initial_course_angle
except TypeError:
    heading = sdr.gim.initial_course_angle
    pt = (sdr.gps_data['lat'].values[0], sdr.gps_data['lon'].values[0])
    alt = getElevation(pt)
    nrange = ((sdr[channel].receive_on_TAC - sdr[channel].transmit_on_TAC) / TAC -
              sdr[channel].pulse_length_S * partial_pulse_percent) * c0 / 2
    frange = ((sdr[channel].receive_off_TAC - sdr[channel].transmit_on_TAC) / TAC -
              sdr[channel].pulse_length_S * partial_pulse_percent) * c0 / 2
    mrange = (nrange + frange) / 2
    origin = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                     (pt[0], pt[1], alt))
# origin = (40.025675, -111.764105, 1413)
# origin = (40.037026, -74.353155, 27)
# origin = (43.020269, -95.579805, 458)
# origin = (40.02612, -74.35176, 26)
origin = (40.032520, -74.347000, 28)
ref_llh = origin

# Generate a platform
print('Generating platform...', end='')

# Bunch of debug files used for testing. These are not necessary for backprojection.
gps_check = True
try:
    for key, val in files.items():
        if f'Channel_{channel + 1}' in key:
            if 'preCorrections' in key:
                preCorr = loadPreCorrectionsGPSData(val)
            elif 'postCorrections' in key:
                postCorr = loadPostCorrectionsGPSData(val)
        elif 'Gimbal' in key:
            gimbal_debug = val
            gimbal_data = loadGimbalData(val)
        elif 'GPSDataPreJump' in key:
            rawGPS = loadGPSData(val)
        elif 'GPSDataPostJump' in key:
            jumpCorrGPS = loadGPSData(val)
except FileNotFoundError:
    gps_check = False
    use_aps_debug = False
    print('Failed to find APS GPS debug outputs.')

if use_aps_debug:
    rp = SDRPlatform(sdr, ref_llh, channel=channel, gps_debug=postCorr, gimbal_debug=gimbal_debug,
                     gps_replace=jumpCorrGPS, use_ecef=use_ecef)
    # rp = SDRPlatform(sdr, ref_llh, channel=channel, gimbal_debug=gimbal_debug)
else:
    rp = SDRPlatform(sdr, ref_llh, channel=channel)

# Atmospheric modeling params
Ns = 313
Nb = 66.65
hb = 12192
if rp.pos(rp.gpst)[2, :].mean() > hb:
    hb = rp.pos(rp.gpst)[2, :].mean() + 1000.
    Nb = 105 * np.exp(-(hb - 9000) / 7023)

# Get reference data
# flight = rp.pos(postCorr['sec'])
fs = sdr[channel].fs
bwidth = sdr[channel].bw
fc = sdr[channel].fc
print('Done.')

# Generate values needed for backprojection
print('Calculating grid parameters...')
# General calculations for slant ranges, etc.
# plat_height = rp.pos(rp.gpst)[2, :].mean()
fdelay = 0
nr = rp.calcPulseLength(fdelay, plp, use_tac=True)
nsam = rp.calcNumSamples(fdelay, plp)
ranges = rp.calcRangeBins(fdelay, upsample, plp)
granges = np.sqrt(ranges**2 - rp.pos(rp.gpst)[2, :].mean()**2)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample

# Chirp and matched filter calculations
if is_sar:
    bpj_wavelength = c0 / (fc - bwidth / 2 - 5e6)
    mfilt = sdr.genMatchedFilter(0)
else:
    try:
        bpj_wavelength = c0 / (fc - bwidth / 2 - sdr[channel].xml['DC_Offset_MHz'] * 1e6) \
            if sdr[channel].xml['Offset_Video_Enabled'] == 'True' else c0 / fc
    except KeyError as e:
        f'Could not find {e}'
        bpj_wavelength = c0 / (fc - bwidth / 2 - 5e6)

    mfilt = GetAdvMatchedFilter(sdr[channel], fft_len=fft_len)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
rbins_gpu = cupy.array(ranges, dtype=np.float64)

# Calculate out points on the arcr grid
ideal_vel = (rp.pos(rp.gpst[-1]) - rp.pos(rp.gpst[0])) / (rp.gpst[-1] - rp.gpst[0])
# Calculate the mean of the points, i.e. the 'center' of the cloud
datamean = rp.pos(rp.gpst).mean(axis=1)

# Do an SVD on the mean-centered data.
uu, dd, vv = np.linalg.svd(rp.pos(rp.gpst).T - datamean, full_matrices=False)
ideal_vel = vv[0] * np.linalg.norm(ideal_vel)
# downrange_vec = azelToVec(np.arctan2(ideal_vel[0], ideal_vel[1]) + np.pi / 2, rp.dep_ang)

coll_time = sdr[0].sys_time / TAC
ideal_path = rp.pos(rp.gpst[0]) + np.outer(coll_time, ideal_vel)
azes = np.arctan2(-ideal_path[:, 1], -ideal_path[:, 0])
eles = np.arcsin(ideal_path[:, 2] / np.linalg.norm(ideal_path, axis=1))
downrange_vec = azelToVec(np.arctan2(ideal_vel[0], ideal_vel[1]) + np.pi / 2,
                          eles[(azes - np.pi / 2) == (azes - np.pi / 2).min()][0])
coa_pt = ideal_path.mean(axis=0)
'''downrange = np.outer(ranges[2000:10000:5], downrange_vec)
gx = sum(np.meshgrid(ideal_path[:, 0], downrange[:, 0]))
gy = sum(np.meshgrid(ideal_path[:, 1], downrange[:, 1]))
gz = sum(np.meshgrid(ideal_path[:, 2], downrange[:, 2]))'''

# Calculate out grid
'''nu = 800
nv = 800
du = (ranges[1] - ranges[0]) * nu / nu
dv = du
u = np.arange(-nu / 2, nu / 2) * du
v = np.arange(-nv / 2, nv / 2) * dv
n_hat = np.cross(ideal_vel / np.linalg.norm(ideal_vel), downrange_vec)
v_hat = np.cross(n_hat, downrange_vec)/np.linalg.norm(np.cross(n_hat, downrange_vec))
u_hat = np.cross(v_hat, n_hat)/np.linalg.norm(np.cross(v_hat, n_hat))
# Represent u and v in (x,y,z)
[uu, vv] = np.meshgrid(u, v)
uu = uu.flatten()
vv = vv.flatten()

A = np.asmatrix(np.hstack((
    np.array([u_hat]).T, np.array([v_hat]).T
)))
b = np.asmatrix(np.vstack((uu, vv)))
pixel_locs = np.asarray(A * b)
gx = pixel_locs[0, :].reshape((nu, nv))
gy = pixel_locs[1, :].reshape((nu, nv))
gz = pixel_locs[2, :].reshape((nu, nv))'''

# Grid using downrange and isodoppler contours
'''downrange_az = np.arctan2(downrange_vec[0], downrange_vec[1])
downrange_el = -np.arcsin(downrange_vec[2])
gx = np.zeros((4000, 4000))
gy = np.zeros_like(gx)
gz = np.zeros_like(gx)
init_rng = 0
for n, ang in enumerate(np.linspace(downrange_az - rp.az_half_bw, downrange_az + rp.az_half_bw, gx.shape[0])):
    vecs = np.outer(azelToVec(ang, downrange_el), ranges) + coa_pt[:, None]
    gx[:, n] = vecs[0, init_rng:init_rng + 4 * gx.shape[1]:4]
    gy[:, n] = vecs[1, init_rng:init_rng + 4 * gx.shape[1]:4]
    gz[:, n] = vecs[2, init_rng:init_rng + 4 * gx.shape[1]:4]'''

normal = np.array([0, 0, 1.])
vel = ideal_vel
scoa = ideal_path.mean(axis=0)
vecs = scoa[:, None] - np.array([gx.flatten(), gy.flatten(), gz.flatten()])
Rg = np.linalg.norm(vecs, axis=0)
fd = (c0 + ideal_vel.dot(vecs / Rg)) / c0 * fc - fc

plt.figure()
plt.scatter(gx.flatten(), gy.flatten(), c=ideal_vel.dot(vecs) - fd * c0 * Rg / fc)


def minfunc(x):
    Rhat = scoa - x
    ncon = normal.dot(x)
    doppcon = vel.dot(Rhat) - fd * c0 * Rg / fc
    return np.sqrt(ncon**2 + doppcon**2)


test = minimize(minfunc, np.array([0, 0, 0]))

bg = SDREnvironment(sdr)
bg.ref = origin
gx, gy, gz = bg.getGrid(origin, grid_width, grid_height, (1000, 1000))
gz = np.zeros_like(gx)


# fig = px.scatter_3d(x=rotmat[:, 0], y=rotmat[:, 1], z=rotmat[:, 2])
# fig.add_scatter3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
# fig.show()

'''path_r = np.linalg.norm(ideal_path, axis=1)
gr_center = np.where(abs(ranges - np.min(path_r)) == abs(ranges - np.min(path_r)).min())[0][0]
path_center = np.where(path_r == path_r.min())[0][0]
gx = sum(np.meshgrid(granges[gr_center - 1000 * upsample:gr_center + 1000 * upsample:2] *
                     np.sin(np.arctan2(ideal_vel[1], ideal_vel[0]) - np.pi / 2),
                     ideal_path[path_center - 4000:path_center + 4000:pulse_dec_factor, 1]))
gy = sum(np.meshgrid(granges[gr_center - 1000 * upsample:gr_center + 1000 * upsample:2] *
                     np.cos(np.arctan2(ideal_vel[1], ideal_vel[0]) - np.pi / 2),
                     ideal_path[path_center - 4000:path_center + 4000:pulse_dec_factor, 0]))
gz = np.zeros_like(gx)'''

nbpj_pts = gx.shape
gx_gpu = cupy.array(gx, dtype=np.float64)
gy_gpu = cupy.array(gy, dtype=np.float64)
gz_gpu = cupy.array(gz, dtype=np.float64)

if debug:
    pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    # pts_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
    # angs_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, nbpj_pts[0] // threads_per_block[0] + 1), nbpj_pts[1] // threads_per_block[1] + 1)
# rng_states = create_xoroshiro128p_states(triangles.shape[0], seed=10)

# Data blocks for imaging
bpj_truedata = np.zeros(nbpj_pts, dtype=np.complex128)

# Run through loop to get data simulated
data_t = sdr[channel].pulse_time[::pulse_dec_factor]
idx_t = sdr[channel].frame_num[::pulse_dec_factor]
test = None
freqs = np.fft.fftfreq(fft_len, 1 / sdr[0].fs)
print('Backprojecting...')
pulse_pos = 0
spline_knots = [(0, 0, 0.)]
for tidx, frames in tqdm(enumerate(idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)),
                         total=len(data_t) // cpi_len + 1):
    ts = data_t[tidx * cpi_len + np.arange(len(frames))]
    tmp_len = len(ts)

    if not np.all(cpudiff(np.arctan2(-rp.pos(ts)[1, :], rp.pos(ts)[0, :]), rp.pan(ts)) -
                  rp.az_half_bw < 0):
        continue
    panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
    postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)
    bpj_grid = cupy.zeros(nbpj_pts, dtype=np.complex128)

    # Armin Doerry's corrections for atmospheric changes to the speed of light
    Hb = (hb - ref_llh[2]) / np.log(Ns / Nb)
    rcatmos = (1 + (Hb * 10e-6 * Ns) / rp.pos(ts)[2, :] *
               (1 - np.exp(-rp.pos(ts)[2, :] / Hb))) ** -1
    if use_rcorr:
        r_corr_gpu = cupy.array(rcatmos, dtype=np.float64)
    else:
        r_corr_gpu = cupy.array(np.ones_like(ts), dtype=np.float64)

    # Reset the grid for truth data
    rtdata = cupy.fft.fft(cupy.array(sdr.getPulses(frames, channel),
                                     dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, :tmp_len]
    upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * upsample, :]
    cupy.cuda.Device().synchronize()

    backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                            elrx_gpu,
                                            panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                            bpj_wavelength, ranges[0] / c0,
                                            rp.fs * upsample, bwidth, rp.az_half_bw, rp.el_half_bw, poly_num, pts_debug,
                                            angs_debug, debug, r_corr_gpu)
    cupy.cuda.Device().synchronize()

    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp = rp.pos(ts[-1])
        test = rtdata.get()
        angd = angs_debug.get()
        locd = pts_debug.get()
    bpj_truedata += bpj_grid.get()

del panrx_gpu
del postx_gpu
del posrx_gpu
del elrx_gpu
del rtdata
del upsample_data
del bpj_grid
del r_corr_gpu
# del shift

del rbins_gpu
del gx_gpu
del gy_gpu
del gz_gpu

# bfig = px.scatter(x=gx.flatten(), y=gy.flatten(), color=db(bpj_truedata).flatten())
# bfig.add_scatter(x=rp.pos(rp.gpst)[0, :], y=rp.pos(rp.gpst)[1, :])
# bfig.show()

if test is not None:
    plt.figure('Doppler data')
    plt.imshow(np.fft.fftshift(db(np.fft.fft(test, axis=1)), axes=1), extent=[0, cpi_len, ranges[0], ranges[-1]])
    plt.axis('tight')

plt.figure('IMSHOW backprojected data')
plt.imshow(db(bpj_truedata), origin='lower')
plt.axis('tight')

try:
    bg = SDREnvironment(sdr_file=sdr)
    bg.ref = origin

    plt.figure('IMSHOW truth data')
    plt.imshow(db(bg.refgrid), origin='lower')
    plt.axis('tight')
    if (nbpj_pts[0] * nbpj_pts[1]) < 400**2:
        fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
        fig.add_scatter3d(x=rp.pos(rp.gpst)[0, :], y=rp.pos(rp.gpst)[1, :], z=rp.pos(rp.gpst)[2, :], mode='markers')
        fig.show()
except:
    pass

from sklearn.preprocessing import power_transform

plot_data = np.fliplr(db(bpj_truedata))
scaled_data = power_transform(db(bpj_truedata).reshape(-1, 1)).reshape(plot_data.shape)

# px.imshow(plot_data, color_continuous_scale=px.colors.sequential.gray, zmin=130, zmax=180).show()
px.imshow(np.fliplr(scaled_data), color_continuous_scale=px.colors.sequential.gray, zmin=-1, zmax=3).show()

# Get FFT output section
slice_bpj = np.zeros_like(bpj_truedata)
slice_bpj = bpj_truedata
fft_bpj = np.fft.fftshift(np.fft.fft2(slice_bpj))
plt.figure('Spatial FFT')
plt.subplot(2, 1, 1)
plt.imshow(db(fft_bpj))
plt.axis('tight')
plt.subplot(2, 1, 2)
plt.imshow(db(slice_bpj))
plt.clim(db(slice_bpj[slice_bpj > 0]).min(), db(slice_bpj).max())
plt.axis('tight')

'''reader = SICDReader('/data5/SAR_DATA/2022/09262022/SAR_09262022_143509LVV_ch2_929500_58.ntf')
structure = reader.sicd_meta
lat, lon, hght = enu2llh(gx.flatten(), gy.flatten(), gz.flatten(), bg.ref)
image_coords, _, _ = point_projection.ground_to_image_geo(np.array([lat, lon, hght]).T, structure)
for x in range(0, image_coords.shape[0], 60000):
    plt.figure(f'SICD image transform_{x}')
    plt.subplot(2, 1, 1)
    plt.title('SICD')
    plt.scatter(image_coords[x:x+60000, 0], image_coords[x:x+60000, 1], c=scaled_data.flatten()[x:x+60000])
    plt.subplot(2, 1, 2)
    plt.title('Normal')
    plt.scatter(lat[x:x + 60000], lon[x:x + 60000], c=scaled_data.flatten()[x:x + 60000])

plt.figure()
plt.scatter(lat[24000:], gy.flatten()[24000:], c=db(bpj_truedata).flatten()[24000:])'''

'''sect = bpj_truedata
fig = plt.figure()
cam = Camera(fig)
for n in range(0, sect.shape[0], 64):
    test = subaperture_processing_array(sect, (n, n+128), 512)
    # plt.figure(f'ap_{n}')
    plt.imshow(db(test))
    plt.axis('tight')
    cam.snap()
anim = cam.animate()'''

xshift = 64
xx, yy = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, xshift))
grangles = np.linspace(np.arcsin(ideal_path.mean(axis=0)[2] / np.linalg.norm(ideal_path.mean(axis=0) + np.array([-50, 0, 0]))),
np.arcsin(ideal_path.mean(axis=0)[2] / np.linalg.norm(ideal_path.mean(axis=0) + np.array([50, 0, 0]))),
                       bpj_truedata.shape[0])
grangles = np.arccos(granges / ranges)
sect = bpj_truedata
'''fig, ax = plt.subplots(1, 2)
cam = Camera(fig)
for n in range(0, sect.shape[0], xshift):
    test = np.fft.fftshift(np.fft.fft2(slice_bpj[n:min(n+xshift, sect.shape[0]), 200:328]))
    shift_test = np.fft.fftshift(
        np.fft.fft2(np.exp(1j * 2 * np.pi * grangles[n * 10] / (c0 / sdr[0].fc) * yy)))
    ax[0].imshow(db(test))
    plt.axis('tight')
    ax[1].imshow(db(shift_test))
    plt.axis('tight')
    cam.snap()
anim = cam.animate()'''

plt.figure('STFT')
plt.imshow(np.fft.fftshift(db(stft(sect[:, 15])[2]), axes=0))
plt.axis('tight')


'''fft_bpj = np.fft.fftshift(fft_bpj).T
fig, ax = plt.subplots(2)
cam = Camera(fig)
for idx in tqdm(range(0, fft_bpj.shape[0], 64)):
    fft_slice = np.zeros_like(fft_bpj)
    fft_slice[idx:idx + 64, :] = fft_bpj[idx:idx + 64, :]
    # fft_slice = np.fft.fftshift(fft_slice)
    ifft_bpj = np.fft.ifft2(fft_slice)
    ax[0].imshow(db(fft_slice))
    plt.axis('tight')
    ax[1].imshow(db(ifft_bpj))
    plt.axis('tight')
    cam.snap()
anim = cam.animate()'''


if gps_check:
    plt.figure('Raw GPS data')
    plt.subplot(2, 2, 1)
    plt.title('Lat')
    plt.plot(rawGPS['gps_ms'], rawGPS['lat'])
    plt.plot(sdr.gps_data.index, sdr.gps_data['lat'])
    plt.subplot(2, 2, 2)
    plt.title('Lon')
    plt.plot(rawGPS['gps_ms'], rawGPS['lon'])
    plt.plot(sdr.gps_data.index, sdr.gps_data['lon'])
    plt.subplot(2, 2, 3)
    plt.title('Alt')
    plt.plot(rawGPS['gps_ms'], rawGPS['alt'])
    plt.plot(sdr.gps_data.index, sdr.gps_data['alt'])

    rerx = rp.rxpos(postCorr['sec'])[0, :]
    rnrx = rp.rxpos(postCorr['sec'])[1, :]
    rurx = rp.rxpos(postCorr['sec'])[2, :]
    gnrx = postCorr['rx_lat'] * postCorr['latConv'] - origin[0] * postCorr['latConv']
    gerx = postCorr['rx_lon'] * postCorr['lonConv'] - origin[1] * postCorr['lonConv']
    gurx = postCorr['rx_alt'] - origin[2]
    gerx, gnrx, gurx = llh2enu(postCorr['rx_lat'], postCorr['rx_lon'], postCorr['rx_alt'], origin)
    retx = rp.txpos(postCorr['sec'])[0, :]
    rntx = rp.txpos(postCorr['sec'])[1, :]
    rutx = rp.txpos(postCorr['sec'])[2, :]
    gntx = postCorr['tx_lat'] * postCorr['latConv'] - origin[0] * postCorr['latConv']
    getx = postCorr['tx_lon'] * postCorr['lonConv'] - origin[1] * postCorr['lonConv']
    gutx = postCorr['tx_alt'] - origin[2]
    plt.figure('ENU diff')
    plt.subplot(2, 2, 1)
    plt.title('E')
    plt.plot(postCorr['sec'], rerx - gerx)
    plt.plot(postCorr['sec'], retx - getx)
    plt.subplot(2, 2, 2)
    plt.title('N')
    plt.plot(postCorr['sec'], rnrx - gnrx)
    plt.plot(postCorr['sec'], rntx - gntx)
    plt.subplot(2, 2, 3)
    plt.title('U')
    plt.plot(postCorr['sec'], rurx - gurx)
    plt.plot(postCorr['sec'], rutx - gutx)
    plt.legend(['Rx', 'Tx'])

    rp_r = rp.att(preCorr['sec'])[0, :]
    rp_p = rp.att(preCorr['sec'])[1, :]
    rp_y = rp.att(preCorr['sec'])[2, :]
    plt.figure('rpy')
    plt.subplot(2, 2, 1)
    plt.title('r')
    plt.plot(preCorr['sec'], rp_r)
    plt.plot(preCorr['sec'], preCorr['r'])
    plt.plot(rawGPS['gps_ms'], rawGPS['r'])
    plt.subplot(2, 2, 2)
    plt.title('p')
    plt.plot(preCorr['sec'], rp_p)
    plt.plot(preCorr['sec'], preCorr['p'])
    plt.plot(rawGPS['gps_ms'], rawGPS['p'])
    plt.subplot(2, 2, 3)
    plt.title('y')
    plt.plot(preCorr['sec'], rp_y - 2 * np.pi)
    plt.plot(postCorr['sec'], postCorr['az'])
    plt.plot(preCorr['sec'], preCorr['az'])
    plt.legend(['sdr', 'interp_sdr', 'pre', 'raw'])

    times = np.interp(gimbal_data['systime'], sdr.gps_data['systime'], sdr.gps_data.index)
    plt.figure('Gimbal')
    plt.subplot(2, 1, 1)
    plt.title('Pan')
    plt.plot(times, gimbal_data['pan'])
    plt.plot(times, sdr.gimbal['pan'])
    plt.subplot(2, 1, 2)
    plt.title('Tilt')
    plt.plot(times, gimbal_data['tilt'])
    plt.plot(times, sdr.gimbal['tilt'])
plt.show()