import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject, cpudiff, backproject_gmti
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
from scipy.signal import medfilt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm
from SDRParsing import SDRParse, load
from SARParsing import SARParse
from celluloid import Camera

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
bg_file = '/data5/SAR_DATA/2022/07282022/SAR_07282022_143945.sar'
# bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
# bg_file = '/data5/SAR_DATA/2022/Redstone/SAR_08122022_170753.sar'
# bg_file = '/data6/SAR_DATA/2023/06202023/SAR_06202023_135617.sar'
# bg_file = '/data6/Tower_Redo_Again/tower_redo_SAR_03292023_120731.sar'
# bg_file = '/data5/SAR_DATA/2022/09272022/SAR_09272022_103053.sar'
# bg_file = '/data5/SAR_DATA/2019/08072019/SAR_08072019_100120.sar'
upsample = 2
poly_num = 1
use_rcorr = False
use_aps_debug = True
rotate_grid = True
use_ecef = False
cpi_len = 512
plp = 0
partial_pulse_percent = .2
debug = True
pts_per_m = .25
grid_width = 100
grid_height = 100
channel = 0

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
ref_llh = origin

# Generate a platform
print('Generating platform...', end='')

# Bunch of debug files used for testing. These are not necessary for backprojection.
gps_check = True
if use_aps_debug:
    if is_sar:
        try:
            sdr_name = bg_file.split('/')[-1].split('.')[0]
            date_name = sdr_name.split('_')[1]
            gps_debug = f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_Band_{channel + 1}_X-Band_VV_Ant_1_postCorrectionsGPSData.dat'
            gimbal_debug = f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_Gimbal.dat'
            gimbal_data = loadGimbalData(gimbal_debug)
            postCorr = loadPostCorrectionsGPSData(gps_debug)
            rawGPS = loadGPSData(f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_GPSDataPreJumpCorrection.dat')
            preCorr = loadPreCorrectionsGPSData(
                f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_Band_{channel + 1}_X-Band_VV_Ant_1_preCorrectionsGPSData.dat')
        except FileNotFoundError:
            gps_check = False
    else:
        try:
            sdr_name = bg_file.split('/')[-1].split('.')[0]
            date_name = sdr_name.split('_')[1]
            gps_debug = f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_Channel_{channel + 1}_X-Band_9_GHz_VV_postCorrectionsGPSData.dat'
            gimbal_debug = f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_Gimbal.dat'
            gimbal_data = loadGimbalData(gimbal_debug)
            postCorr = loadPostCorrectionsGPSData(gps_debug)
            rawGPS = loadGPSData(f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_GPSDataPreJumpCorrection.dat')
            preCorr = loadPreCorrectionsGPSData(
                f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_Channel_{channel + 1}_X-Band_9_GHz_VV_preCorrectionsGPSData.dat')
        except FileNotFoundError:
            gps_check = False

    gps_r = loadGPSData(
        f'/home/jeff/repo/Debug/{date_name}/{sdr_name}_GPSDataPostJumpCorrection.dat')
    # gps_r['r'] = medfilt(gps_r['r'], 21)
    # gps_r['p'] = medfilt(gps_r['p'], 21)
    # gps_r['azimuthX'] = medfilt(gps_r['azimuthX'], 21)
    # gps_r['azimuthY'] = medfilt(gps_r['azimuthY'], 21)
    rp = SDRPlatform(sdr, ref_llh, channel=channel, gps_debug=gps_debug, gimbal_debug=gimbal_debug,
                     gps_replace=gps_r, use_ecef=use_ecef)
    # rp = SDRPlatform(sdr, ref_llh, channel=channel, gimbal_debug=gimbal_debug)
else:
    gps_check = False
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
granges = ranges * np.cos(rp.dep_ang)
nbpj_pts = (cpi_len, len(ranges))
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

# Calculate out points on the ground
max_vel = -sdr[0].prf * bpj_wavelength / 4
gr, gvel = np.meshgrid(ranges, np.linspace(max_vel, -max_vel, cpi_len))
granges_gpu = cupy.array(gr, dtype=np.float64)
gvel_gpu = cupy.array(gvel, dtype=np.float64)

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, gr.shape[0] // threads_per_block[0] + 1), gr.shape[1] // threads_per_block[1] + 1)
# rng_states = create_xoroshiro128p_states(triangles.shape[0], seed=10)

# Data blocks for imaging
bpj_truedata = []
fig = plt.figure('Camera')
cam = Camera(fig)

# Run through loop to get data simulated
data_t = sdr[channel].pulse_time
idx_t = sdr[channel].frame_num
test = None
freqs = np.fft.fftfreq(fft_len, 1 / sdr[0].fs)
print('Backprojecting...')
pulse_pos = 0
for tidx, frames in tqdm(enumerate(idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)),
                         total=len(data_t) // cpi_len + 1):
    ts = data_t[tidx * cpi_len + np.arange(len(frames))]
    tmp_len = len(ts)

    if not np.all(cpudiff(np.arctan2(-rp.pos(ts)[1, :], rp.pos(ts)[0, :]), rp.pan(ts)) -
                  rp.az_half_bw < 0):
        continue
    pantx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    veltx_gpu = cupy.array(rp.vel(ts), dtype=np.float64)
    postx_gpu = cupy.array(rp.txpos(ts)[2, :], dtype=np.float64)
    bpj_grid = cupy.zeros(nbpj_pts, dtype=np.complex128)

    # Reset the grid for truth data
    rtdata = cupy.fft.fft(cupy.array(sdr.getPulses(frames, channel),
                                     dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, :tmp_len]
    upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * upsample, :]
    cupy.cuda.Device().synchronize()

    backproject_gmti[bpg_bpj, threads_per_block](postx_gpu, veltx_gpu, granges_gpu, gvel_gpu, pantx_gpu, rtdata, bpj_grid,
                                            bpj_wavelength, ranges[0] / c0,
                                            rp.fs * upsample, 1 / sdr[0].prf)
    cupy.cuda.Device().synchronize()

    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp = rp.pos(ts[-1])
        test = rtdata.get()
    plt.imshow(db(bpj_grid.get()), origin='lower')
    plt.axis('tight')
    cam.snap()
    # bpj_truedata.append(bpj_grid.get())

del pantx_gpu
del postx_gpu
del veltx_gpu
del rtdata
del upsample_data
del bpj_grid

del granges_gpu
del gvel_gpu

anim = cam.animate()

# bfig = px.scatter(x=gx.flatten(), y=gy.flatten(), color=db(bpj_truedata).flatten())
# bfig.add_scatter(x=rp.pos(rp.gpst)[0, :], y=rp.pos(rp.gpst)[1, :])
# bfig.show()

if test is not None:
    plt.figure('Doppler data')
    plt.imshow(np.fft.fftshift(db(np.fft.fft(test, axis=1)), axes=1), extent=[0, cpi_len, ranges[0], ranges[-1]])
    plt.axis('tight')

try:
    if (nbpj_pts[0] * nbpj_pts[1]) < 400**2:
        bg = SDREnvironment(sdr_file=sdr)
        bg.ref = origin
        cx, cy, cz = bg.getGrid(origin, grid_width, grid_height, nbpj_pts)

        fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
        fig.add_scatter3d(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), mode='markers')
        fig.show()
except:
    pass

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
# plt.show()