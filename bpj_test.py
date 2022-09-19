import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
from SDRParsing import SDRParse

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

# This is the file used to backproject data
bg_file = '/data5/SAR_DATA/2022/03112022/SAR_03112022_135955.sar'
# bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
upsample = 4
cpi_len = 64
plp = 0
debug = True
nbpj_pts = 600

print('Loading SDR file...')
sdr = SDRParse(bg_file, do_exact_matches=False, use_idx=False)
try:
    origin = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                          getElevation((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'])))
except TypeError:
    '''heading = -np.arctan2(sdr.gps_data['ve'].values[0], sdr.gps_data['vn'].values[0])
    hght = sdr.xml['Flight_Line']['Flight_Line_Altitude_M']
    pt = ((sdr.xml['Flight_Line']['Start_Latitude_D'] + sdr.xml['Flight_Line']['Stop_Latitude_D']) / 2,
          (sdr.xml['Flight_Line']['Start_Longitude_D'] + sdr.xml['Flight_Line']['Stop_Longitude_D']) / 2)
    alt = getElevation(pt)
    mrange = hght / np.tan(sdr.ant[0].dep_ang)
    origin = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                    (pt[0], pt[1], alt))'''
    origin = (40.032203, -111.811561, 1525)
ref_llh = origin

# Generate a platform
print('Generating platform...', end='')

# Bunch of debug files used for testing. These are not necessary for backprojection.
'''gps_debug = '/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Channel_1_X-Band_9_GHz_VV_postCorrectionsGPSData.dat'
gimbal_debug = '/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Gimbal.dat'
postCorr = loadPostCorrectionsGPSData(gps_debug)
rawGPS = loadGPSData('/home/jeff/repo/Debug/03112022/SAR_03112022_135854_GPSDataPostJumpCorrection.dat')
preCorr = loadPreCorrectionsGPSData('/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Channel_1_X-Band_9_GHz_VV_preCorrectionsGPSData.dat')'''
rp = SDRPlatform(sdr, ref_llh)

# Get reference data
# flight = rp.pos(postCorr['sec'])
fs = sdr[0].fs
bwidth = sdr[0].bw
fc = sdr[0].fc
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
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample

# Chirp and matched filter calculations
if sdr[0].xml['Offset_Video_Enabled'] == 'True':
    offset_hz = sdr[0].xml['DC_Offset_MHz'] * 1e6
else:
    offset_hz = 0
offset_shift = int(offset_hz / (1 / fft_len * fs))
taywin = int(sdr[0].bw / fs * fft_len)
taywin = taywin + 1 if taywin % 2 != 0 else taywin
taytay = taylor(taywin)
tayd = np.fft.fftshift(taylor(cpi_len))
taydopp = np.fft.fftshift(np.ones((nsam * upsample, 1)).dot(tayd.reshape(1, -1)), axes=1)
# chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, rp.fs, fc,
#                             bwidth) * 1e4, up_fft_len)
chirp = np.fft.fft(np.mean(sdr.getPulses(np.arange(200), 0, is_cal=True), axis=1), fft_len)
mfilt = chirp.conj()
mfilt[:taywin // 2 + offset_shift] *= taytay[taywin // 2 - offset_shift:]
mfilt[-taywin // 2 + offset_shift:] *= taytay[:taywin // 2 - offset_shift]
mfilt[taywin // 2 + offset_shift:-taywin // 2 + offset_shift] = 0
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
rbins_gpu = cupy.array(ranges, dtype=np.float64)

# Calculate out points on the ground
shift_x, shift_y, _ = llh2enu(*origin, ref_llh)
gx, gy = np.meshgrid(np.linspace(-150, 150, nbpj_pts), np.linspace(-150, 150, nbpj_pts))
gx += shift_x
gy += shift_y
latg, long, altg = enu2llh(gx.flatten(), gy.flatten(), np.zeros(gx.flatten().shape[0]), ref_llh)
gz = (getElevationMap(latg, long) - ref_llh[2]).reshape(gx.shape)
gx_gpu = cupy.array(gx, dtype=np.float64)
gy_gpu = cupy.array(gy, dtype=np.float64)
gz_gpu = cupy.array(gz, dtype=np.float64)

if debug:
    # pts_debug = cupy.zeros((triangles.shape[0], 3), dtype=np.float64)
    # angs_debug = cupy.zeros((triangles.shape[0], 3), dtype=np.float64)
    pts_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
    angs_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)
test = None

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, nbpj_pts // threads_per_block[0] + 1), nbpj_pts // threads_per_block[1] + 1)
# rng_states = create_xoroshiro128p_states(triangles.shape[0], seed=10)

# Data blocks for imaging
bpj_truedata = np.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

# Run through loop to get data simulated
data_t = sdr[0].pulse_time
idx_t = np.arange(len(data_t))
print('Simulating...')
pulse_pos = 0
for tidx in tqdm([idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)]):
    ts = data_t[tidx]
    tmp_len = len(ts)
    # att = rp.att(ts)
    panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
    postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)
    data_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    data_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    bpj_grid = cupy.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

    # Reset the grid for truth data
    rtdata = cupy.fft.fft(cupy.array(sdr.getPulses(tidx, 0),
                                     dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, :tmp_len]
    upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * upsample, :]
    cupy.cuda.Device().synchronize()

    backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                            elrx_gpu,
                                            panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                            c0 / (fc - bwidth / 2 - offset_hz), ranges[0] / c0,
                                            rp.fs * upsample, bwidth, rp.az_half_bw, rp.el_half_bw, 0, pts_debug, angs_debug, debug)
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
del data_r
del data_i
del rtdata
del upsample_data
del bpj_grid

del rbins_gpu
del gx_gpu
del gy_gpu
del gz_gpu

'''tx_flight = rp.txpos(postCorr['sec'])
rx_flight = rp.rxpos(postCorr['sec'])

te, tn, tu = llh2enu(postCorr['tx_lat'], postCorr['tx_lon'], postCorr['tx_alt'], ref_llh)
re, rn, ru = llh2enu(postCorr['rx_lat'], postCorr['rx_lon'], postCorr['rx_alt'], ref_llh)
pe, pn, pu = llh2enu(preCorr['lat'], preCorr['lon'], preCorr['alt'], ref_llh)
e, n, u = llh2enu(rawGPS['lat'], rawGPS['lon'], rawGPS['alt'], ref_llh)
ee, nn, uu = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], ref_llh)
ffig = px.scatter_3d(x=flight[0, :], y=flight[1, :], z=flight[2, :])
ffig.add_scatter3d(x=tx_flight[0, :], y=tx_flight[1, :], z=tx_flight[2, :])
ffig.add_scatter3d(x=rx_flight[0, :], y=rx_flight[1, :], z=rx_flight[2, :])
ffig.add_scatter3d(x=te, y=tn, z=tu)
ffig.add_scatter3d(x=re, y=rn, z=ru)
ffig.show()'''

bfig = px.scatter(x=gx.flatten(), y=gy.flatten(), color=db(bpj_truedata).flatten())
bfig.show()

plt.figure('Doppler data')
plt.imshow(np.fft.fftshift(db(np.fft.fft(test, axis=1)), axes=1))
plt.axis('tight')

plt.figure('IMSHOW truedata')
plt.imshow(db(bpj_truedata), origin='lower')
plt.axis('tight')

'''n_diff = flight[1, :] - pn
e_diff = flight[0, :] - pe
u_diff = flight[2, :] - pu
plt.figure('diffs')
plt.plot(n_diff)
plt.plot(e_diff)
plt.plot(u_diff)

nn_diff = (np.interp(rawGPS['gps_ms'], sdr.gps_data.index, sdr.gps_data['lat']) - rawGPS['lat']) * postCorr['latConv']
ee_diff = (np.interp(rawGPS['gps_ms'], sdr.gps_data.index, sdr.gps_data['lon']) - rawGPS['lon']) * postCorr['lonConv']
uu_diff = (np.interp(rawGPS['gps_ms'], sdr.gps_data.index, sdr.gps_data['alt']) - rawGPS['alt'])
plt.figure('rawdiffs')
plt.plot(nn_diff)
plt.plot(ee_diff)
plt.plot(uu_diff)

se, sn, su = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], ref_llh)
plt.figure('enu')
plt.subplot(2, 2, 1)
plt.title('e')
plt.plot(sdr.gps_data.index, se)
plt.plot(postCorr['sec'], flight[0, :])
plt.plot(preCorr['sec'], pe)
plt.plot(rawGPS['gps_ms'], e)
plt.legend(['sdr', 'interp_sdr', 'pre', 'raw'])
plt.subplot(2, 2, 2)
plt.title('n')
plt.plot(sdr.gps_data.index, sn)
plt.plot(postCorr['sec'], flight[1, :])
plt.plot(preCorr['sec'], pn)
plt.plot(rawGPS['gps_ms'], n)
plt.subplot(2, 2, 3)
plt.title('u')
plt.plot(sdr.gps_data.index, su)
plt.plot(postCorr['sec'], flight[2, :])
plt.plot(preCorr['sec'], pu)
plt.plot(rawGPS['gps_ms'], u)

plt.figure('rpy')
plt.subplot(2, 2, 1)
plt.title('r')
plt.plot(sdr.gps_data.index, sdr.gps_data['r'])
plt.plot(preCorr['sec'], preCorr['r'])
plt.plot(rawGPS['gps_ms'], rawGPS['r'])
plt.subplot(2, 2, 2)
plt.title('p')
plt.plot(sdr.gps_data.index, sdr.gps_data['p'])
plt.plot(preCorr['sec'], preCorr['p'])
plt.plot(rawGPS['gps_ms'], rawGPS['p'])
plt.subplot(2, 2, 3)
plt.title('y')
plt.plot(sdr.gps_data.index, sdr.gps_data['y'] - np.pi * 2)
plt.plot(postCorr['sec'], postCorr['az'])
plt.plot(preCorr['sec'], preCorr['az'])
plt.legend(['sdr', 'interp_sdr', 'pre', 'raw'])

gimbal_data = loadGimbalData(gimbal_debug)
times = np.interp(gimbal_data['systime'], sdr.gps_data['systime'], sdr.gps_data.index)
plt.figure('Gimbal')
plt.subplot(2, 1, 1)
plt.title('Pan')
plt.plot(times, gimbal_data['pan'])
plt.plot(times, sdr.gimbal['pan'])
plt.subplot(2, 1, 2)
plt.title('Tilt')
plt.plot(times, gimbal_data['tilt'])
plt.plot(times, sdr.gimbal['tilt'])'''
plt.show()