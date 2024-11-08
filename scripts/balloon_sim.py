import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter, loadMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject, cpudiff
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, basinhopping
import cupy as cupy
from PIL import Image
import cupyx.scipy.signal
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states
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
from SDRWriting import SDRWrite

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
bg_file = '/data5/SAR_DATA/2020/02102020/SAR_Collects/SAR_02102020_091055.sar'
# bg_file = '/data5/SAR_DATA/2019/08072019/SAR_08072019_100120.sar'
output_fnme = f'/data6/SAR_DATA/Simulated_Balloon/balloon_sim_{bg_file.split("/")[-1].split(".")[0].split("_")[-1]}.sar'
upsample = 1
poly_num = 1
use_rcorr = False
rotate_grid = True
use_ecef = True
cpi_len = 32
plp = .5673
prf = 180.
fs = 500e6
presum = 9
attenuation = 31
n_cal_pulses = 50
pts_per_tri = 2
fc = 9600000000.00
bwidth = 240000000.00
el_bw = 10.
az_bw = 4.5
dep_ang = 45.
look_side = 'Right'
init_course_angle = 1.40129391
debug = True
debug_flag = False
m_per_pt = 400
grid_width = 80000
grid_height = 80000
channel = 0

nbpj_pts = (int(grid_width // m_per_pt), int(grid_height // m_per_pt))

# Generate a platform
print('Generating platform...', end='')

# Bunch of debug files used for testing. These are not necessary for backprojection.
gps_check = True
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

# Create platform
ref_llh = (43.246701, -93.787441, 397)
gimbal_offset = np.array([.1375, -.0565, -.2868])
gimbal_rotations = np.array([0.0, 0.0, 60.0])

# Get conversion factor for systime to gps time
e, n, u = llh2enu(rawGPS['lat'], rawGPS['lon'], rawGPS['alt'], ref_llh)
gd = np.array([rawGPS['gps_ms'],
               np.interp(rawGPS['systime'], gimbal_data['systime'], gimbal_data['pan']),
               np.interp(rawGPS['systime'], gimbal_data['systime'], gimbal_data['tilt'])]).T
rp = RadarPlatform(e=e, n=n, u=u, r=rawGPS['r'], p=rawGPS['p'],
                   y=np.angle(rawGPS['azimuthY'] + 1j * rawGPS['azimuthX']),
                   t=rawGPS['gps_ms'], tx_offset=None,
                   rx_offset=None, gimbal=gd[:, 1:], gimbal_offset=gimbal_offset, gimbal_rotations=gimbal_rotations,
                   dep_angle=dep_ang, squint_angle=0., az_bw=az_bw, el_bw=el_bw, fs=fs, gps_data=None, tx_num=0, rx_num=0,
                   wavenumber=0)

wr_obj = SDRWrite(output_fnme, 2000, rawGPS['gps_ms'], rawGPS['lat'], rawGPS['lon'], rawGPS['alt'], rawGPS['r'], rawGPS['p'],
                   np.angle(rawGPS['azimuthY'] + 1j * rawGPS['azimuthX']), rawGPS['vn'], rawGPS['ve'], rawGPS['vu'],
                  gd, settings_alt=rawGPS['alt'].mean(),
                  settings_vel=np.sqrt(rawGPS['vn'].mean()**2 + rawGPS['ve'].mean()**2 + rawGPS['vu'].mean()**2),
                  fs=rp.fs)

bg = MapEnvironment(ref_llh, (grid_width, grid_height), (int(grid_width // m_per_pt), int(grid_height // m_per_pt)))
ng = np.zeros(bg.shape)

ng = Image.open('/home/jeff/Pictures/josh.png').resize(bg.shape, Image.ANTIALIAS)
ng = np.linalg.norm(np.array(ng), axis=2) * 1e11
bg._refgrid = ng

# Atmospheric modeling params
Ns = 313
Nb = 66.65
hb = 12192
if rp.pos(rp.gpst)[2, :].mean() > hb:
    hb = rp.pos(rp.gpst)[2, :].mean() + 1000.
    Nb = 105 * np.exp(-(hb - 9000) / 7023)

# Generate values needed for backprojection
print('Calculating grid parameters...')
# General calculations for slant ranges, etc.
# plat_height = rp.pos(rp.gpst)[2, :].mean()
fdelay = rp.pos(rp.gpst)[2, :].mean()
nr = rp.calcPulseLength(fdelay, plp, use_tac=True) + 1
nsam = rp.calcNumSamples(fdelay, plp)
ranges = rp.calcRangeBins(fdelay, upsample, plp)
granges = ranges * np.cos(rp.dep_ang)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample
up_nsam = nsam * upsample

# Chirp and matched filter calculations
bpj_wavelength = c0 / (fc - bwidth / 2 - 5e6)
chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, rp.fs, fc, bwidth), fft_len) * 1e6
mfilt = chirp.conj()

wvf = np.fft.fft(chirp)[:nr].real
wr_obj.addChannel(fc, bwidth,
                  nr / rp.fs, prf / 1.5, (rp.dep_ang + rp.el_half_bw) / DTR,
                  (rp.dep_ang - rp.el_half_bw) / DTR, plp * 100,
                  presum, trans_on_tac=1000, att_mode='Static', rec_only=False, rec_slice='A', prf_broad=1.5,
                  fs=rp.fs, digital_channel=0, wavenumber=0,
                  adc_channel=0, receiver_slot=4,
                  receiver_channel=0, upconverter_slot=2,
                  select_89=True, band=2, dac_channel=0,
                  ddc_enable=0 if rp.fs == 2e9 else 1, filter_select=0,
                  rx_port=0, tx_port=0, polarization=1,
                  numconsmodes=0, awg_enable=0,
                  rf_ref_wavenumber=0,
                  offset_video=5e6)

calpulse = np.zeros(nsam, dtype=np.complex128)
calpulse[5:nr + 5] += np.fft.ifft(chirp)[:nr]
init_pulse_time = rawGPS['gps_ms'][0] + n_cal_pulses / prf

for fr in tqdm(np.arange(n_cal_pulses)):
    wr_obj.writePulse(init_pulse_time - (n_cal_pulses - fr) * (nsam / rp.fs), 31, calpulse, 0, True)

wr_obj.addAntenna(0, rp.az_half_bw * 2 / DTR, rp.el_half_bw * 2 / DTR, rp.az_half_bw * 2 / DTR, rp.dep_ang / DTR,
                  1, 1, False, 0)

wr_obj.addPort(0, 0., 0., .0656, cable_length=0., max_gain=25, peak_transmit_power=100)

wr_obj.addGimbal(70., 30., rp.dep_ang / DTR, rp.gimbal_offset, rp.gimbal_rotations, init_course_angle,
                 squint=0.0, update_rate=100, look_side=look_side)

ref_coef_gpu = cupy.array(bg.refgrid, dtype=np.float64)
rbins_gpu = cupy.array(ranges, dtype=np.float64)
rmat_gpu = cupy.array(bg.transforms[0], dtype=np.float64)
shift_gpu = cupy.array(bg.transforms[1], dtype=np.float64)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)

# Calculate out points on the ground
gx, gy, gz = bg.getGrid()
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
bpg_ranges = (bg.refgrid.shape[0] // threads_per_block[0] + 1,
              bg.refgrid.shape[1] // threads_per_block[1] + 1)

rng_states = create_xoroshiro128p_states(bpg_ranges[0] * bpg_ranges[1] * threads_per_block[0] * threads_per_block[1],
                                         seed=10)

# Data blocks for imaging
bpj_truedata = np.zeros(nbpj_pts, dtype=np.complex128)

# Run through loop to get data simulated
idx_t = np.arange(int((rawGPS['gps_ms'][-5] - init_pulse_time) * prf) + 1)
data_t = idx_t / prf + init_pulse_time
# data_t = preCorr['sec'][::10][:-20]
# idx_t = np.arange(len(data_t))

test = None
print('Backprojecting...')
for tidx, frames in tqdm(enumerate(idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)),
                         total=len(data_t) // cpi_len + 1):
    ts = data_t[tidx * cpi_len + np.arange(len(frames))]
    tmp_len = len(ts)

    '''if not np.all(cpudiff(np.arctan2(-rp.pos(ts)[1, :], rp.pos(ts)[0, :]), rp.pan(ts)) -
                  rp.az_half_bw < 0):
        continue'''
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

    # Simulate data based on background
    data_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    data_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    genRangeWithoutIntersection[bpg_ranges, threads_per_block](rmat_gpu, shift_gpu, gz_gpu, ref_coef_gpu,
                                                               postx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                               panrx_gpu, elrx_gpu, data_r, data_i, rng_states,
                                                               pts_debug, angs_debug,
                                                               bpj_wavelength,
                                                               ranges[0] / c0,
                                                               rp.fs,
                                                               rp.az_half_bw,
                                                               rp.el_half_bw, pts_per_tri,
                                                               debug_flag)
    cuda.synchronize()
    data_r[np.isnan(data_r)] = 0
    data_i[np.isnan(data_i)] = 0

    rcdata = cupy.fft.fft(data_r + 1j * data_i, fft_len, axis=0) * chirp_gpu[:, :tmp_len]

    rt_cpu = cupy.fft.ifft(rcdata, axis=0)[:nsam, :].get()
    print(rt_cpu.max())
    for fr in np.arange(tmp_len):
        wr_obj.writePulse(ts[fr], attenuation, rt_cpu[:, fr], 0, False)
    rtdata = rcdata * mfilt_gpu[:, :tmp_len]
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
del rcdata
del upsample_data
del bpj_grid
del r_corr_gpu
del shift_gpu
del chirp_gpu
del rmat_gpu
del ref_coef_gpu
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

# Write the XML
print('Finalizing XML...')
wr_obj.finalize(prf * presum)

from sklearn.preprocessing import power_transform

plot_data = np.fliplr(db(bpj_truedata))
scaled_data = power_transform(db(bpj_truedata).reshape(-1, 1)).reshape(plot_data.shape)

# px.imshow(plot_data, color_continuous_scale=px.colors.sequential.gray, zmin=130, zmax=180).show()
px.imshow(np.fliplr(scaled_data), color_continuous_scale=px.colors.sequential.gray, zmin=-1, zmax=3).show()

# Get IPR cut
db_bpj = db(bpj_truedata)
mx = np.where(db_bpj == db_bpj.max())
ipr_gridsz = min(db_bpj.shape[0] - mx[0][0], mx[0][0], db_bpj.shape[1] - mx[1][0], mx[1][0])

cutfig = make_subplots(rows=2, cols=1, subplot_titles=('Azimuth', 'Range'))
cutfig.append_trace(go.Scatter(x=np.arange(ipr_gridsz * 2) * grid_width / nbpj_pts[0] - ipr_gridsz * grid_width / nbpj_pts[0],
                               y=db_bpj[mx[0][0], mx[1][0] - ipr_gridsz:mx[1][0] + ipr_gridsz],
                               mode='lines'), row=1, col=1)
cutfig.append_trace(
    go.Scatter(x=np.arange(ipr_gridsz * 2) * grid_height / nbpj_pts[1] - ipr_gridsz * grid_height / nbpj_pts[1],
               y=db_bpj[mx[0][0] - ipr_gridsz:mx[0][0] + ipr_gridsz, mx[1][0]],
               mode='lines'), row=2, col=1)
cutfig.update_yaxes(range=[120, 160])
cutfig.show()

if gps_check:
    sdr = SDRParse(output_fnme)
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
    gnrx = postCorr['rx_lat'] * postCorr['latConv'] - bg.origin[0] * postCorr['latConv']
    gerx = postCorr['rx_lon'] * postCorr['lonConv'] - bg.origin[1] * postCorr['lonConv']
    gurx = postCorr['rx_alt'] - bg.origin[2]
    gerx, gnrx, gurx = llh2enu(postCorr['rx_lat'], postCorr['rx_lon'], postCorr['rx_alt'], bg.origin)
    retx = rp.txpos(postCorr['sec'])[0, :]
    rntx = rp.txpos(postCorr['sec'])[1, :]
    rutx = rp.txpos(postCorr['sec'])[2, :]
    gntx = postCorr['tx_lat'] * postCorr['latConv'] - bg.origin[0] * postCorr['latConv']
    getx = postCorr['tx_lon'] * postCorr['lonConv'] - bg.origin[1] * postCorr['lonConv']
    gutx = postCorr['tx_alt'] - bg.origin[2]
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

    plt.figure('Matched Filter')
    plt.plot(db(loadMatchedFilter('/home/jeff/repo/Debug/balloon_sim_Channel_1_MatchedFilter.dat')))
    plt.plot(db(loadMatchedFilter('/home/jeff/repo/Debug/balloon_sim_Channel_1_ReferenceChirp.dat')))
plt.show()