import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
from keras.models import load_model
from keras.utils import custom_object_scope
from tensorflow.keras.optimizers import Adam, Adadelta
from wave_train import genTargetPSD, genModel, opt_loss
import open3d as o3d
from scipy.interpolate import CubicSpline, interpn
from scipy.signal import medfilt2d
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
from SDRParsing import SDRParse

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


def compileWaveformData(sdr_f, fft_sz, l_sz, bandwidth, fc, slant_min, slant_max, fs, bin_bw):
    target_data = np.zeros((l_sz, fft_sz, 1))
    for n in range(l_sz):
        tpsd = db(genTargetPSD(bandwidth, fc, slant_min, slant_max,
                               fft_sz, fs)).reshape((fft_sz, 1))
        tpsd = tpsd / np.linalg.norm(tpsd)
        target_data[n, :, :] = tpsd

    td_mu = np.mean(target_data)
    td_std = np.std(target_data)
    target_data = (np.fft.fftshift(target_data, axes=1)[:, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2,
                   :] - td_mu) / td_std

    # Generate the Clutter PSDs from the SDR file
    clutter_data = sdr_f.getPulses(np.arange(0, l_sz), 0)
    cd_fftdata = np.fft.fft(clutter_data, fft_sz, axis=0)
    clutter_data = db(cd_fftdata)
    clutter_data = clutter_data / np.linalg.norm(clutter_data, axis=0)
    clutter_phase_data = np.angle(cd_fftdata)

    cd_mu = np.mean(clutter_data)
    cd_std = np.std(clutter_data)
    cdp_mu = np.mean(clutter_phase_data)
    cdp_std = np.std(clutter_phase_data)
    clutter_data = (clutter_data - cd_mu) / cd_std
    clutter_phase_data = (clutter_phase_data - cdp_mu) / cdp_std

    clutter_data = np.fft.fftshift(clutter_data, axes=0)[fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2,
                   :].reshape(target_data.shape)
    clutter_phase_data = np.fft.fftshift(clutter_phase_data, axes=0)[
                         fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2, :].reshape(target_data.shape)
    return target_data, clutter_data, clutter_phase_data, cd_mu, cd_std


def getWaveFromData(mdl, target_data, clutter_data, clutter_phase_data, cd_mu, cd_std, l_sz, n_ants, fft_sz, bin_bw):
    wave_output = (mdl.predict((clutter_data, target_data, clutter_phase_data)) * cd_mu) + cd_std

    waveforms = np.zeros((l_sz, n_ants, fft_sz))
    waveforms[:, :, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2] = db(
        wave_output[:, :, :bin_bw] + 1j * wave_output[:, :, -bin_bw:])
    waveforms = waveforms / np.linalg.norm(waveforms, axis=2)[:, :, None]
    waveforms = waveforms[:, :, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2]
    return waveforms


def get_autocorr(wave, bw, fs, fft_len, offset_shift, upsample, sec_wave=None):
    taywin = int(bw / fs * fft_len)
    taywin = taywin + 1 if taywin % 2 != 0 else taywin
    taytay = taylor(taywin)
    chirp = np.fft.fft(wave, fft_len)
    if sec_wave is None:
        mfilt = chirp.conj()
    else:
        mfilt = np.fft.fft(sec_wave, fft_len)
    mfilt[:taywin // 2 + offset_shift] *= taytay[taywin // 2 - offset_shift:]
    mfilt[-taywin // 2 + offset_shift:] *= taytay[:taywin // 2 - offset_shift]
    mfilt[taywin // 2 + offset_shift:-taywin // 2 + offset_shift] = 0

    mfilt_chirp = mfilt * chirp
    upsampled_shift = np.zeros((fft_len * upsample,), dtype=np.complex128)

    upsampled_shift[:fft_len // 2] = mfilt_chirp[:fft_len // 2]
    upsampled_shift[-fft_len // 2:] = mfilt_chirp[-fft_len // 2:]
    return db(np.fft.ifft(upsampled_shift))


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

bg_file = '/data5/SAR_DATA/2022/03112022/SAR_03112022_135955.sar'
# bg_file = '/data5/SAR_DATA/2022/06152022/SAR_06152022_145909.sar'
upsample = 1
cpi_len = 64
plp = 0
max_pts_per_tri = 1
debug = True
nbpj_pts = 600
grid_width = 100
grid_height = 100
do_truth_backproject = False
custom_waveform = False

print('Loading SDR file...')
sdr = SDRParse(bg_file, do_exact_matches=False)

# Generate the background for simulation
print('Generating environment...', end='')
# bg = MapEnvironment((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'], sdr.ash['geo']['hRef']), extent=(120, 120))
bg = SDREnvironment(sdr)
'''ng = np.zeros_like(bg.refgrid)
ng[ng.shape[0] // 2, ng.shape[1] // 2] = 100
ng[ng.shape[0] // 2 + 15, ng.shape[1] // 2 + 15] = 100
ng[ng.shape[0] // 2 - 15, ng.shape[1] // 2 - 15] = 100
ng[ng.shape[0] // 2 + 15, ng.shape[1] // 2 - 15] = 100
ng[ng.shape[0] // 2 - 15, ng.shape[1] // 2 + 15] = 100
bg._refgrid = ng'''
print('Done.')

# Generate a platform
print('Generating platform...', end='')
'''gps_debug = '/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Channel_1_X-Band_9_GHz_VV_postCorrectionsGPSData.dat'
gimbal_debug = '/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Gimbal.dat'
postCorr = loadPostCorrectionsGPSData(gps_debug)
rawGPS = loadGPSData('/home/jeff/repo/Debug/03112022/SAR_03112022_135854_GPSDataPostJumpCorrection.dat')
preCorr = loadPreCorrectionsGPSData('/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Channel_1_X-Band_9_GHz_VV_preCorrectionsGPSData.dat')'''
rp = SDRPlatform(sdr, bg.ref)

# Get reference data
# flight = rp.pos(postCorr['sec'])
fc = sdr[0].fc
fs = sdr[0].fs
bwidth = sdr[0].bw
print('Done.')

# Generate a backprojected image
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

# Model for waveform simulation
if custom_waveform:
    mdl_fft = 2048
    n_conv_filters = 4
    kernel_sz = 128
    mdl_bin_bw = int(1500e6 // (fs / mdl_fft))
    mdl = genModel(mdl_bin_bw, n_conv_filters, kernel_sz, mdl_fft)
    mdl.load_weights('/home/jeff/repo/apache_mla/model/weights.h5')

    # Generate model data for waveform
    wfd = compileWaveformData(sdr, mdl_fft, cpi_len, bwidth, fc, ranges[0], ranges[-1], fs, mdl_bin_bw)

    # Get waveforms
    wf_data = getWaveFromData(mdl, *wfd, cpi_len, 2, mdl_fft, mdl_bin_bw)
    chirp = np.fft.fft(np.fft.ifft(wf_data[0, 0, :]), fft_len)
else:
    chirp = np.fft.fft(np.mean(sdr.getPulses(sdr[0].cal_num, 0, is_cal=True), axis=1), fft_len)

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
mfilt = chirp.conj()
mfilt[:taywin // 2 + offset_shift] *= taytay[taywin // 2 - offset_shift:]
mfilt[-taywin // 2 + offset_shift:] *= taytay[:taywin // 2 - offset_shift]
mfilt[taywin // 2 + offset_shift:-taywin // 2 + offset_shift] = 0
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)
mfilt_gpu = cupy.array(np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)

# Calculate out points on the ground
local_grid = bg.createGrid(bg.origin, grid_width, grid_height, (nbpj_pts, nbpj_pts), bg.heading)
newgrid = interpn((np.arange(bg.refgrid.shape[0]) - bg.refgrid.shape[0] / 2,
                   np.arange(bg.refgrid.shape[1]) - bg.refgrid.shape[1] / 2),
                  bg.refgrid, np.array([local_grid[0, :, :].flatten(), localgrid[1, :, :].flatten()]).T).reshape((nbpj_pts, nbpj_pts))
bg.setGrid(newgrid, local_grid, grid_width / nbpj_pts, grid_height / nbpj_pts)

gx_gpu = cupy.array(local_grid[0, :, :], dtype=np.float64)
gy_gpu = cupy.array(local_grid[1, :, :], dtype=np.float64)
gz_gpu = cupy.array(local_grid[2, :, :], dtype=np.float64)

# Generate a test strip of data
grid_gpu = cupy.array(bg.grid, dtype=np.float64)
ref_coef_gpu = cupy.array(bg.refgrid, dtype=np.float64)
rbins_gpu = cupy.array(ranges, dtype=np.float64)

if debug:
    pts_debug = cupy.zeros((bg.refgrid.shape[0], 3), dtype=np.float64)
    angs_debug = cupy.zeros((bg.refgrid.shape[0], 3), dtype=np.float64)
    # pts_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
    # angs_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)
test = None

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_ranges = (bg.refgrid.shape[0] // threads_per_block[0] + 1,
              bg.refgrid.shape[1] // threads_per_block[1] + 1
              )
bpg_bpj = (max(1, nbpj_pts // threads_per_block[0] + 1), nbpj_pts // threads_per_block[1] + 1)

# Data blocks for imaging
bpj_res = np.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)
bpj_truedata = np.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

rng_states = create_xoroshiro128p_states(bpg_ranges[0] * bpg_ranges[1] * threads_per_block[0] * threads_per_block[1],
                                         seed=10)

# Run through loop to get data simulated
data_t = np.unique(sdr[0].pulse_time)
# data_t = rp.gpst.mean() + np.arange(-cpi_len // 2, cpi_len // 2) * (data_t[1] - data_t[0])
idx_t = np.arange(len(data_t))
print('Simulating...')
pulse_pos = 0
for tidx in tqdm([idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)]):
    init_xoroshiro128p_states(rng_states, seed=10)
    ts = data_t[tidx]
    tmp_len = len(ts)
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        debug_flag = debug
    else:
        debug_flag = False
    panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
    postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)
    data_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    data_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)
    bpj_grid = cupy.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)
    genRangeWithoutIntersection[bpg_bpj, threads_per_block](grid_gpu, ref_coef_gpu,
                                                               postx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                               panrx_gpu, elrx_gpu, data_r, data_i, pts_debug,
                                                               angs_debug,
                                                               c0 / fc, ranges[0] / c0,
                                                               rp.fs * upsample, rp.az_half_bw, rp.el_half_bw,
                                                               max_pts_per_tri, rng_states, debug_flag)

    cupy.cuda.Device().synchronize()

    rtdata = cupy.fft.fft(data_r + 1j * data_i, fft_len, axis=0) * chirp_gpu[:, :tmp_len] * mfilt_gpu[:, :tmp_len]
    upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rtdata = cupy.array(np.random.rand(nsam * upsample, tmp_len), dtype=np.complex128) + \
             cupy.fft.ifft(upsample_data, axis=0)[:nsam * upsample, :]
    cupy.cuda.Device().synchronize()

    # Simulated data debug checks
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp = rp.pos(ts[-1])
        test = rtdata.get()
        angd = angs_debug.get()
        locd = pts_debug.get()

    backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                            elrx_gpu, panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                            c0 / fc, ranges[0] / c0,
                                            rp.fs * upsample, bwidth, rp.az_half_bw, rp.el_half_bw, 0, pts_debug,
                                            angs_debug, debug_flag)
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp_bj = rp.pos(ts[-1])
        test_bj = rtdata.get()
        angd_bj = angs_debug.get()
        locd_bj = pts_debug.get()
    cupy.cuda.Device().synchronize()
    bpj_res += bpj_grid.get()

    # Reset the grid for truth data
    if do_truth_backproject:
        trtdata = cupy.fft.fft(cupy.array(sdr.getPulses(tidx, 0),
                                         dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, :tmp_len]
        tupsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
        tupsample_data[:fft_len // 2, :] = trtdata[:fft_len // 2, :]
        tupsample_data[-fft_len // 2:, :] = trtdata[-fft_len // 2:, :]
        trtdata = cupy.fft.ifft(tupsample_data, axis=0)[:nsam * upsample, :]
        cupy.cuda.Device().synchronize()
        bpj_grid2 = cupy.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

        if ts[0] < rp.gpst.mean() <= ts[-1]:
            locp = rp.pos(ts[-1])
            test = rtdata.get()
            angd = angs_debug.get()
            locd = pts_debug.get()

        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                                elrx_gpu,
                                                panrx_gpu, elrx_gpu, trtdata, bpj_grid2,
                                                c0 / (fc - bwidth / 2 - offset_hz), ranges[0] / c0,
                                                rp.fs * upsample, bwidth, rp.az_half_bw, rp.el_half_bw, 0, pts_debug, angs_debug, debug)
        cupy.cuda.Device().synchronize()
        bpj_truedata += bpj_grid2.get()

del panrx_gpu
del postx_gpu
del posrx_gpu
del elrx_gpu
del data_r
del data_i
del rtdata
del upsample_data
del bpj_grid

if do_truth_backproject:
    del trtdata
    del tupsample_data
    del bpj_grid2

del rbins_gpu
del gx_gpu
del gy_gpu
del gz_gpu

# dfig = go.Figure(data=[go.Mesh3d(x=bg.grid[:, :, 0].ravel(), y=bg.grid[:, :, 1].ravel(), z=bg.grid[:, :, 2].ravel(),
#                                  facecolor=bg.refgrid.ravel(), facecolorsrc='teal')])
# flight = rp.pos(rp.gpst)
# dfig.add_scatter3d(x=flight[0, :], y=flight[1, :], z=flight[2, :], mode='markers')
dfig = px.scatter_3d(x=local_grid[0, :, :].flatten(), y=local_grid[1, :, :].flatten(), z=local_grid[2, :, :].flatten())
dfig.add_scatter3d(x=locd[:, 0] + locp[0], y=locd[:, 1] + locp[1], z=locd[:, 2] + locp[2], mode='markers')
dfig.show()

'''te, tn, tu = llh2enu(postCorr['tx_lat'], postCorr['tx_lon'], postCorr['tx_alt'], bg.ref)
re, rn, ru = llh2enu(postCorr['rx_lat'], postCorr['rx_lon'], postCorr['rx_alt'], bg.ref)
pe, pn, pu = llh2enu(preCorr['lat'], preCorr['lon'], preCorr['alt'], bg.ref)
e, n, u = llh2enu(rawGPS['lat'], rawGPS['lon'], rawGPS['alt'], bg.ref)
ee, nn, uu = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], bg.ref)
ffig = px.scatter_3d(x=flight[0, :], y=flight[1, :], z=flight[2, :])
ffig.add_scatter3d(x=tx_flight[0, :], y=tx_flight[1, :], z=tx_flight[2, :])
ffig.add_scatter3d(x=rx_flight[0, :], y=rx_flight[1, :], z=rx_flight[2, :])
ffig.add_scatter3d(x=te, y=tn, z=tu)
ffig.add_scatter3d(x=re, y=rn, z=ru)
ffig.show()'''

plt.figure('Doppler data')
plt.imshow(np.fft.fftshift(db(np.fft.fft(test, axis=1)), axes=1))
plt.axis('tight')

if do_truth_backproject:
    bfig = px.scatter(x=bg.grid[:, :, 0].flatten(), y=bg.grid[:, :, 1].flatten(), color=db(bpj_truedata).flatten())
    bfig.show()

    plt.figure('IMSHOW truedata')
    plt.imshow(db(bpj_truedata), origin='lower')
    plt.axis('tight')

plt.figure('BPJ BackGrid')
plt.imshow(bg.refgrid, origin='lower')
plt.axis('tight')

plt.figure('BPJ Grid')
plt.imshow(db(bpj_res), origin='lower')
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

se, sn, su = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], bg.ref)
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