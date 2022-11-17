import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
from keras.models import load_model
from keras.utils import custom_object_scope
from tensorflow.keras.optimizers import Adam, Adadelta
from wave_train import genTargetPSD, genModel, opt_loss
import open3d as o3d
from scipy.interpolate import CubicSpline, interpn
from PIL import Image
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
from SDRParsing import SDRParse, load

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
mempool = cupy.get_default_memory_pool()


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
    mfilt = chirp.conj() if sec_wave is None else np.fft.fft(sec_wave, fft_len)
    mfilt[:taywin // 2 + offset_shift] *= taytay[taywin // 2 - offset_shift:]
    mfilt[-taywin // 2 + offset_shift:] *= taytay[:taywin // 2 - offset_shift]
    mfilt[taywin // 2 + offset_shift:-taywin // 2 + offset_shift] = 0

    mfilt_chirp = mfilt * chirp
    upsampled_shift = np.zeros((up_nsam,), dtype=np.complex128)

    upsampled_shift[:fft_len // 2] = mfilt_chirp[:fft_len // 2]
    upsampled_shift[-fft_len // 2:] = mfilt_chirp[-fft_len // 2:]
    return db(np.fft.ifft(upsampled_shift))


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

bg_file = '/data5/SAR_DATA/2022/03112022/SAR_03112022_135955.sar'
# bg_file = '/data5/SAR_DATA/2022/03282022/SAR_03282022_082824.sar'
upsample = 1
channel = 0
cpi_len = 128
plp = 0
max_pts_per_tri = 1
debug = True
nbpj_pts = 200
grid_width = 100
grid_height = 100
do_truth_backproject = False
custom_waveform = False

print('Loading SDR file...')
sdr = load(bg_file, export_pickle=False)

# Generate the background for simulation
print('Generating environment...', end='')
# bg = MapEnvironment((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'], sdr.ash['geo']['hRef']), extent=(120, 120))
bg = SDREnvironment(sdr)

print('Done.')

# Generate a platform
print('Generating platform...', end='')
'''gps_debug = '/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Channel_1_X-Band_9_GHz_VV_postCorrectionsGPSData.dat'
gimbal_debug = '/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Gimbal.dat'
postCorr = loadPostCorrectionsGPSData(gps_debug)
rawGPS = loadGPSData('/home/jeff/repo/Debug/03112022/SAR_03112022_135854_GPSDataPostJumpCorrection.dat')
preCorr = loadPreCorrectionsGPSData('/home/jeff/repo/Debug/03112022/SAR_03112022_135854_Channel_1_X-Band_9_GHz_VV_preCorrectionsGPSData.dat')'''
rp = SDRPlatform(sdr, bg.ref, channel=channel)

# Get reference data
# flight = rp.pos(postCorr['sec'])
fc = sdr[channel].fc
fs = sdr[channel].fs
bwidth = sdr[channel].bw
print('Done.')

# Generate a backprojected image
print('Calculating grid parameters...')
# General calculations for slant ranges, etc.
# plat_height = rp.pos(rp.gpst)[2, :].mean()
fdelay = 5
nr = rp.calcPulseLength(fdelay, plp, use_tac=True)
nsam = rp.calcNumSamples(fdelay, plp)
ranges = rp.calcRangeBins(fdelay, upsample, plp)
granges = ranges * np.cos(rp.dep_ang)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample
up_nsam = nsam * upsample

# Model for waveform simulation
if custom_waveform:
    '''mdl_fft = 2048
    n_conv_filters = 4
    kernel_sz = 128
    mdl_bin_bw = int(1500e6 // (fs / mdl_fft))
    mdl = genModel(mdl_bin_bw, n_conv_filters, kernel_sz, mdl_fft)
    mdl.load_weights('/home/jeff/repo/apache_mla/model/weights.h5')

    # Generate model data for waveform
    wfd = compileWaveformData(sdr, mdl_fft, cpi_len, bwidth, fc, ranges[0], ranges[-1], fs, mdl_bin_bw)

    # Get waveforms
    wf_data = getWaveFromData(mdl, *wfd, cpi_len, 2, mdl_fft, mdl_bin_bw)
    chirp = np.fft.fft(np.fft.ifft(wf_data[0, 0, :]), fft_len)'''
    chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10),
                                nr, fs, sdr[channel].baseband_fc, bwidth), fft_len)
else:
    mchirp = np.mean(sdr.getPulses(sdr[channel].cal_num, 0, is_cal=True), axis=1)
    chirp = np.fft.fft(mchirp, up_fft_len)
    chirp /= (10**(np.floor(np.log10(abs(chirp).max()))))

# Chirp and matched filter calculations
if sdr[channel].xml['Offset_Video_Enabled'] == 'True':
    offset_hz = sdr[channel].xml['DC_Offset_MHz'] * 1e6
    bpj_wavelength = c0 / (fc - bwidth / 2 - offset_hz)
    offset_shift = int((offset_hz + bwidth / 2) / (1 / fft_len * fs) * upsample)
else:
    offset_hz = 0
    offset_shift = int(offset_hz / (1 / fft_len * fs) * upsample)
    bpj_wavelength = c0 / fc
taywin = int(sdr[channel].bw / fs * up_fft_len)
taywin = taywin + 1 if taywin % 2 != 0 else taywin
taytay = taylor(taywin)
# tayd = np.fft.fftshift(taylor(cpi_len))
# taydopp = np.fft.fftshift(np.ones((up_nsam, 1)).dot(tayd.reshape(1, -1)), axes=1)
mfilt = chirp.conj()
if offset_shift - taywin // 2 < 0:
    mfilt[:taywin // 2 + offset_shift] *= taytay[-(taywin // 2 + offset_shift):]
    mfilt[taywin - (taywin // 2 + offset_shift):] *= taytay[:taywin - (taywin // 2 + offset_shift)]
    mfilt[taywin // 2 + offset_shift:-taywin // 2 + offset_shift] = 0
else:
    mfilt[offset_shift - taywin // 2:offset_shift + taywin // 2] *= taytay
    mfilt[:offset_shift - taywin // 2] = 0
    mfilt[offset_shift + taywin // 2:] = 0

if do_truth_backproject:
    mfilt_gpu = cupy.array(np.tile(mfilt[::upsample], (cpi_len, 1)).T, dtype=np.complex128)

sim_chirp = chirp * mfilt

# autocorr_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T * np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
autocorr_gpu = cupy.array(np.tile(sim_chirp, (cpi_len, 1)).T, dtype=np.complex128)

bg.resample(bg.origin, grid_width + 10, grid_height + 10, (nbpj_pts * 4, nbpj_pts * 4), bg.heading)
gx, gy, gz = bg.getGrid(bg.origin, grid_width, grid_height, (nbpj_pts, nbpj_pts))
# gx, gy, gz = bg.getGrid()
ngz_gpu = cupy.array(bg.getGrid()[2], dtype=np.float64)
ng = np.zeros(bg.shape)
# ng = bg.grid_function(gx.flatten(), gy.flatten()).reshape(gx.shape)
pts_dist = np.linalg.norm(np.array([n for n in bg.getGrid()]) - np.array(llh2enu(*bg.origin, bg.ref))[:, None, None],
                          axis=0)
ng[np.where(pts_dist < 10)] = 1
ng[ng.shape[0] // 2, ng.shape[1] // 2] = 1
# ng[ng.shape[0] // 2 - 15, :] = 1
# ng[:, ng.shape[1] // 2 - 15] = 1
# ng[::15, ::15] = 1

# ng = Image.open('/home/jeff/Downloads/artemislogo.png').resize(bg.shape, Image.ANTIALIAS)
# ng = np.linalg.norm(np.array(ng), axis=2)
bg._refgrid = ng

bgx_gpu = cupy.array(gx, dtype=np.float64)
bgy_gpu = cupy.array(gy, dtype=np.float64)
bgz_gpu = cupy.array(gz, dtype=np.float64)

# Generate a test strip of data
ref_coef_gpu = cupy.array(bg.refgrid, dtype=np.float64)
rbins_gpu = cupy.array(ranges, dtype=np.float64)
rmat_gpu = cupy.array(bg.transforms[0], dtype=np.float64)
shift_gpu = cupy.array(bg.transforms[1], dtype=np.float64)

if debug:
    pts_debug = cupy.zeros((3, *bg.shape), dtype=np.float64)
    angs_debug = cupy.zeros((3, *bg.shape), dtype=np.float64)
    # pts_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
    # angs_debug = cupy.zeros((nbpj_pts, 3), dtype=np.float64)
else:
    pts_debug = cupy.zeros((1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1), dtype=np.float64)
test = None

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_ranges = (bg.refgrid.shape[0] // threads_per_block[0] + 1,
              bg.refgrid.shape[1] // threads_per_block[1] + 1)
bpg_bpj = (nbpj_pts // threads_per_block[0] + 1,
           nbpj_pts // threads_per_block[1] + 1)

# Data blocks for imaging
bpj_res = np.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)
bpj_truedata = np.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

rng_states = create_xoroshiro128p_states(bpg_ranges[0] * bpg_ranges[1] * threads_per_block[0] * threads_per_block[1],
                                         seed=10)

# Run through loop to get data simulated
data_t = sdr[channel].pulse_time
# data_t = rp.gpst.mean() + np.arange(-cpi_len // 2, cpi_len // 2) * (data_t[1] - data_t[0])
idx_t = sdr[channel].frame_num
print('Simulating...')
pulse_pos = 0
for tidx, frames in tqdm(enumerate(idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len))):
    init_xoroshiro128p_states(rng_states, seed=10)
    # data = sdr.getPulses(sdr[0].frame_num[tidx], 0)
    ts = data_t[tidx * cpi_len + np.arange(len(frames))]
    tmp_len = len(ts)
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        debug_flag = debug
    else:
        debug_flag = False
    panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
    posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
    postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)
    data_r = cupy.random.randn(up_nsam, tmp_len, dtype=np.float64) * 1e-6
    data_i = cupy.random.randn(up_nsam, tmp_len, dtype=np.float64) * 1e-6
    genRangeWithoutIntersection[bpg_ranges, threads_per_block](rmat_gpu, shift_gpu, ngz_gpu, ref_coef_gpu,
                                                               postx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                               panrx_gpu, elrx_gpu, data_r, data_i, rng_states,
                                                               pts_debug, angs_debug, bpj_wavelength,
                                                               ranges[0] / c0, rp.fs * upsample, rp.az_half_bw,
                                                               rp.el_half_bw, max_pts_per_tri, debug_flag)
    cuda.synchronize()

    rtdata = cupy.fft.fft(data_r + 1j * data_i, up_fft_len, axis=0) * autocorr_gpu[:, :tmp_len] # * mfilt_gpu[:, :tmp_len]
    # upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
    # upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
    # upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
    rcdata = cupy.fft.ifft(rtdata, axis=0)[:up_nsam, :]
    cuda.synchronize()

    # Simulated data debug checks
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp = rp.rxpos(ts[0])
        test = rcdata.get()
        angd = angs_debug.get()
        locd = pts_debug.get()

    cuda.synchronize()

    bpj_grid = cupy.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)
    backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, bgx_gpu, bgy_gpu, bgz_gpu, rbins_gpu, panrx_gpu,
                                            elrx_gpu, panrx_gpu, elrx_gpu, rcdata, bpj_grid,
                                            bpj_wavelength, ranges[0] / c0,
                                            rp.fs * upsample, bwidth, rp.az_half_bw, rp.el_half_bw, 0, pts_debug,
                                            angs_debug, debug_flag)

    cuda.synchronize()
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        locp_bj = rp.rxpos(ts[-1])
        test_bj = rcdata.get()
        angd_bj = angs_debug.get()
        locd_bj = pts_debug.get()
    cuda.synchronize()
    bpj_res += bpj_grid.get()

    # Reset the grid for truth data
    if do_truth_backproject:
        trtdata = cupy.fft.fft(cupy.array(sdr.getPulses(frames, 0),
                                         dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, :tmp_len]
        tupsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
        tupsample_data[:fft_len // 2, :] = trtdata[:fft_len // 2, :]
        tupsample_data[-fft_len // 2:, :] = trtdata[-fft_len // 2:, :]
        trtdata = cupy.fft.ifft(tupsample_data, axis=0)[:up_nsam, :]
        cupy.cuda.Device().synchronize()
        bpj_grid2 = cupy.zeros((nbpj_pts, nbpj_pts), dtype=np.complex128)

        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, bgx_gpu, bgy_gpu, bgz_gpu, rbins_gpu, panrx_gpu,
                                                elrx_gpu,
                                                panrx_gpu, elrx_gpu, trtdata, bpj_grid2,
                                                bpj_wavelength, ranges[0] / c0,
                                                rp.fs * upsample, bwidth, rp.az_half_bw, rp.el_half_bw, 0, pts_debug,
                                                angs_debug, False)
        cuda.synchronize()
        bpj_truedata += bpj_grid2.get()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del data_r
    del data_i
    del rtdata
    del rcdata
    # del upsample_data
    del bpj_grid
    mempool.free_all_blocks()

    if do_truth_backproject:
        del trtdata
        del tupsample_data
        del bpj_grid2

del rbins_gpu
del rmat_gpu
del shift_gpu
del bgx_gpu
del bgy_gpu
del bgz_gpu
del ngz_gpu
del autocorr_gpu
if do_truth_backproject:
    del mfilt_gpu
mempool.free_all_blocks()

# dfig = go.Figure(data=[go.Mesh3d(x=bg.grid[:, :, 0].ravel(), y=bg.grid[:, :, 1].ravel(), z=bg.grid[:, :, 2].ravel(),
#                                  facecolor=bg.refgrid.ravel(), facecolorsrc='teal')])
ngx, ngy, ngz = bg.getGrid()
if debug:
    flight = rp.pos(rp.gpst)
    # dfig.add_scatter3d(x=flight[0, :], y=flight[1, :], z=flight[2, :], mode='markers')
    dfig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
    dfig.add_scatter3d(x=ngx.flatten(),
                       y=ngy.flatten(),
                       z=ngz.flatten(), mode='markers')
    # dfig.add_scatter3d(x=flight[0, :], y=flight[1, :], z=flight[2, :], mode='markers')
    dfig.add_scatter3d(x=locd[0, ...].flatten() + locp[0], y=locd[1, ...].flatten() + locp[1],
                       z=locd[2, ...].flatten() + locp[2], marker={'color': angd[2, ...].flatten()}, mode='markers')
    dfig.add_scatter3d(x=locd_bj[0, :nbpj_pts, :nbpj_pts].flatten() + locp[0], y=locd_bj[1, :nbpj_pts, :nbpj_pts].flatten() + locp[1],
                       z=locd_bj[2, :nbpj_pts, :nbpj_pts].flatten() + locp[2], mode='markers')
    # orig = llh2enu(*bg.origin, bg.ref)
    # dfig.add_scatter3d(x=[orig[0]], y=[orig[1]], z=[orig[2]], marker={'size': [10]}, mode='markers')
    # dfig.add_scatter3d(x=bg.grid[0].flatten(), y=bg.grid[1].flatten(), z=bg.grid[2].flatten(), mode='markers')
    dfig.show()

fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=bg.refgrid.flatten(), range_color=[0, 140])
fig.add_scatter(x=gx.flatten(), y=gy.flatten(), mode='markers')
fig.show()

fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=bg.refgrid.flatten(), opacity=.1)
fig.add_scatter(x=gx.flatten(), y=gy.flatten(), marker={'color': db(bpj_res).flatten()}, mode='markers')
fig.show()

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
tayd = taylor(cpi_len)
taydopp = np.ones((up_nsam, 1)).dot(tayd.reshape(1, -1))
plt.subplot(2, 1, 1)
plt.title('Generated')
plt.imshow(np.fft.fftshift(db(np.fft.fft(test * taydopp, axis=1)), axes=1))
plt.axis('tight')
plt.subplot(2, 1, 2)
plt.title('Post-BJ')
plt.imshow(np.fft.fftshift(db(np.fft.fft(test_bj * taydopp, axis=1)), axes=1))
plt.axis('tight')

plt.figure('Time Data')
plt.imshow(db(test))
plt.axis('tight')

if do_truth_backproject:
    bfig = px.scatter(x=gx.flatten(), y=gy.flatten(), color=db(bpj_truedata).flatten())
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

plt.figure('Elevation')
plt.imshow(gz, origin='lower')
plt.axis('tight')
plt.show()

'''bg = SDREnvironment(sdr)
bg.resample(bg.origin, grid_width, grid_height, (nbpj_pts * 1, nbpj_pts * 1), -np.pi / 3)
vgz = bg.getGrid()[2]
vert_reflectivity = bg.refgrid
rot = bg._transforms[0]
shift = bg._transforms[1]
ptx, py = np.meshgrid(np.arange(bg.shape[0]), np.arange(bg.shape[1]))
bx = ptx.astype(float)
by = py.astype(float)
bar_z = vgz[ptx, py]
gpr = vert_reflectivity[ptx, py]
bx -= vgz.shape[0] / 2
by -= vgz.shape[1] / 2
bar_x = rot[0, 0] * bx + rot[0, 1] * by + shift[0]
bar_y = rot[1, 0] * bx + rot[1, 1] * by + shift[1]

shift_x, shift_y, _ = llh2enu(*bg.origin, bg.ref)
gx, gy = np.meshgrid(np.linspace(-grid_width / 2, grid_width / 2, nbpj_pts),
                     np.linspace(-grid_height / 2, grid_height / 2, nbpj_pts))
gx += shift_x
gy += shift_y
latg, long, altg = enu2llh(gx.flatten(), gy.flatten(), np.zeros(gx.flatten().shape[0]), bg.ref)
gz = (getElevationMap(latg, long) - bg.ref[2]).reshape(gx.shape)

fig = px.scatter_3d(x=bar_x.flatten(), y=bar_y.flatten(), z=vgz.flatten())
fig.add_scatter3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten(), mode='markers')
fig.add_scatter3d(x=bar_x.flatten(), y=bar_y.flatten(), z=bar_z.flatten('F'), mode='markers')
fig.show()'''