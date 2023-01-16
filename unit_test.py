import keras.models
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
from scipy.signal import butter, sosfilt
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
from SDRParsing import SDRParse, load, getModeValues
from SDRWriting import SDRWrite
import pickle

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
mempool = cupy.get_default_memory_pool()


def compileWaveformData(sdr_f, fft_sz, l_sz, bandwidth, wf_fc, slant_min, slant_max, fs, bin_bw):
    target_data = np.zeros((l_sz, fft_sz, 1))
    for n in range(l_sz):
        tpsd = db(genTargetPSD(bandwidth, wf_fc, slant_min, slant_max,
                               fft_sz, fs)).reshape((fft_sz, 1))
        tpsd /= np.linalg.norm(tpsd)
        target_data[n, :, :] = tpsd

    td_mu = np.mean(target_data)
    td_std = np.std(target_data)
    target_data = (np.fft.fftshift(target_data, axes=1)[:, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2,
                   :] - td_mu) / td_std

    # Generate the Clutter PSDs from the SDR file
    clutter_data = sdr_f.getPulses(sdr_f[0].frame_num[np.arange(0, l_sz)], 0)
    cd_fftdata = np.fft.fft(clutter_data, fft_sz, axis=0)
    clutter_data = db(cd_fftdata)
    clutter_data /= np.linalg.norm(clutter_data, axis=0)
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


def getWaveFromData(wfd_mdl, target_data, clutter_data, clutter_phase_data, cd_mu, cd_std, l_sz, n_ants, fft_sz,
                    bin_bw):
    wave_output = (wfd_mdl.predict((clutter_data, target_data, clutter_phase_data)) * cd_mu) + cd_std

    waveforms = np.zeros((l_sz, n_ants, fft_sz))
    waveforms[:, :, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2] = db(
        wave_output[:, :, :bin_bw] + 1j * wave_output[:, :, -bin_bw:])
    waveforms = waveforms / np.linalg.norm(waveforms, axis=2)[:, :, None]
    waveforms = waveforms[:, :, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2]
    return waveforms


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

bg_file = '/data5/SAR_DATA/2021/09202021/SAR_09202021_151245.sar'
# bg_file = '/data5/SAR_DATA/2022/03282022/SAR_03282022_082824.sar'
sim_upsample = 1
upsample = 1
channel = 0
cpi_len = 32
plp = .12
max_pts_per_tri = 1
debug = False
nbpj_pts = 100
grid_width = 300
grid_height = 300
grid_center = [40.098826, -111.659694, 0]
cal_num = 500
collect_time = 18.45
collect_alt = 1524
do_truth_backproject = False
custom_waveform = False
use_sdr_file = True
write_file = True
write_fnme = '/home/jeff/repo/Raw/sim_write.sar'

print('Loading SDR file...')
sdr = load(bg_file)
grid_center[2] = getElevation((40.098826, -111.659694))

# Generate the background for simulation
print('Generating environment...', end='')
# bg = MapEnvironment((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'], sdr.ash['geo']['hRef']), extent=(120, 120))
bg = SDREnvironment(sdr)
grid_center = bg.origin

# Get reference data
# flight = rp.pos(postCorr['sec'])
fc = sdr[channel].fc
fs = sdr[channel].fs if use_sdr_file else 2e9
bwidth = sdr[channel].bw if use_sdr_file else 200e6
prf = sdr[channel].prf if use_sdr_file else 1300

print('Done.')

# Generate a platform
print('Generating platform...', end='')
start_loc = llh2enu(bg.sdr.xml['Flight_Line']['Start_Latitude_D'], bg.sdr.xml['Flight_Line']['Start_Longitude_D'],
                        bg.sdr.xml['Flight_Line']['Flight_Line_Altitude_M'], bg.ref)
stop_loc = llh2enu(bg.sdr.xml['Flight_Line']['Stop_Latitude_D'], bg.sdr.xml['Flight_Line']['Stop_Longitude_D'],
                    bg.sdr.xml['Flight_Line']['Flight_Line_Altitude_M'], bg.ref)
course_heading = np.arctan2(start_loc[0] - stop_loc[0], start_loc[1] - stop_loc[1]) + np.pi
if use_sdr_file:
    rp = SDRPlatform(sdr, bg.ref, channel=channel)
else:
    gps_times = sdr.gps_data.index[0] + np.arange(0, collect_time, .01)
    nt = len(gps_times)
    rp = RadarPlatform(e=np.linspace(start_loc[0], stop_loc[0], nt), n=np.linspace(start_loc[1], stop_loc[1], nt),
                       u=np.zeros(nt) + collect_alt,
                       r=np.zeros(nt), p=np.zeros(nt), y=np.zeros(nt) + course_heading, t=gps_times,
                       gimbal=np.zeros((nt, 2)) + np.array([sdr.gimbal['pan'].mean(), sdr.gimbal['tilt'].mean()]),
                       dep_angle=38.0,
                       el_bw=10, az_bw=10,
                       gimbal_offset=np.array([-0.3461, 1.3966, -1.2522]),
                       gimbal_rotations=np.array([0, 0, -np.pi / 2]), fs=fs)
print('Done.')

if write_file:
    print('Initiating file writing...', end='')
    if use_sdr_file:
        trumode = getModeValues(int(sdr[0].mode, 16), not sdr.dataversion)
    else:
        trumode = {'digital_channel': 0, 'wavenumber': 0,
           'operation': 0, 'adc_channel': 0, 'receiver_slot': 5, 'receiver_channel': 0, 'upconverter_slot': 3,
           '8/9_select': 1, 'band': 2, 'dac_channel': 0,
           'ddc_enable': 0, 'filter_select': 0,
           'rx_port': 0, 'tx_port': 0, 'polarization': 3,
           'numconsmodes': 0, 'awg_enable': 0,
           'rf_ref_wavenumber': 0}

    lats, lons, alts = enu2llh(*rp.pos(rp.gpst), bg.ref)
    r, p, y = rp.att(rp.gpst)
    vn, ve, vu = rp.vel(rp.gpst)
    if use_sdr_file:
        sys_to_gps = np.poly1d(np.polyfit(sdr.timesync['systime'], sdr.timesync['secs'], 1))
        gimbal = np.array([sys_to_gps(sdr.gimbal['systime']), sdr.gimbal['pan'],
                           sdr.gimbal['tilt']]).T
        gps_wk = int(sdr.gps_data['gps_wk'].mean())
    else:
        gimbal = np.array([rp.gpst - .02, rp._gimbal[:, 0], rp._gimbal[:, 1]]).T
        gps_wk = 2229

    sdr_wr = SDRWrite(write_fnme, gps_wk, rp.gpst, lats, lons, alts-0.12150114770299338, r, p, y, vn, ve, vu,
                      gimbal,
                      digital_channel=trumode['digital_channel'], wavenumber=trumode['wavenumber'],
                      adc_channel=trumode['adc_channel'], receiver_slot=trumode['receiver_slot'],
                      receiver_channel=trumode['receiver_channel'], upconverter_slot=trumode['upconverter_slot'],
                      select_89=trumode['8/9_select'], band=trumode['band'], dac_channel=trumode['dac_channel'],
                      ddc_enable=trumode['ddc_enable'], filter_select=trumode['filter_select'],
                      rx_port=trumode['rx_port'], tx_port=trumode['tx_port'], polarization=trumode['polarization'],
                      numconsmodes=trumode['numconsmodes'], awg_enable=trumode['awg_enable'],
                      rf_ref_wavenumber=trumode['rf_ref_wavenumber'], settings_alt=collect_alt, settings_vel=150
                      )

    sdr_wr.addChannel(fc, bwidth, sdr[0].pulse_length_S, prf / 1.5, rp.near_range_angle / DTR,
                      rp.far_range_angle / DTR, plp * 100,
                      4, trans_on_tac=1000, att_mode='AGC', rec_only=False, rec_slice='A', prf_broad=1.5,
                      offset_video=0.)
    for idx, a in enumerate(sdr.ant):
        sdr_wr.addAntenna(idx, a.az_bw / DTR, a.el_bw / DTR, a.dopp_bw / DTR, a.dep_ang / DTR,
                          a.az_phase_centers, a.el_phase_centers,
                          False, a.squint)

    for idx, port in enumerate(sdr.port):
        if not isinstance(port, list):
            sdr_wr.addPort(idx, port.x, port.y, port.z, cable_length=0.)

    sdr_wr.addGimbal(70., 30., rp.dep_ang / DTR, rp.gimbal_offset, rp.gimbal_rotations / DTR, course_heading,
                     squint=0.0, update_rate=100, look_side='Left')
    print('Done.')

# Generate a backprojected image
print('Calculating grid parameters...', end='')
# General calculations for slant ranges, etc.
# plat_height = rp.pos(rp.gpst)[2, :].mean()
if use_sdr_file:
    fdelay = 55
    nr = rp.calcPulseLength(fdelay, plp, use_tac=True)
    nsam = rp.calcNumSamples(fdelay, plp)
    ranges = rp.calcRangeBins(fdelay, upsample, plp)
else:
    plat_height = rp.pos(rp.gpst)[2, :].mean()
    nr = rp.calcPulseLength(plat_height, plp, use_tac=True)
    nsam = rp.calcNumSamples(plat_height, plp)
    ranges = rp.calcRangeBins(plat_height, upsample, plp)
granges = ranges * np.cos(rp.dep_ang)
fft_len = findPowerOf2(nsam + nr)
up_fft_len = fft_len * upsample
up_nsam = nsam * upsample
sim_up_fft_len = fft_len * sim_upsample
sim_up_nsam = nsam * sim_upsample
sim_bpj_decimation = sim_upsample // upsample
print('Done.')

# Model for waveform simulation
if custom_waveform:
    print('Using custom waveform from DNN model.')
    with open('/home/jeff/repo/apache_mla/model/model_params.pic', 'rb') as f:
        model_params = pickle.load(f)
    mdl_fft = model_params['mdl_fft']
    n_conv_filters = model_params['n_conv_filters']
    kernel_sz = model_params['kernel_sz']
    mdl_bin_bw = int(model_params['bandwidth'] // (fs / mdl_fft))
    mdl = genModel(mdl_bin_bw, n_conv_filters, kernel_sz, mdl_fft)
    mdl.load_weights('/home/jeff/repo/apache_mla/model/model')

    # Generate model data for waveform
    wfd = compileWaveformData(sdr, mdl_fft, cpi_len, bwidth, fc, ranges[0], ranges[-1], fs, mdl_bin_bw)

    # Get waveforms
    wf_data = getWaveFromData(mdl, *wfd, cpi_len, 2, mdl_fft, mdl_bin_bw)
    chirp = np.fft.fft(np.fft.ifft(wf_data[0, 0, :]), sim_up_fft_len)
else:
    if use_sdr_file:
        mchirp = np.mean(sdr.getPulses(sdr[channel].cal_num, 0, is_cal=True), axis=1)
    else:
        mchirp = genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr, fs, fc, bwidth) * 31 * (10 ** (31 / 20))
    chirp = np.fft.fft(mchirp, sim_up_fft_len)

# Write out cal pulses for .sar file
if write_file:
    sys_to_gps = np.poly1d(np.polyfit(sdr.timesync['systime'], sdr.timesync['secs'], 1))
    if use_sdr_file:
        for fr in np.arange(sdr[channel].ncals):
            pulse = sdr.getPulse(fr, is_cal=True)
            if pulse is not None:
                time = sys_to_gps(sdr[channel].cal_time[fr])
                att = sdr[channel].atts[fr]
                sdr_wr.writePulse(time, 31, pulse, True)
    else:
        cal_time_start = rp.gpst[0] - cal_num / prf
        for p in np.arange(cal_num):
            sdr_wr.writePulse(cal_time_start + p / prf, 31, mchirp, True)

# Chirp and matched filter calculations
print('Calculating Taylor window.')
if sdr[channel].xml['Offset_Video_Enabled'] == 'True' and use_sdr_file:
    offset_hz = sdr[channel].xml['DC_Offset_MHz'] * 1e6
    bpj_wavelength = c0 / (fc - bwidth / 2 - offset_hz)
    offset_shift = int((offset_hz + bwidth / 2) / (1 / fft_len * fs) * sim_upsample)
    # bpj_wavelength = c0 / (fc - bwidth / 2 - offset_hz)
    # offset_shift = int((offset_hz + bwidth / 2) / (1 / fft_len * fs) * upsample)
else:
    offset_hz = 0
    offset_shift = int(offset_hz / (1 / fft_len * fs) * sim_upsample)
    bpj_wavelength = c0 / fc
taywin = int(bwidth / fs * sim_up_fft_len)
taywin = taywin + 1 if taywin % 2 != 0 else taywin
taytay = taylor(taywin)
tmp = np.zeros(sim_up_fft_len)
tmp[:taywin // 2] = taytay[-taywin // 2:]
tmp[-taywin // 2:] = taytay[:taywin // 2]
taytay = np.fft.ifft(tmp)
# tayd = np.fft.fftshift(taylor(cpi_len))
# taydopp = np.fft.fftshift(np.ones((up_nsam, 1)).dot(tayd.reshape(1, -1)), axes=1)

print('Calculating matched filter.')
reflection_freq = c0 / bpj_wavelength - fs * np.round(c0 / bpj_wavelength / fs)
if reflection_freq != 0:
    tayshifted = taytay * np.sin(2 * np.pi * reflection_freq * np.arange(sim_up_fft_len) * 1 / fs)
    mfilt = chirp.conj() * np.fft.fft(tayshifted, sim_up_fft_len)
    if reflection_freq < 0:
        mfilt[sim_up_fft_len // 2:] = 0
    elif reflection_freq > 0:
        mfilt[:sim_up_fft_len // 2] = 0
else:
    mfilt = chirp.conj() * np.fft.fft(taytay, sim_up_fft_len)


# if do_truth_backproject:
mfilt_gpu = cupy.array(np.tile(mfilt[::sim_upsample], (cpi_len, 1)).T, dtype=np.complex128)

# autocorr_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T * np.tile(mfilt, (cpi_len, 1)).T, dtype=np.complex128)
chirp_gpu = cupy.array(np.tile(chirp, (cpi_len, 1)).T, dtype=np.complex128)

bg.resample(grid_center, grid_width, grid_height, (nbpj_pts, nbpj_pts))
# gx, gy, gz = bg.getGrid(bg.origin, grid_width, grid_height, (nbpj_pts, nbpj_pts))
gx, gy, gz = bg.getGrid()
ngz_gpu = cupy.array(bg.getGrid()[2], dtype=np.float64)
ng = np.zeros(bg.shape)
# ng = bg.grid_function(gx.flatten(), gy.flatten()).reshape(gx.shape)
pts_dist = np.linalg.norm(np.array([n for n in bg.getGrid()]) - np.array(llh2enu(*grid_center, bg.ref))[:, None, None],
                          axis=0)
ng[np.where(pts_dist < 10)] = 1000
# ng[ng.shape[0] // 2, ng.shape[1] // 2] = 100
# ng[ng.shape[0] // 2 - 15, :] = 1
# ng[:, ng.shape[1] // 2 - 15] = 1
# ng[::5, ::5] = 1

# ng = Image.open('/home/jeff/Downloads/artemislogo.png').resize(bg.shape, Image.ANTIALIAS)
# ng = np.linalg.norm(np.array(ng), axis=2)
# bg._refgrid = ng

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
    pts_debug = cupy.zeros((1, 1, 1), dtype=np.float64)
    angs_debug = cupy.zeros((1, 1, 1), dtype=np.float64)
test = None

# GPU device calculations
threads_per_block = getMaxThreads()
bpg_ranges = (bg.refgrid.shape[0] // threads_per_block[0] + 1,
              bg.refgrid.shape[1] // threads_per_block[1] + 1)
bpg_bpj = (gx.shape[0] // threads_per_block[0] + 1,
           gx.shape[1] // threads_per_block[1] + 1)

# Data blocks for imaging
bpj_res = np.zeros(gx.shape, dtype=np.complex128)
bpj_truedata = np.zeros(gx.shape, dtype=np.complex128)

rng_states = create_xoroshiro128p_states(bpg_ranges[0] * bpg_ranges[1] * threads_per_block[0] * threads_per_block[1],
                                         seed=10)

# Run through loop to get data simulated
if use_sdr_file:
    data_t = 162785.37023 + np.arange(49146) / sdr[0].prf  #sdr[channel].pulse_time
    # data_t = rp.gpst.mean() + np.arange(-cpi_len // 2, cpi_len // 2) * (data_t[1] - data_t[0])
    idx_t = np.arange(49146)  #sdr[channel].frame_num
else:
    data_t = np.arange(rp.gpst[0], rp.gpst[0] + collect_time, 1 / prf)
    idx_t = np.arange(len(data_t))
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
    data_r = cupy.random.randn(sim_up_nsam, tmp_len, dtype=np.float64) * 0 #1e-7
    data_i = cupy.random.randn(sim_up_nsam, tmp_len, dtype=np.float64) * 0 #1e-7
    genRangeWithoutIntersection[bpg_ranges, threads_per_block](rmat_gpu, shift_gpu, ngz_gpu, ref_coef_gpu,
                                                               postx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                               panrx_gpu, elrx_gpu, data_r, data_i, rng_states,
                                                               pts_debug, angs_debug, bpj_wavelength,
                                                               ranges[0] / c0, rp.fs * sim_upsample, rp.az_half_bw,
                                                               rp.el_half_bw, max_pts_per_tri, debug_flag)
    cuda.synchronize()
    data_r[np.isnan(data_r)] = 0
    data_i[np.isnan(data_i)] = 0

    # Create data using chirp
    rtdata = cupy.fft.fft(data_r + 1j * data_i, sim_up_fft_len, axis=0) * chirp_gpu[:, :tmp_len]
    # Decimate to fit backprojection
    # rtdata = rtdata[::sim_bpj_decimation, :]
    rcdata = cupy.fft.ifft(rtdata * mfilt_gpu[:, :tmp_len], axis=0)[:sim_up_nsam:sim_bpj_decimation, :]
    cuda.synchronize()

    # Simulated data debug checks
    if ts[0] < rp.gpst.mean() <= ts[-1]:
        print(f'Center of data at {tidx * cpi_len}')
        locp = rp.rxpos(ts[0])
        test = rcdata.get()
        test_check = rtdata.get()
        angd = angs_debug.get()
        locd = pts_debug.get()

    if write_file:
        wr_data = cupy.fft.ifft(rtdata, axis=0).get()[:sim_up_nsam:sim_bpj_decimation, :]
        for idx in range(wr_data.shape[1]):
            att = 1
            pulse = wr_data[:, idx]
            while abs(pulse).max() / 10**(att / 20) > 32767 and att < 31:
                att += 1
            if np.any(abs(pulse).max() / 10**(att / 20) > 32767):
                print('TOO BIG OF A VALUE!')
            sdr_wr.writePulse(ts[idx], att, pulse)

    cuda.synchronize()

    bpj_grid = cupy.zeros(gx.shape, dtype=np.complex128)
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
del chirp_gpu
del mfilt_gpu
mempool.free_all_blocks()

if write_file:
    sdr_wr.writeXML(prf * 4)

# dfig = go.Figure(data=[go.Mesh3d(x=bg.grid[:, :, 0].ravel(), y=bg.grid[:, :, 1].ravel(), z=bg.grid[:, :, 2].ravel(),
#                                  facecolor=bg.refgrid.ravel(), facecolorsrc='teal')])
ngx, ngy, ngz = bg.getGrid()
colors = db(bg.refgrid).flatten()
colors = (colors - colors.min()) / (colors.max() - colors.min())
if debug:
    flight = rp.pos(rp.gpst)

    dfig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
    # dfig.add_scatter3d(x=flight[0, :], y=flight[1, :], z=flight[2, :], mode='markers')
    dfig.add_scatter3d(x=ngx.flatten(),
                       y=ngy.flatten(),
                       z=ngz.flatten(), mode='markers', name='BG Grid Positions',
                       marker={'color': db(bg.refgrid).flatten()})
    dfig.add_mesh3d(x=ngx.flatten(),
                    y=ngy.flatten(),
                    z=ngz.flatten(), name='BG Grid Positions',
                    vertexcolor=db(bg.refgrid).flatten())
    # dfig.add_scatter3d(x=flight[0, :], y=flight[1, :], z=flight[2, :], mode='markers')
    dfig.add_scatter3d(x=locd[0, ...].flatten() + locp[0], y=locd[1, ...].flatten() + locp[1],
                       z=locd[2, ...].flatten() + locp[2], marker={'color': angd[2, ...].flatten()}, mode='markers',
                       name='SimGrid Projected Positions')
    dfig.add_scatter3d(x=locd_bj[0, :nbpj_pts, :nbpj_pts].flatten() + locp[0],
                       y=locd_bj[1, :nbpj_pts, :nbpj_pts].flatten() + locp[1],
                       z=locd_bj[2, :nbpj_pts, :nbpj_pts].flatten() + locp[2], mode='markers',
                       name='BPJ Grid Projected Positions')
    # orig = llh2enu(*bg.origin, bg.ref)
    # dfig.add_scatter3d(x=[orig[0]], y=[orig[1]], z=[orig[2]], marker={'size': [10]}, mode='markers')
    # dfig.add_scatter3d(x=bg.grid[0].flatten(), y=bg.grid[1].flatten(), z=bg.grid[2].flatten(), mode='markers')
    dfig.show()

    check_rp = SDRPlatform(sdr, bg.ref, channel=channel)
    check_flight = check_rp.pos(rp.gpst)
    pfig = px.scatter_3d(x=flight[0, :], y=flight[1, :], z=flight[2, :])
    pfig.add_scatter3d(x=check_flight[0, :], y=check_flight[1, :], z=check_flight[2, :], mode='markers')
    pfig.show()

fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=db(bg.refgrid.flatten()), range_color=[0, 140])
fig.add_scatter(x=gx.flatten(), y=gy.flatten(), mode='markers')
fig.show()

fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=db(bg.refgrid.flatten()), opacity=.1)
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
plt.imshow(db(bg.refgrid), origin='lower', cmap='gray')
plt.axis('tight')
plt.axis('off')

plt.figure('BPJ Grid')
plt.imshow(db(bpj_res), origin='lower', cmap='gray')
plt.axis('tight')
plt.axis('off')

plt.figure('Elevation')
plt.imshow(gz, origin='lower')
plt.axis('tight')

plt.figure('Waveforms')
plt.plot(db(chirp))
plt.plot(db(mfilt))

if debug:
    plt.figure('BeamPattern')
    plt.scatter(locd[0, ...].ravel(), locd[1, ...].ravel(), c=angd[2, ...].ravel())

clutter_pulse = sdr.getPulse(1000)
clutter_fft = np.fft.fft(clutter_pulse, fft_len)
plt.figure()
plt.axis('off')
plt.subplot(1, 2, 1)
plt.title('Magnitude')
plt.plot(np.fft.fftshift(db(chirp)))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Phase')
plt.plot(np.angle(chirp))
plt.axis('off')

gps_written = loadPostCorrectionsGPSData('/home/jeff/repo/Debug/sim_write_Channel_1_X-Band_9_GHz_VV_postCorrectionsGPSData.dat')
gps_orig = loadPostCorrectionsGPSData('/home/jeff/repo/Debug/09202021/SAR_09202021_151245_Channel_1_X-Band_9_GHz_VV_postCorrectionsGPSData.dat')
ew, nw, uw = llh2enu(gps_written['rx_lat'], gps_written['rx_lon'], gps_written['rx_alt'], bg.ref)
ew = gps_written['rx_lat'] * gps_written['latConv']
nw = gps_written['rx_lon'] * gps_written['lonConv']
eo, no, uo = llh2enu(gps_orig['rx_lat'], gps_orig['rx_lon'], gps_orig['rx_alt'], bg.ref)
eo = gps_orig['rx_lat'] * gps_orig['latConv']
no = gps_orig['rx_lon'] * gps_orig['lonConv']
plt.figure('GPS datacheck')
plt.subplot(2, 2, 1)
plt.title('e')
plt.plot(gps_written['sec'], ew)
plt.plot(gps_orig['sec'], eo)
# plt.plot(gps_orig['sec'], rp.pos(gps_orig['sec'])[0])
plt.subplot(2, 2, 2)
plt.title('n')
plt.plot(gps_written['sec'], nw)
plt.plot(gps_orig['sec'], no)
# plt.plot(gps_orig['sec'], rp.pos(gps_orig['sec'])[1])
plt.subplot(2, 2, 3)
plt.title('u')
plt.plot(gps_written['sec'], uw)
plt.plot(gps_orig['sec'], uo)
# plt.plot(gps_orig['sec'], rp.pos(gps_orig['sec'])[2])
# plt.legend(['written', 'original', 'platform'])

plt.figure('GPS txdatacheck')
ew, nw, uw = llh2enu(gps_written['tx_lat'], gps_written['tx_lon'], gps_written['tx_alt'], bg.ref)
eo, no, uo = llh2enu(gps_orig['tx_lat'], gps_orig['tx_lon'], gps_orig['tx_alt'], bg.ref)
plt.subplot(2, 2, 1)
plt.title('e')
plt.plot(gps_written['sec'], ew)
plt.plot(gps_orig['sec'], eo)
plt.plot(gps_orig['sec'], rp.pos(gps_orig['sec'])[0])
plt.subplot(2, 2, 2)
plt.title('n')
plt.plot(gps_written['sec'], nw)
plt.plot(gps_orig['sec'], no)
plt.plot(gps_orig['sec'], rp.pos(gps_orig['sec'])[1])
plt.subplot(2, 2, 3)
plt.title('u')
plt.plot(gps_written['sec'], uw)
plt.plot(gps_orig['sec'], uo)
plt.plot(gps_orig['sec'], rp.pos(gps_orig['sec'])[2])
plt.legend(['written', 'original', 'platform'])

plt.figure('GPS timecheck')
plt.subplot(2, 2, 1)
plt.title('e')
plt.scatter(gps_orig['sec'], eo)
plt.scatter(gps_written['sec'], ew)
plt.scatter(rp.gpst, rp.pos(rp.gpst)[0])
plt.subplot(2, 2, 2)
plt.title('n')
plt.scatter(gps_orig['sec'], no)
plt.scatter(gps_written['sec'], nw)
plt.scatter(rp.gpst, rp.pos(rp.gpst)[1])
plt.subplot(2, 2, 3)
plt.title('u')
plt.scatter(gps_orig['sec'], uo)
plt.scatter(gps_written['sec'], uw)
plt.scatter(rp.gpst, rp.pos(rp.gpst)[2])
plt.legend(['original', 'written', 'platform'])

plt.figure('GPS az check')
plt.plot(gps_orig['sec'], gps_orig['az'])
plt.plot(gps_written['sec'], gps_written['az'])
plt.plot(rp.gpst, rp.att(rp.gpst)[2])

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
