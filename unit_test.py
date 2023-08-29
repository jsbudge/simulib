import sys

# sys.path.extend(['/home/jeff/repo/apache_mla', '/home/jeff/repo/data_converter', '/home/jeff/repo/simulib'])

import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, getElevationMap, enu2llh, \
    loadPostCorrectionsGPSData, loadPreCorrectionsGPSData, loadGPSData, loadGimbalData, GetAdvMatchedFilter, \
    loadMatchedFilter
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads, backproject
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
from scipy.interpolate import CubicSpline, interpn
from scipy.signal import butter, sosfilt
from PIL import Image
from scipy.signal import medfilt2d
import cupy as cupy
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states, init_xoroshiro128p_states
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
from SDRParsing import SDRParse, load, getModeValues
from SDRWriting import SDRWrite, SHRT_MAX
import pickle
import json
from celluloid import Camera
from itertools import product

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
mempool = cupy.get_default_memory_pool()

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

command_line_args = sys.argv

print('Getting settings from JSON file...')
with open('./settings.json', 'r') as sf:
    settings = json.load(sf)
    grid_center = settings['grid_center']
    nchan = len(settings['tx']) * len(settings['rx'])
    nrx = len(settings['rx'])
    nants = len(settings['antennas'])
    fs = settings['sample_frequency']
    band_frequency = 9e9 if settings['tx'][0]['center_frequency'] < 31e9 else 32e9

if len(command_line_args) > 1:
    bg_file = command_line_args[1]
else:
    bg_file = '/data5/SAR_DATA/2022/03032022/SAR_03032022_155926.sar'
# bg_file = '/data5/SAR_DATA/2022/03282022/SAR_03282022_082824.sar'

print(f'Loading SAR file {bg_file}...')
sdr = load(bg_file)

grid_center[2] = getElevation((grid_center[0], grid_center[1]))

# Generate the background for simulation
print('Generating environment...', end='')
# bg = MapEnvironment((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'], sdr.ash['geo']['hRef']), extent=(120, 120))
bg = SDREnvironment(sdr)

# Set the channel specific data if not using custom params
if settings['use_sdr_waveform']:
    grid_center = bg.origin
    ants = []
    for idx, ant in enumerate(sdr.port):
        if ant:
            ants.append({'offset': [ant.x, ant.y, ant.z], 'az_beamwidth': sdr.ant[ant.assoc_ant].az_bw / DTR,
                         'el_beamwidth': sdr.ant[ant.assoc_ant].el_bw / DTR,
                         'dopp_beamwidth': sdr.ant[ant.assoc_ant].az_bw / DTR})
    settings['antennas'] = ants
    settings['tx'] = [{'pulse_length_percent': sdr[settings['channel']].xml['Pulse_Length_Percent'] / 100,
                       'center_frequency': sdr[settings['channel']].fc,
                       'bandwidth': sdr[settings['channel']].bw}]
    settings['prf'] = sdr[settings['channel']].prf
    nchan = 1
rx_to_chan = {}
rx2chanconv = 0
for rx in settings['rx']:
    if rx not in rx_to_chan:
        rx_to_chan[rx] = rx2chanconv
        rx2chanconv += 1

print('Done.')

# Generate a platform
print('Generating platform...', end='')
rps = []

settings['nsam'] = 0
if settings['use_sdr_gps']:
    fdelay = 55
    rps.append(SDRPlatform(sdr, bg.ref, channel=settings['channel']))
    rps[0].rx_num = 0
    rps[0].tx_num = 0
    settings['tx'][0]['nr'] = \
        rps[-1].calcPulseLength(fdelay, settings['tx'][0]['pulse_length_percent'], use_tac=True)
    settings['ranges'] = rps[-1].calcRangeBins(fdelay, settings['upsample'], settings['tx'][0]['pulse_length_percent'])
    settings['nsam'] = rps[0].calcNumSamples(fdelay, settings['tx'][0]['pulse_length_percent'])

else:
    start_loc = llh2enu(bg.sdr.xml['Flight_Line']['Start_Latitude_D'], bg.sdr.xml['Flight_Line']['Start_Longitude_D'],
                        bg.sdr.xml['Flight_Line']['Flight_Line_Altitude_M'], bg.ref)
    stop_loc = llh2enu(bg.sdr.xml['Flight_Line']['Stop_Latitude_D'], bg.sdr.xml['Flight_Line']['Stop_Longitude_D'],
                       bg.sdr.xml['Flight_Line']['Flight_Line_Altitude_M'], bg.ref)
    course_heading = np.arctan2(start_loc[0] - stop_loc[0], start_loc[1] - stop_loc[1]) + np.pi
    gps_times = sdr.gps_data.index.values
    nt = len(gps_times)
    for tx_ant, rx_n in np.dstack(np.meshgrid(*[settings['tx'], settings['rx']], indexing='ij')).reshape(-1, 2):
        rps.append(RadarPlatform(e=np.linspace(start_loc[0], stop_loc[0], nt),
                                 n=np.linspace(start_loc[1], stop_loc[1], nt),
                                 u=np.zeros(nt) + settings['collect_alt'],
                                 r=np.zeros(nt), p=np.zeros(nt), y=np.zeros(nt) + course_heading, t=gps_times,
                                 gimbal=np.zeros((nt, 2)) + np.array(
                                     [sdr.gimbal['pan'].mean(), sdr.gimbal['tilt'].mean()]),
                                 dep_angle=settings['gimbal_dep_angle'],
                                 el_bw=settings['antennas'][tx_ant['antenna']]['el_beamwidth'],
                                 az_bw=settings['antennas'][tx_ant['antenna']]['az_beamwidth'],
                                 gimbal_offset=settings['gimbal_offset'],
                                 gimbal_rotations=settings['gimbal_rotations'],
                                 rx_offset=settings['antennas'][rx_n]['offset'],
                                 tx_offset=settings['antennas'][tx_ant['antenna']]['offset'], tx_num=tx_ant['antenna'],
                                 rx_num=rx_n, wavenumber=tx_ant['antenna']))
        plat_height = rps[-1].pos(rps[-1].gpst)[2, :].mean()
        tx_ant['nr'] = rps[-1].calcPulseLength(plat_height, tx_ant['pulse_length_percent'], use_tac=True)
        settings['nsam'] = max(settings['nsam'], rps[-1].calcNumSamples(plat_height,
                                                                        tx_ant['pulse_length_percent']))
    settings['ranges'] = rps[-1].calcRangeBins(rps[-1].pos(rps[-1].gpst)[2, :].mean(), settings['upsample'],
                                               max([n['pulse_length_percent'] for n in settings['tx']]))
print('Done.')

print('Calculating grid parameters...', end='')
# General calculations for slant ranges, etc.
fft_len = findPowerOf2(settings['nsam'] + max([ch['nr'] for ch in settings['tx']]))
up_fft_len = fft_len * settings['upsample']
up_nsam = settings['nsam'] * settings['upsample']
sim_up_fft_len = fft_len * settings['sim_upsample']
sim_up_nsam = settings['nsam'] * settings['sim_upsample']
sim_bpj_decimation = 1
print('Done.')

if settings['write_file']:
    print('Initiating file writing...', end='')

    lats, lons, alts = enu2llh(*rps[0].pos(rps[0].gpst), bg.ref)
    r, p, y = rps[0].att(rps[0].gpst)
    vn, ve, vu = rps[0].vel(rps[0].gpst)
    if settings['use_sdr_gps']:
        sys_to_gps = np.poly1d(np.polyfit(sdr.timesync['systime'], sdr.timesync['secs'], 1))
        gimbal = np.array([sys_to_gps(sdr.gimbal['systime']), sdr.gimbal['pan'],
                           sdr.gimbal['tilt']]).T
        gps_wk = int(sdr.gps_data['gps_wk'].mean())
    else:
        gimbal = np.array([rps[0].gpst, rps[0]._gimbal[:, 0], rps[0]._gimbal[:, 1]]).T
        gps_wk = 2176

    sdr_wr = SDRWrite(settings['write_fnme'], gps_wk, rps[0].gpst, lats, lons, alts, r, p, y, vn, ve, vu,
                      gimbal, settings_alt=settings['collect_alt'],
                      settings_vel=settings['collect_vel'])

    for ch_num, rp in enumerate(rps):
        offset_hz = 0.
        # Get the mode for this channel
        if settings['use_sdr_waveform']:
            trumode = getModeValues(int(sdr[settings['channel']].mode, 16), not sdr.dataversion)
            if sdr[ch_num].xml['Offset_Video_Enabled'] == 'True':
                offset_hz = sdr[ch_num].xml['DC_Offset_MHz'] * 1e6
        else:
            # Get the band
            band_number = 2 if 8e9 < settings['tx'][rp.tx_num]['center_frequency'] < 12e9 \
                else 3 if 26.5e9 < settings['tx'][rp.tx_num]['center_frequency'] < 40e9 \
                else 0 if 1e9 < settings['tx'][rp.tx_num]['center_frequency'] < 2e9 else 1
            trumode = {'digital_channel': ch_num,  # Channel number
                       'wavenumber': rp.wavenumber,  # For AWG waveform files
                       'operation': 0,  # Op mode. 0 for normal operation, 1 for cal
                       'adc_channel': 0,
                       'receiver_slot': 5,
                       'receiver_channel': rp.rx_num,
                       'upconverter_slot': 3,
                       '8/9_select': 1,
                       'band': band_number,  # Band the center frequency is at
                       'dac_channel': 0,
                       'ddc_enable': 0,
                       'filter_select': 0,  # Chooses a filter for reducing the sampling rate. Zero if none
                       'rx_port': rp.rx_num,  # Antenna port location for receive
                       'tx_port': rp.tx_num,  # Antenna port location for transmit
                       'polarization': 3,  # Polarization for this channel
                       'numconsmodes': 0,  # Number of simultaneous modes
                       'awg_enable': 1 if settings['tx'][rp.tx_num]['custom_waveform'] else 0,
                       # True if the AWG is enabled
                       'rf_ref_wavenumber': rp.wavenumber}
            if band_frequency != settings['tx'][rp.tx_num]['center_frequency']:
                offset_hz = settings['tx'][rp.tx_num]['center_frequency'] - band_frequency - \
                            settings['tx'][rp.tx_num]['bandwidth'] / 2

        sdr_wr.addChannel(settings['tx'][rp.tx_num]['center_frequency'], settings['tx'][rp.tx_num]['bandwidth'],
                          settings['tx'][rp.tx_num]['nr'] / fs, settings['prf'] / 1.5, rp.near_range_angle / DTR,
                          rp.far_range_angle / DTR, settings['tx'][rp.tx_num]['pulse_length_percent'] * 100,
                          1, trans_on_tac=1000, att_mode='AGC', rec_only=False, rec_slice='A', prf_broad=1.5,
                          digital_channel=trumode['digital_channel'], wavenumber=trumode['wavenumber'],
                          adc_channel=trumode['adc_channel'], receiver_slot=trumode['receiver_slot'],
                          receiver_channel=trumode['receiver_channel'], upconverter_slot=trumode['upconverter_slot'],
                          select_89=trumode['8/9_select'], band=trumode['band'], dac_channel=trumode['dac_channel'],
                          ddc_enable=trumode['ddc_enable'], filter_select=trumode['filter_select'],
                          rx_port=trumode['rx_port'], tx_port=trumode['tx_port'], polarization=trumode['polarization'],
                          numconsmodes=trumode['numconsmodes'], awg_enable=trumode['awg_enable'],
                          rf_ref_wavenumber=trumode['rf_ref_wavenumber'],
                          offset_video=offset_hz)

    # Add the antennas and ports
    if settings['use_sdr_waveform']:
        for idx, a in enumerate(sdr.ant):
            sdr_wr.addAntenna(idx, a.az_bw / DTR, a.el_bw / DTR, a.dopp_bw / DTR, a.dep_ang / DTR,
                              a.az_phase_centers, a.el_phase_centers,
                              False, a.squint)
        for idx, port in enumerate(sdr.port):
            if not isinstance(port, list):
                sdr_wr.addPort(idx, port.x, port.y, port.z, cable_length=0.)
    else:
        for idx, a in enumerate(settings['antennas']):
            sq = a['squint_angle'] if 'squint_angle' in a else 0.0
            sdr_wr.addAntenna(idx, a['az_beamwidth'], a['el_beamwidth'], a['dopp_beamwidth'],
                              rps[0].dep_ang / DTR, 1, 1, False, sq)
            sdr_wr.addPort(idx, *a['offset'], cable_length=0.)

    sdr_wr.addGimbal(70., 30., rps[0].dep_ang / DTR, rps[0].gimbal_offset, rps[0].gimbal_rotations / DTR,
                     course_heading, squint=0.0, update_rate=100, look_side='Left')
    print('Done.')

for ch_num, ch in enumerate(settings['tx']):
    if sdr[ch_num].xml['Offset_Video_Enabled'] == 'True' and settings['use_sdr_waveform']:
        offset_hz = sdr[ch_num].xml['DC_Offset_MHz'] * 1e6
        ch['bpj_wavelength'] = c0 / (ch['center_frequency'] - ch['bandwidth'] / 2 - offset_hz)
        # settings['bpj_wavelength'] = c0 / (ch['center_frequency'] - bwidth / 2 - offset_hz)
        # offset_shift = int((offset_hz + bwidth / 2) / (1 / fft_len * fs) * upsample)
    else:
        offset_hz = ch['center_frequency'] - band_frequency - ch['bandwidth'] / 2
        ch['bpj_wavelength'] = c0 / ch['center_frequency']

wf_data = None
if not settings['use_sdr_waveform']:
    if np.any([n['custom_waveform'] for n in settings['tx']]):
        # Import all the stuff needed for waveform generation
        import keras.models
        from keras.models import load_model
        from keras.utils import custom_object_scope
        from tensorflow.keras.optimizers import Adam, Adadelta
        from wave_train import genTargetPSD, genModel, opt_loss, compileWaveformData, getWaveFromData

        print('Using custom waveform from DNN model.')
        with open('/home/jeff/repo/apache_mla/model/model_params.pic', 'rb') as f:
            model_params = pickle.load(f)
        mdl_fft = model_params['mdl_fft']
        n_conv_filters = model_params['n_conv_filters']
        kernel_sz = model_params['kernel_sz']
        mdl_bin_bw = model_params['bin_bw']
        mdl = genModel(mdl_bin_bw, n_conv_filters, kernel_sz, mdl_fft)
        mdl.load_weights('/home/jeff/repo/apache_mla/model/model')

        # Generate model data for waveform
        wfd = compileWaveformData(sdr, mdl_fft, settings['cpi_len'], settings['tx'][0]['bandwidth'],
                                  settings['tx'][0]['center_frequency'],
                                  settings['ranges'][0],
                                  settings['ranges'][-1], fs, mdl_bin_bw, target_signal=None)

        # Get waveforms
        wf_data = getWaveFromData(mdl, *wfd[:4], settings['cpi_len'], 2, mdl_fft, mdl_bin_bw)

# Model for waveform simulation
for ch_num, ch in enumerate(settings['tx']):
    if settings['use_sdr_waveform']:
        ch['chirp'] = np.fft.fft(np.mean(sdr.getPulses(sdr[ch_num].cal_num, ch_num, is_cal=True), axis=1),
                                 sim_up_fft_len)
        ch['mchirp'] = np.mean(sdr.getPulses(sdr[ch_num].cal_num, ch_num, is_cal=True), axis=1)
        mfilt = GetAdvMatchedFilter(sdr[ch_num], fft_len=sim_up_fft_len)
        waveform = np.mean(sdr.getPulses(sdr[ch_num].cal_num, ch_num, is_cal=True), axis=1)
    else:
        if ch['custom_waveform']:
            ch['bandwidth'] = model_params['bandwidth']
            mchirp = np.zeros(settings['nsam'], dtype=np.complex128)
            reflection_freq = -(c0 / ch['bpj_wavelength'] - fs * np.round(c0 / ch['bpj_wavelength'] / fs))
            mixup = np.fft.fft(np.exp(-1j * 2 * np.pi * ch['center_frequency'] *
                                      np.arange(sim_up_fft_len) / (fs / sim_up_fft_len)),
                               ch['nr']) / ch['nr']
            wf_ifft = np.fft.ifft(wf_data[0, ch_num, :])
            sp = np.sqrt(np.mean(abs(np.fft.ifft(wf_ifft)) ** 2) / ch['power'])
            waveform = np.zeros(ch['nr'], dtype=np.complex128)
            waveform[:len(wf_ifft)] = wf_ifft / sp
            mchirp[5:5 + ch['nr']] = waveform
        else:
            mchirp = np.zeros(settings['nsam'], dtype=np.complex128)
            direction = np.linspace(0, 1, 10) if ch_num == 0 else np.linspace(1, 0, 10)
            waveform = genPulse(np.linspace(0, 1, 10), direction, ch['nr'], fs,
                                ch['center_frequency'], ch['bandwidth']) * np.sqrt(ch['power']) * 1000
            mchirp[5:5 + ch['nr']] = waveform
        ch['chirp'] = np.fft.fft(mchirp, sim_up_fft_len)
        ch['mchirp'] = mchirp
        if settings['backproject'] or settings['rdmap']:
            print('Calculating Taylor window.')
            taywin = int(ch['bandwidth'] / fs * sim_up_fft_len)
            taywin = taywin + 1 if taywin % 2 != 0 else taywin
            taytay = taylor(taywin, sll=80)
            tmp = np.zeros(sim_up_fft_len, dtype=np.complex128)
            tmp[:taywin // 2] = taytay[-taywin // 2:]
            tmp[-taywin // 2:] = taytay[:taywin // 2]
            taytay = np.fft.ifft(tmp)
            # tayd = np.fft.fftshift(taylor(cpi_len))
            # taydopp = np.fft.fftshift(np.ones((up_nsam, 1)).dot(tayd.reshape(1, -1)), axes=1)

            print('Calculating matched filter.')
            nco_lambda = c0 / band_frequency
            reflection_freq = -(c0 / ch['bpj_wavelength'] - fs * np.round(c0 / ch['bpj_wavelength'] / fs))
            if reflection_freq != 0:
                overshot = int((abs(reflection_freq) - ch['bandwidth'] / 2) / (fs / sim_up_fft_len))
                tayshifted = taytay * np.exp(-1j * 2 * np.pi * reflection_freq * np.arange(sim_up_fft_len) * 1 / fs)
                mfilt = ch['chirp'].conj() * np.fft.fft(tayshifted, sim_up_fft_len)
                if reflection_freq < 0:
                    mfilt[sim_up_fft_len // 2 + overshot:] = 0
                elif reflection_freq > 0:
                    mfilt[:sim_up_fft_len // 2 - overshot] = 0
            else:
                mfilt = ch['chirp'].conj() * np.fft.fft(taytay, sim_up_fft_len)

    ch['mfilt'] = cupy.array(np.tile(mfilt[::settings['sim_upsample']], (settings['cpi_len'], 1)).T,
                             dtype=np.complex128)
    ch['chirp'] = cupy.array(np.tile(ch['chirp'], (settings['cpi_len'], 1)).T, dtype=np.complex128)

    if settings['write_file']:
        real_chirp = np.zeros(len(waveform) * 2, dtype=np.complex128)
        real_chirp[:len(waveform) // 2] = np.fft.fft(waveform)[:len(waveform) // 2]
        real_chirp[-len(waveform) // 2:] = np.fft.fft(waveform)[len(waveform) // 2:]
        real_chirp = np.fft.ifft(real_chirp).real
        if np.any(abs(real_chirp) > SHRT_MAX - 1):
            real_chirp *= (SHRT_MAX - 2) / abs(real_chirp).max()

            for rp in rps:
                if rp.wavenumber == ch_num:
                    sdr_wr.writeWaveform(real_chirp, rp.wavenumber)

# Write out cal pulses for .sar file
sys_to_gps = np.poly1d(np.polyfit(sdr.timesync['systime'], sdr.timesync['secs'], 1))
if settings['write_file']:
    if settings['use_sdr_waveform']:
        for fr in np.arange(sdr[settings['channel']].ncals):
            pulse = sdr.getPulse(fr, channel=settings['channel'], is_cal=True)
            if pulse is not None:
                time = sys_to_gps(sdr[settings['channel']].cal_time[fr])
                att = sdr[settings['channel']].atts[fr]
                sdr_wr.writePulse(time, 31, pulse, settings['channel'], True)
    else:
        for ch_num, rp in enumerate(rps):
            cal_time_start = rp.gpst[0] + .05
            for p in np.arange(settings['ncals']):
                sdr_wr.writePulse(cal_time_start + p / settings['prf'], 31,
                                  settings['tx'][rp.tx_num]['mchirp'],
                                  channel=ch_num, is_cal=True)

bg.resample(grid_center, settings['grid_width'], settings['grid_height'], (settings['nbpj_pts'], settings['nbpj_pts']))
# gx, gy, gz = bg.getGrid(bg.origin, settings['grid_width'], settings['grid_height'], (settings['nbpj_pts'], settings['nbpj_pts']))
gx, gy, gz = bg.getGrid()
ngz_gpu = cupy.array(bg.getGrid()[2], dtype=np.float64)
ng = np.zeros(bg.shape)
# ng = bg.grid_function(gx.flatten(), gy.flatten()).reshape(gx.shape)
pts_dist = np.linalg.norm(np.array([n for n in bg.getGrid()]) - np.array(llh2enu(*grid_center, bg.ref))[:, None, None],
                          axis=0)
# ng[np.where(pts_dist < 10)] = 1000
ng[ng.shape[0] // 2, ng.shape[1] // 2] = 1e5
# ng[ng.shape[0] // 2 - 15, :] = 1
# ng[:, ng.shape[1] // 2 - 15] = 1
# ng[::5, ::5] = 1

# ng = Image.open('/home/jeff/Pictures/artemislogo.png').resize(bg.shape, Image.ANTIALIAS)
# ng = np.linalg.norm(np.array(ng), axis=2)
# bg._refgrid = ng

if settings['backproject']:
    bgx_gpu = cupy.array(gx, dtype=np.float64)
    bgy_gpu = cupy.array(gy, dtype=np.float64)
    bgz_gpu = cupy.array(gz, dtype=np.float64)
ref_coef_gpu = cupy.array(bg.refgrid, dtype=np.float64)
rbins_gpu = cupy.array(settings['ranges'], dtype=np.float64)
rmat_gpu = cupy.array(bg.transforms[0], dtype=np.float64)
shift_gpu = cupy.array(bg.transforms[1], dtype=np.float64)

if settings['debug']:
    pts_debug = cupy.zeros((3, *bg.shape), dtype=np.float64)
    angs_debug = cupy.zeros((3, *bg.shape), dtype=np.float64)
    # pts_debug = cupy.zeros((settings['nbpj_pts'], 3), dtype=np.float64)
    # angs_debug = cupy.zeros((settings['nbpj_pts'], 3), dtype=np.float64)
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
if settings['backproject']:
    bpj_res = np.zeros(gx.shape, dtype=np.complex128)

rng_states = create_xoroshiro128p_states(bpg_ranges[0] * bpg_ranges[1] * threads_per_block[0] * threads_per_block[1],
                                         seed=10)

atmos_effect = cupy.array(np.ones_like(settings['ranges']), dtype=np.float64)

# Run through loop to get data simulated
if settings['use_sdr_waveform']:
    data_t = sdr[settings['channel']].pulse_time[0] + np.arange(sdr[settings['channel']].nframes) / settings['prf']
else:
    data_t = np.arange(rps[0].gpst[0], rps[0].gpst[0] + settings['collect_time'], 1 / settings['prf'])
print('Simulating...')
pulse_pos = 0

# Get MIMO virtual array for processing
ants = settings['antennas']
tx_ant_loc = [np.array(ants[s['antenna']]['offset']) for s in settings['tx']]
rx_ant_loc = [np.array(ants[r]['offset']) for r in settings['rx']]
vx_array = np.concatenate([[rx + tx for tx in tx_ant_loc] for rx in rx_ant_loc])
avec_gpu = cupy.array(np.exp(1j * 2 * np.pi *
                             np.array([[s['center_frequency'] for s in settings['tx']]
                                       for _ in settings['rx']]).flatten() / c0 * vx_array.dot(-azelToVec(0, 0))),
                      dtype=np.complex128)
if settings['rdmap']:
    myDopWindow = taylor(settings['cpi_len'], 11, 70).astype(np.complex128)
    slowTimeWindow = myDopWindow.reshape((settings['cpi_len'], 1)).dot(np.ones((1, up_nsam)))
    slow_time_gpu = cupy.array(slowTimeWindow.T, dtype=np.complex128)
    camfig = plt.figure()
    cam = Camera(camfig)

for tidx in tqdm(np.arange(0, len(data_t), settings['cpi_len'])):
    init_xoroshiro128p_states(rng_states, seed=10)
    ts = data_t[tidx + np.arange(min(settings['cpi_len'], len(data_t) - tidx))]
    tmp_len = len(ts)
    rtdata = cupy.zeros((sim_up_fft_len, tmp_len, nrx), dtype=np.complex128)
    debug_flag = False
    for rp in rps:
        # Run through each tx/rx combination and get antenna values
        debug_flag = settings['debug'] if ts[0] < rp.gpst.mean() <= ts[-1] else False
        panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
        elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
        posrx_gpu = cupy.array(rp.rxpos(ts), dtype=np.float64)
        postx_gpu = cupy.array(rp.txpos(ts), dtype=np.float64)

        # Generate range profiles for this channel
        data_r = cupy.array(np.random.randn(sim_up_nsam, tmp_len) * settings['noise_level'], dtype=np.float64)
        data_i = cupy.array(np.random.randn(sim_up_nsam, tmp_len) * settings['noise_level'], dtype=np.float64)
        genRangeWithoutIntersection[bpg_ranges, threads_per_block](rmat_gpu, shift_gpu, ngz_gpu, ref_coef_gpu,
                                                                   postx_gpu, posrx_gpu, panrx_gpu, elrx_gpu,
                                                                   panrx_gpu, elrx_gpu, data_r, data_i, rng_states,
                                                                   pts_debug, angs_debug,
                                                                   settings['tx'][rp.tx_num]['bpj_wavelength'],
                                                                   settings['ranges'][0] / c0,
                                                                   rp.fs * settings['sim_upsample'],
                                                                   rp.az_half_bw,
                                                                   rp.el_half_bw, settings['pts_per_tri'],
                                                                   debug_flag)
        cuda.synchronize()

        # This fixes a weird thing where NaNs appear
        data_r[np.isnan(data_r)] = 0
        data_i[np.isnan(data_i)] = 0

        # Create data using chirp and range profile - this is in the frequency domain still
        rtdata[:, :, rx_to_chan[rp.rx_num]] += cupy.fft.fft(data_r + 1j * data_i, sim_up_fft_len, axis=0) * \
                                               settings['tx'][rp.tx_num]['chirp'][:, :tmp_len]
        cuda.synchronize()

        # Simulated data debug checks
        if ts[0] < rp.gpst.mean() <= ts[-1] and settings['backproject'] and settings['debug']:
            print(f'Center of data at {tidx}')
            locp = rp.rxpos(ts[0])
            test = cupy.fft.ifft(rtdata, axis=0).get()[:sim_up_nsam, :]
            angd = angs_debug.get()
            locd = pts_debug.get()

        if settings['write_file']:
            wr_data = cupy.fft.ifft(rtdata, axis=0).get()[:sim_up_nsam, :]
            for ch in range(wr_data.shape[2]):
                for idx in range(wr_data.shape[1]):
                    att = 1
                    pulse = wr_data[:, idx, ch]
                    while abs(pulse).max() / 10 ** (att / 20) > SHRT_MAX and att < 31:
                        att += 1
                    if np.any(abs(pulse).max() / 10 ** (att / 20) > SHRT_MAX):
                        print('TOO BIG OF A VALUE!')
                    sdr_wr.writePulse(ts[idx], att, pulse, ch)

        cuda.synchronize()

    posesrx = np.mean([rp.rxpos(ts) for rp in rps], axis=0)
    posestx = np.mean([rp.txpos(ts) for rp in rps], axis=0)
    rcdata = cupy.zeros((up_nsam, tmp_len, len(rps)), dtype=np.complex128)
    for rp_idx, rp in enumerate(rps):
        # Range compress the data, upsample it, and send to backprojection kernel
        rcdata_tmp = rtdata[:, :, rx_to_chan[rp.rx_num]] * settings['tx'][rp.tx_num]['mfilt'][:, :tmp_len]
        upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
        upsample_data[:fft_len // 2, :] = rcdata_tmp[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] = rcdata_tmp[-fft_len // 2:, :]
        rcdata[:, :, rp_idx] = cupy.fft.ifft(upsample_data, axis=0)[:up_nsam, :]
    panrx_gpu = cupy.array(rp.pan(ts), dtype=np.float64)
    elrx_gpu = cupy.array(rp.tilt(ts), dtype=np.float64)
    posrx_gpu = cupy.array(posesrx, dtype=np.float64)
    postx_gpu = cupy.array(posestx, dtype=np.float64)

    # eig_values, Uh = np.linalg.eigh(cupy.cov(cupy.mean(rcdata, axis=1).T).get())
    # rcdata = rcdata.dot(cupy.array(Uh[:, 0].reshape((-1, 1)), dtype=np.complex128)).reshape((up_nsam, tmp_len))
    rcdata = rcdata.dot(avec_gpu).reshape((up_nsam, tmp_len))
    if settings['backproject']:
        if ts[0] < rp.gpst.mean() <= ts[-1]:
            test_bj = np.zeros((up_nsam, tmp_len), dtype=np.complex128)
        bpj_grid = cupy.zeros(gx.shape, dtype=np.complex128)
        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, bgx_gpu, bgy_gpu, bgz_gpu, rbins_gpu,
                                                panrx_gpu,
                                                elrx_gpu, panrx_gpu, elrx_gpu, rcdata, bpj_grid,
                                                settings['tx'][rp.tx_num]['bpj_wavelength'],
                                                settings['ranges'][0] / c0,
                                                rp.fs * settings['upsample'],
                                                settings['tx'][rp.tx_num]['bandwidth'],
                                                rp.az_half_bw, rp.el_half_bw, 0, pts_debug,
                                                angs_debug, debug_flag, atmos_effect)

        cuda.synchronize()
        if ts[0] < rp.gpst.mean() <= ts[-1]:
            locp_bj = rp.rxpos(ts[-1])
            test_bj = rcdata.get()
            angd_bj = angs_debug.get()
            locd_bj = pts_debug.get()
        cuda.synchronize()
        bpj_res += bpj_grid.get()
    if settings['rdmap']:
        # get grazing angles over range bins
        graze_angles = np.arcsin(rp.pos(ts[0])[2] / settings['ranges'])
        rvec = azelToVec(rp.pan(ts[0]), graze_angles)
        dopp_cen = (2 / settings['tx'][rp.tx_num]['bpj_wavelength']) * \
                   rvec.T.dot(rp.vel(ts[0]).flatten()) % settings['prf']
        slowtimes = np.arange(tmp_len).reshape((1, tmp_len)) / settings['prf']
        slowtimePhases = np.exp(1j * 2 * np.pi * -dopp_cen.reshape(up_nsam, 1).dot(slowtimes))
        phases_gpu = cupy.array(slowtimePhases, dtype=np.complex128)
        doppdata = cupy.fft.fft(rcdata * phases_gpu * slow_time_gpu[:, :tmp_len], axis=1).get()
        plt.imshow(db(doppdata))
        plt.axis('tight')
        cam.snap()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del data_r
    del data_i
    del rtdata
    del rcdata
    del rcdata_tmp
    del upsample_data
    if settings['backproject']:
        del bpj_grid
    if settings['rdmap']:
        del phases_gpu
    mempool.free_all_blocks()

del rbins_gpu
del rmat_gpu
del shift_gpu
if settings['backproject']:
    del bgx_gpu
    del bgy_gpu
    del bgz_gpu
del ngz_gpu
for tx_it in settings['tx']:
    tx_it['chirp'] = tx_it['chirp'].get()
    if settings['backproject']:
        tx_it['mfilt'] = tx_it['mfilt'].get()
if settings['rdmap']:
    del slow_time_gpu
mempool.free_all_blocks()

if settings['write_file']:
    sdr_wr.finalize(settings['prf'])
    print(f'File written as {sdr_wr.fnme}')

# dfig = go.Figure(data=[go.Mesh3d(x=bg.grid[:, :, 0].ravel(), y=bg.grid[:, :, 1].ravel(), z=bg.grid[:, :, 2].ravel(),
#                                  facecolor=bg.refgrid.ravel(), facecolorsrc='teal')])

if settings['debug'] and settings['rdmap']:
    anim = cam.animate()

if settings['debug'] and settings['backproject']:
    print('Generating debug plots...')
    ngx, ngy, ngz = bg.getGrid()
    flight = rp.pos(rp.gpst)

    if settings['nbpj_pts'] < 150:
        dfig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
        dfig.add_scatter3d(x=ngx.flatten(),
                           y=ngy.flatten(),
                           z=ngz.flatten(), mode='markers', name='BG Grid Positions',
                           marker={'color': db(bg.refgrid).flatten()})
        dfig.add_mesh3d(x=ngx.flatten(),
                        y=ngy.flatten(),
                        z=ngz.flatten(), name='BG Grid Positions',
                        vertexcolor=db(bg.refgrid).flatten())
        dfig.add_scatter3d(x=locd[0, ...].flatten() + locp[0], y=locd[1, ...].flatten() + locp[1],
                           z=locd[2, ...].flatten() + locp[2], marker={'color': angd[2, ...].flatten()}, mode='markers',
                           name='SimGrid Projected Positions')
        dfig.add_scatter3d(x=locd_bj[0, :settings['nbpj_pts'], :settings['nbpj_pts']].flatten() + locp[0],
                           y=locd_bj[1, :settings['nbpj_pts'], :settings['nbpj_pts']].flatten() + locp[1],
                           z=locd_bj[2, :settings['nbpj_pts'], :settings['nbpj_pts']].flatten() + locp[2],
                           mode='markers',
                           name='BPJ Grid Projected Positions')
        dfig.show()
    else:
        fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=db(bg.refgrid.flatten()), range_color=[0, 140])
        fig.add_scatter(x=gx.flatten(), y=gy.flatten(), mode='markers')
        fig.show()

        fig = px.scatter(x=ngx.flatten(), y=ngy.flatten(), color=db(bg.refgrid.flatten()), opacity=.1)
        fig.add_scatter(x=gx.flatten(), y=gy.flatten(), marker={'color': db(bpj_res[0, ...]).flatten()}, mode='markers')
        fig.show()

    plt.figure('Doppler data')
    for t in range(nrx):
        plt.subplot(2, nrx, t * 2 + 1)
        plt.title('Generated')
        plt.imshow(np.fft.fftshift(db(np.fft.fft(test[:, :, t], axis=1)), axes=1))
        plt.axis('tight')
        plt.subplot(2, nrx, t * 2 + 2)
        plt.title('Post-BJ')
        plt.imshow(np.fft.fftshift(db(np.fft.fft(test_bj, axis=1)), axes=1))
        plt.axis('tight')

    plt.figure('Time Data')
    for t in range(nrx):
        plt.subplot(1, nrx, t + 1)
        plt.imshow(db(test[:, :, t]))
        plt.axis('tight')

    plt.figure('BPJ Reference Grid')
    plt.imshow(db(bg.refgrid), origin='lower', cmap='gray')
    plt.axis('tight')
    plt.axis('off')

    plt.figure('BPJ Sum Grid')
    plt.imshow(db(bpj_res), origin='lower', cmap='gray')
    plt.axis('tight')
    plt.axis('off')

    plt.figure('Elevation')
    plt.imshow(gz, origin='lower')
    plt.axis('tight')

    plt.figure('Waveforms')
    for t in range(nrx):
        plt.subplot(1, nrx, t + 1)
        plt.plot(np.fft.fftshift(np.fft.fftfreq(sim_up_fft_len, 1 / fs)),
                 np.fft.fftshift(db(settings['tx'][t]['chirp'][:, 0])), c='blue')
        plt.plot(np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs)),
                 np.fft.fftshift(db(settings['tx'][t]['mfilt'][:, 0])), c='orange')

    oval_times = rps[0].gpst[::len(rps[0].gpst) // 10]
    plat_height = 0 if settings['use_sdr_gps'] else rps[0].pos(oval_times)[2].mean()
    midrange = (rps[0].calcRanges(plat_height)[0] + rps[0].calcRanges(plat_height)[1]) / 2
    swath = np.sqrt(rps[0].calcRanges(plat_height)[1] ** 2 - plat_height ** 2) - \
            np.sqrt(rps[0].calcRanges(plat_height)[0] ** 2 - plat_height ** 2)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(locd[0, ...].ravel(), locd[1, ...].ravel(), c=angd[2, ...].ravel())
    for ot in oval_times:
        ax.add_patch(Ellipse((rps[0].boresight(ot).flatten() * midrange + rp.pos(ot))[:2],
                             midrange * rps[0].az_half_bw * 2, swath, angle=rps[0].pan(ot) / DTR, fill=False))

    plt.show()
