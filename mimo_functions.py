import numpy as np
from numba.cuda.random import create_xoroshiro128p_states
from simulib import getMaxThreads, getPulseTimeGen, SDREnvironment, genRangeProfile, azelToVec
from simulib.platform_helper import RadarPlatform
import cupy as cupy
from scipy.signal.windows import taylor


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


def genChannels(n_tx, n_rx, tx_pos, rx_pos, plat_e, plat_n, plat_u, plat_r, plat_p, plat_y,
                gpst, gimbal, goff, grot, dep_ang, az_half_bw, el_half_bw, fs, platform,
                platform_kwargs: dict = None):
    if platform_kwargs is None:
        platform_kwargs = {}
    rps = []
    rx_array = []
    vx_perm = [(n, q) for q in range(n_tx) for n in range(n_rx)]
    for tx, rx in vx_perm:
        txpos = np.array(tx_pos[tx])
        rxpos = np.array(rx_pos[rx])
        vx_pos = rxpos + txpos
        rps.append(
            platform(e=plat_e, n=plat_n, u=plat_u, r=plat_r, p=plat_p, y=plat_y, t=gpst, tx_offset=txpos,
                     rx_offset=rxpos, gimbal=gimbal, gimbal_offset=goff, gimbal_rotations=grot, dep_angle=dep_ang,
                     squint_angle=0., az_bw=az_half_bw * 2 / DTR, el_bw=el_half_bw * 2 / DTR,
                     fs=fs, tx_num=tx, rx_num=rx, **platform_kwargs))
        rx_array.append(vx_pos)
    rpref = platform(e=plat_e, n=plat_n, u=plat_u, r=plat_r, p=plat_p, y=plat_y, t=gpst,
                     tx_offset=np.array([0., 0., 0.]), rx_offset=np.array([0., 0., 0.]), gimbal=gimbal,
                     gimbal_offset=goff, gimbal_rotations=grot, dep_angle=dep_ang,
                     squint_angle=0., az_bw=az_half_bw * 2 / DTR, el_bw=el_half_bw * 2 / DTR,
                     fs=fs, **platform_kwargs)
    return rpref, rps, np.array(rx_array)


def genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, fft_len, use_window=True):
    # Get Taylor window of appropriate length and shift it to the aliased frequency of fc
    taywin = int(bwidth / fs * fft_len)
    taywin = taywin + 1 if taywin % 2 != 0 else taywin
    taytay = np.zeros(fft_len, dtype=np.complex128)
    twin_tmp = taylor(taywin, nbar=10, sll=60)
    taytay[:taywin // 2] = twin_tmp[taywin // 2:]
    taytay[-taywin // 2:] = twin_tmp[:taywin // 2]
    alias_shift = int(fft_len + (fc % (fs / 2) - fs / 2) / fs * fft_len)
    taytay = np.roll(taytay, alias_shift).astype(np.complex128)

    # Chirps and Mfilts for each channel
    chirps = []
    mfilt = []
    for rp in rps:
        mf = waves[rp.tx_num].conj() * taytay if use_window else waves[rp.tx_num].conj()
        chirps.append(cupy.array(waves[rp.tx_num], dtype=np.complex128))
        mfilt.append(cupy.array(mf, dtype=np.complex128))
    return taytay, chirps, mfilt


def applyRangeRolloff(bpj_truedata):
    mag_data = np.sqrt(abs(bpj_truedata))
    brightness_raw = np.median(np.sqrt(abs(bpj_truedata)), axis=1)
    brightness_curve = np.polyval(np.polyfit(np.arange(bpj_truedata.shape[0]), brightness_raw, 4),
                                  np.arange(bpj_truedata.shape[1]))
    brightness_curve /= brightness_curve.max()
    brightness_curve = 1. / brightness_curve
    mag_data *= np.outer(np.ones(mag_data.shape[0]), brightness_curve)
    return mag_data


def genSimPulseData(a_rp: RadarPlatform,
                    a_rps: list[RadarPlatform],
                    a_bg: SDREnvironment,
                    a_fdelay: float = 0.,
                    a_plp: float = .5,
                    a_upsample: int = 1,
                    a_grid_width: int = 100,
                    a_grid_height: int = 100,
                    pixels_per_m: int = 1,
                    a_cpi_len: int = 32,
                    a_chirp: np.array = None,
                    a_bpj_wavelength: float = 1.,
                    a_pulse_times: np.array = None,
                    a_transmit_power: list | float = 100.,
                    a_ant_gain: list | float = 20.,
                    a_rotate_grid: bool = False,
                    a_debug: bool = False,
                    a_fft_len: int = None,
                    a_noise_level: float = -300.,
                    a_origin: tuple[float, float, float] = None,
                    a_update_chirp: bool = True):
    nbpj_pts = (int(a_grid_width * pixels_per_m), int(a_grid_height * pixels_per_m))
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, p_fft_len, _ = (
        a_rp.getRadarParams(a_fdelay, a_plp, a_upsample))
    fft_len = p_fft_len if a_fft_len is None else a_fft_len

    # Calculate out points on the ground
    gx, gy, gz = a_bg.getGrid(a_origin, a_grid_width, a_grid_height, *nbpj_pts, a_bg.heading if a_rotate_grid else 0)
    rg = a_bg.getRefGrid(a_origin, a_grid_width, a_grid_height, *nbpj_pts, a_bg.heading if a_rotate_grid else 0)
    gx_gpu = cupy.array(gx, dtype=np.float64)
    gy_gpu = cupy.array(gy, dtype=np.float64)
    gz_gpu = cupy.array(gz, dtype=np.float64)
    refgrid_gpu = cupy.array(rg, dtype=np.float64)

    n_rx = len(np.unique([rp.rx_num for rp in a_rps]))

    if a_debug:
        pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
        angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    else:
        pts_debug = cupy.zeros((1, 1), dtype=np.float64)
        angs_debug = cupy.zeros((1, 1), dtype=np.float64)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = (max(1, nbpj_pts[0] // threads_per_block[0] + 1), nbpj_pts[1] // threads_per_block[1] + 1)

    receive_power_scale = (a_transmit_power / .01 *
                           (10 ** (a_ant_gain / 20)) ** 2
                           * a_bpj_wavelength ** 2 / (4 * np.pi) ** 3)
    noise_level = 10 ** (a_noise_level / 20) / np.sqrt(2) / a_upsample
    rng_states = create_xoroshiro128p_states(threads_per_block[0] * bpg_bpj[0], seed=1)

    # Only if updating the chirp; requires a while loop instead of for
    if a_update_chirp:
        yield

    # Run through loop to get data simulated
    dt = a_pulse_times
    frame_idx = np.arange(len(dt))
    for ts, frames in getPulseTimeGen(dt, frame_idx, a_cpi_len, True):
        tmp_len = len(ts)
        ret_data = (np.random.normal(0, noise_level, (n_rx, tmp_len, fft_len)) +
                    1j * np.random.normal(0, noise_level, (n_rx, tmp_len, fft_len)))
        # If we're updating the chirp, yield this to the send function
        for ch_idx, curr_rp in enumerate(a_rps):
            panrx_gpu = cupy.array(curr_rp.pan(ts), dtype=np.float64)
            elrx_gpu = cupy.array(curr_rp.tilt(ts), dtype=np.float64)
            posrx_gpu = cupy.array(curr_rp.rxpos(ts), dtype=np.float64)
            postx_gpu = cupy.array(curr_rp.txpos(ts), dtype=np.float64)
            pd_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
            pd_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)

            genRangeProfile[bpg_bpj, threads_per_block](gx_gpu, gy_gpu, gz_gpu, refgrid_gpu, postx_gpu, posrx_gpu,
                                                        panrx_gpu, elrx_gpu, panrx_gpu, elrx_gpu, pd_r, pd_i,
                                                        rng_states, pts_debug, angs_debug, a_bpj_wavelength,
                                                        near_range_s, curr_rp.fs, curr_rp.az_half_bw, curr_rp.el_half_bw,
                                                        receive_power_scale, 1, a_debug)

            pdata = pd_r + 1j * pd_i
            rtdata = cupy.fft.fft(pdata, fft_len, axis=0) * a_chirp[ch_idx][:, None]
            cupy.cuda.Device().synchronize()
            ret_data[curr_rp.rx_num, ...] += rtdata.get().T
        # Yielding the chirp here so it can be changed with the send method down the line
        if a_update_chirp:
            # Works backward from normal assignment in send function
            a_chirp = yield ret_data
        else:
            yield ret_data


    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del rtdata
    del gx_gpu
    del gy_gpu
    del gz_gpu


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_converter.SDRParsing import load
    from simulib import llh2enu, db, genPulse

    bg_fnme = '/home/jeff/SDR_DATA/ARCHIVE/05212024/SAR_05212024_123051.sar'
    origin = (40.141223, -111.699440, 1378)
    fs = 500e6
    bwidth = 240e6
    fc = 9.6e9

    gimbal_offset = np.array([0, 0, 2.])
    gimbal_rotation = np.array([0., 0., np.pi / 2])
    gps_times = np.arange(0, 5, .01)
    e = np.linspace(388., -875, len(gps_times))
    n = np.linspace(373.84, -874.26, len(gps_times))
    u = np.linspace(1500, 1524., len(gps_times))
    r = np.zeros_like(e)
    p = np.zeros_like(e)
    y = np.zeros_like(e) + 3.9355

    bg = SDREnvironment(load(bg_fnme, progress_tracker=True, use_jump_correction=False), origin=origin)
    origin_enu = llh2enu(*origin, bg.ref)
    gim_pan = np.zeros_like(e)
    gim_el = np.zeros_like(gim_pan) + np.arccos((u - origin_enu[2]) /
                                                np.sqrt((e - origin_enu[0])**2 +
                                                        (n - origin_enu[1])**2 +
                                                        (u - origin_enu[2])**2))
    gimbal = np.array([gim_pan, gim_el]).T

    tx_array = np.array([[1., 0, 0], [-1., 0, 0]])
    rx_array = np.array([[0, 1., 0], [0, -1., 0]])
    rpi, rps, vx_array = genChannels(2, 2, tx_array, rx_array, e, n, u, r, p, y, gps_times, gimbal,
                                     gimbal_offset, gimbal_rotation, 30, 5 * DTR, 5 * DTR, fs, RadarPlatform)
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, p_fft_len, _ = (
        rpi.getRadarParams(u.mean(), .5, 1))

    ublock = -azelToVec(np.pi / 2, 0)
    fine_ucavec = np.exp(1j * 2 * np.pi / (c0 / fc) * vx_array.dot(ublock))

    waves = np.array([genPulse(np.linspace(0, 1, 10),
                               np.linspace(0, 1, 10), nr, fs, fc, bwidth),
                      genPulse(np.linspace(0, 1, 10),
                               np.linspace(1, 0, 10), nr, fs, fc, bwidth)])
    waves = np.fft.fft(waves, p_fft_len, axis=1)
    second_waves = np.fft.fft(np.random.random(waves.shape) + 1j * np.random.random(waves.shape), axis=1)

    win, chirp, mfilt = genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, p_fft_len)

    collect_times = np.linspace(0, 5, 1288 * 5)

    data_gen = genSimPulseData(rpi, rps, bg, u.mean(), a_grid_width=200, a_grid_height=200,
                               a_pulse_times=collect_times, a_chirp=chirp, a_cpi_len=128,
                               a_bpj_wavelength=c0 / fc, a_origin=origin, a_noise_level=-300, a_debug=True)

    plt.ion()
    for idx, (chirp, pdata) in enumerate(data_gen):
        try:
            compressed_data = np.sum([np.fft.ifft(pdata[rp.tx_num] * mfilt[ch_idx].get(), axis=1)[:, :nsam] *
                                      fine_ucavec[ch_idx] for ch_idx, rp in enumerate(rps)], axis=0)
            compressed_data = np.fft.fft(compressed_data, axis=0)
            plt.gca().cla()
            plt.imshow(db(compressed_data))
            plt.axis('tight')
            plt.title(f'CPI {idx}')
            plt.draw()
            plt.pause(0.1)
            if idx == 17:
                # win, sec_chirp, mfilt = genChirpAndMatchedFilters(second_waves, rps, bwidth, fs, fc, p_fft_len)
                data_gen.send((chirp, pdata))
        except KeyboardInterrupt:
            break
'''def multiplier():
    print("top of generator")
    m = yield # nothing to yield the first time, just a value we get
    print("before loop, m =", m)
    while True:
        print("top of loop, m =", m)
        m = yield m * 2, m * 3         # we always care about the value we're sent
        print("bottom of loop, m =", m)

print("calling generator")
it = multiplier()

print("calling next")
next(it)   # this is equivalent to it.send(None)

print("sending 10")
print(it.send(10))

print("sending 20")
print(it.send(20))

print("sending 100")
print(it.send(100))'''