import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import nvtx
from .cuda_kernels import getMaxThreads, backproject, genRangeProfile, optimizeThreadBlocks
from .grid_helper import SDREnvironment
from .platform_helper import SDRPlatform, RadarPlatform
import cupy
from tqdm import tqdm
from sdrparse import load, SDRParse

_float = np.float32


def applyRangeRolloff(bpj_truedata):
    magnitude_data = np.sqrt(abs(bpj_truedata))
    brightness_raw = np.median(np.sqrt(abs(bpj_truedata)), axis=1)
    brightness_curve = np.polyval(np.polyfit(np.arange(bpj_truedata.shape[0]), brightness_raw, 4),
                                  np.arange(bpj_truedata.shape[1]))
    brightness_curve /= brightness_curve.max()
    brightness_curve = 1. / brightness_curve
    magnitude_data *= np.outer(np.ones(magnitude_data.shape[0]), brightness_curve)
    return magnitude_data


def getPulseTimeGen(data_t, idx_t, a_cpi_len, progress=True):
    if progress:
        for tidx, frames in tqdm(enumerate(idx_t[pos:pos + a_cpi_len] for pos in range(0, len(data_t), a_cpi_len)),
                                 total=len(data_t) // a_cpi_len + 1):
            ts = data_t[tidx * a_cpi_len + np.arange(len(frames))]
            yield ts, frames
    else:
        for tidx, frames in enumerate(idx_t[pos:pos + a_cpi_len] for pos in range(0, len(data_t), a_cpi_len)):
            ts = data_t[tidx * a_cpi_len + np.arange(len(frames))]
            yield ts, frames


def getRadarAndEnvironment(sdr_file: [SDRParse, str], a_channel: int = 0) -> tuple[SDREnvironment, SDRPlatform] | None:
    # Load SAR file into SDRParse object
    if isinstance(sdr_file, str):
        try:
            a_sdr = load(sdr_file, progress_tracker=True)
        except Exception as ex:
            print(ex)
            return None
    else:
        a_sdr = sdr_file
    # Load environment
    a_bg = SDREnvironment(a_sdr)

    # Load the platform
    a_rp = SDRPlatform(a_sdr, a_bg.ref, channel=a_channel)
    return a_bg, a_rp


def runBackproject(a_sdr: SDRParse, a_rp: SDRPlatform, a_bg: SDREnvironment, a_fdelay: float, a_plp: float,
                   a_upsample: int, a_channel: int, a_grid_width: int, a_grid_height: int, pixels_per_m: int,
                   a_cpi_len: int, a_rotate_grid: bool = False, a_debug: bool = False, a_poly_num: int = 0,
                   a_origin: tuple[float, float, float] = None) -> (
        tuple)[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    nbpj_pts = (int(a_grid_width * pixels_per_m), int(a_grid_height * pixels_per_m))
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        a_rp.getRadarParams(a_fdelay, a_plp, a_upsample))

    # Chirp and matched filter calculations
    bpj_wavelength = a_sdr.getBackprojectWavelength(a_channel)
    mfilt_gpu = cupy.array(a_sdr.genMatchedFilter(0, fft_len=fft_len), dtype=np.complex128)

    # Calculate out points on the ground
    gx, gy, gz = a_bg.getGrid(a_origin, a_grid_width, a_grid_height, *nbpj_pts, a_bg.heading if a_rotate_grid else 0)
    gx_gpu = cupy.array(gx, dtype=_float)
    gy_gpu = cupy.array(gy, dtype=_float)
    gz_gpu = cupy.array(gz, dtype=_float)

    if a_debug:
        pts_debug = cupy.zeros((3, *gx.shape), dtype=_float)
        angs_debug = cupy.zeros((3, *gx.shape), dtype=_float)
    else:
        pts_debug = cupy.zeros((1, 1), dtype=_float)
        angs_debug = cupy.zeros((1, 1), dtype=_float)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = (max(1, nbpj_pts[0] // threads_per_block[0] + 1), nbpj_pts[1] // threads_per_block[1] + 1)
    # rng_states = create_xoroshiro128p_states(triangles.shape[0], seed=10)

    # Run through loop to get data simulated
    r_debug_data = None
    print('Backprojecting...')
    # Data blocks for imaging
    r_bpj_final = np.zeros(nbpj_pts, dtype=np.complex128)
    for ts, frames in getPulseTimeGen(a_sdr[a_channel].pulse_time, a_sdr[a_channel].frame_num, a_cpi_len, True):
        tmp_len = len(ts)
        panrx_gpu = cupy.array(a_rp.pan(ts), dtype=_float)
        elrx_gpu = cupy.array(a_rp.tilt(ts), dtype=_float)
        posrx_gpu = cupy.array(a_rp.rxpos(ts), dtype=_float)
        postx_gpu = cupy.array(a_rp.txpos(ts), dtype=_float)
        bpj_grid = cupy.zeros(nbpj_pts, dtype=np.complex128)

        # Reset the grid for truth data
        rtdata = cupy.fft.fft(cupy.array(a_sdr.getPulses(frames, a_channel)[1],
                                         dtype=np.complex128), fft_len, axis=0) * mfilt_gpu[:, None]
        upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
        upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
        rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * a_upsample, :]
        cupy.cuda.Device().synchronize()

        backproject[bpg_bpj, threads_per_block](
            postx_gpu,
            posrx_gpu,
            gx_gpu,
            gy_gpu,
            gz_gpu,
            panrx_gpu,
            elrx_gpu,
            panrx_gpu,
            elrx_gpu,
            rtdata,
            bpj_grid,
            bpj_wavelength,
            near_range_s,
            a_rp.fs * a_upsample,
            a_rp.az_half_bw,
            a_rp.el_half_bw,
            a_poly_num,
            pts_debug,
            angs_debug,
            a_debug and ts[0] < a_rp.gpst.mean() <= ts[-1],
        )
        cupy.cuda.Device().synchronize()

        if (ts[0] < a_rp.gpst.mean() <= ts[-1]) and a_debug:
            r_debug_data = (a_rp.pos(ts[-1]).T, rtdata.get(), angs_debug.get(), pts_debug.get())

        r_bpj_final += bpj_grid.get()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del rtdata
    del upsample_data
    del bpj_grid
    del gx_gpu
    del gy_gpu
    del gz_gpu

    return r_bpj_final, r_debug_data


def backprojectPulseSet(pulse_data: np.ndarray, panrx: np.ndarray, elrx: np.ndarray, posrx: np.ndarray,
                        postx: np.ndarray, gz: np.ndarray, wavelength: float, near_range_s: float, upsample_fs: float,
                        az_half_bw: float, el_half_bw: float, gx: np.ndarray = None,
                        gy: np.ndarray = None, a_poly_num: int = 0) -> np.ndarray:
    nbpj_pts = gz.shape

    # Calculate out points on the ground
    gx_gpu = cuda.to_device(gx)
    gy_gpu = cuda.to_device(gy)
    gz_gpu = cuda.to_device(gz)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = optimizeThreadBlocks(threads_per_block, nbpj_pts)
    rbj = np.zeros(nbpj_pts, dtype=np.complex128)

    with cuda.pinned(panrx, elrx, posrx, postx, rbj, pulse_data):
        panrx_gpu = cuda.to_device(panrx)
        elrx_gpu = cuda.to_device(elrx)
        posrx_gpu = cuda.to_device(posrx)
        postx_gpu = cuda.to_device(postx)
        bpj_grid = cuda.to_device(rbj)
        rtdata = cuda.to_device(pulse_data)
        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, panrx_gpu,
                                                elrx_gpu, panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                                wavelength, near_range_s, upsample_fs, az_half_bw,
                                                el_half_bw, a_poly_num)
        rbj = bpj_grid.copy_to_host()
    cuda.synchronize()
    del gx_gpu, gy_gpu, gz_gpu, panrx_gpu, elrx_gpu, posrx_gpu, postx_gpu, bpj_grid, rtdata

    return rbj

@nvtx.annotate(color='red')
def backprojectPulseStream(pulse_data: list[np.ndarray], panrx: list[np.ndarray], elrx: list[np.ndarray],
                           posrx: list[np.ndarray], postx: list[np.ndarray], gz: np.ndarray, wavelength: float,
                           near_range_s: float, upsample_fs: float, az_half_bw: float, el_half_bw: float,
                           gx: np.ndarray = None, gy: np.ndarray = None, a_poly_num: int = 0,
                           streams: list[cuda.stream]=None) -> np.ndarray:
    nbpj_pts = gz.shape

    # Calculate out points on the ground
    with cuda.defer_cleanup():
        gx_gpu = cuda.to_device(gx)
        gy_gpu = cuda.to_device(gy)
        gz_gpu = cuda.to_device(gz)

        # GPU device calculations
        threads_per_block = getMaxThreads()
        bpg_bpj = optimizeThreadBlocks(threads_per_block, nbpj_pts)

        # Run through loop to get data simulated
        # Data blocks for imaging
        r_bpj = [np.zeros(nbpj_pts, dtype=np.complex128) for _ in panrx]
        for az, el, prx, ptx, data, stream, rbj in zip(panrx, elrx, posrx, postx, pulse_data, streams, r_bpj):
            with cuda.pinned(data, rbj):
                panrx_gpu = cuda.to_device(az, stream=stream)
                elrx_gpu = cuda.to_device(el, stream=stream)
                posrx_gpu = cuda.to_device(prx, stream=stream)
                postx_gpu = cuda.to_device(ptx, stream=stream)
                bpj_grid = cuda.to_device(rbj, stream=stream)
                rtdata = cuda.to_device(data, stream=stream)
                backproject[bpg_bpj, threads_per_block, stream](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, panrx_gpu,
                                                                elrx_gpu, panrx_gpu, elrx_gpu, rtdata, bpj_grid,
                                                                wavelength, near_range_s, upsample_fs, az_half_bw,
                                                                el_half_bw, a_poly_num)
                bpj_grid.copy_to_host(rbj, stream=stream)
                del panrx_gpu, elrx_gpu, posrx_gpu, postx_gpu, bpj_grid, rtdata
        cuda.synchronize()
    del gx_gpu, gy_gpu, gz_gpu

    return np.sum(r_bpj, axis=0)

def genSimPulseData(a_rp: RadarPlatform,
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
                    a_sdr: SDRParse = None,
                    a_channel: int = 0,
                    a_transmit_power: float = 100.,
                    a_ant_gain: float = 20.,
                    a_rotate_grid: bool = False,
                    a_debug: bool = False,
                    a_fft_len: int = None,
                    a_noise_level: float = 0.,
                    a_origin: tuple[float, float, float] = None) -> (
        tuple)[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    nbpj_pts = (int(a_grid_width * pixels_per_m), int(a_grid_height * pixels_per_m))
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, p_fft_len, _ = (
        a_rp.getRadarParams(a_fdelay, a_plp, a_upsample))
    fft_len = p_fft_len if a_fft_len is None else a_fft_len
    up_fft_len = fft_len * a_upsample

    # Chirp and matched filter calculations
    bpj_wavelength = a_sdr.getBackprojectWavelength(a_channel) if a_sdr else a_bpj_wavelength

    # Calculate out points on the ground
    gx, gy, gz = a_bg.getGrid(a_origin, a_grid_width, a_grid_height, *nbpj_pts, a_bg.heading if a_rotate_grid else 0)
    rg = a_bg.getRefGrid(a_origin, a_grid_width, a_grid_height, *nbpj_pts, a_bg.heading if a_rotate_grid else 0)
    gx_gpu = cupy.array(gx, dtype=_float)
    gy_gpu = cupy.array(gy, dtype=_float)
    gz_gpu = cupy.array(gz, dtype=_float)
    refgrid_gpu = cupy.array(rg, dtype=_float)

    if a_debug:
        pts_debug = cupy.zeros((3, *gx.shape), dtype=_float)
        angs_debug = cupy.zeros((3, *gx.shape), dtype=_float)
    else:
        pts_debug = cupy.zeros((1, 1), dtype=_float)
        angs_debug = cupy.zeros((1, 1), dtype=_float)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = (max(1, nbpj_pts[0] // threads_per_block[0] + 1), nbpj_pts[1] // threads_per_block[1] + 1)

    receive_power_scale = (a_transmit_power / .01 *
                           (10 ** (a_ant_gain / 20)) ** 2
                           * bpj_wavelength ** 2 / (4 * np.pi) ** 3)
    noise_level = 10 ** (a_noise_level / 20) / np.sqrt(2) / a_upsample
    chirp_gpu = cupy.array(a_chirp, dtype=np.complex128)
    rng_states = create_xoroshiro128p_states(threads_per_block[0] * bpg_bpj[0], seed=1)

    # Run through loop to get data simulated
    dt = a_sdr[a_channel].pulse_time if a_sdr else a_pulse_times
    frame_idx = a_sdr[a_channel].frame_num if a_sdr else np.arange(len(dt))
    for ts, frames in getPulseTimeGen(dt, frame_idx, a_cpi_len, True):
        tmp_len = len(ts)
        panrx_gpu = cupy.array(a_rp.pan(ts), dtype=_float)
        elrx_gpu = cupy.array(a_rp.tilt(ts), dtype=_float)
        posrx_gpu = cupy.array(a_rp.rxpos(ts), dtype=_float)
        postx_gpu = cupy.array(a_rp.txpos(ts), dtype=_float)
        pd_r = cupy.zeros((nsam, tmp_len), dtype=_float)
        pd_i = cupy.zeros((nsam, tmp_len), dtype=_float)

        genRangeProfile[bpg_bpj, threads_per_block](gx_gpu, gy_gpu, gz_gpu, refgrid_gpu, postx_gpu, posrx_gpu,
                                                    panrx_gpu, elrx_gpu, panrx_gpu, elrx_gpu, pd_r, pd_i,
                                                    rng_states, pts_debug, angs_debug, bpj_wavelength,
                                                    near_range_s, a_rp.fs, a_rp.az_half_bw, a_rp.el_half_bw,
                                                    receive_power_scale, 1, a_debug)

        pdata = pd_r + 1j * pd_i
        rtdata = cupy.fft.fft(pdata, fft_len, axis=0) * chirp_gpu[:, None]
        upsample_data = cupy.array(np.random.normal(0, noise_level, (up_fft_len, tmp_len)) +
                                   1j * np.random.normal(0, noise_level, (up_fft_len, tmp_len)),
                                   dtype=np.complex128)
        upsample_data[:fft_len // 2, :] += rtdata[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] += rtdata[-fft_len // 2:, :]
        # rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * a_upsample, :]
        cupy.cuda.Device().synchronize()
        yield upsample_data.get()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del rtdata
    del upsample_data
    del gx_gpu
    del gy_gpu
    del gz_gpu
    del chirp_gpu
