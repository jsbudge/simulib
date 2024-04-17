import numpy as np
from simulation_functions import llh2enu, db
from cuda_kernels import getMaxThreads, backproject
from grid_helper import SDREnvironment
from platform_helper import SDRPlatform
import cupy as cupy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
from SDRParsing import load, SDRParse

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'


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
            a_sdr = load(bg_file, progress_tracker=True)
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
    gx_gpu = cupy.array(gx, dtype=np.float64)
    gy_gpu = cupy.array(gy, dtype=np.float64)
    gz_gpu = cupy.array(gz, dtype=np.float64)

    if a_debug:
        pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
        angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    else:
        pts_debug = cupy.zeros((1, 1), dtype=np.float64)
        angs_debug = cupy.zeros((1, 1), dtype=np.float64)

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
        panrx_gpu = cupy.array(a_rp.pan(ts), dtype=np.float64)
        elrx_gpu = cupy.array(a_rp.tilt(ts), dtype=np.float64)
        posrx_gpu = cupy.array(a_rp.rxpos(ts), dtype=np.float64)
        postx_gpu = cupy.array(a_rp.txpos(ts), dtype=np.float64)
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


c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

if __name__ == '__main__':
    # This is the file used to backproject data
    # bg_file = '/data5/SAR_DATA/2021/05052021/SAR_05052021_112647.sar'
    # bg_file = '/data5/SAR_DATA/2022/09082022/SAR_09082022_131237.sar'
    # bg_file = '/data5/SAR_DATA/2022/Redstone/SAR_08122022_170753.sar'
    # bg_file = '/data6/SAR_DATA/2023/06202023/SAR_06202023_135617.sar'
    # bg_file = '/data6/Tower_Redo_Again/tower_redo_SAR_03292023_120731.sar'
    # bg_file = '/data5/SAR_DATA/2022/09272022/SAR_09272022_103053.sar'
    # bg_file = '/data5/SAR_DATA/2019/08072019/SAR_08072019_100120.sar'
    bg_file = '/data6/SAR_DATA/2024/04092024/SAR_04092024_133843.sar'
    # bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_122801.sar'
    # bg_file = '/data6/SAR_DATA/2023/07132023/SAR_07132023_123050.sar'
    upsample = 4
    poly_num = 0
    rotate_grid = True
    use_ecef = True
    ipr_mode = False
    cpi_len = 256
    plp = 0
    partial_pulse_percent = .2
    debug = True
    pts_per_m = 5
    grid_width = 20
    grid_height = 20
    channel = 0
    fdelay = 1.5
    origin = (30.562588, -86.436398, 47)

    print('Loading SDR file...')
    sdr = load(bg_file, progress_tracker=True, use_jump_correction=False)
    print('Generating platform...', end='')
    bg, rp = getRadarAndEnvironment(bg_file, channel)
    print('Done.')

    bpj_final, debug_data = runBackproject(sdr, rp, bg, fdelay, plp, upsample, channel, grid_width, grid_height,
                                           pts_per_m, cpi_len, a_rotate_grid=rotate_grid, a_debug=debug,
                                           a_poly_num=poly_num, a_origin=origin)

    mag_data = applyRangeRolloff(bpj_final)

    if debug_data is not None:
        plt.figure('Doppler data')
        plt.imshow(np.fft.fftshift(db(np.fft.fft(debug_data[1], axis=1)), axes=1),
                   extent=(-sdr[channel].prf / 2, sdr[channel].prf / 2,
                           rp.calcRanges(fdelay, partial_pulse_percent)[1],
                           rp.calcRanges(fdelay, partial_pulse_percent)[0]))
        plt.axis('tight')

    plt.figure('IMSHOW backprojected data')
    plt.imshow(db(mag_data), origin='lower')
    plt.axis('tight')

    try:
        if (mag_data.shape[0] * mag_data.shape[1]) < 400 ** 2:
            cx, cy, cz = bg.getGrid(origin, grid_width, grid_height, *mag_data.shape)

            fig = px.scatter_3d(x=cx.flatten(), y=cy.flatten(), z=cz.flatten())
            fig.show()

        plt.figure('IMSHOW truth data')
        plt.imshow(db(bg.refgrid), origin='lower')
        plt.axis('tight')
    except Exception as e:
        print(f'Error in generating background image: {e}')

    from sklearn.preprocessing import QuantileTransformer

    # mag_data = np.sqrt(abs(sdr.loadASI(sdr.files['asi'][0])))
    nbits = 256
    plot_data_init = QuantileTransformer(output_distribution='normal').fit(
        mag_data[mag_data > 0].reshape(-1, 1)).transform(mag_data.reshape(-1, 1)).reshape(mag_data.shape)
    plot_data = plot_data_init
    max_bin = 3
    hist_counts, hist_bins = \
        np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    while hist_counts[-1] == 0:
        max_bin -= .2
        hist_counts, hist_bins = \
            np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    scaled_data = np.digitize(plot_data_init, hist_bins)

    # px.imshow(db(mag_data), color_continuous_scale=px.colors.sequential.gray).show()
    px.imshow(np.fliplr(scaled_data), color_continuous_scale=px.colors.sequential.gray, zmin=0, zmax=nbits).show()
    plt.figure('Image Histogram')
    plt.plot(hist_bins[1:], hist_counts)

    plt.figure('GPS check')
    rpos = rp.pos(rp.gpst)
    e, n, u = llh2enu(sdr.gps_data['lat'].values, sdr.gps_data['lon'].values, sdr.gps_data['alt'].values, bg.ref)
    plt.plot(rpos[:, 2] - u)
    plt.show()
