import numpy as np
from numba.cuda.random import create_xoroshiro128p_states
from simulation_functions import getElevation, llh2enu, db, enu2llh, azelToVec, genPulse
from cuda_kernels import getMaxThreads, backproject, genRangeProfile
from grid_helper import SDREnvironment
from platform_helper import SDRPlatform
from mimo_functions import genChirpAndMatchedFilters, genChannels, applyRangeRolloff
import cupy as cupy
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm import tqdm
import yaml
from sklearn.preprocessing import QuantileTransformer

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


if __name__ == '__main__':
    from SDRParsing import load
    with open('./data_generator.yaml') as y:
        settings = yaml.safe_load(y.read())

    nbpj_pts = (
    int(settings['grid_height'] * settings['pts_per_m']), int(settings['grid_width'] * settings['pts_per_m']))

    print('Loading SDR file...')
    # This pickle should have an ASI file attached and use_jump_correction=False
    sdr = load(settings['bg_file'])

    if settings['origin'] is None:
        try:
            settings['origin'] = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                                  getElevation(sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX']))
            heading = sdr.gim.initial_course_angle
        except TypeError:
            heading = sdr.gim.initial_course_angle
            pt = (sdr.gps_data['lat'].values[0], sdr.gps_data['lon'].values[0])
            alt = getElevation(*pt)
            nrange = ((sdr[settings['channel']].receive_on_TAC - sdr[settings['channel']].transmit_on_TAC) / TAC -
                      sdr[settings['channel']].pulse_length_S * settings['partial_pulse_percent']) * c0 / 2
            frange = ((sdr[settings['channel']].receive_off_TAC - sdr[settings['channel']].transmit_on_TAC) / TAC -
                      sdr[settings['channel']].pulse_length_S * settings['partial_pulse_percent']) * c0 / 2
            mrange = (nrange + frange) / 2
            settings['origin'] = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                                         (pt[0], pt[1], alt))

    bg = SDREnvironment(sdr)

    if settings['ipr_mode']:
        print('IPR mode...', end='')
        dx_shift = np.inf
        iters = 0
        while dx_shift > .5 and iters < 10:
            oe, on, ou = llh2enu(*settings['origin'], bg.ref)
            ix, iy = bg.getIndex(oe, on).astype(int)
            minigrid = bg.refgrid[ix - int(50 / bg.rps):ix + int(50 / bg.rps),
                       iy - int(50 / bg.cps):iy + int(50 / bg.cps)]
            nx, ny = np.where(minigrid == minigrid.max())
            settings['origin'] = enu2llh(*bg.getPos(nx[0] + ix - int(50 / bg.rps), ny[0] + iy - int(50 / bg.cps)), 0,
                                         bg.ref)
            dx_shift = np.sqrt(((nx[0] - 50 / bg.rps) * bg.rps) ** 2 + ((ny[0] - 50 / bg.cps) * bg.cps) ** 2)
            iters += 1
        print(f'Origin set to {settings["origin"]}')
    ref_llh = bg.ref

    # Generate a platform
    print('Generating platform...', end='')

    rpi = SDRPlatform(sdr, ref_llh, channel=settings['channel'])
    plat_e, plat_n, plat_u = rpi.pos(rpi.gpst).T
    plat_r, plat_p, plat_y = rpi.att(rpi.gpst).T
    gimbal = np.array([rpi.pan(rpi.gpst), rpi.tilt(rpi.gpst)]).T
    goff = np.array(
        [sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset])
    grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR])

    rpref, rps, rx_array = genChannels(settings['antenna_params']['n_tx'], settings['antenna_params']['n_rx'],
                                       settings['antenna_params']['tx_pos'], settings['antenna_params']['rx_pos'],
                                       plat_e, plat_n, plat_u, plat_r, plat_p, plat_y, rpi.gpst, gimbal, goff, grot,
                                       rpi.dep_ang, rpi.az_half_bw, rpi.el_half_bw, rpi.fs)

    # Get reference data
    fs = sdr[settings['channel']].fs
    bwidth = sdr[settings['channel']].bw
    fc = sdr[settings['channel']].fc
    print('Done.')

    # Generate values needed for backprojection
    print('Calculating grid parameters...')
    # General calculations for slant ranges, etc.
    # plat_height = rp.pos(rp.gpst)[2, :].mean()
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rpi.getRadarParams(settings['fdelay'], settings['plp'], settings['upsample']))
    rbins_gpu = cupy.array(ranges, dtype=np.float64)

    # Chirp and matched filter calculations
    bpj_wavelength = c0 / fc

    endpoints = [(1, 0) if rp.tx_num == 0 else (0, 1) for rp in rps]
    waves = np.array([np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(*e, 10), nr, fs, fc,
                               bwidth), fft_len) for e in endpoints])
    taytay, chirps, mfilt = genChirpAndMatchedFilters(waves, rps, bwidth, fs, fc, fft_len, settings['cpi_len'])

    bg.resampleGrid(settings['origin'], settings['grid_width'], settings['grid_height'], *nbpj_pts,
                    bg.heading if settings['rotate_grid'] else 0)

    # This replaces the ASI background with a custom image
    '''bg_image = imageio.imread('/data6/Jeff_Backup/Pictures/josh.png').sum(axis=2)
    bg_image = RectBivariateSpline(np.arange(bg_image.shape[0]), np.arange(bg_image.shape[1]), bg_image)(
        np.linspace(0, bg_image.shape[0], nbpj_pts[0]), np.linspace(0, bg_image.shape[1], nbpj_pts[1])) / 750'''
    '''bg_image = np.zeros_like(bg.refgrid)
    bg_image[bg_image.shape[0] // 2, bg_image.shape[1] // 2] = 10'''
    # bg._refgrid = bg_image

    # Constant part of the radar equation
    receive_power_scale = (settings['antenna_params']['transmit_power'][0] / .01 *
                           (10 ** (settings['antenna_params']['gain'][0] / 20)) ** 2
                           * bpj_wavelength ** 2 / (4 * np.pi) ** 3)
    noise_level = 10 ** (settings['noise_level'] / 20) / np.sqrt(2) / settings['upsample']

    # Calculate out points on the ground
    gx, gy, gz = bg.getGrid(settings['origin'], settings['grid_width'], settings['grid_height'], *nbpj_pts,
                            bg.heading if settings['rotate_grid'] else 0)
    gx_gpu = cupy.array(gx, dtype=np.float64)
    gy_gpu = cupy.array(gy, dtype=np.float64)
    gz_gpu = cupy.array(gz, dtype=np.float64)
    refgrid_gpu = cupy.array(bg.refgrid, dtype=np.float64)

    if settings['debug']:
        pts_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
        angs_debug = cupy.zeros((3, *gx.shape), dtype=np.float64)
    else:
        pts_debug = cupy.zeros((1, 1), dtype=np.float64)
        angs_debug = cupy.zeros((1, 1), dtype=np.float64)

    # GPU device calculations
    threads_per_block = getMaxThreads(settings['pts_per_tri'])
    bpg_bpj = (max(1, (nbpj_pts[0]) // threads_per_block[0] + 1), (nbpj_pts[1]) // threads_per_block[1] + 1,
               settings['pts_per_tri'])

    rng_states = create_xoroshiro128p_states(threads_per_block[0] * bpg_bpj[0], seed=1)

    # Get pointing vector for MIMO consolidation
    ublock = -azelToVec(np.pi / 2, 0)
    fine_ucavec = np.exp(1j * 2 * np.pi * sdr[0].fc / c0 * rx_array.dot(ublock))

    # Run through loop to get data simulated
    data_t = sdr[settings['channel']].pulse_time
    idx_t = sdr[settings['channel']].frame_num
    test = None
    print('Backprojecting...')
    pulse_pos = 0
    # Data blocks for imaging
    bpj_truedata = np.zeros(nbpj_pts, dtype=np.complex128)
    for tidx, frames in tqdm(
            enumerate(idx_t[pos:pos + settings['cpi_len']] for pos in range(0, len(data_t), settings['cpi_len'])),
            total=len(data_t) // settings['cpi_len'] + 1):
        ts = data_t[tidx * settings['cpi_len'] + np.arange(len(frames))]
        tmp_len = len(ts)
        # Pan and Tilt are shared by each channel, antennas are all facing the same way
        panrx = rpi.pan(ts)
        elrx = rpi.tilt(ts)
        panrx_gpu = cupy.array(panrx, dtype=np.float64)
        elrx_gpu = cupy.array(elrx, dtype=np.float64)
        bpj_grid = cupy.zeros(nbpj_pts, dtype=np.complex128)

        beamform_data = cupy.zeros((nsam * settings['upsample'], tmp_len), dtype=np.complex128)

        for ch_idx, rp in enumerate(rps):
            posrx = rp.rxpos(ts)
            postx = rp.txpos(ts)
            posrx_gpu = cupy.array(posrx, dtype=np.float64)
            postx_gpu = cupy.array(postx, dtype=np.float64)

            pd_r = cupy.zeros((nsam, tmp_len), dtype=np.float64)
            pd_i = cupy.zeros((nsam, tmp_len), dtype=np.float64)

            genRangeProfile[bpg_bpj, threads_per_block](gx_gpu, gy_gpu, gz_gpu, refgrid_gpu, postx_gpu, posrx_gpu,
                                                        panrx_gpu, elrx_gpu, panrx_gpu, elrx_gpu, pd_r, pd_i,
                                                        rng_states, pts_debug, angs_debug, bpj_wavelength,
                                                        near_range_s, rpref.fs, rpref.az_half_bw, rpref.el_half_bw,
                                                        receive_power_scale, 1, settings['debug'])

            pdata = pd_r + 1j * pd_i
            rtdata = cupy.fft.fft(pdata, fft_len, axis=0) * chirps[ch_idx][:, None] * mfilt[ch_idx][:, None]
            upsample_data = cupy.array(np.random.normal(0, noise_level, (up_fft_len, tmp_len)) +
                                       1j * np.random.normal(0, noise_level, (up_fft_len, tmp_len)),
                                       dtype=np.complex128)
            upsample_data[:fft_len // 2, :] += rtdata[:fft_len // 2, :]
            upsample_data[-fft_len // 2:, :] += rtdata[-fft_len // 2:, :]
            rtdata = cupy.fft.ifft(upsample_data, axis=0)[:nsam * settings['upsample'], :]
            cupy.cuda.Device().synchronize()

            # This is equivalent to a dot product
            beamform_data += rtdata * fine_ucavec[ch_idx]
            # beamform_data = rtdata
            del pd_r
            del pd_i
        posrx_gpu = cupy.array(rpref.rxpos(ts), dtype=np.float64)
        postx_gpu = cupy.array(rpref.txpos(ts), dtype=np.float64)

        # Backprojection only for beamformed final data
        backproject[bpg_bpj, threads_per_block](postx_gpu, posrx_gpu, gx_gpu, gy_gpu, gz_gpu, rbins_gpu, panrx_gpu,
                                                elrx_gpu, panrx_gpu, elrx_gpu, beamform_data, bpj_grid,
                                                bpj_wavelength, near_range_s, rpref.fs * settings['upsample'], bwidth,
                                                rpref.az_half_bw,
                                                rpref.el_half_bw, settings['poly_num'], pts_debug, angs_debug,
                                                settings['debug'])
        cupy.cuda.Device().synchronize()

        # If we're halfway through the collect, grab debug data
        postoorig = llh2enu(*settings['origin'], bg.ref) - rpref.pos(ts)
        angtoorig = np.arctan2(-postoorig[:, 1], postoorig[:, 0]) + np.pi / 2 - panrx
        if np.any(abs(angtoorig) < .1 * DTR):
            locp = rpref.txpos(ts[-1]).T
            test = beamform_data.get()
            angd = angs_debug.get()
            locd = pts_debug.get()

        # bpj_traces.append(go.Heatmap(z=db(bpj_grid.get())))
        bpj_truedata += bpj_grid.get()

    del panrx_gpu
    del postx_gpu
    del posrx_gpu
    del elrx_gpu
    del rtdata
    del beamform_data
    del upsample_data
    del bpj_grid
    del refgrid_gpu
    del rng_states
    del rbins_gpu
    del gx_gpu
    del gy_gpu
    del gz_gpu

    # Apply range roll-off compensation to final image
    mag_data = applyRangeRolloff(bpj_truedata)

    """
    ----------------------------PLOTS-------------------------------
    """

    if test is not None:
        plt.figure('Doppler data')
        plt.imshow(np.fft.fftshift(db(np.fft.fft(test, axis=1)), axes=1),
                   extent=(-sdr[settings['channel']].prf / 2, sdr[settings['channel']].prf / 2, ranges[-1], ranges[0]))
        plt.axis('tight')

    plt.figure('IMSHOW backprojected data')
    plt.imshow(db(mag_data), origin='lower')
    plt.axis('tight')

    try:
        if (nbpj_pts[0] * nbpj_pts[1]) < 400 ** 2:
            cx, cy, cz = bg.getGrid(settings['origin'], width=settings['grid_width'], height=settings['grid_height'],
                                    nrows=nbpj_pts[0], ncols=nbpj_pts[1], az=0)

            fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
            fig.add_scatter3d(x=cx.flatten(), y=cy.flatten(), z=cz.flatten(), mode='markers')
            fig.show()

            pvecs = azelToVec(angd[1, ...].flatten(), angd[0, ...].flatten()) * angd[2, ...].flatten()
            fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
            fig.add_scatter3d(x=locd[0, ...].flatten() + locp[0], y=locd[1, ...].flatten() + locp[1],
                              z=locd[2, ...].flatten() + locp[2], mode='markers')
            fig.add_scatter3d(x=pvecs[0, ...] + locp[0], y=pvecs[1, ...].flatten() + locp[1],
                              z=pvecs[2, ...].flatten() + locp[2], mode='markers')
            fig.show()

        plt.figure('IMSHOW truth data')
        plt.imshow(db(bg.refgrid), origin='lower')
        plt.axis('tight')
    except Exception as e:
        print(f'Error in generating background image: {e}')

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
    px.imshow(scaled_data, color_continuous_scale=px.colors.sequential.gray, zmin=0, zmax=nbits,
              origin='lower').show()

    dbrefgrid = db(bg.refgrid)
    nbits = 256
    plot_data_init = QuantileTransformer(output_distribution='normal').fit(
        dbrefgrid[dbrefgrid > -300].reshape(-1, 1)).transform(dbrefgrid.reshape(-1, 1)).reshape(dbrefgrid.shape)
    plot_data = plot_data_init
    max_bin = 3
    hist_counts, hist_bins = \
        np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    while hist_counts[-1] == 0:
        max_bin -= .2
        hist_counts, hist_bins = \
            np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    scaled_data = np.digitize(plot_data_init, hist_bins)

    px.imshow(scaled_data, color_continuous_scale=px.colors.sequential.gray, zmin=0, zmax=nbits,
              origin='lower').show()
    plt.figure('Image Histogram')
    plt.plot(hist_bins[1:], hist_counts)

    # Get IPR cut
    if settings['ipr_mode']:
        db_bpj = db(bpj_truedata)
        db_bpj -= np.max(db_bpj)
        mx = np.where(db_bpj == db_bpj.max())
        ipr_gridsz = min(db_bpj.shape[0] - mx[0][0], mx[0][0], db_bpj.shape[1] - mx[1][0], mx[1][0])

        cutfig = make_subplots(rows=2, cols=2, subplot_titles=(f'Azimuth', 'Contour', 'Magnitude', 'Range'))
        cutfig.add_trace(
            go.Scatter(
                x=np.arange(ipr_gridsz * 2) * settings['grid_width'] / nbpj_pts[0] - ipr_gridsz * settings[
                    'grid_width'] /
                  nbpj_pts[0],
                y=db_bpj[mx[0][0], mx[1][0] - ipr_gridsz:mx[1][0] + ipr_gridsz] - db_bpj[mx[0][0],
                                                                                  mx[1][0] - ipr_gridsz:mx[1][
                                                                                                            0] + ipr_gridsz].max(),
                mode='lines', showlegend=False), row=1, col=1)
        cutfig.add_trace(
            go.Scatter(
                x=np.arange(ipr_gridsz * 2) * settings['grid_height'] / nbpj_pts[1] - ipr_gridsz * settings[
                    'grid_height'] /
                  nbpj_pts[1],
                y=db_bpj[mx[0][0] - ipr_gridsz:mx[0][0] + ipr_gridsz, mx[1][0]] - db_bpj[mx[0][0] - ipr_gridsz:mx[0][
                                                                                                                   0] + ipr_gridsz,
                                                                                  mx[1][0]].max(),
                mode='lines', showlegend=False), row=2, col=2)
        cutfig.add_trace(
            go.Heatmap(z=db_bpj, colorscale=px.colors.sequential.gray), row=2, col=1)
        cutfig.add_trace(
            go.Contour(z=db_bpj, contours_coloring='lines', line_width=2, contours=dict(
                start=0,
                end=-60,
                size=10,
                showlabels=True,
            ), showscale=False), row=1, col=2)
        cutfig.show()

    plt.show()
