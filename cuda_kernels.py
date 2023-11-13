import cmath
import math
from simulation_functions import findPowerOf2, db
import numpy as np
from jax import numpy as jnp
import jax
from jax import config

config.update("jax_enable_x64", True)

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@jax.jit
def diff(x, y):
    a = y - x
    return (a + np.pi) - jnp.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@jax.jit
def raisedCosine(x, bw, a0):
    """
    Raised Cosine windowing function.
    :param x: float. Azimuth difference between point and beam center in radians.
    :param bw: float. Signal bandwidth in Hz.
    :param a0: float. Factor for raised cosine window generation.
    :return: float. Window value.
    """
    xf = x / bw + .5
    return a0 - (1 - a0) * jnp.cos(2 * np.pi * xf)


@jax.jit
def applyRadiationPattern(el_c, az_c, az_rx, el_rx, az_tx, el_tx, bw_az, bw_el):
    """
    Applies a very simple sinc radiation pattern.
    :param txonly:
    :param el_c: float. Center of beam in elevation, radians.
    :param az_c: float. Azimuth center of beam in radians.
    :param az_rx: float. Azimuth value of Rx antenna in radians.
    :param el_rx: float. Elevation value of Rx antenna in radians.
    :param az_tx: float. Azimuth value of Tx antenna in radians.
    :param el_tx: float. Elevation value of Tx antenna in radians.
    :param bw_az: float. Azimuth beamwidth of antenna in radians.
    :param bw_el: float. elevation beamwidth of antenna in radians.
    :return: float. Value by which a point should be scaled.
    """
    a = 1 / bw_az
    b = 1 / bw_el
    eldiff = diff(el_c, el_tx)
    azdiff = diff(az_c, az_tx)
    txaz = abs(jnp.sinc(a * azdiff))
    txaz = jnp.where(azdiff > bw_az * 2, 0, txaz)
    txel = abs(jnp.sinc(b * eldiff))
    txel = jnp.where(eldiff > bw_el * 2, 0, txel)
    tx_pat = txaz * txel
    # tx_pat = (2 * np.pi - abs(eldiff)) * (2 * np.pi - abs(azdiff))
    eldiff = diff(el_c, el_rx)
    azdiff = diff(az_c, az_rx)
    rxaz = abs(jnp.sinc(a * azdiff))
    rxaz = jnp.where(azdiff > bw_az * 2, 0, rxaz)
    rxel = abs(jnp.sinc(b * eldiff))
    rxel = jnp.where(eldiff > bw_el * 2, 0, rxel)
    rx_pat = rxaz * rxel
    # rx_pat = (2 * np.pi - abs(eldiff)) * (2 * np.pi - abs(azdiff))
    return tx_pat * tx_pat * rx_pat * rx_pat


@jax.jit
def barycentric(bx, by, vgz, vert_reflectivity):
    # Apply barycentric interpolation to get random point height and power
    x1 = jnp.round(bx).astype(int)
    # x2 = x1.copy()
    x3 = jnp.floor(bx)
    x3 = jnp.place(x3, bx % 1 < .5, jnp.ceil(x3), inplace=False).astype(int)
    y1 = jnp.round(by).astype(int)
    y2 = jnp.floor(by)
    y2 = jnp.place(y2, by % 1 < .5, jnp.ceil(y2), inplace=False).astype(int)
    # y3 = y1.copy()

    z1 = vgz[x1, y1]
    z2 = vgz[x1, y2]
    z3 = vgz[x3, y1]
    r1 = vert_reflectivity[x1, y1]
    r2 = vert_reflectivity[x1, y2]
    r3 = vert_reflectivity[x3, y1]

    bary_determinant = 1 / ((y2 - y1) * (x1 - x3))
    bary_determinant = jnp.place(bary_determinant, jnp.isinf(bary_determinant), -1., inplace=False)

    lam1 = ((y2 - y1) * (bx - x3) + (x3 - x1) * (by - y1)) * bary_determinant
    lam2 = ((x1 - x3) * (by - y1)) * bary_determinant
    lam3 = 1 - lam1 - lam2

    # Quick check to see if something's out of whack with the interpolation
    # lam3 + lam1 + lam2 should always be one
    bar_z = z1 * lam1 + lam2 * z2 + lam3 * z3
    gpr = r1 * lam1 + r2 * lam2 + r3 * lam3

    bx -= float(vgz.shape[0]) / 2.
    by -= float(vgz.shape[1]) / 2.
    return bx, by, bar_z, gpr


@jax.jit
def range_profile_vectorized(rot, shift, vgz, vert_reflectivity,
                             source_xyz, receive_xyz, panrx, elrx, pantx, eltx,
                             wavelength, near_range_s, source_fs, bw_az, bw_el, rbins, pts_per_tri):
    # Load in all the parameters that don't change
    wavenumber = 2 * np.pi / wavelength
    ran_key = jax.random.PRNGKey(42)
    px, py = jnp.meshgrid(jnp.arange(vert_reflectivity.shape[0]), jnp.arange(vert_reflectivity.shape[1]))
    pdata = jax.lax.fori_loop(0,
                      pts_per_tri,
                      lambda x, y: jax.lax.cond(x != 0,
                                                gather_data_loop_det,
                                                gather_data_loop,
                                                y, ran_key, px, py, source_xyz, receive_xyz, vgz, vert_reflectivity,
                                                source_fs, wavenumber, panrx, elrx, pantx, eltx, bw_az, bw_el, rot,
                                                shift, rbins, near_range_s),
                      jnp.zeros((len(rbins),), dtype=jnp.complex128))
    return pdata

@jax.jit
def gather_data_loop(pdata, ran_key, px, py, source_xyz, receive_xyz, vgz, vert_reflectivity, source_fs, wavenumber, panrx,
                     elrx, pantx, eltx, bw_az, bw_el, rot, shift, rbins, near_range_s):
    ran_key, *subkey = jax.random.split(ran_key, 3)
    bx = px + .5 - jax.random.uniform(subkey[0], shape=px.shape)
    by = py + .5 - jax.random.uniform(subkey[1], shape=py.shape)

    # Fix the random points outside the grid - this may make the edges have more points
    bx = jnp.place(bx, bx < 0, .1, inplace=False)
    bx = jnp.where(bx >= px.shape[0], px.shape[0] - 1.1, bx)
    by = jnp.where(by < 0, .1, by)
    by = jnp.where(by >= py.shape[1], py.shape[1] - 1.1, by)

    # Apply barycentric interpolation to get random point height and power
    bx, by, bar_z, gpr = barycentric(bx.flatten(), by.flatten(), vgz, vert_reflectivity)
    bar_x = rot[0, 0] * bx + rot[0, 1] * by + shift[0]
    bar_y = rot[1, 0] * bx + rot[1, 1] * by + shift[1]

    # Calculate out the angles in azimuth and elevation for the bounce
    tx = bar_x - source_xyz[0]
    ty = bar_y - source_xyz[1]
    tz = bar_z - source_xyz[2]
    rng = jnp.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

    rx = bar_x - receive_xyz[0]
    ry = bar_y - receive_xyz[1]
    rz = bar_z - receive_xyz[2]
    r_rng = jnp.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
    r_el = -jnp.arcsin(rz / r_rng)
    r_az = jnp.arctan2(rx, ry)

    two_way_rng = rng + r_rng
    # a = abs(b_x * rx / r_rng + b_y * ry / r_rng + b_z * rz / r_rng)
    reflectivity = 1.  # math.pow((1. / -a + 1.) / 20, 10)

    but = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    att = applyRadiationPattern(r_el, r_az, panrx, elrx, pantx, eltx, bw_az, bw_el)
    att = jnp.place(att, jnp.logical_or(but < 0, but > len(rbins) - 1), 0, inplace=False)
    but = jnp.floor(but).astype(int)
    return pdata.at[but].add(att * jnp.exp(-1j * wavenumber * two_way_rng) *
                                  gpr * reflectivity) / (rbins - rbins[0] + 1) ** 4

@jax.jit
def gather_data_loop_det(pdata, ran_key, px, py, source_xyz, receive_xyz, vgz, gpr, source_fs, wavenumber, panrx,
                     elrx, pantx, eltx, bw_az, bw_el, rot, shift, rbins, near_range_s):
    bx = px.flatten() - px.shape[0] / 2
    by = py.flatten() - py.shape[1] / 2
    bar_z = vgz.flatten()

    # Shift and rotate into proper frame
    bar_x = rot[0, 0] * bx + rot[0, 1] * by + shift[0]
    bar_y = rot[1, 0] * bx + rot[1, 1] * by + shift[1]

    # Calculate out the angles in azimuth and elevation for the bounce
    tx = bar_x - source_xyz[0]
    ty = bar_y - source_xyz[1]
    tz = bar_z - source_xyz[2]
    rng = jnp.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

    rx = bar_x - receive_xyz[0]
    ry = bar_y - receive_xyz[1]
    rz = bar_z - receive_xyz[2]
    r_rng = jnp.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
    r_el = -jnp.arcsin(rz / r_rng)
    r_az = jnp.arctan2(-ry, rx) + np.pi / 2

    two_way_rng = rng + r_rng
    # a = abs(b_x * rx / r_rng + b_y * ry / r_rng + b_z * rz / r_rng)
    reflectivity = 1.  # math.pow((1. / -a + 1.) / 20, 10)

    but = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    att = applyRadiationPattern(r_el, r_az, panrx, elrx, pantx, eltx, bw_az, bw_el)
    # att = jnp.place(att, jnp.logical_or(but < 0, but > len(rbins) - 1), 0, inplace=False)
    but = jnp.floor(but).astype(int)
    return pdata.at[but].add(att * jnp.exp(-1j * wavenumber * two_way_rng) *
                                  gpr.flatten() * reflectivity) / (rbins - rbins[0] + 1) ** 4

@jax.jit
def backproject(source_xyz, receive_xyz, gx, gy, gz, panrx, elrx, pantx, eltx, pulse_data, final_grid,
                wavelength, near_range_s, source_fs, signal_bw, bw_az, bw_el):
    """
    Backprojection kernel.
    :param source_xyz: array. XYZ values of the source, usually Tx antenna, in meters.
    :param receive_xyz: array. XYZ values of the receiver, usually Rx antenna, in meters.
    :param gx: array. X values, in meters, of grid.
    :param gy: array. Y values, in meters, of grid.
    :param gz: array. Z values, in meters, of grid.
    :param rbins: array. Range bins, in meters.
    :param panrx: array. Rx azimuth values, in radians.
    :param elrx: array. Rx elevation values, in radians.
    :param pantx: array. Tx azimuth values, in radians.
    :param eltx: array. Tx elevation values, in radians.
    :param pulse_data: array. Complex pulse return data.
    :param final_grid: array. 2D matrix that accumulates all the corrected phase values.
    This is the backprojected image.
    :param wavelength: float. Wavelength used for phase correction.
    :param near_range_s: float. Near range value in seconds.
    :param source_fs: float. Sampling frequency in Hz.
    :param signal_bw: float. Bandwidth of signal in Hz.
    :param bw_az: float. Azimuth beamwidth in radians.
    :param bw_el: float. Elevation beamwidth in radians.
    :param poly: int. Determines the order of polynomial interpolation for range bins.
    :param calc_pts: array. Debug array for calculated ranges. Optional.
    :param calc_angs: array. Debug array for calculated angles to points. Optional.
    :param debug_flag: bool. If True, populates the calc_pts and calc_angs arrays.
    :return: Nothing, technically. final_grid is the returned product.
    """
    # Load in all the parameters that don't change
    k = 2 * np.pi / wavelength

    # Grab pulse data and sum up for this pixel
    for tt in range(pulse_data.shape[0]):
        cp = pulse_data[tt, :]
        # Get LOS vector in XYZ and spherical coordinates at pulse time
        # Tx first
        tx = gx - source_xyz[0, tt]
        ty = gy - source_xyz[1, tt]
        tz = gz - source_xyz[2, tt]
        tx_rng = jnp.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

        # Rx
        rx = gx - receive_xyz[0, tt]
        ry = gy - receive_xyz[1, tt]
        rz = gz - receive_xyz[2, tt]
        rx_rng = jnp.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
        r_el = -jnp.arcsin(rz / rx_rng)
        r_az = jnp.arctan2(-ry, rx) + np.pi / 2

        # Check to see if it's outside of our beam
        az_diffrx = diff(r_az, panrx[tt])
        el_diffrx = diff(r_el, elrx[tt])

        # Get index into range compressed data
        two_way_rng = tx_rng + rx_rng
        but = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = jnp.where(but < 0, 0, but)
        but = jnp.where(but > pulse_data.shape[1], pulse_data.shape[1] - 1, but)

        # Attenuation of beam in elevation and azimuth
        att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt],
                                    bw_az, bw_el)

        # Azimuth window to reduce sidelobes
        # Gaussian window
        # az_win = math.exp(-az_diffrx * az_diffrx / (2 * .001))
        # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
        az_win = raisedCosine(az_diffrx, signal_bw, .5)
        # az_win = 1.

        bi0 = jnp.floor(but).astype(int)
        bi1 = jnp.ceil(but).astype(int)

        b1_range = (bi1 / source_fs + 2 * near_range_s) * c0
        b0_range = (bi0 / source_fs + 2 * near_range_s) * c0

        a = (cp[bi0] * (b1_range - tx_rng) + cp[bi1] * (tx_rng - b0_range)) \
            / (c0 / source_fs)

        '''if poly == 0:
            # This is how APS does it (for reference, I guess)
            a = cp[bi1]
        elif poly == 1:
            # Linear interpolation between bins (slower but more accurate)
            a = (cp[bi0] * (rbins[bi1] - tx_rng) + cp[bi1] * (tx_rng - rbins[bi0])) \
                / (rbins[bi1] - rbins[bi0])
        else:
            # This is a lagrange polynomial interpolation of the specified order
            ar = ai = 0
            kspan = (poly + 1 if poly % 2 != 0 else poly) // 2
            ks = max(bi0 - kspan, 0)
            ke = bi0 + kspan + 1 if bi0 + kspan < n_samples else n_samples
            for jdx in range(ks, ke):
                mm = 1
                for kdx in range(ks, ke):
                    if jdx != kdx:
                        mm *= (tx_rng - rbins[kdx]) / (rbins[jdx] - rbins[kdx])
                ar += mm * cp[jdx].real
                ai += mm * cp[jdx].imag
            a = ar + 1j * ai'''

        # Multiply by phase reference function, attenuation and azimuth window
        # if tt == 0:
        #     print('att ', att, 'rng', tx_rng, 'bin', bi1, 'az_diff', az_diffrx, 'el_diff', el_diffrx)
        exp_phase = k * two_way_rng
        final_grid += a * jnp.exp(1j * exp_phase) * att * az_win
    return final_grid


@jax.jit
def backproject_vectorized(source_xyz, receive_xyz, gx, gy, gz, panrx, elrx, pantx, eltx, pulse_data,
                           wavelength, near_range_s, source_fs, signal_bw, bw_az, bw_el):
    """
    Backprojection kernel.
    :param source_xyz: array. XYZ values of the source, usually Tx antenna, in meters.
    :param receive_xyz: array. XYZ values of the receiver, usually Rx antenna, in meters.
    :param gx: array. X values, in meters, of grid.
    :param gy: array. Y values, in meters, of grid.
    :param gz: array. Z values, in meters, of grid.
    :param rbins: array. Range bins, in meters.
    :param panrx: array. Rx azimuth values, in radians.
    :param elrx: array. Rx elevation values, in radians.
    :param pantx: array. Tx azimuth values, in radians.
    :param eltx: array. Tx elevation values, in radians.
    :param pulse_data: array. Complex pulse return data.
    :param final_grid: array. 2D matrix that accumulates all the corrected phase values.
    This is the backprojected image.
    :param wavelength: float. Wavelength used for phase correction.
    :param near_range_s: float. Near range value in seconds.
    :param source_fs: float. Sampling frequency in Hz.
    :param signal_bw: float. Bandwidth of signal in Hz.
    :param bw_az: float. Azimuth beamwidth in radians.
    :param bw_el: float. Elevation beamwidth in radians.
    :param poly: int. Determines the order of polynomial interpolation for range bins.
    :param calc_pts: array. Debug array for calculated ranges. Optional.
    :param calc_angs: array. Debug array for calculated angles to points. Optional.
    :param debug_flag: bool. If True, populates the calc_pts and calc_angs arrays.
    :return: Nothing, technically. final_grid is the returned product.
    """
    # Load in all the parameters that don't change
    k = 2 * np.pi / wavelength

    # Grab pulse data and sum up for this pixel
    # Get LOS vector in XYZ and spherical coordinates at pulse time
    # Tx first
    tx = gx - source_xyz[0]
    ty = gy - source_xyz[1]
    tz = gz - source_xyz[2]
    tx_rng = jnp.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

    # Rx
    rx = gx - receive_xyz[0]
    ry = gy - receive_xyz[1]
    rz = gz - receive_xyz[2]
    rx_rng = jnp.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
    r_el = -jnp.arcsin(rz / rx_rng)
    r_az = jnp.arctan2(-ry, rx) + np.pi / 2

    # Check to see if it's outside of our beam
    az_diffrx = diff(r_az, panrx)
    el_diffrx = diff(r_el, elrx)

    # Get index into range compressed data
    two_way_rng = tx_rng + rx_rng
    but = (two_way_rng / c0 - 2 * near_range_s) * source_fs

    # Attenuation of beam in elevation and azimuth
    att = applyRadiationPattern(r_el, r_az, panrx, elrx, pantx, eltx,
                                bw_az, bw_el)

    att = jnp.place(att, jnp.logical_or(but < 0, but > len(pulse_data)), 0, inplace=False)
    att = jnp.place(att, abs(az_diffrx) > bw_az, 0, inplace=False)
    att = jnp.place(att, abs(el_diffrx) > bw_el, 0, inplace=False)

    # Azimuth window to reduce sidelobes
    # Gaussian window
    # az_win = jnp.exp(-az_diffrx * az_diffrx / (2 * .001))
    # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
    az_win = raisedCosine(az_diffrx, signal_bw, .5)
    # az_win = 1.

    bi0 = jnp.floor(but).astype(int)
    bi1 = jnp.ceil(but).astype(int)

    b1_range = (bi1 / source_fs + 2 * near_range_s) * c0 / 2
    b0_range = (bi0 / source_fs + 2 * near_range_s) * c0 / 2

    a = (pulse_data[bi0] * (b1_range - tx_rng) + pulse_data[bi1] * (tx_rng - b0_range)) \
        / (b1_range - b0_range)
    # a = pulse_data[bi0]

    # Multiply by phase reference function, attenuation and azimuth window
    return a * jnp.exp(1j * k * two_way_rng) * att * az_win


def split(arr, n_devices=2):
    """Splits the first axis of `arr` evenly across the number of devices."""
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])


if __name__ == '__main__':

    from platform_helper import SDRPlatform
    from grid_helper import SDREnvironment
    from SDRParsing import SDRParse, load
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    fnme = '/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar'
    origin = np.array([40.138538, -111.662090, 1365.8849123907273])
    upsample = 4
    fdelay = 0
    sar = load(fnme)
    bg = SDREnvironment(sar)
    rp = SDRPlatform(sar, bg.ref)
    rp.bwidth = sar[0].bw
    near_range_s = rp.calcRanges(fdelay, partial_pulse_percent=1.)[0] / c0
    nsam = rp.calcNumSamples(1542, .2)
    offset_hz = sar[0].xml['DC_Offset_MHz'] * 1e6
    wavelength = c0 / (sar[0].fc - rp.bwidth / 2 - offset_hz) if offset_hz != 0 else c0 / sar[0].fc
    fft_len = findPowerOf2(nsam + sar[0].pulse_length_N)
    mfilt = sar.genMatchedFilter(0, fft_len=findPowerOf2(nsam)).astype(jnp.complex128)
    chirp = jnp.fft.fft(sar[0].cal_chirp, fft_len).astype(jnp.complex128)
    batch_sz = 32

    avvel = rp.vel(rp.gpst).mean(axis=0)
    bg.resample(origin, 100, 100, (100, 100))
    # bg._refgrid = np.zeros((101, 101))
    # bg._refgrid[50, 50] = 1e9
    rot = bg.transforms[0]
    shift = bg.transforms[1]
    gx, gy, vgz = bg.getGrid()
    vert_reflectivity = bg.refgrid
    bpj_data = np.zeros(vgz.shape, dtype=np.complex128)
    npulses = len(sar[0].pulse_time)
    rbins = rp.calcRangeBins(fdelay, upsample)
    n_devices = jax.device_count()
    mapped_bpj = jax.vmap(backproject_vectorized,
                          in_axes=[0, 0, None, None, None, 0, 0, 0, 0, 0, None, None, None, None, None, None])
    mapped_rpg = jax.vmap(range_profile_vectorized,
                          in_axes=[None, None, None, None, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None])
    grid_center = [np.mean(gx), np.mean(gy), np.mean(vgz)]
    av_heading = rp.heading(rp.gpst).mean() - np.pi / 2

    for tt in tqdm(range(0, npulses, batch_sz)):
        if npulses - tt < batch_sz:
            break
        ts = sar[0].pulse_time[tt:tt + batch_sz]
        frame = sar[0].frame_num[tt:tt + batch_sz]
        source_xyz = rp.txpos(ts)
        receive_xyz = rp.rxpos(ts)
        panrx = rp.pan(ts)
        pantx = rp.pan(ts)
        elrx = rp.tilt(ts)
        eltx = rp.tilt(ts)
        upsample_data = (np.random.randn(batch_sz, fft_len * upsample) +
                         1j * np.random.randn(batch_sz, fft_len * upsample)) * 0
        '''pdata = mapped_rpg(rot, shift, vgz, vert_reflectivity,
                           source_xyz, receive_xyz, panrx, elrx, pantx, elrx,
                           wavelength, near_range_s, rp.fs, rp.az_half_bw, rp.el_half_bw, rbins, 3)

        mfilt_data = jnp.fft.fft(pdata, fft_len, axis=1) * chirp * mfilt'''
        mfilt_data = jnp.fft.fft(sar.getPulses(frame)[1].T, fft_len, axis=1).astype(jnp.complex128) * mfilt
        upsample_data[:, :fft_len // 2] += mfilt_data[:, :fft_len // 2]
        upsample_data[:, -fft_len // 2:] += mfilt_data[:, -fft_len // 2:]

        upsample_data = jnp.fft.ifft(upsample_data, axis=1)[:, :nsam * upsample].astype(jnp.complex128)

        # check_az = np.arctan2(grid_center[0] - source_xyz.mean(axis=0)[0], grid_center[1] - source_xyz.mean(axis=0)[1])
        '''if abs(check_az - av_heading) < 2 * DTR:
            plt.figure('Upsampled Data')
            plt.imshow(db(np.array(jnp.fft.fft(upsample_data, axis=0))).T)
            plt.axis('tight')
            # break
        else:
            pass
            # print(abs(check_az - av_heading))'''

        bpj_data += jnp.sum(mapped_bpj(
            source_xyz, receive_xyz, gx, gy, vgz, panrx, elrx, pantx, eltx,
            upsample_data, wavelength, near_range_s, rp.fs * upsample, rp.bwidth, rp.az_half_bw,
            rp.el_half_bw), axis=0)

    plt.figure('bproject')
    plt.imshow(db(np.array(bpj_data)), origin='lower')

    plt.figure('refgrid')
    plt.imshow((db(bg.refgrid)), origin='lower')
    plt.show()

    '''fig = plt.figure('grid')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(gx.flatten(), gy.flatten(), gz.flatten())
    ax.scatter(np.array(bar_x).flatten(), np.array(bar_y).flatten(), np.array(bar_z).flatten())'''
    source_xyz = source_xyz[0, :]
    receive_xyz = receive_xyz[0, :]
    panrx = panrx[0]
    pantx = pantx[0]
    elrx = elrx[0]
    eltx = eltx[0]
    bw_az = rp.az_half_bw
    bw_el = rp.el_half_bw
    source_fs = rp.fs * upsample
    pulse_data = upsample_data[0, ...]
    gz = vgz
    signal_bw = rp.bwidth

'''fig = plt.figure('grid')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(gx.flatten(), gy.flatten(), gz.flatten(), c=db(np.array(bpj_data)).T.flatten())
ax.scatter(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2])
# ax.plot([source_xyz[0], source_xyz[0] + tx[0, 0]], [source_xyz[1], source_xyz[1] + ty[0, 0]], [source_xyz[2], source_xyz[2] + tz[0, 0]])

plt.figure()
plt.scatter(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1])
plt.scatter(gx.flatten(), gy.flatten(), c=np.array(att).flatten())

fig = plt.figure('projection grid')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(gx.flatten(), gy.flatten(), gz.flatten(), c=db(np.array(bpj_data)).T.flatten())
ax.scatter(source_xyz[0] + tx.flatten(), source_xyz[1] + ty.flatten(), source_xyz[2] + tz.flatten())'''