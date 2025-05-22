import cmath
import math
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64
from .cuda_functions import make_float3, length
from .simulation_functions import findPowerOf2
import numpy as np

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@cuda.jit(device=True)
def diff(x, y):
    return x - y
    # a = y - x
    # return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def cpudiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


@cuda.jit(device=True)
def raisedCosine(x, bw, a0):
    """
    Raised Cosine windowing function.
    :param x: float. Azimuth difference between point and beam center in radians.
    :param bw: float. Signal bandwidth in Hz.
    :param a0: float. Factor for raised cosine window generation.
    :return: float. Window value.
    """
    xf = x / bw + .5
    return a0 - (1 - a0) * math.cos(2 * np.pi * xf)


@cuda.jit(device=True, fastmath=True)
def getRangeAndAngles(v, s):
    t = v - s
    rng = length(t)
    az = math.atan2(t.x, t.y)
    el = -math.asin(t.z / rng)
    return t, rng, az, el


@cuda.jit(device=True, fastmath=True)
def applyRadiationPattern(el_c, az_c, az_rx, el_rx, az_tx, el_tx, bw_az, bw_el):
    """
    Applies a very simple sinc radiation pattern.
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
    a = np.pi / bw_az
    b = np.pi / bw_el
    # Abs shouldn't be a problem since the pattern is symmetrical about zero
    eldiff = diff(el_c, el_rx)
    azdiff = diff(az_c, az_rx)
    # rx_pat = math.sin(a * azdiff) * math.sin(b * eldiff) / (cuda.fma(a, azdiff, 1e-9) + cuda.fma(b, eldiff, 1e-9))
    rx_pat = math.sin(a * azdiff) / cuda.fma(a, azdiff, 1e-9) * math.sin(b * eldiff) / cuda.fma(b, eldiff, 1e-9)
    eldiff = diff(el_c, el_tx)
    azdiff = diff(az_c, az_tx)
    # rx_pat = math.sin(a * azdiff) * math.sin(b * eldiff) / (cuda.fma(a, azdiff, 1e-9) + cuda.fma(b, eldiff, 1e-9))
    tx_pat = math.sin(a * azdiff) / cuda.fma(a, azdiff, 1e-9) * math.sin(b * eldiff) / cuda.fma(b, eldiff, 1e-9)
    return tx_pat * tx_pat * rx_pat * rx_pat


@cuda.jit(device=True, fastmath=True)
def applyOneWayRadiationPattern(el_c, az_c, az_rx, el_rx, bw_az, bw_el):
    """
    Applies a very simple sinc radiation pattern.
    :param el_c: float. Center of beam in elevation, radians.
    :param az_c: float. Azimuth center of beam in radians.
    :param az_rx: float. Azimuth value of Rx antenna in radians.
    :param el_rx: float. Elevation value of Rx antenna in radians.
    :param bw_az: float. Azimuth beamwidth of antenna in radians.
    :param bw_el: float. elevation beamwidth of antenna in radians.
    :return: float. Value by which a point should be scaled.
    """
    a = np.pi / bw_az
    b = np.pi / bw_el
    # Abs shouldn't be a problem since the pattern is symmetrical about zero
    eldiff = diff(el_c, el_rx)
    azdiff = diff(az_c, az_rx)
    # rx_pat = math.sin(a * azdiff) * math.sin(b * eldiff) / (cuda.fma(a, azdiff, 1e-9) + cuda.fma(b, eldiff, 1e-9))
    rx_pat = math.sin(a * azdiff) / cuda.fma(a, azdiff, 1e-9) * math.sin(b * eldiff) / cuda.fma(b, eldiff, 1e-9)
    # rx_pat = math.sin(a * azdiff) / (a * azdiff) * math.sin(b * eldiff) / (b * eldiff)
    return rx_pat * rx_pat


def applyRadiationPatternCPU(el_c, az_c, az_rx, el_rx, az_tx, el_tx, bw_az, bw_el):
    """
    Applies a very simple sinc radiation pattern.
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
    a = np.pi / bw_az
    b = np.pi / bw_el
    # Abs shouldn't be a problem since the pattern is symmetrical about zero
    eldiff = max(1e-9, abs(cpudiff(el_c, el_tx)))
    azdiff = max(1e-9, abs(cpudiff(az_c, az_tx)))
    tx_pat = abs(math.sin(a * azdiff) / (a * azdiff)) * abs(math.sin(b * eldiff) / (b * eldiff))
    eldiff = max(1e-9, abs(cpudiff(el_c, el_rx)))
    azdiff = max(1e-9, abs(cpudiff(az_c, az_rx)))
    rx_pat = abs(math.sin(a * azdiff) / (a * azdiff)) * abs(math.sin(b * eldiff) / (b * eldiff))
    return tx_pat * tx_pat * rx_pat * rx_pat


@cuda.jit()
def genRangeProfile(gx, gy, vgz, vert_reflectivity,
                    source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, rng_states, calc_pts,
                    calc_angs, wavelength, near_range_s, source_fs, bw_az, bw_el, power_scale, pts_per_tri, debug_flag):
    # sourcery no-metrics
    px, py, ntri = cuda.grid(ndim=3)
    if px < vgz.shape[0] and py < vgz.shape[1] and ntri < pts_per_tri:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        # for ntri in range(pts_per_tri):
        # I'm not sure why but vgz and vert_reflectivity need their indexes swapped here
        if ntri != 0 and px < vgz.shape[0] - 1 and py < vgz.shape[1] - 1:
            rand_x = xoroshiro128p_uniform_float64(rng_states, py * vgz.shape[0] + px)
            rand_y = xoroshiro128p_uniform_float64(rng_states, py * vgz.shape[0] + px)
            if 0 < px < vgz.shape[0]:
                bx = px + .5 - rand_x
            elif px == 0:
                bx = .5 * rand_x
            else:
                bx = px - .5 * rand_x
            if 0 < py < vgz.shape[1]:
                by = py + .5 - rand_y
            elif py == 0:
                by = .5 * rand_y
            else:
                by = py - .5 * rand_y

            # Apply barycentric interpolation to get random point height and power
            z1 = vgz[px, py]
            r1 = vert_reflectivity[px, py]
            px3 = px + 1 if bx > px else px - 1
            py3 = py
            z3 = vgz[px3, py3]
            r3 = vert_reflectivity[px3, py3]
            px2 = px
            py2 = py - 1 if by < py else py + 1
            z2 = vgz[px2, py2]
            r2 = vert_reflectivity[px2, py2]

            det = 1. / (py2 - py3) * (px - px3) + (px3 - px2) * (py - py3)

            lam1 = ((py2 - py3) * (bx - px3) + (px3 - px2) * (by - py3)) * det
            lam2 = ((py3 - py) * (bx - px3) + (px - px3) * (by - py3)) * det
            lam3 = 1 - lam1 - lam2

            # Quick check to see if something's out of whack with the interpolation
            # lam3 + lam1 + lam2 should always be one
            if lam3 < 0.:
                lam1 = .33
                lam2 = .33
                lam3 = .33
            bar_x = lam1 * gx[px, py] + lam2 * gx[px2, py2] + lam3 * gx[px3, py3]
            bar_y = lam1 * gy[px, py] + lam2 * gy[px2, py2] + lam3 * gy[px3, py3]
            bar_z = lam1 * z1 + lam2 * z2 + lam3 * z3
            gpr = lam1 * r1 + lam2 * r2 + lam3 * r3
        else:
            bar_z = vgz[px, py]
            gpr = vert_reflectivity[px, py]
            bar_x = gx[px, py]
            bar_y = gy[px, py]

        for tt in range(source_xyz.shape[0]):

            # Calculate out the angles in azimuth and elevation for the bounce
            tx = bar_x - source_xyz[tt, 0]
            ty = bar_y - source_xyz[tt, 1]
            tz = bar_z - source_xyz[tt, 2]
            rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

            rx = bar_x - receive_xyz[tt, 0]
            ry = bar_y - receive_xyz[tt, 1]
            rz = bar_z - receive_xyz[tt, 2]
            r_rng = math.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
            r_el = -math.asin(rz / r_rng)
            r_az = math.atan2(rx, ry)

            if debug_flag and tt == 0:
                calc_pts[0, px, py] = rx
                calc_pts[1, px, py] = ry
                calc_pts[2, px, py] = rz
                calc_angs[0, px, py] = r_el
                calc_angs[1, px, py] = r_az

            two_way_rng = rng + r_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > pd_r.shape[0] or but < 0:
                continue

            # if debug_flag and tt == 0:
            #     calc_angs[2, px, py] = gpr

            if n_samples > but > 0:
                # a = abs(b_x * rx / r_rng + b_y * ry / r_rng + b_z * rz / r_rng)
                reflectivity = 1.  # math.pow((1. / -a + 1.) / 20, 10)
                att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt],
                                            pantx[tt], eltx[tt], bw_az, bw_el) / \
                      (two_way_rng * two_way_rng)
                att *= power_scale
                if debug_flag and tt == 0:
                    calc_angs[2, px, py] = att
                acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * gpr * reflectivity
                cuda.atomic.add(pd_r, (but, np.uint16(tt)), acc_val.real)
                cuda.atomic.add(pd_i, (but, np.uint16(tt)), acc_val.imag)
            cuda.syncthreads()


@cuda.jit()
def backproject(source_xyz, receive_xyz, gx, gy, gz, panrx, elrx, pantx, eltx, pulse_data, final_grid,
                wavelength, near_range_s, source_fs, bw_az, bw_el, poly):
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
    x, y = cuda.grid(ndim=2)
    x_stride, y_stride = cuda.gridsize(2)
    for pcol in range(x, gx.shape[0], x_stride):
        for prow in range(y, gx.shape[1], y_stride):
            acc_val = 0
            gl = make_float3(gx[pcol, prow], gy[pcol, prow], gz[pcol, prow])

            # Grab pulse data and sum up for this pixel
            for tt in range(pulse_data.shape[1]):
                # Get LOS vector in XYZ and spherical coordinates at pulse time
                # Tx first
                tx_rng = length(gl - make_float3(source_xyz[tt, 0], source_xyz[tt, 1], source_xyz[tt, 2]))

                # Rx
                rx_rng = length(gl - make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2]))
                r_az = math.atan2(gl.x - receive_xyz[tt, 0], gl.y - receive_xyz[tt, 1])
                # _, rx_rng, r_az, r_el = getRangeAndAngles(gl, make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2]))

                # Get index into range compressed data
                bi1 = int(((tx_rng + rx_rng) / c0 - 2 * near_range_s) * source_fs) + 1
                if bi1 >= pulse_data.shape[0] or bi1 < 0:
                    continue

                if poly == 0:
                    # This is how APS does it (for reference, I guess)
                    acc_val += pulse_data[bi1, tt] * cmath.exp(1j * k * (tx_rng + rx_rng)) * raisedCosine(diff(r_az, panrx[tt]), bw_az, .5)
                elif poly == 1:
                    bi0 = bi1 - 1
                    # Linear interpolation between bins (slower but more accurate)
                    bi1_rng = c0 / 2 * (bi1 / source_fs + 2 * near_range_s)
                    bi0_rng = c0 / 2 * (bi0 / source_fs + 2 * near_range_s)
                    acc_val += (pulse_data[bi0, tt] * (bi1_rng - tx_rng) + pulse_data[bi1, tt] * (tx_rng - bi0_rng)) \
                        / (bi1_rng - bi0_rng) * cmath.exp(1j * k * (tx_rng + rx_rng)) * raisedCosine(diff(r_az, panrx[tt]), bw_az, .5)

            final_grid[pcol, prow] = acc_val


@cuda.jit('void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], '
          'float64[:], float64[:], float64[:], complex128[:, :], complex128[:, :], float64, float64, float64, float64, '
          'float64, int32)')
def backprojectRegularGrid(source_xyz, receive_xyz, transform, gz, panrx, elrx, pantx, eltx, pulse_data, final_grid,
                wavelength, near_range_s, source_fs, bw_az, bw_el, poly):
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
    pcol, prow = cuda.grid(ndim=2)
    if pcol < gz.shape[0] and prow < gz.shape[1]:
        # Load in all the parameters that don't change
        acc_val = 0
        nPulses = pulse_data.shape[1]
        n_samples = pulse_data.shape[0]
        k = 2 * np.pi / wavelength
        px = transform[0, 0] * pcol + transform[0, 1] * prow + transform[0, 2]
        py = transform[1, 0] * pcol + transform[1, 1] * prow + transform[1, 2]
        gl = make_float3(px, py, gz[pcol, prow])

        # Grab pulse data and sum up for this pixel
        for tt in range(nPulses):
            cp = pulse_data[:, tt]
            # Get LOS vector in XYZ and spherical coordinates at pulse time
            # Rx
            r_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
            rx, rx_rng, r_az, r_el = getRangeAndAngles(gl, r_xyz)

            # Check to see if it's outside of our beam
            az_diffrx = diff(r_az, panrx[tt])
            el_diffrx = diff(r_el, elrx[tt])
            if (abs(az_diffrx) > bw_az) or (abs(el_diffrx) > bw_el):
                continue

            # Tx
            s_xyz = make_float3(source_xyz[tt, 0], source_xyz[tt, 1], source_xyz[tt, 2])
            tx, tx_rng, t_az, t_el = getRangeAndAngles(gl, s_xyz)

            # Get index into range compressed data
            two_way_rng = tx_rng + rx_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            bi0 = int(rng_bin)
            bi1 = bi0 + 1
            if bi1 >= n_samples or bi1 < 0:
                continue

            # Attenuation of beam in elevation and azimuth
            # att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt],
            #                             bw_az, bw_el)
            att = 1.
            # Azimuth window to reduce sidelobes
            # Gaussian window
            # az_win = math.exp(-az_diffrx * az_diffrx / (2 * .001))
            # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
            az_win = raisedCosine(az_diffrx, bw_az, .5)
            # az_win = 1.

            if poly == 0:
                # This is how APS does it (for reference, I guess)
                a = cp[bi1]
            elif poly == 1:
                # Linear interpolation between bins (slower but more accurate)
                bi1_rng = c0 / 2 * (bi1 / source_fs + 2 * near_range_s)
                bi0_rng = c0 / 2 * (bi0 / source_fs + 2 * near_range_s)
                a = (cp[bi0] * (bi1_rng - tx_rng) + cp[bi1] * (tx_rng - bi0_rng)) \
                    / (bi1_rng - bi0_rng)

            # Multiply by phase reference function, attenuation and azimuth window
            # if tt == 0:
            #     print('att ', att, 'rng', tx_rng, 'bin', bi1, 'az_diff', az_diffrx, 'el_diff', el_diffrx)
            acc_val += a * cmath.exp(1j * k * two_way_rng) * att * az_win
        final_grid[pcol, prow] = acc_val


@cuda.jit()
def backproject_gmti(source_hght, source_vel, grange, gvel, pantx, pulse_data, final_grid,
                     wavelength, near_range_s, source_fs, pri):
    """
    Backprojection kernel.
    :param source_hght: array. Altitude of platform in meters.
    :param source_vel: array. ENU velocity values of platform in m/s.
    :param grange: array. Range values to project to. Usually the range bins of a collect.
    :param gvel: array. Radial velocity values in m/s.
    :param pantx: array. Inertial azimuth angle of antenna in radians.
    :param pulse_data: array. Complex pulse return data.
    :param final_grid: array. 2D matrix that accumulates all the corrected phase values.
    This is the backprojected image.
    :param wavelength: float. Wavelength used for phase correction.
    :param near_range_s: float. Near range value in seconds.
    :param source_fs: float. Sampling frequency in Hz.
    :param pri: float. PRI interval of pulses.
    :return: Nothing, technically. final_grid is the returned product.
    """
    px, py = cuda.grid(ndim=2)
    if px < grange.shape[0] and py < grange.shape[1]:
        # Load in all the parameters that don't change
        acc_val = 0
        nPulses = pulse_data.shape[1]
        n_samples = pulse_data.shape[0]
        k = 2 * np.pi / wavelength

        # Grab pulse data and sum up for this pixel
        for tt in range(nPulses):
            cp = pulse_data[:, tt]
            delta_t = pri * (tt - nPulses / 2)

            graze = math.asin(source_hght[tt] / grange[px, py])
            clutter_vel = source_vel[0, tt] * math.cos(graze) * math.sin(pantx[tt]) + \
                          source_vel[1, tt] * math.cos(graze) * math.cos(pantx[tt]) + \
                          source_vel[2, tt] * -math.sin(graze)
            eff_rng = grange[px, py] - (clutter_vel + gvel[px, py]) * delta_t
            rng_bin = (2 * grange[px, py] / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > n_samples:
                continue

            # Multiply by phase reference function, attenuation and azimuth window
            exp_phase = k * eff_rng
            acc_val += cp[but] * cmath.exp(2j * exp_phase)
        final_grid[px, py] = acc_val


@cuda.jit()
def calcRangeProfile(vertices, source_xyz, pan, tilt, pd_r, pd_i, near_range_s, source_fs, az_bw, el_bw, sigma,
               wavenumber, radar_coeff):
    tt, pt = cuda.grid(ndim=2)
    if pt < vertices.shape[0] and tt < source_xyz.shape[0]:
        tx = vertices[pt, 0] - source_xyz[tt, 0]
        ty = vertices[pt, 1] - source_xyz[tt, 1]
        tz = vertices[pt, 2] - source_xyz[tt, 2]
        tx_rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

        rng_bin = (tx_rng * 2 / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin)

        if but < 0 or but > pd_r.shape[1]:
            return

        el = -math.asin(tz)
        az = math.atan2(-ty, tx) + np.pi / 2

        att = applyRadiationPattern(el, az, pan[tt], tilt[tt], pan[tt], tilt[tt], az_bw, el_bw) / (
                tx_rng * tx_rng * 2) * radar_coeff
        acc_val = att * cmath.exp(-1j * wavenumber * tx_rng * 2) * sigma[pt]
        cuda.atomic.add(pd_r, (tt, but), acc_val.real)
        cuda.atomic.add(pd_i, (tt, but), acc_val.imag)
        cuda.syncthreads()


def ambiguity(s1, s2, prf, dopp_bins, a_fs, mag=True, normalize=True):
    fdopp = np.linspace(-prf / 2, prf / 2, dopp_bins)
    fft_sz = findPowerOf2(len(s1)) * 2
    s1f = np.fft.fft(s1, fft_sz).conj().T
    shift_grid = np.arange(len(s2)) / a_fs
    sg = np.fft.fft(np.array([np.exp(2j * np.pi * f * shift_grid) for f in fdopp]) * s2, fft_sz, axis=1)
    A = np.fft.fftshift(np.fft.ifft(sg * s1f, axis=1), axes=1)
    if normalize:
        A = A / abs(A).max()
    return abs(A) if mag else A, fdopp, (np.arange(fft_sz) - fft_sz // 2) * c0 / a_fs


def getMaxThreads(pts_per_tri: int = 0):
    gpuDevice = cuda.get_current_device()
    maxThreads = int(np.sqrt(gpuDevice.MAX_THREADS_PER_BLOCK))
    sqrtMaxThreads = maxThreads
    if pts_per_tri <= 0:
        return sqrtMaxThreads, sqrtMaxThreads
    sqrtMaxThreads = sqrtMaxThreads // pts_per_tri
    return sqrtMaxThreads, sqrtMaxThreads, pts_per_tri


def optimizeThreadBlocks(total_threads, thread_split):
    gpuDevice = cuda.get_current_device()
    best = (0, 0)
    closest = np.inf
    mt = min(thread_split, gpuDevice.MAX_THREADS_PER_BLOCK)
    for n in range(total_threads // gpuDevice.MULTIPROCESSOR_COUNT, 1, -1):
        split = total_threads / (n * gpuDevice.MULTIPROCESSOR_COUNT)
        overhang = split % 1
        if overhang == 0:
            return n * gpuDevice.MULTIPROCESSOR_COUNT, mt
        if overhang < closest:
            best = (n * gpuDevice.MULTIPROCESSOR_COUNT, mt)
            closest = overhang + 0.
    return best


def optimizeThreadBlocks2d(threads_per_block, threads):
    assert len(threads) == len(threads_per_block), 'threads dimension must equal threads_per_block dimension'
    poss_configs = [np.array([th % n for n in range(2, tpb)]) for th, tpb in zip(threads, threads_per_block)]
    new_tpb = tuple(
        (max(1, np.max(np.where(pc == pc.max())[0]) + 2) if np.all(pc > 0) else np.max(np.where(pc == 0)[0]) + 2) if th > 1 else 1
        for th, pc in zip(threads, poss_configs)
    )
    return new_tpb, tuple(
        max(1, th // pc) + (1 if th % pc != 0 else 0) if th > 1 else 1
        for th, pc in zip(threads, new_tpb)
    )


def optimizeStridedThreadBlocks2d(threads):
    gpuDevice = cuda.get_current_device()
    maxThreads = int(gpuDevice.MAX_THREADS_PER_BLOCK**(1 / len(threads)) - 4)
    threads_per_block = (maxThreads for _ in threads)
    poss_configs = [np.array([th % n for n in range(2, tpb)]) for th, tpb in zip(threads, threads_per_block)]
    new_tpb = tuple(
        (max(1, np.max(np.where(pc == pc.max())[0]) + 2) if np.all(pc > 0) else np.max(np.where(pc == 0)[0]) + 2) if th > 1 else 1
        for th, pc in zip(threads, poss_configs)
    )

    n_proc_per_thread = (gpuDevice.MULTIPROCESSOR_COUNT if threads[0] > threads[1] else 1,
                         gpuDevice.MULTIPROCESSOR_COUNT if threads[1] >= threads[0] else 1)

    new_bpg = tuple(np.argsort([th / (tpb * n_proc * n) % 1 for n in range(1, int(th / (tpb * n_proc)))])[-1] * n_proc if th > (tpb * n_proc * 2) else int(th // tpb) for th, tpb, n_proc in zip(threads, new_tpb, n_proc_per_thread))
    return new_tpb, new_bpg
