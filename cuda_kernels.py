import cmath
import math
from numba import cuda, njit
from simulation_functions import findPowerOf2
import numpy as np

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@cuda.jit(device=True)
def diff(x, y):
    a = y - x
    return (a + np.pi) - math.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


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


@cuda.jit(device=True)
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
    a = np.pi / bw_az
    b = np.pi / bw_el
    eldiff = diff(el_c, el_tx)
    azdiff = diff(az_c, az_tx)
    txaz = abs(math.sin(a * azdiff) / (a * azdiff)) if azdiff != 0 else 1.
    txaz = txaz if azdiff <= bw_az * 2 else 0.
    txel = abs(math.sin(b * eldiff) / (b * eldiff)) if eldiff != 0 else 1.
    txel = txel if eldiff <= bw_el * 2 else 0.
    tx_pat = txaz * txel
    # tx_pat = (2 * np.pi - abs(eldiff)) * (2 * np.pi - abs(azdiff))
    eldiff = diff(el_c, el_rx)
    azdiff = diff(az_c, az_rx)
    rxaz = abs(math.sin(a * azdiff) / (a * azdiff)) if azdiff != 0 else 1.
    rxaz = rxaz if azdiff <= bw_az * 2 else 0.
    rxel = abs(math.sin(b * eldiff) / (b * eldiff)) if eldiff != 0 else 1.
    rxel = rxel if eldiff <= bw_el * 2 else 0.
    rx_pat = rxaz * rxel
    # rx_pat = (2 * np.pi - abs(eldiff)) * (2 * np.pi - abs(azdiff))
    return tx_pat * tx_pat * rx_pat * rx_pat


@cuda.jit('void(float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:], '
          'float64[:], float64[:], float64[:], complex128[:, :], complex128[:, :], float64, float64, float64, float64, '
          'float64, float64, int32, float64[:, :, :], float64[:, :, :], int32)')
def backproject(source_xyz, receive_xyz, gx, gy, gz, rbins, panrx, elrx, pantx, eltx, pulse_data, final_grid,
                wavelength, near_range_s, source_fs, signal_bw, bw_az, bw_el, poly, calc_pts, calc_angs, debug_flag):
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
    if pcol < gx.shape[0] and prow < gx.shape[1]:
        # Load in all the parameters that don't change
        acc_val = 0
        nPulses = pulse_data.shape[1]
        n_samples = pulse_data.shape[0]
        k = 2 * np.pi / wavelength

        # Grab pulse data and sum up for this pixel
        for tt in range(nPulses):
            cp = pulse_data[:, tt]
            # Get LOS vector in XYZ and spherical coordinates at pulse time
            # Tx first
            tx = gx[pcol, prow] - source_xyz[tt, 0]
            ty = gy[pcol, prow] - source_xyz[tt, 1]
            tz = gz[pcol, prow] - source_xyz[tt, 2]
            tx_rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

            # Rx
            rx = gx[pcol, prow] - receive_xyz[tt, 0]
            ry = gy[pcol, prow] - receive_xyz[tt, 1]
            rz = gz[pcol, prow] - receive_xyz[tt, 2]
            rx_rng = math.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
            r_el = -math.asin(rz / rx_rng)
            r_az = math.atan2(-ry, rx) + np.pi / 2
            if debug_flag and tt == 0:
                calc_pts[0, pcol, prow] = rx
                calc_pts[1, pcol, prow] = ry
                calc_pts[2, pcol, prow] = rz
                calc_angs[0, pcol, prow] = r_el
                calc_angs[1, pcol, prow] = r_az

            # Check to see if it's outside of our beam
            az_diffrx = diff(r_az, panrx[tt])
            el_diffrx = diff(r_el, elrx[tt])
            if (abs(az_diffrx) > bw_az) or (abs(el_diffrx) > bw_el):
                continue

            # Get index into range compressed data
            two_way_rng = tx_rng + rx_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > n_samples:
                continue

            # Attenuation of beam in elevation and azimuth
            att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt],
                                       bw_az, bw_el)
            # att = 1.
            if debug_flag:
                calc_angs[2, pcol, prow] += 1

            # Azimuth window to reduce sidelobes
            # Gaussian window
            # az_win = math.exp(-az_diffrx * az_diffrx / (2 * .001))
            # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
            az_win = raisedCosine(az_diffrx, bw_az, .5)
            # az_win = 1.

            if rbins[but - 1] < tx_rng < rbins[but]:
                bi0 = but - 1
                bi1 = but
            else:
                bi0 = but
                bi1 = but + 1

            if poly == 0:
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
                a = ar + 1j * ai

            # Multiply by phase reference function, attenuation and azimuth window
            # if tt == 0:
            #     print('att ', att, 'rng', tx_rng, 'bin', bi1, 'az_diff', az_diffrx, 'el_diff', el_diffrx)
            exp_phase = k * two_way_rng
            acc_val += a * cmath.exp(1j * exp_phase) * att * az_win
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


def getMaxThreads():
    gpuDevice = cuda.get_current_device()
    maxThreads = int(np.sqrt(gpuDevice.MAX_THREADS_PER_BLOCK))
    sqrtMaxThreads = maxThreads // 2
    return sqrtMaxThreads, sqrtMaxThreads
