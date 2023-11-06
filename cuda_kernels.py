import cmath
import math
from simulation_functions import findPowerOf2, db
import numpy as np
from jax import numpy as jnp
import jax

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@jax.jit
def diff(x, y):
    a = y - x
    return (a + np.pi) - jnp.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


def cpudiff(x, y):
    a = y - x
    return (a + np.pi) - np.floor((a + np.pi) / (2 * np.pi)) * 2 * np.pi - np.pi


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
def genRangeProfile(pathrx, pathtx, gp, norms, panrx, elrx, pantx, eltx, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gp.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gp[samp_point, 0]
        ty = gp[samp_point, 1]
        tz = gp[samp_point, 2]

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = tx - pathtx[0, tt]
        s_ty = ty - pathtx[1, tt]
        s_tz = tz - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        s_rx = tx - pathrx[0, tt]
        s_ry = ty - pathrx[1, tt]
        s_rz = tz - pathrx[2, tt]
        rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
        rng = (rngtx + rngrx)
        rng_bin = (rng / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            rnorm = norms[samp_point, 0] * s_tx + norms[samp_point, 1] * s_ty + norms[samp_point, 2] * s_tz
            # Reflection of wave
            ref_x = 2 * rnorm * norms[samp_point, 0] - s_tx / rngtx
            ref_y = 2 * rnorm * norms[samp_point, 1] - s_ty / rngtx
            ref_z = 2 * rnorm * norms[samp_point, 2] - s_tz / rngtx
            # Dot product of wave with Rx vector
            gv = abs(ref_x * s_rx / rngrx + ref_y * s_ry / rngrx + ref_z * s_rz / rngrx) * 10.
            att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, panrx[tt], elrx[tt],
                                        pantx[tt], eltx[tt], wavenumber, params[9], params[10]) * 1. * 1 / (rng * rng)
            acc_val = gv * att * cmath.exp(-1j * wavenumber * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@jax.jit
def genRangeData(pathrx, pathtx, gp, norms, panrx, elrx, pantx, eltx, pd_r, pd_i, params):
    tt, samp_point = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and samp_point < gp.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / params[2]

        tx = gp[samp_point, 0]
        ty = gp[samp_point, 1]
        tz = gp[samp_point, 2]

        # Get LOS vector in XYZ and spherical coordinates at pulse time
        s_tx = tx - pathtx[0, tt]
        s_ty = ty - pathtx[1, tt]
        s_tz = tz - pathtx[2, tt]
        rngtx = math.sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz) + c0 / params[4]
        s_rx = tx - pathrx[0, tt]
        s_ry = ty - pathrx[1, tt]
        s_rz = tz - pathrx[2, tt]
        rngrx = math.sqrt(s_rx * s_rx + s_ry * s_ry + s_rz * s_rz) + c0 / params[4]
        rng = (rngtx + rngrx)
        rng_bin = (rng / c0 - 2 * params[3]) * params[4]
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            rnorm = norms[samp_point, 0] * s_tx + norms[samp_point, 1] * s_ty + norms[samp_point, 2] * s_tz
            # Reflection of wave
            ref_x = 2 * rnorm * norms[samp_point, 0] - s_tx / rngtx
            ref_y = 2 * rnorm * norms[samp_point, 1] - s_ty / rngtx
            ref_z = 2 * rnorm * norms[samp_point, 2] - s_tz / rngtx
            # Dot product of wave with Rx vector
            gv = abs(ref_x * s_rx / rngrx + ref_y * s_ry / rngrx + ref_z * s_rz / rngrx) * 10.
            att = applyRadiationPattern(s_tx, s_ty, s_tz, rngtx, s_rx, s_ry, s_rz, rngrx, panrx[tt], elrx[tt],
                                        pantx[tt], eltx[tt], wavenumber, params[9], params[10]) * 1. * 1 / (rng * rng)
            acc_val = gv * att * cmath.exp(-1j * wavenumber * rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)


@jax.jit
def calcBounceFromMesh(rot, shift, vgz, vert_reflectivity,
                       source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, rng_states, calc_pts, calc_angs,
                       wavelength, near_range_s, source_fs, bw_az, bw_el, pts_per_tri, debug_flag):
    # sourcery no-metrics
    px, py = cuda.grid(ndim=2)
    if px < vgz.shape[0] and py < vgz.shape[1]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        for ntri in range(pts_per_tri):
            # I'm not sure why but vgz and vert_reflectivity need their indexes swapped here
            if ntri != 0 and px < vgz.shape[0] - 1 and py < vgz.shape[1] - 1:
                bx = px + .5 - \
                     xoroshiro128p_uniform_float32(rng_states, py * vgz.shape[0] + px)
                by = py + .5 - \
                     xoroshiro128p_uniform_float32(rng_states, py * vgz.shape[0] + px)

                # Apply barycentric interpolation to get random point height and power
                x3 = py - 1 if bx < py else py + 1
                y3 = px
                z3 = vgz[x3, y3]
                r3 = vert_reflectivity[x3, y3]
                x2 = py
                y2 = px - 1 if by < px else px + 1
                z2 = vgz[x2, y2]
                r2 = vert_reflectivity[x2, y2]

                lam1 = ((y2 - y3) * (bx - x3) + (x3 - x2) * (by - y3)) / \
                       ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3))
                lam2 = ((y3 - py) * (bx - x3) + (px - x3) * (by - y3)) / \
                       ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3))
                lam3 = 1 - lam1 - lam2

                # Quick check to see if something's out of whack with the interpolation
                # lam3 + lam1 + lam2 should always be one
                if lam3 < 0.:
                    continue
                bar_z = vgz[py, px] * lam1 + lam2 * z2 + lam3 * z3
                if lam1 < lam2 and lam1 < lam3:
                    gpr = vert_reflectivity[py, px]
                elif lam2 < lam1 and lam2 < lam3:
                    gpr = r2
                else:
                    gpr = r3
                # gpr = lam1 * vert_reflectivity[py, px] + lam2 * r2 + lam3 * r3
            elif ntri != 0:
                continue
            else:
                bx = float(px)
                by = float(py)
                bar_z = vgz[py, px]
                gpr = vert_reflectivity[py, px]
            bx -= float(vgz.shape[0]) / 2.
            by -= float(vgz.shape[1]) / 2.
            bar_x = rot[0, 0] * bx + rot[0, 1] * by + shift[0]
            bar_y = rot[1, 0] * bx + rot[1, 1] * by + shift[1]

            for tt in range(source_xyz.shape[1]):

                # Calculate out the angles in azimuth and elevation for the bounce
                tx = bar_x - source_xyz[0, tt]
                ty = bar_y - source_xyz[1, tt]
                tz = bar_z - source_xyz[2, tt]
                rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

                rx = bar_x - receive_xyz[0, tt]
                ry = bar_y - receive_xyz[1, tt]
                rz = bar_z - receive_xyz[2, tt]
                r_rng = math.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
                r_el = -math.asin(rz / r_rng)
                r_az = math.atan2(-ry, rx) + np.pi / 2
                if debug_flag and tt == 0:
                    calc_pts[0, px, py] = rx
                    calc_pts[1, px, py] = ry
                    calc_pts[2, px, py] = rz
                    calc_angs[0, px, py] = r_el
                    calc_angs[1, px, py] = r_az

                two_way_rng = rng + r_rng
                rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
                but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1

                # if debug_flag and tt == 0:
                #     calc_angs[2, px, py] = gpr

                if n_samples > but > 0:
                    # a = abs(b_x * rx / r_rng + b_y * ry / r_rng + b_z * rz / r_rng)
                    reflectivity = 1.  # math.pow((1. / -a + 1.) / 20, 10)
                    att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt],
                                                pantx[tt], eltx[tt], bw_az, bw_el) / (two_way_rng * two_way_rng)
                    if debug_flag and tt == 0:
                        calc_angs[2, px, py] = att
                    acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * gpr * reflectivity
                    cuda.atomic.add(pd_r, (but, np.uint16(tt)), acc_val.real)
                    cuda.atomic.add(pd_i, (but, np.uint16(tt)), acc_val.imag)
                cuda.syncthreads()


@jax.jit
def checkIntersection(return_sph, is_block):
    pt_idx, tt = cuda.grid(ndim=2)
    if pt_idx < return_sph.shape[0] and tt < return_sph.shape[1]:
        for idx in range(return_sph.shape[0]):
            if idx == pt_idx:
                continue
            ang_diff = math.sqrt(diff(return_sph[pt_idx, tt, 0], return_sph[idx, tt, 0]) *
                                 diff(return_sph[pt_idx, tt, 0], return_sph[idx, tt, 0]) +
                                 diff(return_sph[pt_idx, tt, 1], return_sph[idx, tt, 1]) *
                                 diff(return_sph[pt_idx, tt, 1], return_sph[idx, tt, 1]))
            if ang_diff < 0.01 and return_sph[pt_idx, tt, 2] > return_sph[idx, tt, 2]:
                is_block[pt_idx, tt] = True


@jax.jit
def genRangeProfileFromMesh(ret_xyz, bounce_xyz, receive_xyz, return_pow, is_blocked, panrx, elrx, pantx,
                            eltx, pd_r, pd_i, debug_att, wavelength, near_range_s, source_fs, bw_az, bw_el):
    pt_idx, tt = cuda.grid(ndim=2)
    if tt < pd_r.shape[1] and pt_idx < ret_xyz.shape[0]:
        if is_blocked[pt_idx, tt]:
            return
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        # Convert point to Cartesian coords and get receiver angles
        tx = ret_xyz[pt_idx, tt, 0]
        ty = ret_xyz[pt_idx, tt, 1]
        tz = ret_xyz[pt_idx, tt, 2]

        rx = tx - receive_xyz[0, tt]
        ry = ty - receive_xyz[1, tt]
        rz = tz - receive_xyz[2, tt]
        r_rng = math.sqrt(rx * rx + ry * ry + rz * rz)
        r_el = math.acos(rz / r_rng)
        r_az = math.atan2(rx, ry)
        two_way_rng = math.sqrt(tx * tx + ty * ty + tz * tz) + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin) if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if n_samples > but > 0:
            # Get bounce vector dot product
            a = bounce_xyz[pt_idx, tt, 0] * rx / r_rng + bounce_xyz[pt_idx, tt, 1] * ry / r_rng + \
                bounce_xyz[pt_idx, tt, 2] * rz / r_rng
            # Run scattering coefficient through scaling to simulate real-world scattering
            P = return_pow[pt_idx, tt]
            sigma = .04 * P
            reflectivity = 1  # math.exp(-math.pow((a * a / (2 * sigma)), P))
            att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt], bw_az, bw_el) * \
                  1 / (two_way_rng * two_way_rng) * reflectivity
            debug_att[pt_idx, tt] = att
            acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng)
            cuda.atomic.add(pd_r, (but, np.uint64(tt)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint64(tt)), acc_val.imag)

@jax.jit
def barycentric(bx, by, px, py, vgz, vert_reflectivity):
    # Apply barycentric interpolation to get random point height and power
    x2 = px - 1
    x2mask = bx - px > 0
    x2 = jnp.where(x2mask, px + 1, x2)
    x2 = jnp.where(x2 < 0, 1, x2)
    x2 = jnp.where(x2 >= px.shape[0], px.shape[0] - 2, x2)
    # y2 = py.copy()
    # x3 = px.copy()
    y3 = py - 1
    y3mask = by - py > 0
    y3 = jnp.where(y3mask, py + 1, y3)
    y3 = jnp.where(y3 < 0, 1, y3)
    y3 = jnp.where(y3 >= py.shape[1], py.shape[1] - 2, y3)

    z3 = vgz[px, y3]
    r3 = vert_reflectivity[px, y3]
    z2 = vgz[x2, py]
    r2 = vert_reflectivity[x2, py]

    bary_determinant = (px - x2) * (py - y3)

    lam1 = ((py - y3) * (bx - px) + (px - x2) * (by - y3)) / bary_determinant
    lam2 = ((y3 - py) * (bx - px)) / bary_determinant
    lam3 = 1 - lam1 - lam2

    # Quick check to see if something's out of whack with the interpolation
    # lam3 + lam1 + lam2 should always be one
    bar_z = vgz[px, py] * lam1 + lam2 * z2 + lam3 * z3
    gpr = r3.copy()
    lmask1 = jnp.logical_and(lam1 - lam2 < 0, lam1 - lam3 < 0)
    lmask2 = jnp.logical_and(lam1 - lam2 > 0, lam2 - lam3 < 0)
    gpr = jnp.where(lmask1, vert_reflectivity, gpr)
    gpr = jnp.where(lmask2, r2, gpr)

    bx -= float(vgz.shape[0]) / 2.
    by -= float(vgz.shape[1]) / 2.
    return bx, by, bar_z, gpr


@jax.jit
def genRangeWithoutIntersection(rot, shift, vgz, vert_reflectivity,
                                source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pdata,
                                wavelength, near_range_s, source_fs, bw_az, bw_el, pts_per_tri):
    # Load in all the parameters that don't change
    wavenumber = 2 * np.pi / wavelength
    ran_key = jax.random.PRNGKey(42)

    for _ in pts_per_tri:
        ran_key, *subkey = jax.random.split(ran_key, 3)
        px, py = jnp.meshgrid(jnp.arange(vert_reflectivity.shape[0]), jnp.arange(vert_reflectivity.shape[1]))
        bx = px + .5 - jax.random.uniform(subkey[0], shape=px.shape)
        by = py + .5 - jax.random.uniform(subkey[1], shape=py.shape)

        # Fix the random points outside the grid - this may make the edges have more points
        bx = jnp.where(bx < 0, .1, bx)
        bx = jnp.where(bx >= px.shape[0], px.shape[0] - 1.1, bx)
        by = jnp.where(by < 0, .1, by)
        by = jnp.where(by >= py.shape[1], py.shape[1] - 1.1, by)

        # Apply barycentric interpolation to get random point height and power
        bx, by, bar_z, gpr = barycentric(bx, by, px, py, vgz, vert_reflectivity)
        bar_x = rot[0, 0] * bx + rot[0, 1] * by + shift[0]
        bar_y = rot[1, 0] * bx + rot[1, 1] * by + shift[1]

        for tt in range(source_xyz.shape[1]):
            # Calculate out the angles in azimuth and elevation for the bounce
            tx = bar_x - source_xyz[0, tt]
            ty = bar_y - source_xyz[1, tt]
            tz = bar_z - source_xyz[2, tt]
            rng = jnp.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

            rx = bar_x - receive_xyz[0, tt]
            ry = bar_y - receive_xyz[1, tt]
            rz = bar_z - receive_xyz[2, tt]
            r_rng = jnp.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
            r_el = -jnp.arcsin(rz / r_rng)
            r_az = jnp.arctan2(rx, ry)

            two_way_rng = rng + r_rng
            # a = abs(b_x * rx / r_rng + b_y * ry / r_rng + b_z * rz / r_rng)
            reflectivity = 1.  # math.pow((1. / -a + 1.) / 20, 10)
            att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt],
                                        pantx[tt], eltx[tt], bw_az, bw_el) / \
                  (two_way_rng * two_way_rng * two_way_rng * two_way_rng)
            # att = 1
            but = jnp.floor((two_way_rng / c0 - 2 * near_range_s) * source_fs).astype(int)
            but = jnp.where(but < 0, 0, but)
            but = jnp.where(but > pdata.shape[1], pdata.shape[1] - 1, but).flatten()
            pdata = pdata.at[tt, but].add(att.flatten() * jnp.exp(-1j * wavenumber * two_way_rng.flatten()) * gpr.flatten()
                                          * reflectivity)
    return pdata


@jax.jit
def backproject(source_xyz, receive_xyz, gx, gy, gz, rbins, panrx, elrx, pantx, eltx, pulse_data, final_grid,
                wavelength, near_range_s, source_fs, signal_bw, bw_az, bw_el, range_atmos):
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
        but = jnp.where(but > len(pdata), len(pdata) - 1, but).flatten()

        # Attenuation of beam in elevation and azimuth
        att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt],
                                    bw_az, bw_el) / two_way_rng

        # Azimuth window to reduce sidelobes
        # Gaussian window
        # az_win = math.exp(-az_diffrx * az_diffrx / (2 * .001))
        # Raised Cosine window (a0=.5 for Hann window, .54 for Hamming)
        az_win = raisedCosine(az_diffrx, signal_bw, .5)
        # az_win = 1.

        bi0 = jnp.floor(but).flatten().astype(int)
        bi1 = jnp.ceil(but).flatten().astype(int)

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
        ra = range_atmos[tt] if range_atmos is not None else 1
        exp_phase = k * two_way_rng * ra
        final_grid = final_grid.at[but].add(cp[bi1] * jnp.exp(1j * exp_phase) * att * az_win)
    return final_grid


@jax.jit
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


if __name__ == '__main__':

    from platform_helper import SDRPlatform
    from grid_helper import MapEnvironment
    from SDRParsing import SDRParse
    import matplotlib.pyplot as plt

    fnme = '/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar'
    sar = SDRParse(fnme, progress_tracker=True)
    bg = MapEnvironment(origin=(40.1355753924, -111.6599735794, 1361.83484), extent=(500, 500))
    vref = np.random.rand(*bg.shape)
    vref[220:270, 220:270] += 1e7
    rp = SDRPlatform(sar, origin=bg.origin)
    near_range_s = rp.calcRanges(0)[0] / c0
    nsam = rp.calcNumSamples(1542, .5)
    rot = bg.transforms[0]
    shift = bg.transforms[1]
    vgz = bg.refgrid
    vert_reflectivity = vref
    wavelength = c0 / 9.6e9
    mfilt = sar.genMatchedFilter(0, fft_len=findPowerOf2(nsam))
    chirp = jnp.fft.fft(sar[0].cal_chirp, len(mfilt))

    for tt in range(0, 1000, 128):
        pdata = np.zeros((32, nsam), dtype=np.complex128)
        ts = rp.gpst[tt:tt + 32]
        source_xyz = rp.txpos(ts)
        receive_xyz = rp.rxpos(ts)
        panrx = rp.pan(ts)
        pantx = rp.pan(ts)
        elrx = rp.tilt(ts)
        eltx = rp.tilt(ts)
        test = genRangeWithoutIntersection(rot, shift, vgz, vert_reflectivity,
            source_xyz, receive_xyz, rp.pan(ts), rp.tilt(ts), rp.pan(ts), rp.tilt(ts), pdata,
            wavelength, near_range_s, rp.fs, rp.az_half_bw, rp.el_half_bw, np.array([0]))

        mfilt_data = jnp.fft.ifft(jnp.fft.fft(test, len(mfilt), axis=1) * chirp * mfilt, axis=1)[:, :nsam]

        plt.figure(f'time {tt}')
        plt.imshow(db(np.array(mfilt_data)).T)
        plt.axis('tight')
    plt.show()