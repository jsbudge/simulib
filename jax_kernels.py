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
                             wavelength, near_range_s, source_fs, bw_az, bw_el, rbins, pts_per_tri, power_scaling):
    # Load in all the parameters that don't change
    wavenumber = 2 * np.pi / wavelength
    ran_key = jax.random.PRNGKey(42)
    px, py = jnp.meshgrid(jnp.arange(vert_reflectivity.shape[0]), jnp.arange(vert_reflectivity.shape[1]))
    pdata = jax.lax.fori_loop(0,
                              pts_per_tri,
                              lambda x, y: jax.lax.cond(x != 0,
                                                        gather_data_loop_det,
                                                        gather_data_loop,
                                                        y, ran_key, px, py, source_xyz, receive_xyz, vgz,
                                                        vert_reflectivity,
                                                        source_fs, wavenumber, panrx, elrx, pantx, eltx, bw_az, bw_el,
                                                        rot,
                                                        shift, rbins, near_range_s, power_scaling),
                              jnp.zeros((len(rbins),), dtype=jnp.complex128))
    return pdata


@jax.jit
def gather_data_loop(pdata, ran_key, px, py, source_xyz, receive_xyz, vgz, vert_reflectivity, source_fs, wavenumber,
                     panrx,
                     elrx, pantx, eltx, bw_az, bw_el, rot, shift, rbins, near_range_s, power_scaling):
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
                             gpr * reflectivity) * power_scaling / (rbins - rbins[0] + 1) ** 4


@jax.jit
def gather_data_loop_det(pdata, ran_key, px, py, source_xyz, receive_xyz, vgz, gpr, source_fs, wavenumber, panrx,
                         elrx, pantx, eltx, bw_az, bw_el, rot, shift, rbins, near_range_s, power_scaling):
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
                             gpr.flatten() * reflectivity * power_scaling / two_way_rng ** 4)
