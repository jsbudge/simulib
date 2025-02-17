import cmath
import math
from numba import cuda
import numpy as np
c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
RAYLEIGH_SIGMA_COEFF = .655
CUTOFF = 200000
BOX_CUSHION = .1


@cuda.jit(device=True)
def selectPointOnTriangle(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, rv, ru):
    rw = 1 - rv - ru
    rx = v0x * rv + v1x * ru + v2x * rw
    ry = v0y * rv + v1y * ru + v2y * rw
    rz = v0z * rv + v1z * ru + v2z * rw
    return rx, ry, rz


@cuda.jit()
def applyBeamPatternProjection(x, y, bx, by):
    # bx is equal to tan(el_bw) * d, where d is the depth of the projection
    # by is the same but with azimuth
    # x and y are the projected coordinates of a point from 3d to 2d space.
    return (math.sin(np.pi / bx * x) / (np.pi / bx * x))**2 * (math.sin(np.pi / by * y) / (np.pi / by * y))**2


@cuda.jit()
def cart2bary(x, y, v0x, v0y, v1x, v1y, v2x, v2y):
    det = 1 / ((v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y))
    lam1 = ((v1y - v2y) * (x - v2x) + (v2x - v1x) * (y - v2y)) * det
    lam2 = ((v2y - v0y) * (x - v2x) + (v0x - v2x) * (y - v2y)) * det
    lam3 = 1 - lam2 - lam1
    return lam1, lam2, lam3


@cuda.jit(device=True)
def checkOcclusion(proj_verts, tri_idx, tri_occ_idx, px, py, pz):
    t0x = proj_verts[tri_idx[tri_occ_idx, 0], 0]
    t0y = proj_verts[tri_idx[tri_occ_idx, 0], 1]
    t0z = proj_verts[tri_idx[tri_occ_idx, 0], 2]
    t1x = proj_verts[tri_idx[tri_occ_idx, 1], 0]
    t1y = proj_verts[tri_idx[tri_occ_idx, 1], 1]
    t1z = proj_verts[tri_idx[tri_occ_idx, 1], 2]
    t2x = proj_verts[tri_idx[tri_occ_idx, 2], 0]
    t2y = proj_verts[tri_idx[tri_occ_idx, 2], 1]
    t2z = proj_verts[tri_idx[tri_occ_idx, 2], 2]
    if pz < t0z and pz < t1z and pz < t2z:
        return False
    v0x = t0x / t0z
    v0y = t0y / t0z
    v1x = t1x / t1z
    v1y = t1y / t1z
    v2x = t2x / t2z
    v2y = t2y / t2z

    # Check to see if the point lies within the triangle
    as_x = px - v0x
    as_y = py - v0y
    s_ab = (v1x - v0x) * as_y - (v2y - v0y) * as_x > 0
    if ((v2x - v0x) * as_y - (v2y - v0y) * as_x > 0) == s_ab:
        return False
    if ((v2x - v1x) * (py - v1y) - (v2y - v1y) * (px - v1x) > 0) != s_ab:
        return False

    # Get interpolated depth and check against that
    l0, l1, l2 = cart2bary(px, py, t0x, t0y,
                           t1x, t1y,
                           t2x, t2y)
    if l0 * t0z + l1 * t1z + l2 * t2z > pz:
        return False

    # If it fails all the non-occlusion checks, return True
    return True


@cuda.jit(device=True)
def calcPowerReturn(px, py, pz, tan_bw_az, tan_bw_el, rho, sigma, tnx, tny, tnz):
    corr_bw_az = np.pi / (pz * tan_bw_az)
    corr_bw_el = np.pi / (pz * tan_bw_el)
    att = math.sin(corr_bw_az * py)**4 / (corr_bw_az * py)**4 * math.sin(corr_bw_el * px)**4 / (corr_bw_el * px)**4
    rng = math.sqrt(px**2 + py**2 + pz**2)
    inv_rng = 1 / rng

    pxn = px * inv_rng
    pyn = py * inv_rng
    pzn = pz * inv_rng

    # Calculate out bounce vector
    b = 2 * (pxn * tnx + pyn * tny + pzn * tnz)
    bx = pxn - tnx * b
    by = pyn - tny * b
    bz = pzn - tnz * b

    delta = bx * pxn + by * pyn + bz * pzn

    gamma = 1 - math.exp(-(1 + delta)**2 / (2 * sigma)**2)

    return rho * att * att * (-.9 / 5 * sigma + 1) * gamma * (2 * inv_rng)**2, rng


@cuda.jit(device=True)
def getRayDir(x, y, z, ix, iy, iz):
    occ_x = x - ix
    occ_y = y - iy
    occ_z = z - iz
    inv_occ = 1 / math.sqrt(occ_x ** 2 + occ_y ** 2 + occ_z ** 2)
    occ_x *= inv_occ
    occ_y *= inv_occ
    occ_z *= inv_occ
    return occ_x, occ_y, occ_z


@cuda.jit(device=True)
def checkBox(ro_x, ro_y, ro_z, ray_x, ray_y, ray_z, boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz):
    tmin = -np.inf
    tmax = np.inf
    tx1 = (boxminx - ro_x) / ray_x
    tx2 = (boxmaxx - ro_x) / ray_x
    tmin = max(tmin, min(tx1, tx2))
    tmax = min(tmax, max(tx1, tx2))
    tx1 = (boxminy - ro_y) / ray_y
    tx2 = (boxmaxy - ro_y) / ray_y
    tmin = max(tmin, min(tx1, tx2))
    tmax = min(tmax, max(tx1, tx2))
    tx1 = (boxminz - ro_z) / ray_z
    tx2 = (boxmaxz - ro_z) / ray_z
    tmin = max(tmin, min(tx1, tx2))
    tmax = min(tmax, max(tx1, tx2))
    return tmax - tmin >= 0 and tmax >= 0


@cuda.jit(device=True)
def project(x, y, z, rx, ry, rz, pan, tilt, make_2d):
    el = 3 * np.pi / 2 - tilt
    az = -pan
    sx = x - rx
    sy = y - ry
    sz = z - rz
    px = math.cos(az) * sx + math.sin(az) * sy
    py = -math.sin(az) * math.cos(el) * sx + math.cos(az) * math.cos(el) * sy + math.sin(el) * sz
    pz = math.sin(az) * math.sin(el) * sx - math.cos(az) * math.sin(el) * sy + math.cos(el) * sz
    if make_2d:
        return px / pz, py / pz, 1
    else:
        return px, py, pz


@cuda.jit()
def calcProjectionReturn(source_xyz, ray_power, bounding_box, tri_box_idx, tri_vert, tri_proj_vert, tri_idx,
                   tri_norm, tri_sigma, pd_r, pd_i, pan, tilt, wavelength,
                   near_range_s, source_fs, bw_az, bw_el, tt):
    box, ti = cuda.grid(ndim=2)
    if ti < tri_idx.shape[0] and box < bounding_box.shape[0]:
        n_samples = pd_r.shape[1]
        wavenumber = 2 * np.pi / wavelength
        rx = source_xyz[0]
        ry = source_xyz[1]
        rz = source_xyz[2]
        t0x = tri_vert[tri_idx[ti, 0], 0]
        t0y = tri_vert[tri_idx[ti, 0], 1]
        t0z = tri_vert[tri_idx[ti, 0], 2]
        t1x = tri_vert[tri_idx[ti, 1], 0]
        t1y = tri_vert[tri_idx[ti, 1], 1]
        t1z = tri_vert[tri_idx[ti, 1], 2]
        t2x = tri_vert[tri_idx[ti, 2], 0]
        t2y = tri_vert[tri_idx[ti, 2], 1]
        t2z = tri_vert[tri_idx[ti, 2], 2]
        boxxmin = bounding_box[box, 0, 0]
        boxymin = bounding_box[box, 0, 1]
        boxzmin = bounding_box[box, 0, 2]
        boxxmax = bounding_box[box, 1, 0]
        boxymax = bounding_box[box, 1, 1]
        boxzmax = bounding_box[box, 1, 2]
        for u in range(2):
            for v in range(2):
                if u + v > 2:
                    continue
                intx, inty, intz = selectPointOnTriangle(t0x, t0y, t0z, t1x, t1y, t1z, t2x, t2y, t2z, u / 2., v / 2.)
                rdx, rdy, rdz = getRayDir(intx, inty, intz, rx, ry, rz)
                ipx, ipy, ipz = project(intx, inty, intz, rx, ry, rz, pan, tilt, False)
                ippx, ippy, ippz = project(intx, inty, intz, rx, ry, rz, pan, tilt, True)
                curr_rng = np.inf
                curr_rho = 0.
                if checkBox(rx, ry, rz, rdx, rdy, rdz, boxxmin, boxymin, boxzmin, boxxmax, boxymax, boxzmax):
                    for t_idx in range(tri_box_idx.shape[0]):
                        if not checkOcclusion(tri_proj_vert, tri_idx, t_idx, ippx, ippy, ipz):
                            tnx, tny, tnz = project(tri_norm[t_idx, 0], tri_norm[t_idx, 1], tri_norm[t_idx, 2], rx, ry, rz, pan, tilt, False)
                            rho_o, rng = calcPowerReturn(ipx, ipy, ipz, bw_az, bw_el, ray_power, tri_sigma[t_idx],
                                            tnx, tny, tnz)
                            if rng < curr_rng:
                                curr_rng = rng
                                curr_rho = rho_o + 0.

                if curr_rho > 0:
                    rng_bin = (rng / c0 - 2 * near_range_s) * source_fs
                    but = int(rng_bin)

                    if n_samples > but > 0:
                        acc_val = rho_o * cmath.exp(-1j * wavenumber * rng)
                        cuda.atomic.add(pd_r, (tt, but), acc_val.real)
                        cuda.atomic.add(pd_i, (tt, but), acc_val.imag)


