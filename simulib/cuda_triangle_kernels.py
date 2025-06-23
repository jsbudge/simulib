import cmath
import math
from .cuda_functions import make_float3, cross, dot, length, normalize, make_uint3, rotate
from numba import cuda, prange
import numpy as np
from .cuda_kernels import applyOneWayRadiationPattern, getRangeAndAngles

c0 = 299792458.0
MAX_REGISTERS = 128


@cuda.jit(device=True, fast_math=True)
def calcReturn(inter, re, pan, tilt, bw_az, bw_el, rho):
    r, r_rng, r_az, r_el = getRangeAndAngles(inter, re)

    return applyOneWayRadiationPattern(r_el, r_az, pan, tilt, bw_az, bw_el) / (4 * r_rng * r_rng) * rho

@cuda.jit(device=True, fast_math=True)
def calcTriangleSurfaceArea(ro, p0, p1, p2, Rm, Rp):
    pcycle = cuda.local.array((7, 4), dtype=np.float32)
    pnum = 0

    # This is for l0
    lmm, lmp = lambdaFromRange(ro, p2, p1, Rm)
    lpm, lpp = lambdaFromRange(ro, p2, p1, Rp)
    if 0 < lmp < 1:
        pcycle[pnum, 2] = lmp
        pcycle[pnum, 1] = 1. - lmp
        pnum += 1
    if 0 < lpp < 1:
        pcycle[pnum, 2] = lpp
        pcycle[pnum, 1] = 1. - lpp
        pnum += 1
    if 0 < lpm < 1:
        pcycle[pnum, 2] = lpm
        pcycle[pnum, 1] = 1. - lpm
        pnum += 1
    if 0 < lmm < 1:
        pcycle[pnum, 2] = lmm
        pcycle[pnum, 1] = 1. - lmm
        pnum += 1
    if (1 > lmm > 0 > lpm) or (1 > lmp > 0 > lpp):
        pcycle[pnum, 0] = 0.
        pcycle[pnum, 1] = 1.
        pcycle[pnum, 2] = 0.
        pnum += 1
    elif (1 > lpm > 0 > lmm) or (1 > lpp > 0 > lmp):
        pcycle[pnum, 0] = 0.
        pcycle[pnum, 1] = 1.
        pcycle[pnum, 2] = 0.
        pnum += 1

    # Repeat with side l2
    lmm, lmp = lambdaFromRange(ro, p1, p0, Rm)
    lpm, lpp = lambdaFromRange(ro, p1, p0, Rp)
    if 0 < lmp < 1:
        pcycle[pnum, 1] = lmp
        pcycle[pnum, 0] = 1. - lmp
        pnum += 1
    if 0 < lpp < 1:
        pcycle[pnum, 1] = lpp
        pcycle[pnum, 0] = 1. - lpp
        pnum += 1
    if 0 < lpm < 1:
        pcycle[pnum, 1] = lpm
        pcycle[pnum, 0] = 1. - lpm
        pnum += 1
    if 0 < lmm < 1:
        pcycle[pnum, 1] = lmm
        pcycle[pnum, 0] = 1. - lmm
        pnum += 1
    if (1 > lmm > 0 > lpm) or (1 > lmp > 0 > lpp):
        pcycle[pnum, 0] = 1.
        pcycle[pnum, 1] = 0.
        pcycle[pnum, 2] = 0.
        pnum += 1
    elif (1 > lpm > 0 > lmm) or (1 > lpp > 0 > lmp):
        pcycle[pnum, 0] = 1.
        pcycle[pnum, 1] = 0.
        pcycle[pnum, 2] = 0.
        pnum += 1

    # Repeat with side l1
    lmm, lmp = lambdaFromRange(ro, p0, p2, Rm)
    lpm, lpp = lambdaFromRange(ro, p0, p2, Rp)
    if 0 < lmp < 1:
        pcycle[pnum, 0] = lmp
        pcycle[pnum, 2] = 1. - lmp
        pnum += 1
    if 0 < lpp < 1:
        pcycle[pnum, 0] = lpp
        pcycle[pnum, 2] = 1. - lpp
        pnum += 1
    if 0 < lpm < 1:
        pcycle[pnum, 0] = lpm
        pcycle[pnum, 2] = 1. - lpm
        pnum += 1
    if 0 < lmm < 1:
        pcycle[pnum, 0] = lmm
        pcycle[pnum, 2] = 1. - lmm
        pnum += 1
    if (1 > lmm > 0 > lpm) or (1 > lmp > 0 > lpp):
        pcycle[pnum, 0] = 0.
        pcycle[pnum, 1] = 0.
        pcycle[pnum, 2] = 1.
        pnum += 1
    elif (1 > lpm > 0 > lmm) or (1 > lpp > 0 > lmp):
        pcycle[pnum, 0] = 0.
        pcycle[pnum, 1] = 0.
        pcycle[pnum, 2] = 1.
        pnum += 1

    '''if pnum == 0:
        pcycle[0, 0] = 1.
        pcycle[0, 1] = 0.
        pcycle[0, 2] = 0.
        pcycle[1, 0] = 0.
        pcycle[1, 1] = 1.
        pcycle[1, 2] = 0.
        pcycle[2, 0] = 0.
        pcycle[2, 1] = 0.
        pcycle[2, 2] = 1.
        pnum += 3

        pcycle[1, 3] = 1.
        crossprod = .5 * length(cross(p0 - p1, p0 - p2))'''
    ref_pt = pcycle[0, 0] * p0 + pcycle[0, 1] * p1 + pcycle[0, 2] * p2
    crossprod = .5 * length(cross((pcycle[2, 0] * p0 + pcycle[2, 1] * p1 + pcycle[2, 2] * p2) - ref_pt,
                                             (pcycle[1, 0] * p0 + pcycle[1, 1] * p1 + pcycle[1, 2] * p2) - ref_pt))
    pcycle[1, 3] = crossprod
    for n in range(2, pnum - 1):
        sa = length(cross((pcycle[n, 0] * p0 + pcycle[n, 1] * p1 + pcycle[n, 2] * p2) - ref_pt,
                                     (pcycle[n + 1, 0] * p0 + pcycle[n + 1, 1] * p1 + pcycle[
                                         n + 1, 2] * p2) - ref_pt)) * .5
        crossprod = crossprod + sa
        pcycle[n, 3] = sa
    sa_sum = 0
    for n in range(7):
        if n < pnum:
            sa_sum += pcycle[n, 3] / crossprod
            pcycle[n, 3] = sa_sum
        else:
            pcycle[n, 3] = -1
    return crossprod, pcycle



@cuda.jit(device=True, fast_math=True)
def lambdaFromRange(r0, p0, p1, R):
    # Solving for l2 == 0 and l1 = 1 - l0
    a = 2 * p0.x * p1.x + 2 * p0.y * p1.y + 2 * p0.z * p1.z - p0.x ** 2 - p0.y ** 2 - p0.z ** 2 - p1.x ** 2 - \
        p1.y ** 2 - p1.z ** 2
    c = 2 * p1.x * r0.x + 2 * p1.y * r0.y + 2 * p1.z * r0.z - p1.x ** 2 - p1.y ** 2 - p1.z ** 2 - r0.x ** 2 - \
        r0.y ** 2 - r0.z ** 2 + R ** 2
    b = -2 * p0.x * p1.x - 2 * p0.y * p1.y - 2 * p0.z * p1.z + 2 * p0.x * r0.x + 2 * p0.y * r0.y + 2 * p0.z * \
        r0.z - 2 * p1.x * r0.x - 2 * p1.y * r0.y - 2 * p1.z * r0.z + 2 * p1.x ** 2 + 2 * p1.y ** 2 + 2 * p1.z ** 2

    l_tp = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    l_tm = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return min(l_tp, l_tm), max(l_tp, l_tm)

@cuda.jit(device=True, fast_math=True)
def sampleBinArea(pcycle, p0, p1, p2, rand, idx):
    ran = xoroshiro128p_uniform_float32(rand, idx)
    t0 = pcycle[0, 0] * p0 + pcycle[0, 1] * p1 + pcycle[0, 2] * p2
    for n in range(1, pcycle.shape[0] - 1):
        if ran < pcycle[n, 3] or pcycle[n, 3] == 1.:
            t1 = pcycle[n, 0] * p0 + pcycle[n, 1] * p1 + pcycle[n, 2] * p2
            t2 = pcycle[n + 1, 0] * p0 + pcycle[n + 1, 1] * p1 + pcycle[n + 1, 2] * p2
            r1 = math.sqrt(xoroshiro128p_uniform_float32(rand, idx))
            r2 = xoroshiro128p_uniform_float32(rand, idx)
            return (1 - r1) * t0 + r1 * (1 - r2) * t1 + r1 * r2 * t2
    return None


@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleRangeMinMax(transmit_xyz, tri_verts, tri_idxes, triangle_ranges, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, tri_stride = cuda.gridsize(2)
    for tri_idx in prange(r, tri_idxes.shape[0], tri_stride):
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        for tt in prange(t, transmit_xyz.shape[0], tt_stride):
            tx = make_float3(transmit_xyz[tt, 0], transmit_xyz[tt, 1], transmit_xyz[tt, 2])
            rmin = min(length(tx - p2), length(tx - p0), length(tx - p1))
            rmax = max(length(tx - p2), length(tx - p0), length(tx - p1))
            if rmin > params[1] * c0:
                triangle_ranges[tri_idx, 0] = int(math.floor(((2 * rmin) / c0 - 2 * params[1]) * params[2]))
                triangle_ranges[tri_idx, 1] = int(math.ceil(((2 * rmax) / c0 - 2 * params[1]) * params[2]) -
                                                   math.floor(((2 * rmin) / c0 - 2 * params[1]) * params[2]))
            else:
                triangle_ranges[tri_idx, 0] = 0


@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleBinSurfaceArea(transmit_xyz, tri_verts, tri_idxes, triangle_ranges, triangle_bin_surface_area, params):
    tri, rbin = cuda.grid(ndim=2)
    tri_stride, bin_stride = cuda.gridsize(2)
    tx = make_float3(transmit_xyz[0, 0], transmit_xyz[0, 1], transmit_xyz[0, 2])
    for tri_idx in prange(tri, tri_idxes.shape[0], tri_stride):
        if triangle_ranges[tri_idx, 0] == 0:
            continue
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        for b in prange(rbin, triangle_bin_surface_area.shape[1], bin_stride):
            if b < triangle_ranges[tri_idx, 1]:
                R = ((triangle_ranges[tri_idx, 0] + b) / params[2] + 2 * params[1]) * c0 * .5
                sa, pcycle = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (1 / params[2]) * c0 * .5)
                triangle_bin_surface_area[tri_idx, b] = sa



@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleSampleVariance(transmit_xyz, tri_verts, tri_idxes, triangle_ranges, triangle_bin_surface_area,
                               triangle_sample_variance, pan, tilt, params):
    tri, rbin = cuda.grid(ndim=2)
    tri_stride, bin_stride = cuda.gridsize(2)
    tx = make_float3(transmit_xyz[0, 0], transmit_xyz[0, 1], transmit_xyz[0, 2])
    for tri_idx in prange(tri, tri_idxes.shape[0], tri_stride):
        if triangle_ranges[tri_idx, 0] == 0:
            continue
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        for b in prange(rbin, triangle_bin_surface_area.shape[1], bin_stride):
            if b < triangle_ranges[tri_idx, 1]:
                R = ((triangle_ranges[tri_idx, 0] + b) / params[2] + 2 * params[1]) * c0 * .5
                sa, pcycle = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (1 / params[2]) * c0 * .5)
                r_min = np.inf
                r_max = -np.inf
                for n in range(pcycle.shape[0]):
                    if pcycle[n, 3] != -1:
                        pt = p0 * pcycle[n, 0] + p1 * pcycle[n, 1] + p2 * pcycle[n, 2]
                        rng = length(tx - pt)
                        rd = (pt - tx) / rng
                        nrho = (params[5] * applyOneWayRadiationPattern(pan[0], tilt[0], math.atan2(rd.x, rd.y),
                                                                        -math.asin(rd.z), params[3], params[4]))
                        ret = calcReturn(pt, tx, pan[0], tilt[0], params[3], params[4], nrho)
                        r_min = min(r_min, ret)
                        r_max = max(r_max, ret)
                triangle_bin_surface_area[tri_idx, b] = sa
                triangle_sample_variance[tri_idx, b] = abs(r_max - r_min) * sa


@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleReturns(transmit_xyz, tri_verts, tri_idxes, tri_norm, tri_material, kd_tree, leaf_list,
                        leaf_key, pd_r, pd_i, receive_xyz, pan, tilt, target_samples, rands, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, tri_stride = cuda.gridsize(2)
    for tri_idx in prange(r, tri_idxes.shape[0], tri_stride):
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        for tt in prange(t, transmit_xyz.shape[0], tt_stride):
            tx = make_float3(transmit_xyz[tt, 0], transmit_xyz[tt, 1], transmit_xyz[tt, 2])
            rx = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
            r_min = min(length(tx - p0), length(tx - p1), length(tx - p2))
            r_max = max(length(tx - p0), length(tx - p1), length(tx - p2))
            r_min = int(((2 * r_min) / c0 - 2 * params[1]) * params[2])
            R = (r_min / params[2] + 2 * params[1]) * c0 * .5
            # print(r_min, r_max, R)
            while True:
                sa, pcycle = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (1 / params[2]) * c0 * .5)
                sa = sa / target_samples
                for _ in range(target_samples):
                    inter = sampleBinArea(pcycle, p0, p1, p2, rands, tri_idx)
                    rng = length(tx - inter)
                    rd = (inter - tx) / rng
                    nrho = (params[5] * applyOneWayRadiationPattern(pan[tt], tilt[tt], math.atan2(rd.x, rd.y),
                                                                    -math.asin(rd.z), params[3], params[4]))

                    did_intersect, nrho, inter, int_rng, b = traverseOctreeAndReflection(tx, rd, kd_tree, nrho,
                                                                                         leaf_list, leaf_key, tri_idxes,
                                                                                         tri_verts, tri_norm, tri_material,
                                                                                         rng, params[0])
                    if did_intersect:
                        acc_real, acc_imag, but = calcReturnAndBin(inter, rx, rng, params[1], params[2], pd_r.shape[1],
                                                                   pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
                        if but >= 0:
                            acc_real = acc_real * sa if abs(acc_real) < np.inf else 0.
                            acc_imag = acc_imag * sa if abs(acc_imag) < np.inf else 0.
                            cuda.atomic.add(pd_r, (tt, but), acc_real)
                            cuda.atomic.add(pd_i, (tt, but), acc_imag)
                R += (1 / params[2]) * c0 * .5
                if R > r_max:
                    break