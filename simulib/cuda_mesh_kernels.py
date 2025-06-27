import cmath
import math

from numba.cuda.random import xoroshiro128p_uniform_float32

from .cuda_functions import make_float3, cross, dot, length, normalize, make_uint3, rotate
from numba import cuda, prange
import numpy as np
from .cuda_kernels import applyOneWayRadiationPattern

c0 = 299792458.0
MAX_REGISTERS = 128


@cuda.jit(device=True, fast_math=True)
def getRangeAndAngles(v, s):
    t = v - s
    rng = length(t)
    az = math.atan2(t.x, t.y)
    el = -math.asin(t.z / rng)
    return t, rng, az, el


@cuda.jit(device=True, fast_math=True)
def calcSingleIntersection(rd, ro, v0, v1, v2, vn, get_bounce):
    rcrosse = cross(rd, v2 - v0)
    det = dot(v1 - v0, rcrosse)
    # Check to see if ray is parallel to triangle
    if abs(det) < 1e-9:
        return False, None, None

    inv_det = 1. / det

    u = inv_det * dot(ro - v0, rcrosse)
    if u < 0 or u > 1:
        return False, None, None

    # Recompute cross for s and edge 1
    rcrosse = cross(ro - v0, v1 - v0)
    det = inv_det * dot(rd, rcrosse)
    if det < 0 or u + det > 1:
        return False, None, None

    # Compute intersection point
    det = inv_det * dot(v2 - v0, rcrosse)
    if det < 1e-9:
        return False, None, None

    if not get_bounce:
        return True, None, None

    return True, normalize(det * rd - vn * dot(det * rd, vn) * 2.), ro + det * rd


@cuda.jit(device=True, fast_math=True)
def calcReturnAndBin(inter, re, rng, near_range_s, source_fs, n_samples,
                     pan, tilt, bw_az, bw_el, wavenumber, rho):
    r, r_rng, r_az, r_el = getRangeAndAngles(inter, re)

    rng_bin = int(((rng + r_rng) / c0 - 2 * near_range_s) * source_fs)

    if n_samples > rng_bin > 0:
        acc_val = (applyOneWayRadiationPattern(r_el, r_az, pan, tilt, bw_az, bw_el) /
                   (r_rng * r_rng) * rho * cmath.exp(-1j * wavenumber * (rng + r_rng)))
        return acc_val.real, acc_val.imag, rng_bin

    return 0, 0, -1


@cuda.jit(device=True, fast_math=True)
def calcReturn(inter, re, pan, tilt, bw_az, bw_el, rho):
    r, r_rng, r_az, r_el = getRangeAndAngles(inter, re)

    return applyOneWayRadiationPattern(r_el, r_az, pan, tilt, bw_az, bw_el) / (4 * r_rng * r_rng) * rho


@cuda.jit(device=True, fast_math=True)
def findBoxIntersection(ro, ray, bounds):
    boxmin = make_float3(bounds[0, 0], bounds[0, 1], bounds[0, 2])
    boxmax = make_float3(bounds[1, 0], bounds[1, 1], bounds[1, 2])
    if ray.x >= 0:
        tmin = (boxmin.x - ro.x) * ray.x
        tmax = (boxmax.x - ro.x) * ray.x
    else:
        tmin = (boxmax.x - ro.x) * ray.x
        tmax = (boxmin.x - ro.x) * ray.x
    if ray.y >= 0:
        tminy = (boxmin.y - ro.y) * ray.y
        tmaxy = (boxmax.y - ro.y) * ray.y
    else:
        tminy = (boxmax.y - ro.y) * ray.y
        tmaxy = (boxmin.y - ro.y) * ray.y
    tmin = max(tminy, tmin)
    tmax = min(tmaxy, tmax)

    if ray.z >= 0:
        tminy = (boxmin.z - ro.z) * ray.z
        tmaxy = (boxmax.z - ro.z) * ray.z
    else:
        tminy = (boxmax.z - ro.z) * ray.z
        tmaxy = (boxmin.z - ro.z) * ray.z
    tmin = max(tminy, tmin)
    tmax = min(tmaxy, tmax)
    return tmin, tmax



@cuda.jit(device=True, fast_math=True)
def testIntersection(ro, rd, bounds):
    ray = 1 / rd
    boxmin = make_float3(bounds[0, 0], bounds[0, 1], bounds[0, 2])
    boxmax = make_float3(bounds[1, 0], bounds[1, 1], bounds[1, 2])
    if ray.x >= 0:
        tmin = (boxmin.x - ro.x) * ray.x
        tmax = (boxmax.x - ro.x) * ray.x
    else:
        tmin = (boxmax.x - ro.x) * ray.x
        tmax = (boxmin.x - ro.x) * ray.x
    if ray.y >= 0:
        tminy = (boxmin.y - ro.y) * ray.y
        tmaxy = (boxmax.y - ro.y) * ray.y
    else:
        tminy = (boxmax.y - ro.y) * ray.y
        tmaxy = (boxmin.y - ro.y) * ray.y
    if tmin > tmaxy or tminy > tmax:
        return False

    tmin = max(tminy, tmin)
    tmax = min(tmaxy, tmax)

    if ray.z >= 0:
        tminy = (boxmin.z - ro.z) * ray.z
        tmaxy = (boxmax.z - ro.z) * ray.z
    else:
        tminy = (boxmax.z - ro.z) * ray.z
        tmaxy = (boxmin.z - ro.z) * ray.z
    if tmin <= tmaxy and tminy <= tmax:
        return True
    return False


@cuda.jit(device=True, fast_math=True)
def traverseOctreeForOcclusion(ro, rd, kd_tree, leaf_list, leaf_key, tri_idx, tri_vert, tri_norm):
    """
    Traverse an octree structure, given some triangle indexes, and return the reflected power, bounce angle, and intersection point.
    params:
    ro: float3 = ray origin point
    rd: float3 = normalized ray direction vector
    bounding_box: (N, 2, 3) = array of axis aligned bounding boxes
    leaf_list: (N) = sorted array of triangle indexes based on octree boxes
    leaf_key: (N, 2) = key to look into leaf_list and find the triangles inside of a box
    tri_idx: (N, 3) = indexes of vertices that correspond to an individual triangle
    tri_vert: (N, 3) = vertices for triangles in euclidean space
    tri_norm: (N, 3) = surface normal of triangle
    """
    if not testIntersection(ro, rd, kd_tree[0]):
        return False
    idx = 2
    skip = False

    while 0 < idx < kd_tree.shape[0]:
        if testIntersection(ro, rd, kd_tree[idx]):
            if math.log2(idx + 1) >= math.log2(kd_tree.shape[0] + 1) - 1:
                tri_min = leaf_key[idx, 0]
                for t_idx in prange(tri_min, tri_min + leaf_key[idx, 1]):
                    ti = leaf_list[t_idx]
                    tn = make_float3(tri_norm[ti, 0], tri_norm[ti, 1], tri_norm[ti, 2])
                    curr_intersect, _, _ = (
                        calcSingleIntersection(rd, ro, make_float3(tri_vert[tri_idx[ti, 0], 0],
                                                                   tri_vert[tri_idx[ti, 0], 1],
                                                                   tri_vert[tri_idx[ti, 0], 2]),
                                               make_float3(tri_vert[tri_idx[ti, 1], 0],
                                                           tri_vert[tri_idx[ti, 1], 1],
                                                           tri_vert[tri_idx[ti, 1], 2]),
                                               make_float3(tri_vert[tri_idx[ti, 2], 0],
                                                           tri_vert[tri_idx[ti, 2], 1],
                                                           tri_vert[tri_idx[ti, 2], 2]), tn, False))
                    if curr_intersect:
                        return True
            else:
                # Move down into the box
                idx = idx * 2 + 2
                skip = True
        if not skip:
            while idx % 2 == 1 and idx != 1:
                idx = (idx >> 1)
            if idx == 1:
                break
            idx -= 1
        skip = False
    return False


@cuda.jit(device=True, fast_math=True)
def traverseOctreeAndIntersection(ro, rd, kd_tree, leaf_list,
                                  leaf_key, tri_idx, tri_vert, tri_norm, rng):
    """
    Traverse an octree structure, given some triangle indexes, and return the reflected power, bounce angle, and intersection point.
    params:
    ro: float3 = ray origin point
    rd: float3 = normalized ray direction vector
    bounding_box: (N, 2, 3) = array of axis aligned bounding boxes
    rho: float = ray power in watts
    final_level: int = start index of the final level of the octree for the bounding_box array
    leaf_list: (N) = sorted array of triangle indexes based on octree boxes
    leaf_key: (N, 2) = key to look into leaf_list and find the triangles inside of a box
    tri_idx: (N, 3) = indexes of vertices that correspond to an individual triangle
    tri_vert: (N, 3) = vertices for triangles in euclidean space
    tri_norm: (N, 3) = surface normal of triangle
    tri_material: (N, 3) = material scattering values of triangle - (RCS, ks, kd)
    occlusion_only: bool = set to True to return when the ray intersects something without checking any other triangles
    """

    int_rng = np.inf
    did_intersect = False
    inter = None
    if not testIntersection(ro, rd, kd_tree[0]):
        return False, None, None
    idx = 2
    skip = False

    while 0 < idx < kd_tree.shape[0]:
        # idx, tmp_tmin, tmp_tmax = stack[-1]
        if testIntersection(ro, rd, kd_tree[idx]):
            if math.log2(idx + 1) >= math.log2(kd_tree.shape[0] + 1) - 1:
                tri_min = leaf_key[idx, 0]
                for t_idx in prange(tri_min, tri_min + leaf_key[idx, 1]):
                    ti = leaf_list[t_idx]
                    ti_idx = make_uint3(tri_idx[ti, 0], tri_idx[ti, 1], tri_idx[ti, 2])
                    tn = make_float3(tri_norm[ti, 0], tri_norm[ti, 1], tri_norm[ti, 2])
                    curr_intersect, tb, tinter = (
                        calcSingleIntersection(rd, ro, make_float3(tri_vert[ti_idx.x, 0],
                                                                   tri_vert[ti_idx.x, 1],
                                                                   tri_vert[ti_idx.x, 2]),
                                               make_float3(tri_vert[ti_idx.y, 0],
                                                           tri_vert[ti_idx.y, 1],
                                                           tri_vert[ti_idx.y, 2]),
                                               make_float3(tri_vert[ti_idx.z, 0],
                                                           tri_vert[ti_idx.z, 1],
                                                           tri_vert[ti_idx.z, 2]), tn, True))
                    if curr_intersect:
                        tmp_rng = length(ro - tinter)
                        if 1. < tmp_rng < int_rng:
                            int_rng = tmp_rng + rng
                            inter = tinter + 0.
                            did_intersect = True
            else:
                # Move down into the box
                idx = idx * 2 + 2
                skip = True
        if not skip:
            while idx % 2 == 1 and idx != 1:
                idx = (idx >> 1)
            if idx == 1:
                break
            idx -= 1
        skip = False
    return did_intersect, inter, int_rng


@cuda.jit(device=True, fast_math=True)
def traverseOctreeAndReflection(ro, rd, kd_tree, rho, leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, tri_material,
                                rng, wavenumber):
    """
    Traverse an octree structure, given some triangle indexes, and return the reflected power, bounce angle, and intersection point.
    params:
    ro: float3 = ray origin point
    rd: float3 = normalized ray direction vector
    bounding_box: (N, 2, 3) = array of axis aligned bounding boxes
    rho: float = ray power in watts
    final_level: int = start index of the final level of the octree for the bounding_box array
    leaf_list: (N) = sorted array of triangle indexes based on octree boxes
    leaf_key: (N, 2) = key to look into leaf_list and find the triangles inside of a box
    tri_idx: (N, 3) = indexes of vertices that correspond to an individual triangle
    tri_vert: (N, 3) = vertices for triangles in euclidean space
    tri_norm: (N, 3) = surface normal of triangle
    tri_material: (N, 3) = material scattering values of triangle - (RCS, ks, kd)
    occlusion_only: bool = set to True to return when the ray intersects something without checking any other triangles
    """

    int_rng = np.inf
    did_intersect = False
    inter = None
    if not testIntersection(ro, rd, kd_tree[0]):
        return False, None, None, None, None, None
    idx = 2
    skip = False

    while 0 < idx < kd_tree.shape[0]:
        # idx, tmp_tmin, tmp_tmax = stack[-1]
        if testIntersection(ro, rd, kd_tree[idx]):
            if math.log2(idx + 1) >= math.log2(kd_tree.shape[0] + 1) - 1:
                tri_min = leaf_key[idx, 0]
                for t_idx in prange(tri_min, tri_min + leaf_key[idx, 1]):
                    ti = leaf_list[t_idx]
                    ti_idx = make_uint3(tri_idx[ti, 0], tri_idx[ti, 1], tri_idx[ti, 2])
                    tn = make_float3(tri_norm[ti, 0], tri_norm[ti, 1], tri_norm[ti, 2])
                    curr_intersect, tb, tinter = (
                        calcSingleIntersection(rd, ro, make_float3(tri_vert[ti_idx.x, 0],
                                                                   tri_vert[ti_idx.x, 1],
                                                                   tri_vert[ti_idx.x, 2]),
                                               make_float3(tri_vert[ti_idx.y, 0],
                                                           tri_vert[ti_idx.y, 1],
                                                           tri_vert[ti_idx.y, 2]),
                                               make_float3(tri_vert[ti_idx.z, 0],
                                                           tri_vert[ti_idx.z, 1],
                                                           tri_vert[ti_idx.z, 2]),
                                               tn, True))
                    if curr_intersect:
                        tmp_rng = length(ro - tinter)
                        if 1. < tmp_rng < int_rng:
                            int_rng = tmp_rng + rng
                            inv_rng = 1. / int_rng
                            b = tb + 0.
                            # Some parts of the Fresnel coefficient calculation
                            cosa = abs(dot(rd, tn))
                            sina = tri_material[ti, 0] * math.sqrt(
                                1. - (1. / tri_material[ti, 0] * length(cross(rd, tn))) ** 2)
                            Rs = abs((cosa - sina) / (
                                    cosa + sina)) ** 2  # Reflectance using Fresnel coefficient
                            roughness = math.exp(-.5 * (2. * wavenumber * tri_material[
                                ti, 1] * cosa) ** 2)  # Roughness calculations to get specular/scattering split
                            spec = math.exp(-(1. - cosa) ** 2 / .0000007442)  # This should drop the specular component to zero by 2 degrees
                            L = .7 * ((1 + abs(dot(b, rd))) / 2.) + .3
                            nrho = rho * inv_rng * inv_rng * cosa * Rs * (
                                    roughness * spec + (
                                    1. - roughness) * L ** 2)  # Final reflected power
                            inter = tinter + 0.
                            did_intersect = True
                            inter_tri = ti
            else:
                # Move down into the box
                idx = idx * 2 + 2
                skip = True
        if not skip:
            while idx % 2 == 1 and idx != 1:
                idx = (idx >> 1)
            if idx == 1:
                break
            idx -= 1
        skip = False
    return did_intersect, nrho, inter, int_rng, b, inter_tri


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
    l_tp = -np.inf if math.isnan(l_tp) else l_tp
    l_tm = (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    l_tm = -np.inf if math.isnan(l_tm) else l_tm
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
            return (1 - r1) * t0 + r1 * (1 - r2) * t1 + r1 * r2 * t2, n
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
def calcTriangleReturnsFromVariance(transmit_xyz, pan, tilt, tri_verts, tri_idxes, tri_norm, tri_material, kd_tree,
                                    leaf_key, leaf_list, triangle_ranges, triangle_return_r, triangle_return_i,
                                    triangle_sample_variance, rands, params):
    tri, pulse, rbin = cuda.grid(ndim=3)
    tri_stride, pulse_stride, bin_stride = cuda.gridsize(3)

    for tri_idx in prange(tri, tri_idxes.shape[0], tri_stride):
        if triangle_ranges[tri_idx, 0] == 0:
            continue
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        for t in prange(pulse, transmit_xyz.shape[0], pulse_stride):
            tx = make_float3(transmit_xyz[t, 0], transmit_xyz[t, 1], transmit_xyz[t, 2])
            for b in prange(rbin, triangle_sample_variance.shape[1], bin_stride):
                if b < triangle_ranges[tri_idx, 1]:
                    # if tri_idx == 2:
                    R = ((triangle_ranges[tri_idx, 0] + b) / params[2] + 2 * params[1]) * c0 * .5
                    sa, pcycle = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (1 / params[2]) * c0 * .5)
                    mu = 0j
                    m2 = 0j
                    n = 0
                    error_est = np.inf

                    while n < 10 and (error_est > 1e-5 or n < 3):
                        sample_pt, sample_tri_idx = sampleBinArea(pcycle, p0, p1, p2, rands, tri_idx)
                        ray_dir = sample_pt - tx
                        ray_dir = ray_dir / length(ray_dir)
                        did_intersect, nrho, inter, _, _, inter_tri = traverseOctreeAndReflection(tx, ray_dir, kd_tree, params[5],
                                                                                                  leaf_list,
                                                                                leaf_key, tri_idxes, tri_verts, tri_norm,
                                                                                tri_material, 0, params[0])
                        if did_intersect and inter_tri == tri_idx:
                            r, r_rng, r_az, r_el = getRangeAndAngles(inter, tx)
                            acc_val = (applyOneWayRadiationPattern(r_el, r_az, pan[t], tilt[t], params[3], params[4]) /
                                       (r_rng * r_rng) * nrho * cmath.exp(-1j * params[5] * (2 * r_rng)))
                            # print('intersect')
                            delta = acc_val - mu
                            mu += delta / (n + 1)
                            m2 += delta * (acc_val - mu)
                        else:
                            # print('non-intersect')
                            delta = -mu
                            mu += delta / (n + 1)
                            m2 += delta * -mu
                            pcycle[sample_tri_idx - 1, 3] = pcycle[sample_tri_idx, 3]
                            pcycle[sample_tri_idx, 3] = 0
                        n += 1
                        error_est = sa * math.sqrt(m2.imag**2 / (n - 1.)) / math.sqrt(n)

                    # print(b, mu.real, mu.imag)
                    triangle_sample_variance[tri_idx, b] = error_est
                    cuda.atomic.add(triangle_return_r, (t, triangle_ranges[tri_idx, 0] + b), sa * mu.real)
                    cuda.atomic.add(triangle_return_i, (t, triangle_ranges[tri_idx, 0] + b), sa * mu.imag)



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



@cuda.jit(device=True, fast_math=True)
def barycentricToRange(p0, p1, p2, l0, l1, l2, tx):
    return length(tx - (p0 * l0 + p1 * l1 + p2 * l2))

@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleSampleVariance(transmit_xyz, tri_verts, tri_idxes, triangle_ranges, triangle_bin_surface_area,
                               triangle_sample_variance, pan, tilt, params, target_variance):
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
                triangle_sample_variance[tri_idx, b] = math.ceil((abs(r_max - r_min) * (
                        sa))**2 / target_variance**2)



@cuda.jit()
def calcBounceLoop(ray_origin, ray_dir, ray_distance, ray_power, kd_tree,
                   leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, tri_material, pd_r, pd_i, receive_xyz, pan,
                   tilt, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            # Load in all the parameters that don't change
            rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
            rng = ray_distance[tt, ray_idx]
            did_intersect, nrho, inter, int_rng, b = traverseOctreeAndReflection(
                make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2]),
                make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2]), kd_tree,
                ray_power[tt, ray_idx], leaf_list, leaf_key, tri_idx, tri_vert,
                                                                                   tri_norm, tri_material, rng, params[0])
            if did_intersect:
                rng += int_rng
                # Check for occlusion against the receiver
                if not traverseOctreeForOcclusion(inter, normalize(rec_xyz - inter), kd_tree, leaf_list, leaf_key,
                                                                           tri_idx, tri_vert, tri_norm):
                    acc_real, acc_imag, but = calcReturnAndBin(inter, rec_xyz, rng, params[1], params[2], pd_r.shape[1],
                                                               pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
                    if but >= 0:
                        acc_real = acc_real if abs(acc_real) < np.inf else 0.
                        acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
                        cuda.atomic.add(pd_r, (tt, but), acc_real)
                        cuda.atomic.add(pd_i, (tt, but), acc_imag)

                ray_origin[tt, ray_idx, 0] = inter.x
                ray_origin[tt, ray_idx, 1] = inter.y
                ray_origin[tt, ray_idx, 2] = inter.z
                ray_dir[tt, ray_idx, 0] = b.x
                ray_dir[tt, ray_idx, 1] = b.y
                ray_dir[tt, ray_idx, 2] = b.z
                ray_power[tt, ray_idx] = nrho
            else:
                ray_distance[tt, ray_idx] = rng
                ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcBounceInit(ray_origin, ray_dir, ray_distance, ray_power, kd_tree,
                   leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, tri_material, pd_r, pd_i, receive_xyz, pan,
                   tilt, params, conical_sampling):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
            did_intersect, nrho, inter, rng, b = traverseOctreeAndReflection(rec_xyz, rd,
                                                                             kd_tree, ray_power[tt, ray_idx],
                                                                             leaf_list, leaf_key, tri_idx,
                                                                             tri_vert, tri_norm, tri_material, 0, params[0])
            if did_intersect:
                acc_real, acc_imag, but = calcReturnAndBin(inter, rec_xyz, rng, params[1], params[2], pd_r.shape[1],
                                                           pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
                if but >= 0:
                    acc_real = acc_real if abs(acc_real) < np.inf else 0.
                    acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
                    cuda.atomic.add(pd_r, (tt, but), acc_real)
                    cuda.atomic.add(pd_i, (tt, but), acc_imag)
                    ray_origin[tt, ray_idx, 0] = inter.x
                    ray_origin[tt, ray_idx, 1] = inter.y
                    ray_origin[tt, ray_idx, 2] = inter.z
                    ray_dir[tt, ray_idx, 0] = b.x
                    ray_dir[tt, ray_idx, 1] = b.y
                    ray_dir[tt, ray_idx, 2] = b.z
                    ray_power[tt, ray_idx] = nrho
                    ray_distance[tt, ray_idx] = rng
                # Apply supersampling if wanted
                if params[6]:
                    for n in prange(conical_sampling.shape[0]):
                        if abs(rd.y) > 1e-9:
                            sc = normalize(make_float3(2 * rd.y * rd.z, 0., -2 * rd.x * rd.y))
                        else:
                            sc = normalize(make_float3(-2 * rd.y * rd.z, 2 * rd.x * rd.z, 0.))
                        rd = normalize(inter + rotate(rd, sc, conical_sampling[n, 0]) * conical_sampling[n, 1] - rec_xyz)
                        did_intersect, nrho, cone_inter, rng, b = traverseOctreeAndReflection(rec_xyz, rd, kd_tree,
                                                                                         ray_power[tt, ray_idx],
                                                                                         leaf_list, leaf_key, tri_idx,
                                                                                         tri_vert, tri_norm,
                                                                                         tri_material, 0, params[0])
                        if did_intersect:
                            acc_real, acc_imag, but = calcReturnAndBin(cone_inter, rec_xyz, rng, params[1], params[2],
                                                                       pd_r.shape[1],
                                                                       pan[tt], tilt[tt], params[3], params[4],
                                                                       params[0], nrho)
                            if but >= 0:
                                acc_real = acc_real if abs(acc_real) < np.inf else 0.
                                acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
                                cuda.atomic.add(pd_r, (tt, but), acc_real)
                                cuda.atomic.add(pd_i, (tt, but), acc_imag)
            else:
                ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcBounceWithoutReflect(ray_dir, ray_power, boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, leaf_list, leaf_key, tri_vert, tri_idx,
                   tri_norm, tri_material, pd_r, pd_i, receive_xyz, pan, tilt, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
            rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
            rho = ray_power[tt, ray_idx]
            if rho < 1e-9:
                return
            did_intersect, nrho, inter, rng, _ = traverseOctreeAndReflection(rec_xyz, rd, boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, rho, leaf_list,
                                                                               leaf_key, tri_idx, tri_vert,
                                                                               tri_norm, tri_material, 0, params[0])
            if did_intersect:
                acc_real, acc_imag, but = calcReturnAndBin(inter, rec_xyz, rng, params[1], params[2], pd_r.shape[1],
                                                           pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
                if but >= 0:
                    acc_real = acc_real if abs(acc_real) < np.inf else 0.
                    acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
                    cuda.atomic.add(pd_r, (tt, but), acc_real)
                    cuda.atomic.add(pd_i, (tt, but), acc_imag)


@cuda.jit()
def calcReturnPower(ray_origin, ray_distance, ray_power, ray_poss, pd_r, pd_i, receive_xyz, pan,
                   tilt, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_origin.shape[0], tt_stride):
        for ray_idx in prange(r, ray_origin.shape[1], ray_stride):
            # Check for occlusion against the receiver
            if ray_poss[tt, ray_idx]:
                acc_real, acc_imag, but = calcReturnAndBin(make_float3(ray_origin[tt, ray_idx, 0],
                                                                       ray_origin[tt, ray_idx, 1],
                                                                       ray_origin[tt, ray_idx, 2]),
                                                           make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1],
                                                                       receive_xyz[tt, 2]), ray_distance[tt, ray_idx],
                                                           params[1], params[2], pd_r.shape[1], pan[tt], tilt[tt],
                                                           params[3], params[4], params[0], ray_power[tt, ray_idx])
                if but >= 0:
                    acc_real = acc_real if abs(acc_real) < np.inf else 0.
                    acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
                    cuda.atomic.add(pd_r, (tt, but), acc_real)
                    cuda.atomic.add(pd_i, (tt, but), acc_imag)


@cuda.jit()
def calcSceneOcclusion(ray_origin, ray_poss, boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz,
                   leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, receive_xyz):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_origin.shape[0], tt_stride):
        for ray_idx in prange(r, ray_origin.shape[1], ray_stride):
            inter = make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2])
            # Check for occlusion against the receiver
            if traverseOctreeForOcclusion(inter, normalize(make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2]) - inter), boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, leaf_list, leaf_key,
                                                                       tri_idx, tri_vert, tri_norm):
                ray_poss[tt, ray_idx] = False


@cuda.jit()
def calcOriginDirAtt(rec_xyz, sample_points, pan, tilt, params, ray_dir, ray_origin, ray_power):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rd = normalize(make_float3(sample_points[ray_idx, 0], sample_points[ray_idx, 1], sample_points[ray_idx, 2]) -
                  make_float3(rec_xyz[tt, 0], rec_xyz[tt, 1], rec_xyz[tt, 2]))
            ray_power[tt, ray_idx] = (params[5] *
                                      applyOneWayRadiationPattern(pan[tt], tilt[tt],
                                                                  math.atan2(rd.x, rd.y), -math.asin(rd.z),
                                                                  params[3], params[4]))
            ray_dir[tt, ray_idx, 0] = rd.x
            ray_dir[tt, ray_idx, 1] = rd.y
            ray_dir[tt, ray_idx, 2] = rd.z
            ray_origin[tt, ray_idx, 0] = rec_xyz[tt, 0]
            ray_origin[tt, ray_idx, 1] = rec_xyz[tt, 1]
            ray_origin[tt, ray_idx, 2] = rec_xyz[tt, 2]


@cuda.jit()
def determineSceneRayIntersections(boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, ray_dir, ray_origin, intersects):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            for box in prange(boxminx.shape[0]):
                intersects[tt, ray_idx, box] = findOctreeBox(make_float3(ray_origin[tt, ray_idx, 0],
                                                                         ray_origin[tt, ray_idx, 1],
                                                                         ray_origin[tt, ray_idx, 2]),
                                                             make_float3(ray_dir[tt, ray_idx, 0],
                                                                         ray_dir[tt, ray_idx, 1],
                                                                         ray_dir[tt, ray_idx, 2]),
                                                             boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, box)


@cuda.jit()
def calcIntersectionPoints(ray_origin, ray_dir, ray_power, boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz,
                           leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, tri_material, receive_xyz):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            did_intersect, inter, rng = (
                traverseOctreeAndIntersection(make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2]),
                                              make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2]),
                                              boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz,
                                              leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, 0))
            if did_intersect:
                ray_origin[tt, ray_idx, 0] = inter.x
                ray_origin[tt, ray_idx, 1] = inter.y
                ray_origin[tt, ray_idx, 2] = inter.z
            else:
                ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcClosestIntersection(ray_origin, ray_intersect, ray_dir, ray_bounce, ray_distance, ray_power, ray_bounce_power,
                            boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, leaf_list, leaf_key, tri_vert,
                            tri_idx, tri_norm, tri_material, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rec_xyz = make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2])
            did_intersect, nrho, inter, rng, b = (
                traverseOctreeAndReflection(rec_xyz,
                                              make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2]),
                                              boxminx, boxminy, boxminz, boxmaxx, boxmaxy, boxmaxz, ray_power[tt, ray_idx],
                                              leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, tri_material, ray_distance[tt, ray_idx], params[0]))
            if did_intersect:
                if length(rec_xyz - inter) < length(rec_xyz - make_float3(ray_intersect[tt, ray_idx, 0], ray_intersect[tt, ray_idx, 1], ray_intersect[tt, ray_idx, 2])):
                    ray_intersect[tt, ray_idx, 0] = inter.x
                    ray_intersect[tt, ray_idx, 1] = inter.y
                    ray_intersect[tt, ray_idx, 2] = inter.z
                    ray_bounce[tt, ray_idx, 0] = b.x
                    ray_bounce[tt, ray_idx, 1] = b.y
                    ray_bounce[tt, ray_idx, 2] = b.z
                    ray_distance[tt, ray_idx] = rng
                    ray_bounce_power[tt, ray_idx] = nrho


@cuda.jit()
def calcClosestIntersectionWithoutBounce(ray_origin, ray_intersect, ray_dir, ray_power, kd_tree,
                           leaf_list, leaf_key, tri_vert, tri_idx, tri_norm):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rec_xyz = make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2])
            did_intersect, inter, _ = (
                traverseOctreeAndIntersection(rec_xyz,
                                              make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2]), kd_tree,
                                              leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, 0))
            if did_intersect:
                if length(rec_xyz - inter) < length(rec_xyz - make_float3(ray_intersect[tt, ray_idx, 0], ray_intersect[tt, ray_idx, 1], ray_intersect[tt, ray_idx, 2])):
                    ray_intersect[tt, ray_idx, 0] = inter.x
                    ray_intersect[tt, ray_idx, 1] = inter.y
                    ray_intersect[tt, ray_idx, 2] = inter.z
                    ray_power[tt, ray_idx] = 1


            
            
@cuda.jit(device=True)
def seperatingAxisTest(axis, extent, tut0, tut1, tut2):
    r = extent.x * abs(dot(make_float3(1, 0, 0), axis)) + extent.y * abs(
        dot(make_float3(0, 1, 0), axis)) + extent.z * abs(dot(make_float3(0, 0, 1), axis))
    return (
        max(
            -max(dot(tut0, axis), dot(tut1, axis), dot(tut2, axis)),
            min(dot(tut0, axis), dot(tut1, axis), dot(tut2, axis)),
        )
        <= r
    )
    


@cuda.jit()
def assignBoxPoints(ptx0, ptx1, ptx2, pty0, pty1, pty2, ptz0, ptz1, ptz2, octree_center, octree_extent, point_idx):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for pt in prange(t, ptx0.shape[0], tt_stride):
        for box in prange(r, octree_center.shape[0], ray_stride):
            # Bounding Box test
            c = make_float3(octree_center[box, 0], octree_center[box, 1], octree_center[box, 2])
            extent = make_float3(octree_extent[box, 0], octree_extent[box, 1], octree_extent[box, 2])
            tut0 = make_float3(ptx0[pt], pty0[pt], ptz0[pt]) - c
            tut1 = make_float3(ptx1[pt], pty1[pt], ptz1[pt]) - c
            tut2 = make_float3(ptx2[pt], pty2[pt], ptz2[pt]) - c
            if ((-extent.x <= max(tut0.x, tut1.x, tut2.x) and extent.x >= min(tut0.x, tut1.x, tut2.x)) and
                (-extent.y <= max(tut0.y, tut1.y, tut2.y) and extent.y >= min(tut0.y, tut1.y, tut2.y)) and
                (-extent.z <= max(tut0.z, tut1.z, tut2.z) and extent.z >= min(tut0.z, tut1.z, tut2.z))):

                if seperatingAxisTest(cross(make_float3(1, 0, 0), tut1 - tut0), extent, tut0, tut1, tut2):
                    if seperatingAxisTest(cross(make_float3(0, 1, 0), tut1 - tut0), extent, tut0, tut1, tut2):
                        if seperatingAxisTest(cross(make_float3(0, 0, 1), tut1 - tut0), extent, tut0, tut1, tut2):
                            if seperatingAxisTest(cross(make_float3(1, 0, 0), tut2 - tut1), extent, tut0, tut1, tut2):
                                if seperatingAxisTest(cross(make_float3(0, 1, 0), tut2 - tut1), extent, tut0, tut1, tut2):
                                    if seperatingAxisTest(cross(make_float3(0, 0, 1), tut2 - tut1), extent, tut0, tut1, tut2):
                                        if seperatingAxisTest(cross(make_float3(1, 0, 0), tut0 - tut2), extent, tut0, tut1, tut2):
                                            if seperatingAxisTest(cross(make_float3(0, 1, 0), tut0 - tut2), extent, tut0, tut1, tut2):
                                                if seperatingAxisTest(cross(make_float3(0, 0, 1), tut0 - tut2), extent,
                                                                      tut0, tut1, tut2):
                                                    if seperatingAxisTest(cross(tut1 - tut0, tut2 - tut1), extent,
                                                                          tut0, tut1, tut2):
                                                        point_idx[pt, box] = True
    

