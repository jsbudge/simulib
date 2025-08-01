import cmath
import math
from .cuda_functions import make_float3, cross, dot, length, normalize, make_uint3, rotate
from numba import cuda, prange
import numpy as np
from .cuda_kernels import applyOneWayRadiationPattern

c0_inv = np.float32(1. / 299792458.0)
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
    if abs(det) < np.float32(1e-9):
        return False, None, None

    inv_det = np.float32(1.) / det

    u = inv_det * dot(ro - v0, rcrosse)
    if u < np.float32(0.) or u > np.float32(1.):
        return False, None, None

    # Recompute cross for s and edge 1
    rcrosse = cross(ro - v0, v1 - v0)
    det = inv_det * dot(rd, rcrosse)
    if det < np.float32(0.) or u + det > np.float32(1.):
        return False, None, None

    # Compute intersection point
    det = inv_det * dot(v2 - v0, rcrosse)
    if det < np.float32(1e-9):
        return False, None, None

    if not get_bounce:
        return True, None, None

    return True, normalize(det * rd - vn * dot(det * rd, vn) * np.float32(2.)), ro + det * rd


@cuda.jit(device=True, fast_math=True)
def calcReturnAndBin(inter, re, rng, near_range_s, source_fs, n_samples,
                     pan, tilt, bw_az, bw_el, wavenumber, rho):
    t = inter - re
    r_rng = length(t)
    rng_bin = int(((rng + r_rng) * c0_inv - np.float32(2.) * near_range_s) * source_fs)

    if n_samples > rng_bin > 0:
        # acc_val = (applyOneWayRadiationPattern(r_el, r_az, pan, tilt, bw_az, bw_el) /
        #            (r_rng * r_rng) * rho * cmath.exp(-1j * wavenumber * (rng + r_rng)))
        return (applyOneWayRadiationPattern(-math.asin(t.z / rng), math.atan2(t.x, t.y), pan, tilt, bw_az, bw_el) /
                   (r_rng * r_rng) * rho * cmath.exp(-1j * wavenumber * (rng + r_rng))), rng_bin

    return 0, -1


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
    ray = np.float32(1.) / rd
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
    return tmin <= tmaxy and tminy <= tmax


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
def traverseOctreeAndIntersection(ro, rd, kd_tree, leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, rng):
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


'''@cuda.jit('void(float32[:], float32[:], float32[:, :, :], float32, int32[:], int32[:, :], float32[:, :, :], float32[:, :], float32[:, :], float32, float32, int32, float32, boolean, float32[:], float32, float32[:])', device=True, fast_math=True)
def recurseReflection(ro, rd, kd_tree, rho, leaf_list, leaf_key, tri_vert, tri_norm, tri_material,
                        rng, wavenumber, idx, int_rng, did_intersect, inter, nrho, b):
    if testIntersection0(ro, rd, kd_tree[idx]):
        if math.log2(idx + 1) >= math.log2(kd_tree.shape[0] + 1) - 1:
            tri_min = leaf_key[idx, 0]
            for t_idx in prange(tri_min, tri_min + leaf_key[idx, 1]):

                ti = leaf_list[t_idx]
                curr_intersect = tri_vert[ti, 0, 0] > 7
                if curr_intersect:
                    int_rng = np.float32(7.)
                    nrho = np.float32(14.)
                    b = ro
                    inter = rd
                    did_intersect = True
        else:
            recurseReflection(ro, rd, kd_tree, rho, leaf_list, leaf_key, tri_vert, tri_norm, tri_material,
                            rng, wavenumber, (idx << 1) + 1, int_rng, did_intersect, inter, nrho, b)
            recurseReflection(ro, rd, kd_tree, rho, leaf_list, leaf_key, tri_vert, tri_norm, tri_material,
                              rng, wavenumber, (idx << 1) + 2, int_rng, did_intersect, inter, nrho, b)'''

@cuda.jit("int64(int64)", device=True)
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


@cuda.jit(device=True, fast_math=True)
def traverseOctreeAndReflection(ro, rd, kd_tree, rho, leaf_list, leaf_key, tri_vert, tri_norm, tri_material,
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
        return False, None, None, None, None
    idx = 2
    skip = False

    while 0 < idx < kd_tree.shape[0]:
        # idx, tmp_tmin, tmp_tmax = stack[-1]
        if testIntersection(ro, rd, kd_tree[idx]):
            if math.log2(idx + 1) >= math.log2(kd_tree.shape[0] + 1) - 1:
                tri_min = leaf_key[idx, 0]
                for t_idx in prange(tri_min, tri_min + leaf_key[idx, 1]):
                    ti = leaf_list[t_idx]
                    tn = make_float3(tri_norm[ti, 0], tri_norm[ti, 1], tri_norm[ti, 2])
                    curr_intersect, tb, tinter = (
                        calcSingleIntersection(rd, ro, make_float3(tri_vert[ti, 0, 0],
                                                                   tri_vert[ti, 0, 1],
                                                                   tri_vert[ti, 0, 2]),
                                               make_float3(tri_vert[ti, 1, 0],
                                                           tri_vert[ti, 1, 1],
                                                           tri_vert[ti, 1, 2]),
                                               make_float3(tri_vert[ti, 2, 0],
                                                           tri_vert[ti, 2, 1],
                                                           tri_vert[ti, 2, 2]),
                                               tn, True))
                    if curr_intersect:
                        tmp_rng = length(ro - tinter)
                        if np.float32(1.) < tmp_rng < int_rng:
                            int_rng = tmp_rng
                            inv_rng = np.float32(1.) / (tmp_rng + rng)
                            b = tb + np.float32(0.)
                            # Some parts of the Fresnel coefficient calculation
                            cosa = abs(dot(rd, tn))
                            sina = tri_material[ti, 0] * math.sqrt(
                                max(np.float32(0.), np.float32(1.) - (np.float32(1.) / tri_material[ti, 0] * length(cross(rd, tn))) ** 2))
                            # Rs = abs((cosa - sina) / (
                            #         cosa + sina)) ** 2  # Reflectance using Fresnel coefficient
                            roughness = math.exp(-np.float32(.5) * (np.float32(2.) * wavenumber * tri_material[ti, 1] * cosa) ** 2)  # Roughness calculations to get specular/scattering split
                            # spec = math.exp(-(np.float32(1.) - cosa) ** 2 / np.float32(.0000007442))  # This should drop the specular component to zero by 2 degrees
                            Lsq = ((np.float32(.7) * ((np.float32(1.) + abs(dot(b, rd))) * np.float32(.5)) + np.float32(.3)) *
                                   (np.float32(.7) * ((np.float32(1.) + abs(dot(b, rd))) * np.float32(.5)) + np.float32(.3)))
                            # nrho = rho * inv_rng * inv_rng * cosa * Rs * (
                            #         roughness * spec + (
                            #         np.float32(1.) - roughness) * Lsq)  # Final reflected power
                            nrho = rho * inv_rng * inv_rng * cosa * (abs((cosa - sina) / (
                                    cosa + sina)) ** 2) * (
                                    roughness * math.exp(-(np.float32(1.) - cosa) ** 2 / np.float32(.0000007442)) + (
                                    np.float32(1.) - roughness) * Lsq)  # Final reflected power
                            inter = tinter + np.float32(0.)
                            did_intersect = True
            else:
                # Move down into the box
                idx = idx * 2 + 2
                skip = True
        if not skip:
            while idx % 2 == 1 and idx != 1:
                idx = idx >> 1
            if idx == 1:
                break
            idx -= 1
        skip = False
    return did_intersect, nrho, inter, int_rng + rng, b


@cuda.jit(max_registers=MAX_REGISTERS)
def calcBounceLoop(ray_origin, ray_dir, ray_distance, ray_power, kd_tree,
                   leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, tri_material, pd_r, pd_i, counts, receive_xyz, pan,
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
                ray_power[tt, ray_idx], leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, tri_material, rng, params[0])
            if did_intersect:
                rng += int_rng
                # Check for occlusion against the receiver
                if not traverseOctreeForOcclusion(inter, normalize(rec_xyz - inter), kd_tree, leaf_list, leaf_key,
                                                                           tri_idx, tri_vert, tri_norm):
                    acc, but = calcReturnAndBin(inter, rec_xyz, rng, params[1], params[2], pd_r.shape[1],
                                                               pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
                    if but >= 0:
                        cuda.atomic.add(pd_r, (tt, but), acc.real if abs(acc) < np.inf else np.float32(0.))
                        cuda.atomic.add(pd_i, (tt, but), acc.imag if abs(acc) < np.inf else np.float32(0.))
                        cuda.atomic.add(counts, (tt, but), 1)

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


@cuda.jit('void(float32[:, :, :], float32[:, :, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :, :], int32[:], '
          'int32[:, :], float32[:, :, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], int32[:, :], '
          'float32[:, :], float32[:], float32[:], float32[:], float32[:, :])', max_registers=MAX_REGISTERS)
def calcBounceInit(ray_origin, ray_dir, ray_distance, ray_power, sample_points, kd_tree, leaf_list, leaf_key, tri_vert, tri_norm,
                   tri_material, pd_r, pd_i, counts, receive_xyz, pan, tilt, params, conical_sampling):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rd = normalize(
                make_float3(sample_points[ray_idx, 0], sample_points[ray_idx, 1], sample_points[ray_idx, 2]) - rec_xyz)
            rho = (params[5] *
             applyOneWayRadiationPattern(pan[tt], tilt[tt],
                                         math.atan2(rd.x, rd.y), -math.asin(rd.z),
                                         params[3], params[4]))
            # rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
            did_intersect, nrho, inter, rng, b = traverseOctreeAndReflection(rec_xyz, rd,
                                                                             kd_tree, rho,
                                                                             leaf_list, leaf_key,
                                                                             tri_vert, tri_norm, tri_material, 0, params[0])
            if did_intersect:
                acc, but = calcReturnAndBin(inter, rec_xyz, rng, params[1], params[2], pd_r.shape[1],
                                                           pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
                if pd_r.shape[1] > but >= 0:
                    cuda.atomic.add(pd_r, (tt, but), acc.real if abs(acc) < np.inf else np.float32(0.))
                    cuda.atomic.add(pd_i, (tt, but), acc.imag if abs(acc) < np.inf else np.float32(0.))
                    ray_origin[tt, ray_idx, 0] = inter.x
                    ray_origin[tt, ray_idx, 1] = inter.y
                    ray_origin[tt, ray_idx, 2] = inter.z
                    ray_dir[tt, ray_idx, 0] = b.x
                    ray_dir[tt, ray_idx, 1] = b.y
                    ray_dir[tt, ray_idx, 2] = b.z
                    ray_power[tt, ray_idx] = nrho
                    ray_distance[tt, ray_idx] = rng
                    cuda.atomic.add(counts, (tt, but), 1)
                # Apply supersampling if wanted
                if params[6]:
                    for n in prange(conical_sampling.shape[0]):
                        if abs(rd.y) > np.float32(1e-9):
                            sc = normalize(make_float3(np.float32(2.) * rd.y * rd.z, np.float32(0.), -2 * rd.x * rd.y))
                        else:
                            sc = normalize(make_float3(-2 * rd.y * rd.z, 2 * rd.x * rd.z, 0.))
                        rd = normalize(inter + rotate(rd, sc, conical_sampling[n, 0]) * conical_sampling[n, 1] - rec_xyz)
                        did_intersect, nrho, cone_inter, rng, b = traverseOctreeAndReflection(rec_xyz, rd, kd_tree,
                                                                                         ray_power[tt, ray_idx],
                                                                                         leaf_list, leaf_key,
                                                                                         tri_vert, tri_norm,
                                                                                         tri_material, np.float32(0.), params[0])
                        if did_intersect:
                            acc, but = calcReturnAndBin(cone_inter, rec_xyz, rng, params[1], params[2],
                                                                       pd_r.shape[1],
                                                                       pan[tt], tilt[tt], params[3], params[4],
                                                                       params[0], nrho)
                            if pd_r.shape[1] > but >= 0:
                                cuda.atomic.add(pd_r, (tt, but), acc.real if abs(acc) < np.inf else np.float32(0.))
                                cuda.atomic.add(pd_i, (tt, but), acc.imag if abs(acc) < np.inf else np.float32(0.))
                                cuda.atomic.add(counts, (tt, but), 1)
            else:
                ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcRayBinVariance(ray_origin, ray_dir, ray_distance, ray_power, kd_tree,
                   leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, tri_material, pd_r, pd_i, counts, receive_xyz, pan,
                   tilt, params, conical_sampling):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
            did_intersect, nrho, inter, rng, b, inter_tri = traverseOctreeAndReflection(rec_xyz, rd,
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
                    cuda.atomic.add(counts, (tt, but), 1)
                # Apply supersampling if wanted
                if params[6]:
                    for n in prange(conical_sampling.shape[0]):
                        if abs(rd.y) > 1e-9:
                            sc = normalize(make_float3(2 * rd.y * rd.z, 0., -2 * rd.x * rd.y))
                        else:
                            sc = normalize(make_float3(-2 * rd.y * rd.z, 2 * rd.x * rd.z, 0.))
                        rd = normalize(inter + rotate(rd, sc, conical_sampling[n, 0]) * conical_sampling[n, 1] - rec_xyz)
                        did_intersect, nrho, cone_inter, rng, b, inter_tri = traverseOctreeAndReflection(rec_xyz, rd, kd_tree,
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
                                cuda.atomic.add(counts, (tt, but), 1)
            else:
                ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcReturnPower(ray_origin, ray_distance, ray_power, pd_r, pd_i, counts, receive_xyz, pan,
                   tilt, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_origin.shape[0], tt_stride):
        for ray_idx in prange(r, ray_origin.shape[1], ray_stride):
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
                cuda.atomic.add(counts, (tt, but), 1)




@cuda.jit(max_registers=MAX_REGISTERS)
def calcOriginDirAtt(rec_xyz, sample_points, pan, tilt, params, ray_dir, ray_origin, ray_power):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            # rd = normalize(sample_points[ray_idx] - rec_xyz[tt])
            rd = normalize(make_float3(sample_points[ray_idx, 0], sample_points[ray_idx, 1], sample_points[ray_idx, 2]) -
                  make_float3(rec_xyz[tt, 0], rec_xyz[tt, 1], rec_xyz[tt, 2]))
            ray_power[tt, ray_idx] = (params[5] *
                                      applyOneWayRadiationPattern(pan[tt], tilt[tt],
                                                                  math.atan2(rd.x, rd.y), -math.asin(rd.z),
                                                                  params[3], params[4]))
            ray_dir[tt, ray_idx, 0] = rd.x
            ray_dir[tt, ray_idx, 1] = rd.y
            ray_dir[tt, ray_idx, 2] = rd.z
            # ray_dir[tt, ray_idx] = rd
            # ray_origin[tt, ray_idx] = rec_xyz[tt]
            ray_origin[tt, ray_idx, 0] = rec_xyz[tt, 0]
            ray_origin[tt, ray_idx, 1] = rec_xyz[tt, 1]
            ray_origin[tt, ray_idx, 2] = rec_xyz[tt, 2]


@cuda.jit()
def determineSceneRayIntersections(ray_origin, ray_intersect, ray_dir, ray_distance, ray_bounce, ray_power, ray_bounce_power, kd_tree,
                                   leaf_list, leaf_key, tri_vert, tri_idx, tri_norm, tri_material, params):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
            did_intersect, nrho, inter, rng, b, inter_tri = traverseOctreeAndReflection(make_float3(ray_origin[tt, ray_idx, 0],
                                                                                                    ray_origin[tt, ray_idx, 1],
                                                                                                    ray_origin[tt, ray_idx, 2]), rd,
                                                                                        kd_tree, ray_power[tt, ray_idx],
                                                                                        leaf_list, leaf_key, tri_idx,
                                                                                        tri_vert, tri_norm,
                                                                                        tri_material, 0., params[0])
            if did_intersect and rng < ray_distance[tt, ray_idx]:
                ray_intersect[tt, ray_idx, 0] = inter.x
                ray_intersect[tt, ray_idx, 1] = inter.y
                ray_intersect[tt, ray_idx, 2] = inter.z
                ray_bounce[tt, ray_idx, 0] = b.x
                ray_bounce[tt, ray_idx, 1] = b.y
                ray_bounce[tt, ray_idx, 2] = b.z
                ray_bounce_power[tt, ray_idx] = nrho
                ray_distance[tt, ray_idx] = rng


@cuda.jit()
def calcIntersectionPoints(receive_xyz, ray_intersect, ray_dir, kd_tree, leaf_list, leaf_key, tri_vert,
                           tri_idx, tri_norm):
    t, r = cuda.grid(ndim=2)
    tt_stride, ray_stride = cuda.gridsize(2)
    for tt in prange(t, ray_dir.shape[0], tt_stride):
        for ray_idx in prange(r, ray_dir.shape[1], ray_stride):
            rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
            rxyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
            did_intersect, inter, rng = (
                traverseOctreeAndIntersection(rxyz, rd, kd_tree,
                                              leaf_list, leaf_key, tri_idx, tri_vert, tri_norm, 0))
            # inter = make_float3(0., 1., 0.)
            if did_intersect:
                ray_intersect[tt, ray_idx, 0] = inter.x
                ray_intersect[tt, ray_idx, 1] = inter.y
                ray_intersect[tt, ray_idx, 2] = inter.z


            
            
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
    

