import cmath
import math
from .cuda_functions import make_float3, cross, dot, length, normalize, make_uint3
from numba import cuda, prange
import numpy as np
from .cuda_kernels import applyOneWayRadiationPattern

c0 = 299792458.0


@cuda.jit(device=True, fast_math=True)
def getRangeAndAngles(v, s):
    t = v - s
    rng = length(t)
    az = math.atan2(t.x, t.y)
    el = -math.asin(t.z / rng)
    return t, rng, az, el


@cuda.jit(device=True, fast_math=True)
def checkBox(ro, ray, boxmin, boxmax):
    tmax = min(min(min(np.inf, max((boxmin.x - ro.x) * ray.x, (boxmax.x - ro.x) * ray.x)),
                   max((boxmin.y - ro.y) * ray.y, (boxmax.y - ro.y) * ray.y)),
               max((boxmin.z - ro.z) * ray.z, (boxmax.z - ro.z) * ray.z))
    return (tmax - max(max(max(-np.inf, min((boxmin.x - ro.x) * ray.x, (boxmax.x - ro.x) * ray.x)),
                          min((boxmin.y - ro.y) * ray.y, (boxmax.y - ro.y) * ray.y)),
                      min((boxmin.z - ro.z) * ray.z, (boxmax.z - ro.z) * ray.z)) >= 0) and (tmax >= 0)


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

    rng_bin = ((rng + r_rng) / c0 - 2 * near_range_s) * source_fs

    if n_samples > int(rng_bin) > 0:
        acc_val = applyOneWayRadiationPattern(r_el, r_az, pan, tilt, bw_az, bw_el) / (r_rng * r_rng) * rho * cmath.exp(-1j * wavenumber * (rng + r_rng))
        return acc_val.real, acc_val.imag, int(rng_bin)

    return 0, 0, -1


@cuda.jit(device=True, fast_math=True)
def findOctreeBox(ro, rd, bounding_box, box):
    boxmin = make_float3(bounding_box[box, 0, 0], bounding_box[box, 0, 1], bounding_box[box, 0, 2])
    boxmax = make_float3(bounding_box[box, 1, 0], bounding_box[box, 1, 1], bounding_box[box, 1, 2])
    if boxmin.x == boxmax.x:
        return False
    return bool(checkBox(ro, 1. / rd, boxmin, boxmax))


@cuda.jit(device=True, fast_math=True)
def traverseOctreeForOcclusion(ro, rd, bounding_box, tri_box_idx, tri_box_key, tri_idx, tri_vert, tri_norm):
    """
    Traverse an octree structure, given some triangle indexes, and return the reflected power, bounce angle, and intersection point.
    params:
    ro: float3 = ray origin point
    rd: float3 = normalized ray direction vector
    bounding_box: (N, 2, 3) = array of axis aligned bounding boxes
    tri_box_idx: (N) = sorted array of triangle indexes based on octree boxes
    tri_box_key: (N, 2) = key to look into tri_box_idx and find the triangles inside of a box
    tri_idx: (N, 3) = indexes of vertices that correspond to an individual triangle
    tri_vert: (N, 3) = vertices for triangles in euclidean space
    tri_norm: (N, 3) = surface normal of triangle
    """
    if not findOctreeBox(ro, rd, bounding_box, 0):
        return False
    box = 1
    jump = False
    while 0 < box < bounding_box.shape[0]:
        if findOctreeBox(ro, rd, bounding_box, box):
            if box >= bounding_box.shape[0] >> 3:
                tri_min = tri_box_key[box, 0]
                for t_idx in prange(tri_min, tri_min + tri_box_key[box, 1]):
                    ti = tri_box_idx[t_idx]
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
                box <<= 3
                jump = True
        elif box == 8:
            break
        box += 1
        if (box - 1) >> 3 != (box - 2) >> 3 and not jump:
            if box == 9:
                break
            box >>= 8
            jump = False

    return False


@cuda.jit(device=True, fast_math=True)
def traverseOctreeAndIntersection(ro, rd, bounding_box, rho, tri_box_idx, tri_box_key, tri_idx,
                                  tri_vert, tri_norm, tri_material, rng):
    """
    Traverse an octree structure, given some triangle indexes, and return the reflected power, bounce angle, and intersection point.
    params:
    ro: float3 = ray origin point
    rd: float3 = normalized ray direction vector
    bounding_box: (N, 2, 3) = array of axis aligned bounding boxes
    rho: float = ray power in watts
    final_level: int = start index of the final level of the octree for the bounding_box array
    tri_box_idx: (N) = sorted array of triangle indexes based on octree boxes
    tri_box_key: (N, 2) = key to look into tri_box_idx and find the triangles inside of a box
    tri_idx: (N, 3) = indexes of vertices that correspond to an individual triangle
    tri_vert: (N, 3) = vertices for triangles in euclidean space
    tri_norm: (N, 3) = surface normal of triangle
    tri_material: (N, 3) = material scattering values of triangle - (RCS, ks, kd)
    occlusion_only: bool = set to True to return when the ray intersects something without checking any other triangles
    """
    int_rng = np.inf
    did_intersect = False
    if not findOctreeBox(ro, rd, bounding_box, 0):
        return False, None, None, None, None
    box = 1
    jump = False
    while 0 < box < bounding_box.shape[0]:
        if findOctreeBox(ro, rd, bounding_box, box):
            tn = make_float3(bounding_box[box, 0, 0], bounding_box[box, 0, 1], bounding_box[box, 0, 2])
            if length(ro - tn) <= int_rng:
                if box >= bounding_box.shape[0] >> 3:
                    tri_min = tri_box_key[box, 0]
                    for t_idx in prange(tri_min, tri_min + tri_box_key[box, 1]):
                        ti = tri_box_idx[t_idx]
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
                                inv_rng = 1 / int_rng
                                b = tb + 0.
                                # This is the phong reflection model to get nrho
                                nrho = (tri_material[ti, 1] * max(0, dot(ro - tinter, tn) * inv_rng) * rho +
                                                tri_material[ti, 2] * max(0, 1 - tri_material[ti, 0] * (1 - dot(b, ro) * inv_rng)) ** 2 * rho) * (inv_rng * inv_rng)
                                inter = tinter + 0.
                                did_intersect = True
                else:
                    box <<= 3
                    jump = True
        else:
            if box == 8:
                break
        box += 1
        if (box - 1) >> 3 != (box - 2) >> 3 and not jump:
            if box == 9:
                break
            box >>= 3
            jump = False

    return did_intersect, nrho, inter, int_rng, b

@cuda.jit()
def calcBounceLoop(ray_origin, ray_dir, ray_distance, ray_power, bounding_box, tri_box_idx, tri_box_key, tri_vert, tri_idx,
                   tri_norm, tri_material, pd_r, pd_i, receive_xyz, pan, tilt, params):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:
        # Load in all the parameters that don't change
        ro = make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2])
        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
        rho = ray_power[tt, ray_idx]
        if rho < 1e-9:
            return
        rng = ray_distance[tt, ray_idx]
        did_intersect, nrho, inter, int_rng, b = traverseOctreeAndIntersection(ro, rd, bounding_box, rho, tri_box_idx,
                                                                               tri_box_key, tri_idx, tri_vert,
                                                                               tri_norm, tri_material, rng)
        if did_intersect:
            rng += int_rng
            # Check for occlusion against the receiver
            if not traverseOctreeForOcclusion(inter, normalize(inter - rec_xyz), bounding_box, tri_box_idx, tri_box_key,
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
            ray_distance[tt, ray_idx] = rng
        else:
            ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcBounceInit(ray_origin, ray_dir, ray_distance, ray_power, bounding_box, tri_box_idx, tri_box_key, tri_vert, tri_idx,
                   tri_norm, tri_material, pd_r, pd_i, receive_xyz, pan, tilt, params):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:

        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
        rho = ray_power[tt, ray_idx]
        if rho < 1e-9:
            return
        did_intersect, nrho, inter, rng, b = traverseOctreeAndIntersection(rec_xyz, rd, bounding_box, rho, tri_box_idx,
                                                                               tri_box_key, tri_idx, tri_vert,
                                                                               tri_norm, tri_material, 0)
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
        else:
            ray_power[tt, ray_idx] = 0.


@cuda.jit()
def calcBounceWithoutReflect(ray_dir, ray_power, bounding_box, tri_box_idx, tri_box_key, tri_vert, tri_idx,
                   tri_norm, tri_material, pd_r, pd_i, receive_xyz, pan, tilt, params):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:

        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        rd = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
        rho = ray_power[tt, ray_idx]
        if rho < 1e-9:
            return
        did_intersect, nrho, inter, rng, b = traverseOctreeAndIntersection(rec_xyz, rd, bounding_box, rho, tri_box_idx,
                                                                           tri_box_key, tri_idx, tri_vert,
                                                                           tri_norm, tri_material, 0)
        if did_intersect:
            acc_real, acc_imag, but = calcReturnAndBin(inter, rec_xyz, rng, params[1], params[2], pd_r.shape[1],
                                                       pan[tt], tilt[tt], params[3], params[4], params[0], nrho)
            if but >= 0:
                acc_real = acc_real if abs(acc_real) < np.inf else 0.
                acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
                cuda.atomic.add(pd_r, (tt, but), acc_real)
                cuda.atomic.add(pd_i, (tt, but), acc_imag)


@cuda.jit()
def calcOriginDirAtt(rec_xyz, sample_points, pan, tilt, params, ray_dir, ray_origin, ray_power):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:
        rd = normalize(make_float3(sample_points[ray_idx, 0], sample_points[ray_idx, 1], sample_points[ray_idx, 2]) -
              make_float3(rec_xyz[tt, 0], rec_xyz[tt, 1], rec_xyz[tt, 2]))
        az = math.atan2(rd.x, rd.y)
        el = -math.asin(rd.z)
        ray_power[tt, ray_idx] = params[5] * applyOneWayRadiationPattern(pan[tt], tilt[tt], az, el, params[3], params[4])
        ray_dir[tt, ray_idx, 0] = rd.x
        ray_dir[tt, ray_idx, 1] = rd.y
        ray_dir[tt, ray_idx, 2] = rd.z
        ray_origin[tt, ray_idx, 0] = rec_xyz[tt, 0]
        ray_origin[tt, ray_idx, 1] = rec_xyz[tt, 1]
        ray_origin[tt, ray_idx, 2] = rec_xyz[tt, 2]


@cuda.jit()
def calcIntersectionPoints(ray_origin, ray_dir, ray_power, bounding_box, tri_box_idx, tri_box_key, tri_vert, tri_idx,
                   tri_norm, tri_material, receive_xyz):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:

        did_intersect, nrho, inter, rng, b = traverseOctreeAndIntersection(make_float3(receive_xyz[tt, 0],
                                                                                       receive_xyz[tt, 1],
                                                                                       receive_xyz[tt, 2]),
                                                                           make_float3(ray_dir[tt, ray_idx, 0],
                                                                                       ray_dir[tt, ray_idx, 1],
                                                                                       ray_dir[tt, ray_idx, 2]),
                                                                           bounding_box, ray_power[tt, ray_idx],
                                                                           tri_box_idx, tri_box_key, tri_idx, tri_vert,
                                                                            tri_norm, tri_material, 0)
        if did_intersect:
            ray_origin[tt, ray_idx, 0] = inter.x
            ray_origin[tt, ray_idx, 1] = inter.y
            ray_origin[tt, ray_idx, 2] = inter.z
        else:
            ray_power[tt, ray_idx] = 0.
