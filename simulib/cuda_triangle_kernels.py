import cmath
import math
from .cuda_functions import make_float3, cross, dot, length, normalize, make_uint3, rotate, azelToVec
from numba import cuda, prange
import numpy as np
from .cuda_kernels import applyOneWayRadiationPattern, getRangeAndAngles
from .cuda_mesh_kernels import traverseOctreeAndIntersection, traverseOctreeAndReflection
from numba.cuda.random import xoroshiro128p_uniform_float32

c0 = np.float32(299792458.0)
MAX_REGISTERS = 128

@cuda.jit(device=True, fast_math=True)
def calcTriangleSurfaceArea(ro, p0, p1, p2, Rm, Rp):
    pcycle = cuda.local.array((7, 4), dtype=np.float32)
    pnum = np.int32(0)

    # This is for l0
    lmm, lmp = lambdaFromRange(ro, p2, p1, Rm)
    lpm, lpp = lambdaFromRange(ro, p2, p1, Rp)
    # print(lmm, lmp, lpm, lpp)
    if 0 < lmp < 1:
        pcycle[pnum, 2] = lmp
        pcycle[pnum, 1] = np.float32(1.) - lmp
        pnum += 1
    if 0 < lpp < 1:
        pcycle[pnum, 2] = lpp
        pcycle[pnum, 1] = np.float32(1.) - lpp
        pnum += 1
    if 0 < lpm < 1:
        pcycle[pnum, 2] = lpm
        pcycle[pnum, 1] = np.float32(1.) - lpm
        pnum += 1
    if 0 < lmm < 1:
        pcycle[pnum, 2] = lmm
        pcycle[pnum, 1] = np.float32(1.) - lmm
        pnum += 1
    if (1 > lmm > 0 > lpm) or (1 > lmp > 0 > lpp):
        pcycle[pnum, 0] = np.float32(0.)
        pcycle[pnum, 1] = np.float32(1.)
        pcycle[pnum, 2] = np.float32(0.)
        pnum += 1
    elif (1 > lpm > 0 > lmm) or (1 > lpp > 0 > lmp):
        pcycle[pnum, 0] = np.float32(0.)
        pcycle[pnum, 1] = np.float32(1.)
        pcycle[pnum, 2] = np.float32(0.)
        pnum += 1

    # Repeat with side l2
    lmm, lmp = lambdaFromRange(ro, p1, p0, Rm)
    lpm, lpp = lambdaFromRange(ro, p1, p0, Rp)
    # print(lmm, lmp, lpm, lpp)
    if 0 < lmp < 1:
        pcycle[pnum, 1] = lmp
        pcycle[pnum, 0] = np.float32(1.) - lmp
        pnum += 1
    if 0 < lpp < 1:
        pcycle[pnum, 1] = lpp
        pcycle[pnum, 0] = np.float32(1.) - lpp
        pnum += 1
    if 0 < lpm < 1:
        pcycle[pnum, 1] = lpm
        pcycle[pnum, 0] = np.float32(1.) - lpm
        pnum += 1
    if 0 < lmm < 1:
        pcycle[pnum, 1] = lmm
        pcycle[pnum, 0] = np.float32(1.) - lmm
        pnum += 1
    if (1 > lmm > 0 > lpm) or (1 > lmp > 0 > lpp):
        pcycle[pnum, 0] = np.float32(1.)
        pcycle[pnum, 1] = np.float32(0.)
        pcycle[pnum, 2] = np.float32(0.)
        pnum += 1
    elif (1 > lpm > 0 > lmm) or (1 > lpp > 0 > lmp):
        pcycle[pnum, 0] = np.float32(1.)
        pcycle[pnum, 1] = np.float32(0.)
        pcycle[pnum, 2] = np.float32(0.)
        pnum += 1

    # Repeat with side l1
    lmm, lmp = lambdaFromRange(ro, p0, p2, Rm)
    lpm, lpp = lambdaFromRange(ro, p0, p2, Rp)
    # print(lmm, lmp, lpm, lpp)
    if 0 < lmp < 1:
        pcycle[pnum, 0] = lmp
        pcycle[pnum, 2] = np.float32(1.) - lmp
        pnum += 1
    if 0 < lpp < 1:
        pcycle[pnum, 0] = lpp
        pcycle[pnum, 2] = np.float32(1.) - lpp
        pnum += 1
    if 0 < lpm < 1:
        pcycle[pnum, 0] = lpm
        pcycle[pnum, 2] = np.float32(1.) - lpm
        pnum += 1
    if 0 < lmm < 1:
        pcycle[pnum, 0] = lmm
        pcycle[pnum, 2] = np.float32(1.) - lmm
        pnum += 1
    if (1 > lmm > 0 > lpm) or (1 > lmp > 0 > lpp):
        pcycle[pnum, 0] = np.float32(0.)
        pcycle[pnum, 1] = np.float32(0.)
        pcycle[pnum, 2] = np.float32(1.)
        pnum += 1
    elif (1 > lpm > 0 > lmm) or (1 > lpp > 0 > lmp):
        pcycle[pnum, 0] = np.float32(0.)
        pcycle[pnum, 1] = np.float32(0.)
        pcycle[pnum, 2] = np.float32(1.)
        pnum += 1

    ref_pt = pcycle[0, 0] * p0 + pcycle[0, 1] * p1 + pcycle[0, 2] * p2
    crossprod = np.float32(.5) * length(cross((pcycle[2, 0] * p0 + pcycle[2, 1] * p1 + pcycle[2, 2] * p2) - ref_pt,
                                             (pcycle[1, 0] * p0 + pcycle[1, 1] * p1 + pcycle[1, 2] * p2) - ref_pt))
    # print(crossprod)
    pcycle[1, 3] = crossprod
    for n in range(2, pnum - 1):
        sa = length(cross((pcycle[n, 0] * p0 + pcycle[n, 1] * p1 + pcycle[n, 2] * p2) - ref_pt,
                                     (pcycle[n + 1, 0] * p0 + pcycle[n + 1, 1] * p1 + pcycle[
                                         n + 1, 2] * p2) - ref_pt)) * np.float32(.5)
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
    a = 2. * p0.x * p1.x + 2. * p0.y * p1.y + 2. * p0.z * p1.z - p0.x ** 2 - p0.y ** 2 - p0.z ** 2 - p1.x ** 2 - \
        p1.y ** 2 - p1.z ** 2
    c = 2. * p1.x * r0.x + 2. * p1.y * r0.y + 2. * p1.z * r0.z - p1.x ** 2 - p1.y ** 2 - p1.z ** 2 - r0.x ** 2 - \
        r0.y ** 2 - r0.z ** 2 + R ** 2
    b = -2. * p0.x * p1.x - 2. * p0.y * p1.y - 2. * p0.z * p1.z + 2. * p0.x * r0.x + 2. * p0.y * r0.y + 2. * p0.z * \
        r0.z - 2. * p1.x * r0.x - 2. * p1.y * r0.y - 2. * p1.z * r0.z + 2. * p1.x ** 2 + 2. * p1.y ** 2 + 2. * p1.z ** 2

    l_tp = (-b + math.sqrt(b ** 2 - np.float32(4.) * a * c)) / (2. * a)
    l_tp = -np.inf if math.isnan(l_tp) else l_tp
    l_tm = (-b - math.sqrt(b ** 2 - np.float32(4.) * a * c)) / (2. * a)
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
        tx = make_float3(transmit_xyz[0], transmit_xyz[1], transmit_xyz[2])
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
    tx = make_float3(transmit_xyz[0], transmit_xyz[1], transmit_xyz[2])
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


@cuda.jit('void(float32[:, :], float32[:, :, :], float32[:, :], float32[:])', max_registers=40)
def calcBinTotalSurfaceArea(transmit_xyz, tri_verts, sa_bins, params):
    tri, pulse = cuda.grid(ndim=2)
    tri_stride, pulse_stride = cuda.gridsize(2)
    for tri_idx in prange(tri, tri_verts.shape[0], tri_stride):
        p0 = make_float3(tri_verts[tri_idx, 0, 0], tri_verts[tri_idx, 0, 1], tri_verts[tri_idx, 0, 2])
        p1 = make_float3(tri_verts[tri_idx, 1, 0], tri_verts[tri_idx, 1, 1], tri_verts[tri_idx, 1, 2])
        p2 = make_float3(tri_verts[tri_idx, 2, 0], tri_verts[tri_idx, 2, 1], tri_verts[tri_idx, 2, 2])
        for t in prange(pulse, transmit_xyz.shape[0], pulse_stride):
            tx = make_float3(transmit_xyz[t, 0], transmit_xyz[t, 1], transmit_xyz[t, 2])
            rmin = min(length(tx - p2), length(tx - p0), length(tx - p1))
            '''rmin = length(tx - p2)
            rmax = length(tx - p0)
            if rmin > rmax:
                rmin = rmax
                rmax = length(tx - p2)
            if length(tx - p1) > rmax:
                rmax = length(tx - p1)
            elif length(tx - p1) < rmin:
                rmin = length(tx - p1)'''
            rmax = max(length(tx - p2), length(tx - p0), length(tx - p1))
            if rmin > params[1] * c0:
                first_bin = int(((np.float32(2.) * rmin) / c0 - np.float32(2.) * params[1]) * params[2])
            if np.float32(0.) <= first_bin < sa_bins.shape[1]:
                nbins = int(((np.float32(2.) * rmax) / c0 - np.float32(2.) * params[1]) * params[2] + np.float32(1.)) - first_bin
                for b in prange(first_bin, first_bin + nbins, 1):
                    if b < sa_bins.shape[1]:
                        R = (b / params[2] + np.float32(2.) * params[1]) * c0 * np.float32(.5)
                        sa, _ = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (np.float32(1.) / params[2]) * c0 * np.float32(.5))
                        cuda.atomic.add(sa_bins, (t, b), sa)



@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleSampleVariance(transmit_xyz, pan, tilt, tri_verts, tri_idxes, tri_norm, tri_material, kd_tree, leaf_key,
                               leaf_list, triangle_ranges, triangle_bin_surface_area, triangle_sample_variance, rands,
                               params):
    tri, rbin = cuda.grid(ndim=2)
    tri_stride, bin_stride = cuda.gridsize(2)
    tx = make_float3(transmit_xyz[0], transmit_xyz[1], transmit_xyz[2])
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
                # if tri_idx == 407:
                R = ((triangle_ranges[tri_idx, 0] + b) / params[2] + 2 * params[1]) * c0 * .5
                sa, pcycle = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (1 / params[2]) * c0 * .5)
                mu = 0j
                m2 = 0j
                n = 0
                delta_mu = 9. + 0j
                while n < 10 and (math.sqrt((delta_mu.real - mu.real)**2 + (delta_mu.imag - mu.imag)**2) > 1e-11 or n < 3):
                    sample_pt = sampleBinArea(pcycle, p0, p1, p2, rands, tri_idx)
                    ray_dir = sample_pt - tx
                    ray_dir = ray_dir / length(ray_dir)
                    did_intersect, nrho, inter, _, _, inter_tri = traverseOctreeAndReflection(tx, ray_dir, kd_tree, params[5],
                                                                                              leaf_list,
                                                                            leaf_key, tri_idxes, tri_verts, tri_norm,
                                                                            tri_material, 0, params[0])
                    delta_mu = mu + 0j
                    if did_intersect and inter_tri == tri_idx:
                        r, r_rng, r_az, r_el = getRangeAndAngles(inter, tx)
                        acc_val = (applyOneWayRadiationPattern(r_el, r_az, pan, tilt, params[3], params[4]) /
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
                    n += 1

                # print(b, mu.real, mu.imag)
                triangle_bin_surface_area[tri_idx, b] = sa
                triangle_sample_variance[tri_idx, b] = math.sqrt(m2.real**2 + m2.imag**2) / (n - 1.)



@cuda.jit(max_registers=MAX_REGISTERS)
def calcTriangleReturnsFromVariance(transmit_xyz, pan, tilt, tri_verts, tri_idxes, tri_norm, tri_material, kd_tree,
                                    leaf_key, leaf_list, triangle_ranges, triangle_return_r, triangle_return_i,
                                    triangle_sample_variance, rands, params):
    tri, rbin = cuda.grid(ndim=2)
    tri_stride, bin_stride = cuda.gridsize(2)
    tx = make_float3(transmit_xyz[0], transmit_xyz[1], transmit_xyz[2])
    for tri_idx in prange(tri, tri_idxes.shape[0], tri_stride):
        if triangle_ranges[tri_idx, 0] == 0:
            continue
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        for b in prange(rbin, triangle_sample_variance.shape[1], bin_stride):
            if b < triangle_ranges[tri_idx, 1]:
                # if tri_idx == 407:
                R = ((triangle_ranges[tri_idx, 0] + b) / params[2] + 2 * params[1]) * c0 * .5
                sa, pcycle = calcTriangleSurfaceArea(tx, p0, p1, p2, R, R + (1 / params[2]) * c0 * .5)
                mu = 0j
                m2 = 0j
                n = 0
                delta_mu = 9. + 0j
                while n < 100 and (math.sqrt((delta_mu.real - mu.real)**2 + (delta_mu.imag - mu.imag)**2) > 1e-11 or n < 3):
                    sample_pt = sampleBinArea(pcycle, p0, p1, p2, rands, tri_idx)
                    ray_dir = sample_pt - tx
                    ray_dir = ray_dir / length(ray_dir)
                    did_intersect, nrho, inter, _, _, inter_tri = traverseOctreeAndReflection(tx, ray_dir, kd_tree, params[5],
                                                                                              leaf_list,
                                                                            leaf_key, tri_idxes, tri_verts, tri_norm,
                                                                            tri_material, 0, params[0])
                    delta_mu = mu + 0j
                    if did_intersect and inter_tri == tri_idx:
                        r, r_rng, r_az, r_el = getRangeAndAngles(inter, tx)
                        acc_val = (applyOneWayRadiationPattern(r_el, r_az, pan, tilt, params[3], params[4]) /
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
                    n += 1

                # print(b, mu.real, mu.imag)
                triangle_sample_variance[tri_idx, b] = math.sqrt(m2.real**2 + m2.imag**2) / (n - 1.)
                triangle_return_r[triangle_ranges[tri_idx, 0] + b] = cuda.atomic.add(triangle_return_r, triangle_ranges[tri_idx, 0] + b, sa * mu.real)
                triangle_return_i[triangle_ranges[tri_idx, 0] + b] = cuda.atomic.add(triangle_return_r, triangle_ranges[tri_idx, 0] + b, sa * mu.imag)


@cuda.jit(max_registers=MAX_REGISTERS)
def calcViewVariance(transmit_xyz, pan, tilt, tri_verts, tri_idxes, tri_norm, tri_material, kd_tree, leaf_key, leaf_list,
                     pixel_mu_r, pixel_mu_i, pixel_m2_r, pixel_m2_i, pixel_count, rands, params):
    a, e = cuda.grid(ndim=2)
    a_stride, e_stride = cuda.gridsize(2)
    tx = make_float3(transmit_xyz[0], transmit_xyz[1], transmit_xyz[2])
    for az_idx in prange(a, pixel_count.shape[0], a_stride):
        if az_idx < pixel_count.shape[0]:
            for el_idx in prange(e, pixel_count.shape[1], e_stride):
                if el_idx < pixel_count.shape[1]:
                    mu = 0j
                    m2 = 0j
                    for n in range(100):
                        az = (az_idx + xoroshiro128p_uniform_float32(rands, az_idx)) * 2 * params[3] / pixel_count.shape[0] + pan - params[3]
                        el = (el_idx + xoroshiro128p_uniform_float32(rands, az_idx)) * 2 * params[4] / pixel_count.shape[1] + tilt - params[4]
                        # az = np.float32(az_idx)
                        # el = np.float32(el_idx)
                        # ray_dir = azelToVec(az, el)
                        ray_dir = make_float3(math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), -math.sin(el))
                        did_intersect, nrho, inter, _, _, _ = traverseOctreeAndReflection(tx, ray_dir, kd_tree, params[5],
                                                                                                  leaf_list,
                                                                                                  leaf_key, tri_idxes, tri_verts,
                                                                                                  tri_norm,
                                                                                                  tri_material, 0., params[0])
                        # print(az, el)
                        if did_intersect:
                            r, r_rng, r_az, r_el = getRangeAndAngles(inter, tx)
                            acc_val = (applyOneWayRadiationPattern(r_el, r_az, pan, tilt, params[3], params[4]) /
                                       (r_rng * r_rng) * nrho * cmath.exp(-1j * params[5] * (2. * r_rng)))
                            # acc_val = 1j
                            # print(acc_val.real, acc_val.imag)
                            delta = acc_val - mu
                            mu += delta / (n + 1)
                            m2 += delta * (acc_val - mu)
                        else:
                            # print('non-intersect')
                            delta = -mu
                            mu += delta / (n + 1)
                            m2 += delta * -mu
                    pixel_count[az_idx, el_idx] += n + 1
                    pixel_mu_i[az_idx, el_idx] += mu.imag
                    pixel_mu_r[az_idx, el_idx] += mu.real
                    pixel_m2_i[az_idx, el_idx] += m2.imag
                    pixel_m2_r[az_idx, el_idx] += m2.real


@cuda.jit(max_registers=MAX_REGISTERS)
def calcViewSamples(transmit_xyz, pan, tilt, tri_verts, tri_idxes, tri_norm, kd_tree, leaf_key, leaf_list,
                    target_samples, ray_points, rands, params):
    a, e = cuda.grid(ndim=2)
    a_stride, e_stride = cuda.gridsize(2)
    tx = make_float3(transmit_xyz[0], transmit_xyz[1], transmit_xyz[2])
    for az_idx in prange(a, target_samples.shape[0], a_stride):
        for el_idx in prange(e, target_samples.shape[1], e_stride):
            if target_samples[az_idx, el_idx, 0] > 0:
                for n in prange(target_samples[az_idx, el_idx, 0]):
                    az = (az_idx + xoroshiro128p_uniform_float32(rands, az_idx)) * 2 * params[0] / target_samples.shape[
                        0] + pan - params[0]
                    el = (el_idx + xoroshiro128p_uniform_float32(rands, az_idx)) * 2 * params[1] / target_samples.shape[
                        1] + tilt - params[1]
                    rd = azelToVec(az, el)
                    did_intersect, inter, rng, _ = traverseOctreeAndIntersection(tx, rd, kd_tree, leaf_list, leaf_key,
                                                                                 tri_idxes, tri_verts, tri_norm, 0)
                    if did_intersect:
                        ray_points[target_samples[az_idx, el_idx, 1] + n, 0] = inter.x
                        ray_points[target_samples[az_idx, el_idx, 1] + n, 1] = inter.y
                        ray_points[target_samples[az_idx, el_idx, 1] + n, 2] = inter.z



@cuda.jit(max_registers=MAX_REGISTERS)
def calcFullTriangleSamples(tri_verts, tri_idxes, triangle_ranges,
                        target_samples, ray_points, rands):
    tri, rbin = cuda.grid(ndim=2)
    tri_stride, bin_stride = cuda.gridsize(2)
    for tri_idx in prange(tri, tri_idxes.shape[0], tri_stride):
        if triangle_ranges[tri_idx, 0] == 0:
            continue
        p0 = make_float3(tri_verts[tri_idxes[tri_idx, 0], 0], tri_verts[tri_idxes[tri_idx, 0], 1],
                         tri_verts[tri_idxes[tri_idx, 0], 2])
        p1 = make_float3(tri_verts[tri_idxes[tri_idx, 1], 0], tri_verts[tri_idxes[tri_idx, 1], 1],
                         tri_verts[tri_idxes[tri_idx, 1], 2])
        p2 = make_float3(tri_verts[tri_idxes[tri_idx, 2], 0], tri_verts[tri_idxes[tri_idx, 2], 1],
                         tri_verts[tri_idxes[tri_idx, 2], 2])
        if target_samples[tri_idx, 0] > 0 and rbin == 0:
            for n in prange(target_samples[tri_idx, 0]):
                r1 = math.sqrt(xoroshiro128p_uniform_float32(rands, tri_idx))
                r2 = xoroshiro128p_uniform_float32(rands, tri_idx)
                sample_pt = (1 - r1) * p0 + r1 * (1 - r2) * p1 + r1 * r2 * p2
                ray_points[target_samples[tri_idx, 1] + n, 0] = sample_pt.x
                ray_points[target_samples[tri_idx, 1] + n, 1] = sample_pt.y
                ray_points[target_samples[tri_idx, 1] + n, 2] = sample_pt.z
                ray_points[target_samples[tri_idx, 1] + n, 3] = tri_idx