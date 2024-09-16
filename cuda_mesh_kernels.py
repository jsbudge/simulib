import cmath
import math
from tqdm import tqdm
from numba import cuda, njit
from numba.cuda.random import xoroshiro128p_uniform_float64
import cupy

from simulation_functions import findPowerOf2
import numpy as np
import open3d as o3d
from cuda_kernels import applyRadiationPattern, applyRadiationPatternCPU, getMaxThreads

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@cuda.jit(device=True)
def getRangeAndAngles(vx, vy, vz, sx, sy, sz):
    tx = vx - sx
    ty = vy - sy
    tz = vz - sz
    rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))
    az = math.atan2(tx, ty)
    el = -math.asin(tz / rng)
    return tx, ty, tz, rng, az, el

'''
Calculate out if each ray intersects with each triangle given.
This is meant to get possible occlusions in the mesh.
Pass in the ray origin and direction, as well as the triangles in the possible bounding box, and let it check for
any intersection that isn't the one expected.
'''
@cuda.jit()
def calcIntersection(ray_bounce, ray_xyz, vert_xyz, tri_norm, tri_verts, int_xyz, int_tri):
    ray_idx, tri_idx = cuda.grid(ndim=2)
    if tri_norm.shape[0] > tri_idx and ray_idx < ray_bounce.shape[0]:
        rx = ray_xyz[ray_idx, 0]
        ry = ray_xyz[ray_idx, 1]
        rz = ray_xyz[ray_idx, 2]
        e1x = vert_xyz[tri_verts[tri_idx, 1], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
        e1y = vert_xyz[tri_verts[tri_idx, 1], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
        e1z = vert_xyz[tri_verts[tri_idx, 1], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
        e2x = vert_xyz[tri_verts[tri_idx, 2], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
        e2y = vert_xyz[tri_verts[tri_idx, 2], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
        e2z = vert_xyz[tri_verts[tri_idx, 2], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
        crossx = ray_bounce[ray_idx, 1] * e2z - ray_bounce[ray_idx, 2] * e2y
        crossy = ray_bounce[ray_idx, 2] * e2x - ray_bounce[ray_idx, 0] * e2z
        crossz = ray_bounce[ray_idx, 0] * e2y - ray_bounce[ray_idx, 1] * e2x
        det = e1x * crossx + e1y * crossy + e1z * crossz
        # Check to see if ray is parallel to triangle
        if abs(det) < 1e-9:
            return

        inv_det = 1. / det
        sx = rx - vert_xyz[tri_verts[tri_idx, 0], 0]
        sy = ry - vert_xyz[tri_verts[tri_idx, 0], 1]
        sz = rz - vert_xyz[tri_verts[tri_idx, 0], 2]

        u = inv_det * (sx * crossx + sy * crossy + sz * crossz)
        if u < 0 or u > 1:
            return

        # Recompute cross for s and edge 1
        crossx = sy * e1z - sz * e1y
        crossy = sz * e1x - sx * e1z
        crossz = sx * e1y - sy * e1x
        v = inv_det * (ray_bounce[ray_idx, 0] * crossx + ray_bounce[ray_idx, 1] * crossy + ray_bounce[ray_idx, 2] * crossz)
        if v < 0 or u + v > 1:
            return

        # Compute intersection point
        t = inv_det * (e2x * crossx + e2y * crossy + e2z * crossz)
        if t < 1e-9:
            return

        vnx = tri_norm[tri_idx, 0]
        vny = tri_norm[tri_idx, 1]
        vnz = tri_norm[tri_idx, 2]

        intx = rx + t * ray_bounce[ray_idx, 0]
        inty = ry + t * ray_bounce[ray_idx, 1]
        intz = rz + t * ray_bounce[ray_idx, 2]

        if math.sqrt(abs(intx - rx)**2 + abs(inty - ry)**2 + abs(intz - rz)**2) < math.sqrt(abs(int_xyz[ray_idx, 0] - rx)**2 + abs(int_xyz[ray_idx, 1] - ry)**2 + abs(int_xyz[ray_idx, 2] - rz)**2):

            # Calculate out the angles in azimuth and elevation for the bounce
            tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)

            bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
            bx = tx - vnx * bounce_dot
            by = ty - vny * bounce_dot
            bz = tz - vnz * bounce_dot
            bounce_len = math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz))
            ray_bounce[ray_idx, 0] = bx / bounce_len
            ray_bounce[ray_idx, 1] = by / bounce_len
            ray_bounce[ray_idx, 2] = bz / bounce_len
            int_xyz[ray_idx, 0] = intx
            int_xyz[ray_idx, 1] = inty
            int_xyz[ray_idx, 2] = intz
            int_tri[ray_idx] = tri_idx
        cuda.syncthreads()


@cuda.jit()
def calcInitSpread(ray_power, ray_distance, ray_bounce, vert_xyz, vert_norm, vert_power, source_xyz, panrx, elrx, pd_r, pd_i,
                   wavenumber, near_range_s, source_fs, bw_az, bw_el):
    ray_idx, vert_idx = cuda.grid(ndim=2)
    if ray_idx < ray_power.shape[0] and vert_idx == 0:
        # Calculate the bounce vector for this time
        vx = vert_xyz[ray_idx, 0]
        vy = vert_xyz[ray_idx, 1]
        vz = vert_xyz[ray_idx, 2]
        vnx = vert_norm[ray_idx, 0]
        vny = vert_norm[ray_idx, 1]
        vnz = vert_norm[ray_idx, 2]

        # Calculate out the angles in azimuth and elevation for the bounce
        sx, sy, sz, srng, r_az, r_el = getRangeAndAngles(vx, vy, vz, source_xyz[0], source_xyz[1], source_xyz[2])

        bounce_dot = (sx * vnx + sy * vny + sz * vnz) * 2.
        bx = sx - vnx * bounce_dot
        by = sy - vny * bounce_dot
        bz = sz - vnz * bounce_dot
        bounce_len = math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz))
        ray_bounce[ray_idx, 0] = bx / bounce_len
        ray_bounce[ray_idx, 1] = by / bounce_len
        ray_bounce[ray_idx, 2] = bz / bounce_len

        rx_strength = (sx * bx + sy * by + sz * bz) / (srng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
        rx_strength = max(rx_strength, 0.)
        ray_power[ray_idx] = ray_power[ray_idx] * math.pow(rx_strength, 5)
        ray_distance[ray_idx] = srng
        accumulateRangeProfile(ray_power[ray_idx] / (ray_distance[ray_idx] * ray_distance[ray_idx]) * vert_power[ray_idx], srng, r_el, r_az, panrx,
                               elrx, pd_r, pd_i, wavenumber, near_range_s, source_fs, bw_az, bw_el)
        cuda.syncthreads()


@cuda.jit(device=True)
def accumulateRangeProfile(power, two_way_rng, r_el, r_az, panrx, elrx, pd_r, pd_i, wavenumber, near_range_s,
                            source_fs, bw_az, bw_el):
    rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
    if but > pd_r.shape[0] or but < 0:
        return
    att = applyRadiationPattern(r_el, r_az, panrx, elrx, panrx, elrx, bw_az, bw_el) / (
            two_way_rng * two_way_rng)
    acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * power
    cuda.atomic.add(pd_r, but, acc_val.real)
    cuda.atomic.add(pd_i, but, acc_val.imag)


@cuda.jit()
def genRangeProfileSinglePulse(int_xyz, vert_bounce, vert_reflectivity,
                               source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                               source_fs, bw_az, bw_el, power_scale, do_beampattern, do_bounce):
    # sourcery no-metrics
    pidx = cuda.grid(ndim=1)
    if pidx < int_xyz.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        # Calculate the bounce vector for this time
        vx = int_xyz[pidx, 0]
        vy = int_xyz[pidx, 1]
        vz = int_xyz[pidx, 2]
        bx = vert_bounce[pidx, 0]
        by = vert_bounce[pidx, 1]
        bz = vert_bounce[pidx, 2]

        # Calculate out the angles in azimuth and elevation for the bounce
        _, _, _, rng, _, _ = getRangeAndAngles(vx, vy, vz, source_xyz[0], source_xyz[1], source_xyz[2])
        rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(vx, vy, vz, receive_xyz[0], receive_xyz[1], receive_xyz[2])

        # Calculate bounce vector and strength
        gamma = 1.
        if do_bounce:
            bounce_ang = math.acos((rx * bx + ry * by + rz * bz) / r_rng)
            gamma = abs(math.sin(vert_reflectivity[pidx] * bounce_ang) / (vert_reflectivity[pidx] * bounce_ang))

        two_way_rng = rng + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if but > pd_r.shape[0] or but < 0:
            return

        if n_samples > but > 0:
            att = 1.
            if do_beampattern:
                att = applyRadiationPattern(r_el, r_az, panrx, elrx, pantx, eltx, bw_az, bw_el) / (
                        two_way_rng * two_way_rng)
            att *= power_scale
            acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * vert_reflectivity[pidx] * gamma
            cuda.atomic.add(pd_r, but, acc_val.real)
            cuda.atomic.add(pd_i, but, acc_val.imag)
        cuda.syncthreads()



@cuda.jit()
def genRangeProfileFromMesh(vert_xyz, vert_norm, vert_reflectivity,
                            source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                            source_fs, bw_az, bw_el, power_scale, do_beampattern, do_bounce):
    # sourcery no-metrics
    pidx, t = cuda.grid(ndim=2)
    if pidx < vert_xyz.shape[0] and t < source_xyz.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        # Calculate the bounce vector for this time
        vx = vert_xyz[pidx, 0]
        vy = vert_xyz[pidx, 1]
        vz = vert_xyz[pidx, 2]
        vnx = vert_norm[pidx, 0]
        vny = vert_norm[pidx, 1]
        vnz = vert_norm[pidx, 2]

        # Calculate out the angles in azimuth and elevation for the bounce
        tx, ty, tz, rng, _, _ = getRangeAndAngles(vx, vy, vz, source_xyz[t, 0], source_xyz[t, 1], source_xyz[t, 2])
        rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(vx, vy, vz, receive_xyz[t, 0], receive_xyz[t, 1], receive_xyz[t, 2])

        # Calculate bounce vector and strength
        gamma = 1.
        if do_bounce:
            bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
            bx = tx - vnx * bounce_dot
            by = ty - vny * bounce_dot
            bz = tz - vnz * bounce_dot

            rx_strength = (rx * bx + ry * by + rz * bz) / (r_rng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
            if rx_strength < 0:
                return
            gamma = math.pow(-rx_strength, 5)

        two_way_rng = rng + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if but > pd_r.shape[0] or but < 0:
            return

        if n_samples > but > 0:
            att = 1.
            if do_beampattern:
                att = applyRadiationPattern(r_el, r_az, panrx[t], elrx[t], pantx[t], eltx[t], bw_az, bw_el) / (
                        two_way_rng * two_way_rng)
            att *= power_scale
            acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * vert_reflectivity[pidx] * gamma
            cuda.atomic.add(pd_r, (but, np.uint16(t)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint16(t)), acc_val.imag)
        cuda.syncthreads()


def genRangeProfileFromMeshCPU(vert_xyz, vert_norm, vert_reflectivity,
                               source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                               source_fs, bw_az, bw_el, power_scale):
    # sourcery no-metrics
    for pidx in range(vert_xyz.shape[0]):
        for t in range(source_xyz.shape[0]):
            # Load in all the parameters that don't change
            n_samples = pd_r.shape[0]
            wavenumber = 2 * np.pi / wavelength

            # Calculate the bounce vector for this time
            vx = vert_xyz[pidx, 0]
            vy = vert_xyz[pidx, 1]
            vz = vert_xyz[pidx, 2]
            vnx = vert_norm[pidx, 0]
            vny = vert_norm[pidx, 1]
            vnz = vert_norm[pidx, 2]

            # Calculate out the angles in azimuth and elevation for the bounce
            tx = vx - source_xyz[t, 0]
            ty = vy - source_xyz[t, 1]
            tz = vz - source_xyz[t, 2]
            rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

            rx = vx - receive_xyz[t, 0]
            ry = vy - receive_xyz[t, 1]
            rz = vz - receive_xyz[t, 2]
            r_rng = math.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
            r_el = -math.asin(rz / r_rng)
            r_az = math.atan2(rx, ry)

            # Calculate bounce vector and strength
            bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
            bx = tx - vnx * bounce_dot
            by = ty - vny * bounce_dot
            bz = tz - vnz * bounce_dot

            rx_strength = (rx * bx + ry * by + rz * bz) / (
                    r_rng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
            if rx_strength < 0:
                continue

            two_way_rng = rng + r_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > pd_r.shape[0] or but < 0:
                continue

            if n_samples > but > 0:
                gamma = math.pow(-rx_strength, 5)
                # att = applyRadiationPatternCPU(r_el, r_az, panrx[t], elrx[t], pantx[t], eltx[t], bw_az, bw_el) / (
                #         two_way_rng * two_way_rng)
                att = 1.
                att *= power_scale
                acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * vert_reflectivity[pidx] * gamma
                pd_r[but, t] += acc_val.real
                pd_i[but, t] += acc_val.imag

    return pd_r, pd_i


def barycentric(x0, x1, x2, a0, a1, a2):
    return x0 * a0 + x1 * a1 + x2 * a2


def cart2bary(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1. - v - w
    return u, v, w


def triNormal(x0, x1, x2):
    """
    Calculate a unit normal vector for a triangle.
    """
    A = x1 - x0
    B = x2 - x0
    nx = A[1] * B[2] - A[2] * B[1]
    ny = A[2] * B[0] - A[0] * B[2]
    nz = A[0] * B[1] - A[1] * B[0]
    det = 1 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * det, ny * det, nz * det


def bounceVector(x0, n0):
    """
    Returns the unit vector bouncing off of a normal vector.
    """
    return x0 - 2 * sum(x0 * n0) / sum(n0 * n0) * n0


def readCombineMeshFile(fnme, points=100000):
    full_mesh = o3d.io.read_triangle_model(fnme)
    mesh = full_mesh.meshes[0].mesh
    for me in full_mesh.meshes[1:]:
        mesh += me.mesh

    if len(mesh.triangles) > points:
        mesh = mesh.simplify_quadric_decimation(points)
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    return mesh

def checkBoxIntersection(ray, ray_origin, box_max, box_min):
    tmin = 0.
    tmax = np.inf
    for d in range(3):
        t1 = (box_min[d] - ray_origin[d]) / ray[d]
        t2 = (box_max[d] - ray_origin[d]) / ray[d]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    return tmin < tmax


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from simulation_functions import db, genChirp, azelToVec

    mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)
    obs_pt = np.array([0., -300., 500.])
    nsam = 256
    nr = 4096
    fc = 32.0e9
    standoff = 700.
    fft_len = findPowerOf2(nsam + nr)
    up_fft_len = fft_len * 4

    # GPU device calculations
    threads_per_block = getMaxThreads()

    # Generate bounding box tree
    full_box = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 20.)
    boxes = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(full_box.get_voxel_bounding_points(t.grid_index))
             for t in full_box.get_voxels()]
    mesh_quads = [mesh.crop(b) for b in boxes]
    quad_triangles = [np.asarray(m.triangles) for m in mesh_quads]
    quad_vertices = [np.asarray(m.vertices) for m in mesh_quads]
    quad_normals = [np.asarray(m.triangle_normals) for m in mesh_quads]

    ant_point = mesh.get_center() - obs_pt

    sample_points = mesh.sample_points_poisson_disk(10000)

    face_centers = np.asarray(sample_points.points)
    face_normals = np.asarray(sample_points.normals)

    initial_tvec = face_centers - obs_pt
    initial_tvec = initial_tvec / np.linalg.norm(initial_tvec, axis=1)[:, None]
    az = np.arctan2(initial_tvec[:, 0], initial_tvec[:, 1])
    el = -np.arcsin(initial_tvec[:, 2])

    pointing_az = np.arctan2(ant_point[0], ant_point[1])
    pointing_el = -np.arcsin(ant_point[2] / np.linalg.norm(ant_point))

    rho_o = np.array([applyRadiationPatternCPU(e, a, pointing_el, pointing_az, pointing_el, pointing_az,
                             40 * DTR, 40 * DTR) for e, a in zip(el, az)]) * 100

    int_tri = np.zeros(face_centers.shape[0])

    sigma = np.asarray(sample_points.colors)
    sigma = (sigma[:, 1] - (sigma[:, 0] + sigma[:, 2])**2) * np.pi / 2
    sigma[sigma <= 0] = .1

    # Test each ray and find the correct bounding box
    quad = np.array([[checkBoxIntersection(
        t, obs_pt, np.asarray(b.get_max_bound()), np.asarray(b.get_min_bound())) for t in initial_tvec]
        for b in boxes])

    # Get the observation point onto the GPU
    obs_pt_gpu = cupy.array(obs_pt, dtype=np.float32)

    # Test triangles in correct quads to find intsersection triangles
    for b_idx, box in enumerate(boxes):
        # ray_bounce, ray_xyz, vert_xyz, tri_norm, tri_verts, int_xyz
        poss_pts = quad[:, b_idx]
        if sum(poss_pts) > 0:
            int_gpu = cupy.array(face_centers[poss_pts], dtype=np.float32)
            ray_origin_gpu = cupy.array(np.repeat(obs_pt, (int_gpu.shape[0], 1)), dtype=np.float32)
            ray_dir_gpu = cupy.array(initial_tvec[poss_pts])
            tri_norm_gpu = cupy.array(quad_normals[b_idx], dtype=np.float32)
            tri_verts_gpu = cupy.array(quad_triangles[b_idx], dtype=np.float32)
            vert_xyz_gpu = cupy.array(quad_vertices[b_idx], dtype=np.float32)
            int_tri_gpu = cupy.array(int_gpu.shape[0], dtype=np.int32)

            bprun = (max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
                     tri_norm_gpu.shape[0] // threads_per_block[1] + 1)

            calcIntersection[bprun, threads_per_block](ray_dir_gpu, ray_origin_gpu, vert_xyz_gpu, tri_norm_gpu,
                                                       tri_verts_gpu, int_gpu, int_tri_gpu)
            face_centers[poss_pts] = int_gpu.get()
            initial_tvec[poss_pts] = ray_dir_gpu.get()
            int_tri[poss_pts] = int_tri_gpu.get()

    # Get the range profile from the intersection-tested rays
    bounce_gpu = cupy.array(initial_tvec - 2 * np.sum(initial_tvec * face_normals[int_tri], axis=1) *
                            face_normals[int_tri], dtype=np.float32)
    sigma_gpu = cupy.array(sigma, dtype=np.float32)
    pd_r = cupy.array(np.zeros(nsam), dtype=np.float64)
    pd_i = cupy.array(np.zeros(nsam), dtype=np.float64)

    genRangeProfileSinglePulse(int_gpu, bounce_gpu, sigma_gpu,
                               obs_pt_gpu, obs_pt_gpu, pointing_az, pointing_el, pointing_az, pointing_el, pd_r, pd_i, c0 / 9.6e9,
                               standoff / c0,
                               fs, bw_az, bw_el, 10., True, True)





    for p in tqdm(range(10000)):
        ray = initial_tvec[p]
        intersection = None
        inter_normal = None
        for q in quad[:, p]:
            if q:
                for tri_idx, tri in enumerate(quad_triangles[q - 1]):
                    e1 = quad_vertices[q - 1][tri[1]] - quad_vertices[q - 1][tri[0]]
                    e2 = quad_vertices[q - 1][tri[2]] - quad_vertices[q - 1][tri[0]]
                    cross = np.cross(ray, e2)

                    det = np.dot(e1, cross)
                    # Check to see if ray is parallel to triangle
                    if abs(det) < 1e-9:
                        continue

                    inv_det = 1. / det
                    svec = obs_pt - quad_vertices[q - 1][tri[0]]

                    u = inv_det * svec.dot(cross)
                    if u < 0 or u > 1:
                        continue

                    # Recompute cross for s and edge 1
                    cross = np.cross(svec, e1)
                    v = inv_det * ray.dot(cross)
                    if v < 0 or u + v > 1:
                        continue

                    # Compute intersection point
                    t = inv_det * e2.dot(cross)
                    if t < 1e-9:
                        continue

                    if intersection is not None:
                        new_inter = obs_pt + ray * t
                        if np.linalg.norm(obs_pt - new_inter) < np.linalg.norm(obs_pt - intersection):
                            intersection = obs_pt + ray * t
                            inter_normal = quad_normals[q - 1][tri_idx]
                    else:
                        intersection = obs_pt + ray * t
                        inter_normal = quad_normals[q - 1][tri_idx]
        face_centers[p] = intersection
        face_normals[p] = inter_normal

    # Get corrected ranges for power calculation, and get bounces as well
    tvec = face_centers - obs_pt
    rng = np.linalg.norm(tvec, axis=1)
    tvec = tvec / np.linalg.norm(tvec, axis=1)[:, None]
    bounces = np.array([bounceVector(t, fn) for t, fn in zip(tvec, face_normals)])

    # Calculate returned power based on bounce vector
    delta_phi = np.arccos(np.sum(tvec * bounces, axis=1))
    att = np.sinc(sigma * delta_phi)**2
    rho_final = rho_o * att / rng**4

    face_cloud = o3d.geometry.PointCloud()
    face_cloud.points = o3d.utility.Vector3dVector(np.concatenate((face_centers, obs_pt.reshape(1, 3))))
    face_cloud.normals = o3d.utility.Vector3dVector(np.concatenate((bounces * (rho_final / rho_final.mean())[:, None],
                                                                    -(obs_pt - mesh.get_center()).reshape(1, 3))))

    o3d.visualization.draw_geometries([mesh, face_cloud])
