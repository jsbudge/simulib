import cmath
import math
from tqdm import tqdm
from numba import cuda, njit
from numba.cuda.random import xoroshiro128p_uniform_float64
import cupy
from traitlets import observe

from simulation_functions import findPowerOf2
import numpy as np
import open3d as o3d
from cuda_kernels import applyRadiationPattern, applyRadiationPatternCPU, getMaxThreads
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

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
def calcIntersection(ray_dir, ray_xyz, vert_xyz, tri_norm, tri_verts, int_xyz, int_tri, ray_bounce):
    ray_idx, tri_idx = cuda.grid(ndim=2)
    if tri_norm.shape[0] > tri_idx and ray_idx < ray_dir.shape[0]:
        rx = ray_xyz[ray_idx, 0]
        ry = ray_xyz[ray_idx, 1]
        rz = ray_xyz[ray_idx, 2]
        e1x = vert_xyz[tri_verts[tri_idx, 1], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
        e1y = vert_xyz[tri_verts[tri_idx, 1], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
        e1z = vert_xyz[tri_verts[tri_idx, 1], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
        e2x = vert_xyz[tri_verts[tri_idx, 2], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
        e2y = vert_xyz[tri_verts[tri_idx, 2], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
        e2z = vert_xyz[tri_verts[tri_idx, 2], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
        crossx = ray_dir[ray_idx, 1] * e2z - ray_dir[ray_idx, 2] * e2y
        crossy = ray_dir[ray_idx, 2] * e2x - ray_dir[ray_idx, 0] * e2z
        crossz = ray_dir[ray_idx, 0] * e2y - ray_dir[ray_idx, 1] * e2x
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
        v = inv_det * (ray_dir[ray_idx, 0] * crossx + ray_dir[ray_idx, 1] * crossy + ray_dir[ray_idx, 2] * crossz)
        if v < 0 or u + v > 1:
            return

        # Compute intersection point
        t = inv_det * (e2x * crossx + e2y * crossy + e2z * crossz)
        if t < 1e-9:
            return

        vnx = tri_norm[tri_idx, 0]
        vny = tri_norm[tri_idx, 1]
        vnz = tri_norm[tri_idx, 2]

        intx = rx + t * ray_dir[ray_idx, 0]
        inty = ry + t * ray_dir[ray_idx, 1]
        intz = rz + t * ray_dir[ray_idx, 2]

        if (math.sqrt(abs(intx - rx) ** 2 + abs(inty - ry) ** 2 + abs(intz - rz) ** 2) <=
                math.sqrt(abs(int_xyz[ray_idx, 0] - rx) ** 2 + abs(int_xyz[ray_idx, 1] - ry) ** 2 +
                          abs(int_xyz[ray_idx, 2] - rz) ** 2)):
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
def genRangeProfileSinglePulse(int_xyz, vert_bounce, vert_reflectivity, source_xyz, receive_xyz, ray_dist,
                               ray_init_power, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                               source_fs, bw_az, bw_el):
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
        _, _, _, rng, _, _ = getRangeAndAngles(vx, vy, vz, source_xyz[pidx][0], source_xyz[pidx][1], source_xyz[pidx][2])
        rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(vx, vy, vz, receive_xyz[pidx][0], receive_xyz[pidx][1], receive_xyz[pidx][2])

        # Calculate bounce vector and strength
        # Apply Rayleigh scattering with the sigma being the width of the distribution
        x = max(0., (-rx * bx + -ry * by + -rz * bz) / r_rng)
        gamma = (x / (vert_reflectivity[pidx] * vert_reflectivity[pidx]) *
                 math.exp(-(x * x) / (2 * vert_reflectivity[pidx] * vert_reflectivity[pidx])))

        two_way_rng = ray_dist[pidx] + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin)
        if but > pd_r.shape[0] or but < 0:
            return

        if n_samples > but > 0:
            att = applyRadiationPattern(r_el, r_az, panrx, elrx, pantx, eltx, bw_az, bw_el) / (
                    two_way_rng * two_way_rng) * ray_init_power[pidx]
            acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * gamma
            cuda.atomic.add(pd_r, but, acc_val.real)
            cuda.atomic.add(pd_i, but, acc_val.imag)
        cuda.syncthreads()


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
    tmin = -np.inf
    tmax = np.inf
    if ray[0] != 0.:
        tx1 = (box_min[0] - ray_origin[0]) / ray[0]
        tx2 = (box_max[0] - ray_origin[0]) / ray[0]
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))
    if ray[1] != 0.:
        ty1 = (box_min[1] - ray_origin[1]) / ray[1]
        ty2 = (box_max[1] - ray_origin[1]) / ray[1]
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
    if ray[2] != 0.:
        tz1 = (box_min[2] - ray_origin[2]) / ray[2]
        tz2 = (box_max[2] - ray_origin[2]) / ray[2]
        tmin = max(tmin, min(tz1, tz2))
        tmax = min(tmax, max(tz1, tz2))
    return tmax >= tmin and tmax >= 0


def getRangeProfileFromMesh(mesh, obs_pt, voxel_size=1000, sample_points=10000, bounce_rays=15, num_bounces=3):
    # GPU device calculations
    threads_per_block = getMaxThreads()

    # Generate bounding box tree
    print('Generating bounding box tree...')
    full_box = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    boxes = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(full_box.get_voxel_bounding_points(t.grid_index))
             for t in full_box.get_voxels()]
    mesh_quads = [mesh.crop(b) for b in boxes]
    quad_triangles = [np.asarray(m.triangles) for m in mesh_quads]
    quad_vertices = [np.asarray(m.vertices) for m in mesh_quads]
    quad_normals = [np.asarray(m.triangle_normals) for m in mesh_quads]

    ant_point = mesh.get_center() - obs_pt

    print('Sampling mesh...')
    sample_points = mesh.sample_points_poisson_disk(sample_points)
    face_centers = np.asarray(sample_points.points)

    initial_tvec = face_centers - obs_pt
    initial_tvec = initial_tvec / np.linalg.norm(initial_tvec, axis=1)[:, None]
    az = np.arctan2(initial_tvec[:, 0], initial_tvec[:, 1])
    el = -np.arcsin(initial_tvec[:, 2])

    pointing_az = np.arctan2(ant_point[0], ant_point[1])
    pointing_el = -np.arcsin(ant_point[2] / np.linalg.norm(ant_point))

    rho_o = np.array([applyRadiationPatternCPU(e, a, pointing_az, pointing_el, pointing_az, pointing_el,
                                               bw_az, bw_el) for e, a in zip(el, az)]) * 1e12
    obs_pt_vec = np.repeat(obs_pt.reshape(1, 3), initial_tvec.shape[0], axis=0)
    inter_xyz, inter_bounce_dir, inter_tri_idx = bounce(obs_pt_vec, initial_tvec, boxes, quad_vertices, quad_triangles,
                                                        quad_normals)

    '''sigma = np.asarray(sample_points.colors)
    sigma = (sigma[:, 1] - (sigma[:, 0] + sigma[:, 2])**2) * np.pi / 2
    sigma[sigma <= 0] = .1'''
    sigma = np.ones(face_centers.shape[0])

    # Get the range profile from the intersection-tested rays
    rng = np.linalg.norm(inter_xyz - obs_pt[None, :], axis=1)
    bounce_gpu = cupy.array(inter_bounce_dir, dtype=np.float64)
    sigma_gpu = cupy.array(sigma, dtype=np.float64)
    pd_r = cupy.array(np.zeros(nsam), dtype=np.float64)
    pd_i = cupy.array(np.zeros(nsam), dtype=np.float64)
    int_gpu = cupy.array(inter_xyz, dtype=np.float64)
    ray_dist_gpu = cupy.array(rng, dtype=np.float64)
    ray_power_gpu = cupy.array(rho_o, dtype=np.float64)
    obs_pt_gpu = cupy.array(obs_pt_vec, dtype=np.float64)

    genRangeProfileSinglePulse[max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
    threads_per_block[0]](int_gpu, bounce_gpu, sigma_gpu, obs_pt_gpu, obs_pt_gpu, ray_dist_gpu, ray_power_gpu,
                          pointing_az, pointing_el, pointing_az, pointing_el, pd_r, pd_i, c0 / fc, standoff / c0, fs,
                          bw_az, bw_el)

    single_pulse = pd_r.get() + 1j * pd_i.get()

    # Calculate returned power based on bounce vector
    delta_phi = np.sum(initial_tvec * inter_bounce_dir, axis=1)
    delta_phi[delta_phi < 0] = 0.
    att = delta_phi / sigma ** 2 * np.exp(-delta_phi ** 2 / (2 * sigma ** 2))
    rho_final = rho_o * att / rng ** 4

    for face in tqdm(range(0, inter_xyz.shape[0] - 1, 2)):
        sobs_pt = np.repeat(inter_xyz[face].reshape(1, 3), bounce_rays, axis=0)
        obs_bounce = np.repeat(inter_bounce_dir[face].reshape(1, 3), bounce_rays, axis=0)
        paz = np.arctan2(obs_bounce[:, 0], obs_bounce[:, 1])
        pel = -np.arcsin(obs_bounce[:, 2])
        tvec = azelToVec(paz + np.random.normal(0, sigma[face] * .655, (bounce_rays,)),
                  pel + np.random.normal(0, sigma[face] * .655, (bounce_rays,))).T
        rho_bounce = rho_final[face] * np.ones(bounce_rays)
        curr_sigma = sigma[face] * np.ones(bounce_rays)

        for _ in range(num_bounces):
            # Calulate out the angle and expected power of that angle
            delta_phi = np.sum(tvec * obs_bounce, axis=1)
            delta_phi[delta_phi < 0] = 0.
            att = delta_phi / curr_sigma ** 2 * np.exp(-delta_phi ** 2 / (2 * curr_sigma ** 2))
            rho_bounce = rho_bounce * att

            # Cull rays that are below a significant power
            valids = rho_bounce < 1e-8
            if not np.any(valids):
                break
            tvec = tvec[valids]
            rho_bounce = rho_bounce[valids]
            sobs_pt = sobs_pt[valids]

            nb_xyz, nb_bounce_dir, nb_tri_idx = bounce(sobs_pt, tvec, boxes, quad_vertices, quad_triangles, quad_normals)
            valids = nb_tri_idx >= 0
            nb_xyz = nb_xyz[valids]
            nb_tri_idx = nb_tri_idx[valids]
            nb_bounce_dir = nb_bounce_dir[valids]
            sobs_pt = sobs_pt[valids]
            rho_bounce = rho_bounce[valids]


            b_sigma = np.ones(nb_xyz.shape[0]) * sigma[face]

            acc_rng = np.linalg.norm(nb_xyz - sobs_pt, axis=1)

            # Check for occlusion against the receiver and then accumulate the returns
            check_dir = nb_xyz - obs_pt
            check_dir = check_dir / np.linalg.norm(check_dir, axis=1)[:, None]

            _, _, check_tri_idx = bounce(nb_xyz, check_dir, boxes, quad_vertices, quad_triangles,
                                                                quad_normals)
            valids = check_tri_idx >= 0

            # Get the range profile from the intersection-tested rays
            bounce_gpu = cupy.array(nb_bounce_dir[valids], dtype=np.float32)
            sigma_gpu = cupy.array(b_sigma[valids], dtype=np.float32)
            int_gpu = cupy.array(nb_xyz[valids], dtype=np.float32)
            ray_dist_gpu = cupy.array(rng[face] + acc_rng[valids], dtype=np.float32)
            ray_power_gpu = cupy.array(rho_bounce[valids], dtype=np.float32)
            obs_pt_gpu = cupy.array(sobs_pt[valids], dtype=np.float32)
            pd_r = cupy.array(np.zeros(nsam), dtype=np.float64)
            pd_i = cupy.array(np.zeros(nsam), dtype=np.float64)

            genRangeProfileSinglePulse[max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
            threads_per_block[0]](int_gpu, bounce_gpu, sigma_gpu, obs_pt_gpu, obs_pt_gpu, ray_dist_gpu, ray_power_gpu,
                                  pointing_az, pointing_el, pointing_az, pointing_el, pd_r, pd_i, c0 / fc, standoff / c0,
                                  fs, bw_az, bw_el)
            single_pulse += pd_r.get() + 1j * pd_r.get()
            obs_bounce = tvec + 0.0
            sobs_pt = nb_xyz
            tvec = nb_bounce_dir
            curr_sigma = nb_tri_idx
    return single_pulse


def bounce(ray_origin, ray_dir, bounding_boxes, tri_verts, tri_idxes, tri_norms):
    # GPU device calculations
    threads_per_block = getMaxThreads()

    quad = np.array([[checkBoxIntersection(
        d, o, np.asarray(b.get_max_bound()), np.asarray(b.get_min_bound())) for d, o in zip(ray_dir, ray_origin)]
        for b in bounding_boxes])

    # Test triangles in correct quads to find intsersection triangles
    inter_xyz = np.zeros_like(ray_origin)
    inter_tri_idx = np.zeros(inter_xyz.shape[0]) - 1
    inter_bounce_dir = np.zeros_like(ray_origin)
    for b_idx, box in enumerate(bounding_boxes):
        # ray_bounce, ray_xyz, vert_xyz, tri_norm, tri_verts, int_xyz
        poss_rays = quad[b_idx, :]
        ray_num = sum(poss_rays)
        if ray_num > 0:
            int_gpu = cupy.zeros((ray_num, 3), dtype=np.float32)
            ray_origin_gpu = cupy.array(ray_origin[poss_rays], dtype=np.float32)
            ray_dir_gpu = cupy.array(ray_dir[poss_rays])
            tri_norm_gpu = cupy.array(tri_norms[b_idx], dtype=np.float32)
            tri_idxes_gpu = cupy.array(tri_idxes[b_idx], dtype=np.int32)
            tri_verts_gpu = cupy.array(tri_verts[b_idx], dtype=np.float32)
            int_tri_gpu = cupy.zeros(int_gpu.shape[0], dtype=np.int32)
            bounce_gpu = cupy.zeros_like(ray_dir_gpu)

            bprun = (max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
                     tri_norm_gpu.shape[0] // threads_per_block[1] + 1)

            calcIntersection[bprun, threads_per_block](ray_dir_gpu, ray_origin_gpu, tri_verts_gpu, tri_norm_gpu,
                                                       tri_idxes_gpu, int_gpu, int_tri_gpu, bounce_gpu)
            inter_xyz[poss_rays] = int_gpu.get()
            inter_bounce_dir[poss_rays] = bounce_gpu.get()
            inter_tri_idx[poss_rays] = int_tri_gpu.get()
    return inter_xyz, inter_bounce_dir, inter_tri_idx



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from simulation_functions import db, genChirp, azelToVec

    mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)
    # mesh = o3d.geometry.TriangleMesh.create_sphere(radius=100, resolution=50)
    mesh = mesh.compute_triangle_normals()
    mesh = mesh.compute_vertex_normals()
    obs_pt = np.array([0., -800., 700.])
    nr = 4096
    fc = 32.0e9
    standoff = 700.
    bw_az = 10 * DTR
    bw_el = 10 * DTR

    nsam = int(np.round(standoff / c0 * fs))
    fft_len = findPowerOf2(nsam + nr)
    up_fft_len = fft_len * 4
    chirp = genChirp(nr, fs, fc, 400e6)

    single_rp = getRangeProfileFromMesh(mesh, obs_pt, num_bounces=1)
    single_pulse = np.fft.ifft(np.fft.fft(chirp, fft_len) * np.fft.fft(single_rp, fft_len))[:nsam]
    single_mf_pulse = np.fft.ifft(np.fft.fft(chirp, fft_len) * np.fft.fft(single_rp, fft_len) * np.fft.fft(chirp, fft_len).conj())[:nsam]

    '''first_bounce_cloud = o3d.geometry.PointCloud()
    first_bounce_cloud.points = o3d.utility.Vector3dVector(np.concatenate((face_centers, obs_pt.reshape(1, 3))))
    first_bounce_cloud.normals = o3d.utility.Vector3dVector(
        np.concatenate((bounce_vec, # * (rho_final / rho_scaling)[:, None],
                        -(obs_pt - mesh.get_center()).reshape(1, 3))))



    o3d.visualization.draw_geometries([mesh, first_bounce_cloud])'''


    '''check = o3d.geometry.PointCloud()
    check.points = o3d.utility.Vector3dVector(face_centers)

    c2 = o3d.geometry.PointCloud()
    c2.points = o3d.utility.Vector3dVector(face_second)
    c2.normals = o3d.utility.Vector3dVector(bounce_sec)

    o3d.visualization.draw_geometries([check, c2])'''
    px.scatter(db(single_rp)).show()
    px.scatter(db(single_pulse)).show()
    px.scatter(db(single_mf_pulse)).show()

