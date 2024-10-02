import cmath
import math
from simulation_functions import factors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from numba import cuda
from line_profiler_pycharm import profile
import cupy
from simulation_functions import findPowerOf2, azelToVec
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
RAYLEIGH_SIGMA_COEFF = .655
CUTOFF = 200000


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
def calcIntersection(ray_dir, ray_xyz, vert_xyz, tri_norm, tri_verts, tri_sigmas, tri_poss, tri_box, int_xyz,
                     int_sigma, ray_bounce):
    ray_idx, tri_idx = cuda.grid(ndim=2)
    if tri_norm.shape[0] > tri_idx and ray_idx < ray_dir.shape[0]:
        if tri_poss[ray_idx, tri_box[tri_idx]]:
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
                # print('.')
                # continue
                return

            inv_det = 1. / det
            sx = rx - vert_xyz[tri_verts[tri_idx, 0], 0]
            sy = ry - vert_xyz[tri_verts[tri_idx, 0], 1]
            sz = rz - vert_xyz[tri_verts[tri_idx, 0], 2]

            u = inv_det * (sx * crossx + sy * crossy + sz * crossz)
            if u < 0 or u > 1:
                # print('u')
                # continue
                return

            # Recompute cross for s and edge 1
            crossx = sy * e1z - sz * e1y
            crossy = sz * e1x - sx * e1z
            crossz = sx * e1y - sy * e1x
            v = inv_det * (ray_dir[ray_idx, 0] * crossx + ray_dir[ray_idx, 1] * crossy + ray_dir[ray_idx, 2] * crossz)
            if v < 0 or u + v > 1:
                # print('v')
                # continue
                return

            # Compute intersection point
            t = inv_det * (e2x * crossx + e2y * crossy + e2z * crossz)
            # print(f'{tri_idx} {t}')
            if t < 1e-9:
                return

            vnx = tri_norm[tri_idx, 0]
            vny = tri_norm[tri_idx, 1]
            vnz = tri_norm[tri_idx, 2]

            intx = rx + t * ray_dir[ray_idx, 0]
            inty = ry + t * ray_dir[ray_idx, 1]
            intz = rz + t * ray_dir[ray_idx, 2]

            if int_sigma[ray_idx] < 0:
                # Calculate out the angles in azimuth and elevation for the bounce
                tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)

                bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
                bx = tx - vnx * bounce_dot
                by = ty - vny * bounce_dot
                bz = tz - vnz * bounce_dot
                bounce_len = 1 / math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz))
                ray_bounce[ray_idx, 0] = bx * bounce_len
                ray_bounce[ray_idx, 1] = by * bounce_len
                ray_bounce[ray_idx, 2] = bz * bounce_len
                int_xyz[ray_idx, 0] = intx
                int_xyz[ray_idx, 1] = inty
                int_xyz[ray_idx, 2] = intz
                int_sigma[ray_idx] = tri_sigmas[tri_idx]
            elif (math.sqrt(abs(intx - rx) ** 2 + abs(inty - ry) ** 2 + abs(intz - rz) ** 2) <=
                    math.sqrt(abs(int_xyz[ray_idx, 0] - rx) ** 2 + abs(int_xyz[ray_idx, 1] - ry) ** 2 +
                              abs(int_xyz[ray_idx, 2] - rz) ** 2)):
                # Calculate out the angles in azimuth and elevation for the bounce
                tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)

                bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
                bx = tx - vnx * bounce_dot
                by = ty - vny * bounce_dot
                bz = tz - vnz * bounce_dot
                bounce_len = 1 / math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz))
                ray_bounce[ray_idx, 0] = bx * bounce_len
                ray_bounce[ray_idx, 1] = by * bounce_len
                ray_bounce[ray_idx, 2] = bz * bounce_len
                int_xyz[ray_idx, 0] = intx
                int_xyz[ray_idx, 1] = inty
                int_xyz[ray_idx, 2] = intz
                int_sigma[ray_idx] = tri_sigmas[tri_idx]
        cuda.syncthreads()



def calcIntersectionCPU(ray_dir, ray_xyz, vert_xyz, tri_norm, tri_verts, int_tri):
    tri_edges = vert_xyz[tri_verts]
    e1 = tri_edges[:, 1, :] - tri_edges[:, 0, :]
    e2 = tri_edges[:, 2, :] - tri_edges[:, 0, :]
    ray_bounce = np.zeros((ray_dir.shape[0], 3))
    int_xyz = np.zeros_like(ray_bounce)
    tri_idx_array = np.arange(tri_norm.shape[0])
    for r in range(ray_dir.shape[0]):
        cross = np.cross(ray_dir[r], e2)
        det = np.sum(e1 * cross, axis=1)

        # Check to see if ray is parallel to triangle
        valids = abs(det) > 1e-9
        if not np.any(valids):
            continue

        inv_det = 1. / det[valids]
        s = ray_xyz[r] - tri_edges[valids, 0, :]

        u = inv_det * np.sum(s * cross[valids], axis=1)

        v2 = np.logical_and(u >= 0, u <= 1)
        valids[valids] = v2

        if not np.any(valids):
            continue
        inv_det = inv_det[v2]

        # Recompute cross for s and edge 1
        cross = np.cross(s[v2], e1[valids])
        v = inv_det * np.sum(ray_dir[r] * cross, axis=1)
        v3 = np.logical_and(v >= 0, u[v2] + v <= 1)
        valids[valids] = v3

        if not np.any(valids):
            continue
        inv_det = inv_det[v3]

        # Compute intersection point
        t = inv_det * np.sum(e2[valids] * cross[v3], axis=1)
        v4 = t > 1e-9
        valids[valids] = v4

        if not np.any(valids):
            continue

        vn = tri_norm[valids]
        int_loc = ray_xyz[r] + np.outer(t[v4], ray_dir[r])
        tt = int_loc - ray_xyz[r]
        if len(int_loc.shape) > 1:
            tt_rng = np.linalg.norm(tt, axis=1)
            final_pt = tt_rng == tt_rng.min()
            valids[valids] = final_pt
            int_loc = int_loc[final_pt]
            tt = int_loc - ray_xyz[r]
            vn = vn[final_pt]

        # Calculate out the angles in azimuth and elevation for the bounce
        bounce_dir = tt - vn * (2 * np.sum(tt * vn))
        ray_bounce[r] = bounce_dir / np.linalg.norm(bounce_dir)
        int_xyz[r] = int_loc
        int_tri[r] = tri_idx_array[valids]
    return int_xyz, ray_bounce, int_tri


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
        _, _, _, rng, _, _ = getRangeAndAngles(vx, vy, vz, source_xyz[pidx, 0], source_xyz[pidx, 1],
                                               source_xyz[pidx, 2])
        rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(vx, vy, vz, receive_xyz[pidx, 0], receive_xyz[pidx, 1],
                                                          receive_xyz[pidx, 2])

        # Calculate bounce vector and strength
        # Apply Rayleigh scattering with the sigma being the width of the distribution
        x = max(0., (rx * bx + ry * by + rz * bz) / r_rng)
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
            # print(abs(gamma))
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


def checkBoxIntersection(ray, ray_origin, boxes):
    box_min = boxes[:, 3:]
    box_max = boxes[:, :3]
    quad_return = np.zeros((ray.shape[0], boxes.shape[0]))
    for m in range(boxes.shape[0]):
        tmin = np.ones(ray.shape[0]) * -np.inf
        tmax = np.ones(ray.shape[0]) * np.inf
        for n in range(3):
            tx1 = (box_min[m, n] - ray_origin[:, n]) / ray[:, n]
            tx2 = (box_max[m, n] - ray_origin[:, n]) / ray[:, n]
            tmin = np.maximum(tmin, np.minimum(tx1, tx2))
            tmax = np.minimum(tmax, np.maximum(tx1, tx2))
        quad_return[:, m] = np.logical_and(tmax - tmin >= 0, tmax >= 0)
    return quad_return.astype(bool)


@cuda.jit()
def checkBoxRayIntersection(ray_dir, ray_origin, boxes):
    


def getBoxesSamplesFromMesh(a_mesh, sigma=None, num_boxes=4, sample_points=10000):
    # Generate bounding box tree
    if np.sqrt(num_boxes) % 1 == 0:
        nx = int(np.sqrt(num_boxes))
        ny = int(np.sqrt(num_boxes))
    else:
        facts = factors(num_boxes)
        nx = int(facts[len(facts) // 2])
        ny = int(num_boxes // nx)
    aabb = a_mesh.get_axis_aligned_bounding_box()
    max_bound = aabb.get_max_bound()
    min_bound = aabb.get_min_bound()
    xes = np.linspace(min_bound[0], max_bound[0], nx + 1)
    yes = np.linspace(min_bound[1], max_bound[1], ny + 1)
    boxes = []
    for x in range(1, len(xes)):
        boxes.extend(
            o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(
                    np.array(
                        [
                            [xes[x - 1] - 1, yes[y - 1] - 1, min_bound[2] - 1],
                            [xes[x] + 1, yes[y] + 1, max_bound[2] + 1],
                        ]
                    )
                )
            )
            for y in range(1, len(yes))
        )
    if sigma is not None:
        a_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array([sigma, np.zeros_like(sigma)]).T)
    else:
        a_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.ones((len(a_mesh.triangles), 2)))
    mesh_quads = [a_mesh.crop(b) for b in boxes]
    mesh_quads = [m for m in mesh_quads if len(m.triangles) > 0]
    boxes = np.array([[*b.get_max_bound(), *b.get_min_bound()] for b in mesh_quads])
    quad_triangles = [np.asarray(m.triangles) for m in mesh_quads]
    quad_vertices = [np.asarray(m.vertices) for m in mesh_quads]
    quad_normals = [np.asarray(m.triangle_normals) for m in mesh_quads]
    quad_colors = [np.asarray(m.vertex_colors) for m in mesh_quads]
    quad_checks = [v[t].mean(axis=1) for v, t in zip(quad_colors, quad_triangles)]

    quad_sigmas = [(f[:, 1] - (f[:, 0] + f[:, 2]) ** 2) / f.max() * 5 for f in quad_checks]
    for q in quad_sigmas:
        q[q <= 0] = .1

    nperquad = int(sample_points // boxes.shape[0])
    # points = a_mesh.sample_points_poisson_disk(sample_points)
    points = mesh_quads[0].sample_points_poisson_disk(nperquad + sample_points - nperquad * boxes.shape[0])
    for m in mesh_quads[1:]:
        points += m.sample_points_poisson_disk(nperquad)
    return (boxes, quad_triangles, quad_vertices, quad_normals, quad_sigmas), points

@profile
def getRangeProfileFromMesh(boxes, quad_triangles, quad_vertices, quad_normals, quad_sigmas, sample_points, a_obs_pt,
                            pointing_vec, radar_equation_constant, bw_az, bw_el, nsam, fc, near_range_s, bounce_rays=15,
                            num_bounces=3, debug=False):
    # GPU device calculations
    threads_per_block = getMaxThreads()
    face_centers = np.asarray(sample_points.points)
    pulse_ret = np.zeros((a_obs_pt.shape[0], nsam), dtype=np.complex128)

    tri_box_idx = np.concatenate([np.ones(b.shape[0]) * idx for idx, b in enumerate(quad_triangles)])

    for t in range(a_obs_pt.shape[0]):
        initial_tvec = face_centers - a_obs_pt[t]
        initial_tvec = initial_tvec / np.linalg.norm(initial_tvec, axis=1)[:, None]

        # Generate the pointing vector for the antenna and the antenna pattern
        # that will affect the power received by the intersection point
        pointing_az = np.arctan2(pointing_vec[t, 0], pointing_vec[t, 1])
        pointing_el = -np.arcsin(pointing_vec[t, 2])

        obs_pt_vec = np.repeat(a_obs_pt[t].reshape(1, 3), initial_tvec.shape[0], axis=0)
        inter_xyz, inter_bounce_dir, face_sigma = bounce(obs_pt_vec, initial_tvec, boxes, quad_vertices, quad_triangles,
                                                            quad_normals, quad_sigmas, tri_box_idx)

        valids = face_sigma > 0
        if sum(valids) == 0:
            continue
        inter_xyz = inter_xyz[valids]
        inter_bounce_dir = inter_bounce_dir[valids]
        face_sigma = face_sigma[valids]
        obs_pt_vec = obs_pt_vec[valids]
        rho_o = np.ones(inter_xyz.shape[0]) * radar_equation_constant

        # Get the range profile from the intersection-tested rays
        rng = np.linalg.norm(inter_xyz - a_obs_pt[t, :], axis=1)
        bounce_gpu = cupy.array(inter_bounce_dir, dtype=np.float64)
        sigma_gpu = cupy.array(face_sigma, dtype=np.float64)
        pd_r = cupy.array(np.zeros(nsam), dtype=np.float64)
        pd_i = cupy.array(np.zeros(nsam), dtype=np.float64)
        int_gpu = cupy.array(inter_xyz, dtype=np.float64)
        ray_dist_gpu = cupy.array(rng, dtype=np.float64)
        ray_power_gpu = cupy.array(rho_o, dtype=np.float64)
        obs_pt_gpu = cupy.array(obs_pt_vec, dtype=np.float64)

        genRangeProfileSinglePulse[max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
        threads_per_block[0]](int_gpu, bounce_gpu, sigma_gpu, obs_pt_gpu, obs_pt_gpu, ray_dist_gpu, ray_power_gpu,
                              pointing_az, pointing_el, pointing_az, pointing_el, pd_r, pd_i, c0 / fc, near_range_s, fs,
                              bw_az, bw_el)

        pulse_ret[t, :] = pd_r.get() + 1j * pd_i.get()

        # Calculate returned power based on bounce vector
        delta_phi = np.sum(initial_tvec[valids] * inter_bounce_dir, axis=1)
        delta_phi[delta_phi < 0] = 0.
        att = delta_phi / face_sigma ** 2 * np.exp(-delta_phi ** 2 / (2 * face_sigma ** 2))
        rho_final = rho_o * att

        if debug:
            debug_rays = [inter_xyz]
            debug_raydirs = [inter_bounce_dir]
            debug_raypower = [rho_final]

        nfaces_possible = int(CUTOFF / bounce_rays)

        for face in range(0, inter_xyz.shape[0], nfaces_possible):
            fc_poss = range(face, min(face+nfaces_possible, inter_xyz.shape[0]))
            sobs_pt = np.repeat(inter_xyz[fc_poss], bounce_rays, axis=0)
            obs_bounce = np.repeat(inter_bounce_dir[fc_poss], bounce_rays, axis=0)
            paz = np.arctan2(obs_bounce[:, 0], obs_bounce[:, 1])
            pel = -np.arcsin(obs_bounce[:, 2])
            curr_sigma = np.repeat(face_sigma[fc_poss], bounce_rays)
            tvec = azelToVec(paz + np.random.normal(0, curr_sigma * RAYLEIGH_SIGMA_COEFF),
                             pel + np.random.normal(0, curr_sigma * RAYLEIGH_SIGMA_COEFF)).T
            rho_bounce = np.repeat(rho_final[fc_poss], bounce_rays)
            acc_rng = np.repeat(rng[fc_poss], bounce_rays)

            for _ in range(num_bounces):

                # Cull rays that are below a significant power
                valids = rho_bounce > 1e-8
                if not np.any(valids):
                    break
                tvec = tvec[valids]
                rho_bounce = rho_bounce[valids]
                sobs_pt = sobs_pt[valids]
                acc_rng = acc_rng[valids]

                if debug:
                    debug_rays += [sobs_pt]
                    debug_raydirs += [tvec]
                    debug_raypower += [rho_bounce]

                nb_xyz, nb_bounce_dir, nb_sigma = bounce(sobs_pt, tvec, boxes, quad_vertices, quad_triangles,
                                                           quad_normals, quad_sigmas, tri_box_idx)
                acc_rng += np.linalg.norm(nb_xyz - sobs_pt, axis=1)
                valids = np.logical_and(nb_sigma >= 0, acc_rng > 1e-1)
                if not np.any(valids):
                    break

                nb_xyz = nb_xyz[valids]
                nb_sigma = nb_sigma[valids]
                nb_bounce_dir = nb_bounce_dir[valids]
                sobs_pt = sobs_pt[valids]
                acc_rng = acc_rng[valids]
                rho_bounce = rho_bounce[valids]

                # Check for occlusion against the receiver and then accumulate the returns
                check_dir = nb_xyz - a_obs_pt[t]
                check_dir = check_dir / np.linalg.norm(check_dir, axis=1)[:, None]

                _, _, check_tri_idx = bounce(nb_xyz, check_dir, boxes, quad_vertices, quad_triangles,
                                             quad_normals, quad_sigmas, tri_box_idx)
                valids = check_tri_idx >= 0
                if not np.any(valids):
                    break

                # Get the range profile from the intersection-tested rays
                bounce_gpu = cupy.array(nb_bounce_dir[valids], dtype=np.float32)
                sigma_gpu = cupy.array(nb_sigma[valids], dtype=np.float32)
                int_gpu = cupy.array(nb_xyz[valids], dtype=np.float32)
                ray_dist_gpu = cupy.array(acc_rng[valids], dtype=np.float32)
                ray_power_gpu = cupy.array(rho_bounce[valids], dtype=np.float32)
                obs_pt_gpu = cupy.array(sobs_pt[valids], dtype=np.float32)
                pd_r = cupy.array(np.zeros(nsam), dtype=np.float64)
                pd_i = cupy.array(np.zeros(nsam), dtype=np.float64)

                genRangeProfileSinglePulse[max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
                threads_per_block[0]](int_gpu, bounce_gpu, sigma_gpu, obs_pt_gpu, obs_pt_gpu, ray_dist_gpu, ray_power_gpu,
                                      pointing_az, pointing_el, pointing_az, pointing_el, pd_r, pd_i, c0 / fc,
                                      near_range_s,
                                      fs, bw_az, bw_el)
                pulse_ret[t, :] += pd_r.get() + 1j * pd_i.get()

                # Finished with the occlusion checking
                obs_bounce = nb_bounce_dir + 0.0
                tvec = nb_xyz - sobs_pt
                tvec = tvec / np.linalg.norm(tvec, axis=1)[:, None]
                sobs_pt = nb_xyz
                # Calulate out the angle and expected power of that angle
                try:
                    delta_phi = np.sum(tvec * obs_bounce, axis=1)
                    delta_phi[delta_phi < 0] = 0.
                    att = delta_phi / nb_sigma ** 2 * np.exp(-delta_phi ** 2 / (2 * nb_sigma ** 2))
                    rho_bounce = rho_bounce * att
                except ValueError:
                    break
        del bounce_gpu
        del sigma_gpu
        del pd_r
        del pd_i
        del int_gpu
        del ray_dist_gpu
        del ray_power_gpu
        del obs_pt_gpu
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    if debug:
        return pulse_ret, debug_rays, debug_raydirs, debug_raypower
    else:
        return pulse_ret

@profile
def bounce(ray_origin, ray_dir, bounding_boxes, tri_verts, tri_idxes, tri_norms, tri_sigmas, tri_box_idx):
    # GPU device calculations
    threads_per_block = getMaxThreads()

    quad = checkBoxIntersection(ray_dir, ray_origin, bounding_boxes)

    # Get groupings of boxes to speed up processing
    nrays_per_box = np.sum(quad, axis=0)
    cores_per_box = np.array([t.shape[0] * nrays_per_box[n] for n, t in enumerate(tri_norms)])
    core_idxes = np.argsort(cores_per_box)
    groups = [[core_idxes[0]]]
    nn = 1
    while nn < len(core_idxes):
        if sum(cores_per_box[groups[-1]]) + cores_per_box[core_idxes[nn]] < CUTOFF:
            groups[-1].append(core_idxes[nn])
        else:
            groups.append([core_idxes[nn]])
        nn += 1

    # Test triangles in correct quads to find intsersection triangles
    inter_xyz = np.zeros_like(ray_origin)
    inter_sigma = np.zeros(inter_xyz.shape[0]) - 1
    inter_bounce_dir = np.zeros_like(ray_origin)
    for box in groups:
        poss_rays = np.any(quad[:, box], axis=1)
        ray_num = sum(poss_rays)
        if ray_num > 0:
            int_gpu = cupy.zeros((ray_num, 3), dtype=np.float32)
            ray_origin_gpu = cupy.array(ray_origin[poss_rays], dtype=np.float32)
            ray_dir_gpu = cupy.array(ray_dir[poss_rays])
            int_sigma_gpu = cupy.array(inter_sigma[poss_rays], dtype=np.float32)
            bounce_gpu = cupy.zeros_like(ray_dir_gpu)

            tri_norm_gpu = cupy.array(np.concatenate([tri_norms[b] for b in box]), dtype=np.float32)
            tri_idxes_gpu = cupy.array(np.concatenate([tri_idxes[b] for b in box]), dtype=np.int32)
            tri_verts_gpu = cupy.array(np.concatenate([tri_verts[b] for b in box]), dtype=np.float32)
            tri_sigmas_gpu = cupy.array(np.concatenate([tri_sigmas[b] for b in box]), dtype=np.float32)
            tri_poss_gpu = cupy.array(np.concatenate([quad[poss_rays, b].reshape(ray_num, 1) for b in box], axis=1), dtype=bool)
            tri_box_gpu = cupy.array(np.concatenate([np.ones(tri_norms[b].shape[0]) * b for b in box]), dtype=np.int32)

            bprun = (max(1, int_gpu.shape[0] // threads_per_block[0] + 1),
                     tri_norm_gpu.shape[0] // threads_per_block[1] + 1)

            calcIntersection[bprun, threads_per_block](ray_dir_gpu, ray_origin_gpu, tri_verts_gpu, tri_norm_gpu,
                                                       tri_idxes_gpu, tri_sigmas_gpu, tri_poss_gpu, tri_box_gpu,
                                                       int_gpu, int_sigma_gpu, bounce_gpu)
            inter_xyz[poss_rays] = int_gpu.get()
            inter_bounce_dir[poss_rays] = bounce_gpu.get()
            inter_sigma[poss_rays] = int_sigma_gpu.get()
    return inter_xyz, inter_bounce_dir, inter_sigma


if __name__ == '__main__':

    mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)
