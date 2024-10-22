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
BOX_CUSHION = .1


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
    ray_idx, tri_idx, tt = cuda.grid(ndim=3)
    if tri_norm.shape[0] > tri_idx and ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:
        for box in range(tri_poss.shape[1]):
            if tri_poss[ray_idx, box] and tri_box[tri_idx, box]:
                rx = ray_xyz[tt, ray_idx, 0]
                ry = ray_xyz[tt, ray_idx, 1]
                rz = ray_xyz[tt, ray_idx, 2]
                e1x = vert_xyz[tri_verts[tri_idx, 1], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
                e1y = vert_xyz[tri_verts[tri_idx, 1], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
                e1z = vert_xyz[tri_verts[tri_idx, 1], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
                e2x = vert_xyz[tri_verts[tri_idx, 2], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
                e2y = vert_xyz[tri_verts[tri_idx, 2], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
                e2z = vert_xyz[tri_verts[tri_idx, 2], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
                crossx = ray_dir[tt, ray_idx, 1] * e2z - ray_dir[tt, ray_idx, 2] * e2y
                crossy = ray_dir[tt, ray_idx, 2] * e2x - ray_dir[tt, ray_idx, 0] * e2z
                crossz = ray_dir[tt, ray_idx, 0] * e2y - ray_dir[tt, ray_idx, 1] * e2x
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
                v = inv_det * (ray_dir[tt, ray_idx, 0] * crossx + ray_dir[tt, ray_idx, 1] * crossy + ray_dir[tt, ray_idx, 2] * crossz)
                if v < 0 or u + v > 1:
                    return

                # Compute intersection point
                t = inv_det * (e2x * crossx + e2y * crossy + e2z * crossz)
                if t < 1e-9:
                    return

                vnx = tri_norm[tri_idx, 0]
                vny = tri_norm[tri_idx, 1]
                vnz = tri_norm[tri_idx, 2]

                intx = rx + t * ray_dir[tt, ray_idx, 0]
                inty = ry + t * ray_dir[tt, ray_idx, 1]
                intz = rz + t * ray_dir[tt, ray_idx, 2]

                if int_sigma[tt, ray_idx] < 0:
                    # Calculate out the angles in azimuth and elevation for the bounce
                    tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)

                    bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
                    bx = tx - vnx * bounce_dot
                    by = ty - vny * bounce_dot
                    bz = tz - vnz * bounce_dot
                    bounce_len = 1 / math.sqrt(bx * bx + by * by + bz * bz)
                    ray_bounce[tt, ray_idx, 0] = bx * bounce_len
                    ray_bounce[tt, ray_idx, 1] = by * bounce_len
                    ray_bounce[tt, ray_idx, 2] = bz * bounce_len
                    int_xyz[tt, ray_idx, 0] = intx
                    int_xyz[tt, ray_idx, 1] = inty
                    int_xyz[tt, ray_idx, 2] = intz
                    int_sigma[tt, ray_idx] = tri_sigmas[tri_idx]
                elif (math.sqrt((intx - rx) ** 2 + (inty - ry) ** 2 + (intz - rz) ** 2) <=
                        math.sqrt((int_xyz[tt, ray_idx, 0] - rx) ** 2 + (int_xyz[tt, ray_idx, 1] - ry) ** 2 +
                                  (int_xyz[tt, ray_idx, 2] - rz) ** 2)):
                    # Calculate out the angles in azimuth and elevation for the bounce
                    tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)

                    bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
                    bx = tx - vnx * bounce_dot
                    by = ty - vny * bounce_dot
                    bz = tz - vnz * bounce_dot
                    bounce_len = 1 / math.sqrt(bx * bx + by * by + bz * bz)
                    ray_bounce[tt, ray_idx, 0] = bx * bounce_len
                    ray_bounce[tt, ray_idx, 1] = by * bounce_len
                    ray_bounce[tt, ray_idx, 2] = bz * bounce_len
                    int_xyz[tt, ray_idx, 0] = intx
                    int_xyz[tt, ray_idx, 1] = inty
                    int_xyz[tt, ray_idx, 2] = intz
                    int_sigma[tt, ray_idx] = tri_sigmas[tri_idx]
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


@cuda.jit()
def genRangeProfileMesh(int_xyz, vert_bounce, vert_reflectivity, source_xyz, receive_xyz, ray_dist,
                               ray_init_power, valid, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                               source_fs, bw_az, bw_el):
    # sourcery no-metrics
    tt, pidx = cuda.grid(ndim=2)
    if pidx < int_xyz.shape[1] and tt < int_xyz.shape[0]:
        if valid[tt, pidx]:
            # Load in all the parameters that don't change
            n_samples = pd_r.shape[1]
            wavenumber = 2 * np.pi / wavelength

            # Calculate the bounce vector for this time
            vx = int_xyz[tt, pidx, 0]
            vy = int_xyz[tt, pidx, 1]
            vz = int_xyz[tt, pidx, 2]
            bx = vert_bounce[tt, pidx, 0]
            by = vert_bounce[tt, pidx, 1]
            bz = vert_bounce[tt, pidx, 2]

            # Calculate out the angles in azimuth and elevation for the bounce
            _, _, _, rng, _, _ = getRangeAndAngles(vx, vy, vz, source_xyz[tt, pidx, 0], source_xyz[tt, pidx, 1],
                                                   source_xyz[tt, pidx, 2])
            rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(vx, vy, vz, receive_xyz[tt, pidx, 0], receive_xyz[tt, pidx, 1],
                                                              receive_xyz[tt, pidx, 2])

            # Calculate bounce vector and strength
            # Apply Rayleigh scattering with the sigma being the width of the distribution
            x = max(0., (rx * bx + ry * by + rz * bz) / r_rng)
            gamma = 1 - math.exp(-(1 + x)**2 / (2 * vert_reflectivity[tt, pidx]**2))

            two_way_rng = ray_dist[tt, pidx] + r_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin)
            if but > pd_r.shape[1] or but < 0:
                return

            if n_samples > but > 0:
                att = applyRadiationPattern(r_el, r_az, panrx[tt], elrx[tt], pantx[tt], eltx[tt], bw_az, bw_el) / (
                        two_way_rng * two_way_rng) * ray_init_power[tt, pidx]
                acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * gamma
                # print(abs(gamma))
                cuda.atomic.add(pd_r, (tt, but), acc_val.real)
                cuda.atomic.add(pd_i, (tt, but), acc_val.imag)
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


def readCombineMeshFile(fnme, points=100000, scale=None):
    full_mesh = o3d.io.read_triangle_model(fnme)
    mesh = o3d.geometry.TriangleMesh()
    num_tris = [len(me.mesh.triangles) for me in full_mesh.meshes]
    mids = []
    if sum(num_tris) > points:
        scaling = points / sum(num_tris)
        target_tris = [int(max(1, t * scaling)) for t in num_tris]
        for me_idx, me in enumerate(full_mesh.meshes):
            me.mesh.triangle_uvs = o3d.utility.Vector2dVector([])
            pcd = o3d.geometry.PointCloud(me.mesh.vertices)
            vertex_size = np.mean(pcd.compute_point_cloud_distance(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.mean(np.asarray(pcd.points), axis=0).reshape(1, -1))))) / 2
            tm = me.mesh.simplify_vertex_clustering(vertex_size)
            tm.remove_duplicated_vertices()
            tm.remove_unreferenced_vertices()
            bounce = 1
            while len(tm.triangles) > target_tris[me_idx] or len(tm.triangles) == 0:
                vertex_size *= (2 if len(tm.triangles) > target_tris[me_idx] else .6) * bounce
                tm = me.mesh.simplify_vertex_clustering(vertex_size)
                tm.remove_duplicated_vertices()
                tm.remove_unreferenced_vertices()
                bounce -= .1
                if bounce < .01:
                    break
            mesh += tm
            mids += [me.material_idx for _ in range(len(tm.triangles))]
    else:
        for me in full_mesh.meshes:
            mesh += me.mesh
            mids += [me.material_idx for _ in range(len(me.mesh.triangles))]
    mesh.triangle_material_ids = o3d.utility.IntVector(mids)

    if scale:
        mesh = mesh.scale(scale, mesh.get_center())
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    return mesh


def getBoxesSamplesFromMesh(a_mesh, num_boxes=4, sample_points=10000, material_sigmas=None):
    # Generate bounding box tree
    mesh_tri_idx = np.asarray(a_mesh.triangles)
    mesh_vertices = np.asarray(a_mesh.vertices)
    mesh_normals = np.asarray(a_mesh.triangle_normals)
    mesh_tri_vertices = mesh_vertices[mesh_tri_idx]
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
        for y in range(1, len(yes)):
            bpts = mesh_vertices[np.logical_and(np.logical_or(xes[x - 1] - BOX_CUSHION < mesh_vertices[:, 0],
                                                              mesh_vertices[:, 0] < xes[x] + BOX_CUSHION),
                                                np.logical_or(yes[y - 1] - BOX_CUSHION < mesh_vertices[:, 1],
                                                              mesh_vertices[:, 1] < yes[y] + BOX_CUSHION))]
            boxes.append(np.array(
                [
                    [xes[x - 1] - BOX_CUSHION, yes[y - 1] - BOX_CUSHION, bpts[:, 2].min() - BOX_CUSHION],
                    [xes[x] + BOX_CUSHION, yes[y] + BOX_CUSHION, bpts[:, 2].max() + BOX_CUSHION],
                ]
            ))

    mesh_box_idx = np.zeros((mesh_tri_idx.shape[0], len(boxes))).astype(bool)
    # Get the box index for each triangle, for exclusion in the GPU calculations
    for b_idx, box in enumerate(boxes):
        is_inside = (box[0, 0] < mesh_tri_vertices[:, :, 0]) & (mesh_tri_vertices[:, :, 0] < box[1, 0])
        for dim in range(1, 3):
            is_inside = is_inside & (box[0, dim] < mesh_tri_vertices[:, :, dim]) & (mesh_tri_vertices[:, :, dim] < box[1, dim])
        mesh_box_idx[:, b_idx] = np.any(is_inside, axis=1)

    # Remove any degenerate boxes, those that have no triangles inside
    deg_box = np.any(mesh_box_idx, axis=0)
    mesh_box_idx = mesh_box_idx[:, deg_box]
    boxes = np.array(boxes)[deg_box]

    # Simple calculation to get the grass and dirt with a bigger beta
    try:
        if material_sigmas is None:
            mesh_tri_colors = np.asarray(a_mesh.vertex_colors)[mesh_tri_idx].mean(axis=1)
            mesh_sigmas = np.linalg.norm(mesh_tri_colors - np.array([.4501, .6340, .3228]), axis=1)
            mesh_sigmas = mesh_sigmas.max() / mesh_sigmas
        else:
            mesh_sigmas = np.array([material_sigmas[i] for i in np.asarray(a_mesh.triangle_material_ids)])
    except:
        print('Could not extrapolate sigmas, setting everything to one.')
        mesh_sigmas = np.ones(len(a_mesh.triangles))

    # Sample the mesh to get points for initial raycasting
    points = a_mesh.sample_points_poisson_disk(sample_points)
    return (boxes, mesh_box_idx, mesh_tri_idx, mesh_vertices, mesh_normals, mesh_sigmas), np.asarray(points.points)

@profile
def getRangeProfileFromMesh(bounding_boxes, tri_box_idxes, mesh_tri_idxes, mesh_vertices, mesh_normals, mesh_sigmas,
                            sample_points, a_obs_pt, pointing_vec, radar_equation_constant, bw_az, bw_el, nsam, fc,
                            near_range_s, bounce_rays=15, num_bounces=3, num_rayrounds=1, debug=False):

    r_obs_vec = np.repeat(a_obs_pt.reshape((a_obs_pt.shape[0], 1, a_obs_pt.shape[1])),
                              sample_points.shape[0], axis=1)

    # Generate the pointing vector for the antenna and the antenna pattern
    # that will affect the power received by the intersection point
    pointing_az = cupy.array(np.arctan2(pointing_vec[:, 0], pointing_vec[:, 1]), dtype=np.float32)
    pointing_el = cupy.array(-np.arcsin(pointing_vec[:, 2]), dtype=np.float32)



    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cupy.array(mesh_normals, dtype=np.float32)
    tri_idxes_gpu = cupy.array(mesh_tri_idxes, dtype=np.int32)
    tri_verts_gpu = cupy.array(mesh_vertices, dtype=np.float32)
    tri_sigmas_gpu = cupy.array(mesh_sigmas, dtype=np.float32)
    tri_box_gpu = cupy.array(tri_box_idxes, dtype=bool)
    boxes_gpu = cupy.array(bounding_boxes, dtype=np.float32)
    ray_origin_gpu = cupy.array(r_obs_vec, dtype=np.float32)
    pd_r = cupy.array(np.zeros((a_obs_pt.shape[0], nsam)), dtype=np.float64)
    pd_i = cupy.array(np.zeros((a_obs_pt.shape[0], nsam)), dtype=np.float64)

    for i in range(num_rayrounds):
        # GPU device calculations
        threads_per_block = getMaxThreads(2)

        # Test triangles in correct quads to find intsersection triangles
        inter_xyz = np.zeros_like(r_obs_vec) + np.inf
        inter_sigma = np.zeros((inter_xyz.shape[0], inter_xyz.shape[1])) - 1
        inter_bounce_dir = np.zeros_like(r_obs_vec)

        int_gpu = cupy.array(inter_xyz, dtype=np.float32)
        if i == num_rayrounds - 1:
            ray_init_inter_xyz_gpu = cupy.array(sample_points, np.float32)
        else:
            ray_init_inter_xyz_gpu = cupy.array(sample_points + np.random.normal(0, .5, size=sample_points.shape), np.float32)
        int_sigma_gpu = cupy.array(inter_sigma, dtype=np.float32)
        bounce_gpu = cupy.array(inter_bounce_dir, dtype=np.float32)

        bprun = (max(1, int_gpu.shape[1] // threads_per_block[0] + 1),
                 tri_norm_gpu.shape[0] // threads_per_block[1] + 1, int_gpu.shape[0] // threads_per_block[2] + 1)

        calcInitBounce[bprun, threads_per_block](ray_origin_gpu, ray_init_inter_xyz_gpu, boxes_gpu, tri_box_gpu, tri_verts_gpu,
                                                 tri_idxes_gpu, tri_norm_gpu, tri_sigmas_gpu, int_sigma_gpu, int_gpu,
                                                 bounce_gpu)

        inter_xyz = int_gpu.get()
        inter_bounce_dir = bounce_gpu.get()
        face_sigma = int_sigma_gpu.get()

        valids = face_sigma > 0
        if np.sum(valids) == 0:
            return

        # Remove points that just don't bounce again
        tvals = np.all(valids, axis=0)
        face_sigma = face_sigma[:, tvals]
        inter_xyz = inter_xyz[:, tvals]
        inter_bounce_dir = inter_bounce_dir[:, tvals]
        obs_pt_vec = r_obs_vec[:, tvals]

        rho_o = np.ones(face_sigma.shape) * radar_equation_constant

        # Get the range profile from the intersection-tested rays
        rng = np.linalg.norm(inter_xyz - obs_pt_vec, axis=2)
        bounce_gpu = cupy.array(inter_bounce_dir, dtype=np.float64)
        sigma_gpu = cupy.array(face_sigma, dtype=np.float64)
        int_gpu = cupy.array(inter_xyz, dtype=np.float64)
        ray_dist_gpu = cupy.array(rng, dtype=np.float64)
        ray_power_gpu = cupy.array(rho_o, dtype=np.float64)
        obs_pt_gpu = cupy.array(obs_pt_vec, dtype=np.float64)
        valids_gpu = cupy.array(valids[:, tvals], dtype=bool)

        threads_per_block = getMaxThreads()

        genRangeProfileMesh[
            (max(1, int_gpu.shape[0] // threads_per_block[0] + 1), max(1, int_gpu.shape[1] // threads_per_block[1] + 1)),
            threads_per_block](int_gpu, bounce_gpu, sigma_gpu, obs_pt_gpu, obs_pt_gpu, ray_dist_gpu, ray_power_gpu,
                               valids_gpu, pointing_az, pointing_el, pointing_az, pointing_el, pd_r, pd_i, c0 / fc,
                               near_range_s, fs, bw_az, bw_el)

    if debug:
        debug_rays = [inter_xyz]
        debug_raydirs = [inter_bounce_dir]
        debug_raypower = [rho_o / rng**2]

    if num_bounces > 0:
        # Calc out the origins and directions of the bounces for the bounce loop

        bounce_sigma = np.repeat(face_sigma, bounce_rays, axis=1)
        bounce_origin_gpu = cupy.array(np.repeat(inter_xyz, bounce_rays, axis=1), dtype=np.float32)
        bounce_dirs = np.repeat(inter_bounce_dir, bounce_rays, axis=1)
        bounce_az = np.arctan2(bounce_dirs[:, :, 0], bounce_dirs[:, :, 1]) + np.random.normal(0, bounce_sigma)
        bounce_el = -np.arcsin(bounce_dirs[:, :, 2]) + np.random.normal(0, bounce_sigma)
        bounce_dir_gpu = cupy.array(azelToVec(bounce_az, bounce_el).T.swapaxes(0, 1), dtype=np.float32)
        bounce_power_gpu = cupy.array(np.repeat(rho_o / rng**2, bounce_rays, axis=1), dtype=np.float32)
        bounce_rng_gpu = cupy.array(np.repeat(rng, bounce_rays, axis=1), dtype=np.float32)
        obs_pt_gpu = cupy.array(a_obs_pt, dtype=np.float32)

        bprun = (bounce_origin_gpu.shape[1] // threads_per_block[1] + 1, bounce_origin_gpu.shape[0] // threads_per_block[0] + 1)
        for _ in range(num_bounces):
            calcBounceLoop[bprun, threads_per_block](bounce_origin_gpu, bounce_dir_gpu, bounce_rng_gpu, bounce_power_gpu,
                                                     boxes_gpu, tri_box_gpu, tri_verts_gpu, tri_idxes_gpu, tri_norm_gpu,
                                                     tri_sigmas_gpu, pd_r, pd_i, obs_pt_gpu, pointing_az, pointing_el,
                                                     c0 / fc, near_range_s, fs, bw_az, bw_el)
            if debug:
                debug_rays.append(bounce_origin_gpu.get())
                debug_raydirs.append(bounce_dir_gpu.get())
                debug_raypower.append(bounce_power_gpu.get())

        # Add random rays for additional 'noise'
        '''bounce_sigma = np.repeat(face_sigma, bounce_rays, axis=1)
        bounce_origin_gpu = cupy.array(np.repeat(inter_xyz, bounce_rays, axis=1), dtype=np.float32)
        bounce_dirs = np.repeat(inter_bounce_dir, bounce_rays, axis=1)
        bounce_az = np.arctan2(bounce_dirs[:, :, 0], bounce_dirs[:, :, 1]) + np.random.normal(0, bounce_sigma)
        bounce_el = -np.arcsin(bounce_dirs[:, :, 2]) + np.random.normal(0, bounce_sigma)
        bounce_dir_gpu = cupy.array(azelToVec(bounce_az, bounce_el).T.swapaxes(0, 1), dtype=np.float32)
        bounce_power_gpu = cupy.array(np.repeat(rho_o / rng ** 2, bounce_rays, axis=1), dtype=np.float32)
        bounce_rng_gpu = cupy.array(np.repeat(rng, bounce_rays, axis=1), dtype=np.float32)
        obs_pt_gpu = cupy.array(a_obs_pt, dtype=np.float32)

        bprun = (
        bounce_origin_gpu.shape[1] // threads_per_block[1] + 1, bounce_origin_gpu.shape[0] // threads_per_block[0] + 1)
        for _ in range(num_bounces):
            calcBounceLoop[bprun, threads_per_block](bounce_origin_gpu, bounce_dir_gpu, bounce_rng_gpu,
                                                     bounce_power_gpu,
                                                     boxes_gpu, tri_box_gpu, tri_verts_gpu, tri_idxes_gpu, tri_norm_gpu,
                                                     tri_sigmas_gpu, pd_r, pd_i, obs_pt_gpu, pointing_az, pointing_el,
                                                     c0 / fc, near_range_s, fs, bw_az, bw_el)'''

        del bounce_dir_gpu
        del bounce_origin_gpu
        del bounce_power_gpu
        del bounce_rng_gpu

    pulse_ret = pd_r.get() + 1j * pd_i.get()
    del bounce_gpu
    del sigma_gpu
    del pd_r
    del pd_i
    del int_gpu
    del ray_dist_gpu
    del ray_power_gpu
    del obs_pt_gpu
    del valids_gpu
    del pointing_el
    del pointing_az

    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    if debug:
        return pulse_ret, debug_rays, debug_raydirs, debug_raypower
    else:
        return pulse_ret

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
def calcSingleIntersection(rdx, rdy, rdz, rx, ry, rz, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, vnx, vny, vnz):
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z
    crossx = rdy * e2z - rdz * e2y
    crossy = rdz * e2x - rdx * e2z
    crossz = rdx * e2y - rdy * e2x
    det = e1x * crossx + e1y * crossy + e1z * crossz
    # Check to see if ray is parallel to triangle
    if abs(det) < 1e-9:
        return False, 0, 0, 0, 0, 0, 0

    inv_det = 1. / det
    sx = rx - v0x
    sy = ry - v0y
    sz = rz - v0z

    u = inv_det * (sx * crossx + sy * crossy + sz * crossz)
    if u < 0 or u > 1:
        return False, 0, 0, 0, 0, 0, 0

    # Recompute cross for s and edge 1
    crossx = sy * e1z - sz * e1y
    crossy = sz * e1x - sx * e1z
    crossz = sx * e1y - sy * e1x
    v = inv_det * (rdx * crossx + rdy * crossy + rdz * crossz)
    if v < 0 or u + v > 1:
        return False, 0, 0, 0, 0, 0, 0

    # Compute intersection point
    t = inv_det * (e2x * crossx + e2y * crossy + e2z * crossz)
    if t < 1e-9:
        return False, 0, 0, 0, 0, 0, 0

    intx = rx + t * rdx
    inty = ry + t * rdy
    intz = rz + t * rdz

    # Calculate out the angles in azimuth and elevation for the bounce
    tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)

    bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
    bx = tx - vnx * bounce_dot
    by = ty - vny * bounce_dot
    bz = tz - vnz * bounce_dot
    bounce_len = 1 / math.sqrt(bx * bx + by * by + bz * bz)

    return True, bx * bounce_len, by * bounce_len, bz * bounce_len, intx, inty, intz


@cuda.jit()
def calcInitBounce(ray_origin, init_inter_xyz, bounding_box, tri_box_idx, tri_vert, tri_idx, tri_norm, tri_sigma,
                   inter_sigma, inter_xyz, inter_bounce):
    ray_idx, t_idx, tt = cuda.grid(ndim=3)
    if tri_norm.shape[0] > t_idx and ray_idx < init_inter_xyz.shape[0] and tt < ray_origin.shape[0]:
        # Run for the initial point
        rx = init_inter_xyz[ray_idx, 0] - ray_origin[tt, ray_idx, 0]
        ry = init_inter_xyz[ray_idx, 1] - ray_origin[tt, ray_idx, 1]
        rz = init_inter_xyz[ray_idx, 2] - ray_origin[tt, ray_idx, 2]
        rng = 1 / math.sqrt(rx * rx + ry * ry + rz * rz)
        rx *= rng
        ry *= rng
        rz *= rng
        for box in range(bounding_box.shape[0]):
            if tri_box_idx[t_idx, box]:
                if checkBox(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2], rx, ry,
                            rz, bounding_box[box, 0, 0], bounding_box[box, 0, 1], bounding_box[box, 0, 2],
                            bounding_box[box, 1, 0], bounding_box[box, 1, 1], bounding_box[box, 1, 2]):
                    did_intersect, bx, by, bz, intx, inty, intz = (
                        calcSingleIntersection(rx, ry, rz, ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1],
                                               ray_origin[tt, ray_idx, 2], tri_vert[tri_idx[t_idx, 0], 0],
                                               tri_vert[tri_idx[t_idx, 0], 1], tri_vert[tri_idx[t_idx, 0], 2],
                                               tri_vert[tri_idx[t_idx, 1], 0], tri_vert[tri_idx[t_idx, 1], 1],
                                               tri_vert[tri_idx[t_idx, 1], 2], tri_vert[tri_idx[t_idx, 2], 0],
                                               tri_vert[tri_idx[t_idx, 2], 1], tri_vert[tri_idx[t_idx, 2], 2],
                                               tri_norm[t_idx, 0], tri_norm[t_idx, 1], tri_norm[t_idx, 2]))
                    if did_intersect:
                        if math.sqrt(intx**2 + inty**2 + intz**2) <= math.sqrt(inter_xyz[tt, ray_idx, 0]**2 +
                                                                               inter_xyz[tt, ray_idx, 1]**2 +
                                                                               inter_xyz[tt, ray_idx, 2]**2):
                            inter_sigma[tt, ray_idx] = tri_sigma[t_idx]
                            inter_xyz[tt, ray_idx, 0] = intx
                            inter_xyz[tt, ray_idx, 1] = inty
                            inter_xyz[tt, ray_idx, 2] = intz
                            inter_bounce[tt, ray_idx, 0] = bx
                            inter_bounce[tt, ray_idx, 1] = by
                            inter_bounce[tt, ray_idx, 2] = bz
    cuda.syncthreads()


@cuda.jit(device=True)
def calcReturnAndBin(intx, inty, intz, rex, rey, rez, bx, by, bz, inter_sigma, rng, near_range_s, source_fs, n_samples,
                     pan, tilt, bw_az, bw_el, wavenumber, rho):
    rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(intx, inty, intz, rex,
                                                   rey,
                                                   rez)
    # Calculate return vector and strength
    # Apply Rayleigh scattering with the sigma being the width of the distribution
    x = max(0., (rx * bx + ry * by + rz * bz) / r_rng)
    gamma = 1 - math.exp(-(1 + x)**2 / (2 * inter_sigma**2))

    two_way_rng = rng + r_rng
    rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    but = int(rng_bin)

    if n_samples > but > 0:
        att = applyRadiationPattern(r_el, r_az, pan, tilt, pan, tilt,
                                    bw_az, bw_el) / (
                      two_way_rng * two_way_rng) * rho
        acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * gamma
        return acc_val.real, acc_val.imag, but

    return 0, 0, -1


@cuda.jit()
def calcBounceLoop(ray_origin, ray_dir, ray_distance, ray_power, bounding_box, tri_box_idx, tri_vert, tri_idx,
                   tri_norm, tri_sigma, pd_r, pd_i, receive_xyz, pan, tilt, wavelength, near_range_s, source_fs, bw_az,
                   bw_el):
    ray_idx, tt = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[1]
        wavenumber = 2 * np.pi / wavelength

        rx = ray_origin[tt, ray_idx, 0]
        ry = ray_origin[tt, ray_idx, 1]
        rz = ray_origin[tt, ray_idx, 2]
        rdx = ray_dir[tt, ray_idx, 0]
        rdy = ray_dir[tt, ray_idx, 1]
        rdz = ray_dir[tt, ray_idx, 2]
        rho = ray_power[tt, ray_idx]
        rng = ray_distance[tt, ray_idx]
        int_rng = np.inf
        did_intersect = False
        for box in range(bounding_box.shape[0]):
            boxxmin = bounding_box[box, 0, 0]
            boxymin = bounding_box[box, 0, 1]
            boxzmin = bounding_box[box, 0, 2]
            boxxmax = bounding_box[box, 1, 0]
            boxymax = bounding_box[box, 1, 1]
            boxzmax = bounding_box[box, 1, 2]
            if checkBox(rx, ry, rz, rdx, rdy, rdz, boxxmin, boxymin, boxzmin, boxxmax, boxymax, boxzmax):
                for t_idx in range(tri_box_idx.shape[0]):
                    if tri_box_idx[t_idx, box]:
                        curr_intersect, tbx, tby, tbz, tintx, tinty, tintz = (
                            calcSingleIntersection(rdx, rdy, rdz, rx, ry, rz, tri_vert[tri_idx[t_idx, 0], 0],
                                                   tri_vert[tri_idx[t_idx, 0], 1], tri_vert[tri_idx[t_idx, 0], 2],
                                                   tri_vert[tri_idx[t_idx, 1], 0], tri_vert[tri_idx[t_idx, 1], 1],
                                                   tri_vert[tri_idx[t_idx, 1], 2], tri_vert[tri_idx[t_idx, 2], 0],
                                                   tri_vert[tri_idx[t_idx, 2], 1], tri_vert[tri_idx[t_idx, 2], 2],
                                                   tri_norm[t_idx, 0], tri_norm[t_idx, 1], tri_norm[t_idx, 2]))
                        if curr_intersect:
                            tmp_rng = math.sqrt((rx - tintx)**2 + (ry - tinty)**2 + (rz - tintz)**2)
                            if tmp_rng < int_rng:
                                int_rng = tmp_rng + 0.
                                inter_sigma = tri_sigma[t_idx]
                                delta_phi = tri_norm[t_idx, 0] * tbx + tri_norm[t_idx, 1] * tby + tri_norm[t_idx, 2] * tbz
                                delta_phi = max(0, delta_phi)
                                att = delta_phi / inter_sigma ** 2 * math.exp(-delta_phi ** 2 / (2 * inter_sigma ** 2))
                                nrho = rho * att
                                intx = tintx + 0.
                                inty = tinty + 0.
                                intz = tintz + 0.
                                bx = tbx + 0.
                                by = tby + 0.
                                bz = tbz + 0.
                                did_intersect = True
        if did_intersect:
            rng += int_rng
            curr_intersect = False
            # Check for occlusion against the receiver
            occ_x = intx - receive_xyz[tt, 0]
            occ_y = inty - receive_xyz[tt, 1]
            occ_z = intz - receive_xyz[tt, 2]
            inv_occ = 1 / math.sqrt(occ_x**2 + occ_y**2 + occ_z**2)
            occ_x *= inv_occ
            occ_y *= inv_occ
            occ_z *= inv_occ
            for box in range(bounding_box.shape[0]):
                boxxmin = bounding_box[box, 0, 0]
                boxymin = bounding_box[box, 0, 1]
                boxzmin = bounding_box[box, 0, 2]
                boxxmax = bounding_box[box, 1, 0]
                boxymax = bounding_box[box, 1, 1]
                boxzmax = bounding_box[box, 1, 2]
                if checkBox(intx, inty, intz, occ_x, occ_y, occ_z,
                            boxxmin, boxymin, boxzmin, boxxmax, boxymax, boxzmax):
                    for t_idx in range(tri_box_idx.shape[0]):
                        if tri_box_idx[t_idx, box]:
                            curr_intersect, _, _, _, _, _, _ = (
                                calcSingleIntersection(occ_x, occ_y, occ_z, intx, inty, intz, tri_vert[tri_idx[t_idx, 0], 0],
                                                       tri_vert[tri_idx[t_idx, 0], 1], tri_vert[tri_idx[t_idx, 0], 2],
                                                       tri_vert[tri_idx[t_idx, 1], 0], tri_vert[tri_idx[t_idx, 1], 1],
                                                       tri_vert[tri_idx[t_idx, 1], 2], tri_vert[tri_idx[t_idx, 2], 0],
                                                       tri_vert[tri_idx[t_idx, 2], 1], tri_vert[tri_idx[t_idx, 2], 2],
                                                       tri_norm[t_idx, 0], tri_norm[t_idx, 1], tri_norm[t_idx, 2]))
                            if curr_intersect:
                                break

            if not curr_intersect:
                acc_real, acc_imag, but = calcReturnAndBin(intx, inty, intz, receive_xyz[tt, 0], receive_xyz[tt, 1],
                                                           receive_xyz[tt, 2], bx, by, bz, inter_sigma, rng, near_range_s,
                                                           source_fs, n_samples, pan[tt], tilt[tt], bw_az, bw_el, wavenumber,
                                                           nrho)
                if but >= 0:
                    cuda.atomic.add(pd_r, (tt, but), acc_real)
                    cuda.atomic.add(pd_i, (tt, but), acc_imag)

            ray_origin[tt, ray_idx, 0] = intx
            ray_origin[tt, ray_idx, 1] = inty
            ray_origin[tt, ray_idx, 2] = intz
            ray_dir[tt, ray_idx, 0] = bx
            ray_dir[tt, ray_idx, 1] = by
            ray_dir[tt, ray_idx, 2] = bz
            ray_power[tt, ray_idx] = nrho
            ray_distance[tt, ray_idx] = rng

        cuda.syncthreads()





if __name__ == '__main__':

    mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)
