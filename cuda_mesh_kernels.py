import cmath
import math
from profile import runctx

from cuda_functions import float3, make_float3, cross, dot, length, normalize
from proj_kernels import calcProjectionReturn
from simulation_functions import factors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from numba import cuda, prange
from line_profiler_pycharm import profile
import cupy
from simulation_functions import findPowerOf2, azelToVec
import numpy as np
import open3d as o3d
from cuda_kernels import applyRadiationPattern, applyRadiationPatternCPU, getMaxThreads
import plotly.express as px
from sklearn.cluster import KMeans
import plotly.io as pio
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32, xoroshiro128p_uniform_float32

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
RAYLEIGH_SIGMA_COEFF = .655
CUTOFF = 200000
BOX_CUSHION = .1


@cuda.jit(device=True)
def getRangeAndAngles(v, s):
    t = v - s
    rng = length(t)
    az = math.atan2(t.x, t.y)
    el = -math.asin(t.z / rng)
    return t, rng, az, el


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
        for box in prange(tri_poss.shape[1]):
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


def readCombineMeshFile(fnme: str, points: int=100000, scale: float=None) -> o3d.geometry.TriangleMesh:
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

def genOctree(bounding_box: np.ndarray, points: np.ndarray, num_levels: int = 3):
    octree = np.zeros((sum(8**n for n in range(num_levels)), 2, 3))
    # nest_list = [[0]]
    octree[0, ...] = bounding_box
    for level in range(1, num_levels):
        init_idx = sum(int(8**(n - 1)) for n in range(level))
        # nest_list.append([])
        for parent in range(init_idx, init_idx + int(8**(level - 1))):
            box_extent = np.diff(octree[parent], axis=0)[0]
            box_min = octree[parent, 0]
            box_max = octree[parent, 1]
            child_idx = sum(8**n for n in range(level)) + (parent - init_idx) * 8
            npoints = points[np.logical_and(np.logical_and(points[:, 0] > box_min[0],
                                                           points[:, 0] < octree[parent, 1, 0]),
                                            np.logical_and(points[:, 1] > box_min[1],
                                                           points[:, 1] < octree[parent, 1, 1]),
                                            np.logical_and(points[:, 2] > box_min[2], points[:, 2] < octree[parent, 1, 2]))]
            if len(npoints) == 0:
                continue
            box_div = npoints.mean(axis=0)
            box_hull = np.array([box_min, box_div, box_max])
            bidx = 0
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        '''octree[child_idx + bidx, :] = np.array([[box_min[0] + box_extent[0] * x / 2.,
                                                                         box_min[1] + box_extent[1] * y / 2.,
                                                                         box_min[2] + box_extent[2] * z / 2.],
                                                                        [box_min[0] + box_extent[0] * (x + 1) / 2.,
                                                                         box_min[1] + box_extent[1] * (y + 1) / 2.,
                                                                         box_min[2] + box_extent[2] * (z + 1) / 2.]
                                                                        ])'''
                        octree[child_idx + bidx, :] = np.array([[box_hull[x, 0], box_hull[y, 1], box_hull[z, 2]],
                                                                [box_hull[x + 1, 0], box_hull[y + 1, 1], box_hull[z + 1, 2]]])
                        # nest_list[-1].append(child_idx + bidx)
                        bidx += 1
    '''box = 1
    prev = 0
    final_level = octree.shape[0] // 8
    while True:
        if (box - 1) // 8 != prev:
            if box == 9:
                break
            box = (box - 1) // 8
            prev = (box - 1) // 8
        print(f' {box} ', end='')
        if np.random.rand() > .5:
            if box >= final_level:
                pass
            else:
                prev = box + 0
                box = box * 8
        box += 1'''

    return octree

def getBoxesSamplesFromMesh(a_mesh: o3d.geometry.TriangleMesh, num_box_levels: int=4, sample_points: int=10000,
                            material_sigmas: list=None, material_kd: list=None, material_ks: list = None):
    # Generate bounding box tree
    mesh_tri_idx = np.asarray(a_mesh.triangles)
    mesh_vertices = np.asarray(a_mesh.vertices)
    mesh_normals = np.asarray(a_mesh.triangle_normals)
    mesh_tri_vertices = mesh_vertices[mesh_tri_idx]

    aabb = a_mesh.get_axis_aligned_bounding_box()
    max_bound = aabb.get_max_bound()
    min_bound = aabb.get_min_bound()
    root_box = np.array([min_bound, max_bound])
    boxes = genOctree(root_box, mesh_vertices, num_box_levels)

    mesh_box_idx = np.zeros((mesh_tri_idx.shape[0], len(boxes))).astype(bool)
    # Get the box index for each triangle, for exclusion in the GPU calculations
    for b_idx, box in enumerate(boxes):
        is_inside = (box[0, 0] < mesh_tri_vertices[:, :, 0]) & (mesh_tri_vertices[:, :, 0] < box[1, 0])
        for dim in range(1, 3):
            is_inside = is_inside & (box[0, dim] < mesh_tri_vertices[:, :, dim]) & (mesh_tri_vertices[:, :, dim] < box[1, dim])
        mesh_box_idx[:, b_idx] = np.any(is_inside, axis=1)

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

    if material_kd is None:
        mesh_kd = np.ones(len(a_mesh.triangles))
    else:
        mesh_kd = np.array([material_kd[i] for i in np.asarray(a_mesh.triangle_material_ids)])

    if material_ks is None:
        mesh_ks = np.ones(len(a_mesh.triangles))
    else:
        mesh_ks = np.array([material_ks[i] for i in np.asarray(a_mesh.triangle_material_ids)])

    # Sample the mesh to get points for initial raycasting
    points = a_mesh.sample_points_poisson_disk(sample_points)
    return (boxes, mesh_box_idx, mesh_tri_idx, mesh_vertices, mesh_normals, mesh_sigmas, mesh_kd, mesh_ks), np.asarray(points.points)

@profile
def getRangeProfileFromMesh(bounding_boxes: np.ndarray, tri_box_idxes: np.ndarray, mesh_tri_idxes: np.ndarray,
                            mesh_vertices: np.ndarray, mesh_normals: np.ndarray, mesh_sigmas: np.ndarray,
                            mesh_kd: np.ndarray, mesh_ks: np.ndarray, sample_points: np.ndarray, a_obs_pt: np.ndarray,
                            pointing_vec: np.ndarray,
                            radar_equation_constant: float, bw_az: float, bw_el: float, nsam: int, fc: float,
                            near_range_s: float, bounce_rays: int=15, num_bounces: int=3, debug: bool=False)\
        -> tuple[np.ndarray, list, list, list] | np.ndarray:

    # Generate the pointing vector for the antenna and the antenna pattern
    # that will affect the power received by the intersection point
    pointing_az = cupy.array(np.arctan2(pointing_vec[:, 0], pointing_vec[:, 1]), dtype=np.float32)
    pointing_el = cupy.array(-np.arcsin(pointing_vec[:, 2]), dtype=np.float32)

    debug_rays = []
    debug_raydirs = []
    debug_raypower = []

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bprun = (max(1, a_obs_pt.shape[0] // threads_per_block[0] + 1),
             sample_points.shape[0] // threads_per_block[1] + 1)

    ray_origins = np.repeat(a_obs_pt.reshape((a_obs_pt.shape[0], 1, a_obs_pt.shape[1])), sample_points.shape[0], axis=1)
    ray_dirs = sample_points[None, :, :] - ray_origins
    ray_dirs /= np.linalg.norm(ray_dirs, axis=2)[:, :, None]

    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cupy.array(mesh_normals, dtype=np.float32)
    tri_idxes_gpu = cupy.array(mesh_tri_idxes, dtype=np.int32)
    tri_verts_gpu = cupy.array(mesh_vertices, dtype=np.float32)
    tri_material_gpu = cupy.array(np.concatenate([mesh_sigmas.reshape((-1, 1)),
                                                  mesh_kd.reshape((-1, 1)), mesh_ks.reshape((-1, 1))], axis=1),
                                  dtype=np.float32)
    tri_box_gpu = cupy.array(tri_box_idxes, dtype=bool)
    boxes_gpu = cupy.array(bounding_boxes, dtype=np.float32)
    receive_xyz_gpu = cupy.array(a_obs_pt, dtype=np.float32)
    ray_origin_gpu = cupy.array(ray_origins, dtype=np.float32)
    ray_dir_gpu = cupy.array(ray_dirs, dtype=np.float32)
    ray_power_gpu = cupy.array(np.ones((a_obs_pt.shape[0], sample_points.shape[0])) * radar_equation_constant,
                               dtype=np.float32)
    ray_distance_gpu = cupy.zeros_like(ray_power_gpu)
    pd_r = cupy.array(np.zeros((a_obs_pt.shape[0], nsam)), dtype=np.float64)
    pd_i = cupy.array(np.zeros((a_obs_pt.shape[0], nsam)), dtype=np.float64)

    np.random.seed(12)
    rng_state = cupy.array(np.random.normal(0, .03, (sample_points.shape[0], bounce_rays)), dtype=np.float32)

    for b in range(num_bounces):

        # Test triangles in correct quads to find intersection triangles
        calcBounceLoop[bprun, threads_per_block](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu, ray_power_gpu,
                                                 boxes_gpu, tri_box_gpu, tri_verts_gpu, tri_idxes_gpu, tri_norm_gpu,
                                                 tri_material_gpu, pd_r, pd_i, receive_xyz_gpu, rng_state, bounce_rays,
                                                 pointing_az, pointing_el, c0 / fc, near_range_s, fs, bw_az, bw_el, b == 0)

        if debug:
            debug_rays.append(ray_origin_gpu.get())
            debug_raydirs.append(ray_dir_gpu.get())
            debug_raypower.append(ray_power_gpu.get())

    pulse_ret = pd_r.get() + 1j * pd_i.get()
    del pd_r
    del pd_i
    del boxes_gpu
    del tri_norm_gpu
    del tri_box_gpu
    del tri_idxes_gpu
    del tri_material_gpu
    del tri_verts_gpu

    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    if debug:
        return pulse_ret, debug_rays, debug_raydirs, debug_raypower
    else:
        return pulse_ret

@cuda.jit(device=True, fast_math=True)
def checkBox(ro, ray, boxmin, boxmax):
    tmin = -np.inf
    tmax = np.inf
    tx1 = (boxmin.x - ro.x) / ray.x
    tx2 = (boxmax.x - ro.x) / ray.x
    tmin = max(tmin, min(tx1, tx2))
    tmax = min(tmax, max(tx1, tx2))
    tx1 = (boxmin.y - ro.y) / ray.y
    tx2 = (boxmax.y - ro.y) / ray.y
    tmin = max(tmin, min(tx1, tx2))
    tmax = min(tmax, max(tx1, tx2))
    tx1 = (boxmin.z - ro.z) / ray.z
    tx2 = (boxmax.z - ro.z) / ray.z
    tmin = max(tmin, min(tx1, tx2))
    tmax = min(tmax, max(tx1, tx2))
    return tmax - tmin >= 0 and tmax >= 0


@cuda.jit(device=True, fast_math=True)
def calcSingleIntersection(rd, ro, v0, v1, v2, vn, get_bounce):
    e1 = v1 - v0
    e2 = v2 - v0
    rcrosse = cross(rd, e2)
    det = dot(e1, rcrosse)
    # Check to see if ray is parallel to triangle
    if abs(det) < 1e-9:
        return False, None, None

    inv_det = 1. / det
    s = ro - v0

    u = inv_det * dot(s, rcrosse)
    if u < 0 or u > 1:
        return False, None, None

    # Recompute cross for s and edge 1
    rcrosse = cross(s, e1)
    v = inv_det * dot(rd, rcrosse)
    if v < 0 or u + v > 1:
        return False, None, None

    # Compute intersection point
    t = inv_det * dot(e2, rcrosse)
    if t < 1e-9:
        return False, None, None

    if not get_bounce:
        return True, None, None

    inter = ro + t * rd

    # Calculate out the angles in azimuth and elevation for the bounce
    t, vrng, _, _ = getRangeAndAngles(inter, ro)

    bx = t - vn * dot(t, vn) * 2.

    return True, normalize(bx), inter


@cuda.jit(device=True, fast_math=True)
def calcReturnAndBin(inter, re, rng, near_range_s, source_fs, n_samples,
                     pan, tilt, bw_az, bw_el, wavenumber, rho):
    r, r_rng, r_az, r_el = getRangeAndAngles(inter, re)

    two_way_rng = rng + r_rng
    rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    but = int(rng_bin)

    if n_samples > but > 0:
        att = applyRadiationPattern(r_el, r_az, pan, tilt, pan, tilt,
                                    bw_az, bw_el) / (
                      two_way_rng * two_way_rng) * rho
        acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng)
        return acc_val.real, acc_val.imag, but

    return 0, 0, -1


@cuda.jit(device=True)
def findOctreeBox(ro, rd, bounding_box, box):
    boxmin = make_float3(bounding_box[box, 0, 0], bounding_box[box, 0, 1], bounding_box[box, 0, 2])
    boxmax = make_float3(bounding_box[box, 1, 0], bounding_box[box, 1, 1], bounding_box[box, 1, 2])
    if boxmin.x == boxmax.x:
        return False
    if checkBox(ro, rd, boxmin, boxmax):
        return True
    return False


@cuda.jit()
def calcBounceLoop(ray_origin, ray_dir, ray_distance, ray_power, bounding_box, tri_box_idx, tri_vert, tri_idx,
                   tri_norm, tri_material, pd_r, pd_i, receive_xyz, rng_state, bounce_rays, pan, tilt, wavelength,
                   near_range_s, source_fs, bw_az, bw_el, is_origin):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[1]
        wavenumber = 2 * np.pi / wavelength
        nrho = -1
        final_level = bounding_box.shape[0] // 8

        ro = make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2])
        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        rdi = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
        rho = ray_power[tt, ray_idx]
        rng = ray_distance[tt, ray_idx]
        for b_idx in prange(bounce_rays):
            int_rng = np.inf
            did_intersect = False
            if b_idx == 0:
                rd = rdi + 0.
            else:
                rd = rdi + rng_state[ray_idx, b_idx]
            if not findOctreeBox(ro, rd, bounding_box, 0):
                continue
            box = 1
            prev = 0
            it = 0
            while (0 < box <= bounding_box.shape[0]) and it < bounding_box.shape[0]:
                if (box - 1) // 8 != prev:
                    if box == 9:
                        break
                    box = (box - 1) // 8
                    prev = (box - 1) // 8
                if findOctreeBox(ro, rd, bounding_box, box):
                    if box >= final_level:
                        for t_idx in prange(tri_box_idx.shape[0]):
                            if tri_box_idx[t_idx, box]:
                                t0 = make_float3(tri_vert[tri_idx[t_idx, 0], 0], tri_vert[tri_idx[t_idx, 0], 1],
                                                 tri_vert[tri_idx[t_idx, 0], 2])
                                t1 = make_float3(tri_vert[tri_idx[t_idx, 1], 0], tri_vert[tri_idx[t_idx, 1], 1],
                                                 tri_vert[tri_idx[t_idx, 1], 2])
                                t2 = make_float3(tri_vert[tri_idx[t_idx, 2], 0], tri_vert[tri_idx[t_idx, 2], 1],
                                                 tri_vert[tri_idx[t_idx, 2], 2])
                                tn = make_float3(tri_norm[t_idx, 0], tri_norm[t_idx, 1], tri_norm[t_idx, 2])
                                curr_intersect, tb, tinter = (
                                    calcSingleIntersection(rd, ro, t0, t1, t2, tn, True))
                                if curr_intersect:
                                    tmp_rng = length(ro - tinter)
                                    if tmp_rng < int_rng:
                                        int_rng = tmp_rng + 0.
                                        inv_rng = 1 / tmp_rng
                                        b = tb + 0.
                                        # This is the phong reflection model to get nrho
                                        tdotn = dot(ro - tinter, tn) * inv_rng
                                        reflection = dot(b, ro) * inv_rng
                                        nrho = 1e-6 + (tri_material[t_idx, 1] * max(tdotn, 0) * rho / 100. +
                                                tri_material[t_idx, 2] * max(reflection, 0) ** tri_material[t_idx, 0] * rho)
                                        inter = tinter + 0.
                                        did_intersect = True
                    else:
                        prev = box + 0
                        box = box * 8
                box += 1
                it += 1
            if did_intersect:
                rng += int_rng
                curr_intersect = False
                if not is_origin:
                    # Check for occlusion against the receiver
                    occ = normalize(inter - rec_xyz)
                    box = 1
                    prev = 0
                    it = 0
                    while (0 < box <= bounding_box.shape[0]) and it < bounding_box.shape[0]:
                        if (box - 1) // 8 != prev:
                            if box == 9:
                                break
                            box = (box - 1) // 8
                            prev = (box - 1) // 8
                        if findOctreeBox(inter, occ, bounding_box, box):
                            if box >= final_level:
                                for t_idx in prange(tri_box_idx.shape[0]):
                                    if tri_box_idx[t_idx, box]:
                                        t0 = make_float3(tri_vert[tri_idx[t_idx, 0], 0], tri_vert[tri_idx[t_idx, 0], 1],
                                                         tri_vert[tri_idx[t_idx, 0], 2])
                                        t1 = make_float3(tri_vert[tri_idx[t_idx, 1], 0], tri_vert[tri_idx[t_idx, 1], 1],
                                                         tri_vert[tri_idx[t_idx, 1], 2])
                                        t2 = make_float3(tri_vert[tri_idx[t_idx, 2], 0], tri_vert[tri_idx[t_idx, 2], 1],
                                                         tri_vert[tri_idx[t_idx, 2], 2])
                                        tn = make_float3(tri_norm[t_idx, 0], tri_norm[t_idx, 1], tri_norm[t_idx, 2])
                                        curr_intersect, _, _ = (
                                            calcSingleIntersection(occ, inter, t0, t1, t2, tn, False))
                                        if curr_intersect:
                                            break
                            else:
                                prev = box + 0
                                box = box * 8
                        box += 1
                        it += 1

                if not curr_intersect:
                    acc_real, acc_imag, but = calcReturnAndBin(inter, rec_xyz, rng, near_range_s,
                                                               source_fs, n_samples, pan[tt], tilt[tt], bw_az, bw_el, wavenumber,
                                                               nrho)
                    if but >= 0:
                        cuda.atomic.add(pd_r, (tt, but), acc_real)
                        cuda.atomic.add(pd_i, (tt, but), acc_imag)

                if b_idx == 0:
                    ray_origin[tt, ray_idx, 0] = inter.x
                    ray_origin[tt, ray_idx, 1] = inter.y
                    ray_origin[tt, ray_idx, 2] = inter.z
                    ray_dir[tt, ray_idx, 0] = b.x
                    ray_dir[tt, ray_idx, 1] = b.y
                    ray_dir[tt, ray_idx, 2] = b.z
                    ray_power[tt, ray_idx] = nrho
                    ray_distance[tt, ray_idx] = rng

        cuda.syncthreads()


def makemat(az, el):
    Rzx = np.array([[np.cos(az), -np.cos(el) * np.sin(az), np.sin(el) * np.sin(az)],
                    [np.sin(az), np.cos(el) * np.cos(az), -np.sin(el) * np.cos(az)],
                    [0, np.sin(el), np.cos(el)]])
    return Rzx





if __name__ == '__main__':

    mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)
