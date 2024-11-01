import cmath
import math
from profile import runctx

from scipy.optimize import minimize

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
from cuda_kernels import applyOneWayRadiationPattern, getMaxThreads
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
    octree[0, ...] = bounding_box
    for level in range(1, num_levels):
        init_idx = sum(int(8**(n - 1)) for n in range(level))
        for parent in range(init_idx, init_idx + int(8**(level - 1))):
            box_min = octree[parent, 0]
            # box_max = octree[parent, 1]
            child_idx = sum(8**n for n in range(level)) + (parent - init_idx) * 8
            npoints = points[np.logical_and(np.logical_and(points[:, 0] >= box_min[0],
                                                           points[:, 0] <= octree[parent, 1, 0]),
                                            np.logical_and(points[:, 1] >= box_min[1],
                                                           points[:, 1] <= octree[parent, 1, 1]),
                                            np.logical_and(points[:, 2] >= box_min[2], points[:, 2] < octree[parent, 1, 2]))]
            if len(npoints) == 0:
                continue

            box_div = npoints.mean(axis=0)
            box_hull = np.array([npoints.min(axis=0), box_div, npoints.max(axis=0)])
            bidx = 0
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        # Recalculate to remove any dead space inside the box
                        npo = npoints[np.logical_and(np.logical_and(npoints[:, 0] >= box_hull[x, 0],
                                                                       npoints[:, 0] <= box_hull[x + 1, 0]),
                                                        np.logical_and(npoints[:, 1] >= box_hull[y, 1],
                                                                       npoints[:, 1] <= box_hull[y + 1, 1]),
                                                        np.logical_and(npoints[:, 2] >= box_hull[z, 2],
                                                                       npoints[:, 2] <= box_hull[z + 1, 2]))]
                        if len(npo) == 0:
                            bidx += 1
                            continue
                        box_pt = np.array([npo.min(axis=0), npo.max(axis=0)])
                        octree[child_idx + bidx, :] = np.array([[box_pt[0] - BOX_CUSHION],
                                                                [box_pt[1] + BOX_CUSHION]]).reshape((2, 3))
                        # octree[child_idx + bidx, :] = np.array([[box_hull[x, 0] - BOX_CUSHION, box_hull[y, 1] - BOX_CUSHION, box_hull[z, 2] - BOX_CUSHION],
                        #                                         [box_hull[x + 1, 0] + BOX_CUSHION, box_hull[y + 1, 1] + BOX_CUSHION, box_hull[z + 1, 2] + BOX_CUSHION]])
                        bidx += 1

    return octree

def getBoxesSamplesFromMesh(a_mesh: o3d.geometry.TriangleMesh, num_box_levels: int=4, sample_points: int=10000,
                            material_sigmas: list=None, material_kd: list=None, material_ks: list = None, view_pos: np.ndarray = None):
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
    meshx, meshy = np.where(mesh_box_idx)
    alltri_idxes = np.array([[a, b] for a, b in zip(meshy, meshx)])
    sorted_tri_idx = alltri_idxes[np.argsort(alltri_idxes[:, 0])]
    sorted_tri_idx = sorted_tri_idx[sorted_tri_idx[:, 0] >= sum(8**n for n in range(num_box_levels - 1))]
    box_num, start_idxes = np.unique(sorted_tri_idx[:, 0], return_index=True)
    mesh_extent = np.diff(start_idxes, append=[sorted_tri_idx.shape[0]])
    mesh_idx_key = np.zeros((boxes.shape[0], 2)).astype(int)
    mesh_idx_key[box_num, 0] = start_idxes
    mesh_idx_key[box_num, 1] = mesh_extent

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

    # points = np.sum(mesh_tri_vertices / 3., axis=1)
    if view_pos is None:
        points = np.asarray(a_mesh.sample_points_poisson_disk(sample_points).points)
    else:
        mpoint = mesh_vertices - view_pos
        azes = np.arctan2(mpoint[:, 0], mpoint[:, 1])
        eles = -np.arcsin(mpoint[:, 2] / np.linalg.norm(mpoint, axis=1))
        points = azelToVec(np.random.uniform(azes.min(), azes.max(), sample_points), np.random.uniform(eles.min(), eles.max(), sample_points)).T
    tri_material = np.concatenate([mesh_sigmas.reshape((-1, 1)),
                                                  mesh_kd.reshape((-1, 1)), mesh_ks.reshape((-1, 1))], axis=1)
    return (boxes, sorted_tri_idx[:, 1], mesh_idx_key, mesh_tri_idx, mesh_vertices, mesh_normals, tri_material), points

@profile
def getRangeProfileFromMesh(boxes, tri_box, tri_box_key, tri_idxes, tri_verts, tri_norm,
                                                 tri_material, sample_points: np.ndarray, a_obs_pt: np.ndarray,
                            pointing_vec: np.ndarray,
                            radar_equation_constant: float, bw_az: float, bw_el: float, nsam: int, fc: float,
                            near_range_s: float, bounce_rays: int=15, num_bounces: int=3, debug: bool=False)\
        -> tuple[np.ndarray, list, list, list] | np.ndarray:

    # Generate the pointing vector for the antenna and the antenna pattern
    # that will affect the power received by the intersection point
    paz = np.arctan2(pointing_vec[:, 0], pointing_vec[:, 1])
    pel = -np.arcsin(pointing_vec[:, 2])
    pointing_az = cupy.array(paz, dtype=np.float64)
    pointing_el = cupy.array(pel, dtype=np.float64)

    vert_look = tri_verts - a_obs_pt.mean(axis=0)
    vert_az = np.arctan2(vert_look[:, 0], vert_look[:, 1])
    vert_el = -np.arcsin(vert_look[:, 2] / np.linalg.norm(vert_look, axis=1))

    debug_rays = []
    debug_raydirs = []
    debug_raypower = []

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bprun = (max(1, a_obs_pt.shape[0] // threads_per_block[0] + 1),
             sample_points.shape[0] // threads_per_block[1] + 1)

    ray_origins = np.repeat(a_obs_pt.reshape((a_obs_pt.shape[0], 1, a_obs_pt.shape[1])), sample_points.shape[0], axis=1)
    # ray_dirs = azelToVec(np.random.uniform(vert_az.min(), vert_az.max(), (ray_origins.shape[0], ray_origins.shape[1])),
    #       np.random.uniform(vert_el.min(), vert_el.max(), (ray_origins.shape[0], ray_origins.shape[1]))).T.swapaxes(0, 1)
    ray_dirs = sample_points[None, :, :] - ray_origins
    ray_dirs /= np.linalg.norm(ray_dirs, axis=2)[:, :, None]

    # Calculate out the angles and use a sinc for the beampattern attenuation
    ray_att = (np.sinc((paz[:, None] - np.arctan2(ray_dirs[:, :, 0], ray_dirs[:, :, 1])) / bw_az)**2 *
               np.sinc((pel[:, None] + np.arcsin(ray_dirs[:, :, 2])) / bw_el)**2)
    ray_power = radar_equation_constant * ray_att

    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cupy.array(tri_norm, dtype=np.float64)
    tri_idxes_gpu = cupy.array(tri_idxes, dtype=np.int32)
    tri_verts_gpu = cupy.array(tri_verts, dtype=np.float64)
    tri_material_gpu = cupy.array(tri_material, dtype=np.float64)
    tri_box_gpu = cupy.array(tri_box, dtype=np.int32)
    tri_box_key_gpu = cupy.array(tri_box_key, dtype=np.int32)
    boxes_gpu = cupy.array(boxes, dtype=np.float64)

    receive_xyz_gpu = cupy.array(a_obs_pt, dtype=np.float64)
    ray_origin_gpu = cupy.array(ray_origins, dtype=np.float64)
    ray_dir_gpu = cupy.array(ray_dirs, dtype=np.float64)
    ray_power_gpu = cupy.array(ray_power, dtype=np.float64)
    ray_distance_gpu = cupy.zeros_like(ray_power_gpu)
    pd_r = cupy.array(np.zeros((a_obs_pt.shape[0], nsam)), dtype=np.float64)
    pd_i = cupy.array(np.zeros((a_obs_pt.shape[0], nsam)), dtype=np.float64)

    np.random.seed(12)
    rng_state = cupy.array(np.random.normal(0, .03, (sample_points.shape[0], bounce_rays, 3)),
                           dtype=np.float64) if bounce_rays > 1 else cupy.array([[[0, 0, 0.]]],
                           dtype=np.float64)

    for b in range(num_bounces):

        # Test triangles in correct quads to find intersection triangles
        calcBounceLoop[bprun, threads_per_block](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu, ray_power_gpu,
                                                 boxes_gpu, tri_box_gpu, tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu, tri_norm_gpu,
                                                 tri_material_gpu, pd_r, pd_i, receive_xyz_gpu, rng_state, bounce_rays,
                                                 pointing_az, pointing_el, c0 / fc, near_range_s, fs, bw_az, bw_el, b == 0)

        if debug:
            debug_rays.append(ray_origin_gpu.get())
            debug_raydirs.append(ray_dir_gpu.get())
            debug_raypower.append(ray_power_gpu.get())

    pulse_ret = pd_r.get() + 1j * pd_i.get()
    del pd_r
    del pd_i
    del ray_distance_gpu
    del ray_power_gpu
    del ray_origin_gpu
    del receive_xyz_gpu
    del pointing_az
    del pointing_el
    del tri_material_gpu
    del tri_box_key_gpu
    del tri_idxes_gpu
    del tri_verts_gpu
    del tri_box_gpu
    del boxes_gpu
    del tri_norm_gpu

    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    if debug:
        return pulse_ret, debug_rays, debug_raydirs, debug_raypower
    else:
        return pulse_ret

@cuda.jit(device=True)
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


@cuda.jit(device=True)
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


@cuda.jit(device=True)
def calcReturnAndBin(inter, re, rng, near_range_s, source_fs, n_samples,
                     pan, tilt, bw_az, bw_el, wavenumber, rho):
    r, r_rng, r_az, r_el = getRangeAndAngles(inter, re)

    two_way_rng = rng + r_rng
    rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    but = int(rng_bin)

    if n_samples > but > 0:
        att = applyOneWayRadiationPattern(r_el, r_az, pan, tilt, bw_az, bw_el) / r_rng**2 * rho
        if abs(att) == np.inf:
            print(r_el)
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


@cuda.jit(device=True)
def traverseOctreeAndIntersection(ro, rd, bounding_box, rho, final_level, tri_box_idx, tri_box_key, tri_idx,
                                  tri_vert, tri_norm, tri_material, occlusion_only):
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
    prev = 0
    it = 0
    closest_box_len = np.inf
    while (0 < box <= bounding_box.shape[0]) and it < bounding_box.shape[0]:
        if (box - 1) // 8 != prev:
            if box == 9:
                break
            box = (box - 1) // 8
            prev = (box - 1) // 8
        box_rng = length(
            ro - make_float3(bounding_box[box, 0, 0], bounding_box[box, 0, 1], bounding_box[box, 0, 2]))
        if box_rng <= closest_box_len:
            if findOctreeBox(ro, rd, bounding_box, box):
                if box >= final_level:
                    tri_min = tri_box_key[box, 0]
                    for t_idx in prange(tri_min, tri_min + tri_box_key[box, 1]):
                        ti = tri_box_idx[t_idx]
                        t0 = make_float3(tri_vert[tri_idx[ti, 0], 0], tri_vert[tri_idx[ti, 0], 1],
                                         tri_vert[tri_idx[ti, 0], 2])
                        t1 = make_float3(tri_vert[tri_idx[ti, 1], 0], tri_vert[tri_idx[ti, 1], 1],
                                         tri_vert[tri_idx[ti, 1], 2])
                        t2 = make_float3(tri_vert[tri_idx[ti, 2], 0], tri_vert[tri_idx[ti, 2], 1],
                                         tri_vert[tri_idx[ti, 2], 2])
                        tn = make_float3(tri_norm[ti, 0], tri_norm[ti, 1], tri_norm[ti, 2])
                        curr_intersect, tb, tinter = (
                            calcSingleIntersection(rd, ro, t0, t1, t2, tn, True))
                        if curr_intersect:
                            if occlusion_only:
                                return True, None, None, None, None
                            tmp_rng = length(ro - tinter)
                            if 1. < tmp_rng < int_rng:
                                int_rng = tmp_rng + 0.
                                inv_rng = 1 / tmp_rng
                                b = tb + 0.
                                # This is the phong reflection model to get nrho
                                tdotn = dot(ro - tinter, tn) * inv_rng
                                reflection = dot(b, ro) * inv_rng
                                nrho = min(1e-6 + (tri_material[ti, 1] * abs(tdotn) * rho / 100. +
                                               tri_material[ti, 2] * abs(reflection) ** tri_material[
                                                   ti, 0] * rho), rho) * inv_rng**2
                                inter = tinter + 0.
                                did_intersect = True
                                closest_box_len = box_rng
                else:
                    prev = box + 0
                    box = box * 8
        box += 1
        it += 1
    return did_intersect, nrho, inter, int_rng, b


@cuda.jit()
def calcBounceLoop(ray_origin, ray_dir, ray_distance, ray_power, bounding_box, tri_box_idx, tri_box_key, tri_vert, tri_idx,
                   tri_norm, tri_material, pd_r, pd_i, receive_xyz, rng_state, bounce_rays, pan, tilt, wavelength,
                   near_range_s, source_fs, bw_az, bw_el, is_origin):
    tt, ray_idx = cuda.grid(ndim=2)
    if ray_idx < ray_dir.shape[1] and tt < ray_dir.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[1]
        wavenumber = 2 * np.pi / wavelength
        final_level = bounding_box.shape[0] // 8

        ro = make_float3(ray_origin[tt, ray_idx, 0], ray_origin[tt, ray_idx, 1], ray_origin[tt, ray_idx, 2])
        rec_xyz = make_float3(receive_xyz[tt, 0], receive_xyz[tt, 1], receive_xyz[tt, 2])
        rdi = make_float3(ray_dir[tt, ray_idx, 0], ray_dir[tt, ray_idx, 1], ray_dir[tt, ray_idx, 2])
        rho = ray_power[tt, ray_idx]
        if rho < 1e-9:
            return
        rng = ray_distance[tt, ray_idx]
        for b_idx in prange(bounce_rays):
            if b_idx == 0:
                rd = rdi + 0.
            else:
                rd = rdi + make_float3(rng_state[ray_idx, b_idx, 0], rng_state[ray_idx, b_idx, 1],
                                       rng_state[ray_idx, b_idx, 2])
            did_intersect, nrho, inter, int_rng, b = traverseOctreeAndIntersection(ro, rd, bounding_box, rho,
                                                                                   final_level, tri_box_idx,
                                                                                   tri_box_key, tri_idx, tri_vert,
                                                                                   tri_norm, tri_material, False)
            if did_intersect:
                rng += int_rng
                if not is_origin:
                    # Check for occlusion against the receiver
                    occ = normalize(inter - rec_xyz)
                    curr_intersect, _, _, _, _ = traverseOctreeAndIntersection(inter, occ, bounding_box, rho,
                                                                               final_level, tri_box_idx, tri_box_key,
                                                                               tri_idx, tri_vert, tri_norm,
                                                                               tri_material, True)

                if not curr_intersect:
                    acc_real, acc_imag, but = calcReturnAndBin(inter, rec_xyz, rng, near_range_s, source_fs, n_samples,
                                                               pan[tt], tilt[tt], bw_az, bw_el, wavenumber, nrho)
                    if but >= 0:
                        acc_real = acc_real if abs(acc_real) < np.inf else 0.
                        acc_imag = acc_imag if abs(acc_imag) < np.inf else 0.
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
            else:
                ray_power[tt, ray_idx] = 0.

        cuda.syncthreads()


def makemat(az, el):
    Rzx = np.array([[np.cos(az), -np.cos(el) * np.sin(az), np.sin(el) * np.sin(az)],
                    [np.sin(az), np.cos(el) * np.cos(az), -np.sin(el) * np.cos(az)],
                    [0, np.sin(el), np.cos(el)]])
    return Rzx





if __name__ == '__main__':

    mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)
    def minfunc(x):
        box_hull = np.array([npoints.min(axis=0), x, npoints.max(axis=0)])
        bidx = 0
        stds = np.zeros(3)
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    # Recalculate to remove any dead space inside the box
                    npoints = points[np.logical_and(np.logical_and(points[:, 0] > box_hull[x, 0],
                                                                   points[:, 0] < box_hull[x + 1, 0]),
                                                    np.logical_and(points[:, 1] > box_hull[y, 1],
                                                                   points[:, 1] < box_hull[y + 1, 1]),
                                                    np.logical_and(points[:, 2] > box_hull[z, 2],
                                                                   points[:, 2] < box_hull[z + 1, 2]))]
                    if len(npoints) == 0:
                        bidx += 1
                        continue
                    stds += npoints.std(axis=0)
        return sum(stds)


    box_hull = np.array([npoints.min(axis=0), npoints.max(axis=0)])
    opt_x = minimize(minfunc, npoints.mean(axis=0), bounds=[(box_hull[0, 0], box_hull[1, 0]), (box_hull[0, 1], box_hull[1, 1]), (box_hull[0, 2], box_hull[1, 2])])


