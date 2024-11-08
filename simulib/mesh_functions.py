import itertools
from numba import cuda
from .cuda_mesh_kernels import calcBounceLoop, calcBounceInit, calcOriginDirAtt, calcIntersectionPoints
from .simulation_functions import azelToVec
import numpy as np
import open3d as o3d
from .cuda_kernels import getMaxThreads
import nvtx

c0 = 299792458.0
fs = 2e9
BOX_CUSHION = .1


def readCombineMeshFile(fnme: str, points: int=100000, scale: float=None) -> o3d.geometry.TriangleMesh:
    full_mesh = o3d.io.read_triangle_model(fnme)
    mesh = o3d.geometry.TriangleMesh()
    num_tris = [len(me.mesh.triangles) for me in full_mesh.meshes]
    mids = []
    if sum(num_tris) > points:
        scaling = points / sum(num_tris)
        target_tris = [int(np.ceil(max(1., t * scaling))) for t in num_tris]
        for me_idx, me in enumerate(full_mesh.meshes):
            me.mesh.triangle_uvs = o3d.utility.Vector2dVector([])
            pcd = o3d.geometry.PointCloud(me.mesh.vertices)
            vertex_size = np.mean(pcd.compute_point_cloud_distance(
                o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.mean(np.asarray(pcd.points), axis=0).reshape(1, -1))))) / 4
            tm = me.mesh.simplify_vertex_clustering(vertex_size)
            tm.remove_duplicated_vertices()
            tm.remove_unreferenced_vertices()
            bounce = 1
            while len(tm.triangles) > target_tris[me_idx] or len(tm.triangles) == 0:
                vertex_size *= (1.5 if len(tm.triangles) > target_tris[me_idx] else .6) * bounce
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

def splitBox(bounding_box, npo: np.ndarray = None):
    splits = np.zeros((8, 2, 3))
    if npo is not None:
        try:
            box_hull = np.array([npo.min(axis=0) - BOX_CUSHION, npo.mean(axis=0), npo.max(axis=0) + BOX_CUSHION])
        except ValueError:
            box_hull = np.zeros((3, 3))
    else:
        box_hull = np.array([bounding_box.min(axis=0) - BOX_CUSHION, bounding_box.mean(axis=0),
                             bounding_box.max(axis=0) + BOX_CUSHION])
    for bidx, (x, y, z) in enumerate(itertools.product(range(2), range(2), range(2))):
        try:
            boxpo = npo[np.logical_and(npo[:, 0] >= box_hull[x, 0], npo[:, 0] <= box_hull[x + 1, 0]) &
                         np.logical_and(npo[:, 1] >= box_hull[y, 1], npo[:, 1] <= box_hull[y + 1, 1]) &
                         np.logical_and(npo[:, 2] >= box_hull[z, 2], npo[:, 2] <= box_hull[z + 1, 2])]
            splits[bidx, :] = np.array([boxpo.min(axis=0) - BOX_CUSHION,
                                        boxpo.max(axis=0) + BOX_CUSHION]).reshape((2, 3))
        except ValueError:
            splits[bidx, :] = np.zeros((2, 3))
        # splits[bidx, :] = np.array([[box_hull[x, 0], box_hull[y, 1], box_hull[z, 2]],
        #                             [box_hull[x + 1, 0], box_hull[y + 1, 1], box_hull[z + 1, 2]]]).reshape((2, 3))
    return splits

def genOctree(bounding_box: np.ndarray, num_levels: int = 3, points: np.ndarray = None):
    octree = np.zeros((sum(8**n for n in range(num_levels)), 2, 3))
    octree[0, ...] = bounding_box
    abs_idx = 1
    npo = None
    for level in range(num_levels - 1):
        for parent in range(8**level):
            pbox = octree[parent + sum(8**l for l in range(level))]
            if points is not None:
                npo = points[np.logical_and(points[:, 0] > pbox[0, 0], points[:, 0] < pbox[1, 0]) &
                             np.logical_and(points[:, 1] > pbox[0, 1], points[:, 1] < pbox[1, 1]) &
                             np.logical_and(points[:, 2] > pbox[0, 2], points[:, 2] < pbox[1, 2])]
            octree[abs_idx:abs_idx + 8] = splitBox(pbox, npo)
            abs_idx += 8
    return octree

def getBoxesSamplesFromMesh(a_mesh: o3d.geometry.TriangleMesh, num_box_levels: int=4, sample_points: int=10000,
                            material_sigmas: list=None, material_kd: list=None, material_ks: list = None,
                            view_pos: np.ndarray = None, use_box_pts: bool = True):
    # Generate bounding box tree
    mesh_tri_idx = np.asarray(a_mesh.triangles)
    mesh_vertices = np.asarray(a_mesh.vertices)
    mesh_normals = np.asarray(a_mesh.triangle_normals)
    mesh_tri_vertices = mesh_vertices[mesh_tri_idx]

    aabb = a_mesh.get_axis_aligned_bounding_box()
    max_bound = aabb.get_max_bound()
    min_bound = aabb.get_min_bound()
    root_box = np.array([min_bound, max_bound])
    boxes = genOctree(root_box, num_box_levels, mesh_vertices if use_box_pts else None)

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
    except Exception:
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

    tri_material = np.concatenate([mesh_sigmas.reshape((-1, 1)),
                                   mesh_kd.reshape((-1, 1)), mesh_ks.reshape((-1, 1))], axis=1)

    # Sample the mesh to get points for initial raycasting

    # points = np.sum(mesh_tri_vertices / 3., axis=1)
    if view_pos is None:
        points = np.asarray(a_mesh.sample_points_poisson_disk(sample_points).points)
    else:
        # Calculate out the beamwidths so we don't waste GPU cycles on rays into space
        pvecs = a_mesh.get_center() - view_pos
        pointing_az = np.arctan2(pvecs[:, 0], pvecs[:, 1])
        pointing_el = -np.arcsin(pvecs[:, 2] / np.linalg.norm(pvecs, axis=1))
        mesh_views = mesh_vertices[None, :, :] - view_pos[:, None, :]
        view_az = np.arctan2(mesh_views[:, :, 0], mesh_views[:, :, 1])
        view_el = -np.arcsin(mesh_views[:, :, 2] / np.linalg.norm(mesh_views, axis=2))
        bw_az = abs(pointing_az[:, None] - view_az).max()
        bw_el = abs(pointing_el[:, None] - view_el).max()
        points = detectPoints(boxes, sorted_tri_idx[:, 1], mesh_idx_key, mesh_tri_idx, mesh_vertices, mesh_normals, tri_material,
                     sample_points, view_pos, bw_az, bw_el, pointing_az, pointing_el)

    return (boxes, sorted_tri_idx[:, 1], mesh_idx_key, mesh_tri_idx, mesh_vertices, mesh_normals, tri_material), points

@nvtx.annotate(color='blue')
def getRangeProfileFromMesh(boxes: np.ndarray, tri_box: np.ndarray, tri_box_key: np.ndarray, tri_idxes: np.ndarray,
                            tri_verts: np.ndarray, tri_norm: np.ndarray, tri_material: np.ndarray,
                            sample_points: np.ndarray, a_obs_pt: list[np.ndarray], pan: list[np.ndarray], tilt: list[np.ndarray],
                            radar_equation_constant: float, bw_az: float, bw_el: float, nsam: int, fc: float,
                            near_range_s: float, num_bounces: int=3, debug: bool=False, streams: list[cuda.stream]=None) -> tuple[list, list, list, list] | list:
    npulses = a_obs_pt[0].shape[0]
    npoints = sample_points.shape[0]

    debug_rays = []
    debug_raydirs = []
    debug_raypower = []

    # GPU device calculations
    threads_per_block = getMaxThreads()
    poss_threads = np.array([npoints % n for n in range(1, threads_per_block[1])])
    poss_pulse_threads = np.array([npulses % n for n in range(1, threads_per_block[0])])
    bprun = (max(1, npulses // (np.max(np.where(poss_pulse_threads == poss_pulse_threads.min())[0]) + 1)),
             npoints // (np.max(np.where(poss_threads == poss_threads.min())[0]) + 1))

    # These are the mesh constants that don't change with intersection points
    sample_points_gpu = cuda.to_device(sample_points.astype(np.float64))
    tri_norm_gpu = cuda.to_device(tri_norm.astype(np.float64))
    tri_idxes_gpu = cuda.to_device(tri_idxes.astype(np.int32))
    tri_verts_gpu = cuda.to_device(tri_verts.astype(np.float64))
    tri_material_gpu = cuda.to_device(tri_material.astype(np.float64))
    tri_box_gpu = cuda.to_device(tri_box.astype(np.int32))
    tri_box_key_gpu = cuda.to_device(tri_box_key.astype(np.int32))
    boxes_gpu = cuda.to_device(boxes.astype(np.float64))
    params_gpu = cuda.to_device(np.array([2 * np.pi / (c0 / fc), near_range_s, fs, bw_az, bw_el, radar_equation_constant]).astype(np.float64))

    pd_r = [np.zeros((npulses, nsam), dtype=np.float64) for _ in pan]
    pd_i = [np.zeros((npulses, nsam), dtype=np.float64) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with cuda.defer_cleanup():
        for stream, rec, az, el, pdr_tmp, pdi_tmp in zip(streams, a_obs_pt, pan, tilt, pd_r, pd_i):
            with cuda.pinned(rec, az, el, pdr_tmp, pdi_tmp):
                receive_xyz_gpu = cuda.to_device(rec, stream=stream)
                ray_origin_gpu = cuda.device_array((npulses, npoints, 3), stream=stream)
                ray_dir_gpu = cuda.device_array((npulses, npoints, 3), stream=stream)
                ray_power_gpu = cuda.device_array((npulses, npoints), stream=stream)
                ray_distance_gpu = cuda.device_array((npulses, npoints), stream=stream)
                az_gpu = cuda.to_device(az, stream=stream)
                el_gpu = cuda.to_device(el, stream=stream)
                pd_r_gpu = cuda.device_array((npulses, nsam), stream=stream)
                pd_i_gpu = cuda.device_array((npulses, nsam), stream=stream)
                # Calculate out the attenuation from the beampattern
                calcOriginDirAtt[bprun, threads_per_block, stream](receive_xyz_gpu, sample_points_gpu, az_gpu, el_gpu,
                                                           params_gpu, ray_dir_gpu, ray_origin_gpu, ray_power_gpu)
                # Since we know the first ray can see the receiver, this is a special call to speed things up
                calcBounceInit[bprun, threads_per_block, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                         ray_power_gpu, boxes_gpu, tri_box_gpu,
                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                         receive_xyz_gpu,
                                                         az_gpu, el_gpu, params_gpu)
                if debug:
                    debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                    debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                    debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                if num_bounces > 1:
                    for _ in range(1, num_bounces):

                        calcBounceLoop[bprun, threads_per_block, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                                         ray_power_gpu, boxes_gpu, tri_box_gpu,
                                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                                         receive_xyz_gpu, az_gpu, el_gpu, params_gpu)
                        if debug:
                            debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                            debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                            debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                # We need to copy to host this way so we don't accidentally sync the streams
                pd_r_gpu.copy_to_host(pdr_tmp, stream=stream)
                pd_i_gpu.copy_to_host(pdi_tmp, stream=stream)
                del ray_power_gpu, ray_origin_gpu, ray_dir_gpu, ray_distance_gpu, az_gpu, el_gpu, receive_xyz_gpu, pd_r_gpu, pd_i_gpu
    # cuda.synchronize()
    del tri_material_gpu
    del tri_box_key_gpu
    del tri_idxes_gpu
    del tri_verts_gpu
    del tri_box_gpu
    del boxes_gpu
    del tri_norm_gpu
    del params_gpu
    del sample_points_gpu

    # Combine chunks in comprehension
    final_rp = [pr + 1j * pi for pr, pi in zip(pd_r, pd_i)]
    if debug:
        return final_rp, debug_rays, debug_raydirs, debug_raypower
    else:
        return final_rp


def detectPoints(boxes: np.ndarray, tri_box: np.ndarray, tri_box_key: np.ndarray, tri_idxes: np.ndarray,
                            tri_verts: np.ndarray, tri_norm: np.ndarray, tri_material: np.ndarray,
                            npoints: int, a_obs_pt: np.ndarray, bw_az, bw_el, pointing_az, pointing_el):

    # GPU device calculations
    threads_per_block = getMaxThreads()
    npulses = a_obs_pt.shape[0]
    poss_threads = np.array([npoints % n for n in range(1, threads_per_block[1])])
    poss_pulse_threads = np.array([npulses % n for n in range(1, threads_per_block[0])])
    bprun = (max(1, npulses // (np.max(np.where(poss_pulse_threads == poss_pulse_threads.min())[0]) + 1)),
             npoints // (np.max(np.where(poss_threads == poss_threads.min())[0]) + 1))

    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cuda.to_device(tri_norm.astype(np.float64))
    tri_idxes_gpu = cuda.to_device(tri_idxes.astype(np.int32))
    tri_verts_gpu = cuda.to_device(tri_verts.astype(np.float64))
    tri_material_gpu = cuda.to_device(tri_material.astype(np.float64))
    tri_box_gpu = cuda.to_device(tri_box.astype(np.int32))
    tri_box_key_gpu = cuda.to_device(tri_box_key.astype(np.int32))
    boxes_gpu = cuda.to_device(boxes.astype(np.float64))

    ray_origins_gpu = cuda.to_device(np.zeros((npulses, npoints, 3)))
    receive_xyz_gpu = cuda.to_device(a_obs_pt)

    # Calculate out direction vectors for the beam to hit
    points = np.zeros((1, 3))
    while len(points) < npoints:
        ray_power_gpu = cuda.to_device(np.ones((npulses, npoints)))
        rdirs = azelToVec(pointing_az[:, None] + np.random.uniform(-bw_az, bw_az, (npulses, npoints)),
                      pointing_el[:, None] + np.random.uniform(-bw_el, bw_el, (npulses, npoints))).T.swapaxes(0, 1)
        ray_dir_gpu = cuda.to_device(np.ascontiguousarray(rdirs))

        calcIntersectionPoints[bprun, threads_per_block](ray_origins_gpu, ray_dir_gpu, ray_power_gpu, boxes_gpu, tri_box_gpu,
                                                             tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                             tri_norm_gpu, tri_material_gpu, receive_xyz_gpu)
        newpoints = ray_origins_gpu.copy_to_host()[ray_power_gpu.copy_to_host().astype(bool)]
        points = newpoints if newpoints.shape[0] > npoints else np.concatenate((points, newpoints))

    return points[:npoints, :]