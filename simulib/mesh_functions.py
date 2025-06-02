import itertools
import multiprocessing as mp
from numba import cuda
from .cuda_mesh_kernels import calcBounceLoop, calcBounceInit, calcOriginDirAtt, calcIntersectionPoints, \
    assignBoxPoints, calcClosestIntersection, calcReturnPower, \
    calcClosestIntersectionWithoutBounce, calcSceneOcclusion
from .simulation_functions import azelToVec, factors
import numpy as np
import open3d as o3d
from .cuda_kernels import optimizeStridedThreadBlocks2d
import plotly.graph_objects as go

c0 = 299792458.0
_float = np.float32

BOX_CUSHION = .01
MULTIPROCESSORS = cuda.get_current_device().MULTIPROCESSOR_COUNT
THREADS_PER_BLOCK = 512
BLOCK_MULTIPLIER = 64


def readCombineMeshFile(fnme: str, points: int=100000, scale: float=None) -> o3d.geometry.TriangleMesh:
    full_mesh = o3d.io.read_triangle_model(fnme)
    mesh = o3d.geometry.TriangleMesh()
    num_tris = [len(me.mesh.triangles) for me in full_mesh.meshes]
    mids = []
    if sum(num_tris) > points:
        for me in full_mesh.meshes:
            mesh += me.mesh
            mids += [me.material_idx for _ in range(len(me.mesh.triangles))]
        pcd = o3d.geometry.PointCloud(mesh.vertices)
        vertex_size = np.asarray(pcd.compute_nearest_neighbor_distance()).mean()
        tm = mesh.simplify_vertex_clustering(vertex_size * 2)
        tm.remove_duplicated_vertices()
        tm.remove_unreferenced_vertices()
        brit = 0
        scaling = 2
        while len(tm.triangles) > points and brit < 10:
            scaling *= len(tm.triangles) / points
            tm = mesh.simplify_vertex_clustering(vertex_size * scaling)
            tm.remove_duplicated_vertices()
            tm.remove_unreferenced_vertices()
            brit += 1
        mids = list(np.zeros(len(tm.triangles), dtype=int))
        mesh = tm
    else:
        for me in full_mesh.meshes:
            mesh += me.mesh
            mids += [me.material_idx for _ in range(len(me.mesh.triangles))]
    '''if sum(num_tris) > points:
        scaling = points / sum(num_tris)
        target_tris = [int(np.ceil(max(1., t * scaling))) for t in num_tris]
        for me_idx, me in enumerate(full_mesh.meshes):
            me.mesh.triangle_uvs = o3d.utility.Vector2dVector([])
            pcd = o3d.geometry.PointCloud(me.mesh.vertices)
            vertex_size = np.asarray(pcd.compute_nearest_neighbor_distance()).mean()
            tm = me.mesh.simplify_vertex_clustering(vertex_size)
            tm.remove_duplicated_vertices()
            tm.remove_unreferenced_vertices()
            bounce = 1
            while len(tm.triangles) > target_tris[me_idx] or len(tm.triangles) == 0:
                vertex_size *= (10. if len(tm.triangles) > target_tris[me_idx] else .2) * bounce
                tm = me.mesh.simplify_vertex_clustering(vertex_size)
                tm.remove_duplicated_vertices()
                tm.remove_unreferenced_vertices()
                bounce -= .1
                if bounce < .01:
                    break
            mesh += tm
            mids += [me.material_idx for _ in range(len(tm.triangles))]'''

    mesh.triangle_material_ids = o3d.utility.IntVector(mids)

    if scale:
        mesh = mesh.scale(scale, mesh.get_center())
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    return mesh


def getRangeProfileFromScene(scene, sampled_points: int | np.ndarray, tx_pos: list[np.ndarray], rx_pos: list[np.ndarray],
                            pan: list[np.ndarray], tilt: list[np.ndarray], radar_equation_constant: float, bw_az: float,
                            bw_el: float, nsam: int, fc: float, near_range_s: float, fs: float = 2e9, num_bounces: int=3,
                            debug: bool=False, streams: list[cuda.stream]=None, use_supersampling: bool=True) -> tuple[list, list, list, list] | list:
    # This is here because the single mesh function is more highly optimized for a single mesh and should therefore
    # be used.
    if len(scene.meshes) == 1:
        return getRangeProfileFromMesh(scene.meshes[0], sampled_points, tx_pos, rx_pos, pan, tilt, 
                                       radar_equation_constant, bw_az, bw_el, nsam, fc, near_range_s, fs, num_bounces,
                                       debug, streams, use_supersampling)
    npulses = tx_pos[0].shape[0]
    if isinstance(sampled_points, int):
        sampled_points = tx_pos[0][0] + azelToVec(pan[0].mean() + np.random.normal(0, bw_az, (sampled_points,)),
                                                  tilt[0].mean() + np.random.normal(0, bw_el,
                                                                                    (sampled_points,))).T * 100
        # sampled_points = mesh.sample(sampled_points, view_pos=np.array([tx[0] for tx in tx_pos]))
    npoints = sampled_points.shape[0]

    debug_rays = []
    debug_raydirs = []
    debug_raypower = []

    threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((npulses, npoints))

    # These are the mesh constants that don't change with intersection points

    sample_points_gpu = cuda.to_device(sampled_points.astype(_float))


    '''sxminx_gpu = cuda.to_device(scene.tree[:, 0, 0].astype(_float))
    sxminy_gpu = cuda.to_device(scene.tree[:, 0, 1].astype(_float))
    sxminz_gpu = cuda.to_device(scene.tree[:, 0, 2].astype(_float))
    sxmaxx_gpu = cuda.to_device(scene.tree[:, 1, 0].astype(_float))
    sxmaxy_gpu = cuda.to_device(scene.tree[:, 1, 1].astype(_float))
    sxmaxz_gpu = cuda.to_device(scene.tree[:, 1, 2].astype(_float))'''
    # boxes_gpu = cuda.to_device(mesh.octree.astype(_float))
    params_gpu = cuda.to_device(np.array([2 * np.pi / (c0 / fc), near_range_s, fs, bw_az, bw_el,
                                          radar_equation_constant, use_supersampling]).astype(_float))

    pd_r = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    pd_i = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with cuda.defer_cleanup():
        for tx, rx, az, el, pdr_tmp, pdi_tmp in zip(tx_pos, rx_pos, pan, tilt, pd_r, pd_i):
            with cuda.pinned(tx, rx, az, el, pdr_tmp, pdi_tmp):
                receive_xyz_gpu = cuda.to_device(rx, )
                transmit_xyz_gpu = cuda.to_device(tx, )
                ray_origin_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, )
                ray_dir_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, )
                ray_power_gpu = cuda.device_array((npulses, npoints), dtype=_float, )
                ray_distance_gpu = cuda.device_array((npulses, npoints), dtype=_float, )
                # scene_intersects = cuda.device_array((npulses, npoints, len(scene.meshes)), dtype=bool, )
                az_gpu = cuda.to_device(az, )
                el_gpu = cuda.to_device(el, )
                pd_r_gpu = cuda.device_array((npulses, nsam), dtype=_float, )
                pd_i_gpu = cuda.device_array((npulses, nsam), dtype=_float, )
                # Calculate out the attenuation from the beampattern
                calcOriginDirAtt[blocks_strided, threads_strided](transmit_xyz_gpu, sample_points_gpu, az_gpu,
                                                                          el_gpu,
                                                                          params_gpu, ray_dir_gpu, ray_origin_gpu,
                                                                          ray_power_gpu)

                for _ in range(num_bounces):
                    ray_intersection_gpu = cuda.to_device(np.ones((npulses, npoints, 3)) * 1e9, )
                    ray_bounce_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, )
                    ray_bounce_power_gpu = cuda.device_array((npulses, npoints), dtype=_float, )

                    '''# Get the correct mesh segments for each in the scene
                    determineSceneRayIntersections[blocks_strided, threads_strided](sxminx_gpu, sxminy_gpu,
                                                                            sxminz_gpu, sxmaxx_gpu,
                                                                            sxmaxy_gpu, sxmaxz_gpu, ray_dir_gpu,
                                                   ray_origin_gpu, scene_intersects)

                    sc_inter = scene_intersects.copy_to_host()'''

                    for mesh in scene.meshes:
                        tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float))
                        tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32))
                        tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float))
                        tri_material_gpu = cuda.to_device(mesh.materials.astype(_float))
                        tri_box_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32))
                        tri_box_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32))

                        # This is for optimization purposes
                        bxminx_gpu = cuda.to_device(mesh.bvh[:, 0, 0].astype(_float))
                        bxminy_gpu = cuda.to_device(mesh.bvh[:, 0, 1].astype(_float))
                        bxminz_gpu = cuda.to_device(mesh.bvh[:, 0, 2].astype(_float))
                        bxmaxx_gpu = cuda.to_device(mesh.bvh[:, 1, 0].astype(_float))
                        bxmaxy_gpu = cuda.to_device(mesh.bvh[:, 1, 1].astype(_float))
                        bxmaxz_gpu = cuda.to_device(mesh.bvh[:, 1, 2].astype(_float))

                        calcClosestIntersection[blocks_strided, threads_strided](ray_origin_gpu, ray_intersection_gpu, ray_dir_gpu,
                                                                                ray_bounce_gpu, ray_distance_gpu, ray_power_gpu, ray_bounce_power_gpu, bxminx_gpu, bxminy_gpu,
                                                                                bxminz_gpu, bxmaxx_gpu,
                                                                                bxmaxy_gpu, bxmaxz_gpu, tri_box_gpu,
                                                                                tri_box_key_gpu, tri_verts_gpu,
                                                                                tri_idxes_gpu,
                                                                                tri_norm_gpu, tri_material_gpu, params_gpu)
                    if debug:
                        debug_rays.append(ray_intersection_gpu.copy_to_host())
                        debug_raydirs.append(ray_bounce_gpu.copy_to_host())
                        debug_raypower.append(ray_bounce_power_gpu.copy_to_host())

                    for mesh in scene.meshes:
                        tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float))
                        tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32))
                        tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float))
                        tri_box_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32))
                        tri_box_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32))

                        # This is for optimization purposes
                        bxminx_gpu = cuda.to_device(mesh.bvh[:, 0, 0].astype(_float))
                        bxminy_gpu = cuda.to_device(mesh.bvh[:, 0, 1].astype(_float))
                        bxminz_gpu = cuda.to_device(mesh.bvh[:, 0, 2].astype(_float))
                        bxmaxx_gpu = cuda.to_device(mesh.bvh[:, 1, 0].astype(_float))
                        bxmaxy_gpu = cuda.to_device(mesh.bvh[:, 1, 1].astype(_float))
                        bxmaxz_gpu = cuda.to_device(mesh.bvh[:, 1, 2].astype(_float))

                        ray_poss_gpu = cuda.to_device(np.ones((npulses, npoints)).astype(bool))
                        calcSceneOcclusion[blocks_strided, threads_strided](ray_intersection_gpu, ray_poss_gpu, bxminx_gpu, bxminy_gpu,
                                                                         bxminz_gpu, bxmaxx_gpu,
                                                                         bxmaxy_gpu, bxmaxz_gpu, tri_box_gpu,
                                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                                         tri_norm_gpu,
                                                                         receive_xyz_gpu)
                    calcReturnPower[blocks_strided, threads_strided](ray_intersection_gpu, ray_distance_gpu,
                                                                             ray_bounce_power_gpu, ray_poss_gpu, pd_r_gpu,
                                                                            pd_i_gpu,
                                                                            receive_xyz_gpu,
                                                                            az_gpu, el_gpu, params_gpu)
                    ray_origin_gpu = ray_intersection_gpu
                    ray_dir_gpu = ray_bounce_gpu
                    ray_power_gpu = ray_bounce_power_gpu


                # We need to copy to host this way so we don't accidentally sync the streams
                pd_r_gpu.copy_to_host(pdr_tmp)
                pd_i_gpu.copy_to_host(pdi_tmp)

    # cuda.synchronize()

    # Combine chunks in comprehension
    final_rp = [pr + 1j * pi for pr, pi in zip(pd_r, pd_i)]
    if debug:
        return final_rp, debug_rays, debug_raydirs, debug_raypower
    else:
        return final_rp



def getRangeProfileFromMesh(mesh, sampled_points: int | np.ndarray, tx_pos: list[np.ndarray], rx_pos: list[np.ndarray],
                            pan: list[np.ndarray], tilt: list[np.ndarray], radar_equation_constant: float, bw_az: float,
                            bw_el: float, nsam: int, fc: float, near_range_s: float, fs: float = 2e9, num_bounces: int=3,
                            debug: bool=False, streams: list[cuda.stream]=None, use_supersampling: bool = True) -> tuple[list, list, list, list] | list:
    """
    Generate a range profile, given a Mesh object and some other values

    """
    npulses = tx_pos[0].shape[0]
    if isinstance(sampled_points, int):
        sampled_points = tx_pos[0][0] + azelToVec(pan[0].mean() + np.random.normal(0, bw_az, (sampled_points,)),
                                   tilt[0].mean() + np.random.normal(0, bw_el, (sampled_points,))).T * 100
        # sampled_points = mesh.sample(sampled_points, view_pos=np.array([tx[0] for tx in tx_pos]))
    npoints = sampled_points.shape[0]

    debug_rays = []
    debug_raydirs = []
    debug_raypower = []

    threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((npulses, npoints))

    # These are the mesh constants that don't change with intersection points

    sample_points_gpu = cuda.to_device(sampled_points.astype(_float))
    tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float))
    tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32))
    tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float))
    tri_material_gpu = cuda.to_device(mesh.materials.astype(_float))
    tri_box_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32))
    tri_box_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32))

    #This is for optimization purposes
    kd_tree_gpu = cuda.to_device(mesh.bvh.astype(_float))
    params_gpu = cuda.to_device(np.array([2 * np.pi / (c0 / fc), near_range_s, fs, bw_az, bw_el,
                                          radar_equation_constant, use_supersampling]).astype(_float))
    conical_sampling_gpu = cuda.to_device(np.array([[np.pi / 2, c0 / (2 * fc)],
                                                    [np.pi, c0 / (2 * fc)],
                                                    [-np.pi / 2, c0 / (2 * fc)],
                                                    [-np.pi, c0 / (2 * fc)]]).astype(_float))
    # conical_sampling_gpu = cuda.to_device(np.array([-1., 1., 0]).astype(_float))

    pd_r = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    pd_i = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with cuda.defer_cleanup():
        for stream, tx, rx, az, el, pdr_tmp, pdi_tmp in zip(streams, tx_pos, rx_pos, pan, tilt, pd_r, pd_i):
            with cuda.pinned(tx, rx, az, el, pdr_tmp, pdi_tmp):
                receive_xyz_gpu = cuda.to_device(rx, stream=stream)
                transmit_xyz_gpu = cuda.to_device(tx, stream=stream)
                ray_origin_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, stream=stream)
                ray_dir_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, stream=stream)
                ray_power_gpu = cuda.device_array((npulses, npoints), dtype=_float, stream=stream)
                ray_distance_gpu = cuda.device_array((npulses, npoints), dtype=_float, stream=stream)
                az_gpu = cuda.to_device(az, stream=stream)
                el_gpu = cuda.to_device(el, stream=stream)
                pd_r_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                pd_i_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                # Calculate out the attenuation from the beampattern
                calcOriginDirAtt[blocks_strided, threads_strided, stream](transmit_xyz_gpu, sample_points_gpu, az_gpu, el_gpu,
                                                           params_gpu, ray_dir_gpu, ray_origin_gpu, ray_power_gpu)
                # Since we know the first ray can see the receiver, this is a special call to speed things up
                calcBounceInit[blocks_strided, threads_strided, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                         ray_power_gpu, kd_tree_gpu, tri_box_gpu,
                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                         receive_xyz_gpu,
                                                         az_gpu, el_gpu, params_gpu, conical_sampling_gpu)
                if debug:
                    debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                    debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                    debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                if num_bounces > 1:
                    for _ in range(1, num_bounces):
                        calcBounceLoop[blocks_strided, threads_strided, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                                         ray_power_gpu, bxminx_gpu, bxminy_gpu, bxminz_gpu, bxmaxx_gpu,
                                                                 bxmaxy_gpu, bxmaxz_gpu, tri_box_gpu,
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
                del ray_power_gpu, ray_origin_gpu, ray_dir_gpu, ray_distance_gpu, az_gpu, el_gpu, receive_xyz_gpu, pd_r_gpu, pd_i_gpu, transmit_xyz_gpu
    # cuda.synchronize()
    del tri_material_gpu
    del tri_box_key_gpu
    del tri_idxes_gpu
    del tri_verts_gpu
    del tri_box_gpu
    del kd_tree_gpu
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
    npulses = a_obs_pt.shape[0]

    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cuda.to_device(tri_norm.astype(_float))
    tri_idxes_gpu = cuda.to_device(tri_idxes.astype(np.int32))
    tri_verts_gpu = cuda.to_device(tri_verts.astype(_float))
    tri_material_gpu = cuda.to_device(tri_material.astype(_float))
    tri_box_gpu = cuda.to_device(tri_box.astype(np.int32))
    tri_box_key_gpu = cuda.to_device(tri_box_key.astype(np.int32))
    # This is for optimization purposes
    bxminx_gpu = cuda.to_device(boxes[:, 0, 0].astype(_float))
    bxminy_gpu = cuda.to_device(boxes[:, 0, 1].astype(_float))
    bxminz_gpu = cuda.to_device(boxes[:, 0, 2].astype(_float))
    bxmaxx_gpu = cuda.to_device(boxes[:, 1, 0].astype(_float))
    bxmaxy_gpu = cuda.to_device(boxes[:, 1, 1].astype(_float))
    bxmaxz_gpu = cuda.to_device(boxes[:, 1, 2].astype(_float))

    ray_origins_gpu = cuda.to_device(np.zeros((npulses, npoints, 3)).astype(_float))
    receive_xyz_gpu = cuda.to_device(a_obs_pt.astype(_float))

    # Calculate out direction vectors for the beam to hit
    points = []
    threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((npulses, npoints))
    while len(points) < npoints:
        ray_power_gpu = cuda.to_device(np.ones((npulses, npoints)))

        # Group az and el for faster traversal of octree
        n_anchors = min(16, npoints)
        if npoints < n_anchors**2:
            azes = pointing_az[:, None] + np.random.uniform(-bw_az, bw_az, (npulses, npoints))
            eles = np.sort(pointing_el[:, None] + np.random.uniform(-bw_el, bw_el, (npulses, npoints)))
        else:
            az_anchors, el_anchors = np.meshgrid(np.linspace(-bw_az, bw_az, n_anchors), np.linspace(-bw_el, bw_el, n_anchors))
            azes = np.concatenate([pointing_az[:, None] + np.random.uniform(a - bw_az / n_anchors, a + bw_az / n_anchors,
                                                                            (npulses, npoints // n_anchors**2)) for a in
                                   az_anchors.flatten()], axis=1)
            eles = np.concatenate([pointing_el[:, None] + np.random.uniform(a - bw_el / n_anchors, a + bw_el / n_anchors,
                                                                            (npulses, npoints // n_anchors**2)) for a in
                                   el_anchors.flatten()], axis=1)
        rdirs = azelToVec(azes,eles).T.swapaxes(0, 1)
        ray_dir_gpu = cuda.to_device(np.ascontiguousarray(rdirs.astype(_float)))

        calcIntersectionPoints[blocks_strided, threads_strided](ray_origins_gpu, ray_dir_gpu, ray_power_gpu, bxminx_gpu, bxminy_gpu, bxminz_gpu, bxmaxx_gpu,
                                                                 bxmaxy_gpu, bxmaxz_gpu, tri_box_gpu,
                                                             tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                             tri_norm_gpu, tri_material_gpu, receive_xyz_gpu)

        pts_used = ray_power_gpu.copy_to_host().astype(bool)

        newpoints = ray_origins_gpu.copy_to_host()[pts_used]
        points = (
            newpoints
            if newpoints.shape[0] > npoints or len(points) == 0
            else np.concatenate((points, newpoints))
        )

    points = points[:npoints, :]

    return points


def detectPointsScene(scene,
                            npoints: int, a_obs_pt: np.ndarray, bw_az, bw_el, pointing_az, pointing_el):
    npulses = a_obs_pt.shape[0]

    ray_origin_gpu = cuda.to_device(np.repeat(np.expand_dims(a_obs_pt, 1), npoints, axis=1).astype(_float))

    # Calculate out direction vectors for the beam to hit
    points = []
    threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((npulses, npoints))
    while len(points) < npoints:
        ray_power_gpu = cuda.to_device(np.zeros((npulses, npoints)))
        ray_intersection_gpu = cuda.to_device((np.zeros((npulses, npoints, 3)) + np.inf).astype(_float))

        n_anchors = min(32, npoints)
        if npoints < n_anchors**2:
            azes = pointing_az[:, None] + np.random.uniform(-bw_az, bw_az, (npulses, npoints))
            eles = np.sort(pointing_el[:, None] + np.random.uniform(-bw_el, bw_el, (npulses, npoints)))
        else:
            az_anchors, el_anchors = np.meshgrid(np.linspace(-bw_az, bw_az, n_anchors), np.linspace(-bw_el, bw_el, n_anchors))
            azes = np.concatenate([pointing_az[:, None] + np.random.uniform(a - bw_az / n_anchors, a + bw_az / n_anchors,
                                                                            (npulses, npoints // n_anchors**2)) for a in
                                   az_anchors.flatten()], axis=1)
            eles = np.concatenate([pointing_el[:, None] + np.random.uniform(a - bw_el / n_anchors, a + bw_el / n_anchors,
                                                                            (npulses, npoints // n_anchors**2)) for a in
                                   el_anchors.flatten()], axis=1)
        rdirs = azelToVec(azes,eles).T.swapaxes(0, 1)
        ray_dir_gpu = cuda.to_device(np.ascontiguousarray(rdirs.astype(_float)))

        for mesh in scene.meshes:
            tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float))
            tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32))
            tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float))
            leaf_list_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32))
            leaf_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32))

            # This is for optimization purposes
            kd_tree_gpu = cuda.to_device(mesh.bvh.astype(_float))

            calcClosestIntersectionWithoutBounce[blocks_strided, threads_strided](ray_origin_gpu, ray_intersection_gpu,
                                                                     ray_dir_gpu,
                                                                     ray_power_gpu, kd_tree_gpu, leaf_list_gpu,
                                                                     leaf_key_gpu, tri_verts_gpu,
                                                                     tri_idxes_gpu,
                                                                     tri_norm_gpu)

        pts_used = ray_power_gpu.copy_to_host().astype(bool)

        newpoints = ray_intersection_gpu.copy_to_host()[pts_used]
        points = (
            newpoints
            if newpoints.shape[0] > npoints or len(points) == 0
            else np.concatenate((points, newpoints))
        )

    points = points[:npoints, :]
    return points# [np.argsort(np.linalg.norm(points - scene.center, axis=1))]


def getIntersection(boxes: np.ndarray, tri_box: np.ndarray, tri_box_key: np.ndarray, tri_idxes: np.ndarray,
                            tri_verts: np.ndarray, tri_norm: np.ndarray, tri_material: np.ndarray,
                            sampled_points: np.ndarray, a_obs_pt: np.ndarray):
    npoints = sampled_points.shape[0]
    npulses = a_obs_pt.shape[0]

    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cuda.to_device(tri_norm.astype(_float))
    tri_idxes_gpu = cuda.to_device(tri_idxes.astype(np.int32))
    tri_verts_gpu = cuda.to_device(tri_verts.astype(_float))
    tri_material_gpu = cuda.to_device(tri_material.astype(_float))
    tri_box_gpu = cuda.to_device(tri_box.astype(np.int32))
    tri_box_key_gpu = cuda.to_device(tri_box_key.astype(np.int32))
    bxminx_gpu = cuda.to_device(boxes[:, 0, 0].astype(_float))
    bxminy_gpu = cuda.to_device(boxes[:, 0, 1].astype(_float))
    bxminz_gpu = cuda.to_device(boxes[:, 0, 2].astype(_float))
    bxmaxx_gpu = cuda.to_device(boxes[:, 1, 0].astype(_float))
    bxmaxy_gpu = cuda.to_device(boxes[:, 1, 1].astype(_float))
    bxmaxz_gpu = cuda.to_device(boxes[:, 1, 2].astype(_float))

    ray_origins_gpu = cuda.to_device(np.zeros((npulses, npoints, 3)).astype(_float))
    receive_xyz_gpu = cuda.to_device(a_obs_pt.astype(_float))

    # Calculate out direction vectors for the beam to hit
    points = []
    while len(points) < npoints:
        ray_power_gpu = cuda.to_device(np.ones((npulses, npoints)))
        rdirs = np.linalg.norm(a_obs_pt[:, None, :] - sampled_points[None, :, :], axis=1)
        ray_dir_gpu = cuda.to_device(np.ascontiguousarray(rdirs.astype(_float)))

        calcIntersectionPoints[BLOCK_MULTIPLIER * MULTIPROCESSORS, THREADS_PER_BLOCK](ray_origins_gpu, ray_dir_gpu,
                                                                                      ray_power_gpu, bxminx_gpu, bxminy_gpu, bxminz_gpu, bxmaxx_gpu,
                                                                 bxmaxy_gpu, bxmaxz_gpu, tri_box_gpu,
                                                             tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                             tri_norm_gpu, tri_material_gpu, receive_xyz_gpu)
        newpoints = ray_origins_gpu.copy_to_host()[ray_power_gpu.copy_to_host().astype(bool)]
        points = (
            newpoints
            if newpoints.shape[0] > npoints or len(points) == 0
            else np.concatenate((points, newpoints))
        )

    return points[:npoints, :]


def splitNextLevel(box_hull: np.ndarray = None, perspective: np.ndarray = None):
    splits = np.zeros((8, 2, 3))
    for bidx, (x, y, z) in enumerate(itertools.product(range(2), range(2), range(2))):
        splits[bidx, :] = np.array([[box_hull[x, 0], box_hull[y, 1], box_hull[z, 2]],
                                    [box_hull[x + 1, 0], box_hull[y + 1, 1], box_hull[z + 1, 2]]])
    if perspective is not None:
        split_mean = splits.mean(axis=1)
        dists = np.linalg.norm(split_mean - perspective, axis=1)
        dists[np.sum(split_mean, axis=1) == 0] = 0.
        splits = splits[np.argsort(dists)]
    return splits


def assocPointsWithOctree(octree: object, points: np.array):
    tri_box_idxes = np.zeros((points.shape[0], octree.shape[0]))

    # GPU device optimization for memory access
    ptx0_gpu = cuda.to_device(points[:, 0, 0].astype(_float))
    ptx1_gpu = cuda.to_device(points[:, 1, 0].astype(_float))
    ptx2_gpu = cuda.to_device(points[:, 2, 0].astype(_float))
    pty0_gpu = cuda.to_device(points[:, 0, 1].astype(_float))
    pty1_gpu = cuda.to_device(points[:, 1, 1].astype(_float))
    pty2_gpu = cuda.to_device(points[:, 2, 1].astype(_float))
    ptz0_gpu = cuda.to_device(points[:, 0, 2].astype(_float))
    ptz1_gpu = cuda.to_device(points[:, 1, 2].astype(_float))
    ptz2_gpu = cuda.to_device(points[:, 2, 2].astype(_float))
    for level in range(octree.depth):
        prev_level_idx = sum(8 ** l for l in range(level - 1))
        level_idx = sum(8 ** l for l in range(level))
        next_level_idx = sum(8 ** l for l in range(level + 1))
        threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((points.shape[0], next_level_idx - level_idx))
        centered_octree = octree.octree[level_idx:next_level_idx].mean(axis=1)
        extent_octree = (np.diff(octree.octree[level_idx:next_level_idx], axis=1) * .5)[:, 0]
        centered_octree_gpu = cuda.to_device(centered_octree.astype(_float))
        extent_octree_gpu = cuda.to_device(extent_octree.astype(_float))
        ptidx_gpu = cuda.to_device(np.zeros((points.shape[0], centered_octree_gpu.shape[0])).astype(bool))
        assignBoxPoints[blocks_strided, threads_strided](ptx0_gpu, ptx1_gpu, ptx2_gpu, pty0_gpu, pty1_gpu, pty2_gpu,
                                                         ptz0_gpu, ptz1_gpu, ptz2_gpu, centered_octree_gpu,
                                                         extent_octree_gpu, ptidx_gpu)
        box_idxes = ptidx_gpu.copy_to_host()
        full_boxes = np.any(box_idxes, axis=0)
        octree.mask[prev_level_idx:level_idx] = np.packbits(full_boxes)
        octree.is_occupied[level_idx:next_level_idx] = full_boxes
        tri_box_idxes[:, level_idx:next_level_idx] = box_idxes
    return octree, tri_box_idxes


def assocPointsWithKDTree(tree_bounds: np.array, points: np.array):
    tri_box_idxes = np.zeros((points.shape[0], tree_bounds.shape[0]))


    # GPU device optimization for memory access
    ptx0_gpu = cuda.to_device(points[:, 0, 0].astype(_float))
    ptx1_gpu = cuda.to_device(points[:, 1, 0].astype(_float))
    ptx2_gpu = cuda.to_device(points[:, 2, 0].astype(_float))
    pty0_gpu = cuda.to_device(points[:, 0, 1].astype(_float))
    pty1_gpu = cuda.to_device(points[:, 1, 1].astype(_float))
    pty2_gpu = cuda.to_device(points[:, 2, 1].astype(_float))
    ptz0_gpu = cuda.to_device(points[:, 0, 2].astype(_float))
    ptz1_gpu = cuda.to_device(points[:, 1, 2].astype(_float))
    ptz2_gpu = cuda.to_device(points[:, 2, 2].astype(_float))
    for level in range(int(np.log2(tree_bounds.shape[0] + 1))):
        # prev_level_idx = sum(2 ** l for l in range(level - 1))
        level_idx = sum(2 ** l for l in range(level))
        next_level_idx = sum(2 ** l for l in range(level + 1))
        threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((points.shape[0], next_level_idx - level_idx))
        centered_octree = tree_bounds[level_idx:next_level_idx].mean(axis=1)
        extent_octree = (np.diff(tree_bounds[level_idx:next_level_idx], axis=1) * .5)[:, 0]
        centered_octree_gpu = cuda.to_device(centered_octree.astype(_float))
        extent_octree_gpu = cuda.to_device(extent_octree.astype(_float))
        ptidx_gpu = cuda.to_device(np.zeros((points.shape[0], centered_octree_gpu.shape[0])).astype(bool))
        assignBoxPoints[blocks_strided, threads_strided](ptx0_gpu, ptx1_gpu, ptx2_gpu, pty0_gpu, pty1_gpu, pty2_gpu,
                                                         ptz0_gpu, ptz1_gpu, ptz2_gpu, centered_octree_gpu,
                                                         extent_octree_gpu, ptidx_gpu)
        box_idxes = ptidx_gpu.copy_to_host()
        # full_boxes = np.any(box_idxes, axis=0)
        # octree.mask[prev_level_idx:level_idx] = np.packbits(full_boxes)
        # octree.is_occupied[level_idx:next_level_idx] = full_boxes
        tri_box_idxes[:, level_idx:next_level_idx] = box_idxes
    return tri_box_idxes



'''def genOctree(bounding_box: np.ndarray, num_levels: int = 3, points: np.ndarray = None,
              perspective: np.ndarray = None, use_box_pts: bool = True):
    octree = np.zeros((sum(8 ** n for n in range(num_levels)), 2, 3))
    octree[0, ...] = bounding_box
    tri_box_idxes = np.zeros((points.shape[0], octree.shape[0]))

    # GPU device optimization for memory access
    ptx0_gpu = cuda.to_device(points[:, 0, 0].astype(_float))
    ptx1_gpu = cuda.to_device(points[:, 1, 0].astype(_float))
    ptx2_gpu = cuda.to_device(points[:, 2, 0].astype(_float))
    pty0_gpu = cuda.to_device(points[:, 0, 1].astype(_float))
    pty1_gpu = cuda.to_device(points[:, 1, 1].astype(_float))
    pty2_gpu = cuda.to_device(points[:, 2, 1].astype(_float))
    ptz0_gpu = cuda.to_device(points[:, 0, 2].astype(_float))
    ptz1_gpu = cuda.to_device(points[:, 1, 2].astype(_float))
    ptz2_gpu = cuda.to_device(points[:, 2, 2].astype(_float))
    max_ex = points.max(axis=1)
    min_ex = points.min(axis=1)
    print('Building octree level ', end='')
    for level in range(num_levels):
        print(f'{level}...', end='')
        level_idx = sum(8 ** l for l in range(level))
        next_level_idx = sum(8 ** l for l in range(level + 1))
        threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((points.shape[0], next_level_idx - level_idx))

        if use_box_pts:
            centered_octree_gpu = cuda.to_device(octree[level_idx:next_level_idx].mean(axis=1).astype(_float))
            extent_octree_gpu = cuda.to_device((np.diff(octree[level_idx:next_level_idx], axis=1) * .5)[:, 0].astype(_float))
            ptidx_gpu = cuda.to_device(np.zeros((points.shape[0], centered_octree_gpu.shape[0])).astype(bool))
            assignBoxPoints[blocks_strided, threads_strided](ptx0_gpu, ptx1_gpu, ptx2_gpu, pty0_gpu, pty1_gpu, pty2_gpu,
                                                      ptz0_gpu, ptz1_gpu, ptz2_gpu, centered_octree_gpu, extent_octree_gpu, ptidx_gpu)

            box_idxes = ptidx_gpu.copy_to_host()
            full_boxes = np.any(box_idxes, axis=0)
            used_boxes = np.argsort(box_idxes.sum(axis=0))
            use_box_idx = 0

            for idx in range(box_idxes.shape[1]):
                if full_boxes[idx]:
                    octree[level_idx + idx] = np.array([np.array([min_ex[box_idxes[:, idx]].min(axis=0),
                                                                  octree[level_idx + idx, 0]]).max(axis=0),
                                                        np.array([max_ex[box_idxes[:, idx]].max(axis=0),
                                                                  octree[level_idx + idx, 1]]).min(axis=0)])
                else:
                    split_box = octree[level_idx + used_boxes[use_box_idx]] + 0
                    split_sz = np.argsort(split_box[1] - split_box[0])
                    octree[level_idx + idx, 0] = split_box[0]
                    octree[level_idx + used_boxes[use_box_idx], 1] = split_box[1]
                    if split_sz[0] == 0:
                        octree[level_idx + idx, 1] = np.array(
                            [(split_box[1, 0] + split_box[0, 0]) / 2, split_box[1, 1], split_box[1, 2]])
                        octree[level_idx + used_boxes[use_box_idx], 0] = np.array(
                            [(split_box[1, 0] + split_box[0, 0]) / 2, split_box[0, 1], split_box[0, 2]])
                    elif split_sz[0] == 1:
                        octree[level_idx + idx, 1] = np.array(
                            [split_box[1, 0], (split_box[1, 1] + split_box[0, 1]) / 2, split_box[1, 2]])
                        octree[level_idx + used_boxes[use_box_idx], 0] = np.array(
                            [split_box[1, 0], (split_box[1, 1] + split_box[0, 1]) / 2, split_box[0, 2]])
                    else:
                        octree[level_idx + idx, 1] = np.array(
                            [split_box[1, 0], split_box[1, 1], (split_box[1, 2] + split_box[0, 2]) / 2])
                        octree[level_idx + used_boxes[use_box_idx], 0] = np.array(
                            [split_box[0, 0], split_box[1, 1], (split_box[1, 2] + split_box[0, 2]) / 2])
                    use_box_idx += 1
                    # octree[level_idx + idx] = 0.
        centered_octree_gpu = cuda.to_device(octree[level_idx:next_level_idx].mean(axis=1).astype(_float))
        extent_octree_gpu = cuda.to_device((np.diff(octree[level_idx:next_level_idx], axis=1) * .5)[:, 0].astype(_float))
        ptidx_gpu = cuda.to_device(np.zeros((points.shape[0], centered_octree_gpu.shape[0])).astype(bool))
        assignBoxPoints[blocks_strided, threads_strided](ptx0_gpu, ptx1_gpu, ptx2_gpu, pty0_gpu, pty1_gpu, pty2_gpu,
                                                      ptz0_gpu, ptz1_gpu, ptz2_gpu, centered_octree_gpu, extent_octree_gpu, ptidx_gpu)
        box_idxes = ptidx_gpu.copy_to_host()
        full_boxes = np.any(box_idxes, axis=0)
        for idx in range(box_idxes.shape[1]):
            if not full_boxes[idx]:
                octree[level_idx + idx] = 0.
        tri_box_idxes[:, level_idx:next_level_idx] = box_idxes
        if level < num_levels - 1:
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                npo = [np.array([octree[l, 0], octree[l].mean(axis=0), octree[l, 1]]) for l in range(level_idx, next_level_idx)]
                results = [pool.apply(splitNextLevel, args=(npo[p], perspective)) for p in
                           range(8 ** level)]

            for idx, r in enumerate(results):
                octree[next_level_idx + idx * 8:next_level_idx + (idx + 1) * 8] = r

    return octree, tri_box_idxes'''


def genKDTree(bounding_box: np.ndarray, points: np.ndarray = None, min_tri_per_box: int = 64):
    verts = points.reshape((-1, 3))  # points.mean(axis=1)
    depth = int(np.ceil(np.log2(verts.shape[0] / (3 * min_tri_per_box))))
    vidx = np.zeros((verts.shape[0])).astype(int)
    tree = np.zeros((2 ** depth - 1, 3))
    tree_bounds = np.zeros((2 ** depth - 1, 2, 3))
    tree[0] = [np.median(verts[:, 0]), bounding_box[:, 1].mean(), bounding_box[:, 2].mean()]
    tree_bounds[0] = bounding_box

    # Split based on depth mod 3
    for d in range(1, depth):
        axis = (d - 1) % 3
        for idx in range(sum(2 ** np.arange(d)), sum(2 ** np.arange(d + 1)), 2):
            parent_idx = idx >> 1
            v_act = verts[vidx == parent_idx]
            lefties = tree[parent_idx, axis] < v_act[:, axis]
            righties = tree[parent_idx, axis] >= v_act[:, axis]
            if np.sum(lefties) != 0:
                left = np.median(v_act[lefties], axis=0)
                vidx[np.logical_and(tree[parent_idx, axis] < verts[:, axis], vidx == parent_idx)] = idx
                tree[idx] = left
            if np.sum(righties) != 0:
                right = np.median(v_act[righties], axis=0)
                vidx[np.logical_and(tree[parent_idx, axis] >= verts[:, axis], vidx == parent_idx)] = idx + 1
                tree[idx + 1] = right
            tree_bounds[idx] = tree_bounds[parent_idx]
            tree_bounds[idx, 0, axis] = tree[parent_idx, axis]
            tree_bounds[idx + 1] = tree_bounds[parent_idx]
            tree_bounds[idx + 1, 1, axis] = tree[parent_idx, axis]
    return tree, tree_bounds


def getMeshFig(mesh, triangle_colors=None, title='Title Goes Here', zrange=100):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            # i, j and k give the vertices of triangles
            i=mesh.tri_idx[:, 0],
            j=mesh.tri_idx[:, 1],
            k=mesh.tri_idx[:, 2],
            facecolor=triangle_colors,
            showscale=True
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(zaxis=dict(range=[-30, zrange])),
    )
    return fig


def getSceneFig(scene, triangle_colors=None, title='Title Goes Here', zrange=100):
    # Get for first mesh
    fig = go.Figure(data=[
        go.Mesh3d(
            x=scene.meshes[0].vertices[:, 0],
            y=scene.meshes[0].vertices[:, 1],
            z=scene.meshes[0].vertices[:, 2],
            # i, j and k give the vertices of triangles
            i=scene.meshes[0].tri_idx[:, 0],
            j=scene.meshes[0].tri_idx[:, 1],
            k=scene.meshes[0].tri_idx[:, 2],
            facecolor=triangle_colors,
            showscale=True
        )
    ])
    for m in scene.meshes[1:]:
        fig.add_trace(go.Mesh3d(
                x=m.vertices[:, 0],
                y=m.vertices[:, 1],
                z=m.vertices[:, 2],
                # i, j and k give the vertices of triangles
                i=m.tri_idx[:, 0],
                j=m.tri_idx[:, 1],
                k=m.tri_idx[:, 2],
                facecolor=triangle_colors,
                showscale=True
            ))
    fig.update_layout(
        title=title,
        scene=dict(zaxis=dict(range=[-30, zrange])),
    )
    return fig


def drawOctreeBox(box):
    vertices = []
    for z in range(2):
        vertices.extend(
            (
                [box[0, 0], box[0, 1], box[z, 2]],
                [box[1, 0], box[0, 1], box[z, 2]],
                [box[1, 0], box[0, 1], box[int(not z), 2]],
                [box[1, 0], box[0, 1], box[z, 2]],
                [box[1, 0], box[1, 1], box[z, 2]],
                [box[1, 0], box[1, 1], box[int(not z), 2]],
                [box[1, 0], box[1, 1], box[z, 2]],
                [box[0, 0], box[1, 1], box[z, 2]],
                [box[0, 0], box[1, 1], box[int(not z), 2]],
                [box[0, 0], box[1, 1], box[z, 2]],
                [box[0, 0], box[0, 1], box[z, 2]],
            )
        )
    vertices = np.array(vertices)
    return go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='lines')


def msplit(a: np.uint32):
    x = a & 0x1fffff # we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff # shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff # shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f # shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3 # shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249
    return x


def morton_encode(x: np.uint32, y: np.uint32, z: np.uint32):
    answer = 0
    answer |= msplit(x) | msplit(y) << 1 | msplit(z) << 2
    return answer


def c1b2(x: np.uint32):
    x &= 0x09249249                # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >>  2)) & 0x030c30c3 # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >>  4)) & 0x0300f00f # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >>  8)) & 0xff0000ff # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff # x = ---- ---- ---- ---- ---- --98 7654 3210
    return x

def morton_decode(code: np.uint32):
    return c1b2(code >> 0), c1b2(code >> 1), c1b2(code >> 2)


def DecodeMorton3X(code: np.uint32):
    return c1b2(code >> 0)


def DecodeMorton3Y(code: np.uint32):
    return c1b2(code >> 1)


def DecodeMorton3Z(code: np.uint32):
    return c1b2(code >> 2)