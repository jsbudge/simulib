import itertools
import multiprocessing as mp
from numba import cuda
from .cuda_mesh_kernels import calcBounceLoop, calcBounceInit, calcOriginDirAtt, calcIntersectionPoints, \
    assignBoxPoints, calcClosestIntersection, calcReturnPower, \
    calcClosestIntersectionWithoutBounce, calcSceneOcclusion, calcTriangleRangeMinMax, calcTriangleBinSurfaceArea, \
    calcTriangleSampleVariance, calcTriangleReturns
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from .simulation_functions import azelToVec, factors
import numpy as np
import open3d as o3d
from .cuda_kernels import optimizeStridedThreadBlocks2d
import plotly.graph_objects as go
from tqdm import tqdm

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


def getRangeProfileFromScene(scene, tx_pos: list[np.ndarray], rx_pos: list[np.ndarray],
                            pan: list[np.ndarray], tilt: list[np.ndarray], radar_equation_constant: float, bw_az: float,
                            bw_el: float, nsam: int, fc: float, near_range_s: float, fs: float = 2e9, num_bounces: int=3,
                            debug: bool=False, streams: list[cuda.stream]=None) -> tuple[list, list, list, list] | list:
    # This is here because the single mesh function is more highly optimized for a single mesh and should therefore
    # be used.
    if len(scene.meshes) == 1:
        return getRangeProfileFromMesh(scene.meshes[0], tx_pos, rx_pos, pan, tilt,
                                       radar_equation_constant, bw_az, bw_el, nsam, fc, near_range_s, fs, num_bounces,
                                       debug, streams)
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
                                          radar_equation_constant]).astype(_float))

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



def getRangeProfileFromMesh(mesh, tx_pos: list[np.ndarray], rx_pos: list[np.ndarray],
                            pan: list[np.ndarray], tilt: list[np.ndarray], radar_equation_constant: float, bw_az: float,
                            bw_el: float, nsam: int, fc: float, near_range_s: float, fs: float = 2e9, num_bounces: int=3,
                            debug: bool=False, streams: list[cuda.stream]=None) -> tuple[list, list, list, list] | list:
    """
    Generate a range profile, given a Mesh object and some other values

    """
    npulses = tx_pos[0].shape[0]
    ntris = mesh.tri_idx.shape[0]

    debug_rays = []
    debug_raydirs = []
    debug_raypower = []

    # threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((npulses, npoints))
    threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((npulses, ntris))


    # These are the mesh constants that don't change with intersection points
    tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float))
    tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32))
    tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float))
    tri_material_gpu = cuda.to_device(mesh.materials.astype(_float))
    tri_box_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32))
    tri_box_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32))

    #This is for optimization purposes
    kd_tree_gpu = cuda.to_device(mesh.bvh.astype(_float))
    params_gpu = cuda.to_device(np.array([2 * np.pi / (c0 / fc), near_range_s, fs, bw_az, bw_el,
                                          radar_equation_constant]).astype(_float))

    target_variance = 1e-2

    # Random number generation
    rng_states = create_xoroshiro128p_states([tri_idxes_gpu.shape[0]], seed=1)

    pd_r = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    pd_i = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with (cuda.defer_cleanup()):
        for stream, tx, rx, az, el, pdr_tmp, pdi_tmp in zip(streams, tx_pos, rx_pos, pan, tilt, pd_r, pd_i):
            with cuda.pinned(tx, rx, az, el, pdr_tmp, pdi_tmp):
                receive_xyz_gpu = cuda.to_device(rx, stream=stream)
                transmit_xyz_gpu = cuda.to_device(tx, stream=stream)
                '''tri_ranges = np.zeros((ntris, 2)).astype(np.int32)
                tri_ranges[:, 0] = 32768
                triangle_ranges_gpu = cuda.to_device(tri_ranges, stream=stream)'''
                az_gpu = cuda.to_device(az, stream=stream)
                el_gpu = cuda.to_device(el, stream=stream)
                pd_r_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                pd_i_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)

                # Calculate range bin min/max per pulse per triangle
                '''calcTriangleRangeMinMax[blocks_strided, threads_strided, stream](transmit_xyz_gpu, tri_verts_gpu,
                                                                                 tri_idxes_gpu, triangle_ranges_gpu, params_gpu)'''

                # triangle_ranges = triangle_ranges_gpu.copy_to_host(stream=stream)
                # max_extent = int(triangle_ranges[..., 1].max())
                # threads_strided_sa, blocks_strided_sa = optimizeStridedThreadBlocks2d((ntris, max_extent))
                '''triangle_bin_surface_area_gpu = cuda.device_array((ntris, max_extent), dtype=_float, stream=stream)
                triangle_sample_variance_gpu = cuda.device_array((ntris, max_extent), dtype=_float,
                                                                  stream=stream)
                calcTriangleBinSurfaceArea[blocks_strided_sa, threads_strided_sa, stream](transmit_xyz_gpu,
                                                                                          tri_verts_gpu,
                                                                                          tri_idxes_gpu,
                                                                                          triangle_ranges_gpu,
                                                                                          triangle_bin_surface_area_gpu, 
                                                                                          params_gpu)
                calcTriangleSampleVariance[blocks_strided_sa, threads_strided_sa, stream](transmit_xyz_gpu, tri_verts_gpu,
                                                                                    tri_idxes_gpu, triangle_ranges_gpu,
                                                                                    triangle_bin_surface_area_gpu,
                                                                                    triangle_sample_variance_gpu,
                                                                                    az_gpu, el_gpu, params_gpu, target_variance)'''
                '''tri_var = triangle_sample_variance_gpu.copy_to_host(stream=stream)
                tri_sa = triangle_bin_surface_area_gpu.copy_to_host(stream=stream)
                check = tri_sa.max()'''

                # threads_strided3, blocks_strided3 = optimizeStridedThreadBlocks2d((npulses, ntris, max_extent))
                calcTriangleReturns[blocks_strided, threads_strided, stream](transmit_xyz_gpu, tri_verts_gpu,
                                                                             tri_idxes_gpu, tri_norm_gpu,
                                                                             tri_material_gpu, kd_tree_gpu, tri_box_gpu,
                                                         tri_box_key_gpu, pd_r_gpu, pd_i_gpu, receive_xyz_gpu, az_gpu,
                                                                             el_gpu, 1, rng_states, params_gpu)

                '''if debug:
                    debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                    debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                    debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))'''
                # We need to copy to host this way so we don't accidentally sync the streams
                pd_r_gpu.copy_to_host(pdr_tmp, stream=stream)
                pd_i_gpu.copy_to_host(pdi_tmp, stream=stream)
                del az_gpu, el_gpu, receive_xyz_gpu, pd_r_gpu, pd_i_gpu, transmit_xyz_gpu
    # cuda.synchronize()
    del tri_material_gpu
    del tri_box_key_gpu
    del tri_idxes_gpu
    del tri_verts_gpu
    del tri_box_gpu
    del kd_tree_gpu
    del tri_norm_gpu
    del params_gpu
    del rng_states
    # del sample_points_gpu

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


def detectPointsScene(scene, npoints: int, a_obs_pt: np.ndarray, bw_az, bw_el, pointing_az, pointing_el):
    '''l0 = np.array([0, 0, 1.])
    l1 = np.array([0, 1, 0.])
    for mesh in scene.meshes:
        points = mesh.vertices[mesh.tri_idx]
        rvals = np.linalg.norm(flight_pos[None, :] - (
                    r1[None, :] * l0[:, None] + r2[None, :] * l1[:, None] + r3[None, :] * (1 - l0 - l1)[:, None]),
                               axis=1)
        rbins = np.digitize(rvals, ranges)'''

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


def surfaceAreaHeuristic(tri_area: np.ndarray, centroids: np.ndarray, tri_bounds: np.ndarray, bounding_box: np.ndarray):
    best_score = np.inf
    best_split = (bounding_box[0, 0] + bounding_box[1, 0]) / 2
    best_bounds = [np.stack((np.array([best_split, bounding_box[0, 1], bounding_box[0, 2]]), bounding_box[1]), axis=0),
                   np.stack((bounding_box[0], np.array([best_split, bounding_box[1, 1], bounding_box[1, 2]])), axis=0)]

    best_axis = 0
    if centroids.shape[0] > 0:
        medes = np.median(centroids, axis=0)
        s_a = 2 * sum(
            a * b
            for a, b in itertools.combinations(
                np.diff(bounding_box, axis=0)[0], 2
            )
        )
        for ax in range(3):
            # 13 chosen based on empirical data from a ray-tracing book
            ax_poss_splits = np.linspace(bounding_box[0, ax], bounding_box[1, ax], 13)[1:-1]
            ax_poss_splits = np.concatenate((ax_poss_splits, [medes[ax]]))
            split_scores = np.zeros_like(ax_poss_splits)
            for idx, sp in enumerate(ax_poss_splits):
                lefties = sp < centroids[:, ax]
                righties = sp >= centroids[:, ax]
                if sum(lefties) == 0 or sum(righties) == 0:
                    score = np.inf
                else:
                    left_split = tri_bounds[lefties].max(axis=(0, 1)) - tri_bounds[lefties].min(axis=(0, 1))
                    right_split = tri_bounds[righties].max(axis=(0, 1)) - tri_bounds[righties].min(axis=(0, 1))
                    left_prob = (
                        2
                        * sum(
                            a * b
                            for a, b in itertools.combinations(left_split, 2)
                        )
                        / s_a
                    )
                    right_prob = (
                        2
                        * sum(
                            a * b
                            for a, b in itertools.combinations(right_split, 2)
                        )
                        / s_a
                    )
                    score = sum(tri_area[lefties]) * left_prob + sum(tri_area[righties]) * right_prob
                    if score < best_score:
                        best_score = score + 0.
                        best_bounds = [np.stack((tri_bounds[lefties].min(axis=(0, 1)),
                                                 tri_bounds[lefties].max(axis=(0, 1))), axis=0),
                                       np.stack((tri_bounds[righties].min(axis=(0, 1)),
                                                 tri_bounds[righties].max(axis=(0, 1))), axis=0)]
                        best_split = sp
                        best_axis = ax
                split_scores[idx] = score

    return best_bounds[0], best_bounds[1], best_split, best_axis




def genKDTree(bounding_box: np.ndarray, points: np.ndarray = None, min_tri_per_box: int = 64):
    verts = points.mean(axis=1)
    tri_area = .5 * np.linalg.norm(np.cross(points[:, 1, :] - points[:, 0, :], points[:, 2, :] - points[:, 0, :]),
                                   axis=1)
    centroids = np.mean(points, axis=1)
    tri_bounds = np.stack((points.min(axis=1), points.max(axis=1)), axis=1)
    depth = 0
    max_depth = 12
    vidx = np.zeros((verts.shape[0])).astype(int)
    tree_bounds = bounding_box.reshape((1, 2, 3))
    tri_box_idxes = np.zeros((points.shape[0], 1))

    ptx0_gpu = cuda.to_device(points[:, 0, 0].astype(_float))
    ptx1_gpu = cuda.to_device(points[:, 1, 0].astype(_float))
    ptx2_gpu = cuda.to_device(points[:, 2, 0].astype(_float))
    pty0_gpu = cuda.to_device(points[:, 0, 1].astype(_float))
    pty1_gpu = cuda.to_device(points[:, 1, 1].astype(_float))
    pty2_gpu = cuda.to_device(points[:, 2, 1].astype(_float))
    ptz0_gpu = cuda.to_device(points[:, 0, 2].astype(_float))
    ptz1_gpu = cuda.to_device(points[:, 1, 2].astype(_float))
    ptz2_gpu = cuda.to_device(points[:, 2, 2].astype(_float))

    while True:
        level = sum(2 ** np.arange(depth))
        next_level = sum(2 ** np.arange(depth + 1))

        threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((points.shape[0], next_level - level))
        kd_box_center = tree_bounds[level:next_level].mean(axis=1)
        kd_box_extent = (np.diff(tree_bounds[level:next_level], axis=1) * .5)[:, 0]
        kd_center_gpu = cuda.to_device(kd_box_center.astype(_float))
        kd_extent_gpu = cuda.to_device(kd_box_extent.astype(_float))
        ptidx_gpu = cuda.to_device(np.zeros((points.shape[0], kd_center_gpu.shape[0])).astype(bool))
        assignBoxPoints[blocks_strided, threads_strided](ptx0_gpu, ptx1_gpu, ptx2_gpu, pty0_gpu, pty1_gpu, pty2_gpu,
                                                         ptz0_gpu, ptz1_gpu, ptz2_gpu, kd_center_gpu,
                                                         kd_extent_gpu, ptidx_gpu)
        box_idxes = ptidx_gpu.copy_to_host()
        tri_box_idxes = np.concatenate((tri_box_idxes, box_idxes), axis=1)

        if (
            np.median(box_idxes.sum(axis=0)) <= min_tri_per_box
            or depth >= max_depth
        ):
            break
        depth += 1
        ntree_bounds = np.zeros((2**depth, 2, 3))
        for idx in range(level, next_level):
            act_pts = vidx == idx
            ntree_bounds[(idx - level) * 2], ntree_bounds[(idx - level) * 2 + 1], sp_pt, sp_axis = (
                surfaceAreaHeuristic(tri_area[act_pts], centroids[act_pts], tri_bounds[act_pts], tree_bounds[idx]))
            lefties = np.logical_and(sp_pt < verts[:, sp_axis], vidx == idx)
            righties = np.logical_and(sp_pt >= verts[:, sp_axis], vidx == idx)
            vidx[lefties] = idx * 2 + 1
            vidx[righties] = idx * 2 + 2
        tree_bounds = np.concatenate((tree_bounds, ntree_bounds), axis=0)
    return tree_bounds, tri_box_idxes[:, 1:]


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