import itertools
import multiprocessing as mp
from numba import cuda
from .cuda_mesh_kernels import calcBounceLoop, calcBounceInit, calcOriginDirAtt, calcIntersectionPoints, \
    assignBoxPoints, determineSceneRayIntersections, calcReturnPower
from .cuda_triangle_kernels import calcBinTotalSurfaceArea, calcViewSamples
from .simulation_functions import azelToVec, factors
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np
import open3d as o3d
from .cuda_kernels import optimizeStridedThreadBlocks2d
import plotly.graph_objects as go
from tqdm import tqdm
from pathlib import Path
import pickle

c0 = 299792458.0
_float = np.float32

MULTIPROCESSORS = cuda.get_current_device().MULTIPROCESSOR_COUNT
THREADS_PER_BLOCK = 512
BLOCK_MULTIPLIER = 64


def loadTarget(fnme):
    with open(fnme, 'r') as f:
        targ_data = f.readlines()

    obj_fnme = targ_data[0].strip()
    material_key = {}
    for l in targ_data[1:]:
        parts = l.split(' ')
        material_key[parts[0]] = int(parts[1])
        material_key[int(parts[1])] = [float(parts[2]), float(parts[3])]
    if Path(obj_fnme).suffix == '.obj':
        build_mesh = _loadOBJ(obj_fnme, material_key)
    elif Path(obj_fnme).suffix == '.gltf':
        build_mesh = _loadGLTF(obj_fnme)

    build_mesh.compute_vertex_normals()
    build_mesh.compute_triangle_normals()
    build_mesh.normalize_normals()
    return build_mesh, material_key


def _loadGLTF(obj_fnme):
    return o3d.io.read_triangle_mesh(obj_fnme)



def _loadOBJ(obj_fnme, material_key):
    with open(obj_fnme, 'r') as f:
        lines = f.readlines()
    vertices = []
    vertex_normals = []
    vertex_textures = []
    o = {}
    curr_o = None
    curr_mat = 0
    faces = []
    face_normals = []
    face_mats = []
    for line in lines:
        if isinstance(line, str):
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('vn '):
                vertex_normals.append(list(map(float, line.split()[1:4])))
            elif line.startswith('vt '):
                vertex_textures.append(list(map(float, line.split()[1:4])))
            if line.startswith('f '):
                # Face data (e.g., f v1 v2 v3)
                parts = line.split()
                fces = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                try:
                    fc_norms = [int(p.split('/')[2]) - 1 for p in parts[1:]]
                except IndexError:
                    fc_norms = [0, 0, 0.]
                if len(fces) > 3:
                    for n in range(len(fces) - 2):
                        faces.append([fces[0], fces[n + 1], fces[n + 2]])
                        face_normals.append([fc_norms[0], fc_norms[n + 1], fc_norms[n + 2]])
                        face_mats.append(curr_mat)
                else:
                    faces.append(fces)
                    face_normals.append(fc_norms)
                    face_mats.append(curr_mat)
            elif line.startswith('o '):
                if curr_o is not None and len(faces) > 0:
                    o[curr_o] = [np.array(faces), np.array(face_normals), np.array(face_mats)]
                faces = []
                face_normals = []
                face_mats = []
                curr_o = line[2:].strip()
                curr_mat = material_key[curr_o]
            elif line.startswith('usemtl '):
                curr_mat = material_key[line.split(' ')[1].strip()]
    o[curr_o] = [np.array(faces), np.array(face_normals), np.array(face_mats)]
    vertex_normals = np.array(vertex_normals)
    # Get material numbers correct
    mat_nums = []
    triangles = []
    tri_norms = []
    for i, (key, val) in enumerate(o.items()):
        mat_nums.append(val[2])
        triangles.append(val[0])
        # tri_norms.append(vertex_normals[val[1]].mean(axis=1))
    # tri_norms = np.concatenate(tri_norms)
    # tri_norms = tri_norms / np.linalg.norm(tri_norms, axis=1)[:, None]
    build_mesh = o3d.geometry.TriangleMesh()
    build_mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    build_mesh.triangles = o3d.utility.Vector3iVector(np.concatenate(triangles).astype(int))
    build_mesh.triangle_material_ids = o3d.utility.IntVector(list(np.concatenate(mat_nums).astype(int)))
    # build_mesh.triangle_normals = o3d.utility.Vector3dVector(tri_norms)
    build_mesh.compute_vertex_normals()
    build_mesh.compute_triangle_normals()
    build_mesh.normalize_normals()
    return build_mesh


def readCombineMeshFile(fnme: str, points: int=100000, scale: float=None) -> o3d.geometry.TriangleMesh:
    full_mesh = o3d.io.read_triangle_model(fnme)
    mesh = o3d.geometry.TriangleMesh()
    num_tris = [len(me.mesh.triangles) for me in full_mesh.meshes]
    mids = []
    for me in full_mesh.meshes:
        mesh += me.mesh
        mids += [me.material_idx for _ in range(len(me.mesh.triangles))]
    if sum(num_tris) > points:
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

    params_gpu = cuda.to_device(np.array([2 * np.pi / (c0 / fc), near_range_s, fs, bw_az, bw_el,
                                          radar_equation_constant, use_supersampling]).astype(_float))

    pd_r = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    pd_i = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    sa_bins = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    counts = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with cuda.defer_cleanup():
        for stream, tx, rx, az, el, pdr_tmp, pdi_tmp, sa, count in zip(streams, tx_pos, rx_pos, pan, tilt, pd_r, pd_i, sa_bins, counts):
            with cuda.pinned(tx, rx, az, el, pdr_tmp, pdi_tmp, sa, count):
                receive_xyz_gpu = cuda.to_device(rx, stream=stream)
                transmit_xyz_gpu = cuda.to_device(tx, stream=stream)
                ray_origin_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, stream=stream)
                ray_dir_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, stream=stream)
                ray_power_gpu = cuda.device_array((npulses, npoints), dtype=_float, stream=stream)
                ray_distance_gpu = cuda.to_device((np.zeros((npulses, npoints)) + np.inf).astype(_float), stream=stream)
                az_gpu = cuda.to_device(az, stream=stream)
                el_gpu = cuda.to_device(el, stream=stream)
                pd_r_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                pd_i_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)

                sa_bins_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                count_bins_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)

                # Calculate out the attenuation from the beampattern
                calcOriginDirAtt[blocks_strided, threads_strided, stream](transmit_xyz_gpu, sample_points_gpu, az_gpu,
                                                                          el_gpu, params_gpu, ray_dir_gpu,
                                                                          ray_origin_gpu, ray_power_gpu)

                for bounce in range(num_bounces):
                    ray_intersect_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, stream=stream)
                    ray_bounce_gpu = cuda.device_array((npulses, npoints, 3), dtype=_float, stream=stream)
                    ray_bounce_power_gpu = cuda.device_array((npulses, npoints), dtype=_float, stream=stream)
                    for mesh in scene.meshes:
                        threads_strided_sa, blocks_strided_sa = optimizeStridedThreadBlocks2d(
                            (mesh.tri_idx.shape[0], nsam))
                        tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float), stream=stream)
                        tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32), stream=stream)
                        tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float), stream=stream)
                        tri_material_gpu = cuda.to_device(mesh.materials.astype(_float), stream=stream)
                        tri_box_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32), stream=stream)
                        tri_box_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32), stream=stream)

                        # This is for optimization purposes
                        kd_tree_gpu = cuda.to_device(mesh.bvh.astype(_float))
                        if bounce == 0:
                            calcBinTotalSurfaceArea[blocks_strided_sa, threads_strided_sa, stream](transmit_xyz_gpu, tri_verts_gpu,
                                                                                           tri_idxes_gpu,
                                                                                           sa_bins_gpu, params_gpu)

                        determineSceneRayIntersections[blocks_strided, threads_strided, stream](
                            ray_origin_gpu, ray_intersect_gpu, ray_dir_gpu, ray_distance_gpu, ray_bounce_gpu, ray_power_gpu,
                            ray_bounce_power_gpu, kd_tree_gpu,
                            tri_box_gpu, tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu, tri_norm_gpu, tri_material_gpu,
                            params_gpu)
                        del tri_material_gpu, tri_box_key_gpu, tri_idxes_gpu, tri_verts_gpu, tri_box_gpu, kd_tree_gpu, tri_norm_gpu
                    calcReturnPower[blocks_strided, threads_strided, stream](ray_intersect_gpu, ray_distance_gpu,
                                                                     ray_bounce_power_gpu, pd_r_gpu,
                                                                     pd_i_gpu,
                                                                     count_bins_gpu, receive_xyz_gpu,
                                                                     az_gpu, el_gpu, params_gpu)
                    ray_power_gpu = ray_bounce_power_gpu
                    ray_dir_gpu = ray_bounce_gpu
                    ray_origin_gpu = ray_intersect_gpu
                    if debug:
                        debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                        debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                        debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))


                # We need to copy to host this way so we don't accidentally sync the streams
                pd_r_gpu.copy_to_host(pdr_tmp, stream=stream)
                pd_i_gpu.copy_to_host(pdi_tmp, stream=stream)
                sa_bins_gpu.copy_to_host(sa, stream=stream)
                count_bins_gpu.copy_to_host(count, stream=stream)

    # cuda.synchronize()

    # Combine chunks in comprehension
    for c in counts:
        c[c == 0.] += 1
    # final_rp = [pr + 1j * pi for pr, pi in zip(pd_r, pd_i)]
    final_rp = [s * (pr + 1j * pi) / n for pr, pi, s, n in zip(pd_r, pd_i, sa_bins, counts)]
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
    threads_strided_sa, blocks_strided_sa = optimizeStridedThreadBlocks2d(
        (mesh.tri_idx.shape[0], npulses))

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
    sa_bins = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    counts = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with (cuda.defer_cleanup()):
        for stream, tx, rx, az, el, pdr_tmp, pdi_tmp, sa, count in zip(streams, tx_pos, rx_pos, pan, tilt, pd_r, pd_i, sa_bins, counts):
            with cuda.pinned(tx, rx, az, el, pdr_tmp, pdi_tmp, sa, count):
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

                sa_bins_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                count_bins_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)

                calcBinTotalSurfaceArea[blocks_strided_sa, threads_strided_sa](transmit_xyz_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                                         sa_bins_gpu, params_gpu)
                # Calculate out the attenuation from the beampattern
                calcOriginDirAtt[blocks_strided, threads_strided, stream](transmit_xyz_gpu, sample_points_gpu, az_gpu, el_gpu,
                                                           params_gpu, ray_dir_gpu, ray_origin_gpu, ray_power_gpu)
                # Since we know the first ray can see the receiver, this is a special call to speed things up
                calcBounceInit[blocks_strided, threads_strided, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                         ray_power_gpu, kd_tree_gpu, tri_box_gpu,
                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                         count_bins_gpu, receive_xyz_gpu,
                                                         az_gpu, el_gpu, params_gpu, conical_sampling_gpu)
                if debug:
                    debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                    debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                    debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                if num_bounces > 1:
                    for _ in range(1, num_bounces):
                        calcBounceLoop[blocks_strided, threads_strided, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                                         ray_power_gpu, kd_tree_gpu, tri_box_gpu,
                                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                                         count_bins_gpu, receive_xyz_gpu, az_gpu, el_gpu, params_gpu)
                        if debug:
                            debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                            debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                            debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                # We need to copy to host this way so we don't accidentally sync the streams
                pd_r_gpu.copy_to_host(pdr_tmp, stream=stream)
                pd_i_gpu.copy_to_host(pdi_tmp, stream=stream)
                sa_bins_gpu.copy_to_host(sa, stream=stream)
                count_bins_gpu.copy_to_host(count, stream=stream)
                del ray_power_gpu, ray_origin_gpu, ray_dir_gpu, ray_distance_gpu, az_gpu, el_gpu, receive_xyz_gpu, \
                pd_r_gpu, pd_i_gpu, transmit_xyz_gpu, count_bins_gpu, sa_bins_gpu
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
    for c in counts:
        c[c == 0.] += 1
    final_rp = [s * (pr + 1j * pi) / n for pr, pi, s, n in zip(pd_r, pd_i, sa_bins, counts)]
    # final_rp = [pr + 1j * pi for pr, pi in zip(pd_r, pd_i)]
    if debug:
        return final_rp, debug_rays, debug_raydirs, debug_raypower
    else:
        return final_rp


def loadMesh(fnme: str):
    with open(fnme, 'rb') as f:
        m = pickle.load(f)
    return m


def getMeshErrorBounds(mesh, sampled_points: int | np.ndarray, tx_pos: list[np.ndarray], rx_pos: list[np.ndarray],
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
    threads_strided_sa, blocks_strided_sa = optimizeStridedThreadBlocks2d(
        (mesh.tri_idx.shape[0], nsam))

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
    sa_bins = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]
    counts = [np.zeros((npulses, nsam), dtype=_float) for _ in pan]

    # Use defer_cleanup to make sure things run asynchronously
    with (cuda.defer_cleanup()):
        for stream, tx, rx, az, el, pdr_tmp, pdi_tmp, sa, count in zip(streams, tx_pos, rx_pos, pan, tilt, pd_r, pd_i, sa_bins, counts):
            with cuda.pinned(tx, rx, az, el, pdr_tmp, pdi_tmp, sa, count):
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

                sa_bins_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)
                count_bins_gpu = cuda.device_array((npulses, nsam), dtype=_float, stream=stream)

                calcBinTotalSurfaceArea[blocks_strided_sa, threads_strided_sa](transmit_xyz_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                                         sa_bins_gpu, params_gpu)
                # Calculate out the attenuation from the beampattern
                calcOriginDirAtt[blocks_strided, threads_strided, stream](transmit_xyz_gpu, sample_points_gpu, az_gpu, el_gpu,
                                                           params_gpu, ray_dir_gpu, ray_origin_gpu, ray_power_gpu)
                # Since we know the first ray can see the receiver, this is a special call to speed things up
                calcBounceInit[blocks_strided, threads_strided, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                         ray_power_gpu, kd_tree_gpu, tri_box_gpu,
                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                         count_bins_gpu, receive_xyz_gpu,
                                                         az_gpu, el_gpu, params_gpu, conical_sampling_gpu)
                if debug:
                    debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                    debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                    debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                if num_bounces > 1:
                    for _ in range(1, num_bounces):
                        calcBounceLoop[blocks_strided, threads_strided, stream](ray_origin_gpu, ray_dir_gpu, ray_distance_gpu,
                                                                         ray_power_gpu, kd_tree_gpu, tri_box_gpu,
                                                                         tri_box_key_gpu, tri_verts_gpu, tri_idxes_gpu,
                                                                         tri_norm_gpu, tri_material_gpu, pd_r_gpu, pd_i_gpu,
                                                                         count_bins_gpu, receive_xyz_gpu, az_gpu, el_gpu, params_gpu)
                        if debug:
                            debug_rays.append(ray_origin_gpu.copy_to_host(stream=stream))
                            debug_raydirs.append(ray_dir_gpu.copy_to_host(stream=stream))
                            debug_raypower.append(ray_power_gpu.copy_to_host(stream=stream))
                # We need to copy to host this way so we don't accidentally sync the streams
                pd_r_gpu.copy_to_host(pdr_tmp, stream=stream)
                pd_i_gpu.copy_to_host(pdi_tmp, stream=stream)
                sa_bins_gpu.copy_to_host(sa, stream=stream)
                count_bins_gpu.copy_to_host(count, stream=stream)
                del ray_power_gpu, ray_origin_gpu, ray_dir_gpu, ray_distance_gpu, az_gpu, el_gpu, receive_xyz_gpu, \
                pd_r_gpu, pd_i_gpu, transmit_xyz_gpu, count_bins_gpu, sa_bins_gpu
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
    for c in counts:
        c[c == 0.] += 1
    final_rp = [s * (pr + 1j * pi) / n for pr, pi, s, n in zip(pd_r, pd_i, sa_bins, counts)]
    if debug:
        return final_rp, debug_rays, debug_raydirs, debug_raypower
    else:
        return final_rp


def detectPointsScene(scene, npoints: int, a_obs_pt: np.ndarray = None):
    points = np.zeros((1, 3))
    gridsz = 512
    if a_obs_pt is None:
        azes, eles = np.meshgrid(np.linspace(0, 2 * np.pi, 8), np.linspace(-np.pi / 2, np.pi / 2, 8))
        a_obs_pt = azelToVec(azes.flatten(), eles.flatten()).T * max(scene.bounding_box[1] - scene.bounding_box[0]) / (2 * np.tan(1 * np.pi / 180.)) + scene.center
        cosz = np.cos(azes.flatten())
        sinz = np.sin(azes.flatten())
        cosx = np.cos(eles.flatten())
        sinx = np.sin(eles.flatten())
        cosy = 1.
        siny = 0.
        x = scene.bounding_box[:, 0][:, None] - a_obs_pt[:, 0][None, :]
        y = scene.bounding_box[:, 1][:, None] - a_obs_pt[:, 1][None, :]
        z = scene.bounding_box[:, 2][:, None] - a_obs_pt[:, 2][None, :]
        dx = cosy * (sinz * y + cosz * x) - siny * z
        dy = sinx * (cosy * z + siny * (sinz * y + cosz * x)) + cosx * (cosz * y - sinz * x)
        dz = cosx * (cosy * z + siny * (sinz * y + cosz * x)) - sinx * (cosz * y - sinz * x)
        az_des = np.arctan2(dx, dy)
        az_des[az_des < 0] += 2 * np.pi
        el_des = -np.arcsin(dz / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        el_des[el_des < 0] += 2 * np.pi
        bw_az = az_des.max(axis=0) - az_des.min(axis=0)
        bw_az[bw_az > np.pi] = 2 * np.pi - bw_az[bw_az > np.pi]
        bw_el = el_des.max(axis=0) - el_des.min(axis=0)
        bw_el[bw_el > np.pi] = 2 * np.pi - bw_el[bw_el > np.pi]
        test_ranges = (max(scene.bounding_box[1] - scene.bounding_box[0]) /
                       np.array([2 * np.tan(bw_az), 2 * np.tan(bw_el), np.ones_like(bw_az) * 2 * np.tan(1 * np.pi / 180)])).T
        a_obs_pt = np.ascontiguousarray(azelToVec(azes.flatten(), eles.flatten()).T *
                                        test_ranges.max(axis=1)[:, None] + scene.center)
    else:
        azes = np.arctan2((a_obs_pt - scene.center)[:, 0], (a_obs_pt - scene.center)[:, 1])
        eles = -np.arcsin((a_obs_pt - scene.center)[:, 2] / np.linalg.norm(a_obs_pt - scene.center, axis=1))
    cosz = np.cos(azes.flatten())
    sinz = np.sin(azes.flatten())
    cosy = np.cos(eles.flatten())
    siny = np.sin(eles.flatten())
    cosx = 1.
    sinx = 0.
    x = scene.bounding_box[:, 0][:, None] - a_obs_pt[:, 0][None, :]
    y = scene.bounding_box[:, 1][:, None] - a_obs_pt[:, 1][None, :]
    z = scene.bounding_box[:, 2][:, None] - a_obs_pt[:, 2][None, :]
    dx = cosy * (sinz * y + cosz * x) - siny * z
    dy = sinx * (cosy * z + siny * (sinz * y + cosz * x)) + cosx * (cosz * y - sinz * x)
    dz = cosx * (cosy * z + siny * (sinz * y + cosz * x)) - sinx * (cosz * y - sinz * x)
    az_des = np.arctan2(dx, dy)
    az_des[az_des < 0] += 2 * np.pi
    el_des = -np.arcsin(dz / np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    el_des[el_des < 0] += 2 * np.pi
    bw_az = az_des.max() - az_des.min() if az_des.max() - az_des.min() < np.pi else 2 * np.pi - (
                az_des.max() - az_des.min())
    bw_el = el_des.max() - el_des.min() if el_des.max() - el_des.min() < np.pi else 2 * np.pi - (
                el_des.max() - el_des.min())
    params_gpu = cuda.to_device(np.array([bw_az, bw_el]).astype(_float))
    set_npoints = int(np.ceil(npoints / a_obs_pt.shape[0]))
    for t in range(a_obs_pt.shape[0]):
        transmit_xyz_gpu = cuda.to_device(a_obs_pt[t])
        threads_strided, blocks_strided = optimizeStridedThreadBlocks2d((gridsz, gridsz))
        rng_states = create_xoroshiro128p_states([gridsz], seed=np.random.randint(1, 10000))
        pixel_bins = np.ones((gridsz, gridsz))

        test = (pixel_bins / np.sum(pixel_bins)).ravel()
        fntest = np.zeros((*test.shape, 2))
        fntest[..., 0] = np.concatenate(
            (np.histogram(np.random.choice(len(test), npoints, p=test), np.arange(len(test)))[0],
             [0]))
        fntest[..., 1] = np.cumsum(fntest[..., 0]) * (fntest[..., 0] > 0)
        fntest = fntest.reshape((*pixel_bins.shape, 2))
        target_samples_gpu = cuda.to_device(fntest.astype(np.int32))
        for mesh in scene.meshes:
            tri_norm_gpu = cuda.to_device(mesh.normals.astype(_float))
            tri_idxes_gpu = cuda.to_device(mesh.tri_idx.astype(np.int32))
            tri_verts_gpu = cuda.to_device(mesh.vertices.astype(_float))
            leaf_list_gpu = cuda.to_device(mesh.leaf_list.astype(np.int32))
            leaf_key_gpu = cuda.to_device(mesh.leaf_key.astype(np.int32))

            # This is for optimization purposes
            kd_tree_gpu = cuda.to_device(mesh.bvh.astype(_float))
            center_pt_vec = mesh.center - a_obs_pt[t]
            az = np.float32(np.arctan2(center_pt_vec[0], center_pt_vec[1]))
            el = np.float32(-np.arcsin(center_pt_vec[2] / np.linalg.norm(center_pt_vec)))
            ray_points_gpu = cuda.to_device(np.zeros((npoints, 3)).astype(_float))
            # ray_points_gpu = cuda.device_array((npoints, 3), dtype=_float)

            calcViewSamples[blocks_strided, threads_strided](transmit_xyz_gpu, az, el, tri_verts_gpu, tri_idxes_gpu,
                                                            tri_norm_gpu, kd_tree_gpu, leaf_key_gpu, leaf_list_gpu,
                                                            target_samples_gpu, ray_points_gpu, rng_states, params_gpu)
            npts = np.zeros(ray_points_gpu.shape, dtype=_float)
            ray_points_gpu.copy_to_host(npts)
            # npts = ray_points_gpu.copy_to_host()
            npts = npts[np.sum(npts, axis=1) != 0]

            points = np.concatenate((points, npts[np.random.choice(np.arange(npts.shape[0]), set_npoints)]))
            del ray_points_gpu
        del tri_norm_gpu, tri_idxes_gpu, tri_verts_gpu, leaf_key_gpu, leaf_list_gpu, kd_tree_gpu
    del params_gpu
    return points[1:npoints + 1]


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


def getSceneFig(scene, triangle_colors=None, title='Title Goes Here', zrange=None):
    # Get for first mesh
    if zrange is None:
        zrange = [-30, 100]
    fig = go.Figure(data=[
        go.Mesh3d(
            x=scene.meshes[0].vertices[:, 0],
            y=scene.meshes[0].vertices[:, 1],
            z=scene.meshes[0].vertices[:, 2],
            # i, j and k give the vertices of triangles
            i=scene.meshes[0].tri_idx[:, 0],
            j=scene.meshes[0].tri_idx[:, 1],
            k=scene.meshes[0].tri_idx[:, 2],
            facecolor=triangle_colors if triangle_colors is not None else scene.meshes[0].normals,
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
                facecolor=triangle_colors if triangle_colors is not None else m.normals,
                showscale=True
            ))
    maxis = [
        max(m.vertices[:, 0].max() for m in scene.meshes),
        max(m.vertices[:, 1].max() for m in scene.meshes),
        max(m.vertices[:, 2].max() for m in scene.meshes),
    ]
    mixis = [
        min(m.vertices[:, 0].min() for m in scene.meshes),
        min(m.vertices[:, 1].min() for m in scene.meshes),
        min(m.vertices[:, 2].min() for m in scene.meshes),
    ]
    fig.update_layout(
        title=title,
        scene=dict(zaxis=dict(range=zrange), xaxis=dict(range=[mixis[0], maxis[0]]), yaxis=dict(range=[mixis[1], maxis[1]])),
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