from functools import cached_property, singledispatch
import open3d as o3d
import numpy as np
from .mesh_functions import detectPointsScene, genKDTree, _float, readVTC
from sklearn.cluster import AgglomerativeClustering


class BaseMesh(object):
    bvh: np.ndarray
    bvh_levels: int
    leaf_key: np.ndarray
    leaf_list: np.ndarray
    bounding_box: np.ndarray
    tri_idx: np.ndarray
    vertices: np.ndarray
    normals: np.ndarray
    vertex_normals: np.ndarray
    materials: np.ndarray
    center: np.ndarray
    ntri: int
    is_dynamic: bool = False

    def __init__(self, center, mesh_tri_idx, mesh_vertices, mesh_normals, vertex_normals, triangle_material_ids=None,
                 material_emissivity=None, material_sigma=None):

        assert mesh_tri_idx.shape[-2] != 0, 'No triangle indexes found.'
        assert mesh_vertices.shape[-2] != 0, 'No vertices found.'
        assert mesh_normals.shape[-2] != 0, 'No triangle normals found.'

        # Material triangle stuff
        if material_emissivity is None:
            print('Could not extrapolate sigmas, setting everything to one.')
            mesh_sigmas = np.ones(mesh_tri_idx.shape[-2]) * 1e6
        else:
            mesh_sigmas = np.array([material_emissivity[i] for i in triangle_material_ids])

        if material_sigma is None:
            mesh_kd = np.ones(mesh_tri_idx.shape[-2]) * .0017
        else:
            mesh_kd = np.array([material_sigma[i] for i in triangle_material_ids])

        tri_material = np.concatenate([mesh_sigmas.reshape((-1, 1)),
                                       mesh_kd.reshape((-1, 1))], axis=1)
        assert tri_material.shape[0] == mesh_tri_idx.shape[0], 'Materials do not match triangles.'

        # Set them all as properties of the object
        self.tri_idx = mesh_tri_idx.astype(np.int32)
        self.vertices = mesh_vertices.astype(_float)
        self.normals = mesh_normals.astype(_float)
        self.vertex_normals = vertex_normals.astype(_float)
        self.materials = tri_material.astype(_float)
        self.center = center.astype(_float)
        self.ntri = mesh_tri_idx.shape[0]

    def set_bounding_box(self, a_box_source, max_tris_per_split):
        pass

    def sample(self, sample_points: int):
        sm = o3d.geometry.TriangleMesh()
        sm.triangles = o3d.utility.Vector3iVector(self.tri_idx)
        sm.vertices = o3d.utility.Vector3dVector(self.vertices)
        sm.triangle_normals = o3d.utility.Vector3dVector(self.normals)
        pc = sm.sample_points_poisson_disk(sample_points)
        return np.asarray(pc.points)

    def shift(self, new_center, relative=False):
        _shift = new_center if relative else new_center - self.center
        self.vertices += _shift
        self.center += _shift
        self.bounding_box += _shift
        self.bvh += _shift

    def rotate(self, rot_mat):
        self.vertices = self.vertices @ rot_mat
        self.center = self.center @ rot_mat
        self.normals = self.normals @ rot_mat

class TriangleMesh(BaseMesh):

    def __init__(self, a_mesh: o3d.geometry.TriangleMesh, material_emissivity: list = None, material_sigma: list = None,
                 max_tris_per_split: int = 64):

        # Generate bounding box tree

        mesh_tri_idx = np.asarray(a_mesh.triangles)
        mesh_vertices = np.asarray(a_mesh.vertices)
        a_mesh = a_mesh.compute_vertex_normals()
        mesh_normals = np.asarray(a_mesh.triangle_normals)
        vertex_normals = np.asarray(a_mesh.vertex_normals)

        super().__init__(a_mesh.get_center(), mesh_tri_idx, mesh_vertices, mesh_normals, vertex_normals,
                         np.asarray(a_mesh.triangle_material_ids), material_emissivity, material_sigma)

        self.set_bounding_box(a_mesh, max_tris_per_split)

    def set_bounding_box(self, a_box_source, max_tris_per_split):
        mesh_tri_vertices = self.vertices[self.tri_idx]
        # Generate octree and associate triangles with boxes
        aabb = a_box_source.get_axis_aligned_bounding_box()
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        root_box = np.array([min_bound, max_bound])
        tree_bounds, mesh_box_idx = genKDTree(root_box, mesh_tri_vertices, max_tris_per_split)
        num_box_levels = int(np.log2(tree_bounds.shape[0]) + 1)

        meshx, meshy = np.where(mesh_box_idx)
        alltri_idxes = np.array([[a, b] for a, b in zip(meshy, meshx)])
        sorted_tri_idx = alltri_idxes[np.argsort(alltri_idxes[:, 0])]
        sorted_tri_idx = sorted_tri_idx[sorted_tri_idx[:, 0] >= sum(2 ** n for n in range(num_box_levels - 1))]
        box_num, start_idxes = np.unique(sorted_tri_idx[:, 0], return_index=True)
        mesh_extent = np.diff(start_idxes, append=[sorted_tri_idx.shape[0]])
        mesh_idx_key = np.zeros((tree_bounds.shape[0], 3)).astype(int)
        mesh_idx_key[box_num, 0] = start_idxes
        mesh_idx_key[box_num, 1] = mesh_extent

        self.bvh = tree_bounds.astype(_float)
        self.bounding_box = root_box.astype(_float)
        self.leaf_list = sorted_tri_idx[:, 1].astype(np.int32)
        self.leaf_key = mesh_idx_key.astype(np.int32)
        self.bvh_levels = num_box_levels


class VTCMesh(BaseMesh):

    def __init__(self, a_filepath: str):
        scat_data, angles = readVTC(a_filepath)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(scat_data[:, :3])
        pc.estimate_normals()

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector([.1, 10.]))
        mesh.triangle_material_ids = o3d.utility.IntVector([0 for _ in range(len(mesh.triangles))])

        super().__init__(mesh, material_emissivity=[1e6], material_sigma=[.0017])


class OceanMesh(BaseMesh):

    def __init__(self, center, mesh_tri_idx, mesh_heights, mesh_x, mesh_y):
        # Get the sampled ocean points into something we can use
        mesh_vertices = np.stack([np.array([mesh_x, mesh_y, m]).T for m in mesh_heights])

        # Run through all the time triangles and get normals for the vertices
        mesh_tri_vertices = mesh_vertices[:, mesh_tri_idx]
        e0 = mesh_tri_vertices[:, :, 1] - mesh_tri_vertices[:, :, 0]
        e1 = mesh_tri_vertices[:, :, 2] - mesh_tri_vertices[:, :, 0]
        mesh_normals = np.cross(e0, e1)
        mesh_normals = mesh_normals / np.linalg.norm(mesh_normals, axis=2)[..., None]

        # Make sure all normals point upwards, since the simulated ocean never has breaking waves
        mesh_normals[mesh_normals[:, :, 2] < 0.] = -mesh_normals[mesh_normals[:, :, 2] < 0.]

        vertex_normals = np.zeros_like(mesh_vertices)
        for vidx in range(vertex_normals.shape[1]):
            # Get all triangles associated with this vertex
            vertex_normals[:, vidx, :] = mesh_normals[:, np.any(mesh_tri_idx == vidx, axis=1)].mean(axis=1)

        super().__init__(center, mesh_tri_idx, mesh_vertices, mesh_normals, vertex_normals, np.zeros((mesh_tri_idx.shape[0],)).astype(int),
                 [1e2], [.0001])

        self.set_bounding_box(np.stack([np.stack([v.min(axis=0), v.max(axis=0)]) for v in mesh_vertices]), 64)
        self.is_dynamic = True

    def set_bounding_box(self, a_box_source, max_tris_per_split):
        root_box = np.stack([np.min(a_box_source, axis=(0, 1)), np.max(a_box_source, axis=(0, 1))])
        mesh_tri_vertices = self.vertices[0, self.tri_idx]
        # Generate octree and associate triangles with boxes
        tree_bounds, mesh_box_idx = genKDTree(root_box, mesh_tri_vertices, max_tris_per_split, n_ax_split=2)
        num_box_levels = int(np.log2(tree_bounds.shape[0]) + 1)

        meshx, meshy = np.where(mesh_box_idx)
        alltri_idxes = np.array([[a, b] for a, b in zip(meshy, meshx)])
        sorted_tri_idx = alltri_idxes[np.argsort(alltri_idxes[:, 0])]
        sorted_tri_idx = sorted_tri_idx[sorted_tri_idx[:, 0] >= sum(2 ** n for n in range(num_box_levels - 1))]
        box_num, start_idxes = np.unique(sorted_tri_idx[:, 0], return_index=True)
        mesh_extent = np.diff(start_idxes, append=[sorted_tri_idx.shape[0]])
        mesh_idx_key = np.zeros((tree_bounds.shape[0], 3)).astype(int)
        mesh_idx_key[box_num, 0] = start_idxes
        mesh_idx_key[box_num, 1] = mesh_extent

        self.bvh = tree_bounds.astype(_float)
        self.bounding_box = root_box.astype(_float)
        self.leaf_list = sorted_tri_idx[:, 1].astype(np.int32)
        self.leaf_key = mesh_idx_key.astype(np.int32)
        self.bvh_levels = num_box_levels
        '''self.bvh = np.stack([bbox, bbox, bbox]).astype(_float)
        self.bounding_box = bbox.astype(_float)
        self.leaf_list = np.concatenate((np.arange(self.tri_idx.shape[0]), np.arange(self.tri_idx.shape[0]), np.arange(self.tri_idx.shape[0]))).astype(np.int32)
        self.leaf_key = np.array([[0, self.tri_idx.shape[0], 0],
                                  [self.tri_idx.shape[0], self.tri_idx.shape[0], 0],
                                  [self.tri_idx.shape[0] * 2, self.tri_idx.shape[0], 0]]).astype(np.int32)
        self.bvh_levels = 2'''




class Scene(object):
    tree = list()
    
    def __init__(self, meshes: list[BaseMesh] = None):
        self.meshes = [] if meshes is None else meshes
        if meshes is not None:
            self.tree = np.zeros((len(meshes), 2, 3))
            for idx, m in enumerate(meshes):
                self.tree[idx] = m.bvh[0]
        

    @singledispatch
    def add(self, a):
        self.tree = np.concatenate((self.tree, np.expand_dims(a.bvh[0], 0)), axis=0) if len(
            self.tree) != 0 else np.expand_dims(a.bvh[0], 0)
        self.meshes += [a]

    @add.register
    def _(self, a: list):
        ntree = np.zeros((len(a), 2, 3))
        for idx, m in enumerate(a):
            ntree[idx] = m.bvh[0]
        self.tree = np.concatenate((self.tree, ntree), axis=0) if len(self.tree) != 0 else ntree
        self.meshes += a
        
    def __str__(self):
        return f'Scene with {len(self.meshes)} meshes.'
    
    def sample(self, sample_points: int, a_obs_pts: np.ndarray):
        return detectPointsScene(self, sample_points, a_obs_pts)

    '''def sample(self, sample_points: int):
        # return detectPointsScene(self, sample_points)
        sm = o3d.geometry.TriangleMesh()
        sm.triangles = o3d.utility.Vector3iVector(self.meshes[0].tri_idx)
        sm.vertices = o3d.utility.Vector3dVector(self.meshes[0].vertices)
        sm.triangle_normals = o3d.utility.Vector3dVector(self.meshes[0].normals)
        # pc = sm.compute_convex_hull()[0].sample_points_uniformly(sample_points)
        pc = sm.sample_points_uniformly(sample_points)
        return np.asarray(pc.points)'''''

    def shift(self, new_center, relative=False):
        _shift = new_center - self.center if relative else new_center
        for mesh in self.meshes:
            mesh.shift(_shift, relative)


    def rotate(self, rot_xyz: np.ndarray):
        x, y, z = rot_xyz
        rot_mat = np.array([[np.cos(y) * np.cos(z), np.sin(x) * np.sin(y) * np.cos(z) - np.cos(x) * np.sin(z), np.cos(x) * np.sin(y) * np.cos(z) + np.sin(x) * np.sin(z)],
                            [np.cos(y) * np.sin(z), np.sin(x) * np.sin(y) * np.sin(z) + np.cos(x) * np.cos(z), np.cos(x) * np.sin(y) * np.sin(z) - np.sin(x) * np.cos(z)],
                            [-np.sin(y), np.sin(x) * np.cos(y), np.cos(x) * np.cos(y)]])
        for mesh in self.meshes:
            scene_center = self.center + 0.  # store the scene center, as the method re-calculates it from meshes
            mesh.shift(mesh.center - scene_center, False)
            mesh.rotate(rot_mat)
            mesh.shift(scene_center + mesh.center, False)

    def recalcKDTree(self, max_tris_per_split):
        for mesh in self.meshes:
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            new_mesh.triangles = o3d.utility.Vector3iVector(mesh.tri_idx)
            mesh.set_bounding_box(new_mesh, max_tris_per_split)


    @property
    def bounding_box(self):
        return np.array([[np.min(np.array([m.bounding_box[0, :] for m in self.meshes]), axis=0)],
                         [np.max(np.array([m.bounding_box[1, :] for m in self.meshes]), axis=0)]]).squeeze(1)

    @property
    def center(self):
        return np.mean([np.array(s.center) for s in self.meshes], axis=0)