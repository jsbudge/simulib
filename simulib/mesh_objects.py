from functools import cached_property, singledispatch
import open3d as o3d
import numpy as np
from .mesh_functions import detectPointsScene, genKDTree, _float, readVTC
from sklearn.cluster import AgglomerativeClustering


class Mesh(object):

    def __init__(self, a_mesh: o3d.geometry.TriangleMesh, material_emissivity: list=None,
                 material_sigma: list=None, max_tris_per_split: int = 64):
        # Generate bounding box tree
        mesh_tri_idx = np.asarray(a_mesh.triangles)
        mesh_vertices = np.asarray(a_mesh.vertices)
        mesh_normals = np.asarray(a_mesh.triangle_normals)
        mesh_tri_vertices = mesh_vertices[mesh_tri_idx]

        # Material triangle stuff
        if material_emissivity is None:
            print('Could not extrapolate sigmas, setting everything to one.')
            mesh_sigmas = np.ones(len(a_mesh.triangles)) * 1e6
        else:
            mesh_sigmas = np.array([material_emissivity[i] for i in np.asarray(a_mesh.triangle_material_ids)])

        if material_sigma is None:
            mesh_kd = np.ones(len(a_mesh.triangles)) * .0017
        else:
            mesh_kd = np.array([material_sigma[i] for i in np.asarray(a_mesh.triangle_material_ids)])

        tri_material = np.concatenate([mesh_sigmas.reshape((-1, 1)),
                                       mesh_kd.reshape((-1, 1))], axis=1)

        # Generate octree and associate triangles with boxes
        aabb = a_mesh.get_axis_aligned_bounding_box()
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


        # Set them all as properties of the object
        self.tri_idx = mesh_tri_idx.astype(np.int32)
        self.vertices = mesh_vertices.astype(_float)
        self.normals = mesh_normals.astype(_float)
        self.materials = tri_material.astype(_float)
        self.bvh = tree_bounds.astype(_float)
        self.bounding_box = root_box.astype(_float)
        self.leaf_list = sorted_tri_idx[:, 1].astype(np.int32)
        self.leaf_key = mesh_idx_key.astype(np.int32)
        self.center = a_mesh.get_center().astype(_float)
        self.ntri = mesh_tri_idx.shape[0]
        self.bvh_levels = num_box_levels
        # self.source_mesh = a_mesh

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


class VTCMesh(Mesh):

    def __init__(self, a_filepath: str):
        scat_data, angles = readVTC(a_filepath)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(scat_data[:, :3])
        pc.estimate_normals()

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector([.1, 10.]))
        mesh.triangle_material_ids = o3d.utility.IntVector([0 for _ in range(len(mesh.triangles))])

        super().__init__(mesh, material_emissivity=[1e6], material_sigma=[.0017])



class Scene(object):
    tree = list()
    
    def __init__(self, meshes: list[Mesh] = None):
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


    @property
    def bounding_box(self):
        return np.array([[np.min(np.array([m.bounding_box[0, :] for m in self.meshes]), axis=0)],
                         [np.max(np.array([m.bounding_box[1, :] for m in self.meshes]), axis=0)]]).squeeze(1)

    @property
    def center(self):
        return np.mean([np.array(s.center) for s in self.meshes], axis=0)