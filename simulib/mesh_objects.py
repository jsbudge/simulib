from functools import cached_property, singledispatch
import open3d as o3d
import numpy as np
from .mesh_functions import detectPoints, detectPointsScene, genKDTree


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

        self.tri_area = .5 * np.linalg.norm(np.cross(mesh_tri_vertices[:, 1, :] - mesh_tri_vertices[:, 0, :], mesh_tri_vertices[:, 2, :] - mesh_tri_vertices[:, 0, :]),
                                       axis=1)


        # Set them all as properties of the object
        self.tri_idx = mesh_tri_idx
        self.vertices = mesh_vertices
        self.normals = mesh_normals
        self.materials = tri_material
        self.bvh = tree_bounds
        self.bounding_box = root_box
        self.leaf_list = sorted_tri_idx[:, 1]
        self.leaf_key = mesh_idx_key
        self.center = a_mesh.get_center()
        self.ntri = mesh_tri_idx.shape[0]
        self.bvh_levels = num_box_levels

    def sample(self, sample_points: int, view_pos: np.ndarray, bw_az: float = None, bw_el: float = None):
        # Calculate out the beamwidths so we don't waste GPU cycles on rays into space
        pvecs = self.center - view_pos
        pointing_az = np.arctan2(pvecs[:, 0], pvecs[:, 1])
        pointing_el = -np.arcsin(pvecs[:, 2] / np.linalg.norm(pvecs, axis=1))
        mesh_views = self.vertices[None, :, :] - view_pos[:, None, :]
        if bw_az is None:
            view_az = np.arctan2(mesh_views[:, :, 0], mesh_views[:, :, 1])
            view_el = -np.arcsin(mesh_views[:, :, 2] / np.linalg.norm(mesh_views, axis=2))
            bw_az = abs(pointing_az[:, None] - view_az).max()
            bw_el = abs(pointing_el[:, None] - view_el).max()
        return detectPoints(self.bvh, self.leaf_list, self.leaf_key, self.tri_idx, self.vertices, self.normals,
                            self.materials, sample_points, view_pos, bw_az, bw_el, pointing_az, pointing_el)



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
    
    def sample(self, sample_points: int, view_pos: np.ndarray, bw_az: float = None, bw_el: float = None):
        if bw_az is None:
            bw_az = 0.
            bw_el = 0.
            center = np.mean(self.bounding_box, axis=0)
            # Calculate out the beamwidths so we don't waste GPU cycles on rays into space
            pvecs = center - view_pos
            pointing_az = np.arctan2(pvecs[:, 0], pvecs[:, 1])
            pointing_el = -np.arcsin(pvecs[:, 2] / np.linalg.norm(pvecs, axis=1))
            mesh_views = np.vstack(self.tree)[None, :, :] - view_pos[:, None, :]
            view_az = np.arctan2(mesh_views[:, :, 0], mesh_views[:, :, 1])
            view_el = -np.arcsin(mesh_views[:, :, 2] / np.linalg.norm(mesh_views, axis=2))
            bw_az = max(bw_az, abs(pointing_az[:, None] - view_az).max())
            bw_el = max(bw_el, abs(pointing_el[:, None] - view_el).max())
        return detectPointsScene(self, sample_points, view_pos, bw_az, bw_el, pointing_az, pointing_el)

    @cached_property
    def bounding_box(self):
        return np.array([[np.min(np.array([m.bounding_box[0, :] for m in self.meshes]), axis=0)],
                         [np.max(np.array([m.bounding_box[1, :] for m in self.meshes]), axis=0)]]).squeeze(1)

    @property
    def center(self):
        return np.mean([np.array(s.center) for s in self.meshes], axis=0)