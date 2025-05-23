from functools import cached_property, singledispatch

import open3d as o3d
import numpy as np
from .mesh_functions import detectPoints, detectPointsScene, genBVH, assocPointsWithOctree


class Mesh(object):

    def __init__(self, a_mesh: o3d.geometry.TriangleMesh, num_box_levels: int=4, material_emissivity: list=None,
                 material_sigma: list=None, use_box_pts: bool = True, octree_perspective: np.ndarray = None):
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
        # bvh, leaf_key, leaf_list = genBVH(root_box, num_box_levels, mesh_tri_vertices)
        bvh, mesh_box_idx = assocPointsWithOctree(Octree(num_box_levels, root_box), mesh_tri_vertices)

        meshx, meshy = np.where(mesh_box_idx)
        alltri_idxes = np.array([[a, b] for a, b in zip(meshy, meshx)])
        sorted_tri_idx = alltri_idxes[np.argsort(alltri_idxes[:, 0])]
        sorted_tri_idx = sorted_tri_idx[sorted_tri_idx[:, 0] >= sum(8 ** n for n in range(num_box_levels - 1))]
        box_num, start_idxes = np.unique(sorted_tri_idx[:, 0], return_index=True)
        mesh_extent = np.diff(start_idxes, append=[sorted_tri_idx.shape[0]])
        mesh_idx_key = np.zeros((bvh.shape[0], 2)).astype(int)
        mesh_idx_key[box_num, 0] = start_idxes
        mesh_idx_key[box_num, 1] = mesh_extent


        # Set them all as properties of the object
        self.tri_idx = mesh_tri_idx
        self.vertices = mesh_vertices
        self.normals = mesh_normals
        self.materials = tri_material
        self.bvh = bvh
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
        return np.array([[np.min(np.array([m.bvh[0, 0, :] for m in self.meshes]), axis=0)],
                         [np.max(np.array([m.bvh[0, 1, :] for m in self.meshes]), axis=0)]]).squeeze(1)

    @property
    def center(self):
        return np.mean([np.array(s.center) for s in self.meshes], axis=0)


class Octree(object):

    def __init__(self, depth, bounding_box):
        self.depth = depth
        self.octree = np.zeros((sum(8 ** n for n in range(depth)), 2, 3))
        self.octree[0, ...] = bounding_box
        self.mask = np.zeros(sum(8 ** n for n in range(depth - 1))).astype(np.uint8)
        self.pos = np.zeros((self.octree.shape[0], 3)).astype(np.uint32)
        self.extent = np.diff(bounding_box, axis=0)[0]
        self.center = np.mean(bounding_box, axis=0)
        self.lower = bounding_box[0]

        half_extent = self.extent / 2
        self.pos[0] = [0, 0, 0]
        for bidx in range(8):
            low_ext = np.array([(bidx >> 2) & 1, (bidx >> 1 & 1), (bidx >> 0 & 1)])
            self.octree[1 + bidx] = self.octree[0, 0] + half_extent * np.array([low_ext, low_ext + 1])
            self.pos[1 + bidx] = low_ext
        print('Building octree level ', end='')
        for level in range(1, depth):
            print(f'{level}...', end='')
            level_idx = sum(8 ** l for l in range(level))
            next_level_idx = sum(8 ** l for l in range(level + 1))
            if level < depth - 1:
                half_extent = self.extent / (2 * 2 ** level)
                for idx in range(level_idx, next_level_idx):
                    for bidx in range(8):
                        low_ext = np.array([(bidx >> 2) & 1, (bidx >> 1 & 1), (bidx >> 0 & 1)])
                        self.octree[next_level_idx + (idx - level_idx) * 8 + bidx] = self.octree[
                                                                                    idx, 0] + half_extent * np.array(
                            [low_ext, low_ext + 1])
                        self.pos[next_level_idx + (idx - level_idx) * 8 + bidx] = self.pos[idx] * 2 + low_ext

    def __sizeof__(self):
        return self.octree.shape

    @property
    def shape(self):
        return self.octree.shape

    def __getitem__(self, item):
        return self.octree[item]