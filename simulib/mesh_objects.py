import open3d as o3d
import numpy as np
from .mesh_functions import detectPoints, genOctree


class Mesh(object):

    def __init__(self, a_mesh: o3d.geometry.TriangleMesh, num_box_levels: int=4, material_sigmas: list=None,
                 material_kd: list=None, material_ks: list = None, use_box_pts: bool = True, octree_perspective: np.ndarray = None):
        # Generate bounding box tree
        mesh_tri_idx = np.asarray(a_mesh.triangles)
        mesh_vertices = np.asarray(a_mesh.vertices)
        mesh_normals = np.asarray(a_mesh.triangle_normals)
        mesh_tri_vertices = mesh_vertices[mesh_tri_idx]

        # Material triangle stuff
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

        # Generate octree and associate triangles with boxes
        aabb = a_mesh.get_axis_aligned_bounding_box()
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        root_box = np.array([min_bound, max_bound])
        boxes, mesh_box_idx = genOctree(root_box, num_box_levels, mesh_tri_vertices, octree_perspective, use_box_pts=use_box_pts)

        meshx, meshy = np.where(mesh_box_idx)
        alltri_idxes = np.array([[a, b] for a, b in zip(meshy, meshx)])
        sorted_tri_idx = alltri_idxes[np.argsort(alltri_idxes[:, 0])]
        sorted_tri_idx = sorted_tri_idx[sorted_tri_idx[:, 0] >= sum(8 ** n for n in range(num_box_levels - 1))]
        box_num, start_idxes = np.unique(sorted_tri_idx[:, 0], return_index=True)
        mesh_extent = np.diff(start_idxes, append=[sorted_tri_idx.shape[0]])
        mesh_idx_key = np.zeros((boxes.shape[0], 2)).astype(int)
        mesh_idx_key[box_num, 0] = start_idxes
        mesh_idx_key[box_num, 1] = mesh_extent


        # Set them all as properties of the object
        self.tri_idx = mesh_tri_idx
        self.vertices = mesh_vertices
        self.normals = mesh_normals
        self.materials = tri_material
        self.octree = boxes
        self.sorted_idx = sorted_tri_idx[:, 1]
        self.idx_key = mesh_idx_key
        self.center = a_mesh.get_center()
        self.ntri = mesh_tri_idx.shape[0]

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
        return detectPoints(self.octree, self.sorted_idx, self.idx_key, self.tri_idx, self.vertices, self.normals,
                            self.materials, sample_points, view_pos, bw_az, bw_el, pointing_az, pointing_el)



class Scene(object):
    
    def __init__(self, meshes: list[Mesh] = None):
        self.meshes = [] if meshes is None else meshes
        if meshes is not None:
            self.tree = np.zeros((len(meshes), 2, 3))
            for idx, m in enumerate(meshes):
                self.tree[idx] = m.octree[0]
        
        
        
    def __add__(self, a: list[Mesh]):
        ntree = np.zeros((len(a), 2, 3))
        for idx, m in enumerate(a):
            ntree[idx] = m.octree[0]
        self.tree = np.concatenate((self.tree, ntree), axis=0)
        self.meshes += a
        
    def __str__(self):
        return f'Scene with {len(self.meshes)} meshes.'
    
    def sample(self, sample_points: int, view_pos: np.ndarray, bw_az: float = None, bw_el: float = None):
        total_tris = sum(m.ntri for m in self.meshes)
        mesh_points = []
        for m in self.meshes:
            sp = int(m.ntri / total_tris * sample_points)
            # Calculate out the beamwidths so we don't waste GPU cycles on rays into space
            pvecs = m.center - view_pos
            pointing_az = np.arctan2(pvecs[:, 0], pvecs[:, 1])
            pointing_el = -np.arcsin(pvecs[:, 2] / np.linalg.norm(pvecs, axis=1))
            mesh_views = m.vertices[None, :, :] - view_pos[:, None, :]
            if bw_az is None:
                view_az = np.arctan2(mesh_views[:, :, 0], mesh_views[:, :, 1])
                view_el = -np.arcsin(mesh_views[:, :, 2] / np.linalg.norm(mesh_views, axis=2))
                bw_az = abs(pointing_az[:, None] - view_az).max()
                bw_el = abs(pointing_el[:, None] - view_el).max()
            mesh_points.append(detectPoints(m.octree, m.sorted_idx, m.idx_key, m.tri_idx, m.vertices, m.normals,
                                m.materials, sp, view_pos, bw_az, bw_el, pointing_az, pointing_el))
        return np.concatenate(mesh_points)