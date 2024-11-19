import itertools
import multiprocessing as mp
import open3d as o3d
import functools
import numpy as np
from .mesh_functions import detectPoints

BOX_CUSHION = .01


class Mesh(object):

    def __init__(self, a_mesh: o3d.geometry.TriangleMesh, num_box_levels: int=4, material_sigmas: list=None,
                 material_kd: list=None, material_ks: list = None, use_box_pts: bool = True, octree_perspective: np.ndarray = None):
        # Generate bounding box tree
        mesh_tri_idx = np.asarray(a_mesh.triangles)
        mesh_vertices = np.asarray(a_mesh.vertices)
        mesh_normals = np.asarray(a_mesh.triangle_normals)
        mesh_tri_vertices = mesh_vertices[mesh_tri_idx]

        aabb = a_mesh.get_axis_aligned_bounding_box()
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        root_box = np.array([min_bound, max_bound])
        boxes = genOctree(root_box, num_box_levels, mesh_tri_vertices if use_box_pts else None, octree_perspective)

        mesh_box_idx = np.zeros((mesh_tri_idx.shape[0], len(boxes)), dtype=bool)
        # Get the box index for each triangle, for exclusion in the GPU calculations
        for b_idx, box in enumerate(boxes):
            is_inside = (box[0, 0] < mesh_tri_vertices[:, :, 0]) & (mesh_tri_vertices[:, :, 0] < box[1, 0])
            for dim in range(1, 3):
                is_inside = is_inside & (box[0, dim] < mesh_tri_vertices[:, :, dim]) & (
                            mesh_tri_vertices[:, :, dim] < box[1, dim])
            mesh_box_idx[:, b_idx] = np.any(is_inside, axis=1)
        meshx, meshy = np.where(mesh_box_idx)
        alltri_idxes = np.array([[a, b] for a, b in zip(meshy, meshx)])
        sorted_tri_idx = alltri_idxes[np.argsort(alltri_idxes[:, 0])]
        sorted_tri_idx = sorted_tri_idx[sorted_tri_idx[:, 0] >= sum(8 ** n for n in range(num_box_levels - 1))]
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

        # Set them all as properties of the object
        self.tri_idx = mesh_tri_idx
        self.vertices = mesh_vertices
        self.normals = mesh_normals
        self.materials = tri_material
        self.octree = boxes
        self.sorted_idx = sorted_tri_idx[:, 1]
        self.idx_key = mesh_idx_key
        self.center = a_mesh.get_center()

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

def splitBox(bounding_box, npo: np.ndarray = None, perspective: np.ndarray = None):
    splits = np.zeros((8, 2, 3))
    if npo is None:
        box_hull = np.array([bounding_box[0] - BOX_CUSHION, bounding_box.mean(axis=0),
                             bounding_box[1] + BOX_CUSHION])
    elif len(npo) > 0:
        box_hull = np.array([npo.min(axis=(0, 1)) - BOX_CUSHION, np.mean(npo, axis=(0, 1)), npo.max(axis=(0, 1)) + BOX_CUSHION])
    else:
        box_hull = np.zeros((3, 3))
    for bidx, (x, y, z) in enumerate(itertools.product(range(2), range(2), range(2))):
        splits[bidx, :] = np.array([[box_hull[x, 0], box_hull[y, 1], box_hull[z, 2]],
                                    [box_hull[x + 1, 0], box_hull[y + 1, 1], box_hull[z + 1, 2]]])
    if perspective is not None:
        split_mean = splits.mean(axis=1)
        dists = np.linalg.norm(split_mean - perspective, axis=1)
        dists[np.sum(split_mean, axis=1) == 0] = 0.
        splits = splits[np.argsort(dists)]
    return splits


def getPts(points, octree, p):
    return points[np.all(np.logical_and(points[:, :, 0] >= octree[p, 0, 0], points[:, :, 0] <= octree[p, 1, 0]), axis=1) &
                     np.all(np.logical_and(points[:, :, 1] >= octree[p, 0, 1], points[:, :, 1] <= octree[p, 1, 1]), axis=1) &
                     np.all(np.logical_and(points[:, :, 2] >= octree[p, 0, 2], points[:, :, 2] <= octree[p, 1, 2]), axis=1)]


def genOctree(bounding_box: np.ndarray, num_levels: int = 3, points: np.ndarray = None, perspective: np.ndarray = None):
    octree = np.zeros((sum(8**n for n in range(num_levels)), 2, 3))
    octree[0, ...] = bounding_box
    for level in range(num_levels - 1):
        level_idx = sum(8**l for l in range(level))
        next_level_idx = sum(8 ** l for l in range(level + 1))
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            if points is None:
                npo = [None for _ in range(level_idx, next_level_idx)]
            else:
                npo = [pool.apply(getPts, args=(points, octree, p)) for p in range(level_idx, next_level_idx)]
            results = [pool.apply(splitBox, args=(octree[p + level_idx], npo[p], perspective)) for p in range(8**level)]

        for idx, r in enumerate(results):
            octree[next_level_idx + idx * 8:next_level_idx + (idx + 1) * 8] = r
    return octree
