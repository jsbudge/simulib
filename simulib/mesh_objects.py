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

        aabb = a_mesh.get_axis_aligned_bounding_box()
        max_bound = aabb.get_max_bound()
        min_bound = aabb.get_min_bound()
        root_box = np.array([min_bound, max_bound])
        boxes = genOctree(root_box, num_box_levels, mesh_tri_vertices if use_box_pts else None, mesh_normals if use_box_pts else None, octree_perspective)

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




def getPts(points, normals, octree, p):
    poss_tris = (
                np.all(np.logical_and(points[:, :, 0] >= octree[p, 0, 0], points[:, :, 0] <= octree[p, 1, 0]), axis=1) &
                np.all(np.logical_and(points[:, :, 1] >= octree[p, 0, 1], points[:, :, 1] <= octree[p, 1, 1]), axis=1) &
                np.all(np.logical_and(points[:, :, 2] >= octree[p, 0, 2], points[:, :, 2] <= octree[p, 1, 2]), axis=1))
    pidx = np.arange(points.shape[0])
    if sum(poss_tris) == 0:
        return [], []
    pidx = pidx[poss_tris]
    bpts = points
    # Bounding Box test
    '''poss_tris = (np.any(np.logical_and(points[:, :, 0] >= octree[p, 0, 0], points[:, :, 0] <= octree[p, 1, 0]), axis=1) &
                 np.any(np.logical_and(points[:, :, 1] >= octree[p, 0, 1], points[:, :, 1] <= octree[p, 1, 1]), axis=1) &
                 np.any(np.logical_and(points[:, :, 2] >= octree[p, 0, 2], points[:, :, 2] <= octree[p, 1, 2]), axis=1))
    if sum(poss_tris) == 0:
        return [], []
    pidx = pidx[poss_tris]
    bpts = points[poss_tris]
    bn = normals[poss_tris]
    c = (octree[p, 0] + octree[p, 1]) / 2
    extent = (octree[p, 1] - octree[p, 0]) / 2
    tvs = bpts - c[None, None, :]
    ax = np.eye(3)
    for v, n in itertools.product(range(3), range(3)):
        vn = n + 1 if n < 2 else 0
        axis = np.cross(ax[v], tvs[:, vn, :] - tvs[:, n, :])
        pp = np.sum(tvs * axis[:, :, None], axis=1)
        r = sum(e * abs(np.sum(a * axis, axis=1)) for e, a in zip(extent, ax))
        poss_tris = np.max(np.array([-np.max(pp, axis=1), np.min(pp, axis=1)]), axis=0) - r <= 0
        if sum(poss_tris) == 0:
            return [], []
        bpts = bpts[poss_tris]
        bn = bn[poss_tris]
        tvs = bpts - c[None, None, :]
        pidx = pidx[poss_tris]

    # Test the normal last
    pp = np.sum(tvs * bn[:, :, None], axis=1)
    r = sum(e * abs(np.sum(a * bn, axis=1)) for e, a in zip(extent, ax))
    poss_tris = np.max(np.array([-np.max(pp, axis=1), np.min(pp, axis=1)]), axis=0) - r <= 0
    pidx = pidx[poss_tris]'''
    return bpts[poss_tris], pidx


def reassign_pts(npo, oct_sect):
    '''for n in range(len(npo) - 1):
        for b in range(n + 1, len(npo)):
            if len(npo[n][0]) == 0 or len(npo[b][0]) == 0:
                continue
            # Check if boxes overlap at all
            if np.all([oct_sect[n, 1, q] >= oct_sect[b, 0, q] and oct_sect[b, 1, q] >= oct_sect[n, 0, q] for q in range(3)]):
                if npo[n][0].shape[0] < npo[b][0].shape[0]:
                    dupes = np.logical_not([nn in npo[n][1] for nn in npo[b][1]])
                    npo[b] = (npo[b][0][dupes], npo[b][1][dupes])
                else:
                    dupes = np.logical_not([nn in npo[b][1] for nn in npo[n][1]])
                    npo[n] = (npo[n][0][dupes], npo[n][1][dupes])'''
    return [n[0] for n in npo]



