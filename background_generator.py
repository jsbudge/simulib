import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import minimize

from SDRParsing import load
from cuda_kernels import applyRadiationPatternCPU
from grid_helper import SDREnvironment
from models import ImageSegmenter
from platform_helper import SDRPlatform
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.measure import label, find_contours
from skimage.morphology import medial_axis
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import open3d as o3d
from simulation_functions import db, llh2enu, getElevationMap, azelToVec
from pywt import wavedec2
from trimesh.creation import extrude_polygon
import plotly.io as pio
pio.renderers.default = 'browser'

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180
ROAD_ID = 0
BUILDING_ID = 1
TREE_ID = 2

fnme = '/home/jeff/SDR_DATA/ARCHIVE/07082024/SAR_07082024_112333.sar'
sdr = load(fnme, import_pickle=False, progress_tracker=True)
wavelength = c0 / sdr[0].fc
ant_gain = 25
transmit_power = 100
upsample = 1
nper = 64
pixel_to_m = .25

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
    rp.getRadarParams(2., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)

'''
====================SEGMENTATION=========================
'''
device = o3d.core.Device("CPU:0")

# Load the segmentation model
with open('./segmenter_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
segmenter = ImageSegmenter(**param_dict['model_params'], label_sz=5, params=param_dict)
print('Setting up model...')
segmenter.load_state_dict(torch.load('./model/inference_model.state'))
segmenter.to('cuda:0')

png_fnme = '/home/jeff/repo/simulib/data/base_SAR_07082024_112333.png'


background = np.array(Image.open(png_fnme)) / 65535.
chip = background[:512, :512]

segment = np.zeros((5, *background.shape))

for x in tqdm(range(0, background.shape[0] - 512, 256)):
    for y in range(0, background.shape[1] - 512, 256):
        tense = torch.tensor(background[x:x + 512, y:y + 512], dtype=torch.float32,
                             device=segmenter.device).view(1, 1, 512, 512)
        segment[:, x:x + 512, y:y + 512] = segmenter(tense).cpu()[0, ...].data.numpy()

mesh = o3d.geometry.TriangleMesh()
material_ids = o3d.utility.IntVector()
colors = o3d.utility.Vector3dVector()

gx, gy, gz = bg.getGrid(bg.origin, background.shape[0] * pixel_to_m, background.shape[1] * pixel_to_m, *background.shape)
grid_id = np.zeros_like(gx) - 1
'''
============================================================
==================ROADS=====================================
'''
# Get the roads
roads = binary_dilation(binary_erosion(segment[2] > .9))

# Blob them
blabels, nlabels = label(roads, return_num=True)

# Take the roads and add them to the background
bcld = o3d.geometry.PointCloud()
xp, yp = np.where(blabels > 0)
bpts = np.array([gx[xp, yp], gy[xp, yp], gz[xp, yp]])
bcld.points = o3d.utility.Vector3dVector(bpts.T)
bcld.estimate_normals()
bmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(bcld, o3d.utility.DoubleVector([2., 20.]))
bmesh = bmesh.simplify_vertex_clustering(1.)
material_ids.extend(o3d.utility.IntVector([ROAD_ID for _ in range(len(bmesh.triangles))]))
colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[1., 0, 0]]), len(bmesh.vertices), 0)))
grid_id[xp, yp] = ROAD_ID
mesh += bmesh
print('Roads added to mesh.')

'''
============================================================
==================FIELDS====================================
'''
# Get the fields
'''fields = binary_dilation(binary_erosion(segment[3] > .9))

# Blob them
blabels = label(fields)

# Take the fields and add them to the background
bcld = o3d.geometry.PointCloud()
xp, yp = np.where(blabels > 0)
bpts = np.array([gx[xp, yp], gy[xp, yp], gz[xp, yp]])
bcld.points = o3d.utility.Vector3dVector(bpts.T)
bcld.estimate_normals()
bmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(bcld, o3d.utility.DoubleVector([2.]))
bmesh.triangle_uvs = o3d.utility.Vector2dVector(np.random.rand(len(bmesh.triangles), 2))
bmesh.triangle_material_ids = o3d.utility.IntVector([3 for _ in range(len(bmesh.triangles))])
bmesh = bmesh.paint_uniform_color([1., 1, 0])
mesh += bmesh
print('Fields added to mesh.')'''

'''
============================================================
=======================BUILDINGS============================
'''
# Get the buildings
buildings = binary_dilation(binary_erosion(segment[0] > .9))

# Blob them
blabels, nlabels = label(buildings, return_num=True)

# Run through and get shadows for height estimation
for n in tqdm(range(1, nlabels)):
    bding = blabels == n

    # Locate the shadow
    ypts, xpts = np.where(bding)
    bding_extent = ypts.max() - ypts.min()
    ymin = ypts.min() - bding_extent
    xmin = xpts.min()
    shadow_block = background[ymin:ypts.max(), xmin:xpts.max()]
    bding_block = bding[ymin:ypts.max(), xmin:xpts.max()]

    shadows = binary_dilation(binary_erosion(shadow_block == 0))

    # Get connected shadows
    conn = label(shadows + bding_block)
    reconn = (conn == conn[bding_block].min()) ^ bding_block
    mhght = 0
    for m in range(reconn.shape[1]):
        try:
            bmin = np.where(bding_block[:, m])[0].max()
            reconn[bmin:, m] = False
            mhght = max(mhght, sum(reconn[:, m]))
        except ValueError:
            continue

    mhght = mhght * pixel_to_m if mhght > 0 else 5.
    # Calculate height from image angle
    bding_height = mhght / np.tan(rp.dep_ang)
    foreshortening = mhght / np.tan(rp.dep_ang)**2

    contours = np.concatenate(find_contours(bding_block.astype(int), .9))
    poly = Polygon(contours)
    poly_s = poly.simplify(2)
    bmm = extrude_polygon(poly_s.convex_hull, bding_height, engine='triangle')
    bmesh = o3d.geometry.TriangleMesh()
    bmesh.triangles = o3d.utility.Vector3iVector(bmm.faces)
    bmesh.vertices = o3d.utility.Vector3dVector(bmm.vertices)
    material_ids.extend(o3d.utility.IntVector([BUILDING_ID for _ in range(len(bmesh.triangles))]))
    grid_id[[[int(np.round(x)), int(np.round(y))] for x, y in zip(*poly_s.exterior.coords.xy)]] = BUILDING_ID
    colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[0, 0, 1.]]), len(bmesh.vertices), 0)))
    bmesh = bmesh.translate(np.array([gx[int(poly_s.centroid.x + xmin), int(poly_s.centroid.y + ymin)],
                                      gy[int(poly_s.centroid.x + xmin), int(poly_s.centroid.y + ymin)],
                                      gz[int(poly_s.centroid.x + xmin), int(poly_s.centroid.y + ymin)]]))
    mesh += bmesh
print('Buildings added to mesh.')

'''
============================================================
=======================TREES============================
'''
# Get the trees
trees = binary_dilation(binary_erosion(segment[1] > .9))

# Blob them
blabels, nlabels = label(trees, return_num=True)

# Run through and get shadows for height estimation
for n in tqdm(range(1, nlabels)):
    bding = blabels == n

    # Locate the shadow
    ypts, xpts = np.where(bding)
    bding_extent = ypts.max() - ypts.min()
    ymin = ypts.min() - bding_extent
    xmin = xpts.min()
    shadow_block = background[ymin:ypts.max(), xmin:xpts.max()]
    bding_block = bding[ymin:ypts.max(), xmin:xpts.max()]

    shadows = binary_dilation(binary_erosion(shadow_block == 0))

    # Get connected shadows
    conn = label(shadows + bding_block)
    reconn = (conn == conn[bding_block].min()) ^ bding_block
    mhght = 0
    for m in range(reconn.shape[1]):
        try:
            bmin = np.where(bding_block[:, m])[0].max()
            reconn[bmin:, m] = False
            mhght = max(mhght, sum(reconn[:, m]))
        except ValueError:
            continue

    mhght = mhght * pixel_to_m if mhght > 0 else 3.
    # Calculate height from image angle
    bding_height = mhght / np.tan(rp.dep_ang)
    foreshortening = mhght / np.tan(rp.dep_ang)**2
    skeleton, dists = medial_axis(bding, return_distance=True)

    yd, xd = np.where(skeleton)
    sk_dists = dists[skeleton]

    tree_total = o3d.geometry.TriangleMesh()
    for n in range(0, len(xd), 5):
        nm = o3d.geometry.TriangleMesh.create_sphere(sk_dists[n] * pixel_to_m, create_uv_map=True)
        nm = nm.translate(np.array([gx[xd[n], yd[n]], gy[xd[n], yd[n]], gz[xd[n], yd[n]] + mhght]))
        tree_total += nm
    tree_total = tree_total.compute_convex_hull()[0]

    colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[0, 1., 0]]), len(tree_total.vertices), 0)))
    for n in range(0, len(xd), 5):
        trunk = o3d.geometry.TriangleMesh.create_cylinder(.5, mhght)
        colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[.66, .66, .66]]), len(trunk.vertices), 0)))
        trunk = trunk.translate(np.array([0, 0, -mhght / 2]) + np.array([gx[xd[n], yd[n]], gy[xd[n], yd[n]], gz[xd[n], yd[n]] + mhght]))
        tree_total += trunk
    material_ids.extend(o3d.utility.IntVector([TREE_ID for _ in range(len(tree_total.triangles))]))
    mesh += tree_total
    grid_id[xd, yd] = TREE_ID

print('Trees added to mesh.')

mesh = mesh.compute_triangle_normals()
mesh = mesh.compute_vertex_normals()
mesh.triangle_uvs = o3d.utility.Vector2dVector(np.random.rand(len(mesh.triangles), 2))
mesh.triangle_material_ids = material_ids
mesh.vertex_colors = colors

mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_unreferenced_vertices()
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.merge_close_vertices(.1)

'''
====================================================================
===============OPTIMIZATION ROUTINES================================
====================================================================
'''

face_points = np.asarray(mesh.vertices)
face_idxes = np.asarray(mesh.triangles)
face_materials = np.asarray(mesh.triangle_material_ids)
face_triangles = face_points[face_idxes]

n_samples = 10000

sample_points = mesh.sample_points_poisson_disk(n_samples)
snorms = np.asarray(sample_points.normals)
spts = np.asarray(sample_points.points)
sample_triangles = np.zeros(n_samples) - 1

# Get point's triangle
for n in tqdm(range(n_samples)):
    # Edge 1
    ba = face_triangles[:, 1, :] - face_triangles[:, 0, :]
    m = (np.sum((spts[n, :] - face_triangles[:, 0, :]) * ba, axis=1) / np.linalg.norm(ba, axis=1)**2)[:, None] * ba + face_triangles[:, 0, :]
    s = np.sum((spts[n, :] - m) * (face_triangles[:, 2, :] - m), axis=1)
    final_valids = s > 0
    # Edge 2
    ba = face_triangles[final_valids, 2, :] - face_triangles[final_valids, 0, :]
    m = (np.sum((spts[n, :] - face_triangles[final_valids, 0, :]) * ba, axis=1) / np.linalg.norm(ba, axis=1) ** 2)[:,
        None] * ba + face_triangles[final_valids, 0, :]
    s = np.sum((spts[n, :] - m) * (face_triangles[final_valids, 1, :] - m), axis=1)
    final_valids[final_valids] = s > 0
    # Edge 3
    ba = face_triangles[final_valids, 2, :] - face_triangles[final_valids, 1, :]
    m = (np.sum((spts[n, :] - face_triangles[final_valids, 1, :]) * ba, axis=1) / np.linalg.norm(ba, axis=1) ** 2)[:,
        None] * ba + face_triangles[final_valids, 1, :]
    s = np.sum((spts[n, :] - m) * (face_triangles[final_valids, 0, :] - m), axis=1)
    final_valids[final_valids] = s > 0
    sample_triangles[n] = np.where(final_valids)[0][0]

road_scat = 0.
road_rcs = 0.
road_normal = np.array([0., 0., 1.])
tree_rcs = 0.
tree_scat = 0.

pt_ids = face_materials[sample_triangles.astype(int)]
is_opted = np.zeros_like(pt_ids)
opt_norm = np.zeros((n_samples, 3))
opt_norm[pt_ids == ROAD_ID] = np.array([0, 0, 1.])
opt_norm[pt_ids == TREE_ID] = snorms[pt_ids == TREE_ID]
opt_scat = np.ones(n_samples)
opt_rcs = np.ones(n_samples)

while np.any(np.logical_not(is_opted)):
    print(f'{sum(is_opted) / n_samples * 100.}% complete.')
    all_pts = []
    pt_it = 0
    while len(all_pts) == 0:
        ref_pt = np.random.choice(np.arange(n_samples)[np.logical_not(is_opted)])
        pt_pos = spts[ref_pt]

        rngs = np.zeros(sdr[0].nframes)
        pvecs = np.zeros((sdr[0].nframes, 3))
        pmods = np.ones(sdr[0].nframes)
        norm = np.array([0., 0.])
        pt_it += 1

        access_pts = []
        for t in sdr[0].pulse_time[np.round(np.linspace(0, sdr[0].nframes, nper, endpoint=False)).astype(int)]:
            vecs = np.array([spts[:, 0] - rp.txpos(t)[0], spts[:, 1] - rp.txpos(t)[1],
                             spts[:, 2] - rp.txpos(t)[2]])
            bins = np.round((np.linalg.norm(vecs, axis=0) / c0 - 2 * near_range_s) * fs * upsample).astype(int)
            access_pts.append([a for a in zip(*np.where(bins == bins[ref_pt])) if not is_opted[a]])
        all_pts = np.array(list({x for xs in access_pts for x in xs})).flatten()

    rho_matrix = np.zeros((all_pts.shape[0], nper)).astype(np.complex128)
    rngs = np.zeros((all_pts.shape[0], nper))
    pmods = np.zeros((all_pts.shape[0], nper))
    pvecs = np.zeros((*rho_matrix.shape, 3))

    for n in range(nper):
        vec = np.array([spts[all_pts, 0] - rp.txpos(sdr[0].pulse_time[n])[0],
                        spts[all_pts, 1] - rp.txpos(sdr[0].pulse_time[n])[1],
                        spts[all_pts, 2] - rp.txpos(sdr[0].pulse_time[n])[2]]).T
        tmp_rngs = np.linalg.norm(vec, axis=1)
        rng_bin = np.round((tmp_rngs / c0 - 2 * near_range_s) * fs * upsample).astype(int)
        valids = np.logical_and(rng_bin >= 0, rng_bin < nsam * upsample)
        rng_bin = rng_bin[valids]
        _, pdata = sdr.getPulses([sdr[0].frame_num[n]], 0)
        mfdata = np.fft.fft(pdata, fft_len, axis=0) * mfilt[:, None]
        updata = np.zeros((up_fft_len, 1), dtype=np.complex128)
        updata[:fft_len // 2, :] = mfdata[:fft_len // 2, :]
        updata[-fft_len // 2:, :] = mfdata[-fft_len // 2:, :]
        updata = np.fft.ifft(updata, axis=0)[:nsam * upsample, :].T
        dmag = updata[0, rng_bin]
        rho_matrix[valids, n] = dmag
        rngs[:, n] = tmp_rngs
        pvecs[:, n, :] = vec / tmp_rngs[:, None]
        azes = np.arctan2(pvecs[:, n, 0], pvecs[:, n, 1])
        eles = -np.arcsin(pvecs[:, n, 2])
        pmods[:, n] = [applyRadiationPatternCPU(eles[i], azes[i], rp.pan(sdr[0].pulse_time[n]),
                                                rp.tilt(sdr[0].pulse_time[n]), rp.pan(sdr[0].pulse_time[n]),
                                                rp.tilt(sdr[0].pulse_time[n]), rp.az_half_bw, rp.el_half_bw)
                       for i in range(rho_matrix.shape[0])]

    coeff = 1 / rngs ** 4
    coeff = coeff / coeff.max()
    rhos_scaled = rho_matrix / abs(rho_matrix).max() * coeff.max()

    # Get the expected values to optimize for
    # Roads, then buildings, then trees

    x0 = []
    bounds = []
    is_roads = False
    is_trees = False
    is_buildings = False
    if np.any(pt_ids[all_pts] == ROAD_ID):
        x0 = np.ones(2)
        is_roads = True
        bounds += [(1e-9, 5), (1e-9, 100)]
    if np.any(pt_ids[all_pts] == TREE_ID):
        x0 = np.concatenate((x0, np.ones(2)))
        bounds += [(1e-9, 5), (1e-9, 100)]
        is_trees = True
    if np.any(pt_ids[all_pts] == BUILDING_ID):
        x0 = np.concatenate((x0, np.ones(4 * sum(pt_ids[all_pts] == BUILDING_ID))))
        for _ in range(sum(pt_ids[all_pts] == BUILDING_ID)):
            bounds += [(1e-9, 5), (1e-9, 100), (1e-9, 2 * np.pi), (-np.pi / 2, np.pi / 2)]
        is_buildings = True

    def minfunc(x):
        mnorm = np.zeros((len(all_pts), 3))
        mscat = np.ones(len(all_pts))
        mrcs = np.ones(len(all_pts))
        if is_roads:
            mnorm[pt_ids[all_pts] == ROAD_ID] = np.array([0, 0, 1.])
            mscat[pt_ids[all_pts] == ROAD_ID] = x[0]
            mrcs[pt_ids[all_pts] == ROAD_ID] = x[1]
        if is_trees:
            mnorm[pt_ids[all_pts] == TREE_ID] = snorms[all_pts[pt_ids[all_pts] == TREE_ID]]
            mscat[pt_ids[all_pts] == TREE_ID] = x[is_roads * 2 + 0]
            mrcs[pt_ids[all_pts] == TREE_ID] = x[is_roads * 2 + 1]
        if is_buildings:
            mnorm[pt_ids[all_pts] == BUILDING_ID] = azelToVec(x[is_roads * 2 + is_trees * 2 + 2::4],
                                                              x[
                                                              is_roads * 2 + is_trees * 2 + 3::4]).T
            mscat[pt_ids[all_pts] == BUILDING_ID] = x[is_roads * 2 + is_trees * 2::4]
            mrcs[pt_ids[all_pts] == BUILDING_ID] = x[is_roads * 2 + is_trees * 2 + 1::4]
        xr = np.sum(pvecs * (pvecs - 2 * np.einsum('ji,jk->jik',
                                                   np.sum(pvecs * mnorm[:, None, :], axis=2), mnorm)), axis=2)
        xr[xr < 0] = 0
        x_hat = (coeff * xr / mscat[:, None] ** 2 * np.exp(-(xr**2) / (2 * mscat[:, None] ** 2)) *
                 np.exp(-1j * 2 * np.pi / wavelength * rngs))
        return np.linalg.norm(rhos_scaled - x_hat * mrcs[:, None])

    opt_x = minimize(minfunc, x0, bounds=bounds)

    if is_roads:
        opt_scat[all_pts[pt_ids[all_pts] == ROAD_ID]] = opt_x['x'][0]
        opt_rcs[all_pts[pt_ids[all_pts] == ROAD_ID]] = opt_x['x'][1]
    if is_trees:
        opt_scat[all_pts[pt_ids[all_pts] == TREE_ID]] = opt_x['x'][is_roads * 2 + 0]
        opt_rcs[all_pts[pt_ids[all_pts] == TREE_ID]] = opt_x['x'][is_roads * 2 + 1]
    if is_buildings:
        opt_norm[all_pts[pt_ids[all_pts] == BUILDING_ID]] = azelToVec(opt_x['x'][is_roads * 2 + is_trees * 2 + 2::4],
                                                          opt_x['x'][
                                                          is_roads * 2 + is_trees * 2 + 3::4]).T
        opt_scat[all_pts[pt_ids[all_pts] == BUILDING_ID]] = opt_x['x'][is_roads * 2 + is_trees * 2::4]
        opt_rcs[all_pts[pt_ids[all_pts] == BUILDING_ID]] = opt_x['x'][is_roads * 2 + is_trees * 2 + 1::4]
    is_opted[all_pts] = True

opt_pcd = o3d.geometry.PointCloud()
opt_pcd.points = sample_points.points
opt_pcd.normals = o3d.utility.Vector3dVector(opt_norm)

# o3d.visualization.draw_plotly([pcd])
o3d.visualization.draw_geometries([mesh, opt_pcd])
