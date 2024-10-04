import cupy
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import minimize
from triangle.plot import vertices

from SDRParsing import load
from cuda_kernels import applyRadiationPatternCPU, calcOptRho, calcOptParams, getMaxThreads, assocTriangle
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
from simulation_functions import db, llh2enu, getElevationMap, azelToVec, enu2llh
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

fnme = '/home/jeff/SDR_DATA/RAW/08072024/SAR_08072024_111617.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / sdr[0].fc
ant_gain = 25
transmit_power = 100
upsample = 1
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
chip = background[206:206 + 512, 2000:2000 + 512]

tense = torch.tensor(chip, dtype=torch.float32,
                             device=segmenter.device).view(1, 1, 512, 512)
segment = segmenter(tense).cpu()[0, ...].data.numpy()

mesh = o3d.geometry.TriangleMesh()
material_ids = o3d.utility.IntVector()
colors = o3d.utility.Vector3dVector()

flight_path = rp.rxpos(rp.gpst)

# Get grid position of chip
chip_edge = bg.getPos(206 + 256, 2000 + 256)

gx, gy, gz = bg.getGrid(bg.origin, chip.shape[0] * pixel_to_m, chip.shape[1] * pixel_to_m, *chip.shape)
grid_id = np.zeros_like(gx) - 1
'''
============================================================
==================ROADS=====================================
'''
# Get the roads
roads = binary_dilation(binary_erosion(segment[2] > .9))

# Blob them
blabels = label(roads)

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
shadows = binary_dilation(binary_erosion(chip == 0))

# Blob them
blabels, nlabels = label(buildings, return_num=True)

# Run through and get shadows for height estimation
for n in tqdm(range(1, nlabels)):
    bding = blabels == n

    # Locate the shadow
    ypts, xpts = np.where(bding)
    bding_extent = ypts.max() - ypts.min()

    # Get connected shadows
    conn = label(shadows + bding)
    reconn = (conn == conn[bding].min()) ^ bding
    mhght = reconn.sum(axis=0)
    mhght = mhght[mhght > 0].mean()

    mhght = mhght * pixel_to_m if mhght > 0 else 5.
    # Calculate height from image angle
    flight_vec = flight_path - np.array([gx[xp, yp].mean(), gy[xp, yp].mean(), gz[xp, yp].mean()])
    perp_pt = flight_vec[np.linalg.norm(flight_vec, axis=1) == np.linalg.norm(flight_vec, axis=1).min()].flatten()
    perp_dep_ang = np.arcsin(perp_pt[2] / np.linalg.norm(perp_pt))
    bding_height = mhght / np.tan(perp_dep_ang)
    foreshortening = mhght / np.tan(perp_dep_ang)**2

    contours = np.concatenate(find_contours(bding.astype(int), .9))
    poly = Polygon(contours)
    poly_s = poly.simplify(2)
    bmm = extrude_polygon(poly_s, bding_height, engine='triangle')
    bmesh = o3d.geometry.TriangleMesh()
    bmesh.triangles = o3d.utility.Vector3iVector(bmm.faces)
    bmesh.vertices = o3d.utility.Vector3dVector(bmm.vertices)
    material_ids.extend(o3d.utility.IntVector([BUILDING_ID for _ in range(len(bmesh.triangles))]))
    grid_id[[[int(np.round(x)), int(np.round(y))] for x, y in zip(*poly_s.exterior.coords.xy)]] = BUILDING_ID
    colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[0, 0, 1.]]), len(bmesh.vertices), 0)))
    bmesh = bmesh.translate(np.array([gx[int(poly_s.centroid.x), int(poly_s.centroid.y)],
                                      gy[int(poly_s.centroid.x), int(poly_s.centroid.y)],
                                      gz[int(poly_s.centroid.x), int(poly_s.centroid.y)]]))
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
    shadow_block = chip[ymin:ypts.max(), xmin:xpts.max()]
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
        trunk = trunk.translate(np.array([0, 0, -mhght / 2]) +
                                np.array([gx[xd[n], yd[n]], gy[xd[n], yd[n]], gz[xd[n], yd[n]] + mhght]))
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
chirp_filt = np.fft.fft(sdr[0].cal_chirp, fft_len) * mfilt
threads_per_block = getMaxThreads()

face_points = np.asarray(mesh.vertices)
face_idxes = np.asarray(mesh.triangles)
face_materials = np.asarray(mesh.triangle_material_ids)
face_triangles = face_points[face_idxes]

n_samples = 1000

sample_points = mesh.sample_points_poisson_disk(n_samples)
snorms = np.asarray(sample_points.normals)
spts = np.asarray(sample_points.points)

print('Getting material triangle indexes...')
spts_gpu = cupy.array(spts, dtype=np.float32)
triangles_gpu = cupy.array(face_idxes, dtype=np.int32)
vertices_gpu = cupy.array(face_points, dtype=np.float32)
tri_idx_gpu = cupy.zeros(n_samples, dtype=np.int32)

bprun = (max(1, n_samples // threads_per_block[0] + 1),
             face_triangles.shape[0] // threads_per_block[1] + 1)
assocTriangle[bprun, threads_per_block](spts_gpu, triangles_gpu, vertices_gpu, tri_idx_gpu)

sample_triangles = tri_idx_gpu.get()
del spts_gpu
del triangles_gpu
del tri_idx_gpu
del vertices_gpu

road_scat = 5.
road_rcs = 5.
road_normal = np.array([0., 0., 1.])
tree_rcs = 10.
tree_scat = 10.

pt_ids = face_materials[sample_triangles.astype(int)]
is_opted = np.zeros_like(pt_ids)
opt_norm = np.zeros((n_samples, 3))
opt_norm[pt_ids == ROAD_ID] = np.array([0, 0, 1.])
opt_norm[pt_ids == TREE_ID] = snorms[pt_ids == TREE_ID]
opt_scat = np.ones(n_samples)
opt_rcs = np.ones(n_samples)

prog_bar = tqdm(total=n_samples)
boresight = rp.boresight(sdr[0].pulse_time).mean(axis=0)
pointing_az = np.arctan2(boresight[0], boresight[1])
pointing_el = -np.arcsin(boresight[2] / np.linalg.norm(boresight))

print('Optimizing points...')
while np.any(np.logical_not(is_opted)):
    all_pts = []
    inbeam_times = []
    while len(all_pts) == 0:
        ref_pt = np.random.choice(np.arange(n_samples)[np.logical_not(is_opted)])
        pt_pos = spts[ref_pt]

        vecs = np.array([pt_pos[0] - rp.txpos(sdr[0].pulse_time)[:, 0], pt_pos[1] - rp.txpos(sdr[0].pulse_time)[:, 1],
                         pt_pos[2] - rp.txpos(sdr[0].pulse_time)[:, 2]])
        pt_az = np.arctan2(vecs[0, :], vecs[1, :])
        pt_el = -np.arcsin(vecs[2, :] / np.linalg.norm(vecs, axis=0))
        inbeam_times = np.logical_and(abs(pt_az - pointing_az) < rp.az_half_bw, abs(pt_el - pointing_el) < rp.el_half_bw)

        access_pts = []
        for t in sdr[0].pulse_time[inbeam_times]:
            vecs = np.array([spts[:, 0] - rp.txpos(t)[0], spts[:, 1] - rp.txpos(t)[1],
                             spts[:, 2] - rp.txpos(t)[2]])
            bins = np.round((np.linalg.norm(vecs, axis=0) / c0 - 2 * near_range_s) * fs * upsample).astype(int)
            access_pts.append([a for a in zip(*np.where(bins == bins[ref_pt]))])
        all_pts = np.array(list({x for xs in access_pts for x in xs})).flatten()
    pre_opt_pts = all_pts[is_opted[all_pts].astype(bool)]
    to_opt = all_pts[np.logical_not(is_opted[all_pts].astype(bool))]
    nper = sum(inbeam_times)

    rho_matrix = cupy.zeros((nper, all_pts.shape[0]), dtype=np.complex128)
    rngs = cupy.zeros((nper, all_pts.shape[0]), dtype=np.float32)
    pvecs = cupy.zeros((*rho_matrix.shape, 3), dtype=np.float32)

    _, pdata = sdr.getPulses(sdr[0].frame_num[inbeam_times], 0)
    mfdata = np.fft.fft(pdata, fft_len, axis=0) * mfilt[:, None]
    updata = np.zeros((up_fft_len, mfdata.shape[1]), dtype=np.complex128)
    updata[:fft_len // 2, :] = mfdata[:fft_len // 2, :]
    updata[-fft_len // 2:, :] = mfdata[-fft_len // 2:, :]
    updata = np.fft.ifft(updata, axis=0)[:nsam * upsample, :].T

    updata_gpu = cupy.array(updata, dtype=np.complex128)
    spts_gpu = cupy.array(spts[all_pts], dtype=np.float32)
    source_gpu = cupy.array(rp.txpos(sdr[0].pulse_time[inbeam_times]), dtype=np.float32)
    pan_gpu = cupy.array(rp.pan(sdr[0].pulse_time[inbeam_times]), dtype=np.float32)
    tilt_gpu = cupy.array(rp.tilt(sdr[0].pulse_time[inbeam_times]), dtype=np.float32)

    bprun = (max(1, nper // threads_per_block[0] + 1),
             len(all_pts) // threads_per_block[1] + 1)

    calcOptParams[bprun, threads_per_block](spts_gpu, source_gpu, updata_gpu, near_range_s, fs * upsample, rho_matrix,
                                            pvecs, rngs)

    coeff = 1 / rngs.get() ** 4
    coeff = coeff / coeff.max()
    scaling_coeff = coeff.max() / abs(rho_matrix.get()).max()
    rhos_scaled = rho_matrix.get() * scaling_coeff

    # Get the expected values to optimize for
    # Roads, then buildings, then trees

    x0 = []
    bounds = []
    is_roads = False
    is_trees = False
    is_buildings = False
    if np.any(pt_ids[to_opt] == ROAD_ID):
        x0 = np.array([road_scat, road_rcs])
        is_roads = True
        bounds += [(1e-9, 15), (1e-9, 100)]
    if np.any(pt_ids[to_opt] == TREE_ID):
        x0 = np.concatenate((x0, np.array([tree_scat, tree_rcs])))
        bounds += [(1e-9, 15), (1e-9, 100)]
        is_trees = True
    if np.any(pt_ids[to_opt] == BUILDING_ID):
        for bd_pt in to_opt[pt_ids[to_opt] == BUILDING_ID]:
            bd_az = np.arctan2(snorms[bd_pt, 0], snorms[bd_pt, 1])
            bd_el = -np.arcsin(snorms[bd_pt, 2])
            bd_x0 = np.array([1., 100., bd_az, bd_el])
            x0 = np.concatenate((x0, bd_x0))
            bounds += [(1e-9, 15), (1e-9, 1e6), (bd_az - np.pi / 2, bd_az + np.pi / 2),
                       (bd_el - np.pi / 2, bd_el + np.pi / 2)]
        is_buildings = True

    rng_bins = ((rngs.get() / c0 - 2 * near_range_s) * fs * upsample).astype(int)

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
            not_opted = np.logical_and(pt_ids[all_pts] == BUILDING_ID, np.logical_not(is_opted[all_pts]))
            mnorm[not_opted] = azelToVec(x[is_roads * 2 + is_trees * 2 + 2::4],
                                                              x[
                                                              is_roads * 2 + is_trees * 2 + 3::4]).T
            mscat[not_opted] = x[is_roads * 2 + is_trees * 2::4]
            mrcs[not_opted] = x[is_roads * 2 + is_trees * 2 + 1::4]
            build_opted = np.logical_and(pt_ids[all_pts] == BUILDING_ID, is_opted[all_pts].astype(bool))
            mnorm[build_opted] = snorms[all_pts][build_opted]
            mscat[build_opted] = opt_scat[all_pts][build_opted]
            mrcs[build_opted] = opt_rcs[all_pts][build_opted]

        mnorm_gpu = cupy.array(mnorm, dtype=np.float32)
        mscat_gpu = cupy.array(mscat, dtype=np.float32)
        mrcs_gpu = cupy.array(mrcs, dtype=np.float32)
        pd_r = cupy.zeros(nsam, dtype=np.float64)
        pd_i = cupy.zeros(nsam, dtype=np.float64)
        calcOptRho[bprun, threads_per_block](pan_gpu, tilt_gpu, pd_r, pd_i, near_range_s, fs, rp.az_half_bw,
                                             rp.el_half_bw, mrcs_gpu, pvecs, mnorm_gpu, mscat_gpu, rngs,
                                             2 * np.pi / wavelength)
        x_hat = np.fft.fft(pd_r.get() + 1j * pd_i.get(), fft_len) * chirp_filt
        upx = np.zeros(up_fft_len, dtype=np.complex128)
        upx[:fft_len // 2] = x_hat[:fft_len // 2]
        upx[-fft_len // 2:] = x_hat[-fft_len // 2:]
        upx = np.fft.ifft(upx)[:nsam * upsample]

        return np.linalg.norm(rhos_scaled - upx[rng_bins] * scaling_coeff)

    opt_x = minimize(minfunc, x0, bounds=bounds)

    if is_roads:
        opt_scat[all_pts[pt_ids[all_pts] == ROAD_ID]] = opt_x['x'][0]
        opt_rcs[all_pts[pt_ids[all_pts] == ROAD_ID]] = opt_x['x'][1]
        road_scat = opt_x['x'][0]
        road_rcs = opt_x['x'][1]
    if is_trees:
        opt_scat[all_pts[pt_ids[all_pts] == TREE_ID]] = opt_x['x'][is_roads * 2 + 0]
        opt_rcs[all_pts[pt_ids[all_pts] == TREE_ID]] = opt_x['x'][is_roads * 2 + 1]
        tree_scat = opt_x['x'][is_roads * 2 + 0]
        tree_rcs = opt_x['x'][is_roads * 2 + 1]
    if is_buildings:
        opt_norm[to_opt[pt_ids[to_opt] == BUILDING_ID]] = azelToVec(opt_x['x'][is_roads * 2 + is_trees * 2 + 2::4],
                                                          opt_x['x'][
                                                          is_roads * 2 + is_trees * 2 + 3::4]).T
        opt_scat[to_opt[pt_ids[to_opt] == BUILDING_ID]] = opt_x['x'][is_roads * 2 + is_trees * 2::4]
        opt_rcs[to_opt[pt_ids[to_opt] == BUILDING_ID]] = opt_x['x'][is_roads * 2 + is_trees * 2 + 1::4]
    is_opted[to_opt] = True
    prog_bar.update(len(to_opt))

opt_pcd = o3d.geometry.PointCloud()
opt_pcd.points = sample_points.points
opt_pcd.normals = o3d.utility.Vector3dVector(opt_norm)

# o3d.visualization.draw_plotly([pcd])
o3d.visualization.draw_geometries([mesh, opt_pcd])
