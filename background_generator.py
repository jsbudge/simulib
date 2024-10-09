import cupy
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from triangle.plot import vertices
from scipy.spatial import Delaunay
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
FIELD_ID = 3
UNKNOWN_ID = 4

fnme = '/home/jeff/SDR_DATA/RAW/08072024/SAR_08072024_111617.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / sdr[0].fc
ant_gain = 25
transmit_power = 100
eff_aperture = 10. * 10.
upsample = 1
pixel_to_m = .25

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
    rp.getRadarParams(2., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)

# This is all the constants in the radar equation for this simulation
radar_coeff = transmit_power * 10**(ant_gain / 10) * eff_aperture / (4 * np.pi)**2

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
chip_pos = (206, 2000)
chip_shape = (768, 768)


background = np.array(Image.open(png_fnme)) / 65535.
chip = background[chip_pos[0]:chip_pos[0]+ chip_shape[0], chip_pos[1]:chip_pos[1] + chip_shape[1]]
segment = np.zeros((5, *chip.shape))

if chip.shape[0] > 512 or chip.shape[1] > 512:
    for x in range(0, chip.shape[0], 512):
        for y in range(0, chip.shape[1], 512):
            xrng = (x if x + 512 < chip.shape[0] else x - (x + 512 - chip.shape[0]), x + min(chip.shape[0] - x, 512))
            yrng = (y if y + 512 < chip.shape[1] else y - (y + 512 - chip.shape[1]), y + min(chip.shape[1] - y, 512))
            tense = torch.tensor(chip[xrng[0]:xrng[1], yrng[0]:yrng[1]], dtype=torch.float32,
                                         device=segmenter.device).view(1, 1, 512, 512)
            segment[:, xrng[0]:xrng[1], yrng[0]:yrng[1]] = segmenter(tense).cpu()[0, ...].data.numpy()

mesh = o3d.geometry.TriangleMesh()
material_ids = o3d.utility.IntVector()
colors = o3d.utility.Vector3dVector()

flight_path = rp.rxpos(rp.gpst)

# Get grid position of chip
chip_edge = bg.getPos(chip_pos[0] + chip_shape[0] // 2, chip_pos[1] + chip_shape[1] // 2)

gx, gy, gz = bg.getGrid(bg.origin, chip.shape[0] * pixel_to_m, chip.shape[1] * pixel_to_m, *chip.shape)
grid_id = np.zeros_like(gx) - 1
'''
============================================================
==================UNKNOWNS=====================================
'''
# Get the unclassified stuff
roads = binary_dilation(binary_erosion(segment[4] > .9))

# Blob them
blabels = label(roads)

# Take the roads and add them to the background
xp, yp = np.where(blabels > 0)
del_tri = Delaunay(np.array([xp, yp]).T)
del_idx = np.round(del_tri.points).astype(int).T
bmesh = o3d.geometry.TriangleMesh()
bmesh.vertices = o3d.utility.Vector3dVector(np.array([gx[*del_idx], gy[*del_idx], gz[*del_idx]]).T)
bmesh.triangles = o3d.utility.Vector3iVector(del_tri.simplices)
bmesh = bmesh.merge_close_vertices(2.)
bmesh = bmesh.remove_degenerate_triangles()
bmesh = bmesh.remove_duplicated_triangles()
material_ids.extend(o3d.utility.IntVector([UNKNOWN_ID for _ in range(len(bmesh.triangles))]))
colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[0., 0, 0]]), len(bmesh.vertices), 0)))
grid_id[xp, yp] = UNKNOWN_ID
mesh += bmesh
print('Unknowns added to mesh.')
'''
============================================================
==================ROADS=====================================
'''
# Get the roads
roads = binary_dilation(binary_erosion(segment[2] > .9))

# Blob them
blabels = label(roads)

# Take the roads and add them to the background
xp, yp = np.where(blabels > 0)
contours = np.concatenate(find_contours(blabels.astype(int), .9))
poly = Polygon(contours)
poly_s = poly.simplify(2)
del_tri = Delaunay(np.array(poly_s.exterior.xy).T)
del_idx = np.round(del_tri.points).astype(int).T
bmesh = o3d.geometry.TriangleMesh()
bmesh.vertices = o3d.utility.Vector3dVector(np.array([gx[*del_idx], gy[*del_idx], gz[*del_idx]]).T)
bmesh.triangles = o3d.utility.Vector3iVector(del_tri.simplices)
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
fields = binary_dilation(binary_erosion(segment[3] > .9))

# Blob them
blabels, nlabels = label(fields, return_num=True)

# Take the fields and add them to the background
for n in range(1, nlabels):
    bfield = blabels == n
    xp, yp = np.where(bfield)
    contours = np.concatenate(find_contours(bfield.astype(int), .9))
    poly = Polygon(contours)
    poly_s = poly.simplify(2)
    del_tri = Delaunay(np.array(poly_s.exterior.xy).T)
    del_idx = np.round(del_tri.points).astype(int).T
    bmesh = o3d.geometry.TriangleMesh()
    bmesh.vertices = o3d.utility.Vector3dVector(np.array([gx[*del_idx], gy[*del_idx], gz[*del_idx]]).T)
    bmesh.triangles = o3d.utility.Vector3iVector(del_tri.simplices)
    material_ids.extend(o3d.utility.IntVector([FIELD_ID for _ in range(len(bmesh.triangles))]))
    colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[1., 1., 0]]), len(bmesh.vertices), 0)))
    grid_id[xp, yp] = FIELD_ID
    mesh += bmesh
print('Fields added to mesh.')

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
    grid_contours = np.array([gx[contours[:, 1].astype(int), contours[:, 0].astype(int)], gy[contours[:, 1].astype(int), contours[:, 0].astype(int)]]).T
    poly = Polygon(grid_contours)
    poly_s = poly.simplify(2)
    bmm = extrude_polygon(poly_s, bding_height, engine='triangle')
    bmesh = o3d.geometry.TriangleMesh()
    bmesh.triangles = o3d.utility.Vector3iVector(bmm.faces)
    bmesh.vertices = o3d.utility.Vector3dVector(bmm.vertices)
    material_ids.extend(o3d.utility.IntVector([BUILDING_ID for _ in range(len(bmesh.triangles))]))
    grid_id[bding] = BUILDING_ID
    colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[0, 0, 1.]]), len(bmesh.vertices), 0)))
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
    dmm = squareform(pdist(np.array([yd, xd]).T))
    dist_mat = abs(dmm - sk_dists)
    rep_pts = [np.where(dmm == dmm.max())[0][0]]
    taken = np.zeros(dist_mat.shape[0]).astype(bool)
    taken[rep_pts[-1]] = True

    while not np.all(taken):
        close_ones = np.argsort(dist_mat[rep_pts[-1]])
        idx = 0
        close = 0
        while idx < len(close_ones):
            if not taken[close_ones[idx]]:
                close = close_ones[idx]
                break
            idx += 1
        if idx >= len(close_ones):
            break
        rep_pts.append(close)
        taken[dist_mat[rep_pts[-2]] <= sk_dists[rep_pts[-2]]] = True



    tree_total = o3d.geometry.TriangleMesh()
    for n in rep_pts:
        nm = o3d.geometry.TriangleMesh.create_sphere(sk_dists[n] * pixel_to_m, create_uv_map=True)
        nm = nm.translate(np.array([gx[xd[n], yd[n]], gy[xd[n], yd[n]], gz[xd[n], yd[n]] + mhght]))
        tree_total += nm
    tree_total = tree_total.compute_convex_hull()[0]
    tree_total = tree_total.merge_close_vertices(2.)
    tree_total = tree_total.remove_degenerate_triangles()
    tree_total = tree_total.remove_duplicated_triangles()

    colors.extend(o3d.utility.Vector3dVector(np.repeat(np.array([[0, 1., 0]]), len(tree_total.vertices), 0)))
    for n in rep_pts:
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

'''mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_unreferenced_vertices()
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.merge_close_vertices(.1)'''

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
n_tri = len(mesh.triangles)

print('Getting material triangle indexes...')
triangles_gpu = cupy.array(face_idxes, dtype=np.int32)
vertices_gpu = cupy.array(face_points, dtype=np.float32)
road_scat = 5.
road_rcs = 5.
road_normal = np.array([0., 0., -1.])
tree_rcs = 10.
tree_scat = 10.
field_scat = 5.
field_rcs = 10.

opt_norm = np.zeros((n_tri, 3))
opt_scat = np.ones(n_tri)
opt_rcs = np.ones(n_tri)

prog_bar = tqdm(total=n_tri)
boresight = rp.boresight(sdr[0].pulse_time).mean(axis=0)
pointing_az = np.arctan2(boresight[0], boresight[1])
pointing_el = -np.arcsin(boresight[2] / np.linalg.norm(boresight))

# Get vertex classifications
vertex_ids = np.zeros(face_points.shape[0])
for i in range(face_idxes.shape[0]):
    vertex_ids[face_idxes[i]] = face_materials[i]


print('Optimizing points...')
times = [3788]
prog_bar.update(times[-1])
while times[-1] < sdr[0].nframes:
    opt_vars = 0
    rng_support = 0
    rng_span = (0, np.inf)
    can_opt_tri = np.zeros(face_idxes.shape[0]).astype(bool)
    can_opt_vert = np.zeros(face_points.shape[0]).astype(bool)
    while True:
        plat_pos = rp.txpos(sdr[0].pulse_time[times[-1]])
        vecs = np.array([face_points[:, 0] - plat_pos[0], face_points[:, 1] - plat_pos[1],
                         face_points[:, 2] - plat_pos[2]])
        pt_az = np.arctan2(vecs[0, :], vecs[1, :])
        pt_el = -np.arcsin(vecs[2, :] / np.linalg.norm(vecs, axis=0))
        can_opt_vert[np.logical_and(abs(pt_az - pointing_az) < rp.az_half_bw * 2,
                                    abs(pt_el - pointing_el) < rp.el_half_bw * 2)] = True
        can_opt_tri = np.logical_or(can_opt_tri, np.array([np.any([can_opt_vert[f] for f in i]) for i in face_idxes]))
        rng_bins = ((np.linalg.norm(vecs, axis=0) * 2 / c0 - 2 * near_range_s) * fs * upsample).astype(int)
        rng_span = (max(rng_span[0], rng_bins.max()), min(rng_span[1], rng_bins.min()))

        opt_num = (np.any(face_materials[can_opt_tri] == TREE_ID) * 2 +
                   np.any(face_materials[can_opt_tri] == ROAD_ID) * 2 +
                   np.any(face_materials[can_opt_tri] == FIELD_ID) * 2 +
                   np.sum(face_materials[can_opt_tri] == BUILDING_ID) * 2 +
                   np.sum(face_materials[can_opt_tri] == UNKNOWN_ID) * 2 + np.sum(vertex_ids[can_opt_vert] != TREE_ID))
        rng_support += len(list(set(rng_bins[can_opt_vert])))
        if opt_num < 30:
            times[-1] = times[-1] + 1
            rng_support = 0
            continue
        if opt_num > rng_support or rng_support < 150:
            times.append(times[-1] + 1)
        else:
            break

    nper = len(times)
    ntri = sum(can_opt_tri)
    nvert = sum(can_opt_vert)

    _, pdata = sdr.getPulses(sdr[0].frame_num[times], 0)
    mfdata = np.fft.fft(pdata, fft_len, axis=0) * mfilt[:, None]
    updata = np.zeros((up_fft_len, mfdata.shape[1]), dtype=np.complex128)
    updata[:fft_len // 2, :] = mfdata[:fft_len // 2, :]
    updata[-fft_len // 2:, :] = mfdata[-fft_len // 2:, :]
    updata = np.fft.ifft(updata, axis=0)[:nsam * upsample, :].T

    source_gpu = cupy.array(rp.txpos(sdr[0].pulse_time[times]), dtype=np.float32)
    pan_gpu = cupy.array(rp.pan(sdr[0].pulse_time[times]), dtype=np.float32)
    tilt_gpu = cupy.array(rp.tilt(sdr[0].pulse_time[times]), dtype=np.float32)

    bprun = (max(1, nper // threads_per_block[0] + 1),
             sum(can_opt_tri) // threads_per_block[1] + 1)

    # Get the expected values to optimize for
    # Roads, then buildings, then trees
    print('Building x0...')
    x0 = []
    bounds = []
    is_roads = False
    is_trees = False
    is_buildings = False
    is_fields = False
    is_unknown = False
    # Triangle optimization variables
    nvar = sum(can_opt_tri)
    if np.any(face_materials[can_opt_tri] == ROAD_ID):
        x0 = np.array([road_scat, road_rcs])
        is_roads = True
        bounds += [(1e-9, 15), (1e-9, 100)]
    if np.any(face_materials[can_opt_tri] == TREE_ID):
        x0 = np.concatenate((x0, np.array([tree_scat, tree_rcs])))
        bounds += [(1e-9, 15), (1e-9, 100)]
        is_trees = True
    if np.any(face_materials[can_opt_tri] == FIELD_ID):
        is_fields = True
        x0 = np.concatenate((x0, np.array([field_scat, field_rcs])))
        bounds += [(1e-9, 15), (1e-9, 100)]
    if np.any(face_materials[can_opt_tri] == BUILDING_ID):
        is_buildings = True
        for idx, bd_pt in enumerate(can_opt_tri):
            if bd_pt:
                bd_x0 = np.array([opt_scat[idx], opt_rcs[idx]])
                x0 = np.concatenate((x0, bd_x0))
                bounds += [(1e-9, 15.), (1e-9, 1e6)]
    if np.any(face_materials[can_opt_tri] == UNKNOWN_ID):
        is_unknown = True
        for idx, bd_pt in enumerate(can_opt_tri):
            if bd_pt:
                bd_x0 = np.array([opt_scat[idx], opt_rcs[idx]])
                x0 = np.concatenate((x0, bd_x0))
                bounds += [(1e-9, 15.), (1e-9, 1e6)]

    # Vertex optimization variables
    x0 = np.concatenate((x0, face_points[np.logical_and(vertex_ids != TREE_ID, can_opt_vert), 2]))
    bounds += [(f[2] - 10., f[2] + 10) for f in face_points[np.logical_and(vertex_ids != TREE_ID, can_opt_vert)]]

    def minfunc(x):
        mscat = np.zeros(ntri)
        mrcs = np.zeros(ntri)
        mz = np.zeros(nvert)
        vertex_start = (is_roads * 2 + is_trees * 2 + is_fields * 2 +
                        is_buildings * sum(face_materials[can_opt_tri] == BUILDING_ID) * 2 +
                        is_unknown * sum(face_materials[can_opt_tri] == UNKNOWN_ID) * 2)
        btri_end = (is_roads * 2 + is_trees * 2 + is_fields * 2 + is_buildings *
                    sum(face_materials[can_opt_tri] == BUILDING_ID) * 2)
        if is_roads:
            mscat[face_materials[can_opt_tri] == ROAD_ID] = x[0]
            mrcs[face_materials[can_opt_tri] == ROAD_ID] = x[1]
        if is_trees:
            mscat[face_materials[can_opt_tri] == TREE_ID] = x[is_roads * 2]
            mrcs[face_materials[can_opt_tri] == TREE_ID] = x[is_roads * 2 + 1]
        if is_fields:
            mscat[face_materials[can_opt_tri] == FIELD_ID] = x[is_roads * 2 + is_trees * 2]
            mrcs[face_materials[can_opt_tri] == FIELD_ID] = x[is_roads * 2 + is_trees * 2 + 1]
        if is_buildings:
            mscat[face_materials[can_opt_tri] == BUILDING_ID] = x[is_roads * 2 + is_trees * 2 + is_fields * 2:btri_end:2]
            mrcs[face_materials[can_opt_tri] == BUILDING_ID] = x[is_roads * 2 + is_trees * 2 + is_fields * 2 + 1:btri_end:2]
        if is_unknown:
            mscat[face_materials[can_opt_tri] == UNKNOWN_ID] = x[btri_end:vertex_start:2]
            mrcs[face_materials[can_opt_tri] == UNKNOWN_ID] = x[btri_end + 1:vertex_start:2]

        # Add in vertex unknowns
        if is_roads:
            vertex_add = sum(vertex_ids[can_opt_vert] == ROAD_ID)
            mz[vertex_ids[can_opt_vert] == ROAD_ID] = x[vertex_start:vertex_start + vertex_add]
            vertex_start += vertex_add
        if is_fields:
            vertex_add = sum(vertex_ids[can_opt_vert] == FIELD_ID)
            mz[vertex_ids[can_opt_vert] == FIELD_ID] = x[vertex_start:vertex_start + vertex_add]
            vertex_start += vertex_add
        if is_buildings:
            vertex_add = sum(vertex_ids[can_opt_vert] == BUILDING_ID)
            mz[vertex_ids[can_opt_vert] == BUILDING_ID] = x[vertex_start:vertex_start + vertex_add]
            vertex_start += vertex_add
        if is_unknown:
            vertex_add = sum(vertex_ids[can_opt_vert] == UNKNOWN_ID)
            mz[vertex_ids[can_opt_vert] == UNKNOWN_ID] = x[vertex_start:vertex_start + vertex_add]
            vertex_start += vertex_add
        # Vertices optimization
        spts = face_points + 0.0
        spts[can_opt_vert, 2] = mz
        spts_gpu = cupy.array(spts, dtype=np.float32)

        mfinalscat = opt_scat + 0.0
        mfinalscat[can_opt_tri] = mscat
        mscat_gpu = cupy.array(mfinalscat, dtype=np.float32)
        mfinalrcs = opt_rcs + 0.0
        mfinalrcs[can_opt_tri] = mrcs
        mrcs_gpu = cupy.array(mfinalrcs, dtype=np.float32)

        pd_r = cupy.zeros((nper, nsam), dtype=np.float64)
        pd_i = cupy.zeros((nper, nsam), dtype=np.float64)
        calcOptRho[bprun, threads_per_block](spts_gpu, triangles_gpu, source_gpu, pan_gpu, tilt_gpu, pd_r, pd_i, near_range_s, fs, rp.az_half_bw,
                                             rp.el_half_bw, mscat_gpu, mrcs_gpu, 2 * np.pi / wavelength, radar_coeff)
        x_hat = np.fft.fft(pd_r.get() + 1j * pd_i.get(), fft_len, axis=1) * chirp_filt
        upx = np.zeros((nper, up_fft_len), dtype=np.complex128)
        upx[:, fft_len // 2] = x_hat[:, fft_len // 2]
        upx[:, -fft_len // 2:] = x_hat[:, -fft_len // 2:]
        upx = np.fft.ifft(upx, axis=1)[:, :nsam * upsample]

        return np.linalg.norm(updata[:, rng_span[1]:rng_span[0]] - upx[:, rng_span[1]:rng_span[0]])

    opt_x = minimize(minfunc, x0, bounds=bounds)

    vertex_start = (is_roads * 2 + is_trees * 2 + is_fields * 2 + is_buildings *
                    sum(face_materials[can_opt_tri] == BUILDING_ID) * 2 +
                    is_unknown * sum(face_materials[can_opt_tri] == UNKNOWN_ID) * 2)
    vertex_add = 0
    btri_end = (is_roads * 2 + is_trees * 2 + is_fields * 2 + is_buildings *
                sum(face_materials[can_opt_tri] == BUILDING_ID) * 2)
    if is_roads:
        opt_scat[face_materials == ROAD_ID] = opt_x['x'][0]
        opt_rcs[face_materials == ROAD_ID] = opt_x['x'][1]
        road_scat = opt_x['x'][0]
        road_rcs = opt_x['x'][1]
    if is_trees:
        opt_scat[face_materials == TREE_ID] = opt_x['x'][is_roads * 2 + 0]
        opt_rcs[face_materials == TREE_ID] = opt_x['x'][is_roads * 2 + 1]
        tree_scat = opt_x['x'][is_roads * 2 + 0]
        tree_rcs = opt_x['x'][is_roads * 2 + 1]
    if is_fields:
        opt_scat[face_materials == FIELD_ID] = opt_x['x'][is_roads * 2 + is_trees * 2]
        opt_rcs[face_materials == FIELD_ID] = opt_x['x'][is_roads * 2 + is_trees * 2 + 1]
        field_scat = opt_x['x'][is_roads * 2 + is_trees * 2]
        field_rcs = opt_x['x'][is_roads * 2 + is_trees * 2 + 1]
    if is_buildings:
        opt_scat[np.logical_and(face_materials == BUILDING_ID, can_opt_tri)] = opt_x['x'][is_roads * 2 + is_trees * 2 + is_fields * 2:btri_end:2]
        opt_rcs[np.logical_and(face_materials == BUILDING_ID, can_opt_tri)] = opt_x['x'][is_roads * 2 + is_trees * 2 + is_fields * 2 + 1:btri_end:2]
    if is_unknown:
        opt_scat[np.logical_and(face_materials == UNKNOWN_ID, can_opt_tri)] = opt_x['x'][btri_end:vertex_start:2]
        opt_rcs[np.logical_and(face_materials == UNKNOWN_ID, can_opt_tri)] = opt_x['x'][btri_end + 1:vertex_start:2]

    # Add in vertex unknowns
    mz = np.zeros(nvert)
    if is_roads:
        vertex_add = sum(vertex_ids[can_opt_vert] == ROAD_ID)
        mz[vertex_ids[can_opt_vert] == ROAD_ID] = opt_x['x'][vertex_start:vertex_start + vertex_add]
        vertex_start += vertex_add
    if is_fields:
        vertex_add = sum(vertex_ids[can_opt_vert] == FIELD_ID)
        mz[vertex_ids[can_opt_vert] == FIELD_ID] = opt_x['x'][vertex_start:vertex_start + vertex_add]
        vertex_start += vertex_add
    if is_buildings:
        vertex_add = sum(vertex_ids[can_opt_vert] == BUILDING_ID)
        mz[vertex_ids[can_opt_vert] == BUILDING_ID] = opt_x['x'][vertex_start:vertex_start + vertex_add]
        vertex_start += vertex_add
    if is_unknown:
        vertex_add = sum(vertex_ids[can_opt_vert] == UNKNOWN_ID)
        mz[vertex_ids[can_opt_vert] == UNKNOWN_ID] = opt_x['x'][vertex_start:vertex_start + vertex_add]
        vertex_start += vertex_add

    face_points[can_opt_vert, 2] = mz

    # See if a triangle is fully optimized
    prog_bar.update(times[-1] - times[0] + 1)
    times = [times[-1] + 1]


opt_pcd = o3d.geometry.PointCloud()
opt_pcd.points = o3d.utility.Vector3dVector(face_points)
# opt_pcd.normals = o3d.utility.Vector3dVector(opt_norm)

# o3d.visualization.draw_plotly([pcd])
o3d.visualization.draw_geometries([mesh, opt_pcd])
