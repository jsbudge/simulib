import numpy as np
import torch
import yaml
from PIL import Image
from SDRParsing import load
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
from simulation_functions import db, llh2enu, getElevationMap
from pywt import wavedec2
from trimesh.creation import extrude_polygon
import plotly.io as pio
pio.renderers.default = 'browser'

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180

fnme = '/data6/SAR_DATA/2024/07082024/SAR_07082024_112333.sar'
sdr = load(fnme, import_pickle=False, progress_tracker=True)
wavelength = c0 / 9.6e9
ant_gain = 25
transmit_power = 100
pixel_to_m = .25

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)

device = o3d.core.Device("CPU:0")

# Load the segmentation model
with open('./segmenter_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
segmenter = ImageSegmenter(**param_dict['model_params'], label_sz=5, params=param_dict)
print('Setting up model...')
segmenter.load_state_dict(torch.load('./model/inference_model.state'))
segmenter.to('cuda:0')

png_fnme = '/home/jeff/repo/simulib/data/base_SAR_07082024_112333.png'
# bx, by, bz = bg.getGrid()

background = np.array(Image.open(png_fnme)) / 65535.
chip = background[:512, :512]

segment = np.zeros((5, *background.shape))

for x in tqdm(range(0, background.shape[0] - 512, 256)):
    for y in range(0, background.shape[1] - 512, 256):
        tense = torch.tensor(background[x:x + 512, y:y + 512], dtype=torch.float32,
                             device=segmenter.device).view(1, 1, 512, 512)
        segment[:, x:x + 512, y:y + 512] = segmenter(tense).cpu()[0, ...].data.numpy()

'''activation = {}


def get_activation(name):
    def hook(model, input, output):
        with contextlib.suppress(AttributeError):
            activation[name] = output.detach()

    return hook


sample = torch.tensor(chip, dtype=torch.float32, device=segmenter.device).unsqueeze(0).unsqueeze(0)
for name, module in segmenter.named_modules():
    module.register_forward_hook(get_activation(name))
output = segmenter(sample)

plt.figure('Original')
plt.imshow(sample[0, 0].cpu().data.numpy())
plt.axis('off')

for key, activ in activation.items():
    if '.' not in key:
        act = activ.squeeze(0).cpu().data.numpy()
        ngrid = int(np.floor(np.sqrt(act.shape[0]))) + (1 if np.sqrt(act.shape[0]) % 1 != 0 else 0)
        plt.figure(key)
        for x in range(act.shape[0]):
            plt.subplot(ngrid, ngrid, x + 1)
            plt.imshow(act[x])
            plt.axis('off')
plt.show()'''

pcd = o3d.geometry.TriangleMesh()

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
bpts = np.array([xp, yp, np.zeros_like(xp)]) * pixel_to_m
bcld.points = o3d.utility.Vector3dVector(bpts.T)
bcld.estimate_normals()
bmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(bcld, o3d.utility.DoubleVector([2., 20.]))
bmesh = bmesh.paint_uniform_color([0, 1., 0])
pcd += bmesh
print('Roads added to mesh.')

'''
============================================================
==================FIELDS====================================
'''
# Get the roads
'''fields = binary_dilation(binary_erosion(segment[3] > .9))

# Blob them
blabels = label(fields)

# Take the fields and add them to the background
bcld = o3d.geometry.PointCloud()
xp, yp = np.where(blabels > 0)
bpts = np.array([xp, yp, np.zeros_like(xp)]) * pixel_to_m
bcld.points = o3d.utility.Vector3dVector(bpts.T)
bcld.estimate_normals()
bmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(bcld, o3d.utility.DoubleVector([2.]))
bmesh = bmesh.paint_uniform_color([1., 1, 0])
pcd += bmesh
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
    bmesh = bmesh.translate(np.array([(poly_s.centroid.x + xmin) * pixel_to_m, (poly_s.centroid.y + ymin) * pixel_to_m, 0.]))
    pcd += bmesh
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

    for n in range(0, len(xd), 5):
        nm = o3d.geometry.TriangleMesh.create_sphere(sk_dists[n] * pixel_to_m)
        nm = nm.paint_uniform_color(np.array([0, 1, 0]))
        trunk = o3d.geometry.TriangleMesh.create_cylinder(.5, mhght)
        trunk = trunk.translate(np.array([0, 0, -mhght / 2]))
        nm += trunk
        nm = nm.translate(np.array([xd[n] * pixel_to_m, yd[n] * pixel_to_m, mhght]))
        pcd += nm
print('Trees added to mesh.')

pcd = pcd.compute_triangle_normals()
pcd = pcd.compute_vertex_normals()
# o3d.visualization.draw_plotly([pcd])
o3d.visualization.draw_geometries([pcd])
plt.figure()
plt.imshow(background)
