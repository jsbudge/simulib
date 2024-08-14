import contextlib
import cupy
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial import ConvexHull
from torch.nn import functional as tf
from SDRParsing import load, loadASIFile, loadASHFile
from cuda_mesh_kernels import readCombineMeshFile, calcInitSpread, calcIntersection
from cuda_kernels import getMaxThreads, cpudiff, applyRadiationPatternCPU
from grid_helper import SDREnvironment, mesh
from models import ImageSegmenter
from platform_helper import SDRPlatform, RadarPlatform
from image_segment_loader import prepImage
from scipy.ndimage import sobel, gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes
from scipy.interpolate import interpn
from skimage.measure import label, find_contours
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import open3d as o3d
from simulation_functions import db, llh2enu, getElevation
from pywt import wavedec2

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180

fnme = '/data6/SAR_DATA/2024/07082024/SAR_07082024_112333.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / 9.6e9
ant_gain = 25
transmit_power = 100

# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)

# Load the segmentation model
with open('./segmenter_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
segmenter = ImageSegmenter(**param_dict['model_params'], label_sz=5, params=param_dict)
print('Setting up model...')
segmenter.load_state_dict(torch.load('./model/inference_model.state'))
segmenter.to('cuda:1')

png_fnme = '/home/jeff/repo/simulib/data/base_SAR_07082024_112333.png'

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

    mhght = mhght * .25
    # Calculate height from image angle
    bding_height = mhght / np.tan(rp.dep_ang)
    foreshortening = mhght / np.tan(rp.dep_ang)**2

    contours = np.concatenate(find_contours(bding_block.astype(int), .9))
    poly = Polygon(contours)
    poly_s = poly.simplify(2)

    innerpts = np.array([[y, x] for y, x in zip(*np.where(bding_block)) if Point(y, x).within(poly_s)][::5])
    contours = np.array(poly_s.boundary.coords)

    bcld = o3d.geometry.PointCloud()
    bpt_list = [np.array([contours[:, 1] * 0.25 + ymin * 0.25, contours[:, 0] * 0.25 + xmin * 0.25,
                          np.ones(contours.shape[0]) * bding_height]),
                np.array([innerpts[:, 1] * 0.25 + ymin * 0.25, innerpts[:, 0] * 0.25 + xmin * 0.25,
                          np.ones(innerpts.shape[0]) * bding_height]),
                np.array([innerpts[:, 1] * 0.25 + ymin * 0.25, innerpts[:, 0] * 0.25 + xmin * 0.25,
                          np.ones(innerpts.shape[0]) * 0])
                ]
    bpt_list.extend(
        np.array(
            [
                contours[:, 1] * 0.25 + ymin * 0.25,
                contours[:, 0] * 0.25 + xmin * 0.25,
                np.ones(contours.shape[0]) * inter_height,
            ]
        )
        for inter_height in np.arange(0, mhght, 0.5)
    )
    bpts = np.concatenate(bpt_list, axis=1)
    bcld.points = o3d.utility.Vector3dVector(bpts.T)
    bcld.estimate_normals()
    bmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(bcld, o3d.utility.DoubleVector([1., 5., 10., 100.]))
    pcd += bmesh


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

    mhght = mhght * .25
    # Calculate height from image angle
    bding_height = mhght / np.tan(rp.dep_ang)
    foreshortening = mhght / np.tan(rp.dep_ang)**2

    # Center of mass
    xmass = xpts.mean()
    ymass = ypts.mean()

    contours = np.concatenate(find_contours(bding.astype(int), .9))
    innerpts = np.array(np.where(bding)).T
    innerdists = np.zeros(innerpts.shape[0])

    for i in range(innerpts.shape[0]):
        innerdists[i] = np.linalg.norm(innerpts[i] - contours, axis=1).min()

    shrubbery_size = mhght * .33
    hght_weights = innerdists * .25
    hght_weights = (hght_weights / hght_weights.max())**(1/4) * shrubbery_size

    tcld = o3d.geometry.PointCloud()
    tpts = np.concatenate((np.array([innerpts[:, 0] * .25, innerpts[:, 1] * .25, mhght - hght_weights]),
                           np.array([innerpts[:, 0] * .25, innerpts[:, 1] * .25, mhght + hght_weights])),
                          axis=1)
    tcld.points = o3d.utility.Vector3dVector(tpts.T)
    tcld.estimate_normals()
    tmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(tcld, o3d.utility.DoubleVector([1., 5., 10., 50.]))
    pcd += tmesh

o3d.visualization.draw_geometries([pcd])
