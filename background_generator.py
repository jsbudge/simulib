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
from skimage.measure import label
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

fnme = '/data6/SAR_DATA/2024/08052024/SAR_08052024_105936.sar'
# sdr = load(fnme, progress_tracker=True)
wavelength = c0 / 9.6e9
ant_gain = 25
transmit_power = 100

# Prep the background ASI image
# bg = SDREnvironment(sdr)
# rp = SDRPlatform(sdr, origin=bg.ref)

png_fnme = '/home/jeff/repo/simulib/logs/base_SAR_06212024_124710.png'

background = np.array(Image.open(png_fnme)) / 65535.
chip = background[:1024, 5000:6024]

wavelets = ['coif4', 'db1', 'db4', 'haar', 'coif1', 'bior1.1', 'rbio1.1']

plt.figure('Wavelets')
for idx, w in enumerate(wavelets):
    try:
        plt.subplot(3, 3, idx + 1)
        plt.title(w)
        wave_coeffs = wavedec2(chip, w, level=3)
        plt.imshow(wave_coeffs[1][0])
        plt.axis('off')
    except:
        continue
plt.show()

'''shadows = binary_fill_holes(binary_dilation(binary_erosion(chip == 0)))
blobs, nblobs = label(shadows, return_num=True)

selector = np.zeros_like(chip)

for b in tqdm(range(1, nblobs)):
    picktree = blobs == b
    shadow_start = np.cumsum(picktree, axis=0)
    shadow_start = shadow_start * np.gradient(shadow_start, axis=0)
    shadow_start = np.argmax(shadow_start, axis=0)
    shadow_height = np.sum(picktree, axis=0) * .25
    height_from_shadow = shadow_height / np.tan(np.pi / 2 - rp.dep_ang)
    layover_heights = height_from_shadow / np.tan(np.pi / 2 - rp.dep_ang)

    layover_pix = np.ceil(layover_heights / .25).astype(int)

    tree_correction = (np.ceil((shadow_start[shadow_start != 0].max() - shadow_start[shadow_start != 0]) * .25 /
                               np.tan(np.pi / 2 - rp.dep_ang) ** 2).astype(int) + layover_pix.max() -
                       layover_pix[shadow_start != 0])
    shadow_start[shadow_start != 0] += tree_correction

    tree_mask = np.zeros_like(picktree)
    for n in range(tree_mask.shape[1]):
        tree_mask[shadow_start[n]:shadow_start[n] + layover_pix[n], n] = 1
    tree_mask = binary_dilation(binary_erosion(tree_mask))
    hull = ConvexHull(np.array(np.where(picktree)).T)
    if hull.volume - np.sum(picktree) > 150:
        print(f'Blob {b} fails convex test.')
        plt.figure(f'Blob {b}')
        plt.imshow(picktree)'''

    # selector += chip * tree_mask

'''plt.figure()
plt.imshow(selector)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(chip)
plt.subplot(2, 2, 2)
plt.imshow(blobs)
plt.subplot(2, 2, 3)
plt.imshow(chip * tree_mask)
plt.subplot(2, 2, 4)
plt.imshow(picktree)

# Load the segmentation model
with open('./segmenter_config.yaml') as y:
    param_dict = yaml.safe_load(y.read())
segmenter = ImageSegmenter(**param_dict['model_params'], label_sz=5, params=param_dict)
print('Setting up model...')
segmenter.load_state_dict(torch.load('./model/inference_model.state'))
segmenter.to('cuda:1')

activation = {}


def get_activation(name):
    def hook(model, input, output):
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
