import torch
from numba import cuda
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from scripts.denoise_model import FFDNet
from simulib.platform_helper import SDRPlatform
from scipy.ndimage import sobel
from simulib.grid_helper import SDREnvironment
from scipy.interpolate import RegularGridInterpolator
from simulib.backproject_functions import getRadarAndEnvironment, backprojectPulseStream
from simulib.simulation_functions import db, genChirp, upsamplePulse, llh2enu, genTaylorWindow
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromMesh, _float
from tqdm import tqdm
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sdrparse import load

from simulib.mesh_objects import Mesh

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def addNoise(range_profile, chirp, npower, mf, fft_len):
    data = (chirp * np.fft.fft(range_profile + np.random.normal(0, npower, range_profile.shape) +
     1j * np.random.normal(0, npower, range_profile.shape), fft_len))
    return data * mf


fc = 9.6e9
rx_gain = 22  # dB
tx_gain = 22  # dB
rec_gain = 100  # dB
ant_transmit_power = 110  # watts
noise_power_db = -120
npulses = 64
plp = .75
fdelay = 10.
upsample = 8
num_bounces = 1
pixel_to_m = .25
nbox_levels = 5
nstreams = 2
points_to_sample = 2**15
num_mesh_triangles = 1000000
max_pts_per_run = 2**16

grid_origin = (40.133830, -111.665722, 1385.)
fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / sdr[0].fc

# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

print('Loading mesh...', end='')
# Prep the background ASI image
bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
    rp.getRadarParams(10., .75, upsample))
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
chirp_filt = np.fft.fft(sdr[0].cal_chirp, fft_len) * mfilt
print('Done.')

# This is all the constants in the radar equation for this simulation
radar_coeff = (c0**2 / fc**2 * ant_transmit_power * 10**((rx_gain + 2.15) / 10) * 10**((tx_gain + 2.15) / 10) *
               10**((rec_gain + 2.15) / 10) / (4 * np.pi)**3)
noise_power = 0  #10**(noise_power_db / 10)

chip_shape = (256, 256)
gx, gy, gz = bg.getGrid(grid_origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape)
bg.resampleGrid(grid_origin, chip_shape[0] * pixel_to_m, chip_shape[1] * pixel_to_m, *chip_shape)
pix_data = bg.refgrid[:, :7000]

print('Smoothing ASI...')
model_path = './model_zoo/ffdnet_gray.pth'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = FFDNet(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

img_L = np.float32(pix_data / pix_data.max())
img_std = img_L.std()

img_L = torch.from_numpy(np.ascontiguousarray(img_L)).float().unsqueeze(0).unsqueeze(0)
img_L = img_L.to(device)

sigma = torch.full((1, 1, 1, 1), img_std).type_as(img_L)

img_E = model(img_L, sigma)
img_E = img_E.data.squeeze().float().clamp_(0, 1).cpu().numpy() * pix_data.max()
model.to('cpu')

print('Placing mesh points...')
sobel_h = sobel(img_E, 0)  # horizontal gradient
sobel_v = sobel(img_E, 1)  # vertical gradient
grad_im = np.sqrt(sobel_h**2 + sobel_v**2)
xx, yy = np.meshgrid(np.arange(grad_im.shape[0]), np.arange(grad_im.shape[1]))
interpolator_img = RegularGridInterpolator(np.array([np.arange(grad_im.shape[0]), np.arange(grad_im.shape[1])]), img_E)
interpolator_grad = RegularGridInterpolator(np.array([np.arange(grad_im.shape[0]), np.arange(grad_im.shape[1])]), grad_im)
mu = grad_im.mean()
master_pts = []
while len(master_pts) < 10000:
    pts = np.random.rand(10000, 2) * (grad_im.shape[0] - 1)
    pts = pts[interpolator_grad(pts) > mu]
    master_pts = pts if len(master_pts) == 0 else np.concatenate((master_pts, pts))

rcs = interpolator_img(master_pts)


gnd_points = bg.getPos(master_pts[:, 0], master_pts[:, 1], elevation=True)
tri_ = Delaunay(gnd_points[:, :2])
ground = o3d.geometry.TriangleMesh()
ground.vertices = o3d.utility.Vector3dVector(gnd_points)
ground.triangles = o3d.utility.Vector3iVector(tri_.simplices)
ground.remove_duplicated_vertices()
ground.remove_unreferenced_vertices()
ground.compute_vertex_normals()
ground.compute_triangle_normals()
ground.normalize_normals()

ground.triangle_material_ids = o3d.utility.IntVector([0 for _ in rcs])

# Get initial estimates of kd, ks, and beta
msigmas = rcs / rcs.max() * 2
mkds = rcs / rcs.max() * .5
mkss = rcs / rcs.max() * .5

# Load in boxes and meshes for speedup of ray tracing
print('Loading mesh box structure...', end='')
ptsam = min(points_to_sample, max_pts_per_run)
mesh = Mesh(ground, num_box_levels=nbox_levels, use_box_pts=False, material_emissivity=msigmas, material_sigma=mkds, material_ks=mkss)
print('Done.')

sample_points = mesh.sample(ptsam, view_pos=rp.txpos(rp.gpst[np.linspace(0, len(rp.gpst) - 1, 4).astype(int)]))

boresight = rp.boresight(sdr[0].pulse_time).mean(axis=0)
pointing_az = np.arctan2(boresight[0], boresight[1])

# Locate the extrema to speed up the optimization
flight_path = rp.txpos(sdr[0].pulse_time)
pmax = sample_points.max(axis=0)
vecs = np.array([pmax[0] - flight_path[:, 0], pmax[1] - flight_path[:, 1],
                 pmax[2] - flight_path[:, 2]]).T
pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
max_pts = sdr[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw * 2]
pmin = sample_points.min(axis=0)
vecs = np.array([pmin[0] - flight_path[:, 0], pmin[1] - flight_path[:, 1],
                 pmin[2] - flight_path[:, 2]]).T
pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
min_pts = sdr[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw * 2]
pulse_lims = [max(min(min(max_pts), min(min_pts)) - 1000, 0), min(max(max(max_pts), max(min_pts)) + 1000, sdr[0].frame_num[-1])]
# pulse_lims = [0, sdr_f[0].nframes]
streams = [cuda.stream() for _ in range(nstreams)]

# Single pulse for debugging
bpj_grid = np.zeros_like(gx).astype(np.complex128)

print('Running main loop...')
# Get the data into CPU memory for later
# MAIN LOOP
# If we need to split the point raster, do so
if points_to_sample > max_pts_per_run:
    splits = np.concatenate((np.arange(0, points_to_sample, max_pts_per_run), [points_to_sample]))
else:
    splits = np.array([0, points_to_sample])
for s in range(len(splits) - 1):
    if s > 0:
        sample_points = mesh.sample(int(splits[s + 1] - splits[s]), view_pos=rp.txpos(rp.gpst[np.linspace(0, len(rp.gpst) - 1, 4).astype(int)]))
    for frame in tqdm(list(zip(*(iter(range(pulse_lims[0], pulse_lims[1] - npulses, npulses)),) * nstreams))):
        txposes = [rp.txpos(sdr[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
        rxposes = [rp.rxpos(sdr[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
        pans = [rp.pan(sdr[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
        tilts = [rp.tilt(sdr[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
        trp = getRangeProfileFromMesh(mesh, sample_points, txposes, rxposes, pans, tilts,
                                      radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s,
                                      num_bounces=num_bounces, streams=streams)
        mf_pulses = [np.ascontiguousarray(upsamplePulse(np.fft.fft(range_profile, fft_len, axis=1) * chirp_filt, fft_len, upsample, is_freq=True, time_len=nsam).T, dtype=np.complex128) for range_profile in trp]
        bpj_grid += backprojectPulseStream(mf_pulses, pans, tilts, rxposes, txposes, gz.astype(_float),
                                            c0 / fc, near_range_s, fs * upsample, rp.az_half_bw, rp.el_half_bw,
                                            gx=gx.astype(_float), gy=gy.astype(_float), streams=streams)

def getMeshFig(title='Title Goes Here', zrange=100):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            # i, j and k give the vertices of triangles
            i=mesh.tri_idx[:, 0],
            j=mesh.tri_idx[:, 1],
            k=mesh.tri_idx[:, 2],
            # facecolor=triangle_colors,
            showscale=True
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(zaxis=dict(range=[-30, zrange])),
    )
    return fig

fig = getMeshFig('Full Mesh')
fig.show()


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(db(pix_data))
plt.subplot(1, 2, 2)
plt.imshow(db(img_E))

plt.figure()
plt.scatter(master_pts[:, 0], master_pts[:, 1], c=interpolator_img(master_pts))

plt.figure()
plt.imshow(db(bpj_grid))
plt.show()