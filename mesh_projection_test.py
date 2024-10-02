import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as mtri
from backproject_functions import getRadarAndEnvironment, backprojectPulseSet
from simulation_functions import db, genChirp, upsamplePulse, enu2llh
from cuda_mesh_kernels import readCombineMeshFile, getRangeProfileFromMesh, getBoxesSamplesFromMesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
from SDRParsing import load

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


fc = 9.6e9
ant_gain = 150  # dB
ant_transmit_power = 200  # watts
ant_eff_aperture = 10. * 10.  # m**2
bw_az = 4.5 * DTR
bw_el = 11 * DTR
npulses = 128
plp = .75
fdelay = 0.
upsample = 4
num_bounces = 1
nbounce_rays = 5
nboxes = 36
points_to_sample = 10000
num_mesh_triangles = 10000
grid_origin = (40.138544, -111.664394, 1381.)
fnme = '/home/jeff/SDR_DATA/RAW/08052024/SAR_08052024_110111.sar'


sdr_f = load(fnme)
bg, rp = getRadarAndEnvironment(sdr_f)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(0., plp, upsample))
idx_t = sdr_f[0].frame_num[sdr_f[0].nframes // 2 : sdr_f[0].nframes // 2 + npulses]
data_t = sdr_f[0].pulse_time[idx_t]

pointing_vec = rp.boresight(data_t).mean(axis=0)

print('Loading mesh...')
mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=num_mesh_triangles)  # Has just over 243000 points in the file
# mesh = o3d.geometry.TriangleMesh.create_sphere(radius=150, resolution=10)
mesh = mesh.compute_triangle_normals()
mesh = mesh.compute_vertex_normals()

mesh_extent = mesh.get_max_bound() - mesh.get_min_bound()
gx, gy, gz = bg.getGrid(grid_origin, 400, 400, nrows=200, ncols=200, az=np.pi / 4)

grid_extent = np.array([gx.max() - gx.min(), gy.max() - gy.min(), gz.max() - gz.min()])

mesh = mesh.translate(np.array([gx.mean(), gy.mean(), gz.mean()]), relative=False)
face_points = np.asarray(mesh.vertices)

# Calculate out the sigma values for triangles based on average of vertex colors
face_tris = np.asarray(mesh.triangles)
try:
    face_colors = np.asarray(mesh.vertex_colors)[face_tris].mean(axis=1)
except IndexError:
    face_colors = np.zeros_like(face_tris)

try:
    sigma = (face_colors[:, 1] - (face_colors[:, 0] + face_colors[:, 2]) ** 2) / face_colors.max() * 5
    sigma[sigma <= 0] = .1
except ValueError:
    sigma = np.ones(face_colors.shape[0]) * 1

# This is all the constants in the radar equation for this simulation
radar_coeff = ant_transmit_power * 10**(ant_gain / 10) * ant_eff_aperture / (4 * np.pi)**2

# Generate a chirp
chirp = genChirp(nr, fs, fc, 400e6)
fft_chirp = np.fft.fft(chirp, fft_len)

# Load in boxes and meshes for speedup of ray tracing
box_tree, sample_points = getBoxesSamplesFromMesh(mesh, sigma, num_boxes=nboxes, sample_points=points_to_sample)

# Single pulse for debugging
single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(*box_tree, sample_points,
                                                                             rp.pos(data_t[npulses // 2]).reshape((1, 3)),
                                                                             pointing_vec.reshape((1, 3)), radar_coeff, bw_az, bw_el,
                                                                             nsam, fc, near_range_s,
                                                                             num_bounces=num_bounces,
                                                                             bounce_rays=nbounce_rays,
                                                                             debug=True)
single_pulse = upsamplePulse(fft_chirp * np.fft.fft(single_rp, fft_len), fft_len, upsample,
                             is_freq=True, time_len=nsam)
single_mf_pulse = upsamplePulse(
    fft_chirp * np.fft.fft(single_rp, fft_len) * fft_chirp.conj(), fft_len, upsample,
    is_freq=True, time_len=nsam)
bpj_grid = np.zeros_like(gx).astype(np.complex128)


# MAIN LOOP
for frame in tqdm(range(0, sdr_f[0].nframes - npulses, npulses)):
    dt = sdr_f[0].pulse_time[frame:frame + npulses]
    trp = getRangeProfileFromMesh(*box_tree, sample_points, rp.pos(dt), rp.boresight(dt),
                                  radar_coeff, bw_az, bw_el, nsam, fc, near_range_s, num_bounces=num_bounces, bounce_rays=nbounce_rays)
    mf_pulse = upsamplePulse(
    fft_chirp * np.fft.fft(trp, fft_len) * fft_chirp.conj(), fft_len, upsample,
        is_freq=True, time_len=nsam)
    pulses = mf_pulse
    bpj_grid += backprojectPulseSet(pulses.T, rp.pan(dt), rp.tilt(dt), rp.rxpos(dt), rp.txpos(dt), gx, gy, gz,
                                   c0 / fc, near_range_s, fs * upsample, bw_az, bw_el)

'''
try:
    face_colors = np.asarray(mesh.vertex_colors)[face_tris].mean(axis=1)
except IndexError:
    face_colors = np.zeros_like(face_tris)
obs_pt = rp.pos(data_t[npulses // 2])
ax = plt.figure().add_subplot(projection='3d')
polygons = []
for i in range(face_tris.shape[0]):
    face = face_tris[i]
    polygon = Poly3DCollection([face_points[face]], alpha=.75, facecolor=face_colors[i], linewidths=2)
    polygons.append(polygon)
    ax.add_collection3d(polygon)
scaling = min(r.min() for r in ray_powers), max(r.max() for r in ray_powers)
sc_min = scaling[0]
sc = 1 / (scaling[1] - scaling[0])
ax.quiver([obs_pt[0]], [obs_pt[1]], [obs_pt[2]],
          [pointing_vec[0] * 100], [pointing_vec[1] * 100], [pointing_vec[2]* 100])
for idx, (ro, rd, nrp) in enumerate(zip(ray_origins, ray_directions, ray_powers)):
    scaled_rp = (nrp - sc_min) * sc * 10
    ax.quiver(ro[:, 0], ro[:, 1], ro[:, 2], rd[:, 0],
              rd[:, 1], rd[:, 2], color=cm.jet(idx / len(ray_origins) * np.ones_like(nrp)))
plt.show()'''
px.scatter(db(single_rp.flatten())).show()
px.scatter(db(single_pulse.flatten())).show()
px.scatter(db(single_mf_pulse.flatten())).show()

plt.figure('Data')
plt.imshow(db(np.fft.fft(pulses, axis=0)))
plt.axis('tight')
plt.show()

plt.figure('Backprojection')
plt.imshow(db(bpj_grid))
plt.axis('tight')
plt.show()

points_plot = np.asarray(sample_points.points)
plt.figure()
plt.scatter(gx.flatten(), gy.flatten())
plt.scatter(face_points[:, 0], face_points[:, 1])
plt.scatter(points_plot[:, 0], points_plot[:, 1])
plt.show()

'''ax = plt.figure().add_subplot(projection='3d')
ax.scatter(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2])
ax.quiver(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2], rp.boresight(rp.gpst)[:, 0] * 1500,
          rp.boresight(rp.gpst)[:, 1] * 1500, rp.boresight(rp.gpst)[:, 2] * 1500)
ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2])
ax.scatter(gx.flatten(), gy.flatten(), gz.flatten())'''
