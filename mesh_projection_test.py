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
ant_gain = 52  # dB
ant_transmit_power = 100  # watts
ant_eff_aperture = 10. * 10.  # m**2
bw_az = 4.5 * DTR
bw_el = 11 * DTR
npulses = 128
plp = .75
fdelay = 10.
upsample = 4
num_bounces = 1
nbounce_rays = 5
nboxes = 36
points_to_sample = 1000
num_mesh_triangles = 10000
grid_origin = (40.139343, -111.663541, 1380.)
fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'


sdr_f = load(fnme)
bg, rp = getRadarAndEnvironment(sdr_f)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(0., plp, upsample))
idx_t = sdr_f[0].frame_num[sdr_f[0].nframes // 2 : sdr_f[0].nframes // 2 + npulses]
data_t = sdr_f[0].pulse_time[idx_t]

pointing_vec = rp.boresight(data_t).mean(axis=0)

print('Loading mesh...')
mesh = readCombineMeshFile('/home/jeff/Documents/nissan_sky/NissanSkylineGT-R(R32).obj',
                           points=num_mesh_triangles)  # Has just over 500000 points in the file
mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))


mesh_extent = mesh.get_max_bound() - mesh.get_min_bound()
face_points = np.asarray(mesh.vertices)
grid_vec = face_points[face_points[:, 0] == face_points[:, 0].max()] - face_points[face_points[:, 1] == face_points[:, 1].min()]
head_ang = np.arctan2(grid_vec[0, 0], grid_vec[0, 1])
gx, gy, gz = bg.getGrid(grid_origin, 201 * .1, 199 * .1, nrows=201, ncols=199, az=-68.5715881976 * DTR)
gpcd = o3d.geometry.PointCloud()
gpcd.points = o3d.utility.Vector3dVector(np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T)
gpcd.estimate_normals()
ground = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(gpcd, radii=o3d.utility.DoubleVector(np.array([.08])))
ground = ground.simplify_vertex_clustering(1.)

grid_extent = np.array([gx.max() - gx.min(), gy.max() - gy.min(), gz.max() - gz.min()])

mesh = mesh.translate(np.array([gx.mean(), gy.mean(), gz.mean() + 2.5]), relative=False)
mesh_ids = np.asarray(mesh.triangle_material_ids)
mesh_ids = np.concatenate((mesh_ids, np.array([28 for _ in range(len(ground.triangles))])))
mesh += ground
mesh.triangle_material_ids = o3d.utility.IntVector([int(m) for m in mesh_ids])
face_points = np.asarray(mesh.vertices)

# This is all the constants in the radar equation for this simulation
radar_coeff = ant_transmit_power * 10**(ant_gain / 10) * ant_eff_aperture / (4 * np.pi)**2

# Generate a chirp
chirp = genChirp(nr, fs, fc, 400e6)
fft_chirp = np.fft.fft(chirp, fft_len)

# Load in boxes and meshes for speedup of ray tracing
try:
    msigmas = [1. for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
    msigmas[0] = msigmas[15] = 5  # seats
    msigmas[6] = msigmas[13] = msigmas[17] = .1  # body
    msigmas[12] = msigmas[4] = 5  # windshield
    msigmas[28] = .1
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample,
                                                      material_sigmas=msigmas)
except ValueError:
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample)

# Single pulse for debugging
single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(*box_tree, sample_points,
                                                                             rp.txpos(data_t),
                                                                             rp.boresight(data_t), radar_coeff,
                                                                             bw_az, bw_el,
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
    trp = getRangeProfileFromMesh(*box_tree, sample_points, rp.txpos(dt), rp.boresight(dt),
                                  radar_coeff, bw_az, bw_el, nsam, fc, near_range_s, num_bounces=num_bounces,
                                  bounce_rays=nbounce_rays)
    mf_pulse = upsamplePulse(
    fft_chirp * np.fft.fft(trp, fft_len) * fft_chirp.conj(), fft_len, upsample,
        is_freq=True, time_len=nsam)
    pulses = mf_pulse
    bpj_grid += backprojectPulseSet(pulses.T, rp.pan(dt), rp.tilt(dt), rp.rxpos(dt), rp.txpos(dt), gx, gy, gz,
                                   c0 / fc, near_range_s, fs * upsample, bw_az, bw_el)

face_tris = np.asarray(mesh.triangles)
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
for idx, (ro, rd, nrp) in enumerate(zip(ray_origins, ray_directions, ray_powers)):
    scaled_rp = (nrp - sc_min) * sc * 10
    ax.quiver(ro[0, :, 0], ro[0, :, 1], ro[0, :, 2], rd[0, :, 0] * scaled_rp[0, :],
              rd[0, :, 1] * scaled_rp[0, :], rd[0, :, 2] * scaled_rp[0, :])
ax.set_zlim(face_points[:, 2].min() - 15., face_points[:, 2].max() + 15)
ax.set_ylim(face_points[:, 1].min() - 5., face_points[:, 1].max() + 5)
ax.set_xlim(face_points[:, 0].min() - 5., face_points[:, 0].max() + 5)
plt.show()
px.scatter(db(single_rp[0].flatten())).show()
px.scatter(db(single_pulse[0].flatten())).show()
px.scatter(db(single_mf_pulse[0].flatten())).show()

plt.figure('Data')
plt.imshow(db(single_mf_pulse))
plt.axis('tight')
plt.show()

plt.figure('Backprojection')
plt.imshow(db(bpj_grid), cmap='gray', clim=[80., 125.])
plt.axis('tight')
plt.show()

points_plot = np.asarray(sample_points.points)
plt.figure()
plt.scatter(gx.flatten(), gy.flatten())
plt.scatter(face_points[:, 0], face_points[:, 1])
plt.scatter(points_plot[:, 0], points_plot[:, 1])
plt.show()

ax = plt.figure().add_subplot(projection='3d')
polygons = []
for i in range(face_tris.shape[0]):
    face = face_tris[i]
    polygon = Poly3DCollection([face_points[face]], alpha=.75, facecolor=face_colors[i], linewidths=2)
    polygons.append(polygon)
    ax.add_collection3d(polygon)
for pt in range(ray_origins[0].shape[1]):
    ax.plot([rp.txpos(data_t[0])[0], ray_origins[0][0, pt, 0]],
            [rp.txpos(data_t[0])[1], ray_origins[0][0, pt, 1]],
            [rp.txpos(data_t[0])[2], ray_origins[0][0, pt, 2]])
scaled_rp = (ray_powers[0] - sc_min) * sc
ax.quiver(ray_origins[0][0, :, 0], ray_origins[0][0, :, 1], ray_origins[0][0, :, 2], ray_directions[0][0, :, 0] * scaled_rp[0, :],
          ray_directions[0][0, :, 1] * scaled_rp[0, :], ray_directions[0][0, :, 2] * scaled_rp[0, :])
ax.set_zlim(face_points[:, 2].min(), face_points[:, 2].max())
ax.set_ylim(face_points[:, 1].min(), face_points[:, 1].max())
ax.set_xlim(face_points[:, 0].min(), face_points[:, 0].max())
plt.show()

ax = plt.figure().add_subplot(projection='3d')
for idx, (ro, rd, nrp) in enumerate(zip(ray_origins, ray_directions, ray_powers)):
    scaled_rp = (nrp - sc_min) * sc * 10
    ax.scatter(ro[0, :, 0], ro[0, :, 1], ro[0, :, 2])

'''ax = plt.figure().add_subplot(projection='3d')
ax.scatter(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2])
ax.quiver(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2], rp.boresight(rp.gpst)[:, 0] * 1500,
          rp.boresight(rp.gpst)[:, 1] * 1500, rp.boresight(rp.gpst)[:, 2] * 1500)
ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2])
ax.scatter(gx.flatten(), gy.flatten(), gz.flatten())'''