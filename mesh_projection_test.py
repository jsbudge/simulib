import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as mtri
from scipy.signal.windows import taylor
from scipy.spatial import Delaunay
from backproject_functions import getRadarAndEnvironment, backprojectPulseSet
from simulation_functions import db, genChirp, upsamplePulse, enu2llh, llh2enu
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
ant_gain = 42  # dB
ant_transmit_power = 100  # watts
ant_eff_aperture = 10. * 10.  # m**2
npulses = 512
plp = .75
fdelay = 10.
upsample = 4
num_bounces = 0
nbounce_rays = 5
nboxes = 36
points_to_sample = 1000
num_mesh_triangles = 5000
num_rayrounds = 1
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
mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/ram1500trx2021.gltf',
                           points=num_mesh_triangles, scale=300)
# mesh = readCombineMeshFile('/home/jeff/Documents/nissan_sky/NissanSkylineGT-R(R32).obj',
#                            points=num_mesh_triangles)  # Has just over 500000 points in the file
mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([0, 0, -45. * DTR])))


mesh_extent = mesh.get_max_bound() - mesh.get_min_bound()
face_points = np.asarray(mesh.vertices)
grid_vec = face_points[face_points[:, 0] == face_points[:, 0].max()] - face_points[face_points[:, 1] == face_points[:, 1].min()]
head_ang = np.arctan2(grid_vec[0, 0], grid_vec[0, 1])
gx, gy, gz = bg.getGrid(grid_origin, 201 * .1, 199 * .1, nrows=201, ncols=199, az=-68.5715881976 * DTR)
grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
grid_ranges = np.linalg.norm(rp.txpos(data_t).mean(axis=0) - grid_pts, axis=1)

bpcd = o3d.geometry.PointCloud()
heights = [15, 18]
building_points = np.array([llh2enu(40.139148, -111.664156, 1380., bg.ref),
                            llh2enu(40.139342, -111.664427, 1380., bg.ref),
                            llh2enu(40.139148, -111.664156, 1380. + heights[0], bg.ref),
                            llh2enu(40.139342, -111.664427, 1380. + heights[0], bg.ref),
                            llh2enu(40.139729, -111.663969, 1380., bg.ref),
                            llh2enu(40.139729, -111.663969, 1380. + heights[1], bg.ref),
                            llh2enu(40.139729, -111.663969, 1380. + (heights[0] + heights[1]) / 2, bg.ref),
                            llh2enu(40.139552, -111.663695, 1380., bg.ref),
                            llh2enu(40.139552, -111.663695, 1380. + heights[1], bg.ref),
                            llh2enu(40.139552, -111.663695, 1380. + (heights[0] + heights[1]) / 2, bg.ref),
                            llh2enu(40.139977, -111.663158, 1380., bg.ref),
                            llh2enu(40.140169, -111.663449, 1380., bg.ref),
                            llh2enu(40.139977, -111.663158, 1380. + heights[0], bg.ref),
                            llh2enu(40.140169, -111.663449, 1380. + heights[0], bg.ref),
                            llh2enu(40.139759, -111.663427, 1380. + (heights[0] + heights[1]) / 2, bg.ref),
                            llh2enu(40.139364, -111.663907, 1380. + (heights[0] + heights[1]) / 2, bg.ref),
                            llh2enu(40.139575, -111.664150, 1380. + (heights[0] + heights[1]) / 2, bg.ref),
                            llh2enu(40.139969, -111.663695, 1380. + (heights[0] + heights[1]) / 2, bg.ref)
                            ])
bpcd.points = o3d.utility.Vector3dVector(building_points)
bnormals = building_points - building_points.mean(axis=0)
bpcd.normals = o3d.utility.Vector3dVector(bnormals / np.linalg.norm(bnormals, axis=1)[:, None])
building, _ = bpcd.compute_convex_hull()
bpcd = building.sample_points_poisson_disk(20) + bpcd
bpcd.normals = o3d.utility.Vector3dVector(np.asarray(bpcd.points) - building_points.mean(axis=0))
bpcd.normalize_normals()
building = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(bpcd, radii=o3d.utility.DoubleVector(np.array([1000, 100., 50., 5.])))
building.compute_vertex_normals()
building.compute_triangle_normals()
building.normalize_normals()
building = building.translate(np.array([0, 0, -16.]), relative=True)

gpx, gpy, gpz = bg.getGrid(grid_origin, 400, 400, nrows=400, ncols=400)
gnd_points = np.array([gpx.flatten(), gpy.flatten(), gpz.flatten()]).T
gnd_range = np.linalg.norm(rp.txpos(data_t).mean(axis=0) - gnd_points, axis=1)
gnd_points = gnd_points[np.logical_and(gnd_range > grid_ranges.min() - grid_ranges.std() * 3, gnd_range < grid_ranges.max() + grid_ranges.std() * 3)]
tri_ = Delaunay(gnd_points[:, :2])
ground = o3d.geometry.TriangleMesh()
ground.vertices = o3d.utility.Vector3dVector(gnd_points)
ground.triangles = o3d.utility.Vector3iVector(tri_.simplices)
ground = ground.simplify_vertex_clustering(30.)
ground.remove_duplicated_vertices()
ground.remove_unreferenced_vertices()
ground.compute_vertex_normals()
ground.compute_triangle_normals()
ground.normalize_normals()

grid_extent = np.array([gx.max() - gx.min(), gy.max() - gy.min(), gz.max() - gz.min()])

mesh = mesh.translate(np.array([gx.mean(), gy.mean(), gz.mean() + 2.5]), relative=False).scale(.6, center=np.array([gx.mean(), gy.mean(), gz.mean() + 2.5]))
mesh_ids = np.asarray(mesh.triangle_material_ids)
mesh_ids = np.concatenate((mesh_ids, np.array([28 for _ in range(len(ground.triangles))]), np.array([29 for _ in range(len(building.triangles))])))
# mesh_ids = np.concatenate((mesh_ids, np.array([28 for _ in range(len(building.triangles))])))
mesh += ground
mesh += building
mesh.triangle_material_ids = o3d.utility.IntVector([int(m) for m in mesh_ids])
face_points = np.asarray(mesh.vertices)

# This is all the constants in the radar equation for this simulation
radar_coeff = ant_transmit_power * 10**(ant_gain / 10) * ant_eff_aperture / (4 * np.pi)**2

# Generate a chirp
chirp = genChirp(nr, fs, fc, 400e6)
fft_chirp = np.fft.fft(chirp, fft_len)
twin = taylor(int(np.round(400e6 / fs * fft_len)))
taytay = np.zeros(fft_len, dtype=np.complex128)
winloc = int((fc % fs) * fft_len / fs) - len(twin) // 2
taytay[winloc:winloc + len(twin)] += twin
mf_chirp = fft_chirp * fft_chirp.conj() * taytay


# Load in boxes and meshes for speedup of ray tracing
try:
    msigmas = [2. for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
    msigmas[0] = msigmas[15] = 1.  # seats
    msigmas[6] = msigmas[13] = msigmas[17] = .1  # body
    msigmas[12] = msigmas[4] = 1.  # windshield
    msigmas[28] = 2.5
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample,
                                                      material_sigmas=msigmas)
except ValueError:
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample)

boresight = rp.boresight(sdr_f[0].pulse_time).mean(axis=0)
pointing_az = np.arctan2(boresight[0], boresight[1])

# Locate the extrema to speed up the optimization
flight_path = rp.txpos(sdr_f[0].pulse_time)
pmax = sample_points.max(axis=0)
vecs = np.array([pmax[0] - flight_path[:, 0], pmax[1] - flight_path[:, 1],
                 pmax[2] - flight_path[:, 2]]).T
pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
max_pts = sdr_f[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw * 2]
pmin = sample_points.min(axis=0)
vecs = np.array([pmin[0] - flight_path[:, 0], pmin[1] - flight_path[:, 1],
                 pmin[2] - flight_path[:, 2]]).T
pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
min_pts = sdr_f[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw * 2]
pulse_lims = [min(min(max_pts), min(min_pts)), max(max(max_pts), max(min_pts))]

# Single pulse for debugging
single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(*box_tree, sample_points,
                                                                             rp.txpos(data_t),
                                                                             rp.boresight(data_t), radar_coeff,
                                                                             rp.az_half_bw * 2, rp.el_half_bw * 2,
                                                                             nsam, fc, near_range_s,
                                                                             num_bounces=num_bounces,
                                                                             bounce_rays=nbounce_rays,
                                                                             debug=True)
single_pulse = upsamplePulse(fft_chirp * np.fft.fft(single_rp, fft_len), fft_len, upsample,
                             is_freq=True, time_len=nsam)
single_mf_pulse = upsamplePulse(
    mf_chirp * np.fft.fft(single_rp, fft_len), fft_len, upsample,
    is_freq=True, time_len=nsam)
bpj_grid = np.zeros_like(gx).astype(np.complex128)


# MAIN LOOP
for frame in tqdm(range(pulse_lims[0], pulse_lims[1] - npulses, npulses)):
    dt = sdr_f[0].pulse_time[frame:frame + npulses]
    trp = getRangeProfileFromMesh(*box_tree, sample_points, rp.txpos(dt), rp.boresight(dt),
                                  radar_coeff, rp.az_half_bw * 2, rp.el_half_bw * 2, nsam, fc, near_range_s, num_bounces=num_bounces,
                                  bounce_rays=nbounce_rays, num_rayrounds=num_rayrounds)
    clean_pulse = mf_chirp * np.fft.fft(trp, fft_len)
    noise = (np.random.normal(0, 1e-1, size=clean_pulse.shape) +
             1j * np.random.normal(0, 1e-1, size=clean_pulse.shape))
    mf_pulse = upsamplePulse(clean_pulse + noise, fft_len, upsample, is_freq=True, time_len=nsam)
    bpj_grid += backprojectPulseSet(mf_pulse.T, rp.pan(dt), rp.tilt(dt), rp.rxpos(dt), rp.txpos(dt), gx, gy, gz,
                                   c0 / fc, near_range_s, fs * upsample, rp.az_half_bw * 2, rp.el_half_bw * 2)

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
db_bpj = db(bpj_grid)
plt.imshow(db_bpj, cmap='gray', origin='lower', clim=[np.mean(db_bpj) - np.std(db_bpj), np.mean(db_bpj) + np.std(db_bpj) * 3])
plt.axis('tight')
plt.show()

plt.figure()
plt.scatter(gx.flatten(), gy.flatten())
plt.scatter(face_points[:, 0], face_points[:, 1])
plt.scatter(sample_points[:, 0], sample_points[:, 1])
plt.show()

ax = plt.figure().add_subplot(projection='3d')
polygons = []
for i in range(face_tris.shape[0]):
    face = face_tris[i]
    polygon = Poly3DCollection([face_points[face]], alpha=.75, facecolor=face_colors[i], linewidths=2)
    polygons.append(polygon)
    ax.add_collection3d(polygon)
'''for pt in range(ray_origins[0].shape[1]):
    ax.plot([rp.txpos(data_t[0])[0], ray_origins[0][0, pt, 0]],
            [rp.txpos(data_t[0])[1], ray_origins[0][0, pt, 1]],
            [rp.txpos(data_t[0])[2], ray_origins[0][0, pt, 2]])'''
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