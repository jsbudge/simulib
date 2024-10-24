import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
from scipy.spatial import Delaunay
from backproject_functions import getRadarAndEnvironment, backprojectPulseSet
from simulation_functions import db, genChirp, upsamplePulse, llh2enu
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
ant_gain = 22  # dB
ant_transmit_power = 100  # watts
npulses = 128
plp = .75
fdelay = 10.
upsample = 4
num_bounces = 1
nbounce_rays = 1
nboxes = 1000
points_to_sample = 1000
num_mesh_triangles = 10000
grid_origin = (40.139343, -111.663541, 1360.10812)
fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'


sdr_f = load(fnme)
bg, rp = getRadarAndEnvironment(sdr_f)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(0., plp, upsample))
idx_t = sdr_f[0].frame_num[sdr_f[0].nframes // 2 : sdr_f[0].nframes // 2 + npulses]
data_t = sdr_f[0].pulse_time[idx_t]

pointing_vec = rp.boresight(data_t).mean(axis=0)

gx, gy, gz = bg.getGrid(grid_origin, 201 * .1, 199 * .1, nrows=201, ncols=199, az=-68.5715881976 * DTR)
# gx, gy, gz = bg.getGrid(grid_origin, 400, 200, nrows=400, ncols=200)
grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
grid_ranges = np.linalg.norm(rp.txpos(data_t).mean(axis=0) - grid_pts, axis=1)

print('Loading mesh...', end='')
# mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/ram1500trx2021.gltf',
#                            points=num_mesh_triangles, scale=300)
mesh = o3d.geometry.TriangleMesh()
mesh_ids = []

'''mesh = readCombineMeshFile('/home/jeff/Documents/roman_facade/scene.gltf', points=100000)
mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
mesh_ids = np.asarray(mesh.triangle_material_ids)'''

car = readCombineMeshFile('/home/jeff/Documents/nissan_sky/NissanSkylineGT-R(R32).obj',
                           points=num_mesh_triangles, scale=.6)  # Has just over 500000 points in the file
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([0, 0, -42.51 * DTR])))
# Rotate into antenna frame
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([pointing_el, 0, 0])))
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([0, 0, pointing_az])))
points = np.asarray(car.vertices)
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
hpoints = points.dot(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1 / 100.],[0, 0, 0, 0]]))
proj_points = hpoints / hpoints[:, 3][:, None]
mesh_extent = car.get_max_bound() - car.get_min_bound()
car = car.translate(np.array([gx.mean(), gy.mean(), gz.mean() + mesh_extent[2] / 2]), relative=False)
mesh_ids = np.asarray(car.triangle_material_ids)
mesh += car

'''building = readCombineMeshFile('/home/jeff/Documents/target_meshes/hangar.gltf', points=10000, scale=.8)
building = building.translate(llh2enu(40.139670, -111.663759, 1380, bg.ref) + np.array([-10, -10, -6.]),
                              relative=False).rotate(building.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
building = building.rotate(building.get_rotation_matrix_from_xyz(np.array([0, 0, 42.51 * DTR])))
mesh_ids = np.concatenate((mesh_ids, np.asarray(building.triangle_material_ids) + mesh_ids.max()))
mesh += building'''

gpx, gpy, gpz = bg.getGrid(grid_origin, 201 * .2, 199 * .2, nrows=201, ncols=199, az=-68.5715881976 * DTR)
gnd_points = np.array([gpx.flatten(), gpy.flatten(), gpz.flatten()]).T
gnd_range = np.linalg.norm(rp.txpos(data_t).mean(axis=0) - gnd_points, axis=1)
gnd_points = gnd_points[np.logical_and(gnd_range > grid_ranges.min() - grid_ranges.std() * 3,
                                       gnd_range < grid_ranges.max() + grid_ranges.std() * 3)]
tri_ = Delaunay(gnd_points[:, :2])
ground = o3d.geometry.TriangleMesh()
ground.vertices = o3d.utility.Vector3dVector(gnd_points)
ground.triangles = o3d.utility.Vector3iVector(tri_.simplices)
ground = ground.simplify_vertex_clustering(5.)
ground.remove_duplicated_vertices()
ground.remove_unreferenced_vertices()
ground.compute_vertex_normals()
ground.compute_triangle_normals()
ground.normalize_normals()
mesh += ground
if len(mesh_ids) > 0:
    mesh_ids = np.concatenate((mesh_ids, np.array([mesh_ids.max() + 1 for _ in range(len(ground.triangles))])))
else:
    mesh_ids = np.zeros(len(ground.triangles)).astype(int)

grid_extent = np.array([gx.max() - gx.min(), gy.max() - gy.min(), gz.max() - gz.min()])
mesh.triangle_material_ids = o3d.utility.IntVector([int(m) for m in mesh_ids])
face_points = np.asarray(mesh.vertices)
print('Done.')

# This is all the constants in the radar equation for this simulation
radar_coeff = c0**2 / fc**2 * ant_transmit_power * 10**((ant_gain + 2.15) / 5) / (4 * np.pi)**3

# Generate a chirp
chirp = genChirp(nr, fs, fc, 400e6)
fft_chirp = np.fft.fft(chirp, fft_len)
twin = taylor(int(np.round(400e6 / fs * fft_len)))
taytay = np.zeros(fft_len, dtype=np.complex128)
winloc = int((fc % fs) * fft_len / fs) - len(twin) // 2
taytay[winloc:winloc + len(twin)] += twin
mf_chirp = fft_chirp * fft_chirp.conj() * taytay


# Load in boxes and meshes for speedup of ray tracing
print('Loading mesh box structure...', end='')
try:
    msigmas = [2. for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
    # msigmas[0] = msigmas[15] = 2.  # seats
    # msigmas[6] = msigmas[13] = msigmas[17] = .5  # body
    # msigmas[12] = msigmas[4] = 2.  # windshield
    # msigmas[28] = 2
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample,
                                                      material_sigmas=msigmas)
except ValueError:
    print('Error in getting material sigmas.')
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=nboxes, sample_points=points_to_sample)
print('Done.')

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

rng_sequence = np.random.rand(npulses, nbounce_rays, 2) * .33

# Single pulse for debugging
print('Generating single pulse...')
single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(*box_tree, rng_sequence,
                                                                             rp.txpos(data_t),
                                                                             rp.boresight(data_t), radar_coeff,
                                                                             rp.az_half_bw, rp.el_half_bw,
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

print('Running main loop...')
# MAIN LOOP
for frame in tqdm(range(pulse_lims[0], pulse_lims[1] - npulses, npulses)):
    dt = sdr_f[0].pulse_time[frame:frame + npulses]
    trp = getRangeProfileFromMesh(*box_tree, rng_sequence, rp.txpos(dt), rp.boresight(dt),
                                  radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s, num_bounces=num_bounces,
                                  bounce_rays=nbounce_rays)
    clean_pulse = mf_chirp * np.fft.fft(trp, fft_len)
    noise = (np.random.normal(0, 1e-12, size=clean_pulse.shape) +
             1j * np.random.normal(0, 1e-12, size=clean_pulse.shape))
    mf_pulse = upsamplePulse(clean_pulse + noise, fft_len, upsample, is_freq=True, time_len=nsam)
    bpj_grid += backprojectPulseSet(mf_pulse.T, rp.pan(dt), rp.tilt(dt), rp.txpos(dt), rp.txpos(dt), gx, gy, gz,
                                   c0 / fc, near_range_s, fs * upsample, rp.az_half_bw, rp.el_half_bw)

'''face_tris = np.asarray(mesh.triangles)
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
sc_min = scaling[0] - 1e-3
sc = 1 / (scaling[1] - scaling[0])
for idx, (ro, rd, nrp) in enumerate(zip(ray_origins, ray_directions, ray_powers)):
    scaled_rp = nrp
    ax.quiver(ro[0, :, 0], ro[0, :, 1], ro[0, :, 2], rd[0, :, 0] * scaled_rp[0, :],
              rd[0, :, 1] * scaled_rp[0, :], rd[0, :, 2] * scaled_rp[0, :])
ax.set_zlim(face_points[:, 2].min() - 15., face_points[:, 2].max() + 15)
ax.set_ylim(face_points[:, 1].min() - 5., face_points[:, 1].max() + 5)
ax.set_xlim(face_points[:, 0].min() - 5., face_points[:, 0].max() + 5)
plt.show()'''
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

'''ax = plt.figure('First Bounce Angle').add_subplot(projection='3d')
polygons = []
for i in range(face_tris.shape[0]):
    face = face_tris[i]
    polygon = Poly3DCollection([face_points[face]], alpha=.75, facecolor=face_colors[i], linewidths=2)
    polygons.append(polygon)
    ax.add_collection3d(polygon)
scaled_rp = (ray_powers[0] - sc_min) * sc
ax.quiver(ray_origins[0][0, :, 0], ray_origins[0][0, :, 1], ray_origins[0][0, :, 2], ray_directions[0][0, :, 0] * scaled_rp[0, :],
          ray_directions[0][0, :, 1] * scaled_rp[0, :], ray_directions[0][0, :, 2] * scaled_rp[0, :])
ax.scatter(gx.flatten(), gy.flatten(), gz.flatten())
ax.set_zlim(face_points[:, 2].min(), face_points[:, 2].max())
ax.set_ylim(face_points[:, 1].min(), face_points[:, 1].max())
ax.set_xlim(face_points[:, 0].min(), face_points[:, 0].max())
plt.show()'''

'''ax = plt.figure().add_subplot(projection='3d')
for idx, (ro, rd, nrp) in enumerate(zip(ray_origins, ray_directions, ray_powers)):
    scaled_rp = (nrp - sc_min) * sc * 10
    ax.scatter(ro[0, :, 0], ro[0, :, 1], ro[0, :, 2])'''

'''ax = plt.figure().add_subplot(projection='3d')
ax.scatter(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2])
ax.quiver(rp.pos(rp.gpst)[:, 0], rp.pos(rp.gpst)[:, 1], rp.pos(rp.gpst)[:, 2], rp.boresight(rp.gpst)[:, 0] * 1500,
          rp.boresight(rp.gpst)[:, 1] * 1500, rp.boresight(rp.gpst)[:, 2] * 1500)
ax.scatter(points_plot[:, 0], points_plot[:, 1], points_plot[:, 2])
ax.scatter(gx.flatten(), gy.flatten(), gz.flatten())'''

campos = rp.txpos(sdr_f[0].pulse_time).mean(axis=0)
boresight = np.array([gx.mean() - campos[0], gy.mean() - campos[1], gz.mean() + mesh_extent[2] / 2 - campos[2]])
bnorm = boresight / np.linalg.norm(boresight)
pointing_az = np.arctan2(bnorm[0], bnorm[1])
pointing_el = -np.arcsin(bnorm[2])
car = readCombineMeshFile('/home/jeff/Documents/nissan_sky/NissanSkylineGT-R(R32).obj',
                           points=num_mesh_triangles, scale=.6)  # Has just over 500000 points in the file
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([0, 0, -42.51 * DTR])))
car = car.translate(np.array([gx.mean() - campos[0], gy.mean() - campos[1], gz.mean() + mesh_extent[2] / 2 - campos[2]]), relative=False)
# Rotate into antenna frame
points = np.asarray(car.vertices).dot(car.get_rotation_matrix_from_zyx(np.array([-pointing_az, 0, 3 * np.pi / 2 - pointing_el])))
points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
hpoints = points.dot(np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1 / 100.],[0, 0, 0, 0]]))
proj_points = hpoints / hpoints[:, 3][:, None]
mesh_extent = car.get_max_bound() - car.get_min_bound()


ax = plt.figure().add_subplot(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.quiver([0], [0], [0], [0], [0], [1000])
ax.azim = pointing_az / DTR
ax.elev = pointing_el / DTR
ax.dist = 100

plt.figure()
plt.scatter(proj_points[:, 0], proj_points[:, 1])

az_bw_con = 100 * np.tan(rp.az_half_bw)
el_bw_con = 100 * np.tan(rp.el_half_bw)
p0 = radar_coeff
A = (-.9 / 5 * 2 + 1)
tri_idx = box_tree[2][0]
tri_verts = hpoints[tri_idx]
x = (u * tri_verts[0, 0] + v * tri_verts[1, 0] + w * tri_verts[2, 0])
y = (u * tri_verts[0, 1] + v * tri_verts[1, 1] + w * tri_verts[2, 1])
z = (u * tri_verts[0, 2] + v * tri_verts[1, 2] + w * tri_verts[2, 2])
w = (1 - u - v)
a_t = lambda u, v: (np.sin(np.tan(rp.el_half_bw) * (u * tri_verts[0, 2] + v * tri_verts[1, 2] + (1 - u - v) * tri_verts[2, 2]) *
                             (u * tri_verts[0, 0] + v * tri_verts[1, 0] + (1 - u - v) * tri_verts[2, 0]))**2 /
                       (np.tan(rp.el_half_bw) * (u * tri_verts[0, 2] + v * tri_verts[1, 2] + (1 - u - v) * tri_verts[2, 2]) *
                        (u * tri_verts[0, 0] + v * tri_verts[1, 0] + (1 - u - v) * tri_verts[2, 0]))**2 *
                       np.sin(np.tan(rp.az_half_bw) * (u * tri_verts[0, 2] + v * tri_verts[1, 2] + (1 - u - v) * tri_verts[2, 2]) *
                              (u * tri_verts[0, 1] + v * tri_verts[1, 1] + (1 - u - v) * tri_verts[2, 1]))**2 /
                       (np.tan(rp.az_half_bw) * (u * tri_verts[0, 2] + v * tri_verts[1, 2] + (1 - u - v) * tri_verts[2, 2]) *
                        (u * tri_verts[0, 1] + v * tri_verts[1, 1] + (1 - u - v) * tri_verts[2, 1]))**2)
r_add = lambda u, v: 1 / (2 * np.sqrt((u * tri_verts[0, 0] + v * tri_verts[1, 0] + (1 - u - v) * tri_verts[2, 0])**2 +
                                         (u * tri_verts[0, 1] + v * tri_verts[1, 1] + (1 - u - v) * tri_verts[2, 1])**2 +
                                         (u * tri_verts[0, 2] + v * tri_verts[1, 2] + (1 - u - v) * tri_verts[2, 2])**2))**2

u, v = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
uvadd = u + v
u = u[uvadd > 1]
v = v[uvadd > 1]

atts = a_t(u, v)
r_adds = r_add(u, v)

plt.figure()
plt.scatter(u, v, s=atts * 1e10)

plt.figure()
plt.scatter(u, v, s=r_adds * 1e10)