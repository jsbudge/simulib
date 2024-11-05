import sys
sys.path.extend(['/home/jeff/repo/data_converter', '/home/jeff/repo/simulib'])
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
from scipy.spatial import Delaunay
from backproject_functions import getRadarAndEnvironment, backprojectPulseSet
from simulation_functions import db, genChirp, upsamplePulse, llh2enu
from cuda_mesh_kernels import readCombineMeshFile, getRangeProfileFromMesh, getBoxesSamplesFromMesh
from tqdm import tqdm
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from SDRParsing import load


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
ant_transmit_power = 100  # watts
noise_power_db = -80
npulses = 128
plp = .75
fdelay = 10.
upsample = 4
num_bounces = 1
nbounce_rays = 1
nbox_levels = 4
nstreams = 10
points_to_sample = 100
num_mesh_triangles = 10000
grid_origin = (40.139343, -111.663541, 1360.10812)
fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'

# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'


sdr_f = load(fnme)
bg, rp = getRadarAndEnvironment(sdr_f)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(0., plp, upsample))
idx_t = sdr_f[0].frame_num[sdr_f[0].nframes // 2 : sdr_f[0].nframes // 2 + npulses]
data_t = sdr_f[0].pulse_time[idx_t]

pointing_vec = rp.boresight(data_t).mean(axis=0)

gx, gy, gz = bg.getGrid(grid_origin, 201 * .1, 199 * .1, nrows=201, ncols=199, az=-68.5715881976 * DTR)
# gx, gy, gz = bg.getGrid(grid_origin, 400, 200, nrows=800, ncols=400)
grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
grid_ranges = np.linalg.norm(rp.txpos(data_t).mean(axis=0) - grid_pts, axis=1)

print('Loading mesh...', end='')
# mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/ram1500trx2021.gltf',
#                            points=num_mesh_triangles, scale=300)
mesh = o3d.geometry.TriangleMesh()
mesh_ids = []

'''mesh = readCombineMeshFile('/home/jeff/Documents/roman_facade/scene.gltf', points=1000000)
mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
mesh_ids = np.asarray(mesh.triangle_material_ids)'''

car = readCombineMeshFile('/home/jeff/Documents/nissan_sky/NissanSkylineGT-R(R32).obj',
                           points=num_mesh_triangles, scale=.6)  # Has just over 500000 points in the file
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([0, 0, -42.51 * DTR])))
mesh_extent = car.get_max_bound() - car.get_min_bound()
car = car.translate(np.array([gx.mean(), gy.mean(), gz.mean() + mesh_extent[2] / 2]), relative=False)
mesh_ids = np.asarray(car.triangle_material_ids)
mesh += car

building = readCombineMeshFile('/home/jeff/Documents/target_meshes/hangar.gltf', points=10000, scale=.8)
building = building.translate(llh2enu(40.139670, -111.663759, 1380, bg.ref) + np.array([-10, -10, -6.]),
                              relative=False).rotate(building.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
building = building.rotate(building.get_rotation_matrix_from_xyz(np.array([0, 0, 42.51 * DTR])))
mesh_ids = np.concatenate((mesh_ids, np.asarray(building.triangle_material_ids) + mesh_ids.max()))
mesh += building

gpx, gpy, gpz = bg.getGrid(grid_origin, 201, 199, nrows=201, ncols=199, az=-68.5715881976 * DTR)
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
radar_coeff = (c0**2 / fc**2 * ant_transmit_power * 10**((rx_gain + 2.15) / 10) * 10**((tx_gain + 2.15) / 10) *
               10**((rec_gain + 2.15) / 10) / (4 * np.pi)**3)
noise_power = 10**(noise_power_db / 10)

# Generate a chirp
chirp = genChirp(nr, fs, fc, 400e6)
fft_chirp = np.fft.fft(chirp, fft_len)
twin = taylor(int(np.round(400e6 / fs * fft_len)))
taytay = np.zeros(fft_len, dtype=np.complex128)
winloc = int((fc % fs) * fft_len / fs) - len(twin) // 2
taytay[winloc:winloc + len(twin)] += twin
mf_chirp = fft_chirp.conj() * taytay


# Load in boxes and meshes for speedup of ray tracing
print('Loading mesh box structure...', end='')
try:
    msigmas = [2. for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
    '''msigmas[0] = msigmas[15] = .2  # seats
    msigmas[6] = msigmas[13] = msigmas[17] = .02  # body
    msigmas[12] = msigmas[4] = 1.  # windshield'''
    mkds = [.5 for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
    '''mkds[0] = mkds[15] = 1.  # seats
    mkds[6] = mkds[13] = mkds[17] = 1.  # body
    mkds[12] = mkds[4] = .1  # windshield'''
    mkss = [.5 for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
    '''mkss[0] = mkss[15] = .2  # seats
    mkss[6] = mkss[13] = mkss[17] = 1.  # body
    mkss[12] = mkss[4] = .01  # windshield'''
    # msigmas[28] = 2
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_box_levels=nbox_levels, sample_points=points_to_sample,
                                                      material_sigmas=msigmas, material_kd=mkds, material_ks=mkss)
except ValueError:
    print('Error in getting material sigmas.')
    box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_box_levels=nbox_levels, sample_points=points_to_sample)
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
single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(*box_tree, sample_points,
                                                                             rp.txpos(data_t),
                                                                             rp.boresight(data_t), radar_coeff,
                                                                             rp.az_half_bw, rp.el_half_bw,
                                                                             nsam, fc, near_range_s,
                                                                             num_bounces=num_bounces,
                                                                             bounce_rays=nbounce_rays,
                                                                             debug=True, nstreams=nstreams)
single_pulse = upsamplePulse(fft_chirp * np.fft.fft(single_rp, fft_len), fft_len, upsample,
                             is_freq=True, time_len=nsam)
single_mf_pulse = upsamplePulse(
    addNoise(single_rp, fft_chirp, noise_power, mf_chirp, fft_len), fft_len, upsample,
    is_freq=True, time_len=nsam)
bpj_grid = np.zeros_like(gx).astype(np.complex128)

print('Running main loop...')
# MAIN LOOP
for frame in tqdm(range(pulse_lims[0], pulse_lims[1] - npulses, npulses)):
    dt = sdr_f[0].pulse_time[frame:frame + npulses]
    trp = getRangeProfileFromMesh(*box_tree, sample_points, rp.txpos(dt), rp.boresight(dt),
                                  radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s, num_bounces=num_bounces,
                                  bounce_rays=nbounce_rays, nstreams=nstreams)
    mf_pulse = upsamplePulse(addNoise(trp, fft_chirp, noise_power, mf_chirp, fft_len), fft_len, upsample, is_freq=True, time_len=nsam)
    bpj_grid += backprojectPulseSet(mf_pulse.T, rp.pan(dt), rp.tilt(dt), rp.txpos(dt), rp.txpos(dt), gz,
                                    c0 / fc, near_range_s, fs * upsample, rp.az_half_bw, rp.el_half_bw,
                                    gx=gx, gy=gy)
    # bpj_grid += backprojectPulseSet(mf_pulse.T, rp.pan(dt), rp.tilt(dt), rp.txpos(dt), rp.txpos(dt), gz,
    #                                c0 / fc, near_range_s, fs * upsample, rp.az_half_bw, rp.el_half_bw, transform=bg.transforms[0])

def getMeshFig(title='Title Goes Here'):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=box_tree[4][:, 0],
            y=box_tree[4][:, 1],
            z=box_tree[4][:, 2],
            colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            # intensity=point_rng / point_rng.max(),
            # i, j and k give the vertices of triangles
            # here we represent the 4 triangles of the tetrahedron surface
            i=box_tree[3][:, 0],
            j=box_tree[3][:, 1],
            k=box_tree[3][:, 2],
            name='y',
            showscale=True
        )
    ])
    fig.update_layout(
        title=title,
    )
    return fig

px.scatter(db(single_rp[0].flatten())).show()
px.scatter(db(single_pulse[0].flatten())).show()
px.scatter(db(single_mf_pulse[0].flatten())).show()

'''plt.figure('Data')
plt.imshow(db(single_mf_pulse))
plt.axis('tight')
plt.show()

plt.figure('Backprojection')
db_bpj = db(bpj_grid)
plt.imshow(db_bpj, cmap='gray', origin='lower', clim=[np.mean(db_bpj), np.mean(db_bpj) + np.std(db_bpj) * 2])
plt.axis('tight')
plt.axis('off')
plt.show()'''

scaling = min(r.min() for r in ray_powers), max(r.max() for r in ray_powers)
sc_min = scaling[0] - 1e-3
sc = 1 / (scaling[1] - scaling[0])
scaled_rp = (ray_powers[0] - sc_min) * sc

fig = getMeshFig('Full Mesh')
fig.show()

bounce_colors = ['blue', 'red', 'green', 'yellow']
for bounce in range(len(ray_origins)):
    fig = getMeshFig(f'Bounce {bounce}')
    for idx, (ro, rd, nrp) in enumerate(zip(ray_origins[:bounce + 1], ray_directions[:bounce + 1], ray_powers[:bounce + 1])):
        valids = nrp[0] > 1e-9
        fig.add_trace(go.Cone(x=ro[0, valids, 0], y=ro[0, valids, 1], z=ro[0, valids, 2], u=rd[0, valids, 0],
                          v=rd[0, valids, 1], w=rd[0, valids, 2], sizemode='absolute', sizeref=40, anchor='tail',
                              colorscale=[[0, bounce_colors[idx]], [1, bounce_colors[idx]]]))

    fig.show()


def drawbox(box):
    vertices = []
    for z in range(2):
        vertices.append([box[0, 0], box[0, 1], box[z, 2]])
        vertices.append([box[1, 0], box[0, 1], box[z, 2]])
        vertices.append([box[1, 0], box[0, 1], box[int(not z), 2]])
        vertices.append([box[1, 0], box[0, 1], box[z, 2]])
        vertices.append([box[1, 0], box[1, 1], box[z, 2]])
        vertices.append([box[1, 0], box[1, 1], box[int(not z), 2]])
        vertices.append([box[1, 0], box[1, 1], box[z, 2]])
        vertices.append([box[0, 0], box[1, 1], box[z, 2]])
        vertices.append([box[0, 0], box[1, 1], box[int(not z), 2]])
        vertices.append([box[0, 0], box[1, 1], box[z, 2]])
        vertices.append([box[0, 0], box[0, 1], box[z, 2]])
    vertices = np.array(vertices)
    return go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='lines')

fig = getMeshFig()

for b in box_tree[0][sum(8**n for n in range(nbox_levels - 1)):]:
    if np.sum(b) != 0:
        fig.add_trace(drawbox(b))

fig.show()