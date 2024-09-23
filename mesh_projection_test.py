import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as mtri
from backproject_functions import getRadarAndEnvironment
from simulation_functions import db, genChirp, upsamplePulse, findPowerOf2
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


fc = 32.0e9
ant_gain = 50  # dB
ant_transmit_power = 200  # watts
ant_eff_aperture = 10. * 10.  # m**2
bw_az = 10
bw_el = 10
npulses = 32
plp = .5
fdelay = 2.
upsample = 4
fnme = '/home/jeff/SDR_DATA/ARCHIVE/07082024/SAR_07082024_112333.sar'


sdr_f = load(fnme)
bg, rp = getRadarAndEnvironment(sdr_f)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(fdelay, plp, upsample))
idx_t = sdr_f[0].frame_num[sdr_f[0].nframes // 2:sdr_f[0].nframes // 2 + npulses * 32:32]
data_t = sdr_f[0].pulse_time[idx_t]

pointing_vec = rp.boresight(data_t).mean(axis=0)
mesh_center = rp.pos(rp.gpst).mean(axis=0) + pointing_vec * ranges.mean()

print('Loading mesh...')
mesh = readCombineMeshFile('/home/jeff/Documents/plot.obj', points=260000)  # Has just over 243000 points in the file
# mesh = o3d.geometry.TriangleMesh.create_sphere(radius=150, resolution=10)
mesh = mesh.compute_triangle_normals()
mesh = mesh.compute_vertex_normals()
mesh = mesh.translate(mesh_center, relative=False).scale(5, mesh_center)


# This is all the constants in the radar equation for this simulation
radar_coeff = ant_transmit_power * 10**(ant_gain / 10) * ant_eff_aperture / (4 * np.pi)**2

chirp = genChirp(nr, fs, fc, 400e6)
pulses = np.zeros((npulses, nsam * upsample), dtype=np.complex128)



box_tree, sample_points = getBoxesSamplesFromMesh(mesh, num_boxes=1, sample_points=10000)

single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(*box_tree, sample_points,
                                                                             rp.pos(data_t[npulses // 2]),
                                                                             pointing_vec, radar_coeff, bw_az, bw_el,
                                                                             nsam, fc, near_range_s * c0,
                                                                             num_bounces=1,
                                                                             debug=True)
single_pulse = upsamplePulse(np.fft.fft(chirp, fft_len) * np.fft.fft(single_rp, fft_len), fft_len, upsample,
                             is_freq=True, time_len=nsam)
single_mf_pulse = upsamplePulse(
    np.fft.fft(chirp, fft_len) * np.fft.fft(single_rp, fft_len) * np.fft.fft(chirp, fft_len).conj(), fft_len, upsample,
    is_freq=True, time_len=nsam)
for n in tqdm(range(npulses)):
    trp = getRangeProfileFromMesh(*box_tree, sample_points, rp.pos(data_t[n]), rp.boresight(data_t[n]).flatten(),
                                  radar_coeff, bw_az, bw_el, nsam, fc, near_range_s * c0, num_bounces=1)
    pulse = upsamplePulse(np.fft.fft(chirp, fft_len) * np.fft.fft(trp, fft_len), fft_len, upsample, is_freq=True)
    mf_pulse = upsamplePulse(
    np.fft.fft(chirp, fft_len) * np.fft.fft(trp, fft_len) * np.fft.fft(chirp, fft_len).conj(), fft_len, upsample,
        is_freq=True, time_len=nsam)
    pulses[n] = mf_pulse

face_points = np.asarray(mesh.vertices)
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
ax.quiver([obs_pt[0]], [obs_pt[1]], [obs_pt[2]], [pointing_vec[0] * 100], [pointing_vec[1] * 100], [pointing_vec[2]* 100])
for idx, (ro, rd, nrp) in enumerate(zip(ray_origins, ray_directions, ray_powers)):
    scaled_rp = (nrp - sc_min) * sc * 10
    ax.quiver(ro[:, 0], ro[:, 1], ro[:, 2], rd[:, 0],
              rd[:, 1], rd[:, 2], color=cm.jet(idx / len(ray_origins) * np.ones_like(nrp)))
plt.show()
px.scatter(db(single_rp)).show()
px.scatter(db(single_pulse)).show()
px.scatter(db(single_mf_pulse)).show()

plt.figure('Data')
plt.imshow(db(np.fft.fft(pulses, axis=0)))
plt.axis('tight')
plt.show()