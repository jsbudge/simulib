import cupy
import numpy as np
from SDRParsing import load
from cuda_mesh_kernels import readCombineMeshFile, calcSpread
from cuda_kernels import getMaxThreads, cpudiff
from grid_helper import SDREnvironment, mesh
from platform_helper import SDRPlatform
from scipy.ndimage import sobel, gaussian_filter
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.tri import Triangulation
import open3d as o3d
from simulation_functions import db

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180

fnme = '/data6/SAR_DATA/2024/06212024/SAR_06212024_124611.sar'
sdr = load(fnme, progress_tracker=True)
origin = (40.135107, -111.675027, 1370.67212)
wavelength = c0 / 9.6e9

bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)

gx, gy, gz = bg.getGrid(origin, 500, 500, 500, 500)
refgrid = bg.getRefGrid(origin, 500, 500, 500, 500)
smooth_grid = gaussian_filter(db(refgrid), 25.)
edge_im = np.sqrt(sobel(smooth_grid, 0) ** 2 + sobel(smooth_grid, 1) ** 2)
edge_im = edge_im / edge_im.max()

print('Calculating mesh...')
mx, my, mref, simp = mesh(np.arange(500), np.arange(500), edge_im, 1e-3, 25000,
                          max_iters=60, minimize_vertices=False)


npos = bg.getPos(mx, my, True)

print('Getting face colors...')
facecolors = interpn([np.arange(500), np.arange(500)], db(refgrid),
                     np.array([(mx[simp[:, 0]] + mx[simp[:, 1]] + mx[simp[:, 2]]) / 3,
                               (my[simp[:, 0]] + my[simp[:, 1]] + my[simp[:, 2]]) / 3]).T)
fcx = facecolors - facecolors.min()
fcx /= fcx.max()

print('Generating Open3d mesh...')
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(npos)
pcd.colors = o3d.utility.Vector3dVector(np.array([fcx, fcx, fcx]).T)
pcd.estimate_normals()
background_mesh = o3d.geometry.TriangleMesh()
background_mesh.vertices = o3d.utility.Vector3dVector(npos)
background_mesh.triangles = o3d.utility.Vector3iVector(simp)
background_mesh.remove_degenerate_triangles()
background_mesh.remove_duplicated_vertices()
background_mesh.remove_non_manifold_edges()
background_mesh.compute_vertex_normals()
background_mesh.compute_triangle_normals()
background_mesh.normalize_normals()
background_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([fcx, fcx, fcx]).T)
background_mesh.translate(np.array([0, 0, 0.]))

target_mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/x-wing.obj')
target_mesh.translate(np.array([1050., 800., 10.]))
target_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi / 2, np.pi / 2, 0.])))
full_mesh = background_mesh + target_mesh

'''plt.figure()
plt.imshow(edge_im, origin='lower')
plt.figure()
plt.imshow(db(refgrid), origin='lower')
plt.figure()
plt.tripcolor(mx, my, simp, facecolors=facecolors)
plt.figure()
plt.tricontourf(Triangulation(mx, my, simp), db(bg.refgrid[mx.astype(int), my.astype(int)]), levels=120)'''
nsamples = 10000
# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, nsamples // threads_per_block[0] + 1), nsamples // threads_per_block[1] + 1)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(rp.pos(rp.gpst).mean(axis=0)[2], .1, 1))

samples = full_mesh.sample_points_poisson_disk(nsamples)
face_centers = np.asarray(samples.points)
face_normals = np.asarray(samples.normals)

vert_xyz_gpu = cupy.array(face_centers, dtype=np.float32)
vert_norm_gpu = cupy.array(face_normals, dtype=np.float32)

range_profile = np.zeros((128, len(ranges))).astype(np.complex128)
for idx, t in tqdm(enumerate(rp.gpst[:128])):
    platform_pos = rp.pos(t)
    source_xyz_gpu = cupy.array(platform_pos, dtype=np.float32)
    range_vec = face_centers - platform_pos
    face_ranges = np.linalg.norm(range_vec, axis=1)
    face_az = np.arctan2(range_vec[:, 0], range_vec[:, 1])
    face_el = -np.arcsin(range_vec[:, 2] / face_ranges)
    a = np.pi / (10 * DTR)
    b = np.pi / (10 * DTR)
    # Abs shouldn't be a problem since the pattern is symmetrical about zero
    eldiff = abs(cpudiff(rp.tilt(t), face_el))
    azdiff = abs(cpudiff(rp.pan(t), face_az))
    tx_pat = abs(np.sin(a * azdiff) / (a * azdiff)) * abs(np.sin(b * eldiff) / (b * eldiff))
    ray_init_distance = np.outer(face_ranges, np.ones(nsamples))
    ray_init_power = 100 * np.outer(tx_pat, np.ones(nsamples)) / ray_init_distance ** 2

    ray_power_gpu = cupy.array(ray_init_power, dtype=np.float32)
    ray_distance_gpu = cupy.array(ray_init_distance, dtype=np.float32)
    vrp_r_gpu = cupy.zeros(face_centers.shape[0], dtype=np.float64)
    vrp_i_gpu = cupy.zeros_like(vrp_r_gpu)
    face_bin = np.round((face_ranges * 2 / c0 - 2 * near_range_s) * fs).astype(int)
    for _ in range(2):
        calcSpread[bpg_bpj, threads_per_block](ray_power_gpu, ray_distance_gpu, vert_xyz_gpu, vrp_r_gpu, vrp_i_gpu,
                                               vert_norm_gpu, source_xyz_gpu, 2 * np.pi / wavelength)
        cupy.cuda.Device().synchronize()
        range_profile[idx, face_bin[face_bin < len(ranges)]] += \
            (vrp_r_gpu.get() + 1j * vrp_i_gpu.get())[face_bin < len(ranges)]

plt.figure()
plt.imshow(db(np.fft.fft(range_profile, axis=0)))
plt.axis('tight')
plt.show()

# bpg_bpj = (max(1, face_centers.shape[0] // threads_per_block[0] + 1), len(pan) // threads_per_block[1] + 1)

# o3d.visualization.draw_geometries([pcd, full_mesh])

# Calculate normal vectors for center points
