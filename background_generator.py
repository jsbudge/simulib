import numpy as np
from SDRParsing import load
from cuda_mesh_kernels import readCombineMeshFile
from grid_helper import SDREnvironment, mesh
from scipy.ndimage import sobel, gaussian_filter
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import open3d as o3d
from simulation_functions import db

fnme = '/data6/SAR_DATA/2024/06212024/SAR_06212024_124611.sar'
sdr = load(fnme, progress_tracker=True)
origin = (40.135107, -111.675027, 1370.67212)

bg = SDREnvironment(sdr)

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

tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(full_mesh)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(tensor_mesh)

rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    fov_deg=90,
    center=[1313., 667., 0.],
    eye=[0, 0, 1524.],
    up=[0, 0, 1],
    width_px=640,
    height_px=480,
)
# We can directly pass the rays tensor to the cast_rays function.
lx = scene.list_intersections(rays)
ray_splits = lx['ray_splits'].numpy()
t_hit = lx['t_hit'].numpy()
ans = scene.cast_rays(rays)

plt.figure()
plt.imshow(ans['t_hit'].numpy())
plt.show()

for ray_id, (start, end) in enumerate(zip(ray_splits[:-1], ray_splits[1:])):
    for i,t in enumerate(t_hit[start:end]):
        print(f'ray {ray_id}, intersection {i} at {t}')

# o3d.visualization.draw_geometries([pcd, full_mesh])

# Calculate normal vectors for center points
