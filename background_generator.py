import cupy
import numpy as np
from SDRParsing import load
from cuda_mesh_kernels import readCombineMeshFile, calcInitSpread, calcIntersection
from cuda_kernels import getMaxThreads, cpudiff, applyRadiationPatternCPU
from grid_helper import SDREnvironment, mesh
from platform_helper import SDRPlatform, RadarPlatform
from scipy.ndimage import sobel, gaussian_filter
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import open3d as o3d
from simulation_functions import db, llh2enu, getElevation

c0 = 299792458.0
fs = 2e9
DTR = np.pi / 180

fnme = '/data6/SAR_DATA/2024/06212024/SAR_06212024_124710.sar'
sdr = load(fnme, progress_tracker=True)
wavelength = c0 / 9.6e9
ant_gain = 25
transmit_power = 100

bg = SDREnvironment(sdr)
rp = SDRPlatform(sdr, origin=bg.ref)

'''print('Generating Open3d mesh...')
mesh_lat = [40.09549, 40.09807]
mesh_lon = [-111.67357, -111.66864]
mesh_origin = (40.092961, -111.674558, 1370.)# (40.096657, -111.671529, 1370.)
center_mesh = llh2enu(*mesh_origin, bg.ref)
background_mesh = readCombineMeshFile('/home/jeff/Documents/alacademy.obj')
background_mesh.remove_degenerate_triangles()
background_mesh.remove_duplicated_vertices()
background_mesh.remove_non_manifold_edges()
background_mesh.compute_vertex_normals()
background_mesh.compute_triangle_normals()
background_mesh.normalize_normals()

target_mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/x-wing.obj')
target_mesh.translate(np.array([550., 800., 10.]))
center_mesh += np.array([-115., -35, 0])
background_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0., 0.])))
background_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.array([0., 0., rp.heading(rp.gpst).mean()])))
background_mesh.translate(center_mesh, relative=False)
points = np.asarray(background_mesh.vertices)

# full_mesh = target_mesh
full_mesh = background_mesh#  + target_mesh

print('Sampling background...')
total_samples = 100000
nsamples = 20000
# GPU device calculations
threads_per_block = getMaxThreads()
bpg_bpj = (max(1, nsamples // threads_per_block[0] + 1), len(full_mesh.triangle_normals) // threads_per_block[1] + 1)
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
    rp.getRadarParams(rp.pos(rp.gpst).mean(axis=0)[2], .1, 1))

samples = full_mesh.sample_points_poisson_disk(total_samples)
all_face_centers = np.asarray(samples.points)
all_face_normals = np.asarray(samples.normals)

colors = bg.sample(all_face_centers[:, 0], all_face_centers[:, 1])
colors = colors / colors.max()

vert_xyz_gpu = cupy.array(np.asarray(full_mesh.vertices), dtype=np.float32)
tri_indices_gpu = cupy.array(np.asarray(full_mesh.triangles), dtype=np.int32)
tri_norm_gpu = cupy.array(np.asarray(full_mesh.triangle_normals), dtype=np.float32)

receive_power_scale = (transmit_power / .01 *
                       (10 ** (ant_gain / 20)) ** 2
                       * wavelength ** 2 / (4 * np.pi) ** 3)

range_profile = np.zeros((128, len(ranges))).astype(np.complex128)
for idx, t in tqdm(enumerate(sdr[0].pulse_time[8173: 8173 + 128])):
    platform_pos = rp.pos(t)
    panrx = rp.pan(t).item()
    elrx = rp.tilt(t).item()
    source_xyz_gpu = cupy.array(platform_pos, dtype=np.float32)

    # Get initial power from beampattern
    range_vec = all_face_centers - platform_pos
    face_ranges = np.linalg.norm(range_vec, axis=1)
    face_az = np.arctan2(range_vec[:, 0], range_vec[:, 1])
    face_el = -np.arcsin(range_vec[:, 2] / face_ranges)
    a = np.pi / (10 * DTR)
    b = np.pi / (10 * DTR)
    # Abs shouldn't be a problem since the pattern is symmetrical about zero
    eldiff = abs(cpudiff(rp.tilt(t), face_el))
    azdiff = abs(cpudiff(rp.pan(t), face_az))
    tx_pat = abs(np.sin(a * azdiff) / (a * azdiff)) * abs(np.sin(b * eldiff) / (b * eldiff))
    for nf in range(5):
        face_centers = all_face_centers[nf * nsamples:(nf + 1) * nsamples]
        face_normals = all_face_normals[nf * nsamples:(nf + 1) * nsamples]
        ray_init_power = receive_power_scale * tx_pat[nf * nsamples:(nf + 1) * nsamples] * 1e9

        ray_distance_gpu = cupy.zeros(nsamples, dtype=np.float32)
        ray_power_gpu = cupy.array(ray_init_power, dtype=np.float32)
        ray_bounce_gpu = cupy.zeros((nsamples, 3), dtype=np.float32)
        vert_power_gpu = cupy.array(colors[nf * nsamples:(nf + 1) * nsamples], dtype=np.float32)
        ray_xyz_gpu = cupy.array(face_centers, dtype=np.float32)
        ray_norm_gpu = cupy.array(face_normals, dtype=np.float32)
        vrp_r_gpu = cupy.zeros(nsam, dtype=np.float64)
        vrp_i_gpu = cupy.zeros_like(vrp_r_gpu)
        calcInitSpread[bpg_bpj, threads_per_block](ray_power_gpu, ray_distance_gpu, ray_bounce_gpu, ray_xyz_gpu,
                                                   ray_norm_gpu, vert_power_gpu, source_xyz_gpu, panrx, elrx, vrp_r_gpu, vrp_i_gpu,
                                                   2 * np.pi / wavelength, near_range_s, fs, rp.az_half_bw,
                                                   rp.el_half_bw)
        cupy.cuda.Device().synchronize()
        ray_power = ray_power_gpu.get()
        ray_bounce = ray_bounce_gpu.get()
        ray_xyz = ray_xyz_gpu.get()
        ray_distance = ray_distance_gpu.get()
        for _ in range(1):
            calcIntersection[bpg_bpj, threads_per_block](ray_power_gpu, ray_distance_gpu, ray_bounce_gpu, ray_xyz_gpu,
                                                         vert_xyz_gpu, tri_norm_gpu, tri_indices_gpu,
                                                         source_xyz_gpu, panrx, elrx, vrp_r_gpu, vrp_i_gpu,
                                                         2 * np.pi / wavelength, near_range_s, fs, rp.az_half_bw,
                                                         rp.el_half_bw)
            cupy.cuda.Device().synchronize()
            ray_power_0 = ray_power_gpu.get()
            ray_bounce_0 = ray_bounce_gpu.get()
            ray_xyz_0 = ray_xyz_gpu.get()
            ray_distance_0 = ray_distance_gpu.get()

        range_profile[idx, :] += vrp_r_gpu.get() + 1j * vrp_i_gpu.get()
        ray_norm = ray_norm_gpu.get()

    del source_xyz_gpu
    del ray_distance_gpu
    del ray_power_gpu
    del ray_norm_gpu
    del ray_bounce_gpu
    del ray_xyz_gpu
    del vrp_r_gpu
    del vrp_i_gpu

del vert_xyz_gpu
del tri_norm_gpu
del tri_indices_gpu'''

'''plt.figure()
plt.imshow(db(np.fft.fft(range_profile, axis=0)))
plt.axis('tight')
plt.show()

plt.figure()
plt.imshow(db(range_profile))
plt.axis('tight')
plt.show()

fig = plt.figure('Bounces')
ax = fig.add_subplot(projection='3d')
ax.quiver(ray_xyz[:, 0], ray_xyz[:, 1], ray_xyz[:, 2], ray_bounce[:, 0], ray_bounce[:, 1],
          ray_bounce[:, 2])

plt.figure()
plt.scatter(points[:, 0], points[:, 1], c=db(bg.sample(points[:, 0], points[:, 1])))
plt.axis('tight')
plt.show()'''

'''ray_cloud = o3d.geometry.PointCloud()
ray_cloud.points = o3d.utility.Vector3dVector(ray_xyz)
ray_cloud.normals = o3d.utility.Vector3dVector(ray_bounce)

camera_cloud = o3d.geometry.PointCloud()
camera_cloud.points = o3d.utility.Vector3dVector(platform_pos.reshape((-1, 3)))
camera_cloud.normals = o3d.utility.Vector3dVector(center_mesh - platform_pos.reshape((-1, 3)))

o3d.visualization.draw_geometries([ray_cloud, full_mesh, camera_cloud], zoom=.2, front=[*platform_pos], lookat=[*center_mesh],
                                  up=[0., 0., 1.])

ray_cloud = o3d.geometry.PointCloud()
ray_cloud.points = o3d.utility.Vector3dVector(ray_xyz_0)
ray_cloud.normals = o3d.utility.Vector3dVector(ray_bounce_0)

o3d.visualization.draw_geometries([ray_cloud, full_mesh], zoom=.2, front=[*platform_pos], lookat=[*center_mesh], up=[0., 0., 1.])

plt.figure('Profile')
plt.plot(db(vrp_r_gpu.get() + 1j * vrp_i_gpu.get()))

ray_cloud = o3d.geometry.PointCloud()
ray_cloud.points = o3d.utility.Vector3dVector(all_face_centers)
colors = db(colors)
mappable = cm.ScalarMappable(Normalize(vmin=colors.min(), vmax=colors.max()))
ray_cloud.colors = o3d.utility.Vector3dVector(mappable.to_rgba(colors)[:, :3])
o3d.visualization.draw_geometries([ray_cloud, full_mesh], zoom=.2, front=[*platform_pos], lookat=[*center_mesh], up=[0., 0., 1.])'''

# bpg_bpj = (max(1, face_centers.shape[0] // threads_per_block[0] + 1), len(pan) // threads_per_block[1] + 1)

# o3d.visualization.draw_geometries([full_mesh])

# Calculate normal vectors for center points
# Experiments in scaling for JPEG file
from sklearn.preprocessing import QuantileTransformer
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, binary_fill_holes
from scipy.signal import convolve2d
import skimage.restoration as resto
from skimage.feature import canny, multiscale_basic_features
from skimage.segmentation import chan_vese
from PIL import Image


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for _ in np.arange(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im ** 2
    ones = np.ones(im.shape)

    kernel = np.ones((2 * N + 1, 2 * N + 1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    return np.sqrt((s2 - s ** 2 / ns) / ns)


asi_data = sdr.loadASI(sdr.files['asi'])[:, :7720]
mag_data = np.sqrt(abs(asi_data))
phase_data = np.angle(asi_data)

print('Denoising...')
mag_data = anisodiff(mag_data, 5, gamma=.25, kappa=1000)

# mag_data = np.sqrt(abs(sdr.loadASI(sdr.files['asi'][0])))
print('Binning...')
nbits = 256
plot_data = QuantileTransformer(output_distribution='normal').fit(
    mag_data[mag_data > 0].reshape(-1, 1)).transform(mag_data.reshape(-1, 1)).reshape(mag_data.shape)
max_bin = 3
hist_counts, hist_bins = \
    np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
while hist_counts[-1] == 0:
    max_bin -= .01
    hist_counts, hist_bins = \
        np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
scaled_data = np.sqrt(np.digitize(plot_data, hist_bins))
denoised = multiscale_basic_features(scaled_data)
nsub = int(np.round(np.sqrt(denoised.shape[2])))
plt.figure('Features')
for n in range(denoised.shape[2]):
    plt.subplot(nsub, nsub, n + 1)
    if denoised[:, :, n].min() >= 0:
        denoised[:, :, n] = denoised[:, :, n] / denoised[:, :, n].max()
    else:
        denoised[:, :, n] = denoised[:, :, n] / abs(denoised[:, :, n]).max()
    plt.imshow(denoised[:, :, n], origin='lower')
    plt.axis('off')

print('Chan-Vese...')
cv = chan_vese(
    scaled_data,
    mu=0.25,
    lambda1=1,
    lambda2=1,
    tol=1e-3,
    max_num_iter=200,
    dt=0.5,
    init_level_set="checkerboard",
    extended_output=True,
)

plt.figure('Chan_Vese')
plt.imshow(cv[1])


# selection = gauss_data[:, 5400]
plt.figure('Section')
plt.plot(scaled_data[:, 5400])
plt.plot(cv[1][:, 5400])


'''shadowmask = binary_fill_holes(binary_dilation(binary_erosion(scaled_data < 1.)))
plt.figure()
plt.imshow(shadowmask)
plt.axis('tight')

line_heading = (rp.heading(rp.gpst).mean() - np.pi / 2)
plt.figure()
plt.imshow(shadowmask)
plt.axis('tight')
plt.plot(np.arange(bg.shape[0]) * np.cos(line_heading) + 5250, np.arange(bg.shape[0]))
plt.show()'''

# tree_kernel = denoised[611:655, 5079:5115]