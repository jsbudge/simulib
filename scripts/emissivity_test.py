import torch
import trimesh
import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
from numba import cuda
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from simulib.platform_helper import SDRPlatform, RadarPlatform
from simulib.grid_helper import SDREnvironment
from simulib.backproject_functions import getRadarAndEnvironment, backprojectPulseStream
from simulib.simulation_functions import db, genChirp, upsamplePulse, llh2enu, genTaylorWindow, enu2llh, getRadarCoeff, \
    azelToVec
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromMesh, _float, getRangeProfileFromScene, \
    getMeshFig, getSceneFig, drawOctreeBox
from tqdm import tqdm
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sdrparse import load
import matplotlib as mplib
mplib.use('TkAgg')
from simulib.mesh_objects import Mesh, Scene
from itertools import product

pio.renderers.default = 'browser'

def addNoise(range_profile, a_chirp, npower, mf, a_fft_len):
    data = a_chirp * np.fft.fft(range_profile, a_fft_len)
    data = data + np.random.normal(0, npower, data.shape) + 1j * np.random.normal(0, npower, data.shape)
    return data * mf

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180

if __name__ == '__main__':
    fc = 9.6e9
    fs = 2e9
    rx_gain = 32  # dB
    tx_gain = 32  # dB
    rec_gain = 100  # dB
    ant_transmit_power = 100  # watts
    noise_power_db = -120
    upsample = 8
    exp_range = 1500
    n_samples = 2**15

    nposes = 64
    azes = np.linspace(np.pi, 2 * np.pi, nposes)
    eles = np.ones(nposes) * .41
    poses = np.ascontiguousarray(azelToVec(azes, eles).T * exp_range, dtype=_float)
    pans = azes.astype(_float)
    tilts = eles.astype(_float)
    near_range_s = (exp_range - 50) / c0
    nsam = 4096
    nr = 1024
    fft_len = 8192

    # Get a triangle
    ground = o3d.geometry.TriangleMesh().create_sphere(radius=10, resolution=40)
    '''gx, gy = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
    gz = (gx + gy)**2
    gnd_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()]).T
    tri_ = Delaunay(gnd_points[:, :2])
    ground = o3d.geometry.TriangleMesh()
    ground.vertices = o3d.utility.Vector3dVector(gnd_points)
    ground.triangles = o3d.utility.Vector3iVector(tri_.simplices)'''
    ground.remove_duplicated_vertices()
    ground.remove_unreferenced_vertices()
    ground.compute_vertex_normals()
    ground.compute_triangle_normals()
    ground.normalize_normals()
    ground.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(ground.triangles)).astype(np.int32))

    emissivities = [5.24, 1e6, 1.01]
    sigmas = [.1, .01, .001]

    # Chirp data
    radar_coeff = getRadarCoeff(fc, ant_transmit_power, rx_gain, tx_gain, rec_gain)
    noise_power = 10 ** (noise_power_db / 10)

    # Generate a chirp
    # fft_chirp = np.fft.fft(sdr_f[0].cal_chirp, fft_len)
    # mf_chirp = sdr_f.genMatchedFilter(0, fft_len=fft_len)
    chirp_bandwidth = 400e6
    chirp = genChirp(nr, fs, fc, chirp_bandwidth)
    fft_chirp = np.fft.fft(chirp, fft_len)
    taytay = genTaylorWindow(fc % fs, chirp_bandwidth / 2, fs, fft_len)
    mf_chirp = taytay / fft_chirp

    scene = Scene()
    scene.add(Mesh(ground, max_tris_per_split=8, material_emissivity=[1e6], material_sigma=[.01]))

    sample_points = [scene.sample(n_samples,
                                  view_pos=poses[::30], fc=fc,
                                  fs=fs, near_range_s=near_range_s, radar_equation_constant=radar_coeff)]

    streams = [cuda.stream()]
    rpfig = go.Figure()
    mfig = go.Figure()
    pfig = go.Figure()

    for e, s in product(emissivities, sigmas):

        scene = Scene()
        scene.add(Mesh(ground, max_tris_per_split=8, material_emissivity=[e], material_sigma=[s]))



        # Single pulse for debugging
        print('Generating single pulse...')
        single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromScene(scene, sample_points[0].astype(_float),
                                                                                      [poses.astype(_float)],
                                                                                      [poses.astype(_float)],
                                                                                      [pans.astype(_float)],
                                                                                      [tilts.astype(_float)],
                                                                                      radar_coeff,
                                                                                      np.pi / 4, np.pi / 4,
                                                                                      nsam, fc, near_range_s, fs,
                                                                                      num_bounces=1,
                                                                                      debug=True, streams=streams, use_supersampling=True)
        single_pulse = upsamplePulse(fft_chirp * np.fft.fft(single_rp[0], fft_len), fft_len, upsample,
                                     is_freq=True, time_len=nsam)
        single_mf_pulse = upsamplePulse(
            addNoise(single_rp[0], fft_chirp, noise_power, mf_chirp, fft_len), fft_len, upsample,
            is_freq=True, time_len=nsam)

        rpfig.add_scatter(y=db(single_rp[0][0].flatten()), mode='markers', name=f'E {e}, S {s}')
        pfig.add_scatter(y=db(single_pulse[0].flatten()), mode='markers')
        mfig.add_scatter(y=db(single_mf_pulse[0].flatten()), mode='markers')

        bounce_colors = ['blue', 'red', 'green', 'yellow']
        for bounce in range(len(ray_origins)):
            fig = getSceneFig(scene,
                              title=f'Bounce {bounce}')
            for idx, (ro, rd, nrp) in enumerate(
                    zip(ray_origins[:bounce + 1], ray_directions[:bounce + 1], ray_powers[:bounce + 1])):
                valids = nrp[0] > 0.
                sc = (1 + nrp[0, valids] / nrp[0, valids].max()) * 10
                fig.add_trace(go.Cone(x=ro[0, valids, 0], y=ro[0, valids, 1], z=ro[0, valids, 2], u=rd[0, valids, 0] * sc,
                                      v=rd[0, valids, 1] * sc, w=rd[0, valids, 2] * sc, anchor='tail', sizeref=10,
                                      colorscale=[[0, bounce_colors[idx]], [1, bounce_colors[idx]]]))

            fig.show()


    fig = getSceneFig(scene, title='Depth')

    for mesh in scene.meshes:
        d = mesh.bvh_levels - 1
        for b in mesh.bvh[sum(2 ** n for n in range(d)):sum(2 ** n for n in range(d + 1))]:
            if np.sum(b) != 0:
                fig.add_trace(drawOctreeBox(b))
    fig.show()
    rpfig.show()
    mfig.show()
    pfig.show()

    
