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
from simulib.mesh_functions import loadTarget
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
    ant_transmit_power = 10  # watts
    noise_power_db = -120
    upsample = 8
    exp_range = 500
    n_samples = 2**17

    nposes = 64
    azes = np.linspace(0, 2 * np.pi, nposes)
    eles = np.ones(nposes) * np.pi
    poses = np.ascontiguousarray(azelToVec(azes, eles).T * exp_range, dtype=_float)
    pointing = -poses
    pans = np.arctan2(pointing[:, 0], pointing[:, 1]).astype(_float)
    tilts = -np.arcsin(pointing[:, 2] / np.linalg.norm(pointing, axis=1)).astype(_float)
    near_range_s = (exp_range - 50) / c0
    nsam = 4096
    nr = 1024
    fft_len = 8192

    scene = Scene()

    mesh, mesh_materials = loadTarget('/home/jeff/Documents/target_meshes/air_balloon.targ')
    mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0]))).scale(1 / 50., center=mesh.get_center())
    mesh = mesh.translate(np.array([0, 0, 0.]), relative=False)
    scene.add(
        Mesh(
            mesh,
            max_tris_per_split=256,
            material_sigma=[mesh_materials[mtid][1] for mtid in
                            range(np.asarray(mesh.triangle_material_ids).max() + 1)],
            material_emissivity=[mesh_materials[mtid][0] for mtid in
                                 range(np.asarray(mesh.triangle_material_ids).max() + 1)],
        )
    )

    '''mesh = o3d.geometry.TriangleMesh.create_sphere(10., resolution=10)
    mesh.triangle_material_ids = o3d.utility.IntVector([0 for _ in range(len(mesh.triangles))])
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    scene.add(
        Mesh(
            mesh,
            max_tris_per_split=64,
            material_sigma=[1000000.],
            material_emissivity=[mesh_materials[mtid][1] for mtid in
                                 range(np.asarray(mesh.triangle_material_ids).max() + 1)],
        )
    )'''


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

    # sample_points = [scene.sample(n_samples, view_pos=poses[::16], fc=fc, fs=fs, near_range_s=near_range_s,
    #                               radar_equation_constant=radar_coeff)]
    sample_points = [scene.sample(n_samples)]

    streams = [cuda.stream()]
    rpfig = go.Figure()
    mfig = go.Figure()
    pfig = go.Figure()

    # Single pulse for debugging
    print('Generating single pulse...')
    single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromScene(scene, sample_points[0].astype(_float),
                                                                                  [poses.astype(_float)],
                                                                                  [poses.astype(_float)],
                                                                                  [pans.astype(_float)],
                                                                                  [tilts.astype(_float)],
                                                                                  radar_coeff,
                                                                                  np.pi / 128, np.pi / 128,
                                                                                  nsam, fc, near_range_s, fs,
                                                                                  num_bounces=1,
                                                                                  debug=True, streams=streams, use_supersampling=True)
    single_pulse = upsamplePulse(fft_chirp * np.fft.fft(single_rp[0], fft_len), fft_len, upsample,
                                 is_freq=True, time_len=nsam)
    single_mf_pulse = upsamplePulse(
        addNoise(single_rp[0], fft_chirp, noise_power, mf_chirp, fft_len), fft_len, upsample,
        is_freq=True, time_len=nsam)

    rpfig.add_scatter(y=db(single_rp[0][0].flatten()), mode='markers')
    pfig.add_scatter(y=db(single_pulse[0].flatten()), mode='markers')
    mfig.add_scatter(y=db(single_mf_pulse[0].flatten()), mode='markers')

    bounce_colors = ['blue', 'red', 'green', 'yellow']
    for bounce in range(len(ray_origins)):
        fig = getSceneFig(scene,
                          title=f'Bounce {bounce}', zrange=[-10, 10])
        for idx, (ro, rd, nrp) in enumerate(
                zip(ray_origins[:bounce + 1], ray_directions[:bounce + 1], ray_powers[:bounce + 1])):
            valids = nrp[0] > 0.
            sc = (1 + nrp[0, valids] / nrp[0, valids].max()) * 10
            fig.add_trace(go.Cone(x=ro[0, valids, 0], y=ro[0, valids, 1], z=ro[0, valids, 2], u=rd[0, valids, 0] * sc,
                                  v=rd[0, valids, 1] * sc, w=rd[0, valids, 2] * sc, anchor='tail', sizeref=80,
                                  colorscale=[[0, bounce_colors[idx]], [1, bounce_colors[idx]]]))

        fig.show()


    fig = getSceneFig(scene, title='Depth', zrange=[-10, 10])

    for mesh in scene.meshes:
        d = mesh.bvh_levels - 1
        for b in mesh.bvh[sum(2 ** n for n in range(d)):sum(2 ** n for n in range(d + 1))]:
            if np.sum(b) != 0:
                fig.add_trace(drawOctreeBox(b))
    fig.show()

    fig = getSceneFig(scene, title='Depth', zrange=[-10, 10])
    fig.add_scatter3d(x=sample_points[0][:, 0], y=sample_points[0][:, 1], z=sample_points[0][:, 2], mode='markers')
    fig.show()
    rpfig.show()
    mfig.show()
    pfig.show()

    plt.figure()
    plt.imshow(db(single_mf_pulse))
    plt.axis('tight')

    
