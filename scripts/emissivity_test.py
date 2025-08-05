import os
import pickle
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
from pathlib import Path
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sdrparse import load
import matplotlib as mplib
mplib.use('TkAgg')
import pandas as pd
from simulib.mesh_objects import Mesh, Scene, VTCMesh
from simulib.mesh_functions import loadTarget
from itertools import product
from glob import glob

pio.renderers.default = 'browser'

def addNoise(range_profile, a_chirp, npower, mf, a_fft_len):
    data = a_chirp * np.fft.fft(range_profile, a_fft_len, axis=1)
    data = data + np.random.normal(0, npower, data.shape) + 1j * np.random.normal(0, npower, data.shape)
    return data * mf

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180

if __name__ == '__main__':
    fc = 9.6e9
    fs = 4e9
    rx_gain = 32  # dB
    tx_gain = 32  # dB
    rec_gain = 100  # dB
    ant_transmit_power = 10  # watts
    noise_power_db = -120
    upsample = 8
    exp_range = 500
    n_samples = 2**12
    single_target = '/home/jeff/Documents/target_meshes/tacoma_VTC.dat'

    nposes = 8
    azes, eles = np.meshgrid(np.linspace(0, 2 * np.pi, nposes), np.linspace(-np.pi / 2, np.pi / 2, nposes))
    azes = azes.flatten()
    eles = eles.flatten()
    # azes = np.linspace(0, 2 * np.pi, nposes)
    # azes = np.ones(nposes) * -np.pi / 2
    # eles = np.ones(nposes) * 0
    poses = np.ascontiguousarray(azelToVec(azes, eles).T * exp_range, dtype=_float)
    pointing = -poses
    pans = np.arctan2(pointing[:, 0], pointing[:, 1]).astype(_float)
    tilts = -np.arcsin(pointing[:, 2] / np.linalg.norm(pointing, axis=1)).astype(_float)
    # pans = np.linspace(np.pi / 2 - 1., np.pi / 2 + 1., nposes).astype(_float)
    # tilts = np.zeros(nposes).astype(_float)
    near_range_s = (exp_range - 50) / c0
    nsam = 4096
    nr = 4000
    fft_len = 8192

    target_info = pd.read_csv('/home/jeff/repo/apache/data/target_info.csv')

    targets = [*glob('/home/jeff/Documents/target_meshes/*.targ'), single_target]
    # Generate a chirp
    # fft_chirp = np.fft.fft(sdr_f[0].cal_chirp, fft_len)
    # mf_chirp = sdr_f.genMatchedFilter(0, fft_len=fft_len)
    chirp_bandwidth = 1400e6
    pulse_idx = 16

    # Chirp data
    radar_coeff = getRadarCoeff(fc, ant_transmit_power, rx_gain, tx_gain, rec_gain)
    noise_power = 10 ** (noise_power_db / 10)

    chirp = genChirp(nr, fs, fc, chirp_bandwidth)
    fft_chirp = np.fft.fft(chirp, fft_len)
    taytay = genTaylorWindow(fc % fs, chirp_bandwidth / 2, fs, fft_len)
    mf_chirp = taytay / fft_chirp
    for target_fnme in targets:
        # if target_fnme != single_target:
        #     continue
        print(target_fnme)
        scene = Scene()
        # Check if this is some VTC data and load the mesh data accordingly
        if Path(target_fnme).suffix == '.dat':
            mesh = VTCMesh(target_fnme)
        else:
            with open(target_fnme, 'r') as f:
                tdata = f.readlines()
            try:
                target_scaling = 1 / float(target_info.loc[target_info['filename'] ==
                                                     f'{Path(tdata[0]).stem}{Path(tdata[0]).suffix}'.strip()]['scaling'].iloc[0])
            except IndexError:
                target_scaling = 1.



            print('Loading mesh...', end='')
            source_mesh, mesh_materials = loadTarget(target_fnme)
            source_mesh = source_mesh.rotate(source_mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0]))).scale(target_scaling,
                                                                                                     center=source_mesh.get_center())
            source_mesh = source_mesh.translate(np.array([0, 0, 0.]), relative=False)
            mesh = Mesh(
                source_mesh,
                max_tris_per_split=64,
                material_sigma=[mesh_materials[mtid][1] if mtid in mesh_materials.keys() else mesh_materials[0][1] for mtid in
                                range(np.asarray(source_mesh.triangle_material_ids).max() + 1)],
                material_emissivity=[mesh_materials[mtid][0] if mtid in mesh_materials.keys() else mesh_materials[0][0] for mtid in
                                     range(np.asarray(source_mesh.triangle_material_ids).max() + 1)],
            )
        scene.add(mesh)

        model_name = f'/home/jeff/repo/apache/data/target_meshes/{Path(target_fnme).stem}.model'

        with open(model_name, 'wb') as f:
            pickle.dump(scene, f)

        with open(model_name, 'rb') as f:
            scene = pickle.load(f)
        print('Sampling mesh...', end='')
        sample_points = [scene.sample(n_samples, a_obs_pts=poses[::16])]

        streams = [cuda.stream()]
        rpfig = go.Figure()
        mfig = go.Figure()
        pfig = go.Figure()

        # Single pulse for debugging
        print('Generating single pulse...', end='')
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

        # rpfig.add_scatter(y=db(single_rp[0][0].flatten()), mode='markers')
        # pfig.add_scatter(y=db(single_pulse[0].flatten()), mode='markers')
        mfig.add_scatter(y=db(single_mf_pulse[0].flatten()), mode='markers')

        zranges = [scene.bounding_box[0, 2] - 1, scene.bounding_box[1, 2] + 1]

        bounce_colors = ['blue', 'red', 'green', 'yellow']
        for bounce in range(len(ray_origins)):
            fig = getSceneFig(scene,
                              title=f'Bounce {bounce}', zrange=zranges)
            for idx, (ro, rd, nrp) in enumerate(
                    zip(ray_origins[:bounce + 1], ray_directions[:bounce + 1], ray_powers[:bounce + 1])):
                valids = nrp[pulse_idx] > 0.
                sc = (1 + nrp[pulse_idx, valids] / nrp[pulse_idx, valids].max()) * 10
                fig.add_trace(go.Cone(x=ro[pulse_idx, valids, 0], y=ro[pulse_idx, valids, 1], z=ro[pulse_idx, valids, 2], u=rd[pulse_idx, valids, 0] * sc,
                                      v=rd[pulse_idx, valids, 1] * sc, w=rd[pulse_idx, valids, 2] * sc, anchor='tail', sizeref=80,
                                      colorscale=[[0, bounce_colors[idx]], [1, bounce_colors[idx]]]))

            fig.show()

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Choose a colormap
        cmap = cm.get_cmap('viridis')

        # Normalize the data
        tri_materials = np.asarray(source_mesh.triangle_material_ids)
        norm = mcolors.Normalize(vmin=np.min(tri_materials), vmax=np.max(tri_materials))

        # Map data to colors
        rgba_colors = cmap(norm(tri_materials))


        '''fig = getSceneFig(scene, triangle_colors=rgba_colors[:, :3], title='Depth', zrange=zranges)

        for mesh in scene.meshes:
            d = mesh.bvh_levels - 1
            for b in mesh.bvh[sum(2 ** n for n in range(d)):sum(2 ** n for n in range(d + 1))]:
                if np.sum(b) != 0:
                    fig.add_trace(drawOctreeBox(b))
        fig.show()'''

        fig = getSceneFig(scene, title='Depth', zrange=zranges)
        fig.add_scatter3d(x=sample_points[0][:, 0], y=sample_points[0][:, 1], z=sample_points[0][:, 2], mode='markers')
        fig.show()
        # rpfig.show()
        mfig.show()
        # pfig.show()

        plt.figure()
        plt.imshow(db(single_mf_pulse))
        plt.axis('tight')
        print('Done.')



'''fig = go.Figure()
fig.add_trace(go.Cone(x=[ro[0]], y=[ro[1]], z=[ro[2]], u=[rd[0]], v=[rd[1]], w=[rd[2]]))
fig.add_trace(go.Cone(x=[ro[0]], y=[ro[1]], z=[ro[2]], u=[b[0]], v=[b[1]], w=[b[2]]))
fig.show()'''

