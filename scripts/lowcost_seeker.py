import torch
import trimesh
import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
from numba import cuda
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from simulib.platform_helper import SDRPlatform, RadarPlatform
from simulib.grid_helper import MapEnvironment
from simulib.backproject_functions import getRadarAndEnvironment, backprojectPulseStream
from simulib.simulation_functions import db, genChirp, upsamplePulse, llh2enu, genTaylorWindow, enu2llh, getRadarCoeff, \
    azelToVec
from simulib.mimo_functions import genChannels
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromMesh, _float, getRangeProfileFromScene, \
    getMeshFig, getSceneFig, drawOctreeBox, loadTarget, loadMesh
from tqdm import tqdm
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sdrparse import load
import matplotlib as mplib
import matplotlib.animation as anim
import pickle
mplib.use('TkAgg')
from simulib.mesh_objects import Mesh, Scene

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180


def addNoise(range_profile, a_chirp, npower, mf, a_fft_len):
    data = a_chirp * np.fft.fft(range_profile, a_fft_len)
    data = data + np.random.normal(0, npower, data.shape) + 1j * np.random.normal(0, npower, data.shape)
    return data * mf


if __name__ == '__main__':
    fc = 16700e6
    fs = 500e6
    rx_gain = 60  # dB
    tx_gain = 60  # dB
    rec_gain = 120  # dB
    ant_transmit_power = 220  # watts
    noise_figure = 5
    operating_temperature = 290  # degrees Kelvin
    npulses = 64
    plp = .35
    upsample = 1
    dopp_upsample = 2
    num_bounces = 1
    max_tris_per_split = 64
    points_to_sample = 2**16
    num_mesh_triangles = 1000000
    max_pts_per_run = 2**17
    prf = 3236.
    bw_az = 40
    bw_el = 45
    chirp_bandwidth = 245e6
    # grid_origin = (40.139343, -111.663541, 1360.10812)
    # fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'
    # fnme = '/home/jeff/SDR_DATA/RAW/12172024/SAR_12172024_113146.sar'
    grid_origin = np.array([40.138044, -111.660027, 1365.8849123907273])
    # grid_origin = (40.198354, -111.924774, 1560.)
    # fnme = '/data6/SAR_DATA/2024/08222024/SAR_08222024_121824.sar'
    triangle_colors = None
    supersamples = 0
    nbpj_pts = (256, 256)

    bg = MapEnvironment(grid_origin, nbpj_pts, ref=grid_origin, background=np.ones(nbpj_pts))

    plane_pos = llh2enu(40.138052, -111.660027, 1365, bg.ref)

    # Generate a platform
    print('Generating platform...', end='')
    # Run directly at the plane from the south
    launch_time = 160.
    launch_height = 6096.
    launch_speed = 102.8889
    ngps = int(launch_time * 100)
    pulse_times = np.linspace(0, launch_time, int(prf * launch_time))
    # grid_origin = llh2enu(*bg.origin, bg.ref)

    u = np.ones(ngps) * launch_height
    u = u * np.sqrt(np.linspace(1, 1e-9, ngps))
    e = np.cos(np.ones(ngps) * 1 * np.pi * pulse_times[:ngps]) * 200 + 1600.
    n = np.linspace(launch_speed * launch_time, 0, ngps)
    gpst = np.linspace(0, launch_time, ngps)
    poses = -np.array([e, n, u]).T
    vel = CubicSpline(gpst, median_filter(np.gradient(np.array([e, n, u]).T, axis=0), 15, axes=(0,)) * 100)
    missile_vels = vel(gpst)
    r = np.zeros(ngps)
    p = np.arcsin(missile_vels[:, 2] / np.linalg.norm(missile_vels, axis=1))
    y = -np.arctan2(missile_vels[:, 0], missile_vels[:, 1])

    # Point at the target the whoooole way
    pt_vec = poses / np.linalg.norm(poses, axis=1)[:, None]
    gimbal = np.array([-np.arctan2(pt_vec[:, 0], pt_vec[:, 1]) * 0., np.arcsin(pt_vec[:, 2]) * 0.]).T
    # gimbal = np.array([-(y - np.arctan2(pt_vec[:, 0], pt_vec[:, 1])), p - np.arcsin(pt_vec[:, 2])]).T


    # Get the transmit channels going
    rp = RadarPlatform(e=e, n=n, u=u, r=r, p=p, y=y, t=np.linspace(0, launch_time, ngps), tx_offset=np.zeros(3),
             rx_offset=np.zeros(3), gimbal=gimbal, gimbal_offset=np.array([0., 0, 0]),
                       gimbal_rotations=np.array([0, np.pi / 2, 0]), dep_angle=0,
             squint_angle=0., az_bw=bw_az, el_bw=bw_el, fs=fs, tx_num=0, rx_num=0)

    '''test = azelToVec(rp.pan(rp.gpst), rp.tilt(rp.gpst)).T
    check = (np.sinc(1 / (bw_az * DTR) * (np.arctan2(pt_vec[:, 0], pt_vec[:, 1]) - rp.pan(rp.gpst)))**2 * np.sinc(
        1 / (bw_el * DTR) * (-np.arcsin(pt_vec[:, 2]) - rp.tilt(rp.gpst)))**2)'''

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(launch_height, plp, upsample, a_ranges=(14000, 15000.)))

    '''pulse_times = pulse_times[
        np.logical_and(
            np.linalg.norm(rp.pos(pulse_times), axis=1) > 14000.0,
            np.linalg.norm(rp.pos(pulse_times), axis=1) < 15000,
        )
    ]'''

    # gx, gy, gz = bg.getGrid(grid_origin, 201 * .2, 199 * .2, nrows=256, ncols=256, az=-68.5715881976 * DTR)
    # gx, gy, gz = bg.getGrid(grid_origin, 201 * .3, 199 * .3, nrows=256, ncols=256)
    gx, gy, gz = bg.getGrid(grid_origin, 50, 50, nrows=nbpj_pts[0], ncols=nbpj_pts[1])
    grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T

    print('Loading mesh...', end='')

    scene = Scene()
    # mesh_ids = []

    mesh, mesh_materials = loadTarget('/home/jeff/Documents/roman_facade/scene.targ')
    # mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    # mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
    scene.add(
        Mesh(
            mesh,
            max_tris_per_split=max_tris_per_split,
            material_sigma=[mesh_materials[mtid][0] for mtid in
                            range(np.asarray(mesh.triangle_material_ids).max() + 1)],
            material_emissivity=[mesh_materials[mtid][1] for mtid in
                                 range(np.asarray(mesh.triangle_material_ids).max() + 1)],
        )
    )

    '''with open('/home/jeff/repo/apache/data/target_meshes/air_balloon.model', 'rb') as f:
        scene = pickle.load(f)'''
    scene.shift(llh2enu(*grid_origin, bg.ref))

    print('Done.')

    # This is all the constants in the radar equation for this simulation
    radar_coeff = getRadarCoeff(fc, ant_transmit_power, rx_gain, tx_gain, rec_gain)
    noise_power = 10**(noise_figure / 10) * operating_temperature * 1.38e-23

    # Generate a chirp
    # fft_chirp = np.fft.fft(sdr_f[0].cal_chirp, fft_len)
    # mf_chirp = sdr_f.genMatchedFilter(0, fft_len=fft_len)
    chirp = genChirp(nr, fs, fc, chirp_bandwidth)

    fft_chirp = np.fft.fft(chirp, fft_len)
    taytay = genTaylorWindow(fc % fs, chirp_bandwidth / 2, fs, fft_len)
    mf_chirp = taytay / fft_chirp
    fft_chirps = [fft_chirp]
    mf_chirps = [mf_chirp]


    # Load in boxes and meshes for speedup of ray tracing
    print('Loading mesh box structure...', end='')
    ptsam = min(points_to_sample, max_pts_per_run)
    print('Done.')

    # If we need to split the point raster, do so
    if points_to_sample > max_pts_per_run:
        splits = np.concatenate((np.arange(0, points_to_sample, max_pts_per_run), [points_to_sample]))
    else:
        splits = np.array([0, points_to_sample])

    sample_points = [scene.sample(int(splits[s + 1] - splits[s]), rp.txpos(rp.gpst[::100]))
                     for s in range(len(splits) - 1)]
    streams = [cuda.stream()]
    datasets = [rp]

    # Single pulse block for debugging
    data_t = pulse_times[len(pulse_times) // 2:len(pulse_times) // 2 + npulses]
    print('Generating single pulse...')
    single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromScene(scene, sample_points,
                                                                                  [r.txpos(data_t).astype(_float) for r in datasets],
                                                                                  [r.rxpos(data_t).astype(_float) for r in datasets],
                                                                                  [r.pan(data_t).astype(_float) for r in datasets],
                                                                                  [r.tilt(data_t).astype(_float) for r in datasets],
                                                                                  radar_coeff, rp.az_half_bw, rp.el_half_bw,
                                                                                  nsam, fc, near_range_s, fs,
                                                                                  num_bounces=num_bounces,
                                                                                  debug=True, streams=streams,
                                                                                  supersamples=supersamples)
    single_pulse = [
        sum(
            upsamplePulse(
                f * np.fft.fft(s, fft_len) + np.random.normal(0, noise_power * chirp_bandwidth, f.shape) + 1j * np.random.normal(0, noise_power * chirp_bandwidth, f.shape),
                fft_len,
                upsample,
                is_freq=True,
                time_len=nsam,
            )
            for s, f in zip(srp, fft_chirps)
        )
        for srp in single_rp
    ]
    single_mf_pulse = [
        [upsamplePulse(sum(f * np.fft.fft(s, fft_len) for s, f in zip(srp, fft_chirps)) * mf, fft_len, upsample,
                       is_freq=True, time_len=nsam)
            for mf in mf_chirps
        ]
        for srp in single_rp
    ]
    bpj_grid = np.zeros_like(gx).astype(np.complex64)

    dopp_freq = np.fft.fftshift(np.fft.fftfreq(npulses * dopp_upsample, 1 / prf))

    print('Running main loop...')
    # Get the data into CPU memory for later
    # MAIN LOOP
    fig, ax = plt.subplots()
    ims = []
    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
        txposes = [r.txpos(ptimes).astype(_float) for r in datasets]
        rxposes = [r.rxpos(ptimes).astype(_float) for r in datasets]
        pans = [r.pan(ptimes).astype(_float) for r in datasets]
        tilts = [r.tilt(ptimes).astype(_float) for r in datasets]
        trp = getRangeProfileFromScene(scene, sample_points, txposes, rxposes, pans, tilts,
                                        radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s, fs,
                                        num_bounces=num_bounces, streams=streams, supersamples=supersamples)
        rx_pulse = [sum(f * np.fft.fft(s, fft_len) for s, f in zip(srp, fft_chirps)) for srp in trp]
        mf_pulses = [[np.ascontiguousarray(upsamplePulse(rpulse * mf, fft_len, upsample, is_freq=True,
            time_len=nsam).T, dtype=np.complex64) for rpulse in rx_pulse] for mf in mf_chirps]
        mf_pulses = [[rpulse + np.random.normal(0, noise_power * chirp_bandwidth, rpulse.shape) +
                      1j * np.random.normal(0, noise_power * chirp_bandwidth, rpulse.shape) for rpulse in mf] for
                     mf in mf_pulses]
        '''bpj_grid += backprojectPulseStream(mf_pulses[0], [pans[0]], [rxposes[0]], [txposes[0]], gz.astype(_float),
                                            c0 / fc, near_range_s, fs * upsample, rp.az_half_bw,
                                            gx=gx.astype(_float), gy=gy.astype(_float), streams=streams)'''

        dopp_image = np.fft.fftshift(np.fft.fft(mf_pulses[0][0], npulses * dopp_upsample, axis=1), axes=1)
        '''exp_ranges = np.linalg.norm(txposes[0], axis=1)
        bins = [int((2 * exp_ranges[0] / c0 - 2. * near_range_s) * fs), int((2 * exp_ranges[-1] / c0 - 2. * near_range_s) * fs)]
        plot_image = dopp_image[min(bins)-12:max(bins)+12, :]'''
        # print(f'{frame}: {np.mean(abs(dopp_image))}')
        # dopp_image = np.fft.fftshift(np.fft.fft2(single_mf_pulse[0][0].T, (up_fft_len, 256)), axes=1)
        '''plt.axis('tight')
        im = ax.imshow(db(dopp_image), origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[-1], ranges[0]))
        ims.append([im])'''

        # Calculate out beampattern points
        az_beamlines = [min(rp.pan(ptimes) - rp.az_half_bw), max(rp.pan(ptimes) + rp.az_half_bw)]
        el_beamlines = [min(rp.tilt(ptimes) - rp.az_half_bw), max(rp.tilt(ptimes) + rp.az_half_bw)]
        beamlines = azelToVec(az_beamlines, el_beamlines).T
        beampoints = np.concatenate((beamlines * ranges[0] + txposes[0][0, 0], beamlines * ranges[-1] + txposes[0][0, 0]))
        plt.clf()
        plt.title(f'CPI {frame}')
        plt.subplot(1, 2, 1)
        plt.imshow(db(dopp_image), extent=(dopp_freq[0], dopp_freq[-1], ranges[-1], ranges[0]))
        plt.axis('tight')
        plt.subplot(1, 2, 2)
        plt.scatter(txposes[0][:, 0], txposes[0][:, 1])
        plt.scatter([0], [0])
        plt.plot(beampoints[:, 0], beampoints[:, 1])
        plt.draw()
        plt.pause(.1)

    ani = anim.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save('test.gif', fps=60)




    px.scatter(db(np.stack([[s[0] for s in sp] for sp in single_rp])[0].T)).show()
    px.scatter(db(np.stack([sp[0] for sp in single_pulse]).T)).show()
    px.scatter(db(np.stack([[s[0] for s in sp] for sp in single_mf_pulse])[0].T)).show()

    '''plt.figure('Data')
    plt.imshow(db(single_mf_pulse * 1e8))
    plt.xlabel('Interpolated Range Bin')
    plt.ylabel('Pulse Number')
    plt.title('Example CPI - Generated Data')
    plt.colorbar()
    plt.axis('tight')
    plt.show()'''

    # plt.figure('Backprojection')
    db_bpj = db(bpj_grid)
    '''plt.imshow(db_bpj, cmap='gray', origin='lower', clim=[np.mean(db_bpj), np.mean(db_bpj) + np.std(db_bpj) * 2])
    plt.axis('tight')
    plt.axis('off')
    plt.show()'''

    px.imshow(db_bpj, origin='lower', color_continuous_scale='gray',
              range_color=[np.mean(db_bpj), np.mean(db_bpj) + np.std(db_bpj) * 3]).show()

    scaling = min(r.min() for r in ray_powers), max(r.max() for r in ray_powers)
    sc_min = scaling[0] - 1e-3
    sc = 1 / (scaling[1] - scaling[0])
    scaled_rp = (ray_powers[0] - sc_min) * sc

    flight_path = rp.pos(rp.gpst)
    boresights = rp._tx.boresight(rp.gpst).T
    # boresights = azelToVec(y, p).T

    fig = px.scatter_3d(x=flight_path[:, 0], y=flight_path[:, 1], z=flight_path[:, 2])
    fig.add_trace(go.Cone(x=flight_path[:, 0], y=flight_path[:, 1], z=flight_path[:, 2], u=boresights[:, 0],
                              v=boresights[:, 1], w=boresights[:, 2], anchor='tail', sizeref=10))
    fig.show()

    bounce_colors = ['blue', 'red', 'green', 'yellow']
    for bounce in range(len(ray_origins)):
        fig = getSceneFig(scene, triangle_colors=scene.meshes[0].normals if triangle_colors is None else triangle_colors,
                          title=f'Bounce {bounce}')
        for idx, (ro, rd, nrp) in enumerate(zip(ray_origins[:bounce + 1], ray_directions[:bounce + 1], ray_powers[:bounce + 1])):
            valids = nrp[0] > 0.
            sc = (1 + nrp[0, valids] / nrp[0, valids].max()) * 10
            fig.add_trace(go.Cone(x=ro[0, valids, 0], y=ro[0, valids, 1], z=ro[0, valids, 2], u=rd[0, valids, 0] * sc,
                              v=rd[0, valids, 1] * sc, w=rd[0, valids, 2] * sc, anchor='tail', sizeref=10,
                                  colorscale=[[0, bounce_colors[idx]], [1, bounce_colors[idx]]]))

        fig.show()

    fig = getSceneFig(scene, triangle_colors=scene.meshes[0].normals if triangle_colors is None else triangle_colors, title='Ray trace')
    init_pos = np.repeat(rp.txpos(data_t[0]).reshape((1, -1)), ro.shape[1], 0)
    valids = np.sum([nrp[0] > 0.0 for nrp in ray_powers], axis=0) == len(ray_powers)
    ln = np.stack([init_pos[valids]] + [ro[0, valids] for ro in ray_origins]).swapaxes(0, 1)[::1000, :, :]
    for l in ln:
        fig.add_trace(go.Scatter3d(x=l[:, 0], y=l[:, 1], z=l[:, 2], mode='lines'))
    sc_max = ray_powers[0][0, valids].max()
    for ro, rd, nrp in zip(ray_origins, ray_directions, ray_powers):
        sc = ((1 + nrp[0, valids] / nrp[0, valids].max()) * 10)[::1000]
        fig.add_trace(go.Cone(x=ro[0, valids, 0][::1000], y=ro[0, valids, 1][::1000], z=ro[0, valids, 2][::1000], u=rd[0, valids, 0][::1000] * sc,
                          v=rd[0, valids, 1][::1000] * sc, w=rd[0, valids, 2][::1000] * sc, anchor='tail', sizemode='absolute'))
    '''fig.update_layout(
            scene=dict(zaxis=dict(range=[-30, 100]),
                       xaxis=dict(range=[2150, 2550]),
                       yaxis=dict(range=[-800, -1000])),
        )'''
    fig.show()

    fig = getSceneFig(scene, title='Depth')

    for mesh in scene.meshes:
        d = mesh.bvh_levels - 1
        for b in mesh.bvh[sum(2 ** n for n in range(d)):sum(2 ** n for n in range(d+1))]:
            if np.sum(b) != 0:
                fig.add_trace(drawOctreeBox(b))
    fig.show()

    fig = px.scatter_3d(x=sample_points[0][:256, 0], y=sample_points[0][:256, 1], z=sample_points[0][:256, 2])
    for n in range(256, 8192, 256):
        fig.add_trace(
            go.Scatter3d(x=sample_points[0][n:n + 256, 0], y=sample_points[0][n:n + 256, 1], z=sample_points[0][n:n + 256, 2],
                         mode='markers'))
    fig.show()

    '''tri_pcd = o3d.geometry.PointCloud()
    tri_pcd.points = o3d.utility.Vector3dVector(mesh.vertices[mesh.tri_idx].mean(axis=1))
    tri_pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
    o3d.visualization.draw_geometries([tri_pcd])'''

    '''import simplekml
    from simulib import enu2llh
    kml = simplekml.Kml()
    lat, lon, alt = enu2llh(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], bg.ref)
    lin = kml.newlinestring(name='Flight Line', coords=[(lo, la, al - 1380) for la, lo, al in zip(lat, lon, alt)])
    kml.save('/home/jeff/repo/test.kml')'''

    '''px.scatter(x=np.fft.fftfreq(fft_len, 1 / fs), y=db(mf_chirp)).show()

    plt.figure()
    plt.imshow(db(single_rp[0]))
    plt.axis('tight')    tri_pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
    o3d.visualization.draw_geometries([tri_pcd])'''

    '''import simplekml
    from simulib import enu2llh
    kml = simplekml.Kml()
    lat, lon, alt = enu2llh(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], bg.ref)
    lin = kml.newlinestring(name='Flight Line', coords=[(lo, la, al - 1380) for la, lo, al in zip(lat, lon, alt)])
    kml.save('/home/jeff/repo/test.kml')'''

    px.scatter(x=np.fft.fftfreq(fft_len, 1 / fs), y=db(mf_chirp)).show()

    plt.figure()
    plt.imshow(db(single_rp[0][0]))
    plt.axis('tight')