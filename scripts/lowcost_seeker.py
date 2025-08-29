# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
from numba import cuda
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal.windows import taylor
from simulib.platform_helper import RadarPlatform
from simulib.grid_helper import MapEnvironment
from simulib.simulation_functions import db, genChirp, upsamplePulse, llh2enu, genTaylorWindow, enu2llh, getRadarCoeff, \
    azelToVec
from scipy.signal import sawtooth
from simulib.mimo_functions import genChannels
from simulib.mesh_functions import _float, getRangeProfileFromScene, \
    getMeshFig, getSceneFig, drawOctreeBox, loadTarget, drawAntennaBox
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
from simulib.mesh_objects import TriangleMesh, Scene, OceanMesh
from simulib.mesh_functions import genOceanBackground

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
    fs = 250e6
    rx_gain = 60  # dB
    tx_gain = 60  # dB
    rec_gain = 120  # dB
    up_gain = 150  # dB
    ant_transmit_power = 220  # watts
    noise_figure = 5
    operating_temperature = 290  # degrees Kelvin
    npulses = 64
    plp = .85
    upsample = 4
    dopp_upsample = 1
    num_bounces = 1
    max_tris_per_split = 64
    points_to_sample = 2**18
    num_mesh_triangles = 1000000
    max_pts_per_run = 2**18
    prf = 3236.
    bw_az = 10
    bw_el = 3
    chirp_bandwidth = 123e6
    # grid_origin = (40.139343, -111.663541, 1360.10812)
    # fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'
    # fnme = '/home/jeff/SDR_DATA/RAW/12172024/SAR_12172024_113146.sar'
    grid_origin = np.array([40.138044, -111.660027, 1365.8849123907273])
    # grid_origin = (40.198354, -111.924774, 1560.)
    # fnme = '/data6/SAR_DATA/2024/08222024/SAR_08222024_121824.sar'
    triangle_colors = None
    supersamples = 0
    nbpj_pts = (256, 256)
    cons_ranges = (10000, 15000.)

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
    # u = u * np.sqrt(np.linspace(1, 1e-9, ngps))
    # e = np.cos(np.ones(ngps) * 1 * np.pi * pulse_times[:ngps]) * 200 + 1600.
    # e = np.ones(ngps)
    # n = np.linspace(launch_speed * launch_time, 0, ngps)
    n = np.ones(ngps) * -13700.
    e = np.linspace(-7000., 7000, ngps)
    gpst = np.linspace(0, launch_time, ngps)
    poses = -np.array([e, n, u]).T
    vel = CubicSpline(gpst, median_filter(np.gradient(np.array([e, n, u]).T, axis=0), 15, axes=(0,)) * 100)
    missile_vels = vel(gpst)
    r = np.zeros(ngps)
    p = np.arcsin(missile_vels[:, 2] / np.linalg.norm(missile_vels, axis=1))
    y = np.arctan2(missile_vels[:, 0], missile_vels[:, 1])

    # Point at the target the whoooole way
    pt_vec = poses / np.linalg.norm(poses, axis=1)[:, None]
    gimbal = np.array([np.arctan2(pt_vec[:, 0], pt_vec[:, 1]) - y + np.pi / 2, np.pi / 2+np.arcsin(pt_vec[:, 2]) - p]).T
    # gimbal = np.array([sawtooth(np.pi * (45 / 45) *
    #                     np.arange(ngps) / 100, .5) * 45 / 2 * DTR + .001, p - np.arcsin(pt_vec[:, 2])]).T


    # Get the transmit channels going
    rp = RadarPlatform(e=e, n=n, u=u, r=r, p=p, y=y, t=np.linspace(0, launch_time, ngps), tx_offset=np.zeros(3),
             rx_offset=np.zeros(3), gimbal=gimbal, gimbal_offset=np.array([0., 0, 0]),
                       gimbal_rotations=np.array([0, 0, np.pi/2]), dep_angle=0,
             squint_angle=0., az_bw=bw_az, el_bw=bw_el, fs=fs, tx_num=0, rx_num=0)

    '''test = azelToVec(rp.pan(rp.gpst), rp.tilt(rp.gpst)).T
    check = (np.sinc(1 / (bw_az * DTR) * (np.arctan2(pt_vec[:, 0], pt_vec[:, 1]) - rp.pan(rp.gpst)))**2 * np.sinc(
        1 / (bw_el * DTR) * (-np.arcsin(pt_vec[:, 2]) - rp.tilt(rp.gpst)))**2)'''

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(launch_height, plp, upsample, a_ranges=cons_ranges))

    pulse_times = pulse_times[
        np.logical_and(
            np.linalg.norm(rp.pos(pulse_times), axis=1) > cons_ranges[0],
            np.linalg.norm(rp.pos(pulse_times), axis=1) < cons_ranges[1],
        )
    ]

    # gx, gy, gz = bg.getGrid(grid_origin, 201 * .2, 199 * .2, nrows=256, ncols=256, az=-68.5715881976 * DTR)
    # gx, gy, gz = bg.getGrid(grid_origin, 201 * .3, 199 * .3, nrows=256, ncols=256)
    gx, gy, gz = bg.getGrid(grid_origin, 50, 50, nrows=nbpj_pts[0], ncols=nbpj_pts[1])
    grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T

    print('Loading mesh...', end='')

    # scene = Scene()
    with open('/home/jeff/repo/apache/data/target_meshes/frigate.model', 'rb') as f:
        scene = pickle.load(f)

    scene.shift(np.array([0., 0., 10.]))

    '''mesh, mesh_materials = loadTarget('/home/jeff/Documents/roman_facade/scene.targ')
    scene.add(
        TriangleMesh(
            mesh,
            max_tris_per_split=max_tris_per_split,
            material_sigma=[mesh_materials[mtid][0] for mtid in
                            range(np.asarray(mesh.triangle_material_ids).max() + 1)],
            material_emissivity=[mesh_materials[mtid][1] for mtid in
                                 range(np.asarray(mesh.triangle_material_ids).max() + 1)],
        )
    )'''

    print('Loading ocean...', end='')

    ostack, x, y = genOceanBackground((10000, 10000), pulse_times, repetition_T=1000., fft_grid_sz=(128, 128), u10=10., numrings=30)

    # Reshape the background into something we can use
    tri_ = Delaunay(np.array([x, y]).T)
    ocean = OceanMesh(np.array([x[0], y[0], 0]), tri_.simplices, ostack, x, y)
    ocean.shift(np.array([0, 0, 0.]))
    scene.add(ocean)


    scene.shift(llh2enu(*grid_origin, bg.ref), relative=True)

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

    doppwin = taylor(npulses * dopp_upsample)


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
    idx_t = np.arange(len(pulse_times) // 2, len(pulse_times) // 2 + npulses)
    data_t = pulse_times[idx_t]
    print('Generating single pulse...')
    single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromScene(scene, sample_points,
                                                                                  [r.txpos(data_t).astype(_float) for r in datasets],
                                                                                  [r.rxpos(data_t).astype(_float) for r in datasets],
                                                                                  [r.pan(data_t).astype(_float) for r in datasets],
                                                                                  [r.tilt(data_t).astype(_float) for r in datasets],
                                                                                  radar_coeff, rp.az_half_bw, rp.el_half_bw,
                                                                                  nsam, fc, near_range_s, fs,
                                                                                  num_bounces=num_bounces,
                                                                                  debug=True, supersamples=supersamples, frames=idx_t)
    single_pulse = [
        sum(
            upsamplePulse(
                10**(up_gain / 20) * f * np.fft.fft(s, fft_len) + np.random.normal(0, noise_power * chirp_bandwidth, f.shape) + 1j * np.random.normal(0, noise_power * chirp_bandwidth, f.shape),
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
        if frame[0] + npulses > ocean.normals.shape[0]:
            break
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
        txposes = [r.txpos(ptimes).astype(_float) for r in datasets]
        rxposes = [r.rxpos(ptimes).astype(_float) for r in datasets]
        pans = [r.pan(ptimes).astype(_float) for r in datasets]
        tilts = [r.tilt(ptimes).astype(_float) for r in datasets]
        trp = getRangeProfileFromScene(scene, sample_points, txposes, rxposes, pans, tilts,
                                        radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s, fs,
                                        num_bounces=num_bounces, supersamples=supersamples, frames=np.arange(frame[0], frame[0] + npulses))
        rx_pulse = [sum(10**(up_gain / 20) * (f * np.fft.fft(s, fft_len) + np.random.normal(0, noise_power * chirp_bandwidth, f.shape) +
                      1j * np.random.normal(0, noise_power * chirp_bandwidth, f.shape)) for s, f in zip(srp, fft_chirps)) for srp in trp]
        mf_pulses = [[np.ascontiguousarray(upsamplePulse(rpulse * mf, fft_len, upsample, is_freq=True,
            time_len=nsam).T, dtype=np.complex64) for rpulse in rx_pulse] for mf in mf_chirps]
        '''bpj_grid += backprojectPulseStream(mf_pulses[0], [pans[0]], [rxposes[0]], [txposes[0]], gz.astype(_float),
                                            c0 / fc, near_range_s, fs * upsample, rp.az_half_bw,
                                            gx=gx.astype(_float), gy=gy.astype(_float), streams=streams)'''

        dopp_image = np.fft.fftshift(np.fft.fft(mf_pulses[0][0], npulses * dopp_upsample, axis=1), axes=1) * doppwin[None, :]
        exp_ranges = np.linalg.norm(txposes[0], axis=1)
        bins = [int((2 * exp_ranges[0] / c0 - 2. * near_range_s) * fs), int((2 * exp_ranges[-1] / c0 - 2. * near_range_s) * fs)]
        plot_image = dopp_image[min(bins)-12:max(bins)+12, :]
        # print(f'{frame}: {np.mean(abs(dopp_image))}')
        # dopp_image = np.fft.fftshift(np.fft.fft2(single_mf_pulse[0][0].T, (up_fft_len, 256)), axes=1)
        ax.set_title(f'Pulse {frame[0]}')
        plt.axis('tight')
        im = ax.imshow(db(dopp_image), origin='lower', extent=(dopp_freq[0], dopp_freq[-1], ranges[0], ranges[-1]))
        title_text = ax.text(.5, 1.05, f'Pulse {frame[0]}', transform=ax.transAxes, ha='center', va='top')
        ims.append([im, title_text])

        # Calculate out beampattern points
        az_beamlines = [min(rp.pan(ptimes) - rp.az_half_bw), max(rp.pan(ptimes) + rp.az_half_bw)]
        el_beamlines = [max(rp.tilt(ptimes) - rp.az_half_bw), min(rp.tilt(ptimes) + rp.az_half_bw)]
        ground_ranges = [ranges[0] * np.sin(rp.tilt(ptimes[0]) + rp.el_half_bw), ranges[-1] * np.sin(rp.tilt(ptimes[0]) - rp.el_half_bw)]
        beamlines = azelToVec(az_beamlines, el_beamlines).T
        beampoints = np.roll(np.concatenate((beamlines * ground_ranges[0] + txposes[0][0, 0], beamlines * ground_ranges[-1] + txposes[0][0, 0])), axis=0, shift=1)
        bsights = rp._tx.boresight(ptimes).T
        # plt.title(f'CPI {frame}')
        # plt.draw()
        # plt.pause(.1)


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

    fig = go.Figure(data=[
        go.Mesh3d(
            x=ocean.vertices[0, :, 0],
            y=ocean.vertices[0, :, 1],
            z=ocean.vertices[0, :, 2],
            # i, j and k give the vertices of triangles
            i=ocean.tri_idx[:, 0],
            j=ocean.tri_idx[:, 1],
            k=ocean.tri_idx[:, 2],
            facecolor=ocean.normals[0],
            showscale=True
        )
    ])
    tri_centers = ocean.vertices[0][ocean.tri_idx].mean(axis=1)
    fig.add_trace(go.Cone(x=tri_centers[:, 0],
            y=tri_centers[:, 1],
            z=tri_centers[:, 2],
            # i, j and k give the vertices of triangles
            u=ocean.normals[0, :, 0],
            v=ocean.normals[0, :, 1],
            w=ocean.normals[0, :, 2],))

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
    omax = ocean.vertices.max(axis=(0, 1))
    omin = ocean.vertices.min(axis=(0, 1))
    txposes = rp.txpos(pulse_times[:32]).astype(_float)
    bores = rp._tx.boresight(pulse_times[:32]).T
    '''xrange = [min(omin[0], txposes[:, 0].min()) - 100, max(omax[0], txposes[:, 0].max()) + 100]
    yrange = [min(omin[1], txposes[:, 1].min()) - 100, max(omax[1], txposes[:, 1].max()) + 100]
    zrange = [min(omin[2], -10), txposes[:, 2].max() + 10]'''
    xrange = [min(omin[0], txposes[:, 0].min()) - 15000, max(omax[0], txposes[:, 0].max()) + 15000]
    yrange = [min(omin[1], txposes[:, 1].min()) - 15000, max(omax[1], txposes[:, 1].max()) + 15000]
    zrange = [min(omin[2], -10) - 15000, txposes[:, 2].max() + 10]
    scene_layout = dict(
        xaxis=dict(range=xrange),
        yaxis=dict(range=yrange),
        zaxis=dict(range=zrange),
        aspectratio=dict(x=(xrange[1] - xrange[0]) / (zrange[1] - zrange[0]),y=(yrange[1] - yrange[0]) / (zrange[1] - zrange[0]),z=1),
    )

    pfig = go.Figure(
        data=[go.Cone(x=txposes[:, 0], y=txposes[:, 1], z=txposes[:, 2], u=bores[:, 0], v=bores[:, 1], w=bores[:, 2], showscale=False),
              drawAntennaBox(txposes[0], rp.pan(pulse_times[0]) - rp.az_half_bw, rp.pan(pulse_times[0]) + rp.az_half_bw,
                             rp.tilt(pulse_times[0]) - rp.el_half_bw, rp.tilt(pulse_times[0]) + rp.el_half_bw, 14000, 15000),
              go.Mesh3d(
            x=ocean.vertices[0, :, 0],
            y=ocean.vertices[0, :, 1],
            z=ocean.vertices[0, :, 2],
            # i, j and k give the vertices of triangles
            i=ocean.tri_idx[:, 0],
            j=ocean.tri_idx[:, 1],
            k=ocean.tri_idx[:, 2],
                  facecolor=ocean.normals[0, :], showscale=False
        )],
        layout=go.Layout(
            scene=scene_layout,
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None, {"frame": {"duration": 10, "redraw": True},
                                               "fromcurrent": True,
                                               "transition": {"duration": 6, "easing": "quadratic-in-out"}}]),
                             dict(label="Pause",
                                  method="animate",
                                  args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                 "mode": "immediate",
                                                 "transition": {"duration": 0}}])]
                )
            ],
        )
    )
    pframes = []

    # go.Cone(x=txposes[:, 0], y=txposes[:, 1], z=txposes[:, 2], u=bores[:, 0], v=bores[:, 1], w=bores[:, 2], showscale=False),

    for frame in tqdm(list(zip(*(iter(range(0, len(pulse_times), npulses)),)))):
        ptimes = pulse_times[frame[0]:frame[0] + npulses]
        txposes = rp.txpos(ptimes).astype(_float)
        bores = rp._tx.boresight(ptimes).T
        heights = ocean.vertices[frame[0], ocean.tri_idx].mean(axis=1)[:, 2]
        height_color = mplib.cm.ocean((heights - omin[2]) / (omax[2] - omin[2]))
        pframes.append(go.Frame(data=[go.Cone(x=txposes[:, 0], y=txposes[:, 1], z=txposes[:, 2], u=bores[:, 0],
                                              v=bores[:, 1], w=bores[:, 2], sizeref=1000, sizemode='raw', showscale=False),
                                      drawAntennaBox(txposes[0], rp.pan(ptimes[0]) - rp.az_half_bw,
                                                     rp.pan(ptimes[0]) + rp.az_half_bw,
                                                     rp.tilt(ptimes[0]) - rp.el_half_bw,
                                                     rp.tilt(ptimes[0]) + rp.el_half_bw, 14000, 15000),
                                      go.Mesh3d(
                                          x=ocean.vertices[frame[0], :, 0],
                                          y=ocean.vertices[frame[0], :, 1],
                                          z=ocean.vertices[frame[0], :, 2],
                                          # i, j and k give the vertices of triangles
                                          i=ocean.tri_idx[:, 0],
                                          j=ocean.tri_idx[:, 1],
                                          k=ocean.tri_idx[:, 2],
                                          facecolor=height_color[:, :3], showscale=False
                                      )
                                      ], traces=[0, 1, 2],
                                ))
    pfig.update(frames=pframes)
    pfig.show()

    '''fig, ax = plt.subplots()
    ocean_stack, xx, yy = genOceanBackground((100, 100), pulse_times, repetition_T=1000, S=2., u10=100.)
    plt.figure()

    for o in ocean_stack:
        plt.clf()
        plt.tricontourf(xx, yy, o)
        plt.colorbar()
        # plt.clf()
        # plt.imshow(test.reshape(xx.shape))
        plt.draw()
        plt.pause(.1)'''