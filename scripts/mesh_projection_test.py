from numba import cuda
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from simulib.backproject_functions import getRadarAndEnvironment, backprojectPulseStream
from simulib.simulation_functions import db, genChirp, upsamplePulse, llh2enu, genTaylorWindow
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromMesh, _float
from tqdm import tqdm
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sdrparse import load

from simulib.mesh_objects import Mesh

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def addNoise(range_profile, chirp, npower, mf, fft_len):
    data = (chirp * np.fft.fft(range_profile + np.random.normal(0, npower, range_profile.shape) +
     1j * np.random.normal(0, npower, range_profile.shape), fft_len))
    return data * mf


if __name__ == '__main__':
    fc = 9.6e9
    rx_gain = 22  # dB
    tx_gain = 22  # dB
    rec_gain = 100  # dB
    ant_transmit_power = 100  # watts
    noise_power_db = -120
    npulses = 32
    plp = .75
    fdelay = 10.
    upsample = 8
    num_bounces = 2
    nbox_levels = 5
    nstreams = 2
    points_to_sample = 2**18
    num_mesh_triangles = 1000000
    max_pts_per_run = 2**16
    grid_origin = (40.139343, -111.663541, 1360.10812)
    fnme = '/data6/SAR_DATA/2024/08072024/SAR_08072024_111617.sar'
    triangle_colors = None
    do_randompts = False

    # os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

    sdr_f = load(fnme)
    bg, rp = getRadarAndEnvironment(sdr_f)
    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
            rp.getRadarParams(0., plp, upsample))
    idx_t = sdr_f[0].frame_num[sdr_f[0].nframes // 2 : sdr_f[0].nframes // 2 + npulses]
    data_t = sdr_f[0].pulse_time[idx_t]

    pointing_vec = rp.boresight(data_t).mean(axis=0)

    # gx, gy, gz = bg.getGrid(grid_origin, 201 * .2, 199 * .2, nrows=256, ncols=256, az=-68.5715881976 * DTR)
    # gx, gy, gz = bg.getGrid(grid_origin, 201 * .3, 199 * .3, nrows=256, ncols=256)
    gx, gy, gz = bg.getGrid(grid_origin, 300, 300, nrows=1024, ncols=1024)
    grid_pts = np.array([gx.flatten(), gy.flatten(), gz.flatten()]).T
    grid_ranges = np.linalg.norm(rp.txpos(data_t).mean(axis=0) - grid_pts, axis=1)

    print('Loading mesh...', end='')
    mesh = o3d.geometry.TriangleMesh()
    mesh_ids = []

    '''mesh = readCombineMeshFile('/home/jeff/Documents/roman_facade/scene.gltf', points=3000000)
    mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
    mesh_ids = np.asarray(mesh.triangle_material_ids)'''

    '''mesh = readCombineMeshFile('/home/jeff/Documents/eze_france/scene.gltf', 1e9, scale=1 / 100)
    mesh = mesh.translate(np.array([0, 0, 0]), relative=False)
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox().create_from_points(o3d.utility.Vector3dVector(np.array([[-400, -400, -400],[400, 400, 400]]))))
    mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
    mesh_ids = np.asarray(mesh.triangle_material_ids)'''

    mesh = readCombineMeshFile('/home/jeff/Documents/house_detail/source/1409 knoll lane.obj', 1e6, scale=4)
    mesh = mesh.translate(np.array([0, 0, 0]), relative=False)
    mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
    mesh_ids = np.asarray(mesh.triangle_material_ids)

    '''mesh = readCombineMeshFile('E:\meshes\plot.obj', points=30000000)
    # mesh = mesh.rotate(mesh.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    mesh = mesh.translate(llh2enu(*grid_origin, bg.ref), relative=False)
    mesh_ids = np.asarray(mesh.triangle_material_ids)
    triangle_colors = np.mean(np.asarray(mesh.vertex_colors)[np.asarray(mesh.triangles)], axis=1)'''


    '''car = readCombineMeshFile('/home/jeff/Documents/nissan_sky/NissanSkylineGT-R(R32).obj',
                               points=num_mesh_triangles)  # Has just over 500000 points in the file
    car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    car = car.rotate(car.get_rotation_matrix_from_xyz(np.array([0, 0, -42.51 * DTR])))
    mesh_extent = car.get_max_bound() - car.get_min_bound()
    car = car.translate(np.array([gx.mean() + 1.5, gy.mean() - 1.5, gz.mean() + mesh_extent[2] / 2]), relative=False)
    mesh_ids = np.asarray(car.triangle_material_ids)
    mesh += car
    
    building = readCombineMeshFile('/home/jeff/Documents/target_meshes/long_hangar.obj', points=1e9,
                                   scale=.033).rotate(car.get_rotation_matrix_from_xyz(np.array([np.pi / 2, 0, 0])))
    stretch = np.eye(4)
    stretch[2, 2] = 2.8
    building = building.transform(stretch)
    building = building.translate(llh2enu(40.139642, -111.663817, 1380, bg.ref) + np.array([-30, -50, -12.]),
                                  relative=False)
    building = building.rotate(building.get_rotation_matrix_from_xyz(np.array([0, 0, -42.51 * DTR])))
    building = building.compute_triangle_normals()
    mesh_ids = np.concatenate((mesh_ids, np.asarray(building.triangle_material_ids) + mesh_ids.max()))
    mesh += building
    
    gpx, gpy, gpz = bg.getGrid(grid_origin, 201 / 2, 199 / 2, nrows=201, ncols=199, az=-68.5715881976 * DTR)
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
        mesh_ids = np.zeros(len(ground.triangles)).astype(int)'''

    grid_extent = np.array([gx.max() - gx.min(), gy.max() - gy.min(), gz.max() - gz.min()])
    mesh.triangle_material_ids = o3d.utility.IntVector([int(m) for m in mesh_ids])
    face_points = np.asarray(mesh.vertices)
    print('Done.')

    # This is all the constants in the radar equation for this simulation
    radar_coeff = (c0**2 / fc**2 * ant_transmit_power * 10**((rx_gain + 2.15) / 10) * 10**((tx_gain + 2.15) / 10) *
                   10**((rec_gain + 2.15) / 10) / (4 * np.pi)**3)
    noise_power = 10**(noise_power_db / 10)

    # Generate a chirp
    chirp_bandwidth = 400e6
    chirp = genChirp(nr, fs, fc, chirp_bandwidth)
    fft_chirp = np.fft.fft(chirp, fft_len)
    taytay = genTaylorWindow(fc % fs, chirp_bandwidth / 2, fs, fft_len)
    mf_chirp = fft_chirp.conj() * taytay


    # Load in boxes and meshes for speedup of ray tracing
    print('Loading mesh box structure...', end='')
    ptsam = min(points_to_sample, max_pts_per_run)
    '''try:
        msigmas = [2. for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
        msigmas[0] = msigmas[15] = .5  # seats
        msigmas[6] = msigmas[13] = msigmas[17] = .5  # body
        msigmas[12] = msigmas[4] = .2  # windshield
        mkds = [.5 for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
        mkds[0] = mkds[15] = .8  # seats
        mkds[6] = mkds[13] = mkds[17] = .8  # body
        mkds[12] = mkds[4] = .1  # windshield
        mkss = [.5 for _ in range(np.asarray(mesh.triangle_material_ids).max() + 1)]
        mkss[0] = mkss[15] = .2  # seats
        mkss[6] = mkss[13] = mkss[17] = .8  # body
        mkss[12] = mkss[4] = .01  # windshield
        # msigmas[28] = 2
        mesh = Mesh(mesh, num_box_levels=nbox_levels, material_sigmas=msigmas, material_kd=mkds, material_ks=mkss, use_box_pts=True)
    except ValueError:
        print('Error in getting material sigmas.')
        mesh = Mesh(mesh, num_box_levels=nbox_levels, use_box_pts=True)'''
    mesh = Mesh(mesh, num_box_levels=nbox_levels)
    print('Done.')

    if do_randompts:
        sample_points = ptsam
    else:
        sample_points = mesh.sample(ptsam, view_pos=rp.txpos(rp.gpst[np.linspace(0, len(rp.gpst) - 1, 4).astype(int)]))
    boresight = rp.boresight(sdr_f[0].pulse_time).mean(axis=0)
    pointing_az = np.arctan2(boresight[0], boresight[1])

    # Locate the extrema to speed up the optimization
    flight_path = rp.txpos(sdr_f[0].pulse_time)
    sp = mesh.vertices if do_randompts else sample_points
    pmax = sp.max(axis=0)
    vecs = np.array([pmax[0] - flight_path[:, 0], pmax[1] - flight_path[:, 1],
                     pmax[2] - flight_path[:, 2]]).T
    pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
    max_pts = sdr_f[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw * 2]
    pmin = sp.min(axis=0)
    vecs = np.array([pmin[0] - flight_path[:, 0], pmin[1] - flight_path[:, 1],
                     pmin[2] - flight_path[:, 2]]).T
    pt_az = np.arctan2(vecs[:, 0], vecs[:, 1])
    min_pts = sdr_f[0].frame_num[abs(pt_az - pointing_az) < rp.az_half_bw * 2]
    pulse_lims = [max(min(min(max_pts), min(min_pts)) - 1000, 0), min(max(max(max_pts), max(min_pts)) + 1000, sdr_f[0].frame_num[-1])]
    # pulse_lims = [0, sdr_f[0].nframes]
    streams = [cuda.stream() for _ in range(nstreams)]

    # Single pulse for debugging
    print('Generating single pulse...')
    single_rp, ray_origins, ray_directions, ray_powers = getRangeProfileFromMesh(mesh, sample_points,
                                                                                 [rp.txpos(data_t).astype(_float)], [rp.rxpos(data_t).astype(_float)],
                                                                                 [rp.pan(data_t).astype(_float)], [rp.tilt(data_t).astype(_float)], radar_coeff,
                                                                                 rp.az_half_bw, rp.el_half_bw,
                                                                                 nsam, fc, near_range_s,
                                                                                 num_bounces=num_bounces,
                                                                                 debug=True, streams=streams)
    single_pulse = upsamplePulse(fft_chirp * np.fft.fft(single_rp[0], fft_len), fft_len, upsample,
                                 is_freq=True, time_len=nsam)
    single_mf_pulse = upsamplePulse(
        addNoise(single_rp[0], fft_chirp, noise_power, mf_chirp, fft_len), fft_len, upsample,
        is_freq=True, time_len=nsam)
    bpj_grid = np.zeros_like(gx).astype(np.complex128)

    print('Running main loop...')
    # Get the data into CPU memory for later
    # MAIN LOOP
    # If we need to split the point raster, do so
    if points_to_sample > max_pts_per_run:
        splits = np.concatenate((np.arange(0, points_to_sample, max_pts_per_run), [points_to_sample]))
    else:
        splits = np.array([0, points_to_sample])
    for s in range(len(splits) - 1):
        if s > 0:
            if do_randompts:
                sample_points = ptsam
            else:
                sample_points = mesh.sample(int(splits[s + 1] - splits[s]), view_pos=rp.txpos(rp.gpst[np.linspace(0, len(rp.gpst) - 1, 4).astype(int)]))
        for frame in tqdm(list(zip(*(iter(range(pulse_lims[0], pulse_lims[1] - npulses, npulses)),) * nstreams))):
            txposes = [rp.txpos(sdr_f[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
            rxposes = [rp.rxpos(sdr_f[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
            pans = [rp.pan(sdr_f[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
            tilts = [rp.tilt(sdr_f[0].pulse_time[frame[n]:frame[n] + npulses]).astype(_float) for n in range(nstreams)]
            trp = getRangeProfileFromMesh(mesh, sample_points, txposes, rxposes, pans, tilts,
                                          radar_coeff, rp.az_half_bw, rp.el_half_bw, nsam, fc, near_range_s,
                                          num_bounces=num_bounces, streams=streams)
            mf_pulses = [np.ascontiguousarray(upsamplePulse(addNoise(range_profile, fft_chirp, noise_power, mf_chirp, fft_len), fft_len, upsample, is_freq=True, time_len=nsam).T, dtype=np.complex128) for range_profile in trp]
            bpj_grid += backprojectPulseStream(mf_pulses, pans, tilts, rxposes, txposes, gz.astype(_float),
                                                c0 / fc, near_range_s, fs * upsample, rp.az_half_bw, rp.el_half_bw,
                                                gx=gx.astype(_float), gy=gy.astype(_float), streams=streams)


    def getMeshFig(title='Title Goes Here', zrange=100):
        fig = go.Figure(data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                # i, j and k give the vertices of triangles
                i=mesh.tri_idx[:, 0],
                j=mesh.tri_idx[:, 1],
                k=mesh.tri_idx[:, 2],
                facecolor=triangle_colors,
                showscale=True
            )
        ])
        fig.update_layout(
            title=title,
            scene=dict(zaxis=dict(range=[-30, zrange])),
        )
        return fig

    px.scatter(db(single_rp[0][0].flatten())).show()
    px.scatter(db(single_pulse[0].flatten())).show()
    px.scatter(db(single_mf_pulse[0].flatten())).show()

    '''plt.figure('Data')
    plt.imshow(db(single_mf_pulse * 1e8))
    plt.xlabel('Interpolated Range Bin')
    plt.ylabel('Pulse Number')
    plt.title('Example CPI - Generated Data')
    plt.colorbar()
    plt.axis('tight')
    plt.show()'''

    px.imshow(db(single_mf_pulse * 1e8)).show()

    # plt.figure('Backprojection')
    db_bpj = db(bpj_grid)
    '''plt.imshow(db_bpj, cmap='gray', origin='lower', clim=[np.mean(db_bpj), np.mean(db_bpj) + np.std(db_bpj) * 2])
    plt.axis('tight')
    plt.axis('off')
    plt.show()'''

    px.imshow(db_bpj, origin='lower', color_continuous_scale='gray',
              range_color=[np.mean(db_bpj), np.mean(db_bpj) + np.std(db_bpj) * 2]).show()

    scaling = min(r.min() for r in ray_powers), max(r.max() for r in ray_powers)
    sc_min = scaling[0] - 1e-3
    sc = 1 / (scaling[1] - scaling[0])
    scaled_rp = (ray_powers[0] - sc_min) * sc

    fig = getMeshFig('Full Mesh', flight_path[:, 2].mean() + 10)
    fig.add_trace(go.Scatter3d(x=flight_path[::100, 0], y=flight_path[::100, 1], z=flight_path[::100, 2], mode='lines'))
    fig.show()

    bounce_colors = ['blue', 'red', 'green', 'yellow']
    for bounce in range(len(ray_origins)):
        fig = getMeshFig(f'Bounce {bounce}')
        for idx, (ro, rd, nrp) in enumerate(zip(ray_origins[:bounce + 1], ray_directions[:bounce + 1], ray_powers[:bounce + 1])):
            valids = nrp[0] > 0.
            sc = (1 + nrp[0, valids] / nrp[0, valids].max()) * 10
            fig.add_trace(go.Cone(x=ro[0, valids, 0], y=ro[0, valids, 1], z=ro[0, valids, 2], u=rd[0, valids, 0] * sc,
                              v=rd[0, valids, 1] * sc, w=rd[0, valids, 2] * sc, anchor='tail', sizeref=40,
                                  colorscale=[[0, bounce_colors[idx]], [1, bounce_colors[idx]]]))

        fig.show()

    fig = getMeshFig('Ray trace')
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
    fig.update_layout(
            scene=dict(zaxis=dict(range=[-30, 100]),
                       xaxis=dict(range=[2150, 2550]),
                       yaxis=dict(range=[-800, -1000])),
        )
    fig.show()


    def drawbox(box):
        vertices = []
        for z in range(2):
            vertices.extend(
                (
                    [box[0, 0], box[0, 1], box[z, 2]],
                    [box[1, 0], box[0, 1], box[z, 2]],
                    [box[1, 0], box[0, 1], box[int(not z), 2]],
                    [box[1, 0], box[0, 1], box[z, 2]],
                    [box[1, 0], box[1, 1], box[z, 2]],
                    [box[1, 0], box[1, 1], box[int(not z), 2]],
                    [box[1, 0], box[1, 1], box[z, 2]],
                    [box[0, 0], box[1, 1], box[z, 2]],
                    [box[0, 0], box[1, 1], box[int(not z), 2]],
                    [box[0, 0], box[1, 1], box[z, 2]],
                    [box[0, 0], box[0, 1], box[z, 2]],
                )
            )
        vertices = np.array(vertices)
        return go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='lines')

    fig = getMeshFig()

    for b in mesh.octree[sum(8**n for n in range(nbox_levels - 1)):]:
        if np.sum(b) != 0:
            fig.add_trace(drawbox(b))

    fig.show()

    fig = px.scatter_3d(x=sample_points[:256, 0], y=sample_points[:256, 1], z=sample_points[:256, 2])
    for n in range(256, 8192, 256):
        fig.add_trace(
            go.Scatter3d(x=sample_points[n:n + 256, 0], y=sample_points[n:n + 256, 1], z=sample_points[n:n + 256, 2],
                         mode='markers'))
    fig.show()

    tri_pcd = o3d.geometry.PointCloud()
    tri_pcd.points = o3d.utility.Vector3dVector(mesh.vertices[mesh.tri_idx].mean(axis=1))
    tri_pcd.normals = o3d.utility.Vector3dVector(mesh.normals)
    o3d.visualization.draw_geometries([tri_pcd])

    '''import simplekml
    from simulib import enu2llh
    kml = simplekml.Kml()
    lat, lon, alt = enu2llh(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], bg.ref)
    lin = kml.newlinestring(name='Flight Line', coords=[(lo, la, al - 1380) for la, lo, al in zip(lat, lon, alt)])
    kml.save('/home/jeff/repo/test.kml')'''




