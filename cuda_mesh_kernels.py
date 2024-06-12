import cmath
import math
from numba import cuda, njit
from simulation_functions import findPowerOf2
import numpy as np
import open3d as o3d
from cuda_kernels import applyRadiationPattern, applyRadiationPatternCPU

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def calcBounceFromMesh(vertices, faces, source_xyz, bounce_xyz, bounce_dir):
    pass


@cuda.jit()
def genRangeProfileFromMesh(vert_xyz, vert_norm, vert_reflectivity,
                            source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                            source_fs, bw_az, bw_el, power_scale):
    # sourcery no-metrics
    pidx, t = cuda.grid(ndim=2)
    if pidx < vert_xyz.shape[0] and t < source_xyz.shape[0]:
        # Load in all the parameters that don't change
        n_samples = pd_r.shape[0]
        wavenumber = 2 * np.pi / wavelength

        # Calculate the bounce vector for this time
        vx = vert_xyz[pidx, 0]
        vy = vert_xyz[pidx, 1]
        vz = vert_xyz[pidx, 2]
        vnx = vert_norm[pidx, 0]
        vny = vert_norm[pidx, 1]
        vnz = vert_norm[pidx, 2]

        # Calculate out the angles in azimuth and elevation for the bounce
        tx = vx - source_xyz[t, 0]
        ty = vy - source_xyz[t, 1]
        tz = vz - source_xyz[t, 2]
        rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

        rx = vx - receive_xyz[t, 0]
        ry = vy - receive_xyz[t, 1]
        rz = vz - receive_xyz[t, 2]
        r_rng = math.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
        r_el = -math.asin(rz / r_rng)
        r_az = math.atan2(rx, ry)

        # Calculate bounce vector and strength
        bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
        bx = tx - vnx * bounce_dot
        by = ty - vny * bounce_dot
        bz = tz - vnz * bounce_dot

        rx_strength = (rx * bx + ry * by + rz * bz) / (r_rng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
        if rx_strength < 0:
            return

        two_way_rng = rng + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if but > pd_r.shape[0] or but < 0:
            return

        if n_samples > but > 0:
            gamma = math.pow((1. / -rx_strength + 1.) / 20, 10)
            att = applyRadiationPattern(r_el, r_az, panrx[t], elrx[t], pantx[t], eltx[t], bw_az, bw_el) / (two_way_rng * two_way_rng)
            att *= power_scale
            acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * vert_reflectivity[pidx] * gamma
            cuda.atomic.add(pd_r, (but, np.uint16(t)), acc_val.real)
            cuda.atomic.add(pd_i, (but, np.uint16(t)), acc_val.imag)
        cuda.syncthreads()


def genRangeProfileFromMeshCPU(vert_xyz, vert_norm, vert_reflectivity,
                            source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                            source_fs, bw_az, bw_el, power_scale):
    # sourcery no-metrics
    for pidx in range(vert_xyz.shape[0]):
        for t in range(source_xyz.shape[0]):
            # Load in all the parameters that don't change
            n_samples = pd_r.shape[0]
            wavenumber = 2 * np.pi / wavelength

            # Calculate the bounce vector for this time
            vx = vert_xyz[pidx, 0]
            vy = vert_xyz[pidx, 1]
            vz = vert_xyz[pidx, 2]
            vnx = vert_norm[pidx, 0]
            vny = vert_norm[pidx, 1]
            vnz = vert_norm[pidx, 2]

            # Calculate out the angles in azimuth and elevation for the bounce
            tx = vx - source_xyz[t, 0]
            ty = vy - source_xyz[t, 1]
            tz = vz - source_xyz[t, 2]
            rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))

            rx = vx - receive_xyz[t, 0]
            ry = vy - receive_xyz[t, 1]
            rz = vz - receive_xyz[t, 2]
            r_rng = math.sqrt(abs(rx * rx) + abs(ry * ry) + abs(rz * rz))
            r_el = -math.asin(rz / r_rng)
            r_az = math.atan2(rx, ry)

            # Calculate bounce vector and strength
            bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
            bx = tx - vnx * bounce_dot
            by = ty - vny * bounce_dot
            bz = tz - vnz * bounce_dot

            rx_strength = (rx * bx + ry * by + rz * bz) / (r_rng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
            if rx_strength < 0:
                continue

            two_way_rng = rng + r_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > pd_r.shape[0] or but < 0:
                continue

            if n_samples > but > 0:
                gamma = math.pow(-rx_strength, 5)
                att = applyRadiationPatternCPU(r_el, r_az, panrx[t], elrx[t], pantx[t], eltx[t], bw_az, bw_el) / (two_way_rng * two_way_rng)
                att *= power_scale
                acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * vert_reflectivity[pidx] * gamma
                pd_r[but, t] += acc_val.real
                pd_i[but, t] += acc_val.imag

    return pd_r, pd_i


def barycentric(x0, x1, x2, a0, a1, a2):
    return x0 * a0 + x1 * a1 + x2 * a2


def cart2bary(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1. - v - w
    return u, v, w


def triNormal(x0, x1, x2):
    """
    Calculate a unit normal vector for a triangle.
    """
    A = x1 - x0
    B = x2 - x0
    nx = A[1] * B[2] - A[2] * B[1]
    ny = A[2] * B[0] - A[0] * B[2]
    nz = A[0] * B[1] - A[1] * B[0]
    det = 1 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * det, ny * det, nz * det


def bounceVector(x0, n0):
    """
    Returns the unit vector bouncing off of a normal vector.
    """
    return x0 - 2 * sum(x0 * n0) / sum(n0 * n0) * n0


def transform1(a):
    idx = np.flatnonzero(a[:, -1] == 0)
    out0 = np.empty((a.shape[0], 2, 3), dtype=a.dtype)

    out0[:, 0, 1:] = a[:, 1:-1]
    out0[:, 1, 1:] = a[:, 2:]

    out0[..., 0] = a[:, 0, None]

    out0.shape = (-1, 3)

    mask = np.ones(out0.shape[0], dtype=bool)
    mask[idx * 2 + 1] = 0
    return out0[mask]


def readObjFile(fpath):
    v = []
    fp = []
    fn = []
    vn = []
    with open(fpath, 'r') as f:
        while data := f.readline():
            data = data.strip()
            if len(data) < 2:
                continue
            if data[0] == 'v':
                if data[1] == 'n':
                    un_n = [float(q) for q in data[3:].split(' ')]
                    un_norm = np.linalg.norm(un_n)
                    if un_norm == 0:
                        vn.append([1., 0, 0])
                    else:
                        vn.append([u / un_norm for u in un_n])
                elif data[1] == ' ':
                    v.append([float(q) for q in data[2:].split(' ')])
            elif data[0] == 'f':
                if '/' in data:
                    nodes = [q.split('/') for q in data[2:].split(' ')]
                    if len(nodes) == 3:
                        fp.append([int(q[0]) for q in nodes] + [0])
                        fn.append([int(q[2]) for q in nodes] + [0])
                    elif len(nodes) == 4:
                        fp.append([int(q[0]) for q in nodes])
                        fn.append([int(q[2]) for q in nodes])
                    else:
                        for n in range(4, len(nodes) + 1):
                            fp.append([int(q[0]) for q in nodes[n-4:n]])
                            fn.append([int(q[2]) for q in nodes[n-4:n]])
                else:
                    fp.append([int(q) for q in data[1:].split(' ')])
                    if len(fp[-1]) == 3:
                        fp[-1] += [0]
    return np.array(v), np.array(vn), transform1(np.array(fp)) - 1, transform1(np.array(fn)) - 1


def createMeshFromObj(fnme):
    v, vn, f, fn = readObjFile(fnme)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    return mesh


def readCombineMeshFile(fnme, points=100000):
    full_mesh = o3d.io.read_triangle_model(fnme)
    mesh = full_mesh.meshes[0].mesh
    for me in full_mesh.meshes[1:]:
        mesh += me.mesh

    if len(mesh.triangles) > points:
        mesh = mesh.simplify_quadric_decimation(points)
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()
    return mesh

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from simulation_functions import db, genChirp, azelToVec

    '''full_mesh = o3d.io.read_triangle_model('/home/jeff/Documents/piper_pa18.obj')
    mesh = full_mesh.meshes[0].mesh
    for me in full_mesh.meshes[1:]:
        mesh += me.mesh'''
    mesh = readCombineMeshFile('/home/jeff/Documents/piper_pa18.obj')
    # mesh = sum([t.mesh for t in full_mesh.meshes])
    obs_pt = np.array([100., 0., 50.])
    nsam = 256
    nr = 4096
    fc = 32.0e9
    standoff = 700.
    fft_len = findPowerOf2(nsam + nr)
    up_fft_len = fft_len * 4

    print('Generating Mesh...')
    # mesh.scale(1 / 41.08, center=(0, 0, 0))

    '''mesh = mesh.simplify_vertex_clustering(voxel_size=.01, contraction=o3d.geometry.SimplificationContraction.Average)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh.normalize_normals()'''
    # trimesh.repair.broken_faces(mesh)
    mverts = np.asarray(mesh.vertices)
    mnorms = np.asarray(mesh.vertex_normals)

    print('Sampling Points...')
    sample_points = mesh.sample_points_poisson_disk(3000)

    face_centers = np.asarray(sample_points.points)
    face_normals = np.asarray(sample_points.normals)
    bounces = np.array([bounceVector(fpt - obs_pt, fnorm) for fpt, fnorm in zip(face_centers, face_normals)])
    bounces = bounces / np.linalg.norm(bounces, axis=1)[:, None]
    powers = np.sum(bounces * face_normals, axis=1)
    powers[powers < 0.6] = 0
    bounces *= powers[:, None]

    obs_dir = np.mean([(fpt - obs_pt) / np.linalg.norm(obs_pt - fpt) for fpt in face_centers], axis=0)

    # Get some locations
    pans, tilts = np.meshgrid(np.linspace(0, 2 * np.pi, 16, endpoint=False),
                              np.linspace(np.pi / 2 - .1, -np.pi / 2 + .1, 16))
    pan = pans.flatten()
    tilt = tilts.flatten()
    poses = azelToVec(pan, tilt).T * standoff
    pan[pan == 0] = 1e-9
    pan[pan == 2 * np.pi] = 1e-9
    tilt[tilt == 0] = 1e-9
    tilt[tilt == 2 * np.pi] = 1e-9

    bounces = np.array([bounceVector(fpt - obs_pt, fnorm) for fpt, fnorm in zip(face_centers, face_normals)])
    bounces = bounces / np.linalg.norm(bounces, axis=1)[:, None]
    powers = np.sum(bounces * face_normals, axis=1)
    powers[powers < 0.6] = 0
    bounces *= powers[:, None]

    # Get some pans and tilts
    pd_r = np.zeros((nsam, len(pan)))
    pd_i = np.zeros((nsam, len(pan)))
    near_range_s = (standoff - 10) / c0

    print('Generating range profile...')
    pd_r, pd_i = genRangeProfileFromMeshCPU(face_centers, face_normals, np.ones(face_centers.shape[0]) * 1e9, poses, poses,
                                            pan, tilt, pan, tilt, pd_r, pd_i, c0 / fc, near_range_s, fs, 10 * DTR,
                                            10 * DTR, 1e9)

    pd = pd_r + 1j * pd_i
    pd = pd / abs(pd).max()
    chirp = np.fft.fft(genChirp(nr, fs, fc, 400e6), fft_len)
    mfilt = chirp.conj()

    pd_fft = np.fft.fft(pd, fft_len, axis=0) * chirp[:, None] * mfilt[:, None] / fft_len
    pd_ch = np.zeros((up_fft_len, pd.shape[1]), dtype=np.complex128)
    pd_ch[:fft_len // 2, :] = pd_fft[:fft_len // 2, :]
    pd_ch[-fft_len // 2:, :] = pd_fft[-fft_len // 2:, :]

    pd_ch = np.fft.ifft(pd_ch, axis=0)[:nsam * 4, :]
    # pd_ch += np.random.normal(size=pd_ch.shape) + 1j * np.random.normal(size=pd_ch.shape)

    fig = plt.figure('Mesh')
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mverts[:, 0], mverts[:, 1], mverts[:, 2], triangles=np.asarray(mesh.triangles))

    fig = plt.figure('Normals')
    ax = fig.add_subplot(projection='3d')
    ax.quiver(mverts[:, 0], mverts[:, 1], mverts[:, 2], mnorms[:, 0],
              mnorms[:, 1], mnorms[:, 2])

    fig = plt.figure('Face Normals')
    ax = fig.add_subplot(projection='3d')
    ax.quiver(face_centers[:, 0], face_centers[:, 1], face_centers[:, 2], face_normals[:, 0], face_normals[:, 1],
              face_normals[:, 2])

    fig = plt.figure('Bounces')
    ax = fig.add_subplot(projection='3d')
    ax.quiver(face_centers[:, 0], face_centers[:, 1], face_centers[:, 2], bounces[:, 0], bounces[:, 1],
              bounces[:, 2])
    # ax.quiver(*obs_pt, *obs_dir * 10, color='red')

    plt.figure('RangeProfile')
    plt.subplot(1, 2, 1)
    plt.imshow(db(pd))
    plt.axis('tight')
    plt.subplot(1, 2, 2)
    plt.imshow(db(pd_ch))
    plt.axis('tight')
    plt.show()