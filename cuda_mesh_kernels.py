import cmath
import math

import cupy
from numba import cuda, njit
from numba.cuda.random import xoroshiro128p_uniform_float64

from simulation_functions import findPowerOf2
import numpy as np
import open3d as o3d
from cuda_kernels import applyRadiationPattern, applyRadiationPatternCPU, getMaxThreads

c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


@cuda.jit(device=True)
def getRangeAndAngles(vx, vy, vz, sx, sy, sz):
    tx = vx - sx
    ty = vy - sy
    tz = vz - sz
    rng = math.sqrt(abs(tx * tx) + abs(ty * ty) + abs(tz * tz))
    az = math.atan2(tx, ty)
    el = -math.asin(tz / rng)
    return tx, ty, tz, rng, az, el


@cuda.jit()
def calcIntersection(ray_power, ray_distance, ray_bounce, ray_xyz, vert_xyz, tri_norm, tri_verts, source_xyz, panrx, elrx,
                     pd_r, pd_i, wavenumber, near_range_s, source_fs, bw_az, bw_el):
    ray_idx, tri_idx = cuda.grid(ndim=2)
    if tri_norm.shape[0] > tri_idx and ray_idx < ray_power.shape[0]:
        rx = ray_xyz[ray_idx, 0]
        ry = ray_xyz[ray_idx, 1]
        rz = ray_xyz[ray_idx, 2]
        e1x = vert_xyz[tri_verts[tri_idx, 1], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
        e1y = vert_xyz[tri_verts[tri_idx, 1], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
        e1z = vert_xyz[tri_verts[tri_idx, 1], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
        e2x = vert_xyz[tri_verts[tri_idx, 2], 0] - vert_xyz[tri_verts[tri_idx, 0], 0]
        e2y = vert_xyz[tri_verts[tri_idx, 2], 1] - vert_xyz[tri_verts[tri_idx, 0], 1]
        e2z = vert_xyz[tri_verts[tri_idx, 2], 2] - vert_xyz[tri_verts[tri_idx, 0], 2]
        crossx = ray_bounce[ray_idx, 1] * e2z - ray_bounce[ray_idx, 2] * e2y
        crossy = ray_bounce[ray_idx, 2] * e2x - ray_bounce[ray_idx, 0] * e2z
        crossz = ray_bounce[ray_idx, 0] * e2y - ray_bounce[ray_idx, 1] * e2x
        det = e1x * crossx + e1y * crossy + e1z * crossz
        # Check to see if ray is parallel to triangle
        if abs(det) < 1e-9:
            return

        inv_det = 1. / det
        sx = rx - vert_xyz[tri_verts[tri_idx, 0], 0]
        sy = ry - vert_xyz[tri_verts[tri_idx, 0], 1]
        sz = rz - vert_xyz[tri_verts[tri_idx, 0], 2]

        u = inv_det * (sx * crossx + sy * crossy + sz * crossz)
        if u < 0 or u > 1:
            return

        # Recompute cross for s and edge 1
        crossx = sy * e1z - sz * e1y
        crossy = sz * e1x - sx * e1z
        crossz = sx * e1y - sy * e1x
        v = inv_det * (ray_bounce[ray_idx, 0] * crossx + ray_bounce[ray_idx, 1] * crossy + ray_bounce[ray_idx, 2] * crossz)
        if v < 0 or u + v > 1:
            return

        # Compute intersection point
        t = inv_det * (e2x * crossx + e2y * crossy + e2z * crossz)
        if t < 1e-9:
            return

        vnx = tri_norm[tri_idx, 0]
        vny = tri_norm[tri_idx, 1]
        vnz = tri_norm[tri_idx, 2]

        intx = rx + t * ray_bounce[ray_idx, 0]
        inty = ry + t * ray_bounce[ray_idx, 1]
        intz = rz + t * ray_bounce[ray_idx, 2]

        # Calculate out the angles in azimuth and elevation for the bounce
        tx, ty, tz, vrng, _, _ = getRangeAndAngles(intx, inty, intz, rx, ry, rz)
        sx, sy, sz, srng, r_az, r_el = getRangeAndAngles(intx, inty, intz, source_xyz[0], source_xyz[1], source_xyz[2])

        bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
        bx = tx - vnx * bounce_dot
        by = ty - vny * bounce_dot
        bz = tz - vnz * bounce_dot
        bounce_len = math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz))
        ray_bounce[ray_idx, 0] = bx / bounce_len
        ray_bounce[ray_idx, 1] = by / bounce_len
        ray_bounce[ray_idx, 2] = bz / bounce_len
        ray_xyz[ray_idx, 0] = intx
        ray_xyz[ray_idx, 1] = inty
        ray_xyz[ray_idx, 2] = intz

        ray_distance[ray_idx] += vrng

        # Calculate for return to source
        rx_strength = (sx * bx + sy * by + sz * bz) / (srng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
        rx_strength = max(rx_strength, 0.)
        accumulateRangeProfile(ray_power[ray_idx] * math.pow(rx_strength, 5),
                               ray_distance[ray_idx] + srng, r_el, r_az, panrx,
                               elrx, pd_r, pd_i, wavenumber, near_range_s, source_fs, bw_az, bw_el)

        # Calculate cosine similarity between bounce vector and source vector
        rx_strength = (intx * bx + inty * by + intz * bz) / (srng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
        rx_strength = max(rx_strength, 0.)
        ray_power[ray_idx] = ray_power[ray_idx] * math.pow(rx_strength, 5)

        cuda.syncthreads()


@cuda.jit()
def calcInitSpread(ray_power, ray_distance, ray_bounce, vert_xyz, vert_norm, vert_power, source_xyz, panrx, elrx, pd_r, pd_i,
                   wavenumber, near_range_s, source_fs, bw_az, bw_el):
    ray_idx, vert_idx = cuda.grid(ndim=2)
    if ray_idx < ray_power.shape[0] and vert_idx == 0:
        # Calculate the bounce vector for this time
        vx = vert_xyz[ray_idx, 0]
        vy = vert_xyz[ray_idx, 1]
        vz = vert_xyz[ray_idx, 2]
        vnx = vert_norm[ray_idx, 0]
        vny = vert_norm[ray_idx, 1]
        vnz = vert_norm[ray_idx, 2]

        # Calculate out the angles in azimuth and elevation for the bounce
        sx, sy, sz, srng, r_az, r_el = getRangeAndAngles(vx, vy, vz, source_xyz[0], source_xyz[1], source_xyz[2])

        bounce_dot = (sx * vnx + sy * vny + sz * vnz) * 2.
        bx = sx - vnx * bounce_dot
        by = sy - vny * bounce_dot
        bz = sz - vnz * bounce_dot
        bounce_len = math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz))
        ray_bounce[ray_idx, 0] = bx / bounce_len
        ray_bounce[ray_idx, 1] = by / bounce_len
        ray_bounce[ray_idx, 2] = bz / bounce_len

        rx_strength = (sx * bx + sy * by + sz * bz) / (srng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
        rx_strength = max(rx_strength, 0.)
        ray_power[ray_idx] = ray_power[ray_idx] * math.pow(rx_strength, 5)
        ray_distance[ray_idx] = srng
        accumulateRangeProfile(ray_power[ray_idx] / (ray_distance[ray_idx] * ray_distance[ray_idx]) * vert_power[ray_idx], srng, r_el, r_az, panrx,
                               elrx, pd_r, pd_i, wavenumber, near_range_s, source_fs, bw_az, bw_el)
        cuda.syncthreads()


@cuda.jit(device=True)
def accumulateRangeProfile(power, two_way_rng, r_el, r_az, panrx, elrx, pd_r, pd_i, wavenumber, near_range_s,
                            source_fs, bw_az, bw_el):
    rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
    but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
    if but > pd_r.shape[0] or but < 0:
        return
    att = applyRadiationPattern(r_el, r_az, panrx, elrx, panrx, elrx, bw_az, bw_el) / (
            two_way_rng * two_way_rng)
    acc_val = att * cmath.exp(-1j * wavenumber * two_way_rng) * power
    cuda.atomic.add(pd_r, but, acc_val.real)
    cuda.atomic.add(pd_i, but, acc_val.imag)



@cuda.jit()
def genRangeProfileFromMesh(vert_xyz, vert_norm, vert_reflectivity,
                            source_xyz, receive_xyz, panrx, elrx, pantx, eltx, pd_r, pd_i, wavelength, near_range_s,
                            source_fs, bw_az, bw_el, power_scale, do_beampattern, do_bounce):
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
        tx, ty, tz, rng, _, _ = getRangeAndAngles(vx, vy, vz, source_xyz[t, 0], source_xyz[t, 1], source_xyz[t, 2])
        rx, ry, rz, r_rng, r_az, r_el = getRangeAndAngles(vx, vy, vz, receive_xyz[t, 0], receive_xyz[t, 1], receive_xyz[t, 2])

        # Calculate bounce vector and strength
        gamma = 1.
        if do_bounce:
            bounce_dot = (tx * vnx + ty * vny + tz * vnz) * 2.
            bx = tx - vnx * bounce_dot
            by = ty - vny * bounce_dot
            bz = tz - vnz * bounce_dot

            rx_strength = (rx * bx + ry * by + rz * bz) / (r_rng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
            if rx_strength < 0:
                return
            gamma = math.pow(-rx_strength, 5)

        two_way_rng = rng + r_rng
        rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
        but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
        if but > pd_r.shape[0] or but < 0:
            return

        if n_samples > but > 0:
            att = 1.
            if do_beampattern:
                att = applyRadiationPattern(r_el, r_az, panrx[t], elrx[t], pantx[t], eltx[t], bw_az, bw_el) / (
                        two_way_rng * two_way_rng)
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

            rx_strength = (rx * bx + ry * by + rz * bz) / (
                    r_rng * math.sqrt(abs(bx * bx) + abs(by * by) + abs(bz * bz)))
            if rx_strength < 0:
                continue

            two_way_rng = rng + r_rng
            rng_bin = (two_way_rng / c0 - 2 * near_range_s) * source_fs
            but = int(rng_bin)  # if rng_bin - int(rng_bin) < .5 else int(rng_bin) + 1
            if but > pd_r.shape[0] or but < 0:
                continue

            if n_samples > but > 0:
                gamma = math.pow(-rx_strength, 5)
                # att = applyRadiationPatternCPU(r_el, r_az, panrx[t], elrx[t], pantx[t], eltx[t], bw_az, bw_el) / (
                #         two_way_rng * two_way_rng)
                att = 1.
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
    scalings = [30., .25, 41.08, 1.8, 60, 1., 1., .8, 12., 1., 156.25
                ]
    import matplotlib.pyplot as plt
    from simulation_functions import db, genChirp, azelToVec

    mesh = readCombineMeshFile('/home/jeff/Documents/target_meshes/helic.obj')
    mesh.scale(1 / 3., center=mesh.get_center())
    # mesh = sum([t.mesh for t in full_mesh.meshes])
    obs_pt = np.array([100., 0., 50.])
    nsam = 256
    nr = 4096
    fc = 32.0e9
    standoff = 700.
    fft_len = findPowerOf2(nsam + nr)
    up_fft_len = fft_len * 4

    print('Generating Mesh...')
    mverts = np.asarray(mesh.vertices)
    mnorms = np.asarray(mesh.vertex_normals)

    print('Sampling Points...')
    sample_points = mesh.sample_points_poisson_disk(10000)

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
    pd_r = cupy.zeros((nsam, len(pan)), dtype=np.float32)
    pd_i = cupy.zeros((nsam, len(pan)), dtype=np.float32)
    near_range_s = (standoff - 10) / c0

    print('Generating range profile...')
    face_centers_gpu = cupy.array(face_centers, dtype=np.float32)
    face_normals_gpu = cupy.array(face_normals, dtype=np.float32)
    reflectivity_gpu = cupy.array(np.ones(face_centers.shape[0]) * 1e9, dtype=np.float32)
    poses_gpu = cupy.array(poses, dtype=np.float32)
    pan_gpu = cupy.array(pan, dtype=np.float32)
    tilt_gpu = cupy.array(pan, dtype=np.float32)

    # GPU device calculations
    threads_per_block = getMaxThreads()
    bpg_bpj = (max(1, face_centers.shape[0] // threads_per_block[0] + 1), len(pan) // threads_per_block[1] + 1)
    genRangeProfileFromMesh[bpg_bpj, threads_per_block](face_centers_gpu, face_normals_gpu, reflectivity_gpu, poses_gpu,
                                                        poses_gpu,
                                                        pan_gpu, tilt_gpu, pan_gpu, tilt_gpu, pd_r, pd_i, c0 / fc,
                                                        near_range_s, fs, 10 * DTR,
                                                        10 * DTR, 1e9, False, True)

    pd = pd_r.get() + 1j * pd_i.get()
    pd = pd / abs(pd).max()

    '''pd_r_cpu = np.zeros((nsam, len(pan)), dtype=np.float32)
    pd_i_cpu = np.zeros((nsam, len(pan)), dtype=np.float32)

    pd_r_cpu, pd_i_cpu = genRangeProfileFromMeshCPU(face_centers.astype(np.float32), face_normals.astype(np.float32),
                                                    reflectivity_gpu.get(), poses.astype(np.float32),
                                                    poses.astype(np.float32),
                                                    pan.astype(np.float32), tilt.astype(np.float32),
                                                    pan.astype(np.float32), tilt.astype(np.float32), pd_r_cpu, pd_i_cpu,
                                                    c0 / fc,
                                                    near_range_s, fs, 10 * DTR,
                                                    10 * DTR, 1e9)
    pd_cpu = pd_r_cpu + 1j * pd_i_cpu
    pd_cpu = pd_cpu / abs(pd_cpu).max()'''
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
    plt.subplot(2, 2, 1)
    plt.imshow(db(pd))
    plt.axis('tight')
    plt.subplot(2, 2, 2)
    plt.imshow(db(pd_ch))
    plt.axis('tight')
    '''plt.subplot(2, 2, 3)
    plt.imshow(db(pd_cpu))
    plt.axis('tight')
    plt.subplot(2, 2, 4)
    plt.imshow(db(pd_cpu - pd))
    plt.axis('tight')'''
    plt.show()
