import numpy as np
from osgeo import gdal
from scipy.interpolate import RectBivariateSpline
from scipy.spatial.transform import Rotation as rot
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
import open3d as o3d
import plotly.io as pio
import plotly.graph_objects as go
import os

pio.renderers.default = 'browser'

WGS_A = 6378137.0
WGS_F = 1 / 298.257223563
WGS_B = 6356752.314245179
WGS_E2 = 6.69437999014e-3
c0 = 299792458.0
DTR = np.pi / 180


def getDTEDName(lat, lon):
    """Return the path and name of the dted to load for the given lat/lon"""
    tmplat = int(np.floor(lat))
    tmplon = int(np.floor(lon))
    direw = 'w' if tmplon < 0 else 'e'
    dirns = 's' if tmplat < 0 else 'n'
    if os.name == 'nt':
        return 'Z:\\dted\\%s%03d\\%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))
    else:
        return '/data5/dted/%s%03d/%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))


def detect_local_extrema(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = filters.minimum_filter(arr, footprint=neighborhood) == arr
    # local_max = filters.maximum_filter(arr, footprint=neighborhood) == arr
    background = arr == 0
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_extrema = local_min ^ eroded_background  # + local_max ^ eroded_background
    return np.where(detected_extrema)


def db(x):
    ret = abs(x)
    ret[ret < 1e-15] = 1e-15
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2 ** (np.ceil(np.log2(x))))


def undulationEGM96(lat, lon):
    if os.name == 'nt':
        egmdatfile = "Z:\\dted\\EGM96.DAT"
    else:
        egmdatfile = "/data5/dted/EGM96.DAT"
    with open(egmdatfile, "rb") as f:
        emg96 = np.fromfile(f, 'double', 1441 * 721, '')
        eg_n = np.ceil(lat / .25) * .25
        eg_s = np.floor(lat / .25) * .25
        eg_e = np.ceil(lon / .25) * .25
        eg_w = np.floor(lon / .25) * .25
        eg1 = emg96[((eg_w + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_n + 90 + .25) / .25 - 1).astype(int)]
        eg2 = emg96[((eg_w + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_s + 90 + .25) / .25 - 1).astype(int)]
        eg3 = emg96[((eg_e + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_n + 90 + .25) / .25 - 1).astype(int)]
        eg4 = emg96[((eg_e + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_s + 90 + .25) / .25 - 1).astype(int)]
        egc = (eg2 / ((eg_e - eg_w) * (eg_n - eg_s))) * (eg_e - lon) * (eg_n - lat) + \
              (eg4 / ((eg_e - eg_w) * (eg_n - eg_s))) * (lon - eg_w) * (eg_n - lat) + \
              (eg1 / ((eg_e - eg_w) * (eg_n - eg_s))) * (eg_e - lon) * (lat - eg_s) + \
              (eg3 / ((eg_e - eg_w) * (eg_n - eg_s))) * (lon - eg_w) * (lat - eg_s)
    return egc


def getElevationMap(lats, lons):
    """Returns the digital elevation for a latitude and longitude"""
    dtedName = getDTEDName(lats[0], lons[0])

    # open DTED file for reading
    ds = gdal.Open(dtedName)

    # grab geo transform with resolutions
    gt = ds.GetGeoTransform()
    bin_lat = (lats - gt[3]) / gt[-1]
    bin_lon = (lons - gt[0]) / gt[1]
    blmin = bin_lat.min()
    blmax = bin_lat.max()

    # read in raster data
    raster = ds.GetRasterBand(1).ReadAsArray()

    # Get lats and lons as bins into raster
    bin_lat = (lats - gt[3]) / gt[-1]
    bin_lon = (lons - gt[0]) / gt[1]

    # Linear interpolation using bins
    bx1 = bin_lat.astype(int)
    bx2 = (bin_lat + 1).astype(int)
    by1 = bin_lon.astype(int)
    by2 = (bin_lon + 1).astype(int)
    x1 = bx1 * gt[-1]
    x2 = bx2 * gt[-1]
    y1 = by1 * gt[1]
    y2 = by2 * gt[1]
    x = lats - gt[3]
    y = lons - gt[0]
    dted_mat = np.array([[raster[bx1, by1], raster[bx1, by2], raster[bx2, by1], raster[bx2, by2]]])

    coeff_mat = np.array([[x2 * y2, -y2, -x2, np.ones_like(x2)],
                          [-x2 * y1, y1, x2, -np.ones_like(x2)],
                          [-x1 * y2, y2, x1, -np.ones_like(x2)],
                          [x1 * y1, -y1, -x1, np.ones_like(x2)]])

    poly = np.einsum('ijn,jkn->ikn', dted_mat, coeff_mat)

    sol = np.einsum('ijn,jkn->ikn', poly, np.array([[np.ones_like(x2), x, y, x * y]]).swapaxes(0, 1))

    return 1 / ((x2 - x1) * (y2 - y1)) * sol.flatten() + undulationEGM96(lats, lons)


def getElevation(pt):
    lat = pt[0]
    lon = pt[1]
    """Returns the digital elevation for a latitude and longitude"""
    dtedName = getDTEDName(lat, lon)
    gdal.UseExceptions()
    # open DTED file for reading
    ds = gdal.Open(dtedName)

    # get the geo transform info for the dted
    # ulx is upper left corner longitude
    # xres is the resolution in the x-direction (in degrees/sample)
    # xskew is useless (0.0)
    # uly is the upper left corner latitude
    # yskew is useless (0.0)
    # yres is the resolution in the y-direction (in degrees/sample)
    ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
    # calculate the x and y indices into the DTED data for the lat/lon
    px = int(np.round((lon - ulx) / xres))
    py = int(np.round((lat - uly) / yres))

    # only if these x and y indices are within the bounds of the DTED, get the
    # raster band and try to read in the DTED values
    elevation = -1e20
    if (0 <= px < ds.RasterXSize) and (0 <= py < ds.RasterYSize):
        rasterBand = ds.GetRasterBand(1)
        dtedData = rasterBand.ReadAsArray(px, py, 2, 2)

        # use bilinear interpolation to get the elevation for the lat/lon
        x = (lon - ulx)
        y = (lat - uly)
        x1 = int((lon - ulx) / xres) * xres
        x2 = int((lon - ulx) / xres + 1) * xres
        y1 = int((lat - uly) / yres) * yres
        y2 = int((lat - uly) / yres + 1) * yres
        elevation = 1 / ((x2 - x1) * (y2 - y1)) * \
                    dtedData.ravel().dot(np.array([[x2 * y2, -y2, -x2, 1],
                                                    [-x2 * y1, y1, x2, -1],
                                                    [-x1 * y2, y2, x1, -1],
                                                    [x1 * y1, -y1, -x1, 1]])).dot(np.array([1, x, y, x * y]))

    return elevation + undulationEGM96(lat, lon)


def llh2enu(lat, lon, h, refllh):
    ecef = llh2ecef(lat, lon, h)
    return ecef2enu(*ecef, refllh)


def enu2llh(e, n, u, refllh):
    ecef = enu2ecef(e, n, u, refllh)
    return ecef2llh(*ecef)


def enu2ecef(e, n, u, refllh):
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    enu = np.array([e, n, u])
    tmp_rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                        [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                        [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]]).T
    if len(enu.shape) > 1:
        sz = np.ones((enu.shape[1],))
        ecef = tmp_rot.dot(enu) + np.array([sz * rx, sz * ry, sz * rz])
    else:
        ecef = tmp_rot.dot(enu) + np.array([rx, ry, rz])
    return ecef[0], ecef[1], ecef[2]


def llh2ecef(lat, lon, h):
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y, Z
    given the Geodetic Coordinates lat, lon + Ellipsoid Height h
    """
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180
    N = WGS_A / np.sqrt(1 - WGS_E2 * np.sin(lat_rad) ** 2)
    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (WGS_B ** 2 / WGS_A ** 2 * N + h) * np.sin(lat_rad)
    return X, Y, Z


def ecef2llh(x, y, z):
    # This is the Heikkinen application of the Ferrari solution to Bowring's irrational
    # geodetic-latitude equation to get a geodetic latitude and height.
    # Longitude remains the same between the two.
    r = np.sqrt(x ** 2 + y ** 2)
    ep2 = (WGS_A ** 2 - WGS_B ** 2) / WGS_B ** 2
    F = 54 * WGS_B ** 2 * z ** 2
    G = r ** 2 + (1 - WGS_E2) * z ** 2 - WGS_E2 * (WGS_A ** 2 - WGS_B ** 2)
    c = WGS_E2 ** 2 * F * r ** 2 / G ** 3
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G ** 2)
    Q = np.sqrt(1 + 2 * WGS_E2 ** 2 * P)
    r0 = -P * WGS_E2 * r / (1 + Q) + np.sqrt(
        1 / 2 * WGS_A ** 2 * (1 + 1 / Q) - P * (1 - WGS_E2) * z ** 2 / (Q * (1 + Q)) - 1 / 2 * P * r ** 2)
    U = np.sqrt((r - WGS_E2 * r0) ** 2 + z ** 2)
    V = np.sqrt((r - WGS_E2 * r0) ** 2 + (1 - WGS_E2) * z ** 2)
    z0 = WGS_B ** 2 * z / (WGS_A * V)
    h = U * (1 - WGS_B ** 2 / (WGS_A * V))
    lat = np.arctan((z + ep2 * z0) / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon, h


def ecef2enu(x, y, z, refllh):
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                    [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                    [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]])
    enu = rot.dot(np.array([x - rx, y - ry, z - rz]))
    return enu[0], enu[1], enu[2]


def getLivingRoom(voxel_downsample=.05):
    pcd = o3d.io.read_point_cloud("./livingroom.ply")

    # Stupid thing has walls, need to crop them out
    cube_ext = 3
    cube_hght = 2.5
    cube_hght_below = .1
    cube_points = np.array([
        # Vertices Polygon1
        [(cube_ext / 2), cube_hght, (cube_ext / 2)],  # face-topright
        [-(cube_ext / 2), cube_hght, (cube_ext / 2)],  # face-topleft
        [-(cube_ext / 2), cube_hght, -(cube_ext / 2)],  # rear-topleft
        [(cube_ext / 2), cube_hght, -(cube_ext / 2)],  # rear-topright
        # Vertices Polygon 2
        [(cube_ext / 2), cube_hght_below, (cube_ext / 2)],
        [-(cube_ext / 2), cube_hght_below, (cube_ext / 2)],
        [-(cube_ext / 2), cube_hght_below, -(cube_ext / 2)],
        [-(cube_ext / 2), cube_hght_below, -(cube_ext / 2)],
    ]).astype("float64")
    v3dv = o3d.utility.Vector3dVector(cube_points)
    oriented_bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(v3dv)
    pcd = pcd.crop(oriented_bounding_box)

    # This is not needed after the walls are removed
    pcd = pcd.voxel_down_sample(voxel_size=voxel_downsample)
    return pcd


def getMapLocation(p1, extent, init_llh, npts_background=500, resample=True):
    pt_enu = llh2enu(*p1, init_llh)
    lats = np.linspace(p1[0] - extent[0] / 2 / 111111, p1[0] + extent[0] / 2 / 111111, npts_background)
    lons = np.linspace(p1[1] - extent[1] / 2 / 111111, p1[1] + extent[1] / 2 / 111111, npts_background)
    lt, ln = np.meshgrid(lats, lons)
    ltp = lt.flatten()
    lnp = ln.flatten()
    e, n, u = llh2enu(ltp, lnp, getElevationMap(ltp, lnp), init_llh)
    if resample:
        nlat, nlon, nh = resampleGrid(u.reshape(lt.shape), lats, lons, int(len(u) * .8))
        e, n, u = llh2enu(nlat, nlon, nh + init_llh[2], init_llh)
    point_cloud = np.array([e, n, u]).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd = pcd.translate(pt_enu)
    return pcd


def resampleGrid(grid, x, y, npts):
    gxx, gyy = np.gradient(grid / grid.max())
    gx = RectBivariateSpline(np.arange(len(x)), np.arange(len(y)), gxx)
    gy = RectBivariateSpline(np.arange(len(x)), np.arange(len(y)), gyy)
    ptx = np.random.uniform(0, len(x) - 1, npts)
    pty = np.random.uniform(0, len(y) - 1, npts)
    for n in range(4):
        dx = gx(ptx, pty, grid=False) * 1e3 / (n + 1)
        dy = gy(ptx, pty, grid=False) * 1e3 / (n + 1)
        ptx += dx
        pty += dy
        ptx[ptx > len(x) - 1] = np.random.uniform(0, len(x) - 1, sum(ptx > len(x) - 1))
        pty[pty > len(y) - 1] = np.random.uniform(0, len(y) - 1, sum(pty > len(y) - 1))
    finalgrid = RectBivariateSpline(np.arange(len(x)), np.arange(len(y)), grid)
    return np.interp(ptx, np.arange(len(x)), x), np.interp(pty, np.arange(len(y)), y), finalgrid(ptx, pty, grid=False)


def createMeshFromPoints(pcd):
    pcd = pcd.voxel_down_sample(voxel_size=np.mean(pcd.compute_nearest_neighbor_distance()) / 1.5)
    its = 0
    while np.std(pcd.compute_nearest_neighbor_distance()) > 2. and its < 30:
        dists = pcd.compute_nearest_neighbor_distance()
        pcd = pcd.voxel_down_sample(voxel_size=np.mean(dists) / 1.5)
        its += 1

    avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
    radius = 3 * avg_dist
    radii = [radius, radius * 2]
    pcd.estimate_normals()
    try:
        pcd.orient_normals_consistent_tangent_plane(100)
    except RuntimeError:
        pass

    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    # rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    rec_mesh.remove_duplicated_vertices()
    rec_mesh.remove_duplicated_triangles()
    rec_mesh.remove_degenerate_triangles()
    rec_mesh.remove_unreferenced_vertices()
    return rec_mesh


def genPulse(phase_x, phase_y, nnr, nfs, nfc, bandw):
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * 1 / nfs))


def rotate(az, nel, rot_mat):
    return rot.from_euler('zx', [[-az, 0.],
                                 [0., nel - np.pi / 2]]).apply(rot_mat)


def azelToVec(az, el):
    return -np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def hornPattern(fc, width, height, theta=None, phi=None, deg_per_bin=.5, az_only=False):
    _lambda = c0 / fc
    d = _lambda / 2.
    if theta is None:
        theta = np.arange(0, np.pi, deg_per_bin * DTR)
    if phi is None:
        phi = [0] if az_only else np.arange(-np.pi / 2, np.pi / 2, deg_per_bin * DTR)
    theta, phi = np.meshgrid(theta, phi)
    lcw = np.arange(-width / 2, width / 2, d)
    lch = np.arange(-height / 2, height / 2, d)
    lch, lcw = np.meshgrid(lch, lcw)
    lchm = lch.flatten()
    lcwm = lcw.flatten()
    k = 2 * np.pi / _lambda
    locs = np.array([lcwm, np.zeros_like(lcwm), lchm]).T
    ublock = azelToVec(theta.flatten(), phi.flatten())
    AF = np.sum(np.exp(-1j * k * locs.dot(ublock)), axis=0)
    AF = AF.flatten() if az_only else AF.reshape(theta.shape)

    # Return degree array and antenna pattern
    return theta, phi, AF


def arrayFactor(fc, pos, theta=None, phi=None, weights=None, deg_per_bin=.5, az_only=False):
    _, _, el_pat = hornPattern(fc, .0766, .0646, theta=theta, phi=phi, deg_per_bin=deg_per_bin, az_only=az_only)
    _lambda = c0 / fc
    if theta is None:
        theta = np.arange(0, np.pi, deg_per_bin * DTR)
    if phi is None:
        phi = [0] if az_only else np.arange(-np.pi / 2, np.pi / 2, deg_per_bin * DTR)
    theta, phi = np.meshgrid(theta, phi)
    k = 2 * np.pi / _lambda
    # az, el = np.meshgrid(theta, theta)
    ublock = azelToVec(theta.flatten(), phi.flatten())
    AF = np.exp(-1j * k * pos.dot(ublock)) * el_pat[None, :]
    weights = weights if weights is not None else np.ones(pos.shape[0])
    AF = AF.T.dot(weights).flatten() if az_only else AF.T.dot(weights).reshape(theta.shape)
    # Return degree array and antenna pattern
    return theta, phi, AF


'''
------------------------------------------------------------------------------------
------------------- RENDERING AND PLOTTING FUNCTIONS -------------------------------
------------------------------------------------------------------------------------
'''


class PlotWithSliders(object):

    def __init__(self, bg=None, ntraces=1):
        self._frames = []
        if bg is None:
            self._fig = go.Figure(go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(color="red", size=10)))
            self._traces = [0]
        else:
            self._fig = go.Figure(data=[go.Scatter3d(x=bg[:, 0], y=bg[:, 1], z=bg[:, 2], mode="markers",
                                                     marker=dict(color="red", size=10))] +
                                       [go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                                     marker=dict(color="red", size=10)) for _ in range(ntraces)])
            self._traces = list(range(1, ntraces + 1))

    def addFrame(self, fdata, trace=None, args=None):
        t = [trace] if trace is not None else [self._traces[0]]
        frame_args = dict(data=[go.Scatter3d(x=fdata[:, 0], y=fdata[:, 1], z=fdata[:, 2], mode='lines+markers')],
                                     traces=t, name=f'frame {len(self._frames) + 1}')
        if args is not None:
            for key, val in args.items():
                frame_args[key] = val
        self._frames.append(go.Frame(**frame_args))

    def render(self):
        args = {"frame": {"duration": 0}, "mode": "immediate", "fromcurrent": True,
                "transition": {"duration": 0, "easing": "linear"}, }
        slider = [{"pad": {"b": 1, "t": 1}, "len": 0.9, "x": 0.1, "y": 0,
                   "steps": [{"args": [[f.name], args],
                              "label": str(k), "method": "animate", } for k, f in enumerate(self._fig.frames)]}]

        update_menus = [{"buttons": [{'args': [None, args], 'label': 'Play', 'method': 'animate'},
                            {'args': [[None], args], 'label': 'Pause', 'method': 'animate'}],
                         "direction": "left", "pad": {"r": 10, "t": 1}, "type": "buttons", "x": 0.1, "y": 0, }]
        self._fig.update(frames=self._frames)
        self._fig.update_layout(updatemenus=update_menus, sliders=slider)
        self._fig.update_layout(sliders=slider)
        self._fig.show()
