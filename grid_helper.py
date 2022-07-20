import numpy as np
from simulation_functions import getMapLocation, createMeshFromPoints, getElevationMap, rotate, llh2enu, genPulse, \
    enu2llh, getElevation, detect_local_extrema, db
import open3d as o3d
from SDRParsing import SDRParse
from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Delaunay
import pickle


fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


class Environment(object):
    _mesh = None
    _pcd = None
    _ref_coefs = None
    _scat_coefs = None
    _refscale = 1
    _scatscale = 1

    def __init__(self, pts=None, scattering=None, reflectivity=None, scatscale=1, refscale=1):

        if pts is not None:
            # Create the point cloud for the mesh basis
            scats = scattering if scattering is not None else np.ones((pts.shape[0],))
            refs = reflectivity if reflectivity is not None else np.ones((pts.shape[0],))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            colors = np.zeros((len(reflectivity), 3))
            colors[:, 0] = 1 - reflectivity * 2
            colors[:, 2] = reflectivity * 2
            colors[colors < 0] = 0
            colors[colors > 1] = 1
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self._refscale = refscale
            self._scatscale = scatscale

            # Downsample if possible to reduce number of triangles
            pcd = pcd.voxel_down_sample(voxel_size=np.mean(pcd.compute_nearest_neighbor_distance()))
            '''its = 0
            while np.std(pcd.compute_nearest_neighbor_distance()) > 2. and its < 30:
                dists = pcd.compute_nearest_neighbor_distance()
                pcd = pcd.voxel_down_sample(voxel_size=np.mean(dists) / 1.5)
                its += 1'''

            avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
            radius = 3 * avg_dist
            radii = [radius, radius * 2]
            pcd.estimate_normals()
            try:
                pcd.orient_normals_consistent_tangent_plane(100)
            except RuntimeError:
                pass

            # Generate mesh
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            # rec_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            rec_mesh.remove_duplicated_vertices()
            rec_mesh.remove_duplicated_triangles()
            rec_mesh.remove_degenerate_triangles()
            rec_mesh.remove_unreferenced_vertices()
            self._mesh = rec_mesh
            self._pcd = pcd

    def setScatteringCoeffs(self, coef, scale=10):
        if coef.shape[0] != self.vertices.shape[0]:
            raise RuntimeError('Scattering coefficients must be the same size as vertex points')
        old_stuff = np.asarray(self._pcd.colors)
        old_stuff[:, 1] = coef
        self._pcd.colors = o3d.utility.Vector3dVector(old_stuff)
        self._scatscale = scale

    def setReflectivityCoeffs(self, coef, scale=10):
        if coef.shape[0] != self.vertices.shape[0]:
            raise RuntimeError('Reflectivity coefficients must be the same size as vertex points')
        old_stuff = np.asarray(self._pcd.colors)
        old_stuff[:, 0] = coef
        old_stuff[:, 2] = coef
        self._pcd.colors = o3d.utility.Vector3dVector(old_stuff)
        self._refscale = scale

    def visualize(self):
        o3d.visualization.draw_geometries([self._pcd, self._mesh])

    def save(self, fnme):
        with open(fnme, 'wb') as f:
            pickle.dump(self, f)

    @property
    def vertices(self):
        return np.asarray(self._pcd.points)

    @property
    def triangles(self):
        return np.asarray(self._mesh.triangles)

    @property
    def normals(self):
        return np.asarray(self._mesh.vertex_normals)

    @property
    def ref_coefs(self):
        return np.asarray(self._pcd.colors)[:, 0] * self._refscale

    @property
    def scat_coefs(self):
        return np.asarray(self._pcd.colors)[:, 1] * self._scatscale


class MapEnvironment(Environment):

    def __init__(self, origin, extent, npts_background=500, resample=False):
        lats = np.linspace(origin[0] - extent[0] / 2 / 111111, origin[0] + extent[0] / 2 / 111111, npts_background)
        lons = np.linspace(origin[1] - extent[1] / 2 / 111111, origin[1] + extent[1] / 2 / 111111, npts_background)
        lt, ln = np.meshgrid(lats, lons)
        ltp = lt.flatten()
        lnp = ln.flatten()
        e, n, u = llh2enu(ltp, lnp, getElevationMap(ltp, lnp), origin)
        if resample:
            nlat, nlon, nh = resampleGrid(u.reshape(lt.shape), lats, lons, int(len(u) * .8))
            e, n, u = llh2enu(nlat, nlon, nh + origin[2], origin)
        self.origin = origin
        super().__init__(np.array([e, n, u]).T)


class SDREnvironment(Environment):

    def __init__(self, sdr_file, num_vertices=400000):
        # Load in the SDR file
        sdr = SDRParse(sdr_file) if type(sdr_file) == str else sdr_file
        print('SDR loaded')
        try:
            asi = sdr.loadASI(sdr.files['asi'])
        except KeyError:
            print('ASI not found.')
            asi = np.zeros((1000, 1000)) + .001
            asi[250, 250] = 1
            asi[750, 750] = 1
        self._sdr = sdr
        self._asi = asi
        row_pixel_size = 1
        col_pixel_size = 1
        heading = -np.arctan2(sdr.gps_data['ve'].values[0], sdr.gps_data['vn'].values[0])
        if sdr.ash is None:
            hght = sdr.xml['Flight_Line']['Flight_Line_Altitude_M']
            pt = ((sdr.xml['Flight_Line']['Start_Latitude_D'] + sdr.xml['Flight_Line']['Stop_Latitude_D']) / 2,
                  (sdr.xml['Flight_Line']['Start_Longitude_D'] + sdr.xml['Flight_Line']['Stop_Longitude_D']) / 2)
            alt = getElevation(pt)
            mrange = hght / np.tan(sdr.ant[0].dep_ang)
            ref_llh = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                              (pt[0], pt[1], alt))
        else:
            ref_llh = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'], sdr.ash['geo']['hRef'])
            row_pixel_size = sdr.ash['geo']['rowPixelSizeM']
            col_pixel_size = sdr.ash['geo']['colPixelSizeM']
            heading = -sdr.ash['flight']['flnHdg'] * DTR

        self.origin = ref_llh

        super().__init__(np.array([e, n, u]).T, scattering=np.ones_like(e), reflectivity=asi_pts / asi_max,
                         refscale=asi_max)

    def generate(self):
        grid = db(asi)
        rowup = int(1 / row_pixel_size) if row_pixel_size < 1 else 1
        colup = int(1 / col_pixel_size) if col_pixel_size < 1 else 1
        grid = grid[::rowup, ::colup]
        # Reduce grid to int8
        # mu = grid[grid != -300].mean()
        # std = grid[grid != -300].std()
        # grid = np.digitize(grid, np.linspace(mu - std * 3, mu + std * 3, 255))
        row_pixel_size *= rowup
        col_pixel_size *= colup
        ptx, pty = np.meshgrid(np.arange(0, grid.shape[0]), np.arange(0, grid.shape[1]))
        ptx = ptx.flatten()
        pty = pty.flatten()
        asi_pts = grid[ptx, pty]
        ptx = ptx - grid.shape[0] / 2
        pty = pty - grid.shape[1] / 2
        ptx *= row_pixel_size
        pty *= col_pixel_size
        rotated = rot.from_euler('z', heading).apply(
            np.array([ptx, pty, np.ones_like(ptx)]).T)
        lat, lon, alt = enu2llh(rotated[:, 0], rotated[:, 1], np.zeros_like(ptx), ref_llh)
        e, n, u = llh2enu(lat, lon, getElevationMap(lat, lon), ref_llh)
        self._grid = grid
        self._grid_info = []
        # Get the point cloud information
        asi_max = asi_pts.max()