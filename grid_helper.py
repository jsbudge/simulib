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

    def __init__(self, pts=None, scattering=None, reflectivity=None, triangles=None, scatscale=1, refscale=1):

        if pts is not None:
            # Create the point cloud for the mesh basis
            scats = scattering if scattering is not None else np.ones((pts.shape[0],))
            refs = reflectivity if reflectivity is not None else np.ones((pts.shape[0],))
            colors = np.zeros((len(reflectivity), 3))
            colors[:, 0] = refs
            colors[:, 1] = scats
            colors[:, 2] = refs

            self._refscale = refscale
            self._scatscale = scatscale

            # Downsample if possible to reduce number of triangles
            if triangles is None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd = pcd.voxel_down_sample(voxel_size=np.mean(pcd.compute_nearest_neighbor_distance()))

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

                rec_mesh.remove_duplicated_vertices()
                rec_mesh.remove_duplicated_triangles()
                rec_mesh.remove_degenerate_triangles()
                rec_mesh.remove_unreferenced_vertices()
            else:
                rec_mesh = o3d.geometry.TriangleMesh()
                rec_mesh.vertices = o3d.utility.Vector3dVector(pts)
                rec_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
                rec_mesh.triangles = o3d.utility.Vector3iVector(triangles)
                rec_mesh.compute_vertex_normals()
                pcd = o3d.geometry.PointCloud()
                pcd.points = rec_mesh.vertices
                pcd.colors = rec_mesh.vertex_colors
                pcd.normals = rec_mesh.vertex_normals
            self._mesh = rec_mesh
            self._pcd = pcd
        else:
            pass

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

    def getDistance(self, pos):
        return np.linalg.norm(self.vertices - pos[None, :], axis=1)

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
    rps = 1
    cps = 1
    heading = 0.

    def __init__(self, sdr_file, num_vertices=100000, tri_err=35):
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
        self.heading = -np.arctan2(sdr.gps_data['ve'].values[0], sdr.gps_data['vn'].values[0])
        if sdr.ash is None:
            hght = sdr.xml['Flight_Line']['Flight_Line_Altitude_M']
            pt = ((sdr.xml['Flight_Line']['Start_Latitude_D'] + sdr.xml['Flight_Line']['Stop_Latitude_D']) / 2,
                  (sdr.xml['Flight_Line']['Start_Longitude_D'] + sdr.xml['Flight_Line']['Stop_Longitude_D']) / 2)
            alt = getElevation(pt)
            mrange = hght / np.tan(sdr.ant[0].dep_ang)
            ref_llh = enu2llh(mrange * np.sin(heading), mrange * np.cos(heading), 0.,
                              (pt[0], pt[1], alt))
        else:
            ref_llh = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                       getElevation((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'])))
            self.rps = sdr.ash['geo']['rowPixelSizeM']
            self.cps = sdr.ash['geo']['colPixelSizeM']
            self.heading = -sdr.ash['flight']['flnHdg'] * DTR

        self.origin = ref_llh
        grid = db(asi)

        # Set grid to be one meter resolution
        rowup = int(1 / self.rps) if self.rps < 1 else 1
        colup = int(1 / self.cps) if self.cps < 1 else 1
        grid = grid[::rowup, ::colup]
        self.rps *= rowup
        self.cps *= colup

        # Reduce grid to int8
        # mu = grid[grid != -300].mean()
        # std = grid[grid != -300].std()
        # grid = np.digitize(grid, np.linspace(mu - std * 3, mu + std * 3, 255))

        # Generate 2d triangle mesh
        ptx, pty, reflectivity, simplices = mesh(grid, tri_err, num_vertices)

        # Rotate everything into lat/lon/alt for 3d mesh
        ptx = ptx - grid.shape[0] / 2
        pty = pty - grid.shape[1] / 2
        ptx *= self.rps
        pty *= self.cps
        rotated = rot.from_euler('z', self.heading).apply(
            np.array([ptx, pty, np.zeros_like(ptx)]).T)
        lat, lon, alt = enu2llh(rotated[:, 0], rotated[:, 1], np.zeros_like(ptx), ref_llh)
        e, n, u = llh2enu(lat, lon, getElevationMap(lat, lon), ref_llh)
        self._grid = grid
        self._grid_info = []
        ref_max = reflectivity.max()

        super().__init__(np.array([e, n, u]).T, scattering=np.ones_like(e), reflectivity=reflectivity / ref_max,
                         triangles=simplices, refscale=ref_max)


def mesh(grid, tri_err, num_vertices):
    # Generate a mesh using SVS metrics to make triangles in the right spots
    ptx, pty = np.meshgrid(np.arange(grid.shape[0] - 1), np.arange(grid.shape[1] - 1))
    ptx = ptx.flatten()
    pty = pty.flatten()
    im = grid[ptx, pty]
    nonzeros = im != -300
    ptx = ptx[nonzeros]
    pty = pty[nonzeros]
    im = im[nonzeros]
    pts = np.array([ptx, pty]).T
    init_pts = np.array([[0, 0],
                         [0, grid.shape[1] - 1],
                         [grid.shape[0] - 1, 0],
                         [grid.shape[0] - 1, grid.shape[1] - 1]])
    tri = Delaunay(init_pts, incremental=True, qhull_options='QJ')
    total_err = np.inf
    its = 0
    total_its = 20
    while total_err > tri_err and its < total_its:
        pts_tri = tri.find_simplex(pts).astype(int)
        total_err = 0
        add_pts = []
        for t in range(tri.simplices.shape[0]):
            # Calculate out SVS error of triangle
            pts_idx = np.where(pts_tri == t)[0]
            if len(pts_idx) > 0:
                tric = im[pts_idx].mean()
                errors = abs(im[pts_idx] - tric) ** 2
                if np.mean(errors) > tri_err:
                    winner = pts_idx[errors == errors.max()][0]
                    add_pts.append([ptx[winner], pty[winner]])
                    total_err += sum(errors)
                    if len(add_pts) + tri.points.shape[0] >= num_vertices:
                        its = total_its
                        break
        total_err /= len(pts_tri)
        tri.add_points(np.array(add_pts))
        its += 1
    ptx = tri.points[:, 0]
    pty = tri.points[:, 1]
    reflectivity = grid[ptx.astype(int), pty.astype(int)]
    return ptx, pty, reflectivity, tri.simplices
