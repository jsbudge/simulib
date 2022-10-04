import contextlib
import numpy as np
from simulation_functions import getElevationMap, llh2enu, \
    enu2llh, getElevation, db, resampleGrid
import open3d as o3d
from SDRParsing import SDRParse
from scipy.spatial.transform import Rotation as rot
from scipy.spatial import Delaunay
from scipy.signal import medfilt2d
import pickle

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

'''
Environment
This is a class to represent the environment of a radar. 
'''


class Environment(object):
    _mesh = None
    _pcd = None
    _ref_coefs = None
    _scat_coefs = None
    _refscale = 1
    _scatscale = 1

    def __init__(self, pts=None, scattering=None, reflectivity=None):
        self.pts = pts
        self.scattering = scattering
        self.reflectivity = reflectivity

    def createGrid(self, triangles):
        if self.pts is None:
            return
        # Create the point cloud for the mesh basis
        scats = self.scattering if self.scattering is not None else np.ones((triangles.shape[0],))
        refs = self.reflectivity if self.reflectivity is not None else np.ones((triangles.shape[0],))
        col_scale = (refs - refs.min()) / refs.max()
        col_scale /= col_scale.max()
        colors = np.zeros((len(refs), 3))
        colors[:, 0] = col_scale
        colors[:, 1] = col_scale
        colors[:, 2] = col_scale

        self._ref = refs
        self._scat = scats

        # Downsample if possible to reduce number of triangles
        if triangles is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=np.mean(pcd.compute_nearest_neighbor_distance()))

            avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
            radius = 3 * avg_dist
            radii = [radius, radius * 2]
            pcd.estimate_normals()
            with contextlib.suppress(RuntimeError):
                pcd.orient_normals_consistent_tangent_plane(100)
            # Generate mesh
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))

            rec_mesh.remove_duplicated_vertices()
            rec_mesh.remove_duplicated_triangles()
            rec_mesh.remove_degenerate_triangles()
            rec_mesh.remove_unreferenced_vertices()
        else:
            rec_mesh = o3d.geometry.TriangleMesh()
            rec_mesh.vertices = o3d.utility.Vector3dVector(self.pts)
            rec_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            rec_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            rec_mesh.compute_vertex_normals()
            pcd = o3d.geometry.PointCloud()
            pcd.points = rec_mesh.vertices
            pcd.colors = rec_mesh.vertex_colors
            pcd.normals = rec_mesh.vertex_normals
        self._mesh = rec_mesh
        self._pcd = pcd

    def setScatteringCoeffs(self, coef):
        if coef.shape[0] != self.triangles.shape[0]:
            raise RuntimeError('Scattering coefficients must be the same size as triangle points')
        self._scat = coef

    def setReflectivityCoeffs(self, coef):
        if coef.shape[0] != self.triangles.shape[0]:
            raise RuntimeError('Reflectivity coefficients must be the same size as triangle points')
        self._ref = coef

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
        return self._ref

    @property
    def scat_coefs(self):
        return self._scat


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
        self.createGrid()


class SDREnvironment(Environment):
    rps = 1
    cps = 1
    heading = 0.

    def __init__(self, sdr_file, local_grid=None, origin=None):
        # Load in the SDR file
        sdr = SDRParse(sdr_file) if type(sdr_file) == str else sdr_file
        grid = None
        print('SDR loaded')
        try:
            asi = sdr.loadASI(sdr.files['asi'])
        except KeyError:
            print('ASI not found.')
            asi = np.random.rand(2000, 2000)
            asi[250, 250] = 10
            asi[750, 750] = 10
            grid = asi
        except TypeError:
            asi = sdr.loadASI(sdr.files['asi'][list(sdr.files['asi'].keys())[0]])
            grid = db(asi)
        self._sdr = sdr
        self._asi = asi
        self.heading = -np.arctan2(sdr.gps_data['ve'].values[0], sdr.gps_data['vn'].values[0])
        if sdr.ash is None:
            hght = sdr.xml['Flight_Line']['Flight_Line_Altitude_M']
            pt = ((sdr.xml['Flight_Line']['Start_Latitude_D'] + sdr.xml['Flight_Line']['Stop_Latitude_D']) / 2,
                  (sdr.xml['Flight_Line']['Start_Longitude_D'] + sdr.xml['Flight_Line']['Stop_Longitude_D']) / 2)
            alt = getElevation(pt)
            mrange = hght / np.tan(sdr.ant[0].dep_ang)
            if origin is None:
                ref_llh = origin = enu2llh(mrange * np.sin(self.heading), mrange * np.cos(self.heading), 0.,
                                           (pt[0], pt[1], alt))
            else:
                ref_llh = enu2llh(mrange * np.sin(self.heading), mrange * np.cos(self.heading), 0.,
                                           (pt[0], pt[1], alt))
        else:
            if origin is None:
                origin = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                          getElevation((sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'])))
            ref_llh = (sdr.ash['geo']['refLat'], sdr.ash['geo']['refLon'],
                       sdr.ash['geo']['hRef'])
            self.rps = sdr.ash['geo']['rowPixelSizeM']
            self.cps = sdr.ash['geo']['colPixelSizeM']
            self.heading = -sdr.ash['flight']['flnHdg'] * DTR

        self.origin = origin
        self.ref = ref_llh

        if local_grid is not None:
            grid = local_grid
        else:
            # Set grid to be one meter resolution
            rowup = int(1 / self.rps) if self.rps < 1 else 1
            colup = int(1 / self.cps) if self.cps < 1 else 1
            grid = grid[::rowup, ::colup]
            self.rps *= rowup
            self.cps *= colup

            # Reduce grid to int8
            grid = medfilt2d(grid, 5)
            mu = grid[grid > -200].mean()
            std = grid[grid > -200].std()
            grid = np.digitize(grid, np.linspace(mu - std * 3, mu + std * 3, 1000))
        self._grid = grid
        self._grid_info = []

        super().__init__()

    def genMesh(self, num_vertices, tri_err=35):
        shift_x, shift_y, shift_z = llh2enu(*self.origin, self.ref)
        # Generate 2d triangle mesh
        if num_vertices is None:
            num_vertices = (self._grid.shape[0] * self._grid.shape[1]) // 2
        ptx, pty, reflectivity, simplices = mesh(self._grid, tri_err, num_vertices)

        # Rotate everything into lat/lon/alt for 3d mesh
        ptx = ptx - self._grid.shape[0] / 2
        pty = pty - self._grid.shape[1] / 2
        ptx *= self.rps
        pty *= self.cps
        rotated = rot.from_euler('z', self.heading).apply(
            np.array([ptx, pty, np.zeros_like(ptx)]).T)
        lat, lon, alt = enu2llh(rotated[:, 0] + shift_x, rotated[:, 1] + shift_y,
                                np.zeros_like(ptx) + shift_z, self.ref)
        e, n, u = llh2enu(lat, lon, getElevationMap(lat, lon), self.ref)
        self.pts = np.array([e, n, u]).T
        self.scattering = np.ones_like(e)
        self.reflectivity = reflectivity
        self.createGrid(simplices)


def mesh(grid, tri_err, num_vertices):
    # Generate a mesh using SVS metrics to make triangles in the right spots
    # Generate grid using indices instead of whatever unit it's in
    ptx, pty = np.meshgrid(np.arange(grid.shape[0] - 1), np.arange(grid.shape[1] - 1))
    ptx = ptx.flatten()
    pty = pty.flatten()
    im = grid[ptx, pty]
    nonzeros = im > 0
    ptx = ptx[nonzeros]
    pty = pty[nonzeros]
    im = im[nonzeros]
    pts = np.array([ptx, pty]).T

    # Initial points are the four corners of the grid
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
        sort_args = pts_tri.argsort()
        sorted_arr = pts_tri[sort_args]
        _, cut = np.unique(sorted_arr, return_index=True)
        out = np.split(sort_args, cut)
        total_err = 0
        add_pts = []
        for simplex in out:
            # Calculate out SVS error of triangle
            if len(simplex) > 1:
                tric = im[simplex].mean()
                errors = abs(im[simplex] - tric) ** 2
                if np.mean(errors) > tri_err:
                    winner = simplex[errors == errors.max()][0]
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
    reflectivity = \
        (grid[tri.points[tri.simplices[:, 0], 0].astype(int), tri.points[tri.simplices[:, 0], 1].astype(int)] +
         grid[tri.points[tri.simplices[:, 1], 0].astype(int), tri.points[tri.simplices[:, 1], 1].astype(int)] +
         grid[tri.points[tri.simplices[:, 2], 0].astype(int), tri.points[tri.simplices[:, 2], 1].astype(int)]) / 3
    return ptx, pty, reflectivity, tri.simplices
