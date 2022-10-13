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
    _grid = None
    _refgrid = None

    def __init__(self, grid=None, reflectivity=None):
        self._grid = grid
        self._refgrid = reflectivity

    def getDistance(self, pos):
        return np.linalg.norm(self._grid - pos[None, :], axis=1)

    def save(self, fnme):
        with open(fnme, 'wb') as f:
            pickle.dump(self, f)

    def getPos(self, px, py):
        return self._grid[:, px, py]

    @property
    def refgrid(self):
        return self._refgrid

    @property
    def grid(self):
        return self._grid


class MapEnvironment(Environment):

    def __init__(self, origin, extent, npts_background=500):
        lats = np.linspace(origin[0] - extent[0] / 2 / 111111, origin[0] + extent[0] / 2 / 111111, npts_background)
        lons = np.linspace(origin[1] - extent[1] / 2 / 111111, origin[1] + extent[1] / 2 / 111111, npts_background)
        lt, ln = np.meshgrid(lats, lons)
        ltp = lt.flatten()
        lnp = ln.flatten()
        e, n, u = llh2enu(ltp, lnp, getElevationMap(ltp, lnp), origin)
        self.origin = origin
        super().__init__(grid=np.array([e.reshape(grid.shape), n.reshape(grid.shape), u.reshape(grid.shape)]).T)


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

        shift_x, shift_y, shift_z = llh2enu(*self.origin, self.ref)
        ptx, pty = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
        ptx = ptx.flatten()
        pty = pty.flatten()
        ptx = ptx - grid.shape[0] / 2
        pty = pty - grid.shape[1] / 2
        ptx *= self.rps
        pty *= self.cps
        rotated = rot.from_euler('z', self.heading).apply(
            np.array([ptx, pty, np.zeros_like(ptx)]).T)
        lat, lon, alt = enu2llh(rotated[:, 0] + shift_x, rotated[:, 1] + shift_y,
                                np.zeros_like(ptx) + shift_z, self.ref)
        e, n, u = llh2enu(lat, lon, getElevationMap(lat, lon), self.ref)

        super().__init__(grid=np.array([e.reshape(grid.shape), n.reshape(grid.shape), u.reshape(grid.shape)]),
                         reflectivity=grid)

    def setGrid(self, newgrid, new_elgrid, newrps, newcps):
        self._refgrid = newgrid
        self._grid = new_elgrid
        self.rps = newrps
        self.cps = newcps


def mesh(grid, tri_err, num_vertices):
    # Generate a mesh using SVS metrics to make triangles in the right spots
    ptx = grid[0, :, :].flatten()
    pty = grid[1, :, :].flatten()
    im = grid[2, :, :].flatten()
    pts = np.array([ptx, pty]).T

    # Initial points are the four corners of the grid
    init_pts = np.array([[ptx.min(), pty.min()],
                         [ptx.min(), pty.max()],
                         [ptx.max(), pty.min()],
                         [ptx.max(), pty.max()]])
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
        if not add_pts:
            break
        try:
            tri.add_points(np.array(add_pts))
        except Exception:
            print('Something went wrong.')
            break
        its += 1
    ptx = tri.points[:, 0]
    pty = tri.points[:, 1]
    reflectivity = tri.find_simplex(pts).reshape((grid.shape[1], grid.shape[2]))
    return ptx, pty, reflectivity, tri.simplices
