import numpy as np
from simulation_functions import getElevationMap, llh2enu, enu2llh, getElevation
from scipy.spatial import Delaunay
from scipy.interpolate import interpn
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
    _transforms: tuple
    _refgrid: np.ndarray
    ref: np.ndarray
    origin: np.ndarray

    def __init__(self, rmat=None, shift=None, reflectivity=None):
        if rmat is not None:
            self.setGrid(reflectivity, rmat, shift)

    def getGridParams(self, pos, width, height, npts, az=0.):
        shift_x, shift_y, _ = llh2enu(*pos, self.ref)
        corr_az = np.pi / 2 - az
        # Translation
        rmat = np.array([[1, 0, shift_x],
                         [0, 1, shift_y],
                         [0, 0, 1]])
        # Rotation
        rmat = rmat.dot(np.array([[np.cos(corr_az), -np.sin(corr_az), 0],
                                  [np.sin(corr_az), np.cos(corr_az), 0],
                                  [0, 0, 1.]]))
        # Scaling
        # The -1 offsets the fact that the number of points is one more than the array element index
        w_k = width / (npts[0] - 1)
        h_k = height / (npts[1] - 1)
        rmat = rmat.dot(np.diag([h_k, w_k, 1]))

        return rmat, np.array([shift_x, shift_y])

    def getGrid(self, pos=None, width=None, height=None, nrows=0, ncols=0, az=0, use_elevation=True):
        # This grid is independent of the refgrid or stored transforms
        pos = self.origin if pos is None else pos
        width = self.shape[0] if width is None else width
        height = self.shape[1] if height is None else height
        npts = self.shape if nrows == 0 else (ncols, nrows)
        rmat, (shift_x, shift_y) = self.getGridParams(pos, width, height, npts, az)
        gxx = np.linspace(npts[0] / 2, -npts[0] / 2, npts[0])
        gyy = np.linspace(-npts[1] / 2, npts[1] / 2, npts[1])
        gy, gx = np.meshgrid(gxx, gyy)

        px = rmat[0, 0] * gx + rmat[0, 1] * gy + rmat[0, 2]
        py = rmat[1, 0] * gx + rmat[1, 1] * gy + rmat[1, 2]
        latg, long, altg = enu2llh(px.ravel(), py.ravel(), np.zeros(px.shape[0] * px.shape[1]), self.ref)
        sh = gx.shape
        if use_elevation:
            try:
                gz = (getElevationMap(latg, long, interp_method='splinef2d') - self.ref[2]).reshape(sh)
            except FileNotFoundError:
                gz = np.zeros(px.shape)
        else:
            gz = np.zeros(px.shape)
        return px, py, gz

    def getRefGrid(self, pos=None, width=None, height=None, nrows=0, ncols=0, az=0):
        x, y, _ = self.getGrid(pos, width, height, nrows, ncols, az)
        irmat = np.linalg.pinv(self._transforms[0])
        px = self.shape[1] - (irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2)
        py = self.shape[0] - (irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2)
        pos_r = np.stack([px.ravel(), py.ravel()]).T
        return interpn((np.arange(self.refgrid.shape[1]),
                        np.arange(self.refgrid.shape[0])), self.refgrid.T, pos_r, bounds_error=False,
                       fill_value=0).reshape(x.shape, order='C')

    def setGrid(self, newgrid, rmat, shift):
        self._refgrid = newgrid
        self._transforms = (rmat, shift)

    def resampleGrid(self, pos, width, height, nrows, ncols, az=0):
        x, y, _ = self.getGrid(pos, width, height, nrows, ncols, az)
        irmat = np.linalg.pinv(self._transforms[0])
        px = irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2
        py = irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2
        pos_r = np.stack([px.ravel(), py.ravel()]).T
        self.setGrid(interpn((np.arange(self.refgrid.shape[1]),
                              np.arange(self.refgrid.shape[0])), self.refgrid.T, pos_r, bounds_error=False,
                             fill_value=0).reshape(x.shape, order='C'),
                     *self.getGridParams(pos, width, height, (nrows, ncols), az))

    def sample(self, x, y):
        irmat = np.linalg.pinv(self._transforms[0])
        px = irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2
        py = irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2
        pos_r = np.stack([px.ravel(), py.ravel()]).T
        return interpn((np.arange(self.refgrid.shape[1]),
                        np.arange(self.refgrid.shape[0])), self.refgrid.T, pos_r, bounds_error=False,
                       fill_value=0)

    def save(self, fnme):
        with open(fnme, 'wb') as f:
            pickle.dump(self, f)

    def getPos(self, px: float | list[float], py: float | list[float], elevation: bool = False) -> np.ndarray:
        """
        Calculate the grid position based on the given pixel coordinates.

        Args:
            px (int): The x-coordinate of the pixel element.
            py (int): The y-coordinate of the pixel element.
            elevation (bool, optional): Flag to include elevation data. Defaults to False.

        Returns:
            np.array: An array containing the calculated position coordinates.
            If elevation is True, the array also includes the elevation relative to a reference point.
        """
        gx = px - self.shape[1] / 2
        gy = py - self.shape[0] / 2
        pos_x = self._transforms[0][0, 0] * gx + self._transforms[0][0, 1] * gy + self._transforms[0][0, 2]
        pos_y = self._transforms[0][1, 0] * gx + self._transforms[0][1, 1] * gy + self._transforms[0][1, 2]
        if not elevation:
            return np.array([pos_x, pos_y]).T
        lat, lon, _ = (
            enu2llh(pos_x, pos_y, 0, self.ref)
            if isinstance(px, float)
            else enu2llh(pos_x, pos_y, np.zeros_like(pos_x), self.ref)
        )
        return np.array([pos_x, pos_y, getElevation(lat, lon) - self.ref[2]]) if isinstance(px, float) else (
            np.array([pos_x, pos_y, getElevationMap(lat, lon) - self.ref[2]]).T)

    def getIndex(self, x: float, y: float) -> np.ndarray:
        irmat = np.linalg.pinv(self._transforms[0])
        px = irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2
        py = irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2
        return np.array([px, py])

    def interp(self, x, y):
        return interpn((np.arange(self.refgrid.shape[0]),
                        np.arange(self.refgrid.shape[1])), self.refgrid, self.getIndex(x, y)).reshape(x.shape)

    @property
    def refgrid(self):
        return self._refgrid

    @property
    def shape(self):
        return self._refgrid.shape

    @property
    def transforms(self):
        return self._transforms


class MapEnvironment(Environment):

    def __init__(self, origin, extent, background):
        self.origin = origin
        self.ref = origin
        gp = getGridParams(origin, origin, extent[0], extent[1], background.shape)
        super().__init__(gp[0], gp[1], background)


class SDREnvironment(Environment):
    rps: float = 1
    cps: float = 1
    heading: float = 0.

    def __init__(self, sdr, local_grid=None, origin=None):
        print('SDR loaded')
        try:
            asi = sdr.loadASI(sdr.files['asi'])
            grid = abs(asi)
        except KeyError:
            print('ASI not found.')
            asi = np.random.rand(2000, 2000)
            asi[250, 250] = 10
            asi[750, 750] = 10
            grid = asi
        except TypeError:
            asi = sdr.loadASI(sdr.files['asi'][0])
            grid = abs(asi)
        self._sdr = sdr
        self._asi = asi
        self.heading = np.arctan2(sdr.gps_data['ve'].values[0], sdr.gps_data['vn'].values[0])
        if sdr.ash is None:
            try:
                hght = sdr.xml['Flight_Line']['Flight_Line_Altitude_M']
                pt = ((sdr.xml['Flight_Line']['Start_Latitude_D'] + sdr.xml['Flight_Line']['Stop_Latitude_D']) / 2,
                      (sdr.xml['Flight_Line']['Start_Longitude_D'] + sdr.xml['Flight_Line']['Stop_Longitude_D']) / 2)
                alt = getElevation(*pt)
            except KeyError:
                alt = sdr.gps_data['alt'].mean()
                pt = (sdr.gps_data['lat'].mean(), sdr.gps_data['lon'].mean())
                hght = alt + getElevation(*pt)
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
                          getElevation(sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX']))
            ref_llh = (sdr.ash['geo']['refLat'], sdr.ash['geo']['refLon'],
                       sdr.ash['geo']['hRef'])
            self.rps = sdr.ash['geo']['rowPixelSizeM']
            self.cps = sdr.ash['geo']['colPixelSizeM']
            self.heading = sdr.ash['flight']['flnHdg'] * DTR

        self.origin = origin
        self.ref = ref_llh

        grid = local_grid if local_grid is not None else grid

        rmat, shift = self.getGridParams(self.origin, grid.shape[0] * self.cps, grid.shape[1] * self.rps, grid.shape,
                                         self.heading)

        super().__init__(rmat=rmat, shift=shift, reflectivity=grid)

    @property
    def sdr(self):
        return self._sdr


def mesh(ptx, pty, ref_im, tri_err, max_vertices, max_iters=20, minimize_vertices=True):
    # Generate a mesh using SVS metrics to make triangles in the right spots

    # Initial points are the four corners of the grid
    init_pts = np.array([[ptx.min(), pty.min()],
                         [ptx.min(), pty.max()],
                         [ptx.max(), pty.min()],
                         [ptx.max(), pty.max()]])
    tri = Delaunay(init_pts, incremental=True, qhull_options='QJ')
    total_err = 0.
    for _ in range(max_iters):
        pts = np.array([np.random.rand(max_vertices) * ptx.max(), np.random.rand(max_vertices) * pty.max()]).T
        im = interpn([ptx, pty], ref_im, pts)
        pts_tri = tri.find_simplex(pts).astype(int)
        simp_idx = np.unique(pts_tri)
        add_pts = []
        for sidx in simp_idx:
            i = pts_tri == sidx
            # Calculate out SVS error of triangle
            tric = im[i].mean()
            errors = abs(im[i] - tric) ** 2
            if np.any(errors > tri_err):
                add_pts.append(pts[i][errors == errors.max()][0])
                total_err += sum(errors)
                if len(add_pts) + tri.points.shape[0] >= max_vertices:
                    break
        total_err /= len(pts_tri)
        if not add_pts:
            if minimize_vertices:
                break
            for sidx in simp_idx:
                i = pts_tri == sidx
                # Calculate out SVS error of triangle
                tric = im[i].mean()
                errors = abs(im[i] - tric) ** 2
                add_pts.append(pts[i][errors == errors.max()][0])
                if len(add_pts) + tri.points.shape[0] >= max_vertices:
                    break
        try:
            tri.add_points(np.array(add_pts))
        except IndexError:
            print('Something went wrong.')
            break
    ptx = tri.points[:, 0]
    pty = tri.points[:, 1]
    return ptx, pty, tri.find_simplex(tri.points), tri.simplices


def getGridParams(ref, pos, width, height, npts, az=0):
    shift_x, shift_y, _ = llh2enu(*pos, ref)
    rmat = np.array([[np.cos(az), -np.sin(az)],
                     [np.sin(az), np.cos(az)]]).dot(np.diag([width / npts[0], height / npts[1]]))

    return (shift_x, shift_y), rmat


if __name__ == '__main__':
    from SDRParsing import load
    from simulation_functions import db
    import matplotlib.pyplot as plt

    sdr = load('/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar')

    bg = SDREnvironment(sdr)
    plt.figure('Before')
    plt.imshow(db(bg.refgrid), origin='lower', clim=[130, 160])
    plt.show()
    plane_x = np.arange(100) * np.exp(1j * bg.heading).imag
    plane_y = np.arange(100) * np.exp(1j * bg.heading).real

    x, y, _ = bg.getGrid(width=1225.7, height=1038.25, nrows=4902, ncols=4153, az=bg.heading, use_elevation=False)
    lx, ly, _ = bg.getGrid([40.138538, -111.662090, 1365.8849123907273], 500, 200, 20, 600, use_elevation=False)

    plt.figure('Rel. Positions')
    plt.scatter(plane_x, plane_y)
    plt.scatter(x[::50, ::50].flatten(), y[::50, ::50].flatten())
    plt.scatter(lx.flatten(), ly.flatten())

    bg.resampleGrid([40.138538, -111.662090, 1365.8849123907273], 500, 200, 200, 600)
    plt.figure('After')
    plt.imshow(db(bg.refgrid), origin='lower', clim=[130, 160])
    plt.axis('tight')
