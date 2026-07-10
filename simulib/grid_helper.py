import numpy as np
from .simulation_functions import getElevationMap, llh2enu, enu2llh, getElevation, getElevationTIFF
from .utils import DTR, TAC, c0
from scipy.spatial import Delaunay
from scipy.interpolate import interpn
import pickle

'''
Environment
This is a class to represent the environment of a radar. 
'''


class Environment(object):
    _transform: np.ndarray
    _refgrid: np.ndarray | None
    ref: np.ndarray
    origin: np.ndarray
    rps: float = 1.
    cps: float = 1.

    def __init__(self, rmat: np.ndarray | None = None, reflectivity: np.ndarray | None = None, **kwargs):
        if rmat is not None:
            self.setGrid(reflectivity, rmat)

    def getGridParams(self, pos: tuple[float, float, float] | np.ndarray, along_track_m: float,
                      cross_track_m: float, npts: tuple[int, ...], cross_track_angle=0.) -> np.ndarray:
        shift_x, shift_y, _ = llh2enu(*pos, self.ref)

        # Shift the math such that it rotates from the Y axis clockwise
        cross_corr = cross_track_angle - np.pi / 2

        # Translation
        rmat = np.array([[1, 0, shift_x],
                         [0, 1, shift_y],
                         [0, 0, 1]])
        # Rotation
        rmat = rmat.dot(np.array([[np.cos(cross_corr), np.sin(cross_corr), 0],
                                  [-np.sin(cross_corr), np.cos(cross_corr), 0],
                                  [0, 0, 1.]]))
        # Scaling
        # The -1 offsets the fact that the number of points is one more than the array element index
        # npts is (ncols, nrows)
        w_k = along_track_m / (npts[0] - 1)
        h_k = cross_track_m / (npts[1] - 1)
        rmat = rmat.dot(np.diag([h_k, w_k, 1]))

        self.rps = h_k
        self.cps = w_k

        return rmat

    def getGrid(self, pos: tuple[float, float, float] | None = None, along_track_m: float | None = None,
                cross_track_m: float | None = None, nrows: int = 0, ncols: int = 0, cross_track_angle: float = 0,
                use_elevation: str | bool = True) -> tuple:
        # This grid is independent of the refgrid or stored transforms
        npts = self.shape if nrows == 0 else (ncols, nrows)
        if pos is None and along_track_m is None and cross_track_m is None and nrows == 0 and ncols == 0 and cross_track_angle == 0:
            rmat = self.transforms
        else:
            pos = self.origin if pos is None else pos
            along_track_m = self.shape[0] if along_track_m is None else along_track_m
            cross_track_m = self.shape[1] if cross_track_m is None else cross_track_m
            rmat = self.getGridParams(pos, along_track_m, cross_track_m, npts, cross_track_angle)
        gxx = np.linspace(npts[0] / 2, -npts[0] / 2, npts[0])
        gyy = np.linspace(-npts[1] / 2, npts[1] / 2, npts[1])
        gy, gx = np.meshgrid(gxx, gyy)

        px = rmat[0, 0] * gx + rmat[0, 1] * gy + rmat[0, 2]
        py = rmat[1, 0] * gx + rmat[1, 1] * gy + rmat[1, 2]
        latg, long, altg = enu2llh(px.ravel(), py.ravel(), np.zeros(px.shape[0] * px.shape[1]), self.ref)
        sh = gx.shape
        if isinstance(use_elevation, bool):
            if use_elevation:
                try:
                    gz = (getElevationMap(latg, long, interp_method='splinef2d') - self.ref[2]).reshape(sh)
                except FileNotFoundError:
                    gz = np.zeros(px.shape)
                except Exception as e:
                    gz = np.zeros(px.shape)
                    print(f'Error found: {e}')
            else:
                gz = np.zeros(px.shape)
        else:
            try:
                gz = (getElevationTIFF(use_elevation, latg, long, interp_method='splinef2d') - self.ref[2]).reshape(sh)
            except FileNotFoundError:
                gz = np.zeros(px.shape)
        return (px, py, gz), rmat

    def getRefGrid(self, pos: tuple[float, float, float] | None = None, width: float | None = None,
                   height: float | None = None, nrows: int = 0, ncols: int = 0, az: float = 0,
                   use_elevation: str | None = None) -> np.ndarray:
        x, y, _ = self.getGrid(pos, width, height, nrows, ncols, az, use_elevation)
        irmat = np.linalg.pinv(self._transform)
        px = self.shape[1] - (irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2)
        py = self.shape[0] - (irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2)
        pos_r = np.stack([px.ravel(), py.ravel()]).T
        return interpn((np.arange(self.refgrid.shape[1]),
                        np.arange(self.refgrid.shape[0])), self.refgrid.T, pos_r, bounds_error=False,
                       fill_value=0).reshape(x.shape, order='C')

    def setGrid(self, newgrid: np.ndarray | None, rmat: np.ndarray) -> None:
        self._refgrid = newgrid
        self._transform = rmat

    def resampleGrid(self, pos: tuple[float, float, float], width: float, height: float,
                     nrows: int, ncols: int, az: float = 0) -> None:
        x, y, _ = self.getGrid(pos, width, height, nrows, ncols, az)
        irmat = np.linalg.pinv(self._transform)
        px = irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2
        py = irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2
        pos_r = np.stack([px.ravel(), py.ravel()]).T
        self.setGrid(interpn((np.arange(self.refgrid.shape[1]),
                              np.arange(self.refgrid.shape[0])), self.refgrid.T, pos_r, bounds_error=False,
                             fill_value=0).reshape(x.shape, order='C'),
                     self.getGridParams(pos, width, height, (nrows, ncols), az))

    def sample(self, x: float, y: float) -> np.ndarray:
        irmat = np.linalg.pinv(self._transform)
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
        pos_x = self._transform[0, 0] * gx + self._transform[0, 1] * gy + self._transform[0, 2]
        pos_y = self._transform[1, 0] * gx + self._transform[1, 1] * gy + self._transform[1, 2]
        if not elevation:
            return np.array([pos_x, pos_y]).T
        lat, lon, _ = (
            enu2llh(pos_x, pos_y, 0, self.ref)
            if isinstance(px, float)
            else enu2llh(pos_x, pos_y, np.zeros_like(pos_x), self.ref)
        )
        return np.array([pos_x, pos_y, getElevation(lat, lon) - self.ref[2]]) if isinstance(px, float) else (
            np.array([pos_x, pos_y, getElevationMap(lat, lon) - self.ref[2]]).T)

    def getIndex(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        irmat = np.linalg.pinv(self._transform)
        px = irmat[0, 0] * x + irmat[0, 1] * y + irmat[0, 2] + self.shape[1] / 2
        py = irmat[1, 0] * x + irmat[1, 1] * y + irmat[1, 2] + self.shape[0] / 2
        return np.array([px, py])

    def interp(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        if isinstance(x, float):
            return interpn((np.arange(self.refgrid.shape[0]),
                            np.arange(self.refgrid.shape[1])), self.refgrid, self.getIndex(x, y))
        else:
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
        return self._transform


class MapEnvironment(Environment):

    def __init__(self, origin, extent: tuple[int, int], pixels_per_meter: tuple[float, float] = (1., 1.), ref=None,
                 background=None, az=0.):
        self.origin = origin
        self.ref = origin if ref is None else ref
        self.heading = az
        self.cps = pixels_per_meter[0]
        self.rps = pixels_per_meter[1]
        rmat = self.getGridParams(self.origin, extent[0] * self.cps, extent[1] * self.rps, extent,
                                  az)
        bg = np.ones(extent) if background is None else background
        super().__init__(rmat=rmat, reflectivity=bg)


class SDREnvironment(Environment):
    rps: float = 1
    cps: float = 1
    cross_track_angle: float = 0.

    def __init__(self, sdr, local_grid: np.ndarray | None = None,
                 origin: tuple[float, float, float] | np.ndarray | None = None):
        print('SDR loaded')
        try:
            asi = sdr.loadASI(sdr.files['asi'])
            grid = abs(asi)
        except KeyError:
            print('ASI not found.')
            asi = np.random.rand(200, 200)
            asi[25, 25] = 10
            asi[75, 75] = 10
            grid = asi
        except TypeError:
            asi = sdr.loadASI(sdr.files['asi'][0])
            grid = abs(asi)
        except FileNotFoundError:
            print('ASI not found.')
            asi = np.random.rand(200, 200)
            asi[25, 25] = 10
            asi[75, 75] = 10
            grid = asi
        self._sdr = sdr
        self._asi = asi
        if sdr.gim is not None:
            look_angle = np.pi / 2 if sdr.gim.look_side == 'Right' else -np.pi / 2
            look_angle += sdr.gim.squint_angle * DTR
        else:
            look_angle = -np.pi / 2
        self.cross_track_angle = np.arctan2(sdr.gps_ve.mean(), sdr.gps_vn.mean()) + look_angle
        if sdr.ash is None:
            pt = (sdr.gps_lat.mean(), sdr.gps_lon.mean())
            alt = getElevation(*pt)
            hght = sdr.gps_alt.mean() - alt
            try:
                nrange = ((sdr[0].receive_on_TAC - sdr[0].transmit_on_TAC) / TAC) * c0 / 2
                frange = ((sdr[0].receive_off_TAC - sdr[0].transmit_on_TAC) / TAC -
                          sdr[0].pulse_length_S) * c0 / 2
            except AttributeError:
                nrange = ((sdr[0].Receive_On_TAC - sdr[0].Transmit_On_TAC) / TAC) * c0 / 2
                frange = ((sdr[0].Receive_Off_TAC - sdr[0].Transmit_On_TAC) / TAC -
                          sdr[0].pulse_length_S) * c0 / 2
            mrange = np.sqrt(((frange + nrange) / 2)**2 - hght**2)
            if origin is None:
                ref_llh = origin = enu2llh(mrange * np.sin(self.cross_track_angle), mrange * np.cos(self.cross_track_angle), 0.,
                                           (pt[0], pt[1], alt))
            else:
                ref_llh = enu2llh(mrange * np.sin(self.cross_track_angle), mrange * np.cos(self.cross_track_angle), 0.,
                                  (pt[0], pt[1], alt))
        else:
            if origin is None:
                origin = (sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX'],
                          getElevation(sdr.ash['geo']['centerY'], sdr.ash['geo']['centerX']))
            ref_llh = (sdr.ash['geo']['refLat'], sdr.ash['geo']['refLon'],
                       sdr.ash['geo']['hRef'])
            self.rps = sdr.ash['geo']['rowPixelSizeM']
            self.cps = sdr.ash['geo']['colPixelSizeM']
            self.cross_track_angle = sdr.ash['flight']['flnHdg'] * DTR - np.pi / 2

        self.origin = np.array(origin)
        self.ref = np.array(ref_llh)

        grid = local_grid if local_grid is not None else grid

        rmat = self.getGridParams(self.origin, grid.shape[0] * self.cps, grid.shape[1] * self.rps, grid.shape,
                                  self.cross_track_angle)

        super().__init__(rmat=rmat, reflectivity=grid)

    def gridFromSwath(self, near_range, far_range, bandwidth: float = 6.6e8, beamwidth: float = 0.):
        alt = self._sdr.gps_alt.mean() - getElevation(self._sdr.gps_lat.mean(), self._sdr.gps_lon.mean())
        nrange = np.sqrt(near_range**2 - alt**2)
        frange = np.sqrt(far_range**2 - alt**2)
        bottom_enu = llh2enu(self._sdr.gps_lat[0], self._sdr.gps_lon[0], self._sdr.gps_alt[0], self.ref)
        top_enu = llh2enu(self._sdr.gps_lat[-1], self._sdr.gps_lon[-1], self._sdr.gps_alt[-1], self.ref)
        bottom_corner = np.array([bottom_enu[0] + nrange * np.sin(self.cross_track_angle), bottom_enu[1] + nrange * np.cos(self.cross_track_angle), 0])
        top_corner = np.array([top_enu[0] + frange * np.sin(self.cross_track_angle), top_enu[1] + frange * np.cos(self.cross_track_angle), 0])
        resolution = 1.2 * c0 / (2 * bandwidth)
        origin_enu = (bottom_corner + top_corner) / 2.
        origin_llh = enu2llh(*origin_enu, self.ref)
        width = np.linalg.norm(np.array(bottom_enu) - np.array(top_enu)) - 2 * np.tan(beamwidth) * far_range
        height = float(np.sqrt(np.linalg.norm(bottom_corner - top_corner)**2 - width**2))
        return self.getGrid(origin_llh, width, height, min(512, int(width / resolution)), min(512, int(height / resolution)),
                            cross_track_angle=self.cross_track_angle), width, height, origin_llh

    @property
    def sdr(self):
        return self._sdr


class SAREnvironment(Environment):
    rps: float = 1
    cps: float = 1
    heading: float = 0.

    def __init__(self, sar, local_grid=None, origin=None, local_height: float | None = None, use_tiff: str | None = None):
        print('SDR loaded')
        if local_grid is None:
            try:
                asi = sar.loadASI(sar.files['asi'])
                grid = abs(asi)
            except KeyError:
                print('ASI not found.')
                asi = np.random.rand(2000, 2000)
                asi[250, 250] = 10
                asi[750, 750] = 10
                grid = asi
            except TypeError:
                asi = sar.loadASI(sar.files['asi'][0])
                grid = abs(asi)
            except FileNotFoundError:
                print('ASI not found.')
                asi = np.random.rand(2000, 2000)
                asi[250, 250] = 10
                asi[750, 750] = 10
                grid = asi
        else:
            grid = local_grid
            asi = None
        self._sar = sar
        self._asi = asi
        self.heading = np.arctan2(sar.gps_data['ve'].values[0], sar.gps_data['vn'].values[0])
        if sar.ash is None:
            try:
                hght = sar.xml['Flight_Line']['Flight_Line_Altitude_M']
                pt = ((sar.xml['Flight_Line']['Start_Latitude_D'] + sar.xml['Flight_Line']['Stop_Latitude_D']) / 2,
                      (sar.xml['Flight_Line']['Start_Longitude_D'] + sar.xml['Flight_Line']['Stop_Longitude_D']) / 2)
                alt = local_height if local_height is not None else getElevation(*pt)
            except KeyError:
                alt = sar.gps_data['alt'].mean()
                pt = (sar.gps_data['lat'].mean(), sar.gps_data['lon'].mean())
                hght = alt + (local_height if local_height is not None else getElevation(*pt))
            mrange = hght / np.tan(sar.ant[0].dep_ang)
            if origin is None:
                ref_llh = origin = enu2llh(mrange * np.sin(self.heading), mrange * np.cos(self.heading), 0.,
                                           (pt[0], pt[1], alt))
            else:
                ref_llh = enu2llh(mrange * np.sin(self.heading), mrange * np.cos(self.heading), 0.,
                                  (pt[0], pt[1], alt))
        else:
            if origin is None:
                origin = (sar.ash['geo']['centerY'], sar.ash['geo']['centerX'],
                          local_height if local_height is not None else getElevation(sar.ash['geo']['centerY'], sar.ash['geo']['centerX'])
                          )
            ref_llh = (sar.ash['geo']['refLat'], sar.ash['geo']['refLon'],
                       sar.ash['geo']['hRef'])
            self.rps = sar.ash['geo']['rowPixelSizeM']
            self.cps = sar.ash['geo']['colPixelSizeM']
            self.heading = sar.ash['flight']['flnHdg'] * DTR

        self.origin = np.array(origin)
        self.ref = np.array(ref_llh)

        grid = local_grid if local_grid is not None else grid

        rmat = self.getGridParams(self.origin, grid.shape[0] * self.cps, grid.shape[1] * self.rps, grid.shape,
                                         self.heading)

        super().__init__(rmat=rmat, reflectivity=grid)

    @property
    def sar(self):
        return self._sar


def createMesh(ptx, pty, ref_im, tri_err, max_vertices, max_iters=20, minimize_vertices=True):
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
    bggrid = np.ones((200, 200))
    bggrid[::50, ::50] = 100
    test = MapEnvironment((40.011, -111.-11, 1380), (200, 300), background=bggrid, az=np.pi / 3)
    # from simulation_functions import db
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(test.refgrid)
    tgx, tgy, tgz = test.getGrid()

    plt.figure()
    plt.scatter(tgx.flatten(), tgy.flatten())
    plt.show()
