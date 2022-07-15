import numpy as np
from scipy.interpolate import CubicSpline
from SDRParsing import SDRParse
from simulation_functions import llh2enu, findPowerOf2
from scipy.spatial.transform import Rotation as rot

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254


class Platform(object):
    _pos = None
    _vel = None
    _att = None
    _heading = None

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None, gimbal=None, gimbal_offset=None,
                 gimbal_rotations=None, ant_offset=None):
        self._gpst = t
        self._ant = ant_offset
        self._gimbal = gimbal
        self._gimbal_offset = gimbal_offset

        # attitude spline
        rr = CubicSpline(t, r)
        pp = CubicSpline(t, p)
        yy = CubicSpline(t, y)
        self._att = lambda lam_t: np.array([rr(lam_t), pp(lam_t), yy(lam_t)])

        # Take into account the gimbal if necessary
        gphi = None
        gtheta = None
        if gimbal is not None:
            # Matrix to rotate from body to inertial frame for each INS point
            Rbi = [rot.from_rotvec([p[i], r[i], y[i]]) for i in range(len(p))]

            # Account for gimbal frame mounting rotations
            Rgb2g = rot.from_rotvec(np.array([0, gimbal_rotations[0], 0]))
            Rb2gblg = rot.from_rotvec(np.array([gimbal_rotations[1], 0, 0]))
            Rblgb = rot.from_rotvec(np.array([0, 0, gimbal_rotations[2]]))
            # This is because the gimbal is mounted upside down
            Rmgg = rot.from_rotvec([0, -np.pi, 0])
            Rgb = Rmgg * Rgb2g * Rb2gblg * Rblgb
            ant_offsets = ant_offset if ant_offset is not None else np.array([0., 0., 0.])

            # Convert gimbal angles to rotations
            Rmg = [rot.from_rotvec([gimbal[n, 1], 0, gimbal[n, 0]])
                   for n in range(gimbal.shape[0])]

            # Apply rotations through antenna frame, gimbal frame, and add to gimbal offsets
            gamma_b_gpc = [(Rgb * n).inv().apply(ant_offsets).flatten() + gimbal_offset for n in Rmg]

            # Rotate gimbal/antenna offsets into inertial frame
            rotated_offsets = np.array([Rbi[i].inv().apply(gamma_b_gpc[i]).flatten()
                                        for i in range(gimbal.shape[0])])

            # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
            e += rotated_offsets[:, 1]
            n += rotated_offsets[:, 0]
            u -= rotated_offsets[:, 2]

            # Rotate antenna into inertial frame in the same way as above
            boresight = np.array([0, 0, 1])
            bai = np.array([(Rbi[n].inv() * (Rgb * Rmg[n]).inv()).apply(boresight).flatten()
                            for n in range(len(Rbi))])

            # Calculate antenna azimuth/elevation for beampattern
            # gphi = y - np.pi / 2 if gphi is None else gphi
            # gtheta = np.zeros(len(t)) + 20 * DTR if gtheta is None else gtheta
            gtheta = np.arcsin(-bai[:, 2])
            gphi = np.arctan2(-bai[:, 1], bai[:, 0])

        # Build the position spline
        ee = CubicSpline(t, e)
        nn = CubicSpline(t, n)
        uu = CubicSpline(t, u)
        self._pos = lambda lam_t: np.array([ee(lam_t), nn(lam_t), uu(lam_t)])

        # Build a velocity spline
        ve = CubicSpline(t, np.gradient(e))
        vn = CubicSpline(t, np.gradient(n))
        vu = CubicSpline(t, np.gradient(u))
        self._vel = lambda lam_t: np.array([ve(lam_t), vn(lam_t), vu(lam_t)])

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[0], self._vel(lam_t)[1])

        # Beampattern stuff
        gphi = self._heading(t) - np.pi / 2 if gphi is None else gphi
        gtheta = np.zeros(len(t)) + 45 * DTR if gtheta is None else gtheta
        self.pan = CubicSpline(t, gphi)
        self.tilt = CubicSpline(t, gtheta)

    @property
    def pos(self):
        return self._pos

    @property
    def heading(self):
        return self._heading

    @property
    def att(self):
        return self._att

    @property
    def gpst(self):
        return self._gpst


class RadarPlatform(Platform):

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None, ant_offset=None, gimbal=None,
                 gimbal_offset=None, gimbal_rotations=None, dep_angle=45.,
                 squint_angle=0., az_bw=10., el_bw=10., fs=2e9):
        super().__init__(e, n, u, r, p, y, t, gimbal, gimbal_offset, gimbal_rotations, ant_offset)
        self.dep_ang = dep_angle * DTR
        self.squint_ang = squint_angle * DTR
        self.az_half_bw = az_bw * DTR / 2
        self.el_half_bw = el_bw * DTR / 2
        self.fs = fs
        self.near_range_angle = self.dep_ang + self.el_half_bw
        self.far_range_angle = self.dep_ang - self.el_half_bw

    def calcRanges(self, height):
        nrange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang - self.el_half_bw)
        frange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang + self.el_half_bw)
        return nrange, frange

    def calcPulseLength(self, height, pulse_length_percent=1., use_tac=False):
        nrange, _ = self.calcRanges(height)
        plength_s = (nrange * 2 / c0 - 1 / TAC) * pulse_length_percent
        return int(plength_s * self.fs) if use_tac else plength_s

    def calcNumSamples(self, height, plp=1.):
        nrange, frange = self.calcRanges(height)
        pl_s = self.calcPulseLength(height, plp)
        return int((np.ceil((2 * frange / c0 + pl_s) * TAC) - np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)

    def calcRangeBins(self, height, upsample=1, plp=1.):
        nrange, frange = self.calcRanges(height)
        pl_s = self.calcPulseLength(height, plp)
        nsam = int((np.ceil((2 * frange / c0 + pl_s) * TAC) -
                    np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)
        MPP = c0 / self.fs / upsample / 2
        return nrange + np.arange(nsam * upsample) * MPP + c0 / self.fs

    def intoBodyFrame(self, pt, t):
        return rot.from_euler('zxy', self._att(t)).apply(pt)

    def fromBodyFrame(self, pt, t):
        return rot.from_euler('zxy', self._att(t)).inv().apply(pt)


class SDRPlatform(RadarPlatform):
    _sdr = None

    def __init__(self, sdr_file, origin=None, ant_offsets=None, fs=None, channel=0):
        sdr = SDRParse(sdr_file) if type(sdr_file) == str else sdr_file
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[:, 0])
        e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
        pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                        sdr.gimbal['pan'].values.astype(np.float64))
        tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                        sdr.gimbal['tilt'].values.astype(np.float64))
        goff = np.array([sdr.xml['Common_Channel_Settings']['Gimbal_Settings']['Gimbal_X_Offset_M'],
                         sdr.xml['Common_Channel_Settings']['Gimbal_Settings']['Gimbal_Y_Offset_M'],
                         sdr.xml['Common_Channel_Settings']['Gimbal_Settings']['Gimbal_Z_Offset_M']])
        grot = np.array([sdr.xml['Common_Channel_Settings']['Gimbal_Settings']['Roll_D'] * DTR,
                         sdr.xml['Common_Channel_Settings']['Gimbal_Settings']['Pitch_D'] * DTR,
                         sdr.xml['Common_Channel_Settings']['Gimbal_Settings']['Yaw_D'] * DTR])
        channel_dep = (sdr.xml['Channel_0']['Near_Range_D'] + sdr.xml['Channel_0']['Far_Range_D']) / 2 * DTR
        ant_num = sdr[channel].trans_num
        if ant_offsets is None:
            # ant_offsets = np.array([sdr.port[ant_num].x, sdr.port[ant_num].y, sdr.port[ant_num].z])
            ant_offsets = sum([np.array([sdr.port[n].x, sdr.port[n].y, sdr.port[n].z]) for n in range(len(sdr.port))]) / len(sdr.port)
        super().__init__(e=e, n=n, u=u, r=sdr.gps_data['r'].values, p=sdr.gps_data['p'].values,
                         y=sdr.gps_data['y'].values,
                         t=sdr.gps_data.index.values, ant_offset=ant_offsets, gimbal=np.array([pan, tilt]).T,
                         gimbal_offset=goff, gimbal_rotations=grot, dep_angle=channel_dep,
                         squint_angle=sdr.ant[ant_num].squint / DTR, az_bw=sdr.ant[ant_num].az_bw / DTR,
                         el_bw=sdr.ant[ant_num].el_bw / DTR, fs=fs)
        self._sdr = sdr
        self.origin = origin

    def calcRanges(self, fdelay):
        nrange = ((self._sdr[0].receive_on_TAC - self._sdr[0].transmit_on_TAC - fdelay) / TAC) * c0 / 2
        # nrange = ((self._sdr[0].receive_on_TAC - self._sdr[0].transmit_on_TAC - fdelay) / TAC -
        #           (findPowerOf2(self._sdr[0].nsam + self._sdr[0].pulse_length_N) - self._sdr[
        #               0].nsam) / self.fs) * c0 / 2
        frange = nrange + self._sdr[0].nsam * c0 / 2 / self.fs
        return nrange, frange

    def calcPulseLength(self, height, pulse_length_percent=1., use_tac=False):
        return self._sdr[0].pulse_length_N if use_tac else self._sdr[0].pulse_length_S

    def calcNumSamples(self, height, plp=1.):
        return self._sdr[0].nsam

    def calcRangeBins(self, height, upsample=1, plp=1.):
        nrange, frange = self.calcRanges(height)
        MPP = c0 / self.fs / 2 / upsample
        return nrange + np.arange(self.calcNumSamples(height, plp) * upsample) * MPP
