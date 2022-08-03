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
                 gimbal_rotations=None, tx_offset=None, rx_offset=None, gps_data=None):
        self._gpst = t
        self._txant = tx_offset
        self._rxant = rx_offset
        self._gimbal = gimbal
        self._gimbal_offset = gimbal_offset

        # attitude spline
        rr = CubicSpline(t, r)
        pp = CubicSpline(t, p)
        yy = CubicSpline(t, y)
        self._att = lambda lam_t: np.array([rr(lam_t), pp(lam_t), yy(lam_t)])

        # Take into account the gimbal if necessary
        if gimbal is not None:
            # Matrix to rotate from body to inertial frame for each INS point
            cr = np.cos(gimbal_rotations[0])
            sr = np.sin(gimbal_rotations[0])
            cp = np.cos(gimbal_rotations[1])
            sp = np.sin(gimbal_rotations[1])
            cy = np.cos(gimbal_rotations[2])
            sy = np.sin(gimbal_rotations[2])
            cpan = np.cos(gimbal[:, 0])
            ct = np.cos(gimbal[:, 1])
            span = np.sin(gimbal[:, 0])
            st = np.sin(gimbal[:, 1])
            tx_offset = tx_offset if tx_offset is not None else np.array([0., 0., 0.])
            rx_offset = rx_offset if rx_offset is not None else np.array([0., 0., 0.])

            # Gimbal to body frame rotations
            J = [cpan * tx_offset[0] + span * ct * tx_offset[1] - span * st * tx_offset[2],
                          -span * tx_offset[0] + cpan * ct * tx_offset[1] - cpan * st * tx_offset[2],
                          st * tx_offset[1] + ct * tx_offset[2]]
            g2b = np.array([(-cp * cy - sr * sp * sy) * J[0] + cp * sy * J[1] + (cr * sp * sy - sr * cy) * J[2],
                            (cr * sy - sr * sp * cy) * J[0] + cp * cy * J[1] + (sr * sy + cr * sp * cy) * J[2],
                            sr * cp * J[0] + sp * J[1] - cr * cp * J[2]])
            a2_itx = gimbal_offset[:, None] + g2b

            # Repeat for receive antenna
            J = [cpan * rx_offset[0] + span * ct * rx_offset[1] - span * st * rx_offset[2],
                 -span * rx_offset[0] + cpan * ct * rx_offset[1] - cpan * st * rx_offset[2],
                 st * rx_offset[1] + ct * rx_offset[2]]
            g2b = np.array([(-cp * cy - sr * sp * sy) * J[0] + cp * sy * J[1] + (cr * sp * sy - sr * cy) * J[2],
                            (cr * sy - sr * sp * cy) * J[0] + cp * cy * J[1] + (sr * sy + cr * sp * cy) * J[2],
                            sr * cp * J[0] + sp * J[1] - cr * cp * J[2]])
            a2_irx = gimbal_offset[:, None] + g2b

            # Boresight angle using lever arm in Z direction
            Jb = [-span * st, -cpan * st, ct]
            bore_offsets = np.array([(-cp * cy - sr * sp * sy) * Jb[0] + cp * sy * Jb[1] + (cr * sp * sy - sr * cy) * Jb[2],
                            (cr * sy - sr * sp * cy) * Jb[0] + cp * cy * Jb[1] + (sr * sy + cr * sp * cy) * Jb[2],
                            sr * cp * Jb[0] + sp * Jb[1] - cr * cp * Jb[2]])

            # Final body to inertial frame calc
            cr = np.cos(r)
            sr = np.sin(r)
            cp = np.cos(p)
            sp = np.sin(p)
            cy = np.cos(y)
            sy = np.sin(y)
            b2_itx = np.array([(cr * cy + sr * sp * sy) * a2_itx[0] + cp * a2_itx[1] + (sr * cy - cr * sp * sy) * a2_itx[2],
                            (-cr * sy + sr * sp * sy) * a2_itx[0] + cp * a2_itx[1] + (-sr * sy - cr * sp * cy) * a2_itx[2],
                            -sr * cp * a2_itx[0] + sp * a2_itx[1] + cr * cp * a2_itx[2]])
            b2_irx = np.array(
                [(cr * cy + sr * sp * sy) * a2_irx[0] + cp * a2_irx[1] + (sr * cy - cr * sp * sy) * a2_irx[2],
                 (-cr * sy + sr * sp * sy) * a2_irx[0] + cp * a2_irx[1] + (-sr * sy - cr * sp * cy) * a2_irx[2],
                 -sr * cp * a2_irx[0] + sp * a2_irx[1] + cr * cp * a2_irx[2]])

            # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
            new_t = t
            if gps_data is not None:
                te = gps_data['te']
                tn = gps_data['tn']
                tu = gps_data['tu']
                re = gps_data['re']
                rn = gps_data['rn']
                ru = gps_data['ru']
                e = te
                n = tn
                u = tu
                new_t = gps_data['sec']
            else:
                te = e + b2_itx[0]
                tn = n + b2_itx[1]
                tu = u + b2_itx[2]
                re = e + b2_irx[0]
                rn = n + b2_irx[1]
                ru = u + b2_irx[2]

            # Rotate antenna into inertial frame in the same way as above
            bai = np.array([(cr * cy + sr * sp * sy) * bore_offsets[0] + cp * bore_offsets[1] +
                            (sr * cy - cr * sp * sy) * bore_offsets[2],
                            (-cr * sy + sr * sp * sy) * bore_offsets[0] + cp * bore_offsets[1] +
                            (-sr * sy - cr * sp * cy) * bore_offsets[2],
                            -sr * cp * bore_offsets[0] + sp * bore_offsets[1] + cr * cp * bore_offsets[2]])

            # Calculate antenna azimuth/elevation for beampattern
            # gphi = y - np.pi / 2 if gphi is None else gphi
            # gtheta = np.zeros(len(t)) + 20 * DTR if gtheta is None else gtheta
            if gps_data is not None:
                gtheta = np.interp(new_t, t, np.arcsin(-bai[2, :]))
                gphi = np.interp(new_t, t, np.arctan2(bai[0, :], bai[1, :]))
                t = new_t
            else:
                gtheta = np.arcsin(-bai[2, :])
                gphi = np.arctan2(bai[0, :], bai[1, :])
        else:
            te = re = e
            tn = rn = n
            tu = ru = u
            gphi = y - np.pi / 2
            gtheta = np.zeros(len(t)) + 20 * DTR

        # Build the position spline
        ee = CubicSpline(t, e)
        nn = CubicSpline(t, n)
        uu = CubicSpline(t, u)
        self._pos = lambda lam_t: np.array([ee(lam_t), nn(lam_t), uu(lam_t)])
        tte = CubicSpline(t, te)
        ttn = CubicSpline(t, tn)
        ttu = CubicSpline(t, tu)
        self._txpos = lambda lam_t: np.array([tte(lam_t), ttn(lam_t), ttu(lam_t)])
        rre = CubicSpline(t, re)
        rrn = CubicSpline(t, rn)
        rru = CubicSpline(t, ru)
        self._rxpos = lambda lam_t: np.array([rre(lam_t), rrn(lam_t), rru(lam_t)])

        # Build a velocity spline
        ve = CubicSpline(t, np.gradient(e))
        vn = CubicSpline(t, np.gradient(n))
        vu = CubicSpline(t, np.gradient(u))
        self._vel = lambda lam_t: np.array([ve(lam_t), vn(lam_t), vu(lam_t)])

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[0], self._vel(lam_t)[1])

        # Beampattern stuff
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

    @property
    def rxpos(self):
        return self._rxpos

    @property
    def txpos(self):
        return self._txpos


class RadarPlatform(Platform):

    def __init__(self, e=None, n=None, u=None, r=None, p=None, y=None, t=None, tx_offset=None, rx_offset=None,
                 gimbal=None, gimbal_offset=None, gimbal_rotations=None, dep_angle=45.,
                 squint_angle=0., az_bw=10., el_bw=10., fs=2e9, gps_data=None):
        super().__init__(e, n, u, r, p, y, t, gimbal, gimbal_offset, gimbal_rotations, tx_offset, rx_offset, gps_data)
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

    def __init__(self, sdr_file, origin=None, tx_offset=None, rx_offset=None, fs=None, channel=0, gps_data=None):
        sdr = SDRParse(sdr_file) if type(sdr_file) == str else sdr_file
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[:, 0])
        if gps_data is not None:
            gps_data['te'], gps_data['tn'], gps_data['tu'] = llh2enu(gps_data['tx_lat'], gps_data['tx_lon'],
                                                                     gps_data['tx_alt'], origin)
            gps_data['re'], gps_data['rn'], gps_data['ru'] = llh2enu(gps_data['rx_lat'], gps_data['rx_lon'],
                                                                     gps_data['rx_alt'], origin)
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
        tx_num = sdr[channel].trans_num
        tx_offset = np.array([sdr.port[tx_num].x, sdr.port[tx_num].y, sdr.port[tx_num].z]) if tx_offset is None else tx_offset
        rx_num = sdr[channel].rec_num
        rx_offset = np.array([sdr.port[rx_num].x, sdr.port[rx_num].y, sdr.port[rx_num].z]) if rx_offset is None else rx_offset
        super().__init__(e=e, n=n, u=u, r=sdr.gps_data['r'].values, p=sdr.gps_data['p'].values,
                         y=sdr.gps_data['y'].values,
                         t=sdr.gps_data.index.values, tx_offset=tx_offset, rx_offset=rx_offset, gimbal=np.array([pan, tilt]).T,
                         gimbal_offset=goff, gimbal_rotations=grot, dep_angle=channel_dep,
                         squint_angle=sdr.ant[tx_num].squint / DTR, az_bw=sdr.ant[tx_num].az_bw / DTR,
                         el_bw=sdr.ant[tx_num].el_bw / DTR, fs=fs, gps_data=gps_data)
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
