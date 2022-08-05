import numpy as np
from scipy.interpolate import CubicSpline
from SDRParsing import SDRParse
from simulation_functions import llh2enu, findPowerOf2, loadPostCorrectionsGPSData, loadRawData, loadGimbalData, \
    loadMatchedFilter, loadReferenceChirp
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
        new_t = gps_data['sec'] if gps_data is not None else t

        # attitude spline
        rr = CubicSpline(t, r)
        pp = CubicSpline(t, p)
        yy = CubicSpline(t, y) if gps_data is None else CubicSpline(new_t, gps_data['az'] + 2 * np.pi)
        self._att = lambda lam_t: np.array([rr(lam_t), pp(lam_t), yy(lam_t)])

        # Take into account the gimbal if necessary
        if gimbal is not None:
            gimbal_offset_mat = getRotationOffsetMatrix(*gimbal_rotations)
            # Matrix to rotate from body to inertial frame for each INS point
            tx_offset = tx_offset if tx_offset is not None else np.array([0., 0., 0.])
            rx_offset = rx_offset if rx_offset is not None else np.array([0., 0., 0.])

            # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
            if gps_data is not None:
                te = gps_data['te']
                tn = gps_data['tn']
                tu = gps_data['tu']
                re = gps_data['re']
                rn = gps_data['rn']
                ru = gps_data['ru']
            else:
                tx_corrs = np.array([getPhaseCenterInertialCorrection(gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n],
                                                                 p[n], r[n], tx_offset, gimbal_offset)
                                for n in range(gimbal.shape[0])])
                rx_corrs = np.array(
                    [getPhaseCenterInertialCorrection(gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n],
                                                      p[n], r[n], rx_offset, gimbal_offset)
                     for n in range(gimbal.shape[0])])
                te = e + tx_corrs[:, 0]
                tn = n + tx_corrs[:, 1]
                tu = u + tx_corrs[:, 2]
                re = e + rx_corrs[:, 0]
                rn = n + rx_corrs[:, 1]
                ru = u + rx_corrs[:, 2]

            # Rotate antenna into inertial frame in the same way as above
            bai = np.array([getBoresightVector(gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n], p[n], r[n])
                            for n in range(gimbal.shape[0])])
            bai = bai.reshape((bai.shape[0], bai.shape[1])).T

            # Calculate antenna azimuth/elevation for beampattern
            # gphi = y - np.pi / 2 if gphi is None else gphi
            # gtheta = np.zeros(len(t)) + 20 * DTR if gtheta is None else gtheta
            if gps_data is not None:
                gtheta = np.interp(new_t, t, np.arcsin(-bai[2, :]))
                gphi = np.interp(new_t, t, np.arctan2(bai[0, :], bai[1, :]))
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
        tte = CubicSpline(new_t, te)
        ttn = CubicSpline(new_t, tn)
        ttu = CubicSpline(new_t, tu)
        self._txpos = lambda lam_t: np.array([tte(lam_t), ttn(lam_t), ttu(lam_t)])
        rre = CubicSpline(new_t, re)
        rrn = CubicSpline(new_t, rn)
        rru = CubicSpline(new_t, ru)
        self._rxpos = lambda lam_t: np.array([rre(lam_t), rrn(lam_t), rru(lam_t)])

        # Build a velocity spline
        ve = CubicSpline(t, np.gradient(e))
        vn = CubicSpline(t, np.gradient(n))
        vu = CubicSpline(t, np.gradient(u))
        self._vel = lambda lam_t: np.array([ve(lam_t), vn(lam_t), vu(lam_t)])

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[0], self._vel(lam_t)[1])

        # Beampattern stuff
        self.pan = CubicSpline(new_t, gphi)
        self.tilt = CubicSpline(new_t, gtheta)

    def boresight(self, t):
        # Get a boresight angle from the pan/tilt values
        az = np.exp(1j * self.tilt(t))
        return np.array([az.imag, az.real, -np.sin(self.pan(t))])

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

    @property
    def vel(self):
        return self._vel



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

    def __init__(self, sdr_file, origin=None, tx_offset=None, rx_offset=None, fs=None, channel=0, debug_fnme=None,
                 gimbal_debug=None):
        sdr = SDRParse(sdr_file) if type(sdr_file) == str else sdr_file
        t = sdr.gps_data.index.values
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[0, :])
        if debug_fnme is not None:
            gps_data = loadPostCorrectionsGPSData(debug_fnme)
            gps_data['te'], gps_data['tn'], gps_data['tu'] = llh2enu(gps_data['tx_lat'], gps_data['tx_lon'],
                                                                     gps_data['tx_alt'], origin)
            gps_data['re'], gps_data['rn'], gps_data['ru'] = llh2enu(gps_data['rx_lat'], gps_data['rx_lon'],
                                                                     gps_data['rx_alt'], origin)
        else:
            gps_data = None
        e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
        if gimbal_debug is None:
            pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                            sdr.gimbal['pan'].values.astype(np.float64))
            tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                            sdr.gimbal['tilt'].values.astype(np.float64))
        else:
            gim = loadGimbalData(gimbal_debug)
            times = np.interp(gim['systime'], sdr.gps_data['systime'], sdr.gps_data.index)
            pan = np.interp(t, times, gim['pan'])
            tilt = np.interp(t, times, gim['tilt'])
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
                         t=t, tx_offset=tx_offset, rx_offset=rx_offset, gimbal=np.array([pan, tilt]).T,
                         gimbal_offset=goff, gimbal_rotations=grot, dep_angle=channel_dep,
                         squint_angle=sdr.ant[tx_num].squint / DTR, az_bw=sdr.ant[tx_num].az_bw / DTR,
                         el_bw=sdr.ant[tx_num].el_bw / DTR, fs=fs, gps_data=gps_data)
        self._sdr = sdr
        self.origin = origin
        self._channel = channel

    def calcRanges(self, fdelay, partial_pulse_percent=1.):
        nrange = ((self._sdr[0].receive_on_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                  self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        # nrange = ((self._sdr[0].receive_on_TAC - self._sdr[0].transmit_on_TAC - fdelay) / TAC -
        #           (findPowerOf2(self._sdr[0].nsam + self._sdr[0].pulse_length_N) - self._sdr[
        #               0].nsam) / self.fs) * c0 / 2
        frange = ((self._sdr[0].receive_off_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                  self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        return nrange, frange

    def calcPulseLength(self, height, pulse_length_percent=1., use_tac=False):
        return self._sdr[self._channel].pulse_length_N if use_tac else self._sdr[self._channel].pulse_length_S

    def calcNumSamples(self, height, plp=1.):
        return self._sdr[self._channel].nsam

    def calcRangeBins(self, height, upsample=1, plp=1.):
        nrange, frange = self.calcRanges(height)
        MPP = c0 / self.fs / 2 / upsample
        return nrange * 2 + np.arange(self.calcNumSamples(height, plp) * upsample) * MPP
    
    
def bodyToInertial(yaw, pitch, roll, x, y, z):
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    # compute the inertial to body rotation matrix
    rotItoB = np.array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp],
        [cp * sy, cp * cy, sp],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp]])
    return rotItoB.T.dot(np.array([x, y, z]))


def gimbalToBody(rotBtoMG, pan, tilt, x, y, z):
    cp = np.cos(pan)
    sp = np.sin(pan)
    ct = np.cos(tilt)
    st = np.sin(tilt)

    rotMGtoGP = np.array([
        [cp, -sp, 0],
        [sp * ct, cp * ct, st],
        [-sp * st, -cp * st, ct]])

    # compute the gimbal mounted to gimbal pointing rotation matrix
    rotBtoGP = rotMGtoGP.dot(rotBtoMG)
    return rotBtoGP.T.dot(np.array([x, y, z]))


def getRotationOffsetMatrix(roll0, pitch0, yaw0):
    cps0 = np.cos(yaw0)
    sps0 = np.sin(yaw0)
    cph0 = np.cos(roll0)
    sph0 = np.sin(roll0)
    cth0 = np.cos(pitch0)
    sth0 = np.sin(pitch0)

    Delta1 = cph0 * cps0 + sph0 * sth0 * sps0
    Delta2 = -cph0 * sps0 + sph0 * sth0 * cps0
    Delta3 = -sph0 * cth0
    Delta4 = cth0 * sps0
    Delta5 = cth0 * cps0
    Delta6 = sth0
    Delta7 = sph0 * cps0 - cph0 * sth0 * sps0
    Delta8 = -sph0 * sps0 - cph0 * sth0 * cps0
    Delta9 = cph0 * cth0

    return np.array([[-Delta1, -Delta2, -Delta3], [Delta4, Delta5, Delta6], [-Delta7, -Delta8, -Delta9]])


def getBoresightVector(ROffset, pan, tilt, yaw, pitch, roll):
    """Returns the a 3x1 numpy array with the normalized boresight vector in 
    the inertial frame"""
    # set the boresight pointing vector in the pointed gimbal frame
    delta_gp = np.array([[0], [0], [1.0]])
    # rotate these into the body frame
    delta_b = gimbalToBody(
        ROffset, pan, tilt, *delta_gp)
    # return the boresight in the inertial frame
    return bodyToInertial(yaw, pitch, roll, *delta_b)


def getPhaseCenterInertialCorrection(rotBtoMG, pan, tilt, yaw, pitch, roll, ant_offset, gimbal_offset):
    """Returns the 3x1 numpy array with the inertial translational
    correction to be applied to the INS position value to have the position
    of the antenna phase center."""

    # rotate the antenna offsets (provided in the gimbal frame) into the
    #   the body frame of the INS
    antDelta_b = gimbalToBody(rotBtoMG, pan, tilt, *ant_offset)
    # add the antenna offset in the body frame to the offsets measured from
    #   the INS center to the gimbal axes center of rotation
    totDelta_b = antDelta_b + gimbal_offset
    # return the inertial correction
    return bodyToInertial(yaw, pitch, roll, totDelta_b[0], totDelta_b[1], totDelta_b[2])

