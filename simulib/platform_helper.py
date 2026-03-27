import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from typing import Type
from .simulation_functions import llh2enu, azelToVec
from .rotation_functions import (apply_lever_arm_corrections, get_aesa_rotation_offset_matrix,
                                 get_rotation_offset_matrix, get_aesa_boresight_vector, get_boresight_vector)
from .utils import GRAVITIC_CONSTANT, c0, TAC, DTR, INS_REFRESH_HZ
from itertools import product
SDRBase = Type
SARParse = Type


class DynamicModel(object):
    _pos = None
    _vel = None
    _acc = None
    _att = None
    _att_vel = None
    _att_acc = None
    _txpos = None
    _rxpos = None
    _tx_offset = None
    _rx_offset = None
    gimbal_az = None
    gimbal_el = None
    _gimbal_rot_offset = None
    _gimbal_offset = None
    _az_iner = None
    _el_iner = None
    _t = None
    _boresight = None
    _n_channels = 0
    _history: dict

    def __init__(self, init_t, init_pos, init_att, gimbal_az, gimbal_el, gimbal_rotations, gimbal_offset: np.ndarray = None,
                 tx_offset: np.ndarray = None, rx_offset: np.ndarray = None, init_vel: np.ndarray = None,
                 init_acc: np.ndarray = None, init_att_vel: np.ndarray = None, init_att_acc: np.ndarray = None,
                 buffer_size: int = 16384):
        self._tell = 0
        self._buffer = buffer_size * 2
        # Initialize all the buffers for various parameters
        self._pos = np.zeros((self._buffer, 3))
        self._vel = np.zeros_like(self._pos)
        self._acc = np.zeros_like(self._pos)
        self._att = np.zeros_like(self._pos)
        self._att_vel = np.zeros_like(self._pos)
        self._att_acc = np.zeros_like(self._pos)
        self._boresight = np.zeros_like(self._pos)

        self._t = np.zeros(self._buffer)
        self.gimbal_el = np.zeros_like(self._t)
        self.gimbal_az = np.zeros_like(self._t)
        self._az_iner = np.zeros_like(self._t)
        self._el_iner = np.zeros_like(self._t)

        self._tx_offset = tx_offset if tx_offset is not None else np.array([[0., 0., 0.]])
        self._rx_offset = rx_offset if rx_offset is not None else np.array([[0., 0., 0.]])

        self._txpos = np.zeros((self._tx_offset.shape[0], self._buffer, 3))
        self._rxpos = np.zeros((self._rx_offset.shape[0], self._buffer, 3))

        self._pos[0] = init_pos
        self._vel[0] = init_vel
        self._acc[0] = init_acc
        self._att[0] = init_att
        self._att_vel[0] = init_att_vel
        self._att_acc[0] = init_att_acc
        self._t[0] = init_t

        # Take into account the gimbal/AESA rotations
        self._gimbal_rot_offset = getRotationOffsetMatrix(*gimbal_rotations)
        self._gimbal_offset = gimbal_offset
        self.gimbal_az[0] = gimbal_az
        self.gimbal_el[0] = gimbal_el

        self._history = {}


        # Get rx/tx channel pairs
        self.channel_pairs = list(product(self._rx_offset, self._tx_offset))
        self._n_channels = len(self.channel_pairs)
        self._txpos[:, 0] = np.stack([init_pos + getPhaseCenterInertialCorrection(self._gimbal_rot_offset, gimbal_az, gimbal_el, init_att[2],
                                                 init_att[1], init_att[0], tx, gimbal_offset) for tx in self._tx_offset])
        self._rxpos[:, 0] = np.stack([init_pos + getPhaseCenterInertialCorrection(self._gimbal_rot_offset, gimbal_az, gimbal_el, init_att[2],
                                                 init_att[1], init_att[0], rx, gimbal_offset) for rx in self._rx_offset])


        # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
        '''corrs = getPhaseCenterInertialCorrection(self._gimbal_rot_offset, gimbal_az, gimbal_el, init_att[2],
                                                 init_att[1], init_att[0], self._tx_offset, gimbal_offset)
        self._txpos = (init_pos + corrs).reshape((1, 3))'''

        # Rotate antenna into inertial frame in the same way as above
        self._boresight[0] = getBoresightVector(self._gimbal_rot_offset, gimbal_az, gimbal_el, init_att[2], init_att[1], init_att[0]).T

        # Calculate antenna azimuth/elevation for beampattern
        self._el_iner[0] = np.arcsin(-self._boresight[0, 2])
        self._az_iner[0] = np.arctan2(self._boresight[0, 0], self._boresight[0, 1])

    def update(self, t, new_acc, new_att_acc, gim_az, gim_el):
        if self.next >= self._buffer:
            self._flush()
        self.gimbal_el[self.next] = gim_el
        self.gimbal_az[self.next] = gim_az
        self.updatePos(t, new_acc)
        self.updateAtt(t, new_att_acc)
        self.updatePhaseCenter(1 if isinstance(t, float) else len(t))
        self._t[self.next] = t
        self._tell += 1

    def updatePos(self, t, new_acc):
        if isinstance(t, float):
            delta_t = t - self._t[self._tell]
            self._acc[self.next] = new_acc
            self._vel[self.next] = (self._vel[self._tell] + new_acc * delta_t).reshape((1, 3))
            self._pos[self.next] = (self._pos[self._tell] + self._vel[self._tell] * delta_t).reshape((1, 3))
        else:
            n_ts = len(t)
            delta_t = np.array([t[0] - self._t[self._tell], *np.diff(t)])
            acc = np.zeros((len(t), 3))
            acc[:new_acc.shape[0]] = new_acc
            self._acc = np.concatenate((self._acc, acc), axis=0)
            self._vel = np.concatenate((self._vel, (self._vel[self.prev(n_ts)] + acc * delta_t).reshape((1, 3))), axis=0)
            self._pos = np.concatenate((self._pos, (self._pos[self.prev(n_ts)] + self._vel[self.prev(n_ts)] * delta_t).reshape((1, 3))),
                                       axis=0)

    def updateAtt(self, t, new_att):
        if isinstance(t, float):
            delta_t = t - self._t[self._tell]
            self._att_acc[self.next] = new_att
            self._att_vel[self.next] = (self._att_vel[self._tell] + new_att * delta_t).reshape((1, 3))
            self._att[self.next] = (self._att[self._tell] + self._att_vel[self._tell] * delta_t).reshape((1, 3))
        else:
            n_ts = len(t)
            delta_t = np.array([t[0] - self._t[self._tell], *np.diff(t)])
            att = np.zeros((len(t), 3))
            att[:new_att.shape[0]] = new_att
            self._att_acc = np.concatenate((self._att_acc, att), axis=0)
            self._att_vel = np.concatenate((self._att_vel, (self._att_vel[self.prev(n_ts)] + att * delta_t).reshape((1, 3))),
                                           axis=0)
            self._att = np.concatenate(
                (self._att, (self._att[self.prev(n_ts)] + self._att_vel[self.prev(n_ts)] * delta_t).reshape((1, 3))),
                axis=0)

    def updatePhaseCenter(self, n_ts):
        tx_update = np.zeros((self._tx_offset.shape[0], 1, 3))
        rx_update = np.zeros((self._rx_offset.shape[0], 1, 3))
        for idx, txo in enumerate(self._tx_offset):
            phase_corrections = [getPhaseCenterInertialCorrection(self._gimbal_rot_offset, ga, ge, a[2], a[1], a[0],
                                                                  txo, self._gimbal_offset)
                                 for ga, ge, a in zip(self.gimbal_az[self.prev(n_ts)], self.gimbal_el[self.prev(n_ts)], self._att[self.prev(n_ts)])]
            tx_update[idx] = self._pos[self.prev(n_ts)] + np.array(phase_corrections)
        for idx, rxo in enumerate(self._rx_offset):
            phase_corrections = [getPhaseCenterInertialCorrection(self._gimbal_rot_offset, ga, ge, a[2], a[1], a[0],
                                                                  rxo, self._gimbal_offset)
                                 for ga, ge, a in
                                 zip(self.gimbal_az[self.prev(n_ts)], self.gimbal_el[self.prev(n_ts)], self._att[self.prev(n_ts)])]
            rx_update[idx] = self._pos[self.prev(n_ts)] + np.array(phase_corrections)
        self._txpos[:, self.next] = tx_update
        self._rxpos[:, self.next] = np.squeeze(rx_update, 1)
        bai = [getBoresightVector(self._gimbal_rot_offset, ga, ge, a[2], a[1], a[0])
               for ga, ge, a in zip(self.gimbal_az[self.prev(n_ts)], self.gimbal_el[self.prev(n_ts)], self._att[self.prev(n_ts)])][0]
        self._boresight[self.next] = np.array(bai).T

        # Calculate antenna azimuth/elevation for beampattern
        self._el_iner[self.next] = np.arcsin(-self._boresight[self.next, 2])
        self._az_iner[self.next] = np.arctan2(self._boresight[self.next, 0], self._boresight[self.next, 1])

    def getGimbalUpdatesFromBoresight(self, inertial_boresight):
        return getAzElGimbalFromDesiredBoresight(inertial_boresight, self._gimbal_rot_offset, self.gimbal_az[self._tell],
                                                 self.gimbal_el[self._tell], self._att[self._tell, 2],
                                                 self._att[self._tell, 1], self._att[self._tell, 0])

    def turnToHeading(self, t, tvec, des_bore=None):
        if self.next >= self._buffer:
            self._flush()
        omega_max = GRAVITIC_CONSTANT * np.sqrt((1 / np.cos(np.pi / 8) ** 2 - 1)) / np.linalg.norm(self._vel[self._tell])
        delta_t = t - self._t[self._tell]
        uvec = tvec / np.linalg.norm(tvec)
        turn_axis = np.cross(self._vel[self._tell], uvec)
        turn_angle = np.clip(np.arctan2(np.linalg.norm(np.cross(self._vel[self._tell], uvec)), np.dot(self._vel[self._tell], uvec)),
                             -omega_max * delta_t, omega_max * delta_t)

        skew_sym_cross = np.array([[0, -turn_axis[2], turn_axis[1]],
                                   [turn_axis[2], 0, -turn_axis[0]],
                                   [-turn_axis[1], turn_axis[0], 0]])
        rotation_matrix = np.eye(3) + np.sin(turn_angle) * skew_sym_cross + np.dot(skew_sym_cross, skew_sym_cross) * (
                1 - np.cos(turn_angle))
        self._vel[self.next] = np.dot(rotation_matrix, self._vel[self._tell]).reshape((1, 3))
        self._pos[self.next] = (self._vel[self._tell] * delta_t + self._pos[self._tell]).reshape((1, 3))
        self._acc[self.next] = np.zeros((1, 3))

        # Calculate attitude
        phi_roll = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
        alpha_yaw = np.arctan2(self._vel[self._tell, 0], self._vel[self._tell, 1])
        theta_pitch = -np.arcsin(self._vel[self._tell, 2] / np.linalg.norm(self._vel[self._tell]))
        self._att[self.next] = np.array([[phi_roll, theta_pitch, alpha_yaw]])
        self._att_vel[self.next] = (self._att[self._tell] * delta_t).reshape((1, 3))
        self._att_acc[self.next] = (self._att_vel[self._tell] * delta_t).reshape((1, 3))
        gim_az, gim_el = self.getGimbalUpdatesFromBoresight(des_bore) if des_bore is not None else \
            (self.gimbal_az[self._tell], self.gimbal_el[self._tell])
        self.gimbal_el[self.next] = gim_el
        self.gimbal_az[self.next] = gim_az
        self.updatePhaseCenter(1)
        self._t[self.next] = t
        self._tell += 1

    def _flush(self):
        # Save history just in case
        if 'pos' not in self._history.keys():
            self._history['pos'] = self._pos + 0.0
            self._history['vel'] = self._vel + 0.0
            self._history['txpos'] = self._txpos + 0.0
            self._history['rxpos'] = self._rxpos + 0.0
            self._history['boresight'] = self._boresight + 0.0
            self._history['az'] = self._az_iner + 0.0
            self._history['el'] = self._el_iner + 0.0
        else:
            self._history['pos'] = np.concatenate((self._history['pos'], self._pos), axis=0)
            self._history['vel'] = np.concatenate((self._history['vel'], self._vel), axis=0)
            self._history['txpos'] = np.concatenate((self._history['txpos'], self._txpos), axis=1)
            self._history['rxpos'] = np.concatenate((self._history['rxpos'], self._rxpos), axis=1)
            self._history['boresight'] = np.concatenate((self._history['boresight'], self._boresight), axis=0)
            self._history['az'] = np.concatenate((self._history['az'], self._az_iner), axis=0)
            self._history['el'] = np.concatenate((self._history['el'], self._el_iner), axis=0)

        # shift everything over to make more room
        self._pos[:self._buffer // 2] = self._pos[self._buffer // 2:]
        self._vel[:self._buffer // 2] = self._vel[self._buffer // 2:]
        self._acc[:self._buffer // 2] = self._acc[self._buffer // 2:]
        self._att[:self._buffer // 2] = self._att[self._buffer // 2:]
        self._att_vel[:self._buffer // 2] = self._att_vel[self._buffer // 2:]
        self._att_acc[:self._buffer // 2] = self._att_acc[self._buffer // 2:]
        self._boresight[:self._buffer // 2] = self._boresight[self._buffer // 2:]
        self._t[:self._buffer // 2] = self._t[self._buffer // 2:]
        self.gimbal_az[:self._buffer // 2] = self.gimbal_az[self._buffer // 2:]
        self.gimbal_el[:self._buffer // 2] = self.gimbal_el[self._buffer // 2:]
        self._az_iner[:self._buffer // 2] = self._az_iner[self._buffer // 2:]
        self._el_iner[:self._buffer // 2] = self._el_iner[self._buffer // 2:]
        self._txpos[:, :self._buffer // 2] = self._txpos[:, self._buffer // 2:]
        self._rxpos[:, :self._buffer // 2] = self._rxpos[:, self._buffer // 2:]
        self._tell = self._buffer // 2 - 1

    def getHistory(self):
        h = self._history.copy()
        if 'pos' not in h.keys():
            h['pos'] = self._pos[:self._tell]
            h['vel'] = self._vel[:self._tell]
            h['txpos'] = self._txpos[:self._tell]
            h['rxpos'] = self._rxpos[:self._tell]
            h['boresight'] = self._boresight[:self._tell]
            h['az'] = self._az_iner[:self._tell]
            h['el'] = self._el_iner[:self._tell]
        else:
            h['pos'] = np.concatenate((h['pos'], self._pos[:self._tell]), axis=0)
            h['vel'] = np.concatenate((h['vel'], self._vel[:self._tell]), axis=0)
            h['txpos'] = np.concatenate((h['txpos'], self._txpos[:self._tell]), axis=1)
            h['rxpos'] = np.concatenate((h['rxpos'], self._rxpos[:self._tell]), axis=1)
            h['boresight'] = np.concatenate((h['boresight'], self._boresight[:self._tell]), axis=0)
            h['az'] = np.concatenate((h['az'], self._az_iner[:self._tell]), axis=0)
            h['el'] = np.concatenate((h['el'], self._el_iner[:self._tell]), axis=0)
        return h


    def prev(self, n: int = 1):
        return slice(1) if self._tell == 0 else slice(max(0, self._tell - n), self._tell)

    @property
    def boresight(self):
        return self._boresight

    @property
    def pos(self):
        return self._pos

    @property
    def txpos(self):
        return self._txpos

    @property
    def rxpos(self):
        return self._rxpos

    @property
    def az_iner(self):
        return self._az_iner

    @property
    def el_iner(self):
        return self._el_iner

    @property
    def vel(self):
        return self._vel

    @property
    def att(self):
        return self._att

    @property
    def t(self):
        return self._t

    @property
    def tell(self):
        return self._tell

    @property
    def next(self):
        return self._tell + 1

    @property
    def txoffset(self):
        return self._tx_offset

    @property
    def rxoffset(self):
        return self._rx_offset

    @property
    def gimbaloffset(self):
        return self._gimbal_offset

    @property
    def gimbalrotation(self):
        return self._gimbal_rot_offset






"""
Platform class

Base class used to hold information on radar platform, such as velocity, position, attitude, and gimbal corrections.
This is expected to provide positional information for only one pair of Tx and Rx antennas.
"""


class Platform(object):
    _pos = None
    _vel = None
    _att = None
    _heading = None
    _gimbal_offset_mat = None

    def __init__(self,
                 e: np.ndarray = None,
                 n: np.ndarray = None,
                 u: np.ndarray = None,
                 r: np.ndarray = None,
                 p: np.ndarray = None,
                 y: np.ndarray = None,
                 t: np.ndarray = None,
                 gimbal: np.ndarray = None,
                 gimbal_offset: np.ndarray = None,
                 gimbal_rotations: np.ndarray = None,
                 tx_offset: np.ndarray = None,
                 rx_offset: np.ndarray = None,
                 gps_t: np.ndarray = None,
                 gps_az: np.ndarray = None,
                 aesa: np.ndarray = None,
                 *args,
                 **kwargs):
        """
        Initializes a Platform object with position, velocity, attitude, and gimbal corrections.

        This constructor sets up splines for position, velocity, and attitude, and applies gimbal and antenna corrections if provided.

        Args:
            e: np.ndarray. East coordinates.
            n: np.ndarray. North coordinates.
            u: np.ndarray. Up coordinates.
            r: np.ndarray. Roll angle.
            p: np.ndarray. Pitch angle.
            y: np.ndarray. Yaw angle.
            t: np.ndarray. Time array for each of the [e, n, u] and [r, p, y] arrays. Should be the same length as them.
            gimbal: np.ndarray. Gimbal data.
            gimbal_offset: np.ndarray. Gimbal offset.
            gimbal_rotations: np.ndarray. Gimbal rotations.
            tx_offset: np.ndarray. Transmitter offset.
            rx_offset: np.ndarray. Receiver offset.
            gps_t: np.ndarray. GPS time. Used for SlimSDR collects.
            gps_az: np.ndarray. GPS azimuth. Used for SlimSDR collects.
            gps_rxpos: np.ndarray. GPS receiver position. Used for SlimSDR collects.
            gps_txpos: np.ndarray. GPS transmitter position. Used for SlimSDR collects.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        self._gpst = t
        self._gimbal = gimbal
        self._gimbal_offset = gimbal_offset
        self.gimbal_rotations = gimbal_rotations

        # The yaw is calculated separately here if necessary
        position_matrix = np.stack([e, n, u, r, p, y, t])
        position_matrix[5] = np.interp(t, gps_t, gps_az + 2 * np.pi) if gps_t is not None else y

        self.tx = AntennaPosition(position_matrix, gimbal, gimbal_offset, gimbal_rotations, tx_offset, aesa)
        self.rx = AntennaPosition(position_matrix, gimbal, gimbal_offset, gimbal_rotations, rx_offset, aesa)

        self._att = self.tx.att

        # Take into account the gimbal if necessary
        if gimbal is not None:
            self._gimbal_offset_mat = getRotationOffsetMatrix(*gimbal_rotations)
            self._gimbal_pan = CubicSpline(t, gimbal[:, 0])
            self._gimbal_tilt = CubicSpline(t, gimbal[:, 1])

        # Build the position splines
        self._pos = CubicSpline(t, position_matrix[:3].T)

        # Build a velocity spline
        self._vel = CubicSpline(t, median_filter(np.gradient(position_matrix[:3].T, axis=0), 15, axes=(0,)) *
                                INS_REFRESH_HZ)

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[:, 0], self._vel(lam_t)[:, 1])

    @property
    def pos(self):
        # Return the position lambda
        return self._pos

    @property
    def heading(self):
        # Return the heading lambda
        return self._heading

    @property
    def att(self):
        # Return the attitude lambda
        return self._att

    @property
    def gpst(self):
        # Return the array of GPS times used to create lambdas
        return self._gpst

    @property
    def rxpos(self):
        # Return Rx position lambda
        return self.rx.pos

    @property
    def txpos(self):
        # Return Tx position lambda
        return self.tx.pos

    @property
    def vel(self):
        # Return velocity lambda
        return self._vel

    @property
    def gimbal_offset(self):
        return self._gimbal_offset

    @property
    def az_iner(self):
        return self.tx.az_iner

    @property
    def el_iner(self):
        return self.tx.el_iner


"""
    RadarPlatform
    A class representing a Radar Platform that inherits from Platform. This is used for creating a platform
    object that does not use a SlimSDR collect as a base.

    Args:
        e: np.ndarray. East coordinate.
        n: np.ndarray. North coordinate.
        u: np.ndarray. Up coordinate.
        r: np.ndarray. Right coordinate.
        p: np.ndarray. Pitch angle.
        y: np.ndarray. Yaw angle.
        t: np.ndarray. Time.
        tx_offset: np.ndarray. Transmitter offset.
        rx_offset: np.ndarray. Receiver offset.
        gimbal: np.ndarray. (Nx2) Gimbal array of pan/tilt values.
        gimbal_offset: np.ndarray. Gimbal offset.
        gimbal_rotations: np.ndarray. Gimbal rotations.
        dep_angle: float. Depression angle (default: 45.).
        squint_angle: float. Squint angle (default: 0.).
        az_bw: float. Azimuth beamwidth (default: 10.).
        el_bw: float. Elevation beamwidth (default: 10.).
        fs: float. Sampling frequency.
        gps_t: np.ndarray. GPS time.
        gps_az: np.ndarray. GPS azimuth.
        gps_rxpos: np.ndarray. GPS receiver position.
        gps_txpos: np.ndarray. GPS transmitter position.
        tx_num: int. Transmitter number (default: 0).
        rx_num: int. Receiver number (default: 0).
        wavenumber: int. Wavenumber (default: 0).

    Attributes:
        dep_ang: float. Depression angle in radians.
        squint_ang: float. Squint angle in radians.
        az_half_bw: float. Half of azimuth beamwidth in radians.
        el_half_bw: float. Half of elevation beamwidth in radians.
        fs: float. Frequency.
        near_range_angle: float. Near range angle in degrees.
        far_range_angle: float. Far range angle in degrees.
        rx_num: int. Receiver number.
        tx_num: int. Transmitter number.
        wavenumber: int. Wavenumber.
        pulse: ndarray. Array set aside for a pulse.
        mfilt: ndarray. Array set aside for a matched filter.
        fc: float. Center frequency in Hz.
        bwidth: float. Bandwidth of pulse in Hz.

    Methods:
        calcRanges: Calculates near and far slant ranges.
        calcPulseLength: Calculates pulse length.
        calcNumSamples: Calculates number of samples in a pulse.
        calcRangeBins: Calculates range bins for a given pulse.
        getRadarParams: Gets relevant radar parameters.

    Examples:
        radar = RadarPlatform()
        radar.calcRanges(100)
"""


class RadarPlatform(Platform):
    pulse: np.ndarray
    mfilt: np.ndarray
    fc: float
    bwidth: float

    def __init__(self,
                 e: np.ndarray = None,
                 n: np.ndarray = None,
                 u: np.ndarray = None,
                 r: np.ndarray = None,
                 p: np.ndarray = None,
                 y: np.ndarray = None,
                 t: np.ndarray = None,
                 tx_offset: np.ndarray = None,
                 rx_offset: np.ndarray = None,
                 gimbal: np.ndarray = None,
                 gimbal_offset: np.ndarray = None,
                 gimbal_rotations: np.ndarray = None,
                 dep_angle: float = 45.,
                 squint_angle: float = 0.,
                 az_bw: float = 10.,
                 el_bw: float = 10.,
                 fs: float = 2e9,
                 fc: float = 9.6e9,
                 prf: float = 1200.,
                 gps_t: np.ndarray = None,
                 gps_az: np.ndarray = None,
                 tx_num: int = 0,
                 rx_num: int = 0,
                 wavenumber: int = 0,
                 bwidth: float = 0.,
                 *args,
                 **kwargs):

        """
        Initializes a Radar Platform object with provided parameters.

        Args:
            e: np.ndarray. East coordinates.
            n: np.ndarray. North coordinates.
            u: np.ndarray. Up coordinates.
            r: np.ndarray. Roll angle.
            p: np.ndarray. Pitch angle.
            y: np.ndarray. Yaw angle.
            t: np.ndarray. Time.
            tx_offset: np.ndarray. Transmitter offset.
            rx_offset: np.ndarray. Receiver offset.
            gimbal: np.ndarray. Gimbal.
            gimbal_offset: np.ndarray. Gimbal offset.
            gimbal_rotations: np.ndarray. Gimbal rotations.
            dep_angle: float. Depression angle (default: 45.).
            squint_angle: float. Squint angle (default: 0.).
            az_bw: float. Azimuth beamwidth (default: 10.).
            el_bw: float. Elevation beamwidth (default: 10.).
            fs: float. Frequency.
            gps_t: np.ndarray. GPS time.
            gps_az: np.ndarray. GPS azimuth.
            tx_num: int. Transmitter number (default: 0).
            rx_num: int. Receiver number (default: 0).
            wavenumber: int. Wavenumber (default: 0).
        """

        super().__init__(e, n, u, r, p, y, t, gimbal, np.array(gimbal_offset), np.array(gimbal_rotations),
                         tx_offset, rx_offset, gps_t, gps_az, **kwargs)
        self.dep_ang = dep_angle * DTR
        self.squint_ang = squint_angle * DTR
        self.az_half_bw = az_bw * DTR / 2
        self.el_half_bw = el_bw * DTR / 2
        self.fs = fs
        self.prf = prf
        self.fc = fc
        self.bwidth = bwidth
        self.near_range_angle = self.dep_ang + self.el_half_bw
        self.far_range_angle = self.dep_ang - self.el_half_bw
        self.rx_num = rx_num
        self.tx_num = tx_num
        self.wavenumber = wavenumber

    def calcRanges(self, height, exp_factor=1, **kwargs):
        """
        Calculates out the expected near and far slant ranges of the antenna.
        :param exp_factor: float. Expansion factor for range.
        :param height: float. Height of antenna off the ground in meters.
        :return: tuple of near and far slant ranges in meters.
        """
        nrange = height / np.sin(self.near_range_angle * exp_factor)
        frange = height / np.sin(self.far_range_angle * exp_factor)
        return nrange, frange

    def calcPulseLength(self, height, pulse_length_percent=1., use_tac=False, nrange=None, **kwargs):
        """
        Calculates the pulse length given a height off the ground and a pulse percent.
        :param nrange: float. Near range, for override, if desired.
        :param height: float. Height of antenna off the ground in meters.
        :param pulse_length_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :param use_tac: bool. If True, returns the pulse length in TAC. If False, returns in seconds.
        :return: Expected pulse length in either TAC or seconds.
        """
        if nrange is None:
            nrange, _ = self.calcRanges(height)
        plength_s = (nrange * 2 / c0) * pulse_length_percent
        return int(np.ceil(plength_s * self.fs)) if use_tac else plength_s

    def calcNumSamples(self, height, plp=1., ranges=None, **kwargs):
        """
        Calculate the number of samples in a pulse.
        :param ranges: 2-tuple. (Near range, Far range) for range override if desired.
        :param height: float. Height of antenna off the ground in meters.
        :param plp: float, <1. Percentage of maximum pulse length to use in radar.
        :return: Number of samples in a returned pulse.
        """
        if ranges is None:
            nrange, frange = self.calcRanges(height)
            return int((np.ceil((2 * frange / c0 + self.calcPulseLength(height, plp, nrange=nrange)) * TAC) - np.floor(
                2 * nrange / c0 * TAC)) * self.fs / TAC)
        else:
            nrange = ranges[0]
            frange = ranges[1]
            return int((np.ceil((2 * frange / c0) * TAC) - np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)

    def calcRangeBins(self, height, upsample=1, plp=1., ranges=None, **kwargs):
        """
        Calculates the range bins for a given pulse.
        :param ranges: 2-tuple. (Near range, Far range) for range override if desired.
        :param height: float. Height of antenna off the ground in meters.
        :param upsample: int. Upsample factor.
        :param plp: float, <1. Percentage of maximum pulse length to use in radar.
        :return: array of range bins.
        """
        if ranges is None:
            nrange, frange = self.calcRanges(height)
        else:
            nrange = ranges[0]
        MPP = c0 / self.fs / upsample / 2
        return nrange + np.arange(self.calcNumSamples(height, plp, ranges) * upsample) * MPP + c0 / self.fs

    def getRadarParams(self, fdelay, plp, upsample=1, a_ranges=None):
        """
        A function to get many relevant radar parameters gathered in one spot.

        Args:
            fdelay: The fdelay value.
            plp: The plp value.
            upsample: The upsample value (default: 1).

        Returns:
            nsam: The calculated number of samples.
            nr: The calculated pulse length.
            ranges: The calculated range bins.
            ranges_sampled: The calculated range bins with upsample value of 1.
            near_range_s: The calculated near range in seconds.
            granges: The calculated range bins multiplied by the cosine of the dep_ang.
            fft_len: The calculated FFT length.
            up_fft_len: The calculated upsampled FFT length.
        """
        nsam = self.calcNumSamples(fdelay, plp, a_ranges)
        nr = self.calcPulseLength(fdelay, plp, True, a_ranges[0] if a_ranges is not None else None)
        ranges = self.calcRangeBins(fdelay, upsample, plp, a_ranges)
        ranges_sampled = self.calcRangeBins(fdelay, 1, plp, a_ranges)
        near_range_s = ranges[0] / c0
        granges = ranges * np.cos(self.dep_ang)
        fft_len = int(2 ** (np.ceil(np.log2(nsam + self.calcPulseLength(fdelay, plp, use_tac=True, nrange=a_ranges[0] if a_ranges is not None else None)))))
        up_fft_len = fft_len * upsample
        return nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len

    def calcPRF(self, vel, prf_broadening: float = 1.5):
        return prf_broadening * 2 * vel * np.sin(self.az_half_bw) * (self.fc + self.bwidth / 2) / c0

    def calcRadVelRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the radial velocity resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Radial velocity resolution in meters per second.
        """
        return dopplerBroadeningFactor * c0 * self.prf / (self.fc * cpi_len)

    def calcDopRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the Doppler resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Doppler resolution in Hz.
        """
        return dopplerBroadeningFactor * self.prf / cpi_len

    def calcWrapVel(self):
        """
        Calculate the wrap velocity for this collect.
        :return: Wrap velocity in meters per second.
        """
        return self.prf * (c0 / self.fc) / 4.0


"""
SDRPlatform
More specialized version of RadarPlatform that gets all of the values it needs from a SAR file or instance of
SDRParse.
"""

class SDRPlatform(RadarPlatform):
    _sdr = None

    def __init__(self, sdr: SDRBase,
                 origin: np.ndarray = None,
                 tx_offset: np.ndarray = None,
                 rx_offset: np.ndarray = None,
                 fs: float = 2e9,
                 channel: int = 0,
                 gimbal_offset: np.ndarray = None):
        """
        Init function. Inherits RadarPlatform to allow for storing radar parameters such as center frequency.
        This is a Platform object specifically built for SlimSDR collects. Represents a single channel of data.
        :param sdr: SDRParse object or str. This is path to the SAR file used as a basis for other calculations,
            or the SDRParse object of an already parsed file.
        :param origin: 3-tuple. Point used as the origin for ENU reference frame, in (lat, lon, alt).
        :param tx_offset: 3-tuple. Offset of Tx antenna from body frame in meters. Only used if a different offset is wanted than
        the one in the SlimSDR .xml file.
        :param rx_offset: 3-tuple. Offset of Rx antenna from body frame in meters. Only used if a different offset is wanted than
        the one in the SlimSDR .xml file.
        :param fs: float. Sampling frequency in Hz.
        :param channel: int. Channel of data for this object to represent in the SAR file.
        """

        # Get times, sampling frequency, and set an origin for the local tangent plane
        t = sdr.gps_data.index.values
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[0, :])

        # Load GPS values for location and attitude
        e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
        r = sdr.gps_data['r'].values
        p = sdr.gps_data['p'].values
        y = sdr.gps_data['y'].values

        # Get gimbal values, if any
        if sdr.has_gimbal:
            pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                            sdr.gimbal['pan'].values.astype(np.float64))
            tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                             sdr.gimbal['tilt'].values.astype(np.float64))
        else:
            pan = np.zeros_like(sdr.gps_data['systime'].values)
            tilt = np.zeros_like(sdr.gps_data['systime'].values)
        pan = np.interp(t, sdr.gps_data.index.values, pan)
        tilt = np.interp(t, sdr.gps_data.index.values, tilt)
        goff = np.array(
            [sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset]) if gimbal_offset is None else gimbal_offset
        grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR])
        try:
            channel_dep = (sdr.xml.Channel_0.Near_Range_D + sdr.xml.Channel_0.Far_Range_D) / 2
        except AttributeError:
            channel_dep = (sdr.xml.Interval_0.Near_Range_D + sdr.xml.Interval_0.Far_Range_D) / 2
        if sdr.intervals[channel].is_receive_only:
            tx_num = np.where([n is not None for n in sdr.port])[0][0]
        else:
            tx_num = sdr[channel].trans_num if not sdr.is_v2 else sdr[channel].tx_num
            tx_offset = np.array(
                [[[sdr.port[tx_num].x, sdr.port[tx_num].y, sdr.port[tx_num].z]]]) if tx_offset is None else tx_offset
        rx_num = sdr[channel].rec_num if not sdr.is_v2 else sdr[channel].rx_num
        rx_offset = np.array(
            [[[sdr.port[rx_num].x, sdr.port[rx_num].y, sdr.port[rx_num].z]]]) if rx_offset is None else rx_offset
        try:
            aesa_vals = sdr.aesa.values
            aesa_vals[:, 0] = np.interp(aesa_vals[:, 0], sdr.gps_data['systime'].values, sdr.gps_data.index.values)
            aesa = np.zeros((len(t), 2))
            aesa[:, 0] = np.interp(t, aesa_vals[:, 0], aesa_vals[:, 1]) * DTR
            aesa[:, 1] = np.interp(t, aesa_vals[:, 0], aesa_vals[:, 2]) * DTR

        except AttributeError:
            aesa = None
        super().__init__(e=e, n=n, u=u, r=r, p=p, y=y, t=t, tx_offset=tx_offset, rx_offset=rx_offset,
                         gimbal=np.array([pan, tilt]).T, gimbal_offset=goff, gimbal_rotations=grot,
                         dep_angle=channel_dep, squint_angle=sdr.ant[sdr.port[tx_num].assoc_ant].squint / DTR,
                         az_bw=sdr.ant[sdr.port[tx_num].assoc_ant].az_bw / DTR,
                         el_bw=sdr.ant[sdr.port[tx_num].assoc_ant].el_bw / DTR, fs=fs, tx_num=tx_num,
                         rx_num=rx_num, aesa=aesa)
        self._sdr = sdr
        self.origin = origin
        self._channel = channel

    def getRadarParams(self, fdelay, plp, upsample=1, a_ranges=None):
        """
        A function to get many relevant radar parameters gathered in one spot.

        Args:
            fdelay: The fdelay value.
            plp: The plp value.
            upsample: The upsample value (default: 1).

        Returns:
            nsam: The calculated number of samples.
            nr: The calculated pulse length.
            ranges: The calculated range bins.
            ranges_sampled: The calculated range bins with upsample value of 1.
            near_range_s: The calculated near range in seconds.
            granges: The calculated range bins multiplied by the cosine of the dep_ang.
            fft_len: The calculated FFT length.
            up_fft_len: The calculated upsampled FFT length.
        """
        nsam = self.calcNumSamples(fdelay, plp, a_ranges)
        nr = self.calcPulseLength(fdelay, plp, True, a_ranges[0] if a_ranges is not None else None)
        ranges = self.calcRangeBins(fdelay, upsample, plp, a_ranges)
        ranges_sampled = self.calcRangeBins(fdelay, 1, plp, a_ranges)
        near_range_s = ranges[0] / c0
        granges = ranges * np.cos(self.dep_ang)
        fft_len = int(2 ** (np.ceil(np.log2(nsam + self.calcPulseLength(fdelay, plp, use_tac=True, nrange=a_ranges[0] if a_ranges is not None else None)))))
        up_fft_len = fft_len * upsample
        return nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len

    def calcRanges(self, fdelay, partial_pulse_percent=1., **kwargs):
        """
        Calculate near and far ranges for this collect using the SAR file.
        :param fdelay: float. FDelay desired in TAC.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: tuple of near and far ranges in meters.
        """
        try:
            nrange = ((self._sdr[0].receive_on_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC) * c0 / 2
            frange = ((self._sdr[0].receive_off_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                      self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        except AttributeError:
            nrange = ((self._sdr[0].Receive_On_TAC - self._sdr[self._channel].Transmit_On_TAC - fdelay) / TAC) * c0 / 2
            frange = ((self._sdr[0].Receive_Off_TAC - self._sdr[self._channel].Transmit_On_TAC - fdelay) / TAC -
                      self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        return nrange, frange

    def calcPulseLength(self, height=0, pulse_length_percent=1., use_tac=False, nrange=None, **kwargs):
        """
        Calculate the pulse length for this collect.
        :param height: Not used. Here for compatibility with parent classes.
        :param pulse_length_percent: Not used. Here for compatibility with parent classes.
        :param use_tac: bool. If True, returns pulse length in TAC, otherwise in seconds.
        :return: Pulse length in TAC, otherwise in seconds.
        """
        return self._sdr[self._channel].pulse_length_N if use_tac else self._sdr[self._channel].pulse_length_S

    def calcNumSamples(self, height=0, plp=1., ranges=None, **kwargs):
        """
        Get number of samples in a pulse.
        :param height: Not used. Here for compatibility with parent classes.
        :param plp: Not used. Here for compatibility with parent classes.
        :return: Number of samples in a pulse.
        """
        return self._sdr[self._channel].nsam

    def calcRangeBins(self, fdelay, upsample=1, plp=1., ranges=None, **kwargs):
        """
        Calculate range bins for a pulse/collect.
        :param fdelay: float. FDelay desired for this pulse in TAC.
        :param upsample: int. Upsample factor.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: array of range bins in meters.
        """
        nrange, frange = self.calcRanges(fdelay, partial_pulse_percent=plp)
        MPP = c0 / self.fs / upsample
        return (nrange * 2 + np.arange(self.calcNumSamples() * upsample) * MPP) / 2

    def calcRadVelRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the radial velocity resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Radial velocity resolution in meters per second.
        """
        return dopplerBroadeningFactor * c0 * self._sdr[self._channel].prf / \
            (self._sdr[self._channel].fc * cpi_len)

    def calcDopRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the Doppler resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Doppler resolution in Hz.
        """
        return dopplerBroadeningFactor * self._sdr[self._channel].prf / cpi_len

    def calcWrapVel(self):
        """
        Calculate the wrap velocity for this collect.
        :return: Wrap velocity in meters per second.
        """
        return self._sdr[self._channel].prf * (c0 / self._sdr[self._channel].fc) / 4.0

    def calcIlluminationVector(self, t):
        gps_times = np.interp(t, self._sdr.gps_data.index.values, self._sdr.gps_data['systime'].values.astype(int))
        phi = np.interp(gps_times, self._sdr.aesa['systime'].values.astype(int), self._sdr.aesa['phi'].values.astype(np.float64))
        theta = np.interp(gps_times, self._sdr.aesa['systime'].values.astype(int), self._sdr.aesa['theta'].values.astype(np.float64))
        return azelToVec(phi, theta)


"""
SARPlatform
More specialized version of RadarPlatform that gets all of the values it needs from a SAR file or instance of
SARParse.
"""

class SARPlatform(RadarPlatform):
    _sdr = None

    def __init__(self, sdr: SARParse,
                 origin: np.ndarray = None,
                 tx_offset: np.ndarray = None,
                 rx_offset: np.ndarray = None,
                 fs: float = 500e6,
                 channel: int = 0,
                 gimbal_offset: np.ndarray = None,
                 gimbal_rotations: np.ndarray = None):
        """
        Init function. Inherits RadarPlatform to allow for storing radar parameters such as center frequency.
        This is a Platform object specifically built for SlimSDR collects. Represents a single channel of data.
        :param sdr: SDRParse object or str. This is path to the SAR file used as a basis for other calculations,
            or the SDRParse object of an already parsed file.
        :param origin: 3-tuple. Point used as the origin for ENU reference frame, in (lat, lon, alt).
        :param tx_offset: 3-tuple. Offset of Tx antenna from body frame in meters. Only used if a different offset is wanted than
        the one in the SlimSDR .xml file.
        :param rx_offset: 3-tuple. Offset of Rx antenna from body frame in meters. Only used if a different offset is wanted than
        the one in the SlimSDR .xml file.
        :param fs: float. Sampling frequency in Hz.
        :param channel: int. Channel of data for this object to represent in the SAR file.
        """

        # Get times, sampling frequency, and set an origin for the local tangent plane
        t = sdr.gps_data.index.values
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[0, :])

        # Load GPS values for location and attitude
        e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
        r = sdr.gps_data['r'].values
        p = sdr.gps_data['p'].values
        y = sdr.gps_data['y'].values

        # Get gimbal values, if any
        if sdr.gimbal is not None:
            pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                            sdr.gimbal['pan'].values.astype(np.float64))
            tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                             sdr.gimbal['tilt'].values.astype(np.float64))
        else:
            pan = np.zeros_like(sdr.gps_data['systime'].values)
            tilt = np.zeros_like(sdr.gps_data['systime'].values)
        pan = np.interp(t, sdr.gps_data.index.values, pan)
        tilt = np.interp(t, sdr.gps_data.index.values, tilt)
        if gimbal_offset is None:
            goff = np.array(
                [sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset]) if 'gim' in sdr.__dict__.keys() else np.zeros(3)
        else:
            goff = gimbal_offset
        if gimbal_rotations is None:
            grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR]) if 'gim' in sdr.__dict__.keys() else np.zeros(3)
        else:
            grot = gimbal_rotations
        try:
            channel_dep = (sdr.xml['Band_1']['Band_1_Near_Range_D'] + sdr.xml['Band_1']['Band_1_Far_Range_D']) / 2
        except AttributeError:
            channel_dep = (sdr.xml['Band_1']['Band_1_Near_Range_D'] + sdr.xml['Band_1']['Band_1_Far_Range_D']) / 2
        if sdr.channels[channel].is_receive_only:
            tx_num = np.where([n is not None for n in sdr.port])[0][0]
        else:
            tx_num = sdr[channel].trans_num
            tx_offset = np.array(
                [[[sdr.port[tx_num].x, sdr.port[tx_num].y, sdr.port[tx_num].z]]]) if tx_offset is None else tx_offset
        rx_num = sdr[channel].rec_num
        rx_offset = np.array(
            [[[sdr.port[rx_num].x, sdr.port[rx_num].y, sdr.port[rx_num].z]]]) if rx_offset is None else rx_offset
        aesa = None
        super().__init__(e=e, n=n, u=u, r=r, p=p, y=y, t=t, tx_offset=tx_offset, rx_offset=rx_offset,
                         gimbal=np.array([pan, tilt]).T, gimbal_offset=goff, gimbal_rotations=grot,
                         dep_angle=channel_dep, squint_angle=sdr.ant[sdr.port[tx_num].assoc_ant].squint / DTR,
                         az_bw=sdr.ant[sdr.port[tx_num].assoc_ant].az_bw / DTR,
                         el_bw=sdr.ant[sdr.port[tx_num].assoc_ant].el_bw / DTR, fs=fs, tx_num=tx_num,
                         rx_num=rx_num, aesa=aesa)
        self._sdr = sdr
        self.origin = origin
        self._channel = channel

    def calcRanges(self, fdelay, partial_pulse_percent=1., **kwargs):
        """
        Calculate near and far ranges for this collect using the SAR file.
        :param fdelay: float. FDelay desired in TAC.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: tuple of near and far ranges in meters.
        """
        try:
            nrange = ((self._sdr[0].receive_on_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC) * c0 / 2
            frange = ((self._sdr[0].receive_off_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                      self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        except AttributeError:
            nrange = ((self._sdr[0].Receive_On_TAC - self._sdr[self._channel].Transmit_On_TAC - fdelay) / TAC) * c0 / 2
            frange = ((self._sdr[0].Receive_Off_TAC - self._sdr[self._channel].Transmit_On_TAC - fdelay) / TAC -
                      self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        return nrange, frange

    def calcPulseLength(self, height=0, pulse_length_percent=1., use_tac=False, nrange=None, **kwargs):
        """
        Calculate the pulse length for this collect.
        :param height: Not used. Here for compatibility with parent classes.
        :param pulse_length_percent: Not used. Here for compatibility with parent classes.
        :param use_tac: bool. If True, returns pulse length in TAC, otherwise in seconds.
        :return: Pulse length in TAC, otherwise in seconds.
        """
        return self._sdr[self._channel].pulse_length_N if use_tac else self._sdr[self._channel].pulse_length_S

    def calcNumSamples(self, height=0, plp=1., ranges=None, **kwargs):
        """
        Get number of samples in a pulse.
        :param height: Not used. Here for compatibility with parent classes.
        :param plp: Not used. Here for compatibility with parent classes.
        :return: Number of samples in a pulse.
        """
        return self._sdr[self._channel].nsam

    def calcRangeBins(self, fdelay, upsample=1, plp=1., ranges=None, **kwargs):
        """
        Calculate range bins for a pulse/collect.
        :param fdelay: float. FDelay desired for this pulse in TAC.
        :param upsample: int. Upsample factor.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: array of range bins in meters.
        """
        nrange, frange = self.calcRanges(fdelay, partial_pulse_percent=plp)
        MPP = c0 / self.fs / upsample
        return (nrange * 2 + np.arange(self.calcNumSamples() * upsample) * MPP) / 2

    def calcRadVelRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the radial velocity resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Radial velocity resolution in meters per second.
        """
        return dopplerBroadeningFactor * c0 * self._sdr[self._channel].prf / \
            (self._sdr[self._channel].fc * cpi_len)

    def calcDopRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the Doppler resolution for this collect.
        :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal value.
        :return: Doppler resolution in Hz.
        """
        return dopplerBroadeningFactor * self._sdr[self._channel].prf / cpi_len

    def calcWrapVel(self):
        """
        Calculate the wrap velocity for this collect.
        :return: Wrap velocity in meters per second.
        """
        return self._sdr[self._channel].prf * (c0 / self._sdr[self._channel].fc) / 4.0

    def calcIlluminationVector(self, t):
        gps_times = np.interp(t, self._sdr.gps_data.index.values, self._sdr.gps_data['systime'].values.astype(int))
        phi = np.interp(gps_times, self._sdr.aesa['systime'].values.astype(int), self._sdr.aesa['phi'].values.astype(np.float64))
        theta = np.interp(gps_times, self._sdr.aesa['systime'].values.astype(int), self._sdr.aesa['theta'].values.astype(np.float64))
        return azelToVec(phi, theta)



class AntennaPosition:
    bw_az: float
    bw_el: float
    gain_db: float
    _pos: object
    _vel: object
    _att: object
    _heading: object
    az_iner: object
    el_iner: object
    az_aesa_iner: object
    el_aesa_iner: object
    aesa_frame_phi_theta: object
    boresight: object

    def __init__(self, pos_mat: np.ndarray, gimbal: np.ndarray, gimbal_offset: np.ndarray, gimbal_rotations: np.ndarray,
                 offset: np.ndarray, aesa: np.ndarray = None):

        pos = pos_mat[:3].T

        # attitude spline
        self._att = CubicSpline(pos_mat[6], pos_mat[3:6].T)
        r, p, y = pos_mat[3:6]
        t = pos_mat[6]

        # Take into account the gimbal
        gim = gimbal if gimbal is not None else np.zeros((len(t), 2))
        self.gimbal_rotation = gimbal_rotations
        self.gimbal_offset = gimbal_offset
        self._gimbal = CubicSpline(t, gim)

        # Matrix to rotate from body to inertial frame for each INS point
        if offset is None:
            offset = np.array([[0., 0., 0.]])

        self.rot_b_to_mbs = get_aesa_rotation_offset_matrix(*self.gimbal_rotation) if aesa is not None else (
            get_rotation_offset_matrix(*self.gimbal_rotation))

        boresight_inertial = get_boresight_vector(self.rot_b_to_mbs, gim[:, 0], gim[:, 1], y, p, r)
        self.boresight = CubicSpline(t, boresight_inertial)
        self.az_iner = CubicSpline(t, np.arctan2(boresight_inertial[:, 0], boresight_inertial[:, 1]))
        self.el_iner = CubicSpline(t, -np.arcsin(boresight_inertial[:, 2]))

        if aesa is not None:
            aesa_phi_r, aesa_theta_r = aesa.T
            boresight_aesa = get_aesa_boresight_vector(self.rot_b_to_mbs, aesa_phi_r, aesa_theta_r, y, p, r)
            self.az_aesa_iner = CubicSpline(t, np.arctan2(boresight_aesa[:, 0], boresight_aesa[:, 1]))
            self.el_aesa_iner = CubicSpline(t, -np.arcsin(boresight_aesa[:, 2]))


            self.aesa_frame_phi_theta = CubicSpline(t, np.array([aesa_phi_r, aesa_theta_r]).T)
        else:
            self.az_aesa_iner = None
            self.el_aesa_iner = None

        tpos = np.stack([np.stack([apply_lever_arm_corrections(
            self.rot_b_to_mbs, y, p, r, gim[:, 0], gim[:, 1], self.gimbal_offset,
            off, pos) for off in a], axis=1) for a in offset], axis=1)

        # Build the position splines
        self._pos = CubicSpline(t, tpos)

        # Build a velocity spline
        self._vel = CubicSpline(t, median_filter(np.gradient(pos, axis=0), 15, axes=(0,)) *
                                INS_REFRESH_HZ)

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[:, 0], self._vel(lam_t)[:, 1])

    @property
    def pos(self):
        # Return the position lambda
        return self._pos

    @property
    def heading(self):
        # Return the heading lambda
        return self._heading

    @property
    def att(self):
        # Return the attitude lambda
        return self._att

    @property
    def vel(self):
        # Return velocity lambda
        return self._vel


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


def inertialToBody(yaw, pitch, roll, x, y, z):
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
    return rotItoB.dot(np.array([x, y, z]))


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


def m_p_matrix(pan, tilt, aesa=False):
    # Mounted Gimbal to Gimbal Pointing matrix
    cp = np.cos(pan)
    sp = np.sin(pan)
    ct = np.cos(tilt)
    st = np.sin(tilt)

    if not aesa:
        mg_gp = np.array([
            [cp, -sp, 0. if isinstance(pan, float) else np.zeros(len(cp))],
            [sp * ct, cp * ct, st],
            [-sp * st, -cp * st, ct]])
    else:
        mg_gp = np.array([
            [cp * ct, sp * ct, -st],
            [-sp, cp, 0. if isinstance(pan, float) else np.zeros(len(cp))],
            [cp * st, sp * st, ct]])

    return mg_gp.T if isinstance(pan, float) else mg_gp.swapaxes(0, 2)


def mg_b_matrix(pan, tilt, offset_rotation, offset_xyz, aesa=False):

    if isinstance(pan, np.ndarray):
        _stack = np.stack([np.eye(4) for _ in range(len(pan))])
        _stack[:, :3, 3] = offset_xyz
        _stack[:, :3, :3] = np.einsum('ijk->ikj', m_p_matrix(pan, tilt, aesa).dot(offset_rotation))
    else:
        _stack = np.eye(4)
        _stack[:3, 3] = offset_xyz
        _stack[:3, :3] = np.einsum('jk->kj', m_p_matrix(pan, tilt, aesa).dot(offset_rotation))

    return _stack


def get_inertial_to_body_matrix(
        a_yaw: float, a_pitch: float, a_roll: float, a_x: float, a_y: float, a_z: float) -> np.ndarray:
    cy = np.cos(a_yaw)
    sy = np.sin(a_yaw)
    cp = np.cos(a_pitch)
    sp = np.sin(a_pitch)
    cr = np.cos(a_roll)
    sr = np.sin(a_roll)

    # Compute the inertial to body rotation matrix
    rot_i_to_b = np.array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp, a_x],
        [cp * sy, cp * cy, sp, a_y],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp, a_z],
        [0, 0, 0, 1]])
    return rot_i_to_b


def get_body_to_inertial_matrix(
        a_yaw: float, a_pitch: float, a_roll: float, a_x: float, a_y: float, a_z: float) -> np.ndarray:
    cy = np.cos(a_yaw)
    sy = np.sin(a_yaw)
    cp = np.cos(a_pitch)
    sp = np.sin(a_pitch)
    cr = np.cos(a_roll)
    sr = np.sin(a_roll)

    # Compute the inertial to body rotation matrix
    rot_i_to_b = np.array([
        [cr * cy + sr * sp * sy, -cr * sy + sr * sp * cy, -sr * cp],
        [cp * sy, cp * cy, sp],
        [sr * cy - cr * sp * sy, -sr * sy - cr * sp * cy, cr * cp]])

    if isinstance(a_yaw, float):
        rot_b_to_i = np.eye(4)
        rot_b_to_i[:3, :3] = rot_i_to_b.T
        rot_b_to_i[:3, 3] = [a_x, a_y, a_z]
    else:
        rot_b_to_i = np.stack([np.eye(4) for _ in range(len(a_yaw))])
        rot_b_to_i[:, :3, :3] = rot_i_to_b.T
        rot_b_to_i[:, :3, 3] = np.array([a_x, a_y, a_z]).T
    return rot_b_to_i


def bodyToGimbal(rotBtoMG, pan, tilt, x, y, z):
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
    return rotBtoGP.dot(np.array([x, y, z]))


def getRotationOffsetMatrix(roll0, pitch0, yaw0, aesa = False):
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

    if aesa:
        return np.array([[Delta1, Delta2, Delta3], [Delta4, Delta5, Delta6], [Delta7, Delta8, Delta9]])
    else:
        return np.array([[-Delta1, -Delta2, -Delta3], [Delta4, Delta5, Delta6], [-Delta7, -Delta8, -Delta9]])


def getBoresightVector(ROffset, pan, tilt, yaw, pitch, roll, shift=None):
    """Returns the a 3x1 numpy array with the normalized boresight vector in 
    the inertial frame"""
    # set the boresight pointing vector in the pointed gimbal frame
    delta_gp = np.array([[0], [0], [1.0]]) if shift is None else shift.reshape((3, 1))
    # rotate these into the body frame
    delta_b = gimbalToBody(
        ROffset, pan, tilt, *delta_gp)
    # return the boresight in the inertial frame
    return bodyToInertial(yaw, pitch, roll, *delta_b)


def getAzElGimbalFromDesiredBoresight(boresight, ROffset, pan, tilt, yaw, pitch, roll):
    delta_i = inertialToBody(yaw, pitch, roll, *boresight)
    delta_b = bodyToGimbal(ROffset, pan, tilt, *delta_i)
    delta_b /= np.linalg.norm(delta_b)
    return np.arctan2(delta_b[0], delta_b[2]) + pan, -np.arcsin(delta_b[1]) + tilt



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


def createFlightPath(knot_points, init_pos, init_vel, launch_speed, roll_max, delta_t):

    # Build the gps profile for the missile
    gps_vel = [init_vel]
    gps_pos = [init_pos]
    gps_att = [np.array([0., -np.arcsin(init_vel[2]), np.arctan2(init_vel[0], init_vel[1])])]
    omega_max = GRAVITIC_CONSTANT * np.sqrt((1 / np.cos(roll_max)**2 - 1)) / launch_speed

    for knot in knot_points:
        tvec = knot - gps_pos[-1]
        while np.linalg.norm(tvec) > 10.:
            uvec = tvec / np.linalg.norm(tvec)
            turn_axis = np.cross(gps_vel[-1], uvec)
            turn_angle = np.clip(np.arctan2(np.linalg.norm(np.cross(gps_vel[-1], uvec)), np.dot(gps_vel[-1], uvec)),
                                 -omega_max * delta_t, omega_max * delta_t)

            skew_sym_cross = np.array([[0, -turn_axis[2], turn_axis[1]],
                                       [turn_axis[2], 0, -turn_axis[0]],
                                       [-turn_axis[1], turn_axis[0], 0]])
            rotation_matrix = np.eye(3) + np.sin(turn_angle) * skew_sym_cross + np.dot(skew_sym_cross, skew_sym_cross) * (
                    1 - np.cos(turn_angle))
            gps_vel.append(np.dot(rotation_matrix, gps_vel[-1]))
            gps_pos.append(gps_vel[-1] * delta_t * launch_speed + gps_pos[-1])

            # Calculate attitude
            phi_roll = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
            alpha_yaw = np.arctan2(gps_vel[-1][0], gps_vel[-1][1])
            theta_pitch = -np.arcsin(gps_vel[-1][2])
            gps_att.append(np.array([phi_roll, theta_pitch, alpha_yaw]))
            tvec = knot - gps_pos[-1]
    gps_pos = np.array(gps_pos)
    gps_att = np.array(gps_att)
    gps_vel = np.array(gps_vel) * launch_speed * delta_t

    r, p, y = gps_att.T
    e, n, u = gps_pos.T
    return e, n, u, r, p, y
