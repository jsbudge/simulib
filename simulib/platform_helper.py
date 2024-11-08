import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

from typing import Type
from .simulation_functions import llh2enu, findPowerOf2

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
INS_REFRESH_HZ = 100
SDRBase = Type

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
                 gps_rxpos: np.ndarray = None,
                 gps_txpos: np.ndarray = None,
                 *args,
                 **kwargs):
        self._gpst = t
        self._txant = tx_offset
        self._rxant = rx_offset
        self._gimbal = gimbal
        self._gimbal_offset = gimbal_offset
        self.gimbal_rotations = gimbal_rotations
        use_gps = gps_t is not None
        gps_t = gps_t if use_gps else t

        pos = np.array([e, n, u]).T

        # attitude spline
        yy = np.interp(t, gps_t, gps_az + 2 * np.pi) if use_gps else y
        self._att = CubicSpline(t, np.array([r, p, yy]).T)

        # Take into account the gimbal if necessary
        if gimbal is not None:
            self._gimbal_offset_mat = getRotationOffsetMatrix(*gimbal_rotations)
            # Matrix to rotate from body to inertial frame for each INS point
            tx_offset = tx_offset if tx_offset is not None else np.array([0., 0., 0.])
            rx_offset = rx_offset if rx_offset is not None else np.array([0., 0., 0.])
            self._gimbal_pan = CubicSpline(t, gimbal[:, 0])
            self._gimbal_tilt = CubicSpline(t, gimbal[:, 1])

            # Add to INS positions. X and Y are flipped since it rotates into NEU instead of ENU
            if use_gps:
                tpos = gps_txpos
                rpos = gps_rxpos
            else:
                tx_corrs = np.array(
                    [getPhaseCenterInertialCorrection(self._gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n],
                                                      p[n], r[n], tx_offset, gimbal_offset)
                     for n in range(gimbal.shape[0])])
                rx_corrs = np.array(
                    [getPhaseCenterInertialCorrection(self._gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n],
                                                      p[n], r[n], rx_offset, gimbal_offset)
                     for n in range(gimbal.shape[0])])
                tpos = pos + tx_corrs
                rpos = pos + rx_corrs

            # Rotate antenna into inertial frame in the same way as above
            bai = np.array([getBoresightVector(self._gimbal_offset_mat, gimbal[n, 0], gimbal[n, 1], y[n], p[n], r[n])
                            for n in range(gimbal.shape[0])])
            bai = bai.reshape((bai.shape[0], bai.shape[1])).T

            # Calculate antenna azimuth/elevation for beampattern
            if use_gps:
                gtheta = np.interp(gps_t, t, np.arcsin(-bai[2, :]))
                gphi = np.interp(gps_t, t, np.arctan2(bai[0, :], bai[1, :]))
            else:
                gtheta = np.arcsin(-bai[2, :])
                gphi = np.arctan2(bai[0, :], bai[1, :])
        else:
            tpos = rpos = pos
            gphi = y - np.pi / 2
            gtheta = np.zeros(len(t)) + 20 * DTR

        # Build the position splines
        self._pos = CubicSpline(t, pos)
        self._txpos = CubicSpline(gps_t, tpos)
        self._rxpos = CubicSpline(gps_t, rpos)

        # Build a velocity spline
        self._vel = CubicSpline(t, median_filter(np.gradient(pos, axis=0), 15, axes=(0,)) *
                                INS_REFRESH_HZ)

        # heading check
        self._heading = lambda lam_t: np.arctan2(self._vel(lam_t)[:, 0], self._vel(lam_t)[:, 1])

        # Beampattern stuff
        self.pan = CubicSpline(gps_t, gphi)
        self.tilt = CubicSpline(gps_t, gtheta)

    def boresight(self, t):
        # Get a boresight angle from the pan/tilt values
        r, p, y = self._att(t).T
        if isinstance(t, float):
            return getBoresightVector(self._gimbal_offset_mat, self._gimbal_pan(t), self._gimbal_tilt(t), y, p, r)
        else:
            return np.array([getBoresightVector(
                self._gimbal_offset_mat, self._gimbal_pan(t[i]), self._gimbal_tilt(t[i]), y[i], p[i], r[i])
                for i in range(len(t))]).squeeze(2)

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
        return self._rxpos

    @property
    def txpos(self):
        # Return Tx position lambda
        return self._txpos

    @property
    def vel(self):
        # Return velocity lambda
        return self._vel

    @property
    def gimbal_offset(self):
        return self._gimbal_offset


"""
    A class representing a Radar Platform that inherits from Platform.

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
                 gps_t: np.ndarray = None,
                 gps_az: np.ndarray = None,
                 gps_rxpos: np.ndarray = None,
                 gps_txpos: np.ndarray = None,
                 tx_num: int = 0,
                 rx_num: int = 0,
                 wavenumber: int = 0,
                 *args,
                 **kwargs):

        """
        Initializes a Radar Platform object with provided parameters.

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
            gps_rxpos: np.ndarray. GPS receiver position.
            gps_txpos: np.ndarray. GPS transmitter position.
            tx_num: int. Transmitter number (default: 0).
            rx_num: int. Receiver number (default: 0).
            wavenumber: int. Wavenumber (default: 0).
        """

        super().__init__(e, n, u, r, p, y, t, gimbal, np.array(gimbal_offset), np.array(gimbal_rotations),
                         tx_offset, rx_offset, gps_t, gps_az, gps_rxpos, gps_txpos)
        self.dep_ang = dep_angle * DTR
        self.squint_ang = squint_angle * DTR
        self.az_half_bw = az_bw * DTR / 2
        self.el_half_bw = el_bw * DTR / 2
        self.fs = fs
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
        nrange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang + self.el_half_bw * exp_factor)
        frange = height / np.sin(self._att(self.gpst[0])[0] + self.dep_ang - self.el_half_bw * exp_factor)
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
        else:
            nrange = ranges[0]
            frange = ranges[1]
        pl_s = self.calcPulseLength(height, plp)
        return int((np.ceil((2 * frange / c0 + pl_s) * TAC) - np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)

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
            frange = ranges[1]
        pl_s = self.calcPulseLength(height, plp, nrange=None if ranges is None else ranges[0])
        nsam = int((np.ceil((2 * frange / c0 + pl_s) * TAC) -
                    np.floor(2 * nrange / c0 * TAC)) * self.fs / TAC)
        MPP = c0 / self.fs / upsample
        return nrange + np.arange(nsam * upsample) * MPP + c0 / self.fs

    def getRadarParams(self, fdelay, plp, upsample=1):
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
        nsam = self.calcNumSamples(fdelay, plp)
        nr = self.calcPulseLength(fdelay, plp, True)
        ranges = self.calcRangeBins(fdelay, upsample, plp)
        ranges_sampled = self.calcRangeBins(fdelay, 1, plp)
        near_range_s = ranges[0] / c0
        granges = ranges * np.cos(self.dep_ang)
        fft_len = findPowerOf2(nsam + self.calcPulseLength(fdelay, plp, use_tac=True))
        up_fft_len = fft_len * upsample
        return nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len


"""
SDRPlatform
More specialized version of RadarPlatform that gets all of the values it needs from a SAR file or instance of
SDRParse.
"""


class APSDebugPlatform(RadarPlatform):
    _sdr: object

    def __init__(self,
                 sdr: SDRBase,
                 origin: np.ndarray = None,
                 tx_offset: np.ndarray = None,
                 rx_offset: np.ndarray = None,
                 gimbal_offset: np.ndarray = None,
                 fs: float = 2e9,
                 channel: int = 0,
                 gps_data: dict = None,
                 gimbal_data: dict = None):
        """
        Init function.
        :param sdr: SDRParse object or str. This is path to the SAR file used as a basis for other calculations,
            or the SDRParse object of an already parsed file.
        :param origin: 3-tuple. Point used as the origin for ENU reference frame, in (lat, lon, alt).
        :param tx_offset: 3-tuple. Offset of Tx antenna from body frame in meters.
        :param rx_offset: 3-tuple. Offset of Rx antenna from body frame in meters.
        :param fs: float. Sampling frequency in Hz.
        :param channel: int. Channel of data for this object to represent in the SAR file.
        :param gps_data: This is from postCorrectionGPSData. Loaded in via aps_io in data_converter.
        :param gimbal_data: Gimbal data from aps_io in data_converter.
        """
        t = sdr.gps_data.index.values
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[0, :])
        e, n, u = llh2enu(gps_data['lat'], gps_data['lon'], gps_data['alt'], origin)
        e = np.interp(t, gps_data['gps_ms'], e)
        n = np.interp(t, gps_data['gps_ms'], n)
        u = np.interp(t, gps_data['gps_ms'], u)
        r = sdr.gps_data['r'].values
        p = sdr.gps_data['p'].values
        y = sdr.gps_data['y'].values
        if gimbal_data is None:
            try:
                pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                                sdr.gimbal['pan'].values.astype(np.float64))
                tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                                 sdr.gimbal['tilt'].values.astype(np.float64))
            except TypeError:
                pan = np.zeros_like(sdr.gps_data['systime'].values)
                tilt = np.zeros_like(sdr.gps_data['systime'].values)
            pan = np.interp(t, sdr.gps_data.index.values, pan)
            tilt = np.interp(t, sdr.gps_data.index.values, tilt)
        else:
            if len(gimbal_data['systime']) == 0:
                # This is for the weird interpolated to each channel data
                times = sdr[channel].pulse_time[:len(gimbal_data['pan'])]
            else:
                times = np.interp(gimbal_data['systime'], sdr.gps_data['systime'], sdr.gps_data.index)
            pan = np.interp(t, times, gimbal_data['pan'])
            tilt = np.interp(t, times, gimbal_data['tilt'])
        goff = np.array(
            [sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset]) if gimbal_offset is None else gimbal_offset
        grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR])
        try:
            channel_dep = (sdr.xml['Channel_0']['Near_Range_D'] + sdr.xml['Channel_0']['Far_Range_D']) / 2
        except KeyError:
            channel_dep = sdr.ant[0].dep_ang / DTR
        if sdr[channel].is_receive_only:
            tx_num = np.where([n is not None for n in sdr.port])[0][0]
        else:
            tx_num = sdr[channel].trans_num
            tx_offset = np.array(
                [sdr.port[tx_num].x, sdr.port[tx_num].y, sdr.port[tx_num].z]) if tx_offset is None else tx_offset
        rx_num = sdr[channel].rec_num
        rx_offset = np.array(
            [sdr.port[rx_num].x, sdr.port[rx_num].y, sdr.port[rx_num].z]) if rx_offset is None else rx_offset
        super().__init__(e=e, n=n, u=u, r=r, p=p, y=y, t=t, tx_offset=tx_offset, rx_offset=rx_offset,
                         gimbal=np.array([pan, tilt]).T, gimbal_offset=goff, gimbal_rotations=grot,
                         dep_angle=channel_dep, squint_angle=sdr.ant[sdr.port[tx_num].assoc_ant].squint / DTR,
                         az_bw=sdr.ant[sdr.port[tx_num].assoc_ant].az_bw / DTR,
                         el_bw=sdr.ant[sdr.port[tx_num].assoc_ant].el_bw / DTR, fs=fs, gps_t=gps_data['gps_ms'],
                         tx_num=tx_num, rx_num=rx_num, gps_az=gps_data['az'], gps_rxpos=gps_data['rxpos'],
                         gps_txpos=gps_data['txpos'])
        self._sdr: SDRBase = sdr
        self.origin = origin
        self._channel = channel

    def calcRanges(self, fdelay, partial_pulse_percent=1., **kwargs):
        """
        Calculate near and far ranges for this collect using the SAR file.
        :param fdelay: float. FDelay desired in TAC.
        :param partial_pulse_percent: float, <1. Percentage of maximum pulse length to use in radar.
        :return: tuple of near and far ranges in meters.
        """
        nrange = ((self._sdr[0].receive_on_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                  self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        frange = ((self._sdr[0].receive_off_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
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
        :param plp: float, <1. Percentage of maximum pulse length to use in radar.
        :param ranges: For compatibility. Not used.
        :return: array of range bins in meters.
        """
        nrange, frange = self.calcRanges(fdelay, partial_pulse_percent=plp)
        MPP = c0 / self.fs / upsample
        return (nrange * 2 + np.arange(self.calcNumSamples() * upsample) * MPP) / 2

    def calcRadVelRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the radial velocity resolution for this collect. :param cpi_len: int. Length of CPI in number of
        pulses. :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected
        optimal value. :return: Radial velocity resolution in meters per second.
        """
        return dopplerBroadeningFactor * c0 * self._sdr[self._channel].prf / \
            (self._sdr[self._channel].fc * cpi_len)

    def calcDopRes(self, cpi_len, dopplerBroadeningFactor=2.5):
        """
        Calculate the Doppler resolution for this collect. :param cpi_len: int. Length of CPI in number of pulses.
        :param dopplerBroadeningFactor: float. Factor by which to decrease the resolution from the expected optimal
        :param cpi_len: CPI length.
        value. :return: Doppler resolution in Hz.
        """
        return dopplerBroadeningFactor * self._sdr[self._channel].prf / cpi_len

    def calcWrapVel(self):
        """
        Calculate the wrap velocity for this collect.
        :return: Wrap velocity in meters per second.
        """
        return self._sdr[self._channel].prf * (c0 / self._sdr[self._channel].fc) / 4.0


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
        Init function.
        :param sdr: SDRParse object or str. This is path to the SAR file used as a basis for other calculations,
            or the SDRParse object of an already parsed file.
        :param origin: 3-tuple. Point used as the origin for ENU reference frame, in (lat, lon, alt).
        :param tx_offset: 3-tuple. Offset of Tx antenna from body frame in meters.
        :param rx_offset: 3-tuple. Offset of Rx antenna from body frame in meters.
        :param fs: float. Sampling frequency in Hz.
        :param channel: int. Channel of data for this object to represent in the SAR file.
        """
        t = sdr.gps_data.index.values
        fs = fs if fs is not None else sdr[channel].fs
        origin = origin if origin is not None else (sdr.gps_data[['lat', 'lon', 'alt']].values[0, :])
        e, n, u = llh2enu(sdr.gps_data['lat'], sdr.gps_data['lon'], sdr.gps_data['alt'], origin)
        r = sdr.gps_data['r'].values
        p = sdr.gps_data['p'].values
        y = sdr.gps_data['y'].values
        try:
            pan = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                            sdr.gimbal['pan'].values.astype(np.float64))
            tilt = np.interp(sdr.gps_data['systime'].values, sdr.gimbal['systime'].values.astype(int),
                             sdr.gimbal['tilt'].values.astype(np.float64))
        except TypeError:
            pan = np.zeros_like(sdr.gps_data['systime'].values)
            tilt = np.zeros_like(sdr.gps_data['systime'].values)
        pan = np.interp(t, sdr.gps_data.index.values, pan)
        tilt = np.interp(t, sdr.gps_data.index.values, tilt)
        goff = np.array(
            [sdr.gim.x_offset, sdr.gim.y_offset, sdr.gim.z_offset]) if gimbal_offset is None else gimbal_offset
        grot = np.array([sdr.gim.roll * DTR, sdr.gim.pitch * DTR, sdr.gim.yaw * DTR])
        try:
            channel_dep = (sdr.xml['Channel_0']['Near_Range_D'] + sdr.xml['Channel_0']['Far_Range_D']) / 2
        except KeyError:
            channel_dep = sdr.ant[0].dep_ang / DTR
        if sdr[channel].is_receive_only:
            tx_num = np.where([n is not None for n in sdr.port])[0][0]
        else:
            tx_num = sdr[channel].trans_num
            tx_offset = np.array(
                [sdr.port[tx_num].x, sdr.port[tx_num].y, sdr.port[tx_num].z]) if tx_offset is None else tx_offset
        rx_num = sdr[channel].rec_num
        rx_offset = np.array(
            [sdr.port[rx_num].x, sdr.port[rx_num].y, sdr.port[rx_num].z]) if rx_offset is None else rx_offset
        super().__init__(e=e, n=n, u=u, r=r, p=p, y=y, t=t, tx_offset=tx_offset, rx_offset=rx_offset,
                         gimbal=np.array([pan, tilt]).T, gimbal_offset=goff, gimbal_rotations=grot,
                         dep_angle=channel_dep, squint_angle=sdr.ant[sdr.port[tx_num].assoc_ant].squint / DTR,
                         az_bw=sdr.ant[sdr.port[tx_num].assoc_ant].az_bw / DTR,
                         el_bw=sdr.ant[sdr.port[tx_num].assoc_ant].el_bw / DTR, fs=fs, tx_num=tx_num,
                         rx_num=rx_num)
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
        nrange = ((self._sdr[0].receive_on_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
                  self._sdr[self._channel].pulse_length_S * partial_pulse_percent) * c0 / 2
        frange = ((self._sdr[0].receive_off_TAC - self._sdr[self._channel].transmit_on_TAC - fdelay) / TAC -
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
    return np.linalg.pinv(rotItoB).T.dot(np.array([x, y, z]))


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
