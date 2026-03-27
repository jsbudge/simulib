import numpy as np
from scipy.interpolate import interpn
from scipy.spatial.transform import Rotation as rot
from itertools import product
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy.special import i0
import plotly.io as pio
import os
from functools import reduce
from dted import Tile, LatLon
from pathlib import Path
import rasterio

pio.renderers.default = 'browser'

WGS_A = 6378137.0
WGS_F = 1 / 298.257223563
WGS_B = 6356752.314245179
WGS_E2 = 6.69437999014e-3
c0 = 299792458.0
DTR = np.pi / 180
DAC_FREQ_HZ = 4e9
BASE_COMPLEX_SRATE_HZ = DAC_FREQ_HZ / 2


def getDTEDName(lat: float, lon: float) -> str:
    """Return the path and name of the dted to load for the given lat/lon"""
    tmplat = int(np.floor(lat))
    tmplon = int(np.floor(lon))
    direw = 'w' if tmplon < 0 else 'e'
    dirns = 's' if tmplat < 0 else 'n'
    win_path = f'E:\\dted\\{direw}{abs(tmplon)}\\{dirns}{abs(tmplat)}'
    unix_path = f'/data1/dted/{direw}{abs(tmplon)}/{dirns}{abs(tmplat)}'
    if Path(win_path + '.dt2').exists():
        return win_path + '.dt2'
    elif Path(win_path + '.dt3').exists():
        return win_path + '.dt3'
    elif Path(unix_path + '.dt2').exists():
        return unix_path + '.dt2'
    else:
        return unix_path + '.dt3'


def detect_local_extrema(arr: np.ndarray) -> tuple:
    """
    Given a 2d array of floats, returns the areas of local extrema
    :param arr: NxM array of floats
    :return: tuple of locations of extrema.
    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = filters.minimum_filter(arr, footprint=neighborhood) == arr
    # local_max = filters.maximum_filter(arr, footprint=neighborhood) == arr
    background = arr == 0
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_extrema = local_min ^ eroded_background  # + local_max ^ eroded_background
    return np.where(detected_extrema)


def db(x: float | np.ndarray) -> float | np.ndarray:
    ret = abs(x)
    if isinstance(ret, np.ndarray):
        ret[ret < 1e-15] = 1e-15
    else:
        ret = max(ret, 1e-15)
    return 20 * np.log10(ret)


def findPowerOf2(x: float | np.ndarray) -> float | np.ndarray:
    return int(2 ** (np.ceil(np.log2(x))))


def undulationEGM96(lat: float | np.ndarray, lon: float | np.ndarray) -> float | np.ndarray:
    inp_file = f'{os.path.dirname(os.path.abspath(__file__))}/geoids/EGM96.DAT'
    with open(inp_file, "rb") as f:  # or "rt" as text file with universal newlines
        egm96 = np.fromfile(f, 'double', 1441 * 721, '')
    eg_n = np.ceil(lat / .25) * .25
    eg_s = np.floor(lat / .25) * .25
    eg_e = np.ceil(lon / .25) * .25
    eg_w = np.floor(lon / .25) * .25
    eg1 = egm96[((eg_w + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_n + 90 + .25) / .25 - 1).astype(int)]
    eg2 = egm96[((eg_w + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_s + 90 + .25) / .25 - 1).astype(int)]
    eg3 = egm96[((eg_e + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_n + 90 + .25) / .25 - 1).astype(int)]
    eg4 = egm96[((eg_e + 180 + .25) / .25).astype(int) - 1 + 1441 * ((eg_s + 90 + .25) / .25 - 1).astype(int)]
    return (
        (eg2 / ((eg_e - eg_w) * (eg_n - eg_s))) * (eg_e - lon) * (eg_n - lat)
        + (eg4 / ((eg_e - eg_w) * (eg_n - eg_s))) * (lon - eg_w) * (eg_n - lat)
        + (eg1 / ((eg_e - eg_w) * (eg_n - eg_s))) * (eg_e - lon) * (lat - eg_s)
        + (eg3 / ((eg_e - eg_w) * (eg_n - eg_s))) * (lon - eg_w) * (lat - eg_s)
    )


def getElevationMap(lats: np.ndarray, lons: np.ndarray, und: bool = True, interp_method: str = 'linear') -> np.ndarray:
    """
    Get elevations from DTED library for a block of lat/lon pairs.
    :param lats: Latitude values.
    :param lons: Longitude values.
    :param und: If True, applies undulation to get to EGM96.
    :param interp_method: Passed to the interpolator for the DTED grid.
    :return: List of elevation values same size as lats/lons.
    """
    # First, check to see if multiple DTEDs are needed
    floor_lats = np.floor(lats)
    floor_lons = np.floor(lons)
    hght = np.zeros_like(lats)
    dteds = list(product(np.unique(np.floor(lats)), np.unique(np.floor(lons))))
    for ted in dteds:
        idx = np.logical_and(floor_lats == ted[0], floor_lons == ted[1])
        dtedName = getDTEDName(*ted)
        block = Tile(dtedName, in_memory=False)
        block.load_data(perform_checksum=False)
        data = block.data
        ulx = block.dsi.origin.latitude
        uly = block.dsi.origin.longitude
        xres = 1 / block.dsi.shape[0]
        yres = 1 / block.dsi.shape[1]

        y = (lats[idx] - ulx) / xres
        x = (lons[idx] - uly) / yres

        hght[idx] = interpn(np.array([np.arange(block.dsi.shape[0]), np.arange(block.dsi.shape[1])]), data, np.array([x, y]).T,
                            method=interp_method, bounds_error=False, fill_value=0) + undulationEGM96(lats[idx], lons[
            idx]) if und else hght

    return hght


def getElevationTIFF(tiff_name: str, lats: float | np.ndarray, lons: float | np.ndarray, und: bool = True,
                     interp_method: str = 'linear') -> float | np.ndarray:
    """
    Get elevation from TIFF file.
    :param tiff_name: Path to TIFF file.
    :param lats: Latitude values.
    :param lons: Longitude values.
    :param und: If True, applies undulation to get to EGM96.
    :param interp_method: Passed to the interpolator for the DTED grid.
    :return: Elevation value(s) same size as lats/lons.
    """
    with rasterio.open(tiff_name) as src:
        if isinstance(lats, float):
            row, col = src.index(lons, lats)
            if row > 0 and col > 0:
                return src.read(1)[row, col] + undulationEGM96(lats, lons) if und else 0.
            else:
                print(f'Desired lat and lon ({lats}, {lons}) are not inside of TIFF area.')
                return -32767.
        else:
            data = src.read(1, resampling=rasterio.enums.Resampling(2))
            x = (lats - src.bounds[0]) / src.res[0]
            y = (lons - src.bounds[1]) / src.res[1]
            return interpn(np.array([np.arange(data.shape[0]), np.arange(data.shape[1])]), data, np.array([x, y]).T,
                                method=interp_method, bounds_error=False, fill_value=0) + (
            undulationEGM96(lats, lons) if und else 0.)




def getElevation(lat: float, lon: float, und: bool = True):
    """Returns the digital elevation for a latitude and longitude"""
    ted = getDTEDName(np.floor(lat), np.floor(lon))
    data = Tile(ted, in_memory=False)
    ulx = data.dsi.origin.latitude
    uly = data.dsi.origin.longitude
    xres = 1 / data.dsi.shape[0]
    yres = -1 / data.dsi.shape[1]

    x = data.dsi.shape[0] - (lat - ulx) / xres
    y = (lon - uly) / yres
    x1 = int(x) * xres
    x2 = int(x + 1) * xres
    y1 = int(y) * yres
    y2 = int(y + 1) * yres
    x *= xres
    y *= yres

    dtedData = np.array([data.get_elevation(LatLon(latitude=ulx + 1 - x1, longitude=uly + y1)),
                         data.get_elevation(LatLon(latitude=ulx + 1 - x1, longitude=uly + y2)),
                         data.get_elevation(LatLon(latitude=ulx + 1 - x2, longitude=uly + y1)),
                         data.get_elevation(LatLon(latitude=ulx + 1 - x2, longitude=uly + y2))])

    elevation = 1 / ((x2 - x1) * (y2 - y1)) * \
                dtedData.dot(np.array([[x2 * y2, -y2, -x2, 1],
                                       [-x2 * y1, y1, x2, -1],
                                       [-x1 * y2, y2, x1, -1],
                                       [x1 * y1, -y1, -x1, 1]])).dot(np.array([1, x, y, x * y]))

    return elevation + undulationEGM96(lat, lon) if und else elevation


def llh2enu(lat: float | np.ndarray, lon: float | np.ndarray, h: float | np.ndarray,
            refllh: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    """
    Converts from Lat/Lon/Alt to local tangent plane ENU
    :param lat: Latitude values.
    :param lon: Longitude values.
    :param h: Altitude values.
    :param refllh: Reference lat/lon/alt for local tangent plane ENU.
    :return: Values in ENU.
    """
    ecef = llh2ecef(lat, lon, h)
    return ecef2enu(*ecef, refllh)


def enu2llh(e: float | np.ndarray, n: float | np.ndarray, u: float | np.ndarray,
            refllh: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    """
    Converts from ENU to Lat/Lon/Alt.
    :param e: Easting values.
    :param n: Northing values.
    :param u: Up values.
    :param refllh: Reference lat/lon/alt for local tangent plane ENU.
    :return: Values in Lat/Lon/Alt.
    """
    ecef = enu2ecef(e, n, u, refllh)
    return ecef2llh(*ecef)


def enu2ecef(e: float | np.ndarray, n: float | np.ndarray, u: float | np.ndarray,
            refllh: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    """
    Converts from ENU to ECEF.
    :param e: Easting values.
    :param n: Northing values.
    :param u: Up values.
    :param refllh: Reference lat/lon/alt for local tangent plane ENU.
    :return: Values in ECEF.
    """
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    enu = np.array([e, n, u])
    tmp_rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                        [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                        [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]]).T
    if len(enu.shape) > 1:
        sz = np.ones((enu.shape[1],))
        ecef = tmp_rot.dot(enu) + np.array([sz * rx, sz * ry, sz * rz])
    else:
        ecef = tmp_rot.dot(enu) + np.array([rx, ry, rz])
    return ecef[0], ecef[1], ecef[2]


def llh2ecef(lat: float | np.ndarray, lon: float | np.ndarray, h: float | np.ndarray) -> tuple[float, float, float]:
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y, Z
    given the Geodetic Coordinates lat, lon + Ellipsoid Height h
    """
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180
    N = WGS_A / np.sqrt(1 - WGS_E2 * np.sin(lat_rad) ** 2)
    X = (N + h) * np.cos(lat_rad) * np.cos(lon_rad)
    Y = (N + h) * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (WGS_B ** 2 / WGS_A ** 2 * N + h) * np.sin(lat_rad)
    return X, Y, Z


def ecef2llh(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray) -> tuple[float, float, float]:
    # This is the Heikkinen application of the Ferrari solution to Bowring's irrational
    # geodetic-latitude equation to get a geodetic latitude and height.
    # Longitude remains the same between the two.
    r = np.sqrt(x ** 2 + y ** 2)
    ep2 = (WGS_A ** 2 - WGS_B ** 2) / WGS_B ** 2
    F = 54 * WGS_B ** 2 * z ** 2
    G = r ** 2 + (1 - WGS_E2) * z ** 2 - WGS_E2 * (WGS_A ** 2 - WGS_B ** 2)
    c = WGS_E2 ** 2 * F * r ** 2 / G ** 3
    s = (1 + c + np.sqrt(c ** 2 + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G ** 2)
    Q = np.sqrt(1 + 2 * WGS_E2 ** 2 * P)
    r0 = -P * WGS_E2 * r / (1 + Q) + np.sqrt(
        1 / 2 * WGS_A ** 2 * (1 + 1 / Q) - P * (1 - WGS_E2) * z ** 2 / (Q * (1 + Q)) - 1 / 2 * P * r ** 2)
    U = np.sqrt((r - WGS_E2 * r0) ** 2 + z ** 2)
    V = np.sqrt((r - WGS_E2 * r0) ** 2 + (1 - WGS_E2) * z ** 2)
    z0 = WGS_B ** 2 * z / (WGS_A * V)
    h = U * (1 - WGS_B ** 2 / (WGS_A * V))
    lat = np.arctan((z + ep2 * z0) / r) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return lat, lon, h


def ecef2enu(x: float | np.ndarray, y: float | np.ndarray, z: float | np.ndarray,
             refllh: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                    [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                    [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]])
    enu = rot.dot(np.array([x - rx, y - ry, z - rz]))
    return enu[0], enu[1], enu[2]


def genPulse(phase_x: np.ndarray, phase_y: np.ndarray, nnr: int, nfs: float, nfc: float, bandw: float) -> np.ndarray:
    """
    Generates a pulse with the given phase characteristics.
    :param phase_x: Normalized points along the duration of the pulse to put a phase knot.
    :param phase_y: Normalized locations of phase knots within bandwidth, e.g. 1 = bandwidth frequency
    :param nnr: Number of sample points of pulse.
    :param nfs: Sampling frequency.
    :param nfc: Center frequency of pulse.
    :param bandw: Bandwidth of pulse.
    :return: Pulse array of length nnr with the given phase characteristics.
    """
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * 1 / nfs))


def genChirp(nnr: int, nfs: float, nfc: float, bandw: float) -> np.ndarray:
    """
    Generates a linear frequency modulated chirp.
    :param nnr: Number of sample points of pulse.
    :param nfs: Sampling frequency.
    :param nfc: Center frequency of pulse.
    :param bandw: Bandwidth of pulse.
    :return: Pulse array of length nnr with linear frequency modulated chirp.
    """
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * 1 / nfs))


def azelToVec(az: float | np.ndarray, el: float | np.ndarray) -> np.ndarray:
    """
    Converts from inertial azimuth and depression angle to a pointing vector.
    :param az: Azimuth in radians.
    :param el: Depression in radians.
    :return: Nx3 array of pointing vectors.
    """
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)]).T


def calcSNR(p_s, ant_g, az, el, wavelength, pulse_time, bandw, dopp_mult, rng, rcs_val=None):
    sig = calcPower(p_s, ant_g, az, el, wavelength, pulse_time, bandw, dopp_mult, rng, rcs_val)
    # Boltzmans constant
    kb = 1.3806503e-23
    # the reference temperature (in Kelvin)
    T0 = 290.0
    # Noise figure of 3dB
    F = 10.0 ** (3.0 / 10.0)
    N0 = kb * T0 * F
    sigma_n = np.sqrt(N0 * bandw)
    noise = sigma_n ** 2
    return db(np.array([sig / noise]))[0]


def calcPower(p_s, ant_g, az, el, wavelength, pulse_time, bandw, dopp_mult, rng, rcs_val=None):
    if rcs_val is None:
        sig = p_s * ant_g ** 2 * np.tan(az) * np.tan(el) * wavelength ** 2 * pulse_time * bandw * dopp_mult
        sig /= (4 * np.pi) ** 3 * rng ** 2
    else:
        sig = p_s * ant_g ** 2 * rcs_val * wavelength ** 2 * pulse_time * bandw * dopp_mult
        sig /= (4 * np.pi) ** 3 * rng ** 4
    return sig


def factors(n: int) -> list:
    """
    Returns a list of all factors of n.
    :param n: Number to get factors of.
    :return: List of all factors of n.
    """
    return list(set(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0))))


def genTaylorWindow(baseband_fc: float, half_bw: float, fs: float, fft_len: int, nbar: int = 5,
                    sll: float = -35) -> np.ndarray:
    # Get the basebanded center, start and stop frequency of the chirp
    basebandedStartFreqHz = baseband_fc - half_bw
    basebandedStopFreqHz = baseband_fc + half_bw
    windowSize = \
        int(np.floor(half_bw * 2.0 / fs * fft_len))
    taylorWindow = window_taylor(windowSize, nbar=nbar, sll=sll) if sll != 0 else np.ones(windowSize)

    # IQ baseband vs offset video
    if np.sign(basebandedStartFreqHz) != np.sign(basebandedStopFreqHz):
        aboveZeroLength = int(np.ceil((baseband_fc + half_bw) / fs * fft_len))
        taylorWindowExtended = np.zeros(fft_len)
        taylorWindowExtended[fft_len // 2 - aboveZeroLength: fft_len // 2 - aboveZeroLength + windowSize] = taylorWindow
        taylorWindowExtended = np.fft.fftshift(taylorWindowExtended)
    else:
        bandStartInd = \
            int(np.floor((baseband_fc - half_bw) / fs * fft_len))
        taylorWindowExtended = np.zeros(fft_len)
        if bandStartInd + windowSize > fft_len:
            taylorWindowExtended[bandStartInd:] = taylorWindow[:fft_len - bandStartInd]
            taylorWindowExtended[:windowSize - (fft_len - bandStartInd)] = taylorWindow[fft_len - bandStartInd:]
        else:
            taylorWindowExtended[bandStartInd: bandStartInd + windowSize] = taylorWindow
    return taylorWindowExtended


def getRadarCoeff(fc: float, ant_transmit_power: float, rx_gain: float, tx_gain: float, rec_gain: float) -> float:
    """
    Calculates the coefficient needed for the radar equation that does not change with range or angle of arrival.
    :param fc: Center frequency in Hz.
    :param ant_transmit_power: Antenna transmission power in watts.
    :param rx_gain: Receiver gain in dB.
    :param tx_gain: Transmitter gain in dB.
    :param rec_gain: Upconverter gain in dB.
    :return: Radar coefficient.
    """
    return (c0 ** 2 / fc ** 2 * ant_transmit_power * 10 ** ((rx_gain + 2.15) / 10) * 10 ** ((tx_gain + 2.15) / 10) *
     10 ** ((rec_gain + 2.15) / 10) / (4 * np.pi) ** 3)


def window_taylor(N: int, nbar: float = 4., sll: float = -30.) -> np.ndarray:
    """Taylor tapering window
    Taylor windows allows you to make tradeoffs between the
    mainlobe width and sidelobe level (sll).
    Implemented as described by Carrara, Goodman, and Majewski
    in 'Spotlight Synthetic Aperture Radar: Signal Processing Algorithms'
    Pages 512-513
    :param N: window length
    :param float nbar:
    :param float sll:
    The default values gives equal height
    sidelobes (nbar) and maximum sidelobe level (sll).
    .. warning:: not implemented
    .. seealso:: :func:`create_window`, :class:`Window`
    """
    if sll > 0:
        sll *= -1
    B = 10 ** (-sll / 20)
    A = np.log(B + np.sqrt(B ** 2 - 1)) / np.pi
    s2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)
    ma = np.arange(1, nbar)

    def calc_Fm(m):
        numer = (-1) ** (m + 1) \
                * np.prod(1 - m ** 2 / s2 / (A ** 2 + (ma - 0.5) ** 2))
        denom = 2 * np.prod([1 - m ** 2 / j ** 2 for j in ma if j != m])
        return numer / denom

    Fm = np.array([calc_Fm(m) for m in ma])

    def W(n):
        return 2 * np.sum(
            Fm * np.cos(2 * np.pi * ma * (n - N / 2 + 1 / 2) / N)) + 1

    w = np.array([W(n) for n in range(N)])
    # normalize (Note that this is not described in the original text)
    scale = W((N - 1) / 2)
    w /= scale
    return w


def slant_path_atmospheric_attenuation(gamma0: float, gammaw: float, f: float, el_ang: float):
    """
    Calculates the slant path attenuation at a given elevation. Taken from Armin Doerry report on atmospheric conditions
    affecting attenuation.
    :param gamma0:
    :param gammaw:
    :param f:
    :param el_ang:
    :return:
    """
    h0 = 6
    hw = 1.6 * (1 + 3 / ((f - 22.2)**2 + 5) + 5 / ((f - 183.3)**2 + 6) + 2.5 / ((f - 325.4)**2 + 4))
    return (h0 * gamma0 + hw * gammaw) / np.sin(el_ang)


def marcumq(alpha, T, end, numSamples):
    """My implimentation of the Marcum-Q function"""
    t = np.linspace(T, end, numSamples)
    dt = t[1] - t[0]
    alpha_2 = alpha ** 2
    ret = sum(t * np.exp(-.5 * (t ** 2 + alpha_2)) * i0(alpha * t) * dt)
    return 1.0 if np.isnan(ret) else ret


def getDoppResolution(broad_factor, PRF, lamda, Ncpi):
    DopplerResolutionHz = broad_factor * PRF / Ncpi
    radialVelocityResolutionMPerS = DopplerResolutionHz * lamda / 2.0
    # 3) Velocity uncertainty is dominated by the velocity sample spacing, but our
    #   ability to know the pointing direction of the antenna and estimate the
    #   azimuth angle of the target also comes into play.
    radialVelocityUncertaintyMPerS = PRF * lamda / (2.0 * Ncpi) / 2
    return DopplerResolutionHz, radialVelocityUncertaintyMPerS, radialVelocityResolutionMPerS


def sinc_interp(x, s, u):
    if len(x) != len(s):
        raise ValueError('x and s must be the same length')

    # Find the period
    T = s[1] - s[0]

    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    return np.dot(x, np.sinc(sincM / T))


def getDopplerLine(effAzI, rangeBins, antVel, antPos, nearRangeGrazeR, azBeamwidthHalf, PRF, wavelength, origin=None):
    """Compute the expected Doppler vs range for the given platform geometry"""

    # compute the grazing angle for the near range to start
    if origin is None:
        effectiveHeight = antPos[2]
    else:
        (nearRangeGrazeR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
            effAzI, nearRangeGrazeR, antPos, rangeBins[0], origin)
        effectiveHeight = antPos[2] + origin[2] - surfaceHeight

    # now I need to get the grazing angles across all of the range bins
    grazeOverRanges = np.arcsin(effectiveHeight / rangeBins)

    # this is a special version of Rvec (it is not 3x1, it is 3xNrv)
    Rvec = np.array([
        np.cos(grazeOverRanges) * np.sin(effAzI),
        np.cos(grazeOverRanges) * np.cos(effAzI),
        -np.sin(grazeOverRanges)])
    # perform the dot product and calculate the Doppler
    DopplerCen = ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = np.nonzero(DopplerCen > PRF / 2)
    DopplerCen[ind] -= PRF
    ind = np.nonzero(DopplerCen < -PRF / 2)
    DopplerCen[ind] += PRF

    # generate the radial vector for the forward beamwidth edge
    # (NOTE!!!: this is dependent
    # on the antenna pointing vector attitude with respect to the aircraft heading.
    # if on the left side, negative azimuth will be lower Doppler, and positive
    # azimuth will be higher, but on the right side, it will be the opposite, one
    # could use the sign of the cross-product to determine which it is.)
    # if (xmlData.gimbalSettings.lookSide.lower() == 'left'):
    eff_boresight = np.mean(np.array([
        np.cos(grazeOverRanges) * np.sin(effAzI),
        np.cos(grazeOverRanges) * np.cos(effAzI),
        -np.sin(grazeOverRanges)]), axis=1)
    ant_dir = np.cross(eff_boresight, antVel)
    azBeamwidthHalf *= np.sign(ant_dir[2])

    newAzI = effAzI - azBeamwidthHalf
    Rvec = np.array([
        np.cos(grazeOverRanges) * np.sin(newAzI),
        np.cos(grazeOverRanges) * np.cos(newAzI),
        -np.sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerUp = ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = np.nonzero(DopplerUp > PRF / 2)
    DopplerUp[ind] -= PRF
    ind = np.nonzero(DopplerUp < -PRF / 2)
    DopplerUp[ind] += PRF

    # generate the radial vector for the forward beamwidth edge
    newAzI = effAzI + azBeamwidthHalf
    Rvec = np.array([
        np.cos(grazeOverRanges) * np.sin(newAzI),
        np.cos(grazeOverRanges) * np.cos(newAzI),
        -np.sin(grazeOverRanges)])
    # perform the dot product and calculate the Upper Doppler
    DopplerDown = \
        ((2.0 / wavelength) * Rvec.T.dot(antVel).flatten()) % PRF
    # account for wrapping of the Doppler spectrum
    ind = np.nonzero(DopplerDown > PRF / 2)
    DopplerDown[ind] -= PRF
    ind = np.nonzero(DopplerDown < -PRF / 2)
    DopplerDown[ind] += PRF
    return DopplerCen, DopplerUp, DopplerDown, grazeOverRanges


def computeGrazingAngle(effAzIR, grazeIR, antPos, theRange, origin):
    # initialize the pointing vector to first range bin
    Rvec = np.array([np.cos(grazeIR) * np.sin(effAzIR),
                     np.cos(grazeIR) * np.cos(effAzIR),
                     -np.sin(grazeIR)])

    groundPoint = antPos + Rvec * theRange
    nlat, nlon, alt = enu2llh(*groundPoint, origin)
    # look up the height of the surface below the aircraft
    surfaceHeight = getElevation(nlat, nlon, False)
    # check the error in the elevation compared to what was calculated
    elevDiff = surfaceHeight - alt

    iterationThresh = 2
    heightDiffThresh = 1.0
    numIterations = 0
    newGrazeR = grazeIR + 0.0
    # iterate if the difference is greater than 1.0 m
    while abs(elevDiff) > heightDiffThresh and numIterations < iterationThresh:
        hAgl = antPos[2] + origin[2] - surfaceHeight
        newGrazeR = np.arcsin(hAgl / theRange)
        if np.isnan(newGrazeR) or np.isinf(newGrazeR):
            print('NaN or inf found.')
        Rvec = np.array([np.cos(newGrazeR) * np.sin(effAzIR),
                         np.cos(newGrazeR) * np.cos(effAzIR),
                         -np.sin(newGrazeR)])
        groundPoint = antPos + Rvec * theRange
        nlat, nlon, alt = enu2llh(*groundPoint, origin)
        surfaceHeight = getElevation(nlat, nlon, False)
        # check the error in the elevation compared to what was calculated
        elevDiff = surfaceHeight - alt
        numIterations += 1

    return newGrazeR, Rvec, surfaceHeight, numIterations


def complexMixDown(signalData, mixDownFrequency, srateHz):
    # Get the number of samples
    numSamples = len(signalData)
    # Create the time array
    timeS = np.arange(numSamples) / srateHz
    # Create the mix down signal
    mixDownSignal = np.exp(1j * 2 * np.pi * mixDownFrequency * timeS)

    return signalData * mixDownSignal


def upsamplePulse(p: np.ndarray, fft_len: int, upsample: int, is_freq: bool = False, out_freq: bool = False,
                  time_len: int = 0) -> np.ndarray:
    """
    Given an array of complex data, upsamples it in the frequency domain.
    :param p: Array of complex data. Can be in time domain or frequency domain, but should specify using is_freq.
    :param fft_len: Length of FFT to return. This is the base value and needs to be upsampled.
    :param upsample: Upsampling number.
    :param is_freq: If True, assumes the input p is in the frequency domain.
    :param out_freq: If True, returns the data still in the frequency domain. Otherwise, returns it to time domain and cuts it to time_len, if specified.
    :param time_len: If the user specifies out_freq=False, this is the length to cut the data when returned to the time domain. If not specified, just
        returns the whole thing.
    :return: Upsampled data.
    """
    tl = time_len if time_len else len(p)
    if len(p.shape) == 1:
        op = p if is_freq else np.fft.fft(p, fft_len)
        up = np.zeros(fft_len * upsample, dtype=op.dtype)
        up[:fft_len // 2] = op[:fft_len // 2]
        up[-fft_len // 2:] = op[-fft_len // 2:]
        up = np.fft.ifft(up)[:tl * upsample] if not out_freq else up
    else:
        op = p if is_freq else np.fft.fft(p, axis=-1)
        up = np.zeros((*op.shape[:-1], fft_len * upsample), dtype=op.dtype)
        up[..., :fft_len // 2] = op[..., :fft_len // 2]
        up[..., -fft_len // 2:] = op[..., -fft_len // 2:]
        up = np.fft.ifft(up, axis=-1)[..., :tl * upsample] if not out_freq else up
    return up
