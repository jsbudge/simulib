from io import BytesIO
import numpy as np
import requests
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
from PIL import Image
from importlib import resources as impresources
from simulib import geoids

pio.renderers.default = 'browser'

WGS_A = 6378137.0
WGS_F = 1 / 298.257223563
WGS_B = 6356752.314245179
WGS_E2 = 6.69437999014e-3
c0 = 299792458.0
DTR = np.pi / 180
DAC_FREQ_HZ = 4e9
BASE_COMPLEX_SRATE_HZ = DAC_FREQ_HZ / 2


def getDTEDName(lat, lon):
    """Return the path and name of the dted to load for the given lat/lon"""
    tmplat = int(np.floor(lat))
    tmplon = int(np.floor(lon))
    direw = 'w' if tmplon < 0 else 'e'
    dirns = 's' if tmplat < 0 else 'n'
    if os.name == 'nt':
        return 'E:\\dted\\%s%03d\\%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))
    else:
        return '/data1/dted/%s%03d/%s%02d.dt2' % (direw, abs(tmplon), dirns, abs(tmplat))


def detect_local_extrema(arr):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = filters.minimum_filter(arr, footprint=neighborhood) == arr
    # local_max = filters.maximum_filter(arr, footprint=neighborhood) == arr
    background = arr == 0
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_extrema = local_min ^ eroded_background  # + local_max ^ eroded_background
    return np.where(detected_extrema)


def db(x):
    ret = abs(x)
    if isinstance(ret, np.ndarray):
        ret[ret < 1e-15] = 1e-15
    else:
        ret = min(ret, 1e-15)
    return 20 * np.log10(ret)


def findPowerOf2(x):
    return int(2 ** (np.ceil(np.log2(x))))


def undulationEGM96(lat, lon):
    inp_file = (impresources.files(geoids) / 'EGM96.DAT')
    with inp_file.open("rb") as f:  # or "rt" as text file with universal newlines
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


def getElevationMap(lats, lons, und=True, interp_method='linear'):
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

        hght[idx] = interpn(np.array([np.arange(3601), np.arange(3601)]), data, np.array([x, y]).T,
                            method=interp_method, bounds_error=False, fill_value=0) + undulationEGM96(lats[idx], lons[
            idx]) if und else hght

    return hght


def getElevation(lat, lon, und=True):
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


def llh2enu(lat, lon, h, refllh):
    ecef = llh2ecef(lat, lon, h)
    return ecef2enu(*ecef, refllh)


def enu2llh(e, n, u, refllh):
    ecef = enu2ecef(e, n, u, refllh)
    return ecef2llh(*ecef)


def enu2ecef(e, n, u, refllh):
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


def llh2ecef(lat, lon, h):
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


def ecef2llh(x, y, z):
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


def ecef2enu(x, y, z, refllh):
    latr = refllh[0] * np.pi / 180
    lonr = refllh[1] * np.pi / 180
    rx, ry, rz = llh2ecef(*refllh)
    rot = np.array([[-np.sin(lonr), np.cos(lonr), 0],
                    [-np.sin(latr) * np.cos(lonr), -np.sin(latr) * np.sin(lonr), np.cos(latr)],
                    [np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)]])
    enu = rot.dot(np.array([x - rx, y - ry, z - rz]))
    return enu[0], enu[1], enu[2]


def genPulse(phase_x, phase_y, nnr, nfs, nfc, bandw):
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), phase_x, phase_y)
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * 1 / nfs))


def genChirp(nnr, nfs, nfc, bandw):
    phase = nfc - bandw // 2 + bandw * np.interp(np.linspace(0, 1, nnr), np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    return np.exp(1j * 2 * np.pi * np.cumsum(phase * 1 / nfs))


def rotate(az, nel, rot_mat):
    return rot.from_euler('zx', [[-az, 0.], [0., nel - np.pi / 2]]).apply(rot_mat)


def azelToVec(az, el):
    return np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])


def hornPattern(fc, width, height, theta=None, phi=None, deg_per_bin=.5, az_only=False):
    _lambda = c0 / fc
    d = _lambda / 2.
    if theta is None:
        theta = np.arange(0, np.pi, deg_per_bin * DTR)
    if phi is None:
        phi = [0] if az_only else np.arange(-np.pi / 2, np.pi / 2, deg_per_bin * DTR)
    theta, phi = np.meshgrid(theta, phi)
    lcw = np.arange(-width / 2, width / 2, d)
    lch = np.arange(-height / 2, height / 2, d)
    lch, lcw = np.meshgrid(lch, lcw)
    lchm = lch.flatten()
    lcwm = lcw.flatten()
    k = 2 * np.pi / _lambda
    locs = np.array([lcwm, np.zeros_like(lcwm), lchm]).T
    ublock = -azelToVec(theta.flatten(), phi.flatten())
    AF = np.sum(np.exp(-1j * k * locs.dot(ublock)), axis=0)
    AF = AF.flatten() if az_only else AF.reshape(theta.shape)

    # Return degree array and antenna pattern
    return theta, phi, AF


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


def arrayFactor(fc, pos, theta=None, phi=None, weights=None, deg_per_bin=.5, az_only=False, horn_dim=None,
                horn_pattern=None):
    use_pat = False
    if horn_pattern is not None:
        _, _, el_pat = horn_pattern
        use_pat = True
    elif horn_dim is not None:
        use_pat = True
        _, _, el_pat = hornPattern(fc, horn_dim[0], horn_dim[1], theta=theta, phi=phi,
                                   deg_per_bin=deg_per_bin, az_only=az_only)
    _lambda = c0 / fc
    if theta is None:
        theta = np.arange(0, np.pi, deg_per_bin * DTR)
    if phi is None:
        phi = [0] if az_only else np.arange(-np.pi / 2, np.pi / 2, deg_per_bin * DTR)
    theta, phi = np.meshgrid(theta, phi)
    k = 2 * np.pi / _lambda
    # az, el = np.meshgrid(theta, theta)
    ublock = -azelToVec(theta.flatten(), phi.flatten())
    AF = np.exp(-1j * k * pos.dot(ublock))
    if use_pat:
        AF *= el_pat.flatten()[None, :]
    weights = weights if weights is not None else np.ones(pos.shape[0])
    AF = AF.T.dot(weights).flatten() if az_only else AF.T.dot(weights).reshape(theta.shape)
    # Return degree array and antenna pattern
    return theta, phi, AF


def factors(n):
    return list(set(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0))))


def genTaylorWindow(baseband_fc: float, half_bw: float, fs: float, fft_len: int, nbar: int = 5,
                    sll: float = -35) -> np.array:
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


def GetAdvMatchedFilter(chan, nbar=5, SLL=-35, sar=None, pulseNum=20, fft_len=None):
    # Things the PS will need to know from the configuration
    numSamples = chan.nsam
    samplingFreqHz = chan.fs
    basebandedChirpRateHzPerS = chan.chirp_rate
    # If the NCO was positive it means we will have sampled the reverse spectrum
    #   and the chirp will be flipped
    if chan.NCO_freq_Hz > 0:
        basebandedChirpRateHzPerS *= -1
    halfBandwidthHz = chan.bw / 2.0
    # Get the basebanded center, start and stop frequency of the chirp
    basebandedCenterFreqHz = chan.baseband_fc
    basebandedStartFreqHz = chan.baseband_fc - halfBandwidthHz
    basebandedStopFreqHz = chan.baseband_fc + halfBandwidthHz
    if basebandedChirpRateHzPerS < 0:
        basebandedStartFreqHz = chan.baseband_fc + halfBandwidthHz
        basebandedStopFreqHz = chan.baseband_fc - halfBandwidthHz

    # Get the reference waveform and mix it down by the NCO frequency and
    #   downsample to the sampling rate of the receive data if necessary
    # The waveform input into the DAC has already had the Hilbert transform
    #   and downsample operation performed on it by SDRParsing, so it is
    #   complex sampled data at this point at the SlimSDR base complex sampling
    #   rate.
    # Compute the decimation rate if the data has been low-pass filtered and
    #   downsampled
    decimationRate = 1
    if chan.is_lpf:
        decimationRate = int(np.floor(BASE_COMPLEX_SRATE_HZ / samplingFreqHz))

    # Grab the waveform
    waveformData = chan.ref_chirp

    # Create the plot for the FFT of the waveform
    waveformLen = len(waveformData)

    # Compute the mixdown signal
    mixDown = np.exp(1j * (2 * np.pi * chan.NCO_freq_Hz * np.arange(waveformLen) / BASE_COMPLEX_SRATE_HZ))
    basebandWaveform = mixDown * waveformData

    # Decimate the waveform if applicable
    if decimationRate > 1:
        basebandWaveform = basebandWaveform[:: decimationRate]
    # Calculate the updated baseband waveform length
    basebandWaveformLen = len(basebandWaveform)
    # Grab the calibration data
    calData = chan.cal_chirp + 0.0
    # Grab the pulses
    if sar:
        calData = sar.getPulse(pulseNum, channel=0).T + 0.0

    # Calculate the convolution length
    convolutionLength = numSamples + basebandWaveformLen - 1
    FFTLength = findPowerOf2(convolutionLength) if fft_len is None else fft_len

    # Calculate the inverse transfer function
    FFTCalData = np.fft.fft(calData, FFTLength)
    FFTBasebandWaveformData = np.fft.fft(basebandWaveform, FFTLength)
    inverseTransferFunction = FFTBasebandWaveformData / FFTCalData
    # NOTE! Outside of the bandwidth of the signal, the inverse transfer function
    #   is invalid and should not be viewed. Values will be enormous.

    # Generate the Taylor window
    TAYLOR_NBAR = 5
    TAYLOR_NBAR = nbar
    TAYLOR_SLL_DB = -35
    TAYLOR_SLL_DB = SLL
    windowSize = \
        int(np.floor(halfBandwidthHz * 2.0 / samplingFreqHz * FFTLength))
    taylorWindow = window_taylor(windowSize, nbar=TAYLOR_NBAR, sll=TAYLOR_SLL_DB) if SLL != 0 else np.ones(windowSize)

    # Create the matched filter and polish up the inverse transfer function
    matchedFilter = np.fft.fft(basebandWaveform, FFTLength)
    # IQ baseband vs offset video
    if np.sign(basebandedStartFreqHz) != np.sign(basebandedStopFreqHz):
        # Apply the inverse transfer function
        aboveZeroLength = int(np.ceil((basebandedCenterFreqHz + halfBandwidthHz) / samplingFreqHz * FFTLength))
        belowZeroLength = int(windowSize - aboveZeroLength)
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[int(FFTLength / 2) - aboveZeroLength:int(FFTLength / 2) - aboveZeroLength + windowSize] = \
            taylorWindow
        # Zero out the invalid part of the inverse transfer function
        inverseTransferFunction[aboveZeroLength: -belowZeroLength] = 0
        taylorWindowExtended = np.fft.fftshift(taylorWindowExtended)
    else:
        # Apply the inverse transfer function
        bandStartInd = \
            int(np.floor((basebandedCenterFreqHz - halfBandwidthHz) / samplingFreqHz * FFTLength))
        taylorWindowExtended = np.zeros(FFTLength)
        taylorWindowExtended[bandStartInd: bandStartInd + windowSize] = taylorWindow
        inverseTransferFunction[: bandStartInd] = 0
        inverseTransferFunction[bandStartInd + windowSize:] = 0
    matchedFilter = matchedFilter.conj() * inverseTransferFunction * taylorWindowExtended
    return matchedFilter


def getRadarCoeff(fc, ant_transmit_power, rx_gain, tx_gain, rec_gain):
    return (c0 ** 2 / fc ** 2 * ant_transmit_power * 10 ** ((rx_gain + 2.15) / 10) * 10 ** ((tx_gain + 2.15) / 10) *
     10 ** ((rec_gain + 2.15) / 10) / (4 * np.pi) ** 3)


def window_taylor(N, nbar=4, sll=-30):
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


def antennaGain(N, wN, Nsub, wNsub, Nel, wNel, a, b, d, lamda):
    """
    Inputs are x, y, and z components of the intertial Range vector, which
    points from the aircraft to the point target (full magnitude, not normalized),
    as well as the rotation matrices from inertial-to-body,
    body-to-perpindicular body, and perpindicular body-to-antenna frames.
    As well as the weights for the Tx array, Rx array, and elevation array and
    the number of elements in the Tx, Rx, and Elevation arrays, and
    width (a) and height (b) of a single element, and the wavenumber.
    """
    k = 2 * np.pi / lamda
    thetat = 0.0
    thetar = 0.0
    phit = 0.0
    phir = 0.0
    cttx = np.cos(thetat)
    sttx = np.sin(thetat)
    ctrx = np.cos(thetar)
    strx = np.sin(thetar)
    cptx = np.cos(phit)
    sptx = np.sin(phit)
    cprx = np.cos(phir)
    sprx = np.sin(phir)
    E_i = np.sinc(a * k * sttx * cptx / (2 * np.pi)) * np.sinc(b * k * sttx * sptx / (2 * np.pi))
    # compute the Tx array factor
    AFtx = 0.0
    AFtx2 = 0.0
    for i in range(int(N / 2)):
        AFtx += 2 * wN[i]
        AFtx2 += 2 * (wN[i] ** 2)
    # compute the Rx array factor for the sub-array
    AFsub = 0.0
    AFsub2 = 0.0
    for i in range(int(Nsub / 2)):
        AFsub += 2 * wNsub[i]
        AFsub2 += 2 * (wNsub[i] ** 2)
    # compute the Elevation array factor
    AFel = 0.0
    AFel2 = 0.0
    for i in range(int(Nel / 2)):
        AFel += 2 * wNel[i]
        AFel2 += 2 * (wNel[i] ** 2)

    # let's try the solution from the other PDF from online
    DFtx = 0.0
    DFel = 0.0
    DFde = 0.0
    for i in range(int(N / 2)):
        for j in range(int(Nel / 2)):
            DFtx += 2 * (wN[i] * wNel[j]) * (((i * 2 + 1) / 2.0) ** 2)
            DFel += 2 * (wN[i] * wNel[j]) * (((j * 2 + 1) / 2.0) ** 2)
            DFde += 2 * wN[i] * wNel[j]
    DFtx = np.sqrt(DFtx)
    DFel = np.sqrt(DFel)

    MTI_DF = 8 * np.pi ** 2 * d * d * DFtx * DFel / (DFde * lamda ** 2)

    # let's try the solution from the other PDF from online
    DFsub = 0.0
    DFel = 0.0
    DFde = 0.0
    for i in range(int(Nsub / 2)):
        for j in range(int(Nel / 2)):
            DFsub += 2 * (wNsub[i] * wNel[j]) * ((i * 2 + 1) / 2.0) ** 2
            DFel += 2 * (wNsub[i] * wNel[j]) * ((j * 2 + 1) / 2.0) ** 2
            DFde += 2 * wNsub[i] * wNel[j]

    DFsub = np.sqrt(DFsub)
    DFel = np.sqrt(DFel)

    SAR_DF = 8 * np.pi ** 2 * d * d * DFsub * DFel / (DFde * lamda ** 2)

    # combine the element electric field with all of the array factors to obtain
    # the two way normalized radiation pattern
    MTI_D = AFtx ** 2 * AFel ** 2 / (AFtx2 * AFel2)
    SAR_D = AFsub ** 2 * AFel ** 2 / (AFsub2 * AFel2)
    return MTI_D, SAR_D, MTI_DF, SAR_DF


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


def getDopplerLine(effAzI, rangeBins, antVel, antPos, nearRangeGrazeR, azBeamwidthHalf, PRF, wavelength, origin):
    """Compute the expected Doppler vs range for the given platform geometry"""

    # compute the grazing angle for the near range to start
    (nearRangeGrazeR, Rvec, surfaceHeight, numIter) = computeGrazingAngle(
        effAzI, nearRangeGrazeR, antPos, rangeBins[0], origin)

    # now I need to get the grazing angles across all of the range bins
    grazeOverRanges = np.arcsin((antPos[2] + origin[2] - surfaceHeight) / rangeBins)

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


def getTimeDelayS(
        refDat, secDat, pulseLengthS, chirpRateHzPerS, srateHz, offset):
    # Compute the pulse length in samples
    pulseLengthN = int(pulseLengthS * srateHz)
    cutStart = offset
    cutEnd = cutStart + pulseLengthN - offset * 2
    choppedRef = refDat[cutStart: cutEnd].copy() + 0.0
    choppedSec = secDat[cutStart: cutEnd].copy() + 0.0
    # Look at phase difference between reference and secondary channels
    unwrappedPhaseDif = np.unwrap(np.angle(choppedRef / choppedSec))
    # Get a least squares order 1 polynomial fit of the phase difference
    times = np.arange(choppedRef.shape[0]) / srateHz
    polynomials = np.polyfit(times, unwrappedPhaseDif, 1)
    # Attempt to calculate the time delay based on the delta phase per delta
    #   freq
    tau = polynomials[0] / (2 * np.pi * chirpRateHzPerS)
    print("Time delay tau: %0.3f ps" % (tau / 1e-12))

    """ Now we need to apply the time shift, then compute residual phase. """
    # Prepare for the FFT of the pulses by computing the next power of 2
    fftLength = int(2 ** np.ceil(np.log2(choppedRef.shape[0])))
    # Also estimate the residual remaining phase and return the amplitude bias
    secondaryFreq = np.fft.fftshift(np.fft.fft(choppedSec, fftLength))
    # Generate frequencies for the spectrum
    frequenciesHz = np.arange(fftLength) / fftLength * srateHz
    omegaK = 2 * np.pi * frequenciesHz
    # Apply time shift to the secondary pulse in the frequency domain
    secShiftFreq = secondaryFreq * np.exp(1j * omegaK * tau)
    # IFFT back to the time domain
    secShift = np.fft.ifft(
        np.fft.ifftshift(secShiftFreq))[:choppedRef.shape[0]]
    # Compute the number of sample shifts from the time delay
    shiftSamplesN = int(tau * srateHz)
    chopLopRef = secShift[:shiftSamplesN]
    chopLopSec = secShift[:shiftSamplesN]
    # Compute the new unwrapped phase difference
    unwrappedPhaseDifC = np.unwrap(np.angle(chopLopRef / chopLopSec))
    meanPhaseOffset = unwrappedPhaseDifC.mean()
    print("Residual phase offset: %0.3f deg" % (
            meanPhaseOffset * 180 / np.pi))
    # Apply the mean phase offset and IFFT
    secShiftFreq = \
        secondaryFreq * np.exp(1j * (omegaK * tau + meanPhaseOffset))
    secShift = np.fft.ifft(
        np.fft.ifftshift(secShiftFreq))[:choppedRef.shape[0]]
    # Get the mean amplitutde bias
    amplitudeBias = abs(choppedRef).mean() / abs(secShift).mean()
    print("Amplitude bias: %0.5f" % amplitudeBias)

    return tau, meanPhaseOffset, amplitudeBias


def applyPulseCorrections(
        refPulse, timeDelayS, residualPhase, ampBias, fftLength, numSamples,
        srateHz):
    # Compute the FFT of the ref pulse
    freqRefPulse = np.fft.fftshift(
        np.fft.fft(refPulse * ampBias, fftLength))
    frequenciesHz = np.arange(fftLength) / fftLength * srateHz
    omegaK = 2 * np.pi * frequenciesHz
    freqFixedRef = \
        freqRefPulse * np.exp(1j * (omegaK * timeDelayS + residualPhase))
    fixedRef = np.fft.ifft(
        np.fft.ifftshift(freqFixedRef))[:numSamples]
    corrections = \
        ampBias * np.exp(1j * (omegaK * timeDelayS + residualPhase))

    return fixedRef, corrections


GECOEFF = 156543.03392
TILE_SIZE = 256


def getGoogleMap(lat, lon, mpp, im_width=10, im_height=10):
    zoomlevel = int(np.log2(GECOEFF * np.cos(lat * np.pi / 180.) / mpp))
    gmpp = GECOEFF * np.cos(lat * np.pi / 180.) / (2 ** zoomlevel)

    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    scale = 1 << zoomlevel

    # Convert the latitude to radians and take the sine
    sy = np.sin(lat * np.pi / 180.0)
    world_x = TILE_SIZE * (.5 + lon / 360.)
    world_y = TILE_SIZE * (.5 - np.log((1 + sy) / (1 - sy)) / (4 * np.pi))

    pixel_x = np.floor(world_x * scale)
    pixel_y = np.floor(world_y * scale)

    corner_x = np.floor(pixel_x - (im_height / 2) / gmpp)
    corner_y = np.floor(pixel_y - (im_height / 2) / gmpp)

    tile_x = corner_x // TILE_SIZE
    tile_y = corner_y // TILE_SIZE

    start_x = int(tile_x)
    start_y = int(tile_y)
    pix_shift_x = int((corner_x / TILE_SIZE - start_x) * TILE_SIZE)
    pix_shift_y = int((corner_y / TILE_SIZE - start_y) * TILE_SIZE)

    tile_width = int(np.ceil((im_width / gmpp) / TILE_SIZE)) + 1
    tile_height = int(np.ceil((im_height / gmpp) / TILE_SIZE)) + 1
    im_shift_x = int(im_width / gmpp)
    im_shift_y = int(im_height / gmpp)

    # Determine the size of the image in pixels
    width, height = TILE_SIZE * tile_width, TILE_SIZE * tile_height

    # Create a new image of the size require
    map_img = Image.new('RGB', (width, height))

    for x, y in product(range(tile_width), range(tile_height)):
        url = f'https://mt1.google.com/vt?lyrs=s&x={start_x + x}&y={start_y + y}&z={zoomlevel}'

        response = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0'})

        im = Image.open(BytesIO(response.content))
        map_img.paste(im, (x * TILE_SIZE, y * TILE_SIZE))

    map_img = map_img.crop((pix_shift_x, pix_shift_y, pix_shift_x + im_shift_x, pix_shift_y + im_shift_y))

    return map_img


def resampleGoogleMap(lats, lons, mpp):
    center_lat = (lats.max() + lats.min()) / 2
    center_lon = (lons.max() + lons.min()) / 2
    zoomlevel = int(np.log2(GECOEFF * np.cos(center_lat * np.pi / 180.) / mpp))
    gmpp = GECOEFF * np.cos(center_lat * np.pi / 180.) / (2 ** zoomlevel)

    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    scale = 1 << zoomlevel

    box_coords = np.array([lats, lons]).T
    pix_coords = np.zeros_like(box_coords)
    sy = np.sin(box_coords[:, 0] * np.pi / 180.0)
    pix_coords[:, 0] = np.floor(TILE_SIZE * (.5 + box_coords[:, 1] / 360.) * scale)
    pix_coords[:, 1] = np.floor(TILE_SIZE * (.5 - np.log((1 + sy) / (1 - sy)) / (4 * np.pi)) * scale)

    tile_coords = pix_coords // TILE_SIZE

    start_x = int(tile_coords[:, 0].min())
    start_y = int(tile_coords[:, 1].min())

    tile_width = int(tile_coords[:, 0].max() - tile_coords[:, 0].min()) + 1
    tile_height = int(tile_coords[:, 1].max() - tile_coords[:, 1].min()) + 1

    # Determine the size of the image in pixels
    width, height = TILE_SIZE * tile_width, TILE_SIZE * tile_height

    # Create a new image of the size require
    map_img = Image.new('RGB', (width, height))

    for x, y in product(range(tile_width), range(tile_height)):
        url = f'https://mt1.google.com/vt?lyrs=s&x={start_x + x}&y={start_y + y}&z={zoomlevel}'
        response = requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0'})

        im = Image.open(BytesIO(response.content))
        map_img.paste(im, (x * TILE_SIZE, y * TILE_SIZE))

    # Get interpolated values from image for lat/lons
    im_x = pix_coords[:, 0] - start_x * TILE_SIZE
    im_y = pix_coords[:, 1] - start_y * TILE_SIZE
    return map_img, im_x, im_y


def upsamplePulse(p, fft_len, upsample, is_freq=False, out_freq=False, time_len=0):
    if len(p.shape) == 1:
        op = p if is_freq else np.fft.fft(p, fft_len)
        up = np.zeros(fft_len * upsample, dtype=op.dtype)
        up[:fft_len // 2] = op[:fft_len // 2]
        up[-fft_len // 2:] = op[-fft_len // 2:]
        up = np.fft.ifft(up)[:time_len * upsample] if not out_freq else up
    else:
        op = p if is_freq else np.fft.fft(p, axis=1)
        up = np.zeros((op.shape[0], fft_len * upsample), dtype=op.dtype)
        up[:, :fft_len // 2] = op[:, :fft_len // 2]
        up[:, -fft_len // 2:] = op[:, -fft_len // 2:]
        up = np.fft.ifft(up, axis=1)[:, :time_len * upsample] if not out_freq else up
    return up
