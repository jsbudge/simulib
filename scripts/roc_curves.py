# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:08:29 2014

@author: Josh Bradley

@purpose: To estimate and calculate parameters for the Ka-band radar we are
going to make someday.
"""
from numpy import *
from scipy.signal.windows import chebwin, exponential, blackman, hann
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from simulation_functions import window_taylor, antennaGain, marcumq, getDoppResolution
import yaml
import warnings


def calculateMDV(velP, azWidth, graze, squint, Kexo):
    phiFore = max(squint - Kexo * azWidth / (2 * cos(graze)), 0)
    phiAft = min(squint + Kexo * azWidth / (2 * cos(graze)), pi)

    MDVapproach = velP * cos(graze) * (cos(phiFore) - cos(squint))
    MDVrecede = velP * cos(graze) * (cos(phiAft) - cos(squint))

    MDV = max(abs(MDVapproach), abs(MDVrecede))

    return MDV, MDVapproach, MDVrecede


def dbtonat(x):
    return 10 ** (x / 10.)


def genPdreq0(alpha, req_fa0, array_sz):
    return array([marcumq(alpha[i], sqrt(-2 * log(req_fa0)), 60.0, 10000)
                  for i in range(array_sz)])


# We want a particular font size for the plots
plt.rc('font', size=16)
# Close all open figures
plt.close('all')

# physical constants
c0 = 299792458  # speed of light m/s
DTR = pi / 180.0  # degrees to radians conversion
RTD = 180.0 / pi  # radians to degrees conversion
ft2m = 0.3048  # feet to meter conversion
knts2m = .514444  # knots to m/s
kb = 1.3806503e-23  # Boltzmans constant
eta_rad = 0.7  # assumed radiation efficiency
mph2mps = 0.44704  # conversion from mph to m/s
twoWayFreeAirdBPerkm = 0.0198

# Radar constants
SAMPLE_SIZE_BYTES = 2
COMPLEX_BYTES = 2

if __name__ == '__main__':

    with open('./roc_curve_settings.yaml') as y:
        settings = yaml.safe_load(y.read())
    # Choose a sample rate that satisfies Nyquist for the BW
    reqSampleRate = (settings['radar_params']['BW'] + 5e6) * 2.1 if settings['radar_params']['OFFSET_VIDEO'] else (
            settings['radar_params']['BW'] * 1.1)
    SRATE = 15.625e6 * 2 ** sum(reqSampleRate - 15.625e6 * 2 ** arange(7) > 0)

    # Calculated Radar Constants
    rng_rez = c0 / SRATE  # range resolution
    plat_vel = settings['radar_params']['Vel'] * knts2m
    hAgl = settings['radar_params']['hAgl'] * ft2m
    # meters per sample (sampled range resolution)
    actual_range_resolution = c0 / (2 * settings['radar_params']['BW']) * 1.2
    lamda = c0 / settings['radar_params']['fc']  # wavelength in meters (at center frequency)
    lamda_start = c0 / (settings['radar_params']['fc'] - settings['radar_params']['BW'] / 2.0)
    lamda_stop = c0 / (settings['radar_params']['fc'] + settings['radar_params']['BW'] / 2.0)
    Dop_mbsl = 4 * plat_vel / lamda  # Mainbeam + Sidelobe clutter Doppler spread

    ################################################################################
    # Antenna calculations
    ################################################################################
    # Antenna Parameters
    d = lamda / 2.0  # antenna element spacing (m)
    L = settings['radar_params']['tx_array_n'] * d  # length of antenna array (m)
    Nsub = settings['radar_params']['tx_array_n'] // settings['radar_params'][
        'Nchan']  # number of array elements in sub-array (on Rx)
    Lsub = Nsub * d  # sub-array length
    dsub = Nsub * d
    Lel = settings['radar_params']['el_array_n'] * d  # height of antenna array (m)
    width = lamda / 4.0  # element width (m)
    height = lamda / 4.0  # element height (m)
    print("center frequency: %0.2f GHz" % (settings['radar_params']['fc'] / 1e9))
    print("wavelength: %f cm" % (lamda * 100))
    print("bandwidth: %0.2f MHz" % (settings['radar_params']['BW'] / 1e6))
    print("Transmit Power: %0.1f watts" % (settings['radar_params']['Pt']))
    print("Noise Figure: %0.2f" % (dbtonat(settings['radar_params']['F'])))
    print("CPI size: %d" % settings['Ncpi'])
    print("Height above ground: %0.1f ft" % (settings['radar_params']['hAgl']))
    print("Max Duty Cycle: %0.1f %%" % (settings['radar_params']['maxDutyCycle'] * 100))
    print("Platform Velocity: %0.1f knots" % (settings['radar_params']['Vel']))
    print("Sample Rate: %0.2f MHz" % (SRATE / 1e6))

    # Create the weights for the azimuth elements of the full Tx antenna
    if settings['tx_weight_type'] == 'ones':
        tW = ones(settings['radar_params']['tx_array_n'])
    else:
        tW = window_taylor(settings['radar_params']['tx_array_n'], 5, -30)
    tW = tW[settings['radar_params']['tx_array_n'] // 2:]
    # Create the weights for the azimuth elements of the Rx sub-antennas
    if settings['rx_weight_type'] == 'cheb':
        subW = chebwin(Nsub, -30)
    elif settings['rx_weight_type'] == 'ones':
        subW = ones(Nsub)
    else:
        subW = exponential(Nsub, tau=Nsub / 2.0)
    subW = subW[Nsub // 2:]
    # Create the weights for the elevation elements of the Tx/Rx antenna
    if settings['el_weight_type'] == 'kaiser':
        elW = kaiser(settings['radar_params']['el_array_n'], 1.5)
    else:
        elW = ones(settings['radar_params']['el_array_n'])
    elW = elW[settings['radar_params']['el_array_n'] // 2:]

    # attempt to calculate the gain of the MTI and SAR mode antennas
    MTI_D, SAR_D, MTI_DF, SAR_DF = antennaGain(settings['radar_params']['tx_array_n'], tW, Nsub, subW,
                                               settings['radar_params']['el_array_n'], elW, width, height, d, lamda)
    if settings['radar_params']['calc_antenna']:
        TxGain = MTI_D * eta_rad
        RxGain = SAR_D * eta_rad
    else:
        TxGain = dbtonat(settings['radar_params']['ant_gain_db'])
        RxGain = dbtonat(settings['radar_params']['ant_gain_db'])
    # RxGain = TxGain

    # Define the antenna azimuth and elevation 3dB beamwidth in radians
    az_3dB = settings['radar_params']['az_bw'] * DTR
    el_3dB = settings['radar_params']['el_bw'] * DTR

    print("TxGain: %0.3f dB" % (10 * log10(TxGain)))
    print("RxGain: %0.3f dB" % (10 * log10(RxGain)))
    print("Combined Tx/Rx Gain: %0.3f dB" % (10 * log10(TxGain * RxGain)))

    ################################################################################
    # Geometry and timing calculations
    ################################################################################
    # Calculate the depression angle at the maximum slant range for the far grazing
    farGraze = arcsin(hAgl / settings['radar_params']['maxSlantRange'])
    # Get the near range grazing angle
    nearGraze = farGraze + el_3dB
    # Get the depression angle for the center of the beam
    depAng = (farGraze + nearGraze) / 2.0
    # For the nominal range, use the range at the center depression angle
    centerRange = hAgl / sin(depAng)
    # Calculate the near and far ranges using the associated grazing angles
    nearRange = hAgl / sin(nearGraze)  # near range (meters)
    farRange = hAgl / sin(farGraze)  # far range (meters)
    # Compute the ambiguous range
    unambiguousRange = farRange * 1.5
    # ambiguous_range = (T_r + (rng_gate - tp)) * c0
    unambiguousGrazAngle = arcsin(hAgl / unambiguousRange)
    # the time to the unambiguous range requirement
    uarTimeReq = unambiguousRange * 2.0 / c0
    # Calculate the slant range swath
    slantRangeSwath = farRange - nearRange
    # Compute the ground range swath
    groundRangeSwath = hAgl * (1 / tan(farGraze) - 1 / tan(nearGraze))
    # Compute the number of range samples for the swath
    nominalNumRangeBins = int(slantRangeSwath * 2 / rng_rez)

    # Determine radar timing parameters

    # Define the circuit protection time
    ampPowerUpTime = 1000 / settings['radar_params']['TAC']
    rxOffTime = 100 / settings['radar_params']['TAC']
    receiverProtectionTime = ampPowerUpTime + rxOffTime

    # The ADC on time to the integer TAC
    adcOn = round(nearRange * 2.0 / c0 * settings['radar_params']['TAC']) / settings['radar_params']['TAC']
    # Calculate the pulse length first
    tp = nearRange * 2.0 / c0 * settings['radar_params']['pulse_length_percent']
    if tp > settings['radar_params']['max_pulse_length']:
        warnings.warn(f"Possible pulse time is greater than max_pulse_length. "
                      f"Being reset to {settings['radar_params']['max_pulse_length']}.",
                      RuntimeWarning)
        tp = settings['radar_params']['max_pulse_length']

    # Calculate the PRI for our desired unambiguous velocity
    PRF = 2 * settings['radar_params']['unambVelDesired'] * mph2mps * cos(farGraze) * 2 / lamda
    # Get the PRI to an integer TAC
    PRI = round(1 / PRF * settings['radar_params']['TAC']) / settings['radar_params']['TAC']
    # If the PRI based on this calculation is not larger than the unambiguous range
    #   requirement, then we need to increase the PRI to unambiguously out to that
    #   range
    if PRI < uarTimeReq - adcOn + tp:
        PRI = max(
            round((uarTimeReq - adcOn + tp + receiverProtectionTime) *
                  settings['radar_params']['TAC']) / settings['radar_params']['TAC'],
            round((2 * farRange / c0 + tp + receiverProtectionTime) *
                  settings['radar_params']['TAC']) / settings['radar_params']['TAC'])
        warnings.warn(f"WARNING! PRI was too short for the unambiguous range requirement. "
                      f"Being reset to {PRI}.",
                      RuntimeWarning)
    PRF = 1 / PRI

    # Check to make sure that the duty cycle isn't too high
    if tp / PRI > settings['radar_params']['maxDutyCycle']:
        tp = settings['radar_params']['maxDutyCycle'] * PRI
        warnings.warn("*** WARNING! the duty cycle was too high. %0.2f***" % (tp / PRI), RuntimeWarning)

    dutyCycle = tp / PRI
    # Compute the adc off value
    adcOff = round((farRange * 2.0 / c0 + tp) * settings['radar_params']['TAC']) / settings['radar_params']['TAC']

    # Derived Scalars
    # Compute the chirp rate
    kr = settings['radar_params']['BW'] / tp
    # calculate the maximum unambiguous velocity (m/s)
    unambVel = PRF * lamda / 4.0

    # Calculate the minimum detectable velocities
    doerrySquintR = (90 - settings['squint']) * DTR
    # Calculate the minimum detectable velocities
    MDV, MDVapproach, MDVrecede = calculateMDV(
        plat_vel, az_3dB, farGraze, doerrySquintR, settings['radar_params']['Kexo'])
    clutterDopplerSpread = MDV * 4.0 / lamda
    print("The Doppler spread at MDV: %0.3f kHz" % (clutterDopplerSpread / 1e3))
    # MDV /= 4
    # When the clutter slope is greater than 1.0, then we have Doppler aliasing
    #   which significantly increases the rank of the clutter matrix
    LOSVelocityAtSquintMPerS = plat_vel * cos(settings['squint'] * DTR) * cos(farGraze)
    if settings['radar_params']['calc_antenna']:
        achievedClutterSlope = PRI * 2.0 * LOSVelocityAtSquintMPerS / Lsub
        print("The STAP clutter slope for the achievable PRF: %0.3f" % (
            achievedClutterSlope))
        # Compute the required Doppler sampling for clutter slope less than 1.0
        # desired slope of clutter ridge line
        beta = 1.0 / 1.0
        # This is based on the inter-channel spacing and velocity (I believe it
        #   also assumes a broadside geometry)
        unambiguousPRI = beta * Lsub / (2 * LOSVelocityAtSquintMPerS)
        print("PRF required for STAP clutter slope less than 1.0: %0.3f Hz" % (
                1 / unambiguousPRI))

    # Compute the first blind velocity
    blindVelocityMPerS = PRF * lamda / 2 - MDV

    print("pulse length: %0.5f us" % (tp * 1e6))
    print("PRF = %0.6f kHz" % (PRF / 1e3))
    print("PRI = %0.6f us" % (PRI * 1e6))
    print("duty cycle: %0.3f%%" % (dutyCycle * 100))
    print("chirp rate: %0.3f hz/s" % kr)
    print("adc_on: %0.3f us" % (adcOn * 1e6))
    print("adc_off: %0.3f us" % (adcOff * 1e6))
    print("unambiguous_range: %0.3f km" % (unambiguousRange / 1e3))
    print("unambiguous_range time: %0.3f us" % (uarTimeReq * 1e6))
    print("unambiguous grazing_angle: %0.3f deg" % (unambiguousGrazAngle / DTR))
    print("unambiguous velocity: %0.3f m/s" % unambVel)
    print("1st blind velocity: %0.3f m/s" % blindVelocityMPerS)
    print("MDV: %0.3f m/s" % MDV)
    print("Max target speed: %0.3f m/s" % (settings['target_params']['maxSpeed'] * knts2m))

    # use the near and far ranges to create our first and last range bins
    range_bins = arange(nearRange * 2, farRange * 2, rng_rez) / 2.0
    Nsam = range_bins.size

    # Calculate the number of receive samples
    RxOnS = nearRange * 2 / c0
    RxOffS = farRange * 2 / c0 + tp
    numSimultaneousChannels = 2
    numSamples = int((adcOff - adcOn) * SRATE) * numSimultaneousChannels
    samplesPerS = numSamples * PRF
    dataRateMBPerS = samplesPerS * SAMPLE_SIZE_BYTES * COMPLEX_BYTES / 1e6

    print("near range: %0.3f m\nfar range: %0.3f m" % (
        nearRange, farRange))
    print("near grazing angle: %0.3f deg\nfar grazing angle: %0.3f deg" % (
        nearGraze * RTD, farGraze * RTD))
    print("Depression Angle: %0.3f deg" % (depAng / DTR))
    print("range at center of beam: %0.3f m" % centerRange)
    print("range sample spacing: %0.4f m" % rng_rez)
    print("range resolution: %0.4f m" % actual_range_resolution)
    print("Range Swath: %0.3f km" % slantRangeSwath)
    print("Ground Range Swath: %0.3f km" % groundRangeSwath)
    print("Number of fast-time samples: %d" % numSamples)
    print("Number of Range bins: %d samples" % nominalNumRangeBins)
    print("Estimated Data Rate: %0.2f MBps" % dataRateMBPerS)

    # Determine the resolutions and accuracies (this entails range, velocity,
    #   azimuth and elevation angles)
    # 1) Compute ranging accuracy for different range resolutions (note that there
    #   is 15cm of range error due to the timeing error of the system. This should
    #   be added to the error in the straddling loss)
    rangeUncertaintyM = rng_rez / 2 + 0.15
    # 2) Velocity resolution
    scanExtent = settings['scan_extent'] * DTR
    broadside = settings['squint'] * DTR
    _, radialVelocityResolutionMPerS, _ = getDoppResolution(settings['doppler_broadening_factor'], PRF, lamda,
                                                            settings['Ncpi'])
    # Calculate an expected plane induced radial velocity at the max azimuth scan
    radVelAtScanExtentMPerS = plat_vel * sin(scanExtent) * cos(farGraze)
    worstAzimuthResolutionDeg = arcsin(radVelAtScanExtentMPerS / (plat_vel * cos(farGraze))) * RTD - arcsin(
        (radVelAtScanExtentMPerS - radialVelocityResolutionMPerS)
        / (plat_vel * cos(farGraze))) * RTD
    bestAzimuthResolutionDeg = radialVelocityResolutionMPerS / (plat_vel * cos(farGraze)) * RTD
    # 5) Azimuth angle uncertainty is directly related to the noise in the
    #   monopulse phase (let's say that is 25 deg)
    monopulsePhaseNoiseR = settings['monopulse_phase_noise'] * DTR
    azimuthUncertaintyDeg = (arcsin(broadside * lamda / (pi * L)) -
                             arcsin((broadside - monopulsePhaseNoiseR) * lamda / (pi * L))) * RTD
    # 6) update the radial velocity uncertainty with azimuth uncertaiinty
    #   by converting the azimuth uncertainty to a change in radial velocity
    bestRadialVelocityUncertaintyUpdateMPerS = (PRF * lamda / (4 * settings['Ncpi']) + plat_vel *
                                                cos(farGraze) * (sin(scanExtent) -
                                                                 sin(scanExtent - azimuthUncertaintyDeg * DTR)))
    worstRadialVelocityUncertaintyUpdateMPerS = (PRF * lamda / (4 * settings['Ncpi']) + plat_vel *
                                                 cos(farGraze) * (sin(broadside) - sin(broadside -
                                                                                       azimuthUncertaintyDeg * DTR)))

    # 7) Compute the effective elevation angle accuracy
    rdElevationUncertaintyDeg = tan(nearGraze) * rangeUncertaintyM / nearRange * RTD
    edElevationUncertaintyDeg = settings['dtedErrorM'] / (cos(nearGraze) * nearRange) * RTD
    elevationAngleUncertaintyDeg = rdElevationUncertaintyDeg + edElevationUncertaintyDeg
    # 8) Estimate the effective elevation angle resolution
    elevationAngleResolutionDeg = tan(nearGraze) * actual_range_resolution / nearRange * RTD

    print("@%0.0f MHz BW and %0.1f MHz sample rate..." % (settings['radar_params']['BW'] / 1e6, SRATE / 1e6))
    print("Range resolution: %0.2f m" % actual_range_resolution)
    print("Range uncertainty: %0.2f m" % rangeUncertaintyM)
    print("Effective elevation angle uncertainty @near-range: %0.4f deg" % (
        elevationAngleUncertaintyDeg))
    print("Effective elevation angle resolution @near-range: %0.4f deg" % (
        elevationAngleResolutionDeg))
    print("Radial Velocity Resolution: %0.2f m/s" % (
        radialVelocityResolutionMPerS))
    print("@%0.0f deg scan, radial velocity uncertainty: %0.2f m/s" % (
        scanExtent * RTD, bestRadialVelocityUncertaintyUpdateMPerS))
    print("@%0.0f deg scan, radial velocity uncertainty: %0.2f m/s" % (
        broadside * RTD, worstRadialVelocityUncertaintyUpdateMPerS))
    print("@%0.0f deg scan, Azimuth Resolution: %0.2f deg" % (
        scanExtent * RTD, worstAzimuthResolutionDeg))
    print("@%0.0f deg scan, Azimuth Resolution: %0.2f deg" % (
        broadside * RTD, bestAzimuthResolutionDeg))
    print("Azimuth uncertainty for %0.0f deg monopulse phase noise: %0.2f deg" % (
        monopulsePhaseNoiseR * RTD, azimuthUncertaintyDeg))

    # Create a plot of the timing of the radar.
    PATurnOnTime = 1000 / settings['radar_params']['TAC']
    txOn = PATurnOnTime
    txOff = (txOn + tp) / 1e-3
    rxOn = adcOn / 1e-3
    rxOff = adcOff / 1e-3
    farthestRangeTime = uarTimeReq / 1e-3
    farthestRangeTimePlusPulse = (uarTimeReq + tp) / 1e-3
    # Generate pulse plots
    pulsePlotX = array([
        txOn, txOn, txOff, txOff, rxOn, rxOn, rxOff, rxOff, farthestRangeTime,
        farthestRangeTime, farthestRangeTimePlusPulse,
        farthestRangeTimePlusPulse])
    pulsePlotY = array([0, 0.5, 0.55, 0, 0, 0.4, 0.4, 0, 0, 0.75, 0.75, 0])
    pulse2PlotX = pulsePlotX + PRI / 1e-3
    pulse3PlotX = pulse2PlotX + PRI / 1e-3
    plt.figure()
    plt.plot(pulsePlotX, pulsePlotY)
    plt.plot(pulse2PlotX, pulsePlotY)
    plt.plot(pulse3PlotX, pulsePlotY)
    plt.xlabel('Time (ms)')
    plt.ylabel('Nunya')
    plt.xlim([0, 18])
    plt.ylim([0, 1.05])

    ################################################################################
    # Target characteristics and Radar equation for SNR
    ################################################################################
    # Define the target parameters
    target_RCS = dbtonat(settings['target_params']['RCS'])
    R0 = settings['radar_params']['maxSlantRange']

    # Calculate the expected noise power
    #   the noise term is given by N0
    N0 = kb * settings['radar_params']['T0'] * dbtonat(settings['radar_params']['F'])
    NoisePow = N0 * settings['radar_params']['BW']
    # Compute the range compression gain
    compression_gain = tp * settings['radar_params']['BW']
    # We assume our receiver bandwidth is identical to our transmit pulse bandwidth
    #   since we are using a matched filter with window (pulse compression)
    SNR = (settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * target_RCS * compression_gain /
           ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * NoisePow * R0 ** 4))
    print("Expected SNR for a single pulse on a single channel for target_RCS "
          "of %0.1f at a range of %0.1f m: %0.3f dB" % (
              target_RCS, R0, 10 * log10(SNR)))

    # Calculate the time for a CPI
    Tcpi = settings['Ncpi'] * PRI
    print("The CPI time is %0.3f ms" % (Tcpi / 1e-3))

    ################################################################################
    # Let's make some calculations of the temporal decorrelation due to range walk
    ################################################################################
    # Compute the width of the azimuth lobe on the ground
    az_side = az_3dB * R0
    # Compute the area of the antenna range-azimuth patch on the ground
    az_rng_area = az_side * rng_rez
    # calculate the pulse range walk when pointed toward the nose
    az_rng_area_delta = (PRI * plat_vel) * az_side
    # Equate this toa pulse decorrelation
    pulse_decorrelation = 1 - az_rng_area_delta / az_rng_area
    print("pulse-2-pulse decorrelation: %f" % pulse_decorrelation)
    print("CPI decorrelation: %f" % (pulse_decorrelation ** (settings['Ncpi'] - 1)))
    print("range resolution: %0.6f, pulse range walk: %0.6f, CPI range walk: %0.6f" %
          (rng_rez, PRI * plat_vel, Tcpi * plat_vel))
    # let's compare the rate of change of the area due to range walk (caused by
    #   platform velocity) vs rate of change of the area due to antenna scanning
    #   (when pointed toward the nose)
    # Set our scan rate based on our CPI time for the time-on-target
    scan_rate = az_3dB / (2 * Tcpi)
    azimuth_scan_range = settings['az_scan_range'] * DTR
    mid_revisit_time = azimuth_scan_range / scan_rate
    max_revisit_time = azimuth_scan_range * 2 / scan_rate
    print("scan rate: %0.6f deg/s" % (scan_rate / DTR))
    print("revisit rate: <%0.3f sec" % max_revisit_time)
    print("effective ground scan rate: %0.3f m/s" % (scan_rate * R0))
    print("CPI azimuth ground walk: %0.6f m" % (scan_rate * R0 * Tcpi))
    print("Pulse-2-pulse azimuth ground walk: %0.6f m" % (scan_rate * R0 * PRI))

    # Calculate the CPI area coverage (just the az_rng_area)
    CPIAreaCoverageKM2 = az_side * groundRangeSwath / 1e3 ** 2
    # Calculate the number of those in a scan
    numBeamsPerScan = int(azimuth_scan_range / az_3dB)
    ScanAreaCoverageKM2 = CPIAreaCoverageKM2 * numBeamsPerScan
    print("CPI Coverage Area: %0.3f km^2" % CPIAreaCoverageKM2)
    print("GMTI Area Coverage for a scan: %0.3f km^2" % ScanAreaCoverageKM2)
    ################################################################################
    # Generate the ROC curves for different FA rates that show Pd vs target RCS
    #   for target range from above
    ################################################################################
    # required FA rate in FA/min
    req_fa_rate = array([0.01, 0.1, 1.0, 6.0])
    req_fa = req_fa_rate / 60.0 * Tcpi / (settings['Ncpi'] * nominalNumRangeBins)
    # req_fa = array([1e-6])
    # uncomment if employing consecutive CPI's to mitigate false alarms
    # req_fa = sqrt( req_fa )
    Pd = sqrt(settings['radar_params']['desiredProbDet'])
    # comment if employing consecutive CPI's to mitigate false alarms
    Pd = settings['radar_params']['desiredProbDet']
    assumedSINRLoss = 10.0 ** (settings['SINRLossdB'] / 10)
    STAPGain = settings['radar_params']['Nchan'] * assumedSINRLoss if settings['doSTAP'] else 1.0
    target_RCS_dB = linspace(-10, 20, 100)
    SNR_lin = (settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * dbtonat(
        target_RCS_dB) * compression_gain /
               ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * NoisePow * R0 ** 4))
    CPIGain_lin_snr = SNR_lin * settings['Ncpi'] * STAPGain
    alpha_req_snr = sqrt(2 * CPIGain_lin_snr)
    P_dreq_0 = genPdreq0(sqrt(2 * CPIGain_lin_snr), req_fa[0], SNR_lin.size)
    plt.figure('Pd vs. RCS')
    plt.plot(target_RCS_dB, P_dreq_0, 'b', linewidth=1)
    plt.hlines(Pd, target_RCS_dB[0], target_RCS_dB[-1], 'k', 'dashed')
    plt.vlines(settings['target_params']['RCS'], 0, 1, 'k', 'dashdot')
    plt.xlabel('Target RCS')
    plt.ylabel('Probability of Detection')
    plt.legend([r'$10^{-2}$ FA/min', r'$10^{-1}$ FA/min', r'$10^{0}$ FA/min'], loc='best')
    plt.title(r'%0.1f km slant range from %0.1f ft AGL' % (R0 / 1e3, hAgl / ft2m))
    plt.xlim([target_RCS_dB[0], target_RCS_dB[-1]])
    plt.ylim([0, 1.05])

    ################################################################################
    # Generate the ROC curves for different FA rates that show Pd vs slant range
    #   for the target RCS define above
    ################################################################################

    # Define the target ranges
    target_range = linspace(R0 * 0.75, R0 * 1.5, 200)

    # Total atmospheric losses
    atmosphericLosses = 10 ** ((twoWayFreeAirdBPerkm * target_range / 1e3) / 10.0)
    SNR_lin = (settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * target_RCS * compression_gain /
               ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * settings[
                   'beam_edge_loss'] * atmosphericLosses *
                NoisePow * target_range ** 4))
    CPIGain_req_snr = SNR_lin * settings['Ncpi'] * STAPGain
    P_dreqs = [genPdreq0(sqrt(2 * CPIGain_req_snr), r, SNR_lin.size) for r in req_fa]

    plt.figure('Pd vs. Slant Range')
    for pdreq in P_dreqs:
        plt.plot(target_range / 1e3, pdreq, linewidth=1)
    plt.hlines(Pd, target_range[0] / 1e3, target_range[-1] / 1e3, 'k', 'dashed')
    plt.vlines(R0 / 1e3, 0, 1, 'k', 'dashdot')
    plt.xlabel('Target Slant Range (km)')
    plt.ylabel('Probability of Detection')
    plt.legend([f'{r} FA/min' for r in req_fa_rate], loc='best')
    plt.title(r'%0.1f dBsm target from %0.1f ft AGL' % (settings['target_params']['RCS'], hAgl / ft2m))
    plt.xlim([target_range[0] / 1e3, target_range[-1] / 1e3])
    plt.ylim([0, 1.05])
    ################################################################################
    # Generate the ROC curves for different FA rates that show Pd vs CPI Length
    #   for target range and target RCS from above
    ################################################################################
    # required FA rate in FA/min
    CPILength = arange(32, 256 + 1)
    SNR_lin = settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * target_RCS * compression_gain \
              / ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * NoisePow * R0 ** 4)
    CPIGain_lin_snr = SNR_lin * CPILength * STAPGain
    alpha_req_snr = sqrt(2 * CPIGain_lin_snr)

    P_dreq_4 = genPdreq0(alpha_req_snr, req_fa[2], CPILength.size)
    P_dreq_2 = genPdreq0(alpha_req_snr, req_fa[1], CPILength.size)
    P_dreq_0 = genPdreq0(alpha_req_snr, req_fa[0], CPILength.size)

    plt.figure('Pd vs. CPI size')
    plt.plot(CPILength, P_dreq_0, 'r', linewidth=1)
    plt.plot(CPILength, P_dreq_2, 'b', linewidth=1)
    plt.plot(CPILength, P_dreq_4, 'g', linewidth=1)
    plt.hlines(Pd, CPILength[0], CPILength[-1], 'k', 'dashed')
    plt.vlines(settings['Ncpi'], 0, 1, 'k', 'dashdot')
    plt.xlabel('CPI Size (# pulses)')
    plt.ylabel('Probability of Detection')
    plt.legend(
        [r'$10^{-2}$ FA/min', r'$10^{-1}$ FA/min', r'$10^{0}$ FA/min'],
        loc='best')
    plt.title(r'%0.0f dbsm target at %0.1f km slant range from %0.1f ft AGL' \
              % (settings['target_params']['RCS'], R0 / 1e3, settings['radar_params']['hAgl']))
    plt.xlim([CPILength[0], CPILength[-1]])
    plt.ylim([0, 1.05])

    ################################################################################
    ################################################################################
    # Generate the ROC curves for different FA rates that show Pd vs slant range
    #   for the target RCS define above
    ################################################################################
    # required FA rate in FA/min
    req_fa_rate_dB = linspace(-500, 10, 200)
    req_fa_rate = 10 ** (req_fa_rate_dB / 10.0)
    req_fa = req_fa_rate / 60.0 * Tcpi / (settings['Ncpi'] * nominalNumRangeBins)
    SNR_lin = settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * target_RCS * compression_gain \
              / ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * NoisePow * R0 ** 4)
    CPIGain_req_snr = SNR_lin * settings['Ncpi'] * STAPGain
    alpha_req_snr = sqrt(2 * CPIGain_req_snr)

    P_dreq_0 = ones(req_fa.size)
    for i in range(req_fa.size):
        P_dreq_0[i] = marcumq(
            alpha_req_snr, sqrt(-2 * log(req_fa[i])), 60.0, 10000)

    plt.figure('Pd vs. False Alarm Rate')
    plt.semilogx(req_fa_rate, P_dreq_0, 'r', linewidth=1)
    plt.hlines(Pd, req_fa_rate[0], req_fa_rate[-1], 'k', 'dashed')
    plt.vlines(1.0, 0, 1.0, 'k', 'dashdot')
    plt.xlabel('False Alarm Rate (FA/min)')
    plt.ylabel('Probability of Detection')
    plt.title(r'%0.0f dbsm target at %0.1f km slant range from %0.1f ft AGL' \
              % (settings['target_params']['RCS'], R0 / 1e3, settings['radar_params']['hAgl']))
    plt.xlim([req_fa_rate[0], req_fa_rate[-1]])
    plt.ylim([0, 1.05])

    ################################################################################
    # Generate the ROC curves for different FA rates that show Pd vs swath size
    #   for target range and target RCS from above
    ################################################################################
    # Vary the slant range swath
    slantRangeSwaths = linspace(0.1, 1.0, 200) * slantRangeSwath
    # Compute the number of range samples for the swath
    nominalNumRangeBinsArray = \
        (slantRangeSwaths * 2 / rng_rez).astype('int').reshape(
            (1, len(slantRangeSwaths)))
    # required FA rate in FA/min
    req_fa_rate = array([0.1, 1.0, 10.0]).reshape((3, 1))
    req_fa = (req_fa_rate / 60.0).dot(
        Tcpi / (settings['Ncpi'] * nominalNumRangeBinsArray))
    req_fa *= req_fa
    SNR_lin = (settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * target_RCS * tp /
               ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * N0 * R0 ** 4))
    CPIGain_lin_snr = SNR_lin * settings['Ncpi'] * STAPGain
    alpha_req_snr = sqrt(2 * CPIGain_lin_snr)

    P_dreq_4 = [genPdreq0([alpha_req_snr], req_fa[2, i], 1) for i in range(slantRangeSwaths.size)]
    P_dreq_2 = [genPdreq0([alpha_req_snr], req_fa[1, i], 1) for i in range(slantRangeSwaths.size)]
    P_dreq_0 = [genPdreq0([alpha_req_snr], req_fa[0, i], 1) for i in range(slantRangeSwaths.size)]

    plt.figure('Pd vs. Swath Size')
    plt.plot(slantRangeSwaths / 1e3, P_dreq_0, 'r', linewidth=1)
    plt.plot(slantRangeSwaths / 1e3, P_dreq_2, 'b', linewidth=1)
    plt.plot(slantRangeSwaths / 1e3, P_dreq_4, 'g', linewidth=1)
    plt.hlines(Pd, slantRangeSwaths[0], slantRangeSwaths[-1], 'k', 'dashed')
    plt.vlines(slantRangeSwath / 2.0e3, 0, 1, 'k', 'dashdot')
    plt.xlabel('Slant Range Swath (km)')
    plt.ylabel('Probability of Detection')
    plt.legend([r'$10^{-1}$ FA/min', r'$10^{0}$ FA/min', r'$10^{1}$ FA/min'], loc='best')
    plt.title(r'%0.0f km slant range from %0.0f ft AGL' % (R0 / 1e3, settings['radar_params']['hAgl']))
    plt.xlim([slantRangeSwaths[0] / 1e3, slantRangeSwaths[-1] / 1e3])
    plt.ylim([0, 1.05])

    ################################################################################
    ################################################################################
    # Generate the ROC curves for RCS vs slant range for a given Pd and FAR
    ################################################################################
    # This involves reversing the Marcum-Q, so we're just going to least squares the points
    npts = 200
    target_range = linspace(R0 * 0.75, R0 * 1.5, npts)
    atmosphericLosses = 10 ** ((twoWayFreeAirdBPerkm * target_range / 1e3) / 10.0)
    req_fa_rate = 6.0
    req_fa = req_fa_rate / 60.0 * Tcpi / (settings['Ncpi'] * nominalNumRangeBins)

    rcs_pd = zeros(npts)
    x0 = 1e-2
    for tidx, tr in enumerate(target_range):
        def minRCS(x):
            SNR_lin = (settings['radar_params']['Pt'] * TxGain * RxGain * lamda ** 2 * x[0] * compression_gain /
                       ((4 * pi) ** 3 * dbtonat(settings['radar_params']['Ls']) * settings[
                           'beam_edge_loss'] * atmosphericLosses[tidx] *
                        NoisePow * tr ** 4))
            CPIGain_req_snr = SNR_lin * settings['Ncpi'] * STAPGain
            P_dreqs = genPdreq0([sqrt(2 * CPIGain_req_snr)], req_fa, 1)
            return abs(P_dreqs[0] - settings['desiredPd'])
        rcs_pd[tidx] = minimize(minRCS, array([x0]), bounds=[(0, None)])['x'][0]
        x0 = rcs_pd[tidx]
    rcs_pd = 10 * log(rcs_pd)

    plt.figure('RCS vs. Slant Range')
    plt.plot(target_range / 1e3, rcs_pd, 'r', linewidth=1)
    plt.xlabel('Slant Range (km)')
    plt.ylabel('RCS (dBsm)')
    plt.title(f"{settings['desiredPd']} Pd and {req_fa_rate} FA/min FAR")
    plt.xlim([target_range[0] / 1e3, target_range[-1] / 1e3])
    plt.ylim([min(rcs_pd), max(rcs_pd)])

    plt.show()
