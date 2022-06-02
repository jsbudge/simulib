import numpy as np
import socket
import struct
import sys
import argparse
# from xml.etree.ElementTree import ElementTree, Element, SubElement, tostring, parse, fromstringlist
from lxml import etree
from datetime import datetime, timedelta, tzinfo, timezone
from simulation_functions import llh2enu, azelToVec, getElevation, enu2llh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as rot
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

base_tsdf = '<ns39:TrustedDataObject xmlns="urn:edm:/geoObservation/v2" xmlns:ns2="urn:edm:/device/v2" xmlns:ns3="urn:edm:/spatial/v2" xmlns:ns4="http://www.opengis.net/gml" xmlns:ns5="urn:edm:/antenna/v2" xmlns:ns6="urn:edm:/locatedIn/v2" xmlns:ns7="urn:edm:/commIdentifier/v2" xmlns:ns8="urn:edm:/basic/v2" xmlns:ns9="urn:edm:/party/v1" xmlns:ns10="urn:edm:/file/v1" xmlns:ns11="urn:edm:/commDevice/v2" xmlns:ns12="urn:edm:/signal/v3" xmlns:ns13="urn:edm:/commDeviceTech/v2" xmlns:ns14="http://www.w3.org/1999/xlink" xmlns:ns15="urn:edm:/platform/v2" xmlns:ns16="urn:edm:/weapon/v2" xmlns:ns17="urn:edm:/task/v2" xmlns:ns18="urn:edm:/information/v2" xmlns:ns19="urn:edm:/reference/v1" xmlns:ns20="urn:edm:/processing/v2" xmlns:ns21="urn:edm:/collectionSys/v2" xmlns:ns22="urn:edm:/collectionSysConfig/v2" xmlns:ns23="urn:edm:/track/v2" xmlns:ns24="urn:exo:/processing" xmlns:ns25="urn:exo:/processing/collectionSysConfiguration/v2" xmlns:ns26="urn:exo:/processing/commEvent/v2" xmlns:ns27="urn:exo:/processing/commEventStream/v2" xmlns:ns28="urn:exo:/processing/deviceCharacterization/v1" xmlns:ns29="urn:exo/processing/geotag/v3" xmlns:ns30="urn:exo:/processing/geoMeasurement/v2" xmlns:ns31="urn:exo:/processing/geoResult/v2" xmlns:ns32="urn:exo:/processing/messageCharacterization/v1" xmlns:ns33="urn:edm:/message/v2" xmlns:ns34="urn:exo:/processing/orderOfBattle/v2" xmlns:ns35="urn:exo:/processing/pedigree/v2" xmlns:ns36="urn:exo:/processing/recording/v2" xmlns:ns37="urn:exo:/processing/signalCharacterization/v3" xmlns:ns38="urn:exo:/processing/tracking/v2" xmlns:ns39="urn:us:gov:ic:tdf:alt:v2" xmlns:ns40="urn:us:gov:ic:edh:alt:v2" xmlns:ns41="urn:us:gov:ic:ism" xmlns:ns42="urn:us:gov:ic:ntk" xmlns:ns43="urn:us:gov:ic:arh" xmlns:ns44="urn:edm:edh:v2" ns39:version="1.0"><ns39:HandlingAssertion ns39:scope="TDO"><ns39:HandlingStatement><ns40:Edh><ns44:Identifier>a35d6150-bc74-474c-9df0-c5669b725422</ns44:Identifier><ns44:CreateDateTime>2015-01-01T00:00:05.500-05:00</ns44:CreateDateTime><ns44:ResponsibleEntity>MyEdhEntity</ns44:ResponsibleEntity><ns44:DataSet>MyEdhDataset</ns44:DataSet><ns44:AuthRef>yourAuthRef</ns44:AuthRef><ns44:PolicyRef>yourPolicyRef</ns44:PolicyRef><ns44:ControlSet>CLS:U CUI:FOUO</ns44:ControlSet></ns40:Edh></ns39:HandlingStatement></ns39:HandlingAssertion><ns39:Assertion ns39:scope="TDO" ns39:type="{urn:exo:processing/collectionSysConfiguration/v2}collectionSysConfiguration"><ns39:StructuredStatement><ns25:collectionSysConfiguration authRef="yourAuthRef" policyRef="yourPolicyRef" controlSet="CLS:U CUI:FOUO" specificationVersion="2.0" startDateTime="2015-01-01T00:00:05.500-05:00"><ns24:processingComponent version="someProcessingCompVer" host="somePCHost">somePCVal</ns24:processingComponent><ns25:identifier>df911c9e-9ebd-41f8-8150-9c072908a107</ns25:identifier><ns21:collectionSystem><ns2:integrates><ns11:communicationDevice><ns2:model>SEAMS</ns2:model><ns2:identifier>es-system1</ns2:identifier><ns11:deviceIntent>Sensor</ns11:deviceIntent></ns11:communicationDevice></ns2:integrates></ns21:collectionSystem></ns25:collectionSysConfiguration></ns39:StructuredStatement></ns39:Assertion><ns39:Assertion ns39:scope="TDO" ns39:type="{urn:exo:/processing/geoMeasurement/v2}geoMeasurement"><ns39:StructuredStatement><ns30:geoMeasurement authRef="yourAuthRef" policyRef="yourPolicyRef" controlSet="CLS:U CUI:FOUO" specificationVersion="2.0" startDateTime="2015-01-01T00:00:05.500-05:00"><ns24:processingComponent version="someProcessingCompVer" host="somePCHost">somePCVal</ns24:processingComponent><ns30:identifier>029bc295-58e3-4703-9b54-7516be694861</ns30:identifier><LOB measurementMethod="Synthesized"><identifier>2b19b213-9bcc-4bc6-bfd1-22ab7991c285</identifier><referenceTime>2015-01-01T00:00:05.500-05:00</referenceTime><referenceFrequency numberOfSamples="1" numberOfSamplesQualifier="GreaterThanOrEqualTo">9.0E8</referenceFrequency><observingSensor role="Reference" action="Update"><ns6:locatedIn><ns3:positionArea referencePoint="SensorCOG"><ns4:Point><ns4:pos>39.979822 -74.401882 4573.034183</ns4:pos></ns4:Point></ns3:positionArea></ns6:locatedIn><ns2:model>SEAMS</ns2:model><ns2:serialNumber>es-system1</ns2:serialNumber><ns2:availability>Available</ns2:availability><ns11:sensorType>Unknown</ns11:sensorType><ns11:side>Right</ns11:side></observingSensor><observedEmitter score="15" numberOfObservations="1"><ns8:detectionType>direct</ns8:detectionType><observedEmitterUseIntent>Target</observedEmitterUseIntent></observedEmitter><angle mean="0.0" numberOfSamples="1" numberOfSamplesQualifier="EqualTo">-61.89404753935344</angle><angleError accuracy="1.0E-4" measurementMethod="Synthesized">0.19600000000000004</angleError><ns3:elevation xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:nil="true"/><referenceOrientation>TrueNorth</referenceOrientation><range xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:nil="true"/><bearingQuality>NotApplicable</bearingQuality></LOB></ns30:geoMeasurement></ns39:StructuredStatement></ns39:Assertion><ns39:StringPayload></ns39:StringPayload></ns39:TrustedDataObject>\n' + \
             '<ns39:TrustedDataObject xmlns="urn:edm:/geoObservation/v2" xmlns:ns2="urn:edm:/device/v2" xmlns:ns3="urn:edm:/spatial/v2" xmlns:ns4="http://www.opengis.net/gml" xmlns:ns5="urn:edm:/antenna/v2" xmlns:ns6="urn:edm:/locatedIn/v2" xmlns:ns7="urn:edm:/commIdentifier/v2" xmlns:ns8="urn:edm:/basic/v2" xmlns:ns9="urn:edm:/party/v1" xmlns:ns10="urn:edm:/file/v1" xmlns:ns11="urn:edm:/commDevice/v2" xmlns:ns12="urn:edm:/signal/v3" xmlns:ns13="urn:edm:/commDeviceTech/v2" xmlns:ns14="http://www.w3.org/1999/xlink" xmlns:ns15="urn:edm:/platform/v2" xmlns:ns16="urn:edm:/weapon/v2" xmlns:ns17="urn:edm:/task/v2" xmlns:ns18="urn:edm:/information/v2" xmlns:ns19="urn:edm:/reference/v1" xmlns:ns20="urn:edm:/processing/v2" xmlns:ns21="urn:edm:/collectionSys/v2" xmlns:ns22="urn:edm:/collectionSysConfig/v2" xmlns:ns23="urn:edm:/track/v2" xmlns:ns24="urn:exo:/processing" xmlns:ns25="urn:exo:/processing/collectionSysConfiguration/v2" xmlns:ns26="urn:exo:/processing/commEvent/v2" xmlns:ns27="urn:exo:/processing/commEventStream/v2" xmlns:ns28="urn:exo:/processing/deviceCharacterization/v1" xmlns:ns29="urn:exo/processing/geotag/v3" xmlns:ns30="urn:exo:/processing/geoMeasurement/v2" xmlns:ns31="urn:exo:/processing/geoResult/v2" xmlns:ns32="urn:exo:/processing/messageCharacterization/v1" xmlns:ns33="urn:edm:/message/v2" xmlns:ns34="urn:exo:/processing/orderOfBattle/v2" xmlns:ns35="urn:exo:/processing/pedigree/v2" xmlns:ns36="urn:exo:/processing/recording/v2" xmlns:ns37="urn:exo:/processing/signalCharacterization/v3" xmlns:ns38="urn:exo:/processing/tracking/v2" xmlns:ns39="urn:us:gov:ic:tdf:alt:v2" xmlns:ns40="urn:us:gov:ic:edh:alt:v2" xmlns:ns41="urn:us:gov:ic:ism" xmlns:ns42="urn:us:gov:ic:ntk" xmlns:ns43="urn:us:gov:ic:arh" xmlns:ns44="urn:edm:edh:v2" ns39:version="1.0"><ns39:HandlingAssertion ns39:scope="TDO"><ns39:HandlingStatement><ns40:Edh><ns44:Identifier>2549d061-391c-4a49-8bf0-78b83166b436</ns44:Identifier><ns44:CreateDateTime>2017-03-30T13:19:00.764Z</ns44:CreateDateTime><ns44:ResponsibleEntity>MyEdhEntity</ns44:ResponsibleEntity><ns44:DataSet>MyEdhDataset</ns44:DataSet><ns44:AuthRef>yourAuthRef</ns44:AuthRef><ns44:PolicyRef>yourPolicyRef</ns44:PolicyRef><ns44:ControlSet>CLS:U CUI:FOUO</ns44:ControlSet></ns40:Edh></ns39:HandlingStatement></ns39:HandlingAssertion><ns39:Assertion ns39:scope="TDO" ns39:type="{urn:exo:/processing/geoResult/v2}geoResult"><ns39:StructuredStatement><ns31:geoResult authRef="yourAuthRef" policyRef="yourPolicyRef" controlSet="CLS:U CUI:FOUO" specificationVersion="2.0" startDateTime="2017-03-30T13:19:00.764Z"><ns24:processingComponent version="someProcessingCompVer" host="somePCHost">somePCVal</ns24:processingComponent><ns31:identifier>2d2c8de9-2877-4819-bcfb-00c1c2435eee</ns31:identifier><ns11:communicationDevice><ns6:locatedIn><ns3:positionArea><ns3:Ellipse srsName="WGS84E_3D"><ns4:pos>35.32729196468948 -116.52322014489368 743.6552153180844</ns4:pos><ns3:semiMajorAxis>235.19487284586418</ns3:semiMajorAxis><ns3:semiMinorAxis>235.19487284586418</ns3:semiMinorAxis><ns3:orientation>180.0</ns3:orientation></ns3:Ellipse></ns3:positionArea></ns6:locatedIn><ns2:model>SEAMS</ns2:model><ns2:identifier>tx-2-1-67-1</ns2:identifier><ns11:deviceIntent>Emitter</ns11:deviceIntent></ns11:communicationDevice></ns31:geoResult></ns39:StructuredStatement></ns39:Assertion><ns39:StringPayload></ns39:StringPayload></ns39:TrustedDataObject>\n'

tsdf_root = etree.fromstringlist(['<root>', base_tsdf, '</root>'])
DTR = np.pi / 180.


def getAlt(altpos, altaz, altel):
    en_unit = azelToVec(altaz, altel)
    enu_extpos = altpos + en_unit
    llh_extpos = enu2llh(*enu_extpos, init_llh)
    els = getElevation((llh_extpos[0], llh_extpos[1]))
    dz = llh_extpos[2] - els
    dsl = dz / 2
    iters = 0
    while abs(dz) >= 1 and iters <= 50:
        enu_extpos = altpos + en_unit * dsl
        llh_extpos = enu2llh(*enu_extpos, init_llh)
        els = getElevation((llh_extpos[0], llh_extpos[1]))
        dz = llh_extpos[2] - els
        dsl = dsl + dz / 2
        iters += 1
    return els - init_llh[2]


def getExtPos(altpos, altaz, altel):
    en_unit = azelToVec(altaz, altel)
    enu_extpos = altpos + en_unit
    llh_extpos = enu2llh(*enu_extpos, init_llh)
    els = getElevation((llh_extpos[0], llh_extpos[1]))
    dz = llh_extpos[2] - els
    dsl = dz / 2
    iters = 0
    while abs(dz) >= 1 and iters <= 50:
        enu_extpos = altpos + en_unit * dsl
        llh_extpos = enu2llh(*enu_extpos, init_llh)
        els = getElevation((llh_extpos[0], llh_extpos[1]))
        dz = llh_extpos[2] - els
        dsl = dsl + dz / 2
        iters += 1
    return enu_extpos


def tsdfLOB(time, freq, lat, lon, alt, az_ang, err=1.6):
    lob_base = tsdf_root[0]
    # Create date time
    lob_base[0][0][0][1].text = time
    # startDateTime in sysconfig
    lob_base[1][0][0].attrib['startDateTime'] = time
    # startDateTime in sysconfig 2
    lob_base[2][0][0].attrib['startDateTime'] = time
    # reference time
    lob_base[2][0][0][2][1].text = time
    # Reference frequency
    lob_base[2][0][0][2][2].text = f'{freq}'
    # Platform location
    lob_base[2][0][0][2][3][0][0][0][0].text = f'{lat:.6f} {lon:.6f} {alt:.6f}'
    # Observed azimuth angle
    lob_base[2][0][0][2][5].text = f'{az_ang / DTR}'
    # Angle error
    lob_base[2][0][0][2][6].text = f'{err}'
    lob_tree = etree.ElementTree(lob_base)
    return ' '.join(etree.tostring(lob_tree.getroot(), encoding='unicode', xml_declaration=False).split()).replace('> <', '><') + '\r\n'


def tsdfEllipse(time, lat, lon, alt, smaxis, sminaxis, ell_ang):
    ell_base = tsdf_root[1]
    # Create time
    ell_base[0][0][0][1].text = time
    # startDateTime in sysconfig
    ell_base[1][0][0].attrib['startDateTime'] = time
    # Position of ellipse
    ell_base[1][0][0][2][0][0][0][0].text = f'{lat:.6f} {lon:.6f} {alt:.6f}'
    # semimajoraxis
    ell_base[1][0][0][2][0][0][0][1].text = f'{smaxis}'
    # Semiminoraxis
    ell_base[1][0][0][2][0][0][0][2].text = f'{sminaxis}'
    # orientation
    ell_base[1][0][0][2][0][0][0][3].text = f'{ell_ang / DTR}'
    ell_tree = etree.ElementTree(ell_base)
    return ' '.join(etree.tostring(ell_tree.getroot(), encoding='unicode', xml_declaration=False).split()).replace('> <', '><') + '\r\n'


def getEllipseParams(pts):
    center = np.mean(pts, axis=0)
    cdata = pts - center[None, :]
    ev, trans = np.linalg.eig(cdata.T.dot(cdata))
    axes = np.std(2 * np.linalg.pinv(trans).dot(cdata.T), axis=1)
    return center[0], center[1], axes[0], axes[1], np.arccos(trans[0, 0])


def genEllipsePoints(pos, az):
    init_theta = np.zeros_like(az) - 135 * DTR

    def resid_func(theta):
        r = np.zeros(((len(theta) - 1) * 3,))
        alts = [getAlt(pos[:, m], az[m], theta[m]) for m in range(len(theta))]
        mus = [(pos[2, m] - alts[m]) / np.sin(theta[m]) for m in range(len(theta))]
        dir_vecs = [azelToVec(az[m], theta[m]) for m in range(len(theta))]
        for idx in range(1, len(theta)):
            r[(idx - 1) * 3:idx * 3] = -np.array([dir_vecs[0][0] * mus[0] - dir_vecs[idx][0] * mus[idx],
                                                  dir_vecs[0][1] * mus[0] - dir_vecs[idx][1] * mus[idx],
                                                  dir_vecs[0][2] * mus[0] - dir_vecs[idx][2] * mus[idx]])
            r[(idx - 1) * 3:idx * 3] += np.array(
                [pos[0, 0] - pos[0, idx],
                 pos[1, 0] - pos[1, idx],
                 pos[2, 0] - pos[2, idx]])
        return r

    lsq_sol = least_squares(resid_func, init_theta, bounds=(- 135 * DTR - np.pi / 8, - 135 * DTR + np.pi / 8))
    return np.array([getExtPos(pos[:, m], az[m], lsq_sol['x'][m]) for m in range(num_lobs)]), lsq_sol['x']


def gps_to_datetime(secs, week):
    t = datetime(1980, 1, 6, 0, 0, 0, 0, tzinfo=timezone.utc)
    t += timedelta(seconds=secs)
    t += timedelta(weeks=week)
    return t


def datetime_to_tow(t):
    wk_ref = datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow


new_tsdf = ''
getDatetime = lambda w, s: gps_to_datetime(s + 21600, w).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
gps_week = 2212
sec = 2000
hostip = '192.168.2.138'
destip = '192.168.2.70'
port = 54322

UDPDFSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
UDPDFSocket.bind((destip, port))
UDPDFSocket.listen()
conn_sock, conn_addr = UDPDFSocket.accept()
UDPDFSocket.settimeout(4)

'''
argp = argparse.ArgumentParser()
argp.add_argument('-ip', '-op', '-input_ipadd', '-output_ipadd')


# Read in arguments for ports and IP addresses, use defaults if none given
inp_args = argp.parse_args()
inp_address = (inp_args.Output)

# Set the local IP address, port, and buffer size
localIP = "127.0.0.1"
DFPort = 2721
bufferSize = 1024

# Bind to address and ip
UDPDFSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPDFSocket.bind((localIP, DFPort))
UDPDFSocket.settimeout(40)
print("Listening for UDP")

# Listen for incoming datagrams
while True:
    try:
        # Read from the UDP port
        message, address = UDPDFSocket.recvfrom(bufferSize)

        # Parse the packet id info. Packet type for DF results equals 10
        packetType = int(message[0])
        packetNum = int(message[1])
        collectionID = int(message[2])

        # Parse the DF result data.
        centerFrequency = struct.unpack("<f", message[3:7])[0]
        azimuthAngle = struct.unpack("<f", message[7:11])[0]
        elevationAngle = struct.unpack("<f", message[11:15])[0]
        latitude = struct.unpack("<f", message[15:19])[0]
        longitude = struct.unpack("<f", message[19:23])[0]
        altitude = struct.unpack("<f", message[23:27])[0]
        gpsWeek = int.from_bytes(message[27:29], byteorder="little")
        gpsSeconds = struct.unpack("<f", message[29:33])[0]

        # print the parsed data for verification
        print(  "type = " + str(packetType) + \
                " num = " + str(packetNum) + \
                " id = " + str(collectionID) + \
                " freq = {:.3e}".format(centerFrequency) + \
                " az = {:.3f}".format(azimuthAngle) + \
                " el = {:.3f}".format(elevationAngle) + \
                " lat = " + str(latitude) + \
                " lon = " + str(longitude) + \
                " alt = " + str(altitude) + \
                " week = " + str(gpsWeek) + \
                " sec = " + str(gpsSeconds))

    # Close the socket if no packets are received for a while
    except socket.timeout:
        print("UDP Listener Timed Out")
        UDPDFSocket.close()
        break

'''

# Test parameters to get ellipses and such right
init_llh = (30.895723, -86.529416, 37.525590657451424)
num_trials = 2
max_num_lobs = 6
dep_ang = -135 * DTR
mse_max = 1.6 * DTR
# Nonlinear least squares for some LOB analysis
errorfig = None
for num_lobs in range(3, max_num_lobs):
    # Setup the plane flight line
    e = np.linspace(0, 3000, num_lobs)
    n = np.zeros_like(e) + np.random.rand(len(e)) * 2 - 1
    u = np.zeros_like(e) + 5000 / 3.2808 + np.random.rand(len(e)) * 2 - 1
    lat, lon, height = enu2llh(e, n, u, init_llh)
    u -= init_llh[2]
    pos = np.array([e, n, u])
    err_mu = np.zeros((num_trials,))
    err_std = np.zeros_like(err_mu)
    for trial in tqdm(range(num_trials)):
        mse = mse_max / num_trials * (trial + 1)
        errors = np.zeros((num_trials,))
        for ep in range(num_trials):
            truth = np.array([np.random.randint(0, int(e.max())),
                              np.random.randint(int(u.mean() - 400), int(u.mean() + 400)), 0.])
            llh_truth = enu2llh(*truth, init_llh)
            truth[2] = getElevation((llh_truth[0], llh_truth[1])) - init_llh[2]
            az = np.arctan2(truth[0] - e, truth[1] - n) + np.random.normal(0, mse, e.shape)
            truth_az = np.arctan2(truth[0] - e, truth[1] - n)
            rng = np.sqrt((e - truth[0]) ** 2 + (n - truth[1]) ** 2 + (u - truth[2]) ** 2)
            rng_mean = rng.mean()
            el = -np.arcsin((u - truth[2]) / rng) + np.random.normal(0, mse, e.shape)
            truth_el = -np.arcsin((u - truth[2]) / rng)

            ps, calc_el = genEllipsePoints(pos, az)
            errors[ep] = np.linalg.norm(truth - np.mean(ps, axis=0))
            for lob in range(num_lobs):
                # new_tsdf += tsdfLOB(getDatetime(sec, gps_week), 9e8, lat[lob], lon[lob], height[lob], az[lob])
                conn_sock.sendto(bytes(tsdfLOB(getDatetime(*datetime_to_tow(datetime.now())), 9e8, lat[lob], lon[lob], height[lob],
                                                 az[lob]), 'utf-8'), conn_addr)
            try:
                ell_params = getEllipseParams(ps)
                ell_center = enu2llh(ell_params[0], ell_params[1], 0, init_llh)
                # new_tsdf += tsdfEllipse(getDatetime(sec, gps_week), ell_center[0], ell_center[1],
                #                         ell_center[2], ell_params[2], ell_params[3], ell_params[4])
                conn_sock.sendto(
                    bytes(tsdfEllipse(getDatetime(*datetime_to_tow(datetime.now())), ell_center[0], ell_center[1], ell_center[2],
                                      ell_params[2], ell_params[3], ell_params[4]), 'utf-8'), conn_addr)
            except ValueError:
                print('Ellipse bad.')
            sec += 5
        err_mu[trial] = errors.mean()
        err_std[trial] = errors.std()
    sec += 1
    if errorfig is None:
        errorfig = px.scatter(x=(np.arange(num_trials) + 1) * mse_max / num_trials / DTR, y=err_mu)
    else:
        errorfig.add_scatter(x=(np.arange(num_trials) + 1) * mse_max / num_trials / DTR, y=err_mu)

print("UDP Listener Timed Out")
UDPDFSocket.close()

errorfig.show()


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    x0, y0, ap, bp, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y


ell_xy = get_ellipse_pts(getEllipseParams(ps[:, :2]))
datafig = px.scatter_3d(x=ps[:, 0], y=ps[:, 1], z=ps[:, 2])
datafig.add_scatter3d(x=ell_xy[0], y=ell_xy[1], z=np.zeros_like(ell_xy[0]) + ps[:, 2].mean())
datafig.show()

with open('/home/jeff/repo/out_tsdf.tsdf', 'w') as f:
    f.write(new_tsdf)




