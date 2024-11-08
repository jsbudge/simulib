#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:45:47 2019

@author: jeff

Novatel Log Reader

Reads the logs in a Novatel log (.ASC) file.
"""

import numpy as np
from tqdm import tqdm
import re
# from useful_lib import timetogps, lagrange, timetogps
import pandas as pd
from bitarray import bitarray
from os import walk
# from skyfield.api import load, JulianDate
import subprocess
import ftplib
from _datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

# from mayavi import mlab

c0 = 299792458


def hextoint(s, flip=True, twos=False):
    if flip:
        t = int(''.join(np.flip([s[n:n + 2] for n in range(0, len(s), 2)])), 16)
    else:
        t = int(s, 16)
    if twos:
        t = t - int('0xFFFFFFFF', 16) + 1
    return t


def readFile(filename):
    output = []
    try:
        logs = open(filename, 'r').readlines()
    except:
        print('File not found: ' + filename)
        return output
    for line in logs:
        line = line[:-1].split(';')
        header = line[0].split(',')
        body = line[1].split(',')
        output.append({'header': header, 'body': body})
    return output


def saveProcessedFile(filename, path):
    L1 = 0.1902936727984
    L2 = 0.2442102134246
    L5 = c0 / 1176.45e6
    freq_vals = {0: {0: L1, 5: L2, 9: L2, 14: L5, 16: L1, 17: L2},
                 1: {0: L1, 1: L2, 5: L2}, 5: {0: L1, 14: L5, 16: L1, 17: L2, 27: L5},
                 2: {0: L1, 6: L5}, 6: {0: L5}}
    # LOGS KEY
    # 0: INSPVAA, 1: INSPVAXA, 2: BESTGNSSPOSA, 3:BESTXYZA, 0:TRACKSTATA, 1:SATXYZA, 2: RANGEA
    proc_log = bitarray('0000')
    track_log = bitarray('000')
    proc_data = pd.DataFrame(
        columns=['time', 'type', 'lat', 'lon', 'height', 'height_egm96', 'raw_lat', 'raw_lon', 'raw_height',
                 'r', 'p', 'y',
                 'vel_n', 'vel_e', 'vel_u', 'ecef_x', 'ecef_y', 'ecef_z',
                 'vel_x', 'vel_y', 'vel_z', 'sol_age', 'nsats',
                 'nsats_in_solution', 'sigma_lat', 'sigma_lon',
                 'sigma_height', 'sigma_r', 'sigma_p', 'sigma_y', 'rec_clk_off', 'logs']).set_index(['type', 'time'])
    track_data = pd.DataFrame(columns=['sat', 'time', 'cutoff_angle', 'psr_l1', 'psr_l2', 'dopp_l1', 'dopp_l2',
                                       'cno_l1', 'cno_l2', 'locktime', 'psrl1_res', 'psrl2_res', 'psrl1_weight',
                                       'psrl2_weight', 'phi_l1', 'phi_l2', 'psrl1_sigma', 'psrl2_sigma',
                                       'phil1_sigma', 'phil2_sigma', 'SNRl1', 'SNRl2',
                                       'x', 'y', 'z', 'clk', 'iono', 'trop', 'channel', 'logs']).set_index(
        ['sat', 'time'])
    imu_data = pd.DataFrame(columns=['time', 'sf_x', 'sf_y', 'sf_z', 'dr', 'dp', 'dy',
                                     'acc_x', 'acc_y', 'acc_z', 'dcr', 'dcp', 'dcy']).set_index(['time'])
    satcoeffs = []
    if type(filename) == str:
        filename = [filename]
    for ff in filename:
        logs = open(ff, 'r').readlines()
        for line in tqdm(logs):
            ln = line[:-1].split(';')
            hd = ln[0].split(',')
            bd = ln[1].split(',')
            _type = hd[0]
            t = np.float64(hd[6]) if _type[0] == '#' else np.float64(hd[2])
            if _type == '#INSPVAA':
                proc_log[0] = True
                proc_data.loc[(0, t), ['lat', 'lon', 'height_egm96',
                                       'vel_n', 'vel_e', 'vel_u',
                                       'r', 'p', 'y']] = \
                    [np.float64(bd[2]), np.float64(bd[3]), np.float64(bd[4]),
                     np.float64(bd[5]), np.float64(bd[6]), np.float64(bd[7]),
                     np.float64(bd[9]), np.float64(bd[8]),
                     np.float64(bd[10])]  # Time, Lat, Lon, Height, roll, pitch, heading
            elif _type == '#INSPVAXA':
                proc_log[1] = True
                proc_data.loc[(1, t), ['lat', 'lon', 'height', 'vel_n', 'vel_e', 'vel_u',
                                       'r', 'p', 'y', 'sigma_lat', 'sigma_lon', 'sigma_height',
                                       'sigma_r', 'sigma_p', 'sigma_y']] = \
                    [np.float64(bd[2]), np.float64(bd[3]), np.float64(bd[4]),
                     np.float64(bd[6]), np.float64(bd[7]), np.float64(bd[8]),
                     np.float64(bd[10]), np.float64(bd[9]), np.float64(bd[11]),
                     np.float64(bd[12]), np.float64(bd[13]), np.float64(bd[14]),
                     np.float64(bd[18]), np.float64(bd[19]), np.float64(bd[20])]
            elif _type == '#BESTGNSSPOSA':
                proc_log[2] = True
                proc_data.loc[(2, t), ['raw_lat', 'raw_lon', 'raw_height',
                                       'sol_age', 'nsats', 'nsats_in_solution',
                                       'sigma_lat', 'sigma_lon', 'sigma_height']] = \
                    [np.float64(bd[2]), np.float64(bd[3]),
                     np.float64(bd[4]), np.float64(bd[12]), np.float64(bd[13]),
                     np.float64(bd[14]), np.float64(bd[7]), np.float64(bd[8]),
                     np.float64(bd[9])]
            elif _type == '#BESTXYZA':
                proc_log[3] = True
                proc_data.loc[(3, t), ['ecef_x', 'ecef_y', 'ecef_z',
                                       'vel_x', 'vel_y', 'vel_z',
                                       'sol_age', 'nsats', 'nsats_in_solution']] = \
                    [np.float64(bd[2]), np.float64(bd[3]),
                     np.float64(bd[4]), np.float64(bd[10]), np.float64(bd[11]),
                     np.float64(bd[12]), np.float64(bd[19]), np.float64(bd[20]),
                     np.float64(bd[21])]
            elif _type == '#TRACKSTATA':
                track_log[0] = True
                numsats = int(bd[3])
                cutoff_ang = np.float64(bd[2])
                nfields = 10
                for i in range(numsats):
                    shift = i * nfields
                    if bd[12 + shift] != 'NA':
                        channel = int(bd[6 + shift], 16)
                        sys = (channel & 458752) >> 16
                        st = (channel & 65011712) >> 21  # Numbers are masks for that part of the channel word
                        # grouped = (channel & int('0x00100000', 16)) >> 20
                        try:
                            freq = freq_vals[sys][st]
                        except:
                            freq = L1
                        wv = 'l1' if freq == L1 else 'l2'
                        psr_weight = np.float64(bd[13 + shift]) if i < numsats - 1 else np.float64(
                            bd[13 + shift].split('*')[0])
                        track_data.loc[(np.float64(bd[4 + shift]), t),
                                       ['cutoff_angle', 'cno_' + wv, 'locktime',
                                        'psr' + wv + '_res', 'psr' + wv + '_weight']] = \
                            [cutoff_ang, np.float64(bd[9 + shift]),
                             np.float64(bd[10 + shift]),
                             np.float64(bd[11 + shift]), psr_weight]
            elif _type == '#SATXYZ2A':
                track_log[1] = True
                numsats = int(bd[0]) if len(bd[0].split('*')) == 1 else int(bd[0].split('*')[0])
                nfields = 10
                for i in range(numsats):
                    shift = i * nfields
                    if bd[shift + 1] == 'GPS':
                        track_data.loc[(np.float64(bd[2 + shift]), t),
                                       ['x', 'y', 'z', 'clk', 'iono', 'trop']] = \
                            [np.float64(bd[3 + shift]), np.float64(bd[4 + shift]),
                             np.float64(bd[5 + shift]), np.float64(bd[6 + shift]),
                             np.float64(bd[7 + shift]), np.float64(bd[8 + shift])]
            elif _type == '#RANGEA':
                track_log[2] = True
                numsats = int(bd[0]) if len(bd[0].split('*')) == 1 else int(bd[0].split('*')[0])
                nfields = 10
                for i in range(numsats):
                    shift = i * nfields
                    channel = int(bd[10 + shift], 16) if len(bd[10 + shift].split('*')) == 1 else int(
                        bd[10 + shift].split('*')[0], 16)
                    sys = (channel & 458752) >> 16
                    st = (channel & 65011712) >> 21  # Numbers are masks for that part of the channel word
                    # grouped = (channel & int('0x00100000', 16)) >> 20
                    try:
                        freq = freq_vals[sys][st]
                    except:
                        freq = L1
                    wv = 'l1' if freq == L1 else 'l2'
                    track_data.loc[(np.float64(bd[1 + shift]), t),
                                   ['psr_' + wv, 'psr' + wv + '_sigma',
                                    'phi_' + wv, 'phi' + wv + '_sigma',
                                    'dopp_' + wv, 'SNR' + wv, 'channel']] = \
                        [np.float64(bd[3 + shift]), np.float64(bd[4 + shift]),
                         -np.float64(bd[5 + shift]) * freq, -np.float64(bd[6 + shift]) * freq,
                         np.float64(bd[7 + shift]), np.float64(bd[8 + shift]), 1.0]
            elif _type == '%RAWIMUSA':
                # CONSTANTS
                acc = 2 ** -27 * .3048 * 100;
                rot = 2 ** -33 * 100
                dy = -np.float64(bd[8]) * rot if len(bd[8].split('*')) == 1 else -np.float64(bd[8].split('*')[0]) * rot
                imu_data.loc[t, ['sf_z', 'sf_y', 'sf_x', 'dr', 'dp', 'dy']] = \
                    [-np.float64(bd[3]) * acc, -np.float64(bd[4]) * acc, -np.float64(bd[5]) * acc,
                     -np.float64(bd[6]) * rot, -np.float64(bd[7]) * rot, dy]
            elif _type == '%CORRIMUSA':
                scale = 100 / np.float64(bd[0])
                imu_data.loc[t, ['acc_x', 'acc_y', 'acc_z', 'dcr', 'dcp', 'dcy']] = \
                    [np.float64(bd[4]) * scale, np.float64(bd[5]) * scale, np.float64(bd[6]) * scale,
                     np.float64(bd[1]) * scale, np.float64(bd[2]) * scale, np.float64(bd[3]) * scale]
            elif _type == '#RAWEPHEMA':
                prn = int(bd[0])
                sf_1 = bin(int(bd[3], 16))[2:]
                sf_2 = bin(int(bd[4], 16))[2:]
                sf_3 = bin(int(bd[5].split('*')[0], 16))[2:]
                # Subframe 1 information
                URA = int(sf_1[60:64], 2)
                URA = 2 ** (1 + URA / 2) if URA <= 6 else 2 ** (URA - 2)
                satdata = [URA, int(bd[0]), int(bd[1]), int(bd[2])]  # PRN number, ref week, ref time (s)
                sdd = translateSatCoeffs([satdata, (sf_1, sf_2, sf_3)])
                sdd['sat'] = prn
                sdd['time'] = sdd['t_oc']
                satcoeffs.append(sdd)
            elif _type == '#TIMEA':
                proc_data.loc[(2, t), 'rec_clk_off'] = np.float64(bd[1])
    sat = pd.DataFrame(data=satcoeffs).set_index(['time', 'sat'])
    try:
        proc_data['logs'] = int(proc_log.to01(), 2)
        track_data['logs'] = int(track_log.to01(), 2)
        proc_data = proc_data.astype(np.float64).to_csv(path + '_proc.csv')
        track_data = track_data.astype(np.float64).to_csv(path + '_sat.csv')
        sat.to_csv(path + '_eph.csv')
        imu_data.to_csv(path + '_imu.csv')
    except:
        print('Files failed to CSV.')
        return False
    return True


def translateSatCoeffs(data):
    sf_1, sf_2, sf_3 = data[1]
    scale = lambda x, y: np.float64(x * 2 ** y)
    up = {}
    # FRAME 1
    up['af0'] = scale(twos(sf_1[216:238], 2), -31);
    up['af1'] = scale(twos(sf_1[200:216], 2), -43);
    up['af2'] = scale(twos(sf_1[192:200], 2), -55)
    up['t_oc'] = scale(int(sf_1[176:192], 2), 4);
    up['Tgd'] = scale(twos(sf_1[160:168]), -31)
    # FRAME 2
    up['Crs'] = scale(twos(sf_2[56:72], 2), -5);
    up['delta_n'] = scale(twos(sf_2[72:88], 2), -43) * np.pi;
    up['M0'] = scale(twos(sf_2[88:120], 2), -31) * np.pi
    up['Cmuc'] = scale(twos(sf_2[120:136], 2), -29);
    up['e'] = scale(int(sf_2[136:168], 2), -33);
    up['Cmus'] = scale(twos(sf_2[168:184], 2), -29)
    up['A'] = scale(int(sf_2[184:216], 2), -19) ** 2;
    up['t_oe'] = scale(int(sf_2[216:232], 2), 4)
    # FRAME 3
    up['Cic'] = scale(twos(sf_3[48:64], 2), -29);
    up['omega0'] = scale(twos(sf_3[64:96], 2), -31) * np.pi;
    up['Cis'] = scale(twos(sf_3[96:112], 2), -29)
    up['i0'] = scale(twos(sf_3[112:144], 2), -31) * np.pi;
    up['Crc'] = scale(twos(sf_3[144:160], 2), -5);
    up['w'] = scale(twos(sf_3[160:192], 2), -31) * np.pi
    up['omegadot'] = scale(twos(sf_3[192:216], 2), -43) * np.pi;
    up['IDOT'] = scale(twos(sf_3[224:238], 2), -43) * np.pi
    return up


def compileBalloonFiles(fnme_dir, loc=None):
    # get everything in dir
    (pths, pthnames, all_fnmes) = next(walk(fnme_dir))
    fnmes = {}
    for f in all_fnmes:
        f1 = f.split('.')
        if f1[1] == 'ASC':
            f2 = f1[0].split('_')
            if f2[1] in fnmes:
                fnmes[f2[1]].append(pths + '/' + f)
            else:
                fnmes[f2[1]] = [pths + '/' + f]
    t = pd.DataFrame(columns=['date', 'fnme'])
    for l in fnmes:
        tmp = pd.DataFrame(fnmes[l], columns=['fnme'])
        tmp['date'] = l
        t = t.append(tmp, sort=True)
    try:
        if loc is not None:
            t.to_csv(loc + '_balloon.csv')
        else:
            t.to_csv(pths + '/balloonlogs.csv', index=False)
        return True
    except:
        return False


def saveANTFile(csv_path, apc_df, gpsweek):
    sat_apc = apc_df.loc[np.logical_and(apc_df['start_gps_wk'] < gpsweek, apc_df['end_gps_wk'] > gpsweek)]
    try:
        sat_apc.to_csv(csv_path + '_ant.csv')
        return True
    except:
        return False


def loadANTEXFile(fnme):
    output = pd.DataFrame()
    freq = 'NONE'
    logs = open(fnme, 'r').readlines()
    is_gps = False
    for l in logs:
        data = l.split(' ')
        d = [n for n in data if n != '']
        if d[-4:-1] == ['START', 'OF', 'ANTENNA']:
            block = {'end_gps_wk': 1e6}
        elif d[-4:-1] == ['END', 'OF', 'ANTENNA']:
            if is_gps:
                output = output.append(block, ignore_index=True)
                is_gps = False
            block = {'end_gps_wk': 1e6}
        if d[-5:-1] == ['TYPE', '/', 'SERIAL', 'NO']:
            if d[2][0] == 'G':
                is_gps = True
                block['prn'] = int(d[2][1:])
        if is_gps:
            if d[-6:-1] == ['NORTH', '/', 'EAST', '/', 'UP']:
                block[freq + '_i'] = np.float64(d[0]) / 1000
                block[freq + '_j'] = np.float64(d[1]) / 1000
                block[freq + '_k'] = np.float64(d[2]) / 1000
            elif d[-3] == 'VALID' and d[-2] == 'UNTIL':
                gps_time = timetogps(int(d[0]), int(d[1]), int(d[2]),
                                     int(d[3]), int(d[4]), int(np.float64(d[5])))
                block['end_gps_wk'] = gps_time[0]
                block['end_gps_sec'] = gps_time[1]
            elif d[-3] == 'VALID' and d[-2] == 'FROM':
                gps_time = timetogps(int(d[0]), int(d[1]), int(d[2]),
                                     int(d[3]), int(d[4]), int(np.float64(d[5])))
                block['start_gps_wk'] = gps_time[0]
                block['start_gps_sec'] = gps_time[1]
            elif d[-4:-1] == ['START', 'OF', 'FREQUENCY']:
                freq = 'L' + d[0][2]
    return output


def downloadSP3(gpsweek, secs, _type='s'):
    dayofweek = int(secs / 60 / 60 / 24)
    if _type == 's':
        try:
            fnme = 'igs{}{}.sp3.Z'.format(gpsweek, dayofweek)
            url = 'pub/igs/products/{}'.format(gpsweek)
            ftp = ftplib.FTP('igs.ensg.ign.fr')
            ftp.login('', '')
            ftp.cwd(url)
            ftp.retrbinary("RETR " + fnme, open('./gnss_precise/' + fnme, 'wb').write)
            ftp.quit()
            print('Grabbed precise orbit file.')
        except:
            print('Could not grab precise orbit file')
            return False
    if _type == 'r':
        try:
            fnme = 'igr{}{}.sp3.Z'.format(gpsweek, dayofweek)
            url = 'pub/igs/products/{}'.format(gpsweek)
            ftp = ftplib.FTP('igs.ensg.ign.fr')
            ftp.login('', '')
            ftp.cwd(url)
            ftp.retrbinary("RETR " + fnme, open('./gnss_precise/' + fnme, 'wb').write)
            ftp.quit()
            print('Grabbed Rapid almost-precise orbit file.')
        except:
            print('Could not grab rapid orbit file.')
            return False
    if _type == 'u':
        hr = int(np.around((int(secs / 60 / 60) % 24 - 3) / 6, decimals=0) * 6)
        try:
            fnme = 'igu{}{}_{}.sp3.Z'.format(gpsweek, dayofweek, hr)
            url = 'pub/igs/products/{}'.format(gpsweek)
            ftp = ftplib.FTP('igs.ensg.ign.fr')
            ftp.login('', '')
            ftp.cwd(url)
            ftp.retrbinary("RETR " + fnme, open('./gnss_precise/' + fnme, 'wb').write)
            ftp.quit()
            print('Grabbed Ultra-Rapid almost precise orbit file.')
        except:
            print('Could not grab ultra-rapid orbit file.')
            return False
    subprocess.run(["uncompress", './gnss_precise/' + fnme])
    return True


def downloadIONEX(day_of_yr, yr):
    last_yr = yr[2:]
    for _type in ['jplg', 'c2pg', 'ehrg', 'codg', 'carg', 'igsg', 'gpsg']:
        try:
            fnme = _type + '{}0.{}i.Z'.format(day_of_yr, yr)
            url = 'pub/igs/products/ionosphere/{}/{}'.format(yr, day_of_yr)
            ftp = ftplib.FTP('igs.ensg.ign.fr')
            ftp.login('', '')
            ftp.cwd(url)
            ftp.retrbinary("RETR " + fnme, open('./gnss_precise/' + fnme, 'wb').write)
            ftp.quit()
            print('Grabbed IONEX ' + _type + ' file.')
            subprocess.run(["uncompress", './gnss_precise/' + fnme])
            return _type
        except:
            continue
    return None


def sp3_parse(fnme):
    output = [];
    secs_in_week = 604800.0;
    satlist = []
    logs = open(fnme, 'r').readlines()
    for line in logs:
        if line == "EOF":
            break
        cols = re.split(r' +', line)
        cols[-1] = cols[-1][:-1]
        # This is how we know if a line is something we care about
        key = cols[0]
        if key == '##':
            # Get GPS time and week here
            start_week = int(cols[1]);
            sec_offset = float(cols[2]);
            epoch_interval = float(cols[3])
            start_secs = start_week * secs_in_week + sec_offset
        elif key == '+':
            # Get satellite names, we only care about GPS, or 'G'
            if len(cols) == 3:
                total_sats = cols[1]
                sats = cols[2]
            else:
                sats = cols[1]
            sats = sats.split('G')
            if len(sats) < 2:
                # Nuthin to see here, some other satellite set
                continue
            else:
                for g in sats:
                    try:
                        satlist.append(int(g))
                    except:
                        continue
        elif key == '*':
            # Our epoch thing. Tells us what time these records stand for.
            curr_week, curr_secs = timetogps(int(cols[1]), int(cols[2]), int(cols[3]), int(cols[4]), int(cols[5]),
                                             int(float(cols[6])))
            delta_t = curr_week * secs_in_week + curr_secs - start_secs
        elif key[0] == 'P':
            sat_num = int(key[-2:])
            if key[1] == 'G' and sat_num in satlist:
                # This is our position record.
                output.append({'delta_t': delta_t, 'posx': float(cols[1]) * 1000, 'posy': float(cols[2]) * 1000,
                               'posz': float(cols[3]) * 1000,
                               'clock': float(cols[4]) * 1e-6 if float(cols[4]) != 999999.999999 else -1,
                               'satidx': sat_num, 'time': sec_offset + delta_t, 'epoch_interval': epoch_interval,
                               'start_week': start_week, 'sec_offset': sec_offset})
    out_df = pd.DataFrame(data=output).set_index(['satidx', 'time'])
    return out_df


def twos(binaryStr, yuck=2):
    # TWOSCOMP2DEC(binaryNumber) Converts a two's-complement binary number
    # BINNUMBER (in Matlab it is a string type), represented as a row vector of
    # zeros and ones, to an integer.

    # intNumber = twosComp2dec(binaryNumber)

    # --- Check if the input is string -----------------------------------------
    if not isinstance(binaryStr, str):
        raise IOError('Input must be a string.')

    # --- Convert from binary form to a decimal number -------------------------
    intNumber = int(binaryStr, 2)

    # --- If the number was negative, then correct the result ------------------
    if binaryStr[0] == '1':
        intNumber -= 2 ** len(binaryStr)
    return intNumber


def read_ionex(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    length = len(lines)
    n = 0
    h_bins = [0]
    lat_bins = [0]
    lon_bins = [0]
    flip = [0, 0, 0]
    while n < length:
        line = lines[n]
        n += 1
        description = line[60:-1].strip()
        if description == 'HGT1 / HGT2 / DHGT' or description == 'LAT1 / LAT2 / DLAT' or description == 'LON1 / LON2 / DLON':
            st = np.float64(line[0:8])
            stp = np.float64(line[8:14])
            step = np.float64(line[14:20])
        if description == 'HGT1 / HGT2 / DHGT':
            if step == 0:
                hbsz = 0
                h_bins = [st * 1000]
            else:
                h_bins = np.arange(st, stp + step, step)
                hbsz = len(h_bins)
            flip[0] = 1 if step < 0 else 0
        if description == 'LAT1 / LAT2 / DLAT':
            lat_bins = np.arange(st, stp + step, step) * np.pi / 180
            latbsz = len(lat_bins)
            flip[1] = 1 if step < 0 else 0
        if description == 'LON1 / LON2 / DLON':
            lon_bins = np.arange(st, stp + step, step) * np.pi / 180
            lonbsz = len(lon_bins)
            flip[2] = 1 if step < 0 else 0
        if description == 'EXPONENT':
            expo = np.float64(line[0:6])
        if description == 'END OF HEADER':
            break

    result = {'gpsweek': [], 'gpssecs': [], 'mapnum': [], 'h': h_bins, 'lat': lat_bins, 'lon': lon_bins, 'data': []}
    if flip[1] == 1:
        result['lat'] = np.flipud(result['lat'])
    if flip[2] == 1:
        result['lon'] = np.flipud(result['lon'])

    is_tec = False
    while n < length:
        line = lines[n]
        n += 1
        description = line[60:-1].strip()
        if description == 'START OF TEC MAP':
            k = int(line[:6])
            is_tec = True
            curr_lat = 0
            tec_map = np.zeros((latbsz, lonbsz))

        if description == 'EPOCH OF CURRENT MAP':
            year = int(line[:6])
            month = int(line[6:12])
            day = int(line[12:18])
            hh = int(line[18:24])
            next_day = False
            if hh >= 24:
                hh -= 24
                next_day = True
            mm = int(line[24:30])
            ss = int(line[30:36])
            date = datetime(year, month, day, hh, mm, ss)
            if next_day:
                date = date + timedelta(days=1)
            gpsweek, gpssecs = timetogps(year, month, day, hh, mm, ss)

        if description == 'LAT/LON1/LON2/DLON/H':
            if is_tec:
                lat = float(line[2:8])
                lon1 = float(line[8:14])
                lon2 = float(line[14:20])
                dlon = float(line[20:26])
                h = float(line[26:32])
                n_tec = int((lon2 - lon1) / dlon) + 1
                tec = 0
                while tec < n_tec:
                    line = lines[n]
                    n += 1
                    t = line.split()
                    for curr_lon, v in enumerate(t):
                        tec_map[curr_lat, tec] = np.float64(v) * 10 ** expo
                        tec += 1
                curr_lat += 1

        if description == 'END OF TEC MAP':
            result['gpsweek'].append(gpsweek)
            result['gpssecs'].append(gpssecs)
            if flip[1]:
                tec_map = np.flipud(tec_map)
            if flip[2]:
                tec_map = np.fliplr(tec_map)
            result['data'].append(RegularGridInterpolator((result['lat'], result['lon']),
                                                          tec_map, bounds_error=False, fill_value=0))
            result['mapnum'].append(k)
            is_tec = False

    result['gpsweek'] = np.array(result['gpsweek'])
    result['gpssecs'] = np.array(result['gpssecs'])
    result['mapnum'] = np.array(result['mapnum'])

    return result
