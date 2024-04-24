from SDRParsing import SDRParse, load
import numpy as np
import pandas as pd
from glob import glob
from astropy.time import Time
import matplotlib.pyplot as plt
from simulation_functions import llh2enu, db, enu2llh
from tqdm import tqdm
from backproject_functions import runBackproject, getRadarAndEnvironment
from scipy.interpolate import LinearNDInterpolator
import cupy
from scipy.optimize import minimize

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

fdir = ['/data6/SAR_DATA/2024/04092024']
opt_file = '/data6/SAR_DATA/2024/04092024/SAR_04092024_143605.sar'
origin = (30.562588, -86.436398, 22)
channel = 0

csv_pos = pd.DataFrame()
sar_files = []
csv_files = []
for f in fdir:
    sar_files += glob(f'{f}/SAR_*.sar')
    csv_files += glob(f'{f}/*_R*.csv')

for c in csv_files:
    csv_pos = pd.concat([csv_pos, pd.read_csv(c)], ignore_index=True)
csv_pos['t_gps'] = 0

# Prep the csv_file positions
for row in tqdm(csv_pos.itertuples()):
    csv_pos.loc[row[0], 't_gps'] = Time(f'{row[2]} {row[1]}', format='iso', scale='utc').gps + (5 * 60 * 60)
csv_pos['gps_wk'] = csv_pos['t_gps'] // 604800
csv_pos['gps_sec'] = csv_pos['t_gps'] % 604800

gps_df = pd.DataFrame()
sdr_opt = None
for f in tqdm(sar_files):
    try:
        sdr = load(f, progress_tracker=True, use_jump_correction=False)
        gps_df = pd.concat([gps_df, sdr.gps_data])
    except ValueError:
        continue
    if sdr_opt is None and f == opt_file:
        sdr_opt = sdr

e, n, u = llh2enu(gps_df['lat'], gps_df['lon'], gps_df['alt'], origin)
gps_df['e'] = e
gps_df['n'] = n
gps_df['u'] = u
X = np.array([e - e.mean(), n - n.mean(), u - u.mean()]).T
eigs, evec = np.linalg.eig(X.T.dot(X))

# Fabricated data
nstart = np.array([[e.max(), n.max(), u.mean() + .5]])
fab = nstart + evec[:, 0].reshape((1, 3)) * (csv_pos['Position'].values[:, None] * .0254)

fab_pos = np.array([np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 0]),
                    np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 1]),
                    np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 2])]).T

ax = plt.figure('Full').add_subplot(projection='3d')
ax.scatter(gps_df['e'].values, gps_df['n'].values, gps_df['u'].values)
ax.scatter(fab[:, 0], fab[:, 1], fab[:, 2])

sub_loc = np.logical_and(gps_df.index >= sdr_opt.gps_data.index.min(), gps_df.index <= sdr_opt.gps_data.index.max())
ax = plt.figure('Single Collect').add_subplot(projection='3d')
ax.scatter(gps_df['e'].values[sub_loc], gps_df['n'].values[sub_loc],
           gps_df['u'].values[sub_loc])
ax.scatter(fab_pos[sub_loc, 0], fab_pos[sub_loc, 1], fab_pos[sub_loc, 2])

bg, rp = getRadarAndEnvironment(sdr_opt, 0)

bpj_data, _ = runBackproject(sdr_opt, rp, bg, -1.0, 0, 4, 0, 10, 10, 20, 128,
                             rotate_grid=True, debug=True, poly_num=1, origin=origin)

mag_data = np.sqrt(abs(bpj_data))

plt.figure('Original Data')
plt.imshow(db(mag_data), origin='lower', cmap='gray', clim=[100, 130])
plt.axis('tight')

# Run optimizations
lat, lon, alt = enu2llh(fab_pos[sub_loc, 0], fab_pos[sub_loc, 1], fab_pos[sub_loc, 2] + .5, origin)
scrubbed = sdr_opt.gps_data
scrubbed['lat'] = lat
scrubbed['lon'] = lon
scrubbed['alt'] = alt
sdr_opt.gps_data = scrubbed
bg, rp = getRadarAndEnvironment(sdr_opt, 0)

bpj_data, _ = runBackproject(sdr_opt, rp, bg, -1.0, 0, 4, 0, 10, 10, 20, 128,
                             a_rotate_grid=True, a_debug=True, a_poly_num=1, a_origin=origin,
                             ipr_mode=False)

mag_data = np.sqrt(abs(bpj_data))

plt.figure('Scrubbed Data')
plt.imshow(db(mag_data), origin='lower', cmap='gray', clim=[100, 140])
plt.axis('tight')
best_bpj = None
best_max = 0
best_track = None
shift = np.array([ 0.22494617,  0.02044005, -0.02133452])
lat, lon, alt = enu2llh(fab_pos[sub_loc, 0] + shift[0], fab_pos[sub_loc, 1] + shift[1], fab_pos[sub_loc, 2] + .5 + shift[2], origin)
scrubbed = sdr_opt.gps_data
scrubbed['lat'] = lat
scrubbed['lon'] = lon
scrubbed['alt'] = alt
sdr_opt.gps_data = scrubbed
bg, rp = getRadarAndEnvironment(sdr_opt, 0)

bpj_data, _ = runBackproject(sdr_opt, rp, bg, -1.0, 0, 4, 0, 10, 10, 20, 128,
                             a_rotate_grid=True, a_debug=True, a_poly_num=1, a_origin=origin,
                             ipr_mode=False)

mag_data = db(np.sqrt(abs(bpj_data)))

plt.figure('Best Scrubbed Data')
plt.imshow(mag_data, origin='lower', cmap='gray', clim=[100, 135])
plt.axis('tight')


plt.show()

try:
    bpj_wavelength = c0 / (sdr_opt[channel].fc - sdr_opt[channel].bw / 2 - sdr_opt[channel].xml['DC_Offset_MHz'] * 1e6) \
        if sdr_opt[channel].xml['Offset_Video_Enabled'].lower() == 'true' else c0 / sdr_opt[channel].fc
except KeyError as e:
    f'Could not find {e}'
    bpj_wavelength = c0 / (sdr_opt[channel].fc - sdr_opt[channel].bw / 2 - 5e6)

gx, gy, gz = bg.getGrid(origin, 10, 10, 200, 200, bg.heading, use_elevation=True)
maxpos = np.where(mag_data == mag_data.max())
maxpt = np.array([gx[maxpos], gy[maxpos], gz[maxpos]]).flatten()
nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(-1., 0, 4))


cpi_len = 2048
data_t = sdr_opt[0].pulse_time
idx_t = sdr_opt[0].frame_num

mfilt = cupy.array(sdr_opt.genMatchedFilter(channel, fft_len=fft_len), dtype=np.complex128)
motion = np.array([0, 0, 0, *evec[:, 0]])
def minfunc(motion):
    val = 0
    fab = nstart + motion[:3] + motion[3:].reshape((1, 3)) * (csv_pos['Position'].values[:, None] * .0254)

    fpos = np.array([np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 0])[sub_loc],
                        np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 1])[sub_loc],
                        np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 2])[sub_loc]])
    lat, lon, alt = enu2llh(fpos[0], fpos[1], fpos[2], origin)
    scrubbed = sdr_opt.gps_data
    scrubbed['lat'] = lat
    scrubbed['lon'] = lon
    scrubbed['alt'] = alt
    sdr_opt.gps_data = scrubbed
    bg, rp = getRadarAndEnvironment(sdr_opt, 0)
    for tidx, frames in tqdm(enumerate(idx_t[pos:pos + cpi_len] for pos in range(0, len(data_t), cpi_len)),
                             total=len(data_t) // cpi_len + 1):
        ts = data_t[tidx * cpi_len + np.arange(len(frames))]
        tmp_len = len(ts)
        rng = np.linalg.norm(rp.rxpos(ts) - maxpt, axis=1)
        bins = np.ceil((rng * 2 / c0 - 2 * near_range_s) * rp.fs * 4).astype(int)
        rtdata = cupy.fft.fft(cupy.array(sdr_opt.getPulses(frames, channel)[1], dtype=np.complex128), fft_len, axis=0) * mfilt[:, None]
        upsample_data = cupy.zeros((up_fft_len, tmp_len), dtype=np.complex128)
        upsample_data[:fft_len // 2, :] = rtdata[:fft_len // 2, :]
        upsample_data[-fft_len // 2:, :] = rtdata[-fft_len // 2:, :]
        rtdata = cupy.fft.ifft(upsample_data, axis=0).get()[:nsam * 4, :]
        exp_phase = np.exp(1j * 2 * np.pi / bpj_wavelength * rng * 2)
        val += sum(rtdata[bins, np.arange(tmp_len)] * exp_phase)
    print(f'{motion}: {10 * np.log10(abs(val))}')
    return -10 * np.log10(abs(val))

check = minimize(minfunc, motion)
best_track = check['x']
fab = nstart + best_track[:3] + best_track[3:].reshape((1, 3)) * (csv_pos['Position'].values[:, None] * .0254)

fpos = np.array([np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 0])[sub_loc],
                    np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 1])[sub_loc],
                    np.interp(gps_df.index, csv_pos['gps_sec'].values, fab[:, 2])[sub_loc]])

ax = plt.figure('Best Track').add_subplot(projection='3d')
ax.scatter(gps_df['e'].values[sub_loc], gps_df['n'].values[sub_loc],
           gps_df['u'].values[sub_loc])
ax.scatter(fpos[0], fpos[1], fpos[2])

lat, lon, alt = enu2llh(fpos[0], fpos[1], fpos[2], origin)
scrubbed = sdr_opt.gps_data
scrubbed['lat'] = lat
scrubbed['lon'] = lon
scrubbed['alt'] = alt
sdr_opt.gps_data = scrubbed
bg, rp = getRadarAndEnvironment(sdr_opt, 0)

bpj_data, _ = runBackproject(sdr_opt, rp, bg, -1.0, 0, 4, 0, 10, 10, 20, 128,
                             rotate_grid=True, debug=True, poly_num=1, origin=origin,
                             ipr_mode=False)

mag_data = np.sqrt(abs(bpj_data))

plt.figure('Scrubbed Data')
plt.imshow(db(mag_data), origin='lower', cmap='gray', clim=[100, 130])
plt.axis('tight')
