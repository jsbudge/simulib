import numpy as np
from scipy.special import gamma as gam_func
from scipy.interpolate import interpn

# DEFINES
_float = np.float64
_complex_float = np.complex128

# CONSTANTS
THREADS_PER_BLOCK = 512
BLOCK_MULTIPLIER = 64
MAX_REGISTERS = 128
c0 = _float(299792458.0)
c0_inv = _float(1. / c0)
c0_half = _float(c0 / 2.)
TAC = 125e6
fs = 2e9
DTR = np.pi / 180

# FUNCTIONS

def loadOceanBackground(sample_pts, bg_ext, t, rand_vecs, repetition_T=10.):
    fsz = rand_vecs[0].shape
    bg = wavefunction(bg_ext, npts=fsz, rand_vecs=rand_vecs, T=repetition_T)
    zo = 1 / np.sqrt(2) * (bg[0] * np.exp(1j * bg[2] * t) + bg[1].conj() * np.exp(-1j * bg[2] * t))
    zo[0, 0] = 0
    bg = np.fft.ifft2(np.fft.fftshift(zo)).real
    bg = bg / np.max(bg) * 2
    return interpn((np.linspace(0, bg_ext[0], fsz[0]), np.linspace(0, bg_ext[1], fsz[1])), bg, sample_pts, method='cubic')


def wavefunction(sz, npts=(64, 64), rand_vecs=None, T=10., S=2, u10=10):
    kx = np.arange(-(npts[0] // 2 - 1), npts[0] / 2 + 1) * 2 * np.pi / sz[0]
    ky = np.arange(-(npts[1] // 2 - 1), npts[1] / 2 + 1) * 2 * np.pi / sz[1]
    kkx, kky = np.meshgrid(kx, ky)
    if rand_vecs is None:
        rho = np.random.randn(*kkx.shape)
        sig = np.random.randn(*kkx.shape)
    else:
        rho, sig = rand_vecs
    omega = np.floor(np.sqrt(9.8 * np.sqrt(kkx ** 2 + kky ** 2)) / (2 * np.pi / T)) * (2 * np.pi / T)
    zo = 1 / np.sqrt(2) * (rho - 1j * sig) * np.sqrt(var_phi(kkx, kky, S, u10))
    zoc = 1 / np.sqrt(2) * (rho + 1j * sig) * np.sqrt(var_phi(-kkx, -kky, S, u10))
    return zo, zoc, omega


def var_phi(kx, ky, S=2, u10=10):
    phi = np.cos(np.arctan2(ky, kx) / 2) ** (2 * S)
    # phi = abs(kx * .01 + ky * 0.01) ** (2 * S)
    gamma = Sk(np.sqrt(kx ** 2 + ky ** 2), u10) * phi * gam_func(S + 1) / gam_func(S + .5) * np.sqrt(kx ** 2 + ky ** 2)
    gamma[gamma < 1e-10] = 0.
    gamma[np.logical_and(kx == 0, ky == 0)] = 0
    return gamma


def Sk(k, u10=1):
    # Handles DC case
    k[k == 0] = 1e-9
    g = 9.82
    om_c = .84
    Cd10 = .00144
    ust = np.sqrt(Cd10) * u10
    km = 370
    cm = .23
    lemma = 1.7 if om_c <= 1 else 1.7 + 6 * np.log10(om_c)
    sigma = .08 * (1 + 4 * om_c ** -3)
    alph_p = .006 * om_c ** .55
    alph_m = .01 * (1 + np.log(ust / cm)) if ust <= cm else .01 * (1 + 3 * np.log(ust / cm))
    ko = g / u10 ** 2
    kp = ko * om_c ** 2
    cp = np.sqrt(g / kp)
    cc = np.sqrt((g / kp) * (1 + (k / km) ** 2))
    Lpm = np.exp(-1.25 * (kp / k) ** 2)
    gamma = np.exp(-1 / (2 * sigma ** 2) * (np.sqrt(k / kp) - 1) ** 2)
    Jp = lemma ** gamma
    Fp = Lpm * Jp * np.exp(-.3162 * om_c * (np.sqrt(k / kp) - 1))
    Fm = Lpm * Jp * np.exp(-.25 * (k / km - 1) ** 2)
    Bl = .5 * alph_p * (cp / cc) * Fp
    Bh = .5 * alph_m * (cm / cc) * Fm

    return (Bl + Bh) / k ** 3


if __name__ == '__main__':

    fft_grid_sz = (32, 32)
    xx, yy = np.meshgrid(np.linspace(0, 100, 128), np.linspace(0, 100, 128))
    bgpts = np.array([xx.flatten(), yy.flatten()]).T
    rand_vec = (np.random.randn(*fft_grid_sz), np.random.randn(*fft_grid_sz))


    import matplotlib.pyplot as plt
    import matplotlib as mplib
    mplib.use('TkAgg')

    for t in np.linspace(0, 100, 1000):
        test = loadOceanBackground(bgpts, (100, 100), t, rand_vecs=rand_vec, repetition_T=100.)
        plt.clf()
        plt.title(f'{t}')
        plt.imshow(test.reshape(xx.shape))
        plt.draw()
        plt.pause(.1)