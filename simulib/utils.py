import numpy as np
from sdrparse import load, SDRParse

# DEFINES
_float = np.float64
_complex_float = np.complex128

# CONSTANTS
THREADS_PER_BLOCK = 512
BLOCK_MULTIPLIER = 64
INS_REFRESH_HZ = 100
MAX_REGISTERS = 128
c0 = _float(299792458.0)
c0_inv = _float(1. / c0)
c0_half = _float(c0 / 2.)
TAC = 125e6
fs = 2e9
DTR = np.pi / 180
GRAVITIC_CONSTANT = 9.80665
MAX_DISTANCE = 1e6
inch_to_m = .0254
m_to_ft = 3.2808


def getRadarAndEnvironment(sdr_file: [SDRParse, str], a_channel: int = 0) -> tuple | None:
    from .platform_helper import SDRPlatform
    from .grid_helper import SDREnvironment

    # Load SAR file into SDRParse object
    if isinstance(sdr_file, str):
        try:
            a_sdr = load(sdr_file, progress_tracker=True)
        except Exception as ex:
            print(ex)
            return None
    else:
        a_sdr = sdr_file
    # Load environment
    a_bg = SDREnvironment(a_sdr)

    # Load the platform
    a_rp = SDRPlatform(a_sdr, a_bg.ref, channel=a_channel)
    return a_bg, a_rp


if __name__ == '__main__':

    ocean_stack, xx, yy = genOceanBackground((100, 100), np.linspace(0, 100, 1000), repetition_T=1000, S=2., u10=10., rect_grid=True)


    import matplotlib.pyplot as plt
    import matplotlib as mplib
    from scipy.interpolate import griddata
    mplib.use('TkAgg')

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})


    for o in ocean_stack:
        ax.clear()
        ax.set_zlim(-2., 40.)
        ax.plot_surface(xx, yy, o, cmap=mplib.cm.ocean)
        # plt.clf()
        # plt.imshow(test.reshape(xx.shape))
        plt.draw()
        plt.pause(.1)

