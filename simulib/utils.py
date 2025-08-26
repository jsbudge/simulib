import numpy as np

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
GRAVITIC_CONSTANT = 9.80665


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

