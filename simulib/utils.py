import numpy as np
from sdrparse import load, SDRParse
import functools as ftools

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


def fftn_n( arr ):
    return np.fft.fftn( arr, norm='ortho' )

def ifftn_n( arr ):
    return np.fft.ifftn( arr, norm='ortho' )


chirp = np.mgrid[ 0:1, 0:1, 0:1 ]
chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )

pref0 = 'chirp = tuple( np.fft.fftshift( this )**2 / this.shape[n] for n, this in enumerate( np.mgrid[ '
suff0 = ' ] ) )'

DoNothing = lambda x: x
opdict = { 0:DoNothing, 1:fftn_n, 2:np.flip, 3:ifftn_n }

def frft( arr, alpha ):
    if arr.shape != chirp[0].shape:
        RecalculateChirp( arr.shape )
    ops = CanonicalOps( alpha )
    return frft_base( ops[0]( arr ), ops[1] )

def frft_base( arr, alpha ):
    phi = alpha * np.pi/2.
    cotphi = 1. / np.tan( phi )
    cscphi = np.sqrt( 1. + cotphi**2 )
    scale = np.sqrt( 1. - 1.j*cotphi ) / np.sqrt( np.prod( arr.shape ) )
    modulator = ChirpFunction( cotphi - cscphi )
    filtor = ChirpFunction( cscphi )
    arr_frft = scale * modulator * ifftn_n( fftn_n( filtor ) * fftn_n( modulator * arr ) )
    return arr_frft

def ChirpFunction( x ):
    return np.exp( x * chirp_arg )

def RecalculateChirp( newshape ):
    global chirp_arg
    if len( newshape ) == 1:    # extra-annoying string manipulations needed with 1D data
        pref = pref0.replace( 'np.', '( np.' )
        suff = suff0.replace( ']', '], )' )
    else:
        pref = pref0
        suff = suff0
    regrid = ','.join( tuple( '-%d:%d'%(n//2,n//2) for n in newshape ) ).join( [ pref, suff ] )
    #print( regrid )
    exec( regrid, globals() )
    chirp_arg = 1.j * np.pi * ftools.reduce( lambda x, y: x+y, chirp )
    return

def CanonicalOps( alpha ):
    alpha_0 = alpha % 4.
    if alpha_0 < 0.5:
        return[ ifftn_n, 1.+alpha_0 ]
    flag = 0
    while alpha_0 > 1.5:
        alpha_0 -= 1.
        flag += 1
    return [ opdict[flag], alpha_0 ]


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

