import numpy as np
from sdrparse import load, SDRParse
from sdrparse.SARParsing import SARParse
import functools as ftools

# DEFINES
_float = np.float32
_complex_float = np.complex64

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


def getRadarAndEnvironment(sdr_file: [SDRParse, str], a_channel: int = 0, is_sdr: bool = True) -> tuple | None:
    from .platform_helper import SDRPlatform, SARPlatform
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

    a_rp = SDRPlatform(a_sdr, a_bg.ref, channel=a_channel) if is_sdr else SARPlatform(a_sdr, a_bg.ref, channel=a_channel)
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


def design_element_positions(
        a_design_freq_hz: float, a_sub_num_elements_wide: int,
        a_sub_num_elements_high: int, a_num_sub_wide: int,
        a_num_sub_high: int) -> tuple:
    f_wavelength_m = c0 / a_design_freq_hz
    # Design element width (quarter wavelength for half wavelength spacing to
    #   avoid grating lobes)
    element_width_m = f_wavelength_m / 4
    element_height_m = element_width_m
    x_spacing_m = element_width_m * 2
    y_spacing_m = x_spacing_m
    num_high = a_sub_num_elements_high * a_num_sub_high
    num_wide = a_sub_num_elements_wide * a_num_sub_wide
    antenna_height_m = num_high * y_spacing_m
    antenna_width_m = num_wide * x_spacing_m
    sub_array_height_m = a_sub_num_elements_high * y_spacing_m
    sub_array_width_m = a_sub_num_elements_wide * x_spacing_m

    # Get the positions of the elements in the full array using the antenna
    #   parameters above and assuming 0, 0 is in the center of the array
    #   with positive y-axis up, and positive x-axis to the right, and the
    #   z-axis out perpendicular from the array
    sub_elem_y_pos_m = np.arange(a_sub_num_elements_high) * y_spacing_m
    sub_elem_x_pos_m = np.arange(a_sub_num_elements_wide) * x_spacing_m
    sub_elem_x_pos_2d_m, sub_elem_y_pos_2d_m = np.meshgrid(
        sub_elem_x_pos_m, sub_elem_y_pos_m)
    elem_positions_m = np.zeros((
        a_num_sub_high, a_num_sub_wide, 3, a_sub_num_elements_high,
        a_sub_num_elements_wide))
    for row in range(a_num_sub_high):
        for col in range(a_num_sub_wide):
            # Combine them to get a 3 x num_high x num_wide array
            elem_positions_m[row, col] = \
                np.array([
                    sub_elem_x_pos_2d_m + col * sub_array_width_m
                    - antenna_width_m / 2 + element_width_m,
                    sub_elem_y_pos_2d_m + row * sub_array_height_m
                    - antenna_height_m / 2 + element_height_m,
                    np.zeros_like(sub_elem_x_pos_2d_m)])

    return (
        np.array(elem_positions_m), antenna_width_m, antenna_height_m,
        element_width_m)


def get_element_phases(
        a_element_pos_m: np.ndarray, a_point_theta_r: float,
        a_point_phi_r: float,
        a_wavelength_m: float) -> tuple[np.ndarray, np.ndarray]:
    # Compute the wave number and the steering vector
    k = 2 * np.pi / a_wavelength_m
    steering_vector = np.array([
        np.sin(a_point_theta_r) * np.cos(a_point_phi_r),
        np.sin(a_point_theta_r) * np.sin(a_point_phi_r),
        np.cos(a_point_theta_r)])

    # Compute the distance travelled to each element
    element_phases_r = np.einsum(
        "nmijk,i->nmjk", a_element_pos_m, steering_vector) * k

    return element_phases_r, np.exp(-1j * element_phases_r)


def patch_pattern(
        a_obs_vectors_mbs: np.ndarray, a_patch_size_m: float,
        a_wavelength_m: float) -> np.ndarray:
    """
    Far-field radiation pattern of a square micro-strip patch (approximate
      model)

    :param a_obs_vectors_mbs: 3xNxM array of steering vectors within the
      antenna frame
    :param a_patch_size_m: Patch side length (meters).
    :param a_wavelength_m: Operating wavelength (meters).
    :return: NxM array of the magnitude of the element pattern.
    """
    k0 = 2 * np.pi / a_wavelength_m

    # Avoid division-by-zero in sinc
    def sinc(x):
        return np.sinc(x / np.pi)

    fx = sinc(0.5 * k0 * a_patch_size_m * a_obs_vectors_mbs[0, ...])
    fy = sinc(0.5 * k0 * a_patch_size_m * a_obs_vectors_mbs[1, ...])

    # cos(theta) factor for patch tilt
    ant_response = np.abs(fx * fy * a_obs_vectors_mbs[2, ...])

    return ant_response


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

