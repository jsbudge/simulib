from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from skimage.filters import gabor
import imageio
import time
from SDRParsing import loadASIFile, loadASHFile


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), option=1):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for _ in np.arange(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

    return imgout


def prepImage(asi_f: str | np.ndarray, nrows=6687, ncols=22000, scale=1e0, nbits=255) -> np.ndarray:
    if isinstance(asi_f, str):
        asi_data = loadASIFile(asi_f, nrows, ncols, scale=scale)
        mag_data = np.log(abs(asi_data) + 1)
    else:
        mag_data = asi_f + 0.0

    print('Binning...')
    plot_data = QuantileTransformer(output_distribution='normal').fit(
        mag_data[mag_data > 0].reshape(-1, 1)).transform(mag_data.reshape(-1, 1)).reshape(mag_data.shape)
    max_bin = 3
    hist_counts, hist_bins = \
        np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    while hist_counts[-1] == 0:
        max_bin -= .01
        hist_counts, hist_bins = \
            np.histogram(plot_data, bins=np.linspace(-1, max_bin, nbits))
    print('Digitizing...')
    return np.digitize(plot_data, hist_bins)


if __name__ == '__main__':
    asi_fnme = '/home/jeff/SDR_DATA/ARCHIVE/07082024/SAR_07082024_121552LVV_ch1_926500_20.asi'
    ash_file = loadASHFile(f'{asi_fnme[:-4]}.ash')

    output_image = prepImage(asi_fnme, nrows=int(ash_file['image']['nRows']), ncols=int(ash_file['image']['nCols']))
    plt.figure()
    plt.imshow(output_image, cmap='gray')
    plt.show()

    start_time = time.time()
    import bm3d
    test_image = output_image / 255.
    print('redenoising')
    denoised = np.zeros_like(test_image)
    for x in range(0, denoised.shape[0], 4096):
        for y in range(0, denoised.shape[1], 4096):
            try:
                denoised[x:x+4096, y:y+4096] = bm3d.bm3d(test_image[x:x+4096, y:y+4096], .25,
                                                         stage_arg=bm3d.BM3DStages.ALL_STAGES)
            except:
                continue
    print(f'Time done in {time.time() - start_time}')

    renoised = prepImage(denoised[:, :7630], nbits=65535)

    plt.figure('Renoised')
    plt.imshow(renoised, cmap='gray')
    plt.show()

    plt.figure('Denoised')
    plt.imshow(denoised, cmap='gray')
    plt.show()

    # stem_name = Path(asi_fnme).stem[:19]
    # imageio.imwrite(f'./data/base_{stem_name}.png', renoised.astype(np.uint16))