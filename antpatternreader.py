from imageio.v2 import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pickle



c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
BYTES_PER_SAMPLE = 4
BYTES_TO_MB = 1048576


def colorDistance(n, m):
    rm = .5 * (n[0] + m[:, :, 0])
    return np.sqrt((2 + rm / 256) * (n[0] - m[:, :, 0])**2 + 4 * (n[1] - m[:, :, 1])**2 + (2 + (255 - rm) / 256) * (n[2] - m[:, :, 2])**2)


fnme = '/home/jeff/Pictures/horn329ghzhighgain.png'

img = np.array(imread(fnme)[:, :, :3], dtype=float)

eline = np.array([198, 97, 108])
hline = np.array([101, 111, 157])
plot_dims = [(16, 398), (54, 518)]
pols = ['azimuth', 'elevation']

for idx, pol in enumerate([hline, eline]):
    thresh = colorDistance(pol, img)

    # Grab red line
    img_mask = thresh[plot_dims[0][0]:plot_dims[0][1], plot_dims[1][0]:plot_dims[1][1]] < 50

    pts = np.where(img_mask)
    xes, idxes = np.unique(pts[1], return_index=True)
    yes = pts[0][idxes]

    poly = CubicSpline(xes, yes)

    x = np.linspace(0, max(xes), 1000)

    plt.figure()
    plt.imshow(img / 256)
    plt.scatter(xes + plot_dims[1][0], yes + plot_dims[0][0])
    plt.plot(x + plot_dims[1][0], poly(x) + plot_dims[0][0])

    pic_dict = {'range': (x / max(x) - .5) * np.pi / 2, 'vals': -poly(x) / plot_dims[0][1] * 50}

    with open(f'./{fnme.split("/")[-1].split(".")[0]}_{pols[idx]}.pic', 'wb') as f:
        pickle.dump(pic_dict, f)


_lambda = c0 / fc
k = 2 * np.pi / _lambda
A = 3 * _lambda
B = 3 * _lambda
Rh = Re = 1
Ea = lambda x, y: np.cos(np.pi * x / A) * np.exp(-1j * k / 2 * (x**2 / Rh + y**2 / Re))

azes = np.linspace(-np.pi / 2, np.pi / 2, 100)
eles = np.linspace(-np.pi / 2, np.pi / 2, 100)
E = np.zeros((100,), dtype=np.complex128)
for idx in range(100):
    E[idx] = k / (4 * np.pi) * (1 + np.cos(azes[idx])) * dblquad(lambda x, y: Ea(x, y) * np.exp(1j * k * (x * np.sin(azes[idx]))),
                                                                 -A / 2, A / 2)
