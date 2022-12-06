from imageio.v2 import imread
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pickle


def colorDistance(n, m):
    rm = .5 * (n[0] + m[:, :, 0])
    return np.sqrt((2 + rm / 256) * (n[0] - m[:, :, 0])**2 + 4 * (n[1] - m[:, :, 1])**2 + (2 + (255 - rm) / 256) * (n[2] - m[:, :, 2])**2)


fnme = '/home/jeff/Pictures/horn329ghz.png'

img = np.array(imread(fnme)[:, :, :3], dtype=float)

eline = np.array([198, 97, 108])
hline = np.array([101, 111, 157])
plot_dims = [(49, 556), (83, 695)]
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