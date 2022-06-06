import numpy as np
from simulation_functions import getElevation, llh2enu, genPulse, findPowerOf2, db, azelToVec, PlotWithSliders
from cuda_kernels import genRangeWithoutIntersection, getMaxThreads
from grid_helper import MapEnvironment, SDREnvironment
from platform_helper import RadarPlatform, SDRPlatform
import open3d as o3d
from scipy.interpolate import CubicSpline
import cupy as cupy
import cupyx.scipy.signal
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states
import matplotlib.pyplot as plt
from scipy.signal.windows import taylor
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 1e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254

# The antenna can run Ka and Ku
# on individual pulses
bg_file = '/data5/SAR_DATA/2022/03282022/SAR_03282022_122555.sar'

# Generate the background for simulation
bg = SDREnvironment(bg_file, num_vertices=500000)

# Generate a platform
rp = SDRPlatform(bg_file, bg.origin)

# Grab vertices and such
vertices = bg.vertices
triangles = bg.triangles
normals = bg.normals

init_enu = llh2enu(*init_llh, bg.origin)