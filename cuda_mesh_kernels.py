import cmath
import math
from numba import cuda, njit
from simulation_functions import findPowerOf2
import numpy as np


def calcBounceFromMesh(vertices, faces, source_xyz, bounce_xyz, bounce_dir):
    pass


def genRangeProfile():
    pass


def backprojectFromMesh():
    pass


def barycentric(x0, x1, x2, a0, a1, a2):
    return x0 * a0 + x1 * a1 + x2 * a2


def triNormal(x0, x1, x2):
    """
    Calculate a unit normal vector for a triangle.
    """
    A = x1 - x0
    B = x2 - x0
    nx = A[1] * B[2] - A[2] * B[1]
    ny = A[2] * B[0] - A[0] * B[2]
    nz = A[0] * B[1] - A[1] * B[0]
    det = 1 / math.sqrt(nx * nx + ny * ny + nz * nz)
    return nx * det, ny * det, nz * det


def bounceVector(x0, n0):
    """
    Returns the unit vector bouncing off of a normal vector.
    This assumes the normal vector given is a unit vector.
    """
    u = sum(x0 * n0) * n0
    return x0 - 2 * u

