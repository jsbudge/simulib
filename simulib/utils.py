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
