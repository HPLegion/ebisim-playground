import numpy as np
from numba.extending import get_cython_function_address
from numba import vectorize
import ctypes
# from scipy.special import erf
PI = np.pi
_INVSQRTPI = 1/np.sqrt(PI)

_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erf")
_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
erf = _functype(_addr)

@vectorize(nopython=True)
def f(z, y):
    if z < y:
        e1 = np.exp(-(z-2*y)**2)
        r = y * (5 + 8*y**2)/z**3 * (e1 - np.exp(-4*y**2)) * _INVSQRTPI \
            + 3/(2*z**2) * (e1 - np.exp(-4*z**2)) * _INVSQRTPI \
            + 2 * y * (2*y+z)/z**2 * e1 * _INVSQRTPI \
            + 2 * y * (erf(y) - erf(2*y-z)) \
            + ((3 + 48*y**2 + 64*y**4) * (erf(2 * y - z) - erf(2*y)) + 3 * erf(z))/(4*z**3)
    else:
        r = 1/z**3 * (
            6*y**3 * np.exp(-y**2) * _INVSQRTPI
            + y * (5 + 8*y**2) * (np.exp(-y**2) - np.exp(-4*y**2)) * _INVSQRTPI
            + 3/4 * erf(2*y)
            + (1.5 + 12*y**2 + 16*y**4) * (erf(y) - erf(2*y))
        )
    return r
