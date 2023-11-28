import os
import numpy as np
from numba import jit, prange

# detect whether to use parallel or not
numba_threads = int(os.getenv('NUMBA_NUM_THREADS', '1'))
parallel = True if numba_threads != 1 else False


@jit(nopython=True, parallel=parallel, nogil=True)
def _dirfilts1(xfilt, theta, a, b):
    """numba implementation of _dirfilts1.
    See official documentation for description of variables
    """
    f1 = np.exp(-xfilt ** 2 * (np.cos(theta) ** 2 / a ** 2 +
                               np.sin(theta) ** 2 / b ** 2))
    f4 = xfilt * f1
    return f1, f4

@jit(nopython=True, parallel=parallel, nogil=True)
def _dirfilts2(zfilt, theta, a, b):
    """numba implementation of _dirfiltss.
    See official documentation for description of variables
    """
    f2, f5 = _dirfilts1(zfilt, theta, b, a)
    return f2, f5

@jit(nopython=True, parallel=parallel, nogil=True)
def _dirsmoothing2d_forw_numba(imp, zfilt, xfilt, a, b, theta):
    """numba implementation of _dirsmoothing2d_forw.
    See official documentation for description of variables
    """
    nz, nx  = imp.shape
    nzfilt, nxfilt = zfilt.size, xfilt.size
    out = np.zeros_like(imp)
    alphabase = (a**2 - b**2) / (a**2 * b**2)
    alpha = alphabase * np.sin(2*theta)
    imp1 = imp * alpha
    zfilt = zfilt[::-1]
    xfilt = xfilt[::-1]
    out1 = np.zeros_like(imp)
    out2 = np.zeros_like(imp)
    for iz in range(0, nz):
        izmin = iz - nzfilt//2
        if izmin < 0:
            izfiltmin = -izmin
            izmin = 0
        else:
            izfiltmin = 0
        izmax = iz + nzfilt//2 + 1
        if izmax > nz:
            izfiltmax = nzfilt - (izmax - (nz - 1)) + 1
            izmax = nz
        else:
            izfiltmax = nzfilt
        for ix in range(nx):
            f2, f5 = _dirfilts2(zfilt[izfiltmin:izfiltmax],
                                theta[izmin:izmax, ix], a, b)
            out1[iz, ix] = np.sum(imp[izmin:izmax, ix] * f2)
            out2[iz, ix] = np.sum(imp1[izmin:izmax, ix] * f5)
    for iz in range(0, nz):
        for ix in range(nx):
            ixmin = ix - nxfilt//2
            if ixmin < 0:
                ixfiltmin = -ixmin
                ixmin = 0
            else:
                ixfiltmin = 0
            ixmax = ix + nxfilt//2 + 1
            if ixmax > nx:
                ixfiltmax = nxfilt - (ixmax - (nx - 1)) + 1
                ixmax = nx
            else:
                ixfiltmax = nxfilt
            f1, f4 = _dirfilts1(xfilt[ixfiltmin:ixfiltmax],
                                theta[iz, ixmin:ixmax], a, b)
            out[iz, ix] = np.sum(out1[iz, ixmin:ixmax] * f1) - \
                  np.sum(out2[iz, ixmin:ixmax] * f4)
    return out

@jit(nopython=True, parallel=parallel, nogil=True)
def _dirsmoothing2d_adj_numba(imp, zfilt, xfilt, a, b, theta):
    """numba implementation of _dirsmoothing2d_adj.
    See official documentation for description of variables
    """
    nz, nx  = imp.shape
    nzfilt, nxfilt = zfilt.size, xfilt.size
    out = np.zeros_like(imp)
    alphabase = (a**2 - b**2) / (a**2 * b**2)
    out1 = np.zeros_like(imp)
    out2 = np.zeros_like(imp)
    for iz in range(0, nz):
        for ix in range(nx):
            ixmin = ix - nxfilt//2
            if ixmin < 0:
                ixfiltmin = -ixmin
                ixmin = 0
            else:
                ixfiltmin = 0
            ixmax = ix + nxfilt//2 + 1
            if ixmax > nx:
                ixfiltmax = nxfilt - (ixmax - (nx - 1)) + 1
                ixmax = nx
            else:
                ixfiltmax = nxfilt
            f1, f4 = _dirfilts1(xfilt[ixfiltmin:ixfiltmax],
                                theta[iz, ix], a, b)
            out1[iz, ix] = np.sum(imp[iz, ixmin:ixmax] * f1)
            out2[iz, ix] = np.sum(imp[iz, ixmin:ixmax] * f4)
    for iz in range(0, nz):
        izmin = iz - nzfilt//2
        if izmin < 0:
            izfiltmin = -izmin
            izmin = 0
        else:
            izfiltmin = 0
        izmax = iz + nzfilt//2 + 1
        if izmax > nz:
            izfiltmax = nzfilt - (izmax - (nz - 1)) + 1
            izmax = nz
        else:
            izfiltmax = nzfilt
        for ix in range(nx):
            f2, f5 = _dirfilts2(zfilt[izfiltmin:izfiltmax],
                                theta[iz, ix], a, b)
            alpha = alphabase * np.sin(2*theta[iz, ix])
            out[iz, ix] = np.sum(out1[izmin:izmax, ix] * f2) - \
                  alpha * np.sum(out2[izmin:izmax, ix] * f5)
    return out