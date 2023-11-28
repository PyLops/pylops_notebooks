import logging
import numpy as np
from pylops import LinearOperator

try:
    from numba import jit
    from _DirectionalSmoothing_numba import _dirsmoothing2d_forw_numba, \
        _dirsmoothing2d_adj_numba
except ModuleNotFoundError:
    jit = None


def _dirfilts1(xfilt, theta, a, b):
    """Create inpulse response of x-oriented directional filters
    """
    f1 = np.exp(-xfilt ** 2 * (np.cos(theta) ** 2 / a ** 2 +
                               np.sin(theta) ** 2 / b ** 2))
    f4 = xfilt * f1
    return f1, f4

def _dirfilts2(zfilt, theta, a, b):
    """Create inpulse response of z-oriented directional filters
    """
    f2, f5 = _dirfilts1(zfilt, theta, b, a)
    return f2, f5

def _dirsmoothing2d_forw(imp, zfilt, xfilt, a, b, theta):
    """Apply 2d forward directional smoothing

    Parameters
    ----------
    inp : :obj:`numpy.ndarray`
        Input array
    zfilt : :obj:`int`
        Axis of filter to be applied along z-axis
    xfilt : :obj:`int`
        Axis of filter to be applied along x-axis
    a : :obj:`tuple`
        Ellipse axis along z-axis
    b : :obj:`int`, optional
        Ellipse axis along x-axis
    theta : :obj:`numpy.ndarray`
        Angles along which non-stationary directional smoothing is applied

    """
    nzfilt, nxfilt = zfilt.size, xfilt.size

    out = np.zeros_like(imp)
    alphabase = (a**2 - b**2) / (a**2 * b**2)
    alpha = alphabase * np.sin(2*theta)
    imp1 = imp * alpha
    imp = np.pad(imp, ((nzfilt//2, nzfilt//2),
                       (nxfilt//2, nxfilt//2)), mode='constant')
    imp1 = np.pad(imp1, ((nzfilt//2, nzfilt//2),
                         (nxfilt//2, nxfilt//2)), mode='constant')
    theta = np.pad(theta, ((nzfilt//2, nzfilt//2),
                           (nxfilt//2, nxfilt//2)), mode='constant')
    out1 = np.zeros_like(imp)
    out2 = np.zeros_like(imp)
    for izout, iz in enumerate(range(nzfilt//2, imp.shape[0]-nzfilt//2)):
        for ixout, ix in enumerate(range(nxfilt//2, imp.shape[1]-nxfilt//2)):
            f2, f5 = _dirfilts2(np.flipud(zfilt),
                                theta[iz-nzfilt//2:iz+nzfilt//2+1, ix], a, b)
            out1[iz, ix] = np.sum(imp[iz-nzfilt//2:iz+nzfilt//2+1, ix] * f2)
            out2[iz, ix] = np.sum(imp1[iz-nzfilt//2:iz+nzfilt//2+1, ix] * f5)
    for izout, iz in enumerate(range(nzfilt//2, imp.shape[0]-nzfilt//2)):
        for ixout, ix in enumerate(range(nxfilt//2, imp.shape[1]-nxfilt//2)):
            f1, f4 = _dirfilts1(np.flipud(xfilt),
                                theta[iz, ix-nxfilt//2:ix+nxfilt//2+1], a, b)
            out[izout, ixout] = np.sum(out1[iz, ix-nxfilt//2:ix+nxfilt//2+1] * f1) - \
                  np.sum(out2[iz, ix-nxfilt//2:ix+nxfilt//2+1] * f4)
    return out

def _dirsmoothing2d_adj(imp, zfilt, xfilt, a, b, theta):
    """Apply 2d adjoint directional smoothing

    Parameters
    ----------
    inp : :obj:`numpy.ndarray`
        Input array
    zfilt : :obj:`int`
        Axis of filter to be applied along z-axis
    xfilt : :obj:`int`
        Axis of filter to be applied along x-axis
    a : :obj:`tuple`
        Ellipse axis along z-axis
    b : :obj:`int`, optional
        Ellipse axis along x-axis
    theta : :obj:`numpy.ndarray`
        Angles along which non-stationary directional smoothing is applied

    """
    nzfilt, nxfilt = zfilt.size, xfilt.size

    out = np.zeros_like(imp)
    imp = np.pad(imp, ((nzfilt//2, nzfilt//2),
                       (nxfilt//2, nxfilt//2)), mode='constant')
    out1 = np.zeros_like(imp)
    out2 = np.zeros_like(imp)
    alphabase = (a**2 - b**2) / (a**2 * b**2)
    for izout, iz in enumerate(range(nzfilt//2, imp.shape[0]-nzfilt//2)):
        for ixout, ix in enumerate(range(nxfilt//2, imp.shape[1]-nxfilt//2)):
            f1, f4 = _dirfilts1(xfilt, theta[izout, ixout], a, b)
            out1[iz, ix] = np.sum(imp[iz, ix-nxfilt//2:ix+nxfilt//2+1] * f1)
            out2[iz, ix] = np.sum(imp[iz, ix-nxfilt//2:ix+nxfilt//2+1] * f4)
    for izout, iz in enumerate(range(nzfilt//2, imp.shape[0]-nzfilt//2)):
        for ixout, ix in enumerate(range(nxfilt//2, imp.shape[1]-nxfilt//2)):
            f2, f5 = _dirfilts2(zfilt, theta[izout, ixout], a, b)
            alpha = alphabase * np.sin(2*theta[izout, ixout])
            out[izout, ixout] = np.sum(out1[iz-nzfilt//2:iz+nzfilt//2+1, ix] * f2) - \
                  alpha * np.sum(out2[iz-nzfilt//2:iz+nzfilt//2+1, ix] * f5)
    return out


class DirectionalSmoothing(LinearOperator):
    r"""Directional Smoothing.

    Apply directional smoothing to a multi-dimensional array
    (at least 2 dimensions are required) along either a single common
    angle or different angles for each point of the array.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension.
    theta : :obj:`float` or :obj:`np.ndarray`, optional
        Single angle or group of angles
        (array of size :math:`[n_{d0} \times ... \times n_{n_{dims}}`)
    filtlenghts : :obj:`tuple`
        Length of filter along each direction (must be odd).
    filtaxes: :obj:`tuple`
        Extent of ellipse axis along each direction.
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``numba``). Note that
        ``numba`` can only be used when providing a look-up table
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba``

    Notes
    -----
    The DirectionalSmoothing operator is a stationary convolutional operator
    that performs directional smoothing along a single angle (or
    non-stationary convolutional operator that performs directional smoothing
    along a different angle for each position).

    The filter is composed of two separable filters [1]_:

    .. math::
        h(z, x) = h_1(z) * h_2(x) - \alpha z h_1(z) * x h_2(x)

    where :math:`h_1(z) = e^{-z^2(\frac{cos(\theta)^2}{a^2} +
    \frac{sin(\theta)^2}{b^2})}`,
    :math:`h_2(x) = e^{-x^2(\frac{cos(\theta)^2}{b^2} +
    \frac{sin(\theta)^2}{a^2})}`, and
    :math:`alpha = (a^2-b^2)sin(2\theta)/(a^2b^2).

    In forward mode, the filters along the second direction (x)
    are first applied to the model and the filters first direction (z)
    are subsequently applied to the outputs of the previous filters and
    subtracted. In adjoint mode, the order of operations is simply reversed.

    .. [1] Lakshmanan, V ., "A Separable Filter for Directional Smoothing",
        IEEE Geoscience and Remote Sensing Letters, vol. 1, pp. 192-195, 2004.

    """
    def __init__(self, dims, theta, filtlenghts, filtaxes,
                 engine='numpy', dtype='float64'):
        if engine not in ['numpy', 'numba']:
            raise KeyError('engine must be numpy or numba')
        if engine == 'numba' and jit is not None:
            self.engine = 'numba'
            self._matvecf = _dirsmoothing2d_forw_numba
            self._rmatvecf = _dirsmoothing2d_adj_numba
        else:
            if engine == 'numba' and jit is None:
                logging.warning('numba not available, revert to numpy...')
            self.engine = 'numpy'
            self._matvecf = _dirsmoothing2d_forw
            self._rmatvecf = _dirsmoothing2d_adj
        self.dims = dims
        if isinstance(theta, (int, float)):
            self.theta = theta * np.ones(self.dims)
        else:
            self.theta = theta
        self.filtlenghts = \
            [filtlenght + (1 - filtlenght % 2) for filtlenght in filtlenghts]

        self.zfilt = \
            np.arange(0, self.filtlenghts[0]) - self.filtlenghts[0] // 2
        self.xfilt = \
            np.arange(0, self.filtlenghts[1]) - self.filtlenghts[1] // 2
        self.filtaxes = filtaxes
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = self._matvecf(x, self.zfilt, self.xfilt,
                          self.filtaxes[0], self.filtaxes[1], self.theta)
        y = y.ravel()
        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims)
        y = self._rmatvecf(x, self.zfilt, self.xfilt,
                           self.filtaxes[0], self.filtaxes[1], self.theta)
        y = y.ravel()
        return y
