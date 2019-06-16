import logging
import warnings
import numpy as np

from scipy.sparse.linalg import lsqr
from scipy.ndimage.filters import convolve1d as sp_convolve1d

from pylops import LinearOperator, Diagonal, Identity, Transpose
from pylops.signalprocessing import FFT, Fredholm1
from pylops.utils import dottest as Dottest
from pylops.optimization.leastsquares import PreconditionedInversion

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class MDC(LinearOperator):
    r"""Multi-dimensional convolution.

    Apply multi-dimensional convolution between two seismic data. The input
    model and data must be obtained from 2- or 3-dimensional arrays
    of size :math:`[n_r \times n_{vs} \times n_t]` and
    :math:`[n_s \times n_{vs} \times n_t]`, respectively.

    .. warning:: This class will be deprecated and overwritten by
      :class:`pylops.waveeqprocessing.MDCchain` in version 2.0.0. Until then,
      it is reccomended to start using the new operator importing it with its
      future name ``import pylops.waveqprocessing.MDCchain as MDC``

    Parameters
    ----------
    G : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel in frequency domain of size
        :math:`[n_s \times n_r \times n_{fmax}]`
    nt : :obj:`int`
        Number of samples along time axis
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`
        Sampling of time integration axis
    dr : :obj:`float`
        Sampling of receiver integration axis
    twosided : :obj:`bool`
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    fast : :obj:`bool`
        Fast application of MDC when model has only one virtual
        source (``True``) or not (``False``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    ns : :obj:`int`
        Number of samples along source axis
    nr : :obj:`int`
        Number of samples along receiver axis
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    The so-called multi-dimensional convolution (MDC) is a chained
    operator [1]_. It is composed of a forward Fourier transform,
    a multi-dimensional integration, and an inverse Fourier transform:

    .. math::
        y(s,v,f) = \int_S R(s,r,f) x(r,v,f) dr

    y(f, s, v) = \mathscr{F}^{-1} \Big( \int_S R(f, s, r)
        \mathscr{F}(x(f, r, v)) dr \Big)


    This operation can be discretized and performed by means of a
    linear operator

    .. math::
        \mathbf{D}= \mathbf{F}^H  \mathbf{R} \mathbf{F}

    where :math:`\mathbf{F}` is the Fourier transform applied along
    the time axis and :math:`\mathbf{R}` is the multi-dimensional
    convolution kernel.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a
       systematic comparison", Geophyscial Journal International, vol. 185,
       pp. 1335-1364. 2011.

    """
    def __init__(self, G, nt, nv, dt=1., dr=1.,
                 twosided=True, fast=False, dtype='float64'):
        warnings.warn('Deprecate, use new MDC...',
                      DeprecationWarning, stacklevel=2)

        if twosided and nt % 2 == 0:
            raise ValueError('nt must be odd number')
        self.G = G
        self.ns, self.nr, self.nfmax = G.shape
        self.nt = nt
        self.nv = nv
        self.dt = dt
        self.dr = dr

        self.shape = (self.ns*self.nv*self.nt, self.nr*self.nv*self.nt)
        self.twosided = twosided
        self.fast = fast
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.squeeze(np.reshape(x, (self.nr, self.nv, self.nt)))
        if self.twosided:
            x = np.fft.ifftshift(x, axes=-1)
        x = np.sqrt(1./self.nt)*np.fft.rfft(x, self.nt, axis=-1)
        x = x[..., :self.nfmax]
        if self.nv == 1 and self.fast:
            y = self.dr * self.dt * np.sqrt(self.nt) * \
                np.sum(self.G * np.tile(x, [self.ns, 1, 1]), axis=1)
        else:
            y = np.squeeze(np.zeros((self.ns, self.nv, x.shape[-1]),
                                    dtype=np.complex128))
            for it in range(self.nfmax):
                y[..., it] = self.dr * self.dt * np.sqrt(self.nt) * \
                             np.dot(self.G[:, :, it], x[..., it])

        y = np.real(np.fft.irfft(y, self.nt, axis=-1) * np.sqrt(self.nt))
        y = np.ndarray.flatten(y)
        return y

    def _rmatvec(self, x):
        x = np.squeeze(np.reshape(x, (self.ns, self.nv, self.nt)))
        x = np.sqrt(1./self.nt)*np.fft.rfft(x, self.nt, axis=-1)
        x = x[..., :self.nfmax]

        if self.nv == 1 and self.fast:
            y = self.dr * self.dt * np.sqrt(self.nt) * \
                np.sum(np.conj(self.G) * np.tile(x[:, np.newaxis, :],
                                                 [1, self.nr, 1]), axis=0)
        else:
            y = np.squeeze(np.zeros((self.nr, self.nv, x.shape[-1]),
                                    dtype=np.complex128))
            for it in range(self.nfmax):
                y[..., it] = self.dr * self.dt * np.sqrt(self.nt) * \
                            np.dot(np.conj(self.G[:, :, it].T), x[..., it])
        y = np.fft.irfft(y, self.nt, axis=-1)* np.sqrt(self.nt)
        if self.twosided:
            y = np.fft.fftshift(y, axes=-1)
        y = np.real(y)
        y = np.ndarray.flatten(y)
        return y
