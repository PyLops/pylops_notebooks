import numpy as np
import pylops

# input signal parameters
ifreqs = [41, 25, 66]
amps = [1., 1., 1.]
nt = 200
nfft = 2**11
dt = 0.004
t = np.arange(nt)*dt
f = np.fft.rfftfreq(nfft, dt)

# input signal in frequency domain
X = np.zeros(nfft//2+1, dtype='complex128')
X[ifreqs] = amps

# input signal in time domain
FFTop = pylops.signalprocessing.FFT(nt, nfft=nfft, real=True)
x = FFTop.H*X

# sampling locations
perc_subsampling = 0.2
ntsub = int(np.round(nt*perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(nt))[:ntsub])

# create operator
Rop = pylops.Restriction(nt, iava, dtype='float64')

# apply forward
y = Rop*x
ymask = Rop.mask(x)

# apply adjoint
xadj = Rop.H*y

# apply inverse
xinv = Rop / y

# regularized inversion
D2op = pylops.SecondDerivative(nt, dims=None, dtype='float64')

epsR = np.sqrt(0.1)
epsI = np.sqrt(1e-4)

xne = \
    pylops.optimization.leastsquares.NormalEquationsInversion(Rop, [D2op], y,
                                                              epsI=epsI,
                                                              epsRs=[epsR],
                                                              returninfo=False,
                                                              **dict(maxiter=50))

# sparse inversion
pfista, niterf, costf = \
    pylops.optimization.sparsity.FISTA(Rop*FFTop.H, y, niter=1000,
                                       eps=0.001, tol=1e-7, returninfo=True)
xfista = FFTop.H*pfista