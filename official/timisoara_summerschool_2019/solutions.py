# Solutions for DataAssimilation summer school

class Diagonal(LinearOperator):
    def __init__(self, diag, dtype='float64'):
        self.diag = diag.flatten()
        self.shape = (len(self.diag), len(self.diag))
        self.dtype = np.dtype(dtype)

    def _matvec(self, x):
        y = self.diag*x
        return y

    def _rmatvec(self, x):
        y = self.diag*x
        return y
        

class FirstDerivative(LinearOperator):
    def __init__(self, N, sampling=1., dtype='float64'):
        self.N = N
        self.sampling = sampling
        self.shape = (N, N)
        self.dtype = dtype
        self.explicit = False
        
    def _matvec(self, x):
        x, y = x.squeeze(), np.zeros(self.N, self.dtype)
        y[1:-1] = (0.5 * x[2:] - 0.5 * x[0:-2]) / self.sampling
        # edges
        y[0] = (x[1] - x[0]) / self.sampling
        y[-1] = (x[-1] - x[-2]) / self.sampling
        return y
    
    def _rmatvec(self, x):
        x, y = x.squeeze(), np.zeros(self.N, self.dtype)
        y[0:-2] -= (0.5 * x[1:-1]) / self.sampling
        y[2:] += (0.5 * x[1:-1]) / self.sampling
        # edges
        y[0] -= x[0] / self.sampling
        y[1] += x[0] / self.sampling
        y[-2] -= x[-1] / self.sampling
        y[-1] += x[-1] / self.sampling
        return y