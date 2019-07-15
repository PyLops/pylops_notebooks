det _matvec(x):
    x = x.squeeze()
    y = np.zeros(self.N, self.dtype)
    y[1:-1] = (0.5 * x[2:] - 0.5 * x[0:-2]) / self.sampling
    if self.edge:
        y[0] = (x[1] - x[0]) / self.sampling
        y[-1] = (x[-1] - x[-2]) / self.sampling
