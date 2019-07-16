# forward
det _matvec(x)
    return self.diag * x

# adjoint
det _matvec(x)
    return self.diag * x
