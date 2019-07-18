def RegularizedInversion(G, Reg, d, dreg, epsR, ...):

    ...
    # operator
    Gtot = VStack([G, epsR * Reg], dtype=G.dtype)
    #data
    dtot = np.hstack((d, epsR*dreg))
    ...
    # solver
    minv = lsqr(Gtot, dtot, ...)[0]
