def PreconditionedInversion(G, P, d, ...):

    ...
    # operator
    Gtot = G * P
    ...
    # solver
    minv = lsqr(Gtot, d, ...)[0]
