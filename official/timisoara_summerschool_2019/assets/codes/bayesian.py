def BayesianInversion(G, d, Cm, Cd ...):

    ...
    # operator
    Gbayes = G * Cm * G.H + Cd
    # data
    dbayes = d - G * m0
    ...
    # solver
    minv = m0 + Cm * G.H * lsqr(Gbayes, dbayes, ...)[0]
