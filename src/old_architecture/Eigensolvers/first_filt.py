import numpy as np
from scipy.linalg import qr, eigh
from .lancz_uppbnd import lancz_uppbnd
from .Rayleighritz import Rayleighritz

def first_filt(nev, H, polm=10):
    """
    Input:
      nev   -- number of ritz values to compute (>= # of occupied states)
      H     -- The hamiltonian matrix
      polm  -- polynomial degree  (can be optional)
 
    Output:
      ritzv -- contains the new ritz values
      W     -- the ortho-normal basis of a approximate subspace
    """

    OPTIMIZATIONLEVEL = 0
    
    DEBUG = 0
    Energ_tol = 0.08
    if polm < 10 or polm > 15:
        polm = 10
    max_iter = max(min(int(60 / polm), 5), 3)
    n = H.shape[0]

    # call upper bound estimator to compute an upper bound
    upperb, ritzv = lancz_uppbnd(n, H)
    lowerb = 0.8 * ritzv[0] + 0.2 * ritzv[-2]

    if DEBUG == 1:
        print(f'polm={polm}, nev={nev}, sizeH={n}, max_iter={max_iter}')
        print(f'upperb={upperb},  lowerb={lowerb}')

    # filter random vectors  (the size of W here saves at least half the memory
    # than lanczos or other diagonalization methods)
    W = np.random.rand(n, nev)

    for it in range(1, max_iter + 1):

        W = cheb_filter_slim(H, W, polm, lowerb, upperb)

        W, G = qr(W, mode='economic', pivoting=False)
        Vin = H @ W   # the temp Vin can be further reduced to save memory, not done yet.
        n, n2 = Vin.shape
        if n2 != nev:
            raise ValueError('wrong number of eigenvalues')
        if OPTIMIZATIONLEVEL != 0:
            print('Using Rayleighritz')
            G = Rayleighritz(Vin, W, n2)
        else:
            # reuse G and overwrite it
            G = np.zeros((n2, n2))
            for j in range(1, n2 + 1):
                for i in range(1, j + 1):
                    G[i - 1, j - 1] = np.vdot(Vin[:, i - 1], W[:, j - 1])
                    G[j - 1, i - 1] = G[i - 1, j - 1]

        ritzv_raw, Q = eigh(G)
        order = np.argsort(ritzv_raw)
        ritzv = ritzv_raw[order]
        lowerb = ritzv[-1]
        W = W @ Q[:, order]

        if it == 1:
            tr0 = np.sum(ritzv[:nev])
        else:
            tr1 = np.sum(ritzv[:nev])
            if abs(tr1 - tr0) <= Energ_tol * abs(tr0):
                break
            if DEBUG == 1 and it > 1:
                print(f'iter={it}, lowerb = {lowerb}, ritz_diff={abs(tr1 - tr0)}')
            tr0 = tr1

    return W, ritzv


def cheb_filter_slim(H, x, polm, low, high):
    """
    Parameters:
    H     -- The Hamiltonian matrix.
    x     -- Input matrix.
    polm  -- Polynomial degree.
    low   -- Lower bound.
    high  -- Upper bound.

    Returns:
    y     -- The filtered matrix.
    Slim Chebyshev iteration (non-scaling, normalized).
    """
    e = (high - low) / 2
    center = (high + low) / 2

    y = H @ x
    y = (-y + center * x) / e

    for i in range(2, polm + 1):
        ynew = H @ y
        ynew = (-ynew + center * y) * 2 / e - x
        x = y
        y = ynew

    return y
