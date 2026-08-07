import numpy as np
import time
from scipy.linalg import qr, eigh
from .Rayleighritz import Rayleighritz
from .lanczosForChefsi1 import lanczosForChefsi1
from .ch_filter import ch_filter

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def chefsi1(Vin, ritzv, deg, nev, H):
    """
    Input:
        Vin   -- basis vector
        ritzv -- vector containing previous retz values
        deg   -- polynomial degree
        nev   -- number of occupied states
        H     -- The hamiltonian matrix
    
    Output:
        ritzv -- contains the new ritz values
        W     -- the filtered vectors
    """
    n, n2 = Vin.shape
    G = np.zeros((n2, n2))
    if nev >= n2:
        raise ValueError('Vin should have more than nev columns')

    # call Lanczos with nev = 6 and m=6 to get the upper interval bound
    upperb = lanczosForChefsi1(H, 6, np.random.randn(n, 1), 6, 0.0)

    # Use max of previous eigenvalues for lower bound
    lowerb = np.max(ritzv)

    # Use the smallest eigenvalue estimate for scaling
    lam1 = np.min(ritzv)

    if lowerb > upperb:
        raise ValueError('Bounds are incorrect')

    W = ch_filter(H, Vin, deg, lam1, lowerb, upperb)

    # orthonormalize the basis, should be replaced with better method for real computations
    W, R = qr(W, mode='economic', pivoting=False)

    # Rayleigh-Ritz projection: compute Hhat = W^T @ H @ W
    Vin = H @ W
    if OPTIMIZATIONLEVEL != 0:
        G = Rayleighritz(Vin, W, n2)
    else:
        loop_start = time.perf_counter() if enableMexFilesTest == 1 else None
        for j in range(1, n2 + 1):
            for i in range(1, j + 1):
                G[i - 1, j - 1] = np.vdot(Vin[:, i - 1], W[:, j - 1])
                G[j - 1, i - 1] = G[i - 1, j - 1]
        if enableMexFilesTest == 1:
            loop_time = time.perf_counter() - loop_start
            mex_start = time.perf_counter()
            G2 = Rayleighritz(Vin, W, n2)
            mex_time = time.perf_counter() - mex_start
            max_diff = np.max(np.abs(G - G2))
            print(f"Rayleighritz C++ time: {mex_time:.6f}s, Python time: {loop_time:.6f}s, max diff: {max_diff:.3e}")

    # Diagonalize
    ritzv_raw, Q = eigh(G)
    order = np.argsort(ritzv_raw)
    ritzv = ritzv_raw[order]
    W = W @ Q[:, order]

    return W, ritzv
