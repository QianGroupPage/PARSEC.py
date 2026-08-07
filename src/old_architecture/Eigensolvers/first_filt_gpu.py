import cupy as cp
from cupy.linalg import qr, eigh

try:
    from .gpu_linear_operator import to_gpu_matrix
    from .lancz_uppbnd_gpu import lancz_uppbnd
    from .Rayleighritz_gpu import Rayleighritz
except ImportError:
    from gpu_linear_operator import to_gpu_matrix
    from lancz_uppbnd_gpu import lancz_uppbnd
    from Rayleighritz_gpu import Rayleighritz


def _to_gpu_matrix(H):
    """
    Ensure H is a GPU matrix, sparse matrix, or custom GPU linear operator.
    """
    return to_gpu_matrix(H, return_mode=True)


def first_filt(nev, H, polm=10):
    """
    GPU-first eigensolver with Chebyshev filtering + Rayleigh-Ritz projection.
    Automatically picks dense or sparse GPU path depending on H.
    """
    global tr0
    OPTIMIZATIONLEVEL = 0
    DEBUG = False
    Energ_tol = 0.08

    # sanitize polynomial degree
    if polm < 10 or polm > 15:
        polm = 10
    max_iter = max(min(int(60 / polm), 5), 3)

    # convert H to GPU object
    H, mode = _to_gpu_matrix(H)
    n = H.shape[0]

    # estimate spectral bounds
    upperb, ritzv = lancz_uppbnd(n, H)
    lowerb = 0.8 * ritzv[0] + 0.2 * ritzv[-2]

    if DEBUG:
        print(f"mode={mode}, polm={polm}, nev={nev}, sizeH={n}, max_iter={max_iter}")
        print(f"upperb={upperb}, lowerb={lowerb}")

    # random initial subspace
    W = cp.random.randn(n, nev, dtype=cp.float32)

    # filtering loop
    for it in range(max_iter):
        W = cheb_filter_slim(H, W, polm, lowerb, upperb)

        # QR orthogonalization (GPU)
        W, _ = qr(W, mode="reduced")

        Vin = H @ W
        if Vin.shape[1] != nev:
            raise ValueError("Wrong number of eigenvalues")

        # Rayleigh-Ritz projection
        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, nev)
        else:
            G = W.T @ Vin
            G = (G + G.T) / 2  # enforce symmetry

        # solve reduced eigenproblem
        ritzv, Q = eigh(G)
        lowerb = ritzv[-1]
        W = W @ Q

        # check convergence
        tr = cp.sum(ritzv[:nev])
        if it > 0 and abs(tr - tr0) <= Energ_tol * abs(tr0):
            break
        tr0 = tr

    return W, ritzv


def cheb_filter_slim(H, X, polm, low, high):
    """
    Chebyshev filter using recurrence, works with both dense and sparse CuPy matrices.
    """
    e = (high - low) / 2
    center = (high + low) / 2

    Y = H @ X
    Y = (-Y + center * X) / e

    for _ in range(2, polm + 1):
        Ynew = H @ Y
        Ynew = (-Ynew + center * Y) * 2 / e - X
        X, Y = Y, Ynew

    return Y
