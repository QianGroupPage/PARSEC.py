import cupy as cp
from cupy.linalg import qr, eigh

try:
    from .Rayleighritz_gpu import Rayleighritz
    from .ch_filter import ch_filter
    from .gpu_linear_operator import to_gpu_matrix
    from .lanczosForChefsi1_gpu import lanczosForChefsi1
except ImportError:
    from Rayleighritz_gpu import Rayleighritz
    from ch_filter import ch_filter
    from gpu_linear_operator import to_gpu_matrix
    from lanczosForChefsi1_gpu import lanczosForChefsi1


OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0


def _to_gpu_matrix(H):
    """Convert H to a CuPy matrix, sparse matrix, or custom GPU operator."""
    return to_gpu_matrix(H, return_mode=True)


def chefsi1(Vin, ritzv, deg, nev, H):
    """
    GPU-accelerated CheFSI (Chebyshev Filtered Subspace Iteration)
    using CuPy + cuBLAS/cuSPARSE.
    """
    n, n2 = Vin.shape
    if nev >= n2:
        raise ValueError("Vin should have more than nev columns")

    H, mode = _to_gpu_matrix(H)

    # --- Compute spectral bounds ---
    upperb = lanczosForChefsi1(H, 6, cp.random.randn(n, 1, dtype=cp.float32), 6, 0.0)
    lowerb = cp.max(ritzv)
    lam1 = cp.min(ritzv)
    if lowerb > upperb:
        raise ValueError("Invalid spectral bounds (lower > upper)")

    # --- Apply Chebyshev filter (should use GPU ch_filter) ---
    W = ch_filter(H, Vin, deg, lam1, lowerb, upperb)

    # --- QR orthonormalization (GPU) ---
    W, _ = qr(W, mode="reduced")

    # --- Rayleigh-Ritz projection ---
    Vin_proj = H @ W  # cuBLAS/cuSPARSE or custom operator matvec
    if OPTIMIZATIONLEVEL != 0:
        G = Rayleighritz(Vin_proj, W, n2)
    else:
        G = W.T @ Vin_proj
        G = (G + G.T) / 2  # enforce symmetry

    # --- Small eigen-decomposition (GPU) ---
    ritzv, Q = eigh(G)
    W = W @ Q

    return W, ritzv
