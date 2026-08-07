import cupy as cp


def qinv(A, tol=None):
    """
    GPU quasi-inverse based on the Moore-Penrose pseudoinverse.

    The CPU path uses QR with column pivoting. CuPy does not expose an exact
    drop-in equivalent, so this uses SVD-based pseudoinversion instead.
    """
    A_gpu = cp.asarray(A)
    if A_gpu.ndim != 2:
        raise ValueError("qinv expects a 2D matrix")

    m, n = A_gpu.shape
    if m < n:
        return qinv(A_gpu.T, tol).T

    if tol is None:
        tol = cp.finfo(A_gpu.dtype).eps

    scale = cp.linalg.norm(A_gpu, cp.inf)
    rcond = float((tol * scale).item()) if scale.ndim == 0 else float(tol)
    return cp.linalg.pinv(A_gpu, rcond=rcond)
