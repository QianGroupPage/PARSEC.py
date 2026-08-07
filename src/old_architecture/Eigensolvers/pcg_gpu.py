import cupy as cp
from cupy._core import set_reduction_accelerators, set_routine_accelerators
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as sps


set_reduction_accelerators([])
set_routine_accelerators([])
warned_preconditioner = False


def _to_gpu_matrix(A):
    if isinstance(A, cp.ndarray):
        return A.astype(cp.float32, copy=False)
    if isinstance(A, cpsparse.spmatrix):
        if A.dtype == cp.float32:
            return A
        return A.astype(cp.float32)
    if sps.issparse(A):
        return cpsparse.csr_matrix(A.astype("float32"))
    return cp.asarray(A, dtype=cp.float32)


def pcg(A, rhs, x0, m, tol, *args):
    """
    GPU-accelerated Conjugate Gradient solver using CuPy.

    Parameters:
    A   -- SciPy/CuPy matrix (n x n)
    rhs -- CuPy or NumPy ndarray (n x 1)
    x0  -- Initial guess (n x 1)
    m   -- Max iterations
    tol -- Convergence tolerance

    Returns:
    x   -- Solution vector on the GPU
    its -- Number of iterations
    """
    global warned_preconditioner
    if len(args) >= 2:
        if not warned_preconditioner:
            print("Warning: GPU PCG currently ignores the CPU-side preconditioner arguments.")
            warned_preconditioner = True

    A = _to_gpu_matrix(A)
    rhs = cp.asarray(rhs, dtype=cp.float32).ravel()
    x = cp.asarray(x0, dtype=cp.float32).ravel()
    # Flatten vectors for faster inner products on GPU.
    r = rhs - A @ x
    p = r.copy()

    ro1 = cp.inner(r, r)
    tol1 = (tol ** 2) * ro1

    its = 0
    while its < m and ro1 > tol1:
        its += 1
        Ap = A @ p
        alpha = ro1 / cp.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        ro_new = cp.inner(r, r)
        beta = ro_new / ro1
        p = r + beta * p
        ro1 = ro_new

    return x, its
