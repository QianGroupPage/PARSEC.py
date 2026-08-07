import cupy as cp
from cupy.linalg import norm, eigh

try:
    from .gpu_linear_operator import to_gpu_matrix
except ImportError:
    from gpu_linear_operator import to_gpu_matrix

enableMexFilesTest = 0


def _to_gpu_matrix(B):
    """Convert B to dense or sparse CuPy matrix."""
    return to_gpu_matrix(B, return_mode=True)


def lanczosForChefsi1(B, nev, v, m, tol=1e-6, reorth=False):
    """
    GPU version of the Lanczos iteration used in CheFSI.
    Estimates an upper bound of eigenvalues.
    """
    B, mode = _to_gpu_matrix(B)

    n = B.shape[0]
    v = cp.asarray(v, dtype=cp.float32)
    v = v / norm(v)

    v1 = v
    v0 = cp.zeros_like(v)
    k = 0
    bet = 0.0
    ll = 0

    if reorth:
        VV = cp.zeros((n, m), dtype=cp.float32)
    else:
        VV = None

    Tmat = cp.zeros((m + 1, m + 1), dtype=cp.float32)
    tr = cp.zeros(m, dtype=cp.float32)
    bound = 0.0

    while k < m:
        k += 1
        if reorth:
            VV[:, k - 1] = v1[:, 0]

        # Lanczos step
        v = B @ v1 - bet * v0
        alp = cp.dot(v1.T, v).item()
        v -= alp * v1

        if reorth:
            t = VV[:, :k].T @ v
            v -= VV[:, :k] @ t

        bet = norm(v)
        if bet < tol:
            break

        v0 = v1
        v1 = v / bet

        # update tridiagonal T matrix
        Tmat[k - 1, k - 1] = alp
        Tmat[k, k - 1] = bet
        Tmat[k - 1, k] = bet

        # periodic convergence check
        NTest = min(5 * nev, m)
        if (k >= NTest and k % 10 == 0) or k == m:
            rr, _ = eigh(Tmat[:k, :k])
            bound = cp.max(cp.abs(rr)) + bet
            tr1 = cp.sum(rr[:nev])
            ll += 1
            tr[ll - 1] = tr1

            if ll > 1 and cp.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2]:
                break

    return bound
