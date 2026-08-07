import cupy as cp
from cupy.linalg import qr, eigh

try:
    from .gpu_linear_operator import to_gpu_matrix
    from .Rayleighritz_gpu import Rayleighritz
except ImportError:
    from gpu_linear_operator import to_gpu_matrix
    from Rayleighritz_gpu import Rayleighritz

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0


def _to_gpu_matrix(B):
    return to_gpu_matrix(B)


def lanczosForChsubsp(B, nev, v, m):
    """
    Perform the Lanczos algorithm for eigenvalue computation.

    Args:
        B (cp.ndarray): Icput matrix.
        nev (int): Number of desired eigenvalues.
        v (cp.ndarray): Initial vector.
        m (int): Number of Lanczos steps.

    Returns:
        W (cp.ndarray): Matrix of approximate eigenvectors.
        lam (cp.ndarray): Approximate eigenvalues.
        bound (float): An upper bound for the eigenvalues.
    """

    reorth = 0
    tol = 1e-5

    # Enable or disable convergence testing based on matrix size
    enableConvergenceTest = 0 if B.shape[0] > 35000 else 1
    B = _to_gpu_matrix(B)
    v = cp.asarray(v, dtype=cp.float32).ravel()
    # Matrix dimensions and normalization of the initial vector
    n = B.shape[0]
    v = v / cp.linalg.norm(v)
    v1 = v
    v0 = cp.zeros(n, dtype=cp.float32)
    #ZEROES ARE A GPU KILLER ALWAYS INIT FLOAT32
    k = 0
    bet = 0
    ll = 0

    # Pre-allocating arrays
    Tmat = cp.zeros((m + 1, m + 1), dtype=cp.float32)
    tr = []
    rr = []
    X1 = []
    #indx = []
    bound = 0.0
    VV = cp.zeros((n, m), dtype=cp.float32)

    while k < m:
        k += 1
        VV[:, k - 1] = v1

        v = B @ v1 - bet * v0
            # if enableMexFilesTest == 1:
            #     v2 = BTimesv1MinusbetTimesv0(B, v0, v1, bet, findFirstColumnWithNonZeroElement(B))
            #     if np.any(np.abs(v - v2) > 1e-6):
            #         ValueError("Mex file discrepancy for BTimesv1MinusbetTimesv0")

            # Calculate alpha
        alp = cp.dot(v1, v)

        v = v - alp * v1
            # else:
            #     vTemp = v - alp * v1
            #     vTemp2 = vMinusalpTimesv1(v, v1, alp)
            #     v = vTemp
            #     if np.any(np.abs(vTemp - vTemp2) > 1e-6):
            #         ValueError("Mex file discrepancy for vMinusalpTimesv1")

            # Reorthogonalization, if needed
        if reorth:
            subVV = VV[:, :k]
            v = v - subVV @ (subVV.T @ v)

            # Normalize the vector and update variables
        bet = cp.linalg.norm(v)
        v0 = v1
        v1 = v / bet

            # Update Tmat
        Tmat[k - 1, k - 1] = alp
        Tmat[k, k - 1] = bet
        Tmat[k - 1, k] = bet

        NTest = min(8 * nev, m)

        if enableConvergenceTest and (((k >= NTest) and (k % 10 == 0)) or k == m):
            rr, X1 = eigh(Tmat[:k, :k])
            bound = cp.max(cp.abs(rr)) + bet
            tr1 = cp.sum(rr[:nev])
            tr.append(tr1)
            ll += 1
            # Convergence criterion
        if enableConvergenceTest and (ll > 1 and (cp.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2])):
            rr, X1 = cp.linalg.eigh(Tmat[:k, :k])
            break

    rr, X1 = cp.linalg.eigh(Tmat[:k, :k])

    del Tmat
    del v
    del v0
    del v1

    # Save eigenvalues and compute eigenvectors
    if reorth:
        lam = rr[:nev]
        Y = X1[:, :nev]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
    else:
        rrCopy = rr.copy()

        # Find the number of unique eigenvalues
        counter = 0
        while counter < len(rrCopy) - 1:
            rrCopy = cp.delete(rrCopy, cp.where(cp.abs(rrCopy[counter] - rrCopy[counter+1:]) < 1e-5)[0] + counter + 1)
            counter += 1

        mevMultiplier = min(1.8 ** (len(rr) / len(rrCopy)), 3.2)
        mev = int(min(nev * mevMultiplier, k))

        Y = X1[:, :mev]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
        del VV  # Clean up memory

        # Orthogonalize W using QR decomposition
        W, _ = qr(W, mode='reduced')

        # Calculate the Rayleigh-Ritz matrix G
        Vin = B @ W
        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, mev)
        else:
            G = W.T @ Vin
            G = (G + G.T) * 0.5

        rr, X1 = cp.linalg.eigh(G)
        lam = rr[:nev]
        W = W @ X1[:, :nev]

    return W, lam, bound
