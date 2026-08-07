import numpy as np
from scipy.linalg import qr, eigh
from .Rayleighritz import Rayleighritz

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def lanczosForChsubsp(B, nev, v, m):
    """
    Perform the Lanczos algorithm for eigenvalue computation.

    Args:
        B (np.ndarray): Input matrix.
        nev (int): Number of desired eigenvalues.
        v (np.ndarray): Initial vector.
        m (int): Number of Lanczos steps.

    Returns:
        W (np.ndarray): Matrix of approximate eigenvectors.
        lam (np.ndarray): Approximate eigenvalues.
        bound (float): An upper bound for the eigenvalues.
    """

    reorth = 0
    tol = 1e-5

    # Enable or disable convergence testing based on matrix size
    if B.shape[0] > 35000:
        enableConvergenceTest = 0
    else:
        enableConvergenceTest = 1

    # Matrix dimensions and normalization of the initial vector
    n = B.shape[0]
    v = np.asarray(v).reshape(-1)
    v = v / np.linalg.norm(v)
    v1 = v
    v0 = np.zeros_like(v1)
    # print(f"lanczosForChsubsp: start n={n} nev={nev} m={m}", flush=True)
    k = 0
    bet = 0
    ll = 0

    # Pre-allocating arrays
    Tmat = np.zeros((m + 1, m + 1))
    tr = np.zeros(m)
    rr = None
    X1 = None
    indx = None
    bound = None

    if OPTIMIZATIONLEVEL != 0:
        ValueError("not implemented yet!")
        # # Using a custom memory allocation (placeholder for actual implementation)
        # rr, X1, indx, k, bound = lanWhileLoopForChsubsp(m, v0, v1, B, reorth, nev, VV, Tmat)
    else:
        # print(f"lanczosForChsubsp: allocating VV shape=({n}, {m})", flush=True)
        VV = np.zeros((n, m))
        # print("lanczosForChsubsp: VV allocated", flush=True)
        while k < m:
            k += 1
            VV[:, k - 1] = v1

            v = B @ v1 - bet * v0
            # if enableMexFilesTest == 1:
            #     v2 = BTimesv1MinusbetTimesv0(B, v0, v1, bet, findFirstColumnWithNonZeroElement(B))
            #     if np.any(np.abs(v - v2) > 1e-6):
            #         ValueError("Mex file discrepancy for BTimesv1MinusbetTimesv0")

            # Calculate alpha
            alp = np.vdot(v1, v)

            if enableMexFilesTest == 0:
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
            bet = np.linalg.norm(v)
            v0 = v1
            inverseOfbet = 1.0 / bet
            v1 = v * inverseOfbet

            # Update Tmat
            Tmat[k - 1, k - 1] = alp
            Tmat[k, k - 1] = bet
            Tmat[k - 1, k] = bet

            NTest = min(8 * nev, m)

            if enableConvergenceTest and (((k >= NTest) and (k % 10 == 0)) or k == m):
                if k != m:
                    rr = np.linalg.eigvalsh(Tmat[:k, :k])
                    indx = np.argsort(rr)
                    rr = rr[indx]
                else:
                    rr, X1 = eigh(Tmat[:k, :k])
                    indx = np.argsort(rr)
                    rr = rr[indx]

                bound = np.max(np.abs(rr)) + bet
                tr1 = np.sum(rr[:nev])
                ll += 1
                tr[ll - 1] = tr1

            # Convergence criterion
            if enableConvergenceTest and (ll > 1 and (np.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2])):
                rr, X1 = eigh(Tmat[:k, :k])
                indx = np.argsort(rr)
                rr = rr[indx]
                break

        if enableConvergenceTest == 0:
            rr, X1 = eigh(Tmat[:k, :k])
            indx = np.argsort(rr)
            rr = rr[indx]
            bound = np.max(np.abs(rr)) + bet

    if X1 is None:
        rr, X1 = eigh(Tmat[:k, :k])
        indx = np.argsort(rr)
        rr = rr[indx]
    if bound is None or not np.isfinite(bound) or np.iscomplexobj(bound):
        bound = np.max(np.abs(rr)) + bet

    del Tmat
    del v
    del v0
    del v1

    # Save eigenvalues and compute eigenvectors
    if reorth:
        lam = rr[:nev]
        Y = X1[:, indx[:nev]]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
    else:
        rrCopy = rr.copy()

        # Find the number of unique eigenvalues
        counter = 0
        while counter < len(rrCopy) - 1:
            rrCopy = np.delete(rrCopy, np.where(np.abs(rrCopy[counter] - rrCopy[counter+1:]) < 1e-5)[0] + counter + 1)
            counter += 1

        mevMultiplier = min(1.8 ** (len(rr) / len(rrCopy)), 3.2)
        mev = int(min(nev * mevMultiplier, k))

        Y = X1[:, indx[:mev]]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
        del VV  # Clean up memory

        # Orthogonalize W using QR decomposition
        W, G = qr(W, mode='economic',pivoting=False)

        # Calculate the Rayleigh-Ritz matrix G
        Vin = B @ W
        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, mev)
        else:
            G = np.zeros((mev, mev))
            for j in range(mev):
                for i in range(j + 1):
                    G[i, j] = np.vdot(Vin[:, i], W[:, j])
                    G[j, i] = G[i, j]

            if enableMexFilesTest == 1:
                G2 = Rayleighritz(Vin, W, mev)
                if np.any(abs(G - G2) > 1e-6):
                    ValueError("Mex file discrepancy for Rayleighritz.c")

        rr, X1 = eigh(G)
        indx = np.argsort(rr)
        rr = rr[indx]
        lam = rr[:nev]
        Y = X1[:, indx[:nev]]
        W = W @ Y

    # print("lanczosForChsubsp: done", flush=True)
    return W, lam, bound
