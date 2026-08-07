import numpy as np
from scipy.linalg import qr, eigh
from .ch_filter import ch_filter
from .Rayleighritz import Rayleighritz
from .lanczosForChsubsp import lanczosForChsubsp
from ..updatePercentageComplete import updatePercentageComplete

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def lanczos(B, nev, v, m, tol, *args):
    """
        B   = matrix;
        nev = number of wanted eigenvalues;
        v   = initial vector
        m   = number of steps
        tol = tolerance for stopping
    can do with (full) reorthogonalization (reorth=1)
    or no -- reorthogonalization (reorth=0)
    this version of lanczos is called by RunRDFT and does not calculate
    bound
    """

    # Handle optional arguments for GUI
    if len(args) == 3:
        percentageComplete, handle, barHandle = args
    else:
        percentageComplete = handle = barHandle = None

    reorth = 0
    usePolynomialFiltering = 0
    # testing has shown that most of the time when B is large, the sum of the eigenvalues does not converge, so
    # testing for convergence just wastes cpu time
    if B.shape[0] > 35000:
        enableConvergenceTest = 0
    else:
        enableConvergenceTest = 1

    n = B.shape[0]  # Dimension of the matrix
    v = np.asarray(v, dtype=float).ravel()
    v = v / np.linalg.norm(v)  # Normalize the initial vector
    v1 = v
    v0 = np.zeros(n, dtype=float)
    k = 0
    bet = 0
    ll = 0

    if usePolynomialFiltering == 1:
        Lanc_steps = max(2 * nev, 30)
        W, ritzv, upperb = lanczosForChsubsp(B, nev, v, Lanc_steps)
        lam1 = min(ritzv)
        lowerb = ritzv[min(nev, len(ritzv) - 1)]

        v = ch_filter(B, v, 500, lam1, lowerb, upperb)

        v1 = v / np.linalg.norm(v)

    # Pre-allocate Tmat for tridiagonal matrix storage
    Tmat = np.zeros((m + 1, m + 1))
    tr = []  # Track convergence values

    # TODO: c code
    # if OPTIMIZATIONLEVEL!=0:
    #     # Pre-allocate VV array
    #     VV = np.zeros((n, m))
    #
    #     # Use GUI-related inputs if all 8 arguments are provided
    #     if len(args) == 3:
    #         rr, X1, indx, k = lanWhileLoop(m, v0, v1, B, reorth, nev, VV, Tmat, tr, tol, enableConvergenceTest,
    #                                        percentageComplete, handle, barHandle)
    #     else:
    #         # Text-based run with no progress bar
    #         rr, X1, indx, k = lanWhileLoop(m, v0, v1, B, reorth, nev, VV, Tmat, tr, tol, enableConvergenceTest)
    #
    # else:
        # Allocate VV array for storing basis vectors
    VV = np.zeros((n, m))
    rr = []
    #indx = []
    X1 = []
    while k < m:
        k += 1
        VV[:, k - 1] = v1

        # Perform matrix-vector multiplication and Lanczos step
        v = B @ v1 - bet * v0

        # TODO: compare c and m language
        # if enableMexFilesTest == 1:
        #     v2 = BTimesv1MinusbetTimesv0(B, v0, v1, bet, findFirstColumnWithNonZeroElement(B))
        #     if np.any(np.abs(v - v2) > 1e-6):
        #         ValueError("Mex file discrepancy for BTimesv1MinusbetTimesv0")

        # Compute alpha and perform reorthogonalization if needed
        alp = np.dot(v1, v)

        # if enableMexFilesTest == 0:
        v -= alp * v1
        # TODO: c code
        # else:
        #     vTemp = v - alp * v1;
        #     vTemp2 = vMinusalpTimesv1(v, v1, alp)
        #     v = vTemp
        #     if np.any(np.abs(vTemp - vTemp2) > 1e-6):
        #          ValueError("Mex file discrepancy for BTimesv1MinusbetTimesv0")

        '''
            reorth  -- test for reorth. needed!
        '''
        if reorth:
            subVV = VV[:, :k]
            v -= subVV @ (subVV.T @ v)

        # normalize and store v1
        bet = np.linalg.norm(v)
        v0 = v1
        inverseOfbet = 1.0 / bet
        v1 = v * inverseOfbet

        # Update Tmat (the tridiagonal matrix)
        Tmat[k - 1, k - 1] = alp
        Tmat[k, k - 1] = bet
        Tmat[k - 1, k] = bet
        NTest = min(8 * nev, m) # when to start testing

        # tr, ll, == for plotting
        # Tracking convergence of eigenvalues
        if enableConvergenceTest and (((k >= NTest) and (k % 10 == 0)) or k == m):
            if k != m:
                rr = eigh(Tmat[:k, :k])
            else:
                rr, X1 = eigh(Tmat[:k, :k])
            tr1 = np.sum(rr[:nev])
            ll += 1
            tr[ll - 1] = tr1

        # Stopping criterion based on eigenvalue sum convergence
        if enableConvergenceTest and (ll > 1 and (np.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2])):
            rr, X1 = eigh(Tmat[:k, :k])
            break

        # Update progress bar for GUI
        if handle is not None and k % 25 == 0:
            updatePercentageComplete(percentageComplete + 0.1 * k / m, barHandle, handle)

    if enableConvergenceTest == 0:
        rr, X1 = eigh(Tmat[:k, :k])

    # Print the current state
    print(f"in lanczos.py: nev={nev}, k={k}, m={m}")

    del Tmat
    del v
    del v0
    del v1

    if reorth:
        lam = rr[:nev]
        Y = X1[:, :nev]
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            W = VV[:, :k] @ Y
    else:
        # Copy rr to find unique eigenvalues
        rrCopy = rr.copy()

        # find out how many eigenvalues are copies
        counter = 0

        # Remove near-duplicate eigenvalues
        rrCopy = np.array(rrCopy)
        while counter < len(rrCopy) - 1:
            rrCopy = np.delete(rrCopy, np.where(np.abs(rrCopy[counter] - rrCopy[counter+1:]) < 1e-5)[0] + counter + 1)
            counter += 1

        # Determine multiplicative factor for eigenvalues to target
        mevMultiplier = min(1.8 ** (len(rr) / len(rrCopy)), 3.2)
        # if no reorth.to get meaningful eigenvectors.. get twice as many eigenvectors and use RayleighRitz
        mev = int(min(nev * mevMultiplier, k))

        # Extract top `mev` eigenvectors
        Y = X1[:, :mev]

        # Construct W matrix with VV and Y
        if k == VV.shape[1]:
            W = VV @ Y
        else:
            VV = VV[:, :k]
            W = VV @ Y

        # Clear VV to free memory
        del VV

        # Orthogonalize W using QR decomposition
        W, G = qr(W, mode='economic',pivoting=False)

        # Calculate the Rayleigh quotient matrix
        Vin = B @ W
        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, mev)
        else:
            # Construct the Rayleigh quotient matrix manually
            G = np.zeros((mev, mev))
            for j in range(mev):
                for i in range(j + 1):
                    G[i, j] = np.dot(Vin[:, i], W[:, j])
                    G[j, i] = G[i, j]  # G is symmetric

            # Test against compiled implementation if enabled
            if enableMexFilesTest == 1:
                G2 = Rayleighritz(Vin, W, mev)
                if np.any(abs(G - G2) > 1e-6):
                    ValueError("Mex file discrepancy for Rayleighritz.c")

        # Compute eigenvalues and eigenvectors of G
        rr, X1 = eigh(G)

        # Select the top `nev` eigenvalues and corresponding eigenvectors
        lam = rr[:nev]
        Y = X1[:, :nev]
        W = W @ Y

    return W, lam
