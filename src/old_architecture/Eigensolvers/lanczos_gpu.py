import cupy as cp
from cupy.linalg import qr, eigh

try:
    from .ch_filter import ch_filter
    from .gpu_linear_operator import to_gpu_matrix
    from .Rayleighritz_gpu import Rayleighritz
    from .lanczosForChsubsp_gpu import lanczosForChsubsp
except ImportError:
    from ch_filter import ch_filter
    from gpu_linear_operator import to_gpu_matrix
    from Rayleighritz_gpu import Rayleighritz
    from lanczosForChsubsp_gpu import lanczosForChsubsp
from ..updatePercentageComplete import updatePercentageComplete

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0


def _to_gpu_matrix(B):
    return to_gpu_matrix(B)


def lanczos(B, nev, v, m, tol, *args):
    """
    Parameters:
    B (ndarray): The Hamiltonian matrix (or a matrix-like object).
    nev (int): Number of desired eigenvalues.
    v (ndarray): Initial random vector for Lanczos.
    m (int): Number of Lanczos steps.
    tol (float): Convergence tolerance.
    can do with (full) reorthogonalization (reorth=1)
    or no -- reorthogonalization (reorth=0)

    enable_convergence_test (bool): Flag to enable or disable convergence testing.
    percentage_complete (float): Used for graphical progress bars (optional).
    handle, bar_handle: GUI handles for updating the progress bar (optional).

    Returns:
    W (ndarray): Eigenvector matrix of size (n, nev).
    lam (ndarray): Array of the computed eigenvalues.
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

    B = _to_gpu_matrix(B)
    n = B.shape[0]  # Dimension of the matrix
    v = cp.asarray(v, dtype=cp.float32).ravel()
    v = v / cp.linalg.norm(v)  # Normalize the initial vector
    v1 = v
    v0 = cp.zeros(n, dtype=cp.float32)
    k = 0
    bet = cp.float32(0.0)
    ll = 0

    if usePolynomialFiltering == 1:
        Lanc_steps = max(2 * nev, 30)
        W, ritzv, upperb = lanczosForChsubsp(B, nev, v, Lanc_steps)
        lam1 = min(ritzv)
        lowerb = ritzv[min(nev, len(ritzv) - 1)]

        v = ch_filter(B, v, 500, lam1, lowerb, upperb)

        v1 = v / cp.linalg.norm(v)

    # Pre-allocate Tmat for tridiagonal matrix storage
    Tmat = cp.zeros((m + 1, m + 1))
    tr = []  # Track convergence values


    VV = cp.zeros((n, m))
    rr = []
    #indx = []
    X1 = []
    while k < m:
        k += 1
        VV[:, k - 1] = v1

        # Perform matrix-vector multiplication and Lanczos step
        v = B @ v1 - bet * v0

        # Compute alpha and perform reorthogonalization if needed
        alp = cp.dot(v1, v)

        # if enableMexFilesTest == 0:
        v -= alp * v1

        '''
            reorth  -- test for reorth. needed!
        '''
        if reorth:
            subVV = VV[:, :k]
            v -= subVV @ (subVV.T @ v)

        # normalize and store v1
        bet = cp.linalg.norm(v)
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
            rr, X1 = eigh(Tmat[:k, :k])
            tr1 = cp.sum(rr[:nev])
            tr.append(tr1)
            ll += 1

        # Stopping criterion based on eigenvalue sum convergence
        if enableConvergenceTest and (ll > 1 and (cp.abs(tr[-1] - tr[-2]) < tol * cp.abs(tr[-2]))):
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
        rrCopy = cp.array(rrCopy)
        while counter < len(rrCopy) - 1:
            rrCopy = cp.delete(rrCopy, cp.where(cp.abs(rrCopy[counter] - rrCopy[counter+1:]) < 1e-5)[0] + counter + 1)
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
        W, _ = qr(W, mode='reduced')

        # Calculate the Rayleigh quotient matrix
        Vin = B @ W
        if OPTIMIZATIONLEVEL != 0:
            G = Rayleighritz(Vin, W, mev)
        else:
            # Construct the Rayleigh quotient matrix manually
            G = W.T @ Vin           
            G = (G + G.T) * 0.5

            # Test against compiled implementation if enabled
            if enableMexFilesTest == 1:
                G2 = Rayleighritz(Vin, W, mev)
                if cp.any(abs(G - G2) > 1e-6):
                    ValueError("Mex file discrepancy for Rayleighritz.c")

        # Compute eigenvalues and eigenvectors of G
        rr, X1 = eigh(G)

        # Select the top `nev` eigenvalues and corresponding eigenvectors
        lam = rr[:nev]
        Y = X1[:, :nev]
        W = W @ Y

    return W, lam
