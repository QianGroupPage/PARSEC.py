import numpy as np
from scipy.linalg import eigh

enableMexFilesTest = 0
def lanczosForChefsi1(B, nev, v, m, tol):
    """
        B   = matrix;
        nev = number of wanted egenvalues;
        v   = initial vector
        m   = number of steps
        tol = tolerance for stopping
    can do with (full) reorthogonalization (reorth=1)
    or no -- reorthogonalization (reorth=0)
    this version of lanczos is called by chefsi1 and only does the
    calculations necessary to find bound
    """
    reorth = 0

    n = B.shape[0]
    v = v / np.linalg.norm(v)
    v1 = v
    v0 = np.zeros((n,1))
    k = 0
    bet = 0
    ll = 0

    # Pre-allocating
    if reorth:
        VV = np.zeros((n, m))

    Tmat = np.zeros((m + 1, m + 1))
    tr = np.zeros((1, m))

    # Main Lanczos loop
    while k < m:
        k = k + 1

        if reorth:
            VV[:, k - 1] = v1

        v = B @ v1 - bet * v0

        # TODO: compare performance between c++ and python
        # if enableMexFilesTest == 1:
        #     v2 = BTimesv1MinusbetTimesv0(B, v0, v1, bet, findFirstColumnWithNonZeroElement(B))
        #     if (any(abs(v - v2) > 0.000001))
        #         exception = struct('message', 'Mex file descreptency for BTimesv1MinusbetTimesv0', 'identifier', [],
        #                            'stack', {mfilename('fullpath')
        #         'filler'})
        #         logError(exception)

        alp = np.vdot(v1, v)

        if enableMexFilesTest == 0:
            v = v - alp * v1
        # else:
        # TODO: compare performance between c++ and python

        #     vTemp = v - alp * v1
        #     vTemp2 = vMinusalpTimesv1(v, v1, alp)
        #     v = vTemp
        #     if (any(abs(vTemp - vTemp2) > 0.000001))
        #         exception = struct('message', 'Mex file descreptency for vMinusalpTimesv1', 'identifier', [], 'stack',
        #                            {mfilename('fullpath')
        #         'filler'})
        #         logError(exception)

        # reorth  -- test for reorth. needed!
        if reorth:
            t = VV[:, :k].H @ v
            v = v - VV[:, :k] @ t

        # normalize and store v1
        bet = np.linalg.norm(v)
        v0 = v1
        v1 = v / bet

        # Update the tridiagonal matrix
        Tmat[k - 1, k - 1] = alp
        Tmat[k, k - 1] = bet
        Tmat[k - 1, k] = bet
        NTest = min(5 * nev, m) # when to start testing
        # tr, ll,  == for plotting
        if (k >= NTest and k % 10 == 0) or k == m:
            rr, _ = eigh(Tmat[:k, :k])
            rr = np.sort(rr) # sort increasingly

            # upper bound used by chef_si -- not used otherwise.
            bound = max(np.abs(rr)) + bet
            tr1 = np.sum(rr[:nev])
            ll = ll + 1
            tr[ll - 1] = tr1

        # stopping criterion based on sum of eigenvalues.
        # make sure this is is well-converged!
        if ll > 1 and np.abs(tr[ll - 1] - tr[ll - 2]) < tol * tr[ll - 2]:
            break

    return bound
