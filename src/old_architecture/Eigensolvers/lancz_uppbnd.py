import numpy as np
from scipy.linalg import eigh

def lancz_uppbnd(n, A, k=6):
    """
    apply k steps Lanczos to get the upper bound of abs(eigh(A)).
  
    Input: 
        n  ---  dimension
        k  ---  (optional) perform k steps of Lanczos
                if not provided, k =4 (a relatively small k is enough)
        A  ---  (optional) the matrix (or a script name for MV) 
  
     Output:
        upperb  ---  estimated upper bound for the eigenvalues
        ritzv   ---  ritz values
    """
    k = min(max(k, 6), 10)  # do not go over 10 steps

    T = np.zeros((k, k))
    v = np.random.rand(n, 1)
    v = v / np.linalg.norm(v)

    tol = 2.5e-16  # before ||f|| reaches eps, convergence should
                   # have happened, so no need to ask for a small ||f|| 

    upperb = np.zeros((3, k))  # save the bounds for each step j=2:k

    f = A @ v
    alpha = np.vdot(v, f)
    f = f - alpha * v
    T[0, 0] = alpha
    beta = np.linalg.norm(f)

    # Compute the bounds for j=1, using rayleight quotient and its corresponding r
    upperb[0, 0] = alpha + beta
    upperb[1, 0] = upperb[0, 0]
    upperb[2, 0] = upperb[0, 0]

    isbreak = 0

    for j in range(2, k + 1):  # run k steps

        if beta > tol:
            v0 = v
            v = f / beta
            f = A @ v
            f = f - beta * v0
            alpha = np.vdot(v, f)
            f = f - alpha * v
            T[j - 1, j - 2] = beta
            T[j - 2, j - 1] = beta
            T[j - 1, j - 1] = alpha
        else:
            isbreak = 1
            print(f'j = {j}, invariant subspace found')
            break

        beta = np.linalg.norm(f)
        if isbreak != 1:
            ritzv, X = eigh(T[0: j + 1, 0: j + 1])
        else:
            ritzv, X = eigh(T[0: j, 0: j])

        if beta < 1e-2:
            beta = beta * 10

        upperb[0, j - 1] = ritzv[-1] + beta
        upperb[1, j - 1] = ritzv[-1] + abs(X[-1, -1]) * beta
        upperb[2, j - 1] = ritzv[-1] + max(abs(X[-1, :])) * beta

    uppbnd = (upperb[0, j - 1] + upperb[1, j - 1]) / 2
    uppbnd = (upperb[2, j - 1] + uppbnd) / 2

    return uppbnd, ritzv
