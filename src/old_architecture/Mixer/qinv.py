import numpy as np
from scipy.linalg import qr


def qinv(A, tol=np.finfo(float).eps):
    """
    B = qinv(A [, tol])
    Quasi-inverse of A by QR factorization with column pivoting.
    tol is the tolerance, a small positive scalar.

    The QR factorization of A with column pivoting is denoted by A*P=Q*R,
    where A is the given m-by-n matrix (m>=n), P is permutation matrix,
    Q is an m-by-n matrix of orthogonal columns, R is rank-revealing upper
    triangular matrix in the form [ R11 R12 ; 0 0 ], with R11 nonsingular.

    In some sense S = [ R11^{-1} 0 ; 0 0 ] plays the role of the inverse
    of R. We call B = P*S*Q^T the quasi-inverse of B. If A is square and
    has full-rank, then B is the inverse of A. Note that x=B*b is a
    solution to the least square problem of minimizing ||Ax-b||_2.
    """

    # Get the dimensions of A
    m, n = A.shape

    # Handle the case when m < n by transposing the matrix and calling qinv recursively
    if m < n:
        return qinv(A.T, tol).T

    # Calculate the tolerance
    tol = tol * np.linalg.norm(A, np.inf)

    # Perform QR factorization with column pivoting
    Q, R, P = qr(A, mode='economic', pivoting=True)

    # Determine the numerical rank
    r = n
    for i in range(n):
        if abs(R[i, i]) < tol:
            r = i
            break

    # Construct the inverse of the upper triangular matrix R(1:r,1:r)
    S = np.zeros((r, r))
    for j in range(r - 1, -1, -1):  # Loop from r-1 down to 0
        S[j, j] = 1 / R[j, j]
        for i in range(j - 1, -1, -1):  # Loop from j-1 down to 0
            t = 0
            for k in range(i + 1, r):  # Loop from i+1 up to r-1
                t += R[i, k] * S[k, j]
            S[i, j] = -t / R[i, i]

    # Form the quasi-inverse
    B = np.zeros((n, m))
    B[:r, :] = S @ Q[:, :r].T

    # Permute B back to the original column order
    q = np.zeros(n, dtype=int)
    for i in range(n):
        q[P[i]] = i
    B = B[q, :]

    return B
