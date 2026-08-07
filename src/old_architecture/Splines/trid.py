import numpy as np

def trid(a, d, c, r):
    """
    Solve a tridiagonal system A * x = r, where A is tridiagonal with:
    - `a`: sub-diagonal (below the main diagonal)
    - `d`: main diagonal
    - `c`: super-diagonal (above the main diagonal)

    Parameters:
    a (ndarray): Sub-diagonal elements (length n-1).
    d (ndarray): Main diagonal elements (length n).
    c (ndarray): Super-diagonal elements (length n-1).
    r (ndarray): Right-hand side vector (length n).

    Returns:
    r (ndarray): Solution vector.
    """
    n = len(d)

    # Forward elimination
    for k in range(1, n):
        piv = a[k] / d[k-1]
        d[k] = d[k] - piv * c[k-1]
        r[k] = r[k] - piv * r[k-1]

    # Back substitution
    r[n - 1] = r[n - 1] / d[n - 1]
    for j in range(n - 2, -1, -1):
        r[j] = (r[j] - c[j] * r[j + 1]) / d[j]

    return r
