import numpy as np
from .trid import trid


def fspline(xi, yi):
    """
    Computes the cubic spline coefficients.

    Parameters:
    xi (ndarray): x-values of the data points.
    yi (ndarray): y-values of the data points.

    Returns:
    z (ndarray): Second derivatives of the spline.
    c (ndarray): Spline coefficient c.
    d (ndarray): Spline coefficient d.
    """

    xi = xi.flatten()  # Ensure xi is a 1D array
    yi = yi.flatten()  # Ensure yi is a 1D array
    np1 = len(xi)  # Number of points
    n = np1 - 1  # Number of intervals

    # Get the step sizes (h)
    h = xi[1:np1] - xi[0:n]

    # Calculate the right-hand-side (r)
    r = (yi[1:np1] - yi[0:n]) / h[0:n]
    r = r[1:n] - r[0:n-1]

    # Set up the tridiagonal matrix system
    L = h[0:n-1]  # Lower diagonal
    U = h[1:n]  # Upper diagonal
    D = 2 * (L + U)  # Main diagonal
    # Solve the tridiagonal system
    z = trid(L, D, U, r)

    # Set the end values to 0
    z = np.concatenate([[0], z, [0]])

    # Calculate the cubic spline coefficients
    c = (yi[0:n] / h[0:n]) - (z[0:n] * h[0:n])
    d = (yi[1:np1] / h[0:n]) - (z[1:np1] * h[0:n])

    return z, c, d
