import numpy as np
from .FermiDirac import FermiDirac


def occupations(lam, Temp, Nelec, tol):
    """
    Parameters:
    lam (ndarray): Array of eigenvalues.
    Temp (float): Temperature (in energy units).
    Nelec (float): Number of electrons to be filled.
    tol (float): Tolerance for convergence.

    Returns:
    c (float): Fermi level.
    occup (ndarray): Occupation numbers for each eigenvalue.
    """
    # Initialize variables
    its = 0
    maxits = 200

    # Set up initial values for bisection
    a = np.min(lam) - 1
    fa, _ = FermiDirac(lam, a, Temp, Nelec)
    lmax = min(max(1, int(np.ceil(Nelec / 2)) + 1), len(lam))
    b = lam[lmax - 1] + 1
    fb, _ = FermiDirac(lam, b, Temp, Nelec)

    # Midpoint for the bisection algorithm
    c = (b + a) / 2
    fc, occup = FermiDirac(lam, c, Temp, Nelec)

    # Calculate error: sum of occupation numbers should match `Nelec`
    error = 2 * np.sum(occup) - Nelec

    # If `fa * fb > tol`, return default occupation values
    if fa * fb > tol:
        c = b
        occup = np.ones(len(lam))
        print('In bisect - fa*fb > 0')
        return c, occup

    # Start the bisection loop
    while abs(error) > tol and its < maxits:
        its += 1
        c = (b + a) / 2
        fc, occup = FermiDirac(lam, c, Temp, Nelec)
        error = 2 * np.sum(occup) - Nelec

        # Update interval for bisection
        if fc * fb < 0:
            a = c
        else:
            b = c
            fb = fc

    return c, occup
