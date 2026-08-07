import cupy as cp

try:
    from .FermiDirac_gpu import FermiDirac
except ImportError:
    from .FermiDirac_gpu import FermiDirac


def occupations(lam, Temp, Nelec, tol):
    """
    Uses a bisection algorithm to find the Fermi level and occupation numbers.

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
    lam = cp.asarray(lam, dtype=cp.float32)
    # Set up initial values for bisection
    a = cp.min(lam) - 1.0
    fa, _ = FermiDirac(lam, a, Temp, Nelec)
    lmax = min(max(1, int(cp.ceil(Nelec / 2).item()) + 1), len(lam))
    b = lam[lmax - 1] + 1.0
    fb, _ = FermiDirac(lam, b, Temp, Nelec)

    # Midpoint for the bisection algorithm
    c = (b + a) / 2
    fc, occup = FermiDirac(lam, c, Temp, Nelec)

    # Calculate error: sum of occupation numbers should match `Nelec`
    error = 2 * cp.sum(occup) - Nelec

    # If `fa * fb > tol`, return default occupation values
    if fa * fb > tol:
        c = b
        occup = cp.ones(len(lam), dtype=cp.float32)
        print('In bisect - fa*fb > 0')
        return c, occup

    # Start the bisection loop
    while abs(error) > tol and its < maxits:
        its += 1
        c = (b + a) / 2
        fc, occup = FermiDirac(lam, c, Temp, Nelec)
        error = 2 * cp.sum(occup) - Nelec

        # Update interval for bisection
        if fc * fb < 0:
            a = c
        else:
            b = c
            fb = fc

    return c, occup
