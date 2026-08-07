import numpy as np

OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0


def pcg(A, rhs, x0, m, tol, *args):
    """
    Solve the linear system A * x = rhs using the Preconditioned Conjugate Gradient method.

    Parameters:
    A (ndarray or sparse matrix): The matrix A.
    rhs (ndarray): The right-hand side vector.
    x0 (ndarray): Initial guess for the solution.
    m (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    PRE (optional): Preconditioner structure. Required if `precfun` is provided.
    precfun (optional): Function handle for applying the preconditioner.

    Returns:
    x (ndarray): Solution vector.
    its (int): Number of iterations performed.
    """
    if len(args) == 2:
        PRE, precfun = args
    else:
        PRE = precfun = None

    # Ensure all inputs are real type
    A = A.astype(np.float64)
    rhs = rhs.astype(np.float64)
    x = x0.astype(np.float64)

    if OPTIMIZATIONLEVEL != 0 and len(args) == 0:
        # x, its = pcgMexFile(A, rhs, x0, m, tol)
        ValueError("not implemented")
    else:
        # Initial solution and residual
        r = rhs - A.dot(x)

        # Apply the preconditioner, if available
        if PRE is not None and precfun is not None:
            z = precfun(PRE, r)
        else:
            z = r

        # Initialize the search direction
        p = z
        ro1 = np.dot(z, r)  # ro1 = z' * r
        tol1 = tol * tol * ro1

        # Number of iterations
        its = 0

        # Main PCG iteration loop
        while its < m and ro1 > tol1:
            its += 1
            ro = ro1
            ap = A.dot(p)  # Compute A * p
            alp = ro / np.dot(ap, p)  # Compute alpha = ro / (ap' * p)

            # Update solution and residual
            x += alp * p
            r -= alp * ap

            # Apply the preconditioner again, if available
            if PRE is not None and precfun is not None:
                z = precfun(PRE, r)
            else:
                z = r

            # Update ro1
            ro1 = np.dot(z, r)
            bet = ro1 / ro  # Compute beta = ro1 / ro

            # Update search direction
            p = z + bet * p

        if enableMexFilesTest == 1:
            print("not implemented")
            # x2, its2 = pcgMexFile(A, rhs, x0, m, tol)
            # if np.any(abs(x - x2) > 0.000001) or np.any(abs(its - its2) > 0.000001):
            #     ValueError("Mex file discrepancy for pcgMexFile.c")

    return x, its
