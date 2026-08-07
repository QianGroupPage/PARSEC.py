
OPTIMIZATIONLEVEL = 0
enableMexFilesTest = 0
def ch_filter(A, x, deg, lam1, low, high):
    """
    --> Apply chebyshev filter to x.
    A    = matrix
    x    = vector (s) to be filtered
    lam1 = estimate of lowest eigenvalue - for scaling
           purposes only [rough estimate OK]
    low, high = interval to be damped.
    """

    e = (high - low) / 2
    c = (high + low) / 2
    sigma1 = e / (lam1 - c)
    sigma = sigma1

    # TODO: compare performance between c++ and python
    # If MEX file testing is enabled, log discrepancies
    # if enableMexFilesTest==1:
    #     # Degree 1 Chebyshev term
    #     y = (A @ x - c * x) * (sigma1 / e)
    #     y2 = chebyshevfilterDegree1(A, x, c, sigma1, e)
    #     if not np.allclose(y, y2, atol=1e-6):
    #         raise ValueError("Mex file discrepancy for chebyshevfilterDegree1")
    #
    #     twoDividedbysigma1 = 2 / sigma1
    #     inverseOfe = 1 / e
    #
    # # Loop for higher degree terms
    #     for i in range(2, deg + 1):
    #         # Calculate the next Chebyshev polynomial coefficients
    #         sigma_new = 1 / (twoDividedbysigma1 - sigma)
    #         t1 = 2 * sigma_new * inverseOfe
    #         t2 = sigma * sigma_new
    #
    #         # Compute the new filtered vector
    #         ynew = (A @ y - c * y) * t1 - t2 * x
    #         ynew2 = chebyshevfilterDegreeN(A, x, y, c, t1, t2)
    #
    #         if not np.allclose(ynew, ynew2, atol=1e-6):
    #             raise ValueError("Mex file discrepancy for chebyshevfilterDegreeN")
    #
    #         # Update the vectors for the next iteration
    #         x = y
    #         y = ynew
    #         sigma = sigma_new
    # else:
    #     if OPTIMIZATIONLEVEL !=0:
    #         y = chebyshevfilterDegree1(A, x, c, sigma1, e)
    #
    #         twoDividedbysigma1 = 2 / sigma1
    #         inverseOfe = 1 / e
    #         for i in range(2, deg + 1):
    #             sigma_new = 1 / (twoDividedbysigma1 - sigma)
    #             t1 = 2 * sigma_new * inverseOfe
    #             t2 = sigma * sigma_new
    #             ynew = chebyshevfilterDegreeN(A, x, y, c, t1, t2)
    #
    #             x = y
    #             y = ynew
    #             sigma = sigma_new
    #     else:
    # degree 1 term
    y = (A @ x - c * x) * (sigma1 / e)

    twoDividedbysigma1 = 2.0 / sigma1
    inverseOfe = 1.0 / e

    # loop to degree
    for i in range(2, deg + 1):
        sigma_new = 1.0 / (twoDividedbysigma1 - sigma)
        t1 = 2.0 * sigma_new * inverseOfe
        t2 = sigma * sigma_new
        ynew = (A @ y - c * y) * t1 - t2 * x

        x = y
        y = ynew
        sigma = sigma_new
    return y
