
def fsplevalIO(z, c, d, xi, x, j_in):
    """
    Python version of fsplevalIO to evaluate free spline function at x.

    Parameters:
    z (ndarray): Spline z coefficients.
    c (ndarray): Spline c coefficients.
    d (ndarray): Spline d coefficients.
    xi (ndarray): Spline xi points.
    x (float): Point at which to evaluate the spline.
    j_in (int): Interval index to try first.

    Returns:
    y (float): Evaluated value of the spline at x.
    j_out (int): Output interval index for the next call.
    """
    n = len(xi)

    # Check if the input j_in is out of bounds
    if j_in < 0 or j_in >= n - 1:
        j_in = 0

    # Handle cases where x is out of the xi range
    if x < xi[0]:
        x = xi[0]
        j_in = 0
    elif x > xi[-1]:
        x = xi[-1]
        j_in = n - 2

    j_out = j_in

    # If x is not between xi[j_in] and xi[j_in + 1], perform binary search
    if not (xi[j_in] <= x <= xi[j_in + 1]):
        n = len(xi)
        fflag = 0
        ind_low = 0
        ind_high = n - 1
        if xi[ind_low] <= x <= xi[ind_high]:
            while ind_high - ind_low > 1:
                ind_middle = (ind_high + ind_low) // 2
                val_middle = xi[ind_middle]
                if x < val_middle:
                    ind_high = ind_middle
                else:
                    ind_low = ind_middle
            i_int = ind_low
        else:
            fflag = -1
            i_int = 0
        j_out = i_int

        if (fflag != 0):
            ValueError(' SPLINE ERROR [ in binary search ] ')

    # Compute spline evaluation
    t1 = xi[j_out + 1] - x
    t2 = x - xi[j_out]
    h_j_out = xi[j_out + 1] - xi[j_out]

    y = (t1 * (z[j_out] * t1 * t1 / h_j_out + c[j_out]) +
         t2 * (z[j_out + 1] * t2 * t2 / h_j_out + d[j_out]))

    return y, j_out + 1
