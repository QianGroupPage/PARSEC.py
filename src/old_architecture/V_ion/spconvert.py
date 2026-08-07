from scipy import sparse


def spconvert(D):
    """
    Converts a matrix D to a sparse matrix format.

    If D is a 2D numpy array with size Nx3, constructs a sparse matrix such that:
        S[i[k], j[k]] = re[k], where columns [i, j, re] are from D.

    If D has size Nx4, constructs a sparse matrix such that:
        S[i[k], j[k]] = re[k] + 1j * im[k], where columns [i, j, re, im] are from D.

    If D is already sparse, it returns D unchanged.

    Parameters:
    D (numpy.ndarray or scipy.sparse matrix): Input matrix.

    Returns:
    scipy.sparse.coo_matrix: The resulting sparse matrix.
    """
    if sparse.issparse(D):
        return D

    na = D.shape[1]

    if na == 3:
        # Construct sparse matrix using columns i, j, re
        S = sparse.coo_matrix((D[:, 2], (D[:, 0].astype(int) - 1, D[:, 1].astype(int) - 1)))  # Adjusting for 0-based indexing
    elif na == 4:
        # Construct sparse matrix using columns i, j, re, im
        S = sparse.coo_matrix((D[:, 2] + 1j * D[:, 3], (D[:, 0].astype(int) - 1, D[:, 1].astype(int) - 1)))  # Adjusting for 0-based indexing
    else:
        raise ValueError("Wrong array size: Expected 3 or 4 columns in D.")

    return S
