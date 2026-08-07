import numpy as np

# TODO: C++ version
def Rayleighritz(Vin, W, n):
    """
    Computes the overlap matrix G based on Vin and W.

    Parameters:
    Vin (ndarray): The input matrix Vin of size (m, n).
    W (ndarray): The input matrix W of size (m, n).
    n  (int): Size of the output matrix G (n x n).

    Returns:
    G (ndarray): Output matrix G of size (n, n).
    """
    # Ensure Vin and W are numpy arrays
    Vin = np.asarray(Vin)
    W = np.asarray(W)

    # Initialize the output matrix G with zeros
    G = np.zeros((n, n))

    # Calculate G = W^T * Vin
    for j in range(n):
        for i in range(j + 1):
            # Compute the dot product for G[i, j] and G[j, i]
            G[i, j] = np.dot(Vin[:, i], W[:, j])
            G[j, i] = G[i, j]  # Symmetric matrix

    return G
