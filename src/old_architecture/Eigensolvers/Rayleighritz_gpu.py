import cupy as cp
# TODO: C++ version
def Rayleighritz(Vin, W, n):
    """
    Python translation of the C++ `Rayleighritz` function.

    Computes the overlap matrix G based on Vin and W.

    Parameters:
    Vin (ndarray): The input matrix Vin of size (m, n).
    W (ndarray): The input matrix W of size (m, n).
    n  (int): Size of the output matrix G (n x n).

    Returns:
    G (ndarray): Output matrix G of size (n, n).
    """


    # Calculate G = W^T * Vin
    G = W.T @ Vin
    return 0.5 * (G + G.T)

