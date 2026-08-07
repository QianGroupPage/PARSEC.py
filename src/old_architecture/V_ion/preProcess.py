import numpy as np


def preProcess(v):
    """
    Pre-process the vector `v` by keeping only the relevant points.
    Points where the function is constant (within a tolerance) are removed.

    Parameters:
    v (ndarray): Input column vector to pre-process.

    Returns:
    I (ndarray): List of indexes of relevant points.
    """
    I = [0]  # In Python, indexing starts from 0
    eps = 10e-6

    # Iterate over the vector and check for changes greater than eps
    for i in range(1, len(v) - 1):
        if np.abs(v[i] - v[i - 1]) > eps:
            I.append(i)

    # Always keep the last point
    I.append(len(v) - 1)

    return np.array(I)  # Return as numpy array for indexing
