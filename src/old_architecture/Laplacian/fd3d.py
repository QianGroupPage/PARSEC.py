import numpy as np
from scipy.sparse import spdiags


def handle_duplicate_diagonals(V_diag, N_diag):
    """
    Handle duplicate diagonal entries by summing the corresponding values in V_diag.

    Parameters:
    V_diag (ndarray): The matrix of diagonal values (each row is a diagonal).
    N_diag (list): The list of diagonal positions.

    Returns:
    V_diag_new (ndarray): The updated diagonal matrix with summed values for duplicate diagonals.
    N_diag_new (list): The updated list of diagonal positions with duplicates removed.
    """
    unique_diags = {}

    for i, diag in enumerate(N_diag):
        if diag in unique_diags:
            # If the diagonal already exists, sum the corresponding values in V_diag
            unique_diags[diag] += V_diag[:, i]
        else:
            # Otherwise, add the new diagonal entry
            unique_diags[diag] = V_diag[:, i]

    # Create the new diagonal matrix and diagonal positions
    N_diag_new = sorted(unique_diags.keys())
    V_diag_new = np.column_stack([unique_diags[diag] for diag in N_diag_new])

    return V_diag_new, N_diag_new


def fd3d(nx, ny, nz, degree):
    """
    Constructs a 3D finite difference Laplacian operator using sparse matrices.

    Parameters:
    nx, ny, nz (int): Grid dimensions in the x, y, and z directions.
    degree (int): Finite difference accuracy (2, 4, or higher).

    Returns:
    A (scipy.sparse.spmatrix): The Laplacian operator matrix in sparse form.
    """

    nxy = nx * ny
    nxyz = nxy * nz

    if degree <= 2:
        X1 = np.kron(np.ones(ny * nz), np.concatenate([[-1] * (nx - 1), [0]]))
        Y1 = np.kron(np.ones(nz), np.concatenate([[-1] * (nx * (ny - 1)), [0] * nx]))
        V_diag = np.column_stack([[-1] * nxyz, np.flipud(Y1), np.flipud(X1), [6] * nxyz, X1, Y1, [-1] * nxyz])
        N_diag = [nx * ny, nx, 1, 0, -1, -nx, -nx * ny]

    elif degree <= 4:
        X1 = np.kron(np.ones(ny * nz), np.concatenate([[-4 / 3] * (nx - 1), [0]]))
        X2 = np.kron(np.ones(ny * nz), np.concatenate([[1 / 12] * (nx - 2), [0] * min(2, nx)]))
        Y1 = np.kron(np.ones(nz), np.concatenate([[-4 / 3] * (nx * (ny - 1)), [0] * nx]))
        Y2 = np.kron(np.ones(nz), np.concatenate([[1 / 12] * (nx * (ny - 2)), [0] * min(2 * nx, nxy)]))
        V_diag = np.column_stack([[1 / 12] * nxyz, [-4 / 3] * nxyz, np.flipud(Y2), np.flipud(Y1),
                                  np.flipud(X2), np.flipud(X1), [15 / 2] * nxyz, X1, X2, Y1, Y2, [-4 / 3] * nxyz,
                                  [1 / 12] * nxyz])
        N_diag = [2 * nxy, nxy, 2 * nx, nx, 2, 1, 0, -1, -2, -nx, -2 * nx, -nxy, -2 * nxy]

    elif degree <= 6:
        X1 = np.kron(np.ones(ny * nz), np.concatenate([[-3 / 2] * (nx - 1), [0] * min(1, nx)]))
        X2 = np.kron(np.ones(ny * nz), np.concatenate([[3 / 20] * (nx - 2), [0] * min(2, nx)]))
        X3 = np.kron(np.ones(ny * nz), np.concatenate([[-1 / 90] * (nx - 3), [0] * min(3, nx)]))
        Y1 = np.kron(np.ones(nz), np.concatenate([[-3 / 2] * (nx * (ny - 1)), [0] * min(nx, nxy)]))
        Y2 = np.kron(np.ones(nz), np.concatenate([[3 / 20] * (nx * (ny - 2)), [0] * min(2 * nx, nxy)]))
        Y3 = np.kron(np.ones(nz), np.concatenate([[-1 / 90] * (nx * (ny - 3)), [0] * min(3 * nx, nxy)]))
        V_diag = np.column_stack([[-1 / 90] * nxyz, [3 / 20] * nxyz, [-3 / 2] * nxyz, np.flipud(Y3),
                                  np.flipud(Y2), np.flipud(Y1), np.flipud(X3), np.flipud(X2), np.flipud(X1),
                                  [49 / 6] * nxyz, X1, X2, X3, Y1, Y2, Y3, [-3 / 2] * nxyz, [3 / 20] * nxyz,
                                  [-1 / 90] * nxyz])
        N_diag = [3 * nxy, 2 * nxy, nxy, 3 * nx, 2 * nx, nx, 3, 2, 1, 0, -1, -2, -3, -nx, -2 * nx, -3 * nx, -nxy,
                  -2 * nxy, -3 * nxy]

    else:
        X1 = np.kron(np.ones(ny * nz), np.concatenate([[-8 / 5] * (nx - 1), [0] * min(1, nx)]))
        X2 = np.kron(np.ones(ny * nz), np.concatenate([[1 / 5] * (nx - 2), [0] * min(2, nx)]))
        X3 = np.kron(np.ones(ny * nz), np.concatenate([[-8 / 315] * (nx - 3), [0] * min(3, nx)]))
        X4 = np.kron(np.ones(ny * nz), np.concatenate([[1 / 560] * (nx - 4), [0] * min(4, nx)]))
        Y1 = np.kron(np.ones(nz), np.concatenate([[-8 / 5] * (nx * (ny - 1)), [0] * min(nxy, nx)]))
        Y2 = np.kron(np.ones(nz), np.concatenate([[1 / 5] * (nx * (ny - 2)), [0] * min(nxy, 2 * nx)]))
        Y3 = np.kron(np.ones(nz), np.concatenate([[-8 / 315] * (nx * (ny - 3)), [0] * min(nxy, 3 * nx)]))
        Y4 = np.kron(np.ones(nz), np.concatenate([[1 / 560] * (nx * (ny - 4)), [0] * min(nxy, 4 * nx)]))
        V_diag = np.column_stack([[1 / 560] * nxyz, [-8 / 315] * nxyz, [1 / 5] * nxyz, [-8 / 5] * nxyz,
                                  np.flipud(Y4), np.flipud(Y3), np.flipud(Y2), np.flipud(Y1),
                                  np.flipud(X4), np.flipud(X3), np.flipud(X2), np.flipud(X1),
                                  [205 / 24] * nxyz, X1, X2, X3, X4, Y1, Y2, Y3, Y4, [-8 / 5] * nxyz, [1 / 5] * nxyz,
                                  [-8 / 315] * nxyz, [1 / 560] * nxyz])
        N_diag = [4 * nxy, 3 * nxy, 2 * nxy, nxy, 4 * nx, 3 * nx, 2 * nx, nx, 4, 3, 2, 1, 0, -1, -2, -3, -4, -nx,
                  -2 * nx, -3 * nx, -4 * nx, -nxy, -2 * nxy, -3 * nxy, -4 * nxy]

    # Handle duplicate diagonals by summing their corresponding values
    V_diag, N_diag = handle_duplicate_diagonals(V_diag, N_diag)

    # Transpose V_diag to match Python's spdiags behavior
    A = spdiags(V_diag.T, N_diag, nxyz, nxyz)
    # n = A.shape[0]
    # print(n)
    #B = A.todense()
    #print(B)

    return A

# fd3d(2,2,2,2)