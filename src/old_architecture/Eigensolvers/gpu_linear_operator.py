import cupy as cp
import cupyx.scipy.sparse as cpsparse
import scipy.sparse as sps


def is_gpu_linear_operator(value):
    """Return True for custom GPU operators that support matmul and shape."""
    return getattr(value, "_gpu_linear_operator", False)


def to_gpu_matrix(value, return_mode=False):
    """Convert a matrix-like input to a CuPy object or pass through a GPU operator."""
    if is_gpu_linear_operator(value):
        result = value
        mode = "operator"
    elif isinstance(value, cp.ndarray):
        if value.dtype == cp.float32:
            result = value
        else:
            result = value.astype(cp.float32, copy=False)
        mode = "dense"
    elif isinstance(value, cpsparse.spmatrix):
        if value.dtype == cp.float32:
            result = value
        else:
            result = value.astype(cp.float32)
        mode = "sparse"
    elif sps.issparse(value):
        result = cpsparse.csr_matrix(value.astype("float32"))
        mode = "sparse"
    else:
        result = cp.asarray(value, dtype=cp.float32)
        mode = "dense"

    if return_mode:
        return result, mode
    return result


class ShiftedHamiltonianOperator:
    """GPU-resident linear operator for base_matrix + diag(diagonal)."""

    _gpu_linear_operator = True

    def __init__(self, base_matrix, diagonal):
        self.base_matrix = to_gpu_matrix(base_matrix)
        self.shape = self.base_matrix.shape
        self.dtype = cp.float32
        self.diagonal = None
        self.update_diagonal(diagonal)

    def update_diagonal(self, diagonal):
        """Replace the diagonal term without rebuilding the sparse base matrix."""
        if isinstance(diagonal, cp.ndarray):
            diag = diagonal.astype(cp.float32, copy=False).reshape(-1)
        else:
            diag = cp.asarray(diagonal, dtype=cp.float32).reshape(-1)

        if diag.size != self.shape[0]:
            raise ValueError(
                f"Diagonal length {diag.size} does not match operator size {self.shape[0]}."
            )

        self.diagonal = diag
        return self

    def __matmul__(self, other):
        if isinstance(other, cp.ndarray):
            other_dev = other.astype(cp.float32, copy=False)
        else:
            other_dev = cp.asarray(other, dtype=cp.float32)

        result = self.base_matrix @ other_dev
        if other_dev.ndim == 1:
            return result + self.diagonal * other_dev
        return result + self.diagonal[:, None] * other_dev
