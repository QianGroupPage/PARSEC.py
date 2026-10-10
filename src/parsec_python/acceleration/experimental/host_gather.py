"""Column-friendly host input gathering; outside GPU benchmark timings."""
import numpy as np


def gather_rows_columns(matrix, rows, column_start=0, column_stop=None):
    """Return selected FP64 entries in Fortran order without a C-order gather.

    For contiguous rows, a basic slice is used.  Selecting every row of an
    F-contiguous matrix can therefore return a zero-copy view: callers must
    treat this result as read-only input and must not assume independent
    storage.  Partial contiguous rows require only one F-order copy.

    Arbitrary rows are gathered one column at a time into a single F-order
    destination.  This avoids row-wise traversal of widely separated columns
    in a large F-order memory-mapped file, and preserves row order/duplicates.
    """
    source = np.asanyarray(matrix)
    indices = np.asarray(rows)
    if source.ndim != 2 or source.dtype != np.dtype(np.float64):
        raise ValueError('source must be a two-dimensional FP64 array')
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('rows must be a one-dimensional integer array')
    if np.any(indices < 0) or np.any(indices >= source.shape[0]):
        raise ValueError('row index is outside the source')
    stop = source.shape[1] if column_stop is None else column_stop
    if (int(column_start) != column_start or int(stop) != stop
            or not 0 <= column_start <= stop <= source.shape[1]):
        raise ValueError('invalid column interval')
    first, stop = int(column_start), int(stop)
    if not len(indices):
        return np.empty((0, stop-first), dtype=np.float64, order='F')
    if len(indices) == 1 or np.all(np.diff(indices) == 1):
        return np.asfortranarray(source[int(indices[0]):int(indices[-1])+1, first:stop])
    destination = np.empty((len(indices), stop-first), dtype=np.float64, order='F')
    for local, column in enumerate(range(first, stop)):
        np.take(source[:, column], indices, out=destination[:, local])
    return destination


__all__ = ['gather_rows_columns']
