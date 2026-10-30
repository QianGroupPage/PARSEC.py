"""The nonlocal force of one k-point, contracted where its orbitals are.

``F_a = 4 w_k sum_n f_n sum_q s_q Re[ conj(<beta_q|u_n>) <G_q|u_n> ]``

(:func:`parsec_python.Forces.nonlocal_force.build_periodic_projector_gradients`) needs the overlaps of every occupied
Bloch orbital ``u_n`` with the ``Q`` projectors and with their three gradients.  The orbitals of a k-point stay on the
device that solved it (a calculation of hundreds of states on a million grid points holds gigabytes there), so the
overlaps are formed with sparse products on that device, a block of orbitals at a time, and only the ``(atoms, 3)``
result comes back.  The projector and gradient matrices are sparse (a few thousand grid points per projector); they
are uploaded once per call.

The sums are written against an array module ``xp`` (CuPy) and its sparse module; with NumPy and SciPy they are the
readable contraction of :func:`parsec_python.Forces.nonlocal_force.nonlocal_forces`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from parsec_python.Forces.nonlocal_force import (
    NonlocalProjectorGradients,
    gradient_columns_for,
)

#: Device bytes one block of orbitals and its overlaps may take.
_BLOCK_BYTES = 512 * 1024 * 1024


def _sparse_module(xp):
    if xp is np:
        return sp
    import cupyx.scipy.sparse as device_sparse  # type: ignore[import-not-found]

    return device_sparse


def _conjugate_transpose(matrix) -> sp.csr_matrix:
    """Host CSR of ``M^H``, the matrix that turns an orbital into its overlaps (``M^T`` for real data)."""

    matrix = sp.csc_matrix(matrix)
    transposed = matrix.conj().T if np.iscomplexobj(matrix.data) else matrix.T
    return sp.csr_matrix(transposed, dtype=np.complex128)


def device_nonlocal_forces(
    xp,
    operator,
    gradients: NonlocalProjectorGradients,
    wavefunctions,
    occupations: np.ndarray,
    atom_count: int,
    *,
    sparse=None,
    block_bytes: int = _BLOCK_BYTES,
) -> np.ndarray:
    """``(atoms, 3)`` nonlocal forces in Ry/bohr of the orbitals ``wavefunctions`` (an ``xp`` array).

    ``occupations`` carries the occupation of each column times the k-point weight, as for
    :func:`parsec_python.Forces.nonlocal_force.nonlocal_forces`; empty states are skipped.  ``sparse`` is the sparse
    module of ``xp`` (``cupyx.scipy.sparse`` for CuPy).
    """

    occupations = np.asarray(occupations, dtype=np.float64)
    if operator.projectors.shape[1] == 0 or occupations.size == 0:
        return np.zeros((atom_count, 3), dtype=np.float64)
    sparse = _sparse_module(xp) if sparse is None else sparse
    columns = gradient_columns_for(operator.labels, gradients.labels)
    occupied = np.flatnonzero(occupations != 0.0)
    if not occupied.size:
        return np.zeros((atom_count, 3), dtype=np.float64)

    overlap_matrix = sparse.csr_matrix(_conjugate_transpose(operator.projectors))
    gradient_matrices = [
        sparse.csr_matrix(_conjugate_transpose(sp.csc_matrix(component)[:, columns])) for component in gradients.components
    ]
    projector_count = int(operator.projectors.shape[1])
    grid_size = int(wavefunctions.shape[0])
    # a block holds the orbitals and four sets of overlaps, all complex
    width = max(1, int(block_bytes // (16 * (grid_size + 4 * projector_count))))
    per_projector = np.zeros((projector_count, 3), dtype=np.float64)
    for start in range(0, int(occupied.size), width):
        chosen = occupied[start : start + width]
        block = xp.ascontiguousarray(wavefunctions[:, xp.asarray(chosen)]).astype(xp.complex128, copy=False)
        weights = xp.asarray(occupations[chosen])
        overlaps = overlap_matrix @ block
        conjugate = xp.conj(overlaps)
        for component, matrix in enumerate(gradient_matrices):
            coupling = xp.real(conjugate * (matrix @ block))
            sums = coupling @ weights
            per_projector[:, component] += np.asarray(sums.get() if hasattr(sums, "get") else sums)
    per_projector *= 4.0 * np.asarray(operator.signs, dtype=np.float64)[:, None]

    owners = np.fromiter((label[0] for label in operator.labels), dtype=np.int64, count=len(operator.labels))
    forces = np.zeros((atom_count, 3), dtype=np.float64)
    for component in range(3):
        forces[:, component] = np.bincount(owners, weights=per_projector[:, component], minlength=atom_count)
    return forces


__all__ = ["device_nonlocal_forces"]
