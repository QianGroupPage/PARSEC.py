"""Brillouin-zone k-point sampling for an orthorhombic periodic cell.

Only a uniform, unreduced Monkhorst-Pack grid is implemented: every k-point
carries the *same* weight ``1/(n1*n2*n3)`` (no symmetry-based irreducible-BZ
reduction, no user-supplied explicit k-point list with individual weights).
This deliberately keeps :mod:`~parsec_python.SCF.kpoints`'s Fermi-level
pooling simple: reusing
:func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations`'s existing
scalar ``degeneracy`` parameter (``degeneracy = 2*weight``, spin-unpolarized)
requires every pooled eigenvalue to share one scale factor, which only holds
when every k-point's weight is identical.
"""

from __future__ import annotations

import numpy as np

from ..models import PeriodicCell


def monkhorst_pack_grid(
    cell: PeriodicCell,
    dimensions: tuple[int, int, int],
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(k_points, weights)`` for a uniform Monkhorst-Pack grid.

    ``dimensions=(n1,n2,n3)`` are the number of divisions along each
    reciprocal lattice vector.  For an orthorhombic cell with lattice
    vectors ``diag(Lx,Ly,Lz)``, the reciprocal lattice vectors are
    ``b_a = (2*pi/L_a)*e_a``, and the unshifted grid points are

    ``k_(i1,i2,i3) = sum_a [(2*i_a-n_a-1)/(2*n_a)] * b_a``,
    ``i_a = 1,...,n_a``,

    the standard Monkhorst-Pack construction, optionally displaced by
    ``shift`` (given in the same Cartesian reciprocal units as the returned
    k-points, e.g. a half-grid-step shift to avoid sampling exactly at
    Gamma).  Every returned weight is exactly ``1/(n1*n2*n3)``: this is the
    full uniform grid, not reduced to the irreducible wedge.

    ``k_points`` has shape ``(n1*n2*n3, 3)``, ``weights`` has shape
    ``(n1*n2*n3,)`` and sums to 1.
    """
    dims = tuple(int(value) for value in dimensions)
    if len(dims) != 3 or any(value < 1 for value in dims):
        raise ValueError("dimensions must be three positive integers")
    off_diagonal = cell.lattice_vectors - np.diag(np.diag(cell.lattice_vectors))
    if not np.allclose(off_diagonal, 0.0, atol=1.0e-10):
        raise ValueError(
            "Monkhorst-Pack sampling only supports an orthorhombic cell "
            "(diagonal lattice_vectors)"
        )
    side_lengths = np.diag(cell.lattice_vectors)
    reciprocal = 2.0 * np.pi / side_lengths
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (3,):
        raise ValueError("shift must have shape (3,)")

    fractional_axes = [
        (2.0 * np.arange(1, n + 1) - n - 1) / (2.0 * n) for n in dims
    ]
    grids = np.meshgrid(*fractional_axes, indexing="ij")
    fractional = np.column_stack([axis.reshape(-1) for axis in grids])
    k_points = fractional * reciprocal[None, :] + shift[None, :]
    n_total = int(np.prod(dims))
    weights = np.full(n_total, 1.0 / n_total, dtype=float)
    return k_points, weights


__all__ = ["monkhorst_pack_grid"]
