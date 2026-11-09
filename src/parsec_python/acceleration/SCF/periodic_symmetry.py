"""``Symmetrize_Charge_Density`` of a periodic cell on an array backend (NumPy or CuPy).

The average of :func:`parsec_python.Grid.point_group.symmetrize_scalar` over the operations of a
:class:`~parsec_python.Grid.space_group.PeriodicGridSymmetry` (``symm_scalar`` of ``newrho.F90``),

``f_sym(i) = (1/|G|) sum_g f(images[g, i])``,

for a field that stays on the device of a k-point solver.  Where the group carries the integer affine maps of its
operations (``images[g, i] = (M_g i + c_g) mod n`` on a row-major grid,
:func:`parsec_python.Grid.space_group.periodic_grid_symmetry`), the gather indices of one operation are computed on
the device when they are needed (one elementwise kernel per operation): no table of operations x points is kept,
which for 3 M points and 24 operations is 0.29 GB of int32.  A group without that form uploads its table once.
The terms are added to the total in the order of the operations, as the host does, so the field is the host's bit
for bit; the sums of the report (largest change, charges) are reduced on the backend and may differ from the
host's in their last bits (they are printed only).

The host average at 3 M points costs ~1.5 s per call on a laptop (24 gathers); on an A100 it is milliseconds.
"""

from __future__ import annotations

import numpy as np

from parsec_python.Grid.point_group import ScalarSymmetrization
from parsec_python.Grid.space_group import PeriodicGridSymmetry, affine_grid_images


class PeriodicDensitySymmetrizer:
    """The space-group average of periodic fields on the array module ``xp``.

    ``group`` is the :class:`PeriodicGridSymmetry` of the prepared system (``system.density_symmetry``).
    """

    def __init__(self, group: PeriodicGridSymmetry, xp=np) -> None:
        self.xp = xp
        self.order = int(group.order)
        self.reduced_order = int(group.reduced_order)
        self.size = int(np.asarray(group.images).shape[1])
        maps = getattr(group, "affine_maps", None)
        counts = getattr(group, "counts", None)
        if maps is not None and counts is not None:
            self._maps = tuple(
                (tuple(tuple(int(value) for value in row) for row in np.asarray(matrix)),
                 tuple(int(value) for value in np.asarray(constant)))
                for matrix, constant in maps
            )
            self._counts = tuple(int(value) for value in counts)
            self._table = None
        else:
            self._maps = None
            self._counts = None
            self._table = xp.asarray(np.asarray(group.images))

    def images(self, operation: int):
        """The gather indices ``images[operation, :]`` on the backend."""

        if self._table is not None:
            return self._table[operation]
        matrix, constant = self._maps[operation]
        return affine_grid_images(matrix, constant, self._counts, xp=self.xp)

    def symmetrize_scalar(self, values):
        """``(1/|G|) sum_g f(images[g])`` of ``values`` ``(points,)`` (a backend array)."""

        xp = self.xp
        values = xp.asarray(values, dtype=xp.float64)
        if values.shape != (self.size,):
            raise ValueError(f"a field of {self.size} grid points is required, got shape {values.shape}")
        if self.order <= 1:
            return values.copy()
        total = xp.zeros_like(values)
        for operation in range(self.order):
            total += values[self.images(operation)]
        return total / float(self.order)

    def symmetrize_density(self, density, volume_element: float):
        """Symmetrize a density and report it as :func:`parsec_python.Grid.point_group.symmetrize_density` does."""

        xp = self.xp
        density = xp.asarray(density, dtype=xp.float64)
        symmetric = self.symmetrize_scalar(density)
        before = float(xp.sum(density))
        wedge_sum = before / self.reduced_order
        deviation = float(xp.max(xp.abs(symmetric - density))) if density.size else 0.0
        return symmetric, ScalarSymmetrization(
            maximum_deviation_percent=(deviation / wedge_sum * 100.0 if before else 0.0),
            charge_before=before * float(volume_element),
            charge_after=float(xp.sum(symmetric)) * float(volume_element),
        )


__all__ = ["PeriodicDensitySymmetrizer"]
