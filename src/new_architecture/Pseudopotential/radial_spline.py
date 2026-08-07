"""PARSEC-compatible cubic splines for padded radial pseudopotential data.

PARSEC does not construct its optional pseudopotential splines from only the
positive POTRE radii.  If ``M = Expansion_Order/2``, it prepends the artificial
knots

``[-M-1, -M, ..., -1, 0]``

and continues the first radial value across those knots.  It then uses the
clamped Numerical Recipes cubic spline with zero first derivative at the two
ends of this enlarged table.  The padding makes the spline regular through the
atomic origin while retaining the original positive logarithmic radial grid.

This module is a direct NumPy translation of PARSEC's ``spline``/``splint``
recurrences.  Keeping it separate makes the same construction reusable for
local potentials, core densities, and Kleinman-Bylander radial projectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clamped_second_derivatives(
    knots: np.ndarray,
    values: np.ndarray,
    left_slope: float = 0.0,
    right_slope: float = 0.0,
) -> np.ndarray:
    """Return Numerical Recipes spline second derivatives.

    This is the tridiagonal forward-elimination/back-substitution recurrence
    used by PARSEC's ``spline.f90`` when finite endpoint derivatives are
    supplied.  Pseudopotential callers use ``left_slope=right_slope=0``.
    """
    point_count = knots.size
    second = np.zeros(point_count, dtype=float)
    work = np.zeros(point_count, dtype=float)

    first_step = knots[1] - knots[0]
    second[0] = -0.5
    work[0] = (3.0 / first_step) * (
        (values[1] - values[0]) / first_step - left_slope
    )

    for index in range(1, point_count - 1):
        span = knots[index + 1] - knots[index - 1]
        sigma = (knots[index] - knots[index - 1]) / span
        pivot = sigma * second[index - 1] + 2.0
        second[index] = (sigma - 1.0) / pivot
        right_difference = (
            (values[index + 1] - values[index])
            / (knots[index + 1] - knots[index])
            - (values[index] - values[index - 1])
            / (knots[index] - knots[index - 1])
        )
        work[index] = (
            6.0 * right_difference / span - sigma * work[index - 1]
        ) / pivot

    last_step = knots[-1] - knots[-2]
    right_factor = 0.5
    right_value = (3.0 / last_step) * (
        right_slope - (values[-1] - values[-2]) / last_step
    )
    second[-1] = (
        right_value - right_factor * work[-2]
    ) / (right_factor * second[-2] + 1.0)

    for index in range(point_count - 2, -1, -1):
        second[index] = second[index] * second[index + 1] + work[index]
    return second


@dataclass(frozen=True)
class ParsecRadialSpline:
    """Clamped cubic spline over PARSEC's origin-padded radial table."""

    knots: np.ndarray
    values: np.ndarray
    second_derivatives: np.ndarray

    @classmethod
    def from_positive_grid(
        cls,
        radii: np.ndarray,
        values: np.ndarray,
        padding_width: int,
        *,
        origin_value: float | None = None,
    ) -> "ParsecRadialSpline":
        """Construct the padded spline used by PARSEC pseudopotentials.

        ``radii`` are the strictly positive POTRE samples corresponding to
        Fortran indices ``2:npoint``.  ``padding_width`` is PARSEC's
        ``norder = Expansion_Order/2``.  Fortran indices ``-norder:1`` become
        the artificial coordinates ``-norder-1,...,0``.

        By default the first positive-grid value is continued across the
        padding.  ``origin_value`` supports radial quantities for which PARSEC
        uses a different inner continuation.
        """
        radial_grid = np.asarray(radii, dtype=float)
        radial_values = np.asarray(values, dtype=float)
        if radial_grid.ndim != 1 or radial_values.shape != radial_grid.shape:
            raise ValueError(
                "radii and values must be one-dimensional arrays of equal length"
            )
        if radial_grid.size < 2:
            raise ValueError("at least two positive radial samples are required")
        if not np.all(np.isfinite(radial_grid)) or not np.all(
            np.isfinite(radial_values)
        ):
            raise ValueError("radial spline data must be finite")
        if radial_grid[0] <= 0.0 or np.any(np.diff(radial_grid) <= 0.0):
            raise ValueError("radial spline radii must be strictly increasing and positive")

        width = int(padding_width)
        if width != padding_width or width < 1:
            raise ValueError("padding_width must be a positive integer")
        inner_value = (
            radial_values[0] if origin_value is None else float(origin_value)
        )
        if not np.isfinite(inner_value):
            raise ValueError("origin_value must be finite")

        artificial_radii = np.arange(-width - 1, 1, dtype=float)
        knots = np.concatenate((artificial_radii, radial_grid))
        padded_values = np.concatenate(
            (np.full(artificial_radii.size, inner_value, dtype=float), radial_values)
        )
        second = _clamped_second_derivatives(knots, padded_values)
        return cls(knots=knots, values=padded_values, second_derivatives=second)

    def __call__(self, points: np.ndarray | float) -> np.ndarray | float:
        """Evaluate with PARSEC's ``splint`` cubic formula.

        PARSEC's evaluator uses the first or last interval for values outside
        the knot range.  Physical callers in this package evaluate only
        nonnegative radii below their explicit radial cutoff, but the same
        extrapolation behavior is retained here for source compatibility.
        """
        queries = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(queries)):
            raise ValueError("radial spline evaluation points must be finite")
        flat = queries.reshape(-1)

        lower = np.searchsorted(self.knots, flat, side="right") - 1
        lower = np.clip(lower, 0, self.knots.size - 2)
        upper = lower + 1
        step = self.knots[upper] - self.knots[lower]
        left_weight = (self.knots[upper] - flat) / step
        right_weight = (flat - self.knots[lower]) / step
        result = (
            left_weight * self.values[lower]
            + right_weight * self.values[upper]
            + (
                (left_weight**3 - left_weight) * self.second_derivatives[lower]
                + (right_weight**3 - right_weight) * self.second_derivatives[upper]
            )
            * step**2
            / 6.0
        )
        shaped = result.reshape(queries.shape)
        if queries.ndim == 0:
            return float(shaped)
        return shaped


__all__ = ["ParsecRadialSpline"]
