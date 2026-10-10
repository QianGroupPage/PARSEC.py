"""Allocation-bounded recurrence for isolated Hartree multipoles.

The reference implementation evaluates SciPy spherical harmonics separately
for every ``(l,m)`` and every density update.  On molecular grids that special
function work can dominate the Poisson solve.  This module evaluates the same
normalized complex harmonics through associated-Legendre recurrences using a
few length-N work arrays; it neither stores a dense boundary map nor changes
the multipole truncation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.Hartree import MultipoleExpansion
from parsec_python.Hartree.harmonics import _positive_m_harmonic_rows


@dataclass(frozen=True)
class FastMultipoleExpansion(MultipoleExpansion):
    """Reference-compatible expansion with recurrence-based evaluation."""

    def potential(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("boundary points must have shape (n, 3)")
        radius = np.linalg.norm(points, axis=1)
        if np.any(radius <= 0.0):
            raise ValueError("multipole boundary potential is undefined at the origin")

        result = np.zeros(points.shape[0], dtype=np.complex128)
        for angular_momentum, magnetic, harmonic, _ in _positive_m_harmonic_rows(
            points, self.order
        ):
            factor = (
                4.0
                * np.pi
                / (2 * angular_momentum + 1)
                * radius ** (-(angular_momentum + 1))
            )
            result += (
                factor
                * self.moments[(angular_momentum, magnetic)]
                * harmonic
            )
            if magnetic:
                negative_harmonic = ((-1) ** magnetic) * np.conjugate(harmonic)
                result += (
                    factor
                    * self.moments[(angular_momentum, -magnetic)]
                    * negative_harmonic
                )
        return 2.0 * result.real

    __call__ = potential


def density_multipoles_fast(
    density: np.ndarray,
    grid: RealSpaceGrid,
    order: int = 9,
) -> FastMultipoleExpansion:
    """Compute exactly the reference ``Q_lm`` moments without special calls."""

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    if order < 0:
        raise ValueError("multipole order cannot be negative")

    weighted_density = density * grid.volume_element
    moments: dict[tuple[int, int], complex] = {}
    for angular_momentum, magnetic, harmonic, radius in _positive_m_harmonic_rows(
        grid.coordinates, order
    ):
        moment = complex(
            np.sum(
                weighted_density
                * radius**angular_momentum
                * np.conjugate(harmonic)
            )
        )
        moments[(angular_momentum, magnetic)] = moment
        if magnetic:
            moments[(angular_momentum, -magnetic)] = (
                ((-1) ** magnetic) * np.conjugate(moment)
            )
    return FastMultipoleExpansion(order=order, moments=moments)


__all__ = ["FastMultipoleExpansion", "density_multipoles_fast"]
