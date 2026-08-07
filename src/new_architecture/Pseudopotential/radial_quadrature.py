"""Radial integration weights used by PARSEC pseudopotentials."""

from __future__ import annotations

import numpy as np


def parsec_radial_integral(radii: np.ndarray, integrand: np.ndarray) -> float:
    """Integrate tabulated radial data with PARSEC's endpoint convention.

    PARSEC conceptually inserts ``r=0`` with a zero integrand before the
    strictly positive POTRE samples.  For positive samples ``r_i, f_i``, its
    quadrature is

    ``0.5*r_1*f_0
      + sum_(i=1..N-2) 0.5*(r_(i+1)-r_(i-1))*f_i
      + (r_(N-1)-r_(N-2))*f_(N-1)``.

    The last sample receives a full final-interval weight rather than the
    half weight of the ordinary trapezoidal rule.  POTRE projector integrands
    normally vanish there, but retaining this rule gives source-level parity
    and avoids the removed NumPy ``np.trapz`` API.
    """
    radial_grid = np.asarray(radii, dtype=float)
    values = np.asarray(integrand, dtype=float)
    if radial_grid.ndim != 1 or values.shape != radial_grid.shape:
        raise ValueError(
            "radii and integrand must be one-dimensional arrays of equal length"
        )
    if radial_grid.size < 2:
        raise ValueError("at least two positive radial samples are required")
    if not np.all(np.isfinite(radial_grid)) or not np.all(np.isfinite(values)):
        raise ValueError("radial quadrature data must be finite")
    if radial_grid[0] <= 0.0 or np.any(np.diff(radial_grid) <= 0.0):
        raise ValueError("radial quadrature radii must be strictly increasing and positive")

    result = 0.5 * radial_grid[1] * values[0]
    if radial_grid.size > 2:
        weights = 0.5 * (radial_grid[2:] - radial_grid[:-2])
        result += float(weights @ values[1:-1])
    result += (radial_grid[-1] - radial_grid[-2]) * values[-1]
    return float(result)


__all__ = ["parsec_radial_integral"]
