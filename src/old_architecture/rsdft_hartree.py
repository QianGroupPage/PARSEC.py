"""Hartree-potential setup helpers.

The legacy solver carries the Hartree potential as a split

    V_H[rho] ~= hpot0 + Hpot,

where ``hpot0`` comes from the initial-density builder and ``Hpot`` is solved
from density differences during SCF.  This module centralizes the selectable
Hartree policies so the solver can keep the numerical flow readable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


HARTREE_METHOD_DESCRIPTIONS = {
    "split": "legacy split Hartree: setup hpot0 plus Poisson solve for rho-rho0",
    "consistent_split": "split Hartree with hpot0 recomputed by the active Poisson solver",
    "full": "full Hartree solve from the current density at every SCF iteration",
}

_HARTREE_METHOD_ALIASES = {
    "legacy": "split",
    "legacy_split": "split",
    "delta": "split",
    "difference": "split",
    "diff": "split",
    "consistent": "consistent_split",
    "recompute_initial": "consistent_split",
    "recomputed_initial": "consistent_split",
    "poisson_initial": "consistent_split",
    "full_density": "full",
    "full_rho": "full",
    "direct": "full",
}


def normalize_hartree_method(value: Any) -> str:
    """Normalize user-facing Hartree method names."""
    if value is None:
        return "split"

    method = str(value).strip().lower().replace("-", "_")
    method = _HARTREE_METHOD_ALIASES.get(method, method)
    if method not in HARTREE_METHOD_DESCRIPTIONS:
        valid = ", ".join(sorted(HARTREE_METHOD_DESCRIPTIONS))
        raise ValueError(f"invalid Hartree method {value!r}; choose one of: {valid}")
    return method


def describe_hartree_method(method: str) -> str:
    """Return a short description for a normalized Hartree method."""
    return HARTREE_METHOD_DESCRIPTIONS[normalize_hartree_method(method)]


def hartree_rhs_from_grid_density(density_grid, h: float, backend):
    """Build the Poisson RHS from a density stored as electrons per grid point."""
    xp = backend.array_module
    return (4.0 * np.pi / h**3) * xp.asarray(density_grid).reshape(-1)


def solve_hartree_from_grid_density(
    laplacian,
    density_grid,
    h: float,
    backend,
    initial_guess=None,
    max_iters: int = 200,
    tol: float = 1.0e-5,
):
    """Solve ``A * V_H = 4*pi*rho`` for density stored as electrons/grid point."""
    xp = backend.array_module
    density_grid = xp.asarray(density_grid).reshape(-1)
    if initial_guess is None:
        initial_guess = xp.zeros_like(density_grid)
    else:
        initial_guess = xp.asarray(initial_guess).reshape(-1)

    rhs = hartree_rhs_from_grid_density(density_grid, h, backend)
    potential, iterations = backend.pcg(laplacian, rhs, initial_guess, max_iters, tol)
    return xp.asarray(potential).reshape(-1), int(iterations)
