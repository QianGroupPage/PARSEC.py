"""
Finite-difference derivatives on a grid whose axes follow oblique lattice vectors.

A periodic cell with lattice vectors ``a_i`` is sampled at ``u_1 s_1 + u_2 s_2 + u_3 s_3`` with the integer
indices ``u`` and the steps ``s_i = a_i/n_i`` (the rows of ``S``).  Writing the Cartesian point as
``r = S^T u``, the chain rule gives

    d/dr_k = sum_i C[k, i] d/du_i,       C = S^{-1}

    nabla^2 = sum_i G_ii d^2/du_i^2 + sum_{i<j} 2 G_ij d^2/du_i du_j,       G = (S S^T)^{-1}

so an oblique cell has mixed second derivatives.  Following Natan et al. (the Fortran PARSEC's
``pbc_grid_coefs``), the mixed terms are written as pure second derivatives along a few extra lattice
directions ``d`` (for example ``u_1 + u_2``): since

    d^2/dt^2 f(u + t d) = sum_i d_i^2 d^2/du_i^2 + sum_{i<j} 2 d_i d_j d^2/du_i du_j,

weights ``b_k`` on directions ``d_k`` that satisfy ``sum_k b_k d_ki d_kj = G_ij`` for every pair reproduce
the mixed terms, and the pure coefficients become ``G_ii - sum_k b_k d_ki^2``.  The directions are the
shortest lattice vectors (best accuracy), as in the Fortran code: one diagonal ``u_i +- u_j`` when a
single pair of axes is oblique, the three shortest independent neighbors otherwise.  The result is exact
for any polynomial the axial stencil is exact for, whatever the (possibly unequal) steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

_COUPLING_TOLERANCE = 1.0e-8  # |G_ij|/sqrt(G_ii G_jj) below this: the axes are orthogonal
_WEIGHT_TOLERANCE = 1.0e-7  # relative weights below this are dropped
_DETERMINANT_TOLERANCE = 1.0e-9

_PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class SecondDerivativePlan:
    """``nabla^2 = sum_i axial[i] d^2/du_i^2 + sum_k weight_k d^2/dt^2 (along direction_k)`` in index units."""

    axial: np.ndarray
    directions: tuple[tuple[int, int, int], ...]
    weights: tuple[float, ...]


def inverse_metric(step_vectors: np.ndarray) -> np.ndarray:
    """``G = (S S^T)^{-1}``: the coefficients of the Laplacian in the index coordinates."""
    steps = np.asarray(step_vectors, dtype=np.float64)
    return np.linalg.inv(steps @ steps.T)


def cartesian_derivative_matrix(step_vectors: np.ndarray) -> np.ndarray:
    """``C = S^{-1}``: ``d/dr_k = sum_i C[k, i] d/du_i``."""
    return np.linalg.inv(np.asarray(step_vectors, dtype=np.float64))


def _pair_matrix(directions: list[np.ndarray]) -> np.ndarray:
    return np.array([[d[i] * d[j] for d in directions] for i, j in _PAIRS], dtype=np.float64)


def second_derivative_plan(step_vectors: np.ndarray) -> SecondDerivativePlan:
    """Pure-second-derivative form of the Laplacian on the grid with these step vectors."""
    steps = np.asarray(step_vectors, dtype=np.float64)
    metric = inverse_metric(steps)
    diagonal = np.diag(metric).copy()
    coupling = np.array([metric[i, j] / np.sqrt(diagonal[i] * diagonal[j]) for i, j in _PAIRS])
    oblique = [pair for pair, value in zip(_PAIRS, coupling) if abs(value) > _COUPLING_TOLERANCE]
    if not oblique:
        return SecondDerivativePlan(diagonal, (), ())

    if len(oblique) == 1:
        i, j = oblique[0]
        sign = 1 if metric[i, j] > 0.0 else -1
        direction = np.zeros(3, dtype=int)
        direction[i], direction[j] = 1, sign
        weight = abs(float(metric[i, j]))
        axial = diagonal.copy()
        axial[i] -= weight
        axial[j] -= weight
        return SecondDerivativePlan(axial, (tuple(int(v) for v in direction),), (weight,))

    # Several oblique pairs: the three shortest independent neighbors that are not grid axes.
    candidates = [np.array(d, dtype=int) for d in product((-1, 0, 1), repeat=3) if sum(abs(v) for v in d) > 1]
    lengths = [float(np.linalg.norm(d @ steps)) for d in candidates]
    order = sorted(range(len(candidates)), key=lambda index: (round(lengths[index], 9), index))
    chosen: list[np.ndarray] = []
    for index in order:
        candidate = candidates[index]
        if any(np.array_equal(candidate, -other) for other in chosen):
            continue
        trial = chosen + [candidate]
        if len(trial) == 3 and abs(np.linalg.det(_pair_matrix(trial))) < _DETERMINANT_TOLERANCE:
            continue
        chosen = trial
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        raise ValueError("no set of three independent lattice neighbors represents the cell's mixed derivatives")
    matrix = _pair_matrix(chosen)
    weights = np.linalg.solve(matrix, np.array([metric[i, j] for i, j in _PAIRS]))
    axial = diagonal.copy()
    for direction, weight in zip(chosen, weights):
        axial -= weight * direction.astype(float) ** 2
    keep = np.abs(weights) > _WEIGHT_TOLERANCE * float(np.max(diagonal))
    return SecondDerivativePlan(
        axial,
        tuple(tuple(int(v) for v in d) for d, k in zip(chosen, keep) if k),
        tuple(float(w) for w, k in zip(weights, keep) if k),
    )


__all__ = [
    "SecondDerivativePlan",
    "cartesian_derivative_matrix",
    "inverse_metric",
    "second_derivative_plan",
]
