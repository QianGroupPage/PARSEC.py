"""Brillouin-zone weights of the pooled (k-point, state) levels of a k-point SCF.

Fortran PARSEC weights every level of k-point ``k`` by ``kpwt(k)`` in the
Fermi level (``flevel.f90``: ``weight(jj, isp) = kpwt(kplp)``), in the density
(``newrho.F90:115-131``) and in the band energy (``totnrg.f90:85-93``).  The
k-point drivers of this package pool every level into one Fermi-level search
and add every k-point's density.

Equal weights (this port's unreduced Monkhorst-Pack grid, an explicit list
with equal weights) keep the arithmetic the drivers have always used: one
scalar ``degeneracy * weight`` in
:func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations`, ``weight``
times every k-point's density, ``band_energy_weight = weight``.  Unequal
weights (Fortran's irreducible wedge, a weighted ``Kpoint_List``) use the
per-state weights of the Fermi level and of the band energy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..Occupations import fermi_occupations


@dataclass(frozen=True)
class KPointWeights:
    """The normalized weights of a k-point set and how the drivers apply them.

    ``uniform`` holds where every weight is (numerically) the first one,
    ``weight``; the drivers then use ``weight`` everywhere, as before
    unequal weights existed.
    """

    weights: np.ndarray
    uniform: bool
    weight: float

    def of_k(self, k_index: int) -> float:
        """The weight that multiplies the density of k-point ``k_index``."""

        return self.weight if self.uniform else float(self.weights[k_index])

    def state_weights(self, blocks, k_indices) -> np.ndarray:
        """Per-level weights of pooled arrays: ``blocks[i]`` belongs to ``k_indices[i]``."""

        return np.concatenate(
            [
                np.full(np.asarray(block).size, float(self.weights[k]))
                for block, k in zip(blocks, k_indices)
            ]
        )

    def occupy(
        self,
        stacked: np.ndarray,
        electron_count: float,
        temperature_kelvin: float,
        degeneracy: float,
        state_weights: np.ndarray | None = None,
    ):
        """Fermi occupations of the pooled levels ``stacked``.

        ``degeneracy`` is the electron count of one full level at weight one
        (2 without spin, 1 for a spin channel or a spinor).  Returns
        ``(occupations, fermi_level)`` in the order of ``stacked``.
        ``state_weights`` (from :meth:`state_weights`) is needed for unequal
        weights.
        """

        order = np.argsort(stacked, kind="stable")
        if self.uniform:
            result = fermi_occupations(
                stacked[order],
                electron_count,
                temperature_kelvin,
                degeneracy=degeneracy * self.weight,
            )
        else:
            result = fermi_occupations(
                stacked[order],
                electron_count,
                temperature_kelvin,
                degeneracy=degeneracy,
                weights=np.asarray(state_weights, dtype=float)[order],
            )
        occupations = np.empty_like(stacked)
        occupations[order] = result.occupations
        return occupations, result.fermi_level

    def band_energy_terms(self, occupations: np.ndarray, state_weights: np.ndarray | None = None):
        """``(occupations, band_energy_weight)`` for the ``Energy`` functions.

        Equal weights: the occupations and the common weight.  Unequal: the
        occupations times their level's weight, and one.
        """

        if self.uniform:
            return occupations, self.weight
        return np.asarray(occupations) * np.asarray(state_weights, dtype=float), 1.0


def validated_kpoint_weights(k_points, weights, caller: str) -> tuple[np.ndarray, KPointWeights]:
    """Check ``(k_points, weights)`` and return the k-points and their :class:`KPointWeights`.

    ``weights`` must be one non-negative value per k-point summing to one
    (the input reader and :func:`~parsec_python.Grid.kpoints.kpoint_sampling`
    normalize them).
    """

    k_points = np.asarray(k_points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_kpoints = k_points.shape[0] if k_points.ndim == 2 else 0
    if k_points.ndim != 2 or k_points.shape[1] != 3 or weights.shape != (n_kpoints,):
        raise ValueError("k_points must be (n_kpoints, 3) and weights (n_kpoints,)")
    if n_kpoints < 1:
        raise ValueError("at least one k-point is required")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{caller}: k-point weights must be finite and non-negative")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("k-point weights must sum to 1")
    uniform = bool(np.allclose(weights, weights[0]))
    return k_points, KPointWeights(weights=weights, uniform=uniform, weight=float(weights[0]))


__all__ = ["KPointWeights", "validated_kpoint_weights"]
