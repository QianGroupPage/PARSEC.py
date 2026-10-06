"""Non-collinear exchange-correlation from a collinear spin functional.

For a spinor density the local spin-density matrix is ``(n + m.sigma)/2`` with
charge density ``n`` and magnetization vector ``m(r)``.  Rotating at every
grid point to the local magnetization axis ``m_hat = m/|m|`` makes the matrix
diagonal with eigenvalues ``n_up,down = (n +/- |m|)/2``, so any collinear
spin-polarized functional (LSDA or PBE) applies unchanged,

``E_xc[n, m] = E_xc^collinear[(n + |m|)/2, (n - |m|)/2]``.

Its functional derivative has the form ``V_avg * 1 + B_xc . sigma`` with

``V_avg = (V_up + V_down)/2``,  ``B_xc = (V_up - V_down)/2 * m_hat``,

which follows from the chain rule through ``|m|`` (``d|m|/dm = m_hat``).  For
a gradient functional the gradients are those of the fields ``n_up`` and
``n_down`` built from ``|m|``, the usual local-rotation (``grad |m|``)
treatment.  ``B_xc`` is zero where ``|m|`` vanishes, which is the limit of the
collinear difference ``V_up - V_down``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ca_lda import SpinPolarizedXCResult

_MAGNETIZATION_FLOOR = 1.0e-12


@dataclass(frozen=True)
class NoncollinearXCResult:
    """``V_xc = potential * 1 + field . sigma`` and the XC energy (Rydberg).

    ``potential`` has shape ``(n_grid,)`` and ``field`` has shape
    ``(3, n_grid)``.
    """

    potential: np.ndarray
    field: np.ndarray
    energy_density: np.ndarray
    total_energy: float


def local_spin_densities(
    density: np.ndarray, magnetization: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(n_up, n_down, |m|)`` along the local magnetization axis.

    ``|m|`` is clipped to ``n`` (a pure-state spinor density satisfies
    ``|m| <= n``; round-off can exceed it by ~1e-16).
    """

    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    if magnetization.shape != (3, density.size):
        raise ValueError("magnetization must have shape (3, n_grid)")
    magnitude = np.sqrt(np.sum(magnetization * magnetization, axis=0))
    magnitude = np.minimum(magnitude, np.maximum(density, 0.0))
    return 0.5 * (density + magnitude), 0.5 * (density - magnitude), magnitude


def noncollinear_xc(
    density: np.ndarray,
    magnetization: np.ndarray,
    spin_functional: Callable[[np.ndarray, np.ndarray], SpinPolarizedXCResult],
) -> NoncollinearXCResult:
    """Evaluate non-collinear XC with a collinear ``spin_functional(n_up, n_down)``."""

    density = np.asarray(density, dtype=float)
    magnetization = np.asarray(magnetization, dtype=float)
    n_up, n_down, magnitude = local_spin_densities(density, magnetization)
    collinear = spin_functional(n_up, n_down)

    average = 0.5 * (collinear.potential_up + collinear.potential_down)
    delta = 0.5 * (collinear.potential_up - collinear.potential_down)
    raw_magnitude = np.sqrt(np.sum(magnetization * magnetization, axis=0))
    usable = raw_magnitude > _MAGNETIZATION_FLOOR
    direction = np.zeros_like(magnetization)
    direction[:, usable] = magnetization[:, usable] / raw_magnitude[usable]
    return NoncollinearXCResult(
        potential=average,
        field=delta[None, :] * direction,
        energy_density=collinear.energy_density,
        total_energy=collinear.total_energy,
    )


__all__ = ["NoncollinearXCResult", "local_spin_densities", "noncollinear_xc"]
