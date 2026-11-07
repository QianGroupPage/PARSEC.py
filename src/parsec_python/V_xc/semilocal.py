"""A pluggable semilocal exchange-correlation evaluator and its potential on the grid.

A *semilocal evaluator* is any object with the three members of
:class:`SemilocalXC`.  It works pointwise, in Hartree atomic units, with the
variables and derivatives of libxc:

* ``unpolarized(rho, sigma) -> (e, vrho, vsigma)`` for the total density
  ``rho`` and ``sigma = |grad rho|**2``;
* ``polarized(rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd)
  -> (e, vrho_up, vrho_down, vsigma_uu, vsigma_ud, vsigma_dd)`` with
  ``sigma_ab = grad rho_a . grad rho_b``;
* ``family``: ``"none"`` (no semilocal part, e.g. Hartree-Fock), ``"lda"`` or
  ``"gga"``.

``e`` is the energy per *volume* (libxc's ``zk`` times the density), ``vrho``
and ``vsigma`` are ``de/drho`` and ``de/dsigma`` at fixed other variables.
An LDA evaluator returns ``vsigma = 0``.  Meta-GGA evaluators (``vtau``,
``vlapl``) do not exist in this code yet; they would add arguments and
returns to the same two methods.  The evaluators of this package are
:class:`parsec_python.V_xc.semilocal_components.ComponentSemilocal`; an
evaluator backed by libxc can replace them without changing any caller.

:func:`semilocal_xc` and :func:`semilocal_xc_spin` turn an evaluator into the
results the SCF loops take (:class:`~parsec_python.V_xc.ca_lda.XCResult`,
:class:`~parsec_python.V_xc.ca_lda_spin.SpinXCResult`, Rydberg).  The energy
is the discrete sum ``E = h**3 sum_i e(rho_i, sigma_i)`` with
``sigma = |D rho|**2`` of the centered first-derivative stencil of the run
(:func:`parsec_python.V_xc.pbe.density_gradient`), and the potentials are its
exact derivatives,

``v = vrho + D.T (2 vsigma D rho)`` (unpolarized),
``v_a = vrho_a + D.T (2 vsigma_aa D rho_a + vsigma_ab D rho_b)`` (spin),

the construction of the PBE of :mod:`parsec_python.V_xc.pbe`.  A frozen
nonlinear-core density is added to the valence density (unpolarized) or
split evenly between the channels (spin), as for every functional of the code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .ca_lda import XCResult
from .ca_lda_spin import SpinXCResult, spin_densities_with_core
from .pbe import _DENSITY_THRESHOLD, _gga_density, discrete_gga, discrete_spin_gga_ab


#: The families an evaluator may report.
SEMILOCAL_FAMILIES = ("none", "lda", "gga")


@runtime_checkable
class SemilocalXC(Protocol):
    """The pointwise interface of a semilocal functional (see the module docstring)."""

    @property
    def family(self) -> str: ...

    def unpolarized(
        self, rho: np.ndarray, sigma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def polarized(
        self,
        rho_up: np.ndarray,
        rho_down: np.ndarray,
        sigma_uu: np.ndarray,
        sigma_ud: np.ndarray,
        sigma_dd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def _zero_result(density: np.ndarray) -> XCResult:
    zero = np.zeros_like(density)
    return XCResult(
        potential=zero.copy(),
        energy_per_electron=zero.copy(),
        energy_density=zero,
        total_energy=0.0,
    )


def semilocal_xc(
    evaluator: SemilocalXC,
    valence_density: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    *,
    name: str = "the semilocal functional",
) -> XCResult:
    """Unpolarized semilocal energy and exact discrete potential (Rydberg).

    ``evaluator.unpolarized`` is evaluated at ``rho = valence + core`` and
    ``sigma = |D rho|**2``; ``name`` appears in the error of a negative density.
    """

    density = _gga_density(valence_density, core_density, grid, name)
    if evaluator.family == "none":
        return _zero_result(density)
    energy_hartree, potential_hartree = discrete_gga(density, grid, evaluator.unpolarized)
    # Hartree -> Rydberg at the public boundary, as in pbe.py.
    energy_density = 2.0 * energy_hartree
    epsilon = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    epsilon[positive] = energy_density[positive] / density[positive]
    return XCResult(
        potential=2.0 * potential_hartree,
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
    )


def semilocal_xc_spin(
    evaluator: SemilocalXC,
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    *,
    name: str = "the semilocal functional",
) -> SpinXCResult:
    """Spin-polarized semilocal energy and exact discrete potentials (Rydberg).

    Each channel sees its valence density plus half of the frozen core
    density; ``evaluator.polarized`` gets ``sigma_ab = D rho_a . D rho_b``.
    """

    up, down = spin_densities_with_core(valence_up, valence_down, core_density, functional=name)
    if up.shape != (grid.size,):
        raise ValueError(f"the densities of {name} do not match the real-space grid")
    if evaluator.family == "none":
        zero = np.zeros_like(up)
        return SpinXCResult(
            potential_up=zero.copy(),
            potential_down=zero.copy(),
            energy_density=zero,
            total_energy=0.0,
        )
    result = discrete_spin_gga_ab(up, down, grid, evaluator.polarized)
    return SpinXCResult(
        potential_up=result.potential_up,
        potential_down=result.potential_down,
        energy_density=result.energy_density,
        total_energy=result.total_energy,
    )


__all__ = [
    "SEMILOCAL_FAMILIES",
    "SemilocalXC",
    "semilocal_xc",
    "semilocal_xc_spin",
]
