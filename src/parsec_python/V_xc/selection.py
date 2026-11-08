"""Evaluate the exchange-correlation functional an SCF setting names.

``SCFSettings.xc_functional`` names one of the ``Correlation_Type`` values of
PARSEC (``usrinputfile.F90``:2308-2361):

=====================  =========  ==========================================
``xc_functional``      Fortran    functional
=====================  =========  ==========================================
``ca``                 ``ca``     CA/PZ LDA (:mod:`.ca_lda`, :mod:`.ca_lda_spin`)
``pw92``               ``pl``     PW92 LDA (:mod:`.pw92_lda`)
``xalpha``             ``xa``     Slater X-alpha, alpha = 1 (:mod:`.local_density`)
``wigner``             ``wi``     Wigner (:mod:`.local_density`)
``hedin_lundqvist``    ``hl``     Hedin-Lundqvist (:mod:`.local_density`)
``pbe``                ``pb``     PBE GGA (:mod:`.pbe`, :mod:`.pbe_spin`)
``pw91``               ``pw``     PW91 GGA (:mod:`.pw91`)
``lb94``               ``lb``     CA with the LB94 potential (:mod:`.asymptotic_correction`)
``casida_salahub``     ``cs``     CA with the Casida-Salahub potential
=====================  =========  ==========================================

A libxc functional (``libxc:<name>[+<name>...]``, :mod:`.libxc`, the
optional library) of family LDA or GGA is evaluated by
:func:`.semilocal.semilocal_xc` (``semilocal_xc_spin``) with its
:class:`~.libxc.LibxcSemilocal`: the discrete variational potential and the
frozen core density of every functional of this code.

The X-alpha, Wigner and Hedin-Lundqvist functionals have no spin-polarized
form (neither in Fortran).  The CA and PBE branches of the SCF systems call
their functions directly; this module adds the others.
"""

from __future__ import annotations

import numpy as np

from .asymptotic_correction import lb94, lb94_spin
from .ca_lda import SpinPolarizedXCResult, XCResult, ca_lda
from .ca_lda_spin import SpinXCResult
from .local_density import hedin_lundqvist_lda, wigner_lda, xalpha_lda
from .pbe import pbe
from .pw91 import pw91, pw91_spin
from .pw92_lda import pw92_lda, pw92_lda_spin
from .libxc import is_libxc_functional
from .semilocal import semilocal_xc, semilocal_xc_spin


#: ``xc_functional`` -> the two-letter ``icorr`` code of Fortran PARSEC.
FORTRAN_CORRELATION_CODES = {
    "ca": "ca",
    "pw92": "pl",
    "xalpha": "xa",
    "wigner": "wi",
    "hedin_lundqvist": "hl",
    "pbe": "pb",
    "pw91": "pw",
    "lb94": "lb",
    "casida_salahub": "cs",
}
#: The description line ``usrinputfile.F90`` prints for each functional.
FORTRAN_CORRELATION_DESCRIPTIONS = {
    "ca": "LDA, Ceperley-Alder, Perdew-Zunger parametrization",
    "pw92": "LDA, Perdew-Wang parametrization",
    "xalpha": "LDA, Slater x-alpha",
    "wigner": "LDA, Wigner interpolation",
    "hedin_lundqvist": "LDA, Hedin-Lundqvist",
    "pbe": "GGA, Perdew-Burke-Ernzerhof parametrization",
    "pw91": "GGA, Perdew-Wang parametrization",
    "lb94": "ACLDA, Leeuwen-Baerends correction",
    "casida_salahub": "ACLDA, Casida-Salahub correction",
}
#: Short names of the functionals in reports.
XC_DISPLAY_NAMES = {
    "ca": "CA-LDA",
    "pw92": "PW92-LDA",
    "xalpha": "X-alpha",
    "wigner": "Wigner-LDA",
    "hedin_lundqvist": "Hedin-Lundqvist-LDA",
    "pbe": "PBE",
    "pw91": "PW91",
    "lb94": "LB94",
    "casida_salahub": "Casida-Salahub",
}
XC_FUNCTIONALS = tuple(FORTRAN_CORRELATION_CODES)
#: Functionals without a spin-polarized form.
UNPOLARIZED_ONLY_FUNCTIONALS = frozenset({"xalpha", "wigner", "hedin_lundqvist"})
#: Functionals whose potential is not the derivative of their (CA) energy.
POTENTIAL_ONLY_CORRECTIONS = frozenset({"lb94", "casida_salahub"})


def _libxc_density_functional(functional: str):
    """The evaluator of a libxc LDA/GGA (meta-GGAs and hybrids take other routes, refused here)."""

    from .libxc import libxc_semilocal, parse_libxc_spec

    spec = parse_libxc_spec(functional)
    if spec.is_hybrid:
        raise ValueError(f"{functional} is a hybrid: its semilocal part belongs to the Fock loop (SCF.hybrid)")
    if spec.is_meta_gga:
        raise NotImplementedError(
            f"{functional} is a meta-GGA: it needs the kinetic-energy density of the orbitals and the "
            "meta-GGA Hamiltonian, which this evaluation does not have"
        )
    return libxc_semilocal(functional)


def _cap(functional: str, ionization_energy_difference: float) -> float | None:
    return float(ionization_energy_difference) if functional == "casida_salahub" else None


def evaluate_unpolarized_xc(
    functional: str,
    valence_density: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    *,
    ionization_energy_difference: float = 0.0,
) -> XCResult:
    """The unpolarized functional ``functional`` at ``rho_valence + rho_core``.

    ``grid`` is an isolated or a periodic real-space grid; the gradient
    functionals zero-pad the first or wrap around the second.
    ``ionization_energy_difference`` (Rydberg) is read by
    ``casida_salahub`` only.
    """

    if is_libxc_functional(functional):
        return semilocal_xc(
            _libxc_density_functional(functional), valence_density, grid, core_density, name=functional
        )
    volume_element = grid.volume_element
    if functional == "ca":
        return ca_lda(valence_density, volume_element, core_density)
    if functional == "pbe":
        return pbe(valence_density, grid, core_density)
    if functional == "pw92":
        return pw92_lda(valence_density, volume_element, core_density)
    if functional == "xalpha":
        return xalpha_lda(valence_density, volume_element, core_density)
    if functional == "wigner":
        return wigner_lda(valence_density, volume_element, core_density)
    if functional == "hedin_lundqvist":
        return hedin_lundqvist_lda(valence_density, volume_element, core_density)
    if functional == "pw91":
        return pw91(valence_density, grid, core_density)
    if functional in POTENTIAL_ONLY_CORRECTIONS:
        return lb94(
            valence_density,
            grid,
            core_density,
            _cap(functional, ionization_energy_difference),
        )
    raise ValueError(f"unsupported XC functional {functional!r}")


def evaluate_spin_xc(
    functional: str,
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    *,
    ionization_energy_difference: float = 0.0,
) -> SpinXCResult:
    """The spin-polarized form of ``pw92``, ``pw91``, ``lb94`` or ``casida_salahub``.

    CA and PBE have two spin implementations each, which the SCF systems
    select themselves; the functionals without a spin form are refused.
    """

    if is_libxc_functional(functional):
        return semilocal_xc_spin(
            _libxc_density_functional(functional), valence_up, valence_down, grid, core_density, name=functional
        )
    if functional == "pw92":
        return pw92_lda_spin(valence_up, valence_down, grid.volume_element, core_density)
    if functional == "pw91":
        return pw91_spin(valence_up, valence_down, grid, core_density)
    if functional in POTENTIAL_ONLY_CORRECTIONS:
        return lb94_spin(
            valence_up,
            valence_down,
            grid,
            core_density,
            _cap(functional, ionization_energy_difference),
        )
    if functional in UNPOLARIZED_ONLY_FUNCTIONALS:
        raise ValueError(
            f"the {functional} functional has no spin-polarized form "
            "(neither has it in Fortran PARSEC's exc_spn.f90)"
        )
    raise ValueError(f"unsupported spin-polarized XC functional {functional!r}")


def as_spin_polarized_result(result: SpinXCResult) -> SpinPolarizedXCResult:
    """The same fields in the result type of the periodic and spin-orbit paths."""

    return SpinPolarizedXCResult(
        potential_up=result.potential_up,
        potential_down=result.potential_down,
        energy_density=result.energy_density,
        total_energy=result.total_energy,
    )


__all__ = [
    "FORTRAN_CORRELATION_CODES",
    "FORTRAN_CORRELATION_DESCRIPTIONS",
    "POTENTIAL_ONLY_CORRECTIONS",
    "UNPOLARIZED_ONLY_FUNCTIONALS",
    "XC_DISPLAY_NAMES",
    "XC_FUNCTIONALS",
    "as_spin_polarized_result",
    "evaluate_spin_xc",
    "evaluate_unpolarized_xc",
]
