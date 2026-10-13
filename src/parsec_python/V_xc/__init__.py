"""Exchange-correlation functionals for the native Python port."""

from .b3lyp import (
    B3LYPSemilocalEnergies,
    b3lyp_exchange_correlation_energy,
    b3lyp_semilocal_energies,
    b3lyp_spin_semilocal_energies,
)
from .ca_lda import XCResult, ca_lda
from .ca_lda_spin import SpinXCResult, ca_lda_spin
from .exact_exchange import (
    ExactExchangeResult,
    exact_exchange_energy,
    reference_pair_poisson,
    spin_exact_exchange_energy,
)
from .pbe import first_derivative_coefficients, pbe, pbe_energy_partials
from .pbe_spin import pbe_spin, pbe_spin_energy_partials, pw92_spin_correlation

__all__ = [
    "B3LYPSemilocalEnergies",
    "ExactExchangeResult",
    "SpinXCResult",
    "XCResult",
    "b3lyp_exchange_correlation_energy",
    "b3lyp_semilocal_energies",
    "b3lyp_spin_semilocal_energies",
    "ca_lda",
    "ca_lda_spin",
    "exact_exchange_energy",
    "first_derivative_coefficients",
    "pbe",
    "pbe_energy_partials",
    "pbe_spin",
    "pbe_spin_energy_partials",
    "pw92_spin_correlation",
    "reference_pair_poisson",
    "spin_exact_exchange_energy",
]
