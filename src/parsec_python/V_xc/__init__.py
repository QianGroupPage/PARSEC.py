"""Exchange-correlation functionals for the native Python port."""

from .b3lyp import (
    B3LYPSemilocalEnergies,
    b3lyp_exchange_correlation_energy,
    b3lyp_semilocal_energies,
    b3lyp_spin_semilocal_energies,
)
from .ca_lda import SpinPolarizedXCResult, XCResult, ca_lda, ca_lda_spin_polarized
from .ca_lda_spin import SpinXCResult, ca_lda_spin
from .exact_exchange import (
    ExactExchangeResult,
    exact_exchange_energy,
    reference_pair_poisson,
    spin_exact_exchange_energy,
)
from .noncollinear import NoncollinearXCResult, local_spin_densities, noncollinear_xc
from .pbe import (
    first_derivative_coefficients,
    pbe,
    pbe_energy_partials,
    pbe_spin_polarized,
    pbe_spin_polarized_partials,
)
from .pbe_spin import pbe_spin, pbe_spin_energy_partials, pw92_spin_correlation

__all__ = [
    "B3LYPSemilocalEnergies",
    "ExactExchangeResult",
    "NoncollinearXCResult",
    "SpinPolarizedXCResult",
    "SpinXCResult",
    "XCResult",
    "b3lyp_exchange_correlation_energy",
    "b3lyp_semilocal_energies",
    "b3lyp_spin_semilocal_energies",
    "ca_lda",
    "ca_lda_spin",
    "ca_lda_spin_polarized",
    "exact_exchange_energy",
    "first_derivative_coefficients",
    "local_spin_densities",
    "noncollinear_xc",
    "pbe",
    "pbe_energy_partials",
    "pbe_spin",
    "pbe_spin_energy_partials",
    "pbe_spin_polarized",
    "pbe_spin_polarized_partials",
    "pw92_spin_correlation",
    "reference_pair_poisson",
    "spin_exact_exchange_energy",
]
