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
from .asymptotic_correction import (
    lb94,
    lb94_potential_correction,
    lb94_spin,
    lb94_spin_potential_corrections,
)
from .hybrid import (
    HYBRID_FUNCTIONALS,
    ExactExchangeTerm,
    HybridFunctional,
    hybrid_functional,
    hybrid_semilocal_xc,
    hybrid_semilocal_xc_spin,
    is_hybrid_label,
)
from .local_density import hedin_lundqvist_lda, wigner_lda, xalpha_lda
from .noncollinear import NoncollinearXCResult, local_spin_densities, noncollinear_xc
from .pbe import (
    density_gradient,
    discrete_gga,
    first_derivative_coefficients,
    pbe,
    pbe_energy_partials,
    pbe_spin_polarized,
    pbe_spin_polarized_partials,
)
from .pbe_spin import (
    discrete_spin_gga,
    pbe_spin,
    pbe_spin_energy_partials,
    pw92_spin_correlation,
)
from .pw91 import pw91, pw91_energy_partials, pw91_spin, pw91_spin_energy_partials
from .pw92_lda import pw92_correlation, pw92_lda, pw92_lda_spin, pw92_lsd_correlation
from .selection import (
    FORTRAN_CORRELATION_CODES,
    FORTRAN_CORRELATION_DESCRIPTIONS,
    UNPOLARIZED_ONLY_FUNCTIONALS,
    XC_FUNCTIONALS,
    evaluate_spin_xc,
    evaluate_unpolarized_xc,
)

__all__ = [
    "B3LYPSemilocalEnergies",
    "ExactExchangeResult",
    "ExactExchangeTerm",
    "HYBRID_FUNCTIONALS",
    "HybridFunctional",
    "FORTRAN_CORRELATION_CODES",
    "FORTRAN_CORRELATION_DESCRIPTIONS",
    "NoncollinearXCResult",
    "SpinPolarizedXCResult",
    "SpinXCResult",
    "UNPOLARIZED_ONLY_FUNCTIONALS",
    "XCResult",
    "XC_FUNCTIONALS",
    "b3lyp_exchange_correlation_energy",
    "b3lyp_semilocal_energies",
    "b3lyp_spin_semilocal_energies",
    "ca_lda",
    "ca_lda_spin",
    "ca_lda_spin_polarized",
    "density_gradient",
    "discrete_gga",
    "discrete_spin_gga",
    "evaluate_spin_xc",
    "evaluate_unpolarized_xc",
    "exact_exchange_energy",
    "first_derivative_coefficients",
    "hedin_lundqvist_lda",
    "hybrid_functional",
    "hybrid_semilocal_xc",
    "hybrid_semilocal_xc_spin",
    "is_hybrid_label",
    "lb94",
    "lb94_potential_correction",
    "lb94_spin",
    "lb94_spin_potential_corrections",
    "local_spin_densities",
    "noncollinear_xc",
    "pbe",
    "pbe_energy_partials",
    "pbe_spin",
    "pbe_spin_energy_partials",
    "pbe_spin_polarized",
    "pbe_spin_polarized_partials",
    "pw91",
    "pw91_energy_partials",
    "pw91_spin",
    "pw91_spin_energy_partials",
    "pw92_correlation",
    "pw92_lda",
    "pw92_lda_spin",
    "pw92_lsd_correlation",
    "pw92_spin_correlation",
    "reference_pair_poisson",
    "spin_exact_exchange_energy",
    "wigner_lda",
    "xalpha_lda",
]
