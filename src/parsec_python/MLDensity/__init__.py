"""Optional machine-learned initial densities for isolated PARSEC jobs.

The adapters deliberately sit outside the SCF implementation.  They return a
plain volume-density vector on the already constructed :class:`RealSpaceGrid`,
so every downstream Hartree, XC, Hamiltonian, and convergence algorithm is the
same as for the default superposition-of-atoms (SAD) initial guess.
"""

from .field import (
    BOHR_TO_ANGSTROM,
    DensityLoadResult,
    load_density_for_grid,
    normalize_density_units,
    save_point_density,
)
from .providers import build_initial_density

__all__ = [
    "BOHR_TO_ANGSTROM",
    "DensityLoadResult",
    "build_initial_density",
    "load_density_for_grid",
    "normalize_density_units",
    "save_point_density",
]
