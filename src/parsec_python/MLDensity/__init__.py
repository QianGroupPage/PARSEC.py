"""Optional machine-learned initial densities for isolated PARSEC jobs.

The adapters deliberately sit outside the SCF implementation.  They return a
plain volume-density vector on the already constructed :class:`RealSpaceGrid`,
so every downstream Hartree, XC, Hamiltonian, and convergence algorithm is the
same as for the default superposition-of-atoms (SAD) initial guess.

Models are listed in :mod:`.registry`; :func:`predict_density` runs one at
arbitrary points, and ``ml_density.py`` (:mod:`.cli`) is its command line.
"""

from .box import BoxGrid, box_around_atoms, read_xyz
from .field import (
    BOHR_TO_ANGSTROM,
    DensityLoadResult,
    dense_from_grid,
    load_density_for_grid,
    normalize_density_units,
    save_box_density,
    save_point_density,
    write_cube,
)
from .providers import (
    PointPrediction,
    ResolvedProvider,
    build_initial_density,
    predict_density,
    resolve_provider,
)
from .registry import DensityProvider, get_provider, provider_names, register_provider

__all__ = [
    "BOHR_TO_ANGSTROM",
    "BoxGrid",
    "DensityLoadResult",
    "DensityProvider",
    "PointPrediction",
    "ResolvedProvider",
    "box_around_atoms",
    "build_initial_density",
    "dense_from_grid",
    "get_provider",
    "load_density_for_grid",
    "normalize_density_units",
    "predict_density",
    "provider_names",
    "read_xyz",
    "register_provider",
    "resolve_provider",
    "save_box_density",
    "save_point_density",
    "write_cube",
]
