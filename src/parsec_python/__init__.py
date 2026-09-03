"""PARSEC-style real-space DFT with accelerated execution by default.

The top-level scientific components remain small and independently usable.
The three workflow functions exported here select the optimized backend;
explicit ``*_reference`` aliases retain the readable SciPy implementation for
algorithm inspection and parity tests.
"""

from .Eigensolvers import (
    ChebDavResult,
    ChebDavSettings,
    ChebFFResult,
    ChebFFSettings,
    EigvalResult,
    EigvalSettings,
    EigvalState,
    StrictEigvalSolver,
    SubspaceResult,
    SubspaceSettings,
    run_chebdav,
    solve_eigval,
)
from .Energy import total_energy
from .Grid import RealSpaceGrid, build_cluster_grid
from .Hamiltonian import KohnShamHamiltonian
from .Hartree import (
    DirectCoulombBoundary,
    HartreeResult,
    MultipoleExpansion,
    density_multipoles,
    solve_hartree,
)
from .Input import (
    ANGSTROM_TO_BOHR,
    ParsecInputError,
    ParsecInputTranslation,
    UnsupportedParsecOptionError,
    parse_parsec_input,
    summarize_translation,
)
from .Laplacian import (
    apply_negative_laplacian_boundary,
    build_negative_laplacian,
    second_derivative_coefficients,
)
from .Mixer import AndersonMixer, ResidualMetrics, potential_residual_metrics
from .MLDensity import (
    DensityLoadResult,
    build_initial_density,
    load_density_for_grid,
    save_point_density,
)
from .Occupations import (
    BOLTZMANN_RYDBERG_PER_KELVIN,
    OccupationResult,
    density_from_orbitals,
    fermi_occupations,
)
from .Output import ParsecTextReporter, RYDBERG_TO_EV
from .Pseudopotential import (
    ParsecPseudopotential,
    ParsecRadialSpline,
    parsec_radial_integral,
    read_parsec_pseudopotential,
)
from .V_ion import (
    NonlocalProjectorOperator,
    build_local_ionic_potential,
    build_nonlocal_projectors,
    center_cluster_geometry,
    ion_ion_energy,
    load_pseudopotentials,
    normalize_density,
    superpose_atomic_density,
)
from .V_xc import (
    XCResult,
    ca_lda,
    first_derivative_coefficients,
    pbe,
    pbe_energy_partials,
)
from .flow_map import PARSEC_SINGLE_POINT_FLOW, format_flow_map
from .models import (
    Atom,
    EigensolverSettings,
    EnergyBreakdown,
    GridSettings,
    HartreeSettings,
    InitialDensitySettings,
    MixingSettings,
    PreparationTimings,
    RunTimings,
    SCFIteration,
    SCFSettings,
    SinglePointInput,
    SinglePointResult,
    SpeciesPotential,
    XCFunctional,
)
from .driver import (
    PreparedSinglePointSystem,
    prepare_single_point as prepare_reference_single_point,
    run_scf as run_reference_scf,
    run_single_point as run_reference_single_point,
)
from .acceleration.driver import (
    AcceleratedPreparedSinglePointSystem,
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
    run_single_point,
)
from .acceleration.models import (
    AcceleratedSinglePointResult,
    BackendInfo,
    BackendName,
    BackendStatistics,
    BackendUnavailableError,
    SymmetryMode,
)

__all__ = [
    "AndersonMixer",
    "ANGSTROM_TO_BOHR",
    "AcceleratedPreparedSinglePointSystem",
    "AcceleratedSinglePointResult",
    "Atom",
    "BOLTZMANN_RYDBERG_PER_KELVIN",
    "BackendInfo",
    "BackendName",
    "BackendStatistics",
    "BackendUnavailableError",
    "ChebDavResult",
    "ChebDavSettings",
    "ChebFFResult",
    "ChebFFSettings",
    "DirectCoulombBoundary",
    "DensityLoadResult",
    "EigensolverSettings",
    "EigvalResult",
    "EigvalSettings",
    "EigvalState",
    "EnergyBreakdown",
    "GridSettings",
    "HartreeResult",
    "HartreeSettings",
    "InitialDensitySettings",
    "KohnShamHamiltonian",
    "MixingSettings",
    "MultipoleExpansion",
    "NonlocalProjectorOperator",
    "OccupationResult",
    "PARSEC_SINGLE_POINT_FLOW",
    "ParsecPseudopotential",
    "ParsecRadialSpline",
    "ParsecTextReporter",
    "ParsecInputError",
    "ParsecInputTranslation",
    "PreparationTimings",
    "PreparedSinglePointSystem",
    "RealSpaceGrid",
    "ResidualMetrics",
    "RYDBERG_TO_EV",
    "RunTimings",
    "SCFIteration",
    "SCFSettings",
    "SinglePointInput",
    "SinglePointResult",
    "SpeciesPotential",
    "StrictEigvalSolver",
    "SubspaceResult",
    "SubspaceSettings",
    "SymmetryMode",
    "UnsupportedParsecOptionError",
    "XCResult",
    "XCFunctional",
    "apply_negative_laplacian_boundary",
    "build_cluster_grid",
    "build_initial_density",
    "build_local_ionic_potential",
    "build_negative_laplacian",
    "build_nonlocal_projectors",
    "ca_lda",
    "first_derivative_coefficients",
    "pbe",
    "pbe_energy_partials",
    "center_cluster_geometry",
    "density_from_orbitals",
    "density_multipoles",
    "fermi_occupations",
    "format_flow_map",
    "ion_ion_energy",
    "load_pseudopotentials",
    "load_density_for_grid",
    "normalize_density",
    "parse_parsec_input",
    "parsec_radial_integral",
    "potential_residual_metrics",
    "prepare_single_point",
    "prepare_reference_single_point",
    "profile_hamiltonian_components",
    "read_parsec_pseudopotential",
    "run_scf",
    "run_reference_scf",
    "run_reference_single_point",
    "run_chebdav",
    "run_single_point",
    "save_point_density",
    "second_derivative_coefficients",
    "solve_hartree",
    "solve_eigval",
    "superpose_atomic_density",
    "summarize_translation",
    "total_energy",
]
