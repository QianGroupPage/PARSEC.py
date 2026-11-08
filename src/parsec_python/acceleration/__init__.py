"""Optimized runtimes used by the canonical :mod:`parsec_python` package."""

from parsec_python.Input import parse_parsec_input, summarize_translation

from .SCF import AcceleratedPreparedSinglePointSystem
from .Hartree import (
    CuPyPoissonSolver,
    FastMultipoleExpansion,
    NativePoissonSolver,
    SymmetryReducedPoissonSolver,
    density_multipoles_fast,
    solve_scipy_hartree,
)
from .Symmetry import AxisReflectionReduction, ReflectionRepresentationDecomposition
from .backends import (
    BackendSelection,
    NativeConjugateGradientBackend,
    NativeHamiltonianBackend,
    ScipyHamiltonianBackend,
    build_native_negative_laplacian,
    cupy_available,
    native_available,
    resolve_backend,
)
from .driver import (
    prepare_single_point,
    profile_hamiltonian_components,
    run_scf,
    run_single_point,
)
from .models import (
    AcceleratedSinglePointResult,
    BackendInfo,
    BackendName,
    BackendStatistics,
    BackendUnavailableError,
    SymmetryMode,
)

__all__ = [
    "AcceleratedPreparedSinglePointSystem",
    "AcceleratedSinglePointResult",
    "AxisReflectionReduction",
    "ReflectionRepresentationDecomposition",
    "BackendInfo",
    "BackendName",
    "BackendSelection",
    "BackendStatistics",
    "BackendUnavailableError",
    "SymmetryMode",
    "CuPyPoissonSolver",
    "FastMultipoleExpansion",
    "NativeHamiltonianBackend",
    "NativeConjugateGradientBackend",
    "NativePoissonSolver",
    "SymmetryReducedPoissonSolver",
    "ScipyHamiltonianBackend",
    "build_native_negative_laplacian",
    "cupy_available",
    "density_multipoles_fast",
    "native_available",
    "parse_parsec_input",
    "prepare_single_point",
    "profile_hamiltonian_components",
    "resolve_backend",
    "run_scf",
    "run_single_point",
    "solve_scipy_hartree",
    "summarize_translation",
]

# The k-point drivers and band paths of the reference SCF package run serially
# unless this parallel backend is installed.
from .SCF.kpoint_parallel import install as _install_kpoint_parallelism  # noqa: E402

_install_kpoint_parallelism()

# The Fock loop of a periodic hybrid with k-points runs its exchange on the
# devices of the k-point pool (SCF/hybrid_periodic.py) when that pool does.
from .SCF.hybrid_periodic import install as _install_device_bloch_exchange  # noqa: E402

_install_device_bloch_exchange()

# Preparing a periodic cell scans the whole grid for every atom image unless the fast
# replacements are installed.
from .V_ion.periodic_setup import install as _install_periodic_setup  # noqa: E402

_install_periodic_setup()

# The tetrahedron sums of Create_DOS run on the host unless the device provider is installed.
from .Output.device_dos import install as _install_device_dos  # noqa: E402

_install_device_dos()

# The wire, slab, Calc_Hartree_G and 3-D cell Hartree solves of the reference drivers run on
# the device where those put their eigenproblems (PARSEC_PERIODIC_HARTREE).
from .Hartree.periodic_device import install as _install_periodic_hartree  # noqa: E402

_install_periodic_hartree()

# The spin functionals of a periodic cell that PARSEC_CUPY_XC puts on the device (PBE only
# with PARSEC_CUPY_XC=on), in the reference periodic drivers that put their eigenproblems there.
from .V_xc.periodic_xc import install as _install_periodic_spin_xc  # noqa: E402

_install_periodic_spin_xc()

# The TS correction of a periodic cell whose k-point eigenproblems run on a GPU
# forms its grid and pair sums there.
from .Energy.dispersion_device import install as _install_device_dispersion  # noqa: E402

_install_device_dispersion()
