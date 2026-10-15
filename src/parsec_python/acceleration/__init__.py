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
