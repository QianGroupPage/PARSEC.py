"""Native Python translations of PARSEC eigensolver routines.

The strict isolated-system path is deliberately assembled from small,
independently testable steps:

First SCF eigensolve (selected by the input):

``eigval.solve_eigval`` -> ``chebff.run_chebff`` ->
``lapack_random`` + ``spectral_bounds`` + ``chebyshev`` +
``orthogonalize`` + ``rayleigh_ritz``.

or

``eigval.solve_eigval`` -> ``chebdav.run_chebdav`` -> the same numerical
primitives plus CHEBDAV residual locking and restart orchestration.

Later SCF eigensolves:

``eigval.solve_eigval`` -> ``subspace.run_subspace_filter`` ->
``spectral_bounds`` + ``chebyshev`` + ``orthogonalize`` +
``rayleigh_ritz``.

Only the first requested Ritz pairs return to occupations, while every
requested-plus-buffer pair is saved for the next nonlinear iteration.

Nothing in this package invokes PARSEC Fortran or silently falls back to
ARPACK.  ARPACK support, when added as a separate explicitly selected
solver, belongs behind a different dispatch path.
"""

from .chebdav import (
    ChebDavIteration,
    ChebDavResult,
    ChebDavSettings,
    ChebDavState,
    run_chebdav,
)
from .chebff import (
    ChebFFCycle,
    ChebFFResult,
    ChebFFSettings,
    ChebFFState,
    run_chebff,
)
from .chebyshev import (
    FilterBlock,
    chebff_filter,
    chebyshev_filter,
    subspace_filter,
    subspace_filter_blocks,
)
from .eigval import (
    EigvalResult,
    EigvalSettings,
    EigvalState,
    StrictEigvalSolver,
    solve_eigval,
)
from .lapack_random import LapackRandom, PARSEC_RANDOM_ARRAY_SEED
from .orthogonalize import OrthonormalizationResult, orthonormalize
from .rayleigh_ritz import RayleighRitzResult, rayleigh_ritz
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound
from .subspace import (
    SubspaceResult,
    SubspaceSettings,
    SubspaceState,
    run_subspace_filter,
)

__all__ = [
    "ChebFFCycle",
    "ChebFFResult",
    "ChebFFSettings",
    "ChebFFState",
    "ChebDavIteration",
    "ChebDavResult",
    "ChebDavSettings",
    "ChebDavState",
    "EigvalResult",
    "EigvalSettings",
    "EigvalState",
    "FilterBlock",
    "LanczosBoundResult",
    "LapackRandom",
    "OrthonormalizationResult",
    "PARSEC_RANDOM_ARRAY_SEED",
    "RayleighRitzResult",
    "StrictEigvalSolver",
    "SubspaceResult",
    "SubspaceSettings",
    "SubspaceState",
    "chebff_filter",
    "chebyshev_filter",
    "lanczos_upper_bound",
    "orthonormalize",
    "rayleigh_ritz",
    "run_chebff",
    "run_chebdav",
    "run_subspace_filter",
    "solve_eigval",
    "subspace_filter",
    "subspace_filter_blocks",
]
