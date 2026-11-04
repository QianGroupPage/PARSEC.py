"""ARPACK implicitly restarted Lanczos/Arnoldi eigensolver, as PARSEC calls it.

Provenance
----------
``arpk_diag``/``zarpk_diag`` of PARSEC's ``src/arpk.f90z`` (built with
``-DUSEARPACK``) and the ARPACK branch of ``src/eigval.F90``:

* ``nev = nn`` wanted eigenpairs, ``ncv = 2*nev + 5`` Lanczos vectors
  (arpk.f90z:95-96), the standard problem (``bmat = 'I'``), the smallest
  algebraic eigenvalues (``which = 'SA'`` with the real ``dsaupd``,
  ``'SR'`` with the complex ``znaupd``), exact shifts (``iparam(1) = 1``),
  at most ``180*nev`` implicit restarts (``iparam(3)``), mode 1, the
  tolerance ``solver%toler`` (``Diag_Tolerance``) and eigenvectors from
  ``dseupd``/``zneupd`` (lines 140-153, 211-246);
* ``info = 0`` on every call (eigval.F90:357-360, 430-433: "Residual vector
  is not reused for the moment"), so ARPACK draws a random start vector each
  time with ``dlarnv(idist=2)`` from the LAPACK seed ``dgetv0`` keeps
  between calls (initially ``1, 3, 5, 7``);
* ``nec = iparam(5)`` converged pairs and ``maxmvs = iparam(9)`` products of
  the call, the pairs sorted ascending (``sortreal``) and the first ``nn``
  copied out (lines 304-341);
* eigval.F90 repeats the call up to five times until ``nec >= nn`` and fails
  with ``ierr = 402`` if the last call converged fewer than ``nn`` pairs or
  used ``solver%maxmv`` products or more
  (:mod:`parsec_python.Eigensolvers.iterative`).

The Python port calls :func:`scipy.sparse.linalg.eigsh` (which drives the
same ARPACK routines; a complex Hermitian operator goes through ``znaupd``
with ``'SR'``) with these parameters and passes the ``dlarnv`` start vector
explicitly, which is what ARPACK computes itself for ``info = 0``.

A property of ARPACK, kept: a single-vector Lanczos method can return
only some copies of a (nearly) degenerate eigenvalue when it meets the
tolerance before rounding errors have brought the others into its Krylov
space.  PARSEC solves each symmetry representation separately, which splits
many degenerate partners; the Python port solves the full grid.  Measured:
CH4 (2 of the 3 empty t2 states, the 8th state is the next level instead)
and Si8 (5 of a 6-fold level at the Fermi energy); the occupied states and
the total energy of CH4 were right, Si8 differed by 2.7e-6 Ry (with
``Diag_Tolerance`` 1e-6).  With ``Diag_Tolerance`` 1e-10 ARPACK found all
six copies of Si8; DIAGLA, a block method, found all copies in both cases.

Deviations: ``ncv`` is capped at the dimension (``eigsh`` needs
``ncv <= n``; PARSEC grids are far larger).  When ARPACK stops at its
restart limit, SciPy returns only the converged pairs; PARSEC then copies
``nn`` columns from an array holding fewer (arpk.f90z: "Watch now how I
crash"), the port keeps the converged ones and lets the retry loop and the
``ierr = 402`` test decide, as eigval.F90 does.  The matrix-vector budget
is not checked inside the reverse-communication loop: arpk.f90z compares
``iparam(9)``, which ARPACK sets only on return, so that test never fires
in PARSEC either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh

from .iterative import (
    EIGVAL_MAXIMUM_CALLS,
    IterativeDiagonalizationError,
    MatvecCounter,
)
from .lapack_random import LapackRandom

# ARPACK dgetv0: iseed = 1, 3, 5, 7 on the first call, kept afterwards.
ARPACK_START_SEED = (1, 3, 5, 7)
# arpk.f90z: iparam(3) = nev*180
ARPACK_RESTARTS_PER_STATE = 180


@dataclass(frozen=True)
class ArpackSettings:
    """Controls of PARSEC's ARPACK call.

    ``tolerance`` is ``Diag_Tolerance`` (ARPACK's relative accuracy of the
    Ritz values), ``maximum_matvec`` is ``Maximum_matvec`` before the rule
    ``max(Maximum_matvec, n * n_states)`` raises it, ``maximum_calls`` is
    eigval.F90's ``maxit``.
    """

    tolerance: float = 1.0e-4
    maximum_matvec: int = 30000
    restarts_per_state: int = ARPACK_RESTARTS_PER_STATE
    maximum_calls: int = EIGVAL_MAXIMUM_CALLS
    start_seed: tuple[int, int, int, int] = ARPACK_START_SEED

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("ARPACK tolerance must be nonnegative")
        for name in ("maximum_matvec", "restarts_per_state", "maximum_calls"):
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        # Validates the four LAPACK seed digits.
        object.__setattr__(
            self, "start_seed", tuple(LapackRandom(tuple(self.start_seed)).seed)
        )


@dataclass(frozen=True)
class ArpackCall:
    """One ``arpk_diag`` call: converged pairs and products it used."""

    converged_states: int
    matvecs: int


@dataclass(frozen=True)
class ArpackResult:
    """Outcome of eigval.F90's ARPACK loop.

    ``eigenvalues`` (Ry, ascending) and ``vectors`` (``(n, wanted)``,
    Euclidean-normalized columns) are those of the last call.
    ``converged_states`` and ``matvecs`` are its ``nec`` and ``maxmvs``;
    ``start_seed`` is the LAPACK seed after its start vector, which the next
    solve continues from.
    """

    eigenvalues: np.ndarray
    vectors: np.ndarray
    converged_states: int
    matvecs: int
    calls: tuple[ArpackCall, ...]
    start_seed: tuple[int, int, int, int]

    @property
    def total_matvecs(self) -> int:
        return int(sum(call.matvecs for call in self.calls))


def arpack_start_vector(random: LapackRandom, dimension: int, dtype) -> np.ndarray:
    """ARPACK's random start: ``dlarnv``/``zlarnv`` with ``idist = 2``.

    Real: ``n`` values uniform in (-1, 1).  Complex (``zlarnv``): the real
    and imaginary part of element ``i`` are ``2 u(2i-1) - 1`` and
    ``2 u(2i) - 1``.
    """

    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        pairs = 2.0 * random.uniform_0_1(2 * int(dimension)) - 1.0
        return pairs[0::2] + 1j * pairs[1::2]
    return random.uniform_minus_1_1(int(dimension))


def arpack_diag(
    operator: Any,
    wanted_states: int,
    *,
    tolerance: float,
    maximum_restarts: int,
    start_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """One ``arpk_diag`` call: ``(values, vectors, nec, matvecs)``.

    ``values`` ascending; only the converged pairs are returned when ARPACK
    reaches its restart limit.  Raises :class:`IterativeDiagonalizationError`
    (``ierr`` 401) for an ARPACK error.
    """

    counter = MatvecCounter(operator)
    dimension = counter.shape[0]
    complex_problem = np.issubdtype(counter.dtype, np.complexfloating) or (
        np.iscomplexobj(start_vector)
    )
    # eigsh (dsaupd) needs nev < n; the complex path (znaupd) nev < n - 1.
    largest = dimension - 2 if complex_problem else dimension - 1
    if not 1 <= wanted_states <= largest:
        raise ValueError(
            f"ARPACK needs between 1 and {largest} eigenpairs for an operator "
            f"of dimension {dimension}, got {wanted_states}"
        )
    lanczos_vectors = min(2 * wanted_states + 5, dimension)
    try:
        values, vectors = eigsh(
            counter.as_linear_operator(),
            k=wanted_states,
            which="SA",
            ncv=lanczos_vectors,
            maxiter=maximum_restarts,
            tol=tolerance,
            v0=start_vector,
            return_eigenvectors=True,
        )
        converged = int(np.size(values))
    except ArpackNoConvergence as stopped:
        values = np.asarray(stopped.eigenvalues)
        vectors = np.asarray(stopped.eigenvectors)
        converged = int(np.size(values))
    except ArpackError as error:
        raise IterativeDiagonalizationError(
            "ARPACK",
            ierr=401,
            wanted=wanted_states,
            converged=0,
            matvecs=counter.count,
            budget=0,
            calls=1,
            info=int(getattr(error, "info", -1)),
        ) from error
    values = np.real(np.asarray(values))
    vectors = np.asarray(vectors).reshape(dimension, -1)
    order = np.argsort(values, kind="stable")
    return values[order], vectors[:, order], converged, counter.count


def run_arpack(
    operator: Any,
    wanted_states: int,
    *,
    settings: ArpackSettings = ArpackSettings(),
    matvec_budget: int | None = None,
    start_seed: tuple[int, int, int, int] | None = None,
) -> ArpackResult:
    """Run eigval.F90's ARPACK loop for ``wanted_states`` eigenpairs.

    ``matvec_budget`` is ``solver%maxmv`` (``settings.maximum_matvec`` when
    omitted).  ``start_seed`` continues the LAPACK stream of an earlier
    solve; ``settings.start_seed`` starts a new one.  Raises
    :class:`IterativeDiagonalizationError` with ``ierr`` 402 when the last
    call converged fewer than ``wanted_states`` pairs or used the budget.
    """

    wanted_states = int(wanted_states)
    budget = int(settings.maximum_matvec if matvec_budget is None else matvec_budget)
    random = LapackRandom(settings.start_seed if start_seed is None else start_seed)
    dtype = np.dtype(getattr(operator, "dtype", np.float64))
    dimension = int(operator.shape[0])
    calls: list[ArpackCall] = []
    values = np.empty(0)
    vectors = np.empty((dimension, 0), dtype=dtype)
    converged = matvecs = 0
    for _ in range(settings.maximum_calls):
        start = arpack_start_vector(random, dimension, dtype)
        values, vectors, converged, matvecs = arpack_diag(
            operator,
            wanted_states,
            tolerance=settings.tolerance,
            maximum_restarts=settings.restarts_per_state * wanted_states,
            start_vector=start,
        )
        calls.append(ArpackCall(converged, matvecs))
        if converged >= wanted_states:
            break
    if converged < wanted_states or matvecs >= budget:
        raise IterativeDiagonalizationError(
            "ARPACK",
            ierr=402,
            wanted=wanted_states,
            converged=converged,
            matvecs=matvecs,
            budget=budget,
            calls=len(calls),
        )
    return ArpackResult(
        eigenvalues=values[:wanted_states],
        vectors=vectors[:, :wanted_states],
        converged_states=min(converged, wanted_states),
        matvecs=matvecs,
        calls=tuple(calls),
        start_seed=tuple(random.seed),
    )


__all__ = [
    "ARPACK_RESTARTS_PER_STATE",
    "ARPACK_START_SEED",
    "ArpackCall",
    "ArpackResult",
    "ArpackSettings",
    "arpack_diag",
    "arpack_start_vector",
    "run_arpack",
]
