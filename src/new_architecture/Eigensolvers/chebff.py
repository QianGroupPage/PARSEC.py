"""PARSEC's first-filter (CHEBFF) eigensolver, expressed as Python steps.

Provenance
----------
This module follows ``chebff_diag`` and its scaled filter
``cheby_filterscal1`` in PARSEC's ``src/chebff.f90z``.  It is a direct
Python implementation of the orchestration:

1. create a random trial subspace,
2. estimate the spectral interval with ``lancz_bound``,
3. apply a scaled Chebyshev filter,
4. orthonormalize and perform a Rayleigh--Ritz rotation, and
5. update the filter bounds before the next configured filter cycle.

CHEBFF deliberately does *not* test individual Ritz residuals.  PARSEC marks
all requested pairs as approximate convergence after ``FF_MaxIter`` cycles;
this module preserves that policy and never falls back to ARPACK or another
eigensolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .chebyshev import chebff_filter
from .lapack_random import LapackRandom
from .orthogonalize import orthonormalize
from .rayleigh_ritz import rayleigh_ritz
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


@dataclass(frozen=True)
class ChebFFSettings:
    """Input controls corresponding to PARSEC's CHEBFF solver fields.

    ``polynomial_degree`` is ``Chebdav_Degree`` and ``filter_cycles`` is
    ``FF_MaxIter``.  Every cycle performs that many recurrence steps followed
    by orthonormalization and one Rayleigh--Ritz rotation.  In particular,
    ``filter_cycles`` is a fixed work count, not a maximum checked against an
    eigenpair-residual tolerance.

    ``block_size`` changes only how many vector columns are sent through one
    Hamiltonian block application.  With the production value
    ``reset_recurrence_per_block=False``, the scalar recurrence state is
    carried from one block to the next exactly as in the audited Fortran
    source.

    ``random_seed`` and ``reset_recurrence_per_block`` are explicit Python
    reproducibility/testing controls, not PARSEC input-file labels.  The trial
    basis itself always starts from PARSEC's canonical LAPACK seed; this seed
    controls the separate NumPy short-Lanczos starting vector.
    """

    polynomial_degree: int = 20
    filter_cycles: int = 2
    lanczos_steps: int = 10
    block_size: int = 6
    reset_recurrence_per_block: bool = False
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.polynomial_degree < 1:
            raise ValueError("polynomial_degree must be positive")
        if not 1 <= self.filter_cycles < 20:
            raise ValueError("filter_cycles must be in PARSEC's [1, 19] range")
        if self.lanczos_steps < 1:
            raise ValueError("lanczos_steps must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")


@dataclass(frozen=True)
class ChebFFCycle:
    """Observable state for one filter/Rayleigh--Ritz cycle."""

    number: int
    lower_bound_in: float
    upper_bound_in: float
    lower_bound_out: float
    upper_bound_out: float
    smallest_ritz_value: float
    largest_ritz_value: float


@dataclass(frozen=True)
class ChebFFState:
    """Full buffered working subspace produced by the first SCF solve.

    The arrays contain the requested physical states *and* the safety-buffer
    states.  :mod:`eigval` returns only the requested leading columns to SCF,
    but preserves this complete state so that later SCF iterations start from
    the already filtered subspace instead of new random vectors.
    """

    operator_dimension: int
    wanted_states: int
    eigenvalues: np.ndarray
    vectors: np.ndarray
    filter_lower_bound: float
    spectral_upper_bound: float
    smallest_ritz_value: float


@dataclass(frozen=True)
class ChebFFResult:
    """Result of the fixed number of PARSEC first-filter cycles."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    state: ChebFFState
    lanczos_bound: LanczosBoundResult
    cycles: tuple[ChebFFCycle, ...]

    @property
    def approximate_converged_states(self) -> int:
        """Number PARSEC reports as converged, without a residual test."""

        return self.state.wanted_states


def _operator_dimension(operator: Any) -> int:
    """Validate the square matrix-like interface and return its dimension."""

    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    return int(shape[0])


def _operator_is_complex(operator: Any) -> bool:
    """Report whether the advertised operator dtype needs a complex path."""

    dtype = getattr(operator, "dtype", None)
    return dtype is not None and np.issubdtype(np.dtype(dtype), np.complexfloating)


def _random_vectors(
    dimension: int,
    count: int,
    generator: LapackRandom,
    *,
    complex_values: bool,
) -> np.ndarray:
    """Generate PARSEC's first trial subspace in Fortran column order.

    The trial basis and the short-Lanczos starting vector intentionally use
    two different random streams.  This function consumes the stateful LAPACK
    ``DLARNV(IDIST=2)`` translation, beginning from PARSEC's canonical seed.
    ``settings.random_seed`` is used separately for the Lanczos vector below;
    it does not replace this LAPACK seed.
    """

    if complex_values:
        raise NotImplementedError(
            "strict CHEBFF currently implements PARSEC's real scalar path"
        )
    # random_array calls DLARNV once per complete Fortran column.
    return generator.uniform_minus_1_1(
        (dimension, count),
        column_major=True,
    )


def _initial_filter_lower_bound(
    estimated_lowest_eigenvalue: float,
    spectral_upper_bound: float,
) -> float:
    """Place the first unwanted-spectrum boundary above the low estimate.

    If ``lambda0`` is the lowest Ritz estimate from short Lanczos and ``u``
    is its conservative spectral upper bound, PARSEC chooses

    ``l = (2*lambda0 + u)/3``.

    The desired low-energy states lie below ``l``; ``[l, u]`` is the interval
    suppressed by the Chebyshev polynomial.
    """

    # chebff.f90z: lowb = (lowb0 + (lowb0 + upperb)) / 3
    return (2.0 * estimated_lowest_eigenvalue + spectral_upper_bound) / 3.0


def _updated_filter_bounds(
    lower_bound: float,
    upper_bound: float,
    smallest_ritz: float,
    largest_ritz: float,
) -> tuple[float, float]:
    """Translate CHEBFF's post-Rayleigh--Ritz bound update literally.

    Let ``r_min`` and ``r_max`` be the smallest and largest Ritz values after
    a filter cycle.  If ``r_max`` escapes above the old upper bound ``u``, the
    interval is enlarged according to

    ``u' = r_max + 0.5*(r_max-u) + 1``

    and the lower boundary is allowed to move only downward,

    ``l' = min(l, (3*l+u')/4)``.

    Otherwise only the lower boundary is tightened:

    ``l' = min(r_max+0.001*(u-r_min), r_max+0.05*abs(r_max))``.

    These empirical PARSEC formulas are filter-window updates; they are not
    eigenvalue convergence tests.
    """

    if largest_ritz >= upper_bound:
        new_upper = largest_ritz + 0.5 * (largest_ritz - upper_bound) + 1.0
        new_lower = min(lower_bound, (3.0 * lower_bound + new_upper) / 4.0)
        return float(new_lower), float(new_upper)

    new_lower = min(
        largest_ritz + 0.001 * (upper_bound - smallest_ritz),
        largest_ritz + 0.05 * abs(largest_ritz),
    )
    return float(new_lower), float(upper_bound)


def run_chebff(
    operator: Any,
    wanted_states: int,
    *,
    settings: ChebFFSettings = ChebFFSettings(),
) -> ChebFFResult:
    """Build PARSEC's initial approximate eigensubspace.

    Parameters
    ----------
    operator
        A square Hermitian matrix or matrix-like object supporting ``@``.
    wanted_states
        Size of the complete working subspace.  The caller is responsible for
        adding PARSEC's safety states before calling this routine.
    settings
        First-filter degree, cycle count, bound-estimator length, and seed.

    Notes
    -----
    For an unwanted spectral interval ``[l, u]``, the lower estimated
    eigenvalue is used as the normalization point below that interval.  The
    normalized three-term recurrence itself lives in
    :func:`chebyshev.chebff_filter`.  After each fixed filter cycle this
    routine constructs an orthonormal basis ``Q`` and solves the small
    projected problem

    ``(Q.T @ H @ Q) U = U diag(epsilon)``, ``Psi = Q @ U``.

    No residual threshold terminates this loop early.  PARSEC treats the
    complete working subspace as approximately converged after the requested
    number of cycles.
    """

    dimension = _operator_dimension(operator)
    if not 1 <= wanted_states <= dimension:
        raise ValueError(
            f"wanted_states must be between 1 and {dimension}, got {wanted_states}"
        )

    # Stage 1: form the complete buffered trial basis.  PARSEC random_array
    # (DLARNV) and Fortran RANDOM_NUMBER have independent state, so the basis
    # and short-Lanczos start below deliberately come from different streams.
    basis_generator = LapackRandom()
    bound_generator = np.random.default_rng(settings.random_seed)
    complex_values = _operator_is_complex(operator)
    vectors = _random_vectors(
        dimension,
        wanted_states,
        basis_generator,
        complex_values=complex_values,
    )
    # Stage 2: estimate the unwanted spectrum's upper edge.  The SCF caller
    # requests ten Lanczos steps for CHEBFF; the non-BETA bound routine itself
    # applies PARSEC's inclusive [4, 8] clamp, so production uses eight.
    bound = lanczos_upper_bound(
        operator,
        steps=settings.lanczos_steps,
        rng=bound_generator,
    )

    upper_bound = float(bound.upper_bound)
    # Before the first Rayleigh--Ritz solve, ``smallest_ritz`` is the short-
    # Lanczos lower estimate.  Later cycles replace it with the actual lowest
    # Ritz value of the filtered working subspace.
    smallest_ritz = float(bound.lower_bound)
    lower_bound = _initial_filter_lower_bound(smallest_ritz, upper_bound)
    eigenvalues = np.empty(wanted_states, dtype=float)
    cycle_records: list[ChebFFCycle] = []

    for cycle_number in range(1, settings.filter_cycles + 1):
        lower_in = lower_bound
        upper_in = upper_bound
        # Stage 3a: amplify states below lower_bound while damping the
        # unwanted interval [lower_bound, upper_bound].  H is applied through
        # the matrix-free Kohn--Sham LinearOperator supplied by SCF.
        vectors = chebff_filter(
            operator,
            vectors,
            degree=settings.polynomial_degree,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            reference_eigenvalue=smallest_ritz,
            block_size=settings.block_size,
            reset_recurrence_per_block=settings.reset_recurrence_per_block,
        )
        # Stage 3b: repair the filtered basis, then diagonalize H only in this
        # small working subspace.  The same LAPACK stream is retained for the
        # rare random-replacement branch of orthonormalization.
        vectors = orthonormalize(vectors, rng=basis_generator).basis
        ritz = rayleigh_ritz(operator, vectors)
        eigenvalues = np.asarray(ritz.eigenvalues, dtype=float)
        vectors = np.asarray(ritz.wavefunctions)

        if eigenvalues.shape != (wanted_states,):
            raise RuntimeError(
                "Rayleigh--Ritz changed the CHEBFF working-subspace dimension"
            )
        smallest_ritz = float(eigenvalues[0])
        largest_ritz = float(eigenvalues[-1])
        # Stage 3c: update only the next cycle's filter window.  Rayleigh--Ritz
        # residuals are diagnostic quantities and do not alter fixed CHEBFF
        # work or trigger an alternate solver.
        lower_bound, upper_bound = _updated_filter_bounds(
            lower_bound,
            upper_bound,
            smallest_ritz,
            largest_ritz,
        )
        cycle_records.append(
            ChebFFCycle(
                number=cycle_number,
                lower_bound_in=float(lower_in),
                upper_bound_in=float(upper_in),
                lower_bound_out=lower_bound,
                upper_bound_out=upper_bound,
                smallest_ritz_value=smallest_ritz,
                largest_ritz_value=largest_ritz,
            )
        )

    state = ChebFFState(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        eigenvalues=eigenvalues,
        vectors=vectors,
        filter_lower_bound=lower_bound,
        spectral_upper_bound=upper_bound,
        smallest_ritz_value=smallest_ritz,
    )
    return ChebFFResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        state=state,
        lanczos_bound=bound,
        cycles=tuple(cycle_records),
    )
