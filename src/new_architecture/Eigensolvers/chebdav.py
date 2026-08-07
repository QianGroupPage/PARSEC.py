"""Native real-scalar translation of PARSEC's Chebyshev--Davidson solver.

PARSEC uses ``chebdav_diag`` only for the first diagonalization of an SCF
sequence.  The routine builds a block Davidson subspace, but its expansion
vectors are Chebyshev-filtered low Ritz vectors rather than preconditioned
residuals.  Once that first solve has produced a reusable eigensubspace,
later SCF iterations use PARSEC's separate ``subspace`` routine.

This module follows the non-``BETA`` ``flag='new'`` path in
``src/chebdav.f90z``:

* a five-step Lanczos run estimates the spectral interval;
* a random block is filtered and orthonormalized;
* the projected Hamiltonian is extended incrementally;
* only a contiguous lowest-residual Ritz prefix is locked;
* inner and outer restarts retain low active Ritz vectors; and
* an early/restart-limited exit finishes with filtered subspace iteration
  and Rayleigh--Ritz, which PARSEC deliberately accepts as approximate.

The Hamiltonian remains matrix-free.  Only basis blocks, their cached
Hamiltonian images, and small projected matrices are stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .chebyshev import chebff_filter, chebyshev_filter
from .lapack_random import LapackRandom
from .orthogonalize import orthonormalize
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound


@dataclass(frozen=True)
class ChebDavSettings:
    """Controls for PARSEC's first-SCF CHEBDAV calculation.

    ``polynomial_degree``, ``convergence_tolerance``, and ``block_size`` map
    to ``Chebdav_Degree``, ``Diag_Tolerance``, and ``Matvec_Blocksize``.
    PARSEC requires a degree of at least 15 for CHEBDAV.

    ``workspace_window=12`` is the hard-coded ``solver%winsize`` allocated
    in ``structures.F90`` in addition to the wanted (physical plus safety)
    states.  ``max_outer_restarts=2`` is also a source constant; because the
    Fortran exit test uses ``>`` rather than ``>=``, three outer restarts can
    actually be taken.

    ``max_iterations`` and ``random_seed`` are explicit Python diagnostic
    controls.  If ``max_iterations`` is omitted, the source formula is used.
    The seed controls the compiler-RNG-like Lanczos starting vector; trial
    basis columns use PARSEC's independent canonical LAPACK random stream.
    """

    polynomial_degree: int = 20
    convergence_tolerance: float = 1.0e-4
    block_size: int = 6
    workspace_window: int = 12
    lanczos_steps: int = 5
    max_outer_restarts: int = 2
    max_iterations: int | None = None
    random_seed: int = 7

    def __post_init__(self) -> None:
        if int(self.polynomial_degree) != self.polynomial_degree:
            raise ValueError("polynomial_degree must be an integer")
        if self.polynomial_degree < 15:
            raise ValueError("PARSEC CHEBDAV requires polynomial_degree >= 15")
        if (
            not np.isfinite(self.convergence_tolerance)
            or self.convergence_tolerance <= 0.0
        ):
            raise ValueError("convergence_tolerance must be positive")
        for name in ("block_size", "workspace_window", "lanczos_steps"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} has an invalid value")
        if (
            int(self.max_outer_restarts) != self.max_outer_restarts
            or self.max_outer_restarts < 0
        ):
            raise ValueError("max_outer_restarts has an invalid value")
        if int(self.random_seed) != self.random_seed or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if self.max_iterations is not None:
            if int(self.max_iterations) != self.max_iterations or self.max_iterations < 1:
                raise ValueError("max_iterations must be a positive integer")


@dataclass(frozen=True)
class ChebDavIteration:
    """Diagnostics captured after one projected Davidson solve."""

    number: int
    locked_before: int
    locked_after: int
    active_dimension_before_restart: int
    active_dimension_retained: int
    residual_tolerance: float
    tested_residual_norms: np.ndarray
    inner_restart: bool
    outer_restart: bool


@dataclass(frozen=True)
class ChebDavState:
    """Complete first-solve output and CHEBDAV work diagnostics.

    ``matrix_vector_products`` reproduces PARSEC's printed ``num_mv`` counter.
    That source counter intentionally omits the final cleanup projection's
    Hamiltonian application, even though the operation is performed.
    """

    operator_dimension: int
    wanted_states: int
    maximum_subspace_dimension: int
    eigenvalues: np.ndarray
    vectors: np.ndarray
    residual_norms: np.ndarray
    truly_converged_states: int
    approximate_cleanup_used: bool
    matrix_vector_products: int
    iterations_completed: int
    inner_restarts: int
    outer_restarts: int
    filter_lower_bound: float
    spectral_upper_bound: float


@dataclass(frozen=True)
class ChebDavResult:
    """Lowest CHEBDAV Ritz pairs and the state used by later SUBSPACE."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    residual_norms: np.ndarray
    state: ChebDavState
    lanczos_bound: LanczosBoundResult
    iterations: tuple[ChebDavIteration, ...]

    @property
    def approximate_converged_states(self) -> int:
        """Match PARSEC's post-cleanup reported convergence count."""

        return self.state.wanted_states


def _operator_dimension(operator: Any) -> int:
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    dimension = int(shape[0])
    if dimension < 1:
        raise ValueError("operator dimension must be positive")
    dtype = getattr(operator, "dtype", np.float64)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        raise NotImplementedError(
            "native CHEBDAV currently implements PARSEC's real scalar path"
        )
    return dimension


def _apply(operator: Any, vectors: np.ndarray) -> np.ndarray:
    result = np.asarray(operator @ vectors)
    if result.shape != vectors.shape:
        raise ValueError("operator must preserve the vector or block shape")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("operator application produced nonfinite values")
    return np.asarray(result, dtype=float)


def _rayleigh_ritz_upper(
    operator: Any,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rayleigh--Ritz using CHEBDAV's authoritative upper triangle.

    ``chebdav.f90z`` calls ``DSYEV('U')``.  The lower triangle of the raw
    floating-point product is therefore ignored, even though the exact
    projected Hamiltonian is symmetric.  Mirroring the computed upper
    triangle reproduces that convention before NumPy's dense eigensolve.
    """

    applied = _apply(operator, basis)
    raw = basis.T @ applied
    upper = np.triu(raw)
    projected = upper + np.triu(upper, 1).T
    eigenvalues, rotations = np.linalg.eigh(projected)
    vectors = basis @ rotations
    applied_vectors = applied @ rotations
    residual_norms = np.linalg.norm(
        applied_vectors - vectors * eigenvalues[np.newaxis, :],
        axis=0,
    )
    return (
        np.asarray(eigenvalues, dtype=float),
        np.asarray(vectors, dtype=float),
        np.asarray(residual_norms, dtype=float),
    )


def _active_subspace_limit(
    wanted_states: int,
    block_size: int,
    maximum_subspace_dimension: int,
) -> int:
    """Translate the piecewise ``kactmax`` table in ``chebdav.f90z``."""

    if wanted_states >= 2000:
        limit = min(max(15 * block_size, wanted_states // 10), 400)
    elif wanted_states >= 1000:
        limit = max(15 * block_size, 150)
    elif wanted_states >= 500:
        limit = max(15 * block_size, 80)
    elif wanted_states >= 300:
        limit = max(14 * block_size, 70)
    elif wanted_states >= 80:
        limit = max(10 * block_size, 50)
    elif wanted_states >= 36:
        limit = 36
    else:
        limit = 30
    return max(block_size, min(limit, maximum_subspace_dimension))


def _source_max_iterations(
    wanted_states: int,
    block_size: int,
    active_limit: int,
) -> int:
    iteration_limit = max(min(wanted_states * 5, 400), wanted_states * 3)
    return max(active_limit + 20 * block_size, iteration_limit)


def _initial_lower_bound(lowest_estimate: float, upper_bound: float) -> float:
    """Return ``max(0, (2.2*lambda0 + 0.8*u)/3)`` from CHEBDAV."""

    return max(0.0, (2.2 * lowest_estimate + 0.8 * upper_bound) / 3.0)


def _inner_restart_size(active_limit: int, block_size: int) -> int:
    retained = min(
        max(active_limit // 2, active_limit - 2 * block_size) + 2,
        active_limit - block_size,
    )
    return max(block_size, retained)


def _orthonormalize_appended(
    basis: np.ndarray,
    existing_columns: int,
    appended_columns: int,
    generator: LapackRandom,
) -> None:
    """Run PARSEC ``orth_normal`` only on a newly appended basis block."""

    stop = existing_columns + appended_columns
    view = basis[:, :stop]
    orthonormalize(
        view,
        existing_columns=existing_columns,
        existing_are_orthonormal=True,
        rng=generator,
        copy=False,
    )


def _insert_locked_pair(
    eigenvalues: np.ndarray,
    residual_norms: np.ndarray,
    basis: np.ndarray,
    locked_count: int,
    value: float,
    residual_norm: float,
    vector_column: int,
) -> tuple[int, bool]:
    """Insert one newly locked pair into the ascending locked prefix."""

    vector = basis[:, vector_column].copy()
    position = int(np.searchsorted(eigenvalues[:locked_count], value, side="right"))
    reordered = position < locked_count
    if reordered:
        eigenvalues[position + 1 : locked_count + 1] = eigenvalues[
            position:locked_count
        ].copy()
        residual_norms[position + 1 : locked_count + 1] = residual_norms[
            position:locked_count
        ].copy()
        basis[:, position + 1 : locked_count + 1] = basis[
            :, position:locked_count
        ].copy()
    eigenvalues[position] = value
    residual_norms[position] = residual_norm
    basis[:, position] = vector
    return locked_count + 1, reordered


def _final_approximate_subspace(
    operator: Any,
    basis: np.ndarray,
    locked_values: np.ndarray,
    locked_residuals: np.ndarray,
    locked_count: int,
    source_subspace_dimension: int,
    wanted_states: int,
    polynomial_degree: int,
    block_size: int,
    lower_bound: float,
    upper_bound: float,
    generator: LapackRandom,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply CHEBDAV's restart-limited filtered-subspace cleanup."""

    # chebdav.f90z uses the last ``dim_subsp`` value, which excludes the block
    # filtered at the end of the final loop body.  It keeps one extra Ritz pair
    # iff that source dimension exceeds nwant.
    bounded_source_dimension = min(
        int(source_subspace_dimension),
        basis.shape[1],
    )
    target_columns = (
        wanted_states + 1
        if bounded_source_dimension > wanted_states
        and basis.shape[1] > wanted_states
        else wanted_states
    )

    if bounded_source_dimension < wanted_states:
        # PARSEC assumes its normal iteration/allocation invariant and would
        # consume uninitialized workspace if an artificially tiny iteration
        # limit violated it.  Python makes that exceptional small-test path
        # deterministic by explicitly initializing the missing directions.
        missing = wanted_states - bounded_source_dimension
        basis[:, bounded_source_dimension:wanted_states] = (
            generator.uniform_minus_1_1(
                (basis.shape[0], missing),
                column_major=True,
            )
        )

    active_count = target_columns - locked_count
    if active_count <= 0:
        order = np.argsort(locked_values[:locked_count])
        return (
            locked_values[:locked_count][order][:wanted_states].copy(),
            basis[:, :locked_count][:, order][:, :wanted_states].copy(),
            locked_residuals[:locked_count][order][:wanted_states].copy(),
            0,
        )

    active = basis[:, locked_count:target_columns]
    active[:, :] = chebff_filter(
        operator,
        active,
        degree=polynomial_degree,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        reference_eigenvalue=lower_bound,
        block_size=block_size,
        reset_recurrence_per_block=False,
    )
    _orthonormalize_appended(
        basis,
        locked_count,
        active_count,
        generator,
    )
    active_values, active_vectors, active_residuals = _rayleigh_ritz_upper(
        operator,
        basis[:, locked_count:target_columns],
    )

    values = np.concatenate(
        (locked_values[:locked_count], active_values)
    )
    vectors = np.concatenate(
        (basis[:, :locked_count], active_vectors),
        axis=1,
    )
    residuals = np.concatenate(
        (locked_residuals[:locked_count], active_residuals)
    )
    order = np.argsort(values, kind="stable")[:wanted_states]
    return (
        np.asarray(values[order], dtype=float),
        np.asarray(vectors[:, order], dtype=float),
        np.asarray(residuals[order], dtype=float),
        # PARSEC's printed num_mv includes the filter but omits the final
        # ZmatvecB used for this Rayleigh--Ritz projection.
        polynomial_degree * active_count,
    )


def run_chebdav(
    operator: Any,
    wanted_states: int,
    *,
    settings: ChebDavSettings = ChebDavSettings(),
) -> ChebDavResult:
    """Compute the lowest states with PARSEC's first-SCF CHEBDAV policy.

    ``wanted_states`` is the complete working count: physical states plus
    ``Subspace_Buffer_Size``.  The fixed 12-column Davidson work window is
    allocated separately and is not returned to occupations or SUBSPACE.

    Small matrices are safely capped at their actual dimension.  Production
    PARSEC assumes a grid far larger than its block/work dimensions and can
    otherwise raise an internal subspace size beyond caller allocation.
    """

    dimension = _operator_dimension(operator)
    wanted_states = int(wanted_states)
    if not 1 <= wanted_states <= dimension:
        raise ValueError(
            f"wanted_states must be between 1 and {dimension}, got {wanted_states}"
        )

    # PARSEC's input default is at most six and ordinarily much smaller than
    # the grid.  These caps only make the direct tiny-matrix primitive safe.
    block_size = min(
        int(settings.block_size),
        wanted_states,
        max(1, dimension // 3),
    )
    maximum_subspace_dimension = min(
        dimension,
        wanted_states + int(settings.workspace_window),
    )
    active_limit = _active_subspace_limit(
        wanted_states,
        block_size,
        maximum_subspace_dimension,
    )
    iteration_limit = (
        _source_max_iterations(wanted_states, block_size, active_limit)
        if settings.max_iterations is None
        else int(settings.max_iterations)
    )

    basis_generator = LapackRandom()
    bound_generator = np.random.default_rng(settings.random_seed)
    bound = lanczos_upper_bound(
        operator,
        steps=settings.lanczos_steps,
        rng=bound_generator,
    )
    upper_bound = float(bound.upper_bound)
    lower_bound = _initial_lower_bound(bound.lower_bound, upper_bound)
    if not lower_bound < upper_bound:
        raise RuntimeError("CHEBDAV estimated an invalid filter interval")

    basis = np.zeros((dimension, maximum_subspace_dimension), dtype=float)
    applied_basis = np.zeros((dimension, active_limit), dtype=float)
    projected = np.zeros((active_limit, active_limit), dtype=float)
    locked_values = np.empty(maximum_subspace_dimension, dtype=float)
    locked_residuals = np.empty(maximum_subspace_dimension, dtype=float)

    initial = basis_generator.uniform_minus_1_1(
        (dimension, block_size),
        column_major=True,
    )
    basis[:, :block_size] = chebyshev_filter(
        operator,
        initial,
        degree=settings.polynomial_degree,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        reference_eigenvalue=lower_bound,
    )
    _orthonormalize_appended(basis, 0, block_size, basis_generator)

    matrix_vector_products = bound.steps + settings.polynomial_degree * block_size
    locked_count = 0
    retained_active = 0
    # Mirrors Fortran ``dim_subsp`` at the point used by approximate cleanup.
    # At the end of a normal loop body it excludes the just-filtered next block.
    cleanup_subspace_dimension = block_size
    running_ritz_max = 0.0
    inner_restarts = 0
    outer_restarts = 0
    records: list[ChebDavIteration] = []
    complete = False
    last_ritz_values = np.empty(0, dtype=float)
    last_full_active_size = 0
    last_newly_locked = 0

    # Fortran writes ``do while (num_iter <= max_iter)`` and increments at the
    # top, so it can execute max_iter+1 projected solves.
    for iteration in range(1, iteration_limit + 2):
        locked_before = locked_count
        new_start = locked_count + retained_active
        new_stop = new_start + block_size
        if new_stop > maximum_subspace_dimension:
            # The source reaches this situation through its outer-restart
            # branch.  Treat it as a restart-limited cleanup rather than ever
            # writing outside the owned Python allocation.
            break

        new_block = basis[:, new_start:new_stop]
        applied_new = _apply(operator, new_block)
        applied_basis[:, retained_active : retained_active + block_size] = applied_new
        matrix_vector_products += block_size

        full_active_size = retained_active + block_size
        # chebdav.f90z sets dim_subsp before Rayleigh--Ritz and deliberately
        # retains that pre-inner-restart lifecycle value for the outer-restart
        # exit/cap decisions below.
        source_subspace_dimension = locked_before + full_active_size
        cleanup_subspace_dimension = source_subspace_dimension
        active_basis = basis[
            :, locked_count : locked_count + full_active_size
        ]
        last_columns = active_basis.T @ applied_new
        # PARSEC fills the new columns and calls DSYEV('U'); entries below the
        # diagonal are not authoritative.  Copy only each new column's upper
        # triangle, then mirror it for NumPy's full symmetric eigensolver.
        for local_column in range(block_size):
            projected_column = retained_active + local_column
            projected[: projected_column + 1, projected_column] = (
                last_columns[: projected_column + 1, local_column]
            )
        upper = np.triu(projected[:full_active_size, :full_active_size])
        small = upper + np.triu(upper, 1).T
        ritz_values, rotations = np.linalg.eigh(small)
        last_ritz_values = np.asarray(ritz_values, dtype=float)
        last_full_active_size = full_active_size

        inner_restart = full_active_size + block_size > active_limit
        if inner_restart:
            retained_after_rr = min(
                _inner_restart_size(active_limit, block_size),
                full_active_size,
            )
            inner_restarts += 1
        else:
            retained_after_rr = full_active_size

        # Rotate Q and cached HQ by the same projected eigenvectors.  Keeping
        # HQ avoids reapplying H to all old active columns in the next cycle.
        basis[:, locked_count : locked_count + retained_after_rr] = (
            active_basis @ rotations[:, :retained_after_rr]
        )
        applied_basis[:, :retained_after_rr] = (
            applied_basis[:, :full_active_size]
            @ rotations[:, :retained_after_rr]
        )

        running_ritz_max = max(
            running_ritz_max,
            abs(float(ritz_values[0])),
            abs(float(ritz_values[-1])),
        )
        residual_tolerance = settings.convergence_tolerance * running_ritz_max
        if locked_before > wanted_states // 2:
            residual_tolerance *= 2.0 * (locked_before + 1) / wanted_states

        tested: list[float] = []
        reordered = False
        for active_index in range(retained_after_rr):
            vector_column = locked_before + active_index
            residual = (
                applied_basis[:, active_index]
                - basis[:, vector_column] * ritz_values[active_index]
            )
            residual_norm = float(np.linalg.norm(residual))
            tested.append(residual_norm)
            if residual_norm >= residual_tolerance:
                break
            locked_count, pair_reordered = _insert_locked_pair(
                locked_values,
                locked_residuals,
                basis,
                locked_count,
                float(ritz_values[active_index]),
                residual_norm,
                vector_column,
            )
            reordered = reordered or pair_reordered

        newly_locked = locked_count - locked_before
        last_newly_locked = newly_locked
        if (locked_count >= wanted_states and not reordered) or (
            locked_count >= wanted_states + 1
        ):
            records.append(
                ChebDavIteration(
                    number=iteration,
                    locked_before=locked_before,
                    locked_after=locked_count,
                    active_dimension_before_restart=full_active_size,
                    active_dimension_retained=retained_after_rr,
                    residual_tolerance=float(residual_tolerance),
                    tested_residual_norms=np.asarray(tested),
                    inner_restart=inner_restart,
                    outer_restart=False,
                )
            )
            complete = True
            break

        active_after_lock = retained_after_rr - newly_locked
        # The source tests nconv+hsizerst before subtracting the newly locked
        # prefix from hsizerst.  This intentionally triggers a restart a little
        # earlier when several Ritz pairs lock in one iteration.
        outer_restart_test_dimension = locked_count + retained_after_rr
        outer_restart = (
            outer_restart_test_dimension
            > maximum_subspace_dimension - block_size
        )
        if outer_restart and (
            outer_restarts > settings.max_outer_restarts
            and source_subspace_dimension >= wanted_states
        ):
            # The source exits before deflating hsizerst and later feeds this
            # same pre-restart dim_subsp value to approximate cleanup.
            cleanup_subspace_dimension = source_subspace_dimension
            records.append(
                ChebDavIteration(
                    number=iteration,
                    locked_before=locked_before,
                    locked_after=locked_count,
                    active_dimension_before_restart=full_active_size,
                    active_dimension_retained=active_after_lock,
                    residual_tolerance=float(residual_tolerance),
                    tested_residual_norms=np.asarray(tested),
                    inner_restart=inner_restart,
                    outer_restart=True,
                )
            )
            break

        if outer_restart:
            # Source: dim_subsp=min(n_spdim-2*b, old_dim_subsp), followed by
            # removal of the newly locked active prefix.
            restart_total = min(
                maximum_subspace_dimension - 2 * block_size,
                source_subspace_dimension,
            )
            restart_total = max(locked_count, restart_total)
            requested_active_after_restart = restart_total - locked_count
            # For normal PARSEC-sized problems this min is an equality.  It is
            # a finite-N safety cap when source dimensions would otherwise ask
            # Python to reuse active H images discarded by an inner restart.
            active_after_lock = min(
                active_after_lock,
                requested_active_after_restart,
            )
            outer_restarts += 1

        if newly_locked:
            applied_basis[:, :active_after_lock] = applied_basis[
                :, newly_locked : newly_locked + active_after_lock
            ].copy()
        retained_active = active_after_lock
        cleanup_subspace_dimension = locked_count + retained_active

        projected[:retained_active, :retained_active] = 0.0
        if retained_active:
            diagonal_values = ritz_values[
                newly_locked : newly_locked + retained_active
            ]
            projected[
                np.arange(retained_active), np.arange(retained_active)
            ] = diagonal_values

        if retained_active < block_size:
            # Production PARSEC normally retains at least one full active
            # block.  A tiny/problematic matrix can exhaust it; inject a new
            # LAPACK-stream block rather than reading stale columns.
            candidates = basis_generator.uniform_minus_1_1(
                (dimension, block_size),
                column_major=True,
            )
            cutoff_base = (
                float(locked_values[locked_count - 1])
                if locked_count
                else lower_bound
            )
        else:
            candidates = basis[:, locked_count : locked_count + block_size].copy()
            first_nonconverged = float(ritz_values[newly_locked])
            cutoff_base = (
                max(first_nonconverged, float(locked_values[locked_count - 1]))
                if locked_count
                else first_nonconverged
            )

        second_largest = float(ritz_values[max(0, full_active_size - 2)])
        lower_bound = (cutoff_base + 6.0 * second_largest) / 7.0
        if lower_bound >= upper_bound:
            lower_bound = np.nextafter(upper_bound, -np.inf)

        append_start = locked_count + retained_active
        append_stop = append_start + block_size
        if append_stop > maximum_subspace_dimension:
            break
        basis[:, append_start:append_stop] = chebyshev_filter(
            operator,
            candidates,
            degree=settings.polynomial_degree,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            reference_eigenvalue=lower_bound,
        )
        matrix_vector_products += settings.polynomial_degree * block_size
        _orthonormalize_appended(
            basis,
            append_start,
            block_size,
            basis_generator,
        )

        records.append(
            ChebDavIteration(
                number=iteration,
                locked_before=locked_before,
                locked_after=locked_count,
                active_dimension_before_restart=full_active_size,
                active_dimension_retained=retained_active,
                residual_tolerance=float(residual_tolerance),
                tested_residual_norms=np.asarray(tested),
                inner_restart=inner_restart,
                outer_restart=outer_restart,
            )
        )

    true_converged = min(locked_count, wanted_states)
    approximate_cleanup_used = not complete

    if complete:
        order = np.argsort(locked_values[:locked_count], kind="stable")[:wanted_states]
        eigenvalues = locked_values[:locked_count][order].copy()
        vectors = basis[:, :locked_count][:, order].copy()
        residual_norms = locked_residuals[:locked_count][order].copy()
    else:
        if last_ritz_values.size:
            first_index = min(last_newly_locked, last_ritz_values.size - 1)
            cutoff_base = float(last_ritz_values[first_index])
            if locked_count:
                cutoff_base = max(
                    cutoff_base,
                    float(locked_values[locked_count - 1]),
                )
            second_largest = float(
                last_ritz_values[max(0, last_full_active_size - 2)]
            )
            lower_bound = (cutoff_base + 6.0 * second_largest) / 7.0
            if lower_bound >= upper_bound:
                lower_bound = np.nextafter(upper_bound, -np.inf)
        eigenvalues, vectors, residual_norms, cleanup_mv = _final_approximate_subspace(
            operator,
            basis,
            locked_values,
            locked_residuals,
            locked_count,
            cleanup_subspace_dimension,
            wanted_states,
            settings.polynomial_degree,
            block_size,
            lower_bound,
            upper_bound,
            basis_generator,
        )
        matrix_vector_products += cleanup_mv

    state = ChebDavState(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        maximum_subspace_dimension=maximum_subspace_dimension,
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        truly_converged_states=true_converged,
        approximate_cleanup_used=approximate_cleanup_used,
        matrix_vector_products=matrix_vector_products,
        iterations_completed=len(records),
        inner_restarts=inner_restarts,
        outer_restarts=outer_restarts,
        filter_lower_bound=float(lower_bound),
        spectral_upper_bound=float(upper_bound),
    )
    return ChebDavResult(
        eigenvalues=eigenvalues,
        vectors=vectors,
        residual_norms=residual_norms,
        state=state,
        lanczos_bound=bound,
        iterations=tuple(records),
    )


__all__ = [
    "ChebDavIteration",
    "ChebDavResult",
    "ChebDavSettings",
    "ChebDavState",
    "run_chebdav",
]
