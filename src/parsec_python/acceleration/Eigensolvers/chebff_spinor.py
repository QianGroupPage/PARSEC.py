"""Device eigensolver for complex spinor operators (self-consistent SOC).

The same two-phase policy as :func:`parsec_python.Eigensolvers.eigval.solve_eigval`,
on an array backend (CuPy, or NumPy as the test double) with every large array
resident on the device:

* the first SCF iteration builds ``requested + buffer`` vectors with CHEBFF
  (``filter_cycles`` cycles of filter / orthonormalize / Rayleigh-Ritz);
* every later iteration applies exactly one SUBSPACE step to the saved Ritz
  vectors of the previous iteration (:func:`parsec_python.Eigensolvers.subspace.run_subspace_filter`):
  lower bound from the saved Ritz values, a short Lanczos upper bound, degree
  ``polynomial_degree -/+ degree_delta`` on the lower/upper block groups, one
  orthonormalization and one Rayleigh-Ritz rotation.

Earlier versions of this module ran the full first-solve CHEBFF in every SCF
iteration, about eight times the matrix applications of the reference.

Memory.  The filter runs on chunks of columns, so its temporaries are a few
chunks, not a few copies of the basis; the Rayleigh-Ritz projection is
accumulated chunk by chunk and the rotation overwrites the basis in place.  The
peak is the filtered block plus its QR factor.  A chunk is
``PARSEC_SPINOR_FILTER_COLUMNS`` columns (default 64).

Everything uses only operations NumPy and CuPy share, so the tests run the
exact code of the GPU with NumPy and compare it with the reference.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers.chebff import ChebFFSettings
from parsec_python.Eigensolvers.chebyshev import subspace_filter_blocks
from parsec_python.Eigensolvers.subspace import SubspaceSettings, _adapt_polynomial_degree

from ..backends.cupy import require_cupy

_BREAKDOWN_TOLERANCE = 2.5e-16
FILTER_COLUMNS_ENV = "PARSEC_SPINOR_FILTER_COLUMNS"
_ROW_CHUNK = 1 << 16


def _filter_columns() -> int:
    value = os.environ.get(FILTER_COLUMNS_ENV, "").strip()
    return max(1, int(value)) if value.isdigit() else 64


@dataclass
class SpinorEigvalState:
    """Saved subspace: the next SCF iteration filters these Ritz vectors.

    ``vectors`` stays on the device (``(dimension, working_states)``) and is
    consumed by the next solve, which replaces it.
    """

    vectors: Any
    eigenvalues: np.ndarray
    working_states: int
    operator_dimension: int
    first_filter: bool = True
    filter_lower_bound: float | None = None
    filters_completed: int = 0


# Name kept for the earlier state type.
ChebFFSpinorState = SpinorEigvalState


@dataclass
class SpinorEigvalResult:
    """Duck-typed like ``solve_eigval``'s result; ``vectors`` is a device view."""

    eigenvalues: np.ndarray
    vectors: Any
    state: SpinorEigvalState
    stats: dict[str, float] = field(default_factory=dict)
    residual_norms: np.ndarray | None = None
    """``||H x - e x||`` of the requested Ritz pairs when ``converge_first`` measured them."""


ChebFFSpinorResult = SpinorEigvalResult


class _Stats:
    """Matrix-application counts and seconds per phase of one solve."""

    def __init__(self, sync: Callable[[], None] | None) -> None:
        self.values: dict[str, float] = {}
        self._sync = sync

    def add(self, name: str, amount: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + amount

    def timed(self, name: str):
        stats = self

        class _Timer:
            def __enter__(self):
                if stats._sync is not None:
                    stats._sync()
                self.start = time.perf_counter()

            def __exit__(self, *exc):
                if stats._sync is not None:
                    stats._sync()
                stats.add(name, time.perf_counter() - self.start)

        return _Timer()


class _CountingOperator:
    """Counts applications and columns of an operator (``apply`` on device blocks)."""

    def __init__(self, operator: Any, stats: _Stats) -> None:
        self._operator = operator
        self._stats = stats
        self.shape = operator.shape

    def apply(self, block):
        self._stats.add("applications", 1)
        if block.ndim == 1:  # the Lanczos start vectors are single vectors
            self._stats.add("application_columns", 1)
            return self._operator.apply(block[:, None])[:, 0]
        self._stats.add("application_columns", int(block.shape[1]))
        return self._operator.apply(block)


# --------------------------------------------------------------------- primitives
def _lanczos_upper_bound(cp, operator, dimension: int, steps: int, rng: np.random.Generator):
    """Device counterpart of
    :func:`parsec_python.Eigensolvers.spectral_bounds.lanczos_upper_bound`."""

    target_steps = min(max(int(steps), 4), 8, dimension)
    host_start = rng.random(dimension) + 1j * rng.random(dimension)
    vector = cp.asarray(host_start, dtype=cp.complex128)
    vector = vector / cp.linalg.norm(vector)

    projected = np.zeros((target_steps, target_steps), dtype=float)
    residual = operator.apply(vector)
    alpha = float(cp.real(cp.vdot(vector, residual)))
    residual = residual - alpha * vector
    projected[0, 0] = alpha

    actual_steps = 1
    raw_beta = float(cp.linalg.norm(residual))
    for column in range(1, target_steps):
        raw_beta = float(cp.linalg.norm(residual))
        if raw_beta <= _BREAKDOWN_TOLERANCE:
            break
        previous = vector
        vector = residual / raw_beta
        residual = operator.apply(vector) - raw_beta * previous
        alpha = float(cp.real(cp.vdot(vector, residual)))
        residual = residual - alpha * vector
        projected[column, column - 1] = raw_beta
        projected[column - 1, column] = raw_beta
        projected[column, column] = alpha
        actual_steps = column + 1

    ritz_values = np.linalg.eigvalsh(projected[:actual_steps, :actual_steps])
    lower_bound = float(ritz_values[0])
    spectral_radius = max(float(ritz_values[-1]), abs(lower_bound))
    residual_scale = raw_beta
    if residual_scale < 1.0e-2:
        residual_scale *= 10.0
    elif residual_scale < 1.0e-1:
        residual_scale *= 5.0
    upper_bound = spectral_radius + residual_scale
    return float(upper_bound), lower_bound


def _chebyshev_filter(
    cp,
    operator,
    block,
    degree: int,
    lower_bound: float,
    upper_bound: float,
    reference_eigenvalue: float,
):
    """Device counterpart of
    :func:`parsec_python.Eigensolvers.chebyshev.chebyshev_filter` (one block, one
    polynomial for every column)."""

    half_span = 0.5 * (upper_bound - lower_bound)
    if half_span <= 0.0:
        raise ValueError("upper_bound must be greater than lower_bound")
    center = 0.5 * (upper_bound + lower_bound)
    denominator = reference_eigenvalue - center
    if denominator == 0.0:
        raise ValueError("reference_eigenvalue cannot equal the interval center")

    sigma_one = half_span / denominator
    sigma = sigma_one
    previous = block.copy()
    current = operator.apply(block)
    current = (current - center * block) * (sigma_one / half_span)

    for _ in range(2, int(degree) + 1):
        sigma_next = 1.0 / (2.0 / sigma_one - sigma)
        following = operator.apply(current)
        following = (
            (following - center * current) * (2.0 / half_span) - sigma * previous
        )
        following = following * sigma_next
        previous, current = current, following
        sigma = sigma_next
    return current


def _orthonormalize(cp, vectors):
    basis, _ = cp.linalg.qr(vectors)
    return basis


def _rayleigh_ritz(cp, operator, basis):
    """Device counterpart of
    :func:`parsec_python.Eigensolvers.rayleigh_ritz.rayleigh_ritz` (whole basis at once)."""

    applied = operator.apply(basis)
    raw_projection = basis.conj().T @ applied
    lower = cp.tril(raw_projection)
    hermitian = lower + cp.tril(lower, -1).conj().T
    eigenvalues, rotations = cp.linalg.eigh(hermitian)
    wavefunctions = basis @ rotations
    return eigenvalues, wavefunctions


def _rayleigh_ritz_in_place(xp, operator, basis, chunk: int):
    """Rayleigh-Ritz with the projection accumulated by column chunks and the
    rotation written over ``basis``.  Returns the Ritz values (host)."""

    dimension, count = basis.shape
    projection = xp.empty((count, count), dtype=xp.complex128)
    for c0 in range(0, count, chunk):
        c1 = min(c0 + chunk, count)
        applied = operator.apply(xp.ascontiguousarray(basis[:, c0:c1]))
        for r0 in range(0, count, chunk):
            r1 = min(r0 + chunk, count)
            projection[r0:r1, c0:c1] = xp.ascontiguousarray(basis[:, r0:r1]).conj().T @ applied
        del applied
    lower = xp.tril(projection)
    hermitian = lower + xp.tril(lower, -1).conj().T
    eigenvalues, rotations = xp.linalg.eigh(hermitian)
    for r0 in range(0, dimension, _ROW_CHUNK):
        r1 = min(r0 + _ROW_CHUNK, dimension)
        basis[r0:r1] = basis[r0:r1] @ rotations
    return np.asarray(_to_host(xp, eigenvalues), dtype=np.float64)


def _residual_norms(xp, operator, vectors, eigenvalues: np.ndarray, chunk: int) -> np.ndarray:
    """``||H y_j - e_j y_j||`` of the columns of ``vectors`` (host array), chunk by chunk."""
    count = int(vectors.shape[1])
    norms = np.empty(count, dtype=np.float64)
    for c0 in range(0, count, chunk):
        c1 = min(c0 + chunk, count)
        block = xp.ascontiguousarray(vectors[:, c0:c1])
        applied = operator.apply(block)
        applied -= block * xp.asarray(np.asarray(eigenvalues[c0:c1], dtype=np.float64))[None, :]
        squares = applied.real * applied.real + applied.imag * applied.imag
        norms[c0:c1] = np.sqrt(np.asarray(_to_host(xp, squares.sum(axis=0)), dtype=np.float64))
    return norms


def _to_host(xp, array):
    get = getattr(array, "get", None)
    return get() if get is not None else np.asarray(array)


def _chain(sigma_one: float, initial: float, degree: int):
    """Scalar sequence of the normalized recurrence.

    Returns ``(previous, following)`` lists for the steps ``j = 2 .. degree``
    (the ``sigma_(j-1)`` entering the step and the ``sigma_j`` it produces) and
    the final ``sigma``: the loop of
    :func:`parsec_python.Eigensolvers.chebyshev.chebyshev_filter`.
    """
    sigma = initial
    entering, produced = [], []
    for _ in range(2, degree + 1):
        sigma_next = 1.0 / (2.0 / sigma_one - sigma)
        entering.append(sigma)
        produced.append(sigma_next)
        sigma = sigma_next
    return entering, produced, sigma


def _filter_group(
    xp,
    operator,
    source,
    out,
    columns: slice,
    degree: int,
    center: float,
    half_span: float,
    sigma_one: float,
    entering: np.ndarray,
    produced: np.ndarray,
    chunk: int,
):
    """Chebyshev recurrence for ``source[:, columns]`` with a scalar sequence per column.

    ``entering`` and ``produced`` are ``(degree - 1, columns)`` arrays: column
    ``j`` follows the sigma sequence of the memory block it belongs to, so the
    columns of several blocks are filtered together (one wide application per
    step) with the very coefficients the block-by-block reference uses.
    """
    start, stop = columns.start, columns.stop
    for c0 in range(start, stop, chunk):
        c1 = min(c0 + chunk, stop)
        local = slice(c0 - start, c1 - start)
        block = xp.ascontiguousarray(source[:, c0:c1])
        previous = block.copy()
        current = operator.apply(block)
        current -= center * block
        current *= sigma_one / half_span
        del block
        if degree >= 2:
            sigma_before = xp.asarray(entering[:, local])
            sigma_after = xp.asarray(produced[:, local])
            for step in range(degree - 1):
                following = operator.apply(current)
                following -= center * current
                following *= 2.0 / half_span
                following -= sigma_before[step][None, :] * previous
                following *= sigma_after[step][None, :]
                previous, current = current, following
        del previous
        out[:, c0:c1] = current
        del current


def _check_finite(xp, array) -> None:
    if not bool(xp.all(xp.isfinite(array))):
        raise FloatingPointError("Chebyshev recurrence produced nonfinite values")


# ------------------------------------------------------------------- first solve
def _first_solve(xp, operator, working: int, settings: ChebFFSettings, chunk: int, stats: _Stats):
    dimension = int(operator.shape[0])
    rng = xp.random.default_rng(settings.random_seed)
    vectors = xp.empty((dimension, working), dtype=xp.complex128)
    for c0 in range(0, working, chunk):
        c1 = min(c0 + chunk, working)
        vectors[:, c0:c1] = rng.uniform(-1.0, 1.0, (dimension, c1 - c0)) + 1j * rng.uniform(
            -1.0, 1.0, (dimension, c1 - c0)
        )
    bound_rng = np.random.default_rng(settings.random_seed + 1)
    with stats.timed("lanczos_seconds"):
        upper_bound, smallest_ritz = _lanczos_upper_bound(
            xp, operator, dimension, settings.lanczos_steps, bound_rng
        )
    # chebff.f90z: lowb = (lowb0 + (lowb0 + upperb)) / 3
    lower_bound = (2.0 * smallest_ritz + upper_bound) / 3.0

    eigenvalues = np.empty(working, dtype=float)
    for _ in range(settings.filter_cycles):
        half_span = 0.5 * (upper_bound - lower_bound)
        if half_span <= 0.0:
            raise ValueError("upper_bound must be greater than lower_bound")
        center = 0.5 * (upper_bound + lower_bound)
        denominator = smallest_ritz - center
        if denominator == 0.0:
            raise ValueError("reference_eigenvalue cannot equal the interval center")
        sigma_one = half_span / denominator
        entering, produced, _ = _chain(sigma_one, sigma_one, settings.polynomial_degree)
        shape = (len(entering), working)
        with stats.timed("filter_seconds"):
            filtered = xp.empty_like(vectors)
            _filter_group(
                xp,
                operator,
                vectors,
                filtered,
                slice(0, working),
                int(settings.polynomial_degree),
                center,
                half_span,
                sigma_one,
                np.broadcast_to(np.asarray(entering)[:, None], shape),
                np.broadcast_to(np.asarray(produced)[:, None], shape),
                chunk,
            )
            del vectors
        _check_finite(xp, filtered[:, :1])
        with stats.timed("orthonormalize_seconds"):
            vectors = _orthonormalize(xp, filtered)
            del filtered
        with stats.timed("ritz_seconds"):
            eigenvalues = _rayleigh_ritz_in_place(xp, operator, vectors, chunk)
        if eigenvalues.shape != (working,):
            raise RuntimeError("Rayleigh-Ritz changed the CHEBFF working-subspace dimension")
        smallest_ritz = float(eigenvalues[0])
        largest_ritz = float(eigenvalues[-1])
        if largest_ritz >= upper_bound:
            new_upper = largest_ritz + 0.5 * (largest_ritz - upper_bound) + 1.0
            new_lower = min(lower_bound, (3.0 * lower_bound + new_upper) / 4.0)
            lower_bound, upper_bound = new_lower, new_upper
        else:
            lower_bound = min(
                largest_ritz + 0.001 * (upper_bound - smallest_ritz),
                largest_ritz + 0.05 * abs(largest_ritz),
            )
    return vectors, eigenvalues


# ----------------------------------------------------------------- later solves
def _next_filter_lower_bound(state: SpinorEigvalState) -> float:
    largest_saved_ritz = float(np.max(state.eigenvalues))
    if state.first_filter or state.filter_lower_bound is None:
        return max(largest_saved_ritz, 0.0)
    return max(float(state.filter_lower_bound), largest_saved_ritz)


def _subspace_step(xp, operator, state: SpinorEigvalState, settings: SubspaceSettings, chunk: int, stats: _Stats):
    """One later-SCF SUBSPACE update of ``state`` (see the module docstring)."""
    dimension = state.operator_dimension
    working = state.working_states
    lower_bound = _next_filter_lower_bound(state)
    bound_rng = np.random.default_rng(settings.random_seed + state.filters_completed)
    with stats.timed("lanczos_seconds"):
        upper_bound, _ = _lanczos_upper_bound(
            xp, operator, dimension, settings.lanczos_steps, bound_rng
        )
    degree = _adapt_polynomial_degree(settings.polynomial_degree, lower_bound, upper_bound)
    blocks = subspace_filter_blocks(working, settings.block_size, degree, settings.degree_delta)

    half_span = 0.5 * (upper_bound - lower_bound)
    center = 0.5 * (upper_bound + lower_bound)
    sigma_one = half_span / (lower_bound - center)  # the reference is the lower edge: -1
    # Per block: the sigma it starts from.  The reference carries sigma from one block to
    # the next (reset_recurrence_per_block=False); the sequence does not depend on the data.
    per_column_entering: dict[int, list[np.ndarray]] = {}
    per_column_produced: dict[int, list[np.ndarray]] = {}
    carried: float | None = None
    for block in blocks:
        initial = sigma_one if (settings.reset_recurrence_per_block or carried is None) else carried
        entering, produced, final = _chain(sigma_one, initial, block.degree)
        carried = final
        width = block.stop - block.start
        per_column_entering.setdefault(block.degree, []).append(
            np.repeat(np.asarray(entering)[:, None], width, axis=1)
        )
        per_column_produced.setdefault(block.degree, []).append(
            np.repeat(np.asarray(produced)[:, None], width, axis=1)
        )

    with stats.timed("filter_seconds"):
        filtered = xp.empty_like(state.vectors)
        for group_degree in sorted({block.degree for block in blocks}):
            members = [block for block in blocks if block.degree == group_degree]
            start, stop = members[0].start, members[-1].stop
            if sum(block.stop - block.start for block in members) != stop - start:
                raise RuntimeError("blocks of equal degree are not contiguous")
            _filter_group(
                xp,
                operator,
                state.vectors,
                filtered,
                slice(start, stop),
                int(group_degree),
                center,
                half_span,
                sigma_one,
                np.concatenate(per_column_entering[group_degree], axis=1),
                np.concatenate(per_column_produced[group_degree], axis=1),
                chunk,
            )
        state.vectors = None  # consumed
    _check_finite(xp, filtered[:, :1])
    with stats.timed("orthonormalize_seconds"):
        vectors = _orthonormalize(xp, filtered)
        del filtered
    with stats.timed("ritz_seconds"):
        eigenvalues = _rayleigh_ritz_in_place(xp, operator, vectors, chunk)
    if eigenvalues.shape != (working,):
        raise RuntimeError("Rayleigh-Ritz changed the filtered subspace's eigenvalue count")
    return vectors, eigenvalues, lower_bound, degree


# ---------------------------------------------------------------------- driver
def solve_spinor_eigval(
    operator: Any,
    requested_states: int,
    *,
    settings,
    state: SpinorEigvalState | None = None,
    xp=None,
    sync: Callable[[], None] | None = None,
    converge_first: bool = False,
    tolerance: float | None = None,
    max_refinements: int | None = None,
) -> SpinorEigvalResult:
    """One eigensolve of the SCF loop (``solve_eigval``'s policy on the device).

    ``operator`` exposes ``shape`` and ``apply(block)`` on device blocks
    ``(dimension, vectors)``.  ``settings`` is an ``EigvalSettings`` (only its
    ``chebff``, ``subspace`` and ``safety_buffer`` are read).  With ``state=None``
    the first solve (CHEBFF) runs, otherwise one SUBSPACE step.

    ``converge_first`` makes the first solve a *converged* one, the device counterpart of CHEBDAV:
    after the CHEBFF cycles the SUBSPACE step is repeated on the same operator until the residuals
    ``||H x - e x||`` of the requested pairs fall below ``tolerance * max|e|`` (CHEBDAV's relative
    test; default ``settings.chebdav.convergence_tolerance``) or ``max_refinements`` steps (default 8,
    ``PARSEC_KPOINT_REFINE_STEPS``) were taken.  The state it returns continues as an ordinary
    subspace state, and the residuals come back in ``residual_norms``.
    """
    if xp is None:
        xp, _ = require_cupy()
        if sync is None:
            sync = xp.cuda.Device().synchronize
    stats = _Stats(sync)
    counted = _CountingOperator(operator, stats)
    dimension = int(operator.shape[0])
    working = min(int(requested_states) + int(settings.safety_buffer), dimension)
    if not 1 <= requested_states <= dimension:
        raise ValueError(f"requested_states must be between 1 and {dimension}, got {requested_states}")
    chunk = _filter_columns()

    reusable = (
        state is not None
        and state.vectors is not None
        and state.operator_dimension == dimension
        and state.working_states == working
    )
    if not reusable:
        vectors, eigenvalues = _first_solve(xp, counted, working, settings.chebff, chunk, stats)
        new_state = SpinorEigvalState(
            vectors=vectors,
            eigenvalues=eigenvalues,
            working_states=working,
            operator_dimension=dimension,
            first_filter=True,
            filter_lower_bound=None,
            filters_completed=0,
        )
        stats.add("first_solves", 1)
        residuals = None
        if converge_first:
            relative = float(settings.chebdav.convergence_tolerance if tolerance is None else tolerance)
            if max_refinements is None:
                env = os.environ.get("PARSEC_KPOINT_REFINE_STEPS", "").strip()
                max_refinements = int(env) if env.isdigit() else 8
            for step in range(int(max_refinements) + 1):
                with stats.timed("residual_seconds"):
                    residuals = _residual_norms(
                        xp, counted, new_state.vectors[:, :requested_states], eigenvalues[:requested_states], chunk
                    )
                scale = max(float(np.max(np.abs(eigenvalues[:requested_states]))), 1.0e-8)
                if residuals.max() <= relative * scale or step == int(max_refinements):
                    break
                vectors, eigenvalues, lower_bound, _ = _subspace_step(
                    xp, counted, new_state, settings.subspace, chunk, stats
                )
                new_state = SpinorEigvalState(
                    vectors=vectors,
                    eigenvalues=eigenvalues,
                    working_states=working,
                    operator_dimension=dimension,
                    first_filter=False,
                    filter_lower_bound=lower_bound,
                    filters_completed=new_state.filters_completed + 1,
                )
                stats.add("refinements", 1)
    else:
        vectors, eigenvalues, lower_bound, degree = _subspace_step(
            xp, counted, state, settings.subspace, chunk, stats
        )
        new_state = SpinorEigvalState(
            vectors=vectors,
            eigenvalues=eigenvalues,
            working_states=working,
            operator_dimension=dimension,
            first_filter=False,
            filter_lower_bound=lower_bound,
            filters_completed=state.filters_completed + 1,
        )
        stats.add("subspace_steps", 1)
        stats.add("polynomial_degree", degree)
    return SpinorEigvalResult(
        eigenvalues=eigenvalues[:requested_states].copy(),
        vectors=new_state.vectors[:, :requested_states],
        state=new_state,
        stats=dict(stats.values),
        residual_norms=residuals if not reusable else None,
    )


def run_chebff_spinor(operator, wanted_states, *, settings=None, state=None, xp=None, sync=None):
    """First-solve-only entry point kept for earlier callers (CHEBFF cycles)."""
    from parsec_python.Eigensolvers.eigval import EigvalSettings

    eig = EigvalSettings(
        safety_buffer=0,
        initial_method="chebff",
        chebff=settings if settings is not None else ChebFFSettings(),
    )
    return solve_spinor_eigval(operator, wanted_states, settings=eig, state=None, xp=xp, sync=sync)


def cupy_spinor_eigenproblem_solver(
    operator: Any,
    requested_states: int,
    *,
    settings,
    state: SpinorEigvalState | None = None,
) -> SpinorEigvalResult:
    """``eigenproblem_solver`` hook of
    :func:`parsec_python.SCF.self_consistent_soc.run_self_consistent_soc`."""
    return solve_spinor_eigval(operator, requested_states, settings=settings, state=state)


def spinor_density_on_device(xp, spinors, occupations: np.ndarray, volume_element: float) -> np.ndarray:
    """``rho = (1/dV) sum_n f_n (|psi_n,up|^2 + |psi_n,down|^2)`` computed where the
    spinors are; only the grid-length density comes back to the host."""
    dimension = int(spinors.shape[0])
    n = dimension // 2
    occupations = np.asarray(occupations, dtype=np.float64)
    total = xp.zeros(dimension, dtype=xp.float64)
    chunk = _filter_columns()
    for c0 in range(0, occupations.size, chunk):
        c1 = min(c0 + chunk, occupations.size)
        if not np.any(occupations[c0:c1]):
            continue
        block = spinors[:, c0:c1]
        total += (block.real * block.real + block.imag * block.imag) @ xp.asarray(occupations[c0:c1])
    density = total[:n] + total[n:]
    return np.asarray(_to_host(xp, density), dtype=np.float64) / volume_element


__all__ = [
    "ChebFFSpinorResult",
    "ChebFFSpinorState",
    "SpinorEigvalResult",
    "SpinorEigvalState",
    "cupy_spinor_eigenproblem_solver",
    "run_chebff_spinor",
    "solve_spinor_eigval",
    "spinor_density_on_device",
]
