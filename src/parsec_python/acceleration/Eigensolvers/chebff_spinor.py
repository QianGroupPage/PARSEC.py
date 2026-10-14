"""Correctness-first CuPy CHEBFF for complex spinor/SOC operators.

Deliberately not an extension of the production real-valued
``acceleration.Eigensolvers.chebff``/``chebdav``/``rayleigh_ritz``/
``orthogonalize`` stack (876+334+182 lines of hand-tuned real float64
machinery, entangled with mixed-precision and symmetry-sector code this
module does not need). Instead this is a small, self-contained Chebyshev-
filtered-subspace-iteration eigensolver -- the same numerical method PARSEC
uses, line-for-line structurally mirroring the *reference* (CPU) module
:mod:`parsec_python.Eigensolvers.chebff` plus the Chebyshev recurrence in
:mod:`parsec_python.Eigensolvers.chebyshev`, the bound estimator in
:mod:`parsec_python.Eigensolvers.spectral_bounds`, and the Rayleigh-Ritz
projection in :mod:`parsec_python.Eigensolvers.rayleigh_ritz` -- but built
from CuPy array operations (``cupy.linalg.qr`` for orthonormalization,
``cupy.linalg.eigh`` for the small projected-Hermitian-matrix diagonalization,
both of which support complex128 natively) so vectors stay device-resident
across the whole filter/orthonormalize/Rayleigh-Ritz cycle.

Two deliberate simplifications relative to the reference CHEBFF, both
explicitly noted in its own docstring as not affecting correctness:

* The trial subspace is an ordinary ``cupy`` random complex draw, not
  PARSEC's bit-exact ``DLARNV``-translated stream (the reference docstring:
  "Chebyshev filtering converges to the true subspace regardless of the
  specific trial-vector seed").
* Orthonormalization is a plain ``cupy.linalg.qr`` call rather than the
  reference's modified-Gram-Schmidt-with-random-replacement-on-breakdown
  primitive (:mod:`parsec_python.Eigensolvers.orthogonalize`) -- QR produces
  a mathematically equivalent orthonormal basis for a full-rank input, and a
  rank-deficient filtered block is not expected from a well-posed SCF
  Hamiltonian.

This also does not yet replicate ``solve_eigval``'s two-phase dispatch
(an expensive first CHEBFF solve, then a cheaper SUBSPACE filter reusing the
previous iteration's Ritz vectors on later SCF iterations): every call here
runs the same fixed number of CHEBFF cycles. This is slower per SCF
iteration than the production two-phase approach but not incorrect --
``run_self_consistent_soc``'s SCF loop still carries this module's own
warm-start ``state`` (the previous iteration's filtered vectors and filter
bounds) from one call to the next, so work is not fully repeated from
scratch either. Matching ``solve_eigval``'s full two-phase/mixed-precision
production behavior is explicit follow-up work once this baseline is
validated on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from parsec_python.Eigensolvers.chebff import ChebFFSettings

from ..backends.cupy import require_cupy

_BREAKDOWN_TOLERANCE = 2.5e-16


@dataclass
class ChebFFSpinorState:
    """Warm-start state threaded between SCF iterations.

    ``vectors`` is host ``complex128`` (not device-resident) so it can be
    stored by the reference SCF loop's plain ``eigval_state`` variable
    between calls without that loop needing to know anything about CuPy.
    """

    vectors: np.ndarray
    lower_bound: float
    upper_bound: float
    smallest_ritz: float


@dataclass
class ChebFFSpinorResult:
    """Return shape matching :func:`parsec_python.Eigensolvers.eigval.solve_eigval`'s
    duck-typed result (``.eigenvalues``, ``.vectors``, ``.state``)."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    state: ChebFFSpinorState


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
    :func:`parsec_python.Eigensolvers.chebyshev.chebyshev_filter`."""

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
    :func:`parsec_python.Eigensolvers.rayleigh_ritz.rayleigh_ritz`."""

    applied = operator.apply(basis)
    raw_projection = basis.conj().T @ applied
    lower = cp.tril(raw_projection)
    hermitian = lower + cp.tril(lower, -1).conj().T
    eigenvalues, rotations = cp.linalg.eigh(hermitian)
    wavefunctions = basis @ rotations
    return eigenvalues, wavefunctions


def run_chebff_spinor(
    operator: Any,
    wanted_states: int,
    *,
    settings: ChebFFSettings = ChebFFSettings(),
    state: ChebFFSpinorState | None = None,
) -> ChebFFSpinorResult:
    """CuPy Chebyshev-filtered-subspace-iteration solve for a complex
    Hermitian ``operator`` (``.shape``, ``.apply(block)``).

    ``state``, when given, resumes from the previous call's filtered
    vectors and filter bounds instead of drawing a new random trial
    subspace and re-estimating the spectral bound -- the same warm-start
    role ``solve_eigval``'s ``EigvalState`` plays for the reference/real
    accelerated paths.
    """

    cp, _ = require_cupy()
    dimension = int(operator.shape[0])
    if not 1 <= wanted_states <= dimension:
        raise ValueError(
            f"wanted_states must be between 1 and {dimension}, got {wanted_states}"
        )

    if state is not None and state.vectors.shape == (dimension, wanted_states):
        vectors = cp.asarray(state.vectors, dtype=cp.complex128)
        lower_bound = state.lower_bound
        upper_bound = state.upper_bound
        smallest_ritz = state.smallest_ritz
    else:
        trial_rng = np.random.default_rng(settings.random_seed)
        host_trial = trial_rng.uniform(-1.0, 1.0, (dimension, wanted_states)) + 1j * (
            trial_rng.uniform(-1.0, 1.0, (dimension, wanted_states))
        )
        vectors = cp.asarray(host_trial, dtype=cp.complex128)
        bound_rng = np.random.default_rng(settings.random_seed + 1)
        upper_bound, smallest_ritz = _lanczos_upper_bound(
            cp, operator, dimension, settings.lanczos_steps, bound_rng
        )
        # chebff.f90z: lowb = (lowb0 + (lowb0 + upperb)) / 3
        lower_bound = (2.0 * smallest_ritz + upper_bound) / 3.0

    eigenvalues = np.empty(wanted_states, dtype=float)
    for _ in range(settings.filter_cycles):
        vectors = _chebyshev_filter(
            cp,
            operator,
            vectors,
            settings.polynomial_degree,
            lower_bound,
            upper_bound,
            smallest_ritz,
        )
        vectors = _orthonormalize(cp, vectors)
        eigenvalues_device, vectors = _rayleigh_ritz(cp, operator, vectors)
        eigenvalues = cp.asnumpy(eigenvalues_device).astype(np.float64)
        if eigenvalues.shape != (wanted_states,):
            raise RuntimeError(
                "Rayleigh-Ritz changed the CHEBFF working-subspace dimension"
            )
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

    vectors_host = cp.asnumpy(vectors).astype(np.complex128)
    new_state = ChebFFSpinorState(
        vectors=vectors_host,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        smallest_ritz=smallest_ritz,
    )
    return ChebFFSpinorResult(
        eigenvalues=eigenvalues, vectors=vectors_host, state=new_state
    )


def cupy_spinor_eigenproblem_solver(
    operator: Any,
    requested_states: int,
    *,
    settings,
    state: ChebFFSpinorState | None = None,
) -> ChebFFSpinorResult:
    """``eigenproblem_solver`` DI hook for
    :func:`parsec_python.SCF.self_consistent_soc.run_self_consistent_soc`
    (and its spin-polarized counterpart).  ``settings`` is the full
    ``EigvalSettings`` the reference SCF loop builds each iteration; only
    ``settings.chebff`` is used, matching the reference's own restriction of
    self-consistent SOC to ``eigensolver.method == 'chebff'``.
    """

    return run_chebff_spinor(
        operator, requested_states, settings=settings.chebff, state=state
    )


__all__ = [
    "ChebFFSpinorResult",
    "ChebFFSpinorState",
    "cupy_spinor_eigenproblem_solver",
    "run_chebff_spinor",
]
