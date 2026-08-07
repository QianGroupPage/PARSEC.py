"""Short-Lanczos spectral bounds used by PARSEC's Chebyshev solvers.

Fortran provenance
------------------
PARSEC commit ``f432777750c2efc633adeb26faff641500b39b4a``,
``src/lancz_bound.f90z``, real non-``BETA`` routine ``lancz_bound``.

This module translates the serial numerical recurrence only.  PARSEC's MPI
reductions and distributed leading dimensions become ordinary NumPy inner
products.  Starting from a normalized vector ``q1``, it builds a small real
symmetric tridiagonal matrix ``T`` through

``w_j = H*q_j - beta_(j-1)*q_(j-1)``

``alpha_j = q_j.H @ w_j``

``f_j = w_j - alpha_j*q_j``

``beta_j = ||f_j||``, ``q_(j+1) = f_j/beta_j``.

No Lanczos reorthogonalization is performed.  Requested step counts are
clamped to the inclusive range four through eight, and the final upper-bound
correction uses the ``beta`` that *started* the last Lanczos step, not the norm
of the residual produced after that last step.

There are two documented safety deviations from literal Fortran behavior.
Python also caps the step count at the matrix dimension, whereas PARSEC assumes
``n >> k``.  If the recurrence breaks down early, Python diagonalizes only the
tridiagonal entries that were actually constructed.  The non-``BETA`` Fortran
source instead diagonalizes a fixed leading ``(k-1)`` block, which can include
zero/unfilled rows when breakdown occurs before the final step.  The ordinary
no-breakdown calculation follows the Fortran recurrence and bound formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_BREAKDOWN_TOLERANCE = 2.5e-16


@dataclass(frozen=True)
class LanczosBoundResult:
    """Result of PARSEC's non-``BETA`` short-Lanczos bound estimator.

    ``ritz_values`` are the eigenvalues of the constructed tridiagonal
    projection.  ``lower_bound`` is their smallest value.  ``upper_bound`` is
    PARSEC's empirical spectral-radius estimate plus a ``beta`` safety margin;
    it is the upper endpoint supplied to the Chebyshev filter, not a
    residual-certified eigenvalue bound.  ``raw_beta`` is the unscaled value
    and ``residual_scale`` is the piecewise enlarged margin actually added.
    """

    upper_bound: float
    lower_bound: float
    middle: float
    ritz_values: np.ndarray
    raw_beta: float
    residual_scale: float
    steps: int
    requested_steps: int
    breakdown: bool


def _square_operator_size(operator: Any) -> int:
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square two-dimensional shape")
    size = int(shape[0])
    if size < 1:
        raise ValueError("operator dimension must be positive")
    return size


def _apply(operator: Any, vector: np.ndarray, size: int) -> np.ndarray:
    result = np.asarray(operator @ vector)
    if result.shape != (size,):
        raise ValueError("operator must preserve the shape of an input vector")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("operator application produced nonfinite values")
    return result


def _random_start(size: int, rng: np.random.Generator) -> np.ndarray:
    """Match real Fortran ``random_number`` in distribution, not bit stream.

    PARSEC uses the compiler/runtime intrinsic random state.  NumPy provides
    the same uniform ``[0,1)`` type of starting vector but cannot reproduce
    that implementation-dependent sequence from a portable Python seed.
    """

    return rng.random(size)


def lanczos_upper_bound(
    operator: Any,
    initial_vector: np.ndarray | None = None,
    steps: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """Estimate the full spectral radius with PARSEC's short Lanczos run.

    Let ``theta_min`` and ``theta_max`` be the extreme eigenvalues of the
    small Lanczos tridiagonal.  The non-``BETA`` routine first forms

    ``rho = max(theta_max, abs(theta_min))``

    and reports ``lower_bound = theta_min`` and

    ``middle = (rho + theta_min)/2``.

    It then enlarges the positive spectral endpoint using the retained
    recurrence coefficient ``beta``:

    ``margin = 10*beta`` if ``beta < 1e-2``;

    ``margin = 5*beta`` if ``1e-2 <= beta < 1e-1``;

    ``margin = beta`` otherwise;

    ``upper_bound = rho + margin``.

    This is PARSEC's inexpensive conservative estimate for filter design.  It
    is not tested as a rigorous enclosure of every eigenvalue.

    Parameters
    ----------
    operator
        Square real or complex Hermitian operator supporting ``operator @ x``.
    initial_vector
        Optional deterministic starting vector.  If omitted, a uniform
        ``[0, 1)`` vector is drawn, matching the distribution of Fortran
        ``random_number`` but not its compiler-dependent sequence.
    steps
        Requested Lanczos steps.  The non-``BETA`` PARSEC source clamps this
        to ``[4, 8]``.  For a matrix smaller than four rows, Python safely
        limits the count to the matrix dimension.
    rng
        NumPy generator used only when ``initial_vector`` is omitted.

    Returns
    -------
    LanczosBoundResult
        The estimated upper and lower bounds, PARSEC's ``middle`` value, the
        projected Ritz values, and recurrence diagnostics.
    """

    size = _square_operator_size(operator)
    requested_steps = int(steps)
    if requested_steps != steps:
        raise ValueError("steps must be an integer")
    target_steps = min(max(requested_steps, 4), 8, size)

    if initial_vector is None:
        generator = np.random.default_rng() if rng is None else rng
        vector = _random_start(size, generator)
    else:
        vector = np.asarray(initial_vector)
        if vector.shape != (size,):
            raise ValueError("initial_vector must have shape (operator.shape[0],)")
        vector = vector.copy()
    if not np.all(np.isfinite(vector)):
        raise ValueError("initial_vector must contain only finite values")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("initial_vector must be nonzero")
    vector = vector / norm

    # First Lanczos column: alpha_1=<q_1,Hq_1> and
    # f_1=Hq_1-alpha_1*q_1.  Only the small T matrix is stored; Lanczos vectors
    # other than the current and previous ones are not retained.
    projected = np.zeros((target_steps, target_steps), dtype=float)
    residual = _apply(operator, vector, size)
    alpha = float(np.real(np.vdot(vector, residual)))
    residual = residual - alpha * vector
    projected[0, 0] = alpha

    actual_steps = 1
    raw_beta = float(np.linalg.norm(residual))
    breakdown = False

    for column in range(1, target_steps):
        # This placement is intentional: lancz_bound.f90z retains this beta
        # after the final iteration and later adds it to the Ritz estimate.
        # On the last pass it is beta_(k-1), used to construct q_k; the norm of
        # the newly produced f_k is never evaluated by the Fortran routine.
        raw_beta = float(np.linalg.norm(residual))
        if raw_beta <= _BREAKDOWN_TOLERANCE:
            breakdown = True
            break

        # q_j=f_(j-1)/beta_(j-1), followed by the unreorthogonalized
        # three-term Hermitian Lanczos recurrence.
        previous = vector
        vector = residual / raw_beta
        residual = _apply(operator, vector, size) - raw_beta * previous
        alpha = float(np.real(np.vdot(vector, residual)))
        residual = residual - alpha * vector

        projected[column, column - 1] = raw_beta
        projected[column - 1, column] = raw_beta
        projected[column, column] = alpha
        actual_steps = column + 1

    # Truncating to actual_steps is the mathematically consistent breakdown
    # behavior.  It deliberately avoids PARSEC's zero-padded (k-1)-block
    # diagonalization artifact described in the module docstring.
    ritz_values = np.linalg.eigvalsh(projected[:actual_steps, :actual_steps])
    lower_bound = float(ritz_values[0])
    # PARSEC uses a radius, so a large-magnitude negative Ritz value can set
    # the positive Chebyshev upper endpoint even when theta_max is smaller.
    spectral_radius = max(float(ritz_values[-1]), abs(lower_bound))
    middle = 0.5 * (spectral_radius + lower_bound)

    # Empirical safety enlargement copied from the non-BETA Fortran branch.
    residual_scale = raw_beta
    if residual_scale < 1.0e-2:
        residual_scale *= 10.0
    elif residual_scale < 1.0e-1:
        residual_scale *= 5.0
    upper_bound = spectral_radius + residual_scale

    return LanczosBoundResult(
        upper_bound=float(upper_bound),
        lower_bound=lower_bound,
        middle=float(middle),
        ritz_values=ritz_values,
        raw_beta=raw_beta,
        residual_scale=float(residual_scale),
        steps=actual_steps,
        requested_steps=requested_steps,
        breakdown=breakdown,
    )
