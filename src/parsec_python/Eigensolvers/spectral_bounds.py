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


def orbital_term_margin(operator: Any) -> float:
    """A bound of the eigenvalues of a meta-GGA's orbital term held by ``operator`` (Ry; 0 without).

    The term ``-div(vtau grad)`` (and the first-order term of the orbital
    form) has localized eigenvalues where ``vtau`` has a narrow spike (low
    density), far above the rest of the spectrum; a short Lanczos run from a
    random vector does not see them and the Chebyshev filter would amplify
    them.  The bounds add Gershgorin's bound of the term
    (``spectral_margin``): ``lambda_max(H_0 + M) <= lambda_max(H_0) + m``.
    ``PARSEC_META_GGA_FILTER_BOUND``: ``gershgorin`` (default) or ``lanczos``
    (no margin: the Lanczos bound alone, for a comparison of its cost); the
    host bounds take ``targeted`` (a device policy,
    :func:`parsec_python.acceleration.Eigensolvers.spectral_bounds.with_orbital_term`)
    as ``gershgorin``.
    """

    import os

    choice = os.environ.get("PARSEC_META_GGA_FILTER_BOUND", "gershgorin").strip().lower()
    if choice not in ("targeted", "gershgorin", "lanczos"):
        raise ValueError("PARSEC_META_GGA_FILTER_BOUND must be targeted, gershgorin or lanczos")
    if choice == "lanczos":
        return 0.0
    return float(getattr(operator, "spectral_margin", 0.0) or 0.0)


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


def _lanczos_upper_bound_core(
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
    return lanczos_bound_from_ritz(
        "parsec",
        ritz_values,
        raw_beta,
        steps=actual_steps,
        requested_steps=requested_steps,
        breakdown=breakdown,
    )


#: The rules of the short-Lanczos bound that every filter (reference and device) computes
#: (:func:`lanczos_bound_from_ritz`): ``parsec`` the earlier PARSEC's, ``lanczos`` PARSEC 1.5's
#: (``Bounds_Method: lanczos``).
LANCZOS_BOUND_RULES = ("parsec", "lanczos")


def lanczos_bound_steps(rule: str, requested_steps: int, size: int) -> int:
    """The Lanczos steps of a bound of ``rule``: ``parsec`` clamps ``requested_steps`` to ``[4, 8]``
    (the non-``BETA`` ``lancz_bound.f90z``), ``lanczos`` takes them as given (PARSEC 1.5); never more
    than the dimension ``size``."""

    requested_steps = int(requested_steps)
    if rule == "parsec":
        return min(max(requested_steps, 4), 8, int(size))
    if rule == "lanczos":
        if requested_steps < 1:
            raise ValueError("steps must be a positive integer")
        return min(requested_steps, int(size))
    raise ValueError(f"unknown Lanczos bound rule {rule!r}; the rules are " + ", ".join(LANCZOS_BOUND_RULES))


def lanczos_bound_from_ritz(
    rule: str,
    ritz_values,
    raw_beta: float,
    *,
    steps: int,
    requested_steps: int,
    breakdown: bool,
) -> LanczosBoundResult:
    """The filter bound of a short-Lanczos tridiagonal (its sorted Ritz values ``theta``) under ``rule``.

    The one bound formula of the reference and device filters (CHEBFF, CHEBDAV, SUBSPACE).

    * ``parsec`` (the earlier PARSEC, the default): the radius
      ``rho = max(theta_max, |theta_min|)``, ``u = rho + margin`` with
      ``margin`` = 10, 5 or 1 times the last ``beta`` (below 1e-2, below 0.1,
      otherwise), ``middle = (rho + theta_min)/2``.  The radius assumes that
      the top of the spectrum exceeds the magnitude of its bottom, as for a
      cluster whose potential vanishes far away.  A bulk cell whose
      eigenvalues lie far below 0 (with this code's periodic potential the
      Fermi level of bulk diamond is -237 Ry at 1,000 atoms and -607 Ry at
      4,096, the kinetic spectrum spans ~300 Ry) gets ``u ~ |theta_min|``, an
      interval several times wider than the spectrum, and a filter of
      correspondingly lower effective degree.
    * ``lanczos`` (PARSEC 1.5, ``Bounds_Method: lanczos``): ``u = theta_max +
      margin`` with 1.5's margin (:func:`_parsec15_beta_scale`),
      ``middle = (theta_max + theta_min)/2``; invariant under a shift of the
      spectrum.

    ``lower_bound`` is ``theta_min`` under both rules.
    """

    ritz_values = np.asarray(ritz_values, dtype=float)
    lower_bound = float(ritz_values[0])
    largest = float(ritz_values[-1])
    raw_beta = float(raw_beta)
    if rule == "parsec":
        # PARSEC uses a radius, so a large-magnitude negative Ritz value can set
        # the positive Chebyshev upper endpoint even when theta_max is smaller.
        spectral_radius = max(largest, abs(lower_bound))
        middle = 0.5 * (spectral_radius + lower_bound)
        # Empirical safety enlargement copied from the non-BETA Fortran branch.
        residual_scale = raw_beta
        if residual_scale < 1.0e-2:
            residual_scale *= 10.0
        elif residual_scale < 1.0e-1:
            residual_scale *= 5.0
        upper_bound = spectral_radius + residual_scale
    elif rule == "lanczos":
        residual_scale = _parsec15_beta_scale(raw_beta)
        upper_bound = largest + residual_scale
        middle = 0.5 * (largest + lower_bound)
    else:
        raise ValueError(f"unknown Lanczos bound rule {rule!r}; the rules are " + ", ".join(LANCZOS_BOUND_RULES))
    return LanczosBoundResult(
        upper_bound=float(upper_bound),
        lower_bound=lower_bound,
        middle=float(middle),
        ritz_values=ritz_values,
        raw_beta=raw_beta,
        residual_scale=float(residual_scale),
        steps=int(steps),
        requested_steps=int(requested_steps),
        breakdown=bool(breakdown),
    )


def bound_request(settings) -> tuple[str, int]:
    """``(rule, steps)`` of the short-Lanczos bound that a filter's settings ask for.

    ``settings`` is a :class:`~parsec_python.Eigensolvers.chebff.ChebFFSettings`,
    :class:`~parsec_python.Eigensolvers.chebdav.ChebDavSettings` or
    :class:`~parsec_python.Eigensolvers.subspace.SubspaceSettings`: ``bounds``
    (``parsec`` or ``lanczos``; ``lanczos_full`` is the reference CHEBFF's
    only) and ``lanczos_steps``, replaced by ``bound_steps`` (``max(degree/2,
    Bounds_Iter)``) where the settings carry one and the rule is not ``parsec``
    (1.5's CHEBDAV keeps its ``lanczos_steps``).
    """

    rule = str(getattr(settings, "bounds", "parsec"))
    steps = int(settings.lanczos_steps)
    bound_steps = getattr(settings, "bound_steps", None)
    if rule != "parsec" and bound_steps is not None:
        steps = int(bound_steps)
    return rule, steps


def _parsec15_beta_scale(beta: float) -> float:
    """PARSEC 1.5's enlargement of the last ``beta`` (``lancz_bound.f90z:481-494`` of 1.5).

    ``10 beta`` below 1e-2, ``5 beta`` below 0.1, ``2 beta`` below 1,
    ``1.5 beta`` below 100, ``1.01 beta`` below 1000 (the source writes
    ``10d1`` and ``10d2``), ``beta`` otherwise.
    """

    for limit, factor in ((1.0e-2, 10.0), (1.0e-1, 5.0), (1.0, 2.0), (100.0, 1.5), (1000.0, 1.01)):
        if beta < limit:
            return beta * factor
    return beta


def _start_vector(size: int, initial_vector, rng) -> np.ndarray:
    if initial_vector is None:
        vector = _random_start(size, np.random.default_rng() if rng is None else rng)
    else:
        vector = np.array(initial_vector, copy=True)
        if vector.shape != (size,):
            raise ValueError("initial_vector must have shape (operator.shape[0],)")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("initial_vector must be finite and nonzero")
    return vector / norm


def _lanczos_bound_parsec15_core(
    operator: Any,
    initial_vector: np.ndarray | None = None,
    steps: int = 10,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """PARSEC 1.5's short-Lanczos bound (``lancz_bound.f90z:282-510`` of 1.5).

    The same unreorthogonalized recurrence as :func:`lanczos_upper_bound`,
    with the changes of 1.5: ``steps`` is taken as given (no ``[4, 8]``
    clamp; the callers ask for ``max(degree/2, Bounds_Iter)``), the radius
    ``max(theta_max, abs(theta_min))`` is gone (``OLDLANCZBOUND`` undefined:
    ``theta_max`` alone), ``middle = (theta_max + theta_min)/2`` and the
    margin is :func:`_parsec15_beta_scale` of the ``beta`` that started the
    last step.  The start vector is NumPy's uniform ``[0, 1)`` vector as in
    :func:`lanczos_upper_bound`; 1.5 draws it with ``random_Nvector`` (the
    gfortran generator, re-seeded at every call), which is not reproduced.
    A breakdown keeps the steps that were built (1.5 steps back twice).
    """

    size = _square_operator_size(operator)
    requested_steps = int(steps)
    if requested_steps != steps or requested_steps < 1:
        raise ValueError("steps must be a positive integer")
    target_steps = min(requested_steps, size)
    vector = _start_vector(size, initial_vector, rng)
    projected = np.zeros((target_steps, target_steps), dtype=float)
    residual = _apply(operator, vector, size)
    alpha = float(np.real(np.vdot(vector, residual)))
    residual = residual - alpha * vector
    projected[0, 0] = alpha
    actual_steps = 1
    raw_beta = float(np.linalg.norm(residual))
    breakdown = False
    for column in range(1, target_steps):
        raw_beta = float(np.linalg.norm(residual))
        if raw_beta <= _BREAKDOWN_TOLERANCE:
            breakdown = True
            break
        previous = vector
        vector = residual / raw_beta
        residual = _apply(operator, vector, size) - raw_beta * previous
        alpha = float(np.real(np.vdot(vector, residual)))
        residual = residual - alpha * vector
        projected[column, column - 1] = raw_beta
        projected[column - 1, column] = raw_beta
        projected[column, column] = alpha
        actual_steps = column + 1
    ritz_values = np.linalg.eigvalsh(projected[:actual_steps, :actual_steps])
    return lanczos_bound_from_ritz(
        "lanczos",
        ritz_values,
        raw_beta,
        steps=actual_steps,
        requested_steps=requested_steps,
        breakdown=breakdown,
    )


def _lanczos_bound_full_orthogonalization_core(
    operator: Any,
    initial_vector: np.ndarray | None = None,
    steps: int = 10,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """PARSEC 1.5's ``lancz_bound_full_orth`` (``lancz_bound.f90z:17-180`` of 1.5).

    ``Bounds_Method: lanczos_full`` of the CHEBFF first solve: Lanczos with a
    full reorthogonalization of every new vector against all earlier ones,
    a lucky break when ``(j+1) beta_j^2 < orthtol * w`` (``w`` the running
    sum of ``alpha^2 + 2 beta^2``, ``orthtol = 2.5e-16``, const.F90:106), and
    the bounds of Zhou and Li (Linear Algebra Appl. 435, 480 (2011)): with
    Ritz pairs ``(theta_j, z_j)`` of the tridiagonal and ``b`` its last
    off-diagonal ``beta``, ``lower = min_j (theta_j - |b z_kj|)`` and
    ``upper = max_j (theta_j + |b z_kj|)``.  As in 1.5, ``b`` is the ``beta``
    before the last step (``finalbeta = bet(fsteps-1)``).  ``middle`` is the
    centre of the two bounds (1.5 returns none).
    """

    size = _square_operator_size(operator)
    requested_steps = int(steps)
    if requested_steps != steps or requested_steps < 2:
        raise ValueError("steps must be an integer of at least two")
    target_steps = min(requested_steps, size)
    start = _start_vector(size, initial_vector, rng)
    basis = np.zeros((size, target_steps + 1), dtype=np.result_type(start.dtype, float))
    basis[:, 0] = start
    alphas = np.zeros(target_steps)
    betas = np.zeros(target_steps)
    running = 0.0
    final_steps = target_steps
    for j in range(target_steps):
        candidate = _apply(operator, basis[:, j], size)
        if j > 0:
            candidate = candidate - betas[j - 1] * basis[:, j - 1]
        alphas[j] = float(np.real(np.vdot(basis[:, j], candidate)))
        running += alphas[j] * alphas[j]
        candidate = candidate - alphas[j] * basis[:, j]
        # The full reorthogonalization: one classical Gram-Schmidt pass.
        candidate = candidate - basis[:, : j + 1] @ (basis[:, : j + 1].conj().T @ candidate)
        squared = float(np.real(np.vdot(candidate, candidate)))
        if squared * (j + 2) < _BREAKDOWN_TOLERANCE * running:
            final_steps = j + 1
            break
        running += 2.0 * squared
        betas[j] = np.sqrt(squared)
        basis[:, j + 1] = candidate / betas[j]
    if final_steps < 2:
        raise RuntimeError("the full-reorthogonalization Lanczos bound broke down in its first step")
    tridiagonal = (
        np.diag(alphas[:final_steps])
        + np.diag(betas[: final_steps - 1], 1)
        + np.diag(betas[: final_steps - 1], -1)
    )
    ritz_values, vectors = np.linalg.eigh(tridiagonal)
    final_beta = float(betas[final_steps - 2])
    spread = np.abs(final_beta * vectors[final_steps - 1, :])
    lower_bound = float(np.min(ritz_values - spread))
    upper_bound = float(np.max(ritz_values + spread))
    return LanczosBoundResult(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        middle=0.5 * (upper_bound + lower_bound),
        ritz_values=ritz_values,
        raw_beta=final_beta,
        residual_scale=float(upper_bound - ritz_values[-1]),
        steps=final_steps,
        requested_steps=requested_steps,
        breakdown=final_steps < target_steps,
    )


def _with_orbital_term(core, operator, initial_vector, steps, rng) -> LanczosBoundResult:
    """The bound of ``core`` on ``operator``, with a meta-GGA's orbital term taken in.

    ``max(lanczos, bare + m)``: ``bare`` the same Lanczos bound of the
    operator without the term (``without_orbital_term``), ``m`` the term's
    Gershgorin bound (:func:`orbital_term_margin`), as the device bounds
    (:func:`parsec_python.acceleration.Eigensolvers.spectral_bounds.with_orbital_term`);
    ``lanczos + m`` for an operator that cannot drop its term.
    """

    from dataclasses import replace

    bound = core(operator, initial_vector, steps, rng=rng)
    margin = orbital_term_margin(operator)
    if margin <= 0.0:
        return bound
    strip = getattr(operator, "without_orbital_term", None)
    if callable(strip):
        bare = core(strip(), initial_vector, steps, rng=rng).upper_bound
        upper = max(float(bound.upper_bound), float(bare) + margin)
    else:
        upper = float(bound.upper_bound) + margin
    return replace(bound, upper_bound=float(upper))


def lanczos_upper_bound(
    operator: Any,
    initial_vector: np.ndarray | None = None,
    steps: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """PARSEC's short Lanczos bound (:func:`_lanczos_upper_bound_core`) with a meta-GGA's orbital term taken in."""

    return _with_orbital_term(_lanczos_upper_bound_core, operator, initial_vector, steps, rng)


def lanczos_bound_parsec15(
    operator: Any,
    initial_vector: np.ndarray | None = None,
    steps: int = 10,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """PARSEC 1.5's ``lanczos`` bound (:func:`_lanczos_bound_parsec15_core`) with a meta-GGA's orbital term taken in."""

    return _with_orbital_term(_lanczos_bound_parsec15_core, operator, initial_vector, steps, rng)


def lanczos_bound_full_orthogonalization(
    operator: Any,
    initial_vector: np.ndarray | None = None,
    steps: int = 10,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """PARSEC 1.5's ``lanczos_full`` bound (:func:`_lanczos_bound_full_orthogonalization_core`) with the orbital term."""

    return _with_orbital_term(_lanczos_bound_full_orthogonalization_core, operator, initial_vector, steps, rng)


def spectral_bound(
    operator: Any,
    rule: str = "parsec",
    steps: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """The Lanczos bound of ``rule``: ``parsec`` (:func:`lanczos_upper_bound`, the earlier
    PARSEC), ``lanczos`` (:func:`lanczos_bound_parsec15`) or ``lanczos_full``
    (:func:`lanczos_bound_full_orthogonalization`), PARSEC 1.5's ``Bounds_Method``."""

    if rule == "parsec":
        return lanczos_upper_bound(operator, steps=steps, rng=rng)
    if rule == "lanczos":
        return lanczos_bound_parsec15(operator, steps=steps, rng=rng)
    if rule == "lanczos_full":
        return lanczos_bound_full_orthogonalization(operator, steps=steps, rng=rng)
    raise ValueError("the bound rule must be parsec, lanczos or lanczos_full")
