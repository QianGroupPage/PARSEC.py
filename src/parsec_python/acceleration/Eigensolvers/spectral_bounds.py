"""Short PARSEC Lanczos spectral bounds with device-resident vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from parsec_python.Eigensolvers.spectral_bounds import LanczosBoundResult

from ..backends.cupy import require_cupy


_BREAKDOWN_TOLERANCE = 2.5e-16


def filter_bound_policy() -> str:
    """``PARSEC_META_GGA_FILTER_BOUND``: ``gershgorin`` (default), ``targeted`` or ``lanczos`` (:func:`with_orbital_term`)."""

    import os

    value = os.environ.get("PARSEC_META_GGA_FILTER_BOUND", _DEFAULT_POLICY).strip().lower()
    if value not in ("targeted", "gershgorin", "lanczos"):
        raise ValueError("PARSEC_META_GGA_FILTER_BOUND must be targeted, gershgorin or lanczos")
    return value


_DEFAULT_POLICY = "gershgorin"


def with_orbital_term(operator: Any, upper_bound: float, *, steps: int = 8, rng=None) -> float:
    """The filter's upper bound of an operator that may hold a meta-GGA term.

    The term ``sum_a B_a^T diag(vtau) B_a`` (and the first-order term of the
    orbital form) reports a bound ``m`` of its eigenvalues
    (``spectral_margin``, Gershgorin's row sums).  Where ``vtau`` has a
    narrow spike (low density) the term has localized eigenvalues far above
    the rest of the spectrum, which the short Lanczos run from a random
    vector does not see; the Chebyshev filter would amplify them without
    limit.  With the term, the bound is ``max(lanczos, bare + m)``, ``bare``
    the last Lanczos bound of the same operator without the term (the first
    iteration of a meta-GGA has none), or ``lanczos + m`` before one exists:
    ``lambda_max(H_0 + M) <= lambda_max(H_0) + m``.  ``bare`` is raised by
    the rise of the largest value of the effective potential since it was
    taken.  Without the term the Lanczos bound is recorded as ``bare`` on
    operators that keep one.  This is ``PARSEC_META_GGA_FILTER_BOUND=gershgorin``.

    ``targeted`` runs a second short Lanczos recurrence on the operator from
    a start vector concentrated on the rows where the term is large (its
    Gershgorin row sums with random signs, ``spectral_start_vector``), which
    overlaps the localized modes of the spikes, and takes the larger of the
    two Lanczos bounds: the spectrum's top of ``H_0`` (the checkerboard mode
    of the Laplacian, which the derivative stencil does not see) is not added
    to that of the term.  ``lanczos`` keeps the Lanczos bound alone (a
    comparison; it can end in the filter's failure described above).
    """

    margin = float(getattr(operator, "spectral_margin", 0.0) or 0.0)
    if margin <= 0.0:
        if hasattr(operator, "bare_upper_bound") and getattr(operator, "kinetic_term", None) is None:
            operator.bare_upper_bound = (float(upper_bound), _potential_top(operator))
        return float(upper_bound)
    policy = filter_bound_policy()
    if policy == "lanczos":
        return float(upper_bound)
    start = getattr(operator, "spectral_start_vector", None) if policy == "targeted" else None
    if start is not None:
        cp, _ = require_cupy()
        generator = np.random.default_rng(20261108) if rng is None else rng
        signs = cp.asarray(generator.standard_normal(int(start.shape[0])))
        targeted = _lanczos_bound(operator, cp.asarray(start, dtype=cp.float64) * signs, steps)
        return max(float(upper_bound), float(targeted.upper_bound))
    bare = getattr(operator, "bare_upper_bound", None)
    if bare is None:
        base = float(upper_bound)
    else:
        base, top = bare
        now = _potential_top(operator)
        if top is not None and now is not None:
            base += max(0.0, now - top)
    return max(float(upper_bound), base + margin)


def orbital_term_margin(operator: Any) -> float:
    """The bound of the orbital term an operator holds (``spectral_margin``), 0 under ``PARSEC_META_GGA_FILTER_BOUND=lanczos``."""

    from parsec_python.Eigensolvers.spectral_bounds import orbital_term_margin as host_margin

    return host_margin(operator)


def lanczos_upper_bound(
    operator: Any,
    initial_vector: Any | None = None,
    steps: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """PARSEC's short Lanczos bound (:func:`_lanczos_bound`), with a meta-GGA's orbital term taken in (:func:`with_orbital_term`)."""

    from dataclasses import replace

    bound = _lanczos_bound(operator, initial_vector, steps, rng=rng)
    upper = with_orbital_term(operator, bound.upper_bound, steps=steps, rng=rng)
    return bound if upper == bound.upper_bound else replace(bound, upper_bound=float(upper))


def _potential_top(operator: Any) -> float | None:
    potential = getattr(operator, "effective_potential", None)
    if potential is None:
        return None
    return float(potential.max())


def _lanczos_bound(
    operator: Any,
    initial_vector: Any | None = None,
    steps: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> LanczosBoundResult:
    """Run the non-reorthogonalized 4--8 step PARSEC recurrence on CuPy.

    The ``N``-component current/previous/residual vectors stay on the device.
    Recurrence scalars stay as zero-dimensional device arrays while CUDA
    builds the at-most eight-step Krylov sequence.  They are transferred in
    one small batch before NumPy diagonalizes the tridiagonal.  In
    particular, this avoids two stream synchronizations per Lanczos step for
    ``alpha`` and ``beta``; neither scalar is a host-side decision until the
    recurrence has finished.
    """

    cp, _ = require_cupy()
    array_where = getattr(cp, "where", np.where)
    array_stack = getattr(cp, "stack", np.stack)
    array_concatenate = getattr(cp, "concatenate", np.concatenate)
    shape = getattr(operator, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("operator must expose a square shape")
    size = int(shape[0])
    if size < 1:
        raise ValueError("operator dimension must be positive")
    requested_steps = int(steps)
    if requested_steps != steps:
        raise ValueError("steps must be an integer")
    target_steps = min(max(requested_steps, 4), 8, size)

    if initial_vector is None:
        generator = np.random.default_rng() if rng is None else rng
        # Match the reference stream/distribution on the host, then upload once.
        vector = cp.asarray(generator.random(size), dtype=cp.float64)
    else:
        vector = cp.asarray(initial_vector, dtype=cp.float64)
        if vector.shape != (size,):
            raise ValueError("initial_vector must have shape (dimension,)")
        vector = vector.copy()
    initial_norm = cp.linalg.norm(vector)
    safe_initial_norm = array_where(initial_norm > 0.0, initial_norm, 1.0)
    vector /= safe_initial_norm

    projected = np.zeros((target_steps, target_steps), dtype=np.float64)
    residual = operator @ vector
    alpha = cp.vdot(vector, residual).real
    residual = residual - alpha * vector
    device_alphas = [alpha]
    device_betas = []

    raw_beta = cp.linalg.norm(residual)
    for column in range(1, target_steps):
        # The beta retained at the top of the final pass is the one used by
        # PARSEC's empirical upper-bound enlargement.
        # ``raw_beta`` was already evaluated immediately before the loop.
        # Reuse it for column one; later columns need the newly formed
        # residual norm.  The former code evaluated the identical first norm
        # twice, adding one device reduction and host synchronization to every
        # sector bound without changing a single recurrence value.
        if column > 1:
            raw_beta = cp.linalg.norm(residual)
        device_betas.append(raw_beta)

        # A literal implementation stops here on a Lanczos breakdown.  That
        # requires downloading beta at every pass and serializes all eight
        # representation streams.  A safe divisor lets the queued device
        # recurrence finish instead.  If beta is below tolerance, subsequent
        # vectors may be computed but are discarded after the one batched
        # scalar download below; every accepted column before the breakdown
        # is therefore identical to the literal recurrence.
        safe_beta = array_where(
            raw_beta > _BREAKDOWN_TOLERANCE, raw_beta, 1.0
        )
        previous = vector
        vector = residual / safe_beta
        residual = operator @ vector - raw_beta * previous
        alpha = cp.vdot(vector, residual).real
        residual = residual - alpha * vector
        device_alphas.append(alpha)

    if device_betas:
        scalar_block = array_concatenate(
            (
                cp.asarray(initial_norm).reshape(1),
                array_stack(device_alphas),
                array_stack(device_betas),
            )
        )
    else:
        # A one-dimensional operator has no off-diagonal Lanczos entries,
        # but PARSEC still reports the norm of its first residual as beta.
        scalar_block = array_concatenate(
            (
                cp.asarray(initial_norm).reshape(1),
                array_stack(device_alphas),
                cp.asarray(raw_beta).reshape(1),
            )
        )
    host_scalars = np.asarray(cp.asnumpy(scalar_block), dtype=np.float64)
    if float(host_scalars[0]) == 0.0:
        raise ValueError("initial_vector must be nonzero")
    host_alphas = host_scalars[1 : target_steps + 1]
    host_betas = host_scalars[target_steps + 1 :]

    if target_steps == 1:
        actual_steps = 1
        raw_beta_value = float(host_betas[0])
        breakdown = False
        projected[0, 0] = host_alphas[0]
        breakdown_indices = np.empty(0, dtype=np.int64)
    else:
        breakdown_indices = np.flatnonzero(
            host_betas <= _BREAKDOWN_TOLERANCE
        )

    if target_steps > 1 and breakdown_indices.size:
        first_breakdown = int(breakdown_indices[0])
        actual_steps = first_breakdown + 1
        raw_beta_value = float(host_betas[first_breakdown])
        breakdown = True
    elif target_steps > 1:
        actual_steps = target_steps
        raw_beta_value = float(host_betas[-1])
        breakdown = False

    if target_steps > 1:
        diagonal = host_alphas[:actual_steps]
        projected[np.arange(actual_steps), np.arange(actual_steps)] = diagonal
        if actual_steps > 1:
            off_diagonal = host_betas[: actual_steps - 1]
            indices = np.arange(actual_steps - 1)
            projected[indices + 1, indices] = off_diagonal
            projected[indices, indices + 1] = off_diagonal

    ritz_values = np.linalg.eigvalsh(projected[:actual_steps, :actual_steps])
    lower_bound = float(ritz_values[0])
    spectral_radius = max(float(ritz_values[-1]), abs(lower_bound))
    middle = 0.5 * (spectral_radius + lower_bound)
    residual_scale = raw_beta_value
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
        raw_beta=raw_beta_value,
        residual_scale=float(residual_scale),
        steps=actual_steps,
        requested_steps=requested_steps,
        breakdown=breakdown,
    )


__all__ = ["LanczosBoundResult", "filter_bound_policy", "lanczos_upper_bound", "with_orbital_term"]
