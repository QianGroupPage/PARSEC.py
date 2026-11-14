"""Fourier filtering of pseudopotential radial tables (``fourier_f.f90``).

Briggs, Sullivan and Bernholc, Phys. Rev. B 54, 14362 (1996): a radial
function ``f(r)`` with angular momentum ``l`` is transformed to

``f(q) = (2/pi) * q**2 * integral r**2 f(r) j_l(q r) dr``

and back with the filter ``F``,

``f_filtered(r) = integral f(q) F(q) j_l(q r) dq``,

``F(q) = 1`` for ``q < q0 = alpha*pi/h`` and
``F(q) = exp(-beta1*(q/q0 - 1)**2)`` above, cut where the exponent passes
``-10`` (``beta1 = 0`` cuts at ``q0``).  ``h`` is the grid spacing.  Both
integrals are those of ``fourier_f.f90``: the five-point Newton-Cotes rule
``(7, 32, 12, 32, 7) * 2/45`` in the index of the logarithmic radial grid
(with ``dr/di = b*(r + c)``) and on the uniform mesh ``q = 0.015*(j-1)``,
``j = 1, ..., nq``, ``nq = 2*nint(max(1, alpha, acore)*pi/h/0.015) + 4``.
The spherical Bessel functions are those of its ``besselj`` (series below
``x = 0.001``, upward recurrence).

Per species ``Alpha_Filter`` (``alpha``) and ``Beta1_Filter`` (``beta1``)
filter the atomic valence density ``rho_r`` and the normalized
Kleinman-Bylander projectors ``vw`` of every nonlocal channel, and
``Core_Filter`` (``acore``, default ``alpha``, with ``beta1 = 0``) the core
density.  Nothing is filtered where the parameter is zero.  PARSEC reads the
parameters for periodic systems only (``usrinputfile.F90:1897-1907``) and
filters in ``inipbc.f90`` with the final grid spacing.

What this module leaves out, and why:

* ``fourier_f.f90`` also filters the local potential ``vion``, but no
  periodic consumer reads it: the local potential of a periodic system is
  made in reciprocal space from the transform of the unfiltered table
  (``pbc%vloc``), so the local potential stays unfiltered here as well.
* Of the radial derivatives ``fourier_f.f90`` replaces (``drhodr``,
  ``dvion``, ``ddenc``, ``dvw``), only ``dvw`` has a reader here: the
  tabulated projector slopes of the nonlocal forces (wire and slab forces,
  and the opt-in ``PARSEC_PERIODIC_FORCE_SLOPES=tabulated`` of bulk forces).
  It is ``dfint``, the radial derivative of the filtered projector
  (:func:`filtered_derivative`), as in PARSEC 1.5 (``math/fourier_f.f90:267``);
  the previous tree called ``fint`` there and stored the filtered projector
  itself as its slope.  The default bulk forces differentiate the
  interpolant of the filtered projector instead, and the local and core
  forces of periodic systems do not read ``dvion``/``drhodr``/``ddenc``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .potre import ParsecPseudopotential

#: ``delq`` of ``fourier_f.f90``: spacing of the 1-D reciprocal mesh (1/bohr).
DELTA_Q = 0.015

#: ``tol`` of ``besselj``, a single-precision literal in ``fourier_f.f90``.
_BESSEL_SERIES_LIMIT = float(np.float32(0.001))


def spherical_bessel(order: int, x: np.ndarray) -> np.ndarray:
    """``besselj(l, x)`` of ``fourier_f.f90``, elementwise."""

    values = np.asarray(x, dtype=np.float64)
    small = np.abs(values) <= _BESSEL_SERIES_LIMIT
    safe = np.where(small, 1.0, values)
    half_square = values * values / 2.0
    first = np.where(
        small,
        1.0 - (half_square / 3.0) * (1.0 - half_square / 10.0),
        np.sin(safe) / safe,
    )
    if order == 0:
        return first
    second = np.where(
        small,
        (values / 3.0) * (1.0 - (half_square / 5.0) * (1.0 - half_square / 14.0)),
        (np.sin(safe) / safe - np.cos(safe)) / safe,
    )
    if order == 1:
        return second
    zero = values == 0.0
    inverse = 1.0 / np.where(zero, 1.0, values)
    lower, current = first, second
    for step in range(1, order):
        lower, current = current, (2 * step + 1) * inverse * current - lower
    return np.where(zero, 0.0, current)


def reciprocal_mesh_size(spacing: float, alpha: float, core_alpha: float) -> int:
    """``nq`` of ``fourier_f.f90:103-104``."""

    extent = max(1.0, float(alpha), float(core_alpha)) * np.pi / float(spacing)
    # Fortran ``nint`` rounds halves away from zero.
    return 2 * int(np.floor(extent / DELTA_Q + 0.5)) + 4


def _index_weights(count: int) -> np.ndarray:
    """Weights of the composite rule ``(7, 32, 12, 32, 7)`` over ``count``.

    Panels start at every fourth index ``1, 5, 9, ...`` up to ``count`` and
    may run past its end; the caller pads the integrand with zeros there.
    """

    weights = np.zeros(count + 4)
    for start in range(0, count, 4):
        weights[start : start + 5] += (7.0, 32.0, 12.0, 32.0, 7.0)
    return weights


def fourier_transform(
    potential: ParsecPseudopotential,
    values: np.ndarray,
    order: int,
    count: int,
    *,
    coulomb: float = 0.0,
) -> np.ndarray:
    """``fourier_1d`` of ``fourier_f.f90`` on ``q = DELTA_Q*(j-1)``.

    ``values`` are aligned with ``potential.radii``.  ``coulomb`` is ``vt``:
    the function behaves as ``-vt/r`` far out and that tail is integrated
    analytically (``2*Z`` in Rydberg for a local potential).  Entry ``j = 1``
    (``q = 0``) is left zero; the caller sets it.
    """

    # Fortran index 1 is the inserted origin, whose integrand vanishes.
    radii = np.concatenate(([0.0], potential.radii))
    table = np.concatenate(([0.0], np.asarray(values, dtype=np.float64)))
    slope = (radii + potential.radial_offset) * potential.radial_parameter_b
    integrand = slope * radii * (radii * table + coulomb)
    integrand[0] = 0.0
    weights = _index_weights(radii.size)[: radii.size]
    q = DELTA_Q * np.arange(count, dtype=np.float64)
    result = np.zeros(count)
    if count < 2:
        return result
    bessel = spherical_bessel(order, np.outer(q[1:], radii))
    bessel[:, 0] = 0.0
    sums = bessel @ (weights * integrand)
    result[1:] = (2.0 * q[1:] * q[1:] * sums / 45.0 - coulomb) * 2.0 / np.pi
    return result


def filter_weights(
    count: int, spacing: float, alpha: float, beta1: float
) -> np.ndarray:
    """``F(q)`` of ``fint`` times the Newton-Cotes weights, on the q mesh."""

    cutoff_index = alpha * np.pi / spacing / DELTA_Q
    unfiltered = int(cutoff_index) + 1
    largest = np.sqrt(10.0 / beta1) + 1.0 if beta1 > 0.0 else 1.0
    factor = np.zeros(count)
    factor[: min(unfiltered, count)] = 1.0
    for index in range(unfiltered, count):
        ratio = float(index) / cutoff_index
        if ratio > largest:
            break
        factor[index] = np.exp(-beta1 * (ratio - 1) * (ratio - 1))
    weights = np.zeros(count)
    for start in range(0, count - 4, 4):
        weights[start : start + 5] += (7.0, 32.0, 12.0, 32.0, 7.0)
    return weights * factor


def filtered_function(
    radii: np.ndarray,
    transform: np.ndarray,
    order: int,
    spacing: float,
    alpha: float,
    beta1: float,
) -> np.ndarray:
    """``fint`` of ``fourier_f.f90`` at every radius of ``radii``."""

    count = transform.size
    q = DELTA_Q * np.arange(count, dtype=np.float64)
    weights = filter_weights(count, spacing, alpha, beta1) * transform
    bessel = spherical_bessel(order, np.outer(np.asarray(radii, dtype=np.float64), q))
    return (bessel @ weights) * DELTA_Q * 2.0 / 5.0 / 9.0


def filtered_derivative(
    radii: np.ndarray,
    transform: np.ndarray,
    order: int,
    spacing: float,
    alpha: float,
    beta1: float,
) -> np.ndarray:
    """``dfint`` of PARSEC 1.5 (``math/auxiliary.F90:2569-2630``) at ``radii``.

    The radial derivative of :func:`filtered_function`: the Bessel factor
    ``j_l(q r)`` becomes ``q j_l'(q r)`` with
    ``j_l'(x) = l j_l(x)/x - j_(l+1)(x)``, the same filter and weights, and
    the ``q = 0`` term left out (``ii = 2, ...`` there).  ``radii`` must be
    positive (the file radii; ``fourier_f.f90`` sets ``dvw = 0`` at the
    padded origin).
    """

    count = transform.size
    q = DELTA_Q * np.arange(count, dtype=np.float64)
    weights = filter_weights(count, spacing, alpha, beta1) * transform * q
    weights[:1] = 0.0
    argument = np.outer(np.asarray(radii, dtype=np.float64), q)
    safe = np.where(argument == 0.0, 1.0, argument)
    slope = order * spherical_bessel(order, argument) / safe - spherical_bessel(
        order + 1, argument
    )
    slope[:, :1] = 0.0
    return (slope @ weights) * DELTA_Q * 2.0 / 5.0 / 9.0


def apply_fourier_filter(
    potential: ParsecPseudopotential,
    local_l: int,
    *,
    spacing: float,
    alpha: float,
    beta1: float = 0.0,
    core_alpha: float | None = None,
    valence_density: np.ndarray | None = None,
) -> ParsecPseudopotential:
    """Return ``potential`` with the tables ``fourier_f.f90`` filters.

    ``spacing`` is the grid spacing ``h`` in bohr, ``alpha``, ``beta1`` and
    ``core_alpha`` the ``Alpha_Filter``, ``Beta1_Filter`` and ``Core_Filter``
    of the species (``core_alpha=None`` is ``alpha``).  ``valence_density``
    is ``rho_r`` before filtering (the density rebuilt from the radial
    functions unless the species reads its own); it defaults to the one
    rebuilt from the radial functions.

    With ``alpha > 0`` the valence density and the normalized projector of
    every channel but ``local_l`` are replaced; with ``core_alpha > 0`` and a
    core correction the core density.  The potential is returned unchanged
    when both are zero.
    """

    alpha = float(alpha)
    beta1 = float(beta1)
    core = alpha if core_alpha is None else float(core_alpha)
    for name, value in (("alpha", alpha), ("beta1", beta1), ("core_alpha", core)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"the Fourier filter parameter {name} must be nonnegative")
    if not spacing > 0.0:
        raise ValueError("the grid spacing must be positive")
    if alpha == 0.0 and core == 0.0:
        return potential
    count = reciprocal_mesh_size(spacing, alpha, core)
    radii = potential.radii
    changes: dict[str, object] = {}
    if alpha > 0.0:
        if valence_density is None:
            valence_density = np.zeros_like(radii)
            for angular_momentum, wave in potential.radial_wavefunctions.items():
                valence_density = valence_density + (
                    potential.channel_occupations.get(angular_momentum, 0.0)
                    * wave
                    * wave
                    / (4.0 * np.pi * radii * radii)
                )
        transform = fourier_transform(potential, valence_density, 0, count)
        changes["filtered_valence_density"] = filtered_function(
            radii, transform, 0, spacing, alpha, beta1
        )
        if getattr(potential, "projectors_are_tabulated", False):
            # a potential given as projectors (UPF): every projector, several for one l, is filtered on its own and
            # replaced in the channel, together with its analytic derivative
            channels = []
            for channel in potential.kb_channels:
                transform = fourier_transform(potential, channel.values, channel.angular_momentum, count)
                channels.append(
                    replace(
                        channel,
                        values=filtered_function(radii, transform, channel.angular_momentum, spacing, alpha, beta1),
                        slopes=filtered_derivative(radii, transform, channel.angular_momentum, spacing, alpha, beta1),
                    )
                )
            changes["kb_channels"] = tuple(channels)
        else:
            projectors: dict[int, np.ndarray] = {}
            slopes: dict[int, np.ndarray] = {}
            for angular_momentum, _ in potential.nonlocal_channels(local_l):
                table, _ = potential.radial_projector(angular_momentum, local_l)
                transform = fourier_transform(potential, table, angular_momentum, count)
                projectors[angular_momentum] = filtered_function(
                    radii, transform, angular_momentum, spacing, alpha, beta1
                )
                slopes[angular_momentum] = filtered_derivative(
                    radii, transform, angular_momentum, spacing, alpha, beta1
                )
            changes["filtered_projectors"] = projectors
            changes["filtered_projector_slopes"] = slopes
    if core > 0.0 and potential.has_nonlinear_core_correction:
        transform = fourier_transform(potential, potential.core_density, 0, count)
        changes["core_density"] = filtered_function(
            radii, transform, 0, spacing, core, 0.0
        )
    return replace(potential, **changes)


__all__ = [
    "DELTA_Q",
    "apply_fourier_filter",
    "filter_weights",
    "filtered_derivative",
    "filtered_function",
    "fourier_transform",
    "reciprocal_mesh_size",
    "spherical_bessel",
]
