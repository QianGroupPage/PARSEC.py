"""Normalized complex spherical harmonics by associated-Legendre recurrence.

The generator below evaluates the harmonics of
:func:`scipy.special.sph_harm_y` for every nonnegative ``m`` with three real
work arrays per ``m``.  The accelerated multipole builders and the a-priori
estimate of the Hartree boundary share it.
"""

from __future__ import annotations

import math

import numpy as np


def _normalization(angular_momentum: int, magnetic: int) -> float:
    """Return the normalized complex ``Y_lm`` prefactor for ``m >= 0``."""

    log_ratio = math.lgamma(angular_momentum - magnetic + 1) - math.lgamma(
        angular_momentum + magnetic + 1
    )
    return math.sqrt(
        (2 * angular_momentum + 1)
        * math.exp(log_ratio)
        / (4.0 * math.pi)
    )


def _angular_coordinates(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radius = np.linalg.norm(points, axis=1)
    cosine = np.ones_like(radius)
    nonzero = radius > 0.0
    cosine[nonzero] = points[nonzero, 2] / radius[nonzero]
    cosine = np.clip(cosine, -1.0, 1.0)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    xy_radius = np.hypot(points[:, 0], points[:, 1])
    phase_positive = np.ones(points.shape[0], dtype=np.complex128)
    away_from_axis = xy_radius > 0.0
    phase_positive[away_from_axis] = (
        points[away_from_axis, 0] + 1j * points[away_from_axis, 1]
    ) / xy_radius[away_from_axis]
    return radius, cosine, sine, phase_positive


def _positive_m_harmonic_rows(
    points: np.ndarray,
    order: int,
):
    """Yield ``(l,m,Y_lm,radius)`` for every nonnegative ``m``.

    ``P_l^m`` includes the Condon--Shortley phase, matching
    ``scipy.special.sph_harm_y`` and PARSEC's complex-harmonic convention.
    Only three real associated-Legendre arrays are live for one ``m``.
    """

    radius, cosine, sine, phase_unit = _angular_coordinates(points)
    diagonal = np.ones(points.shape[0], dtype=np.float64)  # P_0^0
    phase = np.ones(points.shape[0], dtype=np.complex128)

    for magnetic in range(order + 1):
        if magnetic:
            diagonal = -(2 * magnetic - 1) * sine * diagonal
            phase = phase * phase_unit

        previous = diagonal
        yield (
            magnetic,
            magnetic,
            _normalization(magnetic, magnetic) * previous * phase,
            radius,
        )
        if magnetic == order:
            continue

        current = (2 * magnetic + 1) * cosine * diagonal
        yield (
            magnetic + 1,
            magnetic,
            _normalization(magnetic + 1, magnetic) * current * phase,
            radius,
        )
        for angular_momentum in range(magnetic + 2, order + 1):
            following = (
                (2 * angular_momentum - 1) * cosine * current
                - (angular_momentum + magnetic - 1) * previous
            ) / (angular_momentum - magnetic)
            yield (
                angular_momentum,
                magnetic,
                _normalization(angular_momentum, magnetic)
                * following
                * phase,
                radius,
            )
            previous, current = current, following
