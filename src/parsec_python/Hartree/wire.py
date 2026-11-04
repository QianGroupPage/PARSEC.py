"""Hartree boundary values of a wire (1-D periodic) system (F ``hartset_wire.F90``).

A wire is periodic along ``x`` with period ``a`` and confined to the cylinder
``rho = sqrt(y**2 + z**2) <= R``.  Outside the cylinder the potential of a
density inside it is the cylindrical multipole series of the Green's function
of a periodic line,

``V(rho, theta, x) = 2 Re{ sum_(m=1..L) Q_m0 e^(i m theta) / (2 m rho^m)
     + sum_(n=1..L) e^(i n k x) [Q_0n K_0(n k rho)
          + sum_(m=1..L) K_m(n k rho) (Q_mn e^(i m theta) + Q_-m,n e^(-i m theta))]
     + Q_00 (ln 2 - gamma - ln(rho/a)) / 2 }``    (Ry),

with ``k = 2 pi/a``, ``L = Solver_Lpole`` and the moments of the density
``n_j`` at the grid points ``(x_j, rho_j, theta_j)``

``Q_m0  = c sum_j n_j rho_j^m e^(-i m theta_j)``,
``Q_0n  = c sum_j n_j e^(-i n k x_j) I_0(n k rho_j)``,
``Q_+-m,n = c sum_j n_j e^(-+i m theta_j) e^(-i n k x_j) I_m(n k rho_j)``,
``Q_00  = c sum_j n_j``,  ``c = 4 h^3 / a``

(``hartset_wire.F90:96-257``).  ``ln 2 - gamma`` fixes the constant of the
logarithm; the same convention enters the ion-ion energy of
:func:`parsec_python.V_ion.wire_slab.wire_ion_ion_energy`, so that the
divergent constants of a neutral wire cancel.  ``theta`` is the angle from
``y`` towards ``z``.

Deviation from Fortran (``skip_axis_points=True`` reproduces it): Fortran
skips grid points on the axis (``rho_j = 0``) in every moment but ``Q_00``
(``hartset_wire.F90:109-111``), which drops their ``I_0(0) = 1``
contribution to ``Q_0n``.  The axis holds grid points only with a zero grid
shift (``Ignore_Symmetry``).  Fortran's Bessel functions are the Numerical
Recipes polynomial fits (relative error ~1e-7); here they are SciPy's, with
the exponentially scaled forms so that large arguments do not overflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ive, kve

#: ``ln2g`` of ``hartset_wire.F90``: ln 2 minus the Euler-Mascheroni constant.
LN2_MINUS_EULER = 0.1159315156584124488


@dataclass(frozen=True)
class WireMultipoles:
    """Moments ``Q_mn`` of one density (already times ``4 h^3/a``).

    ``radial[m-1] = Q_m0`` (``m = 1..L``), ``axial[n-1] = Q_0n``,
    ``positive[n-1, m-1] = Q_mn`` and ``negative[n-1, m-1] = Q_-m,n``;
    ``monopole = Q_00``.  The Bessel ``I`` factors carry ``exp(-n k R)``
    (see :class:`WireBoundaryPlan`).
    """

    radial: np.ndarray
    axial: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    monopole: float


@dataclass(frozen=True)
class WireBoundaryPlan:
    """Static tables of the wire boundary values of one grid.

    Source tables run over the grid points, boundary tables over the
    exterior stencil points.  With ``R`` the cylinder radius every source
    has ``rho_j <= R`` and every exterior point ``rho_b > R``; the modified
    Bessel functions are stored as ``I_m(n k rho_j) exp(-n k R)`` and
    ``K_m(n k rho_b) exp(n k R)``, whose products are those of the series.
    """

    order: int
    period: float
    volume_element: float
    source_radial: np.ndarray  # (N, L) complex: rho^m e^(-i m theta)
    source_angular: np.ndarray  # (N, L) complex: e^(-i m theta)
    source_axial: np.ndarray  # (N, L) complex: e^(-i n k x)
    source_bessel: np.ndarray  # (N, L, L+1): I_m(n k rho) e^(-n k R), [j, n-1, m]
    source_mask: np.ndarray  # (N,) 1.0, or 0.0 for a skipped axis point
    boundary_radius: np.ndarray  # (B,)
    boundary_angular: np.ndarray  # (B, L) complex: e^(i m theta)
    boundary_axial: np.ndarray  # (B, L) complex: e^(i n k x)
    boundary_bessel: np.ndarray  # (B, L, L+1): K_m(n k rho) e^(n k R)

    def multipoles(self, density: np.ndarray) -> WireMultipoles:
        """The moments of ``density`` (electrons/bohr**3 on the grid points)."""

        density = np.asarray(density, dtype=np.float64)
        weights = density * self.source_mask
        factor = 4.0 * self.volume_element / self.period
        radial = factor * (weights @ self.source_radial)
        # (N, L) x (N, L, L+1): sum_j w_j e^(-i n k x_j) I_m(n k rho_j)
        phased = (weights[:, None] * self.source_axial)
        axial_part = np.einsum("jn,jnm->nm", phased, self.source_bessel)
        axial = factor * axial_part[:, 0]
        positive = factor * np.einsum(
            "jn,jnm,jm->nm", phased, self.source_bessel[:, :, 1:], self.source_angular
        )
        negative = factor * np.einsum(
            "jn,jnm,jm->nm",
            phased,
            self.source_bessel[:, :, 1:],
            np.conj(self.source_angular),
        )
        monopole = factor * float(np.sum(density))
        return WireMultipoles(
            radial=radial,
            axial=axial,
            positive=positive,
            negative=negative,
            monopole=monopole,
        )

    def boundary_values(self, density: np.ndarray) -> np.ndarray:
        """Rydberg potential of ``density`` at every exterior stencil point."""

        moments = self.multipoles(density)
        order = self.order
        radius = self.boundary_radius
        powers = np.arange(1, order + 1, dtype=np.float64)
        # sum_m Q_m0 e^(i m theta) / (2 m rho^m)
        value = (
            self.boundary_angular
            * moments.radial[None, :]
            / (2.0 * powers[None, :] * radius[:, None] ** powers[None, :])
        ).sum(axis=1)
        bessel = self.boundary_bessel
        angular = self.boundary_angular
        inner = moments.axial[None, :] * bessel[:, :, 0]
        inner = inner + np.einsum(
            "bnm,nm,bm->bn", bessel[:, :, 1:], moments.positive, angular
        )
        inner = inner + np.einsum(
            "bnm,nm,bm->bn", bessel[:, :, 1:], moments.negative, np.conj(angular)
        )
        value = value + np.sum(self.boundary_axial * inner, axis=1)
        value = value + moments.monopole * (
            LN2_MINUS_EULER - np.log(radius / self.period)
        ) * 0.5
        return 2.0 * value.real


def _bessel_table(function, orders: int, arguments: np.ndarray) -> np.ndarray:
    """``function(m, arguments)`` for ``m = 0..orders`` as the last axis."""

    return np.stack([function(m, arguments) for m in range(orders + 1)], axis=-1)


def build_wire_boundary_plan(
    grid,
    exterior_points: np.ndarray,
    order: int,
    *,
    skip_axis_points: bool = False,
) -> WireBoundaryPlan:
    """Tables of :class:`WireBoundaryPlan` for ``grid`` and its exterior points.

    ``exterior_points`` are the Cartesian coordinates of the missing stencil
    neighbours (:func:`parsec_python.Hartree.confined.exterior_stencil`).
    ``skip_axis_points=True`` reproduces Fortran's omission of grid points on
    the axis from the moments other than ``Q_00``.
    """

    order = int(order)
    if order < 1:
        raise ValueError("the wire boundary needs Solver_Lpole >= 1")
    period = grid.period
    wave = 2.0 * np.pi / period
    radius_limit = grid.confinement_radius
    coordinates = grid.coordinates
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    rho = np.sqrt(y * y + z * z)
    theta = np.arctan2(z, y)
    m = np.arange(1, order + 1)
    n = np.arange(1, order + 1)
    source_angular = np.exp(-1j * m[None, :] * theta[:, None])
    source_radial = rho[:, None] ** m[None, :] * source_angular
    source_axial = np.exp(-1j * wave * n[None, :] * coordinates[:, 0][:, None])
    argument = wave * n[None, :] * rho[:, None]  # (N, L)
    scale = np.exp(argument - wave * n[None, :] * radius_limit)
    source_bessel = _bessel_table(ive, order, argument) * scale[:, :, None]
    mask = np.ones(grid.size, dtype=np.float64)
    if skip_axis_points:
        mask[rho == 0.0] = 0.0

    exterior = np.asarray(exterior_points, dtype=np.float64)
    by = exterior[:, 1]
    bz = exterior[:, 2]
    boundary_radius = np.sqrt(by * by + bz * bz)
    if np.any(boundary_radius <= radius_limit):
        raise ValueError("an exterior stencil point lies inside the wire cylinder")
    boundary_theta = np.arctan2(bz, by)
    boundary_angular = np.exp(1j * m[None, :] * boundary_theta[:, None])
    boundary_axial = np.exp(1j * wave * n[None, :] * exterior[:, 0][:, None])
    boundary_argument = wave * n[None, :] * boundary_radius[:, None]
    boundary_scale = np.exp(-(boundary_argument - wave * n[None, :] * radius_limit))
    boundary_bessel = _bessel_table(kve, order, boundary_argument) * boundary_scale[:, :, None]
    return WireBoundaryPlan(
        order=order,
        period=period,
        volume_element=grid.volume_element,
        source_radial=source_radial,
        source_angular=source_angular,
        source_axial=source_axial,
        source_bessel=source_bessel,
        source_mask=mask,
        boundary_radius=boundary_radius,
        boundary_angular=boundary_angular,
        boundary_axial=boundary_axial,
        boundary_bessel=boundary_bessel,
    )


__all__ = [
    "LN2_MINUS_EULER",
    "WireBoundaryPlan",
    "WireMultipoles",
    "build_wire_boundary_plan",
]
