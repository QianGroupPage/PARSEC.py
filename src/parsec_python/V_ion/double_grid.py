"""Ono-Hirose double grid for the nonlocal projectors (``nonloc.F90``).

With ``Double_Grid_Order`` ``N > 1`` PARSEC does not sample a
Kleinman-Bylander projector at a grid point ``r_i`` but averages it over the
points of a grid ``N`` times finer around it (Ono and Hirose, Phys. Rev.
Lett. 82, 5016 (1999)):

``beta(r_i) = sum_(dx,dy,dz) w(dx) w(dy) w(dz) beta(r_i + (dx,dy,dz)*h/N)``,

``d = -(N-1), ..., N-1``, ``w(d) = (1 - |d|/N)/N``.  The weights of one
direction add up to one; they are those of linear interpolation from the
coarse to the fine grid, so the sum is the adjoint of that interpolation.
``nonloc.F90:116-121`` sets them up, ``205-259`` and ``317-406`` choose the
coarse points (one is kept when any of its fine points lies within the
projector radius ``rc``), and ``505-569`` sums the values and the gradients
(the gradients feed ``forcnloc.f90`` unchanged).  A fine point beyond ``rc``
contributes nothing; the radial table is evaluated at the distance clamped
below to the first radial sample.  ``N = 1`` is the ordinary sampling.

The harmonics of a fine point use its true direction, as the projectors of
:func:`parsec_python.V_ion.build_nonlocal_projectors` do, so that ``N = 1``
reproduces them exactly; ``ylm_real`` of ``nonloc.F90`` divides the vector
by the clamped distance, which differs only for fine points closer to the
atom than the first radial sample (``~1e-4`` bohr).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential, ParsecRadialSpline


def double_grid_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Offsets ``d = -(N-1), ..., N-1`` and weights ``(1 - |d|/N)/N``."""

    order = int(order)
    if order < 1:
        raise ValueError("the double grid order must be at least one")
    offsets = np.arange(-(order - 1), order)
    inverse = 1.0 / float(order)
    weights = (1.0 - np.abs(offsets).astype(np.float64) * inverse) * inverse
    return offsets, weights


def double_grid_points(order: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Displacements of the ``(2N-1)**3`` fine points and their weights.

    The order is that of the loops of ``nonloc.F90`` (x slowest, z
    fastest); displacements are ``d*h/N`` in bohr.
    """

    offsets, weights = double_grid_weights(order)
    inverse = 1.0 / float(order)
    shifts = offsets.astype(np.float64) * float(spacing) * inverse
    dx, dy, dz = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    wx, wy, wz = np.meshgrid(weights, weights, weights, indexing="ij")
    displacements = np.column_stack((dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)))
    return displacements, (wx * wy * wz).reshape(-1)


@dataclass(frozen=True)
class DoubleGridSupport:
    """The coarse points of one projector image and their fine points.

    ``rows`` are ascending rows of the grid; ``relative[p, m]`` is fine point
    ``m`` of row ``p`` minus the atom (image) position, ``radius`` its
    length, and ``weights`` the double-grid weight of the fine point, zero
    beyond the projector radius.
    """

    rows: np.ndarray
    relative: np.ndarray
    radius: np.ndarray
    weights: np.ndarray


def double_grid_support(
    coordinates: np.ndarray,
    center: np.ndarray,
    support_radius: float,
    order: int,
    spacing: float,
) -> DoubleGridSupport:
    """Coarse grid points with a fine point inside ``support_radius``."""

    displacements, weights = double_grid_points(order, spacing)
    reach = float(np.max(np.linalg.norm(displacements, axis=1)))
    coarse = np.asarray(coordinates, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    near = np.linalg.norm(coarse, axis=1) <= support_radius + reach + 1.0e-12 * (1.0 + support_radius)
    candidates = np.flatnonzero(near)
    relative = coarse[candidates][:, None, :] + displacements[None, :, :]
    radius = np.linalg.norm(relative, axis=2)
    inside = radius <= support_radius
    kept = np.any(inside, axis=1)
    return DoubleGridSupport(
        rows=candidates[kept],
        relative=relative[kept],
        radius=radius[kept],
        weights=np.where(inside[kept], weights[None, :], 0.0),
    )


def interpolate_projector(
    potential: ParsecPseudopotential,
    table: np.ndarray,
    radius: np.ndarray,
    spline: ParsecRadialSpline | None,
) -> np.ndarray:
    """The radial projector at ``radius`` as the projector builders take it."""

    if spline is not None:
        return spline(np.maximum(radius, potential.radii[0]))
    return np.interp(radius, potential.radii, table, left=table[0], right=0.0)


def build_double_grid_projectors(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray | None = None,
    k_point: np.ndarray | None = None,
    *,
    order: int | None = None,
):
    """Kleinman-Bylander projectors averaged over the double grid.

    Arguments, result, column order and periodic images are those of
    :func:`parsec_python.V_ion.build_nonlocal_projectors`; ``order``
    defaults to ``grid.settings.double_grid_order``.  The Bloch phase of a
    coarse point is that of the point itself, as in ``nonloc.F90``.
    """

    from .ionic_potential import (
        NonlocalProjectorOperator,
        _periodic_image_translations,
        _projector_support_radius,
        real_spherical_harmonics,
    )

    if order is None:
        order = int(getattr(grid.settings, "double_grid_order", 1))
    complex_valued = k_point is not None
    if complex_valued and lattice_vectors is None:
        raise ValueError("k_point requires lattice_vectors")
    value_dtype = complex if complex_valued else float
    k_point = None if k_point is None else np.asarray(k_point, dtype=float)
    sqrt_dv = np.sqrt(grid.volume_element)
    spacing = float(grid.spacing)
    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    signs: list[float] = []
    labels: list[tuple[int, int, int]] = []
    column = 0

    for atom_index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        position = np.asarray(atom.position, dtype=np.float64)
        support_radius = _projector_support_radius(potential)
        translations = (
            np.zeros((1, 3))
            if lattice_vectors is None
            else _periodic_image_translations(lattice_vectors, support_radius)
        )
        images = []
        for translation in translations:
            support = double_grid_support(
                grid.coordinates, position + translation, support_radius, order, spacing
            )
            if support.rows.size:
                phase = (
                    1.0
                    if k_point is None
                    else np.exp(
                        -1j * ((grid.coordinates[support.rows] - translation) @ k_point)
                    )
                )
                images.append((support, phase))
        if not images:
            continue

        for angular_momentum in sorted(potential.radial_wavefunctions):
            if angular_momentum == local_l:
                continue
            table, sign = potential.radial_projector(angular_momentum, local_l)
            spline = (
                ParsecRadialSpline.from_positive_grid(
                    potential.radii, table, grid.settings.stencil_half_width
                )
                if specification.use_spline
                else None
            )
            width = 2 * angular_momentum + 1
            channel_columns = list(range(column, column + width))
            column += width
            for harmonic_index in range(width):
                signs.append(sign)
                labels.append((atom_index, angular_momentum, harmonic_index))
            for support, phase in images:
                points, fine = support.radius.shape
                radial = interpolate_projector(
                    potential, table, support.radius.reshape(-1), spline
                )
                harmonics = real_spherical_harmonics(
                    angular_momentum, support.relative.reshape(-1, 3)
                )
                scaled = (sqrt_dv * radial)[:, None] * harmonics
                summed = np.einsum(
                    "pm,pmh->ph", support.weights, scaled.reshape(points, fine, width)
                )
                for harmonic_index, target in enumerate(channel_columns):
                    projector = phase * summed[:, harmonic_index]
                    keep = np.abs(projector) > 1.0e-16
                    if not np.any(keep):
                        continue
                    rows.append(support.rows[keep])
                    columns.append(np.full(np.count_nonzero(keep), target, dtype=np.int64))
                    values.append(projector[keep].astype(value_dtype, copy=False))

    if column == 0:
        matrix = sp.csc_matrix((grid.size, 0), dtype=value_dtype)
    else:
        matrix = sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(grid.size, column),
            dtype=value_dtype,
        ).tocsc()
    return NonlocalProjectorOperator(
        projectors=matrix,
        signs=np.asarray(signs, dtype=float),
        labels=tuple(labels),
    )


__all__ = [
    "DoubleGridSupport",
    "build_double_grid_projectors",
    "double_grid_points",
    "double_grid_support",
    "double_grid_weights",
    "interpolate_projector",
]
