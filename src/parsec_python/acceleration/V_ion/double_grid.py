"""Ono-Hirose double-grid projectors and their gradients for the accelerated path.

With ``Double_Grid_Order`` ``N > 1`` PARSEC averages every Kleinman-Bylander
projector over the ``(2N-1)**3`` points of a grid ``N`` times finer around
each coarse grid point (``nonloc.F90:116-121, 205-259, 317-406, 505-657``).
The readable builders of :mod:`parsec_python.V_ion.double_grid` and
:func:`parsec_python.Forces.nonlocal_force.double_grid_projector_gradients`
look for the coarse points of an atom among all points of the grid; the
builders here look in the box of grid points around the atom only, through
the lookup table of the grid, as the native projector builder of the
ordinary grid does (:func:`.native_ionic._projector_box_rows`), and share
the atoms of a large structure among a few host threads.

Every value is formed by the operations of the readable builders on the
same arrays: fine-point offsets and weights, the support rule (a coarse
point is kept when one of its fine points lies within the projector
radius), the radial table or spline at the fine distance clamped below to
the first radial sample, the harmonics of the true fine-point direction,
the weighted sum over the fine points, the ``1e-16`` cut of the projector
values, and the column order ``(atom, l, m)``.  The result is the matrix of
the readable builder; the accelerated Hamiltonians, symmetry sectors and
device force contractions take it as they take the ordinary projectors.

Isolated systems only: a periodic preparation uses the readable builders
(:func:`parsec_python.SCF.pbc.prepare_periodic_single_point`).  Units: bohr,
projector columns carry ``sqrt(dV)`` as every projector of this package.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.models import Atom, SpeciesPotential
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.V_ion.double_grid import (
    DoubleGridSupport,
    double_grid_points,
    interpolate_projector,
)

from .native_ionic import _projector_box_rows


def double_grid_order(grid) -> int:
    """``Double_Grid_Order`` of ``grid`` (1 for the ordinary grid)."""

    return int(getattr(getattr(grid, "settings", None), "double_grid_order", 1) or 1)


def _reach(displacements: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(displacements, axis=1)))


def box_double_grid_support(
    grid: RealSpaceGrid,
    center: np.ndarray,
    support_radius: float,
    displacements: np.ndarray,
    weights: np.ndarray,
) -> DoubleGridSupport:
    """:func:`parsec_python.V_ion.double_grid.double_grid_support` from a box.

    The candidates are the grid points of the box around ``center`` that
    holds every point within ``support_radius`` plus the reach of the fine
    points; the readable function tests every point of the grid with the
    same expression.  Rows come out in grid order.
    """

    center = np.asarray(center, dtype=np.float64)
    reach = _reach(displacements)
    box = _projector_box_rows(grid, center, support_radius + reach)
    coarse = np.asarray(grid.coordinates[box], dtype=np.float64) - center
    near = np.linalg.norm(coarse, axis=1) <= (
        support_radius + reach + 1.0e-12 * (1.0 + support_radius)
    )
    candidates = box[near]
    relative = coarse[near][:, None, :] + displacements[None, :, :]
    radius = np.linalg.norm(relative, axis=2)
    inside = radius <= support_radius
    kept = np.any(inside, axis=1)
    return DoubleGridSupport(
        rows=candidates[kept],
        relative=relative[kept],
        radius=radius[kept],
        weights=np.where(inside[kept], weights[None, :], 0.0),
    )


def _spline(potential, table, grid, enabled):
    if not enabled:
        return None
    return ParsecRadialSpline.from_positive_grid(
        potential.radii, table, grid.settings.stencil_half_width
    )


def _workers(count: int) -> int:
    """Host threads that share the atoms of one build.

    Every atom is independent and NumPy releases the interpreter lock in its
    loops: up to four threads (the count measured for the other set-up
    pools, :func:`..Symmetry.representations.operator_build_workers`), one
    for a small molecule.  The result does not depend on the count.
    """

    if count < 32:
        return 1
    return max(1, min(4, os.cpu_count() or 1))


def _map_atoms(function, count: int) -> list:
    """``[function(0), ..., function(count - 1)]``, on the threads of :func:`_workers`."""

    workers = _workers(count)
    if workers == 1:
        return [function(index) for index in range(count)]
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="parsec-double-grid"
    ) as pool:
        return list(pool.map(function, range(count), chunksize=16))


def _channel_tables(atoms, potentials, specifications, grid, *, slopes: bool) -> dict:
    """Radial table, slope table, sign and spline of every nonlocal channel of a species."""

    from parsec_python.Pseudopotential.radial_derivatives import (
        radial_projector_with_derivative,
    )

    tables: dict[tuple[str, int, int], tuple] = {}
    for atom in atoms:
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        local_l = specification.local_angular_momentum
        for angular_momentum, projector_index in potential.nonlocal_channels(local_l):
            key = (atom.symbol, angular_momentum, projector_index)
            if key in tables:
                continue
            if slopes:
                table, slope_table, sign = radial_projector_with_derivative(
                    potential, angular_momentum, local_l, projector_index
                )
            else:
                table, sign = potential.radial_projector(angular_momentum, local_l, projector_index)
                slope_table = None
            tables[key] = (
                table,
                slope_table,
                sign,
                _spline(potential, table, grid, specification.use_spline),
            )
    return tables


def _atom_support(grid, atom, potential, displacements, fine_weights) -> DoubleGridSupport:
    from parsec_python.V_ion.ionic_potential import _projector_support_radius

    return box_double_grid_support(
        grid,
        np.asarray(atom.position, dtype=np.float64),
        _projector_support_radius(potential),
        displacements,
        fine_weights,
    )


def build_double_grid_projectors(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    *,
    order: int | None = None,
) -> NonlocalProjectorOperator:
    """Kleinman-Bylander projectors averaged over the double grid.

    The matrix, signs and labels of
    :func:`parsec_python.V_ion.double_grid.build_double_grid_projectors` for
    an isolated system.  ``order`` defaults to the order of ``grid``.
    """

    from parsec_python.V_ion.ionic_potential import real_spherical_harmonics

    order = double_grid_order(grid) if order is None else int(order)
    displacements, fine_weights = double_grid_points(order, float(grid.spacing))
    sqrt_dv = np.sqrt(grid.volume_element)
    tables = _channel_tables(atoms, potentials, specifications, grid, slopes=False)

    def atom_columns(atom_index: int) -> list:
        """``(rows, values, sign, label)`` of every column of one atom."""

        atom = atoms[atom_index]
        potential = potentials[atom.symbol]
        local_l = specifications[atom.symbol].local_angular_momentum
        support = _atom_support(grid, atom, potential, displacements, fine_weights)
        if not support.rows.size:
            # The readable builder gives such an atom no columns.
            return []
        points, fine = support.radius.shape
        flat_radius = support.radius.reshape(-1)
        flat_relative = support.relative.reshape(-1, 3)
        columns = []
        for angular_momentum, projector_index in potential.nonlocal_channels(local_l):
            table, _slopes, sign, spline = tables[(atom.symbol, angular_momentum, projector_index)]
            width = 2 * angular_momentum + 1
            radial = interpolate_projector(potential, table, flat_radius, spline)
            harmonics = real_spherical_harmonics(angular_momentum, flat_relative)
            scaled = (sqrt_dv * radial)[:, None] * harmonics
            summed = np.einsum(
                "pm,pmh->ph", support.weights, scaled.reshape(points, fine, width)
            )
            for harmonic_index in range(width):
                projector = summed[:, harmonic_index]
                keep = np.abs(projector) > 1.0e-16
                columns.append(
                    (
                        support.rows[keep],
                        projector[keep],
                        sign,
                        (atom_index, angular_momentum, projector_index * (2 * angular_momentum + 1) + harmonic_index),
                    )
                )
        return columns

    columns = [column for block in _map_atoms(atom_columns, len(atoms)) for column in block]
    if not columns:
        matrix = sp.csc_matrix((grid.size, 0), dtype=np.float64)
    else:
        indptr = np.zeros(len(columns) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([column[0].size for column in columns])
        matrix = sp.csc_matrix(
            (
                np.concatenate([column[1] for column in columns]).astype(
                    np.float64, copy=False
                ),
                np.concatenate([column[0] for column in columns]),
                indptr,
            ),
            shape=(grid.size, len(columns)),
        )
    return NonlocalProjectorOperator(
        projectors=matrix,
        signs=np.asarray([column[2] for column in columns], dtype=np.float64),
        labels=tuple(column[3] for column in columns),
    )


def build_double_grid_projector_gradients(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    *,
    order: int | None = None,
):
    """Projector gradients averaged over the double grid.

    The gradients of
    :func:`parsec_python.Forces.nonlocal_force.double_grid_projector_gradients`
    (``nonloc.F90:544-568``): every fine point within the projector radius
    adds its weight times ``w(|d|) grad Y + w'(|d|) Y d/|d|`` with ``|d|``
    clamped below to the first radial sample.  Columns, rows and labels are
    those of :func:`build_double_grid_projectors` before the cut of zero
    values, as in the readable builder.
    """

    from parsec_python.Forces import NonlocalProjectorGradients
    from parsec_python.Forces.nonlocal_force import real_harmonics_with_gradients
    from parsec_python.Pseudopotential.radial_derivatives import spline_derivative

    order = double_grid_order(grid) if order is None else int(order)
    displacements, fine_weights = double_grid_points(order, float(grid.spacing))
    sqrt_dv = np.sqrt(grid.volume_element)
    tables = _channel_tables(atoms, potentials, specifications, grid, slopes=True)

    def atom_columns(atom_index: int) -> list:
        """``(rows, (points, 3) gradients, sign, label)`` of every column of one atom."""

        atom = atoms[atom_index]
        potential = potentials[atom.symbol]
        local_l = specifications[atom.symbol].local_angular_momentum
        support = _atom_support(grid, atom, potential, displacements, fine_weights)
        points, fine = support.radius.shape
        relative = support.relative.reshape(-1, 3)
        clamped = np.maximum(support.radius.reshape(-1), potential.radii[0])
        inverse = 1.0 / clamped
        unit = relative * inverse[:, None]
        columns = []
        for angular_momentum, projector_index in potential.nonlocal_channels(local_l):
            table, slope_table, sign, spline = tables[(atom.symbol, angular_momentum, projector_index)]
            if spline is not None:
                radial = spline(clamped)
                slope = spline_derivative(spline, clamped)
            else:
                radial = np.interp(clamped, potential.radii, table, right=0.0)
                slope = np.interp(clamped, potential.radii, slope_table, right=0.0)
            harmonics, harmonic_gradients = real_harmonics_with_gradients(
                angular_momentum, unit, inverse
            )
            for harmonic_index in range(2 * angular_momentum + 1):
                gradient = sqrt_dv * (
                    radial[:, None] * harmonic_gradients[:, harmonic_index, :]
                    + (slope * harmonics[:, harmonic_index])[:, None] * unit
                )
                summed = np.einsum(
                    "pm,pmc->pc", support.weights, gradient.reshape(points, fine, 3)
                )
                columns.append(
                    (
                        support.rows,
                        summed,
                        sign,
                        (atom_index, angular_momentum, projector_index * (2 * angular_momentum + 1) + harmonic_index),
                    )
                )
        return columns

    columns = [column for block in _map_atoms(atom_columns, len(atoms)) for column in block]
    indptr = np.zeros(len(columns) + 1, dtype=np.int64)
    rows = np.empty(0, dtype=np.int64)
    if columns:
        indptr[1:] = np.cumsum([column[0].size for column in columns])
        rows = np.concatenate([column[0] for column in columns])

    def assemble(component: int) -> sp.csc_matrix:
        if not columns:
            return sp.csc_matrix((grid.size, 0), dtype=np.float64)
        return sp.csc_matrix(
            (
                np.concatenate([column[1][:, component] for column in columns]),
                rows,
                indptr,
            ),
            shape=(grid.size, len(columns)),
        )

    return NonlocalProjectorGradients(
        components=(assemble(0), assemble(1), assemble(2)),
        signs=np.asarray([column[2] for column in columns], dtype=np.float64),
        labels=tuple(column[3] for column in columns),
    )


def build_projector_gradients(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
):
    """Projector gradients of an isolated system on its grid.

    The double-grid gradients of :func:`build_double_grid_projector_gradients`
    for ``Double_Grid_Order > 1``; otherwise the readable
    :func:`parsec_python.Forces.build_nonlocal_projector_gradients`, unchanged.
    """

    if double_grid_order(grid) > 1:
        return build_double_grid_projector_gradients(
            grid, atoms, potentials, specifications
        )
    from parsec_python.Forces import build_nonlocal_projector_gradients

    return build_nonlocal_projector_gradients(grid, atoms, potentials, specifications)


__all__ = [
    "box_double_grid_support",
    "build_double_grid_projector_gradients",
    "build_double_grid_projectors",
    "build_projector_gradients",
    "double_grid_order",
]
