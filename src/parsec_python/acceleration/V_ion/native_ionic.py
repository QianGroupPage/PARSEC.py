"""C++/OpenMP construction of PARSEC local and KB ionic grid fields.

The public objects returned here are the same NumPy arrays and
``NonlocalProjectorOperator`` used by :mod:`parsec_python`.  Only the
atom-by-grid loops, radial interpolation, and real-harmonic evaluation move to
native code; POTRE parsing, KB denominators, support rules, column ordering,
and sparse assembly stay visible in Python.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid
from parsec_python.models import Atom, SpeciesPotential
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.V_ion import NonlocalProjectorOperator

from ..backends import native as native_backend


_EMPTY = np.empty(0, dtype=np.float64)


def _values64(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _spline_payload(
    potential: ParsecPseudopotential,
    values: np.ndarray,
    grid: RealSpaceGrid,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        return _EMPTY, _EMPTY, _EMPTY
    spline = ParsecRadialSpline.from_positive_grid(
        potential.radii,
        values,
        grid.settings.stencil_half_width,
    )
    return (
        _values64(spline.knots),
        _values64(spline.values),
        _values64(spline.second_derivatives),
    )


def _projector_support_radius(potential: ParsecPseudopotential) -> float:
    # One rule for every builder: the next radial sample beyond the channel
    # cutoffs of a MARTINS_NEW file, Core_Cutoff_Radius for a MARTINS_OLD one.
    return potential.projector_support_radius


def _density_table(potential, specification, *, core=False):
    """Construct a species table once, retaining PARSEC's operation order."""
    if core:
        return _values64(potential.core_density), specification.use_spline
    if specification.read_valence_density:
        return _values64(potential.valence_density), False
    density = np.zeros_like(potential.radii, dtype=np.float64)
    for angular_momentum, wavefunction in potential.radial_wavefunctions.items():
        density += (
            potential.channel_occupations.get(angular_momentum, 0.0)
            * wavefunction * wavefunction
            / (4.0 * np.pi * potential.radii * potential.radii)
        )
    return _values64(density), False


def _projector_box_rows(grid, position, radius):
    """Conservative support box; the native kernel applies the exact sphere.

    One extra grid cell covers rounding at box faces. Sorting restores the
    original full-grid row order, including shifted and clipped domains.
    Uses the existing lookup, never constructs an atom-by-full-grid array.
    """
    position = np.asarray(position, dtype=np.float64)
    shift = np.asarray(grid.settings.shift)
    lower = np.floor((position - radius) / grid.spacing - shift).astype(np.int64) - 1
    upper = np.ceil((position + radius) / grid.spacing - shift).astype(np.int64) + 1
    lower = np.maximum(lower, grid.index_min) - grid.index_min
    upper = np.minimum(upper, grid.index_max) - grid.index_min
    if np.any(lower > upper):
        return np.empty(0, dtype=np.int64)
    rows = grid.lookup[tuple(slice(int(a), int(b) + 1) for a, b in zip(lower, upper))]
    return np.sort(rows[rows >= 0])


def _direct_interpolation(specification: SpeciesPotential) -> bool:
    """``Old_Interpolation_Format`` acts on a species without cubic spline."""

    return bool(getattr(specification, "old_interpolation", False)) and not specification.use_spline


def _coordinate_bounds(coordinates) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    return coordinates.min(axis=0), coordinates.max(axis=0)


def _farthest_distance(bounds, position) -> float:
    """An upper bound of the distance from ``position`` to the points in ``bounds``."""

    position = np.asarray(position, dtype=np.float64)
    corner = np.maximum(np.abs(bounds[0] - position), np.abs(bounds[1] - position))
    return float(np.sqrt(corner @ corner))


def _direct_local_potential(evaluator, bounds, atom, potential, values) -> np.ndarray:
    """``V_local`` of one atom with ``Old_Interpolation_Format`` (``ionpot.f90:107-112``).

    Inside ``rs(ns-1)`` the table ``V`` is interpolated linearly (its first
    value below the first sample), beyond it the Coulomb tail ``-2Z/r``, as
    :meth:`parsec_python.Pseudopotential.ParsecPseudopotential.local_potential`
    with ``direct=True``.  The native kernels give both parts: the density
    kernel interpolates a table linearly inside ``rs(ns-1)`` and is zero
    beyond, and the local-potential kernel of a zero table is zero inside
    and the Coulomb tail beyond.  One of the two is zero at every point, so
    their sum is exact; the tail is evaluated only where the grid reaches
    ``rs(ns-1)``.
    """

    position = _values64(atom.position)
    radii = _values64(potential.radii)
    field = np.asarray(
        evaluator.density(position, radii, values, _EMPTY, _EMPTY, _EMPTY),
        dtype=np.float64,
    )
    if _farthest_distance(bounds, atom.position) >= potential.interpolation_cutoff:
        field = field + np.asarray(
            evaluator.local_potential(
                position,
                radii,
                np.zeros_like(values),
                float(potential.ionic_charge),
                _EMPTY,
                _EMPTY,
                _EMPTY,
            ),
            dtype=np.float64,
        )
    return field


class NativeIonicBuilders:
    """Cache grid coordinates and expose reference-compatible setup hooks."""

    def __init__(self) -> None:
        self._grid_identity: int | None = None
        self._grid_size = 0
        self._evaluator = None

    def _for_grid(self, grid: RealSpaceGrid):
        identity = id(grid)
        if self._evaluator is None or self._grid_identity != identity:
            self._evaluator = native_backend._load_native().RadialGridEvaluator(
                _values64(grid.coordinates)
            )
            self._grid_identity = identity
            self._grid_size = grid.size
        return self._evaluator

    def build_local_ionic_potential(
        self,
        grid: RealSpaceGrid,
        atoms: Sequence[Atom],
        potentials: Mapping[str, ParsecPseudopotential],
        specifications: Mapping[str, SpeciesPotential],
    ) -> np.ndarray:
        """Build ``sum_a V_local,a`` with PARSEC rV/spline interpolation.

        A species with ``Old_Interpolation_Format`` (and no cubic spline,
        which takes precedence) interpolates ``V`` itself
        (:func:`_direct_local_potential`).
        """

        evaluator = self._for_grid(grid)
        total = np.zeros(grid.size, dtype=np.float64)
        tables = {}
        bounds = None
        for atom in atoms:
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            if atom.symbol not in tables:
                values = _values64(potential.channel_potentials[specification.local_angular_momentum])
                tables[atom.symbol] = (values, _spline_payload(
                    potential, values, grid, specification.use_spline
                ))
            values, spline = tables[atom.symbol]
            if _direct_interpolation(specification):
                if bounds is None:
                    bounds = _coordinate_bounds(grid.coordinates)
                total += _direct_local_potential(evaluator, bounds, atom, potential, values)
                continue
            total += np.asarray(
                evaluator.local_potential(
                    _values64(atom.position),
                    _values64(potential.radii),
                    values,
                    float(potential.ionic_charge),
                    *spline,
                ),
                dtype=np.float64,
            )
        return total

    def superpose_atomic_density(
        self,
        grid: RealSpaceGrid,
        atoms: Sequence[Atom],
        potentials: Mapping[str, ParsecPseudopotential],
        specifications: Mapping[str, SpeciesPotential],
        *,
        core: bool = False,
    ) -> np.ndarray:
        """Build initial valence or frozen NLCC density with native loops."""

        evaluator = self._for_grid(grid)
        total = np.zeros(grid.size, dtype=np.float64)
        tables = {}
        for atom in atoms:
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            if core and not potential.has_nonlinear_core_correction:
                continue
            if atom.symbol not in tables:
                radial_density, use_spline = _density_table(potential, specification, core=core)
                tables[atom.symbol] = (radial_density, _spline_payload(
                    potential, radial_density, grid, use_spline
                ))
            radial_density, spline = tables[atom.symbol]
            total += np.asarray(
                evaluator.density(
                    _values64(atom.position),
                    _values64(potential.radii),
                    radial_density,
                    *spline,
                ),
                dtype=np.float64,
            )
        return total

    def build_nonlocal_projectors(
        self,
        grid: RealSpaceGrid,
        atoms: Sequence[Atom],
        potentials: Mapping[str, ParsecPseudopotential],
        specifications: Mapping[str, SpeciesPotential],
    ) -> NonlocalProjectorOperator:
        """Build PARSEC-order KB sparse columns with native radial kernels.

        With ``Double_Grid_Order > 1`` the columns are averaged over the
        double grid (:mod:`.double_grid`, NumPy on the support boxes).
        """

        from .double_grid import build_double_grid_projectors, double_grid_order

        if double_grid_order(grid) > 1:
            return build_double_grid_projectors(grid, atoms, potentials, specifications)
        use_lookup = os.environ.get("PARSEC_NATIVE_PROJECTOR_LOOKUP", "1") == "1"
        evaluator = None if use_lookup else self._for_grid(grid)
        rows: list[np.ndarray] = []
        indptr = [0]
        values: list[np.ndarray] = []
        signs: list[float] = []
        labels: list[tuple[int, int, int]] = []
        column = 0
        square_root_volume = float(np.sqrt(grid.volume_element))
        tables = {}

        for atom_index, atom in enumerate(atoms):
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            local_l = specification.local_angular_momentum
            support_radius = _projector_support_radius(potential)
            box_rows = None
            if use_lookup:
                box_rows = _projector_box_rows(grid, atom.position, support_radius)
                evaluator = native_backend._load_native().RadialGridEvaluator(
                    _values64(grid.coordinates[box_rows])
                )
            for angular_momentum in sorted(potential.radial_wavefunctions):
                if angular_momentum == local_l:
                    continue
                key = (atom.symbol, angular_momentum)
                if key not in tables:
                    radial, denominator_sign = potential.radial_projector(angular_momentum, local_l)
                    radial = _values64(radial)
                    tables[key] = (radial, denominator_sign, _spline_payload(
                        potential, radial, grid, specification.use_spline
                    ))
                radial, denominator_sign, spline = tables[key]
                payload = evaluator.projector_channel(
                    _values64(atom.position),
                    _values64(potential.radii),
                    radial,
                    support_radius,
                    int(angular_momentum),
                    square_root_volume,
                    *spline,
                )
                support_rows = np.asarray(payload["rows"], dtype=np.int64)
                if box_rows is not None:
                    support_rows = box_rows[support_rows]
                channel_values = np.asarray(payload["values"], dtype=np.float64)
                for harmonic_index in range(channel_values.shape[1]):
                    projector = channel_values[:, harmonic_index]
                    keep = np.abs(projector) > 1.0e-16
                    kept_rows = support_rows[keep]
                    rows.append(kept_rows)
                    indptr.append(indptr[-1] + kept_rows.size)
                    values.append(projector[keep])
                    signs.append(denominator_sign)
                    labels.append(
                        (atom_index, angular_momentum, harmonic_index)
                    )
                    column += 1

        if column == 0:
            matrix = sp.csc_matrix((grid.size, 0), dtype=np.float64)
        else:
            matrix = sp.csc_matrix(
                (np.concatenate(values), np.concatenate(rows), np.asarray(indptr)),
                shape=(grid.size, column),
            )
        return NonlocalProjectorOperator(
            projectors=matrix,
            signs=np.asarray(signs, dtype=np.float64),
            labels=tuple(labels),
        )


__all__ = ["NativeIonicBuilders"]
