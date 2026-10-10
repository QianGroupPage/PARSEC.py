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
    requested = max(
        potential.channel_cutoffs.values(), default=potential.radii[0]
    )
    next_index = int(np.searchsorted(potential.radii, requested, side="right"))
    next_index = min(next_index, potential.radii.size - 2)
    return float(potential.radii[next_index])


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
        """Build ``sum_a V_local,a`` with PARSEC rV/spline interpolation."""

        evaluator = self._for_grid(grid)
        total = np.zeros(grid.size, dtype=np.float64)
        tables = {}
        for atom in atoms:
            potential = potentials[atom.symbol]
            specification = specifications[atom.symbol]
            if atom.symbol not in tables:
                values = _values64(potential.channel_potentials[specification.local_angular_momentum])
                tables[atom.symbol] = (values, _spline_payload(
                    potential, values, grid, specification.use_spline
                ))
            values, spline = tables[atom.symbol]
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
        """Build PARSEC-order KB sparse columns with native radial kernels."""

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
