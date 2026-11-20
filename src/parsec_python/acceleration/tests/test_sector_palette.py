"""The palette of the sector stencils of periodic cells: the bound, the count and the ladder of groups (2026-11-19).

A packed sector stencil codes its coefficients in one byte (256 values).  Those of a sector are the stencil values
times the phases +-1 and the ratios sqrt(m_row / m_column) of its orbit sizes, and the sums of the rows whose stencil
holds two points of one orbit.  A cube whose orbits differ in size, an even repeat of a smaller cell on n = 2 mod 4
points per edge, needs more at the default order 12: Si 2x2x2 at Grid_Spacing 0.6 (34^3 points) 295 values in the
weighted sectors of its point group and in the commuting group with translations, where the build of the stencils
stopped the run with PaletteOverflow.  The driver bounds every periodic sector cell (``sector_palette_bounds``),
counts where the bound exceeds the palette and steps down to a group whose stencils fit, the commuting signed
permutations about the origin here (``PARSEC_SECTOR_PALETTE_LADDER``).

The model tests (CPU) compare the bound with the count and the builder; the SCF tests (A100) the energies of the
ladder with the full grid.
"""

from __future__ import annotations

import atexit
from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    detect_point_group_reduction,
    periodic_abelian_reduction,
)
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.Symmetry.sector_stencil import (
    PALETTE_CAPACITY,
    PaletteOverflow,
    _numpy_sector_stencil,
    _padded_lookup,
    _stencil_columns,
    build_sector_stencils,
    merging_orbits,
    sector_palette_bounds,
    sector_palette_sizes,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

_SILICON_EXAMPLE = Path(__file__).resolve().parents[3].parent / "examples" / "3d_Si"
_SILICON_SIDE = 10.26
_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
_DIAMOND = np.vstack([_FCC, _FCC + 0.25])


def _grid(sides, points: int | None = None, *, spacing: float | None = None, order: int = 12, shift: float = 0.5):
    """The grid the parser gives the default routines (``points_rule="fortran"``), with ``points`` per edge of the
    first side or at ``spacing``."""

    sides = np.asarray(sides, dtype=float)
    if spacing is None:
        spacing = float(sides[0]) / points * (1.0 - 1.0e-9)
    return build_periodic_grid(
        PeriodicCell(lattice_vectors=np.diag(sides)),
        PeriodicGridSettings(spacing=spacing, expansion_order=order, shift=(shift,) * 3, points_rule="fortran"),
    )


def _diamond(side: float, repeat: int, symbol: str = "C"):
    return [
        Atom(symbol, tuple((position + np.array([i, j, k])) * side))
        for i in range(repeat)
        for j in range(repeat)
        for k in range(repeat)
        for position in _DIAMOND
    ]


def _groups(grid, atoms):
    """The groups of the ladder: the weighted sectors, the commuting group with translations, the one about the
    origin."""

    origin = SignedPermutationReduction.detect(grid, atoms)
    return (
        detect_point_group_reduction(grid, atoms, origin),
        periodic_abelian_reduction(grid, atoms, origin),
        origin,
    )


def _decomposition(grid, reduction):
    return ReflectionRepresentationDecomposition.build(grid, reduction)


class ColumnTests(unittest.TestCase):
    def test_the_columns_of_a_cell_without_its_wrapped_lookup(self) -> None:
        # the counts take the columns of a periodic cell by index arithmetic: those of the wrapped lookup
        for sides, points, shift in (((8.0,) * 3, 14, 0.5), ((8.0, 8.0, 9.5), 14, 0.0), ((7.0, 8.0, 9.0), 13, 0.5)):
            grid = _grid(sides, points, order=8, shift=shift)
            rows = np.arange(grid.size, dtype=np.int64)
            padded = _stencil_columns(grid, 4, _padded_lookup(grid, 4, True))(rows)
            np.testing.assert_array_equal(_stencil_columns(grid, 4, None)(rows), padded, err_msg=str(sides))


class BoundTests(unittest.TestCase):
    """``sector_palette_bounds``: the whole table and the exact palette of the rows near a symmetry element."""

    cases = (
        # (label, side, repeat, points per edge, order): the benchmark's 18 points per edge, the mixed orbit sizes of
        # n = 2 mod 4 at the default order, a cube at order 20
        ("diamond 2x2x2 n=36 order 8", 6.74, 2, 36, 8),
        ("diamond 2x2x2 n=34 order 12", 6.74, 2, 34, 12),
        ("diamond 1x1x1 n=22 order 20", 6.74, 1, 22, 20),
    )

    def test_the_bound_holds_the_count(self) -> None:
        for label, side, repeat, points, order in self.cases:
            grid = _grid((side * repeat,) * 3, points, order=order)
            reduction = _groups(grid, _diamond(side, repeat))[0]
            self.assertIsInstance(reduction, PointGroupReduction, label)
            decomposition = _decomposition(grid, reduction)
            bounds = sector_palette_bounds(grid, decomposition, wrap=True)
            sizes = sector_palette_sizes(grid, decomposition, wrap=True)
            for bound, size in zip(bounds, sizes):
                self.assertGreaterEqual(bound, size, label)
            if np.unique(reduction.multiplicities).size == 1:
                # one orbit size: the table is the stencil values and their signs alone
                self.assertLessEqual(max(bounds), max(sizes) + 2 * (1 + order // 2), label)
            # the requested sectors alone
            some = (1, len(sizes) - 1)
            self.assertEqual(
                sector_palette_sizes(grid, decomposition, some, wrap=True),
                tuple(size if index in some else None for index, size in enumerate(sizes)),
            )

    def test_the_rows_left_out_add_no_entries(self) -> None:
        # The rows the pass does not flag add no two entries of one sector column in any sector: their palette is
        # within the table (the builder finds the sums from the sector columns, the pass from the orbits).
        grid = _grid((13.48,) * 3, 34, order=12)
        reduction = _groups(grid, _diamond(6.74, 2))[0]
        decomposition = _decomposition(grid, reduction)
        merging = merging_orbits(grid, decomposition, wrap=True)
        self.assertTrue(0 < np.count_nonzero(merging) < merging.size)
        table = _numpy_sector_stencil(
            grid, decomposition, 0, None, palette_only=True, orbit_mask=np.zeros_like(merging)
        )
        for index in range(decomposition.representation_count):
            left_out = _numpy_sector_stencil(grid, decomposition, index, None, palette_only=True, orbit_mask=~merging)
            np.testing.assert_array_equal(left_out, table)
            flagged = _numpy_sector_stencil(grid, decomposition, index, None, palette_only=True, orbit_mask=merging)
            self.assertTrue(set(flagged.tolist()) >= set(table.tolist()))

    def test_the_benchmark_cells_need_no_count(self) -> None:
        # 18 points per edge of the diamond benchmarks (one orbit size, order 8): the bound proves the palette, the
        # count is not asked for
        from parsec_python.acceleration import driver

        grid = _grid((4 * 6.7406,) * 3, 72, order=8)
        reduction = _groups(grid, _diamond(6.7406, 4))[0]

        def count(*args, **kwargs):
            raise AssertionError("counted")

        checked, refusal = driver._sector_palette_check(
            grid, reduction, lambda orbital: (_decomposition(grid, orbital), None), count
        )
        self.assertIsNone(refusal)
        self.assertIs(checked[0], reduction)
        self.assertLessEqual(max(sector_palette_bounds(grid, checked[1][0], wrap=True)), 64)


class LadderTests(unittest.TestCase):
    """Si 2x2x2 (64 atoms) at Grid_Spacing 0.6 and the default order 12: 34^3 points, three orbit sizes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = _grid((2 * _SILICON_SIDE,) * 3, spacing=0.6, order=12)
        cls.atoms = _diamond(_SILICON_SIDE, 2, "Si")
        cls.groups = _groups(cls.grid, cls.atoms)

    def _check(self, reduction):
        from parsec_python.acceleration import driver

        return driver._sector_palette_check(
            self.grid, reduction, lambda orbital: (_decomposition(self.grid, orbital), None)
        )

    def test_the_groups_and_their_palettes(self) -> None:
        self.assertEqual(tuple(self.grid.points_per_axis), (34, 34, 34))
        weighted, abelian, origin = self.groups
        self.assertIsInstance(weighted, PointGroupReduction)
        self.assertGreater(np.unique(weighted.multiplicities).size, 1)
        self.assertEqual(abelian.group_order, 32)
        self.assertEqual(origin.group_order, 4)
        decomposition = _decomposition(self.grid, weighted)
        sizes = sector_palette_sizes(self.grid, decomposition, wrap=True)
        self.assertEqual(max(sizes), 295)
        # the build stops where the count says
        with self.assertRaises(PaletteOverflow) as caught:
            build_sector_stencils(self.grid, decomposition, implementation="numpy", wrap=True)
        self.assertIn(f"more than {PALETTE_CAPACITY} coefficients", str(caught.exception))
        for reduction, fits in ((weighted, False), (abelian, False), (origin, True)):
            checked, refusal = self._check(reduction)
            self.assertEqual(refusal is None, fits, type(reduction).__name__)
            if fits:
                self.assertIs(checked[0], reduction)
                stencils = build_sector_stencils(self.grid, checked[1][0], implementation="numpy", wrap=True)
                self.assertLessEqual(max(int(s.coefficient_palette.size) for s in stencils), PALETTE_CAPACITY)
            else:
                self.assertIsNone(checked)
                self.assertIn(f"more than the {PALETTE_CAPACITY} of their palette (Expansion_Order 12)", refusal)
        self.assertIn("need 295 distinct coefficients", self._check(weighted)[1])

    def test_the_operator_builder_says_why(self) -> None:
        # Reached only where the driver's check did not run: PaletteOverflow with what avoids it, from the stencils
        # of the grid, not after the full-grid matrix of the CSR route
        from parsec_python.acceleration.Laplacian import DeferredPeriodicNegativeLaplacian
        from parsec_python.acceleration.Symmetry.operator_cache import _direct_sector_stencils

        decomposition = _decomposition(self.grid, self.groups[0])
        with self.assertRaises(PaletteOverflow) as caught:
            _direct_sector_stencils(decomposition, DeferredPeriodicNegativeLaplacian(self.grid), None)
        self.assertIn("a lower Expansion_Order", str(caught.exception))

    def test_the_switch(self) -> None:
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_SECTOR_PALETTE_LADDER", None)
            self.assertEqual(driver.sector_palette_ladder_setting(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PALETTE_LADDER": "off"}):
            self.assertEqual(driver.sector_palette_ladder_setting(), "off")
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PALETTE_LADDER": "abelian"}):
            with self.assertRaises(ValueError):
                driver.sector_palette_ladder_setting()


def silicon_supercell(repeat: int, spacing: float, *, order: int = 12, functional: str = "pbe"):
    """The conventional Si cube of ``examples/3d_Si`` repeated ``repeat`` times along each axis, SCF to 1e-9."""

    from parsec_python.Input.parsec_input import parse_parsec_input

    side = _SILICON_SIDE * repeat
    atoms = _diamond(_SILICON_SIDE, repeat, "Si")
    text = f"""Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  {side:.6f} {side:.6f} {side:.6f}
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: {spacing}
Expansion_Order: {order}
States_Num: {2 * len(atoms) + 8}
Fermi_Temp: 300.0
Max_Iter: 100
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-10
Mixing_Method: Anderson
Mixing_Param: 0.3
Correlation_Type: {functional}
Skip_force: .true.
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
""" + "".join(f"  {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in (atom.position for atom in atoms)) + "end Atom_Coord\n"
    directory = Path(tempfile.mkdtemp(prefix="parsec_palette_"))
    atexit.register(shutil.rmtree, directory, True)
    shutil.copy(_SILICON_EXAMPLE / "Si_POTRE.DAT", directory)
    path = directory / "parsec.in"
    path.write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_parsec_input(path).problem


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class PaletteLadderSCFTests(unittest.TestCase):
    def _run(self, problem, environment):
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            result = driver.run_scf(system)
        return result, dict(result.backend.details)

    def _assert_same(self, sectors, full, label, energy: float = 1.0e-8) -> None:
        self.assertTrue(sectors.converged and full.converged, label)
        for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
            delta = energy if field == "total" else max(energy, 1.0e-7)
            self.assertAlmostEqual(
                getattr(sectors.energies, field), getattr(full.energies, field), delta=delta, msg=f"{label} {field}"
            )

    def test_si_2x2x2_at_the_default_order(self) -> None:
        # 34^3 points, order 12: the weighted sectors (295 values) and the commuting group with translations (295)
        # do not fit, D2 about the origin does; the build of the stencils stopped the run before 2026-11-19
        problem = silicon_supercell(2, 0.6)
        self.assertEqual(problem.grid.expansion_order, 12)
        sectors, details = self._run(problem, {})
        detection = details["symmetry_detection"]
        self.assertIn("PARSEC_SECTOR_PALETTE_LADDER=auto", detection)
        self.assertIn("need 295 distinct coefficients", detection)
        self.assertIn("the sectors of the 4 commuting signed permutations about the origin", detection)
        self.assertIn("PARSEC_SECTOR_PALETTE_LADDER=auto", details.get("symmetry_full_grid_note", ""))
        self.assertNotIn("full grid", details["orbital_symmetry"])
        kept, kept_details = self._run(problem, {"PARSEC_SECTOR_PALETTE_LADDER": "off"})
        self.assertIn("fell back to full grid", kept_details["symmetry_detection"])
        full, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        self._assert_same(sectors, full, "Si 2x2x2 in the sectors of D2")
        self._assert_same(kept, full, "Si 2x2x2 on the full grid")


if __name__ == "__main__":
    unittest.main()
