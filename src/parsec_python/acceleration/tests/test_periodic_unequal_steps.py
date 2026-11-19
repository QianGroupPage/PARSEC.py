"""Bulk cells whose grid steps differ between the axes, in the sectors of their point group (stream C of 2026-11-18).

The input-file parser gives the default routines Fortran's ``adjust_step`` counts (``points_rule="fortran"``), so an
orthorhombic or tetragonal cell gets a different step on its axes (10.26 x 10.26 x 12 bohr at 0.64 bohr: 16 x 16 x 18
points, 0.641 / 0.641 / 0.667).  Its grid is a :class:`~parsec_python.Grid.pbc.PeriodicRealSpaceGrid` that is not
``is_orthorhombic``: the detection refused it and the run kept the full grid.  ``PARSEC_PERIODIC_UNEQUAL_STEPS=auto``
gives it the sectors: the mirrors and two-fold axes of every axis and the exchanges of axes of one count and side
only, the sector stencils with a step per axis, the deferred wrapped Laplacian (the symbol of the FFT Hartree solve
from its first row), the device XC with a row of weights per axis and the projectors of all atoms at once.

The model tests (CPU) compare the weighted spectra of the sectors with the full grid and the arrays of every builder
with those of the full-grid matrix; the SCF tests (A100) the energies and forces with ``PARSEC_PERIODIC_SYMMETRY=off``.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.pbc import PeriodicRealSpaceGrid, build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.acceleration.Laplacian import DeferredPeriodicNegativeLaplacian, periodic_negative_laplacian
from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    _lattice_period,
    detect_point_group_reduction,
    periodic_unequal_steps_setting,
    unequal_step_grid,
)
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.Symmetry.sector_stencil import (
    PALETTE_CAPACITY,
    PaletteOverflow,
    SectorStencilUnavailable,
    axis_stencil_values,
    build_sector_stencils,
    sector_palette_sizes,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

_SWITCH = "PARSEC_PERIODIC_UNEQUAL_STEPS"
_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
_DIAMOND = np.vstack([_FCC, _FCC + 0.25])


def _grid(sides, spacing, *, order: int = 4, shift: float = 0.5):
    """The grid the parser gives the default routines (``points_rule="fortran"``)."""

    return build_periodic_grid(
        PeriodicCell(lattice_vectors=np.diag(np.asarray(sides, dtype=float))),
        PeriodicGridSettings(spacing=spacing, expansion_order=order, shift=(shift,) * 3, points_rule="fortran"),
    )


def _hamiltonian(grid, atoms, depth: float = 2.0):
    """The Laplacian of the full grid and Gaussian wells with their nearest images."""

    sides = np.diag(np.asarray(grid.cell.lattice_vectors, dtype=float))
    potential = np.zeros(grid.size)
    for atom in atoms:
        difference = np.asarray(grid.coordinates) - np.asarray(atom.position)
        difference -= sides * np.rint(difference / sides)
        potential -= depth * np.exp(-np.sum(difference * difference, axis=1) / 0.9)
    return (0.5 * build_negative_laplacian(grid) + sp.diags(potential)).tocsr()


def _stretched_diamond(sides):
    return [Atom("C", tuple(position)) for position in _DIAMOND * np.asarray(sides, dtype=float)]


def _weighted_spectrum(decomposition, hamiltonian, count):
    weighted = []
    for index, matrix in enumerate(decomposition.reduce_operators(hamiltonian)):
        values = np.linalg.eigvalsh(matrix.toarray())[:count]
        weighted += [(value, int(decomposition.sector_weights[index])) for value in values]
    return np.array([value for value, weight in sorted(weighted) for _ in range(weight)][:count])


def _native_reduction() -> bool:
    from parsec_python.acceleration.backends.native import _load_native

    return native_available() and hasattr(_load_native(), "reduce_sector_csr")


def _reduce(decomposition, matrix):
    """The sectors of ``matrix`` by the native reduction where the extension has it (the order of the packed
    stencils' sums), by SciPy's otherwise."""

    with mock.patch.dict(os.environ, {"PARSEC_NATIVE_SECTOR_ASSEMBLY": "1" if _native_reduction() else "0"}):
        return decomposition.reduce_operators(matrix)


def _detect(grid, atoms):
    return detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))


class SwitchTests(unittest.TestCase):
    def test_the_switch(self) -> None:
        for value, expected in (("", "auto"), ("auto", "auto"), (" OFF ", "off")):
            with mock.patch.dict(os.environ, {_SWITCH: value}):
                self.assertEqual(periodic_unequal_steps_setting(), expected)
        with mock.patch.dict(os.environ, {_SWITCH: "on"}), self.assertRaises(ValueError):
            periodic_unequal_steps_setting()

    def test_which_grids_have_unequal_steps(self) -> None:
        tetragonal = _grid((6.0, 6.0, 7.2), 0.75)
        self.assertIs(type(tetragonal), PeriodicRealSpaceGrid)
        np.testing.assert_array_equal(tetragonal.points_per_axis, [8, 8, 8])
        self.assertFalse(tetragonal.is_orthorhombic)
        self.assertTrue(unequal_step_grid(tetragonal))
        # a cube of one step, an oblique cell and the general grid of the Fortran-derived routines are not
        self.assertFalse(unequal_step_grid(_grid((6.0, 6.0, 6.0), 0.75)))
        oblique = build_periodic_grid(
            PeriodicCell(lattice_vectors=np.array([[6.0, 0.0, 0.0], [3.0, 5.2, 0.0], [0.0, 0.0, 7.0]])),
            PeriodicGridSettings(spacing=0.75, expansion_order=4, points_rule="fortran"),
        )
        self.assertFalse(unequal_step_grid(oblique))
        general = build_periodic_grid(
            PeriodicCell(lattice_vectors=np.diag([6.0, 6.0, 7.2])),
            PeriodicGridSettings(spacing=0.75, expansion_order=4, shift=(0.5,) * 3, fortran_routines=True),
        )
        self.assertTrue(getattr(general, "is_general_periodic", False))
        self.assertFalse(unequal_step_grid(general))

    def test_the_parser_gives_such_a_grid(self) -> None:
        # what users hit: an input file of a tetragonal cell on the default routines
        problem = tetragonal_silicon(12.0, 0.64)
        self.assertEqual(problem.grid.points_rule, "fortran")
        self.assertFalse(problem.grid.fortran_routines)
        grid = build_periodic_grid(problem.periodic_cell, problem.grid)
        np.testing.assert_array_equal(grid.points_per_axis, [16, 16, 18])
        self.assertTrue(unequal_step_grid(grid))


class DetectionTests(unittest.TestCase):
    def test_the_lattice_period_and_the_switch(self) -> None:
        import tempfile

        from parsec_python.acceleration.Symmetry.geometry_cache import load_or_detect_reflection_reduction

        grid = _grid((6.0, 6.0, 7.2), 0.75)
        np.testing.assert_array_equal(_lattice_period(grid), [6.0, 6.0, 7.2])
        with mock.patch.dict(os.environ, {_SWITCH: "off"}), self.assertRaisesRegex(ValueError, _SWITCH):
            _lattice_period(grid)
        # a symmetry cache written under auto is not read under off
        atoms = _stretched_diamond((6.0, 6.0, 7.2))
        with tempfile.TemporaryDirectory() as directory:
            reduction, info = load_or_detect_reflection_reduction(grid, atoms, cache_directory=directory)
            self.assertEqual(int(reduction.group_order), 4)
            _, again = load_or_detect_reflection_reduction(grid, atoms, cache_directory=directory)
            self.assertEqual(again.status, "hit")
            with mock.patch.dict(os.environ, {_SWITCH: "off"}), self.assertRaisesRegex(ValueError, _SWITCH):
                load_or_detect_reflection_reduction(grid, atoms, cache_directory=directory)

    def test_a_tetragonal_cell_keeps_its_group(self) -> None:
        sides = (6.0, 6.0, 7.2)
        for shift in (0.5, 0.0):
            with self.subTest(shift=shift):
                grid = _grid(sides, 0.75, shift=shift)
                atoms = _stretched_diamond(sides)
                reduction = _detect(grid, atoms)
                self.assertIsInstance(reduction, PointGroupReduction)
                operations = np.asarray(reduction.full_operations)
                # the operations of the stretched crystal that keep z: no exchange of z with x or y
                self.assertTrue(np.all(np.abs(operations[:, 2, 2]) == 1))
                self.assertTrue(np.any(np.abs(operations[:, 0, 1]) == 1))
                self.assertGreaterEqual(operations.shape[0], 32)
                hamiltonian = _hamiltonian(grid, atoms)
                decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
                count = 40
                np.testing.assert_allclose(
                    _weighted_spectrum(decomposition, hamiltonian, count),
                    np.linalg.eigvalsh(hamiltonian.toarray())[:count],
                    rtol=0.0,
                    atol=1.0e-11,
                )

    def test_three_different_steps_keep_the_mirrors(self) -> None:
        # equal counts, different sides: no axis is exchanged with another
        sides = (6.0, 6.6, 7.2)
        grid = _grid(sides, 0.75)
        np.testing.assert_array_equal(grid.points_per_axis, [8, 8, 8])
        atoms = [Atom("C", (0.0, 0.0, 0.0)), Atom("O", (3.0, 3.3, 3.6)), Atom("C", (1.1, 0.0, 0.0)), Atom("C", (-1.1, 0.0, 0.0))]
        reduction = _detect(grid, atoms)
        operations = np.asarray(
            reduction.full_operations
            if isinstance(reduction, PointGroupReduction)
            else [np.diag(signs) for signs in reduction.signs]
        )
        self.assertTrue(np.all(np.abs(operations) == np.eye(3, dtype=int)[None]))
        self.assertEqual(int(reduction.group_order), 8)
        hamiltonian = _hamiltonian(grid, atoms)
        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
        np.testing.assert_allclose(
            _weighted_spectrum(decomposition, hamiltonian, 30),
            np.linalg.eigvalsh(hamiltonian.toarray())[:30],
            rtol=0.0,
            atol=1.0e-11,
        )


class StencilTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sides = (6.0, 6.0, 7.4)
        self.grid = _grid(self.sides, 0.6, order=8)
        self.reduction = _detect(self.grid, _stretched_diamond(self.sides))
        self.decomposition = ReflectionRepresentationDecomposition.build(self.grid, self.reduction)

    def test_the_values_are_those_of_the_full_grid_matrix(self) -> None:
        matrix = build_negative_laplacian(self.grid)
        diagonal, shells = axis_stencil_values(self.grid)
        row = matrix.getrow(0).tocoo()
        entries = dict(zip(row.col.tolist(), row.data.tolist()))
        self.assertEqual(entries[0], diagonal)
        shape = tuple(int(value) for value in self.grid.points_per_axis)
        for axis in range(3):
            for shell in range(1, 5):
                point = [0, 0, 0]
                point[axis] = shell
                self.assertEqual(entries[int(np.ravel_multi_index(point, shape))], shells[axis][shell - 1])
        self.assertNotEqual(shells[0][0], shells[2][0])
        self.assertEqual(shells[0], shells[1])

    def assert_reduced(self, stencil, expected, label) -> None:
        """The packed stencil against the reduced full-grid matrix: equal to the native reduction, within the last
        bits of SciPy's (which adds the repeated columns of a row in another order)."""

        if _native_reduction():
            np.testing.assert_array_equal(stencil.to_csr().toarray(), expected.toarray(), err_msg=label)
        else:
            np.testing.assert_allclose(stencil.to_csr().toarray(), expected.toarray(), rtol=0, atol=2.0e-14, err_msg=label)

    def test_sector_stencils_are_the_reduced_matrix(self) -> None:
        matrix = build_negative_laplacian(self.grid)
        reduced = _reduce(self.decomposition, matrix)
        stencils = build_sector_stencils(self.grid, self.decomposition, implementation="numpy", wrap=True)
        for index, (stencil, expected) in enumerate(zip(stencils, reduced)):
            self.assert_reduced(stencil, expected, f"sector {index}")
        with self.assertRaises(SectorStencilUnavailable):
            build_sector_stencils(self.grid, self.decomposition, implementation="native")

    def test_the_deferred_laplacian(self) -> None:
        from parsec_python.acceleration.Symmetry.operator_cache import _direct_sector_stencils, _wrapped_stencil_grid

        deferred = periodic_negative_laplacian(self.grid)
        self.assertIsInstance(deferred, DeferredPeriodicNegativeLaplacian)
        matrix = build_negative_laplacian(self.grid)
        self.assertEqual(deferred.nnz, matrix.nnz)
        for index in (0, 1, 77, self.grid.size // 2, self.grid.size - 1):
            np.testing.assert_array_equal(deferred.getrow(index).toarray(), matrix.getrow(index).toarray())
        self.assertIs(_wrapped_stencil_grid(deferred), self.grid)
        metadata, route = _direct_sector_stencils(self.decomposition, deferred, None)
        self.assertEqual(route, "direct-numpy-wrapped")
        self.assertFalse(deferred.materialized)
        self.assert_reduced(metadata[0], _reduce(self.decomposition, matrix)[0], "deferred")
        built = deferred.materialize()
        self.assertEqual((built != matrix).nnz, 0)
        # a step per axis in the key: the same counts and shortest step, another z step
        other = DeferredPeriodicNegativeLaplacian(_grid((6.0, 6.0, 7.6), 0.6, order=8))
        self.assertNotEqual(deferred.cache_key, other.cache_key)
        cube = DeferredPeriodicNegativeLaplacian(_grid((6.0, 6.0, 6.0), 0.6, order=8))
        self.assertNotEqual(cube.cache_key, deferred.cache_key)
        # off: the matrix at once, as before
        with mock.patch.dict(os.environ, {_SWITCH: "off"}):
            self.assertTrue(sp.issparse(periodic_negative_laplacian(self.grid)))


class PaletteTests(unittest.TestCase):
    """The palette of a packed stencil holds 256 coefficients (one-byte codes).  Such a cell has two or three times the
    shell values of a cube, each times the phases and multiplicity ratios of a sector: Si stretched along z at order 18
    needs 262 in one sector, where the build of the stencils stopped the run (C2 of review 5).  The driver counts them
    first and keeps such a cell on the full grid."""

    sides = (10.26, 10.26, 12.0)

    def _sectors(self, spacing: float, order: int):
        grid = _grid(self.sides, spacing, order=order)
        reduction = _detect(grid, _stretched_diamond(self.sides))
        return grid, reduction, ReflectionRepresentationDecomposition.build(grid, reduction)

    @staticmethod
    def _check(grid, reduction):
        from parsec_python.acceleration import driver

        asked = []

        def build(orbital):
            asked.append(orbital)
            return ReflectionRepresentationDecomposition.build(grid, orbital), None

        return driver._sector_palette_check(grid, reduction, build), asked

    def test_the_count_is_that_of_the_built_stencils(self) -> None:
        grid, reduction, decomposition = self._sectors(0.64, 12)
        sizes = sector_palette_sizes(grid, decomposition, wrap=True)
        stencils = build_sector_stencils(grid, decomposition, implementation="numpy", wrap=True)
        self.assertEqual(sizes, tuple(int(stencil.coefficient_palette.size) for stencil in stencils))
        self.assertLessEqual(max(sizes), PALETTE_CAPACITY)
        (checked, refusal), asked = self._check(grid, reduction)
        self.assertIsNone(refusal)
        self.assertIs(checked[0], reduction)
        self.assertEqual(asked, [reduction])

    def test_a_cell_beyond_the_palette_keeps_the_full_grid(self) -> None:
        grid, reduction, decomposition = self._sectors(0.5, 18)
        self.assertTrue(unequal_step_grid(grid))
        self.assertIsInstance(reduction, PointGroupReduction)
        sizes = sector_palette_sizes(grid, decomposition, wrap=True)
        self.assertGreater(max(sizes), PALETTE_CAPACITY)
        # Both routes of the stencils refuse it with one error, a ValueError as before; not SectorStencilUnavailable,
        # which would send the direct build to the CSR route.
        with self.assertRaises(PaletteOverflow) as caught:
            build_sector_stencils(grid, decomposition, implementation="numpy", wrap=True)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertNotIsInstance(caught.exception, SectorStencilUnavailable)
        from parsec_python.acceleration.backends.cupy_stencil_major import build_stencil_major_metadata

        worst = int(np.argmax(sizes))
        with self.assertRaises(PaletteOverflow):
            build_stencil_major_metadata(_reduce(decomposition, build_negative_laplacian(grid))[worst])
        (checked, refusal), asked = self._check(grid, reduction)
        self.assertIsNone(checked)
        self.assertIn(f"need {max(sizes)} distinct coefficients, more than the {PALETTE_CAPACITY}", refusal)
        self.assertEqual(asked, [reduction])

    def test_cubes_are_checked_too(self) -> None:
        # since 2026-11-19 (tests/test_sector_palette.py): a cube whose orbits differ in size overflows too
        cube = _grid((10.26,) * 3, 0.5, order=18)
        self.assertFalse(unequal_step_grid(cube))
        reduction = _detect(cube, _stretched_diamond((10.26,) * 3))
        self.assertGreater(reduction.group_order, 1)
        sizes = sector_palette_sizes(cube, ReflectionRepresentationDecomposition.build(cube, reduction), wrap=True)
        (checked, refusal), asked = self._check(cube, reduction)
        self.assertEqual(asked, [reduction])
        if max(sizes) <= PALETTE_CAPACITY:
            self.assertIsNone(refusal)
            self.assertIs(checked[0], reduction)
        else:
            self.assertIsNone(checked)
            self.assertIn(f"need {max(sizes)} distinct coefficients", refusal)
        grid = _grid(self.sides, 0.5, order=18)
        self.assertEqual(self._check(grid, None), ((None, None), []))


class ProjectorTests(unittest.TestCase):
    def test_the_projectors_of_all_atoms_at_once(self) -> None:
        from parsec_python.V_ion import ionic_potential
        from parsec_python.acceleration.V_ion.periodic_projectors import batched_nonlocal_projectors
        from parsec_python.acceleration.tests.test_periodic_projectors import _SILICON, _reference
        from parsec_python.models import SpeciesPotential

        sides = np.array([10.26, 10.26, 12.0])
        lattice = np.diag(sides)
        grid = _grid(sides, 0.64, order=8)
        self.assertTrue(unequal_step_grid(grid))
        atoms = [Atom("Si", tuple(position)) for position in _DIAMOND * sides]
        specifications = {"Si": SpeciesPotential(_SILICON, 2, use_spline=True)}
        potentials = ionic_potential.load_pseudopotentials(specifications, xc_functional="pbe")
        batched = batched_nonlocal_projectors(grid, atoms, potentials, specifications, lattice)
        self.assertIsNotNone(batched)
        reference = _reference(grid, atoms, potentials, specifications, lattice)
        for name in ("indptr", "indices", "data"):
            np.testing.assert_array_equal(getattr(batched.projectors, name), getattr(reference.projectors, name))
        np.testing.assert_array_equal(batched.signs, reference.signs)
        self.assertEqual(batched.labels, reference.labels)
        with mock.patch.dict(os.environ, {_SWITCH: "off"}):
            self.assertIsNone(batched_nonlocal_projectors(grid, atoms, potentials, specifications, lattice))


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceXCTests(unittest.TestCase):
    def test_gradients_with_a_step_per_axis(self) -> None:
        from parsec_python.V_xc.selection import evaluate_unpolarized_xc
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.V_xc import device_xc
        from parsec_python.acceleration.tests.test_device_xc import FIELD_TOLERANCE, _relative

        sides = np.array([6.0, 6.0, 7.2])
        grid = _grid(sides, 0.5, order=8)
        self.assertTrue(unequal_step_grid(grid))
        self.assertTrue(device_xc.device_grid_supported(grid))
        with mock.patch.dict(os.environ, {_SWITCH: "off"}):
            self.assertFalse(device_xc.device_grid_supported(grid))
        atoms = _stretched_diamond(sides)
        density = np.full(grid.size, 1.0e-3)
        for atom in atoms:
            difference = grid.coordinates - np.asarray(atom.position)
            difference -= np.round(difference / sides) * sides
            density += np.exp(-np.sum(difference * difference, axis=1) / 1.1)
        reduction = _detect(grid, atoms)
        for reducer in (None, SymmetrySCFReducer(reduction)):
            values = density if reducer is None else reducer.from_full(density)
            full = density if reducer is None else reducer.to_full(values)
            for functional in ("pbe", "pw91", "lb94"):
                label = f"{functional} {'full grid' if reducer is None else 'orbits'}"
                reference = evaluate_unpolarized_xc(functional, full, grid, None)
                value = device_xc.DeviceXCEvaluator(functional, grid, reducer=reducer)(values)
                expected = reference.potential if reducer is None else reducer.wedge_values(reference.potential)
                self.assertLess(_relative(expected, value.potential), 20 * FIELD_TOLERANCE, label)
                self.assertAlmostEqual(
                    value.total_energy, reference.total_energy, delta=20 * FIELD_TOLERANCE * abs(reference.total_energy),
                    msg=label,
                )


def tetragonal_silicon(c_side: float, spacing: float, *, displaced: bool = False, functional: str = "pbe"):
    """The conventional Si cube of ``examples/3d_Si`` stretched along z to ``c_side`` bohr (the atoms scaled with it):
    D2d about the atom at the origin and the face-centring translations; ``displaced`` moves the four neighbours of
    that atom 0.04 bohr outward along their bonds (D2d stays, the translations go, the atoms have forces)."""

    from parsec_python.acceleration.tests.test_periodic_point_group import silicon

    base = silicon(functional, spacing=spacing)
    lattice = np.asarray(base.periodic_cell.lattice_vectors, dtype=np.float64)
    scale = np.array([1.0, 1.0, c_side / float(lattice[0, 0])])
    sides = np.diag(lattice) * scale
    atoms = []
    for atom in base.atoms:
        position = np.asarray(atom.position, dtype=np.float64) * scale
        bond = position - sides * np.round(position / sides)
        if displaced and 0.0 < np.linalg.norm(bond) < 0.25 * np.linalg.norm(sides) + 1.0e-6:
            position = position + 0.04 * bond / np.linalg.norm(bond)
        atoms.append(Atom(atom.symbol, tuple(position)))
    return replace(
        base,
        atoms=tuple(atoms),
        periodic_cell=replace(base.periodic_cell, lattice_vectors=np.diag(sides)),
        grid=replace(base.grid, box_lengths=tuple(sides)),
    )


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class UnequalStepSCFTests(unittest.TestCase):
    def _run(self, problem, environment, *, forces: bool = False):
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            result = driver.run_scf(system)
            table = driver.compute_forces(system, result, symmetrize=False) if forces else None
        return result, dict(result.backend.details), table

    def _assert_same(self, sectors, full, label, energy: float = 1.0e-8) -> None:
        self.assertTrue(sectors.converged and full.converged, label)
        for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
            delta = energy if field == "total" else max(energy, 1.0e-7)
            self.assertAlmostEqual(
                getattr(sectors.energies, field), getattr(full.energies, field), delta=delta, msg=f"{label} {field}"
            )

    def test_a_tetragonal_cell_in_its_sectors(self) -> None:
        problem = tetragonal_silicon(12.0, 0.64)
        sectors, details, _ = self._run(problem, {})
        self.assertIn("exact group of 32 operations", details["symmetry_detection"])
        self.assertEqual(details["orbital_operator_stencil_builder"], "direct-numpy-wrapped")
        self.assertEqual(details["finite_difference_full_grid_materialization"], "skipped_by_direct_sector_stencil")
        self.assertIn("CuPy", details["xc_implementation"])
        before, before_details, _ = self._run(problem, {_SWITCH: "off"})
        self.assertIn("fell back to full grid", before_details["symmetry_detection"])
        self.assertIn("NumPy", before_details["xc_implementation"])
        full, full_details, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        self.assertIn("CuPy", full_details["xc_implementation"])
        self._assert_same(sectors, full, "tetragonal Si")
        self._assert_same(before, full, "tetragonal Si before")

    def test_forces_of_a_tetragonal_cell(self) -> None:
        problem = tetragonal_silicon(12.0, 0.64, displaced=True)
        sectors, details, sector_forces = self._run(problem, {}, forces=True)
        self.assertIn("exact point group of 8", details["symmetry_detection"])
        full, _, full_forces = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"}, forces=True)
        self._assert_same(sectors, full, "displaced tetragonal Si")
        self.assertGreater(float(np.abs(full_forces.total).max()), 1.0e-3)
        np.testing.assert_allclose(sector_forces.total, full_forces.total, rtol=0, atol=1.0e-6)

    def test_an_orthorhombic_cell_of_three_steps(self) -> None:
        # Si stretched along y and z by different factors: D2 about the atom at the origin (the mirrors of the
        # face-centred crystal go with the stretch only where they keep the bonds), the face-centring translations
        problem = tetragonal_silicon(12.0, 0.64)
        sides = np.array([10.26, 11.1, 12.0])
        base_sides = np.diag(np.asarray(problem.periodic_cell.lattice_vectors))
        atoms = tuple(
            Atom(atom.symbol, tuple(np.asarray(atom.position) * sides / base_sides)) for atom in problem.atoms
        )
        problem = replace(
            problem,
            atoms=atoms,
            periodic_cell=replace(problem.periodic_cell, lattice_vectors=np.diag(sides)),
            grid=replace(problem.grid, box_lengths=tuple(sides)),
        )
        sectors, details, _ = self._run(problem, {})
        self.assertIn("exact", details["symmetry_detection"])
        full, _, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        self._assert_same(sectors, full, "orthorhombic Si")

    def test_a_cell_beyond_the_palette_keeps_the_full_grid(self) -> None:
        # Order 18 on 20 x 20 x 22 points: one sector stencil needs 306 coefficients (PaletteTests), where the build
        # of the stencils stopped the run.  The ladder of 2026-11-19 (tests/test_sector_palette.py): the commuting
        # group with translations needs 307, the mirrors about the origin fit; PARSEC_SECTOR_PALETTE_LADDER=off: the
        # full grid, and the identity sector the MPI runs take then, as before.
        problem = tetragonal_silicon(12.0, 0.5)
        problem = replace(problem, grid=replace(problem.grid, expansion_order=18))
        lower, lower_details, _ = self._run(problem, {})
        self.assertIn("need 306 distinct coefficients", lower_details["symmetry_detection"])
        self.assertIn("PARSEC_SECTOR_PALETTE_LADDER=auto", lower_details["symmetry_detection"])
        self.assertTrue(lower_details["symmetry_detection"].startswith("exact Cartesian axis-reflection subgroup"))
        off = {"PARSEC_SECTOR_PALETTE_LADDER": "off"}
        kept, details, _ = self._run(problem, off)
        self.assertIn("fell back to full grid", details["symmetry_detection"])
        self.assertIn(f"more than the {PALETTE_CAPACITY} of their palette", details["symmetry_detection"])
        identity, identity_details, _ = self._run(problem, dict(off, PARSEC_CUPY_SINGLE_SECTOR="on"))
        self.assertIn("one sector of the identity", identity_details["symmetry_detection"])
        full, _, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        self._assert_same(lower, full, "beyond the palette, the mirrors about the origin")
        self._assert_same(kept, full, "beyond the palette")
        self._assert_same(identity, full, "beyond the palette, identity sector")


if __name__ == "__main__":
    unittest.main()
