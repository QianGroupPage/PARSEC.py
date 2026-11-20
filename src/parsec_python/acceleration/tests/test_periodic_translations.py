"""The translations by half the cell in the group of a periodic cell (``PARSEC_PERIODIC_TRANSLATIONS``).

A cell that repeats a smaller one twice along some axes is invariant under the translations by half of those axes:
exact symmetries of its grid and of its discretized Hamiltonian, whose real characters are the time-reversal
invariant k-points of the smaller cell.  With the signed permutations about the origin they form the group whose
weighted sectors the orbitals take.  The sectors must hold the levels of the full grid (the same weighted spectrum,
the same SCF), ``off`` must give the point group alone, and a cell that does not repeat keeps it.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.linalg as sl

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    detect_point_group_reduction,
)
from parsec_python.acceleration.Symmetry.orbital_groups import orbital_group_plan
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

_SIDE = 10.26
_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
_BASIS = np.vstack([_FCC, _FCC + 0.25])


def _supercell(repeats, points: int = 10):
    cells = np.array([(i, j, k) for i in range(repeats[0]) for j in range(repeats[1]) for k in range(repeats[2])], float)
    atoms = [Atom("Si", tuple(site)) for site in (cells[:, None, :] + _BASIS[None]).reshape(-1, 3) * _SIDE]
    lattice = np.diag(_SIDE * np.asarray(repeats, dtype=float))
    grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=_SIDE / points, expansion_order=8))
    return grid, atoms


def _reduction(grid, atoms, setting: str):
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": setting}):
        return detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))


def _weighted_levels(grid, reduction, count: int = 40) -> np.ndarray:
    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
    levels = []
    for matrix, weight in zip(decomposition.reduce_operators(build_negative_laplacian(grid)), decomposition.sector_weights):
        values = sl.eigh(matrix.toarray(), eigvals_only=True, subset_by_index=[0, min(count, matrix.shape[0]) - 1])
        levels.extend(np.repeat(values, int(weight)))
    return np.sort(levels)


class HalfCellTranslationTests(unittest.TestCase):
    def test_the_group_of_a_twice_repeated_cube(self) -> None:
        grid, atoms = _supercell((2, 2, 2))
        reduction = _reduction(grid, atoms, "auto")
        self.assertIsInstance(reduction, PointGroupReduction)
        # Td about an atom (24) times the seven half-cell translations and the identity
        self.assertEqual(reduction.full_operations.shape[0], 192)
        self.assertEqual(int(np.count_nonzero(np.any(reduction.full_translations != 0, axis=1))), 168)
        self.assertEqual(reduction.group_order, 32)
        self.assertEqual(int(np.sum(reduction.sector_weights)), 32)
        point_group = _reduction(grid, atoms, "off")
        self.assertEqual(point_group.full_operations.shape[0], 24)
        self.assertIsNone(point_group.translations)
        self.assertLess(reduction.wedge_size * 3, point_group.wedge_size)
        # the weighted levels of the sectors are those of the point group alone (and so of the full grid)
        np.testing.assert_allclose(
            _weighted_levels(grid, reduction)[:60], _weighted_levels(grid, point_group)[:60], rtol=0, atol=1.0e-11
        )

    def test_one_axis_repeated(self) -> None:
        grid, atoms = _supercell((2, 1, 1))
        reduction = _reduction(grid, atoms, "auto")
        self.assertIsNotNone(getattr(reduction, "full_translations", None))
        # the doubling along x and the face centring of the conventional cube, (0, a/2, a/2)
        counts = [int(value) for value in grid.points_per_axis]
        translations = {tuple(int(v) for v in row) for row in reduction.full_translations}
        half = (counts[0] // 2, counts[1] // 2, counts[2] // 2)
        self.assertEqual(
            translations, {(0, 0, 0), (half[0], 0, 0), (0, half[1], half[2]), (half[0], half[1], half[2])}
        )
        full = np.sort(np.linalg.eigvalsh(build_negative_laplacian(grid).toarray()))
        np.testing.assert_allclose(_weighted_levels(grid, reduction)[:50], full[:50], rtol=0, atol=1.0e-11)

    def test_the_face_centred_cube_and_its_partner_states(self) -> None:
        from parsec_python.acceleration.tests.test_orbital_groups import _FCC as fcc, _periodic_model, _weighted_spectrum

        side = 6.0
        atoms = [Atom("C", tuple(side * (np.array(b) + s))) for s in (0.0, 0.25) for b in fcc]
        grid, hamiltonian = _periodic_model((side,) * 3, 12, 0.5, atoms, {"C": 2.0})
        full = np.linalg.eigvalsh(hamiltonian.toarray())[:40]
        # Td about an atom with the three face-centring translations (0, a/2, a/2), ... (half: 96 operations);
        # with 12 points per edge a/4 is three steps, and the operations of Oh outside Td with the translation
        # a/4 (1, 1, 1) close the group: Fd-3m modulo the cube, 192 operations (auto)
        for setting, operations in (("half", 96), ("auto", 192)):
            reduction = _reduction(grid, atoms, setting)
            self.assertEqual(reduction.full_operations.shape[0], operations, setting)
            decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
            np.testing.assert_allclose(
                _weighted_spectrum(decomposition, hamiltonian, 40), full, rtol=0.0, atol=1e-11, err_msg=setting
            )
            reduced = decomposition.reduce_operators(hamiltonian)
            for sector, weight in enumerate(decomposition.sector_weights):
                if weight == 1:
                    continue
                values, vectors = np.linalg.eigh(reduced[sector].toarray())
                state = decomposition.expand_vector(vectors[:, 0], sector)
                partners = reduction.partner_states(state[:, None], np.array([int(weight)]))
                self.assertEqual(partners.shape[1], int(weight))
                residual = hamiltonian @ partners - values[0] * partners
                self.assertLess(float(np.max(np.abs(residual))), 1e-10, f"{setting} sector {sector} weight {weight}")

    def test_the_sectors_take_the_projectors_of_one_atom_per_orbit(self) -> None:
        from pathlib import Path

        from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors, load_pseudopotentials
        from parsec_python.acceleration.Symmetry.axis_reflection import compress_projectors_for_sectors
        from parsec_python.models import SpeciesPotential

        carbon = Path(__file__).resolve().parents[4] / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
        specifications = {"C": SpeciesPotential(carbon, 2, use_spline=True)}
        potentials = load_pseudopotentials(specifications, xc_functional="pbe")
        side = 3.567 / 0.529177210903
        for repeats, points, setting in (((1, 1, 1), 12, "auto"), ((2, 2, 2), 10, "auto"), ((2, 2, 2), 10, "off")):
            cells = np.array(
                [(i, j, k) for i in range(repeats[0]) for j in range(repeats[1]) for k in range(repeats[2])], float
            )
            atoms = [Atom("C", tuple(s)) for s in (cells[:, None, :] + _BASIS[None]).reshape(-1, 3) * side]
            lattice = np.diag(side * np.asarray(repeats, dtype=float))
            grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=side / points, expansion_order=8))
            reduction = _reduction(grid, atoms, setting)
            decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
            operator = build_nonlocal_projectors(grid, atoms, potentials, specifications, lattice)
            compressed = compress_projectors_for_sectors(operator, reduction)
            label = f"{repeats} {setting}"
            # the columns of one atom per orbit of the group of the sectors
            orbits = int(np.unique(reduction.atom_representatives).size)
            self.assertEqual(compressed.projectors.shape[1] * len(atoms), operator.projectors.shape[1] * orbits, label)
            every = decomposition.reduce_nonlocal_operators(operator)
            orbit = decomposition.reduce_nonlocal_operators(compressed)
            matrices = [
                (
                    left.projectors @ np.diag(left.signs) @ left.projectors.T.toarray(),
                    right.projectors @ np.diag(right.signs) @ right.projectors.T.toarray(),
                )
                for left, right in zip(every, orbit)
            ]
            scale = max(float(np.max(np.abs(left))) for left, _ in matrices)
            for sector, (left, right) in enumerate(matrices):
                self.assertLess(float(np.max(np.abs(left - right))), 1.0e-12 * scale, f"{label} sector {sector}")
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "off"}):
            self.assertIs(compress_projectors_for_sectors(operator, reduction), operator)

    def test_a_cell_that_does_not_repeat(self) -> None:
        grid, atoms = _supercell((2, 1, 1))
        atoms[3] = Atom("Si", tuple(np.asarray(atoms[3].position) + np.array((0.0, 0.0, 0.0))))
        atoms[9] = Atom("Si", tuple(np.asarray(atoms[9].position) + np.array((0.05, 0.0, 0.0))))
        reduction = _reduction(grid, atoms, "auto")
        self.assertIsNone(getattr(reduction, "translations", None))

    def test_a_species_of_one_atom_off_the_origin(self) -> None:
        # the operations that need a translation by a fraction of the cell are searched with the atoms of the
        # smallest species after its first one: a species of one atom has none (this indexed past it before)
        from parsec_python.acceleration.Symmetry.axis_reflection import _space_group_operations

        grid = build_periodic_grid(PeriodicCell(8.0 * np.eye(3)), PeriodicGridSettings(spacing=0.5, expansion_order=4))
        matrices, shifts = _space_group_operations(grid, [Atom("Au", (1.0, 1.0, 1.0))], 1.0e-7, 1.0e-10)
        self.assertEqual(len(matrices), 48)
        inversion = next(index for index, matrix in enumerate(matrices) if np.array_equal(matrix, -np.eye(3)))
        # r -> -r + t carries (1, 1, 1) bohr onto itself with t = (2, 2, 2) bohr: four steps of 0.5 bohr
        np.testing.assert_array_equal(shifts[inversion], [4, 4, 4])

    def test_the_plan_of_the_group(self) -> None:
        # Td about an atom of diamond with the translations of a 2 x 2 x 2 supercell, as affine matrices
        grid, atoms = _supercell((2, 2, 2), points=8)
        reduction = _reduction(grid, atoms, "auto")
        elements = []
        for matrix, shift in zip(reduction.full_operations, reduction.full_translations):
            element = np.eye(4, dtype=np.int64)
            element[:3, :3] = matrix
            element[:3, 3] = shift
            elements.append(element)
        plan = orbital_group_plan(elements, modulus=grid.points_per_axis)
        self.assertEqual(len(plan.subgroup), 32)
        self.assertAlmostEqual(plan.cost, 0.009765625)

    def test_the_switch(self) -> None:
        from parsec_python.acceleration.Symmetry.axis_reflection import periodic_translation_setting

        for value, expected in (("", "auto"), ("OFF", "off"), ("Half", "half")):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": value}):
                self.assertEqual(periodic_translation_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": "on"}):
            with self.assertRaises(ValueError):
                periodic_translation_setting()


def _silicon_supercell(repeats, spacing: float = 0.64):
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon

    base = silicon("pbe", spacing=spacing)
    lattice = np.asarray(base.periodic_cell.lattice_vectors, dtype=float)
    atoms = []
    for i in range(repeats[0]):
        for j in range(repeats[1]):
            for k in range(repeats[2]):
                shift = np.array([i, j, k]) @ lattice
                atoms += [Atom(atom.symbol, tuple(np.asarray(atom.position) + shift)) for atom in base.atoms]
    count = int(np.prod(repeats))
    return replace(
        base,
        atoms=atoms,
        periodic_cell=replace(base.periodic_cell, lattice_vectors=np.diag(np.diag(lattice) * np.asarray(repeats))),
        scf=replace(base.scf, number_of_states=base.scf.number_of_states * count),
    )


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class HalfCellTranslationSCFTests(unittest.TestCase):
    def _solve(self, problem, environment):
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            result = driver.run_scf(system)
        return result, dict(result.backend.details)

    def test_silicon_supercells(self) -> None:
        for repeats in ((2, 1, 1), (2, 2, 2)):
            problem = _silicon_supercell(repeats)
            label = "x".join(str(value) for value in repeats)
            sectors, details = self._solve(problem, {"PARSEC_PERIODIC_TRANSLATIONS": "auto"})
            self.assertIn("with a translation", details["symmetry_detection"], label)
            full, full_details = self._solve(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
            self.assertTrue(sectors.converged and full.converged, label)
            for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
                # the total to 1e-8 Ry; its parts to 1e-7 (the SCF to 1e-9 converges the total, not each part:
                # another starting density moves them by 1e-8); the sum of the levels 1e-8 Ry per cube
                delta = (
                    1.0e-8 if field == "total"
                    else 1.0e-8 * int(np.prod(repeats)) if field == "eigenvalue"
                    else 1.0e-7
                )
                self.assertAlmostEqual(
                    getattr(sectors.energies, field), getattr(full.energies, field), delta=delta, msg=f"{label} {field}"
                )
            occupied = int(np.count_nonzero(np.asarray(full.occupations) > 1.0e-6))
            np.testing.assert_allclose(
                np.sort(sectors.eigenvalues)[:occupied], np.sort(full.eigenvalues)[:occupied], rtol=0, atol=1.0e-7,
                err_msg=label,
            )

    def test_the_non_symmorphic_operations_of_a_silicon_cube(self) -> None:
        # 16 points per edge: a/4 is four grid steps, and the operations of Oh outside Td with the translation
        # a/4 (1, 1, 1) are symmetries of the grid: Fd-3m modulo the cube, 192 operations
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        problem = silicon("pbe", spacing=10.26 / 16)
        sectors, details = self._solve(problem, {"PARSEC_PERIODIC_TRANSLATIONS": "auto"})
        self.assertIn("exact group of 192 operations", details["symmetry_detection"])
        half, half_details = self._solve(problem, {"PARSEC_PERIODIC_TRANSLATIONS": "half"})
        self.assertIn("exact group of 96 operations", half_details["symmetry_detection"])
        full, _ = self._solve(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        for result, label in ((sectors, "Fd-3m"), (half, "half-cell translations")):
            self.assertTrue(result.converged, label)
            for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
                self.assertAlmostEqual(
                    getattr(result.energies, field), getattr(full.energies, field), delta=1.0e-8, msg=f"{label} {field}"
                )


if __name__ == "__main__":
    unittest.main()
