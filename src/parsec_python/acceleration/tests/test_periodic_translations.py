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
        reduction = _reduction(grid, atoms, "auto")
        # Td about an atom with the three face-centring translations (0, a/2, a/2), ...
        self.assertEqual(reduction.full_operations.shape[0], 96)
        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
        count = 40
        full = np.linalg.eigvalsh(hamiltonian.toarray())[:count]
        np.testing.assert_allclose(_weighted_spectrum(decomposition, hamiltonian, count), full, rtol=0.0, atol=1e-11)
        reduced = decomposition.reduce_operators(hamiltonian)
        for sector, weight in enumerate(decomposition.sector_weights):
            if weight == 1:
                continue
            values, vectors = np.linalg.eigh(reduced[sector].toarray())
            state = decomposition.expand_vector(vectors[:, 0], sector)
            partners = reduction.partner_states(state[:, None], np.array([int(weight)]))
            self.assertEqual(partners.shape[1], int(weight))
            residual = hamiltonian @ partners - values[0] * partners
            self.assertLess(float(np.max(np.abs(residual))), 1e-10, f"sector {sector} weight {weight}")

    def test_a_cell_that_does_not_repeat(self) -> None:
        grid, atoms = _supercell((2, 1, 1))
        atoms[3] = Atom("Si", tuple(np.asarray(atoms[3].position) + np.array((0.0, 0.0, 0.0))))
        atoms[9] = Atom("Si", tuple(np.asarray(atoms[9].position) + np.array((0.05, 0.0, 0.0))))
        reduction = _reduction(grid, atoms, "auto")
        self.assertIsNone(getattr(reduction, "translations", None))

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

        for value, expected in (("", "auto"), ("OFF", "off")):
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
            self.assertIn("translation by half the cell", details["symmetry_detection"], label)
            full, full_details = self._solve(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
            self.assertTrue(sectors.converged and full.converged, label)
            for field in ("total", "exchange_correlation", "hartree", "eigenvalue"):
                # the sum of the levels: 1e-8 Ry for the states of each conventional cube (Diag_Tolerance 1e-10)
                delta = 1.0e-8 * (int(np.prod(repeats)) if field == "eigenvalue" else 1)
                self.assertAlmostEqual(
                    getattr(sectors.energies, field), getattr(full.energies, field), delta=delta, msg=f"{label} {field}"
                )
            occupied = int(np.count_nonzero(np.asarray(full.occupations) > 1.0e-6))
            np.testing.assert_allclose(
                np.sort(sectors.eigenvalues)[:occupied], np.sort(full.eigenvalues)[:occupied], rtol=0, atol=1.0e-7,
                err_msg=label,
            )


if __name__ == "__main__":
    unittest.main()
