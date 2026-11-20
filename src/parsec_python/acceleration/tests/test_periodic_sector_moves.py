"""Moving the atoms of a bulk cell solved in the sectors of its point group (``driver.update_geometry``).

A cell at the Gamma point solved in the one-dimensional sectors of a subgroup ``H`` of its group ``G``
(:class:`parsec_python.acceleration.Symmetry.axis_reflection.PointGroupReduction`) moves its atoms where they keep
``G``: the moved structure must give what a fresh preparation of it gives, in the sectors and on the full grid, and a
relaxation with forces and positions averaged over ``G`` must follow the trajectory of the full grid.  A move that
leaves the group is refused.  The orbital operators of an MPI sector rank (no ionic potential) take the projectors of
the moved atoms alone, and one sector of the identity (``PARSEC_CUPY_SINGLE_SECTOR=on``) takes them too.
"""

from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Forces.periodic_device import with_periodic_operations
from parsec_python.acceleration.Forces.sector_symmetry import (
    MOVE_TOLERANCE,
    check_sector_atoms,
    sector_force_operations,
    sector_group_operations,
    sector_reduction,
)
from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon
from parsec_python.models import Atom


def _moved(problem, shifts: dict) -> tuple:
    """The atoms of ``problem`` with ``shifts[index]`` (bohr) added."""

    return tuple(
        Atom(atom.symbol, tuple(np.asarray(atom.position, dtype=np.float64) + np.asarray(shifts.get(index, 0.0))))
        for index, atom in enumerate(problem.atoms)
    )


def displaced_supercell(distance: float, repeats: tuple[int, int, int] = (2, 2, 2)):
    """:func:`displaced_silicon` repeated ``repeats`` times: ``G`` of Td about the origin with the translations by
    half the cell (192 operations for 2 x 2 x 2), on the same grid spacing, with the states it needs."""

    base = displaced_silicon(distance)
    lattice = np.asarray(base.periodic_cell.lattice_vectors, dtype=np.float64)
    shifts = [i * lattice[0] + j * lattice[1] + k * lattice[2] for i in range(repeats[0]) for j in range(repeats[1])
              for k in range(repeats[2])]
    atoms = tuple(
        Atom(atom.symbol, tuple(np.asarray(atom.position, dtype=np.float64) + shift))
        for shift in shifts
        for atom in base.atoms
    )
    count = len(shifts)
    states = base.scf.number_of_states
    return replace(
        base,
        atoms=atoms,
        periodic_cell=replace(base.periodic_cell, lattice_vectors=lattice * np.asarray(repeats, dtype=np.float64)[:, None]),
        scf=replace(base.scf, number_of_states=None if states is None else int(states) * count),
    )


def _point_group_reduction(problem):
    """The reduction the accelerated driver detects for ``problem`` (host only)."""

    from parsec_python.Grid import build_periodic_grid
    from parsec_python.acceleration.Symmetry import detect_point_group_reduction, load_or_detect_reflection_reduction

    grid = build_periodic_grid(problem.periodic_cell, problem.grid)
    abelian, _ = load_or_detect_reflection_reduction(grid, problem.atoms, cache_directory=None)
    return detect_point_group_reduction(grid, problem.atoms, abelian), grid


class OrbitalOperatorMoveTests(unittest.TestCase):
    """The projectors-only update of :func:`parsec_python.SCF.pbc.update_periodic_geometry`."""

    def test_an_orbital_only_system_takes_the_projectors_of_the_folded_moved_atoms(self) -> None:
        from parsec_python.SCF.pbc import fold_atoms_into_cell, prepare_periodic_single_point, update_periodic_geometry
        from parsec_python.V_ion import build_nonlocal_projectors

        problem = displaced_silicon()
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            operators_only = driver._prepare_periodic_orbital_operators(problem)
        self.assertIsNone(operators_only.ionic_potential)
        # one atom moved a lattice vector out of the cell: the coordinates stay, the projectors take it folded
        atoms = _moved(problem, {2: (0.05, -0.03, 0.02), 5: np.array((-0.02, 0.04, 0.01)) + lattice[0]})
        moved = update_periodic_geometry(operators_only, atoms)
        self.assertEqual(tuple(moved.atoms), atoms)
        self.assertEqual(tuple(moved.input.atoms), atoms)
        for name in ("ionic_potential", "initial_density", "core_density", "gradient", "initial_spin_densities"):
            self.assertIsNone(getattr(moved, name), name)
        self.assertTrue(np.isnan(moved.ion_ion_energy))
        self.assertIs(moved.grid, operators_only.grid)
        self.assertIs(moved.negative_laplacian, operators_only.negative_laplacian)
        folded = fold_atoms_into_cell(atoms, lattice)
        self.assertFalse(np.allclose(folded[5].position, atoms[5].position))
        expected = build_nonlocal_projectors(
            operators_only.grid,
            folded,
            operators_only.pseudopotentials,
            problem.pseudopotentials,
            lattice,
            periodic_dimensions=3,
        )
        self.assertEqual(tuple(moved.nonlocal_operator.labels), tuple(expected.labels))
        np.testing.assert_array_equal(moved.nonlocal_operator.signs, expected.signs)
        self.assertEqual((moved.nonlocal_operator.projectors != expected.projectors).nnz, 0)
        # and those of the full update of a complete preparation, bit for bit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            complete = update_periodic_geometry(prepare_periodic_single_point(problem), atoms)
        self.assertEqual((moved.nonlocal_operator.projectors != complete.nonlocal_operator.projectors).nnz, 0)
        np.testing.assert_array_equal(moved.nonlocal_operator.signs, complete.nonlocal_operator.signs)


class SectorAtomCheckTests(unittest.TestCase):
    """:func:`check_sector_atoms` on the displaced Si8 cell (Td about the origin, ``G`` of 24 operations)."""

    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.problem = displaced_silicon(0.04)
            cls.reduction, cls.grid = _point_group_reduction(cls.problem)
        cls.lattice = np.asarray(cls.problem.periodic_cell.lattice_vectors, dtype=np.float64)

    def test_the_reduction_is_a_point_group_with_orbits(self) -> None:
        self.assertEqual(int(np.asarray(self.reduction.full_operations).shape[0]), 24)
        self.assertIsNotNone(self.reduction.atom_representatives)
        self.assertGreater(int(np.max(self.reduction.atom_orbit_sizes)), 1)

    def test_moves_on_the_group_pass(self) -> None:
        check_sector_atoms(self.reduction, self.problem.atoms, self.grid, self.lattice)
        # further along the bonds: the same group
        check_sector_atoms(self.reduction, displaced_silicon(0.07).atoms, self.grid, self.lattice)
        # an atom named by another image: the same structure
        check_sector_atoms(self.reduction, _moved(self.problem, {3: -self.lattice[1]}), self.grid, self.lattice)
        # round-off below the tolerance
        check_sector_atoms(
            self.reduction, _moved(self.problem, {4: (0.3 * MOVE_TOLERANCE, 0.0, 0.0)}), self.grid, self.lattice
        )

    def test_the_operations_of_the_forces_hold_where_the_check_passes(self) -> None:
        # an atom moved off the group within the tolerance of the check: the operations of its forces (the image
        # table of G) are found, the same as for the atoms on the group
        exact = sector_group_operations(self.reduction, self.problem.atoms, self.grid, self.lattice)
        for fraction in (0.3, 0.9):
            moved = _moved(self.problem, {4: (fraction * MOVE_TOLERANCE, 0.0, 0.0)})
            check_sector_atoms(self.reduction, moved, self.grid, self.lattice)
            operations = sector_group_operations(self.reduction, moved, self.grid, self.lattice)
            self.assertEqual(len(operations), 24)
            for found, wanted in zip(operations, exact):
                np.testing.assert_array_equal(found.image, wanted.image)

    def test_a_move_off_the_group_is_refused(self) -> None:
        for shift in ((0.01, 0.0, 0.0), (3.0 * MOVE_TOLERANCE, 0.0, 0.0)):
            with self.subTest(shift=shift), self.assertRaisesRegex(ValueError, "operation"):
                check_sector_atoms(self.reduction, _moved(self.problem, {4: shift}), self.grid, self.lattice)

    def test_atoms_that_change_their_orbits_are_refused(self) -> None:
        sizes = np.asarray(self.reduction.atom_orbit_sizes)
        single = int(np.flatnonzero(sizes == 1)[0])
        shared = int(np.flatnonzero(sizes > 1)[0])
        atoms = list(self.problem.atoms)
        atoms[single], atoms[shared] = (
            Atom(atoms[single].symbol, atoms[shared].position),
            Atom(atoms[shared].symbol, atoms[single].position),
        )
        with self.assertRaisesRegex(ValueError, "orbits"):
            check_sector_atoms(self.reduction, atoms, self.grid, self.lattice)

    def test_the_translations_of_the_group_are_checked(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = displaced_supercell(0.04)
            reduction, grid = _point_group_reduction(problem)
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
        self.assertEqual(int(np.asarray(reduction.full_operations).shape[0]), 192)
        self.assertTrue(np.any(np.asarray(reduction.translations) != 0))
        check_sector_atoms(reduction, displaced_supercell(0.07).atoms, grid, lattice)
        # the first small cube alone further along its bonds: Td about the origin stays, the translations go
        count = len(self.problem.atoms)
        atoms = displaced_supercell(0.07).atoms[:count] + tuple(problem.atoms[count:])
        with self.assertRaisesRegex(ValueError, "group of the symmetry sectors"):
            check_sector_atoms(reduction, atoms, grid, lattice)

    def test_other_lattice_vectors_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "lattice"):
            check_sector_atoms(self.reduction, self.problem.atoms, self.grid, 1.1 * self.lattice)

    def test_the_operations_of_a_relaxation_follow_the_structure(self) -> None:
        def system_of(atoms):
            problem = replace(self.problem, atoms=tuple(atoms))
            reference = SimpleNamespace(atoms=problem.atoms, input=problem, grid=self.grid)
            sectors = SimpleNamespace(decomposition=SimpleNamespace(reduction=self.reduction))
            return SimpleNamespace(
                reference=reference,
                backend=SimpleNamespace(symmetry_eigensolver=sectors),
                atoms=problem.atoms,
                input=problem,
                grid=self.grid,
            )

        backend = with_periodic_operations(driver.ionic_backend())
        first = system_of(self.problem.atoms)
        operations = backend.symmetry_operations(first)
        self.assertEqual(len(operations), 24)
        self.assertIs(backend.symmetry_operations(first), operations)  # once per structure
        moved = system_of(displaced_silicon(0.07).atoms)
        other = backend.symmetry_operations(moved)
        self.assertIsNot(other, operations)
        for left, right in zip(other, sector_force_operations(moved)):
            np.testing.assert_array_equal(left.matrix, right.matrix)
            np.testing.assert_array_equal(left.image, right.image)


def _prepare(problem, environment: dict):
    with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = driver.prepare_single_point(problem, backend="cupy")
    system.materialize_final_wavefunctions = False
    return system


def _scf(system, warm_start=None):
    result = driver.run_scf(system, warm_start=warm_start)
    if not result.converged:
        raise AssertionError("the SCF loop did not converge")
    return result


SECTORS = {"PARSEC_PERIODIC_SYMMETRY": "auto"}
FULL_GRID = {"PARSEC_PERIODIC_SYMMETRY": "off"}
IDENTITY_SECTOR = {"PARSEC_PERIODIC_SYMMETRY": "off", "PARSEC_CUPY_SINGLE_SECTOR": "on"}


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class PeriodicSectorMoveTests(unittest.TestCase):
    """Moved atoms of a cell in sectors against fresh preparations of the moved structure."""

    def assert_same_structure(self, moved, fresh, label, *, energy, force) -> None:
        (moved_system, moved_result), (fresh_system, fresh_result) = moved, fresh
        self.assertAlmostEqual(moved_result.energies.total, fresh_result.energies.total, delta=energy, msg=label)
        moved_forces = driver.compute_forces(moved_system, moved_result, symmetrize=False)
        fresh_forces = driver.compute_forces(fresh_system, fresh_result, symmetrize=False)
        np.testing.assert_allclose(
            moved_forces.nonlocal_pseudopotential, fresh_forces.nonlocal_pseudopotential, rtol=0, atol=force,
            err_msg=f"{label} nonlocal",
        )
        np.testing.assert_allclose(moved_forces.total, fresh_forces.total, rtol=0, atol=force, err_msg=f"{label} total")
        return moved_forces

    def test_moved_atoms_in_the_sectors_give_what_a_fresh_preparation_gives(self) -> None:
        first = displaced_silicon(0.04)
        target = displaced_silicon(0.07)
        system = _prepare(first, SECTORS)
        reduction = sector_reduction(system)
        self.assertIsNotNone(reduction, "the cell was not solved in the sectors of a point group")
        self.assertTrue(np.any(np.asarray(reduction.sector_weights) > 1))
        result = _scf(system)
        driver.compute_forces(system, result, symmetrize=False)
        # a move off the group is refused before anything changes
        with self.assertRaisesRegex(ValueError, "group of the symmetry sectors"):
            driver.update_geometry(system, _moved(first, {4: (0.01, 0.0, 0.0)}))
        moved_system = driver.update_geometry(system, target.atoms)
        self.assertIs(sector_reduction(moved_system), reduction)
        moved_result = _scf(moved_system, driver.warm_start_for_moved_atoms(system, result, moved_system))
        fresh_system = _prepare(target, SECTORS)
        fresh_result = _scf(fresh_system)
        forces = self.assert_same_structure(
            (moved_system, moved_result), (fresh_system, fresh_result), "sectors", energy=1.0e-8, force=1.0e-7
        )
        self.assertGreater(float(np.abs(forces.total).max()), 1.0e-3)
        self.assertEqual(len(sector_force_operations(moved_system)), 24)
        full_system = _prepare(target, FULL_GRID)
        self.assertIsNone(sector_reduction(full_system))
        full_result = _scf(full_system)
        self.assert_same_structure(
            (moved_system, moved_result), (full_system, full_result), "full grid", energy=1.0e-7, force=2.0e-6
        )

    def test_a_cell_with_half_cell_translations_moves_in_its_sectors(self) -> None:
        first = displaced_supercell(0.04)
        target = displaced_supercell(0.07)
        system = _prepare(first, SECTORS)
        reduction = sector_reduction(system)
        self.assertIsNotNone(reduction, "the cell was not solved in the sectors of a point group")
        self.assertEqual(int(np.asarray(reduction.full_operations).shape[0]), 192)
        self.assertTrue(np.any(np.asarray(reduction.translations) != 0))
        result = _scf(system)
        moved_system = driver.update_geometry(system, target.atoms)
        moved_result = _scf(moved_system, driver.warm_start_for_moved_atoms(system, result, moved_system))
        fresh_system = _prepare(target, SECTORS)
        fresh_result = _scf(fresh_system)
        forces = self.assert_same_structure(
            (moved_system, moved_result), (fresh_system, fresh_result), "2x2x2 sectors", energy=1.0e-8, force=1.0e-7
        )
        self.assertGreater(float(np.abs(forces.total).max()), 1.0e-3)

    def test_one_sector_of_the_identity_takes_the_projectors_of_moved_atoms(self) -> None:
        from parsec_python.SCF.pbc import fold_atoms_into_cell

        first = displaced_silicon(0.04)
        lattice = np.asarray(first.periodic_cell.lattice_vectors, dtype=np.float64)
        shifts = {2: (0.05, -0.03, 0.02), 5: (-0.02, 0.04, 0.01)}
        system = _prepare(first, IDENTITY_SECTOR)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        self.assertIsNotNone(sectors, "the cell was not solved as one sector of the identity")
        result = _scf(system)
        # the second atom named by its image a lattice vector away: the projectors fold it
        moved_atoms = _moved(first, {2: shifts[2], 5: np.asarray(shifts[5]) + lattice[0]})
        moved_system = driver.update_geometry(system, moved_atoms)
        moved_result = _scf(moved_system, driver.warm_start_for_moved_atoms(system, result, moved_system))
        # (a fresh preparation takes the atoms inside the cell)
        fresh_atoms = fold_atoms_into_cell(_moved(first, shifts), lattice)
        fresh_system = _prepare(replace(first, atoms=fresh_atoms), IDENTITY_SECTOR)
        fresh_result = _scf(fresh_system)
        self.assert_same_structure(
            (moved_system, moved_result), (fresh_system, fresh_result), "identity sector", energy=1.0e-8, force=1.0e-7
        )

    def test_a_relaxation_in_the_sectors_follows_the_full_grid(self) -> None:
        from parsec_python.Relaxation import run_relaxation
        from parsec_python.models import RelaxationSettings

        problem = displaced_silicon(0.04)
        settings = RelaxationSettings(method="bfgs", max_movements=3, minimum_force=1.0e-5)
        outcomes = {}
        for label, environment in (("sectors", SECTORS), ("full grid", FULL_GRID)):
            system = _prepare(problem, environment)
            in_sectors = sector_reduction(system) is not None
            self.assertEqual(in_sectors, label == "sectors")
            # a cell on the full grid has no operations to average over (Forces/compute.py)
            backend = with_periodic_operations(driver.ionic_backend()) if in_sectors else driver.ionic_backend()
            with mock.patch.dict(os.environ, environment):
                outcomes[label] = run_relaxation(system, settings, backend, symmetrize=in_sectors)
        sectors, full = outcomes["sectors"], outcomes["full grid"]
        self.assertEqual(len(sectors.records), 4)
        self.assertEqual(len(full.records), len(sectors.records))
        for left, right in zip(sectors.records, full.records):
            self.assertTrue(left.converged and right.converged)
            np.testing.assert_allclose(left.positions, right.positions, rtol=0, atol=1.0e-6, err_msg=f"step {left.step}")
            self.assertAlmostEqual(left.energy, right.energy, delta=1.0e-6, msg=f"step {left.step}")
        # the atoms moved, and on the group
        self.assertGreater(float(np.abs(sectors.records[-1].positions - sectors.records[0].positions).max()), 1.0e-3)
        self.assertLess(sectors.records[-1].energy, sectors.records[0].energy)


if __name__ == "__main__":
    unittest.main()
