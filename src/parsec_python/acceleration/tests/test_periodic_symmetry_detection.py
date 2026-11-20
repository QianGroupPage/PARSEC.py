"""Symmetry of periodic cells that the detection used to miss (stream D of the 2026-11-18 survey).

Four inputs lost operations without saying more than one line of the report:

* a cell whose abelian group about the origin is the identity alone (a supercell of a low-symmetry structure) did not
  look for its group at all, and lost its translations by half the cell (``PARSEC_PERIODIC_TRIVIAL_ORIGIN``);
* a supercell whose symmetric site is off the origin kept the operations about the origin and the half-cell points
  (diamond 2 x 2 x 2 moved by (1, 2, 3) grid steps: 16 of 192, ``PARSEC_PERIODIC_SYMMETRY_CENTRE``);
* the orbitals that keep an abelian group (``occup.in``, ``Fix_Number_Eigenvalues``, ``PARSEC_ORBITAL_GROUP=abelian``)
  took the commuting signed permutations about the origin without the translations of the cell
  (``PARSEC_PERIODIC_ABELIAN_TRANSLATIONS``);
* the atom tolerance was fixed at 1e-7 bohr, and a rounded input lost its group with the report saying "identity
  only" (``PARSEC_SYMMETRY_ATOM_TOLERANCE`` and a note of the operations lost).

The sectors of the new groups are those of the same Hamiltonian: the model tests compare their weighted spectra with
the full grid on the CPU, the SCF tests (A100) the energies and forces with ``PARSEC_PERIODIC_SYMMETRY=off``.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    detect_point_group_reduction,
    lost_operations_note,
    periodic_abelian_reduction,
    periodic_abelian_translation_setting,
    periodic_centre_setting,
    periodic_trivial_origin_setting,
    symmetry_atom_tolerance,
)
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.tests.test_orbital_groups import _periodic_model, _weighted_spectrum
from parsec_python.models import Atom

_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])
_DIAMOND = np.vstack([_FCC, _FCC + 0.25])


def _diamond(repeats, side: float = 6.0, offset=(0.0, 0.0, 0.0), symbol: str = "C"):
    """Diamond cubes of ``side`` bohr repeated ``repeats`` times, moved by ``offset`` bohr (folded into the cell)."""

    cells = np.array([(i, j, k) for i in range(repeats[0]) for j in range(repeats[1]) for k in range(repeats[2])], float)
    sides = side * np.asarray(repeats, dtype=float)
    positions = np.mod((cells[:, None, :] + _DIAMOND[None]).reshape(-1, 3) * side + np.asarray(offset), sides)
    return [Atom(symbol, tuple(position)) for position in positions], sides


def _full_spectrum(hamiltonian, count: int) -> np.ndarray:
    return np.linalg.eigvalsh(hamiltonian.toarray())[:count]


def _detect(grid, atoms, environment=None):
    with mock.patch.dict(os.environ, environment or {}):
        return detect_point_group_reduction(grid, atoms, SignedPermutationReduction.detect(grid, atoms))


class DetectionSwitchTests(unittest.TestCase):
    def test_the_switches(self) -> None:
        for function, name in (
            (periodic_trivial_origin_setting, "PARSEC_PERIODIC_TRIVIAL_ORIGIN"),
            (periodic_centre_setting, "PARSEC_PERIODIC_SYMMETRY_CENTRE"),
            (periodic_abelian_translation_setting, "PARSEC_PERIODIC_ABELIAN_TRANSLATIONS"),
        ):
            for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
                with mock.patch.dict(os.environ, {name: value}):
                    self.assertEqual(function(), expected, name)
            with mock.patch.dict(os.environ, {name: "on"}), self.assertRaises(ValueError):
                function()
        for value, expected in (("", 1.0e-7), ("1e-4", 1.0e-4), (" 0.001 ", 1.0e-3)):
            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": value}):
                self.assertEqual(symmetry_atom_tolerance(), expected)
        for value in ("0", "-1e-5", "0.2", "nan", "tight"):
            with mock.patch.dict(os.environ, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": value}), self.assertRaises(ValueError):
                symmetry_atom_tolerance()


class LowSymmetrySupercellTests(unittest.TestCase):
    """(i) A cell of no symmetry about the origin, repeated: its translations by half the cell."""

    def _cell(self, repeats):
        rng = np.random.default_rng(11)
        basis = rng.uniform(0.0, 6.0, size=(4, 3))
        atoms = [
            Atom("C", tuple(position + 6.0 * np.array((i, j, k))))
            for i in range(repeats[0]) for j in range(repeats[1]) for k in range(repeats[2])
            for position in basis
        ]
        return _periodic_model(6.0 * np.asarray(repeats, dtype=float), 8 * repeats[0], 0.0, atoms, {"C": 2.0}), atoms

    def test_a_cell_repeated_twice(self) -> None:
        (grid, hamiltonian), atoms = self._cell((2, 1, 1))
        abelian = SignedPermutationReduction.detect(grid, atoms)
        self.assertEqual(abelian.group_order, 1)
        reduction = detect_point_group_reduction(grid, atoms, abelian)
        self.assertIsInstance(reduction, PointGroupReduction)
        self.assertEqual(reduction.full_operations.shape[0], 2)
        np.testing.assert_array_equal(reduction.full_translations, [[0, 0, 0], [8, 0, 0]])
        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
        np.testing.assert_allclose(
            _weighted_spectrum(decomposition, hamiltonian, 30), _full_spectrum(hamiltonian, 30), rtol=0, atol=1.0e-11
        )


class OffCentreSupercellTests(unittest.TestCase):
    """(ii) A supercell whose symmetric site is off the origin: the operations about that site."""

    def test_diamond_moved_off_the_origin(self) -> None:
        # 8 points per cube edge of 6 bohr (steps of 0.75): (1, 2, 3) steps put no atom on the origin or a half-cell
        # point; the cell has pure translations besides the half-cell ones (the face centring along x)
        step = 0.75
        about_origin, sides = _diamond((2, 1, 1))
        moved, _ = _diamond((2, 1, 1), offset=step * np.array((1.0, 2.0, 3.0)))
        grid, _ = _periodic_model(sides, 16, 0.0, about_origin, {"C": 2.0})
        reference = _detect(grid, about_origin)
        # D2d about an atom (the operations of Td that keep the x axis) with four half-cell translations
        self.assertEqual(reference.full_operations.shape[0], 32)
        found = _detect(grid, moved)
        self.assertIsInstance(found, PointGroupReduction)
        self.assertEqual(found.full_operations.shape[0], 32)
        self.assertEqual(found.group_order, reference.group_order)
        before = _detect(grid, moved, {"PARSEC_PERIODIC_SYMMETRY_CENTRE": "off"})
        self.assertLess(getattr(before, "full_operations", np.zeros((before.group_order,))).shape[0], 32)
        _, hamiltonian = _periodic_model(sides, 16, 0.0, moved, {"C": 2.0})
        decomposition = ReflectionRepresentationDecomposition.build(grid, found)
        np.testing.assert_allclose(
            _weighted_spectrum(decomposition, hamiltonian, 40), _full_spectrum(hamiltonian, 40), rtol=0, atol=1.0e-11
        )
        # the partner states of the weighted sectors are eigenstates of the full Hamiltonian
        reduced = decomposition.reduce_operators(hamiltonian)
        for sector, weight in enumerate(decomposition.sector_weights):
            if weight == 1:
                continue
            values, vectors = np.linalg.eigh(reduced[sector].toarray())
            state = decomposition.expand_vector(vectors[:, 0], sector)
            partners = found.partner_states(state[:, None], np.array([int(weight)]))
            residual = hamiltonian @ partners - values[0] * partners
            self.assertLess(float(np.max(np.abs(residual))), 1.0e-10, f"sector {sector}")

    def test_the_translation_switch_keeps_its_groups(self) -> None:
        # PARSEC_PERIODIC_TRANSLATIONS=off (the point group about the origin alone) and half (the half-cell
        # translations alone) take no operation about a centre off the origin: (I - R) x0 is a translation of its
        # own.  Before the fix the centre search ran under both, and the moved cell got 8 operations of the cell
        # group under off (1 before the search existed) and 32 under half (4).
        from parsec_python.acceleration.SCF.kpoint_sectors import CellGroup

        moved, sides = _diamond((2, 1, 1), offset=0.75 * np.array((1.0, 2.0, 3.0)))
        grid, _ = _periodic_model(sides, 16, 0.0, moved, {"C": 2.0})
        for value, order in (("off", 1), ("half", 4)):
            found = {}
            for centre in ("auto", "off"):
                environment = {"PARSEC_PERIODIC_TRANSLATIONS": value, "PARSEC_PERIODIC_SYMMETRY_CENTRE": centre}
                with mock.patch.dict(os.environ, environment):
                    found[centre] = (CellGroup(grid, moved), _detect(grid, moved))
            (cell, detected), (cell_off, detected_off) = found["auto"], found["off"]
            with self.subTest(translations=value):
                self.assertEqual(cell.order, order)
                np.testing.assert_array_equal(np.asarray(cell.matrices), np.asarray(cell_off.matrices))
                np.testing.assert_array_equal(np.asarray(cell.shifts), np.asarray(cell_off.shifts))
                self.assertIs(type(detected), type(detected_off))
                self.assertEqual(detected.group_order, detected_off.group_order)
                if isinstance(detected, PointGroupReduction):
                    self.assertEqual(detected.full_operations.shape[0], order)
                    np.testing.assert_array_equal(detected.full_translations, detected_off.full_translations)
                np.testing.assert_array_equal(detected.full_to_wedge, detected_off.full_to_wedge)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": "auto"}):
            self.assertEqual(CellGroup(grid, moved).order, 32)

    def test_a_centred_supercell_keeps_its_group(self) -> None:
        # bit for bit: about the origin the search finds no better centre
        atoms, sides = _diamond((2, 2, 1))
        grid, _ = _periodic_model(sides, 16, 0.0, atoms, {"C": 2.0})
        auto = _detect(grid, atoms)
        off = _detect(grid, atoms, {"PARSEC_PERIODIC_SYMMETRY_CENTRE": "off"})
        np.testing.assert_array_equal(auto.full_operations, off.full_operations)
        np.testing.assert_array_equal(auto.full_translations, off.full_translations)
        np.testing.assert_array_equal(auto.full_to_wedge, off.full_to_wedge)


class AbelianTranslationTests(unittest.TestCase):
    """(iv) The abelian group of the orbitals with the translations of the cell."""

    def test_diamond_supercell(self) -> None:
        atoms, sides = _diamond((2, 1, 1))
        grid, hamiltonian = _periodic_model(sides, 16, 0.0, atoms, {"C": 2.0})
        abelian = SignedPermutationReduction.detect(grid, atoms)
        self.assertEqual(abelian.group_order, 4)
        reduction = periodic_abelian_reduction(grid, atoms, abelian)
        # D2 about the atom with the four half-cell translations
        self.assertEqual(reduction.group_order, 16)
        self.assertIsNotNone(reduction.translations)
        self.assertEqual(int(np.count_nonzero(np.any(reduction.translations != 0, axis=1))), 12)
        decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
        self.assertTrue(np.all(np.asarray(decomposition.sector_weights) == 1))
        np.testing.assert_allclose(
            _weighted_spectrum(decomposition, hamiltonian, 40), _full_spectrum(hamiltonian, 40), rtol=0, atol=1.0e-11
        )
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_ABELIAN_TRANSLATIONS": "off"}):
            self.assertIs(periodic_abelian_reduction(grid, atoms, abelian), abelian)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": "off"}):
            self.assertIs(periodic_abelian_reduction(grid, atoms, abelian), abelian)

    def test_a_cluster_keeps_its_group(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings

        atoms = [Atom("C", (0.0, 0.0, 0.0)), Atom("H", (1.2, 1.2, 1.2)), Atom("H", (-1.2, -1.2, 1.2)),
                 Atom("H", (1.2, -1.2, -1.2)), Atom("H", (-1.2, 1.2, -1.2))]
        grid = build_cluster_grid(GridSettings(radius=5.0, spacing=0.6, expansion_order=4))
        abelian = SignedPermutationReduction.detect(grid, atoms)
        self.assertIs(periodic_abelian_reduction(grid, atoms, abelian), abelian)


class AtomToleranceTests(unittest.TestCase):
    """(v) The atom tolerance: a switch, and a note of the operations a rounded input loses."""

    def test_a_noisy_cell(self) -> None:
        atoms, sides = _diamond((1, 1, 1), side=6.0)
        grid, _ = _periodic_model(sides, 8, 0.0, atoms, {"C": 2.0})
        exact = _detect(grid, atoms)
        self.assertIsNone(lost_operations_note(grid, atoms, exact))
        rng = np.random.default_rng(7)
        noisy = [Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.normal(0.0, 1.0e-5, 3))) for atom in atoms]
        strict = _detect(grid, noisy)
        self.assertEqual(strict.group_order, 1)
        note = lost_operations_note(grid, noisy, strict)
        self.assertIsNotNone(note)
        self.assertIn("PARSEC_SYMMETRY_ATOM_TOLERANCE", note)
        self.assertIn("47 rotations or mirrors", note)
        tolerance = 1.0e-4
        loose = detect_point_group_reduction(
            grid, noisy, SignedPermutationReduction.detect(grid, noisy, atom_tolerance=tolerance),
            atom_tolerance=tolerance,
        )
        self.assertEqual(loose.full_operations.shape[0], exact.full_operations.shape[0])
        self.assertIsNone(lost_operations_note(grid, noisy, loose, atom_tolerance=tolerance))


# ---------------------------------------------------------------------------------------------------------------
# The SCF on the A100: the sectors of the new groups against the full grid (PARSEC_PERIODIC_SYMMETRY=off).


def _moved(problem, offset_steps):
    """``problem`` with every atom moved by ``offset_steps`` grid steps (folded into the cell)."""

    from parsec_python.Grid.pbc import build_periodic_grid

    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    grid = build_periodic_grid(problem.periodic_cell, problem.grid)
    steps = np.diag(lattice) / np.asarray(grid.points_per_axis, dtype=np.float64)
    shift = np.asarray(offset_steps, dtype=np.float64) * steps
    atoms = tuple(
        Atom(atom.symbol, tuple(np.mod(np.asarray(atom.position) + shift, np.diag(lattice)))) for atom in problem.atoms
    )
    return replace(problem, atoms=atoms)


def _folded(problem, atoms):
    """``problem`` with ``atoms`` folded into its cell."""

    sides = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    return replace(problem, atoms=tuple(Atom(atom.symbol, tuple(np.mod(atom.position, sides))) for atom in atoms))


def _repeated(problem, repeats):
    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    atoms = []
    for i in range(repeats[0]):
        for j in range(repeats[1]):
            for k in range(repeats[2]):
                shift = np.array([i, j, k]) @ lattice
                atoms += [Atom(atom.symbol, tuple(np.asarray(atom.position) + shift)) for atom in problem.atoms]
    count = int(np.prod(repeats))
    return replace(
        problem,
        atoms=tuple(atoms),
        periodic_cell=replace(problem.periodic_cell, lattice_vectors=np.diag(np.diag(lattice) * np.asarray(repeats))),
        scf=replace(problem.scf, number_of_states=problem.scf.number_of_states * count),
    )


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class PeriodicDetectionSCFTests(unittest.TestCase):
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

    def test_a_low_symmetry_cell_repeated(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        base = silicon("pbe", spacing=0.64)
        rng = np.random.default_rng(3)
        atoms = tuple(
            Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.uniform(-0.15, 0.15, 3))) for atom in base.atoms
        )
        problem = _repeated(_folded(base, atoms), (2, 1, 1))
        sectors, details, _ = self._run(problem, {})
        self.assertIn("1 of them with a translation", details["symmetry_detection"])
        before, before_details, _ = self._run(problem, {"PARSEC_PERIODIC_TRIVIAL_ORIGIN": "off"})
        self.assertIn("identity only", before_details["symmetry_detection"])
        full, _, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        self._assert_same(sectors, full, "P1 2x1x1")
        self._assert_same(before, full, "P1 2x1x1 before")

    def test_four_ranks_with_the_gpus_of_their_nodes_keep_one_sector_of_a_cell_repeated_twice(self) -> None:
        # The group the detection finds for a P1 cell repeated twice has two sectors; four ranks that take one or
        # more each (the one-rank-per-node MPI runner, threads here) cannot divide them, and before the fix every
        # rank stopped in its sector configuration.  They keep the group about the origin (the identity) and share
        # its one sector, as before the detection looked for the group (review finding C3).
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon
        from parsec_python.acceleration.tests.thread_mpi import run_ranks

        base = silicon("pbe", spacing=0.64)
        rng = np.random.default_rng(3)
        atoms = tuple(
            Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.uniform(-0.15, 0.15, 3))) for atom in base.atoms
        )
        problem = _repeated(_folded(base, atoms), (2, 1, 1))

        def rank_main(thread):
            context = MPISectorContext(thread.comm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy", mpi_context=context)
            solver = system.backend.symmetry_eigensolver
            return (
                solver.decomposition.representation_count,
                context.shared,
                dict(system.backend_info.details)["symmetry_detection"],
            )

        for count, shared, detection in run_ranks(4, rank_main):
            self.assertEqual((count, shared), (1, True))
            self.assertIn("are fewer than the 4 MPI ranks", detection)
            self.assertIn("one sector of the identity", detection)

    def test_a_supercell_off_the_origin_with_forces(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon

        problem = _moved(_repeated(displaced_silicon(), (3, 1, 1)), (1, 2, 3))
        sectors, details, sector_forces = self._run(problem, {}, forces=True)
        # D2d about the displaced atom: the operations of Td that keep the x axis
        self.assertIn("exact group of 8 operations", details["symmetry_detection"])
        _, before_details, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY_CENTRE": "off"})
        self.assertNotIn("exact group of 8 operations", before_details["symmetry_detection"])
        full, _, full_forces = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"}, forces=True)
        self._assert_same(sectors, full, "Si 3x1x1 off the origin")
        self.assertGreater(float(np.abs(full_forces.total).max()), 1.0e-3)
        np.testing.assert_allclose(sector_forces.total, full_forces.total, rtol=0, atol=2.0e-6)

    def test_the_abelian_group_with_translations(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        problem = _repeated(silicon("pbe", spacing=0.64), (2, 1, 1))
        sectors, details, _ = self._run(problem, {"PARSEC_ORBITAL_GROUP": "abelian"})
        self.assertIn("exact commuting group of 16 operations", details["symmetry_detection"])
        before, before_details, _ = self._run(
            problem, {"PARSEC_ORBITAL_GROUP": "abelian", "PARSEC_PERIODIC_ABELIAN_TRANSLATIONS": "off"}
        )
        self.assertNotIn("with a translation", before_details["symmetry_detection"])
        full, _, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        self._assert_same(sectors, full, "Si 2x1x1 abelian")
        self._assert_same(before, full, "Si 2x1x1 abelian before")

    def test_a_noisy_cell_and_the_tolerance(self) -> None:
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        base = silicon("pbe", spacing=0.64)
        rng = np.random.default_rng(5)
        problem = _folded(
            base,
            [Atom(atom.symbol, tuple(np.asarray(atom.position) + rng.normal(0.0, 1.0e-5, 3))) for atom in base.atoms],
        )
        _, details, _ = self._run(problem, {})
        self.assertIn("PARSEC_SYMMETRY_ATOM_TOLERANCE", details["symmetry_detection"])
        loose, loose_details, _ = self._run(problem, {"PARSEC_SYMMETRY_ATOM_TOLERANCE": "1e-4"})
        self.assertIn("exact group of", loose_details["symmetry_detection"])
        full, _, _ = self._run(problem, {"PARSEC_PERIODIC_SYMMETRY": "off"})
        # The sectors solve a Hamiltonian averaged over the group, not that of the input: the projectors of one atom
        # per orbit sit at the images of its representative, the other terms at the noisy positions, so the energy
        # moves at first order in the noise (3.5e-5 Ry for 1e-5 bohr on each coordinate, measured).
        self.assertTrue(loose.converged)
        self.assertAlmostEqual(loose.energies.total, full.energies.total, delta=2.0e-4)


if __name__ == "__main__":
    unittest.main()
