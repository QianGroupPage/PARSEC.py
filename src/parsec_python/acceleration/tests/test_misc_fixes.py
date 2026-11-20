"""Small fixes of 2026-11-18 found by the survey of how the SCF uses symmetry (stream X).

* The count of the sector states that the storage and margin rules read
  (``CuPySymmetrySCFEigensolver._state_fit_bytes``) holds the columns of both
  channels of a spin-polarized loop on an MPI rank that shares a sector basis
  with its group, as it does on one device.

* A cluster with a point group whose Hartree solve (and CA-LDA) leaves the
  sectors for the full grid for want of the native extension says so in a
  NOTE line of parsec.out (``driver._cluster_full_grid_notes``).
* The MPI runner says in a NOTE when ranks share sectors after a CHEBDAV
  first solve, which gives each of them the whole basis
  (``mpi_full_scf._shared_first_solve_note``).
* A ``relax_restart.dat`` that holds an atom of a cell on the default
  routines at -1e-9 bohr (on a face to round-off) is prepared, folded into
  the cell for the builders (``SCF.pbc.atoms_for_cell_builders``); since
  2026-11-19 also one that crossed a face (refused before), and the current
  positions of an ``mdinit.dat`` (``Restart_mode``), with the energy and the
  forces of the input folded into the cell.

All tests run on the CPU: the count reads no device for a shared group.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


class SharedGroupSpinCountTests(unittest.TestCase):
    """``_state_fit_bytes`` of a sector whose basis the MPI ranks of a group share."""

    rows = 1_000_003
    count = 1_234

    def counted(self, *, channels: int, route: str, initial_method: str = "chebff", size: int = 4) -> int:
        from parsec_python.acceleration.Eigensolvers import symmetry

        solver = SimpleNamespace(settings=SimpleNamespace(initial_method=initial_method))
        group = SimpleNamespace(size=size, ritz_route=route)
        stub = SimpleNamespace(
            resident_device_bytes={},
            device_ids=(0,),
            _solvers=[solver],
            decomposition=SimpleNamespace(sector_size=lambda representation: self.rows),
            _sector_device_ids=[0],
            _operators=[SimpleNamespace(_mpi_shared_group=group)],
            spin_channel_count=channels,
        )
        with patch.object(symmetry, "require_cupy", return_value=(None, None)):
            return symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes(stub, [self.count])[0]

    def test_every_shared_route_counts_the_columns_of_both_channels(self) -> None:
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import exchange_bytes

        share = -(-self.count // 4)
        columns = self.rows * share * 8
        for route in ("block", "rounds", "rows"):
            with self.subTest(route=route):
                one = self.counted(channels=1, route=route)
                two = self.counted(channels=2, route=route)
                # The second channel adds this rank's columns of its basis, and nothing else.
                self.assertEqual(two - one, columns)
        # The unpolarized counts are those of before.
        self.assertEqual(self.counted(channels=1, route="block"),
                         columns + 4 * exchange_bytes() + 11 * self.count ** 2 * 8)
        self.assertEqual(self.counted(channels=1, route="rounds"),
                         2 * columns + 3 * exchange_bytes() + 2 * self.count ** 2 * 8)
        self.assertEqual(self.counted(channels=1, route="rows"), 4 * columns + 2 * exchange_bytes())

    def test_a_whole_basis_after_chebdav_counts_both_channels_as_before(self) -> None:
        one = self.counted(channels=1, route="block", initial_method="chebdav")
        two = self.counted(channels=2, route="block", initial_method="chebdav")
        self.assertEqual(two - one, self.rows * self.count * 8)


class ClusterFullGridNoteTests(unittest.TestCase):
    """The NOTE of a cluster whose Hartree solve or CA-LDA leaves the sectors without the native extension."""

    @staticmethod
    def selection(requested, hartree, reasons=()):
        return SimpleNamespace(requested=requested, selected="cupy", hartree_backend=hartree,
                               fallback_reasons=tuple(reasons))

    def test_the_note_names_the_full_grid_and_the_reason(self) -> None:
        from parsec_python.acceleration import driver

        notes = driver._cluster_full_grid_notes(self.selection("cupy", "cupy"), periodic=False, group_found=True,
                                                orbitals_in_sectors=True)
        self.assertEqual(len(notes), 1)
        for words in ("Hartree solve of this cluster runs on the full grid", "--backend cupy",
                      "the orbitals stay in the symmetry sectors"):
            self.assertIn(words, notes[0])
        (missing,) = driver._cluster_full_grid_notes(
            self.selection("auto", "cupy", ["native components skipped: no module named _parsec_native"]),
            periodic=False, group_found=True, orbitals_in_sectors=False)
        self.assertIn("the native C++ extension is not available (native components skipped", missing)
        self.assertNotIn("orbitals stay", missing)
        xc = driver._cluster_xc_full_grid_note(self.selection("cupy", "cupy"), functional="CA-LDA")
        self.assertIn("XC functional (CA-LDA) of this cluster is evaluated on the full grid of the host", xc)

    def test_no_note_where_the_wedge_serves_or_nothing_is_lost(self) -> None:
        from parsec_python.acceleration import driver

        for selection, periodic, group in ((self.selection("auto", "native"), False, True),
                                           (self.selection("cupy", "cupy"), True, True),
                                           (self.selection("cupy", "cupy"), False, False)):
            with self.subTest(hartree=selection.hartree_backend, periodic=periodic, group=group):
                self.assertEqual(driver._cluster_full_grid_notes(selection, periodic=periodic, group_found=group,
                                                                 orbitals_in_sectors=group), [])

    def test_the_report_writes_the_note_at_every_output_level(self) -> None:
        from parsec_python.acceleration.Output.accelerated_output import AcceleratedTextReporter

        written = []
        reporter = SimpleNamespace(reference=SimpleNamespace(setup=lambda system: None),
                                   _start_radius_estimates=lambda system: None, output_level=1,
                                   write=written.append)
        details = (("orbital_symmetry", "CuPy real sectors"), ("symmetry_full_grid_note", "the Hartree solve ..."),
                   ("hartree_backend", "cupy"))
        system = SimpleNamespace(
            backend_info=SimpleNamespace(details=details, requested="cupy", selected="cupy", dtype="float64",
                                         device="cuda:0", implementation="x", fallback_reasons=()),
            backend=SimpleNamespace(statistics=SimpleNamespace(initialization_seconds=0.0)))
        AcceleratedTextReporter.setup(reporter, system)
        text = "\n".join(written)
        self.assertIn(" NOTE: the Hartree solve ...", text)
        self.assertNotIn("symmetry_full_grid_note =", text)
        self.assertIn(" hartree_backend = cupy", text)


class SharedChebdavNoteTests(unittest.TestCase):
    """The NOTE of the MPI runner where ranks share sectors after a CHEBDAV first solve."""

    def test_the_note_recommends_chebff(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        context = SimpleNamespace(shared=True, rank_groups=((0, 1), (2, 3), (4, 5, 6)))
        note = mpi_full_scf._shared_first_solve_note(context, "chebdav")
        self.assertTrue(note.startswith("NOTE: 3 sector(s) are shared by groups of 2/3 MPI ranks"))
        self.assertIn("Eigensolver: chebff divides each basis", note)
        one_sector = SimpleNamespace(shared=True, rank_groups=((0, 1, 2, 3),))
        self.assertIn("groups of 4 MPI ranks", mpi_full_scf._shared_first_solve_note(one_sector, "chebdav"))

    def test_no_note_for_chebff_or_unshared_sectors(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        shared = SimpleNamespace(shared=True, rank_groups=((0, 1),))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(shared, "chebff"))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(shared, "CHEBFF"))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(None, "chebdav"))
        self.assertIsNone(mpi_full_scf._shared_first_solve_note(
            SimpleNamespace(shared=False, rank_groups=((0,), (1,))), "chebdav"))


_SILICON = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
 10.26 10.26 10.26
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: lattice_vectors
Grid_Spacing: 0.75
Expansion_Order: 6
States_Num: 26
Fermi_Temp: 500 K
Correlation_Type: ca
Eigensolver: chebff
Minimization: simple
Movement_Num: 1
Relax_Restart: .true.
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
begin Atom_Coord
 0.00 0.00 0.00
 0.00 0.50 0.50
 0.50 0.00 0.50
 0.50 0.50 0.00
 0.25 0.25 0.25
 0.25 0.75 0.75
 0.75 0.25 0.75
 0.75 0.75 0.25
end Atom_Coord
"""


class RestartOnCellFaceTests(unittest.TestCase):
    """A ``relax_restart.dat`` of a cell on the default routines with an atom at -1e-9 bohr."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        shutil.copy(DATA / "Si_POTRE.DAT", self.directory / "Si_POTRE.DAT")
        (self.directory / "parsec.in").write_text(_SILICON, encoding="utf-8")

    def restarted(self, first):
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.Relaxation.files import RelaxRestart, format_relax_restart
        from parsec_python.Relaxation.run import restart_structure

        translation = parse_parsec_input(self.directory / "parsec.in")
        self.assertFalse(translation.problem.grid.fortran_routines)
        positions = np.array([atom.position for atom in translation.problem.atoms], dtype=np.float64)
        positions[0] = first
        (self.directory / "relax_restart.dat").write_text(
            format_relax_restart(RelaxRestart(positions=positions, iteration=1, movable_atoms=8)), encoding="utf-8")
        restarted, _restart, notes = restart_structure(translation, self.directory)
        self.assertTrue(any("relax_restart.dat" in note for note in notes))
        return translation, restarted

    def test_an_atom_on_a_face_is_prepared_folded_for_the_builders(self) -> None:
        from dataclasses import replace

        from parsec_python.SCF.pbc import atoms_for_cell_builders, prepare_periodic_single_point

        original, restarted = self.restarted([-1.0e-9, 0.0, -1.1e-9])
        atoms = restarted.problem.atoms
        np.testing.assert_array_equal(atoms[0].position, [-1.0e-9, 0.0, -1.1e-9])
        lattice = restarted.problem.periodic_cell.lattice_vectors
        folded = atoms_for_cell_builders(lattice, atoms)
        self.assertTrue(np.all(np.asarray(folded[0].position) >= 0.0))
        self.assertTrue(all(new is old for new, old in zip(folded[1:], atoms[1:])))
        # Inside the cell nothing changes: the same atoms, bit for bit.
        inside = original.problem.atoms
        self.assertIs(atoms_for_cell_builders(lattice, inside)[0], inside[0])
        system = prepare_periodic_single_point(restarted.problem)
        # The system keeps the coordinates of the file (an optimizer continues from them); its fields are those of
        # the atoms folded into the cell, and within round-off those of the atom at the corner.
        np.testing.assert_array_equal(system.atoms[0].position, atoms[0].position)
        at_folded = prepare_periodic_single_point(replace(restarted.problem, atoms=folded))
        np.testing.assert_array_equal(system.ionic_potential, at_folded.ionic_potential)
        self.assertEqual(system.ion_ion_energy, at_folded.ion_ion_energy)
        at_corner = prepare_periodic_single_point(original.problem)
        np.testing.assert_allclose(system.ionic_potential, at_corner.ionic_potential, rtol=0.0, atol=1.0e-6)
        self.assertAlmostEqual(system.ion_ion_energy, at_corner.ion_ion_energy, delta=1.0e-6)

    def test_an_atom_that_crossed_a_face_is_the_folded_input(self) -> None:
        # Since 2026-11-19 (the user's decision): an atom of relax_restart.dat outside the cell by more than round-off
        # (refused before) is prepared folded into the cell for the builders, as after every move, and the system
        # keeps the coordinates of the file, which the BFGS block of the file refers to.  The energy and the forces
        # are those of the input with the atom folded into the cell, bit for bit.
        from dataclasses import replace

        from parsec_python.Forces import compute_periodic_forces
        from parsec_python.SCF import run_scf
        from parsec_python.SCF.pbc import fold_atoms_into_cell, prepare_periodic_single_point

        side = 10.26
        _original, restarted = self.restarted([-0.31, side + 0.12, 0.05 - 2.0 * side])
        atoms = restarted.problem.atoms
        np.testing.assert_allclose(atoms[0].position, [-0.31, side + 0.12, 0.05 - 2.0 * side], rtol=0.0, atol=1e-12)
        folded = fold_atoms_into_cell(atoms, restarted.problem.periodic_cell.lattice_vectors)
        np.testing.assert_allclose(folded[0].position, [side - 0.31, 0.12, 0.05], rtol=0.0, atol=1e-12)
        system = prepare_periodic_single_point(restarted.problem)
        np.testing.assert_array_equal(system.atoms[0].position, atoms[0].position)
        at_folded = prepare_periodic_single_point(replace(restarted.problem, atoms=folded))
        _assert_same_fields(self, system, at_folded)
        result, reference = run_scf(system), run_scf(at_folded)
        self.assertTrue(result.converged)
        self.assertEqual(result.energies.total, reference.energies.total)
        np.testing.assert_array_equal(
            compute_periodic_forces(system, result).total, compute_periodic_forces(at_folded, reference).total
        )

    def test_a_dynamics_restart_with_an_atom_outside_is_the_folded_input(self) -> None:
        # Restart_mode: the current positions of mdinit.dat, one atom 0.4 bohr beyond a face (thermal motion of a
        # corner atom), are prepared folded for the builders; the system keeps them, as the integrator does with
        # the previous positions of the file.  The orbital operators of an MPI rank that shares a sector basis
        # (driver._prepare_periodic_orbital_operators) fold them alike.
        from dataclasses import replace

        from parsec_python.acceleration.driver import _prepare_periodic_orbital_operators
        from parsec_python.Input.parsec_input import parse_parsec_input
        from parsec_python.MolecularDynamics.md_files import format_md_restart
        from parsec_python.MolecularDynamics.moldyn import MolecularDynamicsState
        from parsec_python.MolecularDynamics.run import restart_structure
        from parsec_python.SCF.pbc import fold_atoms_into_cell, prepare_periodic_single_point

        text = _SILICON.replace("Minimization: simple\nMovement_Num: 1\nRelax_Restart: .true.\n", "")
        text = text.replace(
            "Eigensolver: chebff\n",
            "Eigensolver: chebff\nMolecular_Dynamics: .true.\nRestart_mode: .true.\nTinit: 300\nT_Final: 200\n"
            "T_Step: 10\nTime_Step: 20\nStep_Num: 1\n",
        )
        (self.directory / "parsec.in").write_text(text, encoding="utf-8")
        translation = parse_parsec_input(self.directory / "parsec.in")
        self.assertFalse(translation.problem.grid.fortran_routines, translation.routines.reasons)
        self.assertIsNotNone(translation.molecular_dynamics)
        current = np.array([atom.position for atom in translation.problem.atoms], dtype=np.float64)
        current[0] = [-0.4, 0.1, 10.26 + 0.2]
        previous = current - 0.01
        count = len(current)
        state = MolecularDynamicsState(
            masses=np.full(count, 28.0855 * 1822.888486),
            movable=np.ones(count, dtype=np.int64),
            current_positions=current,
            previous_positions=previous,
            velocities=np.zeros((count, 3)),
            accelerations=np.zeros((count, 3)),
            temperature=300.0,
            frame=3,
        )
        (self.directory / "mdinit.dat").write_text(format_md_restart(state), encoding="utf-8")
        restarted, restart, notes = restart_structure(translation, self.directory)
        self.assertIsNotNone(restart)
        self.assertTrue(any("mdinit.dat" in note for note in notes))
        atoms = restarted.problem.atoms
        np.testing.assert_allclose(atoms[0].position, [-0.4, 0.1, 10.46], rtol=0.0, atol=1e-6)
        system = prepare_periodic_single_point(restarted.problem)
        np.testing.assert_array_equal(system.atoms[0].position, atoms[0].position)
        folded = fold_atoms_into_cell(atoms, restarted.problem.periodic_cell.lattice_vectors)
        self.assertTrue(np.all((np.asarray(folded[0].position) >= 0.0) & (np.asarray(folded[0].position) < 10.26)))
        _assert_same_fields(self, system, prepare_periodic_single_point(replace(restarted.problem, atoms=folded)))
        operators = _prepare_periodic_orbital_operators(restarted.problem)
        np.testing.assert_array_equal(operators.atoms[0].position, atoms[0].position)
        _assert_same_projectors(self, operators.nonlocal_operator, system.nonlocal_operator)


def _assert_same_projectors(case, first, second) -> None:
    """The Kleinman-Bylander factors of two prepared systems agree element for element."""

    case.assertEqual(first.labels, second.labels)
    for name in ("projectors", "signs"):
        a, b = getattr(first, name), getattr(second, name)
        a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
        b = b.toarray() if hasattr(b, "toarray") else np.asarray(b)
        np.testing.assert_array_equal(a, b)


def _assert_same_fields(case, system, other) -> None:
    """The position-dependent fields of two prepared periodic systems agree bit for bit."""

    np.testing.assert_array_equal(system.ionic_potential, other.ionic_potential)
    np.testing.assert_array_equal(system.initial_density, other.initial_density)
    if system.core_density is not None or other.core_density is not None:
        np.testing.assert_array_equal(system.core_density, other.core_density)
    case.assertEqual(system.ion_ion_energy, other.ion_ion_energy)
    _assert_same_projectors(case, system.nonlocal_operator, other.nonlocal_operator)


if __name__ == "__main__":
    unittest.main()
