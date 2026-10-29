"""Forces, relaxation and molecular dynamics of a periodic cell: the geometry update, the warm starts, the periodic
backend, and the command line.

The system is two hydrogen atoms in a cubic cell (the 773-point table of ``examples/0d_AuH``).  Its preparation takes a
fraction of a second since the image sums stop where the tables are Coulomb or zero.
"""

from __future__ import annotations

import re

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np

from parsec_python.cli import main as reference_main
from parsec_python.acceleration.cli import main as accelerated_main
from parsec_python.Forces import compute_periodic_forces
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    RelaxationSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Relaxation import periodic_backend, run_relaxation
from parsec_python.SCF import fold_atoms_into_cell, prepare_periodic_single_point, update_periodic_geometry
from parsec_python.SCF.kpoints import run_scf_kpoints
from parsec_python.V_ion.ionic_potential import atomic_density_support_radius, coulomb_onset_radius

_H = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "H_POTRE.DAT"
_SIDE = 6.0
_POSITIONS = ((2.9, 3.1, 3.0), (4.35, 3.0, 3.2))
_K_POINTS = np.array([[0.21, -0.37, 0.11], [-0.40, 0.15, 0.30]])
_K_WEIGHTS = np.array([0.5, 0.5])


def _problem(positions=_POSITIONS) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("H", list(position)) for position in positions],
        pseudopotentials={"H": SpeciesPotential(_H, 0)},
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=_SIDE * np.eye(3)),
        scf=SCFSettings(max_iterations=80, number_of_states=3, convergence_criterion=1.0e-9),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
    )


def _moved(shift=(0.05, -0.04, 0.03)):
    return tuple(Atom("H", list(np.array(position) + np.array(shift) * (1 if index == 0 else -0.5)))
                 for index, position in enumerate(_POSITIONS))


class FoldAtomsTests(unittest.TestCase):
    def test_atoms_inside_the_cell_are_returned_as_they_are(self) -> None:
        atoms = tuple(Atom("H", list(position)) for position in _POSITIONS)
        folded = fold_atoms_into_cell(atoms, _SIDE * np.eye(3))
        self.assertTrue(all(new is old for new, old in zip(folded, atoms)))

    def test_atoms_outside_are_moved_by_lattice_vectors(self) -> None:
        for lattice in (_SIDE * np.eye(3), 5.4 * (np.ones((3, 3)) - np.eye(3))):
            with self.subTest(lattice=lattice[0].tolist()):
                inside = np.array([0.3, 0.6, 0.2]) @ lattice
                outside = inside + np.array([2, -3, 1]) @ lattice
                (folded,) = fold_atoms_into_cell((Atom("H", list(outside)),), lattice)
                np.testing.assert_allclose(folded.position, inside, atol=1.0e-12)
                fraction = np.asarray(folded.position) @ np.linalg.inv(lattice)
                self.assertTrue(np.all(fraction >= 0.0) and np.all(fraction < 1.0))


class SupportRadiusTests(unittest.TestCase):
    def test_the_tables_are_bounded_far_inside_their_length(self) -> None:
        system = prepare_periodic_single_point(_problem())
        potential = system.pseudopotentials["H"]
        self.assertLess(coulomb_onset_radius(potential, 0), 5.0)
        self.assertLess(atomic_density_support_radius(potential, system.input.pseudopotentials["H"], core=False), 40.0)
        self.assertLess(potential.interpolation_cutoff, 200.0)  # the table runs to 118 bohr
        self.assertGreater(potential.interpolation_cutoff, 100.0)

    def test_the_bounded_sums_equal_the_sums_to_the_table_end(self) -> None:
        # the density sum over images out to the end of the table (what the code did) gives the same fields
        from parsec_python.V_ion import ionic_potential

        bounded = prepare_periodic_single_point(_problem())
        original = ionic_potential.atomic_density_support_radius
        ionic_potential.atomic_density_support_radius = lambda potential, specification, core: potential.interpolation_cutoff
        try:
            full = prepare_periodic_single_point(_problem())
        finally:
            ionic_potential.atomic_density_support_radius = original
        np.testing.assert_allclose(bounded.initial_density, full.initial_density, atol=1.0e-12)
        np.testing.assert_allclose(bounded.ionic_potential, full.ionic_potential, atol=1.0e-9)


class UpdateGeometryTests(unittest.TestCase):
    def test_the_update_is_a_fresh_preparation_of_the_new_positions(self) -> None:
        system = prepare_periodic_single_point(_problem())
        moved = _moved()
        updated = update_periodic_geometry(system, moved)
        fresh = prepare_periodic_single_point(replace(_problem(), atoms=moved))
        np.testing.assert_array_equal(updated.ionic_potential, fresh.ionic_potential)
        np.testing.assert_array_equal(updated.initial_density, fresh.initial_density)
        np.testing.assert_array_equal(updated.core_density, fresh.core_density)
        self.assertEqual(updated.ion_ion_energy, fresh.ion_ion_energy)
        self.assertEqual((updated.nonlocal_operator.projectors != fresh.nonlocal_operator.projectors).nnz, 0)
        self.assertIs(updated.grid, system.grid)
        self.assertIs(updated.negative_laplacian, system.negative_laplacian)

    def test_positions_outside_the_cell_are_kept_and_folded_for_the_builders(self) -> None:
        system = prepare_periodic_single_point(_problem())
        moved = _moved()
        away = tuple(
            Atom("H", list(np.array(atom.position) + np.array(shift) * _SIDE))
            for atom, shift in zip(moved, ((1, 0, 0), (-2, 1, 3)))
        )
        outside = update_periodic_geometry(system, away)
        inside = update_periodic_geometry(system, moved)
        # the coordinates are the caller's (an optimizer needs them continuous) ...
        np.testing.assert_allclose([a.position for a in outside.atoms], [a.position for a in away])
        # ... and the physics is that of the folded structure
        np.testing.assert_allclose(outside.ionic_potential, inside.ionic_potential, atol=1.0e-12)
        np.testing.assert_allclose(outside.initial_density, inside.initial_density, atol=1.0e-12)
        self.assertAlmostEqual(outside.ion_ion_energy, inside.ion_ion_energy, places=12)
        backend = periodic_backend()
        a, b = backend.run_scf(outside), backend.run_scf(inside)
        self.assertAlmostEqual(a.energies.total, b.energies.total, places=8)
        np.testing.assert_allclose(
            compute_periodic_forces(outside, a).unsymmetrized, compute_periodic_forces(inside, b).unsymmetrized, atol=1.0e-6
        )

    def test_the_moved_atoms_must_keep_their_species_and_number(self) -> None:
        system = prepare_periodic_single_point(_problem())
        with self.assertRaises(ValueError):
            update_periodic_geometry(system, _moved()[:1])
        with self.assertRaises(ValueError):
            update_periodic_geometry(system, (Atom("H", [1.0, 1.0, 1.0]), Atom("He", [2.0, 2.0, 2.0])))


class WarmStartTests(unittest.TestCase):
    def _check(self, backend, extrapolation: str) -> None:
        system = prepare_periodic_single_point(_problem())
        previous = backend.run_scf(system)
        moved = update_periodic_geometry(system, _moved())
        cold = backend.run_scf(moved)
        warm_start = periodic_backend(
            *( (_K_POINTS, _K_WEIGHTS) if previous.k_points is not None else (None, None) ), extrapolation
        ).warm_start(system, previous, moved)
        warm = backend.run_scf(moved, warm_start=warm_start)
        self.assertTrue(cold.converged and warm.converged)
        self.assertAlmostEqual(warm.energies.total, cold.energies.total, places=7)
        if extrapolation == "none":
            # the previous potential is a better start than the atomic one; the atomic extrapolation pays off for
            # large structures only (it does not for two atoms), so it is held to the energy alone
            self.assertLessEqual(warm.iterations, cold.iterations)

    def test_gamma_point(self) -> None:
        for extrapolation in ("none", "atomic"):
            with self.subTest(extrapolation=extrapolation):
                self._check(periodic_backend(), extrapolation)

    def test_kpoints_start_their_eigenproblems_cold_but_from_the_previous_potential(self) -> None:
        for extrapolation in ("none", "atomic"):
            with self.subTest(extrapolation=extrapolation):
                self._check(periodic_backend(_K_POINTS, _K_WEIGHTS), extrapolation)

    def test_a_looser_criterion_for_one_loop(self) -> None:
        system = prepare_periodic_single_point(_problem())
        tight = run_scf_kpoints(system, _K_POINTS, _K_WEIGHTS)
        loose = run_scf_kpoints(system, _K_POINTS, _K_WEIGHTS, convergence_criterion=1.0e-5)
        self.assertLess(loose.iterations, tight.iterations)
        self.assertAlmostEqual(loose.energies.total, tight.energies.total, places=4)


class PeriodicBackendTests(unittest.TestCase):
    def test_kpoint_runs_carry_their_nonlocal_forces(self) -> None:
        system = prepare_periodic_single_point(_problem())
        backend = periodic_backend(_K_POINTS, _K_WEIGHTS)
        result = backend.run_scf(system)
        self.assertEqual(result.nonlocal_forces.shape, (2, 3))
        forces = backend.compute_forces(system, result)
        self.assertEqual(forces.torque.tolist(), [0.0, 0.0, 0.0])  # a crystal has no torque
        np.testing.assert_allclose(forces.total.sum(axis=0), 0.0, atol=1.0e-12)  # the net force is removed

    def test_the_point_group_of_a_cluster_is_refused(self) -> None:
        system = prepare_periodic_single_point(_problem())
        backend = periodic_backend()
        result = backend.run_scf(system)
        with self.assertRaises(ValueError):
            backend.compute_forces(system, result, symmetrize=True)

    def test_a_relaxation_lowers_the_energy(self) -> None:
        for sampled, backend in ((False, periodic_backend()), (True, periodic_backend(_K_POINTS, _K_WEIGHTS, "atomic"))):
            with self.subTest(kpoints=sampled):
                system = prepare_periodic_single_point(_problem())
                settings = RelaxationSettings(method="bfgs", max_movements=4, minimum_force=1.0e-3)
                outcome = run_relaxation(system, settings, backend, symmetrize=False)
                energies = [record.energy for record in outcome.records]
                self.assertGreaterEqual(len(energies), 3)
                self.assertLess(energies[-1], energies[0])
                self.assertTrue(all(record.converged for record in outcome.records))
                # every structure was made from the positions of the optimizer, which may leave the cell
                self.assertEqual(outcome.records[-1].positions.shape, (2, 3))


_INPUT = """
Periodic_System: .true.
Boundary_Conditions bulk
begin Cell_Shape
 6.0 0.0 0.0
 0.0 6.0 0.0
 0.0 0.0 6.0
end Cell_Shape
Lattice_Vector_Scale: 1.0
Grid_Spacing: 0.3
Expansion_Order: 8
Correlation_Type: ca
States_Num: 3
Convergence_Criterion: 1e-8 Ry
Max_Iter: 80
Eigensolver: chebdav
Diag_Tolerance: 1e-9
Fermi_Temp: 300 K
Mixing_Param: {mixing}
Memory_Param: 6
Mixing_Safeguard: true
{extra}
Atom_Types_Num: 1
Coordinate_Unit: cartesian_bohr
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
 2.9 3.1 3.0
 4.35 3.0 3.2
end Atom_Coord
"""


class CommandLineTests(unittest.TestCase):
    def _run(self, extra: str, main=reference_main, *options: str, mixing: str = "0.1"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name)
        (folder / "parsec.in").write_text(_INPUT.format(extra=extra, mixing=mixing))
        (folder / "H_POTRE.DAT").write_bytes(_H.read_bytes())
        output, error = StringIO(), StringIO()
        previous = os.getcwd()
        os.chdir(folder)
        try:
            with redirect_stdout(output), redirect_stderr(error):
                code = main([str(folder / "parsec.in"), "--no-archive", *options])
        finally:
            os.chdir(previous)
        return code, error.getvalue(), folder

    def test_forces_of_a_periodic_cell(self) -> None:
        code, _, folder = self._run("Skip_Force: .false.")
        self.assertEqual(code, 0)
        text = (folder / "parsec.out").read_text()
        for line in ("Coordinates and local force components", "(Ewald forces included)", "Reporting non-local force components"):
            self.assertIn(line, text)
        self.assertIn("after setting net force to zero", text)
        self.assertNotIn("tot. torque", text)  # Fortran reports none for a crystal
        self.assertNotIn("Forces from ion-ion interaction", text)

    def test_forces_with_kpoints(self) -> None:
        code, _, folder = self._run("Skip_Force: .false.\nKpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid")
        self.assertEqual(code, 0)
        self.assertIn("after setting net force to zero", (folder / "parsec.out").read_text())

    def test_the_accelerated_driver_hands_a_periodic_cell_with_forces_to_the_reference_solver(self) -> None:
        code, error, folder = self._run("Skip_Force: .false.", accelerated_main)
        self.assertEqual(code, 0)
        self.assertIn("running the reference parsec_python.cli solver instead", error)
        self.assertIn("after setting net force to zero", (folder / "parsec.out").read_text())

    def test_relaxation(self) -> None:
        code, _, folder = self._run("Minimization: bfgs\nMovement_Num: 4\nForce_Min: 1e-3 Ry/bohr")
        self.assertIn(code, (0, 4))  # 4: not relaxed within the movements
        for name in ("atom.cor", "relaxation.xyz", "relax_restart.dat"):
            self.assertTrue((folder / name).is_file(), name)
        frames = (folder / "relaxation.xyz").read_text().count("energy_Ry=")  # one frame per structure
        self.assertGreaterEqual(frames, 3)

    def test_molecular_dynamics(self) -> None:
        code, _, folder = self._run("Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 200\nT_Step: 10\nTime_Step: 20\nStep_Num: 3\nRng_Seed: 7")
        self.assertEqual(code, 0)
        rows = (folder / "md_nrg.dat").read_text().strip().splitlines()
        self.assertEqual(len(rows), 1 + 4)  # the header and steps 0 to 3
        total = [float(row.split()[4]) for row in rows[1:]]
        self.assertLess(max(total) - min(total), 5.0e-3)  # eV per atom: the kinetic and potential energies move by ~0.03

    _SPIN = "Net_Charges: 1 e\nSpin_Polarization: .true.\n"

    def test_forces_of_a_spin_polarized_cell(self) -> None:
        code, _, folder = self._run(self._SPIN + "Skip_Force: .false.", mixing="0.3")
        self.assertEqual(code, 0)
        text = (folder / "parsec.out").read_text()
        self.assertIn("Net magnetic moment", text)
        for line in ("(Ewald forces included)", "Reporting non-local force components", "after setting net force to zero"):
            self.assertIn(line, text)

    def test_relaxation_of_a_spin_polarized_cell(self) -> None:
        code, _, folder = self._run(
            self._SPIN + "Minimization: bfgs\nMovement_Num: 4\nForce_Min: 1e-3 Ry/bohr", mixing="0.3"
        )
        self.assertIn(code, (0, 4))
        energies = [float(value) for value in re.findall(r"energy_Ry=(-?\d+\.\d+)", (folder / "relaxation.xyz").read_text())]
        self.assertGreaterEqual(len(energies), 3)
        self.assertLess(energies[-1], energies[0])

    def test_molecular_dynamics_of_a_spin_polarized_cell(self) -> None:
        code, _, folder = self._run(
            self._SPIN + "Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 200\nT_Step: 10\nTime_Step: 20\nStep_Num: 3\nRng_Seed: 7",
            mixing="0.3",
        )
        self.assertEqual(code, 0)
        rows = (folder / "md_nrg.dat").read_text().strip().splitlines()
        self.assertEqual(len(rows), 1 + 4)
        total = [float(row.split()[4]) for row in rows[1:]]
        self.assertLess(max(total) - min(total), 5.0e-3)

    def test_a_spin_polarized_cell_with_kpoints_has_no_forces(self) -> None:
        extra = self._SPIN + "Skip_Force: .false.\nKpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid"
        code, error, _ = self._run(extra)
        self.assertEqual(code, 2)
        self.assertIn("spin-polarized periodic calculation with k-points", error)


if __name__ == "__main__":
    unittest.main()
