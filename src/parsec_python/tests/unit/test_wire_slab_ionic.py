"""Relaxation, dynamics, spin forces and symmetry of wires and slabs.

Covers :func:`parsec_python.SCF.wire_slab.update_wire_slab_geometry` (equal
to a fresh preparation bit for bit, ``back_to_cell``, ``isolat``), the space
group of a wire or slab (:func:`parsec_python.Grid.space_group.find_wire_slab_space_group`,
``alatt`` with ``4 rmax``; operations that do not map the confined grid are
dropped), the density symmetrization on the confined grid, the forces of
spin-polarized wires and slabs (Gamma and k-points) and their symmetrization,
the input of Minimization/Molecular_Dynamics/Monkhorst_Pack_Shift for a wire
or slab and a CLI relaxation.  Fortran numbers are from PARSEC runs of the
inputs of the parity round's wire and slab validation.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from parsec_python.Grid.confined import build_confined_periodic_grid, confined_lattice_vectors
from parsec_python.Grid.point_group import symmetrize_scalar
from parsec_python.Grid.space_group import (
    confined_grid_symmetry,
    confined_symmetry_lattice,
    find_space_group,
    find_wire_slab_space_group,
    grid_preserving_operations,
)
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError, parse_parsec_input
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)

DATA = Path(__file__).resolve().parents[1] / "data"
HYDROGEN = DATA / "water_H_POTRE.DAT"
NA_CHAIN = [
    Atom("Na", (0.0, 2.931429, 0.0)),
    Atom("Na", (0.0, -2.931429, 0.0)),
    Atom("Na", (0.0, 0.0, 5.890929)),
    Atom("Na", (0.0, 0.0, -5.890929)),
]


def hydrogen_problem(periodicity, atoms, *, kpoints=None, weights=None, spin=False, radius=4.0, spacing=0.6):
    if periodicity == 1:
        cell = PeriodicCell(confined_lattice_vectors([[3.0, 0.0, 0.0]]))
    else:
        cell = PeriodicCell(confined_lattice_vectors([[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]]))
    return SinglePointInput(
        atoms=tuple(atoms),
        pseudopotentials={"H": SpeciesPotential(path=HYDROGEN, local_angular_momentum=0)},
        grid=PeriodicGridSettings(
            spacing=spacing,
            expansion_order=8,
            shift=(0.5, 0.5, 0.5),
            periodicity=periodicity,
            confinement_radius=radius,
        ),
        periodic_cell=cell,
        scf=SCFSettings(
            number_of_states=3,
            xc_functional="ca",
            convergence_criterion=1e-9,
            max_iterations=80,
            spin_polarized=spin,
        ),
        eigensolver=EigensolverSettings(method="chebdav", tolerance=1e-10),
        hartree=HartreeSettings(relative_tolerance=1e-12),
        kpoints=kpoints,
        kpoint_weights=weights,
    )


def _same(first, second) -> bool:
    if first is None or second is None:
        return first is second
    first = first.toarray() if hasattr(first, "toarray") else np.asarray(first)
    second = second.toarray() if hasattr(second, "toarray") else np.asarray(second)
    return first.shape == second.shape and np.array_equal(first, second)


class GeometryUpdateTests(unittest.TestCase):
    def assert_systems_equal(self, moved, fresh) -> None:
        for name in (
            "ionic_potential",
            "initial_density",
            "core_density",
            "gaussian_potential",
            "ion_ion_forces",
        ):
            self.assertTrue(_same(getattr(moved, name), getattr(fresh, name)), name)
        self.assertEqual(moved.ion_ion_energy, fresh.ion_ion_energy)
        self.assertTrue(_same(moved.nonlocal_operator.projectors, fresh.nonlocal_operator.projectors))
        self.assertEqual(
            [atom.position for atom in moved.atoms], [atom.position for atom in fresh.atoms]
        )

    def test_update_equals_a_fresh_preparation(self) -> None:
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point, update_wire_slab_geometry

        for periodicity in (1, 2):
            with self.subTest(periodicity=periodicity):
                atoms = [Atom("H", (0.2, 0.1, 0.7)), Atom("H", (-0.3, 0.0, -0.6))]
                system = prepare_wire_slab_single_point(hydrogen_problem(periodicity, atoms))
                moved_atoms = [Atom("H", (0.25, 0.05, 0.72)), Atom("H", (-0.35, 0.1, -0.55))]
                moved = update_wire_slab_geometry(system, moved_atoms)
                fresh = prepare_wire_slab_single_point(hydrogen_problem(periodicity, moved_atoms))
                self.assert_systems_equal(moved, fresh)
                self.assertIs(moved.grid, system.grid)

    def test_back_to_cell_and_isolat(self) -> None:
        from parsec_python.SCF.wire_slab import (
            prepare_wire_slab_single_point,
            update_wire_slab_geometry,
            wire_slab_back_to_cell,
        )

        lattice = confined_lattice_vectors([[3.0, 0.0, 0.0], [1.5, 2.6, 0.0]])
        inside = Atom("H", (0.2, 0.1, 0.7))
        outside = Atom("H", (2.0, 2.0, -0.6))
        placed = wire_slab_back_to_cell([inside, outside], lattice, 2)
        self.assertIs(placed[0], inside)
        # u = (0.1667, 0.7692): the second lattice vector is removed, z kept.
        np.testing.assert_allclose(placed[1].position, (0.5, -0.6, -0.6), atol=1e-12)
        fractional = np.asarray(placed[1].position) @ np.linalg.inv(lattice)
        self.assertTrue(np.all(np.abs(fractional[:2]) <= 0.5))

        atoms = [Atom("H", (0.2, 0.1, 0.7)), Atom("H", (-0.3, 0.0, -0.6))]
        system = prepare_wire_slab_single_point(hydrogen_problem(1, atoms))
        # Atom 2 has left the cell by -0.4 along x: it comes back by +3.
        moved = update_wire_slab_geometry(system, [atoms[0], Atom("H", (-1.9, 0.0, -0.6))])
        np.testing.assert_allclose(moved.atoms[1].position, (1.1, 0.0, -0.6), atol=1e-12)
        np.testing.assert_array_equal(moved.input.atoms[1].position, (-1.9, 0.0, -0.6))
        fresh = prepare_wire_slab_single_point(hydrogen_problem(1, list(moved.atoms)))
        self.assert_systems_equal(moved, fresh)
        with self.assertRaisesRegex(ValueError, "error 291"):
            update_wire_slab_geometry(system, [atoms[0], Atom("H", (0.0, 3.0, 3.0))])


class SpaceGroupTests(unittest.TestCase):
    def test_wire_group_is_fortrans(self) -> None:
        # PARSEC (na_chain, R = 12): tetragonal lattice operations 1-4 17-20
        # 25-28 41-44; the D2h operations 1 2 3 4 25 26 27 28.
        lattice = confined_lattice_vectors([[10.0, 0.0, 0.0]])
        group, dropped = find_wire_slab_space_group(lattice, 1, 12.0, NA_CHAIN)
        self.assertEqual(dropped, 0)
        self.assertEqual(group.candidate_numbers, (1, 2, 3, 4, 25, 26, 27, 28))
        self.assertEqual(group.lattice_operations, (1, 2, 3, 4, 17, 18, 19, 20, 25, 26, 27, 28, 41, 42, 43, 44))
        self.assertEqual(group.crystal_system, "tetragonal")
        np.testing.assert_allclose(
            confined_symmetry_lattice(lattice, 1, 12.0), np.diag([10.0, 48.0, 48.0])
        )

    def test_operations_off_the_grid_are_dropped(self) -> None:
        # Two H atoms of a slab whose mid-point is at z = 0.6: PARSEC finds the
        # inversion about it, a translation along z that moves the slab.
        lattice = confined_lattice_vectors([[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]])
        atoms = [Atom("H", (0.3, 0.2, 0.2)), Atom("H", (1.2, 1.0, 1.0))]
        full = find_space_group(confined_symmetry_lattice(lattice, 2, 5.0), atoms)
        self.assertTrue(any(abs(t[2]) > 1e-6 for t in full.translations))
        group, dropped = grid_preserving_operations(full, 2)
        self.assertGreater(dropped, 0)
        self.assertEqual(group.order + dropped, full.order)
        self.assertTrue(all(abs(t[2]) < 1e-6 for t in group.translations))
        self.assertEqual(group.rotations[0], ((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    def test_density_symmetrization_on_a_wire_grid(self) -> None:
        cell = PeriodicCell(confined_lattice_vectors([[10.0, 0.0, 0.0]]))
        settings = PeriodicGridSettings(
            spacing=0.9, shift=(0.5, 0.5, 0.5), expansion_order=8, periodicity=1, confinement_radius=6.0
        )
        grid = build_confined_periodic_grid(cell, settings)
        group, _ = find_wire_slab_space_group(cell.lattice_vectors, 1, 6.0, NA_CHAIN)
        symmetry = confined_grid_symmetry(grid, group)
        self.assertEqual(symmetry.order, 8)
        for images in symmetry.images:
            self.assertEqual(np.unique(images).size, grid.size)
        field = np.random.default_rng(3).random(grid.size)
        averaged = symmetrize_scalar(field, symmetry)
        self.assertAlmostEqual(float(averaged.sum()), float(field.sum()), places=9)
        np.testing.assert_allclose(symmetrize_scalar(averaged, symmetry), averaged, atol=1e-14)
        for images in symmetry.images:
            np.testing.assert_allclose(averaged[images], averaged, atol=1e-14)


class InputTests(unittest.TestCase):
    HEAD = """
Boundary_Conditions: {boundary}
begin Cell_Shape
{cell}
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.6
States_Num: 2
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 0.7
  0.0 0.0 -0.7
end Atom_Coord
"""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def parse(self, boundary="wire", cell=" 3.0", extra=""):
        path = self.directory / "parsec.in"
        path.write_text(self.HEAD.format(boundary=boundary, cell=cell, extra=extra), encoding="utf-8")
        return parse_parsec_input(path)

    def test_relaxation_and_dynamics_are_accepted(self) -> None:
        relaxation = self.parse("slab", " 3.0 0.0\n 0.0 3.2", "Minimization: bfgs\nMovement_Num: 2")
        self.assertEqual(relaxation.relaxation.method, "bfgs")
        dynamics = self.parse("wire", " 3.0", "Molecular_Dynamics: .true.\nStep_num: 2")
        self.assertIsNotNone(dynamics.molecular_dynamics)
        # Molecular dynamics switches PARSEC's symmetry off.
        self.assertEqual(dynamics.problem.space_group.order, 1)

    def test_spin_kpoint_relaxation_and_single_point_forces_are_accepted(self) -> None:
        kpoints = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2\nend Monkhorst_Pack_Grid\nSpin_Polarization: .true.\n"
        # The spin k-point loop starts each structure from the channel
        # potentials of the one before (parity sweep A, test_parity_sweepa).
        relaxation = self.parse("wire", " 3.0", kpoints + "Minimization: bfgs")
        self.assertEqual(relaxation.relaxation.method, "bfgs")
        translation = self.parse("wire", " 3.0", kpoints + "Skip_Force: .false.")
        self.assertTrue(translation.compute_forces)
        weighted = (
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n"
            " 0.0 0.0 0.0 1.0\n 0.25 0.0 0.0 2.0\nend Kpoint_List\nSkip_Force: .false."
        )
        self.assertTrue(self.parse("slab", " 3.0 0.0\n 0.0 3.2", weighted).compute_forces)

    def test_monkhorst_pack_shift_and_symmetry(self) -> None:
        shift = (
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 4\nend Monkhorst_Pack_Grid\n"
            "begin Monkhorst_Pack_Shift\n 0.5\nend Monkhorst_Pack_Shift"
        )
        translation = self.parse("wire", " 3.0", shift)
        problem = translation.problem
        self.assertEqual(problem.monkhorst_pack_shift, (0.5, 0.0, 0.0))
        self.assertTrue(problem.scf.symmetrize_density)
        self.assertGreater(problem.space_group.order, 1)
        extra = self.parse("wire", " 3.0", shift.replace(" 0.5\nend", " 0.5 0.5 0.0\nend"))
        self.assertEqual(extra.problem.monkhorst_pack_shift, (0.5, 0.0, 0.0))
        self.assertTrue(any("Monkhorst_Pack_Shift: the entries beyond" in note for note in extra.warnings))
        # A slab gets no symmetry (parsec.F90: per = 2 -> ierr = 2) ...
        slab = self.parse("slab", " 3.0 0.0\n 0.0 3.2").problem
        self.assertEqual(slab.space_group.order, 1)
        self.assertEqual(slab.space_group.crystal_system, "")
        # ... unless bandstruc turns it on (ierr = 1).
        bands = self.parse(
            "slab",
            " 3.0 0.0\n 0.0 3.2",
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2 2\nend Monkhorst_Pack_Grid\n"
            "begin bandstruc\n2 0 0 0 0.5 0 0 G-X\nend bandstruc",
        ).problem
        self.assertGreater(bands.space_group.order, 1)
        ignored = self.parse("wire", " 3.0", "Ignore_Symmetry: .true.").problem
        self.assertTrue(ignored.space_group.ignored)


class ForceTests(unittest.TestCase):
    def test_spin_forces_of_an_unpolarized_wire_are_the_scalar_ones(self) -> None:
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF import run_scf
        from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
        from parsec_python.SCF.pbc import with_initial_spin_densities
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        atoms = [Atom("H", (0.1, 0.05, 0.72)), Atom("H", (-0.2, 0.0, -0.66))]
        # Gamma: run_spin_scf (both channels equal) against run_scf.
        scalar = prepare_wire_slab_single_point(hydrogen_problem(1, atoms))
        spin = with_initial_spin_densities(
            prepare_wire_slab_single_point(hydrogen_problem(1, atoms, spin=True))
        )
        scalar_forces = compute_forces(scalar, run_scf(scalar))
        spin_forces = compute_forces(spin, run_scf(spin))
        np.testing.assert_allclose(spin_forces.total, scalar_forces.total, atol=2e-6)
        # k-points with unequal weights.
        kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]) * 2.0 * np.pi / 3.0
        weights = np.array([1.0, 2.0]) / 3.0
        scalar = prepare_wire_slab_single_point(hydrogen_problem(1, atoms, kpoints=kpoints, weights=weights))
        spin = prepare_wire_slab_single_point(
            hydrogen_problem(1, atoms, kpoints=kpoints, weights=weights, spin=True)
        )
        scalar_result = run_scf_kpoints(scalar, kpoints, weights)
        spin_result = run_scf_kpoints_spin_polarized(spin, kpoints, weights, keep_wavefunctions=True)
        self.assertIsNotNone(spin_result.scf_hartree_potential)
        np.testing.assert_allclose(
            compute_forces(spin, spin_result).total,
            compute_forces(scalar, scalar_result).total,
            atol=5e-6,
        )

    def test_wire_forces_are_symmetrized_over_its_group(self) -> None:
        from parsec_python.Forces.symmetry import symmetrize_forces, symmetrize_positions
        from parsec_python.Forces.wire_slab import wire_slab_symmetry_operations
        from types import SimpleNamespace

        lattice = confined_lattice_vectors([[10.0, 0.0, 0.0]])
        group, _ = find_wire_slab_space_group(lattice, 1, 12.0, NA_CHAIN)
        system = SimpleNamespace(
            input=SimpleNamespace(space_group=group, periodic_cell=PeriodicCell(lattice)),
            grid=SimpleNamespace(periodicity=1, confinement_radius=12.0),
            atoms=tuple(NA_CHAIN),
        )
        operations = wire_slab_symmetry_operations(system)
        self.assertEqual(len(operations), 8)
        forces = np.random.default_rng(5).normal(size=(4, 3))
        averaged = symmetrize_forces(forces, operations)
        for operation in operations:
            rotated = np.zeros_like(averaged)
            rotated[operation.image] = averaged @ operation.matrix.T
            np.testing.assert_allclose(rotated, averaged, atol=1e-12)
        positions = np.array([atom.position for atom in NA_CHAIN])
        np.testing.assert_allclose(symmetrize_positions(positions, operations), positions, atol=1e-12)


class CommandLineTests(unittest.TestCase):
    def test_slab_relaxation_and_wire_dynamics_run(self) -> None:
        from parsec_python.cli import main

        directory = Path(tempfile.mkdtemp())
        try:
            shutil.copy(HYDROGEN, directory / "H_POTRE.DAT")
            head = InputTests.HEAD.replace(
                "Grid_Spacing: 0.6", "Grid_Spacing: 0.6\nExpansion_Order: 4\nConvergence_Criterion: 1e-6"
            )
            (directory / "parsec.in").write_text(
                head.format(
                    boundary="slab",
                    cell=" 3.0 0.0\n 0.0 3.2",
                    extra="Minimization: bfgs\nMovement_Num: 1\nForce_Min: 1e-4",
                ),
                encoding="utf-8",
            )
            code = main([str(directory / "parsec.in"), "--quiet"])
            self.assertIn(code, (0, 4))
            text = (directory / "parsec.out").read_text()
            self.assertIn("(Ewald forces included)", text)
            self.assertTrue((directory / "atom.cor").is_file())
            (directory / "parsec.in").write_text(
                head.format(
                    boundary="wire",
                    cell=" 3.0",
                    extra="Molecular_Dynamics: .true.\nStep_num: 1\nTime_Step: 20\ntest_rng: .true.",
                ),
                encoding="utf-8",
            )
            self.assertEqual(main([str(directory / "parsec.in"), "--quiet"]), 0)
            self.assertTrue((directory / "md_mech.dat").is_file())
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class PostProcessingTests(unittest.TestCase):
    """Polarizability, parsec.dat/Restart_Run/NSCF, Super_Cell, DOS, LDA+U."""

    HEAD = InputTests.HEAD.replace(
        "Grid_Spacing: 0.6", "Grid_Spacing: 0.6\nExpansion_Order: 4\nConvergence_Criterion: 1e-7"
    )

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def write(self, boundary="slab", cell=" 3.0 0.0\n 0.0 3.2", extra=""):
        path = self.directory / "parsec.in"
        path.write_text(self.HEAD.format(boundary=boundary, cell=cell, extra=extra), encoding="utf-8")
        return path

    def run_cli(self, path) -> int:
        from parsec_python.cli import main

        return main([str(path), "--quiet"])

    def test_polarizability_of_a_wire_and_a_slab_with_kpoints(self) -> None:
        spin_kpoints = (
            "Polarizability: .true.\nSpin_Polarization: .true.\nKpoint_Method: mp\n"
            "begin Monkhorst_Pack_Grid\n 2\nend Monkhorst_Pack_Grid"
        )
        # Spin with k-points is accepted (parity sweep A: the spin k-point
        # loop of every field starts from the potentials of the one before).
        self.assertIsNotNone(
            parse_parsec_input(self.write("wire", " 3.0", spin_kpoints)).polarizability_field
        )
        # PARSEC error 156 stays for a bulk cell.
        bulk = self.directory / "bulk.in"
        bulk.write_text(
            self.HEAD.format(boundary="bulk", cell=" 3.0 0.0 0.0\n 0.0 3.2 0.0\n 0.0 0.0 3.4", extra="Polarizability: .true.")
            .replace("Boundary_Sphere_Radius: 5.0\n", "Periodic_System: .true.\n"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Cannot do polarizability"):
            parse_parsec_input(bulk)
        for boundary, cell, extra in (
            ("wire", " 3.0", ""),
            (
                "slab",
                " 3.0 0.0\n 0.0 3.2",
                "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
                "begin Kpoint_List\n 0.0 0.0 0.0 1.0\n 0.5 0.0 0.0 1.0\nend Kpoint_List",
            ),
        ):
            with self.subTest(boundary=boundary):
                path = self.write(boundary, cell, "Polarizability: .true.\nElectric_Field: 0.005\n" + extra)
                self.assertEqual(self.run_cli(path), 0)
                lines = (self.directory / "polar.dat").read_text().splitlines()
                self.assertTrue(any("From energy:" in line for line in lines))
                # +x and -x energies of a centred molecule agree; the field
                # along the axis of the H2 (z) moves the dipole most.
                runs = [line for line in lines if line[:12].count(".") == 3]
                self.assertEqual(len(runs), 7)

    def test_parsec_dat_restart_and_nscf_of_a_slab(self) -> None:
        from parsec_python.Output.parsec_dat import read_parsec_dat

        kpoints = (
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
            "begin Kpoint_List\n 0.0 0.0 0.0 1.0\n 0.5 0.0 0.0 1.0\nend Kpoint_List\n"
            "Eigensolver: chebdav\nDiag_Tolerance: 1e-10\n"
        )
        self.assertEqual(self.run_cli(self.write(extra=kpoints)), 0)
        data = read_parsec_dat(self.directory / "parsec.dat", wavefunctions=False)
        self.assertEqual(int(data.system_type[2]), 3)  # stype(3): slab
        self.assertEqual(data.rmax, 5.0)
        np.testing.assert_allclose(data.periodic.box_size, [3.0, 3.2, 0.0])
        np.testing.assert_allclose(np.diag(data.symmetry.alatt), [3.0, 3.2, 20.0])
        first = (self.directory / "parsec.out").read_text()
        energy = [line for line in first.splitlines() if "Total Energy =" in line][-1]
        # Restart_Run: the stored potential converges at once.
        restart = kpoints + "Restart_Run: .true.\nFF_Maxiter: 9\nChebdav_Degree: 40\n"
        self.assertEqual(self.run_cli(self.write(extra=restart)), 0)
        second = (self.directory / "parsec.out").read_text()
        self.assertIn("Restarted Run - from parsec.dat", second)
        self.assertLessEqual(second.count("SRE of pot."), 3)
        restarted = [line for line in second.splitlines() if "Total Energy =" in line][-1]
        self.assertAlmostEqual(float(restarted.split()[3]), float(energy.split()[3]), places=6)
        # NSCF at the SCF k-points: their eigenvalues again.
        nscf = restart + (
            "nscf_kpoint_unit: reciprocal_lattice_vectors\nbegin nscf_kpoints\n"
            " 0.0 0.0 0.0 1.0\n 0.5 0.0 0.0 1.0\nend nscf_kpoints\n"
        )
        self.assertEqual(self.run_cli(self.write(extra=nscf)), 0)
        self.assertIn("NSCF calculation done", (self.directory / "parsec.out").read_text())
        with self.assertRaisesRegex(Exception, "confined"):
            parse_parsec_input(
                self.write(extra=nscf.replace(" 0.5 0.0 0.0 1.0\nend nscf", " 0.5 0.0 0.3 1.0\nend nscf"))
            )

    def test_super_cell_of_a_wire_and_a_slab(self) -> None:
        wire = parse_parsec_input(self.write("wire", " 3.0", "begin Super_Cell\n 2\nend Super_Cell"))
        np.testing.assert_allclose(wire.problem.periodic_cell.lattice_vectors, np.diag([6.0, 1.0, 1.0]))
        self.assertEqual(wire.super_cell, (2, 1, 1))
        self.assertEqual(len(wire.problem.atoms), 4)
        # half a cell along the wire: -1/2 after PARSEC 1.5's start-up back_to_cell (fractional coordinates in [-1/2, 1/2))
        np.testing.assert_allclose(wire.problem.atoms[1].position, (-3.0, 0.0, 0.7))
        slab = parse_parsec_input(
            self.write(
                "slab",
                " 3.0 0.0\n 0.0 3.2",
                "begin Super_Cell\n 1 2\nend Super_Cell\nbegin Super_Cell_Vac\n 1 0\nend Super_Cell_Vac",
            )
        )
        np.testing.assert_allclose(
            slab.problem.periodic_cell.lattice_vectors, np.diag([6.0, 6.4, 1.0])
        )
        self.assertEqual(slab.super_cell_vacuum, (1, 0, 0))
        self.assertEqual(len(slab.problem.atoms), 4)
        np.testing.assert_allclose(slab.problem.atoms[1].position, (0.0, -3.2, 0.7))

    def test_dos_of_a_wire(self) -> None:
        from parsec_python.Output.dos import _image_translations

        lattice = confined_lattice_vectors([[3.0, 0.0, 0.0]])
        translations = _image_translations(lattice, np.zeros(3), 2.0, periodicity=1)
        self.assertTrue(np.all(translations[:, 1:] == 0.0))
        self.assertGreater(translations.shape[0], 1)
        path = self.write(
            "wire",
            " 3.0",
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 4\nend Monkhorst_Pack_Grid\n"
            "begin Monkhorst_Pack_Shift\n 0.5\nend Monkhorst_Pack_Shift\n"
            "Create_DOS: .true.\nDOS_Pnum: 200\nYLM_DOS_L: 1",
        )
        self.assertEqual(self.run_cli(path), 0)
        text = (self.directory / "DOS").read_text().splitlines()
        self.assertIn("Tot. DOS:", text[2])
        self.assertTrue(any(line.startswith("LDOS") for line in text))

    def test_ylm_projection_counts_fortrans_confined_images(self) -> None:
        from parsec_python.Output.dos import ylm_projection_weights

        cell = PeriodicCell(confined_lattice_vectors([[3.0, 0.0, 0.0]]))
        settings = PeriodicGridSettings(
            spacing=0.5, shift=(0.5, 0.5, 0.5), expansion_order=4, periodicity=1, confinement_radius=3.0
        )
        grid = build_confined_periodic_grid(cell, settings)
        orbital = np.exp(-np.sum(grid.coordinates**2, axis=1)) * grid.coordinates[:, 2]
        weights = {
            flag: ylm_projection_weights(
                grid, np.zeros((1, 3)), [1.5], [[orbital[:, None]]], np.zeros((1, 3)),
                l_max=1, fortran_projection=flag,
            )[0]
            for flag in (False, True)
        }
        # p channels (indices 1-3): Fortran's ylm_eig counts each point three
        # times per confined direction (box_size 0 there), 81 times the weight.
        np.testing.assert_allclose(weights[True][..., 1:], 81.0 * weights[False][..., 1:], rtol=1e-12)

    def test_lda_plus_u_of_a_wire(self) -> None:
        from parsec_python.Hamiltonian.hubbard import periodic_replica_translations
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point, update_wire_slab_geometry

        translations = periodic_replica_translations(np.array([[3.0, 0.0, 0.0]]))
        np.testing.assert_allclose(translations, [[-3.0, 0, 0], [0, 0, 0], [3.0, 0, 0]])
        bulk = periodic_replica_translations(np.eye(3) * 2.0)
        self.assertEqual(bulk.shape, (27, 3))
        np.testing.assert_allclose(bulk[0], (-2.0, -2.0, -2.0))
        problem = SinglePointInput(
            atoms=(Atom("Ag", (0.1, 0.0, -1.5)), Atom("H", (0.0, 0.2, 1.6))),
            pseudopotentials={
                "Ag": SpeciesPotential(path=DATA / "Ag_POTRE.DAT", local_angular_momentum=0, hubbard_u=0.3),
                "H": SpeciesPotential(path=HYDROGEN, local_angular_momentum=0),
            },
            grid=PeriodicGridSettings(
                spacing=0.7, expansion_order=8, shift=(0.5, 0.5, 0.5), periodicity=1, confinement_radius=6.0
            ),
            periodic_cell=PeriodicCell(confined_lattice_vectors([[9.0, 0.0, 0.0]])),
            scf=SCFSettings(number_of_states=8, xc_functional="ca", max_iterations=2),
        )
        system = prepare_wire_slab_single_point(problem)
        self.assertIsNotNone(system.hubbard)
        moved = update_wire_slab_geometry(system, [Atom("Ag", (0.15, 0.0, -1.5)), problem.atoms[1]])
        fresh = prepare_wire_slab_single_point(
            replace(problem, atoms=(Atom("Ag", (0.15, 0.0, -1.5)), problem.atoms[1]))
        )
        self.assertTrue(_same(moved.hubbard.projectors, fresh.hubbard.projectors))


if __name__ == "__main__":
    unittest.main()
