"""Combinations of features that meet only once the parity streams are merged.

The wire/slab stream (relaxation, spin forces, LDA+U and Create_DOS of wires and
slabs), the force stream (space-group, spin k-point, spin-orbit and double-grid
forces) and the remaining-gaps stream (LDA+U/TS in the k-point and spinor
drivers, spin-orbit restarts, spin-orbit double grid, the DOS restart shift)
each refused or ignored the others' features.  These tests pin what the
merged package does where they meet: the combinations it computes and the
ones it refuses by name.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parsec_python.Input import UnsupportedParsecOptionError, parse_parsec_input
from parsec_python.tests.environment import pin_kpoints_on_the_cpu

DATA = Path(__file__).resolve().parents[1] / "data"
AG = DATA / "Ag_POTRE.DAT"
HYDROGEN = DATA / "water_H_POTRE.DAT"
AUH = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"


class _Directory(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def write(self, text: str) -> Path:
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return path


_AG_WIRE = """\
Boundary_Conditions: wire
begin Cell_Shape
  6.0
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.6
States_Num: 10
Fermi_Temp: 1000.0
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
ldaplusu_u 4.0 eV
begin Atom_Coord
  0.0 0.0 0.0
end Atom_Coord
"""


class WireSlabLdaPlusUTest(_Directory):
    """LDA+U of a wire or slab at Gamma, with k-points and with spin (the
    Bloch projectors replicate along the periodic directions alone since the
    parity sweep A; the merge had refused the k-point and spin forms)."""

    def parse(self, extra: str = ""):
        shutil.copy(AG, self.directory / "Ag_POTRE.DAT")
        return parse_parsec_input(self.write(_AG_WIRE.format(extra=extra)))

    def test_gamma_kpoints_and_spin_are_accepted(self) -> None:
        self.assertTrue(self.parse().problem.pseudopotentials["Ag"].has_hubbard)
        for extra in (
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2\nend Monkhorst_Pack_Grid",
            "Spin_Polarization: .true.",
        ):
            with self.subTest(extra=extra):
                self.assertTrue(self.parse(extra).problem.pseudopotentials["Ag"].has_hubbard)


_H2_SLAB = """\
Boundary_Conditions: slab
begin Cell_Shape
 3.0 0.0
 0.0 3.2
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.6
States_Num: 2
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
 2 2
end Monkhorst_Pack_Grid
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 0.7
  0.0 0.0 -0.7
end Atom_Coord
"""


class WireSlabDosRestartShiftTest(_Directory):
    def test_dos_shift_of_a_restarted_slab_takes_the_wire_slab_group(self) -> None:
        """Create_DOS + Restart_Run: the restarted SCF of a slab runs with
        DOS_MP_Shift (usrinputfile.F90 l.1188-1193); its group comes from the
        wire/slab search, not from the bulk one."""

        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")
        translation = parse_parsec_input(
            self.write(
                _H2_SLAB.format(
                    extra=(
                        "Create_DOS: .true.\nRestart_Run: .true.\n"
                        "begin DOS_MP_Grid\n 4 4\nend DOS_MP_Grid\n"
                        "begin DOS_MP_Shift\n 0.5 0.5\nend DOS_MP_Shift"
                    )
                )
            )
        )
        problem = translation.problem
        self.assertEqual(problem.monkhorst_pack_shift, (0.5, 0.5, 0.0))
        self.assertEqual(problem.monkhorst_pack_dimensions, (4, 4, 1))
        # A slab without bands, spin-orbit or NSCF: PARSEC skips its symmetry.
        self.assertEqual(problem.space_group.order, 1)


_AU_ATOM = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 20
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-5
FF_MaxIter: 9
Chebdav_Degree: 30
Mixing_Param: 0.25
Eigensolver: chebff
SO_from_scratch: .true.
{extra}
Atom_Types_Num: 2
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   1.445 0 0
End atom_coord
Atom_Type: H
Local_Component: s
Begin atom_coord
   -1.445 0 0
End atom_coord
Correlation_Type: CA
"""


class SpinOrbitCombinationTest(_Directory):
    def setUp(self) -> None:
        super().setUp()
        pin_kpoints_on_the_cpu(self)
        for name in ("Au_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(AUH / name, self.directory / name)

    def test_double_grid_forces_are_accepted_and_wfndat_is_refused(self) -> None:
        double_grid = "Double_Grid_Order: 2\n"
        # The spin-orbit double grid of the single point and of its forces
        # (double-grid spin-orbit projector gradients since the parity sweep
        # A, test_parity_sweepa; the spinor Polarizability with forces is
        # tested there too).
        parse_parsec_input(self.write(_AU_ATOM.format(extra=double_grid)))
        self.assertTrue(
            parse_parsec_input(
                self.write(_AU_ATOM.format(extra=double_grid + "Skip_Force: .false."))
            ).compute_forces
        )
        # wfn.dat holds no spinor components: the spin-orbit loops restart
        # from parsec.dat only.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "wfn.dat holds no spinor"):
            parse_parsec_input(
                self.write(_AU_ATOM.format(extra="Restart_Run: .true.\nRestart_From_Wfndat: .true."))
            )

    def test_restart_with_forces_starts_from_the_spin_orbit_file(self) -> None:
        """Restart_Run of a spin-orbit run with forces: the spinor ionic path
        starts its first SCF loop from the potentials of parsec.dat."""

        from parsec_python.cli import main

        path = self.write(_AU_ATOM.format(extra=""))
        self.assertEqual(main([str(path), "--quiet"]), 0)
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            energy = float(archive["energy_total_ry"])
            fresh_iterations = int(archive["iterations"])
        path = self.write(_AU_ATOM.format(extra="Restart_Run: .true.\nSkip_Force: .false."))
        self.assertEqual(main([str(path), "--quiet"]), 0)
        text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn(" Adding spin-orbit contribution to forces", text)
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            self.assertAlmostEqual(float(archive["energy_total_ry"]), energy, places=6)
            self.assertLessEqual(int(archive["iterations"]), 6)
            self.assertLess(int(archive["iterations"]), fresh_iterations)
            self.assertEqual(archive["forces_spin_orbit"].shape, (2, 3))

    def test_spin_orbit_forces_add_the_ts_term(self) -> None:
        """The TS correction of a spinor loop (efvdw.f90 on its density) enters
        the spin-orbit forces as it enters the scalar ones."""

        from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF import prepare_single_point
        from parsec_python.SCF.self_consistent_soc import run_self_consistent_soc

        translation = parse_parsec_input(
            self.write(_AU_ATOM.format(extra="").replace("1.0e-5", "1.0e-4"))
        )
        system = prepare_single_point(translation.problem)
        projectors = tuple(
            build_spin_orbit_projectors(
                system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials
            )
        )
        result = run_self_consistent_soc(system, projectors)
        self.assertIsNone(result.dispersion)
        plain = compute_forces(system, result, symmetrize=False)
        self.assertIsNone(plain.dispersion)
        term = np.array([[0.01, -0.02, 0.03], [-0.004, 0.005, 0.006]])
        with_term = compute_forces(
            system,
            dataclasses.replace(result, dispersion=SimpleNamespace(forces=term)),
            symmetrize=False,
        )
        np.testing.assert_array_equal(with_term.dispersion, term)
        np.testing.assert_allclose(with_term.unsymmetrized - plain.unsymmetrized, term, atol=1e-15)


# ---------------------------------------------------------------------------
# The final sweep: Save_Intermediate_* (part B) with the warm starts of the
# spinor and wire/slab spin k-point loops (part A)
# ---------------------------------------------------------------------------


class IntermediateFilesWithWarmStartsTests(_Directory):
    """parsec.F90 writes parsec.dat after every SCF iteration of every loop;
    the loops of an ionic or field sequence still start from the one before."""

    def run_counting(self, path: Path) -> tuple[int, list[bool], list[int]]:
        import re

        from parsec_python.Output import output_files
        from parsec_python.cli import main

        pin_kpoints_on_the_cpu(self)
        written: list[bool] = []
        original = output_files.RunOutputFiles._write_parsec_dat

        def counting(files, result, grid, problem, *, report):
            written.append(report)
            return original(files, result, grid, problem, report=report)

        output_files.RunOutputFiles._write_parsec_dat = counting
        try:
            code = main([str(path), "--quiet"])
        finally:
            output_files.RunOutputFiles._write_parsec_dat = original
        loops: list[int] = []
        for line in (self.directory / "parsec.out").read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*\d+-\s*(\d+)\s+(up |dn )?.*SRE", line)
            if match and match.group(2) in (None, "up "):
                iteration = int(match.group(1))
                if iteration == 1 or not loops:
                    loops.append(iteration)
                else:
                    loops[-1] = iteration
        return code, written, loops

    def write_file(self, name: str, text: str) -> None:
        (self.directory / name).write_text(text, encoding="utf-8")

    def test_spin_orbit_relaxation(self) -> None:
        for name in ("Au_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(AUH / name, self.directory / name)
        self.write_file("manual.dat", "   1.460 0.0 0.0\n  -1.445 0.0 0.0\n")
        path = self.write(
            _AUH_SO.format(
                extra="Minimization: manual\nMovement_Num: 1\n"
                "Save_Intermediate_Charge_Density: .true."
            )
        )
        code, written, loops = self.run_counting(path)
        self.assertEqual(code, 0)
        self.assertEqual(len(loops), 2)
        self.assertEqual(written.count(False), sum(loops))
        self.assertLess(loops[1], loops[0] - 3)

    def test_spin_orbit_polarizability(self) -> None:
        for name in ("Au_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(AUH / name, self.directory / name)
        path = self.write(
            _AUH_SO.format(
                extra="Polarizability: .true.\nElectric_Field: 0.005\n"
                "Save_Intermediate_Charge_Density: .true."
            ).replace("1.0e-5", "5.0e-3")
        )
        code, written, loops = self.run_counting(path)
        self.assertIn(code, (0, 3))
        self.assertEqual(len(loops), 7)
        self.assertEqual(written.count(False), sum(loops))
        self.assertLess(max(loops[1:]), loops[0])

    def test_wire_polarizability_with_spin_and_kpoints(self) -> None:
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")
        path = self.write(
            _H3_SPIN_WIRE.format(
                extra=(
                    "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
                    "begin Kpoint_List\n 0.125 0.0 0.0 1.0\n 0.375 0.0 0.0 1.0\nend Kpoint_List\n"
                    "Polarizability: .true.\nElectric_Field: 0.005\n"
                    "Save_Intermediate_Charge_Density: .true."
                )
            )
        )
        code, written, loops = self.run_counting(path)
        self.assertEqual(code, 0)
        self.assertEqual(len(loops), 7)
        self.assertEqual(written.count(False), sum(loops))
        self.assertLess(max(loops[1:]), loops[0])


_AUH_SO = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 20
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 1.0e-5
FF_MaxIter: 9
Chebdav_Degree: 30
Mixing_Param: 0.25
Eigensolver: chebff
SO_from_scratch: .true.
{extra}
Atom_Types_Num: 2
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   1.445 0 0
End atom_coord
Atom_Type: H
Local_Component: s
Begin atom_coord
   -1.445 0 0
End atom_coord
Correlation_Type: CA
"""

_H3_SPIN_WIRE = """\
Boundary_Conditions: wire
begin Cell_Shape
 6.0
end Cell_Shape
Boundary_Sphere_Radius: 4.5
Grid_Spacing: 0.6
Coordinate_Unit: Cartesian_Bohr
States_Num: 5
Convergence_Criterion: 1.0e-5
Max_Iter: 80
Spin_Polarization: .true.
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.00   0.10   0.65
   1.40   0.00  -0.75
  -1.90   0.30   0.20
end Atom_Coord
"""


if __name__ == "__main__":
    unittest.main()
