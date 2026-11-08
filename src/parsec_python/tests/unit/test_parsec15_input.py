"""PARSEC 1.5 keywords of the input, I/O and physics layer (parity round of 2026-11-04).

Fortran references: ``src/io/usrinputfile.F90`` (PARSEC 1.5) and ``io/pseudo.f90``.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from parsec_python.Hartree.poisson import _conjugate_gradient, cg_stopping_threshold
from parsec_python.Input import parsec_input as parser_module
from parsec_python.Input.complete_input import COMPLETE_INPUT_NAME
from parsec_python.Output.complete_input import write_complete_input
from parsec_python.Input.parsec_input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Input.routines import NEUTRAL_KEYS
from parsec_python.models import HartreeSettings

DATA = Path(__file__).resolve().parents[1] / "data"

CLUSTER = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.6 bohr
Expansion_Order: 4
States_Num: {states}
Eigensolver: chebff
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
{species}
begin Atom_Coord
  0.0  0.0  0.0
  0.0  0.0  1.4
end Atom_Coord
Correlation_Type: {correlation}
{extra}
"""

BULK = """\
Boundary_Conditions: bulk
begin Cell_Shape
  6.0  6.0  6.0
end Cell_Shape
Grid_Spacing: 0.6 bohr
Expansion_Order: 4
States_Num: 4
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.0  3.0  3.0
end Atom_Coord
Correlation_Type: ca
{extra}
"""


class _InputDirectory(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def write(self, text: str, name: str = "parsec.in") -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def cluster(self, extra: str = "", *, states: int = 4, correlation: str = "ca", species: str = "", **keywords):
        path = self.write(CLUSTER.format(states=states, correlation=correlation, species=species, extra=extra))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parse_parsec_input(path, **keywords)

    def bulk(self, extra: str = "", **keywords):
        return parse_parsec_input(self.write(BULK.format(extra=extra)), **keywords)


class InertKeywordTests(_InputDirectory):
    def test_tuning_keywords_are_read_and_named_in_one_warning(self) -> None:
        translation = self.cluster(
            "Use_ScaLAPACK: .true.\nScaLAPACK_Block_Size: 8\nUse_ELPA: .false.\nELPA_Block_Size: 8\n"
            "Use_Sfc: .true.\nThread_Random_Init: .true.\nNslices: 1\nNum_Of_Copied_Groups: 2\n"
            "Spectrum_Slicing: .false.\nPolym_SS: 100\nRritz_Logic: none\nTest_Logic: none\n"
            "Max_Taskgroups: 4\nVxc_Matrix_Elements: .false.\nSpectrum_Slicing_Ratio: -1.0\n"
        )
        baseline = self.cluster()
        self.assertEqual(repr(translation.problem), repr(baseline.problem))
        notes = [note for note in translation.warnings if "tune PARSEC 1.5's parallel layout" in note]
        self.assertEqual(len(notes), 1)
        for label in ("Use_ScaLAPACK", "Use_Sfc", "Nslices", "Rritz_Logic", "Spectrum_Slicing", "Max_Taskgroups"):
            self.assertIn(label, notes[0])

    def test_types_and_values_follow_parsec15(self) -> None:
        with self.assertRaisesRegex(ParsecInputError, "Use_ScaLAPACK"):
            self.cluster("Use_ScaLAPACK: maybe")
        with self.assertRaisesRegex(ParsecInputError, "Npcol"):
            self.cluster("Npcol: 2.5")
        # usrinputfile.F90:2768-2784 (ierr 130), case-sensitive
        with self.assertRaisesRegex(ParsecInputError, "ierr 130"):
            self.cluster("Test_Logic: None")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "internal test"):
            self.cluster("Test_Logic: forces")
        # usrinputfile.F90:2667-2735 (ierr 128)
        with self.assertRaisesRegex(ParsecInputError, "ierr 128"):
            self.cluster("Rritz_Logic: fancy")
        self.cluster("Rritz_Logic: dist1")

    def test_npcol_is_the_name_of_mpi_groups(self) -> None:
        with self.assertRaisesRegex(ParsecInputError, "provide one of them"):
            self.cluster("Npcol: 2\nMPI_Groups: 2")
        # usrinputfile.F90:4875-4884: States_Num 4 -> 16 with Npcol 2 and the ScaLAPACK block 8
        translation = self.cluster("Npcol: 2")
        self.assertEqual(translation.problem.scf.number_of_states, 4)
        self.assertTrue(any("16 instead of 4" in note for note in translation.warnings))

    def test_hdf5_and_spectrum_slicing_say_what_runs(self) -> None:
        translation = self.cluster("Use_HDF5: .true.\nSpectrum_Slicing: .true.")
        self.assertTrue(any(note.startswith("Use_HDF5: .true.: this port writes no HDF5") for note in translation.warnings))
        self.assertTrue(any(note.startswith("Spectrum_Slicing: .true.") for note in translation.warnings))

    def test_inert_keywords_do_not_choose_the_routines(self) -> None:
        baseline = self.bulk()
        translation = self.bulk("Use_ScaLAPACK: .false.\nUse_Sfc: .true.\nNpcol: 1\nUse_HDF5: .false.\nUse_LibXC: .false.")
        self.assertEqual(translation.routines, baseline.routines)
        self.assertFalse(translation.routines.fortran)
        accepted = (parser_module._INPUT15_ACCEPTED_GLOBAL | parser_module._INPUT15_SPECIES_KEYS) - set(
            parser_module._INPUT15_ALIASES
        )
        self.assertLessEqual(accepted, NEUTRAL_KEYS)


class RenamedKeywordTests(_InputDirectory):
    MD = "Molecular_Dynamics: .true.\nStep_Num: 2\n"

    def test_mol_rng_seed_is_rng_seed(self) -> None:
        new = self.cluster(self.MD + "Mol_RNG_Seed: 17")
        old = self.cluster(self.MD + "RNG_Seed: 17")
        self.assertEqual(new.molecular_dynamics.rng_seed, 17)
        self.assertEqual(new.molecular_dynamics, old.molecular_dynamics)
        with self.assertRaisesRegex(ParsecInputError, "both are given"):
            self.cluster(self.MD + "Mol_RNG_Seed: 17\nRNG_Seed: 17")


class LibxcTests(_InputDirectory):
    def test_libxc_runs_with_the_library_and_its_ids_are_ignored_without_use_libxc(self) -> None:
        from unittest import mock

        from parsec_python.V_xc import libxc_library

        # Without the library (an empty path): refused with the reason; with it: libxc's functional
        # (tests/unit/test_libxc.py).
        with mock.patch.dict("os.environ", {libxc_library.LIBRARY_VARIABLE: "/nonexistent/libxc.so"}):
            libxc_library._reset_for_tests()
            try:
                with self.assertRaisesRegex(UnsupportedParsecOptionError, "Use_LibXC.*no usable libxc"):
                    self.cluster("Use_LibXC: .true.\nFunc_Id_X: 1\nFunc_Id_C: 9")
            finally:
                libxc_library._reset_for_tests()
        if libxc_library.libxc_available():
            translation = self.cluster("Use_LibXC: .true.\nFunc_Id_X: 1\nFunc_Id_C: 9")
            self.assertEqual(translation.problem.scf.xc_functional, "libxc:lda_x+lda_c_pz")
        translation = self.cluster("Use_LibXC: .false.\nFunc_Id_X: 101\nFunc_Id_C: 130")
        self.assertEqual(translation.problem.scf.xc_functional, "ca")
        self.assertTrue(any(note.startswith("Func_Id_X and Func_Id_C") for note in translation.warnings))
        with self.assertRaisesRegex(ParsecInputError, "Func_Id_X"):
            self.cluster("Func_Id_X: one")


class CorrelationTests(_InputDirectory):
    def test_wb_is_this_ports_pbe(self) -> None:
        wb = self.cluster(correlation="wb")
        pbe = self.cluster(correlation="pbe")
        self.assertEqual(wb.problem.scf.xc_functional, "pbe")
        self.assertEqual(repr(wb.problem), repr(pbe.problem))
        self.assertTrue(any(note.startswith("Correlation_Type: wb") for note in wb.warnings))
        # usrinputfile.F90:4100: the TS correction wants icorr == 'pb' (ierr 158)
        with self.assertRaisesRegex(ParsecInputError, "error 158"):
            self.cluster("Van_Der_Waals: .true.\nIgnore_Symmetry: .true.", correlation="wb")

    def test_a_ca_file_in_a_pbe_run_is_used_with_a_warning(self) -> None:
        # pseudo.f90:482-492 warns; the Tier_xc GGA inputs of 1.5 are such runs
        translation = self.cluster(correlation="pbe")
        self.assertTrue(any("was generated for 'ca', not for PBE" in note for note in translation.warnings))
        from parsec_python.V_ion import load_pseudopotentials

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            potentials = load_pseudopotentials(translation.problem.pseudopotentials, xc_functional="pbe")
        self.assertIn("H", potentials)
        self.assertTrue(any("correlation potential does not" in str(item.message) for item in caught))
        # a PBE file in a CA run stays refused
        shutil.copy(DATA / "spin_pbe_H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        with self.assertRaisesRegex(ValueError, "not the requested CA"):
            load_pseudopotentials(self.cluster().problem.pseudopotentials, xc_functional="ca")


class HartreeCGTests(_InputDirectory):
    def test_without_the_keywords_the_cg_is_this_ports(self) -> None:
        hartree = self.cluster().problem.hartree
        self.assertEqual((hartree.relative_tolerance, hartree.max_iterations, hartree.cg_criterion), (1.0e-7, 1600, 1))
        self.assertNotIn("cg_criterion", repr(hartree))

    def test_either_keyword_selects_parsec15s_cg(self) -> None:
        hartree = self.cluster("Hartree_CG_Tolerance: 1.0d-10").problem.hartree
        self.assertEqual(
            (hartree.relative_tolerance, hartree.absolute_tolerance, hartree.max_iterations, hartree.cg_criterion),
            (1.0e-10, 1.0e-13, 2400, 1),
        )
        hartree = self.cluster("Hartree_CG_Criterion: 2").problem.hartree
        self.assertEqual((hartree.relative_tolerance, hartree.max_iterations, hartree.cg_criterion), (1.0e-13, 2400, 2))
        self.assertIn("cg_criterion=2", repr(hartree))
        # usrinputfile.F90:2838: anything but 2 is the test against the initial residual
        self.assertEqual(self.cluster("Hartree_CG_Criterion: 7").problem.hartree.cg_criterion, 1)
        from parsec_python.Input.parsec_input import summarize_translation

        self.assertIn("Hartree CG: PARSEC 1.5's", summarize_translation(self.cluster("Hartree_CG_Criterion: 2")))
        self.assertNotIn("Hartree CG:", summarize_translation(self.cluster()))
        with self.assertRaisesRegex(ParsecInputError, "must be positive"):
            self.cluster("Hartree_CG_Tolerance: 0")

    def test_criterion_two_stops_against_the_right_hand_side(self) -> None:
        settings = HartreeSettings(relative_tolerance=1.0e-3, absolute_tolerance=0.0, cg_criterion=2)
        self.assertEqual(cg_stopping_threshold(settings, 1.0, 10.0), 1.0e-2)
        self.assertEqual(cg_stopping_threshold(HartreeSettings(relative_tolerance=1.0e-3, absolute_tolerance=0.0), 1.0), 1.0e-3)
        with self.assertRaises(ValueError):
            cg_stopping_threshold(settings, 1.0)
        # a warm start: ||r0|| << ||b||, so the test against ||b|| stops earlier
        size = 60
        operator = sp.diags([-np.ones(size - 1), 2.0 * np.ones(size), -np.ones(size - 1)], [-1, 0, 1], format="csr")
        exact = np.sin(np.linspace(0.0, 3.0, size))
        rhs = operator @ exact
        start = exact + 1.0e-3 * np.cos(np.arange(size))
        first = _conjugate_gradient(operator, rhs, start, HartreeSettings(relative_tolerance=1.0e-6, absolute_tolerance=0.0))
        second = _conjugate_gradient(
            operator, rhs, start, HartreeSettings(relative_tolerance=1.0e-6, absolute_tolerance=0.0, cg_criterion=2)
        )
        self.assertTrue(first[1] and second[1])
        self.assertLess(second[2], first[2])
        self.assertLessEqual(second[4], 1.0e-6 * np.linalg.norm(rhs) * 1.0001)

    def test_criterion_two_of_a_cluster_runs_on_the_reference_cli(self) -> None:
        try:
            from parsec_python.acceleration.cli import _reference_only_features
        except ImportError as error:  # pragma: no cover - acceleration package unavailable
            self.skipTest(str(error))
        name = "the Poisson CG test against the right-hand side (Hartree_CG_Criterion: 2) of a cluster"
        self.assertIn(name, _reference_only_features(self.cluster("Hartree_CG_Criterion: 2"), accelerated_backend_selected=True))
        self.assertNotIn(name, _reference_only_features(self.cluster("Hartree_CG_Tolerance: 1e-9"), accelerated_backend_selected=True))


class PeriodicSystemTests(_InputDirectory):
    def test_it_is_inferred_from_the_boundary_conditions(self) -> None:
        inferred = self.bulk()
        given = self.bulk("Periodic_System: .true.")
        self.assertIsNotNone(inferred.problem.periodic_cell)
        self.assertEqual(repr(inferred.problem), repr(given.problem))
        with self.assertRaisesRegex(ParsecInputError, "disagree"):
            self.bulk("Periodic_System: .false.")

    SLAB = (
        "Boundary_Conditions: slab\nbegin Cell_Shape\n  5.0  0.0  0.0\n  0.0  5.0  0.0\nend Cell_Shape\n"
        "Boundary_Sphere_Radius: 5.0\nGrid_Spacing: 0.45\nExpansion_Order: 6\nCorrelation_Type: ca\nStates_Num: 4\n"
        "Atom_Types_Num: 1\nAtom_Type: H\nLocal_Component: s\n"
        "begin Atom_Coord\n  0.10  -0.10   0.65\n -0.40  -0.20  -0.75\nend Atom_Coord\n"
    )
    WIRE = (
        "Boundary_Conditions: wire\nbegin Cell_Shape\n  5.0\nend Cell_Shape\nBoundary_Sphere_Radius: 5.0\n"
        "Grid_Spacing: 0.45\nExpansion_Order: 6\nCorrelation_Type: ca\nStates_Num: 4\n"
        "Atom_Types_Num: 1\nAtom_Type: H\nLocal_Component: s\n"
        "begin Atom_Coord\n  0.10  -0.10   0.65\n -0.40  -0.20  -0.75\nend Atom_Coord\n"
    )

    def routines_of(self, text: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            routines = parse_parsec_input(self.write(text)).routines
        return routines.fortran, routines.reasons, routines.defaults

    def test_the_routines_of_existing_inputs_do_not_change(self) -> None:
        # The routines before the inference of Periodic_System: a slab reads the line itself,
        # false without it, for the routine decision.
        without_line = "a slab without Periodic_System: .true."
        expected = {
            # (boundary, Periodic_System line, Use_Fortran_Routines) -> (fortran, reasons, defaults)
            ("slab", None, None): (True, (), ("slabs",)),
            ("slab", ".true.", None): (True, (), ("slabs",)),
            ("slab", ".false.", None): (True, (), ("slabs",)),
            ("slab", None, ".false."): (True, (without_line,), ()),
            ("slab", ".false.", ".false."): (True, (without_line,), ()),
            ("slab", ".true.", ".false."): (False, (), ()),
            ("slab", None, ".true."): (True, (), ()),
            ("wire", None, None): (True, ("Boundary_Conditions: wire",), ()),
            ("wire", ".true.", ".false."): (True, ("Boundary_Conditions: wire",), ()),
            ("wire", None, ".true."): (True, (), ()),
        }
        templates = {"slab": self.SLAB, "wire": self.WIRE}
        for (boundary, line, requested), result in expected.items():
            with self.subTest(boundary=boundary, line=line, requested=requested):
                text = templates[boundary]
                if line is not None:
                    text += f"Periodic_System: {line}\n"
                if requested is not None:
                    text += f"Use_Fortran_Routines: {requested}\n"
                self.assertEqual(self.routines_of(text), result)
        # a bulk cell without the line (refused before) runs as with Periodic_System: .true.
        for requested in ("", "Use_Fortran_Routines: .false.\n", "Use_Fortran_Routines: .true.\n"):
            with self.subTest(bulk=requested):
                self.assertEqual(
                    self.routines_of(BULK.format(extra=requested)),
                    self.routines_of(BULK.format(extra=requested + "Periodic_System: .true.\n")),
                )


class PseudopotentialFileTests(_InputDirectory):
    def test_pseudopot_file_names_the_file_of_the_species(self) -> None:
        (self.directory / "pp").mkdir()
        shutil.move(self.directory / "H_POTRE.DAT", self.directory / "pp" / "hydrogen.dat")
        translation = self.cluster(species="Pseudopot_File: pp/hydrogen.dat")
        self.assertEqual(translation.problem.pseudopotentials["H"].path, (self.directory / "pp" / "hydrogen.dat").resolve())
        with self.assertRaisesRegex(ParsecInputError, "not found"):
            self.cluster(species="Pseudopot_File: pp/missing.dat")
        absolute = self.cluster(species=f"Pseudopot_File: '{(self.directory / 'pp' / 'hydrogen.dat').resolve()}'")
        self.assertEqual(absolute.problem.pseudopotentials["H"].path, translation.problem.pseudopotentials["H"].path)

    def test_the_nth_line_goes_to_the_nth_species(self) -> None:
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "Li_POTRE.DAT")
        text = CLUSTER.format(states=4, correlation="ca", species="", extra="").replace(
            "Atom_Types_Num: 1", "Atom_Types_Num: 2"
        )
        text += "Atom_Type: H_b\nElement_Symbol: H\nLocal_Component: s\nPseudopot_File: H_POTRE.DAT\nbegin Atom_Coord\n 0 2 0\nend Atom_Coord\n"
        # only the second species has a line: ESDF would give it to the first
        with self.assertRaisesRegex(ParsecInputError, "Pseudopot_File"):
            parse_parsec_input(self.write(text))


class MinimumStatesTests(_InputDirectory):
    def test_fewer_states_than_parsec15_wants_give_a_warning(self) -> None:
        # H2: N_val = 2, PARSEC wants nstate >= N_val/2 + 2 = 3 (pseudo.f90:886-896, ierr 213)
        few = self.cluster(states=2)
        self.assertTrue(any(note.startswith("States_Num=2 is below PARSEC 1.5's minimum") for note in few.warnings))
        self.assertEqual(few.problem.scf.number_of_states, 2)
        self.assertFalse(any("PARSEC 1.5's minimum" in note for note in self.cluster(states=3).warnings))


class OutputKeywordTests(_InputDirectory):
    def test_ascii_dumps_are_refused_by_name(self) -> None:
        for label in ("Output_Ascii_Data", "Print_Unconverged_Density"):
            with self.subTest(label=label):
                with self.assertRaisesRegex(UnsupportedParsecOptionError, label):
                    self.cluster(f"{label}: .true.")
                self.cluster(f"{label}: .false.")

    def test_complete_input_reads_back_to_the_same_problem(self) -> None:
        for maker, extra in (
            (self.cluster, "Generate_Complete_Parsec_In: .true.\nMixing_Param: 0.25\nHartree_CG_Criterion: 2"),
            (self.bulk, "Generate_Complete_Parsec_In: .true.\nbegin Super_Cell\n 1 1 2\nend Super_Cell"),
        ):
            with self.subTest(maker=maker.__name__):
                translation = maker(extra)
                self.assertIsNotNone(translation.complete_input)
                lines = write_complete_input(translation)
                path = self.directory / COMPLETE_INPUT_NAME
                self.assertTrue(path.is_file())
                self.assertIn(str(path), lines[0])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    again = parse_parsec_input(path)
                self.assertEqual(repr(again.problem), repr(translation.problem))
                self.assertEqual(again.routines.fortran, translation.routines.fortran)
                self.assertIsNone(again.complete_input)
        self.assertIsNone(self.cluster().complete_input)
        self.assertEqual(write_complete_input(self.cluster()), [])

    def read_back(self, translation):
        write_complete_input(translation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parse_parsec_input(self.directory / COMPLETE_INPUT_NAME)

    def test_complete_input_with_a_super_cell_keeps_every_kpoint(self) -> None:
        # The reader converts lattice-unit k-point lists with the input cell where a Super_Cell is given; the bak file
        # holds the expanded cell, so it writes those lists in 1/bohr.
        (self.directory / "kpoints.dat").write_text("0.25 0.0 0.0\n0.0 0.5 0.0\n", encoding="utf-8")
        generate = "Generate_Complete_Parsec_In: .true.\n"
        double_x = "begin Super_Cell\n 2 1 1\nend Super_Cell\n"
        vacuum_z = "begin Super_Cell_Vac\n 0 0 1\nend Super_Cell_Vac\n"
        lattice_list = (
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
            "begin Kpoint_List\n 0.25 0.0 0.0 1.0\n 0.0 0.25 0.5 3.0\nend Kpoint_List\n"
        )
        nscf = (
            "Restart_Run: .true.\nnscf_kpoint_unit: reciprocal_lattice_vectors\n"
            "begin nscf_kpoints\n 0.25 0.0 0.0 1.0\n 0.0 0.0 0.5 2.0\nend nscf_kpoints\n"
        )
        cases = {
            "Kpoint_List, Super_Cell": double_x + lattice_list,
            "Kpoint_List, Super_Cell_Vac": vacuum_z + lattice_list,
            "Kpoint_List in 1/bohr": double_x + "Kpoint_Method: manual\nbegin Kpoint_List\n 0.25 0.0 0.0 1.0\nend Kpoint_List\n",
            "Kpoint_File": double_x
            + "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nKpoint_File: kpoints.dat\n",
            "Monkhorst-Pack grid and shift": double_x
            + "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2 2 1\nend Monkhorst_Pack_Grid\n"
            "begin Monkhorst_Pack_Shift\n 0.5 0.0 0.0\nend Monkhorst_Pack_Shift\n",
            "bandstruc": double_x + "Bandstruc_Points: 30\nbegin bandstruc\n 1 0 0 0 0.5 0 0 G X\nend bandstruc\n",
            "nscf_kpoints": double_x + nscf,
            "nscf_kpoints, Super_Cell_Vac": vacuum_z + nscf,
            "DOS": double_x
            + "Restart_Run: .true.\nCreate_DOS: .true.\nbegin dos_mp_grid\n 2 2 2\nend dos_mp_grid\n"
            "begin dos_mp_shift\n 0.5 0.5 0.5\nend dos_mp_shift\n",
        }
        for name, extra in cases.items():
            with self.subTest(case=name):
                translation = self.bulk(generate + extra)
                again = self.read_back(translation)
                self.assertEqual(repr(again.problem), repr(translation.problem))
                self.assertEqual(again.routines.fortran, translation.routines.fortran)
                for attribute in ("nscf", "dos", "super_cell"):
                    first, second = getattr(translation, attribute), getattr(again, attribute)
                    if attribute == "super_cell":
                        self.assertIsNone(second)  # expanded
                        continue
                    self.assertEqual(first is None, second is None)
                    if first is not None and attribute == "nscf":
                        np.testing.assert_array_equal(second.k_points, first.k_points)
                        np.testing.assert_array_equal(second.weights, first.weights)
                    elif first is not None:
                        self.assertEqual(repr(second), repr(first))
        # the doubled direction keeps its k (it was halved before the fix)
        translation = self.bulk(generate + double_x + lattice_list)
        self.assertAlmostEqual(float(translation.problem.kpoints[0, 0]), 2.0 * np.pi * 0.25 / 6.0, places=12)
        self.assertIn("Kpoint_Unit  cartesian_inverse_bohr", translation.complete_input)

    def test_complete_input_paths_are_relative_and_comment_characters_are_named(self) -> None:
        # ESDF (esdf_mod.f90) ends a line at '#', ';' or '!' with no quoting
        for folder in ("dir with space", "dir#hash", "dir;semi", "dir!bang"):
            with self.subTest(folder=folder):
                directory = self.directory / folder
                directory.mkdir()
                shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
                path = directory / "parsec.in"
                path.write_text(
                    CLUSTER.format(states=4, correlation="ca", species="", extra="Generate_Complete_Parsec_In: .true."),
                    encoding="utf-8",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    translation = parse_parsec_input(path)
                self.assertIn("Pseudopot_File  H_POTRE.DAT\n", translation.complete_input)
                self.assertNotIn("WARNING", translation.complete_input)
                write_complete_input(translation)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    again = parse_parsec_input(directory / COMPLETE_INPUT_NAME)
                self.assertEqual(repr(again.problem), repr(translation.problem))
        # a file whose own name holds one cannot be read back: the header says so, and so does parsec.out
        directory = self.directory / "plain"
        directory.mkdir()
        shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
        shutil.copy(DATA / "H_POTRE.DAT", directory / "h#1.dat")
        path = directory / "parsec.in"
        path.write_text(
            CLUSTER.format(states=4, correlation="ca", species="", extra="Generate_Complete_Parsec_In: .true."),
            encoding="utf-8",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            translation = parse_parsec_input(path)  # the input names the file by its default name
        self.assertIn("Pseudopot_File  H_POTRE.DAT", translation.complete_input)
        from dataclasses import replace

        species = translation.problem.pseudopotentials["H"]
        moved = replace(translation.problem, pseudopotentials={"H": replace(species, path=directory / "h#1.dat")})
        from parsec_python.Input.complete_input import complete_input_text

        text = complete_input_text(replace(translation, problem=moved), {}, {})
        self.assertIn("# WARNING: this file does not read back as it is: Pseudopot_File of H h#1.dat", text)
        lines = write_complete_input(replace(translation, complete_input=text))
        self.assertTrue(any(line.startswith(" WARNING: this file does not read back") for line in lines))

    def test_fermi_temperature_in_an_esdf_energy_unit(self) -> None:
        # esdf_mod.f90: k is an energy (1.38066e-23 J); 1.5 prints 580.226123737922 K for 0.05 eV
        translation = self.cluster("Fermi_Temp: 0.05 eV")
        self.assertAlmostEqual(translation.problem.scf.fermi_temperature_kelvin, 580.226123737922, places=9)


class EsdfLenientTests(_InputDirectory):
    def test_unknown_and_repeated_labels(self) -> None:
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "boundary_mask"):
            self.cluster("boundary_mask false")
        with self.assertRaisesRegex(ParsecInputError, "duplicate max_iter"):
            self.cluster("Max_Iter: 30\nMax_Iter: 40")
        lenient = self.cluster("boundary_mask false\nnscf_kpoints (block)\nMax_Iter: 30\nMax_Iter: 40", esdf_lenient=True)
        self.assertEqual(lenient.problem.scf.max_iterations, 30)
        self.assertTrue(any("boundary_mask, nscf_kpoints are not keywords" in note for note in lenient.warnings))
        self.assertTrue(any("max_iter is given on lines" in note for note in lenient.warnings))

    def test_command_lines_take_the_switch(self) -> None:
        import contextlib
        import io

        from parsec_python.cli import main as reference_main

        path = self.write(CLUSTER.format(states=4, correlation="ca", species="", extra="boundary_mask false"))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(reference_main([str(path), "--dry-run"]), 2)
            self.assertEqual(reference_main([str(path), "--dry-run", "--esdf-lenient"]), 0)
        try:
            from parsec_python.acceleration.cli import _build_parser, _reference_argv
        except ImportError as error:  # pragma: no cover - acceleration package unavailable
            self.skipTest(str(error))
        arguments = _build_parser().parse_args([str(path), "--esdf-lenient"])
        self.assertIn("--esdf-lenient", _reference_argv(arguments))
        self.assertNotIn("--esdf-lenient", _reference_argv(_build_parser().parse_args([str(path)])))


class Parsec15BakFileTests(_InputDirectory):
    """What PARSEC 1.5 writes into its own bak_complete_parsec.in (usrinputfile.F90:5477-6042), read as 1.5 reads it."""

    EMPTY_BLOCKS = (
        "begin super_cell\nend super_cell\nbegin super_cell_vac\nend super_cell_vac\n"
        "begin bandstruc\nend bandstruc\nbegin nscf_kpoints\nend nscf_kpoints\n"
        "begin dos_mp_grid\nend dos_mp_grid\nbegin dos_mp_shift\nend dos_mp_shift\n"
        "begin nscf_kgrid\nend nscf_kgrid\nbegin nscf_kgrid_shift\nend nscf_kgrid_shift\n"
        "begin initial_ncl_moment\nend initial_ncl_moment\n"
    )

    def test_empty_blocks_and_kpoint_unit(self) -> None:
        cluster = self.cluster()
        written = self.cluster(
            self.EMPTY_BLOCKS
            + "begin cell_shape\nend cell_shape\nbegin kpoint_list\nend kpoint_list\n"
            "begin monkhorst_pack_grid\nend monkhorst_pack_grid\nbegin monkhorst_pack_shift\nend monkhorst_pack_shift\n"
            "kpoint_method none\nkpoint_unit cartesian_inverse_bohr\n"
        )
        self.assertEqual(repr(written.problem), repr(cluster.problem))
        self.assertEqual(written.routines, cluster.routines)
        self.assertTrue(any("are read as absent, as 1.5 does" in note and "ierr 158/160" in note for note in written.warnings))
        self.assertTrue(any(note.startswith("Kpoint_Unit is used for a Kpoint_List only") for note in written.warnings))
        mp = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2 2 2\nend Monkhorst_Pack_Grid\n"
        bulk = self.bulk(mp)
        again = self.bulk(mp + self.EMPTY_BLOCKS + "begin kpoint_list\nend kpoint_list\nKpoint_Unit: cartesian_inverse_bohr\n")
        self.assertEqual(repr(again.problem), repr(bulk.problem))
        self.assertEqual(again.routines, bulk.routines)
        # empty blocks 1.5 reads for the run stay refused (1.5: ierr 114 and 102)
        with self.assertRaisesRegex(ParsecInputError, "Monkhorst_Pack_Grid"):
            self.bulk("Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\nend Monkhorst_Pack_Grid\n")
        with self.assertRaisesRegex(ParsecInputError, "Kpoint_List"):
            self.bulk("Kpoint_Method: manual\nbegin Kpoint_List\nend Kpoint_List\n")
        # a Kpoint_Unit with a manual list keeps its meaning
        manual = "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\nbegin Kpoint_List\n 0.25 0 0\nend Kpoint_List\n"
        self.assertAlmostEqual(float(self.bulk(manual).problem.kpoints[0, 0]), 2.0 * np.pi * 0.25 / 6.0, places=12)

    def write_two_species(self, first: str, second: str) -> Path:
        shutil.copy(DATA / "Si_POTR.DAT", self.directory / "Si_POTR.DAT")
        text = (
            "Boundary_Conditions: cluster\nBoundary_Sphere_Radius: 6.0 bohr\nGrid_Spacing: 0.8 bohr\n"
            "Expansion_Order: 4\nStates_Num: 6\nCorrelation_Type: ca\nAtom_Types_Num: 2\n" + first + second
        )
        return self.write(text)

    SILICON = (
        "Atom_Type: Si\nPseudopotential_Format: martins_old\nLocal_Component: p\nPotential_Num: 2\n"
        "Core_Cutoff_Radius: 2.5\nbegin Electron_Per_Orbital\n2 2\nend Electron_Per_Orbital\n"
        "Nonlinear_Core_Correction: F\nbegin Atom_Coord\n 0 0 0\nend Atom_Coord\n"
    )
    HYDROGEN_15 = (
        # what 1.5 writes for a martins_new species: its unread labels with zeros (or garbage) and an empty block
        "Atom_Type: H\nPseudopotential_Format: martins_new\nLocal_Component: s\nPotential_Num: 0\n"
        "Core_Cutoff_Radius: 6.67e-310\nbegin Electron_Per_Orbital\nend Electron_Per_Orbital\n"
        "Nonlinear_Core_Correction: T\nbegin Atom_Coord\n 0 0 2.8\nend Atom_Coord\n"
    )
    HYDROGEN = "Atom_Type: H\nLocal_Component: s\nbegin Atom_Coord\n 0 0 2.8\nend Atom_Coord\n"

    def test_format_labels_of_every_species(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = parse_parsec_input(self.write_two_species(self.SILICON.replace("Nonlinear_Core_Correction: F\n", ""), self.HYDROGEN))
            written = parse_parsec_input(self.write_two_species(self.SILICON, self.HYDROGEN_15))
        self.assertEqual(repr(written.problem), repr(plain.problem))
        self.assertTrue(any("one line under every Atom_Type" in note for note in written.warnings))
        # the martins_new species first: 1.5 would give Si the lines of H (Potential_Num 0, ierr 138): refused
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ParsecInputError, "Potential_Num occurs 2 times"):
                parse_parsec_input(self.write_two_species(self.HYDROGEN_15, self.SILICON))


if __name__ == "__main__":
    unittest.main()
