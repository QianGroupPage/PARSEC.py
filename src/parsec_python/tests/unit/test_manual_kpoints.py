"""Explicit k-point input (Fortran ``Kpoint_Method: manual``): helpers, model,
input parsing, and end-to-end equivalence with the Monkhorst-Pack grid."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Grid import (
    explicit_kpoints,
    kpoint_sampling,
    monkhorst_pack_grid,
    read_kpoint_file,
)
from parsec_python.cli import main as cli_main

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)
_BOX = 8.0
_CELL = PeriodicCell(lattice_vectors=_BOX * np.eye(3))


class ExplicitKpointConversionTests(unittest.TestCase):
    def test_cartesian_rows_pass_through_with_equal_weights(self) -> None:
        k, w = explicit_kpoints([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.5]])
        np.testing.assert_allclose(k, [[0.1, 0.2, 0.3], [-0.1, 0.0, 0.5]])
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_weights_are_normalized(self) -> None:
        _, w = explicit_kpoints([[0, 0, 0, 2.0], [0.1, 0, 0, 6.0]])
        np.testing.assert_allclose(w, [0.25, 0.75])

    def test_reciprocal_units_reproduce_the_monkhorst_pack_points(self) -> None:
        mp_k, _ = monkhorst_pack_grid(_CELL, (2, 2, 1))
        fractional = [
            [fx, fy, 0.0]
            for fx in (-0.25, 0.25)
            for fy in (-0.25, 0.25)
        ]
        k, _ = explicit_kpoints(
            fractional, unit="reciprocal_lattice_vectors", cell=_CELL
        )
        np.testing.assert_allclose(np.sort(k, axis=0), np.sort(mp_k, axis=0), atol=1e-12)

    def test_reciprocal_units_satisfy_the_lattice_duality_for_a_skewed_cell(self) -> None:
        lattice = np.array([[8.0, 0.0, 0.0], [1.5, 7.0, 0.0], [0.5, 0.7, 6.0]])
        cell = PeriodicCell.__new__(PeriodicCell)
        object.__setattr__(cell, "lattice_vectors", lattice)
        fractions = np.array([[0.25, -0.1, 0.4]])
        k, _ = explicit_kpoints(fractions, unit="reciprocal_lattice_vectors", cell=cell)
        # a_i . k = 2*pi*f_i
        np.testing.assert_allclose(lattice @ k[0], 2.0 * np.pi * fractions[0], atol=1e-12)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            explicit_kpoints([[0, 0]])
        with self.assertRaises(ValueError):
            explicit_kpoints([[0, 0, 0, -1.0], [0, 0, 1, 2.0]])
        with self.assertRaises(ValueError):
            explicit_kpoints([[0, 0, 0, 0.0]])
        with self.assertRaises(ValueError):
            explicit_kpoints([[0, 0, np.nan]])
        with self.assertRaises(ValueError):
            explicit_kpoints([[0, 0, 0]], unit="angstrom")
        with self.assertRaises(ValueError):
            explicit_kpoints([[0, 0, 0]], unit="reciprocal_lattice_vectors")


class KpointFileTests(unittest.TestCase):
    def _write(self, text: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "kpts.dat"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_rows_comments_and_blank_lines(self) -> None:
        path = self._write(
            "# a comment\n\n0.0 0.0 0.0   ! gamma\n0.1 0.2 0.3\n  \n"
        )
        np.testing.assert_allclose(read_kpoint_file(path), [[0, 0, 0], [0.1, 0.2, 0.3]])

    def test_reads_a_weight_column(self) -> None:
        path = self._write("0 0 0 1\n0.5 0 0 3\n")
        data = read_kpoint_file(path)
        self.assertEqual(data.shape, (2, 4))

    def test_rejects_malformed_files(self) -> None:
        for text in ("", "# only a comment\n", "0 0\n", "0 0 x\n", "0 0 0\n0 0 0 1\n", "0 0 0 1 2\n"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    read_kpoint_file(self._write(text))


def _problem(**kwargs) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("Au", [4.0, 4.0, 4.0])],
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0)},
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=_CELL,
        scf=SCFSettings(),
        **kwargs,
    )


class SinglePointInputKpointTests(unittest.TestCase):
    def test_explicit_list_is_stored_with_equal_default_weights(self) -> None:
        problem = _problem(kpoints=[[0.0, 0, 0], [0.2, 0, 0]])
        self.assertTrue(problem.has_kpoint_sampling)
        np.testing.assert_allclose(problem.kpoint_weights, [0.5, 0.5])
        k, w = kpoint_sampling(problem)
        self.assertEqual(k.shape, (2, 3))

    def test_monkhorst_pack_and_gamma_only_are_unchanged(self) -> None:
        self.assertFalse(_problem().has_kpoint_sampling)
        self.assertIsNone(kpoint_sampling(_problem()))
        problem = _problem(monkhorst_pack_dimensions=(2, 1, 1))
        self.assertTrue(problem.has_kpoint_sampling)
        self.assertEqual(kpoint_sampling(problem)[0].shape, (2, 3))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            _problem(kpoints=[[0, 0, 0]], monkhorst_pack_dimensions=(2, 1, 1))
        with self.assertRaises(ValueError):
            _problem(kpoint_weights=[1.0])
        with self.assertRaises(ValueError):
            _problem(kpoints=[[0, 0, 0], [1, 0, 0]], kpoint_weights=[1.0])
        with self.assertRaises(ValueError):
            _problem(kpoints=[[0, 0]])
        from parsec_python import GridSettings

        with self.assertRaises(ValueError):
            SinglePointInput(
                atoms=[Atom("Au", [0, 0, 0])],
                pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0)},
                grid=GridSettings(spacing=0.6, radius=4.0, expansion_order=6),
                kpoints=[[0, 0, 0]],
            )


_BASE = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 10
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 1.0e-5

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff
{spin}
Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .false.
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA
{kpoints}
"""

_MP = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"
_K = 2.0 * np.pi / _BOX * 0.25


class ManualKpointCliTests(unittest.TestCase):
    def _run(self, kpoints: str, *, spin: str = "", files: dict | None = None) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "Au_POTRE.DAT").write_text(
                _AU_POTRE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            for name, text in (files or {}).items():
                (directory / name).write_text(text, encoding="utf-8")
            (directory / "parsec.in").write_text(
                _BASE.format(spin=spin, kpoints=kpoints), encoding="utf-8"
            )
            code = cli_main([str(directory / "parsec.in"), "--quiet"])
            self.assertEqual(code, 0)
            archive = np.load(directory / "parsec_python_results.npz")
            return {key: archive[key] for key in archive.files}

    def _dry_run_code(self, kpoints: str, files: dict | None = None) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "Au_POTRE.DAT").write_text(
                _AU_POTRE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            for name, text in (files or {}).items():
                (directory / name).write_text(text, encoding="utf-8")
            (directory / "parsec.in").write_text(
                _BASE.format(spin="", kpoints=kpoints), encoding="utf-8"
            )
            return cli_main([str(directory / "parsec.in"), "--dry-run"])

    def test_explicit_list_reproduces_the_monkhorst_pack_result(self) -> None:
        reference = self._run(_MP)
        variants = {
            "cartesian block": (
                f"Kpoint_Method: manual\nbegin Kpoint_List\n{-_K!r} 0 0\n{_K!r} 0 0\n"
                "end Kpoint_List\n"
            ),
            "reciprocal block with weights": (
                "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
                "begin Kpoint_List\n-0.25 0 0 1.0\n0.25 0 0 1.0\nend Kpoint_List\n"
            ),
        }
        for name, text in variants.items():
            with self.subTest(name):
                result = self._run(text)
                np.testing.assert_allclose(
                    result["eigenvalues_ry"], reference["eigenvalues_ry"], atol=1e-8
                )
                self.assertAlmostEqual(
                    float(result["energy_total_ry"]), float(reference["energy_total_ry"]), places=8
                )

    def test_kpoint_file_reproduces_the_monkhorst_pack_result(self) -> None:
        reference = self._run(_MP)
        result = self._run(
            "Kpoint_Method: manual\nKpoint_File: kpts.dat\nKpoint_Unit: reciprocal_lattice_vectors\n",
            files={"kpts.dat": "# shifted pair\n-0.25 0 0\n 0.25 0 0   ! equal weights\n"},
        )
        self.assertAlmostEqual(
            float(result["energy_total_ry"]), float(reference["energy_total_ry"]), places=8
        )

    def test_single_gamma_point_matches_the_gamma_only_path(self) -> None:
        gamma_only = self._run("")
        manual = self._run("Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0\nend Kpoint_List\n")
        self.assertAlmostEqual(
            float(manual["energy_total_ry"]), float(gamma_only["energy_total_ry"]), delta=1e-5
        )

    def test_spin_polarized_run_accepts_an_explicit_list(self) -> None:
        result = self._run(
            "Kpoint_Method: manual\nKpoint_Unit: reciprocal_lattice_vectors\n"
            "begin Kpoint_List\n-0.25 0 0\n0.25 0 0\nend Kpoint_List\n",
            spin="\nSpin_Polarization: .true.\n",
        )
        self.assertTrue(bool(result["converged"]))
        self.assertGreater(float(result["magnetic_moment"]), 0.5)

    def test_input_errors_are_reported(self) -> None:
        cases = {
            "manual without a list": "Kpoint_Method: manual\n",
            "both block and file": (
                "Kpoint_Method: manual\nKpoint_File: kpts.dat\n"
                "begin Kpoint_List\n0 0 0\nend Kpoint_List\n"
            ),
            "missing file": "Kpoint_Method: manual\nKpoint_File: nothere.dat\n",
            "bad unit": (
                "Kpoint_Method: manual\nKpoint_Unit: angstrom\n"
                "begin Kpoint_List\n0 0 0\nend Kpoint_List\n"
            ),
            "bad row": "Kpoint_Method: manual\nbegin Kpoint_List\n0 0\nend Kpoint_List\n",
            "mixed columns": (
                "Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0\n0 0 1 2\nend Kpoint_List\n"
            ),
            "negative weight": (
                "Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0 -1\n0 0 1 2\nend Kpoint_List\n"
            ),
            "unequal weights": (
                "Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0 1\n0 0 1 2\nend Kpoint_List\n"
            ),
            "list without manual": "begin Kpoint_List\n0 0 0\nend Kpoint_List\n",
            "unit without manual": "Kpoint_Unit: reciprocal_lattice_vectors\n",
        }
        files = {"kpts.dat": "0 0 0\n"}
        for name, text in cases.items():
            with self.subTest(name):
                self.assertEqual(self._dry_run_code(text, files), 2, msg=name)

    def test_cluster_input_with_a_kpoint_list_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "Au_POTRE.DAT").write_text(
                _AU_POTRE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (directory / "parsec.in").write_text(
                "Boundary_Sphere_Radius: 5.0\nGrid_Spacing: 0.6\nExpansion_Order: 6\n"
                "States_Num: 8\nAtom_Types_Num: 1\nTotal_Atom_Num: 1\n"
                "Atom_Type: Au\nLocal_Component: s\nBegin atom_coord\n0 0 0\nEnd atom_coord\n"
                "Correlation_Type: CA\nbegin Kpoint_List\n0 0 0\nend Kpoint_List\n",
                encoding="utf-8",
            )
            self.assertEqual(cli_main([str(directory / "parsec.in"), "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
