"""The header of parsec.out says what the boundary conditions and the wavefunctions are."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.Output.parsec_output import ParsecTextReporter

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

_COMMON = """\
States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 5
Convergence_Criterion: 5.0e-3
Mixing_Method: Anderson
Eigensolver: chebff
Atom_Types_Num: 1
Total_Atom_Num: 1
Atom_Type: Au
Local_Component: s
Correlation_Type: CA
"""
_CLUSTER = "Boundary_Sphere_Radius: 6.0\nGrid_Spacing: 0.6\nExpansion_Order: 6\n" + _COMMON + (
    "Begin atom_coord\n0.0 0.0 0.0\nEnd atom_coord\n"
)
_PERIODIC = (
    "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n8.0 8.0 8.0\nend Cell_Shape\n"
    "Grid_Spacing: 0.6\nExpansion_Order: 6\n"
    + _COMMON
    + "Begin atom_coord\n4.0 4.0 4.0\nEnd atom_coord\n"
)
_KPOINTS = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"


def _header(text: str) -> str:
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        (directory / "Au_POTRE.DAT").write_bytes(_AU_POTRE.read_bytes())
        (directory / "parsec.in").write_text(text, encoding="utf-8")
        translation = parse_parsec_input(directory / "parsec.in", pseudopotential_directory=directory)
        lines: list[str] = []
        ParsecTextReporter(lines.append, translation).header()
    return "\n".join(lines)


class HeaderBoundaryTests(unittest.TestCase):
    def test_a_cluster_is_a_confined_system_with_real_wavefunctions(self) -> None:
        header = _header(_CLUSTER)
        self.assertIn("Confined system (cluster) with zero boundary condition!", header)
        self.assertIn("Spherical domain shape:", header)
        self.assertIn("WAVEFUNCTIONS ARE REAL!", header)

    def test_a_periodic_cell_lists_its_lattice_vectors(self) -> None:
        header = _header(_PERIODIC)
        self.assertNotIn("Confined system", header)
        self.assertIn("Periodic boundary conditions!", header)
        self.assertIn("System is fully periodic", header)
        self.assertIn("Unit lattice vectors [bohr]", header)
        self.assertIn("8.0000000000    0.0000000000    0.0000000000", header)
        self.assertIn("WAVEFUNCTIONS ARE REAL!", header)  # Gamma only, as in PARSEC

    def test_k_points_make_the_wavefunctions_complex(self) -> None:
        self.assertIn("WAVEFUNCTIONS ARE COMPLEX!", _header(_PERIODIC + _KPOINTS))

    def test_spinors_make_the_wavefunctions_complex(self) -> None:
        for name, text in (
            ("non-collinear cluster", _CLUSTER + "Non_Collinear_magnetism: .true.\n"),
            ("non-collinear periodic", _PERIODIC + "Non_Collinear_magnetism: .true.\n"),
        ):
            with self.subTest(name):
                self.assertIn("WAVEFUNCTIONS ARE COMPLEX!", _header(text))


if __name__ == "__main__":
    unittest.main()
