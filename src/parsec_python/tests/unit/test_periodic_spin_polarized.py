"""Tests for the periodic (Gamma-point) collinear spin-polarized SCF path.

``run_scf_spin_polarized`` is written generically against both prepared-
system types; these tests confirm it actually works unchanged on a
:class:`~parsec_python.SCF.pbc.PeriodicPreparedSinglePointSystem`, and that
the CLI dispatches to it correctly for a real text ``parsec.in``.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.cli import main as cli_main
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _periodic_au_problem(box: float = 8.0) -> SinglePointInput:
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return SinglePointInput(
        atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.3)
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=40,
            number_of_states=10,
            spin_polarized=True,
            convergence_criterion=5.0e-3,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class PeriodicSpinPolarizedLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        system = prepare_periodic_single_point(_periodic_au_problem())
        cls.result = run_scf_spin_polarized(system)

    def test_converges(self) -> None:
        self.assertTrue(self.result.converged)

    def test_reproduces_au_atomic_moment(self) -> None:
        # Au's atomic ground state ([Xe]4f14 5d10 6s1) has one unpaired
        # electron; a large enough box should reproduce this even under
        # periodic (Ewald) boundary conditions.
        self.assertAlmostEqual(self.result.magnetic_moment, 1.0, places=1)

    def test_includes_nonzero_ewald_ion_ion_energy(self) -> None:
        """Distinguishes this from the isolated path: a single periodic
        atom still self-interacts with its own periodic images."""
        self.assertNotEqual(self.result.energies.ion_ion, 0.0)


_BASE_PERIODIC_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Coordinate_Unit: Cartesian_Bohr

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 10
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Spin_Polarization: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
Initial_Spin_Polarization: 0.3
Local_Component: s
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA
"""


class PeriodicSpinPolarizedCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.directory = Path(self._tmpdir.name)
        shutil.copy(_AU_POTRE, self.directory / "Au_POTRE.DAT")

    def _write_input(self, name: str, extra: str = "") -> Path:
        path = self.directory / name
        path.write_text(_BASE_PERIODIC_INPUT + extra, encoding="utf-8")
        return path

    def test_full_periodic_run_converges_via_cli(self) -> None:
        input_path = self._write_input("parsec.in")
        exit_code = cli_main([str(input_path), "--quiet"])
        self.assertEqual(exit_code, 0)

        log_text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Spin-polarized SCF iter", log_text)
        self.assertIn("Self-consistency convergence achieved.", log_text)
        # No SO_PSP species in this input, so no SOC section should appear.
        self.assertNotIn("Perturbative spin-orbit correction", log_text)

        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertTrue(bool(archive["converged"]))
        self.assertAlmostEqual(float(archive["magnetic_moment"]), 1.0, places=1)

    def test_periodic_plus_soc_is_rejected(self) -> None:
        text = _BASE_PERIODIC_INPUT.replace(
            "Atom_Type: Au\n", "Atom_Type: Au\nSO_PSP: .true.\n"
        )
        input_path = self.directory / "parsec.in"
        input_path.write_text(text, encoding="utf-8")
        exit_code = cli_main([str(input_path), "--dry-run"])
        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
