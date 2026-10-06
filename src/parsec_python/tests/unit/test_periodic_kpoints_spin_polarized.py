"""Tests for k-point-sampled periodic collinear spin polarization (no
spin-orbit), LDA and PBE."""

from __future__ import annotations

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
from parsec_python.Grid import monkhorst_pack_grid
from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized
from parsec_python.cli import main as cli_main

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
_AU_POTRE = _EXAMPLES / "Au_POTRE.DAT"
_H_POTRE = _EXAMPLES / "H_POTRE.DAT"


def _au_problem(
    *,
    potre: Path = _AU_POTRE,
    spin_polarized: bool = True,
    xc_functional: str = "ca",
    initial_polarization: float = 0.3,
    convergence: float = 5.0e-3,
    max_iterations: int = 80,
) -> SinglePointInput:
    box = 8.0
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return SinglePointInput(
        atoms=[Atom("Au", [box / 2, box / 2, box / 2])],
        pseudopotentials={
            "Au": SpeciesPotential(
                potre, 0, initial_spin_polarization=initial_polarization
            )
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=max_iterations,
            number_of_states=16,
            convergence_criterion=convergence,
            spin_polarized=spin_polarized,
            xc_functional=xc_functional,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _auh_problem(*, spin_polarized: bool) -> SinglePointInput:
    """Closed-shell AuH: its paramagnetic state is stable."""
    box = 9.0
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return SinglePointInput(
        atoms=[
            Atom("Au", [6.0, box / 2, box / 2]),
            Atom("H", [6.0 - 2.89, box / 2, box / 2]),
        ],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.0),
            "H": SpeciesPotential(_H_POTRE, 0),
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=120,
            number_of_states=16 if spin_polarized else 10,
            convergence_criterion=1.0e-4,
            spin_polarized=spin_polarized,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _relabelled_potre(directory: Path, label: str) -> Path:
    lines = _AU_POTRE.read_text(encoding="utf-8").splitlines()
    tokens = lines[0].split()
    tokens[1] = label
    lines[0] = " " + " ".join(tokens) + "  "
    path = directory / "Au_POTRE.DAT"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class KPointSpinPolarizedSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system = prepare_periodic_single_point(_au_problem())
        cls.cell = cls.system.input.periodic_cell

    def test_gamma_point_matches_the_gamma_only_spin_driver(self) -> None:
        """One k-point at Gamma must reproduce run_scf_spin_polarized, an
        independent implementation of the same physics."""
        reference = run_scf_spin_polarized(self.system)
        k_result = run_scf_kpoints_spin_polarized(
            self.system, np.zeros((1, 3)), np.ones(1)
        )
        self.assertTrue(reference.converged and k_result.converged)
        self.assertAlmostEqual(k_result.magnetic_moment, reference.magnetic_moment, delta=5e-3)
        self.assertAlmostEqual(k_result.energies.total, reference.energies.total, delta=5e-3)
        self.assertGreater(k_result.magnetic_moment, 0.5)

    def test_monkhorst_pack_grid_conserves_electrons_and_moment(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        result = run_scf_kpoints_spin_polarized(self.system, k_points, weights)
        self.assertTrue(result.converged)
        total = np.sum(result.occupations_up) + np.sum(result.occupations_down)
        self.assertAlmostEqual(
            float(total) * result.occupation_weight, result.electron_count, places=4
        )
        self.assertAlmostEqual(result.occupation_weight, 0.5)
        # Pooled over 2 k-points x 16 states per spin.
        self.assertEqual(result.eigenvalues_up.size, 32)
        self.assertEqual(result.eigenvalues_down.size, 32)
        self.assertGreater(result.magnetic_moment, 0.5)
        self.assertLess(result.magnetic_moment, 1.5)
        self.assertTrue(np.isfinite(result.energies.total))

    def test_moment_equals_integrated_spin_density(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        result = run_scf_kpoints_spin_polarized(self.system, k_points, weights)
        integrated = self.system.grid.volume_element * float(
            np.sum(result.density_up - result.density_down)
        )
        self.assertAlmostEqual(integrated, result.magnetic_moment, delta=1e-3)

    def test_closed_shell_reproduces_the_unpolarized_kpoint_driver(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        polarized = run_scf_kpoints_spin_polarized(
            prepare_periodic_single_point(_auh_problem(spin_polarized=True)),
            k_points,
            weights,
        )
        unpolarized = run_scf_kpoints(
            prepare_periodic_single_point(_auh_problem(spin_polarized=False)),
            k_points,
            weights,
        )
        self.assertTrue(polarized.converged and unpolarized.converged)
        self.assertAlmostEqual(polarized.magnetic_moment, 0.0, delta=1e-3)
        self.assertAlmostEqual(
            polarized.energies.total, unpolarized.energies.total, delta=2e-3
        )

    def test_rejects_unequal_weights(self) -> None:
        with self.assertRaises(ValueError):
            run_scf_kpoints_spin_polarized(
                self.system,
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                np.array([0.7, 0.3]),
            )

    def test_requires_spin_polarized_settings(self) -> None:
        system = prepare_periodic_single_point(_au_problem(spin_polarized=False))
        with self.assertRaises(ValueError):
            run_scf_kpoints_spin_polarized(system, np.zeros((1, 3)), np.ones(1))


class KPointSpinPolarizedPBETests(unittest.TestCase):
    def test_pbe_converges_with_a_net_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            potre = _relabelled_potre(Path(tmp), "pb")
            system = prepare_periodic_single_point(
                _au_problem(potre=potre, xc_functional="pbe")
            )
            k_points, weights = monkhorst_pack_grid(system.input.periodic_cell, (2, 1, 1))
            result = run_scf_kpoints_spin_polarized(system, k_points, weights)
        self.assertTrue(result.converged)
        self.assertGreater(result.magnetic_moment, 0.5)


_CLI_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 16
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Spin_Polarization: .true.

Kpoint_Method: mp
begin Monkhorst_Pack_Grid
2 1 1
end Monkhorst_Pack_Grid

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
"""


class KPointSpinPolarizedCliTests(unittest.TestCase):
    def test_cli_runs_and_reports_the_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _relabelled_potre(directory, "ca")
            (directory / "parsec.in").write_text(_CLI_INPUT, encoding="utf-8")
            exit_code = cli_main([str(directory / "parsec.in"), "--quiet"])
            self.assertEqual(exit_code, 0)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            archive = np.load(directory / "parsec_python_results.npz")
            moment = float(archive["magnetic_moment"])
            self.assertTrue(bool(archive["converged"]))
        self.assertIn("Converged magnetic moment N_up - N_down", log)
        # Au 6s^1: one unpaired electron per cell, correctly k-weighted.
        self.assertGreater(moment, 0.5)
        self.assertLess(moment, 1.5)


if __name__ == "__main__":
    unittest.main()
