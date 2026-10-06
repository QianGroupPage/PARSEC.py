"""Tests for periodic self-consistent SOC combined with collinear spin
polarization (k-points, spin and spin-orbit together), LDA and PBE."""

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
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.cli import main as cli_main
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.SCF.kpoints_soc import (
    run_self_consistent_soc_kpoints,
    run_self_consistent_soc_kpoints_spin_polarized,
)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)
_H_POTRE = _AU_POTRE.parent / "H_POTRE.DAT"
_BOX = 8.0


def _problem(
    *,
    potre: Path = _AU_POTRE,
    spin_polarized: bool = True,
    xc_functional: str = "ca",
    initial_polarization: float = 0.3,
    convergence: float = 5.0e-3,
    max_iterations: int = 60,
) -> SinglePointInput:
    cell = PeriodicCell(lattice_vectors=_BOX * np.eye(3))
    return SinglePointInput(
        atoms=[Atom("Au", [_BOX / 2, _BOX / 2, _BOX / 2])],
        pseudopotentials={
            "Au": SpeciesPotential(
                potre,
                0,
                spin_orbit=True,
                initial_spin_polarization=initial_polarization,
            )
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=max_iterations,
            number_of_states=20,
            convergence_criterion=convergence,
            spin_polarized=spin_polarized,
            self_consistent_spin_orbit=True,
            xc_functional=xc_functional,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _auh_problem(
    *, spin_polarized: bool, initial_polarization: float = 0.0
) -> SinglePointInput:
    """Closed-shell AuH in a periodic box: its paramagnetic state is stable."""
    box = 9.0
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return SinglePointInput(
        atoms=[
            Atom("Au", [6.0, box / 2, box / 2]),
            Atom("H", [6.0 - 2.89, box / 2, box / 2]),
        ],
        pseudopotentials={
            "Au": SpeciesPotential(
                _AU_POTRE,
                0,
                spin_orbit=True,
                initial_spin_polarization=initial_polarization,
            ),
            "H": SpeciesPotential(_H_POTRE, 0),
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=cell,
        scf=SCFSettings(
            max_iterations=120,
            number_of_states=20,
            convergence_criterion=1.0e-4,
            spin_polarized=spin_polarized,
            self_consistent_spin_orbit=True,
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


class KPointSpinorXCDeltaTests(unittest.TestCase):
    """Operator-level checks before trusting any SCF result built on top."""

    def setUp(self) -> None:
        cell = PeriodicCell(lattice_vectors=10.0 * np.eye(3))
        self.grid = build_periodic_grid(
            cell, PeriodicGridSettings(spacing=0.7, expansion_order=6)
        )
        atoms = [Atom("Au", [5.0, 5.0, 5.0])]
        specs = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
        pots = load_pseudopotentials(specs, xc_functional="ca")
        k_point = np.array([0.3, -0.15, 0.22])
        nonlocal_k = build_nonlocal_projectors(
            self.grid, atoms, pots, specs, cell.lattice_vectors, k_point=k_point
        )
        self.soc = tuple(
            build_spin_orbit_projectors(
                self.grid, atoms, pots, specs, cell.lattice_vectors, k_point=k_point
            )
        )
        rng = np.random.default_rng(1)
        self.veff = rng.normal(size=self.grid.size)
        self.xc_delta = 0.3 * rng.normal(size=self.grid.size)
        self.scalar = KPointKohnShamHamiltonian(
            build_negative_laplacian(self.grid),
            build_gradient(self.grid),
            self.veff,
            nonlocal_k,
            k_point,
        )

    def test_hermitian_with_xc_delta(self) -> None:
        hamiltonian = KPointSpinorKohnShamHamiltonian(
            self.scalar, self.soc, xc_delta=self.xc_delta
        )
        rng = np.random.default_rng(2)
        size = 2 * self.grid.size
        x = rng.normal(size=size) + 1j * rng.normal(size=size)
        y = rng.normal(size=size) + 1j * rng.normal(size=size)
        lhs = np.vdot(y, hamiltonian.apply(x))
        rhs = np.vdot(hamiltonian.apply(y), x)
        self.assertAlmostEqual(lhs.real, rhs.real, delta=1e-8 * abs(lhs))
        self.assertAlmostEqual(lhs.imag, rhs.imag, delta=1e-8 * abs(lhs))

    def test_xc_delta_adds_opposite_zeeman_terms(self) -> None:
        plain = KPointSpinorKohnShamHamiltonian(self.scalar, self.soc)
        split = KPointSpinorKohnShamHamiltonian(
            self.scalar, self.soc, xc_delta=self.xc_delta
        )
        rng = np.random.default_rng(3)
        n = self.grid.size
        x = rng.normal(size=2 * n) + 1j * rng.normal(size=2 * n)
        difference = split.apply(x) - plain.apply(x)
        np.testing.assert_allclose(difference[:n], self.xc_delta * x[:n], atol=1e-12)
        np.testing.assert_allclose(difference[n:], -self.xc_delta * x[n:], atol=1e-12)

    def test_none_matches_omitting_the_field(self) -> None:
        a = KPointSpinorKohnShamHamiltonian(self.scalar, self.soc)
        b = KPointSpinorKohnShamHamiltonian(self.scalar, self.soc, xc_delta=None)
        x = np.random.default_rng(4).normal(size=2 * self.grid.size).astype(complex)
        np.testing.assert_array_equal(a.apply(x), b.apply(x))

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            KPointSpinorKohnShamHamiltonian(
                self.scalar, self.soc, xc_delta=np.zeros(3)
            )


class PeriodicSpinPolarizedSOCSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system = prepare_periodic_single_point(_problem())
        cell = cls.system.input.periodic_cell
        cls.gamma = (np.zeros((1, 3)), np.ones(1))
        cls.result = run_self_consistent_soc_kpoints_spin_polarized(
            cls.system, *cls.gamma
        )
        cls.cell = cell

    def test_converges_with_a_net_moment(self) -> None:
        self.assertTrue(self.result.converged)
        # Au 6s^1: one unpaired electron.
        self.assertGreater(self.result.magnetic_moment, 0.5)
        self.assertTrue(np.isfinite(self.result.energies.total))

    def test_occupations_conserve_electrons(self) -> None:
        self.assertAlmostEqual(
            float(np.sum(self.result.occupations)), self.result.electron_count, places=4
        )

    def test_moment_matches_spinor_expectation_at_gamma(self) -> None:
        """At a single k-point the net moment equals sum_n f_n <sigma_z>_n."""
        expectation = float(
            np.dot(self.result.occupations, self.result.magnetic_moment_per_state)
        )
        self.assertAlmostEqual(self.result.magnetic_moment, expectation, delta=1e-6)

    def test_closed_shell_reproduces_unpolarized_driver(self) -> None:
        """For a closed-shell molecule (AuH) with no symmetry-breaking seed the
        spin-polarized loop must stay paramagnetic and agree with the
        unpolarized SOC driver -- an independent code path for the same
        physics.  (An isolated open-shell atom cannot serve here: its
        paramagnetic state is unstable and numerical noise alone drives it to
        the magnetic one.)"""
        polarized = run_self_consistent_soc_kpoints_spin_polarized(
            prepare_periodic_single_point(_auh_problem(spin_polarized=True)),
            *self.gamma,
        )
        unpolarized = run_self_consistent_soc_kpoints(
            prepare_periodic_single_point(_auh_problem(spin_polarized=False)),
            *self.gamma,
        )
        self.assertTrue(polarized.converged and unpolarized.converged)
        self.assertAlmostEqual(polarized.magnetic_moment, 0.0, delta=1e-3)
        self.assertAlmostEqual(
            polarized.energies.total, unpolarized.energies.total, delta=2e-3
        )

    def test_monkhorst_pack_grid_converges(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.cell, (2, 1, 1))
        result = run_self_consistent_soc_kpoints_spin_polarized(
            self.system, k_points, weights
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.eigenvalues.size, 2 * 20)
        # Each pooled state carries weight 1/2 (two k-points).
        self.assertAlmostEqual(
            float(np.sum(result.occupations)) * 0.5, result.electron_count, places=4
        )
        self.assertGreater(result.magnetic_moment, 0.5)

    def test_rejects_unequal_weights(self) -> None:
        with self.assertRaises(ValueError):
            run_self_consistent_soc_kpoints_spin_polarized(
                self.system,
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                np.array([0.7, 0.3]),
            )

    def test_requires_spin_polarized_settings(self) -> None:
        system = prepare_periodic_single_point(_problem(spin_polarized=False))
        with self.assertRaises(ValueError):
            run_self_consistent_soc_kpoints_spin_polarized(system, *self.gamma)


class PeriodicSpinPolarizedSOCWithPBETests(unittest.TestCase):
    def test_pbe_converges_with_a_net_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            potre = _relabelled_potre(Path(tmp), "pb")
            system = prepare_periodic_single_point(
                _problem(potre=potre, xc_functional="pbe")
            )
            result = run_self_consistent_soc_kpoints_spin_polarized(
                system, np.zeros((1, 3)), np.ones(1)
            )
        self.assertTrue(result.converged)
        self.assertGreater(result.magnetic_moment, 0.5)
        self.assertTrue(np.isfinite(result.energies.total))


_CLI_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

SO_from_scratch: .true.
Spin_Polarization: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: {correlation}
"""


class PeriodicSpinPolarizedSOCCliTests(unittest.TestCase):
    def _run(self, correlation: str, label: str, extra: str = "") -> tuple[str, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _relabelled_potre(directory, label)
            (directory / "parsec.in").write_text(
                _CLI_INPUT.format(correlation=correlation) + extra, encoding="utf-8"
            )
            exit_code = cli_main([str(directory / "parsec.in"), "--quiet"])
            self.assertEqual(exit_code, 0)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            archive = np.load(directory / "parsec_python_results.npz")
            return log, {key: archive[key] for key in archive.files}

    def test_gamma_run_reports_moment(self) -> None:
        log, archive = self._run("CA", "ca")
        self.assertIn("Converged magnetic moment N_up - N_down", log)
        self.assertTrue(bool(archive["converged"]))
        self.assertGreater(float(archive["magnetic_moment"]), 0.5)

    def test_monkhorst_pack_pbe_run(self) -> None:
        log, archive = self._run(
            "PBE",
            "pb",
            extra=(
                "Kpoint_Method: mp\n"
                "begin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"
            ),
        )
        self.assertTrue(bool(archive["converged"]))
        self.assertEqual(archive["eigenvalues_ry"].shape, (40,))
        self.assertGreater(float(archive["magnetic_moment"]), 0.5)


if __name__ == "__main__":
    unittest.main()
