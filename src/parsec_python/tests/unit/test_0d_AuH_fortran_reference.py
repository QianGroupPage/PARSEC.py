"""Regression tests comparing PARSEC.py against a real, freshly-run Fortran
PARSEC reference on a scaled-down AuH molecule (same geometry and
pseudopotentials as examples/0d_AuH/, reduced grid resolution for a fast
comparison).

The Fortran reference values below and the full logs they come from are
recorded in examples/0d_AuH/reduced_perturbative_soc/fortran_reference.out
and examples/0d_AuH/reduced_self_consistent_soc/fortran_reference.out (see
the README.md next to each for exactly how they were generated). Tolerances
are set from the observed agreement (total energy to ~1e-5 Ry, eigenvalues
to ~1e-3 Ry) rather than exact equality, since the two codes use different
eigensolvers/mixing and the grid discretization is only nominally identical.
<S_z> within tightly-spaced near-degenerate multiplets (Au's d-manifold) is
basis-sensitive between the two codes -- this is checked only for
well-separated states, not exhaustively.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.cli import main as cli_main

_FIXTURES = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"


class PerturbativeSOCFortranReferenceTests(unittest.TestCase):
    """examples/0d_AuH/reduced_perturbative_soc: Spin_Polarization=true,
    SO_PSP=true on Au, no SO_from_scratch/SCF_SO (Fortran's default
    perturbative path)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        directory = Path(cls._tmpdir.name)
        shutil.copytree(
            _FIXTURES / "reduced_perturbative_soc", directory, dirs_exist_ok=True
        )
        exit_code = cli_main([str(directory / "parsec.in"), "--quiet"])
        if exit_code != 0:
            raise AssertionError(f"CLI run failed with exit code {exit_code}")
        cls.archive = np.load(directory / "parsec_python_results.npz")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_pre_soc_scf_total_energy_matches_fortran(self) -> None:
        # Fortran: -68.32490203 Ry (fortran_reference.out, converged
        # scalar-relativistic SCF, before the perturbative SOC correction).
        self.assertAlmostEqual(
            float(self.archive["energy_total_ry"]), -68.32490203, delta=5.0e-5
        )

    def test_magnetic_moment_is_zero(self) -> None:
        self.assertAlmostEqual(float(self.archive["magnetic_moment"]), 0.0, delta=1.0e-6)

    def test_soc_split_state_1_matches_fortran(self) -> None:
        # Fortran: -0.818415 Ry, <S_z>=0.9779 (fortran_reference.out).
        self.assertAlmostEqual(
            float(self.archive["soc_eigenvalues_ry"][0]), -0.818415, delta=2.0e-3
        )
        self.assertAlmostEqual(
            float(self.archive["soc_magnetic_moment"][0]), 0.9779, delta=0.01
        )

    def test_kramers_like_pairing_of_soc_split_states(self) -> None:
        eigenvalues = self.archive["soc_eigenvalues_ry"]
        moments = self.archive["soc_magnetic_moment"]
        for i in range(0, len(eigenvalues) - 1, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=2.0e-3)
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.1)


class SelfConsistentSOCFortranReferenceTests(unittest.TestCase):
    """examples/0d_AuH/reduced_self_consistent_soc: SO_from_scratch=true,
    SO_PSP=true on Au, no Spin_Polarization (PARSEC.py's only currently
    supported cluster self-consistent-SOC mode; Fortran's own input parser
    requires Spin_Polarization=true whenever any species has SO_PSP=true,
    even for SCF_SO -- see usrinputfile.F90:2574-2580 -- so the Fortran
    reference run has Spin_Polarization=true where PARSEC.py's does not.
    AuH's ground state has zero net moment in both codes, which is what
    makes the comparison meaningful despite that difference)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        directory = Path(cls._tmpdir.name)
        shutil.copytree(
            _FIXTURES / "reduced_self_consistent_soc", directory, dirs_exist_ok=True
        )
        exit_code = cli_main([str(directory / "parsec.in"), "--quiet"])
        if exit_code != 0:
            raise AssertionError(f"CLI run failed with exit code {exit_code}")
        cls.archive = np.load(directory / "parsec_python_results.npz")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_converged_total_energy_matches_fortran(self) -> None:
        # Fortran: -68.34107156 Ry (fortran_reference.out, converged
        # self-consistent SOC, Spin_Polarization=true forced by Fortran's
        # own input validation).
        self.assertAlmostEqual(
            float(self.archive["energy_total_ry"]), -68.34107156, delta=5.0e-5
        )

    def test_state_1_eigenvalue_matches_fortran(self) -> None:
        # Fortran: -0.821060 Ry (fortran_reference.out, converged
        # iteration). <S_z> is not compared here -- states 1-12 sit inside
        # a tightly-spaced near-degenerate manifold where the exact spin
        # decomposition is basis-sensitive between the two independently
        # converged solutions (see module docstring).
        self.assertAlmostEqual(
            float(self.archive["eigenvalues_ry"][0]), -0.821060, delta=2.0e-3
        )

    def test_kramers_pairing_of_spinor_states(self) -> None:
        eigenvalues = self.archive["eigenvalues_ry"]
        moments = self.archive["magnetic_moment_per_state"]
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1.0e-4)
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)


class SelfConsistentSOCSpinPolarizedFortranReferenceTests(unittest.TestCase):
    """examples/0d_AuH/reduced_self_consistent_soc/parsec_spin_polarized.in:
    SO_from_scratch=true, Spin_Polarization=true, SO_PSP=true on Au -- the
    combined feature (run_self_consistent_soc_spin_polarized), run against
    the *same* fortran_reference.out as
    SelfConsistentSOCFortranReferenceTests above (that Fortran run always
    had Spin_Polarization=true, since Fortran's input parser requires it;
    this is the first PARSEC.py input that matches it structurally, not
    just numerically through AuH's incidental zero net moment). Tolerances
    are noticeably tighter than the no-spin-polarization comparison above,
    since the physics genuinely matches now rather than approximating it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        directory = Path(cls._tmpdir.name)
        shutil.copytree(
            _FIXTURES / "reduced_self_consistent_soc", directory, dirs_exist_ok=True
        )
        exit_code = cli_main(
            [str(directory / "parsec_spin_polarized.in"), "--quiet"]
        )
        if exit_code != 0:
            raise AssertionError(f"CLI run failed with exit code {exit_code}")
        cls.archive = np.load(directory / "parsec_python_results.npz")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_converged_total_energy_matches_fortran(self) -> None:
        # Fortran: -68.34107156 Ry (fortran_reference.out).
        self.assertAlmostEqual(
            float(self.archive["energy_total_ry"]), -68.34107156, delta=3.0e-5
        )

    def test_state_1_eigenvalue_and_moment_match_fortran(self) -> None:
        # Fortran: -0.821060 Ry, <S_z>=0.9766 (fortran_reference.out).
        self.assertAlmostEqual(
            float(self.archive["eigenvalues_ry"][0]), -0.821060, delta=1.5e-3
        )
        self.assertAlmostEqual(
            float(self.archive["magnetic_moment_per_state"][0]), 0.9766, delta=0.01
        )

    def test_states_are_only_approximately_paired(self) -> None:
        """Unlike the no-spin-polarization case above, exact Kramers
        degeneracy is *not* a rigorous guarantee here: the converged
        xc_delta field only needs AuH's *net* moment to vanish, not its
        local spin density pointwise, so it can still break time-reversal
        symmetry locally even at zero net moment. The splitting should
        still be small (a fraction of the underlying SOC splitting itself,
        not comparable to the level spacing)."""
        eigenvalues = self.archive["eigenvalues_ry"]
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertLess(abs(eigenvalues[i] - eigenvalues[i + 1]), 5.0e-3)


if __name__ == "__main__":
    unittest.main()
