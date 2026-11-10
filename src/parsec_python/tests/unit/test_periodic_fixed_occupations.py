"""The occupations of ``occup.in`` (a negative ``Fermi_Temp``) in the periodic spin-polarized and k-point loops.

A hydrogen atom in a periodic box with its electron in the up channel: the energy-only Gamma loop
(``run_scf_spin_polarized``), the 1x1x1 k-point loop (``run_scf_kpoints_spin_polarized``) and the loop of the ionic
path (``run_spin_scf``) reach the same state, which a Fermi level would not (it would split the electron between the
channels of a closed-shell start at a negative temperature); a k-point loop with more than one k-point refuses
``occup.in``.  Inputs without ``occup.in`` take the Fermi level as before (the other tests of the drivers).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

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

LIBRARY = Path(__file__).resolve().parents[3] / "pp_generation"
HYDROGEN = LIBRARY / "NCPPs" / "PBE" / "H" / "PARSEC" / "H_POTRE.DAT"


def _problem() -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("H", [3.0, 3.0, 3.0])],
        pseudopotentials={"H": SpeciesPotential(HYDROGEN, 0, initial_spin_polarization=0.0)},
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
        scf=SCFSettings(
            xc_functional="pbe",
            spin_polarized=True,
            number_of_states=3,
            max_iterations=80,
            convergence_criterion=1.0e-7,
            fermi_temperature_kelvin=-1.0,
            fixed_occupations=(1.0, 0.0, 0.0),
            fixed_occupations_down=(0.0, 0.0, 0.0),
        ),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
    )


class PeriodicFixedOccupationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.SCF import prepare_periodic_single_point, run_scf
        from parsec_python.SCF.pbc import with_initial_spin_densities

        cls.system = prepare_periodic_single_point(_problem())
        cls.reference = run_scf(with_initial_spin_densities(cls.system))

    def test_gamma_loop_takes_occup_in(self) -> None:
        from parsec_python.SCF.spin_polarized import run_scf_spin_polarized

        result = run_scf_spin_polarized(self.system)
        self.assertTrue(result.converged and self.reference.converged)
        np.testing.assert_allclose(result.occupations_up, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result.occupations_down, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(result.magnetic_moment, 1.0, places=12)
        self.assertAlmostEqual(result.energies.total, self.reference.energies.total, delta=1.0e-6)

    def test_one_k_point_loop_takes_occup_in_and_several_refuse_it(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized

        result = run_scf_kpoints_spin_polarized(self.system, np.zeros((1, 3)), np.ones(1))
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.occupations_up, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(result.energies.total, self.reference.energies.total, delta=1.0e-6)
        two = (np.array([[0.0, 0.0, 0.0], [np.pi / 6.0, 0.0, 0.0]]), np.array([0.5, 0.5]))
        with self.assertRaises(NotImplementedError):
            run_scf_kpoints_spin_polarized(self.system, *two)
        unpolarized = replace(
            self.system,
            input=replace(
                self.system.input,
                scf=replace(self.system.input.scf, spin_polarized=False, fixed_occupations=(0.5, 0.0, 0.0), fixed_occupations_down=None),
            ),
        )
        with self.assertRaises(NotImplementedError):
            run_scf_kpoints(unpolarized, *two)


if __name__ == "__main__":
    unittest.main()
