"""Physics-validation tests for the collinear spin-polarized SCF path."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
_C_POTRE = _EXAMPLES / "0d_benzene" / "C_POTRE.DAT"
_H_POTRE_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"


def _carbon_problem(**scf_overrides) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("C", [0.0, 0.0, 0.0])],
        pseudopotentials={
            "C": SpeciesPotential(_C_POTRE, 1, initial_spin_polarization=0.3)
        },
        grid=GridSettings(spacing=0.5, radius=5.0, expansion_order=6),
        scf=SCFSettings(
            max_iterations=60,
            number_of_states=6,
            spin_polarized=True,
            convergence_criterion=1.0e-3,
            **scf_overrides,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class CarbonTripletGroundStateTests(unittest.TestCase):
    """An isolated carbon atom's 2p^2 ground state is a Hund's-rule triplet
    (S=1): two unpaired electrons, so N_up-N_down should converge to 2."""

    @classmethod
    def setUpClass(cls) -> None:
        system = prepare_single_point(_carbon_problem())
        cls.result = run_scf_spin_polarized(system)

    def test_converges(self) -> None:
        self.assertTrue(self.result.converged)

    def test_conserves_total_electron_count(self) -> None:
        total = np.sum(self.result.occupations_up) + np.sum(self.result.occupations_down)
        self.assertAlmostEqual(total, self.result.electron_count, places=2)

    def test_reproduces_hunds_rule_triplet_moment(self) -> None:
        self.assertAlmostEqual(self.result.magnetic_moment, 2.0, places=3)

    def test_energy_is_finite_and_negative(self) -> None:
        self.assertTrue(np.isfinite(self.result.energies.total))
        self.assertLess(self.result.energies.total, 0.0)


class SpinPolarizedEntryGuardsTests(unittest.TestCase):
    def test_requires_spin_polarized_settings(self) -> None:
        problem = SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(_H_POTRE_FIXTURE, 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=4),
            scf=SCFSettings(max_iterations=1, number_of_states=3),
            hartree=HartreeSettings(multipole_order=2),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_single_point(problem)
        with self.assertRaises(ValueError):
            run_scf_spin_polarized(system)

    def test_unpolarized_run_scf_refuses_spin_polarized_settings(self) -> None:
        from parsec_python.SCF.single_point import run_scf

        problem = SinglePointInput(
            atoms=[Atom("H", [0.0, 0.0, 0.0])],
            pseudopotentials={"H": SpeciesPotential(_H_POTRE_FIXTURE, 0)},
            grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=4),
            scf=SCFSettings(max_iterations=1, number_of_states=3, spin_polarized=True),
            hartree=HartreeSettings(multipole_order=2),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_single_point(problem)
        with self.assertRaises(NotImplementedError):
            run_scf(system)


if __name__ == "__main__":
    unittest.main()
