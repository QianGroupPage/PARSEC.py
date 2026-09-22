from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF import prepare_periodic_single_point

_H_POTRE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"


def _hydrogen_box(side_length: float, spacing: float) -> SinglePointInput:
    cell = PeriodicCell(lattice_vectors=side_length * np.eye(3))
    center = side_length / 2.0
    return SinglePointInput(
        atoms=[Atom("H", [center, center, center])],
        pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
        grid=PeriodicGridSettings(spacing=spacing, expansion_order=4),
        periodic_cell=cell,
        scf=SCFSettings(max_iterations=40, number_of_states=3, convergence_criterion=1.0e-5),
        hartree=HartreeSettings(
            relative_tolerance=1.0e-8, absolute_tolerance=1.0e-12, max_iterations=2000
        ),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class SinglePointInputPeriodicValidationTests(unittest.TestCase):
    def test_isolated_input_requires_grid_settings(self) -> None:
        with self.assertRaises(ValueError):
            SinglePointInput(
                atoms=[Atom("H", [0.0, 0.0, 0.0])],
                pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
                grid=PeriodicGridSettings(spacing=0.6),
            )

    def test_periodic_input_requires_periodic_grid_settings(self) -> None:
        cell = PeriodicCell(lattice_vectors=6.0 * np.eye(3))
        with self.assertRaises(ValueError):
            SinglePointInput(
                atoms=[Atom("H", [3.0, 3.0, 3.0])],
                pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
                grid=GridSettings(spacing=0.6, radius=3.0),
                periodic_cell=cell,
            )


class PreparePeriodicSinglePointTests(unittest.TestCase):
    def test_requires_periodic_cell(self) -> None:
        problem = _hydrogen_box(6.0, 0.6)
        stripped = SinglePointInput(
            atoms=problem.atoms,
            pseudopotentials=problem.pseudopotentials,
            grid=GridSettings(spacing=0.6, radius=3.0),
            periodic_cell=None,
        )
        with self.assertRaises(ValueError):
            prepare_periodic_single_point(stripped)

    def test_rejects_atom_outside_cell(self) -> None:
        cell = PeriodicCell(lattice_vectors=6.0 * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("H", [-0.5, 3.0, 3.0])],
            pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
            grid=PeriodicGridSettings(spacing=0.6, expansion_order=4),
            periodic_cell=cell,
        )
        with self.assertRaises(ValueError):
            prepare_periodic_single_point(problem)



if __name__ == "__main__":
    unittest.main()
