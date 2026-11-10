
# With both sets of routines: the periodic grids of this module are those of the Fortran-derived routines
# (PeriodicGridSettings(fortran_routines=True), as Use_Fortran_Routines: .true. gives); the numbers checked
# here are those of the Fortran parity round.
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
from parsec_python.SCF import prepare_periodic_single_point, run_scf

_H_POTRE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"


def _hydrogen_box(side_length: float, spacing: float) -> SinglePointInput:
    cell = PeriodicCell(lattice_vectors=side_length * np.eye(3))
    center = side_length / 2.0
    return SinglePointInput(
        atoms=[Atom("H", [center, center, center])],
        pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
        grid=PeriodicGridSettings(fortran_routines=True, spacing=spacing, expansion_order=4),
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
                grid=PeriodicGridSettings(fortran_routines=True, spacing=0.6),
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

    def test_translates_atom_outside_cell_into_it(self) -> None:
        # Every periodic sum runs over images: an atom outside the cell is
        # moved into it by a lattice vector instead of being refused.
        cell = PeriodicCell(lattice_vectors=6.0 * np.eye(3))
        problem = SinglePointInput(
            atoms=[Atom("H", [-0.5, 3.0, 3.0])],
            pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
            grid=PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=4),
            periodic_cell=cell,
        )
        system = prepare_periodic_single_point(problem)
        np.testing.assert_array_equal(system.atoms[0].position, [5.5, 3.0, 3.0])

    def test_conventions_of_the_potential_and_the_lazy_gradient(self) -> None:
        # PARSEC 1.5's zero average (to the sampling of the short-range sum) and its alpha term, for both sets of
        # routines; the gradient matrices are built where they are read, element for element build_gradient's
        import pickle

        from parsec_python.Laplacian import build_gradient
        from parsec_python.V_ion import load_pseudopotentials
        from parsec_python.V_ion.gspace_local import alpha_coefficient

        for fortran in (True, False):
            with self.subTest(fortran_routines=fortran):
                problem = _hydrogen_box(6.0, 0.6)
                if not fortran:
                    problem = SinglePointInput(
                        atoms=problem.atoms,
                        pseudopotentials=problem.pseudopotentials,
                        grid=PeriodicGridSettings(spacing=0.6, expansion_order=4),
                        periodic_cell=problem.periodic_cell,
                    )
                system = prepare_periodic_single_point(problem)
                # the grid sum of the short-ranged pieces against their integral: 1e-2 Ry for H at 0.6 bohr
                # (4e-4 Ry for the carbon of diamond at 0.374 bohr), against the -0.329 Ry removed here
                self.assertLess(abs(float(np.mean(system.ionic_potential))), 2.0e-2)
                potential = load_pseudopotentials(problem.pseudopotentials)["H"]
                self.assertAlmostEqual(
                    system.alpha_z_energy, system.electron_count * alpha_coefficient(potential, 0) / 216.0, places=12
                )
                self.assertFalse(system.gradient.built)
                expected = build_gradient(system.grid)
                restored = pickle.loads(pickle.dumps(system.gradient))
                for lazy, matrix in zip(system.gradient, expected):
                    self.assertEqual((lazy != matrix).nnz, 0)
                self.assertTrue(system.gradient.built)
                self.assertEqual(len(restored), 3)
                self.assertEqual((restored[1] != expected[1]).nnz, 0)


class RunPeriodicSinglePointTests(unittest.TestCase):
    def test_hydrogen_in_a_box_converges(self) -> None:
        problem = _hydrogen_box(6.0, 0.6)
        system = prepare_periodic_single_point(problem)
        result = run_scf(system)

        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.energies.total))
        self.assertTrue(np.isfinite(result.ionic_potential).all())
        self.assertTrue(np.all(np.isfinite(result.density)))
        self.assertAlmostEqual(
            float(np.sum(result.density) * result.grid.volume_element),
            result.electron_count,
            places=4,
        )
        # A single positive point charge's own Ewald self-energy (against its
        # neutralizing background) is negative.
        self.assertLess(result.energies.ion_ion, 0.0)


if __name__ == "__main__":
    unittest.main()
