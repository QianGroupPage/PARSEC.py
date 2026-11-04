"""Forces of a periodic cell with the Ono-Hirose double grid (``Double_Grid_Order > 1``).

``nonloc.F90`` builds the projector gradients of a periodic cell on the fine
points too (with the images and the Bloch phase of the coarse point).  The
nonlocal force of fixed orbitals is checked against central differences of
their nonlocal energy (Gamma and a k-point), Fortran's ``dvw`` option against
the cluster convention, and the complete force of an SCF loop against the
free energy.  The Fortran comparison (Si8 with order 2) is in the report of
the parity round.
"""


# With both sets of routines: the periodic grids of this module are those of the Fortran-derived routines
# (PeriodicGridSettings(fortran_routines=True), as Use_Fortran_Routines: .true. gives); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces.periodic import build_periodic_projector_gradients, periodic_nonlocal_forces
from parsec_python.Grid.pbc import build_periodic_grid
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
from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend
from parsec_python.SCF import prepare_periodic_single_point
from parsec_python.SCF.pbc import update_periodic_geometry
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials

DATA = Path(__file__).resolve().parents[1] / "data"
SILICON = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)}
CELL = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
ATOMS = (
    Atom("Si", np.array([0.31, 0.22, 0.13])),
    Atom("Si", np.array([2.9, 2.6, 3.1])),
)


def _moved(atoms, index, axis, step):
    out = list(atoms)
    position = np.array(out[index].position, dtype=float)
    position[axis] += step
    out[index] = Atom(out[index].symbol, position)
    return tuple(out)


class DoubleGridNonlocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.potentials = load_pseudopotentials(SILICON)
        cls.grid = build_periodic_grid(
            CELL, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4, double_grid_order=2)
        )

    def test_force_is_the_derivative_of_the_nonlocal_energy(self) -> None:
        rng = np.random.default_rng(2)
        occupations = np.array([1.0, 0.6, 0.2])
        step = 1e-6
        for k_point in (None, np.array([0.1, -0.2, 0.15])):
            vectors = rng.normal(size=(self.grid.size, 3))
            if k_point is not None:
                vectors = vectors + 1j * rng.normal(size=(self.grid.size, 3))
            vectors /= np.linalg.norm(vectors, axis=0)
            operator = build_nonlocal_projectors(
                self.grid, ATOMS, self.potentials, SILICON, CELL.lattice_vectors, k_point=k_point
            )
            gradients = build_periodic_projector_gradients(
                self.grid, ATOMS, self.potentials, SILICON, CELL.lattice_vectors, k_point
            )
            forces = periodic_nonlocal_forces(operator, gradients, vectors, occupations, 2)

            def energy(atoms):
                moved = build_nonlocal_projectors(
                    self.grid, atoms, self.potentials, SILICON, CELL.lattice_vectors,
                    k_point=k_point,
                )
                overlaps = moved.projectors.conj().T @ vectors
                return 2.0 * float(
                    np.sum(moved.signs[:, None] * np.abs(overlaps) ** 2 * occupations[None, :])
                )

            for index, axis in ((0, 0), (1, 2)):
                numeric = -(
                    energy(_moved(ATOMS, index, axis, step))
                    - energy(_moved(ATOMS, index, axis, -step))
                ) / (2.0 * step)
                self.assertAlmostEqual(forces[index, axis], numeric, delta=1e-9)

    def test_tabulated_slopes_are_those_of_the_cluster_double_grid(self) -> None:
        from parsec_python.Forces.nonlocal_force import double_grid_projector_gradients

        # A cell large enough for one image per projector: Fortran's dvw
        # convention of the periodic builder equals the cluster one.
        cell = PeriodicCell(lattice_vectors=12.0 * np.eye(3))
        grid = build_periodic_grid(
            cell, PeriodicGridSettings(fortran_routines=True, spacing=0.8, expansion_order=4, double_grid_order=2)
        )
        atoms = (Atom("Si", np.array([6.1, 5.7, 6.3])),)
        periodic = build_periodic_projector_gradients(
            grid, atoms, self.potentials, SILICON, cell.lattice_vectors, radial_slopes="tabulated"
        )
        cluster = double_grid_projector_gradients(
            grid, atoms, self.potentials, SILICON, order=2
        )
        for left, right in zip(periodic.components, cluster.components):
            np.testing.assert_allclose(left.toarray(), right.toarray(), atol=1e-14)


class DoubleGridSelfConsistentTests(unittest.TestCase):
    def test_forces_of_an_scf_loop(self) -> None:
        problem = SinglePointInput(
            atoms=list(ATOMS),
            pseudopotentials=SILICON,
            grid=PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4, double_grid_order=2),
            periodic_cell=CELL,
            scf=SCFSettings(
                max_iterations=80,
                number_of_states=8,
                convergence_criterion=1e-10,
                fermi_temperature_kelvin=300.0,
            ),
            hartree=HartreeSettings(
                relative_tolerance=1e-12, absolute_tolerance=1e-14, max_iterations=4000
            ),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1e-10),
        )
        backend = periodic_backend()
        system = prepare_periodic_single_point(problem)
        result = backend.run_scf(system)
        forces = backend.compute_forces(system, result, symmetrize=False)

        def free_energy(result, temperature=300.0):
            f = np.clip(np.asarray(result.occupations), 1e-300, 1 - 1e-16)
            entropy = 2.0 * np.sum(f * np.log(f) + (1 - f) * np.log(1 - f))
            return result.energies.total + 6.33327186e-6 * temperature * entropy

        # A step that crosses no radial knot of the linear interpolants.
        step = 5e-7
        energies = [
            free_energy(
                backend.run_scf(update_periodic_geometry(system, _moved(ATOMS, 0, 1, sign * step)))
            )
            for sign in (1.0, -1.0)
        ]
        numeric = -(energies[0] - energies[1]) / (2.0 * step)
        self.assertAlmostEqual(forces.unsymmetrized[0, 1], numeric, delta=2e-8)


if __name__ == "__main__":
    unittest.main()
