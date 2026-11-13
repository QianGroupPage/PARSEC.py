"""Forces, relaxation and molecular dynamics of periodic (bulk) cells.

Every force term is compared with a central difference of the quantity it
differentiates: the Ewald ion-ion energy, the grid integrals of fixed fields
with the local potential, the superposed core and valence densities and the
nonlocal energy of fixed orbitals (Gamma and a k-point, orthorhombic and
general grids), and the total (free) energy of converged SCF loops.  The
Fortran comparison is in the report of the parity round.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces.ion_ion import ewald_ion_ion_forces  # Andrew Grieder's, which replaced ours
from parsec_python.Forces.periodic import (  # the Fortran-derived functions of these names
    build_periodic_projector_gradients,
    periodic_local_forces,
    periodic_nonlocal_forces,
)
from parsec_python.Forces import (
    periodic_symmetry_operations,
    symmetrize_forces,
    symmetrize_positions,
)
from parsec_python.Forces.periodic import interpolant_slope, local_potential_slope
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
from parsec_python.Relaxation import run_relaxation
from parsec_python.Relaxation.domove import RelaxationSettings
from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend
from parsec_python.SCF import prepare_periodic_single_point
from parsec_python.SCF.pbc import update_periodic_geometry
from parsec_python.V_ion import (
    build_nonlocal_projectors,
    ewald_ion_ion_energy,
    ewald_local_ionic_potential,
    load_pseudopotentials,
    superpose_atomic_density,
)

DATA = Path(__file__).resolve().parents[1] / "data"
STEP = 1.0e-6


def _moved(atoms, index, axis, step):
    out = list(atoms)
    position = np.array(out[index].position, dtype=float)
    position[axis] += step
    out[index] = Atom(out[index].symbol, position)
    return tuple(out)


def _central(function, atoms, index, axis, step=STEP):
    return -(
        function(_moved(atoms, index, axis, step)) - function(_moved(atoms, index, axis, -step))
    ) / (2.0 * step)


_FCC = 0.5 * 10.26 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


class ComponentDerivativeTests(unittest.TestCase):
    """Each term against a finite difference, no SCF loop."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.si = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)}
        cls.si_potentials = load_pseudopotentials(cls.si)
        cls.core = {"H": SpeciesPotential(DATA / "H_CORE_POTRE.DAT", 0)}
        cls.core_potentials = load_pseudopotentials(cls.core)

    def cells(self):
        cube = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
        yield (
            "orthorhombic",
            cube,
            build_periodic_grid(cube, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4)),
            (Atom("Si", np.array([0.31, 0.22, 0.13])), Atom("Si", np.array([2.9, 2.6, 3.1]))),
        )
        fcc = PeriodicCell(lattice_vectors=_FCC)
        yield (
            "general",
            fcc,
            build_periodic_grid(fcc, PeriodicGridSettings(fortran_routines=True, spacing=0.9, expansion_order=4)),
            (
                Atom("Si", np.array([0.12, 0.03, 0.10])),
                Atom("Si", 0.25 * 10.26 * np.ones(3) + np.array([0.12, 0.0, -0.07])),
            ),
        )

    def fields(self, grid, lattice):
        x = grid.coordinates
        b = 2.0 * np.pi * np.linalg.inv(lattice).T
        density = 0.05 + 0.02 * np.cos(x @ b[0]) + 0.015 * np.sin(x @ (b[1] + b[2]))
        residual = 0.03 * np.sin(x @ b[1]) + 0.02 * np.cos(x @ (b[0] + b[1]))
        xc = -0.4 + 0.1 * np.cos(x @ b[2])
        return density, residual, xc

    def test_ewald_forces_are_the_gradient_of_the_ewald_energy(self) -> None:
        for name, cell, _, atoms in self.cells():
            with self.subTest(cell=name):
                lattice = cell.lattice_vectors
                forces = ewald_ion_ion_forces(atoms, self.si_potentials, lattice)
                for index in range(2):
                    for axis in range(3):
                        numeric = _central(
                            lambda moved: ewald_ion_ion_energy(moved, self.si_potentials, lattice),
                            atoms,
                            index,
                            axis,
                        )
                        self.assertAlmostEqual(forces[index, axis], numeric, delta=2e-8)
                np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)

    def test_local_force_is_the_gradient_of_the_ewald_split_potential(self) -> None:
        for name, cell, grid, atoms in self.cells():
            with self.subTest(cell=name):
                lattice = cell.lattice_vectors
                density, _, _ = self.fields(grid, lattice)
                zeros = np.zeros(grid.size)
                terms = periodic_local_forces(
                    grid, atoms, self.si_potentials, self.si, lattice, density, zeros, zeros
                )
                self.assertTrue(np.all(terms.self_consistency == 0.0))
                self.assertTrue(np.all(terms.core_correction == 0.0))

                def energy(moved):
                    field = ewald_local_ionic_potential(
                        grid, moved, self.si_potentials, self.si, lattice
                    )
                    return grid.integrate(density * field)

                # Atom 0 sits near a corner of the cell: its images beyond the
                # cutoff in |T| alone are part of the energy and of the force.
                for index in range(2):
                    for axis in range(3):
                        self.assertAlmostEqual(
                            terms.ionic[index, axis],
                            _central(energy, atoms, index, axis),
                            delta=3e-8,
                        )

    def test_core_and_self_consistency_terms_are_density_gradients(self) -> None:
        for name, cell, grid, si_atoms in self.cells():
            with self.subTest(cell=name):
                lattice = cell.lattice_vectors
                atoms = tuple(Atom("H", atom.position) for atom in si_atoms)
                _, residual, xc = self.fields(grid, lattice)
                terms = periodic_local_forces(
                    grid,
                    atoms,
                    self.core_potentials,
                    self.core,
                    lattice,
                    np.zeros(grid.size),
                    residual,
                    xc,
                )
                for axis in range(3):
                    core = _central(
                        lambda moved: grid.integrate(
                            xc
                            * superpose_atomic_density(
                                grid, moved, self.core_potentials, self.core, lattice, core=True
                            )
                        ),
                        atoms,
                        0,
                        axis,
                    )
                    scf = _central(
                        lambda moved: grid.integrate(
                            residual
                            * superpose_atomic_density(
                                grid, moved, self.core_potentials, self.core, lattice
                            )
                        ),
                        atoms,
                        0,
                        axis,
                    )
                    self.assertAlmostEqual(terms.core_correction[0, axis], core, delta=3e-8)
                    self.assertAlmostEqual(terms.self_consistency[0, axis], scf, delta=3e-8)

    def test_nonlocal_force_with_images_and_bloch_phases(self) -> None:
        rng = np.random.default_rng(5)
        for name, cell, grid, atoms in self.cells():
            lattice = cell.lattice_vectors
            reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
            for k_point in (None, 0.3 * reciprocal[0] - 0.2 * reciprocal[2]):
                with self.subTest(cell=name, k=k_point is not None):
                    orbitals = rng.normal(size=(grid.size, 3))
                    if k_point is not None:
                        orbitals = orbitals + 1j * rng.normal(size=(grid.size, 3))
                    orbitals /= np.linalg.norm(orbitals, axis=0)
                    occupations = np.array([1.0, 0.6, 0.1])

                    def energy(moved):
                        operator = build_nonlocal_projectors(
                            grid, moved, self.si_potentials, self.si, lattice, k_point=k_point
                        )
                        overlaps = np.asarray(operator.projectors.conj().T @ orbitals)
                        return 2.0 * float(
                            np.sum(operator.signs[:, None] * np.abs(overlaps) ** 2 * occupations)
                        )

                    operator = build_nonlocal_projectors(
                        grid, atoms, self.si_potentials, self.si, lattice, k_point=k_point
                    )
                    gradients = build_periodic_projector_gradients(
                        grid, atoms, self.si_potentials, self.si, lattice, k_point
                    )
                    forces = periodic_nonlocal_forces(
                        operator, gradients, orbitals, occupations, len(atoms)
                    )
                    for axis in range(3):
                        self.assertAlmostEqual(
                            forces[1, axis], _central(energy, atoms, 1, axis), delta=3e-8
                        )

    def test_an_atom_within_round_off_of_a_grid_point(self) -> None:
        # Found when both sets of routines were compared: the 1/|d| of a round-off distance made |grad beta| 1.8e6 (Si64, 82 Ry/bohr).
        # Below min(first radial knot, ROUND_OFF_DISTANCE) the distance is clamped: continuous, bounded.
        from parsec_python.Forces.periodic import ROUND_OFF_DISTANCE

        cube = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
        grid = build_periodic_grid(cube, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4))
        node = grid.coordinates[int(np.argmin(np.linalg.norm(grid.coordinates - 2.8, axis=1)))]
        first_knot = self.si_potentials["Si"].radii[0]
        clamp = min(first_knot, ROUND_OFF_DISTANCE)

        def gradients(offset, slopes="interpolant"):
            return build_periodic_projector_gradients(
                grid, [Atom("Si", node + offset)], self.si_potentials, self.si, cube.lattice_vectors,
                radial_slopes=slopes,
            )

        def largest(result):
            return max(abs(component).max() for component in result.components)

        bound = {
            "interpolant": largest(gradients(np.array([0.0, clamp, 0.0]))),
            "tabulated": largest(gradients(np.array([0.0, first_knot, 0.0]), "tabulated")),
        }
        for name, offset in (
            ("round-off", np.array([1.8e-15, 0.0, -1.8e-15])),
            ("round-off, one axis", np.array([0.0, 3.6e-15, 0.0])),
            ("half the clamp", np.array([0.0, 0.5 * clamp, 0.0])),
        ):
            for slopes in ("interpolant", "tabulated"):
                with self.subTest(case=name, slopes=slopes):
                    self.assertLess(largest(gradients(offset, slopes)), 1.001 * bound[slopes])
        # just below and just above the clamp: the same gradients to the step (continuous)
        below = gradients(np.array([0.0, 0.999 * clamp, 0.0]))
        above = gradients(np.array([0.0, 1.001 * clamp, 0.0]))
        for a, b in zip(below.components, above.components):
            self.assertLess(abs(a - b).max(), 3.0e-3 * bound["interpolant"])
        # an atom exactly on the grid point keeps the gradients of the Fortran parity round (1/|d| = 0 there)
        exact = gradients(np.zeros(3))
        centre = int(np.argmin(np.linalg.norm(grid.coordinates - node, axis=1)))
        self.assertEqual(abs(exact.components[0][centre]).max(), 0.0)

    def test_tabulated_slopes_are_the_fortran_table(self) -> None:
        # The Fortran convention gives other forces than the exact slope.
        _, cell, grid, atoms = next(iter(self.cells()))
        lattice = cell.lattice_vectors
        exact = build_periodic_projector_gradients(grid, atoms, self.si_potentials, self.si, lattice)
        table = build_periodic_projector_gradients(
            grid, atoms, self.si_potentials, self.si, lattice, radial_slopes="tabulated"
        )
        self.assertEqual(exact.labels, table.labels)
        difference = abs(exact.components[0] - table.components[0]).max()
        self.assertGreater(difference, 0.0)
        with self.assertRaises(ValueError):
            build_periodic_projector_gradients(
                grid, atoms, self.si_potentials, self.si, lattice, radial_slopes="smooth"
            )

    def test_slopes_of_the_interpolants(self) -> None:
        knots = np.array([0.0, 1.0, 3.0])
        values = np.array([1.0, 3.0, 2.0])
        np.testing.assert_allclose(
            interpolant_slope(np.array([-1.0, 0.5, 1.0, 2.5, 3.0]), knots, values),
            [0.0, 2.0, -0.5, -0.5, 0.0],
        )
        potential = self.si_potentials["Si"]
        radius = np.array([0.37, 1.3, 2.71, 150.0])
        step = 1e-7
        numeric = (
            potential.local_potential(radius + step, 2) - potential.local_potential(radius - step, 2)
        ) / (2 * step)
        np.testing.assert_allclose(local_potential_slope(potential, radius, 2), numeric, rtol=1e-6)


def _hydrogen_cell(lattice, atoms, *, spin=False, states=4, species="H_CORE_POTRE.DAT"):
    return SinglePointInput(
        atoms=list(atoms),
        pseudopotentials={"H": SpeciesPotential(DATA / species, 0)},
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=np.asarray(lattice, dtype=float)),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=states,
            convergence_criterion=1.0e-10,
            spin_polarized=spin,
            fermi_temperature_kelvin=300.0,
        ),
        hartree=HartreeSettings(
            relative_tolerance=1.0e-12, absolute_tolerance=1.0e-14, max_iterations=4000
        ),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-10),
    )


def _free_energy(result, temperature=300.0):
    """``E - T S``: the forces are the derivative of the Mermin free energy."""

    occupations = np.clip(np.asarray(result.occupations, dtype=float), 1e-300, 1 - 1e-16)
    degeneracy = 2.0 if getattr(result, "spin", None) is None else 1.0
    weights = np.ones_like(occupations)
    if getattr(result, "k_weights", None) is not None:
        weights = np.repeat(result.k_weights, occupations.size // len(result.k_weights))
    entropy = occupations * np.log(occupations) + (1 - occupations) * np.log(1 - occupations)
    return float(
        result.energies.total
        + 6.33327186e-6 * temperature * degeneracy * np.sum(weights * entropy)
    )


_H2 = (Atom("H", np.array([1.6, 2.4, 2.2])), Atom("H", np.array([3.05, 2.55, 2.45])))


class SelfConsistentForceTests(unittest.TestCase):
    """Analytic forces against central differences of the converged free energy."""

    def check(self, problem, sampling=None, components=((1, 0), (1, 2)), step=1e-5, delta=2e-6):
        backend = periodic_backend(sampling)
        system = prepare_periodic_single_point(problem)
        result = backend.run_scf(system)
        self.assertTrue(result.converged)
        forces = backend.compute_forces(system, result, symmetrize=False)
        self.assertTrue(forces.periodic)
        for index, axis in components:
            energies = []
            for sign in (1.0, -1.0):
                moved = update_periodic_geometry(
                    system, _moved(problem.atoms, index, axis, sign * step)
                )
                energies.append(_free_energy(backend.run_scf(moved)))
            numeric = -(energies[0] - energies[1]) / (2.0 * step)
            self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=delta)
        return forces

    def test_gamma_orthorhombic(self) -> None:
        forces = self.check(_hydrogen_cell(5.0 * np.eye(3), _H2))
        self.assertTrue(np.any(forces.local.core_correction != 0.0))
        self.assertTrue(np.any(forces.nonlocal_pseudopotential != 0.0))

    def test_gamma_general_cell(self) -> None:
        lattice = [[5.0, 0.0, 0.0], [1.5, 5.0, 0.0], [0.0, 0.0, 5.0]]
        self.check(_hydrogen_cell(lattice, _H2), components=((1, 0), (0, 1)))

    def test_kpoints(self) -> None:
        reciprocal = 2.0 * np.pi / 5.0
        k_points = np.array([[0.0, 0.0, 0.25 * reciprocal], [0.0, 0.0, -0.25 * reciprocal]])
        sampling = (k_points, np.array([0.5, 0.5]))
        self.check(_hydrogen_cell(5.0 * np.eye(3), _H2), sampling, components=((1, 0), (1, 2)))

    def test_collinear_spin_at_gamma(self) -> None:
        atoms = (*_H2, Atom("H", np.array([2.3, 3.9, 1.2])))
        forces = self.check(
            _hydrogen_cell(5.0 * np.eye(3), atoms, spin=True), components=((2, 1),)
        )
        self.assertEqual(forces.unsymmetrized.shape, (3, 3))


class GeometryAndSymmetryTests(unittest.TestCase):
    def test_update_equals_a_fresh_preparation_and_wraps(self) -> None:
        problem = _hydrogen_cell(5.0 * np.eye(3), _H2)
        system = prepare_periodic_single_point(problem)
        moved_atoms = (Atom("H", np.array([-0.4, 2.4, 2.2])), _H2[1])
        moved = update_periodic_geometry(system, moved_atoms)
        fresh = prepare_periodic_single_point(
            SinglePointInput(
                atoms=list(moved_atoms),
                pseudopotentials=problem.pseudopotentials,
                grid=problem.grid,
                periodic_cell=problem.periodic_cell,
                scf=problem.scf,
                hartree=problem.hartree,
                eigensolver=problem.eigensolver,
            )
        )
        np.testing.assert_array_equal(moved.atoms[0].position, [4.6, 2.4, 2.2])
        np.testing.assert_array_equal(moved.input.atoms[0].position, [-0.4, 2.4, 2.2])
        for name in ("ionic_potential", "initial_density", "core_density"):
            np.testing.assert_array_equal(getattr(moved, name), getattr(fresh, name))
        self.assertEqual(moved.ion_ion_energy, fresh.ion_ion_energy)
        self.assertEqual(moved.alpha_z_energy, fresh.alpha_z_energy)
        np.testing.assert_array_equal(
            moved.nonlocal_operator.projectors.toarray(),
            fresh.nonlocal_operator.projectors.toarray(),
        )

    def test_operations_of_diamond(self) -> None:
        a = 10.26
        fractions = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
                     (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
        atoms = [Atom("Si", np.array(item) * a) for item in fractions]
        cell = PeriodicCell(lattice_vectors=a * np.eye(3))
        grid = build_periodic_grid(cell, PeriodicGridSettings(fortran_routines=True, spacing=0.64, expansion_order=4))
        operations = periodic_symmetry_operations(atoms, cell.lattice_vectors, grid)
        # The site group Td of an atom at the origin (no fractional translations).
        self.assertEqual(len(operations), 24)
        np.testing.assert_array_equal(operations[0].matrix, np.eye(3))
        fcc = periodic_symmetry_operations(
            [Atom("Si", np.zeros(3)), Atom("Si", 0.25 * a * np.ones(3))], _FCC
        )
        self.assertEqual(len(fcc), 24)
        # A k-point set without the symmetry of the cell keeps fewer operations.
        fewer = periodic_symmetry_operations(
            atoms, cell.lattice_vectors, grid, np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        )
        self.assertEqual(len(fewer), 1)

    def test_symmetrized_positions_remove_lattice_vectors(self) -> None:
        a = 10.26
        atoms = [Atom("Si", np.zeros(3)), Atom("Si", 0.25 * a * np.ones(3))]
        operations = periodic_symmetry_operations(atoms, _FCC)
        positions = np.array([atom.position for atom in atoms]) + 1e-9
        symmetric = symmetrize_positions(positions, operations)
        np.testing.assert_allclose(symmetric, positions - 1e-9, atol=2e-9)
        forces = np.array([[0.1, 0.2, -0.3], [-0.1, -0.2, 0.3]])
        # Td of the diamond site leaves no force on its atoms.
        np.testing.assert_allclose(symmetrize_forces(forces, operations), 0.0, atol=1e-15)


class IonicLoopTests(unittest.TestCase):
    def test_bfgs_relaxation_lowers_energy_and_forces(self) -> None:
        problem = _hydrogen_cell(5.0 * np.eye(3), _H2, species="H_POTRE.DAT")
        problem = SinglePointInput(
            atoms=list(problem.atoms),
            pseudopotentials=problem.pseudopotentials,
            grid=problem.grid,
            periodic_cell=problem.periodic_cell,
            scf=SCFSettings(max_iterations=80, number_of_states=4, convergence_criterion=1e-7),
            hartree=problem.hartree,
            eigensolver=EigensolverSettings(method="chebff", tolerance=1e-7),
        )
        system = prepare_periodic_single_point(problem)
        outcome = run_relaxation(
            system,
            RelaxationSettings(method="bfgs", max_movements=4, minimum_force=1e-3),
            periodic_backend(),
        )
        energies = [record.energy for record in outcome.records]
        self.assertGreaterEqual(len(energies), 2)
        self.assertLess(min(energies[1:]), energies[0])
        self.assertLess(outcome.records[-1].maximum_force, outcome.records[0].maximum_force)


_CLI_INPUT = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.0 5.0 5.0
end Cell_Shape
Grid_Spacing: 0.5
Expansion_Order: 4
Ignore_Symmetry: .true.
States_Num: 4
Fermi_Temp: 300.0
Max_Iter: 60
Convergence_Criterion: 1.0e-6
Mixing_Method: Anderson
Mixing_Param: 0.3
Correlation_Type: ca
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  1.6  2.4  2.2
  3.05 2.55 2.45
end Atom_Coord
"""


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")

    def run_input(self, extra: str) -> tuple[int, str]:
        from parsec_python.cli import main

        path = self.directory / "parsec.in"
        path.write_text(_CLI_INPUT + extra, encoding="utf-8")
        code = main([str(path), "--quiet", "--no-archive"])
        return code, (self.directory / "parsec.out").read_text(encoding="utf-8")

    def test_single_point_forces_print_the_periodic_tables(self) -> None:
        code, text = self.run_input("Skip_force: .false.\n")
        self.assertEqual(code, 0)
        self.assertIn(" (Ewald forces included)", text)
        self.assertIn("coordinates and tot. forces (after setting net force to zero)", text)
        self.assertNotIn("tot. torque", text)
        self.assertNotIn("implemented for isolated systems only", text)

    def test_relaxation_and_molecular_dynamics_run(self) -> None:
        code, text = self.run_input("Minimization: bfgs\nMovement_Num: 2\nForce_Min: 0.001\n")
        self.assertIn(code, (0, 4))
        self.assertTrue((self.directory / "atom.cor").is_file())
        self.assertTrue((self.directory / "relax_restart.dat").is_file())
        code, text = self.run_input(
            "Molecular_Dynamics: .true.\nStep_num: 2\nTinit: 300.0\nT_Final: 300.0\n"
            "T_Step: 0.0\nTime_Step: 20.0\n"
            "test_rng: .true.\n"
        )
        self.assertEqual(code, 0)
        lines = (self.directory / "md_nrg.dat").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)

    def test_molecular_dynamics_writes_levels_at_the_last_structure_only(self) -> None:
        """MD writes the table of levels once, for the last structure; PARSEC_MD_EIGENVALUES=1 at every iteration."""
        from unittest import mock

        md = (
            "Molecular_Dynamics: .true.\nStep_num: 1\nTinit: 300.0\nT_Final: 300.0\n"
            "T_Step: 0.0\nTime_Step: 20.0\ntest_rng: .true.\n"
        )
        header = "   State   Eigenvalue [Ry]"
        with mock.patch.dict(os.environ, {"PARSEC_MD_EIGENVALUES": ""}):
            code, held = self.run_input(md)
        self.assertEqual(code, 0)
        self.assertEqual(held.count("Self-consistency convergence achieved."), 2)
        self.assertGreater(held.count(" SCF iter # "), 2)
        self.assertEqual(held.count(header), 1)
        self.assertEqual(held.count(" Levels of the last SCF iteration:"), 1)
        with mock.patch.dict(os.environ, {"PARSEC_MD_EIGENVALUES": "1"}):
            code, every = self.run_input(md)
        self.assertEqual(every.count(header), every.count(" SCF iter # "))
        self.assertNotIn("Levels of the last SCF iteration", every)
        # a single point is not affected
        code, single = self.run_input("Skip_force: .false.\n")
        self.assertEqual(single.count(header), single.count(" SCF iter # "))

    def test_md_output_interval_thins_the_structures_and_the_force_tables(self) -> None:
        md = (
            "Molecular_Dynamics: .true.\nStep_num: 4\nTinit: 300.0\nT_Final: 300.0\n"
            "T_Step: 0.0\nTime_Step: 20.0\ntest_rng: .true.\n"
        )
        table = "coordinates and tot. forces (after setting net force to zero)"
        code, every = self.run_input(md)
        self.assertEqual(code, 0)
        blocks = lambda: (self.directory / "md_mech.dat").read_text(encoding="utf-8").count("Step #")
        nrg = lambda: len((self.directory / "md_nrg.dat").read_text(encoding="utf-8").splitlines())
        self.assertEqual(every.count(table), 5)
        full_blocks, full_lines = blocks(), nrg()
        code, thin = self.run_input(md + "MD_Output_Interval: 2\n")
        self.assertEqual(code, 0)
        # structures 0, 2 and 4 (the last) carry the force table; md_nrg.dat still has every step
        self.assertEqual(thin.count(table), 3)
        self.assertEqual(thin.count("Time for forces"), 5)
        self.assertEqual(nrg(), full_lines)
        self.assertLess(blocks(), full_blocks)
        self.assertGreaterEqual(blocks(), 3)
        self.assertEqual(thin.count(" Levels of the last SCF iteration:"), 1)

    def test_use_fortran_routines_false_runs_andrews_routines(self) -> None:
        from parsec_python.Input.parsec_input import parse_parsec_input

        text = _CLI_INPUT.replace("Use_Fortran_Routines: .true.", "Use_Fortran_Routines: .false.")
        self.assertIn("Use_Fortran_Routines: .false.", text)
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        translation = parse_parsec_input(path)
        self.assertFalse(translation.routines.fortran)
        self.assertIs(translation.routines.requested, False)
        # a keyword that only the Fortran-derived routines read still switches them on, with its reason
        path.write_text(text + "Periodic_Local_Potential: g_space\n", encoding="utf-8")
        translation = parse_parsec_input(path)
        self.assertTrue(translation.routines.fortran)
        # .true. is the Fortran-derived routines; unset is the choice of the default (Andrew Grieder's routines for
        # an input they read, as on main)
        path.write_text(text.replace("Use_Fortran_Routines: .false.\n", "Use_Fortran_Routines: .true.\n"), encoding="utf-8")
        self.assertTrue(parse_parsec_input(path).routines.fortran)
        path.write_text(text.replace("Use_Fortran_Routines: .false.\n", ""), encoding="utf-8")
        self.assertIsNone(parse_parsec_input(path).routines.requested)


class InputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")

    def parse(self, extra: str):
        from parsec_python.Input.parsec_input import parse_parsec_input

        path = self.directory / "parsec.in"
        path.write_text(_CLI_INPUT + extra, encoding="utf-8")
        return parse_parsec_input(path)

    def test_periodic_movements_are_accepted(self) -> None:
        self.assertEqual(self.parse("Minimization: bfgs\n").relaxation.method, "bfgs")
        self.assertIsNotNone(self.parse("Molecular_Dynamics: .true.\n").molecular_dynamics)
        self.assertTrue(self.parse("Skip_force: .false.\n").compute_forces)

    def test_combinations_with_periodic_forces(self) -> None:
        kpoints = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n1 1 2\nend Monkhorst_Pack_Grid\n"
        spin_k = kpoints + "Spin_Polarization: .true.\n"
        # Collinear spin with k-points and the double grid have periodic
        # forces: nothing is refused or reduced to energies.
        self.assertEqual(self.parse(spin_k + "Minimization: bfgs\n").relaxation.method, "bfgs")
        self.assertTrue(self.parse(spin_k + "Skip_force: .false.\n").compute_forces)
        self.assertIsNotNone(
            self.parse("Double_Grid_Order: 2\nMolecular_Dynamics: .true.\n").molecular_dynamics
        )
        translation = self.parse("Double_Grid_Order: 2\nSkip_force: .false.\n")
        self.assertTrue(translation.compute_forces)
        self.assertFalse(any("reports energies only" in item for item in translation.warnings))
        self.assertTrue(self.parse(kpoints + "Skip_force: .false.\n").compute_forces)

    def test_accelerated_cli_routes_periodic_forces(self) -> None:
        from dataclasses import replace

        from parsec_python.acceleration.cli import _reference_only_features

        kpoints = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n1 1 2\nend Monkhorst_Pack_Grid\n"
        for extra in ("Skip_force: .false.\n", "Minimization: bfgs\n", "Molecular_Dynamics: .true.\n"):
            # Gamma: the accelerated path has the forces of a bulk cell
            # (acceleration/Forces/periodic_device.py).
            features = _reference_only_features(
                self.parse(extra), accelerated_backend_selected=False
            )
            self.assertEqual(features, [], extra)
            # k-points (the reference drivers on the device workers) and an
            # energy refinement stay with the reference solver, by name.
            features = _reference_only_features(
                self.parse(kpoints + extra), accelerated_backend_selected=False
            )
            self.assertTrue(any("k-point sampling" in item for item in features), extra)
            refined = replace(self.parse(extra), energy_refinement="b3lyp")
            features = _reference_only_features(refined, accelerated_backend_selected=False)
            self.assertTrue(
                any("periodic cell together with Energy_Refinement" in item for item in features),
                extra,
            )
        self.assertFalse(
            any(
                "relaxation or molecular dynamics" in item
                for item in _reference_only_features(
                    self.parse(""), accelerated_backend_selected=False
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
