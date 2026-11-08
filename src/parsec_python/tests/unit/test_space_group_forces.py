"""Forces of periodic cells symmetrized over the space group, and with spin and k-points.

``symm_forces.f90`` averages the forces over Fortran's space group ``symm``,
non-symmorphic operations included; the forces of a k-point wedge (an
irreducible ``Monkhorst_Pack_Shift`` grid or a weighted ``Kpoint_List``) are
right only after that average, which completes the k-point sum over the
stars.  The tests check the operations, the position average of a
relaxation, the wedge forces against the forces of the full grid, the forces
of collinear spin with k-points against finite differences of the free
energy, and the input and command-line paths.  The Fortran comparison is in
the report of the parity round.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces import (
    compute_forces,
    periodic_force_operations,
    periodic_symmetry_operations,
    space_group_symmetry_operations,
    symmetrize_forces,
    symmetrize_positions,
)
import parsec_python.Forces.periodic as periodic
from parsec_python.Grid.kpoints import kpoint_sampling, kpoint_sampling_details
from parsec_python.Grid.space_group import (
    find_space_group,
    fortran_reciprocal_lattice,
    space_group_atom_images,
)
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
from parsec_python.SCF import prepare_periodic_single_point, run_scf_kpoints
from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized
from parsec_python.SCF.pbc import update_periodic_geometry

DATA = Path(__file__).resolve().parents[1] / "data"
A = 10.26
FCC = 0.5 * A * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
# Diamond with the two atoms moved along z in opposite senses: a group of
# eight operations, four with the fractional translation (1/4, 1/4, 1/4).
DISPLACED_DIAMOND = (
    Atom("Si", np.array([0.0, 0.0, 0.1539])),
    Atom("Si", 0.25 * A * np.ones(3) - np.array([0.0, 0.0, 0.1539])),
)


def _apply(operation, positions):
    return (positions - operation.translation) @ operation.matrix.T


class SpaceGroupOperationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.group = find_space_group(FCC, DISPLACED_DIAMOND)
        self.operations = space_group_symmetry_operations(DISPLACED_DIAMOND, FCC, self.group)

    def test_operations_carry_the_fractional_translations(self) -> None:
        self.assertEqual(len(self.operations), 8)
        np.testing.assert_array_equal(self.operations[0].matrix, np.eye(3))
        np.testing.assert_array_equal(self.operations[0].translation, 0.0)
        exchanging = [op for op in self.operations if op.image[0] == 1]
        self.assertEqual(len(exchanging), 4)
        positions = np.array([atom.position for atom in DISPLACED_DIAMOND])
        for operation in self.operations:
            # T(R_a) = R_image(a) + a lattice vector.
            mapped = _apply(operation, positions)
            offset = np.linalg.solve(FCC.T, (mapped - positions[operation.image]).T).T
            np.testing.assert_allclose(offset, np.round(offset), atol=1e-10)
        for operation in exchanging:
            self.assertGreater(np.linalg.norm(operation.translation), 1.0)
        np.testing.assert_array_equal(
            space_group_atom_images(self.group, FCC, DISPLACED_DIAMOND),
            np.array([op.image for op in self.operations]),
        )

    def test_forces_and_positions_keep_the_group(self) -> None:
        forces = np.array([[0.1, -0.2, 0.3], [0.05, 0.15, -0.4]])
        symmetric = symmetrize_forces(forces, self.operations)
        # Only opposite forces along z survive the group.
        np.testing.assert_allclose(symmetric[:, :2], 0.0, atol=1e-15)
        self.assertAlmostEqual(symmetric[0, 2], -symmetric[1, 2], places=15)
        self.assertAlmostEqual(symmetric[0, 2], 0.35, places=14)
        positions = np.array([atom.position for atom in DISPLACED_DIAMOND])
        np.testing.assert_allclose(
            symmetrize_positions(positions, self.operations), positions, atol=1e-13
        )
        noisy = positions + np.array([[1e-7, -2e-7, 3e-7], [4e-7, 1e-7, -5e-7]])
        averaged = symmetrize_positions(noisy, self.operations)
        for operation in self.operations:
            mapped = _apply(operation, averaged)
            offset = np.linalg.solve(FCC.T, (mapped - averaged[operation.image]).T).T
            np.testing.assert_allclose(offset, np.round(offset), atol=1e-12)

    def test_an_atom_without_image_is_an_error(self) -> None:
        moved = (DISPLACED_DIAMOND[0], Atom("Si", DISPLACED_DIAMOND[1].position + 0.01))
        with self.assertRaisesRegex(ValueError, "411"):
            space_group_atom_images(self.group, FCC, moved)

    def test_default_operations(self) -> None:
        problem = _cubic_problem()
        plain = prepare_periodic_single_point(problem)
        expected = periodic_symmetry_operations(
            plain.atoms, problem.periodic_cell.lattice_vectors, plain.grid
        )
        found = periodic_force_operations(plain)
        self.assertEqual(len(found), len(expected))
        for left, right in zip(found, expected):
            np.testing.assert_array_equal(left.matrix, right.matrix)
            self.assertIsNone(left.translation)
        with_group = prepare_periodic_single_point(_cubic_problem(space_group=_cubic_group()))
        operations = periodic_force_operations(with_group)
        self.assertEqual(len(operations), 4)
        self.assertTrue(any(np.any(op.translation != 0.0) for op in operations))


# A cubic cell (orthorhombic grid: the stencil and k.grad terms are invariant
# under the signed axis permutations) with a two-fold screw axis along z and
# an inversion centre off the origin.
CUBE = 7.0 * np.eye(3)
SCREW_PAIR = (
    Atom("Si", np.array([0.9, 1.4, 0.35])),
    Atom("Si", np.array([6.1, 5.6, 3.85])),
)


def _cubic_group():
    return find_space_group(CUBE, SCREW_PAIR)


def _cubic_problem(spin=False, **fields):
    return SinglePointInput(
        atoms=list(SCREW_PAIR),
        pseudopotentials={"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)},
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=CUBE),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=8,
            convergence_criterion=1e-10,
            fermi_temperature_kelvin=300.0,
            spin_polarized=spin,
            symmetrize_density=fields.pop("symmetrize", False),
        ),
        hartree=HartreeSettings(
            relative_tolerance=1e-12, absolute_tolerance=1e-14, max_iterations=4000
        ),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1e-10),
        **fields,
    )


class WedgeForceTests(unittest.TestCase):
    """Wedge forces (completed over the group) equal those of the full grid."""

    def compare(self, spin: bool) -> None:
        group = _cubic_group()
        self.assertEqual(group.order, 4)
        driver = run_scf_kpoints_spin_polarized if spin else run_scf_kpoints
        reduced_problem = _cubic_problem(
            spin,
            symmetrize=True,
            monkhorst_pack_dimensions=(3, 1, 3),
            monkhorst_pack_shift=(0.0, 0.0, 0.0),
            space_group=group,
        )
        system = prepare_periodic_single_point(reduced_problem)
        k_reduced, w_reduced = kpoint_sampling(reduced_problem)
        self.assertEqual(len(w_reduced), 4)
        reduced = driver(system, k_reduced, w_reduced, keep_wavefunctions=True)
        wedge = compute_forces(system, reduced, symmetrize=False)
        details = kpoint_sampling_details(reduced_problem)
        stars = []
        for k, count in zip(details.fractional, details.counts):
            members = {}
            for g in group.reciprocal_rotations.astype(float):
                for sign in (1.0, -1.0):
                    q = sign * (g @ k)
                    members.setdefault(tuple(np.round(np.mod(q + 1e-9, 1.0), 6)), q)
            self.assertEqual(len(members), count)
            stars.extend(members.values())
        b = fortran_reciprocal_lattice(CUBE)
        k_full = np.array([b @ q for q in stars])
        plain = prepare_periodic_single_point(_cubic_problem(spin))
        full = driver(plain, k_full, np.full(len(stars), 1.0 / len(stars)), keep_wavefunctions=True)
        self.assertAlmostEqual(reduced.energies.total, full.energies.total, delta=1e-9)
        reference = compute_forces(plain, full, symmetrize=False)
        np.testing.assert_allclose(wedge.unsymmetrized, reference.unsymmetrized, atol=5e-8)
        symmetrized = compute_forces(system, reduced)
        self.assertEqual(symmetrized.symmetry_operations, 4)
        np.testing.assert_allclose(symmetrized.symmetrized, reference.unsymmetrized, atol=2e-7)
        # Without the completion the wedge alone is far off.
        saved = periodic.reduced_kpoint_operations
        periodic.reduced_kpoint_operations = lambda system, result: None
        try:
            raw = compute_forces(system, reduced, symmetrize=False)
        finally:
            periodic.reduced_kpoint_operations = saved
        self.assertGreater(
            np.max(np.abs(raw.nonlocal_pseudopotential - reference.nonlocal_pseudopotential)),
            1e-3,
        )

    def test_scalar_kpoints(self) -> None:
        self.compare(spin=False)

    def test_collinear_spin_kpoints(self) -> None:
        self.compare(spin=True)


_H3 = (
    Atom("H", np.array([1.6, 2.4, 2.2])),
    Atom("H", np.array([3.05, 2.55, 2.45])),
    Atom("H", np.array([2.3, 3.9, 1.2])),
)


def _spin_hydrogen():
    return SinglePointInput(
        atoms=list(_H3),
        pseudopotentials={"H": SpeciesPotential(DATA / "H_CORE_POTRE.DAT", 0)},
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=5.0 * np.eye(3)),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=4,
            convergence_criterion=1e-10,
            spin_polarized=True,
            fermi_temperature_kelvin=300.0,
        ),
        hartree=HartreeSettings(
            relative_tolerance=1e-12, absolute_tolerance=1e-14, max_iterations=4000
        ),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1e-10),
    )


class SpinKpointForceTests(unittest.TestCase):
    """Collinear spin with k-points: forces of ``run_scf_kpoints_spin_polarized``."""

    @classmethod
    def setUpClass(cls) -> None:
        reciprocal = 2.0 * np.pi / 5.0
        cls.k_points = np.array(
            [[0.0, 0.0, 0.25 * reciprocal], [0.1 * reciprocal, 0.0, -0.25 * reciprocal]]
        )
        cls.weights = np.array([0.3, 0.7])
        cls.backend = periodic_backend((cls.k_points, cls.weights))
        cls.system = prepare_periodic_single_point(_spin_hydrogen())
        cls.result = cls.backend.run_scf(cls.system)

    def free_energy(self, result, temperature=300.0) -> float:
        per_k = len(result.occupations_up) // len(self.weights)
        weights = np.repeat(self.weights, per_k)
        entropy = 0.0
        for occupations in (result.occupations_up, result.occupations_down):
            f = np.clip(np.asarray(occupations), 1e-300, 1 - 1e-16)
            entropy += np.sum(weights * (f * np.log(f) + (1 - f) * np.log(1 - f)))
        return float(result.energies.total + 6.33327186e-6 * temperature * entropy)

    def test_forces_are_the_derivative_of_the_free_energy(self) -> None:
        self.assertTrue(self.result.converged)
        self.assertAlmostEqual(self.result.magnetic_moment, 1.0, places=6)
        forces = self.backend.compute_forces(self.system, self.result, symmetrize=False)
        self.assertTrue(forces.periodic)
        step = 1e-5
        for index, axis in ((2, 1), (1, 0)):
            values = []
            for sign in (1.0, -1.0):
                moved = list(_H3)
                position = np.array(moved[index].position, dtype=float)
                position[axis] += sign * step
                moved[index] = Atom("H", position)
                other = update_periodic_geometry(self.system, tuple(moved))
                values.append(self.free_energy(self.backend.run_scf(other)))
            numeric = -(values[0] - values[1]) / (2.0 * step)
            self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=2e-8)

    def test_warm_start_of_a_moved_structure(self) -> None:
        moved = list(_H3)
        moved[2] = Atom("H", np.array([2.35, 3.9, 1.2]))
        other = update_periodic_geometry(self.system, tuple(moved))
        cold = self.backend.run_scf(other)
        warm = self.backend.run_scf(
            other, warm_start=self.backend.warm_start(self.system, self.result, other)
        )
        self.assertTrue(warm.converged)
        self.assertLess(warm.iterations, cold.iterations)
        self.assertAlmostEqual(warm.energies.total, cold.energies.total, delta=1e-9)


_CLI_INPUT = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.0 5.0 5.0
end Cell_Shape
Grid_Spacing: 0.5
Expansion_Order: 4
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
  2.3  3.9  1.2
end Atom_Coord
"""

_KPOINTS = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n1 1 2\nend Monkhorst_Pack_Grid\n"


class InputAndCommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")

    def write(self, extra: str, base: str = _CLI_INPUT) -> Path:
        path = self.directory / "parsec.in"
        path.write_text(base + extra, encoding="utf-8")
        return path

    def parse(self, extra: str, base: str = _CLI_INPUT):
        from parsec_python.Input.parsec_input import parse_parsec_input

        return parse_parsec_input(self.write(extra, base))

    def test_spin_kpoint_forces_and_relaxation_on_the_command_line(self) -> None:
        from parsec_python.cli import main

        path = self.write("Spin_Polarization: .true.\n" + _KPOINTS + "Skip_force: .false.\n")
        self.assertEqual(main([str(path), "--quiet"]), 0)
        text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn(" (Ewald forces included)", text)
        self.assertIn("coordinates and tot. forces (after setting net force to zero)", text)
        self.assertIn("Converged magnetic moment", text)
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            self.assertEqual(archive["forces_total"].shape, (3, 3))
            self.assertIn("density_up_e_per_bohr3", archive.files)
        path = self.write(
            "Spin_Polarization: .true.\n" + _KPOINTS + "Minimization: bfgs\nMovement_Num: 1\n"
            "Force_Min: 0.0001\n"
        )
        self.assertIn(main([str(path), "--quiet", "--no-archive"]), (0, 4))
        self.assertIn("Step #   1", (self.directory / "atom.cor").read_text(encoding="utf-8"))

    def test_reduced_kpoint_sets_and_movements(self) -> None:
        from parsec_python.Input.parsec_input import UnsupportedParsecOptionError

        cubic = _CLI_INPUT.replace(
            "  1.6  2.4  2.2\n  3.05 2.55 2.45\n  2.3  3.9  1.2\n",
            "  1.0 1.0 1.0\n  4.0 4.0 4.0\n",
        )
        shift = (
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 2\nend Monkhorst_Pack_Grid\n"
            "begin Monkhorst_Pack_Shift\n0.5 0.5 0.5\nend Monkhorst_Pack_Shift\n"
        )
        relaxation = self.parse(shift + "Minimization: bfgs\n", cubic)
        self.assertGreater(relaxation.problem.space_group.order, 1)
        # Molecular dynamics and manual movements use no symmetry
        # (parsec.F90 ierr = 2): the wedge of the identity.
        dynamics = self.parse(shift + "Molecular_Dynamics: .true.\n", cubic)
        self.assertEqual(dynamics.problem.space_group.order, 1)
        # The group of a run PARSEC does not search ("skipping non-trivial
        # symmetry operations"; before a fix of the parity round the identity of
        # Ignore_Symmetry, whose report lines name that flag).
        self.assertFalse(dynamics.problem.space_group.ignored)
        self.assertEqual(dynamics.problem.space_group.crystal_system, "")
        # A weighted list with forces gets the space group its forces need.
        weighted = (
            "Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0 1\n0.25 0 0 3\nend Kpoint_List\n"
        )
        single = self.parse(weighted + "Skip_force: .false.\n", cubic)
        self.assertGreater(single.problem.space_group.order, 1)
        self.assertTrue(single.compute_forces)
        self.assertTrue(any("completed over the space group" in w for w in single.warnings))
        self.assertIsNone(self.parse(weighted, cubic).problem.space_group)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Molecular_Dynamics"):
            self.parse(weighted + "Molecular_Dynamics: .true.\n", cubic)
        # Movable flags that the group exchanges with fixed atoms.
        flagged = cubic.replace("Local_Component: s\n", "Local_Component: s\nMove_Flag: 1\n")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Move_Flag"):
            self.parse(shift + "Minimization: bfgs\n", flagged)


if __name__ == "__main__":
    unittest.main()
