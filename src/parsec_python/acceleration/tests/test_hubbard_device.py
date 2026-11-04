"""LDA+U and the TS correction on the accelerated driver.

The LDA+U term enters the device Hamiltonians as fifteen KB-like columns
per atom whose coefficients follow ``D`` (``Hamiltonian/hubbard_device.py``);
the occupation matrices and forces are formed where the orbitals are.  The
TS Hirshfeld sums run on a device (``Energy/dispersion_device.py``).  Every
result is compared with the reference solver on the same input.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Hamiltonian.hubbard import (
    HubbardOperator,
    build_hubbard_projectors,
    hubbard_terms,
    occupation_matrices,
)
from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.Hamiltonian.hubbard_device import (
    EXPANSION_COLUMNS,
    augmented_nonlocal_operator,
    expansion_coefficients,
    expansion_vectors,
)


DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

_AG_INPUT = """
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0
Hartree_Boundary_Tolerance: off
Grid_Spacing: 0.7
Coordinate_Unit: Cartesian_Bohr
States_Num: 16
Fermi_Temp: 1000.0
Correlation_Type: ca
Convergence_Criterion: 1.0e-7
Max_Iter: {iterations}
Eigensolver: chebff
Mixing_Param: 0.2
{extra}
Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
{ag_line}
begin Atom_Coord
  0.0 0.0 {first}
  0.0 0.0 2.4
end Atom_Coord
"""

_H2_INPUT = """
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 6.0
Hartree_Boundary_Tolerance: off
Grid_Spacing: 0.6
Coordinate_Unit: Cartesian_Bohr
States_Num: 4
Correlation_Type: pbe
Ignore_Symmetry: .true.
Van_Der_Waals: .true.
Convergence_Criterion: 1.0e-6
Max_Iter: 40
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 -0.7
  0.0 0.0 0.7
  0.3 0.2 3.9
  0.3 0.2 5.3
end Atom_Coord
"""


_ROOT: tempfile.TemporaryDirectory | None = None


def setUpModule():
    global _ROOT
    _ROOT = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def tearDownModule():
    if _ROOT is not None:
        _ROOT.cleanup()


def _folder() -> str:
    """A new folder that lives until the end of the module's tests."""

    return tempfile.mkdtemp(dir=_ROOT.name)


def _ag_case(directory, *, iterations=5, first=-2.4, extra="", ag_line="ldaplusu_u 4.0 eV"):
    path = Path(directory) / "parsec.in"
    path.write_text(
        _AG_INPUT.format(iterations=iterations, first=first, extra=extra, ag_line=ag_line)
    )
    (Path(directory) / "Ag_POTRE.DAT").write_bytes((DATA / "Ag_POTRE.DAT").read_bytes())
    return path


def _h2_case(directory, extra=""):
    path = Path(directory) / "parsec.in"
    path.write_text(_H2_INPUT.format(extra=extra))
    (Path(directory) / "H_POTRE.DAT").write_bytes((DATA / "spin_pbe_H_POTRE.DAT").read_bytes())
    return path


class _Folder:
    def __enter__(self) -> str:
        return _folder()

    def __exit__(self, *_exc) -> None:
        return None


def _parse(path):
    return parse_parsec_input(path).problem


class ExpansionTest(unittest.TestCase):
    def test_fifteen_terms_rebuild_a_symmetric_matrix(self):
        rng = np.random.default_rng(3)
        matrices = rng.normal(size=(4, 5, 5))
        matrices = matrices + np.swapaxes(matrices, 1, 2)
        coefficients = expansion_coefficients(matrices).reshape(4, EXPANSION_COLUMNS)
        vectors = expansion_vectors()
        rebuilt = np.einsum("ak,km,kn->amn", coefficients, vectors, vectors)
        np.testing.assert_allclose(rebuilt, matrices, rtol=0.0, atol=1.0e-13)

    def test_augmented_factor_applies_the_kb_and_the_lda_u_terms(self):
        from parsec_python.SCF.single_point import prepare_single_point

        with _Folder() as directory:
            system = prepare_single_point(_parse(_ag_case(directory)))
        projectors = system.hubbard
        kb = system.nonlocal_operator
        augmented = augmented_nonlocal_operator(kb, projectors)
        count = len(kb.signs)
        self.assertEqual(augmented.projectors.shape[1], count + EXPANSION_COLUMNS * projectors.atom_count)
        self.assertEqual(augmented.labels[:count], kb.labels)
        rng = np.random.default_rng(5)
        occupations = rng.uniform(0.0, 1.0, size=(projectors.atom_count, 5, 5))
        occupations = 0.5 * (occupations + np.swapaxes(occupations, 1, 2))
        potentials = hubbard_terms(projectors, [occupations]).potentials[0]
        signs = augmented.signs.copy()
        signs[count:] = expansion_coefficients(potentials)
        vectors = rng.normal(size=(system.grid.size, 3))
        factor = augmented.projectors
        applied = factor @ (signs[:, None] * (factor.T @ vectors))
        expected = kb.apply(vectors) + HubbardOperator(projectors, potentials).apply(vectors)
        np.testing.assert_allclose(applied, expected, rtol=0.0, atol=1.0e-12)

    def test_the_j_term_is_refused(self):
        with _Folder() as directory:
            problem = _parse(
                _ag_case(directory, ag_line="ldaplusu_u 4.0 eV\nldaplusu_j 0.5 eV")
            )
        with self.assertRaises(NotImplementedError):
            driver.prepare_single_point(problem, backend="scipy", symmetry="off")


def _reference(problem):
    from parsec_python.Forces import compute_forces
    from parsec_python.SCF.single_point import prepare_single_point, run_scf

    system = prepare_single_point(problem)
    result = run_scf(system)
    return system, result, compute_forces(system, result)


def _accelerated(problem, backend, symmetry):
    system = driver.prepare_single_point(problem, backend=backend, symmetry=symmetry)
    system.materialize_final_wavefunctions = False
    result = driver.run_scf(system)
    return system, result, driver.compute_forces(system, result)


class HostTermTest(unittest.TestCase):
    """The CPU backends add the host term to their bound Hamiltonian."""

    def test_scipy_backend_follows_the_reference_loop(self):
        with _Folder() as directory:
            problem = _parse(_ag_case(directory, first=-2.6))
        _system, expected, expected_forces = _reference(problem)
        _system, result, forces = _accelerated(problem, "scipy", "off")
        self.assertIsNone(_system.hubbard_device)
        # The first filters of the two paths differ by a few 1e-9 Ry already
        # without LDA+U.
        for step, reference in zip(result.history, expected.history):
            self.assertAlmostEqual(step.energies.total, reference.energies.total, delta=5.0e-8)
        np.testing.assert_allclose(result.eigenvalues, expected.eigenvalues, atol=1.0e-7)
        np.testing.assert_allclose(forces.hubbard, expected_forces.hubbard, atol=1.0e-7)
        np.testing.assert_allclose(forces.total, expected_forces.total, atol=1.0e-7)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceTermTest(unittest.TestCase):
    def test_full_grid_occupations_and_action_equal_the_reference(self):
        import cupy as cp

        from parsec_python.SCF.single_point import prepare_single_point

        with _Folder() as directory:
            problem = _parse(_ag_case(directory, first=-2.6))
        reference = prepare_single_point(problem)
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
        device = system.hubbard_device
        self.assertIsNotNone(device)
        rng = np.random.default_rng(7)
        orbitals, _ = np.linalg.qr(rng.normal(size=(reference.grid.size, 12)))
        weights = rng.uniform(0.0, 1.0, size=12)
        matrices = device.occupation_matrices(cp.asarray(orbitals, order="F"), weights)
        expected = occupation_matrices(reference.hubbard, orbitals, weights)
        np.testing.assert_allclose(matrices, expected, rtol=0.0, atol=1.0e-13)
        potentials = hubbard_terms(reference.hubbard, [expected]).potentials[0]
        term = HubbardOperator(reference.hubbard, potentials)
        potential = np.asarray(reference.ionic_potential)
        bound = system.hamiltonian(potential, term)
        vectors = rng.normal(size=(reference.grid.size, 3))
        applied = bound.apply(vectors)
        wanted = reference.hamiltonian(potential, term).apply(vectors)
        np.testing.assert_allclose(applied, wanted, rtol=0.0, atol=1.0e-10)
        # D = 0 again: the KB Hamiltonian alone.
        plain = system.hamiltonian(potential).apply(vectors)
        np.testing.assert_allclose(
            plain, reference.hamiltonian(potential).apply(vectors), rtol=0.0, atol=1.0e-10
        )

    def test_host_and_hand_over_bases_give_the_occupations_of_their_orbitals(self):
        import cupy as cp

        from parsec_python.acceleration.Eigensolvers.handover import RitzHandover

        with _Folder() as directory:
            problem = _parse(_ag_case(directory, first=-2.6))
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
        device = system.hubbard_device
        rng = np.random.default_rng(17)
        basis, _ = np.linalg.qr(rng.normal(size=(system.grid.size, 10)))
        rotation, _ = np.linalg.qr(rng.normal(size=(10, 10)))
        ritz = basis @ rotation
        weights = np.array([1.0, 1.0, 1.0, 0.75, 0.25, 0.0])
        expected = occupation_matrices(system.hubbard, ritz[:, :6], weights)
        # Orbitals spilled to the host: uploaded a chunk at a time.
        host = device.occupation_matrices(np.asfortranarray(ritz), np.concatenate((weights, np.zeros(4))))
        np.testing.assert_allclose(host, expected, rtol=0.0, atol=1.0e-13)
        # A basis that holds the triangular hand-over: Q with the Ritz
        # coefficients beside it.
        record = RitzHandover(coefficients=cp.asarray(rotation), device=0, columns=10)
        occupation, _slopes = device._products(
            None, cp.asarray(basis, order="F"), weights, record
        )
        np.testing.assert_allclose(occupation, expected, rtol=0.0, atol=1.0e-12)

    def test_full_grid_loop_forces_and_spin_follow_the_reference(self):
        for extra in ("", "Spin_Polarization: .true.\nInitial_Spin_Polarization: 0.4"):
            with self.subTest(spin=bool(extra)):
                with _Folder() as directory:
                    problem = _parse(_ag_case(directory, first=-2.6, extra=extra))
                _system, expected, expected_forces = _reference(problem)
                _system, result, forces = _accelerated(problem, "cupy", "off")
                # The first filters of the two paths differ by a few 1e-9 Ry
                # already without LDA+U.
                for step, reference in zip(result.history, expected.history):
                    self.assertAlmostEqual(
                        step.energies.total, reference.energies.total, delta=5.0e-8
                    )
                    if reference.hubbard_energy is not None:
                        self.assertAlmostEqual(
                            step.hubbard_energy, reference.hubbard_energy, delta=5.0e-8
                        )
                np.testing.assert_allclose(result.eigenvalues, expected.eigenvalues, atol=1.0e-7)
                np.testing.assert_allclose(forces.hubbard, expected_forces.hubbard, atol=1.0e-7)
                np.testing.assert_allclose(forces.total, expected_forces.total, atol=1.0e-7)

    def test_sector_occupations_are_those_of_the_full_grid_orbitals(self):
        from parsec_python.acceleration.SCF.single_point import materialize_wavefunctions

        with _Folder() as directory:
            problem = _parse(_ag_case(directory, iterations=3))
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="on")
        self.assertIsNotNone(getattr(system.backend, "symmetry_eigensolver", None))
        system.materialize_final_wavefunctions = False
        result = driver.run_scf(system)
        sectors = system.hubbard_device.occupation_matrices(
            result.wavefunctions, result.occupations
        )
        materialize_wavefunctions(system, result)
        expected = occupation_matrices(system.hubbard, result.wavefunctions, result.occupations)
        np.testing.assert_allclose(sectors, expected, rtol=0.0, atol=1.0e-12)

    def test_sector_loop_converges_to_the_reference_state(self):
        with _Folder() as directory:
            problem = _parse(_ag_case(directory, iterations=60))
        _system, expected, expected_forces = _reference(problem)
        system, result, forces = _accelerated(problem, "cupy", "on")
        self.assertTrue(result.converged and expected.converged)
        self.assertAlmostEqual(result.energies.total, expected.energies.total, delta=1.0e-7)
        np.testing.assert_allclose(result.eigenvalues, expected.eigenvalues, atol=1.0e-6)
        np.testing.assert_allclose(forces.total, expected_forces.total, atol=1.0e-6)

    def test_moved_atoms_take_new_projectors(self):
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF import update_geometry
        from parsec_python.SCF.single_point import prepare_single_point, run_scf
        from parsec_python.models import Atom

        with _Folder() as directory:
            problem = _parse(_ag_case(directory, iterations=60))
        # The sectors were chosen for the first structure: the move keeps them.
        moved_atoms = (Atom("Ag", (0.0, 0.0, -2.5)), Atom("Ag", (0.0, 0.0, 2.5)))
        reference = update_geometry(prepare_single_point(problem), moved_atoms)
        expected = run_scf(reference)
        expected_forces = compute_forces(reference, expected)
        for symmetry in ("off", "on"):
            with self.subTest(symmetry=symmetry):
                system = driver.prepare_single_point(problem, backend="cupy", symmetry=symmetry)
                system.materialize_final_wavefunctions = False
                moved = driver.update_geometry(system, moved_atoms)
                result = driver.run_scf(moved)
                forces = driver.compute_forces(moved, result)
                self.assertTrue(result.converged)
                self.assertAlmostEqual(result.energies.total, expected.energies.total, delta=1.0e-7)
                np.testing.assert_allclose(forces.hubbard, expected_forces.hubbard, atol=1.0e-6)
                np.testing.assert_allclose(forces.total, expected_forces.total, atol=1.0e-6)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DispersionDeviceTest(unittest.TestCase):
    def test_device_volumes_equal_the_reference(self):
        from parsec_python.Energy.dispersion import hirshfeld_volumes
        from parsec_python.SCF.single_point import prepare_single_point
        from parsec_python.acceleration.Energy import device_hirshfeld_volumes

        with _Folder() as directory:
            problem = _parse(_h2_case(directory))
        system = prepare_single_point(problem)
        rng = np.random.default_rng(11)
        density = system.initial_density * rng.uniform(0.5, 1.5, size=system.grid.size)
        arguments = (system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials, density)
        expected = hirshfeld_volumes(*arguments)
        volumes = device_hirshfeld_volumes(*arguments)
        np.testing.assert_allclose(volumes.free, expected.free, rtol=1.0e-12)
        np.testing.assert_allclose(volumes.effective, expected.effective, rtol=1.0e-12)

    def test_device_pair_sum_equals_the_reference(self):
        from parsec_python.Energy.dispersion import ts_pair_energy
        from parsec_python.acceleration.Energy import device_ts_pair_energy

        rng = np.random.default_rng(13)
        positions = rng.uniform(-8.0, 8.0, size=(37, 3))
        c6 = rng.uniform(2.0, 50.0, size=37)
        alpha = rng.uniform(3.0, 12.0, size=37)
        radius = rng.uniform(2.5, 4.0, size=37)
        energy, forces, _shells = ts_pair_energy(positions, c6, alpha, radius)
        device_energy, device_forces, shells = device_ts_pair_energy(positions, c6, alpha, radius)
        self.assertEqual(shells, 0)
        self.assertAlmostEqual(device_energy, energy, delta=1.0e-14 * abs(energy) + 1.0e-18)
        np.testing.assert_allclose(device_forces, forces, rtol=0.0, atol=1.0e-14 * np.abs(forces).max())

    def test_loop_energy_and_forces_follow_the_reference(self):
        with _Folder() as directory:
            problem = _parse(_h2_case(directory))
        _system, expected, expected_forces = _reference(problem)
        system, result, forces = _accelerated(problem, "cupy", "off")
        self.assertEqual(
            system.dispersion_device.__module__,
            "parsec_python.acceleration.Energy.dispersion_device",
        )
        self.assertAlmostEqual(result.dispersion.energy, expected.dispersion.energy, delta=1.0e-12)
        self.assertAlmostEqual(result.energies.total, expected.energies.total, delta=1.0e-9)
        np.testing.assert_allclose(forces.dispersion, expected_forces.dispersion, atol=1.0e-12)
        np.testing.assert_allclose(forces.total, expected_forces.total, atol=1.0e-8)


class RoutingTest(unittest.TestCase):
    def test_what_the_accelerated_driver_lacks_goes_to_the_reference_solver(self):
        from parsec_python.acceleration.cli import _reference_only_features

        with _Folder() as directory:
            plain = parse_parsec_input(_ag_case(directory))
            with_j = parse_parsec_input(
                _ag_case(directory, ag_line="ldaplusu_u 4.0 eV\nldaplusu_j 0.5 eV")
            )
            dispersion = parse_parsec_input(_h2_case(directory))
        self.assertEqual(_reference_only_features(plain, accelerated_backend_selected=False), [])
        self.assertEqual(
            _reference_only_features(dispersion, accelerated_backend_selected=False), []
        )
        self.assertIn(
            "LDA+U with ldaplusu_j != 0",
            _reference_only_features(with_j, accelerated_backend_selected=False),
        )

    def test_mpi_and_j_cases_are_refused_by_the_driver(self):
        from parsec_python.acceleration.driver import _refuse_corrections

        with _Folder() as directory:
            problem = _parse(_ag_case(directory))
            with_j = _parse(_ag_case(directory, ag_line="ldaplusu_u 4.0 eV\nldaplusu_j 0.5 eV"))
            dispersion = _parse(_h2_case(directory))
        with self.assertRaises(NotImplementedError):
            _refuse_corrections(problem, is_periodic=False, mpi_context=object())
        with self.assertRaises(NotImplementedError):
            _refuse_corrections(with_j, is_periodic=True, mpi_context=None)
        # A bulk cell at the Gamma point runs on the accelerated driver.
        _refuse_corrections(problem, is_periodic=True, mpi_context=None)
        _refuse_corrections(dispersion, is_periodic=True, mpi_context=None)
        _refuse_corrections(problem, is_periodic=False, mpi_context=None)
        _refuse_corrections(dispersion, is_periodic=False, mpi_context=object())


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class CommandLineTest(unittest.TestCase):
    def test_forces_run_on_the_accelerated_path(self):
        from parsec_python.acceleration.cli import main

        with _Folder() as directory:
            path = _ag_case(directory, first=-2.6, iterations=60, extra="Skip_Force: .false.")
            log = Path(directory) / "accelerated.out"
            errors = StringIO()
            with redirect_stdout(StringIO()), redirect_stderr(errors):
                code = main([str(path), "--log", str(log), "--no-archive", "--quiet"])
            self.assertEqual(code, 0, errors.getvalue())
            self.assertNotIn("reference parsec_python.cli solver", errors.getvalue())
            text = log.read_text()
        self.assertIn("On-site Coulomb interaction (LDA+U) messages:", text)
        self.assertIn("LDA+U Energy", text)
        self.assertIn("Reporting LDA+U force components", text)


if __name__ == "__main__":
    unittest.main()
