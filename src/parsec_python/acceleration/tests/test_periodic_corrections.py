"""LDA+U and the TS correction of periodic cells on the devices.

A bulk cell at the Gamma point runs on the accelerated driver
(``Hamiltonian/hubbard_device.py`` with the images of the cell,
``Energy/dispersion_device.py`` with the Hirshfeld cells and lattice
shells).  The k-point drivers of the reference solver apply the LDA+U term in
the device Bloch operators of the k-point workers
(``Hamiltonian/device_bloch.py``), form the occupation matrices from the device
orbitals (``SCF/kpoint_hubbard.py``) and, on a GPU, the TS sums on the device.
Every result is compared with the CPU path of the same input.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available


DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

_AG_CELL = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   6.0  6.0  6.0
end Cell_Shape
{kpoints}
Grid_Spacing: 0.6
Expansion_Order: 6
Ignore_Symmetry: .true.
States_Num: 10
Fermi_Temp: 1000.0
Max_Iter: {iterations}
Convergence_Criterion: 1.0e-7
Mixing_Param: 0.2
Eigensolver: chebff
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
{ag_line}
begin Atom_Coord
   3.0 3.0 3.0
end Atom_Coord
"""

_TWO_K = """Kpoint_Method: manual
Kpoint_Unit: reciprocal_lattice_vectors
begin Kpoint_List
 0.0 0.0 0.0 0.5
 0.5 0.0 0.0 0.5
end Kpoint_List"""

_H2_CELL = """Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   7.2  7.2  8.4
end Cell_Shape
{kpoints}
Grid_Spacing: 0.6
Expansion_Order: 6
Ignore_Symmetry: .true.
Van_Der_Waals: .true.
States_Num: 4
Correlation_Type: pbe
Convergence_Criterion: 1.0e-7
Max_Iter: 40
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.6 3.6 1.8
  3.6 3.6 3.2
  3.9 3.8 5.5
  3.9 3.8 6.9
end Atom_Coord
"""

_ROOT: tempfile.TemporaryDirectory | None = None


def setUpModule():
    global _ROOT
    _ROOT = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


def tearDownModule():
    if _ROOT is not None:
        _ROOT.cleanup()


def _ag(*, kpoints="", iterations=30, extra="", ag_line="ldaplusu_u 4.0 eV"):
    directory = Path(tempfile.mkdtemp(dir=_ROOT.name))
    (directory / "parsec.in").write_text(
        _AG_CELL.format(kpoints=kpoints, iterations=iterations, extra=extra, ag_line=ag_line)
    )
    (directory / "Ag_POTRE.DAT").write_bytes((DATA / "Ag_POTRE.DAT").read_bytes())
    return parse_parsec_input(directory / "parsec.in")


def _h2(*, kpoints="", extra=""):
    directory = Path(tempfile.mkdtemp(dir=_ROOT.name))
    (directory / "parsec.in").write_text(_H2_CELL.format(kpoints=kpoints, extra=extra))
    (directory / "H_POTRE.DAT").write_bytes((DATA / "spin_pbe_H_POTRE.DAT").read_bytes())
    return parse_parsec_input(directory / "parsec.in")


def _kpoint_run(system, translation, backend):
    from parsec_python.Grid import kpoint_sampling
    from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized

    k_points, weights = kpoint_sampling(translation.problem)
    runner = (
        run_scf_kpoints_spin_polarized
        if translation.problem.scf.spin_polarized
        else run_scf_kpoints
    )
    with mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": backend}):
        return runner(system, k_points, weights)


class DeviceBlochTermTest(unittest.TestCase):
    """The LDA+U term of the device Bloch operator (the NumPy double runs the GPU code)."""

    def _hamiltonian(self):
        from parsec_python.Hamiltonian.hubbard import hubbard_terms
        from parsec_python.SCF.kpoint_hubbard import KPointHubbard
        from parsec_python.SCF.kpoints import KPointKohnShamHamiltonian
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.V_ion import build_nonlocal_projectors

        problem = _ag(kpoints=_TWO_K).problem
        system = prepare_periodic_single_point(problem)
        k_point = np.array([0.3, -0.2, 0.1])
        hubbard = KPointHubbard.for_system(system, k_point[None, :])
        lattice = problem.periodic_cell.lattice_vectors
        nonlocal_k = build_nonlocal_projectors(
            system.grid, system.atoms, system.pseudopotentials, problem.pseudopotentials,
            lattice, k_point=k_point,
        )
        base = KPointKohnShamHamiltonian(
            system.negative_laplacian, system.gradient, system.ionic_potential, nonlocal_k, k_point
        )
        rng = np.random.default_rng(5)
        occupations = rng.uniform(0.0, 1.0, size=(1, 5, 5))
        occupations = 0.5 * (occupations + np.swapaxes(occupations, 1, 2))
        potentials = hubbard_terms(system.hubbard, [occupations]).potentials[0]
        return system, hubbard, hubbard.hamiltonian(base, 0, potentials)

    def _check(self, backend):
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter

        system, _hubbard, hamiltonian = self._hamiltonian()
        rng = np.random.default_rng(9)
        block = rng.normal(size=(system.grid.size, 3)) + 1j * rng.normal(size=(system.grid.size, 3))
        expected = hamiltonian.apply(block)
        operator = DeviceBlochConverter(backend).operator(hamiltonian)
        self.assertIsNotNone(operator.hubbard)
        np.testing.assert_allclose(operator.matmat(block), expected, rtol=0.0, atol=1.0e-11)
        plain = DeviceBlochConverter(backend).operator(hamiltonian.base)
        self.assertIsNone(plain.hubbard)
        np.testing.assert_allclose(
            plain.matmat(block), hamiltonian.base.apply(block), rtol=0.0, atol=1.0e-11
        )

    def test_numpy_double_applies_the_term(self):
        from parsec_python.acceleration.Hamiltonian.device_bloch import NumpyArrayBackend

        self._check(NumpyArrayBackend())

    @unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
    def test_device_applies_the_term_and_forms_the_occupations(self):
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend

        self._check(CupyArrayBackend(0))
        system, hubbard, _hamiltonian = self._hamiltonian()
        rng = np.random.default_rng(3)
        vectors = rng.normal(size=(system.grid.size, 6)) + 1j * rng.normal(size=(system.grid.size, 6))
        weights = np.array([1.0, 1.0, 0.6, 0.2, 0.0, 0.0])
        expected = hubbard.occupation(0, vectors, weights, 0.5)
        device = hubbard.occupation(0, cp.asarray(vectors), weights, 0.5, xp=cp)
        np.testing.assert_allclose(device, expected, rtol=0.0, atol=1.0e-12)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class KPointLoopTest(unittest.TestCase):
    """The k-point loops with LDA+U and TS: CPU workers against device workers."""

    def test_lda_plus_u_on_device_workers(self):
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        for extra in ("", "Spin_Polarization: .true."):
            with self.subTest(spin=bool(extra)):
                translation = _ag(kpoints=_TWO_K, extra=extra)
                system = prepare_periodic_single_point(translation.problem)
                host = _kpoint_run(system, translation, "cpu")
                device = _kpoint_run(system, translation, "cupy")
                self.assertTrue(host.converged and device.converged)
                self.assertAlmostEqual(device.energies.total, host.energies.total, delta=1.0e-7)
                self.assertAlmostEqual(device.energies.hubbard, host.energies.hubbard, delta=1.0e-6)
                for name in (("eigenvalues_up", "eigenvalues_down") if extra else ("eigenvalues",)):
                    np.testing.assert_allclose(
                        getattr(device, name), getattr(host, name), atol=1.0e-6
                    )

    def test_ts_sums_on_the_device(self):
        from parsec_python.Energy import dispersion as host_dispersion
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.Energy import dispersion_device

        translation = _h2(kpoints=_TWO_K)
        system = prepare_periodic_single_point(translation.problem)
        self.assertIs(
            host_dispersion.device_dispersion_for(system) or dispersion_device.device_dispersion_correction,
            dispersion_device.device_dispersion_correction,
        )
        host = _kpoint_run(system, translation, "cpu")
        # On the CPU backend the host sums: no device correction is in use.
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": "cpu"}):
            self.assertIsNone(host_dispersion.device_dispersion_for(system))
        device = _kpoint_run(system, translation, "cupy")
        self.assertGreater(device.dispersion.shells, 0)
        self.assertEqual(device.dispersion.shells, host.dispersion.shells)
        # The two loops take different eigensolvers (device and host CHEBFF):
        # their densities agree to the SCF criterion, the sums exactly
        # (DeviceSumsTest).
        self.assertAlmostEqual(device.dispersion.energy, host.dispersion.energy, delta=1.0e-8)
        self.assertAlmostEqual(device.energies.total, host.energies.total, delta=1.0e-7)


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceSumsTest(unittest.TestCase):
    def test_hirshfeld_cells_and_lattice_shells_equal_the_reference(self):
        from parsec_python.Energy.dispersion import hirshfeld_volumes, ts_pair_energy
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.acceleration.Energy import (
            device_hirshfeld_volumes,
            device_ts_pair_energy,
        )

        problem = _h2().problem
        system = prepare_periodic_single_point(problem)
        lattice = problem.periodic_cell.lattice_vectors
        rng = np.random.default_rng(21)
        density = system.initial_density * rng.uniform(0.5, 1.5, size=system.grid.size)
        for cells in ((0, 0, 0), (1, 0, 1)):
            with self.subTest(cells=cells):
                arguments = (
                    system.grid, system.atoms, system.pseudopotentials,
                    problem.pseudopotentials, density, lattice, cells,
                )
                expected = hirshfeld_volumes(*arguments)
                volumes = device_hirshfeld_volumes(*arguments)
                np.testing.assert_allclose(volumes.free, expected.free, rtol=1.0e-12)
                np.testing.assert_allclose(volumes.effective, expected.effective, rtol=1.0e-12)
        positions = np.array([atom.position for atom in system.atoms])
        c6 = rng.uniform(2.0, 8.0, size=4)
        alpha = rng.uniform(3.0, 6.0, size=4)
        radius = rng.uniform(2.5, 3.5, size=4)
        energy, forces, shells = ts_pair_energy(positions, c6, alpha, radius, lattice)
        device_energy, device_forces, device_shells = device_ts_pair_energy(
            positions, c6, alpha, radius, lattice
        )
        self.assertEqual(device_shells, shells)
        self.assertAlmostEqual(device_energy, energy, delta=1.0e-13 * abs(energy))
        np.testing.assert_allclose(device_forces, forces, rtol=0.0, atol=1.0e-13 * np.abs(forces).max())


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class GammaCellTest(unittest.TestCase):
    """A bulk cell at the Gamma point on the accelerated driver."""

    def test_lda_plus_u_follows_the_reference_loop(self):
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf

        problem = _ag(iterations=8).problem
        reference = run_scf(prepare_periodic_single_point(problem))
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
        self.assertIsNotNone(system.hubbard_device)
        system.materialize_final_wavefunctions = False
        result = driver.run_scf(system)
        for step, expected in zip(result.history, reference.history):
            self.assertAlmostEqual(step.energies.total, expected.energies.total, delta=1.0e-8)
        self.assertAlmostEqual(result.energies.hubbard, reference.energies.hubbard, delta=1.0e-8)

    def test_lda_plus_u_forces_take_the_images_of_the_cell(self):
        # The device products of Hamiltonian/hubbard_device.py against
        # Forces/hubbard_force.hubbard_forces of the same orbitals, both with the
        # replicas of the cell in the projector gradients.
        from parsec_python.Forces.hubbard_force import hubbard_forces
        from parsec_python.acceleration.SCF.single_point import materialize_wavefunctions

        # Two atoms closer through the cell boundary (2.9 bohr) than inside it (3.1 bohr).
        directory = Path(tempfile.mkdtemp(dir=_ROOT.name))
        text = _AG_CELL.format(kpoints="", iterations=8, extra="", ag_line="ldaplusu_u 4.0 eV")
        text = text.replace("States_Num: 10", "States_Num: 16").replace(
            "   3.0 3.0 3.0", "   0.8 2.9 3.1\n   3.9 3.2 2.8"
        )
        (directory / "parsec.in").write_text(text)
        (directory / "Ag_POTRE.DAT").write_bytes((DATA / "Ag_POTRE.DAT").read_bytes())
        problem = parse_parsec_input(directory / "parsec.in").problem
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
        system.materialize_final_wavefunctions = False
        result = driver.run_scf(system)
        forces = system.hubbard_device.forces(system, result)
        materialize_wavefunctions(system, result)
        expected = hubbard_forces(system, [result.wavefunctions], [result.occupations])
        self.assertGreater(np.max(np.abs(expected)), 1.0e-4)
        np.testing.assert_allclose(forces, expected, rtol=0.0, atol=1.0e-10)

    def test_ts_follows_the_reference(self):
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf

        problem = _h2().problem
        reference = run_scf(prepare_periodic_single_point(problem))
        system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
        self.assertIsNotNone(system.dispersion_device)
        system.materialize_final_wavefunctions = False
        result = driver.run_scf(system)
        self.assertTrue(result.converged and reference.converged)
        self.assertEqual(result.dispersion.shells, reference.dispersion.shells)
        self.assertAlmostEqual(result.dispersion.energy, reference.dispersion.energy, delta=1.0e-10)
        self.assertAlmostEqual(result.energies.total, reference.energies.total, delta=1.0e-7)
        np.testing.assert_allclose(result.dispersion.forces, reference.dispersion.forces, atol=1.0e-9)


class RoutingTest(unittest.TestCase):
    def test_gamma_cells_take_the_accelerated_driver(self):
        from parsec_python.acceleration.cli import _reference_only_features

        self.assertEqual(_reference_only_features(_ag(), accelerated_backend_selected=False), [])
        self.assertEqual(_reference_only_features(_h2(), accelerated_backend_selected=False), [])
        # k-points go to the reference solver by their own entry (device workers there)
        features = _reference_only_features(_ag(kpoints=_TWO_K), accelerated_backend_selected=False)
        self.assertEqual(features, ["k-point sampling (Monkhorst-Pack or explicit list)"])
        with_j = _ag(ag_line="ldaplusu_u 4.0 eV\nldaplusu_j 0.5 eV")
        self.assertIn(
            "LDA+U with ldaplusu_j != 0",
            _reference_only_features(with_j, accelerated_backend_selected=False),
        )

    def test_lda_plus_u_of_a_slab_stays_on_the_reference_solver(self):
        from parsec_python.acceleration.cli import _reference_only_features

        directory = Path(tempfile.mkdtemp(dir=_ROOT.name))
        (directory / "parsec.in").write_text(
            "Boundary_Conditions: slab\nbegin Cell_Shape\n 6.0 0.0\n 0.0 6.0\nend Cell_Shape\n"
            "Boundary_Sphere_Radius: 6.0\nGrid_Spacing: 0.6\nStates_Num: 10\n"
            "Atom_Types_Num: 1\nAtom_Type: Ag\nLocal_Component: s\nldaplusu_u 4.0 eV\n"
            "begin Atom_Coord\n  0.0 0.0 0.0\nend Atom_Coord\n"
        )
        (directory / "Ag_POTRE.DAT").write_bytes((DATA / "Ag_POTRE.DAT").read_bytes())
        translation = parse_parsec_input(directory / "parsec.in")
        self.assertIn(
            "LDA+U of a wire or slab (ldaplusu_u)",
            _reference_only_features(translation, accelerated_backend_selected=False),
        )
        with self.assertRaisesRegex(NotImplementedError, "wire or slab"):
            driver._refuse_corrections(translation.problem, is_periodic=True, mpi_context=None)

    def test_mpi_keeps_refusing_lda_plus_u(self):
        from parsec_python.acceleration.cli import _mpi_unsupported_features

        self.assertTrue(
            any("LDA+U" in item for item in _mpi_unsupported_features(_ag(kpoints=_TWO_K)))
        )


if __name__ == "__main__":
    unittest.main()
