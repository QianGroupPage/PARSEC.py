"""Where the device k-point path, the MPI solver and the fast periodic setup meet the parity features.

The merge of the device-resident k-point eigensolver (qian/main) with the Fortran-parity features
routes every combination the device paths do not carry to a path that does, by name:

* ARPACK and DIAGLA (``Eigensolver: arpack`` / ``diagla``) are host eigensolvers: a device k-point
  worker and a device band path run them on the device Hamiltonian as a linear operator (the route
  before the device solver existed), never the device CHEBFF/SUBSPACE solver in their place;
* ``Dynamic_Diag_Tol`` reaches the device workers (the tolerance of the iteration, which the device
  CHEBDAV counterpart uses as its residual test);
* the MPI-distributed solver refuses ARPACK/DIAGLA, and the accelerated CLI refuses under ``--mpi``
  what needs whole orbitals (LDA+U, periodic or non-collinear forces/relaxation/MD, the Ylm DOS,
  Output_GW);
* the fast periodic setup (index-box support, FFT local potential) is for the plain orthorhombic grid
  only: Fortran's general grid and the wire/slab grids keep the readable definitions;
* a Gamma-only periodic spin run goes through the k-point driver on a device, except with LDA+U (a
  wire or slab too, unless ``PARSEC_WIRE_SLAB_GAMMA_SPIN=gamma``).
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import dataclasses
import os
import unittest
from unittest import mock

import numpy as np

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.acceleration.tests.test_kpoint_parallel import _K, _WEIGHTS, _system
from parsec_python.SCF.kpoint_solver import device_solver_supports
from parsec_python.SCF.kpoints import run_scf_kpoints

_DEVICE = {"PARSEC_KPOINT_BACKEND": "numpy-device", "PARSEC_KPOINT_WORKERS": "1"}
_CPU = {"PARSEC_KPOINT_BACKEND": "cpu", "PARSEC_KPOINT_WORKERS": "1"}


def _refuse_device_solver(*_args, **_kwargs):
    raise AssertionError("the device eigensolver ran for a host eigensolver")


class HostEigensolversOnDeviceWorkersTests(unittest.TestCase):
    def test_support_rule(self) -> None:
        from parsec_python.Eigensolvers import EigvalSettings

        self.assertTrue(device_solver_supports(EigvalSettings(initial_method="chebff")))
        self.assertTrue(device_solver_supports(EigvalSettings(initial_method="chebdav")))
        self.assertFalse(device_solver_supports(EigvalSettings(initial_method="arpack")))
        self.assertFalse(device_solver_supports(EigvalSettings(initial_method="diagla")))

    def test_arpack_kpoint_scf_on_a_device_worker_runs_arpack(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter

        system = _system(method="arpack")
        with mock.patch.dict(os.environ, _CPU):
            host = run_scf_kpoints(system, _K, _WEIGHTS)
        original = DeviceBlochConverter.operator
        uploads = []

        def counted(converter, hamiltonian):
            uploads.append(1)
            return original(converter, hamiltonian)

        with mock.patch.dict(os.environ, _DEVICE), mock.patch(
            "parsec_python.acceleration.SCF.kpoint_parallel.solve_spinor_eigval",
            side_effect=_refuse_device_solver,
        ), mock.patch.object(DeviceBlochConverter, "operator", autospec=True, side_effect=counted):
            device = run_scf_kpoints(system, _K, _WEIGHTS)
        # ARPACK ran on the device worker's operator (the host route of the device plan)
        self.assertGreater(len(uploads), 0)
        self.assertTrue(host.converged and device.converged)
        # the same ARPACK on the device operator: same energy and levels
        self.assertAlmostEqual(device.energies.total, host.energies.total, places=7)
        np.testing.assert_allclose(device.eigenvalues, host.eigenvalues, atol=1.0e-6)

    def test_arpack_band_path_on_a_device_converter_runs_arpack(self) -> None:
        from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter
        from parsec_python.SCF.bands import solve_kpoint_bands

        system = _system(method="arpack")
        potentials = {"effective_potential": np.asarray(system.ionic_potential)}
        with mock.patch.dict(os.environ, _CPU):
            host = solve_kpoint_bands(
                system, _K[:2], mode="scalar", potentials=potentials, number_of_states=8
            )
        with mock.patch.dict(os.environ, _DEVICE), mock.patch.object(
            DeviceBlochConverter, "eigensolve", side_effect=_refuse_device_solver
        ):
            device = solve_kpoint_bands(
                system, _K[:2], mode="scalar", potentials=potentials, number_of_states=8
            )
        np.testing.assert_allclose(device.eigenvalues, host.eigenvalues, atol=1.0e-6)


class DynamicToleranceOnDeviceWorkersTests(unittest.TestCase):
    def test_the_device_worker_receives_the_iteration_tolerance(self) -> None:
        from parsec_python.SCF import kpoints as kpoint_driver

        base = _system(method="chebdav")
        eigensolver = dataclasses.replace(base.input.eigensolver, dynamic_tolerance=True)
        system = dataclasses.replace(
            base, input=dataclasses.replace(base.input, eigensolver=eigensolver)
        )
        seen = []
        original = kpoint_driver._eigval_settings

        def recording(system_, degree, tolerance=None):
            seen.append(tolerance)
            return original(system_, degree, tolerance)

        records = []
        with mock.patch.dict(os.environ, _DEVICE), mock.patch.object(
            kpoint_driver, "_eigval_settings", side_effect=recording
        ):
            result = run_scf_kpoints(system, _K, _WEIGHTS, callback=records.append)
        self.assertTrue(result.converged)
        # the workers asked with the tolerance of each iteration: 0.1 first, then smaller
        handed = [value for value in seen if value is not None]
        self.assertTrue(handed)
        self.assertAlmostEqual(handed[0], 0.1)
        self.assertLess(min(handed), 0.1)
        tolerances = [step.next_eigensolver_tolerance for step in records]
        self.assertTrue(all(b <= a for a, b in zip(tolerances, tolerances[1:])))


class MPIRoutingTests(unittest.TestCase):
    def test_unsupported_features_are_named(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.cli import _mpi_unsupported_features

        def problem(**changes):
            fields = dict(
                pseudopotentials={"Si": SimpleNamespace(has_hubbard=False)},
                eigensolver=SimpleNamespace(method="chebdav"),
                periodic_cell=object(),
                has_kpoint_sampling=True,
                scf=SimpleNamespace(noncollinear=False),
            )
            fields.update(changes)
            return SimpleNamespace(**fields)

        def translation(problem_, **changes):
            fields = dict(
                problem=problem_, compute_forces=False, relaxation=None,
                molecular_dynamics=None, dos=None, gw=None,
            )
            fields.update(changes)
            return SimpleNamespace(**fields)

        self.assertEqual(_mpi_unsupported_features(translation(problem())), [])
        cases = {
            "LDA+U": translation(
                problem(pseudopotentials={"Zn": SimpleNamespace(has_hubbard=True)})
            ),
            "Eigensolver: arpack": translation(problem(eigensolver=SimpleNamespace(method="arpack"))),
            "Eigensolver: diagla": translation(problem(eigensolver=SimpleNamespace(method="diagla"))),
            "forces": translation(problem(), compute_forces=True),
            "Ylm": translation(problem(), dos=SimpleNamespace(ylm_l=1)),
            "Output_GW": translation(problem(), gw=object()),
        }
        for name, case in cases.items():
            with self.subTest(feature=name):
                found = _mpi_unsupported_features(case)
                self.assertEqual(len(found), 1)
                self.assertIn(name, found[0])
        # a cluster's forces do not run through the distributed k-point solver
        cluster = problem(periodic_cell=None, has_kpoint_sampling=False)
        self.assertEqual(_mpi_unsupported_features(translation(cluster, compute_forces=True)), [])
        # the plain DOS (no Ylm projection) needs no orbitals
        self.assertEqual(
            _mpi_unsupported_features(translation(problem(), dos=SimpleNamespace(ylm_l=-1))), []
        )

    def test_distributed_solver_refuses_host_eigensolvers(self) -> None:
        from parsec_python.acceleration.SCF.kpoint_mpi import MPIKPointSolver
        from parsec_python.Eigensolvers import EigvalSettings
        from parsec_python.SCF.kpoint_solver import KPointJob

        solver = MPIKPointSolver.__new__(MPIKPointSolver)  # no communicator needed to refuse
        solver.job = KPointJob(
            1, 2, lambda degree, tolerance=None: EigvalSettings(initial_method="diagla"),
            None, None,
        )
        with self.assertRaisesRegex(NotImplementedError, "diagla"):
            solver.solve({"V": np.zeros(4)}, 15)


class FastPeriodicSetupGridsTests(unittest.TestCase):
    def _general_grids(self):
        from parsec_python.Grid.pbc import build_general_periodic_grid, uses_general_periodic_grid

        oblique = PeriodicCell(
            lattice_vectors=5.1 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        )
        # orthorhombic, but no common spacing: the general grid with its own steps and shift
        uneven = PeriodicCell(lattice_vectors=np.diag([6.0, 6.7, 7.9]))
        grids = []
        for cell in (oblique, uneven):
            settings = PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=6)
            self.assertTrue(uses_general_periodic_grid(cell, settings))
            grids.append(build_general_periodic_grid(cell, settings))
        return grids

    def test_general_grids_keep_the_dense_support_and_the_direct_ewald_sum(self) -> None:
        from parsec_python.acceleration.V_ion import periodic_setup_fortran as periodic_setup
        from parsec_python.V_ion.ionic_potential import dense_support

        for grid in self._general_grids():
            with self.subTest(grid=type(grid).__name__, shape=grid.shape):
                self.assertFalse(periodic_setup._plain_orthorhombic(grid))
                centre = np.array([1.37, 2.11, 0.43])
                found = periodic_setup.grid_ball(grid, centre, 2.3)
                expected = dense_support(grid, centre, 2.3)
                for value, reference in zip(found, expected):
                    np.testing.assert_array_equal(value, reference)
                self.assertIsNone(
                    periodic_setup.fft_local_ionic_potential(
                        grid, [], {}, {}, grid.cell.lattice_vectors, None, 1.0e-8
                    )
                )

    def test_local_forces_differentiate_the_potential_the_scf_used(self) -> None:
        # The FFT provider returns the converged Ewald sum on the plain orthorhombic grid: the
        # periodic local forces then use the converged sum's translations (their derivative is the
        # potential the SCF used; with the default ones they missed ~1e-7 Ry/bohr in an Si2 cube).
        from parsec_python.acceleration.V_ion import periodic_setup_fortran as periodic_setup
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.V_ion.ewald import CONVERGED_TOLERANCE, local_potential_tolerance

        cube = PeriodicCell(lattice_vectors=np.diag([6.0, 6.0, 6.0]))
        plain = build_periodic_grid(cube, PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=6))
        self.assertTrue(periodic_setup._plain_orthorhombic(plain))
        lattice = cube.lattice_vectors
        self.assertEqual(local_potential_tolerance(plain, lattice, 1.0e-14), CONVERGED_TOLERANCE)
        for grid in self._general_grids():
            self.assertEqual(local_potential_tolerance(grid, grid.cell.lattice_vectors, 1.0e-14), 1.0e-14)
        periodic_setup.uninstall()
        try:
            self.assertEqual(local_potential_tolerance(plain, lattice, 1.0e-14), 1.0e-14)
        finally:
            periodic_setup.install()

    def test_wire_and_slab_grids_are_general_grids(self) -> None:
        from parsec_python.Grid.confined import ConfinedPeriodicGrid

        self.assertTrue(ConfinedPeriodicGrid.is_general_periodic)
        self.assertTrue(ConfinedPeriodicGrid.is_confined_periodic)


class GammaSpinRouteExclusionsTests(unittest.TestCase):
    def test_lda_plus_u_goes_along_and_wire_slab_take_the_kpoint_driver(self) -> None:
        # LDA+U on the device workers; wires and slabs on the k-point driver,
        # with LDA+U too, unless PARSEC_WIRE_SLAB_GAMMA_SPIN=gamma.
        from types import SimpleNamespace

        from parsec_python import cli

        def translation(*, hubbard=False, periodicity=3):
            return SimpleNamespace(
                problem=SimpleNamespace(
                    pseudopotentials={"V": SimpleNamespace(has_hubbard=hubbard)},
                    grid=SimpleNamespace(periodicity=periodicity),
                )
            )

        with mock.patch.object(cli, "parallel_backend_uses_device", return_value=True):
            self.assertTrue(cli._gamma_spin_on_kpoint_driver(translation(), True, False))
            # the device workers apply the LDA+U term (SCF/kpoint_hubbard.py)
            self.assertTrue(cli._gamma_spin_on_kpoint_driver(translation(hubbard=True), True, False))
            self.assertTrue(cli._gamma_spin_on_kpoint_driver(translation(periodicity=1), True, False))
            self.assertTrue(cli._gamma_spin_on_kpoint_driver(translation(periodicity=2), True, False))
            self.assertTrue(
                cli._gamma_spin_on_kpoint_driver(translation(hubbard=True, periodicity=2), True, False)
            )
            with mock.patch.dict(os.environ, {"PARSEC_WIRE_SLAB_GAMMA_SPIN": "gamma"}):
                self.assertFalse(cli._gamma_spin_on_kpoint_driver(translation(periodicity=2), True, False))
                self.assertFalse(
                    cli._gamma_spin_on_kpoint_driver(translation(hubbard=True, periodicity=2), True, False)
                )
                self.assertTrue(cli._gamma_spin_on_kpoint_driver(translation(), True, False))
            # clusters and spin-orbit species never
            self.assertFalse(cli._gamma_spin_on_kpoint_driver(translation(), False, False))
            self.assertFalse(cli._gamma_spin_on_kpoint_driver(translation(), True, True))
        with mock.patch.object(cli, "parallel_backend_uses_device", return_value=False):
            self.assertFalse(cli._gamma_spin_on_kpoint_driver(translation(), True, False))


class SpinorIterationDispatchTests(unittest.TestCase):
    def test_spinor_records_with_timings_go_to_the_spinor_report(self) -> None:
        # The ionic loops hand every record to ParsecTextReporter.iteration; the spinor record now
        # carries diagonalization_seconds as the scalar one does.
        from parsec_python.models import SelfConsistentSOCIteration
        from parsec_python.Output.parsec_output import ParsecTextReporter

        record = SelfConsistentSOCIteration(
            iteration=1, weighted_residual=0.1, plain_residual=0.1, energies=None,
            diagonalization_seconds=0.5, total_seconds=1.0,
        )
        reporter = ParsecTextReporter.__new__(ParsecTextReporter)
        with mock.patch.object(
            ParsecTextReporter, "iteration_self_consistent_soc"
        ) as spinor_report:
            ParsecTextReporter.iteration(reporter, record)
        spinor_report.assert_called_once_with(record)


if __name__ == "__main__":
    unittest.main()
