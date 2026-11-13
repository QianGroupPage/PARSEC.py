"""The device parts of the periodic SCF loop for the round's periodic features.

* :mod:`parsec_python.acceleration.Hartree.periodic_device`: the wire and slab
  Hartree solves (boundary values of ``hartset_wire``/``hartset_slab`` and the
  CG of ``hpotcg``) and ``Calc_Hartree_G`` on the device equal the readable
  solves to round-off; which solves go there (``PARSEC_PERIODIC_HARTREE``,
  the k-point plan); the hooks of :mod:`parsec_python.Hartree.confined` and
  :mod:`parsec_python.Hartree.pbc`;
* the accelerated driver runs a Gamma-point wire, slab, ``Calc_Hartree_G``,
  ``Periodic_Local_Potential: g_space`` and ``Zero_Average_Vxc`` input as the
  reference solver does (:mod:`parsec_python.acceleration.SCF.periodic_routes`),
  and the combinations it does not carry stay routed by name;
* :class:`parsec_python.acceleration.V_xc.device_xc.ZeroAverageXC`;
* the Gamma-point spin run of a wire or slab takes the 1x1x1 k-point driver
  where the k-point eigenproblems run on a device.
"""

from __future__ import annotations

from dataclasses import replace
import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.Hartree import periodic_device
from parsec_python.Grid.confined import build_confined_periodic_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hartree import confined as confined_hartree
from parsec_python.Hartree import pbc as periodic_hartree
from parsec_python.Hartree.confined import plan_confined_hartree, solve_confined_hartree
from parsec_python.Hartree.gspace import solve_hartree_g_space
from parsec_python.Hartree.pbc import solve_periodic_hartree
from parsec_python.Input import parse_parsec_input
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.models import Atom, HartreeSettings, PeriodicCell, PeriodicGridSettings
from parsec_python.tests.unit.test_wire_slab import hydrogen_problem

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_DEVICE = {"PARSEC_KPOINT_BACKEND": "cupy", "PARSEC_PERIODIC_HARTREE": "auto"}
_CPU = {"PARSEC_KPOINT_BACKEND": "cpu"}


def _gaussian(grid, centre, width=0.9, charge=1.0):
    offset = grid.coordinates - np.asarray(centre)
    if hasattr(grid, "cell"):
        lattice = np.asarray(grid.cell.lattice_vectors, dtype=float)
        fractional = offset @ np.linalg.inv(lattice)
        periodic = getattr(grid, "periodicity", 3)
        fractional[:, :periodic] -= np.round(fractional[:, :periodic])
        offset = fractional @ lattice
    values = np.exp(-np.sum(offset**2, axis=1) / width**2)
    return charge * values / (np.sum(values) * grid.volume_element)


def _confined(periodicity, *, spacing=0.4, radius=4.0, order=8):
    atoms = [Atom("H", (0.3, 0.1, 0.65)), Atom("H", (-0.2, 0.0, -0.75))]
    problem = hydrogen_problem(periodicity, atoms=atoms, spacing=spacing, radius=radius, order=order)
    grid = build_confined_periodic_grid(problem.periodic_cell, problem.grid)
    laplacian = build_negative_laplacian(grid)
    plan = plan_confined_hartree(grid, problem.hartree.multipole_order)
    density = _gaussian(grid, (0.0, 0.2, 0.3)) + 0.5 * _gaussian(grid, (1.0, -0.4, -0.8), width=0.6)
    return grid, laplacian, plan, density


class ModeTests(unittest.TestCase):
    def test_switch_values(self) -> None:
        for value, mode in (("", "auto"), ("auto", "auto"), ("device", "device"), ("on", "device"),
                            ("host", "host"), ("off", "host"), ("0", "host")):
            with self.subTest(value=value), mock.patch.dict(os.environ, {periodic_device.MODE_ENV: value}):
                self.assertEqual(periodic_device.hartree_device_mode(), mode)
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "gpu-please"}):
            with self.assertRaises(ValueError):
                periodic_device.hartree_device_mode()

    def test_cpu_plans_keep_the_host(self) -> None:
        for backend in ("cpu", "numpy-device"):
            with self.subTest(backend=backend), mock.patch.dict(
                os.environ, {"PARSEC_KPOINT_BACKEND": backend}
            ):
                self.assertIsNone(periodic_device.hartree_device_id())
        grid, laplacian, plan, _ = _confined(1, spacing=0.6, radius=3.0, order=4)
        with mock.patch.dict(os.environ, _CPU):
            self.assertIsNone(periodic_device.confined_device_solver(plan, laplacian))

    def test_hooks_are_installed_by_the_package(self) -> None:
        import parsec_python.acceleration  # noqa: F401

        self.assertIs(confined_hartree._confined_solver, periodic_device.solve_confined_on_device)
        self.assertIs(periodic_hartree._periodic_solver, periodic_device.solve_periodic_on_device)

    def test_host_keeps_the_host_cg_of_a_plain_cell(self) -> None:
        # Up to 2026-10-30 ``auto`` kept the FD-CG of a 3-D cell on the host (this test
        # asserted it for ``auto``); since then it runs on the device by default
        # (test_the_fd_cg_of_a_plain_cell_takes_the_device_by_default), and ``host`` is the way back.
        grid = build_periodic_grid(PeriodicCell(5.0 * np.eye(3)), PeriodicGridSettings(spacing=0.5, expansion_order=4))
        laplacian = build_negative_laplacian(grid)
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "host", "PARSEC_KPOINT_BACKEND": "cupy"}):
            self.assertIsNone(periodic_device.periodic_device_solver(grid, laplacian, HartreeSettings()))
            self.assertIsNone(
                periodic_device.periodic_device_solver(
                    grid, laplacian, HartreeSettings(periodic_g_space=True)
                )
            )
        with mock.patch.dict(os.environ, _CPU):
            for mode in ("", "auto", "device"):
                with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: mode}):
                    self.assertIsNone(periodic_device.periodic_device_solver(grid, laplacian, HartreeSettings()))

    @unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
    def test_the_fd_cg_of_a_plain_cell_takes_the_device_by_default(self) -> None:
        grid = build_periodic_grid(PeriodicCell(5.0 * np.eye(3)), PeriodicGridSettings(spacing=0.5, expansion_order=4))
        laplacian = build_negative_laplacian(grid)
        periodic_device.clear_cache()
        for mode in ("", "auto", "device"):
            with self.subTest(mode=mode), mock.patch.dict(
                os.environ, {"PARSEC_KPOINT_BACKEND": "cupy", periodic_device.MODE_ENV: mode}
            ):
                solver = periodic_device.periodic_device_solver(grid, laplacian, HartreeSettings())
                self.assertIsInstance(solver, periodic_device.DevicePeriodicHartree)
                self.assertFalse(solver.g_space)
        periodic_device.clear_cache()

    def test_text_of_the_default_cell(self) -> None:
        system = SimpleNamespace(
            grid=SimpleNamespace(is_general_periodic=False),
            input=SimpleNamespace(hartree=HartreeSettings()),
        )
        # The host CG (``host``, or no device in the k-point plan) keeps the driver's text.
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "host"}):
            self.assertEqual(periodic_device.describe_periodic_hartree(system, "old text"), "old text")
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "auto", **_CPU}):
            self.assertEqual(periodic_device.describe_periodic_hartree(system, "old text"), "old text")
        if cupy_available():
            for mode in ("auto", "device"):
                with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: mode, "PARSEC_KPOINT_BACKEND": "cupy"}):
                    self.assertEqual(
                        periodic_device.describe_periodic_hartree(system, "old text"),
                        "periodic conjugate-gradient solve, CuPy float64 on the device "
                        f"({periodic_device.MODE_ENV}={mode}; SCF.pbc.solve_periodic_hartree)",
                    )
        general = SimpleNamespace(
            grid=SimpleNamespace(is_general_periodic=True),
            input=SimpleNamespace(hartree=HartreeSettings()),
        )
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "auto"}):
            self.assertEqual(periodic_device.describe_periodic_hartree(general, "old text"), "old text")


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceSolveTests(unittest.TestCase):
    def setUp(self) -> None:
        periodic_device.clear_cache()

    def test_wire_and_slab_boundary_values(self) -> None:
        from parsec_python.acceleration.backends.cupy import require_cupy

        cp, _ = require_cupy()
        for periodicity, device_class in (
            (1, periodic_device.DeviceWireBoundary),
            (2, periodic_device.DeviceSlabBoundary),
        ):
            with self.subTest(periodicity=periodicity):
                grid, _, plan, density = _confined(periodicity)
                host = plan.boundary_values(density)
                device = cp.asnumpy(device_class(plan.boundary, cp).values(cp.asarray(density)))
                self.assertEqual(device.shape, host.shape)
                np.testing.assert_allclose(device, host, rtol=1.0e-12, atol=1.0e-14 * np.max(np.abs(host)))

    def test_wire_and_slab_solves(self) -> None:
        settings = HartreeSettings(relative_tolerance=1.0e-12)
        for periodicity in (1, 2):
            with self.subTest(periodicity=periodicity):
                grid, laplacian, plan, density = _confined(periodicity)
                start = -0.1 * density
                with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "host"}):
                    host = solve_confined_hartree(density, plan, laplacian, settings, start)
                with mock.patch.dict(os.environ, _DEVICE):
                    solver = periodic_device.confined_device_solver(plan, laplacian)
                    self.assertIsNotNone(solver)
                    device = solve_confined_hartree(density, plan, laplacian, settings, start)
                self.assertTrue(device.converged)
                self.assertLessEqual(abs(device.iterations - host.iterations), 2)
                scale = np.max(np.abs(host.potential))
                np.testing.assert_allclose(device.potential, host.potential, rtol=0, atol=1.0e-9 * scale)
                np.testing.assert_allclose(
                    device.boundary_values, host.boundary_values, rtol=1.0e-12,
                    atol=1.0e-14 * np.max(np.abs(host.boundary_values)),
                )
                np.testing.assert_allclose(
                    device.right_hand_side, host.right_hand_side, rtol=0,
                    atol=1.0e-12 * np.max(np.abs(host.right_hand_side)),
                )
                self.assertLess(device.residual_norm, 1.0e-10 * device.initial_residual_norm)
                # the second solve reuses the uploaded problem
                with mock.patch.dict(os.environ, _DEVICE):
                    self.assertIs(periodic_device.confined_device_solver(plan, laplacian), solver)

    def test_batches_change_no_bit_and_runs_are_deterministic(self) -> None:
        # the accepted iterations do the same arithmetic whatever the batch;
        # iterations past the stop are discarded
        grid, laplacian, plan, density = _confined(1)
        settings = HartreeSettings(relative_tolerance=1.0e-10)
        results = []
        for batch in ("1", "3", "8", "8", "64"):
            periodic_device.clear_cache()
            with mock.patch.dict(os.environ, {**_DEVICE, periodic_device.BATCH_ENV: batch}):
                results.append(solve_confined_hartree(density, plan, laplacian, settings, -0.1 * density))
        for result in results[1:]:
            self.assertEqual(result.iterations, results[0].iterations)
            self.assertEqual(result.matrix_vector_products, results[0].matrix_vector_products)
            np.testing.assert_array_equal(result.potential, results[0].potential)
        with mock.patch.dict(os.environ, {periodic_device.BATCH_ENV: "0"}):
            with self.assertRaises(ValueError):
                periodic_device._batch_size()

    def test_the_budget_and_breakdown_rules(self) -> None:
        # the matrix-vector budget stops the device CG where the host CG stops
        grid, laplacian, plan, density = _confined(2)
        tight = HartreeSettings(relative_tolerance=1.0e-14, absolute_tolerance=0.0, max_iterations=7)
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "host"}):
            host = solve_confined_hartree(density, plan, laplacian, tight, None, raise_on_nonconvergence=False)
        with mock.patch.dict(os.environ, _DEVICE):
            device = solve_confined_hartree(density, plan, laplacian, tight, None, raise_on_nonconvergence=False)
            with self.assertRaises(RuntimeError):
                solve_confined_hartree(density, plan, laplacian, tight, None)
        self.assertFalse(device.converged)
        self.assertEqual(device.iterations, host.iterations)
        self.assertEqual(device.matrix_vector_products, host.matrix_vector_products)
        np.testing.assert_allclose(device.potential, host.potential, rtol=0, atol=1.0e-11 * np.max(np.abs(host.potential)))

    def test_calc_hartree_g_on_the_device(self) -> None:
        from parsec_python.Grid.pbc import GeneralPeriodicGrid  # noqa: F401

        cells = (
            (PeriodicCell(np.diag([5.0, 6.0, 7.0])), PeriodicGridSettings(spacing=0.5, expansion_order=4)),
            # Fortran's general grid (fcc): the same spectral solve
            (
                PeriodicCell(5.0 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])),
                PeriodicGridSettings(spacing=0.45, expansion_order=4),
            ),
        )
        for cell, settings in cells:
            grid = build_periodic_grid(cell, settings)
            laplacian = build_negative_laplacian(grid)
            density = _gaussian(grid, grid.coordinates[grid.size // 3]) + 0.02
            hartree = HartreeSettings(periodic_g_space=True)
            with self.subTest(general=getattr(grid, "is_general_periodic", False)):
                expected = solve_hartree_g_space(density, grid)
                with mock.patch.dict(os.environ, _DEVICE):
                    result = solve_periodic_hartree(density, grid, laplacian, hartree, None)
                np.testing.assert_allclose(result.potential, expected, rtol=0, atol=1.0e-13 * np.max(np.abs(expected)))
                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 0)
                self.assertAlmostEqual(float(np.mean(result.potential)), 0.0, places=13)

    def test_opt_in_cg_of_a_plain_cell(self) -> None:
        grid = build_periodic_grid(PeriodicCell(np.diag([5.0, 6.0, 7.0])), PeriodicGridSettings(spacing=0.5, expansion_order=6))
        laplacian = build_negative_laplacian(grid)
        density = _gaussian(grid, grid.coordinates[17])
        settings = HartreeSettings(relative_tolerance=1.0e-12)
        with mock.patch.dict(os.environ, {periodic_device.MODE_ENV: "host"}):
            host = solve_periodic_hartree(density, grid, laplacian, settings, None)
        with mock.patch.dict(os.environ, {**_DEVICE, periodic_device.MODE_ENV: "device"}):
            self.assertIsNotNone(periodic_device.periodic_device_solver(grid, laplacian, settings))
            device = solve_periodic_hartree(density, grid, laplacian, settings, 0.3 * density)
        np.testing.assert_allclose(device.potential, host.potential, rtol=0, atol=1.0e-10 * np.max(np.abs(host.potential)))
        self.assertAlmostEqual(float(np.mean(device.potential)), 0.0, places=12)


def _write_input(folder: Path, text: str) -> Path:
    # The PBE input reads the PBE hydrogen of the spin tests.
    source = "spin_pbe_H_POTRE.DAT" if "Correlation_Type: pb" in text else "H_POTRE.DAT"
    shutil.copy(DATA / source, folder / "H_POTRE.DAT")
    path = folder / "parsec.in"
    path.write_text(text, encoding="utf-8")
    return path


_SLAB = """\
Boundary_Conditions: slab
begin Cell_Shape
  3.0 0.0
  0.0 3.2
end Cell_Shape
Boundary_Sphere_Radius: 4.0
Grid_Spacing: 0.5
Expansion_Order: 8
States_Num: 2
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-9
Eigensolver: chebdav
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.3 0.1 0.65
 -0.2 0.0 -0.75
end Atom_Coord
"""

_CELL = """\
Periodic_System: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.0 5.0 5.0
end Cell_Shape
Grid_Spacing: 0.6
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 1.0e-9
Diag_Tolerance: 1.0e-9
Correlation_Type: {xc}
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  1.6 2.4 2.2
  3.05 2.55 2.45
end Atom_Coord
"""


class RoutingTests(unittest.TestCase):
    def parse(self, text: str):
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parse_parsec_input(_write_input(Path(folder), text))

    def features(self, text: str) -> list[str]:
        from parsec_python.acceleration.cli import _reference_only_features

        return _reference_only_features(self.parse(text), accelerated_backend_selected=False)

    def test_gamma_inputs_of_the_features_run_accelerated(self) -> None:
        for text in (
            _SLAB.format(extra=""),
            _SLAB.format(extra="Correlation_Type: pb"),
            _CELL.format(xc="ca", extra="Calc_Hartree_G: .true."),
            _CELL.format(xc="ca", extra="Periodic_Local_Potential: g_space\nPeriodic_Alpha_Term: fortran"),
            _CELL.format(xc="ca", extra="Zero_Average_Vxc: .true."),
            _CELL.format(xc="pl", extra="Zero_Average_Vxc: .true.\nCalc_Hartree_G: .true."),
        ):
            with self.subTest(text=text.splitlines()[0:3]):
                self.assertEqual(self.features(text), [])

    def test_an_mpi_run_keeps_the_reference_route(self) -> None:
        # --mpi (PARSEC_MPI=1): the routes and host fields of the base
        with mock.patch.dict(os.environ, {"PARSEC_MPI": "1"}):
            for text, name in (
                (_SLAB.format(extra=""), "wire or slab in an MPI run"),
                (_CELL.format(xc="ca", extra="Calc_Hartree_G: .true."), "Calc_Hartree_G) in an MPI run"),
                (_CELL.format(xc="ca", extra="Periodic_Local_Potential: g_space"), "g_space) in an MPI run"),
            ):
                with self.subTest(name=name):
                    features = self.features(text)
                    self.assertTrue(any(name in item for item in features), features)
            self.assertEqual(self.features(_CELL.format(xc="ca", extra="")), [])
            # Zero_Average_Vxc (PARSEC 1.5's potshift) is the default of a 3-D cell since 2026-11-08: the route of
            # the cell, given or not
            self.assertEqual(self.features(_CELL.format(xc="ca", extra="Zero_Average_Vxc: .true.")), [])
            self.assertEqual(self.features(_CELL.format(xc="ca", extra="Zero_Average_Vxc: .false.")), [])

    def test_an_mpi_rank_solves_on_its_own_device(self) -> None:
        # the replicated k-point drivers of an MPI run (since 2026-11-08): the device of the rank's k-point
        # solver, device_ids[local rank % count]; the CPU plans keep the host
        from types import SimpleNamespace

        from parsec_python.acceleration.Hamiltonian.device_bloch import KPointBackend
        from parsec_python.acceleration.Parallel import mpi

        with mock.patch.dict(os.environ, {"PARSEC_MPI": "1"}), mock.patch.object(
            mpi, "context", return_value=SimpleNamespace(local_rank=5)
        ):
            with mock.patch(
                "parsec_python.acceleration.SCF.kpoint_parallel.resolve_backend",
                return_value=KPointBackend("cupy", (0, 1, 2, 3)),
            ):
                self.assertEqual(periodic_device.hartree_device_id(), 1)
            for backend in ("cpu", "numpy-device"):
                with self.subTest(backend=backend), mock.patch.dict(os.environ, {"PARSEC_KPOINT_BACKEND": backend}):
                    self.assertIsNone(periodic_device.hartree_device_id())

    def test_combinations_stay_routed_by_name(self) -> None:
        kpoints = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2 2 1\nend Monkhorst_Pack_Grid"
        for text, name in (
            (_SLAB.format(extra=kpoints.replace("2 2 1", "2 2")), "k-point"),
            (_SLAB.format(extra="Spin_Polarization: .true."), "spin polarization"),
            (_SLAB.format(extra="Skip_Force: .false."), "forces"),
            (_SLAB.format(extra="Restart_Run: .true."), "Restart_Run of a wire or slab"),
            (_CELL.format(xc="ca", extra="Calc_Hartree_G: .true.\n" + kpoints), "k-point"),
            (_CELL.format(xc="ca", extra="Zero_Average_Vxc: .true.\nSpin_Polarization: .true."), "spin polarization"),
            (_CELL.format(xc="ca", extra="Calc_Hartree_G: .true.\nSkip_Force: .false."), "forces"),
        ):
            with self.subTest(name=name):
                features = self.features(text)
                self.assertTrue(any(name in item for item in features), features)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class AcceleratedDriverTests(unittest.TestCase):
    """The Gamma-point inputs of the features on the accelerated driver = the reference solver."""

    def run_both(self, text: str):
        from parsec_python.acceleration import driver
        from parsec_python.SCF import prepare_periodic_single_point, run_scf

        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(_write_input(Path(folder), text)).problem
            with mock.patch.dict(os.environ, {**_CPU, periodic_device.MODE_ENV: "host"}):
                reference = run_scf(prepare_periodic_single_point(problem))
            with mock.patch.dict(os.environ, _DEVICE):
                system = driver.prepare_single_point(problem, backend="cupy")
                accelerated = driver.run_scf(system)
        return reference, accelerated

    def assert_same(self, reference, accelerated, energy=1.0e-8, levels=1.0e-7) -> None:
        self.assertTrue(reference.converged and accelerated.converged)
        self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=energy)
        np.testing.assert_allclose(accelerated.eigenvalues, reference.eigenvalues, rtol=0, atol=levels)

    def test_slab(self) -> None:
        reference, accelerated = self.run_both(_SLAB.format(extra=""))
        self.assert_same(reference, accelerated)
        details = dict(accelerated.backend.details)
        self.assertIn("slab boundary values", details["hartree_implementation"])
        self.assertIn("on the device", details["hartree_implementation"])

    def test_wire_with_pbe(self) -> None:
        text = _SLAB.format(extra="Correlation_Type: pb").replace(
            "Boundary_Conditions: slab\nbegin Cell_Shape\n  3.0 0.0\n  0.0 3.2\nend Cell_Shape",
            "Boundary_Conditions: wire\nbegin Cell_Shape\n  3.0\nend Cell_Shape",
        )
        reference, accelerated = self.run_both(text)
        self.assert_same(reference, accelerated)

    def test_calc_hartree_g_g_space_and_zero_average(self) -> None:
        for xc, extra in (
            ("ca", "Calc_Hartree_G: .true."),
            ("ca", "Periodic_Local_Potential: g_space\nPeriodic_Alpha_Term: fortran\nZero_Average_Vxc: .true."),
            ("pl", "Zero_Average_Vxc: .true."),
        ):
            with self.subTest(xc=xc, extra=extra):
                reference, accelerated = self.run_both(_CELL.format(xc=xc, extra=extra))
                self.assert_same(reference, accelerated)
                if "Zero_Average" in extra:
                    self.assertAlmostEqual(float(np.mean(accelerated.xc_potential)), 0.0, places=12)
                if xc == "pl":
                    self.assertIn(
                        "zero average", dict(accelerated.backend.details)["xc_implementation"]
                    )


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class AcceleratedCLITests(unittest.TestCase):
    def test_calc_hartree_g_cell_runs_accelerated_with_fortrans_caution_line(self) -> None:
        import contextlib
        import io

        from parsec_python.acceleration.cli import main

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = _write_input(Path(folder), _CELL.format(xc="ca", extra="Calc_Hartree_G: .true."))
            errors = io.StringIO()
            with mock.patch.dict(os.environ, _DEVICE), contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
                status = main([str(path), "--quiet"])
            self.assertEqual(status, 0, errors.getvalue())
            self.assertNotIn("running the reference parsec_python.cli", errors.getvalue())
            text = (Path(folder) / "parsec.out").read_text()
            self.assertIn("CAUTION: The Hartree potential will be calculated in reciprocal space", text)
            self.assertIn("Acceleration backend", text)
            with np.load(Path(folder) / "parsec_python_results.npz", allow_pickle=True) as archive:
                details = dict(zip(archive["backend_detail_keys"].tolist(), archive["backend_detail_values"].tolist()))
            self.assertIn("Calc_Hartree_G, CuPy FFT on the device", details["hartree_implementation"])


class ZeroAverageTests(unittest.TestCase):
    def test_every_channel_loses_its_mean(self) -> None:
        from parsec_python.acceleration.V_xc.device_xc import ZeroAverageXC
        from parsec_python.V_xc import SpinPolarizedXCResult, XCResult

        scalar = XCResult(
            potential=np.array([1.0, 2.0, 6.0]),
            energy_per_electron=np.zeros(3),
            energy_density=np.array([0.1, 0.2, 0.3]),
            total_energy=-1.5,
        )
        wrapped = ZeroAverageXC(_Callable(scalar, "CuPy float64 PW92-LDA"))
        result = wrapped(np.ones(3))
        np.testing.assert_allclose(result.potential, [-2.0, -1.0, 3.0])
        self.assertEqual(result.total_energy, -1.5)
        self.assertIn("Zero_Average_Vxc", wrapped.implementation_label)
        spin = SpinPolarizedXCResult(
            potential_up=np.array([1.0, 3.0]),
            potential_down=np.array([-1.0, 0.0]),
            energy_density=np.zeros(2),
            total_energy=0.25,
        )
        result = ZeroAverageXC(_Callable(spin, "spin"))(np.ones(2), np.ones(2))
        np.testing.assert_allclose(result.potential_up, [-1.0, 1.0])
        np.testing.assert_allclose(result.potential_down, [-0.5, 0.5])


class _Callable:
    def __init__(self, result, label: str) -> None:
        self.result = result
        self.implementation_label = label

    def __call__(self, *_densities):
        return self.result


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class PeriodicSpinXCTests(unittest.TestCase):
    """acceleration/V_xc/periodic_xc.py: the spin functionals of a cell in the reference drivers."""

    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)

    def system(self, xc: str, extra: str = ""):
        from parsec_python.SCF import prepare_periodic_single_point

        folder = self.folder / xc
        folder.mkdir(exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = _CELL.format(xc=xc, extra="Spin_Polarization: .true.\n" + extra)
            problem = parse_parsec_input(_write_input(folder, text)).problem
            return prepare_periodic_single_point(problem)

    def test_parity_functionals_on_the_device(self) -> None:
        from parsec_python.acceleration.V_xc import periodic_xc

        periodic_xc.clear_cache()
        for xc in ("pl", "pw"):
            system = self.system(xc)
            density = np.asarray(system.initial_density)
            up, down = 0.6 * density, 0.4 * density
            with mock.patch.dict(os.environ, _CPU):
                self.assertIsNone(periodic_xc.device_spin_evaluator(system))
                host = system.evaluate_xc_spin_polarized(up, down)
                host_spin = system.evaluate_spin_xc(up, down)
            with mock.patch.dict(os.environ, _DEVICE):
                self.assertIsNotNone(periodic_xc.device_spin_evaluator(system))
                device = system.evaluate_xc_spin_polarized(up, down)
                device_spin = system.evaluate_spin_xc(up, down)
            with self.subTest(xc=xc):
                for left, right in ((device, host), (device_spin, host_spin)):
                    np.testing.assert_allclose(left.potential_up, right.potential_up, rtol=0, atol=1.0e-13)
                    np.testing.assert_allclose(left.potential_down, right.potential_down, rtol=0, atol=1.0e-13)
                    self.assertAlmostEqual(left.total_energy, right.total_energy, delta=1.0e-12)

    def test_ca_pbe_and_spinors_keep_the_host(self) -> None:
        from parsec_python.acceleration.V_xc import periodic_xc

        with mock.patch.dict(os.environ, {**_DEVICE, "PARSEC_CUPY_XC": "auto"}):
            self.assertIsNone(periodic_xc.device_spin_evaluator(self.system("ca")))
            pbe = self.system("pb")
            self.assertIsNone(periodic_xc.device_spin_evaluator(pbe))
        with mock.patch.dict(os.environ, {**_DEVICE, "PARSEC_CUPY_XC": "on"}):
            self.assertIsNotNone(periodic_xc.device_spin_evaluator(pbe))
            density = np.asarray(pbe.initial_density)
            device = pbe.evaluate_spin_xc(0.6 * density, 0.4 * density)
        with mock.patch.dict(os.environ, _CPU):
            host = pbe.evaluate_spin_xc(0.6 * density, 0.4 * density)
        # the same pbe_spin formula on the device
        np.testing.assert_allclose(device.potential_up, host.potential_up, rtol=0, atol=1.0e-12)
        system = self.system("pl")
        spinor = replace(system, input=replace(system.input, scf=replace(system.input.scf, noncollinear=True)))
        with mock.patch.dict(os.environ, _DEVICE):
            self.assertIsNone(periodic_xc.device_spin_evaluator(spinor))

    def test_the_hook_is_installed(self) -> None:
        import parsec_python.acceleration  # noqa: F401
        from parsec_python.acceleration.V_xc import periodic_xc
        from parsec_python.SCF import pbc

        self.assertIs(pbc._spin_xc_provider, periodic_xc.provide_spin_xc)


class GammaSpinRouteTests(unittest.TestCase):
    def test_wire_and_slab_take_the_kpoint_driver_on_a_device(self) -> None:
        from parsec_python.cli import _gamma_spin_on_kpoint_driver

        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            translation = parse_parsec_input(
                _write_input(Path(folder), _SLAB.format(extra="Spin_Polarization: .true."))
            )
        with mock.patch("parsec_python.cli.parallel_backend_uses_device", return_value=True):
            self.assertTrue(_gamma_spin_on_kpoint_driver(translation, True, False))
            with mock.patch.dict(os.environ, {"PARSEC_WIRE_SLAB_GAMMA_SPIN": "gamma"}):
                self.assertFalse(_gamma_spin_on_kpoint_driver(translation, True, False))
            # an MPI run keeps the Gamma driver of a wire or slab
            with mock.patch.dict(os.environ, {"PARSEC_MPI": "1"}):
                self.assertFalse(_gamma_spin_on_kpoint_driver(translation, True, False))
        with mock.patch("parsec_python.cli.parallel_backend_uses_device", return_value=False):
            self.assertFalse(_gamma_spin_on_kpoint_driver(translation, True, False))


if __name__ == "__main__":
    unittest.main()
