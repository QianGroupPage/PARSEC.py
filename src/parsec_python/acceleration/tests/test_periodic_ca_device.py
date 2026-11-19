"""CA-LDA of periodic cells on the device (2026-11-18).

The default functional of a periodic cell was evaluated with NumPy on the host over the full grid at every SCF
step, on every periodic route (the native C++ evaluator of clusters needs the native Hartree backend, which a cell
never has).  :mod:`parsec_python.acceleration.V_xc.device_xc` now evaluates it on the device:

* the switches: ``PARSEC_PERIODIC_CA_XC`` (``auto`` | ``host``) and ``PARSEC_CUPY_XC=off`` keep the host; a
  cluster keeps its native evaluator whatever they say;
* the kernels ``lda_ca`` and ``lda_spin_ca`` equal :func:`~parsec_python.V_xc.ca_lda.ca_lda`,
  :func:`~parsec_python.V_xc.ca_lda_spin.ca_lda_spin` and
  :func:`~parsec_python.V_xc.ca_lda.ca_lda_spin_polarized` to round-off (both density branches, empty points and
  channels, a frozen core density);
* the provider of the reference drivers (:mod:`parsec_python.acceleration.V_xc.periodic_xc`) takes CA-LDA in a
  run of one process too, and both spin kinds, and a k-point SCF gives the host's numbers;
* the accelerated driver evaluates it on the orbits of the sectors of a Gamma cell (silicon, Td), with the
  host's numbers, and tells the sector memory rules its bytes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.V_xc import device_xc, periodic_xc
from parsec_python.V_xc.ca_lda import ca_lda, ca_lda_spin_polarized
from parsec_python.V_xc.ca_lda_spin import ca_lda_spin

_DEVICE = {
    "PARSEC_KPOINT_BACKEND": "cupy",
    "PARSEC_PERIODIC_HARTREE": "auto",
    "PARSEC_CUPY_XC": "auto",
    "PARSEC_PERIODIC_CA_XC": "auto",
}
_HOST = {**_DEVICE, "PARSEC_PERIODIC_CA_XC": "host"}


def _cell_system(xc: str = "ca", extra: str = ""):
    """The hydrogen cell of test_periodic_device_fields, prepared by the reference periodic driver."""

    from parsec_python.acceleration.tests.test_periodic_device_fields import _CELL, _write_input
    from parsec_python.Input import parse_parsec_input
    from parsec_python.SCF import prepare_periodic_single_point

    folder = Path(tempfile.mkdtemp(prefix="parsec_ca_"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(_write_input(folder, _CELL.format(xc=xc, extra=extra))).problem
            return problem, prepare_periodic_single_point(problem)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _densities(count: int, seed: int = 7) -> np.ndarray:
    """Densities from 1e-7 to 30 electrons/bohr^3 (r_s from about 0.2 to 130), with empty points."""

    random = np.random.default_rng(seed)
    values = 10.0 ** random.uniform(-7.0, 1.5, size=count)
    values[::17] = 0.0
    values[5::23] = -3.0e-16
    return values


class SelectionTests(unittest.TestCase):
    def test_switch_values(self) -> None:
        for raw, mode in (("", "auto"), ("auto", "auto"), ("device", "auto"), ("host", "host"), ("off", "host")):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": raw}):
                self.assertEqual(device_xc.periodic_ca_xc_mode(), mode)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": "sometimes"}):
            with self.assertRaises(ValueError):
                device_xc.periodic_ca_xc_mode()

    def test_ca_takes_the_device_in_periodic_cells_only(self) -> None:
        selected = device_xc.periodic_device_xc_selected
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": ""}):
            for spin in (False, True):
                self.assertEqual([selected("ca", spin_polarized=spin, mode=m) for m in ("auto", "on", "off")],
                                 [True, True, False])
                # a cluster keeps the native evaluator
                for mode in ("auto", "on", "off"):
                    self.assertFalse(device_xc.device_xc_selected("ca", spin_polarized=spin, mode=mode))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": "host"}):
            for mode in ("auto", "on", "off"):
                self.assertFalse(selected("ca", mode=mode))
                self.assertFalse(selected("ca", spin_polarized=True, mode=mode))
        # every other functional follows PARSEC_CUPY_XC as before
        for functional in ("pw92", "pbe", "pw91", "lb94", "xalpha"):
            for mode in ("auto", "on", "off"):
                self.assertEqual(selected(functional, mode=mode), device_xc.device_xc_selected(functional, mode=mode))

    def test_the_spin_drivers_and_their_forms(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": ""}):
            for kind in ("spin", "spin_polarized"):
                self.assertEqual(
                    [periodic_xc.periodic_spin_device_selected("ca", mode=m, kind=kind) for m in ("auto", "on", "off")],
                    [True, True, False],
                )
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": "host"}):
            self.assertFalse(periodic_xc.periodic_spin_device_selected("ca", mode="on", kind="spin_polarized"))
        self.assertEqual(periodic_xc._form("ca", "spin_polarized"), "spin_polarized")
        self.assertEqual(periodic_xc._form("ca", "spin"), "pbe_spin")
        self.assertEqual(periodic_xc._form("pbe", "spin_polarized"), "ab")
        self.assertEqual(device_xc.kernel_names("ca", spin_polarized=False), ("lda_ca",))
        self.assertEqual(device_xc.kernel_names("ca", spin_polarized=True), ("lda_spin_ca",))
        source = device_xc.kernel_source()
        for name in ("lda_ca", "lda_spin_ca"):
            self.assertIn(f"KERNEL_{name}", source)

    def test_a_cluster_grid_keeps_ca_off_the_device_evaluators(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=3.0))
        reference = SimpleNamespace(
            input=SimpleNamespace(scf=SimpleNamespace(xc_functional="ca", spin_polarized=False)),
            grid=grid,
            core_density=None,
        )
        for mode in ("auto", "on"):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": mode}):
                self.assertEqual(device_xc.accelerated_xc_evaluators(reference, backend="cupy"), (None, None))

    def test_the_provider_rule_of_one_process(self) -> None:
        # one process: CA-LDA asks for the device of the k-point plan, PBE keeps the readable functional
        _, system = _cell_system("ca")
        calls = []

        def get(system_, functional, device_id, *, spin=True, form="pbe_spin"):
            calls.append((functional, device_id, spin))
            return "evaluator"

        with mock.patch.object(periodic_xc._EVALUATORS, "get", side_effect=get), mock.patch(
            "parsec_python.acceleration.Hartree.periodic_device.hartree_device_id", return_value=3
        ), mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": "", "PARSEC_CUPY_XC": "auto"}), mock.patch(
            "parsec_python.acceleration.Parallel.mpi.requested", return_value=False
        ):
            self.assertEqual(periodic_xc.device_unpolarized_evaluator(system), "evaluator")
            self.assertEqual(calls, [("ca", 3, False)])
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_CA_XC": "host"}):
                self.assertIsNone(periodic_xc.device_unpolarized_evaluator(system))
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": "off"}):
                self.assertIsNone(periodic_xc.device_unpolarized_evaluator(system))
            pbe = replace(system, input=replace(system.input, scf=replace(system.input.scf, xc_functional="pbe")))
            self.assertIsNone(periodic_xc.device_unpolarized_evaluator(pbe))
        self.assertEqual(len(calls), 1)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class KernelTests(unittest.TestCase):
    """The device kernels against the NumPy functionals on the full grid of a cell."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.problem, cls.system = _cell_system("ca")
        cls.grid = cls.system.grid

    def assert_close(self, device, host, rtol=2.0e-14) -> None:
        device = np.asarray(device)
        host = np.asarray(host)
        np.testing.assert_allclose(device, host, rtol=rtol, atol=1.0e-300)

    def test_unpolarized(self) -> None:
        size = int(self.grid.size)
        density = _densities(size)
        core = 0.01 * _densities(size, seed=3).clip(0.0)
        for frozen in (None, core):
            evaluator = device_xc.DeviceXCEvaluator("ca", self.grid, core_density=frozen)
            self.assertEqual(evaluator.implementation_label, "CuPy float64 CA-LDA")
            device = evaluator(density)
            host = ca_lda(density, self.grid.volume_element, frozen)
            for name in ("potential", "energy_per_electron", "energy_density"):
                self.assert_close(getattr(device, name), getattr(host, name))
            self.assertAlmostEqual(device.total_energy, host.total_energy, delta=1.0e-13 * abs(host.total_energy))
            # both branches of Perdew-Zunger are taken
            rs = (0.75 / (np.pi * density[density > 0.0])) ** (1.0 / 3.0)
            self.assertTrue(np.any(rs < 1.0) and np.any(rs >= 1.0))
        with self.assertRaisesRegex(ValueError, "CA-LDA requires a nonnegative density"):
            device_xc.DeviceXCEvaluator("ca", self.grid)(np.full(size, -1.0e-12))

    def test_spin_forms(self) -> None:
        size = int(self.grid.size)
        up = _densities(size, seed=11).clip(0.0)
        down = _densities(size, seed=12).clip(0.0)
        down[::7] = 0.0  # fully polarized points
        up[3::29] = 0.0
        core = 0.01 * _densities(size, seed=13).clip(0.0)
        for frozen in (None, core):
            evaluator = device_xc.DeviceSpinXCEvaluator("ca", self.grid, core_density=frozen)
            for form, reference in (("pbe_spin", ca_lda_spin), ("spin_polarized", ca_lda_spin_polarized)):
                evaluator.form = form
                device = evaluator(up, down)
                host = reference(up, down, self.grid.volume_element, frozen)
                with self.subTest(form=form, core=frozen is not None):
                    for name in ("potential_up", "potential_down", "energy_density"):
                        self.assert_close(getattr(device, name), getattr(host, name))
                    self.assertAlmostEqual(
                        device.total_energy, host.total_energy, delta=1.0e-13 * abs(host.total_energy)
                    )
        # a channel below zero: ca_lda_spin refuses it, ca_lda_spin_polarized takes it as empty
        evaluator = device_xc.DeviceSpinXCEvaluator("ca", self.grid)
        negative = up.copy()
        negative[1] = -1.0e-12
        with self.assertRaises(ValueError):
            ca_lda_spin(negative, down, self.grid.volume_element)
        with self.assertRaises(ValueError):
            evaluator(negative, down)
        evaluator.form = "spin_polarized"
        device = evaluator(negative, down)
        host = ca_lda_spin_polarized(negative, down, self.grid.volume_element)
        self.assert_close(device.potential_up, host.potential_up)
        self.assert_close(device.potential_down, host.potential_down)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class ReferenceDriverTests(unittest.TestCase):
    """The provider of :mod:`parsec_python.SCF.pbc` (k-point loops, the Gamma and spin drivers)."""

    def test_every_kind_of_evaluation_takes_the_device(self) -> None:
        _, system = _cell_system("ca")
        density = np.asarray(system.initial_density)
        up, down = 0.6 * density, 0.4 * density
        periodic_xc.clear_cache()
        with mock.patch.dict(os.environ, _DEVICE):
            self.assertIsNotNone(periodic_xc.device_unpolarized_evaluator(system))
            self.assertIsNotNone(periodic_xc.device_spin_evaluator(system, "spin"))
            self.assertIsNotNone(periodic_xc.device_spin_evaluator(system, "spin_polarized"))
            device = (
                system.evaluate_xc(density),
                system.evaluate_spin_xc(up, down),
                system.evaluate_xc_spin_polarized(up, down),
            )
        with mock.patch.dict(os.environ, _HOST):
            self.assertIsNone(periodic_xc.device_unpolarized_evaluator(system))
            self.assertIsNone(periodic_xc.device_spin_evaluator(system, "spin"))
            host = (
                system.evaluate_xc(density),
                system.evaluate_spin_xc(up, down),
                system.evaluate_xc_spin_polarized(up, down),
            )
        self.assertNotIsInstance(host[0].potential, type(None))
        np.testing.assert_allclose(device[0].potential, host[0].potential, rtol=2.0e-14, atol=0.0)
        self.assertAlmostEqual(device[0].total_energy, host[0].total_energy, delta=1.0e-13)
        for left, right in zip(device[1:], host[1:]):
            np.testing.assert_allclose(left.potential_up, right.potential_up, rtol=2.0e-14, atol=0.0)
            np.testing.assert_allclose(left.potential_down, right.potential_down, rtol=2.0e-14, atol=0.0)
            self.assertAlmostEqual(left.total_energy, right.total_energy, delta=1.0e-13)

    def test_zero_average(self) -> None:
        _, system = _cell_system("ca", "Zero_Average_Vxc: .true.")
        density = np.asarray(system.initial_density)
        with mock.patch.dict(os.environ, _DEVICE):
            device = system.evaluate_xc(density)
        with mock.patch.dict(os.environ, _HOST):
            host = system.evaluate_xc(density)
        self.assertAlmostEqual(float(np.mean(device.potential)), 0.0, places=12)
        np.testing.assert_allclose(device.potential, host.potential, rtol=0.0, atol=1.0e-13)

    def test_a_kpoint_scf_of_one_process(self) -> None:
        from parsec_python.SCF import run_scf_kpoints

        problem, system = _cell_system("ca")
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.array(b, dtype=float) * np.pi / side for b in np.ndindex(2, 2, 2)])
        weights = np.full(8, 1.0 / 8.0)
        results = {}
        for label, environment in (("device", _DEVICE), ("host", _HOST)):
            calls = []
            original = periodic_xc.device_unpolarized_evaluator

            def counted(system_, original=original, calls=calls):
                evaluator = original(system_)
                calls.append(evaluator is not None)
                return evaluator

            with mock.patch.dict(os.environ, environment), mock.patch.object(
                periodic_xc, "device_unpolarized_evaluator", counted
            ), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[label] = run_scf_kpoints(system, k_points, weights)
            self.assertTrue(calls)
            self.assertTrue(all(calls) if label == "device" else not any(calls), label)
        device, host = results["device"], results["host"]
        self.assertTrue(device.converged and host.converged)
        self.assertEqual(device.iterations, host.iterations)
        self.assertAlmostEqual(device.energies.total, host.energies.total, delta=1.0e-10)
        np.testing.assert_allclose(np.asarray(device.eigenvalues), np.asarray(host.eigenvalues), rtol=0, atol=1.0e-9)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class AcceleratedDriverTests(unittest.TestCase):
    """A Gamma cell in the sectors of its point group: CA-LDA on the orbits on the device."""

    def solve(self, environment):
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        problem = silicon("ca")
        with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            resident = dict(getattr(system.backend.symmetry_eigensolver, "resident_device_bytes", {}) or {})
            result = driver.run_scf(system)
        return result, dict(result.backend.details), resident

    def test_silicon_on_the_orbits(self) -> None:
        device, device_details, resident = self.solve({"PARSEC_CUPY_XC": "auto", "PARSEC_PERIODIC_CA_XC": "auto"})
        # (with the zero average of Zero_Average_Vxc, the default of a cell)
        self.assertTrue(
            device_details["xc_implementation"].startswith("CuPy float64 CA-LDA on the symmetry orbits"),
            device_details["xc_implementation"],
        )
        self.assertIn("weights", device_details["symmetry_detection"])
        self.assertEqual(len(resident), 1)
        self.assertGreater(next(iter(resident.values())), 0)
        host, host_details, host_resident = self.solve({"PARSEC_PERIODIC_CA_XC": "host"})
        self.assertEqual(host_details["xc_implementation"], "vectorized NumPy CA/PZ-LDA")
        self.assertEqual(host_resident, {})
        self.assertTrue(device.converged and host.converged)
        self.assertEqual(device.iterations, host.iterations)
        for field in ("total", "exchange_correlation", "integral_vxc_rho", "hartree", "eigenvalue"):
            self.assertAlmostEqual(
                getattr(device.energies, field), getattr(host.energies, field), delta=1.0e-10, msg=field
            )
        np.testing.assert_allclose(np.sort(device.eigenvalues), np.sort(host.eigenvalues), rtol=0, atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
