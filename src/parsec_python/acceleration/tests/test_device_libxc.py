"""libxc's functionals on the GPU (``acceleration/V_xc/device_libxc.py``, libxc's CUDA build).

Needs CuPy and a CUDA build of libxc >= 7.1 (``PARSEC_LIBXC_LIBRARY``).  Host
and device run the same libxc source on two processors, so they agree to
round-off:

* pointwise (``LibxcSemilocal`` on NumPy and on CuPy arrays): LDA, GGA,
  hybrids' semilocal parts, meta-GGAs with tau, both spin forms;
* as grid fields (full grid with a core density, symmetry orbits, spin
  channels) against :func:`parsec_python.V_xc.semilocal.semilocal_xc`;
* SCF runs of H2 / the H atom on the accelerated path against the
  reference path (libxc's PBE, unpolarized and spin, full grid and sectors;
  libxc's B3LYP through the device Fock loop), the sectors to 1e-9 Ry
  (second order in the SCF residual, as in ``test_hybrid_device.py``).
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.models import Atom, GridSettings


def _device_libxc() -> bool:
    if not cupy_available():
        return False
    from parsec_python.acceleration.V_xc.device_libxc import device_libxc_available

    return device_libxc_available()


HAVE_DEVICE_LIBXC = _device_libxc()
POINT_TOLERANCE = 1.0e-12
HOLE_TOLERANCE = 1.0e-10
FIELD_TOLERANCE = 1.0e-12


def _gaussians(coordinates, centers, width, weights):
    values = np.zeros(len(coordinates))
    for center, weight in zip(centers, weights):
        difference = coordinates - np.asarray(center)
        values += weight * np.exp(-np.sum(difference * difference, axis=1) / width)
    return values


def _relative_points(reference, value) -> float:
    reference, value = np.asarray(reference), np.asarray(value)
    scale = np.max(np.abs(reference))
    if scale == 0.0:
        return float(np.max(np.abs(value)))
    return float(np.max(np.abs(reference - value)) / scale)


def _relative_field(reference, value) -> float:
    reference = np.asarray(getattr(reference, "values", reference))
    value = np.asarray(getattr(value, "values", value))
    return float(np.max(np.abs(reference - value)) / max(np.max(np.abs(reference)), 1e-300))


def _points():
    rng = np.random.default_rng(11)
    count = 500
    n = 10.0 ** rng.uniform(-3.0, 2.0, count)
    s = rng.uniform(0.0, 4.0, count)
    sigma = (2.0 * (3.0 * np.pi**2 * n) ** (1.0 / 3.0) * n * s) ** 2
    tau_w = sigma / (8.0 * n)
    tau = tau_w + 0.3 * (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0) * n ** (5.0 / 3.0) * rng.uniform(0.1, 3.0, count)
    up = n * rng.uniform(0.3, 0.7, count)
    down = n - up
    s_uu, s_dd = sigma * 0.3, sigma * 0.2
    s_ud = np.sqrt(s_uu * s_dd) * rng.uniform(-1.0, 1.0, count)
    # Each channel's tau above its von Weizsaecker bound (libxc's polarized SCAN returns NaN below it).
    def channel_tau(rho, s):
        uniform = 0.3 * (6.0 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
        return s / (8.0 * rho) + uniform * rng.uniform(0.1, 3.0, count)

    return (n, sigma, tau), (up, down, s_uu, s_ud, s_dd, channel_tau(up, s_uu), channel_tau(down, s_dd))


@unittest.skipUnless(HAVE_DEVICE_LIBXC, "CuPy and a CUDA build of libxc >= 7.1 (PARSEC_LIBXC_LIBRARY) are required")
class DeviceLibxcPointTests(unittest.TestCase):
    SPECS = (
        "libxc:lda_x+lda_c_pw",
        "libxc:gga_x_pbe+gga_c_pbe",
        "libxc:hyb_gga_xc_b3lyp",
        "libxc:hyb_gga_xc_hse06",
        "libxc:hyb_gga_xc_cam_b3lyp",
        "libxc:mgga_x_scan+mgga_c_scan",
        "libxc:mgga_x_r2scan+mgga_c_r2scan;fhc=off",
        "libxc:hyb_mgga_xc_tpssh",
    )

    def test_points(self) -> None:
        import cupy as cp

        from parsec_python.V_xc.libxc import LibxcSemilocal

        (n, sigma, tau), polarized = _points()
        for spec in self.SPECS:
            evaluator = LibxcSemilocal(spec)
            self.assertTrue(evaluator.supports_device)
            meta = evaluator.family == "mgga"
            arguments = (n, sigma, tau, None) if meta else (n, sigma)
            spin_arguments = polarized + (None, None) if meta else polarized[:5]
            for label, host_arguments in (("unpolarized", arguments), ("polarized", spin_arguments)):
                method = evaluator.unpolarized if label == "unpolarized" else evaluator.polarized
                host = method(*host_arguments)
                device = method(*(None if a is None else cp.asarray(a) for a in host_arguments))
                # libxc's wPBEh hole (HSE) evaluates erfc/expint expansions whose last digits differ
                # between the host's libm and CUDA (2e-12 of vsigma measured).
                tolerance = HOLE_TOLERANCE if "hse" in spec else POINT_TOLERANCE
                for index, (reference, value) in enumerate(zip(host, device)):
                    if reference is None:
                        self.assertIsNone(value)
                        continue
                    self.assertIsInstance(value, cp.ndarray)
                    self.assertLess(_relative_points(reference, cp.asnumpy(value)), tolerance, (spec, label, index))

    def test_tau_below_the_binding_floor(self) -> None:
        # tau below 1e-20 and below tau_W (TPSS's z of round-off): the binding floors tau at 1e-20
        # on both sides (libxc's own threshold acts on the host only), vtau = 0 there.
        import cupy as cp

        from parsec_python.V_xc.libxc import LibxcSemilocal

        rho = np.array([0.125, 0.125, 0.01, 0.3])
        sigma = np.array([1e-30, 1e-22, 1e-6, 0.02])  # z 1e-10, 1e-2, > 1: not at the kink z = 1
        tau = np.array([1e-23, 1e-25, 1e-21, 0.4])
        for spec in ("libxc:mgga_x_tpss+mgga_c_tpss", "libxc:mgga_x_tpss+mgga_c_tpss;tau=1e-14", "libxc:mgga_x_scan+mgga_c_scan"):
            evaluator = LibxcSemilocal(spec)
            host = evaluator.unpolarized(rho, sigma, tau, None)
            device = evaluator.unpolarized(cp.asarray(rho), cp.asarray(sigma), cp.asarray(tau), None)
            floored = evaluator.unpolarized(rho, sigma, np.maximum(tau, 1e-20), None)
            np.testing.assert_array_equal(host[3][:3], 0.0)
            np.testing.assert_allclose(host[0], floored[0], rtol=1e-14, atol=0)
            for index in range(4):
                self.assertLess(_relative_points(host[index], cp.asnumpy(device[index])), POINT_TOLERANCE, (spec, index))
            half = [np.ascontiguousarray(value / 2.0) for value in (rho, sigma, tau)]
            polarized = (half[0], half[0], half[1] / 2, half[1] / 2, half[1] / 2, half[2], half[2], None, None)
            host = evaluator.polarized(*polarized)
            device = evaluator.polarized(*(None if a is None else cp.asarray(a) for a in polarized))
            for index in range(8):
                self.assertLess(_relative_points(host[index], cp.asnumpy(device[index])), POINT_TOLERANCE, (spec, index))

    def test_second_device(self) -> None:
        import cupy as cp

        from parsec_python.V_xc.libxc import LibxcSemilocal

        if cp.cuda.runtime.getDeviceCount() < 2:
            self.skipTest("one device")
        (n, sigma, _), _ = _points()
        evaluator = LibxcSemilocal("libxc:gga_x_pbe+gga_c_pbe")
        host = evaluator.unpolarized(n, sigma)
        with cp.cuda.Device(1):
            arrays = (cp.asarray(n), cp.asarray(sigma))
        # Called with device 0 current: the arrays of device 1 are evaluated there.
        with cp.cuda.Device(0):
            device = evaluator.unpolarized(*arrays)
        self.assertEqual(device[0].data.device_id, 1)
        self.assertLess(_relative_points(host[1], cp.asnumpy(device[1])), POINT_TOLERANCE)


@unittest.skipUnless(HAVE_DEVICE_LIBXC, "CuPy and a CUDA build of libxc >= 7.1 (PARSEC_LIBXC_LIBRARY) are required")
class DeviceLibxcFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.45, radius=5.0, expansion_order=12))
        coordinates = cls.grid.coordinates
        cls.density = _gaussians(coordinates, ((0.3, -0.2, 0.1), (-1.1, 0.8, 0.4)), 1.4, (1.1, 0.6))
        # Both channels hold a share of the density: |zeta| < 1 everywhere.  (Where one channel is almost
        # empty, libxc 7.1's polarized LYP loses digits, on the host and the device differently: README.)
        cls.up = 0.55 * cls.density + _gaussians(coordinates, ((0.3, -0.2, 0.1),), 1.2, (0.2,))
        cls.down = 0.45 * cls.density + _gaussians(coordinates, ((-1.1, 0.8, 0.4),), 1.6, (0.2,))
        cls.core = _gaussians(coordinates, ((0.3, -0.2, 0.1),), 0.3, (0.05,))

    def test_fields(self) -> None:
        from parsec_python.V_xc.libxc import libxc_semilocal
        from parsec_python.V_xc.semilocal import semilocal_xc, semilocal_xc_spin
        from parsec_python.acceleration.V_xc.device_libxc import (
            DeviceLibxcXCEvaluator,
            DeviceSpinLibxcXCEvaluator,
        )

        for spec in ("libxc:lda_x+lda_c_pz", "libxc:gga_x_pbe+gga_c_pbe", "libxc:hyb_gga_xc_b3lyp"):
            evaluator = libxc_semilocal(spec)
            reference = semilocal_xc(evaluator, self.density, self.grid, self.core)
            value = DeviceLibxcXCEvaluator(evaluator, self.grid, core_density=self.core)(self.density)
            for name in ("potential", "energy_density", "energy_per_electron"):
                self.assertLess(
                    _relative_field(getattr(reference, name), getattr(value, name)), FIELD_TOLERANCE, (spec, name)
                )
            self.assertAlmostEqual(
                value.total_energy, reference.total_energy, delta=1e-13 * abs(reference.total_energy), msg=spec
            )
            spin_reference = semilocal_xc_spin(evaluator, self.up, self.down, self.grid, self.core)
            spin = DeviceSpinLibxcXCEvaluator(evaluator, self.grid, core_density=self.core)(self.up, self.down)
            for name in ("potential_up", "potential_down", "energy_density"):
                self.assertLess(
                    _relative_field(getattr(spin_reference, name), getattr(spin, name)), FIELD_TOLERANCE, (spec, name)
                )
            self.assertAlmostEqual(
                spin.total_energy, spin_reference.total_energy, delta=1e-13 * abs(spin_reference.total_energy)
            )

    def test_symmetry_orbits(self) -> None:
        from parsec_python.V_xc.libxc import libxc_semilocal
        from parsec_python.V_xc.semilocal import semilocal_xc, semilocal_xc_spin
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry.axis_reflection import AxisReflectionReduction
        from parsec_python.acceleration.V_xc.device_libxc import (
            DeviceLibxcXCEvaluator,
            DeviceSpinLibxcXCEvaluator,
        )

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=8, shift=(0.5, 0.5, 0.5)))
        atoms = (Atom("O", (0.0, 0.0, 0.3)), Atom("H", (1.2, 0.5, 0.4)), Atom("H", (-1.2, -0.5, 0.4)))
        reducer = SymmetrySCFReducer(AxisReflectionReduction.detect(grid, atoms))
        full = _gaussians(grid.coordinates, [a.position for a in atoms], 0.9, (1.0, 0.4, 0.4))
        wedge = reducer.from_full(full)
        evaluator = libxc_semilocal("libxc:gga_x_pbe+gga_c_pbe")
        reference = semilocal_xc(evaluator, reducer.to_full(wedge), grid)
        value = DeviceLibxcXCEvaluator(evaluator, grid, reducer=reducer)(wedge)
        self.assertLess(_relative_field(reducer.wedge_values(reference.potential), value.potential), FIELD_TOLERANCE)
        self.assertAlmostEqual(value.total_energy, reference.total_energy, delta=1e-13 * abs(reference.total_energy))
        up = reducer.from_full(0.6 * full)
        down = reducer.from_full(0.4 * full)
        spin_reference = semilocal_xc_spin(evaluator, reducer.to_full(up), reducer.to_full(down), grid)
        spin = DeviceSpinLibxcXCEvaluator(evaluator, grid, reducer=reducer)(up, down)
        self.assertLess(
            _relative_field(reducer.wedge_values(spin_reference.potential_up), spin.potential_up), FIELD_TOLERANCE
        )
        self.assertAlmostEqual(spin.total_energy, spin_reference.total_energy, delta=1e-13 * abs(spin_reference.total_energy))

    def test_selection(self) -> None:
        from types import SimpleNamespace

        from parsec_python.models import SCFSettings
        from parsec_python.acceleration.V_xc import accelerated_xc_evaluators
        from parsec_python.acceleration.V_xc.device_libxc import DeviceLibxcXCEvaluator, DeviceSpinLibxcXCEvaluator

        def reference(**keywords):
            scf = SCFSettings(xc_functional="libxc:gga_x_pbe+gga_c_pbe", **keywords)
            return SimpleNamespace(input=SimpleNamespace(scf=scf), grid=self.grid, core_density=None)

        unpolarized, spin = accelerated_xc_evaluators(reference(), backend="cupy")
        self.assertIsInstance(unpolarized, DeviceLibxcXCEvaluator)
        self.assertIsNone(spin)
        self.assertIn("libxc CUDA", unpolarized.implementation_label)
        unpolarized, spin = accelerated_xc_evaluators(reference(spin_polarized=True), backend="cupy")
        self.assertIsInstance(spin, DeviceSpinLibxcXCEvaluator)
        self.assertGreater(spin.planned_device_bytes, 0)
        self.assertEqual(accelerated_xc_evaluators(reference(), backend="numpy"), (None, None))


@unittest.skipUnless(HAVE_DEVICE_LIBXC, "CuPy and a CUDA build of libxc >= 7.1 (PARSEC_LIBXC_LIBRARY) are required")
class DeviceLibxcSCFTests(unittest.TestCase):
    def setUp(self) -> None:
        from parsec_python.acceleration.tests.test_hybrid_device import _Inputs

        self.inputs = _Inputs()
        self.inputs.setUp()

    def tearDown(self) -> None:
        self.inputs.tearDown()

    def problem(self, lines, *, states=3, atoms=None, extra="", species=""):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration.tests.test_hybrid_device import BASE, H2

        text = BASE.format(states=states, atoms=atoms or H2, functional="pbe", extra=extra, species=species)
        path = self.inputs.directory / "libxc.in"
        path.write_text(text.replace("Correlation_Type: pbe\n", lines))
        return parse_parsec_input(path).problem

    def test_pbe_full_grid_and_sectors(self) -> None:
        problem = self.problem("LibXC_Functional: gga_x_pbe gga_c_pbe\n")
        for symmetry, tolerance in (("off", 1e-11), ("auto", 1e-9)):
            reference, system, accelerated, _ = self.inputs.both(problem, symmetry=symmetry)
            self.assertTrue(reference.converged and accelerated.converged)
            self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=tolerance, msg=symmetry)
            np.testing.assert_allclose(accelerated.eigenvalues, reference.eigenvalues, atol=1e-6)

    def test_spin_polarized_atom(self) -> None:
        for sectors in ("off", "auto"):
            problem = self.problem(
                "LibXC_Functional: gga_x_pbe gga_c_pbe\n",
                states=2,
                atoms="  0.0 0.0 0.0\n",
                species="Initial_Spin_Polarization: 1.0\n",
                extra=f"Spin_Polarization: .true.\nSpin_Symmetry_Sectors: {sectors}\n",
            )
            reference, _, accelerated, _ = self.inputs.both(problem)
            self.assertTrue(reference.converged and accelerated.converged)
            self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=1e-9, msg=sectors)

    def test_spin_b3lyp_atom(self) -> None:
        # A fully polarized atom: LYP of the empty channel is where libxc's formula is ill-conditioned.
        from parsec_python.acceleration.tests.test_hybrid_device import TIGHT

        problem = self.problem(
            "LibXC_Functional: hyb_gga_xc_b3lyp\n",
            states=2,
            atoms="  0.0 0.0 0.0\n",
            species="Initial_Spin_Polarization: 1.0\n",
            extra="Spin_Polarization: .true.\nSpin_Symmetry_Sectors: off\n" + TIGHT,
        )
        reference, _, accelerated, _ = self.inputs.both(problem, symmetry="off")
        self.assertTrue(reference.converged and accelerated.converged)
        self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=1e-10)

    def test_meta_gga_hybrid_on_the_device_fock_loop(self) -> None:
        # TPSSh: the device Fock loop with meta-GGA inner loops (the orbital term and the exchange
        # in the device Hamiltonians, libxc's meta-GGA on CuPy arrays) = the reference path.
        from parsec_python.acceleration.tests.test_hybrid_device import TIGHT

        for symmetry, tolerance, extra, species, states, atoms in (
            ("off", 1e-9, "", "", 3, None),
            ("auto", 1e-8, "", "", 3, None),
            ("auto", 1e-8, "Spin_Polarization: .true.\n", "Initial_Spin_Polarization: 1.0\n", 2, "  0.0 0.0 0.0\n"),
        ):
            problem = self.problem(
                "LibXC_Functional: hyb_mgga_xc_tpssh\n",
                states=states,
                atoms=atoms,
                species=species,
                extra=extra + TIGHT,
            )
            reference, _, accelerated, _ = self.inputs.both(problem, symmetry=symmetry)
            label = (symmetry, bool(extra))
            self.assertTrue(reference.converged and accelerated.converged, label)
            self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=tolerance, msg=label)
            self.assertAlmostEqual(
                accelerated.exact_exchange.exchange_energy,
                reference.exact_exchange.exchange_energy,
                delta=1e-6,
                msg=label,
            )

    def test_meta_gga_hybrid_in_the_orbital_form_on_the_device(self) -> None:
        # r2SCAN0 with PARSEC_META_GGA_GRADIENT=orbital: W of the orbital form and the exchange
        # in the device Hamiltonians (full grid and sectors) = the reference path.
        from unittest import mock

        from parsec_python.acceleration.tests.test_hybrid_device import TIGHT

        problem = self.problem("LibXC_Functional: hyb_mgga_xc_r2scan0\n", extra=TIGHT)
        with mock.patch.dict("os.environ", {"PARSEC_META_GGA_GRADIENT": "orbital"}):
            for symmetry, tolerance in (("off", 1e-9), ("auto", 1e-8)):
                reference, _, accelerated, _ = self.inputs.both(problem, symmetry=symmetry)
                self.assertTrue(reference.converged and accelerated.converged, symmetry)
                self.assertEqual(accelerated.meta_gga.gradient, "orbital")
                self.assertAlmostEqual(
                    accelerated.energies.total, reference.energies.total, delta=tolerance, msg=symmetry
                )
                self.assertAlmostEqual(
                    accelerated.exact_exchange.exchange_energy,
                    reference.exact_exchange.exchange_energy,
                    delta=1e-6,
                    msg=symmetry,
                )

    def test_libxc_b3lyp_on_the_device_fock_loop(self) -> None:
        from parsec_python.acceleration.tests.test_hybrid_device import TIGHT

        problem = self.problem("LibXC_Functional: hyb_gga_xc_b3lyp\n", extra=TIGHT)
        reference, system, accelerated, _ = self.inputs.both(problem, symmetry="off")
        self.assertTrue(reference.converged and accelerated.converged)
        self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=1e-11)
        self.assertAlmostEqual(
            accelerated.exact_exchange.exchange_energy, reference.exact_exchange.exchange_energy, delta=1e-10
        )
        # The code's own B3LYP on the device: the same functional.
        own = self.inputs.both(self.inputs.problem("b3lyp"), symmetry="off")[2]
        self.assertAlmostEqual(accelerated.energies.total, own.energies.total, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
