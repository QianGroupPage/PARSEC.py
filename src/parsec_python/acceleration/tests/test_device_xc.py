"""Exchange-correlation functionals on the device (``acceleration/V_xc/device_xc.py``).

The device evaluators translate the readable functionals of
:mod:`parsec_python.V_xc` line by line, with the same stencils and the same
discrete variational GGA potential, so they are compared with them to
round-off: field by field on small grids (full grid, symmetry orbits of
several reflection groups, spin channels, a periodic cell), and in complete
SCF calculations of the accelerated driver against the same driver with
``PARSEC_CUPY_XC=off`` (the host functionals) and against the reference
solver.  The CuPy class of ``test_xc_functionals.py`` compares the device
path, the default of the functionals added for Fortran parity, with Fortran
PARSEC.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import prepare_single_point as prepare_reference
from parsec_python.SCF import run_scf as run_reference
from parsec_python.V_xc.pbe_spin import pbe_spin
from parsec_python.V_xc.selection import evaluate_spin_xc, evaluate_unpolarized_xc
from parsec_python.acceleration import driver
from parsec_python.acceleration.V_xc import device_xc
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, GridSettings
from parsec_python.tests.spin_reference import load_cases as load_spin_cases
from parsec_python.tests.spin_reference import write_case
from parsec_python.tests.xc_reference import load_xc_cases, result_levels


UNPOLARIZED = ("pw92", "xalpha", "wigner", "hedin_lundqvist", "lb94", "casida_salahub", "pbe", "pw91")
SPIN = ("pw92", "lb94", "casida_salahub", "pbe", "pw91")
#: Both sides evaluate the same float64 operations; only the elementary
#: functions and the order of the grid sums differ.
FIELD_TOLERANCE = 1.0e-13


def _gaussians(coordinates, centers, width, weights):
    values = np.zeros(len(coordinates))
    for center, weight in zip(centers, weights):
        difference = coordinates - np.asarray(center)
        values += weight * np.exp(-np.sum(difference * difference, axis=1) / width)
    return values


def _relative(reference, value) -> float:
    reference = np.asarray(getattr(reference, "values", reference))
    value = np.asarray(getattr(value, "values", value))
    return float(np.max(np.abs(reference - value)) / max(np.max(np.abs(reference)), 1.0e-300))


class SelectionTests(unittest.TestCase):
    def test_mode(self) -> None:
        for raw, mode in (("", "auto"), ("auto", "auto"), ("on", "on"), ("1", "on"), ("off", "off"), ("0", "off")):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": raw}):
                self.assertEqual(device_xc.device_xc_mode(), mode)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": "sometimes"}):
            with self.assertRaises(ValueError):
                device_xc.device_xc_mode()

    def test_functionals_on_the_device(self) -> None:
        selected = device_xc.device_xc_selected
        for functional in ("pw92", "xalpha", "wigner", "hedin_lundqvist", "pw91", "lb94", "casida_salahub"):
            self.assertTrue(selected(functional, mode="auto"), functional)
        # PBE takes the device by default since 2026-10-30 (it kept the host
        # unless asked before, which this test asserted); CA keeps its own
        # evaluator always.
        self.assertTrue(selected("pbe", mode="auto"))
        self.assertTrue(selected("pbe", spin_polarized=True, mode="auto"))
        self.assertTrue(selected("pbe", mode="on"))
        self.assertTrue(selected("pbe", spin_polarized=True, mode="on"))
        self.assertFalse(selected("pbe", mode="off"))
        self.assertFalse(selected("pbe", spin_polarized=True, mode="off"))
        for mode in ("auto", "on", "off"):
            self.assertFalse(selected("ca", mode=mode))
            self.assertFalse(selected("ca", spin_polarized=True, mode=mode))
        for functional in UNPOLARIZED:
            self.assertFalse(selected(functional, mode="off"))
        # No spin form of these (as in Fortran).
        for functional in ("xalpha", "wigner", "hedin_lundqvist"):
            self.assertFalse(selected(functional, spin_polarized=True, mode="on"))

    def test_the_periodic_spin_drivers_keep_pbe_on_the_host_unless_asked(self) -> None:
        # Decoupled on 2026-10-30, when PBE took the device by default in the accelerated driver:
        # the device kernel is pbe_spin, the k-point and Gamma spin drivers evaluate pbe_spin_polarized.
        from parsec_python.acceleration.V_xc import periodic_xc

        selected = periodic_xc.periodic_spin_device_selected
        self.assertEqual(periodic_xc.HOST_UNLESS_ASKED, frozenset({"pbe"}))
        self.assertEqual([selected("pbe", mode=mode) for mode in ("auto", "on", "off")], [False, True, False])
        for functional in ("pw92", "pw91", "lb94", "casida_salahub"):
            self.assertEqual([selected(functional, mode=mode) for mode in ("auto", "on", "off")], [True, True, False])
        for mode in ("auto", "on", "off"):
            self.assertFalse(selected("ca", mode=mode))
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": ""}):
            self.assertFalse(selected("pbe"))
            self.assertTrue(device_xc.device_xc_selected("pbe", spin_polarized=True))

    def test_other_backends_and_grids_keep_the_host(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=3.0))
        reference = SimpleNamespace(
            input=SimpleNamespace(scf=SimpleNamespace(xc_functional="pw92", spin_polarized=False)),
            grid=grid,
            core_density=None,
        )
        for backend in ("native", "scipy"):
            self.assertEqual(
                device_xc.accelerated_xc_evaluators(reference, backend=backend), (None, None)
            )
        self.assertTrue(device_xc.device_grid_supported(grid))
        for flag in ("is_general_periodic", "is_confined_periodic"):
            self.assertFalse(device_xc.device_grid_supported(SimpleNamespace(**{flag: True})))

    def test_constants_are_written_into_the_kernels(self) -> None:
        from parsec_python.V_xc.pbe_spin import _GAMMA

        source = device_xc.kernel_source()
        self.assertIn(f"#define PBE_GAMMA ({_GAMMA!r})", source)
        self.assertIn("#define PW_GAM (0.5198421)", source)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(GridSettings(spacing=0.45, radius=5.0, expansion_order=12))
        x = cls.grid.coordinates
        cls.atoms = (Atom("O", (0.0, 0.0, 0.3)), Atom("H", (1.4, 0.0, -0.8)), Atom("H", (-1.4, 0.0, -0.8)))
        cls.density = _gaussians(x, [a.position for a in cls.atoms], 1.0, (1.5, 0.4, 0.4))
        cls.core = _gaussians(x, [(0.0, 0.0, 0.3)], 0.3, (0.2,))
        cls.up = 0.6 * cls.density
        cls.down = 0.4 * cls.density + _gaussians(x, [(1.4, 0.0, -0.8)], 0.5, (0.05,))

    def assert_same(self, reference, value, label) -> None:
        for field in ("potential", "potential_up", "potential_down", "energy_density", "energy_per_electron"):
            if hasattr(reference, field):
                self.assertLess(
                    _relative(getattr(reference, field), getattr(value, field)), FIELD_TOLERANCE, f"{label} {field}"
                )
        self.assertAlmostEqual(
            value.total_energy, reference.total_energy, delta=FIELD_TOLERANCE * abs(reference.total_energy), msg=label
        )

    def test_unpolarized_functionals_on_the_full_grid(self) -> None:
        for functional in UNPOLARIZED:
            reference = evaluate_unpolarized_xc(
                functional, self.density, self.grid, self.core, ionization_energy_difference=0.3
            )
            evaluator = device_xc.DeviceXCEvaluator(
                functional, self.grid, core_density=self.core, ionization_energy_difference=0.3
            )
            self.assert_same(reference, evaluator(self.density), functional)

    def test_symmetry_orbits_of_several_groups(self) -> None:
        """The orbit values, also on mirror planes and under a group without its mirrors."""

        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry.axis_reflection import AxisReflectionReduction

        structures = {
            # C2 about z only: a component of the flux is not zero on the x = 0 plane.
            "C2": ((Atom("O", (0.0, 0.0, 0.3)), Atom("H", (1.2, 0.5, 0.4)), Atom("H", (-1.2, -0.5, 0.4))), 2),
            "D2h": (
                (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (1.3, 0.0, 0.0)), Atom("H", (-1.3, 0.0, 0.0)),
                 Atom("H", (0.0, 0.9, 0.0)), Atom("H", (0.0, -0.9, 0.0))),
                8,
            ),
        }
        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            for order in (4, 12):
                grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=order, shift=shift))
                for name, (atoms, group_order) in structures.items():
                    reduction = AxisReflectionReduction.detect(grid, atoms)
                    self.assertEqual(reduction.group_order, group_order)
                    reducer = SymmetrySCFReducer(reduction)
                    full = _gaussians(grid.coordinates, [a.position for a in atoms], 0.9, [1.0] + [0.4] * (len(atoms) - 1))
                    wedge = reducer.from_full(full)
                    for functional in ("pbe", "pw91", "lb94", "pw92"):
                        label = f"{functional} {name} shift {shift} order {order}"
                        reference = evaluate_unpolarized_xc(functional, reducer.to_full(wedge), grid, None)
                        value = device_xc.DeviceXCEvaluator(functional, grid, reducer=reducer)(wedge)
                        self.assertLess(
                            _relative(reducer.wedge_values(reference.potential), value.potential), FIELD_TOLERANCE, label
                        )
                        self.assertAlmostEqual(
                            value.total_energy, reference.total_energy,
                            delta=FIELD_TOLERANCE * abs(reference.total_energy), msg=label,
                        )
                    # A full-grid density is orbit-averaged first, as the native CA
                    # evaluator takes it (the average of equal values rounds).
                    evaluator = device_xc.DeviceXCEvaluator("pbe", grid, reducer=reducer)
                    self.assertLess(
                        _relative(evaluator(wedge).potential, evaluator(reducer.to_full(wedge)).potential),
                        FIELD_TOLERANCE,
                    )

    def test_spin_functionals(self) -> None:
        for functional in SPIN:
            if functional == "pbe":
                reference = pbe_spin(self.up, self.down, self.grid, self.core)
            else:
                reference = evaluate_spin_xc(
                    functional, self.up, self.down, self.grid, self.core, ionization_energy_difference=0.3
                )
            evaluator = device_xc.DeviceSpinXCEvaluator(
                functional, self.grid, core_density=self.core, ionization_energy_difference=0.3
            )
            self.assert_same(reference, evaluator(self.up, self.down), f"spin {functional}")

    def test_periodic_cell(self) -> None:
        from parsec_python.Grid.pbc import build_periodic_grid
        from parsec_python.models import PeriodicCell, PeriodicGridSettings

        lengths = np.array([6.0, 7.0, 8.0])
        grid = build_periodic_grid(PeriodicCell(np.diag(lengths)), PeriodicGridSettings(spacing=0.5, expansion_order=4))
        self.assertTrue(device_xc.device_grid_supported(grid))
        density = np.full(grid.size, 1.0e-3)
        for center in ((1.0, 1.0, 1.0), (5.5, 6.5, 7.6)):
            difference = grid.coordinates - np.asarray(center)
            difference -= np.round(difference / lengths) * lengths
            density += np.exp(-np.sum(difference * difference, axis=1) / 1.3)
        for functional in ("pw92", "lb94", "pbe", "pw91"):
            reference = evaluate_unpolarized_xc(functional, density, grid, None)
            self.assert_same(reference, device_xc.DeviceXCEvaluator(functional, grid)(density), f"periodic {functional}")
        for functional in ("pbe", "pw91"):
            reference = (
                pbe_spin(0.7 * density, 0.3 * density, grid)
                if functional == "pbe"
                else evaluate_spin_xc(functional, 0.7 * density, 0.3 * density, grid)
            )
            value = device_xc.DeviceSpinXCEvaluator(functional, grid)(0.7 * density, 0.3 * density)
            self.assert_same(reference, value, f"periodic spin {functional}")

    def test_core_density_of_the_present_structure(self) -> None:
        present = [self.core]
        evaluator = device_xc.DeviceXCEvaluator("pw91", self.grid, core_density=lambda: present[0])
        evaluator(self.density)
        present[0] = 2.0 * self.core
        reference = evaluate_unpolarized_xc("pw91", self.density, self.grid, present[0])
        self.assert_same(reference, evaluator(self.density), "moved core")
        present[0] = None
        reference = evaluate_unpolarized_xc("pw91", self.density, self.grid, None)
        self.assert_same(reference, evaluator(self.density), "no core")

    def test_planned_bytes_are_what_the_evaluator_takes(self) -> None:
        """``planned_device_bytes`` (what the sector memory rules count) against a pool of its own."""

        import cupy as cp

        from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
        from parsec_python.acceleration.Symmetry.axis_reflection import AxisReflectionReduction

        grid = build_cluster_grid(GridSettings(spacing=0.3, radius=7.0, expansion_order=12))
        atoms = (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (1.3, 0.0, 0.0)), Atom("H", (-1.3, 0.0, 0.0)),
                 Atom("H", (0.0, 0.9, 0.0)), Atom("H", (0.0, -0.9, 0.0)))
        reducer = SymmetrySCFReducer(AxisReflectionReduction.detect(grid, atoms))
        full = _gaussians(grid.coordinates, [a.position for a in atoms], 0.9, [1.0] + [0.4] * 4) + 1.0e-6
        core = _gaussians(grid.coordinates, [(0.0, 0.0, 0.0)], 0.3, (0.2,))
        wedge = reducer.from_full(full)
        for spin, functional in ((False, "pbe"), (False, "lb94"), (True, "pbe"), (True, "pw92")):
            for with_core in (False, True):
                pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(pool.malloc)
                try:
                    keywords = {"core_density": core if with_core else None}
                    if spin:
                        evaluator = device_xc.DeviceSpinXCEvaluator(functional, grid, **keywords)
                        for _ in range(2):
                            evaluator(0.6 * full, 0.4 * full)
                    else:
                        evaluator = device_xc.DeviceXCEvaluator(functional, grid, reducer=reducer, **keywords)
                        for _ in range(2):
                            evaluator(wedge)
                    cp.cuda.Device().synchronize()
                    planned, held = evaluator.planned_device_bytes, pool.total_bytes()
                finally:
                    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
                    del evaluator
                    pool.free_all_blocks()
                label = f"spin={spin} {functional} core={with_core}"
                # The pool holds the planned bytes and the few small blocks of
                # the reductions (cp.min, the energy sum): 60 KB on 6,637 orbits.
                self.assertGreaterEqual(held, planned, label)
                self.assertLessEqual(held, planned + 2**17, label)

    def test_refusals(self) -> None:
        negative = self.density.copy()
        negative[0] = -1.0e-12
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            device_xc.DeviceXCEvaluator("pbe", self.grid)(negative)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            device_xc.DeviceSpinXCEvaluator("pw92", self.grid)(negative, self.down)
        with self.assertRaises(ValueError):
            device_xc.DeviceXCEvaluator("ca", self.grid)
        with self.assertRaises(ValueError):
            device_xc.DeviceSpinXCEvaluator("xalpha", self.grid)


def _xc_case(name: str, **replacements) -> dict:
    """A case of ``xc_parsec_reference.json`` with input lines replaced (by keyword)."""

    case = dict(load_xc_cases()[name])
    replacements.setdefault("Convergence_Criterion", "1.0e-10")
    lines = []
    for line in case["input"]:
        key = line.split(":")[0].strip()
        lines.append(f"{key}: {replacements[key]}" if key in replacements else line)
    case["input"] = lines
    return case


def _write(case: dict, directory: Path) -> Path:
    path = write_case(case, directory)
    functional = next(
        (line.split(":")[1].strip().lower() for line in case["input"] if line.startswith("Correlation_Type")), "ca"
    )
    if functional in {"pb", "pbe"}:
        # The CA files of the fixtures carry a PBE label here: the code under
        # test is the functional, not the pseudopotential.
        for file in Path(directory).glob("*_POTRE.DAT"):
            lines = file.read_text().splitlines(keepends=True)
            lines[0] = lines[0].replace(" ca ", " pb ")
            file.write_text("".join(lines))
    return path


def _solve(case: dict, mode: str | None, *, forces: bool = False):
    """``mode`` None runs the reference solver, else CuPy with ``PARSEC_CUPY_XC=mode``."""

    with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = parse_parsec_input(_write(case, Path(directory))).problem
        if mode is None:
            system = prepare_reference(problem)
            result = run_reference(system)
            label = None
            computed = None
            if forces:
                from parsec_python.Forces import compute_forces

                computed = compute_forces(system, result).total
            return result, label, computed
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": mode}):
            system = driver.prepare_single_point(problem, backend="cupy")
            result = driver.run_scf(system)
            computed = driver.compute_forces(system, result).total if forces else None
    return result, dict(result.backend.details)["xc_implementation"], computed


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class DeviceSCFTests(unittest.TestCase):
    def assert_same_run(self, left, right, *, energy=1.0e-9, level=1.0e-8) -> None:
        self.assertTrue(left.converged and right.converged)
        for field in ("total", "exchange_correlation", "integral_vxc_rho", "hartree", "eigenvalue"):
            self.assertAlmostEqual(
                getattr(left.energies, field), getattr(right.energies, field), delta=energy, msg=field
            )
        left_levels, right_levels = result_levels(left), result_levels(right)
        for channel, values in left_levels.items():
            count = min(len(values), len(right_levels[channel]))
            np.testing.assert_allclose(values[:count], right_levels[channel][:count], rtol=0, atol=level)

    def test_new_functionals_take_the_device(self) -> None:
        """PW92 and Casida-Salahub on the symmetry orbits of water: as the host and the reference solver."""

        for name, label in (("h2o_pl", "PW92-LDA"), ("h2o_cs3", "Casida-Salahub")):
            case = _xc_case(name)
            device, device_label, _ = _solve(case, "auto")
            self.assertEqual(device_label, f"CuPy float64 {label} on the symmetry orbits")
            host, host_label, _ = _solve(case, "off")
            self.assertEqual(host_label, f"vectorized NumPy {label}")
            self.assert_same_run(device, host)
            reference, _, _ = _solve(case, None)
            self.assert_same_run(device, reference, energy=1.0e-7, level=1.0e-6)

    def test_pbe_takes_the_device_unless_off(self) -> None:
        # Up to 2026-10-30 PBE kept the host under ``auto`` (this test asserted it);
        # ``off`` is the way back.
        case = _xc_case("h2o_pb")
        device, device_label, _ = _solve(case, "auto")
        self.assertEqual(device_label, "CuPy float64 discrete-variational PBE on the symmetry orbits")
        host, host_label, _ = _solve(case, "off")
        self.assertEqual(host_label, "vectorized NumPy discrete-variational PBE")
        self.assert_same_run(device, host)
        _, on_label, _ = _solve(case, "on")
        self.assertEqual(on_label, device_label)

    def test_the_sector_memory_rules_are_told_the_evaluator_bytes(self) -> None:
        """Where the device evaluator serves the sectors, their eigensolver counts its bytes on its device."""

        for name, mode, expect in (("h2o_pb", "auto", True), ("h2o_pb", "off", False), ("h2o_ca", "auto", False)):
            with self.subTest(case=name, mode=mode), tempfile.TemporaryDirectory() as directory, \
                    warnings.catch_warnings(), mock.patch.dict(os.environ, {"PARSEC_CUPY_XC": mode}):
                warnings.simplefilter("ignore")
                problem = parse_parsec_input(_write(_xc_case(name), Path(directory))).problem
                system = driver.prepare_single_point(problem, backend="cupy")
                solver = system.backend.symmetry_eigensolver
                self.assertIsNotNone(solver)
                resident = solver.resident_device_bytes
                if not expect:
                    self.assertEqual(resident, {})
                    continue
                self.assertEqual(len(resident), 1)
                (device, count), = resident.items()
                self.assertGreater(count, 0)
                operators = [operator for operator in solver._operators if operator is not None]
                self.assertTrue(operators)
                for operator in operators:
                    self.assertEqual(operator.resident_device_bytes, resident)

    def test_ca_keeps_its_evaluator(self) -> None:
        _, label, _ = _solve(_xc_case("h2o_ca"), "on")
        self.assertEqual(label, "cached C++/OpenMP float64 CA/PZ-LDA")

    def test_spin_polarized_functionals(self) -> None:
        for name, mode, label in (
            ("ch3_default", "auto", "CuPy float64 spin-polarized PW92-LDA"),
            # ``on`` up to 2026-10-30; the default since.
            ("h2o_spin_pb", "auto", "CuPy float64 spin-polarized discrete-variational PBE"),
        ):
            case = _xc_case(name)
            device, device_label, _ = _solve(case, mode)
            self.assertEqual(device_label, label)
            host, _, _ = _solve(case, "off")
            self.assert_same_run(device, host)

    def test_core_corrected_forces(self) -> None:
        """GeH3 with a core-corrected Ge, unpolarized PW91 and spin PW92, with forces."""

        spin = dict(load_spin_cases()["geh3_nlcc_ca"])
        for correlation, polarized in (("pw", ".false."), ("pl", ".true.")):
            case = dict(spin)
            case["input"] = [
                f"Correlation_Type: {correlation}" if line.startswith("Correlation_Type")
                else f"Spin_Polarization: {polarized}" if line.startswith("Spin_Polarization")
                else "Convergence_Criterion: 1.0e-10" if line.startswith("Convergence_Criterion")
                else line
                for line in spin["input"]
            ]
            device, label, device_forces = _solve(case, "auto", forces=True)
            self.assertTrue(label.startswith("CuPy float64"), label)
            host, _, host_forces = _solve(case, "off", forces=True)
            self.assert_same_run(device, host)
            np.testing.assert_allclose(device_forces, host_forces, rtol=0, atol=1.0e-8)

    def test_periodic_cell(self) -> None:
        """The orthorhombic cell of the driver tests: as the periodic reference solver."""

        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.acceleration.tests.test_driver_cli import periodic_hydrogen

        base = periodic_hydrogen()
        for functional in ("pw92", "pw91", "lb94"):
            problem = replace(base, scf=replace(base.scf, xc_functional=functional))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reference = run_reference(prepare_periodic_single_point(problem))
                result = driver.run_scf(driver.prepare_single_point(problem, backend="cupy"))
            self.assertTrue(dict(result.backend.details)["xc_implementation"].startswith("CuPy float64"))
            self.assertEqual(result.iterations, reference.iterations)
            self.assertAlmostEqual(result.energies.total, reference.energies.total, delta=1.0e-10)
            np.testing.assert_allclose(result.eigenvalues, reference.eigenvalues, rtol=0, atol=1.0e-10)

    def test_moved_atoms_read_the_new_core_density(self) -> None:
        spin = load_spin_cases()["geh3_nlcc_ca"]
        case = dict(spin)
        case["input"] = [
            "Correlation_Type: pw" if line.startswith("Correlation_Type")
            else "Spin_Polarization: .false." if line.startswith("Spin_Polarization")
            else "Convergence_Criterion: 1.0e-10" if line.startswith("Convergence_Criterion")
            else line
            for line in spin["input"]
        ]
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(_write(case, Path(directory))).problem
            system = driver.prepare_single_point(problem, backend="cupy", symmetry="off")
            driver.run_scf(system)
            atoms = tuple(
                Atom(atom.symbol, np.asarray(atom.position) + (np.array([0.15, -0.05, 0.1]) if index == 0 else 0.0))
                for index, atom in enumerate(system.atoms)
            )
            moved = driver.run_scf(driver.update_geometry(system, atoms))
            fresh_problem = replace(problem, atoms=atoms)
            fresh = driver.run_scf(driver.prepare_single_point(fresh_problem, backend="cupy", symmetry="off"))
        self.assertAlmostEqual(moved.energies.total, fresh.energies.total, delta=1.0e-8)
        self.assertAlmostEqual(
            moved.energies.exchange_correlation, fresh.energies.exchange_correlation, delta=1.0e-8
        )


if __name__ == "__main__":
    unittest.main()
