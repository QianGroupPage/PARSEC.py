"""The local forces of large cells: one atom per orbit of the group of the sectors, and the streamed series.

A cell solved in the sectors of a point group averages its density and exchange-correlation potential over the full
group ``G`` of the sectors, so its local ionic and core-correction forces are equivariant: the forces of one atom per
orbit of ``G`` turned onto the others (``PARSEC_PERIODIC_SECTOR_LOCAL_FORCES=orbits``) must equal the forces of every
atom (``all``) on such a density.  The reciprocal series of the local forces is streamed over slabs from the real
transform of the density (``PARSEC_PERIODIC_FORCE_SERIES=stream``) and must equal the stack of its three components
(``stack``, the sums to 2026-11-18) to round-off.
"""

from __future__ import annotations

from dataclasses import replace
import os
import types
import unittest
from unittest import mock
import warnings
from pathlib import Path

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.acceleration.Forces import periodic_local
from parsec_python.acceleration.Forces.periodic_local import (
    fast_periodic_local_forces,
    orbit_periodic_local_forces,
    periodic_force_series_setting,
    periodic_sector_local_forces_setting,
)
from parsec_python.acceleration.V_ion.periodic_setup import fast_splitting
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.Grid import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_GE = _DATA / "Ge_POTRE.DAT"
_CELLS = {
    "rectangular": np.diag([7.7, 8.4, 9.1]),
    "fcc": 6.6 * (np.ones((3, 3)) - np.eye(3)),
    "monoclinic": np.array([[8.4, 0.0, 0.0], [7.7 * np.cos(1.4), 7.7 * np.sin(1.4), 0.0], [0.0, 0.0, 9.1]]),
}
_SHIFTS = {"origin": (0.0, 0.0, 0.0), "half": (0.5, 0.5, 0.5), "odd": (0.5, 0.3, 0.8)}


def _fields(grid, lattice):
    """Smooth, arbitrary density and XC fields without any symmetry."""
    phase = 2.0 * np.pi * (grid.coordinates @ np.linalg.inv(lattice))
    density = 0.3 + 0.2 * np.cos(phase[:, 0] + 0.4) * np.sin(phase[:, 1]) + 0.1 * np.sin(phase[:, 2] - phase[:, 0])
    xc = -0.5 + 0.3 * np.sin(phase[:, 0]) * np.cos(phase[:, 2] + 0.3) + 0.2 * np.cos(phase[:, 1] - 2 * phase[:, 2])
    return density, xc


def _series(xp, grid, lattice, positions, charges, density, setting: str) -> np.ndarray:
    eta, _, g_cutoff = fast_splitting(grid, lattice, 1.0e-14)
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_FORCE_SERIES": setting}):
        return periodic_local._reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice)


def _assert_same_series(test, value, reference, label) -> None:
    scale = float(np.max(np.abs(reference)))
    test.assertGreater(scale, 1.0e-3, label)
    test.assertLessEqual(float(np.max(np.abs(value - reference))), 1.0e-12 * scale, label)


class StreamedSeriesTests(unittest.TestCase):
    """The streamed series against the stack of its components, on oblique cells, shifted and odd grids."""

    def test_the_streamed_series_equals_the_stack(self) -> None:
        rng = np.random.default_rng(7)
        for cell, lattice in _CELLS.items():
            for shift in _SHIFTS:
                grid = build_periodic_grid(
                    PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6, shift=_SHIFTS[shift])
                )
                density, _ = _fields(grid, lattice)
                positions = rng.random((37, 3)) @ lattice
                charges = rng.choice([1.0, 4.0, 6.0], size=37)
                label = f"{cell} {shift} {tuple(grid.points_per_axis)}"
                stacked = _series(np, grid, lattice, positions, charges, density, "stack")
                with self.subTest(label):
                    _assert_same_series(self, _series(np, grid, lattice, positions, charges, density, "stream"), stacked, label)
                    _assert_same_series(self, _series(np, grid, lattice, positions, charges, density, "host"), stacked, label)

    def test_slabs_and_blocks_of_any_size_give_the_same_sums(self) -> None:
        lattice = _CELLS["monoclinic"]
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.35, expansion_order=6, shift=_SHIFTS["odd"]))
        density, _ = _fields(grid, lattice)
        positions = np.random.default_rng(3).random((29, 3)) @ lattice
        charges = np.full(29, 4.0)
        whole = _series(np, grid, lattice, positions, charges, density, "stream")
        for elements, atoms in ((1, 1), (5_000, 3), (20_000, 512)):
            with self.subTest(elements=elements, atoms=atoms), mock.patch.object(
                periodic_local, "_SLAB_ELEMENTS", elements
            ), mock.patch.object(periodic_local, "_STREAM_ATOMS", atoms):
                _assert_same_series(self, _series(np, grid, lattice, positions, charges, density, "stream"), whole, "slabs")

    def test_a_module_without_the_real_transform_takes_the_inverse_one(self) -> None:
        # the restricted module of test_periodic_forces: the array functions of CuPy that the sums use, ifftn alone
        allowed = {"asarray", "zeros_like", "searchsorted", "clip", "interp", "where", "exp", "sum", "stack", "matmul", "einsum"}

        class Restricted:
            fft = types.SimpleNamespace(ifftn=np.fft.ifftn)

            def __getattr__(self, name):
                if name not in allowed:
                    raise AttributeError(name)
                return getattr(np, name)

        lattice = _CELLS["fcc"]
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6, shift=_SHIFTS["odd"]))
        density, _ = _fields(grid, lattice)
        positions = np.random.default_rng(5).random((11, 3)) @ lattice
        charges = np.full(11, 4.0)
        stacked = _series(np, grid, lattice, positions, charges, density, "stack")
        _assert_same_series(self, _series(Restricted(), grid, lattice, positions, charges, density, "stream"), stacked, "ifftn")

    def test_the_stack_is_the_former_routine(self) -> None:
        # "stack" runs the routine of before whatever the module: the same numbers as the stack on the host
        lattice = _CELLS["rectangular"]
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6))
        density, _ = _fields(grid, lattice)
        positions = np.random.default_rng(1).random((5, 3)) @ lattice
        charges = np.full(5, 1.0)
        eta, _, g_cutoff = fast_splitting(grid, lattice, 1.0e-14)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_FORCE_SERIES": "stack"}):
            np.testing.assert_array_equal(
                periodic_local._reciprocal_forces(np, grid, positions, charges, density, eta, g_cutoff, lattice),
                periodic_local._stacked_reciprocal_forces(np, grid, positions, charges, density, eta, g_cutoff, lattice),
            )

    def test_settings(self) -> None:
        for name, reader, default, values in (
            ("PARSEC_PERIODIC_FORCE_SERIES", periodic_force_series_setting, "stream", ("stream", "stack", "host")),
            ("PARSEC_PERIODIC_SECTOR_LOCAL_FORCES", periodic_sector_local_forces_setting, "orbits", ("orbits", "all")),
        ):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(name, None)
                self.assertEqual(reader(), default)
            for value in values:
                with mock.patch.dict(os.environ, {name: f" {value.upper()} "}):
                    self.assertEqual(reader(), value)
            with mock.patch.dict(os.environ, {name: "on"}), self.assertRaises(ValueError):
                reader()


def _silicon_cell(repeats=(1, 1, 1), displaced: bool = True):
    """The displaced Si8 of :func:`..test_periodic_point_group.displaced_silicon` (Td about the origin, forces on the
    atoms), repeated ``repeats`` times: a 2 x 2 x 2 repeat adds the translations by half the cell to its group."""

    from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon, silicon

    base = displaced_silicon() if displaced else silicon("pbe")
    lattice = np.asarray(base.periodic_cell.lattice_vectors, dtype=float)
    atoms = []
    for i in range(repeats[0]):
        for j in range(repeats[1]):
            for k in range(repeats[2]):
                shift = np.array([i, j, k]) @ lattice
                atoms += [Atom(atom.symbol, tuple(np.asarray(atom.position) + shift)) for atom in base.atoms]
    return replace(
        base,
        atoms=tuple(atoms),
        periodic_cell=replace(base.periodic_cell, lattice_vectors=np.diag(np.diag(lattice) * np.asarray(repeats))),
    )


def invariant_cell(problem):
    """``(grid, atoms, potentials, specifications, lattice, density, xc, G)``: the cell of ``problem`` with its
    superposition of atomic densities averaged over the full group ``G`` of its point-group reduction (as the SCF in
    sectors averages its density) and a G-invariant exchange-correlation field."""

    from parsec_python.SCF import pbc as periodic
    from parsec_python.acceleration.Forces.sector_symmetry import sector_group_operations
    from parsec_python.acceleration.Symmetry.axis_reflection import PointGroupReduction, detect_point_group_reduction
    from parsec_python.acceleration.Symmetry.geometry_cache import load_or_detect_reflection_reduction

    lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        potentials = periodic.load_pseudopotentials(problem.pseudopotentials, xc_functional=problem.scf.xc_functional)
    grid = periodic.build_periodic_grid(problem.periodic_cell, problem.grid)
    atoms = tuple(problem.atoms)
    abelian, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
    reduction = detect_point_group_reduction(grid, atoms, abelian)
    if not isinstance(reduction, PointGroupReduction):
        raise AssertionError("the cell has no point-group reduction")
    operations = sector_group_operations(reduction, atoms, reduction.grid if reduction.grid is not None else grid, lattice)
    density = periodic.superpose_atomic_density(
        grid, atoms, potentials, problem.pseudopotentials, lattice, periodic_dimensions=3
    )

    def invariant(values):
        orbit = np.bincount(reduction.full_to_wedge, weights=values, minlength=reduction.wedge_size)
        orbit = orbit / reduction.multiplicities
        return reduction.symmetrize_values(orbit)[reduction.full_to_wedge]

    density = invariant(density)
    xc = invariant(-np.cbrt(np.abs(density)) - 0.1 * density)
    return grid, atoms, potentials, problem.pseudopotentials, lattice, density, xc, operations


def _with_a_core(arguments):
    """The arguments of a cell of silicon atoms with the germanium potential, which has a partial core."""

    grid, atoms, _, _, lattice, density, xc, operations = arguments
    potential = read_parsec_pseudopotential(_GE)
    return grid, atoms, {"Si": potential}, {"Si": SpeciesPotential(_GE, 1, use_spline=True)}, lattice, density, xc, operations


class OrbitLocalForceTests(unittest.TestCase):
    """The local forces of one atom per orbit of ``G`` turned onto the others equal those of every atom."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cells = {
            "Si8": invariant_cell(_silicon_cell()),
            "Si 2x2x2": invariant_cell(_silicon_cell((2, 2, 2))),
        }

    def _compare(self, arguments, label, *, xp=np, erf=None, core: bool = False):
        everything = fast_periodic_local_forces(*arguments[:7], xp=xp, erf=erf)
        orbits, details = orbit_periodic_local_forces(*arguments, xp=xp, erf=erf)
        self.assertGreater(float(np.max(np.abs(everything.ionic))), 1.0e-3, label)
        np.testing.assert_allclose(orbits.ionic, everything.ionic, rtol=0, atol=1.0e-12, err_msg=label)
        np.testing.assert_allclose(orbits.core_correction, everything.core_correction, rtol=0, atol=1.0e-12, err_msg=label)
        np.testing.assert_array_equal(orbits.self_consistency, 0.0)
        if core:
            self.assertGreater(float(np.max(np.abs(everything.core_correction))), 1.0e-4, label)
        self.assertLess(details["site_asymmetry"], 1.0e-12, label)
        self.assertLess(details["representatives"], len(arguments[1]), label)
        self.assertGreater(details["sampled"], 0, label)
        return details

    def test_one_atom_per_orbit_gives_the_forces_of_every_atom(self) -> None:
        orders = {"Si8": 24, "Si 2x2x2": 192}
        for label, arguments in self.cells.items():
            with self.subTest(label):
                self.assertEqual(len(arguments[-1]), orders[label])
                details = self._compare(arguments, label)
                # Td about the origin: the atom at the origin, its four displaced neighbours, the other three; the
                # repeat adds the translations by a cube and keeps those three orbits
                self.assertEqual(details["representatives"], 3, label)

    def test_the_core_correction(self) -> None:
        for label, arguments in self.cells.items():
            with self.subTest(label):
                self._compare(_with_a_core(arguments), label, core=True)

    def test_a_density_without_the_symmetry_is_noticed(self) -> None:
        grid, atoms, potentials, specifications, lattice, density, xc, operations = self.cells["Si8"]
        tilted = density * (1.0 + 0.05 * np.sin(2.0 * np.pi * grid.coordinates[:, 0] / lattice[0, 0]))
        _, details = orbit_periodic_local_forces(grid, atoms, potentials, specifications, lattice, tilted, xc, operations)
        self.assertGreater(details["site_asymmetry"], 1.0e-6)

    def test_operations_of_other_atoms_are_refused(self) -> None:
        grid, atoms, potentials, specifications, lattice, density, xc, operations = self.cells["Si8"]
        with self.assertRaises(ValueError):
            orbit_periodic_local_forces(grid, atoms[:-1], potentials, specifications, lattice, density, xc, operations)

    @unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
    def test_on_the_gpu(self) -> None:
        import cupy as cp
        from cupyx.scipy.special import erf

        for label, arguments in self.cells.items():
            for core in (False, True):
                with self.subTest(label, core=core):
                    cell = _with_a_core(arguments) if core else arguments
                    self._compare(cell, label, xp=cp, erf=erf, core=core)
                    host = fast_periodic_local_forces(*cell[:7])
                    device = fast_periodic_local_forces(*cell[:7], xp=cp, erf=erf)
                    np.testing.assert_allclose(device.ionic, host.ionic, rtol=0, atol=1.0e-11)
                    np.testing.assert_allclose(device.core_correction, host.core_correction, rtol=0, atol=1.0e-11)


class GroupImageTableTests(unittest.TestCase):
    """The image table of ``G`` composed from those of ``H`` and of the coset representatives is the direct one."""

    def _reduction(self, problem, translations: str):
        from parsec_python.SCF import pbc as periodic
        from parsec_python.acceleration.Symmetry.axis_reflection import PointGroupReduction, detect_point_group_reduction
        from parsec_python.acceleration.Symmetry.geometry_cache import load_or_detect_reflection_reduction

        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": translations}):
            grid = periodic.build_periodic_grid(problem.periodic_cell, problem.grid)
            atoms = tuple(problem.atoms)
            abelian, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
            reduction = detect_point_group_reduction(grid, atoms, abelian)
        self.assertIsInstance(reduction, PointGroupReduction)
        return reduction, reduction.grid if reduction.grid is not None else grid, atoms

    def test_the_composed_table_is_the_direct_one(self) -> None:
        from parsec_python.acceleration.Symmetry import axis_reflection
        from parsec_python.acceleration.Forces.sector_symmetry import ATOM_TOLERANCE

        # Td about the origin; the perfect cube with the fcc translations (96); the 2 x 2 x 2 repeat with the
        # translations by half the cell (192); the point group alone
        for label, problem, translations, order in (
            ("displaced Si8", _silicon_cell(), "auto", 24),
            ("Si8", _silicon_cell(displaced=False), "auto", 96),
            ("displaced Si 2x2x2", _silicon_cell((2, 2, 2)), "auto", 192),
            ("displaced Si 2x2x2, point group", _silicon_cell((2, 2, 2)), "off", 24),
        ):
            with self.subTest(label):
                reduction, grid, atoms = self._reduction(problem, translations)
                full = np.asarray(reduction.full_operations, dtype=np.float64).reshape(-1, 3, 3)
                self.assertEqual(full.shape[0], order)
                shifts = reduction.full_translations
                direct = axis_reflection.atom_image_table(
                    atoms, full, None if shifts is None else np.asarray(shifts), grid, ATOM_TOLERANCE
                )
                with mock.patch.object(axis_reflection, "atom_image_table", wraps=axis_reflection.atom_image_table) as spy:
                    composed = axis_reflection.group_atom_image_table(atoms, reduction, grid, ATOM_TOLERANCE)
                np.testing.assert_array_equal(composed, direct)
                # the tables of H and of the cosets only
                self.assertEqual(
                    sorted(len(call.args[1]) for call in spy.call_args_list),
                    sorted([len(reduction.operations), len(reduction.coset_operations)]),
                )

    def test_a_cluster(self) -> None:
        # a Td cluster: no lattice, no translations
        from parsec_python.Grid.cluster import build_cluster_grid
        from parsec_python.acceleration.Forces.sector_symmetry import ATOM_TOLERANCE, sector_group_operations
        from parsec_python.acceleration.Symmetry import axis_reflection
        from parsec_python.acceleration.Symmetry.geometry_cache import load_or_detect_reflection_reduction
        from parsec_python.models import GridSettings

        bond = 4.9 / np.sqrt(3.0)
        corners = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        atoms = (
            (Atom("Ge", (0.0, 0.0, 0.0)),)
            + tuple(Atom("Ge", (bond * x, bond * y, bond * z)) for x, y, z in corners)
            + tuple(Atom("H", (-2.0 * bond * x, -2.0 * bond * y, -2.0 * bond * z)) for x, y, z in corners)
        )
        grid = build_cluster_grid(GridSettings(spacing=0.7, radius=9.0, expansion_order=6))
        abelian, _ = load_or_detect_reflection_reduction(grid, atoms, cache_directory=None)
        reduction = axis_reflection.detect_point_group_reduction(grid, atoms, abelian)
        self.assertIsInstance(reduction, axis_reflection.PointGroupReduction)
        full = np.asarray(reduction.full_operations, dtype=np.float64).reshape(-1, 3, 3)
        self.assertEqual(full.shape[0], 24)
        direct = axis_reflection.atom_image_table(atoms, full, None, grid, ATOM_TOLERANCE)
        composed = axis_reflection.group_atom_image_table(atoms, reduction, grid, ATOM_TOLERANCE)
        np.testing.assert_array_equal(composed, direct)
        operations = sector_group_operations(reduction, atoms, grid)
        self.assertTrue(np.array_equal(operations[0].matrix, np.eye(3)))

    def test_without_coset_representatives_every_operation_is_queried(self) -> None:
        from parsec_python.acceleration.Symmetry import axis_reflection
        from parsec_python.acceleration.Forces.sector_symmetry import ATOM_TOLERANCE

        reduction, grid, atoms = self._reduction(_silicon_cell((2, 2, 2)), "auto")
        composed = axis_reflection.group_atom_image_table(atoms, reduction, grid, ATOM_TOLERANCE)
        direct = axis_reflection.group_atom_image_table(atoms, replace(reduction, coset_operations=None), grid, ATOM_TOLERANCE)
        np.testing.assert_array_equal(composed, direct)

    def test_a_structure_that_left_the_group(self) -> None:
        from parsec_python.acceleration.Forces.sector_symmetry import sector_group_operations
        from parsec_python.acceleration.Symmetry import axis_reflection
        from parsec_python.acceleration.Forces.sector_symmetry import ATOM_TOLERANCE

        reduction, grid, atoms = self._reduction(_silicon_cell((2, 2, 2)), "auto")
        moved = list(atoms)
        moved[5] = Atom(moved[5].symbol, tuple(np.asarray(moved[5].position) + 0.01))
        self.assertIsNone(axis_reflection.group_atom_image_table(tuple(moved), reduction, grid, ATOM_TOLERANCE))
        with self.assertRaises(ValueError):
            sector_group_operations(reduction, moved, grid, np.diag(np.asarray(grid.cell.lattice_vectors).diagonal()))


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceSeriesTests(unittest.TestCase):
    """The streamed series on the GPU: the numbers of the stack, the host route, and its device peak."""

    def _peak(self, function):
        import cupy as cp
        from cupy.cuda import memory_hook

        class Hook(memory_hook.MemoryHook):
            name = "s8_live_peak"

            def __init__(self):
                self.live = self.peak = 0

            def malloc_postprocess(self, **values):
                self.live += int(values["mem_size"])
                self.peak = max(self.peak, self.live)

            def free_postprocess(self, **values):
                self.live -= int(values["mem_size"])

        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        hook = Hook()
        with hook:
            value = function()
            cp.cuda.runtime.deviceSynchronize()
        cp.fft.config.get_plan_cache().clear()
        return value, hook.peak

    def test_the_device_series_equals_the_stack_and_stays_within_three_fields(self) -> None:
        import cupy as cp

        lattice = np.diag([10.8, 10.8, 10.8]) * 2.0
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.2, expansion_order=6))
        self.assertEqual(tuple(grid.points_per_axis), (108, 108, 108))
        density, _ = _fields(grid, lattice)
        positions = np.random.default_rng(11).random((40, 3)) @ lattice
        charges = np.full(40, 4.0)
        stacked, stacked_peak = self._peak(lambda: _series(cp, grid, lattice, positions, charges, density, "stack"))
        streamed, streamed_peak = self._peak(lambda: _series(cp, grid, lattice, positions, charges, density, "stream"))
        host = _series(cp, grid, lattice, positions, charges, density, "host")
        _assert_same_series(self, streamed, stacked, "device")
        _assert_same_series(self, host, stacked, "host")
        field = 16 * grid.size
        print(f"\n108^3 series: stack peak {stacked_peak / field:.2f}, stream peak {streamed_peak / field:.2f} complex fields")
        self.assertLessEqual(streamed_peak, 3 * field)
        self.assertLess(streamed_peak, stacked_peak)

    def test_a_device_short_of_memory_takes_the_host(self) -> None:
        import cupy as cp

        lattice = _CELLS["fcc"]
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6, shift=_SHIFTS["odd"]))
        density, _ = _fields(grid, lattice)
        positions = np.random.default_rng(2).random((9, 3)) @ lattice
        charges = np.full(9, 4.0)
        reference = _series(np, grid, lattice, positions, charges, density, "stream")
        with mock.patch("parsec_python.acceleration.Hartree.gpu_memory.device_free_bytes", return_value=1024):
            with mock.patch.object(periodic_local, "_streamed_reciprocal_forces", wraps=periodic_local._streamed_reciprocal_forces) as spy:
                value = _series(cp, grid, lattice, positions, charges, density, "stream")
        self.assertIs(spy.call_args.args[0], np)
        np.testing.assert_array_equal(value, reference)


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class SectorForceOrbitTests(unittest.TestCase):
    """compute_forces of a cell in sectors: the local terms of the orbits against those of every atom."""

    def test_displaced_silicon(self) -> None:
        from parsec_python.acceleration import driver

        problem = _silicon_cell()
        tables = {}
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY": "auto"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(problem, backend="cupy")
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system)
            self.assertTrue(result.converged)
            for setting in ("all", "orbits"):
                with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SECTOR_LOCAL_FORCES": setting}):
                    tables[setting] = driver.compute_forces(system, result, symmetrize=False)
        everything, orbits = tables["all"], tables["orbits"]
        self.assertGreater(float(np.abs(everything.total).max()), 1.0e-3)
        np.testing.assert_allclose(orbits.local.ionic, everything.local.ionic, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(orbits.local.core_correction, everything.local.core_correction, rtol=0, atol=1.0e-10)
        np.testing.assert_allclose(orbits.total, everything.total, rtol=0, atol=1.0e-10)
        for table in (everything, orbits):
            self.assertIn("local_ewald_asymmetry", table.phase_seconds)
            self.assertLess(table.phase_seconds["local_ewald_asymmetry"], 1.0e-9)
        print(
            f"\nSi8 in sectors: local {everything.phase_seconds['local']:.3f} s (all) vs "
            f"{orbits.phase_seconds['local']:.3f} s (orbits); asymmetry {everything.phase_seconds['local_ewald_asymmetry']:.2e}"
        )


if __name__ == "__main__":
    unittest.main()
