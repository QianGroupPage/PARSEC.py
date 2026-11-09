"""The fast periodic preparation reproduces the readable definitions.

``grid_ball`` must return exactly what the dense scan of every grid point returns, so projectors and
atomic densities built through it are identical; the FFT local ionic potential must equal the direct
Ewald sum once that sum is converged (the sum at the default tolerance stops at the radius where
the images it leaves out are still of order 1e-6 Ry).

The bitwise comparisons with the readable definitions are those of the table reach
(``PARSEC_PERIODIC_SETUP_REACH=table``), which the classes below name: since 2026-10-30 the default is
the physical reach, which drops terms below round-off (``test_the_physical_reach_is_the_default...``).
"""

from __future__ import annotations

import functools
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.special import erf

from parsec_python import (
    Atom,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
# The tests of the parity round for its own module (Use_Fortran_Routines: the grids say fortran_routines=True);
# Andrew Grieder's module, the default, has its tests in test_periodic_setup.py.
from parsec_python.acceleration.V_ion import periodic_setup_fortran as periodic_setup
from parsec_python.Forces.periodic import periodic_local_forces
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion import (
    build_nonlocal_projectors,
    load_pseudopotentials,
    superpose_atomic_density,
)
from parsec_python.V_ion.ewald import ewald_local_ionic_potential
from parsec_python.V_ion.ionic_potential import dense_support

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
# The table reach sums to the end of the radial tables, bit for bit the readable definitions (the
# default up to 2026-10-30, when these classes ran without naming it).
_TABLE_REACH = mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_REACH": "table"})
_SIDES = (6.0, 7.2, 8.4)  # spacing 0.6 on all three axes


@functools.lru_cache(maxsize=None)
def _system():
    problem = SinglePointInput(
        atoms=[
            Atom("Au", [1.3, 5.9, 0.2], [0.0, 0.0, 0.0]),
            Atom("H", [5.7, 0.4, 7.9], [0.0, 0.0, 0.0]),
            Atom("H", [3.1, 3.3, 4.6], [0.0, 0.0, 0.0]),
        ],
        pseudopotentials={
            "Au": SpeciesPotential(_EXAMPLES / "Au_POTRE.DAT", 0),
            "H": SpeciesPotential(_EXAMPLES / "H_POTRE.DAT", 0),
        },
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=np.diag(_SIDES)),
        scf=SCFSettings(max_iterations=5, number_of_states=8),
        hartree=HartreeSettings(multipole_order=4),
    )
    periodic_setup.uninstall()
    try:
        return problem, prepare_periodic_single_point(problem)
    finally:
        periodic_setup.install()


class GridBallTests(unittest.TestCase):
    def test_equals_the_dense_scan_inside_and_outside_the_cell(self) -> None:
        _, system = _system()
        generator = np.random.default_rng(3)
        for _ in range(60):
            centre = generator.uniform(-9.0, 17.0, size=3)
            radius = float(generator.uniform(0.2, 9.0))
            rows, relative, distances = periodic_setup.grid_ball(system.grid, centre, radius)
            expected = dense_support(system.grid, centre, radius)
            np.testing.assert_array_equal(rows, expected[0])
            np.testing.assert_array_equal(relative, expected[1])
            np.testing.assert_array_equal(distances, expected[2])

    def test_other_grids_use_the_dense_scan(self) -> None:
        class Plain:
            coordinates = np.random.default_rng(1).uniform(0, 5, size=(40, 3))

        rows, _, _ = periodic_setup.grid_ball(Plain, np.array([2.0, 2.0, 2.0]), 2.0)
        np.testing.assert_array_equal(rows, dense_support(Plain, np.array([2.0, 2.0, 2.0]), 2.0)[0])


@_TABLE_REACH
class SupportBuildersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem, self.system = _system()
        self.addCleanup(periodic_setup.install)

    def _build(self, fast: bool, k_point=None):
        (periodic_setup.install if fast else periodic_setup.uninstall)()
        return build_nonlocal_projectors(
            self.system.grid,
            self.system.atoms,
            self.system.pseudopotentials,
            self.problem.pseudopotentials,
            np.diag(_SIDES),
            k_point=k_point,
        )

    def test_projectors_are_identical(self) -> None:
        for k_point in (None, np.array([0.21, -0.37, 0.11])):
            with self.subTest(k_point=k_point):
                dense, fast = self._build(False, k_point), self._build(True, k_point)
                difference = abs(dense.projectors - fast.projectors)
                self.assertEqual(dense.projectors.nnz, fast.projectors.nnz)
                self.assertEqual(difference.max() if difference.nnz else 0.0, 0.0)

    def test_atomic_densities_are_identical(self) -> None:
        for core in (False, True):
            with self.subTest(core=core):
                results = []
                for fast in (False, True):
                    (periodic_setup.install if fast else periodic_setup.uninstall)()
                    results.append(
                        superpose_atomic_density(
                            self.system.grid,
                            self.system.atoms,
                            self.system.pseudopotentials,
                            self.problem.pseudopotentials,
                            np.diag(_SIDES),
                            core=core,
                        )
                    )
                np.testing.assert_array_equal(results[0], results[1])
                if not core:
                    self.assertGreater(results[0].max(), 0.0)


_CONVERGED: dict[str, np.ndarray] = {}


def _converged_direct_sum(arguments) -> np.ndarray:
    if "value" not in _CONVERGED:
        _CONVERGED["value"] = ewald_local_ionic_potential(*arguments, tolerance=1.0e-26)
    return _CONVERGED["value"]


@_TABLE_REACH
class FFTLocalPotentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem, self.system = _system()
        self.addCleanup(periodic_setup.install)
        self.arguments = (
            self.system.grid,
            self.system.atoms,
            self.system.pseudopotentials,
            self.problem.pseudopotentials,
            np.diag(_SIDES),
        )

    def test_matches_the_converged_direct_sum(self) -> None:
        periodic_setup.uninstall()
        converged = _converged_direct_sum(self.arguments)
        fast = periodic_setup.fft_local_ionic_potential(
            *self.arguments, None, 1.0e-14
        )
        np.testing.assert_allclose(fast, converged, atol=1.0e-9)

    def test_stays_within_the_accuracy_of_the_default_direct_sum(self) -> None:
        periodic_setup.uninstall()
        direct = ewald_local_ionic_potential(*self.arguments)
        periodic_setup.install()
        provided = ewald_local_ionic_potential(*self.arguments)
        self.assertLess(np.max(np.abs(provided - direct)), 1.0e-4)
        # the constant of the G = 0 term is the reference's: the means agree
        self.assertAlmostEqual(float(provided.mean()), float(direct.mean()), delta=1.0e-5)

    def test_the_image_blocks_equal_the_ball_loop_to_the_end_of_the_tables(self) -> None:
        # the table reach (the end of the radial tables, ~118 bohr): ~10^4 images of each atom
        blocks = periodic_setup.fft_local_ionic_potential(*self.arguments, None, 1.0e-14)
        with mock.patch.object(periodic_setup, "_axis_coordinates", return_value=None):
            balls = periodic_setup.fft_local_ionic_potential(*self.arguments, None, 1.0e-14)
        _assert_same_bits(self, blocks, balls)

    def test_an_oblique_cell_falls_back_to_the_direct_sum(self) -> None:
        oblique = np.diag(_SIDES) + np.array([[0, 0.5, 0], [0, 0, 0], [0, 0, 0]])
        result = periodic_setup.fft_local_ionic_potential(
            self.system.grid,
            self.system.atoms,
            self.system.pseudopotentials,
            self.problem.pseudopotentials,
            oblique,
            None,
            1.0e-14,
        )
        self.assertIsNone(result)


_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def _assert_same_bits(case: unittest.TestCase, first, second) -> None:
    first, second = np.ascontiguousarray(first), np.ascontiguousarray(second)
    case.assertEqual(first.shape, second.shape)
    differing = int(np.count_nonzero(first.view(np.uint64) != second.view(np.uint64)))
    case.assertEqual(differing, 0, f"{differing} values differ, max {np.max(np.abs(first - second)):.3e}")


@functools.lru_cache(maxsize=None)
def _mixed_cell():
    """Ge (core correction, 118 bohr tables) and H with core correction and a six-sample table.

    H sits on a grid point (a distance of exactly 0) and its table ends at 1.6 bohr, inside the erfc
    tail of the split: the terms beyond the table are evaluated and then exactly zero.

    The sides are even multiples of 0.6 (Fortran's counts of the orthorhombic grid since PARSEC 1.5's parity
    round; 5.4 bohr, 9 points of the earlier ``round(L/h)``, now takes 8 points of 0.675 and the general grid).
    """
    sides = (4.8, 6.0, 7.2)  # spacing 0.6
    atoms = (
        Atom("Ge", [1.37, 4.02, 2.71], [0.0, 0.0, 0.0]),
        Atom("H", [3.0, 1.2, 4.8], [0.0, 0.0, 0.0]),
        Atom("Ge", [4.11, 0.35, 5.66], [0.0, 0.0, 0.0]),
    )
    specifications = {
        "Ge": SpeciesPotential(_DATA / "Ge_POTRE.DAT", 1),
        "H": SpeciesPotential(_DATA / "H_CORE_POTRE.DAT", 0),
    }
    cell = PeriodicCell(lattice_vectors=np.diag(sides))
    grid = build_periodic_grid(cell, PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=6))
    return grid, atoms, load_pseudopotentials(specifications), specifications, np.diag(sides)


def _readable_density(grid, atoms, potentials, specifications, lattice, core):
    periodic_setup.uninstall()
    try:
        return superpose_atomic_density(grid, atoms, potentials, specifications, lattice, core=core)
    finally:
        periodic_setup.install()


@_TABLE_REACH
class ImageBlockTests(unittest.TestCase):
    """The image blocks sum the same terms in the same order as the ball loop and the readable sum."""

    def setUp(self) -> None:
        self.addCleanup(periodic_setup.install)
        periodic_setup.install()

    def _potential(self, cell, *, balls: bool = False, **environment):
        grid, atoms, potentials, specifications, lattice = cell
        with mock.patch.dict(os.environ, environment):
            if balls:
                with mock.patch.object(periodic_setup, "_axis_coordinates", return_value=None):
                    return periodic_setup.fft_local_ionic_potential(
                        grid, atoms, potentials, specifications, lattice, None, 1.0e-14
                    )
            return periodic_setup.fft_local_ionic_potential(
                grid, atoms, potentials, specifications, lattice, None, 1.0e-14
            )

    def test_short_tables_core_densities_and_flushes_inside_atoms(self) -> None:
        cell = _mixed_cell()
        # many flushes: inside the images of one atom, inside one block, at the change of species
        with mock.patch.object(periodic_setup, "_POINTS_PER_CALL", 7919):
            balls = self._potential(cell, balls=True)
            for threads in ("1", "3"):
                with self.subTest(threads=threads):
                    blocks = self._potential(cell, PARSEC_PERIODIC_SETUP_THREADS=threads)
                    _assert_same_bits(self, blocks, balls)

    def test_the_device_density_is_the_host_sum_bit_for_bit(self) -> None:
        # periodic_device.device_bulk_atomic_density: the terms of every point in the host's order and arithmetic
        # (H on a grid point, a six-sample table, core densities), on the grid of either set of routines
        from parsec_python.acceleration.V_ion import periodic_device

        if periodic_device._cupy() is None:
            self.skipTest("no usable CUDA device")
        grid, atoms, potentials, specifications, lattice = _mixed_cell()
        default_grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=lattice), PeriodicGridSettings(spacing=0.6, expansion_order=6)
        )
        for name, target in (("fortran", grid), ("default", default_grid)):
            for core in (False, True):
                with self.subTest(grid=name, core=core):
                    device = periodic_setup._density_provider(target, atoms, potentials, specifications, lattice, core)
                    with mock.patch.object(periodic_device, "_cupy", return_value=None):
                        host = periodic_setup._density_provider(target, atoms, potentials, specifications, lattice, core)
                    self.assertGreater(float(host.max()), 0.0)
                    _assert_same_bits(self, device, host)

    def test_densities_equal_the_readable_sum(self) -> None:
        grid, atoms, potentials, specifications, lattice = _mixed_cell()
        for core in (False, True):
            readable = _readable_density(grid, atoms, potentials, specifications, lattice, core)
            self.assertGreater(float(readable.max()), 0.0)
            for threads in ("1", "4"):
                with self.subTest(core=core, threads=threads), mock.patch.dict(
                    os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": threads}
                ):
                    provided = periodic_setup.periodic_atomic_density(
                        grid, atoms, potentials, specifications, lattice, core
                    )
                    _assert_same_bits(self, provided, readable)
                    # the installed provider is what the readable entry point returns
                    _assert_same_bits(
                        self,
                        superpose_atomic_density(
                            grid, atoms, potentials, specifications, lattice, core=core
                        ),
                        readable,
                    )

    def test_his_bulk_grid_takes_the_index_boxes_with_his_support_radii(self) -> None:
        # Andrew Grieder's grid of the cell (half a step off the origin, a step per axis): his readable sum, whose
        # images reach atomic_density_support_radius, bit for bit
        _, atoms, potentials, specifications, _ = _mixed_cell()
        lattice = np.diag((4.8, 5.4, 6.0))  # a step per axis in his grid
        grid = build_periodic_grid(
            PeriodicCell(lattice_vectors=lattice),
            PeriodicGridSettings(spacing=0.6, expansion_order=6, shift=(0.5, 0.5, 0.5), points_rule="fortran"),
        )
        self.assertEqual(tuple(int(n) for n in grid.points_per_axis), (8, 8, 10))
        for core in (False, True):
            readable = _readable_density(grid, atoms, potentials, specifications, lattice, core)
            self.assertGreater(float(readable.max()), 0.0)
            self.assertIsNotNone(
                periodic_setup._density_provider(grid, atoms, potentials, specifications, lattice, core)
            )
            for threads in ("1", "4"):
                with self.subTest(core=core, threads=threads), mock.patch.dict(
                    os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": threads}
                ):
                    _assert_same_bits(
                        self,
                        superpose_atomic_density(grid, atoms, potentials, specifications, lattice, core=core),
                        readable,
                    )

    def test_a_spline_species_keeps_the_ball_loop_and_the_readable_core_sum(self) -> None:
        grid, atoms, potentials, _, lattice = _mixed_cell()
        specifications = {
            "Ge": SpeciesPotential(_DATA / "Ge_POTRE.DAT", 1, use_spline=True),
            "H": SpeciesPotential(_DATA / "H_CORE_POTRE.DAT", 0),
        }
        cell = (grid, atoms, potentials, specifications, lattice)
        with mock.patch.object(periodic_setup, "_short_range_by_blocks") as blocks:
            periodic_setup.fft_local_ionic_potential(
                grid, atoms, potentials, specifications, lattice, None, 1.0e-14
            )
        blocks.assert_not_called()
        self.assertIsNone(
            periodic_setup.periodic_atomic_density(grid, atoms, potentials, specifications, lattice, True)
        )
        # the valence density is linear in superpose_atomic_density for every species
        _assert_same_bits(
            self,
            periodic_setup.periodic_atomic_density(grid, atoms, potentials, specifications, lattice, False),
            _readable_density(*cell, False),
        )

    def test_a_spline_species_sums_its_potential_to_the_coulomb_onset_under_both_rules(self) -> None:
        # The cubic spline of V leaves -2Z/r by up to 3e-5 Ry over the last ~40 bohr of the Ge table.
        # Summed to the table end (the table reach of a spline species before this fix), that tail of
        # every image moved this potential by -3.9e-2 Ry on average (4.3e-2 at most) and put the local
        # force 5.1e-4 Ry/bohr off the derivative of its energy.
        grid, atoms, _, _, lattice = _mixed_cell()
        specifications = {
            "Ge": SpeciesPotential(_DATA / "Ge_POTRE.DAT", 1, use_spline=True),
            "H": SpeciesPotential(_DATA / "H_CORE_POTRE.DAT", 0),
        }
        potentials = load_pseudopotentials(specifications)
        cell = (grid, atoms, potentials, specifications, lattice)
        table = self._potential(cell)
        _assert_same_bits(self, table, self._potential(cell, PARSEC_PERIODIC_SETUP_REACH="physical"))
        periodic_setup.uninstall()
        try:
            converged = ewald_local_ionic_potential(*cell, tolerance=1.0e-26)
        finally:
            periodic_setup.install()
        # The converged direct sum keeps the spline's 1e-9 Ry deviation from -2Z/r out to its own erfc
        # reach (~30 bohr): 1.3e-7 Ry here.
        self.assertLess(float(np.max(np.abs(table - converged))), 5.0e-7)

        # The local ionic force (the derivative of the direct sum's terms) differentiates this energy
        # at a fixed density: 8e-9 Ry/bohr with steps of 1e-5 bohr (forces up to 0.36 Ry/bohr).
        density = superpose_atomic_density(grid, atoms, potentials, specifications, lattice)
        analytic = periodic_local_forces(
            grid, atoms, potentials, specifications, lattice, density,
            np.zeros(grid.size), np.zeros(grid.size),
        ).ionic
        step = 1.0e-5
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                moved = list(atoms)
                shift = np.zeros(3)
                shift[axis] = sign * step
                moved[0] = Atom("Ge", np.asarray(atoms[0].position) + shift, [0.0, 0.0, 0.0])
                energies.append(
                    grid.volume_element
                    * float(density @ self._potential((grid, moved, potentials, specifications, lattice)))
                )
            finite_difference = -(energies[0] - energies[1]) / (2.0 * step)
            self.assertAlmostEqual(float(analytic[0, axis]), finite_difference, delta=1.0e-7)

    def test_other_grids_and_lattices_keep_the_readable_density(self) -> None:
        grid, atoms, potentials, specifications, lattice = _mixed_cell()
        sheared = lattice + np.array([[0.0, 0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertIsNone(
            periodic_setup.periodic_atomic_density(grid, atoms, potentials, specifications, sheared, False)
        )
        self.assertIsNone(
            periodic_setup.periodic_atomic_density(grid, atoms, potentials, specifications, None, False)
        )

        class Moved:
            # one point off the product of the axis coordinates
            points_per_axis = grid.points_per_axis
            coordinates = grid.coordinates.copy()
            coordinates[17, 1] += 1.0e-12

        self.assertIsNone(periodic_setup._axis_coordinates(Moved))
        self.assertIsNotNone(periodic_setup._axis_coordinates(grid))

    def test_the_physical_reach_is_the_default_and_changes_only_round_off(self) -> None:
        cell = _mixed_cell()
        grid, atoms, potentials, specifications, lattice = cell
        table = self._potential(cell)
        physical = self._potential(cell, PARSEC_PERIODIC_SETUP_REACH="physical")
        self.assertLess(float(np.max(np.abs(physical - table))), 1.0e-11)
        for core in (False, True):
            readable = _readable_density(grid, atoms, potentials, specifications, lattice, core)
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_REACH": "physical"}):
                dropped = periodic_setup.periodic_atomic_density(
                    grid, atoms, potentials, specifications, lattice, core
                )
            self.assertLess(float(np.max(np.abs(dropped - readable))), 1.0e-14 * float(readable.max()))
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_REACH": "everything"}):
            with self.assertRaises(ValueError):
                periodic_setup.reach_rule()
        # ``table`` was the default up to 2026-10-30 (this test asserted it); ``physical`` since.
        for unset in ("", "  "):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_REACH": unset}):
                self.assertEqual(periodic_setup.reach_rule(), "physical")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_PERIODIC_SETUP_REACH", None)
            self.assertEqual(periodic_setup.reach_rule(), periodic_setup.DEFAULT_REACH_RULE)
            self.assertEqual(periodic_setup.DEFAULT_REACH_RULE, "physical")
            _assert_same_bits(self, self._potential(cell), physical)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_REACH": "Table"}):
            self.assertEqual(periodic_setup.reach_rule(), "table")

    def test_ordered_results_keep_the_order_of_the_items(self) -> None:
        # the blocks of the densities and of the local potential are added in the order of their items
        for threads in (1, 4):
            order = list(periodic_setup._ordered_results(lambda t: t * t, list(range(50)), threads))
            self.assertEqual(order, [t * t for t in range(50)])

    def test_thread_setting(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": "3", "OMP_NUM_THREADS": "7"}):
            self.assertEqual(periodic_setup.setup_threads(), 3)
        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": "", "OMP_NUM_THREADS": "7"}):
            self.assertEqual(periodic_setup.setup_threads(), 7)
        for bad in ("0", "two"):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": bad}):
                with self.assertRaises(ValueError):
                    periodic_setup.setup_threads()

    def test_erf_shortcut(self) -> None:
        # the terms take 2 Z/r for 2 Z erf(eta r)/r only where erf(eta r) is exactly 1.0
        self.assertTrue(periodic_setup._erf_saturates())
        for eta in (0.31, 0.6234, 1.7):
            r = periodic_setup._smallest_with_product_at_least(eta, periodic_setup._ERF_ONE_FROM)
            self.assertGreaterEqual(eta * r, periodic_setup._ERF_ONE_FROM)
            self.assertLess(eta * float(np.nextafter(r, 0.0)), periodic_setup._ERF_ONE_FROM)
            self.assertEqual(float(erf(eta * r)), 1.0)


def _shifted(cell, shift):
    """``_mixed_cell`` on its grid shifted by ``shift`` steps (PARSEC 1.5's parity round: Fortran's points)."""
    grid, atoms, potentials, specifications, lattice = cell
    settings = PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=6, shift=shift)
    shifted = build_periodic_grid(grid.cell, settings)
    return shifted, atoms, potentials, specifications, lattice


def _translated(cell, shift):
    """``_mixed_cell`` on its grid without shift, the atoms moved by ``-shift`` steps: the points of ``_shifted``
    relative to the atoms."""
    grid, atoms, potentials, specifications, lattice = cell
    step = float(grid.spacing)
    moved = tuple(
        Atom(atom.symbol, np.asarray(atom.position, dtype=float) - np.asarray(shift) * step, [0.0, 0.0, 0.0])
        for atom in atoms
    )
    return grid, moved, potentials, specifications, lattice


@_TABLE_REACH
class ShiftedGridTests(unittest.TestCase):
    """The orthorhombic grid of these routines with a shift (``Grid_Shift``, 0.5 without ``Ignore_Symmetry``): the
    index boxes and the FFT phases of the fast preparation follow it."""

    SHIFTS = ((0.5, 0.5, 0.5), (0.25, 0.0, 0.5))

    def setUp(self) -> None:
        self.addCleanup(periodic_setup.install)
        periodic_setup.install()

    def test_grid_ball_equals_the_dense_scan(self) -> None:
        generator = np.random.default_rng(11)
        for shift in self.SHIFTS:
            grid = _shifted(_mixed_cell(), shift)[0]
            self.assertEqual(float(grid.coordinates[-1, 2]), (grid.points_per_axis[2] - 1 + shift[2]) * grid.spacing)
            for _ in range(40):
                centre = generator.uniform(-6.0, 12.0, size=3)
                radius = float(generator.uniform(0.2, 7.0))
                with self.subTest(shift=shift):
                    rows, relative, distances = periodic_setup.grid_ball(grid, centre, radius)
                    expected = dense_support(grid, centre, radius)
                    np.testing.assert_array_equal(rows, expected[0])
                    np.testing.assert_array_equal(relative, expected[1])
                    np.testing.assert_array_equal(distances, expected[2])

    def test_densities_equal_the_readable_sum(self) -> None:
        for shift in self.SHIFTS:
            grid, atoms, potentials, specifications, lattice = _shifted(_mixed_cell(), shift)
            for core in (False, True):
                with self.subTest(shift=shift, core=core):
                    readable = _readable_density(grid, atoms, potentials, specifications, lattice, core)
                    provided = periodic_setup.periodic_atomic_density(
                        grid, atoms, potentials, specifications, lattice, core
                    )
                    _assert_same_bits(self, provided, readable)

    def test_the_local_potential_is_the_direct_sum_and_moves_with_the_grid(self) -> None:
        for shift in self.SHIFTS:
            shifted = _shifted(_mixed_cell(), shift)
            with self.subTest(shift=shift):
                fast = self._potential(shifted)
                blocks_free = self._potential(shifted, balls=True)
                _assert_same_bits(self, fast, blocks_free)
                periodic_setup.uninstall()
                try:
                    direct = ewald_local_ionic_potential(*shifted, tolerance=1.0e-26)
                finally:
                    periodic_setup.install()
                np.testing.assert_allclose(fast, direct, rtol=0.0, atol=1.0e-9)
                # the same points relative to the atoms: the grid without shift and the atoms moved back
                translated = self._potential(_translated(_mixed_cell(), shift))
                np.testing.assert_allclose(fast, translated, rtol=0.0, atol=1.0e-10)

    def test_a_zero_shift_keeps_the_numbers_of_the_grid_from_the_origin(self) -> None:
        cell = _mixed_cell()
        _assert_same_bits(self, self._potential(_shifted(cell, (0.0, 0.0, 0.0))), self._potential(cell))
        grid, atoms, potentials, specifications, lattice = cell
        np.testing.assert_array_equal(_shifted(cell, (0.0, 0.0, 0.0))[0].coordinates, grid.coordinates)

    def _potential(self, cell, *, balls: bool = False):
        grid, atoms, potentials, specifications, lattice = cell
        if balls:
            with mock.patch.object(periodic_setup, "_axis_coordinates", return_value=None):
                return periodic_setup.fft_local_ionic_potential(
                    grid, atoms, potentials, specifications, lattice, None, 1.0e-14
                )
        return periodic_setup.fft_local_ionic_potential(grid, atoms, potentials, specifications, lattice, None, 1.0e-14)


if __name__ == "__main__":
    unittest.main()
