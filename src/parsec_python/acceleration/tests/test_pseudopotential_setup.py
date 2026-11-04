"""Double grid and ``Old_Interpolation_Format`` on the accelerated path.

The builders of the accelerated preparation must give what the readable
builders give: the double-grid projectors and their gradients
(``acceleration/V_ion/double_grid.py``, bit for bit), the local potential
interpolated directly by the native and CuPy kernels, and the device local
forces of such a species.  A complete accelerated calculation of a small
water molecule is compared with the reference solver, and the routing of the
accelerated command line and the prepared-system cache key are checked.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Forces import (
    build_nonlocal_projector_gradients,
    compute_forces as readable_forces,
    local_forces,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF import prepare_single_point as prepare_readable, run_scf as run_readable
from parsec_python.V_ion import (
    build_local_ionic_potential,
    build_nonlocal_projectors,
    load_pseudopotentials,
)
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.cli import _reference_only_features
from parsec_python.acceleration.V_ion.double_grid import (
    build_double_grid_projector_gradients,
    build_double_grid_projectors,
    build_projector_gradients,
)
from parsec_python.acceleration.V_ion.setup_options import hash_pseudopotential_setup
from parsec_python.models import Atom, GridSettings, SpeciesPotential

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
ANGSTROM = 1.0 / 0.529177

# Off the symmetry planes of the grid, so that every term is exercised.
WATER = (
    Atom("O", np.array([0.031, 0.1163, -0.017])),
    Atom("H", np.array([0.02, -0.92304, 1.79524])),
    Atom("H", np.array([-0.01, -0.95, -1.76])),
)


def _species(*, spline: bool = False, old: bool = False):
    return {
        "O": SpeciesPotential(
            DATA / "spin_ca_O_POTRE.DAT", 1, use_spline=spline, old_interpolation=old
        ),
        "H": SpeciesPotential(
            DATA / "spin_ca_H_POTRE.DAT", 0, use_spline=spline, old_interpolation=old
        ),
    }


def _grid(order: int = 1):
    return build_cluster_grid(
        GridSettings(
            spacing=0.3 * ANGSTROM,
            radius=3.5 * ANGSTROM,
            expansion_order=8,
            double_grid_order=order,
        )
    )


def _assert_same_matrix(test, left, right) -> None:
    left, right = left.tocsc(), right.tocsc()
    test.assertEqual(left.shape, right.shape)
    np.testing.assert_array_equal(left.indptr, right.indptr)
    np.testing.assert_array_equal(left.indices, right.indices)
    np.testing.assert_array_equal(left.data, right.data)


def _hybrid_available() -> bool:
    return cupy_available() and native_available()


class DoubleGridBuilderTests(unittest.TestCase):
    def test_projectors_are_those_of_the_readable_builder(self) -> None:
        from parsec_python.acceleration.V_ion import NativeIonicBuilders

        for spline in (False, True):
            species = _species(spline=spline)
            potentials = load_pseudopotentials(species)
            for order in (2, 3):
                with self.subTest(spline=spline, order=order):
                    grid = _grid(order)
                    readable = build_nonlocal_projectors(grid, WATER, potentials, species)
                    fast = build_double_grid_projectors(grid, WATER, potentials, species)
                    self.assertEqual(fast.labels, readable.labels)
                    np.testing.assert_array_equal(fast.signs, readable.signs)
                    _assert_same_matrix(self, fast.projectors, readable.projectors)
                    if native_available():
                        # The builder hook of the accelerated preparation.
                        hooked = NativeIonicBuilders().build_nonlocal_projectors(
                            grid, WATER, potentials, species
                        )
                        _assert_same_matrix(self, hooked.projectors, readable.projectors)

    def test_gradients_are_those_of_the_readable_builder(self) -> None:
        for spline in (False, True):
            species = _species(spline=spline)
            potentials = load_pseudopotentials(species)
            for order in (2, 3):
                with self.subTest(spline=spline, order=order):
                    grid = _grid(order)
                    readable = build_nonlocal_projector_gradients(
                        grid, WATER, potentials, species
                    )
                    fast = build_double_grid_projector_gradients(
                        grid, WATER, potentials, species
                    )
                    self.assertEqual(fast.labels, readable.labels)
                    np.testing.assert_array_equal(fast.signs, readable.signs)
                    for left, right in zip(fast.components, readable.components):
                        _assert_same_matrix(self, left, right)

    def test_atoms_shared_among_threads(self) -> None:
        # From 32 atoms on the atoms are shared among host threads.
        from parsec_python.acceleration.V_ion import double_grid

        species = _species(spline=True)
        potentials = load_pseudopotentials(species)
        generator = np.random.default_rng(11)
        atoms = tuple(
            Atom("O" if index % 3 == 0 else "H", generator.uniform(-4.0, 4.0, 3))
            for index in range(36)
        )
        if (double_grid.os.cpu_count() or 1) > 1:
            self.assertGreater(double_grid._workers(len(atoms)), 1)
        grid = build_cluster_grid(
            GridSettings(
                spacing=0.45 * ANGSTROM,
                radius=3.2 * ANGSTROM,
                expansion_order=8,
                double_grid_order=2,
            )
        )
        readable = build_nonlocal_projectors(grid, atoms, potentials, species)
        fast = build_double_grid_projectors(grid, atoms, potentials, species)
        self.assertEqual(fast.labels, readable.labels)
        _assert_same_matrix(self, fast.projectors, readable.projectors)
        readable_gradients = build_nonlocal_projector_gradients(grid, atoms, potentials, species)
        fast_gradients = build_double_grid_projector_gradients(grid, atoms, potentials, species)
        self.assertEqual(fast_gradients.labels, readable_gradients.labels)
        for left, right in zip(fast_gradients.components, readable_gradients.components):
            _assert_same_matrix(self, left, right)

    def test_order_one_gradients_are_the_readable_ones(self) -> None:
        species = _species()
        potentials = load_pseudopotentials(species)
        grid = _grid(1)
        readable = build_nonlocal_projector_gradients(grid, WATER, potentials, species)
        dispatched = build_projector_gradients(grid, WATER, potentials, species)
        for left, right in zip(dispatched.components, readable.components):
            _assert_same_matrix(self, left, right)

    def test_gradients_are_the_derivatives_of_the_projectors(self) -> None:
        # Nonlocal energy of fixed orbitals: with cubic splines the radial
        # functions are smooth and the double-grid sum is differentiable.
        species = _species(spline=True)
        potentials = load_pseudopotentials(species)
        grid = _grid(2)
        oxygen = [WATER[0]]
        relative = grid.coordinates - WATER[0].position
        radius2 = np.einsum("ij,ij->i", relative, relative)
        orbitals = np.sqrt(grid.volume_element) * np.column_stack(
            [
                np.exp(-0.5 * radius2) * (1.0 + relative @ np.array(direction))
                for direction in ((0.3, -0.2, 0.5), (-0.4, 0.1, 0.2))
            ]
        )
        occupations = np.array([1.0, 0.5])

        def energy(position) -> float:
            operator = build_double_grid_projectors(
                grid, [Atom("O", position)], potentials, species
            )
            overlaps = operator.projectors.T @ orbitals
            return float(
                2.0 * np.sum(operator.signs[:, None] * overlaps**2 * occupations[None, :])
            )

        from parsec_python.Forces import nonlocal_forces

        operator = build_double_grid_projectors(grid, oxygen, potentials, species)
        gradients = build_double_grid_projector_gradients(grid, oxygen, potentials, species)
        analytic = nonlocal_forces(operator, gradients, orbitals, occupations, 1)[0]
        step = 1.0e-5
        for component in range(3):
            shifted = [np.array(WATER[0].position, dtype=float) for _ in range(2)]
            shifted[0][component] += step
            shifted[1][component] -= step
            difference = -(energy(shifted[0]) - energy(shifted[1])) / (2.0 * step)
            # A fine point that crosses the projector radius within the step
            # leaves a kink (the spline slope there is not zero).
            self.assertAlmostEqual(
                analytic[component],
                difference,
                delta=1.0e-6 * max(1.0, abs(difference)),
            )


def _truncated(potential, samples: int):
    """``potential`` with its radial tables cut after ``samples`` samples."""

    return replace(
        potential,
        radii=potential.radii[:samples].copy(),
        channel_potentials={
            l: values[:samples].copy() for l, values in potential.channel_potentials.items()
        },
        valence_density=potential.valence_density[:samples].copy(),
        core_density=potential.core_density[:samples].copy(),
        radial_wavefunctions={
            l: values[:samples].copy() for l, values in potential.radial_wavefunctions.items()
        },
    )


@unittest.skipUnless(native_available(), "native extension is unavailable")
class OldInterpolationNativeTests(unittest.TestCase):
    def test_native_local_potential_interpolates_v_directly(self) -> None:
        from parsec_python.acceleration.V_ion import NativeIonicBuilders

        grid = _grid()
        potentials = load_pseudopotentials(_species())
        # A table that ends inside the grid exercises the Coulomb tail.
        oxygen = potentials["O"]
        samples = int(np.searchsorted(oxygen.radii, 3.0)) + 2
        cut = {"O": _truncated(oxygen, samples), "H": potentials["H"]}
        self.assertLess(cut["O"].interpolation_cutoff, 3.5)
        for tables in (potentials, cut):
            for spline in (False, True):
                with self.subTest(truncated=tables is cut, spline=spline):
                    old = _species(spline=spline, old=True)
                    readable = build_local_ionic_potential(grid, WATER, tables, old)
                    native = NativeIonicBuilders().build_local_ionic_potential(
                        grid, WATER, tables, old
                    )
                    np.testing.assert_allclose(native, readable, rtol=1e-14, atol=1e-14)
                    if not spline:
                        # The option changes the field (a cubic spline takes precedence).
                        default = build_local_ionic_potential(
                            grid, WATER, tables, _species(spline=spline)
                        )
                        self.assertGreater(np.max(np.abs(default - readable)), 1e-5)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class OldInterpolationDeviceTests(unittest.TestCase):
    def test_cupy_local_potential_interpolates_v_directly(self) -> None:
        from parsec_python.acceleration.V_ion.cupy_ionic import CupyIonicBuilders

        grid = _grid()
        potentials = load_pseudopotentials(_species())
        oxygen = potentials["O"]
        samples = int(np.searchsorted(oxygen.radii, 3.0)) + 2
        cut = {"O": _truncated(oxygen, samples), "H": potentials["H"]}
        for tables in (potentials, cut):
            for spline in (False, True):
                with self.subTest(truncated=tables is cut, spline=spline):
                    old = _species(spline=spline, old=True)
                    readable = build_local_ionic_potential(grid, WATER, tables, old)
                    device = CupyIonicBuilders().build_local_ionic_potential(
                        grid, WATER, tables, old
                    )
                    np.testing.assert_allclose(device, readable, rtol=1e-14, atol=1e-14)
        # A mixture of flagged and ordinary species.
        mixed = _species()
        mixed["O"] = replace(mixed["O"], old_interpolation=True)
        np.testing.assert_allclose(
            CupyIonicBuilders().build_local_ionic_potential(grid, WATER, potentials, mixed),
            build_local_ionic_potential(grid, WATER, potentials, mixed),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_device_local_forces_of_direct_interpolation(self) -> None:
        from parsec_python.acceleration.Forces.local_device import cupy_local_forces

        grid = _grid()
        generator = np.random.default_rng(7)
        density = generator.random(grid.size)
        residual = 1e-2 * generator.standard_normal(grid.size)
        xc = -generator.random(grid.size)
        mixed = _species()
        mixed["O"] = replace(mixed["O"], old_interpolation=True)
        for species in (_species(old=True), mixed, _species(spline=True, old=True)):
            potentials = load_pseudopotentials(species)
            readable = local_forces(grid, WATER, potentials, species, density, residual, xc)
            device = cupy_local_forces(grid, WATER, potentials, species, density, residual, xc)
            for name in ("ionic", "self_consistency", "core_correction"):
                np.testing.assert_allclose(
                    getattr(device, name), getattr(readable, name), rtol=0, atol=1e-11
                )
        # The option changes the ionic term.
        potentials = load_pseudopotentials(_species())
        default = local_forces(grid, WATER, potentials, _species(), density, residual, xc)
        old = local_forces(grid, WATER, potentials, _species(old=True), density, residual, xc)
        self.assertGreater(np.max(np.abs(default.ionic - old.ionic)), 1e-6)


INPUT = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: 3.6 ang
Grid_Spacing: 0.35 ang
Expansion_Order: 8
States_Num: 6
Fermi_Temp: 500.0
Convergence_Criterion: 1.0e-8
Diag_Tolerance: 1.0e-9
Max_Iter: 80
Eigensolver: chebff
Mixing_Method: anderson
Correlation_Type: ca
Hartree_Boundary_Tolerance: off
Ignore_Symmetry: {ignore}
Skip_Force: .false.
{extra}
Atom_Types_Num: 2
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.000  -0.488456   0.950000
   0.000  -0.488456  -0.950000
end Atom_Coord
Atom_Type: O
Local_Component: p
begin Atom_Coord
   0.000   0.061544   0.000000
end Atom_Coord
"""


def _write_input(directory: Path, extra: str, *, ignore_symmetry: bool = True) -> Path:
    (directory / "O_POTRE.DAT").write_bytes((DATA / "spin_ca_O_POTRE.DAT").read_bytes())
    (directory / "H_POTRE.DAT").write_bytes((DATA / "spin_ca_H_POTRE.DAT").read_bytes())
    path = directory / "parsec.in"
    path.write_text(
        INPUT.format(extra=extra, ignore=".true." if ignore_symmetry else ".false."),
        encoding="utf-8",
    )
    return path


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)

    def features(self, extra: str) -> list[str]:
        translation = parse_parsec_input(_write_input(self.directory, extra))
        return _reference_only_features(translation, accelerated_backend_selected=True)

    def test_cluster_options_run_accelerated(self) -> None:
        self.assertEqual(self.features(""), [])
        self.assertEqual(self.features("Double_Grid_Order: 2"), [])
        self.assertEqual(self.features("Old_Interpolation_Format: .true."), [])
        self.assertEqual(
            self.features("Double_Grid_Order: 3\nOld_Interpolation_Format: .true."), []
        )
        self.assertEqual(self.features("Double_Grid_Order: 2\nSpin_Polarization: .true."), [])
        self.assertEqual(
            self.features("Double_Grid_Order: 2\nMinimization: bfgs\nMovement_Num: 2"), []
        )

    def test_double_grid_with_spin_orbit_species_stays_on_the_reference(self) -> None:
        translation = parse_parsec_input(
            _write_input(self.directory, "Double_Grid_Order: 2")
        )
        problem = translation.problem
        oxygen = replace(problem.pseudopotentials["O"], spin_orbit=True)
        translation = replace(
            translation,
            problem=replace(
                problem, pseudopotentials={**problem.pseudopotentials, "O": oxygen}
            ),
        )
        self.assertIn(
            "the double grid with spin-orbit projectors (Double_Grid_Order > 1 with SO_PSP)",
            _reference_only_features(translation, accelerated_backend_selected=True),
        )


class BackendDetailTests(unittest.TestCase):
    def test_details_only_where_an_option_is_set(self) -> None:
        from parsec_python.acceleration.V_ion.setup_options import setup_option_details

        native = SimpleNamespace(finite_difference_builder="native")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            def details(extra: str):
                problem = parse_parsec_input(_write_input(directory, extra)).problem
                return dict(setup_option_details(problem, native))

            self.assertEqual(details(""), {})
            self.assertIn("NumPy fine-point sums", details("Double_Grid_Order: 2")["projector_double_grid"])
            self.assertIn(
                "Old_Interpolation_Format",
                details("Old_Interpolation_Format: .true.")["local_potential_interpolation"],
            )


class CacheKeyTests(unittest.TestCase):
    def test_only_the_option_adds_to_the_key(self) -> None:
        default = SpeciesPotential(DATA / "spin_ca_O_POTRE.DAT", 1)
        digest = sha256()
        hash_pseudopotential_setup(digest, default)
        self.assertEqual(digest.hexdigest(), sha256().hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)

            def key(extra: str) -> str:
                return driver._reference_cache_key(
                    parse_parsec_input(_write_input(directory, extra)).problem,
                    SimpleNamespace(finite_difference_builder="native"),
                    defer_native_laplacian=False,
                    cache_directory=None,
                )

            plain = key("")
            self.assertEqual(plain, key(""))
            self.assertNotEqual(plain, key("Old_Interpolation_Format: .true."))
            self.assertNotEqual(plain, key("Double_Grid_Order: 2"))


@unittest.skipUnless(_hybrid_available(), "CUDA runtime or native extension is unavailable")
class CompleteCalculationTests(unittest.TestCase):
    """The accelerated SCF and forces against the reference solver."""

    def _compare(self, extra: str, *, ignore_symmetry: bool = True) -> None:
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(
                _write_input(Path(directory), extra, ignore_symmetry=ignore_symmetry)
            )
            problem = translation.problem
            readable_system = prepare_readable(problem)
            readable = run_readable(readable_system)
            readable_force = readable_forces(readable_system, readable)
            system = driver.prepare_single_point(
                problem, symmetry="off" if ignore_symmetry else "auto"
            )
            result = driver.run_scf(system)
            force = driver.compute_forces(system, result)
        self.assertTrue(readable.converged and result.converged)
        self.assertAlmostEqual(result.energies.total, readable.energies.total, delta=1e-7)
        occupied = np.asarray(readable.occupations) > 1e-6
        np.testing.assert_allclose(
            np.asarray(result.eigenvalues)[occupied],
            np.asarray(readable.eigenvalues)[occupied],
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(force.total, readable_force.total, rtol=0, atol=1e-6)

    def test_double_grid(self) -> None:
        self._compare("Double_Grid_Order: 2")

    def test_double_grid_in_symmetry_sectors(self) -> None:
        self._compare("Double_Grid_Order: 3", ignore_symmetry=False)

    def test_old_interpolation_format(self) -> None:
        self._compare("Old_Interpolation_Format: .true.")


PERIODIC = """Boundary_Conditions: bulk
Periodic_System: .true.
begin Cell_Shape
  5.6  5.6  5.6
end Cell_Shape
Grid_Spacing: 0.56
Expansion_Order: 8
Ignore_Symmetry: .true.
States_Num: 8
Fermi_Temp: 1500.0
Max_Iter: 100
Convergence_Criterion: 1.0e-8
Diag_Tolerance: 1.0e-9
Eigensolver: chebff
Mixing_Method: anderson
Mixing_Param: 0.2
Correlation_Type: ca
Coordinate_Unit: Cartesian_Bohr
{extra}
Atom_Types_Num: 1
Atom_Type: Si
Local_Component: p
Alpha_Filter: 0.8
Beta1_Filter: 5.0
begin Atom_Coord
   0.10  0.05  0.00
   1.50  1.38  1.45
end Atom_Coord
"""


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class PeriodicSetupTests(unittest.TestCase):
    """A periodic cell at Gamma takes the readable preparation on the accelerated path."""

    def test_filters_and_double_grid_at_gamma(self) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "Si_POTRE.DAT").write_bytes((DATA / "Si_POTRE.DAT").read_bytes())
            path = directory / "parsec.in"
            path.write_text(PERIODIC.format(extra="Double_Grid_Order: 2"), encoding="utf-8")
            translation = parse_parsec_input(path)
            self.assertTrue(translation.problem.pseudopotentials["Si"].has_fourier_filter)
            self.assertEqual(
                _reference_only_features(translation, accelerated_backend_selected=True), []
            )
            problem = translation.problem
            readable = run_readable(prepare_periodic_single_point(problem))
            system = driver.prepare_single_point(problem, symmetry="off")
            result = driver.run_scf(system)
        self.assertEqual(system.backend_info.selected, "cupy")
        self.assertTrue(readable.converged and result.converged)
        self.assertAlmostEqual(result.energies.total, readable.energies.total, delta=1e-7)
        occupied = np.asarray(readable.occupations) > 1e-3
        np.testing.assert_allclose(
            np.asarray(result.eigenvalues)[occupied],
            np.asarray(readable.eigenvalues)[occupied],
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
