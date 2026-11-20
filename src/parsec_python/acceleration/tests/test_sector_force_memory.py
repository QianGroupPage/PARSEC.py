"""Memory of the projector gradients of the nonlocal forces in symmetry sectors.

The sectors of a point group whose projectors hold one atom per orbit
(``compress_projectors_for_sectors``) read the gradient columns of the representative atoms only, so the
gradient provider samples those alone (``PARSEC_SECTOR_GRADIENT_ATOMS``, default ``representatives``).
Checked here: the builders give the columns of the whole build bit for bit for any subset of the atoms (cluster,
double grid, periodic cell at Gamma and at a k-point, the driver's provider); the compression gives the same matrices
from the gradients of the representatives and from those of every atom; the device contraction refuses gradients
that do not belong to its projectors, uploads host blocks in capped chunks (``PARSEC_CUPY_FORCE_UPLOAD_CAP``) and puts
the gradients of an operator on a device once per force call.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Forces.nonlocal_force import (
    build_nonlocal_projector_gradients,
    build_periodic_projector_gradients,
    double_grid_projector_gradients,
)
from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import build_nonlocal_projectors
from parsec_python.models import Atom, GridSettings, PeriodicCell, PeriodicGridSettings, SpeciesPotential

from ..Symmetry.axis_reflection import (
    atom_image_table,
    compress_gradients_for_sectors,
    compress_projectors_for_sectors,
    sector_gradient_atoms,
    sector_gradient_atoms_setting,
)
from ..V_ion.double_grid import build_double_grid_projector_gradients, build_projector_gradients
from ..backends.cupy import cupy_available

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_SIDE = 10.26
# The conventional silicon cube (8 atoms, Td about the atom at the origin), the four neighbours of the origin moved
# 0.04 bohr outward along their bonds as in test_periodic_point_group.displaced_silicon.
_SILICON = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
)


def _silicon_atoms() -> list[Atom]:
    positions = _SILICON * _SIDE
    atoms = []
    for position in positions:
        bond = position - _SIDE * np.round(position / _SIDE)
        if 0.0 < np.linalg.norm(bond) < 0.45 * _SIDE:
            position = position + 0.04 * bond / np.linalg.norm(bond)
        atoms.append(Atom("Si", tuple(position)))
    return atoms


def _tetrahedral_operations() -> list[np.ndarray]:
    """The 24 signed permutations of Td about the origin (an even number of minus signs)."""

    from itertools import permutations, product

    operations = []
    for order in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            if np.prod(signs) != 1:
                continue
            matrix = np.zeros((3, 3), dtype=np.int64)
            for row, (column, sign) in enumerate(zip(order, signs)):
                matrix[row, column] = sign
            operations.append(matrix)
    return operations


def _orbits(atoms, grid):
    images = atom_image_table(atoms, _tetrahedral_operations(), None, grid, 1.0e-7)
    assert images is not None
    ordered = np.sort(images, axis=0)
    sizes = 1 + np.count_nonzero(np.diff(ordered, axis=0), axis=0)
    return SimpleNamespace(atom_representatives=images.min(axis=0), atom_orbit_sizes=sizes.astype(np.int64))


def _assert_same_columns(test, whole, part) -> None:
    """``part`` holds the columns of ``whole`` with its labels, bit for bit (values, rows and their order)."""

    position = {label: column for column, label in enumerate(whole.labels)}
    columns = np.asarray([position[label] for label in part.labels], dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(whole.signs)[columns], part.signs)
    for left, right in zip(whole.components, part.components):
        selected = left[:, columns]
        test.assertEqual(selected.shape, right.shape)
        np.testing.assert_array_equal(selected.indptr, right.indptr)
        np.testing.assert_array_equal(selected.indices, right.indices)
        np.testing.assert_array_equal(selected.data, right.data)


def _assert_same_gradients(test, left, right) -> None:
    test.assertEqual(tuple(left.labels), tuple(right.labels))
    np.testing.assert_array_equal(left.signs, right.signs)
    for a, b in zip(left.components, right.components):
        test.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.data, b.data)


class SubsetBuildTests(unittest.TestCase):
    """``atom_indices`` of the gradient builders: the columns of the whole build, bit for bit."""

    @classmethod
    def setUpClass(cls) -> None:
        path = DATA / "Si_POTRE.DAT"
        cls.potentials = {"Si": read_parsec_pseudopotential(path)}
        cls.species = {"Si": SpeciesPotential(path, 2, use_spline=True)}
        cls.lattice = _SIDE * np.eye(3)
        cls.cell_grid = build_periodic_grid(
            PeriodicCell(cls.lattice), PeriodicGridSettings(spacing=0.57, expansion_order=6)
        )
        cls.atoms = _silicon_atoms()

    def test_periodic_subsets_at_gamma_and_at_a_k_point(self) -> None:
        reciprocal = 2.0 * np.pi * np.linalg.inv(self.lattice).T
        for k_point in (None, 0.3 * reciprocal[0] - 0.2 * reciprocal[2]):
            whole = build_periodic_projector_gradients(
                self.cell_grid, self.atoms, self.potentials, self.species, self.lattice, k_point
            )
            for subset in ([0, 1, 5], [7, 2], list(range(8))):
                with self.subTest(k=k_point is not None, atoms=subset):
                    part = build_periodic_projector_gradients(
                        self.cell_grid, self.atoms, self.potentials, self.species, self.lattice, k_point,
                        atom_indices=subset,
                    )
                    self.assertEqual(sorted({label[0] for label in part.labels}), sorted(subset))
                    # in the order given
                    self.assertEqual([label[0] for label in part.labels][0], subset[0])
                    _assert_same_columns(self, whole, part)

    def test_cluster_subsets_with_and_without_the_double_grid(self) -> None:
        path = DATA / "Ge_POTRE.DAT"
        potentials = {"Ge": read_parsec_pseudopotential(path)}
        species = {"Ge": SpeciesPotential(path, 2)}
        bond = 4.9 / np.sqrt(3.0)
        atoms = [Atom("Ge", (0.0, 0.0, 0.0))] + [
            Atom("Ge", (bond * x, bond * y, bond * z))
            for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ]
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6))
        double = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6, double_grid_order=2))
        subset = [0, 3]
        builders = (
            ("readable", grid, build_nonlocal_projector_gradients, {}),
            ("accelerated", grid, build_projector_gradients, {}),
            ("readable double grid", double, double_grid_projector_gradients, {"order": 2}),
            ("accelerated double grid", double, build_double_grid_projector_gradients, {}),
            ("accelerated, double grid by the grid", double, build_projector_gradients, {}),
        )
        for name, on, builder, options in builders:
            with self.subTest(builder=name):
                whole = builder(on, atoms, potentials, species, **options)
                part = builder(on, atoms, potentials, species, atom_indices=subset, **options)
                self.assertEqual(sorted({label[0] for label in part.labels}), subset)
                self.assertGreater(len(whole.labels), len(part.labels))
                _assert_same_columns(self, whole, part)
                every = builder(on, atoms, potentials, species, atom_indices=range(len(atoms)), **options)
                _assert_same_gradients(self, whole, every)

    def test_bad_atom_indices_are_refused(self) -> None:
        for indices in ([8], [-1], [1, 1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                build_periodic_projector_gradients(
                    self.cell_grid, self.atoms, self.potentials, self.species, self.lattice, atom_indices=indices
                )


class CompressionTests(unittest.TestCase):
    """The compressed gradients of the representatives alone equal those of every atom."""

    @classmethod
    def setUpClass(cls) -> None:
        path = DATA / "Si_POTRE.DAT"
        cls.potentials = {"Si": read_parsec_pseudopotential(path)}
        cls.species = {"Si": SpeciesPotential(path, 2, use_spline=True)}
        cls.lattice = _SIDE * np.eye(3)
        cls.grid = build_periodic_grid(PeriodicCell(cls.lattice), PeriodicGridSettings(spacing=0.57, expansion_order=6))
        cls.atoms = _silicon_atoms()
        cls.reduction = _orbits(cls.atoms, cls.grid)

    def _gradients(self, atom_indices=None):
        return build_periodic_projector_gradients(
            self.grid, self.atoms, self.potentials, self.species, self.lattice, atom_indices=atom_indices
        )

    def test_the_orbits_of_the_displaced_cell(self) -> None:
        # the origin alone, its four neighbours, the three face centres
        self.assertEqual(sorted(self.reduction.atom_orbit_sizes.tolist()), [1, 3, 3, 3, 4, 4, 4, 4])

    def test_representatives_alone_give_the_compression_of_every_atom(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_GRADIENT_ATOMS": "", "PARSEC_SECTOR_PROJECTOR_ORBITS": ""}):
            atoms = sector_gradient_atoms(self.reduction)
            self.assertIsNotNone(atoms)
            np.testing.assert_array_equal(atoms, np.flatnonzero(self.reduction.atom_representatives == np.arange(8)))
            self.assertEqual(atoms.size, 3)
            whole = compress_gradients_for_sectors(self._gradients(), self.reduction)
            alone = compress_gradients_for_sectors(self._gradients(atoms), self.reduction)
            _assert_same_gradients(self, whole, alone)
            # the columns of the sector projectors, scaled by the square roots of the orbit sizes
            operator = build_nonlocal_projectors(self.grid, self.atoms, self.potentials, self.species, self.lattice)
            compressed = compress_projectors_for_sectors(operator, self.reduction)
            self.assertEqual(tuple(compressed.labels), tuple(alone.labels))
            self.assertLess(len(alone.labels), len(operator.labels))

    def test_unscaled_where_every_atom_with_projectors_is_its_own_orbit(self) -> None:
        own = [int(atom) for atom in np.flatnonzero(self.reduction.atom_orbit_sizes == 1)]
        gradients = self._gradients(own)
        self.assertIs(compress_gradients_for_sectors(gradients, self.reduction), gradients)

    def test_every_atom_where_the_sectors_read_every_column(self) -> None:
        single = SimpleNamespace(atom_representatives=np.arange(8), atom_orbit_sizes=np.ones(8, dtype=np.int64))
        self.assertIsNone(sector_gradient_atoms(single))
        self.assertIsNone(sector_gradient_atoms(SimpleNamespace()))  # an abelian reduction knows no orbits
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "off"}):
            self.assertIsNone(sector_gradient_atoms(self.reduction))
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_GRADIENT_ATOMS": "all"}):
            self.assertEqual(sector_gradient_atoms_setting(), "all")
            self.assertIsNone(sector_gradient_atoms(self.reduction))
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_GRADIENT_ATOMS": "some"}), self.assertRaises(ValueError):
            sector_gradient_atoms(self.reduction)


class ProviderTests(unittest.TestCase):
    """The driver's gradient provider of a prepared cell samples the atoms it is given."""

    def test_the_provider_of_a_prepared_cell(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon

        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY": "off"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(displaced_silicon(), backend="scipy")
        whole = system.projector_gradients()
        self.assertEqual(tuple(whole.labels), tuple(system.nonlocal_operator.labels))
        for subset in ([0, 1, 5], [6, 2]):
            with self.subTest(atoms=subset):
                part = system.projector_gradients(atom_indices=subset)
                # the projector columns of these atoms, in the order of the projectors
                self.assertEqual(
                    tuple(part.labels),
                    tuple(label for label in system.nonlocal_operator.labels if label[0] in subset),
                )
                _assert_same_columns(self, whole, part)


def _fake_operator(cp, projectors, signs):
    device_projectors = cp.asarray(projectors.toarray())
    device_signs = cp.asarray(signs)
    calls = []

    def signed(block):
        calls.append(int(block.shape[1]))
        return (device_projectors.T @ block) * device_signs[:, None]

    return SimpleNamespace(
        effective_potential=cp.zeros(1),
        _signed_nonlocal_coefficients_or_none=signed,
        calls=calls,
    )


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceContractionTests(unittest.TestCase):
    """The checks, the host chunks and the device copies of the nonlocal force contraction."""

    @classmethod
    def setUpClass(cls) -> None:
        path = DATA / "Si_POTRE.DAT"
        potentials = {"Si": read_parsec_pseudopotential(path)}
        species = {"Si": SpeciesPotential(path, 2, use_spline=True)}
        lattice = _SIDE * np.eye(3)
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.95, expansion_order=4))
        atoms = _silicon_atoms()
        cls.operator = build_nonlocal_projectors(grid, atoms, potentials, species, lattice)
        cls.gradients = build_periodic_projector_gradients(grid, atoms, potentials, species, lattice)
        generator = np.random.default_rng(5)
        cls.block = np.asfortranarray(generator.normal(size=(grid.size, 9)))
        cls.weights = np.linspace(1.0, 0.2, 9)

    def test_gradients_of_other_projectors_are_refused_with_both_counts(self) -> None:
        import cupy as cp

        from ..Forces.nonlocal_device import gradient_transposes, projector_force_terms

        # the projectors of one atom fewer than the gradients
        columns = np.arange(len(self.operator.labels) - 6)
        operator = _fake_operator(cp, self.operator.projectors[:, columns], self.operator.signs[columns])
        transposes = gradient_transposes(self.gradients.components)
        with self.assertRaises(ValueError) as caught:
            projector_force_terms(operator, cp.asarray(self.block), self.weights, transposes)
        message = str(caught.exception)
        self.assertIn(str(columns.size), message)
        self.assertIn(str(len(self.gradients.labels)), message)

    def test_host_blocks_are_uploaded_in_capped_chunks(self) -> None:
        import cupy as cp

        from ..Forces import nonlocal_device
        from ..Forces.nonlocal_device import gradient_transposes, projector_force_terms

        operator = _fake_operator(cp, self.operator.projectors, self.operator.signs)
        transposes = gradient_transposes(self.gradients.components)
        on_device = projector_force_terms(operator, cp.asarray(self.block), self.weights, transposes)
        self.assertEqual(operator.calls, [9])
        operator.calls.clear()
        rows = self.block.shape[0]
        with mock.patch.object(nonlocal_device, "_HOST_UPLOAD_BYTES", 8 * rows * 2):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_FORCE_UPLOAD_CAP": "off"}):
                uncapped = projector_force_terms(operator, self.block, self.weights, transposes)
            self.assertEqual(operator.calls, [9])
            operator.calls.clear()
            capped = projector_force_terms(operator, self.block, self.weights, transposes)
            self.assertEqual(operator.calls, [2, 2, 2, 2, 1])
        np.testing.assert_array_equal(uncapped, on_device)
        np.testing.assert_allclose(capped, uncapped, rtol=1e-13, atol=1e-15)
        self.assertGreater(float(np.abs(capped).max()), 1e-6)

    def test_the_gradients_go_to_a_device_once_per_force_call(self) -> None:
        import cupy as cp

        from ..Forces import nonlocal_device
        from ..Forces.nonlocal_device import gradient_transposes, projector_force_terms, share_device_gradients

        operator = _fake_operator(cp, self.operator.projectors, self.operator.signs)
        block = cp.asarray(self.block)
        made = []
        original = nonlocal_device.DeviceProjectorGradients.__init__

        def counting(instance, transposes):
            made.append(transposes)
            original(instance, transposes)

        first = gradient_transposes(self.gradients.components)
        second = gradient_transposes(self.gradients.components)
        reference = projector_force_terms(operator, block, self.weights, tuple(first))  # a plain tuple: made each time
        with mock.patch.object(nonlocal_device.DeviceProjectorGradients, "__init__", counting):
            self.assertIsInstance(first, tuple)
            once = [projector_force_terms(operator, block, self.weights, first) for _ in range(3)]
            self.assertEqual(len(made), 1)
            share_device_gradients([first, second])
            # one copy per device at a time: the sectors that alternate make theirs again
            for transposes in (first, first, second, second, first):
                once.append(projector_force_terms(operator, block, self.weights, transposes))
            self.assertEqual(len(made), 4)
        for terms in once:
            np.testing.assert_array_equal(terms, reference)


def _native_available() -> bool:
    from ..backends.native import native_available

    return cupy_available() and native_available()


@unittest.skipUnless(_native_available(), "CUDA runtime or native extension is unavailable")
class SectorForceModeTests(unittest.TestCase):
    """The forces of one SCF in compressed sectors with the gradients of every atom and of the representatives."""

    def _both(self, system, result):
        from parsec_python.acceleration import driver

        tables = {}
        sectors = system.backend.symmetry_eigensolver
        for mode in ("all", "representatives"):
            with mock.patch.dict(
                os.environ, {"PARSEC_SECTOR_GRADIENT_ATOMS": mode, "PARSEC_CUPY_FORCE_PREFETCH": "off"}
            ):
                tables[mode] = driver.compute_forces(system, result, symmetrize=False)
                labels = sectors.projector_force_labels
                self.assertLess(len(labels), len(system.nonlocal_operator.labels))
        self.assertGreater(float(np.abs(tables["all"].total).max()), 1.0e-3)
        np.testing.assert_allclose(
            tables["representatives"].nonlocal_pseudopotential, tables["all"].nonlocal_pseudopotential, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(tables["representatives"].total, tables["all"].total, rtol=0, atol=1e-12)

    def test_the_td_cluster(self) -> None:
        from dataclasses import replace
        import tempfile

        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_forces_geometry import _translation

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        problem = _translation("ge2h2_nlcc_tiny", Path(directory.name)).problem
        bond = 4.9 / np.sqrt(3.0)
        atoms = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(
            Atom("Ge", (bond * x, bond * y, bond * z)) for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        )
        problem = replace(
            problem,
            atoms=atoms,
            recenter_geometry=False,
            grid=replace(problem.grid, radius=9.0),
            scf=replace(problem.scf, number_of_states=16),
        )
        system = driver.prepare_single_point(problem, symmetry="on")
        system.materialize_final_wavefunctions = False
        self._both(system, driver.run_scf(system))

    def test_the_displaced_silicon_cell(self) -> None:
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon

        with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SYMMETRY": "auto"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = driver.prepare_single_point(displaced_silicon(), backend="cupy")
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system)
            self._both(system, result)


if __name__ == "__main__":
    unittest.main()
