"""The k-points of a periodic SCF in the sectors of their little groups (``acceleration/SCF/kpoint_sectors.py``).

The union of the spectra of the sectors of a k-point is its spectrum (both kinetic forms), a sector's orbitals and
density on the whole grid are those of ``U v``, and the SCF of a mesh gives the energy of the solver without
sectors: a mesh of time-reversal invariant k-points, a mesh reduced by time reversal, and the reduced mesh of
diamond whose density is symmetrized over its space group (the benchmark's setting); with too few states in some
sectors the solver adds states and solves the step again.  On the device the complex CSR kernel
(``acceleration/Hamiltonian/complex_csr_device.py``) gives the products of SciPy and the operator of a sector
that of the host.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the k-point solver)
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian
from parsec_python.SCF.kpoint_solver import KPointJob, SerialKPointSolver
from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors
from parsec_python.acceleration.SCF import kpoint_sectors as sectors
from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False

_ROOT = Path(__file__).resolve().parents[4]
_DIAMOND = _ROOT / "examples" / "3d_diamond_benchmark"


def _silicon_system():
    from parsec_python.SCF.pbc import prepare_periodic_single_point
    from parsec_python.acceleration.tests.test_periodic_point_group import silicon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = silicon("pbe", spacing=0.855)
        return problem, prepare_periodic_single_point(problem)


def _decomposition(cell, k_point):
    members = sectors.generated_subgroup(cell.table, cell.little_group(k_point, lambda index: True))
    elements, generators = sectors.largest_commuting_involutions(cell.table, members)
    reduction = sectors.little_group_reduction(cell, elements, generators)
    return members, elements, ReflectionRepresentationDecomposition.build(cell.grid, reduction)


def _dense(hamiltonian) -> np.ndarray:
    matrix = hamiltonian.apply(np.eye(hamiltonian.shape[0], dtype=np.complex128))
    return 0.5 * (matrix + matrix.conj().T)


def setUpModule() -> None:
    # the cells of these checks are small: split them all the same (the floor of the basis is checked below)
    patcher = mock.patch.object(sectors, "_MINIMUM_ENTRIES", 0)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


class SettingTests(unittest.TestCase):
    def test_the_switches(self) -> None:
        for value, expected in (("", "auto"), ("OFF", "off"), ("auto", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": value}):
                self.assertEqual(sectors.kpoint_sectors_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "on"}):
            with self.assertRaises(ValueError):
                sectors.kpoint_sectors_setting()
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_MARGIN": ""}):
            self.assertEqual(sectors.sector_margin(), 1.3)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTOR_MARGIN": "0.9"}):
            with self.assertRaises(ValueError):
                sectors.sector_margin()


class LittleGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem, cls.system = _silicon_system()
        cls.cell = sectors.CellGroup(cls.system.grid, cls.system.cell_atoms)
        cls.side = float(np.asarray(cls.problem.periodic_cell.lattice_vectors)[0, 0])

    def test_the_groups_of_the_silicon_cube(self) -> None:
        # Fd-3m modulo the cube: the face-centring translations join the point group (192 operations)
        self.assertEqual(self.cell.order, 192)
        table = self.cell.table
        for k_point, little, largest in (
            (np.zeros(3), 192, 16),
            (np.pi / self.side * np.array([1.0, 0.0, 0.0]), 32, 8),
            (np.array([0.11, 0.07, 0.05]), 4, 4),  # the face-centring translations alone
        ):
            members, elements, decomposition = _decomposition(self.cell, k_point)
            self.assertEqual((len(members), len(elements)), (little, largest), k_point)
            for index in elements:
                np.testing.assert_allclose(self.cell.matrices[index] @ k_point, k_point, atol=1.0e-12)
                self.assertEqual(int(table[index, index]), 0)
                for other in elements:
                    self.assertEqual(int(table[index, other]), int(table[other, index]))
            self.assertEqual(sum(decomposition.sector_sizes), self.system.grid.size)

    def test_the_orbits_take_one_map_at_a_time(self) -> None:
        # little_group_reduction keeps the smallest image of every row, map by map (the 16 maps of the Gamma point of
        # a 20^3 diamond cell held together were 6 GB): the orbits are those of all the maps at once, and no more
        # than two maps are alive at any time (the running minimum and the map of the element at hand).
        import gc
        import weakref

        from parsec_python.acceleration.Symmetry.axis_reflection import _orbit_labels

        members = sectors.generated_subgroup(self.cell.table, self.cell.little_group(np.zeros(3), lambda index: True))
        elements, generators = sectors.largest_commuting_involutions(self.cell.table, members)
        self.assertEqual(len(elements), 16)
        expected = _orbit_labels([self.cell.row_map(index) for index in elements])
        alive, most = [], [0]
        original = sectors.CellGroup.row_map

        def counted(cell, index, rows=None, table=None):
            gc.collect()
            most[0] = max(most[0], sum(reference() is not None for reference in alive) + 1)
            mapping = original(cell, index, rows, table)
            alive.append(weakref.ref(mapping))
            return mapping

        with mock.patch.object(sectors.CellGroup, "row_map", counted):
            reduction = sectors.little_group_reduction(self.cell, elements, generators)
        self.assertEqual(len(alive), 16)
        self.assertLessEqual(most[0], 2)
        for got, want in zip(
            (reduction.representative_rows, reduction.full_to_wedge, reduction.multiplicities), expected
        ):
            np.testing.assert_array_equal(got, want)

    def test_products_of_admitted_operations(self) -> None:
        # the subgroup that the operations about the origin (Td) and the translations generate holds their
        # products (the 96 operations of Td times the face-centring translations); one operation with a quarter
        # translation (non-symmorphic) more gives the whole group
        admitted = [index for index, shift in enumerate(self.cell.shifts) if not np.any(shift)]
        self.assertEqual(len(admitted), 24)
        translations = [
            index for index, matrix in enumerate(self.cell.matrices) if np.array_equal(matrix, np.eye(3, dtype=np.int8))
        ]
        self.assertEqual(len(sectors.generated_subgroup(self.cell.table, admitted + translations)), 96)
        quarter = next(
            index for index, shift in enumerate(self.cell.shifts)
            if np.any(np.mod(shift, self.cell.counts // 2) != 0)
        )
        self.assertEqual(len(sectors.generated_subgroup(self.cell.table, admitted + [quarter])), 192)
        self.assertEqual(sectors.generated_subgroup(self.cell.table, []), [0])

    def test_invariance_of_the_potential(self) -> None:
        # a full mesh is mapped onto itself by every operation; a single k-point off the axes is not
        k_mesh = np.array([np.array(bits, dtype=float) * np.pi / self.side for bits in np.ndindex(2, 2, 2)])
        lattice = np.asarray(self.problem.periodic_cell.lattice_vectors, dtype=float)
        full = sectors.PotentialInvariance(self.cell, None, k_mesh, np.full(8, 0.125), lattice)
        self.assertTrue(all(full(index) for index in range(1, self.cell.order)))
        lone = sectors.PotentialInvariance(self.cell, None, np.array([[0.11, 0.07, 0.05]]), np.ones(1), lattice)
        swaps = [index for index, matrix in enumerate(self.cell.matrices) if matrix[0, 1] != 0 and matrix[2, 2] == 1]
        self.assertFalse(lone(swaps[0]))
        pure = [index for index, matrix in enumerate(self.cell.matrices) if index and np.array_equal(matrix, np.eye(3))]
        self.assertTrue(all(lone(index) for index in pure))


class SectorOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem, cls.system = _silicon_system()
        cls.cell = sectors.CellGroup(cls.system.grid, cls.system.cell_atoms)
        cls.side = float(np.asarray(cls.problem.periodic_cell.lattice_vectors)[0, 0])

    def _hamiltonian(self, k_point, kinetic: str):
        system, problem = self.system, self.problem
        lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
        projectors = build_nonlocal_projectors(
            system.grid, system.cell_atoms, system.pseudopotentials, problem.pseudopotentials, lattice,
            k_point=k_point, periodic_dimensions=problem.periodic_cell.dimensions,
        )
        bloch = build_bloch_negative_laplacian(system.grid, k_point, "u") if kinetic == "bloch" else None
        return KPointKohnShamHamiltonian(
            system.negative_laplacian, system.gradient, system.ionic_potential, projectors, k_point,
            bloch_laplacian=bloch,
        )

    def test_the_sector_spectra_are_the_spectrum_of_the_k_point(self) -> None:
        for k_point in (np.pi / self.side * np.array([1.0, 0.0, 0.0]), np.array([0.11, 0.07, 0.05])):
            _, _, decomposition = _decomposition(self.cell, k_point)
            for kinetic in ("gradient", "bloch"):
                full = self._hamiltonian(k_point, kinetic)
                expected = np.linalg.eigvalsh(_dense(full))[:30]
                split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
                levels = np.sort(np.concatenate([
                    np.linalg.eigvalsh(_dense(split.hamiltonian(sector, full)))
                    for sector in range(decomposition.representation_count)
                ]))[:30]
                np.testing.assert_allclose(levels, expected, rtol=0, atol=1.0e-11, err_msg=f"{k_point} {kinetic}")

    def test_orbitals_and_density_of_a_sector(self) -> None:
        k_point = np.array([0.11, 0.07, 0.05])
        _, _, decomposition = _decomposition(self.cell, k_point)
        full = self._hamiltonian(k_point, "gradient")
        split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
        sector = decomposition.representation_count - 1
        values, vectors = np.linalg.eigh(_dense(split.hamiltonian(sector, full)))
        orbitals = split.orbitals(sector, vectors[:, :3])
        residual = full.apply(orbitals) - orbitals * values[:3]
        self.assertLess(float(np.max(np.abs(residual))), 1.0e-10)
        np.testing.assert_allclose(np.linalg.norm(orbitals, axis=0), 1.0, atol=1.0e-12)
        occupations = np.array([2.0, 1.0, 0.5])
        density = split.density(sector, (np.abs(vectors[:, :3]) ** 2) @ occupations)
        np.testing.assert_allclose(density, (np.abs(orbitals) ** 2) @ occupations, rtol=0, atol=1.0e-15)


class SectorSCFTests(unittest.TestCase):
    def assert_same_scf(self, problem, k_points=None, weights=None, *, delta=1.0e-9, environment=None):
        from parsec_python.Grid.kpoints import kpoint_sampling
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        results = {}
        for setting in ("off", "auto"):
            switches = {"PARSEC_KPOINT_SECTORS": setting, **(environment or {})}
            with mock.patch.dict(os.environ, switches), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
                points, shares = (k_points, weights) if k_points is not None else kpoint_sampling(problem)
                results[setting] = run_scf_kpoints(system, points, shares)
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=delta)
        self.assertNotIn("kpoint_sector_tasks", off.timings.detail)
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], len(points))
        return auto

    def _silicon(self):
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
        return replace(
            base,
            scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=120),
            eigensolver=replace(base.eigensolver, tolerance=1.0e-10),
        )

    def test_time_reversal_invariant_mesh(self) -> None:
        base = self._silicon()
        side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.array(bits, dtype=float) * np.pi / side for bits in np.ndindex(2, 2, 2)])
        auto = self.assert_same_scf(base, k_points, np.full(8, 0.125))
        self.assertEqual(auto.timings.detail["kpoint_sector_largest_group"], 16.0)

    def test_mesh_reduced_by_time_reversal(self) -> None:
        base = self._silicon()
        self.assert_same_scf(replace(base, monkhorst_pack_dimensions=(3, 3, 3), monkhorst_pack_shift=(0.0, 0.0, 0.0)))

    def test_too_few_states_in_a_sector(self) -> None:
        # without the six extra states and margin some sectors miss levels: they get more and the step is solved
        # again
        base = self._silicon()
        side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])])
        with mock.patch.object(sectors, "_EXTRA_STATES", 0):
            auto = self.assert_same_scf(
                base, k_points, np.array([0.5, 0.5]), environment={"PARSEC_KPOINT_SECTOR_MARGIN": "1.0"}
            )
        self.assertGreaterEqual(auto.timings.detail["kpoint_sector_regrowths"], 1.0)
        # the grown sectors were solved again alone: the solver and the Ritz states of the others were kept
        self.assertEqual(auto.timings.detail["kpoint_sector_rebuilds"], 0.0)
        self.assertGreaterEqual(
            auto.timings.detail["kpoint_sector_regrown_tasks"], auto.timings.detail["kpoint_sector_regrowths"]
        )

    @unittest.skipUnless((_DIAMOND / "make_diamond.py").exists(), "the diamond example is not in this checkout")
    def test_reduced_mesh_with_the_symmetrized_density(self) -> None:
        # the benchmark's input: the default routines, the 3 x 3 x 3 mesh reduced by Td about an atom and time
        # reversal, the density symmetrized over the space group (Td): its operations and the face-centring
        # translations generate the group of the sectors
        from parsec_python.Input.parsec_input import parse_parsec_input

        directory = Path(tempfile.mkdtemp(prefix="parsec_ksec_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, True))
        subprocess.run(
            [sys.executable, str(_DIAMOND / "make_diamond.py"), "1", "1", "1", str(directory / "parsec.in"),
             "--kmesh", "3", "3", "3"],
            check=True, capture_output=True,
        )
        (directory / "C_POTRE.DAT").write_bytes((_DIAMOND / "C_POTRE.DAT").read_bytes())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = parse_parsec_input(directory / "parsec.in").problem
        problem = replace(
            problem,
            scf=replace(problem.scf, convergence_criterion=1.0e-9),
            eigensolver=replace(problem.eigensolver, tolerance=1.0e-10),
        )
        self.assertTrue(problem.scf.symmetrize_density)
        auto = self.assert_same_scf(problem)
        self.assertEqual(auto.timings.detail["kpoint_sector_largest_group"], 16.0)


class SectorDriverTests(unittest.TestCase):
    """The sectors in the other drivers of k-points: both spin channels, the forces from the orbitals of the sectors,
    the exchange of a hybrid functional reduced to the sectors and the states carried from loop to loop."""

    def _silicon(self, **scf):
        base = SectorSCFTests._silicon(self)
        return replace(base, scf=replace(base.scf, **scf)) if scf else base

    def _gamma_and_x(self, problem):
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        return np.array([np.zeros(3), np.pi / side * np.array([1.0, 0.0, 0.0])]), np.array([0.5, 0.5])

    def _both(self, run, problem):
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        results = {}
        for setting in ("off", "auto"):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": setting}), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[setting] = run(prepare_periodic_single_point(problem))
        off, auto = results["off"], results["auto"]
        self.assertTrue(off.converged and auto.converged)
        self.assertNotIn("kpoint_sector_tasks", off.timings.detail)
        return off, auto

    def test_spin_channels_share_the_sectors_of_their_k_point(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized

        problem = self._silicon(spin_polarized=True)
        k_points, weights = self._gamma_and_x(problem)
        off, auto = self._both(lambda system: run_scf_kpoints_spin_polarized(system, k_points, weights), problem)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-9)
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], 4.0)
        np.testing.assert_allclose(auto.eigenvalues_up, off.eigenvalues_up, atol=1.0e-8)
        np.testing.assert_allclose(auto.eigenvalues_down, off.eigenvalues_down, atol=1.0e-8)

    def test_forces_from_the_orbitals_of_the_sectors(self) -> None:
        # one atom moved along x: C2x and the two mirrors through x keep the cell, the sectors of Gamma and X
        from parsec_python.SCF.kpoints import run_scf_kpoints

        problem = self._silicon()
        atoms = list(problem.atoms)
        atoms[0] = replace(atoms[0], position=np.asarray(atoms[0].position, dtype=float) + np.array([0.1, 0.0, 0.0]))
        problem = replace(problem, atoms=atoms)
        k_points, weights = self._gamma_and_x(problem)
        off, auto = self._both(lambda system: run_scf_kpoints(system, k_points, weights, compute_forces=True), problem)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=1.0e-9)
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], 2.0)
        self.assertGreater(float(np.max(np.abs(off.nonlocal_forces))), 1.0e-3)
        np.testing.assert_allclose(auto.nonlocal_forces, off.nonlocal_forces, rtol=0.0, atol=1.0e-7)

    def test_hybrid_exchange_reduced_to_the_sectors(self) -> None:
        # PBE0 at the eight time-reversal invariant k-points: the Fock loop carries the states of the sectors from
        # one SCF loop to the next and builds its operator from their orbitals on the whole grid
        from parsec_python.models import ExactExchangeSettings
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf

        problem = self._silicon(
            xc_functional="pbe0", exact_exchange=ExactExchangeSettings(tolerance=1.0e-8), convergence_criterion=1.0e-8
        )
        side = float(np.asarray(problem.periodic_cell.lattice_vectors)[0, 0])
        k_points = np.array([np.array(bits, dtype=float) * np.pi / side for bits in np.ndindex(2, 2, 2)])
        weights = np.full(8, 0.125)
        off, auto = self._both(lambda system: run_periodic_hybrid_scf(system, k_points, weights), problem)
        self.assertAlmostEqual(auto.energies.total, off.energies.total, delta=2.0e-8)
        # The estimate the Fock loop stops on is an energy, second order in the orbitals: E_x, first order, is
        # converged to about its square root, and the two loops take other paths (without sectors, on the host, E_x
        # still moved by 3.8e-6 Ry between the outer steps with estimates 2e-10 and 4e-12; stopped at 1e-10 the
        # two runs were 7.7e-7 Ry apart in E_x and 2e-11 Ry in the total energy; on A100 at 1e-8, 5e-8 to 1.6e-7)
        self.assertAlmostEqual(
            auto.exact_exchange.exchange_energy, off.exact_exchange.exchange_energy, delta=1.0e-6
        )
        self.assertGreater(auto.timings.detail["kpoint_sector_tasks"], 8.0)

    def test_the_loops_of_one_structure_share_the_plans(self) -> None:
        # a caller's cache (the projector cache of the Fock loop): the next loop takes the plans and their static
        # operators; another layout of the job (other k-points) makes its own
        problem, system = _silicon_system()
        cache: dict = {}
        geometry = {
            "grid": system.grid,
            "atoms": tuple(system.cell_atoms),
            "k_points": np.zeros((1, 3)),
            "weights": np.ones(1),
            "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
            "density_symmetry": None,
            "cache": cache,
        }
        job = KPointJob(1, 22, lambda *args: None, lambda *args: None, lambda *args, **kw: {}, geometry=geometry)
        first = sectors.plan_sectors(job)
        self.assertIsNotNone(first)
        self.assertIs(sectors.plan_sectors(job)[0], first[0])
        moved = dict(geometry, k_points=np.array([[0.1, 0.0, 0.0]]))
        other = sectors.plan_sectors(KPointJob(1, 22, None, None, None, geometry=moved))
        self.assertIsNot(other[0], first[0])

    def test_the_reduction_is_the_adjoint_of_the_expansion(self) -> None:
        problem, system = _silicon_system()
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        _, _, decomposition = _decomposition(cell, np.zeros(3))
        split = sectors.KPointSectors(0, np.zeros(3), decomposition, [4] * decomposition.representation_count)
        rng = np.random.default_rng(7)
        whole = rng.normal(size=(system.grid.size, 3)) + 1j * rng.normal(size=(system.grid.size, 3))
        for sector in range(decomposition.representation_count):
            block = rng.normal(size=(split.sizes[sector], 3)) + 1j * rng.normal(size=(split.sizes[sector], 3))
            # <U v, x> = <v, U^H x>
            left = np.vdot(split.orbitals(sector, block), whole)
            right = np.vdot(block, split.reduce(sector, whole))
            self.assertAlmostEqual(left, right, delta=1.0e-10 * abs(left))
            np.testing.assert_allclose(split.reduce(sector, split.orbitals(sector, block)), block, atol=1.0e-12)

    def test_states_of_other_sectors_are_not_taken(self) -> None:
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        problem = self._silicon()
        k_points, weights = self._gamma_and_x(problem)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "auto"}), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            system = prepare_periodic_single_point(problem)
            result = run_scf_kpoints(system, k_points, weights, keep_states=True)
        carried = result.eigensolver_state
        self.assertIn(sectors._STATES_KEY, carried)
        layout = carried[sectors._STATES_KEY]["layout"]
        self.assertGreater(len(layout), 2)

        class Inner:
            imported = None

            def import_states(self, states):
                Inner.imported = states

        solver = object.__new__(sectors.SectorKPointSolver)
        solver.__dict__.update(_items=[(k, None if s < 0 else s) for k, s, _ in layout], _inner=Inner())
        solver._layout = lambda: [tuple(item) for item in layout]
        solver.import_states({0: "a state of a solver without sectors"})
        self.assertIsNone(Inner.imported)
        solver.import_states({sectors._STATES_KEY: dict(carried[sectors._STATES_KEY], layout=layout[:-1])})
        self.assertIsNone(Inner.imported)
        solver._states = lambda item: dict(zip([(k, None if s < 0 else s) for k, s, _ in layout],
                                                carried[sectors._STATES_KEY]["counts"]))[item]
        solver.import_states(carried)
        self.assertIs(Inner.imported, carried[sectors._STATES_KEY]["states"])


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceSectorTests(unittest.TestCase):
    def test_the_complex_csr_kernel(self) -> None:
        import cupy as cp
        import scipy.sparse as sp

        from parsec_python.acceleration.Hamiltonian.complex_csr_device import DeviceComplexCSR

        rng = np.random.default_rng(3)
        # rows of a stencil's length (with a diagonal and accumulated), long rows (the projectors' adjoint), short
        for rows, columns, density, count in ((4000, 4000, 49 / 4000, 7), (300, 20000, 0.1, 13), (20000, 300, 0.01, 1)):
            matrix = (
                sp.random(rows, columns, density=density, format="csr", random_state=1)
                + 1j * sp.random(rows, columns, density=density, format="csr", random_state=2)
            ).tocsr()
            block = rng.normal(size=(columns, count)) + 1j * rng.normal(size=(columns, count))
            device = DeviceComplexCSR(cp, matrix)
            expected = matrix @ block
            scale = float(np.max(np.abs(expected)))
            np.testing.assert_allclose(cp.asnumpy(device.product(cp.asarray(block))), expected, rtol=0, atol=1.0e-14 * scale)
            if rows == columns:
                diagonal = rng.normal(size=rows)
                got = device.product(cp.asarray(block), diagonal=cp.asarray(diagonal))
                np.testing.assert_allclose(
                    cp.asnumpy(got), expected + diagonal[:, None] * block, rtol=0, atol=1.0e-14 * scale
                )
                target = cp.asarray(block)
                device.product(cp.asarray(block), target, accumulate=True)
                np.testing.assert_allclose(cp.asnumpy(target), block + expected, rtol=0, atol=1.0e-14 * scale)
            else:
                self.assertEqual(device.long_rows, rows < columns)

    def test_the_reduction_of_a_device_block(self) -> None:
        import cupy as cp

        problem, system = _silicon_system()
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        _, _, decomposition = _decomposition(cell, np.array([0.11, 0.0, 0.0]))
        split = sectors.KPointSectors(0, np.zeros(3), decomposition, [4] * decomposition.representation_count)
        rng = np.random.default_rng(9)
        whole = rng.normal(size=(system.grid.size, 5)) + 1j * rng.normal(size=(system.grid.size, 5))
        for sector in range(decomposition.representation_count):
            device = split.reduce(sector, cp.asarray(whole))
            self.assertIsInstance(device, cp.ndarray)
            np.testing.assert_allclose(cp.asnumpy(device), split.reduce(sector, whole), rtol=0, atol=1.0e-12)

    def test_the_device_operator_of_a_sector(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter

        problem, system = _silicon_system()
        cell = sectors.CellGroup(system.grid, system.cell_atoms)
        helper = SectorOperatorTests()
        helper.problem, helper.system, helper.cell = problem, system, cell
        rng = np.random.default_rng(5)
        for k_point in (np.zeros(3), np.array([0.11, 0.07, 0.05])):
            _, _, decomposition = _decomposition(cell, k_point)
            for kinetic in ("gradient", "bloch"):
                full = helper._hamiltonian(k_point, kinetic)
                split = sectors.KPointSectors(0, k_point, decomposition, [8] * decomposition.representation_count)
                for fused in (False, True):  # the MPI pools ask for the stencil kernel: a sector takes the CSR one
                    converter = DeviceBlochConverter(CupyArrayBackend(0))
                    converter.fused_stencil = fused
                    for sector in range(decomposition.representation_count):
                        host = split.hamiltonian(sector, full)
                        block = rng.normal(size=(host.shape[0], 5)) + 1j * rng.normal(size=(host.shape[0], 5))
                        expected = host.apply(block)
                        operator = converter.operator(host)
                        self.assertEqual(type(operator.laplacian).__name__, "DeviceComplexCSR")
                        np.testing.assert_allclose(
                            cp.asnumpy(operator.apply(cp.asarray(block))), expected,
                            rtol=0, atol=1.0e-12 * float(np.max(np.abs(expected))),
                            err_msg=f"{k_point} {kinetic} fused={fused} sector {sector}",
                        )


class DeclineTests(unittest.TestCase):
    def test_what_takes_the_solver_of_before(self) -> None:
        problem, system = _silicon_system()
        geometry = {
            "grid": system.grid,
            "atoms": tuple(system.cell_atoms),
            "k_points": np.zeros((1, 3)),
            "weights": np.ones(1),
            "lattice": np.asarray(problem.periodic_cell.lattice_vectors, dtype=float),
            "density_symmetry": None,
        }

        def job(**changes):
            fields = dict(
                n_tasks=1, number_of_states=22, eigval_settings=lambda *args: None,
                make_hamiltonian=lambda task, potentials: None, accumulate=lambda *args, **kw: {}, geometry=geometry,
            )
            fields.update(changes)
            return KPointJob(**fields)

        def plain(inner, **options):
            return SerialKPointSolver(inner)

        self.assertIsInstance(sectors.sector_solver(job(), plain, problem_size=system.grid.size), sectors.SectorKPointSolver)
        self.assertIsNone(sectors.sector_solver(job(geometry=None), plain))
        # extras that sum over any subset of the states (the forces) take the sectors, in one process or on the
        # ranks of the MPI pools; other extras do not
        self.assertIsNotNone(
            sectors.sector_solver(
                job(extras=lambda task, vectors: 0.0, extras_by_columns=True), plain, problem_size=system.grid.size
            )
        )
        self.assertIsNone(sectors.sector_solver(job(extras=lambda task, vectors: 0.0), plain))
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_SECTORS": "off"}):
            self.assertIsNone(sectors.sector_solver(job(), plain))
        # a k-point whose sectors would not halve the work stays one task: with one state, the six extra states
        # of every sector cost more than the whole k-point
        self.assertIsNone(sectors.sector_solver(job(number_of_states=1), plain))
        # a basis of fewer entries than the floor (2^24) stays whole
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", 1 << 24):
            self.assertIsNone(sectors.sector_solver(job(), plain))
        with mock.patch.object(sectors, "_MINIMUM_ENTRIES", system.grid.size * 22):
            self.assertIsNotNone(sectors.sector_solver(job(), plain))


if __name__ == "__main__":
    unittest.main()
