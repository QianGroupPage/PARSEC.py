"""Projectors of one atom per orbit in Abelian sectors, and the weighted plans of D4h-type groups.

``attach_abelian_atom_orbits`` gives the Abelian reductions of clusters and Gamma cells the orbits of their atoms, so
that every sector takes the projector columns of one atom per orbit (``compress_projectors_for_sectors``) and the
forces contract the gradients of those atoms, completed by the average over the group
(``Forces.sector_symmetry.abelian_orbit_operations``); ``PARSEC_SECTOR_PROJECTOR_ORBITS=weighted`` keeps the
projectors of every atom there, as before.  ``PARSEC_ORBITAL_GROUP_MAX_COST`` (default 0.75) lets the plans of
D4h, D2d, D4 and C4v, three quarters of the abelian work, take the weighted sectors.

Checked on the host: the switches; the orbits of a cluster and of a cell; that the compressed projectors give every
sector the operator of every atom to round-off; that the nonlocal force terms of random sector states, contracted with
the compressed gradients and averaged over the group, are those of every atom; the plans that the cost threshold takes;
the sector counts that the ranks of an MPI run divide (a plan they cannot divide keeps the abelian group).
On a GPU: complete calculations of a D2h cluster with the compressed projectors against the full grid and against
``weighted``, and of D4h and D2d clusters in the weighted sectors against the abelian ones; the preparation of a D2d
cluster on four ranks with the GPUs of their nodes (threads), which keeps its four abelian sectors.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.Forces import symmetrize_forces
from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import build_nonlocal_projectors
from parsec_python.models import Atom, GridSettings, PeriodicCell, PeriodicGridSettings, SpeciesPotential

from ..Forces.sector_symmetry import abelian_orbit_operations, check_sector_atoms
from ..Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    attach_abelian_atom_orbits,
    compress_gradients_for_sectors,
    compress_projectors_for_sectors,
    detect_point_group_reduction,
    orbital_group_max_cost,
    projector_orbit_counts,
    sector_gradient_atoms,
    sector_projector_orbits_setting,
)
from ..Symmetry.representations import ReflectionRepresentationDecomposition
from ..V_ion.double_grid import build_projector_gradients
from ..backends.cupy import cupy_available
from ..backends.native import native_available

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

# Ge in D2h: a rectangle in the xy plane (one orbit of four), two atoms on the z axis (an orbit of two) and one at
# the origin (its own orbit).
D2H_GE = (
    Atom("Ge", (0.0, 0.0, 0.0)),
    Atom("Ge", (3.0, 2.6, 0.0)),
    Atom("Ge", (-3.0, 2.6, 0.0)),
    Atom("Ge", (3.0, -2.6, 0.0)),
    Atom("Ge", (-3.0, -2.6, 0.0)),
    Atom("Ge", (0.0, 0.0, 3.6)),
    Atom("Ge", (0.0, 0.0, -3.6)),
)
# Ge6 in D4h: a square in the xy plane and two apices on the z axis at another distance.
D4H_GE = (
    Atom("Ge", (3.3, 0.0, 0.0)),
    Atom("Ge", (-3.3, 0.0, 0.0)),
    Atom("Ge", (0.0, 3.3, 0.0)),
    Atom("Ge", (0.0, -3.3, 0.0)),
    Atom("Ge", (0.0, 0.0, 2.8)),
    Atom("Ge", (0.0, 0.0, -2.8)),
)
_SIDE = 10.26
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


def _c2v_silicon() -> list[Atom]:
    """The conventional silicon cube with the neighbours of the origin pushed out along their bonds and the atom at
    the origin moved along z: C2v about the origin (an Abelian group of four signed permutations)."""

    atoms = []
    for position in _SILICON * _SIDE:
        bond = position - _SIDE * np.round(position / _SIDE)
        if 0.0 < np.linalg.norm(bond) < 0.45 * _SIDE:
            position = position + 0.04 * bond / np.linalg.norm(bond)
        atoms.append(Atom("Si", tuple(position)))
    atoms[0] = Atom("Si", (0.0, 0.0, 0.05))
    return atoms


def _term_forces(decomposition, operator, gradients, vectors, occupations) -> np.ndarray:
    """``F[a, c] = 4 sum_s sum_n f_n sum_(q of a) s_q <beta_q|psi_n><d_c beta_q|psi_n>`` of sector states."""

    count = decomposition.representation_count
    projectors = decomposition.reduce_nonlocal_operators(operator)
    reduced = [
        decomposition.reduce_nonlocal_operators(
            SimpleNamespace(projectors=component, signs=gradients.signs, labels=gradients.labels)
        )
        for component in gradients.components
    ]
    owners = np.fromiter((int(label[0]) for label in gradients.labels), dtype=np.int64, count=len(gradients.labels))
    np.testing.assert_array_equal(
        owners, np.fromiter((int(label[0]) for label in operator.labels), dtype=np.int64, count=len(operator.labels))
    )
    signs = np.asarray(operator.signs, dtype=np.float64)
    terms = np.zeros((len(operator.labels), 3))
    for sector in range(count):
        overlaps = projectors[sector].projectors.T @ vectors[sector]
        for axis in range(3):
            slopes = reduced[axis][sector].projectors.T @ vectors[sector]
            terms[:, axis] += 4.0 * signs * np.sum(overlaps * slopes * occupations[sector][None, :], axis=1)
    forces = np.zeros((int(owners.max()) + 1, 3))
    for axis in range(3):
        forces[:, axis] = np.bincount(owners, weights=terms[:, axis], minlength=forces.shape[0])
    return forces


class SwitchTests(unittest.TestCase):
    def test_projector_orbit_values(self) -> None:
        for value in ("auto", "weighted", "off"):
            with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": value}):
                self.assertEqual(sector_projector_orbits_setting(), value)
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": ""}):
            self.assertEqual(sector_projector_orbits_setting(), "auto")
        with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "abelian"}):
            with self.assertRaises(ValueError):
                sector_projector_orbits_setting()

    def test_orbital_group_max_cost(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": ""}):
            self.assertEqual(orbital_group_max_cost(), 0.75)
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": "0.5"}):
            self.assertEqual(orbital_group_max_cost(), 0.5)
        for bad in ("0", "1.5", "-0.2", "half"):
            with self.subTest(value=bad), mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": bad}):
                with self.assertRaises(ValueError):
                    orbital_group_max_cost()


class _Setup:
    """A cluster in D2h and a silicon cell in C2v, their projectors and gradients (Ge and Si potentials)."""

    @staticmethod
    def cluster():
        path = DATA / "Ge_POTRE.DAT"
        potentials = {"Ge": read_parsec_pseudopotential(path)}
        species = {"Ge": SpeciesPotential(path, 2)}
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6))
        atoms = D2H_GE
        operator = build_nonlocal_projectors(grid, atoms, potentials, species)
        gradients = build_projector_gradients(grid, atoms, potentials, species)
        return SimpleNamespace(grid=grid, atoms=atoms, operator=operator, gradients=gradients, cell=None)

    @staticmethod
    def cell():
        from parsec_python.Forces.nonlocal_force import build_periodic_projector_gradients

        path = DATA / "Si_POTRE.DAT"
        potentials = {"Si": read_parsec_pseudopotential(path)}
        species = {"Si": SpeciesPotential(path, 2, use_spline=True)}
        lattice = _SIDE * np.eye(3)
        cell = PeriodicCell(lattice)
        grid = build_periodic_grid(cell, PeriodicGridSettings(spacing=0.57, expansion_order=6))
        atoms = _c2v_silicon()
        operator = build_nonlocal_projectors(grid, atoms, potentials, species, lattice_vectors=lattice)
        gradients = build_periodic_projector_gradients(grid, atoms, potentials, species, lattice)
        return SimpleNamespace(grid=grid, atoms=tuple(atoms), operator=operator, gradients=gradients, cell=cell)


class AbelianOrbitTests(unittest.TestCase):
    """The orbits, the sector operators and the completed forces of Abelian sectors with compressed projectors."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {"cluster": _Setup.cluster(), "cell": _Setup.cell()}

    def _reduction(self, case):
        reduction = SignedPermutationReduction.detect(case.grid, case.atoms)
        self.assertNotIsInstance(reduction, PointGroupReduction)
        return reduction

    def test_orbits_are_attached_in_auto_only(self) -> None:
        expected = {"cluster": (8, [1, 4, 4, 4, 4, 2, 2]), "cell": (4, None)}
        for name, case in self.cases.items():
            with self.subTest(case=name):
                order, sizes = expected[name]
                for setting in ("weighted", "off"):
                    reduction = self._reduction(case)
                    with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": setting}):
                        self.assertFalse(attach_abelian_atom_orbits(reduction, case.atoms, case.grid))
                    self.assertIsNone(getattr(reduction, "atom_representatives", None))
                    self.assertIs(compress_projectors_for_sectors(case.operator, reduction), case.operator)
                    self.assertIsNone(sector_gradient_atoms(reduction))
                reduction = self._reduction(case)
                self.assertEqual(reduction.group_order, order)
                with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "auto"}):
                    self.assertTrue(attach_abelian_atom_orbits(reduction, case.atoms, case.grid))
                    # once: a reduction that knows its orbits keeps them
                    self.assertFalse(attach_abelian_atom_orbits(reduction, case.atoms, case.grid))
                    representatives = np.asarray(reduction.atom_representatives)
                    found = np.asarray(reduction.atom_orbit_sizes)
                    if sizes is not None:
                        np.testing.assert_array_equal(found, sizes)
                    self.assertGreater(int(found.max()), 1)
                    # every orbit of a group divides its order; the representative is the smallest index
                    self.assertTrue(np.all(order % found == 0))
                    self.assertTrue(np.all(representatives <= np.arange(representatives.size)))
                    kept_atoms, kept_columns = projector_orbit_counts(
                        reduction, case.operator.labels, len(case.atoms)
                    )
                    self.assertEqual(kept_atoms, int(np.unique(representatives).size))
                    self.assertLess(kept_columns, len(case.operator.labels))
                    np.testing.assert_array_equal(sector_gradient_atoms(reduction), np.unique(representatives))
                    lattice = None if case.cell is None else case.cell.lattice_vectors
                    check_sector_atoms(reduction, case.atoms, case.grid, lattice)

    def test_identity_and_point_groups_take_no_abelian_orbits(self) -> None:
        from ..Symmetry.axis_reflection import AxisReflectionReduction

        case = self.cases["cluster"]
        identity = AxisReflectionReduction.identity(case.grid)
        self.assertFalse(attach_abelian_atom_orbits(identity, case.atoms, case.grid))
        self.assertFalse(attach_abelian_atom_orbits(None, case.atoms, case.grid))
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6))
        abelian = SignedPermutationReduction.detect(grid, D4H_GE)
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": "0.75"}):
            weighted = detect_point_group_reduction(grid, D4H_GE, abelian)
        self.assertIsInstance(weighted, PointGroupReduction)
        known = weighted.atom_representatives
        self.assertFalse(attach_abelian_atom_orbits(weighted, D4H_GE, grid))
        self.assertIs(weighted.atom_representatives, known)

    def test_compressed_sector_operators_and_completed_forces(self) -> None:
        generator = np.random.default_rng(20261118)
        for name, case in self.cases.items():
            with self.subTest(case=name), mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "auto"}):
                reduction = self._reduction(case)
                self.assertTrue(attach_abelian_atom_orbits(reduction, case.atoms, case.grid))
                decomposition = ReflectionRepresentationDecomposition.build(case.grid, reduction)
                compressed = compress_projectors_for_sectors(case.operator, reduction)
                self.assertLess(compressed.projectors.shape[1], case.operator.projectors.shape[1])
                whole = decomposition.reduce_nonlocal_operators(case.operator)
                part = decomposition.reduce_nonlocal_operators(compressed)
                for sector in range(decomposition.representation_count):
                    left = whole[sector].projectors
                    right = part[sector].projectors
                    every = (left @ np.diag(case.operator.signs) @ left.T.toarray()) if left.shape[1] else None
                    kept = right @ np.diag(compressed.signs) @ right.T.toarray()
                    scale = max(1.0, float(np.abs(every).max()))
                    np.testing.assert_allclose(kept, every, rtol=0, atol=1e-12 * scale, err_msg=f"sector {sector}")
                # random states of every sector, positive occupations: the density matrix of states of real
                # one-dimensional characters is invariant under the group whatever the occupations
                vectors = [
                    generator.standard_normal((decomposition.sector_size(sector), 3))
                    for sector in range(decomposition.representation_count)
                ]
                occupations = [generator.uniform(0.5, 2.0, 3) for _ in vectors]
                reference = _term_forces(decomposition, case.operator, case.gradients, vectors, occupations)
                representatives = sector_gradient_atoms(reduction)
                for gradients in (case.gradients, _columns_of(case.gradients, representatives)):
                    reduced_gradients = compress_gradients_for_sectors(gradients, reduction)
                    self.assertEqual(tuple(reduced_gradients.labels), tuple(compressed.labels))
                    raw = _term_forces(decomposition, compressed, reduced_gradients, vectors, occupations)
                    forces = np.zeros_like(reference)
                    forces[: raw.shape[0]] = raw
                    system = SimpleNamespace(
                        backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(decomposition=decomposition)),
                        input=SimpleNamespace(periodic_cell=case.cell),
                        atoms=case.atoms,
                        grid=case.grid,
                    )
                    operations = abelian_orbit_operations(system)
                    self.assertEqual(len(operations), reduction.group_order)
                    completed = symmetrize_forces(forces, operations)
                    scale = float(np.abs(reference).max())
                    self.assertGreater(scale, 1.0e-6)
                    np.testing.assert_allclose(completed, reference, rtol=0, atol=1e-11 * scale)
                with mock.patch.dict(os.environ, {"PARSEC_SECTOR_PROJECTOR_ORBITS": "off"}):
                    self.assertIsNone(abelian_orbit_operations(system))


def _columns_of(gradients, atoms):
    """The gradient columns of ``atoms`` alone, in the order of the projectors (what the provider samples)."""

    from parsec_python.Forces import NonlocalProjectorGradients

    wanted = set(int(atom) for atom in atoms)
    columns = [index for index, label in enumerate(gradients.labels) if int(label[0]) in wanted]
    return NonlocalProjectorGradients(
        components=tuple(component[:, columns] for component in gradients.components),
        signs=np.asarray(gradients.signs)[columns],
        labels=tuple(gradients.labels[index] for index in columns),
    )


class PlanThresholdTests(unittest.TestCase):
    """``PARSEC_ORBITAL_GROUP_MAX_COST``: the D4h plan (three quarters of the abelian work) in or out."""

    def test_d4h_takes_the_weighted_sectors_at_three_quarters(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6))
        abelian = SignedPermutationReduction.detect(grid, D4H_GE)
        self.assertEqual(abelian.group_order, 8)
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": "0.5"}):
            self.assertIs(detect_point_group_reduction(grid, D4H_GE, abelian), abelian)
        for value in ("", "0.75"):
            with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP_MAX_COST": value}):
                weighted = detect_point_group_reduction(grid, D4H_GE, abelian)
            self.assertIsInstance(weighted, PointGroupReduction)
            self.assertEqual(weighted.full_operations.shape[0], 16)
            self.assertEqual(weighted.group_order, 8)
            # six sectors: the two partners of the E levels merged into one of weight 2, for g and u
            self.assertEqual(sorted(int(w) for w in weighted.sector_weights), [1, 1, 1, 1, 2, 2])
        with mock.patch.dict(os.environ, {"PARSEC_ORBITAL_GROUP": "abelian"}):
            self.assertIs(detect_point_group_reduction(grid, D4H_GE, abelian), abelian)


# H4 in D2d: two pairs in perpendicular vertical planes, above and below the origin.  Its abelian group is C2v
# (four sectors); the weighted plan takes three (weights 1, 1, 2).
D2D_H4 = (
    Atom("H", (1.0, 0.0, 0.8)),
    Atom("H", (-1.0, 0.0, 0.8)),
    Atom("H", (0.0, 1.0, -0.8)),
    Atom("H", (0.0, -1.0, -0.8)),
)


def _sector_context(size: int, *, gpu_per_rank: bool = False, exchange: bool = False):
    from ..experimental.mpi_scf import MPISectorContext

    context = MPISectorContext(SimpleNamespace(rank=0, size=size), exchange=exchange)
    context.gpu_per_rank = gpu_per_rank
    return context


class MPIRankTests(unittest.TestCase):
    """The weighted sectors of a plan are taken only where the MPI ranks can divide them (review finding C3)."""

    def test_the_sector_counts_that_a_context_divides(self) -> None:
        counts = (1, 2, 3, 4, 6, 8)
        # one rank per node with its GPUs owns one sector or more; one sector is shared by every rank
        self.assertEqual([_sector_context(4).divides(n) for n in counts], [True, False, False, True, True, True])
        self.assertEqual([_sector_context(7).divides(n) for n in counts], [True, False, False, False, False, True])
        # one GPU per rank: groups of ranks share a sector; a hybrid: ranks may own none
        self.assertTrue(all(_sector_context(4, gpu_per_rank=True).divides(n) for n in counts))
        self.assertTrue(all(_sector_context(4, exchange=True).divides(n) for n in counts))
        self.assertTrue(all(_sector_context(1).divides(n) for n in counts))

    def test_a_plan_that_the_ranks_cannot_divide(self) -> None:
        from .. import driver

        grid = build_cluster_grid(GridSettings(spacing=0.8, radius=4.0, expansion_order=4))
        abelian = SignedPermutationReduction.detect(grid, D2D_H4)
        weighted = detect_point_group_reduction(grid, D2D_H4, abelian)
        self.assertIsInstance(weighted, PointGroupReduction)
        self.assertEqual((driver._sector_count(weighted), driver._sector_count(abelian)), (3, 4))
        for context, takes in (
            (None, True),
            (SimpleNamespace(rank=0, size=4), True),  # a context that cannot tell: as before
            (_sector_context(4), False),
            (_sector_context(3), True),
            (_sector_context(4, gpu_per_rank=True), True),
        ):
            self.assertEqual(driver._ranks_divide_sectors(weighted, context), takes)
            self.assertTrue(driver._ranks_divide_sectors(abelian, context))
        grid = build_cluster_grid(GridSettings(spacing=0.6, radius=7.0, expansion_order=6))
        abelian = SignedPermutationReduction.detect(grid, D4H_GE)
        weighted = detect_point_group_reduction(grid, D4H_GE, abelian)
        self.assertEqual((driver._sector_count(weighted), driver._sector_count(abelian)), (6, 8))
        self.assertTrue(driver._ranks_divide_sectors(weighted, _sector_context(4)))
        self.assertFalse(driver._ranks_divide_sectors(weighted, _sector_context(7)))
        self.assertTrue(driver._ranks_divide_sectors(abelian, _sector_context(7)))


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class MPIRankPreparationTests(unittest.TestCase):
    """Four ranks with the GPUs of their node (threads here, as the one-rank-per-node runner launches them)."""

    def test_four_ranks_keep_the_abelian_sectors_of_d2d(self) -> None:
        # Before the fix the three weighted sectors went to the sector context of four ranks, whose configuration
        # failed on every rank ('MPI ranks must be between one and the representation count').
        from parsec_python.Input import parse_parsec_input

        from .. import driver
        from ..experimental.mpi_scf import MPISectorContext
        from .test_gpu_fields import _write_input
        from .thread_mpi import run_ranks

        def sectors_of(system):
            solver = system.backend.symmetry_eigensolver
            return solver.decomposition.reduction, solver.decomposition.representation_count

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            atoms = "\n".join("  " + " ".join(f"{value:.3f}" for value in atom.position) for atom in D2D_H4)
            problem = parse_parsec_input(_write_input(Path(folder), atoms=atoms)).problem
            reduction, count = sectors_of(driver.prepare_single_point(problem, backend="auto"))
            # one process: the weighted sectors, as without MPI
            self.assertIsInstance(reduction, PointGroupReduction)
            self.assertEqual(count, 3)

            def rank_main(thread):
                context = MPISectorContext(thread.comm)
                system = driver.prepare_single_point(problem, backend="auto", mpi_context=context)
                reduction, count = sectors_of(system)
                return (
                    isinstance(reduction, PointGroupReduction),
                    count,
                    context.owned_sectors,
                    dict(system.backend_info.details)["symmetry_detection"],
                )

            from .. import Symmetry

            original = Symmetry.load_or_build_reflection_decomposition
            decomposed = []

            def recorded(grid, reduction, **options):
                decomposed.append(type(reduction).__name__)
                return original(grid, reduction, **options)

            # (the thread of the maps, where it runs, decomposes the abelian group and not the weighted one)
            with mock.patch.object(Symmetry, "load_or_build_reflection_decomposition", recorded):
                outcomes = run_ranks(4, rank_main)
        self.assertEqual(len(decomposed), 4)
        self.assertNotIn("PointGroupReduction", decomposed)
        for rank, (weighted, count, owned, detection) in enumerate(outcomes):
            self.assertFalse(weighted, rank)
            self.assertEqual(count, 4, rank)
            self.assertEqual(owned, (rank,))
            self.assertIn("the 3 sectors of the larger group found are fewer than the 4 MPI ranks", detection)


def _hybrid_available() -> bool:
    return cupy_available() and native_available()


def _translation(name: str, directory: Path):
    from parsec_python.Input import parse_parsec_input
    from parsec_python.tests.force_reference import load_cases, write_case

    return parse_parsec_input(write_case(load_cases()[name], directory))


@unittest.skipUnless(_hybrid_available(), "CUDA runtime and native extension are unavailable")
class DeviceTests(unittest.TestCase):
    """Complete calculations on a GPU."""

    @classmethod
    def setUpClass(cls) -> None:
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.problem = _translation("ge2h2_nlcc_tiny", Path(directory.name)).problem

    def _problem(self, atoms, states):
        return replace(
            self.problem,
            atoms=atoms,
            recenter_geometry=False,
            grid=replace(self.problem.grid, radius=9.0),
            scf=replace(self.problem.scf, number_of_states=states),
        )

    def _run(self, problem, symmetry, environment):
        from parsec_python.acceleration import driver

        with mock.patch.dict(os.environ, environment):
            system = driver.prepare_single_point(problem, symmetry=symmetry)
            system.materialize_final_wavefunctions = False
            result = driver.run_scf(system)
            forces = driver.compute_forces(system, result, symmetrize=False)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        return SimpleNamespace(system=system, result=result, forces=forces, sectors=sectors)

    def test_d2h_cluster_with_compressed_projectors(self) -> None:
        problem = self._problem(D2H_GE, 20)  # 28 valence electrons
        runs = {
            "auto": self._run(problem, "on", {"PARSEC_SECTOR_PROJECTOR_ORBITS": "auto"}),
            "weighted": self._run(problem, "on", {"PARSEC_SECTOR_PROJECTOR_ORBITS": "weighted"}),
            "full": self._run(problem, "off", {}),
        }
        compressed = runs["auto"].sectors
        self.assertNotIsInstance(compressed.decomposition.reduction, PointGroupReduction)
        self.assertEqual(compressed.decomposition.reduction.group_order, 8)
        np.testing.assert_array_equal(compressed.decomposition.reduction.atom_orbit_sizes, [1, 4, 4, 4, 4, 2, 2])
        # the terms were those of the projectors of the three representatives
        self.assertLess(len(compressed.projector_force_labels), len(runs["auto"].system.nonlocal_operator.labels))
        self.assertEqual({int(label[0]) for label in compressed.projector_force_labels}, {0, 1, 5})
        before = runs["weighted"].sectors
        self.assertIsNone(getattr(before.decomposition.reduction, "atom_representatives", None))
        self.assertEqual(len(before.projector_force_labels), len(runs["weighted"].system.nonlocal_operator.labels))
        energies = {name: float(run.result.energies.total) for name, run in runs.items()}
        self.assertAlmostEqual(energies["auto"], energies["weighted"], delta=1e-8)
        self.assertAlmostEqual(energies["auto"], energies["full"], delta=1e-6)
        full = runs["full"].forces
        self.assertGreater(float(np.abs(full.total).max()), 1.0e-3)
        for name in ("auto", "weighted"):
            np.testing.assert_allclose(
                runs[name].forces.nonlocal_pseudopotential, full.nonlocal_pseudopotential, rtol=0, atol=2e-6,
                err_msg=name,
            )
            np.testing.assert_allclose(runs[name].forces.total, full.total, rtol=0, atol=2e-6, err_msg=name)
        np.testing.assert_allclose(
            runs["auto"].forces.total, runs["weighted"].forces.total, rtol=0, atol=1e-7
        )

    def test_d4h_and_d2d_clusters_in_weighted_sectors(self) -> None:
        bond = 4.9 / np.sqrt(3.0)
        d2d = (Atom("Ge", (0.0, 0.0, 0.0)),) + tuple(
            Atom("Ge", (bond * x, bond * y, bond * z))
            for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        ) + (Atom("Ge", (0.0, 0.0, 4.4)), Atom("Ge", (0.0, 0.0, -4.4)))
        for name, atoms, states, weights in (
            ("D4h", D4H_GE, 18, [1, 1, 1, 1, 2, 2]),  # 24 valence electrons
            ("D2d", d2d, 20, [1, 1, 2]),  # 28 valence electrons
        ):
            with self.subTest(group=name):
                problem = self._problem(atoms, states)
                weighted = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_MAX_COST": "0.75"})
                abelian = self._run(problem, "on", {"PARSEC_ORBITAL_GROUP_MAX_COST": "0.5"})
                reduction = weighted.sectors.decomposition.reduction
                self.assertIsInstance(reduction, PointGroupReduction)
                self.assertEqual(sorted(int(w) for w in reduction.sector_weights), weights)
                self.assertNotIsInstance(abelian.sectors.decomposition.reduction, PointGroupReduction)
                self.assertAlmostEqual(
                    float(weighted.result.energies.total), float(abelian.result.energies.total), delta=1e-8
                )
                np.testing.assert_allclose(weighted.forces.total, abelian.forces.total, rtol=0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
