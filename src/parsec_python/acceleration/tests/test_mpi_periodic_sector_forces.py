"""The forces of a periodic cell in the sectors of its point group on several MPI ranks (the full-SCF runner).

Each rank of ``benchmarks/mpi_full_scf.py`` sums the nonlocal force terms of the sectors it solves over the
projectors of one atom per orbit of the group of the sectors (``projector_force_labels``); the ``forces`` command
of :class:`..experimental.mpi_scf.MPISectorContext` adds them on the root, which completes them over the full group
``G`` of the reduction (``Forces/compute._sector_atom_forces``).  Here the real context runs on threads that stand
for the ranks (as in ``test_mpi_ionic_steps``): the root's forces of the summed rows equal those of one process, and
ranks whose rows belong to other projectors (another count, or other labels) stop the command on every rank
instead of summing them, and a move off the group of the sectors fails the geometry command on every rank.  The
runner's decisions for a cell (what it refuses, the symmetry of its solver under the rule of the command line and
PARSEC_PERIODIC_SECTOR_FORCES, one sector of the identity for molecular dynamics, manual movements and fixed atoms,
whether its forces are symmetrized, the ionic backend, the periodic ionic scope before the preparation) are checked
without a GPU.  The GPU runs of the layouts are in ``item3_scripts`` of the branch (A100).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from parsec_python.Forces import NonlocalProjectorGradients
from parsec_python.Forces.symmetry import symmetrize_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration.Forces import compute as accelerated_compute
from parsec_python.acceleration.Forces.sector_symmetry import sector_force_operations
from parsec_python.acceleration.Symmetry.axis_reflection import compress_gradients_for_sectors
from parsec_python.acceleration.benchmarks import mpi_full_scf
from parsec_python.acceleration.experimental.mpi_scf import (
    MPISCFError,
    MPISectorContext,
    force_signature_mismatches,
    force_term_signature,
)
from parsec_python.acceleration.tests.test_driver_cli import PERIODIC_SILICON
from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World
from parsec_python.acceleration.tests.test_mpi_scf import _make_solver
from parsec_python.acceleration.tests.test_periodic_translations import _SIDE, _reduction, _supercell

# s and the three p projectors of every atom (the local channel d), as the labels of the builders.
_CHANNELS = ((0, 0), (1, 0), (1, 1), (1, 2))


def _cell():
    """Si 2x2x2 on 20 points per axis: G of 192 operations (Td and the half-cell translations), H of 32."""

    grid, atoms = _supercell((2, 2, 2))
    reduction = _reduction(grid, atoms, "auto")
    lattice = np.diag(_SIDE * np.full(3, 2.0))
    system = SimpleNamespace(
        backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(decomposition=SimpleNamespace(reduction=reduction))),
        atoms=tuple(atoms),
        grid=grid,
        input=SimpleNamespace(periodic_cell=SimpleNamespace(lattice_vectors=lattice)),
    )
    labels = tuple((atom, l, m) for atom in range(len(atoms)) for l, m in _CHANNELS)
    rng = np.random.default_rng(7)
    columns = len(labels)
    gradients = NonlocalProjectorGradients(
        components=tuple(sp.random(50, columns, density=0.2, format="csc", random_state=seed) for seed in (1, 2, 3)),
        signs=rng.choice((-1.0, 1.0), size=columns),
        labels=labels,
    )
    compressed = compress_gradients_for_sectors(gradients, reduction)
    return system, labels, tuple(compressed.labels)


class _Protocol:
    """The real sector context of every rank on a thread of its own, with toy sector solvers (test_mpi_scf)."""

    def __init__(self, case: unittest.TestCase):
        self.case = case

    def run(self, size, prepare, root_action):
        world = _World(size)

        def rank_main(rank):
            context = MPISectorContext(_Comm(world, rank))
            solver, builder = _make_solver(context)
            prepare(rank, solver, context)
            if rank:
                context.worker_loop(solver, builder)
                return None
            result = root_action(solver)
            context.stop_workers()
            return result

        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(rank_main, rank) for rank in range(size)]
            return [future.result(timeout=40) for future in futures][0]

    def fail(self, size, prepare, root_action, message):
        world = _World(size)
        failed = {}

        def rank_main(rank):
            context = MPISectorContext(_Comm(world, rank))
            solver, builder = _make_solver(context)
            prepare(rank, solver, context)
            with self.case.assertRaisesRegex(MPISCFError, message):
                if rank:
                    context.worker_loop(solver, builder)
                else:
                    root_action(solver)
            failed[rank] = context.failed

        with ThreadPoolExecutor(max_workers=size) as pool:
            for future in [pool.submit(rank_main, rank) for rank in range(size)]:
                future.result(timeout=40)
        self.case.assertEqual(failed, {rank: True for rank in range(size)})


class SectorForceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.system, cls.labels, cls.compressed = _cell()
        rng = np.random.default_rng(11)
        # The rows of the four toy sectors over the compressed projectors.
        cls.rows = {sector: rng.standard_normal((len(cls.compressed), 3)) for sector in range(4)}
        cls.selection = (np.array([0, 1, 0, 2, 3]), np.array([0, 0, 1, 0, 0]), np.array([1.0, 1.0, 0.75, 0.5, 0.25]))

    def setUp(self) -> None:
        self.protocol = _Protocol(self)

    def _sector_rows(self, labels=None, rows=None):
        """What ``_projector_force_terms_local`` does for the toy sectors: the labels set, the owned rows summed."""

        labels = self.compressed if labels is None else labels
        table = self.rows if rows is None else rows

        def prepare(rank, solver, context):
            def local(representations, columns, occupations, solver=solver):
                solver.projector_force_labels = tuple(labels)
                total = np.zeros_like(next(iter(table.values())))
                for sector in solver._owned_representations:
                    total = total + table[sector] * float(occupations[representations == sector].sum())
                return total

            solver._projector_force_terms_local = local

        return prepare

    def _root_forces(self, solver):
        terms = solver.projector_force_terms(*self.selection)
        operator = SimpleNamespace(labels=self.labels)
        return accelerated_compute._sector_atom_forces(self.system, solver, terms, operator, len(self.system.atoms))

    def test_the_cell_compresses_its_projectors_to_one_atom_per_orbit(self) -> None:
        reduction = self.system.backend.symmetry_eigensolver.decomposition.reduction
        self.assertEqual(reduction.full_operations.shape[0], 192)
        self.assertLess(len(self.compressed), len(self.labels))
        self.assertEqual(len(sector_force_operations(self.system)), 192)

    def test_compressed_rows_summed_on_the_root_equal_one_process(self) -> None:
        forces = {size: self.protocol.run(size, self._sector_rows(), self._root_forces) for size in (1, 2, 3, 4)}
        one = forces[1]
        self.assertEqual(one.shape, (len(self.system.atoms), 3))
        self.assertGreater(float(np.abs(one).max()), 1.0e-3)
        for size in (2, 3, 4):
            np.testing.assert_allclose(forces[size], one, rtol=0, atol=1.0e-13, err_msg=f"{size} ranks")
        # Every atom has its term: the average over G of the representatives' rows.
        np.testing.assert_allclose(symmetrize_forces(one, sector_force_operations(self.system)), one, rtol=0,
                                   atol=1.0e-13)

    def test_rows_of_other_projector_labels_fail_on_every_rank(self) -> None:
        other = list(self.compressed)
        other[0], other[1] = other[1], other[0]
        labels = {0: self.compressed, 1: tuple(other)}

        def prepare(rank, solver, context):
            self._sector_rows(labels=labels[rank])(rank, solver, context)

        self.protocol.fail(2, prepare, self._root_forces,
                           r"forces failed: rank 1: its nonlocal force terms \(rows \(\d+, 3\), projector labels")

    def test_rows_of_another_count_fail_on_every_rank(self) -> None:
        # A rank that kept the projectors of every atom (not compressed) beside ranks that compressed them.
        rng = np.random.default_rng(3)
        full = {sector: rng.standard_normal((len(self.labels), 3)) for sector in range(4)}

        def prepare(rank, solver, context):
            if rank == 2:
                self._sector_rows(labels=self.labels, rows=full)(rank, solver, context)
            else:
                self._sector_rows()(rank, solver, context)

        self.protocol.fail(4, prepare, self._root_forces,
                           rf"forces failed: rank 2: its nonlocal force terms \(rows \({len(self.labels)}, 3\)")

    def test_the_spin_forces_are_compared_too(self) -> None:
        def prepare(rank, solver, context):
            def local(selections, solver=solver, rank=rank):
                solver.projector_force_labels = self.compressed if rank == 0 else self.compressed[::-1]
                return np.ones((len(self.compressed), 3))

            solver._spin_projector_force_terms_local = local

        self.protocol.fail(2, prepare, lambda solver: solver.spin_projector_force_terms((self.selection,) * 2),
                           "spin_forces failed: rank 1: its nonlocal force terms")

    def test_every_rank_observes_each_forces_command(self) -> None:
        seen = {}

        def prepare(rank, solver, context):
            self._sector_rows()(rank, solver, context)

            def observe(command, count, rank=rank):
                seen.setdefault(rank, []).append((command, count))
                if rank == 1:
                    raise RuntimeError("an observation that fails stops nothing")

            context.command_observer = observe

        def root_action(solver):
            return [self._root_forces(solver) for _ in range(2)]

        first, second = self.protocol.run(3, prepare, root_action)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(seen, {rank: [("forces", 1), ("forces", 2)] for rank in range(3)})

    def test_the_signature_names_the_rows_and_their_labels(self) -> None:
        terms = np.zeros((len(self.compressed), 3))
        shape, digest = force_term_signature(terms, self.compressed)
        self.assertEqual(shape, (len(self.compressed), 3))
        # Integers of NumPy and of Python give the same digest; another order does not.
        numpy_labels = [tuple(np.int64(value) for value in label) for label in self.compressed]
        self.assertEqual(force_term_signature(terms, numpy_labels), (shape, digest))
        self.assertNotEqual(force_term_signature(terms, self.compressed[::-1])[1], digest)
        self.assertEqual(force_term_signature(terms, None), (shape, None))
        self.assertEqual(force_signature_mismatches([(shape, digest)] * 3), [None] * 3)
        found = force_signature_mismatches([(shape, digest), (shape, digest), ((1, 3), digest)])
        self.assertEqual(found[:2], [None, None])
        self.assertIn("rows (1, 3)", found[2])


class SectorMoveProtocolTests(unittest.TestCase):
    """The geometry command of a cell in sectors: every rank checks the moved atoms against the group of its sectors.

    The driver's ``move_atoms`` (the geometry handler of every rank) first checks the moved atoms
    (``Forces/sector_symmetry.check_sector_atoms``) and only then replaces the projectors; here the handlers run that
    check alone. A move on the group reaches every rank, and one off it fails the command on every rank alike, which
    the runner's root raises as ``MPISCFError`` before any rank has changed its state.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.system, _labels, _compressed = _cell()
        cls.reduction = cls.system.backend.symmetry_eigensolver.decomposition.reduction
        cls.lattice = np.asarray(cls.system.input.periodic_cell.lattice_vectors, dtype=np.float64)
        cls.positions = np.array([atom.position for atom in cls.system.atoms], dtype=np.float64)

    def _checking(self, received):
        from parsec_python.acceleration.Forces.sector_symmetry import check_sector_atoms

        def prepare(rank, solver, context):
            def handler(positions, rank=rank):
                atoms = tuple(replace(atom, position=tuple(row)) for atom, row in zip(self.system.atoms, positions))
                check_sector_atoms(self.reduction, atoms, self.system.grid, self.lattice)
                received[rank] = positions.copy()

            solver.geometry_handler = handler

        return prepare

    def test_moves_on_the_group_reach_every_rank_and_one_off_it_fails_on_every_rank(self) -> None:
        protocol = _Protocol(self)
        received = {}
        # An atom named by another lattice image: the same structure, on the group.
        moved = self.positions.copy()
        moved[3] -= self.lattice[1]
        protocol.run(3, self._checking(received), lambda solver: solver.mpi_context.execute(solver, "geometry", moved))
        self.assertEqual(sorted(received), [0, 1, 2])
        for positions in received.values():
            np.testing.assert_array_equal(positions, moved)
        received.clear()
        off = self.positions.copy()
        off[4, 0] += 0.01
        protocol.fail(3, self._checking(received), lambda solver: solver.mpi_context.execute(solver, "geometry", off),
                      r"geometry failed: rank 0: ValueError: the moved atoms leave the group of the symmetry sectors"
                      r".*; rank 1: ValueError: .*; rank 2: ValueError: ")
        self.assertEqual(received, {})


def _translation(**changes):
    translation = parse_parsec_input(PERIODIC_SILICON)
    return replace(translation, **changes)


class RunnerDecisionTests(unittest.TestCase):
    """What the full-SCF runner decides for the forces and the moves of a periodic cell."""

    def test_forces_and_moves_of_a_cell_pass_and_the_rest_is_refused_by_name(self) -> None:
        refuse = mpi_full_scf._refuse_unsupported_periodic
        cell = _translation()
        atoms = len(cell.problem.atoms)
        refuse(cell)
        refuse(replace(cell, compute_forces=True))
        bfgs = SimpleNamespace(method="bfgs")
        refuse(replace(cell, compute_forces=True, relaxation=bfgs, movable=(True,) * atoms))
        refuse(replace(cell, compute_forces=True, relaxation=SimpleNamespace(method="simple"), movable=()))
        # The moves outside the group run on one sector of the identity (the layout is decided in execute).
        refuse(replace(cell, compute_forces=True, molecular_dynamics=object()))
        refuse(replace(cell, compute_forces=True, relaxation=SimpleNamespace(method="manual")))
        refuse(replace(cell, compute_forces=True, relaxation=bfgs, movable=(False,) + (True,) * (atoms - 1)))
        cases = {
            "Fortran-derived routines": replace(
                cell, compute_forces=True,
                problem=replace(cell.problem, grid=replace(cell.problem.grid, fortran_routines=True, points_rule="round"))),
            "band structure": replace(cell, compute_forces=True,
                                      problem=replace(cell.problem, bands=object())),
        }
        fortran = cases["Fortran-derived routines"]
        # The moves of the Fortran-derived routines stay refused, on one sector of the identity too.
        cases["molecular dynamics of a periodic cell on the Fortran-derived routines"] = replace(
            fortran, molecular_dynamics=object())
        for message, case in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                refuse(case)
        with self.assertRaisesRegex(ValueError, "supercell of a k-point mesh.*main.py --mpi"):
            refuse(replace(cell, compute_forces=True), supercell=True)
        # A supercell without forces runs as before; an SCF alone on the Fortran routines too.
        refuse(cell, supercell=True)
        refuse(replace(cases["Fortran-derived routines"], compute_forces=False))

    def test_the_solver_of_a_cell_keeps_its_point_group_where_the_atoms_do(self) -> None:
        decide = mpi_full_scf._periodic_solver_symmetry
        cell = _translation()
        atoms = len(cell.problem.atoms)
        forces = replace(cell, compute_forces=True)
        relaxation = replace(forces, relaxation=SimpleNamespace(method="bfgs"), movable=(True,) * atoms)
        for translation in (cell, forces, relaxation):
            self.assertEqual(decide(translation, None, "off"), "auto")
            self.assertEqual(decide(translation, "on", "off"), "on")
            self.assertEqual(decide(translation, "off", "off"), "off")
        self.assertEqual(decide(replace(forces, ignore_symmetry=True), None, "off"), "off")
        for leaves, name, reason in (
                (replace(forces, molecular_dynamics=object()), "molecular dynamics", "molecular dynamics uses no symmetry"),
                (replace(forces, relaxation=SimpleNamespace(method="manual")), "a relaxation", "manual movements"),
                (replace(relaxation, movable=(True,) * (atoms - 1) + (False,)), "a relaxation", "Move_Flag fixes atoms")):
            with self.subTest(reason=reason):
                self.assertEqual(decide(leaves, None, "off"), "off")
                self.assertEqual(decide(leaves, None, "auto"), "off")
                with self.assertRaisesRegex(
                        ValueError, f"--symmetry on cannot be used here: {name} of a periodic cell takes one sector "
                                    f"of the identity \\({reason}"):
                    decide(leaves, "on", "off")

    def test_the_switch_of_the_command_line_puts_the_forces_and_moves_of_a_cell_on_one_sector(self) -> None:
        # PARSEC_PERIODIC_SECTOR_FORCES=off, the rule of main.py (acceleration.cli): the forces and the
        # relaxations of a cell take one sector of the identity, as main.py takes the full grid; an SCF alone
        # keeps the sectors of its point group.
        decide = mpi_full_scf._periodic_solver_symmetry
        cell = _translation()
        forces = replace(cell, compute_forces=True)
        relaxation = replace(forces, relaxation=SimpleNamespace(method="bfgs"), movable=(True,) * len(cell.problem.atoms))
        with mock.patch.dict("os.environ", {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}):
            self.assertEqual(decide(cell, None, "off"), "auto")
            for translation in (forces, relaxation):
                self.assertEqual(decide(translation, None, "off"), "off")
                with self.assertRaisesRegex(ValueError, r"one sector of the identity \(PARSEC_PERIODIC_SECTOR_FORCES=off\)"):
                    decide(translation, "on", "off")
            self.assertIn("PARSEC_PERIODIC_SECTOR_FORCES=off", mpi_full_scf._periodic_sector_forces_refusal(forces))
        with mock.patch.dict("os.environ", {"PARSEC_PERIODIC_SECTOR_FORCES": "auto"}):
            self.assertEqual(decide(relaxation, None, "off"), "auto")
            self.assertIsNone(mpi_full_scf._periodic_sector_forces_refusal(relaxation))
            # This runner reads no wfn.dat: the clause of main.py for stored full-grid orbitals does not apply.
            self.assertIsNone(mpi_full_scf._periodic_sector_forces_refusal(
                replace(relaxation, restart_run=True, restart_from_wfndat=True)))
            self.assertEqual(decide(replace(relaxation, restart_run=True, restart_from_wfndat=True), None, "off"), "auto")
            # The supercell of a k-point mesh repeats the atoms (its forces are refused before, by name).
            self.assertIn("supercell", mpi_full_scf._periodic_sector_forces_refusal(forces, supercell=True))
        # An SCF alone and a cluster have no reason.
        self.assertIsNone(mpi_full_scf._periodic_sector_forces_refusal(cell))
        self.assertIsNone(mpi_full_scf._periodic_sector_forces_refusal(
            SimpleNamespace(problem=SimpleNamespace(periodic_cell=None), compute_forces=True)))

    def test_forces_of_a_prepared_cell_are_symmetrized_where_its_sectors_complete_them(self) -> None:
        decide = mpi_full_scf._periodic_force_symmetry
        system, _labels, _compressed = _cell()
        cell = _translation(compute_forces=True)
        relaxation = replace(cell, relaxation=SimpleNamespace(method="bfgs"))
        self.assertTrue(decide(cell, system))
        self.assertTrue(decide(relaxation, system))

        def prepared(reduction):
            return SimpleNamespace(backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(
                decomposition=SimpleNamespace(reduction=reduction))))

        abelian = prepared(SimpleNamespace(group_order=8, full_operations=None))
        identity = prepared(SimpleNamespace(group_order=1))
        dynamics = replace(cell, molecular_dynamics=object())
        # Stand-ins without the operations of a reduction, and one sector of the identity, sum their terms
        # unsymmetrized; their moves replace the projectors of the sectors as well (driver move_atoms).  A real
        # Abelian reduction of a cell is symmetrized over its group (sector_symmetry.abelian_cell_operations;
        # test_cli_periodic_sector_forces, steepest descent in Abelian sectors).
        for translation in (cell, relaxation, dynamics):
            self.assertFalse(decide(translation, abelian))
            self.assertFalse(decide(translation, identity))
            self.assertFalse(decide(translation, SimpleNamespace(backend=SimpleNamespace(symmetry_eigensolver=None))))
        with self.assertRaisesRegex(ValueError, "not a periodic cell"):
            decide(SimpleNamespace(problem=SimpleNamespace(periodic_cell=None)), system)

    def test_the_ionic_backend_of_a_cell_keeps_its_structure_on_its_group(self) -> None:
        from parsec_python.acceleration.driver import ionic_backend

        cell = _translation(compute_forces=True)
        backend = mpi_full_scf._ionic_backend_of(cell, ionic_backend)
        self.assertIsNotNone(backend.symmetry_operations)
        cluster = SimpleNamespace(problem=SimpleNamespace(periodic_cell=None))
        self.assertIsNone(mpi_full_scf._ionic_backend_of(cluster, ionic_backend).symmetry_operations)

    def _prepare_through_the_runner(self, added="", *, species="", ranks=1, environment=None):
        """Run ``execute`` on the silicon cube with forces up to its preparation.

        ``added`` goes before Atom_Types_Num, ``species`` after Local_Component; ``ranks`` ranks one per node (a
        fake communicator in this thread). Returns the options of ``prepare_single_point`` with ``scope``, whether
        the periodic ionic scope was entered for forces before it, or raises what ``execute`` raised before preparing.
        """
        from parsec_python.acceleration import driver as driver_module
        from parsec_python.acceleration.Forces import periodic_device

        class Prepared(Exception):
            pass

        entered, prepared = [], {}

        def scope(translation):
            entered.append(bool(translation.compute_forces))
            return nullcontext(False)

        def prepare(problem, **options):
            # The scope is entered before the preparation.
            prepared.update(options, scope=list(entered))
            raise Prepared()

        def on_devices(cp, devices, run, memory):
            return cp, run(), [], None, {}

        def allgather(value):
            # One rank per node: the host names differ.
            if isinstance(value, str):
                return [value if rank == 0 else f"{value}-{rank}" for rank in range(ranks)]
            return [value] * ranks

        comm = SimpleNamespace(rank=0, size=ranks, allgather=allgather, Barrier=lambda: None,
                               bcast=lambda value, root=0: value)
        text = PERIODIC_SILICON.read_text().replace("Skip_force: .true.", "Skip_force: .false.")
        text = text.replace(" Atom_Types_Num:", added + " Atom_Types_Num:")
        text = text.replace(" Local_Component: p", " Local_Component: p\n" + species)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "parsec.in"
            source.write_text(text)
            shutil.copy(PERIODIC_SILICON.parent / "Si_POTRE.DAT", directory)
            arguments = mpi_full_scf._build_parser().parse_args(
                ["--input", str(source), "--output-dir", str(Path(directory) / "run"), "--quiet"])
            with (
                mock.patch.dict("os.environ", environment or {}),
                mock.patch.object(driver_module, "prepare_single_point", prepare),
                mock.patch.object(periodic_device, "periodic_ionic_scope", scope),
                mock.patch.object(mpi_full_scf, "_prepare_on_devices", on_devices),
            ):
                try:
                    mpi_full_scf.execute(arguments, None, comm, None, 0.0)
                except Prepared:
                    return prepared
        self.fail("the runner did not prepare")

    def test_the_runner_prepares_the_forces_of_a_cell_in_its_sectors_inside_the_ionic_scope(self) -> None:
        prepared = self._prepare_through_the_runner()
        self.assertEqual(prepared["symmetry"], "auto")
        self.assertEqual(prepared["scope"], [True])
        self.assertIsInstance(prepared["mpi_context"], MPISectorContext)
        self.assertIsNotNone(prepared["mpi_context"].command_observer)
        relaxation = self._prepare_through_the_runner("Minimization: bfgs\nMovement_Num: 2\n", ranks=2)
        self.assertEqual(relaxation["symmetry"], "auto")

    def test_the_moves_outside_the_group_take_one_sector_of_the_identity_where_the_ranks_share_it(self) -> None:
        dynamics = ("Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 300\nT_Step: 0\nTime_Step: 20\n"
                    "Step_Num: 2\nRng_Seed: 7\n")
        bfgs = "Minimization: bfgs\nMovement_Num: 2\n"
        for name, added, species, environment in (
                ("molecular dynamics", dynamics, "", None),
                ("a manual relaxation", "Minimization: manual\nMovement_Num: 1\n", "", None),
                ("fixed atoms", bfgs, " Move_Flag: 7\n", None),
                ("forces under off", "", "", {"PARSEC_PERIODIC_SECTOR_FORCES": "off"}),
                ("a relaxation under off", bfgs, "", {"PARSEC_PERIODIC_SECTOR_FORCES": "off"})):
            with self.subTest(name=name):
                prepared = self._prepare_through_the_runner(added, species=species, environment=environment)
                # One rank: one sector of the identity (the driver's single sector in the context of the runner).
                self.assertEqual(prepared["symmetry"], "off")
                self.assertIsInstance(prepared["mpi_context"], MPISectorContext)
        # One rank on each of two nodes does not divide one sector: refused by name before the preparation.
        with self.assertRaisesRegex(ValueError, "molecular dynamics uses no symmetry"):
            self._prepare_through_the_runner(dynamics, ranks=2)
        with self.assertRaisesRegex(ValueError, r"no point-group symmetry \(Move_Flag fixes atoms\)"):
            self._prepare_through_the_runner(bfgs, species=" Move_Flag: 7\n", ranks=2)
        with self.assertRaisesRegex(ValueError, r"no point-group symmetry \(PARSEC_PERIODIC_SECTOR_FORCES=off\)"):
            self._prepare_through_the_runner(environment={"PARSEC_PERIODIC_SECTOR_FORCES": "off"}, ranks=2)
        with self.assertRaisesRegex(ValueError, "PARSEC_PERIODIC_SECTOR_FORCES must be auto or off"):
            self._prepare_through_the_runner(environment={"PARSEC_PERIODIC_SECTOR_FORCES": "sometimes"})


if __name__ == "__main__":
    unittest.main()
