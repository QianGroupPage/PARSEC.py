"""Placement of the symmetry sectors on MPI ranks and GPUs by their size (stream A of 2026-11-19).

The MPI full-SCF runner dealt the sectors by count: sector ``s`` to rank ``s % size`` (the root took the extras on
top of the fields of the SCF loop), ranks of one GPU each in equal groups per sector (ten sectors on sixteen ranks:
six pairs and four single ranks holding a whole basis), and a rank's GPUs in equal groups per sector.
``PARSEC_SECTOR_PLACEMENT=size`` (the default) places them by cost, rows times first states
(``Eigensolvers/symmetry.sector_placement_costs``); ``cyclic`` gives the former placement.  The sectors and their
counts are the same either way: the toy SCF of ``test_mpi_scf`` gives the same eigenvalues and density on any
placement.  CPU only (fake communicators).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.acceleration.experimental.mpi_scf import (
    MPISCFError,
    MPISectorContext,
    MPISymmetryDensityBuilder,
    plan_sector_groups,
    sector_device_groups,
    sector_placement_setting,
    sector_rank_groups,
    sector_rank_plan,
)
from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World

# Rows of the weighted sectors of diamond cells (Td about an atom with the half-cell translations, relative to the
# largest): even cells repeated n times (10 sectors, weights 1 3 3 6 3 3 3 6 1 3; n = 4) and odd ones (5 sectors).
_EVEN = (1.0, 0.9403, 0.9701, 0.9403, 0.9701, 0.9552, 0.9552, 0.9552, 0.9552, 0.9552)
_ODD = (1.0, 0.9803, 0.9901, 0.9803, 0.9901)
_CYCLIC = {"PARSEC_SECTOR_PLACEMENT": "cyclic"}


def _ranks_of(plan, rank):
    return tuple(sector for sector, ranks in enumerate(plan) if rank in ranks)


def _load(plan, costs, rank):
    """The cost per rank of the group of ``rank``."""
    group = next(ranks for ranks in plan if rank in ranks)
    return sum(costs[sector] for sector in _ranks_of(plan, rank)) / len(group)


class SettingTests(unittest.TestCase):
    def test_values(self) -> None:
        for value, expected in (("", "size"), (" SIZE ", "size"), ("cyclic", "cyclic")):
            with patch.dict(os.environ, {"PARSEC_SECTOR_PLACEMENT": value}):
                self.assertEqual(sector_placement_setting(), expected)
        with patch.dict(os.environ, {"PARSEC_SECTOR_PLACEMENT": "count"}), self.assertRaises(ValueError):
            sector_placement_setting()
        with patch.dict(os.environ, {"PARSEC_SECTOR_PLACEMENT": "count"}), self.assertRaises(ValueError):
            sector_device_groups(4, 2, 0, (0, 1))


class RankPlanTests(unittest.TestCase):
    """One rank with the GPUs of its node per group of sectors (the cluster launch)."""

    def test_equal_sectors_that_divide_the_ranks_keep_the_placement_by_count(self) -> None:
        for sectors, size in ((4, 4), (8, 4), (4, 2), (4, 1), (10, 5), (6, 3), (10, 10)):
            with self.subTest(sectors=sectors, size=size):
                self.assertEqual(sector_rank_plan(sectors, size), tuple((s % size,) for s in range(sectors)))
                for rank in range(size):
                    for devices in ((0,), (0, 1), (0, 1, 2, 3)):
                        with patch.dict(os.environ, _CYCLIC):
                            expected = sector_device_groups(sectors, size, rank, devices)
                        self.assertEqual(sector_device_groups(sectors, size, rank, devices), expected)

    def test_the_root_takes_the_lightest_share(self) -> None:
        # Ten equal sectors on four ranks: by count 3/3/2/2 with three on the root; by size the root takes two.
        plan = sector_rank_plan(10, 4)
        counts = [len(_ranks_of(plan, rank)) for rank in range(4)]
        self.assertEqual(sorted(counts), [2, 2, 3, 3])
        self.assertEqual(counts[0], 2)
        with patch.dict(os.environ, _CYCLIC):
            self.assertEqual([len(_ranks_of(sector_rank_plan(10, 4), rank)) for rank in range(4)], [3, 3, 2, 2])
        # The sectors of diamond cells (ten of an even cell, five of an odd one) on four ranks.
        for costs, root_count in ((_EVEN, 2), (_ODD, 1)):
            plan = sector_rank_plan(len(costs), 4, costs=costs)
            loads = [_load(plan, costs, rank) for rank in range(4)]
            self.assertEqual(len(_ranks_of(plan, 0)), root_count)
            self.assertEqual(loads[0], min(loads))
            cyclic = [sum(costs[s] for s in range(rank, len(costs), 4)) for rank in range(4)]
            self.assertLessEqual(max(loads), max(cyclic) + 1e-12)
            self.assertEqual(sorted(s for rank in range(4) for s in _ranks_of(plan, rank)), list(range(len(costs))))

    def test_a_large_sector_gets_a_rank_alone(self) -> None:
        costs = (5.0, 1.0, 1.0, 1.0, 1.0)
        plan = sector_rank_plan(5, 2, costs=costs)
        self.assertEqual(_ranks_of(plan, 1), (0,))
        self.assertEqual(_ranks_of(plan, 0), (1, 2, 3, 4))
        with patch.dict(os.environ, _CYCLIC):
            self.assertEqual(_ranks_of(sector_rank_plan(5, 2, costs=costs), 0), (0, 2, 4))

    def test_counts_that_cannot_divide_are_refused_as_before(self) -> None:
        for placement in ({}, _CYCLIC):
            with patch.dict(os.environ, placement):
                with self.assertRaises(ValueError):
                    sector_device_groups(3, 4, 0, (0, 1, 2, 3))
                with self.assertRaises(ValueError):
                    sector_device_groups(2, 4, 0, (0,), costs=(1.0, 2.0))
        with self.assertRaises(ValueError):
            sector_rank_plan(3, 2, costs=(1.0, -1.0, 1.0))


class DeviceGroupTests(unittest.TestCase):
    """The GPUs of one rank."""

    def test_equal_sectors_up_to_the_device_count_keep_the_groups_of_before(self) -> None:
        for count in range(1, 9):
            devices = tuple(range(count))
            for sectors in range(1, count + 1):
                with patch.dict(os.environ, _CYCLIC):
                    expected = sector_device_groups(sectors, 1, 0, devices)
                self.assertEqual(sector_device_groups(sectors, 1, 0, devices, divided=True), expected,
                                 (sectors, count))
                with patch.dict(os.environ, {"PARSEC_SECTOR_DEVICE_POOL": "off"}):
                    with patch.dict(os.environ, _CYCLIC):
                        expected = sector_device_groups(sectors, 1, 0, devices)
                    self.assertEqual(sector_device_groups(sectors, 1, 0, devices, divided=True), expected,
                                     (sectors, count))
                # Bases that the first solve does not divide (CHEBDAV) never take the pool: the disjoint groups
                # of before without it.
                self.assertEqual(sector_device_groups(sectors, 1, 0, devices), expected, (sectors, count))

    def test_more_sectors_than_devices(self) -> None:
        devices = (0, 1, 2, 3)
        # Eight equal sectors on four devices: two each, round the devices, as before.
        self.assertEqual(sector_device_groups(8, 1, 0, devices), {s: (s % 4,) for s in range(8)})
        self.assertEqual(sector_device_groups(8, 1, 0, devices, divided=True), {s: (s % 4,) for s in range(8)})
        # Five whose first solve divides the bases: the pool (1.25 of a basis per device instead of two on
        # device 0), unless it is switched off.
        self.assertEqual(sector_device_groups(5, 1, 0, devices, divided=True), {s: devices for s in range(5)})
        with patch.dict(os.environ, {"PARSEC_SECTOR_DEVICE_POOL": "off"}):
            self.assertEqual(sector_device_groups(5, 1, 0, devices, divided=True), {s: (s % 4,) for s in range(5)})
        with patch.dict(os.environ, _CYCLIC):
            self.assertEqual(sector_device_groups(5, 1, 0, devices, divided=True), {s: (s % 4,) for s in range(5)})

    def test_bases_that_are_not_divided_keep_one_device_each(self) -> None:
        # Review of 2026-11-19: a basis that the first solve leaves whole lies on the first device of its group,
        # so the pool put every sector of the rank on device 0 (one after another, devices 1-3 idle).  After a
        # CHEBDAV first solve (divided False) or where a sector of the rank is below the threshold of the
        # generalized Ritz step (divisible False), the sectors go round the devices as by count.
        devices = (0, 1, 2, 3)
        for sectors in (5, 6, 7, 9, 10, 11):
            with self.subTest(sectors=sectors):
                cyclic = {s: (s % 4,) for s in range(sectors)}
                self.assertEqual(sector_device_groups(sectors, 1, 0, devices), cyclic)
                self.assertEqual(sector_device_groups(sectors, 1, 0, devices, divided=True,
                                                      divisible=(False,) * sectors), cyclic)
                some = (True,) * (sectors - 1) + (False,)
                self.assertEqual(sector_device_groups(sectors, 1, 0, devices, divided=True, divisible=some), cyclic)
                self.assertEqual(sector_device_groups(sectors, 1, 0, devices, divided=True,
                                                      divisible=(True,) * sectors), {s: devices for s in range(sectors)})
                # Every device holds sectors.
                groups = sector_device_groups(sectors, 1, 0, devices)
                self.assertEqual({group[0] for group in groups.values()}, set(devices))
        # Ten sectors of the diamond cells on two ranks (five each): the same.
        for rank in (0, 1):
            groups = sector_device_groups(10, 2, rank, devices, costs=_EVEN)
            self.assertEqual(sorted(len(group) for group in groups.values()), [1] * 5)
            self.assertEqual({group[0] for group in groups.values()}, set(devices))
            pooled = sector_device_groups(10, 2, rank, devices, costs=_EVEN, divided=True)
            self.assertEqual(set(pooled.values()), {devices})
        # Only the flags of the rank's own sectors count.
        flags = (True,) * 5 + (False,) * 5
        plan = sector_rank_plan(10, 2, costs=_EVEN)
        for rank in (0, 1):
            pooled = all(flags[s] for s in _ranks_of(plan, rank))
            groups = sector_device_groups(10, 2, rank, devices, costs=_EVEN, divided=True, divisible=flags)
            self.assertEqual(set(groups.values()) == {devices}, pooled)
        with self.assertRaises(ValueError):
            sector_device_groups(10, 2, 0, devices, divided=True, divisible=(True,) * 9)
        # By count the pool of three sectors on four devices stays as before 2026-11-19, whatever the flags.
        with patch.dict(os.environ, _CYCLIC):
            self.assertEqual(sector_device_groups(3, 1, 0, devices, divisible=(False,) * 3),
                             {s: devices for s in range(3)})

    def test_groups_in_proportion_to_the_costs(self) -> None:
        self.assertEqual(sector_device_groups(2, 1, 0, (0, 1, 2, 3), costs=(3.0, 1.0)), {0: (0, 1, 2), 1: (3,)})
        self.assertEqual(sector_device_groups(2, 1, 0, (0, 1, 2, 3), costs=(1.0, 0.97)), {0: (0, 1), 1: (2, 3)})
        # Nearly equal sectors that do not divide the devices take the pool, as equal ones did, where the first
        # solve divides their bases; otherwise the disjoint groups (each basis on the first device of its group).
        self.assertEqual(sector_device_groups(3, 1, 0, (0, 1, 2, 3), costs=(1.0, 0.99, 0.98), divided=True),
                         {s: (0, 1, 2, 3) for s in range(3)})
        self.assertEqual(sector_device_groups(3, 1, 0, (0, 1, 2, 3), costs=(1.0, 0.99, 0.98)),
                         {0: (0, 1), 1: (2,), 2: (3,)})
        # Several sectors per device: longest first.
        groups = sector_device_groups(4, 1, 0, (0, 1), costs=(4.0, 1.0, 1.0, 2.0))
        self.assertEqual(groups, {0: (0,), 1: (1,), 2: (1,), 3: (1,)})


class SharedPlanTests(unittest.TestCase):
    """Ranks of one GPU each that share the sectors (--rank-per-gpu)."""

    def assertGroupsCoverTheRanks(self, plan, size) -> None:
        groups = sorted(set(plan))
        self.assertEqual(sorted(rank for group in groups for rank in group), list(range(size)))
        for group in groups:
            self.assertEqual(group, tuple(range(group[0], group[0] + len(group))))

    def test_ten_sectors_on_sixteen_ranks_hold_no_whole_basis_on_a_gpu(self) -> None:
        for costs in (_EVEN, None):
            plan = sector_rank_plan(10, 16, shared=True, costs=costs, divided=True)
            self.assertGroupsCoverTheRanks(plan, 16)
            values = (1.0,) * 10 if costs is None else costs
            loads = [_load(plan, values, rank) for rank in range(16)]
            self.assertLessEqual(max(loads), 0.7 * max(values))
            self.assertEqual(loads[0], min(loads))
            # By count: four single ranks with a whole sector each.
            former = sector_rank_groups(10, 16)
            self.assertEqual(max(_load(former, values, rank) for rank in range(16)), max(values[6:]))
            # Without a first solve that divides the bases (every rank of a group then holds them whole) and by
            # count, the groups of before.
            self.assertEqual(sector_rank_plan(10, 16, shared=True, costs=costs), former)
            with patch.dict(os.environ, _CYCLIC):
                self.assertEqual(sector_rank_plan(10, 16, shared=True, costs=costs, divided=True), former)

    def test_sectors_that_divide_the_ranks_keep_their_groups(self) -> None:
        for sectors, size in ((4, 16), (4, 8), (2, 8), (8, 16)):
            self.assertEqual(sector_rank_plan(sectors, size, shared=True, divided=True),
                             sector_rank_groups(sectors, size))
        # Five nearly equal sectors on sixteen ranks: five groups (1/3 of a basis per GPU), not one of all.
        plan = sector_rank_plan(5, 16, shared=True, costs=_ODD, divided=True)
        self.assertEqual(sorted(len(group) for group in set(plan)), [3, 3, 3, 3, 4])

    def test_the_one_sector_of_a_run_without_symmetry(self) -> None:
        self.assertEqual(sector_rank_plan(1, 4, shared=True, divided=True), ((0, 1, 2, 3),))
        self.assertEqual(sector_device_groups(1, 4, 2, (3,), shared=True), {0: (3,)})

    def test_plan_counts(self) -> None:
        # Ten sectors on four ranks of one GPU: two groups of two with five sectors (2.5 bases per GPU, not 3).
        plan = plan_sector_groups((1.0,) * 10, 4, range(1, 5))
        self.assertEqual(sorted(set(plan)), [(0, 1), (2, 3)])
        self.assertEqual(plan_sector_groups((1.0,) * 8, 4, range(1, 5)), tuple((s % 4,) for s in range(8)))
        with self.assertRaises(ValueError):
            plan_sector_groups((1.0, 1.0), 4, (3,))


def _context(world, rank, *, gpu_per_rank=False, divided=False, costs=None):
    context = MPISectorContext(_Comm(world, rank))
    context.gpu_per_rank, context.divided_bases = gpu_per_rank, divided
    if costs is not None:
        configure = context.configure
        context.configure = lambda count, devices, **kwargs: configure(count, devices, costs=costs)
    return context


def _on_ranks(size, action):
    world = _World(size)
    with ThreadPoolExecutor(max_workers=size) as pool:
        futures = [pool.submit(action, world, rank) for rank in range(size)]
        return [future.result(timeout=25) for future in futures]


class ContextTests(unittest.TestCase):
    def test_configure_places_alike_on_every_rank(self) -> None:
        def action(world, rank):
            context = _context(world, rank)
            groups = context.configure(10, (0, 1, 2, 3), costs=_EVEN)
            return groups, context.rank_groups, context.sector_owners, context.topology
        results = _on_ranks(4, action)
        plan = sector_rank_plan(10, 4, costs=_EVEN)
        for rank, (groups, rank_groups, owners, topology) in enumerate(results):
            self.assertEqual(tuple(groups), _ranks_of(plan, rank))
            self.assertEqual(rank_groups, plan)
            self.assertEqual(owners, tuple(group[0] for group in plan))
            self.assertEqual(topology, tuple(result[0] for result in results))
        self.assertEqual(len(results[0][0]), 2)

    def test_ranks_that_place_otherwise_stop_the_configuration(self) -> None:
        def action(world, rank):
            context = _context(world, rank)
            try:
                context.configure(10, (0, 1, 2, 3), costs=_EVEN if rank else tuple(reversed(_EVEN)))
            except MPISCFError as error:
                return str(error)
            return None
        for message in _on_ranks(2, action):
            self.assertIn("placed the sectors differently", message)

    def test_shared_groups_of_ranks_of_one_gpu(self) -> None:
        def action(world, rank):
            context = _context(world, rank, gpu_per_rank=True, divided=True)
            groups = context.configure(10, (0,), costs=_EVEN)
            return groups, context.shared, context.rank_groups, context.sector_comm is not context.comm
        world_size = 4
        # _Comm has no Split: the shared configuration of several sectors asks for one.
        with patch.object(_Comm, "Split", lambda self, color, key: ("split", color, key), create=True):
            results = _on_ranks(world_size, action)
        plan = plan_sector_groups(_EVEN, 4, range(1, 5))
        for rank, (groups, shared, rank_groups, split) in enumerate(results):
            self.assertTrue(shared and split)
            self.assertEqual(rank_groups, plan)
            self.assertEqual(tuple(groups), _ranks_of(plan, rank))
            self.assertEqual(set(groups.values()), {(0,)})
            self.assertEqual(len(groups), 5)

    def test_sharing_and_division_rules(self) -> None:
        comm = SimpleNamespace(rank=1, size=4)
        context = MPISectorContext(comm)
        context.gpu_per_rank = True
        self.assertFalse(context._shared(10))
        context.divided_bases = True
        self.assertTrue(context._shared(10, _EVEN))
        self.assertFalse(context._shared(8))
        with patch.dict(os.environ, _CYCLIC):
            self.assertFalse(context._shared(10))
        # The counts that divide are those of the placement by count.
        for size in range(1, 9):
            for sectors in range(1, 11):
                for gpu_per_rank in (False, True):
                    for exchange in (False, True):
                        for divided in (False, True):
                            context = MPISectorContext(SimpleNamespace(rank=0, size=size), exchange=exchange)
                            context.gpu_per_rank, context.divided_bases = gpu_per_rank, divided
                            with patch.dict(os.environ, _CYCLIC):
                                expected = context.divides(sectors)
                            self.assertEqual(context.divides(sectors), expected,
                                             (size, sectors, gpu_per_rank, exchange, divided))

    def test_plans_follow_the_costs_the_solver_gives(self) -> None:
        context = MPISectorContext(SimpleNamespace(rank=0, size=4))
        self.assertTrue(context.accepts_sector_costs)
        self.assertEqual(context.planned_groups(10, (0, 1, 2, 3), costs=_EVEN),
                         sector_device_groups(10, 4, 0, (0, 1, 2, 3), costs=_EVEN))
        from parsec_python.acceleration import driver

        self.assertEqual(driver._planned_sector_device_groups(10, (0, 1, 2, 3), context, costs=_EVEN),
                         context.planned_groups(10, (0, 1, 2, 3), costs=_EVEN))

    def test_the_plan_of_the_driver_and_the_configuration_take_the_same_division(self) -> None:
        # Five sectors on one rank of four GPUs (an odd diamond cell on one node, CHEBFF): the pool where every
        # basis is divided, one GPU each where one is not; the driver plans the Hartree device from the same.
        from parsec_python.acceleration import driver

        devices = (0, 1, 2, 3)
        for divisible, pooled in ((None, True), ((True,) * 5, True), ((True,) * 4 + (False,), False)):
            with self.subTest(divisible=divisible):
                world = _World(1)
                context = MPISectorContext(_Comm(world, 0))
                context.divided_bases = True
                planned = context.planned_groups(5, devices, costs=_ODD, divisible=divisible)
                self.assertEqual(driver._planned_sector_device_groups(5, devices, context, costs=_ODD,
                                                                      divisible=divisible), planned)
                self.assertEqual(context.configure(5, devices, costs=_ODD, divisible=divisible), planned)
                self.assertEqual(set(planned.values()) == {devices}, pooled)
                if not pooled:
                    self.assertEqual({group[0] for group in planned.values()}, set(devices))
        # A CHEBDAV first solve (the runner sets divided_bases from the input) divides nothing.
        context = MPISectorContext(SimpleNamespace(rank=0, size=1))
        planned = context.planned_groups(5, devices, costs=_ODD)
        # Longest first: sectors 0, 2, 4 and 1 one GPU each, sector 3 beside the lightest of them.
        self.assertEqual(planned, {0: (0,), 1: (3,), 2: (1,), 3: (3,), 4: (2,)})
        self.assertEqual(planned, context.planned_groups(5, devices, costs=_ODD, divisible=(True,) * 5))


class DivisionTests(unittest.TestCase):
    """Which sector bases a CHEBFF first solve divides over the GPUs of a rank, and where the memory rule counts
    the bases that it does not divide (review of 2026-11-19)."""

    def test_the_flags_follow_the_threshold_of_the_generalized_ritz_step(self) -> None:
        from parsec_python.acceleration.Eigensolvers.symmetry import first_sector_count, sector_bases_divisible
        from parsec_python.acceleration.backends.implicit_stencil import FilterPlan

        weights = np.array([1, 3, 3, 6, 3, 3, 3, 6, 1, 3])
        # Diamond 4 x 4 x 4 cells at 18 points per cubic edge (Si 4^3 of the review): 72^3 points, sectors of
        # about 11.7k rows, 1,030 states and a buffer of 8: rows x columns^2 about 2.5e7, below 1e8.
        small = SimpleNamespace(sector_sizes=(11_700,) * 10, sector_weights=weights, representation_count=10)
        plan = FilterPlan(states=1030, buffer=8, applications=1)
        count = first_sector_count(11_700, weights, 1030, 8)
        self.assertLess(11_700 * count ** 2, 100_000_000)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("PARSEC_CUPY_DISTRIBUTED_STATE", "PARSEC_CUPY_GENERALIZED_RITZ",
                                              "PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION"))}
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(sector_bases_divisible(small, plan), (False,) * 10)
            # Diamond 15^3 (five sectors of about 1.6 million rows and 2,700 states): divided.
            large = SimpleNamespace(sector_sizes=(1_600_000, 1_570_000, 1_580_000, 1_570_000, 1_580_000),
                                    sector_weights=np.array([1, 3, 3, 3, 3]), representation_count=5)
            self.assertEqual(sector_bases_divisible(large, FilterPlan(states=27_006, buffer=8, applications=1)),
                             (True,) * 5)
            # A sector named by its size alone; no plan: not known.
            sized = SimpleNamespace(sector_size=lambda index: 11_700, representation_count=10,
                                    sector_weights=weights)
            self.assertEqual(sector_bases_divisible(sized, plan), (False,) * 10)
            self.assertIsNone(sector_bases_divisible(small))
            with patch.dict(os.environ, {"PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD": "1000"}):
                self.assertEqual(sector_bases_divisible(small, plan), (True,) * 10)
            with patch.dict(os.environ, {"PARSEC_CUPY_DISTRIBUTED_STATE": "off"}):
                self.assertEqual(sector_bases_divisible(large), (False,) * 5)
            with patch.dict(os.environ, {"PARSEC_CUPY_DISTRIBUTED_STATE": "on"}):
                self.assertEqual(sector_bases_divisible(small, plan), (True,) * 10)

    def stub(self, *, divided, group=None, devices=(0, 1, 2, 3), initial_method="chebff"):
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import attach_shared_group  # noqa: F401

        operator = SimpleNamespace(distributed_filter_devices=devices, _mpi_shared_group=group)
        context = None if divided is None else SimpleNamespace(divided_bases=divided)
        return SimpleNamespace(
            resident_device_bytes={}, device_ids=(0, 1, 2, 3), spin_channel_count=1, mpi_context=context,
            # The sector solver of the construction: CHEBFF, whatever the input names.
            _solvers=[SimpleNamespace(settings=SimpleNamespace(initial_method=initial_method))],
            _sector_device_ids=(0,), _operators=[operator],
            decomposition=SimpleNamespace(sector_size=lambda index: 1_000_000))

    def counted(self, stub, count, shared=(0, 1, 2, 3)):
        from contextlib import nullcontext

        from parsec_python.acceleration.Eigensolvers import symmetry

        asked = []

        def devices(operator, columns):
            asked.append(columns)
            return shared

        cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda device: nullcontext()))
        with patch.object(symmetry, "require_cupy", return_value=(cp, None)), \
                patch.object(symmetry, "shared_basis_devices", devices):
            return symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes(stub, [count]), asked

    def test_a_basis_that_is_not_divided_is_counted_whole_on_its_owner(self) -> None:
        from parsec_python.acceleration.Eigensolvers.distributed_state import shared_block_bytes
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import streaming_workspace_bytes

        rows, count = 1_000_000, 500
        whole = rows * count * 8 + streaming_workspace_bytes(rows, count)
        # CHEBDAV (the runner's divided_bases False) on a rank whose GPUs were given to the sector: the whole
        # basis on the first one, which the rule that shares a basis is not asked about.
        counted, asked = self.counted(self.stub(divided=False), count)
        self.assertEqual(counted, {0: whole, 1: 0, 2: 0, 3: 0})
        self.assertEqual(asked, [])
        # CHEBFF: the shared blocks on every device, as before.
        counted, asked = self.counted(self.stub(divided=True), count)
        self.assertEqual(counted, {device: shared_block_bytes(rows, count, 4) for device in range(4)})
        self.assertEqual(asked, [count])
        # Below the threshold the sharing rule declines (None): the whole basis on the owner, as before.
        counted, _ = self.counted(self.stub(divided=True), count, shared=None)
        self.assertEqual(counted, {0: whole, 1: 0, 2: 0, 3: 0})
        # Without an MPI context the settings of the sector solver decide, as before.
        counted, asked = self.counted(self.stub(divided=None, initial_method="chebdav"), count)
        self.assertEqual((counted[0], asked), (whole, []))
        counted, asked = self.counted(self.stub(divided=None), count)
        self.assertEqual(asked, [count])

    def test_a_shared_group_after_chebdav_counts_the_whole_basis_before_the_first_solve(self) -> None:
        # The sector solvers hold CHEBFF until their first solve, which comes after the count: the runner's
        # divided_bases (from the input) decides, so that a group of ranks that each hold the whole basis
        # (CHEBDAV) is counted so, and not as its columns.
        rows, count = 1_000_000, 500
        group = SimpleNamespace(size=4, ritz_route="block")
        with patch.dict(os.environ, {"PARSEC_MPI_SHARED_RITZ": ""}):
            whole, _ = self.counted(self.stub(divided=False, group=group, devices=(0,)), count)
            divided, _ = self.counted(self.stub(divided=True, group=group, devices=(0,)), count)
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import streaming_workspace_bytes

        self.assertEqual(whole[0], rows * count * 8 + streaming_workspace_bytes(rows, count))
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import exchange_bytes

        self.assertEqual(divided[0], rows * (-(-count // 4)) * 8 + 4 * exchange_bytes() + 11 * count ** 2 * 8)


class SchedulerTests(unittest.TestCase):
    """PARSEC_CUPY_SECTOR_SCHEDULER on a rank that solves several sectors of a shared group (review of 2026-11-19).

    The sectors of a group run their collectives on one communicator (sector_comm) and share one set of round
    buffers: on threads of their own (``streams``) they would interleave them differently on the ranks of the
    group.  Such a rank solves them one after another.
    """

    def test_several_shared_sectors_of_a_rank_are_solved_in_turn(self) -> None:
        from parsec_python.acceleration.Eigensolvers.symmetry import sector_scheduler_setting

        shared = SimpleNamespace(shared=True)
        alone = SimpleNamespace(shared=False)
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_SCHEDULER": "streams"}):
            self.assertEqual(sector_scheduler_setting(shared, (2, 6)), "sequential")
            # One sector of a group, sectors of a rank alone, one process: as named.
            self.assertEqual(sector_scheduler_setting(shared, (2,)), "streams")
            self.assertEqual(sector_scheduler_setting(alone, (2, 6)), "streams")
            self.assertEqual(sector_scheduler_setting(None, range(4)), "streams")
        for value in ("", "sequential", " Sequential "):
            with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_SCHEDULER": value}):
                if value:
                    self.assertEqual(sector_scheduler_setting(shared, (2, 6)), "sequential")
        with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_SCHEDULER": "threads"}), self.assertRaises(ValueError):
            sector_scheduler_setting(None, (0,))

    def test_the_groups_of_a_shared_configuration_ask_for_it(self) -> None:
        # Ten sectors on four ranks of one GPU each after a CHEBFF first solve: two groups of two ranks with five
        # sectors each, all on one communicator per group, so the rank solves them in turn.
        from parsec_python.acceleration.Eigensolvers.symmetry import sector_scheduler_setting

        def action(world, rank):
            context = _context(world, rank, gpu_per_rank=True, divided=True)
            context.configure(10, (0,), costs=_EVEN)
            with patch.dict(os.environ, {"PARSEC_CUPY_SECTOR_SCHEDULER": "streams"}):
                return context.shared, len(context.owned_sectors), sector_scheduler_setting(
                    context, context.owned_sectors)

        with patch.object(_Comm, "Split", lambda self, color, key: ("split", color, key), create=True):
            results = _on_ranks(4, action)
        self.assertEqual(results, [(True, 5, "sequential")] * 4)


class RunnerRecordTests(unittest.TestCase):
    def test_the_placement_line_and_record_of_the_runner(self) -> None:
        from parsec_python.acceleration.benchmarks.mpi_full_scf import _placement_line, _placement_record

        context = SimpleNamespace(rank_groups=((1,), (0,), (1,), (2, 3)), shared=True, divided_bases=True)
        self.assertEqual(_placement_line(context), "Sector placement (size): rank 0: 1; rank 1: 0 2; ranks 2-3: 3")
        self.assertEqual(_placement_record(context), dict(rule="size", shared=True, divided_bases=True,
                                                          rank_groups=[[1], [0], [1], [2, 3]]))
        self.assertIsNone(_placement_line(SimpleNamespace(rank_groups=((0, 1),))))
        self.assertIsNone(_placement_line(None))


class ProtocolTests(unittest.TestCase):
    """The toy SCF of test_mpi_scf on sectors placed otherwise than by count: the same numbers."""

    def run_protocol(self, size, costs):
        from parsec_python.acceleration.tests.test_mpi_scf import _make_solver

        def action(world, rank):
            context = _context(world, rank, costs=costs)
            solver, builder = _make_solver(context)
            if rank:
                context.worker_loop(solver, builder)
                return None
            density_builder = MPISymmetryDensityBuilder(builder, context, solver)
            settings = EigvalSettings(safety_buffer=1)
            first = solver(solver.full_operator, 5, settings=settings)
            occupations = np.array([1.0, 1.0, 0.8, 0.2, 0.0])
            density = density_builder(first.vectors, occupations, 0.25)
            second = solver(solver.full_operator, 5, settings=settings, state=first.state)
            second_density = density_builder(second.vectors, occupations, 0.25)
            context.stop_workers()
            return (first.eigenvalues, density.values, second.eigenvalues, second_density.values,
                    context.rank_groups)

        return _on_ranks(size, action)[0]

    def test_any_placement_gives_the_same_numbers(self) -> None:
        costs = (1.0, 0.5, 3.0, 0.5)
        reference = self.run_protocol(1, costs)
        for size in (2, 3, 4):
            result = self.run_protocol(size, costs)
            for ours, theirs in zip(result[:4], reference[:4], strict=True):
                np.testing.assert_allclose(ours, theirs, rtol=0.0, atol=1.0e-14)
            # Not the placement by count: sector 2, the costliest, on a rank of its own away from the root.
            self.assertNotEqual(result[4], tuple((s % size,) for s in range(4)))
            self.assertNotEqual(result[4][2], (0,))


class StateCountTests(unittest.TestCase):
    """The first counts and costs of the sectors, and the memory count of a rank that solves several shared ones."""

    def test_first_counts_and_costs(self) -> None:
        from parsec_python.acceleration.Eigensolvers.symmetry import first_sector_count, sector_placement_costs
        from parsec_python.acceleration.backends.implicit_stencil import FilterPlan

        # Weighted: requested * 1.1 / sum(weights) + buffer; abelian: PARSEC's share + buffer; at most the rows.
        self.assertEqual(first_sector_count(1000, (1, 3, 3, 6, 3), 160, 8), -(-176 // 16) + 8)
        self.assertEqual(first_sector_count(1000, (1, 1, 1, 1), 160, 8), 48)
        self.assertEqual(first_sector_count(20, (1, 1, 1, 1), 160, 8), 20)
        decomposition = SimpleNamespace(sector_sizes=(100, 50, 10), sector_weights=np.array([1, 1, 1]),
                                        representation_count=3)
        self.assertEqual(sector_placement_costs(decomposition), (100.0, 50.0, 10.0))
        plan = FilterPlan(states=30, buffer=2, applications=1)
        self.assertEqual(sector_placement_costs(decomposition, plan), (1200.0, 600.0, 100.0))
        # A decomposition that names the size of each sector alone.
        sized = SimpleNamespace(sector_size=lambda index: (100, 50, 10)[index], representation_count=3)
        self.assertEqual(sector_placement_costs(sized, plan), (1200.0, 600.0, 100.0))

    def test_a_rank_of_several_shared_sectors_counts_the_buffers_of_one_step(self) -> None:
        from parsec_python.acceleration.Eigensolvers import symmetry
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import exchange_bytes

        def stub(sectors, route):
            group = SimpleNamespace(size=4, ritz_route=route)
            return SimpleNamespace(
                resident_device_bytes={}, device_ids=(0,), spin_channel_count=1,
                _solvers=[object()] * sectors, _sector_device_ids=(0,) * sectors,
                _operators=[SimpleNamespace(_mpi_shared_group=group) for _ in range(sectors)],
                decomposition=SimpleNamespace(sector_size=lambda index: 1000))

        # The groups of the sectors of one rank share one set of round buffers where they are handed one dict.
        from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import attach_shared_group

        class Operator:  # a weak reference is kept to it
            effective_potential = SimpleNamespace(device=SimpleNamespace(id=0))

        shared: dict = {}
        operators = [Operator() for _ in range(3)]
        comm = SimpleNamespace(rank=0, size=4)
        groups = [attach_shared_group(operator, comm, ritz_route="block", round_buffers=shared)
                  for operator in operators[:2]]
        self.assertTrue(all(group._buffers is shared for group in groups))
        self.assertIsNot(attach_shared_group(operators[2], comm, ritz_route="block")._buffers, shared)

        count, rows, share = 100, 1000, 25
        with patch.object(symmetry, "require_cupy", return_value=(None, None)), \
                patch.dict(os.environ, {"PARSEC_MPI_SHARED_RITZ": ""}):
            for route, columns, buffers in (
                ("block", 1, 4 * exchange_bytes() + 11 * count ** 2 * 8),
                ("rounds", 1, rows * share * 8 + 3 * exchange_bytes() + 2 * count ** 2 * 8),
                ("rows", 1, 3 * rows * share * 8 + 2 * exchange_bytes()),
            ):
                one = symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes(stub(1, route), [count])
                three = symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes(stub(3, route), [count] * 3)
                self.assertEqual(one[0], columns * rows * share * 8 + buffers, route)
                self.assertEqual(three[0], 3 * rows * share * 8 + buffers, route)


if __name__ == "__main__":
    unittest.main()
