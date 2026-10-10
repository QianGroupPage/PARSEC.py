"""CPU protocol tests run the real symmetry coordinator against local toy solvers."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalResult
from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
from parsec_python.acceleration.Occupations.symmetry_density import CuPySymmetryDensityBuilder
from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
from parsec_python.acceleration.backends.cupy import CuPyTimingStats, cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.experimental.mpi_scf import (
    MPISCFError, MPISectorContext, MPISymmetryDensityBuilder, sector_device_groups)
from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World


class _PrivateArray(np.ndarray):
    def __deepcopy__(self, memo):
        raise AssertionError('orbital matrix was sent through MPI')


class _PrivateState:
    def __deepcopy__(self, memo):
        raise AssertionError('saved eigensolver state was sent through MPI')


class _ToySector:
    # Sectors whose passes say they filtered in FP32.
    float32 = frozenset()

    def __init__(self, sector, dimension):
        self.sector, self.dimension = sector, dimension
        self.reset()

    def reset(self):
        self.device_state = None
        self.calls = []

    def truncate_state(self, count):
        self.device_state.vectors = self.device_state.vectors[:, :count]

    def solve(self, count, settings):
        self.calls.append((count, settings.subspace.polynomial_degree))
        state = _PrivateState()
        state.vectors = np.eye(self.dimension, count).view(_PrivateArray)
        self.device_state = state
        spectra = ((-10., -9., -8., -7., -6., -5., -4., -3.),
                   (-9.5, -8.5, 10., 11., 12.), (0., 1., 2., 3.), (.5, 1.5, 2.5, 3.5))
        return CuPyEigvalResult(np.array(spectra[self.sector][:count]), state.vectors,
                               np.zeros(count), state, 'subspace', False, None, count, count, .001,
                               'float32' if self.sector in self.float32 else 'float64')


def _make_solver(context):
    solver = object.__new__(CuPySymmetrySCFEigensolver)
    multiplicities = np.array([1, 2, 1, 3, 1, 2, 1, 2])
    full_to_wedge = np.repeat(np.arange(8), multiplicities)
    reduction = SimpleNamespace(multiplicities=multiplicities, full_to_wedge=full_to_wedge,
                                wedge_size=8, full_size=len(full_to_wedge))
    orbits = (np.arange(8), np.array([1, 2, 3, 5, 7]), np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6]))
    def wedge(full):
        return np.bincount(full_to_wedge, weights=full, minlength=8)/multiplicities
    solver.decomposition = SimpleNamespace(reduction=reduction, representation_count=4,
        wedge_size=8, full_size=len(full_to_wedge), sector_size=lambda rep: len(orbits[rep]),
        invariant_wedge_values=wedge, phases=np.ones((4, len(full_to_wedge)), dtype=np.int8))
    context.configure(4, (0, 1, 2, 3))
    solver.mpi_context = context
    solver._owned_representations = context.owned_sectors
    # As in the constructor: a rank holds the orbit lists and scales of its own sectors only.
    solver._sector_orbits = tuple(item if index in context.owned_sectors else None
                                  for index, item in enumerate(orbits))
    solver._sector_scales = tuple(None if item is None else 1/np.sqrt(multiplicities[item])
                                  for item in solver._sector_orbits)
    # Every step shares these scales, so the constructor hands them out read-only.
    for item in solver._sector_scales:
        if item is not None:
            item.setflags(write=False)
    solver._solvers = [_ToySector(i, len(orbits[i])) if i in context.owned_sectors else None for i in range(4)]
    solver._sector_counts = None
    solver._state = None
    solver._mpi_latest_results = {}
    solver._sector_timing_stats = [CuPyTimingStats() for _ in range(4)]
    solver.timing_stats = CuPyTimingStats()
    solver._executor = solver._bound_executor = None
    solver._precompute_bounds = solver._serial_per_device = False
    solver.scheduler_batches = 0
    solver.scheduler_wall_seconds = 0.
    solver.device_ids = (0, 1, 2, 3)
    solver._sector_device_ids = (0, 1, 2, 3)
    solver.full_operator = object()
    solver._local_potential_getter = lambda: np.zeros(len(full_to_wedge))
    solver._configure_large_problem_allocator = lambda counts: None
    solver._update_local_potential = lambda potential: None
    solver.restore_memory_allocator = lambda: None
    def run(this, representation, count, settings, *, reset, spectral_bound=None):
        local = this._solvers[representation]
        if reset:
            local.reset()
        return local.solve(count, settings)
    solver._run_one_sector = MethodType(run, solver)
    builder = CuPySymmetryDensityBuilder(
        lambda vectors, weights, volume: (2/volume)*np.sum(np.asarray(vectors)**2*weights[None, :], axis=1),
        reducer=SymmetrySCFReducer(reduction))
    return solver, builder


class MPISCFTests(unittest.TestCase):
    def run_protocol(self, size, root_action):
        world = _World(size)
        def rank_main(rank):
            context = MPISectorContext(_Comm(world, rank))
            solver, builder = _make_solver(context)
            if rank:
                context.worker_loop(solver, builder)
                return solver
            result = root_action(solver, MPISymmetryDensityBuilder(builder, context, solver))
            context.stop_workers()
            return result
        with ThreadPoolExecutor(max_workers=size) as pool:
            futures = [pool.submit(rank_main, rank) for rank in range(size)]
            return [future.result(timeout=25) for future in futures]

    def test_owned_sector_groups_cover_devices_without_overlap(self):
        self.assertEqual(sector_device_groups(4, 1, 0, (0, 1)), {0:(0,), 1:(1,), 2:(0,), 3:(1,)})
        self.assertEqual(sector_device_groups(4, 2, 1, (0, 1, 2, 3)), {1:(0, 1), 3:(2, 3)})
        self.assertEqual(sector_device_groups(4, 4, 2, (0, 1, 2, 3)), {2:(0, 1, 2, 3)})
        with self.assertRaises(ValueError):
            sector_device_groups(4, 5, 0, (0,))

    def construct_solver(self, context, devices='0,1,2,3', scheduler='sequential'):
        """Run the real constructor on one rank; unassembled sectors are None."""
        class Device:
            def __init__(self, index=0):
                self.id = index
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        class Stream:
            null = None
            def __init__(self, **kwargs):
                pass
        # The stand-in has no asarray: construction must not upload any index map.
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=Device, Stream=Stream))
        reduction = SimpleNamespace(full_to_wedge=np.arange(8),
                                    multiplicities=np.array([1., 2., 1., 4., 1., 2., 1., 2.]))
        orbit_lists = [np.arange(8), np.array([1, 3, 5, 7]), np.arange(8), np.array([0, 3, 6])]
        orbit_requests = []
        def orbit_indices(rep):
            orbit_requests.append(rep)
            return orbit_lists[rep]
        decomposition = SimpleNamespace(representation_count=4, reduction=reduction,
            characters=np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]),
            phases=np.ones((4, 8)), sector_orbit_indices=orbit_indices,
            sector_size=lambda rep: 8, full_size=8, wedge_size=8)
        conversions = []
        def to_csr():
            conversions.append(1)
            return np.eye(8)
        metadata = SimpleNamespace(neighbors=np.zeros((1, 8), dtype=int), to_csr=to_csr,
                                   conversions=conversions)
        requested = []
        def reduced_operators(*args, representations=None, **kwargs):
            # As the cache-free builder does: only the named sectors exist.
            requested.append(representations)
            return SimpleNamespace(cache_info=None, nonlocal_operators=(None,)*4,
                stencil_metadata=tuple(
                    metadata if representations is None or index in representations else None
                    for index in range(4)))
        built = []
        def hamiltonian(*args, **kwargs):
            operator = SimpleNamespace(
                effective_potential=SimpleNamespace(data=SimpleNamespace(ptr=len(built))),
                compact_finite_difference=SimpleNamespace(storage_mode='test', neighbors=None),
                projector_count=0, mixed_precision_recurrence=None,
                implicit_tile=kwargs.get('implicit_tile', 'not named'))
            built.append(operator)
            return operator
        module = 'parsec_python.acceleration.Eigensolvers.symmetry.'
        with patch(module+'require_cupy', return_value=(cp, None)), \
             patch(module+'cupy_device_count', return_value=4), \
             patch(module+'load_or_build_reduced_operators', side_effect=reduced_operators), \
             patch(module+'CuPyHamiltonian', side_effect=hamiltonian), \
             patch(module+'CuPyEigvalSolver', return_value=SimpleNamespace()), \
             patch.dict('os.environ', PARSEC_CUPY_DEVICES=devices,
                        PARSEC_CUPY_SECTOR_SCHEDULER=scheduler, PARSEC_CUPY_COLLECTIVE_LANCZOS='0'):
            solver = CuPySymmetrySCFEigensolver(object(), None, None, decomposition,
                                                timing_stats=CuPyTimingStats(), mpi_context=context)
        if solver._executor is not None:
            self.addCleanup(solver._executor.shutdown)
        self.assertEqual(len(requested), 1)
        # What the constructor asked of the decomposition, and what it was given.
        self.orbit_requests, self.orbit_lists, self.multiplicities = (
            orbit_requests, orbit_lists, reduction.multiplicities)
        return solver, built, requested[0]

    def test_constructor_allocates_only_owned_operators_and_keeps_reporting(self):
        context = SimpleNamespace(owned_sectors=(1, 3), size=2, rank=1, root=0,
            configure=lambda count, devices: {1:(0, 1), 3:(2, 3)})
        solver, built, assembled = self.construct_solver(context)
        self.assertEqual(len(built), 2)
        self.assertEqual([index for index, op in enumerate(solver._operators) if op is not None], [1, 3])
        self.assertEqual([op.distributed_filter_devices for op in built], [(0, 1), (2, 3)])
        # No device holds Hartree objects unless the driver names one, and then every sector operator of
        # the rank is told: a sector that shares its basis gives that device no work beside its own.
        self.assertIsNone(solver.hartree_device)
        self.assertEqual([hasattr(op, 'hartree_device') for op in built], [False, False])
        solver.hartree_device = 3
        self.assertEqual((solver.hartree_device, [op.hartree_device for op in built]), (3, [3, 3]))
        # A rank that never solves Poisson leaves the totally symmetric
        # sector (index 0 here) to the root.
        self.assertEqual(tuple(assembled), (1, 3))
        self.assertIsNone(solver.totally_symmetric_stencil)
        self.assertIsNone(solver.totally_symmetric_negative_laplacian)
        self.assertEqual(solver.assembled_sector_count, 2)
        self.assertEqual(solver.finite_difference_storage, 'test')
        self.assertFalse(solver.fused_projector_scatter)
        self.assertFalse(solver.custom_projector_projection)
        self.assertEqual(solver.later_filter_precision, 'float64')
        self.assertEqual(solver.projector_reduction_modes, 'none')
        # Each owned sector's orbit list is asked for once; the others are left out.
        self.assertEqual(self.orbit_requests, [1, 3])
        self.assertEqual([item is None for item in solver._sector_orbits], [True, False, True, False])
        self.assertIs(solver._sector_orbits[1], self.orbit_lists[1])
        self.assertIs(solver._sector_orbits[3], self.orbit_lists[3])
        for name in ('_device_full_to_wedge', '_device_phases', '_device_sector_orbits', '_device_sector_scales'):
            self.assertFalse(hasattr(solver, name))
        # The constant 1/sqrt(multiplicity) of the same sectors is formed here, once.
        self.assertEqual([item is None for item in solver._sector_scales], [True, False, True, False])
        for sector in (1, 3):
            expected = 1.0/np.sqrt(self.multiplicities[self.orbit_lists[sector]])
            np.testing.assert_array_equal(solver._sector_scales[sector], expected)
            self.assertEqual(solver._sector_scales[sector].dtype, np.float64)
            # One array serves every SCF step; a consumer cannot write into it.
            self.assertFalse(solver._sector_scales[sector].flags.writeable)

    def test_filter_precision_report_before_a_later_pass_is_what_the_sectors_hold(self):
        context = SimpleNamespace(owned_sectors=(1, 3), size=2, rank=1, root=0,
            configure=lambda count, devices: {1:(0, 1), 3:(2, 3)})
        solver, built, _ = self.construct_solver(context)
        self.assertEqual(solver.later_filter_precision, 'float64')
        # One sector that holds an FP32 recurrence ends the float64 report. It says "prepared": no pass has run.
        built[1].mixed_precision_recurrence = object()
        self.assertEqual(solver.later_filter_precision,
            'float32 stencil/projectors/recurrence prepared for sectors 3; float64 Ritz and SCF')
        built[0].mixed_precision_recurrence = object()
        self.assertEqual(solver.later_filter_precision,
            'float32 stencil/projectors/recurrence prepared for sectors 1 3; float64 Ritz and SCF')
        # Once later passes have run, they are the report: these two sectors held one and filtered in FP64, as a
        # basis shared among devices does.
        later = SimpleNamespace(solver_path='subspace', filter_precision='float64')
        solver._record_filter_precision(1, later)
        solver._record_filter_precision(3, later)
        self.assertEqual(solver.later_filter_precision, 'float64')
        # A first solve is no later pass.
        solver._record_filter_precision(3, SimpleNamespace(solver_path='chebff', filter_precision='float64'))
        self.assertEqual(solver._later_filter_passes, {1: [1, 0], 3: [1, 0]})
        solver._record_filter_precision(3, SimpleNamespace(solver_path='subspace', filter_precision='float32'))
        self.assertEqual(solver.later_filter_precision,
            'float32 stencil/projectors/recurrence in 1 of 3 later filter passes (sectors 3); float64 Ritz and SCF')

    def test_root_reports_the_filter_precision_of_the_sectors_of_every_rank(self):
        for size in (1, 2, 4):
            def root_action(solver, builder):
                settings = EigvalSettings(safety_buffer=1)
                first = solver(solver.full_operator, 5, settings=settings)
                before = solver.later_filter_precision
                # Sector 3 belongs to the last rank: with several ranks the root learns of it from the solve records.
                with patch.object(_ToySector, 'float32', frozenset({3})):
                    solver(solver.full_operator, 5, settings=settings, state=first.state)
                return before, solver.later_filter_precision, solver._later_filter_passes
            before, after, passes = self.run_protocol(size, root_action)[0]
            with self.subTest(size=size):
                self.assertEqual(before, 'float64')
                # The toy sectors name every solve a later pass; sector 3 is solved once per step.
                self.assertEqual(sorted(passes), [0, 1, 2, 3])
                self.assertEqual(passes[3], [2, 1])
                total = sum(count[0] for count in passes.values())
                self.assertEqual(after, f'float32 stencil/projectors/recurrence in 1 of {total} later filter passes '
                                        '(sectors 3); float64 Ritz and SCF')

    def sector_tiles(self, context, devices='0,1,2,3', **settings):
        """Tile size the constructor names for each sector it builds, under ``settings`` alone."""
        import os
        names = ('PARSEC_CUPY_IMPLICIT_TILE', 'PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS', 'PARSEC_CUPY_MIXED_FILTER',
                 'PARSEC_CUPY_MIXED_FILTER_MIN_ROWS', 'PARSEC_CUPY_DISTRIBUTED_FILTER',
                 'PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS')
        kept = {name: value for name, value in os.environ.items() if name not in names}
        with patch.dict('os.environ', {**kept, **settings}, clear=True):
            solver, built, _ = self.construct_solver(context, devices)
        return [operator.implicit_tile for operator in built], solver

    def test_sector_stencils_pack_tiles_from_the_row_count_of_their_route(self):
        def rank(groups):
            return SimpleNamespace(owned_sectors=tuple(groups), size=1, rank=0, root=0,
                                   configure=lambda count, devices: groups)
        one_each = rank({index: (index,) for index in range(4)})
        pairs = rank({0: (0, 1), 1: (2, 3)})
        # The stand-in sectors have 8 rows: far below the size from which tiles pay.
        tiles, solver = self.sector_tiles(None)
        self.assertEqual(tiles, [0, 0, 0, 0])
        self.assertEqual(solver.finite_difference_neighbors, 'shared_across_representations')
        large = dict(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='8')
        # One process, a sector per device: with and without a sector context.
        for context in (None, one_each):
            tiles, solver = self.sector_tiles(context, **large)
            self.assertEqual(tiles, [16, 16, 16, 16])
            # Packed descriptors belong to their sector.
            self.assertEqual(solver.finite_difference_neighbors, 'private_per_representation')
        # The measured launch: one rank, its filter "distributed" over the one device of each sector.
        self.assertEqual(self.sector_tiles(one_each, **large, PARSEC_CUPY_DISTRIBUTED_FILTER='1')[0], [16]*4)
        # A group of devices shares the basis or its filter and has a row count of its own, which the limit
        # of one device does not move.
        self.assertEqual(self.sector_tiles(pairs, **large)[0], [0, 0])
        self.assertEqual(self.sector_tiles(None, **large, PARSEC_CUPY_DISTRIBUTED_FILTER='1')[0], [0]*4)
        self.assertEqual(self.sector_tiles(None, '2', **large, PARSEC_CUPY_DISTRIBUTED_FILTER='1')[0], [16]*4)
        # From that row count on the owner packs the tiles, and the others read them.
        shared = dict(PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS='8')
        tiles, solver = self.sector_tiles(pairs, **shared)
        self.assertEqual(tiles, [16, 16])
        self.assertEqual(solver.finite_difference_neighbors, 'private_per_representation')
        self.assertEqual(self.sector_tiles(rank({2: (0, 1, 2, 3)}), **shared)[0], [16])
        self.assertEqual(self.sector_tiles(None, **shared, PARSEC_CUPY_DISTRIBUTED_FILTER='1')[0], [16]*4)
        self.assertEqual(self.sector_tiles(pairs, PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS='9')[0], [0, 0])
        self.assertEqual(self.sector_tiles(pairs, **shared, PARSEC_CUPY_IMPLICIT_TILE='0')[0], [0, 0])
        self.assertEqual(self.sector_tiles(pairs, **shared, PARSEC_CUPY_MIXED_FILTER='on')[0], [0, 0])
        # A sector that one device filters does not read it.
        self.assertEqual(self.sector_tiles(one_each, **shared)[0], [0]*4)
        self.assertEqual(self.sector_tiles(None, '2', **shared, PARSEC_CUPY_DISTRIBUTED_FILTER='1')[0], [0]*4)
        with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS'):
            self.sector_tiles(pairs, PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS='many')
        # The opt-in FP32 filter does not read tiles and outranks the default.
        self.assertEqual(self.sector_tiles(None, **large, PARSEC_CUPY_MIXED_FILTER='on')[0], [0]*4)
        self.assertEqual(self.sector_tiles(None, **large, PARSEC_CUPY_MIXED_FILTER='auto')[0], [16]*4)
        self.assertEqual(self.sector_tiles(None, **large, PARSEC_CUPY_MIXED_FILTER='auto',
                                           PARSEC_CUPY_MIXED_FILTER_MIN_ROWS='8')[0], [0]*4)
        # An explicit value is taken as it is.
        self.assertEqual(self.sector_tiles(pairs, PARSEC_CUPY_IMPLICIT_TILE='16')[0], [16, 16])
        self.assertEqual(self.sector_tiles(None, PARSEC_CUPY_IMPLICIT_TILE='32', PARSEC_CUPY_MIXED_FILTER='on')[0], [32]*4)
        tiles, solver = self.sector_tiles(None, **large, PARSEC_CUPY_IMPLICIT_TILE='0')
        self.assertEqual(tiles, [0]*4)
        self.assertEqual(solver.finite_difference_neighbors, 'shared_across_representations')
        with self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_IMPLICIT_TILE'):
            self.sector_tiles(None, PARSEC_CUPY_IMPLICIT_TILE='1')

    def test_sector_stencils_pack_smaller_sectors_where_a_device_filters_several(self):
        def rank(groups):
            return SimpleNamespace(owned_sectors=tuple(groups), size=1, rank=0, root=0,
                                   configure=lambda count, devices: groups)
        # The stand-in sectors have 8 rows; the limit is the setting over the square root of the sectors of a device.
        for limit, expected in (('8', (16, 16, 16)), ('11', (16, 16, 0)), ('16', (16, 0, 0)), ('17', (0, 0, 0))):
            settings = dict(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS=limit)
            # One process on 1, 2 and 4 devices, as the measured launches: four, two and one sector on each.
            for devices, tile in zip(('0', '0,1', '0,1,2,3'), expected):
                indices = tuple(map(int, devices.split(',')))
                groups = {index: (indices[index % len(indices)],) for index in range(4)}
                for context in (None, rank(groups)):
                    with self.subTest(limit=limit, devices=devices, context=context is not None):
                        self.assertEqual(self.sector_tiles(context, devices, **settings)[0], [tile]*4)
        # A rank counts its own sectors only: two of them on one device, one on the other.
        uneven = rank({0: (0,), 2: (0,), 3: (1,)})
        self.assertEqual(self.sector_tiles(uneven, '0,1', PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='11')[0], [16, 16, 0])
        # The sectors of a device do not make up for a basis that several devices share.
        pairs = rank({0: (0, 1), 1: (0, 1)})
        self.assertEqual(self.sector_tiles(pairs, '0,1', PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='11')[0], [0, 0])

    def test_every_launch_gives_its_sectors_the_routes_of_their_devices(self):
        import os
        from parsec_python.acceleration.backends.implicit_stencil import implicit_tile_for_sector
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        # Rows of a sector of C459H204, which has not run, and of 3,480, 5,264, 7,120 and 23,768 electrons.
        sizes = (577_426, 677_542, 1_031_322, 1_194_720, 3_786_832)
        before = (0, 0, 0, 16, 16)
        # GPUs: ranks, devices of a rank, devices that filter a sector, sectors of a device, the tiles of the
        # default, and the setting that gives the launch the stencils it had.
        launches = {
            1: (1, '0', 1, 4, (16, 16, 16, 16, 16), dict(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='2200000')),
            2: (1, '0,1', 1, 2, (16, 16, 16, 16, 16), dict(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='1555634')),
            4: (1, '0,1,2,3', 1, 1, (0, 16, 16, 16, 16), dict(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='1100000')),
            8: (2, '0,1,2,3', 2, 1, before, dict(PARSEC_CUPY_IMPLICIT_TILE='0')),
            16: (4, '0,1,2,3', 4, 1, before, dict(PARSEC_CUPY_IMPLICIT_TILE='0')),
        }
        names = ('PARSEC_CUPY_IMPLICIT_TILE', 'PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS', 'PARSEC_CUPY_MIXED_FILTER',
                 'PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS', 'PARSEC_CUPY_DISTRIBUTED_FILTER',
                 'PARSEC_CUPY_SECTOR_POOL_RELEASE')
        kept = {name: value for name, value in os.environ.items() if name not in names}

        def tiles(filtering, sectors, **settings):
            with patch.dict('os.environ', {**kept, **settings}, clear=True):
                return tuple(implicit_tile_for_sector(rows, filtering, sectors_on_device=sectors) for rows in sizes)

        for gpus, (size, devices, filtering, sectors, default, former) in launches.items():
            listed = tuple(int(item) for item in devices.split(','))
            for rank in range(size):
                with self.subTest(gpus=gpus, rank=rank), patch.dict('os.environ', kept, clear=True):
                    groups = sector_device_groups(4, size, rank, listed)
                    context = SimpleNamespace(owned_sectors=tuple(groups), size=size, rank=rank, root=0,
                                              configure=lambda count, ids: groups)
                    solver, _, _ = self.construct_solver(context, devices)
                    # What the constructor hands the tile rule for every sector of the rank.
                    self.assertEqual({(solver._filter_device_count(index), solver._device_sector_count(index))
                                      for index in solver._owned_representations}, {(filtering, sectors)})
                    # A finished sector returns the pool blocks of its stream on two GPUs only, where a device
                    # solves two sectors, each alone on it.  No device of a shared basis does, on whose other
                    # devices the condition number may be estimated beside the small solve of the owner.
                    self.assertEqual(solver.pool_release_devices, listed if gpus == 2 else ())
                    shared = {device for group in groups.values() if len(group) > 1 for device in group}
                    self.assertEqual(len(shared), 4 if gpus > 4 else 0)
                    self.assertFalse(shared & set(solver.pool_release_devices))
            with self.subTest(gpus=gpus):
                # One device per sector: 650,000 rows over the square root of the sectors of the device.  A
                # group: 1,100,000 rows, whatever the sectors.  0 and a tile size are taken as they are.
                self.assertEqual(tiles(filtering, sectors), default)
                self.assertEqual(tiles(filtering, sectors, PARSEC_CUPY_IMPLICIT_TILE='auto'), default)
                self.assertEqual(tiles(filtering, sectors, PARSEC_CUPY_IMPLICIT_TILE='0'), (0,)*5)
                self.assertEqual(tiles(filtering, sectors, PARSEC_CUPY_IMPLICIT_TILE='16'), (16,)*5)
                self.assertEqual(tiles(filtering, sectors, **former), (0,)*5 if gpus > 4 else before)
                # The limit of one device moves no group, and that of a group no single device.
                self.assertEqual(tiles(filtering, sectors, PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS='1'),
                                 before if gpus > 4 else (16,)*5)
                self.assertEqual(tiles(filtering, sectors, PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS='1'),
                                 (16,)*5 if gpus > 4 else default)
                # The serial control reads the slot-major stencil on the devices of every launch.
                control = dict(kept)
                mpi_full_scf._keep_former_routes(control)
                with patch.dict('os.environ', control, clear=True):
                    self.assertEqual(tuple(implicit_tile_for_sector(rows, filtering, sectors_on_device=sectors)
                                           for rows in sizes), (0,)*5)

    def release_devices(self, context, devices, scheduler='sequential', **settings):
        """Devices on which the constructor lets a finished sector return its pool blocks, under ``settings`` alone."""
        import os
        kept = {name: value for name, value in os.environ.items() if name != 'PARSEC_CUPY_SECTOR_POOL_RELEASE'}
        with patch.dict('os.environ', {**kept, **settings}, clear=True):
            solver = self.construct_solver(context, devices, scheduler)[0]
        self.assertEqual(sorted(solver.sector_pool_releases), list(solver.pool_release_devices))
        self.assertTrue(all(record == dict(releases=0, bytes=0) for record in solver.sector_pool_releases.values()))
        return solver.pool_release_devices

    def test_sectors_release_their_pool_blocks_where_a_device_solves_several_in_turn(self):
        def rank(groups):
            return SimpleNamespace(owned_sectors=tuple(groups), size=1, rank=0, root=0,
                                   configure=lambda count, devices: groups)
        two = rank({0: (0,), 1: (1,), 2: (0,), 3: (1,)})
        one = rank({index: (0,) for index in range(4)})
        # Two devices for four sectors, with and without a sector context: each sector has a stream of its own.
        for context in (two, None):
            self.assertEqual(self.release_devices(context, '0,1'), (0, 1))
            for value in ('1', 'on', ' TRUE '):
                self.assertEqual(self.release_devices(context, '0,1', PARSEC_CUPY_SECTOR_POOL_RELEASE=value), (0, 1))
            for value in ('0', 'off', ' False '):
                self.assertEqual(self.release_devices(context, '0,1', PARSEC_CUPY_SECTOR_POOL_RELEASE=value), ())
        # Only the device that has a second sector, and a rank counts its own sectors.
        self.assertEqual(self.release_devices(None, '0,1,2'), (0,))
        self.assertEqual(self.release_devices(rank({0: (0,), 2: (0,), 3: (1,)}), '0,1'), (0,))
        # A sector per device, and a basis shared by the devices of a group: no second sector to serve.
        self.assertEqual(self.release_devices(rank({index: (index,) for index in range(4)}), '0,1,2,3'), ())
        self.assertEqual(self.release_devices(rank({0: (0, 1), 2: (2, 3)}), '0,1,2,3'), ())
        self.assertEqual(self.release_devices(rank({1: (0, 1, 2, 3)}), '0,1,2,3'), ())
        # One device: its sectors share the default stream, where a freed block serves the next one already.
        for context in (one, None):
            for value in ({}, dict(PARSEC_CUPY_SECTOR_POOL_RELEASE='1'), dict(PARSEC_CUPY_SECTOR_POOL_RELEASE='on')):
                self.assertEqual(self.release_devices(context, '0', **value), ())
        # Sectors that the stream scheduler may run side by side are not solved in turn.
        self.assertEqual(self.release_devices(None, '0', 'streams'), ())
        self.assertEqual(self.release_devices(None, '0,1', 'streams'), ())
        # A value that is neither on nor off is an error of every solver, also of one that would not release.
        for value in ('2', 'always', 'auto', ''):
            for context, devices in ((two, '0,1'), (one, '0')):
                with self.subTest(value=value, devices=devices), \
                     self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_SECTOR_POOL_RELEASE'):
                    self.release_devices(context, devices, PARSEC_CUPY_SECTOR_POOL_RELEASE=value)

    def test_a_limit_that_no_pool_release_can_use_is_refused_when_the_solver_is_built(self):
        import os
        def rank(groups):
            return SimpleNamespace(owned_sectors=tuple(groups), size=1, rank=0, root=0,
                                   configure=lambda count, devices: groups)
        name = 'PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES'
        kept = {key: value for key, value in os.environ.items()
                if key not in (name, 'PARSEC_CUPY_SECTOR_POOL_RELEASE')}
        # Two devices whose sectors return their blocks, a basis shared by four, and one device: the last two
        # return nothing at the end of a sector, and their density step reads the limit after the first eigensolve.
        launches = ((rank({0: (0,), 1: (1,), 2: (0,), 3: (1,)}), '0,1', (0, 1)),
                    (rank({1: (0, 1, 2, 3)}), '0,1,2,3', ()), (None, '0', ()))
        for context, devices, releasing in launches:
            for setting, words in (('-1', 'must be nonnegative'), ('512M', 'must be a nonnegative integer'),
                                   ('', 'must be a nonnegative integer')):
                for switch in ({}, dict(PARSEC_CUPY_SECTOR_POOL_RELEASE='0')):
                    with self.subTest(devices=devices, setting=setting, switch=switch), \
                         patch.dict('os.environ', {**kept, **switch, name: setting}, clear=True), \
                         self.assertRaisesRegex(ValueError, name + ' ' + words):
                        self.construct_solver(context, devices)
            # A limit that can be used moves no route of the constructor.
            for setting in ('0', '1073741824'):
                with self.subTest(devices=devices, setting=setting), \
                     patch.dict('os.environ', {**kept, name: setting}, clear=True):
                    self.assertEqual(self.construct_solver(context, devices)[0].pool_release_devices, releasing)

    def test_root_constructor_adds_the_totally_symmetric_sector(self):
        # Even a root whose own sectors exclude the totally symmetric one
        # (index 0 here) assembles it, because it prepares Poisson.
        root = SimpleNamespace(owned_sectors=(1, 3), size=2, rank=0, root=0,
            configure=lambda count, devices: {1:(0, 1), 3:(2, 3)})
        solver, built, assembled = self.construct_solver(root)
        self.assertEqual(solver.totally_symmetric_representation, 0)
        self.assertEqual(sorted(set(assembled)), [0, 1, 3])
        # It keeps the packed stencil; the CSR is formed once, when asked for,
        # and the eigensolver then holds that form alone.
        stencil = solver.totally_symmetric_stencil
        self.assertEqual(stencil.neighbors.shape, (1, 8))
        self.assertEqual(len(stencil.conversions), 0)
        self.assertEqual(solver.totally_symmetric_negative_laplacian.shape, (8, 8))
        self.assertIsNone(solver.totally_symmetric_stencil)
        self.assertIs(solver.totally_symmetric_negative_laplacian,
                      solver.totally_symmetric_negative_laplacian)
        self.assertEqual(len(stencil.conversions), 1)
        self.assertEqual(solver.assembled_sector_count, 3)
        self.assertEqual([index for index, op in enumerate(solver._operators) if op is not None], [1, 3])
        # Assembling that sector for Poisson asks for no orbit list of it:
        # the root keeps the lists and scales of the sectors it solves.
        self.assertEqual(self.orbit_requests, [1, 3])
        self.assertEqual([item is None for item in solver._sector_scales], [True, False, True, False])
        # One rank keeps the complete preparation.
        single = SimpleNamespace(owned_sectors=(0, 1, 2, 3), size=1, rank=0, root=0,
            configure=lambda count, devices: {index: (index,) for index in range(4)})
        solver, built, assembled = self.construct_solver(single)
        self.assertEqual(sorted(set(assembled)), [0, 1, 2, 3])
        self.assertEqual(len(built), 4)
        self.assertEqual(solver.totally_symmetric_negative_laplacian.shape, (8, 8))

    def test_selected_devices_follow_the_environment_setting(self):
        from parsec_python.acceleration.Eigensolvers.symmetry import selected_device_ids
        module = 'parsec_python.acceleration.Eigensolvers.symmetry.'
        with patch(module+'cupy_device_count', return_value=4):
            for setting, expected in (('auto', (0, 1, 2, 3)), ('', (0, 1, 2, 3)), ('current', (2,)),
                                      ('off', (2,)), ('3, 1,3', (3, 1)), ('0', (0,))):
                with self.subTest(setting=setting), patch.dict('os.environ', PARSEC_CUPY_DEVICES=setting):
                    self.assertEqual(selected_device_ids(2), expected)
            for setting in ('4', '-1', 'gpu0', ','):
                with self.subTest(setting=setting), patch.dict('os.environ', PARSEC_CUPY_DEVICES=setting), \
                     self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_DEVICES'):
                    selected_device_ids(2)

    def test_hartree_placement_plans_the_groups_the_constructor_forms(self):
        from parsec_python.acceleration import driver
        for devices in ('0,1,2,3', '1,3', '2'):
            listed = tuple(int(item) for item in devices.split(','))
            for size, rank in ((1, 0), (2, 0), (2, 1), (4, 3)):
                with self.subTest(devices=devices, size=size, rank=rank):
                    groups = sector_device_groups(4, size, rank, listed)
                    context = SimpleNamespace(owned_sectors=tuple(groups), size=size, rank=rank, root=0,
                        configure=lambda count, ids, size=size, rank=rank:
                            sector_device_groups(count, size, rank, ids))
                    solver, _, _ = self.construct_solver(context, devices)
                    self.assertEqual(solver.device_ids, listed)
                    plan = driver._planned_sector_device_groups(4, solver.device_ids, context)
                    self.assertEqual(plan, solver.sector_device_groups)
                    self.assertEqual({sector: group[0] for sector, group in plan.items()},
                        {sector: solver._sector_device_ids[sector] for sector in solver._owned_representations})
            # One process without a sector context.
            solver, _, _ = self.construct_solver(None, devices)
            plan = driver._planned_sector_device_groups(4, solver.device_ids, None)
            self.assertEqual(solver.sector_device_groups, {})
            self.assertEqual(tuple(plan[index] for index in range(4)),
                             tuple((device,) for device in solver._sector_device_ids))

    def test_serial_control_keeps_the_former_routes_unless_the_launcher_names_others(self):
        import importlib
        import os
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        # The package also exports a function named like this submodule.
        ritz = importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
        names = tuple(mpi_full_scf._SERIAL_CONTROL_SETTINGS)
        with patch.dict('os.environ'):
            for name in (*names, 'PARSEC_CUPY_RITZ_CONDITION'):
                os.environ.pop(name, None)
            mpi_full_scf._keep_former_routes(os.environ)
            # The CUDA contexts before the preparation, and from CuPy also where a launcher asks for the overlap.
            self.assertFalse(mpi_full_scf._context_overlap_requested())
            self.assertFalse(mpi_full_scf._driver_contexts_requested())
            with patch.dict('os.environ', PARSEC_OVERLAP_CUDA_CONTEXTS='1'):
                self.assertTrue(mpi_full_scf._context_overlap_requested())
                self.assertFalse(mpi_full_scf._driver_contexts_requested())
            # The GPU ionic fields in line, on the thread that prepares.
            with patch.dict('os.environ', PARSEC_IONIC_BACKEND='cupy'):
                self.assertFalse(driver._ionic_overlap_requested())
                with patch.dict('os.environ', PARSEC_OVERLAP_IONIC_SETUP='1'):
                    self.assertTrue(driver._ionic_overlap_requested())
            # The host solve with its SVD, a row-wise copy-back, two arrays even for a marked sector,
            # and the Hartree objects where they were.
            self.assertFalse(ritz.dense_solve_on_device())
            self.assertEqual(ritz._condition_policy(on_device=False), 'svd')
            self.assertFalse(ritz._column_copy_requested())
            self.assertFalse(ritz.streaming_ritz_requested(SimpleNamespace(low_memory_ritz=True)))
            self.assertIsNone(driver._hartree_device_request())
            # The slabs of its projection as wide as they were: an eighth of the columns, and the columns of
            # the budget where its launcher asks for one array.
            self.assertEqual(ritz.gram_multiple(), 1)
            with patch.dict('os.environ'):
                for name in ('PARSEC_CUPY_RITZ_GRAM_SLABS', 'PARSEC_CUPY_STREAMING_RITZ_BYTES'):
                    os.environ.pop(name, None)
                sector = SimpleNamespace(shape=(2604846, 1842), nbytes=8 * 2604846 * 1842)
                self.assertEqual((ritz._gram_slab_width(1842), ritz._streaming_shape(sector)[0]), (231, 206))
            # Its overlap has no slabs for the multiple to cut: it is asked of DSYRK, for which nothing stands
            # in here, and of no slab width, while the projection of its two arrays asks for that width once.
            basis = np.asfortranarray(np.random.default_rng(5).normal(size=(9, 4)))
            with patch.object(ritz, 'require_cupy', return_value=(np, None)), \
                 patch.object(ritz, '_gram_slab_width', wraps=ritz._gram_slab_width) as width:
                with self.assertRaises(Exception):
                    ritz._symmetric_overlap(basis)
                width.assert_not_called()
                ritz._lower_triangle_product(basis, basis)
                width.assert_called_once_with(4)
            # The pool blocks of a finished sector stay until the density step, on however many devices.
            from parsec_python.acceleration.Eigensolvers.symmetry import sector_pool_release_requested
            self.assertFalse(sector_pool_release_requested())
            # The ionic sums on the first device alone, also where the default would take them all.
            from parsec_python.acceleration.V_ion.cupy_ionic import ionic_device_ids
            self.assertEqual(ionic_device_ids((0, 1, 2, 3), 1000, sector_rank=True), (0,))
            # The slot-major stencil for a sector of any size, also where its device filters several sectors
            # and where several devices filter it.
            from parsec_python.acceleration.backends.implicit_stencil import implicit_tile_for_sector
            self.assertEqual(implicit_tile_for_sector(4_000_000, 1), 0)
            self.assertEqual(implicit_tile_for_sector(4_000_000, 1, sectors_on_device=4), 0)
            self.assertEqual(implicit_tile_for_sector(4_000_000, 4), 0)
            # The control solves symmetry sectors and records filter graphs too: its sector stencils are
            # reduced from the full-grid matrix, its maps are the former ones and it records a graph per
            # block of a plan, so that none of the three is in both sides of a comparison.
            from parsec_python.acceleration.Eigensolvers.filter_graph import graph_reuse_requested
            from parsec_python.acceleration.Symmetry.axis_reflection import _fast_maps_requested
            from parsec_python.acceleration.Symmetry.sector_stencil import sector_stencil_route
            self.assertEqual(sector_stencil_route(), 'csr')
            self.assertFalse(_fast_maps_requested())
            self.assertFalse(graph_reuse_requested())
            # The full-grid kernels of the Hartree boundary, so that the wedge kernels of an engaged plan are
            # not in both sides. Which boundary values they evaluate stays the launcher's: the plan is physics.
            self.assertEqual(driver._hartree_boundary_kernel(), 'full')
            # The values of the atomic tail from host threads, so that their device kernel is not either.
            self.assertTrue(driver._hartree_tail_on_host())
            self.assertFalse({'PARSEC_HARTREE_BOUNDARY', 'PARSEC_HARTREE_BOUNDARY_TOLERANCE',
                              'PARSEC_HARTREE_ATOMIC_TAIL', 'PARSEC_HARTREE_LPOLE'} & set(names))
            # Which reduction of that matrix runs, and the FP64 filter, are not the control's to choose.
            self.assertFalse({'PARSEC_NATIVE_SECTOR_ASSEMBLY', 'PARSEC_CUPY_MIXED_FILTER'} & set(names))
            # Nor are the switches of a shared basis, which the control never has.  The slab budget among
            # them is that of the one-array route as well: named here, it would give a control whose launcher
            # asks for one array slabs of 4 GiB where they are an eighth of the basis.
            self.assertFalse({'PARSEC_CUPY_STREAMING_RITZ_BYTES', 'PARSEC_CUPY_RITZ_CONDITION_HELPER',
                              'PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS', 'PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS',
                              'PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES', 'PARSEC_CUPY_EXCHANGE_PAIRS',
                              'PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE'} & set(names))
            # Never: only a sector context gives an operator the devices of a group.  The solver of a process
            # without one, as a control is, has four operators that no setting of the sharing lets share.
            from parsec_python.acceleration.Eigensolvers import distributed_state
            solver, built, _assembled = self.construct_solver(None, '0,1,2,3')
            self.assertEqual((len(built), [hasattr(operator, 'distributed_filter_devices') for operator in built]),
                             (4, [False] * 4))
            for policy in ('auto', 'on'):
                with patch.dict('os.environ', PARSEC_CUPY_DISTRIBUTED_STATE=policy):
                    self.assertEqual([distributed_state.shared_basis_devices(operator, 5) for operator in built],
                                     [None] * 4)
            # The former rule of the state storage where the launcher names neither value, and the former
            # download of a basis that it spills.  The values themselves stay the launcher's, and the share
            # of a device belongs to the rule that the control does not ask.
            from parsec_python.acceleration.Eigensolvers.eigval import spill_download_order
            self.assertEqual(os.environ['PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION'], '0.5')
            self.assertEqual(spill_download_order(), 'C')
            self.assertFalse({'PARSEC_CUPY_SECTOR_STATE_STORAGE', 'PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR',
                              'PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION'} & set(names))
            # The report of the sphere is no route of the solver.
            self.assertNotIn('PARSEC_DOMAIN_REPORT', names)
            # DSYRK, and under an explicit auto the fraction that used to decide between one array and two.
            self.assertEqual((os.environ['PARSEC_CUPY_RITZ_SYRK'], os.environ['PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION']),
                             ('on', '0.8'))
        # What the launcher names stays.
        chosen = {'PARSEC_CUPY_STREAMING_RITZ': 'auto', 'PARSEC_HARTREE_DEVICE': '2'}
        environment = dict(chosen)
        mpi_full_scf._keep_former_routes(environment)
        self.assertEqual(set(environment), set(names))
        self.assertEqual({name: environment[name] for name in chosen}, chosen)

    def test_serial_control_log_says_that_a_symmetry_cache_can_replace_its_maps_and_stencils(self):
        import os
        from pathlib import Path
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        with patch.dict('os.environ'):
            for name in mpi_full_scf._SERIAL_CONTROL_SETTINGS:
                os.environ.pop(name, None)
            mpi_full_scf._keep_former_routes(os.environ)
            # Without a cache the control builds everything by the routes it names: one line, the routes.
            (routes,) = mpi_full_scf._serial_control_notes(None)
            self.assertTrue(routes.startswith('Serial control routes: PARSEC_CUPY_RITZ_SYRK=on '))
            for setting in ('PARSEC_SECTOR_STENCIL=csr', 'PARSEC_SYMMETRY_FAST_MAPS=0', 'PARSEC_CUPY_FILTER_GRAPH_REUSE=0',
                            'PARSEC_HARTREE_BOUNDARY_KERNEL=full', 'PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host'):
                self.assertIn(setting, routes.split())
            # An entry of a cache holds the maps and stencils of the run that wrote it, by whatever route.
            # Only a control that was given a directory reads one.
            cache = Path('work') / '.parsec_cache' / 'symmetry'
            same, warning = mpi_full_scf._serial_control_notes(cache)
            self.assertEqual(same, routes)
            self.assertTrue(warning.startswith('WARNING: '))
            for word in (str(cache), 'PARSEC_SECTOR_STENCIL', 'PARSEC_SYMMETRY_FAST_MAPS',
                         'a control without --symmetry-cache reads none'):
                self.assertIn(word, warning)

    def test_real_coordinator_growth_trim_density_and_later_settings(self):
        all_results = []
        for size in (1, 2, 4):
            def root_action(solver, builder):
                settings = EigvalSettings(safety_buffer=1)
                first = solver(solver.full_operator, 5, settings=settings)
                np.testing.assert_array_equal(first.eigenvalues, [-10., -9.5, -9., -8.5, -8.])
                self.assertTrue(first.restarted, 'spectral bracket must grow sector zero')
                self.assertEqual(first.state.sector_state_counts[0], 3)
                # The packed orbitals carry host maps, and only those of the sectors solved here.
                self.assertIsNone(first.vectors.device_full_to_wedge)
                self.assertIs(first.vectors.phases, solver.decomposition.phases)
                for packed in (first.vectors.sector_orbits, first.vectors.sector_scales):
                    self.assertEqual([item is not None for item in packed],
                                     [index in solver._owned_representations for index in range(4)])
                # They are the solver's static arrays, not copies rebuilt for this step.
                self.assertIs(first.vectors.sector_orbits, solver._sector_orbits)
                self.assertIs(first.vectors.sector_scales, solver._sector_scales)
                occupations = np.array([1., 1., .8, .2, 0.])
                density = builder(first.vectors, occupations, .25)
                charge = .25*np.dot(solver.decomposition.reduction.multiplicities, density.values)
                self.assertAlmostEqual(charge, 2*sum(occupations), places=13)
                first.vectors.release_intermediate_storage()
                settings = replace(settings, subspace=replace(settings.subspace, polynomial_degree=12))
                second = solver(solver.full_operator, 5, settings=settings, state=first.state)
                second_density = builder(second.vectors, occupations, .25)
                np.testing.assert_array_equal(density.values, second_density.values)
                self.assertEqual(second.state.solves_completed, 2)
                self.assertEqual(solver._mpi_latest_results, {})
                if size > 1:
                    with self.assertRaisesRegex(RuntimeError, 'distributed export'):
                        second.vectors.to_full_host()
                return second.eigenvalues, second_density.values, solver
            results = self.run_protocol(size, root_action)
            all_results.append(results[0][:2])
            for solver in [results[0][2], *results[1:]]:
                for local in solver._solvers:
                    if local is not None:
                        self.assertEqual(local.calls[-1][1], 12)
        for values, density in all_results[1:]:
            np.testing.assert_array_equal(values, all_results[0][0])
            np.testing.assert_array_equal(density, all_results[0][1])

    def test_every_rank_waits_for_the_pool_release_of_its_density_behind_the_collective_calls(self):
        # The builder of a rank stands for one that hands the pools of its devices to their threads: it says
        # when it was called, what it was told about the wait, and when it was waited for.
        def protocol(size, failing=None):
            world, events = _World(size), []

            class Comm(_Comm):
                def Allreduce(self, send, recv):
                    events.append((self.rank, 'sum'))
                    super().Allreduce(send, recv)

            class Builder(CuPySymmetryDensityBuilder):
                def __call__(self, *arguments):
                    events.append((self.rank, 'density', self.caller_joins_release))
                    if self.rank == failing:
                        raise MemoryError('injected density failure')
                    return super().__call__(*arguments)

                def release_joined(self):
                    events.append((self.rank, 'joined'))
                    super().release_joined()

            def rank_main(rank):
                context = MPISectorContext(Comm(world, rank))
                solver, plain = _make_solver(context)
                builder = Builder(plain.device_builder, reducer=plain.reducer)
                builder.rank = rank
                self.assertFalse(builder.caller_joins_release)
                if rank:
                    context.worker_loop(solver, builder)
                    return
                first = solver(solver.full_operator, 5, settings=EigvalSettings(safety_buffer=1))
                MPISymmetryDensityBuilder(builder, context, solver)(first.vectors, np.array([1., 1., .8, .2, 0.]), .25)
                context.stop_workers()

            with ThreadPoolExecutor(max_workers=size) as pool:
                futures = [pool.submit(rank_main, rank) for rank in range(size)]
                errors = []
                for future in futures:
                    try:
                        future.result(timeout=25)
                    except MPISCFError as error:
                        errors.append(str(error))
            return events, errors

        for size in (1, 2, 4):
            events, errors = protocol(size)
            self.assertEqual(errors, [])
            for rank in range(size):
                # The rank says that it waits itself, sends its density, and then waits: once per density.
                self.assertEqual([event[1:] for event in events if event[0] == rank],
                                 [('density', True), ('sum',), ('joined',)])
        # A density that failed on one rank is no density of any: every rank still waits before it raises.
        events, errors = protocol(2, failing=1)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all('injected density failure' in error for error in errors))
        for rank in range(2):
            self.assertEqual([event[1:] for event in events if event[0] == rank], [('density', True), ('joined',)])

    def test_a_pool_release_that_fails_behind_the_collective_calls_fails_its_rank_alone(self):
        # The builder of a rank stands for one whose device thread could not empty its pool: the wait of the
        # rank raises, behind allgather and Allreduce. ``density`` names a rank whose density fails as well.
        def protocol(release, density=None):
            world, outcome = _World(2), {}

            class Builder(CuPySymmetryDensityBuilder):
                def __call__(self, *arguments):
                    if self.rank == density:
                        raise MemoryError('injected density failure')
                    return super().__call__(*arguments)

                def release_joined(self):
                    if self.rank == release:
                        raise RuntimeError('injected release failure')
                    super().release_joined()

            def rank_main(rank):
                comm = _Comm(world, rank)
                context = MPISectorContext(comm)
                solver, plain = _make_solver(context)
                builder = Builder(plain.device_builder, reducer=plain.reducer)
                builder.rank = rank
                try:
                    if rank:
                        context.worker_loop(solver, builder)
                    else:
                        first = solver(solver.full_operator, 5, settings=EigvalSettings(safety_buffer=1))
                        try:
                            outcome['density'] = MPISymmetryDensityBuilder(builder, context, solver)(
                                first.vectors, np.array([1., 1., .8, .2, 0.]), .25)
                        finally:
                            if context.failed:
                                # As the runner does whatever the SCF raised: a failed context sends nothing.
                                before = comm.sequence
                                context.stop_workers()
                                outcome['root sent stop'] = comm.sequence > before
                            if release == 0 and density is None:
                                # The launcher's Abort is stood in for: the worker waits for its next command.
                                comm.bcast(('stop', None), root=0)
                except Exception as error:
                    outcome[rank] = error
                outcome[rank, 'failed'] = context.failed

            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in [pool.submit(rank_main, rank) for rank in range(2)]:
                    future.result(timeout=25)
            return outcome

        # A worker: it raises the failure of the protocol and is marked, and its Abort ends the job. The
        # root has the density of both ranks and is not told: its next collective call would wait for it.
        outcome = protocol(release=1)
        self.assertIsInstance(outcome[1], MPISCFError)
        self.assertRegex(str(outcome[1]), 'density pool release failed: rank 1: RuntimeError: injected release failure')
        self.assertIsInstance(outcome[1].__cause__, RuntimeError)
        self.assertEqual((outcome[1, 'failed'], outcome[0, 'failed']), (True, False))
        self.assertIn('density', outcome)
        self.assertNotIn(0, outcome)
        # The root: it is marked before the runner stops the workers, so no 'stop' goes into the job.
        outcome = protocol(release=0)
        self.assertIsInstance(outcome[0], MPISCFError)
        self.assertRegex(str(outcome[0]), 'density pool release failed: rank 0: RuntimeError: injected release failure')
        self.assertEqual((outcome[0, 'failed'], outcome['root sent stop'], outcome[1, 'failed']), (True, False, False))
        self.assertNotIn('density', outcome)
        self.assertNotIn(1, outcome)
        # A density that failed on another rank is what every rank raises, also the one whose wait failed.
        outcome = protocol(release=1, density=0)
        for rank in range(2):
            self.assertIsInstance(outcome[rank], MPISCFError)
            self.assertRegex(str(outcome[rank]), 'density failed: rank 0: MemoryError: injected density failure')
            self.assertTrue(outcome[rank, 'failed'])
        self.assertRegex(str(outcome[1].__cause__), 'density pool release failed: rank 1')
        self.assertIsNone(outcome[0].__cause__)
        self.assertFalse(outcome['root sent stop'])

    def test_worker_failure_reaches_root_and_worker_without_orbital_transfer(self):
        world = _World(2)
        def run(rank):
            context = MPISectorContext(_Comm(world, rank))
            solver, builder = _make_solver(context)
            if rank == 1:
                def fail(*args, **kwargs):
                    raise MemoryError('injected worker OOM')
                solver._run_local_sector_jobs = fail
            with self.assertRaisesRegex(MPISCFError, 'injected worker OOM'):
                if rank:
                    context.worker_loop(solver, builder)
                else:
                    solver(solver.full_operator, 5, settings=EigvalSettings(safety_buffer=1))
            self.assertTrue(context.failed)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, rank) for rank in range(2)]
            for future in futures:
                future.result(timeout=25)


class _ContextRecordingCuPy:
    """Stands for cupy in the runner: which thread created which CUDA context, and when.

    A context is created by the first memory query on its device, as the driver creates one on first use.
    ``creating(device)`` runs there and may wait or raise. Selecting a device that the process cannot see
    fails as in the CUDA runtime. ``log`` holds, in their order, the contexts created, the memory queries
    and the start of the memory sampler that ``sampler`` stands for.
    """
    __version__ = 'stand-in'

    def __init__(self, count=4, creating=None):
        self.count, self.creating = count, creating
        self.created, self.selected, self.synchronized, self.log, self.samplers = {}, [], [], [], []
        self._current = threading.local()
        fake = self

        class Device:
            def __init__(self, device=None):
                self.id = fake.current() if device is None else int(device)
                self.pci_bus_id = f'0000:{self.id:02x}:00.0'

            def _select(self):
                if self.id >= fake.count:
                    raise RuntimeError(f'cudaErrorInvalidDevice: invalid device ordinal {self.id}')
                fake._current.device = self.id

            def use(self):
                self._select()
                fake.selected.append((self.id, threading.current_thread().name))

            def __enter__(self):
                self._previous = fake.current()
                self._select()
                return self

            def __exit__(self, *_error):
                fake._current.device = self._previous

        def query_memory():
            device = fake.current()
            fake.create(device)
            fake.log.append(('memory', device))
            return 70 << 30, 80 << 30

        pool = SimpleNamespace(total_bytes=lambda: 0, used_bytes=lambda: 0)
        self.get_default_memory_pool = lambda: pool
        self.cuda = SimpleNamespace(Device=Device, runtime=SimpleNamespace(
            getDeviceCount=lambda: fake.count, memGetInfo=query_memory,
            getDeviceProperties=lambda device: {'name': b'stand-in GPU'},
            deviceSynchronize=lambda: fake.synchronized.append(fake.current())))

    def current(self):
        return getattr(self._current, 'device', 0)

    def create(self, device):
        """Create the context of ``device`` on the calling thread unless it exists."""
        if device not in self.created:
            if self.creating is not None:
                self.creating(device)
            self.created[device] = threading.current_thread().name
            self.log.append(('context', device))

    def sampler(self):
        self.log.append(('sampler', threading.current_thread().name))
        self.samplers.append(SimpleNamespace(stop=lambda records: dict(available=False)))
        return self.samplers[-1]


class _RecordingDriver:
    """Stands for the runner's ``_DriverContexts`` on the stand-in above: the contexts made through the driver.

    ``started`` holds, for each start of the driver, the thread and the contexts that existed then. A
    library that is not ``usable`` creates nothing, and neither does a creation that fails: the stand-in
    for CuPy then creates the context, or fails, in its own call.
    """

    def __init__(self, cp, usable=True):
        self.cp, self.usable = cp, usable
        self.devices, self.seconds, self.started = [], 0., []

    def start(self):
        self.started.append((threading.current_thread().name, sorted(self.cp.created)))
        return self.usable

    def create(self, cp, device):
        if not self.usable:
            return False
        try:
            cp.create(device)
        except OSError:
            return False
        self.devices.append(device)
        self.seconds += 1e-3
        return True


class _CudaLibrary:
    """Stands for the CUDA driver library as ctypes loads it: its calls, their arguments and results.

    ``addresses`` are the PCI addresses of the devices in the order of the driver. A function named in
    ``failing`` returns an error code, one named in ``missing`` is not exported and one named in
    ``faulting`` faults. The handle of a device is not its index, and ``retained`` lists the indices
    whose primary context was asked for, after ``retain(index)`` ran.
    """

    def __init__(self, addresses, failing=(), missing=(), faulting=(), retain=None):
        self.addresses, self.failing, self.faulting = tuple(addresses), set(failing), set(faulting)
        self.calls, self.retained, self.retain = [], [], retain
        for name in ('cuInit', 'cuDeviceGet', 'cuDeviceGetPCIBusId', 'cuDevicePrimaryCtxRetain'):
            if name not in missing:
                setattr(self, name, getattr(self, '_' + name))

    def _result(self, name, *arguments):
        self.calls.append((name, *arguments))
        if name in self.faulting:
            raise OSError('exception: access violation reading 0x0000000000000000')
        return 1 if name in self.failing else 0

    def _cuInit(self, flags):
        return self._result('cuInit', flags.value)

    def _cuDeviceGet(self, handle, index):
        if index.value >= len(self.addresses):
            self.calls.append(('cuDeviceGet', index.value))
            return 101
        handle._obj.value = 100 + index.value
        return self._result('cuDeviceGet', index.value)

    def _cuDeviceGetPCIBusId(self, address, size, handle):
        address.value = self.addresses[handle.value - 100].encode('ascii')
        return self._result('cuDeviceGetPCIBusId', size.value, handle.value)

    def _cuDevicePrimaryCtxRetain(self, context, handle):
        code = self._result('cuDevicePrimaryCtxRetain', handle.value)
        if code == 0:
            if self.retain is not None:
                self.retain(handle.value - 100)
            self.retained.append(handle.value - 100)
            context._obj.value = 0x7f0000000000 + handle.value
        return code


class FullSCFRunnerContextTests(unittest.TestCase):
    """The CUDA contexts of the MPI runner: PARSEC_OVERLAP_CUDA_CONTEXTS and PARSEC_CUDA_CONTEXT_CREATION."""

    def setUp(self):
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        self.runner = mpi_full_scf
        # No test here loads the driver library of the machine: ``drivers`` holds the stand-ins built for it.
        self.drivers, self.driver_usable = [], True

        def driver():
            self.drivers.append(_RecordingDriver(self.cp, self.driver_usable))
            return self.drivers[-1]

        for name, replacement in (('_GpuMemorySampler', lambda: self.cp.sampler()), ('_DriverContexts', driver)):
            patcher = patch.object(mpi_full_scf, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def cupy(self, **options):
        """Return a stand-in for cupy; the runner's memory sampler is then started on its record."""
        self.cp = _ContextRecordingCuPy(**options)
        return self.cp

    @staticmethod
    def contexts_then_sampler(devices, thread):
        """The record of a creation in the former order: each context, the sampler, the first observation."""
        return [*(event for device in devices for event in (('context', device), ('memory', device))),
                ('sampler', thread), *(('memory', device) for device in devices)]

    def environment(self, value, creation=None):
        """Set the switch of the overlap and that of the creation, or remove one for ``None``, until the test ends."""
        import os
        patcher = patch.dict('os.environ')
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('PARSEC_OVERLAP_CUDA_CONTEXTS', None)
        os.environ.pop('PARSEC_CUDA_CONTEXT_CREATION', None)
        os.environ.pop('PARSEC_CUPY_DEVICES', None)
        if value is not None:
            os.environ['PARSEC_OVERLAP_CUDA_CONTEXTS'] = value
        if creation is not None:
            os.environ['PARSEC_CUDA_CONTEXT_CREATION'] = creation
        self.drivers.clear()

    @staticmethod
    def context_threads():
        return [thread for thread in threading.enumerate() if thread.name == 'parsec-cuda-contexts']

    def test_switch_is_on_unless_named_off(self):
        import os
        for value, expected in ((None, True), ('1', True), (' on ', True), ('0', False), (' Off ', False),
                                ('false', False), ('no', False)):
            with self.subTest(value=value), patch.dict('os.environ'):
                os.environ.pop('PARSEC_OVERLAP_CUDA_CONTEXTS', None)
                if value is not None:
                    os.environ['PARSEC_OVERLAP_CUDA_CONTEXTS'] = value
                self.assertIs(self.runner._context_overlap_requested(), expected)

    def test_creation_is_through_the_driver_library_unless_the_runtime_is_named(self):
        import os
        for value, expected in ((None, True), ('driver', True), (' Driver ', True), ('runtime', False),
                                (' RUNTIME ', False)):
            with self.subTest(value=value), patch.dict('os.environ'):
                os.environ.pop('PARSEC_CUDA_CONTEXT_CREATION', None)
                if value is not None:
                    os.environ['PARSEC_CUDA_CONTEXT_CREATION'] = value
                self.assertIs(self.runner._driver_contexts_requested(), expected)
        # Any other value stops a run before a context, a thread or the preparation, also a run that
        # creates its contexts before the preparation and so never reads the switch for them.
        for creation in ('', '0', '1', 'cupy', 'ctypes'):
            for overlap in (None, '0'):
                with self.subTest(creation=creation, overlap=overlap):
                    self.environment(overlap, creation)
                    cp, prepared = self.cupy(), []
                    with patch.dict('sys.modules', cupy=cp), self.assertRaisesRegex(
                            ValueError, 'PARSEC_CUDA_CONTEXT_CREATION must be driver or runtime'):
                        self.runner._prepare_on_devices(cp, (0, 1), lambda: prepared.append(1), [])
                    self.assertEqual((prepared, cp.log, cp.selected, self.drivers), ([], [], [], []))
                    self.assertEqual(self.context_threads(), [])

    def test_contexts_are_created_while_the_preparation_runs(self):
        import os
        for value, creation in ((None, None), ('1', None), (None, 'driver'), ('1', 'runtime'), (None, 'runtime')):
            with self.subTest(switch=value, creation=creation):
                self.environment(value, creation)
                preparing = threading.Event()

                def creating(device):
                    # No context exists before the preparation has begun: creating them first, in line,
                    # would never get past this.
                    if not preparing.wait(timeout=10.):
                        raise AssertionError('the preparation did not start beside the context creation')

                cp = self.cupy(creating=creating)

                def prepare():
                    self.assertEqual(cp.created, {})
                    self.assertEqual(cp.log, [])
                    self.assertEqual(len(self.context_threads()), 1)
                    self.assertEqual(os.environ['PARSEC_CUPY_DEVICES'], '0,1,2,3')
                    preparing.set()
                    return 'prepared system'

                memory = []
                returned, prepared, records, sampler, contexts = self.runner._prepare_on_devices(
                    cp, (0, 1, 2, 3), prepare, memory)
                self.assertIs(returned, cp)
                self.assertEqual(prepared, 'prepared system')
                # Every context by the one thread, in the order of the devices; it has ended at the join.
                self.assertEqual(list(cp.created.items()),
                                 [(device, 'parsec-cuda-contexts') for device in range(4)])
                self.assertEqual(self.context_threads(), [])
                # Device 0 is current in the calling thread without a call of its own.
                self.assertEqual(cp.selected, [])
                self.assertEqual(cp.current(), 0)
                self.assertEqual([item['index'] for item in records], [0, 1, 2, 3])
                self.assertEqual(records[2], dict(index=2, pci_bus_id='0000:02:00.0', name='stand-in GPU',
                                                  total_bytes=80 << 30, initial_free_bytes=70 << 30))
                self.assertTrue(contexts['overlapped'])
                self.assertGreater(contexts['seconds'], 0.)
                self.assertGreaterEqual(contexts['join_wait_seconds'], 0.)
                self.assertEqual([item['phase'] for item in memory],
                                 ['after_cuda_init_during_static_preparation'])
                self.assertEqual([item['index'] for item in memory[0]['devices']], [0, 1, 2, 3])
                # nvidia-smi is started by the thread once every context exists, not beside their creation,
                # and before the first observation: the order without the overlap.
                self.assertEqual(cp.log, self.contexts_then_sampler(range(4), 'parsec-cuda-contexts'))
                self.assertEqual(cp.samplers, [sampler])
                # Unless the runtime is named, the thread loads the driver library, starts it before any
                # context exists and creates every context through it, in the order of the devices: the
                # calls of CuPy that follow find them. With the runtime named no library is loaded.
                if creation == 'runtime':
                    self.assertEqual(self.drivers, [])
                    self.assertEqual((contexts['driver_devices'], contexts['driver_seconds']), ([], 0.))
                else:
                    (driver,) = self.drivers
                    self.assertEqual(driver.started, [('parsec-cuda-contexts', [])])
                    self.assertEqual(contexts['driver_devices'], [0, 1, 2, 3])
                    self.assertIsNot(contexts['driver_devices'], driver.devices)
                    self.assertEqual(contexts['driver_seconds'], driver.seconds)
                    self.assertGreater(contexts['driver_seconds'], 0.)

    def test_a_driver_library_that_cannot_be_used_leaves_the_contexts_to_cupy(self):
        # No library, or one whose calls fail: the thread creates the contexts with the calls of CuPy, as
        # with the runtime named, and the record names no device.
        self.environment(None)
        self.driver_usable = False
        cp = self.cupy()
        memory = []
        _, prepared, records, sampler, contexts = self.runner._prepare_on_devices(
            cp, (0, 1, 2, 3), lambda: 'prepared system', memory)
        self.assertEqual(prepared, 'prepared system')
        (driver,) = self.drivers
        self.assertEqual((driver.started, driver.devices), ([('parsec-cuda-contexts', [])], []))
        self.assertEqual(list(cp.created.items()), [(device, 'parsec-cuda-contexts') for device in range(4)])
        self.assertEqual(cp.log, self.contexts_then_sampler(range(4), 'parsec-cuda-contexts'))
        self.assertEqual([item['index'] for item in records], [0, 1, 2, 3])
        self.assertTrue(contexts['overlapped'])
        self.assertEqual((contexts['driver_devices'], contexts['driver_seconds']), ([], 0.))

    def test_switch_off_creates_the_contexts_before_the_preparation(self):
        # Whatever the switch of the creation says: nothing runs beside the contexts then, and CuPy
        # creates them as it did.
        for creation in (None, 'driver', 'runtime'):
            with self.subTest(creation=creation):
                self.environment('0', creation)
                cp = self.cupy()

                def prepare():
                    self.assertEqual(list(cp.created.items()), [(device, 'MainThread') for device in (1, 3)])
                    self.assertEqual(cp.log, self.contexts_then_sampler((1, 3), 'MainThread'))
                    self.assertEqual(cp.selected, [(1, 'MainThread')])
                    self.assertEqual(self.context_threads(), [])
                    return 'prepared system'

                memory = []
                with patch.dict('sys.modules', cupy=cp):
                    returned, prepared, records, sampler, contexts = self.runner._prepare_on_devices(
                        object(), (1, 3), prepare, memory)
                self.assertIs(returned, cp)
                self.assertEqual(cp.samplers, [sampler])
                self.assertEqual(prepared, 'prepared system')
                self.assertEqual([item['index'] for item in records], [1, 3])
                self.assertEqual(cp.current(), 1)
                self.assertFalse(contexts['overlapped'])
                self.assertEqual(contexts['join_wait_seconds'], 0.)
                self.assertGreater(contexts['seconds'], 0.)
                self.assertEqual([item['phase'] for item in memory], ['after_cuda_init_before_static_preparation'])
                self.assertEqual(self.drivers, [])
                self.assertEqual((contexts['driver_devices'], contexts['driver_seconds']), ([], 0.))

    def test_first_device_other_than_zero_is_selected_by_the_caller(self):
        self.environment(None)
        cp = self.cupy()
        seen = []
        _, _, records, _, contexts = self.runner._prepare_on_devices(
            cp, (2, 3), lambda: seen.append(cp.current()), [])
        # The preparing thread reads its current device, so it is set before the preparation begins.
        self.assertEqual(seen, [2])
        self.assertEqual(cp.selected, [(2, 'MainThread')])
        self.assertEqual([item['index'] for item in records], [2, 3])
        self.assertTrue(contexts['overlapped'])
        # The thread creates the contexts of those devices, by their indices.
        self.assertEqual(contexts['driver_devices'], [2, 3])
        self.assertEqual(list(cp.created.items()), [(2, 'parsec-cuda-contexts'), (3, 'parsec-cuda-contexts')])

    def test_failures_of_the_creation_and_of_the_preparation_are_both_raised(self):
        self.environment(None)

        def failing():
            raise RuntimeError('preparation failed')

        # A device that does not exist is reported, whatever became of the preparation meanwhile.
        for prepare in (lambda: 'prepared system', failing):
            with self.subTest(prepare=prepare.__name__):
                cp = self.cupy(count=2)
                with self.assertRaisesRegex(ValueError, 'only 2 visible CuPy devices') as raised:
                    self.runner._prepare_on_devices(cp, (0, 1, 2, 3), prepare, [])
                if prepare is failing:
                    self.assertIsInstance(raised.exception.__context__, RuntimeError)
                self.assertEqual(self.context_threads(), [])
                # Without contexts no sampler was started.
                self.assertEqual(cp.log, [])
        # A first device that the process cannot see is selected by the caller, which refuses it with
        # the words of the former order before the runtime does, and before any preparation.
        for value in (None, '0'):
            with self.subTest(first_device='not visible', switch=value):
                self.environment(value)
                cp, prepared = self.cupy(count=4), []
                with patch.dict('sys.modules', cupy=cp), self.assertRaisesRegex(
                        ValueError, r'requested devices \(4, 5\); only 4 visible CuPy devices'):
                    self.runner._prepare_on_devices(cp, (4, 5), lambda: prepared.append(cp.current()), [])
                self.assertEqual((prepared, cp.selected, cp.log), ([], [], []))
                self.assertEqual(self.context_threads(), [])
        self.environment(None)
        # A failure of the preparation alone is raised as it is, after the thread has ended.
        cp = self.cupy()
        with self.assertRaisesRegex(RuntimeError, 'preparation failed'):
            self.runner._prepare_on_devices(cp, (0, 1), failing, [])
        self.assertEqual(sorted(cp.created), [0, 1])
        self.assertEqual(self.context_threads(), [])

        def broken(device):
            if device == 1:
                raise OSError('context creation failed')

        # A context that cannot be created: the driver library reports nothing of its own, and the call of
        # CuPy that follows raises as it does where the runtime is named.
        for creation in (None, 'runtime'):
            with self.subTest(context='cannot be created', creation=creation):
                self.environment(None, creation)
                cp = self.cupy(creating=broken)
                with self.assertRaisesRegex(OSError, 'context creation failed'):
                    self.runner._prepare_on_devices(cp, (0, 1), lambda: 'prepared system', [])
                self.assertEqual(cp.samplers, [])
                self.assertEqual([driver.devices for driver in self.drivers], [] if creation else [[0]])

    def run_worker_rank(self, cp, preparing, operators=(), flags=()):
        """Run the runner's ``execute`` as rank 1 of 2 against a scripted communicator; return its record.

        ``operators`` are the sector operators of the rank's solver, ``None`` for a sector it does not own.
        ``flags`` are further words of the command line; ``self.prepared_with`` holds the options that the
        rank gave its preparation.
        """
        from pathlib import Path
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.models import BackendStatistics
        source = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'H_cli_smoke.in'
        gathered = []

        class Comm:
            rank, size = 1, 2

            def allgather(self, value):
                # Two nodes, each with the devices of its own.
                if isinstance(value, str):
                    return ['node-a', 'node-b']
                if isinstance(value, dict) and 'hostname' in value:
                    return [dict(value, hostname='node-a'), dict(value, hostname='node-b')]
                return [value, value]

            def Barrier(self):
                pass

            def bcast(self, value, root=0):
                return ('stop', None) if value is None and not gathered else value

            def gather(self, value, root=0):
                gathered.append(value)

        solver = SimpleNamespace(restore_memory_allocator=lambda: None, _operators=list(operators))
        seen = SimpleNamespace(created=None)

        def prepare(problem, **options):
            seen.created = dict(cp.created)
            self.prepared_with = dict(options)
            preparing.set()
            options['mpi_context'].owned_sectors = (1,)
            return SimpleNamespace(backend_info=SimpleNamespace(selected='cupy'), orbital_density_builder=None,
                                   backend=SimpleNamespace(symmetry_eigensolver=solver,
                                                           statistics=BackendStatistics()))

        # The command line of a run with default flags unless ``flags`` say more: it names no symmetry cache.
        args = self.runner._build_parser().parse_args([
            '--input', str(source), '--output-dir', str(Path.cwd() / '.tmp' / 'runner-context-test-absent'),
            '--devices', '0,1', '--quiet', *flags])
        with patch.object(driver, 'prepare_single_point', prepare), patch.dict('sys.modules', cupy=cp):
            exit_code = self.runner.execute(
                args, cp, Comm(), SimpleNamespace(Get_library_version=lambda: 'stand-in MPI '), 0.)
        self.assertIsNone(exit_code)
        (record,) = gathered
        return record, seen.created

    def test_rank_record_says_how_its_contexts_were_created(self):
        for value, creation, overlapped, first_phase in (
                (None, None, True, 'after_cuda_init_during_static_preparation'),
                (None, 'runtime', True, 'after_cuda_init_during_static_preparation'),
                ('0', None, False, 'after_cuda_init_before_static_preparation')):
            with self.subTest(switch=value, creation=creation):
                self.environment(value, creation)
                preparing = threading.Event()
                # With the overlap the first context waits until the preparation has been entered.
                cp = self.cupy(count=2, creating=(
                    (lambda device: preparing.wait(timeout=10.)) if overlapped else None))
                record, created_before_preparation = self.run_worker_rank(cp, preparing)
                self.assertEqual(sorted(created_before_preparation), [] if overlapped else [0, 1])
                self.assertIs(record['cuda_context_initialization_overlapped'], overlapped)
                self.assertGreater(record['cuda_context_initialization_seconds_in_preparation'], 0.)
                self.assertGreaterEqual(record['cuda_context_join_wait_seconds'], 0.)
                if not overlapped:
                    self.assertEqual(record['cuda_context_join_wait_seconds'], 0.)
                self.assertGreaterEqual(record['preparation_seconds'],
                                        record['cuda_context_join_wait_seconds'])
                self.assertEqual([item['index'] for item in record['devices']], [0, 1])
                self.assertEqual([item['phase'] for item in record['memory_observations']],
                                 [first_phase, 'after_preparation', 'after_scf_or_worker_loop'])
                self.assertEqual(sorted(cp.created), [0, 1])
                self.assertEqual(self.context_threads(), [])
                # One sampler, started where the contexts were created, once both existed.
                self.assertEqual(cp.log[:7], self.contexts_then_sampler(
                    (0, 1), 'parsec-cuda-contexts' if overlapped else 'MainThread'))
                self.assertEqual(len(cp.samplers), 1)
                # The devices whose context the thread created through the driver library, and its seconds
                # inside those calls: none where CuPy created them, by name or before the preparation.
                through_driver = overlapped and creation is None
                self.assertEqual(record['cuda_context_driver_devices'], [0, 1] if through_driver else [])
                self.assertEqual(record['cuda_context_driver_seconds'] > 0., through_driver)

    def test_rank_record_says_how_many_tall_blocks_the_ritz_step_of_a_shared_sector_held(self):
        # An unset PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS has meant two blocks and means one: the record of a
        # shared sector says what its last step held, next to the row blocks that it took from the pool, and
        # how it cut its slabs, which PARSEC_CUPY_DISTRIBUTED_STATE_SLABS decides for two blocks only, with the
        # columns of its widest slab and of the widest right-hand side of a product of its projection.  It also
        # names the device that estimated the condition number of the overlap in the last device small solve,
        # here the second one beside the solve of the owner, and the seconds that such helpers spent on it, and
        # how the devices issued the copies of the last way to their rows.
        # Of the seconds of the Gram stage it says those of the overlap, whose slabs are cut by another rule
        # than the rounds of the projection: a run and its control differ in each by what that rule gave.
        self.environment('0')
        seconds = dict(filter=2., gram=1.5)
        shared = SimpleNamespace(devices=(0, 1), passes=3, seconds=seconds, overlap_seconds=.5, tall_blocks=1,
                                 slab_cut='equal',
                                 slab_columns=2, projection_columns=4, separate_row_blocks=0, condition_device=1,
                                 condition_seconds=.25, to_rows_copies='together',
                                 layout=SimpleNamespace(parts=(((0, 3), (6, 8)), ((3, 6), (8, 10)))))
        idle = SimpleNamespace(devices=(0, 1), passes=0, seconds={}, overlap_seconds=0., tall_blocks=None, slab_cut=None,
                               slab_columns=None, projection_columns=None, separate_row_blocks=0, condition_device=None,
                               condition_seconds=0., to_rows_copies=None, layout=None)
        operators = (None, SimpleNamespace(_sector_device_group=shared), SimpleNamespace(),
                     SimpleNamespace(_sector_device_group=idle))
        # The layout of the stencil the devices of a group filter with, and the seconds their operators took.
        shared.stencil_storage, shared.replica_seconds = 'implicit_affine_tile_16', 0.25
        idle.stencil_storage, idle.replica_seconds = 'slot-major', 0.5
        record, _created = self.run_worker_rank(self.cupy(count=2), threading.Event(), operators)
        self.assertEqual([(entry.pop('stencil_storage'), entry.pop('replica_seconds'))
                          for entry in record['distributed_state']],
                         [('implicit_affine_tile_16', 0.25), ('slot-major', 0.5)])
        self.assertEqual(record['distributed_state'], [
            dict(sector=1, devices=[0, 1], ritz_passes=3, seconds=seconds, overlap_seconds=.5, tall_blocks=1, slab_cut='equal',
                 slab_columns=2, projection_columns=4, separate_row_blocks=0, condition_device=1, condition_seconds=.25,
                 to_rows_copies='together', column_ranges=[[[0, 3], [6, 8]], [[3, 6], [8, 10]]]),
            # A group that has taken no Ritz step yet.
            dict(sector=3, devices=[0, 1], ritz_passes=0, seconds={}, overlap_seconds=0., tall_blocks=None, slab_cut=None,
                 slab_columns=None, projection_columns=None, separate_row_blocks=0, condition_device=None,
                 condition_seconds=0., to_rows_copies=None, column_ranges=None)])
        # A group has what the record reads from its start, before any step.
        from parsec_python.acceleration.Eigensolvers import distributed_state

        class Owner:
            _distributed_filter = SimpleNamespace(devices=(0, 1), owner=0)

        owner = Owner()
        group = distributed_state.SectorDeviceGroup(owner, (0, 1))
        self.assertEqual((group.passes, group.tall_blocks, group.slab_cut, group.slab_columns, group.projection_columns,
                          group.separate_row_blocks, group.layout), (0, None, None, None, None, 0, None))
        self.assertEqual((group.condition_device, group.condition_seconds, group.to_rows_copies), (None, 0., None))
        self.assertEqual(group.overlap_seconds, 0.)

    def test_rank_record_names_the_pool_releases_of_finished_sectors(self):
        # What ``sector_pool_releases`` of the rank's solver says; a solver without it released nothing.
        self.environment('0')
        record, _created = self.run_worker_rank(self.cupy(count=2), threading.Event())
        self.assertEqual(record['sector_pool_release'], {})
        releases = {0: dict(releases=22, bytes=5 << 30)}

        class Releasing(SimpleNamespace):
            sector_pool_releases = releases

        cp = self.cupy(count=2)
        # The stand-in solver of the rank is made of the namespace type of this module.
        with patch(__name__ + '.SimpleNamespace', Releasing):
            record, _created = self.run_worker_rank(cp, threading.Event())
        self.assertEqual(record['sector_pool_release'], releases)

    def test_rank_record_names_the_state_storage_and_the_allocator_of_its_solver(self):
        # What the solver of the rank decided at its first eigensolve and the bytes that ``auto`` counted per
        # device; a solver that has not decided, or is none, leaves them empty.
        self.environment('0')
        record, _created = self.run_worker_rank(self.cupy(count=2), threading.Event())
        self.assertEqual((record['orbital_sector_state_storage'], record['orbital_memory_allocator'],
                          record['sector_state_fit_bytes']), (None, None, None))
        counted = {0: 60176837864, 1: 60176837864}

        class Decided(SimpleNamespace):
            sector_state_storage = 'persistent CUDA representation states'
            memory_allocator_policy = 'cupy default pool; estimated persistent sector orbitals cuda:0=1B,cuda:1=1B'
            state_fit_bytes = counted

        with patch(__name__ + '.SimpleNamespace', Decided):
            record, _created = self.run_worker_rank(self.cupy(count=2), threading.Event())
        self.assertEqual((record['orbital_sector_state_storage'], record['orbital_memory_allocator']),
                         (Decided.sector_state_storage, Decided.memory_allocator_policy))
        self.assertEqual(record['sector_state_fit_bytes'], counted)
        # The record is written as JSON, with the devices as its keys.
        self.assertEqual(self.runner._jsonable(record)['sector_state_fit_bytes'], {'0': 60176837864, '1': 60176837864})

    def test_rank_prepares_without_a_symmetry_cache_unless_a_directory_is_named(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from pathlib import Path
        parser = self.runner._build_parser()
        required = ['--input', 'parsec.in', '--output-dir', 'out']
        defaults = parser.parse_args(required)
        self.assertEqual((defaults.symmetry_cache, defaults.no_symmetry_cache), (None, False))
        named = Path('work') / 'cache'
        # Both flags together, and the directory flag without a directory, stay errors of the command line.
        for flags in (['--symmetry-cache', str(named), '--no-symmetry-cache'], ['--symmetry-cache']):
            with self.subTest(flags=flags), redirect_stderr(StringIO()) as refusal:
                with self.assertRaises(SystemExit):
                    parser.parse_args(required + flags)
                self.assertIn('--symmetry-cache', refusal.getvalue())
        # No directory reaches the preparation by default, the one beside the input that used to be the
        # default included; --no-symmetry-cache says the same, and a named directory is handed on.
        self.environment('0')
        for flags, expected in (((), None), (('--no-symmetry-cache',), None),
                                (('--symmetry-cache', str(named)), named.resolve())):
            with self.subTest(flags=flags):
                self.run_worker_rank(self.cupy(count=2), threading.Event(), flags=flags)
                self.assertEqual(self.prepared_with['symmetry_cache_directory'], expected)

    def run_serial_control(self, directory, flags=(), *, plan=None, details=(), scf_seconds=0., **system):
        """Run ``execute`` as a serial control whose preparation and SCF are stand-ins.

        ``system`` names further attributes of the prepared stand-in.
        Returns its timing.json, its text log and the symmetry cache directory its preparation was given.
        ``plan`` is the Hartree boundary plan of the prepared system, ``details`` the backend details of the
        result, and the stand-in SCF takes ``scf_seconds``.
        """
        import importlib
        import json
        from contextlib import redirect_stdout
        from dataclasses import dataclass
        from io import StringIO
        from pathlib import Path
        from parsec_python.acceleration.models import BackendInfo, BackendStatistics
        # A module that the runner first imported under a stand-in for cupy left sys.modules with it. The
        # runner imports by name, so the ones patched here are those that sys.modules holds now.
        driver, output, reference_cli = (importlib.import_module(name) for name in (
            'parsec_python.acceleration.driver', 'parsec_python.acceleration.Output', 'parsec_python.cli'))
        source = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'H_cli_smoke.in'

        @dataclass
        class Seconds:
            diagonalization_seconds: float = 0.
            occupations_density_seconds: float = 0.
            hartree_seconds: float = 0.
            initial_hartree_seconds: float = 0.

        class Comm:
            rank, size = 0, 1
            allgather = staticmethod(lambda value: [value])
            gather = staticmethod(lambda value, root=0: [value])
            bcast = staticmethod(lambda value, root=0: value)
            Barrier = staticmethod(lambda: None)

        given = []

        def prepare(problem, **options):
            given.append(options['symmetry_cache_directory'])
            return SimpleNamespace(backend_info=SimpleNamespace(selected='cupy'),
                                   backend=SimpleNamespace(statistics=BackendStatistics()),
                                   hartree_boundary=plan, **system)

        result = SimpleNamespace(converged=True, iterations=1, electron_count=1., fermi_level=0., history=[],
                                 energies=Seconds(), timings=Seconds(), backend_statistics=BackendStatistics(),
                                 backend=BackendInfo(requested='auto', selected='cupy', details=tuple(details)))

        def run(system, callback):
            time.sleep(scf_seconds)
            return result
        cp = self.cupy(count=1)
        args = self.runner._build_parser().parse_args([
            '--input', str(source), '--output-dir', str(directory), '--devices', '0', '--serial-control',
            '--quiet', *flags])
        with patch.object(driver, 'prepare_single_point', prepare), \
                patch.object(driver, 'run_scf', run), \
                patch.object(output, 'AcceleratedTextReporter'), \
                patch.object(reference_cli, 'save_result_archive',
                             lambda path, *_, **__: path.write_bytes(b'stand-in archive')), \
                patch.object(self.runner, '_source_provenance', lambda: None), \
                patch.dict('sys.modules', cupy=cp), redirect_stdout(StringIO()):
            exit_code = self.runner.execute(
                args, cp, Comm(), SimpleNamespace(Get_library_version=lambda: 'stand-in MPI '), 0.)
        self.assertEqual(exit_code, 0)
        (cache,) = given
        return (json.loads((directory / 'timing.json').read_text(encoding='utf-8')),
                (directory / 'parsec.out').read_text(encoding='utf-8'), cache)

    def test_timing_record_and_log_say_which_symmetry_cache_the_run_had(self):
        import os
        import shutil
        from pathlib import Path
        self.environment('0')
        directory = Path.cwd() / '.tmp' / f'runner-symmetry-cache-test-{os.getpid()}'
        named = directory / 'named'
        reads = 'WARNING: the serial control reads the symmetry cache'
        try:
            for flags in ((), ('--no-symmetry-cache',)):
                with self.subTest(flags=flags):
                    record, log, cache = self.run_serial_control(directory / f'out{len(flags)}', flags)
                    self.assertIsNone(cache)
                    # null in the record: no cache, and no path of one that was not used anywhere.
                    self.assertIn('symmetry_cache_directory', record)
                    self.assertIsNone(record['symmetry_cache_directory'])
                    self.assertIsNone(record['parameters']['symmetry_cache'])
                    self.assertNotIn(reads, log)
                    self.assertNotIn('.parsec_cache', log)
                    self.assertTrue(any('symmetry_cache_directory' in note for note in record['notes']))
            record, log, cache = self.run_serial_control(directory / 'out-named', ('--symmetry-cache', str(named)))
            self.assertEqual(cache, named.resolve())
            self.assertEqual(record['symmetry_cache_directory'], str(named.resolve()))
            # A control that was given a cache says that it is not independent of the run that filled it.
            self.assertIn(f'{reads} {named.resolve()}.', log)
            # The stand-in preparation wrote nothing, and the runner itself creates no cache directory.
            self.assertFalse(named.exists())
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_timing_record_names_the_boundary_and_keeps_its_check_out_of_the_scf(self):
        import os
        import shutil
        from pathlib import Path
        self.environment('0')
        directory = Path.cwd() / '.tmp' / f'runner-boundary-check-test-{os.getpid()}'
        plan = SimpleNamespace(order=16, minimum_order=9, tolerance=1.0e-3, atomic_tail=True)
        try:
            # The stand-in for run_scf takes 0.3 s, of which its details call 0.25 s the check.
            record, _, _ = self.run_serial_control(
                directory / 'checked', plan=plan, scf_seconds=0.3,
                details=(('hartree_boundary_kernel', 'full (GPU, full grid)'),
                         ('hartree_atomic_tail_values', 'host threads'),
                         ('hartree_boundary_check_seconds', '0.250000')))
            result = record['result']
            # What a comparison of two runs has to find equal, beside the kernels and the evaluator of the
            # tail, which may differ.
            self.assertEqual(result['hartree_boundary'], dict(
                multipole_order=16, solver_lpole=9, tolerance_ry=1.0e-3, atomic_tail=True,
                kernel='full (GPU, full grid)', atomic_tail_values='host threads'))
            self.assertTrue(any('result.hartree_boundary' in note for note in record['notes']))
            stack = result['wall_time_stack_seconds']
            self.assertEqual(result['hartree_boundary_check_seconds'], 0.25)
            self.assertEqual(stack['hartree_boundary_check_seconds'], 0.25)
            # The check is no part of the SCF time; it is finalization and a phase of its own.
            self.assertGreaterEqual(result['scf_seconds'], 0.)
            self.assertLess(result['scf_seconds'], 0.25)
            self.assertGreaterEqual(result['post_scf_finalization_reporting_seconds'], 0.25)
            self.assertAlmostEqual(sum(stack.values()), result['reported_program_seconds'], places=6)
            # A system without a plan and a run without the check.
            record, _, _ = self.run_serial_control(directory / 'plain')
            self.assertIsNone(record['result']['hartree_boundary'])
            self.assertEqual(record['result']['hartree_boundary_check_seconds'], 0.)
            self.assertEqual(
                record['result']['wall_time_stack_seconds']['hartree_boundary_check_seconds'], 0.)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_timing_record_holds_the_fields_and_notes_of_every_switch_of_one_round(self):
        import os
        import shutil
        from pathlib import Path
        from parsec_python.acceleration.Eigensolvers import distributed_state
        from parsec_python.acceleration.Occupations.symmetry_density import CuPySymmetryDensityBuilder
        self.environment('0')
        directory = Path.cwd() / '.tmp' / f'runner-round-record-test-{os.getpid()}'
        # A density builder as a run makes it, with no switch of the release named.
        with patch.dict('os.environ'):
            for name in ('PARSEC_CUPY_DENSITY_RELEASE', 'PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE',
                         'PARSEC_CUPY_DENSITY_COLLECTION', 'PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES'):
                os.environ.pop(name, None)
            builder = CuPySymmetryDensityBuilder(lambda *args: None)
        try:
            record, _, _ = self.run_serial_control(directory / 'out', orbital_density_builder=builder)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        (rank,) = record['per_rank']
        # The sphere and its rule, the creation of the contexts, the release after a density, the storage of
        # the states and the shared sectors: what one rank and the result record of them.
        for name in ('domain_rule_seconds', 'cuda_context_driver_devices', 'cuda_context_driver_seconds',
                     'density_pool_release', 'orbital_sector_state_storage', 'orbital_memory_allocator',
                     'sector_state_fit_bytes', 'distributed_state'):
            self.assertIn(name, rank)
        self.assertIn('domain', record['result'])
        # Of the release: how the pools of several devices are emptied, and whether those of a shared basis
        # are, with the densities after which they were left.
        release = rank['density_pool_release']
        self.assertEqual({name: release[name] for name in ('release', 'device_release_seconds', 'shared', 'shared_kept')},
                         dict(release='threads', device_release_seconds=0., shared='keep', shared_kept=0))
        # Of a shared sector: what a group holds from its start, as a rank of a run on several nodes records
        # it.  The slabs and their cut, the widest product of the projection, the seconds of the overlap and
        # the order of the copies on the way to the rows.

        class Owner:
            compact_finite_difference = SimpleNamespace(storage_mode='slot-major')
            _distributed_filter = SimpleNamespace(devices=(0, 1), owner=0, replica_seconds=0., replicas={1: SimpleNamespace(
                compact_finite_difference=SimpleNamespace(storage_mode='slot-major'))})

        owner = Owner()
        owner._sector_device_group = distributed_state.SectorDeviceGroup(owner, (0, 1))
        worker, _created = self.run_worker_rank(self.cupy(count=2), threading.Event(), (None, owner))
        (sector,) = worker['distributed_state']
        self.assertEqual({name: sector[name] for name in ('slab_cut', 'slab_columns', 'projection_columns',
                                                          'overlap_seconds', 'to_rows_copies', 'tall_blocks')},
                         dict(slab_cut=None, slab_columns=None, projection_columns=None, overlap_seconds=0.,
                              to_rows_copies=None, tall_blocks=None))
        self.assertEqual(set(sector['seconds']), set(distributed_state.SectorDeviceGroup.STAGES))
        # One note says how each is read.  They stand on neighbouring lines of the runner, written by
        # different changes: every one of them, once.
        for words in ('result.domain is the sphere of the run', 'per_rank[].domain_rule_seconds is what the default rule took',
                      'PARSEC_DOMAIN_REPORT=0', 'cuda_context_driver_devices lists the devices',
                      'PARSEC_CUDA_CONTEXT_CREATION=driver',
                      'release names how the pools of several devices are emptied (PARSEC_CUPY_DENSITY_RELEASE)',
                      'device_release_seconds are the sum of what the device threads took',
                      'sector_state_fit_bytes gives, per device', 'PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION',
                      'PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION', 'PARSEC_CUPY_SPILL_DOWNLOAD_ORDER',
                      'slab_columns is the widest slab', 'to_rows_copies is the order', 'PARSEC_CUPY_EXCHANGE_PAIRS',
                      'slab_cut says how that step cut them', 'PARSEC_CUPY_RITZ_GRAM_MULTIPLE columns (64)',
                      'projection_columns is the widest right-hand side', 'overlap_seconds is the part of seconds.gram',
                      'shared says whether the pools of the devices of a shared sector basis are emptied as well '
                      '(PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE)', 'shared_kept counts the densities'):
            with self.subTest(words=words):
                self.assertEqual(sum(words in note for note in record['notes']), 1)
        # The slab limit of two devices is named where the slabs are and where the storage rule counts them.
        self.assertEqual(sum('PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES' in note for note in record['notes']), 2)
        self.assertEqual(len(set(record['notes'])), len(record['notes']))

    def test_timing_record_says_what_the_bytes_counted_on_a_device_are(self):
        import os
        import shutil
        from pathlib import Path
        self.environment('0')
        directory = Path.cwd() / '.tmp' / f'runner-state-count-test-{os.getpid()}'
        try:
            record, _, _ = self.run_serial_control(directory / 'out')
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        # One note for sector_state_fit_bytes: what the count is held against, and that it is a lower limit
        # for a basis on one device only. Devices that shared a basis have sampled less than their count
        # outside their context.
        (note,) = [note for note in record['notes'] if 'sector_state_fit_bytes' in note]
        self.assertIn('PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION (0.978) of the memory of the device', note)
        self.assertIn('For a basis on one device it is a lower limit of what the device must hold', note)
        self.assertIn('For a shared basis it is what the rule that shares it counts', note)
        self.assertIn('a device can hold less outside its context', note)
        self.assertNotIn('. It is a lower limit', note)
        self.assertIn('null where no auto asked', note)
        # Null is also what an auto records that decided by the former rule, as that of a serial control
        # does whose launcher names neither value: the record of such a control asked, and counted nothing.
        self.assertIn('and where auto decided by the former rule that PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION '
                      'under environment names', note)
        self.assertIn('A serial control names 0.5 where its launcher names none, and '
                      'PARSEC_CUPY_SPILL_DOWNLOAD_ORDER=C beside it', note)
        control = {}
        self.runner._keep_former_routes(control)
        names = ('PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION', 'PARSEC_CUPY_SPILL_DOWNLOAD_ORDER')
        self.assertEqual([control[name] for name in names], ['0.5', 'C'])


class DriverContextTests(unittest.TestCase):
    """``_DriverContexts`` of the MPI runner: the calls it makes of the CUDA driver library."""

    addresses = ('0000:03:00.0', '0000:41:00.0', '0000:82:00.0', '0000:c1:00.0')

    def setUp(self):
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        self.runner = mpi_full_scf

    def driver(self, cp=None, **options):
        """The runner's class on a stand-in library, whose contexts are those of ``cp`` where one is given.

        ``self.made`` then says, for each context asked of the library, whether the library made it.
        """
        self.made = []

        def retain(device):
            self.made.append(device not in cp.created)
            cp.create(device)

        self.library = _CudaLibrary(self.addresses, retain=None if cp is None else retain, **options)
        return self.runner._DriverContexts(self.library)

    @staticmethod
    def cupy(addresses):
        """A stand-in for cupy whose devices report these PCI addresses."""
        cp = _ContextRecordingCuPy(count=len(addresses))
        device = cp.cuda.Device

        class Device(device):
            def __init__(self, index=None):
                super().__init__(index)
                self.pci_bus_id = addresses[self.id]

        cp.cuda.Device = Device
        return cp

    def test_context_is_retained_for_the_device_that_cupy_knows_by_the_index(self):
        # The addresses as a node of four A100 reports them, the last in capitals as the runtime writes it.
        cp = self.cupy(self.addresses[:3] + ('0000:C1:00.0',))
        driver = self.driver(cp)
        self.assertTrue(driver.start())
        self.assertEqual(self.library.calls, [('cuInit', 0)])
        for device in (0, 3, 1):
            self.assertTrue(driver.create(cp, device))
        # Per device: its handle from the index, the address of the handle, the context of the handle.
        self.assertEqual(self.library.calls[1:], [
            call for device in (0, 3, 1) for call in (
                ('cuDeviceGet', device), ('cuDeviceGetPCIBusId', 32, 100 + device),
                ('cuDevicePrimaryCtxRetain', 100 + device))])
        self.assertEqual((driver.devices, self.library.retained), ([0, 3, 1], [0, 3, 1]))
        self.assertEqual((sorted(cp.created), self.made), ([0, 1, 3], [True, True, True]))
        self.assertGreaterEqual(driver.seconds, 0.)
        # A context that exists is found: the call adds a reference and the device is listed again.
        self.assertTrue(driver.create(cp, 3))
        self.assertEqual((driver.devices, self.made[-1], cp.log.count(('context', 3))), ([0, 3, 1, 3], False, 1))

    def test_no_context_is_created_where_the_driver_numbers_the_devices_otherwise(self):
        # CuPy has the first two devices in the other order: their contexts are left to it.
        cp = self.cupy(('0000:41:00.0', '0000:03:00.0', '0000:82:00.0', '0000:c1:00.0'))
        driver = self.driver(cp)
        driver.start()
        self.assertEqual([driver.create(cp, device) for device in range(4)], [False, False, True, True])
        self.assertEqual((driver.devices, self.library.retained, sorted(cp.created)), ([2, 3], [2, 3], [2, 3]))
        # An index that the driver does not have.
        self.assertFalse(driver.create(cp, 4))
        self.assertEqual(self.library.calls[-1], ('cuDeviceGet', 4))

    def test_a_call_that_fails_creates_nothing_and_raises_nothing(self):
        steps = ('cuDeviceGet', 'cuDeviceGetPCIBusId', 'cuDevicePrimaryCtxRetain')
        for name in steps:
            for kind in ('failing', 'missing', 'faulting'):
                with self.subTest(call=name, kind=kind):
                    cp = self.cupy(self.addresses)
                    driver = self.driver(cp, **{kind: (name,)})
                    self.assertTrue(driver.start())
                    self.assertFalse(driver.create(cp, 1))
                    self.assertEqual((driver.devices, self.library.retained, cp.created), ([], [], {}))
                    # Nothing is asked of the library after the call that failed.
                    made = [call[0] for call in self.library.calls[1:]]
                    self.assertEqual(made, list(steps[:steps.index(name) + (kind != 'missing')]))
        # A driver that does not start: CuPy's first call reports why, and nothing is created here.
        for kind in ('failing', 'missing', 'faulting'):
            with self.subTest(call='cuInit', kind=kind):
                driver = self.driver(**{kind: ('cuInit',)})
                self.assertFalse(driver.start())
                self.assertEqual(driver.devices, [])

    def test_a_library_that_cannot_be_loaded_creates_nothing(self):
        asked = []

        def absent(name):
            asked.append(name)
            raise OSError(f'{name}: cannot open shared object file')

        for system, name in (('nt', 'nvcuda.dll'), ('posix', 'libcuda.so.1')):
            with self.subTest(system=system):
                with patch.object(self.runner.os, 'name', system), patch.object(self.runner.ctypes, 'CDLL', absent):
                    driver = self.runner._DriverContexts()
                self.assertEqual(asked, [name])
                asked.clear()
                self.assertIsNone(driver.library)
                cp = self.cupy(self.addresses)
                self.assertFalse(driver.start())
                self.assertFalse(driver.create(cp, 0))
                self.assertEqual((driver.devices, cp.created, cp.log), ([], {}, []))

    def test_contexts_made_through_the_library_serve_the_calls_of_cupy(self):
        # ``_create_contexts`` with the class and a library: the driver is started before the device count
        # is asked, every context exists before CuPy selects its device, and the records are those of
        # the calls of CuPy alone.
        records = {}
        for through_driver in (False, True):
            cp = self.cupy(self.addresses)
            driver, started = None, []
            if through_driver:
                driver = self.driver(cp)
                count = cp.cuda.runtime.getDeviceCount
                cp.cuda.runtime.getDeviceCount = lambda: (started.append(list(self.library.calls)), count())[1]
            records[through_driver] = self.runner._create_contexts(cp, (0, 1, 2, 3), driver)
            self.assertEqual(cp.log, [event for device in range(4)
                                      for event in (('context', device), ('memory', device))])
            if through_driver:
                self.assertEqual(started, [[('cuInit', 0)]])
                # Each context was made by the library, before CuPy asked for the memory of its device.
                self.assertEqual((driver.devices, self.made), ([0, 1, 2, 3], [True] * 4))
        self.assertEqual(records[True], records[False])
        self.assertEqual([item['pci_bus_id'] for item in records[True]], list(self.addresses))

    @unittest.skipUnless(cupy_available(), 'a real CUDA device is required')
    def test_cupy_uses_the_contexts_that_the_driver_library_made_on_a_real_device(self):
        # A new process each, so that its first context is the one the switch asks for: the records of
        # the device are the same, and an array and a kernel of CuPy work on the context of either.
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        code = '\n'.join((
            'import json, sys',
            'import cupy as cp',
            'from parsec_python.acceleration.benchmarks import mpi_full_scf as runner',
            'memory = []',
            'cp, prepared, records, sampler, contexts = runner._prepare_on_devices(',
            '    cp, (0,), lambda: sum(range(100000)), memory)',
            'sampler.stop(records)',
            'total = float((cp.arange(8, dtype=cp.float64) ** 2).sum())',
            'print(json.dumps(dict(records=records, contexts=contexts, total=total, prepared=prepared,',
            '                      current=int(cp.cuda.Device().id))))'))
        source = str(Path(__file__).resolve().parents[3])
        outcomes = {}
        for creation in ('driver', 'runtime'):
            environment = {name: value for name, value in os.environ.items()
                           if name not in ('PARSEC_OVERLAP_CUDA_CONTEXTS', 'PARSEC_CUPY_DEVICES')}
            environment.update(PARSEC_CUDA_CONTEXT_CREATION=creation,
                               PYTHONPATH=os.pathsep.join(filter(None, (source, environment.get('PYTHONPATH')))))
            completed = subprocess.run([sys.executable, '-c', code], env=environment, check=False,
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outcomes[creation] = json.loads(completed.stdout.splitlines()[-1])
        for creation, outcome in outcomes.items():
            with self.subTest(creation=creation):
                self.assertEqual((outcome['total'], outcome['prepared'], outcome['current']),
                                 (140., sum(range(100000)), 0))
                self.assertTrue(outcome['contexts']['overlapped'])
                self.assertEqual(outcome['contexts']['driver_devices'], [0] if creation == 'driver' else [])
                self.assertEqual(outcome['contexts']['driver_seconds'] > 0., creation == 'driver')
        # The device is described alike; the memory free at that moment is that of a shared device.
        for outcome in outcomes.values():
            for record in outcome['records']:
                del record['initial_free_bytes']
        self.assertEqual(outcomes['driver']['records'], outcomes['runtime']['records'])

    @unittest.skipUnless(cupy_available(), 'a real CUDA device is required')
    def test_the_thread_of_a_device_empties_the_pool_of_a_context_that_the_driver_library_made(self):
        # After a density the pool of a device is emptied by the thread of that device
        # (PARSEC_CUPY_DENSITY_RELEASE=threads), which is neither the thread that created the context nor the
        # one that computes.  A new process each, whose context of the device is the one the switch asks for
        # and whose context thread has ended: the device thread computes on the device and returns the
        # blocks that it and the main thread left in the pool, and the array of the main thread stays.
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        code = '\n'.join((
            'import json, threading',
            'import cupy as cp',
            'from parsec_python.acceleration.benchmarks import mpi_full_scf as runner',
            'from parsec_python.acceleration.Eigensolvers.distributed_filter import _pool',
            'cp, prepared, records, sampler, contexts = runner._prepare_on_devices(cp, (0,), lambda: None, [])',
            'sampler.stop(records)',
            'pool = cp.get_default_memory_pool()',
            'kept = cp.arange(1 << 20, dtype=cp.float64)',
            'freed = cp.ones(1 << 23, dtype=cp.float64)',
            'del freed',
            'cached = int(pool.total_bytes()) - int(pool.used_bytes())',
            'def empty(device):',
            '    with cp.cuda.Device(device):',
            '        mine = float(cp.arange(8, dtype=cp.float64).sum())',
            '        pool.free_all_blocks()',
            '    return threading.current_thread().name, mine',
            'name, mine = _pool(0).submit(empty, 0).result(timeout=60)',
            'print(json.dumps(dict(contexts=contexts, cached=cached, name=name, mine=mine,',
            '                      left=int(pool.total_bytes()) - int(pool.used_bytes()), used=int(pool.used_bytes()),',
            '                      kept=float(kept.sum()), main=threading.current_thread().name,',
            '                      alive=[thread.name for thread in threading.enumerate()])))'))
        source = str(Path(__file__).resolve().parents[3])
        outcomes = {}
        for creation in ('driver', 'runtime'):
            environment = {name: value for name, value in os.environ.items()
                           if name not in ('PARSEC_OVERLAP_CUDA_CONTEXTS', 'PARSEC_CUPY_DEVICES')}
            environment.update(PARSEC_CUDA_CONTEXT_CREATION=creation,
                               PYTHONPATH=os.pathsep.join(filter(None, (source, environment.get('PYTHONPATH')))))
            completed = subprocess.run([sys.executable, '-c', code], env=environment, check=False,
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outcomes[creation] = json.loads(completed.stdout.splitlines()[-1])
        for creation, outcome in outcomes.items():
            with self.subTest(creation=creation):
                self.assertEqual(outcome['contexts']['driver_devices'], [0] if creation == 'driver' else [])
                # The thread of the device, alive after its task, and no thread that created contexts.
                self.assertTrue(outcome['name'].startswith('orbital-shard-0'))
                self.assertNotEqual(outcome['name'], outcome['main'])
                self.assertIn(outcome['name'], outcome['alive'])
                self.assertEqual(len(outcome['alive']), 2)
                # The 64 MiB that the main thread had freed were cached and went back; its array stayed.
                self.assertGreaterEqual(outcome['cached'], 1 << 26)
                self.assertEqual((outcome['left'], outcome['used']), (0, 8 << 20))
                self.assertEqual((outcome['kept'], outcome['mine']), (float(sum(range(1 << 20))), 28.))
        for name in ('cached', 'left', 'used', 'kept', 'mine', 'name'):
            self.assertEqual(outcomes['driver'][name], outcomes['runtime'][name])


@unittest.skipUnless(cupy_available() and native_available(),
                     'a real CUDA device and the native extension are required')
class RealRolePreparationTests(unittest.TestCase):
    """Complete SCF with two ranks run as threads sharing one CUDA device."""

    def tearDown(self):
        # Prepared systems that a test left in a reference cycle are destroyed here, between the tests, and not
        # by a collection inside the next one.
        gc.collect()

    def run_two_ranks(self, problem, *, complete_worker):
        from parsec_python.acceleration.driver import prepare_single_point, run_scf
        world = _World(2)
        def rank_main(rank):
            context = MPISectorContext(_Comm(world, rank))
            if rank and complete_worker:
                # A rank that takes itself for the root prepares every
                # component; that run is the reference for the reduced one.
                context.root = rank
            system = prepare_single_point(problem, backend='auto', mpi_context=context)
            context.root = 0
            system.materialize_final_wavefunctions = False
            solver = system.backend.symmetry_eigensolver
            if rank:
                context.worker_loop(solver, system.orbital_density_builder)
                return system
            system.orbital_density_builder = MPISymmetryDensityBuilder(
                system.orbital_density_builder, context, solver)
            try:
                return system, run_scf(system)
            finally:
                context.stop_workers()
        outcomes = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(rank_main, rank) for rank in range(2)]:
                try:
                    outcomes.append(future.result(timeout=300))
                except Exception as error:
                    outcomes.append(error)
        errors = [item for item in outcomes if isinstance(item, Exception)]
        if errors:
            # The rank that failed first leaves the other one waiting in a
            # collective; report the cause rather than that timeout.
            raise next((error for error in errors if not isinstance(error, TimeoutError)), errors[0])
        (root, result), worker = outcomes
        return result, root, worker

    def test_reduced_worker_preparation_leaves_the_scf_bitwise_unchanged(self):
        from pathlib import Path
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration.driver import prepare_single_point, run_scf
        source = Path(__file__).resolve().parents[2] / 'tests' / 'data' / 'H_cli_smoke.in'
        problem = parse_parsec_input(source).problem
        # Three steps reach the later-step filter and the trimmed sector states.
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        with patch.dict('os.environ', PARSEC_CUPY_DEVICES='0'):
            # The one-process run also compiles every kernel, so that no rank
            # thread waits long for the other inside a collective.
            serial = run_scf(prepare_single_point(problem, backend='auto'))
            complete, _, complete_worker = self.run_two_ranks(problem, complete_worker=True)
            reduced, root, worker = self.run_two_ranks(problem, complete_worker=False)

        # The worker skipped every component of the SCF loop ...
        self.assertIsNone(worker.reference.ionic_potential)
        self.assertIsNone(worker.reference.initial_density)
        self.assertIsNone(worker.reference.core_density)
        self.assertIsNone(worker.backend.symmetry_eigensolver.totally_symmetric_stencil)
        self.assertIsNone(worker.backend.symmetry_eigensolver.totally_symmetric_negative_laplacian)
        self.assertFalse(hasattr(worker.backend, 'native_poisson_solver'))
        self.assertIn('mpi_rank_preparation', dict(worker.backend_info.details))
        with self.assertRaisesRegex(RuntimeError, 'MPI root rank only'):
            worker.solve_hartree(np.zeros(worker.grid.size))
        # ... which a complete preparation and the root do build.
        self.assertIsNotNone(complete_worker.reference.ionic_potential)
        self.assertTrue(hasattr(complete_worker.backend, 'native_poisson_solver'))
        self.assertIsNotNone(root.reference.ionic_potential)
        self.assertTrue(hasattr(root.backend, 'native_poisson_solver'))
        self.assertNotIn('mpi_rank_preparation', dict(root.backend_info.details))
        self.assertEqual(sorted(root.backend.symmetry_eigensolver._owned_representations
                                + worker.backend.symmetry_eigensolver._owned_representations),
                         list(range(root.backend.symmetry_eigensolver.representation_count)))

        # Same sector solves, same reductions: not one bit may move.
        self.assertEqual(reduced.iterations, complete.iterations)
        np.testing.assert_array_equal(reduced.eigenvalues, complete.eigenvalues)
        np.testing.assert_array_equal(reduced.occupations, complete.occupations)
        np.testing.assert_array_equal(reduced.density, complete.density)
        np.testing.assert_array_equal(reduced.hartree_potential, complete.hartree_potential)
        self.assertEqual(reduced.energies.total, complete.energies.total)
        for actual, expected in zip(reduced.history, complete.history, strict=True):
            np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
        # Against one process only the order of the density reduction differs.
        self.assertEqual(reduced.iterations, serial.iterations)
        np.testing.assert_allclose(reduced.eigenvalues, serial.eigenvalues, rtol=0, atol=1e-9)
        np.testing.assert_allclose(reduced.density, serial.density, rtol=1e-9, atol=1e-11)
        self.assertAlmostEqual(reduced.energies.total, serial.energies.total, places=8)


if __name__ == '__main__':
    unittest.main()
