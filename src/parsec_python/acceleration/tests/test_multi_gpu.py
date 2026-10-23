"""Device residency, export exactness, and bounded multi-GPU scheduling."""
from concurrent.futures import ThreadPoolExecutor
import gc
import os
from threading import Barrier, Lock, local
from time import sleep
from types import MethodType, SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import numpy as np

from parsec_python.acceleration.backends.cupy import CuPyTimingStats, cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalResult
from parsec_python.acceleration.Eigensolvers.symmetry import (
    CuPySymmetryEigvalState, CuPySymmetryOrbitals, CuPySymmetrySCFEigensolver,
)
from parsec_python.acceleration.Occupations import symmetry_density
from parsec_python.acceleration.Occupations.device_density import CuPyDeviceDensityBuilder
from parsec_python.acceleration.Occupations.symmetry_density import CuPySymmetryDensityBuilder
from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.acceleration.SCF.single_point import run_scf


class AllocatorCleanupTests(unittest.TestCase):
    def test_allocator_restored_after_scf_or_export_failure(self):
        for stage in ("run_reference_scf", "_finalize_result"):
            with self.subTest(stage=stage):
                restore = Mock()
                system = SimpleNamespace(
                    backend=SimpleNamespace(symmetry_eigensolver=SimpleNamespace(
                        restore_memory_allocator=restore)),
                    eigenproblem_solver=None, orbital_density_builder=None,
                    mixer_factory=None, residual_metrics_evaluator=None,
                    total_energy_evaluator=None, scalar_field_adapter=None,
                )
                module = "parsec_python.acceleration.SCF.single_point."
                with patch(module + "run_reference_scf"), patch(module + "_finalize_result"):
                    with patch(module + stage, side_effect=MemoryError("injected OOM")):
                        with self.assertRaisesRegex(MemoryError, "injected OOM"):
                            run_scf(system)
                restore.assert_called_once_with()


class SectorLifetimeTests(unittest.TestCase):
    def test_consumed_snapshot_does_not_pin_replaced_orbitals(self):
        vectors = np.zeros((100, 20))
        reference = weakref.ref(vectors)
        owner = SimpleNamespace(state=SimpleNamespace(vectors=vectors))
        snapshot = CuPySymmetryEigvalState(20, (20,), (owner.state,), 3)
        del vectors
        snapshot.release_sector_snapshots()
        self.assertIs(reference(), owner.state.vectors)
        owner.state = None
        gc.collect()
        self.assertIsNone(reference(), "a completed sector must release its obsolete orbitals")
        self.assertEqual((snapshot.requested_states, snapshot.sector_state_counts,
                          snapshot.solves_completed), (20, (20,), 3))
        snapshot.release_sector_snapshots()
        self.assertEqual(snapshot.sector_states, (None,))


class DeviceBatchTests(unittest.TestCase):
    def make_scheduler(self):
        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._sector_device_ids = (0, 1, 0, 1)
        solver._serial_per_device = True
        solver._bound_executor = None
        solver._executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(solver._executor.shutdown, wait=True)
        solver._sector_timing_stats = [CuPyTimingStats() for _ in range(4)]
        solver.timing_stats = CuPyTimingStats()
        solver.scheduler_batches = 0
        solver.scheduler_wall_seconds = 0.0
        return solver

    def test_uneven_sectors_overlap_devices_but_not_workspaces(self):
        solver = self.make_scheduler()
        lock, barrier = Lock(), Barrier(2)
        active = [0, 0]
        maxima = [0, 0]
        total_active = []

        def run(this, representation, *_args, **_kwargs):
            device = this._sector_device_ids[representation]
            with lock:
                active[device] += 1
                maxima[device] = max(maxima[device], active[device])
                total_active.append(sum(active))
            if representation < 2:
                barrier.wait(timeout=2)
            sleep(0.04 if representation == 0 else 0.003)
            with lock:
                active[device] -= 1
            return representation

        solver._run_one_sector = MethodType(run, solver)
        result = solver._run_sector_jobs((0, 1, 2, 3), [1]*4, EigvalSettings(), reset=True)
        self.assertEqual(list(result), [0, 1, 2, 3])
        self.assertEqual(list(result.values()), [0, 1, 2, 3])
        self.assertEqual(maxima, [1, 1])
        self.assertEqual(max(total_active), 2)

    def test_failure_waits_for_other_device_before_returning(self):
        solver = self.make_scheduler()
        completed = []
        barrier = Barrier(2)

        def run(_this, representation, *_args, **_kwargs):
            if representation < 2:
                barrier.wait(timeout=2)
            if representation == 0:
                raise RuntimeError("injected device failure")
            sleep(0.02)
            completed.append(representation)
            return representation

        solver._run_one_sector = MethodType(run, solver)
        with self.assertRaisesRegex(RuntimeError, "injected device failure"):
            solver._run_sector_jobs((0, 1, 2, 3), [1]*4, EigvalSettings(), reset=True)
        self.assertEqual(completed, [1, 3])

    def test_bound_phase_finishes_before_any_sector_and_keeps_results_ordered(self):
        solver = self.make_scheduler()
        solver._precompute_bounds = True
        bounds = []

        def prepare(_this, representation, count, _settings, *, reset):
            self.assertFalse(reset)
            self.assertEqual(count, representation+1)
            bounds.append(representation)
            return ('bound', representation)

        def solve(_this, representation, *_args, spectral_bound, **_kwargs):
            self.assertEqual(bounds, [0, 1, 2, 3])
            self.assertEqual(spectral_bound, ('bound', representation))
            return representation

        solver._run_one_bound = MethodType(prepare, solver)
        solver._run_one_sector = MethodType(solve, solver)
        result = solver._run_sector_jobs((0, 1, 2, 3), [1, 2, 3, 4], EigvalSettings(), reset=False)
        self.assertEqual(list(result.values()), [0, 1, 2, 3])
        self.assertGreater(solver.timing_stats.eigensolver_bound_prepare_wall_seconds, 0)

    def test_bound_failure_prevents_sector_submission(self):
        solver = self.make_scheduler()
        solver._precompute_bounds = True
        solver._run_one_bound = Mock(side_effect=RuntimeError('injected bound failure'))
        solver._run_one_sector = Mock()
        with self.assertRaisesRegex(RuntimeError, 'injected bound failure'):
            solver._run_sector_jobs((0, 1, 2, 3), [1]*4, EigvalSettings(), reset=True)
        solver._run_one_sector.assert_not_called()


class SectorPoolReleaseTests(unittest.TestCase):
    """A finished sector returns the unused pool blocks of its stream (PARSEC_CUPY_SECTOR_POOL_RELEASE)."""

    def make_solver(self, release=(0, 1), spill=False, counts=(64,)*4, owned=(0, 1, 2, 3)):
        """Sectors 0 and 2 on device 0, 1 and 3 on device 1; a stand-in for cupy that records what happens.

        A sector has 2**20 rows, so that a column of its vectors takes 8 MiB; ``counts`` are the columns of the four
        sectors and ``owned`` the sectors that this process solves.
        """
        events, lock, place = [], Lock(), local()

        def devices():
            if not hasattr(place, "devices"):
                place.devices = []
            return place.devices

        class Device:
            def __init__(self, index):
                self.id = index

            def __enter__(self):
                devices().append(self.id)
                return self

            def __exit__(self, *_error):
                devices().pop()

        class Stream:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                place.stream = self.name
                return self

            def __exit__(self, *_error):
                place.stream = None

        held = {0: 5 << 30, 1: 3 << 30}

        class Pool:
            def total_bytes(self):
                return held[devices()[-1]]

            def free_all_blocks(self, stream=None):
                with lock:
                    events.append(("release", devices()[-1], stream.name, place.stream))
                held[devices()[-1]] -= 1 << 30

        def sector(index):
            def solve(count, settings, spectral_bound=None):
                with lock:
                    events.append(("solve", index, devices()[-1], place.stream))
                if index in self.failing:
                    raise RuntimeError("injected sector failure")
                vectors = SimpleNamespace(place="device")
                return CuPyEigvalResult(np.zeros(count), vectors, None, "device state", "subspace",
                                        False, None, count, count, 0.0)

            def offload():
                with lock:
                    events.append(("offload", index))
                return SimpleNamespace(subspace=SimpleNamespace(vectors=SimpleNamespace(place="host")))

            return SimpleNamespace(solve=solve, reset=lambda: None, restore_state_to_device=lambda: None,
                                   offload_state_to_host=offload)

        self.failing = set()
        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._sector_device_ids = (0, 1, 0, 1)
        solver._streams = [Stream(f"stream {index}") for index in range(4)]
        solver._solvers = [sector(index) for index in range(4)]
        solver._pool_releases = {device: [0, 0] for device in release}
        solver.decomposition = SimpleNamespace(sector_size=lambda representation: 1 << 20)
        solver._owned_representations = tuple(owned)
        solver._sector_counts = list(counts)
        solver._host_spill_sector_states = spill
        solver._serial_per_device = True
        solver._bound_executor = None
        solver._executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(solver._executor.shutdown, wait=True)
        solver._sector_timing_stats = [CuPyTimingStats() for _ in range(4)]
        solver.timing_stats = CuPyTimingStats()
        solver.scheduler_batches = 0
        solver.scheduler_wall_seconds = 0.0
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=Device), get_default_memory_pool=Pool)
        for patcher in (patch("parsec_python.acceleration.Eigensolvers.symmetry.require_cupy", return_value=(cp, None)),
                        patch.dict("os.environ")):
            patcher.start()
            self.addCleanup(patcher.stop)
        # The limit of the density step at its default, whatever the environment of the test run names.
        os.environ.pop("PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES", None)
        return solver, events

    def test_release_follows_the_solve_on_the_device_and_stream_of_the_sector(self):
        solver, events = self.make_solver()
        result = solver._run_one_sector(2, 3, EigvalSettings(), reset=False)
        self.assertEqual(result.vectors.place, "device")
        # The blocks of the sector's own stream, with its device and that stream current.
        self.assertEqual(events, [("solve", 2, 0, "stream 2"), ("release", 0, "stream 2", "stream 2")])
        self.assertEqual(solver.sector_pool_releases, {0: dict(releases=1, bytes=1 << 30), 1: dict(releases=0, bytes=0)})

    def test_every_sector_of_a_device_releases_before_the_next_one_starts(self):
        solver, events = self.make_solver()
        solver._run_sector_jobs((0, 1, 2, 3), solver._sector_counts, EigvalSettings(), reset=True)
        for device, first, second in ((0, 0, 2), (1, 1, 3)):
            here = [event for event in events if event[2 if event[0] == "solve" else 1] == device]
            self.assertEqual(here, [("solve", first, device, f"stream {first}"),
                                    ("release", device, f"stream {first}", f"stream {first}"),
                                    ("solve", second, device, f"stream {second}"),
                                    ("release", device, f"stream {second}", f"stream {second}")])
        self.assertEqual(solver.sector_pool_releases, {0: dict(releases=2, bytes=2 << 30), 1: dict(releases=2, bytes=2 << 30)})

    def test_only_the_named_devices_release_and_a_failed_sector_does_not(self):
        solver, events = self.make_solver(release=(1,))
        solver._run_one_sector(0, 1, EigvalSettings(), reset=False)
        solver._run_one_sector(3, 1, EigvalSettings(), reset=False)
        self.assertEqual([event for event in events if event[0] == "release"], [("release", 1, "stream 3", "stream 3")])
        self.assertEqual(solver.sector_pool_releases, {1: dict(releases=1, bytes=1 << 30)})
        self.failing = {1}
        with self.assertRaisesRegex(RuntimeError, "injected sector failure"):
            solver._run_one_sector(1, 1, EigvalSettings(), reset=False)
        self.assertEqual(solver.sector_pool_releases, {1: dict(releases=1, bytes=1 << 30)})
        # No device releases: the former route, also for a solver built before the setting existed.
        for clear in (lambda this: this._pool_releases.clear(), lambda this: delattr(this, "_pool_releases")):
            solver, events = self.make_solver()
            clear(solver)
            solver._run_sector_jobs((0, 1, 2, 3), solver._sector_counts, EigvalSettings(), reset=True)
            self.assertEqual(sorted(event[1] for event in events), [0, 1, 2, 3])
            self.assertTrue(all(event[0] == "solve" for event in events))
            self.assertEqual(solver.sector_pool_releases, {})

    def released(self, counts, sector=0, **options):
        """Releases of one solve of ``sector`` where the four sectors have ``counts`` columns."""
        solver, events = self.make_solver(counts=counts, **options)
        solver._run_one_sector(sector, counts[sector], EigvalSettings(), reset=False)
        count = sum(event[0] == "release" for event in events)
        device = solver._sector_device_ids[sector]
        self.assertEqual(solver.sector_pool_releases[device], dict(releases=count, bytes=count << 30))
        return count

    def test_sectors_release_from_the_bytes_at_which_the_density_step_empties_the_pool(self):
        # The density step sums the vectors of every sector of the process against 512 MiB, 64 columns here: four
        # sectors of 128 MiB reach it together, and their blocks would go back in the same SCF step anyway.
        self.assertEqual(self.released((16, 16, 16, 16)), 1)
        self.assertEqual(self.released((16, 16, 16, 16), sector=3), 1)
        self.assertEqual(self.released((1, 1, 1, 61)), 1)
        # One column fewer, and the density step leaves the pool: the blocks stay and serve the next SCF step.
        self.assertEqual(self.released((16, 16, 16, 15)), 0)
        self.assertEqual(self.released((60, 1, 1, 1)), 0)
        # A rank is handed the vectors of its own sectors at the density step, and counts no others.
        self.assertEqual(self.released((16, 64, 16, 64), owned=(0, 2)), 0)
        self.assertEqual(self.released((32, 1, 32, 1), owned=(0, 2)), 1)
        # The limit is the one of the density step, wherever that has been moved.
        for setting, counts, released in (("1073741824", (32,)*4, 1), ("1073741825", (32,)*4, 0), ("0", (1,)*4, 1)):
            with self.subTest(setting=setting):
                solver, events = self.make_solver(counts=counts)
                os.environ["PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES"] = setting
                solver._run_one_sector(3, counts[3], EigvalSettings(), reset=False)
                self.assertEqual(sum(event[0] == "release" for event in events), released)
        # Before the first solve has named the columns there is nothing to compare.
        for clear in (lambda this: setattr(this, "_sector_counts", None), lambda this: delattr(this, "_sector_counts")):
            solver, events = self.make_solver()
            clear(solver)
            self.assertEqual(solver._sector_vector_bytes(), 0)
            solver._release_sector_blocks(0)
            self.assertEqual(events, [])
        self.assertEqual(self.make_solver(counts=(1, 2, 3, 4), owned=(1, 3))[0]._sector_vector_bytes(), 6 * 8 << 20)

    def test_a_state_spilled_to_the_host_leaves_the_device_before_the_release(self):
        solver, events = self.make_solver(spill=True)
        result = solver._run_one_sector(1, 2, EigvalSettings(), reset=False)
        self.assertEqual(result.vectors.place, "host")
        self.assertEqual([event[0] for event in events], ["solve", "offload", "release"])
        # The vectors count where they lie on the host as well: below the limit of the density step nothing goes back.
        solver, events = self.make_solver(spill=True, counts=(15, 16, 16, 16))
        solver._run_one_sector(1, 16, EigvalSettings(), reset=False)
        self.assertEqual([event[0] for event in events], ["solve", "offload"])

    def test_the_release_and_the_density_step_read_one_limit(self):
        from parsec_python.acceleration.Eigensolvers import symmetry
        self.assertIs(symmetry_density._pool_release_orbital_bytes, symmetry._pool_release_orbital_bytes)
        orbitals = SimpleNamespace(sector_vectors=(SimpleNamespace(nbytes=1 << 30),))
        # A value that is no nonnegative integer is the same error of both, and nothing goes back before it.
        for setting, words in (("-1", "must be nonnegative"), ("512M", "must be a nonnegative integer"),
                               ("", "must be a nonnegative integer")):
            with self.subTest(setting=setting):
                solver, events = self.make_solver()
                builder = CuPySymmetryDensityBuilder(lambda *arguments: None)
                os.environ["PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES"] = setting
                message = "PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES " + words
                with self.assertRaisesRegex(ValueError, message):
                    solver._run_one_sector(0, 64, EigvalSettings(), reset=False)
                self.assertEqual([event[0] for event in events], ["solve"])
                self.assertEqual(solver.sector_pool_releases[0], dict(releases=0, bytes=0))
                with self.assertRaisesRegex(ValueError, message):
                    builder._release_large_unused_pool(orbitals)
                self.assertEqual(builder.pool_release["calls"], 0)
                # The density step first uses it after a complete eigensolve: its builder refuses it when built.
                with self.assertRaisesRegex(ValueError, message):
                    CuPySymmetryDensityBuilder(lambda *arguments: None)
        # A device that releases nothing does not read it.
        solver, events = self.make_solver(release=())
        os.environ["PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES"] = "-1"
        solver._run_one_sector(0, 64, EigvalSettings(), reset=False)
        self.assertEqual([event[0] for event in events], ["solve"])


class PoolFromSectorsToDensityTests(unittest.TestCase):
    """The pool of two devices from the sectors they solve in turn to the density step.

    A stand-in keeps bytes in use per device and free blocks per device and stream, as CuPy's pool does; the
    release of a finished sector, the collection before the pool release and that release are the solver's and the
    density builder's own.
    """

    ROWS, BUFFER, SUMS = 1 << 20, 1 << 30, 8 << 20

    def run_steps(self, release, counts=(16,)*4, steps=4):
        """``steps`` SCF steps of four sectors on two devices.

        Returns the solver, the density builder, the collections that were asked for, the largest reservation of
        the pool on each device, and the free blocks that each density step left.
        """
        class Place(local):
            # The current device and stream belong to the thread, as CuPy's do: the density step empties
            # the pools of the two devices by a thread each.
            device = stream = None

        place, lock = Place(), Lock()
        used, free, peak, collections, left = {0: 0, 1: 0}, {}, {0: 0, 1: 0}, [], []

        class Device:
            def __init__(self, index):
                self.id = index

            def __enter__(self):
                self.previous, place.device = place.device, self.id
                return self

            def __exit__(self, *_error):
                place.device = self.previous

        class Stream:
            def __enter__(self):
                place.stream = self
                return self

            def __exit__(self, *_error):
                place.stream = None

        class Array:
            def __init__(self, device, nbytes):
                self.device, self.nbytes = Device(device), nbytes

        class Pool:
            def malloc(self, size):
                raise AssertionError("not called")

            def used_bytes(self):
                return used[place.device]

            def total_bytes(self):
                return used[place.device] + sum(sum(sizes) for (device, _), sizes in free.items() if device == place.device)

            def free_all_blocks(self, stream=None):
                with lock:
                    for key in [key for key in free if key[0] == place.device and stream in (None, key[1])]:
                        del free[key]

        pool, pinned = Pool(), SimpleNamespace(free_all_blocks=lambda: None)

        def take(size):
            """A block for the current device and stream: one that their free list holds, else one from the driver."""
            cached = free.get((place.device, place.stream), [])
            if size in cached:
                cached.remove(size)
            used[place.device] += size
            peak[place.device] = max(peak[place.device], pool.total_bytes())

        def give(size):
            used[place.device] -= size
            free.setdefault((place.device, place.stream), []).append(size)

        vectors = {}

        def sector(index):
            def solve(count, settings, spectral_bound=None):
                # The vectors stay from the first solve on; every solve takes a Ritz buffer and gives it back.
                if index not in vectors:
                    take(8 * self.ROWS * count)
                    vectors[index] = Array(place.device, 8 * self.ROWS * count)
                take(self.BUFFER)
                give(self.BUFFER)
                return CuPyEigvalResult(np.zeros(count), vectors[index], None, "device state", "subspace",
                                        False, None, count, count, 0.0)

            return SimpleNamespace(solve=solve)

        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._sector_device_ids = (0, 1, 0, 1)
        solver._streams = [Stream() for _ in range(4)]
        solver._solvers = [sector(index) for index in range(4)]
        solver._pool_releases = {device: [0, 0] for device in ((0, 1) if release else ())}
        solver.decomposition = SimpleNamespace(sector_size=lambda representation: self.ROWS)
        solver._owned_representations = (0, 1, 2, 3)
        solver._sector_counts = list(counts)
        solver._host_spill_sector_states = False
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=Device, get_allocator=lambda: pool.malloc), ndarray=Array,
                             get_default_memory_pool=lambda: pool, get_default_pinned_memory_pool=lambda: pinned)

        def collect(generation=None):
            collections.append(generation)
            return 0

        with patch("parsec_python.acceleration.Eigensolvers.symmetry.require_cupy", return_value=(cp, None)), \
             patch("parsec_python.acceleration.backends.cupy.require_cupy", return_value=(cp, None)), \
             patch.object(symmetry_density.gc, "collect", collect), patch.dict("os.environ"):
            for name in ("PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES", "PARSEC_CUPY_DENSITY_COLLECTION"):
                os.environ.pop(name, None)
            builder = CuPySymmetryDensityBuilder(lambda *arguments: None)
            for _ in range(steps):
                for index in (0, 2, 1, 3):
                    solver._run_one_sector(index, counts[index], EigvalSettings(), reset=False)
                # The sums of a density come and go on the default stream of each device.
                for device in (0, 1):
                    with Device(device):
                        take(self.SUMS)
                        give(self.SUMS)
                builder._release_large_unused_pool(SimpleNamespace(sector_vectors=tuple(vectors[index] for index in range(4))))
                left.append({key: sizes for key, sizes in free.items() if sizes})
        return solver, builder, collections, peak, left

    def test_a_device_holds_one_buffer_and_the_density_step_still_collects_young(self):
        held = 2 * 8 * self.ROWS * 16
        for release in (True, False):
            with self.subTest(release=release):
                solver, builder, collections, peak, left = self.run_steps(release)
                # The second sector of a device finds no block in the free list of its own stream: with the
                # release it takes the bytes that the first one returned, without it a second buffer, and the
                # sums of the density come on top of both.
                most = held + self.BUFFER if release else held + 2 * self.BUFFER + self.SUMS
                self.assertEqual(peak, {0: most, 1: most})
                self.assertEqual(solver.sector_pool_releases, {
                    device: dict(releases=8, bytes=8 * self.BUFFER) for device in ((0, 1) if release else ())})
                # A release returns blocks that nothing uses.  The bytes in use are what they were, so the density
                # step collects in full twice and the young generations afterwards, as without the release, ...
                self.assertEqual(collections, [None, None, 1, 1])
                record = builder.pool_release
                self.assertEqual((record["calls"], record["full_collections"], record["young_collections"]), (4, 2, 2))
                # ... and leaves the free list of no stream behind: the next Ritz step starts from the vectors.
                self.assertEqual(left, [{}] * 4)

    def test_below_the_limit_neither_returns_anything(self):
        # One column short of the 512 MiB from which the density step empties the pool: the blocks stay for the
        # next SCF step on both sides, and the second buffer with them.
        solver, builder, collections, peak, left = self.run_steps(True, counts=(16, 16, 16, 15))
        self.assertEqual(solver.sector_pool_releases, {device: dict(releases=0, bytes=0) for device in (0, 1)})
        self.assertEqual((collections, builder.pool_release["calls"]), ([], 0))
        self.assertEqual(peak[0], 2 * 8 * self.ROWS * 16 + 2 * self.BUFFER + self.SUMS)
        self.assertTrue(all(len(blocks) == 6 for blocks in left))


class SectorStateStorageTests(unittest.TestCase):
    """``auto`` of PARSEC_CUPY_SECTOR_STATE_STORAGE and of PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR follows what fits.

    The sectors are those of first runs on A100 nodes as their reports give them
    (``orbital_sector_dimensions`` and the columns of the first solve): rows and columns of one of the four
    sectors of 3,480, 10,456, 14,680, 19,392 and 29,576 electrons.  A device has the 85,093,777,408 bytes that
    the runner recorded for an A100 of 80 GB, 79.25 GiB.
    """

    TOTAL = 85_093_777_408
    K06, K09, K10 = (677_542, 442), (1_924_792, 1314), (2_604_846, 1842)
    K11, K13 = (2_873_400, 2431), (4_130_510, 3704)
    KEPT = "persistent CUDA representation states"
    SPILLED = "exact FP64 host spill; one active representation on CUDA"
    ONE_PER_DEVICE = ((0,), (1,), (2,), (3,))
    # What leaving the pool does, on a rank with these devices.
    LEFT_POOL = {
        devices: [*(("pool emptied", device) for device in devices), ("pinned pool emptied",), ("allocator", None)]
        for devices in ((0,), (0, 1), (0, 1, 2, 3))
    }

    def decide(self, sector, groups, owned=None, total=None, operators=None, **environment):
        """What a process decides for sectors of ``sector`` rows and columns on the devices of ``groups``.

        A sector outside ``owned`` is another rank's.  Every ``PARSEC_CUPY_*`` variable of the launch is
        removed and ``environment`` set.  Returns the solver and what it did to the allocator and the pools.
        The stand-in for cupy reports ``total`` bytes of memory per device and none of them free.
        """
        from parsec_python.acceleration.Eigensolvers import distributed_state, symmetry
        rows, columns = sector
        place, events = [], []

        class Device:
            def __init__(self, index):
                self.id = index

            def __enter__(self):
                place.append(self.id)
                return self

            def __exit__(self, *_error):
                place.pop()

        cp = SimpleNamespace(
            cuda=SimpleNamespace(
                Device=Device, runtime=SimpleNamespace(memGetInfo=lambda: (0, total or self.TOTAL)),
                get_allocator=lambda: "allocator of the pool",
                set_allocator=lambda allocator: events.append(("allocator", allocator))),
            get_default_memory_pool=lambda: SimpleNamespace(
                free_all_blocks=lambda: events.append(("pool emptied", place[-1]))),
            get_default_pinned_memory_pool=lambda: SimpleNamespace(
                free_all_blocks=lambda: events.append(("pinned pool emptied",))))
        owned = tuple(range(len(groups))) if owned is None else owned
        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._memory_allocator_evaluated = False
        solver._previous_memory_allocator = None
        solver.device_ids = tuple(sorted({device for index in owned for device in groups[index]}))
        solver._sector_device_ids = tuple(group[0] for group in groups)
        solver.decomposition = SimpleNamespace(sector_size=lambda representation: rows)
        # As the constructor leaves them on an MPI rank: the group of the sector on its operator.
        solver._operators = [
            SimpleNamespace(shape=(rows, rows), distributed_filter_devices=tuple(group)) if index in owned else None
            for index, group in enumerate(groups)
        ] if operators is None else list(operators)
        solver._solvers = [None if operator is None else object() for operator in solver._operators]
        with patch.object(symmetry, "require_cupy", return_value=(cp, None)), \
             patch.object(distributed_state, "require_cupy", return_value=(cp, None)), \
             patch.object(distributed_state, "_devices_can_filter", return_value=True), patch.dict(os.environ):
            for name in tuple(os.environ):
                if name.startswith("PARSEC_CUPY_"):
                    del os.environ[name]
            os.environ.update(environment)
            solver._configure_large_problem_allocator([columns] * len(groups))
            self.assertEqual(place, [])
        return solver, events

    @staticmethod
    def vectors(sector, sectors=1):
        return 8 * sector[0] * sector[1] * sectors

    def test_sectors_that_fit_keep_the_route_of_the_named_values(self):
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import streaming_workspace_bytes
        named = dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="device", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool")
        both = dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="auto", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="auto")
        # 19,392 electrons on four devices, 52.04 GiB of vectors on each: the run that stopped for memory
        # where neither value was named, and took 184.1 s and 58.8 GiB with both.
        report = ("cupy default pool; estimated persistent sector orbitals "
                  "cuda:0=55881883200B,cuda:1=55881883200B,cuda:2=55881883200B,cuda:3=55881883200B")
        for layout in (
                (self.K11, self.ONE_PER_DEVICE), (self.K10, self.ONE_PER_DEVICE),
                # Two sectors on each of two devices, four on one.
                (self.K10, ((0,), (1,), (0,), (1,))), (self.K09, ((0,), (1,), (0,), (1,))),
                (self.K06, ((0,),) * 4)):
            pinned, pinned_events = self.decide(*layout, **named)
            self.assertIsNone(pinned.state_fit_bytes)
            for environment in ({}, both, dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="auto"),
                                dict(named, PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="auto"),
                                dict(named, PARSEC_CUPY_SECTOR_STATE_STORAGE=" Auto ")):
                with self.subTest(layout=layout, environment=environment):
                    solver, events = self.decide(*layout, **environment)
                    self.assertFalse(solver._host_spill_sector_states)
                    self.assertEqual((events, pinned_events), ([], []))
                    self.assertEqual((solver.sector_state_storage, solver.memory_allocator_policy),
                                     (pinned.sector_state_storage, pinned.memory_allocator_policy))
                    self.assertEqual(solver.sector_state_storage, self.KEPT)
                    self.assertEqual([operator.low_memory_ritz for operator in solver._operators],
                                     [operator.low_memory_ritz for operator in pinned._operators])
                    # Per device: the vectors of its sectors and one buffer of the Ritz step.
                    sector, groups = layout
                    on_device = sum(group == groups[0] for group in groups)
                    self.assertEqual(solver.state_fit_bytes, dict.fromkeys(
                        solver.device_ids, self.vectors(sector, on_device) + streaming_workspace_bytes(*sector)))
            if layout == (self.K11, self.ONE_PER_DEVICE):
                self.assertEqual(pinned.memory_allocator_policy, report)
                # One buffer of 2 GiB beside the vectors, and of 4 GiB under the former limit of one array.
                self.assertEqual(solver.state_fit_bytes[0], 55_881_883_200 + 2_147_467_608)
                former, events = self.decide(*layout, PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES=str(4 << 30))
                self.assertEqual((former.state_fit_bytes[0], former._host_spill_sector_states, events),
                                 (55_881_883_200 + 4_294_954_664, False, []))

    def test_a_shared_basis_is_counted_by_the_blocks_of_its_devices(self):
        from parsec_python.acceleration.Eigensolvers import distributed_state
        gib = 1 << 30
        # The counts of the exchange that the measured series took and whose buffer they hold, the packed one
        # (PARSEC_CUPY_EXCHANGE_STRIDED=0).  The pitched exchange of the default takes none: end of the test.
        packed = dict(PARSEC_CUPY_EXCHANGE_STRIDED="0")
        # 29,576 electrons: a sector on two devices (a rank of 8 GPUs has two of them) or on four (16
        # GPUs, one).  The whole sector, 114 GiB, was counted on its owner: more than a device has.
        whole = self.vectors(self.K13)
        self.assertEqual(whole, 122_395_272_320)
        self.assertGreater(whole, self.TOTAL)
        for groups, owned in ((((0, 1), (9,), (2, 3), (9,)), (0, 2)), (((9,), (0, 1, 2, 3), (9,), (9,)), (1,))):
            with self.subTest(groups=groups):
                solver, events = self.decide(self.K13, groups, owned, **packed)
                self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage, events),
                                 (False, self.KEPT, []))
                self.assertTrue(solver.memory_allocator_policy.startswith("cupy default pool; "))
                devices = len(groups[owned[0]])
                with patch.dict(os.environ):
                    for name in tuple(os.environ):
                        if name.startswith("PARSEC_CUPY_"):
                            del os.environ[name]
                    os.environ.update(packed)
                    blocks = distributed_state.shared_block_bytes(*self.K13, devices)
                self.assertEqual(solver.state_fit_bytes, dict.fromkeys((0, 1, 2, 3), blocks))
                # What the rule that shares the basis lets it take: 0.85 of the device.
                self.assertLess(blocks, .85 * self.TOTAL)
                if devices == 2:
                    # With the two slabs that the Ritz step of two devices cuts, the 36 columns of the budget of
                    # the floor of 32 where 1 GiB holds 32; 1.7 GiB more are counted where
                    # PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES gives back the slabs of 2 GiB and 5.7 GiB more
                    # where it gives the former slabs of 4 GiB.  Those take up to 2 GiB outside a device, so
                    # that a narrower cut of them can be little wider than an exchange chunk: the buffer may
                    # then follow it, and 128 MiB more are counted unless the chunk is named as well, 1 GiB as
                    # it was.
                    self.assertAlmostEqual(blocks / gib, 60.24, delta=.01)
                    pair, events = self.decide(self.K13, groups, owned,
                                               PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(2 * gib), **packed)
                    self.assertEqual((pair._host_spill_sector_states, events), (False, []))
                    self.assertEqual(set(pair.state_fit_bytes.values()), {66_534_064_304})
                    self.assertAlmostEqual((66_534_064_304 - blocks) / gib, 1.72, delta=.01)
                    slabs = dict(PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(4 * gib))
                    for chunk, count in ((dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib)), 70_829_794_704),
                                         ({}, 70_829_794_704 + gib // 8)):
                        former, events = self.decide(self.K13, groups, owned, **slabs, **chunk, **packed)
                        self.assertEqual((former._host_spill_sector_states, events), (False, []))
                        self.assertEqual(set(former.state_fit_bytes.values()), {count})
                        self.assertLess(count, .85 * self.TOTAL)
                    self.assertAlmostEqual(70_829_794_704 / gib, 65.97, delta=.01)
                    self.assertAlmostEqual((70_829_794_704 - blocks) / gib, 5.72, delta=.01)
                    # The slabs of 36 columns, 1.1 GiB, lie within a chunk: the count of two devices names none.
                    named, events = self.decide(self.K13, groups, owned, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib),
                                                **packed)
                    self.assertEqual(named.state_fit_bytes, solver.state_fit_bytes)
                else:
                    # Four devices cut slabs of 36 columns, 1.11 GiB by the floor, of which 0.83 GiB leave a
                    # device: within a chunk, so the count names none.  The former round of 8 GiB cut slabs of
                    # 2 GiB, of which 1.5 GiB leave a device: 128 MiB more than with the chunk named.
                    named, events = self.decide(self.K13, groups, owned, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib),
                                                **packed)
                    self.assertEqual(named.state_fit_bytes, solver.state_fit_bytes)
                    self.assertEqual(blocks, 34_084_778_336)
                    rounds = dict(PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES=str(8 * gib))
                    former, events = self.decide(self.K13, groups, owned, **rounds, **packed)
                    named, events = self.decide(self.K13, groups, owned, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib),
                                                **rounds, **packed)
                    self.assertEqual(former.state_fit_bytes, dict.fromkeys((0, 1, 2, 3), 36_069_464_992))
                    self.assertEqual(named.state_fit_bytes, dict.fromkeys((0, 1, 2, 3), 36_069_464_992 - gib // 8))
        # The report of the allocator still names the whole sector on its owner, as with both values named.
        self.assertEqual(solver.memory_allocator_policy,
                         "cupy default pool; estimated persistent sector orbitals "
                         "cuda:0=122395272320B,cuda:1=0B,cuda:2=0B,cuda:3=0B")
        # The count of a shared basis is the sharing rule's own and no lower limit of what a device holds.
        # A rank of 10,456 electrons on 16 GPUs counted 9,945 MiB on each of its four devices with the
        # exchange chunk of 1 GiB and the slabs of 2 GiB that it had, which the former round of 8 GiB gives
        # back; devices of that run sampled as little as 10,005 MiB, 426 MiB of them the CUDA context.  Where
        # no chunk is named the count holds the most that a buffer which follows a slab can take, 128 MiB
        # more: 10,073 MiB.
        rounds = dict(PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES=str(8 * gib))
        for chunk, count in ((dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib)), 10_428_230_944), ({}, 10_562_448_672)):
            solver, events = self.decide(self.K09, ((9,), (0, 1, 2, 3), (9,), (9,)), (1,), **chunk, **rounds, **packed)
            self.assertEqual(solver.state_fit_bytes, dict.fromkeys((0, 1, 2, 3), count))
            self.assertEqual((solver._host_spill_sector_states, events), (False, []))
        self.assertEqual((round(10_428_230_944 / 2 ** 20), round(10_562_448_672 / 2 ** 20)), (9_945, 10_073))
        self.assertGreater(10_428_230_944, (10_005 - 426) << 20)
        self.assertLess(10_428_230_944, 10_005 << 20)
        # The pitched exchange, the default since 2026-10-22, takes no buffer, and the count holds none: 1 GiB
        # less for 29,576 electrons on two devices and on four, whose slabs lie within a chunk, and the same
        # decisions.
        for groups, owned in ((((0, 1), (9,), (2, 3), (9,)), (0, 2)), (((9,), (0, 1, 2, 3), (9,), (9,)), (1,))):
            default, events = self.decide(self.K13, groups, owned)
            self.assertEqual((default._host_spill_sector_states, default.sector_state_storage, events),
                             (False, self.KEPT, []))
            former, events = self.decide(self.K13, groups, owned, **packed)
            self.assertEqual({device: former.state_fit_bytes[device] - count
                              for device, count in default.state_fit_bytes.items()}, dict.fromkeys((0, 1, 2, 3), gib))
        # The slabs of 1 GiB of the default, 69 columns, are counted 2 GiB lower, 7,889 MiB, and lie within a
        # chunk whether one is named or not.
        for chunk in (dict(PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(gib)), {}):
            solver, events = self.decide(self.K09, ((9,), (0, 1, 2, 3), (9,), (9,)), (1,), **chunk, **packed)
            self.assertEqual(solver.state_fit_bytes, dict.fromkeys((0, 1, 2, 3), 8_272_463_904))
            self.assertEqual((solver._host_spill_sector_states, events), (False, []))
        self.assertEqual(round(8_272_463_904 / 2 ** 20), 7_889)
        # A basis that is not shared lies on the owner of its sector as one array and is counted there:
        # with the sharing off, and where the Ritz step is not the one a shared basis takes.
        groups, owned = ((0, 1), (9,), (2, 3), (9,)), (0, 2)
        for environment in (dict(PARSEC_CUPY_DISTRIBUTED_STATE="off"), dict(PARSEC_CUPY_GENERALIZED_RITZ="off")):
            with self.subTest(environment=environment):
                solver, events = self.decide(self.K13, groups, owned, **environment)
                self.assertGreater(solver.state_fit_bytes[0], whole)
                self.assertEqual((solver.state_fit_bytes[1], solver.state_fit_bytes[3]), (0, 0))
                self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage),
                                 (True, self.SPILLED))
                self.assertEqual(events, self.LEFT_POOL[0, 1, 2, 3])

    def test_sectors_that_cannot_fit_are_spilled_and_leave_the_pool(self):
        # 14,680 electrons on one device: four sectors of 35.75 GiB.
        one_device = (self.K10, ((0,),) * 4)
        solver, events = self.decide(*one_device)
        self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage), (True, self.SPILLED))
        self.assertEqual(events, self.LEFT_POOL[0,])
        self.assertEqual(solver._previous_memory_allocator, "allocator of the pool")
        self.assertEqual(solver.memory_allocator_policy,
                         "direct CUDA allocation for memory-bound sectors; estimated persistent sector orbitals "
                         f"cuda:0={self.vectors(self.K10, 4)}B")
        self.assertGreater(solver.state_fit_bytes[0], self.TOTAL)
        # A named value is taken as it is, and the other follows the count alone.
        for environment, spilled, direct in (
                (dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="device"), False, True),
                (dict(PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool"), True, False),
                (dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="host", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool"), True, False),
                (dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="device", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="direct"),
                 False, True)):
            with self.subTest(environment=environment):
                solver, events = self.decide(*one_device, **environment)
                self.assertEqual((solver._host_spill_sector_states, events),
                                 (spilled, self.LEFT_POOL[0,] if direct else []))
                self.assertEqual(solver.sector_state_storage, self.SPILLED if spilled else self.KEPT)
        # Where the sectors fit, a named spill keeps the pool and a named direct allocator the states.
        fits = (self.K11, self.ONE_PER_DEVICE)
        for environment, spilled, direct in (
                (dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="host"), True, False),
                (dict(PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="direct"), False, True)):
            with self.subTest(environment=environment):
                solver, events = self.decide(*fits, **environment)
                self.assertEqual((solver._host_spill_sector_states, events),
                                 (spilled, self.LEFT_POOL[0, 1, 2, 3] if direct else []))
        # The sectors of another rank are not counted: one sector of the four, alone on the device.
        solver, events = self.decide(self.K10, ((0,),) * 4, owned=(2,))
        self.assertEqual((solver._host_spill_sector_states, events), (False, []))
        self.assertLess(solver.state_fit_bytes[0], self.vectors(self.K10, 2))

    def test_the_count_is_held_against_a_share_of_a_device_and_reads_nothing_of_the_run(self):
        from parsec_python.acceleration.Eigensolvers.rayleigh_ritz import streaming_workspace_bytes
        gib = 1 << 30
        whole = dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="1")
        # The buffer of one array as it was, at most 4 GiB, and the share that was chosen with it.
        limit = dict(PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES=str(4 * gib))
        former = dict(limit, PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.978")
        # 10,456 electrons on one device: 75.38 GiB of vectors and a buffer of 2 GiB, 77.37 GiB of 79.25.
        # The count leaves out what else the device holds: the devices that ran sampled at least 1.0268
        # times theirs, which is 79.5 GiB here.  Held against 0.974 of the device the states are spilled,
        # as the rule before spilled them; held against all of it they stayed, and with the buffer of
        # 2.36 GiB that such a sector took then the run stopped for memory.
        layout = (self.K09, ((0,),) * 4)
        solver, events = self.decide(*layout)
        count = solver.state_fit_bytes[0]
        self.assertEqual(count, self.vectors(self.K09, 4) + streaming_workspace_bytes(*self.K09))
        self.assertEqual(count, 83_081_129_472)
        self.assertAlmostEqual(count / gib, 77.375, delta=.005)
        self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage, events),
                         (True, self.SPILLED, self.LEFT_POOL[0,]))
        self.assertGreater(1.0268 * count, self.TOTAL)
        solver, events = self.decide(*layout, **whole)
        self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage, events),
                         (False, self.KEPT, []))
        self.assertEqual(solver.state_fit_bytes[0], count)
        # With that buffer the count was 77.73 GiB, which the share of then, 0.978, spilled.  The smaller
        # buffer brings the count below that share, which would keep the states: the share has come down
        # with the count, to the reciprocal of the least ratio of peak to count that was measured.
        for environment, counted, spilled in ((former, 83_462_830_704, True), (limit, 83_462_830_704, True),
                                              (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.978"), count, False)):
            solver, events = self.decide(*layout, **environment)
            self.assertEqual((solver.state_fit_bytes[0], solver._host_spill_sector_states, bool(events)),
                             (counted, spilled, spilled))
        self.assertLess(count / self.TOTAL, .978)
        self.assertLess(.978, 83_462_830_704 / self.TOTAL)
        # The fullest device that ran, 14,680 electrons on two: 73.50 GiB counted, and 75.47 and 76.28
        # sampled on its two devices with the buffer of 2 GiB (75.50 counted, 77.47 and 78.28 sampled with
        # that of 4).  The share lies between the two cells, and one below 0.9275 would spill the run that
        # fitted.
        fitted = (self.K10, ((0,), (1,), (0,), (1,)))
        for environment, spilled in (({}, False), (whole, False), (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.92"), True)):
            solver, events = self.decide(*fitted, **environment)
            self.assertEqual((solver._host_spill_sector_states, bool(events)), (spilled, spilled))
            self.assertEqual(solver.state_fit_bytes[0], 78_917_498_592)
        self.assertLess(78_917_498_592 / self.TOTAL, .974)
        self.assertLess(.974, count / self.TOTAL)
        self.assertEqual([round((peak << 20) / 78_917_498_592, 4) for peak in (77_285, 78_109)], [1.0269, 1.0378])
        self.assertGreaterEqual(.974 * (77_285 << 20) / 78_917_498_592, 1.)
        # The former count of that run, and the ratio that the former share was chosen by.
        for environment, spilled in ((former, False), (dict(limit, PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.95"), True)):
            solver, events = self.decide(*fitted, **environment)
            self.assertEqual((solver.state_fit_bytes[0], solver._host_spill_sector_states, bool(events)),
                             (81_064_975_872, spilled, spilled))
        self.assertEqual([round((peak << 20) / 81_064_975_872, 4) for peak in (79_333, 80_157)], [1.0262, 1.0368])
        # The share is of the device: with one half, a device of twice the count keeps its states and
        # two bytes less are too few.  With all of it, a device of exactly the count keeps them and one
        # byte less does not.
        for environment, total, spilled in (
                (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.5"), 2 * count, False),
                (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.5"), 2 * count - 2, True),
                (whole, count, False), (whole, count - 1, True), ({}, count, True)):
            with self.subTest(environment=environment, total=total):
                solver, events = self.decide(*layout, total=total, **environment)
                self.assertEqual((solver._host_spill_sector_states, bool(events)), (spilled, spilled))
        # A named value is taken as it is whatever the share, and the other one follows the count.
        solver, events = self.decide(*layout, PARSEC_CUPY_SECTOR_STATE_STORAGE="device")
        self.assertEqual((solver._host_spill_sector_states, events), (False, self.LEFT_POOL[0,]))
        solver, events = self.decide(*layout, PARSEC_CUPY_SECTOR_STATE_STORAGE="device",
                                     PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool")
        self.assertEqual((solver._host_spill_sector_states, events, solver.state_fit_bytes), (False, [], None))
        # The rule before, named by its fraction, does not ask the share: 75.38 GiB of vectors are below
        # 0.96 of the device and at least 0.95 of it.
        for fraction, spilled in (("0.96", False), ("0.95", True)):
            solver, events = self.decide(*layout, PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION=fraction,
                                         PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.5")
            self.assertEqual((solver._host_spill_sector_states, bool(events)), (spilled, spilled))
        # The budget of the slabs is counted as the Ritz step takes it, whatever the limit of one array: 1 GiB
        # in place of 2, and 3 GiB where 3 are named.
        for budget, counted in ((gib, 1_073_737_728), (3 * gib, 3_221_223_696)):
            for environment in ({}, limit):
                solver, _events = self.decide(*layout, PARSEC_CUPY_STREAMING_RITZ_BYTES=str(budget), **environment)
                self.assertEqual(solver.state_fit_bytes[0] - self.vectors(self.K09, 4), counted)
                self.assertLessEqual(counted, budget)
        # Two arrays in the Ritz step are counted as one.
        solver, _events = self.decide(*layout, PARSEC_CUPY_STREAMING_RITZ="0")
        self.assertEqual(solver.state_fit_bytes[0], count)

    def test_every_measured_first_run_keeps_its_route_and_peaked_above_the_count(self):
        from parsec_python.acceleration.experimental.mpi_scf import sector_device_groups
        # First runs of one series on A100 nodes, all with ``device`` and ``pool`` named: rows and columns of
        # a sector, ranks, devices of a rank, and the sampled peak of the fullest device in MiB.  The first
        # rank of each, with the sector groups that the runner gives it.  A sector of two ranks lies on two
        # devices, whose Ritz step cut slabs of 4 GiB in that series, a sector of four ranks on four, whose
        # step cut slabs of 2 GiB, an even share of the former round of 8 GiB, and a sector of one rank on one
        # device, whose buffer took 4 GiB at most: the count is made with those limits.
        # Every shared basis of the series had an exchange buffer of 1 GiB, which is named for its count.
        # The fourth is of later runs on the code of that series, 14,680 electrons on two devices: the
        # fullest device that ran, 80,157 of its 81,152 MiB, where 77,310 are counted.
        runs = (
            (*self.K06, 1, 1, 11471), (*self.K09, 1, 2, 43115), (*self.K10, 1, 4, 43141), (*self.K10, 1, 2, 80157),
            (*self.K11, 1, 4, 60251), (*self.K11, 2, 4, 37459), (*self.K11, 4, 4, 20417),
            (3_786_832, 2978, 2, 4, 54795), (3_786_832, 2978, 4, 4, 29451),
            (*self.K13, 2, 4, 70817), (*self.K13, 4, 4, 37831), (5_660_200, 4928, 4, 4, 63911))
        one_array = dict(PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES=str(4 << 30))
        self.assertEqual(self.decide(self.K10, ((0,), (1,), (0,), (1,)), **one_array)[0].state_fit_bytes,
                         {0: 81_064_975_872, 1: 81_064_975_872})
        self.assertEqual(self.decide(self.K10, ((0,), (1,), (0,), (1,)))[0].state_fit_bytes,
                         {0: 78_917_498_592, 1: 78_917_498_592})
        # The series ran the packed exchange, whose buffer the counts hold (PARSEC_CUPY_EXCHANGE_STRIDED=0).
        packed = dict(PARSEC_CUPY_EXCHANGE_STRIDED="0")
        chunk = dict(packed, PARSEC_CUPY_EXCHANGE_CHUNK_BYTES=str(1 << 30))
        series = dict(chunk, PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(4 << 30))
        rounds = dict(packed, PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES=str(8 << 30))
        # The same sectors of two devices with the slabs of 2 GiB that were the default then, which the former
        # limit of two devices gives back: peaks of later runs, 3.7 to 3.8 GiB below those of the series,
        # where 4 GiB less are counted.
        pair = dict(packed, PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=str(2 << 30))
        pair_peaks = {2_873_400: 33643, 3_786_832: 50859, 4_130_510: 67031}
        # And sectors on one device with the buffer of 2 GiB that is the default: peaks of later runs with
        # that budget, 2.0 GiB below those with 4 GiB in the same allocation, where 2 GiB less are counted.
        buffer_peaks = {(*self.K10, 2): 78109, (*self.K10, 4): 41049, (*self.K11, 4): 58155}
        former = []
        for rows, columns, ranks, devices, peak in runs:
            with self.subTest(rows=rows, ranks=ranks, devices=devices):
                mine = sector_device_groups(4, ranks, 0, tuple(range(devices)))
                groups = tuple(mine.get(index, (9,)) for index in range(4))
                slabs = series if ranks == 2 else dict(chunk, **one_array) if ranks == 1 else dict(chunk, **rounds)
                solver, events = self.decide((rows, columns), groups, tuple(mine), **slabs)
                if ranks == 4:
                    # Four devices per sector count up to 128 MiB more where no chunk is named, for a
                    # buffer that may follow a slab: still below what the devices sampled.
                    unnamed, left = self.decide((rows, columns), groups, tuple(mine), **rounds)
                    most = max(unnamed.state_fit_bytes.values())
                    self.assertEqual((unnamed._host_spill_sector_states, left), (False, []))
                    self.assertEqual(most - max(solver.state_fit_bytes.values()), 1 << 27)
                    self.assertLess(most, peak << 20)
                # The route of the named values, and a count that the device did exceed.
                self.assertEqual((solver._host_spill_sector_states, events), (False, []))
                self.assertLess(max(solver.state_fit_bytes.values()), peak << 20)
                self.assertGreater(max(solver.state_fit_bytes.values()), (peak << 20) * .85)
                if ranks == 2:
                    default, events = self.decide((rows, columns), groups, tuple(mine), **pair)
                    self.assertEqual((default._host_spill_sector_states, events), (False, []))
                    count = max(default.state_fit_bytes.values())
                    self.assertLess(count, pair_peaks[rows] << 20)
                    self.assertGreater(count, (pair_peaks[rows] << 20) * .85)
                    self.assertAlmostEqual((max(solver.state_fit_bytes.values()) - count) / (1 << 30), 4, delta=.02)
                if ranks > 1:
                    # The slabs of 1 GiB of the default, and of the 36 columns of the budget of the floor where
                    # 1 GiB holds fewer, are counted lower than the slabs of 2 GiB, the chunk unnamed in both.
                    now, left = self.decide((rows, columns), groups, tuple(mine), **packed)
                    self.assertEqual((now._host_spill_sector_states, left), (False, []))
                    self.assertLess(max(now.state_fit_bytes.values()),
                                    max((unnamed if ranks == 4 else default).state_fit_bytes.values()))
                    # The pitched exchange of the default since takes no buffer: 1 GiB less, or up to 1.125
                    # GiB where the buffer follows a slab, and the same route.
                    pitched, left = self.decide((rows, columns), groups, tuple(mine))
                    self.assertEqual((pitched._host_spill_sector_states, left), (False, []))
                    less = max(now.state_fit_bytes.values()) - max(pitched.state_fit_bytes.values())
                    self.assertTrue(1 << 30 <= less <= (1 << 30) + (1 << 27), less)
                if (rows, columns, devices) in buffer_peaks and ranks == 1:
                    default, events = self.decide((rows, columns), groups, tuple(mine))
                    self.assertEqual((default._host_spill_sector_states, events), (False, []))
                    count = max(default.state_fit_bytes.values())
                    later = buffer_peaks[rows, columns, devices]
                    self.assertLess(count, later << 20)
                    self.assertGreater(count, (later << 20) * .85)
                    self.assertAlmostEqual((max(solver.state_fit_bytes.values()) - count) / (1 << 30), 2, delta=.001)
                    self.assertAlmostEqual((peak - later) / 1024, 2, delta=.05)
                solver, events = self.decide((rows, columns), groups, tuple(mine),
                                             PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="0.5", **slabs)
                former.append(solver._host_spill_sector_states and bool(events))
        # The former rule spilled and left the pool from 19,392 electrons up, on 4, 8 and 16 GPUs, and for
        # 14,680 electrons on two devices.
        self.assertEqual(former, [False] * 3 + [True] * 9)

    def test_the_serial_control_of_the_runner_decides_by_the_former_rule(self):
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        control = {}
        mpi_full_scf._keep_former_routes(control)
        control = {name: value for name, value in control.items() if name.startswith("PARSEC_CUPY_")}
        # 14,680 electrons on two devices, 71.5 GiB of vectors on each: a run keeps them there, and the
        # control, whose Ritz step takes a second array of 35.75 GiB, spills them as it did.
        layout = (self.K10, ((0,), (1,), (0,), (1,)))
        run, events = self.decide(*layout)
        self.assertEqual((run._host_spill_sector_states, events), (False, []))
        solver, events = self.decide(*layout, **control)
        self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage), (True, self.SPILLED))
        self.assertEqual(events, self.LEFT_POOL[0, 1])
        self.assertIsNone(solver.state_fit_bytes)
        former, former_events = self.decide(*layout, PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="0.5")
        self.assertEqual((solver.memory_allocator_policy, events), (former.memory_allocator_policy, former_events))
        # Below half of a device both rules keep the states and the pool, with the same report.
        small = (self.K06, ((0,),) * 4)
        solver, events = self.decide(*small, **control)
        run, _events = self.decide(*small)
        self.assertEqual((solver._host_spill_sector_states, events), (False, []))
        self.assertEqual((solver.sector_state_storage, solver.memory_allocator_policy),
                         (run.sector_state_storage, run.memory_allocator_policy))
        # A launcher that names both values is asked nothing, in a control as in a run.
        named = dict(control, PARSEC_CUPY_SECTOR_STATE_STORAGE="device", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool")
        solver, events = self.decide(*layout, **named)
        self.assertEqual((solver._host_spill_sector_states, events, solver.state_fit_bytes), (False, [], None))

    def test_two_named_values_ask_nothing_of_the_sectors(self):
        from parsec_python.acceleration.Eigensolvers import symmetry
        # Operators without a group or a shape, as the solver of a process without a sector context has
        # them: no rule runs, whatever the sizes.
        bare = [SimpleNamespace() for _ in range(4)]
        for storage, allocator in (("device", "pool"), ("host", "direct"), ("host", "pool"), ("device", "direct")):
            with self.subTest(storage=storage, allocator=allocator), \
                 patch.object(symmetry, "shared_basis_devices", side_effect=AssertionError("asked")):
                solver, events = self.decide(
                    self.K10, ((0,),) * 4, operators=bare,
                    PARSEC_CUPY_SECTOR_STATE_STORAGE=storage, PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR=allocator)
                self.assertIsNone(solver.state_fit_bytes)
                self.assertEqual((solver._host_spill_sector_states, bool(events)),
                                 (storage == "host", allocator == "direct"))
        # Without a group the basis of a sector lies on its device: such operators are counted there.
        solver, events = self.decide(self.K10, ((0,),) * 4, operators=bare)
        self.assertEqual((solver._host_spill_sector_states, events), (True, self.LEFT_POOL[0,]))

    def test_naming_the_fraction_restores_the_former_rule(self):
        def former(sizes, fraction, storage, allocator):
            # The rule as it was written: the vectors of whole sectors on their owners against a share
            # of the device, for the storage and for the allocator alike.
            crowded = any(size >= fraction * self.TOTAL for size in sizes)
            return (storage == "host" or (storage == "auto" and crowded),
                    allocator == "direct" or (allocator == "auto" and crowded))

        layouts = (
            (self.K11, self.ONE_PER_DEVICE, None), (self.K10, self.ONE_PER_DEVICE, None),
            (self.K10, ((0,), (1,), (0,), (1,)), None), (self.K09, ((0,),) * 4, None), (self.K06, ((0,),) * 4, None),
            (self.K13, ((0, 1), (9,), (2, 3), (9,)), (0, 2)), (self.K13, ((9,), (0, 1, 2, 3), (9,), (9,)), (1,)))
        for sector, groups, owned in layouts:
            owners = [group[0] for index, group in enumerate(groups) if owned is None or index in owned]
            for fraction in ("0.5", "0.3", "0.9", "1.0"):
                for storage in ("auto", "device", "host"):
                    for allocator in ("auto", "pool", "direct"):
                        with self.subTest(sector=sector, groups=groups, fraction=fraction, storage=storage,
                                          allocator=allocator):
                            solver, events = self.decide(
                                sector, groups, owned, PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION=fraction,
                                PARSEC_CUPY_SECTOR_STATE_STORAGE=storage, PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR=allocator)
                            sizes = [self.vectors(sector, owners.count(device)) for device in solver.device_ids]
                            spilled, direct = former(sizes, float(fraction), storage, allocator)
                            self.assertEqual((solver._host_spill_sector_states, bool(events)), (spilled, direct))
                            self.assertEqual(solver.sector_state_storage, self.SPILLED if spilled else self.KEPT)
                            self.assertEqual(solver.memory_allocator_policy, (
                                "direct CUDA allocation for memory-bound sectors; estimated persistent sector orbitals "
                                if direct else "cupy default pool; estimated persistent sector orbitals ") + ",".join(
                                    f"cuda:{device}={size}B" for device, size in zip(solver.device_ids, sizes)))
                            self.assertIsNone(solver.state_fit_bytes)
        # At its default of one half that rule spilled the run of 19,392 electrons on four devices, and
        # it took the pool from the ranks of 29,576 electrons on 8 and 16 GPUs, whose shared basis no spill moves.
        half = dict(PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="0.5")
        solver, events = self.decide(self.K11, self.ONE_PER_DEVICE, **half)
        self.assertEqual((solver._host_spill_sector_states, events), (True, self.LEFT_POOL[0, 1, 2, 3]))
        solver, events = self.decide(self.K13, ((0, 1), (9,), (2, 3), (9,)), (0, 2), **half)
        self.assertEqual((solver._host_spill_sector_states, events), (True, self.LEFT_POOL[0, 1, 2, 3]))
        self.assertEqual(solver.memory_allocator_policy,
                         "direct CUDA allocation for memory-bound sectors; estimated persistent sector orbitals "
                         "cuda:0=122395272320B,cuda:1=0B,cuda:2=122395272320B,cuda:3=0B")
        # 14,680 electrons on four devices stayed below it, and stay where they were under either rule.
        solver, events = self.decide(self.K10, self.ONE_PER_DEVICE, **half)
        self.assertEqual((solver._host_spill_sector_states, events), (False, []))

    def test_unknown_values_stop_the_configuration_and_it_is_decided_once(self):
        layout = (self.K06, ((0,),) * 4)
        for environment, words in (
                (dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="fit"), "PARSEC_CUPY_SECTOR_STATE_STORAGE must be auto, device, or host"),
                (dict(PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="fit"),
                 "PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR must be auto, pool, or direct"),
                (dict(PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="0"), r"PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION must be in \(0, 1\]"),
                (dict(PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="1.5"), r"PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION must be in \(0, 1\]"),
                (dict(PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION=""), "could not convert string to float"),
                (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0"), r"PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION must be in \(0, 1\]"),
                (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="1.01"),
                 r"PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION must be in \(0, 1\]"),
                (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="nan"), r"PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION must be in \(0, 1\]"),
                (dict(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="all"), "could not convert string to float"),
                # Read by the download of a spill, which comes after the first solve, and by its staging.
                (dict(PARSEC_CUPY_SPILL_DOWNLOAD_ORDER="F"), "PARSEC_CUPY_SPILL_DOWNLOAD_ORDER must be A or C"),
                (dict(PARSEC_CUPY_SPILL_BUFFERS="pinned"), "PARSEC_CUPY_SPILL_BUFFERS must be persistent or fresh"),
                (dict(PARSEC_CUPY_SPILL_PINNED_POOL="0"), "PARSEC_CUPY_SPILL_PINNED_POOL must be keep or release"),
                (dict(PARSEC_CUPY_SPILL_PINNED_BYTES="half"), "PARSEC_CUPY_SPILL_PINNED_BYTES must be a nonnegative integer"),
                (dict(PARSEC_CUPY_SPILL_PINNED_BYTES="-1"), "PARSEC_CUPY_SPILL_PINNED_BYTES must be nonnegative"),
                # Read by the Ritz step of a sector on one device, inside the first solve, after its first filter.
                (dict(PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES="2GiB"), "PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES must be an integer"),
                (dict(PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES="0"), "PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES must be positive")):
            # Also where both values are named and the rule is not asked.
            for named in ({}, dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="device", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool")):
                with self.subTest(environment=environment, named=named), self.assertRaisesRegex(ValueError, words):
                    self.decide(*layout, **{**named, **environment})
        # The first eigensolve decides; a later call changes nothing, whatever the environment says then.
        solver, events = self.decide(*layout)
        before = (solver._host_spill_sector_states, solver.sector_state_storage, solver.memory_allocator_policy,
                  dict(solver.state_fit_bytes))
        with patch.dict(os.environ, PARSEC_CUPY_SECTOR_STATE_STORAGE="host", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="direct"):
            solver._configure_large_problem_allocator([10 ** 6] * 4)
        self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage,
                          solver.memory_allocator_policy, solver.state_fit_bytes), before)

    def test_the_count_holds_the_slabs_of_a_budget_whatever_the_multiple_cuts_them_to(self):
        import importlib
        from parsec_python.acceleration.Eigensolvers import distributed_state
        # The package also exports a function named like this submodule.
        ritz = importlib.import_module("parsec_python.acceleration.Eigensolvers.rayleigh_ritz")
        name = "PARSEC_CUPY_RITZ_GRAM_MULTIPLE"
        # The Ritz step cuts its slabs and rounds down to whole multiples of 64 columns, which the rule of
        # this class came before.  It counts the slabs of their budget, on one device and on several: the
        # same bytes under every multiple, so that the multiple moves no sector to the host or back.
        layouts = (
            (self.K10, ((0,), (1,), (0,), (1,)), None), (self.K11, self.ONE_PER_DEVICE, None),
            (self.K09, ((0,),) * 4, None), (self.K13, ((0, 1), (9,), (2, 3), (9,)), (0, 2)),
            (self.K13, ((9,), (0, 1, 2, 3), (9,), (9,)), (1,)))
        for sector, groups, owned in layouts:
            with self.subTest(sector=sector, groups=groups):
                unnamed, _events = self.decide(sector, groups, owned)
                self.assertTrue(all(unnamed.state_fit_bytes.values()))
                for multiple in ("1", "64", "128", "7"):
                    solver, events = self.decide(sector, groups, owned, **{name: multiple})
                    self.assertEqual((solver.state_fit_bytes, solver._host_spill_sector_states, events),
                                     (unnamed.state_fit_bytes, unnamed._host_spill_sector_states, _events))
        with patch.dict(os.environ):
            for unset in (name, "PARSEC_CUPY_STREAMING_RITZ_BYTES", "PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES",
                          "PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS", "PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES",
                          "PARSEC_CUPY_DISTRIBUTED_STATE_MIN_SLAB_COLUMNS"):
                os.environ.pop(unset, None)
            # One array per sector: the buffer that the step takes is the count under a multiple of 1 and
            # never more.  A slab that is cut down leaves the tile as the larger view of the buffer, which
            # the budget holds to less than a row of it.  The last two have a budget of 130 columns, where
            # the slab is the larger view until it is cut to 128.
            small = dict(PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8 * 1000 * 130))
            shapes = [(shape, {}) for shape in (self.K06, self.K09, self.K10, self.K11, self.K13, (5_660_200, 4928))]
            for (rows, columns), budget in shapes + [((1000, 600), small), ((1000, 131), small)]:
                with patch.dict(os.environ, budget):
                    count = ritz.streaming_workspace_bytes(rows, columns)
                    sector = SimpleNamespace(shape=(rows, columns), nbytes=8 * rows * columns)
                    taken = {}
                    for multiple in ("1", "64", "128", "7"):
                        os.environ[name] = multiple
                        self.assertEqual(ritz.streaming_workspace_bytes(rows, columns), count)
                        width, tile = ritz._streaming_shape(sector)
                        taken[multiple] = 8 * max(rows * width, tile * columns)
                        self.assertLess(count - 8 * columns, taken[multiple])
                        self.assertLessEqual(taken[multiple], count)
                    del os.environ[name]
                    self.assertEqual(taken["1"], count)
                    if budget:
                        self.assertLess(taken["64"], count)
            # A shared basis: the workspace of the rounds that are cut, for the sectors of the measured series
            # on two devices and on four, is never more than the workspace of the count, with the slabs of the
            # default, with the former slabs of 2 GiB and with the slab limit of two devices at the former 4
            # GiB.
            sectors = (
                (1_924_792, (660, 654)), (2_604_846, (924, 918)), (2_873_400, (1218, 1213)),
                (3_786_832, (1494, 1484)), (4_130_510, (1854, 1850)), (1_924_792, (330, 330, 324, 330)),
                (2_604_846, (462, 462, 456, 462)), (2_873_400, (606, 612, 612, 601)),
                (3_786_832, (750, 744, 744, 740)), (4_130_510, (930, 924, 924, 926)),
                (5_660_200, (1230, 1236, 1236, 1226)))
            limits = {"default": {}, "former": dict(PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES="8589934592",
                                                     PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES="2147483648"),
                      "4 GiB": dict(PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES="4294967296")}
            alike = []
            for rows, widths in sectors:
                columns, devices = sum(widths), len(widths)
                for limit, named in limits.items():
                    with patch.dict(os.environ, named):
                        budget = distributed_state.slab_columns(rows, columns, devices, blocks=1)
                        counted = distributed_state.slab_workspace(rows, columns, devices, 1)
                    held = {}
                    for multiple in (1, 64):
                        rounds = distributed_state.block_rounds(widths, budget, multiple)
                        group = object.__new__(distributed_state.SectorDeviceGroup)
                        group.devices, group._work, group._outgrown = tuple(range(devices)), 0, set()
                        held[multiple] = group._cut_capacity(
                            rows, max(last - first for taken in rounds for first, last in taken),
                            max(sum(last - first for first, last in taken) for taken in rounds), 1)
                        self.assertLessEqual(held[multiple], counted, (rows, widths, limit, multiple))
                    self.assertLessEqual(held[64], held[1])
                    if held[64] == held[1]:
                        alike.append((rows, devices, limit))
            # Narrower in all but the sector of 29,576 electrons on two devices with slabs of 2 GiB, whose equal
            # slabs of 64 columns are those of a whole round.
            self.assertEqual(alike, [(4_130_510, 2, "former")])

    def test_the_former_values_of_the_switches_of_one_round_select_every_former_route_together(self):
        import importlib
        from parsec_python.Output.parsec_output import domain_report_requested
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        from parsec_python.acceleration.Eigensolvers import distributed_state
        from parsec_python.acceleration.Eigensolvers.eigval import spill_download_order
        # The package also exports a function named like this submodule.
        ritz = importlib.import_module("parsec_python.acceleration.Eigensolvers.rayleigh_ritz")
        # The rule of this class came together with eight other switches, and the last table of switches in
        # README.md gives each a former value.  They are held together here, where nothing else is, with the
        # two that the slabs of the one-block Ritz step took since, the round of four devices and the floor of
        # a slab: with all eleven named a process reads none of these routes, and with none it reads them all.
        former = dict(
            PARSEC_DOMAIN_REPORT="0", PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES="4294967296",
            PARSEC_CUPY_DENSITY_RELEASE="serial", PARSEC_CUPY_EXCHANGE_PAIRS="0",
            PARSEC_CUDA_CONTEXT_CREATION="runtime", PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="0.5",
            PARSEC_CUPY_SPILL_DOWNLOAD_ORDER="C", PARSEC_CUPY_RITZ_GRAM_MULTIPLE="1",
            PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE="1", PARSEC_CUPY_DISTRIBUTED_STATE_ROUND_BYTES="8589934592",
            PARSEC_CUPY_DISTRIBUTED_STATE_MIN_SLAB_COLUMNS="1")
        # 23,768 electrons on 8 GPUs: a sector of 84 GiB on two devices, whose slabs of 4 GiB hold 141 columns.
        rows, columns = 3_786_832, 2978

        def routes():
            # The report of the sphere, the slab limit of two devices, the round of four devices and the floor of
            # a slab, the slabs they give that sector on two devices and on four, the release after a density,
            # the turns of four devices, the creation of the contexts and the download of a spill.
            return (domain_report_requested(), distributed_state.pair_slab_bytes(), distributed_state.round_bytes(),
                    distributed_state.min_slab_columns(),
                    distributed_state.slab_columns(rows, columns, 2, blocks=1),
                    distributed_state.slab_columns(rows, columns, 4, blocks=1),
                    symmetry_density.density_release(), distributed_state.paired_exchange_requested(),
                    mpi_full_scf._driver_contexts_requested(), spill_download_order())

        def rounds(widths):
            # How many rounds the one-block step of that sector takes on these blocks, and the columns that
            # the first and the last of them bring together.
            cut = distributed_state.block_rounds(
                widths, distributed_state.slab_columns(rows, columns, len(widths), blocks=1), ritz.gram_multiple())
            joined = [sum(last - first for first, last in taken) for taken in cut]
            return len(cut), joined[0], joined[-1]

        def widths():
            # The multiple of the Gram products and what it makes of that sector: the slabs of the overlap and
            # of the projection of one array, and the rounds on two devices and on four.  And whether a
            # shared basis has its pools emptied after a density.
            sector = SimpleNamespace(shape=(rows, columns), nbytes=8 * rows * columns)
            return (ritz.gram_multiple(), ritz._gram_slab_width(columns), ritz._streaming_shape(sector)[0],
                    rounds((1494, 1484)), rounds((750, 744, 744, 740)),
                    distributed_state.shared_pool_release_requested())

        # The buffer of a sector on one device took 4 GiB at most in the code of these nine and takes 2 since:
        # its limit, which came after them, is held here at the value it had.
        later = dict(PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES="4294967296")
        with patch.dict(os.environ, later):
            for name in (*former, "PARSEC_CUPY_STREAMING_RITZ_BYTES", "PARSEC_CUPY_RITZ_GRAM_SLABS"):
                os.environ.pop(name, None)
            # Slabs of the 36 columns of the budget of the floor on two devices and on four, where 1 GiB holds 35
            # of that sector, in rounds of 64 and 128.
            self.assertEqual(routes(), (True, 1 << 30, 4 << 30, 32, 36, 36, "threads", True, True, "A"))
            self.assertEqual(widths(), (64, 320, 128, (47, 34, 64), (24, 34, 128), False))
            os.environ.update(former)
            self.assertEqual(routes(), (False, 4 << 30, 8 << 30, 1, 141, 70, "serial", False, False, "C"))
            # The rounds that the earlier series ran: eleven of 269 to 273 columns on 8 GPUs and on 16.
            self.assertEqual(widths(), (1, 373, 141, (11, 271, 269), (11, 273, 269), True))
            # The slab limit of two devices alone gives back the columns of a slab and not the rounds, which
            # are cut from it at whole multiples: the former rounds take the multiple of 1 beside it.
            del os.environ["PARSEC_CUPY_RITZ_GRAM_MULTIPLE"]
            self.assertEqual((distributed_state.slab_columns(rows, columns, 2, blocks=1), rounds((1494, 1484))),
                             (141, (12, 162, 256)))
        # The storage of the states: 14,680 electrons on two devices keep them where they fit, and with the
        # eleven values are spilled by the vectors against half of a device, which counts no bytes.
        layout = (self.K10, ((0,), (1,), (0,), (1,)))
        solver, events = self.decide(*layout, **later)
        self.assertEqual((solver._host_spill_sector_states, events), (False, []))
        self.assertEqual(solver.state_fit_bytes, {0: 81_064_975_872, 1: 81_064_975_872})
        solver, events = self.decide(*layout, **former, **later)
        self.assertEqual((solver._host_spill_sector_states, events, solver.state_fit_bytes),
                         (True, self.LEFT_POOL[0, 1], None))
        # The count of the new rule is that of the slabs of a budget: it does not move with the multiple.
        for multiple in ("1", "64"):
            solver, _events = self.decide(*layout, PARSEC_CUPY_RITZ_GRAM_MULTIPLE=multiple, **later)
            self.assertEqual(solver.state_fit_bytes, {0: 81_064_975_872, 1: 81_064_975_872})
        # The serial control of the runner names five of the eleven, each at this value.  The other six
        # it leaves alone: it shares no basis, and the report of the sphere is no route of the solver.
        control = {}
        mpi_full_scf._keep_former_routes(control)
        self.assertEqual({name: value for name, value in control.items() if name in former}, {
            name: former[name] for name in (
                "PARSEC_CUPY_DENSITY_RELEASE", "PARSEC_CUDA_CONTEXT_CREATION",
                "PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION", "PARSEC_CUPY_SPILL_DOWNLOAD_ORDER",
                "PARSEC_CUPY_RITZ_GRAM_MULTIPLE")})
        # And the limit of that buffer, at the value it had.
        self.assertEqual({name: control[name] for name in later}, later)
        # The projection kernel with a CUDA block per projector and column came after them, and the former
        # recurrence kernel of the filter after that.
        self.assertEqual(control.pop("PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS"), "1")
        self.assertEqual(control.pop("PARSEC_CUPY_STENCIL_KERNEL"), "former")
        # And the former values of the three switches of the preparation that came later: the ion-ion
        # energy, the symmetry maps and the grid of the ionic sums.
        preparation = {"PARSEC_OVERLAP_ION_ION_SETUP": "0", "PARSEC_OVERLAP_SYMMETRY_SETUP": "0",
                       "PARSEC_IONIC_SUM_GRID": "full"}
        self.assertEqual({name: control[name] for name in preparation}, preparation)
        # And, since, the former staging of a spill: the download by the former call belongs to it.
        self.assertEqual((control["PARSEC_CUPY_SPILL_BUFFERS"], control["PARSEC_CUPY_SPILL_PINNED_POOL"]),
                         ("fresh", "release"))
        self.assertEqual(len(control), 27 + len(preparation))

    def test_the_former_values_of_the_switches_of_the_round_after_select_every_former_route_together(self):
        import importlib
        from parsec_python.acceleration.benchmarks import mpi_full_scf
        from parsec_python.acceleration.Eigensolvers import distributed_state
        # The package also exports a function named like this submodule.
        ritz = importlib.import_module("parsec_python.acceleration.Eigensolvers.rayleigh_ritz")
        # Three defaults came after the nine above, and the rows behind theirs in that table of README.md give
        # each a former value: the slabs of the overlap of a shared basis, the buffer of a sector on one device and
        # the share of a device that the rule of this class holds its count against.  With the three named a
        # process cuts, takes and decides as it did; with none, as it does.
        former = dict(PARSEC_CUPY_RITZ_GRAM_SLABS="8", PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES="4294967296",
                      PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="0.978")
        # Sectors of 3,480 to 39,368 electrons: the nine clusters of the measured series.
        sectors = ((677_542, 442), (1_031_322, 665), (1_194_720, 897), self.K09, self.K10, self.K11,
                   (3_786_832, 2978), self.K13, (5_660_200, 4928))

        def cuts():
            # The slabs of the overlap of a shared basis (None: those of the count of slabs, which follow),
            # and the slab, the tile and the counted buffer of the same sector on one device.
            shapes = [ritz._streaming_shape(SimpleNamespace(shape=sector, nbytes=8 * sector[0] * sector[1]))
                      for sector in sectors]
            return ([distributed_state.overlap_slab_width(columns) for _rows, columns in sectors],
                    [ritz._gram_slab_width(columns) for _rows, columns in sectors],
                    [width for width, _tile in shapes], [tile for _width, tile in shapes],
                    [ritz.streaming_workspace_bytes(*sector) for sector in sectors])

        with patch.dict(os.environ):
            for name in (*former, "PARSEC_CUPY_STREAMING_RITZ_BYTES", "PARSEC_CUPY_RITZ_GRAM_MULTIPLE",
                         "PARSEC_CUPY_DISTRIBUTED_STATE_OVERLAP_COLUMNS"):
                os.environ.pop(name, None)
            overlap, counted, slabs, tiles, buffers = cuts()
            self.assertEqual(overlap, [128] * 9)
            self.assertEqual(counted, [56, 64, 64, 128, 192, 256, 320, 448, 576])
            self.assertEqual(slabs, [192, 128, 64, 128, 64, 64, 64, 64, 64])
            os.environ.update(former)
            before = cuts()
            # The count of slabs is every route's again, and the buffer an eighth of the basis within 1 and
            # 4 GiB: the slabs of the series that ran before these defaults.
            self.assertEqual(before[:3], ([None] * 9, counted, [192, 128, 64, 128, 192, 128, 128, 128, 64]))
            # Up to 16 GiB of basis the buffer is what it was: 3,480, 5,264 and 7,120 electrons.
            self.assertEqual((slabs[:3], tiles[:3], buffers[:3]), (before[2][:3], before[3][:3], before[4][:3]))
            # 10,456 electrons, 18.8 GiB: the slab of 128 columns stays under a lower tile.
            self.assertEqual((slabs[3], before[2][3], tiles[3], before[3][3]), (128, 128, 204_288, 240_599))
            self.assertTrue(all(new <= old for new, old in zip(tiles, before[3])))
            self.assertEqual([round(count / (1 << 30), 2) for count in buffers], [1., 1., 1., 2., 2., 2., 2., 2., 2.7])
            self.assertEqual([round(count / (1 << 30), 2) for count in before[4]], [1., 1., 1., 2.36, 4., 4., 4., 4., 4.])
        # The storage of the states.  10,456 electrons on one device are spilled either way, by 0.974 of the
        # device with the buffer of 2 GiB and by 0.978 with that of 2.36; 14,680 on two keep theirs either
        # way, and so does every layout of the measured series that kept them.
        for layout, spilled, counts in (((self.K09, ((0,),) * 4), True, (83_081_129_472, 83_462_830_704)),
                                        ((self.K10, ((0,), (1,), (0,), (1,))), False, (78_917_498_592, 81_064_975_872)),
                                        ((self.K11, self.ONE_PER_DEVICE), False, (58_029_350_808, 60_176_837_864)),
                                        ((self.K06, ((0,),) * 4), False, (10_656_895_808, 10_656_895_808))):
            with self.subTest(layout=layout):
                now, events = self.decide(*layout)
                then, former_events = self.decide(*layout, **former)
                self.assertEqual((now._host_spill_sector_states, then._host_spill_sector_states), (spilled, spilled))
                self.assertEqual((bool(events), bool(former_events)), (spilled, spilled))
                self.assertEqual((now.state_fit_bytes[0], then.state_fit_bytes[0]), counts)
                self.assertEqual((now.sector_state_storage, now.memory_allocator_policy),
                                 (then.sector_state_storage, then.memory_allocator_policy))
        # The serial control of the runner names the limit of the buffer, at this value.  It shares no basis
        # and decides the storage by the rule before this one, so it names neither of the other two.
        control = {}
        mpi_full_scf._keep_former_routes(control)
        self.assertEqual({name: value for name, value in control.items() if name in former},
                         dict(PARSEC_CUPY_STREAMING_RITZ_LIMIT_BYTES="4294967296"))

    def test_spilled_sectors_get_host_buffers_planned_within_a_bound_and_none_with_the_former_staging(self):
        from parsec_python.acceleration.Eigensolvers import spill_buffers
        # 10,456 electrons on one device are spilled by the rule: four bases of 18.84 GiB on the host.
        layout = (self.K09, ((0,),) * 4)
        basis = self.vectors(self.K09)
        self.assertEqual(basis, 20_233_413_504)
        former, former_events = self.decide(*layout, PARSEC_CUPY_SPILL_BUFFERS="fresh")
        self.assertEqual((former._host_spill_sector_states, former._spill_buffers), (True, None))
        self.assertEqual(former.spill_staging, dict(buffers="fresh", restores=0, restore_seconds=0, offloads=0,
                                                    offload_seconds=0))
        # A named bound of a byte less than the four: the first three may be page-locked, in the order of the
        # sectors.  What is spilled, the allocator and the pools are those of the former staging.
        solver, events = self.decide(*layout, PARSEC_CUPY_SPILL_PINNED_BYTES=str(4 * basis - 1))
        self.assertEqual((solver._host_spill_sector_states, solver.sector_state_storage, solver.memory_allocator_policy,
                          events, solver.state_fit_bytes),
                         (True, former.sector_state_storage, former.memory_allocator_policy, former_events,
                          former.state_fit_bytes))
        record = solver.spill_staging
        self.assertEqual((record["buffers"], record["bound_bytes"], record["bound_from"]),
                         ("persistent", 4 * basis - 1, "PARSEC_CUPY_SPILL_PINNED_BYTES"))
        self.assertEqual({sector: item["planned"] for sector, item in record["sectors"].items()},
                         {0: "page-locked", 1: "page-locked", 2: "page-locked", 3: "pageable"})
        self.assertEqual(solver._spill_buffers.planned_bytes, 3 * basis)
        # Nothing is taken here: a sector takes its buffer when it is first downloaded.
        self.assertEqual((record["page_locked_bytes"], record["pageable_bytes"], record["lock_seconds"],
                          [item["bytes"] for item in record["sectors"].values()]), (0, 0, 0, [0] * 4))
        # Without a named bound the host gives it: half of the memory it has available, the limit of the
        # system on locked memory where that is less, and nothing where its memory cannot be read.
        half, limit, unread = ("half of the available host memory", "RLIMIT_MEMLOCK",
                               "the available memory of the host could not be read")
        for available, locked, bound, source, planned in (
                (256 << 30, None, 128 << 30, half, 4), (256 << 30, 1 << 40, 128 << 30, half, 4),
                (100 << 30, None, 50 << 30, half, 2), (30 << 30, None, 15 << 30, half, 0),
                (256 << 30, 64 << 20, 64 << 20, limit, 0), (256 << 30, 40 << 30, 40 << 30, limit, 2),
                (None, None, 0, unread, 0), (None, 40 << 30, 0, unread, 0)):
            with self.subTest(available=available, locked=locked), \
                 patch.object(spill_buffers, "host_available_bytes", return_value=available), \
                 patch.object(spill_buffers, "locked_memory_limit", return_value=locked):
                record = self.decide(*layout)[0].spill_staging
                self.assertEqual((record["bound_bytes"], record["bound_from"]), (bound, source))
                self.assertEqual([item["planned"] for item in record["sectors"].values()],
                                 ["page-locked"] * planned + ["pageable"] * (4 - planned))
                # Named bytes are taken as they are, whatever the host says: all, and none.
                for named, locks in ((str(1 << 50), 4), ("0", 0)):
                    record = self.decide(*layout, PARSEC_CUPY_SPILL_PINNED_BYTES=named)[0].spill_staging
                    self.assertEqual((record["bound_bytes"], sum(item["planned"] == "page-locked"
                                                                  for item in record["sectors"].values())),
                                     (int(named), locks))
        # A rank plans for the sectors it solves: two of 19,392 electrons on one device, which cannot hold them.
        rank, _events = self.decide(self.K11, ((0,), (9,), (0,), (9,)), (0, 2), PARSEC_CUPY_SPILL_PINNED_BYTES=str(1 << 50))
        self.assertTrue(rank._host_spill_sector_states)
        self.assertEqual(sorted(rank.spill_staging["sectors"]), [0, 2])
        # A rank that drives three devices for two sectors gives the first a group of two, whose basis is to be
        # shared and takes no buffer, and the second one device.  With room for one basis the second is to be
        # locked, as with the first on its own device; the group is read and no device asked.
        from parsec_python.acceleration.Eigensolvers import symmetry
        for operators, planned in ((None, {0: "pageable", 1: "page-locked"}),
                                   ([SimpleNamespace(), SimpleNamespace(), None, None], {0: "page-locked", 1: "pageable"})):
            with patch.object(symmetry, "shared_basis_devices", side_effect=AssertionError("asked")):
                mixed, _events = self.decide(self.K09, ((0, 1), (2,), (9,), (9,)), (0, 1), operators=operators,
                                             PARSEC_CUPY_SECTOR_STATE_STORAGE="host",
                                             PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool",
                                             PARSEC_CUPY_SPILL_PINNED_BYTES=str(basis))
            self.assertEqual({sector: item["planned"] for sector, item in mixed.spill_staging["sectors"].items()}, planned)
        # Sectors that stay on their devices have no buffers and no record, whatever the staging is called.
        for environment in ({}, dict(PARSEC_CUPY_SPILL_BUFFERS="fresh"), dict(PARSEC_CUPY_SPILL_BUFFERS="persistent")):
            kept, _events = self.decide(self.K06, ((0,),) * 4, **environment)
            self.assertEqual((kept._host_spill_sector_states, kept._spill_buffers, kept.spill_staging), (False, None, None))


class SpillDownloadTests(unittest.TestCase):
    """The download of a spilled basis asks for the order the array has (PARSEC_CUPY_SPILL_DOWNLOAD_ORDER)."""

    @staticmethod
    def solver_with(vectors, eigenvalues):
        """A sector solver that holds these as the state of a finished solve."""
        from parsec_python.acceleration.Eigensolvers.eigval import CuPyEigvalDeviceState, CuPyEigvalSolver
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState
        rows, columns = vectors.shape
        solver = object.__new__(CuPyEigvalSolver)
        solver._state = CuPyEigvalDeviceState(
            operator_dimension=rows, requested_states=columns, working_states=columns, initial_method="chebff",
            subspace=DeviceSubspaceState(rows, columns, eigenvalues, vectors, filter_lower_bound=-1.5,
                                         first_filter=False, filters_completed=2, ritz_workspace=object()),
            solves_completed=1)
        return solver

    @staticmethod
    def offload(solver, value):
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_CUPY_SPILL_DOWNLOAD_ORDER", None)
            if value is not None:
                os.environ["PARSEC_CUPY_SPILL_DOWNLOAD_ORDER"] = value
            return solver.offload_state_to_host()

    def test_the_basis_is_downloaded_in_the_order_it_has_unless_c_order_is_named(self):
        from parsec_python.acceleration.Eigensolvers import eigval
        asked, taken, returned = [], [], []

        class DeviceArray:
            """What the download reads of a device array: its values, shape and order."""

            def __init__(self, values):
                self.values, self.shape, self.flags = values, values.shape, values.flags

        def asnumpy(array, order="C"):
            # As CuPy does it: an array that has not the order asked for is first copied on its device.
            asked.append(order)
            if order == "A":
                order = "F" if array.flags.f_contiguous else "C"
            if not (array.flags.c_contiguous if order == "C" else array.flags.f_contiguous):
                taken.append(array.values.nbytes)
            returned.append(np.array(array.values, order=order))
            return returned[-1]

        cp = SimpleNamespace(ndarray=DeviceArray, asnumpy=asnumpy)
        energies = np.array([3., 1., 2.])
        basis = np.asfortranarray(np.arange(12.).reshape(4, 3) / 7.)
        cases = (
            # The basis of a sector, whole and as the leading columns that truncate_state leaves: Fortran order.
            (basis, None, "A", 0), (basis, " a ", "A", 0), (basis[:, :2], None, "A", 0),
            # Named C order is the download as it was: a second array on the device and a copy on the host.
            (basis, "C", "C", 1), (basis, "c", "C", 1), (basis[:, :2], "C", "C", 1),
            # A C-ordered array has no second array on the device under either value.
            (np.ascontiguousarray(basis), None, "A", 0), (np.ascontiguousarray(basis), "C", "C", 0))
        for values, value, order, copies in cases:
            with self.subTest(value=value, shape=values.shape, f=values.flags.f_contiguous):
                del asked[:], taken[:], returned[:]
                solver = self.solver_with(DeviceArray(values), DeviceArray(energies[:values.shape[1]]))
                before = solver._state
                with patch.object(eigval, "require_cupy", lambda: (cp, None)):
                    state = self.offload(solver, value)
                # The eigenvalues first, as before, then the vectors in the order named.
                self.assertEqual(asked, ["C", order])
                self.assertEqual(taken, [values.nbytes] * copies)
                self.assertIs(solver._state, state)
                vectors = state.subspace.vectors
                self.assertTrue(vectors.flags.f_contiguous)
                self.assertEqual(vectors.dtype, np.float64)
                np.testing.assert_array_equal(vectors, values)
                np.testing.assert_array_equal(state.subspace.eigenvalues, energies[:values.shape[1]])
                # In its own order a Fortran-ordered array arrives as the host keeps it: no copy there.
                self.assertEqual(vectors is returned[-1], order == "A" and values.flags.f_contiguous)
                # All else of the state is kept, and the workspace of the Ritz step is dropped.
                self.assertIsNone(state.subspace.ritz_workspace)
                self.assertEqual((state.subspace.filter_lower_bound, state.subspace.first_filter,
                                  state.subspace.filters_completed, state.subspace.working_states,
                                  state.solves_completed, state.requested_states),
                                 (-1.5, False, 2, before.working_states, 1, before.requested_states))
        # Another value stops the download, and the configuration of the storage before the first solve.
        solver = self.solver_with(DeviceArray(basis), DeviceArray(energies))
        with patch.object(eigval, "require_cupy", lambda: (cp, None)), \
             self.assertRaisesRegex(ValueError, "PARSEC_CUPY_SPILL_DOWNLOAD_ORDER must be A or C"):
            self.offload(solver, "F")
        self.assertIsInstance(solver._state.subspace.vectors, DeviceArray)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class SpillDownloadDeviceTests(unittest.TestCase):
    """The download of a spilled basis on one real device: what it takes there, and that the bits come back."""

    def test_the_download_takes_no_second_array_on_the_device(self):
        import cupy as cp
        pool, taken = cp.get_default_memory_pool(), []

        def malloc(size):
            taken.append(int(size))
            return pool.malloc(size)

        rows, columns = 1 << 16, 24
        whole = cp.asfortranarray(cp.random.default_rng(11).random((rows, columns), dtype=cp.float64))
        energies = cp.arange(columns, dtype=cp.float64)
        reference = cp.asnumpy(whole)
        self.addCleanup(cp.cuda.set_allocator, cp.cuda.get_allocator())
        cp.cuda.set_allocator(malloc)
        # The whole basis, and the leading columns of it that truncate_state leaves as the saved vectors.
        for vectors in (whole, whole[:, :columns - 5]):
            self.assertTrue(vectors.flags.f_contiguous and not vectors.flags.c_contiguous)
            hosts = {}
            for value, copies in ((None, 0), ("C", 1)):
                with self.subTest(columns=vectors.shape[1], value=value):
                    solver = SpillDownloadTests.solver_with(vectors, energies[:vectors.shape[1]])
                    del taken[:]
                    state = SpillDownloadTests.offload(solver, value)
                    self.assertEqual(taken, [vectors.nbytes] * copies)
                    hosts[value] = host = state.subspace.vectors
                    self.assertIsInstance(host, np.ndarray)
                    self.assertTrue(host.flags.f_contiguous)
                    self.assertEqual(host.dtype, np.float64)
                    np.testing.assert_array_equal(host, reference[:, :vectors.shape[1]])
                    # The upload of the next step gives the device the same bits.
                    restored = solver.restore_state_to_device().subspace.vectors
                    self.assertIsInstance(restored, cp.ndarray)
                    self.assertTrue(restored.flags.f_contiguous)
                    self.assertTrue(bool(cp.array_equal(restored, vectors)))
            np.testing.assert_array_equal(hosts[None], hosts["C"])


class _StagedArray:
    """What the staging of a spill reads of a device array, on host memory; ``log`` says what was asked of it."""

    def __init__(self, values, log):
        self.values, self.log = values, log
        self.shape, self.dtype, self.ndim, self.size = values.shape, values.dtype, values.ndim, values.size
        self.flags, self.nbytes = values.flags, values.nbytes

    def get(self, out=None):
        # As CuPy does it: a target of the order of the array takes the bytes as they lie.
        self.log.append(("get", "F" if out.flags.f_contiguous else "C"))
        assert out.flags.f_contiguous and self.flags.f_contiguous
        out[...] = self.values
        return out

    def set(self, host):
        self.log.append(("set", host.shape))
        self.values[...] = host


class SpillBufferTests(unittest.TestCase):
    """The host buffers of spilled sectors and their direct copies (PARSEC_CUPY_SPILL_BUFFERS), on stand-ins."""

    def setUp(self):
        from parsec_python.acceleration.Eigensolvers import spill_buffers
        self.module = spill_buffers
        self.log, self.locked, self.refuse = [], [], None

        def lock(nbytes):
            # Page-locked memory of the system: exactly the bytes asked, or the refusal of the driver.
            self.locked.append(int(nbytes))
            if self.refuse is not None:
                raise self.refuse
            return bytearray(nbytes)

        def asnumpy(array, order="C"):
            self.log.append(("asnumpy", order))
            if order == "A":
                order = "F" if array.flags.f_contiguous and not array.flags.c_contiguous else "C"
            return np.array(array.values, order=order)

        def asarray(values, dtype=None, order=None):
            self.log.append(("asarray", np.shape(values), order))
            return _StagedArray(np.array(values, dtype=dtype, order=order), self.log)

        def empty(shape, dtype=None, order="C"):
            self.log.append(("empty", tuple(shape), order))
            return _StagedArray(np.empty(shape, dtype=dtype, order=order), self.log)

        self.cp = SimpleNamespace(
            ndarray=_StagedArray, float64=np.float64, asnumpy=asnumpy, asarray=asarray, empty=empty,
            cuda=SimpleNamespace(
                PinnedMemory=lock, PinnedMemoryPointer=lambda memory, offset: memory,
                get_current_stream=lambda: SimpleNamespace(synchronize=lambda: self.log.append(("synchronize",)))))

    def device(self, values):
        return _StagedArray(np.array(values), self.log)

    def buffers(self, sizes, bound, deferred=()):
        return self.module.SpillBuffers(sizes, (bound, "named by the test"), deferred)

    def test_the_switches_name_their_values_and_refuse_others(self):
        module = self.module
        with patch.dict(os.environ):
            for name in ("PARSEC_CUPY_SPILL_BUFFERS", "PARSEC_CUPY_SPILL_PINNED_BYTES"):
                os.environ.pop(name, None)
            self.assertEqual((module.spill_buffer_mode(), module.spill_pinned_bytes()), ("persistent", None))
            for raw, mode in ((" Fresh ", "fresh"), ("persistent", "persistent")):
                os.environ["PARSEC_CUPY_SPILL_BUFFERS"] = raw
                self.assertEqual(module.spill_buffer_mode(), mode)
                # The builder of a density reads it too, when a calculation is prepared.
                builder = CuPyDeviceDensityBuilder(SimpleNamespace())
                self.assertEqual((builder.direct_host_upload, builder.host_uploads),
                                 (mode == "persistent", dict(direct=mode == "persistent", calls=0, bytes=0, seconds=0.0)))
            for unknown in ("pinned", "0", "former", ""):
                os.environ["PARSEC_CUPY_SPILL_BUFFERS"] = unknown
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_SPILL_BUFFERS must be persistent or fresh"):
                    module.spill_buffer_mode()
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_SPILL_BUFFERS"):
                    CuPyDeviceDensityBuilder(SimpleNamespace())
            for raw, value in (("0", 0), (" 1073741824 ", 1 << 30), (str(1 << 50), 1 << 50)):
                os.environ["PARSEC_CUPY_SPILL_PINNED_BYTES"] = raw
                self.assertEqual(module.spill_pinned_bytes(), value)
                # A named value is the bound, whatever the host has.
                with patch.object(module, "host_available_bytes", return_value=1 << 20), \
                     patch.object(module, "locked_memory_limit", return_value=1 << 10):
                    self.assertEqual(module.page_lock_bound(), (value, "PARSEC_CUPY_SPILL_PINNED_BYTES"))
            for unknown, words in (("-1", "must be nonnegative"), ("1e9", "must be a nonnegative integer"),
                                   ("", "must be a nonnegative integer"), ("all", "must be a nonnegative integer")):
                os.environ["PARSEC_CUPY_SPILL_PINNED_BYTES"] = unknown
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_SPILL_PINNED_BYTES " + words):
                    module.spill_pinned_bytes()
            # What this host says of itself: bytes that are available now, and a limit or none.  The bound
            # reads the host once, and what is available changes from one reading to the next.
            del os.environ["PARSEC_CUPY_SPILL_PINNED_BYTES"]
            limit, read, readings = module.locked_memory_limit(), module.host_available_bytes, []

            def reading():
                readings.append(read())
                return readings[-1]

            with patch.object(module, "host_available_bytes", reading):
                bound, source = module.page_lock_bound()
            (available,) = readings
            self.assertTrue(available is None or available > 0)
            self.assertTrue(limit is None or limit >= 0)
            if available is None:
                self.assertEqual((bound, source), (0, "the available memory of the host could not be read"))
            elif limit is None or limit >= available // 2:
                self.assertEqual((bound, source), (available // 2, "half of the available host memory"))
            else:
                self.assertEqual((bound, source), (limit, "RLIMIT_MEMLOCK"))

    def test_the_host_is_read_as_linux_reports_it_and_not_at_all_where_it_reports_nothing(self):
        import builtins
        import sys
        from unittest.mock import mock_open
        module = self.module
        # MemAvailable of /proc/meminfo, in kB: what new work can take, which is more than the free pages.
        meminfo = ("MemTotal:       263567140 kB\nMemFree:        12345678 kB\nMemAvailable:   251658240 kB\n"
                   "Buffers:          123456 kB\n")
        with patch.object(builtins, "open", mock_open(read_data=meminfo)) as opened:
            self.assertEqual(module.host_available_bytes(), 251658240 * 1024)
        self.assertEqual(opened.call_args.args[0], "/proc/meminfo")
        # A kernel without that line, and a host without the file: the other readers of the platform, or none.
        def missing(*_arguments, **_named):
            raise FileNotFoundError("/proc/meminfo")

        for reader in (mock_open(read_data="MemTotal: 1 kB\nMemFree: 1 kB\n"), missing):
            with patch.object(builtins, "open", reader), patch.object(module.sys, "platform", "linux"), \
                 patch.object(module.os, "sysconf", create=True,
                              side_effect=lambda name: dict(SC_AVPHYS_PAGES=1000, SC_PAGE_SIZE=4096)[name]):
                self.assertEqual(module.host_available_bytes(), 1000 * 4096)
            with patch.object(builtins, "open", reader), patch.object(module.sys, "platform", "linux"), \
                 patch.object(module.os, "sysconf", create=True, side_effect=ValueError("unrecognized")), \
                 patch.dict(os.environ):
                os.environ.pop("PARSEC_CUPY_SPILL_PINNED_BYTES", None)
                self.assertIsNone(module.host_available_bytes())
                self.assertEqual(module.page_lock_bound(), (0, "the available memory of the host could not be read"))
        # The soft limit of the system on locked memory; none where it is unlimited or the platform has none.
        # A negative soft limit where the platform does not call that unlimited is read as none either.
        for soft, infinity, limit in ((65536, -1, 65536), (8 << 20, -1, 8 << 20), (-1, -1, None),
                                      ((1 << 63) - 1, (1 << 63) - 1, None), (0, -1, 0), (-5, (1 << 63) - 1, None)):
            resource = SimpleNamespace(RLIMIT_MEMLOCK=8, RLIM_INFINITY=infinity,
                                       getrlimit=lambda which, soft=soft: (soft, infinity))
            with patch.dict(sys.modules, resource=resource):
                self.assertEqual(module.locked_memory_limit(), limit)
                # Held against half of what the host has available, the lesser is the bound.
                with patch.object(module, "host_available_bytes", return_value=64 << 20), patch.dict(os.environ):
                    os.environ.pop("PARSEC_CUPY_SPILL_PINNED_BYTES", None)
                    self.assertEqual(module.page_lock_bound(), (
                        (32 << 20, "half of the available host memory") if limit is None or limit >= 32 << 20
                        else (limit, "RLIMIT_MEMLOCK")))
        with patch.dict(sys.modules, resource=None):
            self.assertIsNone(module.locked_memory_limit())
        failing = SimpleNamespace(RLIMIT_MEMLOCK=8, RLIM_INFINITY=-1, getrlimit=Mock(side_effect=OSError("not permitted")))
        with patch.dict(sys.modules, resource=failing):
            self.assertIsNone(module.locked_memory_limit())

    def test_the_buffers_that_fit_the_bound_are_to_be_locked_in_the_order_of_the_sectors(self):
        # Bytes of four sectors: the third no longer fits beside the first two, the fourth still does.
        owner = self.buffers({0: 40, 1: 40, 2: 40, 3: 16}, 100)
        self.assertEqual([owner.get(sector).lock for sector in range(4)], [True, True, False, True])
        self.assertEqual((owner.planned_bytes, owner.page_locked_bytes, self.locked), (96, 0, []))
        self.assertIsNone(owner.get(4))
        # The four bases of 10,456 electrons on a node that has 240 GiB available, and those of 19,392, of
        # which the host can lock two within half of that.
        for basis, locks in ((20_233_413_504, 4), (55_881_883_200, 2)):
            owner = self.buffers(dict.fromkeys(range(4), basis), (240 << 30) // 2)
            self.assertEqual(sum(owner.get(sector).lock for sector in range(4)), locks)
        # A bound of nothing locks nothing, and a basis of no bytes is not asked of the system either.
        self.assertEqual([item.lock for item in self.buffers({0: 8, 1: 8}, 0)._buffers.values()], [False, False])
        self.assertFalse(self.buffers({0: 0}, 100).get(0).lock)
        # Sectors whose basis a group of devices may share, which then takes no buffer, are planned after the
        # others with what the bound leaves; the record keeps the order of the sectors.
        owner = self.buffers({0: 40, 1: 40, 2: 40, 3: 16}, 100, deferred=(1,))
        self.assertEqual([owner.get(sector).lock for sector in range(4)], [True, False, True, True])
        owner = self.buffers({0: 40, 1: 40, 2: 40, 3: 16}, 100, deferred=[0])
        self.assertEqual([owner.get(sector).lock for sector in range(4)], [False, True, True, True])
        self.assertEqual((owner.planned_bytes, list(owner.record["sectors"])), (96, [0, 1, 2, 3]))
        # Every sector deferred is the plan in the order of the sectors again.
        owner = self.buffers({0: 40, 1: 40, 2: 40, 3: 16}, 100, deferred=range(4))
        self.assertEqual([owner.get(sector).lock for sector in range(4)], [True, True, False, True])

    def test_a_download_writes_into_the_buffer_of_its_sector_and_an_upload_reads_it_as_it_lies(self):
        cp = self.cp
        rng = np.random.default_rng(5)
        basis = np.asfortranarray(rng.normal(size=(4, 3)))
        owner = self.buffers({0: basis.nbytes, 1: basis.nbytes}, 1 << 20)
        buffer = owner.get(0)
        host = buffer.download(cp, self.device(basis))
        # One block of exactly the bytes of the basis is locked, and the array is read into it and waited for.
        self.assertEqual((self.locked, self.log), ([basis.nbytes], [("get", "F"), ("synchronize",)]))
        self.assertTrue(host.flags.f_contiguous and host.flags.writeable)
        self.assertEqual(host.dtype, np.float64)
        np.testing.assert_array_equal(host, basis)
        self.assertEqual((buffer.page_locked, buffer.nbytes, owner.page_locked_bytes, buffer.allocations),
                         (True, basis.nbytes, basis.nbytes, 1))
        # The next download of the sector writes over the same bytes: nothing is taken again.
        address = host.ctypes.data
        later = np.asfortranarray(rng.normal(size=(4, 3)))
        again = buffer.download(cp, self.device(later))
        self.assertEqual((again.ctypes.data, self.locked, buffer.allocations), (address, [basis.nbytes], 1))
        np.testing.assert_array_equal(host, later)
        # Fewer columns, as a trim leaves them, are the leading bytes of the buffer.
        del self.log[:]
        fewer = buffer.download(cp, self.device(later[:, :2]))
        self.assertEqual((fewer.shape, fewer.ctypes.data, fewer.flags.f_contiguous, buffer.allocations),
                         ((4, 2), address, True, 1))
        np.testing.assert_array_equal(fewer, later[:, :2])
        # An upload takes one device array and reads the host array where it lies; it is complete on return.
        del self.log[:]
        device = buffer.upload(cp, fewer)
        self.assertEqual(self.log, [("empty", (4, 2), "F"), ("set", (4, 2)), ("synchronize",)])
        np.testing.assert_array_equal(device.values, later[:, :2])
        self.assertTrue(device.values.flags.f_contiguous)
        self.assertEqual((buffer.uploads, buffer.upload_bytes, buffer.downloads, buffer.download_bytes,
                          buffer.reordered_downloads), (1, 64, 3, 96 + 96 + 64, 0))
        # The other sector has taken nothing, and the record of the process says so.
        record = owner.record
        self.assertEqual(record["sectors"], {
            0: dict(bytes=96, page_locked=True, planned="page-locked", allocations=1),
            1: dict(bytes=0, page_locked=False, planned="page-locked", allocations=0)})
        self.assertEqual({name: record[name] for name in (
            "bound_bytes", "bound_from", "page_locked_bytes", "pageable_bytes", "lock_failures", "uploads",
            "upload_bytes", "downloads", "reordered_downloads", "download_bytes")}, dict(
                bound_bytes=1 << 20, bound_from="named by the test", page_locked_bytes=96, pageable_bytes=0,
                lock_failures=[], uploads=1, upload_bytes=64, downloads=3, reordered_downloads=0, download_bytes=256))
        self.assertGreaterEqual(record["lock_seconds"], 0.0)

    def test_what_is_locked_stays_within_the_bound_while_an_outgrown_buffer_is_still_read(self):
        cp = self.cp
        owner = self.buffers({0: 96}, 200)
        buffer = owner.get(0)
        columns = lambda count: self.device(np.asfortranarray(np.arange(4. * count).reshape(4, count)))
        old = buffer.download(cp, columns(3))
        self.assertEqual((buffer.page_locked, owner.page_locked_bytes), (True, 96))
        # The sector grows while a result still reads the old buffer: 96 and 128 bytes are more than the bound.
        grown = buffer.download(cp, columns(4))
        self.assertEqual((buffer.page_locked, buffer.nbytes, owner.page_locked_bytes, self.locked), (False, 128, 96, [96]))
        np.testing.assert_array_equal(grown, np.arange(16.).reshape(4, 4))
        # The old one is intact for whoever reads it, and is counted until it has gone.
        np.testing.assert_array_equal(old, np.arange(12.).reshape(4, 3))
        del old
        self.assertEqual(owner.page_locked_bytes, 0)
        # Growing again, it locks what fits now ...
        wider = buffer.download(cp, columns(5))
        self.assertEqual((buffer.page_locked, owner.page_locked_bytes, self.locked), (True, 160, [96, 160]))
        # ... and with nothing left that reads it, the outgrown buffer goes back before the larger one is
        # counted: 160 and 192 bytes together would not fit.
        del wider
        widest = buffer.download(cp, columns(6))
        self.assertEqual((buffer.page_locked, owner.page_locked_bytes, self.locked, buffer.allocations),
                         (True, 192, [96, 160, 192], 4))
        np.testing.assert_array_equal(widest, np.arange(24.).reshape(4, 6))
        # A buffer that is as large as needed is never taken again.
        buffer.download(cp, columns(2))
        self.assertEqual((owner.page_locked_bytes, buffer.allocations), (192, 4))
        # When the solver and every array have gone, nothing is counted.
        del widest, buffer
        owner._buffers.clear()
        self.assertEqual(owner.page_locked_bytes, 0)

    def test_a_solver_that_has_gone_takes_its_locked_buffers_with_it(self):
        cp = self.cp
        owner = self.buffers({0: 96, 1: 64}, 200)
        hosts = [owner.get(sector).download(cp, self.device(np.asfortranarray(np.arange(4. * count).reshape(4, count))))
                 for sector, count in ((0, 3), (1, 2))]
        self.assertEqual(owner.page_locked_bytes, 160)
        # Nothing but the solver holds the buffers and nobody empties them by hand: what watches the locked
        # memory must not be what keeps the solver, or every calculation of a process would stay locked.
        gone, memory = weakref.ref(owner), [weakref.ref(owner.get(sector)) for sector in (0, 1)]
        del owner, hosts
        gc.collect()
        self.assertIsNone(gone())
        self.assertEqual([item() for item in memory], [None, None])

    def test_a_reset_leaves_an_earlier_result_its_values_and_counts_it_until_it_has_gone(self):
        cp = self.cp
        owner = self.buffers({0: 96}, 200)
        buffer = owner.get(0)
        columns = lambda scale: self.device(np.asfortranarray(scale * np.arange(12.).reshape(4, 3)))
        first = buffer.download(cp, columns(1.0))
        # Without a reset the same sector is downloaded into the same memory (one calculation: the states
        # of the last step are the ones wanted).
        again = buffer.download(cp, columns(2.0))
        self.assertTrue(np.shares_memory(first, again))
        np.testing.assert_array_equal(first, 2.0 * np.arange(12.).reshape(4, 3))
        # A reset starts another calculation on the same solver: the result of the one before keeps its
        # values, and its memory stays counted while it is read.
        owner.release()
        self.assertEqual((buffer.nbytes, buffer.page_locked, owner.page_locked_bytes), (0, False, 96))
        second = buffer.download(cp, columns(3.0))
        self.assertFalse(np.shares_memory(first, second))
        np.testing.assert_array_equal(first, 2.0 * np.arange(12.).reshape(4, 3))
        np.testing.assert_array_equal(second, 3.0 * np.arange(12.).reshape(4, 3))
        self.assertEqual((buffer.page_locked, owner.page_locked_bytes, self.locked), (True, 192, [96, 96]))
        del first, again
        self.assertEqual(owner.page_locked_bytes, 96)
        # With nothing that reads it, a reset gives the memory back at once.
        del second
        owner.release()
        self.assertEqual(owner.page_locked_bytes, 0)

    def test_a_lock_that_is_refused_or_not_planned_leaves_a_pageable_buffer_that_moves_the_same_bytes(self):
        cp = self.cp
        basis = np.asfortranarray(np.random.default_rng(6).normal(size=(5, 4)))
        # The driver refuses: the buffer is pageable, nothing stays counted, and the record says what it said.
        owner = self.buffers({0: basis.nbytes}, 1 << 20)
        self.refuse = MemoryError("out of host memory")
        refused = owner.get(0)
        host = refused.download(cp, self.device(basis))
        np.testing.assert_array_equal(host, basis)
        self.assertEqual((refused.page_locked, owner.page_locked_bytes, self.locked), (False, 0, [basis.nbytes]))
        record = owner.record
        self.assertEqual((record["lock_failures"], record["pageable_bytes"], record["page_locked_bytes"]),
                         (["sector 0: MemoryError: out of host memory"], basis.nbytes, 0))
        self.assertEqual(record["sectors"][0], dict(bytes=basis.nbytes, page_locked=False, planned="page-locked",
                                                     allocations=1))
        # The bound allows none: the system is not asked.
        self.refuse = None
        del self.locked[:], self.log[:]
        pageable = self.buffers({0: basis.nbytes}, basis.nbytes - 1).get(0)
        self.assertFalse(pageable.lock)
        kept = pageable.download(cp, self.device(basis))
        device = pageable.upload(cp, kept)
        # The same calls as with a locked buffer, and the same bytes on both sides.
        self.assertEqual((self.locked, self.log), ([], [("get", "F"), ("synchronize",), ("empty", (5, 4), "F"),
                                                        ("set", (5, 4)), ("synchronize",)]))
        self.assertEqual((pageable.page_locked, pageable.owner.record["lock_failures"]), (False, []))
        np.testing.assert_array_equal(kept, basis)
        np.testing.assert_array_equal(device.values, basis)

    def test_a_solver_moves_its_states_through_a_buffer_it_is_handed_and_by_the_former_calls_without(self):
        from parsec_python.acceleration.Eigensolvers import eigval
        cp = self.cp
        energies = np.array([3., 1., 2.])
        basis = np.asfortranarray(np.arange(12.).reshape(4, 3) / 7.)
        owner = self.buffers({0: basis.nbytes}, 1 << 20)
        buffer = owner.get(0)

        def moved(values, order=None, **handed):
            """The host state and the device state again of a solver that holds ``values``, and what each asked."""
            solver = SpillDownloadTests.solver_with(self.device(values), self.device(energies[:values.shape[1]]))
            with patch.object(eigval, "require_cupy", lambda: (cp, None)), patch.dict(os.environ):
                os.environ.pop("PARSEC_CUPY_SPILL_DOWNLOAD_ORDER", None)
                if order is not None:
                    os.environ["PARSEC_CUPY_SPILL_DOWNLOAD_ORDER"] = order
                del self.log[:]
                host = solver.offload_state_to_host(**handed)
                down = list(self.log)
                del self.log[:]
                device = solver.restore_state_to_device(**handed)
                return host.subspace, down, device.subspace, list(self.log)

        # Without a buffer the calls are the former ones: the vectors by cupy.asnumpy in the order named,
        # behind the eigenvalues, and back by cupy.asarray in Fortran order.
        for order, asked in ((None, "A"), ("C", "C")):
            host, down, device, up = moved(basis, order)
            self.assertEqual(down, [("asnumpy", "C"), ("asnumpy", asked)])
            self.assertEqual(up, [("asarray", (3,), None), ("asarray", (4, 3), "F")])
            self.assertTrue(host.vectors.flags.f_contiguous and host.vectors.flags.owndata)
            np.testing.assert_array_equal(host.vectors, basis)
            np.testing.assert_array_equal(device.vectors.values, basis)
        self.assertEqual((self.locked, buffer.nbytes), ([], 0))
        # With one, a Fortran-ordered basis goes down and up by one copy each, whole or as leading columns,
        # whatever the order of the former download is named; the eigenvalues go as they went.
        for values, order in ((basis, None), (basis, "C"), (basis[:, :2], None)):
            host, down, device, up = moved(values, order, buffer=buffer)
            self.assertEqual(down, [("asnumpy", "C"), ("get", "F"), ("synchronize",)])
            self.assertEqual(up, [("asarray", (values.shape[1],), None), ("empty", values.shape, "F"),
                                  ("set", values.shape), ("synchronize",)])
            # The host array of the state is the buffer, and holds what the former download returned.
            self.assertTrue(np.shares_memory(host.vectors, buffer._flat))
            self.assertTrue(host.vectors.flags.f_contiguous)
            np.testing.assert_array_equal(host.vectors, values)
            np.testing.assert_array_equal(host.eigenvalues, energies[:values.shape[1]])
            np.testing.assert_array_equal(device.vectors.values, values)
            self.assertTrue(device.vectors.values.flags.f_contiguous)
            self.assertIsNone(host.ritz_workspace)
            self.assertEqual((host.filter_lower_bound, host.filters_completed), (-1.5, 2))
        self.assertEqual((self.locked, buffer.allocations, buffer.reordered_downloads), ([basis.nbytes], 1, 0))
        # Vectors of another layout, as a solve without the column-major filter leaves them, are downloaded by
        # the former call, in the order named, and laid into the buffer in Fortran order; they go up as above.
        for values, order, asked in ((np.ascontiguousarray(basis), None, "A"), (np.ascontiguousarray(basis), "C", "C"),
                                     (np.ascontiguousarray(basis)[:, :2], None, "A")):
            host, down, device, up = moved(values, order, buffer=buffer)
            self.assertEqual(down, [("asnumpy", "C"), ("asnumpy", asked)])
            self.assertEqual(up[1:], [("empty", values.shape, "F"), ("set", values.shape), ("synchronize",)])
            self.assertTrue(np.shares_memory(host.vectors, buffer._flat))
            self.assertTrue(host.vectors.flags.f_contiguous)
            np.testing.assert_array_equal(host.vectors, values)
            np.testing.assert_array_equal(device.vectors.values, values)
        self.assertEqual((self.locked, buffer.allocations, buffer.reordered_downloads, buffer.downloads),
                         ([basis.nbytes], 1, 3, 6))
        # A state without columns goes by the former call into an array of its own: the buffer is not asked.
        host, down, device, up = moved(np.asfortranarray(basis[:, :0]), None, buffer=buffer)
        self.assertEqual((down, host.vectors.shape, host.vectors.flags.f_contiguous), ([("asnumpy", "C"), ("asnumpy", "A")],
                                                                                       (4, 0), True))
        self.assertFalse(np.shares_memory(host.vectors, buffer._flat))
        self.assertEqual(up[1:], [("asarray", (4, 0), "F")])
        self.assertEqual((self.locked, buffer.allocations, buffer.reordered_downloads, buffer.downloads),
                         ([basis.nbytes], 1, 3, 6))

    def test_vectors_of_another_layout_are_laid_into_the_buffer_column_after_column(self):
        cp = self.cp
        owner = self.buffers({0: 96}, 1 << 20)
        buffer = owner.get(0)
        first = buffer.download(cp, self.device(np.asfortranarray(np.arange(12.).reshape(4, 3))))
        # What the former call returns for vectors that are C-ordered on their device: a C-ordered host array,
        # here of other values than the buffer holds.
        downloaded = np.ascontiguousarray(-np.arange(12.).reshape(4, 3) / 3.)
        stored = buffer.store(cp, downloaded)
        self.assertTrue(np.shares_memory(stored, first) and stored.flags.f_contiguous)
        np.testing.assert_array_equal(stored, downloaded)
        # The buffer holds them in Fortran order, as the next upload reads it, and nothing was taken again.
        np.testing.assert_array_equal(buffer._flat[:12], downloaded.ravel(order="F"))
        self.assertEqual((buffer.downloads, buffer.reordered_downloads, buffer.download_bytes, buffer.allocations,
                          self.locked), (2, 1, 192, 1, [96]))
        # Fewer columns are the leading bytes of the buffer, again column after column.
        fewer = np.ascontiguousarray(2. * downloaded[:, :2])
        np.testing.assert_array_equal(buffer.store(cp, fewer), fewer)
        np.testing.assert_array_equal(buffer._flat[:8], fewer.ravel(order="F"))
        np.testing.assert_array_equal(buffer._flat[8:], downloaded.ravel(order="F")[8:])
        self.assertEqual((buffer.reordered_downloads, buffer.download_bytes), (2, 256))

    def test_a_builder_without_a_device_reads_host_orbitals_as_they_are(self):
        # NumPy standing in for CuPy, as the shims of tests have it: no density kernel and nothing to upload to,
        # so host orbitals are neither copied directly nor counted as uploads, whatever the staging.
        shim = SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray, empty=np.empty, float64=np.float64,
                               cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
        orbitals = np.asfortranarray(np.random.default_rng(9).normal(size=(6, 3)))
        weights = np.array([1.0, 0.5, 0.0])
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_CUPY_SPILL_BUFFERS", None)
            builder = CuPyDeviceDensityBuilder(shim)
        self.assertEqual((builder.kernel, builder.direct_host_upload), (None, True))
        np.testing.assert_allclose(builder(orbitals, weights, 0.5), 4.0 * np.sum(orbitals ** 2 * weights, axis=1),
                                   rtol=1e-14)
        self.assertEqual(builder.host_uploads, dict(direct=True, calls=0, bytes=0, seconds=0.0))

    def make_solver(self, staging, shared=()):
        """A process that keeps the states of two sectors on the host, with stand-ins for their solvers.

        The basis of a sector in ``shared`` stays on the devices that share it: the two calls leave it there.
        """
        calls = []

        def sector(index):
            # Where the saved vectors of the sector are, as the solver says: a solve leaves them on its device
            # or on the devices that share them, a spill moves those of one device to the host and back.
            state = SimpleNamespace(subspace=SimpleNamespace(vectors=None))

            def solve(count, settings, spectral_bound=None):
                calls.append(("solve", index))
                state.subspace.vectors = "shared basis" if index in shared else "device vectors"
                return CuPyEigvalResult(np.zeros(count), "device vectors", None, "device state", "subspace",
                                        False, None, count, count, 0.0)

            def restore(*handed):
                calls.append(("restore", index, handed))
                if isinstance(state.subspace.vectors, np.ndarray):
                    state.subspace.vectors = "device vectors"
                return state

            def offload(*handed):
                calls.append(("offload", index, handed))
                if isinstance(state.subspace.vectors, str) and state.subspace.vectors == "device vectors":
                    state.subspace.vectors = np.full(2, float(index))
                return state

            return SimpleNamespace(solve=solve, reset=lambda: calls.append(("reset", index)), device_state=state,
                                   restore_state_to_device=restore, offload_state_to_host=offload)

        class Place:
            def __init__(self, *_arguments):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_error):
                return False

        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._sector_device_ids = (0, 0)
        solver._streams = [Place(), Place()]
        solver._solvers = [sector(0), sector(1)]
        solver._host_spill_sector_states = True
        solver._spill_staging = staging
        solver._spill_buffers = self.buffers({0: 96, 1: 96}, 1 << 20) if staging == "persistent" else None
        solver._spill_seconds = {}
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=Place))
        patcher = patch("parsec_python.acceleration.Eigensolvers.symmetry.require_cupy", return_value=(cp, None))
        patcher.start()
        self.addCleanup(patcher.stop)
        return solver, calls

    def test_a_sector_is_handed_its_own_buffer_and_none_with_the_former_staging(self):
        solver, calls = self.make_solver("persistent")
        first, second = (solver._spill_buffers.get(index) for index in (0, 1))
        self.assertIsNot(first, second)
        # The first solve of a sector has nothing to bring back; later ones restore before and offload behind.
        result = solver._run_one_sector(1, 3, EigvalSettings(), reset=True)
        self.assertEqual(calls, [("reset", 1), ("solve", 1), ("offload", 1, (second,))])
        np.testing.assert_array_equal(result.vectors, [1.0, 1.0])
        del calls[:]
        for index in (0, 1, 0):
            solver._run_one_sector(index, 3, EigvalSettings(), reset=False)
        self.assertEqual(calls, [("restore", 0, (first,)), ("solve", 0), ("offload", 0, (first,)),
                                 ("restore", 1, (second,)), ("solve", 1), ("offload", 1, (second,)),
                                 ("restore", 0, (first,)), ("solve", 0), ("offload", 0, (first,))])
        record = solver.spill_staging
        # Sector 0 had no state before its first restore, which moved nothing and is not counted.
        self.assertEqual((record["buffers"], record["restores"], record["offloads"]), ("persistent", 2, 4))
        self.assertGreaterEqual(min(record["restore_seconds"], record["offload_seconds"]), 0.0)
        # The record of the buffers is in it.
        self.assertEqual((record["bound_bytes"], sorted(record["sectors"])), (1 << 20, [0, 1]))
        # The former staging: the two calls as they were made, with no argument, and counted like the others.
        solver, calls = self.make_solver("fresh")
        solver._run_one_sector(0, 3, EigvalSettings(), reset=True)
        solver._run_one_sector(0, 3, EigvalSettings(), reset=False)
        self.assertEqual(calls, [("reset", 0), ("solve", 0), ("offload", 0, ()),
                                 ("restore", 0, ()), ("solve", 0), ("offload", 0, ())])
        record = solver.spill_staging
        self.assertEqual(sorted(record), ["buffers", "offload_seconds", "offloads", "restore_seconds", "restores"])
        self.assertEqual((record["buffers"], record["restores"], record["offloads"]), ("fresh", 1, 2))
        # States that stay on their devices are not moved and have no record.
        solver._host_spill_sector_states, solver._spill_staging = False, None
        del calls[:]
        solver._run_one_sector(0, 3, EigvalSettings(), reset=False)
        self.assertEqual((calls, solver.spill_staging), ([("solve", 0)], None))
        # A basis that the devices of its sector share never leaves them: the two calls are made, under either
        # staging, and return without a move, which is not counted.  The sector beside it on one device is.
        for staging in ("persistent", "fresh"):
            solver, calls = self.make_solver(staging, shared=(1,))
            for reset in (True, False, False):
                for index in (0, 1):
                    solver._run_one_sector(index, 3, EigvalSettings(), reset=reset)
            self.assertEqual([call[:2] for call in calls if call[0] in ("restore", "offload")],
                             [("offload", 0), ("offload", 1)] + [("restore", 0), ("offload", 0),
                                                                 ("restore", 1), ("offload", 1)] * 2)
            record = solver.spill_staging
            self.assertEqual((record["restores"], record["offloads"]), (2, 3))
            self.assertEqual(solver._spill_seconds[1], [0, 0.0, 0, 0.0])
            self.assertEqual(solver._spill_seconds[0][::2], [2, 3])

    def test_the_detail_of_a_run_names_the_buffers_that_its_sectors_took_and_not_the_bound(self):
        from parsec_python.acceleration.SCF.single_point import _finalize_result
        from parsec_python.acceleration.models import BackendInfo

        def details(staging):
            # The backend details that a run of the hybrid driver ends with, stand-ins for what it reads.
            backend = SimpleNamespace(synchronize_statistics=lambda: None, statistics=SimpleNamespace(snapshot=lambda: None))
            system = SimpleNamespace(materialize_final_wavefunctions=False, backend=backend, backend_info=BackendInfo(
                "cupy", "cupy", details=(("orbital_sector_spill_staging", "of an earlier run"),)))
            solver = SimpleNamespace(state=SimpleNamespace(sector_state_counts=(3, 3)), memory_allocator_policy="pool",
                                     sector_state_storage="host", spill_staging=staging)
            return dict(_finalize_result(system, "result", solver).backend.details)

        def staging(sectors, bound, locked, pageable):
            return dict(buffers="persistent", bound_bytes=bound, bound_from="half of the available host memory",
                        page_locked_bytes=locked, pageable_bytes=pageable, sectors=dict(enumerate(sectors)))

        taken = (dict(bytes=96, page_locked=True, planned="page-locked", allocations=1),
                 dict(bytes=64, page_locked=False, planned="pageable", allocations=2))
        shared = dict(bytes=0, page_locked=False, planned="page-locked", allocations=0)
        expected = "persistent host buffers with direct copies; 1 of 2 page-locked (96 B), 64 B pageable"
        # What the sectors took, and not the bound: two launches on a host with other memory available have the
        # same detail, which the record of the staging tells apart.
        for bound in (1 << 30, 1 << 20):
            self.assertEqual(details(staging(taken, bound, 96, 64))["orbital_sector_spill_staging"], expected)
        # A sector whose basis is shared took no buffer and is not among them.
        self.assertEqual(details(staging((shared, *taken), 1 << 30, 96, 64))["orbital_sector_spill_staging"], expected)
        # Where no sector took one, as with every basis shared, and where the states stay on their devices or move by
        # the former calls, a run has the details it had: none of a spill.
        for record in (staging((shared, shared), 1 << 30, 0, 0), None,
                       dict(buffers="fresh", restores=2, restore_seconds=1.0, offloads=3, offload_seconds=1.0)):
            self.assertNotIn("orbital_sector_spill_staging", details(record))


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class SpillBufferDeviceTests(unittest.TestCase):
    """The buffers of spilled sectors on one real device: what a copy takes, and that the bits arrive."""

    def setUp(self):
        import cupy as cp
        from parsec_python.acceleration.Eigensolvers import spill_buffers
        self.cp, self.module = cp, spill_buffers
        pool, pinned = cp.get_default_memory_pool(), cp.cuda.PinnedMemoryPool()
        self.taken, self.staged, self.locked = [], [], []

        def malloc(size):
            self.taken.append(int(size))
            return pool.malloc(size)

        def stage(size):
            # What CuPy asks of its pinned pool: the staging block of an upload.
            self.staged.append(int(size))
            return pinned.malloc(size)

        system = cp.cuda.PinnedMemory

        def lock(size):
            # Page-locked memory taken from the system directly.
            self.locked.append(int(size))
            return system(size)

        self.addCleanup(cp.cuda.set_allocator, cp.cuda.get_allocator())
        self.addCleanup(cp.cuda.set_pinned_memory_allocator, cp.get_default_pinned_memory_pool().malloc)
        self.addCleanup(pinned.free_all_blocks)
        for patcher in (patch.object(cp.cuda, "PinnedMemory", lock), patch.dict(os.environ)):
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("PARSEC_CUPY_SPILL_BUFFERS", "PARSEC_CUPY_SPILL_PINNED_BYTES", "PARSEC_CUPY_SPILL_DOWNLOAD_ORDER"):
            os.environ.pop(name, None)
        self.rows, self.columns = 1 << 16, 24
        self.whole = cp.asfortranarray(cp.random.default_rng(13).random((self.rows, self.columns), dtype=cp.float64))
        # In the order it has on the device, as a spill keeps it on the host.
        self.reference = cp.asnumpy(self.whole, order="A")
        cp.cuda.set_allocator(malloc)
        cp.cuda.set_pinned_memory_allocator(stage)

    def test_a_copy_through_a_buffer_takes_no_staging_block_and_no_second_array(self):
        cp = self.cp
        nbytes = self.whole.nbytes
        for bound, locked in ((1 << 40, True), (0, False)):
            for stream in (cp.cuda.Stream.null, cp.cuda.Stream(non_blocking=True)):
                with self.subTest(locked=locked, null=stream is cp.cuda.Stream.null), stream:
                    buffer = self.module.SpillBuffers({0: nbytes}, (bound, "named by the test")).get(0)
                    del self.taken[:], self.staged[:], self.locked[:]
                    host = buffer.download(cp, self.whole)
                    # Nothing on the device and nothing from the pinned pool; one block of exactly the bytes
                    # of the basis from the system where the bound allows it, which the pool would round up.
                    self.assertEqual((self.taken, self.staged, self.locked), ([], [], [nbytes] * locked))
                    self.assertEqual((buffer.page_locked, buffer.nbytes), (locked, nbytes))
                    self.assertTrue(host.flags.f_contiguous)
                    np.testing.assert_array_equal(host, self.reference)
                    # The leading columns that a trim leaves, down into the same bytes and up again.
                    fewer = buffer.download(cp, self.whole[:, :self.columns - 5])
                    self.assertEqual((self.taken, self.staged, len(self.locked), buffer.allocations),
                                     ([], [], int(locked), 1))
                    np.testing.assert_array_equal(fewer, self.reference[:, :self.columns - 5])
                    device = buffer.upload(cp, fewer)
                    # One device array of the bytes that go up, complete when the call is back.
                    self.assertEqual((self.taken, self.staged), ([fewer.nbytes], []))
                    self.assertTrue(device.flags.f_contiguous)
                    fewer[...] = -1.0
                    self.assertTrue(bool(cp.array_equal(device, self.whole[:, :self.columns - 5])))
                    del device
        # The former upload of the same array stages all of it in a block of the pinned pool.
        del self.taken[:], self.staged[:]
        former = cp.asarray(self.reference, dtype=cp.float64, order="F")
        self.assertEqual((self.taken, self.staged), ([nbytes], [nbytes]))
        self.assertTrue(bool(cp.array_equal(former, self.whole)))

    def test_a_solver_spills_and_restores_the_same_bits_through_a_buffer_as_by_the_former_calls(self):
        cp = self.cp
        energies = cp.arange(self.columns, dtype=cp.float64)
        owner = self.module.SpillBuffers({0: self.whole.nbytes}, (1 << 40, "named by the test"))
        # The basis as a column-major solve leaves it, its leading columns, and one in C order.
        for vectors, reordered in ((self.whole, 0), (self.whole[:, :self.columns - 5], 0), (cp.ascontiguousarray(self.whole), 1)):
            with self.subTest(columns=vectors.shape[1], reordered=reordered):
                states = {}
                for name, handed in (("former", {}), ("buffer", dict(buffer=owner.get(0)))):
                    solver = SpillDownloadTests.solver_with(vectors, energies[:vectors.shape[1]])
                    before = owner.get(0).reordered_downloads
                    host = solver.offload_state_to_host(**handed).subspace
                    self.assertIsInstance(host.vectors, np.ndarray)
                    self.assertTrue(host.vectors.flags.f_contiguous)
                    kept = host.vectors.copy()
                    restored = solver.restore_state_to_device(**handed).subspace
                    self.assertIsInstance(restored.vectors, cp.ndarray)
                    self.assertTrue(restored.vectors.flags.f_contiguous)
                    states[name] = (kept, cp.asnumpy(restored.vectors), cp.asnumpy(restored.eigenvalues), host.eigenvalues)
                    if handed:
                        self.assertEqual(owner.get(0).reordered_downloads - before, reordered)
                for former, buffered in zip(states["former"], states["buffer"]):
                    np.testing.assert_array_equal(buffered, former)
                np.testing.assert_array_equal(states["buffer"][0], self.reference[:, :vectors.shape[1]])
        self.assertEqual((self.locked, owner.get(0).allocations), ([self.whole.nbytes], 1))

    def test_the_density_of_host_orbitals_is_the_same_by_a_direct_upload_as_by_the_former_one(self):
        cp = self.cp
        weights = np.linspace(0.0, 1.0, self.columns - 5)
        densities = {}
        for mode in ("persistent", "fresh"):
            os.environ["PARSEC_CUPY_SPILL_BUFFERS"] = mode
            builder = CuPyDeviceDensityBuilder(cp)
            self.assertEqual(builder.direct_host_upload, mode == "persistent")
            # The columns that a density is handed: the leading ones of a Fortran-ordered host array, a view.
            orbitals = self.reference[:, :self.columns - 5]
            self.assertTrue(orbitals.flags.f_contiguous and not orbitals.flags.owndata)
            del self.taken[:], self.staged[:]
            densities[mode] = builder(orbitals, weights, 0.25)
            # The upload takes the device array either way, and the staging block only by the former call.
            self.assertIn(orbitals.nbytes, self.taken)
            self.assertEqual(orbitals.nbytes in self.staged, mode == "fresh")
            self.assertEqual({name: builder.host_uploads[name] for name in ("direct", "calls", "bytes")},
                             dict(direct=mode == "persistent", calls=1, bytes=orbitals.nbytes))
            # A host array of another layout takes the former call under either value, and orbitals that are
            # on the device are not uploaded.
            del self.staged[:]
            c_ordered = np.ascontiguousarray(orbitals)
            np.testing.assert_array_equal(builder(c_ordered, weights, 0.25), densities[mode])
            self.assertIn(c_ordered.nbytes, self.staged)
            np.testing.assert_array_equal(builder(self.whole[:, :self.columns - 5], weights, 0.25), densities[mode])
            self.assertEqual(builder.host_uploads["calls"], 2)
        np.testing.assert_array_equal(densities["persistent"], densities["fresh"])
        expected = 8.0 * np.sum(self.reference[:, :self.columns - 5] ** 2 * weights, axis=1)
        np.testing.assert_allclose(densities["persistent"], expected, rtol=1e-13)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class SectorPoolReleaseDeviceTests(unittest.TestCase):
    """The pool of one real device: sectors with streams of their own, solved one after another."""

    def make_solver(self, solves, release):
        import cupy as cp
        device = int(cp.cuda.Device().id)
        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver._sector_device_ids = (device,)*len(solves)
        solver._streams = [cp.cuda.Stream(non_blocking=True) for _ in solves]
        solver._solvers = [SimpleNamespace(solve=solve) for solve in solves]
        solver._pool_releases = {device: [0, 0]} if release else {}
        # Vectors of 1 GiB a sector, above the limit of the density step.
        solver.decomposition = SimpleNamespace(sector_size=lambda representation: 1 << 20)
        solver._owned_representations = tuple(range(len(solves)))
        solver._sector_counts = [128]*len(solves)
        patcher = patch.dict("os.environ")
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES", None)
        return solver, device

    def test_second_sector_takes_no_second_buffer_once_the_first_has_released(self):
        import cupy as cp
        pool = cp.get_default_memory_pool()
        size = 64 << 20
        for release in (False, True):
            with self.subTest(release=release):
                gc.collect()
                pool.free_all_blocks()
                base = pool.total_bytes()
                held = []

                def solve(count, settings, spectral_bound=None):
                    # As a Ritz step: a buffer taken on the stream of the sector and given back.
                    buffer = cp.empty(size // 8, dtype=cp.float64)
                    buffer[:8] = 1.0
                    held.append(pool.total_bytes() - base)
                    del buffer
                    cp.cuda.get_current_stream().synchronize()
                    return SimpleNamespace(vectors=None)

                solver, device = self.make_solver([solve, solve], release)
                for representation in range(2):
                    solver._run_one_sector(representation, 1, EigvalSettings(), reset=False)
                # Without the release the buffer of the first sector stays beside that of the second.
                self.assertEqual(held, [size, size if release else 2*size])
                self.assertEqual(pool.total_bytes() - base, 0 if release else 2*size)
                self.assertEqual(solver.sector_pool_releases,
                                 {device: dict(releases=2, bytes=2*size)} if release else {})
                del solver
                pool.free_all_blocks()

    def test_a_block_with_a_part_in_use_stays_and_what_lies_in_it_is_intact(self):
        import cupy as cp
        pool = cp.get_default_memory_pool()
        size = 64 << 20
        gc.collect()
        pool.free_all_blocks()
        base = pool.total_bytes()
        kept = []

        def solve(count, settings, spectral_bound=None):
            buffer = cp.empty(size // 8, dtype=cp.float64)
            del buffer
            # A result that outlives the solve, taken out of the block that the buffer gave back.
            kept.append(cp.arange(1000, dtype=cp.float64))
            cp.cuda.get_current_stream().synchronize()
            return SimpleNamespace(vectors=None)

        solver, device = self.make_solver([solve], True)
        solver._run_one_sector(0, 1, EigvalSettings(), reset=False)
        self.assertEqual(pool.total_bytes() - base, size)
        self.assertEqual(solver.sector_pool_releases, {device: dict(releases=1, bytes=0)})
        np.testing.assert_array_equal(cp.asnumpy(kept[0]), np.arange(1000.0))
        kept.clear()
        pool.free_all_blocks()
        self.assertEqual(pool.total_bytes() - base, 0)


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class MultiGPUResidencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cupy as cp
        cls.cp = cp
        if cp.cuda.runtime.getDeviceCount() < 2:
            raise unittest.SkipTest("requires two allocated CUDA devices")

    def test_density_reduces_on_source_device_and_restores_caller(self):
        cp = self.cp
        backing = np.random.default_rng(12).normal(size=(127, 12))
        host = backing[:, :11]
        weights = np.linspace(0.0, 1.0, 11)
        with cp.cuda.Device(1):
            vectors = cp.asarray(backing, order="C")[:, :11]
            self.assertFalse(vectors.flags.c_contiguous or vectors.flags.f_contiguous)
        with cp.cuda.Device(0):
            builder = CuPyDeviceDensityBuilder(cp)
            original = cp.asarray

            def require_local(array, *args, **kwargs):
                if isinstance(array, cp.ndarray):
                    self.assertEqual(array.device.id, cp.cuda.Device().id,
                                     "density must not transfer orbital blocks")
                return original(array, *args, **kwargs)

            with patch.object(cp, "asarray", side_effect=require_local):
                actual = builder(vectors, weights, 0.25)
            self.assertEqual(cp.cuda.Device().id, 0)
        expected = 8.0 * np.sum(host * host * weights, axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=2e-14)

    def test_blocked_host_export_preserves_phases_stabilizers_and_column_order(self):
        cp = self.cp
        rng = np.random.default_rng(42)
        hosts = (rng.normal(size=(4, 3)), rng.normal(size=(3, 2)))
        sector_orbits = (np.arange(4), np.array([0, 2, 3]))
        scales = (np.array([.5, 1., .5, 1.]), np.array([.5, .5, 1.]))
        mapping = np.array([0, 1, 2, 3, 0, 2])
        phases = np.array([[1]*6, [1, 0, 1, 1, -1, -1]], dtype=float)
        reps = np.array([1, 0, 0, 1, 0], dtype=np.int32)
        cols = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        devices = []
        for i, values in enumerate(hosts):
            with cp.cuda.Device(i):
                devices.append(cp.asarray(values, order="F"))
        with cp.cuda.Device(0):
            orbitals = CuPySymmetryOrbitals(
                None, reps, mapping, cp.asarray(mapping), cp.asarray(phases), 6,
                cols, tuple(devices), sector_orbits, scales, 4,
            )
            result = orbitals.to_full_host(block_states=1)
            self.assertEqual(cp.cuda.Device().id, 0)
        expected = np.empty((6, 5))
        for j, (rep, col) in enumerate(zip(reps, cols)):
            wedge = np.zeros(4)
            wedge[sector_orbits[rep]] = hosts[rep][:, col] * scales[rep]
            expected[:, j] = wedge[mapping] * phases[rep]
        np.testing.assert_array_equal(result, expected)


@unittest.skipUnless(cupy_available() and native_available(),
                     "a real CUDA device and the native extension are required")
class SectorStateStorageDeviceTests(unittest.TestCase):
    """The rule of the state storage in a complete SCF on one real device."""

    NAMES = ("PARSEC_CUPY_SECTOR_STATE_STORAGE", "PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR",
             "PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION", "PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION",
             "PARSEC_CUPY_SPILL_DOWNLOAD_ORDER", "PARSEC_CUPY_SPILL_BUFFERS", "PARSEC_CUPY_SPILL_PINNED_POOL",
             "PARSEC_CUPY_SPILL_PINNED_BYTES", "PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES",
             "PARSEC_CUPY_GENERALIZED_RITZ")

    def tearDown(self):
        gc.collect()

    def run_scf(self, **environment):
        """Three SCF steps of the smoke input under these values alone; the result, its two report lines and
        the bytes that ``auto`` counted."""
        from dataclasses import replace
        from pathlib import Path
        from parsec_python.Input import parse_parsec_input
        from parsec_python.acceleration.driver import prepare_single_point, run_scf
        source = Path(__file__).resolve().parents[2] / "tests" / "data" / "H_cli_smoke.in"
        problem = parse_parsec_input(source).problem
        # Three steps reach the later passes, which take up the states that the first solve left.
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        with patch.dict(os.environ, PARSEC_CUPY_DEVICES="0"):
            for name in self.NAMES:
                os.environ.pop(name, None)
            os.environ.update(environment)
            system = prepare_single_point(problem, backend="auto")
            result = run_scf(system)
        details = dict(result.backend.details)
        # Of the last run: its details, and what its solver and its density builder recorded of a spill.
        self.details = details
        self.staging = system.backend.symmetry_eigensolver.spill_staging
        self.release = dict(system.orbital_density_builder.pool_release)
        return (result, details["orbital_sector_state_storage"], details["orbital_memory_allocator"],
                system.backend.symmetry_eigensolver.state_fit_bytes)

    def assert_same_bits(self, actual, expected):
        self.assertEqual(actual.iterations, expected.iterations)
        np.testing.assert_array_equal(actual.eigenvalues, expected.eigenvalues)
        np.testing.assert_array_equal(actual.occupations, expected.occupations)
        np.testing.assert_array_equal(actual.density, expected.density)
        np.testing.assert_array_equal(actual.hartree_potential, expected.hartree_potential)
        self.assertEqual(actual.energies.total, expected.energies.total)
        for step, reference in zip(actual.history, expected.history, strict=True):
            np.testing.assert_array_equal(step.eigenvalues, reference.eigenvalues)

    def test_a_run_that_names_nothing_is_the_run_that_names_device_and_pool(self):
        import cupy as cp
        allocator = cp.cuda.get_allocator()
        named, *named_lines, named_count = self.run_scf(
            PARSEC_CUPY_SECTOR_STATE_STORAGE="device", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool")
        unnamed, *unnamed_lines, count = self.run_scf()
        # The same route by its two report lines, and not one bit of the results moves.
        self.assertEqual(unnamed_lines, named_lines)
        self.assertEqual(unnamed_lines[0], "persistent CUDA representation states")
        self.assertTrue(unnamed_lines[1].startswith("cupy default pool; estimated persistent sector orbitals cuda:0="))
        self.assert_same_bits(unnamed, named)
        # Only the unnamed run counted, and far less than the device has.
        self.assertIsNone(named_count)
        self.assertEqual(list(count), [0])
        with cp.cuda.Device(0):
            self.assertLess(0, count[0])
            self.assertLess(count[0], cp.cuda.runtime.memGetInfo()[1])
        # The former rule with a fraction that every device reaches is the spill and the direct allocator
        # by their names, bit for bit, and gives the allocator back when the SCF has ended.
        spilled, *spilled_lines, _count = self.run_scf(
            PARSEC_CUPY_SECTOR_STATE_STORAGE="host", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="direct")
        former, *former_lines, former_count = self.run_scf(PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION="1e-12")
        self.assertEqual(former_lines, spilled_lines)
        self.assertEqual(former_lines[0], "exact FP64 host spill; one active representation on CUDA")
        self.assertTrue(former_lines[1].startswith("direct CUDA allocation for memory-bound sectors; "))
        self.assertIsNone(former_count)
        self.assert_same_bits(former, spilled)
        self.assertEqual(cp.cuda.get_allocator(), allocator)
        # So is a share of the device that the count exceeds, and all of the device changes nothing here.
        by_share, *share_lines, share_count = self.run_scf(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="1e-9")
        self.assertEqual((share_lines, share_count), (spilled_lines, count))
        self.assert_same_bits(by_share, spilled)
        self.assertEqual(cp.cuda.get_allocator(), allocator)
        whole, *whole_lines, whole_count = self.run_scf(PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION="1")
        self.assertEqual((whole_lines, whole_count), (unnamed_lines, count))
        self.assert_same_bits(whole, unnamed)
        # The spill downloads its states in the order they have; in C order, as it did, the bits are the same.
        c_order, *c_lines, _count = self.run_scf(
            PARSEC_CUPY_SECTOR_STATE_STORAGE="host", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="direct",
            PARSEC_CUPY_SPILL_DOWNLOAD_ORDER="C")
        self.assertEqual(c_lines, spilled_lines)
        self.assert_same_bits(c_order, spilled)
        # The spill is exact: the states come back as they left, and the energy with them.
        self.assertEqual(spilled.iterations, named.iterations)
        np.testing.assert_allclose(spilled.eigenvalues, named.eigenvalues, rtol=0, atol=1e-9)
        self.assertAlmostEqual(spilled.energies.total, named.energies.total, places=8)

    def test_the_staging_of_a_spill_moves_no_bit(self):
        # States on the host with the pool of CuPy, and a density step that releases at every size.
        spill = dict(PARSEC_CUPY_SECTOR_STATE_STORAGE="host", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool",
                     PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="0")
        former_values = dict(PARSEC_CUPY_SPILL_BUFFERS="fresh", PARSEC_CUPY_SPILL_PINNED_POOL="release")
        # As the solve of a small sector leaves its vectors, in C order, and with the Ritz step of large
        # sectors named, which rotates a Fortran-ordered basis in place.
        for route in ({}, dict(PARSEC_CUPY_GENERALIZED_RITZ="on")):
            with self.subTest(route=route):
                # The former staging: no buffer, the pinned pool emptied after every density, and the
                # details of a run before the buffers.
                former, *former_lines, _count = self.run_scf(**spill, **route, **former_values)
                self.assertEqual(former_lines[0], "exact FP64 host spill; one active representation on CUDA")
                self.assertNotIn("orbital_sector_spill_staging", self.details)
                self.assertEqual(sorted(self.staging), ["buffers", "offload_seconds", "offloads", "restore_seconds",
                                                        "restores"])
                moves = self.staging["restores"], self.staging["offloads"]
                self.assertEqual((self.staging["buffers"], self.release["spill_pinned_pool"], self.release["pinned_kept"]),
                                 ("fresh", "release", 0))
                uploads = self.release["host_uploads"]
                self.assertEqual((uploads["direct"], uploads["calls"] > 0), (False, True))
                densities = self.release["calls"]
                self.assertEqual(densities, former.iterations)
                # The buffers, page-locked by name, pageable by name and by the rule of the host: the
                # same bits, the same report lines and the same moves.
                for bound in (str(1 << 40), "0", None):
                    named = {} if bound is None else dict(PARSEC_CUPY_SPILL_PINNED_BYTES=bound)
                    run, *lines, _count = self.run_scf(**spill, **route, **named)
                    self.assertEqual(lines, former_lines)
                    self.assert_same_bits(run, former)
                    record = self.staging
                    self.assertEqual((record["buffers"], record["restores"], record["offloads"]), ("persistent", *moves))
                    self.assertEqual((record["uploads"], record["downloads"]), moves)
                    sectors = record["sectors"].values()
                    self.assertTrue(all(item["bytes"] > 0 and item["allocations"] == 1 for item in sectors))
                    held = sum(item["bytes"] for item in sectors)
                    if bound is not None:
                        locked = bound != "0"
                        self.assertEqual((record["bound_bytes"], record["bound_from"]),
                                         (int(bound), "PARSEC_CUPY_SPILL_PINNED_BYTES"))
                        self.assertEqual((record["page_locked_bytes"], record["pageable_bytes"], record["lock_failures"]),
                                         (held if locked else 0, 0 if locked else held, []))
                        self.assertTrue(all(item["page_locked"] is locked for item in sectors))
                    else:
                        self.assertEqual(record["page_locked_bytes"] + record["pageable_bytes"], held)
                    # A vector that is not Fortran-ordered on its device is downloaded by the former call.
                    if route:
                        self.assertLess(record["reordered_downloads"], record["downloads"])
                    else:
                        self.assertEqual(record["reordered_downloads"], record["downloads"])
                    # The density uploads the same columns by direct copies and leaves the pinned pool.
                    self.assertEqual((self.release["spill_pinned_pool"], self.release["pinned_kept"], self.release["calls"]),
                                     ("keep", densities, densities))
                    self.assertEqual({name: self.release["host_uploads"][name] for name in ("direct", "calls", "bytes")},
                                     dict(direct=True, calls=uploads["calls"], bytes=uploads["bytes"]))
                    self.assertEqual(self.details["orbital_sector_spill_staging"], (
                        f"persistent host buffers with direct copies; {sum(item['page_locked'] for item in sectors)} of "
                        f"{len(sectors)} page-locked ({record['page_locked_bytes']} B), {record['pageable_bytes']} B pageable"))
                # Each switch goes back by itself: the buffers with the pinned pool emptied, and the former
                # calls with the pinned pool kept.
                for one in ({"PARSEC_CUPY_SPILL_PINNED_POOL": "release"}, {"PARSEC_CUPY_SPILL_BUFFERS": "fresh"}):
                    run, *lines, _count = self.run_scf(**spill, **route, **one)
                    self.assertEqual(lines, former_lines)
                    self.assert_same_bits(run, former)
                    kept = "PARSEC_CUPY_SPILL_BUFFERS" in one
                    self.assertEqual((self.staging["buffers"], self.release["spill_pinned_pool"], self.release["pinned_kept"]),
                                     ("fresh" if kept else "persistent", "keep" if kept else "release", densities * kept))
        # States that stay on the device have no record of a spill and no such detail.
        self.run_scf(PARSEC_CUPY_SECTOR_STATE_STORAGE="device", PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR="pool")
        self.assertIsNone(self.staging)
        self.assertNotIn("orbital_sector_spill_staging", self.details)
        self.assertEqual(self.release["host_uploads"]["calls"], 0)


if __name__ == "__main__":
    unittest.main()
