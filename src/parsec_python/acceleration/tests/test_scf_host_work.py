"""Host work of an SCF step: the collection before the pool release, that release, wedge sums and the Anderson step."""

from __future__ import annotations

import hashlib
import os
import threading
from time import sleep
import tracemalloc
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np

from parsec_python.Mixer.anderson import safeguard_anderson_candidate
from parsec_python.acceleration.Eigensolvers.distributed_state import (
    DistributedBasis,
    shared_pool_release_requested,
)
from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetryOrbitals
from parsec_python.acceleration.Occupations import symmetry_density
from parsec_python.acceleration.Occupations.symmetry_density import (
    CuPySymmetryDensityBuilder,
    density_collection,
    density_release,
)
from parsec_python.acceleration.SCF import symmetry_fields
from parsec_python.acceleration.SCF.symmetry_fields import (
    SymmetrySCFReducer,
    scf_buffers_requested,
)
from parsec_python.models import MixingSettings


def _reducer(size: int, seed: int = 5) -> SymmetrySCFReducer:
    """A reducer on ``size`` orbits with every multiplicity of a group of order eight."""

    rng = np.random.default_rng(seed)
    multiplicities = rng.choice(np.array([1, 2, 4, 8]), size=size)
    return SymmetrySCFReducer(
        SimpleNamespace(
            multiplicities=multiplicities,
            full_to_wedge=np.repeat(np.arange(size), multiplicities),
            wedge_size=size,
            full_size=int(multiplicities.sum()),
        )
    )


def _bits(values) -> str:
    """A digest of the float64 bit patterns.

    Not the bytes themselves: unittest diffs the text of two sequences that
    differ, which for a list of whole fields does not end.
    """

    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.float64).tobytes()
    ).hexdigest()


class _RecordedDots:
    """NumPy with every ``dot`` kept: its operands stay alive, so each is one object."""

    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        return getattr(np, name)

    def dot(self, left, right):
        self.calls.append((left, right))
        return np.dot(left, right)


class BitPatternTests(unittest.TestCase):
    def test_fields_are_compared_by_a_short_digest_of_their_bits(self) -> None:
        zeros = np.zeros(50_000)
        negative = zeros.copy()
        negative[-1] = -0.0
        self.assertNotEqual(_bits(zeros), _bits(negative))
        self.assertLessEqual(len(_bits(zeros)), 64)
        # Twelve fields that differ in one bit are reported at once.
        with self.assertRaises(AssertionError):
            self.assertEqual([_bits(zeros)] * 12, [_bits(negative)] * 12)


class ScfBufferSwitchTests(unittest.TestCase):
    def test_switch_is_on_unless_named_off(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_SYMMETRY_SCF_BUFFERS", None)
            self.assertTrue(scf_buffers_requested())
            for value in ("0", "false", "No", " off "):
                os.environ["PARSEC_SYMMETRY_SCF_BUFFERS"] = value
                self.assertFalse(scf_buffers_requested())
            os.environ["PARSEC_SYMMETRY_SCF_BUFFERS"] = "1"
            self.assertTrue(scf_buffers_requested())

    def test_weights_are_the_multiplicities_in_float64(self) -> None:
        reducer = _reducer(100)
        multiplicities = reducer.reduction.multiplicities
        weights = reducer.weights
        self.assertEqual(weights.dtype, np.float64)
        self.assertEqual(multiplicities.dtype.kind, "i")
        np.testing.assert_array_equal(weights, multiplicities)
        self.assertIs(reducer.weights, weights)
        self.assertFalse(weights.flags.writeable)
        self.assertTrue(multiplicities.flags.writeable)
        # The product with a field is the one the integers give.
        field = np.random.default_rng(1).standard_normal(100)
        self.assertEqual(_bits(weights * field), _bits(multiplicities * field))
        self.assertEqual(
            np.dot(weights, field * field), np.dot(multiplicities, field * field)
        )


class WedgeSumTests(unittest.TestCase):
    size = 50_000

    def fields(self, reducer, count, seed=11):
        rng = np.random.default_rng(seed)
        return [reducer.field(rng.standard_normal(self.size)) for _ in range(count)]

    def evaluate(self, reducer):
        density, vin, vion, vh, vxc, vout = self.fields(reducer, 6)
        density = reducer.field(np.abs(density.values))
        rng = np.random.default_rng(2)
        eigenvalues = rng.standard_normal(40)
        occupations = rng.random(40)
        energy = reducer.total_energy(
            eigenvalues, occupations, density, vin, vion, vh, vxc, -3.25, 7.5, 0.054
        )
        metrics = reducer.potential_residual_metrics(vin, vout, density, 0.054, 96.0)
        return (
            energy,
            metrics.weighted,
            metrics.plain,
            _bits(metrics.wedge_residual),
            reducer.weighted_dot(vin, vout),
        )

    def test_energies_and_norms_are_bitwise_those_of_the_integer_multiplicities(self) -> None:
        reducer = _reducer(self.size)
        with patch.dict(os.environ, PARSEC_SYMMETRY_SCF_BUFFERS="1"):
            buffered = self.evaluate(reducer)
        with patch.dict(os.environ, PARSEC_SYMMETRY_SCF_BUFFERS="0"):
            former = self.evaluate(reducer)
        self.assertEqual(buffered, former)
        # Written out once more, as the integrals stood before the switch.
        density, vin, vion, vh, vxc, _ = self.fields(reducer, 6)
        rho = np.abs(density.values)
        multiplicities = reducer.reduction.multiplicities
        energy = buffered[0]
        self.assertEqual(
            energy.hartree, 0.5 * float(0.054 * np.dot(multiplicities * rho, vh.values))
        )
        self.assertEqual(
            energy.integral_vxc_rho,
            float(0.054 * np.dot(multiplicities * rho, vxc.values)),
        )
        self.assertEqual(
            energy.electron_ion,
            float(0.054 * np.dot(multiplicities * rho, vion.values)),
        )

    def test_one_weighted_density_serves_the_four_integrals(self) -> None:
        reducer = _reducer(2_000)
        density, vin, vion, vh, vxc = self.fields_of(reducer)

        def left_operands(setting):
            recorder = _RecordedDots()
            with patch.dict(os.environ, PARSEC_SYMMETRY_SCF_BUFFERS=setting), \
                 patch.object(symmetry_fields, "np", recorder):
                reducer.total_energy(
                    np.zeros(3), np.ones(3), density, vin, vion, vh, vxc, 0.0, 0.0, 1.0
                )
            # The band energy is the first product; the integrals follow.
            return [left for left, _right in recorder.calls[1:]]

        shared = left_operands("1")
        self.assertEqual(len(shared), 4)
        self.assertTrue(all(left is shared[0] for left in shared))
        apart = left_operands("0")
        self.assertEqual(len({id(left) for left in apart}), 4)
        for left in apart:
            self.assertEqual(_bits(left), _bits(shared[0]))

    def fields_of(self, reducer):
        rng = np.random.default_rng(3)
        size = reducer.reduction.wedge_size
        return [reducer.field(rng.random(size)) for _ in range(5)]


class AndersonStepTests(unittest.TestCase):
    # More than three blocks of the buffered step, the last one short.
    size = 3 * symmetry_fields._BLOCK_ROWS + 1_234

    def potentials(self, reducer, steps, seed=23):
        """Inputs and outputs whose residual shrinks, grows at once and shrinks again."""

        rng = np.random.default_rng(seed)
        target = rng.standard_normal(self.size)
        scale = [1.0, 0.4, 0.2, 0.1, 5.0, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
        return [
            (
                reducer.field(target + scale[step] * rng.standard_normal(self.size)),
                reducer.field(target + 0.5 * scale[step] * rng.standard_normal(self.size)),
            )
            for step in range(steps)
        ]

    def run_mixer(self, reducer, settings, setting, steps=12):
        """After every step: the mixed potential, the residual norm the safeguard
        kept, its resets and clips, and the length of the history."""

        with patch.dict(os.environ, PARSEC_SYMMETRY_SCF_BUFFERS=setting):
            mixer = reducer.mixer(settings)
            trace = []
            for step, (vin, vout) in enumerate(self.potentials(reducer, steps), 1):
                mixed = mixer.mix(vin, vout, iteration=step)
                trace.append((
                    _bits(mixed.values),
                    mixer._previous_residual_norm,
                    mixer.safeguard_resets,
                    mixer.safeguard_clips,
                    len(mixer._residuals),
                ))
        return trace, mixer

    def test_mixed_potentials_are_bitwise_those_of_the_unbuffered_step(self) -> None:
        reducer = _reducer(self.size)
        for settings in (
            # The safeguard resets the history at the fifth step and clips others;
            # the restart clears it at the eighth.
            MixingSettings(parameter=0.3, memory=4, restart=7, safeguard=True,
                           step_limit=1.5, growth_trigger=2.0, backoff=0.5),
            MixingSettings(parameter=0.41, memory=3, restart=20, regularization=1e-10),
            MixingSettings(parameter=0.3, memory=1, restart=20, safeguard=True),
            # Every other step is a plain one after a restart: its safeguard forms
            # the weighted residual in the array the step before left its own in.
            MixingSettings(parameter=0.3, memory=4, restart=2, safeguard=True,
                           step_limit=1.5),
        ):
            with self.subTest(settings=settings):
                buffered, mixer = self.run_mixer(reducer, settings, "1")
                former, reference = self.run_mixer(reducer, settings, "0")
                for step, (new, old) in enumerate(zip(buffered, former, strict=True), 1):
                    self.assertEqual(new, old, step)
                for kept, expected in zip(mixer._inputs + mixer._residuals,
                                          reference._inputs + reference._residuals,
                                          strict=True):
                    self.assertEqual(_bits(kept), _bits(expected))
                if settings.safeguard and settings.restart == 7:
                    self.assertGreaterEqual(mixer.safeguard_resets, 1)
                    self.assertGreaterEqual(mixer.safeguard_clips, 1)
                    # The eighth step starts from an empty history.
                    self.assertEqual([entry[4] for entry in buffered[6:9]], [3, 1, 2])
                if settings.restart == 2:
                    self.assertEqual({entry[4] for entry in buffered[::2]}, {1})

    def test_full_grid_potentials_take_the_same_step(self) -> None:
        reducer = _reducer(700)
        rng = np.random.default_rng(4)
        full = reducer.reduction.full_to_wedge
        pairs = [(rng.standard_normal(700)[full], rng.standard_normal(700)[full])
                 for _ in range(4)]

        def mixed(setting):
            with patch.dict(os.environ, PARSEC_SYMMETRY_SCF_BUFFERS=setting):
                mixer = reducer.mixer(MixingSettings(memory=2, safeguard=True))
                return [_bits(mixer.mix(vin, vout)) for vin, vout in pairs]

        self.assertEqual(mixed("1"), mixed("0"))

    def test_safeguard_in_the_mixer_arrays_is_the_reference_function(self) -> None:
        reducer = _reducer(5_000)
        rng = np.random.default_rng(9)
        vin = rng.standard_normal(5_000)
        residual = 0.1 * rng.standard_normal(5_000)
        settings = MixingSettings(parameter=0.3, memory=4, safeguard=True, step_limit=2.0)
        cases = {
            # Previous residual norm and candidate: kept, clipped, and replaced after a
            # growth of the residual, whatever the candidate was.
            "kept": (None, vin + 0.3 * residual),
            "clipped": (None, vin + 40.0 * rng.standard_normal(5_000)),
            "reset": (1e-6, vin + 0.3 * residual),
            "reset from a far candidate": (1e-6, vin + 40.0 * rng.standard_normal(5_000)),
        }
        seen = set()
        for name, (previous, candidate) in cases.items():
            with self.subTest(name):
                expected = safeguard_anderson_candidate(
                    vin, residual, candidate.copy(), settings,
                    previous_residual_norm=previous,
                    weights=reducer.reduction.multiplicities,
                )
                mixer = reducer.mixer(settings)
                mixer._previous_residual_norm = previous
                own = candidate.copy()
                actual = mixer._safeguarded(vin, residual, own, None)
                self.assertIs(actual[0], own)
                self.assertEqual(_bits(actual[0]), _bits(expected[0]))
                self.assertEqual(actual[1:], expected[1:])
                seen.add(expected[2:])
        self.assertEqual(seen, {(False, False), (False, True), (True, False)})

    def test_dense_products_receive_arrays_of_the_former_layout(self) -> None:
        reducer = _reducer(4_000)
        rng = np.random.default_rng(6)
        mixer = reducer.mixer(MixingSettings(memory=3))
        residual = rng.standard_normal(4_000)
        history = [rng.standard_normal(4_000) for _ in range(3)]
        differences, weighted, weighted_residual = mixer._weighted_differences(
            residual, history
        )
        expected = np.column_stack([residual - previous for previous in history])
        multiplicities = reducer.reduction.multiplicities
        for actual, former in (
            (differences, expected),
            (weighted, multiplicities[:, None] * expected),
            (weighted_residual, multiplicities * residual),
        ):
            self.assertEqual(actual.shape, former.shape)
            self.assertEqual(actual.strides, former.strides)
            self.assertTrue(actual.flags.c_contiguous and actual.flags.owndata)
            self.assertEqual(_bits(actual), _bits(former))

    def test_buffered_step_keeps_its_arrays_and_allocates_one_field(self) -> None:
        reducer = _reducer(self.size)
        field_bytes = 8 * self.size
        # No reset of the history: every step from the fourth on is the steady one.
        settings = MixingSettings(parameter=0.3, memory=2, restart=50, safeguard=True,
                                  growth_trigger=1e6)
        pairs = self.potentials(reducer, 12)

        def traced(setting):
            with patch.dict(os.environ, PARSEC_SYMMETRY_SCF_BUFFERS=setting):
                mixer = reducer.mixer(settings)
                history, peaks = [], []
                for step, (vin, vout) in enumerate(pairs, 1):
                    tracemalloc.start()
                    try:
                        before = tracemalloc.get_traced_memory()[0]
                        tracemalloc.reset_peak()
                        mixed = mixer.mix(vin, vout, iteration=step)
                        peaks.append(tracemalloc.get_traced_memory()[1] - before)
                    finally:
                        tracemalloc.stop()
                    del mixed
                    # Held here, so that no later array can take the place of one.
                    history.extend(mixer._inputs + mixer._residuals)
                return peaks, len({id(array) for array in history}), mixer

        peaks, arrays, mixer = traced("1")
        # From the step after the history is full: the mixed potential, and blocks.
        self.assertLess(max(peaks[4:]), 1.25 * field_bytes)
        # Two arrays per history entry and the two that wait for the next step.
        self.assertEqual(arrays, 2 * settings.memory + 2)
        self.assertEqual(
            {name: array.shape for name, array in mixer._buffers.items()},
            {
                "differences": (self.size, 2),
                "weighted_differences": (self.size, 2),
                "weighted": (self.size,),
                "step": (self.size,),
                "term": (symmetry_fields._BLOCK_ROWS,),
                "average_residual": (symmetry_fields._BLOCK_ROWS,),
            },
        )
        # What the root keeps between steps beside the history: two fields per
        # history entry and four more in the mixer, and the float64
        # multiplicities in the reducer.
        resident = sum(array.nbytes for array in mixer._buffers.values()) + sum(
            array.nbytes for array in mixer._spare
        )
        self.assertEqual(
            resident,
            (2 * settings.memory + 4) * field_bytes + 2 * 8 * symmetry_fields._BLOCK_ROWS,
        )
        self.assertEqual(reducer.weights.nbytes, field_bytes)
        former_peaks, former_arrays, former = traced("0")
        self.assertGreater(min(former_peaks[4:]), 4 * field_bytes)
        self.assertEqual(former_arrays, 2 * len(pairs))
        self.assertEqual((former._buffers, former._spare), ({}, []))
        mixer.reset()
        self.assertEqual((mixer._buffers, mixer._spare, mixer._inputs), ({}, [], []))


class _FakePool:
    def __init__(self, cuda) -> None:
        self.cuda = cuda
        self.used = {}
        self.read = []
        self.released = []

    def malloc(self, size):
        raise AssertionError("not called")

    def used_bytes(self) -> int:
        self.read.append(self.cuda.current)
        return self.used.get(self.cuda.current, 0)

    def free_all_blocks(self) -> None:
        self.released.append(self.cuda.current)


class _FakeCuda:
    def __init__(self) -> None:
        # The current device belongs to the thread, as CuPy's does: the pools of several devices are
        # emptied by a thread each.
        self._place = threading.local()
        self.allocator = None
        outer = self

        class Device:
            def __init__(self, index):
                self.id = index

            def __enter__(self):
                self.previous, outer.current = outer.current, self.id
                return self

            def __exit__(self, *args):
                outer.current = self.previous
                return False

        self.Device = Device

    @property
    def current(self):
        return getattr(self._place, "device", None)

    @current.setter
    def current(self, device) -> None:
        self._place.device = device

    def get_allocator(self):
        return self.allocator


class _FakeArray:
    def __init__(self, cuda, device, nbytes) -> None:
        self.device = cuda.Device(device)
        self.nbytes = nbytes


class DensityCollectionTests(unittest.TestCase):
    """The collection before the pool release, on stand-ins for the pools of two devices."""

    def setUp(self) -> None:
        self.cuda = _FakeCuda()
        self.pool = _FakePool(self.cuda)
        self.pinned = _FakePool(self.cuda)
        self.cuda.allocator = self.pool.malloc
        self.pool.used = {0: 1000, 1: 2000}
        self.cp = SimpleNamespace(
            cuda=self.cuda,
            ndarray=_FakeArray,
            get_default_memory_pool=lambda: self.pool,
            get_default_pinned_memory_pool=lambda: self.pinned,
        )
        self.collections = []
        # What a collection of all generations (None) or of the young ones (1) does.
        self.on_collection = lambda generation: 0
        self.orbitals = SimpleNamespace(
            sector_vectors=(
                _FakeArray(self.cuda, 1, 400),
                _FakeArray(self.cuda, 0, 400),
            )
        )

    def collect(self, generation=None):
        self.collections.append(generation)
        return self.on_collection(generation)

    def release(self, builder, times=1):
        with patch.dict(os.environ, PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="0"), \
             patch("parsec_python.acceleration.backends.cupy.require_cupy",
                   return_value=(self.cp, None)), \
             patch.object(symmetry_density.gc, "collect", self.collect):
            for _ in range(times):
                builder._release_large_unused_pool(self.orbitals)

    def builder(self, collection=None):
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_CUPY_DENSITY_COLLECTION", None)
            if collection is not None:
                os.environ["PARSEC_CUPY_DENSITY_COLLECTION"] = collection
            return CuPySymmetryDensityBuilder(lambda *args: None)

    def test_switch_names_two_collections_and_refuses_others(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_CUPY_DENSITY_COLLECTION", None)
            self.assertEqual(density_collection(), "changed")
            os.environ["PARSEC_CUPY_DENSITY_COLLECTION"] = " Full "
            self.assertEqual(density_collection(), "full")
            os.environ["PARSEC_CUPY_DENSITY_COLLECTION"] = "young"
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_DENSITY_COLLECTION"):
                density_collection()
            # A builder refuses it before any density is built.
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_DENSITY_COLLECTION"):
                CuPySymmetryDensityBuilder(lambda *args: None)

    def test_two_full_collections_then_young_ones_while_the_bytes_in_use_stay(self) -> None:
        builder = self.builder()
        self.assertEqual(builder.collection, "changed")
        self.release(builder, times=6)
        self.assertEqual(self.collections, [None, None, 1, 1, 1, 1])
        record = builder.pool_release
        self.assertEqual(
            (record["calls"], record["full_collections"], record["young_collections"]),
            (6, 2, 4),
        )
        # Every call releases the pool of each device and the pinned pool, and has done so when it returns.
        self.assertEqual([sorted(self.pool.released[call:call + 2]) for call in range(0, 12, 2)], [[0, 1]] * 6)
        self.assertEqual(self.pinned.released, [None] * 6)
        self.assertGreaterEqual(record["collection_seconds"], 0.0)
        self.assertGreaterEqual(record["release_seconds"], 0.0)

    def test_other_bytes_in_use_bring_the_full_collection_back(self) -> None:
        builder = self.builder()
        self.release(builder, times=3)
        self.assertEqual(self.collections, [None, None, 1])
        # A cycle of older objects now holds 512 bytes on device 1: the young
        # collection leaves them, the full one returns them.
        self.pool.used[1] = 2512

        def collection(generation):
            if generation is None:
                self.pool.used[1] = 2000
                return 7
            return 0

        self.on_collection = collection
        self.release(builder)
        self.assertEqual(self.collections[3:], [1, None])
        self.assertEqual(self.pool.used[1], 2000)
        self.assertEqual(builder.pool_release["unreachable_objects"], 7)
        self.release(builder)
        self.assertEqual(self.collections[5:], [1])
        # Fewer bytes in use are a change as well, and the new level is the one kept.
        self.pool.used[0] = 600
        self.release(builder, times=2)
        self.assertEqual(self.collections[6:], [1, None, 1])
        # So are other devices.
        self.orbitals.sector_vectors = self.orbitals.sector_vectors[:1]
        self.release(builder, times=2)
        self.assertEqual(self.collections[9:], [1, None, 1])
        self.assertEqual(self.pool.released[-2:], [1, 1])

    def test_young_garbage_is_collected_before_the_bytes_are_compared(self) -> None:
        builder = self.builder()
        self.release(builder, times=2)
        self.pool.used[0] = 1300

        def collection(generation):
            # A cycle made in this step: any collection returns its array.
            self.pool.used[0] = 1000
            return 3

        self.on_collection = collection
        self.release(builder, times=2)
        self.assertEqual(self.collections[2:], [1, 1])
        self.assertEqual(builder.pool_release["unreachable_objects"], 6)

    def test_full_names_the_former_collection_every_time(self) -> None:
        builder = self.builder("full")
        self.pool.used[0] = 5
        self.release(builder, times=4)
        self.assertEqual(self.collections, [None] * 4)
        # It never reads the pools.
        self.assertEqual(self.pool.read, [])
        self.assertEqual([sorted(self.pool.released[call:call + 2]) for call in range(0, 8, 2)], [[0, 1]] * 4)
        self.assertEqual(builder.pool_release["full_collections"], 4)
        self.assertEqual(builder.pool_release["collection"], "full")

    def test_another_allocator_is_collected_in_full(self) -> None:
        builder = self.builder()
        self.cuda.allocator = None
        self.release(builder, times=4)
        self.assertEqual(self.collections, [None] * 4)
        self.assertEqual(self.pool.read, [])
        # Back on the pool, the level is not known yet.
        self.cuda.allocator = self.pool.malloc
        self.release(builder, times=2)
        self.assertEqual(self.collections[4:], [1, None, 1])

    def test_host_kept_sector_states_are_collected_in_full(self) -> None:
        # Sector states kept on the host name no device: no pool is released
        # or read, so nothing would show a cycle that holds device memory.
        builder = self.builder()
        self.orbitals.sector_vectors = (np.zeros((40, 5)), np.zeros((40, 5)))
        for _ in range(5):
            self.pool.used[0] += 512
            self.release(builder)
        self.assertEqual(self.collections, [None] * 5)
        self.assertEqual((self.pool.read, self.pool.released), ([], []))
        self.assertEqual(self.pinned.released, [None] * 5)
        record = builder.pool_release
        self.assertEqual(
            (record["calls"], record["full_collections"], record["young_collections"]),
            (5, 5, 0),
        )
        # Back on a device, the level is not known yet.
        self.orbitals.sector_vectors = (_FakeArray(self.cuda, 0, 400),)
        self.release(builder, times=2)
        self.assertEqual(self.collections[5:], [1, None, 1])

    def test_small_bases_release_nothing(self) -> None:
        builder = self.builder()
        with patch.dict(os.environ, PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="801"), \
             patch("parsec_python.acceleration.backends.cupy.require_cupy",
                   return_value=(self.cp, None)), \
             patch.object(symmetry_density.gc, "collect", self.collect):
            builder._release_large_unused_pool(self.orbitals)
        self.assertEqual((self.collections, self.pool.released), ([], []))
        self.assertEqual(builder.pool_release["calls"], 0)

    def test_sector_density_seconds_cover_the_lazy_route(self) -> None:
        def columns(vectors, weights, volume):
            return (2.0 / volume) * np.sum(np.asarray(vectors) ** 2 * weights[None, :], axis=1)

        builder = CuPySymmetryDensityBuilder(columns)
        orbitals = CuPySymmetryOrbitals(
            scaled_wedge_vectors=None,
            representations=np.zeros(2, dtype=np.int32),
            full_to_wedge=np.arange(3),
            device_full_to_wedge=None,
            phases=np.ones((1, 3)),
            full_size=3,
            representation_columns=np.arange(2, dtype=np.int32),
            sector_vectors=(np.eye(3, 2),),
            sector_orbits=(np.arange(3),),
            sector_scales=(np.ones(3),),
            wedge_size=3,
        )
        density = builder(orbitals, np.array([1.0, 0.5]), 0.5)
        np.testing.assert_array_equal(density, [4.0, 2.0, 0.0])
        self.assertGreater(builder.pool_release["sector_density_seconds"], 0.0)


class _RecordingPool(_FakePool):
    """Says which thread released, after ``before(device)`` has returned in that thread."""

    def __init__(self, cuda, before=None) -> None:
        super().__init__(cuda)
        self.before = before
        self.threads = []

    def free_all_blocks(self) -> None:
        if self.before is not None:
            self.before(self.cuda.current)
        self.threads.append(threading.current_thread().name)
        super().free_all_blocks()


class DensityReleaseTests(unittest.TestCase):
    """The release of the pools after a density (PARSEC_CUPY_DENSITY_RELEASE), on stand-ins for two devices."""

    def setUp(self) -> None:
        self.cuda = _FakeCuda()
        self.pool = _RecordingPool(self.cuda)
        self.pinned = _RecordingPool(self.cuda)
        self.cuda.allocator = self.pool.malloc
        self.cp = SimpleNamespace(
            cuda=self.cuda,
            ndarray=_FakeArray,
            get_default_memory_pool=lambda: self.pool,
            get_default_pinned_memory_pool=lambda: self.pinned,
        )
        # Per collection: the device pools that had been emptied by then.
        self.emptied = []
        self.orbitals = SimpleNamespace(
            sector_vectors=(_FakeArray(self.cuda, 1, 400), _FakeArray(self.cuda, 0, 400))
        )
        self.here = threading.current_thread().name
        # The threads of the devices: those of a filter that is spread over devices.
        self.device_threads = ["orbital-shard-0_0", "orbital-shard-1_0"]

    def collect(self, generation=None):
        self.emptied.append(len(self.pool.released))
        return 0

    def release(self, builder, times=1, orbitals=None):
        with patch.dict(os.environ, PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="0"), \
             patch("parsec_python.acceleration.backends.cupy.require_cupy",
                   return_value=(self.cp, None)), \
             patch.object(symmetry_density.gc, "collect", self.collect):
            for _ in range(times):
                builder._release_large_unused_pool(self.orbitals if orbitals is None else orbitals)

    def builder(self, release=None, shared=None):
        with patch.dict(os.environ):
            for name, value in (("PARSEC_CUPY_DENSITY_RELEASE", release),
                                ("PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE", shared)):
                os.environ.pop(name, None)
                if value is not None:
                    os.environ[name] = value
            return CuPySymmetryDensityBuilder(lambda *args: None)

    @staticmethod
    def shared(*devices, nbytes=400):
        """Stands for the basis of a sector that the devices of a group share: a block of ``nbytes`` on each."""

        basis = object.__new__(DistributedBasis)
        basis.group = SimpleNamespace(devices=devices)
        basis.blocks = tuple(SimpleNamespace(nbytes=nbytes) for _ in devices)
        return basis

    def test_a_shared_basis_keeps_what_its_pools_hold_unless_the_release_is_named(self) -> None:
        name = "PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE"
        with patch.dict(os.environ):
            os.environ.pop(name, None)
            self.assertFalse(shared_pool_release_requested())
            for raw, expected in (("1", True), (" On ", True), ("true", True), ("0", False), ("off", False)):
                os.environ[name] = raw
                self.assertEqual(shared_pool_release_requested(), expected)
            for unknown in ("keep", "2", ""):
                os.environ[name] = unknown
                with self.assertRaisesRegex(ValueError, name):
                    shared_pool_release_requested()
                # A builder refuses it before any density is built.
                with self.assertRaisesRegex(ValueError, name):
                    CuPySymmetryDensityBuilder(lambda *args: None)
        # Two sectors on two devices each, as a rank of a run on 8 GPUs solves them, and one on four (16 GPUs).
        # The sectors that the other ranks solve have no vectors here.  And a process that holds them all.
        pairs = SimpleNamespace(sector_vectors=(self.shared(0, 1), None, self.shared(2, 3), None))
        four = SimpleNamespace(sector_vectors=(None, self.shared(0, 1, 2, 3), None, None))
        whole = SimpleNamespace(sector_vectors=(self.shared(0, 1), self.shared(2, 3)))
        for orbitals in (pairs, four, whole):
            for release in (None, "serial"):
                for joins in (False, True):
                    builder = self.builder(release)
                    builder.caller_joins_release = joins
                    self.assertEqual((builder.shared_release, builder.pool_release["shared"]), (False, "keep"))
                    self.release(builder, times=3, orbitals=orbitals)
                    # Nothing is collected and no pool is read or emptied, the pinned pool neither; no thread
                    # was given anything, so nothing is left to wait for.
                    self.assertEqual((self.emptied, self.pool.read, self.pool.released, self.pinned.released),
                                     ([], [], [], []))
                    self.assertEqual(builder._releasing, ())
                    record = dict(builder.pool_release)
                    self.assertEqual((record["shared_kept"], record["calls"], record["full_collections"],
                                      record["young_collections"]), (3, 0, 0, 0))
                    self.assertEqual((record["collection_seconds"], record["release_seconds"],
                                      record["device_release_seconds"]), (0.0, 0.0, 0.0))
                    builder.release_joined()
                    self.assertEqual(builder.pool_release, record)
        # Named, the release is what it was: every device of every group by its thread, the pinned pool
        # meanwhile, after a collection; or one device after another in the thread that built the density.
        threads = [f"orbital-shard-{device}_0" for device in range(4)]
        for orbitals in (pairs, four):
            for release, expected in ((None, threads), ("serial", [self.here] * 4)):
                builder = self.builder(release, shared="1")
                self.assertEqual((builder.shared_release, builder.pool_release["shared"]), (True, "release"))
                del self.emptied[:], self.pool.released[:], self.pool.threads[:], self.pinned.released[:]
                self.release(builder, times=2, orbitals=orbitals)
                self.assertEqual(sorted(zip(self.pool.released, self.pool.threads)),
                                 sorted(list(zip((0, 1, 2, 3), expected)) * 2))
                self.assertEqual((self.pinned.released, len(self.emptied)), ([None] * 2, 2))
                record = builder.pool_release
                self.assertEqual((record["shared_kept"], record["calls"], record["full_collections"]), (0, 2, 2))
        # A process that also holds a sector as one array of a device empties the pool of that device as
        # before, here in its own thread, and collects for it; the devices of the shared sector keep theirs.
        del self.emptied[:], self.pool.released[:], self.pool.threads[:], self.pinned.released[:]
        mixed = SimpleNamespace(sector_vectors=(self.shared(0, 1), None, _FakeArray(self.cuda, 2, 400)))
        builder = self.builder()
        self.release(builder, times=2, orbitals=mixed)
        self.assertEqual((self.pool.released, self.pool.threads), ([2, 2], [self.here] * 2))
        self.assertEqual((self.pinned.released, self.emptied), ([None] * 2, [0, 1]))
        record = builder.pool_release
        self.assertEqual((record["shared_kept"], record["calls"]), (2, 2))
        # So it does for a device that holds both an array of one sector and a block of a shared one.
        del self.pool.released[:]
        both = SimpleNamespace(sector_vectors=(self.shared(0, 1), _FakeArray(self.cuda, 1, 400)))
        self.release(self.builder(), orbitals=both)
        self.assertEqual(self.pool.released, [1])
        # Sector states that a process keeps on the host beside a shared basis are collected for in full and
        # empty the pinned pool, as they do alone.
        del self.emptied[:], self.pool.released[:], self.pinned.released[:]
        builder = self.builder()
        self.release(builder, orbitals=SimpleNamespace(sector_vectors=(self.shared(0, 1), None, np.zeros((40, 5)))))
        self.assertEqual((self.pool.released, self.pinned.released, len(self.emptied)), ([], [None], 1))
        self.assertEqual((builder.pool_release["shared_kept"], builder.pool_release["full_collections"]), (1, 1))
        # Below the limit of the release nothing is counted either: the rule is asked of what would be released.
        builder = self.builder()
        with patch.dict(os.environ, PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="1601"), \
             patch("parsec_python.acceleration.backends.cupy.require_cupy", return_value=(self.cp, None)), \
             patch.object(symmetry_density.gc, "collect", self.collect):
            builder._release_large_unused_pool(pairs)
            self.assertEqual(builder.pool_release["shared_kept"], 0)
            os.environ["PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES"] = "1600"
            builder._release_large_unused_pool(pairs)
            self.assertEqual(builder.pool_release["shared_kept"], 1)

    def test_a_density_of_shared_sectors_is_back_with_nothing_to_wait_for(self) -> None:
        # Two shared sectors through the builder itself, as an MPI rank calls it: it leaves the wait to its
        # caller, and there is none.
        def sector(*devices):
            basis = self.shared(*devices)
            basis.density = lambda builder, occupations, volume: np.full(3, float(len(occupations)))
            return basis

        orbitals = CuPySymmetryOrbitals(
            scaled_wedge_vectors=None,
            representations=np.array([0, 0, 2, 2, 2], dtype=np.int32),
            full_to_wedge=np.arange(3),
            device_full_to_wedge=None,
            phases=np.ones((4, 3)),
            full_size=3,
            representation_columns=np.array([0, 1, 0, 1, 2], dtype=np.int32),
            # The second sector of this rank is the third of the calculation: another rank solves the one
            # between them and the last.
            sector_vectors=(sector(0, 1), None, sector(2, 3), None),
            sector_orbits=(np.arange(3), None, np.arange(3), None),
            sector_scales=(np.ones(3), None, np.ones(3), None),
            wedge_size=3,
        )
        for shared, released in ((None, []), ("1", [0, 1, 2, 3])):
            builder = self.builder(shared=shared)
            builder.caller_joins_release = True
            del self.pool.released[:]
            with patch.dict(os.environ, PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="0"), \
                 patch("parsec_python.acceleration.backends.cupy.require_cupy", return_value=(self.cp, None)), \
                 patch.object(symmetry_density.gc, "collect", self.collect):
                values = builder(orbitals, np.ones(5), 0.5)
            np.testing.assert_array_equal(values, np.full(3, 5.0))
            # Kept: no thread was handed a pool, and the builder holds nothing back for a wait.
            self.assertEqual((len(builder._releasing), builder._held is None), (len(released), not released))
            builder.release_joined()
            self.assertEqual((sorted(self.pool.released), builder._releasing, builder._held), (released, (), None))
            self.assertGreater(builder.pool_release["sector_density_seconds"], 0.0)

    def test_switch_names_two_releases_and_refuses_others(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_CUPY_DENSITY_RELEASE", None)
            self.assertEqual(density_release(), "threads")
            os.environ["PARSEC_CUPY_DENSITY_RELEASE"] = " Serial "
            self.assertEqual(density_release(), "serial")
            for unknown in ("0", "off", "thread", ""):
                os.environ["PARSEC_CUPY_DENSITY_RELEASE"] = unknown
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_DENSITY_RELEASE"):
                    density_release()
                # A builder refuses it before any density is built.
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_DENSITY_RELEASE"):
                    CuPySymmetryDensityBuilder(lambda *args: None)

    def test_every_device_is_emptied_by_its_thread_beside_the_others_and_the_pinned_pool(self) -> None:
        builder = self.builder()
        self.assertEqual((builder.release, builder.pool_release["release"]), ("threads", "threads"))
        together, pinned_emptied = threading.Barrier(2), threading.Event()

        def device(index):
            # Passed only while the other device is being emptied too, and the pinned pool has been.
            together.wait(timeout=20)
            self.assertTrue(pinned_emptied.wait(timeout=20))

        self.pool.before = device
        self.pinned.before = lambda index: pinned_emptied.set()
        self.release(builder)
        # Both pools are empty when the builder goes on, each by the thread of its device.
        self.assertEqual(sorted(zip(self.pool.released, self.pool.threads)), list(zip((0, 1), self.device_threads)))
        self.assertEqual((self.pinned.released, self.pinned.threads), ([None], [self.here]))
        self.assertEqual(builder._releasing, ())
        record = builder.pool_release
        self.assertEqual((record["calls"], self.emptied), (1, [0]))
        # What the device threads took, and what the thread that built the density waited.
        self.assertGreater(record["device_release_seconds"], 0.0)
        self.assertGreater(record["release_seconds"], 0.0)
        # Nothing is left to wait for.
        before = dict(record)
        builder.release_joined()
        self.assertEqual(record, before)

    def test_serial_names_the_former_release_in_the_thread_that_built_the_density(self) -> None:
        builder = self.builder("serial")
        self.assertEqual((builder.release, builder.pool_release["release"]), ("serial", "serial"))
        order = []
        self.pool.before = lambda index: order.append(("device", index))
        self.pinned.before = lambda index: order.append(("pinned", index))
        self.release(builder, times=3)
        # One device after another in their order, then the pinned pool, all in this thread.
        self.assertEqual(order, [("device", 0), ("device", 1), ("pinned", None)] * 3)
        self.assertEqual(set(self.pool.threads + self.pinned.threads), {self.here})
        self.assertEqual((builder.pool_release["device_release_seconds"], builder._releasing), (0.0, ()))
        # A caller that would wait for device threads has nothing to wait for.
        builder.caller_joins_release = True
        self.release(builder)
        self.assertEqual((order[9:], builder._releasing), ([("device", 0), ("device", 1), ("pinned", None)], ()))
        builder.release_joined()

    def test_one_device_is_emptied_by_the_thread_that_built_the_density_either_way(self) -> None:
        alone = SimpleNamespace(sector_vectors=(_FakeArray(self.cuda, 1, 400), _FakeArray(self.cuda, 1, 400)))
        for release in (None, "serial"):
            builder = self.builder(release)
            self.release(builder, orbitals=alone)
            self.assertEqual(builder.pool_release["device_release_seconds"], 0.0)
        self.assertEqual((self.pool.released, self.pool.threads), ([1, 1], [self.here] * 2))
        self.assertEqual((self.pinned.released, self.pinned.threads), ([None] * 2, [self.here] * 2))
        # So are host-kept sector states, which name no device: the pinned pool alone.
        self.release(self.builder(), orbitals=SimpleNamespace(sector_vectors=(np.zeros((40, 5)),)))
        self.assertEqual((self.pool.released, self.pinned.threads), ([1, 1], [self.here] * 3))

    def test_a_caller_that_waits_itself_has_the_density_first_and_empty_pools_when_it_has_waited(self) -> None:
        builder = self.builder()
        builder.caller_joins_release = True
        go = threading.Event()
        self.pool.before = lambda index: self.assertTrue(go.wait(timeout=20))
        self.release(builder)
        # The builder is back while no device has been emptied; the pinned pool has.
        self.assertEqual((self.pool.released, self.pinned.released, len(builder._releasing)), ([], [None], 2))
        record = builder.pool_release
        spent = record["release_seconds"], record["sector_density_seconds"], record["device_release_seconds"]
        self.assertEqual(spent[2], 0.0)
        go.set()
        builder.release_joined()
        self.assertEqual(sorted(zip(self.pool.released, self.pool.threads)), list(zip((0, 1), self.device_threads)))
        self.assertEqual(builder._releasing, ())
        # The wait is time of the release and of the sector-wise density of the rank.
        self.assertGreater(record["release_seconds"], spent[0])
        self.assertGreater(record["sector_density_seconds"], spent[1])
        self.assertGreater(record["device_release_seconds"], 0.0)
        # A release that no caller has waited for is complete before the next one collects, however long
        # its devices take.
        self.pool.before = lambda index: sleep(0.05)
        self.release(builder, times=2)
        self.assertEqual((self.emptied, len(builder._releasing)), ([0, 2, 4], 2))
        builder.release_joined()
        self.assertEqual(sorted(self.pool.released), [0, 0, 0, 1, 1, 1])

    def test_an_error_of_a_device_thread_is_raised_when_all_have_ended(self) -> None:
        builder = self.builder()
        other_done = threading.Event()

        def device(index):
            if index == 0:
                # The device that fails does so while the other is still at work.
                raise RuntimeError("injected release failure")
            other_done.set()

        self.pool.before = device
        with self.assertRaisesRegex(RuntimeError, "injected release failure"):
            self.release(builder)
        self.assertTrue(other_done.is_set())
        self.assertEqual((self.pool.released, builder._releasing), ([1], ()))
        # Where the caller waits, the error is that of its wait.
        builder.caller_joins_release = True
        self.release(builder)
        with self.assertRaisesRegex(RuntimeError, "injected release failure"):
            builder.release_joined()
        self.assertEqual((self.pool.released, builder._releasing), ([1, 1], ()))
        self.pool.before = None
        self.release(builder)
        builder.release_joined()
        self.assertEqual(sorted(self.pool.released), [0, 1, 1, 1])

    def test_what_a_density_holds_on_a_device_is_freed_behind_the_release_also_where_a_caller_waits(self) -> None:
        # Two sectors, one on each device.  The selection of a sector is an array of its own here, as where
        # the selected columns are no prefix; that of the last sector lives until the builder returns.
        def density(builder, hold=None):
            """Events of one density, pools emptied and arrays freed: up to the wait, and up to the builder's return."""
            events = []

            class Sector(_FakeArray):
                def __getitem__(sector, key):
                    selection = _FakeArray(self.cuda, sector.device.id, 100)
                    weakref.finalize(selection, events.append, f"selection {sector.device.id} freed")
                    return selection

            def device(index):
                if hold is not None:
                    self.assertTrue(hold.wait(timeout=20))
                events.append(f"device {index} emptied")

            self.pool.before = device
            vectors = (Sector(self.cuda, 0, 400), Sector(self.cuda, 1, 400))
            for index, sector in enumerate(vectors):
                weakref.finalize(sector, events.append, f"vectors {index} freed")
            orbitals = CuPySymmetryOrbitals(
                scaled_wedge_vectors=None,
                representations=np.array([0, 0, 1, 1], dtype=np.int32),
                full_to_wedge=np.arange(3),
                device_full_to_wedge=None,
                phases=np.ones((2, 3)),
                full_size=3,
                representation_columns=np.array([0, 1, 1, 0], dtype=np.int32),
                sector_vectors=vectors,
                sector_orbits=(np.arange(3), np.arange(3)),
                sector_scales=(np.ones(3), np.ones(3)),
                wedge_size=3,
            )
            del vectors, sector
            with patch.dict(os.environ, PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES="0"), \
                 patch("parsec_python.acceleration.backends.cupy.require_cupy",
                       return_value=(self.cp, None)), \
                 patch.object(symmetry_density.gc, "collect", self.collect):
                values = builder(orbitals, np.ones(4), 0.5)
            # The caller lets go of the orbitals as soon as it has the density, as an MPI rank does.
            del orbitals
            np.testing.assert_array_equal(values, np.zeros(3))
            found = list(events)
            if hold is not None:
                hold.set()
            builder.release_joined()
            # Read while the builder lives: what it still kept would be freed with it.
            return list(events), found

        def builder(release=None, joins=False):
            made = self.builder(release)
            made.device_builder = lambda vectors, weights, volume: np.zeros(3)
            made.caller_joins_release = joins
            return made

        emptied = {"device 0 emptied", "device 1 emptied"}
        freed = ["selection 1 freed", "vectors 0 freed", "vectors 1 freed"]
        # Where the builder waits itself, as before: the selection of the first sector goes when the second
        # is selected, the pools are emptied, and then what the density held is freed and stays cached.
        for release in ("serial", None):
            events, found = density(builder(release))
            self.assertEqual((events[0], set(events[1:3]), sorted(events[3:])), ("selection 0 freed", emptied, freed))
            self.assertEqual(found, events)
        # Where the caller waits, the builder is back while no pool has been emptied and nothing more is
        # freed, although the caller has let go of the orbitals; the wait frees it, behind the release.
        events, found = density(builder(joins=True), threading.Event())
        self.assertEqual(found, ["selection 0 freed"])
        self.assertEqual((set(events[1:3]), sorted(events[3:])), (emptied, freed))
        # So it does where the device threads were faster than the caller.
        events, found = density(builder(joins=True))
        self.assertEqual(found[0], "selection 0 freed")
        self.assertLessEqual(set(found[1:]), emptied)
        self.assertEqual((set(events[1:3]), sorted(events[3:])), (emptied, freed))


class RunnerRecordTests(unittest.TestCase):
    def test_serial_control_keeps_the_former_host_work(self) -> None:
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        with patch.dict(os.environ):
            for name in mpi_full_scf._SERIAL_CONTROL_SETTINGS:
                os.environ.pop(name, None)
            mpi_full_scf._keep_former_routes(os.environ)
            self.assertEqual(density_collection(), "full")
            self.assertEqual(density_release(), "serial")
            self.assertFalse(scf_buffers_requested())
            # The release of a shared basis is not the control's to name: it shares none, and its density
            # step empties the pool of every device that holds a sector whatever that switch says.
            self.assertNotIn("PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE", mpi_full_scf._SERIAL_CONTROL_SETTINGS)
            harness = DensityReleaseTests("release")
            harness.setUp()
            for shared in (None, "0", "1"):
                os.environ.pop("PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE", None)
                if shared is not None:
                    os.environ["PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE"] = shared
                builder = CuPySymmetryDensityBuilder(lambda *args: None)
                del harness.pool.released[:], harness.pool.threads[:]
                harness.release(builder, times=2)
                self.assertEqual((harness.pool.released, set(harness.pool.threads)), ([0, 1] * 2, {harness.here}))
                self.assertEqual((builder.pool_release["shared_kept"], builder.pool_release["calls"]), (0, 2))
            os.environ.pop("PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE", None)
            # What the launcher names stays.
            os.environ.update(PARSEC_CUPY_DENSITY_COLLECTION="changed", PARSEC_CUPY_DENSITY_RELEASE="threads")
            mpi_full_scf._keep_former_routes(os.environ)
            self.assertEqual((density_collection(), density_release()), ("changed", "threads"))

    def test_timing_record_holds_the_release_record_of_the_rank(self) -> None:
        import shutil
        from pathlib import Path

        from parsec_python.acceleration.tests.test_mpi_scf import FullSCFRunnerContextTests

        harness = FullSCFRunnerContextTests("run_serial_control")
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        harness.environment("0")
        with patch.dict(os.environ, PARSEC_CUPY_DENSITY_COLLECTION="full"):
            builder = CuPySymmetryDensityBuilder(lambda *args: None)
        builder.pool_release.update(calls=3, full_collections=3, collection_seconds=0.25)
        directory = Path.cwd() / ".tmp" / f"runner-host-work-test-{os.getpid()}"
        try:
            record, _, _ = harness.run_serial_control(
                directory / "builder", orbital_density_builder=builder
            )
            (rank,) = record["per_rank"]
            self.assertEqual(rank["density_pool_release"], builder.pool_release)
            self.assertEqual(rank["density_pool_release"]["collection"], "full")
            # How the pools are emptied, and the seconds of the device threads beside those of the release.
            self.assertEqual(rank["density_pool_release"]["release"], builder.release)
            self.assertEqual(rank["density_pool_release"]["device_release_seconds"], 0.0)
            # A serial control has no sector context and so no commands.
            self.assertIsNone(rank["command_seconds"])
            self.assertTrue(any("density_pool_release" in note for note in record["notes"]))
            # A prepared system without a density builder has no record.
            record, _, _ = harness.run_serial_control(directory / "none")
            self.assertIsNone(record["per_rank"][0]["density_pool_release"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_backend_report_names_the_buffer_route(self) -> None:
        from parsec_python.acceleration.tests.test_hybrid_driver import MPIRolePreparationTests

        harness = MPIRolePreparationTests("prepare")
        self.addCleanup(harness.tearDown)
        # A value the switch does not know leaves it on, and the report says so.
        for value, expected in ((None, "on"), ("off", "off"), ("1", "on"), ("former", "on")):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop("PARSEC_SYMMETRY_SCF_BUFFERS", None)
                prepared = harness.prepare(
                    None,
                    environment={} if value is None else {"PARSEC_SYMMETRY_SCF_BUFFERS": value},
                )
                details = dict(prepared.system.backend_info.details)
                self.assertEqual(details["scf_scalar_field_buffers"], expected)

    def test_context_adds_up_the_seconds_of_each_command(self) -> None:
        from parsec_python.acceleration.experimental.mpi_scf import MPISCFError, MPISectorContext
        from parsec_python.acceleration.tests.test_mpi_domain import _Comm, _World

        context = MPISectorContext(_Comm(_World(1), 0))
        context.configure(4, (0,))
        solver = SimpleNamespace(_reset_local=lambda: None)
        context.execute(solver, "reset")
        first = context.command_seconds["reset"]
        self.assertGreater(first, 0.0)
        context.execute(solver, "reset")
        self.assertGreater(context.command_seconds["reset"], first)
        self.assertEqual(context.command_counts, {"reset": 2})
        # A command that fails is timed as well.
        with self.assertRaises(MPISCFError):
            context.execute(solver, "unknown")
        self.assertEqual(set(context.command_seconds), {"reset", "unknown"})


if __name__ == "__main__":
    unittest.main()
