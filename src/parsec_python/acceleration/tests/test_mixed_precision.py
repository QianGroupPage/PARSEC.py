"""The FP32 later filter is an opt-in whose launches follow the kernel declarations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gc
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.acceleration.backends import (
    cupy as cupy_backend,
    cupy_compact,
    cupy_mixed_precision,
    cupy_stencil_major,
    implicit_stencil,
)
from parsec_python.acceleration.backends.cupy import (
    CuPyHamiltonian,
    CuPyTimingStats,
    cupy_available,
    require_cupy,
)
from parsec_python.acceleration.backends.cupy_launch import (
    kernel_parameters,
    launch_arguments,
)
from parsec_python.acceleration.backends.cupy_mixed_precision import (
    CuPyMixedPrecisionRecurrence,
    float32_filter_requested,
    mixed_filter_policy,
)
from parsec_python.acceleration.backends.cupy_projectors import (
    _SOURCE as PROJECTOR_SOURCE,
)
from parsec_python.acceleration.backends.cupy_stencil_major import (
    _CUDA_SOURCE as STENCIL_SOURCE,
    build_stencil_major_metadata,
)
from parsec_python.acceleration.Eigensolvers.chebyshev import (
    _mixed_filter_requested,
    subspace_filter,
)
from parsec_python.acceleration.Output import AcceleratedTextReporter
from parsec_python.acceleration.SCF.single_point import _finalize_result
from parsec_python.acceleration.models import BackendInfo, BackendStatistics


GPU_AVAILABLE = cupy_available()

_SETTINGS = (
    "PARSEC_CUPY_MIXED_FILTER",
    "PARSEC_CUPY_MIXED_FILTER_MIN_ROWS",
    "PARSEC_CUPY_MIXED_FILTER_MIN_WORK",
    # A launcher may name tiles, which the FP32 recurrence refuses, and
    # other routes for the filter than the block loop these tests follow.
    "PARSEC_CUPY_IMPLICIT_TILE",
    "PARSEC_CUPY_FILTER_GRAPHS",
    "PARSEC_CUPY_DISTRIBUTED_FILTER",
    "PARSEC_CUPY_BATCH_FILTERS",
)
# Center, scale, sigma and the sigma that follows.  All four differ and none
# is 0 or 1, so that a value in the place of another changes the result.
_STEP = dict(center=0.5, scale=1.0 / 64.0, sigma=0.4, sigma_next=1.5)
_FLOAT32_TYPES = {
    "int": np.int32,
    "long long": np.int64,
    "float": np.float32,
    "int *": np.int32,
    "unsigned char *": np.uint8,
    "float *": np.float32,
}


def _environment(**settings):
    """The process environment without the filter settings, then ``settings``."""

    cleared = {
        name: value for name, value in os.environ.items() if name not in _SETTINGS
    }
    return patch.dict(os.environ, {**cleared, **settings}, clear=True)


def _operator(dimension=40):
    """A small operator with rows of unequal width and two KB projectors."""

    kinetic = sp.diags(
        (
            -0.25 * np.ones(dimension - 3),
            -np.ones(dimension - 1),
            2.4 * np.ones(dimension),
            -np.ones(dimension - 1),
            0.125 * np.ones(dimension - 5),
        ),
        (-3, -1, 0, 1, 5),
        format="csr",
    )
    potential = np.linspace(-0.75, 0.35, dimension)
    dense = np.zeros((dimension, 2), dtype=np.float64)
    dense[3:7, 0] = (0.2, -0.35, 0.15, 0.1)
    dense[15:20, 1] = (-0.1, 0.25, 0.3, -0.2, 0.05)
    return kinetic, potential, sp.csr_matrix(dense), np.array((1.0, -1.0))


def _dense_hamiltonian(kinetic, potential, projectors, signs):
    return (
        kinetic.toarray()
        + np.diag(potential)
        + projectors.toarray() @ np.diag(signs) @ projectors.toarray().T
    )


def _reference_step(hamiltonian, current, previous=None):
    value = (hamiltonian @ current - _STEP["center"] * current) * _STEP["scale"]
    if previous is not None:
        value -= _STEP["sigma"] * previous
    return value * _STEP["sigma_next"]


def _check_strides(named, prefix, array):
    itemsize = array.dtype.itemsize
    expected = tuple(stride // itemsize for stride in array.strides)
    actual = (named[f"{prefix}_row_stride"], named[f"{prefix}_column_stride"])
    if actual != expected:
        raise AssertionError(f"{prefix} strides {actual} are not {expected}")


def _host_recurrence(named):
    """``stencil_major_chebyshev6`` in NumPy, read by parameter name."""

    rows, slots = int(named["row_count"]), int(named["slot_count"])
    width = int(named["width"])
    current, previous, output = named["current"], named["previous"], named["output"]
    coefficients = named["projector_coefficients"]
    for prefix, array in (
        ("current", current),
        ("previous", previous),
        ("output", output),
        ("coefficient", coefficients),
    ):
        _check_strides(named, prefix, array)
    neighbors = named["neighbors"].reshape(slots, rows)
    codes = named["coefficient_codes"].reshape(slots, rows)
    value = named["local_potential"][:, None] * current[:, :width]
    for slot in range(slots):
        active = neighbors[slot] >= 0
        coefficient = named["coefficient_palette"][codes[slot, active]]
        value[active] += coefficient[:, None] * current[neighbors[slot, active], :width]
    if named["add_nonlocal"]:
        scatter = sp.csr_matrix(
            (
                named["projector_values"],
                named["projector_columns"],
                named["projector_row_offsets"],
            ),
            shape=(rows, coefficients.shape[0]),
        )
        value += scatter @ coefficients[:, :width]
    if named["use_parameters"]:
        first = 4 * int(named["parameter_step"])
        center, scale, sigma, sigma_next = named["recurrence_parameters"][first:first + 4]
    else:
        center, scale, sigma, sigma_next = (
            named[f"{name}_argument"] for name in ("center", "scale", "sigma", "sigma_next")
        )
    value = (value - center * current[:, :width]) * scale
    if named["add_previous"]:
        value -= sigma * previous[:, :width]
    output[:, :width] = value * sigma_next


def _host_projection(named):
    """``sparse_projector_projection`` in NumPy, read by parameter name."""

    count, width = int(named["projector_count"]), int(named["width"])
    vectors, output = named["vectors"], named["output"]
    _check_strides(named, "vector", vectors)
    _check_strides(named, "output", output)
    transpose = sp.csr_matrix(
        (named["projector_values"], named["grid_rows"], named["row_offsets"]),
        shape=(count, vectors.shape[0]),
    )
    output[:, :width] = named["signs"][:, None] * (transpose @ vectors[:, :width])


class _HostKernel:
    """Stands in for a compiled kernel: a launch must fit its declaration."""

    def __init__(self, parameters, evaluate):
        self.parameters = parameters
        self.evaluate = evaluate
        self.launches = 0

    def __call__(self, grid, block, arguments):
        if len(arguments) != len(self.parameters):
            raise AssertionError(
                f"{len(arguments)} arguments for {len(self.parameters)} parameters"
            )
        named = {}
        for (kind, name), value in zip(self.parameters, arguments):
            actual = value.dtype.type if kind.endswith("*") else type(value)
            if actual is not _FLOAT32_TYPES[kind]:
                raise AssertionError(f"{name} is declared {kind}, got {actual}")
            named[name] = value
        self.launches += 1
        self.evaluate(named)


def _float32_host_kernels():
    """The three FP32 kernels in the order of ``_kernels``, as NumPy stand-ins."""

    return tuple(
        _HostKernel(kernel_parameters(cupy_mixed_precision._SOURCE, name), evaluate)
        for name, evaluate in zip(
            cupy_mixed_precision._KERNEL_NAMES,
            (_host_recurrence, _host_projection, _host_projection),
            strict=True,
        )
    )


class _HostRawKernel:
    """A raw kernel that compiles to nothing: its operator is built, not applied."""

    def __init__(self, source, name, options=()):
        self.name = name

    def compile(self):
        pass


class _HostCuPy:
    """NumPy in place of CuPy for building an operator on device 0."""

    RawKernel = _HostRawKernel
    cuda = SimpleNamespace(
        Device=lambda: SimpleNamespace(id=0),
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
    )

    def __getattr__(self, name):
        return getattr(np, name)


@contextmanager
def host_cupy():
    """Let ``CuPyHamiltonian`` be built and held without CUDA.

    SciPy's sparse matrices stand in for cuSPARSE's.  The kernel caches of
    the process are emptied meanwhile and restored afterwards, so that no
    stand-in kernel is left for a device test of the same process.
    """

    from parsec_python.acceleration.backends import cupy_projectors

    with (
        patch.object(cupy_backend, "require_cupy", return_value=(_HostCuPy(), sp)),
        patch.dict(cupy_stencil_major._KERNEL_CACHE, clear=True),
        patch.dict(cupy_compact._KERNEL_CACHE, clear=True),
        patch.dict(cupy_projectors._KERNEL_CACHE, clear=True),
        patch.dict(implicit_stencil._CACHE, clear=True),
    ):
        yield


class _DenseOperator:
    """A dense Hamiltonian with both recurrences, as a filter asks of an operator."""

    def __init__(self, hamiltonian, float32):
        self.hamiltonian = hamiltonian
        self.shape = hamiltonian.shape
        self.timing_stats = CuPyTimingStats()
        self.mixed_precision_recurrence = object() if float32 else None
        self.steps = []

    def _step(self, kind, current, center, scale, sigma_next, previous, sigma):
        self.steps.append(kind)
        if current.dtype != kind:
            raise AssertionError(f"a {kind.__name__} step was given {current.dtype}")
        value = (self.hamiltonian.astype(kind) @ current - kind(center) * current) * kind(scale)
        if previous is not None:
            value -= kind(sigma) * previous
        return (value * kind(sigma_next)).astype(kind)

    def chebyshev_recurrence(
        self, current, *, center, scale, sigma_next=1.0, previous=None, sigma=0.0
    ):
        return self._step(np.float64, current, center, scale, sigma_next, previous, sigma)

    def chebyshev_recurrence_float32(
        self, current, *, center, scale, sigma_next=1.0, previous=None, sigma=0.0
    ):
        return self._step(np.float32, current, center, scale, sigma_next, previous, sigma)


class MixedFilterPolicyTests(unittest.TestCase):
    def test_every_filter_is_float64_unless_asked(self):
        # Large enough for both thresholds of ``auto``.
        operator = SimpleNamespace(
            shape=(4_000_000, 4_000_000), mixed_precision_recurrence=object()
        )
        with _environment():
            self.assertEqual(mixed_filter_policy(), "off")
            self.assertFalse(float32_filter_requested(4_000_000))
            self.assertFalse(_mixed_filter_requested(operator, 3000))
            self.assertFalse(_mixed_filter_requested(operator))

    def test_explicit_requests_keep_their_meaning(self):
        operator = SimpleNamespace(
            shape=(100, 100), mixed_precision_recurrence=object()
        )
        without = SimpleNamespace(shape=(100, 100), mixed_precision_recurrence=None)
        with _environment(PARSEC_CUPY_MIXED_FILTER="auto"):
            self.assertEqual(mixed_filter_policy(), "auto")
            self.assertFalse(float32_filter_requested(99_999))
            self.assertTrue(float32_filter_requested(100_000))
            os.environ["PARSEC_CUPY_MIXED_FILTER_MIN_ROWS"] = "24"
            self.assertFalse(float32_filter_requested(23))
            self.assertTrue(float32_filter_requested(24))
            os.environ["PARSEC_CUPY_MIXED_FILTER_MIN_WORK"] = "1000"
            self.assertFalse(_mixed_filter_requested(operator, 3))
            self.assertTrue(_mixed_filter_requested(operator, 4))
            self.assertFalse(_mixed_filter_requested(without, 4))
        with _environment(PARSEC_CUPY_MIXED_FILTER="on"):
            self.assertEqual(mixed_filter_policy(), "on")
            self.assertTrue(float32_filter_requested(1))
            self.assertTrue(_mixed_filter_requested(operator, 1))
            self.assertFalse(_mixed_filter_requested(without, 1))
        with _environment(PARSEC_CUPY_MIXED_FILTER="off"):
            self.assertFalse(float32_filter_requested(4_000_000))
            self.assertFalse(_mixed_filter_requested(operator, 100))

    def test_setting_spellings_and_errors(self):
        for value, policy in (
            ("0", "off"), ("false", "off"), ("no", "off"), (" OFF ", "off"),
            ("1", "on"), ("true", "on"), ("yes", "on"), ("Auto", "auto"),
        ):
            with self.subTest(value=value), _environment(PARSEC_CUPY_MIXED_FILTER=value):
                self.assertEqual(mixed_filter_policy(), policy)
        with _environment(PARSEC_CUPY_MIXED_FILTER="single"):
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_MIXED_FILTER"):
                mixed_filter_policy()
        for value in ("0", "many"):
            with self.subTest(minimum=value), _environment(
                PARSEC_CUPY_MIXED_FILTER_MIN_ROWS=value
            ):
                with self.assertRaisesRegex(ValueError, "MIN_ROWS"):
                    float32_filter_requested(10)


class KernelLaunchTests(unittest.TestCase):
    def test_parameters_are_read_from_the_declaration(self):
        source = """
        extern "C" __global__
        void other(const int count) {}
        extern "C" __global__
        void sample(
            const long long row_count,
            const int* __restrict__ neighbors,
            const unsigned char* __restrict__ codes,
            double* __restrict__ output,
            const float scale
        ) {}
        """
        self.assertEqual(
            kernel_parameters(source, "sample"),
            (
                ("long long", "row_count"),
                ("int *", "neighbors"),
                ("unsigned char *", "codes"),
                ("double *", "output"),
                ("float", "scale"),
            ),
        )
        self.assertEqual(kernel_parameters(source, "other"), (("int", "count"),))
        with self.assertRaisesRegex(ValueError, "not declared"):
            kernel_parameters(source, "sampl")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            kernel_parameters("void short_kernel(const short value) {}", "short_kernel")

    def test_arguments_are_placed_and_typed_by_the_declaration(self):
        parameters = (
            ("long long", "row_count"),
            ("float *", "values"),
            ("int", "width"),
            ("float", "scale"),
            ("double", "shift"),
        )
        values = np.zeros(3, dtype=np.float32)
        arguments = launch_arguments(
            parameters,
            dict(scale=0.5, width=2, shift=1, values=values, row_count=3),
        )
        self.assertEqual(
            [type(item) for item in arguments],
            [np.int64, np.ndarray, np.int32, np.float32, np.float64],
        )
        self.assertIs(arguments[1], values)
        self.assertEqual(arguments[0], 3)
        self.assertEqual(arguments[3], np.float32(0.5))
        complete = dict(scale=0.5, width=2, shift=1.0, values=values, row_count=3)
        missing = {name: value for name, value in complete.items() if name != "width"}
        with self.assertRaisesRegex(TypeError, "width"):
            launch_arguments(parameters, missing)
        with self.assertRaisesRegex(TypeError, "step"):
            launch_arguments(parameters, dict(complete, step=1))
        with self.assertRaisesRegex(TypeError, "values"):
            launch_arguments(parameters, dict(complete, values=values.astype(np.float64)))

    def test_float32_kernels_declare_what_the_float64_kernels_declare(self):
        for source, name in (
            (STENCIL_SOURCE, "stencil_major_chebyshev6"),
            (PROJECTOR_SOURCE, "sparse_projector_projection"),
            (PROJECTOR_SOURCE, "sparse_projector_projection_serial"),
        ):
            with self.subTest(kernel=name):
                double = kernel_parameters(source, name)
                single = kernel_parameters(
                    cupy_mixed_precision._SOURCE, f"{name}_float32"
                )
                self.assertEqual(
                    single,
                    tuple((kind.replace("double", "float"), item) for kind, item in double),
                )
                self.assertFalse(any("double" in kind for kind, _ in single))
                self.assertIn(f"{name}_float32", cupy_mixed_precision._KERNEL_NAMES)


class HostFloat32RecurrenceTests(unittest.TestCase):
    """The launches of the FP32 recurrence, evaluated by NumPy stand-ins."""

    def recurrence(self, kinetic, potential, projectors, signs):
        metadata = build_stencil_major_metadata(kinetic)
        stencil = SimpleNamespace(
            shape=metadata.shape,
            slot_count=metadata.neighbors.shape[0],
            neighbors=metadata.neighbors,
            coefficient_codes=metadata.coefficient_codes,
            coefficient_palette=metadata.coefficient_palette,
        )
        # In the order of ``_kernels``: recurrence, tree projection, serial one.
        kernels = _float32_host_kernels()
        with patch.object(cupy_mixed_precision, "_kernels", return_value=kernels):
            # NumPy stands in for CuPy: the class asks it for arrays only.
            built = CuPyMixedPrecisionRecurrence(
                np, stencil, projectors, signs, potential
            )
        return built, kernels

    def assert_step(self, actual, expected):
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, expected.shape)
        self.assertLessEqual(
            np.linalg.norm(actual - expected), 1.0e-5 * np.linalg.norm(expected)
        )

    def test_one_step_fills_every_parameter_and_equals_float64(self):
        kinetic, potential, projectors, signs = _operator()
        hamiltonian = _dense_hamiltonian(kinetic, potential, projectors, signs)
        recurrence, kernels = self.recurrence(kinetic, potential, projectors, signs)
        generator = np.random.default_rng(31)
        current = generator.standard_normal((kinetic.shape[0], 8))
        previous = generator.standard_normal((kinetic.shape[0], 8))
        single = current.astype(np.float32, order="F")
        single_previous = previous.astype(np.float32, order="F")

        first = recurrence(
            single,
            center=_STEP["center"],
            scale=_STEP["scale"],
            sigma_next=_STEP["sigma_next"],
        )
        self.assert_step(first, _reference_step(hamiltonian, current))
        # Eight columns are two launches of at most six; one projection.
        self.assertEqual([kernel.launches for kernel in kernels], [2, 0, 1])

        following = recurrence(single, previous=single_previous, **_STEP)
        self.assert_step(following, _reference_step(hamiltonian, current, previous))

        # A row-major view and a single vector take the same launches.
        rows_first = np.ascontiguousarray(single)
        self.assert_step(
            recurrence(rows_first, previous=single_previous, **_STEP),
            _reference_step(hamiltonian, current, previous),
        )
        vector = recurrence(single[:, 0], previous=single_previous[:, 0], **_STEP)
        self.assert_step(
            vector, _reference_step(hamiltonian, current[:, 0], previous[:, 0])
        )

        updated = potential + 0.07 * np.sin(np.arange(potential.size))
        recurrence.update_potential(updated)
        self.assert_step(
            recurrence(single, previous=single_previous, **_STEP),
            _reference_step(
                _dense_hamiltonian(kinetic, updated, projectors, signs),
                current,
                previous,
            ),
        )

    def test_operator_without_projectors(self):
        kinetic, potential, _, _ = _operator()
        empty = sp.csr_matrix((kinetic.shape[0], 0), dtype=np.float64)
        recurrence, kernels = self.recurrence(kinetic, potential, empty, np.empty(0))
        hamiltonian = kinetic.toarray() + np.diag(potential)
        current = np.random.default_rng(32).standard_normal((kinetic.shape[0], 3))
        self.assert_step(
            recurrence(current.astype(np.float32, order="F"), previous=None, **_STEP),
            _reference_step(hamiltonian, current),
        )
        self.assertEqual([kernel.launches for kernel in kernels], [1, 0, 0])

    def test_long_projector_rows_take_the_parallel_projection(self):
        dimension = 320
        kinetic = sp.eye(dimension, format="csr")
        projectors = sp.csr_matrix(np.linspace(0.1, 1.0, dimension)[:, None])
        signs = np.ones(1)
        recurrence, kernels = self.recurrence(
            kinetic, np.zeros(dimension), projectors, signs
        )
        current = np.random.default_rng(33).standard_normal((dimension, 2))
        self.assert_step(
            recurrence(current.astype(np.float32, order="F"), previous=None, **_STEP),
            _reference_step(
                _dense_hamiltonian(kinetic, np.zeros(dimension), projectors, signs),
                current,
            ),
        )
        self.assertEqual([kernel.launches for kernel in kernels], [1, 1, 0])

    def test_implicit_tiles_are_refused(self):
        kinetic, potential, projectors, signs = _operator()
        packed = SimpleNamespace(implicit_statistics=dict(tile=16))
        with self.assertRaisesRegex(ValueError, "implicit tiles"):
            CuPyMixedPrecisionRecurrence(np, packed, projectors, signs, potential)


class HostOperatorTests(unittest.TestCase):
    """Which operators hold an FP32 recurrence: ``CuPyHamiltonian`` without CUDA."""

    def build(self, dimension=40, **settings):
        kinetic, potential, projectors, signs = _operator(dimension)
        with (
            _environment(**settings),
            host_cupy(),
            patch.object(
                cupy_mixed_precision, "_kernels", return_value=_float32_host_kernels()
            ),
        ):
            return CuPyHamiltonian(
                kinetic, potential, projectors=projectors, projector_signs=signs
            )

    def test_default_builds_none_for_an_operator_of_any_size(self):
        # From 100,000 rows on the former default, ``auto``, built one.
        for settings in ({}, {"PARSEC_CUPY_MIXED_FILTER": "off"}):
            for dimension in (40, 100_000):
                with self.subTest(settings=settings, dimension=dimension):
                    gpu = self.build(dimension, **settings)
                    self.assertEqual(gpu.shape, (dimension, dimension))
                    self.assertIsNone(gpu.mixed_precision_recurrence)
                    self.assertEqual(
                        gpu.mixed_precision_filter_reason, "disabled by policy"
                    )

    def test_an_explicit_request_builds_one_that_the_operator_steps_through(self):
        large = self.build(100_000, PARSEC_CUPY_MIXED_FILTER="auto")
        self.assertIsInstance(
            large.mixed_precision_recurrence, CuPyMixedPrecisionRecurrence
        )
        self.assertIsNone(large.mixed_precision_filter_reason)
        small = self.build(40, PARSEC_CUPY_MIXED_FILTER="auto")
        self.assertIsNone(small.mixed_precision_recurrence)
        self.assertEqual(
            small.mixed_precision_filter_reason, "below automatic row threshold"
        )
        lowered = self.build(
            40, PARSEC_CUPY_MIXED_FILTER="auto", PARSEC_CUPY_MIXED_FILTER_MIN_ROWS="40"
        )
        self.assertIsNotNone(lowered.mixed_precision_recurrence)

        gpu = self.build(40, PARSEC_CUPY_MIXED_FILTER="on")
        self.assertIsNone(gpu.mixed_precision_filter_reason)
        kinetic, potential, projectors, signs = _operator(40)
        generator = np.random.default_rng(35)
        current = generator.standard_normal((40, 8))
        previous = generator.standard_normal((40, 8))
        with host_cupy():
            result = gpu.chebyshev_recurrence_float32(
                current.astype(np.float32, order="F"),
                previous=previous.astype(np.float32, order="F"),
                **_STEP,
            )
        expected = _reference_step(
            _dense_hamiltonian(kinetic, potential, projectors, signs), current, previous
        )
        self.assertEqual(result.dtype, np.float32)
        self.assertLessEqual(
            np.linalg.norm(result - expected), 1.0e-5 * np.linalg.norm(expected)
        )
        self.assertEqual(gpu.timing_stats.orbital_vectors_applied, 8)


class LaterFilterRecordTests(unittest.TestCase):
    """What a later pass ran in is counted where it runs and kept with its solve."""

    def setUp(self):
        kinetic, potential, projectors, signs = _operator()
        self.hamiltonian = _dense_hamiltonian(kinetic, potential, projectors, signs)
        self.vectors = np.asfortranarray(
            np.random.default_rng(36).standard_normal((kinetic.shape[0], 8))
        )

    def filter(self, operator, mixed_precision=None, **settings):
        chebyshev = importlib.import_module(
            "parsec_python.acceleration.Eigensolvers.chebyshev"
        )
        # NumPy stands in for CuPy: the block loop asks it for arrays only.
        with (
            _environment(**settings),
            patch.object(chebyshev, "require_cupy", return_value=(np, None)),
        ):
            return subspace_filter(
                operator,
                self.vectors,
                degree=6,
                degree_delta=1,
                lower_bound=1.0,
                upper_bound=7.0,
                mixed_precision=mixed_precision,
            )

    def test_a_pass_is_counted_only_where_it_runs_the_float32_recurrence(self):
        operator = _DenseOperator(self.hamiltonian, float32=True)
        float32_passes = lambda: operator.timing_stats.subspace_filter_float32_passes
        expected = self.filter(operator)
        self.assertEqual(set(operator.steps), {np.float64})
        self.assertEqual(float32_passes(), 0)

        for count in (1, 2):
            del operator.steps[:]
            result = self.filter(operator, PARSEC_CUPY_MIXED_FILTER="on")
            self.assertEqual(set(operator.steps), {np.float32})
            self.assertEqual(float32_passes(), count)
            self.assertEqual(result.dtype, np.float64)
            difference = np.linalg.norm(result - expected)
            self.assertGreater(difference, 0.0)
            self.assertLessEqual(difference, 1.0e-4 * np.linalg.norm(expected))

        # ``auto`` asks every pass for enough work: 40 rows times 8 states squared.
        for work, counted in (("2561", 2), ("2560", 3)):
            del operator.steps[:]
            self.filter(
                operator,
                PARSEC_CUPY_MIXED_FILTER="auto",
                PARSEC_CUPY_MIXED_FILTER_MIN_WORK=work,
            )
            self.assertEqual(float32_passes(), counted)
            self.assertEqual(
                set(operator.steps), {np.float32 if counted == 3 else np.float64}
            )

        # A caller that names FP32 against the policy gets the FP64 blocks.
        del operator.steps[:]
        np.testing.assert_array_equal(
            self.filter(operator, mixed_precision=True), expected
        )
        self.assertEqual(set(operator.steps), {np.float64})
        # The filter spread over devices refuses FP32 before any block runs.
        with self.assertRaisesRegex(ValueError, "FP64"):
            self.filter(
                operator,
                PARSEC_CUPY_MIXED_FILTER="on",
                PARSEC_CUPY_DISTRIBUTED_FILTER="1",
            )
        self.assertEqual(float32_passes(), 3)

        without = _DenseOperator(self.hamiltonian, float32=False)
        np.testing.assert_array_equal(
            self.filter(without, PARSEC_CUPY_MIXED_FILTER="on"), expected
        )
        self.assertEqual(without.timing_stats.subspace_filter_float32_passes, 0)

    def test_a_solve_records_the_precision_its_filter_ran_in(self):
        eigval = importlib.import_module(
            "parsec_python.acceleration.Eigensolvers.eigval"
        )
        rows = 9
        solver = object.__new__(eigval.CuPyEigvalSolver)
        solver.settings = EigvalSettings(initial_method="chebdav", safety_buffer=0)
        solver.operator = SimpleNamespace(shape=(rows, rows))
        solver._state = None
        solver.timing_stats = CuPyTimingStats()
        solver.compute_subspace_residuals, solver.retain_vectors_on_device = False, True
        float32_pass = [False]

        def call(function, operator, argument, **options):
            """Stands for the solvers; a later pass counts as ``subspace_filter`` would."""
            if function is eigval.run_subspace_filter:
                if float32_pass[0]:
                    solver.timing_stats.subspace_filter_float32_passes += 1
                return SimpleNamespace(state=argument, residual_norms=None), 0.0
            state = SimpleNamespace(
                operator_dimension=rows,
                wanted_states=argument,
                eigenvalues=np.zeros(argument),
                vectors=np.zeros((rows, argument), order="F"),
            )
            return SimpleNamespace(state=state, residual_norms=None), 0.0

        with (
            patch.object(eigval, "synchronized_call", call),
            patch.object(eigval, "resolve_device_stages", lambda stats: None),
            patch.object(
                eigval,
                "require_cupy",
                lambda: (SimpleNamespace(asnumpy=np.asarray), None),
            ),
        ):
            recorded = []
            for float32_pass[0] in (True, False, True, True, False):
                result = solver.solve(5)
                recorded.append((result.solver_path, result.filter_precision))
        # The first solve is FP64 whatever a later pass will do.
        self.assertEqual(
            recorded,
            [
                ("chebdav", "float64"),
                ("subspace", "float64"),
                ("subspace", "float32"),
                ("subspace", "float32"),
                ("subspace", "float64"),
            ],
        )


class FilterPrecisionReportTests(unittest.TestCase):
    def test_result_reports_what_ran_in_the_place_of_what_was_prepared(self):
        prepared = "float32 stencil/projectors/recurrence prepared for sectors 0 1; float64 Ritz and SCF"
        details = (
            ("orbital_sector_projector_reduction", "none"),
            ("orbital_sector_later_filter_precision", prepared),
            ("orbital_sector_local_potential_storage", "shared"),
        )
        for ran in ("float64", "float32 stencil/projectors/recurrence in 3 of 18 later filter passes (sectors 1); float64 Ritz and SCF"):
            with self.subTest(ran=ran):
                system = SimpleNamespace(
                    backend_info=BackendInfo(
                        requested="auto", selected="cupy", details=details
                    ),
                    materialize_final_wavefunctions=False,
                    backend=SimpleNamespace(statistics=BackendStatistics()),
                )
                eigensolver = SimpleNamespace(
                    state=SimpleNamespace(sector_state_counts=(3, 4)),
                    memory_allocator_policy="pool",
                    sector_state_storage="device",
                    later_filter_precision=ran,
                )
                final = _finalize_result(system, object(), eigensolver)
                # In its place among the details; the entries of the end follow.
                self.assertEqual(
                    final.backend.details[:3],
                    (details[0], (details[1][0], ran), details[2]),
                )
                self.assertEqual(
                    [key for key, _ in final.backend.details[3:]],
                    [
                        "orbital_sector_final_state_counts",
                        "orbital_memory_allocator",
                        "orbital_sector_state_storage",
                    ],
                )
                self.assertIs(system.backend.info, final.backend)

    def test_text_report_adds_what_ran_only_where_float32_was_asked_for(self):
        key = "orbital_sector_later_filter_precision"
        prepared = "float32 stencil/projectors/recurrence prepared for sectors 0 1; float64 Ritz and SCF"
        ran = "float32 stencil/projectors/recurrence in 3 of 18 later filter passes (sectors 1); float64 Ritz and SCF"

        from parsec_python.Input import parse_parsec_input

        translation = parse_parsec_input(
            Path(__file__).resolve().parents[2] / "tests/data/H_cli_smoke.in"
        )

        def report(at_setup, at_finish):
            messages = []
            reporter = AcceleratedTextReporter(messages.append, translation)
            reporter.reference = Mock()
            system = SimpleNamespace(
                backend_info=BackendInfo(
                    requested="auto", selected="cupy", details=((key, at_setup),)
                ),
                backend=SimpleNamespace(statistics=BackendStatistics()),
            )
            reporter.setup(system)
            result = SimpleNamespace(
                backend=replace(system.backend_info, details=((key, at_finish),)),
                backend_statistics=BackendStatistics(),
            )
            reporter.finish(result, 1.0)
            return "\n".join(messages)

        self.assertNotIn("as run", report("float64", "float64"))
        self.assertIn(f" Later filter precision as run = {ran}", report(prepared, ran))
        # A basis shared among devices filters in FP64 whatever was prepared.
        self.assertIn(
            " Later filter precision as run = float64", report(prepared, "float64")
        )


@unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA are not available")
class DeviceFloat32RecurrenceTests(unittest.TestCase):
    def tearDown(self):
        # An operator that a test left in a reference cycle is destroyed here,
        # between the tests, and not by a collection inside the next one.
        gc.collect()

    def test_one_float32_step_equals_the_float64_step(self):
        cp, _ = require_cupy()
        kinetic, potential, projectors, signs = _operator()
        with _environment(PARSEC_CUPY_MIXED_FILTER="on"):
            gpu = CuPyHamiltonian(
                kinetic, potential, projectors=projectors, projector_signs=signs
            )
        self.assertIsNotNone(
            gpu.mixed_precision_recurrence, gpu.mixed_precision_filter_reason
        )
        self.assertEqual(gpu.projector_count, 2)
        generator = np.random.default_rng(34)
        for columns in (1, 6, 8):
            current = cp.asarray(
                generator.standard_normal((kinetic.shape[0], columns)), order="F"
            )
            former = cp.asarray(
                generator.standard_normal((kinetic.shape[0], columns)), order="F"
            )
            for previous in (None, former):
                with self.subTest(columns=columns, previous=previous is not None):
                    expected = cp.asnumpy(
                        gpu.chebyshev_recurrence(current, previous=previous, **_STEP)
                    )
                    result = gpu.chebyshev_recurrence_float32(
                        current.astype(cp.float32),
                        previous=(
                            None if previous is None else previous.astype(cp.float32)
                        ),
                        **_STEP,
                    )
                    self.assertEqual(result.dtype, cp.dtype(cp.float32))
                    actual = cp.asnumpy(result)
                    self.assertLessEqual(
                        np.linalg.norm(actual - expected),
                        1.0e-5 * np.linalg.norm(expected),
                    )

    def test_default_operator_holds_no_float32_recurrence(self):
        kinetic, potential, projectors, signs = _operator()
        with _environment():
            gpu = CuPyHamiltonian(
                kinetic, potential, projectors=projectors, projector_signs=signs
            )
        self.assertIsNone(gpu.mixed_precision_recurrence)
        self.assertEqual(gpu.mixed_precision_filter_reason, "disabled by policy")


if __name__ == "__main__":
    unittest.main()
