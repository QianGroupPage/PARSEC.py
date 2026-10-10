"""The small generalized Ritz solve.

Its host form must give the coefficients of the plain whole-matrix form bit for
bit; its device form must agree with the host form and with dense references.
"""
import contextlib
import gc
import importlib
import os
import sys
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy

# The package also exports a function named like this submodule.
ritz = importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')


def _device_count():
    if not cupy_available():
        return 0
    cp, _ = require_cupy()
    return int(cp.cuda.runtime.getDeviceCount())


def _gram_pair(columns, seed, condition=None):
    """``X.T X`` and ``X.T H X`` of a random basis; nothing above the diagonals may be read."""
    rng = np.random.default_rng(seed)
    rows = 2*columns+3
    x = rng.normal(size=(rows, columns))
    if condition is not None:
        x = (np.linalg.qr(x)[0]*np.geomspace(1., condition**-.5, columns)) @ np.linalg.qr(rng.normal(size=(columns, columns)))[0]
    h = rng.normal(size=(rows, rows))
    overlap, projection = x.T @ x, x.T @ ((h+h.T) @ x)
    above = np.triu_indices(columns, 1)
    overlap[above] = 1e3*rng.normal(size=above[0].size)
    projection[above] = np.nan
    return overlap, projection


def _whole_matrix_solve(raw_overlap, raw_projection, eigh=np.linalg.eigh):
    """The solve with whole-matrix triangle sums, a symmetric whitened matrix and an explicit identity."""
    overlap = np.tril(raw_overlap)+np.tril(raw_overlap, -1).T
    projected = np.tril(raw_projection)+np.tril(raw_projection, -1).T
    cholesky = np.linalg.cholesky(overlap)
    left = la.solve_triangular(cholesky, projected, lower=True, check_finite=False)
    whitened = la.solve_triangular(cholesky, left.T, lower=True, check_finite=False).T
    whitened = np.tril(whitened)+np.tril(whitened, -1).T
    values, vectors = eigh(whitened)
    coefficients = la.solve_triangular(cholesky.T, vectors, lower=False, check_finite=False)
    audit = float(np.max(np.abs(coefficients.T @ overlap @ coefficients-np.eye(values.size))))
    return values, coefficients, whitened, overlap, audit


# Thresholds and methods of the measured runs, whatever the calling shell exports.
_HOST = dict(PARSEC_CUPY_RITZ_EIGH_BACKEND='host', PARSEC_CUPY_RITZ_CONDITION='symmetric',
             PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8', PARSEC_CUPY_RITZ_DENSE_BACKEND='host')
_DEVICE = {**_HOST, 'PARSEC_CUPY_RITZ_DENSE_BACKEND': 'device'}
# Solver routes of the multi-device tests.
_ROUTES = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_FILTER='0',
               PARSEC_CUPY_FILTER_GRAPHS='1', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1', PARSEC_CUPY_RITZ_ROTATION='reuse',
               PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1', PARSEC_CUPY_STREAMING_RITZ='0')


def _helper_cases(devices):
    """Setting, device with the Hartree objects -> device of the group ``devices`` that estimates the condition number.

    ``any`` asks the device with the Hartree objects only where the sector has no other: a group of three or more
    passes it over as ``auto`` does, and a group of two leaves the estimate to its owner under ``auto``.
    """
    owner, second = devices[:2]
    third = devices[2] if len(devices) > 2 else owner
    return (('off', None, owner), ('auto', None, second), ('auto', second, third),
            ('any', second, second if len(devices) == 2 else third))


def _operator(rows):
    """A small Hamiltonian with projectors on the current device."""
    return CuPyHamiltonian(sp.diags((-np.ones(rows-1), 2.2*np.ones(rows), -np.ones(rows-1)), (-1, 0, 1)),
                           np.linspace(-.1, .1, rows),
                           (sp.csr_matrix(np.random.default_rng(8).normal(size=(rows, 2))*.002), np.array([1., -1.])),
                           retain_generic_laplacian=False)


@contextlib.contextmanager
def _numpy_as_device(events):
    """NumPy under the names that the device small solve asks of CuPy and cupyx.

    ``events`` receives, in their order, the synchronizations of the current stream and the spectra, the
    factorizations and the eigensolves.  A host can then follow the order of that solve, not its arithmetic.
    """
    def named(name, function):
        def call(*arguments, **options):
            events.append(name)
            return function(*arguments, **options)
        return call

    stream = SimpleNamespace(synchronize=lambda: events.append('synchronize'))
    cp = SimpleNamespace(
        tril=np.tril, asfortranarray=np.asfortranarray, asnumpy=np.asarray, isfinite=np.isfinite, abs=np.abs, eye=np.eye,
        float64=np.float64, cuda=SimpleNamespace(get_current_stream=lambda: stream),
        linalg=SimpleNamespace(cholesky=named('cholesky', np.linalg.cholesky), eigh=named('eigh', np.linalg.eigh),
                               eigvalsh=named('eigvalsh', np.linalg.eigvalsh)))
    cupyx = ModuleType('cupyx')
    cupyx.errstate = lambda linalg: contextlib.nullcontext()
    cupyx.scipy = ModuleType('cupyx.scipy')
    cupyx.scipy.linalg = ModuleType('cupyx.scipy.linalg')
    cupyx.scipy.linalg.solve_triangular = lambda factor, right, lower: la.solve_triangular(
        factor, right, lower=lower, check_finite=False)
    modules = {'cupyx': cupyx, 'cupyx.scipy': cupyx.scipy, 'cupyx.scipy.linalg': cupyx.scipy.linalg}
    with patch.object(ritz, 'require_cupy', lambda: (cp, None)), patch.dict(sys.modules, modules):
        yield cp


def _estimated_at_the_join(events, given=None):
    """A ``condition`` of the device solve that starts nothing: the estimate is made when it is waited for."""
    def start(overlap):
        events.append('start')
        if given is not None:
            given.append(overlap)

        def wait():
            events.append('join')
            return ritz._device_overlap_condition(overlap)
        return wait
    return start


class HostGlueTests(unittest.TestCase):
    def test_blocked_mirror_is_the_sum_of_the_two_triangles(self):
        rng = np.random.default_rng(3)
        for block in (1, 7, ritz._MIRROR_BLOCK):
            with patch.object(ritz, '_MIRROR_BLOCK', block):
                for count in (1, 2, 63, 64, 65, 150):
                    host = rng.normal(size=(count, count))
                    expected = np.tril(host)+np.tril(host, -1).T
                    strided = rng.normal(size=(count, 2*count))
                    strided[:, ::2] = host
                    # Row-major, column-major and non-contiguous storage of the same matrix.
                    for matrix in (host, np.asfortranarray(host), strided[:, ::2]):
                        saved = matrix.copy()
                        actual = ritz._mirrored_lower(matrix)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertTrue(actual.flags.c_contiguous)
                        self.assertFalse(np.shares_memory(actual, matrix))
                        np.testing.assert_array_equal(matrix, saved)

    def test_orthogonality_error_is_the_largest_deviation_from_the_identity(self):
        rng = np.random.default_rng(5)
        overlap, _ = _gram_pair(40, 7)
        overlap = np.tril(overlap)+np.tril(overlap, -1).T
        exact = la.solve_triangular(np.linalg.cholesky(overlap).T, np.linalg.qr(rng.normal(size=(40, 40)))[0], lower=False)
        # Round-off only, then one changed coefficient whose sign decides the sign of the largest deviation.
        signs = set()
        for row, column, change in ((0, 0, 0.), (3, 3, 1e-3), (3, 3, -1e-3), (9, 4, 1e-3), (9, 4, -1e-3)):
            coefficients = exact.copy()
            coefficients[row, column] += change*np.abs(exact).max()
            saved = coefficients.copy()
            deviation = coefficients.T @ overlap @ coefficients-np.eye(40)
            signs.add(float(np.sign(deviation.flat[np.argmax(np.abs(deviation))])))
            self.assertEqual(ritz._orthogonality_error(coefficients, overlap), float(np.max(np.abs(deviation))))
            np.testing.assert_array_equal(coefficients, saved)
        self.assertEqual(signs, {-1., 1.})
        self.assertLess(ritz._orthogonality_error(exact, overlap), 1e-12)
        for bad in (np.nan, np.inf, -np.inf):
            coefficients = exact.copy()
            coefficients[5, 6] = bad
            with np.errstate(invalid='ignore'):
                self.assertFalse(np.isfinite(ritz._orthogonality_error(coefficients, overlap)))

    def test_host_solve_gives_the_coefficients_of_the_whole_matrix_form_bit_for_bit(self):
        for policy in ('svd', 'symmetric'):
            with patch.dict(os.environ, {**_HOST, 'PARSEC_CUPY_RITZ_CONDITION': policy}):
                # A condition number of 3e6 is past the symmetric screen, so the SVD decides there.
                for columns, condition in ((1, None), (2, None), (63, None), (65, 1e5), (150, None), (150, 3e6)):
                    overlap, projection = _gram_pair(columns, 11+columns, condition)
                    saved = overlap.copy(), projection.copy()
                    expected = _whole_matrix_solve(overlap, projection)
                    self.assertLess(expected[4], 5e-10)
                    for order in ('C', 'F'):
                        values, coefficients, whitened = ritz.solve_whitened_ritz(
                            np.array(overlap, order=order), np.array(projection, order=order))
                        np.testing.assert_array_equal(values, expected[0])
                        np.testing.assert_array_equal(coefficients, expected[1])
                        np.testing.assert_array_equal(np.tril(whitened), np.tril(expected[2]))
                    np.testing.assert_array_equal(overlap, saved[0])
                    np.testing.assert_array_equal(projection, saved[1])

    def test_host_solve_still_rejects_unsafe_overlaps(self):
        _, projection = _gram_pair(3, 13)
        with patch.dict(os.environ, _HOST):
            with self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, 'condition number'):
                ritz.solve_whitened_ritz(np.ones((3, 3)), projection)
            with self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, 'Cholesky'):
                ritz.solve_whitened_ritz(np.diag([1., -1., 2.]), projection)
            with self.assertRaises(ritz.GeneralizedRitzStabilityError):
                ritz.solve_whitened_ritz(np.full((3, 3), np.nan), projection)


@unittest.skipUnless(cupy_available(), 'CUDA required')
class DeviceEigensolverGlueTests(unittest.TestCase):
    def test_cusolver_reads_the_lower_triangle_of_the_whitened_matrix_only(self):
        import cupyx
        cp, _ = require_cupy()
        cp.cuda.Device(0).use()

        def symmetric_eigh(matrix):
            with cupyx.errstate(linalg='raise'):
                values, vectors = cp.linalg.eigh(cp.asarray(matrix))
            return cp.asnumpy(values), cp.asnumpy(vectors)

        with patch.dict(os.environ, {**_HOST, 'PARSEC_CUPY_RITZ_EIGH_BACKEND': 'cupy'}):
            for columns in (1, 7, 150, 700):
                overlap, projection = _gram_pair(columns, 17+columns)
                # The reference hands cuSOLVER the mirrored matrix, the solve its lower triangle.
                expected = _whole_matrix_solve(overlap, projection, eigh=symmetric_eigh)
                values, coefficients, whitened = ritz.solve_whitened_ritz(overlap, projection)
                np.testing.assert_array_equal(values, expected[0])
                np.testing.assert_array_equal(coefficients, expected[1])
                np.testing.assert_array_equal(np.tril(whitened), np.tril(expected[2]))


class DenseBackendSwitchTests(unittest.TestCase):
    def test_dense_backend_is_the_device_unless_the_host_is_named(self):
        for value, expected in (('host', False), (' Host ', False), ('device', True), (' Device ', True)):
            with patch.dict(os.environ, PARSEC_CUPY_RITZ_DENSE_BACKEND=value):
                self.assertIs(ritz.dense_solve_on_device(), expected)
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_RITZ_DENSE_BACKEND', None)
            self.assertIs(ritz.dense_solve_on_device(), True)
        for value in ('gpu', 'cupy', ''):
            with patch.dict(os.environ, PARSEC_CUPY_RITZ_DENSE_BACKEND=value):
                with self.assertRaises(ValueError):
                    ritz.dense_solve_on_device()

    def test_condition_policy_follows_the_place_of_the_solve_unless_named(self):
        # Unset, the device solve keeps the overlap on the device and the host solve keeps its SVD.
        with patch.dict(os.environ):
            os.environ.pop('PARSEC_CUPY_RITZ_CONDITION', None)
            self.assertEqual((ritz._condition_policy(on_device=True), ritz._condition_policy(on_device=False)),
                             ('symmetric', 'svd'))
            overlap = np.diag([1., 2., 4.])
            with patch('numpy.linalg.cond', wraps=np.linalg.cond) as svd:
                self.assertAlmostEqual(ritz._overlap_condition(overlap), 4., places=12)
            svd.assert_called_once()
        for policy in ('svd', 'symmetric'):
            with patch.dict(os.environ, PARSEC_CUPY_RITZ_CONDITION=policy):
                self.assertEqual((ritz._condition_policy(on_device=True), ritz._condition_policy(on_device=False)),
                                 (policy, policy))
        for value in ('cheap', 'SVD', ''):
            with patch.dict(os.environ, PARSEC_CUPY_RITZ_CONDITION=value):
                for on_device in (True, False):
                    with self.assertRaises(ValueError):
                        ritz._condition_policy(on_device=on_device)


class ConditionBesideTheSolveTests(unittest.TestCase):
    """The order of the device solve whose condition number is estimated elsewhere, followed on the host."""

    def test_the_estimate_starts_before_the_factorization_and_is_judged_after_the_eigensolve(self):
        for columns, condition in ((1, None), (2, None), (40, None), (65, 1e5), (150, 3e6)):
            overlap, projection = _gram_pair(columns, 23+columns, condition)
            saved = overlap.copy(), projection.copy()
            in_front, beside, given = [], [], []
            with patch.dict(os.environ, _DEVICE):
                with _numpy_as_device(in_front):
                    expected = ritz.solve_whitened_ritz_on_device(overlap, projection)
                with _numpy_as_device(beside):
                    actual = ritz.solve_whitened_ritz_on_device(overlap, projection,
                                                                condition=_estimated_at_the_join(beside, given))
            self.assertEqual(in_front, ['eigvalsh', 'cholesky', 'eigh'])
            # The overlap is complete on its stream before another device is given it.
            self.assertEqual(beside, ['synchronize', 'start', 'cholesky', 'eigh', 'join', 'eigvalsh'])
            # It is the mirrored lower triangle, in one piece: a raw copy of it is the same matrix in either order.
            mirrored, = given
            np.testing.assert_array_equal(mirrored, np.tril(overlap)+np.tril(overlap, -1).T)
            self.assertTrue(mirrored.flags.c_contiguous or mirrored.flags.f_contiguous)
            for mine, theirs in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(mine, theirs)
            np.testing.assert_array_equal(overlap, saved[0])
            np.testing.assert_array_equal(projection, saved[1])

    def test_an_estimate_on_another_thread_gives_the_same_coefficients_bit_for_bit(self):
        with ThreadPoolExecutor(max_workers=1) as helper, patch.dict(os.environ, _DEVICE), _numpy_as_device([]):
            for columns, condition in ((2, None), (65, 1e5), (150, 3e6), (400, None)):
                overlap, projection = _gram_pair(columns, 29+columns, condition)
                expected = ritz.solve_whitened_ritz_on_device(overlap, projection)
                threads = []

                def estimate(copy):
                    threads.append(threading.get_ident())
                    return ritz._device_overlap_condition(copy)

                actual = ritz.solve_whitened_ritz_on_device(
                    overlap, projection, condition=lambda mirrored: helper.submit(estimate, mirrored.copy()).result)
                self.assertEqual(len(threads), 1)
                self.assertNotEqual(threads[0], threading.get_ident())
                for mine, theirs in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(mine, theirs)

    def test_the_solve_raises_what_the_solve_with_the_estimate_in_front_raises(self):
        _, projection = _gram_pair(3, 13)
        broken = np.eye(3)
        broken[2, 1] = np.nan

        def no_spectrum(_overlap):
            raise np.linalg.LinAlgError('no spectrum')

        unsafe = (
            # An overlap that is unsafe and has no factor either: the number is the first thing reported.
            (np.ones((3, 3)), projection, '1e8', None, 'condition number'),
            (np.diag([1., -1., 2.]), projection, '1e8', None, 'Cholesky'),
            (np.full((3, 3), np.nan), projection, '1e8', None, 'condition'),
            # The whole solve succeeds before this number is judged.
            (np.diag([1., 2., 4.]), projection, '1.0000001', None, 'condition number'),
            (np.diag([1., 2., 4.]), broken, '1e8', None, 'Cholesky/Ritz solve failed'),
            (*_gram_pair(30, 5, 1e12), '1e300', None, 'orthogonality audit'),
            # An estimate that fails, of an overlap with a factor and of one without.
            (np.diag([1., 2., 4.]), projection, '1e8', no_spectrum, 'condition estimate failed'),
            (np.diag([1., -1., 2.]), projection, '1e8', no_spectrum, 'condition estimate failed'),
        )
        for overlap, projected, limit, estimate, message in unsafe:
            raised = []
            for beside in (False, True):
                events = []
                with patch.dict(os.environ, {**_DEVICE, 'PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX': limit}), \
                     _numpy_as_device(events), np.errstate(invalid='ignore'), contextlib.ExitStack() as stack:
                    if estimate is not None:
                        stack.enter_context(patch.object(ritz, '_device_overlap_condition', estimate))
                    options = dict(condition=_estimated_at_the_join(events)) if beside else {}
                    with self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, message) as caught:
                        ritz.solve_whitened_ritz_on_device(overlap, projected, **options)
                raised.append((str(caught.exception), type(caught.exception.__cause__)))
                # Whatever was raised, the estimate was waited for once: nothing reads the overlap after it.
                self.assertEqual(events.count('join'), int(beside))
            self.assertEqual(raised[1], raised[0])

    def test_the_estimate_is_waited_for_whatever_the_solve_raises(self):
        overlap, projection = _gram_pair(9, 3)

        def no_memory(_whitened):
            raise MemoryError('eigensolver work array')

        # An error that is no stability failure passes, unless the number is unsafe: that is reported first.
        for limit, expected in (('1e8', MemoryError), ('1.0000001', ritz.GeneralizedRitzStabilityError)):
            events = []
            with patch.dict(os.environ, {**_DEVICE, 'PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX': limit}), \
                 _numpy_as_device(events) as cp:
                cp.linalg.eigh = no_memory
                with self.assertRaises(expected):
                    ritz.solve_whitened_ritz_on_device(overlap, projection, condition=_estimated_at_the_join(events))
            self.assertEqual(events, ['synchronize', 'start', 'cholesky', 'join', 'eigvalsh'])

    def test_an_estimate_that_failed_on_another_thread_holds_nothing_once_its_error_is_handled(self):
        # The fallback gathers the basis on the owner as soon as the handler of the stability error has ended,
        # and a graph capture keeps the cyclic collector off: what the failed solve held has to go with its
        # error.  The future of the other thread holds the error of the estimate, whose traceback holds the
        # frames of the solve and of its callers: none of them may still hold the future.
        class Held(np.ndarray):
            """An array that can be referred to weakly."""

        def step(helper, failure, alive):
            # Stands for the Ritz step, a caller of the solve: its column blocks and its Gram pair.
            columns = np.zeros(8).view(Held)
            alive['column blocks of the step'] = weakref.ref(columns)
            overlap, projection = _gram_pair(6, 11)
            overlap = overlap.view(Held)
            alive['overlap of the owner'] = weakref.ref(overlap)

            def estimate(copy):
                copy = copy.view(Held)
                alive['copy of the helper'] = weakref.ref(copy)
                raise failure('no number')

            # The wait of a future, as a sector group hands it over.
            return ritz.solve_whitened_ritz_on_device(
                overlap, projection, condition=lambda mirrored: helper.submit(estimate, mirrored.copy()).result)

        collecting = gc.isenabled()
        with ThreadPoolExecutor(max_workers=1) as helper, patch.dict(os.environ, _DEVICE), _numpy_as_device([]):
            # A failed estimate is a stability error; an error of another kind passes as it is.
            kinds = ((np.linalg.LinAlgError, ritz.GeneralizedRitzStabilityError), (MemoryError, MemoryError))
            for failure, raised in kinds:
                alive, caught = {}, None
                gc.collect()
                gc.disable()
                try:
                    # No assertRaises: it clears the frames of the traceback, which would hide what they hold.
                    try:
                        step(helper, failure, alive)
                    except raised as error:
                        caught = type(error)
                    held = [name for name, array in alive.items() if array() is not None]
                finally:
                    if collecting:
                        gc.enable()
                self.assertEqual((caught, len(alive), held), (raised, 3, []))


@unittest.skipUnless(cupy_available(), 'CUDA required')
class DeviceSolveTests(unittest.TestCase):
    rows = 257

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()

    def tearDown(self):
        # Solvers and filter graphs that a test left in a reference cycle are destroyed here, between the tests,
        # and not by a collection inside the next one.
        gc.collect()

    def test_device_solve_matches_the_host_solve_and_the_dense_generalized_problem(self):
        cp = self.cp
        for columns, condition in ((1, None), (2, None), (40, None), (65, 1e5), (150, 3e6)):
            overlap, projection = _gram_pair(columns, 23+columns, condition)
            full_overlap = np.tril(overlap)+np.tril(overlap, -1).T
            full_projection = np.tril(projection)+np.tril(projection, -1).T
            # Round-off of the whitened problem grows with the condition number of the overlap.
            growth = max(1., condition or 1.)
            with patch.dict(os.environ, _HOST):
                host = ritz.solve_whitened_ritz(overlap, projection)
            scale = np.abs(host[0]).max()
            for order in ('C', 'F'):
                with patch.dict(os.environ, _DEVICE), patch('numpy.linalg.cond', wraps=np.linalg.cond) as svd:
                    values, coefficients, whitened = ritz.solve_whitened_ritz_on_device(
                        cp.array(overlap, order=order), cp.array(projection, order=order))
                # The symmetric screen decides far below the guard, the host SVD from 1% of it on.
                self.assertEqual(svd.call_count, int(growth >= 1e6))
                self.assertTrue(coefficients.flags.f_contiguous)
                values, coefficients, whitened = map(cp.asnumpy, (values, coefficients, whitened))
                np.testing.assert_allclose(values, host[0], rtol=0, atol=1e-11*scale*growth)
                np.testing.assert_allclose(values, la.eigh(full_projection, full_overlap, eigvals_only=True),
                                           rtol=0, atol=1e-11*scale*growth)
                np.testing.assert_allclose(coefficients.T @ full_overlap @ coefficients, np.eye(columns), atol=5e-9)
                np.testing.assert_allclose(coefficients.T @ full_projection @ coefficients, np.diag(values),
                                           rtol=0, atol=1e-11*scale*growth)
                np.testing.assert_allclose(np.tril(whitened), np.tril(host[2]), rtol=0,
                                           atol=1e-11*growth*np.abs(host[2]).max())

    def test_device_solve_rejects_what_the_host_solve_rejects(self):
        cp = self.cp
        _, projection = _gram_pair(3, 13)
        broken = np.eye(3)
        broken[2, 1] = np.nan
        unsafe = (
            (np.ones((3, 3)), projection, '1e8', 'condition number'),
            # Passes the condition screen; only the status word of the factorization reports it.
            (np.diag([1., -1., 2.]), projection, '1e8', 'Cholesky'),
            (np.full((3, 3), np.nan), projection, '1e8', 'condition'),
            (np.diag([1., 2., 4.]), projection, '1.0000001', 'condition number'),
            (np.diag([1., 2., 4.]), broken, '1e8', ''),
            # Factorizes once the condition limit is lifted, but the coefficients are not orthonormal.
            (*_gram_pair(30, 5, 1e12), '1e300', 'orthogonality audit'),
        )
        for overlap, projected, limit, message in unsafe:
            guard = dict(PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX=limit)
            with patch.dict(os.environ, {**_HOST, **guard}), np.errstate(invalid='ignore'):
                with self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, message):
                    ritz.solve_whitened_ritz(overlap, projected)
            with patch.dict(os.environ, {**_DEVICE, **guard}):
                with self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, message):
                    ritz.solve_whitened_ritz_on_device(cp.asarray(overlap), cp.asarray(projected))

    def test_device_solve_keeps_the_condition_policy(self):
        cp = self.cp
        overlap, projection = _gram_pair(9, 3)
        # Unset, the overlap stays on the device: no host SVD in a pass.
        for policy, spectra, decompositions in (('symmetric', 1, 0), ('svd', 0, 1), (None, 1, 0)):
            with patch.dict(os.environ, _DEVICE):
                os.environ.pop('PARSEC_CUPY_RITZ_CONDITION')
                if policy is not None:
                    os.environ['PARSEC_CUPY_RITZ_CONDITION'] = policy
                with patch.object(cp.linalg, 'eigvalsh', wraps=cp.linalg.eigvalsh) as spectrum, \
                     patch('numpy.linalg.cond', wraps=np.linalg.cond) as svd:
                    ritz.solve_whitened_ritz_on_device(cp.asarray(overlap), cp.asarray(projection))
            self.assertEqual((spectrum.call_count, svd.call_count), (spectra, decompositions))
        with patch.dict(os.environ, {**_DEVICE, 'PARSEC_CUPY_RITZ_CONDITION': 'cheap'}):
            with self.assertRaises(ValueError):
                ritz.solve_whitened_ritz_on_device(cp.asarray(overlap), cp.asarray(projection))

    def _beside(self, threads):
        """A ``condition`` that estimates on another thread and stream of this device, from a copy of the overlap."""
        cp = self.cp
        # The thread comes to hold a cuSOLVER handle; it ends with the test, when nothing captures a graph.
        helper = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(helper.shutdown)

        def estimate(overlap):
            threads.append(threading.get_ident())
            with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True) as stream:
                copy = cp.empty(overlap.shape, dtype=cp.float64)
                copy.data.copy_from_device_async(overlap.data, overlap.nbytes, stream)
                return ritz._device_overlap_condition(copy)

        return lambda overlap: helper.submit(estimate, overlap).result

    def test_the_condition_number_estimated_beside_the_device_solve_leaves_every_bit_of_it(self):
        cp, threads = self.cp, []
        beside = self._beside(threads)
        cases = ((1, None), (2, None), (40, None), (65, 1e5), (150, 3e6), (700, None))
        for columns, condition in cases:
            overlap, projection = _gram_pair(columns, 23+columns, condition)
            for order in ('C', 'F'):
                with patch.dict(os.environ, _DEVICE), patch('numpy.linalg.cond', wraps=np.linalg.cond) as svd:
                    expected = ritz.solve_whitened_ritz_on_device(
                        cp.array(overlap, order=order), cp.array(projection, order=order))
                    actual = ritz.solve_whitened_ritz_on_device(
                        cp.array(overlap, order=order), cp.array(projection, order=order), condition=beside)
                # The helper screens and, from 1% of the guard on, downloads its copy for the SVD as the owner does.
                self.assertEqual(svd.call_count, 2*int((condition or 1.) >= 1e6))
                for mine, theirs in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(cp.asnumpy(mine), cp.asnumpy(theirs))
        self.assertEqual(len(threads), 2*len(cases))
        self.assertNotIn(threading.get_ident(), threads)

    def test_the_condition_number_estimated_beside_the_device_solve_rejects_what_it_rejects_in_front(self):
        cp = self.cp
        beside = self._beside([])
        _, projection = _gram_pair(3, 13)
        broken = np.eye(3)
        broken[2, 1] = np.nan
        unsafe = (
            # Unsafe and without a factor: the number is still the first thing reported.
            (np.ones((3, 3)), projection, '1e8', 'condition number'),
            (np.diag([1., -1., 2.]), projection, '1e8', 'Cholesky'),
            # cuSOLVER is now given an overlap that is not finite for the factorization, and completes it.
            (np.full((3, 3), np.nan), projection, '1e8', 'condition'),
            (np.diag([1., np.inf, 4.]), projection, '1e8', 'condition'),
            (np.diag([1., 2., 4.]), projection, '1.0000001', 'condition number'),
            (np.diag([1., 2., 4.]), broken, '1e8', ''),
            (*_gram_pair(30, 5, 1e12), '1e300', 'orthogonality audit'),
        )
        for overlap, projected, limit, message in unsafe:
            raised = []
            for options in ({}, dict(condition=beside)):
                with patch.dict(os.environ, {**_DEVICE, 'PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX': limit}):
                    with self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, message) as caught:
                        ritz.solve_whitened_ritz_on_device(cp.asarray(overlap), cp.asarray(projected), **options)
                raised.append((str(caught.exception), type(caught.exception.__cause__)))
            self.assertEqual(raised[1], raised[0])

    def test_the_estimate_of_a_sector_group_runs_on_the_thread_of_its_device_beside_the_solve(self):
        # A group of real devices needs two of them.  One device can stand for both here: the thread and the
        # stream that a group gives it copy the overlap and estimate while the calling thread solves.
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp = self.cp
        group = object.__new__(module.SectorDeviceGroup)
        group._worker = SimpleNamespace(streams={0: cp.cuda.Stream(non_blocking=True)})
        group.condition_seconds = 0.
        threads = []

        def estimate(copy):
            threads.append(threading.current_thread().name)
            return ritz._device_overlap_condition(copy)

        for columns, condition in ((2, None), (65, 1e5), (150, 3e6), (700, None)):
            overlap, projection = _gram_pair(columns, 31+columns, condition)
            spent = group.condition_seconds
            with patch.dict(os.environ, _DEVICE), patch.object(module, '_device_overlap_condition', estimate):
                expected = ritz.solve_whitened_ritz_on_device(cp.asarray(overlap), cp.asarray(projection))
                actual = ritz.solve_whitened_ritz_on_device(cp.asarray(overlap), cp.asarray(projection),
                                                            condition=group._condition_beside(0))
            self.assertGreater(group.condition_seconds, spent)
            for mine, theirs in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(cp.asnumpy(mine), cp.asnumpy(theirs))
        self.assertEqual(threads, ['orbital-shard-0_0']*4)

    def test_the_estimate_of_a_sector_group_takes_its_copy_of_the_overlap_and_what_the_spectrum_takes(self):
        # What the helper of a sector takes on its device, in a pool of its own: the copy of the overlap, the
        # screen for entries that are not finite and what the symmetric spectrum takes of that copy.  cuSOLVER
        # sizes the work array of the spectrum itself, so that part is measured: the same spectrum alone.
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp = self.cp
        stream = cp.cuda.Stream(non_blocking=True)
        group = object.__new__(module.SectorDeviceGroup)
        group._worker = SimpleNamespace(streams={0: stream})
        group.condition_seconds = 0.
        # An array must not outlive the pool it was taken from.
        pools = []

        def taken(function):
            """Bytes that ``function`` takes from a pool of its own, and what it returns."""
            pool, previous = cp.cuda.MemoryPool(), cp.cuda.get_allocator()
            pools.append(pool)
            cp.cuda.set_allocator(pool.malloc)
            try:
                result = function()
            finally:
                cp.cuda.set_allocator(previous)
            return pool.total_bytes(), result

        def spectrum(overlap):
            with cp.cuda.Device(0), stream:
                cp.linalg.eigvalsh(overlap)

        try:
            for columns in (150, 700):
                overlap, _ = _gram_pair(columns, 37+columns)
                mirrored = cp.asarray(np.tril(overlap)+np.tril(overlap, -1).T)
                with patch.dict(os.environ, _DEVICE):
                    alone, _ = taken(lambda: module._pool(0).submit(spectrum, mirrored).result())
                    beside, number = taken(lambda: group._condition_beside(0)(mirrored)())
                # Far below the guard: the number is that of the spectrum on the device, and no host SVD followed.
                self.assertLess(number, 1e6)
                one = 8*columns*columns
                # The spectrum takes a copy that cuSOLVER overwrites and its work array.
                self.assertGreater(alone, one)
                # Besides, the helper holds its copy of the overlap and, for a moment, a byte per entry of the
                # screen and its verdict; the pool rounds every block up to 512 bytes.
                self.assertGreaterEqual(beside, alone+one)
                self.assertLessEqual(beside, alone+one+columns*columns+8*512)
                del mirrored
        finally:
            module._pool(0).submit(lambda: None).result()
            gc.collect()
            for pool in pools:
                pool.free_all_blocks()

    def test_generalized_ritz_with_the_device_solve_matches_the_host_solve_and_orthonormal_ritz(self):
        from parsec_python.acceleration.Eigensolvers import orthonormalize
        cp, op = self.cp, _operator(self.rows)
        host = np.random.default_rng(82).normal(size=(self.rows, 17))
        reference = ritz.rayleigh_ritz(op, orthonormalize(cp.asarray(host)).basis, compute_residuals=True)
        expected, vectors = cp.asnumpy(reference.eigenvalues), cp.asnumpy(reference.wavefunctions)
        # Two tall arrays (kept, consumed, with residuals), then the single array rotated in place.
        routes = (('0', {}), ('0', dict(consume_basis=True)), ('0', dict(compute_residuals=True)),
                  ('1', dict(consume_basis=True)))
        for rotation in ('allocate', 'reuse'):
            for streaming, options in routes:
                route = dict(PARSEC_CUPY_RITZ_ROTATION=rotation, PARSEC_CUPY_STREAMING_RITZ=streaming,
                             PARSEC_CUPY_STREAMING_RITZ_BYTES=str(8*self.rows*5))
                results = []
                for backend in (_HOST, _DEVICE):
                    with patch.dict(os.environ, {**backend, **route}):
                        with patch.object(ritz, 'solve_whitened_ritz_on_device',
                                          wraps=ritz.solve_whitened_ritz_on_device) as solve:
                            results.append(ritz.generalized_rayleigh_ritz(op, cp.array(host, order='F'), **options))
                    self.assertEqual(solve.call_count, int(backend is _DEVICE))
                on_host, on_device = results
                self.assertEqual(on_device.algorithm, on_host.algorithm)
                self.assertIsInstance(on_device.projected_hamiltonian, cp.ndarray)
                values = cp.asnumpy(on_device.eigenvalues)
                np.testing.assert_allclose(values, cp.asnumpy(on_host.eigenvalues), rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(values, expected, rtol=2e-11, atol=2e-11)
                rotated = cp.asnumpy(on_device.wavefunctions)
                np.testing.assert_allclose(np.abs(vectors.T @ rotated), np.eye(17), rtol=2e-10, atol=2e-10)
                if options.get('compute_residuals'):
                    np.testing.assert_allclose(cp.asnumpy(on_device.residual_norms), cp.asnumpy(reference.residual_norms),
                                               rtol=2e-9, atol=2e-11)

    def test_later_passes_with_the_device_solve_match_the_host_solve(self):
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings, run_subspace_filter
        cp, op = self.cp, _operator(self.rows)
        start = np.linalg.qr(np.random.default_rng(19).normal(size=(self.rows, 17)))[0]
        settings = SubspaceSettings(polynomial_degree=3, degree_delta=1)
        # 5 columns per slab and 75 rows per tile exercise several slabs and tiles of the single-array route.
        common = {**_ROUTES, 'PARSEC_CUPY_STREAMING_RITZ_BYTES': str(8*self.rows*5)}
        for streaming in ('0', '1'):
            common['PARSEC_CUPY_STREAMING_RITZ'] = streaming
            reference = DeviceSubspaceState(self.rows, 17, cp.linspace(.1, 1.5, 17), cp.array(start, order='F'))
            tested = replace(reference, vectors=cp.array(start, order='F'))
            for _ in range(2):
                with patch.dict(os.environ, {**common, **_HOST}):
                    a = run_subspace_filter(op, reference, settings=settings, compute_residuals=False, consume_state=True)
                with patch.dict(os.environ, {**common, **_DEVICE}):
                    b = run_subspace_filter(op, tested, settings=settings, compute_residuals=False, consume_state=True)
                self.assertEqual(b.rayleigh_ritz.algorithm, a.rayleigh_ritz.algorithm)
                self.assertFalse(b.state.generalized_ritz_failed)
                np.testing.assert_allclose(cp.asnumpy(b.eigenvalues), cp.asnumpy(a.eigenvalues), rtol=0, atol=5e-12)
                u, v = cp.asnumpy(a.vectors), cp.asnumpy(b.vectors)
                np.testing.assert_allclose(v.T @ v, np.eye(17), atol=1e-9)
                np.testing.assert_allclose(np.linalg.svd(u.T @ v, compute_uv=False), np.ones(17), atol=5e-9)
                reference, tested = a.state, b.state

    def test_first_solve_with_the_device_solve_matches_the_host_solve(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        run_chebff = importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff').run_chebff
        cp, op = self.cp, _operator(self.rows)
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        with patch.dict(os.environ, {**_ROUTES, **_HOST}):
            expected = run_chebff(op, 19, settings=settings)
        with patch.dict(os.environ, {**_ROUTES, **_DEVICE}):
            with patch.object(ritz, 'solve_whitened_ritz_on_device', wraps=ritz.solve_whitened_ritz_on_device) as solve:
                actual = run_chebff(op, 19, settings=settings)
        # One small solve per filter cycle, none of them on the host.
        self.assertEqual(solve.call_count, 4)
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=5e-11)
        a, b = cp.asnumpy(actual.vectors), cp.asnumpy(expected.vectors)
        np.testing.assert_allclose(a.T @ a, np.eye(19), atol=1e-9)
        np.testing.assert_allclose(np.linalg.svd(a.T @ b, compute_uv=False), np.ones(19), atol=5e-9)


@unittest.skipUnless(_device_count() >= 2, 'at least two CUDA devices required')
class SharedBasisDenseTests(unittest.TestCase):
    rows = 257
    columns = 23

    def setUp(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import uniform_filter_blocks
        from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _operator(self.rows)
        self.devices = tuple(range(min(4, _device_count())))
        self.op.distributed_filter_devices = self.devices
        self.group = SectorDeviceGroup(self.op, self.devices)
        self.blocks = uniform_filter_blocks(self.columns, 6, 5)
        self.ranges = self.group.column_ranges(self.blocks)

    def tearDown(self):
        # Groups, operators and filter graphs that a test left in a reference cycle are destroyed here, between
        # the tests, and not by a collection inside the next one.
        gc.collect()

    def _filtered(self):
        """Host copies of a filtered basis and of ``H`` times it."""
        cp, group = self.cp, self.group
        start = np.linalg.qr(np.random.default_rng(37).normal(size=(self.rows, self.columns)))[0]
        with patch.dict(os.environ, PARSEC_CUPY_MIXED_FILTER='off', PARSEC_CUPY_DISTRIBUTED_STATE_RANGES='fixed'):
            # The replicas of the other devices receive the effective potential in this step.
            filtered = group.filter(group.scatter(cp.array(start, order='F'), self.ranges), self.blocks, 1.2, 6., 1.2, False)
        basis = cp.asnumpy(group.gather(filtered))
        return basis, cp.asnumpy(self.op @ cp.array(basis, order='F'))

    def test_small_solve_receives_the_gram_matrices_summed_over_the_devices(self):
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp, group = self.cp, self.group
        basis, applied = self._filtered()
        overlap, projection = basis.T @ basis, basis.T @ applied
        # The Ritz pairs must be those of the dense generalized problem of the same basis.
        expected, coefficients = la.eigh((projection+projection.T)/2, (overlap+overlap.T)/2)
        summed = {}
        for backend, solver in ((_HOST, 'solve_whitened_ritz'), (_DEVICE, 'solve_whitened_ritz_on_device')):
            on_device = backend is _DEVICE
            columns = group.scatter(cp.array(basis, order='F'), self.ranges)
            pointers = [int(block.data.ptr) for block in columns.blocks]
            with patch.dict(os.environ, backend):
                with patch.object(module, solver, wraps=getattr(ritz, solver)) as solve:
                    values, rotated, whitened = group.ritz(self.op, columns, stages=False)
            solve.assert_called_once()
            summed[on_device] = [np.tril(cp.asnumpy(part) if on_device else part) for part in solve.call_args.args]
            for actual, reference in zip(summed[on_device], (overlap, projection)):
                np.testing.assert_allclose(actual, np.tril(reference), rtol=0, atol=1e-12*np.abs(reference).max())
            # Host Ritz values either way; the whitened matrix stays where the solve ran.
            self.assertIsInstance(values, np.ndarray)
            self.assertEqual(isinstance(whitened, cp.ndarray), on_device)
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9)
            self.assertEqual([int(block.data.ptr) for block in rotated.blocks], pointers)
            vectors = cp.asnumpy(group.gather(rotated))
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(self.columns), atol=1e-9)
            np.testing.assert_allclose(np.linalg.svd((basis @ coefficients).T @ vectors, compute_uv=False),
                                       np.ones(self.columns), atol=5e-9)
        # The owner adds the parts of the devices in the order of the host sum.
        for on_host, on_owner in zip(summed[False], summed[True]):
            np.testing.assert_array_equal(on_owner, on_host)

    def test_another_device_estimates_the_condition_number_and_the_step_keeps_every_bit(self):
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        cp, group = self.cp, self.group
        basis, _applied = self._filtered()
        owner = self.devices[0]
        results = []
        for policy, hartree_device, expected in _helper_cases(self.devices):
            self.op.hartree_device = hartree_device
            columns = group.scatter(cp.array(basis, order='F'), self.ranges)
            spent = group.condition_seconds
            with patch.dict(os.environ, {**_DEVICE, 'PARSEC_CUPY_RITZ_CONDITION_HELPER': policy}):
                with patch.object(module, '_device_overlap_condition', wraps=ritz._device_overlap_condition) as beside, \
                     patch.object(ritz, '_device_overlap_condition', wraps=ritz._device_overlap_condition) as in_front:
                    values, rotated, whitened = group.ritz(self.op, columns, stages=False)
            helped = expected != owner
            self.assertEqual((beside.call_count, in_front.call_count), (int(helped), int(not helped)), policy)
            self.assertEqual(group.condition_device, expected)
            self.assertEqual(group.condition_seconds > spent, helped)
            if helped:
                # The helper estimates from its own copy of the overlap.
                self.assertEqual(int(beside.call_args.args[0].device.id), expected)
            results.append((values, cp.asnumpy(group.gather(rotated)), cp.asnumpy(whitened)))
        # Where the number is estimated changes no bit of the step.
        for result in results[1:]:
            for mine, theirs in zip(result, results[0], strict=True):
                np.testing.assert_array_equal(mine, theirs)

    def test_later_passes_with_the_device_solve_match_one_device_with_the_host_solve(self):
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_subspace
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings, run_subspace_filter
        cp, group = self.cp, self.group
        start = np.linalg.qr(np.random.default_rng(17).normal(size=(self.rows, 23)))[0]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        single = DeviceSubspaceState(self.rows, 23, cp.linspace(.1, 1.5, 23), cp.array(start, order='F'))
        shared = replace(single, vectors=group.scatter(cp.array(start, order='F'), self.ranges))
        for working in (23, 17):
            if working != single.working_states:
                single = replace(single, working_states=working, eigenvalues=single.eigenvalues[:working],
                                 vectors=single.vectors[:, :working])
                shared = replace(shared, working_states=working, eigenvalues=shared.eigenvalues[:working],
                                 vectors=shared.vectors[:, :working])
            with patch.dict(os.environ, {**_ROUTES, **_HOST}):
                a = run_subspace_filter(self.op, single, settings=settings, compute_residuals=False)
            with patch.dict(os.environ, {**_ROUTES, **_DEVICE}):
                b = run_distributed_subspace(self.op, shared, settings=settings, compute_residuals=False, group=group)
            self.assertFalse(b.state.generalized_ritz_failed)
            self.assertEqual(b.rayleigh_ritz.algorithm, 'distributed_generalized_cholesky_rayleigh_ritz')
            np.testing.assert_allclose(cp.asnumpy(b.eigenvalues), cp.asnumpy(a.eigenvalues), rtol=0, atol=5e-12)
            left, right = cp.asnumpy(a.vectors), cp.asnumpy(group.gather(b.vectors))
            np.testing.assert_allclose(right.T @ right, np.eye(working), atol=1e-9)
            np.testing.assert_allclose(np.linalg.svd(left.T @ right, compute_uv=False), np.ones(working), atol=5e-9)
            single, shared = a.state, b.state
        self.assertEqual(group.passes, 2)
        self.assertGreater(group.seconds['dense'], 0.)

    def test_first_solve_with_the_device_solve_matches_one_device_with_the_host_solve(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        from parsec_python.acceleration.Eigensolvers.distributed_state import run_distributed_chebff
        run_chebff = importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff').run_chebff
        cp, group = self.cp, self.group
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        with patch.dict(os.environ, {**_ROUTES, **_HOST}, PARSEC_CUPY_DEVICE_RANDOM='1'):
            expected = run_chebff(self.op, 19, settings=settings)
        with patch.dict(os.environ, {**_ROUTES, **_DEVICE}, PARSEC_CUPY_DEVICE_RANDOM='1'):
            actual = run_distributed_chebff(self.op, 19, settings=settings, group=group)
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=5e-11)
        left, right = cp.asnumpy(expected.vectors), cp.asnumpy(group.gather(actual.vectors))
        np.testing.assert_allclose(np.linalg.svd(left.T @ right, compute_uv=False), np.ones(19), atol=5e-9)
        for mine, theirs in zip(actual.cycles, expected.cycles, strict=True):
            self.assertAlmostEqual(mine.lower_bound_out, theirs.lower_bound_out, delta=1e-9)

    def test_device_solve_that_rejects_the_overlap_falls_back_to_the_orthonormal_route(self):
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, SubspaceSettings, run_subspace_filter
        run_distributed_subspace = module.run_distributed_subspace
        cp, group = self.cp, self.group
        start = np.linalg.qr(np.random.default_rng(23).normal(size=(self.rows, 17)))[0]
        settings = SubspaceSettings(polynomial_degree=5, degree_delta=1)
        single = DeviceSubspaceState(self.rows, 17, cp.linspace(.1, 1.5, 17), cp.array(start, order='F'))
        shared = replace(single, vectors=group.scatter(cp.array(start, order='F'), ((0, 17),)+((17, 17),)*(len(self.devices)-1)))
        # No filtered basis is this well conditioned, so the condition audit itself rejects it on both routes.
        tight = dict(PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1.0000001')
        with patch.dict(os.environ, {**_ROUTES, **_HOST, **tight}):
            expected = run_subspace_filter(self.op, single, settings=settings, compute_residuals=False)
        with patch.dict(os.environ, {**_ROUTES, **_DEVICE, **tight}):
            with patch.object(module, 'solve_whitened_ritz_on_device', wraps=ritz.solve_whitened_ritz_on_device) as solve:
                actual = run_distributed_subspace(self.op, shared, settings=settings, compute_residuals=False, group=group)
        solve.assert_called_once()
        self.assertTrue(expected.state.generalized_ritz_failed)
        self.assertTrue(actual.state.generalized_ritz_failed)
        self.assertEqual(group.passes, 0)
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=5e-11)
        left, right = cp.asnumpy(expected.vectors), cp.asnumpy(group.gather(actual.vectors))
        np.testing.assert_allclose(np.linalg.svd(left.T @ right, compute_uv=False), np.ones(17), atol=5e-9)

    def test_first_solve_whose_device_solve_rejects_the_overlap_falls_back_in_every_cycle(self):
        from parsec_python.Eigensolvers.chebff import ChebFFSettings
        module = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')
        run_chebff = importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff').run_chebff
        cp, group = self.cp, self.group
        settings = ChebFFSettings(polynomial_degree=12, filter_cycles=4)
        # No filtered basis is this well conditioned, so the condition audit rejects every cycle on both routes.
        tight = dict(PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1.0000001')
        with patch.dict(os.environ, {**_ROUTES, **_HOST, **tight}, PARSEC_CUPY_DEVICE_RANDOM='1'):
            expected = run_chebff(self.op, 19, settings=settings)
        with patch.dict(os.environ, {**_ROUTES, **_DEVICE, **tight}, PARSEC_CUPY_DEVICE_RANDOM='1'):
            with patch.object(module, 'solve_whitened_ritz_on_device', wraps=ritz.solve_whitened_ritz_on_device) as solve:
                actual = module.run_distributed_chebff(self.op, 19, settings=settings, group=group)
        # Each cycle tries the device solve again and then orthonormalizes on the owner.
        self.assertEqual(solve.call_count, 4)
        self.assertEqual(group.passes, 0)
        self.assertEqual([cycle.number for cycle in actual.cycles], [1, 2, 3, 4])
        np.testing.assert_allclose(cp.asnumpy(actual.eigenvalues), cp.asnumpy(expected.eigenvalues), rtol=0, atol=5e-11)
        left, right = cp.asnumpy(expected.vectors), cp.asnumpy(group.gather(actual.vectors))
        np.testing.assert_allclose(np.linalg.svd(left.T @ right, compute_uv=False), np.ones(19), atol=5e-9)


if __name__ == '__main__':
    unittest.main()
