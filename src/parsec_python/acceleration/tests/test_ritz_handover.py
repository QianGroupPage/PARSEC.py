"""The triangular hand-over of the Ritz step (PARSEC_CUPY_RITZ_HANDOVER).

Under ``rotation``, the default, every pass hands the next filter its Ritz
vectors, bit for bit as before.  Under ``triangular`` a later pass hands over
``Q = X L^-T`` and keeps the eigenvectors ``Z`` that make its Ritz vectors
``Q Z``; the density and the forces of such a basis are those of all its
columns less the Ritz vectors whose occupation is not one, and a full rotation
is formed where Ritz vectors themselves are read.  For a fixed basis the
hand-over must give the density, the forces and the Ritz values of the
rotation to round-off; over passes the path may differ beyond round-off.
"""
import contextlib
import gc
import importlib
import os
import pickle
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.Eigensolvers import handover

# The package also exports functions named like these submodules.
ritz = importlib.import_module('parsec_python.acceleration.Eigensolvers.rayleigh_ritz')
subspace = importlib.import_module('parsec_python.acceleration.Eigensolvers.subspace')
chebff = importlib.import_module('parsec_python.acceleration.Eigensolvers.chebff')
distributed_state = importlib.import_module('parsec_python.acceleration.Eigensolvers.distributed_state')

_NAMES = ('PARSEC_CUPY_RITZ_HANDOVER', 'PARSEC_CUPY_RITZ_HANDOVER_COLUMNS', 'PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION',
          'PARSEC_CUPY_RITZ_HANDOVER_CORRECTION_SHARE', 'PARSEC_CUPY_RITZ_HANDOVER_PRODUCT')
# The routes of these tests whatever the calling shell exports: the generalized step on one streamed array, the
# small solve on the device with the measured thresholds, filters without graphs on one device.
_ROUTE = dict(PARSEC_CUPY_GENERALIZED_RITZ='on', PARSEC_CUPY_STREAMING_RITZ='1', PARSEC_CUPY_MIXED_FILTER='off',
              PARSEC_CUPY_DISTRIBUTED_FILTER='0', PARSEC_CUPY_FILTER_GRAPHS='1', PARSEC_CUPY_FILTER_COLUMN_MAJOR='1',
              PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ='1', PARSEC_CUPY_RITZ_DENSE_BACKEND='device',
              PARSEC_CUPY_RITZ_CONDITION='symmetric', PARSEC_CUPY_RITZ_EIGH_BACKEND='host',
              PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8', PARSEC_CUPY_DEVICE_RANDOM='1')


def _device_count():
    if not cupy_available():
        return 0
    cp, _ = require_cupy()
    return int(cp.cuda.runtime.getDeviceCount())


def _clean(**environment):
    """``patch.dict`` of the environment without any setting of the hand-over, then ``environment``."""
    patcher = patch.dict(os.environ)

    @contextlib.contextmanager
    def context():
        with patcher:
            for name in _NAMES:
                os.environ.pop(name, None)
            os.environ.update(environment)
            yield
    return context()


def _gram_pair(columns, seed, condition=None):
    """``X.T X`` and ``X.T H X`` of a random basis, with nothing to read above the diagonals."""
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


def _operator(rows, seed=8):
    """A small Hamiltonian with two projectors on the current device."""
    return CuPyHamiltonian(sp.diags((-np.ones(rows-1), 2.2*np.ones(rows), -np.ones(rows-1)), (-1, 0, 1)),
                           np.linspace(-.1, .1, rows),
                           (sp.csr_matrix(np.random.default_rng(seed).normal(size=(rows, 2))*.002), np.array([1., -1.])),
                           retain_generic_laplacian=False)


def _occupations(kind, count):
    """Occupations of ``count`` selected Ritz vectors: Fermi-like, a frontier group at T = 0, holes, all full, none."""
    f = np.ones(count)
    if kind == 'fermi':
        f[-4:] = (.999999999, .73, .2, 1e-13)
    elif kind == 'frontier':
        f[-3:] = 2./3.
    elif kind == 'holes':
        f[1:4] = .8333333333333334
        f[-1] = .5
    elif kind == 'just_below_one':
        f[2] = 1.-1e-16
    elif kind == 'empty':
        f[:] = 0.
    return f


class PolicyTests(unittest.TestCase):
    """The switch and its knobs."""

    def test_unset_is_the_rotation_and_two_values_are_known(self):
        with _clean():
            self.assertEqual(handover.ritz_handover_policy(), 'rotation')
            self.assertEqual(handover.validate_handover_settings(), 'rotation')
        for value, expected in (('rotation', 'rotation'), ('triangular', 'triangular'), (' Triangular ', 'triangular'),
                                ('', 'rotation')):
            with _clean(PARSEC_CUPY_RITZ_HANDOVER=value):
                self.assertEqual(handover.ritz_handover_policy(), expected)
        for value in ('on', 'tri', '1', 'factor', 'values'):
            with _clean(PARSEC_CUPY_RITZ_HANDOVER=value), self.assertRaisesRegex(ValueError, 'PARSEC_CUPY_RITZ_HANDOVER'):
                handover.validate_handover_settings()

    def test_the_knobs_have_their_defaults_and_refuse_what_they_do_not_know(self):
        with _clean():
            self.assertEqual(handover.handover_slab_columns(), 128)
            self.assertEqual(handover.first_cycle_condition_limit(), 1e4)
            self.assertEqual(handover.correction_column_limit(100), 64)
            self.assertEqual(handover.correction_column_limit(4928), 616)
            self.assertEqual(handover.handover_product(), 'columns')
        with _clean(PARSEC_CUPY_RITZ_HANDOVER_COLUMNS='64', PARSEC_CUPY_RITZ_HANDOVER_FIRST_CONDITION='0',
                    PARSEC_CUPY_RITZ_HANDOVER_CORRECTION_SHARE='2', PARSEC_CUPY_RITZ_HANDOVER_PRODUCT='tiles'):
            self.assertEqual((handover.handover_slab_columns(), handover.first_cycle_condition_limit(),
                              handover.correction_column_limit(1000), handover.handover_product()), (64, 0., 500, 'tiles'))
        for name, value in (('COLUMNS', '0'), ('COLUMNS', 'x'), ('FIRST_CONDITION', '-1'), ('FIRST_CONDITION', 'nan'),
                            ('CORRECTION_SHARE', '0'), ('PRODUCT', 'dtrmm')):
            with _clean(**{f'PARSEC_CUPY_RITZ_HANDOVER_{name}': value}), \
                    self.assertRaisesRegex(ValueError, f'PARSEC_CUPY_RITZ_HANDOVER_{name}'):
                handover.validate_handover_settings()

    def test_the_first_solve_hands_over_values_then_factors_and_rotates_last(self):
        self.assertEqual(chebff.first_solve_modes('rotation', 4), ('rotation',)*4)
        self.assertEqual(chebff.first_solve_modes('triangular', 4), ('values', 'factor', 'factor', 'rotation'))
        self.assertEqual(chebff.first_solve_modes('triangular', 2), ('values', 'rotation'))
        self.assertEqual(chebff.first_solve_modes('triangular', 1), ('rotation',))


class OrderTests(unittest.TestCase):
    """The order of the factorization on a shared basis and the output slabs of its product."""

    @staticmethod
    def _rounds(layout, rounds):
        """Spots and the slot of every row position, as _ritz_one_block lays the rounds."""
        states = layout.columns
        spots, order = [], np.empty(states, dtype=np.int64)
        for taken in rounds:
            offsets = [0]
            for first, last in taken:
                offsets.append(offsets[-1]+last-first)
            end = spots[-1][0] if spots else states
            spots.append((end-offsets[-1], end))
            for index, (first, last) in enumerate(taken):
                for column, offset, size in layout.spans(index, first, last):
                    start = spots[-1][0]+offsets[index]+offset-first
                    order[start:start+size] = np.arange(column, column+size)
        return spots, order

    def test_the_lower_half_comes_first_and_no_slab_cuts_the_band(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import subspace_filter_blocks
        rng = np.random.default_rng(3)
        for states, devices, budget, multiple, width in ((53, 2, 4, 1, 8), (53, 4, 3, 1, 8), (442, 4, 32, 64, 128),
                                                         (897, 2, 64, 64, 128), (4928, 4, 32, 64, 128), (61, 3, 5, 1, 4)):
            blocks = subspace_filter_blocks(states, 6, 15, 3)
            halves = sum(b.stop-b.start for b in blocks if b.degree == 12)
            ranges = distributed_state.interleaved_ranges(blocks, devices)
            layout = distributed_state.ColumnLayout(
                ranges if ranges is not None else distributed_state.SectorDeviceGroup.column_ranges(
                    SimpleNamespace(devices=tuple(range(devices))), blocks))
            rounds = distributed_state.block_rounds(layout.widths, budget, multiple)
            spots, order = self._rounds(layout, rounds)
            hand, band = handover.handover_order(order, halves)
            self.assertEqual(sorted(hand), list(range(states)))
            self.assertTrue(np.all(order[hand[:halves]] < halves), (states, devices))
            if band is not None:
                first, stop = band
                self.assertTrue(np.all(order[:first] < halves) and np.all(order[stop:] >= halves))
            # In row order U is zero below every output slab: a dense upper triangle in the order of hand.
            hpos = np.empty_like(hand)
            hpos[hand] = np.arange(states)
            upper = np.triu(rng.normal(size=(states, states)))
            row = upper[np.ix_(hpos, hpos)]
            slabs = handover.handover_slabs(spots, band, width)
            self.assertEqual((slabs[0][0], slabs[-1][1]), (0, states))
            self.assertTrue(all(a[1] == b[0] for a, b in zip(slabs[:-1], slabs[1:])))
            for first, last in slabs:
                self.assertEqual(np.count_nonzero(row[last:, first:last]), 0, (states, devices, first, last))
                if band is not None:
                    self.assertFalse(first < band[0] < last < band[1] or band[0] < first < band[1] < last)
            self.assertTrue(all(last-first >= width for first, last in slabs[:-1]) or len(slabs) == 1)
            # Without halves the order is the row order and nothing is a band.
            plain, none = handover.handover_order(order, None)
            self.assertEqual((list(plain), none), (list(range(states)), None))

    def test_uniform_slabs_cover_the_columns(self):
        self.assertEqual(handover.uniform_slabs(10, 4), ((0, 4), (4, 8), (8, 10)))
        self.assertEqual(handover.uniform_slabs(3, 128), ((0, 3),))

    def test_the_correction_set_is_every_ritz_vector_whose_occupation_is_not_exactly_one(self):
        full, correction, weights = handover.correction_set(np.array([1., 1.-1e-16, .5, 1.]), np.arange(4), 7)
        self.assertEqual(list(correction), [1, 2, 4, 5, 6])
        np.testing.assert_array_equal(weights, [1.-(1.-1e-16), .5, 1., 1., 1.])
        self.assertGreater(weights[0], 0.)
        np.testing.assert_array_equal(full, [1., 1.-1e-16, .5, 1., 0., 0., 0.])
        # A selection that is no prefix: the occupations go to the Ritz vectors they belong to.
        full, correction, _weights = handover.correction_set(np.array([1., .25]), np.array([0, 2]), 4)
        np.testing.assert_array_equal(full, [1., 0., .25, 0.])
        self.assertEqual(list(correction), [1, 2, 3])


class HostSolveTests(unittest.TestCase):
    """The small solve of a hand-over on the host against the rotation's."""

    def test_triangular_has_the_ritz_values_of_the_rotation_bit_for_bit_and_the_inverse_factor(self):
        with _clean(PARSEC_CUPY_RITZ_EIGH_BACKEND='host', PARSEC_CUPY_RITZ_CONDITION='symmetric',
                    PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8'):
            for columns, seed, condition in ((7, 1, None), (31, 2, 1e5), (64, 3, None)):
                overlap, projection = _gram_pair(columns, seed, condition)
                values, coefficients, whitened = ritz.solve_whitened_ritz(overlap, projection)
                solution = ritz.solve_whitened_handover(overlap, projection, mode='triangular')
                self.assertEqual((solution.mode, solution.note), ('triangular', None))
                np.testing.assert_array_equal(solution.eigenvalues, values)
                np.testing.assert_array_equal(solution.whitened, whitened)
                full = np.tril(overlap)+np.tril(overlap, -1).T
                cholesky = np.linalg.cholesky(full)
                np.testing.assert_allclose(solution.factor, np.linalg.inv(cholesky).T, rtol=1e-10, atol=1e-13)
                self.assertTrue(np.allclose(np.tril(solution.factor, -1), 0.))
                self.assertTrue(solution.factor.flags.f_contiguous)
                z = solution.eigenvectors
                np.testing.assert_allclose(z.T @ z, np.eye(columns), atol=1e-12)
                # U Z are the coefficients of the rotation.
                np.testing.assert_allclose(solution.factor @ z, coefficients, rtol=0, atol=1e-10*np.abs(coefficients).max())
                self.assertLessEqual(ritz._orthogonality_error(solution.factor, full), 5e-10)
                for mode in ('factor', 'values'):
                    other = ritz.solve_whitened_handover(overlap, projection, mode=mode, values_condition_limit=1e12)
                    self.assertEqual((other.mode, other.eigenvectors), (mode, None))
                    scale = np.abs(values).max()
                    np.testing.assert_allclose(other.eigenvalues, values, rtol=0, atol=1e-12*scale)
                    self.assertEqual(other.factor is None, mode == 'values')

    def test_values_take_the_factor_above_their_limit_and_with_a_limit_of_zero(self):
        overlap, projection = _gram_pair(9, 4, 1e3)
        with _clean(PARSEC_CUPY_RITZ_EIGH_BACKEND='host', PARSEC_CUPY_RITZ_CONDITION='symmetric'):
            for limit, mode in ((1e6, 'values'), (1e2, 'factor'), (0., 'factor')):
                solution = ritz.solve_whitened_handover(overlap, projection, mode='values', values_condition_limit=limit)
                self.assertEqual(solution.mode, mode, limit)
                self.assertEqual(solution.note is None, mode == 'values')
                self.assertGreater(solution.condition, 1e2)

    def test_the_same_overlaps_are_refused_as_by_the_rotation(self):
        columns = 6
        singular = np.ones((columns, columns))
        unsafe, projection = _gram_pair(columns, 5, 1e12)
        with _clean(PARSEC_CUPY_RITZ_EIGH_BACKEND='host', PARSEC_CUPY_RITZ_CONDITION='symmetric',
                    PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8'):
            for overlap in (singular, unsafe):
                with self.assertRaises(ritz.GeneralizedRitzStabilityError):
                    ritz.solve_whitened_ritz(overlap, projection)
                for mode in ('triangular', 'factor', 'values'):
                    with self.assertRaises(ritz.GeneralizedRitzStabilityError):
                        ritz.solve_whitened_handover(overlap, projection, mode=mode)
        with self.assertRaises(ValueError):
            ritz.solve_whitened_handover(singular, projection, mode='rotation')


@unittest.skipUnless(_device_count() >= 1, 'a CUDA device required')
class DeviceTests(unittest.TestCase):
    """The small solve, the triangular product and the Ritz step of one array on a real device."""

    rows = 257

    def setUp(self):
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()
        self.op = _operator(self.rows)

    def tearDown(self):
        gc.collect()

    def test_the_device_solve_agrees_with_the_host_and_keeps_the_values_of_the_rotation(self):
        cp = self.cp
        with _clean(PARSEC_CUPY_RITZ_EIGH_BACKEND='host', PARSEC_CUPY_RITZ_CONDITION='symmetric',
                    PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8'):
            for columns, seed, condition in ((7, 1, None), (31, 2, 1e5), (64, 3, None)):
                overlap, projection = _gram_pair(columns, seed, condition)
                pair = cp.asarray(overlap), cp.asarray(projection)
                values, _coefficients, _whitened = ritz.solve_whitened_ritz_on_device(*pair)
                host = ritz.solve_whitened_handover(overlap, projection, mode='triangular')
                scale = np.abs(host.eigenvalues).max()
                for mode in ('triangular', 'factor', 'values'):
                    device = ritz.solve_whitened_handover_on_device(*pair, mode=mode, values_condition_limit=1e12)
                    self.assertEqual(device.mode, mode)
                    if mode == 'triangular':
                        # The same statements up to the eigensolve: the values of the rotation, bit for bit.
                        np.testing.assert_array_equal(cp.asnumpy(device.eigenvalues), cp.asnumpy(values))
                        z = cp.asnumpy(device.eigenvectors)
                        np.testing.assert_allclose(np.abs(z.T @ host.eigenvectors), np.eye(columns), atol=1e-9)
                    np.testing.assert_allclose(cp.asnumpy(device.eigenvalues), host.eigenvalues, rtol=0,
                                               atol=1e-11*scale*(condition or 1.))
                    if mode != 'values':
                        np.testing.assert_allclose(cp.asnumpy(device.factor), host.factor, rtol=1e-10,
                                                   atol=1e-12*np.abs(host.factor).max())
                    self.assertGreater(device.condition, 0.)

    def test_a_condition_estimated_beside_the_solve_is_judged_as_in_front_of_it(self):
        cp = self.cp
        overlap, projection = _gram_pair(9, 4, 1e3)
        events = []

        def start(matrix):
            events.append('start')
            return lambda: (events.append('join'), ritz._device_overlap_condition(matrix))[1]
        with _clean(PARSEC_CUPY_RITZ_CONDITION='symmetric', PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8'):
            front = ritz.solve_whitened_handover_on_device(cp.asarray(overlap), cp.asarray(projection), mode='values',
                                                           values_condition_limit=1e6)
            beside = ritz.solve_whitened_handover_on_device(cp.asarray(overlap), cp.asarray(projection), start,
                                                            mode='values', values_condition_limit=1e6)
            self.assertEqual(events, ['start', 'join'])
            self.assertEqual((front.mode, beside.mode, front.condition), ('values', 'values', beside.condition))
            with patch.dict(os.environ, PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='10'), \
                    self.assertRaisesRegex(ritz.GeneralizedRitzStabilityError, 'unsafe'):
                ritz.solve_whitened_handover_on_device(cp.asarray(overlap), cp.asarray(projection), start, mode='factor')

    def test_the_frozen_rotation_solve_gives_todays_bits(self):
        # A copy of the rotation's device solve as it stood when the hand-over was added: the function itself
        # must keep these bits, so that the hand-over beside it leaves the default as it was.
        import cupyx
        from cupyx.scipy.linalg import solve_triangular
        cp = self.cp

        def frozen(raw_overlap, raw_projection):
            lower = cp.tril(raw_overlap)
            overlap = lower+cp.tril(lower, -1).T
            lower = cp.tril(raw_projection)
            projected = lower+cp.tril(lower, -1).T
            ritz._judge_condition(ritz._device_overlap_condition, overlap)
            with cupyx.errstate(linalg='raise'):
                cholesky = cp.asfortranarray(cp.linalg.cholesky(overlap))
                left = solve_triangular(cholesky, projected, lower=True)
                whitened = solve_triangular(cholesky, left.T, lower=True).T
                values, vectors = cp.linalg.eigh(whitened)
                coefficients = solve_triangular(cp.asfortranarray(cholesky.T), vectors, lower=False)
            return values, coefficients, whitened
        with _clean(PARSEC_CUPY_RITZ_CONDITION='symmetric', PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1e8'):
            for seed in range(4):
                pair = tuple(cp.asarray(part) for part in _gram_pair(23, seed, 1e4))
                for today, kept in zip(ritz.solve_whitened_ritz_on_device(*pair), frozen(*pair)):
                    np.testing.assert_array_equal(cp.asnumpy(today), cp.asnumpy(kept))

    def test_the_triangular_product_is_the_product_by_the_factor(self):
        cp, rng = self.cp, np.random.default_rng(5)
        rows, columns = 1003, 37
        host = rng.normal(size=(rows, columns))
        triangle = np.triu(rng.normal(size=(columns, columns)))+3.*np.eye(columns)
        # A dense diagonal block over [12, 20), as the band of a shared basis is, which no slab may cut.
        banded = triangle.copy()
        banded[12:20, 12:20] = rng.normal(size=(8, 8))
        for upper, slabs in ((triangle, handover.uniform_slabs(columns, 5)), (banded, ((0, 12), (12, 20), (20, 33), (33, 37))),
                             (banded, ((0, 37),))):
            expected = host @ upper
            for product, elements in (('columns', rows*37), ('tiles', 211*columns), ('tiles', columns), ('columns', 3*rows)):
                matrix = cp.array(host, order='F')
                pointer = matrix.data.ptr
                buffer = cp.empty(elements, dtype=cp.float64)
                ran = ritz._triangular_in_place(matrix, cp.asarray(upper), buffer, slabs, product)
                widest = max(b-a for a, b in slabs)
                self.assertEqual(ran, 'columns' if product == 'columns' and rows*widest <= elements else 'tiles')
                self.assertEqual(matrix.data.ptr, pointer)
                np.testing.assert_allclose(cp.asnumpy(matrix), expected, rtol=1e-13, atol=1e-13*np.abs(expected).max())
        # Both copy orders of the tiles give the same bits.
        results = []
        for copy in ('1', '0'):
            with patch.dict(os.environ, PARSEC_CUPY_ROTATE_COLUMN_COPY=copy):
                matrix = cp.array(host, order='F')
                ritz._triangular_in_place(matrix, cp.asarray(triangle), cp.empty(97*columns),
                                          handover.uniform_slabs(columns, 7), 'tiles')
                results.append(cp.asnumpy(matrix))
        np.testing.assert_array_equal(*results)
        with self.assertRaisesRegex(ValueError, 'cover'):
            ritz._triangular_in_place(cp.array(host, order='F'), cp.asarray(triangle), cp.empty(rows*columns),
                                      ((0, 5), (6, 37)))

    def test_the_streaming_step_hands_over_the_triangular_basis_and_the_eigenvectors(self):
        cp, rng = self.cp, np.random.default_rng(41)
        host = rng.normal(size=(self.rows, 17))
        with _clean(**_ROUTE, PARSEC_CUPY_STREAMING_RITZ_BYTES='7700', PARSEC_CUPY_RITZ_HANDOVER_COLUMNS='5'):
            rotated = ritz.generalized_rayleigh_ritz(self.op, cp.array(host, order='F'), consume_basis=True)
            self.assertIsNone(rotated.handover)
            for product in ('columns', 'tiles'):
                with patch.dict(os.environ, PARSEC_CUPY_RITZ_HANDOVER_PRODUCT=product):
                    matrix = cp.array(host, order='F')
                    pointer = matrix.data.ptr
                    result = ritz.generalized_rayleigh_ritz(self.op, matrix, consume_basis=True, handover='triangular')
                step = result.handover
                self.assertEqual((step.mode, step.product, result.wavefunctions.data.ptr), ('triangular', product, pointer))
                self.assertEqual(result.algorithm, 'streaming_generalized_cholesky_rayleigh_ritz')
                np.testing.assert_array_equal(cp.asnumpy(result.eigenvalues), cp.asnumpy(rotated.eigenvalues))
                q, z = cp.asnumpy(result.wavefunctions), cp.asnumpy(step.coefficients)
                np.testing.assert_allclose(q.T @ q, np.eye(17), atol=1e-12)
                for count in range(1, 18):
                    basis = np.linalg.qr(host[:, :count])[0]
                    self.assertLess(np.abs(q[:, :count]-basis @ (basis.T @ q[:, :count])).max(), 1e-10)
                np.testing.assert_allclose(np.abs(cp.asnumpy(rotated.wavefunctions).T @ (q @ z)), np.eye(17), atol=2e-10)
            # The filtered columns as they are, bit for bit, and the values of the rotation.
            matrix = cp.array(host, order='F')
            result = ritz.generalized_rayleigh_ritz(self.op, matrix, consume_basis=True, handover='values',
                                                    values_condition_limit=1e12)
            self.assertEqual((result.handover.mode, result.handover.coefficients), ('values', None))
            np.testing.assert_array_equal(cp.asnumpy(result.wavefunctions), host)
            np.testing.assert_allclose(cp.asnumpy(result.eigenvalues), cp.asnumpy(rotated.eigenvalues), rtol=0, atol=1e-12)
            # Two arrays: the step rotates and says why.
            with patch.dict(os.environ, PARSEC_CUPY_STREAMING_RITZ='0'):
                two = ritz.generalized_rayleigh_ritz(self.op, cp.array(host, order='F'), consume_basis=True,
                                                     handover='triangular')
            self.assertEqual(two.handover.mode, 'rotation')
            self.assertIn('two arrays', two.handover.note)
            np.testing.assert_allclose(cp.asnumpy(two.eigenvalues), cp.asnumpy(rotated.eigenvalues), rtol=0, atol=1e-12)
            # The host small solve hands over the same basis.
            with patch.dict(os.environ, PARSEC_CUPY_RITZ_DENSE_BACKEND='host'):
                on_host = ritz.generalized_rayleigh_ritz(self.op, cp.array(host, order='F'), consume_basis=True,
                                                         handover='triangular')
            np.testing.assert_allclose(cp.asnumpy(on_host.wavefunctions), q, rtol=0, atol=1e-12)
            # A failed audit leaves the filtered columns as they were.
            matrix = cp.array(host, order='F')
            with patch.dict(os.environ, PARSEC_CUPY_GENERALIZED_RITZ_CONDITION_MAX='1.0001'), \
                    self.assertRaises(ritz.GeneralizedRitzStabilityError):
                ritz.generalized_rayleigh_ritz(self.op, matrix, consume_basis=True, handover='triangular')
            np.testing.assert_array_equal(cp.asnumpy(matrix), host)
