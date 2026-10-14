"""Tall transposed products reach the library without an even width below 64."""

import importlib
import os
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy
from parsec_python.acceleration.Eigensolvers.tall_product import (
    split_requested,
    split_width,
    transposed_product,
)

_SWITCH = "PARSEC_CUPY_TALL_PRODUCT_SPLIT"
_CLEARED = (
    _SWITCH,
    "PARSEC_CUPY_RITZ_GRAM_MULTIPLE",
    "PARSEC_CUPY_RITZ_GRAM_SLABS",
    "PARSEC_CUPY_RITZ_SYRK",
    "PARSEC_CUPY_STREAMING_RITZ_BYTES",
)


class Recorded(np.ndarray):
    """A host array that notes the shapes of every matrix product it starts."""

    products: list = []

    def __matmul__(self, other):
        Recorded.products.append((tuple(self.shape), tuple(np.shape(other))))
        return np.matmul(np.asarray(self), np.asarray(other))


def _recorded(values):
    """``values``, which is column-major, as an array that records from now on."""

    assert values.flags.f_contiguous
    Recorded.products = []
    return values.view(Recorded)


def _right_widths():
    """Columns of the right-hand operand of every recorded product."""

    return [right[1] for _left, right in Recorded.products]


def _ritz_module():
    # The package also exports a function named like this submodule.
    return importlib.import_module(
        "parsec_python.acceleration.Eigensolvers.rayleigh_ritz"
    )


class TallProductTests(unittest.TestCase):
    """The rule itself; NumPy stands in for CuPy, so no device is needed."""

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _CLEARED:
            os.environ.pop(name, None)
        self.rng = np.random.default_rng(101)

    def _tall(self, columns, rows=131):
        return np.asfortranarray(self.rng.normal(size=(rows, columns)))

    def test_which_widths_are_split(self):
        self.assertTrue(split_requested())
        self.assertEqual(
            [width for width in range(0, 70) if split_width(width)],
            list(range(2, 64, 2)),
        )
        for value in ("0", "off", "false"):
            os.environ[_SWITCH] = value
            self.assertFalse(split_requested())
            self.assertFalse(any(split_width(width) for width in range(0, 70)))
        for value in ("1", "on", "true"):
            os.environ[_SWITCH] = value
            self.assertTrue(split_width(30))
        os.environ[_SWITCH] = "sometimes"
        with self.assertRaisesRegex(ValueError, _SWITCH):
            split_requested()

    def test_no_even_width_below_64_reaches_the_library(self):
        left = self._tall(9)
        wide = self._tall(70)
        for width in range(1, 71):
            right = wide[:, :width]
            whole = left.T @ right
            product = transposed_product(np, _recorded(left), right)
            widths = _right_widths()
            if width % 2 == 0 and width < 64:
                # All columns but the last, an odd number, and the last one.
                self.assertEqual(widths, [width - 1, 1])
                np.testing.assert_allclose(product, whole, rtol=1e-13, atol=1e-13)
                # Each piece is the product of its own columns, to the last bit.
                np.testing.assert_array_equal(
                    product[:, : width - 1], left.T @ right[:, : width - 1]
                )
                np.testing.assert_array_equal(
                    product[:, width - 1 :], left.T @ right[:, width - 1 :]
                )
            else:
                self.assertEqual(widths, [width])
                np.testing.assert_array_equal(product, whole)
            self.assertEqual(product.shape, (9, width))
            self.assertEqual(product.dtype, np.float64)

    def test_the_switch_restores_the_whole_product(self):
        left = self._tall(9)
        right = self._tall(30)
        os.environ[_SWITCH] = "0"
        product = transposed_product(np, _recorded(left), right)
        self.assertEqual(_right_widths(), [30])
        np.testing.assert_array_equal(product, left.T @ right)

    def test_a_kept_array_receives_the_pieces(self):
        left = self._tall(9)
        wide = self._tall(30)
        for width, pieces in ((30, 2), (29, 1), (6, 2), (1, 1)):
            right = wide[:, :width]
            kept = np.full((9, width), np.nan, order="F")
            with patch.object(np, "matmul", wraps=np.matmul) as multiply:
                product = transposed_product(np, left, right, out=kept)
            self.assertIs(product, kept)
            self.assertEqual(multiply.call_count, pieces)
            self.assertEqual(
                [call.args[1].shape[1] for call in multiply.call_args_list],
                [width - 1, 1] if pieces == 2 else [width],
            )
            np.testing.assert_allclose(kept, left.T @ right, rtol=1e-13, atol=1e-13)

    def test_a_vector_is_one_product(self):
        left = self._tall(8)
        vector = self.rng.normal(size=131)
        product = transposed_product(np, _recorded(left), vector)
        self.assertEqual(Recorded.products, [((8, 131), (131,))])
        np.testing.assert_array_equal(product, left.T @ vector)

    def test_single_precision_keeps_its_type(self):
        left = self._tall(5).astype(np.float32)
        right = self._tall(6).astype(np.float32)
        product = transposed_product(np, left, right)
        self.assertEqual((product.shape, product.dtype), ((5, 6), np.float32))
        np.testing.assert_allclose(product, left.T @ right, rtol=1e-5, atol=1e-5)

    def test_the_ritz_step_hands_over_no_even_slab(self):
        ritz = _ritz_module()
        operator = self.rng.normal(size=(131, 131))
        operator += operator.T
        # 30 columns are the 24 states and 6 buffer states of a small
        # molecule: seven slabs of 4 columns and one of 2.  46 columns make
        # slabs of 6 and one of 4, 23 make slabs of 3 and one of 2, and the
        # 38 of 32 states slabs of 5 and one of 3, none of them even.
        for columns, slab in ((30, 4), (46, 6), (23, 3), (38, 5)):
            host = self._tall(columns)
            applied = operator @ host
            with patch.object(ritz, "require_cupy", return_value=(np, None)):
                self.assertEqual(ritz._gram_slab_width(columns), slab)
                lower = ritz._lower_triangle_product(_recorded(host), applied)
                gram_widths = _right_widths()
                overlap = ritz._symmetric_overlap(_recorded(host))
                overlap_widths = _right_widths()
                # The one-array route holds a basis this small in one slab.
                streamed = ritz._streamed_projection(operator, _recorded(host))
                streamed_widths = _right_widths()
            for widths in (gram_widths, overlap_widths, streamed_widths):
                self.assertTrue(all(width % 2 == 1 for width in widths), widths)
                self.assertEqual(sum(widths), columns)
            self.assertEqual(gram_widths, overlap_widths)
            self.assertEqual(
                streamed_widths, [columns] if columns % 2 else [columns - 1, 1]
            )
            for product, reference in (
                (lower, host.T @ applied),
                (overlap, host.T @ host),
                (streamed, host.T @ applied),
            ):
                np.testing.assert_allclose(
                    np.tril(product), np.tril(reference), rtol=1e-12, atol=1e-10
                )
            # Nothing is computed above the diagonal slabs, as before.
            for product in (lower, overlap):
                for start in range(slab, columns, slab):
                    self.assertFalse(product[:start, start : start + slab].any())

    def test_the_ritz_step_with_the_switch_off_is_the_former_one(self):
        ritz = _ritz_module()
        host = self._tall(30)
        other = self._tall(30)
        former = np.zeros((30, 30), order="F")
        for start in range(0, 30, 4):
            former[start:, start : start + 4] = (
                host[:, start:].T @ other[:, start : start + 4]
            )
        os.environ[_SWITCH] = "0"
        with patch.object(ritz, "require_cupy", return_value=(np, None)):
            lower = ritz._lower_triangle_product(_recorded(host), other)
        self.assertEqual(_right_widths(), [4] * 7 + [2])
        np.testing.assert_array_equal(lower, former)


@unittest.skipUnless(cupy_available(), "CUDA required")
class TallProductDeviceTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _CLEARED:
            os.environ.pop(name, None)
        self.cp, _ = require_cupy()
        rng = np.random.default_rng(107)
        self.rows, self.columns = 521, 30
        diagonal = rng.normal(size=self.rows)
        beside = 0.3 * rng.normal(size=self.rows - 1)
        self.host_operator = (
            np.diag(diagonal) + np.diag(beside, 1) + np.diag(beside, -1)
        )
        self.operator = self.cp.asarray(self.host_operator)
        self.host = rng.normal(size=(self.rows, self.columns))

    def test_device_products_match_the_whole_product(self):
        cp = self.cp
        rng = np.random.default_rng(103)
        host_left = rng.normal(size=(4099, 31))
        host_right = rng.normal(size=(4099, 66))
        for order in ("F", "C"):
            left = cp.array(host_left, order=order)
            right = cp.array(host_right, order=order)
            for width in (1, 2, 3, 4, 16, 29, 30, 62, 63, 64, 66):
                product = transposed_product(cp, left, right[:, :width])
                self.assertEqual(product.shape, (31, width))
                np.testing.assert_allclose(
                    cp.asnumpy(product),
                    host_left.T @ host_right[:, :width],
                    rtol=1e-12,
                    atol=1e-11,
                )
            vector = cp.array(host_right[:, 0])
            np.testing.assert_allclose(
                cp.asnumpy(transposed_product(cp, left, vector)),
                host_left.T @ host_right[:, 0],
                rtol=1e-12,
                atol=1e-11,
            )
            # A column-major array kept by the caller, as CHEBDAV and the
            # block orthogonalization keep one beside a row-major basis.
            for width in (6, 15, 16):
                kept = cp.full((31, width), cp.nan, dtype=cp.float64, order="F")
                self.assertIs(
                    transposed_product(cp, left, right[:, :width], out=kept), kept
                )
                np.testing.assert_allclose(
                    cp.asnumpy(kept),
                    host_left.T @ host_right[:, :width],
                    rtol=1e-12,
                    atol=1e-11,
                )

    def test_generalized_ritz_pairs_do_not_depend_on_the_split(self):
        from scipy.linalg import eigh

        ritz = _ritz_module()
        cp = self.cp
        expected = eigh(
            self.host.T @ self.host_operator @ self.host,
            self.host.T @ self.host,
            eigvals_only=True,
        )
        for streaming in ("0", "1"):
            results = {}
            for value in ("1", "0"):
                with patch.dict(
                    os.environ,
                    {_SWITCH: value},
                    PARSEC_CUPY_STREAMING_RITZ=streaming,
                ):
                    self.assertEqual(ritz._gram_slab_width(self.columns), 4)
                    results[value] = ritz.generalized_rayleigh_ritz(
                        self.operator,
                        cp.array(self.host, order="F"),
                        consume_basis=True,
                    )
                self.assertEqual(
                    results[value].algorithm.startswith("streaming"),
                    streaming == "1",
                )
                np.testing.assert_allclose(
                    cp.asnumpy(results[value].eigenvalues),
                    expected,
                    rtol=0,
                    atol=1e-9,
                )
            np.testing.assert_allclose(
                cp.asnumpy(results["1"].eigenvalues),
                cp.asnumpy(results["0"].eigenvalues),
                rtol=0,
                atol=5e-11,
            )
            first = cp.asnumpy(results["1"].wavefunctions)
            second = cp.asnumpy(results["0"].wavefunctions)
            np.testing.assert_allclose(
                np.linalg.svd(first.T @ second, compute_uv=False),
                np.ones(self.columns),
                atol=5e-9,
            )

    def test_orthonormal_ritz_values_do_not_depend_on_the_split(self):
        ritz = _ritz_module()
        cp = self.cp
        basis = np.linalg.qr(self.host)[0]
        expected = np.linalg.eigvalsh(basis.T @ self.host_operator @ basis)
        for value in ("1", "0"):
            os.environ[_SWITCH] = value
            result = ritz.rayleigh_ritz(
                self.operator, cp.array(basis, order="F"), compute_residuals=False
            )
            np.testing.assert_allclose(
                cp.asnumpy(result.eigenvalues), expected, rtol=0, atol=1e-10
            )


if __name__ == "__main__":
    unittest.main()
