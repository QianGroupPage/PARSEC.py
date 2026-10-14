"""Transposed products of two tall arrays without an even right-hand width.

The Gram matrices of the eigensolver are products ``X.T @ Y`` of arrays with
as many rows as the grid has points and a few columns.  CuPy hands one to
cuBLAS as a single DGEMM whose transposed operand is ``Y``.  On an RTX 5070
(cuBLAS 12.9.1, CuPy 14.0.1) that call takes one of two kernels by the
number of columns of ``Y`` alone, whatever ``X`` has.  Timed one product at
a time on the idle device at 1,065,944 rows:

* an odd number: 6.2 to 13.9 ms up to 63 columns of ``Y`` by 31 of ``X``,
  24.8 ms by 64;
* an even number: 221.7 ms, 208 ns per row, for every size from 2 by 2 to
  66 by 64 columns;
* one column: 0.1 to 1.7 ms.

The time of the even kernel follows the rows alone (4.2 ms at 20,000 rows,
55.2 at 266,486, 441 at 2,131,888) until the operands are wide, and then
grows in whole multiples of that: 110, 221 and 331 ms at 266,486 rows for
64, 128 and 256 columns by 665, where 63, 127 and 255 took 68, 147 and 276.
The call returns at once and the device is busy that long.  A row-major
``X`` or ``Y`` changes none of this for an odd number of columns of ``Y``.

The Ritz step of 30 columns formed its two Gram matrices from 16 products
of 4 and 2 columns: 3.54 of the 3.68 s of an SCF step, where 38 columns, in
products of 5 and 3, took 0.32 s for the whole step.

:func:`transposed_product` therefore gives cuBLAS no even width below 64
columns: the product is formed from all columns but the last, an odd number
of them, and from the last one.  From 64 columns on a product is left as it
is.  The even kernel costs there too, but the slabs of a large basis are
whole multiples of 64 on purpose (:func:`rayleigh_ritz.gram_multiple`,
measured on A100 nodes), and neither an odd width nor a single column
beside each slab has been timed on those.  Below 64 columns the second
piece is at most one more small product on a device that has no such
kernel.

The two pieces are other sums than the one product, so the entries differ
from it by round-off.  ``PARSEC_CUPY_TALL_PRODUCT_SPLIT=0`` gives every
product to cuBLAS whole, as before, bit for bit.
"""

from __future__ import annotations

import os
from typing import Any


# Even widths from two columns to below this many are split.
_SPLIT_BELOW = 64


def split_requested() -> bool:
    """Whether an even width below 64 columns is multiplied in two pieces.

    The default.  ``PARSEC_CUPY_TALL_PRODUCT_SPLIT=0`` hands every product to
    cuBLAS as one call.
    """

    value = os.environ.get("PARSEC_CUPY_TALL_PRODUCT_SPLIT", "1").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_TALL_PRODUCT_SPLIT must be on or off")
    return value in {"1", "on", "true"}


def split_width(width: int) -> bool:
    """Whether a right-hand operand of ``width`` columns is multiplied in two pieces."""

    width = int(width)
    return 2 <= width < _SPLIT_BELOW and width % 2 == 0 and split_requested()


def transposed_product(cp: Any, left: Any, right: Any, out: Any | None = None):
    """Return ``left.T @ right`` for two tall arrays of the module ``cp``.

    ``right`` with an even number of columns below 64 is multiplied as its
    columns but the last and its last column (see the module).  Anything
    else, a vector included, is the one product.

    ``out`` receives the product where a caller keeps an array for it, as
    the ``out`` of ``cp.matmul`` does; its columns are those of ``right``.
    """

    if getattr(right, "ndim", 0) != 2 or not split_width(right.shape[1]):
        if out is None:
            return left.T @ right
        cp.matmul(left.T, right, out=out)
        return out
    last = int(right.shape[1]) - 1
    if out is not None:
        cp.matmul(left.T, right[:, :last], out=out[:, :last])
        cp.matmul(left.T, right[:, last:], out=out[:, last:])
        return out
    head = left.T @ right[:, :last]
    product = cp.empty((head.shape[0], last + 1), dtype=head.dtype, order="F")
    product[:, :last] = head
    product[:, last:] = left.T @ right[:, last:]
    return product


__all__ = ["split_requested", "split_width", "transposed_product"]
