"""``PARSEC_CUPY_STENCIL_KERNEL``: the pipelined recurrence kernel against the former stencil kernels.

The host tests read the sources and the launches.  ``former`` compiles the sources as they were before the switch (as
published on 17 October 2026) to the byte, under the cache keys they had.  ``pipelined`` compiles them as well and,
beside them, a recurrence kernel with a fast path in front of the slot loop of ``stencil_major_chebyshev6`` (the
source verified on A100); its operators apply ``H X`` with the former ``stencil_major_spmm6`` and send every launch
of the recurrence that does not take the fast path to the former recurrence kernel.  A generated kernel that does not
compile leaves its operator the former kernels in the layout it was built for, and the record of why.  The backend
details of a run name the kernels that its stencils launch, an unset variable included.

The device tests compile both in one process.  They launch every path of the recurrence kernel (widths 1 to 6, every
flag, seven operand layouts, slot counts below the compiled one) and of the operators' methods (widths up to 14, so
that blocks are cut into launches of six columns and fewer) on synthetic stencils in the slot-major layout and in
tiles of 16 and 32 rows, with finite values, values of wide exponents, signed zeros, subnormals and infinities, and
NaNs: a few, pairs of different payloads that meet in one operation, and every value a NaN of its own payload.  The
outputs, prefilled with a NaN pattern, are compared whole as 64-bit words, NaN payloads included, on devices of
compute capability 8.0 (A100), where the generated kernel was verified; there it must also hold no local memory.  On
other devices a NaN must come out where the former kernel gives one and every other word must be the former one: where
two NaNs of different payloads meet, IEEE 754 leaves the payload of the result open, and the hardware or ptxas of
another architecture may take the other one (a GeForce RTX 5070 Laptop GPU, compute capability 12.0, CUDA 12.9, did,
and its ptxas kept up to 56 bytes of local memory under the launch bounds).  A sweep takes the
slot counts of Expansion_Order 2 to 20 and others (2 to 61 slots) in the slot-major layout and tiles of 16 to 256
rows: the launches of the fast path with every kind of input, two NaNs and all NaNs on a stencil of NaN coefficients
as well, every value -0.0 on one of positive coefficients (a sum started from -0.0 would keep its sign), and the
operators' methods.  ``benchmarks/stencil_pipelined_check.py`` runs the same comparisons on the operators of real
sectors.
"""
from contextlib import contextmanager
import gc
import hashlib
import os
import sys
import time
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends import (
    cupy_compile, cupy_mixed_precision, cupy_stencil_major, implicit_stencil, stencil_pipeline)
from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy
from parsec_python.acceleration.backends.cupy_launch import kernel_parameters
from parsec_python.acceleration.backends.cupy_stencil_major import (
    _CUDA_SOURCE, CuPyStencilMajorFiniteDifference, StencilMajorHostMetadata, build_stencil_major_metadata)
from parsec_python.acceleration.backends.implicit_stencil import implicit_source, stencil_source
from parsec_python.acceleration.backends.stencil_pipeline import (
    DEFAULT, KERNEL, SWITCH, THREADS, WIDTH, pipelined_source, stencil_kernel_setting, takes_fast_path)
from parsec_python.acceleration.Eigensolvers.distributed_filter import DistributedFilter
from parsec_python.acceleration.tests import test_implicit_stencil as implicit_tests
from parsec_python.acceleration.tests import test_mixed_precision as mixed_tests
from parsec_python.acceleration.tests.test_implicit_stencil import _irregular_operator

_KERNELS = ("stencil_major_spmm6", "stencil_major_chebyshev6")
_TILES = (16, 32, 64, 128, 256)
# SHA-256 of the sources before the switch (as published on 17 October 2026): the slot-major kernels, those of each
# tile size, and the FP32 filter's (cupy_mixed_precision._SOURCE, derived from the slot-major one).
_FORMER_DIGESTS = {
    0: "b810e6467d7b66975446f3400672d25bab1465efabae9f130fadbff946fd8847",
    16: "638b886390184e39d193ea7da9abe47ae6c533104e4bc866889901624f43ad7c",
    32: "3e67ab50d8cbc2a01e5197613eb238053c248309716f05fb140b0529c25302ed",
    64: "bcb59657586e3318a803ecbb2cc6502cda1934bf37265d8710dac8dc5dff9f1b",
    128: "ffcba3d7cac605097e3587fa1ce2b72e704b0092ab0289990a82f598ec5c814c",
    256: "86b7d647c02be948b08bed9329d889b4d3098ec13e67db26dc40454678a8d278",
}
_FP32_DIGEST = "0519af9951af141e7c868a04f2dee535b5dce3a59783f4bf1fa1260388d148f8"
# SHA-256 of generated sources (tile, slots) that were compiled on A100 and compared with the former kernels on
# 2026-10-20, 0 differing words in 79,272 launch pairs, every value a NaN among the inputs: the fast path and the
# launch bounds in stencil_major_chebyshev6 alone.  Production sectors have 25 slots.
_VERIFIED_DIGESTS = {
    (16, 25): "0352d37a82f8939b8c4d250d4d6735b40c66e2052fd4c87c5ed9f4784cfe386b",
    (0, 25): "0005e840d17a79478bc1cb565ae1eb50f5422c3cb97743d99266d13f91f47fbe",
    (32, 25): "232765d3c6010161a9fa08e64f54733582ff9c5e888a89b6059095009832975a",
    (16, 13): "a208a1f7429577abe660131cbe0429219140f2479ec1f8ac1010b9f75a7226f5",
    (0, 3): "bdf7a0a354d87a28e699ab7ddc8b630c5aa9891c62f25626b31a2de050c31ef8",
}
# The compute capability of the A100, where NaN payloads and the absence of local memory are demanded.
VERIFIED_ARCHITECTURE = "80"
# Central second-difference weights of the orders 4, 8 and 12 (offsets 0 to order / 2).
_WEIGHTS = {
    4: (-5 / 2, 4 / 3, -1 / 12),
    8: (-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560),
    12: (-5369 / 1800, 12 / 7, -15 / 56, 10 / 189, -1 / 112, 2 / 1925, -1 / 16632),
}
# The word every output is prefilled with (a quiet NaN), and the NaN of a trap at palette[0].
FILL = np.uint64(0x7FF8DEADBEEF0001)
TRAP = np.uint64(0x7FF800000000A5A5)
LAYOUTS = ("F", "C", "colstride", "negcol", "rowrev", "bcast", "offset")
KINDS = ("finite", "wide", "specials", "nan", "twonan", "allnan")
# Every value -0.0, for a stencil of positive coefficients (the sweep).
SIGNED_ZEROS = "negzero"
# The parameters of stencil_major_chebyshev6 by name: where a launch has its width, row stride and slot count.
_RECURRENCE_PARAMETERS = [name for _kind, name in kernel_parameters(_CUDA_SOURCE, KERNEL)]


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@contextmanager
def _switch(value=None, **values):
    """The environment without the stencil and tile settings, ``PARSEC_CUPY_STENCIL_KERNEL`` at ``value``
    (``None``: unset), then ``values``."""
    names = (SWITCH, "PARSEC_CUPY_IMPLICIT_TILE", "PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS",
             "PARSEC_CUPY_IMPLICIT_PACK_WORKERS", "PARSEC_CUPY_STENCIL_MAJOR", "PARSEC_CUPY_MIXED_FILTER",
             "PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS")
    kept = {name: setting for name, setting in os.environ.items() if name not in names}
    if value is not None:
        kept[SWITCH] = value
    with patch.dict(os.environ, {**kept, **values}, clear=True):
        yield


# ----------------------------------------------------------------------------------------------------------------
# Synthetic stencils (slot-major host metadata)
# ----------------------------------------------------------------------------------------------------------------

def _star(shape, order, mask=None, spacing=(0.31, 0.29, 0.33)):
    """The negative Laplacian of a star stencil of ``order`` on the points of ``mask`` (the whole box by default).

    The rows are the points in the order of the grid, the last axis fastest; a neighbour outside the mask is
    missing, so the rows next to the boundary have fewer entries.
    """
    weights = _WEIGHTS[order]
    if mask is None:
        mask = np.ones(shape, dtype=bool)
    index = np.full(shape, -1, dtype=np.int64)
    inside = np.flatnonzero(mask.ravel())
    index.ravel()[inside] = np.arange(inside.size)
    points = np.argwhere(mask)
    rows, columns, values = [np.arange(inside.size)], [np.arange(inside.size)], [
        np.full(inside.size, -sum(weights[0] / h ** 2 for h in spacing))]
    for axis in range(3):
        for reach in range(1, len(weights)):
            for sign in (-1, 1):
                moved = points.copy()
                moved[:, axis] += sign * reach
                there = (moved[:, axis] >= 0) & (moved[:, axis] < shape[axis])
                target = np.full(points.shape[0], -1)
                target[there] = index[tuple(moved[there].T)]
                there = target >= 0
                rows.append(np.flatnonzero(there))
                columns.append(target[there])
                values.append(np.full(int(there.sum()), -weights[reach] / spacing[axis] ** 2))
    return sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                         shape=(inside.size, inside.size))


def _ball(shape, radius):
    grid = np.indices(shape).astype(float)
    return sum((grid[axis] - (shape[axis] - 1) / 2) ** 2 for axis in range(3)) <= radius ** 2


def _metadata(neighbors, codes, palette):
    rows = int(neighbors.shape[1])
    return StencilMajorHostMetadata(shape=(rows, rows), neighbors=np.ascontiguousarray(neighbors, np.int32),
                                    coefficient_codes=np.ascontiguousarray(codes, np.uint8),
                                    coefficient_palette=np.ascontiguousarray(palette, np.float64))


def _palette(generator, size, trap):
    palette = generator.uniform(-3.0, 3.0, size)
    if trap:
        palette.view(np.uint64)[0] = TRAP
    return palette


def _holes(rows=3013, slots=7, seed=29):
    """Neighbours missing in any slot, not only in the last ones of a row.

    Every other block of 32 rows is affine: one offset and one code per slot for all its rows, and some slots
    missing in all of them, in the middle of the slots as well.  The other blocks have random neighbours, missing
    ones among them, and an empty row.  A missing neighbour carries the code 255, which names no coefficient.
    """
    generator = np.random.default_rng(seed)
    palette = np.concatenate(([0.1], generator.uniform(-2.0, 2.0, 10)))
    neighbors = np.full((slots, rows), -1, np.int32)
    codes = np.full((slots, rows), 255, np.uint8)
    for start in range(0, rows, 32):
        stop = min(start + 32, rows)
        here = np.arange(start, stop)
        if (start // 32) % 2 == 0:
            for slot in range(slots):
                offset = int(generator.integers(-70, 71))
                if generator.random() < 0.25 or here[0] + offset < 0 or here[-1] + offset >= rows:
                    continue
                neighbors[slot, start:stop] = here + offset
                codes[slot, start:stop] = generator.integers(0, palette.size)
        else:
            there = generator.random((slots, stop - start)) >= 0.3
            neighbors[:, start:stop] = np.where(there, generator.integers(0, rows, (slots, stop - start)), -1)
            codes[:, start:stop] = np.where(there, generator.integers(0, palette.size, (slots, stop - start)), 255)
            neighbors[:, min(start + 5, stop - 1)] = -1
            codes[:, min(start + 5, stop - 1)] = 255
    return _metadata(neighbors, codes, palette)


def _random_any(rows, slots, seed, missing=0.3, trap=False):
    """Neighbours anywhere, missing in any slot, codes up to 255 where missing, an empty row."""
    generator = np.random.default_rng(seed)
    palette = _palette(generator, 12, trap)
    present = generator.random((slots, rows)) >= missing
    neighbors = np.where(present, generator.integers(0, rows, (slots, rows)), -1)
    low = 1 if trap else 0
    codes = np.where(present, generator.integers(low, palette.size, (slots, rows)),
                     generator.integers(0, 256, (slots, rows)))
    if rows > 3:
        neighbors[:, rows // 2] = -1
    return _metadata(neighbors, codes, palette)


def _affine_blocks(rows, slots, seed, block=512):
    """Blocks of ``block`` rows that share their offsets and codes per slot: regular tiles of every size, slots
    missing in a whole block (INT_MIN entries of regular tiles), a block with no neighbour at all, and blocks of
    random neighbours between them."""
    generator = np.random.default_rng(seed)
    palette = _palette(generator, 13, False)
    neighbors = np.full((slots, rows), -1, np.int64)
    codes = generator.integers(0, 256, (slots, rows))
    for index, start in enumerate(range(0, rows, block)):
        stop = min(start + block, rows)
        here = np.arange(start, stop)
        if index % 3 == 2:
            present = generator.random((slots, stop - start)) >= 0.25
            neighbors[:, start:stop] = np.where(present, generator.integers(0, rows, (slots, stop - start)), -1)
            codes[:, start:stop] = np.where(present, generator.integers(0, palette.size, (slots, stop - start)),
                                            codes[:, start:stop])
            continue
        if index == 4:
            continue
        for slot in range(slots):
            if generator.random() < 0.2:
                continue
            offset = int(generator.integers(-3 * block, 3 * block + 1))
            if here[0] + offset < 0 or here[-1] + offset >= rows:
                offset = 0
            neighbors[slot, start:stop] = here + offset
            codes[slot, start:stop] = int(generator.integers(0, palette.size))
    return _metadata(neighbors.astype(np.int32), codes, palette)


def nan_pattern(shape, number):
    """Quiet NaN words, one payload per position: ``number`` in bits 44 to 50, the position below, the sign
    alternating."""
    index = np.arange(int(np.prod(shape)), dtype=np.uint64).reshape(shape)
    words = (np.uint64(0x7FF8000000000000) | (np.uint64(number & 0x7F) << np.uint64(44))
             | (index & np.uint64(0xFFFFFFFFFFF)))
    return words | ((index & np.uint64(1)) << np.uint64(63))


def _nan_palette(metadata):
    """The stencil of ``metadata`` with every coefficient a NaN of its own payload: every FMA of the slot sum
    meets two NaNs once the sum is a NaN."""
    palette = np.array(metadata.coefficient_palette, dtype=np.float64)
    palette.view(np.uint64)[...] = nan_pattern(palette.shape, 14)
    return _metadata(metadata.neighbors, metadata.coefficient_codes, palette)


# The synthetic stencils of the device tests: name, builder of the slot-major metadata.
CASES = {
    # Three slots: rows that end at the boundary, an empty row, a missing diagonal.
    "irregular": lambda: build_stencil_major_metadata(_irregular_operator(1031)),
    # The order-8 star of production (25 slots) on a box and on a ball: 6,750 and 12,032 rows (the box not a whole
    # number of tiles of 16 or 32), boundary rows with fewer neighbours, regular and irregular tiles.
    "box8": lambda: build_stencil_major_metadata(_star((9, 10, 75), 8)),
    "ball8": lambda: build_stencil_major_metadata(_star((30, 31, 32), 8, _ball((30, 31, 32), 14.2))),
    # Other orders: 13 and 37 slots.
    "box4": lambda: build_stencil_major_metadata(_star((7, 8, 35), 4)),
    "box12": lambda: build_stencil_major_metadata(_star((13, 13, 21), 12)),
    # Seven slots with holes anywhere and codes that name no coefficient where a neighbour is missing.
    "holes": _holes,
    # 25 slots in blocks that share their offsets: regular tiles of every size, whole slots and a whole block
    # missing.
    "affine25": lambda: _affine_blocks(9001, 25, 7),
    # palette[0] a NaN that no present neighbour names: the FMA of a missing slot, if it ran, would show it.
    "trap25": lambda: _random_any(5003, 25, 9, trap=True),
    # Every coefficient a NaN of its own payload.
    "nan_palette8": lambda: _nan_palette(build_stencil_major_metadata(_star((9, 10, 25), 8))),
    # Fewer rows than a tile, and a single row.
    "rows17": lambda: _random_any(17, 25, 2),
    "one_row": lambda: _random_any(1, 25, 1),
}


# The sweep of the device tests: the slot counts of the stencils of Expansion_Order 2 to 20 (1 + 6M slots, M = 1 to
# 10) and others between them, odd and even, each in the slot-major layout and in tiles of every size.
SWEEP_SLOTS = tuple(sorted({1 + 6 * m for m in range(1, 11)} | {2, 9, 26, 40, 61}))
SWEEP_TILES = (0, 16, 32, 64, 128, 256)
SWEEP_ROWS = 4453


def _sweep_stencil(rows, slots, seed):
    """A stencil of ``slots`` slots in blocks of 512 rows of four kinds, in turn.

    Affine blocks (one offset and one code per slot for all their rows: regular tiles of every size) with whole slots
    missing anywhere; affine blocks whose last slots are missing, as at a boundary; blocks of random neighbours with
    holes in any slot and an empty row; and star-like blocks whose neighbours beyond the rows are missing.  A missing
    neighbour carries a code of 0 to 255 that names no coefficient, and palette[0] is a NaN that no present neighbour
    names: the FMA of a missing slot, if it ran, would show it.
    """
    generator = np.random.default_rng(seed)
    size = 13
    palette = _palette(generator, size, True)
    neighbors = np.full((slots, rows), -1, np.int64)
    codes = generator.integers(0, 256, (slots, rows))
    block = 512
    strides = (1, 23, 23 * 19)
    for index, start in enumerate(range(0, rows, block)):
        stop = min(start + block, rows)
        here = np.arange(start, stop)
        kind = index % 4
        if kind in (0, 1):
            present_slots = slots if kind == 0 else int(generator.integers(0, slots + 1))
            for slot in range(slots):
                if kind == 0 and slot and generator.random() < 0.15:
                    continue
                if kind == 1 and slot >= present_slots:
                    continue
                offset = int(generator.integers(-3 * block, 3 * block + 1))
                if here[0] + offset < 0 or here[-1] + offset >= rows:
                    offset = 0
                neighbors[slot, start:stop] = here + offset
                codes[slot, start:stop] = int(generator.integers(1, size))
        elif kind == 2:
            present = generator.random((slots, stop - start)) >= 0.25
            neighbors[:, start:stop] = np.where(present, generator.integers(0, rows, (slots, stop - start)), -1)
            codes[:, start:stop] = np.where(present, generator.integers(1, size, (slots, stop - start)),
                                            codes[:, start:stop])
            neighbors[:, min(start + 7, stop - 1)] = -1
        else:
            neighbors[0, start:stop] = here
            codes[0, start:stop] = 1
            for slot in range(1, slots):
                axis, reach = (slot - 1) % 3, 1 + (slot - 1) // 6
                sign = 1 if ((slot - 1) // 3) % 2 == 0 else -1
                target = here + sign * reach * strides[axis]
                valid = (target >= 0) & (target < rows) & (((here * 7 + slot) % 29) != 0)
                neighbors[slot, start:stop] = np.where(valid, target, -1)
                codes[slot, start:stop] = np.where(valid, 1 + (slot % (size - 1)), codes[slot, start:stop])
    return _metadata(neighbors.astype(np.int32), codes, palette)


def _nan_words(shape, tag):
    """NaN words with ``tag`` in bits 40 to 47 and the position below: the signs alternate, and every third one is
    signalling (quiet bit clear, mantissa not zero)."""
    index = np.arange(int(np.prod(shape)), dtype=np.uint64).reshape(shape)
    words = (np.uint64(0x7FF0000000000000) | (np.uint64(tag & 0xFF) << np.uint64(40))
             | ((index + np.uint64(1)) & np.uint64(0xFFFFFFFFFF)))
    quiet = (index % np.uint64(3)) != np.uint64(0)
    words = np.where(quiet, words | np.uint64(0x0008000000000000), words)
    return words | ((index & np.uint64(1)) << np.uint64(63))


def _sweep_nan_palette(metadata):
    """The stencil of ``metadata`` with every coefficient a NaN of its own payload, quiet and signalling."""
    palette = np.array(metadata.coefficient_palette, dtype=np.float64)
    palette.view(np.uint64)[...] = _nan_words(palette.shape, 0xA7)
    return _metadata(metadata.neighbors, metadata.coefficient_codes, palette)


def _positive_palette(metadata):
    """The stencil of ``metadata`` with every coefficient positive and the NaN at palette[0] kept.  With every value
    -0.0 each product of the slot sum is -0.0, so the sign of the sum, +0.0, says that it started from +0.0 as the
    former loop's does."""
    palette = np.abs(np.array(metadata.coefficient_palette, dtype=np.float64))
    palette.view(np.uint64)[0] = TRAP
    return _metadata(metadata.neighbors, metadata.coefficient_codes, palette)


# ----------------------------------------------------------------------------------------------------------------
# Operands and launch paths, shared with benchmarks/stencil_pipelined_check.py
# ----------------------------------------------------------------------------------------------------------------

class Operands:
    """Device operands of the launches of one stencil: data of one ``kind`` (of :data:`KINDS`, or
    :data:`SIGNED_ZEROS`) in every layout.

    Arrays have ``columns`` columns, enough for the blocks of the operators' methods.  ``projectors`` (the CSR
    arrays of a real operator on the device) and ``projector_count`` replace the seven synthetic projectors.
    """

    def __init__(self, cp, rows, kind, seed, projectors=None, projector_count=None, columns=14):
        if kind not in KINDS + (SIGNED_ZEROS,):
            raise ValueError(kind)
        generator = np.random.default_rng(seed)
        self.cp, self.rows, self.kind, self.columns = cp, rows, kind, columns

        def data(shape):
            if kind == "wide":
                values = np.ldexp(generator.uniform(1.0, 2.0, shape), generator.integers(-30, 31, shape))
                return np.where(generator.random(shape) < 0.5, -values, values)
            return generator.uniform(-1.0, 1.0, shape)

        current, previous = data((rows, columns)), data((rows, columns))
        wide, previous_wide = data((rows, 2 * columns)), data((rows, 2 * columns))
        offset, previous_offset = data((rows + 11, columns)), data((rows + 11, columns))
        vector, previous_vector = data(rows), data(rows)
        potential = generator.uniform(-0.5, 0.5, rows)
        matrices = (current, previous, wide, previous_wide, offset, previous_offset)
        if projectors is None:
            # Seven projectors on runs of rows; most rows have none.
            count, entries = 7, []
            for projector in range(count):
                centre, reach = int(generator.integers(0, rows)), int(generator.integers(2, 80))
                near = np.arange(max(0, centre - reach), min(rows, centre + reach))
                near = near[generator.random(near.size) < 0.6]
                entries.append((near, np.full(near.size, projector), generator.standard_normal(near.size)))
            near, owner, values = (np.concatenate(items) for items in zip(*entries))
            matrix = sp.csr_matrix((values, (near, owner)), shape=(rows, count))
            matrix.sum_duplicates()
            matrix.sort_indices()
            host_projectors = (matrix.indptr.astype(np.int32), matrix.indices.astype(np.int32), matrix.data.copy())
            device_projectors = None
        else:
            count = int(projector_count)
            host_projectors, device_projectors = None, projectors
        coefficients = generator.standard_normal((count, columns))
        table = generator.uniform(-2.0, 2.0, (4, 4))
        scalars = generator.uniform(-2.0, 2.0, 4)
        if kind == "specials":
            # Rows of signed zeros and subnormals, whose products and sums stay subnormal; a few infinities (their
            # neighbours' sums are infinite or NaN, from inf - inf); signed zeros in the potential.
            tiny = np.array([0.0, -0.0, 5e-324, -5e-324, 1e-310, -2.2250738585072e-308, 3e-320, -1e-320])
            for array in matrices:
                count_rows = max(1, array.shape[0] // 50)
                at = generator.integers(0, array.shape[0], count_rows)
                array[at] = generator.choice(tiny, (count_rows, array.shape[1]))
                for _ in range(3):
                    array[int(generator.integers(0, array.shape[0])), int(generator.integers(0, array.shape[1]))] = \
                        generator.choice((np.inf, -np.inf))
            potential[generator.integers(0, rows, max(1, rows // 50))] = -0.0
            vector[generator.integers(0, rows, max(1, rows // 50))] = 5e-324
            previous_vector[generator.integers(0, rows, max(1, rows // 50))] = -0.0
        elif kind == "nan":
            # Quiet and signalling NaNs of several payloads and both signs, scattered.
            payloads = np.array([0x7FF8000000000001, 0xFFF8000000000ABC, 0x7FF4000000000000, 0x7FFFFFFFFFFFFFFF,
                                 0xFFF0000000000001, 0x7FF800000000BEEF], dtype=np.uint64)
            for array in matrices:
                count_nan = max(1, array.size // 2000)
                at = generator.integers(0, array.shape[0], count_nan), generator.integers(0, array.shape[1], count_nan)
                array.view(np.uint64)[at] = generator.choice(payloads, count_nan)
            potential.view(np.uint64)[int(generator.integers(0, rows))] = np.uint64(0xFFF800000000CAFE)
            vector.view(np.uint64)[int(generator.integers(0, rows))] = np.uint64(0x7FF800000000F00D)
            if coefficients.size:
                coefficients.view(np.uint64)[0, min(2, columns - 1)] = np.uint64(0x7FF8000000001234)
        elif kind == "twonan":
            # Two NaNs of different payloads meet in one operation.  In every array of vectors pairs of them in
            # neighbouring rows of one column, each column in turn: the stencil sum of a row near both meets the
            # two.  Where the vectors have one, Y_{k-1} and the potential have another, and so has one projector's
            # coefficient in every column: the recurrence, potential and scatter terms meet two as well.
            pairs = max(1, rows // 300)
            for number, array in enumerate(matrices):
                words = array.view(np.uint64)
                at = generator.integers(0, max(1, array.shape[0] - 1), pairs)
                for index, row in enumerate(at):
                    column = index % array.shape[1]
                    words[row, column] = np.uint64(0x7FF8000000000000 | (number << 16) | (2 * index + 1))
                    words[min(row + 1, array.shape[0] - 1), column] = np.uint64(
                        0xFFF8000000000000 | (number << 16) | (2 * index + 2))
                    if array is current:
                        previous.view(np.uint64)[row, column] = np.uint64(0x7FF4000000000000 | (index + 1))
                        potential.view(np.uint64)[row] = np.uint64(0xFFF8000000000000 | (0xCAFE0000 + index))
            vector.view(np.uint64)[generator.integers(0, rows, pairs)] = np.uint64(0x7FF800000000F00D)
            previous_vector.view(np.uint64)[generator.integers(0, rows, pairs)] = np.uint64(0xFFF800000000D00F)
            if coefficients.size:
                for column in range(columns):
                    coefficients.view(np.uint64)[column % count, column] = np.uint64(
                        0x7FF8000000000000 | (0xC0E0000 + column))
        elif kind == "allnan":
            # Every value a quiet NaN of its own payload (array and position, both signs): every operation of a
            # kernel then meets two or three NaNs, and the payload of a result says which operand won each one.
            for number, array in enumerate((*matrices, vector, previous_vector, potential)):
                array.view(np.uint64)[...] = nan_pattern(array.shape, number + 1)
            coefficients.view(np.uint64)[...] = nan_pattern(coefficients.shape, 11)
            table.view(np.uint64)[...] = nan_pattern(table.shape, 12)
            scalars.view(np.uint64)[...] = nan_pattern(scalars.shape, 13)
        elif kind == SIGNED_ZEROS:
            # Every value -0.0 and the potential positive; centre and sigma negative, scale and sigma_next positive,
            # so that every step after the slot sum keeps the sign of a zero (subtracting +0.0 adds -0.0).  On a
            # stencil of positive coefficients (_positive_palette) every product of the slot sum is -0.0 and the
            # sum +0.0, from the +0.0 the former loop starts with: a sum started from -0.0 would stay -0.0.
            for array in (*matrices, vector, previous_vector):
                array[...] = -0.0
            potential[...] = generator.uniform(0.1, 0.5, rows)
            coefficients[...] = -0.0
            table[...] = (-0.7, 0.9, -0.4, 1.3)
            scalars[...] = (-0.7, 0.9, -0.4, 1.3)
        upload = lambda array: cp.asarray(np.asfortranarray(array))
        self.current, self.previous = upload(current), upload(previous)
        self.wide, self.previous_wide = upload(wide), upload(previous_wide)
        self.offset, self.previous_offset = upload(offset), upload(previous_offset)
        self.vector, self.previous_vector = cp.asarray(vector), cp.asarray(previous_vector)
        self.potential = cp.asarray(potential)
        if device_projectors is None:
            values = host_projectors[2]
            if kind == "allnan":
                values.view(np.uint64)[...] = nan_pattern(values.shape, 10)
            self.projectors = tuple(cp.asarray(array) for array in (*host_projectors[:2], values))
        else:
            self.projectors = device_projectors
            if kind == "allnan":
                values = cp.asnumpy(device_projectors[2]).copy()
                values.view(np.uint64)[...] = nan_pattern(values.shape, 10)
                self.projectors = (device_projectors[0], device_projectors[1], cp.asarray(values))
        self.projector_count = count
        self.coefficients = cp.asarray(np.asfortranarray(coefficients))
        self.coefficients_c = cp.asarray(np.ascontiguousarray(coefficients))
        self.table = cp.asarray(table)
        self.scalars = tuple(float(value) for value in scalars)
        self.step = int(generator.integers(0, 4))

    def view(self, layout, width, previous=False):
        """The vectors of a launch of ``width`` columns in ``layout``, or the Y_{k-1} beside them."""
        cp = self.cp
        base = self.previous if previous else self.current
        if layout == "F":
            return base[:, :width]
        if layout == "C":
            return cp.ascontiguousarray(base[:, :width])
        if layout == "colstride":
            return (self.previous_wide if previous else self.wide)[:, 1:2 * width:2]
        if layout == "negcol":
            return base[:, ::-1][:, :width]
        if layout == "rowrev":
            return base[::-1, :width]
        if layout == "bcast":
            return cp.broadcast_to((self.previous_vector if previous else self.vector)[:, None], (self.rows, width))
        if layout == "offset":
            return (self.previous_offset if previous else self.offset)[5:5 + self.rows, :width]
        raise ValueError(layout)

    def output(self, layout, width):
        """(the whole array to compare, the view a launch writes), prefilled with :data:`FILL`."""
        cp, rows = self.cp, self.rows

        def filled(shape, order="F"):
            array = cp.empty(shape, dtype=cp.float64, order=order)
            array.view(cp.uint64).fill(FILL)
            return array

        if layout == "C":
            whole = filled((rows, width), "C")
            return whole, whole
        if layout == "colstride":
            whole = filled((rows, 2 * width))
            return whole, whole[:, ::2]
        if layout == "negcol":
            whole = filled((rows, width))
            return whole, whole[:, ::-1]
        if layout == "offset":
            whole = filled((rows + 11, width))
            return whole, whole[5:5 + rows, :]
        whole = filled((rows, width))
        return whole, whole

    def coefficient_view(self, layout, width):
        return (self.coefficients_c if layout == "C" else self.coefficients)[:, :width]


def _strides(array):
    return np.int64(array.strides[0] // 8), np.int64(array.strides[1] // 8)


def launch_recurrence(kernel, stencil, operands, *, slot_count, width, layout, add_nonlocal, add_previous,
                      use_parameters, output):
    """One launch of a ``stencil_major_chebyshev6`` kernel on the arrays of ``stencil``, as production makes it."""
    current = operands.view(layout, width)
    previous = operands.view(layout, width, previous=True) if add_previous else current
    if add_nonlocal:
        offsets, columns, values = operands.projectors
        coefficients = operands.coefficient_view(layout, width)
    else:
        offsets, columns, values = stencil.neighbors, stencil.neighbors, stencil.coefficient_palette
        coefficients = current
    center, scale, sigma, sigma_next = operands.scalars
    rows = int(stencil.shape[0])
    kernel(((rows + THREADS - 1) // THREADS,), (THREADS,), (
        np.int64(rows), np.int32(slot_count), stencil.neighbors, stencil.coefficient_codes,
        stencil.coefficient_palette, operands.potential, offsets, columns, values, coefficients,
        *_strides(coefficients), np.int32(add_nonlocal), current, *_strides(current), previous, *_strides(previous),
        np.int32(add_previous), np.int32(width), np.float64(center), np.float64(scale), np.float64(sigma),
        np.float64(sigma_next), operands.table if use_parameters else stencil.coefficient_palette,
        np.int32(operands.step if use_parameters else 0), np.int32(use_parameters), output, *_strides(output)))


def recurrence_paths(slots):
    """Every launch of the recurrence kernel to compare: options of :func:`launch_recurrence`.

    Widths 1 to 6 with and without the scatter, Y_{k-1} and the table, in every layout; then six columns with slot
    counts below the compiled one (the guard of the fast path) in the layouts of unit row stride.
    """
    found = []
    flags = [(nonlocal_term, previous, table) for nonlocal_term in (0, 1) for previous in (0, 1) for table in (0, 1)]
    for layout in LAYOUTS:
        for width in range(1, 7):
            for add_nonlocal, add_previous, use_parameters in flags:
                found.append(dict(slot_count=slots, width=width, layout=layout, add_nonlocal=add_nonlocal,
                                  add_previous=add_previous, use_parameters=use_parameters))
    for fewer in sorted({slots - 1, 1} - {0, slots}):
        for layout in ("F", "colstride", "negcol", "offset"):
            for add_nonlocal, add_previous, use_parameters in flags:
                found.append(dict(slot_count=fewer, width=6, layout=layout, add_nonlocal=add_nonlocal,
                                  add_previous=add_previous, use_parameters=use_parameters))
    return found


def sweep_paths(slots):
    """The launches of the sweep, options of :func:`launch_recurrence`: every launch that the fast path takes (six
    columns of unit row stride in the five layouts that have it, with and without the scatter, Y_{k-1} and the
    table) and a few that it does not, which the generated kernel runs through its former loop (six columns in C
    order and with reversed rows, and one slot fewer)."""
    flags = [(nonlocal_term, previous, table) for nonlocal_term in (0, 1) for previous in (0, 1) for table in (0, 1)]
    found = [dict(slot_count=slots, width=6, layout=layout, add_nonlocal=add_nonlocal, add_previous=add_previous,
                  use_parameters=use_parameters)
             for layout in ("F", "colstride", "negcol", "bcast", "offset")
             for add_nonlocal, add_previous, use_parameters in flags]
    found += [dict(slot_count=slots, width=6, layout=layout, add_nonlocal=add_nonlocal, add_previous=add_previous,
                   use_parameters=use_parameters)
              for layout in ("C", "rowrev") for add_nonlocal, add_previous, use_parameters in flags[::3]]
    if slots > 1:
        found += [dict(slot_count=slots - 1, width=6, layout="F", add_nonlocal=add_nonlocal, add_previous=add_previous,
                       use_parameters=use_parameters) for add_nonlocal, add_previous, use_parameters in flags[::3]]
    return found


def fast_path_taken(options, slots):
    """Whether a launch of :func:`recurrence_paths` takes the fast path of the kernel generated for ``slots``."""
    row_stride = {"F": 1, "colstride": 1, "negcol": 1, "bcast": 1, "offset": 1}.get(options["layout"], -1)
    if options["layout"] == "C":
        row_stride = options["width"]
    return takes_fast_path(options["slot_count"], options["width"], row_stride, slots)


def compare_recurrence_kernels(cp, former, pipelined, operands, paths=None):
    """Launch every path of :func:`recurrence_paths` (or ``paths``) with ``former.recurrence_kernel`` and
    ``pipelined.recurrence_kernel`` and count the 64-bit words of the whole outputs that differ.

    Returns ``launches``, ``fast_launches`` (those the fast path takes), ``differing_words``,
    ``differing_not_as_two_nans`` (words that differ other than as two NaNs that both kernels wrote: a number on one
    side, two numbers, or a NaN against the prefilled word), ``nan_words`` (NaN results of the former kernel, NaN
    payloads compared), ``differing`` (the paths that differ, at most 20) and ``differing_not_as_two_nans_paths``
    (the paths of the second count, at most 20).
    """
    slots = int(former.slot_count)
    if int(pipelined.slot_count) != slots:
        raise RuntimeError("the two operators hold stencils of different slot counts")
    counts, not_payloads, nans, keys = [], [], [], []
    for options in recurrence_paths(slots) if paths is None else paths:
        wholes = []
        for stencil in (former, pipelined):
            whole, target = operands.output(options["layout"], options["width"])
            launch_recurrence(stencil.recurrence_kernel, stencil, operands, output=target, **options)
            wholes.append(whole)
        a, b = (whole.view(cp.uint64) for whole in wholes)
        both = cp.isnan(wholes[0]) & cp.isnan(wholes[1])
        written = both & (a != FILL) & (b != FILL)
        counts.append(cp.count_nonzero(a != b))
        not_payloads.append(cp.count_nonzero((a != b) & ~written))
        nans.append(cp.count_nonzero(both & (a != FILL)))
        keys.append(options)
    cp.cuda.Device().synchronize()
    counts, not_payloads, nans = (cp.asnumpy(cp.stack(items)).astype(np.int64) for items in (counts, not_payloads, nans))
    return dict(launches=len(keys), fast_launches=sum(fast_path_taken(options, slots) for options in keys),
                differing_words=int(counts.sum()), differing_not_as_two_nans=int(not_payloads.sum()),
                nan_words=int(nans.sum()),
                differing=[dict(options, words=int(count)) for options, count in zip(keys, counts) if count][:20],
                differing_not_as_two_nans_paths=[dict(options, words=int(count))
                                                 for options, count in zip(keys, not_payloads) if count][:20])


class Counted:
    """A recurrence kernel that records (width, row stride, slot count) of every launch and passes it on."""

    _WIDTH, _ROW_STRIDE, _SLOTS = (_RECURRENCE_PARAMETERS.index(name)
                                   for name in ("width", "current_row_stride", "slot_count"))

    def __init__(self, kernel):
        self.kernel, self.launches = kernel, []

    def __call__(self, grid, block, arguments, *rest, **options):
        self.launches.append((int(arguments[self._WIDTH]), int(arguments[self._ROW_STRIDE]),
                              int(arguments[self._SLOTS])))
        return self.kernel(grid, block, arguments, *rest, **options)

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def count_launches(stencil):
    """Put :class:`Counted` in front of the two recurrence kernels of ``stencil``; returns (fast, former)."""
    fast = Counted(stencil.recurrence_kernel)
    former = fast if stencil.former_recurrence_kernel is stencil.recurrence_kernel else Counted(
        stencil.former_recurrence_kernel)
    stencil.recurrence_kernel, stencil.former_recurrence_kernel = fast, former
    return fast, former


OPERATOR_WIDTHS = (1, 2, 5, 6, 8, 14)
OPERATOR_LAYOUTS = ("F", "C", "colstride", "offset")


def compare_operator_methods(cp, former, pipelined, operands, layouts=OPERATOR_LAYOUTS, widths=OPERATOR_WIDTHS,
                             apply=True):
    """``chebyshev_recurrence`` and ``apply`` of both stencils on blocks of :data:`OPERATOR_WIDTHS` columns in
    :data:`OPERATOR_LAYOUTS` (or ``widths`` and ``layouts``; ``apply`` False leaves ``apply`` out), with and without
    the scatter, Y_{k-1}, the table and the potential: the launches of production, a block cut into launches of six
    columns and a narrower rest.  Counts the differing words as :func:`compare_recurrence_kernels`; with
    ``count_launches`` in place on ``pipelined`` it also tells which kernel took which launch."""
    counts, not_payloads, nans, keys = [], [], [], []
    center, scale, sigma, sigma_next = operands.scalars
    for layout in layouts:
        for width in widths:
            for nonlocal_term in (0, 1):
                scatter = (dict(projector_data=operands.projectors,
                                projector_coefficients=operands.coefficient_view(layout, width))
                           if nonlocal_term else {})
                launches = []
                for previous in (0, 1):
                    for table in (0, 1):
                        launches.append((("chebyshev6", layout, width, nonlocal_term, previous, table), dict(
                            method="chebyshev_recurrence", previous=previous, table=table)))
                for local in (0, 1) if apply else ():
                    launches.append((("spmm6", layout, width, nonlocal_term, local), dict(method="apply", local=local)))
                for key, how in launches:
                    wholes = []
                    for stencil in (former, pipelined):
                        whole, target = operands.output(layout, width)
                        current = operands.view(layout, width)
                        if how["method"] == "apply":
                            stencil.apply(current, operands.potential if how["local"] else None, output=target,
                                          **scatter)
                        else:
                            stencil.chebyshev_recurrence(
                                current, operands.potential, center=center, scale=scale, sigma_next=sigma_next,
                                previous=operands.view(layout, width, previous=True) if how["previous"] else None,
                                sigma=sigma, output=target,
                                recurrence_parameters=operands.table if how["table"] else None,
                                parameter_step=operands.step if how["table"] else 0, **scatter)
                        wholes.append(whole)
                    a, b = (whole.view(cp.uint64) for whole in wholes)
                    both = cp.isnan(wholes[0]) & cp.isnan(wholes[1])
                    written = both & (a != FILL) & (b != FILL)
                    counts.append(cp.count_nonzero(a != b))
                    not_payloads.append(cp.count_nonzero((a != b) & ~written))
                    nans.append(cp.count_nonzero(both & (a != FILL)))
                    keys.append(key)
    cp.cuda.Device().synchronize()
    counts, not_payloads, nans = (cp.asnumpy(cp.stack(items)).astype(np.int64) for items in (counts, not_payloads, nans))
    return dict(calls=len(keys), differing_words=int(counts.sum()), differing_not_as_two_nans=int(not_payloads.sum()),
                nan_words=int(nans.sum()),
                differing=[dict(key=key, words=int(count)) for key, count in zip(keys, counts) if count][:20],
                differing_not_as_two_nans_paths=[dict(key=key, words=int(count))
                                                 for key, count in zip(keys, not_payloads) if count][:20])


def _report(line):
    """A summary line for the A100 log."""
    print("STENCIL-PIPELINE " + line, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Host tests
# ----------------------------------------------------------------------------------------------------------------

class _SourceKernel:
    """A raw kernel that compiles to nothing and keeps the source and options it was given."""

    def __init__(self, source, name, options=()):
        self.code, self.name, self.options = source, name, tuple(options)

    def compile(self):
        pass


class _Launches:
    """A raw kernel that records its launches."""

    def __init__(self):
        self.calls = []

    def __call__(self, grid, block, arguments):
        self.calls.append((tuple(grid), tuple(block), tuple(arguments)))


class _CompileError(Exception):
    """What NVRTC raises through CuPy, here."""


def _compile_failing_for_generated_sources(attempts):
    """A compile function that records the source of every kernel it is given and raises for a generated one."""

    def compile_raw(kernel):
        attempts.append(kernel.code)
        if "__launch_bounds__" in kernel.code:
            raise _CompileError("nvrtc: error: injected failure of the generated kernel\nthe rest of the log")
        kernel.compile()

    return compile_raw


def _layout(call):
    """The scalars of a recorded launch and the type, shape, strides and dtype of its arrays."""
    return [(type(item), item.shape, item.strides, item.dtype) if isinstance(item, np.ndarray) else (type(item), item)
            for item in call[2]]


@contextmanager
def _host_kernels():
    """``_HostCuPy`` kernels that keep their sources; the kernel caches emptied meanwhile and restored after."""
    with patch.object(mixed_tests._HostCuPy, "RawKernel", _SourceKernel), \
         patch.dict(cupy_stencil_major._KERNEL_CACHE, clear=True), patch.dict(implicit_stencil._CACHE, clear=True):
        yield


def _host_stencil(value, tile, metadata):
    with _switch(value), _host_kernels():
        return CuPyStencilMajorFiniteDifference(mixed_tests._HostCuPy(), metadata=metadata, implicit_tile=tile)


class StencilSourceTests(unittest.TestCase):
    def test_former_sources_are_those_before_the_switch(self):
        self.assertEqual(_digest(_CUDA_SOURCE), _FORMER_DIGESTS[0])
        for tile in _TILES:
            with self.subTest(tile=tile):
                self.assertEqual(_digest(implicit_source(tile)), _FORMER_DIGESTS[tile])
        # ``former`` returns them as they are, whatever the slot count.
        for slots in (1, 3, 25, 37):
            self.assertIs(stencil_source(0, slots, "former"), _CUDA_SOURCE)
            self.assertIs(stencil_source(0, slots), _CUDA_SOURCE)
            for tile in _TILES:
                self.assertEqual(stencil_source(tile, slots, "former"), implicit_source(tile))
        # The FP32 filter derives its kernels from the former slot-major source.
        self.assertEqual(_digest(cupy_mixed_precision._SOURCE), _FP32_DIGEST)

    def test_switch_values(self):
        for value in (None, "", "  ", "\t\n"):
            with self.subTest(value=value), _switch(value):
                self.assertEqual(stencil_kernel_setting(), DEFAULT)
        for value in ("former", " Former ", "FORMER\n"):
            with self.subTest(value=value), _switch(value):
                self.assertEqual(stencil_kernel_setting(), "former")
        for value in ("pipelined", " PIPELINED ", "Pipelined\t"):
            with self.subTest(value=value), _switch(value):
                self.assertEqual(stencil_kernel_setting(), "pipelined")
        for value in ("fast", "pipeline", "pipelined2", "0", "1", "auto", "former,pipelined"):
            with self.subTest(value=value), _switch(value), self.assertRaises(ValueError) as caught:
                stencil_kernel_setting()
            for word in (SWITCH, "former", "pipelined"):
                self.assertIn(word, str(caught.exception))

    def test_the_default_is_the_pipelined_kernel_and_the_serial_control_keeps_the_former_one(self):
        # The value an unset or empty variable gives; ``former`` names the kernels of before in either case.
        from parsec_python.acceleration.benchmarks import mpi_full_scf

        self.assertEqual(DEFAULT, "pipelined")
        self.assertIn(DEFAULT, stencil_pipeline.KERNELS)
        # The serial control of the MPI runner keeps the former kernels unless its launcher names a value.
        self.assertEqual(mpi_full_scf._SERIAL_CONTROL_SETTINGS[SWITCH], "former")
        for named, expected in (({}, "former"), ({SWITCH: "pipelined"}, "pipelined")):
            environment = dict(named)
            mpi_full_scf._keep_former_routes(environment)
            self.assertEqual(environment[SWITCH], expected)

    def test_the_slot_loop_is_the_anchor_of_both_kernels(self):
        anchor = stencil_pipeline._LOOP
        self.assertEqual(anchor, "    for (int slot = 0; slot < slot_count; ++slot) {\n")
        for tile in (0,) + _TILES:
            with self.subTest(tile=tile):
                self.assertEqual(stencil_source(tile, 25).count(anchor), 2)
                self.assertEqual(stencil_source(tile, 25, "pipelined").count(anchor), 2)
        at = _CUDA_SOURCE.index(anchor)
        once = _CUDA_SOURCE[:at] + _CUDA_SOURCE[at + len(anchor):]
        for changed in (once, _CUDA_SOURCE + anchor, _CUDA_SOURCE.replace(anchor, anchor.replace("++slot", "slot++"))):
            with self.assertRaisesRegex(RuntimeError, "production stencil ABI changed"):
                pipelined_source(changed, 25, 0)
        for slots in (0, -1):
            with self.assertRaises(ValueError):
                pipelined_source(_CUDA_SOURCE, slots, 0)

    def test_the_fast_path_is_in_the_recurrence_kernel_alone(self):
        accumulators = "    double accumulator[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};\n"
        bounds = f"void __launch_bounds__({THREADS}, 4) {KERNEL}("
        for tile in (0,) + _TILES:
            for slots in (1, 2, 3, 13, 25, 37):
                with self.subTest(tile=tile, slots=slots):
                    former, pipelined = stencil_source(tile, slots, "former"), stencil_source(tile, slots, "pipelined")
                    guard = f"if (slot_count == {slots} && width == {WIDTH} && current_row_stride == 1) {{"
                    self.assertEqual(pipelined.count(guard), 1)
                    self.assertNotIn(guard, former)
                    self.assertNotIn("vector_row_stride == 1", pipelined)
                    self.assertEqual(pipelined.count("__launch_bounds__"), 1)
                    self.assertEqual(pipelined.count(bounds), 1)
                    self.assertEqual(pipelined.count("    } else\n" + stencil_pipeline._LOOP), 1)
                    # The hints: the potential, six columns of Y_{k-1} and the projector row offsets.
                    self.assertEqual(pipelined.count("prefetch.global.L2"), 8)
                    self.assertNotIn("__ldg", pipelined)
                    # stencil_major_spmm6 is the former text to the byte, up to the declaration of the next kernel.
                    start = former.index("void stencil_major_spmm6(")
                    stop = former.index(f"void {KERNEL}(")
                    self.assertIn(former[start:stop], pipelined)
                    self.assertEqual(pipelined.index("void stencil_major_spmm6("), start)
                    # One block in front of the slot loop of the recurrence kernel and the launch bounds are all
                    # that differ: the former loop, its epilogue and everything else are the former text.
                    block = stencil_pipeline._fast_path(slots, tile)
                    self.assertEqual(pipelined.count(accumulators + block + stencil_pipeline._LOOP), 1)
                    rebuilt = pipelined.replace(block, "", 1).replace(bounds, f"void {KERNEL}(", 1)
                    self.assertEqual(rebuilt, former)
                    # Both kernels declare what they declared.
                    plain = pipelined.replace(f"__launch_bounds__({THREADS}, 4) ", "")
                    for kernel in _KERNELS:
                        self.assertEqual(kernel_parameters(plain, kernel), kernel_parameters(former, kernel))

    def test_the_generated_source_is_the_one_verified_on_the_a100(self):
        for (tile, slots), digest in _VERIFIED_DIGESTS.items():
            with self.subTest(tile=tile, slots=slots):
                self.assertEqual(_digest(stencil_source(tile, slots, "pipelined")), digest)

    def test_the_fast_path_visits_the_slots_in_order_one_slot_ahead(self):
        for tile, slots in ((16, 25), (0, 25), (32, 13), (0, 3), (64, 37), (16, 1)):
            with self.subTest(tile=tile, slots=slots):
                text = stencil_pipeline._fast_path(slots, tile)
                self.assertTrue(text.endswith("    } else\n"))
                # The tile descriptor once, one record and one code per slot.
                self.assertEqual(text.count("neighbors[row / "), 1 if tile else 0)
                self.assertEqual(text.count("meta_n["), slots)
                self.assertEqual(text.count("meta_c["), slots)
                records = [text.index(f"const int e{slot} = meta_n[") for slot in range(slots)]
                values = [text.index(f"const int r{slot} = ") for slot in range(slots)]
                sums = [text.index(f"if (p{slot}) {{") for slot in range(slots)]
                for slot in range(slots):
                    self.assertLess(records[slot], values[slot])
                    self.assertLess(values[slot], sums[slot])
                    if slot + 1 < slots:
                        # The record of the next slot before the values of this one, the values of the next slot
                        # before the sums of this one, and the sums in slot order.
                        self.assertLess(records[slot + 1], values[slot])
                        self.assertLess(values[slot + 1], sums[slot])
                        self.assertLess(sums[slot], sums[slot + 1])
                    # A missing neighbour adds nothing: its six sums are predicated, in the former expression.
                    block = text[sums[slot]:]
                    for column in range(6):
                        line = f"            accumulator[{column}] += w{slot} * x{slot}_{column};"
                        self.assertEqual(text.count(line), 1)
                        self.assertLess(block.index(line), block.index("        }"))
                # The hints for the epilogue follow the values of slot S/2.
                hint = text.index("prefetch.global.L2")
                self.assertLess(text.index(f"const double x{slots // 2}_5 = "), hint)
                if slots // 2 + 1 < slots:
                    self.assertLess(hint, values[slots // 2 + 1])

    def test_the_host_choice_of_the_kernel_is_the_guard_of_the_fast_path(self):
        for slots in (1, 3, 25):
            guard = f"slot_count == {slots} && width == {WIDTH} && current_row_stride == 1"
            self.assertIn(f"if ({guard})", stencil_source(16, slots, "pipelined"))
            for slot_count in (slots - 1, slots, slots + 1):
                for width in range(0, 9):
                    for row_stride in (-6, -1, 0, 1, 2, 6, 14):
                        expected = slot_count == slots and width == 6 and row_stride == 1
                        self.assertEqual(takes_fast_path(slot_count, width, row_stride, slots), expected)
                        self.assertEqual(takes_fast_path(np.int32(slot_count), np.int32(width), np.int64(row_stride),
                                                         slots), expected)


class StencilLaunchTests(unittest.TestCase):
    """Which kernel takes which launch, with NumPy in place of CuPy."""

    def setUp(self):
        self.metadata = build_stencil_major_metadata(_irregular_operator(1031))
        generator = np.random.default_rng(3)
        self.wide = np.asfortranarray(generator.standard_normal((1031, 28)))
        self.potential = generator.standard_normal(1031)
        projectors = sp.csr_matrix(np.where(generator.random((1031, 2)) < 0.1, 1.0, 0.0))
        self.data = (projectors.indptr.astype(np.int32), projectors.indices.astype(np.int32), projectors.data)
        self.coefficients = np.asfortranarray(generator.standard_normal((2, 28)))

    def blocks(self):
        """Blocks of the recurrence: (name, current, the launches of six columns with unit row stride)."""
        wide = self.wide
        return (
            ("fourteen columns", wide[:, :14], 2),
            ("fourteen in C order", np.ascontiguousarray(wide[:, :14]), 0),
            ("every other column", wide[:, ::2], 2),
            ("six columns", wide[:, 3:9], 1),
            ("six in C order", np.ascontiguousarray(wide[:, :6]), 0),
            ("six with reversed rows", wide[::-1, :6], 0),
            ("five columns", wide[:, :5], 0),
            ("one vector", wide[:, 7], 0),
            ("twelve", wide[:, 2:14], 2),
        )

    def run_blocks(self, stencil):
        """One recurrence step of every block of :meth:`blocks`, with Y_{k-1} and the scatter."""
        for _name, current, _fast in self.blocks():
            width = 1 if current.ndim == 1 else current.shape[1]
            coefficients = self.coefficients[:, 0] if current.ndim == 1 else self.coefficients[:, :width]
            stencil.chebyshev_recurrence(current, self.potential, center=0.5, scale=0.25, sigma_next=1.5,
                                         previous=current, sigma=0.4, projector_data=self.data,
                                         projector_coefficients=coefficients)

    def test_launches_that_do_not_take_the_fast_path_go_to_the_former_kernel(self):
        def launch(call, name):
            return int(call[2][_RECURRENCE_PARAMETERS.index(name)])

        for value in ("former", "pipelined"):
            for tile in (0, 16):
                with self.subTest(value=value, tile=tile):
                    stencil = _host_stencil(value, tile, self.metadata)
                    if value == "former":
                        self.assertIs(stencil.recurrence_kernel, stencil.former_recurrence_kernel)
                        self.assertIsNone(stencil.fast_path_slots)
                    else:
                        self.assertIsNot(stencil.recurrence_kernel, stencil.former_recurrence_kernel)
                        self.assertEqual(stencil.fast_path_slots, 3)
                    fast, former = _Launches(), _Launches()
                    stencil.apply_kernel, stencil.recurrence_kernel, stencil.former_recurrence_kernel = (
                        _Launches(), fast, former)
                    self.run_blocks(stencil)
                    every = fast.calls + former.calls
                    # 3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 2 launches of at most six columns.
                    self.assertEqual(len(every), 16)
                    for grid, block, arguments in every:
                        self.assertEqual((grid, block), ((5,), (THREADS,)))
                        self.assertEqual(len(arguments), len(_RECURRENCE_PARAMETERS))
                    if value == "former":
                        # Every launch takes the one kernel, as before.
                        self.assertEqual(former.calls, [])
                        continue
                    self.assertEqual(len(fast.calls), sum(count for _name, _current, count in self.blocks()))
                    for call in fast.calls:
                        self.assertEqual((launch(call, "width"), launch(call, "current_row_stride"),
                                          launch(call, "slot_count")), (6, 1, 3))
                    for call in former.calls:
                        self.assertFalse(takes_fast_path(launch(call, "slot_count"), launch(call, "width"),
                                                         launch(call, "current_row_stride"), 3))
                    # The loop of launches chooses as recurrence_kernel_for does, launch by launch.
                    for kernel in (fast, former):
                        for call in kernel.calls:
                            self.assertIs(stencil.recurrence_kernel_for(launch(call, "width"),
                                                                        launch(call, "current_row_stride")), kernel)
                    # A kernel generated for another slot count than that of the stencil never takes its fast path
                    # there: the stencil holds the former kernel alone, and every launch goes to it.
                    stencil.hold_kernels(stencil_pipeline.StencilKernels(
                        stencil.apply_kernel, fast, former, fast_path_slots=4, setting="pipelined"))
                    self.assertIs(stencil.recurrence_kernel, former)
                    self.assertIsNone(stencil.fast_path_slots)
                    self.assertEqual(stencil.stencil_kernel_fallback,
                                     "the generated kernel has the fast path of 4 slots, the stencil 3")
                    fast.calls.clear()
                    former.calls.clear()
                    self.run_blocks(stencil)
                    self.assertEqual((len(fast.calls), len(former.calls)), (0, 16))

    def test_the_order_of_the_launches_and_their_arguments_are_those_of_the_former_kernel(self):
        recorded = {}
        for value in ("former", "pipelined"):
            for tile in (0, 16):
                stencil = _host_stencil(value, tile, self.metadata)
                # One recorder behind both recurrence kernels keeps the order of the launches.
                launches = _Launches()
                stencil.apply_kernel = _Launches()
                stencil.recurrence_kernel = stencil.former_recurrence_kernel = launches
                self.run_blocks(stencil)
                stencil.apply(self.wide[:, :14], self.potential, projector_data=self.data,
                              projector_coefficients=self.coefficients[:, :14])
                recorded[value, tile] = launches.calls, stencil.apply_kernel.calls
        for tile in (0, 16):
            with self.subTest(tile=tile):
                for index in range(2):
                    former, pipelined = recorded["former", tile][index], recorded["pipelined", tile][index]
                    self.assertEqual([_layout(call) for call in pipelined], [_layout(call) for call in former])
                    for call in pipelined:
                        self.assertEqual(call[1], (256,))

    def test_an_operator_holds_the_former_kernels_and_the_generated_one_beside_them(self):
        for tile in (0, 16, 32):
            with self.subTest(tile=tile):
                former = _host_stencil("former", tile, self.metadata)
                pipelined = _host_stencil("pipelined", tile, self.metadata)
                base = stencil_source(tile, 3, "former")
                for stencil in (former, pipelined):
                    self.assertEqual((stencil.apply_kernel.code, stencil.apply_kernel.name),
                                     (base, "stencil_major_spmm6"))
                    self.assertEqual((stencil.former_recurrence_kernel.code, stencil.former_recurrence_kernel.name),
                                     (base, KERNEL))
                self.assertEqual((pipelined.recurrence_kernel.code, pipelined.recurrence_kernel.name),
                                 (stencil_source(tile, 3, "pipelined"), KERNEL))
                self.assertEqual(pipelined.recurrence_kernel.options, ("--std=c++11",))
                attributes = pipelined.stencil_kernel_attributes
                self.assertEqual(sorted(attributes), ["apply_kernel", "former_recurrence_kernel", "recurrence_kernel"])

    def test_former_cache_keys_are_unchanged_and_the_generated_kernel_is_kept_beside_them(self):
        metadata = self.metadata

        def build(value):
            with _switch(value), mixed_tests.host_cupy(), _host_kernels():
                operators = [CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                             finite_difference_metadata=metadata, implicit_tile=tile)
                             for tile in (16, 0)]
                return dict(implicit_stencil._CACHE), dict(cupy_stencil_major._KERNEL_CACHE), operators

        expected = {}
        for value in ("former", " FORMER "):
            with self.subTest(value=value):
                tiles, plain, operators = build(value)
                self.assertEqual(sorted(tiles), [(0, 16)])
                self.assertEqual(sorted(plain), [0])
                self.assertEqual([kernel.code for kernel in tiles[0, 16]], [implicit_source(16)] * 2)
                self.assertEqual([kernel.code for kernel in plain[0]], [_CUDA_SOURCE] * 2)
                for kernels in (tiles[0, 16], plain[0]):
                    self.assertEqual([(kernel.name, kernel.options) for kernel in kernels],
                                     [(name, ("--std=c++11",)) for name in _KERNELS])
                expected["former"] = sorted(tiles, key=repr), sorted(plain, key=repr)
        for value in ("pipelined", " Pipelined "):
            with self.subTest(value=value):
                tiles, plain, operators = build(value)
                self.assertEqual(sorted(tiles, key=repr), [(0, 16), (0, 16, "pipelined", 3)])
                self.assertEqual(sorted(plain, key=repr), [(0, "pipelined", 3), 0])
                self.assertEqual([kernel.code for kernel in tiles[0, 16]], [implicit_source(16)] * 2)
                self.assertEqual([kernel.code for kernel in plain[0]], [_CUDA_SOURCE] * 2)
                generated = tiles[0, 16, "pipelined", 3], plain[0, "pipelined", 3]
                self.assertEqual([(kernel.code, kernel.name, kernel.options) for kernel in generated],
                                 [(stencil_source(tile, 3, "pipelined"), KERNEL, ("--std=c++11",)) for tile in (16, 0)])
                for operator, former, kernel in zip(operators, (tiles[0, 16], plain[0]), generated):
                    stencil = operator.compact_finite_difference
                    self.assertIs(stencil.apply_kernel, former[0])
                    self.assertIs(stencil.former_recurrence_kernel, former[1])
                    self.assertIs(stencil.recurrence_kernel, kernel)
                expected["pipelined"] = sorted(tiles, key=repr), sorted(plain, key=repr)
        # An unset variable is the default.
        tiles, plain, _operators = build(None)
        self.assertEqual((sorted(tiles, key=repr), sorted(plain, key=repr)), expected[DEFAULT])

    def test_the_devices_of_a_sector_hold_the_kernels_of_their_owner(self):
        # The owner of a sector and the devices that filter beside it, as the sector group tests build them.
        metadata = build_stencil_major_metadata(_irregular_operator(40_000))
        expected = {
            ("former", "16"): ([(device, 16) for device in range(4)], []),
            ("pipelined", "16"): (sorted([(device, 16) for device in range(4)]
                                         + [(device, 16, "pipelined", 3) for device in range(4)], key=repr), []),
            ("former", "0"): ([], [0, 1, 2, 3]),
            ("pipelined", "0"): ([], sorted([0, 1, 2, 3] + [(device, "pipelined", 3) for device in range(4)],
                                            key=repr)),
        }
        expected[None, "16"], expected[None, "0"] = expected[DEFAULT, "16"], expected[DEFAULT, "0"]
        for (value, tile), keys in expected.items():
            with self.subTest(value=value, tile=tile):
                cp = implicit_tests._DevicesCuPy()
                with _switch(value, PARSEC_CUPY_IMPLICIT_TILE=tile), implicit_tests._devices(cp), \
                     patch.object(implicit_tests._DevicesCuPy, "RawKernel", _SourceKernel):
                    owner = CuPyHamiltonian(None, np.zeros(40_000), None, retain_generic_laplacian=False,
                                            finite_difference_metadata=metadata, implicit_tile=int(tile))
                    worker = owner._distributed_filter = DistributedFilter(owner, (0, 1, 2, 3))
                    found = (sorted(implicit_stencil._CACHE, key=repr),
                             sorted(cupy_stencil_major._KERNEL_CACHE, key=repr))
                self.assertEqual(found, keys)
                pipelined = (value or DEFAULT) == "pipelined"
                for held in [owner] + [worker.replicas[device] for device in (1, 2, 3)]:
                    stencil = held.compact_finite_difference
                    self.assertEqual(stencil.fast_path_slots, 3 if pipelined else None)
                    self.assertEqual(stencil.recurrence_kernel.code,
                                     stencil_source(int(tile), 3, "pipelined" if pipelined else "former"))
                    self.assertEqual(stencil.former_recurrence_kernel.code, stencil_source(int(tile), 3))
                    self.assertEqual(stencil.apply_kernel.code, stencil_source(int(tile), 3))

    def test_a_stencil_kernel_the_switch_does_not_know_is_an_error_of_the_operator(self):
        # Not a reason to give way to the slot-major layout or the CSR-order kernel.
        metadata = self.metadata
        for tile in (16, 0):
            with self.subTest(tile=tile), _switch("pipeline"), mixed_tests.host_cupy(), \
                 self.assertRaisesRegex(ValueError, SWITCH):
                CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                finite_difference_metadata=metadata, implicit_tile=tile)

    def test_a_generated_kernel_that_does_not_compile_gives_way_to_the_former_kernels(self):
        # Not to the slot-major layout or to the CSR-order kernel, which have other bits: the operator keeps its
        # layout and the former kernels, says why, and warns once per device, layout and slot count.
        metadata = self.metadata
        for tile, storage in ((16, "implicit_affine_tile_16"),
                              (0, "stencil_major_int32_neighbors_uint8_coefficient_palette")):
            with self.subTest(tile=tile):
                attempts = []
                compile_raw = _compile_failing_for_generated_sources(attempts)
                with _switch("pipelined"), mixed_tests.host_cupy(), _host_kernels(), \
                     patch.object(cupy_stencil_major, "compile_cupy_raw", compile_raw), \
                     patch.object(cupy_compile, "compile_cupy_raw", compile_raw), \
                     warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    operators = [CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                                 finite_difference_metadata=metadata, implicit_tile=tile)
                                 for _ in range(2)]
                    cache = dict(implicit_stencil._CACHE if tile else cupy_stencil_major._KERNEL_CACHE)
                layout = f"tiles of {tile} rows" if tile else "the slot-major layout"
                self.assertEqual([str(item.message).split(" (")[0] for item in caught], [
                    f"{SWITCH}=pipelined: the recurrence kernel generated for 3 slots in {layout} could not be built"])
                self.assertIs(caught[0].category, RuntimeWarning)
                self.assertIn("injected failure of the generated kernel", str(caught[0].message))
                # The generated source was compiled once, for both operators: the failure is kept under its key.
                generated = stencil_source(tile, 3, "pipelined")
                self.assertEqual((attempts.count(generated), attempts.count(stencil_source(tile, 3)), len(attempts)),
                                 (1, 2, 3))
                key = (0, tile, "pipelined", 3) if tile else (0, "pipelined", 3)
                self.assertEqual(sorted(cache, key=repr), sorted([key, (0, tile) if tile else 0], key=repr))
                for operator in operators:
                    stencil = operator.compact_finite_difference
                    self.assertIsInstance(stencil, CuPyStencilMajorFiniteDifference)
                    self.assertEqual(stencil.storage_mode, storage)
                    self.assertIsNone(operator.compact_finite_difference_reason)
                    # The former kernels, as under ``former``; the operator names the value it was built for.
                    self.assertEqual([(kernel.code, kernel.name) for kernel in (stencil.apply_kernel,
                                                                                stencil.recurrence_kernel)],
                                     [(stencil_source(tile, 3), name) for name in _KERNELS])
                    self.assertIs(stencil.recurrence_kernel, stencil.former_recurrence_kernel)
                    self.assertIsNone(stencil.fast_path_slots)
                    self.assertEqual(stencil.stencil_kernel_setting, "pipelined")
                    self.assertEqual(stencil.stencil_kernel_fallback,
                                     "the generated kernel could not be built: "
                                     "_CompileError: nvrtc: error: injected failure of the generated kernel")
                    self.assertEqual(stencil.stencil_kernel_record,
                                     "pipelined: former kernels (the generated kernel could not be built: "
                                     "_CompileError: nvrtc: error: injected failure of the generated kernel)")
                    self.assertIs(stencil.recurrence_kernel_for(6, 1), stencil.former_recurrence_kernel)
        # The former kernels hold no record of a failure, nor does a generated kernel that compiles.
        for value, fast_path_slots in (("former", None), ("pipelined", 3)):
            stencil = _host_stencil(value, 16, metadata)
            self.assertEqual((stencil.stencil_kernel_setting, stencil.fast_path_slots, stencil.stencil_kernel_fallback),
                             (value, fast_path_slots, None))
        # The source of the generated kernel failing is a failure of the generated kernel like its compilation.
        with _switch("pipelined"), _host_kernels(), \
             patch.object(stencil_pipeline, "pipelined_source", side_effect=RuntimeError("production stencil ABI")), \
             warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stencil = CuPyStencilMajorFiniteDifference(mixed_tests._HostCuPy(), metadata=metadata, implicit_tile=0)
        self.assertEqual(len(caught), 1)
        self.assertIs(stencil.recurrence_kernel, stencil.former_recurrence_kernel)
        self.assertEqual(stencil.stencil_kernel_fallback,
                         "the generated kernel could not be built: RuntimeError: production stencil ABI")
        # A filter that turns warnings into errors does not turn the former kernels into a failure.
        with _switch("pipelined"), _host_kernels(), \
             patch.object(cupy_stencil_major, "compile_cupy_raw", _compile_failing_for_generated_sources([])), \
             warnings.catch_warnings():
            warnings.simplefilter("error")
            stencil = CuPyStencilMajorFiniteDifference(mixed_tests._HostCuPy(), metadata=metadata, implicit_tile=0)
        self.assertIsNone(stencil.fast_path_slots)
        self.assertIn("_CompileError", stencil.stencil_kernel_fallback)

    def test_a_run_reports_the_recurrence_kernels_of_its_stencils(self):
        from types import SimpleNamespace

        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver

        def operator(value, tile=16, fail=False):
            with _switch(value), mixed_tests.host_cupy(), _host_kernels(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if fail:
                    failing = _compile_failing_for_generated_sources([])
                    with patch.object(cupy_stencil_major, "compile_cupy_raw", failing), \
                         patch.object(cupy_compile, "compile_cupy_raw", failing):
                        return CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                               finite_difference_metadata=self.metadata, implicit_tile=tile)
                return CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                       finite_difference_metadata=self.metadata, implicit_tile=tile)

        def sectors(*operators):
            # The sector operators of a symmetry eigensolver; None for a sector of another MPI rank.
            solver = object.__new__(CuPySymmetrySCFEigensolver)
            solver._operators = list(operators)
            return solver.stencil_kernel

        fast = "pipelined: fast path for 3 slots"
        failed = ("pipelined: former kernels (the generated kernel could not be built: "
                  "_CompileError: nvrtc: error: injected failure of the generated kernel)")
        for tile in (16, 0):
            with self.subTest(tile=tile):
                former, pipelined, unset = (operator(value, tile) for value in ("former", "pipelined", None))
                fallback = operator("pipelined", tile, fail=True)
                records = [item.compact_finite_difference.stencil_kernel_record
                           for item in (former, pipelined, unset, fallback)]
                # An unset variable is reported as the value it gives.
                self.assertEqual(records, ["former", fast, {"former": "former", "pipelined": fast}[DEFAULT], failed])
                # The sectors of a run: one record where all give the same, as every run has, else each record with
                # its sectors; the sectors of other ranks are not there.
                self.assertEqual(sectors(former, None, former), "former")
                self.assertEqual(sectors(None, pipelined, pipelined), fast)
                self.assertEqual(sectors(pipelined, fallback, None, pipelined),
                                 f"{fast} (sectors 0 3); {failed} (sectors 1)")
        # An operator without the stencil-major kernels, or none at all.
        self.assertEqual(sectors(SimpleNamespace(compact_finite_difference=None)), "none")
        self.assertEqual(sectors(None, None), "none")


class SweepInputTests(unittest.TestCase):
    """The stencils, inputs and launches of the device sweep are what it says they are."""

    def test_the_sweep_names_the_slot_counts_of_every_order_and_others_in_every_layout(self):
        self.assertEqual(SWEEP_SLOTS, (2, 7, 9, 13, 19, 25, 26, 31, 37, 40, 43, 49, 55, 61))
        self.assertEqual(SWEEP_TILES, (0,) + _TILES)

    def test_the_stencils_of_the_sweep(self):
        for slots in (2, 25, 61):
            with self.subTest(slots=slots):
                metadata = _sweep_stencil(SWEEP_ROWS, slots, 1000 + slots)
                present = metadata.neighbors >= 0
                # No present neighbour names the NaN at palette[0]; missing ones carry codes that name nothing.
                self.assertEqual(int(metadata.coefficient_palette.view(np.uint64)[0]), int(TRAP))
                self.assertTrue((metadata.coefficient_codes[present] >= 1).all())
                self.assertTrue((metadata.coefficient_codes[~present] >= metadata.coefficient_palette.size).any())
                # Empty rows, neighbours missing in every slot, and in the middle of the slots of a row.
                self.assertEqual(int(present.sum(axis=0).min()), 0)
                self.assertTrue(all((~present[slot]).any() for slot in range(slots)))
                if slots > 1:
                    self.assertTrue((~present[:-1] & present[1:]).any())
                # The tiles of every size find regular blocks, and others.
                for tile in _TILES:
                    _neighbors, _codes, statistics = implicit_stencil.pack_affine_tiles(metadata, tile)
                    self.assertGreater(statistics["regular_rows"], 0)
                    self.assertLess(statistics["regular_rows"], SWEEP_ROWS)
                positive = _positive_palette(metadata).coefficient_palette
                self.assertEqual(int(positive.view(np.uint64)[0]), int(TRAP))
                self.assertTrue((positive[1:] > 0).all())
                words = _sweep_nan_palette(metadata).coefficient_palette.view(np.uint64)
                self.assertTrue(np.isnan(words.view(np.float64)).all())
                self.assertEqual(len(set(words.tolist())), words.size)
                quiet = (words & np.uint64(0x0008000000000000)) != 0
                self.assertTrue(quiet.any() and not quiet.all())
                self.assertTrue(((words >> np.uint64(63)) == 1).any())

    def test_every_value_minus_zero_tells_a_sum_started_from_minus_zero(self):
        operands = Operands(np, 1031, SIGNED_ZEROS, 5)
        for array in (operands.current, operands.previous, operands.wide, operands.offset, operands.vector,
                      operands.coefficients):
            self.assertTrue((array.view(np.uint64) == np.uint64(0x8000000000000000)).all())
        self.assertTrue((operands.potential > 0).all())
        center, scale, sigma, sigma_next = operands.scalars
        self.assertTrue(center < 0 and scale > 0 and sigma < 0 and sigma_next > 0)
        self.assertEqual(operands.table.tolist(), [list(operands.scalars)] * 4)
        # The slot sum of positive coefficients and -0.0 values: +0.0 from +0.0, -0.0 from -0.0, and the epilogue
        # keeps the sign of either.
        products = np.array([0.7, 1.9, 0.2]) * operands.current[0, 0]
        for start, sign in ((0.0, 0), (-0.0, 1)):
            total = start
            for product in products:
                total += product
            value = (((total + operands.potential[0] * operands.current[0, 0]) - center * operands.current[0, 0])
                     * scale - sigma * operands.previous[0, 0]) * sigma_next
            self.assertEqual((int(np.float64(value).view(np.uint64)) >> 63, value), (sign, 0.0))

    def test_the_launches_of_the_sweep(self):
        for slots in (1, 2, 25, 61):
            with self.subTest(slots=slots):
                paths = sweep_paths(slots)
                fast = [path for path in paths if fast_path_taken(path, slots)]
                # Six columns of unit row stride in five layouts with all eight flags, and nine launches that the
                # fast path does not take (C order, reversed rows, one slot fewer), or six for one slot.
                self.assertEqual(len(fast), 40)
                self.assertEqual(len(paths) - len(fast), 9 if slots > 1 else 6)
                self.assertTrue(all(path["width"] == 6 for path in paths))


# ----------------------------------------------------------------------------------------------------------------
# Device tests
# ----------------------------------------------------------------------------------------------------------------

def build_pair(cp, metadata, tile):
    """The operators of ``metadata`` in the layout ``tile`` with the switch at ``former`` and ``pipelined``."""
    stencils = {}
    for value in ("former", "pipelined"):
        with _switch(value):
            stencils[value] = CuPyStencilMajorFiniteDifference(cp, metadata=metadata, implicit_tile=tile)
    former, pipelined = stencils["former"], stencils["pipelined"]
    if former.fast_path_slots is not None or pipelined.fast_path_slots != int(former.slot_count):
        raise RuntimeError("the pair does not hold the former kernels and the generated one")
    return former, pipelined


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class PipelinedKernelDeviceTests(unittest.TestCase):
    def setUp(self):
        cp, _ = require_cupy()
        self.verified = cp.cuda.Device().compute_capability == VERIFIED_ARCHITECTURE

    def tearDown(self):
        gc.collect()

    def check(self, found, label):
        """Every word the former one; off the verified architecture, every word but the payload of a NaN that both
        kernels give (the module docstring)."""
        if self.verified:
            self.assertEqual(found["differing_words"], 0, f"{label}: {found['differing']}")
        else:
            self.assertEqual(found["differing_not_as_two_nans"], 0,
                             f"{label}: {found['differing_not_as_two_nans_paths']}")

    def check_registers(self, attributes):
        """At most 64 registers (the launch bounds); no local memory on the verified architecture."""
        self.assertLessEqual(attributes["num_regs"], 64)
        if self.verified:
            self.assertEqual(attributes["local_size_bytes"], 0)

    def test_every_launch_of_the_recurrence_kernel_has_the_bits_of_the_former_kernel(self):
        cp, _ = require_cupy()
        for name, build in CASES.items():
            metadata = build()
            slots, rows = metadata.neighbors.shape
            for tile in (0, 16, 32):
                former, pipelined = build_pair(cp, metadata, tile)
                # The apply kernel and the former recurrence kernel are the former ones, compiled once.
                self.assertIs(pipelined.apply_kernel, former.apply_kernel)
                self.assertIs(pipelined.former_recurrence_kernel, former.recurrence_kernel)
                self.assertIn(f"if (slot_count == {slots} && width == 6", pipelined.recurrence_kernel.code)
                self.assertNotIn("if (slot_count ==", former.recurrence_kernel.code)
                regular = former.implicit_statistics["regular_rows"] if tile else None
                totals = dict(launches=0, fast=0, nan_words=0, differing=0)
                for index, kind in enumerate(KINDS):
                    operands = Operands(cp, int(rows), kind, 7919 * index + 31 * tile + slots)
                    found = compare_recurrence_kernels(cp, former, pipelined, operands)
                    with self.subTest(case=name, tile=tile, kind=kind):
                        self.check(found, f"{name} tile {tile} {kind}")
                    totals["launches"] += found["launches"]
                    totals["fast"] += found["fast_launches"]
                    totals["nan_words"] += found["nan_words"]
                    totals["differing"] += found["differing_words"]
                    del operands
                _report(f"recurrence kernel case={name} slots={slots} rows={rows} tile={tile} regular_rows={regular} "
                        f"launch_pairs={totals['launches']} fast_path={totals['fast']} "
                        f"nan_words_compared={totals['nan_words']} differing={totals['differing']}")
                del former, pipelined

    def test_every_slot_count_and_tile_size_keeps_the_bits_of_the_former_kernel(self):
        # The generated source differs with the slot count (straight-line code, the slot of the hints) and the tile
        # size (the descriptor): every count of SWEEP_SLOTS in every layout, on the stencil of _sweep_stencil, with
        # every kind of input, two NaNs and all NaNs also on a stencil of NaN coefficients, and every value -0.0 on
        # one of positive coefficients; then the operators' methods and which kernel took which launch.
        cp, _ = require_cupy()
        for slots in SWEEP_SLOTS:
            metadata = _sweep_stencil(SWEEP_ROWS, slots, 1000 + slots)
            stencils = {"": metadata, "nan palette": _sweep_nan_palette(metadata),
                        "positive palette": _positive_palette(metadata)}
            paths = sweep_paths(slots)
            for tile in SWEEP_TILES:
                started = time.perf_counter()
                pairs = {name: build_pair(cp, item, tile) for name, item in stencils.items()}
                former, pipelined = pairs[""]
                registers = pipelined.stencil_kernel_attributes["recurrence_kernel"]
                with self.subTest(slots=slots, tile=tile, check="kernels"):
                    self.assertIs(pipelined.apply_kernel, former.apply_kernel)
                    self.assertIs(pipelined.former_recurrence_kernel, former.recurrence_kernel)
                    self.assertIn(f"if (slot_count == {slots} && width == 6 && current_row_stride == 1) {{",
                                  pipelined.recurrence_kernel.code)
                    self.assertEqual(pipelined.stencil_kernel_record, f"pipelined: fast path for {slots} slots")
                    self.check_registers(registers)
                found_in_case = dict(pairs=0, differing=0, nan_words=0)

                def count(found, label):
                    found_in_case["pairs"] += found.get("launches", found.get("calls"))
                    found_in_case["differing"] += found["differing_words"]
                    found_in_case["nan_words"] += found["nan_words"]
                    self.check(found, label)

                for index, kind in enumerate(KINDS + (SIGNED_ZEROS,)):
                    operands = Operands(cp, SWEEP_ROWS, kind, 7 * slots + 131 * index + tile)
                    palettes = (("positive palette",) if kind == SIGNED_ZEROS
                                else ("", "nan palette") if kind in ("twonan", "allnan") else ("",))
                    for palette in palettes:
                        with self.subTest(slots=slots, tile=tile, kind=kind, palette=palette):
                            count(compare_recurrence_kernels(cp, *pairs[palette], operands, paths),
                                  f"{slots} slots, tile {tile}, {kind} {palette}")
                    del operands
                former_fast, former_rest = count_launches(former)
                fast, rest = count_launches(pipelined)
                for index, kind in enumerate(("finite", "allnan")):
                    operands = Operands(cp, SWEEP_ROWS, kind, 104729 * index + tile + slots)
                    with self.subTest(slots=slots, tile=tile, kind=kind, check="methods"):
                        count(compare_operator_methods(cp, former, pipelined, operands, layouts=("F", "C"),
                                                       widths=(5, 6, 13), apply=False),
                              f"{slots} slots, tile {tile}, methods {kind}")
                    del operands
                with self.subTest(slots=slots, tile=tile, check="dispatch"):
                    self.assertIs(former_fast, former_rest)
                    self.assertTrue(fast.launches and rest.launches)
                    self.assertTrue(all(launch == (6, 1, slots) for launch in fast.launches))
                    self.assertFalse(any(width == 6 and stride == 1 for width, stride, _slots in rest.launches))
                    self.assertEqual(sorted(fast.launches + rest.launches), sorted(former_fast.launches))
                regular = former.implicit_statistics["regular_rows"] if tile else None
                _report(f"sweep slots={slots} tile={tile} regular_rows={regular} pairs={found_in_case['pairs']} "
                        f"nan_words_compared={found_in_case['nan_words']} differing={found_in_case['differing']} "
                        f"registers={registers.get('num_regs')} local={registers.get('local_size_bytes')} "
                        f"seconds={time.perf_counter() - started:.2f}")
                del pairs, former, pipelined
                cp.get_default_memory_pool().free_all_blocks()

    def test_the_operators_methods_have_the_bits_of_the_former_ones_and_send_narrow_launches_to_the_former_kernel(
            self):
        cp, _ = require_cupy()
        for name in ("irregular", "box8", "box12", "holes", "trap25", "one_row"):
            metadata = CASES[name]()
            slots, rows = metadata.neighbors.shape
            for tile in (0, 16, 32):
                former, pipelined = build_pair(cp, metadata, tile)
                former_fast, former_rest = count_launches(former)
                fast, rest = count_launches(pipelined)
                calls = differing = 0
                for index, kind in enumerate(("finite", "specials", "twonan", "allnan")):
                    operands = Operands(cp, int(rows), kind, 104729 * index + tile + slots)
                    found = compare_operator_methods(cp, former, pipelined, operands)
                    with self.subTest(case=name, tile=tile, kind=kind):
                        self.check(found, f"{name} tile {tile} {kind}")
                    calls += found["calls"]
                    differing += found["differing_words"]
                    del operands
                # The former operator launches its one kernel; the pipelined one its generated kernel for the
                # launches of six columns with unit row stride and the former kernel for all others.
                self.assertIs(former_fast, former_rest)
                self.assertTrue(fast.launches and rest.launches)
                self.assertTrue(all(launch == (6, 1, slots) for launch in fast.launches))
                self.assertFalse(any(takes_fast_path(slot_count, width, stride, slots)
                                     for width, stride, slot_count in rest.launches))
                self.assertEqual(sorted(fast.launches + rest.launches), sorted(former_fast.launches))
                _report(f"operator methods case={name} slots={slots} rows={rows} tile={tile} calls={calls} "
                        f"fast_path_launches={len(fast.launches)} former_kernel_launches={len(rest.launches)} "
                        f"differing={differing}")
                del former, pipelined

    def test_the_generated_kernel_holds_64_registers_and_no_local_memory_on_the_a100(self):
        cp, _ = require_cupy()
        for name in ("irregular", "box4", "box8", "box12"):
            metadata = CASES[name]()
            slots = metadata.neighbors.shape[0]
            for tile in (0, 16, 32):
                former, pipelined = build_pair(cp, metadata, tile)
                attributes = pipelined.stencil_kernel_attributes
                _report(f"registers slots={slots} tile={tile} pipelined={attributes} "
                        f"former={former.stencil_kernel_attributes}")
                with self.subTest(case=name, tile=tile):
                    self.check_registers(attributes["recurrence_kernel"])
                    # The apply kernel and the former recurrence kernel are the former ones.
                    self.assertEqual(attributes["apply_kernel"], former.stencil_kernel_attributes["apply_kernel"])
                    self.assertEqual(attributes["former_recurrence_kernel"],
                                     former.stencil_kernel_attributes["recurrence_kernel"])

    def operators(self, tile, projectors=True):
        """H with three projectors on the box of the order-8 star, built with the switch at either value."""
        metadata = CASES["box8"]()
        rows = metadata.shape[0]
        generator = np.random.default_rng(47)
        dense = np.zeros((rows, 3))
        for column, start in enumerate((300, 2900, 6100)):
            dense[start:start + 400, column] = 0.05 * generator.standard_normal(400)
        potential = 0.1 * generator.standard_normal(rows)
        built = {}
        for value in ("former", "pipelined"):
            with _switch(value):
                built[value] = CuPyHamiltonian(None, potential, (sp.csr_matrix(dense), np.array([1.0, -1.0, 1.0])),
                                               retain_generic_laplacian=False, finite_difference_metadata=metadata,
                                               implicit_tile=tile)
        bound = 1.01 * (25 * float(np.abs(metadata.coefficient_palette).max()) + 0.6 + float((dense * dense).sum()))
        return built, rows, bound

    @staticmethod
    def vectors(rows, width, kind):
        """Columns of one kind: finite, one NaN, two NaNs of different payloads in neighbouring rows of the sixth
        column (where the scatter is added last), or every value a NaN of its own payload."""
        host = np.random.default_rng(5).uniform(-1.0, 1.0, (rows, width))
        words = host.view(np.uint64)
        if kind in ("one_nan", "two_nans"):
            words[3000, 5] = np.uint64(0x7FF800000000BEEF)
        if kind == "two_nans":
            words[3001, 5] = np.uint64(0xFFF800000000F00D)
            words[310, 2] = np.uint64(0x7FF4000000000AB0)
            words[311, 2] = np.uint64(0x7FF8000000000AB1)
        if kind == "all_nan":
            words[...] = nan_pattern(host.shape, 3)
        return host

    def test_filter_graphs_record_the_kernel_of_each_block_and_keep_the_bits(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import FilterBlock, chebff_filter
        from parsec_python.acceleration.Eigensolvers.filter_graph import BlockFilterGraphs

        cp, _ = require_cupy()
        blocks = (FilterBlock(0, 6, 7), FilterBlock(6, 12, 7), FilterBlock(12, 14, 5))
        for tile in (16, 0):
            built, rows, bound = self.operators(tile)
            counted = {value: count_launches(operator.compact_finite_difference) for value, operator in built.items()}
            results = {}
            for kind in ("finite", "one_nan", "two_nans", "all_nan"):
                host = self.vectors(rows, 14, kind)
                for value, operator in built.items():
                    vectors = cp.asarray(host, order="F")
                    applied = operator.apply_into(vectors, cp.full_like(vectors, cp.nan))
                    graphs = BlockFilterGraphs(operator).apply(cp.asarray(host, order="F"), blocks, -bound, bound,
                                                               -1.5 * bound, False)
                    with _switch(value, PARSEC_CUPY_FILTER_GRAPHS="0"):
                        plain = chebff_filter(operator, cp.asarray(host, order="F"), 7, -bound, bound, -1.5 * bound,
                                              block_size=6, batch_blocks=False)
                    results[value, kind] = tuple(cp.asnumpy(array) for array in (applied, graphs, plain))
                for index, route in enumerate(("H X", "filter graphs", "filter without graphs")):
                    with self.subTest(tile=tile, kind=kind, route=route):
                        reference, found = results["former", kind][index], results["pipelined", kind][index]
                        if kind == "finite":
                            self.assertTrue(np.isfinite(reference).all() and reference.any())
                        words, former_words = found.view(np.uint64), reference.view(np.uint64)
                        if index and not self.verified:
                            # As check(): off the verified architecture the payload of a NaN that both kernels
                            # give is open.  H X is the former kernel under both values, so it keeps every word.
                            kept = ~(np.isnan(found) & np.isnan(reference))
                            words, former_words = words[kept], former_words[kept]
                        np.testing.assert_array_equal(words, former_words)
            # A graph records the kernel its launches take when it is recorded: the generated one for the blocks of
            # six columns, the former one for the block of two.
            fast, rest = counted["pipelined"]
            self.assertIn((6, 1, 25), fast.launches)
            self.assertTrue(all(launch == (6, 1, 25) for launch in fast.launches))
            self.assertIn((2, 1, 25), rest.launches)
            self.assertFalse(any(width == 6 and stride == 1 for width, stride, _slots in rest.launches))
            _report(f"routes tile={tile} fast_path_launches={len(fast.launches)} "
                    f"former_kernel_launches={len(rest.launches)} "
                    f"compared={'every word' if self.verified else 'every word but NaN payloads of the filter routes'}")
            del built

    def test_the_backend_details_name_the_recurrence_kernels_of_a_full_grid_operator(self):
        from parsec_python.acceleration.backends.cupy_runtime import CuPyHamiltonianBackend
        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
        from parsec_python.V_ion import NonlocalProjectorOperator

        matrix = _star((9, 10, 25), 8)
        dense = np.zeros((matrix.shape[0], 2))
        dense[100:400, 0], dense[1000:1300, 1] = 0.05, -0.03
        nonlocal_operator = NonlocalProjectorOperator(projectors=sp.csc_matrix(dense), signs=np.array([1.0, -1.0]),
                                                      labels=((0, 0, 0), (1, 0, 0)))
        fast = "pipelined: fast path for 25 slots"
        for value, expected in ((None, {"former": "former", "pipelined": fast}[DEFAULT]), ("former", "former"),
                                ("pipelined", fast)):
            with self.subTest(value=value), _switch(value):
                backend = CuPyHamiltonianBackend(matrix, nonlocal_operator)
                details = dict(backend.info.details)
                self.assertEqual(details["gpu_stencil_kernel"], expected)
                self.assertEqual(backend.device_operator.compact_finite_difference.stencil_kernel_record, expected)
                # The same operator as the sector of a symmetry eigensolver.
                solver = object.__new__(CuPySymmetrySCFEigensolver)
                solver._operators = [backend.device_operator, None]
                self.assertEqual(solver.stencil_kernel, expected)
                del backend, solver

    def test_a_calculation_reports_its_recurrence_kernels_and_has_the_results_of_the_former_ones(self):
        from dataclasses import replace
        from pathlib import Path

        from parsec_python.acceleration.driver import prepare_single_point, run_scf
        from parsec_python.Input import parse_parsec_input

        # The smoke input: 13 slots, three states and a buffer of six.  Without symmetry the filter cuts its nine
        # columns into launches of six and three; with symmetry the sectors have fewer columns than six.
        problem = parse_parsec_input(Path(__file__).resolve().parents[2] / "tests" / "data" / "H_cli_smoke.in").problem
        problem = replace(problem, scf=replace(problem.scf, max_iterations=3))
        for symmetry in ("off", "auto"):
            results = {}
            for value in ("former", "pipelined"):
                with _switch(value, PARSEC_ACCELERATED_RESIDENT="0"):
                    system = prepare_single_point(problem, backend="auto", symmetry=symmetry)
                    prepared = dict(system.backend_info.details)
                    result = run_scf(system)
                details = dict(result.backend.details)
                key = ("orbital_sector_stencil_kernel" if "orbital_sector_stencil_kernel" in details
                       else "gpu_stencil_kernel")
                results[value] = result, details[key], prepared[key]
                del system
            with self.subTest(symmetry=symmetry):
                former, pipelined = results["former"], results["pipelined"]
                self.assertEqual((former[1], former[2]), ("former", "former"))
                self.assertEqual(pipelined[1], pipelined[2])
                self.assertTrue(pipelined[1].startswith("pipelined: fast path for "), pipelined[1])
                self.assertEqual(pipelined[0].iterations, former[0].iterations)
                for name in ("eigenvalues", "density", "hartree_potential"):
                    np.testing.assert_array_equal(np.asarray(getattr(pipelined[0], name)).view(np.uint64),
                                                  np.asarray(getattr(former[0], name)).view(np.uint64))
                self.assertEqual(pipelined[0].energies, former[0].energies)
                _report(f"calculation symmetry={symmetry} former={former[1]!r} pipelined={pipelined[1]!r} "
                        f"iterations={former[0].iterations} identical=True")

    @unittest.skipUnless(cupy_available() and require_cupy()[0].cuda.runtime.getDeviceCount() >= 2,
                         "a shared sector needs two CUDA devices")
    def test_the_devices_of_a_shared_sector_filter_with_the_kernels_of_their_owner(self):
        from parsec_python.acceleration.Eigensolvers.chebyshev import FilterBlock

        cp, _ = require_cupy()
        blocks = (FilterBlock(0, 6, 7), FilterBlock(6, 12, 7), FilterBlock(12, 18, 7), FilterBlock(18, 20, 5))
        for tile in (16, 0):
            built, rows, bound = self.operators(tile)
            results = {}
            for value, operator in built.items():
                with _switch(value):
                    worker = DistributedFilter(operator, (0, 1))
                replica = worker.replicas[1].compact_finite_difference
                held = operator.compact_finite_difference
                self.assertEqual(int(replica.neighbors.device.id), 1)
                self.assertEqual(replica.fast_path_slots, held.fast_path_slots)
                for name in ("apply_kernel", "recurrence_kernel", "former_recurrence_kernel"):
                    self.assertEqual(getattr(replica, name).code, getattr(held, name).code)
                if value == "pipelined":
                    self.assertEqual(replica.fast_path_slots, 25)
                    self.assertIn((1, tile, "pipelined", 25) if tile else (1, "pipelined", 25),
                                  implicit_stencil._CACHE if tile else cupy_stencil_major._KERNEL_CACHE)
                host = self.vectors(rows, 20, "finite")
                results[value] = cp.asnumpy(worker.apply(cp.asarray(host, order="F"), blocks, -bound, bound, False))
                del worker
            with self.subTest(tile=tile):
                self.assertTrue(np.isfinite(results["former"]).all())
                np.testing.assert_array_equal(results["pipelined"].view(np.uint64), results["former"].view(np.uint64))
            del built


if __name__ == "__main__":
    unittest.main()
