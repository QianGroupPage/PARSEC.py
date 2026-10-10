"""Sector subspace shared by several CUDA devices of one node.

The owner-centred multi-device route keeps the whole ``N x m`` Ritz basis of
a symmetry sector on one device and only lends columns to the others for
filtering.  Here no device ever holds the whole basis.  The data flow is the
orbital-parallel scheme of GPU SPARC (Sharma et al., J. Chem. Phys. 158,
204117 (2023), Sec. II), with direct device-to-device copies instead of host
staging:

* each device filters, and applies ``H`` to, its own block of columns - no
  communication;
* the filtered columns and ``H`` times them are exchanged so that each device
  holds a block of rows of every column, forms its part of ``X.T X`` and
  ``X.T H X``, and later rotates its rows;
* the small generalized eigenproblem and all of its stability audits run on
  the owner (:func:`rayleigh_ritz.solve_whitened_ritz_on_device`), or with
  ``PARSEC_CUPY_RITZ_DENSE_BACKEND=host`` on the host exactly as in the
  single-device route (:func:`rayleigh_ritz.solve_whitened_ritz`);
* the rotated rows return to column blocks for the next filter.

Applies to operators that were given a group of at least two devices.
``PARSEC_CUPY_DISTRIBUTED_STATE=auto``, the default, shares a basis whose
blocks fit the devices (see :func:`shared_basis_devices`); ``1`` shares every
one and ``0`` none.

A shared basis is filtered in FP64 and solved in its filtered form in every
pass, with QR on the owner only as the fallback after a failed stability
audit.  That holds for every cycle of the CHEBFF first solve too, where one
device orthonormalizes the basis unless
``PARSEC_CUPY_CHEBFF_GENERALIZED_RITZ`` is set.  ``auto`` leaves a basis on
its owner where ``PARSEC_CUPY_GENERALIZED_RITZ`` and its work threshold
select orthonormalization for later passes; ``1`` shares it regardless.
Results differ from a single-device route that also filters in FP64 and
takes the generalized solve in every pass only by the summation order of
the Gram matrices.

A device holds one tall block in the Ritz step: the columns are re-laid as
rows inside their own block a slab at a time, ``H`` times each slab follows
it to the row layout and is projected there, and the rows are rotated in
place.  ``PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=2`` selects the former step,
which exchanges the filtered columns as a whole into a row block of their
own, and ``3`` the one before it, which also keeps ``H`` times the columns
as a tall block (see :func:`tall_blocks_per_device`).

The block of a device holds two ranges of columns side by side: its share
of the lower and of the upper half of the columns, which later passes filter
at different degrees.  ``PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT=contiguous``
selects the former layout, one range of neighbouring columns per device (see
:func:`interleaved_layout_requested`, :class:`ColumnLayout`).  The row blocks
hold the columns in their own order either way.
"""

from __future__ import annotations

from concurrent.futures import wait
from dataclasses import dataclass
from math import gcd
import os
from time import perf_counter
from typing import Any
from weakref import ref

import numpy as np

from parsec_python.Eigensolvers.chebff import ChebFFCycle, ChebFFSettings
from parsec_python.Eigensolvers.subspace import SubspaceSettings

from ..backends.cupy import device_stage, require_cupy
from ..backends.cupy_stencil_major import CuPyStencilMajorFiniteDifference
from .chebff import (
    DeviceChebFFResult,
    DeviceChebFFState,
    _initial_filter_lower_bound,
    _updated_filter_bounds,
)
from .chebyshev import FilterBlock, subspace_filter_blocks, uniform_filter_blocks
from .distributed_filter import DistributedFilter, _pool, block_partitions, starting_sigmas
from .lapack_random import LapackRandom
from .orthogonalize import orthonormalize_complete_subspace
from .rayleigh_ritz import (
    DeviceRayleighRitzResult,
    GeneralizedRitzStabilityError,
    _condition_policy,
    _device_overlap_condition,
    _lower_triangle_product,
    _mirrored_lower,
    _rotate_in_place,
    _streaming_bytes,
    _symmetric_overlap,
    _whole_multiple,
    dense_solve_on_device,
    generalized_ritz_requested,
    gram_multiple,
    rayleigh_ritz,
    solve_whitened_ritz,
    solve_whitened_ritz_on_device,
)
from .spectral_bounds import LanczosBoundResult, lanczos_upper_bound
from .subspace import (
    DeviceSubspaceResult,
    DeviceSubspaceState,
    _next_filter_lower_bound,
    _validate_state,
    filter_degree,
)


_ALGORITHM = "distributed_generalized_cholesky_rayleigh_ritz"
# The most bytes of a slab of ``H X`` where the blocks of a shared basis fit
# a device with it (see :func:`wider_slabs_fit`); 4 GiB otherwise.
_WIDER_SLAB_BYTES = 6 << 30
# The bytes, in all rows, of the columns that a round of the Ritz step with
# one tall block brings together on four or more devices where no budget is
# set: a slab is an even share of them for every device (see
# :func:`slab_columns`).
_ROUND_BYTES = 8 << 30
# The most bytes of a slab of that step on two devices where no budget is
# set and :func:`pair_slab_bytes` names no other limit: a round of 4 GiB.
_PAIR_SLAB_BYTES = 2 << 30
# The three ways to split four devices into two pairs: the partner of every
# device in each turn of an exchange that takes its copies pair by pair (see
# :func:`paired_exchange_requested`).
_PAIR_TURNS = ((1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0))


def distributed_state_policy() -> str:
    """Return ``off``, ``on`` or ``auto`` from ``PARSEC_CUPY_DISTRIBUTED_STATE``.

    ``auto`` is the default.
    """

    value = os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE", "auto").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false", "auto"}:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE must be on, off, or auto")
    if value == "auto":
        return "auto"
    return "on" if value in {"1", "on", "true"} else "off"


def keeps_column_ranges() -> bool:
    """Whether later passes filter the columns each device already holds.

    The default.  ``PARSEC_CUPY_DISTRIBUTED_STATE_RANGES=balanced`` instead
    moves columns between devices whenever the filter degrees shift the work
    split.  The high-degree half of the usual later pass costs 1.5 times the
    low-degree half, so balancing shortens the filter, but every block then
    needs 25% headroom and a moved block exists twice for as long as the
    caller holds the previous basis: about half as much memory again.
    :func:`interleaved_layout_requested` balances fixed ranges instead.
    """

    value = os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE_RANGES", "fixed").strip().lower()
    if value not in {"fixed", "balanced"}:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE_RANGES must be fixed or balanced")
    return value == "fixed"


def interleaved_layout_requested() -> bool:
    """Whether every device holds columns of both halves of the basis.

    The default.  ``PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT=contiguous`` selects
    the former layout, one range of neighbouring columns per device.  A
    later pass filters the lower half of its blocks at ``degree - delta``
    and the upper half at ``degree + delta``
    (:func:`chebyshev.subspace_filter_blocks`).  With contiguous ranges half
    of the devices then filter at the higher degree only and the others wait
    for them.  Interleaved, a device holds its share of each half
    (:func:`interleaved_ranges`): all devices do the same work in every pass,
    no column moves and no block needs headroom.  The later passes of 19,392
    electrons on 16 GPUs then filtered a sector in 16.4 s instead of 19.5 and
    those of 23,768 electrons in 25.6 instead of 30.5, which took 3.0 and
    4.2 s off runs of 89.3 and 140.0 s at the same peak memory.  The layout
    is that of the trial basis and stays.

    Balanced ranges (:func:`keeps_column_ranges`) move columns between the
    devices instead, and theirs are contiguous: left unset, the layout
    follows the ranges, and ``interleaved`` named together with balanced
    ranges is an error.

    The filtered columns and the row blocks are the same in both layouts.
    What differs is the order in which a device adds up its columns of the
    density and, with one or two tall blocks, how the projection is cut into
    slabs, so the results differ by round-off.
    """

    value = os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT")
    if value is None:
        return keeps_column_ranges()
    value = value.strip().lower()
    if value not in {"contiguous", "interleaved"}:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT must be contiguous or interleaved")
    if value == "interleaved" and not keeps_column_ranges():
        raise ValueError(
            "PARSEC_CUPY_DISTRIBUTED_STATE_LAYOUT=interleaved needs the fixed ranges of "
            "PARSEC_CUPY_DISTRIBUTED_STATE_RANGES"
        )
    return value == "interleaved"


def interleaved_ranges(blocks, devices: int):
    """Two column ranges per device: its shares of the two halves of ``blocks``.

    The lower half is what a later pass filters at its lower degree: half of
    the blocks of full width, rounded up
    (:func:`chebyshev.subspace_filter_blocks`).  Each half is cut into
    ``devices`` parts of whole blocks, as equal in number as they can be,
    and device ``k`` holds part ``k`` of both.  The top block is then the
    last one of the last device, and the leading columns of the basis are
    the leading columns of every device (see :class:`DistributedBasis`).
    Returns ``None`` where a half has fewer blocks than there are devices.
    """

    blocks = tuple(blocks)
    devices = int(devices)
    width = blocks[0].stop - blocks[0].start
    low = (sum(block.stop - block.start == width for block in blocks) + 1) // 2
    high = len(blocks) - low
    if min(low, high) < devices:
        return None
    ranges = []
    for index in range(devices):
        lower = low * index // devices, low * (index + 1) // devices
        # Rounded the other way, so that the larger parts of one half meet
        # the smaller parts of the other.
        upper = tuple(low + -(-high * part // devices) for part in (index, index + 1))
        ranges.append(tuple((blocks[first].start, blocks[last - 1].stop) for first, last in (lower, upper)))
    return tuple(ranges)


@dataclass(frozen=True)
class ColumnLayout:
    """The columns of a basis that each device of a group holds.

    ``parts[i]`` lists the column ranges of device ``i`` in ascending order,
    and its block stores them side by side in that order.  All ranges
    together cover every column once; an empty range holds nothing.  One
    range per device is the contiguous layout, and
    :func:`interleaved_ranges` gives two.

    A device may be given as its ranges or, where it has one, as that
    ``(start, stop)`` pair alone.
    """

    parts: tuple[tuple[tuple[int, int], ...], ...]

    def __post_init__(self) -> None:
        parts = []
        for part in self.parts:
            part = tuple(part)
            if len(part) == 2 and not isinstance(part[0], (tuple, list)):
                part = (part,)
            part = tuple((int(start), int(stop)) for start, stop in part)
            if any(start < 0 or stop < start for start, stop in part):
                raise ValueError("a column range must run forward from a column of the basis")
            if any(later[0] < earlier[1] for earlier, later in zip(part[:-1], part[1:])):
                raise ValueError("the column ranges of a device must ascend")
            parts.append(part)
        end = 0
        for start, stop in sorted(piece for part in parts for piece in part if piece[1] > piece[0]):
            if start != end:
                raise ValueError("the column ranges must cover every column once")
            end = stop
        object.__setattr__(self, "parts", tuple(parts))

    @property
    def widths(self) -> tuple[int, ...]:
        """Columns in the block of each device."""

        return tuple(sum(stop - start for start, stop in part) for part in self.parts)

    @property
    def columns(self) -> int:
        return sum(self.widths)

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        """The one column range of each device; an error where a device holds several."""

        if any(len(part) != 1 for part in self.parts):
            raise ValueError("a device of this layout does not hold one column range")
        return tuple(part[0] for part in self.parts)

    def spans(self, index: int, first: int = 0, last: int | None = None) -> tuple[tuple[int, int, int], ...]:
        """Where the columns ``first:last`` of the block of device ``index`` lie in the basis.

        One ``(column, offset, count)`` for each range of the device that
        they reach: ``count`` columns of the block from ``offset`` on are
        the columns of the basis from ``column`` on.  All of the block by
        default.
        """

        found, offset = [], 0
        for start, stop in self.parts[index]:
            low = max(int(first), offset)
            high = offset + stop - start
            if last is not None:
                high = min(int(last), high)
            if high > low:
                found.append((start + low - offset, low, high - low))
            offset += stop - start
        return tuple(found)

    def offset(self, index: int, column: int) -> int:
        """Where column ``column`` of the basis lies in the block of device ``index``."""

        offset = 0
        for start, stop in self.parts[index]:
            if start <= column < stop:
                return offset + column - start
            offset += stop - start
        raise ValueError("the device does not hold this column")

    def leading(self, count: int) -> "ColumnLayout":
        """The layout of the leading ``count`` columns.

        The ranges of a device ascend, so what it keeps is the leading part
        of its block.
        """

        count = int(count)
        return ColumnLayout(
            tuple(tuple((min(start, count), min(stop, count)) for start, stop in part) for part in self.parts)
        )


def _as_layout(layout) -> ColumnLayout:
    """``layout`` itself, or the layout of one column range or several per device."""

    return layout if isinstance(layout, ColumnLayout) else ColumnLayout(tuple(layout))


def column_copies_requested() -> bool:
    """Whether the copies of an exchange inside one device walk down columns.

    Packing and unpacking copy between a row range of a column block and a
    contiguous piece.  CuPy's elementwise kernel runs its fast index over the
    last axis, which in these column-major arrays goes across the columns,
    so neighbouring threads touch addresses a whole column apart.  By
    default the assignment is therefore made through the transposed views of
    both arrays, whose last axis runs down a column.
    ``PARSEC_CUPY_EXCHANGE_COLUMN_COPIES=0`` assigns the views as they are;
    the values copied are the same.
    """

    value = os.environ.get("PARSEC_CUPY_EXCHANGE_COLUMN_COPIES", "1").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_EXCHANGE_COLUMN_COPIES must be on or off")
    return value in {"1", "on", "true"}


def exchange_chunk_bytes() -> int:
    """Bytes of the one buffer through which a device exchanges its columns.

    ``PARSEC_CUPY_EXCHANGE_CHUNK_BYTES`` sets it; the default is 1 GiB.  The
    part of a column block that lies in the rows of the other devices is
    packed and shipped a chunk of columns at a time, so a device needs room
    for one chunk next to its tall blocks instead of for all of that part.
    A chunk is never narrower than one column.
    """

    raw = os.environ.get("PARSEC_CUPY_EXCHANGE_CHUNK_BYTES", str(1 << 30)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("PARSEC_CUPY_EXCHANGE_CHUNK_BYTES must be an integer") from error
    if value < 1:
        raise ValueError("PARSEC_CUPY_EXCHANGE_CHUNK_BYTES must be positive")
    return value


def exchange_chunks(width: int, away: int, limit: int) -> tuple[tuple[int, int], ...]:
    """Column ranges in which a device exchanges a block of ``width`` columns.

    ``away`` rows of every column belong to other devices.  A chunk holds at
    most ``limit`` elements of them, but always at least one column.
    """

    width, away = int(width), int(away)
    if not width or not away:
        return ()
    step = max(1, int(limit) // away)
    return tuple((first, min(width, first + step)) for first in range(0, width, step))


def concurrent_exchange_requested() -> bool:
    """Whether a device receives from all of its peers at the same time.

    By default every source has a stream of its own on the receiving device
    and the copies of a chunk are issued together, so that the links between
    pairs of devices work side by side.  With
    ``PARSEC_CUPY_EXCHANGE_CONCURRENT=0`` a device queues the copies that
    arrive from the other devices on its one stream, so they run one after
    another.  The data moved is the same.  Four devices on the way to their
    rows receive a chunk that all of them send from one peer at a time
    instead, see :func:`paired_exchange_requested`.
    """

    value = os.environ.get("PARSEC_CUPY_EXCHANGE_CONCURRENT", "1").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_EXCHANGE_CONCURRENT must be on or off")
    return value in {"1", "on", "true"}


def paired_exchange_requested() -> bool:
    """Whether four devices take the copies of the way to their rows pair by pair.

    On that way all devices of a sector start to receive a chunk at the same
    moment, each from every other one (:meth:`SectorDeviceGroup._to_rows`):
    with four devices twelve copies at once.  Pieces of 0.49 GiB issued so
    on the four A100 of a node arrived at 59 GiB/s per device, and at 84
    in three turns of two pairs, both directions of a pair at once.  By
    default the copies of a chunk are therefore issued in the three turns
    of :data:`_PAIR_TURNS`.  In a turn every device receives from one
    partner, which receives from it, on the stream it computes on, and all
    devices have finished a turn before the next one begins: two more
    waits for all devices per chunk (:meth:`SectorDeviceGroup.map`).
    ``PARSEC_CUPY_EXCHANGE_PAIRS=0`` issues the twelve together as before.
    The data moved is the same.  No other piece size has been timed, and
    the turns are taken whatever the size.  For the bases of 3,480 to 7,120
    electrons on 16 GPUs, pieces of 0.05 to 0.17 GiB, the stage seconds
    leave open whether the two more waits cost more than the turns save.

    Only the way to the rows of a sector on four devices takes turns, and
    only where :func:`concurrent_exchange_requested` would have issued its
    copies together.  Of that way only the chunks that every device sends
    to every other one, the twelve copies that were timed: once a device
    has sent all its columns, the chunks that the others still have go
    together as before.  Those have not been timed in turns.  With one
    sender a turn would be a single copy where three started at once, and
    a receiver alone took three sources side by side at three times the
    rate of one.  The step with one tall block, the default, gives all
    devices equally many slabs, and in the bases of 3,480 to 39,368
    electrons on 16 GPUs every chunk has four senders, with the equal slabs
    that the turns were timed with and with the whole rounds of
    :func:`block_rounds`, which are more and whose pieces are smaller.  The
    slab of a whole round can be just wider than a chunk
    (:func:`exchange_chunk_bytes`), 32 columns against 31 for 39,368
    electrons, and then travels as a chunk and a second one of a column,
    each in its three turns.  With those rounds the turns have run on 16
    GPUs for 3,480, 23,768 and 39,368 electrons: the way to the rows of a
    sector of 39,368 electrons took 17.4 s, against 21.6 s for equal slabs
    received at once in the same allocation and 16.5 s for equal slabs in
    turns in another (``README.md``).
    The slabs of the step with two blocks end with their column range,
    and there up to a quarter of the bytes of such a basis travel in
    chunks of fewer senders.  Two devices are one pair already.  Sectors
    on three devices or on more than four have not been timed and keep
    their order.  Neither does the way back change.  There every device
    fetches chunk after chunk at its own pace, with no wait for the
    others, and the stage seconds of runs on 16 GPUs give its four devices
    75 to 81 GiB/s each once the cost of a round is taken out, 61 if none
    is.  Turns would bring that way three waits for all devices per chunk
    where it has none, about what they could save.  Pairs that follow each
    other without such a wait have not been timed.
    """

    value = os.environ.get("PARSEC_CUPY_EXCHANGE_PAIRS", "1").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_EXCHANGE_PAIRS must be on or off")
    return value in {"1", "on", "true"}


def condition_helper_policy() -> str:
    """``auto``, ``any`` or ``off`` from ``PARSEC_CUPY_RITZ_CONDITION_HELPER``.

    The device small solve begins with the condition number of the overlap:
    a symmetric spectrum that is only compared with its limit and takes
    about as long as the factorization and the eigensolve that follow (0.09
    of 0.20 s at 2,978 states on an A100), while the other devices of the
    sector wait.  ``auto``, the default, has another device of the sector
    estimate it meanwhile (:meth:`SectorDeviceGroup._condition_helper`,
    :func:`rayleigh_ritz.solve_whitened_ritz_on_device`): the audits, their
    order and their errors stay what they were and the coefficients are
    the same bit for bit.  The device that holds the Hartree objects of the
    process, its fullest one, is not asked; ``any`` asks it too, where a
    sector has no other.  ``off`` is the former route, the estimate on the
    owner before the factorization.  A sector on one device has no helper,
    and neither has a host solve or a run whose
    ``PARSEC_CUPY_RITZ_CONDITION`` names the host SVD.

    The helper takes memory that it did not take before.  Its copy of the
    overlap and the copy that cuSOLVER overwrites fit what its part of the
    Gram arrays has just freed; the work array of the spectrum does not.
    With CUDA 12.9 on a laptop cuSOLVER asks for 4.05 ``m x m`` arrays
    (274 MiB at 2,978 states, 106 MiB at 1,842), by which the pool of the
    helper grows once: the block stays in the free list of the stream of
    its group for the next step, since the density step leaves the pools
    of a shared basis as they are (:func:`shared_pool_release_requested`;
    with ``PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE=1`` they are emptied
    after every density and it grows in every SCF step again).  The first
    estimate of a thread takes 64 MiB outside the pool for its cuSOLVER
    handle.  That would be about 0.33 GiB more on the helper for 23,768
    electrons and 0.48 for 29,576, on a device that peaked 1.2 and 1.7 GiB
    below the owner of its sector: the largest device of a run should stay
    what it was.  None of this is measured on an A100.  The solve with a
    helper has run where one device stood for both, with the same bits; no
    step of a sector on several devices has run with one yet.
    """

    value = os.environ.get("PARSEC_CUPY_RITZ_CONDITION_HELPER", "auto").strip().lower()
    if value not in {"auto", "any", "off"}:
        raise ValueError("PARSEC_CUPY_RITZ_CONDITION_HELPER must be auto, any or off")
    return value


def tall_blocks_per_device() -> int:
    """Tall blocks a device of a shared basis holds in the Ritz step: 1, 2 or 3.

    ``PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS`` selects them; 1 is the default.
    With three, ``H`` times the columns of a device is a tall block of its
    own next to the columns and the row block of the basis.  With two it
    exists a slab of columns at a time only (see
    :meth:`SectorDeviceGroup.ritz`), and a slab workspace
    (:func:`slab_workspace`) takes the place of the third block.  The
    projection is then summed slab by slab, so the results of the two differ
    by round-off.  Two blocks ran 23,768 electrons on 16 GPUs with a peak of
    55.3 GiB instead of 68.3 in 140.7 against 138.9 s, and hold on 8 GPUs a
    sector basis of 52 GiB that three blocks leave on its owner.

    With one, the row block has no memory of its own either: the columns of
    a device leave their block a slab at a time and the rows arrive in the
    memory that the slabs before have left
    (:meth:`SectorDeviceGroup._ritz_one_block`).  The same sums are again
    cut into other pieces, so the results differ by round-off.  First runs
    on A100 nodes, one block against two: 23,768 electrons on 16 GPUs peaked
    at 33.3 GiB per device in 135.5 s, against 55.3 GiB and 137.1 s with
    slabs of the full budget and 57.9 GiB and 133.9 s with equal slabs;
    19,392 electrons on 8 GPUs at 37.8 instead of 65.1 GiB in 140.9 against
    140.2 s, and on 16 at 23.8 GiB in 88.5 s (38.3 GiB and 86.4 s with two
    blocks in another allocation); 23,768 electrons ran on 8 GPUs, at 54.9
    GiB in 223.9 s, where two blocks do not fit.  That is why one block is
    the default.  Those runs used an earlier form of the step.  In its
    present form, named 1 beside the other defaults of that time, it ran
    23,768 electrons on 16 GPUs at 33.2 GiB in 114.1 and 114.4 s (two
    blocks: 57.9 GiB and 113.1 s with equal slabs, 55.2 GiB and 116.6 s with
    full ones) and 10,456 on 8 at 16.5 GiB in 45.9 s (26.4 GiB and 46.4 s),
    and also 3,480 and 14,680 electrons on 16 GPUs and 19,392 and 23,768 on
    8; no pass took a row block from the pool.  The default itself, the
    variable unset, has run since: 3,480 to 29,576 electrons on 8 GPUs and
    3,480 to 39,368 on 16 in one series of first runs, with one tall block
    in each of its 68 shared sectors and no row block from the pool
    (``README.md``).
    """

    value = os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS", "1").strip()
    if value not in {"1", "2", "3"}:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS must be 1, 2 or 3")
    return int(value)


def equal_slabs_requested() -> bool:
    """Whether a column range is cut into slabs of ``H X`` of equal width.

    The default, and a choice of the two-block Ritz step
    (``PARSEC_CUPY_DISTRIBUTED_STATE_BLOCKS=2``) only: the step with one
    tall block, the default, cuts equal slabs of its own budget
    (:func:`slab_columns`) whatever this says.
    ``PARSEC_CUPY_DISTRIBUTED_STATE_SLABS=full`` selects the
    former cut of the two-block Ritz step: slabs of the full width of the
    budget (:func:`slab_columns`, at most 4 GiB) and what is left of the
    range as a narrower last one.  A sector of 23,768 electrons on 16 GPUs
    holds column ranges of about 372 columns and a slab of 4 GiB has 141:
    its ranges were cut into 141, 141 and 90, and with slabs of 189, which
    cut them into two, the same run took 133.1 instead of 135.8 s for 2.7
    GiB more.  Cut into equal slabs, a range has no short last one
    (:func:`projection_slabs`), a slab may take up to 6 GiB where the device
    has the room (:func:`wider_slabs_fit`), which cuts those ranges into
    two of 184 to 189 columns, and the slab workspace has the size of the
    slabs that were cut instead of that of the budget.  The projection is
    summed over other slabs, and the rows are rotated in tiles of another
    height, since a tile is as tall as a slab of the workspace lets it be:
    the results of the two cuts differ by round-off.  The step with one
    tall block cuts equal slabs either way (:func:`block_rounds`); its
    workspace and its rotation tile follow the slabs that were cut too.
    """

    value = os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE_SLABS", "equal").strip().lower()
    if value not in {"equal", "full"}:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE_SLABS must be equal or full")
    return value == "equal"


def shared_pool_release_requested() -> bool:
    """Whether the density step empties the memory pools of the devices of a shared basis.

    ``PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE=1`` has it empty them
    after every density, as before
    (:meth:`CuPySymmetryDensityBuilder._release_large_unused_pool`).  ``0``,
    the default, leaves them: they hold nothing that the next step does not
    take again.

    The blocks of a shared basis stay where the trial basis was created: a
    filter writes into them, the Ritz step lays the rows into them and
    returns the rotated columns in them, and a trim keeps their leading
    columns.  What a step takes beside them, the slab workspace, the
    exchange buffer and the Gram arrays, it takes on the thread of each
    device under the stream of its group, the first two in capacities that
    only grow so that the pool block of one pass serves the next
    (:meth:`SectorDeviceGroup._capacities`,
    :meth:`SectorDeviceGroup._workspace_capacity`), and gives back to the
    free list of that stream when it ends.  The filter between two Ritz
    steps takes nothing from the pool: it works in the blocks, and the
    buffers of its graphs are allocations of their own
    (:mod:`filter_graph`).  The next step therefore asks the same free
    lists for the same blocks, and a release in between returned them to
    the driver only to have them taken from it again.  The largest step of
    a shared basis is its first solve, which precedes every release: what
    it leaves in the pools was there at the peak.

    One block serves no later request: the workspace of the passes before
    where a pass needs a larger one, as a trim can bring about by moving
    the cut of the slabs.  The group returns the free blocks of its stream
    on every device before it takes the larger workspace
    (:meth:`SectorDeviceGroup._slab_workspace`), which is what the release
    after the density did for that pass.

    First runs on A100 nodes with the release switched off for the whole
    process against the default, where all sectors are shared (the limit
    ``PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES`` above every basis): the
    density stage took 0.58 instead of 1.70 s for 14,680 electrons on 16
    GPUs and 0.89 instead of 1.98 for 23,768, and on 8 GPUs 1.77 instead
    of 3.05 s for 29,576, 1.04 instead of 2.14 for 14,680 and 0.32 instead
    of 0.60 for 5,264 electrons; the later solves, which found their
    workspace in the pool, took 0.3 to 0.8 s less; the fullest device held
    2 to 48 MiB more (15.4, 28.8, 69.2, 28.0 and 6.4 GiB) and the total
    energies were the same to the last bit.  The runs on 8 GPUs cut slabs
    of 4 GiB, as the code did then: 29,576 electrons peak at 65.5 GiB and
    14,680 at 24.8 since (:func:`pair_slab_bytes`).  Emptying the pools on
    the threads of the devices (``PARSEC_CUPY_DENSITY_RELEASE``) had taken
    0.04 to 0.35 s off that stage.

    The default itself, by sector, ran there since against ``1`` in one
    allocation, with those slabs: the density stage took 0.88 instead of
    1.56 s for 14,680 electrons on 8 GPUs, 1.85 instead of 2.64 for 23,768
    and 1.68 instead of 2.70 for 29,576, and on 16 GPUs 0.80 instead of
    1.23, 1.01 instead of 1.67 and 0.99 instead of 1.89 s; the fullest
    device held 2 to 48 MiB more (67,083 MiB for 29,576 electrons on 8
    GPUs) and the total energies were the same to the last bit.

    A process that returns nothing keeps host memory as well.  It no
    longer collects after a density, so its cyclic garbage waits for the
    collector of the interpreter, and the pinned pool keeps the blocks
    through which host arrays go to a device.  In the runs above the root
    rank had a high-water mark 0.3 to 1.1 GiB higher (4.72 instead of 4.10
    GiB for 14,680 electrons on 16 GPUs, 6.96 instead of 5.99 for 29,576
    on 8) and the other ranks held 0.1 to 0.4 GiB more at the end.  On a
    workstation it was the pinned pool that held it, 17 free blocks of 50
    MiB together after a run of 864 electrons, and a collection returned
    nothing (``MULTI_GPU.md``).

    A sector on one device keeps the release.  It takes its arrays in the
    sizes of the step, with no capacity that is kept for the next
    (:func:`rayleigh_ritz._streaming_workspace`), and what its first solve
    leaves is not all taken again: with the release off 14,680 electrons
    on 4 GPUs peaked at 45.3 instead of 42.1 GiB, and 10,456 at 23.0
    either way.
    """

    name = "PARSEC_CUPY_DISTRIBUTED_STATE_POOL_RELEASE"
    value = os.environ.get(name, "0").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError(f"{name} must be on or off")
    return value in {"1", "on", "true"}


def pair_slab_bytes() -> int:
    """The most bytes of a slab of the one-block Ritz step of a sector on two devices.

    ``PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES`` sets it; the default
    is 2 GiB (see :func:`slab_columns`).  It limits the slab that the
    streaming budget sizes where that is not set, an eighth of the basis
    but at least 1 GiB, and nothing else: a budget that is set is the slab
    as before, and one device, three and more, and the steps with two and
    three tall blocks never read it.

    ``4294967296`` selects the former slabs, at most 4 GiB, for a basis of
    any size and in every step of a run, bit for bit: the limit is then
    that of the streaming budget itself.  It is the way back of a sector
    on two devices, beside ``PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1`` since the
    rounds are cut at whole multiples of 64 columns (:func:`block_rounds`).
    ``PARSEC_CUPY_STREAMING_RITZ_BYTES=4294967296``, that of four devices,
    is not: a basis below 32 GiB took an eighth of itself per slab and
    takes 4 GiB with it, also one that a trim has taken below 32 GiB after
    its first solve.
    """

    raw = os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES", str(_PAIR_SLAB_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES must be an integer") from error
    if value < 1:
        raise ValueError("PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES must be positive")
    return value


def slab_columns(rows: int, columns: int, devices: int, wide: bool = False, blocks: int = 2) -> int:
    """The most columns of one slab of ``H X`` in a Ritz step that forms it in slabs.

    A slab of all rows takes the bytes that the one-array route gives its
    projection slab (:func:`rayleigh_ritz._streaming_bytes` of the whole
    basis: ``PARSEC_CUPY_STREAMING_RITZ_BYTES``, or an eighth of the basis
    but at least 1 and at most 4 GiB; at most 6 GiB if ``wide``, which
    :func:`wider_slabs_fit` decides).  It is never wider than half of an
    even share of the columns: a device holds two slabs at a time (see
    :func:`slab_workspace`), which then take no more than the tall block
    they replace.  The width does not depend on how the columns are split
    between the devices.

    In the step with one tall block (``blocks`` of 1) on four or more
    devices, a slab that no set budget sizes takes at most an even share
    of 8 GiB as well: 2 GiB with four devices.  A round of that step
    projects the slabs of all devices in one product, so the width of a
    round and not that of a slab is the width of the product, and the
    workspace is two slabs.  First runs on A100 nodes, four devices per
    sector, with the budget set to 2 GiB against the 4 GiB it was: 23,768
    electrons on 16 GPUs peaked at 30.1 instead of 33.2 GiB per device
    and their eigensolver took 94.2 s instead of the 93.1 to 93.2 s of six
    runs with 4 GiB (rounds of about 271 instead of 496 columns); 29,576
    electrons peaked at 37.9 instead of 41.3 GiB and their eigensolver
    took 137.9 instead of 140.2 s (247 instead of 463).  The smaller slabs
    thus return 3.2 and 3.4 GiB per device, cost the first run about 1%
    of its time and gain the second about 1.5%: which way the time of
    another basis moves is not known.  The cut also changes for 14,680,
    19,392 and 39,368 electrons on 16 GPUs (rounds of at most 371, 349 and
    184 columns instead of 614, 609 and 354); the bases of 10,456
    electrons and fewer are cut as before.  The default itself, the budget
    unset, has run on 16 GPUs since, 3,480 to 39,368 electrons in one
    series of first runs (``README.md``), and 19,392 electrons beside
    slabs of 4 GiB in one allocation: 65.8 against 66.0 s of SCF.  The
    rounds named here are those of a multiple of 1: the default cuts them
    further, at whole multiples of 64 columns (:func:`block_rounds`), into
    rounds of 320, 320 and 128 columns for these three.

    On two devices a slab that no set budget sizes takes at most 2 GiB as
    well (:func:`pair_slab_bytes`): rounds of 4 GiB.  They were left at 4
    GiB at first, because rounds of about 135 columns had not run and the
    products of the two-block step of 23,768 electrons on 16 GPUs took
    17.5 s where they were 90 to 141 columns wide and 14.5 s at 184 to
    189.  First runs on A100 nodes, two
    devices per sector, with the budget set to 2 GiB against the 4 GiB it
    was: 23,768 electrons on 8 GPUs peaked at 49.7 instead of 53.5 GiB per
    device and took 181.8 instead of 179.7 s (rounds of 136 instead of 271
    columns, the Gram stage 35.5 to 36.0 instead of 33.2 to 33.4 s per
    sector); 29,576 electrons peaked at 65.5 instead of 69.2 GiB and took
    256.2 instead of 258.7 s (128 instead of 248 columns, 52.1 to 52.7
    instead of 55.1 to 55.3 s).  The smaller slabs thus return 3.8 and 3.7
    GiB per device, cost the first run 1.2% of its time and gain the
    second 1.0%: the sign differs here too.  With 3 GiB the two peaked at
    51.5 and 67.2 GiB and took 181.4 and 258.3 s.  The cut also changes
    for 14,680 and 19,392 electrons on 8 GPUs (rounds of at most 205 and
    174 columns instead of 369 and 348).  The default itself, the budget
    unset, has run on 8 GPUs since, for these two and the two above: the
    fullest device peaked at 25,373, 33,643, 50,859 and 67,031 MiB, 3.2
    to 3.8 GiB below the former slabs (``README.md``).
    The limit is the same for a basis of any size.  Up to 16 GiB a slab is
    the eighth of the basis that it was, at least 1 GiB; from there to 32
    GiB it took that eighth and now takes 2 GiB, in five to nine rounds
    instead of four or five, where no cluster has run.  The sectors of
    10,456 electrons on 8 GPUs (18.8 GiB) keep their five rounds of 132
    columns, and those of smaller clusters their cut.  A limit that began
    at 32 GiB, where a slab took the full 4 GiB, was tried first and
    dropped: a basis just below held a workspace of nearly 8 GiB and one
    just above 4, so that the few columns which a trim takes from a sector
    after its first solve could double the workspace that the fit rule
    had counted for it (:func:`shared_basis_fits` is asked once, for the
    first solve).  With one limit a slab never narrows as columns are
    added, and what the rule counts for the first solve is the most that
    any later step of the sector holds.

    Three devices keep 4 GiB at every size: no run has had them.

    ``PARSEC_CUPY_STREAMING_RITZ_BYTES=4294967296`` gives a sector on four
    or more devices the slabs it had, whatever the size of its basis.  It
    is the way back for those runs only, and what it gives back is this
    width: the rounds are cut from it at whole multiples of 64 columns
    (:func:`block_rounds`) unless ``PARSEC_CUPY_RITZ_GRAM_MULTIPLE=1`` is
    named beside it.  A sector on two devices has its former slabs with
    ``PARSEC_CUPY_DISTRIBUTED_STATE_PAIR_SLAB_BYTES=4294967296``
    (:func:`pair_slab_bytes`), and one on three devices or on one without
    either.  The budget is not theirs: a basis below 32 GiB took an eighth
    of itself per slab, on two or three devices and in the one-array route
    of a sector on one, and takes 4 GiB with it.  For a sector on two
    devices it gives back the slabs of the steps in which the basis holds
    32 GiB or more, as in the runs above, and no others.
    """

    rows, columns, devices = int(rows), int(columns), int(devices)
    basis = 8 * rows * columns
    if int(blocks) == 1:
        budget = _streaming_bytes(basis)
        if devices >= 4:
            budget = min(budget, _streaming_bytes(basis, _ROUND_BYTES // devices))
        elif devices == 2:
            budget = min(budget, _streaming_bytes(basis, pair_slab_bytes()))
    else:
        budget = _streaming_bytes(basis, _WIDER_SLAB_BYTES) if wide else _streaming_bytes(basis)
    return max(1, min(budget // (8 * rows), columns // (2 * devices)))


def slab_workspace(rows: int, columns: int, devices: int, blocks: int = 2, wide: bool = False) -> int:
    """Elements of the slab workspace of a device for slabs as wide as the budget allows.

    It holds ``H`` times a slab of the device's own columns in all rows and,
    next to it, the rows of the device of the slabs of all devices.  In the
    step with one tall block (``blocks`` of 1) that second part also holds
    the slab itself in all rows while it leaves its block, so it is never
    smaller than the first: a difference for fewer columns than devices
    only.  ``wide`` and ``blocks`` are those of :func:`slab_columns`, which
    gives the slabs of the step with one tall block a budget of their own.
    Slabs that are cut equal can be narrower than the budget allows, and
    their workspace smaller than this
    (:meth:`SectorDeviceGroup._cut_capacity`).
    """

    rows, columns, devices = int(rows), int(columns), int(devices)
    width = slab_columns(rows, columns, devices, wide, blocks)
    # No row block is taller (see SectorDeviceGroup.row_edges).
    tallest = -(-rows // devices)
    arriving = tallest * min(columns, devices * width)
    if int(blocks) == 1:
        arriving = max(arriving, rows * width)
    return rows * width + arriving


def block_rounds(widths, width: int, multiple: int = 1) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Columns of its block that each device re-lays per round of the one-block Ritz step.

    Per round, one ``(first, last)`` range of columns of its own block for
    each device, whose block has ``widths[index]`` columns.  The widest
    block is cut into slabs of at most ``width`` columns and every other
    block into as many, each into slabs as equal as they can be: all
    devices take part in every round, none is left with a short last slab,
    and the columns that a round brings together are as many as they can
    be.  A device gives its last columns first, so that its block empties
    from the end (see :meth:`SectorDeviceGroup._ritz_one_block`), and in
    every round at least the share of its columns that the rounds so far
    are of all rounds.

    That is the cut of a ``multiple`` of 1.  The projection of a round is
    one product as wide as the columns that the round brings together, and
    such a product costs as if that width were rounded up to a multiple of
    64 (:func:`rayleigh_ritz.gram_multiple`, which the step names as
    ``multiple``).  With a multiple above 1 the rounds are therefore whole
    ones as far as the blocks go: every device gives the same slab in each
    of them, the widest that makes the round a multiple, 32 or 64 columns
    each of two devices and 16, 32, 48 or 64 each of four, and as many
    such rounds as the narrowest block has slabs.  What the blocks hold
    beyond those rounds goes first, cut as above: the product of a round
    is as long as the columns that have arrived by then, so the rounds
    that are no multiple are the cheapest ones.  No slab is wider than
    the widest of the equal ones, so the workspace of the slabs that are
    cut does not grow by it, but for less than a row of a round where the
    blocks are unequal.  The sector of 29,576 electrons on two devices,
    blocks of 1,854 and 1,850 columns in slabs of at most 64, gives 62 and
    58 columns and then 28 rounds of 128 where it gave 29 rounds of 126 to
    128, and that of 23,768 electrons 22 and 12 and then 23 rounds of 128
    for 22 rounds of 134 to 136.  Equal slabs of all devices also keep a
    block as full as it was: the rows that arrive take the place of the
    columns that left (:meth:`SectorDeviceGroup._row_block_ends`).  Where
    no such slab exists, with a block or an equal slab narrower than one,
    the cut is that of a multiple of 1.

    The slab of a whole round can be little more than half of the widest
    equal one, and the rounds nearly twice as many: blocks of 1,857
    columns in slabs of at most 64 gave 30 rounds of 122 to 124 columns,
    which cost as 128, and give 1 and 1 and then 58 rounds of 64, for 3 to
    5% of the projection as the rounded widths have it and half the
    workspace.  A round more is not free: in the runs that halved the
    slabs (:func:`slab_columns`) it took 1.2 and 1.3 ms per pass on two
    devices and 6.1 to 8.3 ms on four, in the stages around the product.
    In first runs on A100 nodes against a multiple of 1 the 39 rounds of
    39,368 electrons on four devices took 5.0 s per sector more in those
    stages than its 27 and 3.2 s less in the products, the 20 rounds of
    29,576 electrons 0.5 s more for 0.8 s less, and on two devices the 24
    rounds of 23,768 electrons took 6.4 s less in the products for nothing
    more around them.
    No sector cut at half slabs has run, and no product narrower than 124
    columns was timed alone.  The group counts the seconds of the overlap
    apart (``SectorDeviceGroup.overlap_seconds``), so that a run against
    a multiple of 1 shows what the rounds gave and what they cost.
    """

    widths = tuple(int(held) for held in widths)
    multiple = int(multiple)

    def equal(held, kept, most):
        """Rounds of equal slabs of at most ``most`` of the last ``held[index]`` columns, behind the first ``kept``."""

        count = max(1, -(-max(held) // most))
        return [
            tuple((kept + size - -(-size * (step + 1) // count), kept + size - -(-size * step // count)) for size in held)
            for step in range(count)
        ]

    rounds = equal(widths, 0, max(1, int(width)))
    # The widest of the equal slabs, which the first round takes of the widest block.
    most = max(last - first for first, last in rounds[0])
    # The slabs of all devices together are a multiple where each is one of this.
    unit = multiple // gcd(multiple, len(widths))
    slab = min(most, min(widths)) // unit * unit
    if multiple <= 1 or not slab:
        return tuple(rounds)
    whole = min(widths) // slab
    rounds = equal([held - whole * slab for held in widths], whole * slab, most) if max(widths) > whole * slab else []
    rounds.extend(tuple((slab * step, slab * (step + 1)) for _held in widths) for step in reversed(range(whole)))
    return tuple(rounds)


def projection_slabs(
    layout, width: int, equal: bool | None = None, multiple: int = 1
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Column ranges of the slabs of ``H X``: per round, one range per device.

    ``layout`` gives the column ranges of the devices.  Every range is cut
    into slabs of at most ``width`` columns: as few as that takes and as
    equal as they can be, or, unless ``equal``, slabs of ``width`` columns
    and a narrower last one.  :func:`equal_slabs_requested` says which
    where the caller does not.  In every round each device takes the next
    slab of the range it has got to, one range after the other in the
    order of its block, and an empty range once none is left.  A slab ends
    with the range it lies in, so its columns are neighbours in the basis.

    Every slab is the right-hand side of a product of its own in the
    two-block step.  With a ``multiple`` above 1
    (:func:`rayleigh_ritz.gram_multiple`, which that step names) a slab of
    that many columns and more is cut down to a whole multiple, the slab
    of ``width`` columns or the widest of the equal ones, and a range ends
    with a narrower one either way: ranges of 372 columns that were cut
    into two slabs of 186 are cut into 128, 128 and 116.
    """

    layout = _as_layout(layout)
    width, multiple = max(1, int(width)), int(multiple)
    if equal is None:
        equal = equal_slabs_requested()

    def cut(start, stop):
        slab = width
        if equal:
            count = -(-(stop - start) // width)
            slab = -(-(stop - start) // count)
            if multiple <= 1 or slab < multiple:
                edges = [start + (stop - start) * index // count for index in range(count + 1)]
                return list(zip(edges[:-1], edges[1:]))
        slab = _whole_multiple(slab, multiple)
        return [(first, min(stop, first + slab)) for first in range(start, stop, slab)]

    slabs = [[slab for start, stop in part if stop > start for slab in cut(start, stop)] for part in layout.parts]
    ends = [part[-1][1] if part else 0 for part in layout.parts]
    return tuple(
        tuple(taken[step] if step < len(taken) else (end, end) for taken, end in zip(slabs, ends))
        for step in range(max(map(len, slabs)))
    )


def shared_basis_fits(rows: int, columns: int, devices: int) -> bool:
    """Whether the blocks of a shared ``rows x columns`` basis fit one device.

    The rule counts what :func:`tall_blocks_per_device` says a device holds.
    With one tall block, the default, the row block lies in the column block
    (:meth:`SectorDeviceGroup._ritz_one_block`), which is given about one
    column more for it: one tall block is counted next to the slab workspace
    (:func:`slab_workspace`) and one exchange chunk (see
    :func:`exchange_chunk_bytes`), half of the basis with two devices and a
    quarter with four.  With two, the filtered columns and one row block
    are counted next to the workspace and the chunk: as much as the whole
    basis with two devices, half of it with four.  With three, a tall block
    for ``H`` times the columns is counted in place of the workspace: 1.5
    and 0.75 times the basis.
    A sector basis of 84.0 GiB counts 26.0 GiB on four devices and
    47.0 on two with one tall block, where two count 51.0 and 93.0, and four
    devices of 79.25 GiB share a basis of up to 249 GiB instead of 117.
    Balanced ranges (see :func:`keeps_column_ranges`) add headroom and one
    more block, the new columns.  The workspace counted is that of slabs as
    wide as the budget allows (:func:`slab_columns`): 4 GiB at most, and 2
    GiB in the step with one tall block on four devices and on two.  While
    its slabs took 4 GiB that step counted 30.0 and 51.0 GiB for that
    basis and shared up to 233 GiB on four devices and 117 on two, which
    now share up to 125.  The rule is asked for the columns of a first
    solve; the passes that follow hold as many or fewer, and no budget
    cuts fewer columns into wider slabs.  The two-block step cuts wider
    ones only where the blocks fit with those too
    (:func:`wider_slabs_fit`).

    They fit if they take no more of the current device's memory than
    ``PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION``, 0.85 by default.  The
    rest is for the operators, the filter buffers, the Gram matrices and the
    small solve: a basis with 50.9 GiB of blocks peaked at 55.3 GiB of the
    79.25 GiB of its devices.

    The rule keeps no room for the fallback after a failed stability audit
    (:meth:`SectorDeviceGroup.orthonormal_ritz`), which gathers the whole
    basis on the owner next to the owner's columns: ``1 + 1/devices`` times
    the basis.  A basis can therefore be shared although that fallback could
    not run.  With two tall blocks that holds on two devices as well, where
    three always left the room: a sector basis of 52.2 GiB on two devices of
    79.25 GiB fits, and its fallback would need 78.3 GiB on the owner.  One
    tall block shares bases that no device could gather at all.
    """

    return _blocks_fit(rows, columns, devices, False)


def _auto_fraction() -> float:
    """The share of a device's memory that the blocks of a shared basis may take.

    ``PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION``, 0.85 by default (see
    :func:`shared_basis_fits`).
    """

    message = "PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION must be in (0, 1]"
    try:
        fraction = float(os.environ.get("PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION", "0.85"))
    except ValueError as error:
        raise ValueError(message) from error
    if not 0.0 < fraction <= 1.0:
        raise ValueError(message)
    return fraction


def shared_block_bytes(rows: int, columns: int, devices: int, wide: bool = False) -> int:
    """Bytes that :func:`shared_basis_fits` counts on every device of a shared basis.

    The tall blocks, the exchange chunk and the slab workspace of a
    ``rows x columns`` basis on ``devices`` devices, as that rule describes
    them; ``wide`` is that of :func:`slab_workspace`.  The rule that decides
    where the states of the symmetry sectors are kept counts a shared basis
    by them (:meth:`symmetry.CuPySymmetrySCFEigensolver._state_fit_bytes`).
    """

    rows, devices = int(rows), int(devices)
    even = -(-rows * int(columns) // devices)
    fixed = keeps_column_ranges()
    tall = even if fixed else even + even // 4
    # One chunk, or all of a block outside the device's own rows if that is less.
    away = rows - rows // devices
    chunk = min(tall - tall // devices, max(exchange_chunk_bytes() // 8, away))
    blocks = tall_blocks_per_device()
    needed = 8 * ((blocks if fixed else blocks + 1) * tall + chunk)
    if blocks != 3:
        needed += 8 * slab_workspace(rows, columns, devices, blocks, wide)
    if blocks == 1:
        # The room in which a row block may end behind its column block.
        needed += 8 * devices * -(-rows // devices)
    return needed


def _blocks_fit(rows: int, columns: int, devices: int, wide: bool) -> bool:
    """:func:`shared_basis_fits` for the slab workspace of :func:`slab_workspace` with this ``wide``."""

    cp, _ = require_cupy()
    fraction = _auto_fraction()
    needed = shared_block_bytes(rows, columns, devices, wide)
    _free, total = cp.cuda.runtime.memGetInfo()
    return needed <= fraction * total


def wider_slabs_fit(rows: int, columns: int, devices: int) -> bool:
    """Whether the two-block Ritz step of this basis may cut slabs of up to 6 GiB.

    It may if the slabs are cut equal (:func:`equal_slabs_requested`), if
    the budget is not set (``PARSEC_CUPY_STREAMING_RITZ_BYTES`` is the
    width to cut by where it is) and an eighth of the basis is more than 4
    GiB, and if the blocks fit the current device with a workspace of two
    such slabs as :func:`shared_basis_fits` counts them.  A basis that fits
    with slabs of 4 GiB only keeps those, so the rule shares the same bases
    as before.  The slabs that are then cut are as equal as they can be and
    mostly narrower: 184 to 189 columns, 5.3 GiB, for the ranges of 368 to
    378 columns of 23,768 electrons on 16 GPUs, where 6 GiB are 212.

    Every two-block step asks this, also of a basis that
    ``PARSEC_CUPY_DISTRIBUTED_STATE=1`` shares without asking whether it
    fits.  ``PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION`` then decides
    nothing about the sharing but still the width of the slabs, and
    :meth:`SectorDeviceGroup._capacities` reads it where the trial basis is
    sized, so that a mistyped value stops the run there and not in the first
    Ritz step.  The rule counts against the memory of the device and knows
    no limit of the memory pool (``CUPY_GPU_MEMORY_LIMIT``): a run under
    such a limit names the former cut or sets the budget.

    The step with one tall block has no wider slabs.  A round of it brings
    the slabs of all devices together and projects them on all columns that
    have arrived, the fewer rounds the more of them, and it is chosen for
    its memory: its slabs take 4 GiB at most, and less with two devices
    and with four or more (:func:`slab_columns`).
    """

    if tall_blocks_per_device() != 2 or not equal_slabs_requested():
        return False
    # Nothing to decide, and no device to ask, where the wider budget cuts no wider.
    if slab_columns(rows, columns, devices, True) <= slab_columns(rows, columns, devices):
        return False
    return _blocks_fit(rows, columns, devices, True)


class DistributedBasis:
    """An ``N x m`` basis kept as one block of columns per device.

    ``blocks[i]`` lives on ``group.devices[i]`` and holds in Fortran order,
    side by side, the column ranges that ``layout`` gives the device (see
    :class:`ColumnLayout`); a block may be empty.  Without a ``layout`` the
    blocks are consecutive ranges of columns in the order of the devices.
    Only leading-column selections ``basis[:, :count]`` are supported, and
    they are views of the same device memory: the leading columns of every
    block.
    """

    ndim = 2

    def __init__(self, group: "SectorDeviceGroup", blocks, layout=None) -> None:
        blocks = tuple(blocks)
        if len(blocks) != len(group.devices):
            raise ValueError("one column block per device is required")
        rows = {int(block.shape[0]) for block in blocks}
        if len(rows) != 1 or any(block.ndim != 2 for block in blocks):
            raise ValueError("column blocks must share one row count")
        for device, block in zip(group.devices, blocks, strict=True):
            if int(block.device.id) != device or not block.flags.f_contiguous:
                raise ValueError("column blocks must be Fortran-ordered on their own device")
        widths = tuple(int(block.shape[1]) for block in blocks)
        if layout is None:
            offsets = [0]
            for width in widths:
                offsets.append(offsets[-1] + width)
            layout = ColumnLayout(tuple(zip(offsets[:-1], offsets[1:])))
        else:
            layout = _as_layout(layout)
            if layout.widths != widths:
                raise ValueError("column blocks do not have the widths of their layout")
        self.group = group
        self.blocks = blocks
        self.layout = layout
        self.shape = (rows.pop(), sum(widths))

    @property
    def ranges(self) -> tuple[tuple[int, int], ...]:
        """The column range of each device, where each holds one (see :attr:`layout`)."""

        return self.layout.ranges

    @property
    def nbytes(self) -> int:
        return sum(int(block.nbytes) for block in self.blocks)

    def __getitem__(self, key) -> "DistributedBasis":
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or key[0] != slice(None)
            or not isinstance(key[1], slice)
            or key[1].start not in (None, 0)
            or key[1].step not in (None, 1)
        ):
            raise IndexError("a distributed basis supports only basis[:, :count]")
        layout = self.layout.leading(len(range(self.shape[1])[key[1]]))
        return DistributedBasis(
            self.group,
            (block[:, :width] for block, width in zip(self.blocks, layout.widths, strict=False)),
            layout,
        )

    def take(self) -> list:
        """Hand the blocks to the caller and forget them."""

        blocks, self.blocks = list(self.blocks), ()
        return blocks

    def density(self, builder, occupations: np.ndarray, volume_element: float) -> np.ndarray:
        """Sum ``builder`` over the devices for the leading ``len(occupations)`` columns."""

        weights = np.asarray(occupations, dtype=np.float64)
        if weights.ndim != 1 or weights.size > self.shape[1]:
            raise ValueError("occupations do not select leading columns of this basis")
        selected = self[:, : weights.size]

        def partial(index, device):
            block = selected.blocks[index]
            if not block.shape[1]:
                return None
            # The weights of the columns of the block, in the order in which it holds them.
            held = [weights[column : column + count] for column, _offset, count in selected.layout.spans(index)]
            return builder(block, held[0] if len(held) == 1 else np.concatenate(held), volume_element)

        total = None
        # Fixed device order keeps the host sum deterministic.
        for part in self.group.map(partial):
            if part is not None:
                total = part if total is None else total + part
        if total is None:
            raise ValueError("no occupied column was selected")
        return total


class SectorDeviceGroup:
    """Devices, replicated operators and filter graphs serving one sector."""

    STAGES = ("filter", "apply", "to_rows", "gram", "dense", "rotate", "to_columns", "repartition")

    def __init__(self, operator: Any, devices: tuple[int, ...]) -> None:
        worker = getattr(operator, "_distributed_filter", None)
        if worker is None or tuple(worker.devices) != tuple(devices):
            worker = DistributedFilter(operator, tuple(devices))
            operator._distributed_filter = worker
        self._worker = worker
        self._owner_ref = ref(operator)
        self.devices = tuple(int(device) for device in worker.devices)
        self.owner = int(worker.owner)
        if self.devices[0] != self.owner:
            raise ValueError("the sector owner must be the first device of its group")
        self.seconds = dict.fromkeys(self.STAGES, 0.0)
        self.passes = 0
        # Tall blocks per device of the latest Ritz step, for reports: the
        # setting is read at every step, and unset it has meant 2 and 1.
        self.tall_blocks = None
        # How that step cut the slabs of ``H X``, ``equal`` or ``full``, and
        # the columns of its widest slab, for reports: the step with one
        # tall block does not read the switch of the cut, and says
        # ``multiple`` where it cut whole rounds (:func:`block_rounds`).
        # None for a step with three tall blocks, which cuts no slab.
        # Also the most columns of the right-hand side of a product of its
        # projection: of a round with one tall block, of a slab with two.
        self.slab_cut = self.slab_columns = self.projection_columns = None
        # Row blocks that a step with one tall block could not lay into
        # the allocation of their columns, for reports.
        self.separate_row_blocks = 0
        # The layout of the basis that was filtered last, for reports.
        self.layout = None
        # The device that estimated the condition number of the overlap in
        # the latest device small solve, the owner or its helper, and the
        # seconds that helpers have spent on it beside the solve of the
        # owner, for reports.
        self.condition_device = None
        self.condition_seconds = 0.0
        # The order in which the devices issued the copies of the latest way
        # to their rows, for reports: ``pairs`` (the chunks that all four
        # send in turns, see :func:`paired_exchange_requested`),
        # ``together`` or ``queued``.
        self.to_rows_copies = None
        # The seconds of ``gram`` that the overlap took, for reports: what
        # is left of that stage is the projection, whose products are as
        # wide as the rounds of a step with one tall block and as the slabs
        # of a step with two.  The widths of the two matrices are cut apart
        # (:func:`rayleigh_ritz._gram_slab_width`, :func:`block_rounds`,
        # :func:`projection_slabs`) and one setting moves both
        # (:func:`rayleigh_ritz.gram_multiple`): this says which of them a
        # change of the stage comes from.  A step with three tall blocks
        # forms both matrices in one call and counts nothing here.
        self.overlap_seconds = 0.0
        self._tall = self._chunk = self._work = 0
        # The devices that still hold, in the free list of their stream, a
        # slab workspace that a later pass has outgrown.
        self._outgrown = set()
        self._inbound = {}

    # ----- plumbing -----------------------------------------------------

    def _operator(self, device: int):
        if device == self.owner:
            operator = self._owner_ref()
            if operator is None:
                raise ReferenceError("sector operator has been released")
            return operator
        return self._worker.replicas[device]

    def map(self, function):
        """Run ``function(index, device)`` on each device's thread and stream.

        Every stream is synchronized before this returns, so results may be
        read from any other device or stream afterwards.  The first failure
        is raised only after all devices have finished.
        """

        cp, _ = require_cupy()

        def call(index, device):
            stream = self._worker.streams[device]
            with cp.cuda.Device(device), stream:
                value = function(index, device)
                stream.synchronize()
                return value

        futures = [
            _pool(device).submit(call, index, device)
            for index, device in enumerate(self.devices)
        ]
        wait(futures)
        return [future.result() for future in futures]

    def _timed(self, stage: str, function):
        started = perf_counter()
        try:
            return self.map(function)
        finally:
            self.seconds[stage] += perf_counter() - started

    def _stream(self, device: int):
        return self._worker.streams[device]

    @property
    def stencil_storage(self) -> str:
        """Layout of the stencil that the devices filter with, for reports; ``mixed`` if they differ."""

        modes = {
            self._operator(device).compact_finite_difference.storage_mode for device in self.devices
        }
        return modes.pop() if len(modes) == 1 else "mixed"

    @property
    def replica_seconds(self) -> float:
        """Wall time in which the operators of the other devices were built, for reports."""

        return float(self._worker.replica_seconds)

    def _sum_on_owner(self, parts: list):
        """Sum one array per device into the owner's, in device order.

        The arrays of the other devices arrive by direct device copy and are
        added one after the other, which is the order of the host sum.
        ``parts[0]`` receives the result.
        """

        cp, _ = require_cupy()
        with cp.cuda.Device(self.owner):
            stream = cp.cuda.get_current_stream()
            total = parts[0]
            received = cp.empty(total.shape, dtype=cp.float64)
            for part in parts[1:]:
                if part.shape != total.shape or not part.flags.c_contiguous:
                    raise ValueError("the arrays to sum must be C-ordered and of one shape")
                received.data.copy_from_device_async(part.data, part.nbytes, stream)
                total += received
            # The sources may be released only after every copy has completed.
            stream.synchronize()
        return total

    def _from_owner(self, cp, array, device: int):
        """A Fortran-ordered array of the owner as an array of ``device``.

        Runs on the thread and stream of ``device``; the other devices copy
        device to device, the owner uses the array itself.
        """

        if device == self.owner:
            return array
        if not array.flags.f_contiguous:
            raise ValueError("only a Fortran-ordered array is copied from the owner")
        copy = cp.empty(array.shape, dtype=cp.float64, order="F")
        copy.data.copy_from_device_async(array.data, array.nbytes, self._stream(device))
        return copy

    def _inbound_streams(self, device: int) -> list:
        """Streams of ``device`` for copies arriving from each device of the group.

        Indexed like ``self.devices``, with ``None`` for the device itself.
        Created on first use, on the thread of ``device`` while it is current.
        """

        streams = self._inbound.get(device)
        if streams is None:
            cp, _ = require_cupy()
            streams = self._inbound[device] = [
                None if other == device else cp.cuda.Stream(non_blocking=True) for other in self.devices
            ]
        return streams

    def column_ranges(self, blocks: tuple[FilterBlock, ...]) -> tuple[tuple[int, int], ...]:
        """Contiguous column range per device, balanced by filter work."""

        partitions = block_partitions(blocks, len(self.devices))
        ranges = [(blocks[first].start, blocks[last - 1].stop) for first, last in partitions]
        end = blocks[-1].stop
        ranges.extend([(end, end)] * (len(self.devices) - len(ranges)))
        return tuple(ranges)

    def column_layout(self, blocks: tuple[FilterBlock, ...]) -> ColumnLayout:
        """The layout in which a basis with these filter blocks is created.

        Two ranges per device where :func:`interleaved_layout_requested`
        says so and :func:`interleaved_ranges` has blocks of both halves for
        every device, else the contiguous ranges of :meth:`column_ranges`.
        """

        ranges = interleaved_ranges(blocks, len(self.devices)) if interleaved_layout_requested() else None
        return ColumnLayout(self.column_ranges(blocks) if ranges is None else ranges)

    def row_edges(self, rows: int) -> list[int]:
        count = len(self.devices)
        return [rows * index // count for index in range(count + 1)]

    def _capacities(self, layout, edges) -> tuple[int, int]:
        """Element counts shared by all tall blocks and by all exchange buffers.

        A column block (all rows of some columns) and a row block (some rows
        of all columns) of the same basis differ in size.  A pool block
        released by one layout can serve the next only if it is large enough,
        so every tall block gets one capacity: the largest seen so far.  With
        balanced ranges the column split also changes between the first solve
        and later passes, so the capacity starts with 25% headroom over an
        even split (the largest later share for the usual degree ratio).

        An exchange buffer holds one chunk of the columns of its device for
        the rows of the other devices: :func:`exchange_chunk_bytes` or one
        column, whichever is more, and never more than those rows of a whole
        tall block.  The capacities only ever grow.

        The step with one tall block lays the row block of a device into
        the block of its columns, where it may end a little behind them
        (:meth:`_row_block_ends`): a tall block is then given the room for
        that, a row of all devices per row of the tallest row block, about
        one column.
        """

        count = len(self.devices)
        # The trial basis is sized here before anything is filtered, so the
        # switches of the exchange and of the Ritz step are read too: a
        # mistyped one stops the run now and not inside the first exchange.
        concurrent_exchange_requested()
        paired_exchange_requested()
        column_copies_requested()
        condition_helper_policy()
        gram_multiple()
        shared_pool_release_requested()
        blocks = tall_blocks_per_device()
        if blocks != 3:
            # The slab budget and the cut of those steps, for their form only.
            _streaming_bytes(0)
            if blocks == 1:
                pair_slab_bytes()
            if equal_slabs_requested() and blocks == 2:
                # Such a step asks wider_slabs_fit, which reads the share of
                # a device that the blocks may take: also for a basis that
                # is shared by name and was never held against the fit rule.
                _auto_fraction()
        layout = _as_layout(layout)
        widest = max(layout.widths)
        heights = [high - low for low, high in zip(edges[:-1], edges[1:])]
        rows, columns = edges[-1], layout.columns
        even = -(-rows * columns // count)
        tall = max(rows * widest, max(heights) * columns, even if keeps_column_ranges() else even + even // 4)
        away = rows - min(heights)
        chunk = min(max(away * widest, tall - tall // count), max(exchange_chunk_bytes() // 8, away))
        if blocks == 1:
            tall += count * max(heights)
        self._tall = max(self._tall, tall)
        self._chunk = max(self._chunk, chunk)
        return self._tall, self._chunk

    def _slab_capacity(self, rows: int, columns: int) -> tuple[int, int]:
        """Columns of a slab of ``H X`` and elements of every slab workspace.

        For slabs of the full width of the budget: see :func:`slab_columns`
        and :func:`slab_workspace`.  Like the other capacities the
        workspace only ever grows, so that the pool block of one pass serves
        the next.
        """

        count = len(self.devices)
        return slab_columns(rows, columns, count), self._workspace_capacity(slab_workspace(rows, columns, count))

    def _cut_capacity(self, rows: int, slab: int, joined: int, blocks: int = 2) -> int:
        """Elements of every slab workspace for the slabs that a step has cut.

        ``slab`` is the widest of them and ``joined`` the most columns that
        a round brings together.  The two parts are those of
        :func:`slab_workspace`, which sizes them for the widest slabs that
        the budget allows; slabs cut equal are narrower unless a range is
        a multiple of that.  The workspace only ever grows.
        """

        arriving = -(-rows // len(self.devices)) * joined
        if blocks == 1:
            arriving = max(arriving, rows * slab)
        return self._workspace_capacity(rows * slab + arriving)

    def _workspace_capacity(self, needed: int) -> int:
        """Elements of every slab workspace: the most that a step has needed so far.

        A workspace that is needed larger leaves the one of the passes
        before in the free list of the stream of its device, where no later
        request has a use for it: every device is noted as still holding
        it (:meth:`_slab_workspace`).
        """

        if needed > self._work:
            if self._work:
                self._outgrown = set(self.devices)
            self._work = needed
        return self._work

    def _slab_workspace(self, cp, device: int, work: int):
        """The memory of the slab workspace of ``device``: ``work`` elements.

        Taken on the thread of the device, under its stream, by the step
        that holds it to its end.  The density step leaves the pools of a
        shared basis as they are (:func:`shared_pool_release_requested`),
        so a workspace that a pass has outgrown would stay in the free list
        beside the larger one for the rest of the run.  The device first
        returns the free blocks of this stream to the driver, the smaller
        workspace among them, as the release after the density did before
        every pass.  The thread has no graph capture open here, which
        returning memory would invalidate (:mod:`backends.cupy_capture`):
        it ends its captures within the filter that began them.
        """

        if device in self._outgrown:
            self._outgrown.discard(device)
            if not shared_pool_release_requested():
                cp.get_default_memory_pool().free_all_blocks(stream=self._stream(device))
        return cp.empty(work, dtype=cp.float64)

    @staticmethod
    def _room(block) -> int:
        """Elements from the first one of ``block`` to the end of its allocation."""

        pointer = block.data
        memory = getattr(pointer, "mem", None)
        if memory is None:
            return 0
        return max(0, int(memory.ptr) + int(memory.size) - int(pointer.ptr)) // 8

    @staticmethod
    def _row_block_ends(rounds, widths, heights, rows: int) -> list[int]:
        """Where the row block of each device ends in its column block, in elements.

        The step with one tall block lays the rows of a device into the
        allocation of its column block (:meth:`_ritz_one_block`).  The
        rounds of :func:`block_rounds` take the columns from the end of the
        block.  The rows of a round are stored in front of those of the
        rounds before, and those of the first round up to the end that
        this returns.  After every round the rows that have arrived must
        lie behind the columns that have not left yet, which sets the end:
        at least the size of the row block, and for every round but the
        last the columns still to go and the rows so far.

        A device gives in every round at least the share of its columns
        that the rounds so far are of all rounds.  It has then received
        that share of its rows and what the slabs of the other devices
        were rounded up by, less than a column of each.  The end therefore
        lies less than one element per device and row of the device behind
        the larger of its column block and its row block, which is the
        room that :meth:`_capacities` adds to a tall block.
        """

        columns = sum(widths)
        ends = [height * columns for height in heights]
        arrived = 0
        for taken in rounds[:-1]:
            arrived += sum(last - first for first, last in taken)
            for index, (first, _last) in enumerate(taken):
                ends[index] = max(ends[index], rows * first + heights[index] * arrived)
        return ends

    @staticmethod
    def _block(cp, shape, capacity: int):
        """Fortran-ordered array of ``shape`` backed by ``capacity`` elements."""

        rows, columns = map(int, shape)
        if not rows * columns:
            return cp.empty((rows, columns), dtype=cp.float64, order="F")
        backing = cp.empty(max(int(capacity), rows * columns), dtype=cp.float64)
        return cp.ndarray((rows, columns), dtype=cp.float64, memptr=backing.data, order="F")

    @staticmethod
    def _view(cp, backing, offset: int, shape):
        """Fortran-ordered array of ``shape`` in ``backing``, ``offset`` elements from its start."""

        rows, columns = map(int, shape)
        if not rows * columns:
            return cp.empty((rows, columns), dtype=cp.float64, order="F")
        return cp.ndarray(
            (rows, columns), dtype=cp.float64, memptr=backing.data + 8 * int(offset), order="F"
        )

    @staticmethod
    def _pieces(cp, buffer, edges, own: int, width: int) -> dict:
        """Views of the exchange buffer of device ``own`` for ``width`` columns.

        The buffer holds the rows of every other device one after another,
        each as a Fortran-ordered piece of ``width`` columns.  Returns the
        pieces by device index; devices without rows have none.
        """

        pieces, offset = {}, 0
        for other, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            if other == own or high == low:
                continue
            pieces[other] = cp.ndarray(
                (high - low, width), dtype=cp.float64, memptr=buffer.data + 8 * offset, order="F"
            )
            offset += (high - low) * width
        return pieces

    @staticmethod
    def _copy(target, source, by_column: bool) -> None:
        """Copy between two column-major views of one shape on the current device.

        ``by_column`` is what :func:`column_copies_requested` told the caller.
        """

        if by_column:
            # Transposed, the last axis of both views runs down a column.
            target.T[...] = source.T
        else:
            target[...] = source

    @staticmethod
    def _whole_block_partitions(layout, blocks):
        """Filter blocks of every column range of each device, or ``None``.

        Per device, one ``(first, last)`` pair of block indices for each
        range that holds columns, in the order of its block.  ``None``
        unless every such range consists of whole blocks of this plan.
        """

        if layout.columns != blocks[-1].stop:
            return None
        first = {block.start: index for index, block in enumerate(blocks)}
        last = {block.stop: index + 1 for index, block in enumerate(blocks)}
        partitions = []
        for part in layout.parts:
            held = [(start, stop) for start, stop in part if stop > start]
            if any(start not in first or stop not in last for start, stop in held):
                return None
            partitions.append(tuple((first[start], last[stop]) for start, stop in held))
        return partitions

    # ----- creation and layout changes ----------------------------------

    def random_basis(self, generator: LapackRandom, rows: int, layout) -> DistributedBasis:
        """Fill the column blocks with PARSEC's DLARNV stream in column order.

        Column ranges that follow each other in the basis are consecutive
        segments of the column-major stream, so drawing them one after the
        other reproduces the single-device trial basis bit for bit, whichever
        block each of them lies in.
        """

        cp, _ = require_cupy()
        on_device = os.environ.get("PARSEC_CUPY_DEVICE_RANDOM", "0").lower() in {"1", "true", "on"}
        layout = _as_layout(layout)
        tall, _piece = self._capacities(layout, self.row_edges(rows))
        blocks = []
        for device, width in zip(self.devices, layout.widths, strict=True):
            with cp.cuda.Device(device):
                blocks.append(self._block(cp, (rows, width), tall))
        # In the order of the columns: a device with two ranges is visited twice.
        ranges = sorted(
            (column, index, offset, count)
            for index in range(len(self.devices))
            for column, offset, count in layout.spans(index)
        )
        for _column, index, offset, count in ranges:
            with cp.cuda.Device(self.devices[index]):
                # Whole columns of a column-major block: contiguous like the block.
                target = blocks[index][:, offset : offset + count]
                if on_device:
                    generator.device_uniform_minus_1_1(target.shape, column_major=True, out=target)
                else:
                    target.set(generator.uniform_minus_1_1((rows, count), column_major=True))
                cp.cuda.get_current_stream().synchronize()
        return DistributedBasis(self, blocks, layout)

    def repartition(self, basis: DistributedBasis, layout) -> DistributedBasis:
        """Move whole columns between devices so that ``basis.layout == layout``."""

        layout = _as_layout(layout)
        if basis.layout == layout:
            return basis
        if layout.columns != basis.shape[1]:
            raise ValueError("column ranges do not cover the basis")
        cp, _ = require_cupy()
        rows = basis.shape[0]
        old, old_blocks = basis.layout, basis.blocks
        tall, _piece = self._capacities(layout, self.row_edges(rows))

        def rebuild(index, device):
            block = self._block(cp, (rows, layout.widths[index]), tall)
            for start, offset, count in layout.spans(index):
                for other in range(len(self.devices)):
                    for begin, held, size in old.spans(other):
                        low, high = max(start, begin), min(start + count, begin + size)
                        if low >= high:
                            continue
                        source = old_blocks[other][:, held + low - begin : held + high - begin]
                        target = block[:, offset + low - start : offset + high - start]
                        if other == index:
                            target[...] = source
                        else:
                            target.data.copy_from_device_async(source.data, source.nbytes, self._stream(device))
            return block

        return DistributedBasis(self, self._timed("repartition", rebuild), layout)

    def gather(self, basis: DistributedBasis):
        """Assemble the whole basis on the owner (needs room for it there)."""

        cp, _ = require_cupy()
        layout = basis.layout
        with cp.cuda.Device(self.owner):
            stream = cp.cuda.get_current_stream()
            full = cp.empty(basis.shape, dtype=cp.float64, order="F")
            for index, (device, block) in enumerate(zip(self.devices, basis.blocks, strict=True)):
                for column, offset, count in layout.spans(index):
                    target = full[:, column : column + count]
                    source = block[:, offset : offset + count]
                    if device == self.owner:
                        target[...] = source
                    else:
                        target.data.copy_from_device_async(source.data, source.nbytes, stream)
            stream.synchronize()
        return full

    def scatter(self, full, layout) -> DistributedBasis:
        """Split an owner array into column blocks on the devices."""

        cp, _ = require_cupy()
        full = cp.asfortranarray(full)
        cp.cuda.get_current_stream().synchronize()
        rows = int(full.shape[0])
        layout = _as_layout(layout)

        def take(index, device):
            block = cp.empty((rows, layout.widths[index]), dtype=cp.float64, order="F")
            for column, offset, count in layout.spans(index):
                source = full[:, column : column + count]
                target = block[:, offset : offset + count]
                if device == self.owner:
                    target[...] = source
                else:
                    target.data.copy_from_device_async(source.data, source.nbytes, self._stream(device))
            return block

        return DistributedBasis(self, self.map(take), layout)

    def _pull(self, device: int, arriving, concurrent: bool) -> None:
        """Copy contiguous pieces from other devices into arrays of ``device``.

        ``arriving`` lists ``(other, target, source)``: the index of the
        device that holds ``source``, and two contiguous arrays of one size.
        Called on the thread of ``device``.  The copies are queued on its
        stream, in order with everything else that uses the targets there.
        If ``concurrent`` (see :func:`concurrent_exchange_requested`) each
        source has its own stream instead, and the copies have completed on
        return.
        """

        stream = self._stream(device)
        if not concurrent:
            for _other, target, source in arriving:
                target.data.copy_from_device_async(source.data, source.nbytes, stream)
            return
        # Nothing orders the other streams behind work that still uses the
        # targets on the device's own stream, or ahead of what the caller
        # queues there next: wait on both sides of the copies.
        stream.synchronize()
        inbound = self._inbound_streams(device)
        for other, target, source in arriving:
            target.data.copy_from_device_async(source.data, source.nbytes, inbound[other])
        for other, _target, _source in arriving:
            inbound[other].synchronize()

    def _to_rows(
        self,
        blocks: list,
        layout,
        edges,
        *,
        keep: bool = False,
        targets: list | None = None,
        buffers: list | None = None,
    ) -> list:
        """Column blocks -> one row block of all columns per device.

        ``layout`` says which columns each block holds; a row block holds
        all columns in their own order.  The rows of a column block that
        belong to other devices are packed a chunk of its columns at a time
        into the one exchange buffer of its device, and the others copy each
        chunk out before the next is packed.  Four devices that all have a
        chunk to send copy them out in the turns of :data:`_PAIR_TURNS`,
        each from one partner per turn (:func:`paired_exchange_requested`);
        ``to_rows_copies`` says which order the switches chose.  Unless
        ``keep`` is set, ``blocks`` is emptied once all of it has been
        shipped.  ``targets`` supplies row blocks to overwrite, and
        ``buffers`` the exchange buffer of every device where the caller
        holds them over several conversions.
        """

        cp, _ = require_cupy()
        count = len(self.devices)
        layout = _as_layout(layout)
        total, columns = edges[-1], layout.columns
        tall, capacity = self._capacities(layout, edges)
        limit = exchange_chunk_bytes() // 8
        # Read once, on the calling thread, for every chunk.
        concurrent = concurrent_exchange_requested()
        paired = concurrent and count == 4 and paired_exchange_requested()
        self.to_rows_copies = "pairs" if paired else "together" if concurrent else "queued"
        by_column = column_copies_requested()
        chunks = [
            exchange_chunks(width, total - (edges[index + 1] - edges[index]), limit)
            for index, width in enumerate(layout.widths)
        ]
        # The chunks that take the turns of the pairs: those that every
        # device sends to every other one, which is what was timed.  Once a
        # device has no chunk left, or where one has no rows to receive, a
        # device receives from all the others in one turn, as without pairs.
        whole = min(map(len, chunks)) if paired and all(low < high for low, high in zip(edges[:-1], edges[1:])) else 0

        def prepare(index, device):
            low, high = edges[index], edges[index + 1]
            target = self._block(cp, (high - low, columns), tall) if targets is None else targets[index]
            for column, offset, size in layout.spans(index):
                # The rows that stay on this device need no buffer.
                self._copy(
                    target[:, column : column + size], blocks[index][low:high, offset : offset + size], by_column
                )
            if buffers is not None:
                return target, buffers[index]
            return target, cp.empty(capacity if chunks[index] else 0, dtype=cp.float64)

        row_blocks, held = zip(*self.map(prepare))

        def pack(index, step):
            if step >= len(chunks[index]):
                return {}
            first, last = chunks[index][step]
            packed = self._pieces(cp, held[index], edges, index, last - first)
            for other, piece in packed.items():
                self._copy(piece, blocks[index][edges[other] : edges[other + 1], first:last], by_column)
            return packed

        def receive(index, device, step, packed, partners):
            arriving = []
            for other in range(count) if partners is None else (partners[index],):
                piece = packed[other].get(index)
                if piece is not None:
                    first, last = chunks[other][step]
                    # A chunk that reaches into two column ranges of its
                    # device arrives as two runs of whole columns.
                    for column, offset, size in layout.spans(other, first, last):
                        arriving.append(
                            (
                                other,
                                row_blocks[index][:, column : column + size],
                                piece[:, offset - first : offset - first + size],
                            )
                        )
            # The one source of a turn needs no stream beside the device's own.
            self._pull(device, arriving, concurrent and partners is None)

        for step in range(max(map(len, chunks))):
            # A buffer is packed completely before any device reads it, and
            # read by all of them before it is packed again.
            packed = self.map(lambda index, device: pack(index, step))
            for partners in _PAIR_TURNS if step < whole else (None,):
                # Every device has received from its partner of a turn
                # before any device turns to the next one.
                self.map(lambda index, device: receive(index, device, step, packed, partners))
        # Every copy has completed.  The pieces are dropped here and not with
        # the function that received them, which the thread of a device may
        # still hold when the next conversion asks the pool for its buffer.
        # The buffers allocated here are released on return.
        packed = None
        if not keep:
            blocks.clear()
        return list(row_blocks)

    def _to_columns(
        self,
        rows: list,
        layout,
        edges,
        *,
        targets: list | None = None,
        buffers: list | None = None,
    ) -> list:
        """Row blocks of all columns -> one column block per device.

        Each device fetches the rows of the others a chunk of its columns at
        a time into its one exchange buffer and unpacks them from there.
        ``targets`` supplies column blocks of ``layout`` to overwrite, and
        ``buffers`` the exchange buffer of every device where the caller
        holds them over several conversions.
        """

        cp, _ = require_cupy()
        total = edges[-1]
        layout = _as_layout(layout)
        tall, capacity = self._capacities(layout, edges)
        limit = exchange_chunk_bytes() // 8
        concurrent = concurrent_exchange_requested()
        by_column = column_copies_requested()

        def unpack(index, device):
            width = layout.widths[index]
            low, high = edges[index], edges[index + 1]
            block = self._block(cp, (total, width), tall) if targets is None else targets[index]
            chunks = exchange_chunks(width, total - (high - low), limit)
            if buffers is None:
                buffer = cp.empty(capacity if chunks else 0, dtype=cp.float64)
            else:
                buffer = buffers[index]
            for column, offset, size in layout.spans(index):
                # The rows this device already holds need no buffer.
                self._copy(block[low:high, offset : offset + size], rows[index][:, column : column + size], by_column)
            for first, last in chunks:
                pieces = self._pieces(cp, buffer, edges, index, last - first)
                # A chunk that reaches into two column ranges of the device
                # is fetched as two runs of whole columns.
                spans = layout.spans(index, first, last)
                self._pull(
                    device,
                    [
                        (
                            other,
                            piece[:, offset - first : offset - first + size],
                            rows[other][:, column : column + size],
                        )
                        for other, piece in pieces.items()
                        for column, offset, size in spans
                    ],
                    concurrent,
                )
                # The pieces are unpacked after these copies and before those
                # of the next chunk, which overwrite the buffer.
                for other, piece in pieces.items():
                    self._copy(block[edges[other] : edges[other + 1], first:last], piece, by_column)
            return block, buffer

        # The buffers are released only after every stream has been synchronized.
        unpacked = self.map(unpack)
        return [block for block, _buffer in unpacked]

    # ----- the two numerical steps ---------------------------------------

    def filter(self, basis: DistributedBasis, blocks, lower, upper, reference, reset):
        """Chebyshev-filter every column block in place on its own device.

        Consumes ``basis``: its blocks receive the filtered columns.  Each
        device filters the columns it holds unless the ranges are to be
        balanced or do not consist of whole filter blocks.  A device with
        two column ranges filters both in one call, the blocks side by side
        as its own block holds them, so that its filter graphs are captured
        once for a plan like those of a device with one range.
        """

        cp, _ = require_cupy()
        partitions = self._whole_block_partitions(basis.layout, blocks) if keeps_column_ranges() else None
        if partitions is None:
            basis = self.repartition(basis, self.column_layout(blocks))
            partitions = self._whole_block_partitions(basis.layout, blocks)
        sigmas = starting_sigmas(blocks, lower, upper, reset, reference)
        owner = self._operator(self.owner)
        cp.cuda.get_current_stream().synchronize()
        potential = cp.asnumpy(owner.effective_potential)
        layout = self.layout = basis.layout
        columns = basis.take()

        def run(index, device):
            block = columns[index]
            if not partitions[index]:
                return block
            if device != self.owner:
                self._worker.replicas[device].effective_potential.set(potential, stream=self._stream(device))
            # The blocks of this device where its own block holds them, and
            # the sigma that the plan carries into each: a range does not
            # continue the recurrence of the range stored before it.
            local, starts, offset = [], [], 0
            for first, last in partitions[index]:
                shift = offset - blocks[first].start
                local.extend(FilterBlock(b.start + shift, b.stop + shift, b.degree) for b in blocks[first:last])
                starts.extend(sigmas[first:last])
                offset = local[-1].stop
            return self._worker.graphs[device].apply(
                block, tuple(local), lower, upper, reference, reset, initial_sigma=starts, out=block
            )

        return DistributedBasis(self, self._timed("filter", run), layout)

    def _summed(self, parts: list, on_device: bool):
        """Sum the Gram pairs of the devices: on the owner, or on the host.

        ``parts[0]`` receives the result.
        """

        if on_device:
            return self._sum_on_owner(parts)
        # Summed in device order into the owner's part, which nothing else holds.
        packed = parts[0]
        for part in parts[1:]:
            packed += part
        return packed

    def _condition_helper(self) -> int | None:
        """The device that estimates the condition number of the overlap beside the device small solve.

        The first device after the owner, which waits during that solve,
        that the sector operator does not name as ``hartree_device`` (the
        driver does, through the sector eigensolver): with the Poisson and
        boundary arrays that one is the fullest device of the process, and
        the estimate takes a copy of the overlap and the work arrays of its
        eigensolver there.  A sector whose only other device is that one
        asks it where :func:`condition_helper_policy` says ``any``.
        ``None`` leaves the estimate to the owner, in front of its
        factorization: for such a sector otherwise, for a sector on one
        device, where the policy is ``off``, and where the condition number
        is the host SVD, which needs no device.
        """

        policy = condition_helper_policy()
        if policy == "off" or _condition_policy(on_device=True) != "symmetric":
            return None
        others = self.devices[1:]
        busy = getattr(self._operator(self.owner), "hartree_device", None)
        free = [device for device in others if device != busy]
        if free:
            return free[0]
        return others[0] if others and policy == "any" else None

    def _condition_beside(self, helper: int):
        """The ``condition`` of :func:`rayleigh_ritz.solve_whitened_ritz_on_device` for an estimate on ``helper``.

        Given the symmetric overlap of the owner, complete on its device,
        it starts the estimate on the thread and stream of ``helper`` and
        returns what waits for the number.  The helper copies the overlap
        device to device and estimates as the owner would
        (:func:`rayleigh_ritz._device_overlap_condition`), with the same
        host SVD of its copy near the limit, so every decision is the one
        the owner would have made.  A symmetric matrix reads the same in
        both orders, which is all the raw copy asks of its layout.  An
        error of the helper that is no failed estimate, out of memory
        among them, ends the step as the same error of the owner would:
        nothing falls back to the owner (see :func:`condition_helper_policy`
        for what the helper takes).

        The thread is the one that filters on that device and lives as
        long as the process: it comes to hold a cuSOLVER handle, and the
        end of such a thread invalidates an open graph capture
        (backends/cupy_capture.py).
        """

        cp, _ = require_cupy()
        stream = self._stream(helper)

        def estimate(overlap):
            started = perf_counter()
            try:
                with cp.cuda.Device(helper), stream:
                    copy = cp.empty(overlap.shape, dtype=cp.float64)
                    copy.data.copy_from_device_async(overlap.data, overlap.nbytes, stream)
                    return _device_overlap_condition(copy)
            finally:
                self.condition_seconds += perf_counter() - started

        def start(overlap):
            if not (overlap.flags.c_contiguous or overlap.flags.f_contiguous):
                raise ValueError("only a contiguous overlap is copied from the owner")
            return _pool(helper).submit(estimate, overlap).result

        return start

    def _small_solve(self, packed, on_device: bool):
        """Solve the small generalized problem of a summed Gram pair.

        On the owner if the pair is an array of it, else on the host.
        Returns host Ritz values, and the coefficients and the whitened
        matrix where the solve ran.  The time counts as ``dense``, also that
        of a solve that fails a stability audit.  On the owner, another
        device of the sector estimates the condition number of the overlap
        meanwhile where :meth:`_condition_helper` names one.
        """

        cp, _ = require_cupy()
        started = perf_counter()
        try:
            if on_device:
                helper = self._condition_helper()
                self.condition_device = self.owner if helper is None else helper
                beside = {} if helper is None else {"condition": self._condition_beside(helper)}
                with cp.cuda.Device(self.owner):
                    eigenvalues, coefficients, whitened = solve_whitened_ritz_on_device(packed[0], packed[1], **beside)
                    eigenvalues = cp.asnumpy(eigenvalues)
                    # Whole columns in one piece, complete before any other device copies them.
                    coefficients = cp.asfortranarray(coefficients)
                    cp.cuda.get_current_stream().synchronize()
                return eigenvalues, coefficients, whitened
            return solve_whitened_ritz(packed[0], packed[1])
        finally:
            self.seconds["dense"] += perf_counter() - started

    def ritz(self, operator: Any, filtered: DistributedBasis, *, stages: bool):
        """Generalized Rayleigh--Ritz of a filtered basis; consumes ``filtered``.

        Returns host Ritz values, the rotated basis and the whitened projected
        Hamiltonian.  On :class:`GeneralizedRitzStabilityError` the exception
        carries the intact filtered basis as ``error.filtered``.

        The rotated basis is written back into the column blocks of
        ``filtered``, which stay allocated as the one persistent copy.
        :meth:`_ritz_one_block` takes the step, or :meth:`_ritz_two_blocks`,
        unless :func:`tall_blocks_per_device` says three.  Each device then holds
        at most three tall blocks (the columns, ``H`` times them or its row
        block, and the basis row block) plus one exchange chunk (see
        :func:`exchange_chunk_bytes`).  No tall block is released and
        requested again within a pass; doing so had left a fourth one cached
        in the memory pool.

        With the small solve on the device (see
        :func:`rayleigh_ritz.dense_solve_on_device`) the Gram matrices never
        reach the host: the owner collects and sums the parts of the other
        devices, solves, and each device copies the coefficients from it.
        The whitened matrix is then an owner array.  The work arrays of that
        solve, a few ``m x m`` ones on the owner, come on top of the blocks
        counted above.
        """

        blocks = tall_blocks_per_device()
        self.tall_blocks = blocks
        self.slab_cut = self.slab_columns = self.projection_columns = None
        if blocks == 1:
            return self._ritz_one_block(operator, filtered, stages=stages)
        if blocks == 2:
            return self._ritz_two_blocks(operator, filtered, stages=stages)
        cp, _ = require_cupy()
        on_device = dense_solve_on_device()
        rows, _columns = filtered.shape
        layout = filtered.layout
        edges = self.row_edges(rows)
        tall, _piece = self._capacities(layout, edges)
        columns = filtered.take()

        def stage(name):
            return device_stage(operator, name) if stages else _NoStage()

        with stage("subspace_ritz_hamiltonian_seconds"):
            def apply(index, device):
                block = columns[index]
                target = self._block(cp, block.shape, tall)
                if block.shape[1]:
                    self._operator(device).apply_into(block, target)
                return target

            applied = self._timed("apply", apply)

        with stage("subspace_ritz_projection_seconds"):
            started = perf_counter()
            # H times the columns is moved and released before the columns get their row copy.
            row_applied = self._to_rows(applied, layout, edges)
            row_basis = self._to_rows(columns, layout, edges, keep=True)
            self.seconds["to_rows"] += perf_counter() - started

            def gram(index, device):
                overlap = _symmetric_overlap(row_basis[index])
                # A row block's part of X.T (H X) is not symmetric, but the
                # parts are summed entry by entry and only the lower triangle
                # of the sum is read: the lower triangle of each part suffices.
                projection = _lower_triangle_product(row_basis[index], row_applied[index])
                if on_device:
                    # Row-major whatever the order of the two products: the owner copies raw memory.
                    return cp.ascontiguousarray(cp.stack((overlap, projection)))
                return np.asarray(cp.asnumpy(cp.stack((overlap, projection))), dtype=np.float64)

            parts = self._timed("gram", gram)
            packed = self._summed(parts, on_device)
            del parts

        try:
            eigenvalues, coefficients, whitened = self._small_solve(packed, on_device)
        except GeneralizedRitzStabilityError as error:
            # The row copies go before the fallback gathers the basis on the owner.
            del row_applied, row_basis
            error.filtered = DistributedBasis(self, columns, layout)
            raise
        # The Gram pair is released before the rotation.  On the device it is
        # the owner's part, allocated under the owner's stream of this group
        # like the exchange buffer: the pool carves it from the block of that
        # buffer, which the return to columns then requests whole.  The
        # coefficients and the whitened matrix outlive that return without
        # holding the block: they are allocated on the calling thread, under
        # its stream, and CuPy's pool keeps the free blocks of each stream
        # apart.
        del packed

        with stage("subspace_ritz_rotation_seconds"):
            def rotate(index, device):
                if on_device:
                    device_coefficients = self._from_owner(cp, coefficients, device)
                else:
                    device_coefficients = cp.asarray(coefficients, dtype=cp.float64)
                # (X C).T = C.T X.T written into the C-contiguous view of the
                # H X row block, which is no longer needed; no temporary.
                cp.matmul(device_coefficients.T, row_basis[index].T, out=row_applied[index].T)
                return row_applied[index]

            rotated = self._timed("rotate", rotate)
            del row_basis, row_applied
            started = perf_counter()
            blocks = self._to_columns(rotated, layout, edges, targets=columns)
            del rotated
            self.seconds["to_columns"] += perf_counter() - started
        self.passes += 1
        return eigenvalues, DistributedBasis(self, blocks, layout), whitened

    def _ritz_two_blocks(self, operator: Any, filtered: DistributedBasis, *, stages: bool):
        """:meth:`ritz` without a tall block for ``H`` times the columns.

        The filtered columns go to their row blocks first.  ``H X`` then
        exists a slab of columns at a time (:func:`slab_columns`,
        :func:`projection_slabs`): every device applies ``H`` to a slab of
        its own columns, the slabs travel to the row layout through the
        exchange buffers, and each device adds to its part of ``X.T (H X)``
        what its row block of the basis projects on the rows it received.
        After the small solve every row block is rotated in place a tile of
        rows at a time and returns to the column blocks of ``filtered``,
        which nothing has written to until then.

        A device holds two tall blocks (the columns and the row block), the
        slab workspace and one exchange chunk.  The workspace is that of
        the slabs that were cut (:meth:`_cut_capacity`): equal ones in every
        column range, up to 6 GiB wide where :func:`wider_slabs_fit` allows
        it, unless :func:`equal_slabs_requested` selects the former cut,
        whose workspace is that of the budget (:func:`slab_workspace`).  The
        workspace and the exchange buffer are taken before anything else and
        kept to the end: no smaller array of the step can then be carved
        from the pool block of either, which the next request for it would
        find split.

        All of these are in use while a slab travels: its columns are read
        in all rows, packed into the buffer and written to the row half of
        the workspace on every device.  Besides them a device takes its
        Gram pair, two ``m x m`` arrays, and the overlap product the pair is
        filled from, whose pool block then serves the copy of the
        coefficients.  The owner also takes the pair it receives and the
        work arrays of the device small solve.  For 23,768 electrons on 16
        GPUs, with tall blocks of 21.0 GiB and the former cut into slabs of
        4 GiB, a device sampled 53.0 GiB: the two blocks, 8.0 GiB of
        workspace, the chunk of 1 GiB and the 1.7 GiB of operator, filter
        buffers and CUDA context that it holds between the steps, which
        leaves 0.4 GiB for the Gram arrays and whatever else, as much as
        the step with three tall blocks leaves.  The owner sampled 1.1 GiB
        more.

        The results differ from those of three tall blocks by round-off: the
        tall products of the projection and of the rotation are cut into
        other pieces.
        """

        cp, _ = require_cupy()
        on_device = dense_solve_on_device()
        rows, states = filtered.shape
        layout = filtered.layout
        widths = layout.widths
        edges = self.row_edges(rows)
        _tall, chunk = self._capacities(layout, edges)
        equal = equal_slabs_requested()
        if equal:
            width = slab_columns(rows, states, len(self.devices), wider_slabs_fit(rows, states, len(self.devices)))
        else:
            width, work = self._slab_capacity(rows, states)
        rounds = projection_slabs(layout, width, equal, gram_multiple())
        self.slab_cut = "equal" if equal else "full"
        self.slab_columns = max(last - first for taken in rounds for first, last in taken)
        self.projection_columns = self.slab_columns
        # Where the slabs of a round lie side by side in the row layout.
        places = []
        for taken in rounds:
            offsets = [0]
            for first, last in taken:
                offsets.append(offsets[-1] + last - first)
            places.append(tuple(zip(offsets[:-1], offsets[1:])))
        # The widest round.
        widest = max(place[-1][1] for place in places)
        if equal:
            # The workspace of the slabs that were cut, which are narrower
            # than the budget allows unless a range is a multiple of that.
            # The rotation tile lies in its first part and is as tall as
            # the widest of them lets it be.
            width = self.slab_columns
            work = self._cut_capacity(rows, width, widest)
        columns = filtered.take()

        def stage(name):
            return device_stage(operator, name) if stages else _NoStage()

        def workspace(index, device):
            height = edges[index + 1] - edges[index]
            backing = self._slab_workspace(cp, device, work)
            return (
                # H times a slab of the columns of this device, in all rows ...
                self._view(cp, backing, 0, (rows, min(width, widths[index]))),
                # ... and behind it the rows of this device of the slabs of all devices.
                self._view(cp, backing, rows * width, (height, widest)),
                # The rotation tile lies where the slab was, which is done with by then.
                self._view(cp, backing, 0, (min(height, rows * width // states), states)),
                # As the exchanges allocate it: no buffer where no column leaves the device.
                cp.empty(chunk if widths[index] and rows > height else 0, dtype=cp.float64),
            )

        with stage("subspace_ritz_projection_seconds"):
            started = perf_counter()
            slabs, row_slabs, tiles, buffers = zip(*self.map(workspace))
            row_basis = self._to_rows(columns, layout, edges, keep=True, buffers=buffers)
            self.seconds["to_rows"] += perf_counter() - started

            def overlap(index, device):
                # The overlap of the row block and, zero so far, its part of
                # the projection.  Row-major: the owner copies raw memory.
                pair = cp.zeros((2, states, states), dtype=cp.float64)
                if row_basis[index].shape[0]:
                    pair[0] = _symmetric_overlap(row_basis[index])
                return pair

            before = self.seconds["gram"]
            pairs = self._timed("gram", overlap)
            self.overlap_seconds += self.seconds["gram"] - before

        def apply(index, device, step):
            first, last = rounds[step][index]
            if last > first:
                start = layout.offset(index, first)
                self._operator(device).apply_into(
                    columns[index][:, start : start + last - first], slabs[index][:, : last - first]
                )

        def project(index, device, step):
            block = row_basis[index]
            if not block.shape[0]:
                return
            for (first, last), (begin, end) in zip(rounds[step], places[step]):
                if last > first:
                    # A row block's part of X.T (H X) is not symmetric, but the
                    # parts are summed entry by entry and only the lower
                    # triangle of the sum is read: the basis columns from the
                    # first one of the slab onwards suffice.
                    pairs[index][1][first:, first:last] = block[:, first:].T @ row_slabs[index][:, begin:end]

        for step, taken in enumerate(rounds):
            with stage("subspace_ritz_hamiltonian_seconds"):
                self._timed("apply", lambda index, device: apply(index, device, step))
            with stage("subspace_ritz_projection_seconds"):
                started = perf_counter()
                self._to_rows(
                    [slab[:, : last - first] for slab, (first, last) in zip(slabs, taken)],
                    places[step],
                    edges,
                    keep=True,
                    targets=[row_slab[:, : places[step][-1][1]] for row_slab in row_slabs],
                    buffers=buffers,
                )
                self.seconds["to_rows"] += perf_counter() - started
                self._timed("gram", lambda index, device: project(index, device, step))

        with stage("subspace_ritz_projection_seconds"):
            if on_device:
                parts = pairs
            else:
                parts = self._timed(
                    "gram", lambda index, device: np.asarray(cp.asnumpy(pairs[index]), dtype=np.float64)
                )
            packed = self._summed(parts, on_device)
            del parts, pairs

        try:
            eigenvalues, coefficients, whitened = self._small_solve(packed, on_device)
        except GeneralizedRitzStabilityError as error:
            # The row blocks and the workspace go before the fallback gathers
            # the basis on the owner.  The columns are as they were filtered.
            del row_basis, slabs, row_slabs, tiles, buffers
            error.filtered = DistributedBasis(self, columns, layout)
            raise
        del packed

        with stage("subspace_ritz_rotation_seconds"):
            def rotate(index, device):
                block = row_basis[index]
                if not block.shape[0]:
                    return
                if on_device:
                    device_coefficients = self._from_owner(cp, coefficients, device)
                else:
                    device_coefficients = cp.asarray(coefficients, dtype=cp.float64)
                _rotate_in_place(block, device_coefficients, tiles[index])

            self._timed("rotate", rotate)
            started = perf_counter()
            blocks = self._to_columns(row_basis, layout, edges, targets=columns, buffers=buffers)
            del row_basis
            self.seconds["to_columns"] += perf_counter() - started
        self.passes += 1
        return eigenvalues, DistributedBasis(self, blocks, layout), whitened

    def _in_basis_order(self, packed, order: np.ndarray, on_device: bool):
        """The summed Gram pair of re-laid row blocks, in the order of the basis.

        ``packed`` holds the lower triangles of the two Gram matrices of
        row blocks whose column ``p`` is column ``order[p]`` of the basis.
        The small solve reads lower triangles in the order of the basis, so
        each matrix is mirrored into its symmetric form and its rows and
        columns are put back.  The solve is then given what the row blocks
        of the other steps give it, summed from other pieces.  Returns a
        new pair where ``packed`` is: on the owner or on the host.
        """

        positions = np.empty_like(order)
        positions[order] = np.arange(order.size)
        if not on_device:
            ordered = np.empty_like(packed)
            for part in range(2):
                ordered[part] = _mirrored_lower(packed[part])[np.ix_(positions, positions)]
            return ordered
        cp, _ = require_cupy()
        with cp.cuda.Device(self.owner):
            where = cp.asarray(positions)
            ordered = cp.empty_like(packed)
            for part in range(2):
                lower = cp.tril(packed[part])
                ordered[part] = (lower + cp.tril(lower, -1).T)[where[:, None], where[None, :]]
        return ordered

    def _in_row_order(self, coefficients, order: np.ndarray, on_device: bool):
        """Coefficients that rotate row blocks whose column ``p`` is column ``order[p]`` of the basis.

        Their rows follow the columns that such a block holds, and their
        columns the Ritz vectors that it is to hold in the same places.
        Fortran-ordered where ``coefficients`` are: on the owner, complete
        before another device copies them, or on the host.
        """

        if not on_device:
            return np.asfortranarray(coefficients[np.ix_(order, order)])
        cp, _ = require_cupy()
        with cp.cuda.Device(self.owner):
            where = cp.asarray(order)
            permuted = cp.asfortranarray(coefficients[where[:, None], where[None, :]])
            cp.cuda.get_current_stream().synchronize()
        return permuted

    def _ritz_one_block(self, operator: Any, filtered: DistributedBasis, *, stages: bool):
        """:meth:`ritz` with the row block of a device in the memory of its columns.

        A column block and a row block of an even share of the basis have
        the same size, and the columns are needed only until ``H`` has been
        applied to them.  They are therefore re-laid as rows in place, in
        the rounds of :func:`block_rounds`.  In every round each device

        * applies ``H`` to a slab of its columns, the last ones that its
          block still holds, into the first half of the slab workspace;
        * moves that slab into the second half, which frees its place in
          the block;
        * sends the slab from there to the row layout of all devices
          through the exchange buffers, as :meth:`_to_rows` does it; the
          rows that arrive are stored in front of those of the rounds
          before, at the end of the block (:meth:`_row_block_ends`);
        * sends ``H`` times the slab the same way into the second half,
          which the slab has left by then, and projects the rows of it
          that arrive on the rows of the basis that the block holds by
          then.

        A round thus brings the columns of all devices together and its
        projection is one product as wide as they are.  The rows of this
        round and of those before it are the trailing columns of the row
        block, so the product fills a slab of the lower triangle, in the
        order in which the row block holds the columns.  The overlap is
        formed from the complete row block as in the other steps.
        :meth:`_in_basis_order` puts the summed pair into the order of the
        basis for the small solve, and :meth:`_in_row_order` the
        coefficients into that of the row blocks, which are rotated in
        place a tile of rows at a time.  The way back runs the rounds in
        reverse: each device moves the rows of a round into the second
        half of the workspace and fetches from there, on all devices, its
        columns of that round into their place in the block.

        A device holds its one tall block, the workspace of the slabs that
        were cut (:meth:`_cut_capacity`) and one exchange chunk, and beside
        them the Gram arrays of :meth:`_ritz_two_blocks`.  The whole
        allocation of a column block is the step's to use: where the block
        is the leading part of a wider one, what lies behind it is
        overwritten.
        A block whose allocation has no room for the row block, one that
        :meth:`scatter` made or an empty one, gets a row block of its own
        for the step, counted in ``separate_row_blocks``.

        The basis crosses the links as often as with two tall blocks:
        its columns to the rows, ``H`` times them, and the rows back.
        Inside a device it is copied twice more, into the workspace on
        each way.

        After a failed stability audit the rounds are run in reverse
        without the rotation, which puts every filtered column back where
        it was, bit for bit, in the memory the step has already; only then
        is the error raised.  The results differ from those of two tall
        blocks by round-off: the projection is cut into rounds instead of
        slabs of one device, and overlap and rotation meet the columns in
        another order.
        """

        cp, _ = require_cupy()
        on_device = dense_solve_on_device()
        by_column = column_copies_requested()
        rows, states = filtered.shape
        layout = filtered.layout
        widths = layout.widths
        edges = self.row_edges(rows)
        heights = [high - low for low, high in zip(edges[:-1], edges[1:])]
        tall, chunk = self._capacities(layout, edges)
        budget = slab_columns(rows, states, len(self.devices), blocks=1)
        rounds = block_rounds(widths, budget, gram_multiple())
        ends = self._row_block_ends(rounds, widths, heights, rows)
        # Where the slabs of a round lie side by side in the row layout,
        # which columns of a row block they are, and the column of the
        # basis in every column of a row block.
        places, spots, order = [], [], np.empty(states, dtype=np.int64)
        for taken in rounds:
            offsets = [0]
            for first, last in taken:
                offsets.append(offsets[-1] + last - first)
            places.append(tuple(zip(offsets[:-1], offsets[1:])))
            end = spots[-1][0] if spots else states
            spots.append((end - offsets[-1], end))
            for index, (first, last) in enumerate(taken):
                for column, offset, size in layout.spans(index, first, last):
                    start = spots[-1][0] + offsets[index] + offset - first
                    order[start : start + size] = np.arange(column, column + size)
        # The workspace of the slabs that were cut: equal ones, narrower
        # than the budget allows unless the widest block is a multiple of
        # that, or those of whole rounds, which are narrower still.  The
        # rotation tile lies in its first part and is as tall as the widest
        # of them lets it be.
        width = max(last - first for taken in rounds for first, last in taken)
        self.slab_cut = "equal" if rounds == block_rounds(widths, budget) else "multiple"
        self.slab_columns, self.projection_columns = width, max(place[-1][1] for place in places)
        work = self._cut_capacity(rows, width, self.projection_columns, 1)
        columns = filtered.take()

        def stage(name):
            return device_stage(operator, name) if stages else _NoStage()

        def workspace(index, device):
            height, block = heights[index], columns[index]
            backing = self._slab_workspace(cp, device, work)
            inside = self._room(block) >= ends[index]
            if inside:
                row_block = self._view(cp, block, ends[index] - height * states, (height, states))
            else:
                row_block = self._block(cp, (height, states), tall)
            slabs, applied, leaving, arriving, resident = [], [], [], [], []
            for taken, place, (begin, end) in zip(rounds, places, spots):
                first, last = taken[index]
                # The slab of a round where the column block holds it, ...
                slabs.append(block[:, first:last])
                # ... H times it in the first half of the workspace, ...
                applied.append(self._view(cp, backing, 0, (rows, last - first)))
                # ... the slab on its way in the second half and, in the
                # same memory, the rows of this device of all slabs of the
                # round ...
                leaving.append(self._view(cp, backing, rows * width, (rows, last - first)))
                arriving.append(self._view(cp, backing, rows * width, (height, place[-1][1])))
                # ... and those rows where the row block holds them.
                resident.append(row_block[:, begin:end])
            return (
                row_block,
                # The rotation tile lies where H times a slab was, which is done with by then.
                self._view(cp, backing, 0, (min(height, rows * width // states), states)),
                # As the exchanges allocate it: no buffer where no column leaves the device.
                cp.empty(chunk if widths[index] and rows > height else 0, dtype=cp.float64),
                slabs, applied, leaving, arriving, resident, inside,
            )

        with stage("subspace_ritz_projection_seconds"):
            started = perf_counter()
            row_basis, tiles, buffers, slabs, applied, leaving, arriving, resident, inside = zip(*self.map(workspace))
            self.seconds["to_rows"] += perf_counter() - started
            # Zero so far: the part of every device of the projection.  Row-major: the owner copies raw memory.
            pairs = self._timed("gram", lambda index, device: cp.zeros((2, states, states), dtype=cp.float64))
        self.separate_row_blocks += len(inside) - sum(inside)

        def apply(index, device, step):
            if slabs[index][step].shape[1]:
                self._operator(device).apply_into(slabs[index][step], applied[index][step])

        def leave(index, device, step):
            if slabs[index][step].shape[1]:
                self._copy(leaving[index][step], slabs[index][step], by_column)

        def project(index, device, step):
            block = row_basis[index]
            if not block.shape[0]:
                return
            begin, end = spots[step]
            # A row block's part of X.T (H X) is not symmetric, but the
            # parts are summed entry by entry and only the lower triangle
            # of the sum is read: the columns of this round and of those
            # before it, which lie from its first one onwards, suffice.
            pairs[index][1][begin:, begin:end] = block[:, begin:].T @ arriving[index][step]

        def overlap(index, device):
            if row_basis[index].shape[0]:
                pairs[index][0] = _symmetric_overlap(row_basis[index])

        def lift(index, device, step):
            if resident[index][step].size:
                self._copy(arriving[index][step], resident[index][step], by_column)

        def back():
            # The rows of a round leave the block before its columns are
            # written, which may lie in their place.
            for step in reversed(range(len(rounds))):
                self.map(lambda index, device: lift(index, device, step))
                self._to_columns(
                    [views[step] for views in arriving],
                    places[step],
                    edges,
                    targets=[views[step] for views in slabs],
                    buffers=buffers,
                )

        for step in range(len(rounds)):
            with stage("subspace_ritz_hamiltonian_seconds"):
                self._timed("apply", lambda index, device: apply(index, device, step))
            with stage("subspace_ritz_projection_seconds"):
                started = perf_counter()
                self.map(lambda index, device: leave(index, device, step))
                self._to_rows(
                    [views[step] for views in leaving],
                    places[step],
                    edges,
                    keep=True,
                    targets=[views[step] for views in resident],
                    buffers=buffers,
                )
                self._to_rows(
                    [views[step] for views in applied],
                    places[step],
                    edges,
                    keep=True,
                    targets=[views[step] for views in arriving],
                    buffers=buffers,
                )
                self.seconds["to_rows"] += perf_counter() - started
                self._timed("gram", lambda index, device: project(index, device, step))

        with stage("subspace_ritz_projection_seconds"):
            before = self.seconds["gram"]
            self._timed("gram", overlap)
            self.overlap_seconds += self.seconds["gram"] - before
            if on_device:
                parts = pairs
            else:
                parts = self._timed(
                    "gram", lambda index, device: np.asarray(cp.asnumpy(pairs[index]), dtype=np.float64)
                )
            packed = self._summed(parts, on_device)
            del parts, pairs
            started = perf_counter()
            packed = self._in_basis_order(packed, order, on_device)
            self.seconds["gram"] += perf_counter() - started

        try:
            eigenvalues, coefficients, whitened = self._small_solve(packed, on_device)
        except GeneralizedRitzStabilityError as error:
            del packed
            started = perf_counter()
            back()
            self.seconds["to_columns"] += perf_counter() - started
            # The workspace goes before the fallback gathers the basis on
            # the owner.  The columns are as they were filtered.
            del row_basis, tiles, buffers, slabs, applied, leaving, arriving, resident
            error.filtered = DistributedBasis(self, columns, layout)
            raise
        del packed

        with stage("subspace_ritz_rotation_seconds"):
            started = perf_counter()
            rotation = self._in_row_order(coefficients, order, on_device)

            def rotate(index, device):
                block = row_basis[index]
                if not block.shape[0]:
                    return
                if on_device:
                    device_coefficients = self._from_owner(cp, rotation, device)
                else:
                    device_coefficients = cp.asarray(rotation, dtype=cp.float64)
                _rotate_in_place(block, device_coefficients, tiles[index])

            self.map(rotate)
            self.seconds["rotate"] += perf_counter() - started
            started = perf_counter()
            back()
            self.seconds["to_columns"] += perf_counter() - started
        self.passes += 1
        return eigenvalues, DistributedBasis(self, columns, layout), whitened

    def orthonormal_ritz(self, operator: Any, filtered: DistributedBasis, generator):
        """Robust fallback: gather on the owner, QR, ordinary Rayleigh--Ritz."""

        cp, _ = require_cupy()
        layout = filtered.layout
        try:
            full = self.gather(filtered)
        except cp.cuda.memory.OutOfMemoryError as error:
            raise RuntimeError(
                "the filtered basis failed the generalized-Ritz stability audit and is "
                "too large to orthonormalize on one device"
            ) from error
        filtered.take()
        basis = orthonormalize_complete_subspace(full, rng=generator).basis
        del full
        result = rayleigh_ritz(operator, basis, compute_residuals=False)
        del basis
        return (
            cp.asnumpy(result.eigenvalues),
            self.scatter(result.wavefunctions, layout),
            cp.asnumpy(result.projected_hamiltonian),
        )


class _NoStage:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False


def _devices_can_filter(operator: Any) -> bool:
    """Whether every device of a group can filter columns of ``operator``.

    The other devices get replicas with the stencil of the owner in its
    layout, slot-major or the affine tiles it packed
    (:class:`DistributedFilter`), and all of them filter with CUDA graphs,
    which ask of an operator what :func:`filter_graph.graph_filter` does:
    the stencil-major kernels, and the fused projector kernels where there
    are projectors.
    """

    stencil = getattr(operator, "compact_finite_difference", None)
    if not isinstance(stencil, CuPyStencilMajorFiniteDifference):
        return False
    return not operator.projector_count or bool(
        operator.fused_projector_scatter and operator.custom_projector_projection is not None
    )


def shared_basis_devices(operator: Any, columns: int | None = None) -> tuple[int, ...] | None:
    """Return the devices that are to hold the basis of ``operator`` jointly.

    ``None`` leaves the basis on one device: the policy is off, or the
    operator was given no group of at least two devices.  ``auto`` also
    leaves it there if the devices cannot filter it (see
    :func:`_devices_can_filter`), if a basis of its ``columns`` states is to
    be orthonormalized (:func:`rayleigh_ritz.generalized_ritz_requested`
    declines; a shared basis is always solved in its filtered form), or if
    its blocks do not fit the current (owner) device.  The owner-centred
    route is used then, which with ``PARSEC_CUPY_STREAMING_RITZ=auto`` keeps
    a single array there.
    """

    policy = distributed_state_policy()
    if policy == "off":
        return None
    devices = getattr(operator, "distributed_filter_devices", None)
    if devices is None or len(devices) < 2:
        return None
    devices = tuple(int(device) for device in devices)
    if policy == "auto" and (
        not _devices_can_filter(operator)
        or (
            columns is not None
            and not (
                generalized_ritz_requested(int(operator.shape[0]), columns)
                and shared_basis_fits(int(operator.shape[0]), columns, len(devices))
            )
        )
    ):
        return None
    return devices


def sector_device_group(operator: Any, columns: int | None = None) -> SectorDeviceGroup | None:
    """Return the operator's device group when its basis is to be shared.

    :func:`shared_basis_devices` decides that.
    """

    devices = shared_basis_devices(operator, columns)
    if devices is None:
        return None
    group = getattr(operator, "_sector_device_group", None)
    if group is None or group.devices != devices:
        group = SectorDeviceGroup(operator, devices)
        operator._sector_device_group = group
    return group


def _ritz_result(cp, eigenvalues, basis, whitened) -> DeviceRayleighRitzResult:
    return DeviceRayleighRitzResult(
        eigenvalues=eigenvalues,
        wavefunctions=basis,
        applied_wavefunctions=None,
        projected_hamiltonian=cp.asarray(whitened, dtype=cp.float64),
        residual_norms=None,
        workspace=None,
        algorithm=_ALGORITHM,
    )


def run_distributed_chebff(
    operator: Any,
    wanted_states: int,
    *,
    settings: ChebFFSettings = ChebFFSettings(),
    spectral_bound: LanczosBoundResult | None = None,
    group: SectorDeviceGroup,
) -> DeviceChebFFResult:
    """CHEBFF first solve with the trial basis spread over ``group``.

    Bounds, cycle count, degrees, sigma carry and the random stream are those
    of :func:`chebff.run_chebff`.  Every cycle uses the generalized Ritz
    solve; a failed stability audit falls back to QR on the owner.
    """

    cp, _ = require_cupy()
    dimension = int(operator.shape[0])
    wanted_states = int(wanted_states)
    if not 1 <= wanted_states <= dimension:
        raise ValueError("wanted_states is outside the operator dimension")
    blocks = uniform_filter_blocks(wanted_states, settings.block_size, settings.polynomial_degree)
    basis_generator = LapackRandom()
    basis = group.random_basis(basis_generator, dimension, group.column_layout(blocks))
    stats = getattr(operator, "timing_stats", None)
    if stats is not None and os.environ.get("PARSEC_CUPY_DEVICE_RANDOM", "0").lower() in {"1", "true", "on"}:
        stats.initial_random_device_values += dimension * wanted_states

    if spectral_bound is None:
        bound = lanczos_upper_bound(
            operator, steps=settings.lanczos_steps, rng=np.random.default_rng(settings.random_seed)
        )
    else:
        bound = spectral_bound
    upper_bound = float(bound.upper_bound)
    smallest_ritz = float(bound.lower_bound)
    lower_bound = _initial_filter_lower_bound(smallest_ritz, upper_bound)
    records: list[ChebFFCycle] = []
    host_eigenvalues = whitened = None

    for cycle_number in range(1, settings.filter_cycles + 1):
        lower_in, upper_in = lower_bound, upper_bound
        with device_stage(operator, "initial_filter_seconds"):
            filtered = group.filter(
                basis, blocks, lower_bound, upper_bound, smallest_ritz,
                settings.reset_recurrence_per_block,
            )
        basis = None
        try:
            with device_stage(operator, "initial_projection_seconds"):
                host_eigenvalues, basis, whitened = group.ritz(operator, filtered, stages=False)
        except GeneralizedRitzStabilityError as error:
            filtered = error.filtered
        if basis is None:
            # Outside the handler: its traceback would keep the arrays of the
            # failed solve on the owner while the basis is gathered there.
            with device_stage(operator, "initial_orthogonalization_seconds"):
                host_eigenvalues, basis, whitened = group.orthonormal_ritz(
                    operator, filtered, basis_generator
                )
        del filtered
        if host_eigenvalues.shape != (wanted_states,):
            raise RuntimeError("Rayleigh--Ritz changed the working state count")
        smallest_ritz = float(host_eigenvalues[0])
        largest_ritz = float(host_eigenvalues[-1])
        lower_bound, upper_bound = _updated_filter_bounds(
            lower_bound, upper_bound, smallest_ritz, largest_ritz
        )
        records.append(
            ChebFFCycle(
                number=cycle_number,
                lower_bound_in=float(lower_in),
                upper_bound_in=float(upper_in),
                lower_bound_out=float(lower_bound),
                upper_bound_out=float(upper_bound),
                smallest_ritz_value=smallest_ritz,
                largest_ritz_value=largest_ritz,
            )
        )

    eigenvalues = cp.asarray(host_eigenvalues, dtype=cp.float64)
    state = DeviceChebFFState(
        operator_dimension=dimension,
        wanted_states=wanted_states,
        eigenvalues=eigenvalues,
        vectors=basis,
        filter_lower_bound=lower_bound,
        spectral_upper_bound=upper_bound,
        smallest_ritz_value=smallest_ritz,
    )
    return DeviceChebFFResult(
        eigenvalues=eigenvalues,
        vectors=basis,
        state=state,
        lanczos_bound=bound,
        cycles=tuple(records),
        last_rayleigh_ritz=_ritz_result(cp, eigenvalues, basis, whitened),
    )


def run_distributed_subspace(
    operator: Any,
    state: DeviceSubspaceState,
    *,
    settings: SubspaceSettings = SubspaceSettings(),
    compute_residuals: bool = False,
    spectral_bound: LanczosBoundResult | None = None,
    consume_state: bool = True,
    group: SectorDeviceGroup,
) -> DeviceSubspaceResult:
    """One later-SCF filter and Ritz pass on a :class:`DistributedBasis`.

    Bounds, degree adaptation and block layout are those of
    :func:`subspace.run_subspace_filter`.  The saved basis is always consumed.
    """

    if compute_residuals:
        raise ValueError("distributed sector states do not form Ritz residuals")
    if not isinstance(state.vectors, DistributedBasis):
        raise TypeError("saved vectors are not a distributed basis")
    cp, _ = require_cupy()
    _validate_state(operator, state)
    lower_bound = _next_filter_lower_bound(state)
    generator = np.random.default_rng(settings.random_seed + state.filters_completed)
    with device_stage(operator, "subspace_bound_seconds"):
        bound = (
            lanczos_upper_bound(operator, steps=settings.lanczos_steps, rng=generator)
            if spectral_bound is None
            else spectral_bound
        )
    upper_bound = float(bound.upper_bound)
    degree = filter_degree(settings, state, lower_bound, upper_bound)
    blocks = subspace_filter_blocks(
        state.working_states, settings.block_size, degree, settings.degree_delta
    )
    with device_stage(operator, "subspace_filter_seconds"):
        filtered = group.filter(
            state.vectors, blocks, lower_bound, upper_bound, lower_bound,
            settings.reset_recurrence_per_block,
        )
    generalized_ritz_failed = state.generalized_ritz_failed
    host_eigenvalues = basis = whitened = None
    if not generalized_ritz_failed:
        try:
            with device_stage(operator, "subspace_ritz_seconds"):
                host_eigenvalues, basis, whitened = group.ritz(operator, filtered, stages=True)
        except GeneralizedRitzStabilityError as error:
            generalized_ritz_failed = True
            filtered = error.filtered
    if basis is None:
        with device_stage(operator, "subspace_orthogonalization_seconds"):
            host_eigenvalues, basis, whitened = group.orthonormal_ritz(operator, filtered, generator)
    del filtered
    if host_eigenvalues.shape != (state.working_states,):
        raise RuntimeError("Rayleigh--Ritz changed the working state count")
    eigenvalues = cp.asarray(host_eigenvalues, dtype=cp.float64)
    next_state = DeviceSubspaceState(
        operator_dimension=state.operator_dimension,
        working_states=state.working_states,
        eigenvalues=eigenvalues,
        vectors=basis,
        filter_lower_bound=lower_bound,
        first_filter=False,
        filters_completed=state.filters_completed + 1,
        ritz_workspace=None,
        generalized_ritz_failed=generalized_ritz_failed,
    )
    return DeviceSubspaceResult(
        eigenvalues=eigenvalues,
        vectors=basis,
        residual_norms=None,
        state=next_state,
        lanczos_bound=bound,
        polynomial_degree_used=degree,
        filter_blocks=blocks,
        rayleigh_ritz=_ritz_result(cp, eigenvalues, basis, whitened),
    )


__all__ = [
    "ColumnLayout",
    "DistributedBasis",
    "SectorDeviceGroup",
    "distributed_state_policy",
    "interleaved_layout_requested",
    "interleaved_ranges",
    "exchange_chunk_bytes",
    "exchange_chunks",
    "concurrent_exchange_requested",
    "paired_exchange_requested",
    "keeps_column_ranges",
    "column_copies_requested",
    "tall_blocks_per_device",
    "condition_helper_policy",
    "equal_slabs_requested",
    "shared_pool_release_requested",
    "pair_slab_bytes",
    "wider_slabs_fit",
    "slab_columns",
    "slab_workspace",
    "block_rounds",
    "projection_slabs",
    "shared_basis_devices",
    "shared_basis_fits",
    "shared_block_bytes",
    "run_distributed_chebff",
    "run_distributed_subspace",
    "sector_device_group",
]
