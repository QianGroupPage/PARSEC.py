"""Exact affine-tile compression; no grid points are added.

Regular tiles store one neighbor offset/code per slot. Other tiles retain
every original index/code. Slot order, FP64 coefficients and fused operations
are inherited unchanged from the production stencil. By default the stencil
of a symmetry sector is packed where the filter work of the run repays the
packing (see :func:`tiles_repay_packing` and
:func:`implicit_tile_for_sector`).  A sector that several devices filter is
packed once, by its owner: the others upload the arrays the owner holds
(:class:`PackedTileHostMetadata`).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isqrt
import os
from threading import Lock
from typing import Callable
import numpy as np

_CACHE = {}
_LOCK = Lock()
_MISSING = np.iinfo(np.int32).min
_TILES = (16, 32, 64, 128, 256)
# What ``auto`` weighs for a sector (:func:`tiles_repay_packing`), measured in
# first calculations on A100 nodes with four packing threads.  Seconds of the
# host per row of a stencil it packs: the set-up of the sector operators of a
# rank grew by 63 to 67 ns per row where it packed two or four stencils of
# 620,736 and 1,031,322 rows, and by less elsewhere.  That timer does not
# hold all the tiles cost before the SCF: the wall time there grew by 46 ns
# per row of the four stencils of a rank on 4 GPUs and by 118 to 131 ns per
# row of the two of a rank on 8, two runs of each.  The constant is that of
# a sector alone on its device; for the shared sectors of 8 GPUs it is 53 to
# 59% of what was measured (``GPU_COMPACT_OPTIMIZATION.md``, where the
# counted saving is low as well and the two are set against each other).
# Seconds a device saves
# per row, column and Hamiltonian application of its filter, for every unit of
# the share of the rows that lie in regular tiles above the share at which
# the tiles save nothing: 4.7, 5.9 and 7.1 ps at shares of 0.666, 0.705 and
# 0.755 (677,542, 1,031,322 and 1,924,792 rows) lie on that line to 0.15 ps.
PACKING_SECONDS_PER_ROW = 70.0e-9
SAVED_SECONDS_PER_UPDATE = 28.0e-12
REGULAR_SHARE_WITHOUT_GAIN = 0.50
# SCF steps after the first whose filter is counted for a run, and the tiles
# from which the share of regular ones is taken.
COUNTED_LATER_STEPS = 8
REGULAR_SHARE_SAMPLE = 1024


def implicit_tile_setting():
    """Return ``PARSEC_CUPY_IMPLICIT_TILE``: a tile size, 0, or ``None`` for ``auto``.

    ``auto`` is the default.  It leaves the choice to
    :func:`implicit_tile_for_sector`; a stencil outside symmetry sectors is
    packed by an explicit size only.
    """
    raw = os.environ.get("PARSEC_CUPY_IMPLICIT_TILE", "auto").strip().lower()
    if raw in ("", "auto"):
        return None
    try:
        tile = int(raw)
    except ValueError:
        tile = -1
    if tile and tile not in _TILES:
        raise ValueError(
            "PARSEC_CUPY_IMPLICIT_TILE must be auto, 0, 16, 32, 64, 128 or 256"
        )
    return tile


def counted_filter_applications(eigensolver, max_iterations):
    """Return the Hamiltonian applications per column that a run is counted to filter with.

    ``eigensolver`` holds the eigensolver settings of the input and
    ``max_iterations`` the SCF steps it allows.  What the input fixes is
    counted in full: the first solve of CHEBFF filters every column
    ``first_filter_cycles`` times at ``first_filter_degree``.  CHEBDAV, which
    filters until its residuals are met, counts as one pass at that degree and
    a first solve without a filter as none.  How many steps follow is not
    known before they are made.  ``COUNTED_LATER_STEPS`` of them are counted,
    or as many as the input allows if it allows fewer, each at the lowest
    degree the later filter falls to (``max(10, filter_degree_delta + 1)``,
    or ``filter_degree`` below that).  No first calculation on record
    converged in fewer than ten steps (hydrogen-terminated clusters of 3,480
    to 39,368 electrons: 10 or 11; ten systems of other species: 15 or 16),
    and the later filters of the clusters ran at degrees of 10 to 15.
    """
    method = getattr(eigensolver, "method", "chebff")
    degree = int(eigensolver.first_filter_degree)
    first = (
        int(eigensolver.first_filter_cycles) * degree
        if method == "chebff"
        else degree if method == "chebdav" else 0
    )
    lowest = min(
        int(eigensolver.filter_degree),
        max(10, int(eigensolver.filter_degree_delta) + 1),
    )
    later = min(COUNTED_LATER_STEPS, max(0, int(max_iterations) - 1))
    return first + later * lowest


def regular_tile_share(metadata, tile=16, sample=REGULAR_SHARE_SAMPLE):
    """Return the share of the rows of a slot-major stencil that affine tiles would hold.

    A tile is affine where :func:`pack_affine_tiles` finds it so: every slot
    has its neighbour in all rows of the tile or in none, at one offset from
    the row and with one coefficient.  Up to ``sample`` tiles, spread evenly
    over the rows, are looked at; a stencil with no more tiles than that
    gets the share its packing reports (``regular_rows`` over ``rows``).  The
    tiles are the same for every run of a stencil, so the share is too.
    1,024 of them took 3 to 7 ms on the host of a laptop for stencils of
    105,126 to 5.66 million rows, and gave the share of all tiles to 0.01
    for the four sectors that were compared, all of one family of grids:
    0.443, 0.564, 0.650 and 0.672 against 0.441, 0.571, 0.657 and 0.666 at
    105,126, 280,790, 620,736 and 677,542 rows.  That is no bound.  Where
    the tiles are regular or not independently of one another the sample has
    the error of its size, 0.015 at a share of two thirds (0.014 to 0.016
    in the root mean square over 40 such stencils at each of five sizes,
    0.041 at most): a rule whose two sides lie within about a tenth of each
    other is decided by the sample.
    """
    if tile not in _TILES:
        raise ValueError("implicit tile must be 16, 32, 64, 128, or 256")
    neighbors, codes = metadata.neighbors, metadata.coefficient_codes
    slots, rows = neighbors.shape
    # A last tile that is not full is never affine.
    whole = rows // tile
    if not whole:
        return 0.0
    if whole <= int(sample):
        picked = np.arange(whole, dtype=np.int64)
    else:
        picked = np.unique(
            np.rint(np.linspace(0, whole - 1, int(sample))).astype(np.int64)
        )
    # The rows of the tiles that are looked at, read out of the arrays as
    # they are: nothing of the size of the stencil is copied.
    at = (picked[:, None] * tile + np.arange(tile, dtype=np.int64)).ravel()
    nn = neighbors[:, at].reshape(slots, -1, tile)
    cc = codes[:, at].reshape(slots, -1, tile)
    there = nn >= 0
    delta = nn - at.astype(np.int32).reshape(1, -1, tile)
    same = (delta == delta[:, :, :1]) & (cc == cc[:, :, :1])
    same |= ~there
    same &= there == there[:, :, :1]
    regular = int(np.count_nonzero(same.all(axis=(0, 2))))
    # In integers, so that the share of all tiles is the quotient of the
    # packer's own counts.
    return regular * whole * tile / (int(picked.size) * int(rows))


def tiles_repay_packing(columns, applications, devices, regular_share):
    """Whether the tiles of a sector save a run more seconds than their packing takes.

    The host packs a stencil once, in seconds proportional to its rows.  A
    device saves with the tiles, in every row, column and Hamiltonian
    application of its filter, the index reads of the rows that lie in
    regular tiles and pays a descriptor read in all of them: the saving is
    proportional to the share of regular rows above the share at which the
    two cancel.  The devices of a process work side by side while its host
    packs one stencil after another, so a second saved on a device is a
    second of the run only if every one of them saves it: the packing of a
    row is held against the saving of that row on all ``devices``.  The rows
    cancel:

    ``SAVED_SECONDS_PER_UPDATE * (regular_share - REGULAR_SHARE_WITHOUT_GAIN)
    * columns * applications >= PACKING_SECONDS_PER_ROW * devices``

    ``columns`` are the vectors of the basis of the sector, ``applications``
    those of :func:`counted_filter_applications`, ``devices`` the devices of
    the process, and ``regular_share`` the share of
    :func:`regular_tile_share` or a callable that returns it; it is asked
    for only where the columns could repay a stencil of regular tiles alone.

    A sector with few states gets no tiles at any number of rows, and one
    of many states gets none either where half of its rows or more lie in
    tiles that are not regular, as in a sector of few rows.  The three
    constants were measured on A100 nodes and on no other device or host;
    ``PARSEC_CUPY_IMPLICIT_TILE=0`` is the setting for a device whose tile
    kernels do not gain.
    """
    work = int(columns) * int(applications)
    cost = PACKING_SECONDS_PER_ROW * max(1, int(devices))
    if SAVED_SECONDS_PER_UPDATE * (1.0 - REGULAR_SHARE_WITHOUT_GAIN) * work < cost:
        return False
    share = float(regular_share() if callable(regular_share) else regular_share)
    return (
        SAVED_SECONDS_PER_UPDATE * (share - REGULAR_SHARE_WITHOUT_GAIN) * work >= cost
    )


@dataclass(frozen=True)
class FilterPlan:
    """What an input fixes of the filter work of a run, before its sectors are known.

    ``states`` are the eigenpairs the SCF asks for, ``buffer`` the working
    vectors of a subspace beside its share of them, and ``applications``
    those of :func:`counted_filter_applications`.
    """

    states: int
    buffer: int
    applications: int


@dataclass(frozen=True)
class FilterWork:
    """The filter work of a run in one sector, as :func:`tiles_repay_packing` reads it."""

    columns: int
    applications: int
    devices: int
    regular_share: "float | Callable[[], float]"

    def repays_packing(self) -> bool:
        return tiles_repay_packing(
            self.columns, self.applications, self.devices, self.regular_share
        )


def implicit_tile_for_sector(
    rows, device_count, float32_filter=False, sectors_on_device=1, work=None
):
    """Return the tile size for the stencil of one symmetry sector; 0 packs none.

    An explicit ``PARSEC_CUPY_IMPLICIT_TILE`` is returned as it is.  ``auto``
    packs tiles of 16 rows where ``float32_filter`` is false (the opt-in FP32
    recurrence does not read tiles, and an explicit request for it outranks
    this default) and

    * ``work``, the filter work of the run in this sector
      (:class:`FilterWork`), repays the packing
      (:func:`tiles_repay_packing`).  That is the rule of every sector,
      whatever the number of devices that filter its basis, where the
      environment names no row limit for it.
    * ``PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS`` is set and the sector, whose
      basis one device filters (``device_count`` of one), has at least that
      many rows.  The limit is for a device that filters this sector alone.
      Where it filters ``sectors_on_device`` of them one after another, the
      limit is divided by the square root of that number, of four at most.
      ``650000`` is the limit that was the default before the rule: 459,620
      rows for two sectors on a device, 325,000 for four.
    * ``PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`` is set and the sector,
      whose basis or filter several devices share, has at least that many
      rows (:func:`implicit_tile_for_group`); ``1100000`` was the default.

    The two variables at those values give every sector the choice it had
    before the rule.  A caller that gives no ``work`` gets the limits too,
    at those values where the variables are unset; the solver of the drivers
    gives it.  See ``GPU_COMPACT_OPTIMIZATION.md`` for the runs behind the
    rule and behind the limits.
    """
    if int(device_count) != 1:
        return implicit_tile_for_group(rows, float32_filter, work)
    tile = implicit_tile_setting()
    if tile is not None:
        return tile
    raw = os.environ.get("PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS")
    if raw is None:
        if work is not None:
            return 16 if not float32_filter and work.repays_packing() else 0
        raw = "650000"
    try:
        minimum_rows = int(raw.strip())
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS must be an integer"
        ) from error
    if minimum_rows < 1:
        raise ValueError("PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS must be positive")
    # The limit over the square root of the sectors, of four at most, rounded
    # up, in integers.
    sectors = min(4, max(1, int(sectors_on_device)))
    minimum_rows = isqrt(-(-minimum_rows ** 2 // sectors) - 1) + 1
    if float32_filter or int(rows) < minimum_rows:
        return 0
    return 16


def implicit_tile_for_group(rows, float32_filter=False, work=None):
    """Return the tile size for a sector whose basis or filter several devices share.

    The owner of the sector packs, and the other devices read the arrays it
    packed (:class:`PackedTileHostMetadata`): the group has one layout.  An
    explicit ``PARSEC_CUPY_IMPLICIT_TILE`` is returned as it is.  ``auto``
    packs tiles of 16 rows for a sector that is not to hold the FP32
    recurrence and whose filter work repays the packing (``work``, see
    :func:`implicit_tile_for_sector`), or, where
    ``PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`` is set, that has at least
    that many rows.

    ``1100000`` was the default of that variable before the rule, a row
    count that had been assumed and not measured against others.  Shared
    sectors below it have run with tiles on A100 nodes since: at 677,542,
    1,031,322 and 1,094,788 rows the filter of a sector took 7 to 10% fewer
    seconds on 8 and on 16 GPUs (2.66 to 2.47 s and 1.40 to 1.31 s, 5.83 to
    5.32 and 2.97 to 2.71, 8.38 to 7.65 and 4.29 to 3.92), the set-up of the
    sector operators of a rank at most 0.15 s more, and the total energies
    were the same to the last bit.  The wall time before the SCF grew by
    more than that set-up shows, 0.18 to 0.26 s on 8 GPUs, which leaves the
    677,542-row sectors next to no gain there; on 16 GPUs it could not be
    read.  See ``GPU_COMPACT_OPTIMIZATION.md``.
    """
    tile = implicit_tile_setting()
    if tile is not None:
        return tile
    name = "PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS"
    raw = os.environ.get(name)
    if raw is None:
        if work is not None:
            return 16 if not float32_filter and work.repays_packing() else 0
        raw = "1100000"
    try:
        minimum_rows = int(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if minimum_rows < 1:
        raise ValueError(f"{name} must be positive")
    return 16 if int(rows) >= minimum_rows and not float32_filter else 0


def _pack_affine_tiles_reference(metadata, tile=32):
    if tile not in (16, 32, 64, 128, 256):
        raise ValueError("implicit tile must be 16, 32, 64, 128, or 256")
    neighbors, codes = metadata.neighbors, metadata.coefficient_codes
    slots, rows = neighbors.shape
    count = (rows + tile - 1) // tile
    # Header descriptors are negative for affine tiles, positive otherwise.
    headers = np.empty(count, np.int32)
    packed_n, packed_c = [], []
    offset, regular = count, 0
    for k, start in enumerate(range(0, rows, tile)):
        stop = min(start + tile, rows)
        n, c = neighbors[:, start:stop], codes[:, start:stop]
        active = n >= 0
        delta = np.where(active, n - np.arange(start, stop), _MISSING)
        affine = np.all(delta == delta[:, :1]) and np.all(
            np.where(active, c, 0) == np.where(active[:, :1], c[:, :1], 0))
        if affine:
            nn, cc = delta[:, 0].astype(np.int32), c[:, 0]
            headers[k] = -offset
            regular += stop - start
        else:
            nn = np.full((slots, tile), -1, np.int32)
            cc = np.zeros((slots, tile), np.uint8)
            nn[:, :stop-start], cc[:, :stop-start] = n, c
            nn, cc = nn.ravel(), cc.ravel()
            headers[k] = offset
        packed_n.append(nn)
        packed_c.append(cc)
        offset += nn.size
        if offset > np.iinfo(np.int32).max:
            raise ValueError("packed stencil exceeds int32 offsets")
    return (np.concatenate((headers, *packed_n)),
            np.concatenate((np.zeros(count, np.uint8), *packed_c)),
            dict(tile=tile, regular_rows=regular, rows=rows,
                 original_bytes=neighbors.nbytes+codes.nbytes,
                 packed_bytes=offset*5))


def pack_worker_setting():
    """Return ``PARSEC_CUPY_IMPLICIT_PACK_WORKERS``, 4 by default.

    The threads that pack the chunks of one stencil; 1 packs them one after
    another in the calling thread, as before.  An operator that is to pack
    tiles reads this before it builds them, so that a value that is no
    positive integer is an error there and not a reason to drop the tiles.
    """
    raw = os.environ.get("PARSEC_CUPY_IMPLICIT_PACK_WORKERS", "4").strip()
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_IMPLICIT_PACK_WORKERS must be a positive integer"
        ) from error
    if workers < 1:
        raise ValueError("PARSEC_CUPY_IMPLICIT_PACK_WORKERS must be a positive integer")
    return workers


def pack_affine_tiles(metadata, tile=32):
    """Bounded vectorized packing, without a Python loop over every tile."""
    if tile not in (16,32,64,128,256):
        raise ValueError("implicit tile must be 16,32,64,128,256")
    neighbors,codes=metadata.neighbors,metadata.coefficient_codes
    slots,rows=neighbors.shape; count=(rows+tile-1)//tile
    regular=np.zeros(count,dtype=bool)
    def chunk(first,last):
        size=(last-first)*tile
        nn=np.full((slots,size),-1,np.int32); cc=np.zeros((slots,size),np.uint8)
        valid=min(rows-first*tile,size)
        nn[:,:valid]=neighbors[:,first*tile:first*tile+valid]
        cc[:,:valid]=codes[:,first*tile:first*tile+valid]
        return nn.reshape(slots,-1,tile).transpose(1,0,2),cc.reshape(slots,-1,tile).transpose(1,0,2)
    def classify(first):
        last=min(first+1024,count); nn,cc=chunk(first,last)
        rr=np.arange(first*tile,last*tile,dtype=np.int32).reshape(-1,1,tile)
        delta=np.where(nn>=0,nn-rr,_MISSING)
        active_codes=np.where(nn>=0,cc,0)
        regular[first:last]=np.all(delta==delta[:,:,:1],axis=(1,2)) & np.all(active_codes==active_codes[:,:,:1],axis=(1,2))
    def fill(first):
        last=min(first+1024,count); nn,cc=chunk(first,last)
        for affine in (True,False):
            pick=np.flatnonzero(regular[first:last]==affine)
            if not len(pick): continue
            if affine:
                raw=nn[pick,:,0]
                values=np.where(raw>=0,raw-((first+pick)*tile)[:,None],_MISSING)
                co=cc[pick,:,0]
            else:
                values=nn[pick].reshape(len(pick),-1); co=cc[pick].reshape(len(pick),-1)
            at=starts[first+pick,None]+np.arange(values.shape[1])
            packed_n[at]=values; packed_c[at]=co
    # A chunk classifies its own tiles and, once every tile has its place,
    # writes its own part of the packed arrays: the chunks can run side by
    # side, and the NumPy work in them releases the interpreter lock.
    firsts=range(0,count,1024); workers=min(pack_worker_setting(),len(firsts))
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix="parsec-tile-pack") if workers>1 else None
    def each(work):
        for _ in (pool.map(work,firsts) if pool else map(work,firsts)): pass
    try:
        each(classify)
        if rows%tile: regular[-1]=False
        lengths=np.where(regular,slots,slots*tile)
        starts=count+np.r_[0,np.cumsum(lengths[:-1],dtype=np.int64)]
        end=count+int(lengths.sum())
        if end>np.iinfo(np.int32).max: raise ValueError("packed stencil exceeds int32 offsets")
        packed_n=np.empty(end,np.int32); packed_c=np.zeros(end,np.uint8)
        packed_n[:count]=np.where(regular,-starts,starts)
        each(fill)
    finally:
        if pool: pool.shutdown()
    return packed_n,packed_c,dict(tile=tile,regular_rows=int(regular.sum())*tile,rows=rows,
                                 original_bytes=neighbors.nbytes+codes.nbytes,packed_bytes=end*5)


def unpack_affine_tiles(indices, codes, rows, slots, tile):
    """Independent host reconstruction for exact structural validation."""
    n, c = np.full((slots, rows), -1, np.int32), np.zeros((slots, rows), np.uint8)
    for row in range(rows):
        descriptor = int(indices[row // tile])
        if descriptor < 0:
            at = -descriptor + np.arange(slots)
            delta = indices[at]
            n[:, row] = np.where(delta == _MISSING, -1, row + delta)
        else:
            at = descriptor + np.arange(slots)*tile + row % tile
            n[:, row] = indices[at]
        c[:, row] = codes[at]
    return n, c


@dataclass(frozen=True)
class PackedTileHostMetadata:
    """Host copy of a stencil that is packed into affine tiles already.

    ``neighbors`` and ``coefficient_codes`` are the two arrays of
    :func:`pack_affine_tiles` and ``statistics`` its record of them.  An
    operator that is given this in place of the slot-major metadata uploads
    the arrays as they are and packs nothing (:func:`initialize_implicit`).
    It has no other layout to give way to.
    """

    shape: tuple
    slot_count: int
    neighbors: np.ndarray
    coefficient_codes: np.ndarray
    coefficient_palette: np.ndarray
    statistics: dict

    def __post_init__(self):
        rows, tile = int(self.shape[0]), self.tile
        if tile not in _TILES or int(self.statistics["rows"]) != rows:
            raise ValueError("packed stencil statistics do not match its shape")
        neighbors, codes = self.neighbors, self.coefficient_codes
        if neighbors.dtype != np.int32 or codes.dtype != np.uint8:
            raise ValueError("packed stencil arrays must be int32 and uint8")
        if neighbors.ndim != 1 or codes.shape != neighbors.shape:
            raise ValueError("packed stencil arrays must be flat and of one length")
        if 5 * neighbors.size != int(self.statistics["packed_bytes"]) or int(self.slot_count) < 1:
            raise ValueError("packed stencil arrays do not match their statistics")
        # The kernels read the palette as FP64; the codes are one byte each.
        palette = self.coefficient_palette
        if palette.dtype != np.float64 or palette.ndim != 1 or not 1 <= palette.size <= 256:
            raise ValueError("packed stencil palette must be 1 to 256 float64 coefficients")

    @property
    def tile(self):
        return int(self.statistics["tile"])


def implicit_source(tile):
    """The source of the two stencil kernels, reading tiles of ``tile`` rows."""
    from .cupy_stencil_major import _CUDA_SOURCE
    old = '''const long long metadata_offset =
            static_cast<long long>(slot) * row_count + row;
        const int source_row = neighbors[metadata_offset];'''
    new = f'''const int descriptor = neighbors[row / {tile}];
        const long long metadata_offset = descriptor < 0
            ? static_cast<long long>(-descriptor) + slot
            : static_cast<long long>(descriptor) + slot * {tile} + row % {tile};
        const int encoded = neighbors[metadata_offset];
        const int source_row = descriptor < 0
            ? (encoded == (-2147483647 - 1) ? -1 : row + encoded)
            : encoded;'''
    if _CUDA_SOURCE.count(old) != 2:
        raise RuntimeError("production stencil ABI changed")
    return _CUDA_SOURCE.replace(old, new)


def stencil_source(tile, slot_count, kernel="former"):
    """The source of the two stencil kernels for tiles of ``tile`` rows (0: slot-major).

    ``kernel`` is a value of ``PARSEC_CUPY_STENCIL_KERNEL`` (:mod:`.stencil_pipeline`).
    ``former`` gives the source as it was, :func:`implicit_source` or the slot-major source;
    ``pipelined`` is generated from it for a stencil of ``slot_count`` slots, with the fast path
    in ``stencil_major_chebyshev6`` alone.
    """
    from .cupy_stencil_major import _CUDA_SOURCE
    base = implicit_source(tile) if tile else _CUDA_SOURCE
    if kernel == "former":
        return base
    from .stencil_pipeline import pipelined_source
    return pipelined_source(base, slot_count, tile)


def initialize_implicit(instance, cp, metadata, tile):
    from .cupy_compile import compile_cupy_raw
    from .stencil_pipeline import compiled_kernels, stencil_kernel_setting
    setting = stencil_kernel_setting()
    if isinstance(metadata, PackedTileHostMetadata):
        if metadata.tile != tile:
            raise ValueError(f"stencil is packed into tiles of {metadata.tile}, not {tile}")
        n, c = metadata.neighbors, metadata.coefficient_codes
        instance.implicit_statistics = dict(metadata.statistics)
        instance.slot_count = int(metadata.slot_count)
    else:
        n, c, instance.implicit_statistics = pack_affine_tiles(metadata, tile)
    instance.neighbors = cp.asarray(n)
    instance.coefficient_codes = cp.asarray(c)
    instance.coefficient_palette = cp.asarray(metadata.coefficient_palette, dtype=cp.float64)
    # The former kernels keep the key (device, tile) they had under every value; the recurrence kernel of
    # ``pipelined`` is generated for the slot count, which a stencil that arrives packed names in its
    # metadata alone, and kept beside them.
    instance.hold_kernels(compiled_kernels(
        cp, _CACHE, _LOCK, (int(cp.cuda.Device().id), tile), implicit_source(tile), tile, instance.slot_count,
        setting, compile_cupy_raw))
