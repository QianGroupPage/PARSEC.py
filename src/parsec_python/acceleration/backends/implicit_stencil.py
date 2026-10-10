"""Exact affine-tile compression; no grid points are added.

Regular tiles store one neighbor offset/code per slot. Other tiles retain
every original index/code. Slot order, FP64 coefficients and fused operations
are inherited unchanged from the production stencil. By default a symmetry
sector that one device filters is packed from the size where complete
calculations ran faster (see :func:`implicit_tile_for_sector`).  A sector
that several devices filter is packed once, by its owner: the others upload
the arrays the owner holds (:class:`PackedTileHostMetadata`).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isqrt
import os
from threading import Lock
import numpy as np

_CACHE = {}
_LOCK = Lock()
_MISSING = np.iinfo(np.int32).min
_TILES = (16, 32, 64, 128, 256)


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


def implicit_tile_for_sector(rows, device_count, float32_filter=False, sectors_on_device=1):
    """Return the tile size for the stencil of one symmetry sector; 0 packs none.

    An explicit ``PARSEC_CUPY_IMPLICIT_TILE`` is returned as it is.  ``auto``
    packs tiles of 16 rows where they were measured to pay and are supported:

    * ``device_count``, the devices that filter the basis of the sector, is
      one.  A basis shared among devices, and a filter spread over them, have
      a row count of their own (:func:`implicit_tile_for_group`).
    * The sector has at least ``PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS`` rows,
      650,000 by default.  With four sectors on four A100 the program
      took 22.0 s slot-major and 21.4 s with tiles at 1,031,322 rows, and
      12.6 against 12.2 s at 677,542, the smallest sector that was run:
      the filter took 8.6% and 6.7% fewer seconds, the host packing 0.1
      to 0.3 s before the SCF.  See ``GPU_COMPACT_OPTIMIZATION.md``.
    * That limit is for a device that filters this sector alone.  Where it
      filters ``sectors_on_device`` of them one after another, the limit is
      divided by the square root of that number: 459,620 rows for two
      sectors, 325,000 for four.  The stencils are packed once, on the
      host, while the filter seconds that a device saves add up over its
      sectors (64.1 to 60.0 s and 29.8 to 28.4 s for the same two sizes
      with four sectors on one device); a smaller sector saves less than
      in proportion (fewer states, and 61.2% of its rows in regular tiles
      at 407,478 rows against 66.6% at 677,542), so the limit falls more
      slowly than the number of sectors grows.  No sector below 677,542
      rows was run with tiles: the two lower limits are derived.  More than
      four sectors count as four: a process that solves more of them packs
      more stencils as well, and none was run.
    * ``float32_filter`` is false: the opt-in FP32 recurrence does not read
      tiles, and an explicit request for it outranks this default.

    The limits were measured on A100 alone, with one device for the basis
    of a sector.  A device whose tile kernels do not gain pays the packing
    for nothing (an RTX 5070 Laptop gained nothing from 274,784 to
    1,031,322 rows); ``PARSEC_CUPY_IMPLICIT_TILE=0`` keeps the slot-major
    stencil there.
    """
    if int(device_count) != 1:
        return implicit_tile_for_group(rows, float32_filter)
    tile = implicit_tile_setting()
    if tile is not None:
        return tile
    raw = os.environ.get("PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS", "650000").strip()
    try:
        minimum_rows = int(raw)
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


def implicit_tile_for_group(rows, float32_filter=False):
    """Return the tile size for a sector whose basis or filter several devices share.

    The owner of the sector packs, and the other devices read the arrays it
    packed (:class:`PackedTileHostMetadata`): the group has one layout.  An
    explicit ``PARSEC_CUPY_IMPLICIT_TILE`` is returned as it is.  ``auto``
    packs tiles of 16 rows for a sector of at least
    ``PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS`` rows, 1,100,000 by default,
    that is not to hold the FP32 recurrence.

    That row count is assumed, not measured against others; complete
    calculations with a tiled group have run on 8 and 16 A100 GPUs since.
    It was the limit of a sector on one device until the runs on
    one, two and four devices lowered that
    (:func:`implicit_tile_for_sector`), and it has a setting of its own so
    that the limit of one device, ``PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS``,
    moves without it: the sectors of 677,542 and 1,031,322 rows, which one
    device now reads in tiles, were run slot-major only where a group
    shares them.  On one device per sector the tiles saved 6.6% of the filter
    seconds at 677,542 rows, 8.6% at 1,031,322, 10.6% at 1,924,792 and 11.5%
    at 2,873,400, and packing took 0.02 to 0.07 s per sector.  A group of
    ``d`` devices filters a sector in a ``d``-th of the time and packs it
    once, so those shares of its slot-major filter seconds would be 0.4 s at
    1,194,720 rows on 16 GPUs and 0.1 to 0.5 s at the two sizes below the
    row count, on 16 and 8 GPUs.  A run with a lower setting, or with a
    named tile size, against the default shows whether a group gains there.
    See ``GPU_COMPACT_OPTIMIZATION.md``.
    """
    tile = implicit_tile_setting()
    if tile is not None:
        return tile
    name = "PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS"
    try:
        minimum_rows = int(os.environ.get(name, "1100000").strip())
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


def initialize_implicit(instance, cp, metadata, tile):
    from .cupy_compile import compile_cupy_raw
    source = implicit_source(tile)
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
    key = int(cp.cuda.Device().id), tile
    with _LOCK:
        if key not in _CACHE:
            kernels = tuple(cp.RawKernel(source, name, options=("--std=c++11",))
                            for name in ("stencil_major_spmm6", "stencil_major_chebyshev6"))
            for kernel in kernels:
                compile_cupy_raw(kernel)
            _CACHE[key] = kernels
        instance.apply_kernel, instance.recurrence_kernel = _CACHE[key]
