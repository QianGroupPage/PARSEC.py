"""Host staging of sector states that their device cannot keep between SCF steps.

A process whose sector states are spilled to the host
(``PARSEC_CUPY_SECTOR_STATE_STORAGE``,
:meth:`symmetry.CuPySymmetrySCFEigensolver._configure_large_problem_allocator`)
moves the basis of every sector three times per SCF step: up before its
solve, down behind it and up once more for the density.  This module holds
what those transfers go through: one host buffer per sector that lasts as
long as its solver, page-locked within a bound, and copies between such a
buffer and the device that stage nothing in between.  The bytes that move
are the same, so no result changes.
"""

from __future__ import annotations

import os
import sys
import threading
import weakref
from time import perf_counter
from typing import Any, Mapping

import numpy as np


_BUFFERS = ("persistent", "fresh")
_PINNED_POOLS = ("keep", "release")
# The share of the available host memory that the buffers of a process may
# page-lock where no bytes are named: see page_lock_bound.
_LOCK_DIVISOR = 2


def spill_buffer_mode() -> str:
    """The staging selected by ``PARSEC_CUPY_SPILL_BUFFERS``.

    ``persistent``, the default: every sector whose states are kept on the
    host has one host buffer of the size of its basis, taken once
    (:class:`SpillBuffer`).  A download writes the device array into it
    (``ndarray.get(out=)``), an upload reads it into a new device array
    (``ndarray.set``), and the density step uploads its columns the same
    way (:class:`Occupations.device_density.CuPyDeviceDensityBuilder`).
    Neither call takes a staging buffer or a second array on the device,
    which a hook on both allocators counted on a laptop GPU (CuPy 14.0.1),
    and no page of the host is faulted in or page-locked again after the
    first step.  The host holds every spilled basis once.

    ``fresh`` is the staging as it was: ``cupy.asnumpy`` into a new
    pageable array for every download, whose pages are faulted in each
    time, and ``cupy.asarray`` for every upload, which copies the host
    array into a block of CuPy's pinned pool and from there to the device.
    That pool rounds a request up to a power of two, so the block of a
    basis of 18.84 GiB is one of 32 GiB, on top of the spilled bases.
    ``PARSEC_CUPY_SPILL_DOWNLOAD_ORDER`` belongs to this staging.  The
    buffers read it only for vectors that are not Fortran-ordered on their
    device, which that call still downloads (:meth:`SpillBuffer.store`).

    One array of 1,924,792 x 1,314 doubles (18.84 GiB, a sector of 10,456
    electrons) between an A100 and its host: ``cupy.asarray`` 17.99 s with
    the pinned pool empty, 16.2 s of them the page-locking of that block,
    and 1.75 s (10.7 GiB/s) with the block in the pool; ``cupy.asnumpy``
    into a new array 2.62 s (7.2 GiB/s).  Into and out of a page-locked
    buffer of the array's size 0.758 s up and 0.770 s down (24.8 and 24.5
    GiB/s), and with a pageable one that is kept 1.40 and 1.52 s (13.4 and
    12.4 GiB/s).  Page-locking such a buffer took 9.6 to 9.7 s, once.  All
    paths returned the bits of the source.
    """

    value = os.environ.get("PARSEC_CUPY_SPILL_BUFFERS", "persistent").strip().lower()
    if value not in _BUFFERS:
        raise ValueError("PARSEC_CUPY_SPILL_BUFFERS must be persistent or fresh")
    return value


def spill_pinned_pool() -> str:
    """What the density step does to CuPy's pinned pool where sector states are kept on the host.

    ``PARSEC_CUPY_SPILL_PINNED_POOL``: ``release`` empties it after every
    density, as before; ``keep``, the default, leaves it where the density
    was handed a sector whose states are on the host
    (:meth:`Occupations.symmetry_density.CuPySymmetryDensityBuilder._release_large_unused_pool`).
    A process whose sectors are all on their devices empties it as it did
    under either value.

    The pinned pool holds the blocks through which ``cupy.asarray`` sends a
    host array to a device.  With the staging of ``fresh`` (see
    :func:`spill_buffer_mode`) one of them is the block of a whole basis, and
    emptying the pool returned it after every density only for the first
    upload of the next step to page-lock it again: 5.82 to 5.89 s for the
    return of a block of 32 GiB and 15.9 to 16.3 s for locking it, on an
    A100 node.  Kept, the block serves every upload of the run, and the
    host holds it between the steps as well; its high-water mark is the
    same, since the block stood beside the bases in every step.  With the
    persistent buffers no basis goes through the pool, which then holds
    only the blocks of smaller uploads, the eigenvalues and occupations of
    a sector among them.
    """

    value = os.environ.get("PARSEC_CUPY_SPILL_PINNED_POOL", "keep").strip().lower()
    if value not in _PINNED_POOLS:
        raise ValueError("PARSEC_CUPY_SPILL_PINNED_POOL must be keep or release")
    return value


def spill_pinned_bytes() -> int | None:
    """``PARSEC_CUPY_SPILL_PINNED_BYTES``: the most bytes that the spill buffers of a process page-lock.

    ``None`` where it is not set: the bound then follows from the host
    (:func:`page_lock_bound`).  ``0`` page-locks nothing, so that every
    buffer is pageable.  A named value is taken as it is.
    """

    name = "PARSEC_CUPY_SPILL_PINNED_BYTES"
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def host_available_bytes() -> int | None:
    """Bytes of physical memory that the host reports as available now.

    What new work can take without paging anything out: ``MemAvailable`` of
    ``/proc/meminfo`` under Linux, the available physical memory of
    ``GlobalMemoryStatusEx`` under Windows, and the free pages of
    ``sysconf`` elsewhere.  ``None`` where none of them can be read.
    """

    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong),
                    ("available", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.length = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        except (AttributeError, OSError, ValueError):
            pass
        return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def locked_memory_limit() -> int | None:
    """The soft limit of the system on the locked memory of this process, ``None`` where it has none.

    ``RLIMIT_MEMLOCK``.  Whether the CUDA driver holds its page-locking to
    that limit was not measured, so it is not left to a lock that fails:
    the limit is read as what the system means a process to lock.
    """

    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft == resource.RLIM_INFINITY or soft < 0:
            return None
        return int(soft)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def page_lock_bound() -> tuple[int, str]:
    """The most bytes that the spill buffers of this process page-lock, and where that comes from.

    Page-locked memory is out of reach of everything else on the host
    until it is given back: the system cannot page it out to make room.
    The buffers of a process are therefore page-locked only while their sum
    stays within half of the memory that the host reports as available when
    the storage of the states is decided (:func:`host_available_bytes`), so
    that as much stays available as they take, and within the limit of the
    system on locked memory where it has one (:func:`locked_memory_limit`).
    A host whose memory cannot be read locks nothing.  The buffers beyond
    the bound are pageable, and so is one that the system refuses to lock:
    they are copied directly all the same, at about half the rate.

    ``PARSEC_CUPY_SPILL_PINNED_BYTES`` names the bound instead
    (:func:`spill_pinned_bytes`), for a host that the rule reads wrongly:
    several processes that spill on one host each take the half, and a
    process inside a memory limit of its control group sees the memory of
    the host.  The bound is one of time and not of results: a pageable
    buffer holds the same bytes.

    What the buffers need is not bounded here.  The host holds every
    spilled basis whether its buffer is page-locked or not, as it did
    before these buffers, and the rule that spills does not ask whether it
    can.
    """

    named = spill_pinned_bytes()
    if named is not None:
        return named, "PARSEC_CUPY_SPILL_PINNED_BYTES"
    available = host_available_bytes()
    if available is None:
        return 0, "the available memory of the host could not be read"
    bound, source = available // _LOCK_DIVISOR, "half of the available host memory"
    limit = locked_memory_limit()
    if limit is not None and limit < bound:
        bound, source = limit, "RLIMIT_MEMLOCK"
    return int(bound), source


def copies_directly(array: Any) -> bool:
    """Whether ``array``, on the host or on a device, is a basis that a direct copy can move.

    A Fortran-ordered matrix of doubles, which the basis of a sector is,
    whole or as the leading columns that a smaller state count leaves.
    Any other array takes the former calls, which may copy it first.
    """

    flags = getattr(array, "flags", None)
    return bool(
        flags is not None
        and getattr(array, "ndim", 0) == 2
        and array.dtype == np.float64
        and array.size > 0
        and flags.f_contiguous
    )


def _give_back(owner: Any, nbytes: int) -> None:
    """Count a page-locked buffer off its owner, if the owner still lives.

    ``owner`` is a weak reference.  A finalizer that held the owner itself
    would keep it, and through it the buffer it watches, for as long as the
    process lives.
    """

    owner = owner()
    if owner is not None:
        owner._give(nbytes)


def upload(cp: Any, host: np.ndarray) -> Any:
    """The device array that ``cupy.asarray(host, order="F")`` gives, by one copy from where ``host`` lies.

    ``host`` is as :func:`copies_directly` asks.  ``cupy.asarray`` first
    copies a host array into a block of the pinned pool; ``ndarray.set``
    hands the driver the array itself.  From a page-locked buffer under a
    stream of its own that copy is only queued when the call returns, so
    the stream is synchronized here: the buffer may be written again as
    soon as this returns.
    """

    device = cp.empty(host.shape, dtype=cp.float64, order="F")
    device.set(host)
    cp.cuda.get_current_stream().synchronize()
    return device


class SpillBuffer:
    """The host buffer of one sector whose states are kept on the host.

    It is taken when the sector is first downloaded, at the size of its
    basis then, and kept: a later download of as many columns or fewer
    writes into its leading bytes.  Only a sector that grows, as the
    bracket of the spectrum can ask in the first steps, takes a larger one.

    The host array of a download is a view of the buffer.  It holds the
    states of the sector until the next download of that sector, which
    writes over it: what the density step needs, which reads it before the
    next solve, and what the solver needs, which uploads it first.  A
    result that is read later than that is read from the device or copied
    before.

    One thread uses a buffer at a time, the one that solves its sector.
    """

    def __init__(self, owner: "SpillBuffers", sector: int, lock: bool) -> None:
        self.owner = owner
        self.sector = int(sector)
        # Whether the plan of the process lets this buffer be page-locked.
        self.lock = bool(lock)
        self._flat: np.ndarray | None = None
        self.page_locked = False
        self.allocations = 0
        self.lock_seconds = 0.0
        self.lock_failure: str | None = None
        self.uploads = 0
        self.upload_bytes = 0
        self.downloads = 0
        self.reordered_downloads = 0
        self.download_bytes = 0

    @property
    def nbytes(self) -> int:
        return 0 if self._flat is None else int(self._flat.nbytes)

    def _allocate(self, cp: Any, count: int) -> None:
        """Take a buffer of ``count`` doubles, page-locked if the plan and the bound allow."""

        nbytes = count * np.dtype(np.float64).itemsize
        # An outgrown buffer goes back first if nothing reads it any more;
        # otherwise it is counted against the bound until it does.
        self._flat = None
        flat = None
        if self.lock and self.owner._take(nbytes):
            started = perf_counter()
            try:
                # From the system and not from CuPy's pinned pool, which
                # rounds a request up to a power of two and is emptied by
                # others.
                memory = cp.cuda.PinnedMemoryPointer(cp.cuda.PinnedMemory(nbytes), 0)
                flat = np.frombuffer(memory, dtype=np.float64, count=count)
            except Exception as error:  # a pin that the driver or the system refuses
                self.owner._give(nbytes)
                self.lock_failure = f"{type(error).__name__}: {error}"
                flat = None
            else:
                # The memory is unlocked when the last view of it has gone.
                weakref.finalize(flat, _give_back, weakref.ref(self.owner), nbytes).atexit = False
            self.lock_seconds += perf_counter() - started
        self.page_locked = flat is not None
        if flat is None:
            flat = np.empty(count, dtype=np.float64)
        self._flat = flat
        self.allocations += 1

    def view(self, cp: Any, shape: tuple[int, int]) -> np.ndarray:
        """The leading bytes of the buffer as a Fortran-ordered matrix of ``shape``."""

        rows, columns = int(shape[0]), int(shape[1])
        count = rows * columns
        if self._flat is None or self._flat.size < count:
            self._allocate(cp, count)
        return self._flat[:count].reshape((rows, columns), order="F")

    def download(self, cp: Any, device: Any) -> np.ndarray:
        """Write ``device`` into the buffer and return the host array that holds it.

        ``device`` is as :func:`copies_directly` asks.  ``ndarray.get``
        with a target of the same order takes no array on the device and
        none on the host.
        """

        host = self.view(cp, device.shape)
        device.get(out=host)
        cp.cuda.get_current_stream().synchronize()
        self.downloads += 1
        self.download_bytes += int(host.nbytes)
        return host

    def store(self, cp: Any, downloaded: np.ndarray) -> np.ndarray:
        """Lay a host array that another call downloaded into the buffer, in Fortran order.

        For vectors that are no Fortran-ordered matrix on their device.
        The Ritz step of a small sector, which orthogonalizes its basis
        first, leaves them in C order; the generalized step that large
        sectors take rotates a Fortran-ordered basis in place.  The
        former call downloads such vectors, and its copy into Fortran
        order is made here into the buffer instead of into a second new
        array.  The values are the same, and the next upload reads a
        buffer like any other.
        """

        host = self.view(cp, downloaded.shape)
        host[...] = downloaded
        self.downloads += 1
        self.reordered_downloads += 1
        self.download_bytes += int(host.nbytes)
        return host

    def upload(self, cp: Any, host: np.ndarray) -> Any:
        """A new device array with the bytes of ``host``, a view of this buffer (:func:`upload`)."""

        device = upload(cp, host)
        self.uploads += 1
        self.upload_bytes += int(host.nbytes)
        return device


class SpillBuffers:
    """The spill buffers of the sectors that one process solves, and what they may page-lock.

    ``sizes`` gives, in the order of the sectors, the bytes of the basis of
    each at the state counts of the first solve.  The plan is made from
    them at once and reads the host once (:func:`page_lock_bound`): going
    through the sectors in order, a buffer is to be page-locked if it still
    fits the bound beside those before it, and is pageable otherwise.  Two
    launches of one calculation on a host with the same memory available
    therefore lock the same buffers, whichever thread downloads first.

    No memory is taken here.  A sector takes its buffer when it is first
    downloaded (:class:`SpillBuffer`), so a sector whose basis never leaves
    its devices, a shared one, never takes any.  The bound holds for what
    is locked at any moment: a page-locked buffer is counted from before it
    is taken until its memory has gone back, so that a sector which
    outgrows its buffer locks the larger one only if both fit while a
    result still reads the old one.
    """

    def __init__(self, sizes: Mapping[int, int], bound: tuple[int, str] | None = None) -> None:
        self.bound_bytes, self.bound_from = page_lock_bound() if bound is None else bound
        self.bound_bytes = int(self.bound_bytes)
        # Reentrant: a buffer can go back, and be counted off, inside a call
        # that holds the lock.
        self._ledger = threading.RLock()
        self._locked = 0
        self._buffers: dict[int, SpillBuffer] = {}
        planned = 0
        for sector, nbytes in sizes.items():
            nbytes = int(nbytes)
            lock = 0 < nbytes and planned + nbytes <= self.bound_bytes
            if lock:
                planned += nbytes
            self._buffers[int(sector)] = SpillBuffer(self, sector, lock)
        self.planned_bytes = planned

    def get(self, sector: int) -> SpillBuffer | None:
        return self._buffers.get(int(sector))

    def release(self) -> None:
        """Let every sector go of the memory of its buffer.

        Called when the solver is reset, at the start of each SCF.  A result
        of an earlier solve that still reads the memory keeps it, with its
        values, and it stays counted against the bound until that result has
        gone; the next download of the sector takes a new buffer.
        """

        for item in self._buffers.values():
            item._flat = None
            item.page_locked = False

    def _take(self, nbytes: int) -> bool:
        with self._ledger:
            if self._locked + nbytes > self.bound_bytes:
                return False
            self._locked += nbytes
            return True

    def _give(self, nbytes: int) -> None:
        with self._ledger:
            self._locked -= nbytes

    @property
    def page_locked_bytes(self) -> int:
        """Bytes that are page-locked now, outgrown buffers that are still read included."""

        return int(self._locked)

    @property
    def record(self) -> dict[str, Any]:
        """What the buffers are and what went through them, for the record of a run.

        ``sectors`` gives per sector the bytes of its buffer (0 before its
        first download), whether it is page-locked and whether the plan
        let it be.  ``lock_seconds`` are those of the page-locking, inside
        the first download of each sector, and ``lock_failures`` what the
        driver said where it refused.  ``reordered_downloads`` are those
        of the downloads that the former call made because the vectors
        were not Fortran-ordered on their device (:meth:`SpillBuffer.store`).
        The uploads counted here are those before a solve; the density
        step counts its own.
        """

        buffers = self._buffers.values()
        return dict(
            bound_bytes=self.bound_bytes,
            bound_from=self.bound_from,
            page_locked_bytes=self.page_locked_bytes,
            pageable_bytes=sum(item.nbytes for item in buffers if not item.page_locked),
            lock_seconds=sum(item.lock_seconds for item in buffers),
            lock_failures=[
                f"sector {item.sector}: {item.lock_failure}"
                for item in buffers
                if item.lock_failure is not None
            ],
            uploads=sum(item.uploads for item in buffers),
            upload_bytes=sum(item.upload_bytes for item in buffers),
            downloads=sum(item.downloads for item in buffers),
            reordered_downloads=sum(item.reordered_downloads for item in buffers),
            download_bytes=sum(item.download_bytes for item in buffers),
            sectors={
                item.sector: dict(
                    bytes=item.nbytes,
                    page_locked=item.page_locked,
                    planned="page-locked" if item.lock else "pageable",
                    allocations=item.allocations,
                )
                for item in buffers
            },
        )


__all__ = [
    "SpillBuffer",
    "SpillBuffers",
    "copies_directly",
    "host_available_bytes",
    "locked_memory_limit",
    "page_lock_bound",
    "spill_buffer_mode",
    "spill_pinned_bytes",
    "spill_pinned_pool",
    "upload",
]
