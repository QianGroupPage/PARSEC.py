"""Where the device and host memory of a run goes: a diagnostic, off unless ``PARSEC_MEMORY_CENSUS`` names a folder.

nvidia-smi sees what CuPy's pool has reserved, and the live peak of the runner (``PARSEC_DEVICE_LIVE_PEAK``)
counts the arrays of one thread.  This census counts the device arrays of every thread on every device, from
their allocation to their release, and at the largest total of each device lists the arrays that make it,
grouped by where in the code they were allocated.  It also samples the resident host memory of the process.

* ``PARSEC_MEMORY_CENSUS=<folder>`` installs it (:func:`install`, called by the runners and by
  :func:`backends.cupy.require_cupy`); each process writes ``<folder>/census_<host>_<pid>.json`` every 30 s and
  at its exit.
* It wraps the CuPy allocator, whichever is in force (the pool, or direct ``cudaMalloc`` after
  ``cupy.cuda.set_allocator(None)``, or one given to ``cupy.cuda.using_allocator``): every allocation is handed
  out as memory owned by a small tracker object, whose release (when the last array on it dies) is counted.
* Arrays of at least ``PARSEC_MEMORY_CENSUS_MIN_BYTES`` (default 1 MiB) carry the innermost frames of this
  package that allocated them (``PARSEC_MEMORY_CENSUS_FRAMES``, default 4); smaller ones are counted, not named.

The device totals are the bytes of the arrays (as the pool rounds them), without the CUDA context, library work
spaces and the blocks the pool keeps cached: what the code holds at once.  The census adds Python work to every
allocation and release; its timings are diagnostic, not production numbers.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import socket
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

_ENV = "PARSEC_MEMORY_CENSUS"
_census = None
_install_lock = threading.Lock()
_stage = threading.local()


def requested_folder() -> str | None:
    value = os.environ.get(_ENV, "").strip()
    return value or None


def mark(label: str | None) -> None:
    """Name what the calling thread is doing (a stage of the eigensolver); recorded with the peaks."""
    if _census is not None:
        _stage.label = label


def current_stage() -> str | None:
    return getattr(_stage, "label", None)


def _signature(depth: int) -> str:
    frame = sys._getframe(1)
    parts = []
    while frame is not None and len(parts) < depth:
        code = frame.f_code
        name = code.co_filename.replace("\\", "/")
        if "/parsec_python/" in name and not name.endswith("memory_census.py"):
            module = name.rsplit("/parsec_python/", 1)[1]
            parts.append(f"{module}:{code.co_name}:{frame.f_lineno}")
        frame = frame.f_back
    return " < ".join(parts) if parts else "(outside parsec_python)"


class _Tracker:
    """Owns one allocation for the arrays on it; its release is the release of the allocation."""

    __slots__ = ("pointer", "key", "census")

    def __init__(self, pointer, key, census) -> None:
        self.pointer = pointer
        self.key = key
        self.census = census

    def __del__(self) -> None:
        census = self.census
        if census is not None:
            census._free(self.key)


class _Census:
    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.threshold = int(float(os.environ.get("PARSEC_MEMORY_CENSUS_MIN_BYTES", 1 << 20)))
        self.depth = int(os.environ.get("PARSEC_MEMORY_CENSUS_FRAMES", 4))
        self.step = int(float(os.environ.get("PARSEC_MEMORY_CENSUS_STEP_BYTES", 256 << 20)))
        self.lock = threading.Lock()
        self.start = time.time()
        self.ids = itertools.count()
        self.live = defaultdict(int)
        self.small = defaultdict(int)
        self.blocks = {}   # key -> (device, size, signature, stage)
        self.peak = defaultdict(int)
        self.snapshot = {}
        self.snapshot_at = defaultdict(int)
        self.series = defaultdict(list)
        self.last_series = defaultdict(int)
        self.allocations = 0
        self.releases = 0
        self.installed_from = None
        self.host_series = []
        self.host_peak = 0
        self._stop = threading.Event()
        self.sampler = threading.Thread(target=self._sample_host, name="parsec-census-host", daemon=True)

    # ----- the allocator -----
    def wrap(self, allocator):
        if getattr(allocator, "_parsec_census", False):
            return allocator
        import cupy

        memory = cupy.cuda.memory
        if allocator is None:
            base = getattr(memory, "_malloc", None) or (lambda size: memory.MemoryPointer(memory.Memory(size), 0))
        else:
            base = allocator
        census = self

        def allocate(size):
            try:
                pointer = base(size)
            except Exception as error:
                if "out of memory" in str(error).lower() or "OutOfMemory" in type(error).__name__:
                    census._failure(size)
                raise
            mem = pointer.mem
            device = int(getattr(mem, "device_id", -1))
            if device < 0:
                device = int(cupy.cuda.runtime.getDevice())
            held = int(getattr(mem, "size", size) or size)
            key = census._malloc(device, held)
            tracker = _Tracker(pointer, key, census)
            return memory.MemoryPointer(memory.UnownedMemory(pointer.ptr, held, tracker, device), 0)

        allocate._parsec_census = True
        allocate._parsec_base = allocator
        return allocate

    def _malloc(self, device: int, size: int):
        signature = _signature(self.depth) if size >= self.threshold else None
        stage = current_stage()
        key = next(self.ids)
        with self.lock:
            self.allocations += 1
            self.live[device] += size
            if signature is None:
                self.small[device] += size
            self.blocks[key] = (device, size, signature, stage)
            live = self.live[device]
            if abs(live - self.last_series[device]) >= self.step:
                self.series[device].append((round(time.time() - self.start, 3), live))
                self.last_series[device] = live
            if live > self.peak[device]:
                self.peak[device] = live
                if live >= self.snapshot_at[device] + self.step // 4 or device not in self.snapshot:
                    self._take_snapshot(device, stage)
        return key

    def _failure(self, size: int) -> None:
        """An allocation of ``size`` bytes failed: record what every device holds now and write the census."""
        import cupy

        device = int(cupy.cuda.runtime.getDevice())
        with self.lock:
            self.failure = dict(device=device, requested_bytes=int(size), stage=current_stage(),
                                seconds=round(time.time() - self.start, 3), where=_signature(self.depth))
            for held in list(self.live):
                self._take_snapshot(held, current_stage())
        self.write(final=False)

    def _free(self, key) -> None:
        with self.lock:
            entry = self.blocks.pop(key, None)
            if entry is None:
                return
            device, size, signature, _stage_name = entry
            self.releases += 1
            self.live[device] -= size
            if signature is None:
                self.small[device] -= size
            live = self.live[device]
            if abs(live - self.last_series[device]) >= self.step:
                self.series[device].append((round(time.time() - self.start, 3), live))
                self.last_series[device] = live

    def _take_snapshot(self, device: int, stage) -> None:
        groups = defaultdict(lambda: [0, 0, None])
        for where, size, signature, at_stage in self.blocks.values():
            if where != device or signature is None:
                continue
            group = groups[signature]
            group[0] += size
            group[1] += 1
            group[2] = at_stage
        ranked = sorted(groups.items(), key=lambda item: -item[1][0])[:30]
        self.snapshot[device] = dict(
            seconds=round(time.time() - self.start, 3),
            live_bytes=int(self.live[device]),
            small_bytes=int(self.small[device]),
            stage=stage,
            groups=[dict(bytes=int(b), count=int(c), stage=s, where=sig) for sig, (b, c, s) in ranked],
        )
        self.snapshot_at[device] = int(self.live[device])

    # ----- host -----
    def _rss(self) -> int:
        try:
            with open("/proc/self/statm") as handle:
                return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, AttributeError):
            return 0

    def _sample_host(self) -> None:
        last = -1
        written = time.time()
        while not self._stop.wait(0.05):
            if time.time() - written > float(os.environ.get("PARSEC_MEMORY_CENSUS_WRITE_SECONDS", 30)):
                self.write(final=False)
                written = time.time()
            rss = self._rss()
            if rss > self.host_peak:
                self.host_peak = rss
            if last < 0 or abs(rss - last) >= (64 << 20):
                self.host_series.append((round(time.time() - self.start, 3), rss, current_stage()))
                last = rss

    def write(self, final: bool = True) -> None:
        if final:
            self._stop.set()
        hwm = 0
        try:
            with open("/proc/self/status") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        hwm = int(line.split()[1]) * 1024
        except OSError:
            pass
        rank = next((os.environ[k] for k in ("PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK") if k in os.environ), None)
        with self.lock:
            record = dict(
                host=socket.gethostname(), pid=os.getpid(), rank=rank, argv=sys.argv, final=final,
                seconds=round(time.time() - self.start, 3), allocations=self.allocations, releases=self.releases,
                min_named_bytes=self.threshold, installed_from=self.installed_from,
                failure=getattr(self, "failure", None),
                devices={
                    str(d): dict(peak_live_bytes=int(self.peak[d]), final_live_bytes=int(self.live[d]),
                                 at_peak=self.snapshot.get(d), series=list(self.series.get(d, [])))
                    for d in sorted(self.peak)
                },
                host_memory=dict(rss_peak_sampled_bytes=int(self.host_peak), vm_hwm_bytes=int(hwm),
                                 series=list(self.host_series)),
                environment={k: v for k, v in os.environ.items() if k.startswith(("PARSEC_", "CUPY_", "OMP_"))},
            )
        try:
            Path(self.folder).mkdir(parents=True, exist_ok=True)
            path = Path(self.folder) / f"census_{socket.gethostname()}_{os.getpid()}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=1))
            os.replace(temporary, path)
        except OSError as error:
            print(f"memory census: could not write {self.folder}: {error}", file=sys.stderr)


def install() -> bool:
    """Start the census if ``PARSEC_MEMORY_CENSUS`` names a folder; idempotent.  Returns whether it runs."""
    global _census
    folder = requested_folder()
    if folder is None:
        return False
    with _install_lock:
        if _census is not None:
            return True
        import contextlib

        import cupy
        import cupy.cuda.memory as memory

        census = _Census(folder)
        census.installed_from = threading.current_thread().name
        original_set = memory.set_allocator
        original_using = cupy.cuda.using_allocator

        def set_allocator(allocator=None):
            return original_set(census.wrap(allocator))

        @contextlib.contextmanager
        def using_allocator(allocator=None):
            with original_using(census.wrap(allocator)):
                yield

        for module in (memory, cupy.cuda):
            if hasattr(module, "set_allocator"):
                module.set_allocator = set_allocator
        cupy.cuda.using_allocator = using_allocator
        original_set(census.wrap(memory.get_allocator()))
        census.sampler.start()
        atexit.register(census.write)
        _census = census
    return True


__all__ = ["install", "mark", "requested_folder", "current_stage"]
