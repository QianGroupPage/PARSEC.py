"""Exact exchange of a hybrid functional of a periodic cell on the devices of one process.

The physics and the conventions are those of the host module
:mod:`parsec_python.V_xc.bloch_exchange` (units Ry and bohr; columns
``q = sqrt(dV) u`` of the Bloch factors ``u_nk``, ``sum_p |q|^2 = 1`` over
the grid of the cell; occupations ``f`` in ``[0, 1]`` per spin; spin weight
``s`` 2 for an unpolarized set and 1 per channel).  The k-points of a run are
the representatives ``t`` of the Monkhorst-Pack mesh under time reversal
(weight ``w_t = 2/N`` when ``-t`` is another mesh point, its *partner*, and
``1/N`` otherwise); the exchange sums over the whole mesh of ``N`` points,
``{t} U {-t}``, with ``u_(-t) = conj(u_t)``:

``W_t[:, n] = -(1/N) sum_(k' in mesh) sum_(m occ) f_mk' u_mk' w_(nt,mk')``,
``w = ifftn(K(k_t - k' + G) fftn(conj(u_mk') u_nt / dV))``,

``K = kernel.values(q)`` on the FFT box of the grid (a slab's box padded at
the end of its open axis: the pair density is zero-padded and the potential
cropped), numpy's FFT conventions (``ifftn`` with ``1/N_box``).  Then

``E_x = (s/2) sum_t w_t sum_n f_nt Re <u_nt | W_t[:, n]>``,
``M_t = -U_t^H W_t = L L^H``,  ``xi_t = W_t L^-H``,  ``a_x V_ACE,t = -a_x xi_t xi_t^H``

(adaptively compressed exchange per k-point: Lin, J. Chem. Theory Comput.
12, 2242 (2016); the k-point form as in Lin's paper, Sec. 3; no code of
another program is used).

What this module adds is where the arrays are.  The orbitals of every
k-point stay on the device of the eigensolver worker that owns the k-point
(:class:`parsec_python.acceleration.SCF.kpoint_parallel.PooledKPointSolver`);
``W_t`` and ``xi_t`` are formed there.  Every device of the process receives
one copy of the occupied columns of every representative (rows of a
C-contiguous ``(m, N)`` block); a *tile* is a block of target columns of one
representative, solved against every source of the mesh; tiles go to the
devices, the most expensive first, each to the least loaded device (its
owner on a tie), and a finished tile's columns of ``W_t`` are copied into
place on the owner's device.  A chunk of pair densities (1 GiB of them, at
most 5 % of the device, at least ``Exx_Batch_Size``: :data:`CHUNK_BYTES`) is
formed by one kernel straight into the FFT box (padding and the index box of
a confined slab grid included), transformed in place by cuFFT (a batch of a
few sizes only, so that the plans cached per thread are reused), multiplied
by ``K(q+G)`` (cached per device and per ``q``, :class:`DeviceKernelCache`),
transformed back, and summed into ``W`` by a second kernel that reads every
potential once.  Each device works on a thread of its own kept for the run.

A pair of two occupied states is solved once for both (``PARSEC_CUPY_BLOCH_PAIR_SYMMETRY``,
on by default).  With ``F(rho, q) = ifftn(K(q+G) fftn(rho))`` and a kernel
whose values satisfy ``K(-q)[-G] = K(q)[G]`` (the symmetric Nyquist planes of
the exchange kernel), ``F(conj(rho), -q) = conj(F(rho, q))``: the pair
``(n t | m k')`` of a source ``k'`` that is a representative ``r`` has the
partner ``(m r | n t)`` with the potential ``conj(w)``, which adds
``-(f_nt/N) u_nt conj(w)`` to ``W_r[:, m]``; the pair of a source ``-r`` (the
partner of ``r``) has the partner ``(m r | n -t)`` of the same density and
``q``, which adds ``-(f_nt/N) conj(u_nt) w``, where ``-t`` is a point of the
mesh (``t`` has a partner).  A tile of ``t`` solves its pairs with the
sources of ``r > t`` and the triangle ``m >= n`` of ``r = t`` and adds both
sides; the pairs with ``r < t`` come from the tiles of ``r``.  A third kernel
adds the partner sides to a ``W`` of the occupied states kept on the device,
delivered to the owners at the end.

The Gamma point of a cell (real orbitals) takes the device loop of
isolated systems (:func:`parsec_python.acceleration.Hamiltonian.exchange_device.apply_device_exchange`)
with :class:`DevicePeriodicPairSolvers` in place of the Kronecker solver:
real-to-complex FFTs of the real pair densities times ``K(G)``.

Nothing here prints or opens files.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from parsec_python.V_xc.exact_exchange import OCCUPATION_THRESHOLD


# ---------------------------------------------------------------------------
# The FFT box of the grid.
# ---------------------------------------------------------------------------


def _shape_of(values) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


class DevicePeriodicBox:
    """Where the grid points sit in the FFT box of the exchange kernel.

    ``box_shape`` is the index box of the grid (``(n0, n1, n2)``, row-major),
    ``fft_shape`` the kernel's FFT box (equal, or larger along an open axis:
    the box is then padded at the end of that axis), ``flat`` the row-major
    index of every grid point in the index box (``None``: the grid fills the
    box in order).  ``box_index`` is the row-major index of every grid point
    in the FFT box, ``None`` when that is the identity (a bulk cell).
    """

    def __init__(self, box_shape, fft_shape, flat=None) -> None:
        self.box_shape = _shape_of(box_shape)
        self.fft_shape = _shape_of(fft_shape)
        if len(self.box_shape) != 3 or len(self.fft_shape) != 3:
            raise ValueError("the box and the FFT box of the exchange kernel are three-dimensional")
        if any(fft < box for fft, box in zip(self.fft_shape, self.box_shape)):
            raise ValueError("the FFT box of the kernel cannot be smaller than the box of the grid")
        self.fft_size = int(np.prod(self.fft_shape))
        if flat is None:
            flat = np.arange(int(np.prod(self.box_shape)), dtype=np.int64)
        flat = np.asarray(flat, dtype=np.int64).reshape(-1)
        self.points = int(flat.size)
        coordinates = np.unravel_index(flat, self.box_shape)
        index = np.ravel_multi_index(coordinates, self.fft_shape).astype(np.int64)
        identity = self.fft_shape == self.box_shape and bool(
            np.array_equal(index, np.arange(self.points, dtype=np.int64))
        )
        self.box_index = None if identity else index
        self._device: dict[int, Any] = {}
        self._lock = Lock()

    @classmethod
    def of(cls, box, kernel) -> "DevicePeriodicBox":
        """The box of a host ``GridBox`` (``shape``, ``flat``) and a kernel (``shape``)."""

        return cls(box.shape, kernel.shape, getattr(box, "flat", None))

    @property
    def identity(self) -> bool:
        return self.box_index is None

    def device_index(self, cp):
        """``box_index`` on the current device (``None`` for the identity)."""

        if self.box_index is None:
            return None
        device = int(cp.cuda.Device().id)
        with self._lock:
            found = self._device.get(device)
            if found is None:
                found = cp.asarray(self.box_index)
                self._device[device] = found
        return found

    def to_box(self, xp, rows, *, index=None):
        """``(nb, N)`` rows of grid values as ``(nb, *fft_shape)`` (zero outside the grid)."""

        count = int(rows.shape[0])
        if self.box_index is None:
            return xp.ascontiguousarray(rows).reshape((count,) + self.fft_shape)
        if index is None:
            index = self.box_index if xp is np else self.device_index(xp)
        box = xp.zeros((count, self.fft_size), dtype=rows.dtype)
        box[:, index] = rows
        return box.reshape((count,) + self.fft_shape)

    def from_box(self, xp, box, *, index=None):
        """``(nb, N)`` grid values of ``(nb, *fft_shape)`` box values."""

        count = int(box.shape[0])
        flat = box.reshape(count, self.fft_size)
        if self.box_index is None:
            return flat
        if index is None:
            index = self.box_index if xp is np else self.device_index(xp)
        return flat[:, index]


# ---------------------------------------------------------------------------
# K(q + G) on the devices.
# ---------------------------------------------------------------------------


def _q_key(q) -> tuple[float, ...]:
    """The cache key of a wave vector (Cartesian bohr^-1), rounded far below any mesh spacing."""

    return tuple(float(value) for value in np.round(np.asarray(q, dtype=np.float64).reshape(3), 12))


#: Bytes of the pair densities of one transform of the k-point loop at least (``chunk_pairs``), and at most
#: CHUNK_SHARE of the device's memory.  A100, 2026-11-07, 64-atom Si sizes (68^3, 2x2x2, 128 of 140 states) on
#: four GPUs, a warm build: 31.1 s with 512 MiB (106 pairs), 26.5 s with 1 GiB (213), 30.0 s with 2 GiB; with
#: Exx_Batch_Size 32 alone the threads of the four devices waited on each other's Python calls.
CHUNK_BYTES = 1024 * 1024 * 1024
CHUNK_SHARE = 0.05
#: Share of a device's memory the cached kernel arrays may take (at least one array is kept).
KERNEL_CACHE_FRACTION = 0.05


def _fused_builder():
    """``device_values(kernel, q)`` of the fused device builder of the exchange kernel, where it exists."""

    try:
        from ..Hartree.exchange_kernel_device import device_values
    except ImportError:
        return None
    return device_values


def _fused_kernel(kernel) -> bool:
    """Whether ``kernel`` is the exchange kernel of the code (the fused builder knows it), not a stand-in."""

    try:
        from parsec_python.Hartree.exchange_kernel import PeriodicExchangeKernel
    except ImportError:
        return False
    return isinstance(kernel, PeriodicExchangeKernel)


class DeviceKernelCache:
    """``K(q+G)`` of an exchange kernel on every device, cached per ``q``.

    ``kernel.values(q, xp=cupy)`` builds the array on the current device
    (float64, ``kernel.shape``); a kernel whose ``values`` takes no ``xp``
    is evaluated on the host and uploaded.  Each device keeps the arrays of
    the latest ``q`` up to :data:`KERNEL_CACHE_FRACTION` of its memory (the
    least recently used go first); ``half(cp, device)`` is the half spectrum
    of ``q = 0`` for real-to-complex transforms (``K(-G) = K(G)`` there).
    """

    def __init__(self, kernel, *, budget_bytes: int | None = None) -> None:
        self.kernel = kernel
        self.shape = _shape_of(kernel.shape)
        self.array_bytes = 8 * int(np.prod(self.shape))
        self.budget_bytes = budget_bytes
        self._arrays: dict[int, OrderedDict] = {}
        self._half: dict[int, Any] = {}
        self._lock = Lock()
        self._host_lock = Lock()
        self.built = 0
        self.build_seconds = 0.0
        try:
            import inspect

            self._takes_xp = "xp" in inspect.signature(kernel.values).parameters
        except (TypeError, ValueError):
            self._takes_xp = False

    def _budget(self, cp) -> int:
        if self.budget_bytes is not None:
            return int(self.budget_bytes)
        _free, total = cp.cuda.runtime.memGetInfo()
        return int(KERNEL_CACHE_FRACTION * float(total))

    def _build(self, cp, q):
        started = perf_counter()
        with self._host_lock:
            fused = _fused_builder()
            if fused is not None and _fused_kernel(self.kernel):
                # One fused device kernel per q (acceleration/Hartree/exchange_kernel_device.py), equal to values(q)
                values = fused(self.kernel, np.asarray(q, dtype=np.float64))
            elif self._takes_xp:
                values = self.kernel.values(np.asarray(q, dtype=np.float64), xp=cp)
            else:
                values = self.kernel.values(np.asarray(q, dtype=np.float64))
        values = cp.ascontiguousarray(cp.asarray(values, dtype=cp.float64))
        if values.shape != self.shape:
            raise ValueError("the exchange kernel returned an array of another shape than kernel.shape")
        self.built += 1
        self.build_seconds += perf_counter() - started
        return values

    def values(self, cp, q):
        """``K(q+G)`` on the current device."""

        device = int(cp.cuda.Device().id)
        key = _q_key(q)
        with self._lock:
            arrays = self._arrays.setdefault(device, OrderedDict())
            found = arrays.get(key)
            if found is not None:
                arrays.move_to_end(key)
                return found
        values = self._build(cp, q)
        with self._lock:
            arrays = self._arrays.setdefault(device, OrderedDict())
            arrays[key] = values
            limit = max(1, self._budget(cp) // max(1, self.array_bytes))
            while len(arrays) > limit:
                arrays.popitem(last=False)
        return values

    def half(self, cp):
        """``K(G)`` of ``q = 0`` on the half spectrum of a real-to-complex FFT (current device)."""

        device = int(cp.cuda.Device().id)
        with self._lock:
            found = self._half.get(device)
        if found is not None:
            return found
        full = self._build(cp, np.zeros(3))
        half = cp.ascontiguousarray(full[..., : self.shape[-1] // 2 + 1])
        with self._lock:
            self._half[device] = half
        return half

    def device_bytes(self, cp) -> int:
        """Bytes the cache holds on the current device."""

        device = int(cp.cuda.Device().id)
        with self._lock:
            count = len(self._arrays.get(device, ()))
            half = self._half.get(device)
        return count * self.array_bytes + (0 if half is None else int(half.nbytes))

    def release(self) -> None:
        with self._lock:
            self._arrays.clear()
            self._half.clear()


# ---------------------------------------------------------------------------
# The pair solvers of one run.
# ---------------------------------------------------------------------------


_KERNEL_SOURCE = r"""
#include <cupy/complex.cuh>

// rho[i * ns + j] = conj(u_s,j) u_t,i * scale on the FFT box, where u_s = s (or conj(s) for the
// partner -k of a representative, conjugate = 1); points outside the grid are left as they are.
extern "C" __global__ void bloch_pair_densities(
    const complex<double>* __restrict__ targets,   // (nt, n) rows
    const complex<double>* __restrict__ sources,   // (ns, n) rows
    const int conjugate,
    const double scale,
    const long long* __restrict__ box_index,       // (n) or null (identity)
    complex<double>* __restrict__ box,             // (nt * ns, box_size)
    const long long n,
    const long long box_size,
    const int ns)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int pair = blockIdx.y;
    const int i = pair / ns;
    const int j = pair - i * ns;
    const complex<double> s = sources[(long long)j * n + p];
    const complex<double> cs = conjugate ? s : conj(s);
    const long long q = box_index ? box_index[p] : p;
    box[(long long)pair * box_size + q] = cs * targets[(long long)i * n + p] * scale;
}

// applied[i] -= sum_j c_j u_s,j w_(i, j), every point summing its ns terms in order.
extern "C" __global__ void bloch_pair_accumulate(
    const complex<double>* __restrict__ box,       // (nt * ns, box_size) potentials
    const complex<double>* __restrict__ sources,   // (ns, n) rows
    const double* __restrict__ coefficients,       // (ns) f / N
    const int conjugate,
    const long long* __restrict__ box_index,
    complex<double>* __restrict__ applied,         // (nt, n) rows
    const long long n,
    const long long box_size,
    const int ns)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int i = blockIdx.y;
    const long long q = box_index ? box_index[p] : p;
    complex<double> sum(0.0, 0.0);
    for (int j = 0; j < ns; ++j) {
        const complex<double> s = sources[(long long)j * n + p];
        const complex<double> us = conjugate ? conj(s) : s;
        sum += coefficients[j] * (us * box[((long long)i * ns + j) * box_size + q]);
    }
    applied[(long long)i * n + p] -= sum;
}

// The partner of every pair, the same potential read the other way:
// partner[j] -= sum_i d_i (u_t,i conj(w_(i, j))) for a source of the mesh point itself (w of the partner pair
// is conj(w)), or sum_i d_i (conj(u_t,i) w_(i, j)) for the partner -k of a representative (the same w);
// column skip (the pair of a state with itself) is left out.
extern "C" __global__ void bloch_pair_partner(
    const complex<double>* __restrict__ box,       // (nt * ns, box_size) potentials
    const complex<double>* __restrict__ targets,   // (nt, n) rows
    const double* __restrict__ coefficients,       // (nt) f_t / N
    const int conjugate,
    const long long* __restrict__ box_index,
    complex<double>* __restrict__ partner,         // (ns, n) rows
    const long long n,
    const long long box_size,
    const int nt,
    const int ns,
    const int skip)
{
    const long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int j = blockIdx.y;
    if (j == skip) return;
    const long long q = box_index ? box_index[p] : p;
    complex<double> sum(0.0, 0.0);
    for (int i = 0; i < nt; ++i) {
        const complex<double> t = targets[(long long)i * n + p];
        const complex<double> w = box[((long long)i * ns + j) * box_size + q];
        sum += coefficients[i] * (conjugate ? conj(t) * w : t * conj(w));
    }
    partner[(long long)j * n + p] -= sum;
}
"""
_KERNELS: dict[int, tuple[Any, Any, Any]] = {}
_KERNELS_LOCK = Lock()
_THREADS = 256


def _kernels(cp):
    device = int(cp.cuda.Device().id)
    with _KERNELS_LOCK:
        found = _KERNELS.get(device)
        if found is None:
            module = cp.RawModule(code=_KERNEL_SOURCE, options=("--fmad=false",))
            found = (
                module.get_function("bloch_pair_densities"),
                module.get_function("bloch_pair_accumulate"),
                module.get_function("bloch_pair_partner"),
            )
            _KERNELS[device] = found
    return found


def _shutdown(executors: dict) -> None:
    for executor in executors.values():
        executor.shutdown(wait=False)
    executors.clear()


class DevicePeriodicPairSolvers:
    """The pair solves of a periodic cell's exchange on the devices of the process.

    ``kernel`` is the host kernel (:class:`parsec_python.Hartree.exchange_kernel.PeriodicExchangeKernel`
    or any object with ``values(q[, xp])`` and ``shape``), ``box`` the grid's
    box (``shape`` and the optional index map ``flat``), ``batch_size``
    (``Exx_Batch_Size``) the pair densities of one transform.  The kernel's
    weights are its own: a one-term functional's ``a_x`` scales the operator.

    For the device loop of real orbitals (the Gamma point,
    :func:`parsec_python.acceleration.Hamiltonian.exchange_device.apply_device_exchange`)
    the object has the interface of the Kronecker solver's
    :class:`~parsec_python.acceleration.Hamiltonian.exchange_device.DevicePairSolvers`:
    :meth:`solve` takes real pair densities ``(N, nb)`` (F order) and returns
    their potentials.  :meth:`accumulate_bloch` is the complex chunk of the
    k-point loop.
    """

    def __init__(self, kernel, box, batch_size: int, *, description: str = "") -> None:
        self.kernel = kernel
        self.box = box if isinstance(box, DevicePeriodicBox) else DevicePeriodicBox.of(box, kernel)
        if self.box.fft_shape != _shape_of(kernel.shape):
            raise ValueError("the FFT box of the grid box and the kernel's shape differ")
        self.batch_size = max(1, int(batch_size))
        # Pair densities of one transform of the k-point loop: Exx_Batch_Size, or more where the box is small
        # enough (CHUNK_BYTES of densities): every transform costs a few calls of Python, which the threads of
        # several devices take in turn, so fewer and larger transforms keep the devices busy.
        self.chunk_pairs = max(self.batch_size, min(4096, int(CHUNK_BYTES // (16 * self.box.fft_size))))
        self.cache = DeviceKernelCache(kernel)
        self.setup_seconds = 0.0
        self._streams: dict = {}
        self._buffers: dict = {}
        self._lock = Lock()
        self._label = description
        self._paths: set[str] = set()
        # One thread per device, kept for the run: cuFFT plans are cached per thread, so the builds after the
        # first make none (a pool made per build made them again in every build).
        self._executors: dict[int, ThreadPoolExecutor] = {}
        self._chunks: dict[int, int] = {}
        import weakref

        weakref.finalize(self, _shutdown, self._executors)

    def small_build(self, orbitals, *, columns: str = "all", known=None) -> bool:
        """Whether a build of real orbitals (``ChannelOrbitals`` of the device loop) is below
        :data:`SMALL_BUILD_POINTS`: its pairs (each pair of two occupied states once) times the FFT box."""

        occupied = int(orbitals.occupied_states().size)
        empty = int(orbitals.state_count) - occupied
        pairs = 0 if known is not None else occupied * (occupied + 1) // 2
        if columns == "all" or known is not None:
            pairs += empty * occupied
        return len(orbitals.sectors) == 1 and pairs * float(self.box.fft_size) < SMALL_BUILD_POINTS

    def run_on_devices(self, run, devices) -> None:
        """``run(device)`` for every device, on the device's own thread (the calling thread for one device)."""

        devices = tuple(int(device) for device in devices)
        if len(devices) == 1:
            run(devices[0])
            return
        futures = []
        for device in devices:
            with self._lock:
                executor = self._executors.get(device)
                if executor is None:
                    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"parsec-exx-gpu{device}")
                    self._executors[device] = executor
            futures.append(executor.submit(run, device))
        for future in futures:
            future.result()

    # ----- the interface of the Kronecker solvers (real orbitals) -------------

    @property
    def size(self) -> int:
        return int(self.box.points)

    @property
    def description(self) -> str:
        shape = "x".join(str(value) for value in self.box.fft_shape)
        text = f"FP64 cuFFT pair solver of the periodic exchange kernel on the {shape} FFT box"
        if self.box.fft_shape != self.box.box_shape:
            text += " (the pair densities zero-padded along the open axis)"
        if self._label:
            text += f"; {self._label}"
        return text + f"; devices take tiles of {self.batch_size}-column orbital blocks"

    def record_path(self, path: str) -> None:
        self._paths.add(str(path))

    def parity_layout(self):
        """No reflection-parity classes: a periodic cell's orbitals are on the full grid."""

        return None

    def plan_bytes(self) -> int:
        return int(self.cache.array_bytes)

    def stream(self, device: int, purpose: str = "work"):
        """A persistent non-blocking stream of ``device`` for the exchange (``work`` or ``delivery``)."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        key = (int(device), purpose)
        with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                with cp.cuda.Device(int(device)):
                    stream = cp.cuda.Stream(non_blocking=True)
                self._streams[key] = stream
        return stream

    def release_blocks(self) -> None:
        """Return the pool blocks the exchange streams freed, for the eigensolvers."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        with self._lock:
            streams = list(self._streams.items())
            self._buffers.clear()
        for (device, _purpose), stream in streams:
            with cp.cuda.Device(device):
                stream.synchronize()
                cp.get_default_memory_pool().free_all_blocks(stream=stream)

    def solve(self, cp, device: int, densities):
        """Potentials ``(N, nb)`` (Ry, F order) of real pair densities ``(N, nb)`` (e/bohr^3) on ``device``.

        ``irfftn(K(G) rfftn(rho))`` over the FFT box: the real part of the
        complex transform pair of the host solver, since ``K(-G) = K(G)``.
        """

        import cupyx.scipy.fft as cufft

        rows = densities.T  # (nb, N): C-contiguous for an F-ordered block
        count = int(rows.shape[0])
        box = self.box.to_box(cp, cp.ascontiguousarray(rows))
        axes = (1, 2, 3)
        spectrum = cufft.rfftn(box, axes=axes, overwrite_x=not self.box.identity)
        del box
        spectrum *= self.cache.half(cp)[None]
        values = cufft.irfftn(spectrum, s=self.box.fft_shape, axes=axes, overwrite_x=True)
        del spectrum
        potentials = self.box.from_box(cp, values)
        return cp.asfortranarray(potentials.reshape(count, self.box.points).T)

    # ----- the complex chunk of the k-point loop ------------------------------

    def device_chunk_pairs(self, cp) -> int:
        """``chunk_pairs`` on the current device: at most :data:`CHUNK_SHARE` of its memory (at least Exx_Batch_Size)."""

        device = int(cp.cuda.Device().id)
        with self._lock:
            found = self._chunks.get(device)
        if found is None:
            _free, total = cp.cuda.runtime.memGetInfo()
            limit = int(CHUNK_SHARE * float(total) // (16 * self.box.fft_size))
            found = max(self.batch_size, min(self.chunk_pairs, limit))
            with self._lock:
                self._chunks[device] = found
        return found

    def _buffer(self, cp, pairs: int):
        """The chunk's FFT box ``(pairs, fft_size)`` complex of the current device (reused)."""

        device = int(cp.cuda.Device().id)
        with self._lock:
            found = self._buffers.get(device)
        if found is None or found.shape[0] < pairs:
            found = cp.empty((max(pairs, self.device_chunk_pairs(cp)), self.box.fft_size), dtype=cp.complex128)
            with self._lock:
                self._buffers[device] = found
        return found

    def accumulate_bloch(self, cp, q, targets, sources, coefficients, applied, *, conjugate: bool,
                         inverse_volume: float, timing: list | None = None, partner=None,
                         target_coefficients=None, skip_self: bool = False) -> int:
        """``applied -= sum_j c_j u_s,j w(conj(u_s,j) u_t,i / dV; q)`` for every pair of rows.

        ``targets`` ``(nt, N)`` and ``applied`` ``(nt, N)`` are C-contiguous
        complex rows on the current device, ``sources`` ``(ns, N)`` the rows
        of a representative (``u_s = conj(source)`` with ``conjugate``: the
        partner ``-k``), ``coefficients`` ``(ns,)`` device ``f/N``.  The
        pairs go ``chunk_pairs`` at a time through the FFT box.  Returns the
        pair solves.  ``timing`` collects ``(begin, end)`` CUDA events of the
        transforms.

        ``partner`` ``(ns, N)`` (rows of ``W`` of the source's occupied
        states) also takes the partner of every pair, solved by the same
        transform (module text): ``partner_j -= sum_i d_i u_t,i conj(w_ij)``,
        or ``d_i conj(u_t,i) w_ij`` with ``conjugate``; ``d`` are the
        ``target_coefficients`` ``(nt,)`` ``f_t/N``.  ``skip_self`` (one target
        row) leaves out the partner of the pair with source row 0: the pair of
        a state with itself.
        """

        import cupyx.scipy.fft as cufft

        densities, accumulate, accumulate_partner = _kernels(cp)
        n = int(self.box.points)
        nt, ns = int(targets.shape[0]), int(sources.shape[0])
        if nt == 0 or ns == 0:
            return 0
        checked = (targets, sources, applied) + (() if partner is None else (partner,))
        for array in checked:
            if not array.flags.c_contiguous or int(array.shape[1]) != n:
                raise ValueError("the rows of the Bloch pair solve must be C-contiguous (rows, grid)")
        if skip_self and nt != 1:
            raise ValueError("skip_self is for one target row")
        values = self.cache.values(cp, q)
        index = self.box.device_index(cp)
        index_pointer = index if index is not None else np.uintp(0)
        chunk = self.device_chunk_pairs(cp)
        column_step = min(ns, chunk)
        row_step = max(1, chunk // column_step)
        box = self._buffer(cp, row_step * column_step)
        blocks = (n + _THREADS - 1) // _THREADS
        axes = (1, 2, 3)
        solves = 0
        for j0 in range(0, ns, column_step):
            j1 = min(ns, j0 + column_step)
            width = j1 - j0
            source_rows = sources[j0:j1]
            weights = coefficients[j0:j1]
            for i0 in range(0, nt, row_step):
                i1 = min(nt, i0 + row_step)
                pairs = (i1 - i0) * width
                chunk = box[:pairs]
                if index is not None:
                    chunk.fill(0)
                densities(
                    (blocks, pairs),
                    (_THREADS,),
                    (
                        targets[i0:i1],
                        source_rows,
                        np.int32(1 if conjugate else 0),
                        np.float64(inverse_volume),
                        index_pointer,
                        chunk,
                        np.int64(n),
                        np.int64(self.box.fft_size),
                        np.int32(width),
                    ),
                )
                # The transform takes a batch of a few sizes only (the powers of two below the chunk, and the
                # chunk): cuFFT plans are made per batch size and cached (CuPy keeps 16 per thread and device),
                # and the rows after ``pairs`` are not read.  The pairs of one state with the states after it
                # (one row per call) would otherwise need a plan of every length.
                batch = min(int(box.shape[0]), 1 << max(0, pairs - 1).bit_length())
                shaped = box[:batch].reshape((batch,) + self.box.fft_shape)
                begin = cp.cuda.Event() if timing is not None else None
                if begin is not None:
                    begin.record()
                transformed = cufft.fftn(shaped, axes=axes, overwrite_x=True)
                transformed *= values[None]
                transformed = cufft.ifftn(transformed, axes=axes, overwrite_x=True)
                if timing is not None:
                    end = cp.cuda.Event()
                    end.record()
                    timing.append((begin, end))
                if transformed.data.ptr != chunk.data.ptr:
                    chunk[...] = transformed.reshape(batch, self.box.fft_size)[:pairs]
                del transformed
                accumulate(
                    (blocks, i1 - i0),
                    (_THREADS,),
                    (
                        chunk,
                        source_rows,
                        weights,
                        np.int32(1 if conjugate else 0),
                        index_pointer,
                        applied[i0:i1],
                        np.int64(n),
                        np.int64(self.box.fft_size),
                        np.int32(width),
                    ),
                )
                if partner is not None:
                    accumulate_partner(
                        (blocks, width),
                        (_THREADS,),
                        (
                            chunk,
                            targets[i0:i1],
                            target_coefficients[i0:i1],
                            np.int32(1 if conjugate else 0),
                            index_pointer,
                            partner[j0:j1],
                            np.int64(n),
                            np.int64(self.box.fft_size),
                            np.int32(i1 - i0),
                            np.int32(width),
                            np.int32(0 if (skip_self and j0 == 0) else -1),
                        ),
                    )
                solves += pairs
        return solves

    def tile_bytes(self, columns: int) -> int:
        """Device bytes of one tile beside the source copies: the target rows, their ``W`` and the chunk box."""

        return 16 * (2 * int(columns) * int(self.box.points) + 2 * self.chunk_pairs * self.box.fft_size)


# ---------------------------------------------------------------------------
# The Bloch orbitals of one channel.
# ---------------------------------------------------------------------------


@dataclass
class DeviceBlochChannel:
    """The Bloch factors of one spin channel (``spin_weight`` 1) or of both (2), by representative.

    ``orbitals[t]`` is ``(N, S_t)`` complex (q-normalized columns) on the
    device ``devices[t]`` (its eigensolver worker's) or on the host;
    ``partners[t]`` says whether ``-k_t`` is another mesh point (then
    ``u_(-t) = conj(u_t)`` is a source too).  ``mesh_size`` is ``N``.
    """

    k_points: np.ndarray
    weights: np.ndarray
    orbitals: tuple
    occupations: tuple
    spin_weight: float
    partners: np.ndarray
    mesh_size: int
    volume_element: float
    devices: tuple

    @property
    def representatives(self) -> int:
        return len(self.orbitals)

    def occupied(self, rep: int) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.occupations[rep]) > OCCUPATION_THRESHOLD)

    def sources(self):
        """``(rep, k', conjugate)`` of every point of the mesh: the representative, then its partner."""

        found = []
        for rep in range(self.representatives):
            found.append((rep, np.asarray(self.k_points[rep], dtype=np.float64), False))
            if bool(self.partners[rep]):
                found.append((rep, -np.asarray(self.k_points[rep], dtype=np.float64), True))
        return found


def device_bloch_channel(orbital_set, volume_element: float) -> DeviceBlochChannel:
    """:class:`DeviceBlochChannel` of a ``BlochOrbitalSet`` (:mod:`parsec_python.V_xc.bloch_exchange`).

    The set's orbitals are device arrays where the k-point workers keep them
    (or host arrays); nothing is copied here.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    k_points = np.asarray(orbital_set.k_points, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(orbital_set.weights, dtype=np.float64).reshape(-1)
    partners = np.asarray(orbital_set.partners, dtype=bool).reshape(-1)
    orbitals = tuple(orbital_set.orbitals)
    occupations = tuple(np.asarray(values, dtype=np.float64).reshape(-1) for values in orbital_set.occupations)
    count = len(orbitals)
    if not (k_points.shape[0] == weights.size == partners.size == len(occupations) == count):
        raise ValueError("one k-point, weight, partner flag and occupation set per representative is needed")
    mesh_size = int(orbital_set.mesh_size)
    if mesh_size != int(count + np.count_nonzero(partners)):
        raise ValueError("the representatives and their partners do not make up the mesh")
    devices = []
    for vectors, values in zip(orbitals, occupations):
        if getattr(vectors, "ndim", 0) != 2 or int(vectors.shape[1]) != values.size:
            raise ValueError("each representative needs a (grid, states) block with one occupation per state")
        devices.append(int(vectors.device.id) if isinstance(vectors, cp.ndarray) else int(cp.cuda.Device().id))
    return DeviceBlochChannel(
        k_points=k_points,
        weights=weights,
        orbitals=orbitals,
        occupations=occupations,
        spin_weight=float(orbital_set.spin_weight),
        partners=partners,
        mesh_size=mesh_size,
        volume_element=float(volume_element),
        devices=tuple(devices),
    )


# ---------------------------------------------------------------------------
# Copies between devices.
# ---------------------------------------------------------------------------


def _rows_on(cp, vectors, positions: np.ndarray, device: int):
    """The columns ``positions`` of a ``(N, S)`` block as C-contiguous rows ``(len, N)`` on ``device``.

    A host block is uploaded; a block of another device is gathered there
    and copied peer to peer (the caller's current stream on ``device``).
    """

    positions = np.asarray(positions, dtype=np.int64)
    with cp.cuda.Device(int(device)):
        if not isinstance(vectors, cp.ndarray):
            return cp.asarray(np.ascontiguousarray(np.asarray(vectors)[:, positions].T), dtype=cp.complex128)
        source = int(vectors.device.id)
        if source == int(device):
            return _gathered_rows(cp, vectors, positions)
        with cp.cuda.Device(source):
            rows = _gathered_rows(cp, vectors, positions)
            cp.cuda.get_current_stream().synchronize()
        copy = cp.empty(rows.shape, dtype=cp.complex128)
        copy.data.copy_from_device_async(rows.data, rows.nbytes)
        cp.cuda.get_current_stream().synchronize()
        return copy


def _gathered_rows(cp, vectors, positions: np.ndarray):
    contiguous = positions.size and bool(np.all(np.diff(positions) == 1))
    if contiguous:
        view = vectors[:, int(positions[0]) : int(positions[-1]) + 1]
    else:
        view = vectors[:, cp.asarray(positions)]
    return cp.ascontiguousarray(view.T, dtype=cp.complex128)


# ---------------------------------------------------------------------------
# W = V_x U of every representative.
# ---------------------------------------------------------------------------


@dataclass
class DeviceBlochExchangeApplication:
    """``W_t = V_x U_t`` of every representative of one channel and what it took.

    ``applied[t]`` is ``(N, S_t)`` complex, F order, on ``channel.devices[t]``
    (zero columns for the states not in ``columns[t]``).  ``energy`` is
    ``E_x`` of the channel (Ry, unscaled).
    """

    channel: DeviceBlochChannel
    applied: list
    columns: tuple
    occupied: tuple
    energy: float
    pair_solves: int
    poisson_seconds: float
    seconds: float


@dataclass
class _Tile:
    rep: int
    columns: np.ndarray
    cost: int
    # A block of occupied target columns: ``start`` is its first position in the occupied list of ``rep``.
    start: int | None = None


#: Columns of one tile.
TILE_COLUMNS = 32
#: A build of fewer pair-density points than this (pair solves times FFT box points) runs on one device,
#: the one that holds the most k-points: on several it is bound by kernel launches and the threads that
#: issue them.  A100, 2026-11-07, warm builds with one thread per device kept for the run, one GPU against
#: four: Si at 18^3 and 4x4x4 (3.3e8 points) 0.85 and 1.23 s, h-BN 12x12x108 and 6x6x1 (2.8e8) 0.31 and
#: 0.39 s; 64-atom Si at 68^3, Gamma (real, 3e9) 0.89 and 0.31 s, 2x2x2 (2e11) 75 and 31 s.
SMALL_BUILD_POINTS = 1.0e9
#: ``on`` (default): the two pairs of two occupied states that one transform serves (module text) are
#: solved once; ``off``: every pair of every target column with every source column (the host's order).
PAIR_SYMMETRY_SETTING = "PARSEC_CUPY_BLOCH_PAIR_SYMMETRY"


def pair_symmetry_setting() -> bool:
    """Whether the occupied pairs are solved once for both of their states (``PARSEC_CUPY_BLOCH_PAIR_SYMMETRY``)."""

    import os

    value = os.environ.get(PAIR_SYMMETRY_SETTING, "on").strip().lower() or "on"
    if value not in ("on", "off"):
        raise ValueError(f"{PAIR_SYMMETRY_SETTING} must be on or off, not {value!r}")
    return value == "on"


def _symmetric(channel, target: int, conjugate: bool) -> bool:
    """Whether the pairs of an occupied target of ``target`` with a source are paired with their partners.

    Always for a source of the mesh point itself (the partner pair has the
    potential ``conj(w)``); for the partner ``-k`` of a representative only
    when the target has a partner too (the partner pair's source is then
    ``-k_target``, a point of the sum with the same density and ``q``).
    """

    return (not conjugate) or bool(channel.partners[target])


def _tile_cost(channel, tile: _Tile, sources, counts, symmetric: bool) -> int:
    """Pairs a tile solves."""

    width = int(tile.columns.size)
    total = 0
    for rep, _k, conjugate in sources:
        count = counts[rep]
        if not count:
            continue
        if tile.start is None or not symmetric or not _symmetric(channel, tile.rep, conjugate):
            total += width * count
        elif rep > tile.rep:
            total += width * count
        elif rep == tile.rep:
            total += width * max(0, count - tile.start - width) + width * (width + 1) // 2
    return total


def planned_bloch_exchange_bytes(channel: DeviceBlochChannel, solvers: DevicePeriodicPairSolvers,
                                 devices: Sequence[int]) -> dict[int, int]:
    """Device bytes of one build: the source copies and a tile's work on every solving device, ``W`` on the owners."""

    points = int(solvers.box.points)
    sources = 16 * points * sum(int(channel.occupied(rep).size) for rep in range(channel.representatives))
    if pair_symmetry_setting():
        sources *= 2  # the W of the occupied states that partner pairs reach, as large as the copies
    work = solvers.tile_bytes(TILE_COLUMNS) + solvers.cache.array_bytes
    needed = {int(device): sources + work for device in devices}
    for rep in range(channel.representatives):
        device = int(channel.devices[rep])
        needed[device] = needed.get(device, 0) + 16 * points * int(np.asarray(channel.occupations[rep]).size)
    return needed


def require_bloch_exchange_memory(channel: DeviceBlochChannel, solvers: DevicePeriodicPairSolvers,
                                  devices: Sequence[int]) -> None:
    """Refuse, with the numbers, a build whose copies, tiles or ``W`` do not fit the devices."""

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    for device, count in sorted(planned_bloch_exchange_bytes(channel, solvers, devices).items()):
        with cp.cuda.Device(device):
            free, total = cp.cuda.runtime.memGetInfo()
            available = int(free) + int(cp.get_default_memory_pool().free_bytes())
        if count > available:
            raise MemoryError(
                f"the exact exchange of this periodic hybrid needs {count / 2**30:.1f} GiB on device "
                f"{device} (a copy of the occupied orbitals of every k-point, W = V_x U of the k-points "
                f"there and the work of a tile) and {available / 2**30:.1f} GiB of {int(total) / 2**30:.1f} "
                "GiB are free there; run it on more GPUs, with fewer k-points or States_Num, or a smaller "
                "Exx_Batch_Size"
            )


def apply_device_bloch_exchange(
    channel: DeviceBlochChannel,
    solvers: DevicePeriodicPairSolvers,
    devices: Sequence[int],
    *,
    columns: str = "all",
    known: DeviceBlochExchangeApplication | None = None,
) -> DeviceBlochExchangeApplication:
    """``W_t = V_x U_t`` of every representative of a channel on ``devices`` (see the module text).

    ``columns="occupied"`` forms the occupied columns only (enough for the
    energy); ``"all"`` every column.  ``known``, an application of the
    occupied columns of the same orbitals, is completed with the empty ones
    and consumed.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    started = perf_counter()
    if columns not in ("all", "occupied"):
        raise ValueError("columns must be 'all' or 'occupied'")
    count = channel.representatives
    occupied = tuple(tuple(int(value) for value in channel.occupied(rep)) for rep in range(count))
    if known is not None:
        same = known.channel is channel or (
            known.channel.representatives == count
            and all(a is b for a, b in zip(known.channel.orbitals, channel.orbitals))
        )
        if not same or tuple(known.occupied) != occupied:
            raise ValueError("the known application belongs to other orbitals")
        if tuple(tuple(values) for values in known.columns) != occupied:
            raise ValueError("a known application must hold the occupied columns")
    devices = tuple(int(device) for device in devices) or (channel.devices[0],)
    points = int(solvers.box.points)
    # Targets of every representative.
    wanted = []
    for rep in range(count):
        states = int(np.asarray(channel.occupations[rep]).size)
        if known is not None:
            wanted.append(np.array(sorted(set(range(states)) - set(occupied[rep])), dtype=np.int64))
        elif columns == "all":
            wanted.append(np.arange(states, dtype=np.int64))
        else:
            wanted.append(np.asarray(occupied[rep], dtype=np.int64))
    sources = channel.sources()
    symmetric = pair_symmetry_setting()
    counts = [len(values) for values in occupied]
    tiles = []
    for rep in range(count):
        # Blocks of occupied targets (by their positions in the occupied list) and of empty ones.
        positions = {int(state): index for index, state in enumerate(occupied[rep])}
        targets = [int(state) for state in wanted[rep] if int(state) in positions]
        empty = [int(state) for state in wanted[rep] if int(state) not in positions]
        for start in range(0, len(targets), TILE_COLUMNS):
            block = np.asarray(targets[start : start + TILE_COLUMNS], dtype=np.int64)
            tiles.append(_Tile(rep, block, 0, positions[int(block[0])]))
        for start in range(0, len(empty), TILE_COLUMNS):
            tiles.append(_Tile(rep, np.asarray(empty[start : start + TILE_COLUMNS], dtype=np.int64), 0))
    for tile in tiles:
        if tile.start is not None and not np.array_equal(
            tile.columns, np.asarray(occupied[tile.rep][tile.start : tile.start + tile.columns.size])
        ):
            raise AssertionError("an occupied tile must be a contiguous run of the occupied list")
        tile.cost = _tile_cost(channel, tile, sources, counts, symmetric)
    if len(devices) > 1 and sum(tile.cost for tile in tiles) * float(solvers.box.fft_size) < SMALL_BUILD_POINTS:
        held = [sum(1 for owner in channel.devices if owner == device) for device in devices]
        devices = (devices[int(np.argmax(held))],)
    if known is not None:
        applied = known.applied
    else:
        require_bloch_exchange_memory(channel, solvers, devices)
        applied = []
        for rep in range(count):
            with cp.cuda.Device(channel.devices[rep]):
                states = int(np.asarray(channel.occupations[rep]).size)
                applied.append(cp.zeros((points, states), dtype=cp.complex128, order="F"))
    locks = [Lock() for _ in range(count)]
    # Most expensive tiles first, each to the least loaded device, its owner on a tie.
    loads = {device: 0 for device in devices}
    assigned: dict[int, list[_Tile]] = {device: [] for device in devices}
    for tile in sorted(tiles, key=lambda item: -item.cost):
        owner = channel.devices[tile.rep]
        device = min(
            devices, key=lambda name: (loads[name], 0 if name == owner else 1, devices.index(name))
        )
        loads[device] += tile.cost
        assigned[device].append(tile)
    for device in devices:
        assigned[device].sort(key=lambda item: (item.rep, int(item.columns[0])))
    inverse_volume = 1.0 / float(channel.volume_element)
    timing = {device: 0.0 for device in devices}
    solved = {device: 0 for device in devices}
    for device in sorted(set(channel.devices) | set(devices)):
        cp.cuda.Device(device).synchronize()

    def deliver(rep: int, positions: np.ndarray, rows) -> None:
        """Add the rows ``(b, N)`` to the columns ``positions`` of ``W`` of ``rep`` on the owner's device."""

        owner = channel.devices[rep]
        target = applied[rep]
        ready = cp.cuda.get_current_stream().record()
        with locks[rep]:
            with cp.cuda.Device(owner):
                stream = solvers.stream(owner, "delivery")
                with stream:
                    stream.wait_event(ready)
                    if int(rows.device.id) == owner:
                        received = rows
                    else:
                        received = cp.empty(rows.shape, dtype=cp.complex128)
                        received.data.copy_from_device_async(rows.data, rows.nbytes)
                    if positions.size and bool(np.all(np.diff(positions) == 1)):
                        target[:, int(positions[0]) : int(positions[-1]) + 1] += received.T
                    else:
                        index = cp.asarray(positions)
                        target[:, index] = target[:, index] + received.T
                stream.synchronize()

    def run(device: int) -> None:
        work = assigned[device]
        if not work:
            return
        with cp.cuda.Device(device), solvers.stream(device):
            copies = {}
            for rep in range(count):
                if occupied[rep]:
                    rows = _rows_on(cp, channel.orbitals[rep], np.asarray(occupied[rep]), device)
                    weights = cp.asarray(
                        np.asarray(channel.occupations[rep])[list(occupied[rep])] / float(channel.mesh_size)
                    )
                    copies[rep] = (rows, weights)
            pending: list = []
            # W of the occupied states of every representative that partner pairs reach, on this device.
            partners: dict = {}

            def partner_rows(rep: int):
                found = partners.get(rep)
                if found is None:
                    found = cp.zeros((counts[rep], points), dtype=cp.complex128)
                    partners[rep] = found
                return found

            for tile in work:
                targets = _rows_on(cp, channel.orbitals[tile.rep], tile.columns, device)
                rows = cp.zeros(targets.shape, dtype=cp.complex128)
                k_target = np.asarray(channel.k_points[tile.rep], dtype=np.float64)
                target_weights = None
                if tile.start is not None and symmetric:
                    target_weights = cp.asarray(
                        np.asarray(channel.occupations[tile.rep])[tile.columns] / float(channel.mesh_size)
                    )
                for rep, k_source, conjugate in sources:
                    if rep not in copies:
                        continue
                    source_rows, weights = copies[rep]
                    keywords = dict(conjugate=conjugate, inverse_volume=inverse_volume, timing=pending)
                    q = k_target - k_source
                    if target_weights is None or not _symmetric(channel, tile.rep, conjugate):
                        solved[device] += solvers.accumulate_bloch(
                            cp, q, targets, source_rows, weights, rows, **keywords
                        )
                    elif rep > tile.rep:
                        solved[device] += solvers.accumulate_bloch(
                            cp, q, targets, source_rows, weights, rows,
                            partner=partner_rows(rep), target_coefficients=target_weights, **keywords
                        )
                    elif rep == tile.rep:
                        # The pairs (n, m) with m at or after n in the occupied list: a rectangle of the
                        # columns after the tile, then one row at a time (its own pair without partner).
                        width = int(tile.columns.size)
                        start = int(tile.start)
                        mine = partner_rows(rep)
                        if start + width < counts[rep]:
                            solved[device] += solvers.accumulate_bloch(
                                cp, q, targets, source_rows[start + width :], weights[start + width :], rows,
                                partner=mine[start + width :], target_coefficients=target_weights, **keywords
                            )
                        for row in range(width):
                            begin, end = start + row, start + width
                            solved[device] += solvers.accumulate_bloch(
                                cp, q, targets[row : row + 1], source_rows[begin:end], weights[begin:end],
                                rows[row : row + 1], partner=mine[begin:end],
                                target_coefficients=target_weights[row : row + 1], skip_self=True, **keywords
                            )
                    # rep < tile.rep: the tiles of rep solve these pairs and add them here as partners
                for begin, end in pending:
                    end.synchronize()
                    timing[device] += cp.cuda.get_elapsed_time(begin, end) * 1.0e-3
                pending.clear()
                deliver(tile.rep, tile.columns, rows)
                del targets, rows
            for rep, values in partners.items():
                deliver(rep, np.asarray(occupied[rep], dtype=np.int64), values)
            del copies, partners

    try:
        solvers.run_on_devices(run, devices)
    finally:
        solvers.release_blocks()
    energy = 0.0
    if known is None:
        for rep in range(count):
            positions = np.asarray(occupied[rep], dtype=np.int64)
            if not positions.size:
                continue
            with cp.cuda.Device(channel.devices[rep]):
                vectors = channel.orbitals[rep]
                if not isinstance(vectors, cp.ndarray):
                    vectors = cp.asarray(np.asarray(vectors))
                index = cp.asarray(positions)
                values = cp.sum(cp.conj(vectors[:, index]) * applied[rep][:, index], axis=0).real
                weights = np.asarray(channel.occupations[rep])[positions]
                energy += float(channel.weights[rep]) * float(cp.asnumpy(values) @ weights)
        energy *= 0.5 * channel.spin_weight
    done = tuple(
        tuple(sorted(set(int(value) for value in wanted[rep]) | (set(occupied[rep]) if known is not None else set())))
        for rep in range(count)
    )
    pair_solves = int(sum(solved.values()))
    return DeviceBlochExchangeApplication(
        channel=channel,
        applied=applied,
        columns=done,
        occupied=occupied,
        energy=float(energy) if known is None else float(known.energy),
        pair_solves=pair_solves + (0 if known is None else int(known.pair_solves)),
        poisson_seconds=float(max(timing.values(), default=0.0)) + (0.0 if known is None else known.poisson_seconds),
        seconds=float(perf_counter() - started) + (0.0 if known is None else known.seconds),
    )


# ---------------------------------------------------------------------------
# ACE per k-point.
# ---------------------------------------------------------------------------


def adjoint_product(xp, projectors, block):
    """``xi^H X`` of an F-ordered ``xi`` (rows x rank) and a block (rows x columns), F order.

    On a CUDA device ``X`` is read in its own order (C or F) by one ZGEMM,
    without a copy of either operand.
    """

    if xp is np:
        return projectors.conj().T @ block
    from cupy_backends.cuda.libs import cublas
    from cupy.cuda import device as cuda_device

    if not projectors.flags.f_contiguous:
        projectors = xp.asfortranarray(projectors)
    if not (block.flags.c_contiguous or block.flags.f_contiguous):
        block = xp.ascontiguousarray(block)
    rows, rank = int(projectors.shape[0]), int(projectors.shape[1])
    columns = int(block.shape[1])
    if int(block.shape[0]) != rows:
        raise ValueError("the block and the projectors have different rows")
    result = xp.empty((rank, columns), dtype=xp.complex128, order="F")
    if rows == 0 or rank == 0 or columns == 0:
        result.fill(0)
        return result
    if block.flags.f_contiguous:
        operation, leading = cublas.CUBLAS_OP_N, rows
    else:
        operation, leading = cublas.CUBLAS_OP_T, columns
    handle = cuda_device.get_cublas_handle()
    one = np.array(1.0 + 0.0j, dtype=np.complex128)
    zero = np.array(0.0 + 0.0j, dtype=np.complex128)
    mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    try:
        cublas.zgemm(
            handle,
            cublas.CUBLAS_OP_C,
            operation,
            rank,
            columns,
            rows,
            one.ctypes.data,
            projectors.data.ptr,
            rows,
            block.data.ptr,
            leading,
            zero.ctypes.data,
            result.data.ptr,
            rank,
        )
    finally:
        cublas.setPointerMode(handle, mode)
    return result


@dataclass
class DeviceBlochACE:
    """``a_x V_ACE = -a_x xi xi^H`` of one k-point of one channel, ``xi`` on the k-point's device.

    The interface of the host ``BlochCompressedExchange``
    (:mod:`parsec_python.V_xc.bloch_exchange`): ``projectors`` (``xi``, F
    order), ``fraction`` (``a_x``), ``asymmetry`` of ``M``; :meth:`apply`,
    :meth:`expectation_values`, :attr:`rank`.
    """

    projectors: Any
    fraction: float
    asymmetry: float = 0.0
    released: bool = False

    @property
    def rank(self) -> int:
        return 0 if self.projectors is None else int(self.projectors.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        rows = int(self.projectors.shape[0])
        return rows, rows

    def _xp(self):
        import sys

        cupy = sys.modules.get("cupy")
        if cupy is not None and isinstance(self.projectors, cupy.ndarray):
            return cupy
        return np

    def _block(self, xp, vectors):
        """``(block on the device of xi, whether the caller's were host arrays)``."""

        if xp is np:
            return np.asarray(vectors), False
        if not isinstance(vectors, xp.ndarray):
            with xp.cuda.Device(int(self.projectors.device.id)):
                return xp.asarray(np.asarray(vectors, dtype=np.complex128)), True
        device = int(self.projectors.device.id)
        if int(vectors.device.id) != device:
            source = xp.ascontiguousarray(vectors)
            with xp.cuda.Device(device):
                copy = xp.empty(source.shape, dtype=source.dtype)
                copy.data.copy_from_device(source.data, source.nbytes)
            return copy, False
        return vectors, False

    def apply(self, vectors, *, scaled: bool = True):
        """``-a_x xi (xi^H V)`` (``scaled=False``: without ``a_x``) on the device of ``xi``.

        Host vectors give a host result; vectors of another device are
        copied to the device of ``xi``.
        """

        if self.released:
            raise RuntimeError("the compressed exchange operator was released")
        xp = self._xp()
        block, host = self._block(xp, vectors)
        single = getattr(block, "ndim", 2) == 1
        block = block[:, None] if single else block
        with _device_of(xp, self.projectors):
            coefficients = adjoint_product(xp, self.projectors, block)
            factor = -(self.fraction if scaled else 1.0)
            result = factor * (self.projectors @ coefficients)
        result = result[:, 0] if single else result
        return xp.asnumpy(result) if host else result

    def expectation_values(self, vectors, *, scaled: bool = True):
        """``<v|a_x V_ACE|v>`` of every column (real, on the device of ``xi``; host vectors give host values)."""

        if self.released:
            raise RuntimeError("the compressed exchange operator was released")
        xp = self._xp()
        block, host = self._block(xp, vectors)
        block = block[:, None] if getattr(block, "ndim", 2) == 1 else block
        with _device_of(xp, self.projectors):
            coefficients = adjoint_product(xp, self.projectors, block)
            factor = -(self.fraction if scaled else 1.0)
            values = factor * xp.sum((coefficients.conj() * coefficients).real, axis=0)
        return xp.asnumpy(values) if host else values

    def release(self) -> None:
        """Free the device memory of ``xi``; the scalars stay."""

        self.projectors = None
        self.released = True


class _NoPlace:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


def _device_of(xp, array):
    """The device of a CuPy array as a context (nothing for NumPy)."""

    if xp is np:
        return _NoPlace()
    return xp.cuda.Device(int(array.device.id))


def _right_solve_lower_adjoint(cp, lower, block) -> None:
    """``block <- block L^-H`` in place (``block`` F-contiguous complex, rows x k)."""

    from cupy.cuda import device as cuda_device
    from cupy_backends.cuda.libs import cublas

    lower = cp.asfortranarray(lower)
    rows, count = int(block.shape[0]), int(block.shape[1])
    if not block.flags.f_contiguous:
        raise ValueError("the block of the triangular solve must be F-contiguous")
    if rows == 0 or count == 0:
        return
    handle = cuda_device.get_cublas_handle()
    one = np.array(1.0 + 0.0j, dtype=np.complex128)
    mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    try:
        cublas.ztrsm(
            handle,
            cublas.CUBLAS_SIDE_RIGHT,
            cublas.CUBLAS_FILL_MODE_LOWER,
            cublas.CUBLAS_OP_C,
            cublas.CUBLAS_DIAG_NON_UNIT,
            rows,
            count,
            one.ctypes.data,
            lower.data.ptr,
            count,
            block.data.ptr,
            rows,
        )
    finally:
        cublas.setPointerMode(handle, mode)


def _device_cholesky(cp, matrix):
    """``L`` of ``matrix = L L^H`` on the device, refused (``LinAlgError``) as the host refuses it.

    cuSOLVER reports a matrix that is not positive definite through its info
    flag, which CuPy ignores unless asked (``cupyx.errstate(linalg='raise')``):
    the factor is then partly computed and may be finite.  So the flag is
    asked for, and the diagonal of the factor must be positive and finite.
    """

    import cupyx

    message = (
        "the exchange matrix M = -U^H V_x U of a k-point is not positive definite; "
        "ACE needs orbitals that overlap the occupied ones"
    )
    try:
        with cupyx.errstate(linalg="raise"):
            lower = cp.linalg.cholesky(matrix)
    except Exception as error:  # cuSOLVER's info flag (LinAlgError), or another failure of the factor
        raise np.linalg.LinAlgError(message) from error
    diagonal = cp.diagonal(lower)
    if not bool(cp.all(cp.isfinite(lower))) or not bool(cp.all(diagonal.real > 0.0)):
        raise np.linalg.LinAlgError(message)
    return lower


def compress_device_bloch_exchange(
    application: DeviceBlochExchangeApplication, *, fraction: float = 1.0
) -> list[DeviceBlochACE | None]:
    """``xi_t = W_t L_t^-H`` with ``M_t = -U_t^H W_t = L_t L_t^H``, per representative, on its device.

    ``W`` is overwritten by ``xi`` (the application is consumed).  ``M_t``
    is Hermitized; its asymmetry ``max|M - M^H| / max|M|`` is kept on each
    operator.  ``None`` for a representative without states.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    channel = application.channel
    operators: list[DeviceBlochACE | None] = []
    for rep in range(channel.representatives):
        states = int(np.asarray(channel.occupations[rep]).size)
        if len(application.columns[rep]) != states:
            raise ValueError("ACE needs V_x applied to every orbital of every k-point")
        applied = application.applied[rep]
        if states == 0:
            operators.append(None)
            continue
        with cp.cuda.Device(channel.devices[rep]):
            vectors = channel.orbitals[rep]
            if not isinstance(vectors, cp.ndarray):
                vectors = cp.asarray(np.asarray(vectors))
            overlap = -adjoint_product(cp, cp.asfortranarray(vectors), applied)
            scale = float(cp.abs(overlap).max())
            asymmetry = 0.0
            if scale > 0.0:
                asymmetry = float(cp.abs(overlap - overlap.conj().T).max()) / scale
            overlap = 0.5 * (overlap + overlap.conj().T)
            lower = _device_cholesky(cp, overlap)
            if not applied.flags.f_contiguous:
                applied = cp.asfortranarray(applied)
            _right_solve_lower_adjoint(cp, lower, applied)
            cp.cuda.get_current_stream().synchronize()
        operators.append(DeviceBlochACE(projectors=applied, fraction=float(fraction), asymmetry=asymmetry))
        application.applied[rep] = None
    return operators


def device_cross_energy(xp, operator: DeviceBlochACE, vectors, occupations, weight: float, spin_weight: float):
    """``(s/2) w_t sum_n f_n <u_n|V_ACE|u_n>`` (unscaled) of one k-point as an ``xp`` array of shape ``(1,)``."""

    occupations = np.asarray(occupations, dtype=np.float64)
    occupied = np.flatnonzero(occupations > OCCUPATION_THRESHOLD)
    if operator is None or not occupied.size:
        return xp.zeros(1, dtype=xp.float64)
    columns = vectors[:, : int(occupied[-1]) + 1] if occupied.size == occupied[-1] + 1 else vectors[:, occupied]
    values = operator.expectation_values(columns, scaled=False)
    weights = xp.asarray(occupations[occupied])
    return (0.5 * float(spin_weight) * float(weight) * xp.sum(values * weights)).reshape(1)


__all__ = [
    "DeviceBlochACE",
    "DeviceBlochChannel",
    "DeviceBlochExchangeApplication",
    "DeviceKernelCache",
    "DevicePeriodicBox",
    "DevicePeriodicPairSolvers",
    "adjoint_product",
    "apply_device_bloch_exchange",
    "compress_device_bloch_exchange",
    "device_bloch_channel",
    "device_cross_energy",
    "planned_bloch_exchange_bytes",
    "require_bloch_exchange_memory",
]
