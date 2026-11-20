"""Opt-in FP64 Poisson CG on one CUDA device.

The operator is that of the normalized symmetry wedge of the Hartree problem,
or of the full grid where no reduction applies.
No boundary, discretization, tolerance, or chronological predictor changes.
Static coefficients, work vectors, and recurrence scalars remain on one CUDA
device. A captured group of iterations checks stopping ON DEVICE after every
iteration; converged/broken recurrences are frozen until the next host poll.
Thus graph batching never performs extra numerical iterations after stopping.
The right-hand side and the initial guess are host or device arrays; the
solution comes back on the host unless ``device_result`` asks for a device
array (:mod:`.cupy_resident`).
"""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import scipy.sparse as sp

from ..backends.cupy_capture import capture_graph, collector_paused
from ..backends.cupy_stencil_major import StencilMajorHostMetadata
from .native_poisson import NativePoissonSolver


def packed_stencil(metadata):
    """Slot-major arrays of a stencil that a symmetry sector already packed.

The eigensolver packs every sector from canonical CSR rows into the layout
the CG kernels read: entry ``s`` of a row is in slot ``s`` and its uint8 code
selects the exact float64 coefficient. ``pack_stencil`` on the same matrix
gives the same neighbors and the same coefficient per slot; only the order of
the palette differs (bit pattern instead of value), which no kernel observes.
The arrays are returned as they are, without a copy.
"""
    neighbors = metadata.neighbors
    codes = metadata.coefficient_codes
    palette = metadata.coefficient_palette
    if neighbors.shape[0] > 256:
        raise ValueError("GPU Poisson requires int32 rows and at most 256 entries per row")
    if not np.all(np.isfinite(palette)):
        raise ValueError("Poisson coefficients must be finite")
    return neighbors, codes, palette


def pack_stencil(operator):
    """Canonical CSR -> slot-major indices and exact coefficient palette.

No quantization: a uint8/uint16 entry selects an existing float64 coefficient.
Duplicate entries are summed; rows retain their canonical CSR accumulation order.
This layout is intended for bounded-width finite-difference stencils.
"""
    matrix = sp.csr_matrix(operator, dtype=np.float64, copy=True)
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Poisson operator must be a nonempty square matrix")
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("Poisson coefficients must be finite")
    n = matrix.shape[0]
    counts = np.diff(matrix.indptr)
    width = int(counts.max(initial=0))
    if n > np.iinfo(np.int32).max or width > 256:
        raise ValueError("GPU Poisson requires int32 rows and at most 256 entries per row")
    palette, inverse = np.unique(matrix.data, return_inverse=True)
    if palette.size > 65536:
        raise ValueError("GPU Poisson coefficient palette exceeds 65536 exact values")
    code_dtype = np.uint8 if palette.size <= 256 else np.uint16
    neighbors = np.full((max(width, 1), n), -1, dtype=np.int32)
    codes = np.zeros(neighbors.shape, dtype=code_dtype)
    for slot in range(width):
        valid = counts > slot
        positions = matrix.indptr[:-1][valid] + slot
        neighbors[slot, valid] = matrix.indices[positions]
        codes[slot, valid] = inverse[positions]
    # An all-zero operator is useful for testing breakdown; no valid index
    # reads this placeholder coefficient.
    if not palette.size:
        palette = np.zeros(1, dtype=np.float64)
    return matrix, neighbors, codes, palette


class CuPyPreparedConjugateGradientBackend:
    """Prepared finite-difference CG with one host status poll per graph.

The solve is synchronous at the public boundary and uses an owned CUDA
stream/context. It does not allocate device memory inside the recurrence.
Not thread-safe: create a distinct instance for concurrent Poisson solves.
"""
    def __init__(self, operator, *, device_id=None, graph_iterations=None):
        from ..backends.cupy import require_cupy
        cp, _ = require_cupy()
        self.cp = cp
        self.device_id = int(cp.cuda.Device().id if device_id is None else device_id)
        self.graph_iterations = int(os.environ.get("PARSEC_CUPY_POISSON_GRAPH_STEPS", "8")
                                    if graph_iterations is None else graph_iterations)
        if not 1 <= self.graph_iterations <= 64:
            raise ValueError("Poisson graph iterations must be between 1 and 64")
        if isinstance(operator, StencilMajorHostMetadata):
            # Already in the layout of the kernels: no CSR round trip and no
            # second packing. The CSR is rebuilt only if ``operator`` is read.
            self._stencil = operator
            self._operator = None
            neighbors, codes, palette = packed_stencil(operator)
            self.shape = tuple(int(size) for size in operator.shape)
            self.stencil_packing = "reused from the symmetry-sector stencil"
        else:
            self._stencil = None
            self._operator, neighbors, codes, palette = pack_stencil(operator)
            self.shape = self._operator.shape
            self.stencil_packing = "packed from CSR"
        self.n = np.int32(self.shape[0])
        self.width = np.int32(neighbors.shape[0])
        self.blocks = (int(self.n) + 255) // 256
        self.coefficient_palette_size = len(palette)
        self.worker_count = 0
        self.storage_mode = f"CUDA slot-major int32 + {codes.dtype} exact coefficient palette"
        self.last_host_polls = 0
        source = Path(__file__).with_name("cupy_prepared_kernels.cu").read_text()
        source = source.replace("COEFF_CODE", "unsigned char" if codes.dtype == np.uint8 else "unsigned short")
        with cp.cuda.Device(self.device_id):
            self.stream = cp.cuda.Stream(non_blocking=True)
            with self.stream:
                self.neighbors = cp.asarray(neighbors)
                self.codes = cp.asarray(codes)
                self.palette = cp.asarray(palette)
                self.x, self.r, self.p, self.ap, self.b = [cp.empty(int(self.n), cp.float64) for _ in range(5)]
                self.partial = cp.empty(self.blocks, cp.float64)
                # rr, alpha, beta, tol, status, iterations, matvecs, norm0, norm
                self.state = cp.zeros(9, cp.float64)
                module = cp.RawModule(code=source, options=("-std=c++11", "--fmad=false"))
                self.kernels = {name: module.get_function(name) for name in
                                ("matvec", "residual", "initialize", "alpha", "update_xr", "beta", "update_p", "finish")}
                self.stream.synchronize()
                self.graph = None
                # All kernels are compiled and all arrays allocated before
                # capture; only explicit device launches occur between
                # begin_capture/end_capture.  The capture runs in the relaxed
                # mode, the documented default of CuPy 14.2 (a capture begun
                # without a mode was measured to behave so) and now named: in
                # it no call of the capturing thread but a device
                # synchronization was measured to invalidate a capture, so
                # the pause of the collector is a precaution here.  A capture
                # that fails is ended, so that neither this stream nor its
                # thread stays in capture mode, and one that the end of
                # another thread of the device invalidated is recorded again.
                if self.graph_iterations > 1:
                    with collector_paused():
                        self.graph, _ = capture_graph(
                            self.stream, lambda: self._iterations(self.graph_iterations),
                            mode=cp.cuda.runtime.streamCaptureModeRelaxed)
        self.device_bytes = sum(v.nbytes for v in (self.neighbors, self.codes, self.palette,
                                self.x, self.r, self.p, self.ap, self.b, self.partial, self.state))

    @property
    def operator(self):
        """Canonical CSR operator; rebuilt from a packed stencil on first use."""
        if self._operator is None:
            matrix = self._stencil.to_csr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            self._operator = matrix
        return self._operator

    def _matvec(self, vector, force):
        self.kernels["matvec"]((self.blocks,), (256,),
            (self.n, self.width, self.neighbors, self.codes, self.palette, vector,
             self.ap, self.partial, self.state, np.int32(force)))

    def _iterations(self, count):
        for _ in range(count):
            self._matvec(self.p, 0)
            self.kernels["alpha"]((1,), (256,), (self.partial, np.int32(self.blocks), self.state))
            self.kernels["update_xr"]((self.blocks,), (256,),
                (self.n, self.x, self.r, self.p, self.ap, self.partial, self.state))
            self.kernels["beta"]((1,), (256,), (self.partial, np.int32(self.blocks), self.state))
            self.kernels["update_p"]((self.blocks,), (256,), (self.n, self.r, self.p, self.state))

    def solve(self, rhs, initial, *, relative_tolerance, absolute_tolerance, max_iterations, device_result=False):
        cp = self.cp
        rhs = cp.ascontiguousarray(rhs,dtype=cp.float64) if isinstance(rhs,cp.ndarray) else np.ascontiguousarray(rhs, dtype=np.float64)
        initial = cp.ascontiguousarray(initial,dtype=cp.float64) if isinstance(initial,cp.ndarray) else np.ascontiguousarray(initial, dtype=np.float64)
        if rhs.shape != (int(self.n),) or initial.shape != rhs.shape:
            raise ValueError("Poisson vectors do not match the operator")
        if (not np.isfinite(relative_tolerance) or relative_tolerance < 0
                or not np.isfinite(absolute_tolerance) or absolute_tolerance < 0
                or int(max_iterations) != max_iterations or max_iterations < 1):
            raise ValueError("Invalid Poisson CG controls")
        ready = None
        if isinstance(rhs,cp.ndarray) or isinstance(initial,cp.ndarray):
            ready=cp.cuda.Event(disable_timing=True)
            ready.record(cp.cuda.get_current_stream())
        self.last_host_polls = 0
        with cp.cuda.Device(self.device_id), self.stream:
            if ready is not None: self.stream.wait_event(ready)
            if isinstance(rhs,cp.ndarray): cp.copyto(self.b,rhs)
            else: self.b.set(rhs, stream=self.stream)
            if isinstance(initial,cp.ndarray): cp.copyto(self.x,initial)
            else: self.x.set(initial, stream=self.stream)
            self._matvec(self.x, 1)
            self.kernels["residual"]((self.blocks,), (256,),
                (self.n, self.b, self.ap, self.r, self.p, self.partial))
            self.kernels["initialize"]((1,), (256,),
                (self.partial, np.int32(self.blocks), self.state,
                 np.float64(relative_tolerance), np.float64(absolute_tolerance)))
            state = self.state.get(stream=self.stream)
            self.last_host_polls += 1
            remaining = int(max_iterations) - 1
            initially_converged = state[4] == 1
            while state[4] == 0 and remaining > 0:
                count = min(self.graph_iterations, remaining)
                if self.graph is not None and count == self.graph_iterations:
                    self.graph.launch(stream=self.stream)
                else:
                    self._iterations(count)
                remaining -= count
                state = self.state.get(stream=self.stream)
                self.last_host_polls += 1
            if not initially_converged:
                self._matvec(self.x, 1)
                self.kernels["residual"]((self.blocks,), (256,),
                    (self.n, self.b, self.ap, self.r, self.p, self.partial))
                self.kernels["finish"]((1,), (256,), (self.partial, np.int32(self.blocks), self.state))
                state = self.state.get(stream=self.stream)
            solution = self.x.copy() if device_result else self.x.get(stream=self.stream)
            if device_result: self.stream.synchronize()
        return {"solution": solution, "converged": bool(state[4] == 1),
                "iterations": int(state[5]), "matrix_vector_products": int(state[6]),
                "residual_norm": float(state[8]), "initial_residual_norm": float(state[7]),
                "tolerance": float(state[3]), "breakdown": bool(state[4] < 0)}



# The CG of one Poisson problem on several devices of the process: from this many rows of the operator on
# (``PARSEC_HARTREE_CG_DEVICES=auto``).  The wedge of the D2d sectors of the nanodiamond C19395H2556 has
# 5.9 million rows, that of C795H300 about 0.25 million.
MULTI_DEVICE_MIN_ROWS = 1 << 21


def cg_device_setting():
    """``PARSEC_HARTREE_CG_DEVICES``: ``auto`` (``None``) or how many devices the CG takes (1: one, as before)."""

    raw = os.environ.get("PARSEC_HARTREE_CG_DEVICES", "auto").strip().lower()
    if raw in ("", "auto"):
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("PARSEC_HARTREE_CG_DEVICES must be auto or a number of devices") from error
    if value < 1:
        raise ValueError("PARSEC_HARTREE_CG_DEVICES must be at least 1")
    return value


def cg_devices(rows, device_ids, first):
    """The devices of the CG of a ``rows``-row operator: ``first`` (the Hartree device) and others of ``device_ids``.

    ``auto`` takes every device of ``device_ids`` from :data:`MULTI_DEVICE_MIN_ROWS` rows on, one below;
    a number takes as many as there are.  Every device holds at least one block of 256 rows.
    """

    ordered = [int(first)] + [int(device) for device in device_ids if int(device) != int(first)]
    setting = cg_device_setting()
    if setting is None:
        count = len(ordered) if int(rows) >= MULTI_DEVICE_MIN_ROWS else 1
    else:
        count = min(setting, len(ordered))
    count = max(1, min(count, (int(rows) + 255) // 256))
    return tuple(ordered[:count])


class CuPyMultiDeviceConjugateGradientBackend:
    """The CG of :class:`CuPyPreparedConjugateGradientBackend` with its rows on several devices, bit for bit.

The rows are cut at blocks of 256, the blocks of the reductions of the kernels: every device holds the
stencil of its rows, their parts of x, r, A p and b, and the whole vector p.  Each iteration the devices
form A p of their rows and the block partials of p.A p, the first device sums the partials of all blocks
in their order (the same ``alpha`` kernel over the same values as on one device), and the other devices
take its state; so for x, r and the partials of r.r, and for ``beta``.  Every device then updates the p
of its rows and sends them to the others (peer copies).  The recurrence, its stopping rule and every value
are those of the CG on one device; only where the rows are summed differs, not the order of any sum.
The first device is that of the Hartree objects, where the right-hand side comes from and the solution
goes.  Without a CUDA graph: every launch is made by the calling thread.
"""
    def __init__(self, operator, *, device_ids, graph_iterations=None):
        from types import SimpleNamespace

        from ..backends.cupy import require_cupy
        cp, _ = require_cupy()
        self.cp = cp
        self.device_ids = tuple(int(device) for device in device_ids)
        if len(self.device_ids) < 2 or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("the CG on several devices needs two or more distinct devices")
        self.device_id = self.device_ids[0]
        self.graph_iterations = int(os.environ.get("PARSEC_CUPY_POISSON_GRAPH_STEPS", "8")
                                    if graph_iterations is None else graph_iterations)
        if not 1 <= self.graph_iterations <= 64:
            raise ValueError("Poisson graph iterations must be between 1 and 64")
        if isinstance(operator, StencilMajorHostMetadata):
            self._stencil = operator
            self._operator = None
            neighbors, codes, palette = packed_stencil(operator)
            self.shape = tuple(int(size) for size in operator.shape)
            self.stencil_packing = "reused from the symmetry-sector stencil"
        else:
            self._stencil = None
            self._operator, neighbors, codes, palette = pack_stencil(operator)
            self.shape = self._operator.shape
            self.stencil_packing = "packed from CSR"
        n = int(self.shape[0])
        self.n = np.int32(n)
        self.width = np.int32(neighbors.shape[0])
        self.blocks = (n + 255) // 256
        if self.blocks < len(self.device_ids):
            raise ValueError("fewer blocks of rows than devices")
        self.coefficient_palette_size = len(palette)
        self.worker_count = 0
        self.last_host_polls = 0
        source = Path(__file__).with_name("cupy_prepared_kernels.cu").read_text()
        source = source.replace("COEFF_CODE", "unsigned char" if codes.dtype == np.uint8 else "unsigned short")
        names = ("matvec_rows", "residual", "initialize", "alpha", "update_xr", "beta", "update_p", "finish")
        base, extra = divmod(self.blocks, len(self.device_ids))
        self.parts = []
        block0 = 0
        for index, device in enumerate(self.device_ids):
            count = base + (1 if index < extra else 0)
            row0 = block0 * 256
            rows = min(n, (block0 + count) * 256) - row0
            part = SimpleNamespace(device=device, block0=block0, blocks=count, row0=row0, rows=rows)
            with cp.cuda.Device(device):
                part.stream = cp.cuda.Stream(non_blocking=True)
                with part.stream:
                    part.neighbors = cp.asarray(np.ascontiguousarray(neighbors[:, row0:row0 + rows]))
                    part.codes = cp.asarray(np.ascontiguousarray(codes[:, row0:row0 + rows]))
                    part.palette = cp.asarray(palette)
                    part.p = cp.zeros(n, cp.float64)
                    part.x, part.r, part.ap, part.b = [cp.empty(rows, cp.float64) for _ in range(4)]
                    part.partial = cp.empty(count, cp.float64)
                    part.state = cp.zeros(9, cp.float64)
                    module = cp.RawModule(code=source, options=("-std=c++11", "--fmad=false"))
                    part.kernels = {name: module.get_function(name) for name in names}
                    part.done = cp.cuda.Event(disable_timing=True)
                    part.join = cp.cuda.Event(disable_timing=True)
                    part.stream.synchronize()
            self.parts.append(part)
            block0 += count
        first = self.parts[0]
        with cp.cuda.Device(first.device):
            self.partials = cp.empty(self.blocks, cp.float64)
            self.solution = cp.empty(n, cp.float64)
            self.gathered = cp.cuda.Event(disable_timing=True)
            self.fork = cp.cuda.Event(disable_timing=True)
        for part in self.parts:
            part.target = (self.partials[part.block0:part.block0 + part.blocks]
                           if part is first else part.partial)
        for part in self.parts:
            for other in self.parts:
                if other is not part:
                    with cp.cuda.Device(part.device):
                        try:
                            cp.cuda.runtime.deviceEnablePeerAccess(other.device)
                        except cp.cuda.runtime.CUDARuntimeError:
                            pass  # enabled already, or no peer access: the copies are made either way
        self.storage_mode = (f"CUDA slot-major int32 + {codes.dtype} exact coefficient palette, rows on "
                             f"devices {', '.join(str(device) for device in self.device_ids)}")
        self.device_bytes = sum(sum(array.nbytes for array in (
            part.neighbors, part.codes, part.palette, part.p, part.x, part.r, part.ap, part.b, part.partial,
            part.state)) for part in self.parts) + self.partials.nbytes + self.solution.nbytes
        # A group of iterations captured as one graph on the stream of the first device, the streams of the
        # other devices forked from it and joined to it: one launch from the host per group instead of about
        # forty per iteration.  Where the driver refuses such a capture, the iterations are launched one by one.
        self.graph = None
        self.graph_error = None
        if self.graph_iterations > 1:
            try:
                with collector_paused(), cp.cuda.Device(first.device):
                    self.graph, _ = capture_graph(
                        first.stream, lambda: self._captured(self.graph_iterations),
                        mode=cp.cuda.runtime.streamCaptureModeRelaxed)
            except Exception as error:  # noqa: BLE001 - the uncaptured iterations serve
                self.graph = None
                self.graph_error = f"{type(error).__name__}: {error}"

    @property
    def operator(self):
        """Canonical CSR operator; rebuilt from a packed stencil on first use."""
        if self._operator is None:
            matrix = self._stencil.to_csr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            self._operator = matrix
        return self._operator

    def _copy(self, destination, destination_device, source, source_device, size, stream):
        # Unified addresses name the device of either end: an ordinary asynchronous copy of kind default,
        # which a stream capture takes (a peer copy is refused there).
        del destination_device, source_device
        self.cp.cuda.runtime.memcpyAsync(destination, source, size, self.cp.cuda.runtime.memcpyDefault, stream.ptr)

    def _launch(self, part, name, grid, args):
        part.kernels[name]((grid,), (256,), args)

    def _send_partials(self, part):
        """Queue the copy of the block partials of ``part`` to the first device, and mark its stream."""
        first = self.parts[0]
        if part is not first:
            self._copy(self.partials.data.ptr + part.block0 * 8, first.device, part.partial.data.ptr,
                       part.device, part.blocks * 8, part.stream)
        part.done.record(part.stream)

    def _reduce(self, name, *extra):
        """The first device sums the partials of all blocks once every device sent them; the others take its state."""
        cp, first = self.cp, self.parts[0]
        with cp.cuda.Device(first.device), first.stream:
            for part in self.parts[1:]:
                first.stream.wait_event(part.done)
            self._launch(first, name, 1, (self.partials, np.int32(self.blocks), first.state, *extra))
            first.done.record(first.stream)
        for part in self.parts[1:]:
            with cp.cuda.Device(part.device), part.stream:
                part.stream.wait_event(first.done)
                self._copy(part.state.data.ptr, part.device, first.state.data.ptr, first.device, 72, part.stream)

    def _share(self, vectors):
        """Every device sends ``vectors(part)`` (its rows) into the whole vector p of every other device."""
        cp = self.cp
        for part in self.parts:
            part.done.record(part.stream)
        for part in self.parts:
            with cp.cuda.Device(part.device), part.stream:
                for other in self.parts:
                    if other is part:
                        continue
                    part.stream.wait_event(other.done)
                    self._copy(part.p.data.ptr + other.row0 * 8, part.device, vectors(other).data.ptr,
                               other.device, other.rows * 8, part.stream)

    def _residual(self, force_vector):
        """r = b - A x and p = r on every device (x the whole vector in ``part.p``), their partials summed."""
        cp = self.cp
        for part in self.parts:
            with cp.cuda.Device(part.device), part.stream:
                self._launch(part, "matvec_rows", part.blocks, (
                    np.int32(part.rows), np.int32(part.row0), self.width, part.neighbors, part.codes,
                    part.palette, part.p, part.ap, part.target, part.state, np.int32(1)))
                self._launch(part, "residual", part.blocks, (
                    np.int32(part.rows), part.b, part.ap, part.r, part.p[part.row0:], part.target))
                self._send_partials(part)

    def _iteration(self):
        cp = self.cp
        for part in self.parts:
            with cp.cuda.Device(part.device), part.stream:
                self._launch(part, "matvec_rows", part.blocks, (
                    np.int32(part.rows), np.int32(part.row0), self.width, part.neighbors, part.codes,
                    part.palette, part.p, part.ap, part.target, part.state, np.int32(0)))
                self._send_partials(part)
        self._reduce("alpha")
        for part in self.parts:
            with cp.cuda.Device(part.device), part.stream:
                self._launch(part, "update_xr", part.blocks, (
                    np.int32(part.rows), part.x, part.r, part.p[part.row0:], part.ap, part.target, part.state))
                self._send_partials(part)
        self._reduce("beta")
        for part in self.parts:
            with cp.cuda.Device(part.device), part.stream:
                self._launch(part, "update_p", part.blocks, (
                    np.int32(part.rows), part.r, part.p[part.row0:], part.state))
        self._share(lambda part: part.p[part.row0:])

    def _join(self):
        """The stream of the first device waits for what the other devices queued."""
        cp, first = self.cp, self.parts[0]
        for part in self.parts[1:]:
            with cp.cuda.Device(part.device):
                part.join.record(part.stream)
            first.stream.wait_event(part.join)

    def _fork(self):
        """The streams of the other devices wait for what the first device queued."""
        cp, first = self.cp, self.parts[0]
        with cp.cuda.Device(first.device):
            self.fork.record(first.stream)
        for part in self.parts[1:]:
            with cp.cuda.Device(part.device):
                part.stream.wait_event(self.fork)

    def _captured(self, count):
        """``count`` iterations, forked from and joined to the stream of the first device (for the capture)."""
        self._fork()
        for _ in range(count):
            self._iteration()
        with self.cp.cuda.Device(self.parts[0].device):
            self._join()

    def solve(self, rhs, initial, *, relative_tolerance, absolute_tolerance, max_iterations, device_result=False):
        cp, first = self.cp, self.parts[0]
        n = int(self.n)
        if tuple(np.shape(rhs)) != (n,) or tuple(np.shape(initial)) != (n,):
            raise ValueError("Poisson vectors do not match the operator")
        if (not np.isfinite(relative_tolerance) or relative_tolerance < 0
                or not np.isfinite(absolute_tolerance) or absolute_tolerance < 0
                or int(max_iterations) != max_iterations or max_iterations < 1):
            raise ValueError("Invalid Poisson CG controls")
        self.last_host_polls = 0
        # What the caller queued on its stream (of its current device) before the vectors are read.
        ready = cp.cuda.Event(disable_timing=True)
        ready.record(cp.cuda.get_current_stream())
        with cp.cuda.Device(first.device):
            with first.stream:
                first.stream.wait_event(ready)
                # The whole right-hand side and starting vector on the first device.
                whole_b = (cp.ascontiguousarray(rhs, dtype=cp.float64) if isinstance(rhs, cp.ndarray)
                           and rhs.device.id == first.device else cp.asarray(
                               np.ascontiguousarray(cp.asnumpy(rhs) if isinstance(rhs, cp.ndarray) else rhs,
                                                    dtype=np.float64)))
                whole_x = (cp.ascontiguousarray(initial, dtype=cp.float64) if isinstance(initial, cp.ndarray)
                           and initial.device.id == first.device else cp.asarray(
                               np.ascontiguousarray(cp.asnumpy(initial) if isinstance(initial, cp.ndarray)
                                                    else initial, dtype=np.float64)))
                first.done.record(first.stream)
        for part in self.parts:
            with cp.cuda.Device(part.device), part.stream:
                part.stream.wait_event(first.done)
                # b and x of its rows, the whole x in p for the first product.
                self._copy(part.b.data.ptr, part.device, whole_b.data.ptr + part.row0 * 8, first.device,
                           part.rows * 8, part.stream)
                self._copy(part.x.data.ptr, part.device, whole_x.data.ptr + part.row0 * 8, first.device,
                           part.rows * 8, part.stream)
                self._copy(part.p.data.ptr, part.device, whole_x.data.ptr, first.device, n * 8, part.stream)
        self._residual(None)
        self._reduce("initialize", np.float64(relative_tolerance), np.float64(absolute_tolerance))
        # p = r: every device sends the p of its rows to the others.
        self._share(lambda part: part.p[part.row0:])
        state = first.state.get(stream=first.stream)
        self.last_host_polls += 1
        remaining = int(max_iterations) - 1
        initially_converged = state[4] == 1
        while state[4] == 0 and remaining > 0:
            count = min(self.graph_iterations, remaining)
            if self.graph is not None and count == self.graph_iterations:
                with cp.cuda.Device(first.device):
                    self._join()
                    self.graph.launch(stream=first.stream)
                self._fork()
            else:
                for _ in range(count):
                    self._iteration()
            remaining -= count
            state = first.state.get(stream=first.stream)
            self.last_host_polls += 1
        if not initially_converged:
            # The residual of the solution: the whole x in p on every device.
            for part in self.parts:
                with cp.cuda.Device(part.device), part.stream:
                    cp.copyto(part.p[part.row0:part.row0 + part.rows], part.x)
            self._share(lambda part: part.x)
            self._residual(None)
            self._reduce("finish")
            state = first.state.get(stream=first.stream)
        # The solution on the first device.
        for part in self.parts:
            part.done.record(part.stream)
        with cp.cuda.Device(first.device), first.stream:
            for part in self.parts:
                first.stream.wait_event(part.done)
                self._copy(self.solution.data.ptr + part.row0 * 8, first.device, part.x.data.ptr, part.device,
                           part.rows * 8, first.stream)
            solution = self.solution.copy() if device_result else self.solution.get(stream=first.stream)
            if device_result:
                first.stream.synchronize()
            else:
                first.stream.synchronize()
        return {"solution": solution, "converged": bool(state[4] == 1),
                "iterations": int(state[5]), "matrix_vector_products": int(state[6]),
                "residual_norm": float(state[8]), "initial_residual_norm": float(state[7]),
                "tolerance": float(state[3]), "breakdown": bool(state[4] < 0)}


class CuPyPreparedPoissonSolver(NativePoissonSolver):
    """Use the native wrapper's exact chronological starting-vector policy.

    ``device_ids``, the devices of the process, lets a large operator take several of them
    (:func:`cg_devices`, ``PARSEC_HARTREE_CG_DEVICES``), ``device_id`` first: the same values.
    """
    def __init__(self, operator, *, device_id=None, graph_iterations=None, device_ids=None):
        def factory(op):
            devices = (() if device_ids is None or device_id is None
                       else cg_devices(int(op.shape[0]), device_ids, device_id))
            if len(devices) > 1:
                return CuPyMultiDeviceConjugateGradientBackend(
                    op, device_ids=devices, graph_iterations=graph_iterations)
            return CuPyPreparedConjugateGradientBackend(op, device_id=device_id, graph_iterations=graph_iterations)
        super().__init__(operator, backend_factory=factory)
