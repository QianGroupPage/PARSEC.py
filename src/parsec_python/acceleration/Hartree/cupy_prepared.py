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


class CuPyPreparedPoissonSolver(NativePoissonSolver):
    """Use the native wrapper's exact chronological starting-vector policy."""
    def __init__(self, operator, *, device_id=None, graph_iterations=None):
        super().__init__(operator, backend_factory=lambda op: CuPyPreparedConjugateGradientBackend(
            op, device_id=device_id, graph_iterations=graph_iterations))
