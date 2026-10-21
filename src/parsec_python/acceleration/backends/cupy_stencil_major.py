"""Coalesced CUDA finite-difference action for PARSEC's compact stencil.

The first compact CUDA implementation retains CSR row storage.  That is a
useful exact fallback, but adjacent CUDA threads then read metadata at a
roughly ``row_width`` stride.  This implementation transposes the short CSR
rows once during backend construction:

``neighbor[slot, row]`` and ``coefficient_code[slot, row]``.

For a fixed finite-difference slot, a warp therefore reads contiguous
metadata.  Each thread still visits the slots of its own row in canonical CSR
order, so changing the storage layout does not change the order of the
float64 stencil sum.

The second kernel fuses the elementwise Chebyshev recurrence with the
Hamiltonian action.  It evaluates, in the same operation order used by the
Python recurrence,

``Y = (((H X - center*X)*scale) - sigma*X_previous)*sigma_next``.

For the low-rank nonlocal term, the small signed projector coefficients are
supplied as a device array.  The kernel performs the remaining CSR projector
row scatter after the local finite-difference sum, matching
``CuPyHamiltonian``'s ordinary ``local_result += nonlocal_result`` ordering
without allocating the full nonlocal image.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import scipy.sparse as sp

from .cupy_compact import _coefficient_palette
from .cupy_compile import compile_cupy_raw
from .stencil_pipeline import (
    THREADS,
    WIDTH,
    StencilKernels,
    compiled_kernels,
    stencil_kernel_setting,
)


_CUDA_SOURCE = r"""
extern "C" __global__
void stencil_major_spmm6(
    const long long row_count,
    const int slot_count,
    const int* __restrict__ neighbors,
    const unsigned char* __restrict__ coefficient_codes,
    const double* __restrict__ coefficient_palette,
    const double* __restrict__ local_potential,
    const int add_local,
    const int* __restrict__ projector_row_offsets,
    const int* __restrict__ projector_columns,
    const double* __restrict__ projector_values,
    const double* __restrict__ projector_coefficients,
    const long long coefficient_row_stride,
    const long long coefficient_column_stride,
    const int add_nonlocal,
    const double* __restrict__ vectors,
    const long long vector_row_stride,
    const long long vector_column_stride,
    const int width,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    const long long row =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }

    double accumulator[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int slot = 0; slot < slot_count; ++slot) {
        const long long metadata_offset =
            static_cast<long long>(slot) * row_count + row;
        const int source_row = neighbors[metadata_offset];
        if (source_row < 0) {
            continue;
        }
        const double coefficient = coefficient_palette[
            coefficient_codes[metadata_offset]
        ];
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                accumulator[column] += coefficient * vectors[
                    static_cast<long long>(source_row) * vector_row_stride +
                    static_cast<long long>(column) * vector_column_stride
                ];
            }
        }
    }
    if (add_local) {
        const double potential = local_potential[row];
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                accumulator[column] += potential * vectors[
                    row * vector_row_stride +
                    static_cast<long long>(column) * vector_column_stride
                ];
            }
        }
    }
    if (add_nonlocal) {
        double nonlocal_accumulator[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (
            int position = projector_row_offsets[row];
            position < projector_row_offsets[row + 1];
            ++position
        ) {
            const int projector = projector_columns[position];
            const double projector_value = projector_values[position];
#pragma unroll
            for (int column = 0; column < 6; ++column) {
                if (column < width) {
                    nonlocal_accumulator[column] += projector_value *
                        projector_coefficients[
                            static_cast<long long>(projector) * coefficient_row_stride +
                            static_cast<long long>(column) * coefficient_column_stride
                        ];
                }
            }
        }
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                accumulator[column] += nonlocal_accumulator[column];
            }
        }
    }
#pragma unroll
    for (int column = 0; column < 6; ++column) {
        if (column < width) {
            output[
                row * output_row_stride +
                static_cast<long long>(column) * output_column_stride
            ] = accumulator[column];
        }
    }
}

extern "C" __global__
void stencil_major_chebyshev6(
    const long long row_count,
    const int slot_count,
    const int* __restrict__ neighbors,
    const unsigned char* __restrict__ coefficient_codes,
    const double* __restrict__ coefficient_palette,
    const double* __restrict__ local_potential,
    const int* __restrict__ projector_row_offsets,
    const int* __restrict__ projector_columns,
    const double* __restrict__ projector_values,
    const double* __restrict__ projector_coefficients,
    const long long coefficient_row_stride,
    const long long coefficient_column_stride,
    const int add_nonlocal,
    const double* __restrict__ current,
    const long long current_row_stride,
    const long long current_column_stride,
    const double* __restrict__ previous,
    const long long previous_row_stride,
    const long long previous_column_stride,
    const int add_previous,
    const int width,
    const double center_argument,
    const double scale_argument,
    const double sigma_argument,
    const double sigma_next_argument,
    const double* __restrict__ recurrence_parameters,
    const int parameter_step,
    const int use_parameters,
    double* __restrict__ output,
    const long long output_row_stride,
    const long long output_column_stride
) {
    const double center = use_parameters ? recurrence_parameters[4 * parameter_step] : center_argument;
    const double scale = use_parameters ? recurrence_parameters[4 * parameter_step + 1] : scale_argument;
    const double sigma = use_parameters ? recurrence_parameters[4 * parameter_step + 2] : sigma_argument;
    const double sigma_next = use_parameters ? recurrence_parameters[4 * parameter_step + 3] : sigma_next_argument;
    const long long row =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }

    double accumulator[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int slot = 0; slot < slot_count; ++slot) {
        const long long metadata_offset =
            static_cast<long long>(slot) * row_count + row;
        const int source_row = neighbors[metadata_offset];
        if (source_row < 0) {
            continue;
        }
        const double coefficient = coefficient_palette[
            coefficient_codes[metadata_offset]
        ];
#pragma unroll
        for (int column = 0; column < 6; ++column) {
            if (column < width) {
                accumulator[column] += coefficient * current[
                    static_cast<long long>(source_row) * current_row_stride +
                    static_cast<long long>(column) * current_column_stride
                ];
            }
        }
    }

    double nonlocal_accumulator[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (add_nonlocal) {
        for (
            int position = projector_row_offsets[row];
            position < projector_row_offsets[row + 1];
            ++position
        ) {
            const int projector = projector_columns[position];
            const double projector_value = projector_values[position];
#pragma unroll
            for (int column = 0; column < 6; ++column) {
                if (column < width) {
                    nonlocal_accumulator[column] += projector_value *
                        projector_coefficients[
                            static_cast<long long>(projector) * coefficient_row_stride +
                            static_cast<long long>(column) * coefficient_column_stride
                        ];
                }
            }
        }
    }

    const double potential = local_potential[row];
#pragma unroll
    for (int column = 0; column < 6; ++column) {
        if (column < width) {
            const long long current_offset =
                row * current_row_stride +
                static_cast<long long>(column) * current_column_stride;
            double value = accumulator[column] + potential * current[current_offset];
            if (add_nonlocal) {
                value += nonlocal_accumulator[column];
            }
            value = (value - center * current[current_offset]) * scale;
            if (add_previous) {
                const long long previous_offset =
                    row * previous_row_stride +
                    static_cast<long long>(column) * previous_column_stride;
                value -= sigma * previous[previous_offset];
            }
            value *= sigma_next;
            output[
                row * output_row_stride +
                static_cast<long long>(column) * output_column_stride
            ] = value;
        }
    }
}
"""


_KERNEL_CACHE: dict[Any, Any] = {}
_KERNEL_CACHE_LOCK = Lock()


def _compiled_kernels(cp: Any, slot_count: int) -> StencilKernels:
    """Compile each device's invariant stencil kernels exactly once.

    Every full-grid or representation operator uses the same CUDA source and
    launch ABI.  Recreating and explicitly compiling identical ``RawKernel``
    objects for all eight irreps only repeated module setup; sharing them does
    not share any numerical data or alter a kernel launch.  Returns the apply
    kernel, the recurrence kernel and the former recurrence kernel
    (:func:`.stencil_pipeline.compiled_kernels`): the former pair is kept per
    device, as it always was, and the recurrence kernel of
    ``PARSEC_CUPY_STENCIL_KERNEL=pipelined`` per device and slot count beside
    it, or why it could not be built.
    """

    return compiled_kernels(
        cp,
        _KERNEL_CACHE,
        _KERNEL_CACHE_LOCK,
        int(cp.cuda.Device().id),
        _CUDA_SOURCE,
        0,
        slot_count,
        stencil_kernel_setting(),
        compile_cupy_raw,
    )


def _stencil_major_metadata(
    matrix: sp.spmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transpose canonical CSR rows into contiguous per-slot arrays."""

    host = sp.csr_matrix(matrix, dtype=np.float64)
    host.sum_duplicates()
    host.sort_indices()
    if host.shape[0] != host.shape[1]:
        raise ValueError("finite-difference operator must be square")
    if host.shape[0] > np.iinfo(np.int32).max:
        raise ValueError("stencil-major CUDA storage requires int32 grid rows")
    encoded = _coefficient_palette(host.data)
    if encoded is None:
        raise ValueError("finite-difference operator has more than 256 coefficients")
    palette, csr_codes = encoded

    row_count = int(host.shape[0])
    row_widths = np.diff(host.indptr)
    slot_count = int(row_widths.max(initial=0))
    if slot_count < 1:
        raise ValueError("finite-difference operator contains no entries")

    # C-order (slot, row) arrays flatten exactly as slot*row_count + row,
    # giving coalesced reads to adjacent CUDA row threads.
    neighbors = np.full((slot_count, row_count), -1, dtype=np.int32)
    codes = np.zeros((slot_count, row_count), dtype=np.uint8)
    row_starts = host.indptr[:-1]
    for slot in range(slot_count):
        active = row_widths > slot
        positions = row_starts[active] + slot
        neighbors[slot, active] = host.indices[positions].astype(
            np.int32, copy=False
        )
        codes[slot, active] = csr_codes[positions]
    return neighbors, codes, palette


@dataclass(frozen=True)
class StencilMajorHostMetadata:
    """GPU-ready host representation of one canonical stencil matrix."""

    shape: tuple[int, int]
    neighbors: np.ndarray
    coefficient_codes: np.ndarray
    coefficient_palette: np.ndarray

    def __post_init__(self) -> None:
        rows, columns = map(int, self.shape)
        if rows < 1 or rows != columns:
            raise ValueError("stencil metadata shape must be positive and square")
        neighbors = np.asarray(self.neighbors, dtype=np.int32)
        codes = np.asarray(self.coefficient_codes, dtype=np.uint8)
        palette = np.asarray(self.coefficient_palette, dtype=np.float64)
        if neighbors.ndim != 2 or neighbors.shape[1] != rows:
            raise ValueError("stencil neighbor metadata has an invalid shape")
        if codes.shape != neighbors.shape:
            raise ValueError("stencil coefficient codes do not match neighbors")
        if palette.ndim != 1 or not 1 <= palette.size <= 256:
            raise ValueError("stencil coefficient palette has an invalid shape")
        active_codes = codes[neighbors >= 0]
        if active_codes.size and int(active_codes.max()) >= palette.size:
            raise ValueError("stencil metadata contains an invalid coefficient code")
        object.__setattr__(
            self, "neighbors", np.ascontiguousarray(neighbors, dtype=np.int32)
        )
        object.__setattr__(
            self,
            "coefficient_codes",
            np.ascontiguousarray(codes, dtype=np.uint8),
        )
        object.__setattr__(
            self,
            "coefficient_palette",
            np.ascontiguousarray(palette, dtype=np.float64),
        )

    def to_csr(self) -> sp.csr_matrix:
        """Reconstruct canonical row order for a non-CUDA fallback."""

        active = self.neighbors >= 0
        row_widths = np.count_nonzero(active, axis=0)
        indptr = np.empty(self.shape[0] + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(row_widths, out=indptr[1:])
        indices = self.neighbors.T[active.T]
        codes = self.coefficient_codes.T[active.T]
        data = self.coefficient_palette[codes]
        return sp.csr_matrix((data, indices, indptr), shape=self.shape)


def build_stencil_major_metadata(
    matrix: sp.spmatrix,
) -> StencilMajorHostMetadata:
    """Create exact GPU-ready metadata from one canonical sparse matrix."""

    host = sp.csr_matrix(matrix, dtype=np.float64)
    neighbors, codes, palette = _stencil_major_metadata(host)
    return StencilMajorHostMetadata(
        shape=tuple(map(int, host.shape)),
        neighbors=neighbors,
        coefficient_codes=codes,
        coefficient_palette=palette,
    )


class CuPyStencilMajorFiniteDifference:
    """Device-resident coalesced stencil and fused recurrence kernels."""

    chunk_width = 6
    supports_fused_projector_scatter = True

    def __init__(
        self,
        cp: Any,
        matrix: sp.spmatrix | None = None,
        *,
        metadata: StencilMajorHostMetadata | None = None,
        device_neighbors: Any | None = None,
        implicit_tile: int | None = None,
    ) -> None:
        if metadata is None:
            if matrix is None:
                raise ValueError("matrix or stencil metadata is required")
            metadata = build_stencil_major_metadata(matrix)
        neighbors = metadata.neighbors
        codes = metadata.coefficient_codes
        palette = metadata.coefficient_palette
        self.cp = cp
        self.shape = metadata.shape
        self.slot_count = int(neighbors.shape[0])
        self.palette_size = int(palette.size)
        if implicit_tile is None:
            # The symmetry eigensolver names the tile of a sector.  Any
            # other stencil is packed by an explicit size alone.
            from .implicit_stencil import implicit_tile_setting
            implicit_tile = implicit_tile_setting() or 0
        if implicit_tile:
            from .implicit_stencil import initialize_implicit
            initialize_implicit(self, cp, metadata, implicit_tile)
            return
        if neighbors.ndim != 2:
            raise ValueError("a stencil packed into tiles has no slot-major layout")
        if device_neighbors is None:
            self.neighbors = cp.asarray(neighbors)
        else:
            if tuple(device_neighbors.shape) != tuple(neighbors.shape):
                raise ValueError("shared device neighbors have an invalid shape")
            if device_neighbors.dtype != cp.dtype(cp.int32):
                raise ValueError("shared device neighbors must use int32")
            self.neighbors = device_neighbors
        self.coefficient_codes = cp.asarray(codes)
        self.coefficient_palette = cp.asarray(palette, dtype=cp.float64)
        self.hold_kernels(_compiled_kernels(cp, self.slot_count))

    def hold_kernels(self, kernels: StencilKernels | tuple[Any, Any, Any]) -> None:
        """Keep the apply, recurrence and former recurrence kernels of the stencil.

        ``kernels`` is what :func:`.stencil_pipeline.compiled_kernels`
        returns; a plain triple is the former kernels.
        ``stencil_major_spmm6`` (``apply_kernel``) is the former kernel under
        every value of ``PARSEC_CUPY_STENCIL_KERNEL``.  Under ``pipelined``
        ``recurrence_kernel`` holds the fast path for six columns of unit row
        stride, generated for the slot count of this stencil
        (``fast_path_slots``), and ``former_recurrence_kernel`` the former
        kernel for the other launches; under ``former`` both are the former
        kernel and ``fast_path_slots`` is None.  So they are under
        ``pipelined`` where the generated kernel could not be built:
        ``stencil_kernel_fallback`` then says why (it is None otherwise).
        ``stencil_kernel_setting`` is the value the kernels were compiled for.

        Every launch of the stencil names its slot count, the one part of the
        guard of the fast path that all of them share: it is compared with
        the slot count of the fast path here, once.  A kernel generated for
        another slot count would take its former loop in every launch; the
        stencil then holds the former kernel alone.
        """

        held = StencilKernels(*kernels)
        self.apply_kernel = held.apply
        self.former_recurrence_kernel = held.former_recurrence
        fallback = held.fallback
        fast = held.recurrence is not held.former_recurrence
        if fast and held.fast_path_slots != int(self.slot_count):
            fast = False
            fallback = fallback or (
                f"the generated kernel has the fast path of "
                f"{held.fast_path_slots} slots, the stencil {int(self.slot_count)}"
            )
        self.recurrence_kernel = held.recurrence if fast else held.former_recurrence
        self.fast_path_slots = int(self.slot_count) if fast else None
        self.stencil_kernel_setting = held.setting
        self.stencil_kernel_fallback = fallback

    def recurrence_kernel_for(self, width: int, row_stride: int) -> Any:
        """The kernel of one launch of the recurrence, of ``width`` columns and ``row_stride``.

        Without a fast path (``fast_path_slots`` is None, as under
        ``former``) that is ``recurrence_kernel``, as before.  With one, a
        launch that the fast path takes (the guard of
        :func:`.stencil_pipeline.takes_fast_path`: six columns, unit row
        stride, the slot count it was generated for, which
        :meth:`hold_kernels` compared with that of the stencil) goes to
        ``recurrence_kernel``, and every other launch to the former kernel:
        inside the generated kernel it would take the former loop with the
        same bits, held at the registers of the fast path, 2 to 6 % slower
        than the former kernel on A100.  The choice is made in Python when
        the launch is made, and so when a filter graph records it;
        :meth:`chebyshev_recurrence` makes the same choice in its loop of
        launches, with plain integers and no call.
        """

        if self.fast_path_slots is None or (
            int(width) == WIDTH and int(row_stride) == 1
        ):
            return self.recurrence_kernel
        return self.former_recurrence_kernel

    @property
    def stencil_kernel_record(self) -> str:
        """Which recurrence kernels the stencil launches, for the backend details of a run.

        The value of ``PARSEC_CUPY_STENCIL_KERNEL`` that its kernels were
        compiled for, an unset variable included: ``former``; ``pipelined``
        with the slot count of the fast path (``pipelined: fast path for 25
        slots``: the generated kernel for the launches of six columns of unit
        row stride, the former one for the others); or ``pipelined`` with the
        reason why the stencil holds the former kernels instead
        (``stencil_kernel_fallback``).
        """

        if self.fast_path_slots is not None:
            return (
                f"{self.stencil_kernel_setting}: fast path for "
                f"{self.fast_path_slots} slots"
            )
        if self.stencil_kernel_fallback:
            return (
                f"{self.stencil_kernel_setting}: former kernels "
                f"({self.stencil_kernel_fallback})"
            )
        return self.stencil_kernel_setting

    @property
    def storage_mode(self) -> str:
        if hasattr(self, "implicit_statistics"):
            return f"implicit_affine_tile_{self.implicit_statistics['tile']}"
        return "stencil_major_int32_neighbors_uint8_coefficient_palette"

    @property
    def stencil_kernel_attributes(self) -> dict[str, dict[str, int]]:
        """Registers and local memory of the kernels, by the attribute that holds each.

        ``num_regs`` and ``local_size_bytes`` (where registers spill, the
        spill is local memory) of ``apply_kernel``, ``recurrence_kernel`` and
        ``former_recurrence_kernel`` as the driver reports them for the
        current device.  A stand-in kernel that reports no attributes gives
        an empty entry.
        """

        found = {}
        for name in ("apply_kernel", "recurrence_kernel", "former_recurrence_kernel"):
            attributes = getattr(getattr(self, name), "attributes", None) or {}
            found[name] = {
                key: int(attributes[key])
                for key in ("num_regs", "local_size_bytes")
                if key in attributes
            }
        return found

    def _columns(self, vectors: Any) -> tuple[Any, bool]:
        cp = self.cp
        block = (
            vectors
            if isinstance(vectors, cp.ndarray)
            and vectors.dtype == cp.dtype(cp.float64)
            else cp.asarray(vectors, dtype=cp.float64)
        )
        was_vector = block.ndim == 1
        if was_vector:
            block = block[:, None]
        if block.ndim != 2 or block.shape[0] != self.shape[0]:
            raise ValueError("vectors do not match the stencil")
        # Both raw kernels consume explicit row/column strides.  Preserve an
        # existing device view instead of allocating and copying it merely to
        # impose Fortran order; recurrence outputs themselves remain
        # Fortran-contiguous for the coalesced steady-state path.
        return block, was_vector

    def apply(
        self,
        vectors: Any,
        local_potential: Any | None = None,
        *,
        projector_data: tuple[Any, Any, Any] | None = None,
        projector_coefficients: Any | None = None,
        output: Any | None = None,
    ):
        """Apply stencil, local field, and optional low-rank KB scatter.

        The projector coefficients are ``diag(signs) @ B.T @ vectors``.  The
        kernel evaluates the remaining CSR ``B @ coefficients`` row loop in
        the same pass as the stencil, avoiding an N-by-block temporary and a
        second sparse launch while retaining the separable KB operator.
        """

        cp = self.cp
        block, was_vector = self._columns(vectors)
        if local_potential is None:
            add_local = 0
            potential = self.coefficient_palette
        else:
            potential = cp.asarray(local_potential, dtype=cp.float64)
            if potential.shape != (self.shape[0],):
                raise ValueError("local potential does not match the stencil")
            add_local = 1

        if projector_data is None or projector_coefficients is None:
            projector_row_offsets = self.neighbors
            projector_columns = self.neighbors
            projector_values = self.coefficient_palette
            coefficient_block = block
            add_nonlocal = 0
        else:
            projector_row_offsets, projector_columns, projector_values = projector_data
            coefficient_block = cp.asarray(projector_coefficients, dtype=cp.float64)
            if coefficient_block.ndim == 1:
                coefficient_block = coefficient_block[:, None]
            if coefficient_block.ndim != 2 or coefficient_block.shape[1] != block.shape[1]:
                raise ValueError("projector coefficients do not match the vector block")
            add_nonlocal = 1

        if output is None:
            output_block = cp.empty(block.shape, dtype=cp.float64, order="F")
        else:
            output_block = output
            if (
                not isinstance(output_block, cp.ndarray)
                or output_block.dtype != cp.dtype(cp.float64)
                or output_block.shape != block.shape
            ):
                raise ValueError(
                    "output must be a float64 device array matching vectors"
                )
            if output_block.data.ptr == block.data.ptr:
                raise ValueError("output must not alias the input vectors")
        threads = 256
        grid = ((self.shape[0] + threads - 1) // threads,)
        itemsize = int(block.dtype.itemsize)
        for start in range(0, int(block.shape[1]), self.chunk_width):
            stop = min(start + self.chunk_width, int(block.shape[1]))
            source = block[:, start:stop]
            coefficient_source = coefficient_block[:, start:stop]
            target = output_block[:, start:stop]
            self.apply_kernel(
                grid,
                (threads,),
                (
                    np.int64(self.shape[0]),
                    np.int32(self.slot_count),
                    self.neighbors,
                    self.coefficient_codes,
                    self.coefficient_palette,
                    potential,
                    np.int32(add_local),
                    projector_row_offsets,
                    projector_columns,
                    projector_values,
                    coefficient_source,
                    np.int64(coefficient_source.strides[0] // itemsize),
                    np.int64(coefficient_source.strides[1] // itemsize),
                    np.int32(add_nonlocal),
                    source,
                    np.int64(source.strides[0] // itemsize),
                    np.int64(source.strides[1] // itemsize),
                    np.int32(stop - start),
                    target,
                    np.int64(target.strides[0] // itemsize),
                    np.int64(target.strides[1] // itemsize),
                ),
            )
        return output_block[:, 0] if was_vector else output_block

    def chebyshev_recurrence(
        self,
        current: Any,
        local_potential: Any,
        *,
        center: float,
        scale: float,
        sigma_next: float,
        previous: Any | None = None,
        sigma: float = 0.0,
        projector_data: tuple[Any, Any, Any] | None = None,
        projector_coefficients: Any | None = None,
        output: Any | None = None,
        recurrence_parameters: Any | None = None,
        parameter_step: int = 0,
    ):
        """Fuse one normalized Chebyshev step with the local Hamiltonian."""

        cp = self.cp
        block, was_vector = self._columns(current)
        potential = (
            local_potential
            if isinstance(local_potential, cp.ndarray)
            and local_potential.dtype == cp.dtype(cp.float64)
            else cp.asarray(local_potential, dtype=cp.float64)
        )
        if potential.shape != (self.shape[0],):
            raise ValueError("local potential does not match the stencil")

        if previous is None:
            previous_block = block
            add_previous = 0
        else:
            previous_block, previous_was_vector = self._columns(previous)
            if previous_was_vector != was_vector or previous_block.shape != block.shape:
                raise ValueError("previous and current recurrence blocks must match")
            add_previous = 1

        if projector_data is None or projector_coefficients is None:
            projector_row_offsets = self.neighbors
            projector_columns = self.neighbors
            projector_values = self.coefficient_palette
            coefficient_block = block
            add_nonlocal = 0
        else:
            projector_row_offsets, projector_columns, projector_values = projector_data
            coefficient_block = (
                projector_coefficients
                if isinstance(projector_coefficients, cp.ndarray)
                and projector_coefficients.dtype == cp.dtype(cp.float64)
                else cp.asarray(projector_coefficients, dtype=cp.float64)
            )
            if coefficient_block.ndim == 1:
                coefficient_block = coefficient_block[:, None]
            if coefficient_block.ndim != 2 or coefficient_block.shape[1] != block.shape[1]:
                raise ValueError("projector coefficients do not match the current block")
            add_nonlocal = 1

        if output is None:
            output = cp.empty(block.shape, dtype=cp.float64, order="F")
        elif output.shape != block.shape or output.dtype != cp.dtype(cp.float64):
            raise ValueError("recurrence output must match the float64 block")
        # The block size that the launch bounds of the generated kernel name.
        threads = THREADS
        grid = ((self.shape[0] + threads - 1) // threads,)
        itemsize = int(block.dtype.itemsize)
        # The kernel of each launch, as recurrence_kernel_for chooses it,
        # without a call per launch: every launch takes recurrence_kernel
        # where the stencil has no fast path (fast_path_slots None, as under
        # former); otherwise a launch of six columns of unit row stride takes
        # it and every other launch the former kernel.  hold_kernels compared
        # the slot count.
        fast_path = self.fast_path_slots is not None
        recurrence_kernel = self.recurrence_kernel
        former_recurrence_kernel = self.former_recurrence_kernel
        for start in range(0, int(block.shape[1]), self.chunk_width):
            stop = min(start + self.chunk_width, int(block.shape[1]))
            source = block[:, start:stop]
            previous_source = previous_block[:, start:stop]
            coefficient_source = coefficient_block[:, start:stop]
            target = output[:, start:stop]
            source_row_stride = source.strides[0] // itemsize
            kernel = recurrence_kernel
            if fast_path and (stop - start != WIDTH or source_row_stride != 1):
                kernel = former_recurrence_kernel
            kernel(
                grid,
                (threads,),
                (
                    np.int64(self.shape[0]),
                    np.int32(self.slot_count),
                    self.neighbors,
                    self.coefficient_codes,
                    self.coefficient_palette,
                    potential,
                    projector_row_offsets,
                    projector_columns,
                    projector_values,
                    coefficient_source,
                    np.int64(coefficient_source.strides[0] // itemsize),
                    np.int64(coefficient_source.strides[1] // itemsize),
                    np.int32(add_nonlocal),
                    source,
                    np.int64(source_row_stride),
                    np.int64(source.strides[1] // itemsize),
                    previous_source,
                    np.int64(previous_source.strides[0] // itemsize),
                    np.int64(previous_source.strides[1] // itemsize),
                    np.int32(add_previous),
                    np.int32(stop - start),
                    np.float64(center),
                    np.float64(scale),
                    np.float64(sigma),
                    np.float64(sigma_next),
                    self.coefficient_palette if recurrence_parameters is None else recurrence_parameters,
                    np.int32(parameter_step),
                    np.int32(recurrence_parameters is not None),
                    target,
                    np.int64(target.strides[0] // itemsize),
                    np.int64(target.strides[1] // itemsize),
                ),
            )
        return output[:, 0] if was_vector else output

__all__ = [
    "CuPyStencilMajorFiniteDifference",
    "StencilMajorHostMetadata",
    "build_stencil_major_metadata",
]
