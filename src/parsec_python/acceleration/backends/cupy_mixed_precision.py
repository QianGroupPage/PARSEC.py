"""Optional FP32 Chebyshev recurrence with an FP64 DFT outer algorithm.

Consumer NVIDIA GPUs execute FP32 much faster than FP64.  The Chebyshev
filter is a particularly suitable mixed-precision boundary: it constructs a
subspace rather than a reported observable, and the resulting vectors are
immediately converted back to FP64 before Rayleigh--Ritz, density formation,
SCF convergence tests, and energy evaluation.

This module deliberately owns only the recurrence work.  It mirrors the
production stencil-major and separable Kleinman--Bylander projector kernels,
using FP32 values and vectors while preserving their sparse traversal and
PARSEC polynomial.  It is an opt-in: with ``PARSEC_CUPY_MIXED_FILTER`` unset
every filter runs in the FP64 kernels (see :func:`mixed_filter_policy`).
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

import numpy as np

from .cupy_compile import compile_cupy_raw
from .cupy_launch import kernel_parameters, launch_arguments
from .cupy_projectors import _SOURCE as _PROJECTOR_SOURCE
from .cupy_stencil_major import _CUDA_SOURCE as _STENCIL_SOURCE


_MIXED_FILTER_SETTINGS = {
    "off": "off", "0": "off", "false": "off", "no": "off",
    "auto": "auto",
    "on": "on", "1": "on", "true": "on", "yes": "on",
}


def mixed_filter_policy() -> str:
    """Return ``PARSEC_CUPY_MIXED_FILTER`` as ``off``, ``auto`` or ``on``.

    ``off``, the default, keeps every filter in FP64.  ``auto`` asks for the
    FP32 later filter in operators of at least
    ``PARSEC_CUPY_MIXED_FILTER_MIN_ROWS`` rows, ``on`` in every operator.
    """

    value = os.environ.get("PARSEC_CUPY_MIXED_FILTER", "off").strip().lower()
    try:
        return _MIXED_FILTER_SETTINGS[value]
    except KeyError:
        raise ValueError(
            "PARSEC_CUPY_MIXED_FILTER must be auto, on, or off"
        ) from None


def mixed_filter_minimum_rows() -> int:
    """Rows from which ``auto`` asks for the FP32 later filter."""

    raw = os.environ.get("PARSEC_CUPY_MIXED_FILTER_MIN_ROWS", "100000").strip()
    try:
        minimum_rows = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_MIXED_FILTER_MIN_ROWS must be an integer"
        ) from error
    if minimum_rows < 1:
        raise ValueError("PARSEC_CUPY_MIXED_FILTER_MIN_ROWS must be positive")
    return minimum_rows


def float32_filter_requested(rows: int) -> bool:
    """Whether an operator of ``rows`` rows is to own an FP32 recurrence."""

    policy = mixed_filter_policy()
    minimum_rows = mixed_filter_minimum_rows()
    return policy == "on" or (policy == "auto" and int(rows) >= minimum_rows)


def _float_source(source: str, names: tuple[str, ...]) -> str:
    """Return a separately named FP32 version of one CUDA source string."""

    converted = source
    # Use placeholders because ``..._projection`` is a prefix of
    # ``..._projection_serial``; a direct series of replacements would rename
    # part of the already-renamed longer identifier a second time.
    placeholders: list[tuple[str, str]] = []
    for index, name in enumerate(sorted(names, key=len, reverse=True)):
        placeholder = f"PARSEC_FLOAT_KERNEL_{index}"
        converted = converted.replace(name, placeholder)
        placeholders.append((placeholder, f"{name}_float32"))
    for placeholder, replacement in placeholders:
        converted = converted.replace(placeholder, replacement)
    return converted.replace("double", "float")


_SOURCE = _float_source(
    _STENCIL_SOURCE,
    ("stencil_major_spmm6", "stencil_major_chebyshev6"),
) + _float_source(
    _PROJECTOR_SOURCE,
    (
        "sparse_projector_projection",
        "sparse_projector_projection_serial",
        "sparse_projector_scatter",
    ),
)

_KERNEL_NAMES = (
    "stencil_major_chebyshev6_float32",
    "sparse_projector_projection_float32",
    "sparse_projector_projection_serial_float32",
)
_KERNEL_CACHE: dict[int, tuple[Any, Any, Any]] = {}
_KERNEL_LOCK = Lock()


def _kernels(cp: Any) -> tuple[Any, Any, Any]:
    """Compile the invariant FP32 recurrence kernels once per CUDA device."""

    device_id = int(cp.cuda.Device().id)
    with _KERNEL_LOCK:
        kernels = _KERNEL_CACHE.get(device_id)
        if kernels is None:
            module = cp.RawModule(
                code=_SOURCE,
                options=("--std=c++11",),
                name_expressions=_KERNEL_NAMES,
            )
            compile_cupy_raw(module)
            kernels = tuple(module.get_function(name) for name in _KERNEL_NAMES)
            _KERNEL_CACHE[device_id] = kernels
    return kernels


def _strides(prefix: str, array: Any) -> dict[str, int]:
    """Row and column stride of ``array`` in elements, as the kernels name them."""

    itemsize = int(array.dtype.itemsize)
    return {
        f"{prefix}_row_stride": array.strides[0] // itemsize,
        f"{prefix}_column_stride": array.strides[1] // itemsize,
    }


class CuPyMixedPrecisionRecurrence:
    """FP32 stencil/projector recurrence used only inside Chebyshev filters."""

    chunk_width = 6

    def __init__(
        self,
        cp: Any,
        stencil: Any,
        host_projectors: Any,
        projector_signs: np.ndarray,
        effective_potential: np.ndarray,
    ) -> None:
        if hasattr(stencil, "implicit_statistics"):
            # The FP32 kernel is compiled from the slot-major source and
            # would read the packed tile descriptors as neighbor rows.
            raise ValueError(
                "the FP32 recurrence does not read a stencil packed into "
                "implicit tiles; set PARSEC_CUPY_IMPLICIT_TILE=0"
            )
        host = host_projectors.tocsr(copy=True)
        host.sum_duplicates()
        host.sort_indices()
        host_transpose = host.T.tocsr(copy=True)
        host_transpose.sum_duplicates()
        host_transpose.sort_indices()
        if host.indices.dtype != np.int32 or host.indptr.dtype != np.int32:
            raise ValueError("mixed projector factors require int32 host CSR")
        if (
            host_transpose.indices.dtype != np.int32
            or host_transpose.indptr.dtype != np.int32
        ):
            raise ValueError("mixed projector transpose requires int32 host CSR")

        self.cp = cp
        self.shape = tuple(stencil.shape)
        self.slot_count = int(stencil.slot_count)
        # Integer stencil metadata is immutable and safe to share with FP64.
        self.neighbors = stencil.neighbors
        self.coefficient_codes = stencil.coefficient_codes
        self.coefficient_palette = stencil.coefficient_palette.astype(
            cp.float32
        )
        self.effective_potential = cp.asarray(
            effective_potential, dtype=cp.float32
        )

        self.projector_count = int(host.shape[1])
        self.projector_row_offsets = cp.asarray(host.indptr, dtype=cp.int32)
        self.projector_columns = cp.asarray(host.indices, dtype=cp.int32)
        self.projector_values = cp.asarray(host.data, dtype=cp.float32)
        self.transpose_row_offsets = cp.asarray(
            host_transpose.indptr, dtype=cp.int32
        )
        self.transpose_grid_rows = cp.asarray(
            host_transpose.indices, dtype=cp.int32
        )
        self.transpose_values = cp.asarray(
            host_transpose.data, dtype=cp.float32
        )
        self.projector_signs = cp.asarray(projector_signs, dtype=cp.float32)
        row_lengths = np.diff(host_transpose.indptr)
        self.parallel_projection = bool(
            row_lengths.size and int(row_lengths.max(initial=0)) >= 256
        )

        kernels = _kernels(cp)
        selected = (0, 1 if self.parallel_projection else 2)
        self.recurrence_kernel, self.projection_kernel = (
            kernels[index] for index in selected
        )
        # The FP32 kernels are compiled from the FP64 sources, so their
        # parameter lists move with those.  The launches name their values
        # and are ordered by these declarations.
        self.recurrence_parameters, self.projection_parameters = (
            kernel_parameters(_SOURCE, _KERNEL_NAMES[index])
            for index in selected
        )

    def update_potential(self, effective_potential: Any) -> None:
        """Refresh the FP32 shadow of the current FP64 SCF local field."""

        cp = self.cp
        potential = cp.asarray(effective_potential, dtype=cp.float32)
        if potential.shape != self.effective_potential.shape:
            raise ValueError("effective_potential does not match mixed operator")
        cp.copyto(self.effective_potential, potential)

    def _columns(self, vectors: Any) -> tuple[Any, bool]:
        cp = self.cp
        block = (
            vectors
            if isinstance(vectors, cp.ndarray)
            and vectors.dtype == cp.dtype(cp.float32)
            else cp.asarray(vectors, dtype=cp.float32)
        )
        was_vector = block.ndim == 1
        if was_vector:
            block = block[:, None]
        if block.ndim != 2 or block.shape[0] != self.shape[0]:
            raise ValueError("vectors do not match the mixed operator")
        return block, was_vector

    def _signed_projector_coefficients(self, vectors: Any):
        cp = self.cp
        width = int(vectors.shape[1])
        output = cp.empty(
            (self.projector_count, width), dtype=cp.float32, order="F"
        )
        if self.projector_count == 0:
            return output
        threads = 128
        pair_count = self.projector_count * width
        grid = (
            (pair_count,)
            if self.parallel_projection
            else ((pair_count + threads - 1) // threads,)
        )
        values = dict(
            projector_count=self.projector_count,
            width=width,
            row_offsets=self.transpose_row_offsets,
            grid_rows=self.transpose_grid_rows,
            projector_values=self.transpose_values,
            signs=self.projector_signs,
            vectors=vectors,
            **_strides("vector", vectors),
            output=output,
            **_strides("output", output),
        )
        self.projection_kernel(
            grid,
            (threads,),
            launch_arguments(self.projection_parameters, values),
        )
        return output

    def __call__(
        self,
        current: Any,
        *,
        center: float,
        scale: float,
        sigma_next: float,
        previous: Any | None = None,
        sigma: float = 0.0,
    ):
        """Execute one normalized PARSEC recurrence in FP32."""

        cp = self.cp
        block, was_vector = self._columns(current)
        if previous is None:
            previous_block = block
            add_previous = 0
        else:
            previous_block, previous_was_vector = self._columns(previous)
            if previous_was_vector != was_vector or previous_block.shape != block.shape:
                raise ValueError("previous and current mixed blocks must match")
            add_previous = 1

        coefficients = self._signed_projector_coefficients(block)
        add_nonlocal = int(self.projector_count > 0)
        output = cp.empty(block.shape, dtype=cp.float32, order="F")
        threads = 256
        grid = ((self.shape[0] + threads - 1) // threads,)
        fixed = dict(
            row_count=self.shape[0],
            slot_count=self.slot_count,
            neighbors=self.neighbors,
            coefficient_codes=self.coefficient_codes,
            coefficient_palette=self.coefficient_palette,
            local_potential=self.effective_potential,
            projector_row_offsets=self.projector_row_offsets,
            projector_columns=self.projector_columns,
            projector_values=self.projector_values,
            add_nonlocal=add_nonlocal,
            add_previous=add_previous,
            center_argument=center,
            scale_argument=scale,
            sigma_argument=sigma,
            sigma_next_argument=sigma_next,
            # The coefficient table of the filter graphs is not used here;
            # as in the FP64 launch, the palette stands in for it unread.
            recurrence_parameters=self.coefficient_palette,
            parameter_step=0,
            use_parameters=0,
        )
        for start in range(0, int(block.shape[1]), self.chunk_width):
            stop = min(start + self.chunk_width, int(block.shape[1]))
            source = block[:, start:stop]
            previous_source = previous_block[:, start:stop]
            coefficient_source = coefficients[:, start:stop]
            target = output[:, start:stop]
            values = dict(
                fixed,
                projector_coefficients=coefficient_source,
                **_strides("coefficient", coefficient_source),
                current=source,
                **_strides("current", source),
                previous=previous_source,
                **_strides("previous", previous_source),
                width=stop - start,
                output=target,
                **_strides("output", target),
            )
            self.recurrence_kernel(
                grid,
                (threads,),
                launch_arguments(self.recurrence_parameters, values),
            )
        return output[:, 0] if was_vector else output


__all__ = [
    "CuPyMixedPrecisionRecurrence",
    "float32_filter_requested",
    "mixed_filter_minimum_rows",
    "mixed_filter_policy",
]
