"""Bounded-workspace CUDA graphs for the unchanged FP64 block recurrence.

One operator owns three N-by-block buffers, one projector buffer and a small
coefficient table. Graphs share those buffers on an ordered stream. Updating
the table preserves the exact CPU sigma carry and changing SCF bounds without
recapturing. No graph retains a previous SCF input/output or orbital matrix.

By default one graph is recorded per block width and degree and serves every
block and every plan with that width and degree
(:func:`graph_reuse_requested`).
"""

from __future__ import annotations

import os
from math import prod
from threading import RLock
from typing import Any
from weakref import ref

import numpy as np

from ..backends.cupy import require_cupy
from ..backends.cupy_capture import capture_graph, collector_paused
from ..backends.cupy_stencil_major import CuPyStencilMajorFiniteDifference


def graph_reuse_requested() -> bool:
    """Whether the blocks of one width and degree share one recorded graph.

    The default.  The launches of two blocks of one width and degree differ
    only in the rows of the coefficient table that their steps read.  A
    shared graph reads a table of its own of ``degree`` rows instead, and the
    rows of a block are copied into it from the uploaded table before each
    launch, in order on the stream.  A workspace then records a graph once
    for every width and degree it meets and keeps it, and its recurrence
    buffers, from plan to plan.

    ``PARSEC_CUPY_FILTER_GRAPH_REUSE=0`` records one graph per block of a
    plan, and all of them again for a plan with other columns or degrees, as
    before.  The later degree falls by one per SCF step near convergence, so
    a run of ten steps recorded every graph six times (11,929 captures with
    23,768 electrons on 16 GPUs).  Recording is host work that the device
    threads of a process do one after another.  The kernels, the
    coefficients they read and their order are the same on both routes, and
    so are the filtered columns to the last bit.
    """

    value = os.environ.get("PARSEC_CUPY_FILTER_GRAPH_REUSE", "1").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_FILTER_GRAPH_REUSE must be on or off")
    return value in {"1", "on", "true"}


def graph_route_name(reuse: bool) -> str:
    """What a run reports for graphs recorded with this :func:`graph_reuse_requested`."""

    return "one per block width and degree" if reuse else "one per block of a plan"


def graph_filter(operator, matrix, blocks, lower, upper, reference, reset, out=None):
    """Return None for an unsupported/disabled path; otherwise exact filtering.

    ``out`` may be the input itself; see :meth:`BlockFilterGraphs.apply`.
    """
    if os.environ.get("PARSEC_CUPY_FILTER_GRAPHS", "0").lower() not in {
        "1",
        "on",
        "true",
    }:
        return None
    stencil = getattr(operator, "compact_finite_difference", None)
    if not isinstance(stencil, CuPyStencilMajorFiniteDifference):
        return None
    if operator.projector_count and (
        not operator.fused_projector_scatter
        or operator.custom_projector_projection is None
    ):
        return None
    if not blocks or max(b.stop - b.start for b in blocks) > 6:
        return None
    workspace = getattr(operator, "_filter_graph_workspace", None)
    if workspace is None:
        workspace = BlockFilterGraphs(operator)
        operator._filter_graph_workspace = workspace
    return workspace.apply(matrix, blocks, lower, upper, reference, reset, out=out)


def fill_recurrence_table(table, blocks, center, half_span, sigma_one, reset, initial_sigma=None):
    """Write the coefficients of every recurrence step of ``blocks`` into ``table``.

    One row per step: the center, the scale, sigma and the sigma that
    follows it.  Sigma is carried from block to block unless ``reset``, and
    ``initial_sigma`` is the one carried into the first block.  Where the
    blocks are not consecutive blocks of one filter plan, ``initial_sigma``
    is a sequence with the sigma carried into each of them instead (those of
    :func:`distributed_filter.starting_sigmas` for the plan), so that a block
    takes the steps it has in the plan wherever it lies.
    """

    starts = None
    if initial_sigma is not None and not np.isscalar(initial_sigma):
        starts = tuple(initial_sigma)
        if len(starts) != len(blocks):
            raise ValueError("one starting sigma per block is required")
    carried = None if starts is not None else initial_sigma
    offset = 0
    for index, block in enumerate(blocks):
        if starts is not None:
            carried = starts[index]
        sigma = sigma_one if reset or carried is None else carried
        table[offset] = (center, sigma_one / half_span, 0.0, 1.0)
        for step in range(1, block.degree):
            sigma_next = 1.0 / (2.0 / sigma_one - sigma)
            table[offset + step] = (
                center,
                2.0 / half_span,
                sigma,
                sigma_next,
            )
            sigma = sigma_next
        carried = sigma
        offset += block.degree


class BlockFilterGraphs:
    def __init__(self, operator: Any):
        cp, _ = require_cupy()
        self.cp = cp
        # The operator owns this cache. Avoid a cycle retaining large device
        # allocations after a prepared calculation is released.
        self._operator_ref = ref(operator)
        self.device = int(operator.effective_potential.device.id)
        self.lock = RLock()
        self.key = None
        self.finished = None
        self.uploaded = None
        self.graphs = []
        self.buffer_bytes = 0
        # The route of a workspace stays: the graphs of the two routes read
        # different tables.
        self.reuse = graph_reuse_requested()
        # With shared graphs, per block width and degree: the graph, the
        # buffer that holds its result and the table that its steps read.
        self.shared = {}
        self.buffers = None
        self.table = None
        self.has_upload = False
        self.has_finished = False

    @property
    def operator(self):
        operator = self._operator_ref()
        if operator is None:
            raise ReferenceError("filter graph owner has been released")
        return operator

    @property
    def recorded(self) -> bool:
        """Whether the workspace holds a graph that it has recorded."""

        return bool(self.graphs or self.shared)

    def _persistent_empty(self, shape, order="F"):
        # Long-lived small graph buffers must not split cached multi-GB
        # orbital allocations: CuPy cannot release a split pool block until
        # every piece is free. Allocate only this bounded persistent workspace
        # directly; temporary orbital matrices still use the normal allocator.
        cp = self.cp
        memory = cp.cuda.Memory(prod(shape) * np.dtype(np.float64).itemsize)
        pointer = cp.cuda.MemoryPointer(memory, 0)
        return cp.ndarray(shape, dtype=cp.float64, memptr=pointer, order=order)

    def _capture(self, size, degree, table, first):
        """Record ``degree`` recurrence steps on the first ``size`` columns of the buffers.

        Step ``k`` reads row ``first + k`` of ``table``.  Returns the graph
        and the buffer that holds its result.
        """

        cp = self.cp
        views = [x[:, :size] for x in self.buffers]
        coefficient_view = self.coefficients[:, :size]

        def record():
            previous, current = 0, 0
            for step in range(degree):
                target = (step + 1) % 3
                signed = None
                if self.operator.projector_count:
                    signed = self.operator.custom_projector_projection(
                        views[current], output=coefficient_view
                    )
                self.operator.compact_finite_difference.chebyshev_recurrence(
                    views[current],
                    self.operator.effective_potential,
                    center=0.0,
                    scale=0.0,
                    sigma=0.0,
                    sigma_next=1.0,
                    previous=None if step == 0 else views[previous],
                    projector_data=(
                        self.operator.projector_csr_data
                        if signed is not None
                        else None
                    ),
                    projector_coefficients=signed,
                    output=views[target],
                    recurrence_parameters=table,
                    parameter_step=first + step,
                )
                previous, current = current, target
            return current

        with self.capture_stream:
            graph, current = capture_graph(
                self.capture_stream,
                record,
                mode=cp.cuda.runtime.streamCaptureModeThreadLocal,
            )
        return graph, views[current]

    def _prepare(self, matrix, blocks):
        key = (int(matrix.shape[0]), tuple((b.start, b.stop, b.degree) for b in blocks))
        if key == self.key:
            return
        if self.reuse:
            self._prepare_shared(blocks, key)
        else:
            self._prepare_blocks(blocks, key)

    def _prepare_blocks(self, blocks, key):
        """A graph for every block of the plan ``key``, with buffers and a table of the plan."""

        cp = self.cp
        if self.finished is not None:
            self.finished.synchronize()
        # The graphs of the former key go here.  The key is set again once
        # every block has been captured, so that a capture that fails leaves
        # no key naming graphs that are gone.
        self.key = None
        self.graphs = []
        width = max(b.stop - b.start for b in blocks)
        self.buffers = [self._persistent_empty((key[0], width)) for _ in range(3)]
        self.coefficients = self._persistent_empty((self.operator.projector_count, width))
        steps = sum(b.degree for b in blocks)
        self.parameters = self._persistent_empty((steps, 4), order="C")
        self.pinned = cp.cuda.alloc_pinned_memory(steps * 4 * 8)
        self.host_parameters = np.frombuffer(
            self.pinned, dtype=np.float64, count=steps * 4
        ).reshape(steps, 4)
        self.uploaded = cp.cuda.Event(disable_timing=True)
        self.finished = cp.cuda.Event(disable_timing=True)
        self.has_upload = False
        self.has_finished = False
        self.capture_stream = cp.cuda.Stream(non_blocking=True)
        self.buffer_bytes = (
            sum(x.nbytes for x in self.buffers)
            + self.coefficients.nbytes
            + self.parameters.nbytes
        )
        self.operator.timing_stats.filter_graph_buffer_bytes = self.buffer_bytes
        offset = 0
        # A capture in this mode is invalidated when its own thread takes
        # memory from the driver or returns it, or destroys another graph.
        # The launches below do neither: all they use was allocated above.
        # A destructor can, and the cyclic garbage collector may run
        # destructors in this thread at any time, for instance those of the
        # buffers and graphs of an operator that was released in a reference
        # cycle.  The collector therefore stays off until the last capture
        # has ended.  The end of another thread of this device can still
        # invalidate a capture; capture_graph then records it again.
        with collector_paused():
            for block in blocks:
                # The steps of a block read their rows of the table of the plan.
                graph, output = self._capture(
                    block.stop - block.start, block.degree, self.parameters, offset
                )
                self.graphs.append((graph, output, None))
                offset += block.degree
        self.key = key
        self.operator.timing_stats.filter_graph_builds += len(blocks)

    def _prepare_shared(self, blocks, key):
        """The graphs of the plan ``key`` from those per block width and degree.

        What earlier plans left is kept: the buffers while they have the rows
        and the columns of the widest block, the uploaded table while it has
        a row for every step, and every graph.  Only a width and degree that
        no plan had before is recorded.
        """

        cp = self.cp
        rows = key[0]
        width = max(b.stop - b.start for b in blocks)
        steps = sum(b.degree for b in blocks)
        kinds = [(b.stop - b.start, b.degree) for b in blocks]
        # As in _prepare_blocks: no key names a plan whose graphs are not all there.
        self.key = None
        self.graphs = []
        if self.uploaded is None:
            self.uploaded = cp.cuda.Event(disable_timing=True)
            self.finished = cp.cuda.Event(disable_timing=True)
            # One stream records every graph of the workspace.
            self.capture_stream = cp.cuda.Stream(non_blocking=True)
        fits = (
            self.buffers is not None
            and self.buffers[0].shape[0] == rows
            and self.buffers[0].shape[1] >= width
        )
        enough = self.table is not None and self.table.shape[0] >= steps
        if not (fits and enough and all(kind in self.shared for kind in kinds)):
            # The device has done what was submitted before any of the
            # workspace is replaced and, as on the other route, before a
            # capture.
            if self.has_upload:
                self.uploaded.synchronize()
            if self.has_finished:
                self.finished.synchronize()
        if not fits:
            # Every graph launches on the buffers and goes with them.
            self.shared = {}
            self.buffers = self.coefficients = None
            buffers = [self._persistent_empty((rows, width)) for _ in range(3)]
            self.coefficients = self._persistent_empty((self.operator.projector_count, width))
            self.buffers = buffers
        if not enough:
            pinned = cp.cuda.alloc_pinned_memory(steps * 4 * 8)
            self.table = self._persistent_empty((steps, 4), order="C")
            self.pinned = pinned
            self.host_table = np.frombuffer(
                self.pinned, dtype=np.float64, count=steps * 4
            ).reshape(steps, 4)
        # The rows of this plan; no graph reads them where they are.
        self.parameters = self.table[:steps]
        self.host_parameters = self.host_table[:steps]
        missing = [kind for kind in dict.fromkeys(kinds) if kind not in self.shared]
        tables = [self._persistent_empty((degree, 4), order="C") for _size, degree in missing]
        # See _prepare_blocks: all that the launches use was allocated above,
        # and the collector stays off until the last capture has ended.
        with collector_paused():
            for (size, degree), staged in zip(missing, tables):
                graph, output = self._capture(size, degree, staged, 0)
                self.shared[size, degree] = (graph, output, staged)
        self.buffer_bytes = (
            sum(x.nbytes for x in self.buffers)
            + self.coefficients.nbytes
            + self.table.nbytes
            + sum(staged.nbytes for _graph, _output, staged in self.shared.values())
        )
        self.operator.timing_stats.filter_graph_buffer_bytes = self.buffer_bytes
        self.graphs = [self.shared[kind] for kind in kinds]
        self.key = key
        self.operator.timing_stats.filter_graph_builds += len(missing)

    def apply(self, matrix, blocks, lower, upper, reference, reset, initial_sigma=None, out=None):
        cp = self.cp
        operator = self.operator  # Keep the owner alive throughout submission.
        with self.lock, cp.cuda.Device(self.device):
            self._prepare(matrix, blocks)
            stream = cp.cuda.get_current_stream()
            if self.has_finished:
                stream.wait_event(self.finished)
            if self.has_upload:
                # Only the preceding tiny H2D upload must finish before the
                # pinned CPU coefficient table can be reused, not the kernels.
                self.uploaded.synchronize()
            half_span = 0.5 * (float(upper) - float(lower))
            center = 0.5 * (float(upper) + float(lower))
            if (
                not np.isfinite([lower, upper, reference]).all()
                or half_span <= 0
                or float(reference) == center
            ):
                raise ValueError("invalid graph filter interval/reference")
            sigma_one = half_span / (float(reference) - center)
            fill_recurrence_table(
                self.host_parameters, blocks, center, half_span, sigma_one, reset, initial_sigma
            )
            self.parameters.set(self.host_parameters, stream=stream)
            self.uploaded.record(stream)
            self.has_upload = True
            # Ritz consumes column-contiguous orbitals. Writing directly into
            # that layout avoids an extra N-by-states device copy and its peak
            # allocation; recurrence arithmetic and block order are unchanged.
            if out is None:
                order = "F" if os.environ.get("PARSEC_CUPY_FILTER_COLUMN_MAJOR", "0") == "1" else "K"
                result = cp.empty_like(matrix, dtype=cp.float64, order=order)
            else:
                # A caller that owns the input may pass it as ``out``: every
                # block is copied into the recurrence buffer before its own
                # columns are overwritten, in order on this stream, so no
                # second N-by-states array is needed.
                if out.shape != matrix.shape or out.dtype != cp.dtype(cp.float64):
                    raise ValueError("out must be a float64 device array matching the input")
                result = out
            offset = 0
            for block, (graph, output, staged) in zip(blocks, self.graphs, strict=True):
                if staged is not None:
                    # A shared graph reads the table ``staged``: the rows of
                    # this block go there from the uploaded table, device to
                    # device, behind the upload and the launch of the block
                    # before on this stream.
                    staged.data.copy_from_device_async(
                        self.parameters[offset : offset + block.degree].data,
                        staged.nbytes,
                        stream,
                    )
                cp.copyto(
                    self.buffers[0][:, : block.stop - block.start],
                    matrix[:, block.start : block.stop],
                )
                graph.launch(stream=stream)
                result[:, block.start : block.stop] = output
                offset += block.degree
                operator.timing_stats.filter_graph_launches += 1
                operator.timing_stats.hamiltonian_applications += block.degree
                operator.timing_stats.orbital_vectors_applied += block.degree * (
                    block.stop - block.start
                )
            self.finished.record(stream)
            self.has_finished = True
            return result
