"""Nonlocal forces contracted on the devices that hold the orbitals.

The force of the Kleinman--Bylander projectors
(:mod:`parsec_python.Forces.nonlocal_force`) needs, for every occupied
orbital, its overlaps with the projectors and with their gradients:

``t[q, c] = 4 * sum_n f_n * s_q <beta_q|psi_n> <d_c beta_q|psi_n>``.

Summed over the projectors of an atom this is the force on it.  The orbitals
of a large calculation never leave their GPUs, so the sums are formed there:
the signed projector overlaps come from the Hamiltonian that already holds
``B`` on the device, and the three gradient overlaps from projection kernels
of the same kind built from the gradient matrices.  Only ``t``, one row per
projector, is downloaded.

In a symmetry sector the reduced matrices give the same numbers:
``B_G.T c = B.T psi`` for an orbital ``psi`` of the sector with wedge vector
``c`` (:meth:`ReflectionRepresentationDecomposition.reduce_nonlocal_operators`),
and the gradient matrices are reduced by the same map.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from ..backends.cupy import require_cupy
from ..backends.cupy_projectors import CuPySparseProjectorProjection


#: Device bytes one column chunk may take for its coefficient arrays.
_CHUNK_BYTES = 256 * 1024 * 1024


def gradient_transposes(components: Sequence[Any]) -> tuple[sp.csr_matrix, ...]:
    """Host CSR of ``G_c.T`` for the three gradient matrices ``G_c``.

    Row ``q`` lists the grid rows of gradient column ``q``, in ascending
    order, which is the layout the device projection kernel reads.
    """

    transposes = []
    for component in components:
        transpose = sp.csr_matrix(component, dtype=np.float64).T.tocsr(copy=True)
        transpose.sum_duplicates()
        transpose.sort_indices()
        if transpose.indices.dtype != np.int32 or transpose.indptr.dtype != np.int32:
            transpose = sp.csr_matrix(
                (
                    transpose.data,
                    transpose.indices.astype(np.int32),
                    transpose.indptr.astype(np.int32),
                ),
                shape=transpose.shape,
            )
        transposes.append(transpose)
    return tuple(transposes)


class DeviceProjectorGradients:
    """The three gradient projections of one operator on the current device."""

    def __init__(self, transposes: Sequence[sp.csr_matrix]) -> None:
        cp, _ = require_cupy()
        self.cp = cp
        self.projector_count = int(transposes[0].shape[0])
        unit = cp.ones(self.projector_count, dtype=cp.float64)
        self.projections = []
        for transpose in transposes:
            if int(transpose.shape[0]) != self.projector_count:
                raise ValueError("the gradient matrices differ in their columns")
            lengths = np.diff(transpose.indptr)
            self.projections.append(
                CuPySparseProjectorProjection(
                    cp,
                    (
                        cp.asarray(transpose.indptr, dtype=cp.int32),
                        cp.asarray(transpose.indices, dtype=cp.int32),
                        cp.asarray(transpose.data, dtype=cp.float64),
                        transpose.shape,
                    ),
                    unit,
                    max_row_entries=int(lengths.max(initial=0)),
                )
            )

    def contract(self, operator: Any, block: Any, weights: np.ndarray) -> np.ndarray:
        """Return ``t`` of the module text for the columns of ``block``.

        ``operator`` is the device Hamiltonian whose projectors these
        gradients belong to, ``block`` a ``(rows, columns)`` orbital array
        on the current device or on the host, and ``weights`` the
        occupations of its columns.  The result is a host ``(projectors, 3)``
        array.
        """

        cp = self.cp
        weights = np.asarray(weights, dtype=np.float64)
        width = int(block.shape[1])
        if weights.shape != (width,):
            raise ValueError("one occupation per orbital column is required")
        total = cp.zeros((self.projector_count, 3), dtype=cp.float64)
        if self.projector_count == 0 or width == 0:
            return np.zeros((self.projector_count, 3), dtype=np.float64)
        chunk = max(1, int(_CHUNK_BYTES // (8 * 2 * max(self.projector_count, 1))))
        for start in range(0, width, chunk):
            stop = min(width, start + chunk)
            occupied = weights[start:stop]
            if not np.any(occupied != 0.0):
                continue
            columns = block[:, start:stop]
            if not isinstance(columns, cp.ndarray):
                columns = cp.asarray(columns, dtype=cp.float64, order="F")
            signed = operator._signed_nonlocal_coefficients_or_none(columns)
            if signed is None:
                continue
            signed = signed * cp.asarray(4.0 * occupied, dtype=cp.float64)[None, :]
            for component, projection in enumerate(self.projections):
                total[:, component] += cp.sum(signed * projection(columns), axis=1)
        return np.asarray(cp.asnumpy(total), dtype=np.float64)


def projector_force_terms(
    operator: Any,
    vectors: Any,
    weights: np.ndarray,
    transposes: Sequence[sp.csr_matrix],
) -> np.ndarray:
    """``t`` for the leading ``len(weights)`` columns of one stored basis.

    ``vectors`` is what an eigensolver keeps for the operator: a device
    array, a host array (state spilled to the host), or a basis shared by
    the devices of a sector (:class:`DistributedBasis`), each of which then
    contracts its own columns with its own replica of the operator.
    """

    cp, _ = require_cupy()
    weights = np.asarray(weights, dtype=np.float64)
    count = int(weights.size)
    if count > int(vectors.shape[1]):
        raise ValueError("more occupations than stored orbital columns")
    group = getattr(vectors, "group", None)
    if group is not None:
        selected = vectors[:, :count]

        def partial(index, device):
            block = selected.blocks[index]
            if not block.shape[1]:
                return None
            held = [
                weights[column : column + width]
                for column, _offset, width in selected.layout.spans(index)
            ]
            return DeviceProjectorGradients(transposes).contract(
                group._operator(device),
                block,
                held[0] if len(held) == 1 else np.concatenate(held),
            )

        total = np.zeros((int(transposes[0].shape[0]), 3), dtype=np.float64)
        # Fixed device order keeps the host sum deterministic.
        for part in group.map(partial):
            if part is not None:
                total += part
        return total
    device = int(operator.effective_potential.device.id)
    with cp.cuda.Device(device):
        return DeviceProjectorGradients(transposes).contract(
            operator, vectors[:, :count], weights
        )


__all__ = [
    "DeviceProjectorGradients",
    "gradient_transposes",
    "projector_force_terms",
]
