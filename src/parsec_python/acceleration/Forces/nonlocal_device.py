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

The device copy of the gradients of one operator is made once per device and
force call (:class:`GradientTransposes`): the hand-over corrections of a
sector, the two spin channels and the parts of a shared basis contract with
the copy already there.
"""

from __future__ import annotations

from itertools import count as _count
import os
import threading
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from ..backends.cupy import require_cupy
from ..backends.cupy_projectors import CuPySparseProjectorProjection


#: Device bytes one column chunk may take for its coefficient arrays.
_CHUNK_BYTES = 256 * 1024 * 1024

#: Bytes of orbital columns one chunk may upload from a block on the host
#: (sector states spilled to the host), ``PARSEC_CUPY_FORCE_UPLOAD_CAP``.
_HOST_UPLOAD_BYTES = 1 << 30


def force_upload_cap_setting() -> str:
    """``PARSEC_CUPY_FORCE_UPLOAD_CAP``: ``on`` (default) or ``off``.

    ``on`` uploads the orbital columns of a block held on the host (sector
    states spilled to the host) at most 1 GiB at a time for the nonlocal
    forces; ``off`` takes the chunks of the coefficient budget alone, as
    before (2,097 columns of 1.46 M rows, 24 GB, for 8,000 projectors).  The
    chunks of a block on the device are the same either way.  The sums of
    the chunks follow another order, so ``on`` may change the last bits of
    the forces of spilled states.
    """

    value = os.environ.get("PARSEC_CUPY_FORCE_UPLOAD_CAP", "on").strip().lower() or "on"
    if value in {"1", "true", "yes"}:
        value = "on"
    if value in {"0", "false", "no"}:
        value = "off"
    if value not in {"on", "off"}:
        raise ValueError("PARSEC_CUPY_FORCE_UPLOAD_CAP must be on or off")
    return value


class DeviceGradientCache:
    """The device copies of the gradients of a force call, at most one per device.

    A device keeps the copy it made last: the gradients of the sector it
    contracts.  The next sector on that device drops it before making its
    own, so a device never holds more gradients than one contraction needs,
    and all are freed with the transposes of the call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, tuple[int, "DeviceProjectorGradients"]] = {}

    def gradients(self, transposes: "GradientTransposes") -> "DeviceProjectorGradients":
        """The copy of ``transposes`` on the current device, made where it is not there."""

        cp, _ = require_cupy()
        device = int(cp.cuda.runtime.getDevice())
        token = transposes.token
        with self._lock:
            entry = self._entries.get(device)
            if entry is not None and entry[0] == token:
                return entry[1]
            # Another operator's copy: freed before the new one is made.
            self._entries.pop(device, None)
        del entry
        made = DeviceProjectorGradients(transposes)
        # The upload is complete before another stream of this device reads it.
        cp.cuda.get_current_stream().synchronize()
        with self._lock:
            self._entries[device] = (token, made)
        return made


_TOKENS = _count()


class GradientTransposes(tuple):
    """The three host CSR transposes of :func:`gradient_transposes`, with their device copies.

    A tuple of the three matrices to every reader.  ``device_cache`` holds
    the device copies (:class:`DeviceGradientCache`); the transposes of the
    sectors of one force call share one (:func:`share_device_gradients`).
    """

    def __new__(cls, matrices):
        made = super().__new__(cls, tuple(matrices))
        made.token = next(_TOKENS)
        made.device_cache = None
        return made

    def __reduce__(self):
        # the host matrices only: device copies do not travel
        return (GradientTransposes, (tuple(self),))


def share_device_gradients(transposes: Sequence[Any]) -> None:
    """Give the gradient transposes of the sectors of one force call one device cache.

    Each device then holds the copy of one sector at a time, as when every
    contraction made its own.
    """

    cache = DeviceGradientCache()
    for item in transposes:
        if isinstance(item, GradientTransposes):
            item.device_cache = cache


def _device_gradients(transposes: Sequence[sp.csr_matrix]) -> "DeviceProjectorGradients":
    """The gradients of ``transposes`` on the current device, made once per device and force call."""

    if not isinstance(transposes, GradientTransposes):
        return DeviceProjectorGradients(transposes)
    with _CACHE_LOCK:
        # (the device threads of a shared basis ask at once)
        if transposes.device_cache is None:
            transposes.device_cache = DeviceGradientCache()
        cache = transposes.device_cache
    return cache.gradients(transposes)


_CACHE_LOCK = threading.Lock()


def gradient_transposes(components: Sequence[Any]) -> GradientTransposes:
    """Host CSR of ``G_c.T`` for the three gradient matrices ``G_c``.

    Row ``q`` lists the grid rows of gradient column ``q``, in ascending
    order, which is the layout the device projection kernel reads.  The
    result is a tuple of the three (:class:`GradientTransposes`).
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
    return GradientTransposes(transposes)


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

        The columns go through in chunks whose coefficient arrays take at
        most 256 MiB; a block on the host uploads at most 1 GiB of columns at
        a time besides (``PARSEC_CUPY_FORCE_UPLOAD_CAP``).  An operator with
        fewer signed projector rows than there are gradient columns (the
        projectors of one atom per orbit against gradients of every atom, or
        the reverse) is refused with both counts.
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
        if not isinstance(block, cp.ndarray) and force_upload_cap_setting() == "on":
            # Sector states spilled to the host: each chunk is uploaded whole.
            chunk = min(chunk, max(1, int(_HOST_UPLOAD_BYTES // (8 * max(int(block.shape[0]), 1)))))
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
            if int(signed.shape[0]) < self.projector_count:
                raise ValueError(
                    f"the operator gives {int(signed.shape[0])} signed projector rows but the projector gradients "
                    f"have {self.projector_count} columns: the gradients do not belong to the projectors of this "
                    "operator (sector projectors of one atom per orbit need gradients compressed by the same rule, "
                    "compress_gradients_for_sectors)"
                )
            if int(signed.shape[0]) > self.projector_count:
                # The LDA+U columns appended to the factor of the operator
                # (Hamiltonian.hubbard_device) follow the KB ones; their
                # forces are formed there.
                signed = signed[: self.projector_count]
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
    contracts its own columns with its own replica of the operator.  The
    gradients are put on a device once per force call where ``transposes``
    come from :func:`gradient_transposes` (:class:`GradientTransposes`).
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
            return _device_gradients(transposes).contract(
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
        return _device_gradients(transposes).contract(
            operator, vectors[:, :count], weights
        )


__all__ = [
    "DeviceGradientCache",
    "DeviceProjectorGradients",
    "GradientTransposes",
    "force_upload_cap_setting",
    "gradient_transposes",
    "projector_force_terms",
    "share_device_gradients",
]
