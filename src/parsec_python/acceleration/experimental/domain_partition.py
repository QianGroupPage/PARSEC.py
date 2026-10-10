"""Offline spatial partitions for the experimental distributed Hamiltonian.

This module changes ownership, not the grid or the operator. ``neighbors``
uses the production stencil-major shape ``(slots, rows)`` and -1 means an
absent entry. Coordinates are integer grid triples. The returned local rows
are in original row order, independently of the ordering used to partition.
This keeps existing local ordering where possible; it does not guarantee that
affine tiles crossing a partition boundary remain compressible.

``hilbert`` and ``morton`` order blocks, then use Cartesian order inside each
block. Balanced point counts can split a block at a partition boundary. No
missing points are filled in. ``brick`` is recursive coordinate bisection,
not a guarantee of rectangular partitions for an irregular grid. Metrics use
actual operator connectivity, including any symmetry-induced long edges.

Example (CPU only)::

    python -m parsec_python.acceleration.experimental.domain_partition \
        captured_stencil.npz partitions.json --parts 1 2 4 8 16

These are structural traffic counts, not measured MPI/GPU performance.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class PartitionResult:
    owner: np.ndarray
    local_rows: tuple
    halo_rows: tuple
    metrics: dict


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer_coordinates(coords):
    points = np.asarray(coords)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("coords must have nonempty shape (rows, 3)")
    if points.dtype.kind not in "iu":
        raise ValueError("coords must contain integer grid coordinates")
    if points.dtype.kind == "u" and np.max(points) > np.iinfo(np.int64).max:
        raise ValueError("coordinate exceeds signed 64-bit range")
    points = points.astype(np.int64, copy=False)
    # Python integer subtraction avoids a silent int64 overflow during shifting.
    for axis in range(3):
        if int(points[:, axis].max()) - int(points[:, axis].min()) > np.iinfo(np.int64).max:
            raise ValueError("coordinate extent exceeds signed 64-bit range")
    return points - points.min(axis=0)


def _stencil_neighbors(neighbors, rows):
    values = np.asarray(neighbors)
    if values.ndim != 2 or values.shape[1] != rows:
        raise ValueError("neighbors must have stencil-major shape (slots, rows)")
    if values.dtype.kind not in "iu":
        raise ValueError("neighbors must contain integer row indices")
    if np.any(values < -1) or np.any(values >= rows):
        raise ValueError("neighbors must be -1 or a valid original row index")
    return values


def hilbert_keys(points):
    """Skilling 3-D axes-to-transpose mapping, adapted from sfc_stencil.py.

    Kept NumPy-only so partition planning does not import CuPy. Complete cube
    uniqueness and face-adjacent traversal are checked in the unit tests.
    """
    x = np.asarray(points)
    if x.ndim != 2 or x.shape[1] != 3 or x.dtype.kind not in "iu" or np.any(x < 0):
        raise ValueError("nonnegative integer triples required")
    bits = max(1, int(x.max(initial=0)).bit_length())
    if bits > 21:
        raise ValueError("Hilbert keys exceed uint64")
    x = x.astype(np.int64, copy=True)
    q = 1 << (bits - 1)
    while q > 1:
        mask = q - 1
        for axis in range(3):
            invert = (x[:, axis] & q) != 0
            t = (x[:, 0] ^ x[:, axis]) & mask
            t[invert] = 0
            x[:, 0] ^= np.where(invert, mask, t)
            x[:, axis] ^= t
        q >>= 1
    for axis in range(1, 3):
        x[:, axis] ^= x[:, axis - 1]
    t = np.zeros(len(x), dtype=np.int64)
    q = 1 << (bits - 1)
    while q > 1:
        t ^= np.where((x[:, 2] & q) != 0, q - 1, 0)
        q >>= 1
    x ^= t[:, None]
    keys = np.zeros(len(x), dtype=np.uint64)
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            keys = (keys << np.uint64(1)) | ((x[:, axis] >> bit) & 1).astype(np.uint64)
    return keys


def morton_keys(points):
    """Interleave 3-D integer coordinate bits into unsigned 64-bit keys."""
    x = np.asarray(points)
    if x.ndim != 2 or x.shape[1] != 3 or x.dtype.kind not in "iu" or np.any(x < 0):
        raise ValueError("nonnegative integer triples required")
    bits = max(1, int(x.max(initial=0)).bit_length())
    if bits > 21:
        raise ValueError("Morton keys exceed uint64")
    x = x.astype(np.uint64, copy=False)
    keys = np.zeros(len(x), dtype=np.uint64)
    for bit in range(bits - 1, -1, -1):
        for axis in range(3):
            keys = (keys << np.uint64(1)) | ((x[:, axis] >> np.uint64(bit)) & np.uint64(1))
    return keys


def _axis_order(points, rows):
    axis = int(np.argmax(np.ptp(points[rows], axis=0)))
    others = [item for item in range(3) if item != axis]
    return rows[np.lexsort((rows, points[rows, others[1]], points[rows, others[0]], points[rows, axis]))]


def _assign_bricks(points, sizes, owner):
    def recurse(rows, first, stop):
        if stop - first == 1:
            owner[rows] = first
            return
        middle = (first + stop) // 2
        ordered = _axis_order(points, rows)
        split = int(sizes[first:middle].sum())
        recurse(ordered[:split], first, middle)
        recurse(ordered[split:], middle, stop)

    recurse(np.arange(len(points), dtype=np.int64), 0, len(sizes))


def partition_metrics(neighbors, owner, nparts=None, *, local_rows=None):
    """Return actual directed halo rows and JSON-compatible structural counts.

    A received row is counted once per destination, even if several stencil
    slots/local rows need it. ``recv_rows_by_source[dst][src]`` counts FP64
    values per orbital column; multiplying by 8 and the column block size gives
    payload bytes for one Hamiltonian application. This excludes protocol
    overhead, nonlocal-projector reductions, dense algebra, and SCF collectives.
    """
    owner = np.asarray(owner)
    if owner.ndim != 1 or owner.dtype.kind not in "iu" or not len(owner):
        raise ValueError("owner must be a nonempty integer vector")
    if nparts is None:
        nparts = int(owner.max()) + 1
    nparts = _positive_integer(nparts, "nparts")
    if np.any(owner < 0) or np.any(owner >= nparts):
        raise ValueError("owner contains an invalid rank")
    neighbors = _stencil_neighbors(neighbors, len(owner))
    canonical = tuple(np.flatnonzero(owner == rank) for rank in range(nparts))
    if any(not len(rows) for rows in canonical):
        raise ValueError("each rank must own at least one row")
    if local_rows is not None:
        if len(local_rows) != nparts or any(not np.array_equal(rows, canonical[rank]) for rank, rows in enumerate(local_rows)):
            raise ValueError("local_rows must match owner in original row order")
    recv = np.zeros((nparts, nparts), dtype=np.int64)
    halos, nonzeros, remote_references, boundary = [], [], [], []
    for rank, rows in enumerate(canonical):
        # Peak temporary storage scales with the local stencil, not ranks*N.
        entries = neighbors[:, rows]
        valid = entries >= 0
        remote = np.zeros(entries.shape, dtype=bool)
        remote[valid] = owner[entries[valid]] != rank
        halo = np.unique(entries[remote]).astype(np.int64, copy=False)
        halos.append(halo)
        recv[rank] = np.bincount(owner[halo], minlength=nparts)
        nonzeros.append(int(valid.sum()))
        remote_references.append(int(remote.sum()))
        boundary.append(int(remote.any(axis=0).sum()))
    counts = np.array([len(rows) for rows in canonical], dtype=np.int64)
    receive_rows = recv.sum(axis=1)
    send_rows = recv.sum(axis=0)
    metrics = {
        "rows": len(owner), "parts": nparts, "stencil_slots": neighbors.shape[0],
        "row_counts": counts.tolist(), "load_max_over_mean": float(counts.max() / counts.mean()),
        "nonzero_entries_per_rank": nonzeros,
        "remote_stencil_references_per_rank": remote_references,
        "boundary_rows_per_rank": boundary,
        "interior_rows_per_rank": (counts - boundary).tolist(),
        "halo_rows_per_rank": receive_rows.tolist(),
        "halo_over_owned_rows_per_rank": (receive_rows / counts).tolist(),
        "recv_rows_by_source": recv.tolist(),
        "send_rows_per_rank": send_rows.tolist(),
        "recv_neighbor_counts": np.count_nonzero(recv, axis=1).tolist(),
        "send_neighbor_counts": np.count_nonzero(recv, axis=0).tolist(),
        "recv_bytes_fp64_per_column": (receive_rows * 8).tolist(),
        "send_bytes_fp64_per_column": (send_rows * 8).tolist(),
        "total_payload_bytes_fp64_per_column": int(receive_rows.sum() * 8),
        "payload_scope": "One stencil application, unique rows per destination; excludes protocol overhead and all non-stencil communication.",
    }
    return tuple(halos), metrics


def make_partition(coords, neighbors, nparts, method="axis", tile_size=2):
    """Make a deterministic, point-balanced ownership plan without padding.

    All methods have point-count imbalance at most one. The block size affects
    only Hilbert/Morton ownership ordering; local rows always retain original
    global row order. Coordinate shifts do not affect ownership.
    """
    points = _integer_coordinates(coords)
    neighbors = _stencil_neighbors(neighbors, len(points))
    nparts = _positive_integer(nparts, "nparts")
    tile_size = _positive_integer(tile_size, "tile_size")
    if nparts > len(points):
        raise ValueError("nparts cannot exceed the number of active grid points")
    if method not in ("axis", "brick", "morton", "hilbert"):
        raise ValueError("method must be axis, brick, morton, or hilbert")
    sizes = np.full(nparts, len(points) // nparts, dtype=np.int64)
    sizes[:len(points) % nparts] += 1
    owner = np.empty(len(points), dtype=np.int32)
    if method == "brick":
        _assign_bricks(points, sizes, owner)
    else:
        rows = np.arange(len(points), dtype=np.int64)
        if method == "axis":
            ordered = _axis_order(points, rows)
        else:
            blocks = points // tile_size
            key = (hilbert_keys if method == "hilbert" else morton_keys)(blocks)
            # Cartesian intra-block order, deterministic original-row tie break.
            ordered = rows[np.lexsort((rows, points[:, 2], points[:, 1], points[:, 0], key))]
        owner[ordered] = np.repeat(np.arange(nparts, dtype=np.int32), sizes)
    local_rows = tuple(np.flatnonzero(owner == rank) for rank in range(nparts))
    halo_rows, metrics = partition_metrics(neighbors, owner, nparts, local_rows=local_rows)
    metrics.update(method=method, tile_size=tile_size, local_order="original_global_row")
    if method in ("morton", "hilbert"):
        sorted_keys = key[ordered]
        cuts = np.cumsum(sizes)[:-1]
        split = sorted_keys[cuts - 1] == sorted_keys[cuts]
        metrics["split_blocks"] = int(np.unique(sorted_keys[cuts[split]]).size)
        metrics["block_partition_boundaries"] = int(split.sum())
    return PartitionResult(owner, local_rows, halo_rows, metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--methods", nargs="+", default=["axis", "brick", "morton", "hilbert"])
    parser.add_argument("--tile-size", type=int, default=2)
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args(argv)
    _positive_integer(args.columns, "columns")
    digest = hashlib.sha256()
    with args.input.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with np.load(args.input, allow_pickle=False) as data:
        coords, neighbors = data["coordinates"], data["neighbors"]
    records = []
    for parts in args.parts:
        for method in args.methods:
            started = time.perf_counter()
            result = make_partition(coords, neighbors, parts, method, args.tile_size)
            record = dict(result.metrics)
            record["setup_seconds"] = time.perf_counter() - started
            record["column_block_size"] = args.columns
            record["total_payload_bytes_fp64_per_block"] = record["total_payload_bytes_fp64_per_column"] * args.columns
            records.append(record)
            print(json.dumps({key: record[key] for key in ("method", "parts", "setup_seconds", "total_payload_bytes_fp64_per_block")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": 1, "kind": "offline_structural_audit_not_performance", "input": str(args.input), "input_sha256": digest.hexdigest(), "records": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
