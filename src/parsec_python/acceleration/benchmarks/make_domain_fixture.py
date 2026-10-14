"""Deterministic synthetic capture for MPI/GPU correctness smoke tests only.

The files share capture_distributed.py's schema, but represent no material,
pseudopotential or SCF calculation. Timings are not physical performance
evidence. A clipped spherical integer grid gives irregular boundaries; signed
compact projector columns exercise distributed nonlocal reductions.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .capture_distributed import _git_provenance, _sha256, write_basis_memmap


def make_fixture(destination, *, radius=6, states=24, projectors=16, seed=20261006):
    destination = Path(destination).expanduser().resolve()
    if radius < 2 or states < 1 or projectors < 1:
        raise ValueError("radius >= 2 and positive states/projectors required")
    extent = np.arange(-radius, radius + 1, dtype=np.int64)
    coordinates = np.stack(np.meshgrid(extent, extent, extent, indexing="ij"), axis=-1).reshape(-1, 3)
    coordinates = coordinates[np.sum(coordinates * coordinates, axis=1) <= radius * radius]
    rows = len(coordinates)
    if states > rows or projectors > rows:
        raise ValueError("states and projectors cannot exceed grid points")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"fixture directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "CAPTURE_INCOMPLETE"
    marker.write_text("Synthetic fixture generation is incomplete.\n", encoding="utf-8")
    rng = np.random.default_rng(seed)
    lookup = {tuple(point): row for row, point in enumerate(coordinates)}
    neighbors = np.full((7, rows), -1, dtype=np.int32)
    neighbors[0] = np.arange(rows, dtype=np.int32)
    codes = np.zeros_like(neighbors, dtype=np.uint8)
    codes[0] = 1
    for axis in range(3):
        for offset, sign in enumerate((-1, 1)):
            slot = 1 + 2 * axis + offset
            for row, point in enumerate(coordinates):
                adjacent = point.copy()
                adjacent[axis] += sign
                neighbors[slot, row] = lookup.get(tuple(adjacent), -1)
            codes[slot, neighbors[slot] >= 0] = 2
    palette = np.array([0., 6., -1.], dtype=np.float64)
    potential = 1.5 + 0.025 * np.sum(coordinates.astype(float) ** 2, axis=1)
    potential += 0.003 * coordinates[:, 0] - 0.002 * coordinates[:, 1]

    # Each compact column has norm 0.08, keeping the signed correction small
    # relative to the positive local potential without dropping negative signs.
    data, row_ids, column_ids = [], [], []
    centers = rng.choice(rows, size=projectors, replace=False)
    for column, center in enumerate(centers):
        distance2 = np.sum((coordinates - coordinates[center]) ** 2, axis=1)
        support = np.flatnonzero(distance2 <= 5)
        values = np.exp(-0.4 * distance2[support]) * rng.normal(size=len(support))
        values *= 0.08 / np.linalg.norm(values)
        data.extend(values)
        row_ids.extend(support)
        column_ids.extend([column] * len(support))
    factors = sp.coo_matrix((data, (row_ids, column_ids)), shape=(rows, projectors)).tocsr()
    factors.sum_duplicates()
    factors.sort_indices()
    signs = np.where(np.arange(projectors) % 2, -1., 1.)

    q, _ = np.linalg.qr(rng.normal(size=(rows, states)))
    hq = potential[:, None] * q
    for slot in range(len(neighbors)):
        valid = neighbors[slot] >= 0
        hq[valid] += palette[codes[slot, valid], None] * q[neighbors[slot, valid]]
    hq += factors @ (signs[:, None] * (factors.T @ q))
    projected = q.T @ hq
    projected = 0.5 * (projected + projected.T)
    eigenvalues, rotation = np.linalg.eigh(projected)
    basis = np.asfortranarray(q @ rotation)
    orthogonality_error = float(np.max(np.abs(basis.T @ basis - np.eye(states))))
    projected_residual = float(np.linalg.norm((hq @ rotation) - basis * eigenvalues[None, :]))
    if orthogonality_error > 1e-12:
        raise RuntimeError("synthetic fixture lost basis orthogonality")
    np.savez(destination / "operator.npz", coordinates=coordinates,
             neighbors=neighbors, codes=codes, palette=palette,
             local_potential=potential, B_data=factors.data,
             B_indices=factors.indices, B_indptr=factors.indptr,
             B_shape=np.asarray(factors.shape, dtype=np.int64), signs=signs,
             multiplicity=np.ones(rows, dtype=np.int64),
             representative_rows=np.arange(rows, dtype=np.int64),
             sector_orbits=np.arange(rows, dtype=np.int64))
    write_basis_memmap(destination / "basis.npy", basis)
    np.save(destination / "eigenvalues.npy", eigenvalues, allow_pickle=False)
    manifest = {
        "schema": 1, "kind": "synthetic_mpi_gpu_smoke_fixture", "complete": True,
        "synthetic": True, "physical_performance_evidence": False,
        "converged_scf": False, "scf_iteration": 0, "sector": 0,
        "sector_count": 1, "sector_solves_completed": 0,
        "global_requested_states": states, "sector_requested_states": states,
        "full_working_states": states, "captured_columns": states,
        "columns_truncated": False, "basis_shape": [rows, states],
        "basis_order": "F", "precision": "float64", "units": "Ry",
        "units_note": "Synthetic numerical values using the replay Ry convention; no material or DFT energy is represented.",
        "coordinate_units": "synthetic integer grid units",
        "basis_metric": "Euclidean inner product; multiplicities are all one",
        "hamiltonian": "stencil + diag(local_potential) + B @ diag(signs) @ B.T",
        "projector_shape": list(factors.shape), "projector_nnz": int(factors.nnz),
        "stencil_shape": list(neighbors.shape), "device_id": None,
        "source_git": _git_provenance(), "input_hashes": {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_parameters": {"radius": radius, "states": states, "projectors": projectors, "seed": seed},
        "basis_description": "Random orthonormal subspace rotated to diagonalize its projected Hamiltonian, not converged eigenvectors",
        "orthogonality_max_abs_error": orthogonality_error,
        "full_hamiltonian_residual_frobenius": projected_residual,
        "files": {name: {"bytes": (destination / name).stat().st_size,
                          "sha256": _sha256(destination / name)}
                  for name in ("operator.npz", "basis.npy", "eigenvalues.npy")},
    }
    temporary_path = destination / "capture.json.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(destination / "capture.json")
    marker.unlink()
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--projectors", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    manifest = make_fixture(args.destination, radius=args.radius, states=args.states,
                            projectors=args.projectors, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ("kind", "basis_shape", "projector_shape", "orthogonality_max_abs_error")}))


if __name__ == "__main__":
    main()
