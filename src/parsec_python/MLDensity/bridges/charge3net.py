"""Run an official ChargE3Net checkpoint at arbitrary PARSEC probe points.

This file intentionally has no dependency on ``parsec_python`` so it can run
inside a dedicated ChargE3Net virtual environment.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys

import numpy as np


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ML_Density_Device=cuda, but PyTorch cannot see CUDA")
    return torch.device(requested)


def _move_batch(torch, batch, device):
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def predict(args: argparse.Namespace) -> None:
    repository = Path(args.repository).resolve()
    sys.path.insert(0, str(repository))

    try:
        import ase
        import torch
        from scipy.spatial import cKDTree
        from src.charge3net.data.collate import collate_list_of_dicts
        from src.charge3net.data.graph_construction import KdTreeGraphConstructor
        from src.charge3net.models.e3 import E3DensityModel
    except ImportError as error:
        raise RuntimeError(
            "ChargE3Net dependencies are missing in this Python environment; "
            "install the official repository requirements"
        ) from error

    torch.set_default_dtype(torch.float32)
    device = _device(torch, args.device)
    with np.load(args.request, allow_pickle=False) as request:
        symbols = [str(value) for value in request["symbols"]]
        atom_positions = np.asarray(request["atom_positions_angstrom"], dtype=float)
        probes = np.asarray(request["probe_positions_angstrom"], dtype=float)
        target_coordinates = np.asarray(request["target_coordinates_bohr"], dtype=float)
        cell = np.asarray(request["cell_angstrom"], dtype=float)

    atoms = ase.Atoms(symbols=symbols, positions=atom_positions, cell=cell, pbc=False)
    model = E3DensityModel(
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=20,
    ).to(device)
    # PyTorch 2.6 changed ``torch.load`` to weights-only by default, while
    # ChargE3Net checkpoints also contain optimizer/training metadata.  Keep
    # compatibility with both new and old PyTorch releases.
    try:
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    if "pytorch-lightning_version" in checkpoint:
        state = {
            key.replace("network.", "", 1): value
            for key, value in checkpoint["state_dict"].items()
        }
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    constructor = KdTreeGraphConstructor(
        cutoff=4.0,
        num_probes=None,
        disable_pbc=True,
        sorted_edges=True,
    )
    predictions: list[np.ndarray] = []
    atom_representation = None
    atom_tree = cKDTree(atom_positions)
    probe_cutoff = 4.0
    # e3nn tensor-product intermediates scale with atom-to-probe edges, not
    # merely with the probe count.  A fixed 10k-probe chunk can therefore fit
    # for a molecule yet exhaust an 8-GB GPU for a dense cluster.  Bound the
    # actual edge workload while retaining ``--chunk-size`` as an outer cap.
    max_probe_edges = int(
        os.environ.get("PARSEC_CHARGE3NET_MAX_PROBE_EDGES", "80000")
    )
    if max_probe_edges < 1:
        raise ValueError("PARSEC_CHARGE3NET_MAX_PROBE_EDGES must be positive")
    with torch.inference_mode():
        for start in range(0, probes.shape[0], args.chunk_size):
            chunk = probes[start : start + args.chunk_size]
            # The published graph has no atom-to-probe edge beyond its 4-A
            # cutoff, hence the learned local density is exactly zero there.
            # Passing a completely disconnected probe chunk to the upstream
            # constructor triggers an empty-array indexing error.  Filter such
            # points before graph construction, which also avoids expensive GPU
            # work throughout the vacuum portion of a cluster sphere.
            distances, _ = atom_tree.query(
                chunk, k=1, distance_upper_bound=probe_cutoff
            )
            active = np.isfinite(distances)
            chunk_density = np.zeros(chunk.shape[0], dtype=np.float64)
            if not np.any(active):
                predictions.append(chunk_density)
                continue
            active_chunk = chunk[active]
            edge_counts = atom_tree.query_ball_point(
                active_chunk, r=probe_cutoff, return_length=True
            )
            active_rows = np.flatnonzero(active)
            segment_start = 0
            edge_budget = max_probe_edges
            while segment_start < active_chunk.shape[0]:
                cumulative_edges = np.cumsum(
                    edge_counts[segment_start:], dtype=np.int64
                )
                take = int(
                    np.searchsorted(
                        cumulative_edges, edge_budget, side="right"
                    )
                )
                take = max(1, take)
                segment_end = min(active_chunk.shape[0], segment_start + take)
                segment = active_chunk[segment_start:segment_end]
                # Density is merely the graph constructor's target placeholder.
                graph = constructor(np.zeros(segment.shape[0]), atoms, segment)
                batch = _move_batch(
                    torch,
                    collate_list_of_dicts([graph], pin_memory=False),
                    device,
                )
                if atom_representation is None:
                    atom_representation = model.atom_model(batch)
                try:
                    values = model.probe_model(batch, atom_representation)
                except RuntimeError as error:
                    # The persistent atom representation grows with the number
                    # of atoms, so an edge budget that fits one molecule can
                    # still exhaust the same GPU for a larger cluster.  Probe
                    # predictions are independent once ``atom_representation``
                    # is built: retrying the *same* rows in smaller batches
                    # changes memory use only, not the model or physical data.
                    is_cuda_oom = (
                        device.type == "cuda"
                        and "out of memory" in str(error).lower()
                    )
                    actual_edges = int(cumulative_edges[take - 1])
                    if not is_cuda_oom or take <= 1 or actual_edges <= 1:
                        raise
                    del batch, graph
                    gc.collect()
                    torch.cuda.empty_cache()
                    edge_budget = max(1, min(edge_budget - 1, actual_edges // 2))
                    print(
                        "ChargE3Net CUDA OOM: retrying the current probe "
                        f"segment with edge budget {edge_budget}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                chunk_density[
                    active_rows[segment_start:segment_end]
                ] = values.reshape(-1).detach().cpu().numpy()
                segment_start = segment_end
            predictions.append(chunk_density)

    density = np.concatenate(predictions).astype(np.float64, copy=False)
    if density.shape != (probes.shape[0],) or not np.all(np.isfinite(density)):
        raise RuntimeError("ChargE3Net returned an invalid density vector")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray("parsec_python.ml_density.v1"),
        density=density,
        coordinates_bohr=target_coordinates,
        units=np.asarray("e_per_angstrom3"),
        provider=np.asarray("charge3net"),
        metadata_model=np.asarray(args.model),
        metadata_checkpoint=np.asarray(str(Path(args.checkpoint).resolve())),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="qm9", choices=("qm9", "mp", "nmc"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    predict(args)


if __name__ == "__main__":
    main()
