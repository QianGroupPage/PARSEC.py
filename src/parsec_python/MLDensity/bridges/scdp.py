"""Run an official SCDP checkpoint at arbitrary PARSEC probe points.

This bridge is dependency-isolated and imports no PARSEC.py modules.  SCDP
predicts atom-centred Gaussian-orbital coefficients once; evaluating those
coefficients is then chunked over the exact DFT grid points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np


# PyTorch 2.6 changed ``torch.load`` to weights-only mode by default.  Official
# SCDP checkpoints (and e3nn 0.4's packaged constants) predate that change and
# contain trusted framework objects, so retain the loader semantics under
# which those public files were produced.  This must be set before importing
# torch/e3nn below.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ML_Density_Device=cuda, but PyTorch cannot see CUDA")
    return torch.device(requested)


def _virtual_nodes(model: str, checkpoint: Path) -> str:
    """Virtual-node method of the model, checked against the checkpoint path.

    The fast model has no virtual nodes and the accurate one bond midpoints
    (``qm9`` is an alias of the fast model).  A checkpoint does not record
    the method, but the directories of the published archive name it
    (``qm9_none_...``, ``qm9_bond_...``); a mismatch would silently give a
    wrong density, so it is refused.
    """

    method = "bond" if model in {"accurate", "qm9-accurate"} else "none"
    path = checkpoint.as_posix().lower()
    named = {name for name in ("none", "bond") if f"_{name}_" in path}
    if len(named) == 1 and method not in named:
        expected = "accurate" if "bond" in named else "fast"
        raise RuntimeError(
            f"ML_Density_Model {model} uses virtual nodes '{method}', but the "
            f"checkpoint {checkpoint.name} in {checkpoint.parent.name} is a "
            f"'{named.pop()}' model; set ML_Density_Model: {expected}"
        )
    return method


def predict(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    # Before the imports, which take a while: a mismatch fails at once.
    vnode_method = _virtual_nodes(args.model, Path(args.checkpoint))
    repository = Path(args.repository).resolve()
    sys.path.insert(0, str(repository))
    try:
        import torch
        from ase.data import atomic_numbers
        from scdp.common.pyg import Batch
        from scdp.data.data import AtomicData, AtomicNumberTable
        from scdp.model.module import ChgLightningModule
    except ImportError as error:
        raise RuntimeError(
            "SCDP dependencies are missing in this Python environment "
            f"({error}); use a dedicated environment compatible with the "
            "official requirements"
        ) from error

    device = _device(torch, args.device)
    with np.load(args.request, allow_pickle=False) as request:
        symbols = [str(value) for value in request["symbols"]]
        atom_positions = np.asarray(request["atom_positions_angstrom"], dtype=float)
        probes = np.asarray(request["probe_positions_angstrom"], dtype=float)
        target_coordinates = np.asarray(request["target_coordinates_bohr"], dtype=float)
        cell = np.asarray(request["cell_angstrom"], dtype=float)
    numbers = np.asarray([atomic_numbers[symbol] for symbol in symbols], dtype=np.int64)

    model = ChgLightningModule.load_from_checkpoint(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        map_location=device,
    ).to(device)
    model.eval()
    model.ema.copy_to(model.parameters())
    if bool(model.pbc):
        raise RuntimeError(
            "this SCDP checkpoint is periodic, but PARSEC.py currently supports "
            "only isolated ML-density inference"
        )
    supported = {int(value) for value in model.unique_atom_types.detach().cpu().tolist()}
    unsupported = sorted(set(int(value) for value in numbers).difference(supported))
    if unsupported:
        raise RuntimeError(
            "the SCDP checkpoint has no Gaussian basis for atomic number(s) "
            + ", ".join(str(value) for value in unsupported)
        )
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    print(
        f"progress: SCDP {args.model} on {device_name} (torch "
        f"{torch.__version__}), loaded in {time.perf_counter() - started:.1f} s; "
        f"{probes.shape[0]:,} points",
        flush=True,
    )
    dummy_density = torch.zeros((1, 1, 1), dtype=torch.float32)
    data = AtomicData.build_graph_with_vnodes(
        atom_types=torch.as_tensor(numbers, dtype=torch.long),
        atom_coords=torch.as_tensor(atom_positions, dtype=torch.float32),
        cell=torch.as_tensor(cell, dtype=torch.float32),
        chg_density=dummy_density,
        origin=torch.zeros(3, dtype=torch.float32),
        metadata="parsec_python",
        z_table=AtomicNumberTable(list(range(100))),
        atom_cutoff=6.0,
        disable_pbc=True,
        vnode_method=vnode_method,
        # Every neighbour within the cutoff, as SCDP's preprocessing built
        # the training graphs (``preprocess.py``: max_neighbors=None).  The
        # 24 of this function's signature would cut the graph of a molecule
        # of more than 24 nodes, and the bond virtual nodes of the accurate
        # model count as nodes.
        max_neighbors=None,
        device=str(device),
    )
    batch = Batch.from_data_list([data]).to(device)

    predictions: list[np.ndarray] = []
    reported = 0
    with torch.inference_mode():
        coeffs, expo_scaling = model.predict_coeffs(batch)
        for start in range(0, probes.shape[0], args.chunk_size):
            percent = int(100 * start / probes.shape[0])
            if percent >= reported + 10:
                reported = percent - percent % 10
                print(
                    f"progress: {reported}% of the points "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
            chunk = torch.as_tensor(
                probes[start : start + args.chunk_size],
                dtype=coeffs.dtype,
                device=device,
            )
            n_probe = torch.tensor([chunk.shape[0]], dtype=torch.long, device=device)
            values = model.orbital_inference(
                batch, coeffs, expo_scaling, n_probe, chunk
            )
            predictions.append(values.reshape(-1).detach().cpu().numpy())

    density = np.concatenate(predictions).astype(np.float64, copy=False)
    if density.shape != (probes.shape[0],) or not np.all(np.isfinite(density)):
        raise RuntimeError("SCDP returned an invalid density vector")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray("parsec_python.ml_density.v1"),
        density=density,
        coordinates_bohr=target_coordinates,
        units=np.asarray("e_per_angstrom3"),
        provider=np.asarray("scdp"),
        metadata_model=np.asarray(args.model),
        metadata_checkpoint=np.asarray(str(Path(args.checkpoint).resolve())),
        metadata_virtual_nodes=np.asarray(vnode_method),
        metadata_device=np.asarray(device_name),
        metadata_torch=np.asarray(torch.__version__),
        metadata_seconds=np.asarray(f"{time.perf_counter() - started:.3f}"),
    )
    print(f"progress: done in {time.perf_counter() - started:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        default="fast",
        choices=("qm9", "fast", "accurate", "qm9-fast", "qm9-accurate"),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    predict(args)


if __name__ == "__main__":
    main()
