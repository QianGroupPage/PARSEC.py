"""Template bridge: the protocol between PARSEC.py and a density model.

Copy this file, replace :func:`predict_density` with a call to your model, and
use it with

    Initial_Density: custom
    ML_Density_Bridge: my_bridge.py
    ML_Density_Python: /path/to/the/model/environment/bin/python
    ML_Density_Model: my-model-v1          (a free label, passed to --model)

or ``python src/parsec_python/ml_density.py predict --provider custom
--bridge my_bridge.py ...``.  As it stands this file is a working toy model,
a sum of hydrogen-like 1s densities, which the tests run.

The bridge runs in the model's own environment and must not import
PARSEC.py.  It is called as

    python bridge.py --request REQUEST.npz --output OUTPUT.npz
                     --model NAME --device auto|cpu|cuda --chunk-size N
                     [--repository DIR] [--checkpoint FILE]

with the working directory set to ``--repository`` when one is configured.

Request ``REQUEST.npz`` (schema ``parsec_python.ml_request.v1``):

    symbols                   (N,)    chemical symbols such as "C"; a
                                      core-hole species ("C-1s") arrives as
                                      its element
    atom_positions_angstrom   (N, 3)
    probe_positions_angstrom  (M, 3)  the points where the density is wanted
    cell_angstrom             (3, 3)  an orthorhombic box (rows are its edges)
                                      holding the atoms and the points
    periodic                  ()      False: the system is isolated
    target_coordinates_bohr   (M, 3)  the same points in the frame of the
                                      calculation; copy it to the output

Atoms and points are translated together into the positive box, so a
translation-invariant model needs nothing else.

Output ``OUTPUT.npz`` (schema ``parsec_python.ml_density.v1``):

    density            (M,)    the density at the points, in their order
    coordinates_bohr   (M, 3)  ``target_coordinates_bohr`` of the request
    units              ()      "e_per_angstrom3" or "e_per_bohr3"
    provider           ()      a short name of the model
    metadata_<key>     ()      optional strings: model, checkpoint, device, ...

Exit with a nonzero status and a message on stderr when the prediction
fails.  Lines printed (to stdout or stderr) that start with ``progress:``
are shown to the user while the model runs.  PARSEC.py validates the
result, clips small negative values, normalizes the integral to the
valence-electron count, and uses it only as the first SCF density.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


BOHR_TO_ANGSTROM = 0.529177210903
# Nominal valence charges of the toy model.  A real model learns these.
_VALENCE = {"H": 1, "He": 2, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7}


def predict_density(
    symbols: list[str],
    atoms_angstrom: np.ndarray,
    points_angstrom: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    """Replace this function by your model.  Returns (density, units)."""

    exponent = 1.0  # 1/bohr: a hydrogen 1s orbital
    density = np.zeros(points_angstrom.shape[0])
    for start in range(0, points_angstrom.shape[0], args.chunk_size):
        chunk = points_angstrom[start : start + args.chunk_size]
        for symbol, atom in zip(symbols, atoms_angstrom):
            distance = np.linalg.norm(chunk - atom, axis=1) / BOHR_TO_ANGSTROM
            density[start : start + chunk.shape[0]] += (
                _VALENCE.get(symbol, 1) * exponent**3 / np.pi * np.exp(-2.0 * exponent * distance)
            )
    return density, "e_per_bohr3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="custom")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--repository")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    started = time.perf_counter()
    with np.load(args.request, allow_pickle=False) as request:
        if str(request["schema"]) != "parsec_python.ml_request.v1":
            sys.exit(f"unsupported request schema {request['schema']}")
        symbols = [str(value) for value in request["symbols"]]
        atoms = np.asarray(request["atom_positions_angstrom"], dtype=float)
        points = np.asarray(request["probe_positions_angstrom"], dtype=float)
        target = np.asarray(request["target_coordinates_bohr"], dtype=float)
    print(f"progress: {len(symbols)} atoms, {points.shape[0]:,} points", flush=True)

    density, units = predict_density(symbols, atoms, points, args)

    density = np.asarray(density, dtype=np.float64)
    if density.shape != (points.shape[0],) or not np.all(np.isfinite(density)):
        sys.exit("the model returned an invalid density vector")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray("parsec_python.ml_density.v1"),
        density=density,
        coordinates_bohr=target,
        units=np.asarray(units),
        provider=np.asarray("template"),
        metadata_model=np.asarray(args.model),
        metadata_device=np.asarray("cpu"),
        metadata_seconds=np.asarray(f"{time.perf_counter() - started:.3f}"),
    )
    print(f"progress: done in {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
