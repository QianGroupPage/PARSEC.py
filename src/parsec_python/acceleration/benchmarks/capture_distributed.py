"""Capture one actual symmetry-sector Hamiltonian and SCF Ritz basis.

Run as ``python -m ...benchmarks.capture_distributed DIRECTORY parsec.in``
followed by the ordinary accelerated CLI arguments. Run on an allocated GPU.
PARSEC_MPI_CAPTURE_ITERATION (default 3) counts completed outer symmetry
eigensolver calls; PARSEC_MPI_CAPTURE_SECTOR (default 0) chooses the sector;
PARSEC_MPI_CAPTURE_MAX_COLUMNS (default 0, all) optionally truncates columns.

The ordinary solver is unmodified and a completed capture deliberately stops
before the density update/convergence decision for that SCF iteration. This is
replay data, NOT a converged SCF result. The completion manifest is written last.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np


class CaptureComplete(SystemExit):
    """Successful intentional termination, not an SCF convergence claim."""

    def __str__(self):
        return "capture complete; deliberately stopped before SCF convergence"


def _env_integer(name, default, minimum=0):
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance():
    root = next((item for item in Path(__file__).resolve().parents if (item / ".git").exists()), None)
    result = {"root": str(root) if root else None, "commit": None, "status": None}
    if root:
        for key, arguments in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
            try:
                run = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, timeout=15, check=True)
                result[key] = run.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                result[key] = None
    result["capture_script_sha256"] = _sha256(Path(__file__))
    return result


def write_basis_memmap(path, vectors, *, columns=None, chunk_columns=32, to_host=None):
    """Write finite FP64 vectors with at most 32 columns staged on the host.

    ``to_host`` receives only a column slice. The default supports NumPy arrays;
    the GPU caller supplies ``cupy.asnumpy``. Output is Fortran-contiguous so
    these column chunks are contiguous on disk as well as in the source basis.
    """
    if len(vectors.shape) != 2 or np.dtype(vectors.dtype) != np.dtype(np.float64):
        raise ValueError("capture requires a two-dimensional FP64 basis")
    if not isinstance(chunk_columns, int) or not 1 <= chunk_columns <= 32:
        raise ValueError("chunk_columns must be in [1, 32]")
    rows, available = (int(item) for item in vectors.shape)
    columns = available if columns is None else int(columns)
    if rows < 1 or not 1 <= columns <= available:
        raise ValueError("invalid captured basis dimensions")
    to_host = np.asarray if to_host is None else to_host
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                      shape=(rows, columns), fortran_order=True)
    try:
        for start in range(0, columns, chunk_columns):
            stop = min(start + chunk_columns, columns)
            block = np.asarray(to_host(vectors[:, start:stop]))
            if block.dtype != np.float64 or block.shape != (rows, stop - start):
                raise ValueError("host transfer changed basis dtype or dimensions")
            if not np.isfinite(block).all():
                raise ValueError("captured basis contains nonfinite values")
            target[:, start:stop] = block
        target.flush()
    finally:
        del target
    return rows, columns


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    destination = Path(arguments.pop(0)).expanduser().resolve()
    if not arguments:
        raise ValueError("an ordinary accelerated-CLI input is required after DIRECTORY")
    iteration = _env_integer("PARSEC_MPI_CAPTURE_ITERATION", 3, 1)
    sector = _env_integer("PARSEC_MPI_CAPTURE_SECTOR", 0)
    max_columns = _env_integer("PARSEC_MPI_CAPTURE_MAX_COLUMNS", 0)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"capture directory must be empty: {destination}")
    marker = destination / "CAPTURE_INCOMPLETE"
    marker.write_text("No completed capture exists until capture.json is written.\n", encoding="utf-8")

    from parsec_python.acceleration import cli as accelerated_cli
    from parsec_python.acceleration.Eigensolvers import symmetry
    from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition as Decomposition
    from parsec_python.acceleration.backends.cupy import _host_csr, require_cupy

    cp, _ = require_cupy()
    held = {}
    build = Decomposition.build.__func__
    load_operators = symmetry.load_or_build_reduced_operators
    solve = symmetry.CuPySymmetrySCFEigensolver.__call__
    parse_input = accelerated_cli.parse_parsec_input
    source_git = _git_provenance()
    started = time.perf_counter()

    def parsed(*args, **kwargs):
        translation = parse_input(*args, **kwargs)
        held["translation"] = translation
        paths = [translation.source.resolve()]
        paths += [item.path.resolve() for item in translation.problem.pseudopotentials.values()]
        held["input_hashes"] = {str(path): _sha256(path) for path in dict.fromkeys(paths)}
        return translation

    def built(cls, grid, *args, **kwargs):
        result = build(cls, grid, *args, **kwargs)
        held["grid"] = grid
        held["decomposition"] = result
        return result

    def loaded(decomposition, *args, **kwargs):
        result = load_operators(decomposition, *args, **kwargs)
        held["bundle"] = result
        held["operator_decomposition"] = decomposition
        return result

    def solved(self, *args, **kwargs):
        result = solve(self, *args, **kwargs)
        actual_iteration = int(result.state.solves_completed)
        if actual_iteration != iteration:
            return result
        if sector >= len(self._solvers):
            raise ValueError(f"sector {sector} is unavailable; this problem has {len(self._solvers)} sectors")
        decomposition = self.decomposition
        if decomposition is not held.get("decomposition") or decomposition is not held.get("operator_decomposition"):
            raise RuntimeError("captured grid/operator do not belong to the active symmetry decomposition")
        state = self._solvers[sector].device_state
        if state is None:
            raise RuntimeError("chosen sector state is host-spilled or unavailable; capture with device state storage")
        basis = state.subspace.vectors
        if not isinstance(basis, cp.ndarray):
            raise RuntimeError("chosen sector basis is not on a GPU; capture with device state storage")
        rows, full_columns = (int(item) for item in basis.shape)
        if rows != decomposition.sector_size(sector) or full_columns != int(state.working_states):
            raise RuntimeError("sector state dimensions disagree with its decomposition")
        columns = min(full_columns, max_columns) if max_columns else full_columns
        metadata = held["bundle"].stencil_metadata[sector]
        nonlocal_operator = held["bundle"].nonlocal_operators[sector]
        projectors = _host_csr(nonlocal_operator.projectors)
        signs = np.asarray(nonlocal_operator.signs, dtype=np.float64)
        if metadata.neighbors.shape[1] != rows or projectors.shape != (rows, len(signs)):
            raise RuntimeError("static operator does not match captured basis")
        orbits = decomposition.sector_orbit_indices(sector)
        representative_rows = decomposition.reduction.representative_rows[orbits]
        coords = held["grid"].integer_coordinates[representative_rows]
        multiplicity = decomposition.reduction.multiplicities[orbits]
        device_id = int(basis.device.id)
        operator = self._operators[sector]
        with cp.cuda.Device(device_id):
            cp.cuda.Device(device_id).synchronize()
            potential = cp.asnumpy(operator.effective_potential)
            eigenvalues = cp.asnumpy(state.subspace.eigenvalues)[:columns]
            if potential.dtype != np.float64 or potential.shape != (rows,) or not np.isfinite(potential).all():
                raise ValueError("capture requires a finite FP64 local potential")
            if eigenvalues.dtype != np.float64 or eigenvalues.shape != (columns,) or not np.isfinite(eigenvalues).all():
                raise ValueError("capture requires finite FP64 eigenvalues matching the basis")
            write_basis_memmap(destination / "basis.npy", basis, columns=columns, to_host=cp.asnumpy)
        np.save(destination / "eigenvalues.npy", eigenvalues, allow_pickle=False)
        np.savez(destination / "operator.npz", coordinates=coords,
                 neighbors=metadata.neighbors, codes=metadata.coefficient_codes,
                 palette=metadata.coefficient_palette, local_potential=potential,
                 B_data=projectors.data, B_indices=projectors.indices,
                 B_indptr=projectors.indptr, B_shape=np.asarray(projectors.shape, dtype=np.int64),
                 signs=signs, multiplicity=multiplicity,
                 representative_rows=representative_rows, sector_orbits=orbits)
        # Recheck the small input files; a capture must not label changed inputs
        # with the earlier hashes even if edits occurred while SCF was running.
        if any(_sha256(path) != digest for path, digest in held["input_hashes"].items()):
            raise RuntimeError("an input or pseudopotential changed during capture")
        manifest = {
            "schema": 1, "kind": "single_sector_scf_replay_capture", "complete": True,
            "converged_scf": False,
            "stop_point": "After symmetry eigensolver, before density update and convergence decision",
            "scf_iteration": actual_iteration, "sector": sector,
            "sector_count": len(self._solvers), "sector_solves_completed": int(state.solves_completed),
            "global_requested_states": int(result.state.requested_states),
            "sector_requested_states": int(state.requested_states),
            "full_working_states": int(state.working_states), "captured_columns": columns,
            "columns_truncated": columns != full_columns, "basis_shape": [rows, columns],
            "basis_order": "F", "precision": "float64", "units": "Ry",
            "coordinate_units": "integer multiples of grid spacing",
            "basis_metric": "Euclidean inner product in normalized U_Gamma symmetry representation; do not multiply inner products by orbit multiplicities",
            "hamiltonian": "stencil + diag(local_potential) + B @ diag(signs) @ B.T",
            "projector_shape": list(projectors.shape), "projector_nnz": int(projectors.nnz),
            "stencil_shape": list(metadata.neighbors.shape), "device_id": device_id,
            "node": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_to_capture_seconds": time.perf_counter() - started,
            "source_git": source_git, "input_hashes": held["input_hashes"],
            "cli_arguments": arguments,
            "settings": {key: value for key, value in os.environ.items() if key.startswith("PARSEC_")},
            "files": {name: {"bytes": (destination / name).stat().st_size,
                              "sha256": _sha256(destination / name)}
                      for name in ("operator.npz", "basis.npy", "eigenvalues.npy")},
        }
        manifest_path = destination / "capture.json"
        temporary_path = destination / "capture.json.tmp"
        temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary_path.replace(manifest_path)
        marker.unlink()
        print("DISTRIBUTED_CAPTURE_COMPLETE", json.dumps({"directory": str(destination), "sector": sector,
              "scf_iteration": actual_iteration, "basis_shape": [rows, columns], "converged_scf": False}), flush=True)
        raise CaptureComplete(0)

    with patch.object(Decomposition, "build", classmethod(built)), \
         patch.object(symmetry, "load_or_build_reduced_operators", loaded), \
         patch.object(symmetry.CuPySymmetrySCFEigensolver, "__call__", solved), \
         patch.object(accelerated_cli, "parse_parsec_input", parsed):
        try:
            status = accelerated_cli.main(arguments)
        except CaptureComplete:
            return 0
    raise RuntimeError(f"solver ended with status {status} before requested capture iteration {iteration}")


if __name__ == "__main__":
    raise SystemExit(main())
