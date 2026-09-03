"""Model-independent orchestration for ML initial-density providers.

SCDP and ChargE3Net have mutually incompatible (and fairly heavy) dependency
stacks.  PARSEC.py therefore exchanges a small ``.npz`` request/result with a
provider-specific Python process.  This keeps PyTorch/e3nn out of the DFT
environment and makes a cached prediction indistinguishable from a density
supplied directly by the user.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np

from ..Grid import RealSpaceGrid
from ..models import Atom, InitialDensitySettings, SpeciesPotential
from .field import BOHR_TO_ANGSTROM, DensityLoadResult, load_density_for_grid


_PROVIDER_ENV = {
    "charge3net": "CHARGE3NET_REPO",
    "scdp": "SCDP_REPO",
}
_PYTHON_ENV = {
    "charge3net": "CHARGE3NET_PYTHON",
    "scdp": "SCDP_PYTHON",
}


def _physical_symbol(
    species: str, specifications: Mapping[str, SpeciesPotential]
) -> str:
    """Map a PP label such as ``C-1s`` back to its chemical element."""

    explicit = specifications[species].element_symbol
    if explicit:
        return explicit
    import re

    match = re.match(r"^([A-Z][a-z]?)", species)
    if match is None:
        raise ValueError(
            f"cannot infer a chemical element from species label {species!r}; "
            "set Element_Symbol for ML-density prediction"
        )
    return match.group(1)


def _candidate_workspace() -> Path | None:
    configured = os.environ.get("PARSEC_ML_WORKSPACE")
    if configured:
        return Path(configured).expanduser()
    # The recommended local layout places the workspace beside parsec_python.
    checkout = Path(__file__).resolve().parents[3]
    sibling = checkout.parent / "parsec_ml_workspace"
    return sibling if sibling.is_dir() else None


def _resolve_repository(provider: str, configured: Path | None) -> Path:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    environment = os.environ.get(_PROVIDER_ENV[provider])
    if environment:
        candidates.append(Path(environment).expanduser())
    workspace = _candidate_workspace()
    if workspace is not None:
        candidates.append(workspace / provider)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    option = "ML_Density_Repository"
    env = _PROVIDER_ENV[provider]
    raise FileNotFoundError(
        f"cannot locate the {provider} repository; set {option}, {env}, or "
        "PARSEC_ML_WORKSPACE"
    )


def _resolve_python(
    provider: str, configured: Path | None, repository: Path
) -> Path:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(configured)
    environment = os.environ.get(_PYTHON_ENV[provider])
    if environment:
        candidates.append(Path(environment).expanduser())
    executable_name = "python.exe" if os.name == "nt" else "python"
    candidates.extend(
        [
            repository / ".venv" / "Scripts" / executable_name,
            repository / ".venv" / "bin" / executable_name,
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    if configured is None and environment is None:
        return Path(sys.executable).resolve()
    raise FileNotFoundError(f"ML provider Python executable does not exist: {candidates[0]}")


def _resolve_checkpoint(
    provider: str,
    configured: Path | None,
    repository: Path,
    model: str,
) -> Path:
    if configured is not None:
        candidate = configured.resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"ML density checkpoint does not exist: {candidate}")
    if provider == "charge3net":
        candidate = repository / "models" / f"charge3net_{model}.pt"
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(
            f"ChargE3Net model {model!r} was not found at {candidate}; "
            "set ML_Density_Checkpoint explicitly"
        )

    # SCDP publishes weights separately from its Git repository.  Accept a
    # single unambiguous checkpoint if the user placed it below models/.
    candidates = sorted((repository / "models").glob("**/*.ckpt"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise FileNotFoundError(
        "SCDP checkpoints are distributed separately on Zenodo; set "
        "ML_Density_Checkpoint to the desired .ckpt file"
    )


def provider_source_fingerprint(settings: InitialDensitySettings) -> bytes:
    """Fingerprint external code/checkpoint inputs for resident DFT caches."""

    provider = settings.method
    if provider not in {"charge3net", "scdp"}:
        raise ValueError("a provider fingerprint requires charge3net or scdp")
    repository = _resolve_repository(provider, settings.repository)
    checkpoint = _resolve_checkpoint(
        provider, settings.checkpoint, repository, settings.model
    )
    digest = sha256()
    digest.update(provider.encode("ascii"))
    digest.update(settings.model.encode("utf-8"))
    digest.update(
        (Path(__file__).with_name("bridges") / f"{provider}.py").read_bytes()
    )
    digest.update(str(repository).encode("utf-8"))
    digest.update(str(checkpoint).encode("utf-8"))
    stat = checkpoint.stat()
    digest.update(np.asarray([stat.st_size, stat.st_mtime_ns], dtype=np.int64).tobytes())
    head = repository / ".git" / "HEAD"
    if head.is_file():
        head_text = head.read_text(encoding="utf-8").strip()
        digest.update(head_text.encode("ascii", errors="replace"))
        if head_text.startswith("ref: "):
            ref = repository / ".git" / head_text[5:]
            if ref.is_file():
                digest.update(ref.read_bytes())
    return digest.digest()


def _cache_key(
    provider: str,
    settings: InitialDensitySettings,
    repository: Path,
    checkpoint: Path,
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    symbols: Sequence[str],
) -> str:
    digest = sha256()
    digest.update(b"parsec-python-ml-density-v1")
    digest.update(provider.encode())
    bridge = Path(__file__).with_name("bridges") / f"{provider}.py"
    digest.update(bridge.read_bytes())
    digest.update(settings.model.encode())
    digest.update(str(repository).encode("utf-8"))
    stat = checkpoint.stat()
    digest.update(str(checkpoint).encode("utf-8"))
    digest.update(np.asarray([stat.st_size, stat.st_mtime_ns], dtype=np.int64).tobytes())
    # Include the checked-out repository revision when available.  A changed
    # model implementation must not reuse a prediction made by older code.
    head = repository / ".git" / "HEAD"
    if head.is_file():
        head_text = head.read_text(encoding="utf-8").strip()
        digest.update(head_text.encode("ascii", errors="replace"))
        if head_text.startswith("ref: "):
            ref = repository / ".git" / head_text[5:]
            if ref.is_file():
                digest.update(ref.read_bytes())
    digest.update(repr(grid.settings).encode("utf-8"))
    digest.update(np.ascontiguousarray(grid.coordinates, dtype=np.float64).tobytes())
    for atom, symbol in zip(atoms, symbols):
        digest.update(symbol.encode("ascii"))
        digest.update(np.ascontiguousarray(atom.position, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _write_request(
    path: Path,
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    symbols: Sequence[str],
) -> None:
    """Write exact DFT probes using a harmless translated molecular box.

    The translation places the underlying Cartesian grid inside a positive
    orthorhombic cell.  Both supported molecular models run with PBC disabled,
    so translating atoms and probes together cannot change their prediction.
    """

    spacing = grid.spacing
    lower_bohr = (grid.index_min + np.asarray(grid.settings.shift) - 0.5) * spacing
    translation_bohr = -lower_bohr
    atom_positions_bohr = np.asarray([atom.position for atom in atoms], dtype=float)
    atom_positions_angstrom = (
        atom_positions_bohr + translation_bohr[None, :]
    ) * BOHR_TO_ANGSTROM
    probe_positions_angstrom = (
        grid.coordinates + translation_bohr[None, :]
    ) * BOHR_TO_ANGSTROM
    cell_angstrom = np.diag(np.asarray(grid.lookup.shape) * spacing * BOHR_TO_ANGSTROM)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray("parsec_python.ml_request.v1"),
        symbols=np.asarray(symbols, dtype="U3"),
        atom_positions_angstrom=atom_positions_angstrom,
        probe_positions_angstrom=probe_positions_angstrom,
        target_coordinates_bohr=np.asarray(grid.coordinates, dtype=float),
        cell_angstrom=cell_angstrom,
        periodic=np.asarray(False),
    )


def _invoke_provider(
    provider: str,
    settings: InitialDensitySettings,
    repository: Path,
    checkpoint: Path,
    python: Path,
    request: Path,
    output: Path,
) -> None:
    bridge = Path(__file__).with_name("bridges") / f"{provider}.py"
    command = [
        str(python),
        str(bridge),
        "--request",
        str(request),
        "--output",
        str(output),
        "--repository",
        str(repository),
        "--checkpoint",
        str(checkpoint),
        "--model",
        settings.model,
        "--device",
        settings.device,
        "--chunk-size",
        str(settings.prediction_chunk_size),
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"{provider} density prediction failed in {python}\n"
            f"command: {' '.join(command)}\n{detail}"
        )
    if not output.is_file():
        raise RuntimeError(f"{provider} bridge completed without creating {output}")


def build_initial_density(
    settings: InitialDensitySettings,
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    specifications: Mapping[str, SpeciesPotential],
) -> DensityLoadResult:
    """Return a validated ML/file density on the exact active DFT grid."""

    if settings.method == "sad":
        raise ValueError("SAD is constructed by the SCF preparation path, not MLDensity")
    if settings.file is not None:
        # A named provider may also consume a prediction made previously by
        # that provider.  This is useful for regression tests, offline runs,
        # and moving densities between machines: ``Initial_Density`` retains
        # the scientific provenance (SCDP or ChargE3Net), while the portable
        # file supplies the actual field and avoids loading the model stack.
        # If no file is supplied, the ordinary direct-provider path below is
        # unchanged.
        return load_density_for_grid(
            settings.file,
            grid,
            units=settings.units,
            interpolation=settings.interpolation,
            negative_policy=settings.negative_policy,
        )

    provider = settings.method
    repository = _resolve_repository(provider, settings.repository)
    python = _resolve_python(provider, settings.python_executable, repository)
    checkpoint = _resolve_checkpoint(
        provider, settings.checkpoint, repository, settings.model
    )
    symbols = [_physical_symbol(atom.symbol, specifications) for atom in atoms]
    key = _cache_key(
        provider, settings, repository, checkpoint, grid, atoms, symbols
    )
    cache_root = (
        settings.cache_directory.resolve()
        if settings.cache_directory is not None
        else Path.cwd().resolve() / ".parsec_ml_density_cache"
    )
    entry = cache_root / f"{provider}-{key}"
    request = entry / "request.npz"
    output = entry / "density.npz"
    if settings.regenerate or not output.is_file():
        _write_request(request, grid, atoms, symbols)
        _invoke_provider(
            provider,
            settings,
            repository,
            checkpoint,
            python,
            request,
            output,
        )
        manifest = {
            "provider": provider,
            "model": settings.model,
            "repository": str(repository),
            "checkpoint": str(checkpoint),
            "python": str(python),
        }
        (entry / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return load_density_for_grid(
        output,
        grid,
        units="auto",
        interpolation=settings.interpolation,
        negative_policy=settings.negative_policy,
    )


__all__ = ["build_initial_density", "provider_source_fingerprint"]
