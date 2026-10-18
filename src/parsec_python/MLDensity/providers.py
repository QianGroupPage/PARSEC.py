"""Model-independent orchestration for ML initial-density providers.

SCDP and ChargE3Net have mutually incompatible (and fairly heavy) dependency
stacks.  PARSEC.py therefore exchanges a small ``.npz`` request/result with a
provider-specific Python process, the bridge (protocol in
``bridges/template.py``; models in :mod:`.registry`).  This keeps
PyTorch/e3nn out of the DFT environment and makes a cached prediction
indistinguishable from a density supplied directly by the user.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from ..Grid import RealSpaceGrid
from ..models import Atom, InitialDensitySettings, SpeciesPotential
from .field import (
    BOHR_TO_ANGSTROM,
    DensityLoadResult,
    load_density_for_grid,
    normalize_density_units,
)
from .registry import DensityProvider, get_provider


_REQUEST_SCHEMA = "parsec_python.ml_request.v1"
# Hex digits of the cache key in an entry's directory name.  The full key is
# kept in the entry's manifest; a short name keeps deep Windows paths under
# the 248-character limit of a directory.
_ENTRY_KEY_LENGTH = 16
ProgressCallback = Callable[[str], None]


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


# A start whose electron count is off by more than this fraction is reported:
# a model that far off is outside its training domain.
_ELECTRON_COUNT_TOLERANCE = 0.10


def electron_count_warning(integral: float, electrons: float, source: str) -> str | None:
    """A message when a starting density holds the wrong number of electrons.

    Normalization rescales any start to ``electrons``; when the model
    missed that number badly (SCDP gives tens of thousands of electrons for
    a 71-atom nanodiamond), the shape of the density is no better and the
    start can be worse than SAD.
    """

    if electrons <= 0 or abs(integral / electrons - 1.0) <= _ELECTRON_COUNT_TOLERANCE:
        return None
    return (
        f"the initial density from {source} holds {integral:.4g} electrons, the "
        f"calculation {electrons:g}: the model is probably outside its training "
        f"domain; normalization scales it by {electrons / integral:.3g}. "
        "Compare the SCF with Initial_Density: sad"
    )


def physical_symbols(
    atoms: Sequence[Atom], specifications: Mapping[str, SpeciesPotential]
) -> list[str]:
    """Chemical symbols of ``atoms`` as a density model receives them."""

    return [_physical_symbol(atom.symbol, specifications) for atom in atoms]


def _candidate_workspace() -> Path | None:
    configured = os.environ.get("PARSEC_ML_WORKSPACE")
    if configured:
        return Path(configured).expanduser()
    # The recommended local layout places the workspace beside parsec_python.
    checkout = Path(__file__).resolve().parents[3]
    sibling = checkout.parent / "parsec_ml_workspace"
    return sibling if sibling.is_dir() else None


@dataclass(frozen=True)
class ResolvedProvider:
    """A density model with every path it runs from.

    ``origins`` says where each path came from (the input, an environment
    variable, the workspace, ...), for ``ml_density.py check``.
    """

    provider: DensityProvider
    model: str
    bridge: Path
    python: Path
    repository: Path | None
    checkpoint: Path | None
    origins: Mapping[str, str]

    @property
    def name(self) -> str:
        return self.provider.name


def _resolve_repository(
    provider: DensityProvider, configured: Path | None
) -> tuple[Path | None, str]:
    candidates: list[tuple[Path, str]] = []
    if configured is not None:
        candidates.append((Path(configured).expanduser(), "given"))
    variable = provider.repository_variable
    if variable and os.environ.get(variable):
        candidates.append((Path(os.environ[variable]).expanduser(), variable))
    workspace = _candidate_workspace()
    if workspace is not None and provider.needs_repository:
        candidates.append((workspace / provider.name, "PARSEC_ML_WORKSPACE"))
    for candidate, origin in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved, origin
    if configured is not None:
        raise FileNotFoundError(f"ML_Density_Repository does not exist: {configured}")
    if not provider.needs_repository:
        return None, "not used"
    hint = f"set ML_Density_Repository, {variable}," if variable else "set ML_Density_Repository"
    raise FileNotFoundError(
        f"cannot locate the {provider.title} repository; {hint} or "
        f"PARSEC_ML_WORKSPACE (official source: {provider.source})"
    )


def _resolve_python(
    provider: DensityProvider, configured: Path | None, repository: Path | None
) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    if configured is not None:
        candidates.append((Path(configured).expanduser(), "given"))
    variable = provider.python_variable
    environment = os.environ.get(variable) if variable else None
    if environment:
        candidates.append((Path(environment).expanduser(), variable))
    if repository is not None:
        executable_name = "python.exe" if os.name == "nt" else "python"
        for folder in ("Scripts", "bin"):
            candidates.append(
                (repository / ".venv" / folder / executable_name, "repository .venv")
            )
    for candidate, origin in candidates:
        # Absolute, but symbolic links are kept: the ``bin/python`` of a
        # Linux venv links to the base interpreter, which run under its own
        # name would not see the packages of the venv.
        absolute = Path(os.path.abspath(candidate))
        if absolute.is_file():
            return absolute, origin
    if configured is None and environment is None:
        # No model environment was named or found: the bridge then runs in
        # the DFT interpreter and reports any missing dependency itself.
        return Path(os.path.abspath(sys.executable)), "this interpreter (no model environment found)"
    raise FileNotFoundError(
        f"ML provider Python executable does not exist: {candidates[0][0]}"
    )


def _resolve_checkpoint(
    provider: DensityProvider,
    configured: Path | None,
    repository: Path | None,
    model: str,
) -> tuple[Path | None, str]:
    if configured is not None:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate, "given"
        raise FileNotFoundError(f"ML density checkpoint does not exist: {candidate}")
    if provider.find_checkpoint is not None and repository is not None:
        found = provider.find_checkpoint(repository, model)
        if found is not None:
            return found.resolve(), "found in the repository"
    if not provider.needs_checkpoint:
        return None, "not used"
    raise FileNotFoundError(
        f"no {provider.title} checkpoint for model {model!r}: {provider.checkpoint_hint}"
    )


def resolve_provider(
    name: str,
    *,
    model: str | None = None,
    repository: str | Path | None = None,
    python: str | Path | None = None,
    checkpoint: str | Path | None = None,
    bridge: str | Path | None = None,
) -> ResolvedProvider:
    """Find the bridge, interpreter, checkout, and checkpoint of a model."""

    provider = get_provider(name)
    model = provider.check_model(provider.default_model if model is None else model)
    if provider.bridge is not None:
        if bridge is not None:
            raise ValueError(
                f"ML_Density_Bridge is for Initial_Density: custom; {provider.title} "
                "has its own bridge"
            )
        bridge_path, bridge_origin = provider.bridge, "built in"
    else:
        if bridge is None:
            raise ValueError(
                f"Initial_Density: {provider.name} needs ML_Density_Bridge, the "
                "script that runs the model"
            )
        bridge_path = Path(bridge).expanduser().resolve()
        bridge_origin = "given"
        if not bridge_path.is_file():
            raise FileNotFoundError(f"ML density bridge does not exist: {bridge_path}")
    repository_path, repository_origin = _resolve_repository(
        provider, None if repository is None else Path(repository)
    )
    python_path, python_origin = _resolve_python(
        provider, None if python is None else Path(python), repository_path
    )
    checkpoint_path, checkpoint_origin = _resolve_checkpoint(
        provider,
        None if checkpoint is None else Path(checkpoint),
        repository_path,
        model,
    )
    return ResolvedProvider(
        provider=provider,
        model=model,
        bridge=bridge_path,
        python=python_path,
        repository=repository_path,
        checkpoint=checkpoint_path,
        origins={
            "bridge": bridge_origin,
            "python": python_origin,
            "repository": repository_origin,
            "checkpoint": checkpoint_origin,
        },
    )


def resolve_settings(settings: InitialDensitySettings) -> ResolvedProvider:
    """:func:`resolve_provider` for the model of ``settings``."""

    return resolve_provider(
        settings.method,
        model=settings.model,
        repository=settings.repository,
        python=settings.python_executable,
        checkpoint=settings.checkpoint,
        bridge=settings.bridge,
    )


def _source_digest(digest, resolved: ResolvedProvider) -> None:
    """Hash everything outside PARSEC.py that a prediction depends on."""

    digest.update(resolved.name.encode("utf-8"))
    digest.update(resolved.model.encode("utf-8"))
    digest.update(resolved.bridge.read_bytes())
    digest.update(str(resolved.bridge).encode("utf-8"))
    digest.update(str(resolved.python).encode("utf-8"))
    if resolved.checkpoint is not None:
        stat = resolved.checkpoint.stat()
        digest.update(str(resolved.checkpoint).encode("utf-8"))
        digest.update(
            np.asarray([stat.st_size, stat.st_mtime_ns], dtype=np.int64).tobytes()
        )
    repository = resolved.repository
    if repository is not None:
        digest.update(str(repository).encode("utf-8"))
        # A changed model implementation must not reuse a prediction made
        # by older code: include the checked-out revision when available.
        head = repository / ".git" / "HEAD"
        if head.is_file():
            head_text = head.read_text(encoding="utf-8").strip()
            digest.update(head_text.encode("ascii", errors="replace"))
            if head_text.startswith("ref: "):
                ref = repository / ".git" / head_text[5:]
                if ref.is_file():
                    digest.update(ref.read_bytes())


def provider_source_fingerprint(settings: InitialDensitySettings) -> bytes:
    """Fingerprint external code/checkpoint inputs for resident DFT caches."""

    if settings.method in {"sad", "file"}:
        raise ValueError("a provider fingerprint requires a density model")
    digest = sha256()
    _source_digest(digest, resolve_settings(settings))
    return digest.digest()


def _cache_key(
    resolved: ResolvedProvider,
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    symbols: Sequence[str],
) -> str:
    digest = sha256()
    digest.update(b"parsec-python-ml-density-v2")
    _source_digest(digest, resolved)
    digest.update(repr(grid.settings).encode("utf-8"))
    digest.update(np.ascontiguousarray(grid.coordinates, dtype=np.float64).tobytes())
    for atom, symbol in zip(atoms, symbols):
        digest.update(symbol.encode("ascii"))
        digest.update(np.ascontiguousarray(atom.position, dtype=np.float64).tobytes())
    return digest.hexdigest()


def write_request(
    path: Path,
    symbols: Sequence[str],
    atom_positions_bohr: np.ndarray,
    points_bohr: np.ndarray,
    *,
    lower_bohr: np.ndarray,
    cell_bohr: np.ndarray,
) -> None:
    """Write a bridge request with atoms and points in a positive box.

    ``lower_bohr`` is the lower corner of an orthorhombic box of edges
    ``cell_bohr`` that holds every point.  Atoms and points are translated
    together by ``-lower_bohr``; every supported model runs with PBC
    disabled, so the translation cannot change their prediction.
    """

    translation_bohr = -np.asarray(lower_bohr, dtype=float)
    atoms = np.asarray(atom_positions_bohr, dtype=float).reshape(-1, 3)
    points = np.asarray(points_bohr, dtype=float).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema=np.asarray(_REQUEST_SCHEMA),
        symbols=np.asarray(symbols, dtype="U3"),
        atom_positions_angstrom=(atoms + translation_bohr) * BOHR_TO_ANGSTROM,
        probe_positions_angstrom=(points + translation_bohr) * BOHR_TO_ANGSTROM,
        target_coordinates_bohr=points,
        cell_angstrom=np.diag(np.asarray(cell_bohr, dtype=float)) * BOHR_TO_ANGSTROM,
        periodic=np.asarray(False),
    )


def _grid_box(grid: RealSpaceGrid) -> tuple[np.ndarray, np.ndarray]:
    """Lower corner and edges of the Cartesian box of a PARSEC grid."""

    spacing = grid.spacing
    lower = (grid.index_min + np.asarray(grid.settings.shift) - 0.5) * spacing
    return lower, np.asarray(grid.lookup.shape) * spacing


def run_bridge(
    resolved: ResolvedProvider,
    request: Path,
    output: Path,
    *,
    device: str = "auto",
    chunk_size: int = 50000,
    progress: ProgressCallback | None = None,
    matplotlib_cache: Path | None = None,
) -> None:
    """Run the model on ``request`` and leave its result at ``output``.

    The bridge writes to a temporary name that is renamed only after a
    successful exit, so an interrupted run never leaves a partial result
    for a later run to reuse.
    """

    partial = output.with_name(output.stem + ".partial.npz")
    command = [
        str(resolved.python),
        str(resolved.bridge),
        "--request",
        str(request),
        "--output",
        str(partial),
        "--model",
        resolved.model,
        "--device",
        device,
        "--chunk-size",
        str(int(chunk_size)),
    ]
    if resolved.repository is not None:
        command += ["--repository", str(resolved.repository)]
    if resolved.checkpoint is not None:
        command += ["--checkpoint", str(resolved.checkpoint)]
    # Isolated model environments often import matplotlib indirectly.  Give
    # those subprocesses a cache directory beside the request instead of
    # assuming that the user's home directory is writable.
    environment = os.environ.copy()
    matplotlib_cache = matplotlib_cache or request.parent / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    for name, value in resolved.provider.environment.items():
        environment.setdefault(name, value)
    working_directory = resolved.repository or request.parent
    partial.unlink(missing_ok=True)
    tail: deque[str] = deque(maxlen=80)
    with subprocess.Popen(
        command,
        cwd=working_directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            tail.append(line)
            if progress is not None and line.startswith("progress:"):
                progress(line[len("progress:"):].strip())
        returncode = process.wait()
    if returncode != 0:
        detail = "\n".join(line for line in tail if line)
        raise RuntimeError(
            f"{resolved.provider.title} density prediction failed in "
            f"{resolved.python}\ncommand: {' '.join(command)}\n{detail}"
        )
    if not partial.is_file():
        raise RuntimeError(
            f"{resolved.provider.title} bridge completed without creating {partial}"
        )
    os.replace(partial, output)


@dataclass(frozen=True)
class PointPrediction:
    """A model density at given points, in electrons per bohr^3."""

    density: np.ndarray
    seconds: float
    metadata: Mapping[str, str]


def _read_prediction(path: Path, points_bohr: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["density"], dtype=float)
        coordinates = np.asarray(archive["coordinates_bohr"], dtype=float)
        units = normalize_density_units(str(np.asarray(archive["units"]).item()))
        metadata = {
            key[len("metadata_"):]: str(np.asarray(archive[key]).item())
            for key in archive.files
            if key.startswith("metadata_")
        }
    if values.shape != (points_bohr.shape[0],) or not np.all(np.isfinite(values)):
        raise RuntimeError("the model returned an invalid density vector")
    if coordinates.shape != points_bohr.shape or not np.allclose(
        coordinates, points_bohr, rtol=0.0, atol=1.0e-10
    ):
        raise RuntimeError("the model returned the density at other points")
    if units == "e_per_angstrom3":
        values = values * BOHR_TO_ANGSTROM**3
    elif units != "e_per_bohr3":
        raise RuntimeError(f"a bridge must return a volume density, not {units}")
    return values, metadata


def predict_density(
    provider: str | ResolvedProvider,
    symbols: Sequence[str],
    atom_positions_bohr: np.ndarray,
    points_bohr: np.ndarray,
    *,
    model: str | None = None,
    repository: str | Path | None = None,
    python: str | Path | None = None,
    checkpoint: str | Path | None = None,
    bridge: str | Path | None = None,
    device: str = "auto",
    chunk_size: int = 50000,
    lower_bohr: np.ndarray | None = None,
    cell_bohr: np.ndarray | None = None,
    work_directory: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> PointPrediction:
    """Run a density model at arbitrary points of an isolated system.

    Positions are Cartesian bohr in any common frame.  The raw model density
    is returned in electrons per bohr^3: no clipping or normalization (the
    SCF import does both).  ``lower_bohr``/``cell_bohr`` default to the
    bounding box of the points with half a spacing of margin.
    """

    resolved = (
        provider
        if isinstance(provider, ResolvedProvider)
        else resolve_provider(
            provider,
            model=model,
            repository=repository,
            python=python,
            checkpoint=checkpoint,
            bridge=bridge,
        )
    )
    atoms = np.asarray(atom_positions_bohr, dtype=float).reshape(-1, 3)
    points = np.asarray(points_bohr, dtype=float).reshape(-1, 3)
    if len(symbols) != atoms.shape[0] or atoms.shape[0] == 0:
        raise ValueError("give one chemical symbol per atom")
    if points.shape[0] == 0:
        raise ValueError("no points to predict at")
    if lower_bohr is None or cell_bohr is None:
        everything = np.vstack([atoms, points])
        margin = 0.5
        lower_bohr = everything.min(axis=0) - margin
        cell_bohr = everything.max(axis=0) - everything.min(axis=0) + 2 * margin
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="parsec-ml-", dir=None if work_directory is None else str(work_directory)
    ) as directory:
        folder = Path(directory)
        request = folder / "request.npz"
        output = folder / "density.npz"
        write_request(
            request,
            symbols,
            atoms,
            points,
            lower_bohr=np.asarray(lower_bohr, dtype=float),
            cell_bohr=np.asarray(cell_bohr, dtype=float),
        )
        run_bridge(
            resolved,
            request,
            output,
            device=device,
            chunk_size=chunk_size,
            progress=progress,
        )
        values, metadata = _read_prediction(output, points)
    return PointPrediction(
        density=values,
        seconds=time.perf_counter() - started,
        metadata=metadata,
    )


def build_initial_density(
    settings: InitialDensitySettings,
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    specifications: Mapping[str, SpeciesPotential],
    *,
    progress: ProgressCallback | None = None,
) -> DensityLoadResult:
    """Return a validated ML/file density on the exact active DFT grid."""

    if settings.method == "sad":
        raise ValueError("SAD is constructed by the SCF preparation path, not MLDensity")
    positions = np.asarray([atom.position for atom in atoms], dtype=float)
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
            atom_positions_bohr=positions,
        )

    symbols = physical_symbols(atoms, specifications)
    resolved = resolve_settings(settings)
    key = _cache_key(resolved, grid, atoms, symbols)
    cache_root = (
        Path(settings.cache_directory).resolve()
        if settings.cache_directory is not None
        else Path.cwd().resolve() / ".parsec_ml_density_cache"
    )
    entry = cache_root / f"{resolved.name}-{key[:_ENTRY_KEY_LENGTH]}"
    request = entry / "request.npz"
    output = entry / "density.npz"
    manifest_path = entry / "manifest.json"
    reusable = False
    if not settings.regenerate and output.is_file() and manifest_path.is_file():
        try:
            reusable = json.loads(manifest_path.read_text(encoding="utf-8")).get("key") == key
        except (OSError, ValueError):
            reusable = False
    if not reusable:
        lower, cell = _grid_box(grid)
        write_request(request, symbols, positions, grid.coordinates, lower_bohr=lower, cell_bohr=cell)
        started = time.perf_counter()
        run_bridge(
            resolved,
            request,
            output,
            device=settings.device,
            chunk_size=settings.prediction_chunk_size,
            progress=progress,
            matplotlib_cache=cache_root / "matplotlib",
        )
        manifest = {
            "key": key,
            "provider": resolved.name,
            "model": resolved.model,
            "bridge": str(resolved.bridge),
            "repository": None if resolved.repository is None else str(resolved.repository),
            "checkpoint": None if resolved.checkpoint is None else str(resolved.checkpoint),
            "python": str(resolved.python),
            "grid_points": int(grid.size),
            "prediction_seconds": round(time.perf_counter() - started, 3),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return load_density_for_grid(
        output,
        grid,
        units="auto",
        interpolation=settings.interpolation,
        negative_policy=settings.negative_policy,
    )


__all__ = [
    "PointPrediction",
    "ResolvedProvider",
    "build_initial_density",
    "electron_count_warning",
    "physical_symbols",
    "predict_density",
    "provider_source_fingerprint",
    "resolve_provider",
    "resolve_settings",
    "run_bridge",
    "write_request",
]
