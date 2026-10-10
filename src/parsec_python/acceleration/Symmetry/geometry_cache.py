"""Persistent, exactly validated caches for reflection symmetry metadata.

Symmetry detection depends only on the real-space grid, the labeled atomic
geometry, and the two detector tolerances.  Repeating it for every run is
therefore unnecessary.  This module stores only integer orbit/phase metadata;
no Hamiltonian, density, or floating-point physics result is cached.

The cache is content addressed.  Any change to a grid point, atom label,
atomic coordinate, tolerance, or cache format produces a different key.  A
loaded entry is also checked structurally before it is accepted.  Missing,
old, or damaged entries are rebuilt by the same exact routines used when the
cache is disabled.

A key only addresses a cache entry.  Without a cache directory nothing looks
it up, so the full-grid arrays are not hashed and the key is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.models import Atom

from .axis_reflection import AxisReflectionReduction, SignedPermutationReduction
from .representations import ReflectionRepresentationDecomposition


_REDUCTION_FORMAT = 2
_REPRESENTATION_FORMAT = 3


@dataclass(frozen=True)
class SymmetryCacheInfo:
    """Provenance and wall time for one symmetry metadata cache lookup.

    ``key`` is ``None`` when the cache is disabled and no key was hashed.
    """

    status: str
    key: str | None
    path: Path | None
    hash_seconds: float = 0.0
    load_seconds: float = 0.0
    build_seconds: float = 0.0
    write_seconds: float = 0.0


def _update_array(digest, name: str, values: np.ndarray) -> None:
    """Hash one array including its role, dtype, shape, and exact bytes."""

    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))


def _geometry_key(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    atom_tolerance: float,
    lattice_tolerance: float,
) -> str:
    """Return an exact key for every input that can affect detection."""

    digest = sha256()
    digest.update(f"signed-permutation-reduction-v{_REDUCTION_FORMAT}".encode())
    digest.update(np.float64(atom_tolerance).tobytes())
    digest.update(np.float64(lattice_tolerance).tobytes())
    _update_array(digest, "integer_coordinates", grid.integer_coordinates)
    _update_array(digest, "physical_coordinates", grid.coordinates)
    _update_array(digest, "grid_shift", np.asarray(grid.settings.shift, dtype=np.float64))
    for index, atom in enumerate(atoms):
        digest.update(f"atom:{index}:".encode())
        digest.update(str(atom.symbol).encode("utf-8"))
        digest.update(b"\0")
        _update_array(
            digest,
            "position",
            np.asarray(atom.position, dtype=np.float64),
        )
    return digest.hexdigest()


def _representation_key(
    reduction_key: str,
    reduction: AxisReflectionReduction,
) -> str:
    digest = sha256()
    digest.update(
        f"axis-reflection-representations-v{_REPRESENTATION_FORMAT}".encode()
    )
    digest.update(reduction_key.encode("ascii"))
    _update_array(digest, "signs", reduction.signs)
    if isinstance(reduction, SignedPermutationReduction):
        _update_array(digest, "operations", reduction.operations)
        _update_array(digest, "generator_bits", reduction.generator_bits)
    _update_array(digest, "representative_rows", reduction.representative_rows)
    _update_array(digest, "full_to_wedge", reduction.full_to_wedge)
    _update_array(digest, "multiplicities", reduction.multiplicities)
    return digest.hexdigest()


def _validated_reduction(data, full_size: int) -> AxisReflectionReduction:
    """Reconstruct one cache entry only after validating all orbit invariants."""

    format_number = int(np.asarray(data["format"]).reshape(()))
    if format_number != _REDUCTION_FORMAT:
        raise ValueError("unsupported reflection-reduction cache format")
    kind = str(np.asarray(data["kind"]).reshape(()))
    signs = np.ascontiguousarray(data["signs"], dtype=np.int8)
    representatives = np.ascontiguousarray(
        data["representative_rows"], dtype=np.int64
    )
    full_to_wedge = np.ascontiguousarray(data["full_to_wedge"], dtype=np.int64)
    multiplicities = np.ascontiguousarray(data["multiplicities"], dtype=np.int64)
    if signs.ndim != 2 or signs.shape[1] != 3 or not 1 <= signs.shape[0] <= 8:
        raise ValueError("invalid cached reflection signs")
    if not np.all((signs == -1) | (signs == 1)):
        raise ValueError("cached reflection signs are not +/-1")
    if not np.any(np.all(signs == 1, axis=1)):
        raise ValueError("cached reflection group omits identity")
    if representatives.ndim != 1 or representatives.size < 1:
        raise ValueError("invalid cached representative rows")
    if full_to_wedge.shape != (full_size,):
        raise ValueError("cached symmetry map does not match the grid")
    if multiplicities.shape != representatives.shape:
        raise ValueError("cached orbit multiplicities have the wrong shape")
    if np.any(np.diff(representatives) <= 0):
        raise ValueError("cached representative rows are not strictly ordered")
    if representatives[0] < 0 or representatives[-1] >= full_size:
        raise ValueError("cached representative row is outside the grid")
    if np.any(full_to_wedge < 0) or np.any(full_to_wedge >= representatives.size):
        raise ValueError("cached full-to-wedge map is out of bounds")
    expected = np.bincount(full_to_wedge, minlength=representatives.size)
    if not np.array_equal(expected, multiplicities) or np.any(multiplicities <= 0):
        raise ValueError("cached orbit multiplicities are inconsistent")
    if not np.array_equal(
        full_to_wedge[representatives],
        np.arange(representatives.size, dtype=np.int64),
    ):
        raise ValueError("cached rows are not their orbit representatives")
    common = dict(
        signs=signs,
        representative_rows=representatives,
        full_to_wedge=full_to_wedge,
        multiplicities=multiplicities,
    )
    if kind == "axis_reflection":
        return AxisReflectionReduction(**common)
    if kind != "signed_permutation":
        raise ValueError("invalid cached symmetry reduction kind")
    operations = np.ascontiguousarray(data["operations"], dtype=np.int8)
    generator_bits = np.ascontiguousarray(data["generator_bits"], dtype=np.int8)
    order = signs.shape[0]
    if operations.shape != (order, 3, 3):
        raise ValueError("cached signed-permutation operations have the wrong shape")
    if generator_bits.ndim != 2 or generator_bits.shape[0] != order:
        raise ValueError("cached signed-permutation generator bits are invalid")
    if not np.all((generator_bits == 0) | (generator_bits == 1)):
        raise ValueError("cached generator bits are not binary")
    if np.unique(generator_bits, axis=0).shape[0] != order:
        raise ValueError("cached generator bits are not unique")
    identity = np.eye(3, dtype=np.int8)
    for operation in operations:
        if not np.array_equal(operation @ operation, identity):
            raise ValueError("cached operation is not an involution")
        if not np.all(np.sum(np.abs(operation), axis=0) == 1) or not np.all(
            np.sum(np.abs(operation), axis=1) == 1
        ):
            raise ValueError("cached operation is not a signed permutation")
    return SignedPermutationReduction(
        **common,
        operations=operations,
        generator_bits=generator_bits,
    )


def _validated_decomposition(
    data,
    reduction: AxisReflectionReduction,
) -> ReflectionRepresentationDecomposition:
    format_number = int(np.asarray(data["format"]).reshape(()))
    if format_number != _REPRESENTATION_FORMAT:
        raise ValueError("unsupported reflection-representation cache format")
    characters = np.ascontiguousarray(data["characters"], dtype=np.int8)
    phases = np.ascontiguousarray(data["phases"], dtype=np.int8)
    orbit_to_sector = np.ascontiguousarray(
        data["orbit_to_sector"], dtype=np.int64
    )
    order = reduction.group_order
    if characters.shape != (order, order):
        raise ValueError("cached reflection character table has the wrong shape")
    if phases.shape != (order, reduction.full_size):
        raise ValueError("cached representation phases have the wrong shape")
    if orbit_to_sector.shape != (order, reduction.wedge_size):
        raise ValueError("cached sector-orbit maps have the wrong shape")
    if not np.all((characters == -1) | (characters == 1)):
        raise ValueError("cached characters are not +/-1")
    if not np.all((phases == -1) | (phases == 0) | (phases == 1)):
        raise ValueError("cached phases are not -1/0/+1")
    gram = characters.astype(np.int64) @ characters.astype(np.int64).T
    if not np.array_equal(gram, order * np.eye(order, dtype=np.int64)):
        raise ValueError("cached reflection characters are not orthogonal")
    for representation in range(order):
        mapping = orbit_to_sector[representation]
        admitted = mapping >= 0
        if not np.any(admitted):
            raise ValueError("cached representation has zero dimension")
        if not np.array_equal(
            np.sort(mapping[admitted]),
            np.arange(np.count_nonzero(admitted), dtype=np.int64),
        ):
            raise ValueError("cached sector-orbit map is not contiguous")
        if np.any(mapping[~admitted] != -1):
            raise ValueError("cached rejected orbit has an invalid sector index")
        representatives = reduction.representative_rows
        if not np.all(phases[representation, representatives[admitted]] == 1):
            raise ValueError("cached phase convention is invalid")
        # ``full_to_wedge`` is already the direct full-row -> orbit map.
        # Indexing the Boolean admission mask is O(N) and avoids sorting or
        # hashing the rejected orbit indices inside ``np.isin`` for every
        # representation during a cache hit.
        rejected_rows = ~admitted[reduction.full_to_wedge]
        if np.any(phases[representation, rejected_rows] != 0):
            raise ValueError("cached rejected orbit has a nonzero phase")
    return ReflectionRepresentationDecomposition(
        reduction=reduction,
        characters=characters,
        phases=phases,
        orbit_to_sector=orbit_to_sector,
    )


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically publish an uncompressed metadata entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{id(arrays):x}"
    )
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_or_detect_reflection_reduction(
    grid: RealSpaceGrid,
    atoms: Sequence[Atom],
    *,
    cache_directory: os.PathLike[str] | str | None,
    atom_tolerance: float = 1.0e-7,
    lattice_tolerance: float = 1.0e-10,
) -> tuple[AxisReflectionReduction, SymmetryCacheInfo]:
    """Load exact orbit metadata, or run and cache the exact detector."""

    # Preserve the detector's public validation order.  This also lets
    # lightweight backend-wiring objects fall back cleanly before any grid
    # metadata is inspected.
    if not atoms:
        raise ValueError("axis-reflection detection requires at least one atom")
    # Hashing both full-grid coordinate arrays is needed only to address a
    # cache entry.
    key = None
    hash_seconds = 0.0
    if cache_directory is not None:
        hash_started = perf_counter()
        key = _geometry_key(grid, atoms, atom_tolerance, lattice_tolerance)
        hash_seconds = perf_counter() - hash_started
    path = (
        None
        if cache_directory is None
        else Path(cache_directory) / f"reflection-reduction-v{_REDUCTION_FORMAT}-{key}.npz"
    )
    load_seconds = 0.0
    if path is not None and path.is_file():
        started = perf_counter()
        try:
            with np.load(path, allow_pickle=False) as data:
                reduction = _validated_reduction(data, grid.size)
        except (OSError, ValueError, KeyError, EOFError):
            pass
        else:
            load_seconds = perf_counter() - started
            return reduction, SymmetryCacheInfo(
                status="hit",
                key=key,
                path=path,
                hash_seconds=hash_seconds,
                load_seconds=load_seconds,
            )
        load_seconds = perf_counter() - started

    started = perf_counter()
    reduction = SignedPermutationReduction.detect(
        grid,
        atoms,
        atom_tolerance=atom_tolerance,
        lattice_tolerance=lattice_tolerance,
    )
    build_seconds = perf_counter() - started
    write_seconds = 0.0
    status = "disabled-built"
    if path is not None:
        started = perf_counter()
        _write_npz(
            path,
            format=np.asarray(_REDUCTION_FORMAT, dtype=np.int64),
            kind=np.asarray(
                "signed_permutation"
                if isinstance(reduction, SignedPermutationReduction)
                else "axis_reflection"
            ),
            signs=reduction.signs,
            representative_rows=reduction.representative_rows,
            full_to_wedge=reduction.full_to_wedge,
            multiplicities=reduction.multiplicities,
            **(
                {
                    "operations": reduction.operations,
                    "generator_bits": reduction.generator_bits,
                }
                if isinstance(reduction, SignedPermutationReduction)
                else {}
            ),
        )
        write_seconds = perf_counter() - started
        status = "miss-written"
    return reduction, SymmetryCacheInfo(
        status=status,
        key=key,
        path=path,
        hash_seconds=hash_seconds,
        load_seconds=load_seconds,
        build_seconds=build_seconds,
        write_seconds=write_seconds,
    )


def load_or_build_reflection_decomposition(
    grid: RealSpaceGrid,
    reduction: AxisReflectionReduction,
    *,
    reduction_key: str | None,
    cache_directory: os.PathLike[str] | str | None,
) -> tuple[ReflectionRepresentationDecomposition, SymmetryCacheInfo]:
    """Load or build the exact character phases for a free grid action.

    ``reduction_key`` is the content key of ``reduction``.  It may be ``None``
    only together with a disabled cache, where no key is derived from it.
    """

    key = None
    hash_seconds = 0.0
    if cache_directory is not None:
        if reduction_key is None:
            raise ValueError(
                "a cached reflection decomposition requires the content key "
                "of its reduction"
            )
        hash_started = perf_counter()
        key = _representation_key(reduction_key, reduction)
        hash_seconds = perf_counter() - hash_started
    path = (
        None
        if cache_directory is None
        else Path(cache_directory)
        / f"reflection-representations-v{_REPRESENTATION_FORMAT}-{key}.npz"
    )
    load_seconds = 0.0
    if path is not None and path.is_file():
        started = perf_counter()
        try:
            with np.load(path, allow_pickle=False) as data:
                decomposition = _validated_decomposition(data, reduction)
        except (OSError, ValueError, KeyError, EOFError):
            pass
        else:
            load_seconds = perf_counter() - started
            return decomposition, SymmetryCacheInfo(
                status="hit",
                key=key,
                path=path,
                hash_seconds=hash_seconds,
                load_seconds=load_seconds,
            )
        load_seconds = perf_counter() - started

    started = perf_counter()
    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
    build_seconds = perf_counter() - started
    write_seconds = 0.0
    status = "disabled-built"
    if path is not None:
        started = perf_counter()
        _write_npz(
            path,
            format=np.asarray(_REPRESENTATION_FORMAT, dtype=np.int64),
            characters=decomposition.characters,
            phases=decomposition.phases,
            orbit_to_sector=decomposition.orbit_to_sector,
        )
        write_seconds = perf_counter() - started
        status = "miss-written"
    return decomposition, SymmetryCacheInfo(
        status=status,
        key=key,
        path=path,
        hash_seconds=hash_seconds,
        load_seconds=load_seconds,
        build_seconds=build_seconds,
        write_seconds=write_seconds,
    )


__all__ = [
    "SymmetryCacheInfo",
    "load_or_build_reflection_decomposition",
    "load_or_detect_reflection_reduction",
]
