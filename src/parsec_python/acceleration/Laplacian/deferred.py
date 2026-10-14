"""Defer the full-grid finite-difference CSR until something reads it.

The finite-difference operator is determined completely by the active integer
grid, its lookup table, the stencil order, and the grid spacing.  The symmetry
sector route builds its packed stencils from those inputs directly, and a
SHA-256 fingerprint of them validates a cached reduced operator, so neither
needs the much larger full-grid CSR matrix.

This object is deliberately narrow: it exposes the grid, ``shape`` and ``nnz``
for setup reporting, an exact ``cache_key`` for downstream content-addressed
caches, and :meth:`materialize` for every consumer of the full-grid matrix;
``operator @ vectors`` materializes too.  Materialization still calls the
validated C++ builder and checks its shape and nonzero count, so deferral
changes setup order rather than the operator.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid import RealSpaceGrid


_KEY_FORMAT = 1
_NNZ_CACHE_FORMAT = 1
_MEMORY_NNZ_CACHE: dict[tuple[str, str], int] = {}


def _hash_array(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))


def _operator_key(grid: RealSpaceGrid) -> str:
    """Hash every discrete input used by the native stencil builder."""

    digest = sha256()
    digest.update(f"native-negative-laplacian-v{_KEY_FORMAT}".encode("ascii"))
    digest.update(np.int64(grid.settings.expansion_order).tobytes())
    digest.update(np.float64(grid.spacing).tobytes())
    _hash_array(digest, "integer_coordinates", grid.integer_coordinates)
    _hash_array(digest, "index_min", grid.index_min)
    _hash_array(digest, "lookup", grid.lookup)
    return digest.hexdigest()


def _operator_nnz(grid: RealSpaceGrid) -> int:
    """Count the exact centered and in-domain axial stencil entries.

    A row has its centered entry and one entry for every stencil neighbor
    inside the domain.  Two active points ``j`` steps apart along an axis are
    neighbors of each other, so the count is the active points plus twice the
    active pairs of every axis and shell.  The pairs are read off the Boolean
    mask of the lookup table, which is far cheaper than gathering a neighbor
    row for every grid point and shell.  The total is checked again if the
    matrix is eventually materialized.
    """

    active = np.asarray(grid.lookup) >= 0
    width = int(grid.settings.expansion_order) // 2
    pairs = 0
    for axis in range(3):
        along = np.moveaxis(active, axis, 0)
        for shell in range(1, min(width, along.shape[0] - 1) + 1):
            pairs += int(np.count_nonzero(along[shell:] & along[:-shell]))
    return int(np.count_nonzero(active)) + 2 * pairs


def _nnz_cache_path(
    cache_directory: os.PathLike[str] | str | None,
    cache_key: str | None,
) -> Path | None:
    if cache_directory is None:
        return None
    return (
        Path(cache_directory)
        / f"negative-laplacian-nnz-v{_NNZ_CACHE_FORMAT}-{cache_key}.json"
    )


def _validated_cached_nnz(
    payload: Any,
    cache_key: str,
    shape: tuple[int, int],
    maximum_nnz: int,
) -> int:
    """Validate a tiny reporting cache before accepting its integer value."""

    if not isinstance(payload, dict):
        raise ValueError("NNZ cache payload is not an object")
    if int(payload.get("format", -1)) != _NNZ_CACHE_FORMAT:
        raise ValueError("NNZ cache format does not match")
    if payload.get("cache_key") != cache_key:
        raise ValueError("NNZ cache key does not match")
    if payload.get("shape") != list(shape):
        raise ValueError("NNZ cache shape does not match")
    value = payload.get("nnz")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("NNZ cache value is not an integer")
    if not shape[0] <= value <= maximum_nnz:
        raise ValueError("NNZ cache value lies outside stencil bounds")
    return value


def _load_or_count_nnz(
    grid: RealSpaceGrid,
    cache_key: str | None,
    shape: tuple[int, int],
    cache_directory: os.PathLike[str] | str | None,
) -> tuple[int, str, Path | None]:
    """Resolve exact NNZ metadata without rescanning a known full grid.

    ``cache_key`` may be ``None`` only without a cache directory.
    """

    path = _nnz_cache_path(cache_directory, cache_key)
    if path is None:
        return _operator_nnz(grid), "disabled", None

    memory_key = (str(path.resolve()), cache_key)
    remembered = _MEMORY_NNZ_CACHE.get(memory_key)
    if remembered is not None:
        return remembered, "memory-hit", path

    maximum_nnz = shape[0] * (1 + 3 * int(grid.settings.expansion_order))
    invalid_cache = False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = _validated_cached_nnz(
            payload,
            cache_key,
            shape,
            maximum_nnz,
        )
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        invalid_cache = True
    else:
        _MEMORY_NNZ_CACHE[memory_key] = value
        return value, "disk-hit", path

    value = _operator_nnz(grid)
    payload = {
        "format": _NNZ_CACHE_FORMAT,
        "cache_key": cache_key,
        "shape": list(shape),
        "nnz": value,
    }
    status = "invalid-rebuilt" if invalid_cache else "miss-written"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        status = "miss-unwritten"
    _MEMORY_NNZ_CACHE[memory_key] = value
    return value, status, path


class DeferredNativeNegativeLaplacian:
    """Exact lazy proxy for one native full-grid ``-nabla_FD^2`` matrix.

    ``sharing_matrix_with`` is an earlier descriptor of the same grid.  The
    two then hold one memoized matrix, whichever of them builds it.  A
    resident process gives every calculation a descriptor of its own for the
    report, and the matrix must not be built again for each of them.
    """

    def __init__(
        self,
        grid: RealSpaceGrid,
        *,
        cache_directory: os.PathLike[str] | str | None = None,
        sharing_matrix_with: DeferredNativeNegativeLaplacian | None = None,
    ) -> None:
        if sharing_matrix_with is not None and sharing_matrix_with.grid is not grid:
            raise ValueError("descriptors sharing a matrix must have the same grid")
        self.grid = grid
        self.shape = (int(grid.size), int(grid.size))
        self._cache_key: str | None = None
        self.hash_seconds = 0.0
        # Only a cache looks the key up.  Without a cache directory it is
        # hashed when something asks for it, which a calculation that builds
        # its sector stencils from the grid never does.
        if cache_directory is not None:
            self._cache_key = self._hashed_key()
        started = perf_counter()
        self.nnz, self.nnz_cache_status, self.nnz_cache_path = (
            _load_or_count_nnz(
                grid,
                self._cache_key,
                self.shape,
                cache_directory,
            )
        )
        self.nnz_count_seconds = perf_counter() - started
        self.materialization_seconds = 0.0
        # One cell for every descriptor that shares the matrix.
        self._memo: list[sp.csr_matrix | None] = (
            [None] if sharing_matrix_with is None else sharing_matrix_with._memo
        )
        # ``built`` once this descriptor has run the builder, ``shared`` once
        # it has handed out a matrix another descriptor built.
        self.matrix_origin: str | None = None

    def _hashed_key(self) -> str:
        started = perf_counter()
        key = _operator_key(self.grid)
        self.hash_seconds += perf_counter() - started
        return key

    @property
    def cache_key(self) -> str:
        """Exact content key of the operator, hashed on first request."""

        if self._cache_key is None:
            self._cache_key = self._hashed_key()
        return self._cache_key

    @property
    def hashed_cache_key(self) -> str | None:
        """The content key if something has asked for it, else ``None``."""

        return self._cache_key

    @property
    def materialized(self) -> bool:
        """Whether the matrix is in memory, built here or shared."""

        return self._memo[0] is not None

    def materialize(self) -> sp.csr_matrix:
        """Build and memoize the validated native CSR on first demand."""

        matrix = self._memo[0]
        if matrix is None:
            # Local import avoids importing the optional extension merely to
            # construct an exact cache key.
            from ..backends.native import build_native_negative_laplacian

            started = perf_counter()
            matrix = build_native_negative_laplacian(self.grid)
            self.materialization_seconds += perf_counter() - started
            if matrix.shape != self.shape or int(matrix.nnz) != self.nnz:
                raise RuntimeError(
                    "materialized finite-difference operator does not match "
                    "its exact deferred descriptor"
                )
            self._memo[0] = matrix
            self.matrix_origin = "built"
        elif self.matrix_origin is None:
            self.matrix_origin = "shared"
        return matrix

    def __matmul__(self, vectors: Any) -> Any:
        """Apply the operator, building its matrix on first use.

        The reference Hamiltonian and Poisson solve of a prepared system
        apply their kinetic operator this way.  The accelerated SCF calls
        neither; a diagnostic that does gets the full-grid action.
        """

        return self.materialize() @ vectors

    def __repr__(self) -> str:
        state = "materialized" if self.materialized else "deferred"
        return (
            f"DeferredNativeNegativeLaplacian(shape={self.shape}, "
            f"nnz={self.nnz}, state={state!r})"
        )


def materialize_negative_laplacian(operator: Any) -> sp.csr_matrix:
    """Return a canonical CSR from either a matrix or the lazy proxy."""

    if isinstance(operator, DeferredNativeNegativeLaplacian):
        return operator.materialize()
    matrix = sp.csr_matrix(operator, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


__all__ = [
    "DeferredNativeNegativeLaplacian",
    "materialize_negative_laplacian",
]
