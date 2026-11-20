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


_MATRIX_FIELDS = frozenset({"data", "indices", "indptr", "has_sorted_indices", "has_canonical_format", "maxprint"})
"""Instance fields of a CSR matrix (not attributes of its class) that the deferred descriptor hands to the matrix."""


class DeferredPeriodicNegativeLaplacian(DeferredNativeNegativeLaplacian):
    """The wrapped ``-nabla_FD^2`` of a periodic box, built when something reads it.

    The matrix of :func:`parsec_python.Laplacian.finite_difference.build_negative_laplacian` for a grid whose every
    axis holds more than ``2 M`` points: ``6 M + 1`` entries per row, the stencil wrapped on every axis.  The sector
    stencils of a point-group reduction come from the grid
    (:func:`parsec_python.acceleration.Symmetry.operator_cache._wrapped_stencil_grid`), so a Gamma-point cell solved in
    its sectors never forms the matrix (5.8 s and 14 GB of host memory per rank for the 46.7 M points of a 20^3 diamond
    cell); every other reader gets it from :meth:`materialize` (``operator @ vectors`` and the attributes of a sparse
    matrix materialize too), the same matrix bit for bit.
    """

    def __init__(self, grid, *, sharing_matrix_with: "DeferredPeriodicNegativeLaplacian | None" = None) -> None:
        if sharing_matrix_with is not None and sharing_matrix_with.grid is not grid:
            raise ValueError("descriptors sharing a matrix must have the same grid")
        self.grid = grid
        size = int(grid.size)
        self.shape = (size, size)
        self._cache_key = None
        self.hash_seconds = 0.0
        self.nnz = size * (6 * int(grid.settings.stencil_half_width) + 1)
        self.nnz_cache_status = "stencil"
        self.nnz_cache_path = None
        self.nnz_count_seconds = 0.0
        self.materialization_seconds = 0.0
        self._memo: list[sp.csr_matrix | None] = (
            [None] if sharing_matrix_with is None else sharing_matrix_with._memo
        )
        # One build for every view and thread that asks for the matrix at once (the device threads of a k-point or
        # band-structure pass): the lock goes with the memo.
        import threading

        self._build_lock = threading.Lock() if sharing_matrix_with is None else sharing_matrix_with._build_lock
        self.matrix_origin: str | None = None

    @property
    def parsec_wrapped_stencil_of(self):
        """The grid the matrix is the wrapped stencil of (as the built matrix names it)."""

        return self.grid

    def _hashed_key(self) -> str:
        # the inputs of the wrapped builder, under a name of its own (the Dirichlet stencil of a cluster on the same
        # integer points is another operator)
        started = perf_counter()
        digest = sha256()
        digest.update(f"periodic-wrapped-negative-laplacian-v{_KEY_FORMAT}".encode("ascii"))
        digest.update(np.int64(self.grid.settings.expansion_order).tobytes())
        digest.update(np.float64(self.grid.spacing).tobytes())
        _hash_array(digest, "points_per_axis", np.asarray(self.grid.points_per_axis, dtype=np.int64))
        if _unequal_step_box(self.grid):
            # a step per axis (a cell of one spacing keeps the key of before)
            _hash_array(digest, "axis_steps", np.diag(np.asarray(self.grid.step_vectors, dtype=np.float64)))
        self.hash_seconds += perf_counter() - started
        return digest.hexdigest()

    def materialize(self) -> sp.csr_matrix:
        matrix = self._memo[0]
        if matrix is None:
            with self._build_lock:
                matrix = self._memo[0]
                if matrix is None:
                    from parsec_python.Laplacian.finite_difference import build_negative_laplacian

                    started = perf_counter()
                    matrix = build_negative_laplacian(self.grid)
                    self.materialization_seconds += perf_counter() - started
                    if matrix.shape != self.shape or int(matrix.nnz) != self.nnz:
                        raise RuntimeError(
                            "the periodic finite-difference operator does not match its deferred descriptor"
                        )
                    self._memo[0] = matrix
                    self.matrix_origin = "built"
                    return matrix
        if self.matrix_origin is None:
            self.matrix_origin = "shared"
        return matrix

    def getrow(self, index: int) -> sp.csr_matrix:
        """Row ``index`` of the matrix, from the stencil without the matrix: the same entries as the built matrix (no
        two neighbours of a point coincide on such a grid; the symbol of the FFT Hartree solve reads row 0)."""

        from parsec_python.Laplacian.finite_difference import second_derivative_coefficients

        shape = tuple(int(value) for value in self.grid.points_per_axis)
        strides = (shape[1] * shape[2], shape[2], 1)
        width = int(self.grid.settings.stencil_half_width)
        point = (index // strides[0], (index // strides[1]) % shape[1], index % shape[2])
        if _unequal_step_box(self.grid):
            # a step per axis: the values of the builder of such a cell (unequal_step_values)
            diagonal, shells = unequal_step_values(self.grid)
            entries = {int(index): diagonal}
        else:
            coeff = second_derivative_coefficients(self.grid.settings.expansion_order)
            inv_h2 = 1.0 / float(self.grid.spacing) ** 2
            entries = {int(index): -3.0 * coeff[width] * inv_h2}
            shells = (tuple(-coeff[width + shell] * inv_h2 for shell in range(1, width + 1)),) * 3
        for axis in range(3):
            for shift in range(-width, width + 1):
                if shift:
                    moved = list(point)
                    moved[axis] = (moved[axis] + shift) % shape[axis]
                    column = moved[0] * strides[0] + moved[1] * strides[1] + moved[2]
                    entries[column] = shells[axis][abs(shift) - 1]
        columns = np.array(sorted(entries), dtype=np.int64)
        values = np.array([entries[column] for column in columns], dtype=np.float64)
        return sp.csr_matrix((values, columns, np.array([0, columns.size])), shape=(1, self.shape[1]))

    def __getattr__(self, name: str):
        # a reader that takes it for the sparse matrix (dot, diagonal, data, T, ...): the matrix; any other name (an
        # attribute a report asks for with a default) is not one of it
        if name.startswith("_") or not (hasattr(sp.csr_matrix, name) or name in _MATRIX_FIELDS):
            raise AttributeError(name)
        return getattr(self.materialize(), name)

    def __repr__(self) -> str:
        state = "materialized" if self.materialized else "deferred"
        return f"DeferredPeriodicNegativeLaplacian(shape={self.shape}, nnz={self.nnz}, state={state!r})"


PERIODIC_DEFER_ENV = "PARSEC_PERIODIC_DEFER_LAPLACIAN"


def periodic_laplacian_builder():
    """``PARSEC_PERIODIC_DEFER_LAPLACIAN``: ``auto`` (default, since 2026-11-18) gives the accelerated preparation of a
    periodic box :func:`periodic_negative_laplacian` (the matrix built when something reads it); ``off``: ``None``,
    the matrix built at once, as before."""

    value = os.environ.get(PERIODIC_DEFER_ENV, "auto").strip().lower() or "auto"
    if value not in ("auto", "off"):
        raise ValueError(f"{PERIODIC_DEFER_ENV} must be auto or off, got {value!r}")
    return periodic_negative_laplacian if value == "auto" else None


def _unequal_step_box(grid) -> bool:
    """Whether ``grid`` is a bulk box of the default routines with its sides along the axes whose steps differ, every
    point a row in row-major order and each axis longer than ``2 M`` (``_wrapped_box`` of
    :mod:`parsec_python.Laplacian.finite_difference` but for the one spacing), under
    ``PARSEC_PERIODIC_UNEQUAL_STEPS=auto``: its matrix is the wrapped axial stencil with a step per axis (the builder
    of oblique cells, without mixed terms), ``6 M + 1`` entries a row."""

    from parsec_python.acceleration.Symmetry.axis_reflection import periodic_unequal_steps_setting, unequal_step_grid

    if not unequal_step_grid(grid) or periodic_unequal_steps_setting() != "auto":
        return False
    from parsec_python.Laplacian.lattice_metric import second_derivative_plan

    if second_derivative_plan(grid.step_vectors).directions:
        return False
    shape = tuple(int(value) for value in grid.points_per_axis)
    lookup = np.asarray(grid.lookup)
    width = int(grid.settings.stencil_half_width)
    return (
        lookup.shape == shape
        and int(grid.size) == int(np.prod(shape))
        and min(shape) > 2 * width
        and int(lookup.reshape(-1)[0]) == 0
        and int(lookup.reshape(-1)[-1]) == int(grid.size) - 1
    )


def unequal_step_values(grid) -> tuple[float, tuple[tuple[float, ...], ...]]:
    """``(diagonal, shells)`` of ``-nabla_FD^2`` on a box of a step per axis, bit for bit the entries of its built
    matrix (``_build_oblique_negative_laplacian`` of :mod:`parsec_python.Laplacian.finite_difference`):
    ``-sum_a p_a c_0`` and, per axis, ``-p_a c_j`` for ``j = 1 .. M``, with ``p_a`` (``1/h_a^2``) the axial
    coefficients of :func:`parsec_python.Laplacian.lattice_metric.second_derivative_plan`."""

    from parsec_python.Laplacian.finite_difference import second_derivative_coefficients
    from parsec_python.Laplacian.lattice_metric import second_derivative_plan

    plan = second_derivative_plan(grid.step_vectors)
    if plan.directions:
        raise ValueError("the cell has mixed derivatives: no axial stencil")
    coeff = second_derivative_coefficients(grid.settings.expansion_order)
    width = int(grid.settings.stencil_half_width)
    total = float(np.sum(plan.axial)) + float(np.sum(plan.weights))
    diagonal = float(-total * coeff[width])
    shells = tuple(
        tuple(float(-plan.axial[axis] * coeff[width + shell]) for shell in range(1, width + 1)) for axis in range(3)
    )
    return diagonal, shells


def periodic_negative_laplacian(grid):
    """The deferred wrapped Laplacian of a periodic box (:class:`DeferredPeriodicNegativeLaplacian`), or the built
    matrix of :func:`parsec_python.Laplacian.finite_difference.build_negative_laplacian` for every other grid (oblique
    or general cells, an axis of ``2 M`` points or fewer, whose stencil folds onto itself).

    A box whose steps differ between the axes (:func:`_unequal_step_box`, ``PARSEC_PERIODIC_UNEQUAL_STEPS=auto``) is
    deferred too; ``off`` builds its matrix at once, as before."""

    from parsec_python.Laplacian.finite_difference import _wrapped_box, build_negative_laplacian

    # _wrapped_box: an orthorhombic bulk box of every point in row-major order, each axis longer than 2 M
    if not getattr(grid, "is_general_periodic", False) and (_wrapped_box(grid) or _unequal_step_box(grid)):
        return DeferredPeriodicNegativeLaplacian(grid)
    return build_negative_laplacian(grid)


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
    "DeferredPeriodicNegativeLaplacian",
    "PERIODIC_DEFER_ENV",
    "materialize_negative_laplacian",
    "periodic_laplacian_builder",
    "periodic_negative_laplacian",
    "unequal_step_values",
]
