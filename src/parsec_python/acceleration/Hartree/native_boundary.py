"""Cached C++/OpenMP construction of PARSEC's isolated Hartree boundary.

The static grid geometry and the missing finite-difference stencil neighbors
are prepared once.  :meth:`NativeMultipoleBoundaryBuilder.build` then accepts
one SCF density and returns the same normalized multipole object and

``b_eff = 8*pi*rho_I - A_IB V_B``

as the readable Python implementation, while doing the all-grid and exterior
loops in float64 C++/OpenMP.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from parsec_python.Grid import RealSpaceGrid
from parsec_python.models import MAXIMUM_MULTIPOLE_ORDER

from ..backends.native import _load_native
from ..SCF.symmetry_fields import SymmetryScalarField
from ..Symmetry import AxisReflectionReduction
from .fast_multipole import FastMultipoleExpansion


_SYMMETRY_BOUNDARY_CACHE_FORMAT = 2


@dataclass(frozen=True)
class NativeBoundaryCacheInfo:
    """Persistent native Hartree-geometry cache provenance and timings.

    ``key`` is ``None`` when the cache is disabled and no key was hashed.
    """

    status: str
    key: str | None
    path: Path | None
    hash_seconds: float = 0.0
    load_seconds: float = 0.0
    build_seconds: float = 0.0
    write_seconds: float = 0.0


def _checked_order(multipole_order) -> int:
    order = int(multipole_order)
    if order != multipole_order or not 0 <= order <= MAXIMUM_MULTIPOLE_ORDER:
        raise ValueError(
            f"multipole_order must be between 0 and {MAXIMUM_MULTIPOLE_ORDER}"
        )
    return order


def _require_native_order(native, order: int) -> None:
    """Refuse an order the loaded extension stores no arrays for.

    Extensions before 0.6.0 hold the angular arrays of order 9 and report no
    maximum.
    """

    if order <= 9:
        return
    supported = 9
    build_info = getattr(native, "build_info", None)
    if build_info is not None:
        supported = int(dict(build_info()).get("maximum_multipole_order", 9))
    if order > supported:
        raise RuntimeError(
            f"multipole order {order} of the Hartree boundary needs the native "
            f"extension 0.6.0 or newer; the loaded one stops at {supported}. "
            "Rebuild it from src/parsec_python/acceleration/native, or keep "
            "Solver_Lpole at 9 or below with Hartree_Boundary_Tolerance: off"
        )


def _hash_array(digest, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))


def _symmetry_boundary_key(
    grid: RealSpaceGrid,
    reduction: AxisReflectionReduction,
    order: int,
    seed: str | None,
) -> str:
    """Hash every static value that can change the cached native buffers."""

    digest = sha256()
    digest.update(
        f"native-symmetry-boundary-v{_SYMMETRY_BOUNDARY_CACHE_FORMAT}".encode()
    )
    digest.update(np.int64(order).tobytes())
    digest.update(np.int64(grid.settings.expansion_order).tobytes())
    digest.update(np.float64(grid.spacing).tobytes())
    _hash_array(digest, "shift", np.asarray(grid.settings.shift, dtype=np.float64))
    if seed is None:
        _hash_array(digest, "integer_coordinates", grid.integer_coordinates)
        _hash_array(digest, "physical_coordinates", grid.coordinates)
        _hash_array(digest, "representatives", reduction.representative_rows)
        _hash_array(digest, "mapping", reduction.full_to_wedge)
        _hash_array(digest, "multiplicities", reduction.multiplicities)
    else:
        # The geometry cache key already hashes the exact grid, labeled atoms,
        # detector tolerances, and resulting reduction metadata.
        digest.update(seed.encode("ascii"))
    return digest.hexdigest()


def _moment_cache_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}-moments.npy")


def _write_cache(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish metadata plus a memory-mappable moment table.

    ``moment_coefficients`` dominates this cache (about 55 MiB for the
    naphthalene benchmark).  Keeping it inside NPZ forces NumPy to allocate
    and fill one array before pybind copies it into native-owned storage.
    A standalone NPY sidecar is memory mapped, so the native constructor is
    the only full copy.  The metadata archive is replaced last and therefore
    acts as the atomic commit marker for both files.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(payload)
    moments = np.ascontiguousarray(
        arrays.pop("moment_coefficients"), dtype=np.complex128
    )
    moment_path = _moment_cache_path(path)
    token = f"{os.getpid()}-{id(payload):x}"
    temporary = path.with_name(f".{path.name}.tmp-{token}")
    temporary_moments = moment_path.with_name(
        f".{moment_path.name}.tmp-{token}"
    )
    try:
        with temporary_moments.open("wb") as stream:
            np.save(stream, moments, allow_pickle=False)
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary_moments, moment_path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
        if temporary_moments.exists():
            temporary_moments.unlink()


class NativeMultipoleBoundaryBuilder:
    """Reusable native multipole/RHS builder for one spherical cluster grid."""

    def __init__(self, grid: RealSpaceGrid, multipole_order: int = 9) -> None:
        if grid.settings.domain_shape != "sphere":
            raise ValueError(
                "the origin-centered native multipole boundary requires a "
                "spherical grid"
            )
        order = _checked_order(multipole_order)

        native = _load_native()
        _require_native_order(native, order)
        self.grid = grid
        self.order = order
        self._native_builder = native.MultipoleBoundaryBuilder(
            np.ascontiguousarray(grid.integer_coordinates, dtype=np.int64),
            np.ascontiguousarray(grid.coordinates, dtype=np.float64),
            np.ascontiguousarray(grid.index_min, dtype=np.int64),
            np.ascontiguousarray(grid.lookup, dtype=np.int64),
            np.ascontiguousarray(grid.settings.shift, dtype=np.float64),
            int(grid.settings.expansion_order),
            float(grid.spacing),
            order,
        )

    # Static atomic tail added to every right-hand side, ``None`` for the
    # multipole expansion alone.
    boundary_tail = None

    def set_boundary_tail(self, tail) -> None:
        """Add the rows of ``tail`` to every later right-hand side.

        ``tail`` is an :class:`parsec_python.Hartree.AtomicTail` built for
        the order of this builder; ``None`` removes it.
        """
        if tail is not None and tail.order != self.order:
            raise ValueError(
                f"the atomic tail was built for multipole order {tail.order}, "
                f"not {self.order}"
            )
        self.boundary_tail = tail

    @property
    def boundary_term_count(self) -> int:
        """Number of missing axial stencil entries cached by the builder."""
        return int(self._native_builder.boundary_term_count)

    def build(
        self,
        density: np.ndarray,
    ) -> tuple[np.ndarray, FastMultipoleExpansion]:
        """Build positive/negative moments and the boundary-corrected RHS."""
        density = np.ascontiguousarray(density, dtype=np.float64)
        if density.shape != (self.grid.size,):
            raise ValueError("density does not match the active grid")
        payload = self._native_builder.build(density)
        positive = np.asarray(payload["positive_m_moments"], dtype=np.complex128)

        moments: dict[tuple[int, int], complex] = {}
        for angular_momentum in range(self.order + 1):
            for magnetic in range(angular_momentum + 1):
                moment = complex(positive[angular_momentum, magnetic])
                moments[(angular_momentum, magnetic)] = moment
                if magnetic:
                    moments[(angular_momentum, -magnetic)] = (
                        ((-1) ** magnetic) * np.conjugate(moment)
                    )
        boundary = FastMultipoleExpansion(order=self.order, moments=moments)
        right_hand_side = np.asarray(
            payload["right_hand_side"], dtype=np.float64
        )
        if self.boundary_tail is not None:
            right_hand_side = np.require(right_hand_side, requirements="W")
            right_hand_side[self.boundary_tail.rows] += self.boundary_tail.values
        return right_hand_side, boundary


class NativeSymmetryMultipoleBoundaryBuilder(NativeMultipoleBoundaryBuilder):
    """Native multipoles and normalized RHS built directly on exact orbits."""

    def __init__(
        self,
        grid: RealSpaceGrid,
        reduction: AxisReflectionReduction,
        multipole_order: int = 9,
        *,
        cache_directory: os.PathLike[str] | str | None = None,
        cache_key_seed: str | None = None,
    ) -> None:
        if reduction.full_size != grid.size:
            raise ValueError("symmetry reduction does not match the Hartree grid")
        order = _checked_order(multipole_order)
        # The key only addresses the cache entry.  Without a seed it hashes
        # both full-grid coordinate arrays, so skip it when nothing reads it.
        key = None
        hash_seconds = 0.0
        if cache_directory is not None:
            hash_started = perf_counter()
            key = _symmetry_boundary_key(grid, reduction, order, cache_key_seed)
            hash_seconds = perf_counter() - hash_started
        path = (
            None
            if cache_directory is None
            else Path(cache_directory)
            / f"native-symmetry-boundary-v{_SYMMETRY_BOUNDARY_CACHE_FORMAT}-{key}.npz"
        )
        moment_path = None if path is None else _moment_cache_path(path)
        native = _load_native()
        _require_native_order(native, order)
        self.grid = grid
        self.order = order
        self.reduction = reduction
        load_seconds = 0.0
        if (
            path is not None
            and path.is_file()
            and moment_path is not None
            and moment_path.is_file()
        ):
            started = perf_counter()
            try:
                with np.load(path, allow_pickle=False) as data:
                    if int(np.asarray(data["format"]).reshape(())) != (
                        _SYMMETRY_BOUNDARY_CACHE_FORMAT
                    ):
                        raise ValueError("unsupported native boundary cache format")
                    if int(np.asarray(data["multipole_order"]).reshape(())) != order:
                        raise ValueError("cached multipole order does not match")
                    volume = float(np.asarray(data["volume_element"]).reshape(()))
                    if volume != float(grid.volume_element):
                        raise ValueError("cached grid volume element does not match")
                    multiplicities = np.ascontiguousarray(
                        data["multiplicities"], dtype=np.int64
                    )
                    if not np.array_equal(
                        multiplicities, reduction.multiplicities
                    ):
                        raise ValueError("cached orbit multiplicities do not match")
                    moment_coefficients = np.load(
                        moment_path,
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    expected_angular = (order + 1) * (order + 2) // 2
                    if (
                        moment_coefficients.dtype != np.dtype(np.complex128)
                        or moment_coefficients.shape
                        != (multiplicities.size, expected_angular)
                        or not moment_coefficients.flags.c_contiguous
                    ):
                        raise ValueError(
                            "cached multipole moment table has the wrong layout"
                        )
                    self._native_builder = native.MultipoleBoundaryBuilder(
                        order,
                        volume,
                        multiplicities,
                        moment_coefficients,
                        np.ascontiguousarray(data["boundary_indptr"], dtype=np.int64),
                        np.ascontiguousarray(
                            data["boundary_operator_coefficient"], dtype=np.float64
                        ),
                        np.ascontiguousarray(data["boundary_radius"], dtype=np.float64),
                        np.ascontiguousarray(data["boundary_cosine"], dtype=np.float64),
                        np.ascontiguousarray(data["boundary_sine"], dtype=np.float64),
                        np.ascontiguousarray(
                            data["boundary_phase_real"], dtype=np.float64
                        ),
                        np.ascontiguousarray(
                            data["boundary_phase_imag"], dtype=np.float64
                        ),
                    )
            except (OSError, ValueError, KeyError, EOFError, RuntimeError, TypeError):
                pass
            else:
                load_seconds = perf_counter() - started
                self.cache_info = NativeBoundaryCacheInfo(
                    status="hit",
                    key=key,
                    path=path,
                    hash_seconds=hash_seconds,
                    load_seconds=load_seconds,
                )
                return
            load_seconds = perf_counter() - started

        started = perf_counter()
        super().__init__(grid, order)
        if not hasattr(self._native_builder, "configure_symmetry"):
            raise RuntimeError(
                "the installed native extension does not provide the "
                "symmetry multipole kernel; install version 0.4.0 or newer"
            )
        self._native_builder.configure_symmetry(
            np.ascontiguousarray(reduction.representative_rows, dtype=np.int64),
            np.ascontiguousarray(reduction.full_to_wedge, dtype=np.int64),
            np.ascontiguousarray(reduction.multiplicities, dtype=np.int64),
        )
        build_seconds = perf_counter() - started
        write_seconds = 0.0
        status = "disabled-built"
        if path is not None and hasattr(
            self._native_builder, "export_symmetry_cache"
        ):
            started = perf_counter()
            payload = dict(self._native_builder.export_symmetry_cache())
            payload["format"] = np.asarray(
                _SYMMETRY_BOUNDARY_CACHE_FORMAT, dtype=np.int64
            )
            _write_cache(path, payload)
            write_seconds = perf_counter() - started
            status = "miss-written"
        self.cache_info = NativeBoundaryCacheInfo(
            status=status,
            key=key,
            path=path,
            hash_seconds=hash_seconds,
            load_seconds=load_seconds,
            build_seconds=build_seconds,
            write_seconds=write_seconds,
        )

    _reduced_tail = None

    def set_boundary_tail(self, tail) -> None:
        """Add ``U.T`` of the rows of ``tail`` to every later right-hand side."""
        super().set_boundary_tail(tail)
        self._reduced_tail = None
        if tail is not None:
            reduced = self.reduction.reduce_vector(tail.full(self.grid.size))
            rows = np.flatnonzero(reduced)
            self._reduced_tail = (rows, reduced[rows])

    def build_reduced(
        self,
        density: np.ndarray,
    ) -> tuple[np.ndarray, FastMultipoleExpansion]:
        """Return ``U.T b_eff`` without constructing a full-grid RHS."""

        if isinstance(density, SymmetryScalarField):
            if density.reduction is self.reduction:
                wedge_density = density.values
            else:
                full_density = np.ascontiguousarray(
                    density.values[density.reduction.full_to_wedge]
                )
                wedge_density = self.reduction.reduce_vector(
                    full_density
                ) / np.sqrt(self.reduction.multiplicities)
        else:
            density = np.asarray(density, dtype=np.float64)
            if density.shape != (self.grid.size,):
                raise ValueError("density does not match the active grid")
            # reduce_vector/sqrt(m) is the physical orbit average.  It removes
            # only roundoff-level symmetry noise before using the exact
            # invariant multipole formula.
            wedge_density = self.reduction.reduce_vector(density) / np.sqrt(
                self.reduction.multiplicities
            )
        payload = self._native_builder.build_reduced(
            np.ascontiguousarray(wedge_density, dtype=np.float64)
        )
        positive = np.asarray(payload["positive_m_moments"], dtype=np.complex128)
        moments: dict[tuple[int, int], complex] = {}
        for angular_momentum in range(self.order + 1):
            for magnetic in range(angular_momentum + 1):
                moment = complex(positive[angular_momentum, magnetic])
                moments[(angular_momentum, magnetic)] = moment
                if magnetic:
                    moments[(angular_momentum, -magnetic)] = (
                        ((-1) ** magnetic) * np.conjugate(moment)
                    )
        right_hand_side = np.asarray(payload["right_hand_side"], dtype=np.float64)
        if self._reduced_tail is not None:
            rows, values = self._reduced_tail
            right_hand_side = np.require(right_hand_side, requirements="W")
            right_hand_side[rows] += values
        return (
            right_hand_side,
            FastMultipoleExpansion(order=self.order, moments=moments),
        )


__all__ = [
    "NativeBoundaryCacheInfo",
    "NativeMultipoleBoundaryBuilder",
    "NativeSymmetryMultipoleBoundaryBuilder",
]
