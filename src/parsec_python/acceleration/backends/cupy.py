"""Optional CuPy Hamiltonian backend for the accelerated architecture.

The module is intentionally importable on machines that do not have CuPy.
CuPy and ``cupyx.scipy.sparse`` are imported only when a GPU object is
constructed or :func:`require_cupy` is called.

The production action keeps every large operand on the device and evaluates

``H X = A X + V[:, None] X + B diag(signs) (B.T X)``.

``B`` and the explicitly cached transpose ``B.T`` are float64 sparse matrices.
``A`` retains a generic CSR allocation for compatibility and, when possible,
uses coalesced stencil-major metadata for production applications.  No
synchronization is performed between the kinetic, local, and nonlocal terms;
synchronization is reserved for coarse timing boundaries.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, fields
import os
import sys
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from .cupy_compact import CuPyCompactFiniteDifference
from .cupy_stencil_major import (
    CuPyStencilMajorFiniteDifference,
    StencilMajorHostMetadata,
)
from .cupy_mixed_precision import (
    CuPyMixedPrecisionRecurrence,
    float32_filter_requested,
    mixed_filter_policy,
)
from .cupy_projectors import (
    CuPySparseProjectorFactors,
    CuPySparseProjectorProjection,
)
from .implicit_stencil import (
    PackedTileHostMetadata,
    implicit_tile_setting,
    pack_worker_setting,
)


class CuPyUnavailableError(RuntimeError):
    """Raised when the optional CuPy backend cannot use a CUDA device."""


_CUPY: Any | None = None
_CUPYX_SPARSE: Any | None = None
_CUPY_RUNTIME_VALIDATED = False
_CUPY_DEVICE_COUNT: int | None = None


def require_cupy() -> tuple[Any, Any]:
    """Return ``(cupy, cupyx.scipy.sparse)`` or raise a useful error.

    Import failures and CUDA runtime/device failures use the same public error
    type so callers can implement a clean ``auto -> scipy`` fallback.
    """

    global _CUPY, _CUPYX_SPARSE, _CUPY_RUNTIME_VALIDATED, _CUPY_DEVICE_COUNT
    if _CUPY is None or _CUPYX_SPARSE is None:
        try:
            import cupy as cp  # type: ignore[import-not-found]
            import cupyx.scipy.sparse as cpsparse  # type: ignore[import-not-found]
        except Exception as exc:
            # Some binary-loader failures surface as AttributeError or a CUDA
            # runtime exception after Python has inserted a partially
            # initialized CuPy module.  Remove only that failed import family
            # so a later probe is deterministic and ``auto`` can fall back.
            for module_name in tuple(sys.modules):
                if module_name == "cupy" or module_name.startswith(("cupy.", "cupyx.")):
                    sys.modules.pop(module_name, None)
            raise CuPyUnavailableError(
                "CuPy is not installed or its CUDA libraries cannot be loaded. "
                "Install the CuPy wheel matching the local CUDA runtime to use "
                "backend='cupy'."
            ) from exc
        _CUPY = cp
        _CUPYX_SPARSE = cpsparse

    # Import/device discovery is a process-level prerequisite, not a
    # Hamiltonian-operation check.  Calling cudaGetDeviceCount inside every
    # Chebyshev recurrence adds thousands of driver round trips to a normal
    # SCF run.  Once one usable device has been established, return the cached
    # modules directly; subsequent CUDA failures still surface at the actual
    # operation that encountered them.
    if _CUPY_RUNTIME_VALIDATED and _CUPY_DEVICE_COUNT is not None:
        return _CUPY, _CUPYX_SPARSE

    try:
        device_count = int(_CUPY.cuda.runtime.getDeviceCount())
    except Exception as exc:  # CuPy exposes several runtime-specific errors.
        raise CuPyUnavailableError(
            "CuPy imported, but the CUDA runtime is not usable."
        ) from exc
    if device_count < 1:
        raise CuPyUnavailableError("CuPy found no CUDA-capable device.")
    _CUPY_DEVICE_COUNT = device_count
    _CUPY_RUNTIME_VALIDATED = True
    return _CUPY, _CUPYX_SPARSE


def cupy_device_count() -> int:
    """Return the process-cached CUDA device count.

    Device discovery can initialize the CUDA driver and is surprisingly
    expensive on some Windows systems.  All CuPy consumers therefore share
    the result established by :func:`require_cupy` instead of probing the
    runtime again during eigensolver construction.
    """

    require_cupy()
    if _CUPY_DEVICE_COUNT is None:  # Defensive guard for monkeypatched tests.
        raise CuPyUnavailableError("CuPy device discovery did not complete.")
    return _CUPY_DEVICE_COUNT


def cupy_available() -> bool:
    """Return whether CuPy and at least one CUDA device are usable."""

    try:
        require_cupy()
    except CuPyUnavailableError:
        return False
    return True


def synchronize() -> None:
    """Wait for work in CuPy's current stream to finish."""

    cp, _ = require_cupy()
    cp.cuda.get_current_stream().synchronize()


@dataclass
class CuPyTimingStats:
    """Coarse synchronized GPU timings accumulated by one solver.

    These fields intentionally surround whole transfers or eigensolver passes.
    Timing each Hamiltonian term would insert a device synchronization into
    every Chebyshev recurrence and substantially slow the production path.
    """

    initialization_seconds: float = 0.0
    potential_update_seconds: float = 0.0
    first_solve_seconds: float = 0.0
    subspace_solve_seconds: float = 0.0
    download_seconds: float = 0.0
    density_seconds: float = 0.0
    final_wavefunction_download_seconds: float = 0.0
    subspace_bound_seconds: float = 0.0
    subspace_filter_seconds: float = 0.0
    subspace_orthogonalization_seconds: float = 0.0
    subspace_ritz_seconds: float = 0.0
    subspace_ritz_hamiltonian_seconds: float = 0.0
    subspace_ritz_projection_seconds: float = 0.0
    subspace_ritz_rotation_seconds: float = 0.0
    initial_bound_seconds: float = 0.0
    initial_filter_seconds: float = 0.0
    initial_orthogonalization_seconds: float = 0.0
    initial_projection_seconds: float = 0.0
    initial_rotation_seconds: float = 0.0
    initial_residual_seconds: float = 0.0
    initial_cleanup_seconds: float = 0.0
    initial_block_orth_calls: int = 0
    initial_block_orth_fallbacks: int = 0
    solve_calls: int = 0
    first_solve_calls: int = 0
    subspace_solve_calls: int = 0
    # Later filter passes that ran the FP32 recurrence, an opt-in.
    subspace_filter_float32_passes: int = 0
    density_calls: int = 0
    hamiltonian_applications: int = 0
    orbital_vectors_applied: int = 0
    filter_graph_launches: int = 0
    filter_graph_builds: int = 0
    filter_graph_buffer_bytes: int = 0
    initial_random_device_values: int = 0
    eigensolver_bound_prepare_wall_seconds: float = 0.0
    _pending_stage_events: list[tuple[str, Any, Any]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def as_dict(self) -> dict[str, float | int]:
        """Return a plain serializable snapshot."""

        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }


def _stage_timing_requested() -> bool:
    """Return whether optional asynchronous CheFSI stage events are enabled."""

    value = os.environ.get("PARSEC_CUPY_STAGE_TIMING", "0").strip().lower()
    return value not in {"0", "false", "no", "off"}


@contextmanager
def device_stage(operator: Any, name: str):
    """Record one GPU stage without synchronizing the production stream.

    CUDA events are inserted into the current stream and resolved only after
    the eigensolver's existing coarse synchronization. With stage timing
    disabled this is a zero-event context, so production keeps its previous
    synchronization and launch schedule.
    """

    stats = getattr(operator, "timing_stats", None)
    if stats is None or not _stage_timing_requested():
        yield
        return
    if not hasattr(stats, name):
        raise ValueError(f"unknown CuPy timing stage {name!r}")
    cp, _ = require_cupy()
    start = cp.cuda.Event()
    finish = cp.cuda.Event()
    start.record()
    try:
        yield
    finally:
        finish.record()
        stats._pending_stage_events.append((name, start, finish))


def resolve_device_stages(stats: CuPyTimingStats) -> None:
    """Accumulate stage events after an existing eigensolver synchronization."""

    if not stats._pending_stage_events:
        return
    cp, _ = require_cupy()
    pending = stats._pending_stage_events
    stats._pending_stage_events = []
    for name, start, finish in pending:
        elapsed = 1.0e-3 * float(cp.cuda.get_elapsed_time(start, finish))
        setattr(stats, name, float(getattr(stats, name)) + elapsed)


def synchronized_call(function, /, *args, **kwargs):
    """Execute one coarse GPU stage and return ``(result, seconds)``."""

    synchronize()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    synchronize()
    return result, time.perf_counter() - started


def _host_csr(matrix: Any, shape: tuple[int, int] | None = None) -> sp.csr_matrix:
    """Normalize a host matrix to float64 CSR before one device upload."""

    if matrix is None:
        if shape is None:
            raise ValueError("shape is required for an empty sparse matrix")
        return sp.csr_matrix(shape, dtype=np.float64)
    if sp.issparse(matrix):
        result = matrix.astype(np.float64, copy=False).tocsr()
    else:
        result = sp.csr_matrix(np.asarray(matrix, dtype=np.float64))
    result.sum_duplicates()
    result.sort_indices()
    return result


class CuPyHamiltonian:
    """Float64 device-resident Kohn--Sham Hamiltonian.

    Parameters are accepted in their reference SciPy/NumPy representation and
    uploaded once.  ``nonlocal_operator`` may be the reference
    ``NonlocalProjectorOperator`` (with ``projectors`` and ``signs``
    attributes) or a ``(projectors, signs)`` pair.  The local potential can be
    replaced between SCF iterations without rebuilding any sparse matrix.
    """

    def __init__(
        self,
        negative_laplacian: Any,
        effective_potential: Any,
        nonlocal_operator: Any | None = None,
        *,
        projectors: Any | None = None,
        projector_signs: Any | None = None,
        timing_stats: CuPyTimingStats | None = None,
        retain_generic_laplacian: bool = True,
        prefer_stencil_major: bool = True,
        use_compact_finite_difference: bool = True,
        finite_difference_metadata: (
            StencilMajorHostMetadata | PackedTileHostMetadata | None
        ) = None,
        shared_stencil_neighbors: Any | None = None,
        shared_effective_potential: Any | None = None,
        implicit_tile: int | None = None,
    ) -> None:
        self.timing_stats = timing_stats or CuPyTimingStats()

        def initialize() -> None:
            cp, cpsparse = require_cupy()
            host_a = (
                sp.csr_matrix(finite_difference_metadata.shape, dtype=np.float64)
                if finite_difference_metadata is not None
                else _host_csr(negative_laplacian)
            )
            if host_a.shape[0] != host_a.shape[1]:
                raise ValueError("negative_laplacian must be square")
            dimension = int(host_a.shape[0])

            supplied_projectors = projectors
            supplied_signs = projector_signs
            if nonlocal_operator is not None:
                if projectors is not None or projector_signs is not None:
                    raise ValueError(
                        "pass either nonlocal_operator or projectors/signs, not both"
                    )
                if isinstance(nonlocal_operator, tuple) and len(nonlocal_operator) == 2:
                    supplied_projectors, supplied_signs = nonlocal_operator
                else:
                    supplied_projectors = getattr(nonlocal_operator, "projectors", None)
                    supplied_signs = getattr(nonlocal_operator, "signs", None)
                    if supplied_projectors is None or supplied_signs is None:
                        raise TypeError(
                            "nonlocal_operator must expose projectors and signs"
                        )

            host_b = _host_csr(supplied_projectors, (dimension, 0))
            if host_b.shape[0] != dimension:
                raise ValueError("projector rows must match the Hamiltonian dimension")
            projector_count = int(host_b.shape[1])
            if supplied_signs is None:
                host_signs = np.empty(projector_count, dtype=np.float64)
                if projector_count:
                    raise ValueError("projector_signs are required for nonempty projectors")
            else:
                host_signs = np.asarray(supplied_signs, dtype=np.float64)
            if host_signs.shape != (projector_count,):
                raise ValueError("projector_signs must have one entry per projector")

            host_potential = np.asarray(effective_potential, dtype=np.float64)
            if host_potential.shape != (dimension,):
                raise ValueError("effective_potential must match the Hamiltonian dimension")

            # ``compact_finite_difference`` is the common selected-kernel
            # slot retained for API compatibility.  Prefer the transposed
            # stencil-major layout on CUDA; if allocation/JIT fails, retain
            # the earlier CSR-order compact kernel and finally generic CSR.
            self.compact_finite_difference = None
            self.compact_finite_difference_reason = "CuPy RawKernel unavailable"
            if hasattr(cp, "RawKernel") and use_compact_finite_difference:
                failures: list[str] = []
                stencil_requested = os.environ.get(
                    "PARSEC_CUPY_STENCIL_MAJOR", "1"
                ).strip().lower() not in {"0", "false", "no", "off"}
                stencil_requested = stencil_requested and bool(
                    prefer_stencil_major
                )
                if stencil_requested:
                    # Affine tiles are a layout of the stencil-major kernels.
                    # A size that the environment names is built or is an
                    # error; one that the default chose for a sector gives
                    # way to the slot-major layout, which every route reads,
                    # and never to the CSR-order kernel below.
                    named_tile = implicit_tile_setting()
                    tile = (
                        (named_tile or 0)
                        if implicit_tile is None
                        else int(implicit_tile)
                    )
                    if tile:
                        # A setting of the packer that is no thread count is
                        # an error as well, whoever chose the tiles.
                        pack_worker_setting()
                    # A stencil that arrives packed, the tiles of a sector
                    # for another of its devices, has that layout alone.
                    packed = isinstance(
                        finite_difference_metadata, PackedTileHostMetadata
                    )
                    for layout in dict.fromkeys((tile, 0)):
                        try:
                            self.compact_finite_difference = (
                                CuPyStencilMajorFiniteDifference(
                                    cp,
                                    host_a,
                                    metadata=finite_difference_metadata,
                                    device_neighbors=shared_stencil_neighbors,
                                    implicit_tile=layout,
                                )
                            )
                            break
                        except Exception as error:
                            if packed or (layout and named_tile is not None):
                                raise
                            failures.append(
                                (
                                    f"affine tiles of {layout} "
                                    if layout
                                    else "stencil-major "
                                )
                                + f"{type(error).__name__}: {error}"
                            )
                else:
                    failures.append(
                        "stencil-major disabled by operator policy/environment"
                    )
                try:
                    if self.compact_finite_difference is None:
                        if finite_difference_metadata is not None:
                            host_a = finite_difference_metadata.to_csr()
                        self.compact_finite_difference = CuPyCompactFiniteDifference(
                            cp, host_a
                        )
                except Exception as error:
                    failures.append(
                        "compact-CSR " f"{type(error).__name__}: {error}"
                    )
                # Generic CSR remains an exact and fully supported path;
                # expose every fallback reason so provenance is auditable.
                self.compact_finite_difference_reason = (
                    "; ".join(failures) or None
                )
            elif not use_compact_finite_difference:
                self.compact_finite_difference_reason = (
                    "compact kernels disabled by operator policy"
                )
            # The generic CuPy CSR is an exact fallback, but it duplicates all
            # finite-difference values/indices already owned by the selected
            # compact stencil.  Representation eigensolvers never expose this
            # diagnostic matrix, so omit the duplicate allocation there.
            self.negative_laplacian = (
                cpsparse.csr_matrix(
                    finite_difference_metadata.to_csr()
                    if finite_difference_metadata is not None
                    and host_a.nnz == 0
                    else host_a
                )
                if retain_generic_laplacian
                or self.compact_finite_difference is None
                else None
            )
            self.projector_signs = cp.asarray(host_signs, dtype=cp.float64)
            fused_projectors_requested = os.environ.get(
                "PARSEC_CUPY_FUSED_PROJECTORS", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}
            fused_projectors_eligible = bool(
                fused_projectors_requested
                and projector_count
                and getattr(
                    self.compact_finite_difference,
                    "supports_fused_projector_scatter",
                    False,
                )
                and host_b.indptr.dtype == np.int32
                and host_b.indices.dtype == np.int32
            )
            custom_projection_requested = os.environ.get(
                "PARSEC_CUPY_CUSTOM_PROJECTOR_DOT", "1"
            ).strip().lower() not in {"0", "false", "no", "off"}
            self._projector_count = projector_count
            self.raw_projector_factors = None
            self.projectors = None
            self.projectors_transpose = None
            self.custom_projector_projection = None
            # Representation operators omit the generic Laplacian and never
            # need a cuSPARSE projector object in production.  Upload raw CSR
            # factors directly, avoiding both duplicate storage and cold
            # cuSPARSE dynamic-library initialization.  The generic/full-grid
            # backend keeps its public sparse matrices for modular inspection.
            raw_factor_path = bool(
                not retain_generic_laplacian
                and fused_projectors_eligible
                and custom_projection_requested
            )
            if raw_factor_path:
                try:
                    self.raw_projector_factors = CuPySparseProjectorFactors(
                        cp, host_b, self.projector_signs
                    )
                    self.custom_projector_projection = (
                        self.raw_projector_factors.projection
                    )
                except Exception:
                    # NVRTC/cache failures are optimization failures only.
                    # Construct the exact cuSPARSE fallback below.
                    self.raw_projector_factors = None
                    self.custom_projector_projection = None

            if projector_count == 0:
                # Do not initialize cuSPARSE merely to own an empty factor.
                # asarray also supports the small NumPy-backed CuPy shim used
                # to validate the optional-runtime Poisson path.
                empty_offsets = cp.asarray(
                    np.zeros(dimension + 1, dtype=np.int32)
                )
                self.projector_csr_data = (
                    empty_offsets,
                    cp.asarray(np.empty(0, dtype=np.int32)),
                    cp.asarray(np.empty(0, dtype=np.float64)),
                )
            elif self.raw_projector_factors is None:
                self.projectors = cpsparse.csr_matrix(host_b)
                # Cache CSR B.T explicitly. Rebuilding/converting it in every
                # Hamiltonian application would defeat the low-rank factors.
                host_b_transpose = host_b.T.tocsr(copy=True)
                host_b_transpose.sum_duplicates()
                host_b_transpose.sort_indices()
                projector_row_lengths = np.diff(host_b_transpose.indptr)
                max_projector_entries = int(
                    projector_row_lengths.max(initial=0)
                )
                self.projectors_transpose = cpsparse.csr_matrix(
                    host_b_transpose
                )
                if custom_projection_requested and projector_count:
                    try:
                        self.custom_projector_projection = (
                            CuPySparseProjectorProjection(
                                cp,
                                self.projectors_transpose,
                                self.projector_signs,
                                max_row_entries=max_projector_entries,
                            )
                        )
                    except Exception:
                        self.custom_projector_projection = None
                self.projector_csr_data = (
                    self.projectors.indptr,
                    self.projectors.indices,
                    self.projectors.data,
                )
            else:
                self.projector_csr_data = self.raw_projector_factors.csr_data
            self.fused_projector_scatter = bool(
                fused_projectors_eligible
                and (
                    self.raw_projector_factors is not None
                    or (
                        self.projectors is not None
                        and self.projectors.indptr.dtype == cp.dtype(cp.int32)
                        and self.projectors.indices.dtype == cp.dtype(cp.int32)
                    )
                )
            )
            # FP64 unless PARSEC_CUPY_MIXED_FILTER asks for the FP32 filter.
            mixed_filter = mixed_filter_policy()
            mixed_filter_requested = float32_filter_requested(dimension)
            self.mixed_precision_recurrence = None
            self.mixed_precision_filter_reason = (
                "below automatic row threshold"
                if mixed_filter == "auto" and not mixed_filter_requested
                else "disabled by policy"
            )
            if mixed_filter_requested and isinstance(
                self.compact_finite_difference,
                CuPyStencilMajorFiniteDifference,
            ):
                try:
                    self.mixed_precision_recurrence = (
                        CuPyMixedPrecisionRecurrence(
                            cp,
                            self.compact_finite_difference,
                            host_b,
                            host_signs,
                            host_potential,
                        )
                    )
                    self.mixed_precision_filter_reason = None
                except Exception as error:
                    self.mixed_precision_filter_reason = (
                        f"{type(error).__name__}: {error}"
                    )
                    if mixed_filter == "on":
                        raise
            elif mixed_filter_requested:
                self.mixed_precision_filter_reason = (
                    "stencil-major CUDA kernel unavailable"
                )
            if shared_effective_potential is None:
                self.effective_potential = cp.asarray(
                    host_potential, dtype=cp.float64
                )
            else:
                shared = cp.asarray(shared_effective_potential)
                if shared.shape != (dimension,) or (
                    shared.dtype != cp.dtype(cp.float64)
                ):
                    raise ValueError(
                        "shared_effective_potential must be one float64 "
                        "device vector matching the Hamiltonian dimension"
                    )
                self.effective_potential = shared
            self.shape = (dimension, dimension)
            self.dtype = cp.dtype(cp.float64)

        _, elapsed = synchronized_call(initialize)
        self.timing_stats.initialization_seconds += elapsed

    @property
    def projector_count(self) -> int:
        return self._projector_count

    def update_local_potential(self, effective_potential: Any) -> None:
        """Upload one new SCF local field while retaining ``A``, ``B``, and state."""

        cp, _ = require_cupy()

        def update() -> None:
            potential = cp.asarray(effective_potential, dtype=cp.float64)
            if potential.shape != (self.shape[0],):
                raise ValueError(
                    "effective_potential must match the Hamiltonian dimension"
                )
            # Retain the allocation.  Besides avoiding a repeated allocator
            # round trip, stable device pointers make this operator compatible
            # with future CUDA-graph capture.  copyto also preserves ownership
            # when the caller supplied another device array.
            if hasattr(cp, "copyto"):
                cp.copyto(self.effective_potential, potential)
            else:
                # NumPy-backed optional-runtime test adapters do not expose
                # cupy.copyto but do preserve ordinary slice assignment.
                self.effective_potential[...] = potential
            if self.mixed_precision_recurrence is not None:
                self.mixed_precision_recurrence.update_potential(potential)

        _, elapsed = synchronized_call(update)
        self.timing_stats.potential_update_seconds += elapsed

    def apply_kinetic(self, vectors: Any):
        """Apply the finite-difference CSR operator without synchronization."""

        cp, _ = require_cupy()
        block = cp.asarray(vectors, dtype=cp.float64)
        if self.compact_finite_difference is not None:
            return self.compact_finite_difference.apply(block)
        if self.negative_laplacian is None:
            raise RuntimeError("finite-difference operator has no usable storage")
        return self.negative_laplacian @ block

    def apply_local(self, vectors: Any):
        """Apply the current diagonal local field without synchronization."""

        cp, _ = require_cupy()
        block = cp.asarray(vectors, dtype=cp.float64)
        if block.ndim == 1:
            return self.effective_potential * block
        return self.effective_potential[:, None] * block

    def apply_nonlocal(self, vectors: Any):
        """Apply ``B diag(signs) B.T`` without materializing it."""

        cp, _ = require_cupy()
        block = cp.asarray(vectors, dtype=cp.float64)
        if self.projector_count == 0:
            return cp.zeros_like(block)
        if self.raw_projector_factors is not None:
            return self.raw_projector_factors.apply(block)
        coefficients = self.projectors_transpose @ block
        if block.ndim == 1:
            coefficients *= self.projector_signs
        else:
            coefficients *= self.projector_signs[:, None]
        return self.projectors @ coefficients

    def _nonlocal_action_or_none(self, block: Any):
        """Return the low-rank action, avoiding an N-by-block zero allocation."""

        if self.projector_count == 0:
            return None
        if self.raw_projector_factors is not None:
            return self.raw_projector_factors.apply(block)
        coefficients = self.projectors_transpose @ block
        if block.ndim == 1:
            coefficients *= self.projector_signs
        else:
            coefficients *= self.projector_signs[:, None]
        return self.projectors @ coefficients

    def _signed_nonlocal_coefficients_or_none(self, block: Any):
        """Return ``diag(signs) B.T X`` without the full-grid KB scatter."""

        if self.projector_count == 0:
            return None
        if self.custom_projector_projection is not None:
            return self.custom_projector_projection(block)
        coefficients = self.projectors_transpose @ block
        if block.ndim == 1:
            coefficients *= self.projector_signs
        else:
            coefficients *= self.projector_signs[:, None]
        return coefficients

    def apply(self, vectors: Any):
        """Apply the complete Hamiltonian with no intermediate synchronization."""

        cp, _ = require_cupy()
        block = cp.asarray(vectors, dtype=cp.float64)
        if block.ndim not in (1, 2) or block.shape[0] != self.shape[0]:
            raise ValueError("vectors must have shape (dimension,) or (dimension, count)")
        # Counting a Python-level operator call does not synchronize CUDA and
        # is therefore safe inside Chebyshev recurrences.  We deliberately do
        # not time each call: accurate device timing would require a stream
        # synchronization that materially changes the production workload.
        self.timing_stats.hamiltonian_applications += 1
        self.timing_stats.orbital_vectors_applied += (
            1 if block.ndim == 1 else int(block.shape[1])
        )
        if self.compact_finite_difference is not None and self.fused_projector_scatter:
            coefficients = self._signed_nonlocal_coefficients_or_none(block)
            return self.compact_finite_difference.apply(
                block,
                self.effective_potential,
                projector_data=self.projector_csr_data,
                projector_coefficients=coefficients,
            )
        if self.compact_finite_difference is not None:
            result = self.compact_finite_difference.apply(
                block, self.effective_potential
            )
        else:
            if self.negative_laplacian is None:
                raise RuntimeError("finite-difference operator has no usable storage")
            result = self.negative_laplacian @ block
            if block.ndim == 1:
                result += self.effective_potential * block
            else:
                result += self.effective_potential[:, None] * block
        nonlocal_values = self._nonlocal_action_or_none(block)
        if nonlocal_values is not None:
            result += nonlocal_values
        return result

    def apply_into(self, vectors: Any, output: Any):
        """Apply ``H`` into a caller-owned device workspace.

        Later generalized Rayleigh--Ritz needs both ``X`` and ``H X``.  On a
        large full-grid problem, allocating a fresh ``N x states`` result in
        every SCF iteration can force CuPy's memory pool to evict other large
        blocks.  This explicit-output variant preserves the same fused kernel
        and arithmetic while allowing the eigensolver to retain one workspace
        across iterations.
        """

        cp, _ = require_cupy()
        block = cp.asarray(vectors, dtype=cp.float64)
        target = output
        if block.ndim not in (1, 2) or block.shape[0] != self.shape[0]:
            raise ValueError(
                "vectors must have shape (dimension,) or (dimension, count)"
            )
        if (
            not isinstance(target, cp.ndarray)
            or target.dtype != cp.dtype(cp.float64)
            or target.shape != block.shape
        ):
            raise ValueError(
                "output must be a float64 device array matching vectors"
            )
        was_vector = block.ndim == 1
        kernel_block = block[:, None] if was_vector else block
        kernel_target = target[:, None] if was_vector else target
        self.timing_stats.hamiltonian_applications += 1
        self.timing_stats.orbital_vectors_applied += (
            1 if was_vector else int(block.shape[1])
        )
        if (
            self.compact_finite_difference is not None
            and self.fused_projector_scatter
        ):
            coefficients = self._signed_nonlocal_coefficients_or_none(
                kernel_block
            )
            self.compact_finite_difference.apply(
                kernel_block,
                self.effective_potential,
                projector_data=self.projector_csr_data,
                projector_coefficients=coefficients,
                output=kernel_target,
            )
            return target

        # Exact compatibility fallback for optional compact/generic paths
        # that do not expose an output buffer.  Production stencil-major GPU
        # calculations take the allocation-free branch above.
        if self.compact_finite_difference is not None:
            result = self.compact_finite_difference.apply(
                kernel_block, self.effective_potential
            )
        else:
            if self.negative_laplacian is None:
                raise RuntimeError(
                    "finite-difference operator has no usable storage"
                )
            result = self.negative_laplacian @ kernel_block
            result += self.effective_potential[:, None] * kernel_block
        nonlocal_values = self._nonlocal_action_or_none(kernel_block)
        if nonlocal_values is not None:
            result += nonlocal_values
        kernel_target[...] = result
        return target

    def chebyshev_recurrence(
        self,
        current: Any,
        *,
        center: float,
        scale: float,
        sigma_next: float = 1.0,
        previous: Any | None = None,
        sigma: float = 0.0,
    ):
        """Apply ``H`` and the normalized Chebyshev update in one CUDA pass.

        The method is deliberately part of the concrete CuPy operator rather
        than the generic backend protocol.  Eigensolvers feature-detect it and
        retain the ordinary ``operator @ X`` implementation for reference,
        CPU, fake-device, and compact-CSR fallback operators.
        """

        cp, _ = require_cupy()
        block = (
            current
            if isinstance(current, cp.ndarray)
            and current.dtype == cp.dtype(cp.float64)
            else cp.asarray(current, dtype=cp.float64)
        )
        if block.ndim not in (1, 2) or block.shape[0] != self.shape[0]:
            raise ValueError("current must match the Hamiltonian dimension")
        if previous is not None:
            previous_block = (
                previous
                if isinstance(previous, cp.ndarray)
                and previous.dtype == cp.dtype(cp.float64)
                else cp.asarray(previous, dtype=cp.float64)
            )
            if previous_block.shape != block.shape:
                raise ValueError("previous and current recurrence blocks must match")
        else:
            previous_block = None

        selected = self.compact_finite_difference
        fused = getattr(selected, "chebyshev_recurrence", None)
        if fused is None or (
            self.projector_count and not self.fused_projector_scatter
        ):
            applied = self.apply(block)
            result = (applied - float(center) * block) * float(scale)
            if previous_block is not None:
                result -= float(sigma) * previous_block
            result *= float(sigma_next)
            return result

        self.timing_stats.hamiltonian_applications += 1
        self.timing_stats.orbital_vectors_applied += (
            1 if block.ndim == 1 else int(block.shape[1])
        )
        if self.fused_projector_scatter:
            projector_coefficients = self._signed_nonlocal_coefficients_or_none(block)
            projector_data = self.projector_csr_data
        else:
            projector_coefficients = None
            projector_data = None
        return fused(
            block,
            self.effective_potential,
            center=float(center),
            scale=float(scale),
            sigma_next=float(sigma_next),
            previous=previous_block,
            sigma=float(sigma),
            projector_data=projector_data,
            projector_coefficients=projector_coefficients,
        )

    def chebyshev_recurrence_float32(
        self,
        current: Any,
        *,
        center: float,
        scale: float,
        sigma_next: float = 1.0,
        previous: Any | None = None,
        sigma: float = 0.0,
    ):
        """Execute a later-filter recurrence in FP32 when safely selected."""

        recurrence = self.mixed_precision_recurrence
        if recurrence is None:
            raise RuntimeError("mixed-precision filtering is unavailable")
        cp, _ = require_cupy()
        block = cp.asarray(current, dtype=cp.float32)
        self.timing_stats.hamiltonian_applications += 1
        self.timing_stats.orbital_vectors_applied += (
            1 if block.ndim == 1 else int(block.shape[1])
        )
        return recurrence(
            block,
            center=float(center),
            scale=float(scale),
            sigma_next=float(sigma_next),
            previous=previous,
            sigma=float(sigma),
        )

    def __matmul__(self, vectors: Any):
        return self.apply(vectors)


__all__ = [
    "CuPyHamiltonian",
    "CuPyTimingStats",
    "CuPyUnavailableError",
    "cupy_available",
    "device_stage",
    "require_cupy",
    "resolve_device_stages",
    "synchronize",
    "synchronized_call",
]
