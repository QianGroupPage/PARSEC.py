"""Readable public workflow for the additive accelerated architecture."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Callable

import numpy as np

from parsec_python.SCF.single_point import (
    prepare_single_point as prepare_reference_single_point,
)
from parsec_python.models import (
    PreparationTimings,
    SCFIteration,
    SinglePointInput,
)

from .SCF.single_point import (
    AcceleratedPreparedSinglePointSystem,
    run_scf as run_accelerated_scf,
)
from .Hartree.poisson import build_hartree_problem, solve_scipy_hartree
from .backends.scipy import ScipyHamiltonianBackend
from .backends.selection import BackendSelection, resolve_backend
from .models import (
    AcceleratedSinglePointResult,
    BackendName,
    BackendUnavailableError,
    SymmetryMode,
)


_REFERENCE_CACHE_FORMAT = 1
_REFERENCE_CACHE: OrderedDict[str, object] = OrderedDict()
_REFERENCE_CACHE_LOCK = Lock()


def _resident_reference_cache_size() -> int:
    """Return the bounded resident static-system cache capacity."""

    if os.environ.get("PARSEC_ACCELERATED_RESIDENT", "0").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return 0
    raw = os.environ.get("PARSEC_RESIDENT_REFERENCE_CACHE_SIZE", "1").strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE must be an integer"
        ) from error
    if value < 0:
        raise ValueError(
            "PARSEC_RESIDENT_REFERENCE_CACHE_SIZE cannot be negative"
        )
    return value


def _reference_cache_key(
    problem: SinglePointInput,
    selection: BackendSelection,
    *,
    defer_native_laplacian: bool,
    cache_directory: os.PathLike[str] | str | None,
) -> str:
    """Hash every physical/static input used by reference preparation."""

    digest = sha256()
    digest.update(f"resident-reference-v{_REFERENCE_CACHE_FORMAT}".encode("ascii"))
    digest.update(selection.finite_difference_builder.encode("ascii"))
    digest.update(bytes((int(bool(defer_native_laplacian)),)))
    cache_path = (
        "disabled"
        if cache_directory is None
        else str(Path(cache_directory).resolve())
    )
    digest.update(cache_path.encode("utf-8"))
    for settings in (
        problem.grid,
        problem.scf,
        problem.hartree,
        problem.eigensolver,
        problem.mixing,
        problem.initial_density_settings,
        problem.recenter_geometry,
    ):
        digest.update(repr(settings).encode("utf-8"))
    initial_source = problem.initial_density_settings.file
    if initial_source is not None:
        source_path = Path(initial_source).resolve()
        digest.update(str(source_path).encode("utf-8"))
        if source_path.is_file():
            with source_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    model_checkpoint = problem.initial_density_settings.checkpoint
    if model_checkpoint is not None:
        checkpoint_path = Path(model_checkpoint).resolve()
        digest.update(str(checkpoint_path).encode("utf-8"))
        if checkpoint_path.is_file():
            checkpoint_stat = checkpoint_path.stat()
            digest.update(
                np.asarray(
                    [checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns],
                    dtype=np.int64,
                ).tobytes()
            )
    if (
        problem.initial_density_settings.method in {"charge3net", "scdp"}
        and problem.initial_density_settings.file is None
    ):
        from parsec_python.MLDensity.providers import provider_source_fingerprint

        digest.update(
            provider_source_fingerprint(problem.initial_density_settings)
        )
    for atom in problem.atoms:
        digest.update(atom.symbol.encode("utf-8"))
        digest.update(
            np.ascontiguousarray(atom.position, dtype=np.float64).tobytes()
        )
    for symbol, specification in sorted(problem.pseudopotentials.items()):
        path = Path(specification.path).resolve()
        digest.update(symbol.encode("utf-8"))
        digest.update(str(path).encode("utf-8"))
        digest.update(np.int64(specification.local_angular_momentum).tobytes())
        digest.update(bytes((
            int(bool(specification.read_valence_density)),
            int(bool(specification.use_spline)),
        )))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _cached_reference_view(
    reference,
    *,
    cache_directory: os.PathLike[str] | str | None,
    lookup_seconds: float,
):
    """Return fresh timing/descriptor metadata over immutable cached arrays."""

    negative_laplacian = reference.negative_laplacian
    from .Laplacian import DeferredNativeNegativeLaplacian

    if isinstance(negative_laplacian, DeferredNativeNegativeLaplacian):
        negative_laplacian = DeferredNativeNegativeLaplacian(
            reference.grid,
            cache_directory=cache_directory,
        )
        negative_laplacian.reference_static_cache_status = "hit"
        negative_laplacian.reference_static_cache_lookup_seconds = lookup_seconds
    return replace(
        reference,
        negative_laplacian=negative_laplacian,
        timings=PreparationTimings(total_seconds=lookup_seconds),
    )


def _remember_reference(cache_key: str, reference, capacity: int) -> None:
    """Insert one immutable prepared system into the bounded process cache."""

    if capacity < 1:
        return
    with _REFERENCE_CACHE_LOCK:
        _REFERENCE_CACHE[cache_key] = reference
        _REFERENCE_CACHE.move_to_end(cache_key)
        while len(_REFERENCE_CACHE) > capacity:
            _REFERENCE_CACHE.popitem(last=False)


def _prepare_reference_physics(
    problem: SinglePointInput,
    selection: BackendSelection,
    *,
    defer_native_laplacian: bool = False,
    deferred_laplacian_cache_directory: os.PathLike[str] | str | None = None,
):
    cache_capacity = _resident_reference_cache_size()
    # A forced fresh model prediction must not be bypassed by the warmed
    # prepared-system cache.  The ordinary exact-key ML prediction cache still
    # makes repeat calculations inexpensive when ``regenerate`` is false.
    initial_settings = getattr(problem, "initial_density_settings", None)
    if initial_settings is not None and initial_settings.regenerate:
        cache_capacity = 0
    cache_started = perf_counter()
    cache_key = None
    if cache_capacity:
        cache_key = _reference_cache_key(
            problem,
            selection,
            defer_native_laplacian=defer_native_laplacian,
            cache_directory=deferred_laplacian_cache_directory,
        )
        with _REFERENCE_CACHE_LOCK:
            cached_reference = _REFERENCE_CACHE.get(cache_key)
            if cached_reference is not None:
                _REFERENCE_CACHE.move_to_end(cache_key)
        if cached_reference is not None:
            return _cached_reference_view(
                cached_reference,
                cache_directory=deferred_laplacian_cache_directory,
                lookup_seconds=perf_counter() - cache_started,
            )

    # Static construction and repeated Hamiltonian execution are independent
    # choices.  In the default hybrid path C++ builds the compressed-grid
    # finite-difference CSR once, then CuPy owns the repeated H@Q operations.
    if selection.finite_difference_builder == "native":
        from .backends.native import _load_native, build_native_negative_laplacian
        from .V_ion import NativeIonicBuilders

        try:
            native_module = _load_native()
        except BackendUnavailableError:
            # Selection normally establishes availability first.  Keeping
            # this optional optimization guard also makes the finite-
            # difference builder independently testable and supports older
            # extension builds that predate the radial kernel.
            native_module = None
        ionic_builders = (
            NativeIonicBuilders()
            if native_module is not None
            and hasattr(native_module, "RadialGridEvaluator")
            else None
        )

        if defer_native_laplacian:
            from .Laplacian import DeferredNativeNegativeLaplacian

            def laplacian_builder(grid):
                return DeferredNativeNegativeLaplacian(
                    grid,
                    cache_directory=deferred_laplacian_cache_directory,
                )
        else:
            laplacian_builder = build_native_negative_laplacian
        builder_options = {"negative_laplacian_builder": laplacian_builder}
        if ionic_builders is not None:
            builder_options.update(
                local_ionic_builder=(
                    ionic_builders.build_local_ionic_potential
                ),
                nonlocal_projector_builder=(
                    ionic_builders.build_nonlocal_projectors
                ),
                atomic_density_builder=(
                    ionic_builders.superpose_atomic_density
                ),
            )
        reference = prepare_reference_single_point(problem, **builder_options)
        if cache_key is not None:
            _remember_reference(cache_key, reference, cache_capacity)
            negative_laplacian = reference.negative_laplacian
            if defer_native_laplacian:
                negative_laplacian.reference_static_cache_status = "miss-stored"
                negative_laplacian.reference_static_cache_lookup_seconds = (
                    perf_counter() - cache_started
                )
        return reference
    if selection.finite_difference_builder != "reference":
        raise RuntimeError(
            "unhandled finite-difference builder "
            f"{selection.finite_difference_builder!r}"
        )
    reference = prepare_reference_single_point(problem)
    if cache_key is not None:
        _remember_reference(cache_key, reference, cache_capacity)
    return reference


def _build_backend(
    reference,
    selection: BackendSelection,
    *,
    defer_cupy_device_operator: bool = False,
):
    common = {
        "requested": selection.requested,
        "fallback_reasons": selection.fallback_reasons,
    }
    if selection.selected == "scipy":
        return ScipyHamiltonianBackend(
            reference.negative_laplacian,
            reference.nonlocal_operator,
            **common,
        )
    if selection.selected == "native":
        from .backends.native import NativeHamiltonianBackend

        return NativeHamiltonianBackend(
            reference.negative_laplacian,
            reference.nonlocal_operator,
            **common,
        )
    if selection.selected == "cupy":
        from .backends.cupy_runtime import CuPyHamiltonianBackend

        return CuPyHamiltonianBackend(
            reference.negative_laplacian,
            reference.nonlocal_operator,
            defer_device_operator=defer_cupy_device_operator,
            **common,
        )
    raise RuntimeError(f"unhandled backend selection {selection.selected!r}")


def _normalize_symmetry_mode(value: SymmetryMode | str) -> SymmetryMode:
    """Validate the public auto/on/off symmetry policy."""

    normalized = str(value).strip().lower()
    if normalized not in {"auto", "on", "off"}:
        raise ValueError("symmetry must be one of 'auto', 'on', or 'off'")
    return normalized  # type: ignore[return-value]


def _build_native_boundary_builder(
    reference,
    hartree_reduction,
    *,
    symmetry_cache_directory: os.PathLike[str] | str | None,
    symmetry_geometry_cache_info,
):
    """Construct the reusable isolated-boundary geometry.

    This setup depends only on the already-built real-space grid and exact
    symmetry metadata.  It is deliberately kept separate from the Poisson
    solver so the hybrid path can prepare it on a CPU worker while the main
    thread uploads/builds the independent GPU orbital operators.
    """

    from .Hartree.native_boundary import (
        NativeMultipoleBoundaryBuilder,
        NativeSymmetryMultipoleBoundaryBuilder,
    )

    started = perf_counter()
    if hartree_reduction is None:
        builder = NativeMultipoleBoundaryBuilder(
            reference.grid,
            reference.input.hartree.multipole_order,
        )
        symmetry_boundary = False
        cache_info = None
    else:
        try:
            builder = NativeSymmetryMultipoleBoundaryBuilder(
                reference.grid,
                hartree_reduction,
                reference.input.hartree.multipole_order,
                cache_directory=symmetry_cache_directory,
                cache_key_seed=(
                    None
                    if symmetry_geometry_cache_info is None
                    else symmetry_geometry_cache_info.key
                ),
            )
        except RuntimeError:
            # A previously installed 0.3 extension remains a valid full-grid
            # fallback until the rebuilt wheel is installed.
            builder = NativeMultipoleBoundaryBuilder(
                reference.grid,
                reference.input.hartree.multipole_order,
            )
            symmetry_boundary = False
            cache_info = None
        else:
            symmetry_boundary = True
            cache_info = getattr(builder, "cache_info", None)
    return (
        builder,
        symmetry_boundary,
        cache_info,
        perf_counter() - started,
    )


def _resolve_and_prepare_reference(
    problem: SinglePointInput,
    backend: BackendName | str,
    *,
    defer_native_laplacian: bool = False,
    deferred_laplacian_cache_directory: os.PathLike[str] | str | None = None,
):
    """Overlap CUDA driver discovery with independent CPU setup.

    ``cudaGetDeviceCount`` initializes the NVIDIA driver and can consume one
    to three seconds in a fresh Windows process.  The reference grid,
    pseudopotential, ionic, and finite-difference construction does not depend
    on that driver state.  For the production ``auto`` path, execute both
    prerequisites concurrently and join them before symmetry/device
    construction. Explicit backend requests retain their strict sequential
    validation order.

    The provisional choice is used only to select the static finite-difference
    builder.  It is validated against the authoritative backend resolution;
    an availability change or a test double that selects another builder
    triggers an exact rebuild instead of silently composing incompatible
    components.
    """

    normalized = str(backend).strip().lower()

    def prepare_reference(selection):
        if defer_native_laplacian:
            return _prepare_reference_physics(
                problem,
                selection,
                defer_native_laplacian=True,
                deferred_laplacian_cache_directory=(
                    deferred_laplacian_cache_directory
                ),
            )
        return _prepare_reference_physics(problem, selection)

    overlap_requested = os.environ.get(
        "PARSEC_OVERLAP_CUDA_INITIALIZATION", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    can_overlap = overlap_requested and normalized == "auto"
    if not can_overlap:
        resolution_started = perf_counter()
        selection = resolve_backend(backend, problem)
        resolution_seconds = perf_counter() - resolution_started
        reference_started = perf_counter()
        reference = prepare_reference(selection)
        reference_seconds = perf_counter() - reference_started
        return selection, reference, {
            "cuda_initialization_overlap": "disabled",
            "backend_resolution_seconds": resolution_seconds,
            "reference_preparation_seconds": reference_seconds,
            "backend_reference_overlapped_seconds": 0.0,
        }

    # The auto static builder depends only on native availability, not on the
    # CUDA probe running in the worker.  This inexpensive check may be
    # repeated by authoritative resolution after CUDA discovery.
    from .backends.selection import _native_status

    native_available, _ = _native_status()
    provisional = BackendSelection(
        requested="auto",
        selected="native" if native_available else "scipy",
        finite_difference_builder=(
            "native" if native_available else "reference"
        ),
        hartree_backend="native" if native_available else "scipy",
    )

    overlap_started = perf_counter()

    def timed_resolution():
        started = perf_counter()
        selected = resolve_backend(backend, problem)
        return selected, perf_counter() - started

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="parsec-cuda-init",
    ) as executor:
        future = executor.submit(timed_resolution)
        reference_started = perf_counter()
        reference = prepare_reference(provisional)
        reference_seconds = perf_counter() - reference_started
        selection, resolution_seconds = future.result()
    overlap_wall_seconds = perf_counter() - overlap_started

    if (
        selection.finite_difference_builder
        != provisional.finite_difference_builder
    ):
        # Availability changed during the probe, or a caller supplied a
        # backend-selection test double.  Correctness takes priority over the
        # discarded speculative setup.
        reference_started = perf_counter()
        reference = prepare_reference(selection)
        reference_seconds += perf_counter() - reference_started
        overlapped_seconds = 0.0
        overlap_status = "rebuild_after_backend_change"
    else:
        overlapped_seconds = max(
            0.0,
            resolution_seconds + reference_seconds - overlap_wall_seconds,
        )
        overlap_status = "cuda_probe_with_cpu_reference_setup"

    return selection, reference, {
        "cuda_initialization_overlap": overlap_status,
        "backend_resolution_seconds": resolution_seconds,
        "reference_preparation_seconds": reference_seconds,
        "backend_reference_overlapped_seconds": overlapped_seconds,
    }


def prepare_single_point(
    problem: SinglePointInput,
    *,
    backend: BackendName | str = "auto",
    symmetry: SymmetryMode | str = "auto",
    symmetry_cache_directory: os.PathLike[str] | str | None = None,
) -> AcceleratedPreparedSinglePointSystem:
    """Prepare physics, automatically reducing exact symmetry when usable.

    ``auto`` (the default) uses every exactly detected Cartesian reflection
    that the selected backend supports and otherwise keeps the full-grid
    algorithm.  ``on`` turns an unusable/nontrivial symmetry into an error;
    ``off`` skips detection and is the reproducible full-grid comparison path.
    """

    symmetry_mode = _normalize_symmetry_mode(symmetry)
    selection, reference, preparation_overlap = _resolve_and_prepare_reference(
        problem,
        backend,
        defer_native_laplacian=(
            symmetry_mode != "off" and symmetry_cache_directory is not None
        ),
        deferred_laplacian_cache_directory=symmetry_cache_directory,
    )
    # A few backend-wiring tests use a deliberately minimal reference object.
    # Preserve the historical CA default for such result-like objects while
    # real prepared systems always carry the parsed SCF functional.
    xc_functional = getattr(
        getattr(getattr(reference, "input", None), "scf", None),
        "xc_functional",
        "ca",
    )

    symmetry_reduction = None
    symmetry_geometry_cache_info = None
    symmetry_representation_cache_info = None
    detected_group_order = 1
    symmetry_detection = "disabled by option"
    if symmetry_mode != "off":
        from .Symmetry import load_or_detect_reflection_reduction

        try:
            candidate, symmetry_geometry_cache_info = (
                load_or_detect_reflection_reduction(
                    reference.grid,
                    getattr(reference, "atoms", ()),
                    cache_directory=symmetry_cache_directory,
                )
            )
        except (ValueError, RuntimeError) as error:
            if symmetry_mode == "on":
                raise ValueError(
                    "symmetry='on' requested, but exact symmetry detection "
                    f"failed: {error}"
                ) from error
            symmetry_detection = (
                f"automatic detection fell back to full grid: "
                f"{type(error).__name__}: {error}"
            )
        else:
            detected_group_order = candidate.group_order
            if candidate.group_order > 1:
                symmetry_reduction = candidate
                from .Symmetry import SignedPermutationReduction

                symmetry_detection = (
                    "exact commuting signed-permutation subgroup detected"
                    if isinstance(candidate, SignedPermutationReduction)
                    else "exact Cartesian axis-reflection subgroup detected"
                )
            else:
                symmetry_detection = (
                    "identity only; no nontrivial supported symmetry detected"
                )
                if symmetry_mode == "on":
                    raise ValueError(
                        "symmetry='on' requested, but only the identity "
                        "operation preserves the labeled atoms and active grid"
                    )

    # Establish whether the orbital problem can use exact representation
    # sectors before constructing the GPU backend.  This ordering matters:
    # a sector calculation never applies the full-grid GPU Hamiltonian, so
    # uploading that duplicate allocation would be pure setup and memory cost.
    orbital_decomposition = None
    orbital_reduction = symmetry_reduction
    stabilizer_policy = os.environ.get(
        "PARSEC_ORBITAL_STABILIZER_SYMMETRY", "auto"
    ).strip().lower()
    if stabilizer_policy not in {
        "auto", "on", "off", "1", "0", "true", "false", "yes", "no"
    }:
        raise ValueError(
            "PARSEC_ORBITAL_STABILIZER_SYMMETRY must be auto, on, or off"
        )
    stabilizer_disabled = stabilizer_policy in {
        "off", "0", "false", "no"
    }
    if (
        orbital_reduction is not None
        and not np.all(
            orbital_reduction.multiplicities == orbital_reduction.group_order
        )
        and stabilizer_disabled
    ):
        # A/B control and conservative architecture override: Hartree retains
        # the detected scalar reduction, while orbitals use the full grid.
        orbital_reduction = None
    orbital_symmetry = "full grid"
    if symmetry_mode == "off":
        orbital_symmetry = "full grid (symmetry disabled)"
    elif orbital_reduction is None or orbital_reduction.group_order <= 1:
        orbital_symmetry = "full grid (no nontrivial supported symmetry)"
    elif selection.selected != "cupy":
        orbital_symmetry = (
            "full grid (representation decomposition is currently a CuPy path)"
        )
    else:
        from .Symmetry import load_or_build_reflection_decomposition

        try:
            orbital_decomposition, symmetry_representation_cache_info = (
                load_or_build_reflection_decomposition(
                    reference.grid,
                    orbital_reduction,
                    reduction_key=symmetry_geometry_cache_info.key,
                    cache_directory=symmetry_cache_directory,
                )
            )
        except (ValueError, RuntimeError) as error:
            if symmetry_mode == "on":
                raise ValueError(
                    "symmetry='on' requested, but the GPU orbital "
                    f"representation decomposition is unusable: {error}"
                ) from error
            orbital_symmetry = (
                "full grid (automatic representation fallback: "
                f"{type(error).__name__}: {error})"
            )
        else:
            sector_sizes = orbital_decomposition.sector_sizes
            orbital_symmetry = (
                "CuPy real one-dimensional representations with exact "
                "orbit-stabilizer character selection"
                if len(set(sector_sizes)) > 1
                else "CuPy real one-dimensional reflection representations"
            )

    # Decide the Hartree reduction before constructing the execution backend.
    # This lets the independent boundary-geometry setup overlap the much more
    # expensive GPU orbital-operator construction below.
    hartree_reduction = symmetry_reduction
    legacy_hartree_setting = os.environ.get(
        "PARSEC_HARTREE_SYMMETRY", "auto"
    ).strip().lower()
    legacy_hartree_disabled = legacy_hartree_setting in {
        "0", "false", "no", "off"
    }
    legacy_hartree_forced = legacy_hartree_setting in {
        "1", "true", "yes", "on"
    }
    if selection.hartree_backend != "native":
        hartree_reduction = None
    elif symmetry_mode == "off" or (
        symmetry_mode == "auto" and legacy_hartree_disabled
    ):
        hartree_reduction = None
    elif legacy_hartree_forced and hartree_reduction is None:
        raise ValueError(
            "PARSEC_HARTREE_SYMMETRY requests reduction, but no nontrivial "
            "supported symmetry was detected"
        )
    if (
        symmetry_mode == "on"
        and orbital_decomposition is None
        and not (
            selection.hartree_backend == "native"
            and hartree_reduction is not None
        )
    ):
        raise ValueError(
            "symmetry='on' requested, but the selected backend has no usable "
            "symmetry-reduced component for this calculation"
        )

    boundary_method = reference.input.hartree.boundary_method
    boundary_setup_eligible = (
        selection.hartree_backend == "native"
        and reference.grid.settings.domain_shape == "sphere"
        and boundary_method in {"auto", "multipole"}
    )
    boundary_overlap_enabled = os.environ.get(
        "PARSEC_OVERLAP_HARTREE_SETUP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    # The worker is useful when there is independent GPU setup to hide it
    # behind.  Native/SciPy-only paths keep deterministic inline construction
    # and avoid paying thread-pool overhead for no overlap opportunity.
    boundary_setup_future = None
    boundary_setup_executor = None
    boundary_setup_status = "not applicable"
    boundary_setup_seconds = 0.0
    boundary_setup_wait_seconds = 0.0
    boundary_setup_overlapped_seconds = 0.0
    boundary_setup_reduction = hartree_reduction
    if (
        boundary_setup_eligible
        and boundary_overlap_enabled
        and selection.selected == "cupy"
    ):
        boundary_setup_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="parsec-hartree-setup",
        )
        boundary_setup_future = boundary_setup_executor.submit(
            _build_native_boundary_builder,
            reference,
            hartree_reduction,
            symmetry_cache_directory=symmetry_cache_directory,
            symmetry_geometry_cache_info=symmetry_geometry_cache_info,
        )
        boundary_setup_status = "overlapped with GPU orbital setup"

    defer_full_gpu_operator = (
        orbital_decomposition is not None
        and selection.hartree_backend != "cupy"
    )
    implementation = _build_backend(
        reference,
        selection,
        defer_cupy_device_operator=defer_full_gpu_operator,
    )
    native_kernel_names: tuple[str, ...] = ()
    if (
        selection.finite_difference_builder == "native"
        or selection.hartree_backend == "native"
    ):
        from .backends.native import native_build_info

        native_kernel_names = tuple(
            str(name)
            for name in native_build_info().get("implemented_kernels", ())
        )
    native_boundary_builder = None
    native_boundary_cache_info = None
    native_symmetry_boundary = False
    native_xc_evaluator = None
    scf_reducer = None
    symmetry_eigensolver = None
    if orbital_decomposition is not None:
        from .Eigensolvers import CuPySymmetrySCFEigensolver

        symmetry_eigensolver = CuPySymmetrySCFEigensolver(
            implementation.eigensolver_operator,
            reference.negative_laplacian,
            reference.nonlocal_operator,
            orbital_decomposition,
            timing_stats=implementation.timing_stats,
            local_potential_getter=lambda: implementation.local_potential,
            operator_cache_directory=(
                None
                if symmetry_cache_directory is None
                else Path(symmetry_cache_directory)
            ),
            kinetic_cache_key=getattr(
                reference.negative_laplacian, "cache_key", None
            ),
            decomposition_cache_key=(
                None
                if symmetry_representation_cache_info is None
                else symmetry_representation_cache_info.key
            ),
        )
        implementation.symmetry_eigensolver = symmetry_eigensolver
        implementation.eigenproblem_solver = symmetry_eigensolver
        # Density and every local scalar potential are totally symmetric.
        # Retain one physical value per orbit for Anderson history, residual
        # norms, and energy quadrature; expand only the mixed field required
        # by the existing Hamiltonian/Hartree interfaces.
        from .SCF import SymmetrySCFReducer

        scf_reducer = SymmetrySCFReducer(orbital_reduction)
        from .Occupations import CuPySymmetryDensityBuilder

        implementation.orbital_density_builder = CuPySymmetryDensityBuilder(
            implementation.orbital_density_builder,
            implementation.timing_stats,
            scf_reducer,
        )
        implementation.statistics.initialization_seconds = (
            implementation.timing_stats.initialization_seconds
        )

    if (
        selection.hartree_backend == "native"
        and xc_functional == "ca"
    ):
        from .V_xc import NativeCALDAEvaluator

        if "CALDAEvaluator" in native_kernel_names:
            native_xc_evaluator = NativeCALDAEvaluator(
                reference.core_density,
                reference.grid.volume_element,
                scf_reducer,
            )

    if scf_reducer is not None and xc_functional == "pbe":
        # The readable PBE evaluator operates on the complete Cartesian
        # cluster because its density gradient couples neighboring symmetry
        # orbits.  Expand the invariant density for that operation, then
        # retain one value per orbit again for the rest of accelerated SCF.
        # This is an exact representation adapter, not a different functional.
        from parsec_python.V_xc import XCResult

        def symmetry_pbe_evaluator(density):
            evaluated = reference.evaluate_xc(scf_reducer.to_full(density))
            return XCResult(
                potential=scf_reducer.from_full(evaluated.potential),
                energy_per_electron=scf_reducer.from_full(
                    evaluated.energy_per_electron
                ),
                energy_density=scf_reducer.from_full(evaluated.energy_density),
                total_energy=evaluated.total_energy,
            )

        native_xc_evaluator = symmetry_pbe_evaluator

    if selection.hartree_backend == "cupy":
        from .Hartree.cupy_poisson import CuPyPoissonSolver

        if selection.selected != "cupy":
            raise RuntimeError("CuPy Hartree requires the CuPy execution backend")
        poisson_solver = CuPyPoissonSolver(implementation.device_operator)
        implementation.poisson_solver = poisson_solver

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            return poisson_solver.solve(
                density,
                reference.grid,
                reference.input.hartree,
                initial_potential,
                **kwargs,
            )

    elif selection.hartree_backend == "native":
        from .Hartree.native_poisson import NativePoissonSolver
        from .Hartree.symmetry_poisson import SymmetryReducedPoissonSolver
        from .Laplacian import (
            DeferredNativeNegativeLaplacian,
            materialize_negative_laplacian,
        )

        def concrete_negative_laplacian(operator):
            return (
                materialize_negative_laplacian(operator)
                if isinstance(operator, DeferredNativeNegativeLaplacian)
                else operator
            )

        if hartree_reduction is None:
            poisson_solver = NativePoissonSolver(
                concrete_negative_laplacian(reference.negative_laplacian)
            )
        else:
            try:
                reusable_reduced_operator = (
                    symmetry_eigensolver is not None
                    and orbital_reduction is hartree_reduction
                )
                poisson_solver = SymmetryReducedPoissonSolver(
                    (
                        symmetry_eigensolver.totally_symmetric_negative_laplacian
                        if reusable_reduced_operator
                        else concrete_negative_laplacian(
                            reference.negative_laplacian
                        )
                    ),
                    hartree_reduction,
                    operator_is_reduced=reusable_reduced_operator,
                )
            except (ValueError, RuntimeError):
                if symmetry_mode == "on" or legacy_hartree_forced:
                    raise
                hartree_reduction = None
                poisson_solver = NativePoissonSolver(
                    concrete_negative_laplacian(reference.negative_laplacian)
                )
        if boundary_setup_eligible:
            if boundary_setup_future is not None:
                wait_started = perf_counter()
                try:
                    (
                        native_boundary_builder,
                        native_symmetry_boundary,
                        native_boundary_cache_info,
                        boundary_setup_seconds,
                    ) = boundary_setup_future.result()
                finally:
                    boundary_setup_wait_seconds = perf_counter() - wait_started
                    boundary_setup_executor.shutdown(wait=True)
                boundary_setup_overlapped_seconds = max(
                    0.0,
                    boundary_setup_seconds - boundary_setup_wait_seconds,
                )
                # A rare reduced-Poisson construction failure changes the
                # required boundary representation after the worker began.
                # Discard that speculative symmetry builder and reconstruct
                # the exact full-grid fallback rather than mixing layouts.
                if (
                    boundary_setup_reduction is not None
                    and hartree_reduction is None
                    and native_symmetry_boundary
                ):
                    (
                        native_boundary_builder,
                        native_symmetry_boundary,
                        native_boundary_cache_info,
                        rebuild_seconds,
                    ) = _build_native_boundary_builder(
                        reference,
                        None,
                        symmetry_cache_directory=symmetry_cache_directory,
                        symmetry_geometry_cache_info=(
                            symmetry_geometry_cache_info
                        ),
                    )
                    boundary_setup_seconds += rebuild_seconds
                    boundary_setup_status = (
                        "overlap discarded after reduced-Poisson fallback"
                    )
            else:
                (
                    native_boundary_builder,
                    native_symmetry_boundary,
                    native_boundary_cache_info,
                    boundary_setup_seconds,
                ) = _build_native_boundary_builder(
                    reference,
                    hartree_reduction,
                    symmetry_cache_directory=symmetry_cache_directory,
                    symmetry_geometry_cache_info=symmetry_geometry_cache_info,
                )
                boundary_setup_status = (
                    "inline (overlap disabled)"
                    if selection.selected == "cupy"
                    else "inline (no independent GPU setup)"
                )
        # CuPy's statistics bridge expects ``poisson_solver`` to expose CuPy
        # event timings.  Keep the native solver under a distinct, inspectable
        # name and accumulate its host timings directly below.
        implementation.native_poisson_solver = poisson_solver
        implementation.native_boundary_builder = native_boundary_builder

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            total_started = perf_counter()
            rhs_started = perf_counter()
            # A symmetry-wedge calculation represents only the invariant
            # density.  Project before constructing multipoles so the
            # returned boundary object and corrected RHS describe the same
            # physical source, even if full-grid eigensolver roundoff left a
            # tiny difference between symmetry images.
            boundary_density = density
            if native_symmetry_boundary:
                right_hand_side, boundary = native_boundary_builder.build_reduced(
                    density
                )
            elif native_boundary_builder is None:
                boundary_density = (
                    density
                    if hartree_reduction is None
                    else hartree_reduction.project_invariant(density)
                )
                right_hand_side, boundary = build_hartree_problem(
                    boundary_density,
                    reference.grid,
                    reference.input.hartree,
                )
            else:
                right_hand_side, boundary = native_boundary_builder.build(
                    boundary_density
                )
            rhs_seconds = perf_counter() - rhs_started
            solve_started = perf_counter()
            if native_symmetry_boundary:
                compact_hartree_output = (
                    scf_reducer is not None
                    and scf_reducer.reduction is hartree_reduction
                )
                native_result = poisson_solver.solve_reduced(
                    right_hand_side,
                    initial_potential,
                    reference.input.hartree,
                    return_wedge=compact_hartree_output,
                    **kwargs,
                )
                if scf_reducer is not None and not compact_hartree_output:
                    from dataclasses import replace as dataclass_replace

                    native_result = dataclass_replace(
                        native_result,
                        potential=scf_reducer.from_full(native_result.potential),
                        right_hand_side=scf_reducer.from_full(
                            native_result.right_hand_side
                        ),
                    )
            else:
                native_result = poisson_solver.solve(
                    right_hand_side,
                    initial_potential,
                    reference.input.hartree,
                    **kwargs,
                )
            solve_seconds = perf_counter() - solve_started
            implementation.statistics.hartree_solve_calls += 1
            implementation.statistics.hartree_rhs_seconds += rhs_seconds
            implementation.statistics.hartree_linear_solve_seconds += solve_seconds
            implementation.statistics.hartree_total_seconds += (
                perf_counter() - total_started
            )
            return native_result.as_hartree_result(boundary)

    elif selection.hartree_backend == "scipy":

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            started = perf_counter()
            result = solve_scipy_hartree(
                density,
                reference.grid,
                reference.negative_laplacian,
                reference.input.hartree,
                initial_potential,
                **kwargs,
            )
            implementation.statistics.hartree_solve_calls += 1
            implementation.statistics.hartree_total_seconds += (
                perf_counter() - started
            )
            return result

    else:
        raise RuntimeError(
            f"unhandled Hartree backend {selection.hartree_backend!r}"
        )

    native_boundary_description = (
        "orbit-summed C++/OpenMP multipoles and direct wedge RHS"
        if native_symmetry_boundary
        else "cached C++/OpenMP multipole boundary/RHS"
        if native_boundary_builder is not None
        else "Python boundary/RHS"
    )
    native_cg_description = (
        "symmetry-wedge C++/OpenMP CG"
        if hartree_reduction is not None
        else "cached C++/OpenMP CG"
    )
    hartree_implementation = {
        "scipy": "fast multipole boundary plus reference-equivalent SciPy CG",
        "native": f"{native_boundary_description} plus {native_cg_description}",
        "cupy": "fast multipole boundary plus shared-device-CSR CuPy CG",
    }[selection.hartree_backend]
    finite_difference_implementation = {
        "reference": "validated vectorized Python/SciPy compressed-grid CSR builder",
        "native": "C++17 compressed-grid CSR builder (exact stencil parity)",
    }[selection.finite_difference_builder]
    component_details = (
        (
            "cuda_initialization_overlap",
            str(preparation_overlap["cuda_initialization_overlap"]),
        ),
        (
            "backend_resolution_seconds",
            f"{preparation_overlap['backend_resolution_seconds']:.6f}",
        ),
        (
            "reference_preparation_seconds",
            f"{preparation_overlap['reference_preparation_seconds']:.6f}",
        ),
        (
            "backend_reference_overlapped_seconds",
            f"{preparation_overlap['backend_reference_overlapped_seconds']:.6f}",
        ),
        ("symmetry_mode", symmetry_mode),
        ("symmetry_detection", symmetry_detection),
        ("detected_symmetry_group_order", str(detected_group_order)),
        ("orbital_symmetry", orbital_symmetry),
        ("finite_difference_builder", finite_difference_implementation),
        (
            "ionic_setup_implementation",
            (
                "cached-grid C++/OpenMP radial interpolation and KB sampling"
                if selection.finite_difference_builder == "native"
                and "RadialGridEvaluator" in native_kernel_names
                else "vectorized NumPy radial interpolation and KB sampling"
            ),
        ),
        ("hartree_backend", selection.hartree_backend),
        ("hartree_implementation", hartree_implementation),
        (
            "xc_implementation",
            (
                "cached C++/OpenMP float64 CA/PZ-LDA"
                if xc_functional == "ca"
                and native_xc_evaluator is not None
                else (
                    "vectorized NumPy discrete-variational PBE"
                    if xc_functional == "pbe"
                    else "vectorized NumPy CA/PZ-LDA"
                )
            ),
        ),
    )
    from .Laplacian import DeferredNativeNegativeLaplacian

    deferred_laplacian = reference.negative_laplacian
    if isinstance(deferred_laplacian, DeferredNativeNegativeLaplacian):
        component_details += (
            (
                "finite_difference_full_grid_materialization",
                (
                    "performed"
                    if deferred_laplacian.materialized
                    else "skipped_by_exact_reduced_operator_cache"
                ),
            ),
            (
                "finite_difference_provenance_key",
                deferred_laplacian.cache_key,
            ),
            (
                "finite_difference_provenance_hash_seconds",
                f"{deferred_laplacian.hash_seconds:.6f}",
            ),
            (
                "finite_difference_nnz_count_seconds",
                f"{deferred_laplacian.nnz_count_seconds:.6f}",
            ),
            (
                "finite_difference_nnz_cache",
                deferred_laplacian.nnz_cache_status,
            ),
            (
                "finite_difference_nnz_cache_path",
                (
                    str(deferred_laplacian.nnz_cache_path)
                    if deferred_laplacian.nnz_cache_path is not None
                    else "disabled"
                ),
            ),
            (
                "finite_difference_materialization_seconds",
                f"{deferred_laplacian.materialization_seconds:.6f}",
            ),
            (
                "reference_static_cache",
                getattr(
                    deferred_laplacian,
                    "reference_static_cache_status",
                    "disabled",
                ),
            ),
            (
                "reference_static_cache_lookup_seconds",
                f"{getattr(deferred_laplacian, 'reference_static_cache_lookup_seconds', 0.0):.6f}",
            ),
        )
    if symmetry_geometry_cache_info is not None:
        cache_info = symmetry_geometry_cache_info
        component_details += (
            ("symmetry_geometry_cache", cache_info.status),
            ("symmetry_geometry_cache_key", cache_info.key),
            (
                "symmetry_geometry_cache_path",
                str(cache_info.path) if cache_info.path is not None else "disabled",
            ),
            ("symmetry_geometry_hash_seconds", f"{cache_info.hash_seconds:.6f}"),
            ("symmetry_geometry_cache_load_seconds", f"{cache_info.load_seconds:.6f}"),
            ("symmetry_geometry_build_seconds", f"{cache_info.build_seconds:.6f}"),
            ("symmetry_geometry_cache_write_seconds", f"{cache_info.write_seconds:.6f}"),
        )
    if symmetry_representation_cache_info is not None:
        cache_info = symmetry_representation_cache_info
        component_details += (
            ("symmetry_representation_cache", cache_info.status),
            ("symmetry_representation_cache_key", cache_info.key),
            (
                "symmetry_representation_cache_path",
                str(cache_info.path) if cache_info.path is not None else "disabled",
            ),
            ("symmetry_representation_hash_seconds", f"{cache_info.hash_seconds:.6f}"),
            ("symmetry_representation_cache_load_seconds", f"{cache_info.load_seconds:.6f}"),
            ("symmetry_representation_build_seconds", f"{cache_info.build_seconds:.6f}"),
            ("symmetry_representation_cache_write_seconds", f"{cache_info.write_seconds:.6f}"),
        )
    if selection.selected == "cupy":
        from .Eigensolvers.orthogonalize import (
            _complete_subspace_policy,
            chebdav_block_orth_requested,
        )
        from .Eigensolvers.rayleigh_ritz import generalized_ritz_requested

        reference_input = getattr(reference, "input", None)
        scf_input = getattr(reference_input, "scf", None)
        eigensolver_input = getattr(reference_input, "eigensolver", None)
        requested_states = getattr(scf_input, "number_of_states", None)
        subspace_buffer = getattr(eigensolver_input, "subspace_buffer", None)
        if (
            orbital_decomposition is not None
            or requested_states is None
            or subspace_buffer is None
        ):
            subspace_orthogonalization = (
                "size-adaptive per representation: audited PARSEC MGS for "
                "small bases; generalized Cholesky-whitened Rayleigh--Ritz "
                "with Householder QR fallback for large bases"
            )
        else:
            working_states = min(
                reference.grid.size,
                int(requested_states)
                + int(subspace_buffer),
            )
            selected_orthogonalization = _complete_subspace_policy(
                reference.grid.size,
                working_states,
            )
            if generalized_ritz_requested(
                reference.grid.size, working_states
            ):
                subspace_orthogonalization = (
                    "audited generalized Cholesky-whitened Rayleigh--Ritz "
                    f"selected for {reference.grid.size}x{working_states} "
                    "filtered basis; Householder QR stability fallback"
                )
            else:
                subspace_orthogonalization = (
                    f"{selected_orthogonalization} selected for "
                    f"{reference.grid.size}x{working_states} saved basis"
                )
        component_details += (
            (
                "gpu_initial_random_basis",
                "bit-exact 2048-lane skip-ahead DLARNV generator",
            ),
            (
                "gpu_chebdav_ritz_scalar_source",
                (
                    "reuse values already returned by host LAPACK"
                    if os.environ.get(
                        "PARSEC_CUPY_REUSE_HOST_RITZ_VALUES", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "explicit device scalar transfers"
                ),
            ),
            (
                "gpu_chebdav_appended_orthogonalization",
                (
                    "not used by the selected first eigensolver"
                    if eigensolver_input is None
                    or eigensolver_input.method != "chebdav"
                    else (
                        "audited FP64 block-CGS2/device-MGS2 with "
                        "Householder and PARSEC-MGS fallbacks"
                        if chebdav_block_orth_requested(
                            (
                                max(orbital_decomposition.sector_sizes)
                                if orbital_decomposition is not None
                                else reference.grid.size
                            ),
                            eigensolver_input.matvec_block_size,
                        )
                        else "PARSEC selective MGS"
                    )
                ),
            ),
            (
                "gpu_chebdav_prefix_projection",
                (
                    (
                        "full C-order coefficient GEMM plus fused "
                        "active-prefix CUDA update for blocks up to six"
                        if os.environ.get(
                            "PARSEC_CUPY_CHEBDAV_FUSED_PREFIX_UPDATE", "1"
                        ).strip().lower()
                        not in {"0", "false", "no", "off"}
                        else "full C-order workspace GEMM with zero "
                        "inactive coefficients"
                    )
                    if os.environ.get(
                        "PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_CGS", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "active noncontiguous prefix GEMM"
                ),
            ),
            (
                "gpu_chebdav_ritz_projection",
                (
                    "full contiguous C-order Davidson workspace GEMM; "
                    "use only the active row interval"
                    if os.environ.get(
                        "PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_RITZ", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "active noncontiguous Davidson basis GEMM"
                ),
            ),
            (
                "gpu_subspace_orthogonalization",
                subspace_orthogonalization,
            ),
        )
    if orbital_decomposition is not None:
        from .Symmetry import operator_build_workers

        orbital_reduction = orbital_decomposition.reduction
        component_details += (
            ("symmetry_full_grid_points", str(orbital_reduction.full_size)),
            ("symmetry_wedge_points", str(orbital_reduction.wedge_size)),
            (
                "symmetry_reduction_ratio",
                f"{orbital_reduction.reduction_ratio:.6g}",
            ),
        )
    if orbital_decomposition is not None:
        cache_info = symmetry_eigensolver.operator_cache_info
        if symmetry_eigensolver.scheduler_mode == "sequential":
            sector_scheduler = "sequential on one CUDA device"
        elif symmetry_eigensolver.scheduler_mode == "multi-gpu":
            device_list = ",".join(map(str, symmetry_eigensolver.device_ids))
            sector_scheduler = (
                "independent representations distributed across "
                f"CUDA devices {device_list}"
            )
        else:
            sector_scheduler = (
                "concurrent nonblocking CUDA streams "
                f"({symmetry_eigensolver.scheduler_workers} workers)"
            )
        component_details += (
            (
                "orbital_symmetry_representations",
                str(orbital_decomposition.representation_count),
            ),
            (
                "orbital_sector_dimensions",
                " ".join(map(str, orbital_decomposition.sector_sizes)),
            ),
            (
                "orbital_sector_stabilizer_handling",
                "exact character selection on every orbit stabilizer",
            ),
            (
                "orbital_sector_state_policy",
                "floor(global_states/representations) + Subspace_Buffer_Size, "
                "then grow sectors to bracket the global cutoff",
            ),
            ("orbital_sector_scheduler", sector_scheduler),
            (
                "orbital_sector_cuda_devices",
                " ".join(map(str, symmetry_eigensolver.device_ids)),
            ),
            (
                "orbital_sector_lanczos_scheduler",
                (
                    "concurrent nonblocking streams; filters remain sequential"
                    if symmetry_eigensolver.collective_lanczos
                    else "same scheduler as sector solves"
                ),
            ),
            (
                "orbital_sector_finite_difference_storage",
                symmetry_eigensolver.finite_difference_storage,
            ),
            (
                "orbital_sector_neighbor_storage",
                symmetry_eigensolver.finite_difference_neighbors,
            ),
            (
                "orbital_sector_nonlocal_application",
                (
                    (
                        "canonical-order custom CUDA B.T projection plus KB "
                        "scatter fused into CUDA stencil"
                        if symmetry_eigensolver.custom_projector_projection
                        else "cuSPARSE B.T projection plus KB scatter fused "
                        "into CUDA stencil"
                    )
                    if symmetry_eigensolver.fused_projector_scatter
                    else "two sparse KB contractions"
                ),
            ),
            (
                "orbital_sector_projector_reduction",
                symmetry_eigensolver.projector_reduction_modes,
            ),
            (
                "orbital_sector_later_filter_precision",
                (
                    "float32 stencil/projectors/recurrence; float64 Ritz and SCF"
                    if symmetry_eigensolver.mixed_precision_filter
                    else "float64"
                ),
            ),
            (
                "orbital_sector_local_potential_storage",
                symmetry_eigensolver.local_potential_storage,
            ),
            (
                "orbital_density_storage",
                "normalized wedge orbitals; physical scalar orbit values "
                "remain compact through SCF",
            ),
            ("orbital_operator_cache", cache_info.status),
            ("orbital_operator_cache_key", cache_info.key),
            (
                "orbital_operator_cache_path",
                str(cache_info.path) if cache_info.path is not None else "disabled",
            ),
            (
                "orbital_operator_hash_seconds",
                f"{cache_info.hash_seconds:.6f}",
            ),
            (
                "orbital_operator_cache_load_seconds",
                f"{cache_info.load_seconds:.6f}",
            ),
            (
                "orbital_operator_build_seconds",
                f"{cache_info.build_seconds:.6f}",
            ),
            (
                "orbital_operator_build_workers",
                str(
                    operator_build_workers(
                        orbital_decomposition.representation_count
                    )
                ),
            ),
            (
                "orbital_operator_cache_write_seconds",
                f"{cache_info.write_seconds:.6f}",
            ),
            (
                "scf_scalar_field_storage",
                "one physical value per symmetry orbit with multiplicity weights",
            ),
        )
    if selection.hartree_backend == "native":
        component_details += (
            ("hartree_boundary_setup", boundary_setup_status),
            (
                "hartree_boundary_setup_seconds",
                f"{boundary_setup_seconds:.6f}",
            ),
            (
                "hartree_boundary_setup_wait_seconds",
                f"{boundary_setup_wait_seconds:.6f}",
            ),
            (
                "hartree_boundary_setup_overlapped_seconds",
                f"{boundary_setup_overlapped_seconds:.6f}",
            ),
            ("hartree_cg_storage", poisson_solver.storage_mode),
            ("hartree_cg_openmp_workers", str(poisson_solver.worker_count)),
            (
                "hartree_initial_guess",
                (
                    "two-step chronological RHS predictor with "
                    "previous-potential fallback"
                    if os.environ.get(
                        "PARSEC_HARTREE_CHRONOLOGICAL_GUESS", "1"
                    ).strip().lower()
                    not in {"0", "false", "no", "off"}
                    else "previous SCF Hartree potential"
                ),
            ),
            (
                "hartree_cg_coefficient_palette_size",
                str(poisson_solver.coefficient_palette_size),
            ),
        )
        if hartree_reduction is None:
            component_details += (("hartree_symmetry", "full grid"),)
        else:
            component_details += (
                (
                    "hartree_symmetry",
                    "normalized Cartesian axis-reflection wedge",
                ),
                (
                    "hartree_symmetry_group_order",
                    str(hartree_reduction.group_order),
                ),
                ("hartree_full_grid_points", str(hartree_reduction.full_size)),
                ("hartree_wedge_points", str(hartree_reduction.wedge_size)),
                (
                    "hartree_reduction_ratio",
                    f"{hartree_reduction.reduction_ratio:.6g}",
                ),
            )
        if native_boundary_cache_info is not None:
            cache_info = native_boundary_cache_info
            component_details += (
                ("hartree_geometry_cache", cache_info.status),
                ("hartree_geometry_cache_key", cache_info.key),
                (
                    "hartree_geometry_cache_path",
                    str(cache_info.path)
                    if cache_info.path is not None
                    else "disabled",
                ),
                ("hartree_geometry_hash_seconds", f"{cache_info.hash_seconds:.6f}"),
                ("hartree_geometry_cache_load_seconds", f"{cache_info.load_seconds:.6f}"),
                ("hartree_geometry_build_seconds", f"{cache_info.build_seconds:.6f}"),
                ("hartree_geometry_cache_write_seconds", f"{cache_info.write_seconds:.6f}"),
            )
    if selection.selected != "native" and (
        selection.finite_difference_builder == "native"
        or selection.hartree_backend == "native"
    ):
        # A hybrid is reported as a CuPy execution backend, so copy the native
        # runtime configuration into its provenance explicitly.
        from .backends.native import native_build_info

        native_build = native_build_info()
        component_details += (
            (
                "native_openmp_detected_processors",
                str(native_build.get("openmp_detected_processors", 1)),
            ),
            (
                "native_openmp_reserved_threads",
                str(native_build.get("openmp_reserved_threads", 0)),
            ),
            (
                "native_openmp_max_threads",
                str(native_build.get("openmp_max_threads", 1)),
            ),
            (
                "native_openmp_thread_source",
                str(native_build.get("openmp_thread_source", "unknown")),
            ),
        )
    implementation.info = replace(
        implementation.info,
        details=implementation.info.details + component_details,
    )
    return AcceleratedPreparedSinglePointSystem(
        reference=reference,
        backend=implementation,
        backend_info=implementation.info,
        eigenproblem_solver=getattr(
            implementation, "eigenproblem_solver", None
        ),
        hartree_solver=accelerated_hartree,
        orbital_density_builder=getattr(
            implementation, "orbital_density_builder", None
        ),
        xc_evaluator=native_xc_evaluator,
        mixer_factory=(None if scf_reducer is None else scf_reducer.mixer),
        residual_metrics_evaluator=(
            None
            if scf_reducer is None
            else scf_reducer.potential_residual_metrics
        ),
        total_energy_evaluator=(
            None if scf_reducer is None else scf_reducer.total_energy
        ),
        scalar_field_adapter=scf_reducer,
    )


def profile_hamiltonian_components(
    system: AcceleratedPreparedSinglePointSystem,
    *,
    block_size: int | None = None,
    repeats: int = 1,
    random_seed: int = 19,
) -> dict[str, float]:
    """Benchmark the three matrix-free Hamiltonian actions explicitly.

    This opt-in diagnostic synchronizes between components, so it is kept out
    of the production Chebyshev recurrence.  The diagonal field is set to the
    local ionic potential, making the reported local action specifically the
    requested ``V_ion,local`` benchmark.  Initial CA-LDA construction is timed
    separately by the reference SCF timing model.
    """

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if block_size is None:
        block_size = system.input.eigensolver.matvec_block_size
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")

    previous = getattr(system.backend, "local_potential", None)
    previous = None if previous is None else np.asarray(previous).copy()
    system.backend.bind(system.ionic_potential)
    generator = np.random.default_rng(random_seed)
    vectors = generator.standard_normal((system.grid.size, block_size))

    # One unreported warm-up prevents import/allocation/JIT startup from being
    # confused with a physical component cost in the optional microprofile.
    system.backend.synchronize()
    started = perf_counter()
    system.backend.apply_kinetic(vectors)
    system.backend.synchronize()
    system.backend.statistics.warmup_seconds += perf_counter() - started

    totals: dict[str, float] = {}
    for _ in range(repeats):
        sample = system.backend.profile_components(vectors)
        for name, value in sample.items():
            totals[name] = totals.get(name, 0.0) + float(value)
    averages = {name: value / repeats for name, value in totals.items()}
    system.backend.statistics.component_profile_seconds = dict(averages)
    if previous is not None:
        system.backend.update_local(previous)
    return averages


def run_scf(
    system: AcceleratedPreparedSinglePointSystem,
    *,
    callback: Callable[[SCFIteration], None] | None = None,
) -> AcceleratedSinglePointResult:
    """Run the validated SCF loop using the selected execution backend."""

    return run_accelerated_scf(system, callback=callback)


def run_single_point(
    problem: SinglePointInput,
    *,
    backend: BackendName | str = "auto",
    symmetry: SymmetryMode | str = "auto",
    callback: Callable[[SCFIteration], None] | None = None,
) -> AcceleratedSinglePointResult:
    """Prepare and run one accelerated isolated single-point calculation."""

    system = prepare_single_point(
        problem, backend=backend, symmetry=symmetry
    )
    return run_scf(system, callback=callback)


__all__ = [
    "AcceleratedPreparedSinglePointSystem",
    "prepare_single_point",
    "profile_hamiltonian_components",
    "run_scf",
    "run_single_point",
]
