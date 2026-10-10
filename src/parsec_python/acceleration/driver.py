"""Readable public workflow for the additive accelerated architecture."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from importlib.util import find_spec
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from typing import Callable

import numpy as np

from parsec_python.SCF.single_point import (
    complete_single_point as complete_reference_single_point,
    prepare_single_point as prepare_reference_single_point,
)
from parsec_python.SCF.pbc import (
    prepare_periodic_single_point as prepare_reference_periodic_single_point,
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
from .Grid import build_cluster_grid_by_slabs, fast_grid_requested
from .Hartree.poisson import build_hartree_problem, solve_scipy_hartree
from .Symmetry.axis_reflection import _fast_maps_requested
from .SCF.symmetry_fields import scf_buffers_requested
# Imported here and not where it is read: the module binds ``require_cupy``
# when it is first imported, and a preparation may run under a double of it.
from .V_ion.cupy_ionic import ionic_gpu_count_setting
from .backends.implicit_stencil import pack_worker_setting
from .backends.scipy import ScipyHamiltonianBackend
from .backends.selection import BackendSelection, resolve_backend
from .models import (
    AcceleratedSinglePointResult,
    BackendName,
    BackendUnavailableError,
    SymmetryMode,
)


_REFERENCE_CACHE_FORMAT = 1
# GPU boundary builders, by class name: the import needs CuPy.
_GPU_BOUNDARY_BUILDERS = frozenset(
    {
        "CuPyMultipoleBoundaryBuilder",
        "CuPySymmetryMultipoleBoundaryBuilder",
        "CuPyPointMultipoleBoundaryBuilder",
    }
)
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
    digest.update(os.environ.get("PARSEC_IONIC_BACKEND", "native").lower().encode("ascii"))
    digest.update(os.environ.get("PARSEC_NATIVE_PROJECTOR_LOOKUP", "1").encode("ascii"))
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
        problem.periodic_cell,
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
    """Return fresh timing/descriptor metadata over immutable cached arrays.

    A deferred Laplacian gets a descriptor of its own for this calculation's
    report.  The full-grid matrix, once a calculation has had to build it, is
    shared with the cached descriptor and so with every later one.
    """

    negative_laplacian = reference.negative_laplacian
    from .Laplacian import DeferredNativeNegativeLaplacian

    if isinstance(negative_laplacian, DeferredNativeNegativeLaplacian):
        negative_laplacian = DeferredNativeNegativeLaplacian(
            negative_laplacian.grid,
            cache_directory=cache_directory,
            sharing_matrix_with=negative_laplacian,
        )
        negative_laplacian.reference_static_cache_status = "hit"
        negative_laplacian.reference_static_cache_lookup_seconds = lookup_seconds
    return replace(
        reference,
        negative_laplacian=negative_laplacian,
        timings=PreparationTimings(total_seconds=lookup_seconds),
    )


def _cupy_installed() -> bool:
    """Whether CuPy could be imported, decided without importing it."""

    if "cupy" in sys.modules:
        return True
    try:
        return find_spec("cupy") is not None
    except (ImportError, ValueError):
        return False


def _reference_with_matrix(reference):
    """Replace a deferred Laplacian by its matrix for a backend that reads it.

    Only the CuPy backend understands the deferred descriptor.  The build
    time joins the finite-difference stage the matrix was deferred from.
    """

    from .Laplacian import DeferredNativeNegativeLaplacian

    operator = getattr(reference, "negative_laplacian", None)
    if not isinstance(operator, DeferredNativeNegativeLaplacian):
        return reference
    built_before = operator.materialization_seconds
    matrix = operator.materialize()
    seconds = operator.materialization_seconds - built_before
    return replace(
        reference,
        negative_laplacian=matrix,
        timings=replace(
            reference.timings,
            finite_difference_seconds=(
                reference.timings.finite_difference_seconds + seconds
            ),
            total_seconds=reference.timings.total_seconds + seconds,
        ),
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


def _ionic_overlap_requested() -> bool:
    """Whether the root-only ionic fields may leave the preparing thread.

    ``PARSEC_OVERLAP_IONIC_SETUP`` is on by default; ``0`` keeps the stages
    in line.  It applies to the GPU ionic fields
    (``PARSEC_IONIC_BACKEND=cupy``), whose host thread only waits for two
    kernels.  The C++/OpenMP builders work on the cores the host stages use
    and stay in line; an overlap of them was not measured.
    """

    return os.environ.get(
        "PARSEC_OVERLAP_IONIC_SETUP", "1"
    ).strip().lower() not in {"0", "false", "no", "off"} and (
        os.environ.get("PARSEC_IONIC_BACKEND", "native").lower() == "cupy"
    )


class _OverlappedIonicFields:
    """The root-only ionic fields, built on a thread beside host set-up.

    Only the XC evaluator and the SCF loop read the local ionic potential,
    the two densities and the ionic energies.  The thread runs
    ``complete_single_point`` of the reference package on a system that was
    prepared without them, so each field is the one the in-line stages
    return.  It makes no MPI call and takes no cuBLAS or cuSOLVER handle:
    its end leaves an open graph capture valid (``backends/cupy_capture.py``).
    """

    def __init__(self) -> None:
        self._executor = None
        self._future = None
        self._sum_devices = None
        self.seconds = 0.0
        self.wait_seconds = 0.0
        # The devices the sums of the thread ran on, known at the join.
        self.devices: tuple[int, ...] = ()

    @property
    def started(self) -> bool:
        return self._future is not None

    def start(self, reference, *, sum_devices=None, **builders) -> None:
        """Begin the fields of ``reference``, a system prepared without them.

        ``sum_devices`` is called at the join for the devices the sums of
        ``builders`` ran on.
        """

        if self._future is not None:
            raise RuntimeError("the ionic fields of a preparation start once")
        self._sum_devices = sum_devices

        def build():
            started = perf_counter()
            complete = complete_reference_single_point(reference, **builders)
            return complete, perf_counter() - started

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="parsec-ionic-setup",
        )
        self._future = self._executor.submit(build)
        # Where a failure of the caller leaves the thread unjoined, it still
        # ends with its task and not when the executor is collected.
        self._executor.shutdown(wait=False)

    def result(self):
        """Join the thread and return the complete reference system.

        Its ``timings.total_seconds`` is still the wall time of the
        preparation that left the fields out.  The thread time and the wait
        here are in no stage of the calling thread; ``details`` reports them.
        """

        started = perf_counter()
        try:
            complete, self.seconds = self._future.result()
        finally:
            self._executor.shutdown(wait=True)
            self.wait_seconds = perf_counter() - started
        if self._sum_devices is not None:
            self.devices = tuple(self._sum_devices())
        return complete

    def details(self) -> tuple[tuple[str, str], ...]:
        """Report the thread time, the wait at the join and what was hidden."""

        if self._future is None:
            return (("ionic_setup", "inline"),)
        return (
            ("ionic_setup", "overlapped with symmetry and orbital setup"),
            ("ionic_setup_seconds", f"{self.seconds:.6f}"),
            ("ionic_setup_wait_seconds", f"{self.wait_seconds:.6f}"),
            (
                "ionic_setup_overlapped_seconds",
                f"{max(0.0, self.seconds - self.wait_seconds):.6f}",
            ),
        )


def _ionic_field_overlap(mpi_context) -> _OverlappedIonicFields | None:
    """Return the overlap of the ionic fields for a rank that may use one.

    The root rank of an MPI run and a process without one build the fields.
    Where they are GPU kernels and the overlap is requested, they run beside
    the host set-up that follows the reference preparation and are joined at
    their first reader.  ``None`` keeps them in line.
    """

    builds_fields = mpi_context is None or mpi_context.rank == mpi_context.root
    if builds_fields and _ionic_overlap_requested():
        return _OverlappedIonicFields()
    return None


def _prepare_reference_physics(
    problem: SinglePointInput,
    selection: BackendSelection,
    *,
    defer_native_laplacian: bool = False,
    deferred_laplacian_cache_directory: os.PathLike[str] | str | None = None,
    orbital_operators_only: bool = False,
    ionic_sector_rank: bool = False,
    ionic_report: dict | None = None,
    ionic_field_overlap: _OverlappedIonicFields | None = None,
):
    """Build the static physics with the builders of the selected backend.

    ``ionic_sector_rank`` tells the GPU ionic sums that this process is an
    MPI rank whose sector groups cover its devices (see
    :func:`V_ion.cupy_ionic.ionic_device_ids`).  ``ionic_report`` receives
    under ``devices`` the devices those sums ran on; it is left as it is
    where none ran here: another ionic backend, a rank that builds no ionic
    field, a reference taken from the resident cache, or fields left to
    ``ionic_field_overlap``.  The thread of the overlap sums after this call
    has returned, on the devices the process setting gives the calling
    thread, and the overlap holds the devices once it is joined.
    """

    cache_capacity = _resident_reference_cache_size()
    # A forced fresh model prediction must not be bypassed by the warmed
    # prepared-system cache.  The ordinary exact-key ML prediction cache still
    # makes repeat calculations inexpensive when ``regenerate`` is false.
    initial_settings = getattr(problem, "initial_density_settings", None)
    if initial_settings is not None and initial_settings.regenerate:
        cache_capacity = 0
    # A system without ionic fields and densities must not be stored under
    # the key of a complete one.
    if orbital_operators_only:
        cache_capacity = 0
    reduced_options = (
        {"orbital_operators_only": True} if orbital_operators_only else {}
    )
    # PARSEC_FAST_GRID: the same grid arrays without the cube of triples.
    if fast_grid_requested():
        reduced_options["grid_builder"] = build_cluster_grid_by_slabs
    # The static atomic tail of the Hartree boundary is built for the
    # Hartree solver of the selected backend, beside its boundary geometry.
    reduced_options["build_atomic_tail"] = False
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

    if getattr(problem, "periodic_cell", None) is not None:
        reference = prepare_reference_periodic_single_point(problem)
        if cache_key is not None:
            _remember_reference(cache_key, reference, cache_capacity)
        return reference

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
        gpu_ionic_fields = False
        if (ionic_builders is not None and selection.selected == "cupy"
                and os.environ.get("PARSEC_IONIC_BACKEND", "native").lower() == "cupy"):
            from .V_ion.cupy_ionic import CupyIonicBuilders

            ionic_builders = CupyIonicBuilders(sector_rank=ionic_sector_rank)
            gpu_ionic_fields = True

        if defer_native_laplacian:
            from .Laplacian import DeferredNativeNegativeLaplacian

            def laplacian_builder(grid):
                return DeferredNativeNegativeLaplacian(
                    grid,
                    cache_directory=deferred_laplacian_cache_directory,
                )
        else:
            laplacian_builder = build_native_negative_laplacian
        builder_options = {
            "negative_laplacian_builder": laplacian_builder,
            **reduced_options,
        }
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
        if (
            ionic_field_overlap is not None
            and gpu_ionic_fields
            and not orbital_operators_only
            # A resident process stores the complete system under its key.
            and not cache_capacity
        ):
            reference = prepare_reference_single_point(
                problem, **builder_options, orbital_operators_only=True
            )
            # A thread starts with device 0 current, whatever the caller's.
            ionic_builders.keep_current_device()
            ionic_field_overlap.start(
                reference,
                sum_devices=lambda: ionic_builders.device_ids,
                local_ionic_builder=builder_options["local_ionic_builder"],
                atomic_density_builder=builder_options["atomic_density_builder"],
                build_atomic_tail=False,
            )
            return reference
        reference = prepare_reference_single_point(problem, **builder_options)
        ionic_devices = getattr(ionic_builders, "device_ids", ())
        if ionic_report is not None and ionic_devices:
            ionic_report["devices"] = tuple(ionic_devices)
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
    reference = prepare_reference_single_point(problem, **reduced_options)
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


def _root_rank_only(component: str):
    """Return a hook that refuses to run ``component`` on a sector worker.

    A rank that only serves symmetry sectors prepares no Hartree or XC
    objects.  Failing with the reason is preferable to an attribute error on
    a missing field if the SCF loop is ever entered there.
    """

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            f"{component} exists on the MPI root rank only; this rank "
            "prepared symmetry-sector operators"
        )

    return unavailable


def _hartree_device_request() -> int | str | None:
    """Parse ``PARSEC_HARTREE_DEVICE``.

    ``auto``, the default, or a CuPy device index of this process puts the
    GPU Hartree objects together on one device.  ``off`` returns ``None`` and
    leaves them on their default devices: the CG on the first sector device,
    the boundary geometry on the device current in the thread that builds it.
    """

    raw = os.environ.get("PARSEC_HARTREE_DEVICE", "auto").strip().lower()
    if raw in {"", "auto"}:
        return "auto"
    if raw == "off":
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_HARTREE_DEVICE must be auto, off or a CuPy device index"
        ) from error
    if value < 0:
        raise ValueError("PARSEC_HARTREE_DEVICE cannot be negative")
    return value


def _planned_sector_device_groups(
    representation_count: int,
    device_ids: tuple[int, ...],
    mpi_context,
) -> dict[int, tuple[int, ...]]:
    """Return the device group of every sector this process will own.

    This is the assignment ``CuPySymmetrySCFEigensolver`` makes, computed
    without it: the boundary geometry is uploaded while that constructor
    still runs.  An MPI rank gets the groups of its sector context; one
    process spreads all sectors over its devices in turn.
    """

    if mpi_context is not None:
        from .experimental.mpi_scf import sector_device_groups

        return sector_device_groups(
            representation_count,
            mpi_context.size,
            mpi_context.rank,
            device_ids,
        )
    return {
        index: (device_ids[index % len(device_ids)],)
        for index in range(representation_count)
    }


def _select_hartree_device(
    request: int | str | None,
    device_ids: tuple[int, ...],
    sector_device_groups: dict[int, tuple[int, ...]],
) -> int | None:
    """Choose the CUDA device of the Poisson CG, boundary and resident maps.

    The first device of a process also holds the owner's share of a sector,
    so with these objects it sets the memory peak.  (The sector index and
    expansion maps of the eigensolver do not add to it: they stay on the
    host.)  ``auto`` takes the device with the smallest share of sector
    bases, then one that owns no sector, then the last one listed.  With
    sector groups that is never the first device of a process that has
    several.  ``None`` leaves every object on its default device.
    """

    if request is None:
        return None
    if request != "auto":
        if request not in device_ids:
            raise ValueError(
                "PARSEC_HARTREE_DEVICE is not one of the CUDA devices of "
                "this process"
            )
        return request
    if len(device_ids) < 2:
        return None
    share: dict[int, float] = {}
    owners = set()
    for group in sector_device_groups.values():
        owners.add(group[0])
        for device in group:
            share[device] = share.get(device, 0.0) + 1.0 / len(group)
    return min(
        reversed(device_ids),
        key=lambda device: (share.get(device, 0.0), device in owners),
    )


def _reported_cache_key(key: str | None) -> str:
    """Return the provenance text for a content key that may be absent.

    A cache that is switched off does not hash its key, so the report states
    that instead of a digest.
    """

    return "not computed (cache disabled)" if key is None else key


def _reported_pack_workers(storage: str) -> str:
    """Return the threads that packed the tiles of sector stencils stored as ``storage``.

    ``PARSEC_CUPY_IMPLICIT_PACK_WORKERS`` as the packer reads it.  ``mixed``
    storage holds tiles or none: a setting that is no thread count says
    none, since an operator that is to pack tiles refuses it.
    """

    if "tile" not in storage and storage != "mixed":
        return "no tiles packed"
    try:
        return str(pack_worker_setting())
    except ValueError:
        return "no tiles packed"


def _apply_hartree_boundary_switches(problem):
    """Apply the environment switches of the Hartree boundary to the input.

    ``PARSEC_HARTREE_BOUNDARY=legacy`` restores PARSEC's boundary values:
    no tolerance and no atomic tail.  ``PARSEC_HARTREE_BOUNDARY_TOLERANCE``
    (a number in Ry, or ``off``) and ``PARSEC_HARTREE_ATOMIC_TAIL``
    (``auto``, ``on`` or ``off``) replace the input keywords
    ``Hartree_Boundary_Tolerance`` and ``Hartree_Atomic_Tail``, and
    ``PARSEC_HARTREE_LPOLE`` replaces ``Solver_Lpole``.  They act before the
    reference is prepared, so the plan of the boundary and the key of the
    resident reference cache see the settings that run.
    """

    mode = os.environ.get("PARSEC_HARTREE_BOUNDARY", "auto").strip().lower()
    if mode not in {"auto", "legacy"}:
        raise ValueError("PARSEC_HARTREE_BOUNDARY must be auto or legacy")
    changes: dict[str, object] = {}
    tolerance = os.environ.get("PARSEC_HARTREE_BOUNDARY_TOLERANCE", "").strip().lower()
    if tolerance in {"off", "none"}:
        changes["boundary_tolerance"] = None
    elif tolerance:
        try:
            changes["boundary_tolerance"] = float(tolerance)
        except ValueError as error:
            raise ValueError(
                "PARSEC_HARTREE_BOUNDARY_TOLERANCE must be a number in Ry or off"
            ) from error
    tail = os.environ.get("PARSEC_HARTREE_ATOMIC_TAIL", "").strip().lower()
    if tail:
        if tail not in {"auto", "on", "off"}:
            raise ValueError("PARSEC_HARTREE_ATOMIC_TAIL must be auto, on, or off")
        changes["atomic_tail"] = tail
    if mode == "legacy":
        if (
            changes.get("boundary_tolerance") is not None
            or changes.get("atomic_tail") == "on"
        ):
            raise ValueError(
                "PARSEC_HARTREE_BOUNDARY=legacy excludes a boundary tolerance "
                "and PARSEC_HARTREE_ATOMIC_TAIL=on"
            )
        changes = {"boundary_tolerance": None, "atomic_tail": "off"}
    order = os.environ.get("PARSEC_HARTREE_LPOLE", "").strip()
    if order:
        try:
            changes["multipole_order"] = int(order)
        except ValueError as error:
            raise ValueError("PARSEC_HARTREE_LPOLE must be an integer") from error
        # The switch is the Solver_Lpole of this run, also for an input an
        # earlier plan had resolved.
        changes["minimum_multipole_order"] = None
    if not changes:
        return problem
    return replace(problem, hartree=replace(problem.hartree, **changes))


def _hartree_boundary_tail(reference, *, on_device: bool, device_id: int | None):
    """Build the static atomic tail the boundary plan of ``reference`` has.

    Returns ``None`` for a plan without one.  ``on_device`` evaluates the
    tail values with the kernel of :mod:`.Hartree.cupy_atomic_tail` on
    ``device_id`` (``None``: the current device of the calling thread);
    otherwise host threads do.  The tail records which, for the report.
    """

    plan = getattr(reference, "hartree_boundary", None)
    if plan is None or not plan.atomic_tail:
        return None
    built = getattr(reference, "hartree_boundary_tail", None)
    if built is not None:
        return built
    from parsec_python.Hartree import valence_point_charges

    from .Hartree.atomic_tail import build_atomic_tail_fast

    tail_values = None
    if on_device:
        from functools import partial

        from .Hartree.cupy_atomic_tail import device_tail_values

        tail_values = partial(device_tail_values, device_id=device_id)
    tail = build_atomic_tail_fast(
        reference.grid,
        *valence_point_charges(
            reference.atoms, reference.pseudopotentials, reference.electron_count
        ),
        plan.order,
        tail_values=tail_values,
    )
    return replace(tail, values_from="device kernel") if on_device else tail


def _attached_boundary_tail(builder):
    """Return the atomic tail a boundary builder adds, ``None`` without one."""

    from parsec_python.Hartree import AtomicTail

    tail = getattr(builder, "boundary_tail", None)
    return tail if isinstance(tail, AtomicTail) else None


def _hartree_boundary_kernel() -> str:
    """Return the requested kernels of the GPU boundary builder.

    ``PARSEC_HARTREE_BOUNDARY_KERNEL`` is ``auto``, ``full`` or ``wedge``.
    Where the boundary plan is engaged and the boundary is built on a GPU,
    ``auto`` takes the wedge kernels if an axis-reflection group reduces
    the Hartree problem and the same kernels under the identity alone, one
    value per unique exterior point of the full grid, if none does.
    Elsewhere, and with ``full``, the full-grid kernels of PARSEC's boundary
    run.
    """

    kernel = os.environ.get("PARSEC_HARTREE_BOUNDARY_KERNEL", "auto").strip().lower()
    if kernel not in {"auto", "full", "wedge"}:
        raise ValueError("PARSEC_HARTREE_BOUNDARY_KERNEL must be auto, full, or wedge")
    return kernel


def _hartree_tail_on_host() -> bool:
    """Whether ``PARSEC_HARTREE_ATOMIC_TAIL_VALUES`` asks for host threads.

    ``auto``, the default, takes the values of the static atomic tail from
    the device kernel wherever the orbital backend is CuPy and from host
    threads elsewhere.  ``host`` takes the host threads there too: the same
    Coulomb sum and series in NumPy, which differ from the kernel's at
    round-off.  The serial control of the MPI runner asks for them, so that
    the kernel is not in both sides of a comparison.
    """

    values = os.environ.get("PARSEC_HARTREE_ATOMIC_TAIL_VALUES", "auto").strip().lower()
    if values not in {"auto", "host"}:
        raise ValueError("PARSEC_HARTREE_ATOMIC_TAIL_VALUES must be auto or host")
    return values == "host"


def _hartree_boundary_check_count() -> int:
    """Return the number of points of ``PARSEC_HARTREE_BOUNDARY_CHECK``, 0 for off."""

    raw = os.environ.get("PARSEC_HARTREE_BOUNDARY_CHECK", "0").strip()
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError(
            "PARSEC_HARTREE_BOUNDARY_CHECK must be a number of points"
        ) from error
    if count < 0:
        raise ValueError("PARSEC_HARTREE_BOUNDARY_CHECK cannot be negative")
    return count


def _hartree_boundary_check(system, result) -> tuple[tuple[str, str], ...]:
    """Measure the boundary values of the final density against the direct sum.

    With ``PARSEC_HARTREE_BOUNDARY_CHECK=n`` and a GPU boundary builder, the
    boundary values at ``n`` unique exterior points are compared with the
    direct Coulomb sum of the final density over the whole grid, on the
    Hartree device.  The points are a sample: the innermost ones, those of
    the largest atomic tail, those nearest to an atom and random ones in
    equal shares, so ``hartree_boundary_check_max`` is the maximum over
    that sample; ``n`` at or above the number of points takes them all.
    ``hartree_boundary_check_legacy_max`` is the same error of the
    expansion at ``Solver_Lpole`` without a tail, which is PARSEC's.
    """

    count = _hartree_boundary_check_count()
    if not count:
        return ()
    builder = getattr(system.backend, "native_boundary_builder", None)
    kind = type(builder).__name__
    if kind not in _GPU_BOUNDARY_BUILDERS:
        return (
            (
                "hartree_boundary_check",
                "not run: the boundary values are not built on a GPU",
            ),
        )
    started = perf_counter()
    reference = system.reference
    plan = getattr(reference, "hartree_boundary", None)
    minimum_order = None if plan is None else plan.minimum_order
    density = np.ascontiguousarray(result.density, dtype=np.float64)
    from parsec_python.Hartree import valence_point_charges

    positions, charges = valence_point_charges(
        reference.atoms, reference.pseudopotentials, reference.electron_count
    )
    if kind == "CuPySymmetryMultipoleBoundaryBuilder":
        check = builder.boundary_check(
            density[builder.reduction.representative_rows],
            count,
            minimum_order=minimum_order,
            positions=positions,
        )
    elif kind == "CuPyPointMultipoleBoundaryBuilder":
        check = builder.boundary_check(
            density, count, minimum_order=minimum_order, positions=positions
        )
    else:
        check = builder.boundary_check(
            density,
            count,
            minimum_order=minimum_order,
            tail_charges=(
                (positions, charges)
                if plan is not None and plan.atomic_tail
                else None
            ),
            positions=positions,
        )
    error = check["boundary"] - check["direct"]
    legacy = check["legacy"] - check["direct"]
    total = int(check["total"])
    where = (
        "unique exterior points of the symmetry wedge"
        if kind == "CuPySymmetryMultipoleBoundaryBuilder"
        else "unique exterior points"
    )
    return (
        ("hartree_boundary_check_points", str(error.size)),
        (
            "hartree_boundary_check_sample",
            f"all {total} {where}"
            if error.size == total
            else f"{error.size} of {total} {where}: the innermost, those of "
            "the largest atomic tail, those nearest to an atom and random "
            "ones in equal shares",
        ),
        ("hartree_boundary_check_max", f"{np.abs(error).max():.6e} Ry"),
        ("hartree_boundary_check_rms", f"{np.sqrt(np.mean(error * error)):.6e} Ry"),
        ("hartree_boundary_check_legacy_max", f"{np.abs(legacy).max():.6e} Ry"),
        ("hartree_boundary_check_seconds", f"{perf_counter() - started:.6f}"),
    )


def hartree_boundary_check_seconds(result) -> float:
    """Return the seconds the check after the SCF took, 0 where none ran.

    :func:`run_scf` makes the check before it returns, so a caller that
    times the SCF around that call subtracts these seconds.
    """

    details = dict(getattr(getattr(result, "backend", None), "details", ()))
    return float(details.get("hartree_boundary_check_seconds", 0.0))


def _hartree_boundary_details(
    reference, tail, builder=None
) -> tuple[tuple[str, str], ...]:
    """Report the boundary plan of ``reference`` and the tail that was built.

    ``tail`` is the tail in rows; a wedge GPU ``builder`` holds its own, per
    exterior point, and reports its maximum and seconds.  Either one says
    what evaluated its values (``PARSEC_HARTREE_ATOMIC_TAIL_VALUES``): the
    entry is what was built, not what the switch asked for.
    """

    plan = getattr(reference, "hartree_boundary", None)
    if plan is None:
        return ()
    maximum = getattr(tail, "maximum", None)
    seconds = getattr(tail, "seconds", None)
    evaluator = getattr(tail, "values_from", None)
    if maximum is None:
        maximum = getattr(builder, "atomic_tail_maximum", None)
        seconds = getattr(builder, "atomic_tail_seconds", None)
        evaluator = getattr(builder, "atomic_tail_values", None)
    details = (
        (
            "hartree_boundary",
            "PARSEC multipole expansion"
            if plan.legacy
            else "multipole expansion with a-priori control "
            "(PARSEC_HARTREE_BOUNDARY=legacy restores PARSEC's)",
        ),
        (
            "hartree_multipole_order",
            f"{plan.order} (Solver_Lpole {plan.minimum_order})",
        ),
        (
            "hartree_boundary_tolerance",
            "off" if plan.tolerance is None else f"{plan.tolerance:.6e} Ry",
        ),
        (
            "hartree_boundary_estimate",
            f"none ({plan.status})"
            if plan.estimates is None
            else f"{plan.estimate_minimum:.6e} Ry at order {plan.minimum_order}, "
            f"{plan.estimate_order:.6e} Ry at order {plan.order} ({plan.status})",
        ),
        ("hartree_atomic_tail", "applied" if plan.atomic_tail else "not applied"),
        # The estimate runs in the first step of the preparation of the
        # rank, before the grid is built.
        ("hartree_boundary_plan_seconds", f"{plan.seconds:.6f}"),
    )
    if plan.cap_reached:
        details += (
            ("hartree_boundary_cap", "tolerance missed at the largest order"),
        )
    if isinstance(maximum, float) and isinstance(seconds, float):
        details += (
            ("hartree_atomic_tail_max", f"{maximum:.6e} Ry"),
            ("hartree_atomic_tail_seconds", f"{seconds:.6f}"),
        )
    if isinstance(evaluator, str):
        details += (("hartree_atomic_tail_values", evaluator),)
    return details


def _build_native_boundary_builder(
    reference,
    hartree_reduction,
    *,
    symmetry_cache_directory: os.PathLike[str] | str | None,
    symmetry_geometry_cache_info,
    device_id: int | None = None,
    tail_on_device: bool = False,
    boundary_tail=None,
    gpu_kernels: bool = False,
):
    """Construct the reusable isolated-boundary geometry.

    This setup depends only on the already-built real-space grid and exact
    symmetry metadata.  It is deliberately kept separate from the Poisson
    solver so the hybrid path can prepare it on a CPU worker while the main
    thread uploads/builds the independent GPU orbital operators.

    ``device_id`` places a GPU builder on that CUDA device; ``None`` keeps
    the current device of the constructing thread.

    Where the boundary plan of ``reference`` has an atomic tail, it is built
    here too, from the atoms and the grid alone, and attached to the
    builder: ``boundary_tail`` if the caller has it already, else a new one
    whose values come from that device if ``tail_on_device`` and from host
    threads otherwise, or wherever ``PARSEC_HARTREE_ATOMIC_TAIL_VALUES=host``
    asks for them.

    ``gpu_kernels`` says that the orbital backend runs on a GPU.  A native
    builder has no orbit table without a reduction, or where the table of a
    raised order exceeds its storage limit, and then repeats the full-grid
    recurrences in every solve.  An engaged plan takes the GPU kernels
    there instead, unless ``PARSEC_HARTREE_BOUNDARY_BACKEND=native`` or
    ``PARSEC_HARTREE_BOUNDARY_KERNEL=full`` asks for the former route by
    name.
    """

    from .Hartree.native_boundary import (
        NativeMultipoleBoundaryBuilder,
        NativeSymmetryMultipoleBoundaryBuilder,
    )

    started = perf_counter()
    tail_on_host = _hartree_tail_on_host()

    def with_tail(builder):
        tail = boundary_tail
        if tail is None:
            tail = _hartree_boundary_tail(
                reference,
                on_device=tail_on_device and not tail_on_host,
                device_id=device_id,
            )
        if tail is not None:
            builder.set_boundary_tail(tail)
        return builder

    kernel = _hartree_boundary_kernel()
    requested = os.environ.get("PARSEC_HARTREE_BOUNDARY_BACKEND", "").strip().lower()
    if requested not in {"", "native", "cupy", "auto"}:
        raise ValueError("PARSEC_HARTREE_BOUNDARY_BACKEND must be native, cupy, or auto")
    boundary_backend = requested or "native"
    plan = getattr(reference, "hartree_boundary", None)
    engaged = plan is not None and plan.engaged
    order = reference.input.hartree.multipole_order
    if boundary_backend == "auto":
        from .Hartree.cupy_boundary import auto_gpu_boundary_requested

        wedge_rows = reference.grid.size if hartree_reduction is None else hartree_reduction.wedge_size
        boundary_backend = "cupy" if (
            os.environ.get("PARSEC_HARTREE_LINEAR_BACKEND", "native").strip().lower() == "cupy"
            and auto_gpu_boundary_requested(wedge_rows, order)
        ) else "native"
    if kernel == "wedge" and boundary_backend != "cupy":
        raise ValueError(
            "PARSEC_HARTREE_BOUNDARY_KERNEL=wedge needs the GPU boundary "
            "(PARSEC_HARTREE_BOUNDARY_BACKEND=cupy)"
        )

    def gpu_builder():
        """Return a GPU builder and whether it works on the symmetry wedge."""

        from .Hartree.cupy_boundary import (
            CuPyMultipoleBoundaryBuilder,
            CuPyPointMultipoleBoundaryBuilder,
            CuPySymmetryMultipoleBoundaryBuilder,
            wedge_kernels_supported,
        )

        wedge = wedge_kernels_supported(hartree_reduction)
        if kernel == "wedge" and not wedge:
            raise ValueError(
                "PARSEC_HARTREE_BOUNDARY_KERNEL=wedge needs an axis-reflection "
                "reduction of the Hartree problem"
            )
        if not (kernel == "wedge" or (kernel == "auto" and engaged)):
            builder = CuPyMultipoleBoundaryBuilder(
                reference.grid, order, device_id=device_id
            )
            return with_tail(builder), False
        if wedge:
            builder = CuPySymmetryMultipoleBoundaryBuilder(
                reference.grid, hartree_reduction, order, device_id=device_id
            )
        else:
            # No group of axis reflections: the same kernels under the
            # identity alone, with the full-grid interface.
            builder = CuPyPointMultipoleBoundaryBuilder(
                reference.grid, order, device_id=device_id
            )
        if plan is not None and plan.atomic_tail:
            from parsec_python.Hartree import valence_point_charges

            # Evaluated per unique exterior point, on the device of the
            # builder unless host threads are asked for.
            host_values = {}
            if tail_on_host:
                from .Hartree.atomic_tail import host_tail_values

                host_values["tail_values"] = host_tail_values
            builder.set_atomic_tail(
                *valence_point_charges(
                    reference.atoms,
                    reference.pseudopotentials,
                    reference.electron_count,
                ),
                **host_values,
            )
        return builder, wedge

    if boundary_backend == "cupy":
        builder, symmetry_boundary = gpu_builder()
        return builder, symmetry_boundary, None, perf_counter() - started
    builder = None
    symmetry_boundary = False
    cache_info = None
    if hartree_reduction is not None:
        try:
            builder = NativeSymmetryMultipoleBoundaryBuilder(
                reference.grid,
                hartree_reduction,
                order,
                cache_directory=symmetry_cache_directory,
                cache_key_seed=(
                    None
                    if symmetry_geometry_cache_info is None
                    else symmetry_geometry_cache_info.key
                ),
            )
        except RuntimeError:
            # No orbit table: it exceeds its storage limit, or a previously
            # installed 0.3 extension has none.  The full grid remains.
            pass
        else:
            symmetry_boundary = True
            cache_info = getattr(builder, "cache_info", None)
    if (
        builder is None
        and gpu_kernels
        and engaged
        and requested != "native"
        and kernel == "auto"
    ):
        try:
            builder, symmetry_boundary = gpu_builder()
        except (RuntimeError, MemoryError):
            # An extension without the geometry exporter of the GPU builders,
            # or a device without the room: the route that was not asked
            # for must not fail where the native one runs.
            builder = None
        else:
            return builder, symmetry_boundary, None, perf_counter() - started
    if builder is None:
        builder = NativeMultipoleBoundaryBuilder(reference.grid, order)
    return (
        with_tail(builder),
        symmetry_boundary,
        cache_info,
        perf_counter() - started,
    )


def _selection_for_problem(
    selection: BackendSelection, problem: SinglePointInput
) -> BackendSelection:
    """Route a periodic problem's static construction to the reference builder.

    The native C++ finite-difference/ionic construction kernels and the
    native/CuPy multipole-boundary Hartree solve are both isolated-cluster
    specific -- neither is periodic-wraparound aware. Forcing
    ``finite_difference_builder``/``hartree_backend`` to the reference/scipy
    values here means every downstream branch keyed on those two fields
    (native boundary setup, native XC, the diagnostic implementation
    strings) is automatically inert for a periodic problem without needing
    its own ``periodic_cell`` check.

    ``selected`` -- the Hamiltonian-apply execution backend used for the
    eigensolver's repeated matrix-vector products -- is deliberately left
    alone: ``negative_laplacian``/``nonlocal_operator`` are plain sparse
    matrices regardless of periodicity (see the comment in
    :func:`_build_backend`), so CuPy/native can still accelerate that step
    for a periodic problem. Only the Hartree *solve* itself still needs an
    explicit ``periodic_cell`` branch in :func:`prepare_single_point`, since
    forcing ``hartree_backend="scipy"`` would otherwise route into the
    isolated-only scipy Hartree path rather than skipping it.
    """

    if getattr(problem, "periodic_cell", None) is None:
        return selection
    if (
        selection.finite_difference_builder == "reference"
        and selection.hartree_backend == "scipy"
    ):
        return selection
    return replace(
        selection,
        finite_difference_builder="reference",
        hartree_backend="scipy",
        fallback_reasons=selection.fallback_reasons
        + (
            "periodic cell: finite-difference/ionic construction and the "
            "Hartree solve are routed to the reference (Python) "
            "implementation -- native/CuPy construction and boundary "
            "kernels are isolated-cluster only",
        ),
    )


def _resolve_and_prepare_reference(
    problem: SinglePointInput,
    backend: BackendName | str,
    *,
    defer_native_laplacian: bool = False,
    deferred_laplacian_cache_directory: os.PathLike[str] | str | None = None,
    orbital_operators_only: bool = False,
    ionic_sector_rank: bool = False,
    ionic_field_overlap: _OverlappedIonicFields | None = None,
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

    ``defer_native_laplacian`` asks for the deferred descriptor in place of
    the full-grid matrix.  Only the CuPy backend can use it, so another
    selected backend gets the matrix, as without the request.

    ``ionic_sector_rank`` is passed on to the GPU ionic sums, which exclude
    the overlap.  The devices they ran on are returned under
    ``ionic_sum_devices``, ``None`` if none ran in this call: sums left to
    ``ionic_field_overlap`` run after it, and their devices are read where
    the overlap is joined.
    """

    normalized = str(backend).strip().lower()
    gpu_ionic = os.environ.get("PARSEC_IONIC_BACKEND", "native").lower() == "cupy"
    ionic_report: dict = {}

    def prepare_reference(selection, defer):
        options = {}
        if defer:
            options.update(
                defer_native_laplacian=True,
                deferred_laplacian_cache_directory=(
                    deferred_laplacian_cache_directory
                ),
            )
        if orbital_operators_only:
            options.update(orbital_operators_only=True)
        if gpu_ionic:
            options.update(
                ionic_sector_rank=ionic_sector_rank, ionic_report=ionic_report
            )
        if ionic_field_overlap is not None:
            options.update(ionic_field_overlap=ionic_field_overlap)
        return _prepare_reference_physics(problem, selection, **options)

    overlap_requested = os.environ.get(
        "PARSEC_OVERLAP_CUDA_INITIALIZATION", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    # GPU setup must wait for authoritative device selection; the speculative
    # native selection used by the CPU overlap path cannot select it safely.
    can_overlap = overlap_requested and normalized == "auto" and not gpu_ionic
    if not can_overlap:
        resolution_started = perf_counter()
        selection = _selection_for_problem(resolve_backend(backend, problem), problem)
        resolution_seconds = perf_counter() - resolution_started
        reference_started = perf_counter()
        reference = prepare_reference(
            selection,
            defer_native_laplacian and selection.selected == "cupy",
        )
        reference_seconds = perf_counter() - reference_started
        return selection, reference, {
            "cuda_initialization_overlap": "disabled",
            "backend_resolution_seconds": resolution_seconds,
            "reference_preparation_seconds": reference_seconds,
            "backend_reference_overlapped_seconds": 0.0,
            "ionic_sum_devices": ionic_report.get("devices"),
        }

    # The auto static builder depends only on native availability, not on the
    # CUDA probe running in the worker.  This inexpensive check may be
    # repeated by authoritative resolution after CUDA discovery.
    from .backends.selection import _native_status

    native_available, _ = _native_status()
    provisional = _selection_for_problem(
        BackendSelection(
            requested="auto",
            selected="native" if native_available else "scipy",
            finite_difference_builder=(
                "native" if native_available else "reference"
            ),
            hartree_backend="native" if native_available else "scipy",
        ),
        problem,
    )

    # The probe decides whether CuPy executes.  Defer for it only where CuPy
    # is installed at all, so that a host without it prepares the matrix
    # inside the overlap exactly as before.
    defer_provisionally = defer_native_laplacian and _cupy_installed()

    overlap_started = perf_counter()

    def timed_resolution():
        started = perf_counter()
        selected = _selection_for_problem(resolve_backend(backend, problem), problem)
        return selected, perf_counter() - started

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="parsec-cuda-init",
    ) as executor:
        future = executor.submit(timed_resolution)
        reference_started = perf_counter()
        reference = prepare_reference(provisional, defer_provisionally)
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
        reference = prepare_reference(
            selection,
            defer_native_laplacian and selection.selected == "cupy",
        )
        reference_seconds += perf_counter() - reference_started
        overlapped_seconds = 0.0
        overlap_status = "rebuild_after_backend_change"
    else:
        overlapped_seconds = max(
            0.0,
            resolution_seconds + reference_seconds - overlap_wall_seconds,
        )
        overlap_status = "cuda_probe_with_cpu_reference_setup"
        if selection.selected != "cupy":
            # CuPy is installed but was not selected.
            reference_started = perf_counter()
            reference = _reference_with_matrix(reference)
            reference_seconds += perf_counter() - reference_started

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
    mpi_context=None,
) -> AcceleratedPreparedSinglePointSystem:
    """Prepare physics, automatically reducing exact symmetry when usable.

    ``auto`` (the default) uses every exactly detected Cartesian reflection
    that the selected backend supports and otherwise keeps the full-grid
    algorithm.  ``on`` turns an unusable/nontrivial symmetry into an error;
    ``off`` skips detection and is the reproducible full-grid comparison path.

    ``symmetry_cache_directory`` names a persistent symmetry cache.  Without
    one, the default here and of the command-line entry points, nothing is
    read or written, which is the fastest first calculation of a structure;
    with one, a process loads or builds the stencils of all sectors.

    With an ``mpi_context`` every rank prepares the symmetry sectors it owns.
    Only the root rank also prepares what the SCF loop itself needs (ionic
    fields, densities, Hartree and XC); on the other ranks those hooks raise.
    """

    is_periodic = getattr(problem, "periodic_cell", None) is not None
    # A periodic cell has no Hartree boundary values.  The switches are read
    # for it too, so that a value they do not know is refused in every run,
    # and leave its input as it is.
    boundary_switched = _apply_hartree_boundary_switches(problem)
    if not is_periodic:
        problem = boundary_switched
    # Read here so that a value they do not know is refused in every run.
    _hartree_boundary_kernel()
    _hartree_boundary_check_count()
    tail_on_host = _hartree_tail_on_host()
    ionic_fields = _ionic_field_overlap(mpi_context)
    symmetry_mode = _normalize_symmetry_mode(symmetry)
    if is_periodic and symmetry_mode == "on":
        raise ValueError(
            "symmetry='on' is not supported for periodic problems: exact "
            "space-group symmetry detection is not implemented for the "
            "accelerated periodic path"
        )
    if is_periodic and mpi_context is not None:
        raise ValueError(
            "an MPI sector context is not supported for periodic problems: "
            "the ranks of an MPI run divide the symmetry sectors of an "
            "isolated cluster, and the accelerated periodic path uses the "
            "full grid"
        )
    # Only the root rank of an MPI run enters the SCF loop.  The other ranks
    # answer eigensolver and density commands for the sectors they own.
    sector_worker = (
        mpi_context is not None and mpi_context.rank != mpi_context.root
    )
    from .Symmetry.sector_stencil import sector_stencil_route

    # The symmetry sectors take their packed stencils from the grid, or from
    # the cache, so the full-grid matrix is built only for a consumer that
    # reads it.  ``PARSEC_SECTOR_STENCIL=csr`` keeps the former route, which
    # builds the matrix first unless a cache may make it unnecessary.  The
    # switch is read here whatever the symmetry mode, so that a value it does
    # not know is refused in every run.
    stencil_route = sector_stencil_route()
    from .Eigensolvers.filter_graph import graph_reuse_requested, graph_route_name

    # Read here for the same reason: the first filter that records a graph
    # comes after the whole preparation, on every rank.
    shared_filter_graphs = graph_reuse_requested()
    selection, reference, preparation_overlap = _resolve_and_prepare_reference(
        problem,
        backend,
        defer_native_laplacian=(
            symmetry_mode != "off"
            and (symmetry_cache_directory is not None or stencil_route != "csr")
        ),
        deferred_laplacian_cache_directory=symmetry_cache_directory,
        orbital_operators_only=sector_worker,
        # Every device of an MPI rank belongs to one of its sector groups.
        ionic_sector_rank=mpi_context is not None,
        **({} if ionic_fields is None else {"ionic_field_overlap": ionic_fields}),
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
    if is_periodic:
        symmetry_detection = (
            "skipped: periodic cell (space-group symmetry detection is not "
            "implemented for the accelerated path; the full grid is used)"
        )
    elif symmetry_mode != "off":
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
    hartree_linear_backend = os.environ.get(
        "PARSEC_HARTREE_LINEAR_BACKEND", "native"
    ).strip().lower()
    if hartree_linear_backend not in {"native", "cupy"}:
        raise ValueError("PARSEC_HARTREE_LINEAR_BACKEND must be native or cupy")
    if is_periodic:
        # The switch chooses the CG of the isolated-cluster Hartree solver.
        # A periodic cell is solved by the reference periodic CG below.
        hartree_linear_backend = "native"
    if hartree_linear_backend == "cupy" and not (
        selection.selected == "cupy" and selection.hartree_backend == "native"
    ):
        raise ValueError(
            "PARSEC_HARTREE_LINEAR_BACKEND=cupy requires the auto/hybrid CUDA "
            "backend with native boundary construction"
        )
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
    # The Poisson CG arrays, the GPU boundary geometry and the resident
    # Hartree maps exchange device arrays and therefore share one device.  It
    # is fixed here because the boundary worker starts before the sector
    # eigensolver exists.  ``None`` leaves every object on its default device.
    # A sector worker builds none of them and has nothing to place.
    hartree_device = None
    hartree_device_request = _hartree_device_request()
    # ``auto`` asks for a device only where one of the three is built on a
    # GPU: the CG, or the boundary geometry, whose own ``auto`` follows the
    # CG.  An index is checked against the devices of the process even so.
    hartree_on_gpu = hartree_linear_backend == "cupy" or (
        os.environ.get("PARSEC_HARTREE_BOUNDARY_BACKEND", "native").strip().lower()
        == "cupy"
    )
    if (
        hartree_device_request is not None
        and (hartree_device_request != "auto" or hartree_on_gpu)
        and selection.selected == "cupy"
        and not sector_worker
    ):
        from .Eigensolvers.symmetry import selected_device_ids
        from .backends.cupy import require_cupy

        cp, _ = require_cupy()
        current_device = int(cp.cuda.Device().id)
        process_devices = selected_device_ids(current_device)
        hartree_device = _select_hartree_device(
            hartree_device_request,
            process_devices,
            (
                # One full-grid operator on the current device.
                {0: (current_device,)}
                if orbital_decomposition is None
                else _planned_sector_device_groups(
                    orbital_decomposition.representation_count,
                    process_devices,
                    mpi_context,
                )
            ),
        )
    if (
        boundary_setup_eligible
        and boundary_overlap_enabled
        and selection.selected == "cupy"
        and not sector_worker
    ):
        # The thread uploads the geometry, compiles and launches kernels,
        # the one of the atomic tail among them, and synchronizes its own
        # stream.  It synchronizes no device and takes no cuBLAS or cuSOLVER
        # handle, so the graph capture of the Poisson solver, which may be
        # open on the same device before the join, stays valid while the
        # thread works and when it ends (``backends/cupy_capture.py``).
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
            device_id=hartree_device,
            tail_on_device=True,
            gpu_kernels=True,
        )
        boundary_setup_status = "overlapped with GPU orbital setup"

    if mpi_context is not None and orbital_decomposition is None:
        raise ValueError("MPI SCF requires a CuPy orbital symmetry decomposition")

    # The full-grid GPU operator serves only the pure-CuPy Poisson solver,
    # which a sector worker does not build.
    defer_full_gpu_operator = orbital_decomposition is not None and (
        selection.hartree_backend != "cupy" or sector_worker
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
    hartree_boundary_tail = None
    resident_hartree = None
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
            # The key of a deferred Laplacian if a cache directory made it
            # hash one.  Without it the reduced-operator cache asks the
            # descriptor itself, and only when it looks a key up.
            kinetic_cache_key=getattr(
                reference.negative_laplacian, "hashed_cache_key", None
            ),
            decomposition_cache_key=(
                None
                if symmetry_representation_cache_info is None
                else symmetry_representation_cache_info.key
            ),
            **({"mpi_context": mpi_context} if mpi_context is not None else {}),
        )
        implementation.symmetry_eigensolver = symmetry_eigensolver
        implementation.eigenproblem_solver = symmetry_eigensolver
        # With the Hartree objects a device is the fullest of the process: a
        # sector that shares its basis gives it no work beside its own.
        symmetry_eigensolver.hartree_device = hartree_device
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

    if ionic_fields is not None and ionic_fields.started:
        # The XC evaluator below is the first reader of an ionic field, and
        # the Poisson solver after it captures a graph: the thread has ended
        # before either.
        reference = ionic_fields.result()
        if ionic_fields.devices:
            preparation_overlap["ionic_sum_devices"] = ionic_fields.devices

    if sector_worker:
        # The frozen core density was not built on this rank.
        native_xc_evaluator = _root_rank_only("the XC evaluator")
    elif (
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

    if scf_reducer is not None and native_xc_evaluator is None:
        # The readable XC evaluators accept complete Cartesian arrays.  PBE
        # specifically needs that layout because its density gradient couples
        # neighboring symmetry orbits; CA-LDA is pointwise but uses the same
        # public array contract.  Expand the invariant density for this narrow
        # operation, then retain one value per orbit again for the rest of the
        # accelerated SCF.  This is an exact representation adapter, not a
        # different functional or quadrature.
        from parsec_python.V_xc import XCResult

        def symmetry_xc_evaluator(density):
            evaluated = reference.evaluate_xc(scf_reducer.to_full(density))
            return XCResult(
                potential=scf_reducer.from_full(evaluated.potential),
                energy_per_electron=scf_reducer.from_full(
                    evaluated.energy_per_electron
                ),
                energy_density=scf_reducer.from_full(evaluated.energy_density),
                total_energy=evaluated.total_energy,
            )

        native_xc_evaluator = symmetry_xc_evaluator

    if sector_worker:
        # No Poisson solver, boundary geometry or resident Hartree maps:
        # the root rank solves Hartree and sends the effective potential.
        accelerated_hartree = _root_rank_only("the Hartree solver")

    elif is_periodic:
        # solve_scipy_hartree/CuPyPoissonSolver/NativePoissonSolver all build
        # an isolated-cluster boundary (multipole or direct Coulomb); none
        # apply to a periodic cell. Delegate straight to the already-
        # validated periodic CG solve (SCF.pbc.solve_periodic_hartree via
        # reference.solve_hartree) -- _selection_for_problem forces
        # hartree_backend to "scipy" for periodic precisely so none of the
        # branches below this one can be reached instead.
        def accelerated_hartree(density, initial_potential=None, **kwargs):
            return reference.solve_hartree(density, initial_potential, **kwargs)

    elif selection.hartree_backend == "cupy":
        from .Hartree.cupy_poisson import CuPyPoissonSolver

        if selection.selected != "cupy":
            raise RuntimeError("CuPy Hartree requires the CuPy execution backend")
        poisson_solver = CuPyPoissonSolver(implementation.device_operator)
        implementation.poisson_solver = poisson_solver
        hartree_boundary_tail = _hartree_boundary_tail(
            reference, on_device=not tail_on_host, device_id=None
        )

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            # ``CuPyPoissonSolver`` owns the full-grid finite-difference
            # operator and boundary builder.  The symmetry-aware nonlinear
            # SCF path, however, keeps scalar fields as one physical value per
            # orbit.  Expand only across this full-grid solver boundary, then
            # immediately compact the returned invariant fields again.  This
            # is an exact representation conversion; no density, boundary
            # condition, or Poisson tolerance is changed.
            full_density = (
                density
                if scf_reducer is None
                else scf_reducer.to_full(density)
            )
            full_initial = (
                initial_potential
                if scf_reducer is None or initial_potential is None
                else scf_reducer.to_full(initial_potential)
            )
            result = poisson_solver.solve(
                full_density,
                reference.grid,
                reference.input.hartree,
                full_initial,
                boundary_tail=hartree_boundary_tail,
                **kwargs,
            )
            if scf_reducer is None:
                return result
            return replace(
                result,
                potential=scf_reducer.from_full(result.potential),
                right_hand_side=scf_reducer.from_full(
                    result.right_hand_side
                ),
            )

    elif selection.hartree_backend == "native":
        from .Hartree.native_poisson import NativePoissonSolver
        from .Hartree.symmetry_poisson import SymmetryReducedPoissonSolver
        from .Laplacian import (
            DeferredNativeNegativeLaplacian,
            materialize_negative_laplacian,
        )

        poisson_factory = NativePoissonSolver
        if hartree_linear_backend == "cupy":
            from functools import partial
            from .Hartree.cupy_prepared import CuPyPreparedPoissonSolver

            poisson_factory = partial(
                CuPyPreparedPoissonSolver,
                device_id=(
                    hartree_device
                    if hartree_device is not None
                    else symmetry_eigensolver.device_ids[0]
                    if symmetry_eigensolver is not None else None
                ),
            )

        def concrete_negative_laplacian(operator):
            return (
                materialize_negative_laplacian(operator)
                if isinstance(operator, DeferredNativeNegativeLaplacian)
                else operator
            )

        if hartree_reduction is None:
            poisson_solver = poisson_factory(
                concrete_negative_laplacian(reference.negative_laplacian)
            )
        else:
            try:
                reusable_reduced_operator = (
                    symmetry_eigensolver is not None
                    and orbital_reduction is hartree_reduction
                )
                if not reusable_reduced_operator:
                    reduced_operator = concrete_negative_laplacian(
                        reference.negative_laplacian
                    )
                elif hartree_linear_backend == "cupy":
                    # The GPU CG reads the slot-major stencil the totally
                    # symmetric sector was packed into.  Converting it to CSR
                    # and packing that again would put the same neighbor and
                    # the same coefficient into every slot.
                    reduced_operator = (
                        symmetry_eigensolver.totally_symmetric_stencil
                    )
                else:
                    # The native CG copies CSR buffers, formed here on demand.
                    reduced_operator = (
                        symmetry_eigensolver.totally_symmetric_negative_laplacian
                    )
                poisson_solver = SymmetryReducedPoissonSolver(
                    reduced_operator,
                    hartree_reduction,
                    operator_is_reduced=reusable_reduced_operator,
                    solver_factory=poisson_factory,
                )
            except (ValueError, RuntimeError):
                if (symmetry_mode == "on" or legacy_hartree_forced
                        or hartree_linear_backend == "cupy"):
                    raise
                hartree_reduction = None
                poisson_solver = poisson_factory(
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
                        device_id=hartree_device,
                        # The tail in rows of the discarded builder, or a
                        # new one from the evaluator that builder was given.
                        boundary_tail=_attached_boundary_tail(
                            native_boundary_builder
                        ),
                        tail_on_device=selection.selected == "cupy",
                        gpu_kernels=selection.selected == "cupy",
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
                    device_id=hartree_device,
                    tail_on_device=selection.selected == "cupy",
                    gpu_kernels=selection.selected == "cupy",
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
        hartree_boundary_tail = _attached_boundary_tail(native_boundary_builder)

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
                    hartree_boundary_tail,
                )
            else:
                # The size-adaptive full-grid C++ multipole builder can be
                # selected even when the nonlinear SCF stores an invariant
                # scalar field on the wedge.  Expand physical point values at
                # this narrow boundary only; the multipole/RHS equations and
                # all later reduced-Poisson algebra are unchanged.
                if scf_reducer is not None:
                    from .SCF.symmetry_fields import SymmetryScalarField

                    if isinstance(boundary_density, SymmetryScalarField):
                        boundary_density = scf_reducer.to_full(
                            boundary_density
                        )
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
                if scf_reducer is not None and hartree_reduction is not None:
                    from dataclasses import replace as dataclass_replace
                    from .SCF.symmetry_fields import SymmetryScalarField

                    if not isinstance(
                        native_result.potential, SymmetryScalarField
                    ):
                        native_result = dataclass_replace(
                            native_result,
                            potential=scf_reducer.from_full(
                                native_result.potential
                            ),
                            right_hand_side=scf_reducer.from_full(
                                native_result.right_hand_side
                            ),
                        )
            solve_seconds = perf_counter() - solve_started
            implementation.statistics.hartree_solve_calls += 1
            implementation.statistics.hartree_rhs_seconds += rhs_seconds
            implementation.statistics.hartree_linear_solve_seconds += solve_seconds
            implementation.statistics.hartree_total_seconds += (
                perf_counter() - total_started
            )
            result = native_result.as_hartree_result(boundary)
            if hartree_boundary_tail is not None:
                result = replace(result, boundary_tail=hartree_boundary_tail)
            return result

        resident_policy = os.environ.get("PARSEC_CUPY_RESIDENT_HARTREE", "0").strip().lower()
        if resident_policy not in {"0", "1", "auto"}:
            raise ValueError("PARSEC_CUPY_RESIDENT_HARTREE must be 0, 1, or auto")
        resident_eligible = (
            type(native_boundary_builder).__name__ in _GPU_BOUNDARY_BUILDERS
            and hartree_linear_backend == "cupy"
            and hartree_reduction is not None and scf_reducer is not None
            and scf_reducer.reduction is hartree_reduction
        )
        if resident_policy == "1" and not resident_eligible:
            raise ValueError("resident Hartree requires GPU boundary/CG and matching scalar/Poisson wedge")
        if resident_policy in {"1", "auto"} and resident_eligible:
            from .Hartree.cupy_resident import CuPyResidentHartree
            resident_hartree = CuPyResidentHartree(native_boundary_builder,hartree_reduction,
                poisson_solver.solver.backend,reference.input.hartree)
            def accelerated_hartree(density,initial_potential=None,**kwargs):
                started=perf_counter()
                result=resident_hartree.solve(density,initial_potential,**kwargs)
                implementation.statistics.hartree_solve_calls+=1
                implementation.statistics.hartree_rhs_seconds+=resident_hartree.rhs_seconds
                implementation.statistics.hartree_linear_solve_seconds+=resident_hartree.solve_seconds
                implementation.statistics.hartree_total_seconds+=perf_counter()-started
                return result

    elif selection.hartree_backend == "scipy":
        hartree_boundary_tail = _hartree_boundary_tail(
            reference, on_device=False, device_id=None
        )

        def accelerated_hartree(density, initial_potential=None, **kwargs):
            started = perf_counter()
            result = solve_scipy_hartree(
                density,
                reference.grid,
                reference.negative_laplacian,
                reference.input.hartree,
                initial_potential,
                boundary_tail=hartree_boundary_tail,
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

    boundary_on_gpu = type(native_boundary_builder).__name__ in _GPU_BOUNDARY_BUILDERS
    boundary_on_wedge_gpu = (
        type(native_boundary_builder).__name__ == "CuPySymmetryMultipoleBoundaryBuilder"
    )
    boundary_on_point_gpu = (
        type(native_boundary_builder).__name__ == "CuPyPointMultipoleBoundaryBuilder"
    )
    native_boundary_description = (
        "FP64 GPU multipoles on the symmetry wedge, boundary values per "
        "unique exterior point and gathered wedge RHS"
        if boundary_on_wedge_gpu
        else
        "FP64 GPU multipoles of the full grid, boundary values per unique "
        "exterior point and gathered RHS"
        if boundary_on_point_gpu
        else
        "FP64 GPU streaming multipoles and exterior-stencil RHS (bounded O(N) storage)"
        if boundary_on_gpu
        else
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
    if hartree_linear_backend == "cupy":
        native_cg_description = (
            "FP64 CUDA graph CG on the normalized symmetry wedge"
            if hartree_reduction is not None
            else "FP64 CUDA graph CG on the full grid"
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
    if is_periodic:
        hartree_implementation = (
            "periodic Ewald-consistent conjugate-gradient solve "
            "(SCF.pbc.solve_periodic_hartree); no accelerated periodic "
            "Hartree backend yet"
        )
        finite_difference_implementation = (
            "validated vectorized Python/SciPy periodic-wraparound CSR "
            "builder (SCF.pbc.prepare_periodic_single_point)"
        )
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
        (
            # PARSEC_SYMMETRY_FAST_MAPS, where symmetry maps were built.
            "symmetry_fast_maps",
            "not used"
            if symmetry_mode == "off" or is_periodic
            else "on" if _fast_maps_requested() else "off",
        ),
        (
            # The directory the caller named.  Without one, the default of
            # every entry point, no symmetry cache is read or written.
            "symmetry_cache_directory",
            "disabled"
            if symmetry_cache_directory is None
            else "not used"
            if symmetry_mode == "off" or is_periodic
            else str(Path(symmetry_cache_directory).resolve()),
        ),
        ("orbital_symmetry", orbital_symmetry),
        # PARSEC_FAST_GRID, as the reference preparation read it.  The grid
        # of a periodic cell has the reference builder alone.
        (
            "grid_builder",
            "slabs" if fast_grid_requested() and not is_periodic else "reference",
        ),
        ("finite_difference_builder", finite_difference_implementation),
        (
            "ionic_setup_implementation",
            (
                ("FP64 CUDA atom-ordered local/density sums; C++ KB sampling"
                 if selection.selected == "cupy"
                 and os.environ.get("PARSEC_IONIC_BACKEND", "native").lower() == "cupy"
                 else "cached-grid C++/OpenMP radial interpolation and KB sampling")
                if selection.finite_difference_builder == "native"
                and "RadialGridEvaluator" in native_kernel_names
                else "vectorized NumPy radial interpolation and KB sampling"
            ),
        ),
        ("ionic_gpu_count_requested", ionic_gpu_count_setting()),
        (
            # The devices the GPU ionic sums of this preparation ran on.
            "ionic_gpu_devices",
            " ".join(map(str, preparation_overlap.get("ionic_sum_devices") or ()))
            or "none",
        ),
        ("ionic_projector_lookup", os.environ.get("PARSEC_NATIVE_PROJECTOR_LOOKUP", "1")),
        # A sector worker builds no ionic fields.
        *(
            ()
            if sector_worker
            else (("ionic_setup", "inline"),)
            if ionic_fields is None
            else ionic_fields.details()
        ),
        ("hartree_backend", "cupy-boundary/cupy-linear"
         if boundary_on_gpu and hartree_linear_backend == "cupy"
         else "native-boundary/cupy-linear" if hartree_linear_backend == "cupy"
         else "cupy-boundary/native-linear" if boundary_on_gpu
         else selection.hartree_backend),
        ("hartree_implementation", hartree_implementation),
        ("hartree_full_grid_transfer_policy", "device-resident" if resident_hartree is not None else "host-interface"),
        ("hartree_resident_predictor", resident_hartree.predictor if resident_hartree is not None else "disabled"),
        *_hartree_boundary_details(
            reference, hartree_boundary_tail, native_boundary_builder
        ),
        (
            # The kernels that build the boundary values of every solve.
            "hartree_boundary_kernel",
            "none (no Hartree solver on this rank)"
            if sector_worker
            else "none (periodic cell: no boundary values)"
            if is_periodic
            else "wedge (GPU, symmetry wedge and unique exterior points)"
            if boundary_on_wedge_gpu
            else "points (GPU, full grid and unique exterior points)"
            if boundary_on_point_gpu
            else "full (GPU, full grid)"
            if boundary_on_gpu
            else "native wedge (C++ orbit table)"
            if native_symmetry_boundary
            else "native full grid (C++)"
            if native_boundary_builder is not None
            else "python",
        ),
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
        if deferred_laplacian.matrix_origin == "built":
            materialization = "performed"
        elif deferred_laplacian.matrix_origin == "shared":
            # An earlier calculation of this resident process built it.
            materialization = "reused_from_resident_reference"
        elif (
            symmetry_eigensolver is not None
            and symmetry_eigensolver.operator_cache_info.stencil_builder
            != "cached"
        ):
            materialization = "skipped_by_direct_sector_stencil"
        else:
            materialization = "skipped_by_exact_reduced_operator_cache"
        component_details += (
            (
                "finite_difference_full_grid_materialization",
                materialization,
            ),
            (
                "finite_difference_provenance_key",
                _reported_cache_key(deferred_laplacian.hashed_cache_key),
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
            ("symmetry_geometry_cache_key", _reported_cache_key(cache_info.key)),
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
            (
                "symmetry_representation_cache_key",
                _reported_cache_key(cache_info.key),
            ),
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
                (
                    "bit-exact device DLARNV for initial CHEBFF basis"
                    if os.environ.get("PARSEC_CUPY_DEVICE_RANDOM", "0").lower()
                    in {"1", "true", "on"}
                    else "bit-exact host DLARNV before device upload"
                ),
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
                    "serial bound preparation on owning streams before sector workers"
                    if symmetry_eigensolver.bound_schedule == "before_sectors"
                    else "concurrent nonblocking streams; filters remain sequential"
                    if symmetry_eigensolver.collective_lanczos
                    else "same scheduler as sector solves"
                ),
            ),
            (
                "orbital_sector_finite_difference_storage",
                symmetry_eigensolver.finite_difference_storage,
            ),
            (
                "orbital_sector_tile_pack_workers",
                _reported_pack_workers(
                    symmetry_eigensolver.finite_difference_storage
                ),
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
                symmetry_eigensolver.later_filter_precision,
            ),
            (
                # PARSEC_CUPY_FILTER_GRAPH_REUSE, for a filter that records
                # graphs: none has run yet.  A result says what the filters
                # of its sectors recorded (SCF/single_point.py).
                "orbital_sector_filter_graphs",
                graph_route_name(shared_filter_graphs),
            ),
            (
                "orbital_sector_local_potential_storage",
                symmetry_eigensolver.local_potential_storage,
            ),
            (
                "orbital_density_storage",
                "representation-sector Ritz views; density accumulated "
                "directly on physical scalar orbits without packed "
                "wedge-by-state duplication",
            ),
            ("orbital_operator_cache", cache_info.status),
            ("orbital_operator_cache_key", _reported_cache_key(cache_info.key)),
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
            ("orbital_operator_stencil_builder", cache_info.stencil_builder),
            (
                "orbital_operator_build_workers",
                str(
                    operator_build_workers(
                        symmetry_eigensolver.assembled_sector_count
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
            (
                # PARSEC_SYMMETRY_SCF_BUFFERS as it is set now; the wedge sums
                # and the mixer read it again at every call.
                "scf_scalar_field_buffers",
                "on" if scf_buffers_requested() else "off",
            ),
        )
    if sector_worker:
        # The component entries above name the selected implementations;
        # this rank built only the ones its sectors use.
        component_details += (
            (
                "mpi_rank_preparation",
                "symmetry-sector operators only; ionic fields, densities, "
                "Hartree and XC are prepared on the root rank",
            ),
        )
    elif selection.hartree_backend == "native":
        if hartree_linear_backend == "cupy":
            gpu_poisson = (poisson_solver.solver if hartree_reduction is not None
                           else poisson_solver).backend
            component_details += (
                ("hartree_cg_device", str(gpu_poisson.device_id)),
                ("hartree_cg_graph_steps", str(gpu_poisson.graph_iterations)),
                ("hartree_cg_device_bytes", str(gpu_poisson.device_bytes)),
                ("hartree_cg_stencil_packing", gpu_poisson.stencil_packing),
            )
        component_details += (
            ("hartree_boundary_device_bytes", str(
                native_boundary_builder.device_storage_bytes if boundary_on_gpu else 0)),
            ("hartree_boundary_device", str(
                native_boundary_builder.device_id) if boundary_on_gpu else "host"),
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
                (
                    "hartree_geometry_cache_key",
                    _reported_cache_key(cache_info.key),
                ),
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
        boundary_tail_maximum=(
            hartree_boundary_tail.maximum
            if hartree_boundary_tail is not None
            else native_boundary_builder.atomic_tail_maximum
            if boundary_on_wedge_gpu or boundary_on_point_gpu
            else None
        ),
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

    result = run_accelerated_scf(system, callback=callback)
    boundary_check = _hartree_boundary_check(system, result)
    if boundary_check:
        result.backend = replace(
            result.backend, details=result.backend.details + boundary_check
        )
    return result


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
