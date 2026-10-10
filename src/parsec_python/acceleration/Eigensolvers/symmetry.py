"""Stateful CuPy eigensolver over real reflection representations.

PARSEC assigns an independent eigenproblem to every Abelian representation,
then globally sorts their Ritz values for occupations.  This adapter applies
the same decomposition while reusing the existing CuPy CHEBFF/CHEBDAV and
later-SUBSPACE implementations for each reduced Hamiltonian.

Each representation is assigned round-robin to the CUDA devices selected by
``PARSEC_CUPY_DEVICES``.  Independent sectors run concurrently across devices
and are gathered in fixed PARSEC representation order before the stable global
eigenvalue sort.  A single GPU remains serialized by default because each
sparse/filter kernel already saturates it; ``PARSEC_CUPY_SECTOR_SCHEDULER``
can explicitly request stream overlap without changing the mathematics.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.V_ion import NonlocalProjectorOperator

from ..Symmetry import ReflectionRepresentationDecomposition
from ..Symmetry.operator_cache import load_or_build_reduced_operators
from ..backends.cupy import (
    CuPyHamiltonian,
    CuPyTimingStats,
    cupy_device_count,
    require_cupy,
    synchronize,
)
from ..backends.cupy_mixed_precision import float32_filter_requested
from ..backends.implicit_stencil import implicit_tile_for_sector
from .distributed_state import shared_basis_devices, shared_block_bytes
from .eigval import CuPyEigvalResult, CuPyEigvalSolver, spill_download_order
from .filter_graph import graph_route_name
from .rayleigh_ritz import (
    gram_multiple,
    streaming_ritz_requested,
    streaming_workspace_bytes,
)


@dataclass(frozen=True)
class CuPySymmetryEigvalState:
    """Per-representation device states retained across SCF iterations."""

    requested_states: int
    sector_state_counts: tuple[int, ...]
    sector_states: tuple[object, ...]
    solves_completed: int

    def release_sector_snapshots(self) -> None:
        """Release consumed aliases while sector solvers own restart state.

        Otherwise this previous global result pins already-replaced orbital
        matrices until every device finishes the successor solve. Metadata
        remains valid; the successor returns a fresh sector-state snapshot.
        """
        object.__setattr__(self, "sector_states", (None,) * len(self.sector_states))


@dataclass(frozen=True)
class CuPySymmetryEigvalResult:
    """Global lowest eigenpairs expanded from their representation wedges."""

    eigenvalues: np.ndarray
    vectors: Any
    residual_norms: np.ndarray | None
    state: CuPySymmetryEigvalState
    solver_path: str
    restarted: bool
    restart_reason: str | None
    representations: np.ndarray
    representation_columns: np.ndarray


@dataclass(frozen=True)
class CuPySymmetryOrbitals:
    """Selected representation vectors retained on the normalized wedge.

    ``scaled_wedge_vectors`` has one row per scalar-field orbit and already
    includes ``1/sqrt(|O_w|)`` on the orbits admitted by each selected
    representation.  Rejected stabilizer orbits are exactly zero.  Its
    squared rows are therefore the physical density on every point in an
    orbit; phases are needed only when materializing signed full-grid states.

    The SCF loop never reads the two full-grid expansion maps, so they may
    stay on the host: ``phases`` is then the host character table and
    ``device_full_to_wedge`` is ``None``.  :meth:`to_full_device` uploads what
    it needs when it is called.  Device arrays are accepted as before.
    """

    scaled_wedge_vectors: Any | None
    representations: np.ndarray
    full_to_wedge: np.ndarray
    device_full_to_wedge: Any
    phases: Any
    full_size: int
    representation_columns: np.ndarray | None = None
    sector_vectors: tuple[Any, ...] | None = None
    sector_orbits: tuple[np.ndarray | None, ...] | None = None
    sector_scales: tuple[np.ndarray | None, ...] | None = None
    wedge_size: int | None = None
    remote_sectors: bool = False

    def to_full_host(self, block_states: int = 8) -> np.ndarray:
        """Export orbitals in bounded host blocks, without a primary-GPU gather.

        The requested final host matrix is necessarily full-sized, but neither
        the full-grid matrix nor the union of all sector states is allocated
        on any GPU. This also supports systems larger than one device's VRAM.
        """
        if block_states < 1:
            raise ValueError("block_states must be positive")
        if self.remote_sectors:
            raise RuntimeError("MPI symmetry orbitals require distributed export; disable final materialization")
        if any(hasattr(item, "group") for item in self.sector_vectors or ()):
            raise RuntimeError("a sector basis shared by several devices cannot be exported as one host array here")
        cp, _ = require_cupy()
        full = np.empty(self.shape, dtype=np.float64, order="F")
        for representation in range(int(self.phases.shape[0])):
            output_columns = np.flatnonzero(self.representations == representation)
            if not output_columns.size:
                continue
            # A phase table kept on the host passes through unchanged.
            phases = cp.asnumpy(self.phases[representation])
            for start in range(0, output_columns.size, block_states):
                columns = output_columns[start : start + block_states]
                if self.scaled_wedge_vectors is not None:
                    source = self.scaled_wedge_vectors
                    with source.device:
                        wedge = cp.asnumpy(source[:, columns])
                else:
                    if (self.sector_vectors is None or self.sector_orbits is None
                            or self.sector_scales is None or self.wedge_size is None
                            or self.representation_columns is None):
                        raise RuntimeError("lazy symmetry orbitals are incomplete")
                    source = self.sector_vectors[representation]
                    source_columns = self.representation_columns[columns]
                    if isinstance(source, cp.ndarray):
                        with source.device:
                            selected = cp.asnumpy(source[:, source_columns])
                    else:
                        selected = np.asarray(source[:, source_columns])
                    wedge = np.zeros((self.wedge_size, columns.size), dtype=np.float64)
                    wedge[self.sector_orbits[representation], :] = (
                        selected * self.sector_scales[representation][:, None]
                    )
                full[:, columns] = wedge[self.full_to_wedge, :] * phases[:, None]
        return full

    @property
    def shape(self) -> tuple[int, int]:
        if self.scaled_wedge_vectors is not None:
            count = int(self.scaled_wedge_vectors.shape[1])
        else:
            count = int(self.representations.size)
        return self.full_size, count

    @property
    def ndim(self) -> int:
        return 2

    def release_intermediate_storage(self) -> None:
        """Drop the preceding iteration's sector references before its successor."""

        if self.scaled_wedge_vectors is None:
            object.__setattr__(self, "sector_vectors", None)

    def to_full_device(self):
        """Expand signed orbitals once, preserving global eigenvalue order."""

        if self.remote_sectors:
            raise RuntimeError("MPI symmetry orbitals cannot be materialized on one device")
        if any(hasattr(item, "group") for item in self.sector_vectors or ()):
            raise RuntimeError("a sector basis shared by several devices cannot be materialized on one device")
        cp, _ = require_cupy()
        wedge = self.scaled_wedge_vectors
        if wedge is None:
            if (
                self.representation_columns is None
                or self.sector_vectors is None
                or self.sector_orbits is None
                or self.sector_scales is None
                or self.wedge_size is None
            ):
                raise RuntimeError("lazy symmetry orbitals are incomplete")
            wedge = cp.zeros(
                (self.wedge_size, self.representations.size),
                dtype=cp.float64,
                order="F",
            )
            for representation, source in enumerate(self.sector_vectors):
                output_columns = np.flatnonzero(
                    self.representations == representation
                )
                if output_columns.size == 0:
                    continue
                source_columns = self.representation_columns[output_columns]
                selected = cp.asfortranarray(source[:, source_columns])
                selected *= cp.asarray(
                    self.sector_scales[representation], dtype=cp.float64
                )[:, None]
                wedge[
                    cp.asarray(
                        self.sector_orbits[representation], dtype=cp.int64
                    )[:, None],
                    cp.asarray(output_columns, dtype=cp.int64)[None, :],
                ] = selected
        full = cp.empty(self.shape, dtype=cp.float64, order="F")
        # Maps kept on the host are uploaded here, one phase row at a time;
        # ``cp.asarray`` returns an already matching device array unchanged.
        device_full_to_wedge = cp.asarray(
            self.full_to_wedge
            if self.device_full_to_wedge is None
            else self.device_full_to_wedge,
            dtype=cp.int64,
        )
        for representation in range(int(self.phases.shape[0])):
            output_columns = np.flatnonzero(
                self.representations == representation
            )
            if output_columns.size == 0:
                continue
            expanded = wedge[
                :, output_columns
            ][device_full_to_wedge, :]
            expanded *= cp.asarray(
                self.phases[representation], dtype=cp.float64
            )[:, None]
            full[:, output_columns] = expanded
        return full


def selected_device_ids(primary_device_id: int) -> tuple[int, ...]:
    """Return the CUDA devices ``PARSEC_CUPY_DEVICES`` gives to this process.

    ``primary_device_id`` is the current device of the calling thread, the
    only one used under the ``current`` setting.  Preparation needs this list
    before the eigensolver exists to place the Hartree objects.
    """

    available_devices = cupy_device_count()
    device_setting = os.environ.get("PARSEC_CUPY_DEVICES", "auto").strip().lower()
    if device_setting in {"", "auto"}:
        return tuple(range(available_devices))
    if device_setting in {"current", "off"}:
        return (int(primary_device_id),)
    try:
        device_ids = tuple(
            dict.fromkeys(
                int(value.strip())
                for value in device_setting.split(",")
                if value.strip()
            )
        )
    except ValueError as error:
        raise ValueError(
            "PARSEC_CUPY_DEVICES must be 'auto', 'current', or a "
            "comma-separated list of CUDA device indices"
        ) from error
    if not device_ids or any(
        value < 0 or value >= available_devices for value in device_ids
    ):
        raise ValueError("PARSEC_CUPY_DEVICES contains an unavailable device")
    return device_ids


def sector_pool_release_requested() -> bool:
    """Whether a finished sector returns the unused pool blocks of its stream.

    CuPy's memory pool keeps the blocks that arrays gave back in one free
    list per stream, and a process with several devices gives each sector a
    stream of its own.  The Ritz buffer that a sector gave back (1 to 4 GiB,
    :func:`rayleigh_ritz._streaming_workspace`) can then not serve the next
    sector of the same device, which takes another one: both stay until the
    density step empties the pool, although the sectors of a device are
    solved one after another so that they need not hold workspace together.

    ``PARSEC_CUPY_SECTOR_POOL_RELEASE=1``, the default, returns the unused
    blocks of a sector's stream to the device when the sector has finished,
    where the device solves several sectors one after another on streams of
    their own (two devices for four sectors), from the bytes of sector
    vectors at which the density step empties the pool
    (:meth:`CuPySymmetrySCFEigensolver._release_sector_blocks`).  ``0``
    leaves every block to the density step, as before.  Neither changes a
    result.
    """

    value = os.environ.get("PARSEC_CUPY_SECTOR_POOL_RELEASE", "1").strip().lower()
    if value not in {"0", "1", "on", "off", "true", "false"}:
        raise ValueError("PARSEC_CUPY_SECTOR_POOL_RELEASE must be on or off")
    return value in {"1", "on", "true"}


def _pool_release_orbital_bytes() -> int:
    """``PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES``, 512 MiB by default.

    The bytes of sector vectors from which the density step empties the pool
    (:meth:`CuPySymmetryDensityBuilder._release_large_unused_pool`) and a
    finished sector returns the blocks of its stream
    (:meth:`CuPySymmetrySCFEigensolver._release_sector_blocks`).  Both read
    it here: one value moves both, and a value that is no nonnegative
    integer is the same error of either.  The solver and the density builder
    read it when they are built as well, as they do their switches: its
    first use follows a complete eigensolve of every sector.
    """

    name = "PARSEC_CUPY_POOL_RELEASE_ORBITAL_BYTES"
    try:
        threshold = int(os.environ.get(name, str(512 * 1024 * 1024)))
    except ValueError as error:
        raise ValueError(f"{name} must be a nonnegative integer") from error
    if threshold < 0:
        raise ValueError(f"{name} must be nonnegative")
    return threshold


class CuPySymmetrySCFEigensolver:
    """Merge persistent wedge eigensolvers into one SCF eigensolver callable."""

    def __init__(
        self,
        full_operator: Any,
        negative_laplacian: Any,
        nonlocal_operator: NonlocalProjectorOperator,
        decomposition: ReflectionRepresentationDecomposition,
        *,
        timing_stats: CuPyTimingStats,
        local_potential_getter: Callable[[], np.ndarray] | None = None,
        operator_cache_directory: Path | None = None,
        kinetic_cache_key: str | None = None,
        decomposition_cache_key: str | None = None,
        mpi_context: Any | None = None,
    ) -> None:
        self.full_operator = full_operator
        self.decomposition = decomposition
        self.timing_stats = timing_stats
        self._local_potential_getter = local_potential_getter
        self._state: CuPySymmetryEigvalState | None = None
        self._sector_counts: list[int] | None = None
        self._solvers: list[CuPyEigvalSolver] = []
        self._operators: list[CuPyHamiltonian] = []
        self._sector_timing_stats: list[CuPyTimingStats] = []
        self.scheduler_batches = 0
        self.scheduler_wall_seconds = 0.0
        self.memory_allocator_policy = "cupy default pool (not evaluated)"
        self.sector_state_storage = "device"
        # Per device, what ``auto`` held against its memory (see
        # ``_state_fit_bytes``); ``None`` where no ``auto`` asked, and
        # where it decided by the former rule, which counts no such bytes.
        self.state_fit_bytes: dict[int, int] | None = None
        self._previous_memory_allocator = None
        self._memory_allocator_evaluated = False
        self.mpi_context = mpi_context
        self._mpi_latest_results = {}
        # Sector -> [later passes, later passes that filtered in FP32].
        self._later_filter_passes: dict[int, list[int]] = {}

        cp, _ = require_cupy()
        self._primary_device_id = int(cp.cuda.Device().id)
        device_ids = selected_device_ids(self._primary_device_id)
        self.device_ids = device_ids
        self._owned_representations = tuple(range(decomposition.representation_count))
        self.sector_device_groups = {}
        if mpi_context is not None:
            self.sector_device_groups = mpi_context.configure(decomposition.representation_count, device_ids)
            self._owned_representations = mpi_context.owned_sectors
        self._sector_device_ids = tuple(
            self.sector_device_groups[index][0]
            if index in self.sector_device_groups else device_ids[index % len(device_ids)]
            for index in range(decomposition.representation_count)
        )
        scheduler = os.environ.get(
            "PARSEC_CUPY_SECTOR_SCHEDULER", "sequential"
        ).strip().lower()
        if scheduler not in {"streams", "sequential"}:
            raise ValueError(
                "PARSEC_CUPY_SECTOR_SCHEDULER must be 'streams' or 'sequential'"
            )
        requested_streams = int(
            os.environ.get(
                "PARSEC_CUPY_SECTOR_STREAMS",
                str(decomposition.representation_count),
            )
        )
        if requested_streams < 1:
            raise ValueError("PARSEC_CUPY_SECTOR_STREAMS must be positive")
        self.scheduler_workers = (
            min(len(device_ids), decomposition.representation_count)
            if len(device_ids) > 1 and scheduler == "sequential"
            else 1
            if scheduler == "sequential"
            else min(requested_streams, decomposition.representation_count)
        )
        self.scheduler_mode = (
            "sequential"
            if self.scheduler_workers == 1
            else "multi-gpu"
            if len(device_ids) > 1
            else "cuda-streams"
        )
        self._serial_per_device = scheduler == "sequential" and len(device_ids) > 1
        self._streams = []
        for representation, device_id in enumerate(self._sector_device_ids):
            if representation not in self._owned_representations:
                self._streams.append(None)
                continue
            with cp.cuda.Device(device_id):
                self._streams.append(
                    cp.cuda.Stream.null
                    if self.scheduler_workers == 1
                    else cp.cuda.Stream(non_blocking=True)
                )
        self._executor = (
            None
            if self.scheduler_workers == 1
            else ThreadPoolExecutor(
                max_workers=self.scheduler_workers,
                thread_name_prefix="parsec-cuda-sector",
            )
        )
        # Devices on which a finished sector returns the unused pool blocks
        # of its stream (see ``sector_pool_release_requested``): those that
        # solve several sectors of this process one after another, each on
        # a stream of its own.  The sectors of a single device share the
        # default stream, where a freed block serves the next one already.
        release = sector_pool_release_requested() and self._serial_per_device
        # The limit of that release and of the density step is next read
        # when the first sector has been solved: a value that neither can
        # use is refused here, by every solver.  So is the multiple at which
        # the Ritz step cuts its Gram products, which a sector on one device
        # first reads behind its first filter.
        _pool_release_orbital_bytes()
        gram_multiple()
        self.pool_release_devices = tuple(sorted({
            self._sector_device_ids[representation]
            for representation in self._owned_representations
            if release and self._device_sector_count(representation) > 1
        }))
        # Per device: the releases and the bytes that went back to the driver.
        self._pool_releases = {
            device_id: [0, 0] for device_id in self.pool_release_devices
        }
        self.bound_schedule = os.environ.get("PARSEC_CUPY_BOUND_SCHEDULE", "inline").strip().lower()
        if self.bound_schedule not in {"inline", "before_sectors"}:
            raise ValueError("PARSEC_CUPY_BOUND_SCHEDULE must be inline or before_sectors")
        self._precompute_bounds = self.bound_schedule == "before_sectors"
        collective_lanczos_requested = os.environ.get(
            "PARSEC_CUPY_COLLECTIVE_LANCZOS", "0"
        ).strip().lower() not in {"0", "false", "no", "off"}
        # Narrow one-vector Lanczos kernels underfill the GPU.  Overlap only
        # those independent representation bounds; the large Chebyshev and
        # orthogonalization phases retain the measured-fast sequential policy.
        self.collective_lanczos = bool(
            collective_lanczos_requested
            and not self._precompute_bounds
            and self.scheduler_workers == 1
            and decomposition.representation_count > 1
        )
        self._bound_streams = (
            [cp.cuda.Stream.null] * decomposition.representation_count
            if not self.collective_lanczos
            else [
                cp.cuda.Stream(non_blocking=True)
                for _ in range(decomposition.representation_count)
            ]
        )
        self._bound_executor = (
            None
            if not self.collective_lanczos
            else ThreadPoolExecutor(
                max_workers=decomposition.representation_count,
                thread_name_prefix="parsec-cuda-lanczos",
            )
        )
        if self._precompute_bounds:
            # Use each sector's own device/stream. Each unchanged Lanczos
            # recurrence completes its scalar download before any large
            # filter or host Ritz solve is submitted by the worker threads.
            self._bound_streams = self._streams
        # Host orbit index list of every sector this process solves; an MPI
        # rank keeps none for the sectors of other ranks.  No device copy of
        # these lists or of the full-grid expansion maps is made here: the
        # SCF loop reads none, and the orbital export uploads its own.
        self._sector_orbits = tuple(
            decomposition.sector_orbit_indices(representation)
            if representation in self._owned_representations
            else None
            for representation in range(decomposition.representation_count)
        )
        # ``1/sqrt(|O_w|)`` on the orbits of the same sectors.  Every density
        # command reads these constant arrays, so they are formed once here
        # and, like the orbit lists, shared read-only by all SCF steps.
        sector_scales: list[np.ndarray | None] = []
        for orbits in self._sector_orbits:
            if orbits is None:
                sector_scales.append(None)
                continue
            scales = np.ascontiguousarray(
                1.0 / np.sqrt(decomposition.reduction.multiplicities[orbits]),
                dtype=np.float64,
            )
            scales.setflags(write=False)
            sector_scales.append(scales)
        self._sector_scales = tuple(sector_scales)

        symmetric_rows = np.flatnonzero(
            np.all(decomposition.characters == 1, axis=1)
        )
        if symmetric_rows.size != 1:
            raise RuntimeError(
                "reflection character table has no unique totally symmetric sector"
            )
        self.totally_symmetric_representation = int(symmetric_rows[0])
        # An MPI rank assembles only the sectors it owns.  The root rank adds
        # the totally symmetric one, whose reduced Laplacian its Poisson
        # setup reuses; the other ranks never solve Poisson.
        prepares_poisson = (
            mpi_context is None or mpi_context.rank == mpi_context.root
        )
        static_operators = load_or_build_reduced_operators(
            decomposition,
            negative_laplacian,
            nonlocal_operator,
            cache_directory=operator_cache_directory,
            kinetic_key_seed=kinetic_cache_key,
            decomposition_key_seed=decomposition_cache_key,
            representations=(
                None
                if mpi_context is None
                else (*self._owned_representations,
                      self.totally_symmetric_representation)
                if prepares_poisson
                else self._owned_representations
            ),
        )
        self.operator_cache_info = static_operators.cache_info
        # The GPU Poisson solver reads this packed stencil as it is.  Its
        # CSR form is rebuilt only for a consumer that asks for one, and
        # then takes its place.
        self.totally_symmetric_stencil = (
            static_operators.stencil_metadata[
                self.totally_symmetric_representation
            ]
            if prepares_poisson
            else None
        )
        self._totally_symmetric_negative_laplacian = None
        built_metadata = [
            item for item in static_operators.stencil_metadata if item is not None
        ]
        # Sectors assembled by this process; sizes its stencil-packing pool.
        self.assembled_sector_count = len(built_metadata)
        first_neighbors = built_metadata[0].neighbors
        neighbors_are_shared = all(
            np.array_equal(first_neighbors, item.neighbors)
            for item in built_metadata[1:]
        )
        # Tile size of the stencil of every sector solved here; 0 keeps the
        # slot-major layout (see ``implicit_tile_for_sector``).
        sector_tiles = {
            representation: implicit_tile_for_sector(
                decomposition.sector_size(representation),
                self._filter_device_count(representation),
                float32_filter_requested(
                    decomposition.sector_size(representation)
                ),
                sectors_on_device=self._device_sector_count(representation),
            )
            for representation in self._owned_representations
        }
        # Affine-tile descriptors depend on coefficients as well as neighbor
        # indices; different representations must own their packed metadata.
        if any(sector_tiles.values()):
            neighbors_are_shared = False
        shared_device_neighbors: dict[int, Any] = {}
        shared_device_potentials: dict[tuple[int, bytes], Any] = {}
        sector_local_potentials: list[Any] = []
        for representation, (stencil_metadata, reduced_nonlocal) in enumerate(zip(
            static_operators.stencil_metadata,
            static_operators.nonlocal_operators,
            strict=True,
        )):
            if representation not in self._owned_representations:
                self._operators.append(None)
                self._solvers.append(None)
                self._sector_timing_stats.append(CuPyTimingStats())
                sector_local_potentials.append(None)
                continue
            device_id = self._sector_device_ids[representation]
            sector_orbits = self._sector_orbits[representation]
            potential_key = (device_id, sector_orbits.tobytes())
            zero_local = np.zeros(
                decomposition.sector_size(representation), dtype=np.float64
            )
            with cp.cuda.Device(device_id):
                sector_timing = CuPyTimingStats()
                operator = CuPyHamiltonian(
                    None,
                    zero_local,
                    reduced_nonlocal,
                    timing_stats=sector_timing,
                    retain_generic_laplacian=False,
                    finite_difference_metadata=stencil_metadata,
                    shared_stencil_neighbors=(
                        shared_device_neighbors.get(device_id)
                        if neighbors_are_shared
                        else None
                    ),
                    shared_effective_potential=shared_device_potentials.get(
                        potential_key
                    ),
                    implicit_tile=sector_tiles[representation],
                )
                if mpi_context is not None:
                    operator.distributed_filter_devices = self.sector_device_groups[representation]
                if neighbors_are_shared and device_id not in shared_device_neighbors:
                    selected_stencil = operator.compact_finite_difference
                    shared_device_neighbors[device_id] = getattr(
                        selected_stencil, "neighbors", None
                    )
                if potential_key not in shared_device_potentials:
                    shared_device_potentials[potential_key] = (
                        operator.effective_potential
                    )
                sector_local_potentials.append(operator.effective_potential)
                self._sector_timing_stats.append(sector_timing)
                self._operators.append(operator)
                self._solvers.append(
                    CuPyEigvalSolver(
                        operator,
                        settings=EigvalSettings(safety_buffer=0),
                        timing_stats=sector_timing,
                        retain_vectors_on_device=True,
                        compute_subspace_residuals=False,
                    )
                )
        self._sector_local_potentials = tuple(sector_local_potentials)
        local_potential_buffer_count = len(
            {
                (device_id, int(potential.data.ptr))
                for device_id, potential in zip(
                    self._sector_device_ids,
                    self._sector_local_potentials,
                    strict=True,
                )
                if potential is not None
            }
        )
        device_count = len({self._sector_device_ids[index] for index in self._owned_representations})
        if local_potential_buffer_count == device_count:
            self.local_potential_storage = (
                "one persistent shared device buffer per CUDA device"
            )
        else:
            self.local_potential_storage = (
                "one persistent device buffer per distinct "
                "stabilizer-filtered sector map and CUDA device"
            )
        # Static uploads occurred sequentially during construction and are
        # therefore additive wall time.  Later per-sector counters are merged
        # only after worker completion, avoiding shared Python mutations.
        self.timing_stats.initialization_seconds += sum(
            item.initialization_seconds for item in self._sector_timing_stats
        )
        storage_modes = {
            (
                operator.compact_finite_difference.storage_mode
                if operator.compact_finite_difference is not None
                else "float64_csr"
            )
            for operator in self._operators
            if operator is not None
        }
        self.finite_difference_storage = (
            storage_modes.pop() if len(storage_modes) == 1 else "mixed"
        )
        self.finite_difference_neighbors = (
            "shared_across_representations"
            if neighbors_are_shared and shared_device_neighbors is not None
            else "private_per_representation"
        )

    def _filter_device_count(self, representation: int) -> int:
        """Return how many devices filter the basis of one sector solved here.

        An MPI rank counts the group it was given for the sector, which may
        share the basis (:mod:`distributed_state`) or its filter.  Without a
        sector context the filter is spread over every device of the process
        under ``PARSEC_CUPY_DISTRIBUTED_FILTER=1``
        (:func:`distributed_filter.distributed_filter`) and stays on the
        device of the sector otherwise.
        """

        group = self.sector_device_groups.get(representation)
        if group is not None:
            return len(group)
        if os.environ.get("PARSEC_CUPY_DISTRIBUTED_FILTER", "0") == "1":
            return len(self.device_ids)
        return 1

    def _device_sector_count(self, representation: int) -> int:
        """Return how many sectors solved here have the device of this one.

        The device of a sector is the one that holds its basis, or the first
        of the group that shares it.  The default scheduler solves the
        sectors of a device one after another.
        """

        device_id = self._sector_device_ids[representation]
        return sum(
            self._sector_device_ids[index] == device_id
            for index in self._owned_representations
        )

    @property
    def hartree_device(self) -> int | None:
        """The device that holds the Hartree objects of this process, as the driver names it.

        ``None`` where they share no device.  Every sector operator is told:
        a sector that shares its basis gives that device, the fullest of the
        process, no work beside its own
        (:meth:`distributed_state.SectorDeviceGroup._condition_helper`).
        """

        return getattr(self, "_hartree_device", None)

    @hartree_device.setter
    def hartree_device(self, device_id: int | None) -> None:
        self._hartree_device = device_id
        for operator in self._operators:
            if operator is not None:
                operator.hartree_device = device_id

    @property
    def state(self) -> CuPySymmetryEigvalState | None:
        return self._state

    @property
    def representation_count(self) -> int:
        return self.decomposition.representation_count

    @property
    def totally_symmetric_negative_laplacian(self):
        """Canonical CSR of the totally symmetric sector, built on first use.

        The CSR then replaces ``totally_symmetric_stencil``, which becomes
        ``None``.  ``None`` on an MPI rank that does not prepare the Poisson
        solver.
        """

        if (
            self._totally_symmetric_negative_laplacian is None
            and self.totally_symmetric_stencil is not None
        ):
            self._totally_symmetric_negative_laplacian = (
                self.totally_symmetric_stencil.to_csr()
            )
            # A consumer takes one form or the other.  Keeping the packed
            # arrays beside the CSR would hold this sector twice on the host.
            self.totally_symmetric_stencil = None
        return self._totally_symmetric_negative_laplacian

    @property
    def fused_projector_scatter(self) -> bool:
        """Whether every nonempty sector fuses the large KB scatter."""

        relevant = [operator for operator in self._operators if operator is not None and operator.projector_count]
        return bool(relevant) and all(
            operator.fused_projector_scatter for operator in relevant
        )

    @property
    def custom_projector_projection(self) -> bool:
        """Whether every nonempty sector replaces tiny cuSPARSE B.T calls."""

        relevant = [operator for operator in self._operators if operator is not None and operator.projector_count]
        return bool(relevant) and all(
            operator.custom_projector_projection is not None
            for operator in relevant
        )

    @property
    def projector_reduction_modes(self) -> str:
        """Report the selected short/long-row CUDA dot policies."""

        modes = {
            getattr(operator.custom_projector_projection, "reduction_mode", "none")
            for operator in self._operators
            if operator is not None and operator.projector_count
        }
        return "+".join(sorted(modes)) if modes else "none"

    @property
    def later_filter_precision(self) -> str:
        """Precision of the later filter passes of the symmetry sectors.

        Once a later pass has run, this is what ran, in every sector of the
        calculation (the record of a solve says so, also for a sector of
        another MPI rank): ``float64`` if no pass took the FP32 recurrence,
        the number of passes and the sectors that did otherwise.  A sector
        that holds an FP32 recurrence still filters in FP64 where its basis
        is shared among devices and, under ``auto``, where a pass is below
        the work threshold.

        Before any later pass it is what this process prepared: ``float64``
        if none of its sectors holds an FP32 recurrence, which only
        ``PARSEC_CUPY_MIXED_FILTER`` asks for.
        """

        passes = getattr(self, "_later_filter_passes", None) or {}
        total = sum(count[0] for count in passes.values())
        if total:
            float32 = {
                sector: count[1] for sector, count in passes.items() if count[1]
            }
            if not float32:
                return "float64"
            sectors = " ".join(map(str, sorted(float32)))
            return (
                "float32 stencil/projectors/recurrence in "
                f"{sum(float32.values())} of {total} later filter passes "
                f"(sectors {sectors}); float64 Ritz and SCF"
            )
        prepared = [
            index
            for index, operator in enumerate(self._operators)
            if operator is not None
            and operator.mixed_precision_recurrence is not None
        ]
        if not prepared:
            return "float64"
        sectors = " ".join(map(str, prepared))
        return (
            "float32 stencil/projectors/recurrence prepared for sectors "
            f"{sectors}; float64 Ritz and SCF"
        )

    @property
    def recorded_filter_graphs(self) -> str:
        """How the filters of the sectors of this process recorded their graphs.

        Reports what ran, so it is read after a solve:
        :func:`filter_graph.graph_route_name` of the graph workspaces that
        hold a graph, those of the devices that filter for a sector
        included, or ``none recorded``.  A filter records graphs only where
        ``PARSEC_CUPY_FILTER_GRAPHS`` or ``PARSEC_CUPY_DISTRIBUTED_FILTER``
        is set or the basis of its sector is shared, so
        ``PARSEC_CUPY_FILTER_GRAPH_REUSE`` alone says nothing about a run.
        """

        routes = set()
        for operator in self._operators:
            if operator is None:
                continue
            workspaces = [getattr(operator, "_filter_graph_workspace", None)]
            shared = getattr(operator, "_distributed_filter", None)
            if shared is not None:
                workspaces.extend(shared.graphs.values())
            routes.update(
                workspace.reuse
                for workspace in workspaces
                if workspace is not None and workspace.recorded
            )
        if not routes:
            return "none recorded"
        return "; ".join(graph_route_name(reuse) for reuse in sorted(routes, reverse=True))

    @property
    def low_memory_ritz_sectors(self) -> tuple[int, ...]:
        """Sectors of this process that keep their basis as one array.

        Reports what ran, so it is read after a solve.  A sector is left out
        if its first solve spread the basis over the devices of its group
        (:func:`distributed_state.sector_device_group`): only a route that
        holds the basis on one device reads the ``low_memory_ritz`` mark.
        For the others :func:`rayleigh_ritz.streaming_ritz_requested` answers
        under the setting in force, an explicit on or off included.
        """

        return tuple(
            representation
            for representation, operator in enumerate(self._operators)
            if operator is not None
            and getattr(operator, "_sector_device_group", None) is None
            and streaming_ritz_requested(operator)
        )

    def reset(self) -> None:
        """Discard every saved representation subspace."""

        context = getattr(self, "mpi_context", None)
        if context is not None:
            context.execute(self, "reset")
            return
        self._reset_local()

    def _reset_local(self) -> None:
        for solver in self._solvers:
            if solver is not None:
                solver.reset()
        self._state = None
        self._sector_counts = None

    def _state_fit_bytes(self, counts: list[int]) -> dict[int, int]:
        """Bytes that the rule of the state storage counts on each device for the sectors solved here.

        A sector whose basis lies on one device counts its one array there,
        and the device counts, beside the arrays of all its sectors, the
        buffer of the largest of them
        (:func:`rayleigh_ritz.streaming_workspace_bytes`): its sectors are
        solved one after another and each gives the buffer back.  A sector
        whose basis the devices of its group are to share counts on each of
        them what the rule that shares it counts
        (:func:`distributed_state.shared_block_bytes`).  Whether they are
        to share it is asked as a CHEBFF first solve asks
        (:func:`distributed_state.shared_basis_devices`, with the device of
        the sector current); another first solver leaves the basis on the
        owner, which is then counted too low.

        For a basis on one device the count is a lower limit of what the
        device must hold in a Ritz step and nothing else: the operators,
        the buffers of the filter graphs, the Gram matrices, the small solve
        and the CUDA context are left out, and the Ritz step with two arrays
        (``PARSEC_CUPY_STREAMING_RITZ``) is counted as the one with one.  A
        device that this count overfills cannot keep the states of its
        sectors on any route; one that it does not may still be too small
        for them (58.8 GiB were sampled where it gives 56.0).

        For a shared basis the count is not a lower limit.  It is the count
        of the rule that shares the basis, whose workspace is that of slabs
        as wide as their budget allows
        (:func:`distributed_state.slab_workspace`); the slabs that are cut
        can be narrower.  Of 10,456 electrons on 16 GPUs every device counts
        9,945 MiB, and devices of that run sampled as little as 10,005 MiB,
        426 of them the CUDA context: 366 MiB less than the count outside
        the context.  So it was, by that or less, on 61 of the 208 devices
        with shared blocks in the first runs of one series, all from 14,680
        electrons down; from 19,392 up the devices sampled 0.5 to 5.0 GiB
        more than the count and the context.  The sharing rule holds this
        count to its own share of a device, so nothing here is decided by
        it.

        The count reads the memory of no device and no state of the run, so
        two launches of one calculation count the same.
        """

        cp, _ = require_cupy()
        states = {device_id: 0 for device_id in self.device_ids}
        buffers = dict(states)
        for representation, count in enumerate(counts):
            if self._solvers[representation] is None:
                continue
            rows = self.decomposition.sector_size(representation)
            owner = self._sector_device_ids[representation]
            with cp.cuda.Device(owner):
                shared = shared_basis_devices(
                    self._operators[representation], int(count)
                )
            if shared is None:
                states[owner] += rows * int(count) * np.dtype(np.float64).itemsize
                buffers[owner] = max(
                    buffers[owner], streaming_workspace_bytes(rows, count)
                )
                continue
            for device_id in shared:
                states[device_id] += shared_block_bytes(rows, count, len(shared))
        return {
            device_id: states[device_id] + buffers[device_id]
            for device_id in self.device_ids
        }

    def _configure_large_problem_allocator(self, counts: list[int]) -> None:
        """Decide where the sector states are kept and which allocator serves them.

        ``PARSEC_CUPY_SECTOR_STATE_STORAGE`` (``device``, ``host`` or
        ``auto``) and ``PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR`` (``pool``,
        ``direct`` or ``auto``) name the two; a named value is taken as it
        is.  ``auto``, the default of both, keeps the states on their
        devices and the pool of CuPy unless the sectors of some device
        cannot fit it (:meth:`_state_fit_bytes` above
        ``PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION`` of the memory of the
        device, 0.978 by default): then the states are spilled to the host
        after every solve and the allocations bypass the pool, so that a
        device holds the blocks of one sector at a time.  A run whose
        sectors fit thus takes the route of ``device`` and ``pool`` without
        naming them.

        The count leaves out what else a device holds, so it is held
        against a share of the device and not all of it.  In first runs on
        A100 nodes every device with a basis of its own sampled at least
        1.026 times its count (39 devices of 15 runs, 3.2 to 75.5 GiB
        counted), and from 50 GiB counted up 1.8 GiB or more above it.  At
        0.978 the rule therefore takes a device as too small only where
        that ratio puts its peak above its memory, 1.74 GiB below the
        79.25 GiB of such a device.  The fullest device that ran, 75.5 GiB
        counted and 78.3 sampled, keeps its states.  10,456 electrons on
        one device count 77.7 GiB, 1.5 GiB below the device, and are
        spilled, as the rule before spilled them: by those figures they
        would not have fitted.  ``1`` holds the count against all of the
        device.  A shared basis is counted as the rule that shares it
        counts it and is held by that rule to
        ``PARSEC_CUPY_DISTRIBUTED_STATE_AUTO_FRACTION`` of a device (0.85),
        so this share decides nothing for it unless that one is raised
        above it or the sharing is forced.

        The rule before this one spilled and left the pool as soon as the
        vectors of a device, the whole basis of a shared sector counted on
        its owner, reached half of its memory.  It dates from two or three
        arrays per sector.  With one, 19,392 electrons on four devices of
        79.25 GiB hold 52.0 GiB of vectors each and peaked at 58.8 GiB with
        the states on the devices; left to that rule the same run stopped
        for memory where its first eigensolve ends, as had been derived
        from the code: the download of a spill first made a second copy of
        the basis on its device, which it no longer does
        (:func:`eigval.spill_download_order`).  Naming
        ``PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION`` selects that rule, with
        the fraction it names; ``0.5`` was its default.
        """

        if self._memory_allocator_evaluated:
            return
        self._memory_allocator_evaluated = True
        policy = os.environ.get(
            "PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR", "auto"
        ).strip().lower()
        if policy not in {"auto", "pool", "direct"}:
            raise ValueError(
                "PARSEC_CUPY_LARGE_PROBLEM_ALLOCATOR must be auto, pool, or direct"
            )
        threshold = float(
            os.environ.get("PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION", "0.5")
        )
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                "PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION must be in (0, 1]"
            )
        storage_policy = os.environ.get(
            "PARSEC_CUPY_SECTOR_STATE_STORAGE", "auto"
        ).strip().lower()
        if storage_policy not in {"auto", "device", "host"}:
            raise ValueError(
                "PARSEC_CUPY_SECTOR_STATE_STORAGE must be auto, device, or host"
            )
        share = float(
            os.environ.get("PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION", "0.978")
        )
        if not 0.0 < share <= 1.0:
            raise ValueError(
                "PARSEC_CUPY_SECTOR_STATE_AUTO_FRACTION must be in (0, 1]"
            )
        # Read by the download of a spill, after the first solve: a value
        # it does not know stops the run here, before that solve.
        spill_download_order()

        cp, _ = require_cupy()
        persistent_bytes: dict[int, int] = {
            device_id: 0 for device_id in self.device_ids
        }
        for representation, count in enumerate(counts):
            if self._solvers[representation] is None:
                continue
            device_id = self._sector_device_ids[representation]
            persistent_bytes[device_id] += (
                self.decomposition.sector_size(representation)
                * int(count)
                * np.dtype(np.float64).itemsize
            )
        totals: dict[int, int] = {}
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                _, total = cp.cuda.runtime.memGetInfo()
                totals[device_id] = int(total)
        # What ``auto`` of either asks: whether some device is too small
        # for the states of its sectors.  Two named values ask nothing.
        too_small = False
        self.state_fit_bytes = None
        if "auto" in (policy, storage_policy):
            if "PARSEC_CUPY_DIRECT_ALLOCATOR_FRACTION" in os.environ:
                too_small = any(
                    persistent_bytes[device_id] >= threshold * totals[device_id]
                    for device_id in self.device_ids
                )
            else:
                self.state_fit_bytes = self._state_fit_bytes(counts)
                too_small = any(
                    self.state_fit_bytes[device_id] > share * totals[device_id]
                    for device_id in self.device_ids
                )
        use_direct = policy == "direct" or (policy == "auto" and too_small)
        self._host_spill_sector_states = bool(
            storage_policy == "host" or (storage_policy == "auto" and too_small)
        )
        self.sector_state_storage = (
            "exact FP64 host spill; one active representation on CUDA"
            if self._host_spill_sector_states
            else "persistent CUDA representation states"
        )
        # PARSEC_CUPY_STREAMING_RITZ=auto: a sector whose basis lies on one
        # device keeps one array of its subspace.  The ordinary Ritz solve
        # keeps two (measured peak about twice the persistent orbitals plus
        # 3 GiB) and takes as long.  PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION
        # above zero returns to two arrays on a device where they stay within
        # that fraction of its memory.
        crowded = float(
            os.environ.get("PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION", "0")
        )
        if not 0.0 <= crowded <= 1.0:
            raise ValueError(
                "PARSEC_CUPY_STREAMING_RITZ_AUTO_FRACTION must be in [0, 1]"
            )
        # Only a route that holds the basis on one device reads the mark;
        # ``low_memory_ritz_sectors`` reports where it acted.
        for representation, operator in enumerate(self._operators):
            if operator is None:
                continue
            device_id = self._sector_device_ids[representation]
            operator.low_memory_ritz = bool(
                2.0 * persistent_bytes[device_id] + 3 * 1024**3
                > crowded * totals[device_id]
            )
        if not use_direct:
            self.memory_allocator_policy = (
                "cupy default pool; estimated persistent sector orbitals "
                + ",".join(
                    f"cuda:{device_id}={persistent_bytes[device_id]}B"
                    for device_id in self.device_ids
                )
            )
            return

        # Static Hamiltonian buffers allocated earlier remain valid.  Only
        # subsequent allocations bypass the caching pool, so temporary
        # ChebDav/SUBSPACE workspaces are returned to CUDA as soon as their
        # owning arrays die instead of accumulating split blocks across SCF.
        self._previous_memory_allocator = cp.cuda.get_allocator()
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.set_allocator(None)
        self.memory_allocator_policy = (
            "direct CUDA allocation for memory-bound sectors; estimated "
            "persistent sector orbitals "
            + ",".join(
                f"cuda:{device_id}={persistent_bytes[device_id]}B"
                for device_id in self.device_ids
            )
        )

    def restore_memory_allocator(self) -> None:
        """Restore the process allocator after this eigensolver is finished."""

        if self._previous_memory_allocator is None:
            return
        cp, _ = require_cupy()
        cp.cuda.set_allocator(self._previous_memory_allocator)
        self._previous_memory_allocator = None

    def _initial_sector_count(
        self,
        representation: int,
        requested_states: int,
        safety_buffer: int,
    ) -> int:
        # PARSEC initeigval: nadd + nstate/nrep (integer division).  The ceil
        # guard matters only for an unusual user choice nadd=0 and ensures the
        # union contains at least the globally requested number of states.
        count = max(
            (requested_states + self.representation_count - 1)
            // self.representation_count,
            requested_states // self.representation_count + safety_buffer,
        )
        return min(self.decomposition.sector_size(representation), max(1, count))

    def _update_local_potential(self, full_potential) -> None:
        """Upload one shared invariant wedge field for all sector operators."""

        cp, _ = require_cupy()
        from ..SCF.symmetry_fields import SymmetryScalarField

        if isinstance(full_potential, SymmetryScalarField):
            if full_potential.reduction is not self.decomposition.reduction:
                raise ValueError("local potential uses a different symmetry map")
            wedge = full_potential.values
        else:
            wedge = self.decomposition.invariant_wedge_values(full_potential)
        started = perf_counter()
        host_wedge = np.ascontiguousarray(wedge, dtype=np.float64)
        updated_buffers: set[tuple[int, int]] = set()
        for representation, (device_id, device) in enumerate(
            zip(
                self._sector_device_ids,
                self._sector_local_potentials,
                strict=True,
            )
        ):
            if device is None:
                continue
            pointer = (device_id, int(device.data.ptr))
            if pointer in updated_buffers:
                continue
            with cp.cuda.Device(device_id):
                device.set(host_wedge[self._sector_orbits[representation]])
            updated_buffers.add(pointer)
        # Sector Hamiltonians share the FP64 field above.  Their optional
        # mixed-precision Chebyshev recurrences own private FP32 shadows, so
        # refresh those shadows from the just-updated device field before any
        # sector is scheduled.  No host transfer is repeated here.
        for device_id, operator in zip(
            self._sector_device_ids, self._operators, strict=True
        ):
            if operator is None:
                continue
            recurrence = operator.mixed_precision_recurrence
            if recurrence is not None:
                with cp.cuda.Device(device_id):
                    recurrence.update_potential(
                        operator.effective_potential
                    )
        for device_id in self.device_ids:
            with cp.cuda.Device(device_id):
                synchronize()
        self.timing_stats.potential_update_seconds += perf_counter() - started

    @staticmethod
    def _global_order(
        results: list[CuPyEigvalResult],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.concatenate([result.eigenvalues for result in results])
        representations = np.concatenate(
            [
                np.full(result.eigenvalues.size, index, dtype=np.int32)
                for index, result in enumerate(results)
            ]
        )
        columns = np.concatenate(
            [
                np.arange(result.eigenvalues.size, dtype=np.int32)
                for result in results
            ]
        )
        order = np.argsort(values, kind="stable")
        return values[order], representations[order], columns[order]

    def _solve_sectors(
        self,
        settings: EigvalSettings,
        counts: list[int],
        previous_state_supplied: bool,
    ) -> tuple[list[CuPyEigvalResult], bool, list[str]]:
        sector_settings = replace(settings, safety_buffer=0)
        results: list[CuPyEigvalResult] = []
        restarted = False
        reasons: list[str] = []
        solved = self._run_sector_jobs(
            tuple(range(self.representation_count)),
            counts,
            sector_settings,
            reset=not previous_state_supplied,
        )
        for representation in range(self.representation_count):
            result = solved[representation]
            results.append(result)
            self._record_filter_precision(representation, result)
            restarted = restarted or result.restarted
            if result.restart_reason is not None:
                reasons.append(
                    f"representation {representation + 1}: {result.restart_reason}"
                )
        return results, restarted, reasons

    def _record_filter_precision(
        self, representation: int, result: CuPyEigvalResult
    ) -> None:
        """Count a later pass of one sector and whether it filtered in FP32.

        The record of the solve says so whichever MPI rank ran it;
        ``later_filter_precision`` reports from these counts.
        """

        if result.solver_path != "subspace":
            return
        passes = getattr(self, "_later_filter_passes", None)
        if passes is None:
            passes = self._later_filter_passes = {}
        count = passes.setdefault(representation, [0, 0])
        count[0] += 1
        count[1] += result.filter_precision == "float32"

    def _run_one_sector(
        self,
        representation: int,
        count: int,
        settings: EigvalSettings,
        *,
        reset: bool,
        spectral_bound: Any | None = None,
    ) -> CuPyEigvalResult:
        """Run one independent solver on its assigned CUDA stream."""

        cp, _ = require_cupy()
        with cp.cuda.Device(self._sector_device_ids[representation]):
            with self._streams[representation]:
                solver = self._solvers[representation]
                if reset:
                    solver.reset()
                elif getattr(self, "_host_spill_sector_states", False):
                    solver.restore_state_to_device()
                result = solver.solve(
                    count,
                    settings=settings,
                    spectral_bound=spectral_bound,
                )
                if getattr(self, "_host_spill_sector_states", False):
                    host_state = solver.offload_state_to_host()
                    assert host_state is not None
                    result = replace(
                        result,
                        vectors=host_state.subspace.vectors,
                        state=host_state,
                    )
                self._release_sector_blocks(representation)
                return result

    def _sector_vector_bytes(self) -> int:
        """Bytes of the vectors of the sectors solved here, at their counts.

        What the density step sums before it empties the pool
        (:meth:`CuPySymmetryDensityBuilder._release_large_unused_pool`): it
        is handed the vectors of these sectors and no others, each as many
        float64 columns as the solve in progress was asked for.
        """

        counts = getattr(self, "_sector_counts", None)
        if counts is None:
            return 0
        return np.dtype(np.float64).itemsize * sum(
            self.decomposition.sector_size(representation)
            * int(counts[representation])
            for representation in self._owned_representations
        )

    def _release_sector_blocks(self, representation: int) -> None:
        """Return the unused pool blocks of a finished sector to its device.

        Only on the devices of ``pool_release_devices`` (see
        :func:`sector_pool_release_requested`), and only where the vectors
        of the sectors solved here together reach the bytes from which the
        density step empties the pool (:meth:`_sector_vector_bytes`).  With
        the states on their devices the blocks go back there in the same SCF
        step anyway, so that the next step takes no allocation that it did
        not take before; below that size the blocks stay through the density
        step and serve the next solve.  With the states spilled to the host
        (``PARSEC_CUPY_SECTOR_STATE_STORAGE``) the density step empties no
        device pool: there the release returns blocks that stayed cached
        before, so that the device holds the blocks of one sector at a time,
        and every sector takes its vectors and workspace from the driver
        again in each step.  So it does under the direct allocator, which a
        device too small for the states of its sectors is given together
        with the spill where both settings are ``auto``
        (:meth:`_configure_large_problem_allocator`), and where the pool has
        nothing to return.

        Called by the thread that solved the sector, with its device and
        stream current: the solve has synchronized that stream, so nothing
        queued reads a freed block, and the thread has no graph capture
        open, which returning memory to the driver would invalidate
        (:mod:`backends.cupy_capture`).  A capture that another thread has
        open on the device stays valid; for one on another device that was
        not measured, and a capture that is invalidated is recorded again
        and counted.  A block of which a part is still in use stays in the
        pool.
        """

        device_id = self._sector_device_ids[representation]
        counts = (getattr(self, "_pool_releases", None) or {}).get(device_id)
        if counts is None or (
            self._sector_vector_bytes() < _pool_release_orbital_bytes()
        ):
            return
        cp, _ = require_cupy()
        pool = cp.get_default_memory_pool()
        held = int(pool.total_bytes())
        pool.free_all_blocks(stream=self._streams[representation])
        counts[0] += 1
        # Exact while no other thread uses the pool of the device meanwhile.
        counts[1] += max(0, held - int(pool.total_bytes()))

    @property
    def sector_pool_releases(self) -> dict[int, dict[str, int]]:
        """Per device: how often a finished sector returned pool blocks, and the bytes.

        Reports what ran, so it is read after a solve; empty where no device
        releases (see :func:`sector_pool_release_requested`).
        """

        return {
            device_id: dict(releases=int(count), bytes=int(size))
            for device_id, (count, size) in (
                getattr(self, "_pool_releases", None) or {}
            ).items()
        }

    def _run_one_bound(
        self,
        representation: int,
        count: int,
        settings: EigvalSettings,
        *,
        reset: bool,
    ):
        """Prepare one sector's unchanged PARSEC Lanczos bound on its stream."""

        cp, _ = require_cupy()
        with cp.cuda.Device(self._sector_device_ids[representation]):
            with self._bound_streams[representation]:
                return self._solvers[representation].prepare_spectral_bound(
                    count,
                    settings=settings,
                    reset=reset,
                )

    def _run_sector_jobs(
        self,
        representations: tuple[int, ...],
        counts: list[int],
        settings: EigvalSettings,
        *,
        reset: bool,
    ) -> dict[int, CuPyEigvalResult]:
        context = getattr(self, "mpi_context", None)
        if context is None:
            return self._run_local_sector_jobs(representations, counts, settings, reset=reset)
        return context.execute(self, "solve", dict(
            representations=representations, counts=counts, settings=settings, reset=reset))

    def _run_local_sector_jobs(
        self,
        representations: tuple[int, ...],
        counts: list[int],
        settings: EigvalSettings,
        *,
        reset: bool,
    ) -> dict[int, CuPyEigvalResult]:
        """Submit independent sectors and merge timing after synchronization."""

        before = [item.as_dict() for item in self._sector_timing_stats]
        started = perf_counter()
        spectral_bounds: dict[int, Any] = {}
        if getattr(self, "_precompute_bounds", False):
            bound_started = perf_counter()
            spectral_bounds = {
                representation: self._run_one_bound(
                    representation, counts[representation], settings, reset=reset)
                for representation in representations
            }
            self.timing_stats.eigensolver_bound_prepare_wall_seconds += perf_counter() - bound_started
        elif self._bound_executor is not None and len(representations) > 1:
            bound_futures = {
                representation: self._bound_executor.submit(
                    self._run_one_bound,
                    representation,
                    counts[representation],
                    settings,
                    reset=reset,
                )
                for representation in representations
            }
            wait(bound_futures.values())
            spectral_bounds = {
                representation: bound_futures[representation].result()
                for representation in representations
            }
        if self._executor is None:
            results = {
                representation: self._run_one_sector(
                    representation,
                    counts[representation],
                    settings,
                    reset=reset,
                    spectral_bound=spectral_bounds.get(representation),
                )
                for representation in representations
            }
        elif self._serial_per_device:
            # A shared queue of individual sectors can start a second sector
            # on a busy GPU when another GPU finishes early. One batch per
            # device bounds concurrent workspace and preserves the intended
            # sequential-on-each-device policy even with uneven sector sizes.
            def run_device_batch(device_id):
                return {
                    representation: self._run_one_sector(
                        representation, counts[representation], settings,
                        reset=reset,
                        spectral_bound=spectral_bounds.get(representation),
                    )
                    for representation in representations
                    if self._sector_device_ids[representation] == device_id
                }

            futures = {
                device_id: self._executor.submit(run_device_batch, device_id)
                for device_id in dict.fromkeys(
                    self._sector_device_ids[index] for index in representations
                )
            }
            # Finish all in-flight CUDA work before propagating a failure;
            # callers may then safely restore allocator policy or free state.
            wait(futures.values())
            by_representation = {}
            for future in futures.values():
                by_representation.update(future.result())
            results = {index: by_representation[index] for index in representations}
        else:
            futures = {
                representation: self._executor.submit(
                    self._run_one_sector,
                    representation,
                    counts[representation],
                    settings,
                    reset=reset,
                    spectral_bound=spectral_bounds.get(representation),
                )
                for representation in representations
            }
            wait(futures.values())
            # Dictionary insertion and retrieval follow the supplied PARSEC
            # representation order, independent of task completion order.
            results = {
                representation: futures[representation].result()
                for representation in representations
            }
        self.scheduler_batches += 1
        self.scheduler_wall_seconds += perf_counter() - started

        for sector_stats, snapshot in zip(
            self._sector_timing_stats, before, strict=True
        ):
            current = sector_stats.as_dict()
            for name, old_value in snapshot.items():
                if name == "initialization_seconds":
                    continue
                delta = current[name] - old_value
                setattr(
                    self.timing_stats,
                    name,
                    getattr(self.timing_stats, name) + delta,
                )
        return results

    def _ensure_spectral_bracket(
        self,
        results: list[CuPyEigvalResult],
        counts: list[int],
        requested_states: int,
        settings: EigvalSettings,
    ) -> tuple[list[CuPyEigvalResult], bool, list[str]]:
        """Grow sectors whose last computed value does not bracket the cutoff."""

        restarted = False
        reasons: list[str] = []
        for _ in range(8):
            values, representations, _ = self._global_order(results)
            if values.size < requested_states:
                raise RuntimeError("symmetry sectors returned too few eigenvalues")
            cutoff = float(values[requested_states - 1])
            grow: list[int] = []
            for representation, result in enumerate(results):
                sector_size = self.decomposition.sector_size(representation)
                if counts[representation] >= sector_size:
                    continue
                scale = max(1.0, abs(cutoff), abs(float(result.eigenvalues[-1])))
                if float(result.eigenvalues[-1]) <= cutoff + 1.0e-11 * scale:
                    grow.append(representation)
            if not grow:
                return results, restarted, reasons

            increment = max(1, settings.safety_buffer)
            sector_settings = replace(settings, safety_buffer=0)
            for representation in grow:
                old_count = counts[representation]
                counts[representation] = min(
                    self.decomposition.sector_size(representation),
                    max(old_count + increment, (3 * old_count + 1) // 2),
                )
                restarted = True
                reasons.append(
                    f"representation {representation + 1}: spectral bracket "
                    f"grew {old_count}->{counts[representation]}"
                )
            grown = self._run_sector_jobs(
                tuple(grow), counts, sector_settings, reset=False
            )
            for representation in grow:
                results[representation] = grown[representation]
                self._record_filter_precision(
                    representation, grown[representation]
                )
        raise RuntimeError(
            "symmetry representation state allocation did not bracket the "
            "global requested eigenspectrum"
        )

    def _trim_sector_states_like_parsec(
        self,
        results: list[CuPyEigvalResult],
        counts: list[int],
        requested_states: int,
        safety_buffer: int,
    ) -> None:
        """Apply ``eigen_sort``'s active-state count for the next SCF step.

        PARSEC first counts each representation among the lowest
        ``N_states-1`` globally sorted values, then admits at most ``nadd``
        additional values per representation from the remainder.  Its
        eigenspace allocation is not shrunk, but ``nn`` limits later work.
        Here a device view of the leading Ritz columns is the equivalent.
        """

        if safety_buffer < 1:
            return
        _, representations, _ = self._global_order(results)
        base = np.bincount(
            representations[: max(0, requested_states - 1)],
            minlength=self.representation_count,
        )
        extra = np.zeros(self.representation_count, dtype=np.int64)
        for representation in representations[max(0, requested_states - 1) :]:
            index = int(representation)
            if extra[index] < safety_buffer:
                extra[index] += 1
            if np.all(extra >= safety_buffer):
                break
        desired = np.maximum(1, base + extra)
        context = getattr(self, "mpi_context", None)
        changes = []
        for representation, solver in enumerate(self._solvers):
            new_count = min(counts[representation], int(desired[representation]))
            if new_count < counts[representation]:
                # truncate_state creates only leading-column views; it does
                # not launch a kernel and is therefore device-context neutral.
                if context is None:
                    solver.truncate_state(new_count)
                else:
                    changes.append((representation, new_count))
                counts[representation] = new_count
        if context is not None and changes:
            context.execute(self, "trim", changes)

    def _pack_selected_wedge_vectors(
        self,
        results: list[CuPyEigvalResult],
        selected_representations: np.ndarray,
        selected_columns: np.ndarray,
    ) -> CuPySymmetryOrbitals:
        """Retain sector views instead of allocating a dense packed wedge.

        The old representation packed every selected orbital into a
        ``wedge_size x requested_states`` array.  That duplicated all sector
        Ritz vectors solely to sum their squared magnitudes for the density.
        Keeping the already resident sector arrays and their global-column
        map removes that copy; :class:`CuPySymmetryDensityBuilder` performs
        the identical representation-wise sum.
        """

        return CuPySymmetryOrbitals(
            scaled_wedge_vectors=None,
            representations=np.ascontiguousarray(
                selected_representations, dtype=np.int32
            ),
            full_to_wedge=self.decomposition.reduction.full_to_wedge,
            # The expansion maps stay on the host until signed full-grid
            # states are materialized; see CuPySymmetryOrbitals.
            device_full_to_wedge=None,
            phases=self.decomposition.phases,
            full_size=self.decomposition.full_size,
            representation_columns=np.ascontiguousarray(
                selected_columns, dtype=np.int32
            ),
            sector_vectors=tuple(result.vectors for result in results),
            # The static orbit lists and scales formed at construction are
            # handed out as they are.  Sectors solved by another MPI rank
            # have none: a rank's density command selects only its own
            # sectors, and the root's density hook reads just the selection.
            sector_orbits=self._sector_orbits,
            sector_scales=self._sector_scales,
            wedge_size=self.decomposition.wedge_size,
            remote_sectors=(getattr(self, "mpi_context", None) is not None
                            and self.mpi_context.size > 1),
        )

    def __call__(
        self,
        operator: Any,
        requested_states: int,
        *,
        settings: EigvalSettings,
        state: object | None = None,
    ) -> CuPySymmetryEigvalResult:
        """Solve all sectors, globally sort them, and expand selected states."""

        if operator is not self.full_operator:
            raise ValueError("symmetry SCF received a different full Hamiltonian")
        if state is None:
            self.reset()
        elif state is not self._state:
            raise ValueError("SCF state does not belong to this symmetry eigensolver")

        requested_states = int(requested_states)
        if requested_states < 1:
            raise ValueError("requested_states must be positive")
        if self._sector_counts is None:
            self._sector_counts = [
                self._initial_sector_count(
                    representation,
                    requested_states,
                    settings.safety_buffer,
                )
                for representation in range(self.representation_count)
            ]
        counts = self._sector_counts
        if self.mpi_context is None:
            self._configure_large_problem_allocator(counts)
        else:
            self.mpi_context.execute(self, "configure_memory", counts)

        # CuPyHamiltonianBackend.bind has already retained this exact host
        # field on the full backend.  It is invariant by construction after
        # symmetry-sector densities are used; orbit averaging removes only
        # roundoff before the wedge upload.
        full_potential = (
            np.asarray(self.full_operator.effective_potential.get(), dtype=np.float64)
            if self._local_potential_getter is None
            else self._local_potential_getter()
        )
        if self.mpi_context is None:
            self._update_local_potential(full_potential)
        else:
            self.mpi_context.update_potential(self, full_potential)

        previous = state is not None
        if previous and len(self.device_ids) > 1:
            # Two A40s otherwise exhaust VRAM on the 1625-atom example.
            # Keep the established single-device allocation/reuse pattern:
            # early release did not demonstrate a one-GPU speed benefit.
            state.release_sector_snapshots()
        results, restarted, reasons = self._solve_sectors(
            settings, counts, previous
        )
        results, bracket_restarted, bracket_reasons = self._ensure_spectral_bracket(
            results, counts, requested_states, settings
        )
        restarted = restarted or bracket_restarted
        reasons.extend(bracket_reasons)

        values, representations, columns = self._global_order(results)
        selected_values = np.asarray(values[:requested_states], dtype=np.float64)
        selected_representations = representations[:requested_states]
        selected_columns = columns[:requested_states]
        vectors = self._pack_selected_wedge_vectors(
            results, selected_representations, selected_columns
        )

        residual_norms = None
        if all(result.residual_norms is not None for result in results):
            residual_norms = np.asarray(
                [
                    results[int(rep)].residual_norms[int(column)]
                    for rep, column in zip(
                        selected_representations,
                        selected_columns,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )

        self._trim_sector_states_like_parsec(
            results,
            counts,
            requested_states,
            settings.safety_buffer,
        )

        solves_completed = 1 if self._state is None else self._state.solves_completed + 1
        self._state = CuPySymmetryEigvalState(
            requested_states=requested_states,
            sector_state_counts=tuple(counts),
            sector_states=(
                tuple(None for _ in self._solvers)
                if getattr(self, "_host_spill_sector_states", False)
                else tuple(None if solver is None else solver.device_state for solver in self._solvers)
            ),
            solves_completed=solves_completed,
        )
        paths = {result.solver_path for result in results}
        solver_path = paths.pop() if len(paths) == 1 else "mixed"
        return CuPySymmetryEigvalResult(
            eigenvalues=selected_values,
            vectors=vectors,
            residual_norms=residual_norms,
            state=self._state,
            solver_path=solver_path,
            restarted=restarted,
            restart_reason="; ".join(reasons) if reasons else None,
            representations=np.asarray(selected_representations, dtype=np.int32),
            representation_columns=np.asarray(selected_columns, dtype=np.int32),
        )


__all__ = [
    "CuPySymmetryEigvalResult",
    "CuPySymmetryEigvalState",
    "CuPySymmetryOrbitals",
    "CuPySymmetrySCFEigensolver",
    "selected_device_ids",
]
