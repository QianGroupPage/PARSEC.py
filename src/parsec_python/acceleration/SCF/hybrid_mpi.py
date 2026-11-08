"""The Fock loop of a hybrid functional on the GPUs of several MPI ranks.

The MPI runner (``benchmarks/mpi_full_scf.py``) divides the symmetry sectors
among its ranks (:class:`~parsec_python.acceleration.experimental.mpi_scf.MPISectorContext`);
only the root runs the SCF loop, the others answer its commands.  Here the
root runs :func:`parsec_python.SCF.hybrid.run_hybrid_scf` unchanged with an
engine (:class:`MPIFockEngine`) whose every piece is a command ``exx_*`` that
all ranks execute together (:class:`MPIExchangeService`).  The loop decides
alone, on the root, from scalars that the ranks hand it: the stop rule, the
inner criterion and the outer DIIS take the same decisions as in one process.

The physics is that of :mod:`..Hamiltonian.exchange_device` (units Ry and
bohr, ``q = sqrt(h^3) psi``, ``W = V_x Psi``, ``M = -Psi^T W = L L^T``,
``xi = W L^-T``); what this module adds is where the arrays are:

- every rank keeps the sector orbitals of its last solve (both channels of a
  spin-polarized loop) on its devices; a channel of the loop is a handle and
  the global selection of its states (sector, column, occupation);
- a build of ``W`` makes the blocks and tiles of one process
  (:func:`~..Hamiltonian.exchange_device.plan_exchange`), assigns the tiles to
  the devices of all ranks, the most expensive first, each to the least loaded
  device (:func:`~..Hamiltonian.exchange_device.assign_tiles`, rank-major
  device order), and broadcasts from its owner the columns of every sector
  that the tiles read: a rank holds a replica of the other ranks' occupied
  orbitals (and of their empty states for a build of all columns) on its
  least loaded device;
- each rank runs its tiles with the kernels of one process (reflection-parity
  classes or the full grid); the parts of ``W`` of another rank's sectors are
  sent to their owner, which adds them in rank order
  (:meth:`ExchangeTransport.reduce_to_owners`);
- ``M``, its Cholesky factor and ``xi_G`` are formed on the owner of each
  sector, where ``xi_G`` stays and enters the sector operator
  (``exx_bind``); the energies of the loop (``E_x``, ``E_x(P, P0)``) are sums
  over the owners' sectors that the root adds in sector order, as one process
  does;
- the window of the outer DIIS keeps its copies on the owners
  (:class:`DistributedOccupiedBlock`); the small matrices of the extrapolation
  come to the root.

Arrays travel as device buffers where the MPI library takes them
(``MPICH_GPU_SUPPORT_ENABLED=1``, which the runner sets for a hybrid input
before ``MPI_Init``) and through the host otherwise
(``PARSEC_MPI_EXCHANGE_TRANSPORT``: ``auto``, ``cuda`` or ``host``), in
chunks of ``PARSEC_MPI_EXCHANGE_CHUNK_BYTES`` (256 MiB).

Every sector basis lies on one device of its owner (the exchange term is not
held by a basis that several devices share), and a rank may own no sector: it
then solves pair tiles only.  Nothing here prints or opens files.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import itertools
import os
from time import perf_counter
from types import SimpleNamespace
from typing import Any
import weakref

import numpy as np

from parsec_python.SCF.fock_diis import OccupiedBlock, orthonormal_complement
from parsec_python.V_xc.exact_exchange import OCCUPATION_THRESHOLD

from ..Hamiltonian.exchange_device import (
    ChannelOrbitals,
    DeviceCompressedExchange,
    DeviceExchangeApplication,
    OrbitalSector,
    SectorExpansion,
    assign_tiles,
    compress_device_exchange,
    parity_context,
    parity_tile_bytes,
    plan_exchange,
    run_exchange_tiles,
    sector_exchange_energies,
    touched_blocks,
    _TILE_ARRAYS,
)
from .hybrid import DeviceExchange, DeviceFockEngine, DeviceFockOperator

#: The tag of the exchange's point-to-point messages (on a communicator of their own).
_TAG = 3141


def exchange_transport() -> str:
    """``PARSEC_MPI_EXCHANGE_TRANSPORT``: ``cuda`` or ``host`` (``auto``, the default, decides).

    ``auto`` hands device arrays to MPI where ``MPICH_GPU_SUPPORT_ENABLED``
    is ``1`` and stages them through the host otherwise.
    """

    value = os.environ.get("PARSEC_MPI_EXCHANGE_TRANSPORT", "auto").strip().lower() or "auto"
    if value not in {"auto", "cuda", "host"}:
        raise ValueError("PARSEC_MPI_EXCHANGE_TRANSPORT must be auto, cuda or host")
    if value == "auto":
        return "cuda" if os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0").strip() == "1" else "host"
    return value


def exchange_overlap() -> bool:
    """``PARSEC_MPI_EXCHANGE_OVERLAP``: the tiles of a rank's own sectors run during the broadcasts (``1``, default).

    ``0`` broadcasts first and runs every tile after (the same sums).
    """

    value = os.environ.get("PARSEC_MPI_EXCHANGE_OVERLAP", "1").strip().lower() or "1"
    if value not in {"0", "1"}:
        raise ValueError("PARSEC_MPI_EXCHANGE_OVERLAP must be 0 or 1")
    return value == "1"


def exchange_chunk_bytes() -> int:
    """``PARSEC_MPI_EXCHANGE_CHUNK_BYTES``: the largest message of the exchange (default 256 MiB)."""

    value = int(os.environ.get("PARSEC_MPI_EXCHANGE_CHUNK_BYTES", str(256 << 20)).strip())
    if value < 8:
        raise ValueError("PARSEC_MPI_EXCHANGE_CHUNK_BYTES must be at least 8")
    return value


def column_runs(columns) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` runs of sorted column indices."""

    columns = np.asarray(columns, dtype=np.int64)
    if not columns.size:
        return []
    breaks = np.flatnonzero(np.diff(columns) != 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [columns.size]))
    return [(int(columns[a]), int(columns[b - 1]) + 1) for a, b in zip(starts, stops)]


class _BareBuffers:
    """What the transport takes of ``mpi4py.MPI`` where mpi4py is missing (buffers without datatypes)."""

    DOUBLE = None

    class Request:
        @staticmethod
        def Waitall(requests) -> None:
            for request in requests:
                request.Wait()


class ExchangeTransport:
    """Broadcasts and owner sums of F-contiguous ``rows x columns`` FP64 device blocks.

    ``kind`` is ``cuda`` (device buffers to MPI) or ``host`` (staging);
    messages hold whole columns, at most :func:`exchange_chunk_bytes` (one
    column where a column is more).  ``seconds`` and ``bytes`` count what this
    rank spent and moved.
    """

    def __init__(self, comm, kind: str | None = None) -> None:
        self.comm = comm
        self.kind = exchange_transport() if kind is None else str(kind)
        self.chunk_bytes = exchange_chunk_bytes()
        self.seconds = {"broadcast": 0.0, "reduce": 0.0}
        self.bytes = {"broadcast_sent": 0, "broadcast_received": 0, "reduce_sent": 0, "reduce_received": 0}
        self._pinned = None

    def _chunks(self, block):
        rows, columns = int(block.shape[0]), int(block.shape[1])
        step = max(1, self.chunk_bytes // max(1, 8 * rows))
        for start in range(0, columns, step):
            yield block[:, start : start + step]

    def _host_buffer(self, size: int):
        import cupyx

        if self._pinned is None or self._pinned.size < size:
            self._pinned = cupyx.empty_pinned((max(int(size), self.chunk_bytes // 8),), dtype=np.float64)
        return self._pinned[:size]

    @staticmethod
    def _mpi():
        """mpi4py's ``MPI``; a stand-in where it is missing (a communicator that takes bare buffers)."""

        try:
            from mpi4py import MPI
        except ImportError:  # the thread communicators of the tests
            return _BareBuffers
        return MPI

    @staticmethod
    def _message(MPI, buffer):
        return buffer if MPI.DOUBLE is None else [buffer, MPI.DOUBLE]

    def broadcast_columns(self, rows: int, count: int, root: int, *, device: int, target=None, fill=None) -> None:
        """Broadcast ``count`` columns of ``rows`` FP64 values from rank ``root``, a chunk at a time.

        The root hands each chunk by ``fill(start, stop)`` (an F-contiguous
        device block of those columns, made only for the chunk), the other ranks
        receive it into ``target[:, start:stop]`` (an F-contiguous device block
        of ``count`` columns).  When this returns, what the ranks received is in
        place for every stream of ``device``.
        """

        import ctypes

        import cupy as cp

        MPI = self._mpi()
        started = perf_counter()
        rank = self.comm.rank
        step = max(1, self.chunk_bytes // max(1, 8 * int(rows)))
        with cp.cuda.Device(int(device)):
            stream = cp.cuda.get_current_stream()
            for start in range(0, int(count), step):
                stop = min(int(count), start + step)
                if rank == root:
                    chunk = fill(start, stop)
                    # (written on this stream; MPI reads it from the device)
                    stream.synchronize()
                else:
                    chunk = target[:, start:stop]
                if not chunk.flags.f_contiguous:
                    raise ValueError("an exchange message must be an F-contiguous block of columns")
                if self.kind == "cuda":
                    self.comm.Bcast(self._message(MPI, chunk), root=root)
                else:
                    host = self._host_buffer(chunk.size)
                    pointer = ctypes.c_void_p(host.ctypes.data)
                    if rank == root:
                        chunk.data.copy_to_host(pointer, chunk.nbytes)
                    self.comm.Bcast(self._message(MPI, host), root=root)
                    if rank != root:
                        chunk.data.copy_from_host(pointer, chunk.nbytes)
                key = "broadcast_sent" if rank == root else "broadcast_received"
                self.bytes[key] += int(chunk.nbytes)
                del chunk
            # Whatever wrote the received columns on this stream is done before a reader on another
            # stream of the device may look at them.
            stream.synchronize()
        self.seconds["broadcast"] += perf_counter() - started

    def broadcast(self, block, root: int) -> None:
        """Broadcast the device block ``block`` (the same shape on every rank) from rank ``root``."""

        self.broadcast_columns(
            int(block.shape[0]), int(block.shape[1]), root, device=int(block.device.id),
            target=block, fill=lambda start, stop: block[:, start:stop],
        )

    def reduce_to_owners(self, sends, receives) -> None:
        """Send blocks of columns to their owners and add what the owners receive.

        ``sends`` are ``(destination, block, columns)``: the columns of a device
        block that go to ``destination``; ``receives`` are ``(source, target,
        columns)``: ``target[:, columns] += `` what ``source`` sent, in the
        order of the list.  The messages between two ranks are taken in the
        order they were sent, so both lists name the sectors of a pair of ranks
        in the same order.
        """

        import cupy as cp
        import cupyx

        MPI = self._mpi()
        started = perf_counter()
        requests = []
        keep = []
        for destination, block, columns in sends:
            with cp.cuda.Device(int(block.device.id)):
                cp.cuda.Device().synchronize()
                for start, stop in column_runs(columns):
                    for chunk in self._chunks(block[:, start:stop]):
                        if self.kind == "cuda":
                            message = chunk
                        else:
                            # Page-locked staging (CuPy's pinned pool): the copy runs at the bus rate.
                            message = cupyx.empty_pinned(chunk.shape, dtype=np.float64, order="F")
                            chunk.get(out=message)
                        keep.append(message)
                        requests.append(self.comm.Isend(self._message(MPI, message), dest=int(destination), tag=_TAG))
                        self.bytes["reduce_sent"] += int(chunk.nbytes)
        for source, target, columns in receives:
            with cp.cuda.Device(int(target.device.id)):
                for start, stop in column_runs(columns):
                    for chunk in self._chunks(target[:, start:stop]):
                        if self.kind == "cuda":
                            buffer = cp.empty(chunk.shape, dtype=cp.float64, order="F")
                            cp.cuda.Device().synchronize()
                            self.comm.Recv(self._message(MPI, buffer), source=int(source), tag=_TAG)
                        else:
                            host = cupyx.empty_pinned(chunk.shape, dtype=np.float64, order="F")
                            self.comm.Recv(self._message(MPI, host), source=int(source), tag=_TAG)
                            buffer = cp.asarray(host, order="F")
                        chunk += buffer
                        self.bytes["reduce_received"] += int(chunk.nbytes)
                cp.cuda.Device().synchronize()
        if all(isinstance(request, MPI.Request) for request in requests):
            MPI.Request.Waitall(requests)
        else:  # requests of a communicator that is not mpi4py's (the tests)
            for request in requests:
                request.Wait()
        del keep
        self.seconds["reduce"] += perf_counter() - started


# ---------------------------------------------------------------------------
# What every rank holds.
# ---------------------------------------------------------------------------


@dataclass
class _Channel:
    """A channel of the loop on one rank: the orbitals of its own sectors and replicas of the others.

    ``orbitals.sectors`` are in the global sector order of one process; a
    sector of another rank has ``vectors`` ``None`` until a build ships its
    columns (``shipped`` lists the sector positions its replica holds, the
    same on every rank) and ``local`` maps sector positions to replica
    columns (``-1``: not shipped).
    """

    orbitals: ChannelOrbitals
    owners: tuple[int, ...]
    shipped: dict = field(default_factory=dict)


class ExchangeAgreementError(RuntimeError):
    """A rank could not go on with an exchange command; every rank raises it at the same point."""


class _SoftMemory(Exception):
    """The exchange does not fit the devices of some rank: the root raises MemoryError (DIIS stops)."""


class MPIExchangeService:
    """The ``exx_*`` commands of one rank (see the module text).

    ``exchange`` is the rank's :class:`~.hybrid.DeviceExchange` (its devices,
    sector expansions, pair solvers), ``sectors`` its symmetry eigensolver,
    ``grid`` and ``definition`` those of the hybrid.
    """

    def __init__(self, context, exchange: DeviceExchange, sectors, grid, definition) -> None:
        self.context = context
        self.comm = context.comm
        self.rank, self.size = int(self.comm.rank), int(self.comm.size)
        self.exchange = exchange
        self.sectors = sectors
        self.decomposition = sectors.decomposition
        self.grid = grid
        self.definition = definition
        self.devices = tuple(int(device) for device in exchange.devices)
        self.global_devices = tuple(
            (rank, int(device))
            for rank, devices in enumerate(self.comm.allgather(self.devices))
            for device in devices
        )
        self.owners = {}
        for rank, groups in enumerate(context.topology):
            for representation in groups:
                self.owners[int(representation)] = rank
        if set(self.owners) != set(range(self.decomposition.representation_count)):
            raise ValueError("every symmetry sector of a hybrid needs an owner rank")
        self.transport = ExchangeTransport(self.comm.Dup())
        self.objects: dict[int, Any] = {}
        # The operator bound into the sector operators of this rank (exx_bind).
        self.bound = None
        # Wall seconds and counts of the phases of this rank (timing.json).
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.pair_solves = 0
        self.tile_seconds = 0.0
        # The pair solvers of the hybrid (their host plans on a thread beside the start).
        self.solvers = exchange.pair_solvers(SimpleNamespace(grid=grid), definition)

    # ----- bookkeeping -------------------------------------------------------

    def _timed(self, name: str, started: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + perf_counter() - started

    def record(self) -> dict:
        """What this rank spent on the exchange (seconds by phase, bytes moved, tiles)."""

        return dict(
            transport=self.transport.kind,
            chunk_bytes=self.transport.chunk_bytes,
            devices=list(self.devices),
            seconds=dict(self.seconds),
            commands=dict(self.counts),
            transport_seconds=dict(self.transport.seconds),
            transport_bytes=dict(self.transport.bytes),
            pair_solves=int(self.pair_solves),
            tile_solve_seconds=float(self.tile_seconds),
            # The build of the host plans of the pair solver on its thread (0 before it ended).
            solver_setup_seconds=float(getattr(self.solvers, "setup_seconds", 0.0)),
        )

    def _settle(self, error: str | None = None, memory: str | None = None) -> None:
        """Agree on the outcome of a rank's local work before the next collective call of a command.

        One ``allgather``: where any rank failed (``error``) every rank raises
        :class:`ExchangeAgreementError` here, where any rank ran out of device
        memory (``memory``) every rank raises :class:`_SoftMemory` (the root then
        raises ``MemoryError``, as one process does).  No rank is left in a
        collective call the others never reach.
        """

        outcomes = self.comm.allgather((error, memory))
        failed = [f"rank {rank}: {item[0]}" for rank, item in enumerate(outcomes) if item[0]]
        if failed:
            raise ExchangeAgreementError("; ".join(failed))
        short = [f"rank {rank}: {item[1]}" for rank, item in enumerate(outcomes) if item[1]]
        if short:
            raise _SoftMemory("; ".join(short))

    @staticmethod
    def _outcome(step, *arguments):
        """``(value, error, memory)`` of a rank's local ``step`` (no communication inside)."""

        try:
            return step(*arguments), None, None
        except MemoryError as exc:  # cupy.cuda.memory.OutOfMemoryError is one
            return None, None, f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}", None

    def _local(self, step, *arguments):
        """Run a rank's local ``step`` and agree on its outcome with every rank; its value."""

        value, error, memory = self._outcome(step, *arguments)
        self._settle(error, memory)
        return value

    def _collective(self, step, *arguments):
        """Run a step whose collective calls every rank must reach: a rank that fails there ends the job.

        The others would wait for it in a broadcast or a receive; the MPI
        runner's abort reaches every rank, a hang does not.  Only the
        communication itself runs here: what can fail before it (allocations,
        the data a rank sends) is agreed first (:meth:`_settle`).
        """

        try:
            return step(*arguments)
        except BaseException:
            import sys
            import traceback

            print(f"EXCHANGE_COLLECTIVE_FAILED rank={self.rank}", file=sys.stderr, flush=True)
            traceback.print_exc()
            self.comm.Abort(1)
            raise

    def _expansion(self, representation: int) -> SectorExpansion:
        expansions = self.exchange.expansions
        expansion = expansions.get(representation)
        if expansion is None:
            decomposition = self.decomposition
            orbits = decomposition.sector_orbit_indices(representation)
            scales = 1.0 / np.sqrt(decomposition.reduction.multiplicities[orbits])
            expansion = SectorExpansion(
                decomposition.reduction.full_to_wedge,
                np.asarray(decomposition.phases[representation]),
                orbits,
                scales,
                int(decomposition.wedge_size),
            )
            expansions[representation] = expansion
        return expansion

    # ----- dispatch ------------------------------------------------------------

    def handle(self, command: str, payload: dict):
        """Execute ``command`` on this rank; what the root returns to the engine (``None`` elsewhere)."""

        for key in payload.get("free", ()):
            self._release(self.objects.pop(int(key), None))
        method = getattr(self, "_" + command[len("exx_"):], None)
        if method is None:
            raise ValueError(f"unknown exchange command: {command}")
        self.counts[command] = self.counts.get(command, 0) + 1
        started = perf_counter()
        try:
            return method(payload)
        except _SoftMemory as problem:
            return {"memory_error": str(problem)}
        finally:
            self._timed(command, started)

    @staticmethod
    def _release(item) -> None:
        if isinstance(item, _Channel):
            for sector in item.orbitals.sectors:
                sector.vectors = None
        elif isinstance(item, DeviceCompressedExchange):
            item.release()

    def _free(self, payload):
        """Free what the root names now (``ids``)."""

        for key in payload.get("ids", ()):
            self._release(self.objects.pop(int(key), None))

    def _trim(self, payload):
        """Let everything of the last Fock loop go (an SCF has ended) and return the freed pool blocks.

        Channels with their replicas, applications and window blocks; the
        operator of a loop was released by the loop itself, and an operator
        still bound into the sector operators stays (the root holds it).  With
        ``final`` (the run leaves this exchange: the spin-sector fallback) the
        sector operators lose the exchange term, the bound operator goes as
        well, and so do the device plans of the pair solvers.
        """

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        final = bool(payload.get("final"))
        if final:
            for operator in self.exchange.operators().values():
                operator.bind_exchange(None)
            bound, self.bound = self.bound, None
            self._release(bound)
        for key in list(self.objects):
            item = self.objects[key]
            if isinstance(item, DeviceCompressedExchange) and not item.released and self.bound is item:
                continue
            self._release(self.objects.pop(key))
        if final:
            self.solvers.release_blocks()
            self.solvers._devices.clear()
            self.exchange._solvers.clear()
            self.exchange.expansions.clear()
        for device in self.devices:
            with cp.cuda.Device(device):
                cp.cuda.Device().synchronize()
                cp.get_default_memory_pool().free_all_blocks()

    # ----- channels --------------------------------------------------------------

    def _retained(self, channel):
        """The sector results of the last solve of ``channel`` (``None``: an unpolarized loop) on this rank."""

        solver = self.sectors
        index = 0 if channel is None else int(channel)
        if getattr(solver, "_spin_channels", None) is None:
            if index != 0:
                raise RuntimeError("this rank serves no second spin channel")
            return solver._mpi_retained_results
        return solver._channel_value(index, "_mpi_retained_results")

    def _local_device(self, loads: dict, nbytes: int) -> int:
        device = min(self.devices, key=lambda name: (loads[name], self.devices.index(name)))
        loads[device] += int(nbytes)
        return device

    def _build_channel(self, selection: dict) -> _Channel:
        """A :class:`_Channel` of the states a selection names (from the retained results, or given vectors)."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        representations = np.asarray(selection["representations"], dtype=np.int64)
        columns = np.asarray(selection["columns"], dtype=np.int64)
        occupations = np.asarray(selection["occupations"], dtype=np.float64)
        retained = None
        sectors = []
        owners = []
        loads = {device: 0 for device in self.devices}
        device_ids = getattr(self.sectors, "_sector_device_ids", ())
        for representation in sorted(set(int(value) for value in representations)):
            if self.owners[representation] == self.rank:
                device = int(device_ids[representation])
                loads[device] = loads.get(device, 0) + 8 * self.decomposition.sector_size(representation) * int(
                    np.count_nonzero(representations == representation)
                )
        for representation in sorted(set(int(value) for value in representations)):
            selected = np.flatnonzero(representations == representation)
            owner = self.owners[representation]
            expansion = self._expansion(representation)
            if owner == self.rank:
                if retained is None:
                    retained = self._retained(selection.get("channel"))
                result = retained.get(representation)
                vectors = None if result is None else result.vectors
                if vectors is None:
                    raise RuntimeError(
                        f"rank {self.rank} holds no orbitals of sector {representation + 1} for the exchange"
                    )
                if hasattr(vectors, "group"):
                    raise NotImplementedError(
                        "the exchange operator of a hybrid functional needs every sector basis on one device "
                        "(a basis shared by several devices is not supported)"
                    )
                handover = getattr(result, "handover", None)
                if handover is not None and not getattr(handover, "rotated", True):
                    raise NotImplementedError(
                        "the exchange operator of a hybrid functional needs Ritz vectors; the triangular Ritz "
                        "hand-over (PARSEC_CUPY_RITZ_HANDOVER) is not supported with it"
                    )
                device = int(vectors.device.id) if isinstance(vectors, cp.ndarray) else int(device_ids[representation])
                local = np.asarray(columns[selected], dtype=np.int64)
            else:
                vectors = None
                device = self._local_device(loads, 8 * expansion.rows * selected.size)
                local = np.full(selected.size, -1, dtype=np.int64)
            sectors.append(
                OrbitalSector(
                    representation=representation,
                    device=device,
                    vectors=vectors,
                    local=local,
                    states=selected.astype(np.int64),
                    expansion=expansion,
                )
            )
            owners.append(owner)
        orbitals = ChannelOrbitals(
            tuple(sectors),
            occupations,
            float(selection["spin_weight"]),
            int(self.decomposition.full_size),
            float(selection["volume_element"]),
        )
        return _Channel(orbitals, tuple(owners))

    def _channel(self, payload):
        """Register the channels ``payload['channels']`` (``id`` and selection each)."""

        for selection in payload["channels"]:
            self.objects[int(selection["id"])] = self._build_channel(selection)

    def _drop(self, payload):
        """Let the orbitals of channels go (their replicas included)."""

        for key in payload["ids"]:
            item = self.objects.pop(int(key), None)
            self._release(item)

    # ----- W = V_x Psi -----------------------------------------------------------

    @staticmethod
    def _missing(entry: _Channel, wanted: dict, index: int) -> np.ndarray:
        have = entry.shipped.get(index, np.zeros(0, dtype=np.int64))
        return np.setdiff1d(np.asarray(wanted.get(index, ()), dtype=np.int64), have)

    def _prepare_replicas(self, entry: _Channel, wanted: dict) -> list:
        """The replica arrays a build fills, made before any tile or broadcast runs.

        One record ``(index, replica, start, missing)`` per sector that lacks
        positions ``missing``: a sector of this rank (``replica`` ``None``: it
        sends them) or another rank's, whose new F-ordered replica holds its
        present columns first (copied here) and receives ``missing`` from column
        ``start`` on.  Every device is synchronized at the end, so the copies are
        complete before a tile on another stream may read them.
        """

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        records = []
        if self.size == 1:
            return records
        for index, sector in enumerate(entry.orbitals.sectors):
            missing = self._missing(entry, wanted, index)
            if not missing.size:
                continue
            if entry.owners[index] == self.rank:
                records.append((index, None, 0, missing))
                continue
            rows = int(sector.expansion.rows)
            current = 0 if sector.vectors is None else int(sector.vectors.shape[1])
            with cp.cuda.Device(sector.device):
                replica = cp.empty((rows, current + int(missing.size)), dtype=cp.float64, order="F")
                if current:
                    replica[:, :current] = sector.vectors
            records.append((index, replica, current, missing))
        for device in self.devices:
            cp.cuda.Device(device).synchronize()
        return records

    def _broadcast_replicas(self, entry: _Channel, records: list, arrived: dict | None = None) -> None:
        """Broadcast the ``records`` of :meth:`_prepare_replicas` from their owners, in sector order.

        ``arrived[index]`` (a :class:`threading.Event`), where given, is set once
        the replica of sector ``index`` holds its new columns on every stream of
        its device and is in place.
        """

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        orbitals = entry.orbitals
        for index, replica, start, missing in records:
            sector = orbitals.sectors[index]
            owner = entry.owners[index]
            rows = int(sector.expansion.rows)
            have = entry.shipped.get(index, np.zeros(0, dtype=np.int64))
            if owner == self.rank:

                def fill(first, last, sector=sector, missing=missing):
                    block = sector.columns(cp, missing[first:last])
                    return block if block.flags.f_contiguous else cp.asfortranarray(block)

                self.transport.broadcast_columns(rows, int(missing.size), owner, device=sector.device, fill=fill)
            else:
                self.transport.broadcast_columns(
                    rows, int(missing.size), owner, device=sector.device, target=replica[:, start:]
                )
                local = np.array(sector.local, copy=True)
                local[missing] = np.arange(start, start + missing.size, dtype=np.int64)
                sector.vectors = replica
                sector.local = local
            entry.shipped[index] = np.concatenate([have, missing])
            if arrived is not None:
                arrived[index].set()

    def _plan(self, payload):
        """The parts of a build every rank makes alike: entry, known application, plan, parity, tiles.

        First the host plans of the pair solver, built on a thread since the
        preparation (a wait only at the first build): a failure of that thread
        on one rank is agreed on with the rest of this step.
        """

        started = perf_counter()
        self.solvers.parts
        self._timed("apply_solver_wait", started)
        entry = self.objects[int(payload["channel"])]
        orbitals = entry.orbitals
        known = None if payload.get("known") is None else self.objects[int(payload["known"])]
        plan = plan_exchange(orbitals, self.solvers.batch_size, columns=str(payload["columns"]), known=known)
        parity = parity_context(orbitals, self.solvers)
        assigned = assign_tiles(plan.tiles, self.global_devices)
        touched = {
            rank: touched_blocks(
                [tile for (owner, _device), work in assigned.items() if owner == rank for tile in work]
            )
            for rank in range(self.size)
        }
        return entry, known, plan, parity, assigned, touched

    def _allocate(self, entry, plan, known, parity, wanted, touched_sectors, owned):
        """The arrays of a build on this rank: owned ``W``, parts of the others' ``W``, replicas."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        orbitals = entry.orbitals
        self._require_memory(entry, wanted, touched_sectors, known, parity)
        if known is not None:
            applied = known.applied
        else:
            applied = [None] * len(orbitals.sectors)
            for index in owned:
                sector = orbitals.sectors[index]
                with cp.cuda.Device(sector.device):
                    applied[index] = cp.zeros(
                        (int(sector.expansion.rows), sector.local.size), dtype=cp.float64, order="F"
                    )
        partial = {}
        for index in sorted(touched_sectors):
            if entry.owners[index] == self.rank:
                continue
            sector = orbitals.sectors[index]
            with cp.cuda.Device(sector.device):
                partial[index] = cp.zeros((int(sector.expansion.rows), sector.local.size), dtype=cp.float64, order="F")
        records = self._prepare_replicas(entry, wanted)
        return applied, partial, records

    def _apply(self, payload):
        """``W`` of a channel on all ranks; the owners keep their sectors' ``W``."""

        import threading

        started = perf_counter()
        entry, known, plan, parity, assigned, touched = self._local(self._plan, payload)
        self.solvers.record_path("full" if parity is None else "parity")
        orbitals = entry.orbitals
        local = {device: assigned[(self.rank, device)] for device in self.devices}
        # Positions of each sector the tiles of this build read: the occupied
        # blocks and the wanted empty blocks (every rank receives the same).
        wanted: dict[int, list] = {}
        for block in plan.blocks:
            wanted.setdefault(int(block.sector), []).extend(int(value) for value in block.positions)
        owned = [index for index, owner in enumerate(entry.owners) if owner == self.rank]
        # W of the owned sectors; parts of the others where this rank's tiles touch them; the replicas.
        touched_sectors = {int(plan.blocks[index].sector) for index in touched[self.rank]}
        allocated, error, memory = self._outcome(
            self._allocate, entry, plan, known, parity, wanted, touched_sectors, owned
        )
        self._settle(error, memory)  # a refusal leaves the known application where it was, on every rank
        applied, partial, records = allocated
        if known is not None:
            self.objects.pop(int(payload["known"]))  # consumed by the completion (as in one process)
        self._timed("apply_plan", started)
        targets = [partial.get(index, applied[index]) for index in range(len(orbitals.sectors))]
        # With the overlap the devices start on their tiles while the main thread broadcasts the
        # orbitals: a tile waits for the replicas of its blocks' sectors, and each device takes its
        # tiles in the order the sectors arrive (its own sectors first, then the others' in sector
        # order), grouped by their first block within that.
        count = len(orbitals.sectors)
        arrived = {index: threading.Event() for index in range(count)}
        position = {index: -1 for index in range(count)}
        for index, replica, _start, _missing in records:
            if replica is not None:
                position[index] = index
        for index in range(count):
            if position[index] < 0:
                arrived[index].set()
        overlap = exchange_overlap() and self.size > 1

        def stage_of(tile):
            return max(position[plan.blocks[tile.first].sector], position[plan.blocks[tile.second].sector])

        def ready(tile) -> None:
            for block in (tile.first, tile.second):
                arrived[plan.blocks[block].sector].wait()

        ordered = {
            device: sorted(work, key=lambda tile: (stage_of(tile), tile.first, tile.second)) if overlap else work
            for device, work in local.items()
        }
        failure = error = None
        timing: dict = {}
        stage = perf_counter()
        running = None
        if overlap:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parsec-exx-tiles")
            running = pool.submit(
                run_exchange_tiles, orbitals, self.solvers, plan, ordered, targets, parity=parity, ready=ready
            )
            pool.shutdown(wait=False)
        try:
            self._collective(self._broadcast_replicas, entry, records, arrived)
        finally:
            for event in arrived.values():
                event.set()  # (a tile never waits for a broadcast that failed)
        self._timed("apply_replicas", stage)
        try:
            timing = (
                running.result()
                if running is not None
                else run_exchange_tiles(orbitals, self.solvers, plan, ordered, targets, parity=parity)
            )
        except MemoryError as exc:  # cupy.cuda.memory.OutOfMemoryError is one
            failure = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._settle(error, failure)  # (a completion that ran out of memory has consumed its known application)
        self._timed("apply_tiles", stage)
        self.pair_solves += int(sum(tile.pairs for work in local.values() for tile in work))
        self.tile_seconds += float(sum(timing.values()))
        stage = perf_counter()
        # The parts of the other ranks' sectors to their owners, in sector order per pair of ranks.
        positions_of = {}
        for rank, indices in touched.items():
            for block_index in indices:
                block = plan.blocks[block_index]
                positions_of.setdefault((rank, int(block.sector)), []).extend(int(v) for v in block.positions)
        sends = []
        for index in sorted(partial):
            positions = np.unique(np.asarray(positions_of[(self.rank, index)], dtype=np.int64))
            sends.append((entry.owners[index], partial[index], positions))
        sends.sort(key=lambda item: item[0])
        receives = []
        for index in owned:
            for rank in range(self.size):
                if rank == self.rank or (rank, index) not in positions_of:
                    continue
                positions = np.unique(np.asarray(positions_of[(rank, index)], dtype=np.int64))
                receives.append((rank, applied[index], positions))
        self._collective(self.transport.reduce_to_owners, sends, receives)
        del partial, sends
        self._timed("apply_reduce", stage)
        stage = perf_counter()
        energies = (
            self._local(sector_exchange_energies, orbitals, applied, set(owned)) if known is None else {}
        )
        gathered = self.comm.gather((energies, max(timing.values(), default=0.0)), root=0)
        application = DeviceExchangeApplication(
            orbitals=orbitals,
            applied=applied,
            columns=plan.done() if known is None else tuple(sorted(set(known.columns) | set(plan.done()))),
            occupied=plan.occupied,
            energy=0.0 if known is None else float(known.energy),
            pair_solves=int(plan.pair_solves + (0 if known is None else known.pair_solves)),
            poisson_seconds=0.0,
            seconds=0.0,
        )
        self.objects[int(payload["id"])] = application
        self._timed("apply_energy", stage)
        if self.rank != 0:
            return None
        energy = 0.0
        if known is None:
            values = {}
            for item, _seconds in gathered:
                values.update(item)
            for index in sorted(values):
                energy += values[index]
            energy *= 0.5 * orbitals.spin_weight
        return dict(
            energy=float(energy),
            pair_solves=int(plan.pair_solves),
            poisson_seconds=float(max(seconds for _item, seconds in gathered)),
            columns=application.columns,
            occupied=tuple(plan.occupied),
        )

    def _build_bytes(self, entry, wanted, touched_sectors, known, parity) -> dict[int, int]:
        """The device bytes a build holds on this rank, per device (:meth:`_require_memory`).

        The ``W`` of the owned sectors (a build of the occupied columns or of
        all), a new replica of every other rank's sector that lacks columns
        (its present columns and the new ones; the old one is freed after the
        build), the part of ``W`` of the other ranks' sectors the tiles touch,
        the messages of the owned sectors, and the work of one tile on every
        device.  A message of an owned sector holds two chunks at once: the
        columns it broadcasts are gathered and then copied to F order, and a
        sum of ``W`` it receives lands in a new buffer while the last one is
        still held.  The messages of a device go one at a time, so its largest
        owned sector counts.
        """

        orbitals = entry.orbitals
        needed = {device: 0 for device in self.devices}
        messages = {device: 0 for device in self.devices}
        chunk = int(self.transport.chunk_bytes)
        for index, sector in enumerate(orbitals.sectors):
            rows = int(sector.expansion.rows)
            device = int(sector.device)
            missing = self._missing(entry, wanted, index) if self.size > 1 else np.zeros(0)
            if entry.owners[index] == self.rank:
                if known is None:
                    needed[device] = needed.get(device, 0) + 8 * rows * int(sector.local.size)
                if self.size > 1:
                    # The columns it sends, or the sums of W it receives (at most every column of W).
                    columns = max(int(missing.size), int(sector.local.size))
                    message = 2 * min(8 * rows * columns, max(chunk, 8 * rows))
                    messages[device] = max(messages.get(device, 0), message)
                continue
            if missing.size:
                current = 0 if sector.vectors is None else int(sector.vectors.shape[1])
                needed[device] = needed.get(device, 0) + 8 * rows * (current + int(missing.size))
            if index in touched_sectors:
                needed[device] = needed.get(device, 0) + 8 * rows * int(sector.local.size)
        if parity is None:
            work = 8 * int(self.solvers.batch_size) * _TILE_ARRAYS * int(orbitals.full_size)
        else:
            work = parity_tile_bytes(
                coset_rows=parity.coset_rows,
                octant_points=int(parity.layout.size),
                octant_box=int(parity.layout.half_size),
                classes=len(parity.group.cosets),
                batch_size=int(self.solvers.batch_size),
                densities_per_solve=parity.densities_per_solve,
                box_work=False,
            )
        for device in self.devices:
            needed[device] = needed.get(device, 0) + work
        for device, message in messages.items():
            needed[device] = needed.get(device, 0) + message
        return needed

    def _require_memory(self, entry, wanted, touched_sectors, known, parity) -> None:
        """Refuse, with the numbers, a build whose arrays (:meth:`_build_bytes`) do not fit on this rank."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        needed = self._build_bytes(entry, wanted, touched_sectors, known, parity)
        for device, count in sorted(needed.items()):
            with cp.cuda.Device(device):
                free, total = cp.cuda.runtime.memGetInfo()
                available = int(free) + int(cp.get_default_memory_pool().free_bytes())
            if count > available:
                raise MemoryError(
                    f"the exact exchange of this hybrid needs {count / 2**30:.1f} GiB on device {device} of rank "
                    f"{self.rank} (W of its sectors, the replicas of the other ranks' orbitals, their parts of W and "
                    f"the tile work) and {available / 2**30:.1f} GiB of {int(total) / 2**30:.1f} GiB are free there; "
                    "run it on more nodes, with fewer States_Num, or a smaller Exx_Batch_Size"
                )

    # ----- ACE -------------------------------------------------------------------

    def _compress_application(self, application, fraction: float, owned):
        """``compress_device_exchange`` of the owned sectors (a method: the tests make one rank fail here)."""

        return compress_device_exchange(application, fraction=fraction, sectors=owned)

    def _compress(self, payload):
        """``xi_G`` of the owned sectors of applications (the applications are consumed)."""

        from dataclasses import replace as _replace

        def compress():
            # The operators join ``objects`` only once every rank made its own: where one rank
            # fails, the root tracks none of them and no rank keeps one (as one process drops them).
            made, records = {}, []
            for item in payload["channels"]:
                application = self.objects.pop(int(item["application"]))
                owned = {
                    index
                    for index, sector in enumerate(application.orbitals.sectors)
                    if self.owners[sector.representation] == self.rank
                }
                operator = self._compress_application(application, float(payload["fraction"]), owned)
                # (E_x of the application is known on the root alone.)
                made[int(item["id"])] = _replace(operator, energy=float(item["energy"]))
                records.append((operator.asymmetry, operator.rank))
            return made, records

        made, records = self._local(compress)
        self.objects.update(made)
        gathered = self.comm.gather(records, root=0)
        if self.rank != 0:
            return None
        return [
            dict(
                asymmetry=max(rows[position][0] for rows in gathered),
                rank=int(sum(rows[position][1] for rows in gathered)),
            )
            for position in range(len(records))
        ]

    def _bind(self, payload):
        """Write ``a_x V_ACE`` of a channel (``id``; ``None`` removes it) into the sector operators of this rank."""

        key = payload.get("id")
        channel = None if key is None else self.objects.get(int(key))
        operators = self.exchange.operators()
        for representation, operator in operators.items():
            projectors = None if channel is None or channel.released else channel.projectors.get(representation)
            if projectors is None:
                operator.bind_exchange(None)
            else:
                operator.bind_exchange(projectors, -float(channel.fraction))
        # (The service's own record: on the root, the engine's bind keeps its handle on the exchange object.)
        self.bound = channel

    # ----- energies of an operator -----------------------------------------------

    def _band_sums(self, payload) -> list:
        """Per pair ``(operator, orbitals)`` of ``payload``, this rank's sector sums (local work only)."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        values = []
        for item in payload["pairs"]:
            channel = self.objects.get(int(item["operator"]))
            if "channel" in item:
                orbitals = self.objects[int(item["channel"])].orbitals
            else:
                orbitals = self._build_channel(item["selection"]).orbitals
            sums = {}
            if channel is not None and not channel.released:
                for index, sector in enumerate(orbitals.sectors):
                    xi = channel.projectors.get(sector.representation)
                    if xi is None or sector.vectors is None or self.owners[sector.representation] != self.rank:
                        continue
                    weights = orbitals.occupations[sector.states]
                    occupied = np.flatnonzero(weights > OCCUPATION_THRESHOLD)
                    if not occupied.size:
                        continue
                    with cp.cuda.Device(int(xi.device.id)):
                        overlaps = sector.products(cp, occupied, xi)
                        squares = cp.sum(overlaps * overlaps, axis=1)
                        sums[index] = float(cp.asnumpy(squares) @ weights[occupied])
            values.append(sums)
        return values

    def _band(self, payload):
        """Per pair ``(operator, orbitals)``, the sector sums of ``sum_i f_i |xi_G^T q_i|^2`` of this rank."""

        values = self._local(self._band_sums, payload)
        gathered = self.comm.gather(values, root=0)
        if self.rank != 0:
            return None
        merged = []
        for position in range(len(values)):
            sums = {}
            for rows in gathered:
                sums.update(rows[position])
            merged.append(sums)
        return merged

    # ----- the window of the outer DIIS --------------------------------------------

    def _owned_sets(self, keys) -> list:
        """The owned sectors of the channels ``keys`` as :class:`ChannelOrbitals` (the fit checks of the window)."""

        sets = []
        for key in keys:
            orbitals = self.objects[int(key)].orbitals
            owned = tuple(sector for sector in orbitals.sectors if self.owners[sector.representation] == self.rank)
            sets.append(ChannelOrbitals(owned, orbitals.occupations, orbitals.spin_weight,
                                        orbitals.full_size, orbitals.volume_element))
        return sets

    def _copy(self, payload):
        """Copies of the occupied columns of the owned sectors of channels (``blocks``: representation -> id)."""

        from ..backends.cupy import require_cupy

        from .fock_device import window_fits

        cp, _ = require_cupy()
        owned_sets, error, memory = self._outcome(self._owned_sets, [item["channel"] for item in payload["channels"]])
        if error is None and memory is None:
            fits, error, memory = self._outcome(window_fits, owned_sets)
            if error is None and memory is None and not fits:
                memory = "the DIIS window does not fit the device memory"
        self._settle(error, memory)

        def copy():
            made = {}
            for item, orbitals in zip(payload["channels"], owned_sets):
                for sector in orbitals.sectors:
                    key = item["blocks"].get(sector.representation)
                    if key is None:
                        continue
                    weights = orbitals.occupations[sector.states]
                    positions = np.flatnonzero(weights > OCCUPATION_THRESHOLD)
                    if not positions.size:
                        continue
                    with cp.cuda.Device(sector.device):
                        vectors = cp.array(sector.columns(cp, positions), dtype=cp.float64, order="F", copy=True)
                    made[int(key)] = OccupiedBlock(vectors, np.array(weights[positions], dtype=np.float64))
            return made

        self.objects.update(self._local(copy))
        return None

    def _overlap(self, payload):
        """``Y_left^T Y_right`` of two window blocks of one sector, on their owner; the root receives it."""

        owner = int(payload["owner"])

        def product():
            if owner != self.rank:
                return None
            return self.objects[int(payload["left"])].overlap(self.objects[int(payload["right"])])

        value = self._local(product)
        if owner == 0:
            return value
        if self.rank == owner:
            self.comm.send(value, dest=0, tag=_TAG + 1)
            return None
        if self.rank == 0:
            return self.comm.recv(source=owner, tag=_TAG + 1)
        return None

    def _combine(self, payload):
        """``Phi = sum_k Y_k C_k`` of window blocks on their owner, kept under ``id``."""

        def combine():
            if int(payload["owner"]) != self.rank:
                return None
            terms = [(self.objects[int(key)], np.asarray(part, dtype=np.float64)) for key, part in payload["terms"]]
            return OccupiedBlock.combine(terms)

        phi = self._local(combine)
        if phi is not None:
            self.objects[int(payload["id"])] = phi
        return None

    def _natural_parts(self, payload, sources) -> list:
        """Per channel, the new sector blocks of this rank: ``Phi`` and its empty states orthonormalized."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        built = []
        for item, orbitals in zip(payload["channels"], sources):
            parts = {}
            for sector in orbitals.sectors:
                representation = sector.representation
                if self.owners[representation] != self.rank:
                    continue
                weights = orbitals.occupations[sector.states]
                empty = np.flatnonzero(weights <= OCCUPATION_THRESHOLD)
                phi_key = item["phi"].get(representation)
                phi = None if phi_key is None else self.objects.pop(int(phi_key))
                with cp.cuda.Device(sector.device):
                    rest = sector.columns(cp, empty) if empty.size else None
                    if phi is not None and rest is not None:
                        rest = orthonormal_complement(phi, rest)
                    pieces = [piece for piece in (phi, rest) if piece is not None and piece.shape[1]]
                    if pieces:
                        parts[representation] = cp.asfortranarray(cp.concatenate(pieces, axis=1))
            built.append(parts)
        return built

    def _natural(self, payload):
        """Channels of natural orbitals: each owned sector holds its ``Phi`` and its empty states orthonormalized."""

        from .fock_device import build_fits

        keys = [item["source"] for item in payload["channels"]]
        owned_sets, error, memory = self._outcome(self._owned_sets, keys)
        if error is None and memory is None:
            fits, error, memory = self._outcome(build_fits, owned_sets)
            if error is None and memory is None and not fits:
                memory = "the channel of the extrapolated density matrix does not fit the devices"
        self._settle(error, memory)
        sources = [self.objects[int(key)].orbitals for key in keys]
        built = self._local(self._natural_parts, payload, sources)
        counts = self.comm.allgather(
            [{representation: int(vectors.shape[1]) for representation, vectors in parts.items()} for parts in built]
        )
        result = []
        for position, (item, orbitals, parts) in enumerate(zip(payload["channels"], sources, built)):
            sizes = {}
            for rows in counts:
                sizes.update(rows[position])
            sectors = []
            owners = []
            occupations = []
            start = 0
            loads = {device: 0 for device in self.devices}
            for sector in orbitals.sectors:
                representation = sector.representation
                count = sizes.get(representation, 0)
                if not count:
                    continue
                values = np.asarray(item["values"].get(representation, ()), dtype=np.float64)
                owner = self.owners[representation]
                if owner == self.rank:
                    vectors = parts[representation]
                    device = int(vectors.device.id)
                    local = np.arange(count, dtype=np.int64)
                else:
                    vectors = None
                    device = self._local_device(loads, 8 * int(sector.expansion.rows) * count)
                    local = np.full(count, -1, dtype=np.int64)
                sectors.append(
                    OrbitalSector(
                        representation=representation,
                        device=device,
                        vectors=vectors,
                        local=local,
                        states=np.arange(start, start + count, dtype=np.int64),
                        expansion=sector.expansion,
                    )
                )
                owners.append(owner)
                occupations.append(np.concatenate([values, np.zeros(count - values.size)]))
                start += count
            channel = ChannelOrbitals(
                tuple(sectors),
                np.concatenate(occupations) if occupations else np.zeros(0),
                float(orbitals.spin_weight),
                int(orbitals.full_size),
                float(orbitals.volume_element),
            )
            self.objects[int(item["id"])] = _Channel(channel, tuple(owners))
            result.append(channel.occupations)
        return result if self.rank == 0 else None


# ---------------------------------------------------------------------------
# The root's side: the engine of the Fock loop.
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class DistributedChannel:
    """A channel of the Fock loop whose orbitals the ranks hold (the handle of :class:`_Channel`)."""

    key: int
    occupations: np.ndarray
    spin_weight: float
    representations: tuple[int, ...]


@dataclass(eq=False)
class DistributedApplication:
    """``W`` of a channel on the owners of its sectors: the numbers the loop reads."""

    key: int
    orbitals: DistributedChannel
    columns: tuple[int, ...]
    occupied: tuple[int, ...]
    energy: float
    pair_solves: int
    poisson_seconds: float
    seconds: float


@dataclass(eq=False)
class DistributedCompressedExchange:
    """``a_x V_ACE`` of a channel whose ``xi_G`` lie on the owners of the sectors."""

    key: int
    fraction: float
    energy: float
    spin_weight: float
    asymmetry: float
    pair_solves: int
    seconds: float
    rank: int = 0
    released: bool = False

    @property
    def projectors(self) -> dict:
        return {}

    def release(self) -> None:
        self.released = True


@dataclass(eq=False)
class DistributedOccupiedBlock(OccupiedBlock):
    """A window block of the outer DIIS whose copy lies on the owner of its sector (``vectors`` is its key)."""

    owner: int = 0
    exchange: Any = None

    def overlap(self, other: "OccupiedBlock") -> np.ndarray:
        if not isinstance(other, DistributedOccupiedBlock) or other.owner != self.owner:
            raise ValueError("the blocks of one sector lie on the same rank")
        return np.asarray(
            self.exchange.execute("exx_overlap", dict(left=self.vectors, right=other.vectors, owner=self.owner)),
            dtype=np.float64,
        )

    @classmethod
    def combine(cls, terms):
        first = terms[0][0]
        key = first.exchange.new_key()
        first.exchange.execute(
            "exx_combine",
            dict(id=key, owner=first.owner,
                 terms=[(block.vectors, np.asarray(part, dtype=np.float64)) for block, part in terms]),
        )
        # Freed with the next command once the loop lets it go (the natural channel takes it over).
        return first.exchange._track(_RemoteArray(key, first.owner, first.exchange), key)


@dataclass(eq=False)
class _RemoteArray:
    """An array kept on rank ``owner`` under ``key`` (the natural orbitals of a sector)."""

    key: int
    owner: int
    exchange: Any


class MPIDeviceExchange(DeviceExchange):
    """:class:`~.hybrid.DeviceExchange` of an MPI run: every piece is a command to all ranks.

    ``context`` is the run's sector context, ``service`` the root's
    :class:`MPIExchangeService`.
    """

    def __init__(self, *, context, service: MPIExchangeService, **keywords) -> None:
        super().__init__(**keywords)
        self.context = context
        self.service = service
        self._keys = itertools.count(1)
        # Keys whose proxies are gone; a finalizer may append while a command takes them (deque: atomic ends).
        self._pending: deque = deque()
        self._bound_key: int | None = None

    # ----- commands -------------------------------------------------------------

    def new_key(self) -> int:
        return next(self._keys)

    def _track(self, proxy, *keys: int):
        """Free ``keys`` on every rank with the next command once ``proxy`` is gone."""

        weakref.finalize(proxy, self._pending.extend, [int(key) for key in keys])
        return proxy

    def execute(self, command: str, payload: dict):
        frees = []
        while self._pending:
            frees.append(self._pending.popleft())
        result = self.context.execute(self.sectors, command, dict(payload, free=frees))
        if isinstance(result, dict) and "memory_error" in result:
            raise MemoryError(result["memory_error"])
        return result

    # ----- orbitals --------------------------------------------------------------

    def _selection(self, wavefunctions, occupations, spin_weight: float, channel) -> dict:
        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals

        if not isinstance(wavefunctions, CuPySymmetryOrbitals):
            raise TypeError("the exchange of an MPI run needs the orbitals of the symmetry sectors")
        if channel is None:
            channel = getattr(wavefunctions, "spin_channel", None)
        return dict(
            channel=channel,
            representations=np.asarray(wavefunctions.representations, dtype=np.int32),
            columns=np.asarray(wavefunctions.representation_columns, dtype=np.int32),
            occupations=np.asarray(occupations, dtype=np.float64),
            spin_weight=float(spin_weight),
            volume_element=float(self.volume_element),
        )

    def _register(self, selections) -> tuple[DistributedChannel, ...]:
        for selection in selections:
            selection["id"] = self.new_key()
        self.execute("exx_channel", dict(channels=selections))
        return tuple(
            self._track(
                DistributedChannel(
                    selection["id"],
                    selection["occupations"],
                    selection["spin_weight"],
                    tuple(sorted(set(int(value) for value in selection["representations"]))),
                ),
                selection["id"],
            )
            for selection in selections
        )

    def channel(self, wavefunctions, occupations, spin_weight: float, volume_element: float | None = None):
        (channel,) = self._register([self._selection(wavefunctions, occupations, spin_weight, None)])
        return channel

    def orbital_sets(self, result):
        channels = getattr(result, "spin", None)
        if channels is None:
            return self._register([self._selection(result.wavefunctions, result.occupations, 2.0, None)])
        return self._register(
            [
                self._selection(item.wavefunctions, item.occupations, 1.0, index)
                for index, item in enumerate(channels)
            ]
        )

    # ----- the term in the Hamiltonians ---------------------------------------------

    def bind(self, channel) -> None:
        key = None if channel is None or getattr(channel, "released", False) else int(channel.key)
        if key != self._bound_key:
            self.execute("exx_bind", dict(id=key))
            self._bound_key = key
        self.bound = channel

    def planned_device_bytes(self, reference, definition, states: int) -> dict[int, int]:
        """:meth:`~.hybrid.DeviceExchange.planned_device_bytes` of this rank's sectors, with the replicas.

        Beside the ``W``/``xi`` of its own sectors, the plans and the tile
        work, a rank holds during a build the replica of every other rank's
        sector and its part of that sector's ``W``: twice the sector's share of
        ``states`` per channel, whole on the device where the service places
        it (:meth:`MPIExchangeService._build_channel`: the least loaded one,
        its own sectors' orbitals counted first).
        """

        counts = super().planned_device_bytes(reference, definition, states)
        decomposition = self.sectors.decomposition
        device_ids = getattr(self.sectors, "_sector_device_ids", ())
        owned = {index for index, operator in enumerate(self.sectors._operators) if operator is not None}
        total_rows = sum(decomposition.sector_size(index) for index in range(decomposition.representation_count))
        channels = 2 if bool(getattr(reference.input.scf, "spin_polarized", False)) else 1
        devices = tuple(int(device) for device in self.devices)
        loads = {device: 0 for device in devices}

        def share(index: int) -> int:
            rows = decomposition.sector_size(index)
            return min(int(states), int(np.ceil(int(states) * rows / max(1, total_rows))) + 8)

        for index in sorted(owned):
            device = int(device_ids[index])
            loads[device] = loads.get(device, 0) + 8 * decomposition.sector_size(index) * share(index)
        for index in range(decomposition.representation_count):
            if index in owned:
                continue
            nbytes = 8 * decomposition.sector_size(index) * share(index)
            device = min(devices, key=lambda name: (loads[name], devices.index(name)))
            loads[device] += nbytes
            counts[device] = counts.get(device, 0) + 2 * nbytes * channels
        return counts

    def release_build_memory(self, *, final: bool = False) -> None:
        """After an SCF: every rank lets the channels, replicas and parts of ``W`` of the loop go.

        ``final``: the run leaves this exchange (the spin-sector fallback to the
        full grid in the root process): the device plans of the pair solvers
        and the bound operator go as well.
        """

        if getattr(self.context, "failed", False) or getattr(self.context, "stopped", False):
            return
        self.execute("exx_trim", {"final": True} if final else {})
        if final:
            self._bound_key = None
            self.bound = None

    def fock_engine(self, system, definition) -> "MPIFockEngine":
        if definition.pair_solver != "kronecker":
            raise NotImplementedError(
                "the device exchange operator solves the pair densities with the Kronecker "
                "solver only (Exx_Pair_Solver: kronecker)"
            )
        return MPIFockEngine(self, system, definition)


@dataclass(frozen=True)
class MPIFockOperator(DeviceFockOperator):
    """:class:`~.hybrid.DeviceFockOperator` whose ``xi_G`` lie on the owners of the sectors."""

    def cross_energy(self, orbitals, occupations) -> float:
        """``E_x(P, P0)`` of an iteration's orbitals (on their ranks) with this operator."""

        if len(orbitals) != len(self.channels):
            raise ValueError("one orbital block per spin channel of the operator is needed")
        exchange = self.exchange
        pairs = []
        positions = []
        for index, (channel, block, weights) in enumerate(zip(self.channels, orbitals, occupations)):
            if channel is None:
                continue
            channel_index = None if len(self.channels) == 1 else index
            pairs.append(dict(operator=channel.key,
                              selection=exchange._selection(block, weights, channel.spin_weight, channel_index)))
            positions.append(channel)
        if not pairs:
            return 0.0
        merged = exchange.execute("exx_band", dict(pairs=pairs))
        total = 0.0
        for channel, sums in zip(positions, merged):
            value = 0.0
            for key in sorted(sums):
                value += sums[key]
            total += float(channel.spin_weight * -1.0 * value)
        return 0.5 * total


class MPIFockEngine(DeviceFockEngine):
    """The pieces of :func:`parsec_python.SCF.hybrid.run_hybrid_scf` across the ranks of an MPI run."""

    def __init__(self, exchange: MPIDeviceExchange, system, definition) -> None:
        self.device = exchange
        self.solvers = exchange.pair_solvers(system, definition)

    def exchange(self, sets, *, columns: str = "occupied", known=None):
        exchange = self.device
        applications = []
        for index, orbitals in enumerate(sets):
            started = perf_counter()
            key = exchange.new_key()
            previous = None if known is None else known[index]
            numbers = exchange.execute(
                "exx_apply",
                dict(channel=orbitals.key, id=key, columns=columns,
                     known=None if previous is None else previous.key),
            )
            seconds = perf_counter() - started
            application = DistributedApplication(
                key=key,
                orbitals=orbitals,
                columns=tuple(numbers["columns"]),
                occupied=tuple(numbers["occupied"]),
                energy=float(numbers["energy"]) if previous is None else float(previous.energy),
                pair_solves=int(numbers["pair_solves"] + (0 if previous is None else previous.pair_solves)),
                poisson_seconds=float(numbers["poisson_seconds"] + (0.0 if previous is None else previous.poisson_seconds)),
                seconds=float(seconds + (0.0 if previous is None else previous.seconds)),
            )
            if previous is not None:
                # Consumed by the completion (as in one process).
                previous.key = -1
            applications.append(exchange._track(application, key))
        return applications

    def fock_operator(self, sets, applications, fraction: float) -> MPIFockOperator:
        exchange = self.device
        reference_energy = float(sum(application.energy for application in applications))
        requests = []
        for application in applications:
            if not application.occupied:
                continue
            requests.append(dict(application=application.key, id=exchange.new_key(), energy=application.energy))
        numbers = exchange.execute("exx_compress", dict(channels=requests, fraction=float(fraction))) if requests else []
        channels = []
        position = 0
        for application in applications:
            if not application.occupied:
                channels.append(None)
                continue
            request, record = requests[position], numbers[position]
            position += 1
            application.key = -1
            channels.append(
                exchange._track(
                    DistributedCompressedExchange(
                        key=request["id"],
                        fraction=float(fraction),
                        energy=float(application.energy),
                        spin_weight=float(application.orbitals.spin_weight),
                        asymmetry=float(record["asymmetry"]),
                        pair_solves=int(application.pair_solves),
                        seconds=float(application.seconds),
                        rank=int(record["rank"]),
                    ),
                    request["id"],
                )
            )
        return MPIFockOperator(
            channels=tuple(channels),
            fraction=float(fraction),
            reference_energy=reference_energy,
            exchange=exchange,
        )

    def cross_energy(self, operator, sets) -> float:
        pairs = []
        weights = []
        for channel, orbitals in zip(operator.channels, sets):
            if channel is None:
                continue
            pairs.append(dict(operator=channel.key, channel=orbitals.key))
            weights.append(channel.spin_weight)
        if not pairs:
            return 0.0
        merged = self.device.execute("exx_band", dict(pairs=pairs))
        total = 0.0
        for weight, sums in zip(weights, merged):
            value = 0.0
            for key in sorted(sums):
                value += sums[key]
            total += float(weight * -1.0 * value)
        return 0.5 * total

    def drop_orbitals(self, sets, result=None) -> None:
        keys = [int(orbitals.key) for orbitals in sets]
        if keys:
            self.device.execute("exx_drop", dict(ids=keys))
        if result is not None:
            channels = getattr(result, "spin", None)
            for wavefunctions in (
                [result.wavefunctions] if channels is None else [channel.wavefunctions for channel in channels]
            ):
                release = getattr(wavefunctions, "release_intermediate_storage", None)
                if callable(release):
                    release()

    def occupied_blocks(self, sets) -> list | None:
        exchange = self.device
        requests = []
        layout = []
        for orbitals in sets:
            blocks = {}
            for representation in orbitals.representations:
                blocks[int(representation)] = exchange.new_key()
            requests.append(dict(channel=orbitals.key, blocks=blocks))
            layout.append(blocks)
        try:
            exchange.execute("exx_copy", dict(channels=requests))
        except MemoryError:
            return None
        channels = []
        for orbitals, blocks in zip(sets, layout):
            # The occupations of each sector's occupied states, in sector order (as the copies).
            selection = exchange.service.objects.get(int(orbitals.key))
            made = {}
            for representation, key in blocks.items():
                weights = _sector_occupations(orbitals, selection, representation, exchange)
                occupied = weights[weights > OCCUPATION_THRESHOLD]
                if not occupied.size:
                    continue
                made[representation] = exchange._track(
                    DistributedOccupiedBlock(
                        key, np.array(occupied, dtype=np.float64),
                        owner=exchange.service.owners[representation], exchange=exchange,
                    ),
                    key,
                )
            channels.append(made)
        return channels

    def natural_sets(self, sets, natural):
        exchange = self.device
        requests = []
        for orbitals, blocks in zip(sets, natural):
            if set(blocks) - set(orbitals.representations):
                return None
            requests.append(
                dict(
                    source=orbitals.key,
                    id=exchange.new_key(),
                    phi={int(key): phi.key for key, (phi, _values) in blocks.items()},
                    values={int(key): np.asarray(values, dtype=np.float64) for key, (_phi, values) in blocks.items()},
                )
            )
        occupations = exchange.execute("exx_natural", dict(channels=requests))
        result = []
        for orbitals, request, values in zip(sets, requests, occupations):
            result.append(
                exchange._track(
                    DistributedChannel(
                        request["id"],
                        np.asarray(values, dtype=np.float64),
                        orbitals.spin_weight,
                        tuple(int(key) for key in orbitals.representations),
                    ),
                    request["id"],
                )
            )
        return tuple(result)

    def release(self, operator) -> None:
        self.device.bind(None)
        keys = [channel.key for channel in operator.channels if channel is not None]
        for channel in operator.channels:
            if channel is not None:
                channel.release()
        if keys:
            self.device.execute("exx_free", dict(ids=keys))


def _sector_occupations(orbitals: DistributedChannel, entry, representation: int, exchange) -> np.ndarray:
    """The occupations of the states of ``representation`` in a channel, in its sector order."""

    sector = None
    if entry is not None:
        for item in entry.orbitals.sectors:
            if item.representation == representation:
                sector = item
                break
    if sector is None:
        raise RuntimeError("the root holds no layout of this channel")
    return np.asarray(orbitals.occupations)[sector.states]


def mpi_device_exchange_for(exchange: DeviceExchange, context, sectors, reference):
    """The MPI form of a prepared :class:`~.hybrid.DeviceExchange`, with the rank's service on ``context``.

    Called on every rank; the service answers the ``exx_*`` commands, the
    returned exchange is what the root's Fock loop uses.
    """

    from parsec_python.SCF.hybrid import hybrid_definition

    if sectors is None:
        raise NotImplementedError(
            "the exchange of a hybrid on several MPI ranks needs the symmetry-sector eigensolver"
        )
    mpi_exchange = MPIDeviceExchange(
        context=context,
        service=None,
        backend=exchange.backend,
        sectors=sectors,
        devices=exchange.devices,
        start_xc=exchange.start_xc,
        multipole_order=exchange.multipole_order,
        batch_size=exchange.batch_size,
        volume_element=exchange.volume_element,
    )
    service = MPIExchangeService(
        context, mpi_exchange, sectors, reference.grid, hybrid_definition(reference.input.scf)
    )
    mpi_exchange.service = service
    context.exchange_service = service
    return mpi_exchange


__all__ = [
    "DistributedApplication",
    "DistributedChannel",
    "DistributedCompressedExchange",
    "DistributedOccupiedBlock",
    "ExchangeTransport",
    "MPIDeviceExchange",
    "MPIExchangeService",
    "MPIFockEngine",
    "MPIFockOperator",
    "column_runs",
    "exchange_chunk_bytes",
    "exchange_transport",
    "mpi_device_exchange_for",
]
