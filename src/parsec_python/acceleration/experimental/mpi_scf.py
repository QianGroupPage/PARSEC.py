"""Root-controlled SCF with persistent, independently owned symmetry sectors.

Only rank zero enters the production SCF loop. Other ranks call worker_loop.
The existing first/subsequent eigensolver policies remain on each sector's
owner. The world communicator carries scalar potentials/densities and small
Ritz metadata, never orbital matrices. With one rank per node a sector stays on
the GPUs of its own rank; several of them share its basis by default
(Eigensolvers/distributed_state.py), otherwise the optional node-local filtering
gathers onto its owner GPU. With one rank per GPU (``gpu_per_rank``) and more
ranks than sectors, disjoint groups of consecutive ranks share the sectors, one
group per sector, which may span nodes: each group's communicator carries the
filter, layout changes, Gram sums and rotations of its basis
(Eigensolvers/mpi_shared_basis.py), and a run without symmetry is the case of one
sector. A hybrid functional keeps its own ownership rules.
"""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np


class MPISCFError(RuntimeError):
    """A worker failed; the outer launcher must abort the MPI job."""


def sector_rank_groups(representations, size):
    """The ranks of each sector: balanced groups of consecutive ranks where there are more ranks than sectors.

    With no more ranks than sectors, the sectors go round the ranks, one rank each (the former assignment).
    Otherwise every sector gets ``size // representations`` consecutive ranks and the first
    ``size % representations`` sectors one more, so that every rank serves one sector; consecutive ranks
    keep a group on as few nodes as the counts allow, and a group may span nodes.
    """
    representations, size = int(representations), int(size)
    if representations < 1 or size < 1:
        raise ValueError('MPI ranks and representations must be positive')
    if size <= representations:
        return tuple((sector % size,) for sector in range(representations))
    base, extra = divmod(size, representations)
    edges = [0]
    for sector in range(representations):
        edges.append(edges[-1] + base + (sector < extra))
    return tuple(tuple(range(edges[sector], edges[sector + 1])) for sector in range(representations))


def sector_device_groups(representations, size, rank, devices, shared=False, exchange=False):
    """Assign sectors cyclically to ranks and disjoint GPU groups when possible.

    With ``shared`` every sector belongs to a group of ranks (:func:`sector_rank_groups`; the one sector of
    a run without symmetry to all of them), each of which holds a share of its basis on its first device
    (:mod:`..Eigensolvers.mpi_shared_basis`).  With ``exchange`` (a hybrid
    functional, ``SCF/hybrid_mpi.py``) every sector basis lies on one device
    of its rank, which the exchange operator needs, and a rank may own no
    sector: every device of every rank solves pair densities of the exchange.
    """
    representations, size, rank = int(representations), int(size), int(rank)
    devices = tuple(int(device) for device in devices)
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError('devices must be a nonempty list of distinct nonnegative indices')
    if exchange:
        if representations < 1 or size < 1 or not 0 <= rank < size:
            raise ValueError('MPI ranks and representations must be positive')
        owned = tuple(range(rank, representations, size))
        return {sector: (devices[index % len(devices)],) for index, sector in enumerate(owned)}
    if shared:
        if representations < 1 or not 0 <= rank < size or size < representations:
            raise ValueError('a shared sector needs at least one rank per sector')
        return {sector: (devices[0],) for sector, ranks in enumerate(sector_rank_groups(representations, size))
                if rank in ranks}
    if representations < 1 or not 1 <= size <= representations or not 0 <= rank < size:
        raise ValueError('MPI ranks must be between one and the representation count')
    owned = tuple(range(rank, representations, size))
    if len(devices) < len(owned):
        return {sector: (devices[index % len(devices)],)
                for index, sector in enumerate(owned)}
    groups = np.array_split(devices, len(owned))
    if (len(owned) > 1 and len(devices) % len(owned) and sector_device_pool_setting() == 'auto'
            and len(owned) / len(devices) < 1.0 / min(len(group) for group in groups)):
        # Sectors that do not divide the devices evenly (3 on 4: groups of 2, 1 and 1 device) leave a device with
        # a whole sector; every sector over all the devices instead (solved one after another, their bases split
        # over every device) holds len(owned)/len(devices) of a sector per device (PARSEC_SECTOR_DEVICE_POOL).
        return {sector: tuple(int(x) for x in devices) for sector in owned}
    return {sector: tuple(int(x) for x in group)
            for sector, group in zip(owned, groups, strict=True)}


def sector_device_pool_setting():
    """``PARSEC_SECTOR_DEVICE_POOL``: ``auto`` (default) spreads every sector of a rank over all its devices where
    the sectors do not divide them evenly (:func:`sector_device_groups`); ``off`` keeps disjoint groups, as before."""
    import os

    value = os.environ.get('PARSEC_SECTOR_DEVICE_POOL', 'auto').strip().lower() or 'auto'
    if value not in {'auto', 'off'}:
        raise ValueError(f'PARSEC_SECTOR_DEVICE_POOL must be auto or off, not {value!r}')
    return value


class MPISectorContext:
    """MPI command protocol; all MPI calls occur on the calling main thread.

    Construct/configure on every rank. Preparation with this context builds
    the SCF-loop components (ionic fields, densities, Hartree, XC) on root
    only; non-root ranks enter worker_loop(solver, unwrapped_density_builder).
    Root wraps that builder with MPISymmetryDensityBuilder and runs the normal
    SCF. Call stop_workers after success. Any exception requires MPI.Abort;
    stop_workers deliberately does not communicate after a collective failure.
    One failure is not collective: a pool release that a rank waits for behind
    the collective calls of its density command (_release_joined) is raised by
    that rank alone, and the others learn of it by the launcher's Abort.

    ``exchange`` (a hybrid functional): every sector basis on one device, a
    rank may own no sector, every rank keeps the sector orbitals of its last
    solve (``_mpi_retained_results`` of the solver) for the exchange, and the
    commands ``exx_*`` go to the rank's ``exchange_service``
    (``SCF/hybrid_mpi.py``).  Every command carries the spin channel it is
    for (the active one of the root's solver, or the channel of the orbitals
    of a density); a rank serves both channels of a spin-polarized loop in
    the sectors it owns.
    """
    def __init__(self, comm, *, exchange=False):
        self.comm = comm
        self.rank, self.size = int(comm.rank), int(comm.size)
        self.root = 0
        self.failed = False
        self.stopped = False
        self.owned_sectors = ()
        self.device_groups = {}
        # Whether groups of ranks share the sector bases (configure), and the communicator of this rank's group.
        self.shared = False
        self.sector_comm = comm
        self.rank_groups = ()
        self.sector_owners = ()
        # One GPU per rank (the runner's --rank-per-gpu launch): more ranks than several sectors then share them.
        self.gpu_per_rank = False
        # The Ritz step of a shared basis where PARSEC_MPI_SHARED_RITZ names none
        # (mpi_shared_basis.shared_ritz_route): the runner asks for rounds for a periodic cell.
        self.shared_ritz_route = "rows"
        self.command_counts = {}
        # Wall seconds of each command on this rank, collective calls included.
        self.command_seconds = {}
        self.exchange = bool(exchange)
        self.retain_orbitals = bool(exchange)
        self.exchange_service = None

    def planned_groups(self, representations, devices):
        """The device groups of the sectors this rank will own (what ``configure`` assigns)."""
        return sector_device_groups(representations, self.size, self.rank, devices,
                                    shared=self._shared(representations), exchange=self.exchange)

    def _shared(self, representations):
        # More ranks than sectors share their bases: the one sector of a run without symmetry, or several with
        # one GPU per rank (not with a hybrid).
        representations = int(representations)
        if self.exchange or self.size <= 1:
            return False
        return representations == 1 or (self.gpu_per_rank and self.size > representations)

    def configure(self, representations, devices):
        shared = self._shared(representations)
        error = None
        try:
            groups = sector_device_groups(representations, self.size, self.rank, devices, shared=shared,
                                          exchange=self.exchange)
        except Exception as exc:
            groups = {}
            error = f'{type(exc).__name__}: {exc}'
        records = self.comm.allgather((int(representations), groups, error))
        errors = [f'rank {rank}: {record[2]}' for rank, record in enumerate(records) if record[2]]
        if errors or len({record[0] for record in records}) != 1:
            self.failed = True
            raise MPISCFError('sector configuration failed: ' + '; '.join(errors or ['representation counts differ']))
        self.representation_count = int(representations)
        self.device_groups = groups
        self.owned_sectors = tuple(groups)
        self.shared = shared
        self.topology = tuple(record[1] for record in records)
        self.rank_groups = sector_rank_groups(representations, self.size)
        self.sector_owners = tuple(group[0] for group in self.rank_groups)
        if shared and int(representations) > 1:
            # Every rank joins the communicator of its sector's group; no collective of an eigensolver
            # crosses the groups after this.
            self.sector_comm = self.comm.Split(color=self.owned_sectors[0], key=self.rank)
        else:
            self.sector_comm = self.comm
        return groups

    def execute(self, solver, command, payload=None, *, density_builder=None, channel=None):
        if self.rank != self.root:
            raise MPISCFError('only root may dispatch SCF commands')
        if self.failed or self.stopped:
            raise MPISCFError('MPI sector context is no longer active')
        if channel is None:
            channel = self._active_channel(solver)
        self.comm.bcast((command, payload, channel), root=self.root)
        return self._execute_local(solver, command, payload, density_builder, channel)

    def worker_loop(self, solver, density_builder):
        if self.rank == self.root:
            raise MPISCFError('root must run SCF, not the worker loop')
        try:
            while True:
                message = self.comm.bcast(None, root=self.root)
                # (command, payload, spin channel); a command without a channel names none.
                command, payload = message[0], message[1]
                channel = message[2] if len(message) > 2 else None
                if command == 'stop':
                    self.stopped = True
                    return
                self._execute_local(solver, command, payload, density_builder, channel)
        finally:
            solver.restore_memory_allocator()

    def stop_workers(self):
        if self.rank != self.root:
            raise MPISCFError('only root may stop workers')
        if not self.failed and not self.stopped:
            self.comm.bcast(('stop', None, None), root=self.root)
            self.stopped = True

    @staticmethod
    def _active_channel(solver):
        """The spin channel the root's solver serves now, ``None`` without spin channels."""
        if getattr(solver, '_spin_channels', None) is None:
            return None
        return int(solver._active_spin_channel)

    @staticmethod
    def _select_channel(solver, channel):
        """Put spin channel ``channel`` in place in ``solver``; return the channel to put back (or None)."""
        if channel is None or solver is None:
            return None
        if getattr(solver, '_spin_channels', None) is None:
            solver.for_spin_channel(int(channel))
        previous = int(solver._active_spin_channel)
        solver._activate_spin_channel(int(channel))
        return previous

    def update_potential(self, solver, potential):
        from ..SCF.symmetry_fields import SymmetryScalarField
        if isinstance(potential, SymmetryScalarField):
            if potential.reduction is not solver.decomposition.reduction:
                raise ValueError('local potential uses a different symmetry map')
            values = potential.values
        else:
            values = solver.decomposition.invariant_wedge_values(potential)
        self.execute(solver, 'potential', np.ascontiguousarray(values, dtype=np.float64))

    def _execute_local(self, solver, command, payload, density_builder, channel=None):
        started = perf_counter()
        previous = self._select_channel(solver, channel)
        try:
            return self._execute_counted(solver, command, payload, density_builder)
        finally:
            if previous is not None:
                solver._activate_spin_channel(previous)
            self.command_seconds[command] = self.command_seconds.get(command, 0.0) + perf_counter() - started

    def _execute_counted(self, solver, command, payload, density_builder):
        self.command_counts[command] = self.command_counts.get(command, 0) + 1
        error, value, packed = None, None, None
        try:
            value = self._apply(solver, command, payload, density_builder)
            if command == 'solve':
                # Strip both orbital arrays AND saved eigensolver states, and the record of a
                # hand-over (a device array); its plain summary, handover_step, travels.
                packed = {sector: replace(result, vectors=None, state=None, handover=None)
                          for sector, result in value.items()}
            elif command == 'density':
                value = np.ascontiguousarray(value, dtype=np.float64)
                if value.shape != (solver.decomposition.wedge_size,) or not np.all(np.isfinite(value)):
                    raise ValueError('local density must be a finite physical wedge field')
            elif command in ('forces', 'spin_forces'):
                value = np.ascontiguousarray(value, dtype=np.float64)
                if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
                    raise ValueError('local force terms must be finite, three per projector')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        errors = self.comm.allgather(error)
        if any(item is not None for item in errors):
            self.failed = True
            details = '; '.join(f'rank {rank}: {item}' for rank, item in enumerate(errors) if item)
            failure = MPISCFError(f'{command} failed: {details}')
            try:
                self._release_joined(density_builder)
            except MPISCFError as released:
                # The failure that every rank reports is the one to raise.
                raise failure from released
            raise failure
        if command == 'solve' and self.shared and self.representation_count == 1:
            # Every rank solved the shared sector together: the Ritz values are those of all of them.
            return value if self.rank == self.root else None
        if command == 'solve':
            if self.shared:
                # The ranks of a group hold the same Ritz values: its first rank alone reports them (the
                # density then sums the columns of every rank of the group once).
                packed = {sector: result for sector, result in packed.items()
                          if self.sector_owners[sector] == self.rank}
            records = self.comm.allgather(packed)
            if self.rank != self.root:
                return None
            merged = {}
            for rank, record in enumerate(records):
                for sector, result in record.items():
                    owner = self.sector_owners[sector] if self.shared else sector % self.size
                    if owner != rank or sector in merged:
                        raise MPISCFError('sector result has duplicate or incorrect ownership')
                    merged[sector] = value[sector] if rank == self.rank else result
            if set(merged) != set(payload['representations']):
                raise MPISCFError('sector solve omitted a requested representation')
            return merged
        if command == 'density':
            total = np.empty_like(value)
            self.comm.Allreduce(value, total)
            # The pools that the builder left to their device threads (_apply) are empty before this
            # rank goes on: the root to its Hartree solve, a worker to the next command.
            self._release_joined(density_builder)
            return total
        if command == 'sector_margin':
            # Every sector takes the least share of the ranks that solve it.
            records = self.comm.allgather((tuple(self.owned_sectors), value))
            if self.rank != self.root:
                return None
            fractions = {}
            for sectors, fraction in records:
                for sector in sectors:
                    fractions[sector] = min(fractions.get(sector, 1.0), fraction)
            return fractions
        if command in ('forces', 'spin_forces'):
            # One row per projector on every rank: the ranks hold the same atoms.
            total = np.empty_like(value)
            self.comm.Allreduce(value, total)
            return total
        return value

    def _release_joined(self, density_builder):
        """Wait for the pool release of ``density_builder`` where it left that to its caller.

        A release that failed in a device thread is raised here, behind the collective calls of the
        command, and so by this rank alone: as MPISCFError, with the context marked failed, so that a
        root does not send 'stop' into a job that its launcher is about to abort. The other ranks have
        their densities and are not told; they wait in their next collective call until that Abort.
        With PARSEC_CUPY_DENSITY_RELEASE=serial the release runs inside the builder, and its failure
        is reported through the command by every rank, as before.
        """
        joined = getattr(density_builder, 'release_joined', None)
        if not callable(joined):
            return
        try:
            joined()
        except Exception as exc:
            self.failed = True
            raise MPISCFError(f'density pool release failed: rank {self.rank}: {type(exc).__name__}: {exc}') from exc

    def _apply(self, solver, command, payload, density_builder):
        if command.startswith('exx_'):
            # A piece of the Fock loop of a hybrid functional (SCF/hybrid_mpi.py).
            if self.exchange_service is None:
                raise RuntimeError(f'rank {self.rank} has no exchange service for {command}')
            return self.exchange_service.handle(command, payload)
        if command == 'reset':
            solver._reset_local()
            solver._mpi_latest_results = {}
            solver._mpi_retained_results = {}
        elif command == 'configure_memory':
            solver._configure_large_problem_allocator(payload)
        elif command == 'sector_margin':
            # The share of the margin of the first counts that the devices of this rank hold.
            return float(solver._margin_fraction_local(*payload))
        elif command == 'potential':
            from ..SCF.symmetry_fields import SymmetryScalarField
            solver._update_local_potential(SymmetryScalarField(solver.decomposition.reduction, payload))
        elif command == 'solve':
            requested = tuple(index for index in payload['representations'] if index in self.device_groups)
            latest = solver._mpi_latest_results
            # The orbitals that a hybrid's exchange reads: those of the last solve of every sector,
            # kept past the density and a new SCF loop (as the views of one process are).
            retained = solver._mpi_retained_results if self.retain_orbitals else {}
            for index in requested:
                latest.pop(index, None)
                retained.pop(index, None)
            if self.rank != self.root:
                solver._sector_counts = list(payload['counts'])
            result = solver._run_local_sector_jobs(requested, payload['counts'], payload['settings'],
                                                   reset=payload['reset'])
            latest.update(result)
            retained.update(result)
            if self.rank != self.root:
                # The root counts every sector from the records it gathers; a worker its own.
                record = getattr(solver, '_record_handover', None)
                if callable(record):
                    for sector, item in result.items():
                        record(sector, item)
            return result
        elif command == 'trim':
            for sector, count in payload:
                if sector in self.device_groups:
                    solver._solvers[sector].truncate_state(count)
                solver._sector_counts[sector] = count
        elif command == 'density':
            # Keep the pre-trim result snapshots through density construction:
            # the selected columns can exceed the NEXT iteration's active set.
            representations, columns, occupations, volume = payload
            owned = np.isin(representations, self.owned_sectors)
            results = [solver._mpi_latest_results.get(index, SimpleNamespace(vectors=None))
                       for index in range(self.representation_count)]
            orbitals = solver._pack_selected_wedge_vectors(results, representations[owned], columns[owned])
            # The pool release after the density (PARSEC_CUPY_DENSITY_RELEASE) runs on the device threads
            # beside the collective calls of this command, which touch no device, and is waited for
            # behind them (_execute_counted) instead of in the builder.
            if hasattr(density_builder, 'caller_joins_release'):
                density_builder.caller_joins_release = True
            local_density = density_builder(orbitals, occupations[owned], volume)
            from ..SCF.symmetry_fields import SymmetryScalarField
            values = (local_density.values if isinstance(local_density, SymmetryScalarField)
                      else solver.decomposition.invariant_wedge_values(local_density))
            solver._mpi_latest_results.clear()
            return values
        elif command == 'new_scf_loop':
            # The SCF loop of the next structure continues from the saved subspaces.
            solver._start_new_scf_loop_local()
        elif command == 'geometry':
            # Every rank rebuilds what it holds for the moved atoms: all of them the projectors of
            # their sectors, the root also the fields of the SCF loop.
            handler = getattr(solver, 'geometry_handler', None)
            if handler is None:
                raise RuntimeError('this rank was prepared without a geometry handler')
            handler(np.asarray(payload, dtype=np.float64))
        elif command == 'forces':
            # Nonlocal force terms of the sectors of this rank, from the states its solvers hold.
            representations, columns, occupations = payload
            return solver._projector_force_terms_local(representations, columns, occupations)
        elif command == 'spin_forces':
            # The same for both spin channels of a spin-polarized loop, summed.
            return solver._spin_projector_force_terms_local(payload)
        else:
            raise ValueError(f'unknown MPI SCF command: {command}')


class MPISymmetryDensityBuilder:
    """Root SCF density hook; reduce physical wedge densities, not orbitals."""
    def __init__(self, local_builder, context, solver):
        self.local_builder, self.context, self.solver = local_builder, context, solver

    def __call__(self, wavefunctions, occupations, volume_element):
        from ..Eigensolvers.symmetry import CuPySymmetryOrbitals
        if not isinstance(wavefunctions, CuPySymmetryOrbitals):
            raise TypeError('MPI SCF requires lazy symmetry orbitals')
        weights = np.asarray(occupations, dtype=np.float64)
        if weights.shape != wavefunctions.representations.shape:
            raise ValueError('occupation count differs from the global selected states')
        payload = (wavefunctions.representations, wavefunctions.representation_columns,
                   weights, float(volume_element))
        # The orbitals of a spin channel are that channel's on every rank, whichever channel the
        # root's solver serves now.
        values = self.context.execute(self.solver, 'density', payload, density_builder=self.local_builder,
                                      channel=getattr(wavefunctions, 'spin_channel', None))
        reducer = getattr(self.local_builder, 'reducer', None)
        return reducer.field(values) if reducer is not None else values[self.solver.decomposition.reduction.full_to_wedge]


__all__ = ['MPISCFError', 'MPISectorContext', 'MPISymmetryDensityBuilder', 'sector_device_groups',
           'sector_rank_groups']
