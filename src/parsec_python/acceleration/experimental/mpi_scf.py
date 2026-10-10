"""Root-controlled SCF with persistent, independently owned symmetry sectors.

Only rank zero enters the production SCF loop. Other ranks call worker_loop.
The existing first/subsequent eigensolver policies remain on each sector's
owner. MPI carries scalar potentials/densities and small Ritz metadata, never
orbital matrices. This is experimental full-SCF orchestration, not distributed
Ritz across ranks: a sector stays on the GPUs of its own rank. Several of them
share its basis by default (Eigensolvers/distributed_state.py); otherwise the
optional node-local filtering gathers onto its owner GPU.
"""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np


class MPISCFError(RuntimeError):
    """A worker failed; the outer launcher must abort the MPI job."""


def sector_device_groups(representations, size, rank, devices):
    """Assign sectors cyclically to ranks and disjoint GPU groups when possible."""
    representations, size, rank = int(representations), int(size), int(rank)
    devices = tuple(int(device) for device in devices)
    if representations < 1 or not 1 <= size <= representations or not 0 <= rank < size:
        raise ValueError('MPI ranks must be between one and the representation count')
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError('devices must be a nonempty list of distinct nonnegative indices')
    owned = tuple(range(rank, representations, size))
    if len(devices) < len(owned):
        return {sector: (devices[index % len(devices)],)
                for index, sector in enumerate(owned)}
    return {sector: tuple(int(x) for x in group)
            for sector, group in zip(owned, np.array_split(devices, len(owned)), strict=True)}


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
    """
    def __init__(self, comm):
        self.comm = comm
        self.rank, self.size = int(comm.rank), int(comm.size)
        self.root = 0
        self.failed = False
        self.stopped = False
        self.owned_sectors = ()
        self.device_groups = {}
        self.command_counts = {}
        # Wall seconds of each command on this rank, collective calls included.
        self.command_seconds = {}

    def configure(self, representations, devices):
        error = None
        try:
            groups = sector_device_groups(representations, self.size, self.rank, devices)
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
        self.topology = tuple(record[1] for record in records)
        return groups

    def execute(self, solver, command, payload=None, *, density_builder=None):
        if self.rank != self.root:
            raise MPISCFError('only root may dispatch SCF commands')
        if self.failed or self.stopped:
            raise MPISCFError('MPI sector context is no longer active')
        self.comm.bcast((command, payload), root=self.root)
        return self._execute_local(solver, command, payload, density_builder)

    def worker_loop(self, solver, density_builder):
        if self.rank == self.root:
            raise MPISCFError('root must run SCF, not the worker loop')
        try:
            while True:
                command, payload = self.comm.bcast(None, root=self.root)
                if command == 'stop':
                    self.stopped = True
                    return
                self._execute_local(solver, command, payload, density_builder)
        finally:
            solver.restore_memory_allocator()

    def stop_workers(self):
        if self.rank != self.root:
            raise MPISCFError('only root may stop workers')
        if not self.failed and not self.stopped:
            self.comm.bcast(('stop', None), root=self.root)
            self.stopped = True

    def update_potential(self, solver, potential):
        from ..SCF.symmetry_fields import SymmetryScalarField
        if isinstance(potential, SymmetryScalarField):
            if potential.reduction is not solver.decomposition.reduction:
                raise ValueError('local potential uses a different symmetry map')
            values = potential.values
        else:
            values = solver.decomposition.invariant_wedge_values(potential)
        self.execute(solver, 'potential', np.ascontiguousarray(values, dtype=np.float64))

    def _execute_local(self, solver, command, payload, density_builder):
        started = perf_counter()
        try:
            return self._execute_counted(solver, command, payload, density_builder)
        finally:
            self.command_seconds[command] = self.command_seconds.get(command, 0.0) + perf_counter() - started

    def _execute_counted(self, solver, command, payload, density_builder):
        self.command_counts[command] = self.command_counts.get(command, 0) + 1
        error, value, packed = None, None, None
        try:
            value = self._apply(solver, command, payload, density_builder)
            if command == 'solve':
                # Strip both orbital arrays AND saved eigensolver states.
                packed = {sector: replace(result, vectors=None, state=None)
                          for sector, result in value.items()}
            elif command == 'density':
                value = np.ascontiguousarray(value, dtype=np.float64)
                if value.shape != (solver.decomposition.wedge_size,) or not np.all(np.isfinite(value)):
                    raise ValueError('local density must be a finite physical wedge field')
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
        if command == 'solve':
            records = self.comm.allgather(packed)
            if self.rank != self.root:
                return None
            merged = {}
            for rank, record in enumerate(records):
                for sector, result in record.items():
                    if sector % self.size != rank or sector in merged:
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
        if command == 'reset':
            solver._reset_local()
            solver._mpi_latest_results = {}
        elif command == 'configure_memory':
            solver._configure_large_problem_allocator(payload)
        elif command == 'potential':
            from ..SCF.symmetry_fields import SymmetryScalarField
            solver._update_local_potential(SymmetryScalarField(solver.decomposition.reduction, payload))
        elif command == 'solve':
            requested = tuple(index for index in payload['representations'] if index in self.device_groups)
            latest = solver._mpi_latest_results
            for index in requested:
                latest.pop(index, None)
            if self.rank != self.root:
                solver._sector_counts = list(payload['counts'])
            result = solver._run_local_sector_jobs(requested, payload['counts'], payload['settings'],
                                                   reset=payload['reset'])
            latest.update(result)
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
        values = self.context.execute(self.solver, 'density', payload, density_builder=self.local_builder)
        reducer = getattr(self.local_builder, 'reducer', None)
        return reducer.field(values) if reducer is not None else values[self.solver.decomposition.reduction.full_to_wedge]


__all__ = ['MPISCFError', 'MPISectorContext', 'MPISymmetryDensityBuilder', 'sector_device_groups']
