"""Density construction that keeps representation orbitals on the wedge."""

from __future__ import annotations

import gc
import os
from concurrent.futures import wait
from contextlib import nullcontext
from time import perf_counter
from typing import Any

import numpy as np

from ..Eigensolvers.distributed_filter import _pool
from ..Eigensolvers.distributed_state import DistributedBasis, shared_pool_release_requested
from ..Eigensolvers.symmetry import (
    CuPySymmetryOrbitals,
    _pool_release_orbital_bytes,
)
from ..SCF.symmetry_fields import SymmetrySCFReducer

_COLLECTIONS = ("changed", "full")
_RELEASES = ("threads", "serial")


def density_collection() -> str:
    """Return the collection selected by ``PARSEC_CUPY_DENSITY_COLLECTION``.

    The pool release after a density is preceded by a collection of the
    cyclic garbage collector, so that device arrays held only by a reference
    cycle are returned with the cached blocks.  ``full`` collects every
    generation each time, as before.  ``changed`` (the default) does so after
    the first two densities, which follow the first solve and the pass that
    drops its state, and afterwards whenever a pool of the released devices
    has other bytes in use than the last full collection left there: an
    unreachable cycle that holds device memory beside the live arrays shows
    as such a difference.  Otherwise it collects the young generations only:
    an unreachable cycle of older objects then waits for the next full
    collection, with the bytes in use on every released device at the level
    the last one left.  Only arrays of the default pool are seen this way:
    pinned host memory and device memory a library holds outside the pool
    (a captured graph, a solver workspace) are not.  A process that does not
    allocate from the default pool, and a call that releases the pool of no
    device because the sector states are kept on the host, always collect in
    full.
    """

    mode = (
        os.environ.get("PARSEC_CUPY_DENSITY_COLLECTION", "changed").strip().lower()
    )
    if mode not in _COLLECTIONS:
        raise ValueError("PARSEC_CUPY_DENSITY_COLLECTION must be changed or full")
    return mode


def density_release() -> str:
    """Return the pool release selected by ``PARSEC_CUPY_DENSITY_RELEASE``.

    After a density the unused pool blocks of the devices of the sector
    vectors go back to the driver, and the pinned pool with them
    (:meth:`CuPySymmetryDensityBuilder._release_large_unused_pool`).
    ``serial`` empties one device after another and then the pinned pool, in
    the thread that built the density and before it goes on, as before.
    ``threads`` (the default) hands the pool of every device to the thread
    of that device, the one that filters on it where a filter is spread over
    devices and that lives as long as the process, empties the pinned pool
    meanwhile and then waits for the devices: they are emptied side by side.
    An MPI rank waits behind the collective calls of its density command
    instead (:meth:`CuPySymmetryDensityBuilder.release_joined`), so that the
    release also runs beside those.  A process that releases one device does
    so in its own thread either way.

    The release is on the path of every SCF step: it runs inside the
    collective density command, in which the ranks wait for the slowest.
    First runs on A100 nodes spent 0.01 to 0.12 s of a step in it on that
    rank, 0.10 to 1.24 s of a run and 0.1 to 2.8% of the program (3,480 to
    39,368 electrons on 1 to 16 GPUs), the sum over its devices.  Side by
    side that is the longest device at best, three quarters less on four,
    if the driver returns the memory of several devices at once.  On A100
    nodes it does so in part: against ``serial`` the stage of the density
    took 0.35 and 0.34 s less for 14,680 and 23,768 electrons on 16 GPUs,
    0.04 s less for 5,264 on 8 and 0.25 s less for 14,680 on 4, while the
    device threads together took 1.2 to 3.7 times what ``serial`` did.
    On a workstation, two pools of its one GPU standing for two devices,
    it did not: 2 x 1.1 GiB went back in 11.5 to 12.2 ms side by side and
    in 12.1 to 17.4 ms one after another, the two frees together taking 18
    to 23 ms instead of 12 to 17, and five releases of 2 x 0.6 GiB by this
    builder took 30 to 33 ms where ``serial`` took 29 to 31.  Where the
    ``release_seconds`` of a run are not below those of its ``serial``
    control, the threads gain nothing and ``serial`` is the setting to
    keep; ``device_release_seconds`` at or above them then says that the
    driver freed one device at a time.

    Every pool is empty before the thread that built the density does
    anything else, as it was: the Hartree solve of the root, which takes
    arrays on its device at once, finds the device as before, and no device
    holds more at any moment than it did.  That is why the release is not
    left to run beside that solve, where it would be off the path
    altogether.  It is dropped where the blocks that stay are those that
    the next step asks for, for a sector whose basis several devices share
    (:func:`distributed_state.shared_pool_release_requested`), and kept
    for a sector on one device, where they are not always.  No graph
    capture is open in a thread that releases, since a thread of a device
    ends its captures within the task that began them; the threads do not
    end with the release, no device is synchronized, and memory that
    another thread returns leaves a capture valid
    (:mod:`backends.cupy_capture`).  Neither setting changes a result.
    """

    mode = os.environ.get("PARSEC_CUPY_DENSITY_RELEASE", "threads").strip().lower()
    if mode not in _RELEASES:
        raise ValueError("PARSEC_CUPY_DENSITY_RELEASE must be threads or serial")
    return mode


class CuPySymmetryDensityBuilder:
    """Build the full scalar density without expanding every wavefunction.

    A real one-dimensional representation differs between symmetry images
    only by a sign.  Since the density contains ``|psi|**2``, all images have
    the same value.  The wrapped ordinary CuPy builder therefore evaluates
    the globally ordered, already-normalized wedge columns once, downloads a
    wedge-length density, and expands that scalar field by orbit lookup.
    """

    def __init__(
        self,
        device_builder: Any,
        timing_stats: Any | None = None,
        reducer: SymmetrySCFReducer | None = None,
    ) -> None:
        self.device_builder = device_builder
        self.timing_stats = timing_stats
        self.reducer = reducer
        # Read here so that a value it does not know is refused in every run,
        # and before the first eigensolve: the collection, the threads of
        # the release, and its limit, which is used after the first density.
        self.collection = density_collection()
        self.release = density_release()
        self.shared_release = shared_pool_release_requested()
        _pool_release_orbital_bytes()
        # Of this builder: pool releases, their collections by kind, the
        # unreachable objects those found, and the wall seconds of the
        # collections, of the release of the pools and of the whole
        # sector-wise density, both included.  The seconds of the release
        # are those of the thread that built the density, its wait for the
        # device threads included; ``device_release_seconds`` are the sum of
        # what those threads took, each for its device.  ``shared`` says
        # whether the pools of the devices of a shared basis are emptied as
        # well, ``release`` or ``keep``, and ``shared_kept`` counts the
        # densities after which they were left; ``calls`` are the releases
        # of a pool.
        self.pool_release = dict(
            collection=self.collection,
            release=self.release,
            shared="release" if self.shared_release else "keep",
            shared_kept=0,
            calls=0,
            full_collections=0,
            young_collections=0,
            unreachable_objects=0,
            collection_seconds=0.0,
            release_seconds=0.0,
            device_release_seconds=0.0,
            sector_density_seconds=0.0,
        )
        # Devices and their pool bytes in use after the last full collection.
        self._collected_use = None
        # Set by a caller that waits for the device threads itself, behind
        # work of its own (:meth:`release_joined`), what it waits for, and
        # what the density that left the release to it still held on a
        # device: kept until the wait.
        self.caller_joins_release = False
        self._releasing = ()
        self._held = None

    def __call__(
        self,
        wavefunctions: CuPySymmetryOrbitals,
        occupations: np.ndarray,
        volume_element: float,
    ) -> np.ndarray:
        if not isinstance(wavefunctions, CuPySymmetryOrbitals):
            return self.device_builder(
                wavefunctions, occupations, volume_element
            )

        if wavefunctions.scaled_wedge_vectors is not None:
            wedge_density = self.device_builder(
                wavefunctions.scaled_wedge_vectors,
                occupations,
                volume_element,
            )
        else:
            if (
                wavefunctions.representation_columns is None
                or wavefunctions.sector_vectors is None
                or wavefunctions.sector_orbits is None
                or wavefunctions.sector_scales is None
                or wavefunctions.wedge_size is None
            ):
                raise RuntimeError("lazy symmetry orbitals are incomplete")
            sectors_started = perf_counter()
            wedge_density = np.zeros(
                wavefunctions.wedge_size, dtype=np.float64
            )
            selected = None
            # rho is invariant even when psi changes sign between symmetry
            # images.  Accumulate each representation directly from the Ritz
            # vectors the sector eigensolver already owns:
            #   rho_w = (2/h^3) sum_r sum_{n in r}
            #           f_n |q_{r,w,n}|^2 / |O_w|.
            # This is algebraically identical to first scattering all scaled
            # orbitals into a dense wedge matrix, but avoids that large copy.
            for representation, vectors in enumerate(
                wavefunctions.sector_vectors
            ):
                output_columns = np.flatnonzero(
                    wavefunctions.representations == representation
                )
                if output_columns.size == 0:
                    continue
                source_columns = wavefunctions.representation_columns[
                    output_columns
                ]
                # Eigenvalues within each representation are sorted, so the
                # globally selected states form a prefix.  Preserve a
                # zero-copy view in the common case; retain exact advanced
                # indexing for defensive support of a non-prefix selection.
                prefix = np.array_equal(
                    source_columns,
                    np.arange(source_columns.size, dtype=source_columns.dtype),
                )
                if isinstance(vectors, DistributedBasis):
                    # Each device reduces its own columns; no orbital moves.
                    if not prefix:
                        raise RuntimeError(
                            "a distributed sector basis supports only leading-column selections"
                        )
                    sector_density = vectors.density(
                        self.device_builder,
                        occupations[output_columns],
                        volume_element,
                    )
                else:
                    device = getattr(vectors, "device", None)
                    context = device if hasattr(device, "__enter__") else nullcontext()
                    with context:
                        if prefix:
                            selected = vectors[:, : source_columns.size]
                        else:
                            selected = vectors[:, source_columns]
                    sector_density = self.device_builder(
                        selected,
                        occupations[output_columns],
                        volume_element,
                    )
                scales = wavefunctions.sector_scales[representation]
                wedge_density[
                    wavefunctions.sector_orbits[representation]
                ] += sector_density * scales * scales
            self._release_large_unused_pool(wavefunctions)
            if self._releasing:
                # The caller waits for the device threads.  The sector
                # vectors and a selection that is no view of them are kept
                # until it does: freed now, beside a thread that empties
                # their pool, they would stay cached or go back to the
                # driver as the two threads happen to meet.  Freed behind
                # the release they stay cached, as where the builder waits.
                self._held = (wavefunctions, selected)
            self.pool_release["sector_density_seconds"] += (
                perf_counter() - sectors_started
            )
        if self.reducer is not None:
            return self.reducer.field(wedge_density)
        expansion_started = perf_counter()
        full_to_wedge = np.asarray(
            wavefunctions.full_to_wedge, dtype=np.int64
        )
        density = np.ascontiguousarray(wedge_density[full_to_wedge])
        if self.timing_stats is not None:
            self.timing_stats.density_seconds += (
                perf_counter() - expansion_started
            )
        return density

    def _release_large_unused_pool(
        self,
        wavefunctions: CuPySymmetryOrbitals,
    ) -> None:
        """Return cached ChebDav workspaces after a large sector solve.

        CuPy's default allocator deliberately caches freed blocks.  That is a
        speed win for ordinary systems, but a first ChebDav solve near the
        WDDM commit limit can retain many gigabytes of fragmented temporary
        workspaces.  A later SUBSPACE solve then fails despite the live Ritz
        vectors themselves fitting.  ``free_all_blocks`` touches only unused
        pool blocks; arrays referenced by the persistent eigensolver state are
        unaffected.

        The pools of several devices are emptied by the threads of those
        devices unless ``PARSEC_CUPY_DENSITY_RELEASE=serial``
        (:func:`density_release`).

        A sector whose basis several devices share returns nothing: the
        blocks in the pools of its devices are those that its next step
        takes again
        (:func:`distributed_state.shared_pool_release_requested`, which
        also names the setting that empties them as before).  A process
        whose sectors are all shared, as on 8 and 16 GPUs, therefore
        collects nothing and releases nothing here, the pinned pool
        included: its cyclic garbage waits for the collector of the
        interpreter and the pinned pool keeps its blocks, 0.3 to 1.1 GiB
        of host memory more on the root rank of the runs on A100 nodes.
        """

        # A release that no caller has waited for is complete before the next.
        if self._releasing:
            self.release_joined()
        vectors = wavefunctions.sector_vectors
        if not vectors:
            return
        # The limit from which a finished sector returns its blocks as well.
        threshold = _pool_release_orbital_bytes()
        live_bytes = sum(int(getattr(item, "nbytes", 0)) for item in vectors)
        if live_bytes < threshold:
            return
        try:
            from ..backends.cupy import require_cupy

            cp, _ = require_cupy()
        except Exception:
            return
        record = self.pool_release
        # The sectors of this process, which has no vectors of those that
        # another MPI rank solves, and among them the shared ones, whose
        # devices keep what their pools hold.
        held = [item for item in vectors if item is not None]
        kept = [
            item
            for item in held
            if isinstance(item, DistributedBasis) and not self.shared_release
        ]
        if kept:
            record["shared_kept"] += 1
            if len(kept) == len(held):
                return
        # Device arrays created by the density-builder calls above are out of
        # scope before collection.  Release each participating device pool in
        # deterministic device order; pinned host staging uses one global pool.
        device_ids = sorted(
            {
                int(item.device.id)
                for item in vectors
                if isinstance(item, cp.ndarray)
            }
            | {
                int(device)
                for item in vectors
                if isinstance(item, DistributedBasis) and self.shared_release
                for device in item.group.devices
            }
        )
        started = perf_counter()
        pool = cp.get_default_memory_pool()

        def in_use() -> tuple:
            used = []
            for device_id in device_ids:
                with cp.cuda.Device(device_id):
                    used.append(int(pool.used_bytes()))
            return tuple(device_ids), tuple(used)

        # Host-kept sector states name no device: nothing to compare then.
        watched = (
            self.collection == "changed"
            and bool(device_ids)
            and cp.cuda.get_allocator() == pool.malloc
        )
        full = not watched or record["calls"] < 2
        if not full:
            record["unreachable_objects"] += gc.collect(1)
            record["young_collections"] += 1
            full = in_use() != self._collected_use
        if full:
            record["unreachable_objects"] += gc.collect()
            record["full_collections"] += 1
            self._collected_use = in_use() if watched else None
        record["calls"] += 1
        collected = perf_counter()
        record["collection_seconds"] += collected - started
        if self.release == "serial" or len(device_ids) < 2:
            for device_id in device_ids:
                with cp.cuda.Device(device_id):
                    pool.free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        else:

            def empty(device_id: int) -> float:
                begun = perf_counter()
                with cp.cuda.Device(device_id):
                    pool.free_all_blocks()
                return perf_counter() - begun

            # Every device by its own thread, side by side; the pinned pool
            # here meanwhile.  See density_release.
            self._releasing = tuple(
                _pool(device_id).submit(empty, device_id)
                for device_id in device_ids
            )
            cp.get_default_pinned_memory_pool().free_all_blocks()
            if not self.caller_joins_release:
                self._join_release()
        record["release_seconds"] += perf_counter() - collected

    def _join_release(self) -> None:
        """Wait for the device threads; the first error is raised when all have ended."""

        futures, self._releasing = self._releasing, ()
        wait(futures)
        # What the density held on a device goes back to its pool now.
        self._held = None
        self.pool_release["device_release_seconds"] += sum(
            future.result() for future in futures
        )

    def release_joined(self) -> None:
        """Wait until the pools that the last density handed to their device threads are empty.

        For a caller that set ``caller_joins_release``: the builder then
        returns the density while the device threads still release, and the
        caller waits here once work of its own that touches no device is
        done.  An MPI rank does so behind the collective calls of its density
        command (:meth:`MPISectorContext._execute_counted`), in which it
        sends its density and waits for those of the other ranks.

        Until it has waited, the caller uses no device, frees no array of
        one and does not compute in the interpreter.  The release is to be
        complete before anything else takes memory, as it is where the
        builder waits itself.  A thread that frees an array while the pool
        of its device is being emptied stands at the lock of that pool
        until it is empty, and the array stays cached; freed just before,
        it goes back to the driver with the rest.  The builder therefore
        keeps what the density still holds on a device, the sector vectors
        and a selection that is no view of them, until the wait: on a
        workstation such a density came back after 32 to 40 ms, when the
        1.4 GiB pool of its selection was empty, and comes back after 24
        to 28 now.  The production route selects leading columns, views
        that free nothing.  And CuPy takes the interpreter lock again
        around every call into the driver: beside a thread in a Python
        loop the same workstation returned 24 blocks of 48 MiB in 351 to
        384 ms instead of 5 to 7.  The collective calls of MPI give that
        lock up while they wait.

        The wait counts as seconds of the release and of the sector-wise
        density.  An error of a device thread is raised by it once all
        threads have ended.  Nothing to wait for is no error: a density
        that released nothing, one device, or
        ``PARSEC_CUPY_DENSITY_RELEASE=serial``.
        """

        if not self._releasing:
            return
        started = perf_counter()
        try:
            self._join_release()
        finally:
            waited = perf_counter() - started
            self.pool_release["release_seconds"] += waited
            self.pool_release["sector_density_seconds"] += waited


__all__ = ["CuPySymmetryDensityBuilder"]
