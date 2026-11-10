"""The exchange of a hybrid functional divided among MPI ranks (``SCF/hybrid_mpi.py``).

The ranks are threads of this process with a communicator of their own
(broadcasts, point-to-point messages and gathers through host copies), all on
the device of the laptop: each owns the symmetry sectors ``sector % size``,
receives replicas of the others' orbitals, runs its share of the pair tiles
and sends the parts of ``W`` of the other ranks' sectors to their owner.
``W``, ``E_x``, ``xi`` of ACE and the band energies of the operator must be
those of one process (:func:`apply_device_exchange`,
:func:`compress_device_exchange`) to round-off, for builds of all columns, of
the occupied ones and their completion, with two and three ranks and with a
rank that owns no sector.  The tile plan of one rank is that of one process.
A broadcast whose data lands late on its stream (a busy device) still gives
the tiles that read it the data; a failure or a shortage of device memory on
one rank inside a command reaches every rank, with no rank left waiting in a
collective call (the commands are driven through the real
:class:`~parsec_python.acceleration.experimental.mpi_scf.MPISectorContext`).
"""

from __future__ import annotations

from collections import defaultdict, deque
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available


class _World:
    """The shared state of the thread communicators of one test."""

    def __init__(self, size: int, timeout: float = 120.0) -> None:
        self.size = size
        self.timeout = float(timeout)
        self.condition = threading.Condition()
        self.collectives: dict = {}
        self.queues: dict = defaultdict(deque)
        self.duplicates: dict = {}

    def collective(self, sequence, rank, kind, value, root=0):
        with self.condition:
            state = self.collectives.setdefault(sequence, dict(kind=kind, values={}, read=0))
            if state["kind"] != kind:
                raise RuntimeError(f"collectives called in different order: {state['kind']} and {kind}")
            state["values"][rank] = value
            if len(state["values"]) == self.size:
                values = [state["values"][r] for r in range(self.size)]
                state["result"] = values[root] if kind == "broadcast" else values
                self.condition.notify_all()
            if not self.condition.wait_for(lambda: "result" in state, timeout=self.timeout):
                raise TimeoutError(f"collective {sequence} {kind} incomplete")
            state["read"] += 1
            result = state["result"]
            if state["read"] == self.size:
                del self.collectives[sequence]
            return result

    def duplicate(self, number: int) -> "_World":
        with self.condition:
            world = self.duplicates.get(number)
            if world is None:
                world = self.duplicates[number] = _World(self.size, self.timeout)
            return world


class _Done:
    def Wait(self) -> None:
        return None


def _buffer(message):
    """The array of an MPI message (a bare buffer or ``[buffer, datatype]``)."""

    return message[0] if isinstance(message, (list, tuple)) else message


def _host(buffer):
    import cupy as cp

    buffer = _buffer(buffer)

    return cp.asnumpy(buffer) if isinstance(buffer, cp.ndarray) else np.array(buffer, copy=True)


def _store(buffer, values) -> None:
    import cupy as cp

    buffer = _buffer(buffer)

    if isinstance(buffer, cp.ndarray):
        buffer[...] = cp.asarray(values)
    else:
        buffer[...] = values


class _Comm:
    def __init__(self, world: _World, rank: int) -> None:
        self.world, self.rank, self.size = world, rank, world.size
        self.sequence = 0
        self.duplicates = 0

    def _call(self, kind, value, root=0):
        sequence = self.sequence
        self.sequence += 1
        return self.world.collective(sequence, self.rank, kind, value, root)

    def allgather(self, value):
        return list(self._call("gather", value))

    def gather(self, value, root=0):
        values = self._call("gather", value)
        return list(values) if self.rank == root else None

    def bcast(self, value, root=0):
        return self._call("broadcast", value, root)

    def Barrier(self) -> None:
        self._call("gather", None)

    def Bcast(self, buffer, root=0) -> None:
        values = self._call("broadcast", _host(buffer) if self.rank == root else None, root)
        if self.rank != root:
            _store(buffer, values)

    def Isend(self, buffer, dest, tag):
        with self.world.condition:
            self.world.queues[(self.rank, int(dest), int(tag))].append(_host(buffer))
            self.world.condition.notify_all()
        return _Done()

    def _take(self, source, tag):
        key = (int(source), self.rank, int(tag))
        with self.world.condition:
            if not self.world.condition.wait_for(lambda: self.world.queues[key], timeout=self.world.timeout):
                raise TimeoutError(f"no message {key}")
            return self.world.queues[key].popleft()

    def Recv(self, buffer, source, tag) -> None:
        _store(buffer, self._take(source, tag))

    def send(self, value, dest, tag) -> None:
        with self.world.condition:
            self.world.queues[(self.rank, int(dest), int(tag))].append(value)
            self.world.condition.notify_all()

    def recv(self, source, tag):
        return self._take(source, tag)

    def Dup(self):
        number = self.duplicates
        self.duplicates += 1
        return _Comm(self.world.duplicate(number), self.rank)

    def Abort(self, code=1):
        raise RuntimeError(f"rank {self.rank} aborted")


def _run_ranks(size, work):
    """``work(rank)`` on ``size`` threads at once; their results by rank (a failure is raised)."""

    results, errors = [None] * size, []

    def run(rank):
        import cupy as cp

        try:
            with cp.cuda.Device(0):
                results[rank] = work(rank)
        except BaseException as error:  # noqa: BLE001 -- reported below
            errors.append((rank, error))

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0][1]
    return results


def _selection(orbitals):
    """The global selection of a channel's states: their sector and sector column."""

    representations = np.zeros(orbitals.state_count, dtype=np.int32)
    columns = np.zeros(orbitals.state_count, dtype=np.int32)
    for sector in orbitals.sectors:
        representations[sector.states] = sector.representation
        columns[sector.states] = sector.local
    return representations, columns


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DistributedExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.tests.test_exchange_parity import sector_channel
        from parsec_python.models import GridSettings

        cls.grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        # D2: four sectors; a few occupied and empty states in each.
        cls.orbitals, cls.reduction = sector_channel(cls.grid, "D2", states=14, occupied=9)

    def _service(self, rank, comm, *, batch=3, real_exchange=False):
        from parsec_python.acceleration.Hamiltonian.exchange_device import DevicePairSolvers
        from parsec_python.acceleration.SCF.hybrid_mpi import MPIExchangeService
        from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition

        decomposition = ReflectionRepresentationDecomposition.build(self.grid, self.reduction)
        count = decomposition.representation_count
        topology = tuple({sector: (0,) for sector in range(owner, count, comm.size)} for owner in range(comm.size))
        retained = {
            sector.representation: SimpleNamespace(vectors=sector.vectors, handover=None)
            for sector in self.orbitals.sectors
            if sector.representation % comm.size == rank
        }
        sectors = SimpleNamespace(
            decomposition=decomposition,
            _sector_device_ids=(0,) * count,
            _mpi_retained_results=retained,
            _operators=[None] * count,
        )
        context = SimpleNamespace(comm=comm, topology=topology)
        if real_exchange:
            # The rank's DeviceExchange as the runner makes it: its own pair_solvers on a system that names a
            # grid only (no input; the periodic check of the GPU path of periodic cells must pass it by).
            from parsec_python.SCF.hybrid import hybrid_definition
            from parsec_python.acceleration.SCF.hybrid import DeviceExchange
            from parsec_python.models import SCFSettings

            exchange = DeviceExchange(backend=None, sectors=sectors, devices=(0,), multipole_order=6,
                                      batch_size=batch, volume_element=float(self.grid.volume_element))
            definition = hybrid_definition(SCFSettings(xc_functional="pbe0"))
            service = MPIExchangeService(context, exchange, sectors, self.grid, definition)
            self.assertIsInstance(service.solvers, DevicePairSolvers)
            return service
        solvers = DevicePairSolvers(self.grid, (("coulomb", 1.0, None),), 6, batch, parity=True)
        exchange = SimpleNamespace(devices=(0,), expansions={}, bound=None, _solvers={},
                                   operators=lambda: {},
                                   pair_solvers=lambda _system, _definition: solvers)
        return MPIExchangeService(context, exchange, sectors, self.grid, None)

    def _reference(self, *, batch=3, two_steps=False):
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            apply_device_exchange,
        )

        solvers = DevicePairSolvers(self.grid, (("coulomb", 1.0, None),), 6, batch, parity=True)
        if two_steps:
            occupied = apply_device_exchange(self.orbitals, solvers, (0,), columns="occupied")
            return occupied, apply_device_exchange(self.orbitals, solvers, (0,), columns="all", known=occupied)
        return None, apply_device_exchange(self.orbitals, solvers, (0,), columns="all")

    def _distributed(self, size, *, two_steps=False, real_exchange=False):
        import cupy as cp

        representations, columns = _selection(self.orbitals)
        world = _World(size)
        selection = dict(id=1, channel=None, representations=representations, columns=columns,
                         occupations=self.orbitals.occupations, spin_weight=2.0,
                         volume_element=float(self.grid.volume_element))

        def work(rank):
            service = self._service(rank, _Comm(world, rank), real_exchange=real_exchange)
            service.handle("exx_channel", dict(channels=[selection]))
            numbers = []
            if two_steps:
                numbers.append(service.handle("exx_apply", dict(channel=1, id=2, columns="occupied", known=None)))
                numbers.append(service.handle("exx_apply", dict(channel=1, id=3, columns="all", known=2)))
                key = 3
            else:
                numbers.append(service.handle("exx_apply", dict(channel=1, id=3, columns="all", known=None)))
                key = 3
            application = service.objects[key]
            owned = {
                sector.representation: cp.asnumpy(application.applied[index])
                for index, sector in enumerate(application.orbitals.sectors)
                if service.owners[sector.representation] == rank
            }
            records = service.handle(
                "exx_compress", dict(channels=[dict(application=key, id=4, energy=1.0)], fraction=0.25)
            )
            xi = {representation: cp.asnumpy(block)
                  for representation, block in service.objects[4].projectors.items()}
            band = service.handle("exx_band", dict(pairs=[dict(operator=4, channel=1)]))
            record = service.record()
            # After the SCF nothing of the loop stays (the operator was released by the loop).
            service.objects[4].release()
            service.handle("exx_trim", {})
            return dict(numbers=numbers, applied=owned, records=records, xi=xi, band=band,
                        record=record, left=len(service.objects))

        return _run_ranks(size, work)

    def _compare(self, size, *, two_steps=False, real_exchange=False):
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.exchange_device import compress_device_exchange

        occupied, reference = self._reference(two_steps=two_steps)
        applied = {sector.representation: cp.asnumpy(reference.applied[index])
                   for index, sector in enumerate(self.orbitals.sectors)}
        results = self._distributed(size, two_steps=two_steps, real_exchange=real_exchange)
        root = results[0]
        energy = root["numbers"][0]["energy"]
        self.assertAlmostEqual(energy, reference.energy, delta=1e-12 * abs(reference.energy))
        self.assertEqual(sum(item["pair_solves"] for item in root["numbers"]), reference.pair_solves)
        scale = max(float(np.abs(block).max()) for block in applied.values())
        seen = set()
        for result in results:
            for representation, block in result["applied"].items():
                self.assertNotIn(representation, seen)
                seen.add(representation)
                self.assertLess(float(np.abs(block - applied[representation]).max()), 1e-12 * scale)
        self.assertEqual(seen, set(applied))
        operator = compress_device_exchange(reference, fraction=0.25)
        for result in results:
            for representation, block in result["xi"].items():
                expected = cp.asnumpy(operator.projectors[representation])
                self.assertLess(float(np.abs(block - expected).max()), 1e-11 * float(np.abs(expected).max()))
        self.assertEqual(root["records"][0]["rank"], operator.rank)
        self.assertAlmostEqual(root["records"][0]["asymmetry"], operator.asymmetry, delta=1e-12)
        # sum_i f_i <q_i|xi xi^T|q_i> per sector, as band_energy sums it.
        expected_band = operator.band_energy(self.orbitals, scaled=False)
        sums = root["band"][0]
        total = 0.0
        for key in sorted(sums):
            total += sums[key]
        self.assertAlmostEqual(-2.0 * total, expected_band, delta=1e-12 * abs(expected_band))
        self.assertEqual([result["left"] for result in results], [0] * size)
        return results

    def test_service_with_the_ranks_own_pair_solvers(self) -> None:
        """The service of every rank built with the rank's DeviceExchange (its pair_solvers not replaced)."""

        self._compare(2, real_exchange=True)

    def test_all_columns_on_two_and_three_ranks(self) -> None:
        for size in (2, 3):
            with self.subTest(ranks=size):
                results = self._compare(size)
                moved = sum(item["record"]["transport_bytes"]["reduce_sent"] for item in results)
                self.assertGreater(moved, 0)

    def test_occupied_columns_then_their_completion(self) -> None:
        import os

        # With the tiles of the own sectors during the broadcasts (the default) and after them.
        for overlap in ("1", "0"):
            saved = os.environ.get("PARSEC_MPI_EXCHANGE_OVERLAP")
            os.environ["PARSEC_MPI_EXCHANGE_OVERLAP"] = overlap
            try:
                with self.subTest(overlap=overlap):
                    self._compare(2, two_steps=True)
            finally:
                if saved is None:
                    os.environ.pop("PARSEC_MPI_EXCHANGE_OVERLAP", None)
                else:
                    os.environ["PARSEC_MPI_EXCHANGE_OVERLAP"] = saved

    def test_a_rank_without_sectors_solves_tiles(self) -> None:
        # Five ranks and four sectors: rank 4 owns none and still takes tiles.
        results = self._compare(5)
        self.assertEqual(results[4]["applied"], {})
        self.assertGreater(results[4]["record"]["pair_solves"], 0)

    def test_replicas_are_in_place_before_a_tile_reads_them(self) -> None:
        # A device busy on the stream that writes a received replica (a spin kernel before every
        # broadcast lands): the tiles started meanwhile on their own streams must still read the
        # replica's data, in the first build and in the completion that grows the replicas.
        import os
        from unittest.mock import patch

        import cupy as cp

        from parsec_python.acceleration.tests import test_exchange_mpi as module

        spin = cp.RawKernel(
            r"""
            extern "C" __global__ void spin(long long cycles) {
                long long start = clock64();
                while (clock64() - start < cycles) { }
            }
            """,
            "spin",
        )
        original = module._store

        def late_store(buffer, values):
            target = _buffer(buffer)
            if not isinstance(target, cp.ndarray):
                return original(buffer, values)
            data = cp.asarray(values)  # (a synchronous copy: the host waits for it, not for what follows)
            spin((1,), (1,), (np.int64(400_000_000),))
            target[...] = data  # lands on the stream behind the spin, after this returns

        # Device arrays to the communicator (as GPU-aware MPI takes them), the overlap on.
        settings = {"PARSEC_MPI_EXCHANGE_OVERLAP": "1", "PARSEC_MPI_EXCHANGE_TRANSPORT": "cuda"}
        with patch.dict(os.environ, settings), patch.object(module, "_store", late_store):
            self._compare(2, two_steps=True)

    # ----- a failure on one rank inside a command ------------------------------------

    def _command(self, size, command, payload, *, setup=None, inject=None):
        """``command`` through the sector context of ``size`` thread ranks after ``setup`` (direct commands).

        A ``setup`` (``()`` too) first registers the channel ``1`` of the test orbitals.
        ``inject(rank, service)`` changes a rank before the command.  Returns per rank what the command
        gave (the root) or how the worker loop ended, and the keys the service holds afterwards.
        """

        from parsec_python.acceleration.experimental.mpi_scf import MPISCFError, MPISectorContext

        representations, columns = _selection(self.orbitals)
        selection = dict(id=1, channel=None, representations=representations, columns=columns,
                         occupations=self.orbitals.occupations, spin_weight=2.0,
                         volume_element=float(self.grid.volume_element))
        world = _World(size, timeout=60.0)
        stub = SimpleNamespace(restore_memory_allocator=lambda: None)

        def work(rank):
            comm = _Comm(world, rank)
            service = self._service(rank, comm)
            if setup is not None:
                service.handle("exx_channel", dict(channels=[selection]))
            for name, arguments in setup or ():
                service.handle(name, arguments)
            if inject is not None:
                inject(rank, service)
            context = MPISectorContext(comm, exchange=True)
            context.exchange_service = service
            try:
                if rank == 0:
                    outcome = context.execute(stub, command, payload)
                    context.stop_workers()
                else:
                    context.worker_loop(stub, None)
                    outcome = "stopped"
            except MPISCFError as error:
                outcome = f"MPISCFError: {error}"
            return outcome, sorted(service.objects)

        return _run_ranks(size, work)

    _ALL = ("exx_apply", dict(channel=1, id=3, columns="all", known=None))
    _OCCUPIED = ("exx_apply", dict(channel=1, id=2, columns="occupied", known=None))

    def test_a_window_product_whose_owner_fails_raises_everywhere(self) -> None:
        from parsec_python.SCF.fock_diis import OccupiedBlock

        def inject(rank, service):
            if rank == 1:  # the owner holds the right block, not the left one
                service.objects[11] = OccupiedBlock(np.eye(3), np.ones(3))

        results = self._command(2, "exx_overlap", dict(left=10, right=11, owner=1), inject=inject)
        for outcome, _keys in results:
            self.assertIn("KeyError", str(outcome))

    def test_an_ace_that_fails_on_one_rank_raises_everywhere(self) -> None:
        # An error, and a shortage of memory (the extrapolated build of the outer DIIS, whose
        # refusal the loop takes as plain steps): no rank keeps an operator the root does not track.
        failures = (
            np.linalg.LinAlgError("M = -Psi^T V_x Psi is not positive definite (test)"),
            MemoryError("out of memory in the Cholesky factor (test)"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):

                def inject(rank, service, failure=failure):
                    if rank == 1:

                        def broken(*_arguments):
                            raise failure

                        service._compress_application = broken

                results = self._command(
                    2, "exx_compress", dict(channels=[dict(application=3, id=4, energy=1.0)], fraction=0.25),
                    setup=(self._ALL,), inject=inject,
                )
                (root, root_keys), (worker, worker_keys) = results
                if isinstance(failure, MemoryError):
                    self.assertIn("test", root["memory_error"])
                    self.assertEqual(worker, "stopped")
                else:
                    for outcome in (root, worker):
                        self.assertIn("LinAlgError", str(outcome))
                self.assertNotIn(4, root_keys)
                self.assertNotIn(4, worker_keys)

    def test_host_plans_that_failed_on_one_rank_raise_everywhere_with_their_message(self) -> None:
        from concurrent.futures import Future

        def inject(rank, service):
            if rank == 1:
                failed = Future()
                failed.set_exception(RuntimeError("the host plans of the pair solver failed (test)"))
                service.solvers._built, service.solvers._future = None, failed

        results = self._command(2, "exx_apply", dict(self._ALL[1]), setup=(), inject=inject)
        for outcome, _keys in results:
            self.assertIn("rank 1: RuntimeError: the host plans of the pair solver failed (test)", str(outcome))

    def test_a_band_energy_that_fails_on_one_rank_raises_everywhere(self) -> None:
        def inject(rank, service):
            if rank == 1:
                del service.objects[1]

        results = self._command(
            2, "exx_band", dict(pairs=[dict(operator=4, channel=1)]),
            setup=(self._ALL, ("exx_compress", dict(channels=[dict(application=3, id=4, energy=1.0)], fraction=0.25))),
            inject=inject,
        )
        for outcome, _keys in results:
            self.assertIn("KeyError", str(outcome))

    def test_a_completion_without_room_on_one_rank_is_refused_everywhere_and_keeps_its_build(self) -> None:
        # The memory rule, and the allocation of a grown replica, of one rank: a refusal with the numbers
        # for the root (MemoryError there), the occupied build kept on every rank.
        def no_room(rank, service):
            if rank == 1:

                def refuse(*_arguments):
                    raise MemoryError("the exact exchange of this hybrid needs 9.9 GiB on device 0 (test)")

                service._require_memory = refuse

        def no_replica(rank, service):
            if rank == 1:

                def refuse(*_arguments):
                    raise MemoryError("out of memory allocating a replica (test)")

                service._prepare_replicas = refuse

        for inject in (no_room, no_replica):
            with self.subTest(inject=inject.__name__):
                results = self._command(
                    2, "exx_apply", dict(channel=1, id=3, columns="all", known=2),
                    setup=(self._OCCUPIED,), inject=inject,
                )
                (root, root_keys), (worker, worker_keys) = results
                self.assertIn("memory_error", root)
                self.assertIn("test", root["memory_error"])
                self.assertEqual(worker, "stopped")
                self.assertIn(2, root_keys)
                self.assertIn(2, worker_keys)
                self.assertNotIn(3, root_keys)

    def test_a_final_trim_lets_the_plans_go(self) -> None:
        def work(rank):
            service = self._service(rank, _Comm(world, rank))
            service.solvers.solvers(0)
            service.handle("exx_trim", {"final": True})
            return len(service.solvers._devices), service.bound

        world = _World(1)
        self.assertEqual(_run_ranks(1, work), [(0, None)])

    def test_a_trim_keeps_the_bound_operator_and_a_final_trim_releases_it(self) -> None:
        representations, columns = _selection(self.orbitals)
        selection = dict(id=1, channel=None, representations=representations, columns=columns,
                         occupations=self.orbitals.occupations, spin_weight=2.0,
                         volume_element=float(self.grid.volume_element))
        calls = []
        sector_operator = SimpleNamespace(bind_exchange=lambda projectors, *scale: calls.append(projectors is not None))

        def work(rank):
            service = self._service(rank, _Comm(world, rank))
            service.exchange.operators = lambda: {0: sector_operator}
            service.handle("exx_channel", dict(channels=[selection]))
            service.handle(*self._ALL)
            service.handle("exx_compress", dict(channels=[dict(application=3, id=4, energy=1.0)], fraction=0.25))
            service.handle("exx_bind", dict(id=4))
            operator = service.objects[4]
            service.handle("exx_trim", {})
            kept = (sorted(service.objects), service.bound is operator, operator.released)
            service.handle("exx_trim", {"final": True})
            return kept, (sorted(service.objects), service.bound, operator.released, operator.projectors)

        world = _World(1)
        [(kept, final)] = _run_ranks(1, work)
        self.assertEqual(kept, ([4], True, False))
        self.assertEqual(final, ([], None, True, {}))
        self.assertEqual(calls, [True, False])

    def test_a_build_counts_two_chunks_of_the_messages_of_an_owned_sector(self) -> None:
        # Rank 0 of two: it owns sector 0 (100 rows, 5 columns to send) and touches sector 1 of rank 1
        # (50 rows, 4 columns it lacks).  With a chunk of two columns of sector 0 a message holds 2 x 1600 bytes.
        def sector(rows, count):
            return SimpleNamespace(expansion=SimpleNamespace(rows=rows), device=0, vectors=None,
                                   local=np.arange(count, dtype=np.int64))

        entry = SimpleNamespace(
            orbitals=SimpleNamespace(sectors=[sector(100, 5), sector(50, 4)], full_size=1000),
            owners=[0, 1], shipped={},
        )
        wanted = {0: list(range(5)), 1: list(range(4))}
        world = _World(2)

        def work(rank):
            service = self._service(rank, _Comm(world, rank))
            service.transport.chunk_bytes = 1600
            return service._build_bytes(entry, wanted, {1}, None, None)

        needed = _run_ranks(2, work)[0]
        from parsec_python.acceleration.Hamiltonian.exchange_device import _TILE_ARRAYS

        work_bytes = 8 * 3 * _TILE_ARRAYS * 1000
        self.assertEqual(needed, {0: 8 * 100 * 5 + 2 * 1600 + 8 * 50 * 4 + 8 * 50 * 4 + work_bytes})

    def test_one_rank_plans_the_tiles_of_one_process(self) -> None:
        from parsec_python.acceleration.Hamiltonian.exchange_device import assign_tiles, plan_exchange

        plan = plan_exchange(self.orbitals, 3, columns="all")
        local = assign_tiles(plan.tiles, (0, 1))
        distributed = assign_tiles(plan.tiles, ((0, 0), (0, 1)))
        self.assertEqual(
            [[(tile.first, tile.second) for tile in local[device]] for device in (0, 1)],
            [[(tile.first, tile.second) for tile in distributed[(0, device)]] for device in (0, 1)],
        )


class SpinFallbackTests(unittest.TestCase):
    def test_the_fallback_frees_the_exchange_of_every_rank_before_the_full_grid(self) -> None:
        from unittest.mock import patch

        from parsec_python.acceleration import driver

        calls = []
        exchange = SimpleNamespace(
            bind=lambda channel: calls.append(("bind", channel)),
            release_build_memory=lambda final=False: calls.append(("release", final)),
            _solvers={"plans": 1},
            expansions={"rows": 1},
        )
        system = SimpleNamespace(
            backend=SimpleNamespace(symmetry_eigensolver=None),
            exchange_device=exchange,
            reference=SimpleNamespace(atoms=()),
        )
        fallback = driver._SpinSectorFallback(
            SimpleNamespace(pseudopotentials={}), (), backend="cupy", symmetry="auto", symmetry_cache_directory=None
        )
        report = SimpleNamespace(reason="test", measure="m", largest=1.0, tolerance=0.1)
        with patch.object(driver._SpinSectorFallback, "_problem_of", return_value=SimpleNamespace(pseudopotentials={})), \
                patch.object(driver, "prepare_single_point", side_effect=RuntimeError("full grid prepared")):
            with self.assertRaisesRegex(RuntimeError, "full grid prepared"):
                fallback.replace_with_full_grid(system, "start", report)
        self.assertEqual(calls, [("bind", None), ("release", True)])
        self.assertEqual((exchange._solvers, exchange.expansions), ({}, {}))


class SectorGroupTests(unittest.TestCase):
    def test_exchange_groups_put_every_sector_on_one_device_and_allow_ranks_without_sectors(self) -> None:
        from parsec_python.acceleration.experimental.mpi_scf import sector_device_groups

        self.assertEqual(sector_device_groups(4, 2, 0, (0, 1, 2, 3), exchange=True), {0: (0,), 2: (1,)})
        self.assertEqual(sector_device_groups(4, 4, 3, (0, 1, 2, 3), exchange=True), {3: (0,)})
        self.assertEqual(sector_device_groups(4, 6, 5, (0, 1, 2, 3), exchange=True), {})
        self.assertEqual(sector_device_groups(1, 4, 0, (0,), exchange=True), {0: (0,)})
        # Without the exchange a rank has a group of devices per sector, as before.
        self.assertEqual(sector_device_groups(4, 4, 3, (0, 1, 2, 3)), {3: (0, 1, 2, 3)})

    def test_exchange_context_plans_what_it_configures_and_shares_no_basis(self) -> None:
        from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext

        world = _World(1)
        comm = SimpleNamespace(rank=1, size=3, allgather=None)
        context = MPISectorContext(comm, exchange=True)
        self.assertEqual(context.planned_groups(4, (0, 1, 2, 3)), {1: (0,)})
        # The one sector of a run without symmetry lies on the root; the other ranks own none.
        self.assertEqual(context.planned_groups(1, (0, 1, 2, 3)), {})
        self.assertFalse(context._shared(1))
        semilocal = MPISectorContext(comm)
        self.assertTrue(semilocal._shared(1))
        self.assertEqual(semilocal.planned_groups(4, (0, 1, 2, 3)), {1: (0, 1, 2, 3)})
        # Configuration agrees with the plan on every rank (one rank here).
        single = MPISectorContext(_Comm(world, 0), exchange=True)
        self.assertEqual(single.configure(4, (0, 1)), single.planned_groups(4, (0, 1)))
        self.assertEqual(single.topology, ({0: (0,), 1: (1,), 2: (0,), 3: (1,)},))

    def test_transport_follows_the_mpi_library_unless_named(self) -> None:
        import os
        from unittest.mock import patch

        from parsec_python.acceleration.SCF.hybrid_mpi import exchange_overlap, exchange_transport

        with patch.dict(os.environ, {"MPICH_GPU_SUPPORT_ENABLED": "1"}, clear=False):
            os.environ.pop("PARSEC_MPI_EXCHANGE_TRANSPORT", None)
            self.assertEqual(exchange_transport(), "cuda")
            os.environ["PARSEC_MPI_EXCHANGE_TRANSPORT"] = "host"
            self.assertEqual(exchange_transport(), "host")
            os.environ["PARSEC_MPI_EXCHANGE_TRANSPORT"] = "gpu"
            with self.assertRaisesRegex(ValueError, "PARSEC_MPI_EXCHANGE_TRANSPORT"):
                exchange_transport()
        with patch.dict(os.environ, {"MPICH_GPU_SUPPORT_ENABLED": "0", "PARSEC_MPI_EXCHANGE_TRANSPORT": "auto"}):
            self.assertEqual(exchange_transport(), "host")
        with patch.dict(os.environ, {"PARSEC_MPI_EXCHANGE_OVERLAP": "0"}):
            self.assertFalse(exchange_overlap())
        with patch.dict(os.environ, {"PARSEC_MPI_EXCHANGE_OVERLAP": "yes"}):
            with self.assertRaisesRegex(ValueError, "PARSEC_MPI_EXCHANGE_OVERLAP"):
                exchange_overlap()

    def test_runner_sees_a_hybrid_input_before_mpi_starts(self) -> None:
        import tempfile
        from pathlib import Path

        from parsec_python.acceleration.benchmarks.mpi_full_scf import _input_has_hybrid

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            for text, expected in (
                ("Correlation_Type: pbe0   # PBE0\n", True),
                ("correlation_type = HSE06\n", True),
                ("Correlation_Type: ca\n", False),
                ("Grid_Spacing: 0.3\n", False),
            ):
                path.write_text("Boundary_Sphere_Radius: 8.0\n" + text, encoding="utf-8")
                self.assertEqual(_input_has_hybrid(path), expected, text)
            self.assertFalse(_input_has_hybrid(Path(directory) / "missing.in"))

    def test_column_runs(self) -> None:
        from parsec_python.acceleration.SCF.hybrid_mpi import column_runs

        self.assertEqual(column_runs([0, 1, 2, 5, 6, 9]), [(0, 3), (5, 7), (9, 10)])
        self.assertEqual(column_runs([]), [])


if __name__ == "__main__":
    unittest.main()
