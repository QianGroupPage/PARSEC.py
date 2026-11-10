"""The Fock loop of a periodic cell with k-points on the GPUs of one node.

:func:`parsec_python.SCF.hybrid.run_hybrid_scf` is the loop, unchanged, and
its SCF loops are the k-point drivers of the reference path
(:func:`parsec_python.SCF.kpoints.run_scf_kpoints` and the spin-polarized
one), whose eigenproblems the k-point pool solves on the GPUs
(:class:`parsec_python.acceleration.SCF.kpoint_parallel.PooledKPointSolver`:
one worker per device, each k-point's Ritz subspace resident there).
:class:`DeviceBlochExchange` is the engine of the loop, the device
counterpart of :class:`parsec_python.SCF.hybrid_periodic.BlochExchange`:

- the orbitals of a result stay where the workers keep them (``keep_vectors``
  hands the device arrays, the Ritz states stay on their devices between the
  loops);
- ``W = V_x U`` of every k-point is formed on every device of the node
  (:func:`~parsec_python.acceleration.Hamiltonian.bloch_exchange_device.apply_device_bloch_exchange`)
  and compressed where the k-point's orbitals are
  (:func:`~parsec_python.acceleration.Hamiltonian.bloch_exchange_device.compress_device_bloch_exchange`);
- the operator of a k-point, ``-a_x xi xi^H`` with ``xi`` on its worker's
  device, joins the device Bloch operator of the worker
  (:class:`parsec_python.acceleration.Hamiltonian.device_bloch.DeviceBlochConverter`),
  so every product of the device eigensolver holds it;
- ``E_x``, ``E_x(P, P0)`` and the overlaps of the outer DIIS are formed where
  the orbitals are; only scalars and small matrices reach the host;
- the result of the loop holds host copies of the orbitals (forces, output).

The pair potentials are those of the host engine (its kernel and box,
:func:`parsec_python.SCF.hybrid_periodic.periodic_pair_potentials`), so the
loop gives the host loop's numbers to round-off.

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from parsec_python.SCF.fock_diis import OccupiedBlock, array_module
from parsec_python.SCF.hybrid_periodic import (
    BlochChannelExchange,
    BlochExchange,
    PeriodicFockOperator,
)
from parsec_python.V_xc.bloch_exchange import BlochOrbitalSet, time_reversal_partners
from parsec_python.V_xc.exact_exchange import OCCUPATION_THRESHOLD

from ..Hamiltonian.bloch_exchange_device import (
    DevicePeriodicBox,
    DevicePeriodicPairSolvers,
    apply_device_bloch_exchange,
    compress_device_bloch_exchange,
    device_bloch_channel,
    device_cross_energy,
)


#: Share of each device's memory that the window of the outer DIIS leaves free.
WINDOW_MARGIN = 0.10


def _split(pooled, blocks) -> list[np.ndarray]:
    values = np.asarray(pooled, dtype=np.float64)
    parts, offset = [], 0
    for block in blocks:
        count = int(block.shape[1])
        parts.append(values[offset : offset + count])
        offset += count
    if offset != values.size:
        raise ValueError("the occupations do not match the orbitals of the k-points")
    return parts


@dataclass
class DeviceBlochOccupiedBlock(OccupiedBlock):
    """The occupied Bloch factors of one (channel, k-point) of the window, on the k-point's device.

    ``weight`` is the number of points of the full mesh the k-point stands for
    (2 for a time-reversal pair, 1 for a self-partner point), its share in the
    norm of the outer DIIS, as for the host's ``BlochOccupiedBlock``.
    """

    weight: float = 1.0

    def overlap(self, other: OccupiedBlock) -> np.ndarray:
        """``Y_self^H Y_other`` (host), formed on the device of ``self``."""

        xp = array_module(self.vectors)
        if xp is np:
            return np.asarray(np.conj(self.vectors).T @ _host(other.vectors))
        with xp.cuda.Device(int(self.vectors.device.id)):
            second = _on_device(xp, other.vectors, int(self.vectors.device.id))
            return xp.asnumpy(xp.conj(self.vectors).T @ second)

    @classmethod
    def combine(cls, terms):
        """``Phi = sum_k Y_k C_k`` (complex) on the device of the first block."""

        phi = None
        for block, part in terms:
            stored = block.vectors
            xp = array_module(stored)
            if xp is np:
                term = stored @ np.asarray(part)
                phi = term if phi is None else phi + _host(term)
                continue
            with xp.cuda.Device(int(stored.device.id)):
                term = stored @ xp.asarray(np.asarray(part, dtype=np.complex128))
            if phi is None:
                phi = term
            else:
                with xp.cuda.Device(int(phi.device.id)):
                    phi = phi + _on_device(xp, term, int(phi.device.id))
        return phi


def _host(array) -> np.ndarray:
    xp = array_module(array)
    return np.asarray(array if xp is np else xp.asnumpy(array))


def _on_device(cp, array, device: int):
    """``array`` on ``device`` (a host array uploaded, another device's copied)."""

    if array_module(array) is np:
        with cp.cuda.Device(device):
            return cp.asarray(array)
    if int(array.device.id) == int(device):
        return array
    source = cp.ascontiguousarray(array) if not array.flags.c_contiguous else array
    with cp.cuda.Device(device):
        copy = cp.empty(source.shape, dtype=source.dtype)
        copy.data.copy_from_device(source.data, source.nbytes)
    return copy


def _device_complement(cp, phi, empty):
    """``empty`` orthogonalized against the orthonormal columns ``phi`` and among themselves (Loewdin, complex)."""

    rest = empty - phi @ (cp.conj(phi).T @ empty)
    gram = cp.conj(rest).T @ rest
    values, vectors = cp.linalg.eigh(0.5 * (gram + cp.conj(gram).T))
    values = cp.asnumpy(values)
    keep = values > 1.0e-8 * max(float(values.max(initial=0.0)), 1.0e-300)
    index = cp.asarray(np.flatnonzero(keep))
    return rest @ (vectors[:, index] / cp.asarray(np.sqrt(values[keep]))[None, :])


@dataclass(frozen=True)
class _Owners:
    """The device of every k-point task of the pool (``spin * n_k + k``)."""

    devices: tuple

    def of(self, spin: int, k: int, count: int) -> int:
        return int(self.devices[(spin * count + k) % len(self.devices)])


class DeviceBlochExchange(BlochExchange):
    """The engine of :func:`parsec_python.SCF.hybrid.run_hybrid_scf` for k-points on the devices (see the module).

    ``devices`` are the CUDA devices of the k-point pool (all of them solve
    pair densities; worker ``i`` owns the tasks ``i, i + n, ...`` on device
    ``devices[i]``).
    """

    #: The k-point backend of the SCF loops: the run's (the device workers).
    kpoint_backend = None

    def __init__(self, system, definition, k_points, k_weights, *, devices, potentials=None) -> None:
        super().__init__(system, definition, k_points, k_weights, potentials)
        self.devices = tuple(int(device) for device in devices)
        if not self.devices:
            raise ValueError("the device exchange needs at least one device")
        box = self.potentials.box
        describe = getattr(self.potentials.kernel, "describe", None)
        self.solvers = DevicePeriodicPairSolvers(
            self.potentials.kernel,
            DevicePeriodicBox(box.shape, self.potentials.fft_shape, getattr(box, "flat", None)),
            self.batch_size,
            description=str(describe()) if callable(describe) else "",
        )
        self.volume_element = float(box.volume_element)
        tasks = self.k_points.shape[0] * (2 if self.spin_polarized else 1)
        workers = max(1, min(len(self.devices), tasks))
        # Worker i of the pool runs on devices[i] and owns the tasks i::workers (Parallel/pool.py).
        self.owners = _Owners(tuple(self.devices[index % workers] for index in range(workers)))
        self.description = (
            f"{self.description}; W = V_x U by cuFFT on {len(self.devices)} GPU(s), "
            "ACE of every k-point on the GPU of its eigensolver"
        )
        self._channels: dict[int, tuple] = {}

    # -- the SCF loops ------------------------------------------------------

    def _keep_vectors(self, solver, task: int):
        """The orbitals of a task where its worker keeps them; its Ritz state then stays there too."""

        keep = getattr(solver, "keep_states_on_devices", None)
        if callable(keep):
            keep()
        fetch = getattr(solver, "device_vectors", None)
        if not callable(fetch):
            return solver.vectors(task)
        return fetch(task)

    def run_scf(self, system, *, callback=None, warm_start=None, exchange_term=None,
                convergence_criterion=None, iteration_state_callback=None, mixer_factory=None, **hooks):
        """One SCF loop of the Fock loop on the device k-point pool (see :meth:`BlochExchange.run_scf`)."""

        from parsec_python.SCF.kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
        from parsec_python.SCF.single_point import refuse_exchange_hooks

        refuse_exchange_hooks(**hooks)
        keywords = dict(
            callback=callback,
            keep_wavefunctions=True,
            keep_states=True,
            keep_vectors=self._keep_vectors,
            backend=self.kpoint_backend,
            warm_start=warm_start,
            convergence_criterion=convergence_criterion,
            exchange_term=exchange_term,
            mixer_factory=mixer_factory,
            projector_cache=self.projector_cache,
        )
        if self.spin_polarized and warm_start is None:
            # The start of the Gamma loop (run_spin_scf), so that every route of a spin-polarized hybrid begins
            # in the same state: the per-species spin densities of initchrg.f90.
            from parsec_python.SCF.pbc import with_initial_spin_densities

            keywords["initial_spin_densities"] = with_initial_spin_densities(system).initial_spin_densities
        if iteration_state_callback is not None:
            keywords["iteration_state_callback"] = iteration_state_callback
        driver = run_scf_kpoints_spin_polarized if self.spin_polarized else run_scf_kpoints
        return driver(system, self.k_points, self.k_weights, **keywords)

    def finish(self, result):
        """The result of the loop with host copies of the orbitals (what the forces and the output read)."""

        vectors = getattr(result, "k_wavefunctions", None)
        if vectors is None:
            return result
        if self.spin_polarized:
            host = tuple(tuple(_host(block) for block in channel) for channel in vectors)
        else:
            host = tuple(_host(block) for block in vectors)
        return replace(result, k_wavefunctions=host)

    # -- the orbitals ----------------------------------------------------------

    def _set(self, orbitals, occupations, spin_weight: float, spin: int) -> BlochOrbitalSet:
        """A Bloch set of device blocks, each on the device of its k-point's worker."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        count = self.k_points.shape[0]
        blocks = []
        for k, block in enumerate(orbitals):
            device = self.owners.of(spin, k, count)
            if array_module(block) is np or int(block.device.id) != device:
                block = _on_device(cp, np.ascontiguousarray(block) if array_module(block) is np else block, device)
            blocks.append(block)
        values = tuple(np.asarray(item, dtype=np.float64) for item in occupations)
        partners = time_reversal_partners(self.k_points, self.box.reciprocal)
        return BlochOrbitalSet(
            self.k_points,
            self.k_weights,
            tuple(blocks),
            values,
            float(spin_weight),
            partners,
            int(np.sum(np.where(partners, 2, 1))),
        )

    def orbital_sets(self, result) -> tuple[BlochOrbitalSet, ...]:
        vectors = getattr(result, "k_wavefunctions", None)
        if vectors is None:
            raise ValueError("the k-point result holds no orbitals (keep_wavefunctions)")
        if hasattr(result, "occupations_up"):
            return tuple(
                self._set(blocks, _split(pooled, blocks), 1.0, spin)
                for spin, (blocks, pooled) in enumerate(zip(vectors, (result.occupations_up, result.occupations_down)))
            )
        return (self._set(vectors, _split(result.occupations, vectors), 2.0, 0),)

    def _channel(self, orbital_set, spin: int = 0):
        """The device channel of a set, made once (an application and its completion share it).

        A set whose blocks are on the host or on another device than their
        k-point's worker (the orbitals of the structure before a move, carried
        on the host) is first placed there, so that ``xi`` is built where the
        worker applies it.
        """

        found = self._channels.get(id(orbital_set))
        if found is not None and found[0] is orbital_set:
            return found[1]
        count = self.k_points.shape[0]
        placed = orbital_set
        if any(
            array_module(block) is np or int(block.device.id) != self.owners.of(spin, k, count)
            for k, block in enumerate(orbital_set.orbitals)
        ):
            placed = self._set(orbital_set.orbitals, orbital_set.occupations, orbital_set.spin_weight, spin)
        channel = device_bloch_channel(placed, self.volume_element)
        self._channels[id(orbital_set)] = (orbital_set, channel)
        return channel

    # -- the exchange --------------------------------------------------------

    def exchange(self, sets, *, columns: str = "occupied", known=None):
        return [
            apply_device_bloch_exchange(
                self._channel(orbital_set, index),
                self.solvers,
                self.devices,
                columns=columns,
                known=None if known is None else known[index],
            )
            for index, orbital_set in enumerate(sets)
        ]

    def fock_operator(self, sets, applications, fraction: float) -> PeriodicFockOperator:
        channels = []
        for application in applications:
            if not any(len(columns) for columns in application.occupied):
                channels.append(None)
                continue
            channels.append(BlochChannelExchange(tuple(compress_device_bloch_exchange(application, fraction=fraction))))
        return PeriodicFockOperator(
            channels=tuple(channels),
            fraction=float(fraction),
            reference_energy=float(sum(application.energy for application in applications)),
            spin_weight=float(sets[0].spin_weight),
        )

    def cross_energy(self, operator: PeriodicFockOperator, sets) -> float:
        """``E_x(P, P0) = (s/2) sum_t w_t sum_n f <u|V_ACE|u>`` (Ry, unscaled), on the devices of the orbitals."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        total = 0.0
        for channel, orbital_set in zip(operator.channels, sets):
            if channel is None:
                continue
            for t, (ace, block, values) in enumerate(
                zip(channel.operators, orbital_set.orbitals, orbital_set.occupations)
            ):
                if ace is None:
                    continue
                with cp.cuda.Device(int(ace.projectors.device.id)):
                    vectors = _on_device(cp, block, int(ace.projectors.device.id))
                    value = device_cross_energy(
                        cp, ace, vectors, values, float(orbital_set.weights[t]), float(orbital_set.spin_weight)
                    )
                    total += float(cp.asnumpy(value)[0])
        return float(total)

    def release(self, operator) -> None:
        """Free the ``xi`` of every k-point (the workers that held them are gone)."""

        for channel in getattr(operator, "channels", ()):
            if channel is None:
                continue
            for ace in channel.operators:
                if ace is not None:
                    ace.release()

    def drop_orbitals(self, sets, result=None) -> None:
        """Forget the device channels of ``sets`` (views of the Ritz subspaces, which the next loop overwrites)."""

        for orbital_set in sets:
            self._channels.pop(id(orbital_set), None)

    # -- the outer DIIS ------------------------------------------------------

    def _window_fits(self, sets) -> bool:
        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        needed: dict[int, int] = {}
        for orbital_set in sets:
            for block, values in zip(orbital_set.orbitals, orbital_set.occupations):
                device = int(block.device.id)
                occupied = int(np.count_nonzero(np.asarray(values) > OCCUPATION_THRESHOLD))
                # the copy, and the build of an extrapolated operator (three sets of the states)
                needed[device] = needed.get(device, 0) + 16 * int(block.shape[0]) * (occupied + 3 * int(block.shape[1]))
        for device, count in needed.items():
            with cp.cuda.Device(device):
                free, total = cp.cuda.runtime.memGetInfo()
            if count + WINDOW_MARGIN * float(total) > float(free):
                return False
        return True

    def occupied_blocks(self, sets) -> list | None:
        """Copies of the occupied Bloch factors of each set, one block per k-point, on its device."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        if not self._window_fits(sets):
            return None
        blocks = []
        for orbital_set in sets:
            channel = {}
            for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
                occupied = np.flatnonzero(np.asarray(values) > OCCUPATION_THRESHOLD)
                if not occupied.size:
                    continue
                with cp.cuda.Device(int(block.device.id)):
                    copy = cp.array(block[:, cp.asarray(occupied)], dtype=cp.complex128, copy=True)
                channel[t] = DeviceBlochOccupiedBlock(
                    copy, np.array(np.asarray(values)[occupied]), 2.0 if orbital_set.partners[t] else 1.0
                )
            blocks.append(channel)
        return blocks

    def natural_sets(self, sets, natural):
        """The sets of the natural orbitals ``natural`` (per set ``{k: (Phi, n)}``) and the empty states, on the devices."""

        from ..backends.cupy import require_cupy

        cp, _ = require_cupy()
        result = []
        for spin, (orbital_set, blocks) in enumerate(zip(sets, natural)):
            orbitals, occupations = [], []
            for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
                device = int(block.device.id)
                with cp.cuda.Device(device):
                    empty_index = np.flatnonzero(np.asarray(values) <= OCCUPATION_THRESHOLD)
                    empty = block[:, cp.asarray(empty_index)]
                    phi, numbers = blocks.get(t, (None, np.zeros(0)))
                    if phi is None:
                        columns, weights = cp.ascontiguousarray(empty), np.zeros(empty_index.size)
                    else:
                        phi = _on_device(cp, phi, device).astype(cp.complex128, copy=False)
                        rest = _device_complement(cp, phi, empty) if empty_index.size else empty
                        columns = cp.ascontiguousarray(cp.concatenate([phi, rest], axis=1))
                        weights = np.concatenate(
                            [np.asarray(numbers, dtype=np.float64), np.zeros(int(rest.shape[1]))]
                        )
                orbitals.append(columns)
                occupations.append(weights)
            result.append(self._set(orbitals, occupations, orbital_set.spin_weight, spin))
        return tuple(result)


def device_bloch_engine(system, definition, k_points, k_weights):
    """The device engine of a periodic hybrid with k-points, or ``None`` where the k-points run on the CPU.

    Installed into :mod:`parsec_python.SCF.hybrid_periodic` by
    :func:`install`: the k-point pool of this run resolves to CUDA devices
    (``--device``, ``PARSEC_KPOINT_BACKEND``, ``PARSEC_CUPY_DEVICES``), the
    device eigensolver is selected (``PARSEC_KPOINT_SOLVER``) and the run is
    one process (no ``--mpi``).
    """

    from ..Parallel import mpi
    from .kpoint_parallel import _device_solver_requested, resolve_backend

    if mpi.context() is not None or mpi.requested():
        return None
    from parsec_python.SCF.kpoint_hubbard import kpoint_backend

    plan = resolve_backend(kpoint_backend(system))
    if plan is None or plan.name != "cupy" or not _device_solver_requested():
        return None
    return DeviceBlochExchange(system, definition, k_points, k_weights, devices=plan.device_ids)


def install() -> None:
    """Make :func:`device_bloch_engine` the engine factory of the periodic Fock loop with k-points."""

    from parsec_python.SCF import hybrid_periodic

    setter = getattr(hybrid_periodic, "set_device_engine", None)
    if callable(setter):
        setter(device_bloch_engine)


__all__ = [
    "DeviceBlochExchange",
    "DeviceBlochOccupiedBlock",
    "device_bloch_engine",
    "install",
]
