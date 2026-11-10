"""Self-consistent hybrid functionals of a periodic cell with k-points: the exchange engine of the Fock loop.

The Fock loop is :func:`parsec_python.SCF.hybrid.run_hybrid_scf`, unchanged:
start from a semilocal SCF, ``V_ACE`` of the orbitals of the loop before,
inner SCF loops with the operator fixed, ``dexx = a_x [2 E_x(P, P0) -
E_x(P0) - E_x(P)]``, the stop rule, the adaptive inner criterion and the
outer DIIS of the cluster loop.  What a periodic cell with k-points changes
lives in :class:`BlochExchange`, the engine the loop asks for its pieces:

* the orbitals are the Bloch factors ``u_nk`` of the k-points solved, the
  representatives of the full Monkhorst-Pack mesh under time reversal
  (:class:`parsec_python.V_xc.bloch_exchange.BlochOrbitalSet`, one per spin
  channel or one unpolarized set);
* ``V_x u`` sums over the full mesh, ``-k`` taking ``conj(u_k)``, with the pair
  potentials of the cell's exchange kernel by FFT
  (:func:`parsec_python.V_xc.bloch_exchange.apply_bloch_exchange`), and the
  operator is the ACE of every (channel, k-point)
  (:class:`PeriodicFockOperator`);
* the SCF loops are the k-point drivers
  :func:`parsec_python.SCF.kpoints.run_scf_kpoints` and
  :func:`~parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized` with the
  operator in every Bloch Hamiltonian, ``E_x(P, P0)`` accumulated where each
  k-point's orbitals are, and the warm start of a Fock loop (the Ritz states
  of every k-point, the carried mixer, the relative criterion, the level
  passes);
* the energies are BZ averages: ``E_x = (s/2) sum_k w_k sum_n f_nk <u|V_x u>``
  and ``E_x(P, P0) = (s/2) sum_k w_k sum_n f_nk <u|V_ACE|u>`` (Ry);
* the outer DIIS keeps one block per (channel, k-point), its overlaps
  Hermitian (:class:`BlochOccupiedBlock`).

At the Gamma point alone (no k-point input) the orbitals are real and the
cluster engine (:class:`parsec_python.SCF.hybrid.HostExchange`) runs with the
FFT pair solver of :func:`periodic_pair_potentials`
(:func:`parsec_python.SCF.hybrid.default_pair_solver`).

Units Ry and bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..V_xc.bloch_exchange import (
    BlochCompressedExchange,
    BlochOrbitalSet,
    GridBox,
    PeriodicPairPotentials,
    apply_bloch_exchange,
    bloch_band_energy,
    bloch_orbital_set,
    compress_bloch_exchange,
    periodic_pair_solver,
)
from ..V_xc.exact_exchange import OCCUPATION_THRESHOLD
from .fock_diis import OccupiedBlock


#: Reduced coordinates closer than this are one value (and steps of a mesh this close are equal).
MESH_TOLERANCE = 1.0e-6


def uniform_mesh_counts(fractions) -> tuple[int, int, int]:
    """The counts ``(N_1, N_2, N_3)`` of the uniform mesh that the points ``fractions`` (reduced coordinates, rows) are.

    A Monkhorst-Pack mesh, shifted or not: along every axis the reduced
    coordinates (modulo 1) are ``s_i + j/N_i``, ``j = 0 .. N_i - 1`` (equal
    steps ``1/N_i``, one shift), and the points are every combination of
    them, each once (the full Cartesian product).  Any other set is refused
    with the reason: the exchange sum and the kernel's treatment of the
    Coulomb singularity (``k_mesh``) assume that mesh.
    """

    points = np.atleast_2d(np.asarray(fractions, dtype=np.float64))
    reduced = points - np.floor(points)
    reduced[np.abs(reduced - 1.0) < MESH_TOLERANCE] = 0.0
    counts, indices = [], []
    for axis in range(3):
        values = np.sort(reduced[:, axis])
        distinct = [values[0]]
        for value in values[1:]:
            if value - distinct[-1] > MESH_TOLERANCE:
                distinct.append(value)
        if len(distinct) > 1 and 1.0 - distinct[-1] + distinct[0] < MESH_TOLERANCE:
            distinct.pop()  # 0 and 1 - epsilon are one value modulo 1
        count = len(distinct)
        # s + j/N: every distinct value at an integer number of steps 1/N from the first
        steps = (reduced[:, axis] - distinct[0]) * count
        steps = steps - np.floor(steps / count + MESH_TOLERANCE) * count
        rounded = np.round(steps)
        if np.any(np.abs(steps - rounded) > MESH_TOLERANCE * count) or len(np.unique(rounded % count)) != count:
            raise NotImplementedError(
                "the exchange of a periodic cell needs the k-points of a uniform Monkhorst-Pack mesh: the reduced "
                f"coordinates along b{axis + 1} ({', '.join(f'{v:.6g}' for v in distinct)}) are not equally spaced "
                "by 1/N (Monkhorst_Pack_Grid, or a Kpoint_List that writes out every point of one)"
            )
        counts.append(count)
        indices.append(rounded.astype(np.int64) % count)
    combined = np.ravel_multi_index(tuple(indices), tuple(counts))
    if points.shape[0] != int(np.prod(counts)) or np.unique(combined).size != points.shape[0]:
        raise NotImplementedError(
            "the exchange of a periodic cell needs the k-points of a uniform Monkhorst-Pack mesh: "
            f"{points.shape[0]} points are not every point of the {'x'.join(str(n) for n in counts)} mesh "
            "their coordinates span, each once (Monkhorst_Pack_Grid, or a Kpoint_List that writes out every point)"
        )
    return tuple(counts)


def mesh_counts(problem) -> tuple[int, int, int]:
    """The counts of the full Monkhorst-Pack mesh of ``problem`` (``(1, 1, 1)`` without one).

    ``Monkhorst_Pack_Grid``, or for a ``Kpoint_List`` of equal weights the
    mesh it writes out (:func:`uniform_mesh_counts`); any other list is
    refused.
    """

    dimensions = getattr(problem, "monkhorst_pack_dimensions", None)
    if dimensions is not None:
        return tuple(int(value) for value in dimensions)
    listed = getattr(problem, "kpoints", None)
    if listed is None:
        return (1, 1, 1)
    from ..Grid.cell import reciprocal_lattice

    points = np.atleast_2d(np.asarray(listed, dtype=np.float64))
    weights = np.asarray(problem.kpoint_weights, dtype=np.float64)
    if not np.allclose(weights, weights[0], rtol=1.0e-12, atol=0.0):
        raise NotImplementedError(
            "a hybrid functional of a periodic cell needs a uniform k-point mesh: a Kpoint_List with unequal "
            "weights is not one"
        )
    return uniform_mesh_counts(points @ np.linalg.inv(reciprocal_lattice(problem.periodic_cell.lattice_vectors)))


def full_mesh_counts(k_points, partners, reciprocal) -> tuple[int, int, int]:
    """The counts ``(N_1, N_2, N_3)`` of the uniform mesh whose time-reversal representatives are ``k_points``.

    The full mesh is ``k_points`` and the partners ``-k`` (``partners``),
    checked by :func:`uniform_mesh_counts` along the reciprocal vectors of
    ``reciprocal`` (rows).
    """

    points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
    full = np.vstack([points, -points[np.asarray(partners, dtype=bool)]])
    return uniform_mesh_counts(full @ np.linalg.inv(np.asarray(reciprocal, dtype=np.float64)))


def periodic_pair_potentials(system, definition, k_mesh=None) -> PeriodicPairPotentials:
    """The FFT pair potentials of the exchange kernel of ``definition`` on the grid of ``system``.

    ``k_mesh`` defaults to the mesh of the input (``(1, 1, 1)`` at the Gamma
    point), which the kernel's treatment of the Coulomb singularity reads.
    """

    settings = getattr(system.input.scf, "exact_exchange", None)
    return periodic_pair_solver(
        system.grid,
        definition.functional,
        definition.fraction,
        k_mesh=mesh_counts(system.input) if k_mesh is None else k_mesh,
        divergence=getattr(settings, "divergence", None),
        auxiliary_exponent=getattr(settings, "auxiliary_exponent", None),
        slab_coulomb=getattr(settings, "slab_coulomb", "padded"),
    )


# ---------------------------------------------------------------------------
# The operator of one inner loop.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlochChannelExchange:
    """The compressed exchange operators of one channel, one per k-point solved."""

    operators: tuple

    @property
    def rank(self) -> int:
        return max((operator.rank for operator in self.operators if operator is not None), default=0)

    @property
    def asymmetry(self) -> float:
        return max((operator.asymmetry for operator in self.operators if operator is not None), default=0.0)


@dataclass(frozen=True)
class PeriodicFockOperator:
    """The exact-exchange term of one inner SCF loop of a periodic cell with k-points.

    ``channels`` holds a :class:`BlochChannelExchange` per spin channel (one
    for an unpolarized loop), ``None`` for a channel without electrons;
    ``spin_weight`` is that of the channels (2 or 1).  ``reference_energy``
    is ``E_x(P0)`` of the orbitals the operators were built from (Ry,
    unscaled, all channels).
    """

    channels: tuple
    fraction: float
    reference_energy: float
    spin_weight: float = 2.0

    def operator(self, spin: int, k: int) -> BlochCompressedExchange | None:
        """``a_x V_ACE`` of channel ``spin`` at k-point ``k`` for its Bloch Hamiltonian."""

        channel = self.channels[spin]
        return None if channel is None else channel.operators[k]

    def task_cross(self, spin: int, k: int, vectors, occupations, weight: float, xp=np):
        """``(s/2) w_k sum_n f_nk <u_nk|V_ACE|u_nk>`` of one task (unscaled), as an ``xp`` array of one value.

        Called by the k-point job where the task's orbitals are (``xp`` the
        array module of ``vectors``); a host operator brings device orbitals
        to the host.
        """

        operator = self.operator(spin, k)
        values = np.asarray(occupations, dtype=np.float64)
        occupied = np.flatnonzero(values > OCCUPATION_THRESHOLD)
        if operator is None or not occupied.size:
            return xp.zeros(1)
        if xp is not np and isinstance(operator.projectors, np.ndarray):
            vectors = xp.asnumpy(vectors)
        expectation = operator.expectation_values(vectors[:, occupied], scaled=False)
        if not isinstance(expectation, np.ndarray):
            expectation = xp.asnumpy(expectation)
        value = 0.5 * self.spin_weight * float(weight) * float(np.dot(values[occupied], expectation))
        return xp.asarray([value])

    def energies_from_cross(self, energies, cross: float):
        """The energies of an inner iteration from its ``E_x(P, P0)`` (:func:`parsec_python.SCF.hybrid.linearized_energies`)."""

        from .hybrid import linearized_energies

        return linearized_energies(energies, float(cross), self.fraction, self.reference_energy)


# ---------------------------------------------------------------------------
# The orbitals of a k-point result.
# ---------------------------------------------------------------------------


def _split(pooled, blocks) -> list[np.ndarray]:
    values = np.asarray(pooled, dtype=np.float64)
    parts, offset = [], 0
    for block in blocks:
        count = int(np.shape(block)[1])
        parts.append(values[offset : offset + count])
        offset += count
    if offset != values.size:
        raise ValueError("the occupations do not match the orbitals of the k-points")
    return parts


def bloch_orbital_sets_of(result, box: GridBox | None = None) -> tuple[BlochOrbitalSet, ...]:
    """The Bloch orbital sets of a k-point result: one unpolarized set or the two channels.

    ``result`` is one of :func:`parsec_python.SCF.kpoints.run_scf_kpoints`
    (spin weight 2) or :func:`~parsec_python.SCF.kpoints.run_scf_kpoints_spin_polarized`
    (two channels, spin weight 1), with its orbitals kept.
    """

    if box is None:
        box = GridBox(result.grid)
    vectors = getattr(result, "k_wavefunctions", None)
    if vectors is None:
        raise ValueError("the k-point result holds no orbitals (keep_wavefunctions)")
    k_points = np.asarray(result.k_points, dtype=np.float64)
    weights = np.asarray(result.k_weights, dtype=np.float64)
    if hasattr(result, "occupations_up"):
        return tuple(
            bloch_orbital_set(k_points, weights, blocks, _split(pooled, blocks), 1.0, box)
            for blocks, pooled in zip(vectors, (result.occupations_up, result.occupations_down))
        )
    return (bloch_orbital_set(k_points, weights, vectors, _split(result.occupations, vectors), 2.0, box),)


# ---------------------------------------------------------------------------
# The outer DIIS of Bloch orbitals.
# ---------------------------------------------------------------------------


@dataclass
class BlochOccupiedBlock(OccupiedBlock):
    """The occupied Bloch factors of one (channel, k-point) of the window: Hermitian overlaps.

    ``weight`` is the number of points of the full mesh the k-point stands for
    (2 for a time-reversal pair, 1 for a self-partner point): the share of the
    k-point in the norm of the outer DIIS (``fock_diis._trace``).
    """

    weight: float = 1.0

    def overlap(self, other: OccupiedBlock) -> np.ndarray:
        """``Y_self^H Y_other`` on the host."""

        return np.asarray(np.conj(self.vectors).T @ np.asarray(other.vectors))

    @classmethod
    def combine(cls, terms) -> np.ndarray:
        """``Phi = sum_k Y_k C_k`` (complex, on the host)."""

        phi = None
        for block, part in terms:
            term = np.asarray(block.vectors) @ np.asarray(part)
            phi = term if phi is None else phi + term
        return phi


def _complement(phi: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """``empty`` orthogonalized against the orthonormal columns ``phi`` and among themselves (Loewdin, complex)."""

    rest = empty - phi @ (np.conj(phi).T @ empty)
    gram = np.conj(rest).T @ rest
    values, vectors = np.linalg.eigh(0.5 * (gram + np.conj(gram).T))
    keep = values > 1.0e-8 * max(float(values.max(initial=0.0)), 1.0e-300)
    return rest @ (vectors[:, keep] / np.sqrt(values[keep]))


# ---------------------------------------------------------------------------
# The engine.
# ---------------------------------------------------------------------------


class BlochExchange:
    """The pieces of :func:`parsec_python.SCF.hybrid.run_hybrid_scf` for a periodic cell with k-points (host).

    ``k_points`` and ``k_weights`` are the k-points solved: the full
    Monkhorst-Pack mesh reduced by time reversal (weights ``2/N``, or ``1/N``
    where ``-k = k``).  ``potentials`` replaces the pair potentials of the
    kernel of the run (:func:`periodic_pair_potentials`).  The SCF loops of
    the Fock loop are the k-point drivers (:meth:`run_scf`); they keep the
    orbitals and Ritz states of every k-point.
    """

    takes_hooks = False
    #: The k-point backend of the SCF loops (``make_kpoint_solver``): the host, where the operator is.
    kpoint_backend = "cpu"

    def __init__(self, system, definition, k_points, k_weights, potentials: PeriodicPairPotentials | None = None) -> None:
        self.k_points = np.atleast_2d(np.asarray(k_points, dtype=np.float64))
        self.k_weights = np.asarray(k_weights, dtype=np.float64)
        self.box = GridBox(system.grid)
        # A uniform mesh reduced by time reversal only, or refused here (before any SCF).
        probe = bloch_orbital_set(
            self.k_points,
            self.k_weights,
            [np.zeros((self.box.size, 1), dtype=np.complex128)] * self.k_points.shape[0],
            [np.zeros(1)] * self.k_points.shape[0],
            2.0,
            self.box,
        )
        # The counts of the full mesh the exchange sums over (the kernel's k_mesh), from the k-points themselves.
        self.mesh = full_mesh_counts(self.k_points, probe.partners, self.box.reciprocal)
        self.potentials = (
            periodic_pair_potentials(system, definition, k_mesh=self.mesh) if potentials is None else potentials
        )
        self.description = (
            f"{self.potentials.description}; {self.k_points.shape[0]} k-points of the time-reversal-reduced mesh "
            f"{'x'.join(str(n) for n in self.mesh)}"
        )
        self.batch_size = int(system.input.scf.exact_exchange.batch_size)
        self.spin_polarized = bool(system.input.scf.spin_polarized)
        # The Bloch-phase projectors of the k-points of this structure, kept for every SCF loop.
        self.projector_cache: dict = {}

    # -- the SCF loops ------------------------------------------------------

    def run_scf(
        self,
        system,
        *,
        callback: Callable | None = None,
        warm_start=None,
        exchange_term=None,
        convergence_criterion: float | None = None,
        iteration_state_callback: Callable | None = None,
        mixer_factory: Callable | None = None,
        **hooks,
    ):
        """One SCF loop of the Fock loop (the semilocal start, or an inner loop with ``exchange_term``)."""

        from .kpoints import run_scf_kpoints, run_scf_kpoints_spin_polarized
        from .single_point import refuse_exchange_hooks

        refuse_exchange_hooks(**hooks)
        keywords = dict(
            callback=callback,
            keep_wavefunctions=True,
            keep_states=True,
            # The host operator: the eigenproblems on the host as well (the device path brings its own engine).
            backend=self.kpoint_backend,
            warm_start=warm_start,
            convergence_criterion=convergence_criterion,
            exchange_term=exchange_term,
            mixer_factory=mixer_factory,
            # the projectors of the k-points, built once for every loop of the structure
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

    def start_system(self, system, definition):
        from .hybrid import semilocal_start_system

        return semilocal_start_system(system, definition)

    def transition_warm_start(self, system, start):
        """The first inner loop after the semilocal start: ``V_ion + V_H[rho] + V_xc^semilocal[rho]``, the
        Ritz states of every k-point kept (Quantum ESPRESSO after its first SCF)."""

        from .single_point import SCFWarmStart, state_for_new_potential

        ionic = np.asarray(system.ionic_potential, dtype=np.float64)
        if self.spin_polarized:
            hartree = np.asarray(start.scf_hartree_potential, dtype=np.float64)
            xc = system.evaluate_xc_spin_polarized(
                np.asarray(start.density_up, dtype=np.float64), np.asarray(start.density_down, dtype=np.float64)
            )
            potential = (ionic + hartree + xc.potential_up, ionic + hartree + xc.potential_down)
        else:
            hartree = np.asarray(start.hartree_potential, dtype=np.float64)
            xc = system.evaluate_xc(np.asarray(start.density, dtype=np.float64))
            potential = ionic + hartree + xc.potential
        return SCFWarmStart(
            input_potential=potential,
            hartree_potential=hartree,
            eigensolver_state=state_for_new_potential(start.eigensolver_state),
        )

    def next_warm_start(self, system, result):
        """The next inner loop: the mixed potential, the Hartree potential and the Ritz states of ``result``."""

        from .single_point import SCFWarmStart, state_for_new_potential

        if self.spin_polarized:
            return SCFWarmStart(
                input_potential=tuple(np.asarray(value) for value in result.next_potentials),
                hartree_potential=np.asarray(result.scf_hartree_potential),
                eigensolver_state=state_for_new_potential(result.eigensolver_state),
            )
        return SCFWarmStart(
            input_potential=np.asarray(result.next_effective_potential),
            hartree_potential=np.asarray(result.hartree_potential),
            eigensolver_state=state_for_new_potential(result.eigensolver_state),
        )

    def start_total_energy(self, system, start, exchange_energy: float, fraction: float) -> float:
        """``E_start - E_xc^start[rho] + E_xc^semilocal[rho] + a_x E_x(P)`` (Ry)."""

        if self.spin_polarized:
            semilocal = system.evaluate_xc_spin_polarized(
                np.asarray(start.density_up, dtype=np.float64), np.asarray(start.density_down, dtype=np.float64)
            ).total_energy
        else:
            semilocal = system.evaluate_xc(np.asarray(start.density, dtype=np.float64)).total_energy
        energies = start.energies
        return float(energies.total - energies.exchange_correlation + semilocal + fraction * exchange_energy)

    # -- the exchange --------------------------------------------------------

    def orbital_sets(self, result) -> tuple[BlochOrbitalSet, ...]:
        return bloch_orbital_sets_of(result, self.box)

    def exchange(self, sets, *, columns: str = "occupied", known=None):
        return [
            apply_bloch_exchange(
                orbital_set,
                self.potentials,
                columns=columns,
                known=None if known is None else known[index],
                batch_size=self.batch_size,
            )
            for index, orbital_set in enumerate(sets)
        ]

    def fock_operator(self, sets, applications, fraction: float) -> PeriodicFockOperator:
        channels = []
        for orbital_set, application in zip(sets, applications):
            if not application.has_electrons:
                channels.append(None)
                continue
            channels.append(
                BlochChannelExchange(
                    tuple(
                        compress_bloch_exchange(block, applied, fraction=fraction)
                        for block, applied in zip(orbital_set.orbitals, application.applied)
                    )
                )
            )
        return PeriodicFockOperator(
            channels=tuple(channels),
            fraction=float(fraction),
            reference_energy=float(sum(application.energy for application in applications)),
            spin_weight=float(sets[0].spin_weight),
        )

    def cross_energy(self, operator: PeriodicFockOperator, sets) -> float:
        """``E_x(P, P0)`` of the orbital sets ``sets`` with ``operator`` (Ry, unscaled)."""

        return float(
            sum(
                0.0 if channel is None else bloch_band_energy(channel.operators, orbital_set)
                for channel, orbital_set in zip(operator.channels, sets)
            )
        )

    def release(self, operator) -> None:
        """Nothing to free: the host operator goes with its last reference."""

    def drop_orbitals(self, sets, result=None) -> None:
        """Nothing to drop: host orbitals pin no eigensolver memory."""

    def occupied_blocks(self, sets) -> list:
        """Copies of the occupied Bloch factors of each set, one block per k-point (``Exx_Outer_Mixing: diis``)."""

        blocks = []
        for orbital_set in sets:
            channel = {}
            for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
                occupied = np.flatnonzero(values > OCCUPATION_THRESHOLD)
                if occupied.size:
                    channel[t] = BlochOccupiedBlock(
                        np.array(block[:, occupied]),
                        np.array(values[occupied]),
                        2.0 if orbital_set.partners[t] else 1.0,
                    )
            blocks.append(channel)
        return blocks

    def natural_sets(self, sets, natural):
        """The orbital sets of the natural orbitals ``natural`` (per set ``{k: (Phi, n)}``) and the empty states."""

        result = []
        for orbital_set, blocks in zip(sets, natural):
            orbitals, occupations = [], []
            for t, (block, values) in enumerate(zip(orbital_set.orbitals, orbital_set.occupations)):
                empty = block[:, values <= OCCUPATION_THRESHOLD]
                phi, numbers = blocks.get(t, (None, np.zeros(0)))
                if phi is None:
                    columns, weights = empty, np.zeros(empty.shape[1])
                else:
                    phi = np.asarray(phi, dtype=np.complex128)
                    rest = _complement(phi, empty) if empty.shape[1] else empty
                    columns = np.hstack([phi, rest])
                    weights = np.concatenate([np.asarray(numbers, dtype=np.float64), np.zeros(rest.shape[1])])
                orbitals.append(np.ascontiguousarray(columns))
                occupations.append(weights)
            result.append(
                bloch_orbital_set(
                    orbital_set.k_points, orbital_set.weights, orbitals, occupations, orbital_set.spin_weight, self.box
                )
            )
        return tuple(result)


def bloch_warm_start(previous_system, previous_result, system, unpolarized=None):
    """The warm start of the Fock loop of a moved structure from the hybrid k-point result before.

    Spin-polarized: the two mixed potentials with the ionic potential of the
    new structure in place of the old (``parsec.F90`` after ``domove``), the
    Hartree potential, the Ritz states of every task and the Bloch orbitals
    that build the first exchange operator.  Unpolarized: ``unpolarized``
    (:func:`parsec_python.SCF.warm_start_for_moved_atoms` with the
    extrapolation of the run, which carries the orbitals as well).
    """

    from .single_point import SCFWarmStart, state_for_new_potential, warm_start_for_moved_atoms

    if not hasattr(previous_result, "occupations_up"):
        build = warm_start_for_moved_atoms if unpolarized is None else unpolarized
        return build(previous_system, previous_result, system)
    shift = np.asarray(system.ionic_potential, dtype=np.float64) - np.asarray(previous_system.ionic_potential)
    return SCFWarmStart(
        input_potential=tuple(np.asarray(value, dtype=np.float64) + shift for value in previous_result.next_potentials),
        hartree_potential=np.asarray(previous_result.scf_hartree_potential, dtype=np.float64),
        eigensolver_state=state_for_new_potential(previous_result.eigensolver_state),
        exchange_orbitals=bloch_orbital_sets_of(previous_result, GridBox(previous_system.grid)),
    )


def bloch_nonlocal_forces(system, result) -> np.ndarray:
    """The nonlocal force of an unpolarized k-point result from its kept Bloch factors (Ry/bohr, ``(atoms, 3)``).

    ``4 w_k sum_n f_nk sum_q s_q Re[conj(<beta_q|u_nk>) <grad beta_q|u_nk>]`` summed over the k-points solved
    (the weights of the time-reversal representatives count their partners), as the step of
    :func:`parsec_python.SCF.kpoints.run_scf_kpoints` with ``compute_forces`` accumulates it.
    """

    from ..Forces.nonlocal_force import bloch_projector_gradients, nonlocal_forces
    from ..V_ion import build_nonlocal_projectors

    atoms = system.cell_atoms
    lattice = system.input.periodic_cell.lattice_vectors
    occupations = _split(result.occupations, result.k_wavefunctions)
    total = np.zeros((len(atoms), 3))
    for k_point, weight, vectors, values in zip(result.k_points, result.k_weights, result.k_wavefunctions, occupations):
        projectors = build_nonlocal_projectors(
            system.grid,
            atoms,
            system.pseudopotentials,
            system.input.pseudopotentials,
            lattice,
            k_point=k_point,
            periodic_dimensions=system.input.periodic_cell.dimensions,
        )
        gradients = bloch_projector_gradients(
            system.grid, atoms, system.pseudopotentials, system.input.pseudopotentials, lattice, k_point
        )
        total += nonlocal_forces(projectors, gradients, np.asarray(vectors), float(weight) * values, len(atoms))
    return total


# The engine factory of the device path (installed by parsec_python.acceleration), or None.
_device_engine = None


def set_device_engine(factory=None) -> None:
    """Install (or, with ``None``, remove) the factory of a device engine of the Fock loop with k-points.

    ``factory(system, definition, k_points, k_weights)`` returns an engine
    with the methods of :class:`BlochExchange` whose orbitals and operators
    live on the devices of the k-point pool, or ``None`` for a run whose
    k-points are solved on the CPU (the host engine then runs).  The
    performance counterpart lives in
    ``parsec_python.acceleration.SCF.hybrid_periodic``; importing
    :mod:`parsec_python.acceleration` installs it.
    """

    global _device_engine
    _device_engine = factory


def run_periodic_hybrid_scf(
    system,
    k_points,
    k_weights,
    *,
    callback: Callable | None = None,
    warm_start=None,
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable | None = None,
    compute_forces: bool = False,
):
    """The Fock loop of a hybrid functional of a periodic cell with k-points (see the module).

    ``k_points``/``k_weights``: the time-reversal-reduced Monkhorst-Pack mesh
    (:func:`parsec_python.Grid.kpoints.kpoint_sampling` of a hybrid input).
    The result is that of the last inner loop (a
    :class:`~parsec_python.models.SinglePointResult` with ``k_points`` or a
    :class:`~parsec_python.models.SpinPolarizedSinglePointResult`) with the
    energies of the hybrid, the history of every loop, the orbitals and
    Ritz states of every k-point and the
    :class:`parsec_python.SCF.hybrid.FockLoopSummary` in ``exact_exchange``.
    A system that holds its exchange on devices (``exchange_device`` with a
    ``bloch_engine`` method) supplies its own engine.  ``compute_forces``
    adds the nonlocal force of an unpolarized result (``nonlocal_forces``,
    :func:`bloch_nonlocal_forces`), what
    :func:`parsec_python.Forces.compute_periodic_forces` of Andrew Grieder's
    routines reads.
    """

    from .hybrid import hybrid_definition, require_supported, run_hybrid_scf

    definition = require_supported(hybrid_definition(system.input.scf))
    if system.input.scf.fixed_occupations is not None:
        raise NotImplementedError(
            "the occupations of occup.in (a negative Fermi_Temp) belong to the Gamma point without k-point input; "
            "the k-point loops of a periodic hybrid find a Fermi level"
        )
    device = getattr(system, "exchange_device", None)
    factory = getattr(device, "bloch_engine", None)
    engine = None
    if factory is not None:
        engine = factory(system, definition, k_points, k_weights)
    elif _device_engine is not None:
        # The k-point pool on the GPUs: the exchange there too (None: the pool runs on the CPU).
        engine = _device_engine(system, definition, k_points, k_weights)
    if engine is None:
        engine = BlochExchange(system, definition, k_points, k_weights)
    result = run_hybrid_scf(
        system,
        callback=callback,
        warm_start=warm_start,
        convergence_criterion=convergence_criterion,
        iteration_state_callback=iteration_state_callback,
        engine=engine,
    )
    # A device engine hands host copies of the orbitals to what reads the result (the forces below too).
    finish = getattr(engine, "finish", None)
    if finish is not None:
        result = finish(result)
    if compute_forces and not system.input.scf.spin_polarized:
        from dataclasses import replace

        result = replace(result, nonlocal_forces=bloch_nonlocal_forces(system, result))
    return result


__all__ = [
    "BlochChannelExchange",
    "BlochExchange",
    "BlochOccupiedBlock",
    "PeriodicFockOperator",
    "bloch_nonlocal_forces",
    "bloch_orbital_sets_of",
    "bloch_warm_start",
    "full_mesh_counts",
    "uniform_mesh_counts",
    "mesh_counts",
    "periodic_pair_potentials",
    "run_periodic_hybrid_scf",
    "set_device_engine",
]
