"""Self-consistent hybrid functionals of an isolated system: the Fock loop.

Generalized Kohn-Sham with a nonlocal exchange operator.  The functional is

``E_xc = a_x E_x^HF[kernel] + E_xc^semilocal``,

the semilocal part with the hybrid's reduced semilocal exchange (for PBE0
``0.75 E_x^PBE + E_c^PBE``), and the Hamiltonian of every orbital

``H = -nabla^2 + V_ion + V_H + V_xc^semilocal + V_NL + a_x V_x``

(Rydberg).  ``V_x`` is the Fock exchange operator of the occupied orbitals of
the same spin (:mod:`parsec_python.V_xc.exchange_operator`), applied in its
adaptively compressed form ``V_ACE`` (Lin, J. Chem. Theory Comput. 12, 2242
(2016); SPARC: Jing, Suryanarayana, arXiv:2406.16998, Sec. 2.2, and Jing,
Sharma, Pask, Suryanarayana, arXiv:2501.16572, Sec. 2-3, from whose
principles this loop is built; no SPARC code is used).

The loop (the scheme of Quantum ESPRESSO's ACE path, ``electrons.f90``):

0. a semilocal SCF with the start functional (PBE for every hybrid of this
   module: PBE0, PBEh, HSE and HF start from the PBE density and orbitals,
   B3LYP and B3PW91 too -- a GGA ground state is a close start for all of
   them and needs no further functional), unless the SCF of the structure
   before a move supplies the orbitals;
1. ``V_ACE`` from the current orbitals ``P0`` (``n_states`` orbitals of each
   spin channel, the occupied ones and the empty states the input asks for)
   and ``E_x(P0)``;
2. an inner SCF (:func:`parsec_python.SCF.run_scf` with that operator fixed in
   every Hamiltonian, the eigensolvers and their Rayleigh-Ritz steps
   included), warm-started from the loop before;
3. ``E_x(P)`` of the new orbitals ``P`` and the estimated exchange error

   ``dexx = a_x [2 E_x(P, P0) - E_x(P0) - E_x(P)] >= 0``

   (QE's ``dexx = fock1 - (fock0 + fock2)/2``; with ``B`` the positive Coulomb
   form of two density matrices, ``E_x(P, Q) = -B(P, Q)/2`` and ``dexx =
   a_x B(P - P0, P - P0)/2``, an estimate of the energy error of the step;
   ``E_x(P, P0) = (w/2) sum f <q|V_ACE[P0]|q>`` is evaluated with the
   compressed operator, which makes ``dexx`` larger, never smaller);
4. stop when ``dexx < Exx_Tolerance`` (and the inner loop converged), else
   back to 1 with ``P0 = P``.

Energies.  The band energy ``w sum_i f_i e_i`` of the inner loop holds
``a_x w sum_i f_i <q_i|V_ACE|q_i> = 2 a_x E_x(P, P0)`` of exchange.  The inner
loop minimizes the functional with the exchange linearized about ``P0``,

``E_inner = E_formula - 2 a_x E_x(P, P0) + a_x [2 E_x(P, P0) - E_x(P0)]
          = E_formula - a_x E_x(P0)``,

``E_formula`` the total of :func:`parsec_python.Energy.total_energy` with the
semilocal exchange-correlation energy; that is what each inner iteration
reports (QE's ``etot`` inside ``electrons_scf``).  After the inner loop the
total energy of the hybrid functional is

``E = E_formula - 2 a_x E_x(P, P0) + a_x E_x(P) = E_inner - dexx``.

Forces.  With norm-conserving pseudopotentials ``V_x`` has no explicit
dependence on the ions, so the Hellmann-Feynman forces keep their form with
the hybrid orbitals and density; the core-correction force reads the
semilocal potential (``result.xc_potential``), the derivative of the only
term that sees the core density.

Spin.  Each channel exchanges with itself: one operator per channel, built
from the channel's orbitals with one electron per state (spin weight 1); an
unpolarized orbital set stands for two equal channels (weight 2).

Pair solves and defaults.  The pair potentials come from the Kronecker-product
solver of the finite-difference Poisson problem on the bounding box of the
domain (``Exx_Pair_Solver: kronecker``, :mod:`parsec_python.Hartree.kronecker`,
every kernel), or from the conjugate gradients of the Hartree solver
(``cg``, 1/r only).  ``Exx_Tolerance`` 1e-6 Ry is Quantum ESPRESSO's default
threshold of ``dexx``; measured on H2, H2O and CH4 (2026-11-06) the energy
still to be gained after an outer iteration is about ``dexx/10``, and
``dexx`` falls about 100x per outer iteration with PBE0 and B3LYP, 4-13x with
HF.  The last outer iteration evaluates only the occupied pairs.

This module composes the SCF loop and the exchange operator; it does not
print or open files.  The outer iterations go to the SCF ``callback`` as
:class:`parsec_python.models.ExchangeLoopIteration` records, between the
:class:`~parsec_python.models.SCFIteration` records of the inner loops.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from ..models import (
    ExchangeLoopIteration,
    HybridEnergyBreakdown,
    RunTimings,
    SinglePointResult,
)
from ..V_xc.exact_exchange import OCCUPATION_THRESHOLD
from ..V_xc.exchange_operator import (
    CompressedExchange,
    apply_exchange_to_orbitals,
    cached_pair_solver,
    compress_exchange,
    exchange_band_energy,
)


# ---------------------------------------------------------------------------
# The functionals: parsec_python.V_xc.hybrid (registry, fractions, kernels and
# semilocal parts) for the hybrid of an SCFSettings.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridDefinition:
    """What the Fock loop needs of the hybrid of an input.

    ``functional`` is the :class:`parsec_python.V_xc.hybrid.HybridFunctional`
    with the overrides of the input and ``terms`` its exact exchange as
    ``(kernel, fraction, omega)``: ``"coulomb"`` ``1/r``, ``"erfc"``
    ``erfc(omega r)/r``, ``"erf"`` ``erf(omega r)/r``, omega in bohr^-1.  The
    loop builds one exchange operator from one pair solver: with one term
    ``fraction`` is its fraction ``a_x`` and ``kernel``/``omega`` its kernel;
    with two (CAM-B3LYP) the solver returns the weighted sum of the two
    potentials, ``fraction`` is 1 and ``kernel`` names both.
    ``start_functional`` is the semilocal ``xc_functional`` of the SCF that
    starts the loop, ``pair_solver`` the ``Exx_Pair_Solver`` of the input.
    """

    label: str
    display_name: str
    fraction: float
    kernel: str = "coulomb"
    omega: float | None = None
    start_functional: str = "pbe"
    functional: object | None = None
    terms: tuple = ()
    pair_solver: str = "kronecker"


def hybrid_display_name(label: str) -> str:
    """Report name of a hybrid ``xc_functional`` label."""

    from ..V_xc.hybrid import HYBRID_FUNCTIONALS

    functional = HYBRID_FUNCTIONALS.get(label)
    return label.upper() if functional is None else functional.name


@lru_cache(maxsize=64)
def _functional(label: str, fraction: float | None, screening: float | None):
    from ..V_xc.hybrid import hybrid_functional

    return hybrid_functional(
        label, exact_exchange_fraction=fraction, screening_parameter=screening
    )


def hybrid_functional_of(scf):
    """The :class:`~parsec_python.V_xc.hybrid.HybridFunctional` of ``scf`` (overrides applied)."""

    settings = scf.exact_exchange
    return _functional(
        scf.xc_functional,
        None if settings is None else settings.fraction,
        None if settings is None else settings.screening,
    )


def hybrid_definition(scf) -> HybridDefinition:
    """The functional of ``scf.xc_functional`` with ``Exact_Exchange_Fraction``
    and ``Screening_Parameter`` of the input (ValueError where the functional
    does not take them, :func:`parsec_python.V_xc.hybrid.hybrid_functional`).

    One kernel: ``fraction`` is its fraction and the operator is built with a
    pair solver of that kernel.  Two kernels (CAM-B3LYP: ``0.19/r + 0.46
    erf(w r)/r``): the pair solver returns the weighted sum of their
    potentials and ``fraction`` is 1, so ``E_x`` is the weighted exact
    exchange.
    """

    functional = hybrid_functional_of(scf)
    terms = tuple(functional.exact_exchange_terms)
    if not terms:
        raise ValueError(f"{functional.name} has no exact exchange")
    if len(terms) == 1:
        term = terms[0]
        fraction, kernel = float(term.fraction), term.kernel
        omega = None if term.omega is None else float(term.omega)
    else:
        fraction, kernel, omega = 1.0, "+".join(term.kernel for term in terms), None
    return HybridDefinition(
        label=scf.xc_functional,
        display_name=functional.name,
        fraction=fraction,
        kernel=kernel,
        omega=omega,
        start_functional=functional.initial_functional,
        functional=functional,
        terms=tuple((term.kernel, float(term.fraction), term.omega) for term in terms),
        pair_solver=getattr(scf.exact_exchange, "pair_solver", "kronecker"),
    )


def scf_semilocal_xc(scf, valence_density, grid, core_density=None):
    """The semilocal part of the hybrid ``scf.xc_functional`` (unpolarized), in Ry."""

    from ..V_xc.hybrid import hybrid_semilocal_xc

    return hybrid_semilocal_xc(hybrid_functional_of(scf), valence_density, grid, core_density)


def scf_semilocal_spin_xc(scf, valence_up, valence_down, grid, core_density=None):
    """The semilocal part of the hybrid ``scf.xc_functional`` for two spin densities, in Ry.

    Each channel sees half of the core density, as in the semilocal functionals.
    """

    from ..V_xc.hybrid import hybrid_semilocal_xc_spin

    return hybrid_semilocal_xc_spin(
        hybrid_functional_of(scf), valence_up, valence_down, grid, core_density
    )


# ---------------------------------------------------------------------------
# The operator of one inner loop and its energies.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitalSet:
    """The orbitals of one spin channel (or of both, ``spin_weight`` 2), on the host."""

    orbitals: np.ndarray
    occupations: np.ndarray
    spin_weight: float


def orbital_sets_of(result) -> tuple[OrbitalSet, ...]:
    """The orbital sets of an SCF result: one unpolarized set or two channels."""

    channels = getattr(result, "spin", None)
    if channels is None:
        return (
            OrbitalSet(
                _host(result.wavefunctions),
                np.asarray(result.occupations, dtype=np.float64),
                2.0,
            ),
        )
    return tuple(
        OrbitalSet(
            _host(channel.wavefunctions),
            np.asarray(channel.occupations, dtype=np.float64),
            1.0,
        )
        for channel in channels
    )


def _host(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=np.float64)
    full = getattr(values, "to_full_host", None)
    if callable(full):
        return np.asarray(full(), dtype=np.float64)
    download = getattr(values, "get", None)
    if callable(download):
        return np.asarray(download(), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class FockOperator:
    """The exact-exchange term of one inner SCF loop.

    ``channels`` holds the compressed operator of each spin channel (one
    entry for an unpolarized loop), ``None`` for a channel without electrons;
    each carries the fraction ``a_x``.  ``reference_energy`` is ``E_x(P0)``
    of the orbitals the operators were built from (Ry, unscaled, all
    channels).
    """

    channels: tuple[CompressedExchange | None, ...]
    fraction: float
    reference_energy: float

    def operator(self, spin: int = 0) -> CompressedExchange | None:
        """``a_x V_ACE`` of channel ``spin`` for the Hamiltonian."""

        return self.channels[spin]

    def cross_energy(self, orbitals: Sequence, occupations: Sequence) -> float:
        """``E_x(P, P0) = sum_s (w/2) sum_i f_i <q_i|V_ACE[P0]|q_i>`` (Ry, unscaled)."""

        if len(orbitals) != len(self.channels):
            raise ValueError("one orbital block per spin channel of the operator is needed")
        return 0.5 * sum(
            exchange_band_energy(channel, _host(block), weights, scaled=False)
            for channel, block, weights in zip(self.channels, orbitals, occupations)
        )

    def iteration_energies(self, energies, orbitals: Sequence, occupations: Sequence) -> HybridEnergyBreakdown:
        """The energies of an inner iteration with the exchange of this loop.

        ``energies`` is the :func:`parsec_python.Energy.total_energy` (or
        spin) breakdown with the semilocal exchange-correlation energy and
        the band energy of the Hamiltonian with ``a_x V_ACE``; the
        operator's part of the band energy, ``2 a_x E_x(P, P0)``, is removed
        and the linearized exact exchange ``a_x [2 E_x(P, P0) - E_x(P0)]``
        added.
        """

        cross = self.cross_energy(orbitals, occupations)
        band = 2.0 * self.fraction * cross
        exact = self.fraction * (2.0 * cross - self.reference_energy)
        semilocal = float(energies.exchange_correlation)
        return HybridEnergyBreakdown(
            eigenvalue=float(energies.eigenvalue),
            hartree=float(energies.hartree),
            integral_vxc_rho=float(energies.integral_vxc_rho),
            exchange_correlation=semilocal + exact,
            electron_ion=float(energies.electron_ion),
            ion_ion=float(energies.ion_ion),
            electronic=float(energies.electronic) - band + exact,
            total=float(energies.total) - band + exact,
            semilocal_exchange_correlation=semilocal,
            exact_exchange=exact,
            exact_exchange_band=band,
            exact_exchange_fraction=self.fraction,
            linearized=True,
        )


def energies_with_exchange(energies: HybridEnergyBreakdown, exchange_energy: float) -> HybridEnergyBreakdown:
    """``energies`` of the last inner iteration with ``a_x E_x(P)`` of its orbitals."""

    exact = energies.exact_exchange_fraction * float(exchange_energy)
    change = exact - energies.exact_exchange
    return replace(
        energies,
        exchange_correlation=energies.semilocal_exchange_correlation + exact,
        electronic=energies.electronic + change,
        total=energies.total + change,
        exact_exchange=exact,
        linearized=False,
    )


@dataclass(frozen=True)
class FockLoopSummary:
    """The Fock loop of a hybrid SCF (``SinglePointResult.exact_exchange``).

    ``iterations`` are the records reported to the callback, ``exchange_energy``
    is ``E_x`` of the final orbitals (Ry, unscaled), ``estimated_error`` the
    last ``dexx``; ``start_iterations`` the SCF iterations of the semilocal
    start (0 when the orbitals of a structure before a move started the loop).

    Diagnostics of the loop (:func:`fock_loop_diagnostics`):
    ``convergence_ratio`` is the factor by which the energy steps of the last
    outer iterations shrank (``None`` with fewer than three steps), and
    ``remaining_energy`` the energy still to be gained if they go on
    shrinking so, ``step * r / (1 - r)`` (Ry; ``None`` unless ``0 < r < 1``).
    ``instability`` is the outer iteration at which dexx rose more than
    tenfold above its smallest earlier value (the state the loop approached
    was unstable under exact exchange, for example a symmetric ensemble that
    breaks its symmetry), ``None`` otherwise.  ``fractional_occupation`` is
    the largest ``min(f, 1 - f)`` of the final occupations.
    """

    functional: str
    display_name: str
    fraction: float
    kernel: str
    omega: float | None
    iterations: tuple[ExchangeLoopIteration, ...]
    converged: bool
    exchange_energy: float
    estimated_error: float
    tolerance: float
    start_functional: str | None
    start_iterations: int
    pair_solver: str
    pair_solves: int
    poisson_seconds: float
    exchange_seconds: float
    seconds: float
    convergence_ratio: float | None = None
    remaining_energy: float | None = None
    instability: int | None = None
    fractional_occupation: float = 0.0


#: dexx rising this many times above its smallest earlier value marks an instability.
INSTABILITY_FACTOR = 10.0
#: A final occupation farther than this from 0 and 1 is reported as fractional.
FRACTIONAL_REPORT = 0.01


def fock_loop_diagnostics(records: Sequence[ExchangeLoopIteration], occupations) -> dict:
    """``convergence_ratio``, ``remaining_energy``, ``instability`` and
    ``fractional_occupation`` of a Fock loop (see :class:`FockLoopSummary`).

    Measured on a ring of 12 hydrogen atoms (bond 1.40 bohr, PBE0, 80 K;
    2026-11-06): the PBE start is a symmetric ensemble (the degenerate +-3
    pair half filled), a stationary point of PBE0 that is unstable (dexx 2.6e-7 at outer 2, then
    rising to 3e-2 at outer 6); the loop then reaches a bond-alternating
    state 67 mRy lower and creeps along its soft rotation with energy steps
    shrinking by 0.90 per outer iteration.
    """

    totals = [record.total_energy for record in records if record.iteration >= 1 and record.total_energy is not None]
    steps = [earlier - later for earlier, later in zip(totals, totals[1:])]
    ratio = remaining = None
    if len(steps) >= 3 and steps[-2] > 0.0 and steps[-3] > 0.0:
        ratios = [steps[-1] / steps[-2], steps[-2] / steps[-3]]
        ratio = float(np.median(ratios))
        if 0.0 < ratio < 1.0 and steps[-1] > 0.0:
            remaining = float(steps[-1] * ratio / (1.0 - ratio))
    instability = None
    smallest = float("inf")
    for record in records:
        error = record.estimated_error
        if error is None:
            continue
        if smallest < float("inf") and error > INSTABILITY_FACTOR * smallest and instability is None:
            instability = int(record.iteration)
        smallest = min(smallest, max(float(error), 1.0e-300))
    values = np.concatenate([np.ravel(np.asarray(item, dtype=np.float64)) for item in occupations] or [np.zeros(0)])
    fractional = float(np.max(np.minimum(values, 1.0 - values), initial=0.0)) if values.size else 0.0
    return {
        "convergence_ratio": ratio,
        "remaining_energy": remaining,
        "instability": instability,
        "fractional_occupation": max(fractional, 0.0),
    }


def _final_occupations(result):
    channels = getattr(result, "spin", None)
    if channels is None:
        return [np.asarray(result.occupations, dtype=np.float64)]
    return [np.asarray(channel.occupations, dtype=np.float64) for channel in channels]


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


#: Pair solvers (``Exx_Pair_Solver``) and the kernels they serve.
PAIR_SOLVER_KERNELS = {
    "kronecker": ("coulomb", "erfc", "erf"),
    "cg": ("coulomb",),
}


def require_supported(definition: HybridDefinition) -> HybridDefinition:
    """Refuse a functional this build cannot run: a semilocal piece that is not
    implemented yet, or a kernel the chosen pair solver does not serve."""

    functional = definition.functional
    missing = [
        component.name
        for component in getattr(functional, "components", ())
        if component.weight != 0.0 and not getattr(component.form, "implemented", True)
    ]
    if missing:
        raise NotImplementedError(
            f"{definition.display_name} needs the semilocal part(s) {', '.join(missing)}, "
            "which parsec_python.V_xc does not implement yet"
        )
    served = PAIR_SOLVER_KERNELS[definition.pair_solver]
    for kernel, _, _ in definition.terms:
        if kernel not in served:
            raise NotImplementedError(
                f"the {kernel} exchange kernel of {definition.display_name} needs "
                "Exx_Pair_Solver: kronecker (the conjugate gradients serve 1/r only)"
            )
    return definition


# Kept for the parser and callers of the first version.
require_pair_solver = require_supported


class WeightedPairPoisson:
    """``sum_t a_t v_t[rho]``: one pair solver per kernel of a two-kernel functional."""

    def __init__(self, parts) -> None:
        self.parts = tuple((float(weight), solver) for weight, solver in parts)
        self.description = " + ".join(
            f"{weight:g} x [{getattr(solver, 'description', type(solver).__name__)}]"
            for weight, solver in self.parts
        )

    def solve_batch(self, densities: np.ndarray) -> np.ndarray:
        total = None
        for weight, solver in self.parts:
            values = weight * np.asarray(solver.solve_batch(densities), dtype=np.float64)
            total = values if total is None else total + values
        return total


def _kernel_solver(system, kernel: str, omega, choice: str):
    from ..V_xc.exact_exchange import pair_hartree_settings

    settings = pair_hartree_settings(system.input.hartree)
    if choice == "cg":
        return cached_pair_solver(system.grid, system.negative_laplacian, system.input.hartree)
    from ..Hartree.kronecker import KroneckerPoisson

    solver = KroneckerPoisson(
        system.grid,
        kernel=kernel,
        omega=None if kernel == "coulomb" else float(omega),
        multipole_order=settings.multipole_order,
    )
    if not hasattr(solver, "description"):
        solver.description = (
            f"Kronecker-product FD solver on the bounding box {solver.box_shape}, {kernel} kernel"
            + ("" if kernel == "coulomb" else f" (w = {float(omega):g} bohr^-1)")
            + f", multipole boundary of order {settings.multipole_order}"
        )
    return solver


def _weighted_kernel_solver(system, definition: HybridDefinition):
    """One Kronecker solve of ``a/r + b K(r)`` for a 1/r term and one screened term.

    ``KroneckerPoisson(..., weights=(a, b))`` (split screening): the same
    potentials as the two solves of :class:`WeightedPairPoisson` summed, to
    round-off (measured 2026-11-06: 0 to 1.5e-16 relative on the H2O grid,
    CAM-B3LYP and erfc mixtures), at the cost of one.  ``None`` for any
    other pair of terms or for ``Exx_Pair_Solver: cg``.
    """

    if definition.pair_solver != "kronecker" or len(definition.terms) != 2:
        return None
    coulomb = [term for term in definition.terms if term[0] == "coulomb"]
    screened = [term for term in definition.terms if term[0] != "coulomb"]
    if len(coulomb) != 1 or len(screened) != 1:
        return None
    from ..Hartree.kronecker import KroneckerPoisson
    from ..V_xc.exact_exchange import pair_hartree_settings

    settings = pair_hartree_settings(system.input.hartree)
    kernel, beta, omega = screened[0]
    alpha = coulomb[0][1]
    solver = KroneckerPoisson(
        system.grid,
        kernel=kernel,
        omega=float(omega),
        multipole_order=settings.multipole_order,
        weights=(float(alpha), float(beta)),
    )
    if not hasattr(solver, "description"):
        solver.description = (
            f"Kronecker-product FD solver on the bounding box {solver.box_shape}, kernel "
            f"{alpha:g}/r + {beta:g} {kernel}(w r)/r (w = {float(omega):g} bohr^-1) in one solve, "
            f"multipole boundary of order {settings.multipole_order}"
        )
    return solver


def default_pair_solver(system, definition: HybridDefinition):
    """The pair-density Poisson solver of the run (``Exx_Pair_Solver``).

    ``kronecker`` (the default): the direct solver of the finite-difference
    Poisson problem on the bounding box of the domain
    (:class:`parsec_python.Hartree.kronecker.KroneckerPoisson`), every kernel,
    multipole boundary values at the order of the run.  ``cg``: the conjugate
    gradients of the Hartree solver on the domain itself
    (:func:`~parsec_python.V_xc.exchange_operator.cached_pair_solver`), the
    Coulomb kernel only.  A two-kernel functional gets the weighted sum.
    """

    require_supported(definition)
    if len(definition.terms) == 1:
        kernel, _, omega = definition.terms[0]
        return _kernel_solver(system, kernel, omega, definition.pair_solver)
    weighted = _weighted_kernel_solver(system, definition)
    if weighted is not None:
        return weighted
    return WeightedPairPoisson(
        (fraction, _kernel_solver(system, kernel, omega, definition.pair_solver))
        for kernel, fraction, omega in definition.terms
    )


def semilocal_start_system(system, definition: HybridDefinition):
    """``system`` with the start functional in place of the hybrid."""

    scf = replace(system.input.scf, xc_functional=definition.start_functional, exact_exchange=None)
    return replace(system, input=replace(system.input, scf=scf))


def _transition_warm_start(system, start: SinglePointResult):
    """The first inner loop after the semilocal start.

    As Quantum ESPRESSO after its first SCF: the potential is rebuilt from the
    converged density with the semilocal part of the hybrid
    (``V_ion + V_H[rho] + V_xc^semilocal[rho]``) and the eigensolver keeps
    its subspace; the exchange operator joins the Hamiltonian.
    """

    from .single_point import SCFWarmStart, state_for_new_potential

    ionic = np.asarray(system.ionic_potential, dtype=np.float64)
    hartree = np.asarray(start.hartree_potential, dtype=np.float64)
    channels = getattr(start, "spin", None)
    if channels is None:
        xc = system.evaluate_xc(np.asarray(start.density, dtype=np.float64))
        return SCFWarmStart(
            input_potential=ionic + hartree + xc.potential,
            hartree_potential=hartree,
            eigensolver_state=state_for_new_potential(start.eigensolver_state),
        )
    xc = system.evaluate_spin_xc(
        np.asarray(channels[0].density, dtype=np.float64),
        np.asarray(channels[1].density, dtype=np.float64),
    )
    return SCFWarmStart(
        input_potential=tuple(ionic + hartree + potential for potential in xc.potentials),
        hartree_potential=hartree,
        eigensolver_state=tuple(
            state_for_new_potential(channel.eigensolver_state) for channel in channels
        ),
    )


def _start_total_energy(system, start: SinglePointResult, exchange_energy: float, fraction: float) -> float:
    """The hybrid total energy of the start orbitals and density (Ry).

    ``E_start - E_xc^start[rho] + E_xc^semilocal[rho] + a_x E_x(P)``: kinetic,
    ionic and Hartree terms are the start loop's, the exchange-correlation
    energy is that of the hybrid.
    """

    channels = getattr(start, "spin", None)
    if channels is None:
        semilocal = system.evaluate_xc(np.asarray(start.density, dtype=np.float64)).total_energy
    else:
        semilocal = system.evaluate_spin_xc(
            np.asarray(channels[0].density, dtype=np.float64),
            np.asarray(channels[1].density, dtype=np.float64),
        ).total_energy
    energies = start.energies
    return float(
        energies.total - energies.exchange_correlation + semilocal + fraction * exchange_energy
    )


def _occupied(occupations: np.ndarray) -> list[int]:
    return [int(index) for index in np.flatnonzero(np.asarray(occupations) > OCCUPATION_THRESHOLD)]


def _exchange_of(sets, volume_element, solver, batch_size, known=None, columns="occupied"):
    """``V_x`` of each set on its occupied columns (``columns="occupied"``) or all."""

    applications = []
    for index, orbital_set in enumerate(sets):
        wanted = (
            _occupied(orbital_set.occupations) if columns == "occupied" else None
        )
        applications.append(
            apply_exchange_to_orbitals(
                orbital_set.orbitals,
                orbital_set.occupations,
                volume_element,
                solver,
                spin_weight=orbital_set.spin_weight,
                columns=wanted,
                known=None if known is None else known[index],
                batch_size=batch_size,
            )
        )
    return applications


def _fock_operator(sets, applications, fraction) -> FockOperator:
    channels = []
    for orbital_set, application in zip(sets, applications):
        if not application.occupied:
            channels.append(None)
            continue
        channels.append(
            compress_exchange(
                orbital_set.orbitals,
                application,
                fraction=fraction,
                spin_weight=orbital_set.spin_weight,
            )
        )
    return FockOperator(
        channels=tuple(channels),
        fraction=float(fraction),
        reference_energy=float(sum(application.energy for application in applications)),
    )


@dataclass(frozen=True)
class _Timed:
    """The timings of one SCF loop of the Fock loop."""

    timings: RunTimings


def _sum_timings(results: Sequence[SinglePointResult], seconds: float) -> RunTimings:
    first = results[0].timings
    names = (
        "initial_hartree_seconds",
        "initial_xc_seconds",
        "hamiltonian_binding_seconds",
        "diagonalization_seconds",
        "occupations_density_seconds",
        "hartree_seconds",
        "xc_seconds",
        "mixing_energy_seconds",
    )
    return replace(
        first,
        **{name: float(sum(getattr(result.timings, name) for result in results)) for name in names},
        total_seconds=float(seconds),
    )


class HostExchange:
    """The exchange pieces of the Fock loop on the host (the reference path).

    :func:`run_hybrid_scf` asks an object of this interface for the orbitals
    of an SCF result, ``V_x`` of them, the compressed operator, its energy
    with other orbitals and the warm starts between the loops.  An
    accelerated system brings its own (``system.exchange_device``,
    :mod:`parsec_python.acceleration.SCF.hybrid`), whose operators live in
    its device Hamiltonians.  This one keeps every array on the host and in
    the reference Hamiltonian.
    """

    def __init__(self, system, definition: HybridDefinition, pair_solver=None) -> None:
        self.solver = default_pair_solver(system, definition) if pair_solver is None else pair_solver
        self.description = str(getattr(self.solver, "description", type(self.solver).__name__))
        self.volume_element = system.grid.volume_element
        self.batch_size = system.input.scf.exact_exchange.batch_size

    def start_system(self, system, definition: HybridDefinition):
        return semilocal_start_system(system, definition)

    def orbital_sets(self, result):
        return orbital_sets_of(result)

    def transition_warm_start(self, system, start):
        return _transition_warm_start(system, start)

    def next_warm_start(self, system, result):
        from .single_point import _parsec_warm_start

        return _parsec_warm_start(system, result, system)

    def start_total_energy(self, system, start, exchange_energy: float, fraction: float) -> float:
        return _start_total_energy(system, start, exchange_energy, fraction)

    def exchange(self, sets, *, columns: str = "occupied", known=None):
        return _exchange_of(
            sets, self.volume_element, self.solver, self.batch_size, known=known, columns=columns
        )

    def fock_operator(self, sets, applications, fraction: float) -> FockOperator:
        return _fock_operator(sets, applications, fraction)

    def cross_energy(self, operator: FockOperator, sets) -> float:
        return operator.cross_energy(
            [orbital_set.orbitals for orbital_set in sets],
            [orbital_set.occupations for orbital_set in sets],
        )

    def release(self, operator: FockOperator) -> None:
        """Nothing to free: the host operator goes with its last reference."""

    def drop_orbitals(self, sets, result=None) -> None:
        """Nothing to drop: host orbitals pin no eigensolver memory."""


def run_hybrid_scf(
    system,
    *,
    callback: Callable[[object], None] | None = None,
    warm_start=None,
    convergence_criterion: float | None = None,
    iteration_state_callback: Callable[[object], None] | None = None,
    pair_solver=None,
    **hooks,
) -> SinglePointResult:
    """The Fock loop of a hybrid ``system.input.scf.xc_functional`` (see the module).

    ``warm_start`` continues from the SCF of a previous structure; when it
    carries that structure's orbitals (``exchange_orbitals``) they build the
    first exchange operator and no semilocal start is run.
    ``convergence_criterion`` is that of every inner loop.  ``pair_solver``
    replaces :func:`default_pair_solver` (an object with ``solve_batch``).
    The execution hooks of an accelerated backend (``hooks``) are refused
    unless the system holds the exchange operator in its own Hamiltonians
    (``system.exchange_device``, whose ``fock_engine`` supplies the pieces
    of :class:`HostExchange` on its devices); they then go to every SCF loop
    of the Fock loop.

    The result is the one of the last inner loop with the energies of the
    hybrid functional, the history of every loop, ``converged`` only when
    the last inner loop and the Fock loop converged, and the
    :class:`FockLoopSummary` in ``exact_exchange``.
    """

    from .single_point import refuse_exchange_hooks, run_scf

    started = perf_counter()
    scf = system.input.scf
    definition = hybrid_definition(scf)
    settings = scf.exact_exchange
    fraction = definition.fraction
    device = getattr(system, "exchange_device", None)
    if device is None:
        refuse_exchange_hooks(**hooks)
        engine = HostExchange(system, definition, pair_solver)
        keywords = {}
    else:
        if pair_solver is not None:
            raise ValueError("the device exchange operator brings its own pair solver")
        engine = device.fock_engine(system, definition)
        keywords = {name: hook for name, hook in hooks.items() if hook is not None}
    if convergence_criterion is not None:
        keywords["convergence_criterion"] = convergence_criterion
    if iteration_state_callback is not None:
        keywords["iteration_state_callback"] = iteration_state_callback

    def report(record: ExchangeLoopIteration) -> None:
        records.append(record)
        if callback is not None:
            callback(record)

    records: list[ExchangeLoopIteration] = []
    history: list = []
    # The timings of every SCF loop (the results themselves would keep
    # their orbitals).
    loops: list = []
    carried = None if warm_start is None else getattr(warm_start, "exchange_orbitals", None)
    pair_solves = 0
    poisson_seconds = 0.0
    exchange_seconds = 0.0

    # 0. The orbitals the first operator is built from.
    if carried is None:
        start_system = engine.start_system(system, definition)
        start = run_scf(start_system, callback=callback, warm_start=warm_start, **keywords)
        history.extend(start.history)
        loops.append(_Timed(start.timings))
        sets = engine.orbital_sets(start)
        inner_start = engine.transition_warm_start(system, start)
        start_iterations = int(start.iterations)
        start_functional = definition.start_functional
    else:
        sets = tuple(carried)
        inner_start = replace(warm_start, exchange_orbitals=None)
        start = None
        start_iterations = 0
        start_functional = None
    if len(sets) != (2 if scf.spin_polarized else 1):
        raise ValueError("the start orbitals do not match the spin channels of the system")

    # 1. The operator of the start orbitals.
    step_started = perf_counter()
    applications = engine.exchange(sets, columns="all")
    operator = engine.fock_operator(sets, applications, fraction)
    # The orbitals of the start (or of the structure before a move) are not
    # read again: an engine whose orbitals are views of an eigensolver's
    # subspace lets them go before that subspace moves on.
    engine.drop_orbitals(sets, start)
    step_seconds = perf_counter() - step_started
    step_solves = sum(application.pair_solves for application in applications)
    step_poisson = sum(application.poisson_seconds for application in applications)
    pair_solves += step_solves
    poisson_seconds += step_poisson
    exchange_seconds += step_seconds
    previous_energy = operator.reference_energy
    # (A device engine may have built its solver beside the start.)
    solver_name = engine.description
    report(
        ExchangeLoopIteration(
            iteration=0,
            functional=definition.display_name,
            fraction=fraction,
            exchange_energy=previous_energy,
            total_energy=(
                None
                if start is None
                else engine.start_total_energy(system, start, previous_energy, fraction)
            ),
            tolerance=settings.tolerance,
            inner_iterations=start_iterations,
            inner_converged=True if start is None else bool(start.converged),
            pair_solves=step_solves,
            poisson_seconds=step_poisson,
            exchange_seconds=step_seconds,
            operator_ranks=tuple(0 if channel is None else channel.rank for channel in operator.channels),
            start_functional=start_functional,
            pair_solver=solver_name,
            operator_asymmetry=max(
                (channel.asymmetry for channel in operator.channels if channel is not None),
                default=0.0,
            ),
        )
    )

    converged = False
    estimated_error = float("inf")
    inner = None
    exchange_energy = previous_energy
    for outer in range(1, settings.max_iterations + 1):
        # 2. The inner SCF with the operator fixed.
        inner = run_scf(
            system,
            callback=callback,
            warm_start=inner_start,
            exchange_term=operator,
            **keywords,
        )
        history.extend(inner.history)
        loops.append(_Timed(inner.timings))
        sets = engine.orbital_sets(inner)
        # 3. E_x of the new orbitals with the operator of this loop (which is
        # then no longer needed), E_x of them with their own (occupied
        # columns only) and dexx.
        step_started = perf_counter()
        cross = engine.cross_energy(operator, sets)
        engine.release(operator)
        occupied_applications = engine.exchange(sets)
        exchange_energy = float(sum(application.energy for application in occupied_applications))
        estimated_error = fraction * (2.0 * cross - previous_energy - exchange_energy)
        energies = energies_with_exchange(inner.energies, exchange_energy)
        inner = replace(inner, energies=energies)
        converged = bool(estimated_error < settings.tolerance and inner.converged)
        last = converged or outer == settings.max_iterations
        ranks: tuple[int, ...] = ()
        asymmetry = 0.0
        if not last:
            # 1'. The operator of the new orbitals: the empty columns too.
            applications = engine.exchange(sets, known=occupied_applications, columns="all")
            operator = engine.fock_operator(sets, applications, fraction)
            engine.drop_orbitals(sets, inner)
            ranks = tuple(0 if channel is None else channel.rank for channel in operator.channels)
            asymmetry = max(
                (channel.asymmetry for channel in operator.channels if channel is not None),
                default=0.0,
            )
        else:
            applications = occupied_applications
        step_seconds = perf_counter() - step_started
        step_solves = sum(application.pair_solves for application in applications)
        step_poisson = sum(application.poisson_seconds for application in applications)
        pair_solves += step_solves
        poisson_seconds += step_poisson
        exchange_seconds += step_seconds
        report(
            ExchangeLoopIteration(
                iteration=outer,
                functional=definition.display_name,
                fraction=fraction,
                exchange_energy=exchange_energy,
                total_energy=float(energies.total),
                previous_exchange_energy=previous_energy,
                cross_exchange_energy=cross,
                estimated_error=float(estimated_error),
                tolerance=settings.tolerance,
                inner_iterations=int(inner.iterations),
                inner_converged=bool(inner.converged),
                converged=converged,
                pair_solves=step_solves,
                poisson_seconds=step_poisson,
                exchange_seconds=step_seconds,
                operator_ranks=ranks,
                pair_solver=solver_name,
                operator_asymmetry=asymmetry,
            )
        )
        if last:
            break
        previous_energy = exchange_energy
        inner_start = engine.next_warm_start(system, inner)

    seconds = perf_counter() - started
    summary = FockLoopSummary(
        functional=definition.label,
        display_name=definition.display_name,
        fraction=fraction,
        kernel=definition.kernel,
        omega=definition.omega,
        iterations=tuple(records),
        converged=converged,
        exchange_energy=float(exchange_energy),
        estimated_error=float(estimated_error),
        tolerance=settings.tolerance,
        start_functional=start_functional,
        start_iterations=start_iterations,
        pair_solver=solver_name,
        pair_solves=pair_solves,
        poisson_seconds=poisson_seconds,
        exchange_seconds=exchange_seconds,
        seconds=seconds,
        **fock_loop_diagnostics(records, _final_occupations(inner)),
    )
    return replace(
        inner,
        converged=converged,
        iterations=len(history),
        history=history,
        timings=_sum_timings(loops, seconds),
        exact_exchange=summary,
    )


def exchange_start_orbitals(result) -> tuple[OrbitalSet, ...] | None:
    """The orbitals of a hybrid result for the first operator of the next structure.

    ``None`` for a result without a Fock loop.
    """

    if getattr(result, "exact_exchange", None) is None:
        return None
    return orbital_sets_of(result)


__all__ = [
    "FockLoopSummary",
    "FockOperator",
    "HostExchange",
    "HybridDefinition",
    "OrbitalSet",
    "default_pair_solver",
    "energies_with_exchange",
    "exchange_start_orbitals",
    "hybrid_definition",
    "hybrid_display_name",
    "hybrid_functional_of",
    "orbital_sets_of",
    "PAIR_SOLVER_KERNELS",
    "WeightedPairPoisson",
    "require_pair_solver",
    "require_supported",
    "run_hybrid_scf",
    "scf_semilocal_spin_xc",
    "scf_semilocal_xc",
    "semilocal_start_system",
]
