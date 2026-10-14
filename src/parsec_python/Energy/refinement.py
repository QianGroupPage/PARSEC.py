"""Post-SCF refinement of the total energy by a hybrid functional.

The refinement of Xu, Prendergast and Qian (J. Chem. Theory Comput. 18,
5471 (2022); ``Lrefine=T`` in ARES) is one non-self-consistent evaluation:
the exchange-correlation energy of the converged orbitals and density is
evaluated once more with B3LYP and replaces the one of the self-consistent
functional in the total energy,

``E_refined = E_total - E_xc[SCF functional] + E_xc[B3LYP]``.

Kinetic, ionic, Hartree and ion-ion energies are those of the SCF, and so is
the atomic all-electron-minus-pseudo reference correction: the
pseudopotentials and their atomic reference energies are the ones of the
self-consistent functional.  With PBE as that functional the result is the
"PP-PBE(B3LYP)" energy of the paper.

Nothing here changes the SCF result.  :func:`refine_energy` reads a
converged result and returns an :class:`EnergyRefinement`;
:func:`refinement_report_lines` formats it for ``parsec.out``.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from time import perf_counter
from typing import Callable

import numpy as np

from ..V_xc.b3lyp import (
    B3LYP_B88_EXCHANGE,
    B3LYP_EXACT_EXCHANGE,
    B3LYP_LYP_CORRELATION,
    B3LYP_SLATER_EXCHANGE,
    B3LYP_VWN_CORRELATION,
    B3LYPSemilocalEnergies,
    b3lyp_exchange_correlation_energy,
    b3lyp_semilocal_energies,
    b3lyp_spin_semilocal_energies,
)
from ..V_xc.exact_exchange import (
    ExactExchangeResult,
    exact_exchange_energy,
    reference_pair_poisson,
    spin_exact_exchange_energy,
)


#: Functionals ``Energy_Refinement`` accepts.
REFINEMENT_FUNCTIONALS = ("b3lyp",)

_SCF_FUNCTIONAL_NAMES = {"ca": "CA-LDA", "pbe": "PBE"}


def normalize_refinement(value: str | None) -> str | None:
    """Return the functional a refinement request names, ``None`` for off.

    Accepts the functional names of :data:`REFINEMENT_FUNCTIONALS` and the
    spellings of "off" (``none``, ``off``, ``false``, ``no``, empty).
    """

    if value is None:
        return None
    label = "".join(
        character for character in str(value).strip().lower() if character.isalnum()
    )
    if label in {"", "none", "off", "false", "no", "f", "0"}:
        return None
    if label in REFINEMENT_FUNCTIONALS:
        return label
    raise ValueError(
        f"Energy_Refinement={value!r}; supported choices are "
        + ", ".join(REFINEMENT_FUNCTIONALS)
        + " and none"
    )


def check_refinement_request(
    functional: str | None,
    *,
    periodic: bool,
    molecular_dynamics: bool,
) -> None:
    """Refuse a refinement the calculation cannot give; ``None`` is always fine."""

    if functional is None:
        return
    if periodic:
        raise ValueError(
            "Energy_Refinement is implemented for isolated systems "
            "(Boundary_Conditions: cluster) only: the exact exchange uses the "
            "open-boundary Poisson solver"
        )
    if molecular_dynamics:
        raise ValueError(
            "Energy_Refinement is not available with Molecular_Dynamics; "
            "refine the structures of interest as single points"
        )


def apply_refinement_option(translation, option: str | None):
    """Let a command-line ``--energy-refinement`` overrule the input.

    ``option`` is ``None`` where the command line says nothing: the
    translation is then returned as it is.  Otherwise it names a functional
    or ``none`` and replaces ``Energy_Refinement`` of the input.
    """

    if option is None:
        return translation
    from dataclasses import replace

    functional = normalize_refinement(option)
    check_refinement_request(
        functional,
        periodic=translation.problem.periodic_cell is not None,
        molecular_dynamics=translation.molecular_dynamics is not None,
    )
    return replace(translation, energy_refinement=functional)


@dataclass(frozen=True)
class EnergyRefinement:
    """B3LYP energy of converged orbitals; every energy in Rydberg.

    ``exact_exchange``, ``slater_exchange``, ``b88_exchange``,
    ``vwn_correlation`` and ``lyp_correlation`` are the unweighted parts
    (``b88_exchange`` includes its Slater part, ``vwn_correlation`` is the
    RPA parametrization); ``exchange_correlation`` is their B3LYP
    combination.  ``scf_exchange_correlation`` and ``scf_total`` are the
    values of the self-consistent functional, ``total`` is the refined
    pseudopotential total energy.
    """

    functional: str
    scf_functional: str
    exact_exchange: float
    slater_exchange: float
    b88_exchange: float
    vwn_correlation: float
    lyp_correlation: float
    exchange_correlation: float
    scf_exchange_correlation: float
    scf_total: float
    total: float
    atomic_reference_correction: float
    exchange: ExactExchangeResult
    core_correction: bool
    scf_converged: bool
    pair_solver: str
    semilocal_seconds: float
    seconds: float
    # Resident high-water mark of the process before and after the
    # refinement, in bytes; ``None`` where the system does not tell.
    memory_before: int | None = None
    memory_after: int | None = None
    # True for the orbitals and densities of two spin channels: the exact
    # exchange is then that of each channel and the density functionals are
    # the spin-polarized ones.
    spin_polarized: bool = False

    @property
    def all_electron_total(self) -> float:
        """Refined total plus the atomic reference correction, in Ry."""

        return float(self.total + self.atomic_reference_correction)

    @property
    def scf_all_electron_total(self) -> float:
        """Self-consistent total plus the atomic reference correction, in Ry."""

        return float(self.scf_total + self.atomic_reference_correction)

    @property
    def correction(self) -> float:
        """``E_xc[B3LYP] - E_xc[SCF functional]`` in Ry."""

        return float(self.exchange_correlation - self.scf_exchange_correlation)


def process_memory_high_water() -> int | None:
    """Resident high-water mark of this process in bytes, where POSIX tells.

    ``None`` on a system without the ``resource`` module (Windows); the
    accelerated driver passes a probe that covers it.
    """

    try:
        import resource
    except ImportError:
        return None
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


# Largest deviation, relative to the largest density, by which the orbitals
# of a result may miss its density.  The orbitals of the last eigenproblem
# give it to round-off (1e-16 on the host and device paths tested, with and
# without symmetry); the bound leaves room for a density that a backend
# symmetrized and is far below what another set of orbitals gives.
ORBITAL_DENSITY_TOLERANCE = 1.0e-4


def orbital_density_deviation(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    density: np.ndarray,
    volume_element: float,
) -> float:
    """Largest ``|2 sum_i f_i psi_i**2 - density|`` over the largest density."""

    occupations = np.asarray(occupations, dtype=np.float64)
    rebuilt = np.zeros(density.shape, dtype=np.float64)
    for column in np.flatnonzero(occupations > 0.0):
        orbital = np.asarray(wavefunctions[:, column], dtype=np.float64)
        rebuilt += (2.0 * occupations[column] / volume_element) * orbital * orbital
    scale = float(np.max(np.abs(density), initial=0.0))
    if scale == 0.0:
        return float(np.max(np.abs(rebuilt), initial=0.0))
    return float(np.max(np.abs(rebuilt - density))) / scale


def refine_energy(
    system,
    result,
    functional: str = "b3lyp",
    *,
    pair_poisson: Callable[[np.ndarray], object] | None = None,
    pair_solver: str | None = None,
    progress: Callable[[int, int, float], None] | None = None,
    memory_probe: Callable[[], int | None] = process_memory_high_water,
    memory_before: int | None = None,
) -> EnergyRefinement:
    """Evaluate the B3LYP energy of the orbitals and density of ``result``.

    Parameters
    ----------
    system
        The prepared system the result belongs to (grid, Hartree settings and
        finite-difference Laplacian).
    result
        Its SCF result.  ``result.wavefunctions`` must be a host array of the
        full-grid orbitals; an accelerated caller materializes them first.
    pair_poisson
        Open-boundary Poisson solver of a pair density, see
        :func:`parsec_python.V_xc.exact_exchange.exact_exchange_energy`.  The
        default solves with :func:`parsec_python.Hartree.solve_hartree` on
        the Laplacian of ``system``.
    pair_solver
        One line describing ``pair_poisson`` for the report.
    memory_probe
        Returns the resident high-water mark of the process in bytes, or
        ``None``; called before and after the evaluation for the report.
    memory_before
        The mark a caller took before it prepared the orbitals for this
        call (an accelerated run downloads them first); taken here otherwise.

    The nonlinear core correction, where the pseudopotentials have one,
    enters as in the SCF: the four density functionals see valence plus core
    density, the exact exchange the valence orbitals alone.

    A spin-polarized result (``result.spin``) is refined with the orbitals
    and densities of its two channels: the exact exchange of each channel
    (:func:`parsec_python.V_xc.exact_exchange.spin_exact_exchange_energy`)
    and the spin-polarized density functionals
    (:func:`parsec_python.V_xc.b3lyp.b3lyp_spin_semilocal_energies`), each
    channel with half of the core density.  Two identical channels give the
    unpolarized refinement.
    """

    functional = normalize_refinement(functional)
    if functional != "b3lyp":
        raise ValueError("refine_energy needs the functional 'b3lyp'")
    problem = system.input
    if getattr(problem, "periodic_cell", None) is not None:
        raise NotImplementedError(
            "Energy_Refinement is implemented for isolated systems "
            "(Boundary_Conditions: cluster): the exact exchange uses the "
            "open-boundary Poisson solver"
        )
    started = perf_counter()
    if memory_before is None:
        memory_before = memory_probe()
    grid = system.grid
    density = np.asarray(result.density, dtype=np.float64)
    core_density = np.asarray(result.core_density, dtype=np.float64)
    core_correction = bool(np.any(core_density != 0.0))
    # The two channels of a spin-polarized result, each with its orbitals,
    # one electron per state, and its density.
    channels = getattr(result, "spin", None)

    semilocal_started = perf_counter()
    semilocal: B3LYPSemilocalEnergies = (
        b3lyp_semilocal_energies(
            density, grid, core_density if core_correction else None
        )
        if channels is None
        else b3lyp_spin_semilocal_energies(
            np.asarray(channels[0].density, dtype=np.float64),
            np.asarray(channels[1].density, dtype=np.float64),
            grid,
            core_density if core_correction else None,
        )
    )
    semilocal_seconds = perf_counter() - semilocal_started

    # (orbitals, occupations in the convention of the exact exchange, the
    # density twice those occupations give).  A state of a spin channel
    # holds one electron, half of what the unpolarized convention counts.
    orbital_sets = (
        [(result.wavefunctions, result.occupations, density)]
        if channels is None
        else [
            (
                channel.wavefunctions,
                0.5 * np.asarray(channel.occupations, dtype=np.float64),
                np.asarray(channel.density, dtype=np.float64),
            )
            for channel in channels
        ]
    )
    for wavefunctions, occupations, orbital_density in orbital_sets:
        if (
            isinstance(wavefunctions, np.ndarray)
            and not np.iscomplexobj(wavefunctions)
            and wavefunctions.ndim == 2
            and wavefunctions.shape[0] == density.shape[0]
            and np.shape(occupations) == (wavefunctions.shape[1],)
        ):
            # Other shapes and types are refused by the exact exchange itself.
            deviation = orbital_density_deviation(
                wavefunctions, occupations, orbital_density, grid.volume_element
            )
            if not deviation <= ORBITAL_DENSITY_TOLERANCE:
                raise ValueError(
                    "the orbitals of the result do not give its density (largest "
                    f"deviation {deviation:.3e} of the largest density): the "
                    "refinement needs the orbitals and occupations of the last "
                    "SCF step"
                )

    if pair_poisson is None:
        pair_poisson = reference_pair_poisson(
            grid, system.negative_laplacian, problem.hartree
        )
        if pair_solver is None:
            pair_solver = "SciPy conjugate gradients on the full grid"
    if channels is None:
        exchange = exact_exchange_energy(
            result.wavefunctions,
            result.occupations,
            grid.volume_element,
            pair_poisson,
            progress=progress,
        )
    else:
        exchange = spin_exact_exchange_energy(
            [channel.wavefunctions for channel in channels],
            [channel.occupations for channel in channels],
            grid.volume_element,
            pair_poisson,
            progress=progress,
        )
    exchange_correlation = b3lyp_exchange_correlation_energy(
        exchange.energy, semilocal
    )
    energies = result.energies
    scf_xc = float(energies.exchange_correlation)
    scf_total = float(energies.total)
    return EnergyRefinement(
        functional=functional,
        scf_functional=str(problem.scf.xc_functional),
        exact_exchange=float(exchange.energy),
        slater_exchange=semilocal.slater_exchange,
        b88_exchange=semilocal.b88_exchange,
        vwn_correlation=semilocal.vwn_correlation,
        lyp_correlation=semilocal.lyp_correlation,
        exchange_correlation=exchange_correlation,
        scf_exchange_correlation=scf_xc,
        scf_total=scf_total,
        total=float(scf_total - scf_xc + exchange_correlation),
        atomic_reference_correction=float(
            getattr(result, "atomic_reference_correction", 0.0)
        ),
        exchange=exchange,
        core_correction=core_correction,
        scf_converged=bool(result.converged),
        pair_solver=pair_solver or "caller-supplied Poisson solver",
        semilocal_seconds=semilocal_seconds,
        seconds=perf_counter() - started,
        memory_before=memory_before,
        memory_after=memory_probe(),
        spin_polarized=channels is not None,
    )


def refinement_report_lines(
    refinement: EnergyRefinement,
    rydberg_to_ev: float,
) -> list[str]:
    """Format a refinement for ``parsec.out``.

    ``rydberg_to_ev`` is the display conversion of the reporter
    (:data:`parsec_python.Output.RYDBERG_TO_EV`).
    """

    name = refinement.functional.upper()
    scf_name = _SCF_FUNCTIONAL_NAMES.get(
        refinement.scf_functional, refinement.scf_functional.upper()
    )
    exchange = refinement.exchange
    title = (
        f" Energy refinement: {name} exchange-correlation energy of the "
        f"{scf_name} orbitals and density"
    )

    def part(label: str, value: float, weight: float) -> str:
        return f"   {label:<29s} = {value:20.8f} [Ry]   x {weight:4.2f}"

    lines = [
        "",
        title,
        " " + "-" * (len(title) - 1),
        " One non-self-consistent evaluation; the SCF result above is unchanged.",
        "",
        part("Exact (Hartree-Fock) exchange", refinement.exact_exchange, B3LYP_EXACT_EXCHANGE),
        part("Slater exchange", refinement.slater_exchange, B3LYP_SLATER_EXCHANGE),
        part("Becke 88 exchange", refinement.b88_exchange, B3LYP_B88_EXCHANGE),
        part("VWN (RPA) correlation", refinement.vwn_correlation, B3LYP_VWN_CORRELATION),
        part("Lee-Yang-Parr correlation", refinement.lyp_correlation, B3LYP_LYP_CORRELATION),
        "",
        f"   Exc ({scf_name}, self-consistent)".ljust(33)
        + f"= {refinement.scf_exchange_correlation:20.8f} [Ry]",
        f"   Exc ({name})".ljust(33)
        + f"= {refinement.exchange_correlation:20.8f} [Ry]",
        f"   Exc ({name}) - Exc ({scf_name})".ljust(33)
        + f"= {refinement.correction:20.8f} [Ry]",
        "",
        f"   Total Energy ({scf_name})".ljust(33)
        + f"= {refinement.scf_total:20.8f} [Ry]",
        f"   Refined Total Energy ({name})".ljust(33)
        + f"= {refinement.total:20.8f} [Ry]",
        f"   Refined Total Energy ({name})".ljust(33)
        + f"= {refinement.total * rydberg_to_ev:20.8f} [eV]",
    ]
    if refinement.atomic_reference_correction != 0.0:
        lines.extend(
            [
                "",
                (
                    "Atomic AE-minus-pseudo reference correction : "
                    f"{refinement.atomic_reference_correction:20.8f} [Ry]"
                    f"   (of the {scf_name} pseudopotentials)"
                ),
                (
                    f"{scf_name} all-electron total (as above)".ljust(44)
                    + f": {refinement.scf_all_electron_total:20.8f} [Ry]"
                ),
                (
                    f"{name}-refined all-electron total".ljust(44)
                    + f": {refinement.all_electron_total:20.8f} [Ry]"
                ),
                (
                    f"{name}-refined all-electron total".ljust(44)
                    + f": {refinement.all_electron_total * rydberg_to_ev:20.8f} [eV]"
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                f" Exact exchange: {len(exchange.orbitals)} occupied orbitals"
                f"{' of the two spin channels' if refinement.spin_polarized else ''}, "
                f"{exchange.pair_solves} pair-density Poisson solves"
                + (
                    f", {exchange.poisson_iterations} CG iterations"
                    if exchange.poisson_iterations
                    else ""
                )
            ),
            f" Pair-density Poisson solver: {refinement.pair_solver}",
        ]
    )
    if refinement.spin_polarized:
        lines.append(
            " Spin-polarized: the orbitals of each spin channel exchange among "
            "themselves, -1/2 sum_ij f_i f_j (ij|ij) per channel; the density "
            "functionals are those of the two spin densities."
        )
    if exchange.fractional and refinement.spin_polarized:
        lines.append(
            " NOTE: fractional occupations (largest distance from 0 or 1: "
            f"{exchange.largest_fractional_deviation:.3e}). The exact exchange "
            "is that of the ensemble density matrix of each spin channel; it "
            "is not the energy of one determinant."
        )
    elif exchange.fractional:
        lines.append(
            " NOTE: fractional occupations (largest distance from 0 or 1: "
            f"{exchange.largest_fractional_deviation:.3e}). The exact exchange "
            "is that of the spin-unpolarized ensemble density matrix, "
            "-sum_ij f_i f_j (ij|ij); it is not a spin-polarized open-shell energy."
        )
    if refinement.core_correction:
        lines.append(
            " NOTE: nonlinear core correction present. The density functionals "
            "see valence + core density as in the SCF, the exact exchange the "
            "valence orbitals; this combination has not been checked against ARES."
        )
    if not refinement.scf_converged:
        lines.append(
            " WARNING: the SCF did not converge; the refined energy belongs "
            "to orbitals that are not self-consistent."
        )
    lines.extend(
        [
            "",
            (
                f"Time for energy refinement [sec] : {refinement.seconds:10.2f},"
                f"    exact exchange = {exchange.seconds:10.2f},"
                f"    density functionals = {refinement.semilocal_seconds:8.2f}"
            ),
            (
                "Time per pair-density Poisson solve [sec] : "
                f"{exchange.poisson_seconds / max(exchange.pair_solves, 1):10.3f}"
            ),
        ]
    )
    if refinement.memory_before is not None and refinement.memory_after is not None:
        gibibyte = float(1 << 30)
        lines.append(
            "Process memory high water [GiB] : "
            f"{refinement.memory_after / gibibyte:8.3f} after the refinement, "
            f"{refinement.memory_before / gibibyte:8.3f} before it"
        )
    lines.append("")
    return lines


__all__ = [
    "REFINEMENT_FUNCTIONALS",
    "EnergyRefinement",
    "apply_refinement_option",
    "check_refinement_request",
    "normalize_refinement",
    "orbital_density_deviation",
    "process_memory_high_water",
    "refine_energy",
    "refinement_report_lines",
]
