"""``parsec.out`` lines of a self-consistent hybrid functional (Fock loop).

The loop is :func:`parsec_python.SCF.hybrid.run_hybrid_scf`; its outer
iterations reach the reporter as
:class:`parsec_python.models.ExchangeLoopIteration` records, the energies of
its iterations as :class:`parsec_python.models.HybridEnergyBreakdown`.  This
module formats them; it computes nothing.
"""

from __future__ import annotations

import numpy as np

from ..models import ExchangeLoopIteration, HybridEnergyBreakdown

#: Kernel names of the report.
_KERNELS = {
    "coulomb": "Coulomb 1/r",
    "erfc": "short-range erfc(w r)/r",
    "erf": "long-range erf(w r)/r",
}


def hybrid_correlation_lines(scf, *, periodic: bool = False) -> list[str]:
    """The functional lines under "Correlation data" for a hybrid.

    ``periodic``: a periodic cell, whose pair potentials are solved by FFT
    with the cell's exchange kernel (:mod:`parsec_python.V_xc.bloch_exchange`).
    """

    from ..SCF.hybrid import hybrid_definition
    from ..V_ion.ionic_potential import hybrid_pseudopotential_family

    definition = hybrid_definition(scf)
    settings = scf.exact_exchange
    kernel = " + ".join(
        f"{fraction / definition.fraction:g} " * (len(definition.terms) > 1)
        + _KERNELS.get(name, name)
        + ("" if omega is None else f", w = {omega:.4f} bohr^-1")
        for name, fraction, omega in definition.terms
    )
    return [
        f" Exchange-Correlation functional is {scf.xc_functional}",
        f" Hybrid (self-consistent, generalized Kohn-Sham) {definition.functional.describe()}",
        f"   exact exchange in the Hamiltonian: {definition.fraction:.4f} x ({kernel})",
        (
            f" Fock loop: start from a {definition.start_functional.upper()} SCF, adaptively "
            "compressed exchange (ACE) of the States_Num orbitals,"
        ),
        (
            f"   stop at an estimated exchange error below {settings.tolerance:.2E} Ry or after "
            f"{settings.max_iterations} outer iterations, {settings.batch_size} pair "
            f"densities per Poisson batch, pair solver "
            + ("FFT with the exchange kernel of the cell" if periodic else settings.pair_solver)
        ),
        (
            f"   inner criterion {settings.inner_criterion}, mixer history {settings.mixer_history}, "
            f"outer mixing {settings.outer_mixing}"
        ),
        f" Pseudopotentials: files of {hybrid_pseudopotential_family(scf.xc_functional)}",
    ] + (_libxc_lines(scf.xc_functional) if getattr(scf, "is_libxc", False) else []) + _meta_gga_lines(scf)


def _meta_gga_lines(scf) -> list[str]:
    """tau and the form of the density gradient of a meta-GGA hybrid's inner loops."""

    if not getattr(scf, "is_meta_gga", False):
        return []
    from ..V_xc.meta_gga import meta_gga_lines

    return meta_gga_lines(scf)[2:]


def _libxc_lines(functional: str) -> list[str]:
    from ..V_xc.libxc import libxc_citation_lines

    return libxc_citation_lines(functional)


def hybrid_energy_lines(energies) -> list[str]:
    """The exact-exchange lines of an iteration's energy block (nothing otherwise)."""

    if not isinstance(energies, HybridEnergyBreakdown):
        return []
    note = "   (linearized: fixed operator)" if energies.linearized else ""
    return [
        (
            "   Exc semilocal part            = "
            f"{energies.semilocal_exchange_correlation:20.8f} [Ry]"
        ),
        (
            "   Exact exchange a_x*E_x        = "
            f"{energies.exact_exchange:20.8f} [Ry]{note}"
        ),
        (
            "   Exact exchange in eigenvalues = "
            f"{energies.exact_exchange_band:20.8f} [Ry]"
        ),
    ]


def exchange_loop_lines(record: ExchangeLoopIteration) -> list[str]:
    """The block of one outer (Fock) iteration."""

    ranks = record.operator_ranks
    operator_line = (
        f"   ACE operator: {'/'.join(str(rank) for rank in ranks)} orbitals"
        + (" (up/down)" if len(ranks) == 2 else "")
        + f", max|M - M^T|/max|M| = {record.operator_asymmetry:.1E}"
        if ranks
        else None
    )
    work = (
        f"   Exact exchange: {record.pair_solves} pair-density Poisson solves, "
        f"{record.poisson_seconds:.2f} s of {record.exchange_seconds:.2f} s "
        f"({record.pair_solver})"
    )
    if record.iteration == 0:
        criterion = getattr(record, "inner_criterion", None)
        source = (
            (
                f"the {record.start_functional.upper()} start orbitals"
                if criterion is not None
                else f"the converged {record.start_functional.upper()} orbitals"
            )
            if record.start_functional
            else "the orbitals of the structure before the move"
        )
        lines = [
            "",
            f" Fock loop ({record.functional}): exchange operator of {source}",
            " " + "-" * 66,
            f"   E_x (start orbitals)          = {record.exchange_energy:20.8f} [Ry]",
        ]
        if record.total_energy is not None:
            lines.append(
                f"   Total Energy ({record.functional}) of them   = {record.total_energy:20.8f} [Ry]"
            )
        if not record.inner_converged:
            lines.append(
                f"   WARNING: the {record.start_functional.upper() if record.start_functional else 'start'} "
                f"SCF did not converge in {record.inner_iterations} iterations; the Fock loop starts "
                "from its last orbitals"
            )
        if criterion is not None:
            lines.append(
                f"   The start stopped below a residual of {criterion:.2E} Ry (adaptive criterion): its energy "
                "is that of a start, not a converged semilocal energy"
            )
        if getattr(record, "outer_note", ""):
            lines.append(f"   Start: {record.outer_note}")
        lines.extend([line for line in (operator_line, work) if line is not None])
        lines.append("")
        return lines
    status = (
        "converged"
        if record.converged
        else ("inner SCF not converged" if not record.inner_converged else "continue")
    )
    lines = [
        "",
        (
            f" Fock loop iteration {record.iteration:3d} ({record.functional}): "
            f"inner SCF of {record.inner_iterations} iterations"
            + ("" if record.inner_converged else " (NOT converged)")
        ),
        " " + "-" * 66,
        f"   E_x (new orbitals)            = {record.exchange_energy:20.8f} [Ry]",
        f"   E_x (new with old operator)   = {record.cross_exchange_energy:20.8f} [Ry]",
        (
            "   Change of E_x                 = "
            f"{record.exchange_energy - record.previous_exchange_energy:20.8f} [Ry]"
        ),
        (
            "   Estimated exchange error      = "
            f"{record.estimated_error:20.3E} [Ry]   (tolerance {record.tolerance:.1E}): {status}"
        ),
        f"   Total Energy ({record.functional})".ljust(34) + f"= {record.total_energy:20.8f} [Ry]",
    ]
    lines.extend(inner_loop_control_lines(record))
    lines.extend([line for line in (operator_line, work) if line is not None])
    lines.append("")
    return lines


def inner_loop_control_lines(record: ExchangeLoopIteration) -> list[str]:
    """How the inner loop of an outer iteration was run (``Exx_Inner_Criterion``, ``Exx_Mixer_History``)."""

    parts = []
    criterion = getattr(record, "inner_criterion", None)
    if criterion is not None:
        parts.append(f"stopped below a residual of {criterion:.2E} Ry (adaptive criterion)")
    seeds = int(getattr(record, "mixer_seeds", 0) or 0)
    if seeds:
        parts.append(f"{seeds} secant pairs carried into its mixer")
    passes = int(getattr(record, "level_passes", 0) or 0)
    if passes:
        parts.append(
            f"levels converged by {passes} eigensolver passes at its last Hamiltonian "
            f"(last move {float(getattr(record, 'level_change', 0.0)):.1E} Ry)"
        )
    lines = [f"   Inner SCF: {'; '.join(parts)}"] if parts else []
    if getattr(record, "extrapolated", False):
        lines.append("   Outer step: the next operator is that of the DIIS-extrapolated density matrix")
    elif getattr(record, "outer_note", ""):
        lines.append(f"   Outer step: {record.outer_note}")
    return lines


def hybrid_finish_lines(result) -> list[str]:
    """The summary of the Fock loop after the SCF (nothing without one)."""

    summary = getattr(result, "exact_exchange", None)
    if summary is None:
        return []
    energies = result.energies
    outer = max((record.iteration for record in summary.iterations), default=0)
    status = (
        f"Fock loop converged after {outer} outer iterations "
        f"(estimated exchange error {summary.estimated_error:.2E} Ry < {summary.tolerance:.1E} Ry)."
        if summary.converged
        else (
            f"Fock loop NOT converged after {outer} outer iterations "
            f"(estimated exchange error {summary.estimated_error:.2E} Ry, tolerance "
            f"{summary.tolerance:.1E} Ry, or an inner SCF that did not converge)."
        )
    )
    lines = [
        "",
        f" Hybrid functional {summary.display_name}: generalized Kohn-Sham energies",
        " " + "-" * 66,
        f"   Eigenvalue Energy             = {energies.eigenvalue:20.8f} [Ry]",
        f"   Hartree Energy                = {energies.hartree:20.8f} [Ry]",
        f"   Exc semilocal part            = {energies.semilocal_exchange_correlation:20.8f} [Ry]",
        f"   Exact exchange E_x            = {summary.exchange_energy:20.8f} [Ry]",
        f"   Exact exchange a_x*E_x        = {energies.exact_exchange:20.8f} [Ry]",
        f"   Exact exchange in eigenvalues = {energies.exact_exchange_band:20.8f} [Ry]",
        f"   Exc = semilocal + a_x*E_x     = {energies.exchange_correlation:20.8f} [Ry]",
        f"   Electron-Ion energy           = {energies.electron_ion:20.8f} [Ry]",
        f"   Ion-Ion Energy                = {energies.ion_ion:20.8f} [Ry]",
        "",
        f"   Total Energy ({summary.display_name})".ljust(30) + f" = {energies.total:22.8f} [Ry]",
        "",
        " " + status,
        (
            f" Exact exchange: {summary.pair_solves} pair-density Poisson solves in "
            f"{summary.exchange_seconds:.2f} s ({summary.poisson_seconds:.2f} s in the solver); "
            f"Fock loop {summary.seconds:.2f} s"
            + (
                f", {summary.start_iterations} SCF iterations of the "
                f"{summary.start_functional.upper()} start"
                if summary.start_functional
                else ""
            )
        ),
        f" Pair-density Poisson solver: {summary.pair_solver}",
    ]
    lines.extend(band_edge_lines(result))
    lines.extend(fock_loop_warning_lines(summary))
    lines.append("")
    return lines


def _level_blocks(result):
    """``(eigenvalues, occupations, k index, channel)`` of every k-point (and channel) of a periodic result."""

    k_points = getattr(result, "k_points", None)
    count = 1 if k_points is None else len(k_points)
    if hasattr(result, "eigenvalues_up"):
        channels = (
            ("up", result.eigenvalues_up, result.occupations_up),
            ("down", result.eigenvalues_down, result.occupations_down),
        )
    elif getattr(result, "spin", None) is not None:
        channels = tuple(
            (name, channel.eigenvalues, channel.occupations) for name, channel in zip(("up", "down"), result.spin)
        )
    else:
        channels = ((None, result.eigenvalues, result.occupations),)
    for name, eigenvalues, occupations in channels:
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        occupations = np.asarray(occupations, dtype=np.float64)
        per_k = eigenvalues.size // count
        for index in range(count):
            block = slice(index * per_k, (index + 1) * per_k)
            yield eigenvalues[block], occupations[block], index, name


def band_edge_lines(result) -> list[str]:
    """The band edges of a periodic hybrid result: VBM, CBM and the gaps (nothing for a cluster).

    A state with occupation above one half counts as occupied.  The k-points
    are those of the loop (the time-reversal representatives; the band edges
    of the full mesh are the same).
    """

    grid = getattr(result, "grid", None)
    if getattr(result, "k_points", None) is None and getattr(grid, "cell", None) is None:
        return []
    from . import RYDBERG_TO_EV

    highest = lowest = None
    direct = None
    for eigenvalues, occupations, index, channel in _level_blocks(result):
        occupied = eigenvalues[occupations > 0.5]
        empty = eigenvalues[occupations <= 0.5]
        if occupied.size and (highest is None or occupied.max() > highest[0]):
            highest = (float(occupied.max()), index, channel)
        if empty.size and (lowest is None or empty.min() < lowest[0]):
            lowest = (float(empty.min()), index, channel)
        if occupied.size and empty.size:
            gap = float(empty.min() - occupied.max())
            if direct is None or gap < direct[0]:
                direct = (gap, index, channel)
    if highest is None or lowest is None:
        return []

    def where(index, channel) -> str:
        return f"k-point {index + 1}" + ("" if channel is None else f", spin {channel}")

    gap = lowest[0] - highest[0]
    lines = [
        f" Valence band maximum          = {highest[0]:20.8f} [Ry] = {highest[0] * RYDBERG_TO_EV:12.6f} [eV] ({where(*highest[1:])})",
        f" Conduction band minimum       = {lowest[0]:20.8f} [Ry] = {lowest[0] * RYDBERG_TO_EV:12.6f} [eV] ({where(*lowest[1:])})",
        f" Band gap ({'direct' if highest[1] == lowest[1] else 'indirect'})".ljust(31)
        + f"= {gap:20.8f} [Ry] = {gap * RYDBERG_TO_EV:12.6f} [eV]",
    ]
    if direct is not None and highest[1] != lowest[1]:
        lines.append(
            " Smallest direct gap           = "
            f"{direct[0]:20.8f} [Ry] = {direct[0] * RYDBERG_TO_EV:12.6f} [eV] ({where(*direct[1:])})"
        )
    return lines


def fock_loop_warning_lines(summary) -> list[str]:
    """What the diagnostics of a Fock loop (:class:`parsec_python.SCF.hybrid.FockLoopSummary`) warn of."""

    lines = []
    instability = getattr(summary, "instability", None)
    if instability is not None:
        lines.append(
            f" NOTE: the estimated exchange error rose at outer iteration {instability}: the state the loop "
            "approached was unstable under exact exchange (for example a symmetric ensemble that breaks its "
            "symmetry); the loop went on to a lower one."
        )
    ratio = getattr(summary, "convergence_ratio", None)
    remaining = getattr(summary, "remaining_energy", None)
    if not summary.converged and ratio is not None and remaining is not None and ratio > 0.5:
        lines.append(
            f" WARNING: the Fock loop converges slowly: its energy steps shrink by {ratio:.2f} per outer "
            f"iteration, about {remaining:.2E} Ry is still to be gained (a soft mode of the hybrid state); "
            "raise Exx_Max_Iter."
        )
    fractional = float(getattr(summary, "fractional_occupation", 0.0))
    if fractional > 0.01:
        lines.append(
            f" WARNING: fractional occupations in the final state (largest min(f, 1 - f) = {fractional:.3f}). "
            "Exact exchange penalizes them: a symmetric ensemble of degenerate levels can be a saddle "
            "point of the hybrid functional, with a symmetry-broken state of lower energy (compare a "
            "tighter Exx_Tolerance, integer occupations of occup.in, or another start)."
        )
    return lines


__all__ = [
    "exchange_loop_lines",
    "fock_loop_warning_lines",
    "band_edge_lines",
    "hybrid_correlation_lines",
    "hybrid_energy_lines",
    "hybrid_finish_lines",
]
