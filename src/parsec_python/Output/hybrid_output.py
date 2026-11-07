"""``parsec.out`` lines of a self-consistent hybrid functional (Fock loop).

The loop is :func:`parsec_python.SCF.hybrid.run_hybrid_scf`; its outer
iterations reach the reporter as
:class:`parsec_python.models.ExchangeLoopIteration` records, the energies of
its iterations as :class:`parsec_python.models.HybridEnergyBreakdown`.  This
module formats them; it computes nothing.
"""

from __future__ import annotations

from ..models import ExchangeLoopIteration, HybridEnergyBreakdown

#: Kernel names of the report.
_KERNELS = {
    "coulomb": "Coulomb 1/r",
    "erfc": "short-range erfc(w r)/r",
    "erf": "long-range erf(w r)/r",
}


def hybrid_correlation_lines(scf) -> list[str]:
    """The functional lines under "Correlation data" for a hybrid."""

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
            f"densities per Poisson batch, pair solver {settings.pair_solver}"
        ),
        f" Pseudopotentials: files of {hybrid_pseudopotential_family(scf.xc_functional)}",
    ]


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
        source = (
            f"the converged {record.start_functional.upper()} orbitals"
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
    lines.extend([line for line in (operator_line, work) if line is not None])
    lines.append("")
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
    lines.extend(fock_loop_warning_lines(summary))
    lines.append("")
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
    "hybrid_correlation_lines",
    "hybrid_energy_lines",
    "hybrid_finish_lines",
]
