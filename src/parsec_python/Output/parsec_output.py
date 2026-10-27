"""PARSEC-like text reporting for the Python single-point CLI.

The formatter intentionally omits data Python does not calculate, including
point-group representation numbers, forces, dipoles, and MPI statistics.
"""

from __future__ import annotations

from datetime import datetime
import math
import os
from time import perf_counter
from types import SimpleNamespace
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from ..Hartree.domain import (
    ANGSTROM,
    DOMAIN_ENERGY_TOLERANCE,
    FIT_WINDOW,
    MINIMUM_VACUUM,
    NO_TABLE_VACUUM,
    RADIUS_STEP_LIMIT,
    RULE_VERSION,
    SHELL_WIDTH,
    WALL_SHARE,
    auto_boundary_tolerance,
    boundary_energy,
    default_domain,
    fit_wall,
    free_atom_tables,
    radius_for_budget,
    sphere_shell_sums,
    wall_energy,
)
from .force_output import force_report_lines
from ..models import MAXIMUM_MULTIPOLE_ORDER, SCFIteration, SinglePointResult
from ..V_ion import center_cluster_geometry, ionic_charge

if TYPE_CHECKING:
    from ..Eigensolvers.perturbative_soc import PerturbativeSpinOrbitResult
    from ..Input.parsec_input import ParsecInputTranslation
    from ..models import (
        SelfConsistentSOCIteration,
        SelfConsistentSOCResult,
        SpinPolarizedSCFIteration,
        SpinPolarizedSinglePointResult,
    )
    from ..SCF import PreparedSinglePointSystem


# Match the display conversion used by this PARSEC build. Internal Python
# physics remains in Rydberg and does not depend on this reporting constant.
RYDBERG_TO_EV = 13.6058


def hartree_boundary_lines(system, input_settings=None) -> list[str]:
    """Say which Hartree boundary values a prepared system uses.

    A system prepared without a plan of the boundary has PARSEC's multipole
    expansion at ``Solver_Lpole`` and gets no lines.  ``input_settings`` are
    the Hartree settings the header of the report printed: where the system
    was prepared with others, as under the ``PARSEC_HARTREE_*`` switches of
    the accelerated driver, a line names each one that was replaced.
    """

    plan = getattr(system, "hartree_boundary", None)
    if plan is None:
        return []
    tolerance = "off" if plan.tolerance is None else f"{plan.tolerance:.3E} Ry"
    lines = [
        (
            f" Hartree boundary multipole order = {plan.order:3d}"
            f"   (Solver_Lpole = {plan.minimum_order:d}, tolerance = {tolerance})"
        ),
    ]
    if plan.estimates is None:
        lines.append(f" Hartree boundary estimate: none ({plan.status})")
    else:
        lines.append(
            " Hartree boundary estimate [Ry] = "
            f"{plan.estimate_minimum:.3E} at order {plan.minimum_order:d}, "
            f"{plan.estimate_order:.3E} at order {plan.order:d} ({plan.status})"
        )
        if plan.cap_reached:
            lines.append(
                " WARNING: the Hartree boundary misses its tolerance at the "
                "largest multipole order; the domain leaves too little vacuum"
            )
    if plan.atomic_tail:
        maximum = getattr(system, "hartree_boundary_tail_maximum", None)
        lines.append(
            " Hartree boundary atomic tail: applied"
            + ("" if maximum is None else f", max |tail| = {maximum:.3E} Ry")
        )
    else:
        lines.append(" Hartree boundary atomic tail: not applied")
    lines.append(
        " Hartree boundary values are PARSEC's multipole expansion"
        if plan.legacy
        else " Hartree boundary values differ from PARSEC's; "
        "Hartree_Boundary_Tolerance: off restores them"
    )
    used = getattr(getattr(system, "input", None), "hartree", None)
    if input_settings is not None and used is not None:

        def tolerance_text(value):
            return "off" if value is None else f"{value:.3E} Ry"

        replaced = [
            f"{name} {before} -> {after}"
            for name, before, after in (
                ("Solver_Lpole", input_settings.multipole_order, plan.minimum_order),
                (
                    "tolerance",
                    tolerance_text(input_settings.boundary_tolerance),
                    tolerance_text(plan.tolerance),
                ),
                ("atomic tail", input_settings.atomic_tail, used.atomic_tail),
            )
            if before != after
        ]
        if replaced:
            lines.append(
                " Hartree boundary settings of this run replace those of the "
                "input (PARSEC_HARTREE_* switches): " + ", ".join(replaced)
            )
    return lines


def domain_report_requested() -> bool:
    """Whether the reporter estimates what a sphere adds to the energy.

    ``PARSEC_DOMAIN_REPORT`` is on by default.  ``0`` leaves out what an
    input that gives its radius gained with the default rule: the estimates
    at set-up, the block after the SCF with its shell sums and its search
    for a radius, the record of the domain and what a dry run says of the
    rule.  Its ``parsec.out`` and the details of its result are then what
    they were before.  A radius the rule chose keeps its lines and its
    record and loses the block after the SCF, which is the part that costs.
    Both drivers read the switch, in their reporter.
    """

    return os.environ.get("PARSEC_DOMAIN_REPORT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _species_shares(by_species) -> str:
    """Whole per cent of an estimate by species, those that round to none left out."""

    total = sum(value for _symbol, value in by_species)
    if not total > 0.0:
        return "none"
    shares = [
        (symbol, round(100.0 * value / total)) for symbol, value in by_species
    ]
    return ", ".join(f"{symbol} {share:d} %" for symbol, share in shares if share)


def _species_left_out(symbols) -> str:
    """Name the species an estimate from the free atoms does not hold."""

    symbols = sorted(symbols)
    if not symbols:
        return ""
    return (
        f"; without {', '.join(symbols)}, which "
        f"{'has' if len(symbols) == 1 else 'have'} no complete free-atom density"
    )


def _domain_input_lines(radius_text, tolerance_text) -> list[str]:
    """The lines of an input that give a domain, as they are typed there."""

    lines = [f"       Boundary_Sphere_Radius: {radius_text}"]
    if tolerance_text is not None:
        lines.append(f"       Hartree_Boundary_Tolerance: {tolerance_text}")
    return lines


def domain_header_lines(translation) -> list[str]:
    """Say what the default rule chose for the sphere, from the input alone.

    No lines for an input that gives its radius, or a box.  The numbers are
    those of the parser: the order of the Hartree boundary and what the
    ``PARSEC_HARTREE_*`` switches make of the tolerance are known only once
    the system is prepared, and :func:`domain_setup_lines` reports them.
    """

    choice = getattr(translation, "domain", None)
    if choice is None:
        return []
    lines = [
        (
            f" --- Radius {choice.radius_text} chosen by the default rule "
            f"{choice.rule_version:d} (Domain_Energy_Tolerance = "
            f"{choice.budget:.1E} Ry)"
        ),
        (
            f" --- Outermost atom: {choice.outermost_symbol} at "
            f"{choice.outermost_distance / ANGSTROM:.3f} ang from the centre; "
            f"vacuum beyond it {choice.vacuum / ANGSTROM:.3f} ang"
        ),
    ]
    if choice.extra_electrons > 1.0e-6:
        lines.append(
            " --- Energy the sphere adds: no estimate before the SCF, the "
            f"system holds {choice.extra_electrons:g} electron(s) more than "
            "its free atoms"
        )
    elif not choice.tables:
        lines.append(
            " --- Energy the sphere adds: no estimate before the SCF, no "
            "species has a complete free-atom density"
        )
    else:
        lines.append(
            " --- Energy the sphere adds, estimated from the free atoms: "
            f"{choice.wall:.1E} Ry (aim {choice.wall_share:.1E} Ry; an "
            "estimate, not a bound"
            + _species_left_out(symbol for symbol, _reason in choice.without_table)
            + ")"
        )
    if choice.tables:
        lines.append(
            " --- Free-atom densities: "
            + ", ".join(
                f"{symbol} {source} {charge:g} e"
                for symbol, source, charge in choice.tables
            )
            + "; share of the estimate: "
            + _species_shares(choice.wall_by_species)
        )
    for symbol, reason in choice.without_table:
        lines.append(
            f" --- NOTE: {symbol} has {reason}: its atoms get "
            f"{NO_TABLE_VACUUM:g} ang of vacuum, which is a guess"
        )
    lines.extend(
        [
            f" --- Radius set by: {choice.set_by}",
            (
                " --- The rule covers the total energy and the occupied "
                "levels; empty levels need more vacuum."
            ),
            (
                " --- These two lines reproduce this domain bit for bit "
                "(with the same PARSEC_HARTREE_* switches):"
                if choice.tolerance_text is not None
                else " --- This line, beside the others of the input, "
                "reproduces this domain bit for bit (with the same "
                "PARSEC_HARTREE_* switches):"
            ),
            *_domain_input_lines(choice.radius_text, choice.tolerance_text),
        ]
    )
    return lines


def domain_setup_lines(system, translation) -> tuple[list[str], dict | None]:
    """Estimate what the sphere and its boundary values add, before the SCF.

    Returns the lines of the set-up block and the record of the domain that
    the timing files keep; no lines and ``None`` for a box.  For a radius of
    the input the wall estimate of its sphere is made here, from the
    pseudopotentials the system has loaded; for a radius of the rule the
    parser made it.  The energy the boundary values leave is read from the
    plan in use, so it shows what the ``PARSEC_HARTREE_*`` switches did.
    With ``Output_Level`` 2 and above an input with a radius is also told
    what the rule would give.
    """

    grid = getattr(getattr(system, "input", None), "grid", None)
    if grid is None or grid.domain_shape != "sphere":
        return [], None
    budget = float(
        getattr(translation, "domain_energy_tolerance", DOMAIN_ENERGY_TOLERANCE)
    )
    share = WALL_SHARE * budget
    choice = getattr(translation, "domain", None)
    tolerance_from = getattr(translation, "boundary_tolerance_from", "default")
    positions = np.array(
        [atom.position for atom in system.atoms], dtype=np.float64
    ).reshape(-1, 3)
    symbols = [atom.symbol for atom in system.atoms]
    distance = np.linalg.norm(positions, axis=1)
    outermost = int(np.argmax(distance))
    vacuum = float(grid.radius - distance[outermost])
    electrons = float(system.electron_count)
    record: dict[str, object] = dict(
        rule_version=RULE_VERSION,
        domain_energy_tolerance_ry=budget,
        radius_from="input" if choice is None else "default rule",
        radius_bohr=float(grid.radius),
        boundary_sphere_radius=None if choice is None else choice.radius_text,
        hartree_boundary_tolerance=(
            None if choice is None else choice.tolerance_text
        ),
        hartree_boundary_tolerance_from=tolerance_from,
        set_by="input" if choice is None else choice.set_by,
        outermost_atom=symbols[outermost],
        outermost_distance_bohr=float(distance[outermost]),
        vacuum_bohr=vacuum,
        vacuum_angstrom=vacuum / ANGSTROM,
        electrons=electrons,
        rule_seconds=float(getattr(translation, "domain_rule_seconds", 0.0)),
    )
    lines: list[str] = []
    if choice is None:
        started = perf_counter()
        tables, reasons = free_atom_tables(
            {symbol: system.pseudopotentials[symbol] for symbol in sorted(set(symbols))}
        )
        wall, by_species = wall_energy(positions, symbols, tables, grid.radius)
        extra = electrons - sum(
            tables[symbol].charge
            if symbol in tables
            else float(system.pseudopotentials[symbol].ionic_charge)
            for symbol in symbols
        )
        record.update(
            wall_estimate_ry=wall,
            wall_estimate_by_species=dict(sorted(by_species.items())),
            free_atom_tables={
                symbol: dict(source=table.source, electrons=table.charge)
                for symbol, table in sorted(tables.items())
            },
            species_without_table=dict(sorted(reasons.items())),
            extra_electrons=extra,
            wall_estimate_seconds=perf_counter() - started,
        )
        lines.append(
            " --- Sphere of the input: outermost atom "
            f"{symbols[outermost]} at {distance[outermost] / ANGSTROM:.3f} ang "
            f"from the centre; vacuum beyond it {vacuum / ANGSTROM:.3f} ang"
        )
        if extra > 1.0e-6:
            lines.append(
                " --- Energy the sphere adds: no estimate before the SCF, the "
                f"system holds {extra:g} electron(s) more than its free atoms"
            )
        elif not tables:
            lines.append(
                " --- Energy the sphere adds: no estimate before the SCF, no "
                "species has a complete free-atom density"
            )
        else:
            lines.append(
                " --- Energy the sphere adds, estimated from the free atoms: "
                f"{wall:.1E} Ry (wall share of Domain_Energy_Tolerance "
                f"{share:.1E} Ry; an estimate, not a bound"
                f"{_species_left_out(reasons)})"
            )
            if wall > share:
                # A radius taken from the block after an earlier SCF comes
                # here: the free atoms are no verdict on it.
                lines.append(
                    " NOTE: the free-atom estimate exceeds the wall share. It "
                    "is several times too large for some surfaces "
                    "(hydrogen-terminated carbon): the estimate from the "
                    "density after the SCF decides"
                )
    else:
        record.update(
            wall_estimate_ry=choice.wall,
            wall_estimate_by_species=dict(choice.wall_by_species),
            free_atom_tables={
                symbol: dict(source=source, electrons=charge)
                for symbol, source, charge in choice.tables
            },
            species_without_table=dict(choice.without_table),
            extra_electrons=choice.extra_electrons,
            rule_estimates=choice.estimates,
        )

    plan = getattr(system, "hartree_boundary", None)
    energy = boundary_energy(plan, electrons, vacuum)
    record["boundary"] = dict(
        about_ry=energy.about, bound_ry=energy.bound, plan=energy.reason
    )
    aim = f"(aim {budget - share:.1E} Ry)"
    if energy.exact:
        lines.append(" --- Energy left by the boundary values: none, they are exact")
    elif not energy.calibrated:
        # Outside what was calibrated: no number is claimed.
        lines.append(
            " --- Energy left by the boundary values: no estimate "
            f"({energy.reason})"
        )
        lines.append(
            " WARNING: the boundary values of this run are not the ones the "
            "default radius was chosen for; Domain_Energy_Tolerance covers "
            "the wall only"
            if choice is not None
            else " NOTE: the boundary values of this run are outside the "
            "calibration of Domain_Energy_Tolerance, which then covers the "
            "wall only"
        )
    else:
        tolerance = f"tolerance {plan.tolerance:.1E} Ry"
        if energy.about is None:
            lines.append(
                f" --- Energy left by the boundary values at order {plan.order:d} "
                "(PARSEC's values, their estimate is within a tenth of the "
                f"{tolerance}): calibrated bound {energy.bound:.1E} Ry {aim}"
            )
            lines.append(
                "       calibration: hydrogen-terminated clusters, 176 to "
                "3,480 electrons"
            )
        else:
            lines.append(
                f" --- Energy left by the boundary values at order {plan.order:d} "
                f"({tolerance}, atomic tail): about {energy.about:.1E} Ry, "
                f"calibrated bound {energy.bound:.1E} Ry {aim}"
            )
            lines.append(
                "       calibration: hydrogen-terminated clusters, 176 to "
                "23,768 electrons"
            )
        if energy.bound > budget - share:
            # The tolerance of the rule keeps the bound within the share
            # wherever some order up to the largest meets it.
            tolerance = auto_boundary_tolerance(electrons, budget, vacuum)
            lines.append(
                " NOTE: the calibrated bound exceeds the boundary share of "
                "Domain_Energy_Tolerance; "
                + (
                    f"Hartree_Boundary_Tolerance: {tolerance:.1e} Ry would "
                    "meet it"
                    if plan.estimates[MAXIMUM_MULTIPOLE_ORDER] <= tolerance
                    else "no multipole order meets it at this radius"
                )
            )
    if choice is None and getattr(translation, "output_level", 1) >= 2:
        lines.extend(rule_would_give_lines(system, translation))
    return lines, record


def rule_would_give_lines(system, translation) -> list[str]:
    """Say what the default rule gives for an input that has a radius.

    The rule is run on the atoms and pseudopotentials of ``system``, a
    prepared one or any object that has ``atoms``, ``pseudopotentials`` and
    ``electron_count``, with the settings of the input and its tolerance
    where it has a line for one.  Where the rule has no radius for them, as
    with a tolerance that no multipole order meets, the line says why: the
    input has its radius and runs with it.
    """

    problem = translation.problem
    hartree = problem.hartree
    multipole = hartree.boundary_method != "direct"
    given = getattr(translation, "boundary_tolerance_from", "default") == "input"
    if multipole and (
        hartree.boundary_tolerance is None or hartree.atomic_tail == "off"
    ):
        return [
            " --- The default rule chooses no radius with "
            "Hartree_Boundary_Tolerance: off or Hartree_Atomic_Tail: off"
        ]
    try:
        choice = default_domain(
            system.atoms,
            system.pseudopotentials,
            system.electron_count,
            spacing=problem.grid.spacing,
            shift=problem.grid.shift[0],
            stencil_half_width=problem.grid.stencil_half_width,
            budget=getattr(
                translation, "domain_energy_tolerance", DOMAIN_ENERGY_TOLERANCE
            ),
            boundary_tolerance=hartree.boundary_tolerance if given else None,
            multipole_boundary=multipole,
        )
    except ValueError as error:
        return [
            " --- Without Boundary_Sphere_Radius the default rule "
            f"{RULE_VERSION:d} would give no radius: {error}"
        ]
    wall = (
        f"; wall estimate {choice.wall:.1E} Ry"
        if choice.tables and not choice.extra_electrons > 1.0e-6
        else ""
    )
    return [
        (
            " --- Without Boundary_Sphere_Radius the default rule "
            f"{choice.rule_version:d} would give (set by {choice.set_by}{wall}):"
        ),
        *_domain_input_lines(choice.radius_text, choice.tolerance_text),
    ]


def contained_domain_report(
    place: str, report: Callable[..., tuple[list[str], dict | None]], *arguments
) -> tuple[list[str], dict | None]:
    """Run a report of the sphere so that it cannot stop a calculation.

    The reports estimate and advise; no result depends on them.  Whatever
    fails in one is printed after ``place``, where its lines would stand,
    and kept in the record, instead of stopping an input that can run or
    losing a calculation that has finished.
    """

    try:
        return report(*arguments)
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        return (
            [f"{place}, the report failed ({reason})"],
            dict(status="failed", failure=reason),
        )


def domain_dry_run_lines(translation, pseudopotentials) -> list[str]:
    """Say of the sphere what a dry run can: no grid is built for it.

    The choice of the default rule where the input left the radius to it,
    and for an input with a radius what the rule would give, from the
    loaded ``pseudopotentials`` of its species; nothing for such an input
    under ``PARSEC_DOMAIN_REPORT=0``.
    """

    problem = translation.problem
    if problem.grid.domain_shape != "sphere":
        return []
    if getattr(translation, "domain", None) is not None:
        return domain_header_lines(translation)
    if not domain_report_requested():
        return []

    def would_give():
        atoms = (
            center_cluster_geometry(problem.atoms)
            if problem.recenter_geometry
            else tuple(problem.atoms)
        )
        electrons = ionic_charge(atoms, pseudopotentials) - problem.scf.net_charge
        if electrons <= 0:
            return [], None
        return (
            rule_would_give_lines(
                SimpleNamespace(
                    input=problem,
                    atoms=atoms,
                    pseudopotentials=pseudopotentials,
                    electron_count=electrons,
                ),
                translation,
            ),
            None,
        )

    return contained_domain_report(
        " --- What the default rule would give without Boundary_Sphere_Radius: "
        "no answer",
        would_give,
    )[0]


def _radius_comment(wanted) -> str:
    """Say behind the radius after the SCF what, other than the density, set it."""

    parts = []
    if wanted.limit == "at least":
        parts.append(f"at least: the step is limited to {RADIUS_STEP_LIMIT:g} ang")
    if wanted.set_by.startswith("multipole order"):
        parts.append(
            f"order {MAXIMUM_MULTIPOLE_ORDER:d} misses the tolerance below this "
            f"radius; the wall alone would allow {wanted.wall_radius_text}"
        )
    elif wanted.set_by.startswith("minimum vacuum"):
        parts.append(f"the minimum vacuum of {MINIMUM_VACUUM:g} ang")
    elif wanted.set_by.startswith("stencil"):
        parts.append("the stencil of the grid needs this much")
    elif wanted.limit == "limit":
        parts.append(
            f"the step is limited to {RADIUS_STEP_LIMIT:g} ang; a smaller "
            "sphere may do"
        )
    # "+0.05 ang: a grid point lay on the sphere", where the radius moved.
    moved = wanted.set_by.partition(" (")[2].rstrip(")")
    if moved:
        parts.append(moved)
    return "   # " + "; ".join(parts) if parts else ""


def domain_finish_lines(
    result, translation, hartree=None, prepared=None
) -> tuple[list[str], dict | None]:
    """Estimate from the final density what the sphere adds to the energy.

    Returns the lines after the SCF and what the timing files keep of them;
    no lines and ``None`` for a box.  The density of a run that did not
    converge gives no estimate: one stopped after one to three steps was up
    to twice too small.  ``hartree`` are the Hartree settings of the input,
    which say whether the radius that would meet the wall share comes with a
    tolerance of the rule.  The shells are summed on the host arrays of the
    result, in blocks.

    The radius is the one the wall asks for.  With a tolerance of the rule
    it is raised where the largest multipole order would miss that tolerance,
    which takes estimates of the omitted potential (several for clusters of
    20,000 electrons and more): the line then says so, and the record keeps
    their number and the seconds of the shells and of the radius apart.
    ``prepared`` may hold such estimates, made for the atoms of the result
    before this is called
    (:func:`~parsec_python.Hartree.domain.prepare_radius_search`): the lines
    and the record are the same with and without them, apart from the
    seconds of the radius, which then count a search that had its estimates.
    """

    settings = getattr(getattr(result, "grid", None), "settings", None)
    if settings is None or settings.domain_shape != "sphere":
        return [], None
    if not result.converged:
        return (
            [" Density at the sphere: no estimate, the SCF did not converge"],
            dict(status="not converged"),
        )
    started = perf_counter()
    budget = float(
        getattr(translation, "domain_energy_tolerance", DOMAIN_ENERGY_TOLERANCE)
    )
    share = WALL_SHARE * budget
    sums = sphere_shell_sums(
        result.grid.coordinates,
        result.density,
        settings.radius,
        result.grid.volume_element,
    )
    record: dict[str, object] = dict(
        status="no fit",
        shell_width_bohr=SHELL_WIDTH,
        fit_window_bohr=list(FIT_WINDOW),
        shell_charge_per_bohr=[float(value) for value in sums],
    )
    fit = fit_wall(sums)
    record["shell_seconds"] = perf_counter() - started
    if fit is None:
        record["seconds"] = perf_counter() - started
        return (
            [
                " Density at the sphere: no estimate, fewer than three shells "
                "inside the sphere hold charge"
            ],
            record,
        )
    eigenvalues = np.asarray(result.eigenvalues, dtype=np.float64)
    occupations = np.asarray(result.occupations, dtype=np.float64)
    occupied = occupations > 0.5 * occupations.max()
    highest = float(eigenvalues[occupied].max())
    lowest_empty = float(eigenvalues[~occupied].min()) if np.any(~occupied) else None
    multipole = (
        hartree is None
        or hartree.boundary_method != "direct"
        and hartree.boundary_tolerance is not None
    )
    given = getattr(translation, "boundary_tolerance_from", "default") == "input"
    try:
        wanted, no_radius = (
            radius_for_budget(
                fit,
                settings.radius,
                result.atoms,
                result.pseudopotentials,
                result.electron_count,
                spacing=settings.spacing,
                shift=settings.shift[0],
                stencil_half_width=settings.stencil_half_width,
                budget=budget,
                highest_occupied=highest,
                boundary_tolerance=(
                    hartree.boundary_tolerance
                    if given and hartree is not None
                    else None
                ),
                multipole_boundary=multipole,
                prepared=prepared,
            ),
            None,
        )
    except (ValueError, ArithmeticError) as error:
        # The estimate stands without a radius to go with it, and the
        # calculation it belongs to has finished: the block says why.
        wanted, no_radius = None, str(error)
    radius_seconds = perf_counter() - started - record["shell_seconds"]
    rough = []
    if fit.rms > 0.3:
        rough.append(f"the fit of the shells has an rms of {fit.rms:.2f}")
    if fit.at_scan_end:
        rough.append("the decay constant is at an end of its scan")
    lines = [
        (
            " Density at the sphere: the sphere adds an estimated "
            f"{fit.energy:.1E} Ry to the total energy (decay constant "
            f"{fit.decay:.2f} /bohr: a factor 10 per "
            f"{math.log(10.0) / (2.0 * fit.decay) / ANGSTROM:.2f} ang)"
        ),
    ]
    if rough:
        lines.append("   This estimate is rough: " + " and ".join(rough) + ".")
    if fit.energy > budget:
        lines.append(
            f"   WARNING: this exceeds Domain_Energy_Tolerance ({budget:.1E} Ry)"
            + (
                "."
                if wanted is None
                else "; the wall share needs a radius of "
                + wanted.radius_text
                + (
                    f" at least (the step is limited to {RADIUS_STEP_LIMIT:g} "
                    "ang)."
                    if wanted.limit == "at least"
                    else "."
                )
            )
        )
    elif fit.energy > share:
        lines.append(
            "   NOTE: this exceeds the wall share of Domain_Energy_Tolerance "
            f"({share:.1E} Ry)."
        )
    else:
        lines.append(
            f"   Wall share of Domain_Energy_Tolerance ({share:.1E} Ry): "
            "within it."
        )
    if wanted is None:
        lines.append(f"   No radius for the wall share: {no_radius}.")
    else:
        lines.append("   For this system the wall share would be met by:")
        suggestion = _domain_input_lines(wanted.radius_text, wanted.tolerance_text)
        suggestion[0] += _radius_comment(wanted)
        lines.extend(suggestion)
    if highest > 0.0:
        lines.append(
            f"   NOTE: the highest occupied level, {highest:+.4f} Ry, is "
            "above zero: it is held by the sphere; the energy has no limit "
            "for a large sphere."
        )
    if lowest_empty is not None:
        lines.append(
            f"   NOTE: lowest empty level {lowest_empty:+.4f} Ry; the rule "
            "does not cover empty levels, which need more vacuum."
        )
    record.update(
        status="estimated",
        energy_ry=fit.energy,
        decay_per_bohr=fit.decay,
        minus_energy_slope_ry_per_bohr=fit.prefactor,
        fit_rms=fit.rms,
        fit_shells=fit.shells,
        decay_at_scan_end=fit.at_scan_end,
        rough=fit.rough,
        highest_occupied_ry=highest,
        lowest_empty_ry=lowest_empty,
        boundary_sphere_radius=None if wanted is None else wanted.radius_text,
        hartree_boundary_tolerance=(
            None if wanted is None else wanted.tolerance_text
        ),
        radius_bohr=None if wanted is None else wanted.radius,
        radius_limit="" if wanted is None else wanted.limit,
        radius_set_by=(
            f"none: {no_radius}" if wanted is None else wanted.set_by
        ),
        radius_of_the_wall_alone=(
            None if wanted is None else wanted.wall_radius_text
        ),
        decay_of_the_step_per_bohr=None if wanted is None else wanted.decay,
        estimates=None if wanted is None else wanted.estimates,
        radius_seconds=radius_seconds,
        seconds=perf_counter() - started,
    )
    return lines, record


class ParsecTextReporter:
    """Stateful writer for the subset of ``parsec.out`` Python can support.

    ``domain`` is ``None`` until the set-up is reported and then, for a
    sphere, the record of the domain: radius, vacuum, the estimates of what
    the wall and the boundary values add, and after the SCF the estimate
    from the density under ``after_scf``.  With ``PARSEC_DOMAIN_REPORT=0``
    (:func:`domain_report_requested`) it stays ``None`` for an input that
    gives its radius, and nothing is estimated after the SCF.

    ``prepared_estimates`` is ``None`` unless a caller sets it before
    :meth:`finish`: estimates of the omitted potential that the block after
    the SCF may read in place of making them (:func:`domain_finish_lines`).
    This reporter prepares none.
    """

    def __init__(
        self,
        write: Callable[[str], None],
        translation: "ParsecInputTranslation",
    ) -> None:
        self.write = write
        self.translation = translation
        self.problem = translation.problem
        self._previous_total: float | None = None
        self._diagonalization_total = 0.0
        self._hartree_total = 0.0
        self._hamiltonian_binding_total = 0.0
        self._occupations_density_total = 0.0
        self._xc_total = 0.0
        self._mixing_energy_total = 0.0
        self.domain: dict | None = None
        self.prepared_estimates = None
        self._domain_report = domain_report_requested()
        # The lines of the Hartree boundary are those of an isolated domain.
        self._periodic = getattr(self.problem, "periodic_cell", None) is not None
        # Two spin channels change the header and the report of every step.
        self._spin = bool(getattr(self.problem.scf, "spin_polarized", False))

    def _complex_wavefunctions(self) -> bool:
        scf = self.problem.scf
        return bool(
            getattr(scf, "self_consistent_spin_orbit", False)
            or getattr(scf, "noncollinear", False)
            or getattr(self.problem, "has_kpoint_sampling", False)
        )

    def _boundary_lines(self, shape_name: str, domain_line: str) -> list[str]:
        """The boundary conditions and the domain: the wording of usrinputfile.F90 for each."""
        cell = getattr(self.problem, "periodic_cell", None)
        if cell is None:
            return [
                " Confined system (cluster) with zero boundary condition!",
                f"  {shape_name} domain shape:",
                domain_line,
                *domain_header_lines(self.translation),
            ]
        vectors = np.asarray(cell.lattice_vectors, dtype=float)
        return [
            " Periodic boundary conditions!",
            " System is fully periodic ",
            " Unit lattice vectors [bohr] : ",
            *(" ".join(f"{value:15.10f}" for value in row) for row in vectors),
        ]

    def header(self) -> None:
        grid = self.problem.grid
        scf = self.problem.scf
        eigensolver = self.problem.eigensolver
        mixing = self.problem.mixing
        xc_name = scf.xc_functional
        now = datetime.now().astimezone().strftime("%d-%b-%Y %H:%M:%S %z")
        shifted = any(abs(value) > 1.0e-14 for value in grid.shift)
        shape_name = "Spherical" if grid.domain_shape == "sphere" else "Box"
        if grid.domain_shape == "sphere":
            domain_line = f" --- Radius is {grid.radius:10.6f} bohrs"
        else:
            domain_line = (
                " --- Full side lengths are "
                + " ".join(f"{value:10.6f}" for value in grid.box_lengths)
                + " bohrs"
            )

        lines = [
            "",
            " =================================================================",
            "",
            "  PARSEC-PYTHON - Modular real-space DFT program",
            "  PARSEC-like report; Python real-space implementation",
            "",
            f" starting run on {now}",
            f" input file: {self.translation.source}",
            "",
            " =================================================================",
            "",
            (
                " Initial Run - starting from atomic potentials"
                if self.problem.initial_density_settings.method == "sad"
                else " Initial Run - starting from an imported/ML density"
            ),
            (
                " Initial density method = "
                f"{self.problem.initial_density_settings.method}"
            ),
            (
                " ignoresym= T"
                if getattr(self.translation, "ignore_symmetry", False)
                else " ignoresym= F"
            ),
            "",
            " Grid data:",
            " ~~------~~",
            "",
            *self._boundary_lines(shape_name, domain_line),
            f" Grid spacing is {grid.spacing:9.6f} bohrs",
            " Order of double grid is    1",
            (
                " Grid points are shifted from origin!"
                if shifted
                else " Grid points include the origin!"
            ),
            (
                " shift vector = "
                + " ".join(f"{value:8.4f}" for value in grid.shift)
                + "   [units of grid spacing]"
            ),
            # usrinputfile.F90: complex with k-point sampling, a magnetic field or complex_wfn; here also
            # the two-component spinors of spin-orbit coupling and non-collinear magnetism
            (
                " WAVEFUNCTIONS ARE COMPLEX!"
                if self._complex_wavefunctions()
                else " WAVEFUNCTIONS ARE REAL!"
            ),
            (
                " The Finite-difference expansion is of order "
                f"{grid.expansion_order:2d}"
            ),
            (
                " Python constructs the full active grid; the selected "
                "execution backend reports any exact symmetry reduction."
            ),
            "",
            " Eigenvalue data:",
            " ----------------",
            f" Number of states: {scf.number_of_states:12d}",
            f" Net cluster charge = {scf.net_charge:8.3f}      [e]",
            (
                f" Fermi temperature = {scf.fermi_temperature_kelvin:9.2f} [K]"
                if scf.fixed_occupations is None
                else " WARNING: reading occupations from occup.in file"
            ),
            "",
            " Self-consistency data:",
            " ----------------------",
            f" Maximum number of iterations is {scf.max_iterations:12d}",
            (
                " Performing Chebyshev subspace filtering"
                if eigensolver.method == "chebff"
                else (
                    " Performing Chebyshev-Davidson diagonalization"
                    if eigensolver.method == "chebdav"
                    else " Performing Lanczos/ARPACK diagonalization"
                )
            ),
            (
                " Polynomial degree for First-filter is "
                f"{eigensolver.first_filter_degree:6d}"
                if eigensolver.method == "chebff"
                else (
                    " Polynomial degree for Chebyshev-Davidson is "
                    f"{eigensolver.first_filter_degree:6d}"
                    if eigensolver.method == "chebdav"
                    else ""
                )
            ),
            (
                " The matvec operations use block size "
                f"{eigensolver.matvec_block_size:6d}"
            ),
            (
                " Polynomial degree for Chebyshev filtering is "
                f"{eigensolver.filter_degree:6d}"
            ),
            (
                " Change in polynomial degree (dpm) for Chebyshev filtering is "
                f"{eigensolver.filter_degree_delta:6d}"
            ),
            (
                " Self-consistency convergence criterion is "
                f"{scf.convergence_criterion:25.16E}  Ry"
            ),
            f" Diagonalization tolerance is {eigensolver.tolerance:25.16E}",
            f" Buffer size in subspace is {eigensolver.subspace_buffer:6d}",
            "",
            " Mixer data:",
            " -----------",
            f" solver lpole is : {self.problem.hartree.multipole_order:12d}",
            # The input, like the line above; the setup block below says
            # what the run made of the three.  A tolerance of the default
            # rule is no line of the input.  A periodic cell has no Hartree
            # boundary values and gets no line.
            *(
                ()
                if self._periodic
                else (
                    (
                        " Hartree boundary tolerance of the default rule is : "
                        if getattr(self.translation, "boundary_tolerance_from", "")
                        == "rule"
                        else " Hartree boundary tolerance in the input is : "
                    )
                    + (
                        "off"
                        if self.problem.hartree.boundary_tolerance is None
                        else f"{self.problem.hartree.boundary_tolerance:.6E}  Ry"
                    )
                    + f"   atomic tail : {self.problem.hartree.atomic_tail}",
                )
            ),
            " Anderson mixer",
            (
                f" Initial Jacobian: {mixing.parameter:6.3f}"
                f"  Mixing memory is {mixing.memory:2d}"
            ),
            f" Mixing restarted after {mixing.restart:12d} iterations",
            (
                " Anderson safeguard enabled: "
                f"step_limit={mixing.step_limit:.3g}, "
                f"growth_trigger={mixing.growth_trigger:.3g}, "
                f"backoff={mixing.backoff:.3g}"
                if mixing.safeguard
                else " Anderson safeguard disabled (strict PARSEC path)"
            ),
            "",
            " Correlation data:",
            " -----------------",
            f" Exchange-Correlation functional is {xc_name}",
            (
                " LDA, Ceperley-Alder, Perdew-Zunger parametrization"
                if xc_name == "ca"
                else " GGA, Perdew-Burke-Ernzerhof parametrization"
            ),
            # Only where Energy_Refinement asks for it.
            *(
                ()
                if getattr(self.translation, "energy_refinement", None) is None
                else (
                    " Post-SCF energy refinement: "
                    f"{self.translation.energy_refinement.upper()} "
                    "exchange-correlation energy of the converged orbitals "
                    "(one non-self-consistent evaluation)",
                )
            ),
            "",
            " Other input data:",
            " -----------------",
            f" output level [1 - 6] = {self.translation.output_level:2d}",
            *(
                (
                    " Spin-polarized computation!",
                    "",
                    " Initial spin polarization: ",
                    *(
                        f"  {symbol:<2s}     {potential.initial_spin_polarization:11.4f}"
                        for symbol, potential in self.problem.pseudopotentials.items()
                    ),
                )
                if self._spin
                else (" No spin effects!",)
            ),
            " No minimization!",
            "",
        ]
        self.write("\n".join(lines))

    def setup(self, system: "PreparedSinglePointSystem") -> None:
        timings = getattr(system, "timings", None)
        # What the rule chose is reported whatever the switch says; a radius
        # of the input only where the report is wanted.
        domain_lines: list[str] = []
        if self._domain_report or getattr(self.translation, "domain", None) is not None:
            domain_lines, self.domain = contained_domain_report(
                " --- Energy the sphere adds: no estimate",
                domain_setup_lines,
                system,
                self.translation,
            )
        lines = [
            " Atom data:",
            " --~--~---",
            "",
            f" Tot. # of atom types is {len(system.pseudopotentials):5d}",
            "",
        ]
        for symbol, potential in system.pseudopotentials.items():
            atoms = [atom for atom in system.atoms if atom.symbol == symbol]
            specification = self.problem.pseudopotentials[symbol]
            lines.extend(
                [
                    f" Chemical element : {symbol}",
                    (
                        " Physical element : "
                        f"{specification.element_symbol or potential.symbol}"
                    ),
                    f" {potential.file_format}",
                    (
                        "  pseudopotential format : old Martins"
                        if potential.file_format == "martins_old"
                        else "  pseudopotential format : new Martins"
                    ),
                    f"  pseudopotential file   : {potential.source}",
                    (
                        f"  radial points/channels : {potential.radii.size}"
                        f" / {potential.number_of_channels}"
                    ),
                    (
                        f"  selected local channel : "
                        f"l={specification.local_angular_momentum}"
                    ),
                    f" There are {len(atoms):6d} {symbol}  atoms",
                    " and their initial coordinates are:",
                    "",
                    "    x [bohr]          y [bohr]          z [bohr]",
                ]
            )
            for atom in atoms:
                lines.append(
                    f" {atom.position[0]:16.9f}"
                    f" {atom.position[1]:16.9f}"
                    f" {atom.position[2]:16.9f}"
                )
            lines.append("")
        lines.extend(
            [
                f" Tot. number of atoms = {len(system.atoms):7d}",
                "",
                " Real-space setup:",
                " -----------------",
                f" Full active grid points = {system.grid.size:12d}",
                f" Hamiltonian dimension   = {system.grid.size:12d}",
                (
                    " Sparse Laplacian nonzeros = "
                    f"{system.negative_laplacian.nnz:11d}"
                ),
                (
                    " Nonlocal projector columns = "
                    f"{system.nonlocal_operator.projectors.shape[1]:8d}"
                ),
                f" Number of electrons = {system.electron_count:14.8f}",
                (
                    " Initial density integral = "
                    f"{system.grid.integrate(system.initial_density):14.8f}"
                ),
                *self._initial_moment_lines(system),
                *hartree_boundary_lines(system, self.problem.hartree),
                *domain_lines,
                "",
            ]
        )
        if timings is not None:
            lines.extend(
                [
                    " Setup timings [sec]",
                    " --------------------------------------------------",
                    (
                        " Pseudopotential loading       : "
                        f"{timings.pseudopotential_loading_seconds:12.6f}"
                    ),
                    f" Grid-domain construction      : {timings.grid_seconds:12.6f}",
                    (
                        " Finite-difference construction: "
                        f"{timings.finite_difference_seconds:12.6f}"
                    ),
                    f" Local ionic potential setup   : {timings.local_ionic_seconds:12.6f}",
                    (
                        " Nonlocal ionic projector setup: "
                        f"{timings.nonlocal_ionic_seconds:12.6f}"
                    ),
                    (
                        " Initial valence-density setup : "
                        f"{timings.initial_density_seconds:12.6f}"
                    ),
                    f" Core-density setup            : {timings.core_density_seconds:12.6f}",
                    f" Ion-ion energy setup           : {timings.ion_ion_seconds:12.6f}",
                    # Neither is prepared for a periodic cell.
                    *(
                        ()
                        if self._periodic
                        else (
                            " Hartree boundary estimate/tail : "
                            f"{getattr(timings, 'hartree_boundary_seconds', 0.0):12.6f}",
                        )
                    ),
                    f" Preparation wall time          : {timings.total_seconds:12.6f}",
                    " --------------------------------------------------",
                    "",
                ]
            )
        self.write("\n".join(lines))

    def _initial_moment_lines(self, system) -> list[str]:
        """The starting magnetic moment, as ``initchrg.f90`` prints it."""

        channels = getattr(system, "initial_spin_densities", None)
        if channels is None:
            return []
        moment = system.grid.integrate(channels[0]) - system.grid.integrate(channels[1])
        return [f" Initial magnetic moment [Bohr magnetons] = {moment:12.5f}"]

    @staticmethod
    def _state_lines(eigenvalues, occupations, representations) -> list[str]:
        """One line per state, in the format of ``flevel.f90``."""

        return [
            f"{index:5d}   {eigenvalue:18.10f}"
            f"   {eigenvalue * RYDBERG_TO_EV:18.10f}"
            f" {occupation:9.4f}   {representation:6d}"
            for index, (eigenvalue, occupation, representation) in enumerate(
                zip(eigenvalues, occupations, representations), start=1
            )
        ]

    def _spin_level_lines(self, step: SCFIteration) -> list[str]:
        """Electrons, moment, Fermi level and the levels of both channels."""

        count = len(step.eigenvalues) // 2
        representations = (
            step.representations
            if len(step.representations) == len(step.eigenvalues)
            else (1,) * len(step.eigenvalues)
        )
        lines = [
            (
                " Electrons in up, down spins: "
                f"{step.spin_electrons[0]:7.2f} {step.spin_electrons[1]:7.2f}"
            ),
            (
                " Net magnetic moment (in Bohr magnetons): "
                f"{step.net_magnetic_moment:7.2f}"
            ),
            "",
            # No Fermi level is searched with the occupations of occup.in.
            *(
                (f" Fermi level at {step.fermi_level:10.4f} [Ry]", "")
                if self.problem.scf.fixed_occupations is None
                else ()
            ),
        ]
        for label, channel in (("up", slice(0, count)), ("dn", slice(count, None))):
            lines.extend(
                [
                    f"{label} State   Eigenvalue [Ry]      Eigenvalue [eV]"
                    "    Occup.     Repr.",
                    "",
                    *self._state_lines(
                        step.eigenvalues[channel],
                        step.occupations[channel],
                        representations[channel],
                    ),
                    "",
                    "",
                ]
            )
        for label, maximum, minimum in zip(
            ("up", "dn"), step.spin_density_maximum, step.spin_density_minimum
        ):
            lines.extend(
                [
                    " Max and min values of charge density [e/bohr^3]:"
                    f" {maximum:12.4E} {minimum:12.4E}   {label}",
                    "",
                ]
            )
        return lines

    def iteration(self, step: SCFIteration) -> None:
        self._diagonalization_total += step.diagonalization_seconds
        self._hartree_total += step.hartree_seconds
        self._hamiltonian_binding_total += getattr(
            step, "hamiltonian_binding_seconds", 0.0
        )
        self._occupations_density_total += getattr(
            step, "occupations_density_seconds", 0.0
        )
        self._xc_total += getattr(step, "xc_seconds", 0.0)
        self._mixing_energy_total += getattr(step, "mixing_energy_seconds", 0.0)
        energies = step.energies
        atom_count = len(self.problem.atoms)
        energy_per_atom_ev = (
            energies.total * RYDBERG_TO_EV / max(atom_count, 1)
        )
        lines = [
            f" SCF iter # {step.iteration:3d}",
            "",
            (
                " Diagonalization time [sec] : "
                f"{step.diagonalization_seconds:10.2f},"
                f"      tdiag_sum = {self._diagonalization_total:11.2f}"
            ),
            *(
                (f" Iteration time [sec]       : {step.total_seconds:10.2f}",)
                if getattr(step, "total_seconds", 0.0) > 0.0
                else ()
            ),
            "",
            *self._residual_lines(step),
        ]
        if getattr(step, "spin_polarized", False):
            # flevel.f90 and rho_hart.F90 report each channel.
            lines.extend(self._spin_level_lines(step))
        else:
            lines.extend(
                [
                    # flevel.f90 reports no Fermi level with the occupations
                    # of occup.in: none is searched.
                    *(
                        (f" Fermi level at {step.fermi_level:10.4f} [Ry]", "")
                        if self.problem.scf.fixed_occupations is None
                        else ()
                    ),
                    *self._kpoint_note(),
                    "   State   Eigenvalue [Ry]      Eigenvalue [eV]"
                    "    Occup.     Repr.",
                    "",
                    *self._state_lines(
                        step.eigenvalues,
                        step.occupations,
                        (
                            step.representations
                            if len(step.representations) == len(step.eigenvalues)
                            else (1,) * len(step.eigenvalues)
                        ),
                    ),
                    "",
                    (
                        " Max and min values of charge density [e/bohr^3]:"
                        f" {step.density_maximum:12.4E}"
                        f" {step.density_minimum:12.4E}"
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                f" Hartree potential time [sec]: {step.hartree_seconds:10.2f}",
                "",
                f"   Eigenvalue Energy             = {energies.eigenvalue:20.8f} [Ry]",
                f"   Hartree Energy                = {energies.hartree:20.8f} [Ry]",
                (
                    "   Integral_{Vxc*rho}            = "
                    f"{energies.integral_vxc_rho:20.8f} [Ry]"
                ),
                (
                    "   Exc = Integral{eps_xc*rho}    = "
                    f"{energies.exchange_correlation:20.8f} [Ry]"
                ),
                (
                    "   Electron-Ion energy           = "
                    f"{energies.electron_ion:20.8f} [Ry]"
                ),
                (
                    "   Ion-Ion Energy                = "
                    f"{energies.ion_ion:20.8f} [Ry]"
                ),
            ]
        )
        if self._previous_total is not None:
            delta_ev_per_atom = (
                (energies.total - self._previous_total)
                * RYDBERG_TO_EV
                / max(atom_count, 1)
            )
            lines.append(
                "   (E(new)-E(old))/atom  = "
                f"{delta_ev_per_atom:22.8f} [eV]"
            )
        lines.extend(
            [
                "",
                f"   Total Energy = {energies.total:22.8f} [Ry]",
                f"   Energy/atom  = {energy_per_atom_ev:22.8f} [eV]",
                "",
            ]
        )
        if getattr(step, "spin_polarized", False):
            # getsre.f90 prints the residuals of each channel.
            for label, plain, weighted in zip(
                ("up", "dn"), step.spin_plain_residuals, step.spin_weighted_residuals
            ):
                lines.extend(
                    [
                        (
                            f"  0-{step.iteration:3d} {label} "
                            "SRE of pot. & charge weighted pot = "
                            f"{plain:14.10f} {weighted:14.10f}"
                        ),
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    (
                        f"  0-{step.iteration:3d}    "
                        "SRE of pot. & charge weighted pot = "
                        f"{step.plain_residual:14.10f}"
                        f" {step.weighted_residual:14.10f}"
                    ),
                    "",
                ]
            )
        self._previous_total = energies.total
        self.write("\n".join(lines))

    def _kpoint_note(self) -> list[str]:
        """Reminder in each iteration block that only one k-point is listed."""
        if not self.problem.has_kpoint_sampling:
            return []
        return [
            " (eigenvalues of the first k-point only; the converged states of "
            "every k-point are listed at the end)",
            "",
        ]

    def _kpoint_header(
        self, index: int, count: int, k_point: Sequence[float], weight: float
    ) -> list[str]:
        """Banner for one k-point block: Cartesian k [bohr^-1], fractional
        coordinates of the reciprocal lattice vectors, and the weight."""
        k_point = np.asarray(k_point, dtype=float)
        cell = self.problem.periodic_cell
        fractional = (
            cell.lattice_vectors @ k_point / (2.0 * np.pi) if cell is not None else None
        )
        text = (
            f" k-point {index:3d} of {count:3d}   "
            f"k = ({k_point[0]:10.6f} {k_point[1]:10.6f} {k_point[2]:10.6f}) [1/bohr]"
        )
        if fractional is not None:
            text += (
                f"   frac = ({fractional[0]:8.5f} {fractional[1]:8.5f} "
                f"{fractional[2]:8.5f})"
            )
        return [text + f"   weight = {weight:9.6f}"]

    def _k_blocks(self, result, array_size: int):
        """Yield ``(slice, header_lines)`` per k-point for pooled arrays of
        ``array_size`` entries; a single unlabeled block without k-points."""
        k_points = getattr(result, "k_points", None)
        if k_points is None or len(k_points) == 0:
            yield slice(0, array_size), []
            return
        count = len(k_points)
        per_k = array_size // count
        weights = getattr(result, "k_weights", None)
        if weights is None:
            weights = np.full(count, 1.0 / count)
        for index in range(count):
            yield (
                slice(index * per_k, (index + 1) * per_k),
                self._kpoint_header(index + 1, count, k_points[index], weights[index]),
            )

    def _final_k_tables(self, result) -> list[str]:
        """Converged eigenvalues of every k-point (scalar k-point runs)."""
        if getattr(result, "k_points", None) is None:
            return []
        lines = ["", "Converged eigenvalues by k-point", ""]
        eigenvalues = np.asarray(result.eigenvalues)
        occupations = np.asarray(result.occupations)
        for block, header in self._k_blocks(result, eigenvalues.size):
            lines.extend(header)
            lines.append("   State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.")
            for index, (eigenvalue, occupation) in enumerate(
                zip(eigenvalues[block], occupations[block]), start=1
            ):
                lines.append(
                    f"{index:5d}   {eigenvalue:18.10f}"
                    f"   {eigenvalue * RYDBERG_TO_EV:18.10f} {occupation:9.4f}"
                )
            lines.append("")
        return lines

    def finish(
        self,
        result: SinglePointResult,
        elapsed_seconds: float,
        *,
        forces=None,
    ) -> None:
        """Report the end of one SCF loop.

        ``forces`` is the :class:`parsec_python.Forces.ForceResult` of the
        structure, where one was evaluated; its tables then take the place
        of the statement that forces are not calculated.
        """
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            status,
            "",
            (
                "Time for self-consistent field [sec] : "
                f"{elapsed_seconds:10.2f},"
                f"    tdiag_sum = {self._diagonalization_total:11.2f}"
            ),
            f"Time spent on Hartree potential [sec] : {self._hartree_total:11.2f}",
        ]
        domain_lines, after_scf = (
            contained_domain_report(
                " Density at the sphere: no estimate",
                domain_finish_lines,
                result,
                self.translation,
                self.problem.hartree,
                self.prepared_estimates,
            )
            if self._domain_report
            else ([], None)
        )
        if after_scf is not None:
            if self.domain is None:
                self.domain = {}
            self.domain["after_scf"] = after_scf
            lines.extend(["", *domain_lines])
        correction = float(getattr(result, "atomic_reference_correction", 0.0))
        if correction != 0.0:
            lines.extend(
                [
                    "",
                    (
                        "Atomic AE-minus-pseudo reference correction : "
                        f"{correction:20.8f} [Ry]"
                    ),
                    (
                        "Reference-corrected all-electron total      : "
                        f"{result.all_electron_total:20.8f} [Ry]"
                    ),
                    (
                        "Reference-corrected all-electron total      : "
                        f"{result.all_electron_total * RYDBERG_TO_EV:20.8f} [eV]"
                    ),
                ]
            )
        timings = getattr(result, "timings", None)
        if timings is not None:
            preparation = timings.preparation
            xc_timing_name = (
                "CA-LDA"
                if self.problem.scf.xc_functional == "ca"
                else self.problem.scf.xc_functional.upper()
            )
            initial_hamiltonian_total = (
                preparation.finite_difference_seconds
                + preparation.local_ionic_seconds
                + preparation.nonlocal_ionic_seconds
                + timings.initial_xc_seconds
            )
            lines.extend(
                [
                    "",
                    " Initial Hamiltonian component timings [sec]",
                    " --------------------------------------------------",
                    (
                        " Finite-difference (-nabla^2) : "
                        f"{preparation.finite_difference_seconds:12.6f}"
                    ),
                    (
                        " V_ion diagonal/local         : "
                        f"{preparation.local_ionic_seconds:12.6f}"
                    ),
                    (
                        " V_ion nonlocal projectors    : "
                        f"{preparation.nonlocal_ionic_seconds:12.6f}"
                    ),
                    (
                        f" Initial V_xc ({xc_timing_name})"
                        "             : "
                        f"{timings.initial_xc_seconds:12.6f}"
                    ),
                    f" Component subtotal             : {initial_hamiltonian_total:12.6f}",
                    " --------------------------------------------------",
                    "",
                    " SCF timing analysis [sec]",
                    " --------------------------------------------------",
                    f" Initial Hartree potential      : {timings.initial_hartree_seconds:12.6f}",
                    f" Initial exchange-correlation   : {timings.initial_xc_seconds:12.6f}",
                    (
                        " Hamiltonian binding subtotal  : "
                        f"{timings.hamiltonian_binding_seconds:12.6f}"
                    ),
                    f" Diagonalization subtotal       : {timings.diagonalization_seconds:12.6f}",
                    (
                        " Occupation/density subtotal   : "
                        f"{timings.occupations_density_seconds:12.6f}"
                    ),
                    f" Hartree potential subtotal     : {timings.hartree_seconds:12.6f}",
                    f" Exchange-correlation subtotal  : {timings.xc_seconds:12.6f}",
                    f" Mixing/energy subtotal         : {timings.mixing_energy_seconds:12.6f}",
                    f" SCF wall time                   : {timings.total_seconds:12.6f}",
                    " --------------------------------------------------",
                ]
            )
            lines.extend(self._kpoint_detail_lines(timings))
        lines.extend(self._final_k_tables(result))
        if forces is None:
            lines.extend(
                [
                    "",
                    "Forces, dipoles, and MPI statistics",
                    "are not calculated by this Python single-point implementation.",
                    "",
                    " =================================================================",
                ]
            )
        else:
            lines.extend(
                [
                    *force_report_lines(forces, self.translation.output_level),
                    f"Time for forces [sec] : {forces.seconds:11.2f}",
                    "",
                    "Dipoles and MPI statistics",
                    "are not calculated by this Python implementation.",
                    "",
                    " =================================================================",
                ]
            )
        self.write("\n".join(lines))

    def _spin_channel_table(
        self,
        label: str,
        eigenvalues: Sequence[float],
        occupations: Sequence[float],
    ) -> list[str]:
        lines = [
            f"   Spin {label}",
            "   State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.",
            "",
        ]
        for index, (eigenvalue, occupation) in enumerate(
            zip(eigenvalues, occupations), start=1
        ):
            lines.append(
                f"{index:5d}   {eigenvalue:18.10f}"
                f"   {eigenvalue * RYDBERG_TO_EV:18.10f}"
                f" {occupation:9.4f}"
            )
        lines.append("")
        return lines

    def iteration_spin_polarized(self, step: "SpinPolarizedSCFIteration") -> None:
        """Per-iteration report for :func:`run_scf_spin_polarized`.

        A leaner counterpart to :meth:`iteration`: the spin-polarized driver
        does not yet carry per-stage timing instrumentation (see its module
        docstring), so this omits the timing breakdown ``iteration`` prints.
        """
        energies = step.energies
        atom_count = len(self.problem.atoms)
        energy_per_atom_ev = energies.total * RYDBERG_TO_EV / max(atom_count, 1)
        lines = [
            f" Spin-polarized SCF iter # {step.iteration:3d}",
            "",
            *(
                (
                    f" Diagonalization time [sec] : {step.diagonalization_seconds:10.2f},"
                    f"      iteration time = {step.total_seconds:10.2f}",
                    "",
                )
                if getattr(step, "total_seconds", 0.0) > 0.0
                else ()
            ),
            *self._residual_lines(step),
            f" Fermi level at {step.fermi_level:10.4f} [Ry]",
            "",
            *self._kpoint_note(),
        ]
        lines.extend(
            self._spin_channel_table("up", step.eigenvalues_up, step.occupations_up)
        )
        lines.extend(
            self._spin_channel_table(
                "down", step.eigenvalues_down, step.occupations_down
            )
        )
        moment = sum(step.occupations_up) - sum(step.occupations_down)
        lines.extend(
            [
                f"   Magnetic moment N_up - N_down = {moment:12.4f}",
                "",
                f"   Eigenvalue Energy             = {energies.eigenvalue:20.8f} [Ry]",
                f"   Hartree Energy                = {energies.hartree:20.8f} [Ry]",
                (
                    "   Integral_{Vxc*rho}            = "
                    f"{energies.integral_vxc_rho:20.8f} [Ry]"
                ),
                (
                    "   Exc = Integral{eps_xc*rho}    = "
                    f"{energies.exchange_correlation:20.8f} [Ry]"
                ),
                (
                    "   Electron-Ion energy           = "
                    f"{energies.electron_ion:20.8f} [Ry]"
                ),
                f"   Ion-Ion Energy                = {energies.ion_ion:20.8f} [Ry]",
                "",
                f"   Total Energy = {energies.total:22.8f} [Ry]",
                f"   Energy/atom  = {energy_per_atom_ev:22.8f} [eV]",
                "",
                (
                    f"  0-{step.iteration:3d}    "
                    "SRE of pot. & charge weighted pot = "
                    f"{step.plain_residual:14.10f}"
                    f" {step.weighted_residual:14.10f}"
                ),
                "",
            ]
        )
        self.write("\n".join(lines))

    def _final_spin_k_tables(self, result) -> list[str]:
        """Converged up/down eigenvalues of every k-point."""
        if getattr(result, "k_points", None) is None:
            return []
        lines = ["Converged eigenvalues by k-point", ""]
        up = np.asarray(result.eigenvalues_up)
        down = np.asarray(result.eigenvalues_down)
        for block, header in self._k_blocks(result, up.size):
            lines.extend(header)
            lines.append("")
            lines.extend(
                self._spin_channel_table(
                    "up", up[block], np.asarray(result.occupations_up)[block]
                )
            )
            lines.extend(
                self._spin_channel_table(
                    "down", down[block], np.asarray(result.occupations_down)[block]
                )
            )
        return lines

    @staticmethod
    def _residual_lines(step) -> list[str]:
        """The largest eigenpair residual ``||H x - e x||`` of an iteration, where one was measured."""
        value = getattr(step, "eigen_residual_max", float("nan"))
        if value != value:  # NaN
            return []
        return [f" Eigensolver residual max [Ry] : {value:12.4e}", ""]

    @staticmethod
    def _kpoint_stage_lines(timings) -> list[str]:
        """Stage table of a k-point SCF that carries no ``SinglePointResult`` table (spin)."""
        if timings is None or timings.total_seconds <= 0.0:
            return []
        lines = [
            " SCF timing analysis [sec]",
            " --------------------------------------------------",
            f" Initial Hartree potential      : {timings.initial_hartree_seconds:12.6f}",
            f" Initial exchange-correlation   : {timings.initial_xc_seconds:12.6f}",
            f" Diagonalization subtotal       : {timings.diagonalization_seconds:12.6f}",
            f" Occupation/density subtotal    : {timings.occupations_density_seconds:12.6f}",
            f" Hartree potential subtotal     : {timings.hartree_seconds:12.6f}",
            f" Exchange-correlation subtotal  : {timings.xc_seconds:12.6f}",
            f" Mixing/energy subtotal         : {timings.mixing_energy_seconds:12.6f}",
            f" SCF wall time                   : {timings.total_seconds:12.6f}",
            " --------------------------------------------------",
        ]
        lines.extend(ParsecTextReporter._kpoint_detail_lines(timings))
        lines.append("")
        return lines

    @staticmethod
    def _kpoint_detail_lines(timings) -> list[str]:
        """What the k-point solver measured: its set-up and the seconds inside its workers."""
        detail = getattr(timings, "detail", None)
        if not detail:
            return []
        lines = ["", " K-point solver detail", " --------------------------------------------------"]
        if "workers" in detail:
            lines.append(f" {'Workers':30s} : {int(detail['workers']):12d}")
        lines.append(f" {'Solver and worker start-up':30s} : {detail.get('solver_setup', 0.0):12.6f}")
        worker = (
            ("worker_hamiltonian_seconds", "Bloch Hamiltonian (projectors)"),
            ("worker_upload_seconds", "Operator upload"),
            ("worker_solve_seconds", "Eigensolver"),
            ("worker_spill_seconds", "Subspace to the host"),
            ("worker_download_seconds", "Eigenvectors to the host"),
            ("worker_density_seconds", "Density of the k-points"),
            ("worker_gather_seconds", "Density joined over the MPI ranks"),
            ("worker_halo_begin_seconds", "Halo rows packed and sent (rank 0)"),
            ("worker_halo_wait_seconds", "Waiting for the halo rows (rank 0)"),
            ("worker_allsum_seconds", "Projector overlaps summed over ranks (rank 0)"),
        )
        present = [(key, label) for key, label in worker if key in detail]
        if present:
            lines.append(" Summed over the workers [worker-sec]")
            for key, label in present:
                lines.append(f"   {label:28s} : {detail[key]:12.6f}")
        solver = (
            ("lanczos_seconds", "Lanczos bounds"),
            ("filter_seconds", "Chebyshev filter"),
            ("orthonormalize_seconds", "Orthonormalization"),
            ("ritz_seconds", "Rayleigh-Ritz"),
            ("residual_seconds", "Residuals"),
        )
        present = [(key, label) for key, label in solver if key in detail]
        if present:
            lines.append(" Device eigensolver [worker-sec]")
            for key, label in present:
                lines.append(f"   {label:28s} : {detail[key]:12.6f}")
        counters = (
            ("first_solves", "First solves"),
            ("subspace_steps", "Subspace steps"),
            ("extra_cycles", "Extra CHEBFF cycles"),
            ("applications", "Operator applications"),
            ("application_columns", "Application columns"),
        )
        present = [(key, label) for key, label in counters if key in detail]
        if present:
            lines.append(" Eigensolver counters")
            for key, label in present:
                lines.append(f"   {label:28s} : {int(detail[key]):12d}")
        lines.append(" --------------------------------------------------")
        return lines

    def finish_spin_polarized(
        self, result: "SpinPolarizedSinglePointResult", elapsed_seconds: float
    ) -> None:
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            status,
            "",
            f"Time for self-consistent field [sec] : {elapsed_seconds:10.2f}",
            "",
            *self._kpoint_stage_lines(getattr(result, "timings", None)),
            f"Converged magnetic moment N_up - N_down = {result.magnetic_moment:12.4f}",
            f"Total Energy = {result.energies.total:22.8f} [Ry]",
            (
                "Energy/atom  = "
                f"{result.energies.total * RYDBERG_TO_EV / max(len(result.atoms), 1):22.8f} [eV]"
            ),
            "",
            *self._final_spin_k_tables(result),
            (
                "Forces, dipoles, and MPI statistics are not calculated by "
                "this Python single-point implementation."
            ),
            "",
            " =================================================================",
        ]
        self.write("\n".join(lines))

    def spin_orbit_correction(self, result: "PerturbativeSpinOrbitResult") -> None:
        """Report PARSEC's ``pls.F90``-style perturbative spin-orbit split.

        Diagonalizing the small ``2N x 2N`` spin-mixing matrix replaces the
        two independent spin channels above with ``2N`` SOC-split spinor
        eigenstates; ``magnetic_moment`` here is each spinor's own
        ``<S_z>``, not a single system-wide total.
        """
        lines = [
            " Perturbative spin-orbit correction (pls.F90-style)",
            " ---------------------------------------------------",
            "",
            "   State   Eigenvalue [Ry]      Eigenvalue [eV]   <S_z>",
            "",
        ]
        for index, (eigenvalue, moment) in enumerate(
            zip(result.eigenvalues, result.magnetic_moment), start=1
        ):
            lines.append(
                f"{index:5d}   {eigenvalue:18.10f}"
                f"   {eigenvalue * RYDBERG_TO_EV:18.10f}"
                f" {moment:9.4f}"
            )
        lines.append("")
        self.write("\n".join(lines))

    def iteration_self_consistent_soc(self, step: "SelfConsistentSOCIteration") -> None:
        """Per-iteration report for :func:`run_self_consistent_soc`.

        Eigenvalues here are already SOC-split spinor states (one set, not
        two spin channels), each holding at most one electron.
        """
        energies = step.energies
        atom_count = len(self.problem.atoms)
        energy_per_atom_ev = energies.total * RYDBERG_TO_EV / max(atom_count, 1)
        lines = [
            f" Self-consistent SOC iter # {step.iteration:3d}",
            "",
            *(
                (
                    f" Diagonalization time [sec] : {step.diagonalization_seconds:10.2f},"
                    f"      iteration time = {step.total_seconds:10.2f}",
                    "",
                )
                if getattr(step, "total_seconds", 0.0) > 0.0
                else ()
            ),
            *self._residual_lines(step),
            f" Fermi level at {step.fermi_level:10.4f} [Ry]",
            "",
            *self._kpoint_note(),
            "   State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.",
            "",
        ]
        for index, (eigenvalue, occupation) in enumerate(
            zip(step.eigenvalues, step.occupations), start=1
        ):
            lines.append(
                f"{index:5d}   {eigenvalue:18.10f}"
                f"   {eigenvalue * RYDBERG_TO_EV:18.10f}"
                f" {occupation:9.4f}"
            )
        lines.extend(
            [
                "",
                f"   Eigenvalue Energy             = {energies.eigenvalue:20.8f} [Ry]",
                f"   Hartree Energy                = {energies.hartree:20.8f} [Ry]",
                (
                    "   Integral_{Vxc*rho}            = "
                    f"{energies.integral_vxc_rho:20.8f} [Ry]"
                ),
                (
                    "   Exc = Integral{eps_xc*rho}    = "
                    f"{energies.exchange_correlation:20.8f} [Ry]"
                ),
                (
                    "   Electron-Ion energy           = "
                    f"{energies.electron_ion:20.8f} [Ry]"
                ),
                f"   Ion-Ion Energy                = {energies.ion_ion:20.8f} [Ry]",
                "",
                f"   Total Energy = {energies.total:22.8f} [Ry]",
                f"   Energy/atom  = {energy_per_atom_ev:22.8f} [eV]",
                "",
                (
                    f"  0-{step.iteration:3d}    "
                    "SRE of pot. & charge weighted pot = "
                    f"{step.plain_residual:14.10f}"
                    f" {step.weighted_residual:14.10f}"
                ),
                "",
            ]
        )
        self.write("\n".join(lines))

    @staticmethod
    def _soc_timing_lines(result: "SelfConsistentSOCResult") -> list[str]:
        """The stage table of a self-consistent SOC or non-collinear run (``result.timings``)."""
        timings = getattr(result, "timings", None)
        if not timings:
            return []
        stages = (
            ("initial_magnetization", "Initial magnetization"),
            ("setup_potentials", "Initial Hartree + exchange-correlation"),
            ("solver_setup", "Solver and worker start-up"),
            ("hamiltonian", "Hamiltonian binding subtotal"),
            ("diagonalization", "Diagonalization subtotal"),
            ("density", "Occupation/density subtotal"),
            ("hartree", "Hartree potential subtotal"),
            ("xc", "Exchange-correlation subtotal"),
            ("mixing_energy", "Mixing/energy subtotal"),
            ("final_download", "Orbitals to the host"),
        )
        lines = [" SCF timing analysis [sec]", " " + "-" * 50]
        total = 0.0
        for key, label in stages:
            if key in timings:
                lines.append(f" {label:30s} : {timings[key]:12.6f}")
                total += timings[key]
        lines.append(f" {'Sum of the stages':30s} : {total:12.6f}")
        lines.append(" " + "-" * 50)
        counters = {k[4:]: v for k, v in timings.items() if k.startswith("eig_")}
        if counters:
            lines.append(" Eigensolver detail")
            lines.append(" " + "-" * 50)
            for key, label in (
                ("workers", "workers"),
                ("first_solves", "first_solves"),
                ("subspace_steps", "subspace_steps"),
                ("extra_cycles", "extra CHEBFF cycles"),
                ("applications", "applications"),
                ("application_columns", "application_columns"),
                ("lanczos_seconds", "lanczos_seconds"),
                ("filter_seconds", "filter_seconds"),
                ("orthonormalize_seconds", "orthonormalize_seconds"),
                ("ritz_seconds", "ritz_seconds"),
                ("residual_seconds", "residual_seconds"),
                ("worker_hamiltonian_seconds", "worker: Bloch Hamiltonian"),
                ("worker_upload_seconds", "worker: operator upload"),
                ("worker_solve_seconds", "worker: eigensolver"),
                ("worker_spill_seconds", "worker: subspace to the host"),
                ("worker_download_seconds", "worker: eigenvectors to the host"),
                ("worker_density_seconds", "worker: density of the states"),
                ("worker_gather_seconds", "worker: density joined over ranks"),
                ("worker_halo_begin_seconds", "rank 0: halo rows packed and sent"),
                ("worker_halo_wait_seconds", "rank 0: waiting for halo rows"),
                ("worker_allsum_seconds", "rank 0: sums over the ranks"),
            ):
                if key in counters:
                    value = counters[key]
                    text = f"{value:12.6f}" if key.endswith("seconds") else f"{int(value):12d}"
                    lines.append(f" {label:34s} : {text}")
            lines.append(" " + "-" * 50)
            lines.append(" (worker rows are seconds summed over the workers)")
        lines.append("")
        return lines

    def finish_self_consistent_soc(
        self, result: "SelfConsistentSOCResult", elapsed_seconds: float
    ) -> None:
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            status,
            "",
            f"Time for self-consistent field [sec] : {elapsed_seconds:10.2f}",
            "",
        ]
        lines.extend(self._soc_timing_lines(result))
        lines.extend(
            [
                "Converged spin-orbit-split spinor states",
                "",
            ]
        )
        moments = np.asarray(result.magnetic_moment_per_state)
        eigenvalues = np.asarray(result.eigenvalues)
        occupations = np.asarray(result.occupations)
        for block, header in self._k_blocks(result, eigenvalues.size):
            lines.extend(header)
            lines.append(
                "   State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.    <S_z>"
            )
            lines.append("")
            for index, (eigenvalue, occupation, moment) in enumerate(
                zip(eigenvalues[block], occupations[block], moments[block]), start=1
            ):
                lines.append(
                    f"{index:5d}   {eigenvalue:18.10f}"
                    f"   {eigenvalue * RYDBERG_TO_EV:18.10f}"
                    f" {occupation:9.4f} {moment:9.4f}"
                )
            lines.append("")
        if result.magnetic_moment_vector is not None:
            mx, my, mz = (float(v) for v in result.magnetic_moment_vector)
            lines.extend(
                [
                    "",
                    "Converged magnetic moment vector (mx, my, mz) = "
                    f"{mx:10.4f} {my:10.4f} {mz:10.4f}   |m| = "
                    f"{float(result.magnetic_moment):10.4f}",
                ]
            )
        elif result.magnetic_moment is not None:
            lines.extend(
                [
                    "",
                    "Converged magnetic moment N_up - N_down = "
                    f"{result.magnetic_moment:12.4f}",
                ]
            )
        lines.extend(
            [
                "",
                f"Total Energy = {result.energies.total:22.8f} [Ry]",
                (
                    "Energy/atom  = "
                    f"{result.energies.total * RYDBERG_TO_EV / max(len(result.atoms), 1):22.8f} [eV]"
                ),
                "",
                (
                    "Forces, dipoles, and MPI statistics are not calculated by "
                    "this Python single-point implementation."
                ),
                "",
                " =================================================================",
            ]
        )
        self.write("\n".join(lines))


    def band_structure(self, result, files: Sequence[object] = (), elapsed_seconds: float = 0.0) -> None:
        """Summary of the post-SCF band structure (Fortran ``calc_bands``)."""
        counts = [int(value) for value in result.segment_counts]
        states = result.eigenvalues.shape[2]
        worst = float(np.max(result.residual_norms)) if result.residual_norms.size else 0.0
        lines = [
            " Band structure (non-self-consistent, Fortran calc_bands)",
            " -------------------------------------------------------",
            f" Hamiltonian                   : {result.mode}"
            + (f" (channels: {', '.join(result.channels)})" if len(result.channels) > 1 else ""),
            f" Segments / k-points / states  : {len(counts)} / {sum(counts)} / {states}",
            f" Fermi level (SCF)             : {result.fermi_level:16.8f} [Ry]",
            "",
            "   Segment  Name                       k-points",
        ]
        for index, (count, name) in enumerate(zip(counts, result.segment_names), start=1):
            lines.append(f"   {index:7d}  {name:<26s} {count:8d}")
        lines.extend(
            [
                "",
                f" Largest eigenpair residual |H v - e v| : {worst:12.4E} [Ry]",
            ]
        )
        attempts = getattr(result, "attempts", None)
        if attempts is not None and int(attempts.max()) > 1:
            retried = int(np.count_nonzero(attempts > 1))
            lines.append(
                f" CHEBDAV re-solved {retried} k-point(s) with a larger block "
                f"(up to {int(attempts.max())} attempts) to converge degenerate levels."
            )
        if worst > 1.0e-2:
            if self.problem.eigensolver.method == "chebff":
                advice = (
                    "CHEBFF has no convergence test: increase FF_MaxIter and/or "
                    "Chebdav_Degree, or use Eigensolver: chebdav."
                )
            else:
                advice = (
                    "the unconverged ones are usually the highest reported bands: raise "
                    "States_Num or Subspace_Buffer_Size."
                )
            lines.append(" WARNING: some bands have a residual above 1e-2 Ry;")
            lines.append(f"          {advice}")
        if elapsed_seconds:
            lines.append(f" Time for band structure [sec] : {elapsed_seconds:10.2f}")
        for path in files:
            lines.append(f" Wrote {path}")
        lines.append("")
        self.write("\n".join(lines))


__all__ = ["ParsecTextReporter", "RYDBERG_TO_EV"]
