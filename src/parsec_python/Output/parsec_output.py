"""PARSEC-like text reporting for the Python single-point CLI.

The formatter intentionally omits data Python does not calculate, including
point-group representation numbers, forces (unless requested) and MPI
statistics.  The total charge and dipole of an isolated system follow every
SCF loop, as ``dipole.f90`` prints them.
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
from .correction_output import (
    dispersion_header_lines,
    dispersion_report_lines,
    energy_correction_lines,
    hubbard_iteration_lines,
    hubbard_parameter_lines,
    hubbard_setup_lines,
)
from .force_output import force_report_lines
from .hybrid_output import hybrid_energy_lines, hybrid_finish_lines
from .kpoint_output import (
    kpoint_generation_lines,
    monkhorst_pack_input_lines,
    space_group_lines,
)
from ..Energy.dipole import DipoleMoment, cluster_dipole_moment
from ..models import MAXIMUM_MULTIPOLE_ORDER, SCFIteration, SinglePointResult
from ..V_ion import (
    center_cluster_geometry,
    ionic_charge,
    pseudopotential_matches_functional,
)
from ..V_xc.selection import (
    FORTRAN_CORRELATION_CODES,
    FORTRAN_CORRELATION_DESCRIPTIONS,
    XC_DISPLAY_NAMES,
)

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


def dipole_report_lines(dipole: DipoleMoment, net_charge: float) -> list[str]:
    """The lines ``dipole.f90`` writes after an SCF loop (formats 50-54)."""

    x, y, z = (float(value) for value in dipole.dipole_debye)
    return [
        f" Tot. charge (user input) [e] = {float(net_charge):12.8f}",
        f" Tot. charge (computed) [e] = {dipole.charge:12.8f}",
        " Dipole components [Debye]:",
        f" x-component = {x:12.8f}",
        f" y-component = {y:12.8f}",
        f" z-component = {z:12.8f}",
    ]


def _fortran_columns(values) -> str:
    """``3(f15.9,3x)`` of ``usrinputfile.F90`` and ``initial.f90``."""

    return "".join(f"{float(value):15.9f}   " for value in values).rstrip()


def point_charge_report_lines(
    input_charges, placed_charges, type_counts: Sequence[int]
) -> list[str]:
    """What ``usrinputfile.F90`` and ``initial.f90`` print of point charges.

    ``input_charges`` are the charges at the coordinates of the input and
    ``placed_charges`` where the calculation put them; ``type_counts`` the
    number of charges of each ``Point_Typ_Num`` type (one type per charge
    where it is empty).
    """

    input_charges = tuple(input_charges)
    if not input_charges:
        return []
    counts = list(type_counts) or [1] * len(input_charges)
    lines = [
        " Point Charge data:",
        " ------------------",
        f" Tot. # of point charge types is {len(counts):12d}",
    ]
    start = 0
    for count in counts:
        group = input_charges[start : start + count]
        start += count
        lines += [
            f" There are {count:4d} point charges with charge {group[0].charge:5.2f} e",
            " and their coordinates are:",
            "",
            "    x [bohr]          y [bohr]          z [bohr]",
            *(_fortran_columns(item.position) for item in group),
        ]
    lines += ["", f" Tot. number of point charges = {len(input_charges):4d}", ""]
    moved = any(
        not np.array_equal(np.asarray(old.position), np.asarray(new.position))
        for old, new in zip(input_charges, placed_charges)
    )
    if moved:
        lines += [
            "WARNING: point charge coordinates were shifted",
            "in accordance with shift of atom coordinates!",
            "The new point charge coordinates are:",
            *(_fortran_columns(item.position) for item in placed_charges),
            "",
        ]
    return lines


def slab_recentring_lines(translation) -> list[str]:
    """What ``initial.f90:215-224`` prints when it re-centred a slab along z.

    The shifted coordinates of every atom (format ``3(f15.9,3x)``), PARSEC's
    lines between the pseudopotential messages and the symmetry block;
    nothing for a slab that was not moved
    (``ParsecInputTranslation.slab_shift``) and for every other system.
    """

    if not getattr(translation, "slab_shift", 0.0):
        return []
    return [
        "",
        " WARNING: atom coordinates were shifted so",
        " that the slab is centered at the origin",
        " The new atom coordinates are:",
        *(_fortran_columns(atom.position) for atom in translation.problem.atoms),
        "",
    ]


def density_symmetrization_lines(reports) -> list[str]:
    """What ``parsec.F90`` and ``symm_scalar`` print for each density.

    ``reports`` are :class:`parsec_python.Grid.ScalarSymmetrization` records,
    the total density first (``isp = 1``) and the spin channels after it.
    """

    lines: list[str] = []
    for index, report in enumerate(reports, start=1):
        lines += [
            "",
            f"  Symmetrizing charge isp = {index:12d}",
            " maximum deviation after symmetrization: "
            f"{report.maximum_deviation_percent:10.4f} %",
            f" electron charge before symmetrization: {report.charge_before:10.4f}",
            f" electron charge after symmetrization: {report.charge_after:10.4f}",
        ]
    return lines


def dynamic_tolerance_lines(step) -> list[str]:
    """``parsec.F90:1307-1310`` after ``getsre``: the eigensolver tolerance of
    the next iteration with ``Dynamic_Diag_Tol``, nothing without it."""

    next_tolerance = getattr(step, "next_eigensolver_tolerance", None)
    if next_tolerance is None:
        return []
    return [
        "Dynamic tolerance is used in eigensolver",
        "",
        f"New tolerance:{next_tolerance:11.5f}",
        "",
    ]


def noncollinear_moment_lines(moments) -> list[str]:
    """``nclmom`` and the end of ``bxc.F90`` for one iteration.

    ``moments`` is a :class:`parsec_python.SCF.noncollinear.NoncollinearMoments`.
    """

    lines = [
        "",
        "Atomic magnetic moments (Bohr magnetons): ",
        "",
        "-atom-  --Mx--  --My--  --Mz--   [bohr magnetons]       ",
        "",
    ]
    for index, moment in enumerate(np.asarray(moments.atomic), start=1):
        lines.append(
            (f"{index:4d}  " + "".join(f"{float(value):7.2f} " for value in moment)).rstrip()
        )
    lines += [
        "",
        "",
        " Tot. Magnetization (Bohr magnetons)"
        + "".join(f"     {float(value):7.2f}" for value in moments.total),
        "",
        f" Integral{{|m|}} = {moments.absolute:7.2f}",
    ]
    return lines


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


def gspace_local_potential_lines(system) -> list[str]:
    """``pot_local.f90``'s lines for Fortran's G-space local potential, else none.

    Formats 14 and 16 (``'  in fft for local potential n = ',3i4``, ``'  max
    and min of potential ',2f10.4``, Hartree as Fortran prints them), then the
    alpha term (``totnrg.f90`` format 55; this port adds it to the eigenvalue
    energy, so it is reported here once).  Printed only with
    ``PeriodicGridSettings.local_potential = "g_space"``.
    """

    settings = getattr(getattr(system, "grid", None), "settings", None)
    if getattr(settings, "local_potential", "real_space") != "g_space":
        return []
    counts = tuple(int(value) for value in system.grid.points_per_axis)
    potential = np.asarray(system.ionic_potential, dtype=np.float64) / 2.0
    return [
        "",
        "  in fft for local potential n = " + "".join(f"{value:4d}" for value in counts),
        "",
        f"  max and min of potential {potential.max():10.4f}{potential.min():10.4f}",
        "",
        f"   Alpha Energy                  = {float(system.alpha_z_energy):20.8f} [Ry]"
        f" ({settings.alpha_term.replace('_', ' ')}, part of the eigenvalue energy)",
    ]


def periodic_geometry_lines(system, translation) -> list[str]:
    """PARSEC's lines of a super cell and of the grid of a general lattice.

    ``Super_Cell``/``Super_Cell_Vac`` (F ``usrinputfile.F90:741-769``), the
    lattice of Fortran's general grid (``usrinputfile.F90:894-904``,
    ``inipbc.f90:77-81, 213-222``) and its Laplacian directions
    (``pbc_grid_coefs.f90:353-362, 699-706``).  An orthorhombic cell on this
    port's own grid, without a super cell, gets no lines.  A note says when
    atoms were translated into the cell.
    """

    lines: list[str] = []
    super_cell = getattr(translation, "super_cell", None)
    vacuum = getattr(translation, "super_cell_vacuum", None)
    # A wire or slab prints its periodic directions (supercell(1:pbc%per)).
    periodic_count = int(
        getattr(getattr(getattr(translation, "problem", None), "grid", None), "periodicity", 3)
    )
    if super_cell is not None:
        lines.append(" Periodic cell is being replicated according to super-cell")
        lines.append(
            " specification. Super-cell size = "
            + " ".join(f"{value:d}" for value in super_cell[:periodic_count])
        )
    if vacuum is not None:
        lines.append(" Vacuum is added to super-cell")
        lines.append(
            " amount of vacuum space = "
            + " ".join(f"{value:d}" for value in vacuum[:periodic_count])
        )
    problem = getattr(system, "input", None)
    cell = getattr(problem, "periodic_cell", None)
    # (The Fortran-derived routines translate every atom about the cell centre at start, PARSEC 1.5's
    # back_to_cell, and the preparation into [0, 1): the note is for atoms the input gave outside [0, 1).)
    if cell is not None and getattr(translation, "atoms_outside_cell", None) is not False and any(
        not np.array_equal(np.asarray(given.position), np.asarray(placed.position))
        for given, placed in zip(problem.atoms, system.atoms)
    ):
        lines.append(
            " Atoms outside the periodic cell were translated into it by "
            "lattice vectors; the coordinates above are the translated ones"
        )
    grid = getattr(system, "grid", None)
    if getattr(grid, "is_confined_periodic", False):
        return lines + wire_slab_geometry_lines(system)
    if not getattr(grid, "is_general_periodic", False):
        return lines + ([""] if lines else [])
    vectors = cell.lattice_vectors
    reciprocal = 2.0 * np.pi * np.linalg.inv(vectors).T
    lines.extend([" Periodic boundary conditions!", " Unit lattice vectors [bohr] : "])
    lines.extend(
        "".join(f"{value:15.10f} " for value in row).rstrip() for row in vectors
    )
    lines.append(" Reciprocal Lattice (a.u.):")
    lines.extend(
        "".join(f"{value:15.10f} " for value in row).rstrip() for row in reciprocal
    )
    lines.append(f" Input grid spacing: {grid.settings.spacing:6.3f} bohr")
    lines.append(
        " Final grid spacings: "
        + "".join(f"{value:6.3f} " for value in grid.spacings)
        + "bohr"
    )
    lines.append(
        " Number of grid points along each direction:"
        + "".join(f" {int(value):3d}" for value in grid.points_per_axis)
    )
    lines.append(
        " shift vector = "
        + " ".join(f"{value:8.4f}" for value in grid.shift)
        + "   [units of grid spacing]"
    )
    coefficients = getattr(system, "laplacian_coefficients", None)
    if coefficients is not None and coefficients.number_of_directions > 0:
        lines.append(
            f" Laplacian number of directions: {coefficients.number_of_directions:2d}"
        )
        one_plane = coefficients.active[1] == 0 and coefficients.active[2] == 0 and (
            not np.any(coefficients.directions[1:])
        )
        shown = 1 if one_plane else 3
        if not one_plane:
            lines.append(" Laplacian directions:")
        for index in range(shown):
            offset = coefficients.directions[index]
            lines.append(
                "".join(f"{int(value):12d}" for value in offset)
                + f"   weight is: {coefficients.coefficients[3 + index]:25.17G}"
            )
        lines.append(" Laplacian coefficients data:")
        lines.append(
            "".join(f"{value:5.2f} " for value in coefficients.coefficients).rstrip()
        )
    return lines + [""]


def wire_slab_grid_lines(problem) -> tuple[str, ...]:
    """The grid data ``usrinputfile.F90:860-895`` prints for a wire or slab."""

    grid = problem.grid
    vectors = problem.periodic_cell.lattice_vectors
    if grid.periodicity == 1:
        return (
            " Tube boundary conditions!",
            " System is non-periodic on the yz plane",
            " and periodic along x direction.",
            "",
            " Periodicity length [bohr] : ",
            f"{vectors[0, 0]:15.10f}",
            "",
            f" Boundary cylinder radius is {grid.confinement_radius:10.6f} bohrs",
            f" User-provided grid spacing is {grid.spacing:9.6f} bohrs",
            " WARNING: grid spacing may be rescaled below!!",
            "",
        )
    return (
        " Periodic boundary conditions!",
        " System is periodic on the xy plane",
        " and non-periodic on the z direction.",
        "",
        " Unit lattice vectors [bohr] : ",
        *("".join(f"{value:15.10f} " for value in row).rstrip() for row in vectors),
        "",
        f" Boundary slab height is {grid.confinement_radius:10.6f} bohrs",
        f" User-provided grid spacing is {grid.spacing:9.6f} bohrs",
        " WARNING: grid spacing may be rescaled below!!",
        "",
    )


def _fortran_g(value: float, width: int = 11, digits: int = 3) -> str:
    """Fortran ``Gw.d`` of one number (``F`` with four trailing blanks or ``E``)."""

    magnitude = abs(float(value))
    if magnitude == 0.0 or 0.1 - 0.5 * 10.0 ** (-digits - 1) <= magnitude < 10.0**digits - 0.5:
        exponent = 0 if magnitude == 0.0 else int(np.floor(np.log10(magnitude))) + 1
        exponent = min(max(exponent, 0), digits)
        decimals = digits - exponent
        return f"{value:{width - 4}.{decimals}f}    "
    mantissa_exponent = int(np.floor(np.log10(magnitude))) + 1
    mantissa = float(value) / 10.0**mantissa_exponent
    if abs(round(mantissa, digits)) >= 1.0:
        mantissa /= 10.0
        mantissa_exponent += 1
    sign = "-" if mantissa < 0 else ""
    text = f"{sign}0.{int(round(abs(mantissa) * 10**digits)):0{digits}d}E{mantissa_exponent:+03d}"
    return text.rjust(width)


def wire_slab_geometry_lines(system) -> list[str]:
    """Setup lines of ``inipbc.f90`` and ``ionpot_wire/slab.f90`` for a wire or slab.

    The reciprocal lattice of the periodic directions, the steps and point
    counts of the grid (``inipbc.f90:77-81, 213-222``), the Laplacian of a
    non-orthogonal slab (``pbc_grid_coefs.f90``) and the charge neutrality
    check of the Gaussian charges (``ionpot_wire.f90:194-201``).
    """

    grid = system.grid
    periodicity = grid.periodicity
    reciprocal = 2.0 * np.pi * np.linalg.inv(grid.cell.lattice_vectors).T
    lines = [
        " Reciprocal Space Data:",
        " ----------------------",
        " Reciprocal Lattice (a.u.):",
        *(
            "".join(f"{value:15.10f} " for value in reciprocal[row, :periodicity]).rstrip()
            for row in range(periodicity)
        ),
        "",
        f" Input grid spacing: {grid.settings.spacing:6.3f} bohr",
        " Final grid spacings: " + "".join(f"{value:6.3f} " for value in grid.spacings) + "bohr",
        " Number of grid points along each direction:"
        + "".join(f" {int(value):3d}" for value in grid.points_per_axis),
    ]
    coefficients = getattr(system, "laplacian_coefficients", None)
    if coefficients is not None and coefficients.number_of_directions > 0:
        lines.append("")
        lines.append(
            f" Laplacian number of directions: {coefficients.number_of_directions:2d}"
        )
        offset = coefficients.directions[0]
        lines.append(
            "".join(f"{int(value):12d}" for value in offset)
            + f"   weight is: {coefficients.coefficients[3]:25.17G}"
        )
        lines.append(" Laplacian coefficients data:")
        lines.append(
            "".join(f"{value:5.2f} " for value in coefficients.coefficients).rstrip()
        )
    short_range = getattr(system, "short_range", None)
    if short_range is not None:
        net = short_range.nominal_charge - short_range.grid_charge
        lines += [
            "",
            " Charge neutrality check.",
            " net charge in unit cell [e] = "
            + _fortran_g(net)
            + "   out of "
            + _fortran_g(short_range.nominal_charge),
            "",
        ]
        if abs(net) > 1.0e-2:
            lines.append(
                "  WARNING: net charge is not zero. The  compensated ionic charge "
                f"is not neutral: {short_range.nominal_charge} {short_range.grid_charge}"
            )
    forces = getattr(system, "ion_ion_forces", None)
    if periodicity == 2 and forces is not None:
        # ewald_slab.f90 prints its forces whatever the output level
        # (parsec.F90:874 passes ipr = 1), with the charged sheets' added.
        from .force_output import _table

        positions = np.array([atom.position for atom in system.atoms], dtype=float)
        if getattr(system, "charged_sheet_forces", None) is not None:
            lines += [
                "",
                " Forces from ion/charged sheet interaction:",
                " (combined forces)",
                " =========================================",
                *_table(positions, forces),
                "",
            ]
        lines += [
            "",
            " Forces from ion-ion interaction:",
            " ================================",
            *_table(positions, forces),
            "",
        ]
    return lines + [""]


def charged_sheet_report_lines(sheets) -> list[str]:
    """``usrinputfile.F90:2232-2297``: the charged sheets of a slab."""

    if not sheets:
        return []
    lines = [" Charged sheet data:", " ------------------"]
    # One line per sheet type of the input: consecutive sheets of one charge.
    groups: list[list] = []
    for sheet in sheets:
        if groups and groups[-1][0].charge == sheet.charge:
            groups[-1].append(sheet)
        else:
            groups.append([sheet])
    lines.append(f" Tot. # of charged sheets is {len(groups):12d}")
    for group in groups:
        lines += [
            f" There are {len(group):4d} charged sheets with charge {group[0].charge:5.2f} e",
            " and their coordinates are:",
            "",
            "    x [bohr]          y [bohr]          z [bohr]",
            *(f"{0.0:15.9f}   {0.0:15.9f}   {sheet.height:15.9f}" for sheet in group),
        ]
    lines += ["", f" Tot. number of charged sheets = {len(sheets):4d}", ""]
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


def _mixer_lines(mixing) -> list[str]:
    """The mixer lines of the header (``usrinputfile.F90:1604-1661``).

    The Anderson lines, with the line of the safeguard of this port, are
    those printed before the other mixers existed; the generalized Broyden
    and multisecant lines are PARSEC's.
    """

    method = getattr(mixing, "method", "anderson")
    if method == "anderson":
        return [
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
        ]
    if method == "broyden":
        return [
            "  Generalized Broyden mixer",
            (
                f" Initial Jacobian: {mixing.parameter:6.3f}"
                f"  Mixing memory is {mixing.memory:2d}"
            ),
            f" Mixing restarted after {mixing.restart:12d} iterations",
        ]
    style = "Broyden-like" if mixing.en_like == 0 else "EN-like"
    group = (
        "  All secant equations taken into account in one group"
        if mixing.group_size == 0 and method != "msecant3"
        else f"  Group size of secant equations: {mixing.group_size:2d}"
    )
    if method == "msecant3":
        head = [
            f" Multisecant method, {style} hybrid update",
            " Preferred type of update is "
            + ("Type-I" if mixing.preferred_type == 1 else "Type-II"),
        ]
    else:
        head = [
            f" Multisecant method, {style} "
            + ("Type-I" if method == "msecant1" else "Type-II")
            + " update"
        ]
    return [
        *head,
        f" Mixing parameter: {mixing.parameter:6.3f}{group}",
        f" Restarting factor is: {mixing.restart_factor:6.3f}",
    ]


def parsec15_solver_header_lines(eigensolver, scf) -> tuple[str, ...]:
    """The lines of the PARSEC 1.5 solver options an input sets (none at the defaults).

    The wording of ``usrinputfile.F90`` of 1.5 where 1.5 prints one (Vector
    Seed, S.M.A.R.T, Scaled Chebyshev, the filter restart); the PARSEC.py
    extensions say what they select.
    """

    lines = []
    seed = getattr(eigensolver, "vector_seed", None)
    if seed is not None:
        lines += [" **Vector Seed Specified", f" Vector Seed: {seed:12d}"]
    if getattr(eigensolver, "exact_state_count", False):
        lines.append(" Working space of exactly States_Num vectors (Exact_State_Count, PARSEC 1.5)")
    test = getattr(eigensolver, "first_filter_residual_test", "off")
    if test != "off":
        lines.append(
            " First filter stops when the largest residual is below the diagonalization "
            f"tolerance ({'stop' if test == 'stop' else 'continue'} after FF_MaxIter cycles)"
        )
    if getattr(eigensolver, "scaled_subspace_filter", False):
        lines.append(" Performing (EXPERIMENTAL) Scaled Chebyshev subspace filtering")
    rule = getattr(eigensolver, "subspace_lower_bound", "monotone")
    if rule == "smart":
        lines.append(" Using (EXPERIMENTAL) S.M.A.R.T adaptive lower filter bound")
    elif rule == "adaptive":
        lines.append(" Using the adaptive lower filter bound of PARSEC 1.5 (Smart_Subspace false)")
    bounds = getattr(eigensolver, "lanczos_bounds", "parsec")
    if bounds != "parsec":
        lines.append(
            f" Lanczos bounds of PARSEC 1.5 ({bounds}), max(degree/2, "
            f"{getattr(eigensolver, 'lanczos_bound_steps', 10)}) steps"
        )
    threshold = getattr(eigensolver, "reorthogonalization_threshold", 0.1)
    if threshold != 0.1:
        lines.append(f" Gram-Schmidt reorthogonalization threshold {threshold:8.4f}")
    after = getattr(scf, "degree_decrease_after", 4)
    if after != 4:
        lines.append(f" Chebyshev polynomial degree may decrease after iteration {after:4d}")
    restart = getattr(eigensolver, "filter_restart", None)
    if restart is not None:
        lines += [
            f" Will restart from scratch after {restart[0]:12d} ionic steps",
            f" restart_cheb_remained_times {restart[1]:12d}",
            f" remained_times_atomic_chrg_den {restart[2]:12d}",
        ]
    return tuple(lines)


def iterative_eigensolver_header_lines(eigensolver) -> tuple[str, ...]:
    """Eigensolver lines of usrinputfile.F90 for ARPACK, DIAGLA and TEST mode."""

    if eigensolver.method == "none":
        return (" Eigensolver is 'none'. Preparing for Special TEST MODE",)
    lines = [
        (
            " Using ARPACK (implicitly restarted Arnoldi eigensolver)"
            if eigensolver.method == "arpack"
            else " Using DIAGLA (block preconditioned Lanczos eigen-problem solver)"
        ),
        (
            " The matvec operations use block size "
            f"{eigensolver.matvec_block_size:6d}"
        ),
    ]
    if eigensolver.filters_later_iterations:
        lines.extend(
            [
                " Performing Chebyshev subspace filtering",
                (
                    " Polynomial degree for Chebyshev filtering is "
                    f"{eigensolver.filter_degree:6d}"
                ),
                (
                    " Change in polynomial degree (dpm) for Chebyshev filtering is "
                    f"{eigensolver.filter_degree_delta:6d}"
                ),
                " Performing Rayleigh-Ritz rotation on all SCF iterations",
            ]
        )
    else:
        lines.append(" No Chebyshev subspace filtering")
    return tuple(lines)


def iterative_eigensolver_count_lines(
    counts, cumulative: int, spin_polarized: bool
) -> tuple[list[str], int]:
    """eigval.F90's report of an ARPACK or DIAGLA solve, and the new total.

    ``counts`` holds ``(nec, maxmvs)`` per spin channel, each an integer or
    a tuple of one integer per representation (gfortran's list-directed
    ``write(7,*)`` of ``nec(:)`` and ``maxmvs(:)``: 12 columns a value).
    """

    def listed(value) -> str:
        if isinstance(value, (tuple, list)):
            return "".join(f"{int(item):12d}" for item in value)
        return f"{int(value):12d}"

    lines: list[str] = []
    for channel, (converged, matvecs) in enumerate(counts):
        if spin_polarized:
            lines.append(" SPIN UP" if channel == 0 else " SPIN DOWN")
        cumulative += (
            sum(int(item) for item in matvecs)
            if isinstance(matvecs, (tuple, list))
            else int(matvecs)
        )
        lines.extend(
            [
                " Number of converged eigen-pairs per representation:"
                f"{listed(converged)}",
                f" Number of matrix-vector multiplications: {listed(matvecs)}",
                f" Cumulated number of matrix-vector mult. = {cumulative:12d}",
            ]
        )
    return lines, cumulative


def fourier_filter_lines(problem) -> tuple[str, ...]:
    """The lines ``fourier_f.f90:95-132`` prints about the Fourier filters.

    Only for a periodic system with a filtered species: PARSEC prints them
    with the transforms of every periodic run, which this report leaves
    out otherwise.
    """

    from ..Pseudopotential.fourier_filter import DELTA_Q, reciprocal_mesh_size

    cell = getattr(problem, "periodic_cell", None)
    species = problem.pseudopotentials
    if cell is None or not any(
        getattr(item, "has_fourier_filter", False) for item in species.values()
    ):
        return ()
    from ..Grid.pbc import adjust_step, fortran_grid_counts, fortran_orthorhombic_counts

    if getattr(problem.grid, "fortran_routines", False):
        counts = fortran_orthorhombic_counts(cell, problem.grid)
        if counts is None:
            # grid%step(1) of Fortran's general grid (adjust_step with the space group's translations).
            spacing = float(fortran_grid_counts(cell, problem.grid)[1][0])
        else:
            # the one spacing L_0/n_0 of the orthorhombic grid (Fortran's counts)
            side = float(np.diag(cell.lattice_vectors)[0])
            spacing = side / float(counts[0])
    elif _round_rule_general(cell, problem.grid):
        # (Andrew Grieder's grid: the lines as they have been printed.)
        spacing = adjust_step(
            float(np.linalg.norm(cell.lattice_vectors[0])), problem.grid.spacing
        )[1]
    else:
        side = float(np.diag(cell.lattice_vectors)[0])
        spacing = side / float(np.round(side / problem.grid.spacing))
    lines = [" Performing Fourier transform of pseudopotentials:"]
    for symbol, item in species.items():
        alpha = item.fourier_filter_alpha
        core = alpha if item.fourier_filter_core is None else item.fourier_filter_core
        count = reciprocal_mesh_size(spacing, alpha, core)
        lines += ["", f" {symbol:<2s}     nql={count:5d}     delq={DELTA_Q:6.3f}"]
        if alpha > 0.0:
            lines.append(" Actually also performing fourier filtering in pseudopotentials ")
        if core > 0.0:
            lines.append(" Actually also performing fourier filtering in core charge ")
        if alpha > 0.0 or core > 0.0:
            lines.append(
                f"  alpha = {alpha:10.4f}  beta1 = {item.fourier_filter_beta1:10.4f}"
                f"  acore = {core:10.4f}"
            )
    return tuple(lines)


def _round_rule_general(cell, settings, tolerance: float = 1.0e-6) -> bool:
    """The test of ``uses_general_periodic_grid`` before PARSEC 1.5's parity round (``round(L/h)`` points), which
    the Fourier-filter lines of Andrew Grieder's grids keep."""

    from ..Grid.cell import is_diagonal_lattice

    vectors = cell.lattice_vectors
    if not is_diagonal_lattice(vectors):
        return True
    sides = np.diag(vectors)
    if np.any(sides <= 0.0):
        return True
    counts = np.round(sides / settings.spacing).astype(np.int64)
    if np.any(counts <= 2 * settings.stencil_half_width):
        return True
    steps = sides / counts
    return not np.allclose(steps, steps[0], rtol=tolerance)


def _correlation_lines(scf) -> list[str]:
    """The functional lines under "Correlation data" (usrinputfile.F90:2330-2361).

    CA and PBE keep the lines this report has always had (PBE is named
    ``pbe`` where Fortran prints ``pb``); the other functionals print the
    two-letter code and the description of the Fortran.  Fortran writes
    the X-alpha line at every evaluation of the functional; it is written
    once here.
    """

    functional = scf.xc_functional
    if getattr(scf, "is_hybrid", False):
        from .hybrid_output import hybrid_correlation_lines

        return hybrid_correlation_lines(scf)
    if functional == "ca":
        return [
            " Exchange-Correlation functional is ca",
            " LDA, Ceperley-Alder, Perdew-Zunger parametrization",
        ]
    if functional == "pbe":
        return [
            " Exchange-Correlation functional is pbe",
            " GGA, Perdew-Burke-Ernzerhof parametrization",
        ]
    lines = [
        f" Exchange-Correlation functional is {FORTRAN_CORRELATION_CODES[functional]}",
        f" {FORTRAN_CORRELATION_DESCRIPTIONS[functional]}",
    ]
    if functional == "casida_salahub":
        lines.append(
            " Ionization energy difference is: "
            f"{scf.ionization_energy_difference:7.4f} Ry"
        )
    if functional == "xalpha":
        lines.append(" performed x-alpha correlation with alpha =  1.000")
    return lines


def _routines_lines(translation) -> tuple[str, ...]:
    from ..Input.routines import routines_report_line

    line = routines_report_line(translation)
    return () if line is None else (line,)


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
        # eigval.F90's solver%totalmv, for the ARPACK and DIAGLA reports.
        self._cumulative_matvecs = 0
        self.domain: dict | None = None
        self.prepared_estimates = None
        # The DipoleMoment of the last SCF loop of an isolated system.
        self.last_dipole: DipoleMoment | None = None
        self._system = None
        self._domain_report = domain_report_requested()
        # The lines of the Hartree boundary are those of an isolated domain.
        self._periodic = getattr(self.problem, "periodic_cell", None) is not None
        # A cell periodic in three dimensions; parsec.F90 calls dipole.f90
        # for every other system (a cluster, a wire, a slab).
        self._bulk = self._periodic and getattr(self.problem.grid, "periodicity", 3) == 3
        # The force report of a crystal of Andrew Grieder's routines (Fortran's
        # form: no ion-ion table, no torque); the Fortran-derived routines'
        # forces carry their own (ForceResult.periodic).
        self._crystal_report = self._periodic and not getattr(
            getattr(translation, "routines", None), "fortran", False
        )
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
        if getattr(cell, "dimensions", 3) == 2:
            return [
                " Periodic boundary conditions!",
                " System is periodic on the xy plane",
                " and non-periodic on the z direction.",
                "",
                " Unit lattice vectors [bohr] : ",
                *(" ".join(f"{value:15.10f}" for value in row) for row in vectors[:2]),
                "",
                f" Boundary slab height is {0.5 * vectors[2, 2]:12.6f} bohrs",
            ]
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
        now = datetime.now().astimezone().strftime("%d-%b-%Y %H:%M:%S %z")
        shifted = any(abs(value) > 1.0e-14 for value in grid.shift)
        shape_name = {
            "sphere": "Spherical",
            "ellipsoid": "Ellipsoidal",
            "cylinder": "Cylindrical",
        }.get(grid.domain_shape, "Box")
        if grid.domain_shape == "sphere":
            domain_line = f" --- Radius is {grid.radius:10.6f} bohrs"
        elif grid.domain_shape == "ellipsoid":
            # usrinputfile.F90 '(a,f10.6,a)' lines of the ellipsoid.
            domain_line = "\n".join(
                f" --- {name} radius is {value:10.6f} bohrs"
                for name, value in zip("xyz", grid.domain_parameters)
            )
        elif grid.domain_shape == "cylinder":
            radius, length, axis = grid.domain_parameters
            domain_line = (
                f" --- Oriented along the {'xyz'[int(axis) - 1]} axis\n"
                f" --- Radius is {radius:10.6f} bohrs\n"
                f" --- Length is {length:10.6f} bohrs"
            )
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
            # The routines of the run (parsec_python.Input.routines), where the choice matters.
            *_routines_lines(self.translation),
            "",
            " =================================================================",
            "",
            (
                " Restarted Run - from parsec.dat"
                if getattr(self.translation, "restart_run", False)
                else (
                    " Initial Run - starting from atomic potentials"
                    if self.problem.initial_density_settings.method == "sad"
                    else " Initial Run - starting from an imported/ML density"
                )
            ),
            # usrinputfile.F90 l.372-375 (Restart_From_Wfndat; its text names
            # parsec.dat although the file read is wfn.dat).
            *(
                (
                    " WARNING! Restarting from file parsec.dat ",
                    " You should think seriously about upgrading source code!",
                )
                if getattr(self.translation, "restart_from_wfndat", False)
                else ()
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
            *(
                wire_slab_grid_lines(self.problem)
                if getattr(grid, "periodicity", 3) in (1, 2)
                else (
                    # a cluster, or a bulk cell (usrinputfile.F90's wording for each)
                    *self._boundary_lines(shape_name, domain_line),
                    f" Grid spacing is {grid.spacing:9.6f} bohrs",
                    # usrinputfile.F90:917-921 prints the order and the warning only
                    # for N > 1; this report has always printed the line for N = 1.
                    f" Order of double grid is {getattr(grid, 'double_grid_order', 1):4d}",
                    *(
                        (
                            " WARNING! DOUBLE GRID WILL MODIFY FORCES AND ",
                            "",
                            "TOTAL ENERGY.",
                            "",
                        )
                        if getattr(grid, "double_grid_order", 1) > 1
                        else ()
                    ),
                )
            ),
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
            *fourier_filter_lines(self.problem),
            # usrinputfile.F90:957-993, only for Fortran's k-point path (an
            # input with Monkhorst_Pack_Shift).
            *(
                monkhorst_pack_input_lines(
                    self.problem.monkhorst_pack_dimensions,
                    self.problem.monkhorst_pack_shift,
                )
                if getattr(self.problem, "monkhorst_pack_shift", None) is not None
                else ()
            ),
            # usrinputfile.F90: only where External_Mag_Field is not zero.
            *(
                (
                    "",
                    " Applying external uniform magnetic field",
                    " Magnetic field: H ="
                    f"{self.translation.magnetic_field_gauss:11.5f} Gauss",
                )
                if getattr(self.translation, "magnetic_field_gauss", 0.0)
                else ()
            ),
            # usrinputfile.F90: complex with k-point sampling, a magnetic field or complex_wfn; here also
            # the two-component spinors of spin-orbit coupling and non-collinear magnetism
            (
                " WAVEFUNCTIONS ARE COMPLEX!"
                if getattr(self.translation, "complex_wavefunctions", False)
                # usrinputfile.F90:1226: k-points make them complex (Fortran's
                # k-point path, an input with Monkhorst_Pack_Shift, and every
                # k-point run).
                or getattr(self.problem, "monkhorst_pack_shift", None) is not None
                or self._complex_wavefunctions()
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
            *(
                (
                    (
                        " Performing Chebyshev subspace filtering"
                        if eigensolver.method == "chebff"
                        else " Performing Chebyshev-Davidson diagonalization"
                    ),
                    (
                        " Polynomial degree for First-filter is "
                        f"{eigensolver.first_filter_degree:6d}"
                        if eigensolver.method == "chebff"
                        else " Polynomial degree for Chebyshev-Davidson is "
                        f"{eigensolver.first_filter_degree:6d}"
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
                        " Change in polynomial degree (dpm) for Chebyshev "
                        f"filtering is {eigensolver.filter_degree_delta:6d}"
                    ),
                )
                if eigensolver.method in {"chebff", "chebdav"}
                else iterative_eigensolver_header_lines(eigensolver)
            ),
            # usrinputfile.F90:1444-1450, only with Fix_Number_Eigenvalues.
            *(
                (
                    " Fixing the number of eigenvalues per representation",
                    " WARNING!! This option should only be used when "
                    "restarting from previous SCF.",
                    "",
                )
                if getattr(self.translation, "fix_number_eigenvalues", False)
                else ()
            ),
            (
                " Self-consistency convergence criterion is "
                f"{scf.convergence_criterion:25.16E}  Ry"
            ),
            # Only where the input gives Convergence_Criterion_Approach.
            *(
                ()
                if getattr(scf, "convergence_criterion_approach", None) is None
                else (
                    " Self-consistency convergence indicator is at "
                    f"{scf.convergence_criterion_approach:25.16E}  Ry",
                    " If applicable, Chebyshev polynomial order will decrease to ",
                    " 10 after SRE becomes less than the self-consistency ",
                    " convergence indicator.",
                )
            ),
            f" Diagonalization tolerance is {eigensolver.tolerance:25.16E}",
            *(
                (
                    " YOU ARE USING DYNAMIC TOLERANCE FOR CONVERGENCE",
                    " (first SCF iteration 0.1, then max(min(tol, SRE/10), "
                    "Diagonalization tolerance))",
                )
                if getattr(eigensolver, "dynamic_tolerance", False)
                else ()
            ),
            # usrinputfile.F90:1493-1494: Dynamic_Diag_Tol with Restart_Run.
            *(
                (" dynamic tolerance is not working with restart run!",)
                if getattr(self.translation, "dynamic_tolerance_off_for_restart", False)
                else ()
            ),
            f" Buffer size in subspace is {eigensolver.subspace_buffer:6d}",
            # The PARSEC 1.5 solver options of the input (none by default).
            *parsec15_solver_header_lines(eigensolver, scf),
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
            *_mixer_lines(mixing),
            "",
            " Correlation data:",
            " -----------------",
            *_correlation_lines(scf),
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
            # Only with Van_Der_Waals (usrinputfile.F90).
            *dispersion_header_lines(scf, self._periodic),
            "",
            " Other input data:",
            " -----------------",
            f" output level [1 - 6] = {self.translation.output_level:2d}",
            *(
                (
                    " Symmetrizing charge density according to symm. operations "
                    "in the full point group",
                )
                if getattr(scf, "symmetrize_density", False)
                else ()
            ),
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
            # usrinputfile.F90:2584-2589; the r*V line of the default is not
            # part of this report.
            *(
                ("", " Using old (direct) interpolation of pseudopot. !")
                if any(
                    getattr(potential, "old_interpolation", False)
                    for potential in self.problem.pseudopotentials.values()
                )
                else ()
            ),
            " No minimization!",
            "",
            # usrinputfile.F90, only for a run that asks for it.
            *(
                ()
                if getattr(self.translation, "polarizability_field", None) is None
                else (
                    " Polarizability data:",
                    " --------------------",
                    " Using seven-fields polarizability approach",
                    " Electric field  = "
                    f"{self.translation.polarizability_field:8.5f} [Ry/au]",
                    "",
                )
            ),
        ]
        self.write("\n".join(lines))

    def setup(self, system: "PreparedSinglePointSystem") -> None:
        # The grid and the atoms of the drivers whose results hold neither,
        # for the dipole after their SCF loop.
        self._system = system
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
                    # pseudo.f90 warns and goes on; CA and PBE refuse the file (but a CA file
                    # in a PBE run, as PARSEC 1.5's Tier_xc inputs have).
                    *(
                        ()
                        if pseudopotential_matches_functional(
                            potential, self.problem.scf.xc_functional
                        )
                        else (
                            " *** warning in pseudo : correlation potential "
                            "does not match",
                        )
                    ),
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
            # Only for a species with ldaplusu_u/ldaplusu_j.
            lines.extend(hubbard_parameter_lines(specification))
            lines.append("")
        lines.extend(
            [
                f" Tot. number of atoms = {len(system.atoms):7d}",
                "",
                *point_charge_report_lines(
                    getattr(self.problem, "point_charges", ()),
                    getattr(system, "point_charges", ()),
                    getattr(self.translation, "point_charge_type_counts", ()),
                ),
                *charged_sheet_report_lines(getattr(self.problem, "charged_sheets", ())),
                " Real-space setup:",
                " -----------------",
                f" Full active grid points = {system.grid.size:12d}",
                f" Hamiltonian dimension   = {system.grid.size:12d}",
                # parsec.F90:687-691, where Maximum_matvec acts (ARPACK, DIAGLA).
                *(
                    (
                        " Maximum number of matrix-vector operations = "
                        f"{self._matvec_budget(system):12d}",
                    )
                    if self.problem.eigensolver.method in {"arpack", "diagla"}
                    else ()
                ),
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
                # Only with LDA+U (upot.F90).
                *hubbard_setup_lines(system),
                # initial.f90:201-228, only for a slab it re-centred along z.
                *slab_recentring_lines(self.translation),
                "",
                # The symmetry block of symmetries.f90 (parsec.F90 l.464-492),
                # for the report: the space group the run uses where it has one
                # (Fortran's k-point path, Symmetrize_Charge_Density of a cell),
                # else the group PARSEC finds for it.
                *self._report_only(self._space_group_lines, system, "symmetry report"),
                # forceion.f90 / ewald_sum.f90 / ewald_slab.f90 at Output_Level >= 1.
                *self._report_only(self._ion_ion_force_lines, system, "ion-ion force table"),
                # Only a super cell or the grid of a general lattice.
                *(
                    periodic_geometry_lines(system, self.translation)
                    if self._periodic
                    else ()
                ),
                # irrbz.f90's table, for Fortran's k-point path.
                *self._kpoint_generation_lines(),
                # pot_local.f90, with Fortran's G-space local potential only.
                *gspace_local_potential_lines(system),
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

    @staticmethod
    def _report_only(build, system, name: str) -> list[str]:
        """Lines of a block that only reports; a failure there never stops the run."""

        try:
            return build(system)
        except Exception as error:  # noqa: BLE001 - a report must not end a run
            return ["", f" NOTE: the {name} of PARSEC is not printed: {type(error).__name__}: {error}", ""]

    def _space_group_lines(self, system=None) -> list[str]:
        """``symmetries``/``symgen`` lines of the run, as PARSEC prints them.

        ``parsec.F90`` l.464-492 decides what PARSEC computes
        (:func:`parsec_python.Grid.symmetry_analysis.symmetry_search_mode`):
        the group of the cell (a cluster: the cube of edge ``4 rmax``), its
        Abelian subgroup and ``symchk`` where the run would use them
        (``info = 0``), only the group (``info = 1``) or nothing
        (``info = 2``).  This port keeps every field on the full grid: the
        block is a report and changes no number.  A problem that carries a
        space group prints that one.
        """

        from ..Grid.symmetry_analysis import (
            abelian_rotation_flag,
            report_space_group,
            symmetry_search_mode,
        )
        from ..Grid.space_group import skipped_space_group

        group = getattr(self.problem, "space_group", None)
        if system is None:
            return [] if group is None else [*space_group_lines(group), ""]
        if self._crystal_report:
            # Andrew Grieder's routines find no space group of a crystal and print none
            return []
        problem = self.problem
        translation = self.translation
        cell = getattr(problem, "periodic_cell", None)
        grid = getattr(system, "grid", None)
        periodicity = 0 if cell is None else int(getattr(grid, "periodicity", 3) or 3)
        ignore = bool(getattr(translation, "ignore_symmetry", False))
        lattice = None
        shift = [0.0, 0.0, 0.0] if ignore else [0.5, 0.5, 0.5]
        if cell is not None:
            lattice = np.asarray(cell.lattice_vectors, dtype=float)
            # usrinputfile.F90 l.786-794: no shift along a periodic direction
            # that is not orthogonal to another periodic one.
            metric = lattice @ lattice.T
            for first in range(periodicity):
                for second in range(periodicity):
                    if first != second and abs(metric[first, second]) > 0.0:
                        shift[first] = 0.0
        requested = getattr(translation, "grid_shift", None)
        if requested is not None:
            # PARSEC 1.5's Grid_Shift block: the shift of the grid (the rule above without it)
            shift = [float(value) for value in problem.grid.shift]
        relaxation = getattr(translation, "relaxation", None)
        mode = symmetry_search_mode(
            ignore_symmetry=ignore,
            kpoints=bool(getattr(problem, "has_kpoint_sampling", False)),
            grid_shift=shift,
            periodicity=periodicity,
            spinors=bool(
                problem.scf.self_consistent_spin_orbit
                or problem.scf.noncollinear
                or any(
                    getattr(specification, "spin_orbit", False)
                    for specification in problem.pseudopotentials.values()
                )
            ),
            bands=getattr(problem, "bands", None) is not None,
            nscf=getattr(translation, "nscf", None) is not None,
            molecular_dynamics=getattr(translation, "molecular_dynamics", None) is not None,
            polarizability=getattr(translation, "polarizability_field", None) is not None,
            manual_movement=relaxation is not None and getattr(relaxation, "method", "") == "manual",
            non_spherical_cluster=cell is None
            and getattr(problem.grid, "domain_shape", "sphere") != "sphere",
        )
        level = int(getattr(translation, "output_level", 1))
        if group is not None and (group.ignored or not group.crystal_system):
            return [*space_group_lines(group), ""]
        if mode.info == 2:
            from ..Grid.space_group import identity_space_group

            shown = identity_space_group() if mode.suppressed else skipped_space_group()
            return [*space_group_lines(shown, domain_warning=mode.domain_warning), ""]
        if cell is None:
            from .parsec_dat import cluster_alatt, grid_rmax

            search_lattice = cluster_alatt(grid_rmax(grid))
        elif periodicity < 3:
            from ..Grid.space_group import confined_symmetry_lattice
            from .parsec_dat import grid_rmax

            search_lattice = confined_symmetry_lattice(lattice, periodicity, grid_rmax(grid))
        else:
            search_lattice = lattice
        if group is None:
            try:
                group = report_space_group(search_lattice, problem.atoms if cell is not None else system.atoms)
            except ValueError as error:
                return ["", " Symmetry Properties:", " --------------------", f" NOTE: {error}", ""]
        norot = abelian_rotation_flag(search_lattice, mode.info)
        return [*space_group_lines(group, info=mode.info, norot=norot, output_level=level), ""]

    def _ion_ion_force_lines(self, system) -> list[str]:
        """The ion-ion forces PARSEC prints during its setup (``ipr >= 1``).

        ``forceion.f90`` (a cluster, point charges included as ``forcept.f90``
        adds them), ``ewald_sum.f90`` (a bulk cell) and the wire/slab sums
        (``forceion_wire.f90``, ``ewald_slab.f90``): format 22 and rows of
        format 20, positions and forces in bohr and Ry/bohr.  A report: the
        forces of a run come from :mod:`parsec_python.Forces`.
        """

        if int(getattr(self.translation, "output_level", 1)) < 1 or self._crystal_report:
            # (Andrew Grieder's routines print no setup table of a crystal's ion-ion forces)
            return []
        atoms = getattr(system, "atoms", None)
        potentials = getattr(system, "pseudopotentials", None)
        if not atoms or potentials is None:
            return []
        try:
            cell = getattr(self.problem, "periodic_cell", None)
            if getattr(getattr(system, "grid", None), "is_confined_periodic", False):
                forces = getattr(system, "ion_ion_forces", None)
                if forces is None:
                    return []
            elif cell is not None:
                from ..Forces.ion_ion import ewald_ion_ion_forces

                forces = ewald_ion_ion_forces(atoms, potentials, cell.lattice_vectors)
            else:
                from ..Forces.ion_ion import ion_ion_forces, point_charge_forces

                forces = ion_ion_forces(atoms, potentials)
                charges = getattr(system, "point_charges", ())
                if charges:
                    forces = forces + point_charge_forces(atoms, potentials, charges)
        except (ValueError, NotImplementedError):
            return []
        forces = np.asarray(forces, dtype=float)
        lines = [
            "",
            " Forces from ion-ion interaction:",
            " ================================",
            "-atom-  ----x----   ----y----   ----z-----   ----Fx----  ----Fy----   ----Fz---",
            "                       [bohr]                             [Ry/bohr]            ",
            "",
        ]
        for index, (atom, force) in enumerate(zip(atoms, forces), start=1):
            position = np.asarray(atom.position, dtype=float)
            # gfortran does not write the trailing 1x of format 20.
            lines.append(
                (
                    f"{index:4d}  "
                    + "".join(f"{value:11.6f} " for value in position)
                    + "  "
                    + "".join(f"{value:11.6f} " for value in force)
                ).rstrip()
            )
        lines.append("")
        return lines

    def _kpoint_generation_lines(self) -> list[str]:
        """``irrbz`` lines of Fortran's k-point path, nothing otherwise."""

        if getattr(self.problem, "monkhorst_pack_shift", None) is None:
            return []
        from ..Grid.kpoints import kpoint_sampling_details

        return [*kpoint_generation_lines(kpoint_sampling_details(self.problem)), ""]

    def _matvec_budget(self, system) -> int:
        """``solver%maxmv = max(Maximum_matvec, n_grid * States_Num)``."""

        from ..Eigensolvers import matvec_budget

        return matvec_budget(
            self.problem.eigensolver.maximum_matvec,
            system.grid.size,
            int(self.problem.scf.number_of_states or 0),
        )

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
        symmetrization = density_symmetrization_lines(
            getattr(step, "density_symmetrization", ())
        )
        if symmetrization:
            lines.extend([*symmetrization[1:], ""])
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
        # The ionic loops hand every iteration here; those of the spin-polarized
        # k-point and spin-orbit drivers (forces of their structures) have
        # reports of their own.
        if hasattr(step, "eigenvalues_up"):
            self.iteration_spin_polarized(step)
            return
        # (The spinor records also carry diagonalization_seconds now: test the type.)
        from ..models import ExchangeLoopIteration, SelfConsistentSOCIteration

        if isinstance(step, SelfConsistentSOCIteration):
            self.iteration_self_consistent_soc(step)
            return
        if isinstance(step, ExchangeLoopIteration):
            # An outer (Fock) iteration of a hybrid functional.
            from .hybrid_output import exchange_loop_lines

            self.write("\n".join(exchange_loop_lines(step)))
            return
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
            # LDA+U from the second iteration on (calcden.f90z, before the
            # eigenproblem).
            *hubbard_iteration_lines(step),
        ]
        counts = getattr(step, "eigensolver_counts", ())
        if counts:
            count_lines, self._cumulative_matvecs = iterative_eigensolver_count_lines(
                counts, self._cumulative_matvecs, len(counts) == 2
            )
            lines.extend([*count_lines, ""])
        lines.extend(
            [
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
        )
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
                    *density_symmetrization_lines(
                        getattr(step, "density_symmetrization", ())
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
                # LDA+U Energy where it is not zero (totnrg.f90).
                *energy_correction_lines(energies),
                # A hybrid functional: its exact exchange.
                *hybrid_energy_lines(energies),
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
        # parsec.F90:1307-1310, after getsre (Dynamic_Diag_Tol only).
        lines.extend(dynamic_tolerance_lines(step))
        self._previous_total = energies.total
        self.write("\n".join(lines))

    def test_mode_complete(self) -> None:
        """parsec.F90:1085-1089: ``Eigensolver: none`` stops after setup."""

        self.write(" TEST MODE COMPLETE")

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
        of the statement that forces are not calculated.  The results of the
        spin-polarized k-point and spin-orbit drivers go to their own reports.
        """
        if hasattr(result, "density_up"):
            self.finish_spin_polarized(result, elapsed_seconds, forces=forces)
            return
        if hasattr(result, "spinors"):
            self.finish_self_consistent_soc(result, elapsed_seconds, forces=forces)
            return
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            # The TS report of parsec.F90 after convergence (Van_Der_Waals).
            *dispersion_report_lines(result),
            # A hybrid functional: the energies of the Fock loop.
            *hybrid_finish_lines(result),
            status,
            "",
            (
                "Time for self-consistent field [sec] : "
                f"{elapsed_seconds:10.2f},"
                f"    tdiag_sum = {self._diagonalization_total:11.2f}"
            ),
            f"Time spent on Hartree potential [sec] : {self._hartree_total:11.2f}",
            *self._dipole_lines(
                getattr(result, "density", None),
                getattr(result, "grid", None),
                getattr(result, "atoms", ()),
                getattr(result, "pseudopotentials", None),
            ),
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
            xc_timing_name = XC_DISPLAY_NAMES.get(
                self.problem.scf.xc_functional, self.problem.scf.xc_functional.upper()
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
                    (
                        "Forces, dipoles, and MPI statistics"
                        if self._bulk
                        else "Forces and MPI statistics"
                    ),
                    "are not calculated by this Python single-point implementation.",
                    "",
                    " =================================================================",
                ]
            )
        else:
            lines.extend(
                [
                    *force_report_lines(
                        forces, self.translation.output_level, periodic=self._crystal_report
                    ),
                    f"Time for forces [sec] : {forces.seconds:11.2f}",
                    "",
                    (
                        "Dipoles and MPI statistics"
                        if self._bulk
                        else "MPI statistics"
                    ),
                    "are not calculated by this Python implementation.",
                    "",
                    " =================================================================",
                ]
            )
        self.write("\n".join(lines))

    def _dipole_lines(self, density, grid, atoms, pseudopotentials) -> list[str]:
        """The block of ``dipole.f90`` for an isolated system, else nothing.

        ``parsec.F90`` calls ``dipole`` after every SCF loop of a system
        that is not periodic in three dimensions.  The moment is kept in
        ``last_dipole`` for the archive.  Nothing is printed where the
        density is not on the full grid of ``grid``.
        """

        self.last_dipole = None
        if self._bulk or density is None or grid is None or not atoms:
            return []
        coordinates = getattr(grid, "coordinates", None)
        density = np.asarray(density)
        if (
            pseudopotentials is None
            or coordinates is None
            or density.ndim != 1
            or density.shape[0] != np.asarray(coordinates).shape[0]
        ):
            return []
        self.last_dipole = cluster_dipole_moment(density, grid, atoms, pseudopotentials)
        return ["", *dipole_report_lines(self.last_dipole, self.problem.scf.net_charge)]

    def _extended_driver_dipole_lines(self, density, atoms) -> list[str]:
        """:meth:`_dipole_lines` with the grid of the prepared system."""

        system = getattr(self, "_system", None)
        return self._dipole_lines(
            density,
            getattr(system, "grid", None),
            atoms,
            getattr(system, "pseudopotentials", None),
        )

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
            # LDA+U from the second iteration on (calcden.f90z).
            *hubbard_iteration_lines(step),
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
        # Symmetrize_Charge_Density of a periodic cell (isp = 1, 2, 3).
        symmetrization = density_symmetrization_lines(
            getattr(step, "density_symmetrization", ())
        )
        if symmetrization:
            lines.extend([*symmetrization[1:], ""])
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
                # LDA+U Energy where it is not zero (totnrg.f90).
                *energy_correction_lines(energies),
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
                # Dynamic_Diag_Tol only (parsec.F90:1307-1310).
                *dynamic_tolerance_lines(step),
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
        self,
        result: "SpinPolarizedSinglePointResult",
        elapsed_seconds: float,
        *,
        forces=None,
    ) -> None:
        """Report the end of a periodic spin-polarized SCF loop.

        ``forces`` (a wire, slab or bulk cell with k-points) prints the force
        tables in place of the statement that forces are not calculated.
        """
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            # The TS report of parsec.F90 after convergence (Van_Der_Waals).
            *dispersion_report_lines(result),
            # A hybrid functional: the energies of the Fock loop.
            *hybrid_finish_lines(result),
            status,
            "",
            f"Time for self-consistent field [sec] : {elapsed_seconds:10.2f}",
            *self._extended_driver_dipole_lines(
                np.asarray(result.density_up) + np.asarray(result.density_down),
                result.atoms,
            ),
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
            *self._closing_lines(forces),
        ]
        self.write("\n".join(lines))

    def _closing_lines(self, forces) -> list[str]:
        """The end of the report of an extended driver's SCF loop.

        Without forces the statement that they are not calculated (as
        before); with forces their tables, as :meth:`finish` prints them.
        """

        if forces is None:
            return [
                (
                    ("Forces, dipoles, and MPI statistics" if self._bulk else "Forces and MPI statistics")
                    + " are not calculated by this Python single-point implementation."
                ),
                "",
                " =================================================================",
            ]
        return [
            *force_report_lines(
                forces, self.translation.output_level, periodic=self._crystal_report
            ),
            f"Time for forces [sec] : {forces.seconds:11.2f}",
            "",
            "Dipoles and MPI statistics" if self._bulk else "MPI statistics",
            "are not calculated by this Python implementation.",
            "",
            " =================================================================",
        ]

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
        moments = getattr(step, "magnetic_moments", None)
        if moments is not None:
            # bxc.F90 after flevel: the moments of the new spinors.
            lines.extend(noncollinear_moment_lines(moments))
        # Symmetrize_Charge_Density of a periodic spin-orbit run.
        lines.extend(
            density_symmetrization_lines(getattr(step, "density_symmetrization", ()))
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
                # LDA+U Energy where it is not zero (totnrg.f90).
                *energy_correction_lines(energies),
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
                # Dynamic_Diag_Tol only (parsec.F90:1307-1310).
                *dynamic_tolerance_lines(step),
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
        self,
        result: "SelfConsistentSOCResult",
        elapsed_seconds: float,
        *,
        forces=None,
    ) -> None:
        status = (
            "Self-consistency convergence achieved."
            if result.converged
            else "Maximum SCF iterations reached without convergence."
        )
        lines = [
            # The TS report of parsec.F90 after convergence (Van_Der_Waals).
            *dispersion_report_lines(result),
            # A hybrid functional: the energies of the Fock loop.
            *hybrid_finish_lines(result),
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
                *self._extended_driver_dipole_lines(
                    getattr(result, "density", None), result.atoms
                ),
                "",
                f"Total Energy = {result.energies.total:22.8f} [Ry]",
                (
                    "Energy/atom  = "
                    f"{result.energies.total * RYDBERG_TO_EV / max(len(result.atoms), 1):22.8f} [eV]"
                ),
                "",
                *self._closing_lines(forces),
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
