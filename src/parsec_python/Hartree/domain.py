"""Default of the boundary sphere and of the tolerance of its Hartree values.

A cluster calculation confines the orbitals to a sphere of radius ``R`` and
takes the Hartree potential on it from a multipole expansion.  Both raise
the total energy: the wall by cutting the tails of the orbitals, the
expansion by what it omits.  ``Domain_Energy_Tolerance`` (``B``, 1e-3 Ry
unless the input says otherwise) is the estimated sum of the two, half
each.  This module chooses ``R`` and the tolerance of the expansion from it
where the input leaves the radius out, and estimates both parts for every
sphere, also one the input gives.

Before the SCF the wall is estimated from the geometry and the valence
density ``rho_s`` of each species' own free atom, as its pseudopotential
file holds it:

``W(R) = sum_a  integral F_a(t; R) * 4 * 4 pi t**2 (d sqrt(rho_s)/dt)**2 dt``

in Ry, where ``F_a`` is the part of the shell of radius ``t`` about the atom
``a`` that lies outside the sphere.  ``4 |grad sqrt(rho)|**2`` is the energy
a hard wall adds per electron it cuts off, to first order, for a density
that decays like the free atom's.  No constant of ``W`` belongs to a
species.  ``W`` is an estimate and not a bound: it is several times too
large for hydrogen-terminated carbon and too small where the highest level
of the system lies above that of its atoms, as in clusters of closed-shell
metal atoms, and it knows nothing of electrons no free atom holds.

The radius of the rule is the smallest multiple of 0.1 angstrom with at
least 3 angstrom of vacuum beyond the outermost atom and ``W(R) <= B/2``.
The tolerance of the expansion follows from the electron count ``N_e``:
with the atomic tail, the energy the boundary values leave was measured at
or below ``K sqrt(N_e) e(L)`` on hydrogen-terminated clusters of 176 to
23,768 electrons, ``e(L)`` being the estimate of
:func:`~parsec_python.Hartree.boundary.estimate_omitted_potential` at the
order in use, so ``tau = (B/2) / (K sqrt(N_e))`` keeps that part within its
half.  Where no order up to the largest meets ``tau`` the radius grows.

After a converged SCF the wall is estimated again from the density itself.
An orbital that vanishes on the sphere is ``sinh(kappa s)`` at the distance
``s`` inside it, so the charge of thin shells follows
``S(s) = G sinh(kappa s)**2 / kappa**2`` with ``G = -dE/dR`` (Hadamard),
and the wall adds about ``G / (2 kappa)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
import math
from time import perf_counter

import numpy as np

from ..models import MAXIMUM_MULTIPOLE_ORDER
from .boundary import estimate_omitted_potential, valence_point_charges


RULE_VERSION = 1
# Default of Domain_Energy_Tolerance in Ry, and the part of it the wall gets.
# The boundary values get the rest.
DOMAIN_ENERGY_TOLERANCE = 1.0e-3
WALL_SHARE = 0.5
# Bohr per angstrom as the input reader takes "ang" (PARSEC's ESDF table), so
# that a radius of the rule is the number its text reads back to.
ANGSTROM = 1.0 / 0.529177
# The radius is a multiple of this many angstrom, with at least this much
# vacuum beyond the outermost atom.
RADIUS_LATTICE = Decimal("0.1")
MINIMUM_VACUUM = 3.0
# Only atoms this close (angstrom) to the outermost one enter W.
WALL_REACH = 7.0
# A density is used where its integral is within this fraction of the
# occupations of its file.  The atoms of a species without one get this much
# vacuum (angstrom): a guess, the largest single-atom value of H to Kr.
TABLE_COMPLETENESS = 0.02
NO_TABLE_VACUUM = 7.5
# Electrons no free-atom density holds: the vacuum v becomes
# max(1.6 v, v + 2.5 angstrom).
EXTRA_ELECTRON_FACTOR = 1.6
EXTRA_ELECTRON_VACUUM = 2.5
# Energy left by the boundary values over sqrt(N_e) e(L): the envelope of the
# measured ratios from 4 angstrom of vacuum, below it, and their typical
# value; and over sqrt(N_e) e(Solver_Lpole) for PARSEC's boundary values.
BOUNDARY_CONSTANT = 6.25e-3
BOUNDARY_CONSTANT_THIN = 1.25e-2
THIN_VACUUM = 4.0
BOUNDARY_TYPICAL = 2.0e-3
BOUNDARY_NOT_ENGAGED = 5.0e-2
LARGEST_BOUNDARY_TOLERANCE = 1.0e-3
# Shells of the density inside the sphere (bohr): their width, the window of
# the fit, and the decay constants it scans (1/bohr).
SHELL_WIDTH = 0.4
FIT_WINDOW = (0.8, 4.0)
_DECAY_SCAN = np.linspace(0.15, 2.0, 371)
ROUGH_FIT_RMS = 0.3
# The radius after the SCF aims at the wall share over this factor and moves
# by at most this many angstrom.
RADIUS_SAFETY = 1.5
RADIUS_STEP_LIMIT = 2.5
_SHELL_BLOCK_ROWS = 1 << 20
_WALL_BLOCK_ATOMS = 2000


def _bohr(value: Decimal) -> float:
    return float(value) * ANGSTROM


def lattice_radius(radius: float, lattice: Decimal = RADIUS_LATTICE) -> Decimal:
    """Return the smallest lattice value (angstrom) at or above ``radius`` (bohr).

    A relative 1e-9 is taken off before the ceiling, so a sum that is a
    lattice value stays on it.  ``radius`` may be a NumPy scalar, whose own
    ``repr`` is no number from NumPy 2 on.
    """

    value = Decimal(repr(float(radius) / ANGSTROM * (1.0 - 1.0e-9)))
    return (value / lattice).to_integral_value(rounding=ROUND_CEILING) * lattice


def _quadrature_weights(radii: np.ndarray) -> np.ndarray:
    """Weights of :func:`~parsec_python.Pseudopotential.parsec_radial_integral`."""

    weights = np.empty_like(radii)
    weights[0] = 0.5 * radii[1]
    weights[1:-1] = 0.5 * (radii[2:] - radii[:-2])
    weights[-1] = radii[-1] - radii[-2]
    return weights


@dataclass(frozen=True)
class FreeAtomTable:
    """Valence density of one free atom, as the wall estimate reads it.

    ``wall`` is ``4 * 4 pi r**2 (d sqrt(rho)/dr)**2`` on ``radii`` times the
    weights of the radial quadrature, in Ry.  ``source`` names what the
    density came from, ``charge`` is the sum of the occupations of the file,
    the electrons the density holds, and ``integral`` what its quadrature
    below the interpolation cutoff gives.
    """

    radii: np.ndarray
    wall: np.ndarray
    source: str
    charge: float
    integral: float


def free_atom_tables(
    pseudopotentials,
) -> tuple[dict[str, FreeAtomTable], dict[str, str]]:
    """Return the free-atom table of every species, or why it has none.

    The valence density of the file is taken below its interpolation cutoff,
    where the code itself reads it.  It is used only if complete: its
    integral must be within ``TABLE_COMPLETENESS`` of the sum of the file's
    own occupations.  A table that fails gives way to
    ``sum_l f_l u_l**2 / (4 pi r**2)`` of the pseudo-wavefunctions under the
    same test.  A file generated for an ion passes with the charge of that
    ion, which is the charge the table then holds.
    """

    tables: dict[str, FreeAtomTable] = {}
    reasons: dict[str, str] = {}
    for symbol, potential in pseudopotentials.items():
        radii = np.asarray(potential.radii, dtype=np.float64)
        occupations = dict(potential.channel_occupations)
        charge = float(
            sum(occupations.values()) if occupations else potential.ionic_charge
        )
        below_cutoff = (radii > 0.0) & (radii < potential.interpolation_cutoff)
        shell_area = 4.0 * np.pi * radii * radii
        candidates = [
            (
                "valence table",
                shell_area * np.asarray(potential.valence_density, dtype=np.float64),
            )
        ]
        if occupations and sum(occupations.values()) > 0:
            candidates.append(
                (
                    "pseudo-wavefunctions",
                    sum(
                        occupation
                        * np.asarray(
                            potential.radial_wavefunctions[angular_momentum],
                            dtype=np.float64,
                        )
                        ** 2
                        for angular_momentum, occupation in occupations.items()
                    ),
                )
            )
        seen = []
        for source, shell in candidates:
            if np.count_nonzero(below_cutoff) < 2:
                break
            integral = float(
                _quadrature_weights(radii[below_cutoff]) @ shell[below_cutoff]
            )
            seen.append(f"{source} {integral:.4f} e")
            if charge > 0 and abs(integral - charge) <= TABLE_COMPLETENESS * charge:
                density = shell / np.where(radii > 0.0, shell_area, 1.0)
                keep = below_cutoff & (density > 1.0e-200)
                kept = radii[keep]
                slope = np.gradient(np.sqrt(density[keep]), kept)
                tables[symbol] = FreeAtomTable(
                    radii=kept,
                    wall=_quadrature_weights(kept)
                    * 4.0
                    * 4.0
                    * np.pi
                    * kept
                    * kept
                    * slope
                    * slope,
                    source=source,
                    charge=charge,
                    integral=integral,
                )
                break
        else:
            reasons[symbol] = (
                f"no complete free-atom density ({'; '.join(seen)}; occupations "
                f"{charge:g} e, table ends at "
                f"{potential.interpolation_cutoff:.2f} bohr)"
            )
        if symbol not in tables and symbol not in reasons:
            reasons[symbol] = "no free-atom density below the interpolation cutoff"
    return tables, reasons


def wall_energy(
    positions: np.ndarray,
    symbols,
    tables: dict[str, FreeAtomTable],
    radius: float,
) -> tuple[float, dict[str, float]]:
    """Return ``W(radius)`` in Ry and its parts by species.

    Only the atoms within ``WALL_REACH`` of the outermost one are summed,
    and of those the ones whose table reaches the sphere.  Atoms at the same
    distance from the centre, to 1e-6 bohr, share one integral.  The sums
    are NumPy's own, whose bits do not depend on the threads of a BLAS.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    symbols = np.asarray(symbols)
    distance = np.linalg.norm(positions, axis=1)
    reach = distance.max() - WALL_REACH * ANGSTROM
    by_species: dict[str, float] = {}
    for symbol, table in tables.items():
        chosen = distance[symbols == symbol]
        chosen = chosen[(radius - chosen < table.radii[-1]) & (chosen > reach)]
        if not chosen.size:
            continue
        unique, counts = np.unique(np.round(chosen, 6), return_counts=True)
        shell = table.radii[None, :]
        total = 0.0
        for start in range(0, unique.size, _WALL_BLOCK_ATOMS):
            atom = unique[start : start + _WALL_BLOCK_ATOMS, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                cosine = (radius * radius - atom * atom - shell * shell) / (
                    2.0 * atom * shell
                )
            outside = np.clip(0.5 * (1.0 - cosine), 0.0, 1.0)
            # An atom at the centre: the whole shell from the radius on.
            outside = np.where(atom > 1.0e-9, outside, (shell >= radius) * 1.0)
            total += float(
                (
                    counts[start : start + _WALL_BLOCK_ATOMS]
                    * (outside * table.wall[None, :]).sum(axis=1)
                ).sum()
            )
        by_species[symbol] = total
    return float(sum(by_species.values())), by_species


def boundary_constant(vacuum: float) -> float:
    """Return ``K`` of the boundary energy for a vacuum width in bohr."""

    return (
        BOUNDARY_CONSTANT
        if vacuum / ANGSTROM >= THIN_VACUUM - 1.0e-9
        else BOUNDARY_CONSTANT_THIN
    )


def auto_boundary_tolerance(electrons: float, budget: float, vacuum: float) -> float:
    """Return the Hartree boundary tolerance of the rule in Ry.

    ``min(1e-3, ((1 - WALL_SHARE) budget) / (K sqrt(electrons)))``, cut to
    two significant digits towards the tighter value, with ``K`` of the
    vacuum width (bohr).
    """

    value = min(
        LARGEST_BOUNDARY_TOLERANCE,
        (1.0 - WALL_SHARE)
        * budget
        / (boundary_constant(vacuum) * math.sqrt(electrons)),
    )
    exponent = math.floor(math.log10(value))
    mantissa = math.floor(value / 10.0**exponent * 10.0 + 1.0e-9) / 10.0
    return float(f"{mantissa:.1f}e{exponent:d}")


def series_bound(
    positions: np.ndarray, charges: np.ndarray, radius: float, order: int
) -> float:
    """Bound what the expansion of the point charges omits above ``order`` (Ry).

    ``sum_a (2 q_a / R) (r_a/R)**(order+1) / (1 - r_a/R)``: the geometric
    series of every charge in its own direction, where it peaks.  It is 23
    to 29 times the estimate of the benchmark clusters, which is enough to
    spare small systems the estimate.
    """

    ratio = np.linalg.norm(np.asarray(positions, dtype=np.float64), axis=1) / radius
    if ratio.size and ratio.max() >= 1.0:
        return math.inf
    charges = np.asarray(charges, dtype=np.float64)
    return float(np.sum(2.0 * charges / radius * ratio ** (order + 1) / (1.0 - ratio)))


def _on_grid_shell(radius: float, spacing: float, shift: float) -> bool:
    """Whether ``radius**2`` is a value ``|r|**2`` of the grid points, to 1e-9.

    The points of the shifted grid have ``|r|**2 / h**2 = 2 k + 3/4`` and
    those of a grid through the origin an integer.  The grid builders decide
    such points by round-off.
    """

    squared = (radius / spacing) ** 2
    if abs(shift - 0.5) < 1.0e-12:
        nearest = 2.0 * round((squared - 0.75) / 2.0) + 0.75
    else:
        nearest = float(round(squared))
    return abs(squared - nearest) / squared < 1.0e-9


def _settle_radius(
    value: Decimal,
    set_by: str,
    positions: np.ndarray,
    charges: np.ndarray,
    spacing: float,
    shift: float,
    tolerance: float | None,
) -> tuple[Decimal, str, int]:
    """Grow a lattice radius until the largest order meets the tolerance and
    no grid point lies on the sphere.

    Returns the radius, what set it, and the estimates that were made.  With
    ``tolerance=None`` the boundary values are exact and only the grid is
    looked at.
    """

    estimates = 0
    if tolerance is not None:

        def misses(candidate: Decimal) -> bool:
            nonlocal estimates
            radius = _bohr(candidate)
            if (
                series_bound(positions, charges, radius, MAXIMUM_MULTIPOLE_ORDER)
                <= tolerance
            ):
                return False
            estimates += 1
            return bool(
                estimate_omitted_potential(positions, charges, radius)[
                    MAXIMUM_MULTIPOLE_ORDER
                ]
                > tolerance
            )

        if misses(value):
            low, step = value, Decimal("0.5")
            high = value + step
            while misses(high):
                low = high
                step *= 2
                high = low + step
                if high - value > 200:
                    raise ValueError(
                        "no radius meets the Hartree boundary tolerance of "
                        f"{tolerance:.1e} Ry at the largest multipole order "
                        f"{MAXIMUM_MULTIPOLE_ORDER}"
                    )
            while high - low > RADIUS_LATTICE:
                middle = (
                    low
                    + ((high - low) / RADIUS_LATTICE / 2).to_integral_value()
                    * RADIUS_LATTICE
                )
                if misses(middle):
                    low = middle
                else:
                    high = middle
            value, set_by = high, "multipole order"
    base, step = value, RADIUS_LATTICE
    for _ in range(9):
        if not _on_grid_shell(_bohr(value), spacing, shift):
            break
        # The next lattice value, and half the step each time that one
        # coincides too, as every one does on an unshifted grid whose
        # spacing divides the lattice.
        value = base + step
        step /= 2
    else:
        raise ValueError(
            "no radius near the lattice keeps the grid points off the sphere"
        )
    if value != base:
        set_by += f" (+{value - base} ang: a grid point lay on the sphere)"
    return value, set_by, estimates


@dataclass(frozen=True)
class DomainChoice:
    """What the default rule chose for one calculation, and why.

    ``radius_text`` and ``tolerance_text`` are the values of the input lines
    ``Boundary_Sphere_Radius`` and ``Hartree_Boundary_Tolerance`` that give
    this domain again; ``radius`` (bohr) and ``boundary_tolerance`` (Ry) are
    what they read back to.  ``tolerance_text`` is ``None`` where the rule
    chose no tolerance: the input has one, or the boundary values are the
    direct Coulomb sum.  ``wall`` is ``W`` at the radius in Ry, with its
    parts by species, and ``wall_radius`` the radius (bohr) before the
    electrons no free atom holds stretched the vacuum.  ``tables`` holds
    species, source and charge of the free-atom densities,
    ``without_table`` the species that have none and why, and ``set_by``
    names the step of the rule that decided the radius.  ``estimates``
    counts the estimates of the omitted potential the rule had to make.
    """

    rule_version: int
    budget: float
    radius_text: str
    radius: float
    tolerance_text: str | None
    boundary_tolerance: float | None
    set_by: str
    outermost_symbol: str
    outermost_distance: float
    electrons: float
    wall: float
    wall_by_species: tuple[tuple[str, float], ...]
    tables: tuple[tuple[str, str, float], ...]
    without_table: tuple[tuple[str, str], ...]
    extra_electrons: float
    wall_radius: float
    estimates: int = 0
    seconds: float = 0.0

    @property
    def vacuum(self) -> float:
        """Vacuum beyond the outermost atom in bohr."""

        return self.radius - self.outermost_distance

    @property
    def wall_vacuum(self) -> float:
        """The vacuum in bohr before electrons no free atom holds widened it."""

        return self.wall_radius - self.outermost_distance

    @property
    def wall_share(self) -> float:
        """The part of the budget the wall may take, in Ry."""

        return WALL_SHARE * self.budget


def default_domain(
    atoms,
    pseudopotentials,
    electron_count: float,
    *,
    spacing: float,
    shift: float = 0.5,
    stencil_half_width: int = 6,
    budget: float = DOMAIN_ENERGY_TOLERANCE,
    boundary_tolerance: float | None = None,
    multipole_boundary: bool = True,
) -> DomainChoice:
    """Choose the sphere radius and the Hartree boundary tolerance.

    ``atoms`` are the atoms as the driver places them, after its recentring,
    and ``pseudopotentials`` the loaded files of their species.
    ``boundary_tolerance`` is the tolerance the input gives, ``None`` to
    have the rule choose it.  ``multipole_boundary=False`` is a calculation
    with the direct Coulomb sum, whose boundary values are exact: the rule
    then chooses the radius alone.

    The steps, in this order: the smallest lattice radius with the minimum
    vacuum whose wall estimate is within its share; more for the atoms of a
    species without a complete density; the stretch for electrons no free
    atom holds; the stencil; the tolerance of the electron count; a larger
    radius where the largest order misses it; and a step off the grid
    points.
    """

    started = perf_counter()
    if not (np.isfinite(budget) and budget > 0):
        raise ValueError("Domain_Energy_Tolerance must be positive")
    symbols = np.array([atom.symbol for atom in atoms])
    positions, charges = valence_point_charges(atoms, pseudopotentials, electron_count)
    distance = np.linalg.norm(positions, axis=1)
    outermost = int(np.argmax(distance))
    largest = float(distance[outermost])
    tables, reasons = free_atom_tables(
        {symbol: pseudopotentials[symbol] for symbol in sorted(set(symbols.tolist()))}
    )
    held = sum(
        tables[symbol].charge
        if symbol in tables
        else float(pseudopotentials[symbol].ionic_charge)
        for symbol in symbols.tolist()
    )
    extra = float(electron_count) - held
    target = WALL_SHARE * budget

    def wall(candidate: Decimal) -> float:
        return wall_energy(positions, symbols, tables, _bohr(candidate))[0]

    floor = lattice_radius(largest + MINIMUM_VACUUM * ANGSTROM)
    value = floor
    if wall(value) > target:
        low, high = value, value + 1
        while wall(high) > target:
            low, high = high, high + 1
            if high - value > 60:
                raise ValueError(
                    "the atomic densities do not decay: no default radius"
                )
        while high - low > RADIUS_LATTICE:
            middle = (
                low
                + ((high - low) / RADIUS_LATTICE / 2).to_integral_value()
                * RADIUS_LATTICE
            )
            if wall(middle) > target:
                low = middle
            else:
                high = middle
        value = high
    set_by = "atomic densities" if value > floor else "minimum vacuum"
    if reasons:
        farthest = max(float(distance[symbols == symbol].max()) for symbol in reasons)
        guess = lattice_radius(farthest + NO_TABLE_VACUUM * ANGSTROM)
        if guess > value:
            value = guess
            set_by = "no atomic density for " + ", ".join(sorted(reasons))
    wall_value = value
    if extra > 1.0e-6:
        vacuum = _bohr(value) - largest
        value = lattice_radius(
            largest
            + max(
                EXTRA_ELECTRON_FACTOR * vacuum,
                vacuum + EXTRA_ELECTRON_VACUUM * ANGSTROM,
            )
        )
        set_by = "electrons no free atom holds"
    stencil = largest + (stencil_half_width + 1) * spacing
    if _bohr(value) < stencil:
        value = lattice_radius(stencil)
        set_by = "stencil"
    tolerance_text = None
    if multipole_boundary and boundary_tolerance is None:
        boundary_tolerance = auto_boundary_tolerance(
            electron_count, budget, _bohr(value) - largest
        )
        tolerance_text = f"{boundary_tolerance:.1e} Ry"
    value, set_by, estimates = _settle_radius(
        value,
        set_by,
        positions,
        charges,
        spacing,
        shift,
        boundary_tolerance if multipole_boundary else None,
    )
    radius = _bohr(value)
    total, by_species = wall_energy(positions, symbols, tables, radius)
    return DomainChoice(
        rule_version=RULE_VERSION,
        budget=float(budget),
        radius_text=f"{value} ang",
        radius=radius,
        tolerance_text=tolerance_text,
        boundary_tolerance=boundary_tolerance if multipole_boundary else None,
        set_by=set_by,
        outermost_symbol=str(symbols[outermost]),
        outermost_distance=largest,
        electrons=float(electron_count),
        wall=total,
        wall_by_species=tuple(sorted(by_species.items())),
        tables=tuple(
            (symbol, table.source, table.charge)
            for symbol, table in sorted(tables.items())
        ),
        without_table=tuple(sorted(reasons.items())),
        extra_electrons=extra,
        wall_radius=_bohr(wall_value),
        estimates=estimates,
        seconds=perf_counter() - started,
    )


@dataclass(frozen=True)
class BoundaryEnergy:
    """Energy the boundary values of a plan leave, as far as it was calibrated.

    ``about`` and ``bound`` are in Ry and ``None`` where the plan lies
    outside the calibration, which ``reason`` then names.  ``exact`` says
    that the plan has no multipole boundary.
    """

    about: float | None
    bound: float | None
    reason: str
    exact: bool = False

    @property
    def calibrated(self) -> bool:
        return self.bound is not None


def boundary_energy(plan, electron_count: float, vacuum: float) -> BoundaryEnergy:
    """Estimate the energy the Hartree boundary values of ``plan`` leave.

    An engaged plan with the atomic tail: about
    ``BOUNDARY_TYPICAL sqrt(N_e) e(L)`` and at most ``K sqrt(N_e) e(L)``
    with ``K`` of the vacuum width (bohr).  PARSEC's values, which a plan
    keeps where its estimate is within a tenth of the tolerance: at most
    ``BOUNDARY_NOT_ENGAGED sqrt(N_e) e(Solver_Lpole)``.  Both were measured
    on hydrogen-terminated clusters and on no other surface.  A plan
    without tolerance or without the tail, one at the largest order, and
    one that is not engaged and has the tail all the same
    (``Hartree_Atomic_Tail: on``: neither PARSEC's values nor measured) has
    no number.
    """

    if plan is None:
        return BoundaryEnergy(None, None, "no plan of the boundary")
    if plan.estimates is None:
        # No estimate: the tolerance is off, or there is no multipole
        # boundary to estimate.
        if plan.status == "tolerance off":
            return BoundaryEnergy(None, None, plan.status)
        return BoundaryEnergy(0.0, 0.0, plan.status, exact=True)
    scale = math.sqrt(electron_count)
    if not plan.engaged:
        if plan.atomic_tail:
            return BoundaryEnergy(
                None, None, "atomic tail on a plan that is not engaged"
            )
        return BoundaryEnergy(
            None, BOUNDARY_NOT_ENGAGED * scale * plan.estimate_minimum, "not engaged"
        )
    if plan.cap_reached:
        return BoundaryEnergy(None, None, "largest multipole order reached")
    if not plan.atomic_tail:
        return BoundaryEnergy(None, None, "atomic tail off")
    return BoundaryEnergy(
        BOUNDARY_TYPICAL * scale * plan.estimate_order,
        boundary_constant(vacuum) * scale * plan.estimate_order,
        "engaged",
    )


def sphere_shell_sums(
    coordinates: np.ndarray,
    density: np.ndarray,
    radius: float,
    volume_element: float,
) -> np.ndarray:
    """Return the charge per bohr of the shells of the fit window.

    ``S_j = sum rho_i h**3 / SHELL_WIDTH`` over the grid points at a
    distance ``s = radius - |r_i|`` from the sphere with
    ``j SHELL_WIDTH <= s < (j + 1) SHELL_WIDTH``, for the shells up to the
    end of ``FIT_WINDOW``.  The grid is read in blocks, so the work arrays
    hold a few megabytes whatever its size.
    """

    count = int(round(FIT_WINDOW[1] / SHELL_WIDTH))
    sums = np.zeros(count, dtype=np.float64)
    innermost = max(radius - FIT_WINDOW[1], 0.0) ** 2
    for start in range(0, coordinates.shape[0], _SHELL_BLOCK_ROWS):
        block = coordinates[start : start + _SHELL_BLOCK_ROWS]
        squared = np.einsum("ij,ij->i", block, block)
        near = np.flatnonzero(squared > innermost)
        index = np.floor(
            (radius - np.sqrt(squared[near])) / SHELL_WIDTH
        ).astype(np.int64)
        inside = (index >= 0) & (index < count)
        sums += np.bincount(
            index[inside],
            weights=density[start : start + _SHELL_BLOCK_ROWS][near[inside]],
            minlength=count,
        )
    return sums * (volume_element / SHELL_WIDTH)


@dataclass(frozen=True)
class WallFit:
    """Fit of ``G sinh(kappa s)**2 / kappa**2`` to the shells near the sphere.

    ``energy = prefactor / (2 decay)`` is the estimate of what the wall adds
    in Ry, ``decay`` is ``kappa`` in 1/bohr and ``rms`` the root mean square
    of the fit in the logarithm.  ``rough`` says that the number is to be
    read as an order of magnitude.
    """

    prefactor: float
    decay: float
    energy: float
    rms: float
    at_scan_end: bool
    shells: int

    @property
    def rough(self) -> bool:
        return self.rms > ROUGH_FIT_RMS or self.at_scan_end


def fit_wall(shell_sums: np.ndarray) -> WallFit | None:
    """Fit the shells of :func:`sphere_shell_sums` inside ``FIT_WINDOW``.

    ``ln S_j = ln G + ln m_j(kappa)`` in least squares, with ``m_j`` the mean
    of ``(sinh(kappa s) / kappa)**2`` over the shell from ``a`` to ``b``,

    ``m_j = ((sinh 2 kappa b - sinh 2 kappa a) / (4 kappa) - (b - a) / 2)
            / (SHELL_WIDTH kappa**2)``.

    Returns ``None`` where fewer than three shells of the window hold
    charge.
    """

    sums = np.asarray(shell_sums, dtype=np.float64)
    lower = SHELL_WIDTH * np.arange(sums.size)
    chosen = (lower >= FIT_WINDOW[0] - 1.0e-9) & (sums > 0.0)
    if np.count_nonzero(chosen) < 3:
        return None
    lower = lower[chosen]
    upper = lower + SHELL_WIDTH
    logarithm = np.log(sums[chosen])
    best = None
    for decay in _DECAY_SCAN:
        shape = np.log(
            (
                (np.sinh(2.0 * decay * upper) - np.sinh(2.0 * decay * lower))
                / (4.0 * decay)
                - 0.5 * SHELL_WIDTH
            )
            / (decay * decay * SHELL_WIDTH)
        )
        offset = float(np.mean(logarithm - shape))
        rms = float(np.sqrt(np.mean((logarithm - shape - offset) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(decay), math.exp(offset))
    rms, decay, prefactor = best
    return WallFit(
        prefactor=prefactor,
        decay=decay,
        energy=prefactor / (2.0 * decay),
        rms=rms,
        at_scan_end=bool(
            decay <= _DECAY_SCAN[0] + 1.0e-9 or decay >= _DECAY_SCAN[-1] - 1.0e-9
        ),
        shells=int(np.count_nonzero(chosen)),
    )


def wall_from_density(
    coordinates: np.ndarray,
    density: np.ndarray,
    radius: float,
    volume_element: float,
) -> WallFit | None:
    """Estimate from a converged density the energy its sphere adds."""

    return fit_wall(sphere_shell_sums(coordinates, density, radius, volume_element))


@dataclass(frozen=True)
class RadiusForBudget:
    """The radius that would meet the wall share, from a fit after the SCF.

    ``set_by`` names what decided it: the density at the sphere, the minimum
    vacuum, the stencil, or the multipole order where the largest one misses
    the tolerance of the rule at a smaller radius.  ``wall_radius_text`` is
    the radius before that last step, the one the wall alone asks for.
    ``limit`` is ``"at least"`` where the step out reached
    ``RADIUS_STEP_LIMIT``, so that the radius is a lower bound, ``"limit"``
    where the step in did and nothing else held the radius, and empty
    otherwise.  ``decay`` is the constant the step was taken with, and
    ``estimates`` counts the estimates of the omitted potential.
    """

    radius_text: str
    radius: float
    tolerance_text: str | None
    boundary_tolerance: float | None
    set_by: str
    limit: str
    decay: float
    estimates: int = 0
    wall_radius_text: str = ""


def radius_for_budget(
    fit: WallFit,
    radius: float,
    atoms,
    pseudopotentials,
    electron_count: float,
    *,
    spacing: float,
    shift: float = 0.5,
    stencil_half_width: int = 6,
    budget: float = DOMAIN_ENERGY_TOLERANCE,
    highest_occupied: float | None = None,
    boundary_tolerance: float | None = None,
    multipole_boundary: bool = True,
) -> RadiusForBudget:
    """Return the radius at which the fitted wall energy meets its share.

    The energy falls like ``exp(-2 kappa x)`` with the wall moved out by
    ``x``, so ``R' = R + ln(RADIUS_SAFETY E / (B/2)) / (2 kappa)``.  A fit
    in a thin sphere finds too small a ``kappa``, hence that of the highest
    occupied level (Ry), ``sqrt(-e)``, where it is larger, and the limit of
    the step.  The radius keeps the minimum vacuum and the stencil, goes up
    to the lattice and gets the tolerance, the largest order and the step
    off the grid points of the rule before the SCF.

    ``boundary_tolerance`` is the tolerance the input gives.  The radius is
    then the wall's alone and no estimate of the omitted potential is made:
    the largest order is asked only for the tolerance of the rule, and what
    the orders make of the input's own is said by the set-up of a run.
    """

    decay = fit.decay
    if highest_occupied is not None and highest_occupied < 0.0:
        decay = max(decay, math.sqrt(-highest_occupied))
    step = math.log(RADIUS_SAFETY * fit.energy / (WALL_SHARE * budget)) / (2.0 * decay)
    limit = ""
    if step > RADIUS_STEP_LIMIT * ANGSTROM:
        step, limit = RADIUS_STEP_LIMIT * ANGSTROM, "at least"
    elif step < -RADIUS_STEP_LIMIT * ANGSTROM:
        step, limit = -RADIUS_STEP_LIMIT * ANGSTROM, "limit"
    positions, charges = valence_point_charges(atoms, pseudopotentials, electron_count)
    largest = float(np.linalg.norm(positions, axis=1).max())
    floor = lattice_radius(largest + MINIMUM_VACUUM * ANGSTROM)
    stepped = lattice_radius(radius + step)
    value = max(stepped, floor)
    set_by = "density at the sphere" if value > floor else "minimum vacuum"
    stencil = largest + (stencil_half_width + 1) * spacing
    if _bohr(value) < stencil:
        value = lattice_radius(stencil)
        set_by = "stencil"
    if value > stepped and limit == "limit":
        # The vacuum or the stencil held the radius, not the step.
        limit = ""
    wall_value = value
    tolerance_text = rule_tolerance = None
    if multipole_boundary and boundary_tolerance is None:
        rule_tolerance = boundary_tolerance = auto_boundary_tolerance(
            electron_count, budget, _bohr(value) - largest
        )
        tolerance_text = f"{boundary_tolerance:.1e} Ry"
    value, set_by, estimates = _settle_radius(
        value, set_by, positions, charges, spacing, shift, rule_tolerance
    )
    return RadiusForBudget(
        radius_text=f"{value} ang",
        radius=_bohr(value),
        tolerance_text=tolerance_text,
        boundary_tolerance=boundary_tolerance if multipole_boundary else None,
        set_by=set_by,
        limit=limit,
        decay=decay,
        estimates=estimates,
        wall_radius_text=f"{wall_value} ang",
    )


__all__ = [
    "ANGSTROM",
    "BOUNDARY_CONSTANT",
    "BOUNDARY_CONSTANT_THIN",
    "BOUNDARY_NOT_ENGAGED",
    "BOUNDARY_TYPICAL",
    "BoundaryEnergy",
    "DOMAIN_ENERGY_TOLERANCE",
    "DomainChoice",
    "FIT_WINDOW",
    "FreeAtomTable",
    "MINIMUM_VACUUM",
    "RADIUS_LATTICE",
    "RULE_VERSION",
    "RadiusForBudget",
    "SHELL_WIDTH",
    "WALL_SHARE",
    "WallFit",
    "auto_boundary_tolerance",
    "boundary_constant",
    "boundary_energy",
    "default_domain",
    "fit_wall",
    "free_atom_tables",
    "lattice_radius",
    "radius_for_budget",
    "series_bound",
    "sphere_shell_sums",
    "wall_energy",
    "wall_from_density",
]
