"""A-priori control of the multipole boundary values of the Hartree potential.

The Dirichlet values of the Poisson problem come from a multipole expansion
of the density about the origin, truncated at the order ``L`` of
``Solver_Lpole``.  The expansion converges like ``(r_source/R)**l``, so what
it omits on the sphere of radius ``R`` grows with the cluster: the electrons
of an atom at ``0.86 R`` still contribute at ``l = 30``.  The local ionic
potential is summed exactly, atom by atom, so nothing cancels the omitted
part of the electrons' potential.

Most of that part is the potential of the valence charges placed at the
nuclei, which depends on the geometry alone.  With charges
``q_a = Z_ion,a * N_e / sum_b Z_ion,b`` at the positions ``R_a`` this module
builds, once per prepared system,

``e(L) = max_P | 2 sum_a q_a/|P-R_a| - M_L[q](P) |``     on the sphere,

the estimate of the omitted potential at every order ``L``, from which the
order in use is chosen, and the static *atomic tail* at the exterior stencil
points,

``C_L(P) = 2 sum_a q_a/|P-R_a| - M_L[q](P)``,

where ``M_L[q]`` is the order-``L`` multipole series of the point charges in
the convention of :class:`~parsec_python.Hartree.MultipoleExpansion`.  The
boundary values with the tail are ``V_B = M_L[rho] + C_L``: their error is
the part above ``L`` of the grid density minus the point atoms, and
``C_L -> 0`` as ``L`` grows.  The tail enters the Poisson right-hand side
linearly, as the rows ``t = -A_IB C_L`` on the grid points that have a
missing stencil neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import warnings

import numpy as np

from ..Grid import RealSpaceGrid
from ..Laplacian import apply_negative_laplacian_boundary, neighbor_shells
from ..models import MAXIMUM_MULTIPOLE_ORDER, GridSettings, HartreeSettings
from .harmonics import _positive_m_harmonic_rows
from .poisson import MultipoleExpansion


# Orders the estimate covers, directions of its Fibonacci sphere, and the
# shell (bohr) of outermost atoms whose own directions are added, where the
# omitted terms peak.
ESTIMATE_MAXIMUM_ORDER = MAXIMUM_MULTIPOLE_ORDER
_ESTIMATE_DIRECTIONS = 2000
_OUTER_ATOM_SHELL = 2.0
_OUTER_ATOM_LIMIT = 2000
# Pairs in one block of the Coulomb sum of the estimate.  The work arrays of
# a block of this size stay in the cache of a core, which makes the sum over
# the sample points three to four times faster than blocks of 2e6 pairs.
_ESTIMATE_CHUNK_PAIRS = 1 << 15
# An estimate at Solver_Lpole below this fraction of the tolerance leaves
# PARSEC's boundary as it is.
ENGAGEMENT_FRACTION = 0.1


def valence_point_charges(
    atoms,
    pseudopotentials,
    electron_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positions and charges of the valence electrons at the nuclei.

    The charges are the ionic valence charges scaled to ``electron_count``,
    so a net cluster charge is spread over the atoms in proportion, as the
    normalization of the superposed atomic densities does.
    """

    positions = np.array(
        [atom.position for atom in atoms], dtype=np.float64
    ).reshape(-1, 3)
    ionic = np.array(
        [pseudopotentials[atom.symbol].ionic_charge for atom in atoms],
        dtype=np.float64,
    )
    return positions, ionic * (float(electron_count) / float(ionic.sum()))


def point_charge_potential(
    points: np.ndarray,
    positions: np.ndarray,
    charges: np.ndarray,
    *,
    chunk_pairs: int = 2_000_000,
) -> np.ndarray:
    """Evaluate ``2 sum_a q_a/|P-R_a|`` (Ry) from coordinate differences."""

    points = np.asarray(points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    result = np.empty(points.shape[0], dtype=np.float64)
    x, y, z = (np.ascontiguousarray(positions[:, axis]) for axis in range(3))
    step = max(1, chunk_pairs // max(1, positions.shape[0]))
    with np.errstate(divide="ignore"):
        for start in range(0, points.shape[0], step):
            block = points[start : start + step]
            distance = (block[:, 0:1] - x[None, :]) ** 2
            distance += (block[:, 1:2] - y[None, :]) ** 2
            distance += (block[:, 2:3] - z[None, :]) ** 2
            np.sqrt(distance, out=distance)
            np.divide(1.0, distance, out=distance)
            result[start : start + step] = 2.0 * (distance @ charges)
    return result


def point_charge_multipoles(
    positions: np.ndarray,
    charges: np.ndarray,
    order: int,
) -> MultipoleExpansion:
    """Return ``Q_lm = sum_a q_a |R_a|**l conj(Y_lm(R_a))`` up to ``order``.

    These are the exact moments of any spherical charges ``q_a`` centred at
    ``R_a``.  A charge at the origin enters ``l = 0`` only.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64)
    moments: dict[tuple[int, int], complex] = {}
    for angular_momentum, magnetic, harmonic, radius in _positive_m_harmonic_rows(
        positions, order
    ):
        moment = complex(
            np.sum(charges * radius**angular_momentum * np.conjugate(harmonic))
        )
        moments[(angular_momentum, magnetic)] = moment
        if magnetic:
            moments[(angular_momentum, -magnetic)] = (
                ((-1) ** magnetic) * np.conjugate(moment)
            )
    return MultipoleExpansion(order=order, moments=moments)


def _fibonacci_directions(count: int) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ],
        axis=1,
    )


def estimate_omitted_potential(
    positions: np.ndarray,
    charges: np.ndarray,
    radius: float,
    *,
    maximum_order: int = ESTIMATE_MAXIMUM_ORDER,
    directions: int = _ESTIMATE_DIRECTIONS,
) -> np.ndarray:
    """Return ``e(L)`` in Ry for ``L = 0..maximum_order``.

    ``e(L)`` is the largest difference, over sample points of the sphere of
    ``radius``, between the Coulomb potential of the point charges and their
    multipole series truncated at ``L``.  The sample points are a Fibonacci
    sphere plus the directions of the outermost atoms.  With the scaled
    moments ``Qs_lm = sum_a q_a (r_a/R)**l conj(Y_lm(R_a))`` the term of
    order ``l`` at the direction ``u`` is

    ``(8*pi / ((2l+1) R)) sum_{m>=0} w_m Re(Qs_lm Y_lm(u))``,

    with ``w_0 = 1`` and ``w_m = 2`` otherwise.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64)
    distance = np.linalg.norm(positions, axis=1)
    unit = _fibonacci_directions(directions)
    if distance.size:
        outer = np.flatnonzero(
            (distance > distance.max() - _OUTER_ATOM_SHELL) & (distance > 0.0)
        )
        if outer.size > _OUTER_ATOM_LIMIT:
            outer = outer[
                np.argsort(distance[outer], kind="stable")[-_OUTER_ATOM_LIMIT:]
            ]
        unit = np.concatenate(
            [unit, positions[outer] / distance[outer][:, None]]
        )
    exact = point_charge_potential(
        radius * unit, positions, charges, chunk_pairs=_ESTIMATE_CHUNK_PAIRS
    )

    scaled_radius = distance / radius
    powers: dict[int, np.ndarray] = {}
    moments = np.zeros((maximum_order + 1, maximum_order + 1), dtype=np.complex128)
    for angular_momentum, magnetic, harmonic, _ in _positive_m_harmonic_rows(
        positions, maximum_order
    ):
        if angular_momentum not in powers:
            powers[angular_momentum] = charges * scaled_radius**angular_momentum
        # Summed without the BLAS: its dot product starts threads from
        # 10,000 atoms up, which costs ten times the sum itself in each of
        # the 1,891 rows.
        moments[angular_momentum, magnetic] = np.conjugate(
            np.einsum("a,a->", powers[angular_momentum], harmonic)
        )

    terms = np.zeros((unit.shape[0], maximum_order + 1), dtype=np.float64)
    for angular_momentum, magnetic, harmonic, _ in _positive_m_harmonic_rows(
        unit, maximum_order
    ):
        factor = 8.0 * np.pi / ((2 * angular_momentum + 1) * radius)
        if magnetic:
            factor *= 2.0
        terms[:, angular_momentum] += factor * (
            moments[angular_momentum, magnetic] * harmonic
        ).real
    return np.abs(exact[:, None] - np.cumsum(terms, axis=1)).max(axis=0)


@dataclass(frozen=True)
class AtomicTail:
    """Static right-hand-side rows of the atomic tail of one grid.

    ``rows`` are the grid points with a missing stencil neighbour, in
    ascending order, and ``values`` the entries ``t = -A_IB C_L`` on them.
    ``order`` is the ``L`` the tail was built for: it complements the
    expansion of that order and no other.  ``maximum`` is ``max |C_L|`` over
    the exterior stencil points, in Ry.  ``values_from`` names what
    evaluated ``C_L`` there: host threads, or the device kernel of the
    accelerated path.
    """

    order: int
    rows: np.ndarray
    values: np.ndarray
    maximum: float
    seconds: float = 0.0
    values_from: str = "host threads"

    def full(self, size: int) -> np.ndarray:
        """Return the rows as a vector of the active grid."""

        vector = np.zeros(size, dtype=np.float64)
        vector[self.rows] = self.values
        return vector


def build_atomic_tail(
    grid: RealSpaceGrid,
    positions: np.ndarray,
    charges: np.ndarray,
    order: int,
) -> AtomicTail:
    """Build the atomic tail through the stencil walk of the reference path.

    ``C_L`` is handed to :func:`apply_negative_laplacian_boundary` as a
    boundary function of a zero source, which visits exactly the exterior
    points the multipole boundary values are evaluated at.
    """

    if grid.settings.domain_shape != "sphere":
        raise ValueError("the atomic tail belongs to a spherical multipole boundary")
    started = perf_counter()
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64)
    expansion = point_charge_multipoles(positions, charges, order)
    maximum = 0.0

    def tail(points: np.ndarray) -> np.ndarray:
        nonlocal maximum
        values = point_charge_potential(
            points, positions, charges
        ) - expansion.potential(points)
        if values.size:
            maximum = max(maximum, float(np.abs(values).max()))
        return values

    vector = apply_negative_laplacian_boundary(
        np.zeros(grid.size, dtype=np.float64), grid, tail
    )
    touched = np.zeros(grid.size, dtype=bool)
    for _axis, _shell, neighbor_rows, _points in neighbor_shells(grid):
        touched |= neighbor_rows < 0
    rows = np.flatnonzero(touched)
    return AtomicTail(
        order=int(order),
        rows=rows,
        values=vector[rows],
        maximum=maximum,
        seconds=perf_counter() - started,
    )


@dataclass(frozen=True)
class HartreeBoundaryPlan:
    """What a prepared system does at the Hartree boundary, and why.

    ``order`` is the multipole order in use and ``minimum_order`` the input
    ``Solver_Lpole``.  ``engaged`` says that the estimate at
    ``minimum_order`` exceeded ``ENGAGEMENT_FRACTION`` of the tolerance, in
    which case ``order`` is the smallest one whose estimate meets the
    tolerance; ``cap_reached`` says that none up to the largest order did.
    ``atomic_tail`` says that the tail is part of the boundary values.  With
    the order unchanged and no tail, the boundary is PARSEC's.  ``estimates``
    holds ``e(L)`` for every order where an estimate was made.
    """

    order: int
    minimum_order: int
    tolerance: float | None
    engaged: bool
    atomic_tail: bool
    status: str
    estimates: tuple[float, ...] | None = field(default=None, repr=False)
    seconds: float = 0.0
    cap_reached: bool = False

    @property
    def estimate_minimum(self) -> float | None:
        """``e(Solver_Lpole)`` in Ry, ``None`` without an estimate."""

        return None if self.estimates is None else self.estimates[self.minimum_order]

    @property
    def estimate_order(self) -> float | None:
        """``e(order)`` in Ry, ``None`` without an estimate."""

        return None if self.estimates is None else self.estimates[self.order]

    @property
    def legacy(self) -> bool:
        """Whether the boundary is the order-``Solver_Lpole`` expansion alone."""

        return self.order == self.minimum_order and not self.atomic_tail


def plan_hartree_boundary(
    settings: HartreeSettings,
    grid_settings: GridSettings,
    positions: np.ndarray,
    charges: np.ndarray,
) -> HartreeBoundaryPlan:
    """Decide the boundary of one calculation from its geometry.

    Nothing is estimated for a box domain or a direct Coulomb boundary,
    which are exact already, or when the tolerance is off: the order is then
    ``Solver_Lpole``.  Otherwise the estimate ``e(Solver_Lpole)`` is compared
    with the tolerance: at or below ``ENGAGEMENT_FRACTION`` of it the
    boundary stays PARSEC's.  Above it the order becomes the smallest one
    from ``Solver_Lpole`` up whose estimate is within the tolerance, or the
    largest order with a warning, and the atomic tail is applied unless
    ``atomic_tail`` is ``"off"``; ``atomic_tail="on"`` applies it whatever
    the estimate.

    Settings a plan has resolved before carry the ``Solver_Lpole`` they
    started from in ``minimum_multipole_order``.  The plan starts there
    again, so it is the same: tested at the raised order alone, an estimate
    within a tenth of the tolerance would drop the tail.
    """

    started = perf_counter()
    minimum = int(
        settings.multipole_order
        if settings.minimum_multipole_order is None
        else settings.minimum_multipole_order
    )
    multipole = grid_settings.domain_shape == "sphere" and (
        settings.boundary_method in {"auto", "multipole"}
    )

    def plan(engaged, status, estimates=None, order=minimum, cap_reached=False):
        return HartreeBoundaryPlan(
            order=order,
            minimum_order=minimum,
            tolerance=settings.boundary_tolerance,
            engaged=engaged,
            atomic_tail=multipole
            and (
                settings.atomic_tail == "on"
                or (settings.atomic_tail == "auto" and engaged)
            ),
            status=status,
            estimates=estimates,
            seconds=perf_counter() - started,
            cap_reached=cap_reached,
        )

    if not multipole:
        return plan(False, "direct Coulomb sum, no multipole boundary")
    if settings.boundary_tolerance is None:
        return plan(False, "tolerance off")
    estimates = tuple(
        float(value)
        for value in estimate_omitted_potential(
            positions, charges, float(grid_settings.radius)
        )
    )
    tolerance = settings.boundary_tolerance
    if not estimates[minimum] > ENGAGEMENT_FRACTION * tolerance:
        return plan(False, "estimate within a tenth of the tolerance", estimates)
    for order in range(minimum, ESTIMATE_MAXIMUM_ORDER + 1):
        if estimates[order] <= tolerance:
            return plan(True, "engaged", estimates, order)
    warnings.warn(
        "the multipole expansion of the Hartree boundary misses the tolerance "
        f"of {tolerance:.3e} Ry at its largest order {ESTIMATE_MAXIMUM_ORDER}: "
        f"the estimate of the omitted potential of the atoms is "
        f"{estimates[ESTIMATE_MAXIMUM_ORDER]:.3e} Ry "
        f"({estimates[minimum]:.3e} Ry at Solver_Lpole {minimum}); "
        "the domain leaves too little vacuum around the outermost atoms",
        RuntimeWarning,
        stacklevel=2,
    )
    return plan(
        True,
        "engaged, largest order reached",
        estimates,
        ESTIMATE_MAXIMUM_ORDER,
        cap_reached=True,
    )


__all__ = [
    "AtomicTail",
    "ENGAGEMENT_FRACTION",
    "ESTIMATE_MAXIMUM_ORDER",
    "HartreeBoundaryPlan",
    "build_atomic_tail",
    "estimate_omitted_potential",
    "plan_hartree_boundary",
    "point_charge_multipoles",
    "point_charge_potential",
    "valence_point_charges",
]
