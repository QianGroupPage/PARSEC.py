"""Fock exchange operator of real-space orbitals and its adaptively compressed form.

Conventions (those of :mod:`parsec_python.V_xc.exact_exchange`): orbitals are
the columns ``q_i = sqrt(h**3) psi_i`` of the solver's wavefunction array,
energies and potentials are in Rydberg, and the occupation ``f_i`` of an
orbital lies in ``[0, 1]`` per spin.  An unpolarized orbital set stands for
two spin channels with the same orbitals (``spin_weight = 2``, ``f = 1`` a
doubly occupied orbital), the orbitals of one channel of a spin-polarized
calculation for one (``spin_weight = 1``, ``f = 1`` one electron).

Exchange operator
-----------------
In Hartree units the exchange operator of spin channel ``s`` acts on an
orbital ``phi`` as

``(V_x phi)(r) = - sum_j f_j psi_j(r) integral K(r, r') psi_j(r') phi(r') dr'``,

``j`` over the occupied orbitals of the channel and ``K`` the exchange kernel
(``1/|r - r'|``, or a screened form, of the pair Poisson solver).  The pair
solver returns ``v[rho] = 2 integral K rho`` in Rydberg (the convention of the
Hartree potential, ``-nabla**2 v = 8 pi rho`` for the Coulomb kernel), so on
the grid, with ``rho_jk = q_j q_k / h**3``,

``(V_x q)_k = - sum_j f_j q_j v[q_j q / h**3]``                       (Ry).

The exchange energy of the channel set is

``E_x = -(w/2) sum_ij f_i f_j (ij|ij) = (w/2) sum_i f_i <q_i | V_x q_i>``  (Ry),

with ``w`` the spin weight; for ``w = 2`` this is
:func:`parsec_python.V_xc.exact_exchange.exact_exchange_energy`, and two
channels with ``w = 1`` give
:func:`~parsec_python.V_xc.exact_exchange.spin_exact_exchange_energy`.
``E_x`` is a quartic form of the orbitals; its gradient on the grid is

``dE_x / dq_i = 2 w f_i (V_x q_i)``

(Euler: ``sum_i q_i . dE_x/dq_i = 4 E_x``).  The band energy of a generalized
Kohn-Sham Hamiltonian ``H = H_semilocal + a_x V_x`` (``w sum_i f_i e_i``)
therefore holds ``2 a_x E_x`` of exchange, twice what the total energy has:
the double counting that :func:`exchange_band_energy` measures.

The pair potentials ``v[q_i q_j]`` of two occupied orbitals serve both
``V_x q_i`` and ``V_x q_j``: the occupied block needs ``n_occ (n_occ + 1)/2``
solves, an orbital without occupation ``n_occ`` more (Jing, Sharma, Pask and
Suryanarayana, arXiv:2501.16572, Sec. 2; the same count in SPARC's CPU code
is described in arXiv:2406.16998).

Adaptively compressed exchange (ACE)
------------------------------------
Lin, J. Chem. Theory Comput. 12, 2242 (2016), as used by SPARC
(arXiv:2406.16998, Sec. 2.2) with the signs and the triangular factor fixed
here (the printed SPARC formulas are inconsistent): with
``W = V_x Psi`` for the orbital block ``Psi``, ``V_x`` negative semidefinite,

``M = -Psi^T W = L L^T`` (Cholesky),   ``xi = W L^-T``,   ``V_ACE = -xi xi^T``.

``V_ACE Psi = -W M^-1 (W^T Psi) = W``: the compressed operator equals the
exchange operator on the span of ``Psi`` and costs two GEMMs per
application.  Off the span ``0 >= <phi|V_ACE|phi> >= <phi|V_x|phi>``, so the
compressed exchange energy of other orbitals is an upper bound.

The discrete pair solver need not be exactly symmetric (the multipole
boundary values of the open-boundary solve make ``v`` slightly
non-self-adjoint); ``M`` is symmetrized before the Cholesky factorization and
its asymmetry reported in :class:`ExchangeApplication`.

Numerical modules do not print or open files: everything is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy.linalg import solve_triangular

from .exact_exchange import OCCUPATION_THRESHOLD, occupied_columns


#: Pair densities solved together unless a caller chooses (``Exx_Batch_Size``).
DEFAULT_BATCH_SIZE = 32


class PairPoissonSolver(Protocol):
    """Open-boundary potentials of a batch of pair densities.

    ``solve_batch(densities)`` takes an array ``(grid points, nb)`` of
    densities in electrons/bohr**3 and returns their potentials ``(grid
    points, nb)`` in Rydberg, ``2 integral K(r, r') rho(r') dr'``.  It must be
    linear in each column.
    """

    def solve_batch(self, densities: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class CallablePairPoisson:
    """Batch interface around a solver of one pair density.

    ``solve_pair(rho)`` returns the potential as an array or as an object
    with ``potential`` and, optionally, ``iterations`` (the Hartree result of
    :func:`parsec_python.V_xc.exact_exchange.reference_pair_poisson`, or the
    device CG solver of :mod:`parsec_python.acceleration.Hartree.pair_poisson`).
    ``solves``, ``iterations`` and ``seconds`` count the work done so far.
    """

    def __init__(self, solve_pair: Callable[[np.ndarray], object], description: str = "") -> None:
        self._solve_pair = solve_pair
        self.description = description or "caller-supplied pair solver"
        self.solves = 0
        self.iterations = 0
        self.seconds = 0.0

    def solve_batch(self, densities: np.ndarray) -> np.ndarray:
        densities = np.asarray(densities, dtype=np.float64)
        if densities.ndim != 2:
            raise ValueError("pair densities must have shape (grid points, batch)")
        started = perf_counter()
        potentials = np.empty_like(densities)
        for column in range(densities.shape[1]):
            solved = self._solve_pair(np.ascontiguousarray(densities[:, column]))
            potential = np.asarray(getattr(solved, "potential", solved), dtype=np.float64)
            if potential.shape != (densities.shape[0],):
                raise ValueError("the pair potential does not match the grid")
            potentials[:, column] = potential
            self.iterations += int(getattr(solved, "iterations", 0) or 0)
        self.solves += densities.shape[1]
        self.seconds += perf_counter() - started
        return potentials


def reference_pair_solver(grid, negative_laplacian, hartree_settings) -> CallablePairPoisson:
    """Coulomb pair solver of :func:`~parsec_python.V_xc.exact_exchange.reference_pair_poisson`.

    Conjugate gradients on the full grid with the multipole (sphere) or
    direct (box) boundary values of each pair density at the order of
    ``hartree_settings``, without the atomic tail.  Slow (one CG solve per
    pair): meant for small molecules and as the reference of faster solvers.
    """

    from .exact_exchange import reference_pair_poisson

    return CallablePairPoisson(
        reference_pair_poisson(grid, negative_laplacian, hartree_settings),
        "SciPy conjugate gradients on the full grid (one solve per pair)",
    )


class MultipolePairPoisson:
    """The pair solves of :func:`reference_pair_solver` on a sphere, with the boundary cached.

    The same equations as :func:`parsec_python.Hartree.solve_hartree` with the
    pair settings (multipole boundary of the pair density at the order of the
    run, no atomic tail, conjugate gradients from zero to the same stopping
    rule).  What a pair solve spends most of its time on there -- the
    spherical harmonics at the source and boundary points -- depends on the
    grid only, so it is formed once: the moments are ``Q = C rho`` and the
    boundary term of the right-hand side ``-A_IB V_B[Q] = U Q``, with
    ``C`` (``2 (L+1)**2`` real rows: real and imaginary parts of
    ``h**3 r**l conj(Y_lm)``) and ``U`` (``-A_IB`` times
    ``2 [4 pi/(2l+1)] Y_lm / r**(l+1)`` at the exterior stencil points).  The
    conjugate gradients of a batch run side by side (one sparse product per
    step for every column still running), each column with its own steps
    and stopping test, so a column ends where its own solve would.

    Memory: ``4 (L+1)**2`` grid vectors.  For a box domain (direct
    boundary sum) use :func:`reference_pair_solver`.
    """

    def __init__(self, grid, negative_laplacian, hartree_settings) -> None:
        from scipy.special import sph_harm_y

        from ..Laplacian.finite_difference import (
            neighbor_shells,
            second_derivative_coefficients,
        )
        from .exact_exchange import pair_hartree_settings

        settings = pair_hartree_settings(hartree_settings)
        method = settings.boundary_method
        if method == "auto":
            method = "multipole" if grid.settings.domain_shape == "sphere" else "direct"
        if method != "multipole":
            raise ValueError(
                "the cached multipole pair solver serves the multipole boundary of a "
                "sphere; use reference_pair_solver for a direct boundary sum"
            )
        started = perf_counter()
        self.settings = settings
        self.size = int(grid.size)
        self.operator = negative_laplacian
        order = int(settings.multipole_order)
        points = grid.coordinates
        radius = np.linalg.norm(points, axis=1)
        theta = np.zeros_like(radius)
        nonzero = radius > 0
        theta[nonzero] = np.arccos(np.clip(points[nonzero, 2] / radius[nonzero], -1.0, 1.0))
        phi = np.arctan2(points[:, 1], points[:, 0])
        # Exterior stencil points: rows, -A_IB coefficients, coordinates.
        coefficients = second_derivative_coefficients(grid.settings.expansion_order)
        width = grid.settings.stencil_half_width
        inverse_h2 = 1.0 / grid.spacing**2
        rows, weights, exterior = [], [], []
        for _axis, signed_shell, neighbor_rows, integer_points in neighbor_shells(grid):
            missing = neighbor_rows < 0
            if not np.any(missing):
                continue
            rows.append(np.flatnonzero(missing))
            # rhs -= A_IB V_B with A_IB = -c/h**2: rhs += (c/h**2) V_B.
            weights.append(
                np.full(int(np.count_nonzero(missing)), coefficients[width + abs(signed_shell)] * inverse_h2)
            )
            exterior.append(grid.physical_coordinates(integer_points[missing]))
        rows_all = np.concatenate(rows)
        weights_all = np.concatenate(weights)
        exterior_all = np.concatenate(exterior)
        outer_radius = np.linalg.norm(exterior_all, axis=1)
        outer_theta = np.arccos(np.clip(exterior_all[:, 2] / outer_radius, -1.0, 1.0))
        outer_phi = np.arctan2(exterior_all[:, 1], exterior_all[:, 0])
        count = (order + 1) ** 2
        moments = np.empty((2 * count, self.size), dtype=np.float64)
        boundary = np.zeros((self.size, 2 * count), dtype=np.float64)
        column = 0
        for angular in range(order + 1):
            source_radial = grid.volume_element * radius**angular
            factor = 2.0 * 4.0 * np.pi / (2 * angular + 1)
            outer_radial = outer_radius ** (-(angular + 1))
            for magnetic in range(-angular, angular + 1):
                source = sph_harm_y(angular, magnetic, theta, phi)
                target = sph_harm_y(angular, magnetic, outer_theta, outer_phi)
                # Q = sum w r^l conj(Y): Re Q and Im Q.
                moments[2 * column] = source_radial * source.real
                moments[2 * column + 1] = -source_radial * source.imag
                # V_B = 2 Re(sum Q b) = Re Q (2 Re b) - Im Q (2 Im b).
                values = factor * outer_radial
                np.add.at(boundary[:, 2 * column], rows_all, weights_all * values * target.real)
                np.add.at(boundary[:, 2 * column + 1], rows_all, -weights_all * values * target.imag)
                column += 1
        self.moments = moments
        self.boundary = boundary
        self.description = (
            f"conjugate gradients on the full grid, multipole boundary of order {order} "
            "(cached harmonics)"
        )
        self.setup_seconds = perf_counter() - started
        self.solves = 0
        self.iterations = 0
        self.seconds = 0.0

    def right_hand_sides(self, densities: np.ndarray) -> np.ndarray:
        """``8 pi rho - A_IB V_B[rho]`` of each column (Ry/bohr**2)."""

        densities = np.asarray(densities, dtype=np.float64)
        return 8.0 * np.pi * densities + self.boundary @ (self.moments @ densities)

    def solve_batch(self, densities: np.ndarray) -> np.ndarray:
        densities = np.asarray(densities, dtype=np.float64)
        if densities.ndim != 2 or densities.shape[0] != self.size:
            raise ValueError("pair densities must have shape (grid points, batch)")
        started = perf_counter()
        rhs = self.right_hand_sides(densities)
        settings = self.settings
        solution = np.zeros_like(rhs)
        residual = rhs.copy()
        initial = np.linalg.norm(residual, axis=0)
        from ..Hartree.poisson import cg_stopping_threshold

        thresholds = np.array(
            [cg_stopping_threshold(settings, value, value) for value in initial]
        )
        running = np.flatnonzero(initial > thresholds)
        direction = residual.copy()
        squared = np.einsum("ij,ij->j", residual, residual)
        products = 1
        iterations = np.zeros(rhs.shape[1], dtype=np.int64)
        while running.size and products < settings.max_iterations:
            applied = np.asarray(self.operator @ direction[:, running])
            products += 1
            denominators = np.einsum("ij,ij->j", direction[:, running], applied)
            if np.any(denominators <= 0.0) or not np.all(np.isfinite(denominators)):
                break
            alpha = squared[running] / denominators
            solution[:, running] += alpha * direction[:, running]
            residual[:, running] -= alpha * applied
            iterations[running] += 1
            norms = np.linalg.norm(residual[:, running], axis=0)
            done = norms <= thresholds[running]
            new_squared = np.einsum("ij,ij->j", residual[:, running], residual[:, running])
            beta = new_squared / squared[running]
            squared[running] = new_squared
            keep = running[~done]
            direction[:, keep] = residual[:, keep] + beta[~done] * direction[:, keep]
            running = keep
        if running.size:
            raise RuntimeError(
                "pair-density conjugate-gradient solve did not converge "
                f"({running.size} of {rhs.shape[1]} columns)"
            )
        self.solves += rhs.shape[1]
        self.iterations += int(iterations.sum())
        self.seconds += perf_counter() - started
        return solution


def cached_pair_solver(grid, negative_laplacian, hartree_settings):
    """:class:`MultipolePairPoisson` on a sphere, :func:`reference_pair_solver` otherwise."""

    try:
        return MultipolePairPoisson(grid, negative_laplacian, hartree_settings)
    except ValueError:
        return reference_pair_solver(grid, negative_laplacian, hartree_settings)


@dataclass(frozen=True)
class ExchangeApplication:
    """``W = V_x Psi`` of an orbital block and what it took.

    ``applied`` has the shape of the orbitals; its columns not in
    ``columns`` are zero.  ``energy`` is the exchange energy ``E_x`` (Ry,
    without any hybrid fraction) of the occupied orbitals, ``(w/2) sum_i f_i
    <q_i|W_i>``; it needs only the occupied columns.  ``pair_solves`` is the
    number of pair densities solved, ``poisson_seconds`` their time,
    ``seconds`` the whole evaluation.
    """

    applied: np.ndarray
    columns: tuple[int, ...]
    occupied: tuple[int, ...]
    energy: float
    pair_solves: int
    poisson_seconds: float
    seconds: float


def _checked_block(orbitals: np.ndarray, occupations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(orbitals, np.ndarray):
        raise TypeError(
            "the exchange operator needs the orbitals as a host array on the full grid"
        )
    if np.iscomplexobj(orbitals):
        raise NotImplementedError("the exchange operator is implemented for real orbitals")
    if orbitals.ndim != 2:
        raise ValueError("orbitals must have shape (grid points, states)")
    occupations = np.asarray(occupations, dtype=np.float64)
    if occupations.shape != (orbitals.shape[1],):
        raise ValueError("one occupation per orbital column is required")
    return np.asarray(orbitals, dtype=np.float64), occupations


def _pair_batches(pairs: Sequence[tuple[int, int]], batch_size: int):
    for start in range(0, len(pairs), batch_size):
        yield pairs[start : start + batch_size]


def apply_exchange_to_orbitals(
    orbitals: np.ndarray,
    occupations: np.ndarray,
    volume_element: float,
    solver: PairPoissonSolver,
    *,
    spin_weight: float = 2.0,
    columns: Sequence[int] | None = None,
    known: ExchangeApplication | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    occupation_threshold: float = OCCUPATION_THRESHOLD,
) -> ExchangeApplication:
    """``W = V_x Psi`` for the columns ``columns`` of the orbital block itself.

    The occupied orbitals (occupation above ``occupation_threshold``) form
    the operator.  Pairs of two occupied target columns are solved once and
    serve both; a target without occupation pairs with every occupied
    orbital.  ``known`` is an earlier application to the same orbitals (for
    example of the occupied columns only, enough for the energy): its
    columns are kept and only the others are solved.  ``columns`` defaults
    to every column.  ``batch_size`` pair densities go to each
    ``solver.solve_batch`` call.
    """

    started = perf_counter()
    orbitals, occupations = _checked_block(orbitals, occupations)
    if not np.isfinite(volume_element) or volume_element <= 0.0:
        raise ValueError("volume_element must be positive")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    states = orbitals.shape[1]
    occupied = [int(index) for index in occupied_columns(occupations, occupation_threshold)]
    occupied_set = set(occupied)
    wanted = list(range(states)) if columns is None else sorted({int(c) for c in columns})
    if any(column < 0 or column >= states for column in wanted):
        raise ValueError("a requested column is outside the orbital block")
    if known is not None:
        if known.applied.shape != orbitals.shape or tuple(known.occupied) != tuple(occupied):
            raise ValueError("the known application belongs to another orbital block")
        applied = known.applied.copy()
        done = set(known.columns)
    else:
        applied = np.zeros_like(orbitals)
        done = set()
    targets = [column for column in wanted if column not in done]
    occupied_targets = [column for column in targets if column in occupied_set]
    empty_targets = [column for column in targets if column not in occupied_set]
    # Occupied targets: unordered pairs (i <= j) of the occupied set where at
    # least one of the two is a target.  A pair with one known member serves
    # the other member only.
    target_set = set(occupied_targets)
    # (a, b, a receives, b receives): W_a -= f_b q_b v_ab and W_b -= f_a q_a v_ab.
    pairs: list[tuple[int, int, bool, bool]] = []
    for position, first in enumerate(occupied):
        for second in occupied[position:]:
            first_wants = first in target_set
            second_wants = second in target_set and second != first
            if first_wants or second_wants:
                pairs.append((first, second, first_wants, second_wants))
    # An orbital without occupation adds nothing to the operator: it only receives.
    pairs.extend(
        (target, source, True, False) for target in empty_targets for source in occupied
    )
    inverse_volume = 1.0 / volume_element
    poisson_seconds = 0.0
    for batch in _pair_batches(pairs, batch_size):
        first = np.fromiter((pair[0] for pair in batch), dtype=np.int64, count=len(batch))
        second = np.fromiter((pair[1] for pair in batch), dtype=np.int64, count=len(batch))
        densities = orbitals[:, first] * orbitals[:, second]
        densities *= inverse_volume
        solve_started = perf_counter()
        potentials = np.asarray(solver.solve_batch(densities), dtype=np.float64)
        poisson_seconds += perf_counter() - solve_started
        if potentials.shape != densities.shape:
            raise ValueError("the pair solver returned potentials of another shape")
        for index, (a, b, a_receives, b_receives) in enumerate(batch):
            potential = potentials[:, index]
            if a_receives:
                applied[:, a] -= occupations[b] * (orbitals[:, b] * potential)
            if b_receives:
                applied[:, b] -= occupations[a] * (orbitals[:, a] * potential)
    columns_done = tuple(sorted(done | set(targets)))
    occupied_done = [column for column in occupied if column in columns_done]
    energy = 0.0
    if len(occupied_done) == len(occupied):
        energy = 0.5 * spin_weight * float(
            sum(
                occupations[column] * float(orbitals[:, column] @ applied[:, column])
                for column in occupied
            )
        )
    elif known is not None:
        energy = known.energy
    else:
        energy = float("nan")
    return ExchangeApplication(
        applied=applied,
        columns=columns_done,
        occupied=tuple(occupied),
        energy=float(energy),
        pair_solves=len(pairs) + (0 if known is None else known.pair_solves),
        poisson_seconds=poisson_seconds + (0.0 if known is None else known.poisson_seconds),
        seconds=perf_counter() - started + (0.0 if known is None else known.seconds),
    )


def apply_exchange(
    orbitals: np.ndarray,
    occupations: np.ndarray,
    vectors: np.ndarray,
    volume_element: float,
    solver: PairPoissonSolver,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    occupation_threshold: float = OCCUPATION_THRESHOLD,
) -> np.ndarray:
    """``V_x X`` for arbitrary grid vectors ``X`` (``(grid points,)`` or ``(grid points, m)``).

    The operator is that of the occupied columns of ``orbitals``; every
    vector pairs with every occupied orbital (``m n_occ`` solves).  This is
    the uncompressed operator, the reference of :class:`CompressedExchange`.
    """

    orbitals, occupations = _checked_block(orbitals, occupations)
    vectors = np.asarray(vectors, dtype=np.float64)
    single = vectors.ndim == 1
    block = vectors[:, None] if single else vectors
    if block.ndim != 2 or block.shape[0] != orbitals.shape[0]:
        raise ValueError("vectors must have shape (grid points,) or (grid points, m)")
    occupied = [int(index) for index in occupied_columns(occupations, occupation_threshold)]
    result = np.zeros_like(block)
    pairs = [(target, source) for target in range(block.shape[1]) for source in occupied]
    inverse_volume = 1.0 / volume_element
    for batch in _pair_batches(pairs, int(batch_size)):
        target = np.fromiter((pair[0] for pair in batch), dtype=np.int64, count=len(batch))
        source = np.fromiter((pair[1] for pair in batch), dtype=np.int64, count=len(batch))
        densities = block[:, target] * orbitals[:, source] * inverse_volume
        potentials = np.asarray(solver.solve_batch(densities), dtype=np.float64)
        for index, (a, b) in enumerate(batch):
            result[:, a] -= occupations[b] * (orbitals[:, b] * potentials[:, index])
    return result[:, 0] if single else result


@dataclass(frozen=True)
class CompressedExchange:
    """``V_ACE = -xi xi^T`` of one orbital block (one spin channel).

    ``projectors`` is ``xi`` ``(grid points, states)``.  ``fraction`` scales
    the operator where it joins a Hamiltonian (``a_x``); the methods with
    ``scaled=False`` give the bare compressed exchange.  ``energy`` is the
    exchange energy ``E_x`` of the orbitals the operator was built from
    (Ry, unscaled, that channel set's spin weight included) and
    ``spin_weight`` that weight.  ``asymmetry`` is
    ``max|M - M^T| / max|M|`` of the pair solver on this block.
    """

    projectors: np.ndarray
    fraction: float = 1.0
    energy: float = 0.0
    spin_weight: float = 2.0
    asymmetry: float = 0.0
    pair_solves: int = 0
    seconds: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        size = int(self.projectors.shape[0])
        return (size, size)

    @property
    def rank(self) -> int:
        return int(self.projectors.shape[1])

    def apply(self, vectors: np.ndarray, *, scaled: bool = True) -> np.ndarray:
        """``a_x V_ACE X = -a_x xi (xi^T X)`` (``scaled=False``: without ``a_x``)."""

        vectors = np.asarray(vectors, dtype=np.float64)
        xi = self.projectors
        factor = -(self.fraction if scaled else 1.0)
        return factor * (xi @ (xi.T @ vectors))

    def expectation_values(self, vectors: np.ndarray, *, scaled: bool = True) -> np.ndarray:
        """``<x_k|a_x V_ACE|x_k> = -a_x ||xi^T x_k||^2`` for each column ``x_k``."""

        vectors = np.asarray(vectors, dtype=np.float64)
        block = vectors[:, None] if vectors.ndim == 1 else vectors
        overlaps = self.projectors.T @ block
        factor = -(self.fraction if scaled else 1.0)
        return factor * np.einsum("ij,ij->j", overlaps, overlaps)

    def as_sparse(self):
        """Dense ``a_x V_ACE`` as a sparse matrix, for small diagnostic grids only."""

        import scipy.sparse as sp

        size = self.shape[0]
        if size > 20000:
            raise NotImplementedError(
                "the compressed exchange operator is dense; it is formed only for "
                "grids of at most 20000 points"
            )
        return sp.csr_matrix(-self.fraction * (self.projectors @ self.projectors.T))


def compress_exchange(
    orbitals: np.ndarray,
    application: ExchangeApplication,
    *,
    fraction: float = 1.0,
    spin_weight: float = 2.0,
) -> CompressedExchange:
    """Build ``V_ACE`` from ``W = V_x Psi`` (``application``, every column).

    ``M = -Psi^T W`` is symmetrized and factorized, ``M = L L^T``, and
    ``xi = W L^-T`` (one triangular solve).  ``M`` is positive definite when
    every column overlaps some occupied orbital: ``M_kk = sum_j f_j (kj|kj)``.
    """

    orbitals = np.asarray(orbitals, dtype=np.float64)
    applied = application.applied
    if applied.shape != orbitals.shape:
        raise ValueError("the exchange application does not match the orbitals")
    if len(application.columns) != orbitals.shape[1]:
        raise ValueError("ACE needs V_x applied to every orbital of the block")
    started = perf_counter()
    overlap = -(orbitals.T @ applied)
    scale = float(np.max(np.abs(overlap), initial=0.0))
    asymmetry = (
        float(np.max(np.abs(overlap - overlap.T), initial=0.0)) / scale if scale > 0.0 else 0.0
    )
    overlap = 0.5 * (overlap + overlap.T)
    try:
        lower = np.linalg.cholesky(overlap)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "the exchange matrix M = -Psi^T V_x Psi of the orbital block is not "
            "positive definite; ACE needs orbitals that overlap the occupied ones"
        ) from error
    # xi = W L^-T  <=>  xi^T = L^-1 W^T.
    projectors = solve_triangular(lower, applied.T, lower=True).T
    return CompressedExchange(
        projectors=np.ascontiguousarray(projectors),
        fraction=float(fraction),
        energy=float(application.energy),
        spin_weight=float(spin_weight),
        asymmetry=asymmetry,
        pair_solves=int(application.pair_solves),
        seconds=float(application.seconds + perf_counter() - started),
    )


def exchange_band_energy(
    operator: CompressedExchange | None,
    orbitals: np.ndarray,
    occupations: np.ndarray,
    *,
    scaled: bool = True,
) -> float:
    """``w sum_i f_i <q_i|a_x V_ACE|q_i>`` (Ry): the exchange in the band energy.

    ``w`` is the spin weight of the operator.  Without an operator (a spin
    channel with no occupied orbital) the term is zero.  The bare form
    (``scaled=False``) is ``2 E_x(P, P0)``, twice the exchange energy of the
    orbitals ``P`` with the operator of the orbitals ``P0`` it was built from.
    """

    if operator is None:
        return 0.0
    occupations = np.asarray(occupations, dtype=np.float64)
    occupied = np.flatnonzero(occupations > OCCUPATION_THRESHOLD)
    if occupied.size == 0:
        return 0.0
    values = operator.expectation_values(
        np.asarray(orbitals, dtype=np.float64)[:, occupied], scaled=scaled
    )
    return float(operator.spin_weight * np.dot(occupations[occupied], values))


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "CallablePairPoisson",
    "CompressedExchange",
    "ExchangeApplication",
    "MultipolePairPoisson",
    "PairPoissonSolver",
    "apply_exchange",
    "apply_exchange_to_orbitals",
    "cached_pair_solver",
    "compress_exchange",
    "exchange_band_energy",
    "reference_pair_solver",
]
