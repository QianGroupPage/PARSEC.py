"""Finite-cluster Hartree boundary construction and Poisson solve.

For a positive electron number density ``rho``, the Hartree potential in
Hartree atomic units is

``V_H(r) = integral rho(r') / |r-r'| d^3r'``.

Because ``nabla**2 (1/|r-r'|) = -4*pi*delta(r-r')``, it obeys

``-nabla**2 V_H = 4*pi*rho``                       (Hartree units).

PARSEC stores potentials in Rydberg, where ``1 Ha = 2 Ry``.  This module
therefore solves

``-nabla**2 V_H = 8*pi*rho``                       (Rydberg units).

An isolated calculation is physically defined on infinite space, with
``V_H(r) -> 0`` only as ``|r| -> infinity``.  The numerical grid is finite,
and its boundary value is generally not zero: its leading term is
``2*N_e/r`` in Rydberg.  This module obtains finite-boundary values either
from a multipole expansion or a direct discrete Coulomb sum.

If ``I`` denotes active grid points and ``B`` denotes missing exterior
stencil points, the finite-difference equation is partitioned as

``A_II V_I + A_IB V_B = 8*pi*rho_I``.

Only ``V_I`` is unknown, so :func:`apply_negative_laplacian_boundary`
constructs the effective right-hand side

``A_II V_I = 8*pi*rho_I - A_IB V_B``.

The resulting symmetric positive-definite interior system is solved by
conjugate gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.special import sph_harm_y

from ..Grid import RealSpaceGrid
from ..Laplacian import apply_negative_laplacian_boundary
from ..models import HartreeSettings


@dataclass(frozen=True)
class MultipoleExpansion:
    """Far-field representation of the Hartree potential.

    For a source density contained inside the evaluation radius, define

    ``Q_lm = integral rho(r') * r'**l * conj(Y_lm(Omega')) d^3r'``.

    The Rydberg Hartree potential outside that density is

    ``V_H(r) = 2 * sum_lm [4*pi/(2*l+1)]
                        * Q_lm * Y_lm(Omega) / r**(l+1)``.

    ``order`` is the largest retained angular momentum ``l``.  The moments
    are complex because complex normalized spherical harmonics are used, but
    the potential of a real density is real.  The ``l=0`` term reduces to
    the familiar monopole tail ``2*N_e/r``.

    The expansion is safe only when each evaluation radius is larger than
    every source radius.  That condition is naturally satisfied for an
    origin-centered spherical domain, but not necessarily on the faces of a
    general box.
    """

    order: int
    moments: dict[tuple[int, int], complex]

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the truncated multipole potential at Cartesian points.

        ``points`` must have shape ``(n, 3)`` and be expressed in bohr
        relative to the multipole origin.  The returned values are in
        Rydberg.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("boundary points must have shape (n, 3)")
        radius = np.linalg.norm(points, axis=1)
        if np.any(radius <= 0):
            raise ValueError("multipole boundary potential is undefined at the origin")

        # scipy.special.sph_harm_y uses the polar angle theta measured from
        # +z and the azimuthal angle phi in the x-y plane.
        theta = np.arccos(np.clip(points[:, 2] / radius, -1.0, 1.0))
        phi = np.arctan2(points[:, 1], points[:, 0])
        result = np.zeros(points.shape[0], dtype=complex)

        # Sum [4*pi/(2*l+1)] Q_lm Y_lm(Omega) / r**(l+1).
        for angular_momentum in range(self.order + 1):
            factor = 4.0 * np.pi / (2 * angular_momentum + 1)
            radial_factor = radius ** (-(angular_momentum + 1))
            for magnetic in range(-angular_momentum, angular_momentum + 1):
                result += (
                    factor
                    * self.moments[(angular_momentum, magnetic)]
                    * sph_harm_y(angular_momentum, magnetic, theta, phi)
                    * radial_factor
                )

        # The imaginary remainder should be roundoff for a real density.
        # Factor two converts the Hartree-unit Coulomb value to Rydberg.
        return 2.0 * result.real

    __call__ = potential


@dataclass
class DirectCoulombBoundary:
    """Direct boundary potential of the density represented on the grid.

    For source coordinates ``r_s`` and quadrature weights
    ``w_s = rho(r_s)*DeltaV``, this class evaluates

    ``V_H(R_b) = 2 * sum_s w_s / |R_b-r_s|``.

    The factor two gives Rydberg units.  This is exact for the discretized
    density and its chosen volume quadrature, rather than an analytic
    continuum integral.

    This corresponds to PARSEC's optional ``Full_Hartree`` boundary sum.  It is
    appropriate for a box, where a face point is not guaranteed to lie
    outside a sphere enclosing every source point.  Its nominal cost is
    ``O(number_of_boundary_points * number_of_source_points)``, so values are
    evaluated in chunks and cached.
    """

    source_coordinates: np.ndarray
    source_weights: np.ndarray
    chunk_size: int = 16
    _cache: dict[tuple[float, float, float], float] = field(
        default_factory=dict, init=False, repr=False
    )

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the direct discrete Coulomb sum at boundary points.

        ``points`` must have shape ``(n, 3)`` in bohr.  Repeated coordinates
        are served from ``_cache`` because the same missing finite-difference
        point can be requested from several active rows or later evaluations
        of the same boundary object.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("boundary points must have shape (n, 3)")
        result = np.empty(points.shape[0], dtype=float)
        missing_keys: list[tuple[float, float, float]] = []
        missing_rows: dict[tuple[float, float, float], list[int]] = {}

        # Collect only coordinates that have not already been evaluated.
        # missing_rows also coalesces duplicates within this call.
        for row, point in enumerate(points):
            key = (float(point[0]), float(point[1]), float(point[2]))
            if key in self._cache:
                result[row] = self._cache[key]
            else:
                if key not in missing_rows:
                    missing_keys.append(key)
                    missing_rows[key] = []
                missing_rows[key].append(row)

        # Use |R-r|**2 = |R|**2 + |r|**2 - 2*R.r.  This performs each chunk
        # with a matrix multiplication without allocating an (n_query,n,3)
        # array of coordinate differences.
        source_squared = np.einsum(
            "ij,ij->i", self.source_coordinates, self.source_coordinates
        )
        for start in range(0, len(missing_keys), self.chunk_size):
            keys = missing_keys[start : start + self.chunk_size]
            queries = np.asarray(keys, dtype=float)
            distance_squared = (
                np.einsum("ij,ij->i", queries, queries)[:, None]
                + source_squared[None, :]
                - 2.0 * queries @ self.source_coordinates.T
            )
            distance_squared = np.maximum(distance_squared, 0.0)
            if np.any(distance_squared == 0.0):
                raise ValueError("direct Hartree boundary coincides with a source point")

            # source_weights contains rho_s*DeltaV.  The leading factor two
            # converts the Coulomb potential from Hartree to Rydberg.
            values = 2.0 * (
                (1.0 / np.sqrt(distance_squared)) @ self.source_weights
            )
            for key, value in zip(keys, values):
                scalar = float(value)
                self._cache[key] = scalar
                for row in missing_rows[key]:
                    result[row] = scalar
        return result

    __call__ = potential


def density_multipoles(
    density: np.ndarray,
    grid: RealSpaceGrid,
    order: int = 9,
) -> MultipoleExpansion:
    """Compute discrete spherical-harmonic moments of the density.

    The continuum definition is

    ``Q_lm = integral rho(r) * r**l * conj(Y_lm(Omega)) d^3r``.

    On the uniform real-space grid it is approximated by

    ``Q_lm = sum_s rho_s * DeltaV * r_s**l
                   * conj(Y_lm(Omega_s))``.

    All moments from ``l=0`` through ``l=order`` are returned.  At the
    coordinate origin the angles are arbitrary; choosing ``theta=0`` is
    harmless because ``r**l`` vanishes there for ``l>0`` and ``Y_00`` is
    angle independent.
    """
    density = np.asarray(density, dtype=float)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    if order < 0:
        raise ValueError("multipole order cannot be negative")

    points = grid.coordinates
    radius = np.linalg.norm(points, axis=1)
    theta = np.zeros_like(radius)
    nonzero = radius > 0
    theta[nonzero] = np.arccos(np.clip(points[nonzero, 2] / radius[nonzero], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])

    # rho has units electron/bohr**3 and volume_element has bohr**3, so each
    # weighted_density entry is the electron count represented by one point.
    weighted_density = density * grid.volume_element
    moments: dict[tuple[int, int], complex] = {}
    for angular_momentum in range(order + 1):
        radial_weight = radius**angular_momentum
        for magnetic in range(-angular_momentum, angular_momentum + 1):
            harmonic = sph_harm_y(angular_momentum, magnetic, theta, phi)
            moments[(angular_momentum, magnetic)] = complex(
                np.sum(weighted_density * radial_weight * np.conjugate(harmonic))
            )
    return MultipoleExpansion(order=order, moments=moments)


@dataclass(frozen=True)
class HartreeResult:
    """Hartree solution and diagnostics.

    ``potential``
        Active-grid Hartree potential in Rydberg.
    ``right_hand_side``
        Boundary-corrected vector
        ``8*pi*rho_I - A_IB*V_B`` actually passed to CG.
    ``boundary``
        Boundary model constructed from the current density.
    ``iterations`` and ``matrix_vector_products``
        CG work counters.  Matrix-vector products include initial/final
        residual evaluations as well as iteration products.
    ``residual_norm`` and ``initial_residual_norm``
        Euclidean norms of ``b-A*V`` before and after the solve.
    """

    potential: np.ndarray
    right_hand_side: np.ndarray
    boundary: MultipoleExpansion | DirectCoulombBoundary
    converged: bool
    iterations: int
    matrix_vector_products: int
    residual_norm: float
    initial_residual_norm: float


def _conjugate_gradient(
    operator: sp.spmatrix,
    rhs: np.ndarray,
    initial: np.ndarray,
    settings: HartreeSettings,
) -> tuple[np.ndarray, bool, int, int, float, float]:
    """Solve ``A*x=b`` with unpreconditioned conjugate gradients.

    The interior negative Laplacian ``A_II`` with Dirichlet boundary
    conditions is symmetric positive definite, so CG is applicable.  With
    residual ``r_k = b-A*x_k``, convergence is declared when

    ``||r_k||_2 <= relative_tolerance*||r_0||_2 + absolute_tolerance``.

    The return tuple is ``(solution, converged, iterations, matvecs,
    final_residual_norm, initial_residual_norm)``.
    """
    solution = np.asarray(initial, dtype=float).copy()

    # r_0 = b - A*x_0.  A previous SCF Hartree potential can be supplied as
    # x_0; otherwise solve_hartree uses zero.
    residual = rhs - operator @ solution
    matrix_vector_products = 1
    initial_norm = float(np.linalg.norm(residual))
    tolerance = (
        settings.relative_tolerance * initial_norm + settings.absolute_tolerance
    )
    if initial_norm <= tolerance:
        return solution, True, 0, matrix_vector_products, initial_norm, initial_norm

    direction = residual.copy()
    residual_squared = float(residual @ residual)
    iterations = 0
    converged = False
    while matrix_vector_products < settings.max_iterations:
        operator_direction = operator @ direction
        matrix_vector_products += 1
        denominator = float(direction @ operator_direction)

        # For an SPD operator p.T@A@p must be positive.  A nonpositive or
        # nonfinite value indicates numerical breakdown or an invalid matrix.
        if denominator <= 0.0 or not np.isfinite(denominator):
            break

        # Standard CG update:
        # alpha_k = (r_k.r_k)/(p_k.A.p_k)
        # x_(k+1) = x_k + alpha_k*p_k
        # r_(k+1) = r_k - alpha_k*A*p_k
        alpha = residual_squared / denominator
        solution += alpha * direction
        residual -= alpha * operator_direction
        iterations += 1
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            converged = True
            break
        new_residual_squared = float(residual @ residual)

        # beta_k = (r_(k+1).r_(k+1))/(r_k.r_k), followed by construction of
        # the next A-conjugate search direction.
        beta = new_residual_squared / residual_squared
        direction = residual + beta * direction
        residual_squared = new_residual_squared

    # Recompute rather than relying on the recursively updated residual,
    # which can accumulate floating-point drift in a long CG solve.
    residual_norm = float(np.linalg.norm(rhs - operator @ solution))
    matrix_vector_products += 1
    return (
        solution,
        converged,
        iterations,
        matrix_vector_products,
        residual_norm,
        initial_norm,
    )


def solve_hartree(
    density: np.ndarray,
    grid: RealSpaceGrid,
    negative_laplacian: sp.spmatrix,
    settings: HartreeSettings = HartreeSettings(),
    initial_potential: np.ndarray | None = None,
    *,
    raise_on_nonconvergence: bool = True,
) -> HartreeResult:
    """Construct and solve the isolated-system Hartree Poisson problem.

    Parameters
    ----------
    density
        Positive valence-electron number density in ``electron/bohr**3``.
    grid
        Active real-space grid; its coordinates and spacing are in bohr.
    negative_laplacian
        The active-to-active finite-difference matrix
        ``A_II = -nabla_h**2`` in ``bohr**(-2)``.
    settings
        Boundary-method, multipole/direct, and CG settings.
    initial_potential
        Optional CG starting vector in Rydberg, normally the Hartree
        potential from an earlier SCF iteration.
    raise_on_nonconvergence
        Raise ``RuntimeError`` when CG misses its tolerance.  If false,
        return the unconverged result and its diagnostics.

    Notes
    -----
    The calculation consists of four steps:

    1. Construct ``V_B`` from the current density.  ``auto`` uses a multipole
       expansion for an origin-centered sphere and a direct Coulomb sum for
       a box.
    2. Form the Rydberg Poisson source ``b_I = 8*pi*rho_I``.
    3. Fold the known exterior stencil values into the source:
       ``b_eff = b_I - A_IB*V_B``.
    4. Solve ``A_II*V_I = b_eff`` with conjugate gradients.

    The boundary model accounts only for density represented inside the
    grid.  The domain must therefore be large enough that omitted exterior
    electron density is negligible.
    """
    density = np.asarray(density, dtype=float)
    if density.shape != (grid.size,):
        raise ValueError("density does not match the active grid")
    if negative_laplacian.shape != (grid.size, grid.size):
        raise ValueError("negative_laplacian shape does not match the grid")

    boundary_method = settings.boundary_method
    if boundary_method == "auto":
        # Every point outside an origin-centered sphere lies outside all
        # active source radii, satisfying multipole convergence.  That is not
        # generally true just outside a face of a box.
        boundary_method = (
            "multipole" if grid.settings.domain_shape == "sphere" else "direct"
        )
    if boundary_method == "multipole":
        if grid.settings.domain_shape != "sphere":
            raise ValueError(
                "an origin-centered multipole boundary is not convergent at "
                "general box faces; use boundary_method='direct' or 'auto'"
            )
        boundary: MultipoleExpansion | DirectCoulombBoundary = density_multipoles(
            density, grid, settings.multipole_order
        )
    else:
        boundary = DirectCoulombBoundary(
            source_coordinates=grid.coordinates,
            source_weights=density * grid.volume_element,
            chunk_size=settings.direct_chunk_size,
        )

    # Rydberg-unit Poisson source: -nabla**2 V_H = 8*pi*rho.
    rhs = 8.0 * np.pi * density

    # build_negative_laplacian stores only A_II.  Add the known exterior
    # values through b_eff = 8*pi*rho_I - A_IB*V_B.
    rhs = apply_negative_laplacian_boundary(rhs, grid, boundary.potential)

    # Reusing the previous SCF potential can greatly reduce the initial
    # residual; zero is a valid starting guess for the first solve.
    if initial_potential is None:
        initial = np.zeros(grid.size, dtype=float)
    else:
        initial = np.asarray(initial_potential, dtype=float)
        if initial.shape != (grid.size,):
            raise ValueError("initial Hartree potential does not match the grid")

    potential, converged, iterations, matvecs, residual, initial_residual = (
        _conjugate_gradient(negative_laplacian, rhs, initial, settings)
    )
    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "Hartree conjugate-gradient solve did not converge: "
            f"residual={residual:.3e}, matvecs={matvecs}"
        )
    return HartreeResult(
        potential=potential,
        right_hand_side=rhs,
        boundary=boundary,
        converged=converged,
        iterations=iterations,
        matrix_vector_products=matvecs,
        residual_norm=residual,
        initial_residual_norm=initial_residual,
    )
