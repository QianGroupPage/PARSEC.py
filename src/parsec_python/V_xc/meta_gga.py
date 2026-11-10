"""Meta-GGA exchange-correlation on the real-space grid.

A meta-GGA evaluator (:mod:`parsec_python.V_xc.semilocal`, family ``"mgga"``)
depends on the kinetic-energy density of the orbitals,

``tau_s = 1/2 sum_n w_n |grad psi_n|^2``   (Hartree units, libxc's convention),

and possibly on the Laplacian of the density.  This module holds the discrete
energy, its exact derivatives and the orbital operator they give.

Discretization.  With ``G = (G_x, G_y, G_z)`` the centered first-derivative
operators of :func:`parsec_python.Laplacian.build_gradient` (the stencil of
:func:`parsec_python.V_xc.pbe.density_gradient`, order ``Expansion_Order``:
zero outside a cluster domain or a confined axis, wrapped around periodic
axes, made Cartesian on oblique cells) and ``L = -negative_laplacian`` (the
kinetic stencil),

``E_xc = h^3 sum_i e(n_i, sigma_i, tau_i, lapl_i)``,
``sigma = |G n|^2``, ``tau_s = (d/2h^3) sum_m f_m |G q_m|^2``, ``lapl = L n``,

where ``q_m`` are the Euclidean-normalized grid vectors of the eigensolver
(``q = sqrt(h^3) psi``), ``f_m`` their occupations and ``d`` the spin
degeneracy (2 unpolarized, 1 per spin channel).  The potentials are the exact
derivatives of this sum,

``v = vrho + G^T (2 vsigma G n) + L^T vlapl``  (spin: ``v_a = vrho_a + G^T (2 vs_aa G n_a + vs_ab G n_b) + L^T vlapl_a``),

and the derivative with respect to an orbital gives the Hamiltonian term
(Rydberg)

``H_tau q = sum_c G_c^T diag(vtau) G_c q``,

the discrete ``-div(vtau grad q)``: symmetric on the grid by construction and
positive semidefinite where ``vtau >= 0``, so the SCF energy is variational
and no force term arises from it.  ``vtau = de/dtau`` is dimensionless; the
band energy holds ``sum_m d f_m <q_m|H_tau|q_m> = 2 h^3 sum_i vtau_i tau_i`` (Ry),
which the total energy removes (double counting).  For a Bloch factor ``u``
of wave vector ``k`` the derivative is ``G_c + i k_c``:
``tau = (d/2h^3) sum w |(G + ik) u|^2`` and
``H_tau = sum_c (G_c + i k_c)^H diag(vtau) (G_c + i k_c)``.

A frozen nonlinear-core density enters ``n`` (and so ``sigma`` and ``lapl``)
only; ``tau`` is that of the valence orbitals (no core ``tau``), as in other
plane-wave and real-space codes.  Units: the evaluator works in Hartree; the
results here are Rydberg (potentials and energies), ``vtau`` stays the
dimensionless ``de/dtau``.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np
import scipy.sparse as sp

from .ca_lda import XCResult
from .ca_lda_spin import SpinXCResult, spin_densities_with_core
from .pbe import _DENSITY_THRESHOLD, _gga_density


@dataclass(frozen=True)
class MetaGGAXCResult(XCResult):
    """:class:`~parsec_python.V_xc.ca_lda.XCResult` of a meta-GGA, with its ``vtau``.

    ``kinetic_potential`` is ``vtau = de/dtau`` on the grid (dimensionless):
    the orbital term of the Hamiltonian is ``sum_c G_c^T diag(vtau) G_c``
    (Rydberg).  ``potential`` is ``dE/dn`` (Ry) at fixed ``tau``.
    ``fermi_hole_clamped`` counts the points where ``tau < tau_W = sigma/(8n)``
    (where libxc's Fermi-hole-curvature clamp acts).
    """

    kinetic_potential: np.ndarray
    fermi_hole_clamped: int = 0
    # The orbital form of the density gradient (meta_gga_gradient() == "orbital"):
    # W = de/dg (Hartree, shape (3, n)) of the orbital term 2 sum_c [W_c G_c + G_c^T W_c],
    # and dE/dn_core (Ry) of the core-correction force; None in the density form.
    gradient_flux: np.ndarray | None = None
    core_potential: np.ndarray | None = None


@dataclass(frozen=True)
class MetaGGASpinXCResult(SpinXCResult):
    """:class:`~parsec_python.V_xc.ca_lda_spin.SpinXCResult` of a meta-GGA, with ``vtau`` per channel."""

    kinetic_potential_up: np.ndarray
    kinetic_potential_down: np.ndarray
    fermi_hole_clamped: int = 0
    # The orbital form (see MetaGGAXCResult): W_a = de/dg_a of each channel and dE/dn_core (Ry).
    gradient_flux_up: np.ndarray | None = None
    gradient_flux_down: np.ndarray | None = None
    core_potential: np.ndarray | None = None

    @property
    def gradient_fluxes(self):
        return (self.gradient_flux_up, self.gradient_flux_down)

    @property
    def kinetic_potentials(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.kinetic_potential_up, self.kinetic_potential_down)


# ----- the gradient operators of a grid ----------------------------------------------

_GRADIENT_CACHE: list[tuple[object, "GridGradient"]] = []
_GRADIENT_LOCK = threading.Lock()


@dataclass(frozen=True)
class GridGradient:
    """The three sparse first-derivative operators ``G_c`` of a grid (``build_gradient``)."""

    matrices: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]

    @property
    def size(self) -> int:
        return int(self.matrices[0].shape[0])

    def apply(self, field: np.ndarray) -> np.ndarray:
        """``G f`` of a grid field, shape ``(3, n)`` (or ``(3, n, k)`` for a block)."""

        return np.stack([matrix @ field for matrix in self.matrices])

    def adjoint(self, fluxes) -> np.ndarray:
        """``sum_c G_c^T flux_c``."""

        total = self.matrices[0].T @ fluxes[0]
        for matrix, flux in zip(self.matrices[1:], fluxes[1:]):
            total = total + matrix.T @ flux
        return total


def grid_gradient(grid) -> GridGradient:
    """:class:`GridGradient` of ``grid``, built once per grid object (a few kept)."""

    with _GRADIENT_LOCK:
        for known, gradient in _GRADIENT_CACHE:
            if known is grid:
                return gradient
    from ..Laplacian import build_gradient

    gradient = GridGradient(tuple(sp.csr_matrix(matrix, dtype=np.float64) for matrix in build_gradient(grid)))
    with _GRADIENT_LOCK:
        _GRADIENT_CACHE.insert(0, (grid, gradient))
        del _GRADIENT_CACHE[4:]
    return gradient


# ----- tau of orbitals --------------------------------------------------------------


def kinetic_energy_density(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    gradient: GridGradient,
    volume_element: float,
    *,
    degeneracy: float = 2.0,
    k_point: np.ndarray | None = None,
) -> np.ndarray:
    """``tau = (degeneracy/(2 h^3)) sum_m f_m |G q_m|^2`` of Euclidean-normalized grid orbitals.

    ``degeneracy`` is 2 for the orbitals of an unpolarized calculation and 1
    for those of one spin channel (as :func:`parsec_python.Occupations.density_from_orbitals`).
    With ``k_point`` the columns are Bloch factors ``u`` and the derivative is
    ``G + ik``.  Hartree units (libxc's ``tau``).
    """

    vectors = np.asarray(wavefunctions)
    occupations = np.asarray(occupations, dtype=np.float64)
    if vectors.ndim != 2 or occupations.shape != (vectors.shape[1],):
        raise ValueError("wavefunctions must be (grid, states) with one occupation per state")
    if volume_element <= 0:
        raise ValueError("volume_element must be positive")
    weights = occupations
    selected = np.flatnonzero(weights != 0.0)
    tau = np.zeros(vectors.shape[0], dtype=np.float64)
    if selected.size == 0:
        return tau
    block = vectors[:, selected]
    weights = weights[selected]
    complex_block = np.iscomplexobj(block)
    k = None if k_point is None else np.asarray(k_point, dtype=np.float64)
    for axis, matrix in enumerate(gradient.matrices):
        if complex_block:
            real_view = np.ascontiguousarray(block).view(np.float64)
            derivative = (matrix @ real_view).view(np.complex128)
        else:
            derivative = matrix @ block
        if k is not None and k[axis] != 0.0:
            derivative = derivative + 1j * k[axis] * block
        if np.iscomplexobj(derivative):
            squares = derivative.real * derivative.real + derivative.imag * derivative.imag
        else:
            squares = derivative * derivative
        tau += squares @ weights
    return (0.5 * float(degeneracy) / float(volume_element)) * tau


_GRADIENT_CHOICES = ("auto", "density", "orbital")
_DEFAULT_GRADIENT = "auto"


def meta_gga_gradient() -> str:
    """``PARSEC_META_GGA_GRADIENT``: ``auto`` (default), ``density`` or ``orbital`` (see :func:`meta_gga_gradient_form`).

    ``orbital`` takes the density gradient of a meta-GGA from the orbitals,
    ``g = (d/h^3) sum_m f_m 2 q_m G q_m``, the form whose discrete von
    Weizsaecker density ``|g|^2/(8n)`` never exceeds ``tau`` (Cauchy-Schwarz)
    and equals it where one orbital carries the density; ``density`` takes
    ``G n``; ``auto`` decides per functional.
    """

    value = __import__("os").environ.get("PARSEC_META_GGA_GRADIENT", _DEFAULT_GRADIENT).strip().lower()
    if value not in _GRADIENT_CHOICES:
        raise ValueError("PARSEC_META_GGA_GRADIENT must be auto, density or orbital")
    return value


def _probe_unpolarized(evaluator, rho, sigma, tau):
    lapl = np.zeros_like(rho) if _needs(evaluator, "needs_laplacian") else None
    floored = _floored(evaluator)
    options = tau_floor_options(evaluator)
    given, below = weizsaecker_floor(np, rho, sigma, tau, **options) if floored else (tau, None)
    _, vrho, vsigma, vtau, _ = evaluator.unpolarized(rho, sigma, given, lapl)
    if floored:
        vrho, vsigma, vtau = weizsaecker_chain(np, rho, sigma, below, vrho, vsigma, vtau)
    return [np.asarray(value, dtype=np.float64) for value in (vrho, vsigma, vtau)]


def _probe_polarized(evaluator, up, down, s_uu, s_ud, s_dd, t_up, t_down):
    lapl = np.zeros_like(up) if _needs(evaluator, "needs_laplacian") else None
    floored = _floored(evaluator)
    options = tau_floor_options(evaluator)
    if floored:
        (t_up, below_up), (t_down, below_down) = (
            weizsaecker_floor(np, up, s_uu, t_up, **options),
            weizsaecker_floor(np, down, s_dd, t_down, **options),
        )
    values = list(evaluator.polarized(up, down, s_uu, s_ud, s_dd, t_up, t_down, lapl, lapl))
    if floored:
        values[1], values[3], values[6] = weizsaecker_chain(np, up, s_uu, below_up, values[1], values[3], values[6])
        values[2], values[5], values[7] = weizsaecker_chain(
            np, down, s_dd, below_down, values[2], values[5], values[7]
        )
    return [np.asarray(values[index], dtype=np.float64) for index in range(1, 8)]


def smooth_at_weizsaecker(evaluator, *, step: float = 1.0e-6, tolerance: float = 1.0e-3) -> bool:
    """Whether the derivatives of a meta-GGA are continuous across ``tau = tau_W`` (a probe of a few points).

    The orbital form of the density gradient puts every point of a
    one-orbital region (all of H2, the tails of every molecule) at
    ``tau = tau_W`` up to round-off.  A functional with a kink there (TPSS's
    correlation clamps ``z = tau_W/tau <= 1``; any evaluator floored at
    ``tau_W``) has derivatives that jump between the two sides, and the
    points flip sides from one SCF iteration to the next.  The probe
    evaluates ``vrho``, ``vsigma`` and ``vtau`` at ``tau_W (1 -+ step)`` for
    densities ``1e-3 .. 1``, reduced gradients 0.3 and 2, unpolarized and
    with two spin channels, through the grid layer's floor where the
    evaluator has one; a relative jump above ``tolerance`` is a kink.
    """

    rho = np.repeat(np.array([1.0e-3, 5.0e-2, 1.0]), 2)
    reduced = np.tile(np.array([0.3, 2.0]), 3)
    gradient = 2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0) * rho ** (4.0 / 3.0) * reduced
    sigma = gradient * gradient
    weizsaecker = sigma / (8.0 * rho)

    def jumped(lower, upper) -> bool:
        for left, right in zip(lower, upper):
            if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
                return True
            scale = np.maximum(np.abs(left), np.abs(right))
            floor = 1.0e-12 * max(float(np.max(scale)), 1.0e-300)
            if np.any(np.abs(left - right) > tolerance * (scale + floor)):
                return True
        return False

    with np.errstate(all="ignore"):
        if jumped(
            _probe_unpolarized(evaluator, rho, sigma, weizsaecker * (1.0 - step)),
            _probe_unpolarized(evaluator, rho, sigma, weizsaecker * (1.0 + step)),
        ):
            return False
        for share in (0.5, 0.7):
            up, down = share * rho, (1.0 - share) * rho
            s_uu, s_dd = share * share * sigma, (1.0 - share) ** 2 * sigma
            s_ud = share * (1.0 - share) * sigma
            w_up, w_down = s_uu / (8.0 * up), s_dd / (8.0 * down)
            if jumped(
                _probe_polarized(evaluator, up, down, s_uu, s_ud, s_dd, w_up * (1 - step), w_down * (1 - step)),
                _probe_polarized(evaluator, up, down, s_uu, s_ud, s_dd, w_up * (1 + step), w_down * (1 + step)),
            ):
                return False
    return True


_GRADIENT_FORMS: dict = {}


def meta_gga_gradient_form(scf_settings) -> tuple[str, str]:
    """``(form, reason)``: the density gradient of the meta-GGA of an SCF's settings, ``orbital`` or ``density``.

    ``PARSEC_META_GGA_GRADIENT=orbital|density`` decides; with ``auto`` the
    orbital form where the functional is smooth across ``tau = tau_W``
    (:func:`smooth_at_weizsaecker`: SCAN, r2SCAN, TPSS exchange) and the
    density form where it has a kink there (TPSS-type correlation, an
    evaluator floored at ``tau_W``).
    """

    request = meta_gga_gradient()
    if request != "auto":
        return request, "PARSEC_META_GGA_GRADIENT=" + request
    evaluator = meta_gga_evaluator(scf_settings)
    label = getattr(scf_settings, "xc_functional", None)
    key = (label, tuple(sorted(tau_floor_options(evaluator).items())), meta_gga_density_cut())
    smooth = _GRADIENT_FORMS.get(key)
    if smooth is None:
        smooth = smooth_at_weizsaecker(evaluator)
        _GRADIENT_FORMS[key] = smooth
    if smooth:
        return "orbital", "auto: the functional is smooth across tau = tau_W"
    return "density", "auto: the functional has a kink at tau = tau_W"


def orbital_density_gradient(
    wavefunctions: np.ndarray,
    occupations: np.ndarray,
    gradient: GridGradient,
    volume_element: float,
    *,
    degeneracy: float = 2.0,
) -> np.ndarray:
    """``g = (d/h^3) sum_m f_m 2 Re(q_m^* (G q_m))``, shape ``(3, n)``: the density gradient from the orbitals.

    Complex columns are Bloch factors ``u`` (or orbitals): ``(G + ik)`` gives
    the same ``g``, since ``Re(i k |u|^2) = 0``.
    """

    vectors = np.asarray(wavefunctions)
    if not np.iscomplexobj(vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
    occupations = np.asarray(occupations, dtype=np.float64)
    result = np.zeros((3, vectors.shape[0]), dtype=np.float64)
    selected = np.flatnonzero(occupations != 0.0)
    if selected.size == 0:
        return result
    block = vectors[:, selected]
    weights = occupations[selected]
    complex_block = np.iscomplexobj(block)
    if complex_block:
        block = np.ascontiguousarray(block, dtype=np.complex128)
    for axis, matrix in enumerate(gradient.matrices):
        if complex_block:
            derivative = (matrix @ block.view(np.float64)).view(np.complex128)
            result[axis] = (block.real * derivative.real + block.imag * derivative.imag) @ weights
        else:
            result[axis] = (block * (matrix @ block)) @ weights
    return (2.0 * float(degeneracy) / float(volume_element)) * result


def kinetic_double_counting(kinetic_potential, tau, volume_element: float, weights=None) -> float:
    """``2 h^3 sum_i vtau_i tau_i`` (Ry): the band-energy part of ``H_tau`` for the orbitals of ``tau``.

    ``weights`` are the orbit multiplicities of fields kept on symmetry orbits.
    """

    product = np.asarray(kinetic_potential, dtype=np.float64) * np.asarray(tau, dtype=np.float64)
    if weights is not None:
        product = product * np.asarray(weights, dtype=np.float64)
    return float(2.0 * volume_element * np.sum(product))


# ----- the operator -----------------------------------------------------------------


@dataclass(frozen=True)
class KineticPotentialOperator:
    """``H_tau = sum_c (G_c + i k_c)^H diag(vtau) (G_c + i k_c)`` (Rydberg) on grid vectors.

    ``kinetic_potential`` is ``vtau``; ``k_point`` is ``None`` for real
    orbitals (clusters, the Gamma point).  A term of
    :class:`parsec_python.Hamiltonian.KohnShamHamiltonian` (and of the k-point
    Hamiltonians) beside the kinetic, local and nonlocal ones.
    """

    gradient: GridGradient
    kinetic_potential: np.ndarray
    k_point: np.ndarray | None = None
    # The orbital form of the density gradient: W = de/dg (3, n), the term 2 sum_c [W_c G_c + G_c^T W_c].
    gradient_flux: np.ndarray | None = None

    def __post_init__(self) -> None:
        potential = np.ascontiguousarray(self.kinetic_potential, dtype=np.float64)
        if potential.shape != (self.gradient.size,):
            raise ValueError("vtau does not match the gradient operators")
        object.__setattr__(self, "kinetic_potential", potential)
        if self.k_point is not None:
            k = np.asarray(self.k_point, dtype=np.float64)
            if k.shape != (3,):
                raise ValueError("k_point must have shape (3,)")
            object.__setattr__(self, "k_point", None if not np.any(k) else k)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.gradient.size, self.gradient.size)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        block = np.asarray(vectors)
        was_vector = block.ndim == 1
        if was_vector:
            block = block[:, None]
        potential = self.kinetic_potential[:, None]
        complex_block = np.iscomplexobj(block) or self.k_point is not None
        if complex_block:
            block = np.ascontiguousarray(block, dtype=np.complex128)
        result = None
        for axis, matrix in enumerate(self.gradient.matrices):
            if complex_block:
                plain = (matrix @ block.view(np.float64)).view(np.complex128)
                derivative = plain
                if self.k_point is not None and self.k_point[axis] != 0.0:
                    derivative = plain + 1j * self.k_point[axis] * block
                flux = np.ascontiguousarray(potential * derivative)
                term = (matrix.T @ flux.view(np.float64)).view(np.complex128)
                if self.k_point is not None and self.k_point[axis] != 0.0:
                    term -= 1j * self.k_point[axis] * flux
                if self.gradient_flux is not None:
                    # 2 [W (G + ik) + (G + ik)^H W] u: the k parts cancel for a real W.
                    flux_axis = np.asarray(self.gradient_flux[axis], dtype=np.float64)[:, None]
                    weighted = np.ascontiguousarray(flux_axis * block)
                    term = term + 2.0 * (
                        flux_axis * plain + (matrix.T @ weighted.view(np.float64)).view(np.complex128)
                    )
            else:
                term = matrix.T @ (potential * (matrix @ block))
                if self.gradient_flux is not None:
                    flux_axis = np.asarray(self.gradient_flux[axis])[:, None]
                    term = term + 2.0 * (flux_axis * (matrix @ block) + matrix.T @ (flux_axis * block))
            result = term if result is None else result + term
        return result[:, 0] if was_vector else result

    @property
    def spectral_margin(self) -> float:
        """Gershgorin's bound of the eigenvalues of the term (Ry): the largest row sum of a bound of its ``|M|``.

        ``sum_c |D_c|^T (|vtau| |D_c| 1) + 2 |W_c| |G_c| 1 + 2 |G_c|^T |W_c|``
        with ``|D_c| = |G_c| + |k_c|``.  A narrow spike of ``vtau`` (low
        density) gives localized eigenvalues that a short Lanczos run does
        not see; the eigensolvers' bounds add this one.
        """

        size = self.gradient.size
        ones = np.ones(size, dtype=np.float64)
        potential = np.abs(self.kinetic_potential)
        total = np.zeros(size, dtype=np.float64)
        for axis, matrix in enumerate(self.gradient.matrices):
            absolute = abs(matrix)
            if self.k_point is not None and self.k_point[axis] != 0.0:
                absolute = absolute + abs(float(self.k_point[axis])) * sp.identity(size, format="csr")
            rows = np.asarray(absolute @ ones).ravel()
            weights = potential * rows
            if self.gradient_flux is not None:
                flux = 2.0 * np.abs(np.asarray(self.gradient_flux[axis], dtype=np.float64))
                weights = weights + flux
                total += flux * rows
            total += np.asarray(absolute.T @ weights).ravel()
        return float(total.max()) if size else 0.0

    def as_sparse(self) -> sp.csr_matrix:
        """The real operator as a sparse matrix (diagnostics; real orbitals only)."""

        if self.k_point is not None:
            raise NotImplementedError("the sparse form is built for real orbitals only")
        diagonal = sp.diags(self.kinetic_potential)
        total = None
        for axis, matrix in enumerate(self.gradient.matrices):
            term = matrix.T @ diagonal @ matrix
            if self.gradient_flux is not None:
                flux = sp.diags(np.asarray(self.gradient_flux[axis], dtype=np.float64))
                term = term + 2.0 * (flux @ matrix + matrix.T @ flux)
            total = term if total is None else total + term
        return sp.csr_matrix(total)


# ----- energies and potentials ---------------------------------------------------------


def _laplacian_of(negative_laplacian, density):
    if negative_laplacian is None:
        raise ValueError("this meta-GGA needs the Laplacian of the density: pass negative_laplacian")
    return -(negative_laplacian @ density)


def _laplacian_adjoint(negative_laplacian, values):
    return -(negative_laplacian.T @ values)


def _needs(evaluator, name: str) -> bool:
    return bool(getattr(evaluator, name, False))


def _clamped_count(sigma, density, tau) -> int:
    return int(np.count_nonzero((density > _DENSITY_THRESHOLD) & (sigma > 8.0 * density * tau)))


def weizsaecker_floor(xp, density, sigma, tau, *, weizsaecker: bool = True, minimum: float | None = None):
    """``(tau_eff, below)``: ``tau_eff = max(tau, floor)`` of one channel and which floor acts where.

    The floors of the ``tau`` an evaluator sees, which it declares
    (:func:`tau_floor_options`): ``tau_floor = "weizsaecker"``, the von
    Weizsaecker density ``sigma/(8n)``, for an evaluator that is not defined
    (or has a kink) below it, as libxc's spin-polarized SCAN, which returns
    NaN there, or a functional of ``z = tau_W/tau <= 1``; the finite-difference
    ``tau`` of the grid can fall below ``tau_W`` (tails, and near nuclei where
    the frozen core density enters ``n`` and not ``tau``).  ``tau_minimum``,
    an absolute floor (``tau`` of points where every orbital has a vanishing
    gradient: a functional of ``z`` is ``0/0`` there).  The evaluator sees
    ``tau_eff``: the energy of a point is never dropped, it is that of the
    regularized input, and :func:`weizsaecker_chain` gives the exact
    derivatives of that energy.  ``below`` holds 1 where ``tau_W`` acts, 2
    where the minimum acts, 0 elsewhere.
    """

    positive = density > _DENSITY_THRESHOLD
    safe = xp.where(positive, density, 1.0)
    floor = xp.where(positive, sigma / (8.0 * safe), 0.0) if weizsaecker else xp.zeros_like(tau)
    code = xp.where(positive & (tau < floor), 1, 0).astype(xp.int8)
    if minimum is not None:
        absolute = positive & (tau < minimum) & (floor < minimum)
        code = xp.where(absolute, 2, code).astype(xp.int8)
        floor = xp.where(absolute, minimum, floor)
    return xp.where(code > 0, floor, tau), code


def weizsaecker_chain(xp, density, sigma, below, vrho, vsigma, vtau):
    """The derivatives of ``e(n, sigma, max(tau, floor))`` from those at ``tau_eff`` (:func:`weizsaecker_floor`).

    Where ``tau_W`` acts: ``d/dn += vtau d(sigma/8n)/dn = -vtau sigma/(8 n^2)``,
    ``d/dsigma += vtau/(8n)``, ``d/dtau = 0``; where the minimum acts
    ``d/dtau = 0``.  ``below`` may also be a boolean mask of the ``tau_W`` floor.
    """

    weizsaecker = below == 1
    safe = xp.where(weizsaecker, density, 1.0)
    vrho = vrho - xp.where(weizsaecker, vtau * sigma / (8.0 * safe * safe), 0.0)
    vsigma = vsigma + xp.where(weizsaecker, vtau / (8.0 * safe), 0.0)
    vtau = xp.where(below > 0, 0.0, vtau)
    return vrho, vsigma, vtau


def tau_floor_options(evaluator) -> dict:
    """The floors an evaluator declares: ``tau_floor = "weizsaecker"`` and ``tau_minimum`` (Hartree, per channel)."""

    minimum = getattr(evaluator, "tau_minimum", None)
    return dict(
        weizsaecker=getattr(evaluator, "tau_floor", None) == "weizsaecker",
        minimum=None if minimum is None else float(minimum),
    )


def _floored(evaluator) -> bool:
    options = tau_floor_options(evaluator)
    return bool(options["weizsaecker"] or options["minimum"] is not None)


def meta_gga_xc(
    evaluator,
    valence_density: np.ndarray,
    tau: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    *,
    negative_laplacian=None,
    gradient: GridGradient | None = None,
    name: str = "the meta-GGA",
    density_gradient: np.ndarray | None = None,
) -> MetaGGAXCResult:
    """Unpolarized meta-GGA energy, ``dE/dn`` and ``vtau`` on the grid (Rydberg).

    ``tau`` is the kinetic-energy density of the valence orbitals (Hartree,
    :func:`kinetic_energy_density`); ``negative_laplacian`` (the kinetic
    stencil) is needed by evaluators with ``needs_laplacian``.
    """

    density = _gga_density(valence_density, core_density, grid, name)
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape != density.shape:
        raise ValueError(f"the kinetic-energy density of {name} does not match the grid")
    gradient = grid_gradient(grid) if gradient is None else gradient
    orbital = density_gradient is not None
    if orbital:
        gradients = np.asarray(density_gradient, dtype=np.float64)
        if core_density is not None:
            gradients = gradients + gradient.apply(np.asarray(core_density, dtype=np.float64))
    else:
        gradients = gradient.apply(density)
    sigma = np.einsum("ij,ij->j", gradients, gradients)
    lapl = _laplacian_of(negative_laplacian, density) if _needs(evaluator, "needs_laplacian") else None
    floored = _floored(evaluator) and _needs(evaluator, "needs_tau")
    given_tau = weizsaecker_floor(np, density, sigma, tau, **tau_floor_options(evaluator)) if floored else (tau, None)
    energy, vrho, vsigma, vtau, vlapl = evaluator.unpolarized(
        density, sigma, given_tau[0] if _needs(evaluator, "needs_tau") else None, lapl
    )
    if floored:
        vrho, vsigma, vtau = weizsaecker_chain(np, density, sigma, given_tau[1], vrho, vsigma, vtau)
    density_potential = gradient.adjoint(2.0 * vsigma[None, :] * gradients)
    potential = vrho if orbital else vrho + density_potential
    if vlapl is not None:
        potential = potential + _laplacian_adjoint(negative_laplacian, vlapl)
    extra = {}
    if orbital:
        core_potential = vrho + density_potential
        if vlapl is not None:
            core_potential = core_potential + _laplacian_adjoint(negative_laplacian, vlapl)
        extra = dict(gradient_flux=2.0 * vsigma[None, :] * gradients, core_potential=2.0 * core_potential)
    energy_density = 2.0 * np.asarray(energy, dtype=np.float64)
    epsilon = np.zeros_like(density)
    positive = density > _DENSITY_THRESHOLD
    epsilon[positive] = energy_density[positive] / density[positive]
    return MetaGGAXCResult(
        potential=2.0 * np.asarray(potential, dtype=np.float64),
        energy_per_electron=epsilon,
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
        kinetic_potential=np.zeros_like(density) if vtau is None else np.asarray(vtau, dtype=np.float64),
        fermi_hole_clamped=_clamped_count(sigma, density, tau),
        **extra,
    )


def meta_gga_xc_spin(
    evaluator,
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    tau_up: np.ndarray,
    tau_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
    *,
    negative_laplacian=None,
    gradient: GridGradient | None = None,
    name: str = "the meta-GGA",
    density_gradients=None,
) -> MetaGGASpinXCResult:
    """Spin-polarized meta-GGA energy, ``dE/dn_a`` and ``vtau_a`` (Rydberg).

    Each channel sees its valence density plus half of the frozen core
    density; ``tau_a`` is that of the channel's valence orbitals.
    """

    up, down = spin_densities_with_core(valence_up, valence_down, core_density, functional=name)
    if up.shape != (grid.size,):
        raise ValueError(f"the densities of {name} do not match the real-space grid")
    tau_up = np.asarray(tau_up, dtype=np.float64)
    tau_down = np.asarray(tau_down, dtype=np.float64)
    if tau_up.shape != up.shape or tau_down.shape != up.shape:
        raise ValueError(f"the kinetic-energy densities of {name} do not match the grid")
    gradient = grid_gradient(grid) if gradient is None else gradient
    orbital = density_gradients is not None
    if orbital:
        # g_a of the orbitals of channel a, with half the core density's gradient.
        grad_up = np.asarray(density_gradients[0], dtype=np.float64)
        grad_down = np.asarray(density_gradients[1], dtype=np.float64)
        if core_density is not None:
            half = gradient.apply(0.5 * np.asarray(core_density, dtype=np.float64))
            grad_up = grad_up + half
            grad_down = grad_down + half
    else:
        grad_up = gradient.apply(up)
        grad_down = gradient.apply(down)
    s_uu = np.einsum("ij,ij->j", grad_up, grad_up)
    s_ud = np.einsum("ij,ij->j", grad_up, grad_down)
    s_dd = np.einsum("ij,ij->j", grad_down, grad_down)
    laplacians = (None, None)
    if _needs(evaluator, "needs_laplacian"):
        laplacians = (_laplacian_of(negative_laplacian, up), _laplacian_of(negative_laplacian, down))
    taus = (tau_up, tau_down) if _needs(evaluator, "needs_tau") else (None, None)
    floored = _floored(evaluator) and _needs(evaluator, "needs_tau")
    if floored:
        (floor_up, below_up), (floor_down, below_down) = (
            weizsaecker_floor(np, up, s_uu, tau_up, **tau_floor_options(evaluator)),
            weizsaecker_floor(np, down, s_dd, tau_down, **tau_floor_options(evaluator)),
        )
        taus = (floor_up, floor_down)
    (energy, v_up, v_down, vs_uu, vs_ud, vs_dd, vt_up, vt_down, vl_up, vl_down) = evaluator.polarized(
        up, down, s_uu, s_ud, s_dd, taus[0], taus[1], laplacians[0], laplacians[1]
    )
    if floored:
        v_up, vs_uu, vt_up = weizsaecker_chain(np, up, s_uu, below_up, v_up, vs_uu, vt_up)
        v_down, vs_dd, vt_down = weizsaecker_chain(np, down, s_dd, below_down, v_down, vs_dd, vt_down)
    flux_up = 2.0 * vs_uu[None, :] * grad_up + vs_ud[None, :] * grad_down
    flux_down = 2.0 * vs_dd[None, :] * grad_down + vs_ud[None, :] * grad_up
    divergence_up = gradient.adjoint(flux_up)
    divergence_down = gradient.adjoint(flux_down)
    potential_up = v_up if orbital else v_up + divergence_up
    potential_down = v_down if orbital else v_down + divergence_down
    if vl_up is not None:
        potential_up = potential_up + _laplacian_adjoint(negative_laplacian, vl_up)
        potential_down = potential_down + _laplacian_adjoint(negative_laplacian, vl_down)
    extra = {}
    if orbital:
        # dE/dn_core: half of the core density in each channel.
        core_up = v_up + divergence_up
        core_down = v_down + divergence_down
        if vl_up is not None:
            core_up = core_up + _laplacian_adjoint(negative_laplacian, vl_up)
            core_down = core_down + _laplacian_adjoint(negative_laplacian, vl_down)
        extra = dict(
            gradient_flux_up=flux_up,
            gradient_flux_down=flux_down,
            core_potential=core_up + core_down,
        )
    energy_density = 2.0 * np.asarray(energy, dtype=np.float64)
    zero = np.zeros_like(up)
    clamped = _clamped_count(s_uu, up, tau_up) + _clamped_count(s_dd, down, tau_down)
    return MetaGGASpinXCResult(
        potential_up=2.0 * np.asarray(potential_up, dtype=np.float64),
        potential_down=2.0 * np.asarray(potential_down, dtype=np.float64),
        energy_density=energy_density,
        total_energy=float(grid.volume_element * np.sum(energy_density)),
        kinetic_potential_up=zero.copy() if vt_up is None else np.asarray(vt_up, dtype=np.float64),
        kinetic_potential_down=zero.copy() if vt_down is None else np.asarray(vt_down, dtype=np.float64),
        fermi_hole_clamped=clamped,
        **extra,
    )


# ----- the density switch of the orbital terms -------------------------------------


def meta_gga_density_cut() -> tuple[float, float] | None:
    """``PARSEC_META_GGA_DENSITY_CUT``: ``off`` or ``low,high`` (bohr^-3), see :class:`DensitySwitchedMetaGGA`."""

    import os

    value = os.environ.get("PARSEC_META_GGA_DENSITY_CUT", _DEFAULT_DENSITY_CUT).strip().lower()
    if value in ("off", "none", ""):
        return None
    try:
        low, high = (float(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("PARSEC_META_GGA_DENSITY_CUT must be off or low,high") from error
    if not 0.0 < low < high:
        raise ValueError("PARSEC_META_GGA_DENSITY_CUT needs 0 < low < high")
    return low, high


_DEFAULT_DENSITY_CUT = "off"

# (3/10)(3 pi^2)^(2/3) and (3/10)(6 pi^2)^(2/3): tau of the uniform gas, unpolarized and of one channel.
_UNIFORM_TAU = 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0)
_UNIFORM_TAU_CHANNEL = 0.3 * (6.0 * np.pi**2) ** (2.0 / 3.0)


def _array_module(array):
    module = type(array).__module__
    if module.startswith("cupy"):
        import cupy

        return cupy
    return np


class DensitySwitchedMetaGGA:
    """A meta-GGA whose dependence on ``tau`` is switched off smoothly at low density.

    ``e = S(n) e_m(n, sigma, tau) + (1 - S(n)) e_m(n, sigma, tau_1(n, sigma))``
    with ``tau_1 = sigma/(8n) + tau_unif(n)`` (``alpha = 1`` of SCAN: the
    functional's own GGA limit, per channel in spin) and ``S`` the quintic
    switch from 0 at ``low`` to 1 at ``high`` in ``log n`` (``n`` the total
    density).  The derivatives are those of this energy:
    ``vtau = S vtau_m``, ``vsigma`` and ``vrho`` hold the chain rule through
    ``tau_1`` and ``S'(n) (e_m - e_GGA)``, so the potential stays the exact
    derivative.  Where the density is below ``low`` the orbital terms vanish:
    ``vtau`` of SCAN grows like ``n^(-1/3)`` into the vacuum, where its narrow
    spikes give the Hamiltonian localized eigenvalues far above the rest of
    its spectrum and widen the Chebyshev filter's interval.

    Same interface as the wrapped evaluator (NumPy or CuPy arrays).
    """

    family = "mgga"

    def __init__(self, evaluator, low: float, high: float) -> None:
        if not 0.0 < low < high:
            raise ValueError("the density switch needs 0 < low < high")
        self.evaluator = evaluator
        self.low = float(low)
        self.high = float(high)
        for name in ("needs_tau", "needs_laplacian", "supports_device", "tau_floor", "name", "label"):
            if hasattr(evaluator, name):
                setattr(self, name, getattr(evaluator, name))

    def __getattr__(self, name):
        return getattr(self.evaluator, name)

    def _switch(self, xp, density):
        span = np.log(self.high / self.low)
        safe = xp.where(density > self.low, density, self.low)
        t = xp.clip((xp.log(safe) - np.log(self.low)) / span, 0.0, 1.0)
        switch = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        slope = 30.0 * t * t * (1.0 - t) * (1.0 - t) / (span * safe)
        return switch, slope

    @staticmethod
    def _limit_tau(xp, density, sigma, uniform):
        positive = density > 1.0e-30
        safe = xp.where(positive, density, 1.0)
        tau = xp.where(positive, sigma / (8.0 * safe) + uniform * safe ** (5.0 / 3.0), 0.0)
        # d tau_1/dn and d tau_1/dsigma
        by_density = xp.where(positive, -sigma / (8.0 * safe * safe) + (5.0 / 3.0) * uniform * safe ** (2.0 / 3.0), 0.0)
        by_sigma = xp.where(positive, 1.0 / (8.0 * safe), 0.0)
        return tau, by_density, by_sigma

    def unpolarized(self, rho, sigma, tau, lapl=None):
        xp = _array_module(rho)
        switch, slope = self._switch(xp, rho)
        e_m, vr_m, vs_m, vt_m, vl_m = self.evaluator.unpolarized(rho, sigma, tau, lapl)
        if not bool(xp.any(switch < 1.0)):
            return e_m, vr_m, vs_m, vt_m, vl_m
        limit, by_density, by_sigma = self._limit_tau(xp, rho, sigma, _UNIFORM_TAU)
        e_g, vr_g, vs_g, vt_g, vl_g = self.evaluator.unpolarized(rho, sigma, limit, lapl)
        vr_g = vr_g + vt_g * by_density
        vs_g = vs_g + vt_g * by_sigma
        rest = 1.0 - switch
        energy = switch * e_m + rest * e_g
        vrho = switch * vr_m + rest * vr_g + slope * (e_m - e_g)
        vsigma = switch * vs_m + rest * vs_g
        vtau = switch * vt_m
        vlapl = None if vl_m is None else switch * vl_m + rest * vl_g
        return energy, vrho, vsigma, vtau, vlapl

    def polarized(self, rho_up, rho_down, s_uu, s_ud, s_dd, tau_up, tau_down, lapl_up=None, lapl_down=None):
        xp = _array_module(rho_up)
        switch, slope = self._switch(xp, rho_up + rho_down)
        meta = self.evaluator.polarized(rho_up, rho_down, s_uu, s_ud, s_dd, tau_up, tau_down, lapl_up, lapl_down)
        if not bool(xp.any(switch < 1.0)):
            return meta
        limit_up, by_density_up, by_sigma_up = self._limit_tau(xp, rho_up, s_uu, _UNIFORM_TAU_CHANNEL)
        limit_down, by_density_down, by_sigma_down = self._limit_tau(xp, rho_down, s_dd, _UNIFORM_TAU_CHANNEL)
        gga = list(self.evaluator.polarized(rho_up, rho_down, s_uu, s_ud, s_dd, limit_up, limit_down, lapl_up, lapl_down))
        # (e, v_up, v_down, vs_uu, vs_ud, vs_dd, vt_up, vt_down, vl_up, vl_down)
        gga[1] = gga[1] + gga[6] * by_density_up
        gga[2] = gga[2] + gga[7] * by_density_down
        gga[3] = gga[3] + gga[6] * by_sigma_up
        gga[5] = gga[5] + gga[7] * by_sigma_down
        rest = 1.0 - switch
        difference = slope * (meta[0] - gga[0])
        result = [switch * meta[0] + rest * gga[0]]
        result.append(switch * meta[1] + rest * gga[1] + difference)
        result.append(switch * meta[2] + rest * gga[2] + difference)
        for index in (3, 4, 5):
            result.append(switch * meta[index] + rest * gga[index])
        result.append(switch * meta[6])
        result.append(switch * meta[7])
        for index in (8, 9):
            result.append(None if meta[index] is None else switch * meta[index] + rest * gga[index])
        return tuple(result)


def _with_density_cut(evaluator):
    cut = meta_gga_density_cut()
    if evaluator is None or cut is None or not getattr(evaluator, "needs_tau", False):
        return evaluator
    return DensitySwitchedMetaGGA(evaluator, *cut)


# ----- selection --------------------------------------------------------------------

#: Meta-GGAs of this package (``Correlation_Type``): label -> evaluator factory.
def _native_scan():
    from .scan import NativeScan

    scan = NativeScan()
    import os

    if os.environ.get("PARSEC_SCAN_TAU_FLOOR", "").strip().lower() == "weizsaecker":
        scan.tau_floor = "weizsaecker"
    return scan


NATIVE_META_GGA_FUNCTIONALS = {
    "scan": _native_scan,
}


def _libxc_provider(scf_settings):
    """The evaluator of a libxc meta-GGA (``libxc:mgga_...``), ``None`` for anything else."""

    label = getattr(scf_settings, "xc_functional", None)
    from .libxc import is_libxc_functional, libxc_semilocal, parse_libxc_spec

    if not is_libxc_functional(label) or not parse_libxc_spec(label).is_meta_gga:
        return None
    return libxc_semilocal(label)


#: Providers of meta-GGA evaluators beyond the native ones: ``provider(scf_settings)``
#: returns an evaluator for the settings or ``None`` (libxc, :mod:`parsec_python.V_xc.libxc`).
_META_GGA_PROVIDERS: list = [_libxc_provider]
_PROVIDED: dict = {}


def register_meta_gga_provider(provider) -> None:
    """Add ``provider(scf_settings) -> evaluator | None`` to the meta-GGA selection."""

    if provider not in _META_GGA_PROVIDERS:
        _META_GGA_PROVIDERS.append(provider)


def meta_gga_evaluator(scf_settings):
    """The meta-GGA evaluator of an SCF's settings, ``None`` for a functional without ``tau`` or Laplacian."""

    label = getattr(scf_settings, "xc_functional", None)
    factory = NATIVE_META_GGA_FUNCTIONALS.get(label)
    if factory is not None:
        return _with_density_cut(factory())
    known = _PROVIDED.get(label)
    if known is not None:
        return _with_density_cut(known)
    for provider in _META_GGA_PROVIDERS:
        evaluator = provider(scf_settings)
        if evaluator is not None and getattr(evaluator, "family", None) == "mgga":
            # One evaluator per label and process (a libxc handle is opened once).
            _PROVIDED[label] = evaluator
            return _with_density_cut(evaluator)
    return None


def meta_gga_lines(scf_settings) -> list[str]:
    """The functional lines of ``parsec.out`` for a meta-GGA of this package."""

    label = getattr(scf_settings, "xc_functional", None)
    descriptions = {
        "scan": " meta-GGA, SCAN of Sun, Ruzsinszky and Perdew (PRL 115, 036402 (2015)), native",
    }
    form, reason = meta_gga_gradient_form(scf_settings)
    return [
        f" Exchange-Correlation functional is {label}",
        descriptions.get(label, " meta-GGA"),
        " tau from the orbitals; the first iteration with PBE (vtau = 0)",
        f" density gradient: {form} form ({reason})",
    ]


def is_meta_gga(scf_settings) -> bool:
    """Whether the functional of these settings depends on ``tau`` or the Laplacian."""

    label = getattr(scf_settings, "xc_functional", None)
    if label in NATIVE_META_GGA_FUNCTIONALS:
        return True
    from .libxc import is_libxc_functional, parse_libxc_spec

    if is_libxc_functional(label):
        return parse_libxc_spec(label).is_meta_gga
    return bool(_META_GGA_PROVIDERS[1:]) and meta_gga_evaluator(scf_settings) is not None


__all__ = [
    "GridGradient",
    "KineticPotentialOperator",
    "MetaGGASpinXCResult",
    "MetaGGAXCResult",
    "NATIVE_META_GGA_FUNCTIONALS",
    "grid_gradient",
    "is_meta_gga",
    "kinetic_double_counting",
    "kinetic_energy_density",
    "meta_gga_evaluator",
    "meta_gga_lines",
    "meta_gga_xc",
    "meta_gga_xc_spin",
    "register_meta_gga_provider",
    "tau_floor_options",
    "weizsaecker_chain",
    "weizsaecker_floor",
]
