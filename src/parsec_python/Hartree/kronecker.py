"""Kronecker-product Poisson solver for many densities of an isolated system.

Exact exchange needs the potential of every product of two occupied orbitals
(:mod:`parsec_python.V_xc.exact_exchange`).  The conjugate-gradient Hartree
solver (:func:`parsec_python.Hartree.solve_hartree`) costs tens to hundreds of
Laplacian applications per density; this module solves the same
finite-difference Poisson problem directly, for a batch of densities at once,
following the Kronecker-product method of the SPARC papers (Jing, Sharma,
Pask, Suryanarayana, arXiv:2501.16572, Sec. 3; Jing, Suryanarayana,
arXiv:2406.16998).  The derivations below are our own.

Discrete problem
----------------
The densities live on the active points of a cluster grid (sphere, box,
ellipsoid or cylinder).  The solve works on the *bounding box* ``B`` of those
points (optionally padded), ``n_x * n_y * n_z`` points.  On ``B`` the
negative Laplacian of the run, with the centered stencil of order ``2M``
(:func:`parsec_python.Laplacian.second_derivative_coefficients`) and
homogeneous Dirichlet ends, is a Kronecker sum

``A_B = -(D_x (+) D_y (+) D_z)``,
``D_d[i, j] = c_|i-j| / h**2`` for ``|i - j| <= M``,

of three symmetric Toeplitz matrices.  With ``D_d = Q_d diag(mu_d) Q_d^T``
(computed once) the eigenvalues of ``A_B`` are
``lambda_ijk = -(mu_x,i + mu_y,j + mu_z,k) > 0`` and every function of
``A_B`` is

``f(A_B) b = (Q_x (x) Q_y (x) Q_z) [f(lambda) o (Q_x^T (x) Q_y^T (x) Q_z^T) b]``:

three batched matrix products along x, y and z, a Hadamard product and three
products back.  For ``n`` points per axis a solve costs ``12 n**4`` flops in
dense GEMMs, independent of the conditioning of the Laplacian.

Isolated boundary values
------------------------
The Rydberg Hartree equation is ``A V = 8 pi rho`` (:mod:`.poisson`).  The
stencil of a point near a face of ``B`` reaches up to ``M`` points beyond
it; there the potential is set to the multipole expansion of the density
about the origin, as the Hartree solver of a sphere does
(:class:`parsec_python.Hartree.MultipoleExpansion`), and moved to the right
hand side:

``A_B V_B = 8 pi rho_B - A_BE V_E``,
``V_E(r) = 2 sum_lm (4 pi / (2l + 1)) q_lm Y_lm(Omega) / r**(l+1)``,
``q_lm = h**3 sum_s rho_s r_s**l Y_lm(Omega_s)``

(real orthonormal harmonics ``Y_lm``; the addition theorem makes this the
expansion of the complex harmonics of :mod:`.poisson` term by term).

Convergence of the boundary
---------------------------
The expansion about the origin converges at an exterior point ``r`` only if
``|r| > |r'|`` for every point ``r'`` that carries density, and its error
then falls like ``(|r'|/|r|)**(L+1)``; beyond that radius it diverges, the
more so the larger ``L``.  The solver therefore pads its box automatically
(``automatic_padding=True``, the default) until the exterior grid point
nearest to the origin beyond every face lies strictly farther from the
origin than the farthest active point, ``source_radius``: then the expansion
converges for any density on the domain, at every exterior point the face
values and the screened tails use.  This is the situation of a sphere
domain, which needs no padding by construction (its exterior points are
outside its radius), so the solves of a sphere are those of the unpadded
box.  A box, ellipsoid or cylinder domain is padded until the faces lie
beyond its farthest corner: an elongated box of 26 x 10 x 10 bohr gives a
58**3-point box at ``h = 0.5`` instead of 52 x 20 x 20 points, and only then
does its boundary converge (the potential error of two charges at
``x = +-8`` falls from 1.8 Ry at ``L = 9`` to the finite-difference level).
Measured alternatives: summing the Coulomb potential directly at the
``6 M n**2`` exterior stencil points (the boundary of the conjugate-gradient
solver on such domains) costs ``6 M n**5`` per density against
``12 n**4`` for a padded solve, and gives nothing for the exterior tails of
the screened kernels; padding is the cheaper of the two at every size.
``padding`` adds vacuum on every side before the automatic padding.

The moments are computed separably: the Cartesian moments
``M_abc = h**3 sum rho x**a y**b z**c`` (``a, b, c <= L``) are three small
matrix products on the box, and ``q_lm = sum C_lm,abc M_abc`` with the exact
polynomial coefficients of the solid harmonics ``r**l Y_lm`` (coordinates
scaled by a length of the box, so that every number is of order one).  The
exterior values of the ``M`` planes beyond each face are folded into one
precomputed matrix that maps the ``K = (L+1)**2`` moments to the correction
of the ``M`` planes inside the six faces (``6 M n**2 K`` numbers).

Screened kernels
----------------
Besides ``1/r`` (``kernel="coulomb"``) the solver gives the potentials of
``erfc(w r)/r`` (``"erfc"``, the short-range kernel of HSE) and
``erf(w r)/r`` (``"erf"``, the long-range kernel of LC and CAM
functionals), again in Rydberg (twice the Hartree-unit integrals).  Their
Fourier transforms are

``4 pi / k**2 * (1 - exp(-k**2 / (4 w**2)))`` (erfc),
``4 pi / k**2 * exp(-k**2 / (4 w**2))`` (erf),

because ``erf(w r)/r`` is the Coulomb potential of the normalized Gaussian
``g_w(r) = (w / sqrt(pi))**3 exp(-w**2 r**2)``, whose transform is
``exp(-k**2 / (4 w**2))``.  (The exponent printed in arXiv:2501.16572 is
not this one.)  The corresponding spectral functions of the discrete
Laplacian are :func:`spectral_function`.

Applied on ``B`` with any fixed boundary values these functions are not the
isolated potentials: ``f(A_B)`` is the kernel of the box with odd image
charges beyond every face, and at HSE's ``w = 0.11/bohr`` the images at
``2 * 9.4 bohr`` (twice the usual vacuum margin) still carry
``erfc(2.1) = 0.003`` of the kernel.  The default treatment
(``screening="split"``) avoids images altogether.  It uses

``v_erf = g_w * v_coulomb``          (``erf(wr)/r = g_w * (1/r)``),
``v_erfc = v_coulomb - v_erf``,

where ``v_coulomb`` is the isolated Coulomb potential of the box problem
above: the finite-difference solution on ``B`` and the multipole expansion
``V_mp`` beyond it.  The discrete convolution with ``g_w`` splits into the
part over ``B``, the separable matrix ``G_B = G_x (x) G_y (x) G_z``,
``G_d[i, j] = h (w / sqrt(pi)) exp(-w**2 (x_i - x_j)**2)``, and the part over
the exterior grid points, which is linear in the moments:

``(g_w * v)(r) = (G_B v_B)(r) + 2 sum_lm (4 pi/(2l+1)) q_lm T_lm(r)``,
``T_lm = sum_(r' not in B) g_w(r - r') h**3 I_lm(r') = g_w * I~_lm - G_B I~_lm``.

``I_lm = Y_lm / r**(l+1)`` is singular at the origin; ``I~_lm`` replaces it
by the potential of the same multipole smeared over a Gaussian of exponent
``beta``, ``I~_lm = Y_lm P(l + 1/2, beta r**2) / r**(l+1)`` (``P`` the
regularized lower incomplete gamma function; Hobson's theorem applied to
``erf(sqrt(beta) r)/r``), equal to ``I_lm`` beyond the box to 1e-17 for
``beta = x_L / d**2`` (``Q(L + 1/2, x_L) = 1e-17``, ``d`` the distance of
the nearest exterior plane).  Its Gaussian convolution is again of that
form, with ``1/gamma = 1/beta + 1/w**2``.  The closed form is the
continuous convolution, the tail a lattice sum: the two agree only while the
smearing is wide on the grid, and the lattice sum of the order-``l`` term
aliases like ``(2 pi/h)**l exp(-(2 pi/h)**2/(4 beta))``.  ``beta`` is
therefore capped, ``beta h**2 <= smearing_limit`` (0.25 by default, which
puts the exponent at ``pi**2/0.25 = 39``); where ``x_L/d**2`` exceeds the
cap, the box is padded until ``d >= h sqrt(x_L/smearing_limit)``.  ``G_B`` acts on the spectral
coefficients of ``v_B`` through ``G_d Q_d``; ``G_d`` is a Gaussian kernel
matrix of small numerical rank (14 at ``w = 0.11`` and 34 at ``w = 0.4``
for a 20-bohr box at ``rank_tolerance = 1e-14``) and is applied in its
truncated eigenbasis.  The erf potential therefore costs less than a
Coulomb solve (no full inverse transform; 0.8 of it measured on an A100 at
``n = 170``) and the erfc potential a little more (1.2).  The tails ``T``
are stored on the active points, ``K n_active`` numbers (2 GB at
``n_active = 2.6e6``, ``L = 9``).  The approximations beyond those of the
Coulomb solve are the truncation of the eigenbasis (relative
``rank_tolerance``) and the aliasing of the smeared multipoles, which the
cap keeps below the finite-difference error; the finite-difference error of
the erf part is that of the Coulomb solve filtered by
``exp(-k**2/(4 w**2))``, negligible.

``screening="spectral"`` is the plain alternative, kept for comparison: it
applies ``8 pi f(lambda)`` to the density on a box enlarged by ``padding``
and adds the boundary correction of the screened multipole values
``Y_lm P(l+1/2, w**2 r**2)/r**(l+1)`` (erf) or
``Y_lm Q(l+1/2, w**2 r**2)/r**(l+1)`` (erfc) with ``1/lambda``.  It needs
a large padding at small ``w``.

Reflection-parity classes
-------------------------
Where the box is symmetric about the origin along all three axes (a domain
centred at the origin with points at ``(i + 1/2) h`` or ``i h``), the
reflections ``x -> -x``, ``y -> -y``, ``z -> -z`` map it onto itself and
commute with ``A_B`` and with the Gaussian matrices ``G_d``.  A density of
definite parity ``t = (t_x, t_y, t_z)`` (``f(s r) = chi_t(s) f(r)`` for the
eight sign operations ``s``, ``chi_t(s)`` the product of ``t_d`` over the
flipped axes) has a potential of the same parity, and both are fixed by
their values on the octant ``x, y, z >= 0`` of the box
(:meth:`KroneckerPoisson.parity_plan`).  Along an axis of ``n`` points the
even and odd functions are the ranges of the isometries ``E_+`` and ``E_-``
with columns ``(delta_i + t delta_i')/sqrt(2)`` for a point ``i >= 0`` and
its mirror ``i'`` (and ``delta_0`` for the point on the plane of an odd
``n``, even functions only); ``D_t = E_t^T D E_t`` are the two half-size
blocks of the one-dimensional factor, diagonalized on their own (so no
numerical eigenvector is ever sorted by parity).  The class solve is the
Kronecker solve of ``-(D_x,tx (+) D_y,ty (+) D_z,tz)`` on the octant
(``(n/2)**3`` points, one sixteenth of the flops of the box transforms);
octant values ``a`` and subspace coordinates are related by
``alpha = sqrt(2) a`` (``a`` on the plane of an odd axis, even class), which
the forward and inverse matrices absorb.  The boundary values and the
exterior tails of a class-``t`` density involve only the moments ``q_lm``
of the harmonics of parity ``t`` (``r**l Y_lm`` is a polynomial of definite
parity in each coordinate) and only the octant rows of the box's face
correction, the Cartesian moments only the monomials ``x**a`` with
``(-1)**a = t_x`` (weighted 2 for a point and its mirror).  The result is
the potential of the full-box solve to round-off.  A pair density of two
orbitals of reflection symmetry sectors has the parities of the sector's
character; :mod:`parsec_python.acceleration.Hamiltonian.exchange_device`
forms them from the sector orbitals without the full grid.

Units and conventions follow :func:`parsec_python.Hartree.solve_hartree`:
densities in electrons/bohr**3, potentials in Rydberg,
``-nabla_h**2 V = 8 pi rho`` for the Coulomb kernel, so the solver can
stand in for
:func:`parsec_python.V_xc.exact_exchange.reference_pair_poisson`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import gammainc, gammaincc

from ..Grid import RealSpaceGrid
from ..Laplacian import second_derivative_coefficients
from ..models import HartreeSettings
from .harmonics import _positive_m_harmonic_rows

Kernel = Literal["coulomb", "erfc", "erf"]
Screening = Literal["split", "spectral"]

KERNELS: tuple[str, ...] = ("coulomb", "erfc", "erf")
SCREENINGS: tuple[str, ...] = ("split", "spectral")

# The Gaussian-smeared multipole potentials I~_lm equal I_lm beyond the box to
# this relative accuracy (choice of beta).
SMEARING_TOLERANCE = 1.0e-17
# Default memory for the work arrays of one batch (bytes).
DEFAULT_BATCH_MEMORY = 1 << 30
# Cap of the smearing exponent of the split screening: beta h**2 <= this.
DEFAULT_SMEARING_LIMIT = 0.25


def _automatic_padding(lower, upper, shift, spacing, source_squared, smearing_distance):
    """Planes ``(below, above)`` (three each) to add to the box ``[lower, upper]`` (integer indices).

    Two conditions, each side of each axis on its own:

    * convergence of the exterior expansion: the exterior grid point nearest
      to the origin beyond the face (the face plane moved one spacing out, at
      the coordinates nearest to zero of the two other axes) lies farther
      from the origin than every active point, ``|r_E|**2 > source_squared``;
    * the smearing cap of the split screening: the face plane moved one
      spacing out is at least ``smearing_distance`` from the origin.

    The squared distances are summed in the order of the grid
    (``x**2 + y**2 + z**2``), so that a sphere domain, whose exterior points
    are by construction farther than its radius, gets no plane.

    An axis whose box is symmetric about the origin gets the same number of
    planes on both sides (the larger count), so that the padded box keeps the
    reflection symmetry the parity classes of :meth:`KroneckerPoisson.parity_plan`
    need.  The two counts of such an axis are equal anyway (the conditions
    are mirror images); the rule only makes it explicit.
    """

    lower = np.asarray(lower, dtype=np.int64)
    upper = np.asarray(upper, dtype=np.int64)
    nearest = []
    for d in range(3):
        coordinates = (np.arange(lower[d], upper[d] + 1, dtype=np.float64) + shift[d]) * spacing
        nearest.append(float(coordinates[np.argmin(np.abs(coordinates))]))
    below = np.zeros(3, dtype=np.int64)
    above = np.zeros(3, dtype=np.int64)
    for d in range(3):
        for side, index in ((-1, lower[d]), (1, upper[d])):
            added = 0
            while True:
                plane = float(index + side * (added + 1)) + shift[d]
                point = list(nearest)
                point[d] = plane * spacing
                squared = point[0] * point[0] + point[1] * point[1] + point[2] * point[2]
                if squared > source_squared and abs(point[d]) >= smearing_distance:
                    break
                added += 1
            if side < 0:
                below[d] = added
            else:
                above[d] = added
        if lower[d] + upper[d] + 2.0 * shift[d] == 0.0:
            below[d] = above[d] = max(below[d], above[d])
    return below, above


def dirichlet_second_derivative_matrix(
    points: int, expansion_order: int, spacing: float
) -> np.ndarray:
    """Return the ``points x points`` matrix ``d**2/dx**2`` with zero Dirichlet ends.

    ``D[i, j] = c_|i-j| / h**2`` for ``|i - j| <= M = expansion_order / 2``,
    with the centered coefficients of
    :func:`parsec_python.Laplacian.second_derivative_coefficients`: the
    one-dimensional factor of the run's negative Laplacian
    (``A = -(D_x (+) D_y (+) D_z)`` on a box).  Values beyond the ends are
    zero (they are moved to the right-hand side by the caller).
    """

    points = int(points)
    if points < 1:
        raise ValueError("a one-dimensional grid needs at least one point")
    coefficients = second_derivative_coefficients(expansion_order)
    width = len(coefficients) // 2
    matrix = np.zeros((points, points), dtype=np.float64)
    for shell in range(-width, width + 1):
        if abs(shell) >= points:
            continue
        diagonal = np.full(points - abs(shell), coefficients[width + shell])
        matrix += np.diag(diagonal, k=shell)
    return matrix / float(spacing) ** 2


def spectral_function(
    kernel: str, eigenvalues: np.ndarray, omega: float | None = None
) -> np.ndarray:
    """Return ``f(lambda)`` with ``v = 4 pi f(-nabla**2) rho`` in Hartree units.

    ``coulomb``: ``1/lambda``; ``erfc``: ``(1 - exp(-lambda/(4 w**2)))/lambda``;
    ``erf``: ``exp(-lambda/(4 w**2))/lambda``, from the Fourier transforms
    ``4 pi/k**2``, ``4 pi/k**2 (1 - exp(-k**2/(4 w**2)))`` and
    ``4 pi/k**2 exp(-k**2/(4 w**2))`` of ``1/r``, ``erfc(w r)/r`` and
    ``erf(w r)/r`` with ``k**2 -> lambda``.  The eigenvalues must be
    positive.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("the spectral functions need positive eigenvalues")
    if kernel == "coulomb":
        return 1.0 / eigenvalues
    if omega is None or not omega > 0.0:
        raise ValueError(f"the {kernel} kernel needs a positive screening parameter")
    scaled = eigenvalues / (4.0 * float(omega) ** 2)
    if kernel == "erfc":
        return -np.expm1(-scaled) / eigenvalues
    if kernel == "erf":
        return np.exp(-scaled) / eigenvalues
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")


def harmonic_index(angular_momentum: int, magnetic: int) -> int:
    """Column of the real harmonic ``(l, m)``: ``l**2 + l + m``."""

    return angular_momentum * angular_momentum + angular_momentum + magnetic


def real_harmonics(points: np.ndarray, order: int) -> np.ndarray:
    """Real orthonormal spherical harmonics at points, shape ``(n, (order+1)**2)``.

    ``Y_l0`` and, for ``m > 0``, ``sqrt(2) Re Y_lm`` (column ``(l, m)``)
    and ``sqrt(2) Im Y_lm`` (column ``(l, -m)``) of the complex harmonics
    of :func:`scipy.special.sph_harm_y` (evaluated by the recurrence of
    :mod:`.harmonics`).  The angles at the origin are those of the +z axis.
    """

    points = np.asarray(points, dtype=np.float64)
    columns = np.empty((points.shape[0], (order + 1) ** 2), dtype=np.float64)
    root_two = math.sqrt(2.0)
    for angular_momentum, magnetic, harmonic, _ in _positive_m_harmonic_rows(points, order):
        if magnetic == 0:
            columns[:, harmonic_index(angular_momentum, 0)] = harmonic.real
        else:
            columns[:, harmonic_index(angular_momentum, magnetic)] = root_two * harmonic.real
            columns[:, harmonic_index(angular_momentum, -magnetic)] = root_two * harmonic.imag
    return columns


def solid_harmonic_coefficients(order: int) -> np.ndarray:
    """Monomial coefficients of the real solid harmonics ``r**l Y_lm``.

    Returns ``C`` of shape ``((order+1)**2, order+1, order+1, order+1)`` with
    ``r**l Y_lm(Omega) = sum_abc C[lm, a, b, c] x**a y**b z**c`` for the
    harmonics of :func:`real_harmonics`.  The complex solid harmonics follow
    from the associated-Legendre recurrence of :mod:`.harmonics` multiplied
    by ``r**l``:

    ``R_mm = -(2m - 1) (x + i y) R_(m-1,m-1)``, ``R_(m+1,m) = (2m + 1) z R_mm``,
    ``(l - m) R_lm = (2l - 1) z R_(l-1,m) - (l + m - 1) r**2 R_(l-2,m)``,

    applied to coefficient arrays (multiplication by ``x``, ``y``, ``z`` is
    a shift of an index), so the coefficients are exact up to rounding.
    """

    order = int(order)
    if order < 0:
        raise ValueError("multipole order cannot be negative")
    size = order + 1

    def shifted(poly: np.ndarray, axis: int, steps: int = 1) -> np.ndarray:
        result = np.zeros_like(poly)
        source = [slice(None)] * 3
        target = [slice(None)] * 3
        source[axis] = slice(0, size - steps)
        target[axis] = slice(steps, size)
        result[tuple(target)] = poly[tuple(source)]
        return result

    def times_r2(poly: np.ndarray) -> np.ndarray:
        return shifted(poly, 0, 2) + shifted(poly, 1, 2) + shifted(poly, 2, 2)

    coefficients = np.zeros((size * size, size, size, size), dtype=np.float64)
    diagonal = np.zeros((size, size, size), dtype=np.complex128)
    diagonal[0, 0, 0] = 1.0
    root_two = math.sqrt(2.0)

    def store(angular_momentum: int, magnetic: int, poly: np.ndarray) -> None:
        log_ratio = math.lgamma(angular_momentum - magnetic + 1) - math.lgamma(
            angular_momentum + magnetic + 1
        )
        norm = math.sqrt((2 * angular_momentum + 1) * math.exp(log_ratio) / (4.0 * math.pi))
        if magnetic == 0:
            coefficients[harmonic_index(angular_momentum, 0)] = norm * poly.real
        else:
            coefficients[harmonic_index(angular_momentum, magnetic)] = root_two * norm * poly.real
            coefficients[harmonic_index(angular_momentum, -magnetic)] = root_two * norm * poly.imag

    for magnetic in range(size):
        if magnetic:
            diagonal = -(2 * magnetic - 1) * (shifted(diagonal, 0) + 1j * shifted(diagonal, 1))
        previous = diagonal
        store(magnetic, magnetic, previous)
        if magnetic == order:
            continue
        current = (2 * magnetic + 1) * shifted(diagonal, 2)
        store(magnetic + 1, magnetic, current)
        for angular_momentum in range(magnetic + 2, size):
            following = (
                (2 * angular_momentum - 1) * shifted(current, 2)
                - (angular_momentum + magnetic - 1) * times_r2(previous)
            ) / (angular_momentum - magnetic)
            store(angular_momentum, magnetic, following)
            previous, current = current, following
    return coefficients


def _smearing_exponent_argument(order: int, tolerance: float = SMEARING_TOLERANCE) -> float:
    """Smallest ``x`` with ``Q(order + 1/2, x) <= tolerance`` (upper regularized gamma)."""

    low, high = 0.0, 64.0
    while gammaincc(order + 0.5, high) > tolerance:
        high *= 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if gammaincc(order + 0.5, middle) > tolerance:
            low = middle
        else:
            high = middle
    return high


def _smeared_radial(
    radius: np.ndarray, order: int, scale: float, exponent: float | None, upper: bool = False
) -> np.ndarray:
    """``(scale/r)**(l+1) * P(l+1/2, exponent r**2)`` for ``l = 0..order``, shape ``(order+1, n)``.

    ``exponent=None`` gives the point-multipole factor ``(scale/r)**(l+1)``;
    ``upper=True`` uses ``Q = 1 - P`` instead of ``P``.  At ``r = 0`` the
    smeared value is its limit ``scale**(l+1) exponent**(l+1/2) r**l /
    Gamma(l+3/2)``: nonzero only for ``l = 0``.
    """

    radius = np.asarray(radius, dtype=np.float64).reshape(-1)
    distinct, inverse = np.unique(radius, return_inverse=True)
    if distinct.size < radius.size:
        # Points of a box repeat their distances (reflections, permutations):
        # the special functions are evaluated once per distinct value.
        return _smeared_radial(distinct, order, scale, exponent, upper)[:, inverse.reshape(-1)]
    positive = radius > 0.0
    at_origin = not bool(np.all(positive))
    if at_origin and (exponent is None or upper):
        raise ValueError("a point-multipole value is undefined at the origin")
    safe = np.where(positive, radius, 1.0)
    result = np.empty((order + 1, radius.size), dtype=np.float64)
    for angular_momentum in range(order + 1):
        value = (scale / safe) ** (angular_momentum + 1)
        if exponent is not None:
            argument = exponent * safe * safe
            incomplete = gammaincc if upper else gammainc
            value = value * incomplete(angular_momentum + 0.5, argument)
        if at_origin:
            limit = scale * 2.0 * math.sqrt(exponent / math.pi) if angular_momentum == 0 else 0.0
            value = np.where(positive, value, limit)
        result[angular_momentum] = value
    return result


@dataclass(frozen=True)
class BoxFace:
    """The boundary correction of one face of the box.

    ``rows`` are the flat box indices of the ``M`` planes inside the face and
    ``matrix`` (shape ``(K, len(rows))``) maps the scaled moments to their
    right-hand-side correction ``-A_BE V_E``.
    """

    axis: int
    side: int
    rows: np.ndarray
    matrix: np.ndarray


def _merged_faces(faces) -> tuple[np.ndarray, np.ndarray]:
    """One ``(rows, matrix)`` pair for all faces: rows near an edge or corner sum their columns."""

    rows = np.concatenate([face.rows for face in faces])
    columns = np.concatenate([face.matrix for face in faces], axis=1)
    distinct, inverse = np.unique(rows, return_inverse=True)
    merged = np.zeros((columns.shape[0], distinct.size), dtype=np.float64)
    np.add.at(merged.T, inverse.reshape(-1), columns.T)
    return distinct, merged


def _separable(data, mx, my, mz):
    """``(M_x (x) M_y (x) M_z) data`` for ``data`` of shape ``(nb, px, py, pz)``.

    ``M_d`` has shape ``(r_d, p_d)``; the result has shape
    ``(nb, r_x, r_y, r_z)``.  Three (batched) matrix products, z first.
    Only ``@`` and ``reshape`` are used, so NumPy and CuPy arrays both work.
    """

    nb, px, py, pz = data.shape
    rz, ry, rx = mz.shape[0], my.shape[0], mx.shape[0]
    work = data.reshape(nb * px * py, pz) @ mz.T
    work = my @ work.reshape(nb * px, py, rz)
    work = mx @ work.reshape(nb, px, ry * rz)
    return work.reshape(nb, rx, ry, rz)


@dataclass
class SolvePlan:
    """The arrays of a prepared solve (NumPy, or CuPy for the device solver).

    ``eigenvectors`` are ``Q_x, Q_y, Q_z``; ``eigenvectors_t`` their
    transposes (contiguous copies for the forward transform).
    ``inverse_eigenvalues`` is ``1/lambda`` on the box; ``density_factor``
    (``8 pi f(lambda)``) and ``boundary_factor`` (``1/lambda``) are set for
    the spectral screening only.  ``monomials_t`` (``(L+1, n_d)``) and
    ``moment_map`` (``((L+1)**3, K)``) give the scaled moments.  ``faces``
    holds ``(rows, matrix)`` pairs with distinct rows within a pair (the
    reference merges the six faces into one).  ``gaussian_contract`` (``diag(s) U^T Q_d``),
    ``gaussian_expand`` (``U_d``) and ``exterior_tail`` (``(K, n_active)``)
    are the split screening, whose result is
    ``coulomb_weight * v_coulomb + long_range_weight * v_erf``.
    """

    kernel: str
    screening: str
    box_shape: tuple[int, int, int]
    box_index: object
    eigenvectors: tuple
    eigenvectors_t: tuple
    inverse_eigenvalues: object
    density_factor: object
    boundary_factor: object
    monomials_t: tuple
    moment_map: object
    faces: tuple
    gaussian_contract: tuple | None
    gaussian_expand: tuple | None
    exterior_tail: object
    coulomb_weight: float = 1.0
    long_range_weight: float = 0.0

    def mapped(self, convert) -> "SolvePlan":
        """The same plan with every array passed through ``convert``."""

        def each(value):
            if value is None:
                return None
            if isinstance(value, tuple):
                return tuple(each(item) for item in value)
            return convert(value)

        return SolvePlan(
            kernel=self.kernel,
            screening=self.screening,
            box_shape=self.box_shape,
            box_index=convert(self.box_index),
            eigenvectors=each(self.eigenvectors),
            eigenvectors_t=each(self.eigenvectors_t),
            inverse_eigenvalues=convert(self.inverse_eigenvalues),
            density_factor=each(self.density_factor),
            boundary_factor=each(self.boundary_factor),
            monomials_t=each(self.monomials_t),
            moment_map=convert(self.moment_map),
            faces=tuple((convert(rows), convert(matrix)) for rows, matrix in self.faces),
            gaussian_contract=each(self.gaussian_contract),
            gaussian_expand=each(self.gaussian_expand),
            exterior_tail=each(self.exterior_tail),
            coulomb_weight=self.coulomb_weight,
            long_range_weight=self.long_range_weight,
        )

    def nbytes(self) -> int:
        """Bytes held by the arrays of the plan."""

        total = 0

        def add(value):
            nonlocal total
            if value is None:
                return
            if isinstance(value, tuple):
                for item in value:
                    add(item)
                return
            total += int(value.nbytes)

        for name in (
            "box_index", "eigenvectors", "eigenvectors_t", "inverse_eigenvalues",
            "density_factor", "boundary_factor", "monomials_t", "moment_map",
            "gaussian_contract", "gaussian_expand", "exterior_tail",
        ):
            add(getattr(self, name))
        for rows, matrix in self.faces:
            add(rows)
            add(matrix)
        return total


def solve_block(plan: SolvePlan, densities, xp=np):
    """Potentials (Ry) of densities ``(n_active, nb)`` as an array ``(nb, n_active)``.

    ``xp`` is the array module of the plan (``numpy`` or ``cupy``).  The steps:
    scatter to the box, scaled moments, boundary correction of each face,
    forward transform, spectral factor, inverse transform (Coulomb, erfc)
    and the Gaussian smoothing with the exterior tails (erf, erfc) of the
    split screening, gather.
    """

    nb = int(densities.shape[1])
    nx, ny, nz = plan.box_shape
    box_size = nx * ny * nz
    box = xp.zeros((nb, box_size), dtype=xp.float64)
    box[:, plan.box_index] = densities.T
    # The solve holds the only reference to the scattered box, so that it is
    # freed after the forward transform (the peak of the work arrays).
    holder = [box.reshape(nb, nx, ny, nz)]
    del box
    return _solve_held_box(plan, holder, xp, prescaled=False)


def solve_box(plan: SolvePlan, box, xp=np, *, prescaled: bool = False):
    """Potentials (Ry) of box densities ``(nb, nx, ny, nz)`` as ``(nb, n_active)``: :func:`solve_block` after its scatter.

    With ``prescaled`` the box holds ``8 pi`` times the densities and becomes
    the right-hand side in place (the moments are scaled back); a caller
    that forms its densities on the box itself (the parity classes of
    :mod:`parsec_python.acceleration.Hamiltonian.exchange_device`) saves the
    scatter and a copy so.  The spectral screening takes unscaled densities.
    """

    return _solve_held_box(plan, [box], xp, prescaled=prescaled)


def _solve_held_box(plan: SolvePlan, holder: list, xp, *, prescaled: bool):
    """:func:`solve_box` of the box in ``holder`` (taken out: the solve drops it after the forward transform)."""

    box = holder.pop()
    nb = int(box.shape[0])
    nx, ny, nz = plan.box_shape
    box_size = nx * ny * nz
    cartesian = _separable(box, *plan.monomials_t)
    moments = cartesian.reshape(nb, -1) @ plan.moment_map
    if prescaled:
        if plan.density_factor is not None:
            raise ValueError("the spectral screening solves unscaled densities")
        moments *= 1.0 / (8.0 * math.pi)

    qx, qy, qz = plan.eigenvectors
    qxt, qyt, qzt = plan.eigenvectors_t
    if plan.density_factor is None:
        right_hand_side = (
            box.reshape(nb, box_size) if prescaled else (8.0 * math.pi) * box.reshape(nb, box_size)
        )
        for rows, matrix in plan.faces:
            right_hand_side[:, rows] += moments @ matrix
        spectral = _separable(right_hand_side.reshape(nb, nx, ny, nz), qxt, qyt, qzt)
        spectral *= plan.inverse_eigenvalues
    else:
        correction = xp.zeros((nb, box_size), dtype=xp.float64)
        for rows, matrix in plan.faces:
            correction[:, rows] += moments @ matrix
        spectral = _separable(box, qxt, qyt, qzt)
        spectral *= plan.density_factor
        boundary = _separable(correction.reshape(nb, nx, ny, nz), qxt, qyt, qzt)
        boundary *= plan.boundary_factor
        spectral += boundary
        del boundary, correction
    del box

    if plan.kernel == "coulomb" or plan.screening == "spectral":
        return _separable(spectral, qx, qy, qz).reshape(nb, box_size)[:, plan.box_index]

    # Split screening: v_erf = G_B v_B + sum_lm q_lm T_lm.
    reduced = _separable(spectral, *plan.gaussian_contract)
    smooth = _separable(reduced, *plan.gaussian_expand).reshape(nb, box_size)
    long_range = smooth[:, plan.box_index]
    del smooth
    long_range += moments @ plan.exterior_tail
    if plan.coulomb_weight == 0.0:
        if plan.long_range_weight != 1.0:
            long_range *= plan.long_range_weight
        return long_range
    coulomb = _separable(spectral, qx, qy, qz).reshape(nb, box_size)[:, plan.box_index]
    if plan.coulomb_weight != 1.0:
        coulomb *= plan.coulomb_weight
    if plan.long_range_weight == -1.0:
        coulomb -= long_range
    elif plan.long_range_weight != 0.0:
        long_range *= plan.long_range_weight
        coulomb += long_range
    return coulomb


# ---------------------------------------------------------------------------
# Reflection-parity classes (module docstring).
# ---------------------------------------------------------------------------

#: The eight sign operations: index ``4 b_x + 2 b_y + b_z``, axis ``d`` flipped where ``b_d = 1``.
SIGN_OPERATIONS: tuple[tuple[int, int, int], ...] = tuple(
    tuple(-1 if (index >> (2 - d)) & 1 else 1 for d in range(3)) for index in range(8)
)
#: The eight parity classes, in the same order: ``t_d = -1`` where bit ``d`` is set.
PARITY_CLASSES: tuple[tuple[int, int, int], ...] = SIGN_OPERATIONS


def parity_index(parity) -> int:
    """Index of a parity class ``(t_x, t_y, t_z)`` in :data:`PARITY_CLASSES`."""

    values = tuple(int(value) for value in parity)
    if len(values) != 3 or any(value not in (-1, 1) for value in values):
        raise ValueError("a parity class is three signs (t_x, t_y, t_z)")
    return PARITY_CLASSES.index(values)


def parity_character(parity: int, operation: int) -> int:
    """``chi_t(s)``: the sign of a class-``t`` function under operation ``s`` (both indices)."""

    return -1 if bin(int(parity) & int(operation)).count("1") % 2 else 1


def parity_character_table() -> np.ndarray:
    """``(8, 8)`` table ``chi[t, s]`` (a Sylvester--Hadamard matrix, ``chi chi^T = 8``)."""

    return np.array(
        [[parity_character(t, s) for s in range(8)] for t in range(8)], dtype=np.float64
    )


@dataclass(frozen=True)
class ParityLayout:
    """The octant of a box symmetric about the origin and its images on the active grid.

    Along axis ``d`` the box has ``n_d`` points; the octant keeps the
    ``half_shape[d] = n_d - centre[d]`` points from box index
    ``centre[d] = n_d // 2`` on (coordinates ``>= 0``; for odd ``n_d`` the
    first one lies on the mirror plane, ``on_plane[d]``).  ``box_index``
    are the flat octant indices (increasing) of the points ``R`` whose
    images include an active point; ``images[s, k]`` is the active-grid row
    of the image of point ``k`` under sign operation ``s``
    (:data:`SIGN_OPERATIONS`), ``-1`` where that image is not active.
    """

    box_shape: tuple[int, int, int]
    centre: tuple[int, int, int]
    half_shape: tuple[int, int, int]
    on_plane: tuple[bool, bool, bool]
    box_index: np.ndarray
    images: np.ndarray
    octant_coordinates: np.ndarray

    @property
    def size(self) -> int:
        """Points ``R`` of the octant."""

        return int(self.box_index.size)

    @property
    def half_size(self) -> int:
        return int(np.prod(self.half_shape))

    def plane_mask(self, parity: int) -> np.ndarray:
        """Points of ``R`` on a mirror plane of an axis that class ``parity`` makes odd (they vanish)."""

        mask = np.zeros(self.size, dtype=bool)
        for d in range(3):
            if self.on_plane[d] and PARITY_CLASSES[int(parity)][d] < 0:
                mask |= self.octant_coordinates[:, d] == 0
        return mask

    def components(self, values: np.ndarray) -> np.ndarray:
        """Parity components ``f_t`` on ``R`` of functions on the active grid: ``(8, size, ...)``.

        ``f_t(r) = (1/8) sum_s chi_t(s) f(s r)``, with ``f = 0`` at images
        that are not active.
        """

        values = np.asarray(values, dtype=np.float64)
        images = np.zeros((8, self.size) + values.shape[1:], dtype=np.float64)
        for operation in range(8):
            rows = self.images[operation]
            present = rows >= 0
            images[operation][present] = values[rows[present]]
        table = parity_character_table()
        return np.tensordot(table, images, axes=(1, 0)) / 8.0

    def values(self, components: np.ndarray, active_points: int) -> np.ndarray:
        """Functions on the ``active_points`` of the grid from their parity components ``(8, size, ...)``.

        ``f(s r) = sum_t chi_t(s) f_t(r)`` at every active image.
        """

        components = np.asarray(components, dtype=np.float64)
        result = np.zeros((int(active_points),) + components.shape[2:], dtype=np.float64)
        table = parity_character_table()
        for operation in range(8):
            rows = self.images[operation]
            present = rows >= 0
            image = np.tensordot(table[:, operation], components, axes=(0, 0))
            result[rows[present]] = image[present]
        return result


def _parity_layout(integer_coordinates, box_lower, box_shape) -> ParityLayout:
    """The :class:`ParityLayout` of active points (integer coordinates) in a symmetric box."""

    local = np.asarray(integer_coordinates, dtype=np.int64) - np.asarray(box_lower, dtype=np.int64)
    shape = np.asarray(box_shape, dtype=np.int64)
    centre = shape // 2
    half = shape - centre
    mirrored = local < centre
    octant = np.where(mirrored, (shape - 1 - local) - centre, local - centre)
    flat = (octant[:, 0] * half[1] + octant[:, 1]) * half[2] + octant[:, 2]
    # The occupied octant points in increasing order and each active point's
    # position among them, by a lookup over the octant box (no sort).
    occupied = np.zeros(int(np.prod(half)), dtype=bool)
    occupied[flat] = True
    box_index = np.flatnonzero(occupied)
    lookup = np.full(occupied.size, -1, dtype=np.int64)
    lookup[box_index] = np.arange(box_index.size, dtype=np.int64)
    position = lookup[flat]
    operation = (mirrored[:, 0].astype(np.int64) << 2) | (mirrored[:, 1].astype(np.int64) << 1) | mirrored[:, 2]
    # A point on the plane of an odd axis is its own mirror image.
    plane_bits = np.zeros(local.shape[0], dtype=np.int64)
    for d in range(3):
        if shape[d] % 2 == 1:
            plane_bits |= (octant[:, d] == 0).astype(np.int64) << (2 - d)
    images = np.full((8, box_index.size), -1, dtype=np.int64)
    rows = np.arange(local.shape[0], dtype=np.int64)
    for subset in range(8):
        selected = (subset & ~plane_bits) == 0
        images[operation[selected] ^ subset, position[selected]] = rows[selected]
    coordinates = np.empty((box_index.size, 3), dtype=np.int64)
    coordinates[:, 0] = box_index // (half[1] * half[2])
    coordinates[:, 1] = (box_index // half[2]) % half[1]
    coordinates[:, 2] = box_index % half[2]
    return ParityLayout(
        box_shape=tuple(int(v) for v in shape),
        centre=tuple(int(v) for v in centre),
        half_shape=tuple(int(v) for v in half),
        on_plane=tuple(bool(v % 2) for v in shape),
        box_index=box_index,
        images=images,
        octant_coordinates=coordinates,
    )


def _reflection_bases(points: int):
    """Even and odd isometries of an axis of ``points`` points symmetric about its middle.

    Returns ``{t: (E, octant, scale)}`` for ``t = +1, -1``: ``E`` (``points
    x dim``) has orthonormal columns ``(delta_i + t delta_i')/sqrt(2)`` for
    the box index ``i = centre + k`` and its mirror ``i' = points - 1 - i``
    (``delta_i`` alone where ``i = i'``, even class only); ``octant[j] = k``
    is the octant index of column ``j`` and ``scale[j]`` (``sqrt(2)`` or 1)
    turns octant values of a class function into its coordinates,
    ``alpha_j = scale_j a_(octant_j)``.
    """

    centre = points // 2
    half = points - centre
    bases = {}
    for parity in (1, -1):
        columns, octant, scale = [], [], []
        for k in range(half):
            index = centre + k
            mirror = points - 1 - index
            column = np.zeros(points)
            if mirror == index:
                if parity < 0:
                    continue
                column[index] = 1.0
                scale.append(1.0)
            else:
                column[index] = math.sqrt(0.5)
                column[mirror] = parity * math.sqrt(0.5)
                scale.append(math.sqrt(2.0))
            columns.append(column)
            octant.append(k)
        bases[parity] = (
            np.column_stack(columns) if columns else np.zeros((points, 0)),
            np.asarray(octant, dtype=np.int64),
            np.asarray(scale, dtype=np.float64),
        )
    return bases


def _octant_matrices(vectors: np.ndarray, octant: np.ndarray, scale: np.ndarray, half: int):
    """Forward ``(dim x half)`` and inverse ``(half x dim)`` matrices of a block's eigenvectors.

    Forward: octant values -> coefficients, ``c = V^T (scale o a[octant])``;
    inverse: ``a[octant] = (V c) / scale`` (zero rows elsewhere: the plane
    point of an odd class).
    """

    dim = vectors.shape[1]
    forward = np.zeros((dim, half))
    forward[:, octant] = (vectors * scale[:, None]).T
    inverse = np.zeros((half, dim))
    inverse[octant, :] = vectors / scale[:, None]
    return np.ascontiguousarray(forward), np.ascontiguousarray(inverse)


class KroneckerPoisson:
    """Direct finite-difference Poisson solver of many densities on the bounding box.

    Parameters
    ----------
    grid
        The active cluster grid of the run.
    finite_difference_order
        Order of the centered stencil, the run's ``Expansion_Order`` by
        default (that is the operator of
        :func:`parsec_python.Laplacian.build_negative_laplacian`).
    kernel
        ``"coulomb"`` (``1/r``), ``"erfc"`` (``erfc(w r)/r``) or ``"erf"``
        (``erf(w r)/r``).
    omega
        Screening parameter ``w`` in 1/bohr for the screened kernels.
    multipole_order
        Largest angular momentum of the boundary expansion
        (``Solver_Lpole``).
    padding
        Extra vacuum around the bounding box in bohr (rounded up to whole
        points on every side), before the automatic padding.
    screening
        ``"split"`` (default) or ``"spectral"`` for the screened kernels,
        see the module docstring.
    rank_tolerance
        Relative eigenvalue below which the Gaussian kernel matrices of the
        split treatment are truncated.
    batch_memory
        Bytes for the work arrays of one batch; :meth:`solve_batch` splits
        larger requests.
    weights
        ``(a, b)``: the potential of the kernel ``a/r + b K(r)`` with ``K``
        the screened kernel (``erfc`` or ``erf``), in one solve (split
        screening).  ``None`` is ``K`` alone.  The exchange kernel of a
        range-separated hybrid, ``alpha/r + beta erfc(w r)/r`` in the
        convention of ``V_xc.hybrid``, is ``weights=(alpha, beta)`` with
        ``kernel="erfc"``.
    automatic_padding
        Pad the box until the multipole expansion converges at every
        exterior point it is used at, and until the smearing cap holds
        (module docstring, "Convergence of the boundary").  ``False`` keeps
        the bounding box (plus ``padding``): correct only for densities
        inside the sphere through the nearest exterior point.  The planes
        added below and above each axis are :attr:`automatic_padding`.
    smearing_limit
        Cap of ``beta h**2`` for the smeared multipoles of the split
        screening.

    :meth:`solve_batch` takes densities of shape ``(n_active, nb)`` and
    returns the potentials in Rydberg, same shape; calling the solver with
    one density returns its potential (the contract of
    :func:`parsec_python.V_xc.exact_exchange.exact_exchange_energy`).
    """

    def __init__(
        self,
        grid: RealSpaceGrid,
        finite_difference_order: int | None = None,
        kernel: Kernel = "coulomb",
        omega: float | None = None,
        multipole_order: int = HartreeSettings().multipole_order,
        *,
        padding: float = 0.0,
        screening: Screening = "split",
        rank_tolerance: float = 1.0e-14,
        batch_memory: int = DEFAULT_BATCH_MEMORY,
        weights: tuple[float, float] | None = None,
        automatic_padding: bool = True,
        smearing_limit: float = DEFAULT_SMEARING_LIMIT,
    ) -> None:
        if not isinstance(grid, RealSpaceGrid):
            raise TypeError(
                "the Kronecker Poisson solver works on isolated cluster grids "
                "(RealSpaceGrid); periodic cells are not supported"
            )
        if kernel not in KERNELS:
            raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")
        if kernel == "coulomb":
            if omega is not None:
                raise ValueError("the Coulomb kernel takes no screening parameter")
        elif omega is None or not (np.isfinite(omega) and omega > 0.0):
            raise ValueError(f"the {kernel} kernel needs a positive screening parameter omega")
        if screening not in SCREENINGS:
            raise ValueError(f"unknown screening treatment {screening!r}; expected {SCREENINGS}")
        if weights is not None:
            if kernel == "coulomb":
                raise ValueError("kernel weights combine 1/r with a screened kernel (erfc or erf)")
            if screening != "split":
                raise ValueError("kernel weights need the split screening")
            weights = tuple(float(value) for value in weights)
            if len(weights) != 2 or not all(np.isfinite(weights)):
                raise ValueError("kernel weights are two finite numbers (a, b) of a/r + b K(r)")
        order = int(
            grid.settings.expansion_order
            if finite_difference_order is None
            else finite_difference_order
        )
        multipole_order = int(multipole_order)
        if multipole_order < 0:
            raise ValueError("multipole order cannot be negative")
        if not (np.isfinite(padding) and padding >= 0.0):
            raise ValueError("padding must be a nonnegative length")
        if not (np.isfinite(smearing_limit) and smearing_limit > 0.0):
            raise ValueError("smearing_limit must be positive")

        self.grid = grid
        self.kernel = kernel
        self.omega = None if omega is None else float(omega)
        self.finite_difference_order = order
        self.multipole_order = multipole_order
        self.screening = screening if kernel != "coulomb" else "split"
        self.rank_tolerance = float(rank_tolerance)
        self.batch_memory = int(batch_memory)
        self.weights = weights
        # v = coulomb_weight * v_coulomb + long_range_weight * v_erf.
        full, screened = (0.0, 1.0) if weights is None else weights
        if kernel == "coulomb":
            coulomb_weight, long_range_weight = 1.0, 0.0
        elif kernel == "erf":
            coulomb_weight, long_range_weight = full, screened
        else:  # erfc = 1/r - erf
            coulomb_weight, long_range_weight = full + screened, -screened

        spacing = float(grid.spacing)
        self.spacing = spacing
        self.volume_element = spacing**3
        coefficients = second_derivative_coefficients(order)
        width = len(coefficients) // 2
        self.stencil_half_width = width

        # The box: bounding box of the active integer points plus padding.
        pad = int(math.ceil(padding / spacing - 1.0e-12)) if padding > 0.0 else 0
        self.padding_points = pad
        integers = np.asarray(grid.integer_coordinates, dtype=np.int64)
        shift = np.asarray(grid.settings.shift, dtype=np.float64)
        lower = integers.min(axis=0) - pad
        upper = integers.max(axis=0) + pad
        # Automatic padding (module docstring, "Convergence of the boundary").
        source_squared = float(np.max(np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)))
        self.source_radius = math.sqrt(source_squared)
        self.smearing_limit = float(smearing_limit)
        self.smearing_argument = None
        smearing_distance = 0.0
        if kernel != "coulomb" and self.screening == "split":
            self.smearing_argument = _smearing_exponent_argument(multipole_order)
            # beta = x_L / d**2 <= smearing_limit / h**2.
            smearing_distance = spacing * math.sqrt(self.smearing_argument / self.smearing_limit)
        self.smearing_distance = smearing_distance
        added_lower = np.zeros(3, dtype=np.int64)
        added_upper = np.zeros(3, dtype=np.int64)
        if automatic_padding:
            added_lower, added_upper = _automatic_padding(
                lower, upper, shift, spacing, source_squared, smearing_distance
            )
        lower = lower - added_lower
        upper = upper + added_upper
        self.automatic_padding = (tuple(int(v) for v in added_lower), tuple(int(v) for v in added_upper))
        shape = tuple(int(v) for v in upper - lower + 1)
        self.box_lower = lower
        self.box_shape = shape
        self.axes = tuple(
            (np.arange(lower[d], upper[d] + 1, dtype=np.float64) + shift[d]) * spacing
            for d in range(3)
        )
        for d in range(3):
            if not (self.axes[d][0] < 0.0 < self.axes[d][-1]):
                raise ValueError("the multipole origin must lie inside the box of the domain")
        local = integers - lower
        self.box_index = (local[:, 0] * shape[1] + local[:, 1]) * shape[2] + local[:, 2]
        self.box_size = int(np.prod(shape))
        self.size = int(integers.shape[0])

        # 1D eigenvectors and eigenvalues of D_d.
        self.eigenvectors = []
        eigenvalues = []
        for d in range(3):
            values, vectors = np.linalg.eigh(
                dirichlet_second_derivative_matrix(shape[d], order, spacing)
            )
            self.eigenvectors.append(np.ascontiguousarray(vectors))
            eigenvalues.append(values)
        self.eigenvectors = tuple(self.eigenvectors)
        laplacian_eigenvalues = -(
            eigenvalues[0][:, None, None] + eigenvalues[1][None, :, None] + eigenvalues[2][None, None, :]
        )
        if np.any(laplacian_eigenvalues <= 0.0):
            raise RuntimeError("the box Laplacian is not positive definite")
        self.laplacian_eigenvalues = laplacian_eigenvalues

        # Scale of the coordinates of the multipole arithmetic and the
        # nearest point beyond the box.
        self.length_scale = float(max(max(abs(a[0]), abs(a[-1])) for a in self.axes))
        self.exterior_distance = float(
            min(min(abs(a[0]), abs(a[-1])) for a in self.axes) + spacing
        )
        self._monomials = tuple(
            np.vander(a / self.length_scale, multipole_order + 1, increasing=True)
            for a in self.axes
        )
        size = multipole_order + 1
        self._moment_map = (
            solid_harmonic_coefficients(multipole_order).reshape(size * size, size**3).T
            * self.volume_element
        )
        self._prefactors = np.array(
            [
                2.0 * 4.0 * math.pi / (2 * l + 1) / self.length_scale
                for l in range(size)
                for _ in range(2 * l + 1)
            ]
        )

        # Spectral factors and boundary corrections.
        density_factor = boundary_factor = None
        if kernel == "coulomb" or self.screening == "split":
            self.faces = self._build_faces(coefficients, None)
        else:
            density_factor = 8.0 * math.pi * spectral_function(
                kernel, laplacian_eigenvalues, self.omega
            )
            boundary_factor = 1.0 / laplacian_eigenvalues
            self.faces = self._build_faces(coefficients, kernel)

        self.smearing_exponent = None
        self.gaussian_ranks = None
        split = None
        if kernel != "coulomb" and self.screening == "split":
            split = self._build_split()
        self.plan = SolvePlan(
            kernel=kernel,
            screening=self.screening,
            box_shape=shape,
            box_index=self.box_index,
            eigenvectors=self.eigenvectors,
            eigenvectors_t=tuple(np.ascontiguousarray(q.T) for q in self.eigenvectors),
            inverse_eigenvalues=1.0 / laplacian_eigenvalues,
            density_factor=density_factor,
            boundary_factor=boundary_factor,
            monomials_t=tuple(np.ascontiguousarray(m.T) for m in self._monomials),
            moment_map=self._moment_map,
            faces=(_merged_faces(self.faces),),
            gaussian_contract=None if split is None else split[0],
            gaussian_expand=None if split is None else split[1],
            exterior_tail=None if split is None else split[2],
            coulomb_weight=coulomb_weight,
            long_range_weight=long_range_weight,
        )

    # ------------------------------------------------------------------ setup

    def _exterior_basis(self, points: np.ndarray, kernel: str | None) -> np.ndarray:
        """Exterior potential per scaled moment at points (Ry), shape ``(n, K)``.

        ``kernel=None`` is the Coulomb expansion; ``"erf"``/``"erfc"`` multiply
        the term of order ``l`` by ``P``/``Q(l+1/2, w**2 r**2)``.
        """

        order = self.multipole_order
        radius = np.linalg.norm(points, axis=1)
        if kernel is None:
            radial = _smeared_radial(radius, order, self.length_scale, None)
        else:
            radial = _smeared_radial(
                radius, order, self.length_scale, self.omega**2, upper=(kernel == "erfc")
            )
        basis = real_harmonics(points, order)
        degree = np.repeat(np.arange(order + 1), 2 * np.arange(order + 1) + 1)
        basis *= radial[degree].T
        basis *= self._prefactors
        return basis

    def _build_faces(self, coefficients: np.ndarray, kernel: str | None) -> tuple[BoxFace, ...]:
        width = self.stencil_half_width
        inverse_h2 = 1.0 / self.spacing**2
        shape = self.box_shape
        flat = np.arange(self.box_size).reshape(shape)
        faces = []
        for axis in range(3):
            others = [d for d in range(3) if d != axis]
            first, second = (self.axes[d] for d in others)
            plane = np.empty((first.size * second.size, 3))
            plane[:, others[0]] = np.repeat(first, second.size)
            plane[:, others[1]] = np.tile(second, first.size)
            axis_values = self.axes[axis]
            for side in (-1, 1):
                inner = min(width, shape[axis])
                exterior = []
                for step in range(1, width + 1):
                    points = plane.copy()
                    if side > 0:
                        points[:, axis] = axis_values[-1] + step * self.spacing
                    else:
                        points[:, axis] = axis_values[0] - step * self.spacing
                    exterior.append(self._exterior_basis(points, kernel))
                blocks = []
                rows = []
                for depth in range(inner):
                    block = np.zeros_like(exterior[0])
                    for step in range(1, width - depth + 1):
                        # The coupling of the point ``depth`` planes inside the
                        # face to the exterior plane ``step`` is -c/h**2 in A;
                        # rhs -= A_BE V_E adds c V_E / h**2.
                        block += coefficients[width + depth + step] * inverse_h2 * exterior[step - 1]
                    index = shape[axis] - 1 - depth if side > 0 else depth
                    selector = [slice(None)] * 3
                    selector[axis] = index
                    rows.append(flat[tuple(selector)].reshape(-1))
                    blocks.append(block)
                faces.append(
                    BoxFace(
                        axis=axis,
                        side=side,
                        rows=np.concatenate(rows),
                        matrix=np.ascontiguousarray(np.concatenate(blocks, axis=0).T),
                    )
                )
        return tuple(faces)

    def _build_split(self) -> tuple[tuple, tuple, np.ndarray]:
        """``(contract, expand, tails)`` of the split treatment.

        ``contract[d] = diag(s_d) U_d^T Q_d`` and ``expand[d] = U_d`` from the
        truncated eigendecomposition ``G_d ~ U_d diag(s_d) U_d^T``; ``tails``
        is ``T_lm`` per scaled moment on the active points, shape
        ``(K, n_active)``, in Rydberg.
        """

        omega = self.omega
        order = self.multipole_order
        argument = self.smearing_argument
        beta = argument / self.exterior_distance**2
        gamma = beta * omega**2 / (beta + omega**2)
        self.smearing_exponent = beta

        contract, expand, ranks, box_contract = [], [], [], []
        weight = self.spacing * omega / math.sqrt(math.pi)
        for d in range(3):
            axis = self.axes[d]
            kernel_matrix = weight * np.exp(-(omega * (axis[:, None] - axis[None, :])) ** 2)
            values, vectors = np.linalg.eigh(kernel_matrix)
            keep = values > self.rank_tolerance * values.max()
            values, vectors = values[keep], vectors[:, keep]
            ranks.append(int(values.size))
            expand.append(np.ascontiguousarray(vectors))
            # On box values diag(s) U^T; on spectral coefficients diag(s) U^T Q_d.
            box_contract.append(np.ascontiguousarray(values[:, None] * vectors.T))
            contract.append(np.ascontiguousarray(box_contract[-1] @ self.eigenvectors[d]))
        self.gaussian_ranks = tuple(ranks)

        # T_lm = g * I~_lm - G_B I~_lm on the active points (Ry per scaled moment).
        nx, ny, nz = self.box_shape
        points = np.empty((self.box_size, 3))
        points[:, 0] = np.repeat(self.axes[0], ny * nz)
        points[:, 1] = np.tile(np.repeat(self.axes[1], nz), nx)
        points[:, 2] = np.tile(self.axes[2], nx * ny)
        radius = np.linalg.norm(points, axis=1)
        smeared = _smeared_radial(radius, order, self.length_scale, beta)
        smoothed = _smeared_radial(radius[self.box_index], order, self.length_scale, gamma)
        count = (order + 1) ** 2
        tails = np.empty((self.size, count), dtype=np.float64)
        chunk = max(1, int(self.batch_memory // (3 * 8 * self.box_size)))
        pending: list[tuple[int, np.ndarray]] = []

        def flush() -> None:
            if not pending:
                return
            columns = [column for column, _ in pending]
            stack = np.stack([values for _, values in pending]).reshape(len(pending), nx, ny, nz)
            reduced = _separable(stack, *box_contract)
            convolved = _separable(reduced, *expand).reshape(len(pending), self.box_size)
            tails[:, columns] -= convolved[:, self.box_index].T
            pending.clear()

        root_two = math.sqrt(2.0)
        for l, m, harmonic, _ in _positive_m_harmonic_rows(points, order):
            parts = [(m, harmonic.real if m == 0 else root_two * harmonic.real)]
            if m:
                parts.append((-m, root_two * harmonic.imag))
            for magnetic, angular in parts:
                column = harmonic_index(l, magnetic)
                factor = self._prefactors[column]
                tails[:, column] = factor * angular[self.box_index] * smoothed[l]
                pending.append((column, factor * angular * smeared[l]))
                if len(pending) >= chunk:
                    flush()
        flush()
        return tuple(contract), tuple(expand), np.ascontiguousarray(tails.T)

    # ------------------------------------------------------------------ solve

    @property
    def description(self) -> str:
        nx, ny, nz = self.box_shape
        text = (
            f"Kronecker-product FD Poisson solve (order {self.finite_difference_order}) on the "
            f"{nx}x{ny}x{nz} box, multipole boundary of order {self.multipole_order}"
        )
        below, above = self.automatic_padding
        if any(below) or any(above):
            text += (
                " (box padded automatically by "
                + "/".join(f"{low}+{high}" for low, high in zip(below, above))
                + " planes in x/y/z for the convergence of the boundary)"
            )
        if self.kernel == "coulomb":
            return text + ", 1/r kernel"
        kernel = f"{self.kernel}(w r)/r kernel, w = {self.omega:g}/bohr"
        if self.weights is not None:
            kernel = f"{self.weights[0]:g}/r + {self.weights[1]:g} {kernel}"
        if self.screening == "split":
            return text + f", {kernel} as Coulomb minus/plus its Gaussian smoothing (ranks {self.gaussian_ranks})"
        return text + f", {kernel} as the spectral function of the box Laplacian"

    @property
    def batch_size(self) -> int:
        """Densities per batch within :attr:`batch_memory` (at least one)."""

        return max(1, int(self.batch_memory // (4 * 8 * self.box_size)))

    def scaled_moments(self, box_densities: np.ndarray) -> np.ndarray:
        """Moments ``q_lm / s**l`` of box densities ``(nb, nx, ny, nz)``, shape ``(nb, K)``.

        ``s`` is :attr:`length_scale`.  The Cartesian moments are three
        matrix products with the monomials of the scaled coordinates.
        """

        nb = box_densities.shape[0]
        cartesian = _separable(box_densities, *self.plan.monomials_t)
        return cartesian.reshape(nb, -1) @ self._moment_map

    def solve_batch(self, densities: np.ndarray) -> np.ndarray:
        """Potentials (Ry) of densities of shape ``(n_active, nb)``, same shape.

        A one-dimensional density gives a one-dimensional potential.  Large
        requests are split into batches of :attr:`batch_size`.
        """

        densities = np.asarray(densities, dtype=np.float64)
        single = densities.ndim == 1
        if single:
            densities = densities[:, None]
        if densities.ndim != 2 or densities.shape[0] != self.size:
            raise ValueError("densities must have shape (active grid points, densities)")
        result = np.empty_like(densities)
        step = self.batch_size
        for start in range(0, densities.shape[1], step):
            stop = min(start + step, densities.shape[1])
            result[:, start:stop] = solve_block(self.plan, densities[:, start:stop]).T
        return result[:, 0] if single else result

    def __call__(self, density: np.ndarray) -> np.ndarray:
        density = np.asarray(density, dtype=np.float64)
        if density.shape != (self.size,):
            raise ValueError("density does not match the active grid")
        return self.solve_batch(density)

    # ------------------------------------------------------------ parity classes

    @property
    def parity_symmetric(self) -> bool:
        """Whether the box is symmetric about the origin along every axis (parity classes exist)."""

        return all(float(axis[0]) == -float(axis[-1]) for axis in self.axes)

    @property
    def parity_layout(self) -> ParityLayout:
        """The octant of the box and the active images of its points (module docstring)."""

        layout = getattr(self, "_parity_layout", None)
        if layout is None:
            if not self.parity_symmetric:
                raise ValueError(
                    "the parity classes need a box symmetric about the origin along every axis"
                )
            layout = _parity_layout(self.grid.integer_coordinates, self.box_lower, self.box_shape)
            self._parity_layout = layout
        return layout

    def _parity_axis(self, axis: int) -> dict:
        """Per parity ``+1``/``-1`` of one axis: block eigenvalues, octant matrices, monomials, Gaussians."""

        cache = self.__dict__.setdefault("_parity_axes", {})
        found = cache.get(axis)
        if found is not None:
            return found
        points = self.box_shape[axis]
        centre = points // 2
        half = points - centre
        bases = _reflection_bases(points)
        operator = dirichlet_second_derivative_matrix(points, self.finite_difference_order, self.spacing)
        size = self.multipole_order + 1
        vander = np.vander(self.axes[axis][centre:] / self.length_scale, size, increasing=True)
        weights = np.full(half, 2.0)
        if points % 2:
            weights[0] = 1.0  # the plane point is its own mirror
        gaussian = None
        if self.kernel != "coulomb" and self.screening == "split":
            coordinates = self.axes[axis]
            gaussian = (self.spacing * self.omega / math.sqrt(math.pi)) * np.exp(
                -(self.omega * (coordinates[:, None] - coordinates[None, :])) ** 2
            )
            spectra = {}
            for parity, (isometry, _octant, _scale) in bases.items():
                spectra[parity] = np.linalg.eigh(isometry.T @ gaussian @ isometry)
            largest = max(float(values.max()) for values, _ in spectra.values() if values.size)
        result = {}
        for parity, (isometry, octant, scale) in bases.items():
            values, vectors = np.linalg.eigh(isometry.T @ operator @ isometry)
            forward, inverse = _octant_matrices(vectors, octant, scale, half)
            rows = [power for power in range(size) if (-1) ** power == parity]
            entry = {
                "eigenvalues": values,
                "forward": forward,
                "inverse": inverse,
                "powers": rows,
                "monomials": np.ascontiguousarray((vander[:, rows] * weights[:, None]).T),
            }
            if gaussian is not None:
                kernel_values, kernel_vectors = spectra[parity]
                keep = kernel_values > self.rank_tolerance * largest
                kernel_values, kernel_vectors = kernel_values[keep], kernel_vectors[:, keep]
                # On spectral coefficients diag(s) U^T V; back to octant values U / scale.
                entry["contract"] = np.ascontiguousarray(
                    (kernel_values[:, None] * kernel_vectors.T) @ vectors
                )
                entry["expand"] = _octant_matrices(kernel_vectors, octant, scale, half)[1]
            result[parity] = entry
        cache[axis] = result
        return result

    def _harmonic_parities(self) -> np.ndarray:
        """Parity class index of every real harmonic column ``(l, m)`` (from its monomials)."""

        found = getattr(self, "_harmonic_parity", None)
        if found is not None:
            return found
        coefficients = solid_harmonic_coefficients(self.multipole_order)
        classes = np.empty(coefficients.shape[0], dtype=np.int64)
        for column in range(coefficients.shape[0]):
            a, b, c = np.nonzero(coefficients[column])
            bits = np.stack([a % 2, b % 2, c % 2], axis=1)
            if not np.all(bits == bits[0]):
                raise RuntimeError("a real solid harmonic without definite parity")
            classes[column] = (int(bits[0, 0]) << 2) | (int(bits[0, 1]) << 1) | int(bits[0, 2])
        self._harmonic_parity = classes
        return classes

    def _parity_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """The octant rows of the merged face correction and its columns there, ``(rows, (K, rows))``.

        All faces (near the inner planes only for boxes thinner than the
        stencil); each class keeps the rows of its moments.
        """

        found = getattr(self, "_parity_face_cache", None)
        if found is not None:
            return found
        layout = self.parity_layout
        face_rows, face_matrix = self.plan.faces[0]
        i, j, k = np.unravel_index(np.asarray(face_rows), self.box_shape)
        cx, cy, cz = layout.centre
        _hx, hy, hz = layout.half_shape
        inside = np.flatnonzero((i >= cx) & (j >= cy) & (k >= cz))
        rows = ((i[inside] - cx) * hy + (j[inside] - cy)) * hz + (k[inside] - cz)
        found = (np.ascontiguousarray(rows.astype(np.int64)), np.asarray(face_matrix)[:, inside])
        self._parity_face_cache = found
        return found

    def _parity_tails(self) -> np.ndarray:
        """The exterior tails ``T_lm`` on ``R``, ``(K, R)``.

        ``T_lm`` has the parity of its moment: at a point of ``R`` it is
        ``chi_t(s) T(s r)`` at its first active image ``s r`` (the point
        itself on a symmetric domain).  One gather serves all eight classes.
        (The full-box tails carry a round-off asymmetry of ~1e-10 of their
        largest value, from the cancellation in ``g * I~ - G_B I~``; averaging
        over the images changed no class solve beyond round-off.)
        """

        found = getattr(self, "_parity_tail_cache", None)
        if found is not None:
            return found
        layout = self.parity_layout
        source = np.asarray(self.plan.exterior_tail)
        classes = self._harmonic_parities()
        present = layout.images >= 0
        first = np.argmax(present, axis=0)
        rows = layout.images[first, np.arange(layout.size)]
        tails = np.take(source, rows, axis=1)
        for operation in np.unique(first):
            if operation == 0:
                continue
            signs = np.array([parity_character(t, int(operation)) for t in classes], dtype=np.float64)
            where = np.flatnonzero(first == operation)
            tails[:, where] *= signs[:, None]
        self._parity_tail_cache = tails
        return tails

    def parity_plan(self, parity) -> SolvePlan:
        """The :class:`SolvePlan` of class ``parity`` (index or signs) on the octant.

        Its box is the octant (:attr:`ParityLayout.half_shape`) and its
        active points the points ``R`` of :attr:`parity_layout`;
        :func:`solve_block` with it takes the class densities on ``R``
        (octant values) and returns their potentials there, and
        :func:`solve_box` takes them on the octant box.  Built once per
        class.
        """

        index = int(parity) if np.isscalar(parity) else parity_index(parity)
        if not 0 <= index < 8:
            raise ValueError("a parity class index is 0 ... 7")
        plans = self.__dict__.setdefault("_parity_plans", {})
        found = plans.get(index)
        if found is not None:
            return found
        layout = self.parity_layout
        signs = PARITY_CLASSES[index]
        axes = [self._parity_axis(d)[signs[d]] for d in range(3)]
        eigenvalues = -(
            axes[0]["eigenvalues"][:, None, None]
            + axes[1]["eigenvalues"][None, :, None]
            + axes[2]["eigenvalues"][None, None, :]
        )
        if np.any(eigenvalues <= 0.0):
            raise RuntimeError("the octant Laplacian of a parity class is not positive definite")
        size = self.multipole_order + 1
        columns = np.flatnonzero(self._harmonic_parities() == index)
        rows = [
            (a * size + b) * size + c
            for a in axes[0]["powers"]
            for b in axes[1]["powers"]
            for c in axes[2]["powers"]
        ]
        moment_map = np.ascontiguousarray(self._moment_map[np.ix_(rows, columns)])
        octant_rows, octant_faces = self._parity_faces()
        face = (octant_rows, np.ascontiguousarray(octant_faces[columns]))
        density_factor = boundary_factor = None
        if self.kernel != "coulomb" and self.screening == "spectral":
            density_factor = 8.0 * math.pi * spectral_function(self.kernel, eigenvalues, self.omega)
            boundary_factor = 1.0 / eigenvalues
        contract = expand = tails = None
        if self.plan.exterior_tail is not None:
            contract = tuple(entry["contract"] for entry in axes)
            expand = tuple(entry["expand"] for entry in axes)
            tails = np.ascontiguousarray(self._parity_tails()[columns])
            tails[:, layout.plane_mask(index)] = 0.0
        plan = SolvePlan(
            kernel=self.kernel,
            screening=self.screening,
            box_shape=layout.half_shape,
            box_index=layout.box_index,
            eigenvectors=tuple(entry["inverse"] for entry in axes),
            eigenvectors_t=tuple(entry["forward"] for entry in axes),
            inverse_eigenvalues=1.0 / eigenvalues,
            density_factor=density_factor,
            boundary_factor=boundary_factor,
            monomials_t=tuple(entry["monomials"] for entry in axes),
            moment_map=moment_map,
            faces=(face,),
            gaussian_contract=contract,
            gaussian_expand=expand,
            exterior_tail=tails,
            coulomb_weight=self.plan.coulomb_weight,
            long_range_weight=self.plan.long_range_weight,
        )
        plans[index] = plan
        return plan

    def solve_parity(self, parity, densities: np.ndarray) -> np.ndarray:
        """Potentials (Ry) on ``R`` of class-``parity`` densities on ``R``, shape ``(size, nb)``.

        The densities are the octant values of functions of that parity
        (:meth:`ParityLayout.components`); the potentials are the octant
        values of their full-box potentials.
        """

        plan = self.parity_plan(parity)
        layout = self.parity_layout
        densities = np.asarray(densities, dtype=np.float64)
        single = densities.ndim == 1
        if single:
            densities = densities[:, None]
        if densities.ndim != 2 or densities.shape[0] != layout.size:
            raise ValueError("class densities must have shape (octant points R, densities)")
        result = np.empty_like(densities)
        step = max(1, int(self.batch_memory // (4 * 8 * layout.half_size)))
        for start in range(0, densities.shape[1], step):
            stop = min(start + step, densities.shape[1])
            result[:, start:stop] = solve_block(plan, densities[:, start:stop]).T
        return result[:, 0] if single else result


__all__ = [
    "DEFAULT_BATCH_MEMORY",
    "KERNELS",
    "PARITY_CLASSES",
    "SCREENINGS",
    "SIGN_OPERATIONS",
    "BoxFace",
    "KroneckerPoisson",
    "ParityLayout",
    "SolvePlan",
    "dirichlet_second_derivative_matrix",
    "harmonic_index",
    "parity_character",
    "parity_character_table",
    "parity_index",
    "real_harmonics",
    "solid_harmonic_coefficients",
    "solve_block",
    "solve_box",
    "spectral_function",
]
