"""Coulomb kernels of the exact exchange in periodic cells: bulk (3-D) with k-points and slabs (2-D).

The exact exchange of Bloch orbitals needs the potential of every pair density

``rho_{nk,mk'}(r) = conj(psi_mk'(r)) psi_nk(r) = e^{i q.r} p(r)``,  ``q = k - k'``,

whose periodic part ``p`` lives on the grid of the cell.  The potential of the infinite crystal is again a Bloch
function, ``v(r) = e^{i q.r} w(r)``, and in reciprocal space ``w`` is ``p`` times the Fourier transform of the
interaction at the shifted wave vectors ``q + G``.  This module gives that kernel,
:meth:`PeriodicExchangeKernel.values`, on the FFT grid of the cell, for the interactions of the hybrid functionals
(:mod:`parsec_python.V_xc.hybrid`, libxc's CAM convention ``alpha/r + beta erfc(omega r)/r``), with the treatment of
the integrable singularity at ``q + G = 0`` that a finite k-point mesh needs, and for slabs with the Coulomb
interaction truncated along the vacuum direction.  It solves nothing itself; the Fock loops (host and device)
multiply with it.

Normalization (the contract of the callers)
-------------------------------------------
Units: Rydberg, ``e^2 = 2`` (a unit point charge has the potential ``2/r``), lengths in bohr.  The orbitals of a
k-point are Bloch factors ``u`` on the grid box, Euclidean normalized (``sum |u|^2 = 1``, ``u = sqrt(dV) psi``,
``dV`` the volume element).  With numpy's FFT conventions (``fftn`` unnormalized, ``ifftn`` with ``1/N``):

* ``p = conj(u_mk') u_nk / dV`` (electrons/bohr^3);
* ``w = ifftn(values(q) * fftn(p))`` is the periodic part of ``v(r) = 2 int rho(r') v(r - r') d^3r'`` (Ry), the
  potential of the pair density of the infinite crystal: no ``1/Omega`` and no ``1/N_k`` inside ``values``.  For a
  slab ``p`` is zero-padded at the end of axis 2 to :attr:`PeriodicExchangeKernel.fft_shape` and ``w`` cropped back
  to :attr:`PeriodicExchangeKernel.box_shape` (:meth:`PeriodicExchangeKernel.potential` does all of it);
* exchange operator (``f`` in ``[0, 1]`` per spin, ``N_k`` points of the full mesh):
  ``(V_x u_nk)(r) = -(1/N_k) sum_{k', m} f_mk' u_mk'(r) w_{nk,mk'}(r)``;
* exchange energy ``E_x = (s/2) sum_k w_k sum_n f_nk Re <u_nk | V_x u_nk>`` (``s = 2`` spin-unpolarized, ``1`` per
  spin channel, ``w_k`` the k weights summing to one), the formula of
  :mod:`parsec_python.V_xc.exact_exchange` with the k averages.

There is no separate energy constant.  The whole treatment of the singularity sits in one element of
``values(q)``, the one at ``q + G = 0`` (it exists when ``q`` is a reciprocal lattice vector, at the index of
``G = -q``): it multiplies the ``G = 0`` coefficient of the pair density, the overlap ``delta_nm / dV`` of two
orbitals of the same k-point, with the same ``1/N_k`` as every other element.  Its value is
:attr:`PeriodicExchangeKernel.divergence_element`.

Kernels
-------
Bulk (Hartree atomic units here; ``values`` returns twice these numbers), ``Q = |q + G|``: the two components

``K_coulomb(Q) = 4 pi/Q^2``,  ``K_erfc(Q) = (4 pi/Q^2) (1 - exp(-Q^2/(4 omega^2)))``,  ``K = alpha K_coulomb + beta K_erfc``

(``erfc(omega r)/r`` is finite at ``Q -> 0``, ``pi/omega^2``; ``erf(omega r)/r`` transforms to
``(4 pi/Q^2) exp(-Q^2/(4 omega^2))``, ``alpha = -beta``).  The FFT index ``m`` of an axis is numpy's ``fftfreq``
order; ``G = sum m_i b_i`` (``b_i = 2 pi inv(A)^T``, rows).  Along an axis with an even number ``n`` of points the
index ``-n/2`` stands for ``-n/2`` and ``+n/2`` alike; the element there is the mean of the kernel over both aliases
(over all ``2^d`` of them where ``d`` axes meet at their Nyquist index), so that ``values(-q)[-G] = values(q)[G]``
holds exactly (time reversal: ``u_{-k} = conj(u_k)``) and the potential of a real pair density at ``q = 0`` is real on
an oblique cell too.

The singularity at ``q + G = 0`` (Gygi and Baldereschi, Phys. Rev. B 34, 4405 (1986); in the form of Carrier,
Rohra and Goerling, Phys. Rev. B 75, 205126 (2007)).  The average over ``k'`` is a quadrature,

``(1/N_k) sum_{q in mesh} sum_G w(q+G) K(q+G) S(q+G)  ~  (Omega/(2 pi)^3) int K(Q) S(Q) d^3Q``,

on the Gamma-centred mesh of the ``q = k - k'`` (``N_1 x N_2 x N_3`` points ``j_i/N_i`` along ``b_i`` for any
Monkhorst-Pack mesh of those counts, shifted or not; together with ``G`` the reciprocal lattice of the Born-von Karman
supercell), with weights ``w`` (one, or the extrapolation below), where ``S`` is smooth with ``S(0) = S_0``, the
squared ``G = 0`` coefficient of the pair density of an orbital with itself.  With the auxiliary function
``F(Q) = K(Q) exp(-a Q^2)`` of each component, ``int K S = int (K S - S_0 F) + S_0 int F``: the first integrand is
regular and goes to the mesh, its value at ``Q = 0`` estimated as ``S_0 L``, ``L = lim_{Q->0} K(Q)(1 - exp(-a Q^2))``
(``4 pi a`` for ``1/r``, ``0`` for erfc); the second integral is analytic.  The mesh sum without its singular term
plus the element

``K_0 = L + N_k [ (Omega/(2 pi)^3) int F(Q) d^3Q - (1/N_k) sum'_{q,G} w F(q+G) ]``

at ``q + G = 0`` is the result (``sum'`` leaves ``Q = 0`` out).  The integrals are

``(Omega/(2 pi)^3) int F_coulomb d^3Q = Omega/sqrt(pi a)``,
``(Omega/(2 pi)^3) int F_erfc d^3Q = (Omega/sqrt(pi)) [1/sqrt(a) - 1/sqrt(a + 1/(4 omega^2))]``,

and the sums over the reciprocal lattice ``L*`` of the Born-von Karman supercell ``L`` (rows ``N_i a_i``, volume
``Omega_sc = N_k Omega``) are evaluated in closed form, at a cost independent of ``a``, the cell and the mesh, by
Ewald's identity

``sum'_{Q in L*} 4 pi exp(-c Q^2)/Q^2 = Omega_sc M_L(c) + 4 pi c + Omega_sc/sqrt(pi c)``,
``M_L(c) = phi_L - sum'_{R in L} erfc(R/(2 sqrt(c)))/R``,

``phi_L`` the potential at the origin of unit charges on ``L`` (the origin's own left out) in a neutralizing
background (the Madelung constant of the supercell, by Ewald's method).  The divergent ``Omega_sc/sqrt(pi a)`` cancels
against the integral, and with ``b = 1/(4 omega^2)``, ``f = 2^d/(2^d - 1)`` and ``L/2`` the lattice whose reciprocal
is the double grid below (rows ``N_i a_i/2``):

* ``1/r``: ``K_0 = -Omega_sc M_L(a)``; extrapolated ``-f [Omega_sc M_L(a) - (Omega_sc/8) M_L/2(a)]``;
* erfc: ``K_0 = 4 pi b - Omega_sc [M_L(a) - M_L(a + b)]``; extrapolated
  ``-f {Omega_sc [M_L(a) - M_L(a + b)] - (Omega_sc/8) [M_L/2(a) - M_L/2(a + b)]}``.

They equal the direct reciprocal sums to 1e-13 (unit test) and Quantum ESPRESSO's element to 2e-13 at its ``a``.

``L`` estimates the regular integrand at the origin with ``S`` flat there; the curvature of ``S`` (the spread of the
pair density) is missed, an error ``-(4 pi/(Omega N_k)) S''``-like that falls only as ``1/N_k``.  The extrapolation of
Nguyen and de Gironcoli (Phys. Rev. B 79, 205114 (2009)) avoids that value altogether: the weights are
``w = 2^d/(2^d - 1)`` off the "double grid" (the points ``q + G`` whose coordinates along every periodic ``b_i`` are
multiples of ``2/N_i``, the mesh of twice the spacing) and zero on it, ``d`` the number of periodic axes (a
quadrature of the same accuracy for smooth integrands, without the origin); ``L`` is then dropped.  For a component
without singularity the extrapolation only halves the resolution of the mesh: the erfc kernel has a peak of width
``omega`` at ``Q = 0`` (``0.11/bohr`` for HSE, comparable to the spacing of the mesh) that the plain mesh with the
element ``K_0`` integrates far better.  ``x_gamma_extrapolation`` therefore chooses per component:

* ``"coulomb"`` (the default, except for the truncated slab kernel): the weights on the ``1/r`` component only, the
  erfc component unweighted;
* ``True``: on both (Quantum ESPRESSO's default, ``x_gamma_extrapolation = .true.``);
* ``False`` (the default of the truncated slab kernel, see below): on neither (Quantum ESPRESSO's
  ``x_gamma_extrapolation = .false.``).

Measured on Gaussian pair densities in the primitive cell of Si (``tests/unit/test_exchange_kernel.py``), errors of
the k-averaged energy against the isolated limit at ``4 x 4 x 4``, spherical / anisotropic Gaussian: ``1/r``
5e-4 / 1.3e-3 Ry (False) and 2e-8 / 3.7e-6 Ry (extrapolated); erfc (HSE, ``omega = 0.11``) 1e-6 / 3e-6 Ry
(unweighted) and 4e-4 Ry (extrapolated); erf, the sum of both, 5e-4 / 1.3e-3 (False), 4e-4 / 3.4e-4 (True) and
1e-6 / 6e-7 Ry (``"coulomb"``; the auxiliary exponent at its default, the limit below).  The weights help the erfc component only on meshes far too coarse to resolve ``omega`` (spacing
above about ``5 omega``), where every treatment errs by 1e-3 Ry or more.

With a negative erfc coefficient (``alpha/r + beta erfc(omega r)/r``, ``beta < 0``: LC and CAM functionals whose
long range is ``alpha``, ``alpha + beta = 0`` the pure erf kernel) the ``"coulomb"`` weights leave
``beta K_erfc < 0`` on the double grid.  Where the mesh resolves ``omega`` the sum over ``q`` stays positive; where it
does not, the exchange operator becomes indefinite: exchange matrices ``-U^H V_x U`` of random band-limited orbitals
with empty bands in the Si cell had negative eigenvalues (-7e-4 to -8e-3 of the largest) for ``omega = 0.3`` at
1 x 1 x 1 (mesh spacing 3.5 omega) and ``omega = 0.11`` up to 3 x 3 x 3 (3.2 omega), none at spacings of 1.2 and
1.8 omega or with weights on both components (``K >= 0``).  ``ACE`` needs ``-U^H V_x U`` positive (a Cholesky
factor), so the automatic choice for ``beta < 0`` weights both components unless the largest mesh spacing
``max_i |b_i|/N_i`` is at most ``1.5 omega`` (unit test).

``divergence="none"``: ``K_0`` is the finite part of the kernel at ``Q -> 0`` with the singular ``1/r`` term
dropped (``0`` for ``1/r``, ``pi/omega^2`` for erfc), zero for a component with the extrapolation weights.

The exponent ``a`` (bohr^2, :attr:`auxiliary_exponent`) only shapes the regular remainder.  Its default is the limit
``a -> 0`` of the closed forms (``M_L(0) = phi_L``): the ``1/r`` element is then the Madelung term of the supercell
(and of its half), the erfc element ``pi/omega^2 - Omega_sc sum'_R erfc(omega R)/R`` (the exact limit minus the
supercell images of a point pair density), with no length scale of the grid in them.  For ``1/r`` the result does not
depend on ``a`` while ``a`` is small against the mesh (Si Gaussians: identical to 1e-10 Ry for ``a`` from 0 to 0.2
bohr^2 at every mesh, to 0.5 from 2 x 2 x 2 on; ``a = 2`` alters 1 x 1 x 1 and 2 x 2 x 2 by 7e-2 and 1e-4 Ry).
The erfc component depends on ``a`` wherever the mesh does not resolve ``omega``, because there ``a`` decides how the
unresolved peak is shared between the element and the remainder: Si Gaussians at 1 x 1 x 1 / 2 x 2 x 2 / 3 x 3 x 3 /
4 x 4 x 4 with ``a`` from 0 to 0.53 bohr^2 change by 5e-2 / 3e-3 / 1.3e-4 / 1.6e-6 Ry and the error changes sign in
that range; the HSE ``E_x`` of the Si cell at Gamma moved by 0.037 Ry between ``a`` = 0.25 and 1/6.  Such numbers are
the k-point error of the coarse mesh, not of ``a``; an ``a`` tied to the grid step (as Quantum ESPRESSO's
``10/ecutwfc``) moves them in a convergence study of the step.  Quantum ESPRESSO's value is to be passed for a
like-for-like comparison.  The truncated slab kernel (below) sums in two dimensions directly and needs ``a > 0``:
its default is ``10/E_w``, ``E_w = (G_box/2)^2`` Ry, ``G_box = min_i pi/h_i`` (the wavefunction cutoff of a
plane-wave grid holding the density, the analogue of Quantum ESPRESSO's choice).

Slabs
-----
Periodic along ``a_1``, ``a_2`` (in the ``xy`` plane), open along axis 2, ``a_3 = n_3 h_z e_z`` (the grid box; the
orbitals vanish beyond it).  The kernel works on a box padded to ``P`` planes along ``z`` (:attr:`fft_shape`,
``vacuum_planes``; default ``2 n_3``): the pair densities are zero-padded, the potentials cropped.  Two kernels
(``slab_coulomb``):

``"padded"`` (default): the bulk kernel above on the padded box, ``a_3 -> P h_z e_z`` with one k-point along ``z``
(the Gygi-Baldereschi sums over the reciprocal lattice of that box, the extrapolation weights ``8/7`` and ``0`` with
the double grid of the even ``G_z``): the exchange of a stack of layers ``P h_z`` apart, at least twice the height of
the box.  This is Quantum ESPRESSO's exchange of a slab (it has no two-dimensional truncation of the exact exchange;
its ``assume_isolated='2D'`` cuts the Hartree, local and Ewald terms only) on a box twice as high.  The images of a
layer cost little (Gaussian models below: 2e-6 to 9e-6 Ry against ``P = 4 n_3``; the reference stream found
QE's gbx energies of h-BN independent of the cell height from 20 to 40 bohr to 5e-6 Ry), and the three-dimensional
quadrature of the in-plane k-mesh with the extrapolation on ``1/r`` handles the anisotropy of a slab's pair
densities, which the truncated kernel cannot (next paragraph).

``"truncated"``: the Coulomb interaction truncated at ``z_c = P h_z/2``.  For ``P >= 2 n_3 - 1`` every separation of
two points of the box is shorter than ``z_c`` and every periodic image farther, so the potential in the box is that of
the slab isolated along ``z`` exactly (the convention of the slab Hartree solvers, :mod:`.slab` and
:mod:`.slab_fortran`); ``P = n_3`` is the unpadded truncation, exact only for densities within the central half of
the box.  ``Q_par = |q + G_par|``, ``G_z = 2 pi m_3/(P h_z)``:

* ``1/r``: the truncated kernel (Ismail-Beigi, Phys. Rev. B 73, 233103 (2006); Rozzi et al., Phys. Rev. B 73,
  205119 (2006)) ``(4 pi/Q^2) [1 - exp(-Q_par z_c) cos(G_z z_c)]``, the Fourier coefficient of ``1/r`` on
  ``|z| < z_c``;
* ``erf(omega r)/r``: the in-plane transform of the interaction at height ``z`` (the potential of a Gaussian sheet,
  as in the Ewald sum of slabs)
  ``v(Q_par, z) = (pi/Q_par) [e^{Q_par z} erfc(Q_par/(2 omega) + omega z) + e^{-Q_par z} erfc(Q_par/(2 omega) - omega z)]``,
  sampled at the plane separations ``z_s = s h_z`` (``|s| <= P/2``) and transformed along ``z``:
  ``K(Q_par, G_z) = h_z sum_s v(Q_par, z_s) e^{-i G_z z_s}``.  This is the discrete convolution of the planes,
  spectrally accurate because ``v`` is smooth on the length ``1/omega``.  At ``Q_par = 0`` the part ``2 pi/Q_par``
  is constant in ``z`` (``G_z = 0`` only) and the rest is ``-2 pi [z erf(omega z) + exp(-omega^2 z^2)/(omega sqrt(pi))]``;
* ``erfc = 1/r - erf``.

The remaining singularity is in-plane, ``K_coulomb(Q_par, G_z = 0) ~ 2 pi H/Q_par`` (``H = P h_z``).  The element at
``q + G = 0`` is the two-dimensional form of the same construction,
``K_0 = lim_{Q->0}[K(Q, 0) - F(Q)] + N_k [ (A/(2 pi)^2) int F d^2Q_par - (1/N_k) sum'_{q, G_par} w F(q + G_par) ]``,
``A = |a_1 x a_2|``, with auxiliary functions that are a pure ``1/Q`` singularity times an even analytic function:
``F = (2 pi H/Q) erfc(sqrt(a) Q)`` for ``1/r`` (integral ``A H/sqrt(pi a)``, limit ``-2 pi z_c^2 + 4 H sqrt(pi a)``)
and the thin sheet ``F = H (2 pi/Q) erf(Q/(2 omega)) exp(-a Q^2)`` for erfc (integral
``A H atan(1/(2 omega sqrt(a)))/sqrt(pi a)``, limit ``K(0, 0) - 2 sqrt(pi) H/omega``).  (``K(Q_par, 0) exp(-a Q^2)``
itself would not do: the truncated kernel carries a term ``(2 pi/3) z_c^3 |Q|`` that only the ``G_z != 0`` elements
compensate.)  Here the value at the origin is exact, so the extrapolation is not used (its default is ``False``).  But
the exact integration along ``z`` turns the direction-dependent limit of an anisotropic pair density at ``Q = 0`` into
a cusp ``|Q_par|`` of the in-plane integrand, and the k-average converges only as ``N_k^(-3/2)``: Gaussian pair
densities of exponents (1, 0.6, 0.35) and (0.25, 0.25, 1.5) in a 6 x 6 x 16 bohr box, errors of ``1/r`` at
``N x N x 1``, truncated / padded: N = 3: 2.3e-3 / 5.4e-5 and 5.2e-3 / 2.9e-4 Ry; N = 4: 9.9e-4 / 1.4e-5 and
2.2e-3 / 6.5e-5; N = 6: 3.0e-4 / 4.0e-6 and 6.5e-4 / 1.6e-5; erf alike, erfc equal (both converge exponentially).
For isotropic densities the truncated kernel is exact from 2 x 2 on.  ``divergence="none"``: the finite part
``-2 pi z_c^2`` of the ``1/r`` component at ``Q_par -> 0`` (its ``1/Q_par`` term dropped) and the limit of the erfc
component.

Quantum ESPRESSO's own convention is the bulk kernel on the unpadded box, ``periodic_axes=(0, 1, 2)`` here with
``k_mesh = (N_1, N_2, 1)``.

Device arrays
-------------
``values(q, xp=cupy)`` builds the same array with CuPy (elementwise; a truncated slab kernel with an erfc or erf
part adds one FFT along ``z``); the scalars (the element ``K_0``, the reciprocal metric) are computed once on the
host in the constructor.  The performance counterpart
:func:`parsec_python.acceleration.Hartree.exchange_kernel_device.device_values` builds the bulk and padded-slab
arrays in one CUDA launch.
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from ..Grid.cell import reciprocal_lattice

#: Treatments of the ``q + G = 0`` singularity.
DIVERGENCE_TREATMENTS = ("gygi-baldereschi", "none")
#: Choices of ``x_gamma_extrapolation``: on no component, on all, on the ``1/r`` component only.
X_GAMMA_CHOICES = (False, True, "coulomb")
#: Kernels of a slab: 3-D periodic on the box padded along z (default) or the Coulomb interaction truncated along z.
SLAB_COULOMB_CHOICES = ("padded", "truncated")
# The auxiliary sums and integrals run while exp(-a Q^2) exceeds exp(-_AUXILIARY_CUTOFF) (1e-24 relative).
_AUXILIARY_CUTOFF = 56.0
# A q + G whose squared length is below this fraction of the smallest mesh spacing squared is q + G = 0.
_SINGULAR_TOLERANCE = 1.0e-16
# Fractional coordinates this close to a multiple count as on it (q on the mesh, the double grid).
_FRACTION_TOLERANCE = 1.0e-7
# Upper bound of the number of points of one slice of the auxiliary sums (memory of the host evaluation).
_SUM_CHUNK = 4_000_000
# The automatic extrapolation of a kernel with a negative erfc coefficient (LC, CAM with alpha + beta = 0): the
# "coulomb" weights only where the mesh spacing max_i |b_i|/N_i is at most this many omega (module docstring).
_LONG_RANGE_RESOLUTION = 1.5


def _lattice_points(rows: np.ndarray, radius: float) -> np.ndarray:
    """Cartesian points ``n . rows`` (integer ``n``) of length at most ``radius``, the origin excluded."""

    rows = np.asarray(rows, dtype=np.float64)
    duals = np.linalg.inv(rows).T  # rows: the dual vectors (n_i = r . dual_i)
    bounds = [int(math.ceil(radius * np.linalg.norm(dual))) + 1 for dual in duals]
    grids = np.meshgrid(*(np.arange(-bound, bound + 1) for bound in bounds), indexing="ij")
    integers = np.stack([grid.reshape(-1) for grid in grids], axis=1).astype(np.float64)
    points = integers @ rows
    lengths = np.linalg.norm(points, axis=1)
    keep = (lengths > 0.0) & (lengths <= radius)
    return points[keep]


def _madelung_remainder(rows: np.ndarray, exponents) -> np.ndarray:
    """``M_L(c) = phi_L - sum'_R erfc(R/(2 sqrt(c)))/R`` (Hartree, bohr^-1) of the lattice ``L`` with the ``rows``.

    ``phi_L`` is the potential at the origin of unit point charges on ``L`` without the one at the origin, in a
    uniform neutralizing background (Ewald: ``sum'_R erfc(eta R)/R + (4 pi/Omega) sum'_Q exp(-Q^2/(4 eta^2))/Q^2
    - pi/(Omega eta^2) - 2 eta/sqrt(pi)``, independent of ``eta``).  With it the reciprocal sums of the auxiliary
    functions take a closed form for any ``c`` (Ewald's identity with ``eta = 1/(2 sqrt(c))``):

    ``sum'_{Q in L*} 4 pi exp(-c Q^2)/Q^2 = Omega M_L(c) + 4 pi c + Omega/sqrt(pi c)``.
    """

    from scipy.special import erfc

    rows = np.asarray(rows, dtype=np.float64)
    volume = abs(float(np.linalg.det(rows)))
    eta = math.sqrt(math.pi) / volume ** (1.0 / 3.0)
    real = _lattice_points(rows, 6.5 / eta)
    lengths = np.linalg.norm(real, axis=1)
    reciprocal = _lattice_points(2.0 * np.pi * np.linalg.inv(rows).T, 2.0 * eta * math.sqrt(42.0))
    squared = np.sum(reciprocal * reciprocal, axis=1)
    phi = (
        float(np.sum(erfc(eta * lengths) / lengths))
        + 4.0 * np.pi / volume * float(np.sum(np.exp(-squared / (4.0 * eta * eta)) / squared))
        - np.pi / (volume * eta * eta)
        - 2.0 * eta / math.sqrt(math.pi)
    )
    out = []
    for c in np.atleast_1d(np.asarray(exponents, dtype=np.float64)):
        if c == 0.0:
            out.append(phi)  # erfc(R/(2 sqrt(c))) = 0
            continue
        reach = 2.0 * math.sqrt(c) * 6.5
        points = _lattice_points(rows, reach)
        distances = np.linalg.norm(points, axis=1)
        out.append(phi - float(np.sum(erfc(distances / (2.0 * math.sqrt(c))) / distances)))
    return np.asarray(out)


def _special(xp):
    """``(erf, erfc, erfcx)`` of the array module ``xp`` (scipy for numpy, cupyx for CuPy)."""

    if xp is np:
        from scipy.special import erf, erfc, erfcx

        return erf, erfc, erfcx
    from cupyx.scipy.special import erf, erfc, erfcx  # pragma: no cover - device only

    return erf, erfc, erfcx  # pragma: no cover


def _fft_frequencies(n: int) -> np.ndarray:
    """Integer FFT frequencies of ``n`` points in numpy's order (``-n/2 .. n/2 - 1``)."""

    return np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(np.int64)


def _sheet_terms(transverse, z, omega, xp):
    """``(e^{Qz} erfc(Q/2w + w z), e^{-Qz} erfc(Q/2w - w z))`` without overflow (``erfcx`` where its argument >= 0)."""

    _erf, erfc, erfcx = _special(xp)
    half = transverse / (2.0 * omega)
    gauss = xp.exp(-(half * half) - (omega * z) ** 2)
    upper = half + omega * z
    lower = half - omega * z
    plus = xp.where(
        upper >= 0.0,
        erfcx(xp.maximum(upper, 0.0)) * gauss,
        xp.exp(xp.minimum(transverse * z, 0.0)) * erfc(upper),
    )
    minus = xp.where(
        lower >= 0.0,
        erfcx(xp.maximum(lower, 0.0)) * gauss,
        xp.exp(xp.minimum(-transverse * z, 0.0)) * erfc(lower),
    )
    return plus, minus


def erf_sheet_residual(transverse, z, omega, xp=np):
    """``v(Q, z) - 2 pi/Q`` (Hartree) of ``erf(omega r)/r`` for ``Q > 0``: the in-plane transform minus its singular part.

    ``v(Q, z) = (pi/Q)[e^{Qz} erfc(Q/(2 omega) + omega z) + e^{-Qz} erfc(Q/(2 omega) - omega z)]`` is evaluated
    with ``erfcx`` where its argument is positive (``e^{Qz} erfc(Q/(2 omega) + omega z) =
    erfcx(Q/(2 omega) + omega z) exp(-Q^2/(4 omega^2) - omega^2 z^2)``), so that neither factor overflows.
    ``transverse`` and ``z`` broadcast against each other.
    """

    transverse = xp.asarray(transverse, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)
    plus, minus = _sheet_terms(transverse, z, omega, xp)
    return (np.pi / transverse) * (plus + minus - 2.0)


def erf_sheet_limit(z, omega, xp=np):
    """``lim_{Q->0} [v(Q, z) - 2 pi/Q] = -2 pi [z erf(omega z) + exp(-omega^2 z^2)/(omega sqrt(pi))]`` (Hartree)."""

    erf, _erfc, _erfcx = _special(xp)
    z = xp.asarray(z, dtype=xp.float64)
    return -2.0 * np.pi * (z * erf(omega * z) + xp.exp(-((omega * z) ** 2)) / (omega * math.sqrt(math.pi)))


class PeriodicExchangeKernel:
    """``K(q + G)`` of the exact exchange on the FFT grid of a periodic cell (module docstring: the contract).

    Parameters
    ----------
    lattice
        Rows: the vectors (bohr) of the box the orbitals live on, ``step_vectors * n`` of the grid: the cell of a
        bulk system; ``a_1``, ``a_2`` in the ``xy`` plane and ``a_3 = n_3 h_z e_z`` for a slab.
    shape
        Grid points ``(n_1, n_2, n_3)`` of the box along the three lattice vectors.
    k_mesh
        Counts ``(N_1, N_2, N_3)`` of the full Monkhorst-Pack mesh the exchange sums over (``N_3 = 1`` for a slab,
        ``(1, 1, 1)`` at Gamma).  The differences ``q = k - k'`` lie on the Gamma-centred mesh of these counts.
    alpha, beta, omega
        libxc's CAM coefficients: the interaction ``alpha/r + beta erfc(omega r)/r`` (``omega`` in bohr^-1, needed
        when ``beta != 0``).
    periodic_axes
        ``(0, 1, 2)`` bulk; ``(0, 1)`` slab, open along axis 2 (which must be perpendicular to the other two).
    divergence
        ``"gygi-baldereschi"`` (default) or ``"none"``: the element at ``q + G = 0``.
    x_gamma_extrapolation
        ``"coulomb"``, ``True`` or ``False``: which components carry the extrapolation weights; ``None`` (default):
        ``"coulomb"``, ``False`` for a truncated slab kernel (module docstring: the measurements behind the choice).
    auxiliary_exponent
        ``a`` of the Gaussian factor ``exp(-a Q^2)`` of the auxiliary functions (bohr^2); default the limit ``a -> 0``
        (``10/E_w`` for the truncated slab kernel).
    vacuum_planes
        Slab only: planes ``P`` of the padded box along ``z`` (default ``2 n_3``; at least ``n_3``).
    slab_coulomb
        Slab only: ``"padded"`` (default, the bulk kernel on the padded box) or ``"truncated"`` (module docstring).
    """

    def __init__(
        self,
        lattice,
        shape: Sequence[int],
        k_mesh: Sequence[int],
        *,
        alpha: float,
        beta: float,
        omega: float | None = None,
        periodic_axes: Sequence[int] = (0, 1, 2),
        divergence: str = "gygi-baldereschi",
        x_gamma_extrapolation: bool | str | None = None,
        auxiliary_exponent: float | None = None,
        vacuum_planes: int | None = None,
        slab_coulomb: str | None = None,
    ) -> None:
        lattice = np.array(lattice, dtype=np.float64)
        if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
            raise ValueError("the lattice must be three finite vectors (rows)")
        volume = abs(float(np.linalg.det(lattice)))
        if not volume > 0.0:
            raise ValueError("the lattice vectors are linearly dependent")
        box_shape = tuple(int(value) for value in shape)
        if len(box_shape) != 3 or min(box_shape) < 1:
            raise ValueError("shape must be three positive point counts")
        mesh = tuple(int(value) for value in k_mesh)
        if len(mesh) != 3 or min(mesh) < 1:
            raise ValueError("k_mesh must be three positive counts")
        axes = tuple(int(axis) for axis in periodic_axes)
        if axes not in ((0, 1, 2), (0, 1)):
            raise ValueError(
                "periodic_axes must be (0, 1, 2) (bulk) or (0, 1) (slab, open along axis 2); "
                "wires are not supported"
            )
        alpha = float(alpha)
        beta = float(beta)
        if not (np.isfinite(alpha) and np.isfinite(beta)):
            raise ValueError("alpha and beta must be finite")
        if beta != 0.0:
            if omega is None or not np.isfinite(omega) or float(omega) <= 0.0:
                raise ValueError("a range-separated kernel (beta != 0) needs a positive omega (bohr^-1)")
            omega = float(omega)
        else:
            omega = None
        if divergence not in DIVERGENCE_TREATMENTS:
            raise ValueError(f"divergence must be one of {DIVERGENCE_TREATMENTS}, not {divergence!r}")
        slab = axes == (0, 1)
        if slab:
            slab_coulomb = "padded" if slab_coulomb is None else slab_coulomb
            if slab_coulomb not in SLAB_COULOMB_CHOICES:
                raise ValueError(f"slab_coulomb must be one of {SLAB_COULOMB_CHOICES}, not {slab_coulomb!r}")
        elif slab_coulomb is not None:
            raise ValueError("slab_coulomb belongs to slabs (periodic_axes (0, 1))")
        truncated = slab and slab_coulomb == "truncated"
        if x_gamma_extrapolation is None:
            x_gamma_extrapolation = False if truncated else "coulomb"
            if not truncated and beta < 0.0:
                # alpha/r + beta erfc/r with beta < 0: the "coulomb" weights leave -beta K_erfc on the double grid,
                # negative; on a mesh that does not resolve omega that makes the exchange operator indefinite.
                reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
                spacing = max(float(np.linalg.norm(reciprocal[axis])) / mesh[axis] for axis in axes)
                if spacing > _LONG_RANGE_RESOLUTION * omega:
                    x_gamma_extrapolation = True
        elif isinstance(x_gamma_extrapolation, (bool, np.bool_)):
            x_gamma_extrapolation = bool(x_gamma_extrapolation)
        elif x_gamma_extrapolation != "coulomb":
            raise ValueError(f"x_gamma_extrapolation must be one of {X_GAMMA_CHOICES}, not {x_gamma_extrapolation!r}")

        if slab:
            if mesh[2] != 1:
                raise ValueError("a slab has no k-points along its open axis 2 (k_mesh[2] must be 1)")
            a1, a2, a3 = lattice
            normal = np.cross(a1, a2)
            if np.linalg.norm(np.cross(normal, a3)) > 1.0e-10 * np.linalg.norm(normal) * np.linalg.norm(a3):
                raise ValueError("the open axis of a slab must be perpendicular to its two periodic vectors")
            planes = 2 * box_shape[2] if vacuum_planes is None else int(vacuum_planes)
            if planes < box_shape[2]:
                raise ValueError("vacuum_planes must be at least the planes of the box")
            fft_shape = (box_shape[0], box_shape[1], planes)
            fft_lattice = lattice.copy()
            fft_lattice[2] = lattice[2] * (planes / box_shape[2])
        else:
            if vacuum_planes is not None:
                raise ValueError("vacuum_planes belongs to slabs")
            fft_shape = box_shape
            fft_lattice = lattice.copy()

        self.lattice = lattice
        self.box_shape = box_shape
        self.fft_shape = fft_shape
        self.fft_lattice = fft_lattice
        self.k_mesh = mesh
        self.periodic_axes = axes
        self.is_slab = slab
        self.slab_coulomb = slab_coulomb
        self.truncated = truncated
        # The axes of the reciprocal sums: a padded slab is a 3-D periodic system with one k-point along z.
        self.sum_axes = (0, 1) if truncated else (0, 1, 2)
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.divergence = divergence
        self.x_gamma_extrapolation = x_gamma_extrapolation
        # Which components carry the extrapolation weights: (1/r, erfc).
        self.weighted = (
            x_gamma_extrapolation is not False,
            x_gamma_extrapolation is True,
        )
        self.reciprocal = reciprocal_lattice(fft_lattice)
        self.metric = self.reciprocal @ self.reciprocal.T
        self.volume = abs(float(np.linalg.det(fft_lattice)))
        self.cell_volume = volume
        self.k_points = int(np.prod(mesh))
        self.dimension = len(self.sum_axes)
        self.extrapolation_factor = 2.0**self.dimension / (2.0**self.dimension - 1.0)
        if auxiliary_exponent is None:
            if truncated:
                # The two-dimensional sums of the truncated kernel need a > 0: 10/E_w, E_w = (G_box/2)^2 Ry.
                steps = np.linalg.norm(lattice, axis=1) / np.asarray(box_shape, dtype=np.float64)
                auxiliary_exponent = 10.0 / (0.25 * float(np.min(np.pi / steps)) ** 2)
            else:
                auxiliary_exponent = 0.0  # the limit a -> 0 of the closed forms (module docstring)
        auxiliary_exponent = float(auxiliary_exponent)
        if not (np.isfinite(auxiliary_exponent) and auxiliary_exponent >= 0.0):
            raise ValueError("auxiliary_exponent must be finite and not negative (bohr^2)")
        if truncated and auxiliary_exponent == 0.0:
            raise ValueError("the truncated slab kernel needs a positive auxiliary_exponent")
        self.auxiliary_exponent = auxiliary_exponent
        if slab:
            self.plane_spacing = float(np.linalg.norm(lattice[2])) / box_shape[2]
            self.padded_height = self.plane_spacing * fft_shape[2]
            self.cutoff_height = 0.5 * self.padded_height
            self.area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))
            separations = _fft_frequencies(fft_shape[2]).astype(np.float64)
            if fft_shape[2] % 2 == 0:
                separations[fft_shape[2] // 2] = fft_shape[2] // 2  # z_c (the kernel is even)
            self.separations = separations * self.plane_spacing
        # The smallest |q + G|^2 of the mesh other than zero sets the test of q + G = 0.
        spacing = np.min(
            np.linalg.norm(self.reciprocal[list(self.sum_axes)], axis=1) / np.asarray(mesh)[list(self.sum_axes)]
        )
        self.singular_threshold = _SINGULAR_TOLERANCE * float(spacing) ** 2
        self.divergence_element = self._divergence_element()

    # -- shapes and helpers ------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """The shape of :meth:`values` (= :attr:`fft_shape`)."""

        return self.fft_shape

    def describe(self) -> str:
        """One line for output files: the geometry and the treatment of ``q + G = 0``."""

        if not self.is_slab:
            geometry = "bulk"
        elif self.truncated:
            geometry = (
                f"slab, Coulomb truncated along a3 on a box of {self.fft_shape[2]} planes "
                f"(z_c = {self.cutoff_height:.4f} bohr)"
            )
        else:
            geometry = (
                f"slab, 3-D periodic on the box padded to {self.fft_shape[2]} planes "
                f"(period {self.padded_height:.4f} bohr along a3)"
            )
        extrapolation = {False: "none", True: "all components", "coulomb": "1/r component"}[
            self.x_gamma_extrapolation
        ]
        return (
            f"exchange kernel: {geometry}, k-mesh {'x'.join(str(n) for n in self.k_mesh)}, "
            f"q+G=0: {self.divergence} (x_gamma extrapolation: {extrapolation}, "
            f"auxiliary exponent {self.auxiliary_exponent:.6g} bohr^2"
            + (" (the limit a -> 0)" if self.auxiliary_exponent == 0.0 else "")
            + f", element {self.divergence_element:.10g} Ry)"
        )

    def fractional(self, q) -> np.ndarray:
        """Coordinates of the Cartesian ``q`` (bohr^-1) along the reciprocal vectors of the box (``q.a_i/(2 pi)``)."""

        q = np.asarray(q, dtype=np.float64).reshape(3)
        return self.lattice @ q / (2.0 * np.pi)

    def mesh_fraction(self, q) -> np.ndarray:
        """:meth:`fractional` of a ``q`` checked against the mesh and snapped onto it.

        Along a periodic axis the coordinate times ``N_i`` must lie within 1e-7 of an integer ``j``; the result
        is ``j/N_i`` exactly, so that every later test (``q + G = 0``, the double grid) is exact on the host and
        the device alike.  A slab's ``q`` must lie in its plane; its axis 2 is returned as 0.
        """

        fraction = self.fractional(q)
        if self.is_slab and abs(fraction[2]) > _FRACTION_TOLERANCE:
            raise ValueError("a slab's q must lie in the plane of its periodic vectors")
        for axis in self.periodic_axes:
            scaled = fraction[axis] * self.k_mesh[axis]
            if abs(scaled - round(scaled)) > _FRACTION_TOLERANCE * max(1.0, abs(scaled)):
                raise ValueError(
                    f"q = {np.asarray(q).tolist()} is not a difference of two points of the {self.k_mesh} mesh "
                    "(the kernel's k_mesh must be the mesh of the exchange sum)"
                )
            fraction[axis] = round(scaled) / self.k_mesh[axis]
        if self.is_slab:
            fraction[2] = 0.0
        return fraction

    def is_singular(self, q) -> bool:
        """Whether ``q`` is a reciprocal lattice vector (``q + G = 0`` for one ``G``), after :meth:`mesh_fraction`."""

        fraction = self.mesh_fraction(q)
        return bool(all(fraction[axis] == round(fraction[axis]) for axis in self.periodic_axes))

    def pad(self, box, xp=np):
        """``box`` (``box_shape`` in its last three axes) zero-padded to :attr:`fft_shape` along axis 2."""

        if not self.is_slab or self.fft_shape[2] == self.box_shape[2]:
            return box
        width = [(0, 0)] * (box.ndim - 1) + [(0, self.fft_shape[2] - self.box_shape[2])]
        return xp.pad(box, width)

    def crop(self, padded):
        """The box part of an array of :attr:`fft_shape` in its last three axes."""

        if not self.is_slab or self.fft_shape[2] == self.box_shape[2]:
            return padded
        return padded[..., : self.box_shape[2]]

    def potential(self, pair_density, q, xp=np):
        """``w = crop(ifftn(values(q) fftn(pad(p))))``: the periodic part (Ry) of the potential of ``p`` (box shape).

        The reference application of the contract (``p`` may carry leading batch axes); the result is complex
        unless ``p`` is real and ``q = 0``, where it is real (the real part; ``K(G) = K(-G)`` there).
        """

        density = xp.asarray(pair_density)
        if tuple(density.shape[-3:]) != self.box_shape:
            raise ValueError("the pair density does not have the box shape of the kernel")
        values = self.values(q, xp=xp)
        transform = xp.fft.fftn(self.pad(density, xp), axes=(-3, -2, -1))
        potential = self.crop(xp.fft.ifftn(transform * values, axes=(-3, -2, -1)))
        if not xp.iscomplexobj(density) and not np.any(self.mesh_fraction(q)):
            return potential.real  # K(G) = K(-G) at q = 0 only
        return potential

    # -- the components (Hartree units) --------------------------------------

    def _components(self):
        """``(name, coefficient, weighted)`` of the components present: ``1/r`` (alpha) and erfc (beta)."""

        parts = []
        if self.alpha != 0.0:
            parts.append(("coulomb", self.alpha, self.weighted[0]))
        if self.beta != 0.0:
            parts.append(("erfc", self.beta, self.weighted[1]))
        return parts

    def _bulk_component(self, name, squared, xp):
        """``K_coulomb`` or ``K_erfc`` (Hartree) at ``Q^2 > 0``."""

        if name == "coulomb":
            return 4.0 * np.pi / squared
        return (4.0 * np.pi / squared) * (-xp.expm1(-squared / (4.0 * self.omega * self.omega)))

    def _bulk_kernel(self, squared, weight, xp):
        """The weighted kernel (Hartree) at ``Q^2 > 0``; ``weight`` the extrapolation weights (or ``None``).

        Where both components carry the same weights, ``(4 pi/Q^2)[alpha e^{-x} + (alpha + beta)(1 - e^{-x})]``,
        ``x = Q^2/(4 omega^2)``: exact for ``alpha + beta = 0`` (erf) and ``alpha = 0`` (erfc) alike.
        """

        if self.beta == 0.0:
            kernel = (4.0 * np.pi * self.alpha) / squared
            return kernel if weight is None or not self.weighted[0] else kernel * weight
        x = squared / (4.0 * self.omega * self.omega)
        if self.weighted[0] == self.weighted[1]:
            kernel = (4.0 * np.pi / squared) * (self.alpha * xp.exp(-x) - (self.alpha + self.beta) * xp.expm1(-x))
            return kernel if weight is None or not self.weighted[0] else kernel * weight
        long_range = self.alpha if weight is None else self.alpha * weight
        return (4.0 * np.pi / squared) * (long_range - self.beta * xp.expm1(-x))

    def _slab_coulomb(self, transverse, gz, xp):
        """Truncated ``1/r`` (Hartree): ``(4 pi/Q^2)[1 - e^{-Q_par z_c} cos(G_z z_c)]`` for ``Q^2 > 0``."""

        zc = self.cutoff_height
        squared = transverse * transverse + gz * gz
        decay = xp.exp(-transverse * zc)
        numerator = -xp.expm1(-transverse * zc) + decay * (2.0 * xp.sin(0.5 * gz * zc) ** 2)
        return 4.0 * np.pi * numerator / squared

    def _slab_erf_planes(self, transverse, xp):
        """``h_z sum_s [v(Q_par, z_s) - 2 pi/Q_par] e^{-i G_z z_s}`` (Hartree) of ``erf(omega r)/r``, ``Q_par > 0``.

        Shape ``transverse.shape + (P,)``; the constant ``2 pi/Q_par`` contributes ``2 pi P h_z/Q_par`` at
        ``G_z = 0`` only, which the caller adds.
        """

        separations = xp.asarray(self.separations)
        residual = erf_sheet_residual(transverse[..., None], separations, self.omega, xp)
        return self.plane_spacing * xp.fft.fft(residual, axis=-1).real

    def _slab_erf_planes_at_zero(self, xp):
        """The same at ``Q_par = 0``: the transform of the finite part ``erf_sheet_limit`` (Hartree), shape ``(P,)``."""

        separations = xp.asarray(self.separations)
        return self.plane_spacing * xp.fft.fft(erf_sheet_limit(separations, self.omega, xp)).real

    # -- the element at q + G = 0 ------------------------------------------

    def _mesh_weight(self, coordinates):
        """Extrapolation weights of points with fractional coordinates ``coordinates`` (list over the periodic axes)."""

        on_double = None
        for axis, values in zip(self.sum_axes, coordinates):
            scaled = 0.5 * values * self.k_mesh[axis]
            flag = np.abs(scaled - np.rint(scaled)) <= _FRACTION_TOLERANCE
            on_double = flag if on_double is None else (on_double & flag)
        return np.where(on_double, 0.0, self.extrapolation_factor)

    def _auxiliary_sum(self, name: str, weighted: bool) -> float:
        """``(1/N_k) sum'_{q, G} w F(q + G)`` (Hartree) of a component, over the reciprocal lattice of the supercell."""

        a = self.auxiliary_exponent
        limit = math.sqrt(_AUXILIARY_CUTOFF / a)
        axes = self.sum_axes
        bounds = [
            int(math.ceil(limit * np.linalg.norm(self.fft_lattice[axis]) / (2.0 * np.pi) * self.k_mesh[axis])) + 1
            for axis in axes
        ]
        ranges = [np.arange(-bound, bound + 1) / self.k_mesh[axis] for bound, axis in zip(bounds, axes)]
        metric = self.metric[: len(axes), : len(axes)]
        total = 0.0
        first = ranges[0]
        per_slice = max(1, int(np.prod([r.size for r in ranges[1:]])))
        chunk = max(1, _SUM_CHUNK // per_slice)
        for start in range(0, first.size, chunk):
            coordinates = [first[start : start + chunk]] + ranges[1:]
            grids = np.meshgrid(*coordinates, indexing="ij")
            squared = sum(metric[i, j] * grids[i] * grids[j] for i in range(len(grids)) for j in range(len(grids)))
            inside = (squared > self.singular_threshold) & (a * squared < _AUXILIARY_CUTOFF)
            if not np.any(inside):
                continue
            terms = self._auxiliary_function(name, squared[inside])
            if weighted:
                terms = terms * self._mesh_weight([grid[inside] for grid in grids])
            total += float(np.sum(terms))
        return total / self.k_points

    def _auxiliary_function(self, name: str, squared: np.ndarray) -> np.ndarray:
        """``F(Q)`` (Hartree) of a component at ``Q^2 > 0`` (host).

        Bulk and the erfc component of a slab: ``K(Q) exp(-a Q^2)``.  The ``1/r`` component of a slab:
        ``(2 pi H/Q) erfc(sqrt(a) Q)`` (``H = P h_z``), the singular part of ``K(Q, 0)`` times an analytic function of
        ``Q^2`` only, so that the remainder keeps no cusp of its own (``K(Q, 0) exp(-a Q^2)`` would: the truncated
        kernel carries a term ``(2 pi/3) z_c^3 Q`` that only the ``G_z != 0`` elements compensate).
        """

        a = self.auxiliary_exponent
        if not self.truncated:
            return self._bulk_component(name, squared, np) * np.exp(-a * squared)
        transverse = np.sqrt(squared)
        from scipy.special import erf, erfc

        if name == "coulomb":
            return (2.0 * np.pi * self.padded_height / transverse) * erfc(math.sqrt(a) * transverse)
        sheet = (2.0 * np.pi / transverse) * erf(transverse / (2.0 * self.omega))
        return self.padded_height * sheet * np.exp(-a * squared)

    def _origin_value(self, name: str) -> float:
        """``lim_{Q->0} [K(Q) - F(Q)]`` at ``G_z = 0`` (Hartree): the regular remainder at the origin per ``S_0``.

        Bulk ``1/r``: ``4 pi a`` (``S`` taken flat at the origin); bulk erfc: zero.  Slab, exact (the finite parts
        of the kernel and of ``F``): ``1/r`` ``-2 pi z_c^2 + 4 H sqrt(pi a)``; erfc ``K(0, 0) - 2 sqrt(pi) H/omega``.
        """

        a = self.auxiliary_exponent
        if not self.truncated:
            return 4.0 * np.pi * a if name == "coulomb" else 0.0
        if name == "coulomb":
            return -2.0 * np.pi * self.cutoff_height**2 + 4.0 * self.padded_height * math.sqrt(math.pi * a)
        return self._finite_limit(name) - 2.0 * math.sqrt(math.pi) * self.padded_height / self.omega

    def _auxiliary_integral(self, name: str) -> float:
        """``(Omega/(2 pi)^3) int F d^3Q`` (bulk) or ``(A/(2 pi)^2) int F d^2Q`` (slab) of a component, Hartree."""

        a = self.auxiliary_exponent
        if not self.truncated:
            if name == "coulomb":
                return self.volume / math.sqrt(math.pi * a)
            return self.volume / math.sqrt(math.pi) * (
                1.0 / math.sqrt(a) - 1.0 / math.sqrt(a + 1.0 / (4.0 * self.omega**2))
            )
        if name == "coulomb":
            return self.area * self.padded_height / math.sqrt(math.pi * a)
        # (A/(2 pi)) int_0^inf 2 pi H erf(Q/(2 w)) e^{-a Q^2} dQ = A H arctan(1/(2 w sqrt(a)))/sqrt(pi a)
        return (
            self.area
            * self.padded_height
            * math.atan(1.0 / (2.0 * self.omega * math.sqrt(a)))
            / math.sqrt(math.pi * a)
        )

    def _finite_limit(self, name: str) -> float:
        """Finite part (Hartree) of a component at ``q + G -> 0``, its singular term dropped."""

        if not self.truncated:
            return 0.0 if name == "coulomb" else np.pi / self.omega**2
        zc = self.cutoff_height
        coulomb = -2.0 * np.pi * zc * zc
        if name == "coulomb":
            return coulomb
        sheets = self.plane_spacing * float(np.sum(erf_sheet_limit(self.separations, self.omega)))
        return coulomb - sheets

    def component_element(self, name: str, weighted: bool) -> float:
        """``K_0`` (Hartree) of one component (``"coulomb"`` or ``"erfc"``) with or without the extrapolation."""

        if self.divergence == "none":
            return 0.0 if weighted else self._finite_limit(name)
        if not self.truncated:
            return self._ewald_element(name, weighted)
        limit = 0.0 if weighted else self._origin_value(name)
        return limit + self.k_points * (self._auxiliary_integral(name) - self._auxiliary_sum(name, weighted))

    def _ewald_element(self, name: str, weighted: bool) -> float:
        """``K_0`` (Hartree) of a 3-D component by Ewald's identity (module docstring), any auxiliary exponent.

        ``L`` the Born-von Karman supercell (rows ``N_i a_i``, volume ``Omega_sc = N_k Omega``), ``L/2`` the lattice
        whose reciprocal is the double grid (rows ``N_i a_i/2``, volume ``Omega_sc/8``), ``f = 8/7``, ``b = 1/(4 w^2)``:
        ``1/r``: ``-Omega_sc M_L(a)``, extrapolated ``-f [Omega_sc M_L(a) - Omega_sc/8 M_L/2(a)]``;
        erfc: ``4 pi b - Omega_sc [M_L(a) - M_L(a + b)]``, extrapolated
        ``-f {Omega_sc [M_L(a) - M_L(a + b)] - Omega_sc/8 [M_L/2(a) - M_L/2(a + b)]}``.
        The divergent ``Omega/sqrt(pi a)`` of the sums and the integral cancel in closed form.
        """

        a = self.auxiliary_exponent
        supercell = np.asarray(self.k_mesh, dtype=np.float64)[:, None] * self.fft_lattice
        volume = abs(float(np.linalg.det(supercell)))
        exponents = [a] if name == "coulomb" else [a, a + 1.0 / (4.0 * self.omega**2)]
        remainder = _madelung_remainder(supercell, exponents)
        if name == "coulomb":
            full = volume * remainder[0]
        else:
            full = volume * (remainder[0] - remainder[1])
        if not weighted:
            return -full if name == "coulomb" else np.pi / self.omega**2 - full
        half = _madelung_remainder(0.5 * supercell, exponents)
        coarse = (volume / 8.0) * (half[0] if name == "coulomb" else half[0] - half[1])
        return -self.extrapolation_factor * (full - coarse)

    def _divergence_element(self) -> float:
        """``K_0`` (Rydberg), the element of :meth:`values` at ``q + G = 0`` (module docstring)."""

        return 2.0 * sum(
            coefficient * self.component_element(name, weighted)
            for name, coefficient, weighted in self._components()
        )

    # -- values --------------------------------------------------------------

    def values(self, q, xp=np):
        """``K(q + G)`` (Rydberg) on :attr:`fft_shape`, the element at ``q + G = 0`` regularized (module docstring).

        ``q`` is Cartesian (bohr^-1), a difference of two points of the mesh ``k_mesh`` (any reciprocal lattice
        vector added).  ``xp`` is the array module of the result (numpy or CuPy).
        """

        fraction = self.mesh_fraction(q)
        frequencies = [_fft_frequencies(n) for n in self.fft_shape]
        nyquist = [axis for axis in self.sum_axes if self.fft_shape[axis] % 2 == 0]
        block = self._slab_block if self.truncated else self._bulk_block
        result = block(fraction, frequencies, xp)
        if nyquist:
            for size in range(1, len(nyquist) + 1):
                for subset in itertools.combinations(nyquist, size):
                    shifted = list(frequencies)
                    index: list = [slice(None)] * 3
                    for axis in subset:
                        n = self.fft_shape[axis]
                        shifted[axis] = np.asarray([n // 2])  # the alias +n/2 of the index -n/2
                        index[axis] = slice(n // 2, n // 2 + 1)
                    result[tuple(index)] += block(fraction, shifted, xp)
            for axis in nyquist:
                index = [slice(None)] * 3
                index[axis] = self.fft_shape[axis] // 2
                result[tuple(index)] *= 0.5
        return result

    def _weights(self, coordinates, xp):
        """Extrapolation weights on the product grid of ``coordinates`` (per axis, ``xp`` arrays), shape ``(.., .., ..)``."""

        on_double = None
        for axis in self.sum_axes:
            scaled = 0.5 * coordinates[axis] * self.k_mesh[axis]
            flag = xp.abs(scaled - xp.rint(scaled)) <= _FRACTION_TOLERANCE
            shape = [1, 1, 1]
            shape[axis] = -1
            flag = flag.reshape(shape)
            on_double = flag if on_double is None else (on_double & flag)
        return xp.where(on_double, 0.0, self.extrapolation_factor)

    def _bulk_block(self, fraction, frequencies, xp):
        """Rydberg kernel on the product of the frequency lists (``q + G`` with ``G = sum m_i b_i``)."""

        coordinates = [xp.asarray(fraction[axis] + frequencies[axis].astype(np.float64)) for axis in range(3)]
        c0 = coordinates[0][:, None, None]
        c1 = coordinates[1][None, :, None]
        c2 = coordinates[2][None, None, :]
        g = self.metric
        squared = (
            g[0, 0] * c0 * c0
            + g[1, 1] * c1 * c1
            + g[2, 2] * c2 * c2
            + 2.0 * (g[0, 1] * c0 * c1 + g[0, 2] * c0 * c2 + g[1, 2] * c1 * c2)
        )
        singular = squared <= self.singular_threshold
        safe = xp.where(singular, 1.0, squared)
        weight = self._weights(coordinates, xp) if any(self.weighted) else None
        kernel = 2.0 * self._bulk_kernel(safe, weight, xp)
        return xp.where(singular, self.divergence_element, kernel)

    def _slab_block(self, fraction, frequencies, xp):
        """Rydberg slab kernel on the product of the frequency lists (in-plane ``q + G_par``, ``G_z``)."""

        coordinates = [xp.asarray(fraction[axis] + frequencies[axis].astype(np.float64)) for axis in range(2)]
        c0 = coordinates[0][:, None]
        c1 = coordinates[1][None, :]
        g = self.metric
        transverse_squared = xp.maximum(g[0, 0] * c0 * c0 + g[1, 1] * c1 * c1 + 2.0 * g[0, 1] * c0 * c1, 0.0)
        at_zero = transverse_squared <= self.singular_threshold
        transverse = xp.sqrt(xp.where(at_zero, 1.0, transverse_squared))
        gz = xp.asarray(frequencies[2].astype(np.float64) * (2.0 * np.pi / self.padded_height))
        coulomb = self._slab_coulomb(transverse[:, :, None], gz[None, None, :], xp)
        # Q_par = 0: (4 pi/G_z^2)(1 - cos(G_z z_c)); G_z = 0 is the singular element, set below.
        gz_safe = xp.where(gz == 0.0, 1.0, gz)
        coulomb_zero = 4.0 * np.pi * (2.0 * xp.sin(0.5 * gz * self.cutoff_height) ** 2) / (gz_safe * gz_safe)
        coulomb = xp.where(at_zero[:, :, None], coulomb_zero[None, None, :], coulomb)
        weight = None
        if any(self.weighted):
            weight = self._weights(coordinates + [xp.zeros(1)], xp)[:, :, :1]
        coefficient = self.alpha + self.beta
        if weight is None:
            kernel = coefficient * coulomb
        else:
            long_range = self.alpha * weight if self.weighted[0] else self.alpha
            short_range = self.beta * weight if self.weighted[1] else self.beta
            kernel = (long_range + short_range) * coulomb
        if self.beta != 0.0:
            if frequencies[2].size != self.fft_shape[2]:
                raise ValueError("the slab kernel is evaluated on all planes of the padded box")
            planes = self._slab_erf_planes(transverse, xp)
            planes[:, :, 0] += (2.0 * np.pi * self.padded_height) / transverse
            planes = xp.where(at_zero[:, :, None], self._slab_erf_planes_at_zero(xp)[None, None, :], planes)
            short_range = self.beta * weight if (weight is not None and self.weighted[1]) else self.beta
            kernel = kernel - short_range * planes
        singular = at_zero[:, :, None] & (gz == 0.0)[None, None, :]
        return xp.where(singular, self.divergence_element, 2.0 * kernel)


__all__ = [
    "DIVERGENCE_TREATMENTS",
    "PeriodicExchangeKernel",
    "SLAB_COULOMB_CHOICES",
    "X_GAMMA_CHOICES",
    "erf_sheet_limit",
    "erf_sheet_residual",
]
