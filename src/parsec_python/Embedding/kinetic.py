"""Non-additive kinetic energy and potential of frozen-density embedding.

PARSEC 1.5 ``potential/kinetic.f90`` (Dingxin Fan, 2026): for the density
``rho_tot = rho_B + rho_core + rho_A`` of the active tip ``B`` (with its core
density) and the frozen sample ``A``,

``T_nad = T[rho_tot] - T[rho_A] - T[rho_tot - rho_A]``,
``v_nad = dT_nad/drho_B = v_T[rho_tot] - v_T[rho_tot - rho_A]``

(``kinetic_energy``, lines 8-54).  ``T`` is one of five semilocal kinetic
functionals, each written as ``T[rho] = C_TF sum_i rho_i^(5/3) F(s_i) h^3``
with ``C_TF = (3/5)(3 pi^2)^(2/3)`` in Rydberg units, and its potential is
evaluated point by point (``compute_kin``, lines 57-177) from the centered
finite-difference gradient of ``gradmvs`` (:func:`parsec_python.V_xc.pbe.density_gradient`),
the gradient of ``|grad rho|`` and the Laplacian of the kinetic stencil
(``-(-nabla^2)``, the same coefficients as :func:`parsec_python.Laplacian.build_negative_laplacian`),
with zero density outside the cluster.  Points with ``rho < 1e-18`` give
neither energy nor potential.  With

``k_F = (3 pi^2 rho)^(1/3)``, ``s = |grad rho|/(2 k_F rho)``,
``u = grad rho . grad|grad rho| / (rho^2 (2 k_F)^3)``, ``v = lap rho/(rho (2 k_F)^2)``

the functionals (``kin_name`` of ``usrinputfile.F90:2200-2227``) are:

=========  ======================================================  =========
name       ``T``                                                    Fortran
=========  ======================================================  =========
``tf``     Thomas-Fermi, ``F = 1`` (the default)                    3
``tw``     Thomas-Fermi plus the second-order gradient expansion,   4 (tfwk)
           ``T_TF + lambda T_vW``, ``T_vW = (1/4) |grad rho|^2/rho``
           in Ry
``pw``     PW91k of Lembarki and Chermette (LC94), PW91 exchange    1 (pw91k)
           enhancement with the constants of Lacks and Gordon
``pb``     PBE-like, ``F = 1 + kappa - kappa/(1 + mu s^2/kappa)``,   2 (pbek)
           ``mu = 0.2319``, ``kappa = 0.8438`` (Tran-Wesolowski)
``none``   no kinetic correction                                    0
=========  ======================================================  =========

For a GGA ``T = C_TF integral rho^(5/3) F(s)`` the functional derivative is

``v_T = C_TF rho^(2/3) [(5/3) F - (u - (4/3) s^3) F_ss - v F_s - s^2 F_s/3]``

with ``F_s = F'(s)/s`` and ``F_ss = dF_s/ds`` (Burke's EASYPBE algebra with
the exponent 5/3), which is the formula of ``pw91k`` and ``pbek``.

Two defects of the 1.5 code are corrected here by default
(``form="corrected"``), and reproduced with ``form="parsec1.5"``
(``FDET_Kinetic_Form``) for comparisons with PARSEC 1.5:

* ``kinetic.f90:150`` sets ``u = 2 grad rho . grad|grad rho| / (rho^2
  (2 k_F)^3)``, twice the ``u`` of the derivative above (the same slip as
  PARSEC's unpolarized PW91 exchange, ``pbe.f90:64``).  The ``pw`` and ``pb``
  potentials are then not the derivatives of their energies; the energies
  themselves are unaffected.  ``tests/unit/test_fdet_kinetic.py`` measures it
  against finite differences of the energy.
* ``tfwk`` (``kinetic.f90:276-280``) adds ``(1/72) |grad rho|^2/rho``, the
  coefficient of ``T_vW/9`` in Hartree units, to an energy in Rydberg: the
  functional is ``T_TF + T_vW/18``.  The gradient expansion that the 1.5 input
  reader names ("Thomas-Fermi plus GEA") is ``T_TF + T_vW/9``, coefficient
  ``1/36`` in Rydberg, which is the default here.  Its potential is the
  derivative of its energy in both forms.

TF and ``none`` are the same in both forms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d

from ..Laplacian.finite_difference import second_derivative_coefficients
from ..V_xc.pbe import _embed_active, _gather_active, density_gradient

#: ``kinetic_energy_functional`` names of PARSEC 1.5, by ``kin_name``.
KINETIC_FUNCTIONALS = ("tf", "tw", "pw", "pb", "none")
#: The two forms of the potentials (:mod:`parsec_python.Embedding.kinetic`).
KINETIC_FORMS = ("corrected", "parsec1.5")
#: ``ctf = 0.6 (3 pi^2)^(2/3)``: Thomas-Fermi constant in Rydberg units.
THOMAS_FERMI_CONSTANT = 0.6 * (3.0 * np.pi**2) ** (2.0 / 3.0)
#: Densities below this give no kinetic energy or potential (``compute_kin``).
DENSITY_FLOOR = 1.0e-18
#: Default ``pbe_m`` and ``pbe_k`` of ``usrinputfile.F90:2209-2210``.
PBE_KINETIC_MU = 0.2319
PBE_KINETIC_KAPPA = 0.8438
#: The ``|grad rho|^2/rho`` coefficient of ``tw`` in Ry: T_vW/9 (corrected) and 1.5's 1/72.
GRADIENT_EXPANSION_COEFFICIENTS = {"corrected": 1.0 / 36.0, "parsec1.5": 1.0 / 72.0}
#: The factor of ``u`` in the GGA potentials: Burke's ``u`` (corrected) and 1.5's twice that.
U_FACTORS = {"corrected": 1.0, "parsec1.5": 2.0}

# PW91k (LC94) constants of pw91k, kinetic.f90:191-196.
_A1 = 0.093907
_A2 = 0.26608
_A3 = 0.0809615
_A4 = 100.0
_A = 76.32
_B1 = 0.000057767


@dataclass(frozen=True)
class KineticFunctional:
    """One kinetic functional with its parameters.

    ``name`` is one of :data:`KINETIC_FUNCTIONALS`, ``form`` one of
    :data:`KINETIC_FORMS`; ``mu`` and ``kappa`` are read by ``pb`` only.
    """

    name: str = "tf"
    form: str = "corrected"
    mu: float = PBE_KINETIC_MU
    kappa: float = PBE_KINETIC_KAPPA

    def __post_init__(self) -> None:
        if self.name not in KINETIC_FUNCTIONALS:
            raise ValueError(
                "the kinetic functional must be one of " + ", ".join(KINETIC_FUNCTIONALS)
            )
        if self.form not in KINETIC_FORMS:
            raise ValueError("the kinetic form must be one of " + ", ".join(KINETIC_FORMS))
        if self.name == "pb" and not (float(self.kappa) != 0.0):
            raise ValueError("pbe_k (kappa) of the PBE-like kinetic functional must not be zero")

    @property
    def uses_gradients(self) -> bool:
        return self.name in ("tw", "pw", "pb")


@dataclass(frozen=True)
class KineticTerms:
    """``T[rho]`` in Ry and ``v_T = dT/drho`` in Ry, one value per grid point."""

    energy: float
    potential: np.ndarray


@dataclass(frozen=True)
class NonadditiveKinetic:
    """``T_nad`` (Ry) and ``v_nad`` (Ry per point) of :func:`nonadditive_kinetic`."""

    energy: float
    potential: np.ndarray


def _positive_laplacian(values: np.ndarray, grid) -> np.ndarray:
    """``+nabla^2`` of a cluster field with zero outside (``-my_laplacian``).

    The axial stencil of :func:`parsec_python.Laplacian.build_negative_laplacian`
    applied to the zero-padded box, with the sign of ``+d^2/dx^2``.
    """

    coefficients = second_derivative_coefficients(grid.settings.expansion_order) / grid.spacing**2
    box = _embed_active(values, grid)
    total = np.zeros(grid.size, dtype=np.float64)
    for axis in range(3):
        total += _gather_active(correlate1d(box, coefficients, axis=axis, mode="constant", cval=0.0), grid)
    return total


def _pw91k(rho, s, u, v, *, xp=np):
    """``pw91k`` of kinetic.f90:180-226: ``e_k`` and ``v_k`` per ``C_TF`` (``xp``: the array module)."""

    fac = rho ** (2.0 / 3.0)
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    p0 = 1.0 / xp.sqrt(1.0 + _A * _A * s2)
    p1 = xp.log(_A * s + 1.0 / p0)
    p2 = xp.exp(-_A4 * s2)
    p3 = 1.0 / (1.0 + _A1 * s * p1 + _B1 * s4)
    p4 = 1.0 + _A1 * s * p1 + (_A2 - _A3 * p2) * s2
    f = p3 * p4
    ek = fac * f
    p5 = _B1 * s2 - (_A2 - _A3 * p2)
    p6 = _A1 * s * (p1 + _A * s * p0)
    p7 = 2.0 * (_A2 - _A3 * p2) + 2.0 * _A3 * _A4 * s2 * p2 - 4.0 * _B1 * s2 * f
    fs = p3 * (p3 * p5 * p6 + p7)
    p8 = 2.0 * s * (_B1 - _A3 * _A4 * p2)
    p9 = _A1 * p1 + _A * _A1 * s * p0 * (3.0 - _A * _A * s2 * p0 * p0)
    p10 = 4.0 * _A3 * _A4 * s * p2 * (2.0 - _A4 * s2) - 8.0 * _B1 * s * f - 4.0 * _B1 * s3 * fs
    p11 = -p3 * p3 * (_A1 * p1 + _A * _A1 * s * p0 + 4.0 * _B1 * s3)
    fss = p3 * p3 * (p5 * p9 + p6 * p8) + 2.0 * p3 * p5 * p6 * p11 + p3 * p10 + p7 * p11
    vk = fac * ((5.0 / 3.0) * f - (u - (4.0 / 3.0) * s3) * fss - v * fs - fs * s2 / 3.0)
    return ek, vk


def _pbek(rho, s, u, v, mu, kappa):
    """``pbek`` of kinetic.f90:229-258: ``e_k`` and ``v_k`` per ``C_TF``."""

    exunif = rho ** (2.0 / 3.0)
    ul = mu / kappa
    s2 = s * s
    p0 = 1.0 + ul * s2
    fxpbe = 1.0 + kappa - kappa / p0
    ek = exunif * fxpbe
    fs = 2.0 * kappa * ul / (p0 * p0)
    fss = -4.0 * ul * s * fs / p0
    vk = exunif * ((5.0 / 3.0) * fxpbe - (u - (4.0 / 3.0) * s2 * s) * fss - v * fs - fs * s2 / 3.0)
    return ek, vk


def _tfwk(rho, absgrad, laplacian, coefficient):
    """``tfwk`` of kinetic.f90:261-283 with the gradient coefficient ``coefficient`` (Ry)."""

    ek = rho ** (2.0 / 3.0) + (coefficient * (absgrad * absgrad) / (rho * rho)) / THOMAS_FERMI_CONSTANT
    vk = (5.0 / 3.0) * rho ** (2.0 / 3.0) + (
        coefficient * (absgrad * absgrad / (rho * rho) - 2.0 * laplacian / rho)
    ) / THOMAS_FERMI_CONSTANT
    return ek, vk


class HostKineticOperators:
    """The grid operations of :func:`kinetic_functional` on the host (NumPy, SciPy).

    ``gradient`` is :func:`parsec_python.V_xc.pbe.density_gradient` (``gradmvs``),
    ``laplacian`` the ``+nabla^2`` of the kinetic stencil, both with zero
    outside the cluster.  A device counterpart (:mod:`parsec_python.acceleration.Embedding`)
    gives the same operations on its arrays.
    """

    xp = np

    def __init__(self, grid) -> None:
        self.grid = grid

    def asarray(self, values):
        return np.asarray(values, dtype=np.float64)

    def gradient(self, values):
        return density_gradient(values, self.grid)

    def laplacian(self, values):
        return _positive_laplacian(values, self.grid)


def kinetic_functional(density: np.ndarray, grid, functional: KineticFunctional, *, operators=None) -> KineticTerms:
    """``compute_kin`` of kinetic.f90:57-177: ``T[rho]`` and its pointwise potential.

    ``density`` is a cluster field in e/bohr^3 (one value per point of
    ``grid``, zero outside the domain); the energy is
    ``C_TF h^3 sum_i rho_i e_k,i`` and the potential ``C_TF v_k,i`` (Ry).
    TF and ``none`` read no gradient.  ``operators`` holds the grid
    operations and the array module (:class:`HostKineticOperators` by
    default); the potential is an array of that module.
    """

    ops = HostKineticOperators(grid) if operators is None else operators
    xp = ops.xp
    rho = ops.asarray(density)
    if rho.shape != (grid.size,):
        raise ValueError("the density must hold one value per grid point")
    potential = xp.zeros(grid.size, dtype=np.float64)
    if functional.name == "none":
        return KineticTerms(energy=0.0, potential=potential)
    active = rho >= DENSITY_FLOOR
    if functional.name == "tf":
        r = rho[active]
        ek = r ** (2.0 / 3.0)
        vk = (5.0 / 3.0) * r ** (2.0 / 3.0)
    else:
        gradient = ops.gradient(rho)
        absgrad = xp.sqrt(xp.einsum("ij,ij->j", gradient, gradient))
        laplacian = ops.laplacian(rho)
        r = rho[active]
        if functional.name == "tw":
            ek, vk = _tfwk(
                r, absgrad[active], laplacian[active],
                GRADIENT_EXPANSION_COEFFICIENTS[functional.form],
            )
        else:
            gradgrad = ops.gradient(absgrad)
            prod = xp.einsum("ij,ij->j", gradient[:, active], gradgrad[:, active])
            fk = (3.0 * np.pi * np.pi * r) ** (1.0 / 3.0)
            s = absgrad[active] / (2.0 * fk * r)
            u = U_FACTORS[functional.form] * prod / (r * r * (2.0 * fk) ** 3)
            v = laplacian[active] / (r * (2.0 * fk) ** 2)
            if functional.name == "pw":
                ek, vk = _pw91k(r, s, u, v, xp=xp)
            else:
                ek, vk = _pbek(r, s, u, v, float(functional.mu), float(functional.kappa))
    potential[active] = THOMAS_FERMI_CONSTANT * vk
    energy = float(THOMAS_FERMI_CONSTANT * xp.sum(r * ek) * grid.volume_element)
    return KineticTerms(energy=energy, potential=potential)


def nonadditive_kinetic(
    total_density: np.ndarray,
    frozen_density: np.ndarray,
    grid,
    functional: KineticFunctional,
    *,
    frozen_energy: float | None = None,
    operators=None,
) -> NonadditiveKinetic:
    """``kinetic_energy`` of kinetic.f90:8-54.

    ``total_density`` is ``rho_B + rho_core + rho_A`` (what ``exc_nspn``
    passes: the XC input density), ``frozen_density`` is ``rho_A``.  Returns
    ``T_nad`` (Ry) and ``v_nad`` (Ry per point).  ``rho_tot - rho_A`` is formed
    as the Fortran forms it.  ``frozen_energy`` is ``T[rho_A]`` where the
    caller keeps it (it does not change during an SCF loop); it is computed
    here otherwise.  ``operators`` as in :func:`kinetic_functional`; the
    potential is then an array of their module.
    """

    ops = HostKineticOperators(grid) if operators is None else operators
    total = ops.asarray(total_density)
    frozen = ops.asarray(frozen_density)
    if functional.name == "none":
        return NonadditiveKinetic(energy=0.0, potential=ops.xp.zeros(grid.size, dtype=np.float64))
    whole = kinetic_functional(total, grid, functional, operators=ops)
    sample = (
        kinetic_functional(frozen, grid, functional, operators=ops)
        if frozen_energy is None
        else KineticTerms(energy=float(frozen_energy), potential=None)
    )
    active = kinetic_functional(total - frozen, grid, functional, operators=ops)
    return NonadditiveKinetic(
        energy=float(whole.energy - sample.energy - active.energy),
        potential=whole.potential - active.potential,
    )


__all__ = [
    "DENSITY_FLOOR",
    "HostKineticOperators",
    "GRADIENT_EXPANSION_COEFFICIENTS",
    "KINETIC_FORMS",
    "KINETIC_FUNCTIONALS",
    "KineticFunctional",
    "KineticTerms",
    "NonadditiveKinetic",
    "PBE_KINETIC_KAPPA",
    "PBE_KINETIC_MU",
    "THOMAS_FERMI_CONSTANT",
    "U_FACTORS",
    "kinetic_functional",
    "nonadditive_kinetic",
]
