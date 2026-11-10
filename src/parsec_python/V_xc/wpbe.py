"""Short-range omega-PBE exchange of the HSE functionals (Ernzerhof-Perdew hole, Heyd's integration).

The exchange energy per volume of an unpolarized density ``n`` with the
interaction ``erfc(omega r)/r`` is

``e_x^SR = -(3/4)(3/pi)**(1/3) n**(4/3) F(s, nu)``,
``s = |grad n| / (2 k_F n)``, ``nu = omega / k_F``, ``k_F = (3 pi**2 n)**(1/3)``,

and spin channels follow from the exact spin scaling (``k_F`` of ``2 n_s``).
The enhancement factor is the integral of the Ernzerhof-Perdew model of the
PBE exchange hole (M. Ernzerhof, J. P. Perdew, J. Chem. Phys. 109, 3313
(1998)) with the screened interaction,

``F = -(8/9) int_0^inf y J(s, y) erfc(nu y) dy``,
``J = [-A/(y**2 (1 + (4/9) A y**2)) + (A/y**2 + B + C (1 + s**2 F(s)) y**2
       + E (1 + s**2 G(s)) y**4) exp(-D y**2)] exp(-s**2 H(s) y**2)``,

done as Heyd, Scuseria and Ernzerhof did it (J. Chem. Phys. 118, 8207 (2003);
erratum 124, 219906 (2006); J. Heyd's thesis): the Gaussian terms in closed
form (``term2 .. term5`` below), the terms with ``A`` by replacing ``erfc(x)``
by the fit ``(1 + sum_k ea_k x**k) exp(-eb1 x**2)`` and integrating
term by term, which gives ``exp(x) erfc(sqrt x)``, ``exp(x) E_1(x)`` and
rational terms (``term1 = t1 + t2t9 + t10``).  For ``nu > 14`` the fit is
the Gaussian ``exp(-2 x**2)`` (``term1_largew``).  ``G(s)`` (as ``E G``) is
fixed by the normalization of the hole; below ``s = 0.08`` it is the
polynomial ``EGa1 + EGa2 s**2 + EGa3 s**4`` of Heyd's code.
``H(s) = (Ha1 s**2 + Ha2 s**4)/(1 + Ha3 s**4 + Ha4 s**5 + Ha5 s**6)`` and
``F(s) = Fc1 H(s) + Fc2`` are those of HSE.  ``s`` is rescaled above 8.3,
``s -> s_max - s_const/s**2`` (Heyd and Scuseria, J. Chem. Phys. 120, 7274
(2004), Eq. 16), so that the hole stays normalizable.

This is the definition of Quantum ESPRESSO (``pbexsr``/
``wpbe_analy_erfc_approx_grad``) and VASP (``EXCHWPBE_R``), which follow
Heyd's code, with all their constants.  libxc's ``GGA_X_WPBEH`` follows the
same code but rescales ``s`` with Henderson, Izmaylov, Scalmani and Scuseria
(J. Chem. Phys. 131, 044108 (2009), Eq. 8) from ``s = 1``; libxc 7 moreover
derives ``E``, ``Fc1``, ``Fc2`` and the small-``s`` ``EG`` from ``A .. D``
instead of using the published digits (libxc 6 has the published ones).
``rescaling="henderson"`` selects libxc's rescaling (used to check this
module against libxc 6.2.2).

The derivatives ``dF/ds`` and ``dF/dnu`` are carried along by forward-mode
differentiation (:class:`_Dual`): each operation propagates its exact
derivative, so ``vrho`` and ``vsigma`` are the derivatives of the evaluated
energy to round-off.  ``exp(x) E_1(x)`` is evaluated by its power series
(``x <= 1``) and its continued fraction (``x > 1``, depth 80) to 1e-15, the
same algorithm as the CUDA kernel.  Units: Hartree, bohr.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfcx

# Ernzerhof-Perdew hole (EP98) and the HSE fits, with the digits of Heyd's code.
_A = 1.0161144
_B = -0.37170836
_C = -0.077215461
_D = 0.57786348
_E = -0.051955731
_HA1 = 0.00979681
_HA2 = 0.0410834
_HA3 = 0.187440
_HA4 = 0.00120824
_HA5 = 0.0347188
_FC1 = 6.4753871
_FC2 = 0.47965830
_EGA1 = -0.02628417880
_EGA2 = -0.07117647788
_EGA3 = 0.08534541323
_EG_CUTOFF = 0.08
_NU_CUTOFF = 14.0
# Heyd's fit erfc(x) ~ (1 + sum_k ea_k x**k) exp(-eb1 x**2).
_EA = (
    -1.128223946706117,
    1.452736265762971,
    -1.243162299390327,
    0.971824836115601,
    -0.568861079687373,
    0.246880514820192,
    -0.065032363850763,
    0.008401793031216,
)
_EB1 = 1.455915450052607
# Rescaling of s (Heyd and Scuseria 2004, Eq. 16): continuous at s = 8.3.
_S_TRANSITION = 8.3
_S_MAX = 8.572844
_S_CONST = 18.79622316
# Henderson et al. 2009, Eq. 8 (libxc's s_scaling_2).
_S_HENDERSON_START = 1.0
_S_HENDERSON_END = 15.0
_S_FLOOR = 1.0e-15
_SLATER_CX = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
_KF_COEFFICIENT = (3.0 * np.pi * np.pi) ** (1.0 / 3.0)
_SQRT_PI = np.sqrt(np.pi)
_EULER = 0.57721566490153286060651209
_E1_SERIES_TERMS = 30
_E1_FRACTION_DEPTH = 80
# Below this s the s-derivative is taken as dF/ds / s at this s (relative change below 1e-16).
_S_DERIVATIVE_FLOOR = 1.0e-8


def scaled_exp1(x: np.ndarray) -> np.ndarray:
    """``exp(x) E_1(x)`` for ``x > 0``: power series for ``x <= 1``, continued fraction (depth 80) above."""

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    small = x <= 1.0
    if np.any(small):
        xs = x[small]
        total = np.zeros_like(xs)
        term = np.ones_like(xs)
        for k in range(1, _E1_SERIES_TERMS + 1):
            term = term * (-xs) / k
            total = total + term / k
        out[small] = np.exp(xs) * (-_EULER - np.log(xs) - total)
    if np.any(~small):
        xl = x[~small]
        f = xl + (2 * _E1_FRACTION_DEPTH + 1)
        for k in range(_E1_FRACTION_DEPTH, 0, -1):
            f = xl + (2 * k - 1) - (k * k) / f
        out[~small] = 1.0 / f
    return out


class _Dual:
    """``value`` with its derivatives with respect to ``s`` and ``nu`` (forward mode, arrays)."""

    __slots__ = ("v", "s", "w")

    def __init__(self, v, s, w):
        self.v = v
        self.s = s
        self.w = w

    @staticmethod
    def _wrap(other):
        if isinstance(other, _Dual):
            return other
        return _Dual(other, 0.0, 0.0)

    def __add__(self, other):
        other = self._wrap(other)
        return _Dual(self.v + other.v, self.s + other.s, self.w + other.w)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._wrap(other)
        return _Dual(self.v - other.v, self.s - other.s, self.w - other.w)

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __neg__(self):
        return _Dual(-self.v, -self.s, -self.w)

    def __mul__(self, other):
        if not isinstance(other, _Dual):
            return _Dual(self.v * other, self.s * other, self.w * other)
        return _Dual(self.v * other.v, self.s * other.v + self.v * other.s, self.w * other.v + self.v * other.w)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, _Dual):
            return _Dual(self.v / other, self.s / other, self.w / other)
        q = self.v / other.v
        return _Dual(q, (self.s - q * other.s) / other.v, (self.w - q * other.w) / other.v)

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    def chain(self, value, derivative):
        """``f(self)`` with ``f(v) = value`` and ``f'(v) = derivative``."""

        return _Dual(value, derivative * self.s, derivative * self.w)

    def __pow__(self, exponent: float):
        value = self.v**exponent
        return self.chain(value, exponent * value / self.v)


def _sqrt(x: _Dual) -> _Dual:
    root = np.sqrt(x.v)
    return x.chain(root, 0.5 / root)


def _log(x: _Dual) -> _Dual:
    return x.chain(np.log(x.v), 1.0 / x.v)


def _erfcx(x: _Dual) -> _Dual:
    value = erfcx(x.v)
    return x.chain(value, 2.0 * x.v * value - 2.0 / _SQRT_PI)


def _scaled_exp1(x: _Dual) -> _Dual:
    value = scaled_exp1(x.v)
    return x.chain(value, value - 1.0 / x.v)


def _enhancement(s: _Dual, w: _Dual, large_nu: bool, small_s: bool) -> _Dual:
    """``F(s, nu)`` of the module docstring on points of one branch (``nu > 14``, ``s <= 0.08``)."""

    s2 = s * s
    s4 = s2 * s2
    h = (_HA1 * s2 + _HA2 * s4) / (1.0 + _HA3 * s4 + _HA4 * s4 * s + _HA5 * s4 * s2)
    f = _FC1 * h + _FC2
    eb1 = 2.0 if large_nu else _EB1
    w2 = w * w
    s2h = s2 * h
    aux1 = _D + s2h
    aux2 = (9.0 / (4.0 * _A)) * s2h
    aux3 = aux1 + w2
    aux4 = s2h + eb1 * w2
    aux5 = (9.0 / (4.0 * _A)) * aux4
    aux6 = _D + aux4

    if small_s:
        eg = _EGA1 + _EGA2 * s2 + _EGA3 * s4
    else:
        aux1_72 = aux1**3.5
        ga = _SQRT_PI * (
            15.0 * _E + 6.0 * _C * (1.0 + f * s2) * aux1 + 4.0 * _B * aux1 * aux1 + 8.0 * _A * aux1 * aux1 * aux1
        ) / (16.0 * aux1_72) - (3.0 * np.pi / 4.0) * np.sqrt(_A) * _erfcx(_sqrt(aux2))
        gb = (15.0 * _SQRT_PI) * s2 / (16.0 * aux1_72)
        eg = -((3.0 * np.pi / 4.0) + ga) / gb

    csf = _C * s2 * f
    s2eg = s2 * eg
    term2 = (aux1 * aux1 * _B + aux1 * _C + 2.0 * _E + aux1 * csf + 2.0 * s2eg) / (2.0 * aux1 * aux1 * aux1)
    aux3_52 = aux3**2.5
    term3 = -w * (4.0 * aux3 * aux3 * _B + 6.0 * aux3 * _C + 15.0 * _E + 6.0 * aux3 * csf + 15.0 * s2eg) / (
        8.0 * aux1 * aux3_52
    )
    w3 = w2 * w
    term4 = -w3 * (aux3 * _C + 5.0 * _E + aux3 * csf + 5.0 * s2eg) / (2.0 * aux1 * aux1 * aux3_52)
    w5 = w3 * w2
    term5 = -w5 * (_E + s2eg) / (aux1 * aux1 * aux1 * aux3_52)

    if large_nu:
        term1 = -0.5 * _A * (-_scaled_exp1(aux5) + _log(aux6) - _log(aux4))
    else:
        ea1, ea2, ea3, ea4, ea5, ea6, ea7, ea8 = _EA
        a12 = np.sqrt(_A)
        w4 = w2 * w2
        w6 = w4 * w2
        w7 = w6 * w
        w8 = w4 * w4
        np1 = (
            -1.5 * ea1 * a12 * w
            + (27.0 * ea3 / (8.0 * a12)) * w3
            - (243.0 * ea5 / (32.0 * _A**1.5)) * w5
            + (2187.0 * ea7 / (128.0 * _A**2.5)) * w7
        )
        np2 = (
            -_A
            + (9.0 * ea2 / 4.0) * w2
            - (81.0 * ea4 / (16.0 * _A)) * w4
            + (729.0 * ea6 / (64.0 * _A**2)) * w6
            - (6561.0 * ea8 / (256.0 * _A**3)) * w8
        )
        t1 = 0.5 * (np1 * np.pi * _erfcx(_sqrt(aux5)) - np2 * _scaled_exp1(aux5))
        sqrt_aux4 = _sqrt(aux4)
        sqrt_aux6 = _sqrt(aux6)
        aux4_32 = aux4 * sqrt_aux4
        aux4_52 = aux4_32 * aux4
        aux6_32 = aux6 * sqrt_aux6
        aux6_52 = aux6_32 * aux6
        aux6_72 = aux6_52 * aux6
        aux6_2 = aux6 * aux6
        aux6_4 = aux6_2 * aux6_2
        aux4_2 = aux4 * aux4
        f2 = (0.5 * ea1 * _SQRT_PI * _A) / sqrt_aux6
        f3 = (0.5 * ea2 * _A) / aux6
        f4 = ea3 * _SQRT_PI * (-9.0 / (8.0 * sqrt_aux4) + 0.25 * _A / aux6_32)
        f5 = (ea4 / 128.0) * (-144.0 / aux4 + 64.0 * _A / aux6_2)
        f6 = ea5 * (3.0 * _SQRT_PI * (3.0 * aux6_52 * (9.0 * aux4 - 2.0 * _A) + 4.0 * aux4_32 * _A * _A)) / (
            32.0 * aux6_52 * aux4_32 * _A
        )
        f7 = ea6 * (32.0 * _A / (aux6_2 * aux6) + (-36.0 + (81.0 / _A) * s2h) / aux4_2) / 32.0
        f8 = ea7 * (
            -3.0 * _SQRT_PI * (-40.0 * aux4_52 * _A**3 + 9.0 * aux6_72 * (27.0 * aux4_2 - 6.0 * aux4 * _A + 4.0 * _A * _A))
        ) / (128.0 * aux6_72 * aux4_52 * _A * _A)
        f9 = (
            324.0 * ea6 * eb1 * aux6_4 * aux4 * _A
            + ea8 * (384.0 * aux4_2 * aux4 * _A**3 + aux6_4 * (-729.0 * aux4_2 + 324.0 * aux4 * _A - 288.0 * _A * _A))
        ) / (128.0 * aux6_4 * aux4_2 * aux4 * _A * _A)
        t2t9 = f2 * w + f3 * w2 + f4 * w3 + f5 * w4 + f6 * w5 + f7 * w6 + f8 * w7 + f9 * w8
        t10 = 0.5 * _A * _log(aux4 / aux6)
        term1 = t1 + t2t9 + t10

    return (-8.0 / 9.0) * (term1 + term2 + term3 + term4 + term5)


def rescaled_s(s: np.ndarray, rescaling: str = "heyd") -> tuple[np.ndarray, np.ndarray]:
    """The reduced gradient the hole is evaluated at and ``d s_eff/d s``."""

    s = np.asarray(s, dtype=np.float64)
    if rescaling == "heyd":
        large = s > _S_TRANSITION
        safe = np.where(large, s, 1.0)
        effective = np.where(large, _S_MAX - _S_CONST / (safe * safe), s)
        derivative = np.where(large, 2.0 * _S_CONST / (safe * safe * safe), 1.0)
    elif rescaling == "henderson":
        clipped = np.clip(s, _S_HENDERSON_START, _S_HENDERSON_END)
        tail = np.exp(clipped - _S_MAX)
        effective = np.where(
            s < _S_HENDERSON_START, s, np.where(s > _S_HENDERSON_END, _S_MAX, clipped - np.log1p(tail))
        )
        derivative = np.where(
            s < _S_HENDERSON_START, 1.0, np.where(s > _S_HENDERSON_END, 0.0, 1.0 - tail / (1.0 + tail))
        )
    else:
        raise ValueError("rescaling must be 'heyd' or 'henderson'")
    floored = effective < _S_FLOOR
    return np.where(floored, _S_FLOOR, effective), np.where(floored, 0.0, derivative)


def wpbe_enhancement(s, nu, *, rescaling: str = "heyd"):
    """``F(s, nu)``, ``dF/ds`` and ``dF/dnu`` (``s`` before rescaling, ``nu = omega/k_F > 0``).

    ``dF/ds`` includes ``d s_eff/d s``.  Also returns ``dF/ds / s``, taken at
    ``max(s, 1e-8)`` for the derivative with respect to ``sigma`` (``s**2``
    is proportional to ``sigma``).
    """

    s = np.asarray(s, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    effective, slope = rescaled_s(s, rescaling)
    value = np.empty_like(s)
    d_s = np.empty_like(s)
    d_nu = np.empty_like(s)
    for large_nu in (False, True):
        for small_s in (False, True):
            mask = ((nu > _NU_CUTOFF) == large_nu) & ((effective <= _EG_CUTOFF) == small_s)
            if not np.any(mask):
                continue
            ones = np.ones(int(np.count_nonzero(mask)))
            zeros = np.zeros_like(ones)
            result = _enhancement(
                _Dual(effective[mask], ones, zeros), _Dual(nu[mask], zeros, ones), large_nu, small_s
            )
            value[mask] = result.v
            d_s[mask] = result.s * slope[mask]
            d_nu[mask] = result.w
    # dF/ds / s: below 1e-8 the s-derivative at 1e-8 (dF/ds is linear in s there).
    tiny = s < _S_DERIVATIVE_FLOOR
    d_s_over_s = np.empty_like(s)
    d_s_over_s[~tiny] = d_s[~tiny] / s[~tiny]
    if np.any(tiny):
        floor = np.full(int(np.count_nonzero(tiny)), _S_DERIVATIVE_FLOOR)
        ones = np.ones_like(floor)
        zeros = np.zeros_like(floor)
        for large_nu in (False, True):
            sub = (nu[tiny] > _NU_CUTOFF) == large_nu
            if np.any(sub):
                result = _enhancement(
                    _Dual(floor[sub], ones[sub], zeros[sub]), _Dual(nu[tiny][sub], zeros[sub], ones[sub]), large_nu, True
                )
                values = d_s_over_s[tiny]
                values[sub] = result.s / _S_DERIVATIVE_FLOOR
                d_s_over_s[tiny] = values
    return value, d_s, d_nu, d_s_over_s


def wpbe_sr_exchange_partials(n, sigma, omega, *, rescaling: str = "heyd"):
    """Short-range omega-PBE exchange ``(e, de/dn, de/dsigma)`` of a positive unpolarized density (Hartree).

    ``e = -C_x n**(4/3) F(s, nu)`` with ``ds/dn = -(4/3) s/n``,
    ``dnu/dn = -nu/(3n)`` and ``ds/dsigma = s/(2 sigma)``, so
    ``de/dsigma = -C_x n**(4/3) (F_s/s) / (8 k_F**2 n**2)``.
    """

    n = np.asarray(n, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n13 = np.cbrt(n)
    kf = _KF_COEFFICIENT * n13
    s = np.sqrt(sigma) / (2.0 * kf * n)
    nu = float(omega) / kf
    value, d_s, d_nu, d_s_over_s = wpbe_enhancement(s, nu, rescaling=rescaling)
    prefactor = -_SLATER_CX * n * n13
    energy = prefactor * value
    d_n = (4.0 / 3.0) * (-_SLATER_CX * n13) * value + prefactor * (
        d_s * (-(4.0 / 3.0) * s / n) + d_nu * (-nu / (3.0 * n))
    )
    d_sigma = prefactor * d_s_over_s / (8.0 * kf * kf * n * n)
    return energy, d_n, d_sigma


__all__ = [
    "rescaled_s",
    "scaled_exp1",
    "wpbe_enhancement",
    "wpbe_sr_exchange_partials",
]
