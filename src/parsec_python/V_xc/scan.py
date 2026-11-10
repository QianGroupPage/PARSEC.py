"""The SCAN meta-GGA of Sun, Ruzsinszky and Perdew (PRL 115, 036402 (2015)), pointwise.

A native evaluator of the meta-GGA interface of :mod:`parsec_python.V_xc.semilocal`
(family ``"mgga"``): SCAN exchange and correlation in libxc's variables and
units (Hartree; ``tau = 1/2 sum_i f_i |grad psi_i|^2``), unpolarized and
spin-polarized, on NumPy or CuPy arrays (the code is array-module agnostic).

Exchange (unpolarized; spin by ``E_x[n_u, n_d] = (E_x[2 n_u] + E_x[2 n_d])/2``,
which scales ``sigma`` by 4 and ``tau`` by 2):

``e_x = -(3/4)(3/pi)^(1/3) n^(4/3) F_x``,
``F_x = [h1x(y) (1 - f_x(alpha)) + h0x f_x(alpha)] g_x(p)``,
``p = sigma / (4 (3 pi^2)^(2/3) n^(8/3))``,
``alpha = (tau - tau_W) / tau_unif``, ``tau_W = sigma/(8 n)``,
``tau_unif = (3/10)(3 pi^2)^(2/3) n^(5/3)``,
``y = mu p + b4 p^2 exp(-b4 p/mu) + [b1 p + b2 (1 - alpha) exp(-b3 (1 - alpha)^2)]^2``,
``h1x = 1 + k1 y/(k1 + y)``, ``g_x = 1 - exp(-a1 p^(-1/4))``, ``h0x = 1.174``,
``f(alpha) = exp(-c1 alpha/(1 - alpha))`` (alpha < 1), ``-d exp(c2/(1 - alpha))``
(alpha > 1), with ``mu = 10/81``, ``b2 = sqrt(5913/405000)``,
``b1 = (511/13500)/(2 b2)``, ``b3 = 1/2``, ``b4 = mu^2/k1 - 1606/18225 - b1^2``,
``k1 = 0.065``, ``a1 = 4.9479``, ``c1 = 0.667``, ``c2 = 0.8``, ``d = 1.24``.

Correlation: ``e_c = n [eps_1 + f_c(alpha_c) (eps_0 - eps_1)]`` with
``eps_1 = eps_PW92(rs, zeta) + gamma phi^3 ln[1 + w_1 (1 - g(A t^2))]``,
``w_1 = exp(-eps_PW92/(gamma phi^3)) - 1``, ``A = beta(rs)/(gamma w_1)``,
``g(x) = (1 + 4x)^(-1/4)``, ``beta(rs) = 0.06672455060314922 (1 + 0.1 rs)/(1 + 0.1778 rs)``,
``gamma = (1 - ln 2)/pi^2``, ``t = |grad n|/(2 phi k_s n)``;
``eps_0 = [eps_LDA0 + b1c ln(1 + w_0 (1 - g_inf(s)))] G_c(zeta)``,
``eps_LDA0 = -b1c/(1 + b2c rs^(1/2) + b3c rs)``, ``w_0 = exp(-eps_LDA0/b1c) - 1``,
``g_inf(s) = (1 + 4 chi s^2)^(-1/4)``, ``G_c = [1 - 2.363 (2^(1/3) - 1) f_zeta(zeta)] (1 - zeta^12)``;
``alpha_c = (tau - tau_W)/(tau_unif d_s(zeta))`` of the total density,
``d_s = [(1 + zeta)^(5/3) + (1 - zeta)^(5/3)]/2``, ``f_c`` the ``f`` above with
``c1 = 0.64``, ``c2 = 1.5``, ``d = 0.7``; ``b1c = 0.0285764``, ``b2c = 0.0889``,
``b3c = 0.125541``, ``chi = 0.12802585262625815``.  PW92 has the parameters of
PBE's correlation (``A = 0.0310907, 0.01554535, 0.0168869``,
``f''(0) = 1.709920934161365617563962776245``).

The numerical conventions are libxc's (7.1.2, ``mgga_x_scan``, ``mgga_c_scan``):
the constants above (``G_c`` with 2.363 as libxc, the paper prints 2.3631),
``f(alpha)`` cut to zero where its exponential falls below machine epsilon
(``alpha`` within ``|ln eps|/(|ln eps| + c1)`` and
``(|ln(eps/d)| + c2)/|ln(eps/d)|`` of one), density screening below ``1e-15``
(total density; a spin channel's exchange below it), the floors
``tau >= 1e-20`` and ``sigma_aa >= 1e-40``, ``|zeta|`` kept ``eps`` away from one.
The Fermi-hole-curvature clamp ``sigma_aa <= 8 n_a tau_a`` (``tau >= tau_W``),
libxc's default, is off by default as in Quantum ESPRESSO (``enforce_fhc``
switches it on); SCAN is defined for ``alpha < 0``.  libxc returns the partial
derivatives of the formula at a clamped point; here ``derivatives="exact"``
(the default) differentiates through the curvature clamp, so the potentials
are the exact derivatives of the energy that is returned (``vsigma = 0`` and
the ``vtau`` of the clamped ``sigma`` where it acts); the floors keep the
partials at the floor, as libxc (``_floor``: they change the energy by less
than ``1e-15`` of their variable).  ``derivatives="libxc"`` gives libxc's
partials everywhere.  The energies are the same either way.

Derivatives are taken by complex-step differentiation (exact to round-off,
no finite-difference error), one perturbed copy of the inputs per variable,
evaluated together; the step is ``1e-20`` of each value (an absolute step
would be large against ``sigma`` of the far tails).  Integer powers are
products: a complex power of a negative base (``zeta < 0``) goes through
``exp(a log z)`` on CuPy and carries an imaginary part far above the step.
"""

from __future__ import annotations

import math

import numpy as np

#: libxc's density screening of SCAN.
SCAN_DENSITY_THRESHOLD = 1.0e-15
_TAU_FLOOR = 1.0e-20
_SIGMA_FLOOR = SCAN_DENSITY_THRESHOLD ** (4.0 / 3.0) * SCAN_DENSITY_THRESHOLD ** (4.0 / 3.0)
_EPSILON = float(np.finfo(np.float64).eps)
_STEP = 1.0e-20
_STEP_FLOOR = 1.0e-200

_PI = math.pi
_MU = 10.0 / 81.0
_K1 = 0.065
_B2 = math.sqrt(5913.0 / 405000.0)
_B1 = (511.0 / 13500.0) / (2.0 * _B2)
_B3 = 0.5
_B4 = _MU * _MU / _K1 - 1606.0 / 18225.0 - _B1 * _B1
_A1 = 4.9479
_H0X = 1.174
_EXCHANGE_ALPHA = (0.667, 0.8, 1.24)
_CORRELATION_ALPHA = (0.64, 1.5, 0.7)
_CX = 0.75 * (3.0 / _PI) ** (1.0 / 3.0)
_P_DENOMINATOR = 4.0 * (3.0 * _PI * _PI) ** (2.0 / 3.0)
_TAU_UNIFORM = 0.3 * (3.0 * _PI * _PI) ** (2.0 / 3.0)

_GAMMA = (1.0 - math.log(2.0)) / (_PI * _PI)
_BETA_A, _BETA_B, _BETA_C = 0.066724550603149220, 0.1, 0.1778
_B1C, _B2C, _B3C = 0.0285764, 0.0889, 0.125541
_CHI = 0.12802585262625815
_G_CONSTANT = 2.363
# PW92 with the parameters of PBE correlation (libxc's "modified" set):
# (A, alpha1, beta1, beta2, beta3, beta4) of eps_c(zeta=0), eps_c(zeta=1), -alpha_c.
_PW92 = (
    (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
)
_FZ20 = 1.709920934161365617563962776245
_FZ_DENOMINATOR = 2.0 ** (4.0 / 3.0) - 2.0
_KS_COEFFICIENT = 4.0 * (3.0 * _PI * _PI) ** (1.0 / 3.0) / _PI  # k_s^2 = this * n^(1/3)
_S_DENOMINATOR = 2.0 * (3.0 * _PI * _PI) ** (1.0 / 3.0)


def _module(array):
    """NumPy or CuPy, whichever holds ``array``."""

    module = type(array).__module__
    if module.startswith("cupy"):
        import cupy

        return cupy
    return np


# ----- helpers analytic in complex-step arguments --------------------------------


def _expm1(xp, z):
    """``exp(z) - 1`` accurate for small real parts; first order in the imaginary step."""

    if not xp.iscomplexobj(z):
        return xp.expm1(z)
    return xp.expm1(z.real) + 1j * z.imag * xp.exp(z.real)


def _log1p(xp, z):
    """``log(1 + z)`` accurate for small real parts; first order in the imaginary step."""

    if not xp.iscomplexobj(z):
        return xp.log1p(z)
    return xp.log1p(z.real) + 1j * z.imag / (1.0 + z.real)


def _real(z):
    return z.real if hasattr(z, "real") else z


def _where(xp, condition, left, right):
    return xp.where(condition, left, right)


def _floor(xp, value, floor):
    """``max(value, floor)`` whose derivative passes through where the floor acts.

    The floors are far below any physical value (``1e-15`` of density,
    ``1e-20`` of ``tau``, ``1e-40`` of ``sigma``): there the partials at the
    floor are taken, as libxc does, so that a zero ``sigma`` (a symmetry
    point) or an empty spin channel keeps the derivative of its limit; energy
    and potential then disagree by less than the floor times that derivative.
    """

    real = _real(value)
    return _where(xp, real > floor, value, value - real + floor)


def _ceiling(xp, value, ceiling):
    """``min(value, ceiling)`` of two arrays, differentiated exactly."""

    return _where(xp, _real(value) < _real(ceiling), value, ceiling)


def _opz_pow(xp, value, exponent):
    """``(1 + zeta)^exponent`` with ``1 + zeta`` kept at least machine epsilon (libxc's opz_pow_n)."""

    base = 1.0 + value
    small = _real(base) <= _EPSILON
    safe = _where(xp, small, 1.0 + 0.0 * base, base)
    return _where(xp, small, _EPSILON**exponent + 0.0 * base, safe**exponent)


def _f_alpha(xp, alpha, parameters):
    """SCAN's interpolation ``f(alpha)`` with libxc's machine-epsilon cut-offs."""

    c1, c2, d = parameters
    log_epsilon = -math.log(_EPSILON)
    left_cutoff = log_epsilon / (log_epsilon + c1)
    log_epsilon_d = -math.log(_EPSILON / abs(d))
    right_cutoff = (log_epsilon_d + c2) / log_epsilon_d
    real = _real(alpha)
    left_argument = _where(xp, real < left_cutoff, alpha, left_cutoff + 0.0 * alpha)
    right_argument = _where(xp, real > right_cutoff, alpha, right_cutoff + 0.0 * alpha)
    left = xp.exp(-c1 * left_argument / (1.0 - left_argument))
    right = -d * xp.exp(c2 / (1.0 - right_argument))
    zero = 0.0 * alpha
    left = _where(xp, real > left_cutoff, zero, left)
    right = _where(xp, real < right_cutoff, zero, right)
    return _where(xp, real <= 1.0, left, right)


# ----- exchange --------------------------------------------------------------------


def _exchange_unpolarized(xp, n, sigma, tau):
    """SCAN exchange energy per volume of an unpolarized density (inputs already clamped)."""

    n13 = n ** (1.0 / 3.0)
    n53 = n * n13 * n13
    p = sigma / (_P_DENOMINATOR * n53 * n)
    alpha = (tau - sigma / (8.0 * n)) / (_TAU_UNIFORM * n53)
    one_minus_alpha = 1.0 - alpha
    gradient_term = _B1 * p + _B2 * one_minus_alpha * xp.exp(-_B3 * one_minus_alpha * one_minus_alpha)
    y = _MU * p + _B4 * p * p * xp.exp(-_B4 * p / _MU) + gradient_term * gradient_term
    h1x = 1.0 + _K1 * y / (_K1 + y)
    interpolation = _f_alpha(xp, alpha, _EXCHANGE_ALPHA)
    gx = -_expm1(xp, -_A1 / p**0.25)
    enhancement = (h1x * (1.0 - interpolation) + _H0X * interpolation) * gx
    return -_CX * n * n13 * enhancement


# ----- correlation -----------------------------------------------------------------


def _pw92_g(xp, rs, parameters):
    a, alpha1, beta1, beta2, beta3, beta4 = parameters
    sqrt_rs = rs**0.5
    polynomial = beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs
    return -2.0 * a * (1.0 + alpha1 * rs) * _log1p(xp, 1.0 / (2.0 * a * polynomial))


def _correlation(xp, n, zeta, sigma_total, tau_total, polarized: bool):
    """SCAN correlation energy per volume (inputs already clamped)."""

    n13 = n ** (1.0 / 3.0)
    rs = (3.0 / (4.0 * _PI)) ** (1.0 / 3.0) / n13
    g0 = _pw92_g(xp, rs, _PW92[0])
    if polarized:
        up43 = _opz_pow(xp, zeta, 4.0 / 3.0)
        down43 = _opz_pow(xp, -zeta, 4.0 / 3.0)
        f_zeta = (up43 + down43 - 2.0) / _FZ_DENOMINATOR
        # Integer powers by products: a complex power of a negative base goes
        # through exp(a log z) on CuPy, whose imaginary part of order 1e-16 of
        # the value would swamp the complex step.
        zeta2 = zeta * zeta
        zeta4 = zeta2 * zeta2
        g1 = _pw92_g(xp, rs, _PW92[1])
        g2 = _pw92_g(xp, rs, _PW92[2])
        eps_pw = g0 + zeta4 * f_zeta * (g1 - g0 + g2 / _FZ20) - f_zeta * g2 / _FZ20
        phi = 0.5 * (_opz_pow(xp, zeta, 2.0 / 3.0) + _opz_pow(xp, -zeta, 2.0 / 3.0))
        d_s = 0.5 * (_opz_pow(xp, zeta, 5.0 / 3.0) + _opz_pow(xp, -zeta, 5.0 / 3.0))
        g_c = (1.0 - _G_CONSTANT * (2.0 ** (1.0 / 3.0) - 1.0) * f_zeta) * (1.0 - zeta4 * zeta4 * zeta4)
    else:
        eps_pw = g0
        phi = 1.0
        d_s = 1.0
        g_c = 1.0
    phi3 = phi * phi * phi
    beta = _BETA_A * (1.0 + _BETA_B * rs) / (1.0 + _BETA_C * rs)
    w1 = _expm1(xp, -eps_pw / (_GAMMA * phi3))
    a_parameter = beta / (_GAMMA * w1)
    t2 = sigma_total / (4.0 * phi * phi * _KS_COEFFICIENT * n13 * n * n)
    one_minus_g = -_expm1(xp, -0.25 * _log1p(xp, 4.0 * a_parameter * t2))
    h1 = _GAMMA * phi3 * _log1p(xp, w1 * one_minus_g)
    eps_1 = eps_pw + h1

    eps_lda0 = -_B1C / (1.0 + _B2C * rs**0.5 + _B3C * rs)
    s2 = sigma_total / (_S_DENOMINATOR * _S_DENOMINATOR * n * n * n13 * n13)
    one_minus_g_infinity = -_expm1(xp, -0.25 * _log1p(xp, 4.0 * _CHI * s2))
    h0 = _B1C * _log1p(xp, _expm1(xp, -eps_lda0 / _B1C) * one_minus_g_infinity)
    eps_0 = (eps_lda0 + h0) * g_c

    n53 = n * n13 * n13
    alpha = (tau_total - sigma_total / (8.0 * n)) / (_TAU_UNIFORM * d_s * n53)
    interpolation = _f_alpha(xp, alpha, _CORRELATION_ALPHA)
    return n * (eps_1 + interpolation * (eps_0 - eps_1))


# ----- the evaluator ---------------------------------------------------------------


class NativeScan:
    """SCAN exchange and correlation as a :class:`~parsec_python.V_xc.semilocal.SemilocalXC` of family ``"mgga"``.

    ``exchange_weight`` and ``correlation_weight`` scale the two parts (1 and 1
    for SCAN).  ``enforce_fhc`` applies libxc's Fermi-hole-curvature clamp
    ``sigma_aa <= 8 n_a tau_a`` (off by default, as Quantum ESPRESSO runs
    libxc); ``derivatives`` is
    ``"exact"`` (the derivatives of the returned energy, through every clamp)
    or ``"libxc"`` (the partials at the clamped point, as libxc returns
    them).  The methods take NumPy or CuPy arrays and return the same kind.
    """

    family = "mgga"
    needs_tau = True
    needs_laplacian = False
    supports_device = True

    def __init__(
        self,
        exchange_weight: float = 1.0,
        correlation_weight: float = 1.0,
        *,
        enforce_fhc: bool = False,
        derivatives: str = "exact",
    ) -> None:
        if derivatives not in ("exact", "libxc"):
            raise ValueError("derivatives must be 'exact' or 'libxc'")
        self.exchange_weight = float(exchange_weight)
        self.correlation_weight = float(correlation_weight)
        self.enforce_fhc = bool(enforce_fhc)
        self.derivatives = derivatives

    def __repr__(self) -> str:
        return (
            f"NativeScan(exchange_weight={self.exchange_weight!r}, correlation_weight="
            f"{self.correlation_weight!r}, enforce_fhc={self.enforce_fhc!r}, derivatives={self.derivatives!r})"
        )

    # ----- clamps --------------------------------------------------------------

    def _clamped_channel(self, xp, n, sigma, tau):
        n = _floor(xp, n, SCAN_DENSITY_THRESHOLD)
        sigma = _floor(xp, sigma, _SIGMA_FLOOR)
        tau = _floor(xp, tau, _TAU_FLOOR)
        if self.enforce_fhc:
            sigma = _ceiling(xp, sigma, 8.0 * n * tau)
        return n, sigma, tau

    # ----- energies of clamped or raw stacked inputs ----------------------------

    def _unpolarized_energy(self, xp, n, sigma, tau, *, clamp: bool):
        if clamp:
            n, sigma, tau = self._clamped_channel(xp, n, sigma, tau)
        energy = 0.0 * n
        if self.exchange_weight:
            energy = energy + self.exchange_weight * _exchange_unpolarized(xp, n, sigma, tau)
        if self.correlation_weight:
            energy = energy + self.correlation_weight * _correlation(xp, n, None, sigma, tau, polarized=False)
        return energy

    def _clamped_spin(self, xp, n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d):
        n_u, s_uu, t_u = self._clamped_channel(xp, n_u, s_uu, t_u)
        n_d, s_dd, t_d = self._clamped_channel(xp, n_d, s_dd, t_d)
        average = 0.5 * (s_uu + s_dd)
        s_ud = _where(xp, _real(s_ud) >= -_real(average), s_ud, -average)
        s_ud = _where(xp, _real(s_ud) <= _real(average), s_ud, average)
        return n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d

    def _polarized_energy(self, xp, values, raw_up, raw_down, *, clamp: bool):
        if clamp:
            values = self._clamped_spin(xp, *values)
        n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d = values
        energy = 0.0 * n_u
        if self.exchange_weight:
            exchange = 0.0 * n_u
            for density, sigma, tau, raw in ((n_u, s_uu, t_u, raw_up), (n_d, s_dd, t_d, raw_down)):
                part = 0.5 * _exchange_unpolarized(xp, 2.0 * density, 4.0 * sigma, 2.0 * tau)
                exchange = exchange + _where(xp, raw > SCAN_DENSITY_THRESHOLD, part, 0.0 * part)
            energy = energy + self.exchange_weight * exchange
        if self.correlation_weight:
            n = n_u + n_d
            zeta = (n_u - n_d) / n
            limit = 1.0 - _EPSILON
            zeta = _where(xp, _real(zeta) > limit, limit + 0.0 * zeta, zeta)
            zeta = _where(xp, _real(zeta) < -limit, -limit + 0.0 * zeta, zeta)
            energy = energy + self.correlation_weight * _correlation(
                xp, n, zeta, s_uu + 2.0 * s_ud + s_dd, t_u + t_d, polarized=True
            )
        return energy

    # ----- complex-step driver ---------------------------------------------------

    def _differentiate(self, xp, inputs, energy_of):
        """Energy and partials of ``energy_of(stacked, clamp)`` at the given (active) inputs."""

        count = len(inputs)
        size = inputs[0].shape[0]
        if self.derivatives == "libxc":
            inputs = list(energy_of.clamp(xp, inputs))
            clamp = False
        else:
            clamp = True
        # A step relative to each variable: an absolute 1e-30 is no small step
        # for sigma ~ 1e-30 of a far tail, where alpha holds sigma / n^(8/3).
        steps = [_STEP * xp.maximum(xp.abs(value), _STEP_FLOOR) for value in inputs]
        stacked = []
        for index, value in enumerate(inputs):
            column = xp.empty((count, size), dtype=xp.complex128)
            column[...] = value[None, :]
            column[index] += 1j * steps[index]
            stacked.append(column)
        energy = energy_of(xp, stacked, clamp)
        values = energy.real[0].copy()
        partials = [energy.imag[index] / steps[index] for index in range(count)]
        return values, partials

    def unpolarized(self, rho, sigma, tau, lapl=None):
        """``(e, vrho, vsigma, vtau, None)`` of SCAN for the total density (Hartree)."""

        xp = _module(rho)
        rho = xp.asarray(rho, dtype=xp.float64)
        sigma = xp.asarray(sigma, dtype=xp.float64)
        tau = xp.asarray(tau, dtype=xp.float64)
        if sigma.shape != rho.shape or tau.shape != rho.shape:
            raise ValueError("SCAN inputs must share one shape")
        outputs = [xp.zeros_like(rho) for _ in range(4)]
        active = rho >= SCAN_DENSITY_THRESHOLD
        if not bool(active.any()):
            return (*outputs, None)
        selected = [rho[active], sigma[active], tau[active]]

        def energy_of(module, stacked, clamp):
            return self._unpolarized_energy(module, *stacked, clamp=clamp)

        energy_of.clamp = lambda module, values: self._clamped_channel(module, *values)
        values, partials = self._differentiate(xp, selected, energy_of)
        outputs[0][active] = values
        for output, partial in zip(outputs[1:], partials):
            output[active] = partial
        return (*outputs, None)

    def polarized(self, rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down, lapl_up=None, lapl_down=None):
        """``(e, vrho_u, vrho_d, vs_uu, vs_ud, vs_dd, vtau_u, vtau_d, None, None)`` of SCAN (Hartree)."""

        xp = _module(rho_up)
        arrays = [
            xp.asarray(value, dtype=xp.float64)
            for value in (rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down)
        ]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("SCAN spin inputs must share one shape")
        outputs = [xp.zeros(shape, dtype=xp.float64) for _ in range(8)]
        active = (arrays[0] + arrays[1]) >= SCAN_DENSITY_THRESHOLD
        if not bool(active.any()):
            return (*outputs, None, None)
        selected = [array[active] for array in arrays]
        raw_up = selected[0]
        raw_down = selected[1]

        def energy_of(module, stacked, clamp):
            return self._polarized_energy(module, stacked, raw_up[None, :], raw_down[None, :], clamp=clamp)

        energy_of.clamp = lambda module, values: self._clamped_spin(module, *values)
        values, partials = self._differentiate(xp, selected, energy_of)
        outputs[0][active] = values
        for output, partial in zip(outputs[1:], partials):
            output[active] = partial
        return (*outputs, None, None)


__all__ = ["NativeScan", "SCAN_DENSITY_THRESHOLD"]
