"""Semilocal exchange and correlation components and their weighted sums.

A hybrid functional combines exact exchange with a weighted sum of semilocal
pieces (:mod:`parsec_python.V_xc.hybrid`).  Each piece here is a pointwise
function of the density and its gradient, in Hartree atomic units, returning
the energy per volume and its first derivatives (what the variational grid
potential of :mod:`parsec_python.V_xc.semilocal` needs).

============== ============ ================================================ ==============================
name           kind         definition                                       libxc
============== ============ ================================================ ==============================
``slater_x``   exchange     Dirac/Slater LDA exchange                        ``LDA_X`` (1)
``pbe_x``      exchange     PBE exchange (:mod:`.pbe`)                       ``GGA_X_PBE`` (101)
``b88_x``      exchange     Becke 1988 exchange, Slater part included        ``GGA_X_B88`` (106)
``pw91_x``     exchange     PW91 exchange (:mod:`.pw91`)                     ``GGA_X_PW91`` (109)
``wpbe_sr_x``  exchange     short-range omega-PBE exchange of HSE (:mod:`.wpbe`) ``GGA_X_WPBEH`` (524)
``b88_sr_x``   exchange     short-range B88 exchange of ITYH (:mod:`.ityh`)  ``GGA_X_ITYH`` (529)
``pbe_c``      correlation  PBE correlation (:mod:`.pbe`, :mod:`.pbe_spin`)  ``GGA_C_PBE`` (130)
``pw91_c``     correlation  PW91 correlation (:mod:`.pw91`)                  ``GGA_C_PW91`` (134)
``pw92_c``     correlation  PW92 LDA correlation, the local part of PW91     ``LDA_C_PW`` (12)
``vwn_rpa_c``  correlation  VWN fit to the RPA gas, f(zeta) interpolation    ``LDA_C_VWN_RPA`` (8)
``vwn5_c``     correlation  VWN fit V to the Ceperley-Alder gas               ``LDA_C_VWN`` (7)
``lyp_c``      correlation  Lee-Yang-Parr correlation                        ``GGA_C_LYP`` (131)
============== ============ ================================================ ==============================

Conventions of every component function (internal, inputs already checked):

* unpolarized: ``f(n, sigma, omega) -> (e, de/dn, de/dsigma)``,
  ``sigma = |grad n|**2``;
* spin: ``f(n_up, n_down, s_uu, s_ud, s_dd, omega) -> (e, de/dn_up,
  de/dn_down, de/ds_uu, de/ds_ud, de/ds_dd)`` with ``s_ab = grad n_a . grad n_b``
  (libxc's variables and ``vrho``, ``vsigma``); a component that depends on the
  total gradient forms ``|grad n|**2 = s_uu + 2 s_ud + s_dd`` itself.

Exchange obeys the exact spin scaling ``E_x[n_up, n_down] = (E_x[2 n_up] +
E_x[2 n_down]) / 2`` for any interaction (also the screened ones), so each
exchange component is written once for an unpolarized density.  Below a
density of 1e-20 (the vacuum branch of :mod:`.pbe`) a point carries nothing.

:class:`ComponentSemilocal` is a weighted sum of components with the
interface :class:`parsec_python.V_xc.semilocal.SemilocalXC`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .pbe import _DENSITY_THRESHOLD
from .pbe import _correlation_partials as _pbe_correlation_partials
from .pbe import _exchange_partials as _pbe_exchange_partials
from .pbe_spin import _spin_correlation_partials as _pbe_spin_correlation_partials
from .pw91 import _correlation_partials as _pw91_correlation_partials
from .pw91 import _exchange_partials as _pw91_exchange_partials
from .pw91 import _spin_correlation_partials as _pw91_spin_correlation_partials
from .pw92_lda import pw92_lsd_correlation
from .ityh import ityh_b88_exchange_partials
from .wpbe import wpbe_sr_exchange_partials
from .becke_lyp import (
    b88_exchange_partials,
    lyp_correlation_partials,
    lyp_spin_correlation_partials,
    slater_exchange_partials,
    vwn_correlation_partials,
    vwn_spin_correlation_partials,
)


#: Kinds of components.
EXCHANGE = "exchange"
CORRELATION = "correlation"


@dataclass(frozen=True)
class ComponentForm:
    """What a component is and the functions that evaluate it."""

    name: str
    kind: str
    description: str
    libxc: str
    gradient: bool
    range_separated: bool
    unpolarized: Callable | None
    spin: Callable | None

    @property
    def implemented(self) -> bool:
        return self.unpolarized is not None and self.spin is not None


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------


def positive_unpolarized(core: Callable) -> Callable:
    """Full-array unpolarized component from ``core(n, sigma, omega)`` on points with ``n > 1e-20``."""

    def evaluate(n, sigma, omega=None):
        energy = np.zeros_like(n)
        d_n = np.zeros_like(n)
        d_sigma = np.zeros_like(n)
        positive = n > _DENSITY_THRESHOLD
        if np.any(positive):
            e, en, es = core(n[positive], sigma[positive], omega)
            energy[positive] = e
            d_n[positive] = en
            d_sigma[positive] = es
        return energy, d_n, d_sigma

    evaluate.core = core
    return evaluate


def spin_scaled_exchange(core: Callable) -> Callable:
    """Spin form of an exchange component from its unpolarized ``core(n, sigma, omega)``.

    ``E_x[n_up, n_down] = (E_x[2 n_up] + E_x[2 n_down]) / 2``: a channel is an
    unpolarized density ``2 n_s`` with squared gradient ``4 s_ss`` and carries
    half of its energy, so ``de/dn_s = e_n(2 n_s)`` and ``de/ds_ss = 2
    e_sigma(4 s_ss)`` (as :func:`parsec_python.V_xc.pbe_spin.pbe_spin_energy_partials`).
    """

    def evaluate(up, down, s_uu, s_ud, s_dd, omega=None):
        energy = np.zeros_like(up)
        d_up = np.zeros_like(up)
        d_down = np.zeros_like(up)
        d_uu = np.zeros_like(up)
        d_dd = np.zeros_like(up)
        for density, sigma, d_density, d_sigma in ((up, s_uu, d_up, d_uu), (down, s_dd, d_down, d_dd)):
            doubled = 2.0 * density
            occupied = doubled > _DENSITY_THRESHOLD
            if not np.any(occupied):
                continue
            e, en, es = core(doubled[occupied], np.maximum(4.0 * sigma[occupied], 0.0), omega)
            energy[occupied] += 0.5 * e
            d_density[occupied] += en
            d_sigma[occupied] += 2.0 * es
        return energy, d_up, d_down, d_uu, np.zeros_like(up), d_dd

    evaluate.core = core
    return evaluate


def positive_spin(core: Callable) -> Callable:
    """Full-array spin component from ``core(n, up, down, s_uu, s_ud, s_dd, omega)`` where ``n > 1e-20``."""

    def evaluate(up, down, s_uu, s_ud, s_dd, omega=None):
        outputs = [np.zeros_like(up) for _ in range(6)]
        density = up + down
        positive = density > _DENSITY_THRESHOLD
        if np.any(positive):
            values = core(
                density[positive],
                up[positive],
                down[positive],
                s_uu[positive],
                s_ud[positive],
                s_dd[positive],
                omega,
            )
            for output, value in zip(outputs, values):
                output[positive] = value
        return tuple(outputs)

    evaluate.core = core
    return evaluate


# ---------------------------------------------------------------------------
# PBE (the functions of pbe.py and pbe_spin.py, bit for bit)
# ---------------------------------------------------------------------------


def _pbe_exchange_core(n, sigma, omega):
    return _pbe_exchange_partials(n, sigma)


def _pbe_correlation_core(n, sigma, omega):
    return _pbe_correlation_partials(n, sigma)


def _total_gradient_form(partials):
    """A correlation of ``(n, zeta, |grad n|**2)`` in libxc's spin variables."""

    def core(n, up, down, s_uu, s_ud, s_dd, omega):
        s_tot = np.maximum(s_uu + 2.0 * s_ud + s_dd, 0.0)
        energy, d_up, d_down, d_tot = partials(n, up, down, s_tot)
        return energy, d_up, d_down, d_tot, 2.0 * d_tot, d_tot

    return core


_pbe_spin_correlation_core = _total_gradient_form(_pbe_spin_correlation_partials)


SEMILOCAL_COMPONENTS: dict[str, ComponentForm] = {}


def _register(form: ComponentForm) -> None:
    SEMILOCAL_COMPONENTS[form.name] = form


_register(
    ComponentForm(
        "pbe_x",
        EXCHANGE,
        "PBE exchange",
        "GGA_X_PBE (101)",
        True,
        False,
        positive_unpolarized(_pbe_exchange_core),
        spin_scaled_exchange(_pbe_exchange_core),
    )
)
_register(
    ComponentForm(
        "pbe_c",
        CORRELATION,
        "PBE correlation",
        "GGA_C_PBE (130)",
        True,
        False,
        positive_unpolarized(_pbe_correlation_core),
        positive_spin(_pbe_spin_correlation_core),
    )
)
# ---------------------------------------------------------------------------
# Slater, B88, VWN, LYP (becke_lyp.py), PW91 and PW92 (pw91.py, pw92_lda.py)
# ---------------------------------------------------------------------------


def _pw91_exchange_core(n, sigma, omega):
    return _pw91_exchange_partials(n, sigma)


def _pw91_correlation_core(n, sigma, omega):
    return _pw91_correlation_partials(n, sigma)


_pw91_spin_correlation_core = _total_gradient_form(_pw91_spin_correlation_partials)


def _pw92_correlation_core(n, sigma, omega):
    """PW92 ``e = n e_c(r_s, 0)`` with the digits of :mod:`.pw92_lda` (the local part of PW91)."""

    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    epsilon, epsilon_rs, _ = pw92_lsd_correlation(rs, np.zeros_like(n))
    return n * epsilon, epsilon - rs * epsilon_rs / 3.0, np.zeros_like(n)


def _pw92_spin_correlation_core(n, up, down, s_uu, s_ud, s_dd, omega):
    rs = np.cbrt(3.0 / (4.0 * np.pi * n))
    zeta = (up - down) / n
    epsilon, epsilon_rs, epsilon_zeta = pw92_lsd_correlation(rs, zeta)
    common = epsilon - rs * epsilon_rs / 3.0
    zero = np.zeros_like(n)
    return n * epsilon, common + (1.0 - zeta) * epsilon_zeta, common - (1.0 + zeta) * epsilon_zeta, zero, zero, zero


def _vwn(parametrization: str):
    def unpolarized(n, sigma, omega):
        return vwn_correlation_partials(n, sigma, parametrization=parametrization)

    def spin(n, up, down, s_uu, s_ud, s_dd, omega):
        return vwn_spin_correlation_partials(n, up, down, parametrization=parametrization)

    return positive_unpolarized(unpolarized), positive_spin(spin)


def _lyp_spin_core(n, up, down, s_uu, s_ud, s_dd, omega):
    return lyp_spin_correlation_partials(n, up, down, s_uu, s_ud, s_dd)


for _form in (
    ComponentForm(
        "slater_x", EXCHANGE, "Slater (Dirac) LDA exchange", "LDA_X (1)", False, False,
        positive_unpolarized(slater_exchange_partials), spin_scaled_exchange(slater_exchange_partials),
    ),
    ComponentForm(
        "b88_x", EXCHANGE, "Becke 1988 exchange (Slater included)", "GGA_X_B88 (106)", True, False,
        positive_unpolarized(b88_exchange_partials), spin_scaled_exchange(b88_exchange_partials),
    ),
    ComponentForm(
        "pw91_x", EXCHANGE, "PW91 exchange", "GGA_X_PW91 (109)", True, False,
        positive_unpolarized(_pw91_exchange_core), spin_scaled_exchange(_pw91_exchange_core),
    ),
    ComponentForm(
        "pw91_c", CORRELATION, "PW91 correlation", "GGA_C_PW91 (134)", True, False,
        positive_unpolarized(_pw91_correlation_core), positive_spin(_pw91_spin_correlation_core),
    ),
    ComponentForm(
        "pw92_c", CORRELATION, "PW92 LDA correlation (digits of Fortran PARSEC)", "LDA_C_PW (12)", False, False,
        positive_unpolarized(_pw92_correlation_core), positive_spin(_pw92_spin_correlation_core),
    ),
    ComponentForm(
        "vwn_rpa_c", CORRELATION, "VWN RPA correlation, f(zeta) interpolation", "LDA_C_VWN_RPA (8)", False, False,
        *_vwn("rpa"),
    ),
    ComponentForm(
        "vwn5_c", CORRELATION, "VWN5 correlation", "LDA_C_VWN (7)", False, False,
        *_vwn("vwn5"),
    ),
    ComponentForm(
        "lyp_c", CORRELATION, "LYP correlation", "GGA_C_LYP (131)", True, False,
        positive_unpolarized(lyp_correlation_partials), positive_spin(_lyp_spin_core),
    ),
):
    _register(_form)
def _wpbe_core(n, sigma, omega):
    return wpbe_sr_exchange_partials(n, sigma, omega)


def _ityh_core(n, sigma, omega):
    return ityh_b88_exchange_partials(n, sigma, omega)


for _form in (
    ComponentForm(
        "wpbe_sr_x", EXCHANGE, "short-range omega-PBE exchange of HSE (Ernzerhof-Perdew hole)", "GGA_X_WPBEH (524)",
        True, True, positive_unpolarized(_wpbe_core), spin_scaled_exchange(_wpbe_core),
    ),
    ComponentForm(
        "b88_sr_x", EXCHANGE, "short-range B88 exchange (ITYH attenuation)", "GGA_X_ITYH (529)",
        True, True, positive_unpolarized(_ityh_core), spin_scaled_exchange(_ityh_core),
    ),
):
    _register(_form)


# ---------------------------------------------------------------------------
# Weighted sums
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemilocalComponent:
    """One weighted semilocal piece: ``weight * E[name](omega)``.

    ``omega`` (bohr**-1) is the range-separation parameter of a short-range
    exchange component and ``None`` for every other one.
    """

    name: str
    weight: float
    omega: float | None = None

    def __post_init__(self) -> None:
        form = SEMILOCAL_COMPONENTS.get(self.name)
        if form is None:
            raise ValueError(f"unknown semilocal component {self.name!r}")
        if form.range_separated:
            if self.omega is None or not np.isfinite(self.omega) or self.omega < 0.0:
                raise ValueError(f"{self.name} needs a nonnegative range-separation parameter omega")
        elif self.omega is not None:
            raise ValueError(f"{self.name} takes no range-separation parameter")
        if not np.isfinite(self.weight):
            raise ValueError("a component weight must be finite")

    @property
    def form(self) -> ComponentForm:
        return SEMILOCAL_COMPONENTS[self.name]

    @property
    def kind(self) -> str:
        return self.form.kind

    def describe(self) -> str:
        text = f"{self.weight:g} {self.name}"
        return text if self.omega is None else f"{text}(omega={self.omega:g})"


def _checked_unpolarized(rho, sigma):
    rho = np.asarray(rho, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if rho.shape != sigma.shape:
        raise ValueError("density and squared gradient must have the same shape")
    if np.any(rho < -1.0e-14):
        raise ValueError("the semilocal functional requires a nonnegative density")
    if np.any(sigma < -1.0e-14):
        raise ValueError("the semilocal functional requires a nonnegative squared gradient")
    return np.maximum(rho, 0.0), np.maximum(sigma, 0.0)


@dataclass(frozen=True)
class ComponentSemilocal:
    """The weighted sum of :class:`SemilocalComponent` pieces as a :class:`~.semilocal.SemilocalXC`."""

    components: tuple[SemilocalComponent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))

    @property
    def family(self) -> str:
        active = [component for component in self.components if component.weight != 0.0]
        if not active:
            return "none"
        return "gga" if any(component.form.gradient for component in active) else "lda"

    @property
    def implemented(self) -> bool:
        return all(component.form.implemented for component in self.components)

    @property
    def exchange(self) -> tuple[SemilocalComponent, ...]:
        return tuple(c for c in self.components if c.kind == EXCHANGE)

    @property
    def correlation(self) -> tuple[SemilocalComponent, ...]:
        return tuple(c for c in self.components if c.kind == CORRELATION)

    def describe(self) -> str:
        return " + ".join(component.describe() for component in self.components) or "none"

    def _forms(self):
        for component in self.components:
            if component.weight == 0.0:
                continue
            form = component.form
            if not form.implemented:
                raise NotImplementedError(
                    f"the semilocal component {component.name} is not implemented yet"
                )
            yield component, form

    def unpolarized(self, rho, sigma):
        """``(e, vrho, vsigma)`` of the weighted sum (Hartree, per volume)."""

        rho, sigma = _checked_unpolarized(rho, sigma)
        energy = np.zeros_like(rho)
        d_rho = np.zeros_like(rho)
        d_sigma = np.zeros_like(rho)
        for component, form in self._forms():
            e, en, es = form.unpolarized(rho, sigma, component.omega)
            weight = component.weight
            energy += weight * e
            d_rho += weight * en
            d_sigma += weight * es
        return energy, d_rho, d_sigma

    def polarized(self, rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd):
        """libxc's ``(e, vrho_up, vrho_down, vsigma_uu, vsigma_ud, vsigma_dd)`` (Hartree, per volume)."""

        arrays = [np.asarray(value, dtype=np.float64) for value in (rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd)]
        up, down, s_uu, s_ud, s_dd = arrays
        if any(value.shape != up.shape for value in arrays):
            raise ValueError("spin densities and gradients must share one shape")
        if np.any(up < -1.0e-14) or np.any(down < -1.0e-14):
            raise ValueError("the semilocal functional requires nonnegative spin densities")
        if np.any(s_uu < -1.0e-14) or np.any(s_dd < -1.0e-14):
            raise ValueError("the semilocal functional requires nonnegative squared spin-density gradients")
        up = np.maximum(up, 0.0)
        down = np.maximum(down, 0.0)
        s_uu = np.maximum(s_uu, 0.0)
        s_dd = np.maximum(s_dd, 0.0)
        outputs = [np.zeros_like(up) for _ in range(6)]
        for component, form in self._forms():
            values = form.spin(up, down, s_uu, s_ud, s_dd, component.omega)
            weight = component.weight
            for output, value in zip(outputs, values):
                output += weight * value
        return tuple(outputs)


__all__ = [
    "CORRELATION",
    "EXCHANGE",
    "ComponentForm",
    "ComponentSemilocal",
    "SEMILOCAL_COMPONENTS",
    "SemilocalComponent",
    "positive_spin",
    "positive_unpolarized",
    "spin_scaled_exchange",
]
