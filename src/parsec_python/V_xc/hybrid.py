"""Hybrid exchange-correlation functionals: definitions, the label registry and their semilocal parts.

A hybrid functional of this code is

``E_xc = alpha E_x^HF[1/r] + beta E_x^HF[erfc(omega r)/r] + E_xc^sl[n]``

with the exact (Hartree-Fock) exchange of the occupied orbitals for an
interaction kernel (:mod:`parsec_python.V_xc.exact_exchange`) and a
semilocal part ``E_xc^sl`` (a weighted sum of LDA/GGA pieces,
:mod:`parsec_python.V_xc.semilocal_components`).  ``(alpha, beta, omega)``
are libxc's CAM coefficients (``xc_hyb_cam_coef``); ``omega`` is in
bohr**-1.  The fraction of exact exchange is ``alpha + beta`` at short
range (``r -> 0``) and ``alpha`` at long range.  The forms map as

==================  =======================  ==================================
form                (alpha, beta, omega)     kernels of the exchange operator
==================  =======================  ==================================
global hybrid       (a, 0, 0)                ``a/r``
screened (HSE)      (0, b, w)                ``b erfc(w r)/r``
long-range (LC)     (a, -a, w)               ``a erf(w r)/r``
CAM                 (a, b, w), b < 0         ``(a+b)/r + (-b) erf(w r)/r``
general             (a, b, w), b > 0         ``a/r + b erfc(w r)/r``
==================  =======================  ==================================

(``erf = 1 - erfc``); :attr:`HybridFunctional.exact_exchange_terms` gives
that decomposition with nonnegative fractions where one exists, and
:meth:`HybridFunctional.exchange_kernel_spectrum` the Fourier transform of the
whole weighted kernel (one Poisson-type solve per pair density for any form).

The registry (:data:`HYBRID_FUNCTIONALS`, :func:`hybrid_functional`) holds
``hf``, ``pbe0``, ``pbeh`` (adjustable fraction), ``hse06``, ``hse03``,
``b3lyp``, ``b3lyp5``, ``b3pw91``, ``lc-wpbe`` and ``cam-b3lyp``; each entry
states its original definition, its references and libxc id, and how
libxc, Gaussian and Quantum ESPRESSO differ (:attr:`HybridFunctional.variants`).

The semilocal part is a pluggable evaluator (:class:`parsec_python.V_xc.semilocal.SemilocalXC`);
:func:`hybrid_semilocal_xc` and :func:`hybrid_semilocal_xc_spin` evaluate it
on the grid with the frozen core density, the discrete variational potential
of :mod:`parsec_python.V_xc.pbe` and the Rydberg units of the SCF.  The
exact exchange is evaluated from the valence orbitals alone (no core density
enters it), as in the reference codes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .ca_lda import XCResult
from .ca_lda_spin import SpinXCResult
from .semilocal import SemilocalXC, semilocal_xc, semilocal_xc_spin
from .semilocal_components import ComponentSemilocal, SemilocalComponent


#: Names of the interaction kernels of exact exchange.
EXCHANGE_KERNELS = ("coulomb", "erfc", "erf")
# Fractions this close are the same (|alpha + beta| of an LC functional).
_FRACTION_TOLERANCE = 1.0e-14


@dataclass(frozen=True)
class ExactExchangeTerm:
    """``fraction * E_x^HF[kernel]``: ``coulomb`` 1/r, ``erfc`` erfc(omega r)/r, ``erf`` erf(omega r)/r."""

    kernel: str
    fraction: float
    omega: float | None = None

    def __post_init__(self) -> None:
        if self.kernel not in EXCHANGE_KERNELS:
            raise ValueError(f"unknown exchange kernel {self.kernel!r}")
        if (self.kernel == "coulomb") != (self.omega is None):
            raise ValueError("the coulomb kernel takes no omega; erfc and erf need one")


@dataclass(frozen=True)
class HybridFunctional:
    """A hybrid functional: CAM coefficients of exact exchange and a semilocal evaluator.

    ``alpha``, ``beta``, ``omega``
        libxc's CAM coefficients: ``E_x^HF = alpha E_x[1/r] + beta E_x[erfc(omega r)/r]``
        (``omega`` in bohr**-1, ``None`` without range separation).
    ``semilocal``
        the semilocal part, any :class:`~parsec_python.V_xc.semilocal.SemilocalXC`;
        the functionals of the registry use a
        :class:`~parsec_python.V_xc.semilocal_components.ComponentSemilocal`
        (its ``components`` name the pieces and weights).
    ``spin_supported``
        whether the semilocal part has a spin-polarized form (exact exchange
        always has one: only orbitals of the same spin exchange).
    ``initial_functional``
        the semilocal ``Correlation_Type`` of this code suggested for the SCF
        that precedes the first exact-exchange step.
    ``references``, ``libxc``, ``variants``
        the papers, the libxc functional this one corresponds to, and the
        differences between the definitions of libxc, Gaussian and Quantum
        ESPRESSO and this code.
    """

    label: str
    name: str
    alpha: float
    beta: float = 0.0
    omega: float | None = None
    semilocal: SemilocalXC = field(default_factory=ComponentSemilocal)
    spin_supported: bool = True
    initial_functional: str = "pbe"
    references: tuple[str, ...] = ()
    libxc: str = ""
    variants: str = ""

    def __post_init__(self) -> None:
        for value in (self.alpha, self.beta):
            if not np.isfinite(value):
                raise ValueError("the exact-exchange coefficients must be finite")
        if self.beta != 0.0:
            if self.omega is None or not np.isfinite(self.omega) or self.omega <= 0.0:
                raise ValueError("a range-separated hybrid needs a positive omega (bohr**-1)")
        elif self.omega is not None:
            raise ValueError("omega belongs to range-separated hybrids (beta != 0) only")

    # -- exact exchange ----------------------------------------------------

    @property
    def cam_coefficients(self) -> tuple[float, float, float]:
        """``(alpha, beta, omega)`` as libxc's ``xc_hyb_cam_coef`` reports them (omega 0 if none)."""

        return float(self.alpha), float(self.beta), float(self.omega or 0.0)

    @property
    def range_separated(self) -> bool:
        return self.beta != 0.0

    @property
    def short_range_fraction(self) -> float:
        """Fraction of exact exchange at short range, ``alpha + beta``."""

        return float(self.alpha + self.beta)

    @property
    def long_range_fraction(self) -> float:
        """Fraction of exact exchange at long range, ``alpha``."""

        return float(self.alpha)

    @property
    def needs_exact_exchange(self) -> bool:
        return bool(self.exact_exchange_terms)

    @property
    def exact_exchange_terms(self) -> tuple[ExactExchangeTerm, ...]:
        """The exact exchange as a sum of kernels with nonnegative fractions where possible (module table)."""

        alpha, beta, omega = float(self.alpha), float(self.beta), self.omega
        if beta == 0.0:
            return () if alpha == 0.0 else (ExactExchangeTerm("coulomb", alpha),)
        if alpha == 0.0:
            return (ExactExchangeTerm("erfc", beta, omega),)
        if abs(alpha + beta) <= _FRACTION_TOLERANCE * max(abs(alpha), 1.0):
            return (ExactExchangeTerm("erf", alpha, omega),)
        if beta < 0.0:
            return (
                ExactExchangeTerm("coulomb", alpha + beta),
                ExactExchangeTerm("erf", -beta, omega),
            )
        return (ExactExchangeTerm("coulomb", alpha), ExactExchangeTerm("erfc", beta, omega))

    @property
    def kernels(self) -> tuple[str, ...]:
        return tuple(term.kernel for term in self.exact_exchange_terms)

    def exchange_kernel_spectrum(self, k_squared) -> np.ndarray:
        """Fourier transform of the weighted exchange kernel, ``(4 pi/k**2)[alpha + beta(1 - exp(-k**2/(4 omega**2)))]``.

        Hartree atomic units (the potential of a unit point charge; twice it in
        Rydberg); ``k_squared`` in bohr**-2, positive.  For a range-separated
        functional the short-range part is finite at ``k -> 0``:
        ``pi beta / omega**2``.
        """

        k2 = np.asarray(k_squared, dtype=np.float64)
        if np.any(k2 <= 0.0):
            raise ValueError("the kernel spectrum is evaluated at positive k**2")
        values = self.alpha * np.ones_like(k2)
        if self.beta != 0.0:
            values = values - self.beta * np.expm1(-k2 / (4.0 * self.omega * self.omega))
        return 4.0 * np.pi * values / k2

    def exact_exchange_energy(self, energies: Mapping[str, float]) -> float:
        """``sum_k fraction_k E_k`` for the exact-exchange energies of the kernels (any one unit)."""

        return float(sum(term.fraction * float(energies[term.kernel]) for term in self.exact_exchange_terms))

    # -- semilocal part ----------------------------------------------------

    @property
    def components(self) -> tuple[SemilocalComponent, ...]:
        """The weighted pieces of a component-built semilocal part (empty otherwise)."""

        return tuple(getattr(self.semilocal, "components", ()))

    def describe(self) -> str:
        """One line for output files: name, exact exchange and semilocal pieces."""

        terms = " + ".join(
            f"{term.fraction:g} HF[{term.kernel}"
            + ("" if term.omega is None else f", omega={term.omega:g}")
            + "]"
            for term in self.exact_exchange_terms
        ) or "no exact exchange"
        describe = getattr(self.semilocal, "describe", None)
        local = describe() if callable(describe) else type(self.semilocal).__name__
        return f"{self.name}: {terms} + {local}"

    def semilocal_xc(self, valence_density, grid, core_density=None) -> XCResult:
        return hybrid_semilocal_xc(self, valence_density, grid, core_density)

    def semilocal_xc_spin(self, valence_up, valence_down, grid, core_density=None) -> SpinXCResult:
        return hybrid_semilocal_xc_spin(self, valence_up, valence_down, grid, core_density)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: HSE03 screening parameters (Heyd, Scuseria, Ernzerhof, erratum 2006): omega_HF = 0.15/sqrt(2),
#: omega_PBE = 0.15 * 2**(1/3) bohr**-1.
HSE03_OMEGA_HF = 0.15 / np.sqrt(2.0)
HSE03_OMEGA_PBE = 0.15 * 2.0 ** (1.0 / 3.0)
#: HSE06 (Krukau, Vydrov, Izmaylov, Scuseria 2006): omega = 0.11 bohr**-1 in both parts.
HSE06_OMEGA = 0.11

_PBE_REFERENCES = (
    "J. P. Perdew, K. Burke, M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996)",
)
_EP_HOLE_REFERENCES = (
    "M. Ernzerhof, J. P. Perdew, J. Chem. Phys. 109, 3313 (1998) (the exchange hole)",
    "J. Heyd, G. E. Scuseria, M. Ernzerhof, J. Chem. Phys. 118, 8207 (2003)",
    "J. Heyd, G. E. Scuseria, M. Ernzerhof, J. Chem. Phys. 124, 219906 (2006) (erratum)",
    "J. Heyd, G. E. Scuseria, J. Chem. Phys. 120, 7274 (2004) (rescaling of s)",
)
_B3_REFERENCES = (
    "A. D. Becke, J. Chem. Phys. 98, 5648 (1993) (the three-parameter form, with PW91)",
    "A. D. Becke, Phys. Rev. A 38, 3098 (1988) (B88 exchange)",
)
_LYP_REFERENCES = (
    "C. Lee, W. Yang, R. G. Parr, Phys. Rev. B 37, 785 (1988)",
    "B. Miehlich, A. Savin, H. Stoll, H. Preuss, Chem. Phys. Lett. 157, 200 (1989) (the gradient-only form)",
    "S. H. Vosko, L. Wilk, M. Nusair, Can. J. Phys. 58, 1200 (1980)",
)


def _sl(*components: SemilocalComponent) -> ComponentSemilocal:
    return ComponentSemilocal(tuple(components))


def _pbeh(label: str, name: str, fraction: float) -> HybridFunctional:
    return HybridFunctional(
        label=label,
        name=name,
        alpha=float(fraction),
        semilocal=_sl(SemilocalComponent("pbe_x", 1.0 - float(fraction)), SemilocalComponent("pbe_c", 1.0)),
        references=_PBE_REFERENCES
        + (
            "J. P. Perdew, M. Ernzerhof, K. Burke, J. Chem. Phys. 105, 9982 (1996) (the 1/4 mixing)",
            "C. Adamo, V. Barone, J. Chem. Phys. 110, 6158 (1999) (PBE0)",
            "M. Ernzerhof, G. E. Scuseria, J. Chem. Phys. 110, 5029 (1999) (PBEh)",
        ),
        libxc="HYB_GGA_XC_PBEH (406), ext. parameter _beta = the fraction",
        variants=(
            "E_xc = a E_x^HF + (1 - a) E_x^PBE + E_c^PBE, a = 1/4 for PBE0.  The same in libxc (PBEH), "
            "Gaussian (PBE1PBE; other fractions with IOp(3/76)) and Quantum ESPRESSO (PBE0, exx_fraction).  "
            "The PBE of this code has the PW92 A = (1 - ln 2)/pi**2 (libxc: 0.0310907), 3e-7 of the local "
            "correlation; the exchange is identical."
        ),
    )


def _hse(label: str, name: str, fraction: float, omega_hf: float, omega_pbe: float) -> HybridFunctional:
    return HybridFunctional(
        label=label,
        name=name,
        alpha=0.0,
        beta=float(fraction),
        omega=float(omega_hf),
        semilocal=_sl(
            SemilocalComponent("pbe_x", 1.0),
            SemilocalComponent("wpbe_sr_x", -float(fraction), float(omega_pbe)),
            SemilocalComponent("pbe_c", 1.0),
        ),
        references=_PBE_REFERENCES
        + _EP_HOLE_REFERENCES
        + ("A. V. Krukau, O. A. Vydrov, A. F. Izmaylov, G. E. Scuseria, J. Chem. Phys. 125, 224106 (2006) (HSE06)",),
        libxc="HYB_GGA_XC_HSE06 (428)" if label == "hse06" else "HYB_GGA_XC_HSE03 (427)",
        variants=(
            "E_xc = a E_x^HF,SR(w_HF) + E_x^PBE - a E_x^wPBE,SR(w_PBE) + E_c^PBE, a = 1/4, with the "
            "short-range PBE exchange of the Ernzerhof-Perdew hole (Heyd's analytic integration with his "
            "erfc fit), as in Quantum ESPRESSO (HSE) and VASP; the long-range PBE part is the full PBE "
            "exchange minus the short-range one, as in Quantum ESPRESSO.  HSE06: w_HF = w_PBE = 0.11 bohr**-1 (libxc, Gaussian "
            "HSEH1PBE); Quantum ESPRESSO's default screening_parameter is 0.106 and VASP's HFSCREEN = 0.2/A "
            "(0.1058 bohr**-1), the same in both parts.  HSE03: w_HF = 0.15/sqrt(2), w_PBE = 0.15*2**(1/3) "
            "(the erratum; libxc HSE03, Gaussian HSE2PBE).  libxc differs: its full-range part is the wPBEh "
            "hole at w = 0 instead of PBE exchange, it rescales s with Henderson et al. 2009 Eq. 8 instead "
            "of Heyd and Scuseria 2004 (s > 8.3), and libxc 7 derives the hole constants E, F(s) and the "
            "small-s EG(s) from A..D instead of the published digits (see V_xc.wpbe)."
        ),
    )


def _b3(
    label: str, name: str, local_correlation: str, gradient_correlation: str, libxc: str, variants: str
) -> HybridFunctional:
    return HybridFunctional(
        label=label,
        name=name,
        alpha=0.20,
        semilocal=_sl(
            SemilocalComponent("slater_x", 0.08),
            SemilocalComponent("b88_x", 0.72),
            SemilocalComponent(local_correlation, 0.19),
            SemilocalComponent(gradient_correlation, 0.81),
        ),
        references=_B3_REFERENCES
        + (
            _LYP_REFERENCES
            + ("P. J. Stephens, F. J. Devlin, C. F. Chabalowski, M. J. Frisch, J. Phys. Chem. 98, 11623 (1994) (B3LYP)",)
            if gradient_correlation == "lyp_c"
            else ("J. P. Perdew et al., Phys. Rev. B 46, 6671 (1992) (PW91)",)
        ),
        libxc=libxc,
        variants=variants,
    )


def _registry() -> dict[str, HybridFunctional]:
    entries = [
        HybridFunctional(
            label="hf",
            name="Hartree-Fock",
            alpha=1.0,
            semilocal=_sl(),
            references=("exact exchange, no correlation",),
            libxc="none (the exchange of the orbitals only)",
            variants=(
                "E_xc = E_x^HF, no correlation.  Gaussian HF, Quantum ESPRESSO input_dft='HF'.  The SCF before "
                "the first exchange step uses a semilocal functional (initial_functional)."
            ),
        ),
        _pbeh("pbe0", "PBE0", 0.25),
        _pbeh("pbeh", "PBEh", 0.25),
        _hse("hse06", "HSE06", 0.25, HSE06_OMEGA, HSE06_OMEGA),
        _hse("hse03", "HSE03", 0.25, HSE03_OMEGA_HF, HSE03_OMEGA_PBE),
        _b3(
            "b3lyp",
            "B3LYP",
            "vwn_rpa_c",
            "lyp_c",
            "HYB_GGA_XC_B3LYP (402)",
            "E_xc = 0.20 E_x^HF + 0.08 E_x^Slater + 0.72 E_x^B88 + 0.19 E_c^VWN-RPA + 0.81 E_c^LYP (B88 "
            "includes Slater: 0.80 Slater + 0.72 B88 gradient correction).  The local correlation is the VWN "
            "fit to the RPA gas with the f(zeta) spin interpolation (libxc LDA_C_VWN_RPA; Gaussian's "
            "'VWN III'), as in libxc B3LYP, Gaussian B3LYP and this code's Energy_Refinement: b3lyp.  "
            "Quantum ESPRESSO's 'B3LYP' uses VWN5 instead (our b3lyp5); its 'B3LYP-V1R' is this one.",
        ),
        _b3(
            "b3lyp5",
            "B3LYP5 (VWN5)",
            "vwn5_c",
            "lyp_c",
            "HYB_GGA_XC_B3LYP5 (475)",
            "B3LYP with the VWN5 local correlation (fit to Ceperley-Alder, VWN Eq. 4.4 with the spin "
            "stiffness interpolation): libxc B3LYP5, Quantum ESPRESSO 'B3LYP', Turbomole b3-lyp.",
        ),
        _b3(
            "b3pw91",
            "B3PW91",
            "pw92_c",
            "pw91_c",
            "HYB_GGA_XC_B3PW91 (401)",
            "E_xc = 0.20 E_x^HF + 0.08 E_x^Slater + 0.72 E_x^B88 + E_c^PW92 + 0.81 Delta E_c^PW91 (Becke "
            "1993), written as 0.19 PW92 + 0.81 PW91 correlation as in libxc.  The PW91 "
            "and PW92 of this code carry the digits of Fortran PARSEC (V_xc.pw91, V_xc.pw92_lda), which "
            "differ from libxc's in the 6th-9th digit of a few constants.",
        ),
        HybridFunctional(
            label="lc-wpbe",
            name="LC-wPBE",
            alpha=1.0,
            beta=-1.0,
            omega=0.4,
            semilocal=_sl(SemilocalComponent("wpbe_sr_x", 1.0, 0.4), SemilocalComponent("pbe_c", 1.0)),
            references=_PBE_REFERENCES
            + _EP_HOLE_REFERENCES[:2]
            + (
                "O. A. Vydrov, G. E. Scuseria, J. Chem. Phys. 125, 234109 (2006) (LC-wPBE, omega = 0.4)",
                "O. A. Vydrov, J. Heyd, A. V. Krukau, G. E. Scuseria, J. Chem. Phys. 125, 074106 (2006)",
            ),
            libxc="HYB_GGA_XC_LC_WPBE (478)",
            variants=(
                "E_xc = E_x^HF,LR(w) + E_x^wPBE,SR(w) + E_c^PBE, w = 0.4 bohr**-1, with the short-range "
                "omega-PBE exchange of HSE (Ernzerhof-Perdew hole), as in libxc LC_WPBE (which uses its wPBEh "
                "variant, see hse06).  The LC-wPBE of Weintraub, Henderson and Scuseria (libxc LC_WPBE_WHS, "
                "486) uses the HJS hole model instead."
            ),
        ),
        HybridFunctional(
            label="cam-b3lyp",
            name="CAM-B3LYP",
            alpha=0.65,
            beta=-0.46,
            omega=0.33,
            semilocal=_sl(
                SemilocalComponent("b88_x", 0.35),
                SemilocalComponent("b88_sr_x", 0.46, 0.33),
                SemilocalComponent("vwn5_c", 0.19),
                SemilocalComponent("lyp_c", 0.81),
            ),
            references=_B3_REFERENCES[1:]
            + _LYP_REFERENCES
            + (
                "T. Yanai, D. P. Tew, N. C. Handy, Chem. Phys. Lett. 393, 51 (2004) (CAM-B3LYP)",
                "H. Iikura, T. Tsuneda, T. Yanai, K. Hirao, J. Chem. Phys. 115, 3540 (2001) (short-range GGA exchange)",
            ),
            libxc="HYB_GGA_XC_CAM_B3LYP (433)",
            variants=(
                "E_xc = 0.19 E_x^HF + 0.46 E_x^HF,LR(w) + 0.35 E_x^B88 + 0.46 E_x^B88,SR(w) + 0.19 E_c^VWN5 "
                "+ 0.81 E_c^LYP, w = 0.33 bohr**-1 (CAM alpha = 0.65, beta = -0.46).  The short-range B88 is "
                "the ITYH attenuation of the B88 exchange; the local correlation is VWN5, as in libxc "
                "(LDA_C_VWN)."
            ),
        ),
    ]
    return {entry.label: entry for entry in entries}


#: The hybrid functionals by label, at their defining parameters.
HYBRID_FUNCTIONALS: dict[str, HybridFunctional] = _registry()
#: Other spellings accepted by :func:`hybrid_functional`.
HYBRID_ALIASES = {
    "hartree-fock": "hf",
    "pbe1pbe": "pbe0",
    "hse": "hse06",
    "hseh1pbe": "hse06",
    "hse2pbe": "hse03",
    "lcwpbe": "lc-wpbe",
    "camb3lyp": "cam-b3lyp",
}
#: Labels whose exact-exchange fraction an input may set (``Exact_Exchange_Fraction``).
ADJUSTABLE_FRACTION = frozenset({"pbeh", "hse06", "hse03"})
#: Labels whose omega an input may set (``Screening_Parameter``; one omega for both parts).
ADJUSTABLE_OMEGA = frozenset({"hse06", "lc-wpbe", "cam-b3lyp"})


def canonical_hybrid_label(label: str) -> str:
    """The registry label of ``label`` (case, ``_`` for ``-`` and the aliases ignored)."""

    key = str(label).strip().lower().replace("_", "-")
    key = HYBRID_ALIASES.get(key.replace("-", ""), HYBRID_ALIASES.get(key, key))
    if key not in HYBRID_FUNCTIONALS:
        raise ValueError(
            f"unknown hybrid functional {label!r}; known: {', '.join(sorted(HYBRID_FUNCTIONALS))}"
        )
    return key


def is_hybrid_label(label: str) -> bool:
    try:
        canonical_hybrid_label(label)
    except ValueError:
        return False
    return True


def _with_omega(functional: HybridFunctional, omega: float) -> HybridFunctional:
    components = tuple(
        replace(component, omega=omega) if component.form.range_separated else component
        for component in functional.components
    )
    return replace(functional, omega=omega, semilocal=ComponentSemilocal(components))


def hybrid_functional(
    label: str,
    *,
    exact_exchange_fraction: float | None = None,
    screening_parameter: float | None = None,
) -> HybridFunctional:
    """The registered functional ``label``, optionally with another fraction or omega.

    ``exact_exchange_fraction`` sets ``a`` of ``pbeh`` (default 1/4) and the
    short-range fraction of ``hse06``/``hse03`` (HSE(a): ``a E_x^HF,SR -
    a E_x^wPBE,SR``); the other functionals are defined by their fractions and
    accept only those.  ``screening_parameter`` (bohr**-1) sets omega of the
    exact and the semilocal part together for ``hse06``, ``lc-wpbe`` and
    ``cam-b3lyp``; ``hse03`` is defined by its two different omegas and the
    global hybrids have none.  Violations raise ``ValueError``.
    """

    key = canonical_hybrid_label(label)
    functional = HYBRID_FUNCTIONALS[key]
    if exact_exchange_fraction is not None:
        fraction = float(exact_exchange_fraction)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError("the exact-exchange fraction must lie between 0 and 1")
        if key == "pbeh":
            functional = _pbeh("pbeh", f"PBEh({fraction:g})", fraction)
        elif key in ("hse06", "hse03"):
            omega_pbe = next(c.omega for c in functional.components if c.name == "wpbe_sr_x")
            functional = _hse(key, functional.name if fraction == 0.25 else f"{functional.name}(a={fraction:g})",
                              fraction, functional.omega, omega_pbe)
        else:
            defined = functional.short_range_fraction
            if abs(fraction - defined) > 1.0e-12:
                raise ValueError(
                    f"{functional.name} is defined with {defined:g} exact exchange; "
                    + ("use pbeh for another fraction" if key == "pbe0" else "its fraction cannot be changed")
                )
    if screening_parameter is not None:
        omega = float(screening_parameter)
        if not np.isfinite(omega) or omega <= 0.0:
            raise ValueError("the screening parameter must be positive (bohr**-1)")
        if key == "hse03":
            raise ValueError(
                "HSE03 is defined by omega_HF = 0.15/sqrt(2) and omega_PBE = 0.15*2**(1/3); "
                "use hse06 with a screening parameter for one omega in both parts"
            )
        if key not in ADJUSTABLE_OMEGA:
            raise ValueError(f"{functional.name} has no screening parameter")
        functional = _with_omega(functional, omega)
    return functional


def _checked_functional(functional: HybridFunctional) -> HybridFunctional:
    if not isinstance(functional, HybridFunctional):
        raise TypeError("a HybridFunctional is required")
    return functional


def hybrid_semilocal_xc(
    functional: HybridFunctional,
    valence_density: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
) -> XCResult:
    """The semilocal part of ``functional`` for an unpolarized density (Rydberg; potential, energy density, total).

    The energy is that of the semilocal part only; the SCF adds the exact
    exchange (:meth:`HybridFunctional.exact_exchange_energy`).
    """

    functional = _checked_functional(functional)
    return semilocal_xc(functional.semilocal, valence_density, grid, core_density, name=functional.name)


def hybrid_semilocal_xc_spin(
    functional: HybridFunctional,
    valence_up: np.ndarray,
    valence_down: np.ndarray,
    grid,
    core_density: np.ndarray | None = None,
) -> SpinXCResult:
    """The semilocal part of ``functional`` for two spin densities (Rydberg)."""

    functional = _checked_functional(functional)
    if not functional.spin_supported:
        raise ValueError(f"{functional.name} has no spin-polarized form")
    return semilocal_xc_spin(
        functional.semilocal, valence_up, valence_down, grid, core_density, name=functional.name
    )


__all__ = [
    "ADJUSTABLE_FRACTION",
    "ADJUSTABLE_OMEGA",
    "EXCHANGE_KERNELS",
    "ExactExchangeTerm",
    "HSE03_OMEGA_HF",
    "HSE03_OMEGA_PBE",
    "HSE06_OMEGA",
    "HYBRID_ALIASES",
    "HYBRID_FUNCTIONALS",
    "HybridFunctional",
    "canonical_hybrid_label",
    "hybrid_functional",
    "hybrid_semilocal_xc",
    "hybrid_semilocal_xc_spin",
    "is_hybrid_label",
]
