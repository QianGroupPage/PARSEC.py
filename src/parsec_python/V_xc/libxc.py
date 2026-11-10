"""libxc's functionals in this code: names, the registry, the semilocal evaluators and the hybrids.

A libxc functional of a calculation is named in its settings by
``SCFSettings.xc_functional = "libxc:<name>[+<name>...][;fhc=on|off][;threshold=<t>][;tau=<t>]"``:
libxc's lowercase names without the ``xc_`` prefix (``libxc:gga_x_pbe+gga_c_pbe``,
``libxc:hyb_gga_xc_b3lyp``, ``libxc:mgga_x_r2scan+mgga_c_r2scan``), in the
order of the input; the components are summed.  ``fhc`` is the
Fermi-hole-curvature condition of meta-GGAs, ``tau >= tau_W = sigma/(8 n)``
per channel; absent, it is off (as in Quantum ESPRESSO, which switches off
libxc 7's default clamp, and as the native SCAN of this code).  libxc's own
enforcement lowers ``sigma`` to ``8 n tau`` without carrying it into the
derivatives, and in its polarized SCAN/r2SCAN correlation that clamp can
leave ``|grad n|**2 = 0`` and return NaN in ``vsigma`` (libxc 7.1.2: a
channel below ``tau_W`` with anti-parallel channel gradients; no NaN with
the clamp off, unpolarized or polarized, ``alpha`` down to -1.5).  ``fhc=on``
therefore asks the meta-GGA grid layer for its floor ``tau_s ->
max(tau_s, tau_W,s)`` with the exact chain rule (``tau_floor``), under which
libxc's clamp never acts.  A functional of ``z = tau_W/tau`` that libxc
clamps at ``z = 1`` (TPSS-type correlation: a jump of the derivatives at
``tau = tau_W``, found by a probe of the functional,
:func:`parsec_python.V_xc.meta_gga.smooth_at_weizsaecker`) has that floor
whatever ``fhc`` says: ``z <= 1`` for all its components, with derivatives
that are those of the floored energy.  ``threshold`` replaces libxc's density
threshold of every component (``xc_func_set_dens_threshold``; libxc's
per-functional default otherwise): below it libxc sets a point to zero and
floors each spin density at it, so that the empty channel of a fully
polarized density is not empty (2.6e-9 Ry of the PBE H atom at libxc's
default).  ``tau`` is the absolute floor of the ``tau`` of each channel of
a meta-GGA (``tau_minimum`` of the grid layer,
:func:`parsec_python.V_xc.meta_gga.weizsaecker_floor`): a point below it
keeps the energy of the floored input, never dropped, with ``vtau = 0``
there, the exact derivative of that energy.  Without the option it is
:data:`DEFAULT_TAU_THRESHOLD`.  Below it, for direct pointwise calls, the
binding floors ``tau`` at :data:`LIBXC_TAU_THRESHOLD` (1e-20, libxc's
default) on the host and on the device alike, with ``vtau = 0`` there (the
CUDA build does not apply ``xc_func_set_tau_threshold`` on the device).  Where
every occupied orbital has a zero gradient at a density maximum (the bond
centre of Si at Gamma, of H2) ``tau`` and ``tau_W`` are round-off (~1e-16),
``z = tau_W/tau`` of TPSS is a ratio of round-off and ``vtau``, ``vsigma``
reach 1e9 to 1e13 at libxc's floor (1e-20), and the SCF does not converge;
at the floor ``z = sigma/(8 n t)`` goes to 0 with ``sigma``.  Quantum
ESPRESSO (``XClib/xc_wrapper_mgga.f90``) drops a point whose ``tau`` is at
most 1e-12; on a real-space grid such a point weighs ``h^3`` (0.03 Ry of Si
at Gamma at h 0.4), so it keeps its energy here.  The family is in the names (``lda_``,
``gga_``, ``mgga_``, ``hyb_lda_``, ``hyb_gga_``, ``hyb_mgga_``), so the
settings are checked (:func:`parse_libxc_spec`) without loading the library;
:func:`canonical_libxc_functional` checks a spec against the library and
writes it in this form.

The registry comes from libxc itself (:func:`libxc_functional_info`,
:func:`available_libxc_functionals`): number, name, description, kind,
family, whether tau or the Laplacian is needed, the CAM coefficients of a
hybrid (``alpha`` at long range, ``alpha + beta`` at short range, ``omega``
in bohr**-1: the convention of :mod:`parsec_python.V_xc.hybrid`), the VV10
coefficients and the references.  :meth:`LibxcFunctionalInfo.refusal` names
what this code does not run: kinetic functionals, the nonlocal VV10
correlation, functionals fitted with an empirical dispersion term (-D, -D3:
libxc holds only their semilocal part), Yukawa-screened hybrids (CAMY/LCY), functionals of one or two
dimensions, functionals without an energy or a potential, families other
than LDA/GGA/meta-GGA and their hybrids.

:class:`LibxcSemilocal` is the pointwise evaluator of a spec
(:class:`parsec_python.V_xc.semilocal.SemilocalXC`, meta-GGA extension
included): libxc's variables and Hartree units, the energy per volume
``e = zk (rho_u + rho_d)``; NumPy arrays are evaluated by the host
functionals of libxc, CuPy arrays by its device functionals (a CUDA build of
libxc >= 7.1).  The grid functions of :mod:`parsec_python.V_xc.semilocal`
turn it into the potentials of the SCF (Rydberg, the exact discrete
derivative, frozen core density), as for every functional of the code.  For
a libxc hybrid the semilocal part is libxc's hybrid functional itself (libxc
returns its semilocal energy only) and the exact exchange comes from this
code's Fock loop (:func:`libxc_hybrid_functional`), which PARSEC 1.5 omits.

The libxc versions of the functionals of this code (``ca`` = LDA_X +
LDA_C_PZ, ``pw92`` = LDA_X + LDA_C_PW_MOD, ``pbe`` = GGA_X_PBE + GGA_C_PBE,
``pw91``, ``b3lyp``, ``pbe0`` = HYB_GGA_XC_PBEH, ``hse06``) are listed with
their differences in :data:`OWN_FUNCTIONAL_COUNTERPARTS`.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from .libxc_library import (
    FAMILIES,
    XC_FLAGS_3D,
    XC_FLAGS_DEVELOPMENT,
    XC_FLAGS_HAVE_EXC,
    XC_FLAGS_HAVE_VXC,
    XC_FLAGS_HYB_CAMY,
    XC_FLAGS_HYB_LCY,
    XC_FLAGS_NEEDS_LAPLACIAN,
    XC_FLAGS_NEEDS_TAU,
    XC_FLAGS_VV10,
    XC_KINDS,
    LibxcUnavailable,
    load_libxc,
)


#: The prefix of a libxc functional in ``SCFSettings.xc_functional``.
LIBXC_PREFIX = "libxc:"


class LibxcRefusal(ValueError):
    """A libxc functional (or combination) that this code does not run, with the reason."""


_NAME = re.compile(r"^(hyb_)?(lda|gga|mgga)_[a-z0-9_]+$")
_SPEC = re.compile(
    r"^libxc:([a-z0-9_]+(?:\+[a-z0-9_]+)*)(?:;fhc=(on|off))?(?:;threshold=([0-9.eE+-]+))?(?:;tau=([0-9.eE+-]+))?$"
)
#: The absolute floor of tau of each channel of a meta-GGA (Hartree/bohr^3) without ``;tau=``.
DEFAULT_TAU_THRESHOLD = 1.0e-12
#: The binding's own floor of tau (libxc 7's default threshold), below the floor of the grid layer;
#: applied before the call on the host and on the device (the device ignores libxc's threshold).
LIBXC_TAU_THRESHOLD = 1.0e-20
#: Whether the functional of a list of libxc components has a kink at tau = tau_W (by names and density threshold).
_KINKED: dict = {}
_FAMILY_ORDER = {"lda": 0, "gga": 1, "mgga": 2}

#: The libxc spellings of the functionals of this code, and how they differ.
OWN_FUNCTIONAL_COUNTERPARTS = {
    "ca": (
        "libxc:lda_x+lda_c_pz",
        "Ceperley-Alder/Perdew-Zunger: the same formulas; the digits of Fortran PARSEC's ca_lda "
        "(V_xc.ca_lda) and libxc's LDA_C_PZ agree.",
    ),
    "pw92": (
        "libxc:lda_x+lda_c_pw_mod",
        "Perdew-Wang 92: this code keeps the digits of Fortran PARSEC (V_xc.pw92_lda), which differ "
        "from libxc's LDA_C_PW_MOD in the 6th-9th digit of a few constants.",
    ),
    "pbe": (
        "libxc:gga_x_pbe+gga_c_pbe",
        "PBE: the exchange is identical; the correlation uses the PW92 A = (1 - ln 2)/pi**2 = "
        "0.0310906908 (libxc's PW_MOD: 0.0310907), 3e-7 of the local correlation, and the spin "
        "interpolation f''(0) of the code's digits.",
    ),
    "pw91": (
        "libxc:gga_x_pw91+gga_c_pw91",
        "PW91: Fortran PARSEC's digits (V_xc.pw91), libxc's differ in the 6th-9th digit of a few constants.",
    ),
    "b3lyp": (
        "libxc:hyb_gga_xc_b3lyp",
        "B3LYP with VWN-RPA (Gaussian's, libxc's): the same definition (V_xc.hybrid).",
    ),
    "pbe0": (
        "libxc:hyb_gga_xc_pbeh",
        "PBE0: as PBE (the PW92 A of the correlation).",
    ),
    "hse06": (
        "libxc:hyb_gga_xc_hse06",
        "HSE06: libxc's full-range part is the wPBEh hole at w = 0, it rescales s after Henderson et al. "
        "2009 and derives the hole constants from A..D; this code follows Quantum ESPRESSO and Heyd's "
        "published digits (V_xc.hybrid, V_xc.wpbe).",
    ),
}


# ---------------------------------------------------------------------------
# Specs (no library needed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibxcSpec:
    """A parsed ``libxc:`` spec: component names and the FHC choice (``None`` = libxc's default)."""

    names: tuple[str, ...]
    enforce_fhc: bool | None = None
    density_threshold: float | None = None
    tau_threshold: float | None = None

    @property
    def text(self) -> str:
        suffix = "" if self.enforce_fhc is None else f";fhc={'on' if self.enforce_fhc else 'off'}"
        if self.density_threshold is not None:
            suffix += f";threshold={float(self.density_threshold)!r}"
        if self.tau_threshold is not None:
            suffix += f";tau={float(self.tau_threshold)!r}"
        return LIBXC_PREFIX + "+".join(self.names) + suffix

    @property
    def family(self) -> str:
        """The family of the sum: ``lda`` < ``gga`` < ``mgga``."""

        families = [_NAME.match(name).group(2) for name in self.names]
        return max(families, key=_FAMILY_ORDER.__getitem__)

    @property
    def hybrid_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.names if name.startswith("hyb_"))

    @property
    def is_hybrid(self) -> bool:
        return bool(self.hybrid_names)

    @property
    def is_meta_gga(self) -> bool:
        return self.family == "mgga"


def is_libxc_functional(xc_functional) -> bool:
    """Whether ``xc_functional`` (an ``SCFSettings`` value) names libxc functionals."""

    return isinstance(xc_functional, str) and xc_functional.startswith(LIBXC_PREFIX)


@lru_cache(maxsize=256)
def parse_libxc_spec(text: str) -> LibxcSpec:
    """The components of a ``libxc:`` spec (syntax only; ``ValueError`` otherwise)."""

    match = _SPEC.match(str(text))
    if match is None:
        raise ValueError(
            f"{text!r} is not a libxc functional of the form "
            "'libxc:<name>[+<name>...][;fhc=on|off][;threshold=<t>][;tau=<t>]' "
            "with libxc's lowercase names (gga_x_pbe, hyb_gga_xc_b3lyp)"
        )
    names = tuple(match.group(1).split("+"))
    for name in names:
        if _NAME.match(name) is None:
            raise ValueError(
                f"{name!r} is not the name of a libxc functional of the families lda, gga, mgga or their "
                "hybrids (libxc's names begin with lda_, gga_, mgga_ or hyb_)"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"{text!r} names a libxc functional twice")
    fhc = match.group(2)

    def positive(group: int, what: str):
        if match.group(group) is None:
            return None
        try:
            value = float(match.group(group))
        except ValueError as error:
            raise ValueError(f"{text!r}: the {what} is not a number") from error
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{text!r}: the {what} must be positive")
        return value

    return LibxcSpec(
        names=names,
        enforce_fhc=None if fhc is None else fhc == "on",
        density_threshold=positive(3, "density threshold"),
        tau_threshold=positive(4, "tau threshold"),
    )


def libxc_pseudopotential_functional(text: str) -> str:
    """``ca`` for an LDA (not hybrid), ``pbe`` otherwise: the pseudopotentials a libxc run reads."""

    spec = parse_libxc_spec(text)
    return "ca" if spec.family == "lda" and not spec.is_hybrid else "pbe"


# ---------------------------------------------------------------------------
# The registry (from the library)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LibxcFunctionalInfo:
    """One libxc functional as libxc reports it.

    ``alpha``, ``beta``, ``omega``: ``xc_hyb_cam_coef`` (``omega`` bohr**-1);
    ``nlc_b``, ``nlc_c``: ``xc_nlc_coef`` (VV10).  ``family`` is that of the
    semilocal part (``lda``, ``gga``, ``mgga``), ``hybrid`` whether libxc
    files it as a hybrid.
    """

    number: int
    name: str
    description: str
    kind: str
    family: str
    hybrid: bool
    flags: int
    alpha: float
    beta: float
    omega: float
    nlc_b: float
    nlc_c: float
    references: tuple[str, ...]
    dois: tuple[str, ...]
    ext_params: tuple[tuple[str, float, str], ...]

    @property
    def needs_tau(self) -> bool:
        return bool(self.flags & XC_FLAGS_NEEDS_TAU)

    @property
    def needs_laplacian(self) -> bool:
        return bool(self.flags & XC_FLAGS_NEEDS_LAPLACIAN)

    @property
    def development(self) -> bool:
        return bool(self.flags & XC_FLAGS_DEVELOPMENT)

    @property
    def yukawa(self) -> bool:
        return bool(self.flags & (XC_FLAGS_HYB_CAMY | XC_FLAGS_HYB_LCY))

    @property
    def vv10(self) -> bool:
        return bool(self.flags & XC_FLAGS_VV10) or self.nlc_c != 0.0

    @property
    def empirical_dispersion(self) -> bool:
        """A functional fitted together with an empirical dispersion term (-D, -D3) that libxc does not hold.

        libxc has no flag for it; its names end in ``_d`` or ``_d<n>`` (``gga_xc_b97_d``,
        ``hyb_gga_xc_wb97x_d``, ``hyb_gga_xc_wb97x_d3``, ``gga_x_ssb_d``, ``mgga_xc_otpss_d``).
        """

        return re.search(r"_d\d?$", self.name) is not None

    @property
    def has_exact_exchange(self) -> bool:
        return self.hybrid and (self.alpha != 0.0 or self.beta != 0.0)

    def refusal(self) -> str | None:
        """Why this code does not run the functional, ``None`` where it does."""

        name = f"libxc's {self.name} ({self.description})"
        if self.kind == "kinetic":
            return f"{name} is a kinetic-energy functional, not an exchange-correlation functional"
        if self.family not in _FAMILY_ORDER:
            return f"{name} belongs to the libxc family {self.family!r}; LDA, GGA, meta-GGA and their hybrids run"
        if not self.flags & XC_FLAGS_3D:
            return f"{name} is a functional of one or two dimensions"
        if not (self.flags & XC_FLAGS_HAVE_EXC and self.flags & XC_FLAGS_HAVE_VXC):
            return f"{name} has no energy or no potential in libxc; the SCF needs both"
        if self.vv10:
            return (
                f"{name} includes the nonlocal VV10 correlation (b = {self.nlc_b:g}, C = {self.nlc_c:g}), "
                "which is not implemented"
            )
        if self.empirical_dispersion:
            return (
                f"{name} is fitted together with an empirical dispersion term (-D) that is not part of libxc "
                "and not implemented here; run without it, it would not be the named functional"
            )
        if self.yukawa:
            return (
                f"{name} screens its exact exchange with a Yukawa kernel exp(-omega r)/r (CAMY/LCY), which the "
                "exchange operator of this code does not have (erf/erfc kernels only)"
            )
        return None

    @property
    def summary(self) -> str:
        """One line: number, name, family, hybrid coefficients, needs."""

        family = ("hybrid " if self.hybrid else "") + {"lda": "LDA", "gga": "GGA", "mgga": "meta-GGA"}.get(
            self.family, self.family
        )
        parts = [f"{self.number:4d} {self.name}: {self.description} [{self.kind}, {family}"]
        if self.hybrid:
            parts.append(f", CAM alpha={self.alpha:g} beta={self.beta:g} omega={self.omega:g}")
        if self.needs_tau:
            parts.append(", tau")
        if self.needs_laplacian:
            parts.append(", laplacian")
        if self.development:
            parts.append(", development")
        parts.append("]")
        refusal = self.refusal()
        return "".join(parts) + ("" if refusal is None else "  (refused)")


@lru_cache(maxsize=None)
def _info(number: int) -> LibxcFunctionalInfo:
    data = load_libxc().data(number)
    family, hybrid = FAMILIES.get(data.family_code, ("unknown", False))
    return LibxcFunctionalInfo(
        number=data.number,
        name=data.name,
        description=data.description,
        kind=data.kind,
        family=family,
        hybrid=hybrid,
        flags=data.flags,
        alpha=data.alpha,
        beta=data.beta,
        omega=data.omega,
        nlc_b=data.nlc_b,
        nlc_c=data.nlc_c,
        references=data.references,
        dois=data.dois,
        ext_params=data.ext_params,
    )


def libxc_functional_number(item) -> int:
    """libxc's number of a name (``GGA_X_PBE``, ``xc_gga_x_pbe``, ``gga_x_pbe``) or of a number."""

    library = load_libxc()
    text = str(item).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        number = int(text)
        if library.name(number) is None:
            raise ValueError(f"libxc {library.version_string} has no functional number {number}")
        return number
    key = text.lower()
    if key.startswith("xc_"):
        key = key[3:]
    number = library.number(key)
    if number < 0:
        raise ValueError(
            f"libxc {library.version_string} has no functional {text!r} "
            "(python -m parsec_python.V_xc.libxc --list prints the names)"
        )
    return number


def libxc_functional_info(item) -> LibxcFunctionalInfo:
    """The registry entry of a libxc functional given by name or number."""

    return _info(libxc_functional_number(item))


def available_libxc_functionals() -> tuple[LibxcFunctionalInfo, ...]:
    """Every functional of the loaded libxc, by number."""

    return tuple(_info(number) for number in load_libxc().numbers())


def canonical_libxc_functional(
    items: Iterable,
    *,
    enforce_fhc: bool | None = None,
    density_threshold: float | None = None,
    tau_threshold: float | None = None,
) -> str:
    """The ``libxc:`` spec of the functionals ``items`` (names or numbers), checked against libxc.

    ``ValueError`` for an unknown functional or one given twice;
    :class:`LibxcRefusal` for one that :meth:`LibxcFunctionalInfo.refusal`
    names and for two hybrid components (their exact exchange would have to
    be added).  A development functional of libxc runs with a warning.
    """

    numbers = []
    for item in items:
        number = libxc_functional_number(item)
        if number in numbers:
            raise ValueError(f"libxc functional {_info(number).name} is given twice")
        numbers.append(number)
    if not numbers:
        raise ValueError("no libxc functional is given")
    infos = [_info(number) for number in numbers]
    for info in infos:
        reason = info.refusal()
        if reason is not None:
            raise LibxcRefusal(reason)
        if info.development:
            warnings.warn(
                f"libxc flags {info.name} ({info.description}) as a functional under development",
                UserWarning,
                stacklevel=2,
            )
    hybrids = [info for info in infos if info.hybrid]
    if len(hybrids) > 1:
        raise LibxcRefusal(
            "two libxc hybrids in one functional ("
            + ", ".join(info.name for info in hybrids)
            + "): give one hybrid and, if needed, semilocal components"
        )
    names = tuple(info.name for info in infos)
    for name in names:
        if _NAME.match(name) is None:
            raise ValueError(f"libxc's name {name!r} does not follow the family prefixes this code reads")
    for value, what in ((density_threshold, "density"), (tau_threshold, "tau")):
        if value is not None and not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"the libxc {what} threshold must be positive")
    return LibxcSpec(
        names=names, enforce_fhc=enforce_fhc, density_threshold=density_threshold, tau_threshold=tau_threshold
    ).text


def libxc_components(text: str) -> tuple[LibxcFunctionalInfo, ...]:
    """The registry entries of the components of a spec (checked as :func:`canonical_libxc_functional`)."""

    spec = parse_libxc_spec(text)
    infos = []
    for name in spec.names:
        info = libxc_functional_info(name)
        reason = info.refusal()
        if reason is not None:
            raise LibxcRefusal(reason)
        infos.append(info)
    return tuple(infos)


def describe_libxc_functional(text: str) -> str:
    """One line for output files: libxc's version, the components and their descriptions."""

    spec = parse_libxc_spec(text)
    library = load_libxc()
    parts = [f"{info.name.upper()} ({info.description})" for info in libxc_components(text)]
    fhc = ""
    if spec.is_meta_gga:
        semilocal = LibxcSemilocal(text)
        if semilocal.tau_minimum is not None:
            # The floors of the grid layer (V_xc.meta_gga.weizsaecker_floor).
            fhc += f", tau floor {semilocal.tau_minimum:g} Hartree/bohr^3"
            if semilocal.tau_floor == "weizsaecker":
                why = "fhc=on" if spec.enforce_fhc else "the functional has a kink at tau_W"
                fhc += f", tau >= tau_W floor (Fermi-hole curvature; {why})"
            elif spec.enforce_fhc is False:
                fhc += ", no tau_W floor (fhc=off)"
    if spec.density_threshold is not None:
        fhc += f", density threshold {spec.density_threshold:g}"
    return f"libxc {library.version_string}: " + " + ".join(parts) + fhc


def libxc_citation_lines(text: str) -> list[str]:
    """The lines of an output file: the components, libxc's version and the paper to cite."""

    library = load_libxc()
    lines = [f" {describe_libxc_functional(text)}"]
    for info in libxc_components(text):
        for reference in info.references:
            lines.append(f"   {info.name.upper()}: {reference}")
    lines.append(f" libxc {library.version_string} ({library.path}): {library.reference} doi:{library.reference_doi}")
    return lines


# ---------------------------------------------------------------------------
# The pointwise evaluator
# ---------------------------------------------------------------------------


def _module_of(array):
    """``(numpy, None)`` for host arrays, ``(cupy, device id)`` for CuPy arrays."""

    if hasattr(array, "__cuda_array_interface__"):
        import cupy

        # The device of the array's memory (``ndarray.device`` may name the current one).
        return cupy, int(array.data.device_id)
    return np, None


def _on_device(xp, device):
    """The device context of CuPy arrays on ``device`` (CuPy copies an array of another device to the current one)."""

    from contextlib import nullcontext

    return nullcontext() if device is None else xp.cuda.Device(device)


def _pointer(array) -> int:
    if hasattr(array, "__cuda_array_interface__"):
        return int(array.data.ptr)
    return int(array.ctypes.data)


class _BareFunctional:
    """An evaluator's functional without the floors of the grid layer (the probe of a kink at ``tau_W``)."""

    def __init__(self, evaluator) -> None:
        self._evaluator = evaluator
        self.family = evaluator.family
        self.needs_tau = evaluator.needs_tau
        self.needs_laplacian = evaluator.needs_laplacian

    def unpolarized(self, *arguments):
        return self._evaluator.unpolarized(*arguments)

    def polarized(self, *arguments):
        return self._evaluator.polarized(*arguments)


class LibxcSemilocal:
    """The semilocal evaluator of a ``libxc:`` spec (``SemilocalXC`` with the meta-GGA extension).

    ``family`` is ``lda``, ``gga`` or ``mgga`` (the highest of the
    components); a meta-GGA has ``needs_tau`` and ``needs_laplacian`` and
    fixed-arity methods with ``None`` placeholders (the meta-GGA extension
    of ``SemilocalXC`` in ``V_xc/semilocal.py``):

    * ``unpolarized(rho, sigma[, tau, lapl])`` -> ``(e, vrho, vsigma)``
      for LDA and GGA, ``(e, vrho, vsigma, vtau, vlapl)`` for a meta-GGA
      (``vlapl`` ``None`` unless ``needs_laplacian``; ``vtau`` ``None`` unless
      ``needs_tau``);
    * ``polarized(rho_u, rho_d, s_uu, s_ud, s_dd[, tau_u, tau_d, lapl_u, lapl_d])``
      -> six arrays, or ten for a meta-GGA.

    Hartree atomic units, energy per volume ``e = zk (rho_u + rho_d)``,
    ``sigma_ab = grad n_a . grad n_b``, ``tau_s = 1/2 sum_i f_i |grad psi_i|**2``.
    An LDA returns ``vsigma = 0``.  NumPy arrays run on the host, CuPy arrays
    on their device (``supports_device``: a CUDA build of libxc >= 7.1).
    """

    def __init__(self, spec: str) -> None:
        parsed = parse_libxc_spec(spec)
        self.spec = parsed.text
        # fhc=on: the grid layer floors tau at tau_W (tau_floor); libxc's own clamp is never switched on.
        self.enforce_fhc = bool(parsed.enforce_fhc)
        # None: the default (the tau_W floor where the functional has a kink there); False: never.
        self.fhc_request = parsed.enforce_fhc
        self.density_threshold = parsed.density_threshold
        self.tau_threshold = parsed.tau_threshold
        self.functionals = libxc_components(spec)
        self.family = max((info.family for info in self.functionals), key=_FAMILY_ORDER.__getitem__)
        self.needs_tau = any(info.needs_tau for info in self.functionals)
        self.needs_laplacian = any(info.needs_laplacian for info in self.functionals)

    @property
    def supports_device(self) -> bool:
        return load_libxc().device_capable

    def tau_floor_value(self, polarized: bool = False) -> float:
        """The absolute floor of ``tau`` of a channel (``;tau=`` or :data:`DEFAULT_TAU_THRESHOLD`)."""

        if self.tau_threshold is not None:
            return float(self.tau_threshold)
        return DEFAULT_TAU_THRESHOLD

    @property
    def tau_minimum(self) -> float | None:
        """The absolute floor of ``tau`` of each channel in the grid layer (a meta-GGA that needs ``tau``)."""

        return self.tau_floor_value() if self.family == "mgga" and self.needs_tau else None

    @property
    def tau_floor(self) -> str | None:
        """``"weizsaecker"``: the grid layer floors ``tau`` at ``tau_W`` (``fhc=on``, or a kink at ``tau_W``)."""

        if not (self.family == "mgga" and self.needs_tau) or self.fhc_request is False:
            return None
        if self.fhc_request or self.kink_at_weizsaecker:
            return "weizsaecker"
        return None

    @property
    def kink_at_weizsaecker(self) -> bool:
        """Whether the derivatives of the functional jump at ``tau = tau_W`` (libxc's clamp ``z <= 1`` inside).

        A probe of the bare functional
        (:func:`parsec_python.V_xc.meta_gga.smooth_at_weizsaecker`), once
        per process for a list of components: TPSS-type correlation clamps
        ``z = tau_W/tau`` at 1 (``vtau = 0`` and another ``vsigma`` above),
        SCAN, r2SCAN and TPSS exchange are smooth there.
        """

        if not (self.family == "mgga" and self.needs_tau):
            return False
        key = (tuple(info.name for info in self.functionals), self.density_threshold)
        known = _KINKED.get(key)
        if known is None:
            from .meta_gga import smooth_at_weizsaecker

            known = not smooth_at_weizsaecker(_BareFunctional(self))
            _KINKED[key] = known
        return known

    @property
    def library(self):
        return load_libxc()

    def describe(self) -> str:
        return describe_libxc_functional(self.spec)

    def __repr__(self) -> str:
        return f"LibxcSemilocal({self.spec!r})"

    # -- evaluation ------------------------------------------------------------

    def _handles(self, polarized: bool, device: int | None):
        library = load_libxc()
        return [
            (
                info,
                library.functional(
                    info.number,
                    polarized=polarized,
                    device=device,
                    enforce_fhc=False,  # libxc's clamp never: fhc=on is the floor of the grid layer
                    density_threshold=self.density_threshold,
                    # libxc's own floor below that of the grid layer (tau_minimum).
                    tau_threshold=min(LIBXC_TAU_THRESHOLD, self.tau_floor_value()) if info.needs_tau else None,
                ),
            )
            for info in self.functionals
        ]

    def _meta_inputs(self, xp, count: int, tau, lapl, columns: int):
        if self.family != "mgga":
            if tau is not None or lapl is not None:
                raise ValueError(f"{self.spec} is not a meta-GGA: it takes no tau or laplacian")
            return None, None
        if self.needs_tau and tau is None:
            raise ValueError(f"{self.spec} needs the kinetic-energy density tau")
        if self.needs_laplacian and lapl is None:
            raise ValueError(f"{self.spec} needs the laplacian of the density")
        zeros = None
        if tau is None or lapl is None:
            zeros = xp.zeros(count * columns, dtype=np.float64)
        return (zeros if tau is None else tau), (zeros if lapl is None else lapl)

    def evaluate_layout(self, rho, sigma, tau=None, lapl=None, *, polarized: bool):
        """The sum over the components in libxc's layout: ``(e, vrho, vsigma, vtau, vlapl)``.

        ``rho`` is ``(np,)`` or, ``polarized``, ``(np, 2)``; ``sigma`` ``(np,)``
        or ``(np, 3)``; ``tau`` and ``lapl`` like ``rho`` (meta-GGA only).
        Returns the energy per volume ``e = sum_c zk_c (rho_u + rho_d)``
        (``(np,)``) and the derivatives in the layout of their variables
        (``vtau``, ``vlapl`` ``None`` for LDA/GGA).  C-contiguous float64
        NumPy arrays are evaluated on the host, CuPy arrays on their device
        (whatever device is current: the outputs are allocated on that of
        ``rho``).
        """

        xp, device = _module_of(rho)
        if device is None:
            return self._evaluate_layout(xp, None, rho, sigma, tau, lapl, polarized)
        with xp.cuda.Device(device):
            return self._evaluate_layout(xp, device, rho, sigma, tau, lapl, polarized)

    def _evaluate_layout(self, xp, device, rho, sigma, tau, lapl, polarized: bool):
        columns = 2 if polarized else 1
        rho = xp.ascontiguousarray(rho, dtype=np.float64)
        count = int(rho.shape[0])
        if rho.size != count * columns:
            raise ValueError("rho does not have libxc's layout")
        sigma = xp.ascontiguousarray(sigma, dtype=np.float64)
        if sigma.size != count * (3 if polarized else 1):
            raise ValueError("rho and sigma differ in size")
        if tau is not None:
            tau = xp.ascontiguousarray(tau, dtype=np.float64)
        if lapl is not None:
            lapl = xp.ascontiguousarray(lapl, dtype=np.float64)
        tau_in, lapl_in = self._meta_inputs(xp, count, tau, lapl, columns)
        meta = self.family == "mgga"
        clipped = None
        if meta and self.needs_tau and count:
            # The binding's floor, the same on both sides: libxc applies its tau threshold on the
            # host only. vtau = 0 where it acts (the derivative of e(max(tau, t))).
            floor = min(LIBXC_TAU_THRESHOLD, self.tau_floor_value())
            clipped = tau_in < floor
            tau_in = xp.where(clipped, floor, tau_in)
        rho_shape = (count, 2) if polarized else (count,)
        sigma_shape = (count, 3) if polarized else (count,)
        energy = xp.zeros(count, dtype=np.float64)
        vrho = xp.zeros(rho_shape, dtype=np.float64)
        vsigma = xp.zeros(sigma_shape, dtype=np.float64)
        vtau = xp.zeros(rho_shape, dtype=np.float64) if meta else None
        vlapl = xp.zeros(rho_shape, dtype=np.float64) if meta else None
        if count:
            total = rho[:, 0] + rho[:, 1] if polarized else rho
            zk = xp.empty(count, dtype=np.float64)
            dr = xp.empty(rho_shape, dtype=np.float64)
            ds = xp.empty(sigma_shape, dtype=np.float64)
            dt = xp.empty(rho_shape, dtype=np.float64) if meta else None
            dl = xp.empty(rho_shape, dtype=np.float64) if meta else None
            for info, handle in self._handles(polarized, device):
                if info.family == "lda":
                    handle.lda(count, _pointer(rho), _pointer(zk), _pointer(dr))
                elif info.family == "gga":
                    handle.gga(count, _pointer(rho), _pointer(sigma), _pointer(zk), _pointer(dr), _pointer(ds))
                    vsigma += ds
                else:
                    # libxc writes vtau/vlapl only for a component that needs them.
                    dt.fill(0.0)
                    dl.fill(0.0)
                    handle.mgga(
                        count,
                        _pointer(rho),
                        _pointer(sigma),
                        _pointer(lapl_in),
                        _pointer(tau_in),
                        _pointer(zk),
                        _pointer(dr),
                        _pointer(ds),
                        _pointer(dl),
                        _pointer(dt),
                    )
                    vsigma += ds
                    vtau += dt
                    vlapl += dl
                energy += zk * total
                vrho += dr
            if clipped is not None:
                vtau = xp.where(clipped, 0.0, vtau)
        return energy, vrho, vsigma, vtau, vlapl

    def unpolarized(self, rho, sigma, tau=None, lapl=None):
        xp, device = _module_of(rho)
        with _on_device(xp, device):
            return self._unpolarized(xp, rho, sigma, tau, lapl)

    def _unpolarized(self, xp, rho, sigma, tau, lapl):
        energy, vrho, vsigma, vtau, vlapl = self.evaluate_layout(
            xp.asarray(rho, dtype=np.float64).reshape(-1),
            xp.asarray(sigma, dtype=np.float64).reshape(-1),
            None if tau is None else xp.asarray(tau, dtype=np.float64).reshape(-1),
            None if lapl is None else xp.asarray(lapl, dtype=np.float64).reshape(-1),
            polarized=False,
        )
        if self.family != "mgga":
            return energy, vrho, vsigma
        return (
            energy,
            vrho,
            vsigma,
            vtau if self.needs_tau else None,
            vlapl if self.needs_laplacian else None,
        )

    def polarized(
        self,
        rho_up,
        rho_down,
        sigma_uu,
        sigma_ud,
        sigma_dd,
        tau_up=None,
        tau_down=None,
        lapl_up=None,
        lapl_down=None,
    ):
        xp, device = _module_of(rho_up)
        with _on_device(xp, device):
            return self._polarized(
                xp, rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down, lapl_up, lapl_down
            )

    def _polarized(self, xp, rho_up, rho_down, sigma_uu, sigma_ud, sigma_dd, tau_up, tau_down, lapl_up, lapl_down):
        def flat(values):
            return xp.asarray(values, dtype=np.float64).reshape(-1)

        def pair(first, second, label):
            if first is None and second is None:
                return None
            if first is None or second is None:
                raise ValueError(f"give {label} of both channels")
            return xp.stack([flat(first), flat(second)], axis=1)

        energy, vrho, vsigma, vtau, vlapl = self.evaluate_layout(
            xp.stack([flat(rho_up), flat(rho_down)], axis=1),
            xp.stack([flat(sigma_uu), flat(sigma_ud), flat(sigma_dd)], axis=1),
            pair(tau_up, tau_down, "tau"),
            pair(lapl_up, lapl_down, "the laplacian"),
            polarized=True,
        )

        def column(array, index):
            return xp.ascontiguousarray(array[:, index])

        outputs = (
            energy,
            column(vrho, 0),
            column(vrho, 1),
            column(vsigma, 0),
            column(vsigma, 1),
            column(vsigma, 2),
        )
        if self.family != "mgga":
            return outputs
        return outputs + (
            column(vtau, 0) if self.needs_tau else None,
            column(vtau, 1) if self.needs_tau else None,
            column(vlapl, 0) if self.needs_laplacian else None,
            column(vlapl, 1) if self.needs_laplacian else None,
        )


def libxc_meta_gga_evaluator(scf_settings):
    """The meta-GGA provider of :mod:`parsec_python.V_xc.meta_gga`: a libxc meta-GGA's evaluator, else ``None``.

    For the settings of a run whose ``xc_functional`` is a libxc spec of
    family ``mgga`` (a hybrid included: its semilocal part) the
    :class:`LibxcSemilocal` of the spec; ``None`` for every other functional
    (decided from the spec alone, without the library).
    """

    functional = getattr(scf_settings, "xc_functional", None)
    if not is_libxc_functional(functional) or not parse_libxc_spec(functional).is_meta_gga:
        return None
    return libxc_semilocal(functional)


@lru_cache(maxsize=64)
def libxc_semilocal(spec: str) -> LibxcSemilocal:
    """The evaluator of ``spec`` (one per spec and process)."""

    return LibxcSemilocal(spec)


# ---------------------------------------------------------------------------
# Hybrids
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def libxc_hybrid_functional(spec: str):
    """The :class:`~parsec_python.V_xc.hybrid.HybridFunctional` of a libxc hybrid spec.

    The exact exchange takes libxc's CAM coefficients of the hybrid
    component (``E_x^HF = alpha E_x[1/r] + beta E_x[erfc(omega r)/r]``), the
    semilocal part is :class:`LibxcSemilocal` of the whole spec (libxc's
    hybrid energy is its semilocal part only).  The SCF before the first
    exchange step uses this code's PBE (``initial_functional``), as for the
    hybrids of :mod:`parsec_python.V_xc.hybrid`.
    """

    from .hybrid import HybridFunctional

    parsed = parse_libxc_spec(spec)
    components = libxc_components(spec)
    hybrids = [info for info in components if info.hybrid]
    if len(hybrids) != 1:
        raise ValueError(f"{spec} has {len(hybrids)} hybrid components; a libxc hybrid has one")
    hybrid = hybrids[0]
    alpha, beta, omega = float(hybrid.alpha), float(hybrid.beta), float(hybrid.omega)
    if beta == 0.0:
        omega_value = None
    elif omega <= 0.0:
        raise ValueError(f"libxc's {hybrid.name} has beta = {beta:g} but no positive omega")
    else:
        omega_value = omega
    names = " + ".join(info.name.upper() for info in components)
    library = load_libxc()
    return HybridFunctional(
        label=parsed.text,
        name=f"libxc {names}",
        alpha=alpha,
        beta=beta,
        omega=omega_value,
        semilocal=libxc_semilocal(parsed.text),
        spin_supported=True,
        initial_functional="pbe",
        references=tuple(ref for info in components for ref in info.references)
        + (f"libxc {library.version_string}: {library.reference}",),
        libxc=", ".join(f"{info.name.upper()} ({info.number})" for info in components),
        variants=(
            "libxc's definition: the semilocal part from libxc, the exact exchange from libxc's CAM "
            "coefficients through this code's exchange operator (PARSEC 1.5 evaluates libxc hybrids "
            "as plain GGAs without exact exchange)."
        ),
    )


__all__ = [
    "LIBXC_PREFIX",
    "LibxcFunctionalInfo",
    "LibxcRefusal",
    "LibxcSemilocal",
    "LibxcSpec",
    "LibxcUnavailable",
    "DEFAULT_TAU_THRESHOLD",
    "LIBXC_TAU_THRESHOLD",
    "OWN_FUNCTIONAL_COUNTERPARTS",
    "available_libxc_functionals",
    "canonical_libxc_functional",
    "describe_libxc_functional",
    "is_libxc_functional",
    "libxc_citation_lines",
    "libxc_components",
    "libxc_functional_info",
    "libxc_functional_number",
    "libxc_hybrid_functional",
    "libxc_meta_gga_evaluator",
    "libxc_pseudopotential_functional",
    "libxc_semilocal",
    "parse_libxc_spec",
]


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m parsec_python.V_xc.libxc",
        description="What libxc this code finds, and its functionals.",
    )
    parser.add_argument("names", nargs="*", help="functionals (names or numbers) to describe")
    parser.add_argument("--list", action="store_true", help="list every functional of the library")
    arguments = parser.parse_args(argv)
    try:
        library = load_libxc()
    except LibxcUnavailable as error:
        print(error)
        return 1
    print(f"libxc {library.version_string} at {library.path}")
    print(
        "  host evaluation: " + ("yes" if library.host_capable else "no")
        + "; device evaluation (CUDA build >= 7.1): " + ("yes" if library.device_capable else "no")
    )
    print(f"  {library.reference} doi:{library.reference_doi}")
    if arguments.list:
        for info in available_libxc_functionals():
            print(info.summary)
    for item in arguments.names:
        info = libxc_functional_info(item)
        print(info.summary)
        for reference in info.references:
            print(f"      {reference}")
        for name, value, description in info.ext_params:
            print(f"      parameter {name} = {value:g}: {description}")
        refusal = info.refusal()
        if refusal:
            print(f"      refused: {refusal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
