"""Molecular dynamics input of PARSEC.

Translation of the molecular dynamics part of subroutine ``usrinput``
(``usrinputfile.F90``), of the checks subroutine ``initial``
(``initial.f90``) applies to it, and of the pieces of the ESDF reader
(``esdf_mod.f90``) those keywords go through.

Keywords, in the spelling of ``esdf_key_mod.f90``.  ESDF ignores case and the
characters ``.`` and ``-`` in a keyword, but not ``_``.

====================  =========  ==========  =====================================
keyword               default    unit        meaning
====================  =========  ==========  =====================================
Molecular_Dynamics    .false.    logical     switch molecular dynamics on
Restart_mode          .false.    logical     continue from ``mdinit.dat``
Cooling_Method        stair      word        ``stair``, ``linear`` or ``log``
Tinit                 500        K           initial temperature
T_Final               500        K           final temperature
T_Step                150        K           temperature step of ``stair``
Time_Step             150        a.u. (!)    time step, see below
Friction_Coefficient  0          1/a.u.      Langevin friction; below 1e-6 the
                                             run is microcanonical
Step_num              250        integer     number of time steps
rng_seed              0          integer     seed, 0 means "from the clock"
test_rng              .false.    logical     use PARSEC's own simple generator
====================  =========  ==========  =====================================

Unit rules (``esdf_physical``): a number may be followed by a unit of the
same dimension as the unit the program asks for; the value is multiplied by
``phy_u(given)/phy_u(asked)``.  ESDF files the kelvin under *energy*, so a
temperature may also be given in ``eV``, ``Ry``, ``hartree``, ``cm-1`` and so
on.  PARSEC asks for ``Time_Step`` in ``ps`` and then uses the number as
atomic units of time (the output file prints it as ``[au]`` and multiplies by
2.4188843e-5 to show picoseconds).  A bare ``Time_Step 150`` is therefore 150
a.u. = 3.6 fs, while ``Time_Step 3.6 fs`` becomes 0.0036 a.u.  This mismatch
is reproduced: :attr:`MolecularDynamicsSettings.time_step` is the number
PARSEC integrates with, in atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping

import numpy as np

#: ``timeaups`` of ``const.f90``: one atomic unit of time in picoseconds.
TIME_AU_TO_PS = 2.4188843e-05

#: ``mol_dynamic%nrandom`` set in ``molecular_dynamic_turn_on``.
RANDOM_BUFFER_SIZE = 10000

#: ``MD_STAIR``, ``MD_LINEAR`` and ``MD_LOG`` of ``const.f90``.
COOLING_CODES = {"stair": 1, "linear": 2, "log": 3}

# moldyn.f90 chooses between Newtonian and Langevin dynamics with
# ``beta < 1.0d-6``.  The number of degrees of freedom of the first
# velocity rescaling, and the message in parsec.out, use ``beta < 10d-6``.
MICROCANONICAL_FRICTION = 1.0e-6
RESCALE_FRICTION = 10.0e-6

# esdf_mod.f90, phy_u: the unit in SI for the dimensions 'e' (energy, which
# contains the kelvin) and 't' (time).  Keys are reduced as esdf_reduce does.
_ESDF_ENERGY_UNITS = {
    "j": 1.0,
    "erg": 1.0e-7,
    "ev": 1.60219e-19,
    "mev": 1.60219e-22,
    "ry": 2.17991e-18,
    "mry": 2.17991e-21,
    "hartree": 4.35982e-18,
    "kcal/mol": 6.94780e-21,
    "mhartree": 4.35982e-21,
    "kj/mol": 1.6606e-21,
    "hz": 6.6262e-34,
    "thz": 6.6262e-22,
    "cm1": 1.986e-23,
    "cm^1": 1.986e-23,
    "cm**1": 1.986e-23,
    "k": 1.38066e-23,
}
_ESDF_TIME_UNITS = {
    "s": 1.0,
    "ns": 1.0e-9,
    "ps": 1.0e-12,
    "fs": 1.0e-15,
}

_FORTRAN_REAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eEdDqQ][+-]?\d+)?")
_FORTRAN_INTEGER = re.compile(r"[+-]?\d+")


class MolecularDynamicsInputError(ValueError):
    """Molecular dynamics input PARSEC would stop on.

    ``parsec_error_code`` is the value of ``ierr`` in the Fortran program
    (154 from ``usrinput``, 231 and 232 from ``initial``, 155 and 157 for the
    incompatible run modes), or ``None`` when the Fortran program has no code
    for it: a line ESDF cannot read, or a value this module refuses although
    PARSEC would go on with it.
    """

    def __init__(self, message: str, parsec_error_code: int | None = None):
        super().__init__(message)
        self.parsec_error_code = parsec_error_code


def esdf_reduce(text: str) -> str:
    """``esdf_reduce``: lower case, without ``.`` and ``-``, trimmed."""
    return text.lower().replace(".", "").replace("-", "").strip()


def _tokens(raw: str) -> list[str]:
    """Split the remainder of an input line as ESDF does (blank, ``=``, ``:``)."""
    return [token for token in re.split(r"[ \t=:]+", str(raw).strip()) if token]


def _esdf_real(keyword: str, raw: str) -> float:
    tokens = _tokens(raw)
    if not tokens or _FORTRAN_REAL.fullmatch(tokens[0]) is None:
        raise MolecularDynamicsInputError(
            f'Unable to parse "{keyword}": expected a number, got {raw!r}'
        )
    return float(re.sub(r"[dDqQ]", "e", tokens[0]))


def _esdf_integer(keyword: str, raw: str) -> int:
    tokens = _tokens(raw)
    if not tokens or _FORTRAN_INTEGER.fullmatch(tokens[0]) is None:
        raise MolecularDynamicsInputError(
            f'Unable to parse "{keyword}": expected an integer, got {raw!r}'
        )
    return int(tokens[0])


def _esdf_boolean(keyword: str, raw: str) -> bool:
    """``esdf_boolean``: a bare keyword is true; else yes/true/t or no/false/f."""
    tokens = _tokens(raw)
    if not tokens:
        return True
    word = esdf_reduce(tokens[0])
    if word in ("yes", "true", "t"):
        return True
    if word in ("no", "false", "f"):
        return False
    raise MolecularDynamicsInputError(
        f'Unable to parse boolean value of "{keyword}": {raw!r}'
    )


def _esdf_physical(
    keyword: str, raw: str, units: Mapping[str, float], default_unit: str
) -> float:
    """``esdf_physical``: ``number [unit]`` converted to ``default_unit``."""
    value = _esdf_real(keyword, raw)
    tokens = _tokens(raw)
    unit = esdf_reduce(tokens[1]) if len(tokens) > 1 else default_unit
    if unit not in units:
        known = _ESDF_ENERGY_UNITS.keys() | _ESDF_TIME_UNITS.keys()
        if unit in known:
            raise MolecularDynamicsInputError(
                f'"{keyword}": Dimensions Do not match : {unit} vs {default_unit}'
            )
        raise MolecularDynamicsInputError(
            f'"{keyword}": Units not recognised in input file : {unit}'
        )
    # esdf_convfac(from,to) * value, in this order
    return (units[unit] / units[default_unit]) * value


def _truncating_quotient(numerator: int, denominator: int) -> int:
    """Fortran integer division: the quotient is truncated toward zero."""
    quotient = abs(numerator) // abs(denominator)
    return quotient if (numerator >= 0) == (denominator >= 0) else -quotient


@dataclass(frozen=True)
class MolecularDynamicsSettings:
    """Every molecular dynamics input of PARSEC (``type molecular_dynamic``).

    The comment after each field gives the PARSEC keyword and the component
    of ``mol_dynamic`` it fills.  Temperatures are in kelvin.  ``time_step``
    is in atomic units of time (2.4188843e-5 ps) and ``friction_coefficient``
    in inverse atomic units of time; see the module documentation for what
    PARSEC does with a unit written after ``Time_Step``.

    Checks of the Fortran program, raised as
    :class:`MolecularDynamicsInputError` with its error code:

    * 154: ``Cooling_Method`` is not ``stair``, ``linear`` or ``log``
      (checked even if molecular dynamics is off, as in ``usrinput``);
    * 231: ``stair`` with a temperature step of the wrong sign.  ``T_Step``
      is what is *subtracted* per stair, so cooling needs a positive and
      heating a negative step;
    * 232: ``stair`` with ``|T_Step|/2 >= |Tinit - T_Final|`` in a run that
      is not isothermal.

    Checks added here, because the Fortran program would continue and
    produce not-a-number, divide an integer by zero, or integrate nonsense
    (``parsec_error_code`` is ``None``); they apply when molecular dynamics
    is on:

    * all numbers must be finite and ``Step_num`` must not be negative;
    * ``Time_Step`` must be positive;
    * temperatures must not be negative, and ``Tinit`` must be positive
      unless the run is a restart with ``stair`` or ``linear`` cooling
      (a fresh start scales the velocities by ``sqrt(Tinit/T_actual)``,
      which is 0/0 at ``Tinit = 0``; ``log`` divides by ``Tinit``).
    """

    enabled: bool = False  # Molecular_Dynamics -> mol_dynamic%is_on
    restart: bool = False  # Restart_mode -> mol_dynamic%is_restart
    cooling_method: str = "stair"  # Cooling_Method -> mol_dynamic%cool_type
    initial_temperature: float = 500.0  # Tinit [K] -> mol_dynamic%tempi
    final_temperature: float = 500.0  # T_Final [K] -> mol_dynamic%tempf
    temperature_step: float = 150.0  # T_Step [K] -> mol_dynamic%step_tmp
    time_step: float = 150.0  # Time_Step [a.u.] -> mol_dynamic%time_step
    friction_coefficient: float = 0.0  # Friction_Coefficient -> friction_coef
    step_num: int = 250  # Step_num -> mol_dynamic%step_num
    rng_seed: int = 0  # rng_seed -> mol_dynamic%rng_seed
    use_test_rng: bool = False  # test_rng -> mol_dynamic%use_test_rng

    def __post_init__(self) -> None:
        method = esdf_reduce(str(self.cooling_method))
        if method not in COOLING_CODES:
            raise MolecularDynamicsInputError(
                f"unknown molecular dynamics cooling scheme: {self.cooling_method!r}",
                154,
            )
        for name in ("step_num", "rng_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise MolecularDynamicsInputError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "cooling_method", method)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "restart", bool(self.restart))
        object.__setattr__(self, "use_test_rng", bool(self.use_test_rng))
        reals = (
            "initial_temperature",
            "final_temperature",
            "temperature_step",
            "time_step",
            "friction_coefficient",
        )
        for name in reals:
            object.__setattr__(self, name, float(getattr(self, name)))
        if not self.enabled:
            return

        for name in reals:
            if not math.isfinite(getattr(self, name)):
                raise MolecularDynamicsInputError(f"{name} must be finite")
        tempi = self.initial_temperature
        tempf = self.final_temperature
        if method == "stair":
            # initial.f90: tscale = -tstep
            tscale = -self.temperature_step
            if (tempi > tempf and tscale >= 0.0) or (tempi < tempf and tscale <= 0.0):
                raise MolecularDynamicsInputError(
                    "temperature step of wrong sign: T_Step is subtracted at "
                    "every stair, so it must be positive for cooling and "
                    "negative for heating",
                    231,
                )
            if abs(tempi - tempf) > 1.0e-5 and abs(tscale / 2.0) >= abs(tempi - tempf):
                raise MolecularDynamicsInputError(
                    "temperature step too large: it goes far beyond the "
                    "final temperature",
                    232,
                )
        if self.step_num < 0:
            raise MolecularDynamicsInputError("Step_num must not be negative")
        if self.time_step <= 0.0:
            raise MolecularDynamicsInputError("Time_Step must be positive")
        if tempi < 0.0 or tempf < 0.0:
            raise MolecularDynamicsInputError("temperatures must not be negative")
        if tempi == 0.0 and (not self.restart or method == "log"):
            raise MolecularDynamicsInputError(
                "Tinit must be positive: PARSEC scales the initial velocities "
                "by sqrt(Tinit/T_actual) and the log cooling divides by Tinit"
            )

    # ------------------------------------------------------------------
    # construction from the input file
    # ------------------------------------------------------------------
    @classmethod
    def from_keywords(cls, keywords: Mapping[str, str]) -> "MolecularDynamicsSettings":
        """Build the settings from ``{keyword: raw text after the keyword}``.

        The mapping may hold the whole input file; only the molecular
        dynamics keywords are looked at.  Keys are compared as ESDF compares
        labels (case, ``.`` and ``-`` do not matter).  Each value is the rest
        of the input line: a number with an optional unit, a word, or an
        empty string for a bare logical keyword, which means true.

        As in ``usrinput``, ``Restart_mode`` is read only when molecular
        dynamics is on.  If the mapping also contains ``Minimization`` or
        ``Polarizability``, the two compatibility checks of ``usrinput`` are
        applied (errors 155 and 157).
        """
        table = {esdf_reduce(str(key)): str(value) for key, value in keywords.items()}

        def given(keyword: str) -> str | None:
            return table.get(esdf_reduce(keyword))

        values: dict[str, object] = {}
        raw = given("Molecular_Dynamics")
        if raw is not None:
            values["enabled"] = _esdf_boolean("molecular_dynamics", raw)
        raw = given("Cooling_Method")
        if raw is not None:
            # esdf_string returns the rest of the line
            values["cooling_method"] = str(raw).strip(" \t=:")
            if esdf_reduce(values["cooling_method"]) not in COOLING_CODES:
                raise MolecularDynamicsInputError(
                    "unknown molecular dynamics cooling scheme: "
                    + repr(values["cooling_method"]),
                    154,
                )
        for keyword, name in (
            ("Tinit", "initial_temperature"),
            ("T_Final", "final_temperature"),
            ("T_Step", "temperature_step"),
        ):
            raw = given(keyword)
            if raw is not None:
                values[name] = _esdf_physical(
                    esdf_reduce(keyword), raw, _ESDF_ENERGY_UNITS, "k"
                )
        raw = given("Time_Step")
        if raw is not None:
            values["time_step"] = _esdf_physical(
                "time_step", raw, _ESDF_TIME_UNITS, "ps"
            )
        raw = given("Friction_Coefficient")
        if raw is not None:
            values["friction_coefficient"] = _esdf_real("friction_coefficient", raw)
        raw = given("Step_num")
        if raw is not None:
            values["step_num"] = _esdf_integer("step_num", raw)
        raw = given("rng_seed")
        if raw is not None:
            values["rng_seed"] = _esdf_integer("rng_seed", raw)
        raw = given("test_rng")
        if raw is not None:
            values["use_test_rng"] = _esdf_boolean("test_rng", raw)
        raw = given("Restart_mode")
        if raw is not None and values.get("enabled", False):
            values["restart"] = _esdf_boolean("restart_mode", raw)
        # usrinput tests the run modes before initial tests the stair
        minimization = given("Minimization")
        polarizability = given("Polarizability")
        _check_run_mode(
            bool(values.get("enabled", False)),
            "none" if minimization is None else minimization,
            False
            if polarizability is None
            else _esdf_boolean("polarizability", polarizability),
        )
        return cls(**values)

    # ------------------------------------------------------------------
    # quantities PARSEC derives from the input
    # ------------------------------------------------------------------
    @property
    def cool_type(self) -> int:
        """``mol_dynamic%cool_type``: 1 stair, 2 linear, 3 log."""
        return COOLING_CODES[self.cooling_method]

    @property
    def is_microcanonical(self) -> bool:
        """The test of ``moldyn``: ``beta < 1.0d-6`` integrates Newton's equations."""
        return self.friction_coefficient < MICROCANONICAL_FRICTION

    @property
    def time_step_ps(self) -> float:
        """The time step in picoseconds as PARSEC prints it."""
        return self.time_step * TIME_AU_TO_PS

    @property
    def temperature_scale(self) -> float:
        """``mol_dynamic%tscale`` of ``initial.f90``.

        ``stair``: the temperature added per stair, ``-T_Step``.
        ``linear``: the temperature added per step, ``(T_Final-Tinit)/Step_num``.
        ``log``: the factor per step, ``(T_Final/Tinit)**(1/Step_num)``.
        """
        tempi = np.float64(self.initial_temperature)
        tempf = np.float64(self.final_temperature)
        steps = np.float64(self.step_num)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.cooling_method == "stair":
                return float(-self.temperature_step)
            if self.cooling_method == "linear":
                return float((tempf - tempi) / steps)
            return float((tempf / tempi) ** (np.float64(1.0) / steps))

    @property
    def stride(self) -> int:
        """``mol_dynamic%stride``: steps between two stairs (``stair`` only).

        ``usrinput`` computes, in integer arithmetic,
        ``i = (Tinit-T_Final)/T_Step + 1``, ``stride = Step_num/i`` and adds
        one if ``stride*i < Step_num``.  For the other cooling methods PARSEC
        leaves the component unset; zero is returned.  An isothermal run with
        ``T_Step = 0`` is 0/0 in Fortran and ends with ``stride = 1`` on x86;
        that value is returned.
        """
        if self.cooling_method != "stair":
            return 0
        difference = self.initial_temperature - self.final_temperature
        if self.temperature_step == 0.0:
            if difference == 0.0:
                return 1
            raise MolecularDynamicsInputError("temperature step of wrong sign", 231)
        stairs = int(difference / self.temperature_step + 1.0)
        if stairs == 0:
            raise MolecularDynamicsInputError("temperature step of wrong sign", 231)
        stride = _truncating_quotient(self.step_num, stairs)
        if stride * stairs < self.step_num:
            stride += 1
        return stride

    def random_source(
        self,
        generator: np.random.Generator | int | None = None,
        *,
        compiler: str | None = None,
    ):
        """Return the uniform random source ``moldyn`` uses with this input.

        ``test_rng``: PARSEC's own generator started from ``rng_seed``; the
        Python numbers are those of the Fortran program.  Otherwise the
        Fortran program takes the numbers from the compiler's
        ``random_number``; here they come from ``generator`` (a NumPy
        generator, or a seed for one), buffered as PARSEC buffers them.  With
        ``rng_seed > 0`` PARSEC puts that seed before every refill of the
        buffer, and so does the returned source, with
        ``numpy.random.default_rng(rng_seed)``; ``generator`` is then not
        used.

        ``compiler="gfortran"`` asks instead for the numbers a PARSEC built
        by gfortran draws without ``test_rng``.  It needs ``rng_seed > 0``.
        """
        from .random_numbers import BufferedRandom, GfortranRandom, ParsecTestRandom

        if self.use_test_rng:
            return ParsecTestRandom(self.rng_seed)
        if compiler is not None:
            if compiler.lower() != "gfortran":
                raise ValueError(
                    f"no emulation of the random_number of compiler {compiler!r}"
                )
            return GfortranRandom(self.rng_seed)
        return BufferedRandom(generator, rng_seed=self.rng_seed)


def check_run_mode(
    settings: MolecularDynamicsSettings,
    *,
    minimization: str = "none",
    polarizability: bool = False,
) -> None:
    """Compatibility checks of ``usrinput`` between the run modes.

    ``minimization`` is the value of the keyword ``Minimization``.  PARSEC
    cannot move atoms in a polarizability run (error 155) and cannot do
    minimization and molecular dynamics together (error 157).
    """
    _check_run_mode(settings.enabled, minimization, polarizability)


def _check_run_mode(enabled: bool, minimization: str, polarizability: bool) -> None:
    minimizing = esdf_reduce(str(minimization).strip(" \t=:")) != "none"
    if (enabled or minimizing) and polarizability:
        raise MolecularDynamicsInputError(
            "Cannot do polarizability and move atoms", 155
        )
    if enabled and minimizing:
        raise MolecularDynamicsInputError("Cannot do minimization and MD", 157)
