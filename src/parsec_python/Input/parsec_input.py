"""Focused ESDF/PARSEC input reader for isolated CA-LDA single points."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from time import perf_counter
from typing import TYPE_CHECKING, Iterable

import numpy as np

from ..models import (
    DEFAULT_INITIAL_SPIN_POLARIZATION,
    DEFAULT_RELAXATION_MAXIMUM_STEP,
    Atom,
    BandSegment,
    BandStructureSettings,
    ChargedSheet,
    DOSSettings,
    GRID_DATA_FIELDS,
    EigensolverSettings,
    GWSettings,
    GridSettings,
    HartreeSettings,
    InitialDensitySettings,
    MIXING_METHODS,
    MixingSettings,
    NSCFSettings,
    OutputFileSettings,
    RelaxationSettings,
    WavefunctionSelection,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    PeriodicCell,
    PeriodicGridSettings,
    PointCharge,
    UNPOLARIZED_ONLY_XC_FUNCTIONALS,
)
from ..Grid.cell import primitive_lattice, replicate_positions, supercell_lattice
from ..Grid.pbc import fortran_grid_shift
from ..Grid.kpoints import explicit_kpoints, read_kpoint_file
from ..Grid.pbc import periodic_grid_shift, uses_general_periodic_grid
from . import routines as _routines_module
from .routines import RoutineChoice, UseFortranRoutines, decide_routines
from ..MLDensity.field import normalize_density_units
from ..Pseudopotential.potrw import read_martins_wang_header
from ..MolecularDynamics.settings import MolecularDynamicsSettings

if TYPE_CHECKING:
    from ..Hartree.domain import DomainChoice
    from ..MLFF.settings import MLFFSettings


# PARSEC's ESDF table uses 1 bohr = 0.529177 angstrom exactly.
ANGSTROM_TO_BOHR = 1.0 / 0.529177
EV_TO_RYDBERG = 1.0 / 13.605693122994
# Energies of ESDF's table in joule (esdf_mod.f90, phy_u), for the keywords
# whose conversion follows the Fortran exactly (ldaplusu_u, ldaplusu_j).
_ESDF_ENERGY_JOULE = {
    "ry": 2.17991e-18,
    "mry": 2.17991e-21,
    "ev": 1.60219e-19,
    "mev": 1.60219e-22,
    "hartree": 4.35982e-18,
    "ha": 4.35982e-18,
    "mhartree": 4.35982e-21,
}

_FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)
_COMMENT_MARKERS = ("#", ";", "!")
# Correlation_Type labels of usrinputfile.F90:2328-2361 (after esdf_reduce)
# and the functional of SCFSettings.xc_functional each one selects.
_CORRELATION_LABELS = {
    "xa": "xalpha",
    "wi": "wigner",
    "hl": "hedin_lundqvist",
    "ca": "ca",
    "pz": "ca",
    "lda": "ca",
    "pl": "pw92",
    "pw92": "pw92",
    "pwlda": "pw92",
    "pw": "pw91",
    "pw91": "pw91",
    "pwgga": "pw91",
    "pb": "pbe",
    "pbe": "pbe",
    "lb": "lb94",
    "cs": "casida_salahub",
}


class ParsecInputError(ValueError):
    """Invalid or unsupported input in the scoped PARSEC reader."""


class UnsupportedParsecOptionError(ParsecInputError):
    """A valid PARSEC feature lies outside this single-point implementation."""


@dataclass(frozen=True)
class ParsecInputTranslation:
    """Translated Python problem plus nonfatal compatibility notes."""

    source: Path
    problem: SinglePointInput
    warnings: tuple[str, ...]
    output_all_states: bool
    output_level: int
    ignore_symmetry: bool = False
    # Domain_Energy_Tolerance in Ry.  ``domain`` is what the default rule
    # chose where the input leaves the sphere radius to it, and ``None`` for
    # an input with a radius or a box.  ``boundary_tolerance_from`` says
    # where the Hartree boundary tolerance of the problem comes from: its
    # own line ("input"), the fixed default beside a radius ("default"), or
    # the rule ("rule").  ``domain_rule_seconds`` is what the rule took.
    domain_energy_tolerance: float = 1.0e-3
    domain: "DomainChoice | None" = None
    boundary_tolerance_from: str = "default"
    domain_rule_seconds: float = 0.0
    # Structural relaxation (``Minimization``), ``None`` for a single point.
    # ``movable`` holds one flag per atom of the problem (``Move_Flag`` and
    # the optional fourth column of ``Atom_Coord``).  ``compute_forces`` says
    # that the forces of the structure are wanted: always with a relaxation
    # or dynamics, and for a single point when the input says
    # ``Skip_Force: false``.  ``use_symmetry`` is false where PARSEC switches
    # the point group off although ``Ignore_Symmetry`` is not set: manual
    # movements and molecular dynamics.
    relaxation: "RelaxationSettings | None" = None
    movable: tuple[bool, ...] = ()
    compute_forces: bool = False
    use_symmetry: bool = True
    # Molecular dynamics (``Molecular_Dynamics``), ``None`` when it is off.
    molecular_dynamics: "MolecularDynamicsSettings | None" = None
    # ``Potential_Extrapolation``: how the SCF loop of a moved structure
    # starts.  ``none`` as PARSEC (the last potential with the ionic
    # potential exchanged), ``atomic`` with the superposed atomic densities
    # moved along with the atoms, ``second_order`` with the change of the
    # density extrapolated along the trajectory
    # (:mod:`parsec_python.SCF.single_point`).
    potential_extrapolation: str = "none"
    # ``Initial_Structure: mlff``: the input structure is relaxed with a
    # machine-learned force field before the run (:mod:`parsec_python.MLFF`),
    # ``None`` when the run starts from the coordinates of the input.
    mlff: "MLFFSettings | None" = None
    # ``Energy_Refinement``: the hybrid functional whose exchange-correlation
    # energy is evaluated once on the converged orbitals after the SCF
    # (:mod:`parsec_python.Energy.refinement`), ``None`` when it is off.  It
    # is a reporting step and is not part of the problem the SCF solves.
    energy_refinement: str | None = None
    # ``Polarizability``: the field F of the seven finite-field SCF loops of
    # polar.f90 (``Electric_Field`` in ry/bohr/e, used as PARSEC uses it:
    # V = 2 F d.r, see parsec_python.SCF.polarizability), ``None`` when off.
    polarizability_field: float | None = None
    # ``Point_Typ_Num`` groups of the point charges of the problem: the
    # number of charges of each ``Pt_Chg``/``Point_Coord`` type, in order.
    point_charge_type_counts: tuple[int, ...] = ()
    # Inert keywords of PARSEC, kept for the lines it prints: the value of
    # ``External_Mag_Field`` divided by 2.5e9 (what PARSEC prints in Gauss;
    # it applies no field) and whether the wavefunctions are declared
    # complex (``Complex_Wfn``, or forced by a magnetic field).
    magnetic_field_gauss: float = 0.0
    complex_wavefunctions: bool = False
    # ``parsec.dat``, ``eigen.dat``, cube files and ``Output_File_Name``
    # (parsec_python.Output.output_files); they change no number of the run.
    output_files: OutputFileSettings = OutputFileSettings()
    # ``Restart_Run``: the SCF loop starts from the potential of parsec.dat
    # beside the input (parsec_python.SCF.restart).
    restart_run: bool = False
    # ``Restart_From_Wfndat`` with ``Restart_Run``: the restart reads the
    # legacy wfn.dat (potential, density and wavefunctions) instead of
    # parsec.dat (restart.F90; parsec_python.Output.wfn_dat).
    restart_from_wfndat: bool = False
    # ``Fix_Number_Eigenvalues`` of a restart (usrinputfile.F90:1444-1449):
    # the eigensolver keeps the number of eigenpairs of the restart file
    # (EigensolverSettings.fixed_state_count); True also for an NSCF run,
    # where calc_nscf.F90 sets the number itself (the lines are printed).
    fix_number_eigenvalues: bool = False
    # ``Dynamic_Diag_Tol`` given together with ``Restart_Run``:
    # usrinputfile.F90:1492-1495 switches it off and says so.
    dynamic_tolerance_off_for_restart: bool = False
    # ``Super_Cell`` and ``Super_Cell_Vac`` of a periodic input (``None``
    # without the block): the replication and the vacuum factors applied to
    # the lattice vectors, for the report.
    super_cell: tuple[int, int, int] | None = None
    super_cell_vacuum: tuple[int, int, int] | None = None
    # ``nscf_kpoints``: a non-self-consistent run from the potential of
    # parsec.dat instead of the SCF (parsec_python.SCF.nscf), ``None`` without.
    nscf: NSCFSettings | None = None
    # ``Create_DOS``: the DOS file after the SCF (parsec_python.Output.dos).
    dos: DOSSettings | None = None
    # ``Output_GW``: the BerkeleyGW files WFN, RHO and VXC after the SCF or
    # the NSCF run (parsec_python.Output.berkeleygw), ``None`` without.
    gw: GWSettings | None = None
    # The z shift (bohr) PARSEC's initial.f90:201-228 subtracted from every
    # atom of a slab to centre it in its box (0.0: not shifted); the problem
    # holds the shifted atoms.
    slab_shift: float = 0.0
    # The routines of the run (parsec_python.Input.routines): Andrew
    # Grieder's (the default) or the Fortran-derived ones
    # (``Use_Fortran_Routines``, or an input the default does not compute).
    routines: RoutineChoice = RoutineChoice()


@dataclass(frozen=True)
class _InputItem:
    key: str
    value: str | tuple[str, ...]
    line: int
    is_block: bool = False


#: Keywords of ``usrinputfile.F90`` for molecular dynamics.
_MOLECULAR_DYNAMICS_KEYS = (
    "molecular_dynamics",
    "restart_mode",
    "cooling_method",
    "tinit",
    "t_final",
    "t_step",
    "time_step",
    "friction_coefficient",
    "step_num",
    "rng_seed",
    "test_rng",
)

#: ``Boundary_Conditions`` labels and their number of periodic directions
#: (``pbc%per``, ``usrinputfile.F90:392-411``).
_BOUNDARY_PERIODICITY = {
    "cluster": 0,
    "0d": 0,
    "wire": 1,
    "tube": 1,
    "1d": 1,
    "slab": 2,
    "2d": 2,
    "bulk": 3,
    "3d": 3,
    "pbc": 3,
}


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", value.strip().lower())


def _strip_comment(line: str) -> str:
    positions = [line.find(marker) for marker in _COMMENT_MARKERS]
    positions = [position for position in positions if position >= 0]
    if positions:
        line = line[: min(positions)]
    return line.strip()


def _split_scalar(line: str, line_number: int) -> tuple[str, str]:
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(?::|=|\s)\s*(.*?)\s*$", line)
    if match is None:
        # ESDF permits a bare boolean label, interpreted as true.
        bare = line.strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+", bare):
            return bare, "true"
        raise ParsecInputError(f"line {line_number}: cannot parse input statement {line!r}")
    key, value = match.groups()
    return key, value or "true"


def _read_items(path: Path) -> list[_InputItem]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ParsecInputError(f"cannot read PARSEC input {path}: {error}") from error

    items: list[_InputItem] = []
    index = 0
    while index < len(lines):
        line_number = index + 1
        line = _strip_comment(lines[index])
        index += 1
        if not line:
            continue
        begin = re.match(r"^begin\s+(.+?)\s*$", line, flags=re.IGNORECASE)
        if begin is None:
            key, value = _split_scalar(line, line_number)
            items.append(_InputItem(_normalize_label(key), value, line_number))
            continue

        block_name = begin.group(1).strip()
        normalized_name = _normalize_label(block_name)
        block_lines: list[str] = []
        while index < len(lines):
            candidate_number = index + 1
            candidate = _strip_comment(lines[index])
            index += 1
            if not candidate:
                continue
            end = re.match(r"^end\s+(.+?)\s*$", candidate, flags=re.IGNORECASE)
            if end is not None:
                end_name = _normalize_label(end.group(1))
                if end_name != normalized_name:
                    raise ParsecInputError(
                        f"line {candidate_number}: block {block_name!r} "
                        f"closed by mismatched {end.group(1)!r}"
                    )
                break
            block_lines.append(candidate)
        else:
            raise ParsecInputError(
                f"line {line_number}: unterminated begin {block_name}"
            )
        items.append(
            _InputItem(
                normalized_name,
                tuple(block_lines),
                line_number,
                is_block=True,
            )
        )
    return items


def _fortran_float(value: str, *, label: str) -> float:
    token = value.strip().replace("D", "E").replace("d", "e")
    try:
        result = float(token)
    except ValueError as error:
        raise ParsecInputError(f"{label}: expected a number, got {value!r}") from error
    if not np.isfinite(result):
        raise ParsecInputError(f"{label}: value must be finite")
    return result


def _first_float(value: str, *, label: str) -> float:
    match = _FLOAT_PATTERN.search(value)
    if match is None:
        raise ParsecInputError(f"{label}: expected a numeric value")
    return _fortran_float(match.group(0), label=label)


def _strict_float(value: str, *, label: str) -> float:
    token = value.strip()
    if _FLOAT_PATTERN.fullmatch(token) is None:
        raise ParsecInputError(f"{label}: expected one number, got {value!r}")
    return _fortran_float(token, label=label)


def _integer(value: str, *, label: str) -> int:
    number = _strict_float(value, label=label)
    integer = int(number)
    if integer != number:
        raise ParsecInputError(f"{label}: expected an integer, got {value!r}")
    return integer


def _boolean(value: str, *, label: str) -> bool:
    normalized = re.sub(r"[.\s_-]", "", value.lower())
    if normalized in {"yes", "true", "t", "1", "on"}:
        return True
    if normalized in {"no", "false", "f", "0", "off"}:
        return False
    raise ParsecInputError(f"{label}: expected a boolean, got {value!r}")


def _physical_length(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "bohr", "bohrs", "au", "atomicunit", "atomicunits"}:
        factor = 1.0
    elif unit in {"ang", "angstrom", "angstroms"}:
        factor = ANGSTROM_TO_BOHR
    else:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported length unit {remainder!r}; use bohr or ang"
        )
    return number * factor


def _energy_rydberg(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "ry", "rydberg", "rydbergs"}:
        factor = 1.0
    elif unit in {"ha", "hartree", "hartrees"}:
        factor = 2.0
    elif unit in {"ev", "electronvolt", "electronvolts"}:
        factor = EV_TO_RYDBERG
    else:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported energy unit {remainder!r}; use Ry, Hartree, or eV"
        )
    return number * factor


def _esdf_energy_rydberg(value: str, *, label: str) -> float:
    """An energy in Ry with the conversion factors of ESDF (``esdf_physical``).

    ESDF's electronvolt is ``1.60219e-19/2.17991e-18`` Ry (1 Ry =
    13.60563 eV), not the CODATA value of :func:`_energy_rydberg`.
    """

    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder) or "ry"
    if unit not in _ESDF_ENERGY_JOULE:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported energy unit {remainder!r}; use Ry, eV or Hartree"
        )
    return number * _ESDF_ENERGY_JOULE[unit] / _ESDF_ENERGY_JOULE["ry"]


def _force_rydberg_per_bohr(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit in {"", "ry/bohr", "rydberg/bohr", "ry/au"}:
        factor = 1.0
    elif unit in {"ha/bohr", "hartree/bohr", "ha/au", "hartree/au"}:
        factor = 2.0
    elif unit in {"ev/ang", "ev/angstrom"}:
        factor = EV_TO_RYDBERG / ANGSTROM_TO_BOHR
    else:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported force unit {remainder!r}; use Ry/bohr, "
            "Hartree/bohr, or eV/ang"
        )
    return number * factor


def _force_ev_per_angstrom(value: str, *, label: str) -> float:
    """A force in eV/angstrom, the unit when the value names none."""

    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip()
    if not remainder:
        return _first_float(value, label=label)
    return _force_rydberg_per_bohr(value, label=label) * ANGSTROM_TO_BOHR / EV_TO_RYDBERG


def _unquoted(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    return text


def _mlff_settings(one, scalar, source: Path):
    """``MLFFSettings`` of ``Initial_Structure: mlff`` from the ``MLFF_*`` keywords.

    A model file, a ``python:file.py:function`` and ``MLFF_Python`` are
    taken relative to the input.  The interpreter path is made absolute but
    not resolved: a virtual environment's interpreter is a link.
    """

    import os
    import shutil

    from ..MLFF.settings import (
        DEFAULT_FMAX,
        DEFAULT_MODEL,
        DEFAULT_STEPS,
        DEFAULT_TASK,
        MLFFSettings,
    )

    def beside_input(text: str) -> str:
        path = Path(text).expanduser()
        return text if path.is_absolute() else os.path.abspath(source.parent / path)

    model = _unquoted(one("mlff_model", DEFAULT_MODEL))
    lower = model.lower()
    if lower.startswith("python:"):
        target = model.split(":", 1)[1]
        location, separator, function = target.rpartition(":")
        if separator and location.lower().endswith(".py"):
            model = f"python:{beside_input(location)}:{function}"
    elif lower.endswith((".model", ".pt")):
        model = beside_input(model)
    else:
        model = lower
    python = None
    if "mlff_python" in scalar:
        raw = _unquoted(one("mlff_python"))
        if not raw:
            raise ParsecInputError("MLFF_Python cannot be empty")
        if any(mark in raw for mark in ("/", "\\")):
            python = Path(beside_input(raw))
        else:
            found = shutil.which(raw)
            if found is None:
                raise ParsecInputError(f"MLFF_Python: {raw} is not on the PATH")
            python = Path(found)

    def optional_integer(key: str, label: str):
        return _integer(one(key), label=label) if key in scalar else None

    try:
        return MLFFSettings(
            model=model,
            task=_normalize_label(one("mlff_task", DEFAULT_TASK)),
            fmax=(
                _force_ev_per_angstrom(one("mlff_fmax"), label="MLFF_Fmax")
                if "mlff_fmax" in scalar
                else DEFAULT_FMAX
            ),
            steps=_integer(one("mlff_steps", str(DEFAULT_STEPS)), label="MLFF_Steps"),
            optimizer=_normalize_label(one("mlff_optimizer", "lbfgs")),
            device=_normalize_label(one("mlff_device", "auto")),
            python=python,
            charge=optional_integer("mlff_charge", "MLFF_Charge"),
            spin=optional_integer("mlff_spin", "MLFF_Spin"),
            reuse=_boolean(one("mlff_reuse", "true"), label="MLFF_Reuse"),
        )
    except ValueError as error:
        raise ParsecInputError(str(error)) from error


def _temperature_kelvin(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit not in {"", "k", "kelvin"}:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported temperature unit {remainder!r}; use K"
        )
    return number


def _charge_electrons(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[.\s_-]", "", remainder)
    if unit not in {"", "e", "electron", "electrons"}:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported charge unit {remainder!r}; use e"
        )
    return number


#: Electric-field units of ``esdf_mod.f90`` (``phy_u`` 48-53) in V/m, and
#: the aliases this reader also accepts.
ELECTRIC_FIELD_UNITS_V_PER_M = {
    "v/m": 1.0,
    "v/nm": 1.0e9,
    "v/ang": 1.0e10,
    "v/bohr": 1.8897268e10,
    "ry/bohr/e": 2.5711273e11,
    "har/bohr/e": 5.1422546e11,
    # Aliases of this reader, not of ESDF.
    "rydberg/bohr/e": 2.5711273e11,
    "ha/bohr/e": 5.1422546e11,
    "hartree/bohr/e": 5.1422546e11,
    "ev/ang/e": 1.0e10,
}


def _electric_field_value(value: str, *, label: str) -> float:
    """An electric field in ``ry/bohr/e``, the unit of ``usrinputfile.F90``.

    The third token names the unit as in ``esdf_physical``; ESDF drops case
    and the characters ``.`` and ``-``.  (The Fortran reads that token
    list-directed, which ends a word at a slash, so that a unit given there
    stops PARSEC with "Dimensions Do not match"; only the default unit
    works in the Fortran.  This reader converts the units as ESDF intends.)
    """

    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[\s.-]", "", remainder)
    if not unit:
        return number
    factor = ELECTRIC_FIELD_UNITS_V_PER_M.get(unit)
    if factor is None:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported electric-field unit {remainder!r}; use one of "
            "v/m, v/nm, v/ang, v/bohr, ry/bohr/e, har/bohr/e"
        )
    return number * factor / ELECTRIC_FIELD_UNITS_V_PER_M["ry/bohr/e"]


def _block_numbers(lines: tuple[str, ...], *, label: str) -> tuple[list[float], str]:
    values: list[float] = []
    unit_tokens: list[str] = []
    for line in lines:
        for token in re.split(r"[\s,]+", line.strip()):
            if not token:
                continue
            if _FLOAT_PATTERN.fullmatch(token):
                values.append(_fortran_float(token, label=label))
            else:
                unit_tokens.append(token)
    unit = " ".join(unit_tokens)
    return values, unit


def _parse_cell_shape(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    *,
    slab: bool = False,
) -> PeriodicCell:
    """Parse a ``Cell_Shape`` block into a :class:`PeriodicCell`.

    A bulk cell takes one line (orthorhombic side lengths) or three lines (lattice vectors).  A slab takes the two
    lattice vectors of the xy plane (two or three components each) and the height ``2R`` of its box along z from
    ``Boundary_Sphere_Radius``.
    """

    label = "Cell_Shape"
    cell_blocks = blocks.get("cell_shape", [])
    if len(cell_blocks) != 1:
        raise ParsecInputError("a periodic cell requires exactly one Cell_Shape block")
    lines = cell_blocks[0].value
    if slab:
        if len(lines) != 2:
            raise ParsecInputError(f"a slab Cell_Shape block has 2 lines (the in-plane lattice vectors), not {len(lines)}")
        rows = []
        for row_index, line in enumerate(lines, start=1):
            row_values, unit = _block_numbers((line,), label=label)
            if unit or len(row_values) not in (2, 3) or (len(row_values) == 3 and row_values[2] != 0.0):
                raise ParsecInputError(
                    f"slab Cell_Shape line {row_index} must be an in-plane vector: two components, or three with z = 0"
                )
            rows.append([row_values[0], row_values[1]])
        radius_occurrences = scalar.get("boundary_sphere_radius", [])
        if len(radius_occurrences) != 1:
            raise ParsecInputError("a slab needs Boundary_Sphere_Radius, the half height of its box along z")
        height = 2.0 * _physical_length(str(radius_occurrences[0].value), label="Boundary_Sphere_Radius")
        vectors = np.zeros((3, 3))
        vectors[:2, :2] = np.array(rows, dtype=np.float64)
    elif len(lines) == 1:
        values, unit = _block_numbers(lines, label=label)
        if len(values) != 3 or unit:
            raise ParsecInputError(
                "a one-line Cell_Shape block must contain exactly the three "
                "orthorhombic side lengths"
            )
        vectors = np.diag(values)
    elif len(lines) == 3:
        rows = []
        for row_index, line in enumerate(lines, start=1):
            row_values, unit = _block_numbers((line,), label=label)
            if len(row_values) != 3 or unit:
                raise ParsecInputError(
                    f"Cell_Shape line {row_index} must contain exactly three components"
                )
            rows.append(row_values)
        vectors = np.array(rows, dtype=np.float64)
    else:
        raise ParsecInputError(
            "Cell_Shape must contain 1 line (orthorhombic side lengths) or "
            f"3 lines (general lattice vectors), not {len(lines)}"
        )

    scale_occurrences = scalar.get("lattice_vector_scale", [])
    if len(scale_occurrences) > 1:
        lines_str = ", ".join(str(item.line) for item in scale_occurrences)
        raise ParsecInputError(
            f"duplicate lattice_vector_scale values on lines {lines_str}; provide it once"
        )
    scale_text = (
        str(scale_occurrences[0].value) if scale_occurrences else "1 bohr"
    )
    scale = _physical_length(scale_text, label="Lattice_Vector_Scale")
    if slab:
        vectors = vectors * scale
        vectors[2, 2] = height  # Boundary_Sphere_Radius is a length of its own, not scaled
        return PeriodicCell(lattice_vectors=vectors, dimensions=2)
    return PeriodicCell(lattice_vectors=vectors * scale)


def _lattice_vector_scale(scalar: dict[str, list["_InputItem"]]) -> float:
    scale_occurrences = scalar.get("lattice_vector_scale", [])
    if len(scale_occurrences) > 1:
        lines_str = ", ".join(str(item.line) for item in scale_occurrences)
        raise ParsecInputError(
            f"duplicate lattice_vector_scale values on lines {lines_str}; provide it once"
        )
    scale_text = str(scale_occurrences[0].value) if scale_occurrences else "1 bohr"
    return _physical_length(scale_text, label="Lattice_Vector_Scale")


def _parse_wire_slab_cell(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    periodicity: int,
) -> PeriodicCell:
    """``Cell_Shape`` of a wire or slab (``usrinputfile.F90:460-526``).

    A wire has one line with its period ``a`` (``latt_vec(1,1)``, along
    ``x``); a slab two lines with ``a1`` and ``a2`` in the ``xy`` plane.  The
    confined directions get unit vectors (``usrinputfile.F90:777-779``), and
    ``Lattice_Vector_Scale`` multiplies the vectors.  Fortran reads three
    numbers per slab line (a line of two is error 107, although the guide
    shows two); here two numbers or three with a zero third are accepted.
    Fortran reads only the first number of a wire line; further numbers
    other than zero are refused here.
    """

    label = "Cell_Shape"
    cell_blocks = blocks.get("cell_shape", [])
    if len(cell_blocks) != 1:
        # usrinputfile.F90 error 108.
        raise ParsecInputError(
            "a wire or slab requires exactly one Cell_Shape block"
        )
    lines = cell_blocks[0].value
    if len(lines) != periodicity:
        # usrinputfile.F90 error 106.
        raise ParsecInputError(
            f"the Cell_Shape block of a {'wire' if periodicity == 1 else 'slab'} "
            f"has {periodicity} line(s), not {len(lines)}"
        )
    rows = []
    for row_index, line in enumerate(lines, start=1):
        values, unit = _block_numbers((line,), label=label)
        if unit or not values:
            raise ParsecInputError(f"Cell_Shape line {row_index} is not a list of numbers")
        if periodicity == 1:
            if any(value != 0.0 for value in values[1:]):
                raise ParsecInputError(
                    "the Cell_Shape of a wire is its period along x, one number "
                    "(PARSEC reads only the first)"
                )
            if not values[0] > 0.0:
                raise ParsecInputError("the period of a wire must be positive")
            rows.append([values[0], 0.0, 0.0])
            continue
        if len(values) == 2:
            values = [values[0], values[1], 0.0]
        if len(values) != 3:
            raise ParsecInputError(
                f"Cell_Shape line {row_index} of a slab needs the x and y "
                "components of a lattice vector (and a zero z component)"
            )
        if values[2] != 0.0:
            raise ParsecInputError(
                "the lattice vectors of a slab lie in the xy plane: the third "
                f"component of Cell_Shape line {row_index} must be zero"
            )
        rows.append(list(values))
    from ..Grid.confined import confined_lattice_vectors

    vectors = confined_lattice_vectors(np.asarray(rows, dtype=np.float64) * _lattice_vector_scale(scalar))
    try:
        return PeriodicCell(lattice_vectors=vectors)
    except ValueError as error:
        raise ParsecInputError(f"Cell_Shape: {error}") from error


def _charged_sheets(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    requested: bool,
    periodicity: int,
    coordinate_matrix: np.ndarray | None,
    coordinate_factor: float,
    warnings: list[str],
) -> tuple[ChargedSheet, ...]:
    """The charged sheets of a slab (``usrinputfile.F90:2222-2299``).

    ``add_charged_sheets`` (false), ``num_charged_sheets`` types (error 146
    without it), and for every type the next ``sheet_charge`` (e, ESDF's
    cursor) and the next ``Sheet_Coord`` block (error 148 without it), one
    sheet per line; the coordinates take ``Coordinate_Unit`` and only ``z``
    is used.  Charged sheets need a slab (error 145).
    """

    keys = ("num_charged_sheets", "sheet_charge")
    if not requested:
        if any(key in scalar for key in keys) or "sheet_coord" in blocks:
            warnings.append(
                "num_charged_sheets, sheet_charge and Sheet_Coord are ignored "
                "without add_charged_sheets: .true., as in PARSEC"
            )
        return ()
    if periodicity != 2:
        raise ParsecInputError(
            "support for charged sheets is only available in 2-D (slab) systems "
            "(PARSEC error 145)"
        )
    counts = scalar.get("num_charged_sheets", [])
    if not counts:
        raise ParsecInputError(
            "unknown number of charged sheets: num_charged_sheets is missing "
            "(PARSEC error 146)"
        )
    if len(counts) > 1:
        raise ParsecInputError("provide num_charged_sheets once")
    types = _integer(str(counts[0].value), label="num_charged_sheets")
    if types < 1:
        raise ParsecInputError("num_charged_sheets must be at least 1")
    charges = scalar.get("sheet_charge", [])
    coordinate_blocks = blocks.get("sheet_coord", [])
    if len(charges) < types:
        raise ParsecInputError(
            f"num_charged_sheets is {types}, but {len(charges)} sheet_charge "
            "line(s) were found (PARSEC reads one per type)"
        )
    if len(coordinate_blocks) < types:
        raise ParsecInputError(
            f"no charged sheet coordinates were found for sheet type "
            f"{len(coordinate_blocks) + 1} (PARSEC error 148)"
        )
    if len(charges) > types or len(coordinate_blocks) > types:
        warnings.append(
            "more sheet_charge lines or Sheet_Coord blocks than num_charged_sheets: "
            "PARSEC reads the first ones only, and so does this run"
        )
    sheets = []
    for index in range(types):
        charge_values, unit = _block_numbers((str(charges[index].value),), label="sheet_charge")
        if len(charge_values) != 1 or unit.strip().lower() not in {"", "e"}:
            raise ParsecInputError("sheet_charge must be a charge in e")
        block = coordinate_blocks[index]
        if not block.value:
            raise ParsecInputError(
                f"Sheet_Coord block {index + 1} holds no coordinates (PARSEC error 148)"
            )
        for line in block.value:
            values, unit = _block_numbers((line,), label="Sheet_Coord")
            if unit or len(values) < 3:
                raise ParsecInputError("every Sheet_Coord line needs x y z")
            if coordinate_matrix is None:
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
            else:
                position = np.asarray(values[:3], dtype=float) @ coordinate_matrix
            sheets.append(ChargedSheet(charge=charge_values[0], height=float(position[2])))
    return tuple(sheets)


def _parse_super_cell_block(
    blocks: dict[str, list["_InputItem"]],
    key: str,
    *,
    minimum: int,
    periodic: bool,
    count: int = 3,
) -> tuple[int, int, int] | None:
    """One ``Super_Cell``/``Super_Cell_Vac`` block (F ``usrinputfile.F90:723-773``).

    One line (errors 158/160) whose first ``count`` entries are integers
    (159/161); later entries are ignored, as by Fortran's list-directed
    read.  Fortran does not check the values; here a replication below 1 or
    a negative vacuum factor, which would give a degenerate or reversed
    cell, is refused.  A cluster reads nothing from the line, so there only
    the line count is checked.  A wire or slab reads ``count = pbc%per``
    entries (``supercell(1:pbc%per)``); the confined directions keep one
    copy and no vacuum.
    """

    label = "Super_Cell" if key == "super_cell" else "Super_Cell_Vac"
    items = blocks.get(key, [])
    if not items:
        return None
    if len(items) > 1:
        raise ParsecInputError(f"provide a single {label} block")
    lines = items[0].value
    if len(lines) != 1:
        raise ParsecInputError(
            f"there should be 1 line instead of {len(lines)} in the {label} block"
        )
    if not periodic:
        return None
    tokens = [token for token in re.split(r"[\s,]+", lines[0].strip()) if token]
    if len(tokens) < count or not all(
        re.fullmatch(r"[+-]?\d+", token) for token in tokens[:count]
    ):
        raise ParsecInputError(
            f"bad specification of the {label} block: give "
            + ("three integers" if count == 3 else f"{count} integer(s)")
            + f", one per periodic lattice vector, got {lines[0].strip()!r}"
        )
    values = tuple(int(token) for token in tokens[:count])
    if any(value < minimum for value in values):
        raise ParsecInputError(
            f"{label} values must be at least {minimum}, got {list(values)}"
        )
    return values + ((1,) if minimum == 1 else (0,)) * (3 - count)


def _coordinate_factor(value: str) -> float:
    normalized = _normalize_label(value)
    if normalized in {"cartesian_ang", "cartesianang"}:
        return ANGSTROM_TO_BOHR
    if normalized in {
        "cartesian_bohr",
        "cartesianbohr",
        "cartesian_au",
        "cartesianau",
    }:
        return 1.0
    raise UnsupportedParsecOptionError(
        "Coordinate_Unit must be Cartesian_Ang, Cartesian_Bohr or (for a "
        "periodic system) Lattice_Vectors"
    )


def _lattice_is_diagonal(lattice_vectors: np.ndarray) -> bool:
    return bool(np.allclose(lattice_vectors, np.diag(np.diag(lattice_vectors)), atol=1.0e-10))


def _is_lattice_coordinate_unit(value: str) -> bool:
    """``Coordinate_Unit: lattice_vectors``: coordinates in units of the lattice vectors (fractional)."""
    return _normalize_label(value) in {"lattice_vectors", "latticevectors"}


#: The formats of the species that read a per-species label in
#: ``usrinputfile.F90:1806-1867``.  ESDF keeps one cursor per label
#: (``esdf_mod.f90``), so the n-th occurrence of such a label goes to the n-th
#: species that reads it, whatever the species in between.  Every other
#: per-species label is read by every species.
FORMATS_READING = {
    "local_component": frozenset({"martins_new", "martins_old", "fhipp"}),
    "core_cutoff_radius": frozenset({"martins_old", "martins_wang", "fhipp"}),
    "electron_per_orbital": frozenset({"martins_old", "fhipp"}),
    "nonlinear_core_correction": frozenset({"fhipp"}),
}

#: File name of each format: ``<Atom_Type>_<suffix>.DAT`` (``pseudo.f90:143-152``).
POTENTIAL_FILE_SUFFIX = {
    "martins_new": "POTRE",
    "martins_old": "POTR",
    "martins_wang": "POTRW",
    "fhipp": "FHIPP",
}


def _format_input(
    symbol: str,
    pseudo_format: str,
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    species_index: int,
    cursors: dict[str, int],
) -> dict[str, object]:
    """Read what a ``martins_old``, ``martins_wang`` or ``fhipp`` species adds.

    ``usrinputfile.F90`` reads ``Potential_Num`` for every species,
    ``Core_Cutoff_Radius`` for the old, Martins-Wang and FHIPP formats,
    ``Electron_Per_Orbital`` for the old and FHIPP formats and
    ``Nonlinear_Core_Correction`` for FHIPP only, and ESDF hands out the
    occurrences of a label in the order of the calls.  A species therefore
    takes the ``Potential_Num`` of its position among all species
    (``species_index``) and each of the others of its position among the
    species that read it: ``cursors[label]``, which is advanced here.

    A Martins-Wang species takes everything but ``Core_Cutoff_Radius`` from
    its file; PARSEC reads its ``Potential_Num`` and replaces it by the
    channels of the file.

    Where PARSEC would go on with a default (no nonlocal projectors without
    a cutoff radius, no ionic charge or no occupations without the block)
    this reader stops.
    """

    def take(label: str) -> int:
        position = cursors.get(label, 0)
        cursors[label] = position + 1
        return position

    if pseudo_format == "martins_wang":
        return {
            "pseudopotential_format": pseudo_format,
            "core_cutoff_radius": _core_cutoff_radius(
                symbol, pseudo_format, scalar, take("core_cutoff_radius")
            ),
        }
    counts = scalar.get("potential_num", [])
    if species_index >= len(counts):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a {pseudo_format} pseudopotential needs "
            "Potential_Num.  PARSEC gives the n-th Potential_Num of the input "
            f"to the n-th Atom_Type whatever its format: {len(counts)} were "
            f"found and {symbol} is Atom_Type {species_index + 1}, so give one "
            "to every Atom_Type before it as well"
        )
    channels = _integer(
        str(counts[species_index].value), label=f"{symbol} Potential_Num"
    )
    if not 1 <= channels <= 4:
        raise ParsecInputError(
            f"Atom_Type {symbol}: impossible number of potentials "
            f"{channels}; Potential_Num must be 1 to 4"
        )
    radius = _core_cutoff_radius(
        symbol, pseudo_format, scalar, take("core_cutoff_radius")
    )
    orbital_index = take("electron_per_orbital")
    orbital_blocks = blocks.get("electron_per_orbital", [])
    if orbital_index >= len(orbital_blocks):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a {pseudo_format} pseudopotential needs an "
            + (
                "Electron_Per_Orbital block; its sum is the ionic charge"
                if pseudo_format == "martins_old"
                else "Electron_Per_Orbital block, the occupations of its channels"
            )
        )
    lines = orbital_blocks[orbital_index].value
    if len(lines) != 1:
        raise ParsecInputError(
            f"Atom_Type {symbol}: there should be only one line in the "
            "Electron_Per_Orbital block"
        )
    electrons, unit = _block_numbers(lines, label=f"{symbol} Electron_Per_Orbital")
    if unit or len(electrons) < channels:
        raise ParsecInputError(
            f"Atom_Type {symbol}: Electron_Per_Orbital must hold the "
            f"occupations of the {channels} channels of Potential_Num"
        )
    result: dict[str, object] = {
        "pseudopotential_format": pseudo_format,
        "number_of_channels": channels,
        # PARSEC reads Potential_Num values and ignores the rest of the line.
        "electrons_per_orbital": tuple(electrons[:channels]),
        "core_cutoff_radius": radius,
    }
    if pseudo_format == "fhipp":
        flags = scalar.get("nonlinear_core_correction", [])
        position = take("nonlinear_core_correction")
        result["nonlinear_core_correction"] = (
            _boolean(
                str(flags[position].value),
                label=f"{symbol} Nonlinear_Core_Correction",
            )
            if position < len(flags)
            else False
        )
    return result


def _core_cutoff_radius(
    symbol: str,
    pseudo_format: str,
    scalar: dict[str, list["_InputItem"]],
    index: int,
) -> float:
    radii = scalar.get("core_cutoff_radius", [])
    if index >= len(radii):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a {pseudo_format} pseudopotential needs "
            "Core_Cutoff_Radius, the radius of its nonlocal projectors"
        )
    radius = _physical_length(
        str(radii[index].value), label=f"{symbol} Core_Cutoff_Radius"
    )
    if not radius > 0.0:
        raise ParsecInputError(
            f"Atom_Type {symbol}: Core_Cutoff_Radius must be positive"
        )
    return radius


def _has_d_channel(path: Path, format_input: dict[str, object]) -> bool:
    """Whether the pseudopotential file holds a d pseudo-wavefunction."""

    if format_input:
        return int(format_input["number_of_channels"]) > 2
    from ..Pseudopotential import read_parsec_pseudopotential

    return 2 in read_parsec_pseudopotential(path).radial_wavefunctions


def _find_potential(
    symbol: str,
    input_directory: Path,
    pseudopotential_directory: Path | None,
    extension: str = "POTRE",
) -> Path:
    filename = f"{symbol}_{extension}.DAT"
    directories: list[Path] = []
    if pseudopotential_directory is not None:
        directories.append(pseudopotential_directory)
    directories.append(input_directory)
    current = Path.cwd().resolve()
    if current not in directories:
        directories.append(current)

    tried: list[Path] = []
    for directory in directories:
        candidate = (directory / filename).resolve()
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    attempted = "\n  ".join(str(path) for path in tried)
    raise ParsecInputError(
        f"missing pseudopotential for Atom_Type {symbol!r}; searched:\n  {attempted}"
    )


def _parse_parsec_input(
    path: str | Path,
    *,
    pseudopotential_directory: str | Path | None = None,
    routines: RoutineChoice | None = None,
) -> ParsecInputTranslation:
    source = Path(path).expanduser().resolve()
    pp_directory = (
        Path(pseudopotential_directory).expanduser().resolve()
        if pseudopotential_directory is not None
        else None
    )
    items = _read_items(source)

    scalar: dict[str, list[_InputItem]] = {}
    blocks: dict[str, list[_InputItem]] = {}
    for item in items:
        target = blocks if item.is_block else scalar
        target.setdefault(item.key, []).append(item)

    species_keys = {
        "atom_type",
        "element_symbol",
        "pseudopotential_format",
        "local_component",
        "read_vcd",
        "cubic_spline",
        "so_psp",
        "atomic_energy_correction",
        "move_flag",
        "potential_num",
        "core_cutoff_radius",
        "initial_spin_polarization",
        "atomic_radius",
        # LDA+U of the d channel (usrinputfile.F90: esdf_physical, in Ry).
        "ldaplusu_u",
        "ldaplusu_j",
        "nonlinear_core_correction",
        "alpha_filter",
        "beta1_filter",
        "core_filter",
        # Radius of the Ylm-projected DOS (usrinputfile.F90 l.1913).
        "ylm_dos_cutoff",
    }
    block_keys = {
        "atom_coord",
        "domain_shape_parameters",
        "cell_shape",
        "monkhorst_pack_grid",
        "monkhorst_pack_shift",
        "initial_ncl_moment",
        "kpoint_list",
        "bandstruc",
        "electron_per_orbital",
        "point_coord",
        "save_wfn_bands",
        "super_cell",
        "super_cell_vac",
        # Charged sheets of a slab (usrinputfile.F90:2222-2299).
        "sheet_coord",
        # Non-self-consistent run and DOS (usrinputfile.F90 l.1056-1197).
        "nscf_kpoints",
        "nscf_kgrid",
        "nscf_kgrid_shift",
        "dos_mp_grid",
        "dos_mp_shift",
    }
    accepted_global = {
        "restart_run",
        "restart_from_wfndat",
        "relax_restart",
        # Output files of usrinputfile.F90 (parsec.dat, eigen.dat, cube files).
        "output_data",
        "output_file_name",
        "output_eigenvalues",
        "save_intermediate_eigenvalues",
        "save_intermediate_charge_density",
        "output_grid_data",
        "output_gw",
        # Non-self-consistent run, DOS and the G-space Hartree solve
        # (parsec_python.SCF.nscf, parsec_python.Output.dos, Hartree.gspace).
        "nscf_kpoint_unit",
        "nscf_states_num",
        "nscf_fermi_level",
        "create_dos",
        "dos_pnum",
        "ylm_dos_l",
        # PARSEC.py extension: the projection of dos.F90 with its defects.
        "ylm_dos_fortran_projection",
        "calc_hartree_g",
        # PARSEC.py extensions: Fortran's G-space local potential of a 3-D
        # cell, its alpha term, and its potshift (parsec_python.V_ion.gspace_local).
        "periodic_local_potential",
        "periodic_alpha_term",
        "zero_average_vxc",
        "old_pseudopotential_format",
        "periodic_system",
        "boundary_conditions",
        "cluster_domain_shape",
        "boundary_sphere_radius",
        "grid_spacing",
        "coordinate_unit",
        "ignore_symmetry",
        "expansion_order",
        "states_num",
        "net_charges",
        "fermi_temp",
        "spin_polarization",
        "scf_so",
        "so_from_scratch",
        "non_collinear_magnetism",
        "kpoint_method",
        "kpoint_unit",
        "kpoint_file",
        "bandstruc_points",
        "max_iter",
        "convergence_criterion",
        "use_plain_sre",
        "diag_tolerance",
        "eigensolver",
        "chebdav_degree",
        "ff_maxiter",
        "matvec_blocksize",
        "chebyshev_degree",
        "chebyshev_degree_delta",
        "subspace_buffer_size",
        # Eigensolver controls of usrinputfile.F90:1310-1531.
        "chebyshev_filtering",
        "convergence_criterion_approach",
        "maximum_matvec",
        "fix_number_eigenvalues",
        "exp_comm",
        "max_lap_buffers",
        "mpi_groups",
        "mixing_method",
        "mixing_param",
        "memory_param",
        "restart_mixing",
        "mixing_group_size",
        "mixing_en_like",
        "mixing_restart_factor",
        "mixing_memory_expand_factor",
        "mixing_preferred_type",
        "mixing_safeguard",
        "mixing_step_limit",
        "mixing_growth_trigger",
        "mixing_backoff",
        "solver_lpole",
        "full_hartree",
        "hartree_boundary_tolerance",
        "hartree_atomic_tail",
        "domain_energy_tolerance",
        "atom_types_num",
        "correlation_type",
        "electric_field",
        "polarizability",
        "add_point_charges",
        "point_typ_num",
        "pt_chg",
        "add_charged_sheets",
        "num_charged_sheets",
        "sheet_charge",
        "external_mag_field",
        "orb_mag",
        "complex_wfn",
        "symmetrize_charge_density",
        "output_all_states",
        "output_level",
        "old_interpolation_format",
        "double_grid_order",
        "minimization",
        "movement_num",
        "skip_force",
        "total_atom_num",
        "ion_energy_diff",
        "force_min",
        "max_step",
        "min_step",
        "bfgs_number_corr",
        # PARSEC.py extension: a looser SCF criterion while the forces are large.
        "relaxation_scf_upscale",
        *_MOLECULAR_DYNAMICS_KEYS,
        "dynamic_diag_tol",
        # PARSEC.py extension: the start of the SCF loop after a move.
        "potential_extrapolation",
        # PARSEC.py extension: the input structure relaxed with a
        # machine-learned force field first (parsec_python.MLFF).
        "initial_structure",
        "mlff_model",
        "mlff_task",
        "mlff_fmax",
        "mlff_steps",
        "mlff_optimizer",
        "mlff_device",
        "mlff_python",
        "mlff_charge",
        "mlff_spin",
        "mlff_reuse",
        # PARSEC.py extension: a modular initial-density source.  These
        # labels do not alter the DFT grid or any post-initialization physics.
        "initial_density",
        "density_method",
        "ml_density_file",
        "ml_file_path",
        "ml_density_units",
        "ml_density_negative_policy",
        "ml_density_interpolation",
        "ml_density_repository",
        "ml_density_checkpoint",
        "ml_density_python",
        "ml_density_model",
        "ml_density_device",
        "ml_density_cache",
        "ml_density_regenerate",
        "ml_density_chunk_size",
        "ml_density_bridge",
        "normalize_initial_density",
        "lattice_vector_scale",
        # PARSEC.py extension: one post-SCF evaluation of a hybrid functional
        # on the converged orbitals (ARES ``Lrefine``).  It changes nothing
        # of the SCF.
        "energy_refinement",
        # Tkatchenko-Scheffler dispersion correction (usrinputfile.F90).
        "van_der_waals",
        "hf_one",
        "hf_two",
        "hf_three",
        # The routines of the run (parsec_python.Input.routines): Andrew
        # Grieder's by default, the Fortran-derived ones with .true.
        "use_fortran_routines",
    }
    unknown_scalars = sorted(set(scalar).difference(species_keys | accepted_global))
    unknown_blocks = sorted(set(blocks).difference(block_keys))
    if unknown_scalars or unknown_blocks:
        names = unknown_scalars + [f"begin {name}" for name in unknown_blocks]
        raise UnsupportedParsecOptionError(
            "unsupported or unknown PARSEC option(s): " + ", ".join(names)
        )

    def one(key: str, default: str | None = None) -> str:
        occurrences = scalar.get(key, [])
        if len(occurrences) > 1 and key not in species_keys | {"pt_chg"}:
            lines = ", ".join(str(item.line) for item in occurrences)
            raise ParsecInputError(
                f"duplicate {key} values on lines {lines}; provide it once"
            )
        if occurrences:
            return str(occurrences[0].value)
        if default is None:
            raise ParsecInputError(f"required PARSEC option {key} is missing")
        return default

    def optional_bool(key: str, default: bool = False) -> bool:
        if key not in scalar:
            return default
        return _boolean(one(key), label=key)

    # The routines of the run (parsec_python.Input.routines): decided once
    # from the labels here; a value that the default routines do not compute
    # raises UseFortranRoutines below (fall_back), and parse_parsec_input
    # starts again with the Fortran-derived routines.
    # The two sets of routines differ for periodic cells and for the forces and
    # ionic loops of spinor orbitals; every other input (a cluster) runs the same
    # code under both and keeps the default without a note.
    choice_matters = _routines_choice_matters(scalar, one, optional_bool)
    if routines is None:
        # Use_Fortran_Routines: .true., .false. or not given (None: the default of each feature)
        requested = optional_bool(_routines_module.KEYWORD) if _routines_module.KEYWORD in scalar else None
        routines = (
            decide_routines(scalar, blocks, requested=requested)
            if choice_matters or requested
            else _routines_module.DEFAULT
        )

    def fall_back(reason: str) -> None:
        """An input of the default routines that they do not compute: start again on the Fortran-derived ones."""
        if not routines.fortran and choice_matters:
            raise UseFortranRoutines(routines.with_reason(reason))

    def fortran_by_default(feature: str) -> None:
        """A feature whose default is the Fortran-derived routines (the parameter not given).

        Andrew Grieder's routines so far: start again on the Fortran-derived ones.  Already on them (a label his
        routines do not read): the choice records the feature, so that the run says it is the default.
        """
        nonlocal routines
        if not choice_matters or routines.requested is not None:
            return
        if not routines.fortran:
            raise UseFortranRoutines(routines.with_default(feature))
        routines = routines.with_default(feature)

    if not routines.fortran and choice_matters:
        refusal = _default_routines_value_refusal(scalar, one, optional_bool)
        if refusal is not None:
            fall_back(refusal)

    def input_relative_path(key: str, *, follow_links: bool = True) -> Path | None:
        """Resolve an optional quoted path relative to the input file.

        ``follow_links=False`` keeps symbolic links: an interpreter of a
        Linux venv is a link to the base interpreter, which run under its
        own name would not see the packages of the venv.
        """

        if key not in scalar:
            return None
        raw = one(key).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1]
        if not raw:
            raise ParsecInputError(f"{key} cannot be empty")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        return candidate.resolve() if follow_links else Path(os.path.abspath(candidate))

    # Restart_Run (usrinputfile.F90 l.361-368): the SCF loop starts from the
    # potential of parsec.dat (parsec_python.SCF.restart).
    restart_run = optional_bool("restart_run")
    # Restart_From_Wfndat (usrinputfile.F90 l.371-375): with Restart_Run the
    # restart reads wfn.dat (restart.F90) instead of parsec.dat; alone it is
    # read and changes nothing (parsec.F90 l.917-933).
    restart_from_wfndat = optional_bool("restart_from_wfndat")
    # Output_GW (usrinputfile.F90 l.2481-2545): the WFN, RHO and VXC files of
    # BerkeleyGW after the run (parsec_python.Output.berkeleygw).
    output_gw = optional_bool("output_gw")
    # Not a keyword of usrinputfile.F90, which ignores labels it does not
    # know: the format is chosen per species by Pseudopotential_Format.
    old_pseudopotential_format = optional_bool("old_pseudopotential_format")
    boundary = _normalize_label(one("boundary_conditions", "cluster"))
    # usrinputfile.F90:392-411: cluster|0d, wire|tube|1d, slab|2d, bulk|3d|pbc.
    periodicity = _BOUNDARY_PERIODICITY.get(boundary.replace(".", "").replace("-", ""))
    if periodicity is None:
        # usrinputfile.F90 error 101.
        raise ParsecInputError(
            f"Boundary_Conditions={boundary!r} is unknown; PARSEC accepts cluster "
            "(0d), wire (tube, 1d), slab (2d) and bulk (3d, pbc)"
        )
    is_periodic = periodicity > 0
    # Boundary_Conditions: wire or slab (Fortran pbc%per = 1 or 2).
    wire_or_slab = periodicity in (1, 2)
    periodic_flag = optional_bool("periodic_system")
    if periodicity == 1:
        # the default routines have no wire
        fall_back("Boundary_Conditions: wire")
    if periodicity == 2:
        # a slab runs on the Fortran-derived routines unless the input says Use_Fortran_Routines: .false.
        fortran_by_default(_routines_module.SLAB_FEATURE)
    if periodicity == 2 and not periodic_flag:
        # Andrew Grieder's slab needs Periodic_System: .true. (PARSEC does not)
        fall_back("a slab without Periodic_System: .true.")
    # Andrew Grieder's slab (qian/slab_boundary): a 3-D periodic cell of PeriodicCell.dimensions = 2 for the code
    # below, which keeps the wire/slab path of the Fortran-derived routines for wire_or_slab.
    andrew_slab = periodicity == 2 and not routines.fortran
    if andrew_slab:
        wire_or_slab = False
    if not wire_or_slab and periodic_flag != is_periodic:
        raise UnsupportedParsecOptionError(
            "Periodic_System and Boundary_Conditions disagree: "
            "Periodic_System=true requires Boundary_Conditions=bulk or slab, and "
            "Boundary_Conditions=cluster requires Periodic_System=false"
        )
    spin_polarized = optional_bool("spin_polarization")
    noncollinear = optional_bool("non_collinear_magnetism")
    # Fortran only actually routes through the complex spinor path when
    # SO_from_scratch=true (or, a combination this port does not implement,
    # Non_Collinear_magnetism=true together with SCF_SO or a species'
    # SO_PSP).  SCF_SO=true alone is otherwise inert in Fortran -- it is only
    # consulted inside that NCL branch.  Treating SCF_SO=true here as also
    # requesting self-consistent SOC is a deliberate simplification (users
    # asking for "self-consistent spin-orbit" by either name get the same
    # behavior) rather than strict key-for-key Fortran fidelity.
    self_consistent_spin_orbit = optional_bool("so_from_scratch") or optional_bool(
        "scf_so"
    )
    # usrinputfile.F90:1490-1500: dynamic_diag_tol, switched off with
    # Restart_Run ("dynamic tolerance is not working with restart run!").
    dynamic_tolerance = optional_bool("dynamic_diag_tol")
    dynamic_tolerance_off_for_restart = bool(dynamic_tolerance and restart_run)
    if dynamic_tolerance_off_for_restart:
        dynamic_tolerance = False
    # usrinputfile.F90:2583: the local potential and its slope are
    # interpolated directly instead of as r*V and r**2*dV/dr; only the
    # cluster builders (ionpot.f90, forcloc.f90) read it.
    old_interpolation = optional_bool("old_interpolation_format")
    double_grid_order = _integer(one("double_grid_order", "1"), label="Double_Grid_Order")
    if double_grid_order < 1:
        # usrinputfile.F90 error 113.
        raise ParsecInputError(
            f"Double_Grid_Order must be greater than zero, got {double_grid_order}"
        )
    # usrinputfile.F90: Polarizability (false) and Electric_Field (0.001
    # ry/bohr/e, read whatever the flag says).  polar.f90 alone applies the
    # field: without Polarizability PARSEC ignores it.
    polarizability = optional_bool("polarizability")
    electric_field = _electric_field_value(
        one("electric_field", "0.001"), label="Electric_Field"
    )
    # A spin-polarized wire or slab with k-points: the seven field loops of
    # the spin k-point loop, each from the channel potentials of the loop
    # before (usrinputfile.F90 allows pbc%per <= 2; SCF/polarizability.py).
    if polarizability and is_periodic and not wire_or_slab:
        # usrinputfile.F90, error 156.
        raise UnsupportedParsecOptionError(
            "Cannot do polarizability with periodic boundary conditions"
        )

    # usrinputfile.F90:2308-2361: without the line a spin-polarized run takes
    # the Perdew-Wang LSDA ("pl"), an unpolarized one CA.
    correlation = _normalize_label(
        one("correlation_type", "pl" if spin_polarized else "ca")
    )
    xc_functional = _CORRELATION_LABELS.get(correlation)
    if xc_functional is None:
        raise ParsecInputError(
            f"Correlation_Type={correlation!r} is unknown; PARSEC accepts "
            + ", ".join(sorted(_CORRELATION_LABELS))
        )
    if xc_functional in UNPOLARIZED_ONLY_XC_FUNCTIONALS and (
        spin_polarized or noncollinear
    ):
        # exc_spn.f90 has no branch for these labels: Fortran would return
        # the spin densities as potentials and no energy.
        raise UnsupportedParsecOptionError(
            f"Correlation_Type={correlation!r} has no spin-polarized form "
            "(PARSEC's exc_spn.f90 does not implement it); it cannot be "
            "combined with Spin_Polarization or Non_Collinear_magnetism"
        )
    # Read only for Casida-Salahub, as esdf_physical in the 'cs' case of
    # usrinputfile.F90 (default 0 Ry).  Fortran ignores the line silently
    # otherwise, and so does this reader once the value has been checked.
    ionization_energy_difference = 0.0
    if "ion_energy_diff" in scalar:
        value = _energy_rydberg(one("ion_energy_diff"), label="Ion_Energy_Diff")
        if xc_functional == "casida_salahub":
            ionization_energy_difference = value

    periodic_cell = None
    general_periodic_grid = False
    monkhorst_pack_dimensions: tuple[int, int, int] | None = None
    monkhorst_pack_shift: tuple[float, float, float] | None = None
    kpoint_notes: list[str] = []
    explicit_kpoint_array: np.ndarray | None = None
    explicit_kpoint_weights: np.ndarray | None = None
    shape: str | None = None
    # Only the sphere of a cluster can leave its radius to the default rule.
    default_radius = False
    # Super_Cell / Super_Cell_Vac (None without the block) and PARSEC's
    # ``lattvec``: the cell of the input before them, the unit of
    # lattice-vector coordinates and of the replication.
    super_cell = _parse_super_cell_block(
        blocks,
        "super_cell",
        minimum=1,
        periodic=is_periodic,
        count=periodicity if wire_or_slab else 3,
    )
    super_cell_vacuum = _parse_super_cell_block(
        blocks,
        "super_cell_vac",
        minimum=0,
        periodic=is_periodic,
        count=periodicity if wire_or_slab else 3,
    )
    super_cell_notes: list[str] = []
    # Notes of the k-point input of a wire or slab.
    warnings_from_kpoints: list[str] = []
    if not is_periodic and ("super_cell" in blocks or "super_cell_vac" in blocks):
        super_cell_notes.append(
            "Super_Cell/Super_Cell_Vac are read but have no effect for a "
            "cluster, as in PARSEC."
        )
    primitive_vectors: np.ndarray | None = None
    if wire_or_slab:
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        ignore_symmetry = optional_bool("ignore_symmetry")
        periodic_cell = _parse_wire_slab_cell(scalar, blocks, periodicity)
        primitive_vectors = periodic_cell.lattice_vectors
        if super_cell is not None or super_cell_vacuum is not None:
            # usrinputfile.F90:723-773: the periodic vectors are replicated
            # and stretched (supercell(1:per), cell_vac(1:per)); the confined
            # unit vectors stay.  The atoms are copied below.
            replication = super_cell or (1, 1, 1)
            vacuum = super_cell_vacuum or (0, 0, 0)
            periodic_cell = PeriodicCell(
                lattice_vectors=supercell_lattice(
                    periodic_cell.lattice_vectors, replication, vacuum
                )
            )
            primitive_vectors = primitive_lattice(
                periodic_cell.lattice_vectors, replication, vacuum
            )
        if "boundary_sphere_radius" not in scalar:
            # usrinputfile.F90 error 109.
            raise ParsecInputError(
                "Boundary_Sphere_Radius is required for a wire or slab: the "
                "radius of the cylinder or the half height of the slab"
            )
        confinement_radius = _physical_length(
            one("boundary_sphere_radius"), label="Boundary_Sphere_Radius"
        )
        if not confinement_radius > 0.0:
            raise ParsecInputError("Boundary_Sphere_Radius must be positive")
        if double_grid_order > 1:
            raise UnsupportedParsecOptionError(
                "Double_Grid_Order > 1 is not implemented for a wire or slab"
            )
        grid = PeriodicGridSettings(
            spacing=spacing,
            expansion_order=_integer(
                one("expansion_order", "12"), label="Expansion_Order"
            ),
            # usrinputfile.F90:786-794: 0.5, zero along a periodic direction
            # not orthogonal to the other one, zero with Ignore_Symmetry.
            shift=periodic_grid_shift(
                periodic_cell.lattice_vectors, ignore_symmetry=ignore_symmetry
            ),
            box_lengths=tuple(
                float(value)
                for value in np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
            ),
            periodicity=periodicity,
            confinement_radius=confinement_radius,
            # wires and slabs of this path are those of the Fortran-derived routines
            fortran_routines=True,
        )
    if is_periodic:
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        ignore_symmetry = optional_bool("ignore_symmetry")
        if not wire_or_slab:
            # Andrew Grieder's slab: two in-plane vectors and the height 2R of its box (dimensions = 2)
            periodic_cell = _parse_cell_shape(scalar, blocks, slab=andrew_slab)
            primitive_vectors = periodic_cell.lattice_vectors
            if super_cell is not None or super_cell_vacuum is not None:
                replication = super_cell or (1, 1, 1)
                vacuum = super_cell_vacuum or (0, 0, 0)
                periodic_cell = PeriodicCell(
                    lattice_vectors=supercell_lattice(
                        periodic_cell.lattice_vectors, replication, vacuum
                    )
                )
                primitive_vectors = primitive_lattice(
                    periodic_cell.lattice_vectors, replication, vacuum
                )
            grid = PeriodicGridSettings(
                spacing=spacing,
                expansion_order=_integer(
                    one("expansion_order", "12"), label="Expansion_Order"
                ),
                box_lengths=tuple(np.diag(periodic_cell.lattice_vectors)),
                double_grid_order=double_grid_order,
                fortran_routines=routines.fortran,
            )
            if andrew_slab:
                # Andrew Grieder's slab: its own grid (Grid.pbc.build_slab_grid) reads no shift or points rule
                grid = replace(
                    grid,
                    box_lengths=tuple(
                        np.diag(periodic_cell.lattice_vectors)
                        if _lattice_is_diagonal(periodic_cell.lattice_vectors)
                        else np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
                    ),
                )
            elif not routines.fortran:
                # Andrew Grieder's grid: the Fortran grid, half a step off the origin along vectors
                # orthogonal to the others (zero with Ignore_Symmetry), and ``adjust_step``'s even
                # number of points per direction; the lengths of the lattice vectors of an oblique cell
                grid = replace(
                    grid,
                    shift=fortran_grid_shift(periodic_cell.lattice_vectors, ignore_symmetry),
                    points_rule="fortran",
                    box_lengths=tuple(
                        np.diag(periodic_cell.lattice_vectors)
                        if _lattice_is_diagonal(periodic_cell.lattice_vectors)
                        else np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
                    ),
                )
            elif uses_general_periodic_grid(periodic_cell, grid):
                general_periodic_grid = True
                if double_grid_order > 1:
                    raise UnsupportedParsecOptionError(
                        "Double_Grid_Order > 1 is not implemented on Fortran's general "
                        "periodic grid (non-orthogonal lattice vectors, or sides without "
                        "one common grid spacing): its projectors use one grid step"
                    )
                # Fortran's grid (non-orthogonal cell, or sides without one
                # common spacing): its shift rule and the lengths of the lattice
                # vectors.  The orthorhombic grid keeps its settings.
                grid = replace(
                    grid,
                    shift=periodic_grid_shift(
                        periodic_cell.lattice_vectors, ignore_symmetry=ignore_symmetry
                    ),
                    box_lengths=tuple(
                        float(value)
                        for value in np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
                    ),
                )
        kpoint_method = _normalize_label(one("kpoint_method", "none"))
        if kpoint_method == "mp":
            mp_blocks = blocks.get("monkhorst_pack_grid", [])
            if len(mp_blocks) != 1 or len(mp_blocks[0].value) != 1:
                raise ParsecInputError(
                    "Kpoint_Method=mp requires exactly one Monkhorst_Pack_Grid "
                    "block with exactly one line of three integers"
                )
            values, unit = _block_numbers(mp_blocks[0].value, label="Monkhorst_Pack_Grid")
            if wire_or_slab and not unit and len(values) >= periodicity:
                # usrinputfile.F90:975-990 reads the first per entries; the
                # confined directions have one point.
                if any(value != 1 for value in values[periodicity:3]):
                    warnings_from_kpoints.append(
                        "Monkhorst_Pack_Grid: the entries beyond the periodic "
                        "direction(s) of a wire or slab are ignored, as in PARSEC"
                    )
                values = list(values[:periodicity]) + [1.0] * (3 - periodicity)
            if andrew_slab and len(values) == 2:
                values = [*values, 1.0]  # a slab samples its two periodic directions
            if len(values) != 3 or unit or any(value <= 0 for value in values):
                raise ParsecInputError(
                    "Monkhorst_Pack_Grid must contain exactly "
                    + ("two (slab) or three" if andrew_slab else "three")
                    + " positive integers"
                )
            if andrew_slab and values[2] != 1.0:
                # Andrew Grieder's slab has one k-point along z (PARSEC ignores the third number)
                fall_back("a slab Monkhorst_Pack_Grid with more than one point along z")
            monkhorst_pack_dimensions = tuple(int(value) for value in values)
            if any(value != original for value, original in zip(monkhorst_pack_dimensions, values)):
                raise ParsecInputError("Monkhorst_Pack_Grid values must be integers")
            if "monkhorst_pack_shift" in blocks:
                if periodicity == 3:
                    # a bulk cell's shifted grid runs on the Fortran-derived routines (Fortran's wedge and weights)
                    # unless the input says Use_Fortran_Routines: .false.
                    fortran_by_default(_routines_module.SHIFTED_MESH_FEATURE)
                if not routines.fortran:
                    # Andrew Grieder's routines (qian/slab_boundary, bulk cells and slabs; Use_Fortran_Routines:
                    # .false. for both):
                    # Fortran (irrbz.f90): the points are k = (i + shift)/n, i = 0 ... n-1, in coordinates of the
                    # reciprocal lattice vectors, so a zero shift gives a Gamma-centred mesh.  The mesh is kept as an
                    # explicit list of equally weighted points.
                    shift_blocks = blocks["monkhorst_pack_shift"]
                    if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
                        raise ParsecInputError("Monkhorst_Pack_Shift needs one line of numbers")
                    shift_values, shift_unit = _block_numbers(shift_blocks[0].value, label="Monkhorst_Pack_Shift")
                    if andrew_slab and len(shift_values) == 2:
                        shift_values = [*shift_values, 0.0]
                    if len(shift_values) != 3 or shift_unit:
                        raise ParsecInputError("Monkhorst_Pack_Shift must contain " + ("two (slab) or three" if andrew_slab else "three") + " numbers")
                    axes = [
                        (np.arange(count) + offset) / count
                        for count, offset in zip(monkhorst_pack_dimensions, shift_values)
                    ]
                    mesh = np.meshgrid(*axes, indexing="ij")
                    fractions = np.column_stack([m.reshape(-1) for m in mesh])
                    reciprocal = 2.0 * np.pi * np.linalg.inv(periodic_cell.lattice_vectors).T
                    explicit_kpoint_array = fractions @ reciprocal
                    explicit_kpoint_weights = np.full(len(fractions), 1.0 / len(fractions))
                    monkhorst_pack_dimensions = None
                else:
                    # usrinputfile.F90:977-993: one line, the shift of each
                    # periodic direction, >= 0 (errors 116, 117).  With it the
                    # grid is Fortran's ((i-1)+s)/n, reduced to the irreducible
                    # wedge (irrbz.f90); without it this port's own grid is kept.
                    shift_blocks = blocks["monkhorst_pack_shift"]
                    if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
                        raise ParsecInputError(
                            "error in specifying Monkhorst-Pack Shift: give one "
                            "Monkhorst_Pack_Shift block with one line (PARSEC error 116)"
                        )
                    values, unit = _block_numbers(
                        shift_blocks[0].value, label="Monkhorst_Pack_Shift"
                    )
                    if wire_or_slab and not unit and len(values) >= periodicity:
                        # usrinputfile.F90:986-987 reads mpshift(1:per); the
                        # confined directions keep zero.
                        if any(value != 0.0 for value in values[periodicity:3]):
                            warnings_from_kpoints.append(
                                "Monkhorst_Pack_Shift: the entries beyond the periodic "
                                "direction(s) of a wire or slab are ignored, as in PARSEC"
                            )
                        values = list(values[:periodicity]) + [0.0] * (3 - periodicity)
                    if unit or len(values) != 3 or any(value < 0.0 for value in values):
                        raise ParsecInputError(
                            "error in specifying Monkhorst-Pack Shift: three "
                            "non-negative numbers are required (PARSEC error 117)"
                        )
                    monkhorst_pack_shift = tuple(float(value) for value in values)
        elif kpoint_method == "manual":
            list_blocks = blocks.get("kpoint_list", [])
            file_path = input_relative_path("kpoint_file")
            if len(list_blocks) > 1:
                raise ParsecInputError("provide a single Kpoint_List block")
            if bool(list_blocks) == (file_path is not None):
                raise ParsecInputError(
                    "Kpoint_Method=manual requires exactly one of a Kpoint_List "
                    "block or a Kpoint_File (rows of 'kx ky kz [weight]')"
                )
            try:
                if list_blocks:
                    rows = []
                    for line in list_blocks[0].value:
                        values, unit = _block_numbers((line,), label="Kpoint_List")
                        if unit or len(values) not in (3, 4):
                            raise ParsecInputError(
                                f"line {list_blocks[0].line}: every Kpoint_List row "
                                "must be 'kx ky kz' or 'kx ky kz weight'"
                            )
                        rows.append(list(values))
                    if len({len(row) for row in rows}) != 1:
                        raise ParsecInputError(
                            "every Kpoint_List row must have the same number of columns"
                        )
                    raw_kpoints = np.asarray(rows, dtype=float)
                else:
                    raw_kpoints = read_kpoint_file(file_path)
                explicit_kpoint_array, explicit_kpoint_weights = explicit_kpoints(
                    raw_kpoints,
                    unit=_normalize_label(one("kpoint_unit", "cartesian_inverse_bohr")),
                    # Fortran converts with the reciprocal vectors of the
                    # input cell (``lattvec``), not of a Super_Cell.
                    cell=(
                        periodic_cell
                        if super_cell is None and super_cell_vacuum is None
                        else PeriodicCell(lattice_vectors=primitive_vectors)
                    ),
                )
            except (OSError, ValueError) as error:
                raise ParsecInputError(f"Kpoint_Method=manual: {error}") from error
            if wire_or_slab and np.any(
                np.abs(explicit_kpoint_array[:, periodicity:]) > 1.0e-12
            ):
                raise ParsecInputError(
                    "Kpoint_List: a wire or slab has Bloch vectors along its periodic "
                    "direction(s) only; the k-point components along the confined "
                    "direction(s) must be zero"
                )
            # Unequal weights are normalized to one (init_var.F90:365-367);
            # the SCF drivers weight every level by its k-point (flevel.f90).
        elif kpoint_method != "none":
            raise ParsecInputError(
                f"Kpoint_Method must be 'none', 'mp', or 'manual', got {kpoint_method!r}"
            )
        if kpoint_method != "manual" and (
            "kpoint_list" in blocks or "kpoint_file" in scalar or "kpoint_unit" in scalar
        ):
            raise ParsecInputError(
                "Kpoint_List/Kpoint_File/Kpoint_Unit require Kpoint_Method: manual"
            )
        if kpoint_method != "mp" and "monkhorst_pack_shift" in blocks:
            kpoint_notes.append(
                "Monkhorst_Pack_Shift is read only with Kpoint_Method: mp; it is "
                "ignored, as in PARSEC"
            )
    else:
        if "monkhorst_pack_shift" in blocks:
            kpoint_notes.append(
                "Monkhorst_Pack_Shift applies to periodic systems; it is ignored "
                "for a cluster, as in PARSEC"
            )
        if "kpoint_list" in blocks or "kpoint_file" in scalar or "kpoint_unit" in scalar:
            raise ParsecInputError(
                "Kpoint_List/Kpoint_File/Kpoint_Unit apply only to periodic systems"
            )
        shape = _normalize_label(one("cluster_domain_shape", "sphere"))
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        # A sphere whose radius the input leaves out, or gives as "auto", gets
        # it from the default rule, below, once the atoms and the charge are
        # known.
        default_radius = (
            shape == "sphere"
            and _normalize_label(one("boundary_sphere_radius", "auto")) == "auto"
        )
        if default_radius:
            radius = None
            box_lengths = None
        elif shape == "sphere":
            radius = _physical_length(
                one("boundary_sphere_radius"), label="Boundary_Sphere_Radius"
            )
            box_lengths = None
        elif shape == "box":
            domain_blocks = blocks.get("domain_shape_parameters", [])
            if len(domain_blocks) != 1:
                raise ParsecInputError(
                    "box domain requires exactly one Domain_Shape_Parameters block"
                )
            values, unit = _block_numbers(
                domain_blocks[0].value, label="Domain_Shape_Parameters"
            )
            if len(values) != 3:
                raise ParsecInputError(
                    "box Domain_Shape_Parameters must contain three full side lengths"
                )
            # PARSEC reads this block directly in bohr. Permit an explicit unit as
            # a convenience, while retaining raw-bohr behavior when absent.
            factor = (
                _physical_length(f"1 {unit}", label="Domain_Shape_Parameters")
                if unit
                else 1.0
            )
            box_lengths = tuple(float(value * factor) for value in values)
            radius = 0.5 * float(np.linalg.norm(box_lengths))
        elif shape in {"ellipsoid", "cylinder"}:
            # usrinputfile.F90 l.559-653: one line of Domain_Shape_Parameters,
            # x/y/z radii or radius, length and orientation (1, 2, 3); error 110.
            domain_blocks = blocks.get("domain_shape_parameters", [])
            if len(domain_blocks) != 1 or len(domain_blocks[0].value) != 1:
                raise ParsecInputError(
                    f"A Domain_Shape_Parameters block of one line is required for "
                    f"{shape} cluster domains (PARSEC error 110)"
                )
            values, unit = _block_numbers(
                domain_blocks[0].value, label="Domain_Shape_Parameters"
            )
            if len(values) < 3:
                raise ParsecInputError(
                    "bad specification of the "
                    + (
                        "ellipsoidal radii: x_radius y_radius z_radius"
                        if shape == "ellipsoid"
                        else "cylindrical boundaries: radius length orientation"
                    )
                    + " (PARSEC error 110)"
                )
            factor = (
                _physical_length(f"1 {unit}", label="Domain_Shape_Parameters")
                if unit
                else 1.0
            )
            if shape == "ellipsoid":
                parameters = tuple(float(value * factor) for value in values[:3])
                if min(parameters) <= 0.0:
                    raise ParsecInputError(
                        "bad specification of ellipsoidal radii (PARSEC error 110)"
                    )
                radius = max(parameters)
            else:
                axis = values[2]
                if values[0] <= 0.0 or values[1] <= 0.0:
                    raise ParsecInputError(
                        "bad specification of the cylindrical radius or length "
                        "(should be positive; PARSEC error 110)"
                    )
                if axis not in (1, 2, 3):
                    raise ParsecInputError(
                        "bad specification of cylindrical orientation: 1 for the x "
                        "axis, 2 for y, 3 for z (PARSEC error 110)"
                    )
                parameters = (float(values[0] * factor), float(values[1] * factor), float(axis))
                radius = float(np.sqrt((parameters[1] / 2.0) ** 2 + parameters[0] ** 2))
            box_lengths = None
            domain_parameters = parameters
        else:
            raise UnsupportedParsecOptionError(
                f"Cluster_Domain_Shape={shape!r}; PARSEC's shapes are sphere, "
                "ellipsoid, cylinder and box"
            )

        ignore_symmetry = optional_bool("ignore_symmetry")
        shift = (0.0, 0.0, 0.0) if ignore_symmetry else (0.5, 0.5, 0.5)
        grid = GridSettings(
            spacing=spacing,
            # The settings are checked here whatever the radius: with the
            # default rule a stand-in, which the rule replaces below.
            radius=spacing if default_radius else radius,
            expansion_order=_integer(
                one("expansion_order", "12"), label="Expansion_Order"
            ),
            shift=shift,
            domain_shape=shape,
            box_lengths=box_lengths,
            double_grid_order=double_grid_order,
            **(
                {"domain_parameters": domain_parameters}
                if shape in {"ellipsoid", "cylinder"}
                else {}
            ),
        )

    # Coordinate_Unit (usrinputfile.F90:1685-1720): lattice_vectors gives
    # r = sum_j f_j a_j with the vectors of the input cell (``lattvec``).
    coordinate_matrix: np.ndarray | None = None
    coordinate_factor = 1.0
    fractional_coordinates = False
    if _normalize_label(one("coordinate_unit", "cartesian_bohr")) in {
        "lattice_vectors",
        "latticevectors",
    }:
        if not is_periodic:
            raise ParsecInputError(
                "Coordinate_Unit: lattice_vectors needs a periodic "
                "Boundary_Conditions; a cluster has no lattice vectors"
            )
        coordinate_matrix = primitive_vectors
        # Andrew Grieder's routines fold fractional coordinates into the cell (below).
        fractional_coordinates = not routines.fortran
    else:
        coordinate_factor = _coordinate_factor(
            one("coordinate_unit", "cartesian_bohr")
        )
    # Super_Cell: every input atom is followed by its copies.
    replicate = super_cell is not None and int(np.prod(super_cell)) != 1
    symbols = [str(item.value).strip() for item in scalar.get("atom_type", [])]
    declared_types = _integer(one("atom_types_num"), label="Atom_Types_Num")
    if declared_types != len(symbols):
        raise ParsecInputError(
            f"Atom_Types_Num={declared_types}, but {len(symbols)} Atom_Type "
            "occurrences were found"
        )
    if len(set(symbols)) != len(symbols):
        raise ParsecInputError("each Atom_Type must have a unique element symbol")
    for key in species_keys.difference({"atom_type"}):
        occurrences = scalar.get(key, [])
        if len(occurrences) > declared_types:
            lines = ", ".join(str(item.line) for item in occurrences)
            raise ParsecInputError(
                f"{key} occurs {len(occurrences)} times on lines {lines}, "
                f"but Atom_Types_Num is {declared_types}"
            )
    coordinate_blocks = blocks.get("atom_coord", [])
    if len(coordinate_blocks) != declared_types:
        raise ParsecInputError(
            f"expected {declared_types} Atom_Coord blocks, found {len(coordinate_blocks)}"
        )

    def species_value(key: str, index: int, default: str | None = None) -> str:
        occurrences = scalar.get(key, [])
        if index < len(occurrences):
            return str(occurrences[index].value)
        if default is None:
            raise ParsecInputError(
                f"Atom_Type {symbols[index]} is missing occurrence {index + 1} of {key}"
            )
        return default

    type_lines = [item.line for item in scalar.get("atom_type", [])]

    def require_own_section(key: str, label: str) -> None:
        """Refuse a species keyword ESDF would hand to another species.

        ``usrinputfile.F90`` reads it once per Atom_Type and ESDF returns the
        next occurrence each time, so the n-th line goes to the n-th
        Atom_Type wherever it stands.
        """

        for position, item in enumerate(scalar.get(key, [])):
            section = sum(1 for line in type_lines if line < item.line) - 1
            section = max(section, 0)
            if section != position:
                raise ParsecInputError(
                    f"line {item.line}: {label} stands after Atom_Type "
                    f"{symbols[section]}, but PARSEC gives the n-th {label} of "
                    "the input to the n-th Atom_Type, which is "
                    f"{symbols[position]} for this one; give every Atom_Type "
                    f"before it its own {label} (0 for none)"
                )

    require_own_section("ldaplusu_u", "ldaplusu_u")
    require_own_section("ldaplusu_j", "ldaplusu_j")

    moment_blocks = blocks.get("initial_ncl_moment", [])
    if moment_blocks and not noncollinear:
        raise ParsecInputError(
            "Initial_NCL_Moment blocks require Non_Collinear_magnetism: .true."
        )
    if moment_blocks and len(moment_blocks) != declared_types:
        raise ParsecInputError(
            f"expected {declared_types} Initial_NCL_Moment blocks (one per "
            f"Atom_Type, in order), found {len(moment_blocks)}"
        )
    minimization = _normalize_label(one("minimization", "none"))
    if minimization == "":
        minimization = "none"
    if minimization not in {"none", "simple", "bfgs", "manual"}:
        raise ParsecInputError(f"unknown minimization method: {minimization!r}")
    # usrinputfile.F90 reads the molecular dynamics input with the
    # minimization: the two exclude each other, and either one makes the
    # movable flags of the atoms count.
    molecular_dynamics = MolecularDynamicsSettings.from_keywords(
        {
            key: one(key)
            for key in (*_MOLECULAR_DYNAMICS_KEYS, "minimization", "polarizability")
            if key in scalar
        }
    )
    atoms_move = minimization != "none" or molecular_dynamics.enabled

    local_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    atoms: list[Atom] = []
    movable: list[bool] = []
    coordinate_notes: list[str] = []
    specifications: dict[str, SpeciesPotential] = {}
    global_cubic_occurrences = scalar.get("cubic_spline", [])
    old_format_species = 0
    hubbard_notes: list[str] = []
    # The position of each format-dependent label's cursor (FORMATS_READING).
    cursors: dict[str, int] = {}
    formats: list[str] = []
    filter_values: list[float] = []
    for species_index, (symbol, coordinate_item) in enumerate(
        zip(symbols, coordinate_blocks)
    ):
        pseudo_format = _normalize_label(
            species_value("pseudopotential_format", species_index, "martins_new")
        )
        if pseudo_format == "martinsnew":
            pseudo_format = "martins_new"
        if pseudo_format not in POTENTIAL_FILE_SUFFIX:
            # usrinputfile.F90 error 134.
            raise ParsecInputError(
                f"Atom_Type {symbol}: unknown Pseudopotential_Format "
                f"{pseudo_format!r}; PARSEC reads martins_new, martins_old, "
                "martins_wang and fhipp"
            )
        formats.append(pseudo_format)
        local_label = None
        if pseudo_format in FORMATS_READING["local_component"]:
            # No Local_Component for a Martins-Wang species: its file names
            # the local channel (usrinputfile.F90:1806-1827).
            local_index = cursors.get("local_component", 0)
            cursors["local_component"] = local_index + 1
            local_occurrences = scalar.get("local_component", [])
            if local_index >= len(local_occurrences):
                raise ParsecInputError(
                    f"Atom_Type {symbol} is missing occurrence {local_index + 1} "
                    "of local_component"
                )
            local_label = _normalize_label(str(local_occurrences[local_index].value))
            if local_label not in local_map:
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Local_Component must be s, p, d, or f"
                )
        spin_orbit = _boolean(
            species_value("so_psp", species_index, "false"),
            label=f"{symbol} SO_PSP",
        )
        read_vcd = _boolean(
            species_value("read_vcd", species_index, "false"),
            label=f"{symbol} Read_VCD",
        )
        use_spline = (
            _boolean(
                str(global_cubic_occurrences[species_index].value),
                label=f"{symbol} Cubic_Spline",
            )
            if species_index < len(global_cubic_occurrences)
            else False
        )
        format_input: dict[str, object] = {}
        if pseudo_format != "martins_new":
            if read_vcd:
                # usrinputfile.F90 error 234.
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Read_VCD needs a martins_new "
                    "pseudopotential; an old Martins file holds no valence "
                    "charge density"
                    if pseudo_format == "martins_old"
                    else f"Atom_Type {symbol}: Read_VCD needs a martins_new "
                    f"pseudopotential; a {pseudo_format} file holds no valence "
                    "charge density"
                )
            if spin_orbit and pseudo_format in {"martins_wang", "fhipp"}:
                # usrinputfile.F90 error 135 for FHIPP; pseudo.f90 reads the
                # spin-orbit columns of a Martins-Wang file into the wrong
                # array index.
                raise UnsupportedParsecOptionError(
                    f"Atom_Type {symbol}: SO_PSP is not supported for a "
                    f"{pseudo_format} pseudopotential"
                )
            format_input = _format_input(
                symbol, pseudo_format, scalar, blocks, species_index, cursors
            )
        potential_path = _find_potential(
            symbol,
            source.parent,
            pp_directory,
            POTENTIAL_FILE_SUFFIX[pseudo_format],
        )
        if pseudo_format == "martins_wang":
            try:
                local_l = read_martins_wang_header(potential_path).local_angular_momentum
            except (OSError, ValueError) as error:
                raise ParsecInputError(f"Atom_Type {symbol}: {error}") from error
        else:
            local_l = local_map[local_label]
        if is_periodic:
            # usrinputfile.F90:1897-1907 reads the filters of every species of
            # a periodic system; Core_Filter defaults to Alpha_Filter.
            alpha = _strict_float(
                species_value("alpha_filter", species_index, "0"),
                label=f"{symbol} Alpha_Filter",
            )
            beta1 = _strict_float(
                species_value("beta1_filter", species_index, "0"),
                label=f"{symbol} Beta1_Filter",
            )
            core_text = species_value("core_filter", species_index, "")
            core = (
                _strict_float(core_text, label=f"{symbol} Core_Filter")
                if core_text
                else None
            )
            for name, value in (
                ("Alpha_Filter", alpha),
                ("Beta1_Filter", beta1),
                ("Core_Filter", 0.0 if core is None else core),
            ):
                if value < 0.0:
                    raise ParsecInputError(
                        f"Atom_Type {symbol}: {name} must be zero or positive"
                    )
            if alpha != 0.0 or beta1 != 0.0 or core is not None:
                format_input.update(
                    fourier_filter_alpha=alpha,
                    fourier_filter_beta1=beta1,
                    fourier_filter_core=core,
                )
        else:
            for key in ("alpha_filter", "beta1_filter", "core_filter"):
                text_value = species_value(key, species_index, "")
                if text_value:
                    filter_values.append(abs(_first_float(text_value, label=key)))
        element_symbol = species_value(
            "element_symbol", species_index, ""
        ).strip()
        if not element_symbol:
            match = re.match(r"^([A-Z][a-z]?)(?:[-_].*)?$", symbol)
            if match is None:
                raise ParsecInputError(
                    f"Atom_Type {symbol!r} is not a recognizable element/species label; "
                    "provide Element_Symbol for this species"
                )
            element_symbol = match.group(1)
        correction = _energy_rydberg(
            species_value("atomic_energy_correction", species_index, "0 Ry"),
            label=f"{symbol} Atomic_Energy_Correction",
        )
        # usrinputfile.F90 reads it for every species, whatever the spin
        # setting; initchrg.f90 uses it in a spin-polarized run only.
        polarization = _strict_float(
            species_value(
                "initial_spin_polarization",
                species_index,
                repr(DEFAULT_INITIAL_SPIN_POLARIZATION),
            ),
            label=f"{symbol} Initial_Spin_Polarization",
        )
        # usrinputfile.F90 reads Atomic_Radius (bohr, 0 by default) for each
        # type in turn; only the moments of a non-collinear run read it.
        atomic_radius = _first_float(
            species_value("atomic_radius", species_index, "0"),
            label=f"{symbol} Atomic_Radius",
        )
        # usrinputfile.F90: p_pot%uu(3,ity), p_pot%jj(3,ity) (d channel).
        hubbard_u = _esdf_energy_rydberg(
            species_value("ldaplusu_u", species_index, "0"),
            label=f"{symbol} ldaplusu_u",
        )
        hubbard_j = _esdf_energy_rydberg(
            species_value("ldaplusu_j", species_index, "0"),
            label=f"{symbol} ldaplusu_j",
        )
        if (hubbard_u != 0.0 or hubbard_j != 0.0) and not _has_d_channel(
            potential_path, format_input
        ):
            # upot.F90 visits the channels of the file only: a U of a species
            # without a d pseudo-wavefunction is never used.
            hubbard_notes.append(
                f"Atom_Type {symbol}: ldaplusu_u/ldaplusu_j is ignored, as in "
                "PARSEC: the pseudopotential has no d channel"
            )
            hubbard_u = hubbard_j = 0.0
        if old_interpolation and not is_periodic:
            format_input["old_interpolation"] = True
        specifications[symbol] = SpeciesPotential(
            potential_path,
            local_l,
            read_valence_density=read_vcd,
            use_spline=use_spline,
            element_symbol=element_symbol,
            atomic_energy_correction=correction,
            spin_orbit=spin_orbit,
            initial_spin_polarization=polarization,
            atomic_radius=atomic_radius,
            hubbard_u=hubbard_u,
            hubbard_j=hubbard_j,
            **format_input,
        )

        species_moments: list[np.ndarray | None] = [None] * len(coordinate_item.value)
        if moment_blocks:
            moment_item = moment_blocks[species_index]
            if len(moment_item.value) != len(coordinate_item.value):
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Initial_NCL_Moment has "
                    f"{len(moment_item.value)} rows but there are "
                    f"{len(coordinate_item.value)} atoms of this type"
                )
            for row_index, moment_line in enumerate(moment_item.value):
                moment_values, moment_unit = _block_numbers(
                    (moment_line,), label=f"{symbol} Initial_NCL_Moment"
                )
                if moment_unit or len(moment_values) != 3:
                    raise ParsecInputError(
                        f"line {moment_item.line}: every Initial_NCL_Moment row "
                        "must contain exactly three numbers"
                    )
                species_moments[row_index] = np.asarray(moment_values, dtype=float)

        # Move_Flag of usrinputfile.F90: 0 (default) moves every atom of the
        # type; n > 0 moves its first n atoms; a negative value reads a
        # fourth column (1 moves, 0 fixes) when the atoms move at all.
        move_flag = _integer(
            species_value("move_flag", species_index, "0"),
            label=f"{symbol} Move_Flag",
        )
        if move_flag > len(coordinate_item.value):
            raise ParsecInputError(
                f"Atom_Type {symbol}: invalid Move_Flag value {move_flag}"
            )
        for row_index, coordinate_line in enumerate(coordinate_item.value):
            values, unit = _block_numbers((coordinate_line,), label=f"{symbol} Atom_Coord")
            wanted = 4 if (move_flag < 0 and atoms_move) else 3
            if unit or len(values) < wanted:
                raise ParsecInputError(
                    f"line {coordinate_item.line}: every Atom_Coord row must "
                    f"contain {'three coordinates and a movement index' if wanted == 4 else 'exactly three numbers'}"
                )
            if len(values) > 3 and move_flag >= 0 and row_index == 0:
                # PARSEC reads three numbers and leaves the rest of the row.
                coordinate_notes.append(
                    f"Atom_Type {symbol}: the Atom_Coord rows hold more than three "
                    "numbers; without a negative Move_Flag the further columns are "
                    "ignored, as in PARSEC."
                )
            if move_flag == 0 or not atoms_move:
                moves = True
            elif move_flag > 0:
                moves = row_index < move_flag
            else:
                if values[3] not in (0.0, 1.0):
                    raise ParsecInputError(
                        f"Atom_Type {symbol}: invalid atom movement index "
                        f"{values[3]:g}; use 0 or 1"
                    )
                moves = values[3] == 1.0
            if fractional_coordinates and andrew_slab:
                # Andrew Grieder's slab: in-plane fractional coordinates folded into the cell, and z the
                # Cartesian coordinate in bohr, as PARSEC reads it (qian/slab_boundary multiplied z by the
                # height 2R of the box, which is no lattice vector: corrected here)
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
                fraction = position[:2] - np.floor(position[:2])
                position = np.array([*(fraction @ periodic_cell.lattice_vectors[:2, :2]), position[2]])
            elif fractional_coordinates:
                # Andrew Grieder's routines: fractional coordinates are folded into the cell
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
                fraction = position - np.floor(position)
                position = fraction @ periodic_cell.lattice_vectors
            elif coordinate_matrix is None:
                position = np.asarray(values[:3], dtype=float) * coordinate_factor
            else:
                position = np.asarray(values[:3], dtype=float) @ coordinate_matrix
            for copy in (
                replicate_positions(position, primitive_vectors, super_cell)
                if replicate
                else (position,)
            ):
                movable.append(moves)
                atoms.append(
                    Atom(
                        symbol,
                        copy,
                        initial_moment=species_moments[row_index],
                    )
                )
        if not any(atom.symbol == symbol for atom in atoms):
            raise ParsecInputError(f"Atom_Type {symbol} has an empty Atom_Coord block")
    if is_periodic and not wire_or_slab and not andrew_slab and not routines.fortran:
        from ..SCF.pbc import atom_outside_cell

        outside = atom_outside_cell(periodic_cell.lattice_vectors, atoms)
        if outside is not None:
            # Andrew Grieder's preparation takes the atoms inside the cell only
            fall_back(f"atom {outside.symbol} at {np.asarray(outside.position).tolist()} outside the cell")
    only_martins = not any(name in {"martins_wang", "fhipp"} for name in formats)
    for key, table in (
        ("core_cutoff_radius", scalar),
        ("electron_per_orbital", blocks),
        ("nonlinear_core_correction", scalar),
        ("local_component", scalar),
    ):
        # PARSEC reads these for some formats only, in their order: a line
        # under another species would go to the next species that reads it.
        readers = FORMATS_READING[key]
        count = sum(name in readers for name in formats)
        if len(table.get(key, [])) <= count:
            continue
        if key == "local_component":
            raise ParsecInputError(
                f"Local_Component occurs {len(table[key])} times, but only "
                f"{count} Atom_Type(s) read it: PARSEC reads none for a "
                "martins_wang species, whose file names its local channel, and "
                "gives the n-th Local_Component to the n-th species that reads "
                "it; remove the lines of the martins_wang species"
            )
        if only_martins and key != "nonlinear_core_correction":
            raise ParsecInputError(
                f"{key} occurs {len(table[key])} times, but only "
                f"{count} Atom_Type(s) have "
                "Pseudopotential_Format: martins_old, the format that reads it"
            )
        names = " or ".join(sorted(readers))
        raise ParsecInputError(
            f"{key} occurs {len(table[key])} times, but only {count} "
            f"Atom_Type(s) have Pseudopotential_Format {names}, the "
            f"format{'s' if len(readers) > 1 else ''} that read{'' if len(readers) > 1 else 's'} "
            "it; PARSEC gives the n-th occurrence to the n-th species that reads it"
        )
    if any(value != 0.0 for value in filter_values):
        # usrinputfile.F90:1897-1907 reads them only for a periodic system.
        coordinate_notes.append(
            "Alpha_Filter, Beta1_Filter and Core_Filter are read by PARSEC for "
            "periodic systems only; they are ignored for this cluster"
        )
    if old_interpolation and is_periodic:
        # Only the cluster builders ionpot.f90 and forcloc.f90 read it.
        coordinate_notes.append(
            "Old_Interpolation_Format applies to clusters only; PARSEC ignores "
            "it for a periodic system"
        )
    if (
        any(item.read_valence_density for item in specifications.values())
        and specifications[symbols[-1]].pseudopotential_format != "martins_new"
    ):
        # pseudo.f90 tests the format of the last Atom_Type, whichever
        # species has Read_VCD, and stops with error 23.
        raise ParsecInputError(
            "Read_VCD is set and the last Atom_Type has a "
            f"{specifications[symbols[-1]].pseudopotential_format} "
            "pseudopotential: PARSEC stops there, because it tests the format "
            "of the last Atom_Type; put a martins_new Atom_Type last"
        )

    if (
        is_periodic
        and any(specification.spin_orbit for specification in specifications.values())
        and not self_consistent_spin_orbit
        and not noncollinear
    ):
        raise UnsupportedParsecOptionError(
            "SO_PSP=true with a periodic Boundary_Conditions is only supported "
            "together with SO_from_scratch/SCF_SO=true (self-consistent SOC, "
            "run_self_consistent_soc_kpoints); the default perturbative "
            "spin-orbit correction does not yet handle periodic k-point "
            "sampling"
        )
    # Double_Grid_Order > 1 with SO_PSP: the spin-orbit projectors are averaged
    # over the double grid as well (nonloc.F90; Eigensolvers/perturbative_soc.py),
    # and so are the LDA+U projectors (upot.F90; Hamiltonian/hubbard.py).

    warnings: list[str] = (
        list(coordinate_notes) + super_cell_notes + warnings_from_kpoints + kpoint_notes
    )
    # initial.f90:201-228: in a first run (istart == 0) other than manual
    # movements PARSEC re-centres a slab along z (|z_mov| > 0.1 bohr) before
    # the symmetry search; every later step, force table and output file is
    # in that frame.  Wires are not re-centred.
    slab_shift = 0.0
    if wire_or_slab and periodicity == 2 and minimization != "manual":
        from ..V_ion import center_slab_geometry

        placed, would_shift = center_slab_geometry(atoms)
        if not restart_run:
            atoms = list(placed)
            slab_shift = would_shift
        elif would_shift != 0.0:
            warnings.append(
                "Restart_Run: the slab is used at the coordinates of the input "
                "(PARSEC re-centres a slab along z only in a first run); a first "
                f"run would have moved the atoms by {-would_shift:.4f} bohr along "
                "z, so the parsec.dat of such a run belongs to other coordinates: "
                "give the coordinates that run printed (PARSEC: 'The new atom "
                "coordinates are') to restart from its potential"
            )
    if old_pseudopotential_format:
        warnings.append(
            "Old_Pseudopotential_Format is not a PARSEC keyword and is ignored; "
            "the format is chosen per species by Pseudopotential_Format "
            "(martins_old for *_POTR.DAT files)"
        )
    # Polarizability with spin-orbit coupling or non-collinear magnetism: as
    # PARSEC (polar.f90 keeps vold and the wavefunctions), every field loop
    # after the first starts from the loop before (its last mixed potential,
    # and the spinors of the isolated spin-orbit loops; cli.py).
    if not polarizability and "electric_field" in scalar and electric_field != 0.0:
        warnings.append(
            "Electric_Field is ignored without Polarizability: .true., as in "
            "PARSEC, which applies a field only in the seven polarizability runs"
        )
    if "atomic_radius" in scalar and not noncollinear:
        warnings.append(
            "Atomic_Radius is read only by the atomic moments of a "
            "non-collinear calculation; here it changes nothing"
        )
    point_charges, point_charge_type_counts = _point_charges(
        scalar, blocks, optional_bool("add_point_charges"), is_periodic,
        coordinate_factor, warnings,
    )
    charged_sheets = _charged_sheets(
        scalar,
        blocks,
        optional_bool("add_charged_sheets"),
        periodicity,
        coordinate_matrix,
        coordinate_factor,
        warnings,
    )
    # External_Mag_Field and Orb_Mag are inert in PARSEC: usrinputfile.F90
    # stores them, and with a field forces complex wavefunctions, but no
    # Zeeman term or vector potential is applied anywhere.
    magnetic_field = (
        _first_float(one("external_mag_field"), label="External_Mag_Field")
        if "external_mag_field" in scalar
        else 0.0
    )
    if magnetic_field != 0.0:
        warnings.append(
            "External_Mag_Field is accepted as in PARSEC, where it applies no "
            "field to the electrons (no Zeeman term or vector potential); the "
            "calculation is that without it"
        )
    if optional_bool("orb_mag"):
        warnings.append(
            "Orb_Mag is accepted as in PARSEC, where it changes nothing of the "
            "calculation"
        )
    complex_wavefunctions = optional_bool("complex_wfn") or magnetic_field != 0.0
    if complex_wavefunctions and not is_periodic:
        warnings.append(
            "Complex wavefunctions (Complex_Wfn or External_Mag_Field): the "
            "Hamiltonian of an isolated system is real, and this code "
            "computes its states as real vectors, which are the same states"
        )
    warnings.extend(hubbard_notes)
    if "total_atom_num" in scalar:
        declared_atoms = _integer(
            one("total_atom_num"), label="Total_Atom_Num"
        )
        if declared_atoms != len(atoms):
            raise ParsecInputError(
                f"Total_Atom_Num={declared_atoms}, but {len(atoms)} coordinates "
                "were found"
            )
    skip_force = optional_bool("skip_force")
    movement_count = None
    if "movement_num" in scalar:
        movement_count = _integer(one("movement_num"), label="Movement_Num")
        if movement_count < 0:
            raise ParsecInputError("Movement_Num must be nonnegative")
    # Ion_Energy_Diff is not one of them: it is the Casida-Salahub parameter
    # (read with Correlation_Type above).
    ionic_control_keys = {
        "movement_num",
        "force_min",
        "max_step",
        "min_step",
    }
    ionic_controls_present = bool(ionic_control_keys.intersection(scalar))
    force_minimum = (
        _force_rydberg_per_bohr(one("force_min"), label="Force_Min")
        if "force_min" in scalar
        else 1.0e-2
    )
    maximum_step = (
        _physical_length(one("max_step"), label="Max_Step")
        if "max_step" in scalar
        else DEFAULT_RELAXATION_MAXIMUM_STEP
    )
    minimum_step = (
        _physical_length(one("min_step"), label="Min_Step")
        if "min_step" in scalar
        else 1.0e-4
    )
    states = _integer(one("states_num"), label="States_Num")

    eigensolver_method = _normalize_label(one("eigensolver", "chebdav"))
    if eigensolver_method == "arpk":
        eigensolver_method = "arpack"
    if eigensolver_method in {"trlanc", "trlan"}:
        raise UnsupportedParsecOptionError(
            "Eigensolver=trlanc: the thick-restart Lanczos solver was removed "
            "from PARSEC (usrinputfile.F90 ierr 126) and has no source to port"
        )
    if eigensolver_method not in {"chebff", "chebdav", "arpack", "diagla", "none"}:
        raise UnsupportedParsecOptionError(
            f"Eigensolver={eigensolver_method!r} is not supported"
        )
    if eigensolver_method == "diagla":
        warnings.append(
            "Eigensolver: diagla -- the present Fortran PARSEC refuses it in "
            "its input reader (usrinputfile.F90 ierr 126, 'DIAGLA eigensolver "
            "is to be removed from PARSEC'); PARSEC.py runs the DIAGLA block "
            "Davidson solver of diagla.f90z, as earlier PARSEC versions did."
        )
        if is_periodic:
            warnings.append(
                "Eigensolver: diagla on a periodic grid: its preconditioner "
                "(fdo5scl) annihilates the alternating grid mode when every "
                "axis has an even number of points, and DIAGLA can stall once "
                "the residual is that mode (an odd number of points per axis "
                "avoids it)."
            )

    # usrinputfile.F90 reads Chebdav_Degree for CHEBFF and CHEBDAV only.
    first_filter_degree = 20
    if eigensolver_method in {"chebff", "chebdav"}:
        first_filter_degree = _integer(
            one("chebdav_degree", "20"), label="Chebdav_Degree"
        )
    elif "chebdav_degree" in scalar:
        warnings.append(
            f"Chebdav_Degree is not read for Eigensolver={eigensolver_method} "
            "(usrinputfile.F90 reads it for chebff and chebdav only)."
        )
    if eigensolver_method == "chebdav" and first_filter_degree < 15:
        raise ParsecInputError(
            "Chebdav_Degree must be at least 15 when Eigensolver=chebdav"
        )
    if eigensolver_method == "chebff" and first_filter_degree < 10:
        warnings.append(
            f"Chebdav_Degree={first_filter_degree} is below 10; "
            "PARSEC resets it to 15."
        )
        first_filter_degree = 15

    if eigensolver_method == "chebff":
        first_filter_cycles = _integer(
            one("ff_maxiter", "2"), label="FF_MaxIter"
        )
        if not 1 <= first_filter_cycles <= 9:
            warnings.append(
                f"FF_MaxIter={first_filter_cycles} is outside 1..9; "
                "PARSEC resets it to 2."
            )
            first_filter_cycles = 2
    else:
        # usrinputfile.F90 reads FF_MaxIter only inside the CHEBFF case.
        # Retain the model's harmless default without treating this label as
        # a CHEBDAV or ARPACK convergence control.
        first_filter_cycles = 2

    # usrinputfile.F90: min(nstate/2, 4) for ARPACK, min(nstate/2, 6) for
    # CHEBFF/CHEBDAV; DIAGLA (refused there) takes the latter.
    block_limit = 4 if eigensolver_method == "arpack" else 6
    default_block_size = max(1, min(states // 2, block_limit))
    matvec_block_size = _integer(
        one("matvec_blocksize", str(default_block_size)),
        label="Matvec_Blocksize",
    )
    if matvec_block_size < 1:
        raise ParsecInputError("Matvec_Blocksize must be positive")

    filter_degree = _integer(
        one("chebyshev_degree", "15"), label="Chebyshev_Degree"
    )
    default_delta = "1" if filter_degree < 10 else "3"
    filter_delta = _integer(
        one("chebyshev_degree_delta", default_delta),
        label="Chebyshev_Degree_Delta",
    )
    subspace_buffer = _integer(
        one("subspace_buffer_size", "6"), label="Subspace_Buffer_Size"
    )
    # Chebyshev_filtering (solver%do_subsp): forced on for CHEBFF and CHEBDAV
    # (usrinputfile.F90:1452-1455), default off for ARPACK and DIAGLA.
    chebyshev_filtering = None
    if eigensolver_method in {"arpack", "diagla"}:
        chebyshev_filtering = optional_bool("chebyshev_filtering")
    elif (
        eigensolver_method in {"chebff", "chebdav"}
        and "chebyshev_filtering" in scalar
        and not optional_bool("chebyshev_filtering")
    ):
        warnings.append(
            f"Chebyshev_Filtering=false is ignored: PARSEC always runs the "
            f"Chebyshev subspace filter after {eigensolver_method}."
        )
    filtering = eigensolver_method in {"chebff", "chebdav"} or bool(
        chebyshev_filtering
    )
    if not filtering and eigensolver_method != "none":
        unread = [
            label
            for key, label in (
                ("chebyshev_degree", "Chebyshev_Degree"),
                ("chebyshev_degree_delta", "Chebyshev_Degree_Delta"),
            )
            if key in scalar
        ]
        if unread:
            warnings.append(
                f"{' and '.join(unread)} not read: Eigensolver="
                f"{eigensolver_method} without Chebyshev_Filtering runs no "
                "Chebyshev subspace filter."
            )
    if filtering and subspace_buffer < 6:
        warnings.append(
            f"Subspace_Buffer_Size={subspace_buffer} is below 6 for "
            f"{eigensolver_method} filtering; PARSEC resets it to 6."
        )
        subspace_buffer = 6
    maximum_matvec = _integer(
        one("maximum_matvec", "30000"), label="Maximum_Matvec"
    )
    if maximum_matvec < 1:
        raise ParsecInputError("Maximum_Matvec must be positive")
    # usrinputfile.F90:1444-1449 / parsec.F90:1046 / eigval.F90:679-685:
    # without initeigval the number of eigenpairs per representation is that
    # of the restart file (restart_run.F90, restart.F90), kept in every solve.
    fix_number_eigenvalues = optional_bool("fix_number_eigenvalues")
    if dynamic_tolerance_off_for_restart:
        warnings.append(
            "Dynamic_Diag_Tol is not used with Restart_Run: PARSEC switches it off "
            "(usrinputfile.F90: dynamic tolerance is not working with restart run)"
        )
    if fix_number_eigenvalues and not restart_run:
        # PARSEC then has no number at all (nn = -1 of init_var.F90): its
        # first diagonalization fails (eigval error flag -3, segmentation fault).
        raise UnsupportedParsecOptionError(
            "Fix_Number_Eigenvalues=true keeps the number of eigenvalues of the "
            "restart file; it needs Restart_Run (PARSEC's first diagonalization "
            "fails without one)"
        )
    communication = [
        label
        for key, label in (
            ("exp_comm", "Exp_Comm"),
            ("max_lap_buffers", "Max_Lap_Buffers"),
            # usrinputfile.F90 l.2863: the number of processor groups.
            ("mpi_groups", "MPI_Groups"),
        )
        if key in scalar
    ]
    if communication:
        if "max_lap_buffers" in scalar:
            _integer(one("max_lap_buffers"), label="Max_Lap_Buffers")
        if "mpi_groups" in scalar:
            _integer(one("mpi_groups"), label="MPI_Groups")
        if "exp_comm" in scalar:
            optional_bool("exp_comm")
        warnings.append(
            f"{' and '.join(communication)} tune the MPI communication of "
            "PARSEC's matrix-vector product and have no effect here."
        )

    eigensolver = EigensolverSettings(
        method=eigensolver_method,
        tolerance=_strict_float(
            one("diag_tolerance", "1e-4"), label="Diag_Tolerance"
        ),
        first_filter_degree=first_filter_degree,
        first_filter_cycles=first_filter_cycles,
        matvec_block_size=matvec_block_size,
        subspace_buffer=subspace_buffer,
        filter_degree=filter_degree,
        filter_degree_delta=filter_delta,
        chebyshev_filtering=chebyshev_filtering,
        dynamic_tolerance=dynamic_tolerance,
        maximum_matvec=maximum_matvec,
        fixed_state_count=fix_number_eigenvalues,
    )

    # usrinputfile.F90:1540-1600: the mixer and its keywords, every one read
    # with its PARSEC default whatever the method.
    mixing_name = _normalize_label(one("mixing_method", "anderson"))
    if mixing_name not in MIXING_METHODS:
        raise ParsecInputError(
            f"Mixing_Method: unknown mixer type {mixing_name!r}; PARSEC's "
            "mixers are " + ", ".join(MIXING_METHODS)
        )
    safeguard = optional_bool("mixing_safeguard")
    if safeguard and mixing_name != "anderson":
        raise UnsupportedParsecOptionError(
            f"Mixing_Safeguard is an option of the Anderson mixer of "
            f"PARSEC.py; Mixing_Method={mixing_name!r} has none"
        )
    mixing = MixingSettings(
        parameter=_strict_float(one("mixing_param", "0.30"), label="Mixing_Param"),
        memory=_integer(one("memory_param", "4"), label="Memory_Param"),
        restart=_integer(one("restart_mixing", "20"), label="Restart_Mixing"),
        safeguard=safeguard,
        step_limit=_strict_float(
            one("mixing_step_limit", "2.0"), label="Mixing_Step_Limit"
        ),
        growth_trigger=_strict_float(
            one("mixing_growth_trigger", "2.0"),
            label="Mixing_Growth_Trigger",
        ),
        backoff=_strict_float(
            one("mixing_backoff", "0.5"), label="Mixing_Backoff"
        ),
        method=mixing_name,
        group_size=_integer(
            one("mixing_group_size", "1"), label="Mixing_Group_Size"
        ),
        en_like=_integer(one("mixing_en_like", "0"), label="Mixing_EN_Like"),
        restart_factor=_strict_float(
            one("mixing_restart_factor", "0.3"), label="Mixing_Restart_Factor"
        ),
        expand_factor=_strict_float(
            one("mixing_memory_expand_factor", "2.0"),
            label="Mixing_Memory_Expand_Factor",
        ),
        preferred_type=_integer(
            one("mixing_preferred_type", "1"), label="Mixing_Preferred_Type"
        ),
    )
    # Keywords PARSEC reads but the selected mixer never uses.
    multisecant_keys = (
        "Mixing_Group_Size",
        "Mixing_EN_Like",
        "Mixing_Restart_Factor",
        "Mixing_Memory_Expand_Factor",
        "Mixing_Preferred_Type",
    )
    inert_mixing_keys = {
        "anderson": multisecant_keys,
        "broyden": multisecant_keys,
        "msecant1": ("Restart_Mixing", "Mixing_Preferred_Type"),
        "msecant2": ("Restart_Mixing", "Mixing_Preferred_Type"),
        "msecant3": ("Restart_Mixing",),
    }[mixing_name]
    for keyword in inert_mixing_keys:
        if keyword.lower() in scalar:
            warnings.append(
                f"{keyword} has no effect with Mixing_Method={mixing_name}; "
                "PARSEC reads it and its mixer does not use it."
            )
    # genbro.f90 keeps its pairs in files of mixer%block - 1 slots and drops
    # some from the steps after the block (see Mixer/broyden.py); this port
    # keeps them all.
    broyden_block = max(20, mixing.memory + 1)
    if mixing_name == "broyden" and mixing.memory > 0 and mixing.restart > broyden_block:
        warnings.append(
            f"Restart_Mixing={mixing.restart} exceeds the Broyden block of "
            f"PARSEC ({broyden_block}): genbro.f90 then drops the pairs of the "
            f"steps 1+k*{broyden_block - 1} (k >= 1) of each restart cycle "
            "from the later steps; PARSEC.py keeps every pair since the last "
            "restart, so the SCF histories differ from that step on."
        )
    # Solver_Lpole is PARSEC's keyword.  The two below are not: they control
    # what a prepared system adds to the multipole boundary values, and
    # "Hartree_Boundary_Tolerance: off" restores PARSEC's boundary.  "auto",
    # and no line beside a radius left to the default rule, is the tolerance
    # that rule derives from the electron count: it replaces the default
    # below.
    boundary_tolerance_text = one("hartree_boundary_tolerance", "1e-3 Ry")
    boundary_tolerance_label = re.sub(
        r"[.\s_-]", "", boundary_tolerance_text.lower()
    )
    default_tolerance = boundary_tolerance_label == "auto" or (
        default_radius and "hartree_boundary_tolerance" not in scalar
    )
    if boundary_tolerance_label == "auto":
        boundary_tolerance_text = "1e-3 Ry"
    if boundary_tolerance_label in {"off", "none", "no", "false"}:
        boundary_tolerance = None
    else:
        boundary_tolerance = _energy_rydberg(
            boundary_tolerance_text, label="Hartree_Boundary_Tolerance"
        )
    atomic_tail_text = re.sub(
        r"[.\s_-]", "", one("hartree_atomic_tail", "auto").lower()
    )
    if atomic_tail_text == "auto":
        atomic_tail = "auto"
    else:
        atomic_tail = (
            "on"
            if _boolean(atomic_tail_text, label="Hartree_Atomic_Tail")
            else "off"
        )
    try:
        hartree = HartreeSettings(
            multipole_order=_integer(
                one("solver_lpole", "9"), label="Solver_Lpole"
            ),
            boundary_method="direct" if optional_bool("full_hartree") else "auto",
            boundary_tolerance=boundary_tolerance,
            atomic_tail=atomic_tail,
        )
    except ValueError as error:
        if isinstance(error, ParsecInputError):
            raise
        raise ParsecInputError(
            f"Solver_Lpole/Hartree_Boundary_Tolerance: {error}"
        ) from error
    fermi_temperature = _temperature_kelvin(
        one("fermi_temp", "80"), label="Fermi_Temp"
    )
    fixed_occupations = None
    fixed_occupations_down = None
    if fermi_temperature < 0 and spin_polarized:
        # initial.f90 reads one section per spin channel, the up one first.
        from ..Occupations import read_spin_occupation_file

        spin_file = read_spin_occupation_file(source.parent / "occup.in", states)
        fixed_occupations = spin_file.occupations_up
        fixed_occupations_down = spin_file.occupations_down
        if spin_file.unread_lines:
            warnings.append(
                f"{spin_file.source.name}: {spin_file.unread_lines} line(s) "
                "after the sections of the two spin channels are ignored."
            )
    elif fermi_temperature < 0:
        # initial.f90: a negative temperature reads the occupations from the
        # file occup.in of the run directory.
        from ..Occupations import read_occupation_file

        occupation_file = read_occupation_file(source.parent / "occup.in", states)
        fixed_occupations = occupation_file.occupations
        if occupation_file.unread_lines:
            warnings.append(
                f"{occupation_file.source.name}: only the first section is "
                "read, as PARSEC does without spin polarization; "
                f"{occupation_file.unread_lines} further line(s), the "
                "sections of other spin channels, are ignored."
            )
    # usrinputfile.F90, "VDW PART": the TS correction needs PBE (ierr 158) and
    # no symmetry (ierr 159); a periodic system must be bulk (ierr 160, the
    # only periodic case here) and reads the Hirshfeld cells Hf_One/Two/Three.
    van_der_waals = optional_bool("van_der_waals")
    hirshfeld_cells = (0, 0, 0)
    hirshfeld_keys = [key for key in ("hf_one", "hf_two", "hf_three") if key in scalar]
    if van_der_waals:
        if correlation not in {"pb", "pbe"}:
            raise ParsecInputError(
                "Van_Der_Waals: the TS correction is implemented only for the "
                "GGA-PBE functional (Correlation_Type pb; PARSEC error 158)"
            )
        if not ignore_symmetry:
            raise ParsecInputError(
                "Van_Der_Waals: for the TS correction symmetry has to be "
                "switched off (Ignore_Symmetry: true; PARSEC error 159)"
            )
        if is_periodic:
            values = []
            for key, label in (("hf_one", "Hf_One"), ("hf_two", "Hf_Two"), ("hf_three", "Hf_Three")):
                value = _integer(one(key, "0"), label=label)
                if value < 0:
                    raise ParsecInputError(f"{label} must be nonnegative")
                values.append(value)
            hirshfeld_cells = tuple(values)
        elif hirshfeld_keys:
            warnings.append(
                "Hf_One/Hf_Two/Hf_Three are read for a periodic system only "
                "and are ignored, as in PARSEC"
            )
    elif hirshfeld_keys:
        warnings.append(
            "Hf_One/Hf_Two/Hf_Three are present without Van_Der_Waals: true "
            "and are ignored"
        )

    scf = SCFSettings(
        max_iterations=_integer(one("max_iter", "50"), label="Max_Iter"),
        convergence_criterion=_energy_rydberg(
            one("convergence_criterion", "2e-4 Ry"),
            label="Convergence_Criterion",
        ),
        fermi_temperature_kelvin=fermi_temperature,
        fixed_occupations=fixed_occupations,
        spin_polarized=spin_polarized,
        fixed_occupations_down=fixed_occupations_down,
        number_of_states=states,
        net_charge=_charge_electrons(
            one("net_charges", "0"), label="Net_Charges"
        ),
        use_plain_residual=optional_bool("use_plain_sre"),
        convergence_criterion_approach=(
            _energy_rydberg(
                one("convergence_criterion_approach"),
                label="Convergence_Criterion_Approach",
            )
            if "convergence_criterion_approach" in scalar
            else None
        ),
        normalize_initial_density=optional_bool(
            "normalize_initial_density", True
        ),
        xc_functional=xc_functional,
        ionization_energy_difference=ionization_energy_difference,
        self_consistent_spin_orbit=self_consistent_spin_orbit
        or (
            noncollinear
            and any(specification.spin_orbit for specification in specifications.values())
        ),
        noncollinear=noncollinear,
        van_der_waals=van_der_waals,
        hirshfeld_cells=hirshfeld_cells,
    )

    initial_labels = []
    for key in ("initial_density", "density_method"):
        if key in scalar:
            initial_labels.append((key, _normalize_label(one(key))))
    if len(initial_labels) == 2 and initial_labels[0][1] != initial_labels[1][1]:
        raise ParsecInputError(
            "Initial_Density and legacy Density_Method specify different methods"
        )
    initial_method = initial_labels[0][1] if initial_labels else "sad"
    if initial_method == "ml":
        initial_method = "file"
    if initial_method == "sad_ml_grid":
        raise UnsupportedParsecOptionError(
            "Density_Method=sad_ml_grid changed the DFT grid in the old architecture. "
            "The modular implementation keeps Grid_Spacing/domain authoritative; "
            "use Initial_Density=sad, file, charge3net, scdp, or custom."
        )
    file_keys = [key for key in ("ml_density_file", "ml_file_path") if key in scalar]
    if len(file_keys) > 1:
        raise ParsecInputError(
            "provide only one of ML_Density_File and legacy ML_File_Path"
        )
    density_file = input_relative_path(file_keys[0]) if file_keys else None
    model_method = initial_method not in {"sad", "file"}
    default_model = "qm9"
    if model_method:
        from ..MLDensity.registry import get_provider, initial_density_methods

        try:
            default_model = get_provider(initial_method).default_model
        except ValueError:
            raise ParsecInputError(
                f"unknown Initial_Density {initial_method!r}; use one of "
                + ", ".join(initial_density_methods())
            ) from None
    density_cache = input_relative_path("ml_density_cache")
    if density_cache is None and model_method:
        density_cache = source.parent / ".parsec_ml_density_cache"
    initial_density_settings = InitialDensitySettings(
        method=initial_method,
        file=density_file,
        units=normalize_density_units(one("ml_density_units", "auto")),
        negative_policy=_normalize_label(
            one("ml_density_negative_policy", "clip")
        ),
        interpolation=_normalize_label(
            one("ml_density_interpolation", "linear")
        ),
        repository=input_relative_path("ml_density_repository"),
        checkpoint=input_relative_path("ml_density_checkpoint"),
        python_executable=input_relative_path("ml_density_python", follow_links=False),
        model=one("ml_density_model", default_model).strip().lower(),
        device=_normalize_label(one("ml_density_device", "auto")),
        cache_directory=density_cache,
        regenerate=optional_bool("ml_density_regenerate"),
        prediction_chunk_size=_integer(
            one("ml_density_chunk_size", "50000"),
            label="ML_Density_Chunk_Size",
        ),
        bridge=input_relative_path("ml_density_bridge"),
    )

    scf_upscale = 1.0
    if "relaxation_scf_upscale" in scalar:
        scf_upscale = _strict_float(
            one("relaxation_scf_upscale"), label="Relaxation_SCF_Upscale"
        )
        if not (np.isfinite(scf_upscale) and scf_upscale >= 1.0):
            raise ParsecInputError("Relaxation_SCF_Upscale must be a number of at least 1")
        if minimization not in {"bfgs", "simple"}:
            if scf_upscale != 1.0:
                warnings.append(
                    "Relaxation_SCF_Upscale loosens the SCF loops of a BFGS or "
                    "steepest-descent relaxation; here it is ignored."
                )
            scf_upscale = 1.0

    relaxation = None
    use_symmetry = True
    # A manual run keeps the coordinates of the input and of manual.dat as
    # they are; every other calculation recentres the cluster, but for a
    # restart (initial.f90 l.133-134: istart == 0 only).
    recenter = minimization != "manual" and not restart_run
    if minimization != "none":
        # A periodic cell (also a wire or slab) relaxes with a fixed cell, as
        # in PARSEC; the movements a reduced k-point set cannot follow are
        # refused below (_check_reduced_kpoint_movements).
        relaxation = RelaxationSettings(
            method=minimization,
            max_movements=500 if movement_count is None else movement_count,
            minimum_force=force_minimum,
            maximum_step=maximum_step,
            minimum_step=minimum_step,
            bfgs_corrections=_integer(
                one("bfgs_number_corr", "7"), label="BFGS_number_corr"
            ),
            restart=optional_bool("relax_restart"),
            scf_upscale=scf_upscale,
        )
        if minimization == "manual":
            # parsec.F90 uses no symmetry when the structures are imposed,
            # and usrinputfile.F90 makes every atom movable.
            use_symmetry = False
            movable = [True] * len(atoms)
        elif not any(movable):
            raise ParsecInputError(
                "Minimization is requested, but Move_Flag leaves no atom movable"
            )
    else:
        if ionic_controls_present:
            warnings.append(
                "Ionic relaxation thresholds/step controls are present without an "
                "active minimizer and are ignored for this fixed single point."
            )
        if optional_bool("relax_restart"):
            warnings.append(
                "Relax_Restart is set without Minimization and is ignored."
            )
    if molecular_dynamics.enabled:
        if not any(movable):
            raise ParsecInputError(
                "Molecular_Dynamics is requested, but Move_Flag leaves no atom movable"
            )
        # parsec.F90 uses no symmetry in molecular dynamics.
        use_symmetry = False
    if polarizability:
        # parsec.F90 uses no symmetry when it applies an electric field.
        use_symmetry = False
    if point_charges:
        # PARSEC finds the point group of the atoms alone (a defect of the
        # Fortran unless Ignore_Symmetry is set); a group the point charges
        # do not have is not used here.
        use_symmetry = False
    symmetrize_density = optional_bool("symmetrize_charge_density")
    space_group = None
    if wire_or_slab:
        # The space group of a wire or slab (and what it decides) is found
        # below, once bandstruc and nscf_kpoints are read.
        pass
    elif is_periodic and not routines.fortran:
        # Andrew Grieder's routines find no space group of a crystal (no IBZ, no
        # symmetrization, no fractional translations in the grid).
        pass
    elif is_periodic:
        weighted_kpoints = bool(
            explicit_kpoint_weights is not None
            and not np.allclose(explicit_kpoint_weights, explicit_kpoint_weights[0])
        )
        forces_requested = atoms_move or ("skip_force" in scalar and not skip_force)
        symmetrize_density, space_group, grid = _periodic_symmetry(
            atoms,
            periodic_cell,
            grid,
            general_periodic_grid=general_periodic_grid,
            ignore_symmetry=ignore_symmetry,
            monkhorst_pack_shift=monkhorst_pack_shift,
            symmetrize_density=symmetrize_density,
            symmetrize_given="symmetrize_charge_density" in scalar,
            noncollinear=noncollinear,
            warnings=warnings,
            forces_of_weighted_kpoints=forces_requested and weighted_kpoints,
            search_info=_bulk_symmetry_search_info(
                periodic_cell,
                ignore_symmetry=ignore_symmetry,
                kpoints=monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None,
                spinors=bool(
                    noncollinear
                    or self_consistent_spin_orbit
                    or any(specification.spin_orbit for specification in specifications.values())
                ),
                bands=bool(blocks.get("bandstruc")),
                nscf="nscf_kpoints" in blocks,
                molecular_dynamics=molecular_dynamics.enabled,
                manual_movement=minimization == "manual",
            ),
        )
        if forces_requested:
            _check_reduced_kpoint_movements(
                atoms,
                periodic_cell,
                space_group,
                movable=movable,
                relaxation=minimization in {"bfgs", "simple"},
                molecular_dynamics=molecular_dynamics.enabled or minimization == "manual",
                reduced=bool(monkhorst_pack_shift is not None or weighted_kpoints),
                ignore_symmetry=ignore_symmetry,
            )
    elif symmetrize_density:
        # The isolated spin-orbit and non-collinear drivers average their
        # densities like the collinear ones (symm_scalar of every rho(:,isp);
        # non-collinear: rho and |m| as scalars, the direction of m kept).
        if ignore_symmetry or not use_symmetry or shape != "sphere":
            warnings.append(
                "Symmetrize_Charge_Density changes nothing here: the "
                "calculation uses no point group (Ignore_Symmetry, a box domain, "
                "manual movements, molecular dynamics, Polarizability or point "
                "charges), so PARSEC's group has the identity alone"
            )
            symmetrize_density = False
    if symmetrize_density and not wire_or_slab:
        scf = replace(scf, symmetrize_density=True)
    # PARSEC always evaluates the forces.  Here a single point evaluates
    # them only on request, so that inputs without the line keep their cost.
    compute_forces = atoms_move or ("skip_force" in scalar and not skip_force)
    if periodicity == 3 and choice_matters and routines.requested is None:
        # the bulk features whose default is the Fortran-derived routines (Gamma-point forces, a space group
        # that sets Fortran's grid), unless the input says Use_Fortran_Routines
        kpoint_input = monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None
        for feature in _routines_module.bulk_default_features(
            atoms,
            periodic_cell,
            grid,
            kpoints=kpoint_input,
            gamma_only=_gamma_only_sampling(monkhorst_pack_dimensions, explicit_kpoint_array),
            forces=compute_forces,
            spinors=bool(self_consistent_spin_orbit or noncollinear),
            spin_polarized=spin_polarized,
            ignore_symmetry=ignore_symmetry,
            spline_species=all(specification.use_spline for specification in specifications.values()),
            search_info=_bulk_symmetry_search_info(
                periodic_cell,
                ignore_symmetry=ignore_symmetry,
                kpoints=kpoint_input,
                spinors=bool(
                    noncollinear
                    or self_consistent_spin_orbit
                    or any(specification.spin_orbit for specification in specifications.values())
                ),
                bands=bool(blocks.get("bandstruc")),
                nscf="nscf_kpoints" in blocks,
                molecular_dynamics=molecular_dynamics.enabled,
                manual_movement=minimization == "manual",
            ),
        ):
            fortran_by_default(feature)
    if not molecular_dynamics.enabled and any(
        key in scalar for key in _MOLECULAR_DYNAMICS_KEYS if key != "molecular_dynamics"
    ):
        warnings.append(
            "Molecular dynamics controls are present without "
            "Molecular_Dynamics: true and are ignored."
        )
    if is_periodic and not routines.fortran:
        # Andrew Grieder's routines: a crystal's forces are not symmetrized: PARSEC.py finds no space-group
        # operations (the point-group operations of a cluster do not apply), and the structure is relaxed as it is.
        use_symmetry = False

    potential_extrapolation = "none"
    if "potential_extrapolation" in scalar:
        from ..SCF.single_point import POTENTIAL_EXTRAPOLATIONS

        potential_extrapolation = _normalize_label(one("potential_extrapolation"))
        if potential_extrapolation not in POTENTIAL_EXTRAPOLATIONS:
            raise ParsecInputError(
                "Potential_Extrapolation must be one of "
                + ", ".join(POTENTIAL_EXTRAPOLATIONS)
            )
        if not atoms_move:
            warnings.append(
                "Potential_Extrapolation is set, but no atom moves; it is ignored."
            )

    mlff = None
    structure_source = _normalize_label(one("initial_structure", "input"))
    if structure_source not in {"input", "mlff"}:
        raise ParsecInputError("Initial_Structure must be input or mlff")
    if structure_source == "mlff":
        if is_periodic:
            raise UnsupportedParsecOptionError(
                "Initial_Structure: mlff with Boundary_Conditions=bulk: the force-field "
                "relaxation is implemented for isolated systems only"
            )
        mlff = _mlff_settings(one, scalar, source)
    elif any(key.startswith("mlff_") for key in scalar):
        warnings.append(
            "MLFF_* keywords are present without Initial_Structure: mlff and are ignored."
        )

    energy_refinement = None
    if "energy_refinement" in scalar:
        from ..Energy.refinement import check_refinement_request, normalize_refinement

        try:
            energy_refinement = normalize_refinement(one("energy_refinement"))
            check_refinement_request(
                energy_refinement,
                periodic=is_periodic,
                molecular_dynamics=molecular_dynamics.enabled,
            )
        except ValueError as error:
            raise UnsupportedParsecOptionError(str(error)) from error

    if ignore_symmetry and not is_periodic:
        warnings.append(
            "Ignore_Symmetry=true reproduces PARSEC's zero grid shift and "
            "disables accelerated symmetry reduction unless --symmetry=on "
            "explicitly overrides it."
        )
    if not is_periodic and shape == "box" and hartree.boundary_method != "direct":
        warnings.append(
            "Box Hartree boundaries use the exact direct Coulomb sum in auto mode "
            "and may be expensive."
        )

    band_settings: BandStructureSettings | None = None
    band_blocks = blocks.get("bandstruc", [])
    if band_blocks or "bandstruc_points" in scalar:
        if not is_periodic:
            raise ParsecInputError("bandstruc applies only to periodic systems")
        if len(band_blocks) != 1:
            raise ParsecInputError(
                "bandstruc_points requires exactly one bandstruc block"
                if not band_blocks
                else "provide a single bandstruc block"
            )
        segments = []
        for line in band_blocks[0].value:
            tokens = line.replace("\t", " ").split()
            try:
                if len(tokens) < 7:
                    raise ValueError("expected 'n kx0 ky0 kz0 kx1 ky1 kz1 label'")
                int(tokens[0])
                numbers = [float(token.lower().replace("d", "e")) for token in tokens[1:7]]
                segments.append(
                    BandSegment(
                        start=tuple(numbers[:3]),
                        end=tuple(numbers[3:]),
                        name=" ".join(tokens[7:]),
                    )
                )
            except ValueError as error:
                raise ParsecInputError(
                    f"bandstruc line {band_blocks[0].line}: {error}; every line is "
                    "'n kx0 ky0 kz0 kx1 ky1 kz1 label' in reciprocal-lattice coordinates"
                ) from error
        points = _integer(one("bandstruc_points", "20"), label="Bandstruc_Points")
        try:
            band_settings = BandStructureSettings(segments=tuple(segments), points=points)
        except ValueError as error:
            raise ParsecInputError(f"bandstruc: {error}") from error

    domain_energy_tolerance = _energy_rydberg(
        one("domain_energy_tolerance", "1e-3 Ry"), label="Domain_Energy_Tolerance"
    )
    if not domain_energy_tolerance > 0.0:
        raise ParsecInputError("Domain_Energy_Tolerance must be positive")
    multipole_boundary = shape == "sphere" and hartree.boundary_method != "direct"
    if default_radius and multipole_boundary and (
        boundary_tolerance is None or atomic_tail == "off"
    ):
        raise ParsecInputError(
            "Boundary_Sphere_Radius is left to the default rule, and the "
            "default radius is chosen for the controlled Hartree boundary: "
            "with Hartree_Boundary_Tolerance: off or Hartree_Atomic_Tail: off "
            "give Boundary_Sphere_Radius"
        )
    domain = None
    domain_rule_seconds = 0.0
    boundary_tolerance_from = (
        "input" if "hartree_boundary_tolerance" in scalar else "default"
    )
    if default_radius or (default_tolerance and multipole_boundary):
        # The rule reads the geometry as the driver places it and the free
        # atoms of the pseudopotential files.  An input with a radius and
        # without "Hartree_Boundary_Tolerance: auto" does not come here.
        from ..Hartree.domain import auto_boundary_tolerance, default_domain
        from ..V_ion import (
            center_cluster_geometry,
            ionic_charge,
            load_pseudopotentials,
        )

        rule_started = perf_counter()
        placed = center_cluster_geometry(atoms) if recenter else tuple(atoms)
        potentials = load_pseudopotentials(
            specifications, xc_functional=xc_functional
        )
        electrons = ionic_charge(placed, potentials) - scf.net_charge
        if electrons <= 0:
            raise ParsecInputError("the requested system has no valence electrons")
        if default_radius:
            domain = default_domain(
                placed,
                potentials,
                electrons,
                spacing=grid.spacing,
                shift=grid.shift[0],
                stencil_half_width=grid.stencil_half_width,
                budget=domain_energy_tolerance,
                boundary_tolerance=None if default_tolerance else boundary_tolerance,
                multipole_boundary=multipole_boundary,
            )
            # The run takes what the two texts read back to, as an input
            # that holds them does.
            grid = replace(
                grid,
                radius=_physical_length(
                    domain.radius_text, label="Boundary_Sphere_Radius"
                ),
            )
            rule_tolerance_text = domain.tolerance_text
            if domain.extra_electrons > 1.0e-6:
                warnings.append(
                    f"{domain.extra_electrons:g} electron(s) more than the free "
                    "atoms hold: their tail is unknown before the SCF; the "
                    "vacuum was raised from "
                    f"{domain.wall_vacuum / ANGSTROM_TO_BOHR:.1f} to "
                    f"{domain.vacuum / ANGSTROM_TO_BOHR:.1f} ang. This does not "
                    'guarantee Domain_Energy_Tolerance. Read "Density at the '
                    'sphere" after the SCF; with the highest occupied level '
                    "above zero the energy has no limit for a large sphere."
                )
        else:
            vacuum = grid.radius - max(
                float(np.linalg.norm(atom.position)) for atom in placed
            )
            tolerance = auto_boundary_tolerance(
                electrons, domain_energy_tolerance, vacuum
            )
            rule_tolerance_text = f"{tolerance:.1e} Ry"
        if rule_tolerance_text is not None:
            hartree = replace(
                hartree,
                boundary_tolerance=_energy_rydberg(
                    rule_tolerance_text, label="Hartree_Boundary_Tolerance"
                ),
            )
            boundary_tolerance_from = "rule"
        domain_rule_seconds = perf_counter() - rule_started

    if wire_or_slab:
        refused = [
            label
            for condition, label in (
                (
                    any(specification.spin_orbit for specification in specifications.values())
                    or self_consistent_spin_orbit,
                    "spin-orbit coupling",
                ),
                (noncollinear, "non-collinear magnetism"),
                # LDA+U: upot.F90 replicates along the periodic directions
                # (nreplica(1:per)); the combinations the LDA+U code refuses
                # for any periodic system are refused below.
                # usrinputfile.F90:2380-2386: the TS correction needs per = 3.
                (van_der_waals, "Van_Der_Waals"),
                (energy_refinement is not None, "Energy_Refinement"),
                (initial_density_settings.method != "sad", "an initial density other than SAD"),
            )
            if condition
        ]
        if refused:
            raise UnsupportedParsecOptionError(
                "Boundary_Conditions wire or slab is not implemented together with "
                + ", ".join(refused)
            )
    _check_hubbard_and_dispersion(
        specifications,
        van_der_waals=van_der_waals,
        is_periodic=is_periodic,
        spin_polarized=spin_polarized,
        kpoints=monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None,
        spin_orbit=bool(
            self_consistent_spin_orbit
            or noncollinear
            or any(specification.spin_orbit for specification in specifications.values())
        ),
        bands=band_settings is not None,
        energy_refinement=energy_refinement is not None,
        forces=compute_forces or atoms_move,
    )
    # Double_Grid_Order > 1 with SO_PSP and forces: the spin-orbit projector
    # gradients are those of the double-grid projectors (Forces.spin_orbit).
    if any(specification.hubbard_j != 0.0 for specification in specifications.values()):
        warnings.append(
            "ldaplusu_j: the J term uses PARSEC's P3/P4 tensors (upot.F90) on the "
            "occupation matrix of the real d harmonics, as the Fortran does; "
            "its potential matrix is not symmetric there"
        )
    post_processing = _post_processing_input(
        scalar,
        blocks,
        one,
        optional_bool,
        is_periodic=is_periodic,
        wire_or_slab=wire_or_slab,
        periodicity=periodicity,
        periodic_cell=periodic_cell,
        # Fortran converts nscf_kpoint_unit with the input cell (lattvec).
        input_cell=(
            None
            if periodic_cell is None
            else (
                periodic_cell
                if super_cell is None and super_cell_vacuum is None
                else PeriodicCell(lattice_vectors=primitive_vectors)
            )
        ),
        restart_run=restart_run,
        states=states,
        symbols=symbols,
        monkhorst_pack_dimensions=monkhorst_pack_dimensions,
        kpoints=explicit_kpoint_array,
        kpoint_weights=explicit_kpoint_weights,
        spin_orbit=bool(self_consistent_spin_orbit or noncollinear),
        noncollinear=bool(noncollinear),
        warnings=warnings,
        output_gw=output_gw,
        monkhorst_pack_shift=monkhorst_pack_shift,
    )
    if post_processing.nscf is not None and eigensolver.fixed_state_count:
        # calc_nscf.F90:146-147 sets the number of eigenpairs of the NSCF
        # solve itself (nn = nstate): Fix_Number_Eigenvalues changes nothing.
        eigensolver = replace(eigensolver, fixed_state_count=False)
    # Create_DOS with Restart_Run: the SCF runs on the Monkhorst-Pack grid of
    # the DOS (usrinputfile.F90 l.1188-1193) with mpshift = dos_mpshift.  The
    # space group, its wedge and the default of Symmetrize_Charge_Density
    # belong to the shift of the restarted SCF: build them again for it.  A
    # wire or slab finds its group below (_wire_slab_symmetry), with this shift.
    if post_processing.monkhorst_pack_shift is not None and (
        post_processing.monkhorst_pack_shift != monkhorst_pack_shift
    ):
        monkhorst_pack_shift = post_processing.monkhorst_pack_shift
        if not wire_or_slab:
            symmetrize_density, space_group, grid = _periodic_symmetry(
                atoms,
                periodic_cell,
                grid,
                general_periodic_grid=general_periodic_grid,
                ignore_symmetry=ignore_symmetry,
                monkhorst_pack_shift=monkhorst_pack_shift,
                symmetrize_density=optional_bool("symmetrize_charge_density"),
                symmetrize_given="symmetrize_charge_density" in scalar,
                noncollinear=noncollinear,
                warnings=warnings,
                # The restarted SCF runs on the DOS grid (no weighted list).
                search_info=_bulk_symmetry_search_info(
                    periodic_cell,
                    ignore_symmetry=ignore_symmetry,
                    kpoints=True,
                    spinors=bool(
                        noncollinear
                        or self_consistent_spin_orbit
                        or any(
                            specification.spin_orbit for specification in specifications.values()
                        )
                    ),
                    bands=band_settings is not None,
                    nscf=post_processing.nscf is not None,
                    molecular_dynamics=molecular_dynamics.enabled,
                    manual_movement=minimization == "manual",
                ),
            )
            scf = replace(scf, symmetrize_density=bool(symmetrize_density))
            if forces_requested:
                _check_reduced_kpoint_movements(
                    atoms,
                    periodic_cell,
                    space_group,
                    movable=movable,
                    relaxation=minimization in {"bfgs", "simple"},
                    molecular_dynamics=(
                        molecular_dynamics.enabled or minimization == "manual"
                    ),
                    reduced=True,
                    ignore_symmetry=ignore_symmetry,
                )
    monkhorst_pack_dimensions = post_processing.monkhorst_pack_dimensions
    explicit_kpoint_array = post_processing.kpoints
    explicit_kpoint_weights = post_processing.kpoint_weights
    if post_processing.g_space_hartree:
        hartree = replace(hartree, periodic_g_space=True)
    if post_processing.nscf is not None and (
        band_settings is not None
        or compute_forces
        or atoms_move
        or polarizability
    ):
        raise UnsupportedParsecOptionError(
            "nscf_kpoints skips the SCF and everything after it in PARSEC "
            "(parsec.F90 l.1098): bandstruc, forces (Skip_Force: false), "
            "Minimization and Molecular_Dynamics are not run with it; remove them"
        )
    if wire_or_slab:
        symmetrize_density, space_group, grid = _wire_slab_symmetry(
            atoms,
            periodic_cell,
            grid,
            periodicity=periodicity,
            ignore_symmetry=ignore_symmetry,
            monkhorst_pack_shift=monkhorst_pack_shift,
            symmetrize_density=symmetrize_density,
            symmetrize_given="symmetrize_charge_density" in scalar,
            noncollinear=noncollinear,
            # parsec.F90:470-486: what turns the symmetry of a slab on, and
            # what switches every symmetry off.
            spin_orbit=bool(
                noncollinear
                or any(specification.spin_orbit for specification in specifications.values())
            ),
            bands=band_settings is not None,
            nscf=post_processing.nscf is not None,
            no_symmetry=bool(
                molecular_dynamics.enabled or polarizability or minimization == "manual"
            ),
            warnings=warnings,
        )
        if symmetrize_density:
            scf = replace(scf, symmetrize_density=True)
        if compute_forces:
            # The movements a reduced k-point set cannot follow, as for a bulk
            # cell (the wedge forces are completed over the wire/slab group in
            # Forces.wire_slab, which the ionic loop restricts to the
            # operations Move_Flag respects and molecular dynamics drops).
            _check_wire_slab_kpoint_movements(
                atoms,
                periodic_cell,
                grid,
                space_group,
                periodicity=periodicity,
                movable=movable,
                relaxation=minimization in {"bfgs", "simple"},
                molecular_dynamics=molecular_dynamics.enabled or minimization == "manual",
                weighted_kpoints=bool(
                    explicit_kpoint_weights is not None
                    and not np.allclose(explicit_kpoint_weights, explicit_kpoint_weights[0])
                ),
                monkhorst_pack_shift=monkhorst_pack_shift,
                ignore_symmetry=ignore_symmetry,
                slab_symmetry_on=bool(
                    band_settings is not None
                    or post_processing.nscf is not None
                    or noncollinear
                    or any(
                        specification.spin_orbit for specification in specifications.values()
                    )
                ),
            )
    has_kpoints = monkhorst_pack_dimensions is not None or explicit_kpoint_array is not None
    has_soc_species = any(
        specification.spin_orbit for specification in specifications.values()
    )
    # LDA+U of a wire or slab with k-points, spin polarization or a band
    # structure: the Bloch projectors replicate along the periodic directions
    # alone (nreplica(1:per), Hamiltonian.hubbard.hubbard_replica_lattice) and
    # the occupations of an irreducible wedge are averaged over the wire/slab
    # group of alatt (hubbard_symmetry_lattice).
    # A wire or slab relaxes and moves with spin polarization and k-points as
    # a bulk cell does: the spin k-point loop starts every structure from the
    # channel potentials of the one before (Relaxation.wire_slab_backend).
    # The drivers of a plain SCF loop (run_scf, and run_spin_scf for an
    # isolated spin-polarized input without spin-orbit species): the others
    # (k-points, spin-orbit, non-collinear, periodic spin) keep no fields of
    # their iterations and take no starting potential, but for the scalar
    # k-point loop.
    plain_loop = not (
        has_kpoints
        or self_consistent_spin_orbit
        or noncollinear
        or (spin_polarized and (is_periodic or has_soc_species))
    )
    gw = None
    if output_gw:
        gw = _gw_settings(
            post_processing,
            atoms=atoms,
            periodic_cell=periodic_cell,
            is_periodic=is_periodic,
            wire_or_slab=wire_or_slab,
            spinors=bool(self_consistent_spin_orbit or noncollinear or has_soc_species),
            bands=band_settings is not None,
            moves=atoms_move,
            forces=compute_forces,
            monkhorst_pack_dimensions=monkhorst_pack_dimensions,
            monkhorst_pack_shift=monkhorst_pack_shift,
            explicit_kpoints=explicit_kpoint_array is not None,
            space_group=space_group,
            ignore_symmetry=ignore_symmetry,
            warnings=warnings,
        )
    output_files = _output_file_settings(scalar, blocks, one, optional_bool, has_kpoints)
    # Save_Intermediate_Charge_Density / Save_Intermediate_Eigenvalues: every
    # SCF driver hands its iterations to the output files (parsec.F90
    # l.1381-1402 writes them in PARSEC's one SCF loop for every kind of run).
    # A wire or slab writes and reads parsec.dat as wfnsave.F90 /
    # restart_run.F90 do for pbc%per = 1, 2 (stype(3) = 2, 3; Output/parsec_dat.py).
    # Fortran's general periodic grid (non-orthogonal cells) has its parsec.dat
    # (shift and per-direction steps, Output/parsec_dat.py), so Restart_Run
    # and Save_Intermediate_Charge_Density work there as on the other grids.
    if restart_from_wfndat and not restart_run:
        warnings.append(
            "Restart_From_Wfndat is read only together with Restart_Run: true "
            "(parsec.F90); this run starts from the atomic potentials"
        )
        restart_from_wfndat = False
    if restart_from_wfndat and (
        has_kpoints
        or post_processing.nscf is not None
        or post_processing.dos is not None
    ):
        # restart.F90 has no k-points (kpnum = 1).
        raise UnsupportedParsecOptionError(
            "Restart_From_Wfndat: wfn.dat has no k-points (restart.F90); it "
            "restarts a cluster or a Gamma-point cell, without nscf_kpoints or "
            "Create_DOS"
        )
    if restart_from_wfndat and (self_consistent_spin_orbit or noncollinear):
        # The spin-orbit loops restart from the two spin components of a
        # spin-orbit parsec.dat; wfn.dat holds one potential per channel.
        raise UnsupportedParsecOptionError(
            "Restart_From_Wfndat with spin-orbit coupling or non-collinear "
            "magnetism is not implemented: wfn.dat holds no spinor components; "
            "Restart_Run restarts a spin-orbit run from its parsec.dat"
        )
    if restart_run:
        # A non-self-consistent run reads the potential of every spin channel
        # and runs no SCF loop, so it needs none of the restartable drivers.
        # A self-consistent spin-orbit run (isolated, or a bulk cell at Gamma
        # or with k-points) restarts from the parsec.dat of a spin-orbit run
        # (restart_run.F90 with stype(2) = 2).
        isolated_spin_orbit = bool(
            self_consistent_spin_orbit and not noncollinear and not wire_or_slab
        )
        if post_processing.nscf is None and not (plain_loop or isolated_spin_orbit or (has_kpoints and not spin_polarized and not self_consistent_spin_orbit and not noncollinear)):
            raise UnsupportedParsecOptionError(
                "Restart_Run is implemented for the SCF loop of a cluster or a "
                "periodic cell (Gamma point or scalar k-points), for an isolated "
                "spin-polarized input and for a self-consistent spin-orbit run; "
                "not with non-collinear magnetism or periodic spin polarization"
            )
        if initial_density_settings.method != "sad":
            raise UnsupportedParsecOptionError(
                "Restart_Run takes the potential of parsec.dat; Initial_Density: "
                f"{initial_density_settings.method} would not be used: remove it"
            )
        if mlff is not None:
            raise UnsupportedParsecOptionError(
                "Restart_Run takes the potential of parsec.dat for the structure of "
                "the input; Initial_Structure: mlff would move the atoms first"
            )
        if not is_periodic and minimization != "manual":
            from ..V_ion import center_cluster_geometry

            # initial.f90:133-134 centres a cluster only with istart = 0, and
            # parsec.dat stores no coordinates (wfnsave.F90): PARSEC restarts
            # the stored potential at the input coordinates, and so does this
            # port (the same fixed points as the Fortran binary, also when the
            # first run had moved the atoms).
            placed = center_cluster_geometry(atoms)
            moved = max(
                float(np.linalg.norm(np.asarray(a.position) - np.asarray(b.position)))
                for a, b in zip(atoms, placed)
            )
            if moved > 0.0:
                warnings.append(
                    "Restart_Run: the atom coordinates are used as the input gives "
                    "them (PARSEC centres a cluster only in a first run); a first "
                    f"run would have moved them by {moved:.4f} bohr, so the "
                    "parsec.dat of such a run belongs to other coordinates: give "
                    "the centred coordinates that run printed (PARSEC: 'The new "
                    "atom coordinates are'; this port: 'initial coordinates') to "
                    "restart from its potential"
                )

    grid, scf = _periodic_potential_conventions(
        scalar,
        one,
        optional_bool,
        grid,
        scf,
        is_periodic=is_periodic,
        wire_or_slab=wire_or_slab,
        warnings=warnings,
    )
    if is_periodic and not wire_or_slab and not routines.fortran:
        refusal = _default_routines_periodic_refusal(
            periodic_cell,
            grid,
            atoms,
            slab=andrew_slab,
            kpoints=explicit_kpoint_array,
            kpoint_weights=explicit_kpoint_weights,
            monkhorst_pack_dimensions=monkhorst_pack_dimensions,
            spin_polarized=spin_polarized,
            ionic=bool(compute_forces or atoms_move),
            functional=scf.xc_functional,
        )
        if refusal is not None:
            fall_back(refusal)
    if routines.warning() is not None:
        warnings.append(routines.warning())
    if andrew_slab:
        if (
            self_consistent_spin_orbit
            or noncollinear
            or any(specification.spin_orbit for specification in specifications.values())
        ):
            # Andrew Grieder's slab refuses them
            fall_back("spin-orbit coupling or non-collinear magnetism of a slab")
        # in-plane positions are folded into the cell (they are periodic)
        inverse_plane = np.linalg.inv(periodic_cell.lattice_vectors[:2, :2])
        folded = []
        for atom in atoms:
            position = np.array(atom.position, dtype=float)
            fraction = position[:2] @ inverse_plane
            fraction -= np.floor(fraction + 1.0e-12)
            position[:2] = fraction @ periodic_cell.lattice_vectors[:2, :2]
            folded.append(replace(atom, position=position))
        atoms = folded
        if recenter and atoms:
            # Fortran (initial.f90) centres a slab along z when the atoms are off centre by more than 0.1 bohr
            z_values = [float(atom.position[2]) for atom in atoms]
            shift = 0.5 * (min(z_values) + max(z_values))
            if abs(shift) > 0.1:
                warnings.append(
                    f"atom coordinates were shifted by {-shift:.6f} bohr along z so that the slab is centred at the origin"
                )
                atoms = [
                    replace(atom, position=np.asarray(atom.position, dtype=float) - np.array([0.0, 0.0, shift]))
                    for atom in atoms
                ]
        height = float(periodic_cell.lattice_vectors[2, 2])
        if any(abs(float(atom.position[2])) >= 0.5 * height for atom in atoms):
            # Andrew Grieder's slab preparation takes the atoms inside its box only (|z| < R)
            fall_back("a slab atom at |z| >= Boundary_Sphere_Radius")

    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials=specifications,
        grid=grid,
        periodic_cell=periodic_cell,
        scf=scf,
        hartree=hartree,
        eigensolver=eigensolver,
        mixing=mixing,
        initial_density_settings=initial_density_settings,
        recenter_geometry=recenter and not is_periodic,
        monkhorst_pack_dimensions=monkhorst_pack_dimensions,
        kpoints=explicit_kpoint_array,
        kpoint_weights=explicit_kpoint_weights,
        bands=band_settings,
        point_charges=point_charges,
        **({"charged_sheets": charged_sheets} if charged_sheets else {}),
        monkhorst_pack_shift=monkhorst_pack_shift,
        space_group=space_group,
    )
    return ParsecInputTranslation(
        source=source,
        problem=problem,
        warnings=tuple(warnings),
        output_all_states=optional_bool("output_all_states"),
        output_level=_integer(one("output_level", "1"), label="Output_Level"),
        ignore_symmetry=ignore_symmetry,
        domain_energy_tolerance=domain_energy_tolerance,
        domain=domain,
        boundary_tolerance_from=boundary_tolerance_from,
        domain_rule_seconds=domain_rule_seconds,
        relaxation=relaxation,
        movable=tuple(movable),
        compute_forces=compute_forces,
        use_symmetry=use_symmetry,
        molecular_dynamics=molecular_dynamics if molecular_dynamics.enabled else None,
        potential_extrapolation=potential_extrapolation,
        mlff=mlff,
        energy_refinement=energy_refinement,
        polarizability_field=electric_field if polarizability else None,
        point_charge_type_counts=point_charge_type_counts,
        magnetic_field_gauss=magnetic_field / 2.5e9,
        complex_wavefunctions=complex_wavefunctions,
        output_files=output_files,
        restart_run=restart_run,
        restart_from_wfndat=restart_from_wfndat,
        fix_number_eigenvalues=fix_number_eigenvalues,
        dynamic_tolerance_off_for_restart=dynamic_tolerance_off_for_restart,
        super_cell=super_cell,
        super_cell_vacuum=super_cell_vacuum,
        nscf=post_processing.nscf,
        dos=post_processing.dos,
        gw=gw,
        slab_shift=slab_shift,
        routines=routines,
    )


def _periodic_symmetry(
    atoms,
    periodic_cell: PeriodicCell,
    grid: PeriodicGridSettings,
    *,
    general_periodic_grid: bool,
    ignore_symmetry: bool,
    monkhorst_pack_shift,
    symmetrize_density: bool,
    symmetrize_given: bool,
    noncollinear: bool,
    warnings: list[str],
    forces_of_weighted_kpoints: bool = False,
    search_info: int = 0,
):
    """The space group of a periodic input and what it decides.

    Returns ``(symmetrize_density, space_group, grid)``:

    * ``Symmetrize_Charge_Density`` defaults to true where Fortran's k-point
      path is taken (an input with ``Monkhorst_Pack_Shift``), as
      ``usrinputfile.F90:2548-2552`` sets it for every k-point run; other
      periodic inputs keep this port's default, false;
    * the space group (``symmetries.f90``; the identity with
      ``Ignore_Symmetry``) is kept in the problem where the irreducible wedge,
      the symmetrization or the forces use it (``symm_forces.f90``: the forces
      of an unequally weighted ``Kpoint_List`` are completed over it,
      ``forces_of_weighted_kpoints``), ``None`` otherwise;
    * ``search_info`` is the ``ierr`` that ``parsec.F90:464-492`` hands to
      ``symmetries.f90`` (:func:`_bulk_symmetry_search_info`).  2 (molecular
      dynamics, ``manual`` movements): no search, the group is the identity
      (``skipping non-trivial symmetry operations``), so ``adjust_step`` sees
      no translation and with ``Monkhorst_Pack_Shift`` the wedge is that of
      time reversal alone.  0 (a Gamma-point run on a grid shifted by 0.5,
      i.e. orthogonal lattice vectors, without spinors, bands or NSCF):
      ``symgen.f90:209-210`` checks with ``symchk`` that the operations form
      a group modulo the lattice and keeps the identity alone when they do
      not (``ERROR: repeated symmetry operations``; a centred structure in a
      conventional cell, whose translations are fixed only up to its
      centring vectors: the 8-atom diamond cube, also stretched along z).
      The problem then carries no group, as for the identity.  1 (k-points,
      oblique cells, spinors, bands, NSCF): the whole group, unchecked;
    * Fortran's general grid gets the fractional translations ``adjust_step``
      makes its steps commensurate with (``inipbc.f90:327-429``); the
      orthorhombic grid of this port keeps its rule, so a symmetrization it
      is not commensurate with is refused by name, and so is one of the
      general grid when no spacing is commensurate with a translation
      (:func:`parsec_python.Grid.pbc.adjust_step`).

    A non-collinear calculation is refused with a non-trivial group: its
    magnetization is a vector field, which the scalar average of
    ``symm_scalar`` does not treat.
    """

    from ..Grid.space_group import (
        find_space_group,
        grid_commensurability_problem,
        identity_space_group,
        skipped_space_group,
    )
    from ..Grid.symmetry_analysis import symmetry_check

    fortran_kpoints = monkhorst_pack_shift is not None
    if not symmetrize_given and fortran_kpoints:
        symmetrize_density = True
    if not (
        symmetrize_density
        or fortran_kpoints
        or general_periodic_grid
        or forces_of_weighted_kpoints
    ):
        return symmetrize_density, None, grid
    failed_check = False
    if ignore_symmetry:
        group = identity_space_group()
    elif search_info == 2:
        # parsec.F90 l.483-485: no symmetry operations for molecular dynamics
        # and manual movements (ierr = 2): adjust_step sees no translation
        # and the wedge of a k-point grid is that of time reversal alone.
        group = skipped_space_group()
        if forces_of_weighted_kpoints:
            # The structure's own group instead, so that the movement check
            # (_check_reduced_kpoint_movements) refuses by name the unequally
            # weighted Kpoint_List (a wedge of that group) such a run cannot
            # follow.
            try:
                group = find_space_group(periodic_cell.lattice_vectors, atoms)
            except ValueError as error:
                raise ParsecInputError(f"symmetry operations: {error}") from error
    else:
        try:
            group = find_space_group(periodic_cell.lattice_vectors, atoms)
        except ValueError as error:
            raise ParsecInputError(f"symmetry operations: {error}") from error
        if search_info == 0 and symmetry_check(group).error:
            # symgen.f90 l.209-210: the operations do not form a group modulo
            # the lattice (symchk), and PARSEC keeps the identity alone.
            group = identity_space_group(ignored=False)
            failed_check = True
        if general_periodic_grid:
            grid = replace(grid, fractional_translations=group.commensurate_translations())
    if symmetrize_density and noncollinear:
        if group.order > 1:
            # PARSEC (newrho.F90, parsec.F90:1223-1232) averages rho and the
            # local-frame densities (rho +- |m|)/2 as scalars and keeps the
            # direction of m: SCF/noncollinear.py does the same.
            warnings.append(
                "Symmetrize_Charge_Density with Non_Collinear_magnetism averages "
                "the density and |m| as scalars and keeps the direction of the "
                "magnetization, as PARSEC does"
            )
        else:
            if symmetrize_given:
                warnings.append(
                    "Symmetrize_Charge_Density changes nothing here: the group has "
                    "the identity alone"
                )
            symmetrize_density = False
    problem = None
    if general_periodic_grid and group.order > 1:
        from ..Grid.pbc import adjust_step

        lengths = np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
        translations = grid.fractional_translations or ((), (), ())
        try:
            counts = [
                adjust_step(length, grid.spacing, translations=taus)[0]
                for length, taus in zip(lengths, translations)
            ]
        except ValueError:
            counts = None  # the grid builder says why
        if counts is not None:
            problem = grid_commensurability_problem(group, counts, grid.shift)
        if problem is not None and not symmetrize_density:
            warnings.append(
                "symmetry operations: " + problem + ": no grid spacing near "
                "Grid_Spacing is commensurate with that fractional translation; the "
                f"grid has {'x'.join(str(count) for count in counts)} points, the number "
                "PARSEC takes (2 nint(L/2h)), with the step L/n (PARSEC keeps the input "
                "spacing, a grid that does not span the cell)"
            )
    if symmetrize_density and general_periodic_grid:
        if problem is not None:
            raise UnsupportedParsecOptionError(
                "Symmetrize_Charge_Density: " + problem + ": no grid spacing is "
                "commensurate with that fractional translation (PARSEC's adjust_step "
                "then keeps the input spacing, and symm_scalar averages over points "
                "that are not images of each other).  Set Ignore_Symmetry: true, or "
                "Symmetrize_Charge_Density: false (with Monkhorst_Pack_Shift the "
                "density of the irreducible wedge then misses its rotated k-points)"
            )
    if symmetrize_density and not general_periodic_grid:
        side_lengths = np.diag(periodic_cell.lattice_vectors)
        counts = np.round(side_lengths / grid.spacing).astype(np.int64)
        problem = grid_commensurability_problem(group, counts, (0.0, 0.0, 0.0))
        if problem is not None:
            raise UnsupportedParsecOptionError(
                "Symmetrize_Charge_Density: " + problem + ".  PARSEC chooses its "
                "grid spacing commensurate with the fractional translations "
                "(adjust_step); this orthorhombic cell keeps the grid of this "
                "port (n = round(L/h), no shift).  Choose a Grid_Spacing for which "
                "every n_i times the translations is a whole number, or set "
                "Symmetrize_Charge_Density: false (with Monkhorst_Pack_Shift the "
                "density of the irreducible wedge then misses its rotated "
                "k-points) or Ignore_Symmetry: true"
            )
    if fortran_kpoints and not symmetrize_density and group.order > 1:
        warnings.append(
            "Symmetrize_Charge_Density: false with a Monkhorst_Pack_Shift grid: "
            "the irreducible k-points alone do not give the density of the "
            "cell, which only the average over the space group restores "
            "(PARSEC does the same)"
        )
    if forces_of_weighted_kpoints and not symmetrize_density and group.order > 1:
        warnings.append(
            "Kpoint_List with unequal weights and forces: the forces of the listed "
            "k-points are completed over the space group (an irreducible wedge), "
            "but without Symmetrize_Charge_Density: true the density is that of "
            "the listed points alone"
        )
    keep = (
        symmetrize_density or fortran_kpoints or forces_of_weighted_kpoints
    ) and not failed_check
    return symmetrize_density, (group if keep else None), grid


def _bulk_symmetry_search_info(
    periodic_cell: PeriodicCell,
    *,
    ignore_symmetry: bool,
    kpoints: bool,
    spinors: bool,
    bands: bool,
    nscf: bool,
    molecular_dynamics: bool,
    manual_movement: bool,
) -> int:
    """The ``ierr`` ``parsec.F90:464-492`` hands to ``symmetries.f90`` for a bulk cell.

    :func:`parsec_python.Grid.symmetry_analysis.symmetry_search_mode` with
    Fortran's grid shift (``usrinputfile.F90:786-794``: 0.5 along orthogonal
    lattice vectors, 0 along oblique ones and with ``Ignore_Symmetry``),
    which the report of the run uses as well.
    """

    from ..Grid.pbc import periodic_grid_shift
    from ..Grid.symmetry_analysis import symmetry_search_mode

    return symmetry_search_mode(
        ignore_symmetry=ignore_symmetry,
        kpoints=kpoints,
        grid_shift=periodic_grid_shift(
            periodic_cell.lattice_vectors, ignore_symmetry=ignore_symmetry
        ),
        periodicity=3,
        spinors=spinors,
        bands=bands,
        nscf=nscf,
        molecular_dynamics=molecular_dynamics,
        polarizability=False,
        manual_movement=manual_movement,
        non_spherical_cluster=False,
    ).info


def _wire_slab_symmetry(
    atoms,
    periodic_cell: PeriodicCell,
    grid: PeriodicGridSettings,
    *,
    periodicity: int,
    ignore_symmetry: bool,
    monkhorst_pack_shift,
    symmetrize_density: bool,
    symmetrize_given: bool,
    noncollinear: bool,
    spin_orbit: bool,
    bands: bool,
    nscf: bool,
    no_symmetry: bool,
    warnings: list[str],
):
    """The space group of a wire or slab and what it decides.

    Returns ``(symmetrize_density, space_group, grid)``.  Fortran's choice
    (``parsec.F90:470-486``): ``Ignore_Symmetry`` suppresses the search; a
    slab gets none (``ierr = 2``) unless band structure, spin-orbit /
    non-collinear or an NSCF run sets ``ierr = 1`` after it; molecular
    dynamics, Polarizability and manual movements get none.  Otherwise
    ``symmetries.f90`` searches with ``alatt`` = ``latt_vec`` and ``4 rmax``
    along the confined directions
    (:func:`parsec_python.Grid.space_group.find_wire_slab_space_group`); the
    operations that do not map the confined grid onto itself are dropped (a
    Fortran defect: with ``bandstruc`` a slab's symmetrization averages
    over points that are not images of each other).  The group is always
    kept in the problem (its block is printed, the forces are averaged over
    it); the fractional translations along the periodic directions make
    ``adjust_step`` commensurate, ``Monkhorst_Pack_Shift`` reduces the
    k-points with it and turns ``Symmetrize_Charge_Density`` on by default
    (``usrinputfile.F90:2548-2552``).
    """

    from ..Grid.space_group import (
        find_wire_slab_space_group,
        identity_space_group,
        skipped_space_group,
    )

    fortran_kpoints = monkhorst_pack_shift is not None
    if not symmetrize_given and fortran_kpoints:
        symmetrize_density = True
    if ignore_symmetry:
        group = identity_space_group()
    elif no_symmetry or (periodicity == 2 and not (bands or spin_orbit or nscf)):
        group = skipped_space_group()
    else:
        try:
            group, dropped = find_wire_slab_space_group(
                periodic_cell.lattice_vectors, periodicity, grid.confinement_radius, atoms
            )
        except ValueError as error:
            raise ParsecInputError(f"symmetry operations: {error}") from error
        if dropped:
            warnings.append(
                f"{dropped} symmetry operation(s) of PARSEC's search are not used: "
                "they translate along a confined direction (or mix it with a "
                "periodic one) and do not map the wire or slab grid onto itself "
                "(PARSEC applies them anyway)"
            )
        translations = group.commensurate_translations(periodicity)
        if any(translations):
            grid = replace(
                grid,
                fractional_translations=tuple(translations) + ((),) * (3 - periodicity),
            )
    if symmetrize_density and noncollinear and group.order > 1:
        raise UnsupportedParsecOptionError(
            "Symmetrize_Charge_Density with Non_Collinear_magnetism is not "
            "implemented: the magnetization is a vector field, which the "
            "scalar average of symm_scalar does not treat (PARSEC averages "
            "its components as scalars).  Set Symmetrize_Charge_Density: "
            "false or Ignore_Symmetry: true"
        )
    if fortran_kpoints and not symmetrize_density and group.order > 1:
        warnings.append(
            "Symmetrize_Charge_Density: false with a Monkhorst_Pack_Shift grid: "
            "the irreducible k-points alone do not give the density of the "
            "cell, which only the average over the space group restores "
            "(PARSEC does the same)"
        )
    return symmetrize_density, group, grid


def _check_reduced_kpoint_movements(
    atoms,
    periodic_cell: PeriodicCell,
    space_group,
    *,
    movable,
    relaxation: bool,
    molecular_dynamics: bool,
    reduced: bool,
    ignore_symmetry: bool,
    symmetry_lattice=None,
) -> None:
    """Refuse movements a reduced k-point set cannot follow.

    The forces of a k-point wedge are completed over the space group
    (:func:`parsec_python.Forces.periodic.reduced_kpoint_operations`), which
    holds only while the structure keeps its space group.  A relaxation keeps
    it when its movable flags are mapped onto movable flags by every
    operation; molecular dynamics and ``manual`` movements keep no group
    (their ``Monkhorst_Pack_Shift`` wedge is the identity's, see
    :func:`_periodic_symmetry`), which an unequally weighted ``Kpoint_List``
    would need.  (A symmetrized density with an unreduced k-point set is
    left as it was.)  ``symmetry_lattice`` is the lattice of the group's
    ``M`` and ``t`` where it is not the cell's (``alatt`` of a wire or slab,
    :func:`_check_wire_slab_kpoint_movements`).
    """

    if space_group is None or space_group.order <= 1 or ignore_symmetry or not reduced:
        return
    if molecular_dynamics:
        raise UnsupportedParsecOptionError(
            "Molecular_Dynamics and manual movements use no symmetry (as PARSEC), "
            "so the structure leaves the space group that an unequally weighted "
            "Kpoint_List (an irreducible wedge) relies on; give a complete "
            "k-point set with equal weights, or Monkhorst_Pack_Shift (its wedge "
            "is then that of time reversal alone)"
        )
    if not relaxation:
        return
    from ..Grid.space_group import space_group_atom_images

    mask = np.asarray(movable, dtype=bool)
    images = space_group_atom_images(
        space_group,
        periodic_cell.lattice_vectors if symmetry_lattice is None else symmetry_lattice,
        atoms,
    )
    if any(not np.array_equal(mask[row], mask) for row in images):
        raise UnsupportedParsecOptionError(
            "Move_Flag fixes atoms that a space-group operation exchanges with "
            "movable ones: the relaxation would leave the space group that the "
            "irreducible k-point wedge relies on; "
            "move them all, or set Ignore_Symmetry: true"
        )


def _check_wire_slab_kpoint_movements(
    atoms,
    periodic_cell: PeriodicCell,
    grid,
    space_group,
    *,
    periodicity: int,
    movable,
    relaxation: bool,
    molecular_dynamics: bool,
    weighted_kpoints: bool,
    monkhorst_pack_shift,
    ignore_symmetry: bool,
    slab_symmetry_on: bool,
) -> None:
    """:func:`_check_reduced_kpoint_movements` for a wire or slab.

    The forces of a wire or slab are averaged over the group the input
    reader keeps (:func:`_wire_slab_symmetry`), which completes an
    irreducible wedge (``Monkhorst_Pack_Shift`` with a group, or an unequally
    weighted ``Kpoint_List``).  The bulk rules apply with that group and its
    lattice ``alatt``: a relaxation whose ``Move_Flag`` a group operation
    does not respect (the ionic loop keeps the respecting operations only)
    and molecular dynamics or ``manual`` movements with a weighted list are
    refused.  Molecular dynamics switches the group off (``parsec.F90:481``,
    ``ierr = 2``), so for that test the group is the one the structure would
    have without it; a slab without band structure, spin-orbit coupling or
    NSCF has none in PARSEC (its weighted list is a plain quadrature there,
    as with ``Ignore_Symmetry``).  ``Monkhorst_Pack_Shift`` with molecular
    dynamics is the wedge of time reversal alone (no group): accepted.
    """

    from ..Grid.space_group import confined_symmetry_lattice, find_wire_slab_space_group

    if ignore_symmetry or not (weighted_kpoints or monkhorst_pack_shift is not None):
        return
    lattice = confined_symmetry_lattice(
        periodic_cell.lattice_vectors, periodicity, grid.confinement_radius
    )
    group = space_group
    if molecular_dynamics and weighted_kpoints:
        if periodicity == 2 and not slab_symmetry_on:
            return
        try:
            group, _ = find_wire_slab_space_group(
                periodic_cell.lattice_vectors, periodicity, grid.confinement_radius, atoms
            )
        except ValueError as error:
            raise ParsecInputError(f"symmetry operations: {error}") from error
    _check_reduced_kpoint_movements(
        atoms,
        periodic_cell,
        group,
        movable=movable,
        relaxation=relaxation,
        molecular_dynamics=molecular_dynamics and weighted_kpoints,
        reduced=True,
        ignore_symmetry=ignore_symmetry,
        symmetry_lattice=lattice,
    )


def _save_wfn_bands(lines: tuple[str, ...]) -> tuple[int, ...]:
    """The state numbers of a ``Save_Wfn_Bands`` block (``usrinputfile.F90``).

    An empty block saves none; one line ``lo hi`` saves ``lo..hi``; one line
    with one number, or one number per line, saves those states.  PARSEC
    reads list-directed, so commas separate numbers too and numbers after
    the first two of a single line are ignored.
    """

    def numbers(line: str) -> list[str]:
        return [token for token in re.split(r"[\s,]+", line.strip()) if token]

    def integer(token: str) -> int:
        try:
            value = float(token.replace("d", "e").replace("D", "E"))
        except ValueError:
            raise ParsecInputError(
                'bad specification of "Save_Wfn_Bands" block: '
                f"{token!r} is not an integer"
            ) from None
        if value != int(value):
            raise ParsecInputError(
                f'bad specification of "Save_Wfn_Bands" block: {token!r} is not an integer'
            )
        return int(value)

    if not lines:
        return ()
    if len(lines) == 1:
        tokens = numbers(lines[0])
        if len(tokens) >= 2:
            low, high = integer(tokens[0]), integer(tokens[1])
            if high < low:
                raise ParsecInputError(
                    f"Save_Wfn_Bands range {low} {high} is empty; give 'first last' "
                    "with first <= last"
                )
            states = tuple(range(low, high + 1))
        elif tokens:
            states = (integer(tokens[0]),)
        else:
            states = ()
    else:
        states = []
        for line in lines:
            tokens = numbers(line)
            if not tokens:
                raise ParsecInputError('bad specification of "Save_Wfn_Bands" block')
            states.append(integer(tokens[0]))
        states = tuple(states)
    if any(value < 1 for value in states):
        raise ParsecInputError("Save_Wfn_Bands state numbers start at 1")
    return states


def _output_grid_data(value: str) -> tuple[str, ...]:
    """The fields of ``Output_Grid_Data`` (``usrinputfile.F90`` l.2598-2638).

    ``esdf_reduce`` lowers the case and removes dots and hyphens;
    ``esdf_split`` splits the words at commas (a semicolon starts a comment
    in this reader, and PARSEC notes that everything after one is ignored).
    ``none`` ends the list, ``all`` gives the four fields.
    """

    reduced = re.sub(r"[.\-]", "", value.lower()).strip()
    words = [word.strip() for word in reduced.split(",")][:10]
    fields: list[str] = []
    for word in words:
        if word == "none":
            break
        if word == "all":
            # PARSEC sets the first four flags and stops reading.
            return GRID_DATA_FIELDS
        if word not in GRID_DATA_FIELDS:
            raise ParsecInputError(f"Wrong input to output_grid_data: {word}")
        fields.append(word)
    return tuple(fields)


def _output_file_settings(scalar, blocks, one, optional_bool, has_kpoints: bool) -> OutputFileSettings:
    """``Output_Data`` and the other output-file keywords of ``usrinputfile.F90``."""

    keywords = {
        key
        for key in (
            "output_data",
            "output_all_states",
            "output_eigenvalues",
            "save_intermediate_eigenvalues",
            "save_intermediate_charge_density",
            "output_grid_data",
            "output_file_name",
        )
        if key in scalar
    }
    all_states = optional_bool("output_all_states")
    if all_states:
        selection = WavefunctionSelection(mode="all", explicit=True)
    elif "save_wfn_bands" in blocks:
        band_blocks = blocks["save_wfn_bands"]
        if len(band_blocks) != 1:
            raise ParsecInputError("provide a single Save_Wfn_Bands block")
        selection = WavefunctionSelection(
            mode="list", indices=_save_wfn_bands(band_blocks[0].value), explicit=True
        )
        keywords.add("save_wfn_bands")
    else:
        selection = WavefunctionSelection()
    name = one("output_file_name", "parsec.out").strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in {"'", '"'}:
        name = name[1:-1].strip()
    if not name:
        raise ParsecInputError("Output_File_Name cannot be empty")
    return OutputFileSettings(
        output_data=optional_bool("output_data", True),
        wavefunctions=selection,
        intermediate_density=optional_bool("save_intermediate_charge_density"),
        # usrinputfile.F90 l.2645: eigen.dat is always written with k-points.
        eigenvalues=optional_bool("output_eigenvalues") or has_kpoints,
        all_states=all_states,
        intermediate_eigenvalues=(
            (optional_bool("output_eigenvalues") or has_kpoints)
            and optional_bool("save_intermediate_eigenvalues")
        ),
        grid_data=_output_grid_data(one("output_grid_data", "none")),
        output_file_name=name,
        keywords=frozenset(keywords),
    )


@dataclass(frozen=True)
class _PostProcessingInput:
    """What :func:`_post_processing_input` reads: NSCF, DOS, Calc_Hartree_G."""

    nscf: NSCFSettings | None
    dos: DOSSettings | None
    g_space_hartree: bool
    monkhorst_pack_dimensions: tuple[int, int, int] | None
    kpoints: np.ndarray | None
    kpoint_weights: np.ndarray | None
    # Output_GW with nscf_kpoints: nscf_kgrid, nscf_kgrid_shift and
    # nscf_fermi_level (Ry); ``None`` otherwise.
    gw_kgrid: tuple[int, int, int] | None = None
    gw_kshift: tuple[float, float, float] | None = None
    gw_fermi_level: float | None = None
    # Create_DOS with Restart_Run on Fortran's Monkhorst-Pack path: the shift
    # of the restarted SCF (``mpshift = dos_mpshift``), ``None``: unchanged.
    monkhorst_pack_shift: tuple[float, float, float] | None = None


#: The NSCF keys ``usrinputfile.F90`` reads only with ``Output_GW``.
_NSCF_GW_KEYS = ("nscf_kgrid", "nscf_kgrid_shift", "nscf_fermi_level")


def _nscf_gw_keys(scalar, blocks, one) -> tuple[tuple[int, int, int], tuple[float, float, float], float]:
    """``nscf_kgrid``, ``nscf_kgrid_shift`` and ``nscf_fermi_level`` (``usrinputfile.F90`` l.2492-2543).

    PARSEC reads them with ``Output_GW`` in a non-self-consistent run and
    stops (error 114) when one is missing; a block of more than one line is
    error 114 (kgrid) or 116 (shift), a negative shift error 117.  The grid
    must hold three integers of at least one (what BerkeleyGW's reader checks;
    PARSEC does not), the Fermi level is an ESDF physical value (default Ry).
    """

    grid_blocks = blocks.get("nscf_kgrid", [])
    if not grid_blocks:
        raise ParsecInputError(
            "Output_GW with nscf_kpoints: Have to specify nscf_kgrid (PARSEC error 114)"
        )
    if len(grid_blocks) != 1 or len(grid_blocks[0].value) != 1:
        raise ParsecInputError(
            "error in specifying kgrid for nscf calc: give one nscf_kgrid line "
            "(PARSEC error 114)"
        )
    values, unit = _block_numbers(grid_blocks[0].value, label="nscf_kgrid")
    if unit or len(values) != 3 or any(value != int(value) for value in values):
        raise ParsecInputError("nscf_kgrid: give three integers")
    kgrid = tuple(int(value) for value in values)
    if any(value < 1 for value in kgrid):
        raise ParsecInputError(
            "nscf_kgrid: every dimension must be at least 1 (BerkeleyGW refuses a "
            "smaller k-grid in the WFN header)"
        )
    shift_blocks = blocks.get("nscf_kgrid_shift", [])
    if not shift_blocks:
        raise ParsecInputError(
            "Output_GW with nscf_kpoints: Have to specify nscf_kgrid_shift (PARSEC error 114)"
        )
    if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
        raise ParsecInputError("error in specifying Monkhorst-Pack Shift (PARSEC error 116)")
    values, unit = _block_numbers(shift_blocks[0].value, label="nscf_kgrid_shift")
    if unit or len(values) != 3 or any(value < 0 for value in values):
        raise ParsecInputError("error in specifying Monkhorst-Pack Shift (PARSEC error 117)")
    kshift = tuple(float(value) for value in values)
    if "nscf_fermi_level" not in scalar:
        raise ParsecInputError(
            "ERROR: specification of Fermi level for non self consistent calculation "
            "not found! Please specify nscf_fermi_level taken from the scf calculation "
            "(PARSEC error 114)"
        )
    fermi_level = _esdf_energy_rydberg(one("nscf_fermi_level"), label="nscf_fermi_level")
    return kgrid, kshift, fermi_level


def _post_processing_input(
    scalar,
    blocks,
    one,
    optional_bool,
    *,
    is_periodic: bool,
    wire_or_slab: bool = False,
    periodicity: int = 3,
    periodic_cell: PeriodicCell | None,
    input_cell: PeriodicCell | None,
    restart_run: bool,
    states: int,
    symbols: list[str],
    monkhorst_pack_dimensions: tuple[int, int, int] | None,
    kpoints: np.ndarray | None,
    kpoint_weights: np.ndarray | None,
    spin_orbit: bool,
    warnings: list[str],
    output_gw: bool = False,
    noncollinear: bool = False,
    monkhorst_pack_shift: tuple[float, float, float] | None = None,
) -> _PostProcessingInput:
    """``nscf_*``, ``Create_DOS`` and its keys, ``Calc_Hartree_G`` (``usrinputfile.F90``).

    NSCF (l.1056-1118): ``nscf_kpoints`` (``kx ky kz w`` rows, ``w >= 0``,
    error 119; weights renormalized), ``nscf_kpoint_unit`` (the reciprocal
    lattice of the input cell ``lattvec``), ``nscf_states_num`` (default
    ``States_Num``), and ``Restart_Run`` (error 114).  ``nscf_kgrid``,
    ``nscf_kgrid_shift`` and ``nscf_fermi_level`` are read only with
    ``Output_GW`` (l.2493-2543, :func:`_nscf_gw_keys`), and are inert without.
    DOS (l.1122-1197, 1913): ``Create_DOS``, ``DOS_Pnum``, and for a cell
    ``DOS_MP_Grid``, ``DOS_MP_Shift``, ``YLM_DOS_L`` and the per-species
    ``ylm_dos_cutoff``; with ``Restart_Run`` the SCF becomes a Monkhorst-Pack
    run on the DOS grid (l.1188-1193), so the returned k-point sampling
    replaces the input's.  ``Calc_Hartree_G`` (l.907-912, bulk only).
    Keys PARSEC does not read in a given setting are accepted with a warning.
    """

    nscf_keys_given = sorted(
        {key for key in ("nscf_kpoint_unit", "nscf_states_num", "nscf_fermi_level") if key in scalar}
        | {f"begin {key}" for key in ("nscf_kpoints", "nscf_kgrid", "nscf_kgrid_shift") if key in blocks}
    )
    nscf = None
    gw_kgrid = gw_kshift = gw_fermi_level = None
    gw_keys = [key for key in _NSCF_GW_KEYS if key in scalar or key in blocks]
    if not is_periodic:
        if nscf_keys_given:
            warnings.append(
                f"{', '.join(nscf_keys_given)}: PARSEC reads the non-self-consistent "
                "keywords for periodic systems only; they have no effect here"
            )
    elif "nscf_kpoints" in blocks:
        nscf_blocks = blocks["nscf_kpoints"]
        if len(nscf_blocks) != 1:
            raise ParsecInputError("provide a single nscf_kpoints block")
        rows = []
        for line in nscf_blocks[0].value:
            values, unit = _block_numbers((line,), label="nscf_kpoints")
            if unit or len(values) != 4:
                raise ParsecInputError(
                    f"line {nscf_blocks[0].line}: every nscf_kpoints row must be "
                    "'kx ky kz weight' (four numbers, as usrinputfile.F90 reads it)"
                )
            rows.append(values)
        if not rows:
            raise ParsecInputError("the nscf_kpoints block is empty")
        array = np.asarray(rows, dtype=float)
        if np.any(array[:, 3] < 0.0):
            raise ParsecInputError(
                "nscf_kpoints: Sampling weight must be positive (PARSEC error 119)"
            )
        unit_name = _normalize_label(one("nscf_kpoint_unit", "cartesian_inverse_bohr"))
        try:
            k_points, weights = explicit_kpoints(array, unit=unit_name, cell=input_cell)
        except ValueError as error:
            raise ParsecInputError(f"nscf_kpoints: {error}") from error
        if wire_or_slab and np.any(np.abs(k_points[:, periodicity:]) > 1.0e-12):
            # calc_nscf.F90 would add -2i k.grad along a confined direction.
            raise ParsecInputError(
                "nscf_kpoints: a wire or slab has Bloch vectors along its periodic "
                "direction(s) only; the k-point components along the confined "
                "direction(s) must be zero"
            )
        number = _integer(one("nscf_states_num", str(states)), label="nscf_States_Num")
        if number < 1:
            raise ParsecInputError("nscf_States_Num must be positive")
        if not restart_run:
            raise ParsecInputError(
                "In case of non self consistent calculations please specify the "
                "restart flag (Restart_Run: .true.) to read the potential from a "
                "previous run (PARSEC error 114)"
            )
        if noncollinear:
            # A non-collinear parsec.dat (stype(2) 3 or 4) stores no direction
            # of the magnetization; restart_run.F90 restarts nothing from it.
            raise UnsupportedParsecOptionError(
                "nscf_kpoints with non-collinear magnetism is not implemented: the "
                "parsec.dat of a non-collinear run stores no magnetization direction "
                "(restart_run.F90 reads spin-orbit files only); a spin-orbit run "
                "(SO_from_scratch/SCF_SO without Non_Collinear_magnetism) is supported"
            )
        nscf = NSCFSettings(k_points=k_points, weights=weights, number_of_states=number)
        if output_gw:
            gw_kgrid, gw_kshift, gw_fermi_level = _nscf_gw_keys(scalar, blocks, one)
        elif gw_keys:
            warnings.append(
                f"{', '.join(gw_keys)}: PARSEC reads them only with Output_GW; they "
                "have no effect: the NSCF Fermi level is the one of the SCF that wrote "
                "parsec.dat"
            )
    elif nscf_keys_given:
        warnings.append(
            f"{', '.join(nscf_keys_given)} without a nscf_kpoints block: PARSEC reads "
            "them only for a non-self-consistent run; they have no effect"
        )

    # Create_DOS and DOS_Pnum are read in every run (l.1122-1124).
    create_dos = optional_bool("create_dos")
    points = _integer(one("dos_pnum", "1000"), label="DOS_Pnum")
    periodic_dos_keys = sorted(
        {f"begin {key}" for key in ("dos_mp_grid", "dos_mp_shift") if key in blocks}
        | {key for key in ("ylm_dos_l",) if key in scalar}
    )
    dos = None
    if create_dos and points < 1:
        raise ParsecInputError("DOS_Pnum must be positive")
    if not create_dos:
        given = periodic_dos_keys + (["dos_pnum"] if "dos_pnum" in scalar else [])
        if given:
            warnings.append(
                f"{', '.join(given)} without Create_DOS: .true. have no effect, as in PARSEC"
            )
    elif not is_periodic:
        if periodic_dos_keys:
            warnings.append(
                f"{', '.join(periodic_dos_keys)}: PARSEC reads them for periodic "
                "systems only; a cluster's DOS is the histogram of its occupations"
            )
        dos = DOSSettings(points=points)
    else:
        # A wire or slab: dos.F90 takes its Monkhorst-Pack grid (one point
        # along the confined directions) as a three-dimensional one, so the
        # tetrahedra are flat along those directions and the linear
        # interpolation runs along the periodic ones.
        if spin_orbit:
            raise UnsupportedParsecOptionError(
                "Create_DOS with spin-orbit coupling or non-collinear magnetism is not "
                "implemented (dos.F90 reads an undefined spin index for spinors)"
            )
        if nscf is not None:
            raise UnsupportedParsecOptionError(
                "Create_DOS together with nscf_kpoints is not implemented: PARSEC "
                "would build the tetrahedra of the SCF grid on the NSCF levels"
            )
        grid_blocks = blocks.get("dos_mp_grid", [])
        dos_grid = None
        if grid_blocks:
            if len(grid_blocks) != 1 or len(grid_blocks[0].value) != 1:
                raise ParsecInputError(
                    "error in specifying DOS MP Grid: give one DOS_MP_Grid block of one line"
                )
            values, unit = _block_numbers(grid_blocks[0].value, label="DOS_MP_Grid")
            if wire_or_slab and not unit and len(values) >= periodicity:
                # usrinputfile.F90:1134-1135 reads dos_mpgrid(1:per).
                values = list(values[:periodicity]) + [1.0] * (3 - periodicity)
            if (
                unit
                or len(values) != 3
                or any(value <= 0 or value != int(value) for value in values)
            ):
                raise ParsecInputError(
                    "error in specifying DOS MP Grid: three positive integers"
                )
            dos_grid = tuple(int(value) for value in values)
        dos_shift = None
        restart_shift = None
        if "dos_mp_shift" in blocks:
            if dos_grid is None:
                warnings.append(
                    "DOS_MP_Shift without DOS_MP_Grid is not read, as in PARSEC"
                )
            else:
                shift_blocks = blocks["dos_mp_shift"]
                if len(shift_blocks) != 1 or len(shift_blocks[0].value) != 1:
                    raise ParsecInputError("error in specifying DOS MP Shift")
                values, unit = _block_numbers(shift_blocks[0].value, label="DOS_MP_Shift")
                if wire_or_slab and not unit and len(values) >= periodicity:
                    values = list(values[:periodicity]) + [0.0] * (3 - periodicity)
                if unit or len(values) != 3 or any(value < 0 for value in values):
                    raise ParsecInputError("error in specifying DOS MP Shift")
                dos_shift = tuple(float(value) for value in values)
        manual = kpoints is not None
        if dos_grid is None and manual:
            if restart_run:
                raise UnsupportedParsecOptionError(
                    "Create_DOS with Restart_Run and a Kpoint_List needs DOS_MP_Grid: "
                    "PARSEC then runs the SCF on that Monkhorst-Pack grid"
                )
            warnings.append(
                "DOS cannot be created based on a manual MP grid, please specify a DOS "
                "MP grid. DOS creation aborted ! (as PARSEC)"
            )
        else:
            if dos_grid is None:
                dos_grid = monkhorst_pack_dimensions
            if restart_run:
                # usrinputfile.F90 l.1188-1193: the restarted SCF runs on the
                # Monkhorst-Pack grid of the DOS.
                if dos_grid is None:
                    raise UnsupportedParsecOptionError(
                        "Create_DOS with Restart_Run for a Gamma-point input needs "
                        "DOS_MP_Grid"
                    )
                if dos_shift is not None or monkhorst_pack_shift is not None:
                    # PARSEC's Monkhorst-Pack path (the shift block of the SCF
                    # or DOS_MP_Shift): mpshift = dos_mpshift, the block of
                    # DOS_MP_Shift, zero without it when DOS_MP_Grid is given,
                    # and the shift of the SCF without DOS_MP_Grid.  Without
                    # either block this port's grid of the DOS divisions is
                    # used, as for every input without a shift block.
                    if "dos_mp_grid" in blocks:
                        restart_shift = dos_shift if dos_shift is not None else (0.0, 0.0, 0.0)
                    else:
                        restart_shift = monkhorst_pack_shift
                    dos_shift = restart_shift
                monkhorst_pack_dimensions = dos_grid
                kpoints = None
                kpoint_weights = None
            elif monkhorst_pack_dimensions is None:
                raise UnsupportedParsecOptionError(
                    "Create_DOS for a periodic cell needs the Monkhorst-Pack k-points of "
                    "the SCF (Kpoint_Method: mp), or DOS_MP_Grid with Restart_Run (PARSEC "
                    "stops: 'DOS: kpoint map is not set up!')"
                )
            else:
                if dos_shift is not None:
                    warnings.append(
                        "DOS_MP_Shift has no effect without Restart_Run: the DOS uses the "
                        "k-points of the SCF, as in PARSEC"
                    )
                if dos_grid != monkhorst_pack_dimensions:
                    warnings.append(
                        "You are creating the DOS using the original MP Grid; in order to "
                        "use the DOS grid perform a restart run with the same input (as "
                        "PARSEC)"
                    )
                dos_grid = monkhorst_pack_dimensions
            ylm_l = _integer(one("ylm_dos_l", "-1"), label="YLM_DOS_L")
            if ylm_l > 3:
                raise UnsupportedParsecOptionError(
                    "YLM_DOS_L above 3: dos.F90 has the real harmonics s, p, d and f only"
                )
            ylm_l = max(ylm_l, -1)
            occurrences = scalar.get("ylm_dos_cutoff", [])
            cutoffs = []
            for index, symbol in enumerate(symbols):
                text = str(occurrences[index].value) if index < len(occurrences) else "1.0"
                radius = _physical_length(text, label=f"{symbol} ylm_dos_cutoff")
                if not radius > 0.0:
                    raise ParsecInputError(f"Atom_Type {symbol}: ylm_dos_cutoff must be positive")
                cutoffs.append((symbol, radius))
            dos = DOSSettings(
                points=points,
                monkhorst_pack_grid=dos_grid,
                monkhorst_pack_shift=dos_shift,
                ylm_l=ylm_l,
                ylm_cutoffs=tuple(cutoffs),
                fortran_projection=optional_bool("ylm_dos_fortran_projection"),
            )
    if "ylm_dos_fortran_projection" in scalar and (dos is None or dos.ylm_l < 0):
        warnings.append("YLM_DOS_Fortran_Projection without a projected DOS has no effect")

    g_space = False
    if optional_bool("calc_hartree_g"):
        # usrinputfile.F90:907 reads it in the bulk case (pbc%per = 3) only.
        if not is_periodic or wire_or_slab:
            warnings.append(
                "Calc_Hartree_G is read for a bulk cell only, as in PARSEC; no effect here"
            )
        else:
            g_space = True
            warnings.append(
                "Calc_Hartree_G: PARSEC's hpot_G is broken (it prints 'IT WILL NOT "
                "WORK': it reads the star-averaged density with the index of each "
                "G-vector, and its Hartree energy is not used); the Python port solves "
                "the intended V_H(G) = 8 pi rho(G)/|G|^2 exactly by FFT "
                "(parsec_python.Hartree.gspace). The default finite-difference solve "
                "is unchanged without the keyword"
            )
    return _PostProcessingInput(
        nscf=nscf,
        dos=dos,
        g_space_hartree=g_space,
        monkhorst_pack_dimensions=monkhorst_pack_dimensions,
        kpoints=kpoints,
        kpoint_weights=kpoint_weights,
        gw_kgrid=gw_kgrid,
        gw_kshift=gw_kshift,
        gw_fermi_level=gw_fermi_level,
        monkhorst_pack_shift=restart_shift if create_dos and is_periodic else None,
    )


def _gw_settings(
    post_processing: _PostProcessingInput,
    *,
    atoms,
    periodic_cell: PeriodicCell | None,
    is_periodic: bool,
    wire_or_slab: bool,
    spinors: bool,
    bands: bool,
    moves: bool,
    forces: bool,
    monkhorst_pack_dimensions,
    monkhorst_pack_shift,
    explicit_kpoints: bool,
    space_group,
    ignore_symmetry: bool,
    warnings: list[str],
) -> GWSettings:
    """``Output_GW``: what ``gw_write`` needs, and the cases refused by name.

    PARSEC calls ``gw_write`` after the SCF or the NSCF run (``parsec.F90``
    l.2095-2147) with the full space group (``symm``; the identity with
    ``Ignore_Symmetry``).  With ``nscf_kpoints`` the WFN header carries
    ``nscf_kgrid``/``nscf_kgrid_shift``; after an SCF PARSEC writes the
    never-set ``nscf%kgrid`` (``0 0 0``), which BerkeleyGW's reader refuses:
    this port writes the Monkhorst-Pack grid of the SCF there (shift in
    BerkeleyGW's convention ``k = (i + s)/n``), and ``0 0 0`` for a
    ``Kpoint_List`` (with a warning).
    """

    from ..Grid.space_group import find_space_group, identity_space_group

    if not is_periodic:
        raise UnsupportedParsecOptionError(
            "Output_GW is implemented for a bulk periodic cell only: gw_write needs the "
            "reciprocal space of the cell (PARSEC's cluster branch reads periodic data "
            "that a cluster run never sets)"
        )
    if wire_or_slab:
        raise UnsupportedParsecOptionError(
            "Output_GW with Boundary_Conditions wire or slab is not implemented: "
            "gw_write maps the grid onto a three-dimensional FFT box, which the confined "
            "grid of a wire or slab does not fill"
        )
    if spinors:
        raise UnsupportedParsecOptionError(
            "Output_GW with spin-orbit coupling or non-collinear magnetism is not "
            "implemented: gw_write writes no two-component spinors"
        )
    if bands:
        raise UnsupportedParsecOptionError(
            "Output_GW together with bandstruc is not implemented: PARSEC's calc_bands "
            "replaces the k-points and levels before gw_write writes them"
        )
    if moves or forces:
        raise UnsupportedParsecOptionError(
            "Output_GW with Minimization, Molecular_Dynamics or forces (Skip_Force: "
            "false) is not implemented: run it as a single point of the final structure"
        )
    nscf = post_processing.nscf
    if nscf is None and monkhorst_pack_dimensions is None and not explicit_kpoints:
        raise UnsupportedParsecOptionError(
            "Output_GW after a Gamma-point SCF is not implemented: PARSEC writes no "
            "k-point (nk = 0), which BerkeleyGW cannot read; use Kpoint_Method mp with "
            "a 1 1 1 grid, or a non-self-consistent run (nscf_kpoints)"
        )
    if space_group is None:
        if ignore_symmetry:
            space_group = identity_space_group()
        else:
            try:
                space_group = find_space_group(periodic_cell.lattice_vectors, atoms)
            except ValueError as error:
                raise ParsecInputError(f"symmetry operations: {error}") from error
    if nscf is not None:
        return GWSettings(
            space_group=space_group,
            kgrid=post_processing.gw_kgrid,
            kshift=post_processing.gw_kshift,
            fermi_level=post_processing.gw_fermi_level,
        )
    if monkhorst_pack_dimensions is not None:
        dimensions = tuple(int(value) for value in monkhorst_pack_dimensions)
        if monkhorst_pack_shift is not None:
            shift = tuple(float(value) for value in monkhorst_pack_shift)
        else:
            # This port's grid (2i - n - 1)/(2n) is k = (i + s)/n with s = 1/2
            # for even n and s = 0 for odd n, modulo a reciprocal lattice vector.
            shift = tuple(0.5 if value % 2 == 0 else 0.0 for value in dimensions)
        warnings.append(
            "Output_GW after an SCF: the WFN header gets the Monkhorst-Pack grid "
            f"{dimensions} and shift {shift} of the SCF (PARSEC writes 0 0 0, which "
            "BerkeleyGW's reader refuses)"
        )
        return GWSettings(space_group=space_group, kgrid=dimensions, kshift=shift)
    warnings.append(
        "Output_GW after an SCF on a Kpoint_List: the WFN header gets the k-grid 0 0 0 "
        "as in PARSEC; BerkeleyGW's reader refuses it unless told not to check the grid"
    )
    return GWSettings(space_group=space_group)


def _point_charges(
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    enabled: bool,
    is_periodic: bool,
    coordinate_factor: float,
    warnings: list[str],
) -> tuple[tuple[PointCharge, ...], tuple[int, ...]]:
    """The point charges of ``usrinputfile.F90`` (lines 2143-2220).

    ``Add_Point_Charges`` (false), ``Point_Typ_Num`` types, and for the n-th
    type the n-th ``Pt_Chg`` (in e, zero where it is missing) and the n-th
    ``Point_Coord`` block (``x y z`` rows in the unit of ``Coordinate_Unit``).
    Returns the charges and the number of charges of each type.
    """

    given = [key for key in ("point_typ_num", "pt_chg") if key in scalar]
    if "point_coord" in blocks:
        given.append("begin Point_Coord")
    if not enabled:
        if given:
            warnings.append(
                f"{', '.join(given)} without Add_Point_Charges: .true. are ignored, "
                "as in PARSEC"
            )
        return (), ()
    if is_periodic:
        # usrinputfile.F90, error 145.
        raise UnsupportedParsecOptionError(
            "point charges (Add_Point_Charges) are only available in "
            "non-periodic systems"
        )
    occurrences = scalar.get("point_typ_num", [])
    types = (
        0
        if not occurrences
        else _integer(str(occurrences[0].value), label="Point_Typ_Num")
    )
    if types <= 0:
        # usrinputfile.F90, error 146 (0 is ESDF's default).
        raise ParsecInputError(
            "unknown number of point charge types: Point_Typ_Num is missing"
        )
    charge_items = scalar.get("pt_chg", [])
    coordinate_blocks = blocks.get("point_coord", [])
    if len(coordinate_blocks) < types:
        # usrinputfile.F90, error 148.
        raise ParsecInputError(
            f"no point charge coordinates were found for point charge type "
            f"{len(coordinate_blocks) + 1}: give one Point_Coord block per type"
        )
    for label, count in (("Pt_Chg", len(charge_items)), ("Point_Coord", len(coordinate_blocks))):
        if count > types:
            warnings.append(
                f"{count} {label} entries for Point_Typ_Num {types}: the "
                f"last {count - types} are not read, as in PARSEC"
            )
    charges: list[PointCharge] = []
    counts: list[int] = []
    for index in range(types):
        charge = (
            _charge_electrons(str(charge_items[index].value), label="Pt_Chg")
            if index < len(charge_items)
            else 0.0
        )
        item = coordinate_blocks[index]
        if not item.value:
            raise ParsecInputError(
                f"line {item.line}: the Point_Coord block of type {index + 1} is empty"
            )
        for row in item.value:
            values, unit = _block_numbers((row,), label="Point_Coord")
            if unit or len(values) < 3:
                raise ParsecInputError(
                    f"line {item.line}: every Point_Coord row must contain three "
                    "coordinates"
                )
            charges.append(
                PointCharge(charge, np.asarray(values[:3], dtype=float) * coordinate_factor)
            )
        counts.append(len(item.value))
    return tuple(charges), tuple(counts)


def _check_hubbard_and_dispersion(
    specifications,
    *,
    van_der_waals: bool,
    is_periodic: bool,
    spin_polarized: bool,
    kpoints: bool,
    spin_orbit: bool,
    bands: bool,
    energy_refinement: bool,
    forces: bool,
) -> None:
    """Refuse LDA+U and TS combinations the drivers do not implement.

    LDA+U runs in the scalar loops (isolated, isolated spin, periodic Gamma,
    periodic spin), the scalar and collinear-spin k-point loops (with the
    Bloch projectors this port uses in place of PARSEC's defective k-point
    form, :mod:`parsec_python.SCF.kpoint_hubbard`) and the band structures
    of their converged Hamiltonians; the TS correction after every loop (the
    spin-orbit and non-collinear ones included), and ``Energy_Refinement``
    (isolated systems) refines the energy that holds them.  LDA+U with
    spin-orbit coupling or non-collinear magnetism is refused (PARSEC's path
    is not a spinor LDA+U, see below).  Forces with ``ldaplusu_j`` are
    refused: PARSEC's J term is not the derivative of its energy, and its
    LDA+U forces are zero; forces with LDA+U and k-points likewise (the
    Bloch projector gradients are not implemented).
    """

    hubbard = any(specification.has_hubbard for specification in specifications.values())
    for active, name in ((hubbard, "LDA+U (ldaplusu_u/ldaplusu_j)"), (van_der_waals, "Van_Der_Waals")):
        if not active:
            continue
        refused = [
            label
            for condition, label in (
                # PARSEC's spin-orbit path builds N from the up components of
                # the spinors alone (calcden.f90z) and applies the matrix of one
                # channel to both components (zmatvec_so -> plusu): there is
                # no spinor LDA+U to port.
                (
                    spin_orbit and name.startswith("LDA"),
                    "spin-orbit coupling or non-collinear magnetism",
                ),
            )
            if condition
        ]
        if refused:
            raise UnsupportedParsecOptionError(
                f"{name} is not implemented together with {', '.join(refused)}"
            )
    if forces and any(
        specification.hubbard_j != 0.0 for specification in specifications.values()
    ):
        raise UnsupportedParsecOptionError(
            "forces, relaxation and molecular dynamics with ldaplusu_j are not "
            "implemented: PARSEC's J term is not the derivative of its LDA+U "
            "energy (and PARSEC's LDA+U forces are zero); give ldaplusu_j 0"
        )
    if forces and hubbard and kpoints:
        raise UnsupportedParsecOptionError(
            "forces, relaxation and molecular dynamics with LDA+U and k-point "
            "sampling are not implemented (PARSEC's LDA+U forces are zero)"
        )




def _default_routines_periodic_refusal(
    periodic_cell,
    grid,
    atoms,
    *,
    slab: bool,
    kpoints,
    kpoint_weights,
    monkhorst_pack_dimensions,
    spin_polarized: bool,
    ionic: bool,
    functional: str = "ca",
) -> str | None:
    """What Andrew Grieder's routines refuse of a periodic input his parser takes, as a reason, or ``None``.

    Unequal k-point weights; a slab with forces, relaxation or dynamics (his slab has energies only); collinear spin
    with k-points other than Gamma and forces (his CLI); a grid his rule refuses (an axis of ``n <= 2 M`` points, ``M``
    the stencil half width, with Fortran's ``adjust_step`` for a bulk cell; fewer than ``2 M`` in the plane of a slab
    with ``2 nint(L/2h)``; a side of 1000 steps or more).
    """

    from ..Grid.pbc import fortran_points_per_axis

    if kpoint_weights is not None and not np.allclose(kpoint_weights, kpoint_weights[0], rtol=1.0e-12, atol=0.0):
        return "a Kpoint_List with unequal weights"
    if slab and ionic:
        return "forces, relaxation or molecular dynamics of a slab"
    gamma_only = (
        (kpoints is None and monkhorst_pack_dimensions is None)
        or (monkhorst_pack_dimensions is not None and tuple(monkhorst_pack_dimensions) == (1, 1, 1))
        or (kpoints is not None and len(kpoints) == 1 and not np.any(kpoints))
    )
    if spin_polarized and ionic and not gamma_only:
        return "forces, relaxation or molecular dynamics of a spin-polarized cell with k-points"
    if spin_polarized and ionic and functional == "pbe":
        # his spin loop of a cell evaluates pbe_spin, which takes no periodic grid there (it stops)
        return "forces, relaxation or molecular dynamics of a spin-polarized PBE cell"
    half_width = grid.stencil_half_width
    lattice = np.asarray(periodic_cell.lattice_vectors, dtype=float)
    if slab:
        in_plane = (2 * np.round(np.linalg.norm(lattice[:2], axis=1) / (2.0 * grid.spacing))).astype(np.int64)
        if np.any(in_plane < 2 * half_width):
            return f"a slab grid of {in_plane.tolist()} in-plane points (fewer than {2 * half_width})"
        return None
    try:
        counts = fortran_points_per_axis(np.linalg.norm(lattice, axis=1), grid.spacing)
    except ValueError as error:
        return f"a grid the default rule cannot build ({error})"
    if np.any(counts <= 2 * half_width):
        return f"a grid of {counts.tolist()} points (the default rule needs more than {2 * half_width} per axis)"
    return None


def _gamma_only_sampling(monkhorst_pack_dimensions, explicit_kpoint_array) -> bool:
    """Whether a k-point input samples the Gamma point alone (``Monkhorst_Pack_Grid 1 1 1``, a list of k = 0)."""
    if monkhorst_pack_dimensions is not None:
        return explicit_kpoint_array is None and tuple(int(n) for n in monkhorst_pack_dimensions) == (1, 1, 1)
    if explicit_kpoint_array is None:
        return False
    points = np.asarray(explicit_kpoint_array, dtype=float)
    return points.shape[0] == 1 and not np.any(points)


def _routines_choice_matters(scalar, one, optional_bool) -> bool:
    """Whether the two sets of routines (parsec_python.Input.routines) can differ for this input.

    A periodic cell, or the forces, relaxation or dynamics of spinor orbitals (``SO_from_scratch``/``SCF_SO``,
    ``Non_Collinear_magnetism``).  Every other calculation runs the same code under both.
    """

    boundary = _normalize_label(one("boundary_conditions", "cluster"))
    if boundary not in {"cluster", "0d"}:
        return True
    spinor = any(
        key in scalar and optional_bool(key)
        for key in ("so_from_scratch", "scf_so", "non_collinear_magnetism")
    )
    if not spinor:
        return False
    moves = _normalize_label(one("minimization", "none")) not in {"none", ""}
    dynamics = "molecular_dynamics" in scalar and optional_bool("molecular_dynamics")
    forces = "skip_force" in scalar and not optional_bool("skip_force")
    return bool(moves or dynamics or forces)


#: Values of the labels Andrew Grieder's parser (or his SCF drivers) refuse although the labels are his
#: (qian/pbc_forces and qian/slab_boundary).
_DEFAULT_CORRELATIONS = frozenset({"ca", "pz", "lda", "pb", "pbe"})
_DEFAULT_EIGENSOLVERS = frozenset({"chebff", "chebdav"})
_DEFAULT_FIELD_UNITS = frozenset({"", "ry/bohr/e", "rydberg/bohr/e", "ha/bohr/e", "hartree/bohr/e", "ev/ang/e"})


def _default_routines_value_refusal(scalar, one, optional_bool) -> str | None:
    """The first value of the input that Andrew Grieder's routines refuse, as a reason, or ``None``.

    Only labels his parser reads are looked at (an unknown label is decided by
    :func:`parsec_python.Input.routines.decide_routines`); the geometry, grid and combination conditions are
    checked where the parser has them.
    """

    if "restart_run" in scalar and optional_bool("restart_run"):
        return "Restart_Run: .true."
    if "old_pseudopotential_format" in scalar and optional_bool("old_pseudopotential_format"):
        return "Old_Pseudopotential_Format: .true."
    if "dynamic_diag_tol" in scalar and optional_bool("dynamic_diag_tol"):
        return "Dynamic_Diag_Tol: .true."
    if "old_interpolation_format" in scalar and optional_bool("old_interpolation_format"):
        return "Old_Interpolation_Format: .true."
    if "double_grid_order" in scalar:
        try:
            order = int(float(one("double_grid_order")))
        except ValueError:
            order = 1
        if order > 1:
            return "Double_Grid_Order > 1"
    if "electric_field" in scalar:
        text = one("electric_field")
        value = _first_float(text, label="Electric_Field")
        remainder = _FLOAT_PATTERN.sub("", text, count=1).strip().lower()
        unit = re.sub(r"[\s_-]", "", remainder)
        if value != 0.0 or unit not in _DEFAULT_FIELD_UNITS:
            return "a nonzero Electric_Field (or its unit)"
    if "spin_polarization" in scalar and optional_bool("spin_polarization") and "correlation_type" not in scalar:
        return "Spin_Polarization without Correlation_Type (PARSEC's default PW LSDA)"
    correlation = _normalize_label(one("correlation_type", "ca"))
    if correlation not in _DEFAULT_CORRELATIONS:
        return f"Correlation_Type: {correlation}"
    shape = _normalize_label(one("cluster_domain_shape", "sphere"))
    if shape not in {"sphere", "box"}:
        return f"Cluster_Domain_Shape: {shape}"
    for item in scalar.get("pseudopotential_format", []):
        name = _normalize_label(str(item.value))
        if name not in {"martins_new", "martinsnew", "martins_old"}:
            return f"Pseudopotential_Format: {name}"
    eigensolver = _normalize_label(one("eigensolver", "chebdav"))
    if eigensolver not in _DEFAULT_EIGENSOLVERS:
        return f"Eigensolver: {eigensolver}"
    mixing = _normalize_label(one("mixing_method", "anderson"))
    if mixing != "anderson":
        return f"Mixing_Method: {mixing}"
    return None

def parse_parsec_input(
    path: str | Path,
    *,
    pseudopotential_directory: str | Path | None = None,
) -> ParsecInputTranslation:
    """Translate a supported ``parsec.in`` (isolated or periodic) into ``SinglePointInput``.

    The routines of the run are decided here (:mod:`parsec_python.Input.routines`): an input that the default
    routines do not compute is translated again for the Fortran-derived ones.
    """
    try:
        try:
            return _parse_parsec_input(
                path,
                pseudopotential_directory=pseudopotential_directory,
            )
        except UseFortranRoutines as fallback:
            return _parse_parsec_input(
                path,
                pseudopotential_directory=pseudopotential_directory,
                routines=fallback.choice,
            )
    except ParsecInputError:
        raise
    except ValueError as error:
        raise ParsecInputError(str(error)) from error


def _mixing_summary(mixing: MixingSettings) -> str:
    """The mixer line of :func:`summarize_translation`."""

    if mixing.method == "anderson":
        return (
            f"Mixing: Anderson alpha={mixing.parameter:.6g}, "
            f"memory={mixing.memory}, restart={mixing.restart}, "
            f"safeguard={mixing.safeguard}"
        )
    if mixing.method == "broyden":
        return (
            f"Mixing: generalized Broyden alpha={mixing.parameter:.6g}, "
            f"memory={mixing.memory}, restart={mixing.restart}"
        )
    return (
        f"Mixing: {mixing.method} alpha={mixing.parameter:.6g}, "
        f"memory={mixing.memory}, group_size={mixing.group_size}, "
        f"en_like={mixing.en_like}, restart_factor={mixing.restart_factor:.6g}, "
        f"expand_factor={mixing.expand_factor:.6g}"
        + (
            f", preferred_type={mixing.preferred_type}"
            if mixing.method == "msecant3"
            else ""
        )
    )


def _periodic_potential_conventions(
    scalar, one, optional_bool, grid, scf, *, is_periodic, wire_or_slab, warnings
):
    """``Periodic_Local_Potential``, ``Periodic_Alpha_Term``, ``Zero_Average_Vxc``.

    Python extensions (Fortran has no keywords: it always does the second
    choice) for a 3-D periodic cell:

    * ``Periodic_Local_Potential: real_space`` (default; this port's Ewald
      split) or ``g_space`` (Fortran's ``v_first``/``pot_local``);
    * ``Periodic_Alpha_Term: analytic`` (default, the ``q -> 0`` limit) or
      ``fortran`` (the finite difference of ``fourier_f.f90``), with
      ``g_space`` only;
    * ``Zero_Average_Vxc: .true.``: Fortran's ``potshift``, the average of the
      exchange-correlation potential of each spin channel removed at every
      evaluation (default false).

    A cluster reads none of them (a warning says so); a wire or slab refuses
    the G-space potential (Fortran's ``ionpot_wire``/``ionpot_slab`` build
    theirs otherwise) and ignores ``Zero_Average_Vxc`` as Fortran does
    (``pbc%per == 3`` only, with a warning); non-collinear magnetism refuses
    ``Zero_Average_Vxc``.
    """

    given = [
        key
        for key in ("periodic_local_potential", "periodic_alpha_term", "zero_average_vxc")
        if key in scalar
    ]
    if not given:
        return grid, scf
    if not is_periodic:
        warnings.append(
            "Periodic_Local_Potential, Periodic_Alpha_Term and Zero_Average_Vxc are "
            "read for a periodic cell only; no effect here"
        )
        return grid, scf
    local = _normalize_label(one("periodic_local_potential", "real_space"))
    local = {"realspace": "real_space", "gspace": "g_space"}.get(local, local)
    if local not in ("real_space", "g_space"):
        raise ParsecInputError(
            "Periodic_Local_Potential must be real_space (this port's Ewald split) "
            "or g_space (Fortran's v_first/pot_local)"
        )
    alpha = _normalize_label(one("periodic_alpha_term", "analytic"))
    alpha = {"finite_difference": "finite_difference", "finitedifference": "finite_difference", "fortran": "finite_difference"}.get(alpha, alpha)
    if alpha not in ("analytic", "finite_difference"):
        raise ParsecInputError(
            "Periodic_Alpha_Term must be analytic or fortran (finite_difference)"
        )
    if wire_or_slab and local == "g_space":
        raise UnsupportedParsecOptionError(
            "Periodic_Local_Potential: g_space is implemented for a 3-D periodic "
            "cell; a wire or slab has its own local potential (Fortran "
            "ionpot_wire/ionpot_slab)"
        )
    if alpha != "analytic" and local != "g_space":
        raise UnsupportedParsecOptionError(
            "Periodic_Alpha_Term belongs to Periodic_Local_Potential: g_space (the "
            "real-space Ewald split carries the average of the potential itself)"
        )
    if local != "real_space" or alpha != "analytic":
        grid = replace(grid, local_potential=local, alpha_term=alpha)
    if optional_bool("zero_average_vxc"):
        if wire_or_slab:
            warnings.append(
                "Zero_Average_Vxc: PARSEC's potshift is applied to 3-D periodic cells "
                "only (rho_hart.F90, pbc%per == 3); no effect for a wire or slab"
            )
        elif scf.noncollinear:
            raise UnsupportedParsecOptionError(
                "Zero_Average_Vxc (PARSEC's potshift) is not implemented for "
                "non-collinear magnetism"
            )
        else:
            scf = replace(scf, zero_average_xc_potential=True)
            if scf.spin_polarized:
                warnings.append(
                    "Zero_Average_Vxc with spin polarization: as PARSEC's potshift, each "
                    "channel loses its own average, which also removes the average "
                    "exchange splitting between the channels (one Fermi level for both)"
                )
    return grid, scf


def summarize_translation(translation: ParsecInputTranslation) -> str:
    """Return a concise, deterministic input summary for CLI/log output."""
    problem = translation.problem
    if problem.eigensolver.method == "chebff":
        first_solver_summary = (
            f"first_filter={problem.eigensolver.first_filter_degree}"
            f"x{problem.eigensolver.first_filter_cycles}"
        )
    elif problem.eigensolver.method == "chebdav":
        first_solver_summary = (
            f"chebdav_degree={problem.eigensolver.first_filter_degree}"
        )
    elif problem.eigensolver.method == "none":
        first_solver_summary = "TEST mode (setup only)"
    else:
        first_solver_summary = (
            f"first_solver={problem.eigensolver.method}, "
            "chebyshev_filtering="
            f"{problem.eigensolver.filters_later_iterations}, "
            f"maximum_matvec={problem.eigensolver.maximum_matvec}"
        )
    if problem.eigensolver.dynamic_tolerance:
        first_solver_summary += ", dynamic_diag_tol=True"
    if problem.periodic_cell is None:
        grid_summary = (
            f"Grid: {problem.grid.domain_shape}, h={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        )
    elif getattr(problem.grid, "periodicity", 3) in (1, 2):
        # A wire (cylinder) or slab: the periodic lattice vectors and radius.
        vectors = problem.periodic_cell.lattice_vectors[: problem.grid.periodicity]
        grid_summary = (
            f"Grid: {'wire' if problem.grid.periodicity == 1 else 'slab'}, "
            "lattice vectors="
            + str([[float(f"{value:.10g}") for value in row] for row in vectors])
            + f" bohr, Boundary_Sphere_Radius={problem.grid.confinement_radius:.10g} bohr, "
            f"h_target={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        )
        if problem.charged_sheets:
            grid_summary += f", charged sheets={len(problem.charged_sheets)}"
    elif getattr(problem.grid, "fortran_routines", False) and uses_general_periodic_grid(
        problem.periodic_cell, problem.grid
    ):
        # Fortran's grid of a general cell: the lattice vectors (rows).
        vectors = problem.periodic_cell.lattice_vectors
        grid_summary = (
            "Grid: periodic (general lattice), lattice vectors="
            + str([[float(f"{value:.10g}") for value in row] for row in vectors])
            + f" bohr, h_target={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        )
    else:
        vectors = problem.periodic_cell.lattice_vectors
        side_lengths = (
            np.diag(vectors)
            if _lattice_is_diagonal(vectors)
            else np.linalg.norm(vectors, axis=1)
        )
        grid_summary = (
            f"Grid: periodic, cell={list(side_lengths)} bohr, "
            f"h_target={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}"
        )
    if problem.grid.double_grid_order > 1:
        grid_summary += f", double_grid_order={problem.grid.double_grid_order}"
    if any(item.old_interpolation for item in problem.pseudopotentials.values()):
        grid_summary += ", old (direct) interpolation of the local pseudopotential"
    lines = [
        f"Input: {translation.source}",
        f"Atoms: {len(problem.atoms)}",
        f"Species: {', '.join(problem.pseudopotentials)}",
        grid_summary,
        #AG (
        #AG     f"Grid: {problem.grid.domain_shape}, h={problem.grid.spacing:.12g} bohr, "
        #AG     f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        #AG ),
        (
            f"SCF: states={problem.scf.number_of_states}, "
            f"max_iter={problem.scf.max_iterations}, "
            f"vconv={problem.scf.convergence_criterion:.6g} Ry, "
            + (
                f"T={problem.scf.fermi_temperature_kelvin:.6g} K"
                if problem.scf.fixed_occupations is None
                else "occupations fixed by occup.in"
            )
        ),
        (
            f"Eigensolver: {problem.eigensolver.method}, "
            f"tol={problem.eigensolver.tolerance:.3g}, "
            f"{first_solver_summary}, "
            f"block={problem.eigensolver.matvec_block_size}, "
            f"later_filter={problem.eigensolver.filter_degree}"
            f"+/-{problem.eigensolver.filter_degree_delta}, "
            f"buffer={problem.eigensolver.subspace_buffer}"
        ),
        _mixing_summary(problem.mixing),
        (
            "Initial density: "
            f"{problem.initial_density_settings.method}, "
            f"normalize={problem.scf.normalize_initial_density}"
        ),
        *(
            ()
            if translation.mlff is None
            else (
                f"Initial structure: relaxed with the force field {translation.mlff.model}"
                + (
                    f" (task {translation.mlff.task})"
                    if translation.mlff.model.startswith(("uma", "esen", "allscaip"))
                    else ""
                )
                + f", fmax={translation.mlff.fmax:g} eV/ang, steps={translation.mlff.steps}, "
                f"{translation.mlff.optimizer}, device={translation.mlff.device}",
            )
        ),
        # A periodic cell has no Hartree boundary values to control.
        *(
            ()
            if problem.periodic_cell is not None
            else (
                f"Hartree boundary: Solver_Lpole={problem.hartree.multipole_order}, "
                "tolerance="
                + (
                    "off"
                    if problem.hartree.boundary_tolerance is None
                    else f"{problem.hartree.boundary_tolerance:.6g} Ry"
                )
                + f", atomic_tail={problem.hartree.atomic_tail}"
                + (
                    " (direct Coulomb sum)"
                    if problem.hartree.boundary_method == "direct"
                    else ""
                ),
            )
        ),
        (
            f"Output: level={translation.output_level}, "
            f"all_states={translation.output_all_states}"
        ),
        # Only for a spin-polarized run: other summaries keep their text.
        *(
            (
                "Spin: polarized, initial polarization "
                + ", ".join(
                    f"{symbol} {potential.initial_spin_polarization:g}"
                    for symbol, potential in problem.pseudopotentials.items()
                ),
            )
            if problem.scf.spin_polarized
            else ()
        ),
        # Only where they are asked for: other summaries keep their text.
        *(
            ()
            if translation.polarizability_field is None
            else (
                "Polarizability: seven-field approach, Electric_Field="
                f"{translation.polarizability_field:.6g} ry/bohr/e",
            )
        ),
        *(
            ()
            if not problem.point_charges
            else (
                f"Point charges: {len(problem.point_charges)}, total "
                f"{sum(item.charge for item in problem.point_charges):.6g} e",
            )
        ),
        *(
            (
                "Charge density: symmetrized over the point group"
                if problem.periodic_cell is None
                else "Charge density: symmetrized over the space group",
            )
            if problem.scf.symmetrize_density
            else ()
        ),
        # Only for Fortran's k-point path (a Monkhorst_Pack_Shift block).
        *(
            ()
            if getattr(problem, "monkhorst_pack_shift", None) is None
            else (
                "K-points: Monkhorst-Pack "
                + "x".join(str(value) for value in problem.monkhorst_pack_dimensions)
                + ", shift "
                + " ".join(f"{value:g}" for value in problem.monkhorst_pack_shift)
                + " (PARSEC's grid, reduced to the irreducible wedge by "
                + (
                    "the identity"
                    if problem.space_group is None or problem.space_group.order == 1
                    else f"{problem.space_group.order} space-group operations"
                )
                + (" without time reversal)" if problem.scf.noncollinear else " and time reversal)"),
            )
        ),
        # Only where the refinement is requested: other summaries keep their text.
        *(
            ()
            if translation.energy_refinement is None
            else (
                f"Energy refinement: {translation.energy_refinement} "
                "(one post-SCF evaluation on the converged orbitals)",
            )
        ),
        # Lines of the post-processing keywords only where the input has them.
        *(
            (
                f"NSCF: {len(translation.nscf.weights)} k-points, "
                f"nscf_states_num={translation.nscf.number_of_states}, potential of "
                "parsec.dat (no SCF)",
            )
            if getattr(translation, "nscf", None) is not None
            else ()
        ),
        *(
            (
                f"DOS: DOS_Pnum={translation.dos.points}"
                + (
                    ""
                    if translation.dos.monkhorst_pack_grid is None
                    else f", grid={translation.dos.monkhorst_pack_grid}"
                )
                + (
                    f", YLM_DOS_L={translation.dos.ylm_l}"
                    if translation.dos.ylm_l >= 0
                    else ""
                ),
            )
            if getattr(translation, "dos", None) is not None
            else ()
        ),
        *(
            (
                f"Output_GW: BerkeleyGW files WFN, RHO, VXC; k-grid {translation.gw.kgrid}, "
                f"shift {translation.gw.kshift}, {translation.gw.space_group.order} symmetry "
                "operations",
            )
            if getattr(translation, "gw", None) is not None
            else ()
        ),
        *(
            ("Hartree: Calc_Hartree_G (V_H(G) = 8 pi rho(G)/|G|^2 by FFT)",)
            if getattr(problem.hartree, "periodic_g_space", False)
            else ()
        ),
        *(
            (
                "Local potential: Fortran's G-space construction (v_first/pot_local), "
                f"alpha term {problem.grid.alpha_term.replace('_', ' ')}",
            )
            if getattr(problem.grid, "local_potential", "real_space") == "g_space"
            else ()
        ),
        *(
            ("Exchange-correlation potential: average removed every iteration (potshift)",)
            if getattr(problem.scf, "zero_average_xc_potential", False)
            else ()
        ),
    ]
    for symbol, potential in problem.pseudopotentials.items():
        lines.append(
            f"PP {symbol}: {potential.path} "
            f"(local l={potential.local_angular_momentum}, "
            f"Read_VCD={potential.read_valence_density}"
            + (
                ""
                if potential.pseudopotential_format == "martins_new"
                else (
                    f", {potential.pseudopotential_format}: "
                    f"Core_Cutoff_Radius={potential.core_cutoff_radius:.6g} bohr"
                )
                if potential.pseudopotential_format == "martins_wang"
                else (
                    f", {potential.pseudopotential_format}: "
                    f"Potential_Num={potential.number_of_channels}, "
                    f"Electron_Per_Orbital={potential.electrons_per_orbital}, "
                    f"Core_Cutoff_Radius={potential.core_cutoff_radius:.6g} bohr"
                    + (
                        ", Nonlinear_Core_Correction=True"
                        if potential.nonlinear_core_correction
                        else ""
                    )
                )
            )
            + (
                f", Alpha_Filter={potential.fourier_filter_alpha:g}, "
                f"Beta1_Filter={potential.fourier_filter_beta1:g}, Core_Filter="
                + (
                    f"{potential.fourier_filter_core:g}"
                    if potential.fourier_filter_core is not None
                    else f"{potential.fourier_filter_alpha:g}"
                )
                if potential.has_fourier_filter
                or potential.fourier_filter_beta1 != 0.0
                or potential.fourier_filter_core is not None
                else ""
            )
            + ")"
        )
    return "\n".join(lines)


__all__ = [
    "ANGSTROM_TO_BOHR",
    "EV_TO_RYDBERG",
    "ParsecInputError",
    "ParsecInputTranslation",
    "UnsupportedParsecOptionError",
    "parse_parsec_input",
    "summarize_translation",
]
