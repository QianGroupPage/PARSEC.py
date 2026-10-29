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
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    InitialDensitySettings,
    MixingSettings,
    RelaxationSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    PeriodicCell,
    PeriodicGridSettings,
)
from ..Grid.pbc import fortran_grid_shift
from ..Grid.kpoints import explicit_kpoints, read_kpoint_file
from ..MLDensity.field import normalize_density_units
from ..MolecularDynamics.settings import MolecularDynamicsSettings

if TYPE_CHECKING:
    from ..Hartree.domain import DomainChoice
    from ..MLFF.settings import MLFFSettings


# PARSEC's ESDF table uses 1 bohr = 0.529177 angstrom exactly.
ANGSTROM_TO_BOHR = 1.0 / 0.529177
EV_TO_RYDBERG = 1.0 / 13.605693122994

_FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
)
_COMMENT_MARKERS = ("#", ";", "!")


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


def _electric_field_value(value: str, *, label: str) -> float:
    number = _first_float(value, label=label)
    remainder = _FLOAT_PATTERN.sub("", value, count=1).strip().lower()
    unit = re.sub(r"[\s_-]", "", remainder)
    supported = {
        "",
        "ry/bohr/e",
        "rydberg/bohr/e",
        "ha/bohr/e",
        "hartree/bohr/e",
        "ev/ang/e",
    }
    if unit not in supported:
        raise UnsupportedParsecOptionError(
            f"{label}: unsupported electric-field unit {remainder!r}"
        )
    return number


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
) -> PeriodicCell:
    """Parse a ``Cell_Shape`` block into a :class:`PeriodicCell`.

    """

    label = "Cell_Shape"
    cell_blocks = blocks.get("cell_shape", [])
    if len(cell_blocks) != 1:
        raise ParsecInputError("a periodic cell requires exactly one Cell_Shape block")
    lines = cell_blocks[0].value
    if len(lines) == 1:
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
    return PeriodicCell(lattice_vectors=vectors * scale)


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
        "Coordinate_Unit must be Cartesian_Ang, Cartesian_Bohr, or (periodic "
        "cells) lattice_vectors"
    )


def _lattice_is_diagonal(lattice_vectors: np.ndarray) -> bool:
    return bool(np.allclose(lattice_vectors, np.diag(np.diag(lattice_vectors)), atol=1.0e-10))


def _is_lattice_coordinate_unit(value: str) -> bool:
    """``Coordinate_Unit: lattice_vectors``: coordinates in units of the lattice vectors (fractional)."""
    return _normalize_label(value) in {"lattice_vectors", "latticevectors"}


def _martins_old_input(
    symbol: str,
    scalar: dict[str, list["_InputItem"]],
    blocks: dict[str, list["_InputItem"]],
    species_index: int,
    old_index: int,
) -> dict[str, object]:
    """Read what a ``martins_old`` species adds to its file.

    ``usrinputfile.F90`` reads ``Potential_Num`` for every species and
    ``Core_Cutoff_Radius`` and ``Electron_Per_Orbital`` for old-format
    species only, and ESDF hands out the occurrences of a label in the order
    of the calls.  A species therefore takes the ``Potential_Num`` of its
    position among all species (``species_index``) and the other two of its
    position among the old-format ones (``old_index``).

    Where PARSEC would go on with a default (no nonlocal projectors without
    a cutoff radius, no ionic charge without occupations) this reader stops.
    """

    counts = scalar.get("potential_num", [])
    if species_index >= len(counts):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a martins_old pseudopotential needs "
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
    radii = scalar.get("core_cutoff_radius", [])
    if old_index >= len(radii):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a martins_old pseudopotential needs "
            "Core_Cutoff_Radius, the radius of its nonlocal projectors"
        )
    radius = _physical_length(
        str(radii[old_index].value), label=f"{symbol} Core_Cutoff_Radius"
    )
    if not radius > 0.0:
        raise ParsecInputError(
            f"Atom_Type {symbol}: Core_Cutoff_Radius must be positive"
        )
    orbital_blocks = blocks.get("electron_per_orbital", [])
    if old_index >= len(orbital_blocks):
        raise ParsecInputError(
            f"Atom_Type {symbol}: a martins_old pseudopotential needs an "
            "Electron_Per_Orbital block; its sum is the ionic charge"
        )
    lines = orbital_blocks[old_index].value
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
    return {
        "pseudopotential_format": "martins_old",
        "number_of_channels": channels,
        # PARSEC reads Potential_Num values and ignores the rest of the line.
        "electrons_per_orbital": tuple(electrons[:channels]),
        "core_cutoff_radius": radius,
    }


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
    }
    accepted_global = {
        "restart_run",
        "relax_restart",
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
        "mixing_method",
        "mixing_param",
        "memory_param",
        "restart_mixing",
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
        if len(occurrences) > 1 and key not in species_keys:
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

    if optional_bool("restart_run"):
        raise UnsupportedParsecOptionError(
            "Restart_Run (restart of the electronic structure from parsec.dat) "
            "is not supported; Relax_Restart is"
        )
    if optional_bool("old_pseudopotential_format"):
        raise UnsupportedParsecOptionError(
            "Old_Pseudopotential_Format=true is not supported; use PARSEC's "
            "new Martins pseudopotential format"
        )
    boundary = _normalize_label(one("boundary_conditions", "cluster"))
    is_periodic = boundary in {"bulk", "3d", "pbc"}
    if boundary not in {"cluster", "0d", "bulk", "3d", "pbc"}:
        raise UnsupportedParsecOptionError(
            f"Boundary_Conditions={boundary!r}; only cluster (isolated) and "
            "bulk (periodic, any cell shape) are supported "
            "-- wire and slab are not"
        )
    periodic_flag = optional_bool("periodic_system")
    if periodic_flag != is_periodic:
        raise UnsupportedParsecOptionError(
            "Periodic_System and Boundary_Conditions disagree: "
            "Periodic_System=true requires Boundary_Conditions=bulk, and "
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
    if optional_bool("dynamic_diag_tol"):
        raise UnsupportedParsecOptionError("Dynamic_Diag_Tol is not supported")
    if optional_bool("old_interpolation_format"):
        raise UnsupportedParsecOptionError("Old_Interpolation_Format is not supported")
    double_grid_order = _integer(one("double_grid_order", "1"), label="Double_Grid_Order")
    if double_grid_order != 1:
        raise UnsupportedParsecOptionError(
            "Double_Grid_Order > 1 (Ono-Hirose subgrid integration) is not implemented"
        )
    electric_field = _electric_field_value(
        one("electric_field", "0"), label="Electric_Field"
    )
    if electric_field != 0.0:
        raise UnsupportedParsecOptionError("nonzero Electric_Field is not supported")

    if spin_polarized and "correlation_type" not in scalar:
        # usrinputfile.F90: without the line a spin-polarized run takes the
        # Perdew-Wang LSDA ("pl"), an unpolarized one CA.
        raise UnsupportedParsecOptionError(
            "Spin_Polarization without Correlation_Type: PARSEC then uses the "
            "Perdew-Wang LSDA (pl), which is not implemented; give "
            "Correlation_Type: ca or pbe"
        )
    correlation = _normalize_label(one("correlation_type", "ca"))
    if correlation in {"ca", "pz", "lda"}:
        xc_functional = "ca"
    elif correlation in {"pb", "pbe"}:
        xc_functional = "pbe"
    else:
        raise UnsupportedParsecOptionError(
            f"Correlation_Type={correlation!r}; supported choices are CA/PZ LDA and PBE"
        )

    periodic_cell = None
    monkhorst_pack_dimensions: tuple[int, int, int] | None = None
    explicit_kpoint_array: np.ndarray | None = None
    explicit_kpoint_weights: np.ndarray | None = None
    shape: str | None = None
    # Only the sphere of a cluster can leave its radius to the default rule.
    default_radius = False
    if is_periodic:
        spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
        ignore_symmetry = optional_bool("ignore_symmetry")
        periodic_cell = _parse_cell_shape(scalar, blocks)
        grid = PeriodicGridSettings(
            spacing=spacing,
            expansion_order=_integer(
                one("expansion_order", "12"), label="Expansion_Order"
            ),
            # the Fortran grid: half a step off the origin along vectors orthogonal to the others (zero with
            # Ignore_Symmetry), and ``adjust_step``'s even number of points per direction
            shift=fortran_grid_shift(periodic_cell.lattice_vectors, ignore_symmetry),
            points_rule="fortran",
            box_lengths=tuple(
                np.diag(periodic_cell.lattice_vectors)
                if _lattice_is_diagonal(periodic_cell.lattice_vectors)
                else np.linalg.norm(periodic_cell.lattice_vectors, axis=1)
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
            if len(values) != 3 or unit or any(value <= 0 for value in values):
                raise ParsecInputError(
                    "Monkhorst_Pack_Grid must contain exactly three positive integers"
                )
            monkhorst_pack_dimensions = tuple(int(value) for value in values)
            if any(value != original for value, original in zip(monkhorst_pack_dimensions, values)):
                raise ParsecInputError("Monkhorst_Pack_Grid values must be integers")
            if "monkhorst_pack_shift" in blocks:
                raise UnsupportedParsecOptionError(
                    "Monkhorst_Pack_Shift is not supported; only an unshifted "
                    "Monkhorst-Pack grid is implemented"
                )
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
                    cell=periodic_cell,
                )
            except (OSError, ValueError) as error:
                raise ParsecInputError(f"Kpoint_Method=manual: {error}") from error
            if not np.allclose(explicit_kpoint_weights, explicit_kpoint_weights[0]):
                raise UnsupportedParsecOptionError(
                    "Kpoint_Method=manual with unequal sampling weights is not "
                    "supported yet: the SCF drivers pool every k-point into one "
                    "Fermi level and need equal weights (an unreduced uniform "
                    "sample).  Give equal weights, or omit the weight column"
                )
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
    else:
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
        else:
            raise UnsupportedParsecOptionError(
                f"Cluster_Domain_Shape={shape!r}; only sphere and box are supported"
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
        )

    fractional_coordinates = _is_lattice_coordinate_unit(
        one("coordinate_unit", "cartesian_bohr")
    )
    if fractional_coordinates and periodic_cell is None:
        raise UnsupportedParsecOptionError(
            "Coordinate_Unit: lattice_vectors needs a periodic cell"
        )
    coordinate_factor = (
        1.0
        if fractional_coordinates
        else _coordinate_factor(one("coordinate_unit", "cartesian_bohr"))
    )
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
            for key in (*_MOLECULAR_DYNAMICS_KEYS, "minimization")
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
    for species_index, (symbol, coordinate_item) in enumerate(
        zip(symbols, coordinate_blocks)
    ):
        pseudo_format = _normalize_label(
            species_value("pseudopotential_format", species_index, "martins_new")
        )
        if pseudo_format not in {"martins_new", "martinsnew", "martins_old"}:
            raise UnsupportedParsecOptionError(
                f"Atom_Type {symbol}: Pseudopotential_Format must be "
                "martins_new or martins_old; martins_wang and fhipp are not "
                "supported"
            )
        local_label = _normalize_label(
            species_value("local_component", species_index)
        )
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
        if pseudo_format == "martins_old":
            if read_vcd:
                raise ParsecInputError(
                    f"Atom_Type {symbol}: Read_VCD needs a martins_new "
                    "pseudopotential; an old Martins file holds no valence "
                    "charge density"
                )
            format_input = _martins_old_input(
                symbol, scalar, blocks, species_index, old_format_species
            )
            old_format_species += 1
        potential_path = _find_potential(
            symbol,
            source.parent,
            pp_directory,
            "POTR" if format_input else "POTRE",
        )
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
        specifications[symbol] = SpeciesPotential(
            potential_path,
            local_map[local_label],
            read_valence_density=read_vcd,
            use_spline=use_spline,
            element_symbol=element_symbol,
            atomic_energy_correction=correction,
            spin_orbit=spin_orbit,
            initial_spin_polarization=polarization,
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
            movable.append(moves)
            position = np.asarray(values[:3], dtype=float) * coordinate_factor
            if fractional_coordinates:
                # fractional coordinates are folded into the cell, as the Fortran code does
                fraction = position - np.floor(position)
                position = fraction @ periodic_cell.lattice_vectors
            atoms.append(
                Atom(
                    symbol,
                    position,
                    initial_moment=species_moments[row_index],
                )
            )
        if not any(atom.symbol == symbol for atom in atoms):
            raise ParsecInputError(f"Atom_Type {symbol} has an empty Atom_Coord block")
    for key, table in (("core_cutoff_radius", scalar), ("electron_per_orbital", blocks)):
        # PARSEC reads these for martins_old species only, in their order: a
        # line under another species would go to the next old-format one.
        if len(table.get(key, [])) > old_format_species:
            raise ParsecInputError(
                f"{key} occurs {len(table[key])} times, but only "
                f"{old_format_species} Atom_Type(s) have "
                "Pseudopotential_Format: martins_old, the format that reads it"
            )
    if (
        any(item.read_valence_density for item in specifications.values())
        and specifications[symbols[-1]].pseudopotential_format != "martins_new"
    ):
        # pseudo.f90 tests the format of the last Atom_Type, whichever
        # species has Read_VCD, and stops with error 23.
        raise ParsecInputError(
            "Read_VCD is set and the last Atom_Type has a martins_old "
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

    warnings: list[str] = list(coordinate_notes)
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
    ionic_control_keys = {
        "movement_num",
        "ion_energy_diff",
        "force_min",
        "max_step",
        "min_step",
    }
    ionic_controls_present = bool(ionic_control_keys.intersection(scalar))
    if "ion_energy_diff" in scalar:
        _energy_rydberg(one("ion_energy_diff"), label="Ion_Energy_Diff")
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
    if eigensolver_method not in {"chebff", "chebdav", "arpack"}:
        raise UnsupportedParsecOptionError(
            f"Eigensolver={eigensolver_method!r} is not supported"
        )

    first_filter_degree = _integer(
        one("chebdav_degree", "20"), label="Chebdav_Degree"
    )
    if eigensolver_method == "chebdav" and first_filter_degree < 15:
        raise ParsecInputError(
            "Chebdav_Degree must be at least 15 when Eigensolver=chebdav"
        )
    if eigensolver_method != "chebdav" and first_filter_degree < 10:
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
    if (
        eigensolver_method in {"chebff", "chebdav"}
        and subspace_buffer < 6
    ):
        warnings.append(
            f"Subspace_Buffer_Size={subspace_buffer} is below 6 for "
            f"{eigensolver_method} filtering; PARSEC resets it to 6."
        )
        subspace_buffer = 6

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
    )

    mixing_name = _normalize_label(one("mixing_method", "anderson"))
    if mixing_name != "anderson":
        raise UnsupportedParsecOptionError(
            f"Mixing_Method={mixing_name!r}; only Anderson is supported"
        )
    mixing = MixingSettings(
        parameter=_strict_float(one("mixing_param", "0.30"), label="Mixing_Param"),
        memory=_integer(one("memory_param", "4"), label="Memory_Param"),
        restart=_integer(one("restart_mixing", "20"), label="Restart_Mixing"),
        safeguard=optional_bool("mixing_safeguard"),
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
        normalize_initial_density=optional_bool(
            "normalize_initial_density", True
        ),
        xc_functional=xc_functional,
        self_consistent_spin_orbit=self_consistent_spin_orbit
        or (
            noncollinear
            and any(specification.spin_orbit for specification in specifications.values())
        ),
        noncollinear=noncollinear,
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
    # they are; every other calculation recentres the cluster.
    recenter = minimization != "manual"
    if minimization != "none":
        if is_periodic:
            raise UnsupportedParsecOptionError(
                f"Minimization={minimization!r} with Boundary_Conditions=bulk: "
                "structural relaxation is implemented for isolated systems "
                "(Boundary_Conditions: cluster) only"
            )
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
        if is_periodic:
            raise UnsupportedParsecOptionError(
                "Molecular_Dynamics with Boundary_Conditions=bulk: molecular "
                "dynamics is implemented for isolated systems "
                "(Boundary_Conditions: cluster) only"
            )
        if not any(movable):
            raise ParsecInputError(
                "Molecular_Dynamics is requested, but Move_Flag leaves no atom movable"
            )
        # parsec.F90 uses no symmetry in molecular dynamics.
        use_symmetry = False
    # PARSEC always evaluates the forces.  Here a single point evaluates
    # them only on request, so that inputs without the line keep their cost.
    compute_forces = atoms_move or ("skip_force" in scalar and not skip_force)
    if not molecular_dynamics.enabled and any(
        key in scalar for key in _MOLECULAR_DYNAMICS_KEYS if key != "molecular_dynamics"
    ):
        warnings.append(
            "Molecular dynamics controls are present without "
            "Molecular_Dynamics: true and are ignored."
        )
    if compute_forces and is_periodic:
        compute_forces = False
        warnings.append(
            "Skip_force=false requests forces, which are implemented for "
            "isolated systems only; this periodic run reports energies only."
        )

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
    )


def parse_parsec_input(
    path: str | Path,
    *,
    pseudopotential_directory: str | Path | None = None,
) -> ParsecInputTranslation:
    """Translate a supported ``parsec.in`` (isolated or periodic) into ``SinglePointInput``."""
    try:
        return _parse_parsec_input(
            path,
            pseudopotential_directory=pseudopotential_directory,
        )
    except ParsecInputError:
        raise
    except ValueError as error:
        raise ParsecInputError(str(error)) from error


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
    else:
        first_solver_summary = "first_solver=arpack"
    if problem.periodic_cell is None:
        grid_summary = (
            f"Grid: {problem.grid.domain_shape}, h={problem.grid.spacing:.12g} bohr, "
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
        (
            f"Mixing: Anderson alpha={problem.mixing.parameter:.6g}, "
            f"memory={problem.mixing.memory}, restart={problem.mixing.restart}, "
            f"safeguard={problem.mixing.safeguard}"
        ),
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
        # Only where the refinement is requested: other summaries keep their text.
        *(
            ()
            if translation.energy_refinement is None
            else (
                f"Energy refinement: {translation.energy_refinement} "
                "(one post-SCF evaluation on the converged orbitals)",
            )
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
                    f"Potential_Num={potential.number_of_channels}, "
                    f"Electron_Per_Orbital={potential.electrons_per_orbital}, "
                    f"Core_Cutoff_Radius={potential.core_cutoff_radius:.6g} bohr"
                )
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
