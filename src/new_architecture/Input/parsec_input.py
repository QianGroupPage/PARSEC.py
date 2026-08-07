"""Focused ESDF/PARSEC input reader for isolated CA-LDA single points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from ..models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    MixingSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)


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


@dataclass(frozen=True)
class _InputItem:
    key: str
    value: str | tuple[str, ...]
    line: int
    is_block: bool = False


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
        "Coordinate_Unit must be Cartesian_Ang or Cartesian_Bohr "
        "for this isolated solver"
    )


def _find_potential(
    symbol: str,
    input_directory: Path,
    pseudopotential_directory: Path | None,
) -> Path:
    filename = f"{symbol}_POTRE.DAT"
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
        "pseudopotential_format",
        "local_component",
        "read_vcd",
        "cubic_spline",
        "so_psp",
    }
    block_keys = {"atom_coord", "domain_shape_parameters"}
    accepted_global = {
        "restart_run",
        "relax_restart",
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
        "solver_lpole",
        "full_hartree",
        "atom_types_num",
        "correlation_type",
        "electric_field",
        "output_all_states",
        "output_level",
        "old_interpolation_format",
        "double_grid_order",
        "minimization",
        "movement_num",
        "dynamic_diag_tol",
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

    if optional_bool("restart_run") or optional_bool("relax_restart"):
        raise UnsupportedParsecOptionError("restart calculations are not supported")
    boundary = _normalize_label(one("boundary_conditions", "cluster"))
    if boundary not in {"cluster", "0d"}:
        raise UnsupportedParsecOptionError(
            f"Boundary_Conditions={boundary!r} is not an isolated cluster"
        )
    if optional_bool("spin_polarization"):
        raise UnsupportedParsecOptionError("spin-polarized calculations are not supported")
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

    correlation = _normalize_label(one("correlation_type", "ca"))
    if correlation not in {"ca", "pz", "lda"}:
        raise UnsupportedParsecOptionError(
            f"Correlation_Type={correlation!r}; only CA/PZ LDA is supported"
        )

    shape = _normalize_label(one("cluster_domain_shape", "sphere"))
    spacing = _physical_length(one("grid_spacing"), label="Grid_Spacing")
    if shape == "sphere":
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
        radius=radius,
        expansion_order=_integer(
            one("expansion_order", "12"), label="Expansion_Order"
        ),
        shift=shift,
        domain_shape=shape,
        box_lengths=box_lengths,
    )

    coordinate_factor = _coordinate_factor(
        one("coordinate_unit", "cartesian_bohr")
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

    local_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    atoms: list[Atom] = []
    specifications: dict[str, SpeciesPotential] = {}
    global_cubic_occurrences = scalar.get("cubic_spline", [])
    for species_index, (symbol, coordinate_item) in enumerate(
        zip(symbols, coordinate_blocks)
    ):
        pseudo_format = _normalize_label(
            species_value("pseudopotential_format", species_index, "martins_new")
        )
        if pseudo_format not in {"martins_new", "martinsnew"}:
            raise UnsupportedParsecOptionError(
                f"Atom_Type {symbol}: only Pseudopotential_Format=martins_new is supported"
            )
        local_label = _normalize_label(
            species_value("local_component", species_index)
        )
        if local_label not in local_map:
            raise ParsecInputError(
                f"Atom_Type {symbol}: Local_Component must be s, p, d, or f"
            )
        if _boolean(
            species_value("so_psp", species_index, "false"),
            label=f"{symbol} SO_PSP",
        ):
            raise UnsupportedParsecOptionError("spin-orbit pseudopotentials are not supported")
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
        potential_path = _find_potential(
            symbol, source.parent, pp_directory
        )
        specifications[symbol] = SpeciesPotential(
            potential_path,
            local_map[local_label],
            read_valence_density=read_vcd,
            use_spline=use_spline,
        )

        for coordinate_line in coordinate_item.value:
            values, unit = _block_numbers((coordinate_line,), label=f"{symbol} Atom_Coord")
            if unit or len(values) != 3:
                raise ParsecInputError(
                    f"line {coordinate_item.line}: every Atom_Coord row must "
                    "contain exactly three numbers"
                )
            atoms.append(
                Atom(symbol, np.asarray(values, dtype=float) * coordinate_factor)
            )
        if not any(atom.symbol == symbol for atom in atoms):
            raise ParsecInputError(f"Atom_Type {symbol} has an empty Atom_Coord block")

    warnings: list[str] = []
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
    )
    hartree = HartreeSettings(
        multipole_order=_integer(one("solver_lpole", "9"), label="Solver_Lpole"),
        boundary_method="direct" if optional_bool("full_hartree") else "auto",
    )
    scf = SCFSettings(
        max_iterations=_integer(one("max_iter", "50"), label="Max_Iter"),
        convergence_criterion=_energy_rydberg(
            one("convergence_criterion", "2e-4 Ry"),
            label="Convergence_Criterion",
        ),
        fermi_temperature_kelvin=_temperature_kelvin(
            one("fermi_temp", "80"), label="Fermi_Temp"
        ),
        number_of_states=states,
        net_charge=_charge_electrons(
            one("net_charges", "0"), label="Net_Charges"
        ),
        use_plain_residual=optional_bool("use_plain_sre"),
    )

    minimization = _normalize_label(one("minimization", "none"))
    if minimization in {"", "none"}:
        recenter = True
    elif minimization == "manual":
        movement_count = _integer(one("movement_num", "0"), label="Movement_Num")
        if movement_count != 0:
            raise UnsupportedParsecOptionError(
                "manual ionic movements are not part of a single-point calculation"
            )
        recenter = False
    else:
        raise UnsupportedParsecOptionError(
            f"Minimization={minimization!r}; relaxation and dynamics are not supported"
        )

    if ignore_symmetry:
        warnings.append(
            "Ignore_Symmetry=true reproduces PARSEC's zero grid shift; Python "
            "already uses the complete grid with no symmetry reduction."
        )
    if shape == "box" and hartree.boundary_method != "direct":
        warnings.append(
            "Box Hartree boundaries use the exact direct Coulomb sum in auto mode "
            "and may be expensive."
        )

    problem = SinglePointInput(
        atoms=atoms,
        pseudopotentials=specifications,
        grid=grid,
        scf=scf,
        hartree=hartree,
        eigensolver=eigensolver,
        mixing=mixing,
        recenter_geometry=recenter,
    )
    return ParsecInputTranslation(
        source=source,
        problem=problem,
        warnings=tuple(warnings),
        output_all_states=optional_bool("output_all_states"),
        output_level=_integer(one("output_level", "1"), label="Output_Level"),
    )


def parse_parsec_input(
    path: str | Path,
    *,
    pseudopotential_directory: str | Path | None = None,
) -> ParsecInputTranslation:
    """Translate a supported isolated ``parsec.in`` into ``SinglePointInput``."""
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
    lines = [
        f"Input: {translation.source}",
        f"Atoms: {len(problem.atoms)}",
        f"Species: {', '.join(problem.pseudopotentials)}",
        (
            f"Grid: {problem.grid.domain_shape}, h={problem.grid.spacing:.12g} bohr, "
            f"order={problem.grid.expansion_order}, shift={problem.grid.shift}"
        ),
        (
            f"SCF: states={problem.scf.number_of_states}, "
            f"max_iter={problem.scf.max_iterations}, "
            f"vconv={problem.scf.convergence_criterion:.6g} Ry, "
            f"T={problem.scf.fermi_temperature_kelvin:.6g} K"
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
            f"memory={problem.mixing.memory}, restart={problem.mixing.restart}"
        ),
        (
            f"Output: level={translation.output_level}, "
            f"all_states={translation.output_all_states}"
        ),
    ]
    for symbol, potential in problem.pseudopotentials.items():
        lines.append(
            f"PP {symbol}: {potential.path} "
            f"(local l={potential.local_angular_momentum}, "
            f"Read_VCD={potential.read_valence_density})"
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
