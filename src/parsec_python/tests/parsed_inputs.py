"""What the parser makes of an input, in a form that can be stored and compared.

The tests of the default domain keep, in
``data/parsed_inputs_given_radius.json``, the records this module made of
every input with a sphere radius under the last tree that knew no default
rule.  A record
holds the settings of the parsed problem as they print, a digest of the
atoms, and a digest of everything the reference cache key of the accelerated
driver hashes.  Directories are replaced by tokens and line ends by ``\\n``,
so a record does not depend on where the tree lies, on its system, or on how
its text files were checked out.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from parsec_python.Input import parse_parsec_input


REPOSITORY = Path(__file__).resolve().parents[3]
SETTINGS = (
    "grid",
    "scf",
    "hartree",
    "eigensolver",
    "mixing",
    "initial_density_settings",
)
_KEY_SWITCHES = ("PARSEC_IONIC_BACKEND", "PARSEC_NATIVE_PROJECTOR_LOOKUP")
# C795H300, the smallest of the benchmark clusters (3,480 electrons),
# with the sphere of 5 angstrom of vacuum of its benchmark input.  Its
# pseudopotentials are those of examples/0d_benzene.
NANODIAMOND = Path(__file__).parent / "data" / "C795H300_parsec.in"
BASE_PSEUDOPOTENTIALS = REPOSITORY / "examples" / "0d_benzene"


RECORDS = Path(__file__).parent / "data" / "parsed_inputs_given_radius.json"
# The periodic inputs of the tree (``Boundary_Conditions: bulk``).  They have
# no sphere and no stored record.
PERIODIC_INPUTS = ("examples/3d_Si/parsec.in",)


def shipped_inputs() -> dict[str, tuple[Path, Path | None]]:
    """The cluster inputs of the tree: name, path and pseudopotential directory.

    They are the inputs the stored records hold.  Any other input found
    under ``examples`` is left out: a periodic one, which has no sphere
    (:func:`periodic_inputs` holds those), and the inputs of a folder that
    is not part of the tree, such as the working files of a checkout, which
    have no record and need not parse.
    """

    return _inputs_named(json.loads(RECORDS.read_text(encoding="utf-8"))["inputs"])


def periodic_inputs() -> dict[str, tuple[Path, Path | None]]:
    """The periodic inputs of the tree: name, path and pseudopotential directory."""

    return _inputs_named(PERIODIC_INPUTS)


def _inputs_named(names) -> dict[str, tuple[Path, Path | None]]:
    """The named inputs in the order they are found; all of them must exist."""

    found = _every_input()
    missing = sorted(set(names) - set(found))
    if missing:
        raise AssertionError(f"inputs that are not in the tree: {missing}")
    return {name: source for name, source in found.items() if name in names}


def _every_input() -> dict[str, tuple[Path, Path | None]]:
    """Every PARSEC input found in the tree: name, path and pseudopotential directory."""

    inputs: dict[str, tuple[Path, Path | None]] = {}
    examples = REPOSITORY / "examples"
    shared = examples / "ml_initial_density" / "pseudopotentials"
    for path in sorted(examples.rglob("parsec.in")):
        inputs[path.relative_to(REPOSITORY).as_posix()] = (
            path,
            shared if shared.parent in path.parents else None,
        )
    for path in sorted((Path(__file__).parent / "data").glob("*.in")):
        inputs[path.relative_to(REPOSITORY).as_posix()] = (
            path,
            BASE_PSEUDOPOTENTIALS if path == NANODIAMOND else None,
        )
    return inputs


def portable(text: str, roots: dict[str, Path]) -> str:
    """Replace the directories of ``roots`` in a text by their tokens."""

    found = False
    for token, root in roots.items():
        for form in (str(root), root.as_posix()):
            if form in text:
                text = text.replace(form, token)
                found = True
    if not found:
        return text
    return re.sub(r"(Windows|Posix)Path\(", "Path(", text).replace("\\\\", "/").replace(
        "\\", "/"
    )


class _Recorder:
    """Stands in for the hash of the driver and keeps what is hashed."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def update(self, data) -> None:
        self.chunks.append(bytes(data))

    def hexdigest(self) -> str:
        return sha256(b"".join(self.chunks)).hexdigest()


def reference_cache_key(problem, roots: dict[str, Path]) -> tuple[str, str]:
    """Return the reference cache key of the driver, as it is and portable.

    The portable one hashes the same bytes with the directories of ``roots``
    replaced and the line ends of the files normalized.
    """

    from parsec_python.acceleration import driver

    recorder = _Recorder()
    environment = {
        name: value for name, value in os.environ.items() if name not in _KEY_SWITCHES
    }
    with (
        patch.object(driver, "sha256", lambda: recorder),
        patch.dict(os.environ, environment, clear=True),
    ):
        key = driver._reference_cache_key(
            problem,
            SimpleNamespace(finite_difference_builder="native"),
            defer_native_laplacian=False,
            cache_directory=None,
        )
    hashed = list(recorder.chunks)
    if problem.periodic_cell is None:
        # The tree of the stored records knew no periodic cell.  The driver now hashes
        # it behind the settings and the recentring, for a cluster as the
        # text "None": that text is left out, so that the digest of a cluster
        # stays the one that was stored from that tree.
        place = hashed.index(repr(problem.initial_density_settings).encode("utf-8")) + 2
        if hashed[place] != b"None":
            raise AssertionError(
                "the driver does not hash the periodic cell behind the settings"
            )
        del hashed[place]
    chunks = []
    for chunk in hashed:
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            chunks.append(chunk)
        else:
            chunks.append(portable(text, roots).encode("utf-8"))
    return key, sha256(b"".join(chunks).replace(b"\r\n", b"\n")).hexdigest()


def problem_record(translation, roots: dict[str, Path]) -> dict:
    """Describe a parsed input: settings, atoms, species and the cache key."""

    problem = translation.problem
    record: dict[str, object] = {
        name: portable(repr(getattr(problem, name)), roots) for name in SETTINGS
    }
    record["recenter_geometry"] = bool(problem.recenter_geometry)
    atoms = sha256()
    for atom in problem.atoms:
        atoms.update(atom.symbol.encode("utf-8"))
        atoms.update(np.ascontiguousarray(atom.position, dtype=np.float64).tobytes())
    record["atoms"] = [len(problem.atoms), atoms.hexdigest()]
    record["pseudopotentials"] = {
        symbol: [
            portable(str(specification.path), roots),
            specification.local_angular_momentum,
            specification.read_valence_density,
            specification.use_spline,
            specification.element_symbol,
            repr(specification.atomic_energy_correction),
        ]
        for symbol, specification in problem.pseudopotentials.items()
    }
    record["translation"] = [
        [portable(warning, roots) for warning in translation.warnings],
        translation.output_all_states,
        translation.output_level,
        translation.ignore_symmetry,
    ]
    record["reference_cache_key"] = reference_cache_key(problem, roots)[1]
    return record


def input_record(
    path: Path, pseudopotential_directory: Path | None, roots: dict[str, Path]
) -> dict:
    """Parse an input and describe the problem, or the error it stops with."""

    try:
        translation = parse_parsec_input(
            path, pseudopotential_directory=pseudopotential_directory
        )
    except Exception as error:
        return {"error": [type(error).__name__, portable(str(error), roots)]}
    return problem_record(translation, roots)


# Benzene with every keyword the mistakes below touch.  Its pseudopotentials
# are those of examples/0d_benzene.
BASE_INPUT = """\
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 6.0 ang
Grid_Spacing: 0.30 ang
Expansion_Order: 8
Coordinate_Unit: Cartesian_Bohr
Atom_Types_Num: 2
Atom_Type: C
Local_Component: p
begin Atom_Coord
  2.60782206   0.00000000   0.00000
  1.30391103   2.25844016   0.00000
 -1.30391103   2.25844016   0.00000
 -2.60782206   0.00000000   0.00000
 -1.30391103  -2.25844016   0.00000
  1.30391103  -2.25844016   0.00000
end Atom_Coord
Atom_Type: H
Local_Component: s
begin Atom_Coord
  4.66762355   0.00000000   0.00000
  2.33381177   4.04228057   0.00000
 -2.33381177   4.04228057   0.00000
 -4.66762355   0.00000000   0.00000
 -2.33381177  -4.04228057   0.00000
  2.33381177  -4.04228057   0.00000
end Atom_Coord
Correlation_Type: ca
States_Num: 21
Net_Charges: 0 e
Fermi_Temp: 500 K
Max_Iter: 40
Convergence_Criterion: 1e-4 Ry
Eigensolver: chebff
Mixing_Method: Anderson
Solver_Lpole: 9
Output_Level: 1
"""


def _replace(pattern: str, replacement: str):
    def edit(text: str) -> str:
        if len(re.findall(pattern, text, flags=re.MULTILINE)) != 1:
            raise AssertionError(f"{pattern!r} does not match one place")
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)

    return edit


def _append(*lines: str):
    return lambda text: text + "\n" + "\n".join(lines) + "\n"


# Mistakes in an input that has a sphere radius.  Several hold two of them:
# the error that is reported shows the order in which the parser reads.
MALFORMED = {
    "no grid spacing": (_replace(r"^Grid_Spacing.*\n", ""),),
    "radius with a unit that is none": (
        _replace(r"^(Boundary_Sphere_Radius:).*$", r"\1 9 furlong"),
    ),
    "radius twice": (
        _replace(r"^(Boundary_Sphere_Radius:.*)$", r"\1\n\1"),
    ),
    "radius of zero": (_replace(r"^(Boundary_Sphere_Radius:).*$", r"\1 0 ang"),),
    "radius without a value": (
        _replace(r"^Boundary_Sphere_Radius:.*$", "Boundary_Sphere_Radius"),
    ),
    "odd expansion order": (_replace(r"^(Expansion_Order:).*$", r"\1 7"),),
    "odd expansion order and a wrong number of types": (
        _replace(r"^(Expansion_Order:).*$", r"\1 7"),
        _replace(r"^(Atom_Types_Num:).*$", r"\1 3"),
    ),
    "wrong number of types": (_replace(r"^(Atom_Types_Num:).*$", r"\1 3"),),
    "negative spacing and a species without a file": (
        _replace(r"^(Grid_Spacing:).*$", r"\1 -0.3 ang"),
        _replace(r"^Atom_Type: C$", "Atom_Type: Xx"),
    ),
    "species without a file": (_replace(r"^Atom_Type: C$", "Atom_Type: Xx"),),
    "unknown keyword": (_append("Boundary_Vacuum: 5 ang"),),
    "tolerance with a unit that is none": (
        _append("Hartree_Boundary_Tolerance: 1e-3 furlong"),
    ),
    "tolerance of zero": (_append("Hartree_Boundary_Tolerance: 0 Ry"),),
    "order of the expansion above the largest": (
        _replace(r"^(Solver_Lpole:).*$", r"\1 99"),
    ),
    "no number of states": (_replace(r"^States_Num.*\n", ""),),
    "a coordinate row of two numbers": (
        _replace(r"^  2.60782206   0.00000000   0.00000$", "  2.60782206   0.0"),
    ),
    "a coordinate row of two numbers and an output level that is none": (
        _replace(r"^  2.60782206   0.00000000   0.00000$", "  2.60782206   0.0"),
        _replace(r"^(Output_Level:).*$", r"\1 many"),
    ),
    "an output level that is none": (_replace(r"^(Output_Level:).*$", r"\1 many"),),
    "a mixer that is none and a charge with a unit": (
        _replace(r"^(Mixing_Method:).*$", r"\1 Broyden"),
        _replace(r"^(Net_Charges:).*$", r"\1 0 coulomb"),
    ),
    "restart": (_append("Restart_Run: true"),),
}


def malformed_text(name: str) -> str:
    """Return ``BASE_INPUT`` with the mistakes of ``MALFORMED[name]``."""

    text = BASE_INPUT
    for edit in MALFORMED[name]:
        text = edit(text)
    return text


_RADIUS_LINE = r"^[ \t]*Boundary_Sphere_Radius\b.*\n"


def without_radius(text: str) -> str:
    """Return the text of an input with its sphere radius line taken out."""

    text = text.replace("\r\n", "\n")
    if len(re.findall(_RADIUS_LINE, text, flags=re.MULTILINE | re.IGNORECASE)) != 1:
        raise AssertionError("the input does not have one Boundary_Sphere_Radius line")
    return re.sub(_RADIUS_LINE, "", text, flags=re.MULTILINE | re.IGNORECASE)


_PBE = REPOSITORY / "examples" / "0_CH4_CF4" / "python_pbe"
# One fluorine atom with an electron its free atom does not hold.
FLUORIDE_INPUT = """\
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 5.0 ang
Grid_Spacing: 0.2 ang
Expansion_Order: 8
Coordinate_Unit: Cartesian_Ang
Atom_Types_Num: 1
Atom_Type: F
Local_Component: p
Read_VCD: true
begin Atom_Coord
  0.0  0.0  0.0
end Atom_Coord
Correlation_Type: pbe
States_Num: 10
Net_Charges: -1 e
Fermi_Temp: 500 K
Eigensolver: chebff
Mixing_Method: Anderson
"""


def two_atom_smoke(*lines: str) -> str:
    """The smoke input of the command lines with two atoms, and ``lines``.

    The atoms lie off the centre, so that the multipole series of their
    charges has a term at every order, and the SCF may converge.
    """

    text = (Path(__file__).parent / "data" / "H_cli_smoke.in").read_text(
        encoding="utf-8"
    )
    for edit in (
        _replace(r"^(Boundary_Sphere_Radius:).*$", r"\1 5.0 bohr"),
        _replace(r"^(Max_Iter:).*$", r"\1 40"),
        _replace(r"^    0.0  0.0  0.0$", "    0.0  0.0  0.7\n    0.0  0.0 -0.7"),
    ):
        text = edit(text.replace("\r\n", "\n"))
    return text + "\n" + "".join(line + "\n" for line in lines)


def _example(*parts: str) -> str:
    return (_PBE.joinpath(*parts) / "parsec.in").read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    )


def constructed_inputs() -> dict[str, tuple[str, Path]]:
    """Inputs made from shipped ones for a branch of the default rule each.

    Name, text with a sphere radius, and pseudopotential directory.
    """

    carbon_tetrafluoride = _replace(r"^(Ignore_Symmetry:).*$", r"\1 true")(
        _example("CF4", "IS")
    )
    return {
        "fluoride anion": (FLUORIDE_INPUT, _PBE / "pseudopotentials"),
        "methane with the core-hole carbon, neutral": (
            _replace(r"^(Net_Charges:).*$", r"\1 0 e")(_example("CH4", "FS_1s")),
            _PBE / "pseudopotentials",
        ),
        "CF4 on a grid of 0.1 ang through the origin": (
            _replace(r"^(Grid_Spacing:).*$", r"\1 0.1 ang")(carbon_tetrafluoride),
            _PBE / "pseudopotentials",
        ),
        "CF4 on a grid of 0.05 ang through the origin": (
            _replace(r"^(Grid_Spacing:).*$", r"\1 0.05 ang")(carbon_tetrafluoride),
            _PBE / "pseudopotentials",
        ),
        "CF4 on a grid of 0.2 ang through the origin": (
            _replace(r"^(Grid_Spacing:).*$", r"\1 0.2 ang")(carbon_tetrafluoride),
            _PBE / "pseudopotentials",
        ),
    }
