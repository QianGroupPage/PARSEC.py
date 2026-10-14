"""Files of a PARSEC molecular dynamics run.

PARSEC opens four files in ``initial.f90`` and appends to them in every call
of ``moldyn`` (``moldyn.f90``); a restart reads a fifth one:

``md_nrg.dat`` (unit 78)
    one line per time step: step, time, kinetic, potential and total energy
    per atom in eV, target and actual temperature.
``md_mech.dat`` (unit 79)
    for every time step a line ``Step #nnnn`` and one line per atom with
    thirteen numbers in ``1x,e15.8``: the next position, the current
    position, the current velocity estimate, the current acceleration and
    the movable flag.  Atomic units throughout.
``mdinit.dat`` (unit 76)
    the restart file: the last step of ``md_mech.dat``, copied by the user.
``atom.cor`` (unit 66)
    the coordinates of every step in bohr, grouped by atom type.
``movie.xyz`` (unit 92)
    the coordinates of every step in angstrom, in XYZ format.

Every function returns the text of complete lines, each ended by a newline,
or parses such text.  Nothing here opens a file: the caller chooses the
paths and appends.  :func:`initial_records` and :func:`step_records` give, per
file, what ``initial.f90`` writes when a run starts and what one call of
``moldyn`` writes.

The fixed formats follow the Fortran ``format`` statements.  The
list-directed records (``write(unit,*)``) are compiler dependent; they are
written as gfortran writes them.  Two things are kept although they look
like slips: the second header line of ``md_mech.dat`` reads
``current positioncurrent velocity``, and the atom count of ``movie.xyz``
is written with ``i3``, so that a cluster of a thousand atoms gets ``***``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .fortran_format import (
    fortran_e,
    fortran_f,
    fortran_i,
    list_directed_integer,
    read_fortran_integer,
    read_fortran_real,
)
from .moldyn import (
    BOHR_ANGSTROM,
    MolecularDynamicsState,
    MoldynStep,
)
from .settings import RESCALE_FRICTION, MolecularDynamicsSettings

MD_NRG_HEADER = (
    "Step  Time[ps]  Ekin[eV/at]  Epot[eV/at]  Etot[eV/at]     T[K]  Tcalc[K]\n"
)
MD_MECH_HEADER = (
    "Each line contains: six coordinates, three velocities, and three accelerations\n"
    "for each molecular dynamics step\n"
    "The data are: next position, current positioncurrent velocity\n"
    "current acceleration\n"
    "The last step of this file should be copied to mdinit.dat for a restart\n"
)


def _type_name(name: str) -> str:
    """An atom type name as the two characters of ``clust%name``."""
    return name.strip()[:2].ljust(2)


def _types(species: Sequence[str]) -> list[tuple[str, int]]:
    """Consecutive atoms of one name form one PARSEC atom type."""
    groups: list[tuple[str, int]] = []
    for name in species:
        if groups and groups[-1][0] == name:
            groups[-1] = (name, groups[-1][1] + 1)
        else:
            groups.append((name, 1))
    return groups


def _positions(positions: np.ndarray | Sequence, count: int | None = None) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("positions must have shape (atoms, 3)")
    if count is not None and array.shape[0] != count:
        raise ValueError("one species name per atom is required")
    return array


# ----------------------------------------------------------------------
# md_nrg.dat
# ----------------------------------------------------------------------
def format_md_nrg_line(step: MoldynStep) -> str:
    """One line of ``md_nrg.dat``.

    ``format(i4,3x,f7.3,2x,f11.5,2(2x,f11.5),2x,f7.2,3x,f7.2)``: step, time
    in ps, kinetic energy per movable atom, potential energy per atom and
    their sum in eV, target temperature and actual temperature in K.  A
    number that does not fit its field appears as asterisks, as in Fortran.
    """
    return (
        fortran_i(step.frame, 4)
        + "   "
        + fortran_f(step.time_ps, 7, 3)
        + "  "
        + fortran_f(step.kinetic_energy_ev, 11, 5)
        + "  "
        + fortran_f(step.potential_energy_ev, 11, 5)
        + "  "
        + fortran_f(step.total_energy_ev, 11, 5)
        + "  "
        + fortran_f(step.target_temperature, 7, 2)
        + "   "
        + fortran_f(step.temperature, 7, 2)
        + "\n"
    )


# ----------------------------------------------------------------------
# md_mech.dat and mdinit.dat
# ----------------------------------------------------------------------
def _mechanics_lines(state: MolecularDynamicsState) -> str:
    """``format(13(1x,e15.8))`` for every atom."""
    table = np.hstack(
        [
            state.current_positions,
            state.previous_positions,
            state.velocities,
            state.accelerations,
            state.movable.astype(float)[:, None],
        ]
    )
    return "".join(
        "".join(" " + fortran_e(value, 15, 8) for value in row) + "\n"
        for row in table
    )


def format_md_mech_step(state: MolecularDynamicsState) -> str:
    """The block ``moldyn`` appends to ``md_mech.dat`` after a step.

    ``state`` is the state *after* the step.  The header is
    ``format('Step #',i4)`` with ``state.frame``; from step 10000 on the
    number is written as ``****`` and the block cannot be used for a restart.
    This block is also the content of ``mdinit.dat``.
    """
    return "Step #" + fortran_i(state.frame, 4) + "\n" + _mechanics_lines(state)


def format_md_restart(state: MolecularDynamicsState) -> str:
    """Text of a restart file ``mdinit.dat`` for ``state``."""
    return format_md_mech_step(state)


def format_md_mech_restart_step(state: MolecularDynamicsState) -> str:
    """The block ``initial.f90`` writes to ``md_mech.dat`` when a run restarts.

    Its header is list-directed (``write(79,*) 'Step #', iframe``) and so
    differs from the header of the blocks ``moldyn`` writes.
    """
    return (
        " Step #" + list_directed_integer(state.frame) + "\n" + _mechanics_lines(state)
    )


@dataclass(frozen=True, eq=False)
class MolecularDynamicsRestart:
    """Content of ``mdinit.dat``.

    ``frame`` is the number of the time step to continue with.  The four
    arrays have shape ``(atoms, 3)`` and the meaning of the state arrays of
    the same name; they carry the eight significant digits of the file.
    ``movable`` holds the thirteenth column as it is written, and PARSEC
    takes its integer part as ``clust%mvat``, which replaces the flags of
    the input file.
    """

    frame: int
    current_positions: np.ndarray
    previous_positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    movable: np.ndarray

    @property
    def atom_count(self) -> int:
        return int(self.current_positions.shape[0])


_STEP_HEADER = re.compile(r"\s*Step\s*#\s*(.*?)\s*$", flags=re.IGNORECASE)


def parse_md_restart(text: str, atom_count: int | None = None) -> MolecularDynamicsRestart:
    """Read ``mdinit.dat`` as ``initial.f90`` does.

    The first line is the header ``Step #nnnn``.  Each following line is
    read with ``format(13(1x,e15.8))``: thirteen fields of sixteen columns,
    by column position and not by blanks, with the Fortran rules for blank
    fields and missing decimal points.  ``atom_count`` lines are read; if it
    is ``None`` all lines up to the end or to the next ``Step #`` line are.

    PARSEC reads the header with ``format('Step #',i4)``.  A character
    constant in an input format is not standard Fortran: gfortran stops with
    "Constant string in input format", so a PARSEC built with gfortran
    cannot restart molecular dynamics at all, while compilers that accept
    it skip the six characters.  This function reads the number after
    ``Step #`` and also accepts the list-directed header that
    ``initial.f90`` itself writes.
    """
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        raise ValueError("the restart file is empty")
    match = _STEP_HEADER.match(lines[0])
    if match is None:
        raise ValueError(f"restart file: expected 'Step #' header, got {lines[0]!r}")
    try:
        frame = read_fortran_integer(match.group(1))
    except ValueError:
        raise ValueError(
            f"restart file: cannot read the step number in {lines[0]!r}"
        ) from None

    rows: list[list[float]] = []
    for line in lines[1:]:
        if atom_count is not None and len(rows) == atom_count:
            break
        if atom_count is None and (not line.strip() or _STEP_HEADER.match(line)):
            break
        padded = line.ljust(13 * 16)
        rows.append(
            [read_fortran_real(padded[16 * k + 1 : 16 * k + 16], 8) for k in range(13)]
        )
    if atom_count is not None and len(rows) != atom_count:
        raise ValueError(
            f"restart file: {len(rows)} atom lines found, {atom_count} expected"
        )
    if not rows:
        raise ValueError("restart file: no atom lines")
    table = np.array(rows, dtype=float)
    return MolecularDynamicsRestart(
        frame=frame,
        current_positions=table[:, 0:3].copy(),
        previous_positions=table[:, 3:6].copy(),
        velocities=table[:, 6:9].copy(),
        accelerations=table[:, 9:12].copy(),
        movable=table[:, 12].copy(),
    )


def last_md_mech_step(text: str) -> str:
    """Return the last step of ``md_mech.dat``: what belongs in ``mdinit.dat``."""
    lines = text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if re.match(r"\s*Step\s*#", lines[index]):
            return "\n".join(lines[index:]) + "\n"
    raise ValueError("no 'Step #' block in the md_mech.dat text")


def restart_molecular_dynamics(
    settings: MolecularDynamicsSettings,
    restart: MolecularDynamicsRestart,
    masses: np.ndarray | Sequence,
) -> MolecularDynamicsState:
    """State of a restarted run, as ``initial.f90`` prepares it.

    Positions, velocities, accelerations, the step counter and the movable
    flags come from ``mdinit.dat``.  The target temperature starts again at
    ``Tinit`` of the input and the call counter at zero, so the cooling
    schedule of the new input begins anew.

    The first self-consistent calculation is done at
    ``state.current_positions``.  ``initial.f90`` recentres that structure
    for the electronic structure if its bounding box is more than 0.1 bohr
    off the origin (and ``Restart_run`` is not set), but it does not shift
    the positions the integrator continues from; the structure returned by
    the first step is therefore not recentred.
    """
    return MolecularDynamicsState(
        masses=masses,
        movable=restart.movable.astype(np.int64),
        current_positions=restart.current_positions,
        previous_positions=restart.previous_positions,
        velocities=restart.velocities,
        accelerations=restart.accelerations,
        temperature=settings.initial_temperature,
        frame=restart.frame,
        move_count=0,
        restarted=True,
    )


# ----------------------------------------------------------------------
# atom.cor
# ----------------------------------------------------------------------
def _atom_cor_block(
    species: Sequence[str], positions: np.ndarray, frame: int, width: int, decimals: int
) -> str:
    positions = _positions(positions, len(species))
    lines = [" Coordinates for step #" + list_directed_integer(frame) + "\n"]
    atom = 0
    for name, count in _types(species):
        lines.append(" " + _type_name(name) + "\n")
        for _ in range(count):
            lines.append(
                "".join("  " + fortran_f(value, width, decimals) for value in positions[atom])
                + "\n"
            )
            atom += 1
    return "".join(lines)


def format_atom_cor_step(species: Sequence[str], positions: np.ndarray, frame: int) -> str:
    """Coordinates of one step as ``moldyn`` writes them to ``atom.cor``.

    A line `` Coordinates for step #`` with the step number, then for every
    atom type its name and the coordinates of its atoms in bohr with
    ``format(3(2x,f14.9))``.  ``species`` holds one name per atom; atoms of
    one type are consecutive, as in the input file.
    """
    return _atom_cor_block(species, positions, frame, 14, 9)


def format_atom_cor_restart(species: Sequence[str], positions: np.ndarray, frame: int) -> str:
    """The block ``initial.f90`` writes to ``atom.cor`` when a run restarts.

    Same layout with ``format(3(2x,f11.6))``.
    """
    return _atom_cor_block(species, positions, frame, 11, 6)


def format_atom_cor_next_run(positions: np.ndarray) -> str:
    """The closing block of ``atom.cor``: an empty line, the title
    `` Coordinates for next Run `` and the coordinates without type names in
    ``format(3(2x,f11.6))``."""
    positions = _positions(positions)
    lines = ["\n", "  Coordinates for next Run \n"]
    for row in positions:
        lines.append("".join("  " + fortran_f(value, 11, 6) for value in row) + "\n")
    return "".join(lines)


# ----------------------------------------------------------------------
# movie.xyz
# ----------------------------------------------------------------------
def format_movie_xyz_frame(species: Sequence[str], positions: np.ndarray, frame: int) -> str:
    """One frame of ``movie.xyz``.

    The number of atoms (``i3``), a list-directed title `` Step # `` with the
    step number, and for every atom ``format(a2, 3(3x, f10.6))`` with the
    coordinates in angstrom (1 bohr = 0.529177 angstrom, PARSEC's value).
    ``positions`` are in bohr.
    """
    positions = _positions(positions, len(species))
    lines = [
        fortran_i(len(species), 3) + "\n",
        " Step # " + list_directed_integer(frame) + "\n",
    ]
    for name, row in zip(species, positions):
        lines.append(
            _type_name(name)
            + "".join("   " + fortran_f(value * BOHR_ANGSTROM, 10, 6) for value in row)
            + "\n"
        )
    return "".join(lines)


# ----------------------------------------------------------------------
# parsec.out
# ----------------------------------------------------------------------
def format_parsec_out_summary(settings: MolecularDynamicsSettings) -> str:
    """The molecular dynamics paragraph ``usrinput`` writes to ``parsec.out``.

    One line ``No molecular dynamics!`` when it is off.  Otherwise the
    warnings for a restart, a user seed and the simple generator, the cooling
    scheme, the temperatures, the time step, the friction coefficient and the
    number of steps.  Two details of the Fortran text are kept: the time step
    is labelled ``[au]`` whatever unit the input named, and the ensemble
    line tests ``friction < 1e-5`` whereas ``moldyn`` integrates Newton's
    equations only below ``1e-6``.
    """
    if not settings.enabled:
        return " No molecular dynamics!\n"
    lines = [" Molecular dynamics data:", " -------------------------"]
    if settings.restart:
        lines.append(" WARNING: restarted molecular dynamics run")
        lines.append(" restart data assumed present in mdinit.dat")
    if settings.rng_seed > 0:
        lines.append(
            " WARNING: running with user defined RNG seed: "
            + list_directed_integer(settings.rng_seed)
        )
    if settings.use_test_rng:
        lines.append(" WARNING: running with simple RNG engine ")
    if settings.cooling_method == "stair":
        lines.append(' "Stairstep" cooling scheme ')
        lines.append(
            " Temperature step = " + fortran_f(settings.temperature_step, 6, 1) + " [K]"
        )
        lines.append(
            " Temperature modified after "
            + list_directed_integer(settings.stride)
            + "  iterations"
        )
    elif settings.cooling_method == "linear":
        lines.append(" Linear cooling scheme ")
    else:
        lines.append(" Logarithmic cooling scheme ")
    lines.append(
        " Initial Temperature = " + fortran_f(settings.initial_temperature, 6, 1) + " [K]"
    )
    lines.append(
        " Final Temperature = " + fortran_f(settings.final_temperature, 6, 1) + " [K]"
    )
    lines.append(
        " Time step = "
        + fortran_f(settings.time_step, 8, 2)
        + " [au] = "
        + fortran_f(settings.time_step_ps, 6, 3)
        + " [ps]"
    )
    lines.append(
        " Friction Coefficient = "
        + fortran_f(settings.friction_coefficient, 8, 5)
        + " [au]"
    )
    if settings.friction_coefficient < RESCALE_FRICTION:
        lines.append(" Small coefficient, doing MICROcanonical ensemble")
    else:
        lines.append(" Large coefficient, doing (MACRO)canonical ensemble")
    lines.append("  ")
    lines.append(" Number of MD iterations = " + fortran_i(settings.step_num, 5))
    lines.append("  ")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# what is written when
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MoldynRecords:
    """Text to append to each of the four output files."""

    md_nrg: str = ""
    md_mech: str = ""
    atom_cor: str = ""
    movie_xyz: str = ""

    def __add__(self, other: "MoldynRecords") -> "MoldynRecords":
        return MoldynRecords(
            self.md_nrg + other.md_nrg,
            self.md_mech + other.md_mech,
            self.atom_cor + other.atom_cor,
            self.movie_xyz + other.movie_xyz,
        )


def initial_records(
    state: MolecularDynamicsState,
    species: Sequence[str],
    positions: np.ndarray | None = None,
) -> MoldynRecords:
    """What ``initial.f90`` writes when it opens the files of a run.

    The headers of ``md_nrg.dat`` and ``md_mech.dat`` and the first frame of
    ``movie.xyz``; for a restart also the restart step in ``md_mech.dat``
    and the starting coordinates in ``atom.cor``.  ``positions`` are the
    cluster coordinates of the first self-consistent calculation
    (``clust%xatm`` after the optional recentring); they default to
    ``state.current_positions``.

    In a restart the Fortran program writes the thirteenth column of
    ``mdinit.dat`` back unchanged; here it is the integer flag derived from
    it, which is the same number in any file PARSEC or this module wrote.
    """
    if positions is None:
        positions = state.current_positions
    md_mech = MD_MECH_HEADER
    atom_cor = ""
    if state.restarted:
        md_mech += format_md_mech_restart_step(state)
        atom_cor = format_atom_cor_restart(species, positions, state.frame)
    return MoldynRecords(
        md_nrg=MD_NRG_HEADER,
        md_mech=md_mech,
        atom_cor=atom_cor,
        movie_xyz=format_movie_xyz_frame(species, positions, state.frame),
    )


def step_records(
    previous: MolecularDynamicsState,
    step: MoldynStep,
    species: Sequence[str],
) -> MoldynRecords:
    """What one call of ``moldyn`` writes.

    ``previous`` is the state that went into :func:`moldyn_step` and
    ``step`` what came out.  The first call of a fresh run puts the starting
    coordinates into ``atom.cor`` first.  Every call adds a block to
    ``md_mech.dat`` and a line to ``md_nrg.dat``.  Then the new coordinates
    go to ``atom.cor`` and ``movie.xyz``, except after the last step, where
    ``atom.cor`` gets the block "Coordinates for next Run" instead and
    ``movie.xyz`` nothing.
    """
    atom_cor = ""
    if previous.is_first_step:
        atom_cor += format_atom_cor_step(
            species, previous.current_positions, previous.frame
        )
    movie_xyz = ""
    if step.last_step:
        atom_cor += format_atom_cor_next_run(step.positions)
    else:
        atom_cor += format_atom_cor_step(species, step.positions, step.state.frame)
        movie_xyz = format_movie_xyz_frame(species, step.positions, step.state.frame)
    return MoldynRecords(
        md_nrg=format_md_nrg_line(step),
        md_mech=format_md_mech_step(step.state),
        atom_cor=atom_cor,
        movie_xyz=movie_xyz,
    )
