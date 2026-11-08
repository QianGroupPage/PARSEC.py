"""``polar.dat`` of a polarizability run, character for character as PARSEC.

``initial.f90`` (lines 246-271) writes the header when the run starts,
``polar.f90`` one line after each of the seven SCF loops (format 93) and the
polarizabilities after the last one (formats 94-98).  List-directed records
follow gfortran: a leading blank and twelve columns for a default integer.
"""

from __future__ import annotations

from typing import Sequence

from ..MolecularDynamics.fortran_format import fortran_f, list_directed_integer


def _fortran_f(value: float, width: int, decimals: int) -> str:
    """``Fw.d``, with gfortran's point after the digits of ``d = 0``."""

    if decimals:
        return fortran_f(value, width, decimals)
    text = "%.0f." % (float(value) + 0.0)
    return "*" * width if len(text) > width else text.rjust(width)


def polar_header(atoms: Sequence, species: Sequence[str], field: float) -> str:
    """The header ``initial.f90`` writes into ``polar.dat``.

    ``atoms`` as the calculation placed them and ``species`` the atom types
    in the order of the input.
    """

    lines = [
        " Tot. number of atoms is: " + list_directed_integer(len(atoms)),
        " Number of different types of atoms is: " + list_directed_integer(len(species)),
    ]
    for symbol in species:
        members = [atom for atom in atoms if atom.symbol == symbol]
        lines += ["", f"{len(members):3d} {symbol[:2]:<2s} atoms with coordinates:"]
        for atom in members:
            lines.append(
                "".join("   " + _fortran_f(value, 10, 6) for value in atom.position)
            )
    lines += [
        "",
        " Applied Electric field  = " + _fortran_f(field, 8, 5) + " [Ry/au]",
        "",
        "   Field           Tot.              Dipole  Moment [Deb.]",
        " Component      Energy [Ry]          X          Y          Z",
    ]
    return "\n".join(lines) + "\n"


def polar_run_line(direction, total_energy: float, dipole_debye) -> str:
    """Format 93 of ``polar.f90``: direction, energy [Ry], dipole [Debye]."""

    x, y, z = direction
    return (
        _fortran_f(x, 4, 0)
        + _fortran_f(y, 3, 0)
        + _fortran_f(z, 3, 0)
        + "  "
        + _fortran_f(total_energy, 17, 12)
        + "  "
        + "".join(_fortran_f(value, 11, 6) for value in dipole_debye)
        + "\n"
    )


def polar_summary(result) -> str:
    """What ``polar.f90`` writes after the seventh run (formats 94-98).

    ``result`` is a :class:`parsec_python.SCF.polarizability.PolarizabilityResult`.
    """

    energy = result.from_energy
    dipole = result.from_dipole
    average_energy = float(sum(energy)) / 3.0
    average_dipole = float(sum(dipole)) / 3.0
    lines = [
        "",
        " Tot. charge of the cluster: " + _fortran_f(result.charge, 7, 3),
        "",
        " Polarizability of the cluster (A**3):",
        "                 alfaXX    alfaYY    alfaZZ       average",
        " From energy: "
        + "".join(_fortran_f(value, 10, 4) for value in energy)
        + "   "
        + _fortran_f(average_energy, 10, 4),
        " From dipole: "
        + "".join(_fortran_f(value, 10, 4) for value in dipole)
        + "   "
        + _fortran_f(average_dipole, 10, 4),
        "",
        " Polarizability per atom ="
        + _fortran_f(average_energy / result.atom_count, 10, 4)
        + " A**3  (energy)",
        "                          "
        + _fortran_f(average_dipole / result.atom_count, 10, 4)
        + " A**3  (dipole)",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["polar_header", "polar_run_line", "polar_summary"]
