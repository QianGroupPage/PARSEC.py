"""``parsec.out`` lines of the LDA+U term and of the TS dispersion correction.

The Fortran writes them from ``usrinputfile.F90`` (input echo), ``upot.F90``
(projector set-up), ``calcden.f90z`` (two energies before the eigenproblem of
every iteration but the first), ``totnrg.f90`` (``LDA+U Energy`` and
``TS-vDW correction (included)``), ``parsec.F90`` and ``efvdw.f90`` (the
report after convergence).  ``efvdw.f90`` writes its forces and volumes to
standard output; this reporter has one log and puts them there.

List-directed records (``write(7,*)``) follow gfortran: a record starts
with a blank, a character item is copied, a default integer takes twelve
columns and a double precision real 26 with its separator
(:func:`list_directed_real`).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..MolecularDynamics.fortran_format import list_directed_integer


RYDBERG_TO_EV = 13.6058
#: ``51.42208245`` of ``efvdw.f90``: Ha/bohr to eV/angstrom.
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 51.42208245


def list_directed_real(value: float) -> str:
    """A ``real(dp)`` item of a gfortran list-directed record, with its separator.

    gfortran writes a blank and 17 significant digits in 25 columns: in
    ``F`` form right-justified in 20 columns and followed by five blanks for
    ``0.1 <= |x| < 1e17`` (and zero), otherwise in ``E`` form with a
    three-digit exponent, right-justified in 25 columns.  The record's
    leading blank is the separator of its first item, so a row of reals is
    the concatenation of these strings; a character item after a real gets a
    blank separator of its own.
    """

    value = float(value)
    if not math.isfinite(value):
        text = "NaN" if math.isnan(value) else ("-Infinity" if value < 0 else "Infinity")
        return " " + text.rjust(25)
    if value == 0.0:
        return " " + "0.0000000000000000".rjust(20) + " " * 5
    mantissa, exponent = ("%.16e" % value).split("e")
    exponent = int(exponent)
    # The decimal exponent after rounding to 17 significant digits decides
    # the form, as G editing does.
    if 0.1 <= abs(value) and exponent < 17:
        decimals = 16 - exponent
        if decimals >= 0:
            return " " + ("%.*f" % (decimals, value)).rjust(20) + " " * 5
    return " " + (mantissa + "E" + "%+04d" % exponent).rjust(25)


def hubbard_parameter_lines(specification) -> list[str]:
    """The echo of ``usrinputfile.F90`` (``'(/,a,f10.3,a,f10.3,a)'``)."""

    if not getattr(specification, "has_hubbard", False):
        return []
    return [
        "",
        " LDA+U parameters : U = "
        f"{specification.hubbard_u * RYDBERG_TO_EV:10.3f} eV   J = "
        f"{specification.hubbard_j * RYDBERG_TO_EV:10.3f} eV",
    ]


def hubbard_setup_lines(system) -> list[str]:
    """The ``On-site Coulomb interaction (LDA+U) messages`` of ``upot.F90``."""

    projectors = getattr(system, "hubbard", None)
    if projectors is None:
        return []
    specifications = system.input.pseudopotentials
    lines = [
        "",
        " On-site Coulomb interaction (LDA+U) messages:",
        " ---------------------------------------------",
        "",
        "  Cut-off radii [bohr] ",
    ]
    species = [symbol for symbol in specifications if symbol in projectors.cutoff_radii]
    for symbol in species:
        lines.append(f"{symbol:<2.2s}{projectors.cutoff_radii[symbol]:12.4f}")
    lines += ["", "  Potential constant (eV) "]
    for symbol in species:
        specification = specifications[symbol]
        lines.append(
            f"{symbol:<2.2s} U = {specification.hubbard_u:12.4f} Ry       J = "
            f"{specification.hubbard_j:12.4f} Ry  ( l = {2:2d} )"
        )
    sizes = list(projectors.support_sizes)
    lines += [
        "",
        " Max # of non-local points for one atom = "
        + list_directed_integer(max(sizes, default=0)),
        "",
        " Sizes of all non-local blocks ",
    ]
    for start in range(0, len(sizes), 12):
        lines.append("".join(f" {size:5d}" for size in sizes[start : start + 12]))
    lines.append("")
    return lines


def hubbard_iteration_lines(step) -> list[str]:
    """What ``calcden.f90z`` prints before the eigenproblem of an iteration."""

    first = getattr(step, "hubbard_first_contribution", None)
    energy = getattr(step, "hubbard_energy", None)
    if first is None or energy is None:
        return []
    return [
        " ----------------------------------",
        "  LDA+U energy term (first contribution) = "
        + list_directed_real(first)
        + "  Ry",
        "  LDA+U energy term = " + list_directed_real(energy) + "  Ry",
        " ----------------------------------",
        "",
    ]


def energy_correction_lines(energies, *, dispersion: bool = False) -> list[str]:
    """``TS-vDW correction (included)`` and ``LDA+U Energy`` of ``totnrg.f90``.

    In the order of ``totnrg.f90`` after ``Ion-Ion Energy``: the TS line
    only in the report after convergence, the LDA+U line where its energy
    is not zero.
    """

    lines = []
    if dispersion:
        lines.append(
            "   TS-vDW correction (included)  = "
            f"{getattr(energies, 'dispersion', 0.0):20.8f} [Ry]"
        )
    hubbard = float(getattr(energies, "hubbard", 0.0))
    if hubbard != 0.0:
        lines.append(f"   LDA+U Energy                  = {hubbard:20.8f} [Ry]")
    return lines


def dispersion_header_lines(scf, periodic: bool) -> list[str]:
    """The echo of the ``VDW PART`` of ``usrinputfile.F90``."""

    if not getattr(scf, "van_der_waals", False):
        return []
    lines = [" including Tkatchenko-Scheffler dispersion correction!"]
    if periodic:
        cells = tuple(scf.hirshfeld_cells)
        lines += [
            " TS-VDW: with periodic calculation",
            " Got the following Hf parameters:",
            " " + "".join(list_directed_integer(value) for value in cells),
            " The numbers for the Hf cells are: 2*(Hf_One, Hf_Two, Hf_Three)+1 ",
            " " + "".join(list_directed_integer(2 * value + 1) for value in cells),
            " Note: The thresold for the TS energy component is 10^-6 [eV]",
        ]
    return lines


def _real_row(values: Sequence[float]) -> str:
    return "".join(list_directed_real(value) for value in values)


def dispersion_report_lines(result, atom_count: int | None = None) -> list[str]:
    """The report after convergence: ``efvdw.f90`` and the second ``totnrg``.

    ``result`` holds the :class:`~parsec_python.Energy.dispersion.DispersionResult`
    in ``dispersion`` and the energies with it in ``energies``; nothing is
    read from it without one.  ``atom_count`` defaults to its atoms.
    """

    dispersion = getattr(result, "dispersion", None)
    if dispersion is None:
        return []
    if atom_count is None:
        atom_count = len(result.atoms)
    energies = result.energies
    forces = np.asarray(dispersion.forces_hartree)
    volumes = dispersion.volumes
    lines = [" NOTE: Writing VDW force info to STDOUT", " vdW force components in units of eV/A"]
    lines += [_real_row(row * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM) for row in forces]
    lines += ["", " vdW force components in units of Ha/bohr"]
    lines += [_real_row(row) for row in forces]
    lines += ["", "", " Free and effective volumes of each atom"]
    lines += [
        _real_row((free, effective))
        for free, effective in zip(volumes.free, volumes.effective)
    ]
    lines += ["", " Ratio of effective and free volumes"]
    lines += [_real_row((ratio,)) for ratio in volumes.ratio]
    lines += ["", "", "     TOTAL ENERGY REPORT WITH TS-VDW CORRECTIONS", ""]
    lines += [
        f"   Eigenvalue Energy             = {energies.eigenvalue:20.8f} [Ry]",
        f"   Hartree Energy                = {energies.hartree:20.8f} [Ry]",
        f"   Integral_{{Vxc*rho}}            = {energies.integral_vxc_rho:20.8f} [Ry]",
        f"   Exc = Integral{{eps_xc*rho}}    = {energies.exchange_correlation:20.8f} [Ry]",
        f"   Electron-Ion energy           = {energies.electron_ion:20.8f} [Ry]",
        f"   Ion-Ion Energy                = {energies.ion_ion:20.8f} [Ry]",
        *energy_correction_lines(energies, dispersion=True),
        # eold is the electronic energy of the same iteration.
        f"   (E(new)-E(old))/atom  = {0.0:22.8f} [eV]",
        "",
        f"   Total Energy = {energies.total:22.8f} [Ry]",
        "   Energy/atom  = "
        f"{energies.total * RYDBERG_TO_EV / max(atom_count, 1):22.8f} [eV]",
        "",
    ]
    return lines


__all__ = [
    "dispersion_header_lines",
    "dispersion_report_lines",
    "energy_correction_lines",
    "hubbard_iteration_lines",
    "hubbard_parameter_lines",
    "hubbard_setup_lines",
    "list_directed_real",
]
