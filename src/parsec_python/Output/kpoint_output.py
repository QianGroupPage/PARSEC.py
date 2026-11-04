"""PARSEC's report of the space group and of the Monkhorst-Pack k-points.

Formats of ``parsec.F90:464-492`` (" Symmetry Properties:"),
``symgen.f90``/``atftmt`` (crystal system, symmorphic or not, operation
numbers, ``901`` rotation matrices and fractional translations),
``symmetries.f90:225-227`` (the Abelian-subgroup line), ``irrbz.f90:180-209``
(" K-point generation:" table and ``qpoints.dat``) and
``usrinputfile.F90:957-993`` (the Monkhorst-Pack lines of the input data),
reproduced character for character.  With the search mode of the run
(:func:`parsec_python.Grid.symmetry_analysis.symmetry_search_mode`) the
symmetry block also has ``symchk`` and the Abelian subgroup with its
character table (``symgen.f90`` l.209-271) and, at ``Output_Level >= 2``,
the multiplication tables of ``atftmt`` and ``symchk``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np


def _integers(values: Sequence[int], per_line: int, width: int, indent: int) -> list[str]:
    """Fortran ``(indent x, per_line i<width>)`` with format reversion."""

    values = [int(value) for value in values]
    lines = []
    for start in range(0, len(values), per_line):
        lines.append(
            " " * indent + "".join(f"{value:{width}d}" for value in values[start : start + per_line])
        )
    return lines


def _list_directed_items(*items) -> str:
    """One gfortran list-directed record of character and numeric items.

    The record starts with a blank; character items follow one another
    without separators, a number (integer: 12 columns, real: see
    :func:`parsec_python.Output.correction_output.list_directed_real`) and a
    character item after a number get a blank of their own.
    """

    from .correction_output import list_directed_real

    text = ""
    previous_number = None
    for item in items:
        if isinstance(item, str):
            text += (" " if previous_number is None or previous_number else "") + item
            previous_number = False
        elif isinstance(item, (int, np.integer)):
            text += " " + f"{int(item):11d}"
            previous_number = True
        else:
            text += list_directed_real(float(item))
            previous_number = True
    return text


def atftmt_table_lines(group, table) -> list[str]:
    """``atftmt``'s multiplication table (``symgen.f90`` l.501-571, ``ipr >= 2``).

    ``table`` is :func:`parsec_python.Grid.symmetry_analysis.atftmt_multiplication_table`.
    A non-symmorphic group (PARSEC's verdict) gets the fractional
    translations ``v = -tnp`` beside the rows of the first 24 columns.
    """

    numbers = list(group.candidate_numbers)
    count = len(numbers)
    translations = [tuple(-float(value) for value in vector) for vector in group.translations]
    lines: list[str] = []
    passes = [(0, min(count, 24))]
    if count > 24:
        passes.append((24, count))
    for pass_index, (start, stop) in enumerate(passes):
        with_translations = pass_index == 0 and not group.symmorphic
        if pass_index == 0 and with_translations:
            lines += [
                "",
                "   Multiplication table" + " " * 30 + "Fractional translations",
                "     " + "".join(f"{value:4d}" for value in numbers[start:stop]),
                "          v(1)      v(2)      v(3)",
            ]
        else:
            if pass_index == 0:
                lines += ["", "   Multiplication table"]
            lines.append("    x" + "".join(f"{value:4d}" for value in numbers[start:stop]))
        for row in range(count):
            lines.append(
                f"{numbers[row]:5d}" + "".join(f"{int(value):4d}" for value in table[row, start:stop])
            )
            if with_translations:
                lines.append("          " + "".join(f"{value:10.4f}" for value in translations[row]))
    return lines


def _two_digit(value: int) -> str:
    text = f"{int(value):2d}"
    return "**" if len(text) > 2 else text


def symmetry_check_lines(check, output_level: int) -> list[str]:
    """What ``symchk`` writes (``symgen.f90`` l.907-1053)."""

    lines: list[str] = []
    if check.duplicate is not None:
        first, second = check.duplicate
        if output_level >= 2:
            lines += ["", f" symmetry operations{first:3d} and{second:3d} are equal"]
        return lines + ["  ERROR: repeated symmetry operations"]
    if output_level >= 2:
        lines.append(" Checking in r-space")
    for i, j, k, column, error in check.failures:
        lines.append(_list_directed_items(i, j, k, column, float(error)))
    if output_level >= 2:
        lines += [" Multiplication table", ""]
        for row in check.table:
            lines.append(" " + "".join(_two_digit(value) for value in row))
    if check.error:
        lines.append("  ERROR: repeated symmetry operations")
    return lines


def abelian_lines(group, subgroup) -> list[str]:
    """The Abelian subgroup and its character table (``symgen.f90`` l.214-231)."""

    labels = [group.labels[index - 1].strip() for index in subgroup.operations]
    dashes = "-" * (18 + 5 * subgroup.order)
    lines = [
        "",
        _list_directed_items(" Found Abelian subgroup ", subgroup.name, " with ", subgroup.order, " symmetry operations"),
        "",
        "  Character table : character = exp(i*Pi*x/3)",
        dashes,
        " Representation |" + "".join(f"{label:>5s}" for label in labels),
    ]
    for row, (label, exponents) in enumerate(
        zip(subgroup.labels, subgroup.character_exponents()), start=1
    ):
        lines.append(
            f"   {label:3.3s} {row:5d}   | " + "".join(f"  {value:3d}" for value in exponents)
        )
    lines.append(dashes)
    numbers = list(subgroup.operations)
    lines.append(
        " Operation number:"
        + f" {numbers[0]:3d}"
        + "".join(f"  {value:3d}" for value in numbers[1:])
    )
    return lines


def space_group_lines(
    group,
    *,
    info: int = 1,
    norot: int = -1,
    output_level: int = 1,
    domain_warning: bool = False,
) -> list[str]:
    """The symmetry block of a run, from " Symmetry Properties:" on.

    ``group`` is a :class:`~parsec_python.Grid.space_group.SpaceGroup`; its
    ``ignored`` flag gives the lines of ``Ignore_Symmetry``.  ``info`` and
    ``norot`` are those of ``symmetries.f90`` (see
    :func:`parsec_python.Grid.symmetry_analysis.symmetry_search_mode`): with
    the defaults (``info = 1``, every k-point run) the Abelian subgroup is the
    identity and ``symchk`` is not run; with ``norot >= 0`` the block has
    ``symchk``'s lines and the Abelian subgroup PARSEC finds (the identity
    after a failed ``symchk``, which sets ``ntrans = 1``).
    ``output_level >= 2`` adds the multiplication tables;
    ``domain_warning`` the lines of a non-spherical cluster domain.
    """

    from ..Grid.symmetry_analysis import (
        abelian_subgroup,
        atftmt_multiplication_table,
        symmetry_check,
    )

    lines = ["", " Symmetry Properties:", " --------------------"]
    if domain_warning:
        lines += [
            "",
            " WARNING: Symmetry is not currently supported with",
            " non-spherical domain cluster calculations.",
            " Symmetry will not be used.",
        ]
    if group.ignored:
        return lines + [
            " Symmetry suppressed by the Ignore_Symmetry flag.",
            "",
            "skipping non-trivial symmetry operations",
            "Abelian subgroup contains only the Identity operation",
            "",
        ]
    if not group.crystal_system:
        # skipped_space_group: no search (info = 2 without Ignore_Symmetry).
        return lines + [
            "",
            "skipping non-trivial symmetry operations",
            "Abelian subgroup contains only the Identity operation",
            "",
        ]
    lines.append("")
    if group.full_holohedry:
        lines.append(f" The point group of the system is the full {group.crystal_system:>12s} group")
    else:
        lines.append(
            f" The crystallographic system is {group.crystal_system} with operations: "
        )
        lines.extend(_integers(group.lattice_operations, 16, 3, 5))
        if len(group.lattice_operations) % 16 == 0:
            # Format 900 is 4(/5x,16i3): after a full row the next
            # repetition's slash still ends a record (an empty line).
            lines.append("")
    if group.symmorphic:
        lines += [
            "",
            " The space group of the crystal is symmorphic",
            " (this usually means that symmetry operations do not",
            " involve rotations and translations simultaneously)",
            "",
        ]
    else:
        lines += [
            "",
            " The space group is non-symmorphic",
            " (Or a non-standard origin of coordinates is used)",
            "",
        ]
    lines.append(" Operation number: ")
    numbers = list(group.candidate_numbers)
    lines.extend(_integers(numbers[:24], 12, 3, 5))
    if len(numbers) > 24:
        lines.extend(_integers(numbers[24:], 12, 3, 5))
    if output_level >= 2:
        lines.extend(atftmt_table_lines(group, atftmt_multiplication_table(group)))
    lines += [
        "",
        "",
        "    Rotation matrices (r-lattice) and fractional translations (r-lattice)",
        "",
    ]
    for number, (matrix, translation, label) in enumerate(
        zip(group.rotations, group.translations, group.labels), start=1
    ):
        lines.append(
            f"{number:5d}"
            + "".join("   " + "".join(f"{int(value):3d}" for value in row) for row in matrix)
            + "    "
            + "".join(f"{float(value):9.5f}" for value in translation)
            + " "
            + f"{label:>5s}"
        )
    if norot < 0:
        # symgen.f90 l.211-215: no Abelian subgroup (nab = 1).
        return lines + ["", "Abelian subgroup contains only the Identity operation", ""]
    check = symmetry_check(group)
    lines.extend(symmetry_check_lines(check, output_level))
    # symgen.f90 l.210: a failed check leaves the identity alone.
    analysed = group if not check.error else replace(
        group,
        rotations=group.rotations[:1],
        translations=group.translations[:1],
        cartesian_rotations=group.cartesian_rotations[:1],
        labels=group.labels[:1],
        candidate_numbers=group.candidate_numbers[:1],
    )
    table = check.table if not check.error else np.ones((1, 1), dtype=np.int64)
    subgroup = abelian_subgroup(analysed, table, norot)
    lines.extend(abelian_lines(analysed, subgroup))
    lines.append("")
    if subgroup.order == 1:
        lines += ["Abelian subgroup contains only the Identity operation", ""]
    return lines


def kpoint_generation_lines(sampling) -> list[str]:
    """``irrbz``'s table of the irreducible k-points.

    ``sampling`` is a :class:`~parsec_python.Grid.kpoints.MonkhorstPackSampling`.
    The weights are ``count/N`` as ``irrbz`` prints them, before ``init_var``
    renormalizes them.
    """

    lines = [
        "",
        " K-point generation:",
        " -------------------",
        f" {sampling.count:5d} K points generated from Monkhorst-Pack grid :",
        "    order = "
        + "".join(f"{int(value):5d}" for value in sampling.dimensions)
        + "      shift = "
        + "".join(f"{float(value):8.4f}" for value in sampling.shift),
        "",
        "    I     weight        relative coord.        Cartesian coord.     E_kin (Ry)",
        "",
    ]
    weights = sampling.printed_weights
    for index in range(sampling.count):
        lines.append(
            f" {index + 1:4d} {weights[index]:11.7f} "
            + "".join(f"{float(value):8.4f}" for value in sampling.fractional[index])
            + "   "
            + "".join(f"{float(value):7.3f}" for value in sampling.k_points[index])
            + f"   {float(sampling.kinetic_energies[index]):7.3f}"
        )
    return lines


def qpoints_dat_text(sampling) -> str:
    """The ``qpoints.dat`` file ``irrbz`` writes (``(3f14.10,a,i1)`` rows).

    The last column is 1 for the Gamma point (``|k|^2 < 1e-6`` in
    reciprocal-lattice coordinates), as BerkeleyGW reads it.
    """

    rows = ["begin qpoints"]
    for point in np.asarray(sampling.fractional, dtype=float):
        gamma = 1 if float(point[0] ** 2 + point[1] ** 2 + point[2] ** 2) < 1.0e-6 else 0
        rows.append("".join(f"{float(value):14.10f}" for value in point) + "  1.0  " + f"{gamma:1d}")
    rows.append("end qpoints")
    return "\n".join(rows) + "\n"


def monkhorst_pack_input_lines(dimensions, shift) -> list[str]:
    """What ``usrinputfile.F90`` prints for ``Kpoint_Method: mp`` with a shift."""

    n1, n2, n3 = (int(value) for value in dimensions)
    s1, s2, s3 = (float(value) for value in shift)
    return [
        "  kpoints selected by Monkhort-Pack algorithm",
        "  Monkhorst-Pack grid dimensions specified",
        f" {n1:3d} X{n2:3d} X{n3:3d}",
        "  Monkhorst-Pack shift specified",
        f" {s1:7.3f} X{s2:7.3f} X{s3:7.3f}",
    ]


__all__ = [
    "kpoint_generation_lines",
    "monkhorst_pack_input_lines",
    "qpoints_dat_text",
    "space_group_lines",
]
