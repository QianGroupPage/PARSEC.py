"""``Output_Grid_Data``: Gaussian cube files as ``export_grid_data.f90`` writes them.

PARSEC writes, at the end of a run, one file per requested field:

==========================  ======================  ==========================
``Output_Grid_Data`` value  file                    field
==========================  ======================  ==========================
``charge_density``          ``charge_den.cube``     valence density, e/bohr^3
``v_external``              ``v_external.cube``     local ionic potential, Ry
``v_hartree``               ``v__hartree.cube``     Hartree potential, Ry
``v_external+v_hartree``    ``v_elecstat.cube``     their sum, Ry
==========================  ======================  ==========================

The layout (``write_cube``): two title lines written list-directed (with a
leading blank), ``(I5,3F12.6)`` with one atom and the origin, three
``(I5,3F12.6)`` voxel lines (``n_i, h e_i`` for a confined direction,
``n_i, a_i/n_i`` for a periodic one), a dummy hydrogen atom at the origin
``(I5,5F12.6)``, and the values ``(6E13.5)``: x outer, y middle, z inner,
a new line for every (x, y).  The box is the whole Cartesian box of the
grid (``n = int(2 rmax/h) + 2`` points per axis for a cluster), zero outside
the domain.

Deviation: with symmetry PARSEC fills only the points of the irreducible
wedge (the file comments that it "probably does not work with symmetry"; the
other points hold whatever the array held).  Python fills every point of the
full grid.  The origin is that of the Python grid: ``(shift + k_min) h`` for
a cluster as in PARSEC, ``0`` for a periodic cell, whose points are
``i h, i = 0..n-1`` in Python.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

#: ``Output_Grid_Data`` value -> file stem of ``export_grid_data.f90``.
CUBE_FILE_STEMS = {
    "charge_density": "charge_den",
    "v_external": "v_external",
    "v_hartree": "v__hartree",
    "v_external+v_hartree": "v_elecstat",
}


def fortran_exponential(value: float, width: int = 13, digits: int = 5) -> str:
    """``Ew.d`` of gfortran: ``0.ddddd`` times a power of ten, e.g. ``  0.12345E-02``.

    The exponent takes three digits and drops the ``E`` above 99, as the
    Fortran standard asks.
    """

    value = float(value)
    if not math.isfinite(value):
        text = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return text.rjust(width)[-width:]
    if value == 0.0:
        mantissa, exponent = "0" * digits, 0
        negative = math.copysign(1.0, value) < 0
    else:
        negative = value < 0
        scientific = f"{abs(value):.{digits - 1}E}"
        head, power = scientific.split("E")
        mantissa = head.replace(".", "")
        exponent = int(power) + 1
    if abs(exponent) <= 99:
        tail = f"E{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    else:
        tail = f"{'-' if exponent < 0 else '+'}{abs(exponent):03d}"
    text = f"{'-' if negative else ''}0.{mantissa}{tail}"
    if len(text) > width:
        text = text.replace("0.", ".", 1)
    return "*" * width if len(text) > width else text.rjust(width)


def _is_periodic(grid) -> bool:
    return hasattr(grid, "points_per_axis")


def dense_box(values: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place a field of ``grid`` in its Cartesian box, zero outside the domain.

    Returns the ``(n1, n2, n3)`` array with the smallest index first, the
    position of its first element and the three voxel vectors (bohr).
    """

    values = np.asarray(values, dtype=float)
    if values.shape != (grid.size,):
        raise ValueError("the field does not hold one value per grid point")
    if getattr(grid, "is_confined_periodic", False):
        # A wire or slab: its index box from index_min, zero outside the
        # domain; voxels a_i/n_i along the periodic directions and h e_i
        # along the confined ones (export_grid_data.f90).
        dense = grid.to_box(values)
        origin = grid.physical_coordinates(np.asarray(grid.index_min)[None, :])[0]
        voxels = (grid.normalized_lattice * grid.spacings[None, :]).T
        return dense, origin, voxels
    if _is_periodic(grid):
        counts = tuple(int(n) for n in grid.points_per_axis)
        dense = np.zeros(counts, dtype=float)
        points = np.asarray(grid.integer_coordinates, dtype=np.int64) % np.asarray(counts)
        dense[points[:, 0], points[:, 1], points[:, 2]] = values
        lattice = np.asarray(grid.cell.lattice_vectors, dtype=float)
        voxels = lattice / np.asarray(counts, dtype=float)[:, None]
        if getattr(grid, "is_general_periodic", False):
            # Fortran's general grid starts at its shift along each direction.
            return dense, grid.physical_coordinates(np.zeros((1, 3)))[0], voxels
        return dense, np.zeros(3), voxels
    dense = np.zeros(grid.lookup.shape, dtype=float)
    local = np.asarray(grid.integer_coordinates) - np.asarray(grid.index_min)[None, :]
    dense[local[:, 0], local[:, 1], local[:, 2]] = values
    origin = (np.asarray(grid.index_min, dtype=float) + np.asarray(grid.settings.shift)) * grid.spacing
    return dense, origin, np.eye(3) * float(grid.spacing)


def cube_text(name: str, dense: np.ndarray, origin: np.ndarray, voxels: np.ndarray) -> str:
    """The text of ``write_cube`` for the field file stem ``name``."""

    lines = [
        " Cubefile created from PARSEC calculation",
        f" Volumetric {name} data",
        f"{1:5d}" + "".join(f"{value:12.6f}" for value in origin),
    ]
    for count, vector in zip(dense.shape, voxels):
        lines.append(f"{count:5d}" + "".join(f"{value:12.6f}" for value in vector))
    lines.append(f"{1:5d}" + "".join(f"{value:12.6f}" for value in (1.0, 0.0, 0.0, 0.0)))
    rows = dense.reshape(-1, dense.shape[2])
    for row in rows:
        for start in range(0, row.size, 6):
            lines.append("".join(fortran_exponential(value) for value in row[start : start + 6]))
    return "\n".join(lines) + "\n"


def write_parsec_cube(path: str | Path, name: str, values: np.ndarray, grid) -> Path:
    """Write ``values`` (one per point of ``grid``) as ``export_grid_data.f90`` does."""

    output = Path(path)
    dense, origin, voxels = dense_box(values, grid)
    output.write_text(cube_text(name, dense, origin, voxels), encoding="ascii", newline="\n")
    return output


def grid_data_fields(result) -> dict[str, np.ndarray]:
    """The fields of ``Output_Grid_Data`` from a Python SCF result.

    ``rho(:,1)`` is the total valence density; ``v_external`` the local
    ionic potential ``vion`` and ``v_hartree`` the Hartree potential of the
    last density, both in Ry.
    """

    density = getattr(result, "density", None)
    if density is None and hasattr(result, "density_up"):
        density = np.asarray(result.density_up) + np.asarray(result.density_down)
    ionic = getattr(result, "ionic_potential", None)
    hartree = getattr(result, "hartree_potential", None)
    fields = {}
    if density is not None:
        fields["charge_density"] = np.asarray(density, dtype=float)
    if ionic is not None:
        fields["v_external"] = np.asarray(ionic, dtype=float)
    if hartree is not None:
        fields["v_hartree"] = np.asarray(hartree, dtype=float)
    if ionic is not None and hartree is not None:
        fields["v_external+v_hartree"] = np.asarray(ionic, dtype=float) + np.asarray(hartree, dtype=float)
    return fields


__all__ = [
    "CUBE_FILE_STEMS",
    "cube_text",
    "dense_box",
    "fortran_exponential",
    "grid_data_fields",
    "write_parsec_cube",
]
