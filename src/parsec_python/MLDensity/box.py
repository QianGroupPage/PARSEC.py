"""Regular boxes of points around a molecule, and XYZ input.

A box prediction is for densities wanted apart from one PARSEC grid: to look
at, to give another code, or, as in the old-architecture ML4Den workflow, to
supply a calculation through interpolation.  Two of box size, points per
direction, and spacing fix the box; the points are placed symmetrically about
the midpoint of the bounding box of the atoms, which is where PARSEC.py puts
the origin of a recentred calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .field import BOHR_TO_ANGSTROM


ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn"
).split()


def atomic_number(symbol: str) -> int:
    try:
        return ELEMENTS.index(symbol) + 1
    except ValueError:
        raise ValueError(f"unknown chemical element {symbol!r}") from None


@dataclass(frozen=True)
class BoxGrid:
    """Points ``origin + [i, j, k] * spacing`` for ``0 <= i, j, k < shape``.

    Lengths are bohr.  A point stands for a voxel of ``spacing`` edges, so
    the box it covers is ``shape * spacing`` long.
    """

    origin_bohr: np.ndarray
    spacing_bohr: np.ndarray
    shape: tuple[int, int, int]

    @property
    def voxel_vectors_bohr(self) -> np.ndarray:
        return np.diag(self.spacing_bohr)

    @property
    def lengths_bohr(self) -> np.ndarray:
        return np.asarray(self.shape) * self.spacing_bohr

    @property
    def lower_bohr(self) -> np.ndarray:
        """Lower corner of the covered box."""

        return self.origin_bohr - 0.5 * self.spacing_bohr

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume_bohr3(self) -> float:
        return float(np.prod(self.spacing_bohr))

    def points(self) -> np.ndarray:
        """All points, ``k`` fastest (the order of ``density.reshape(shape)``)."""

        axes = [
            self.origin_bohr[axis] + np.arange(count) * self.spacing_bohr[axis]
            for axis, count in enumerate(self.shape)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([item.reshape(-1) for item in mesh], axis=1)

    def integrate(self, values: np.ndarray) -> float:
        return float(np.sum(values) * self.voxel_volume_bohr3)


def _three(value, label: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.size == 1:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{label} needs one or three positive values")
    return array


def box_around_atoms(
    atom_positions_bohr: np.ndarray,
    *,
    box_bohr=None,
    vacuum_bohr=None,
    points=None,
    spacing_bohr=None,
) -> BoxGrid:
    """A box of points centred on the atoms.

    Give exactly two of: the box size (``box_bohr``, edge lengths, or
    ``vacuum_bohr``, the distance added on each side of the atoms' extent),
    ``points`` per direction, and ``spacing_bohr``.  Box and points fix the
    spacing; box and spacing round the number of points up, so the box can
    grow by less than one spacing; points and spacing fix the box.
    """

    positions = np.asarray(atom_positions_bohr, dtype=float).reshape(-1, 3)
    if positions.shape[0] == 0:
        raise ValueError("a box needs at least one atom")
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    extent, centre = upper - lower, 0.5 * (lower + upper)
    if box_bohr is not None and vacuum_bohr is not None:
        raise ValueError("give the box size either as edge lengths or as vacuum")
    if vacuum_bohr is not None:
        vacuum = np.atleast_1d(np.asarray(vacuum_bohr, dtype=float))
        if vacuum.size == 1:
            vacuum = np.repeat(vacuum, 3)
        if vacuum.shape != (3,) or np.any(vacuum < 0):
            raise ValueError("vacuum needs one or three nonnegative values")
        lengths = extent + 2.0 * vacuum
    else:
        lengths = _three(box_bohr, "the box size")
    counts = None
    if points is not None:
        counts = np.atleast_1d(np.asarray(points))
        if counts.size == 1:
            counts = np.repeat(counts, 3)
        if counts.shape != (3,) or np.any(counts < 1) or np.any(counts != np.round(counts)):
            raise ValueError("points per direction needs one or three positive integers")
        counts = counts.astype(int)
    spacing = _three(spacing_bohr, "the spacing")
    given = sum(item is not None for item in (lengths, counts, spacing))
    if given != 2:
        raise ValueError(
            "give exactly two of: box size (or vacuum), points per direction, spacing"
        )
    if lengths is None:
        lengths = counts * spacing
    elif counts is None:
        counts = np.asarray(
            [max(1, math.ceil(length / step - 1.0e-9)) for length, step in zip(lengths, spacing)]
        )
        lengths = counts * spacing
    else:
        spacing = lengths / counts
    if np.any(lengths + 1.0e-9 < extent):
        raise ValueError(
            "the box is smaller than the molecule: edges "
            + " x ".join(f"{value * BOHR_TO_ANGSTROM:.3f}" for value in lengths)
            + " angstrom, atoms span "
            + " x ".join(f"{value * BOHR_TO_ANGSTROM:.3f}" for value in extent)
        )
    origin = centre - 0.5 * (counts - 1) * spacing
    return BoxGrid(
        origin_bohr=origin,
        spacing_bohr=np.asarray(spacing, dtype=float),
        shape=tuple(int(value) for value in counts),
    )


def read_xyz(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Symbols and positions (angstrom) of the first frame of an XYZ file."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    try:
        count = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"{source.name}: the first line must give the number of atoms") from None
    if len(lines) < count + 2:
        raise ValueError(f"{source.name}: expected {count} atom lines")
    symbols: list[str] = []
    positions = np.empty((count, 3))
    for row, line in enumerate(lines[2 : 2 + count]):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{source.name}: atom line {row + 1} needs a symbol and x y z")
        label = fields[0]
        if label.isdigit():
            symbol = ELEMENTS[int(label) - 1]
        else:
            symbol = label[:1].upper() + label[1:2].lower()
            if len(label) > 1 and not label[1].isalpha():
                symbol = label[:1].upper()
        atomic_number(symbol)
        symbols.append(symbol)
        positions[row] = [float(value) for value in fields[1:4]]
    return symbols, positions


__all__ = [
    "BoxGrid",
    "ELEMENTS",
    "atomic_number",
    "box_around_atoms",
    "read_xyz",
]
