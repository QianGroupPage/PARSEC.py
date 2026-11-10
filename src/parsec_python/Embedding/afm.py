"""AFM scans of an FDET tip (PARSEC.py extension; PARSEC 1.5 leaves them to the closed PARSEC-Edge app).

PARSEC 1.5 computes the energy of one tip position; its README describes the
HR-AFM workflow of the closed PARSEC-Edge application, which runs the tip over
a lateral grid at three heights with ``Minimization: manual`` per scan line
and forms the force gradient ``k_ts = d^2 E / dz^2`` by finite differences.
This module is that scan:

* the tip moves rigidly; the scan points are the input tip displaced by
  ``(dx, dy, (k - c) dz)`` for every lateral point ``(dx, dy)`` of
  ``AFM_Scan_Lateral_Points`` (one point ``(0, 0)`` without the block) and
  ``k = 0 .. n-1`` (``AFM_Scan_Heights``, ``c = (n-1) // 2``, so the input
  height is the middle one), ``dz`` = ``AFM_Scan_Height_Step``;
* the points run as one manual sequence (heights up along one lateral point,
  down along the next), each SCF loop continuing from the one before; with
  the tip sphere and ``dz`` a whole number of grid steps the tip keeps its
  grid points and only the sample field moves, in the classical mode the
  density and orbitals are translated (:mod:`parsec_python.Embedding.warm_start`);
* ``k_ts(z_k) = (E(z_k - dz) - 2 E(z_k) + E(z_k + dz)) / dz^2`` at every
  interior height.  No tilt correction (its formulas are not published) and
  no frequency shift are applied.  The table ``afm_scan.dat`` is written by
  :mod:`parsec_python.Output.afm_scan`.

``AFM_Scan_Line_Range: first last`` (1-based, inclusive) runs a part of the
lateral points, so that several processes or GPUs can share a scan; the
tables of the parts join line by line.  Each part runs in a folder of its own
(the parts write the same parsec.out, archive and restart files; a second
part in a folder is refused, :func:`parsec_python.Output.afm_scan.claim_scan_folder`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 1 Ry/bohr^2 in N/m (13.605693122994 eV / 0.529177210903^2 A^2, 1 eV/A^2 = 16.02176634 N/m).
RYDBERG_PER_BOHR2_IN_N_PER_M = 13.605693122994 / 0.529177210903**2 * 16.02176634


@dataclass(frozen=True)
class AFMScanSettings:
    """``AFM_Scan`` of an FDET tip input (lengths in bohr).

    ``lateral_points`` are the ``(dx, dy)`` of the scan lines, ``heights``
    the number of heights per line (at least 3), ``height_step`` ``dz``;
    ``line_range`` the 1-based lines this run takes, ``None`` for all;
    ``tip_positions`` the atoms of the input tip, which the scan displaces.
    """

    heights: int
    height_step: float
    lateral_points: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    line_range: tuple[int, int] | None = None
    tip_positions: tuple[tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if int(self.heights) < 3:
            raise ValueError("AFM_Scan_Heights must be at least 3 (k_ts needs three heights)")
        if not np.isfinite(self.height_step) or float(self.height_step) <= 0.0:
            raise ValueError("AFM_Scan_Height_Step must be positive")
        points = tuple((float(x), float(y)) for x, y in self.lateral_points)
        if not points:
            raise ValueError("AFM_Scan_Lateral_Points needs at least one point")
        object.__setattr__(self, "heights", int(self.heights))
        object.__setattr__(self, "height_step", float(self.height_step))
        object.__setattr__(self, "lateral_points", points)
        if self.line_range is not None:
            first, last = (int(value) for value in self.line_range)
            if not 1 <= first <= last <= len(points):
                raise ValueError(f"AFM_Scan_Line_Range must lie within 1..{len(points)}")
            object.__setattr__(self, "line_range", (first, last))

    @property
    def lines(self) -> range:
        """The 0-based lateral points of this run."""

        if self.line_range is None:
            return range(len(self.lateral_points))
        return range(self.line_range[0] - 1, self.line_range[1])

    def offsets(self) -> list[tuple[int, int, np.ndarray]]:
        """``(line, height index, displacement)`` of every scan point, in the order they run."""

        centre = (self.heights - 1) // 2
        points = []
        for order, line in enumerate(self.lines):
            heights = range(self.heights) if order % 2 == 0 else range(self.heights - 1, -1, -1)
            dx, dy = self.lateral_points[line]
            for k in heights:
                points.append((line, k, np.array([dx, dy, (k - centre) * self.height_step])))
        return points

    def structures(self, positions) -> list[np.ndarray]:
        """The positions of every scan point: the tip ``positions`` displaced rigidly."""

        base = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        return [base + displacement for _, _, displacement in self.offsets()]


@dataclass(frozen=True)
class AFMScanPoint:
    """One scan point: line, height index, displacement (bohr), energy (Ry), SCF converged."""

    line: int
    height: int
    displacement: np.ndarray
    energy: float
    converged: bool


def force_gradients(settings: AFMScanSettings, points) -> dict[tuple[int, int], float]:
    """``k_ts`` (Ry/bohr^2) by line and interior height index: the three-point second difference of ``E(z)``.

    Only where the three energies are there and all three SCF loops converged: a ``k_ts`` that would use an
    unconverged energy is left out (:func:`unconverged_gradients` names those heights).
    """

    energies = {(point.line, point.height): float(point.energy) for point in points}
    converged = {(point.line, point.height): bool(point.converged) for point in points}
    gradients = {}
    for line in settings.lines:
        for k in range(1, settings.heights - 1):
            keys = [(line, k - 1), (line, k), (line, k + 1)]
            if all(key in energies for key in keys) and all(converged[key] for key in keys):
                below, middle, above = (energies[key] for key in keys)
                gradients[(line, k)] = (below - 2.0 * middle + above) / settings.height_step**2
    return gradients


def unconverged_gradients(settings: AFMScanSettings, points) -> set[tuple[int, int]]:
    """The ``(line, height)`` whose three energies are there but not all converged: no ``k_ts`` is given there."""

    converged = {(point.line, point.height): bool(point.converged) for point in points}
    left_out = set()
    for line in settings.lines:
        for k in range(1, settings.heights - 1):
            keys = [(line, k - 1), (line, k), (line, k + 1)]
            if all(key in converged for key in keys) and not all(converged[key] for key in keys):
                left_out.add((line, k))
    return left_out


def scan_points(settings: AFMScanSettings, energies, converged) -> list[AFMScanPoint]:
    """The :class:`AFMScanPoint` of the structures of a finished scan, in the order they ran."""

    return [
        AFMScanPoint(line=line, height=k, displacement=displacement, energy=float(energy), converged=bool(done))
        for (line, k, displacement), energy, done in zip(settings.offsets(), energies, converged)
    ]


__all__ = [
    "AFMScanPoint",
    "AFMScanSettings",
    "RYDBERG_PER_BOHR2_IN_N_PER_M",
    "force_gradients",
    "scan_points",
    "unconverged_gradients",
]
