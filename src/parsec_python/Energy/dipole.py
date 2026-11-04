"""Total charge and dipole moment of a confined system (``dipole.f90``).

After every SCF loop of a cluster ``parsec.F90`` (lines 1644-1645) calls
``dipole``, which takes the zeroth and first moments of the charge of the
electrons and of the pseudo-ions:

``Q  = -h^3 sum_i rho_i       + sum_a Z_a``          [e]
``mu = -h^3 sum_i rho_i r_i   + sum_a Z_a R_a``      [e bohr]

``rho`` is the valence electron density (positive), ``r_i`` the grid point
``(shift + k) h`` measured from the origin of the grid and ``Z_a`` the ionic
charge of atom ``a``.  The dipole is reported in Debye with PARSEC's factor
``debye = 2.54`` of ``const.f90`` (the exact factor is 2.541746).  Point
charges (``Add_Point_Charges``) are no part of either sum, as in the Fortran.

``dipole.f90`` sums over the irreducible wedge and applies the operations of
the reduced group to each point; the full-grid density used here gives the
same sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

#: ``debye`` of ``const.f90``: e bohr -> Debye as PARSEC converts.
DEBYE_PER_E_BOHR = 2.54


@dataclass(frozen=True)
class DipoleMoment:
    """Charge in e and dipole moment in e bohr of a confined system."""

    charge: float
    dipole: np.ndarray

    @property
    def dipole_debye(self) -> np.ndarray:
        """The dipole in Debye with PARSEC's conversion factor 2.54."""

        return DEBYE_PER_E_BOHR * np.asarray(self.dipole, dtype=np.float64)


def dipole_moment(
    density: np.ndarray,
    coordinates: np.ndarray,
    volume_element: float,
    positions: Sequence[Sequence[float]] | np.ndarray,
    ionic_charges: Sequence[float] | np.ndarray,
) -> DipoleMoment:
    """Return the total charge and dipole of electrons plus pseudo-ions.

    ``density`` has shape ``(points,)`` in e/bohr^3, ``coordinates`` shape
    ``(points, 3)`` in bohr, ``positions`` shape ``(atoms, 3)`` in bohr and
    ``ionic_charges`` one ``Z_a`` per atom.  Nothing is mutated.
    """

    density = np.asarray(density, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape != (density.size, 3):
        raise ValueError("coordinates must have shape (points, 3) matching the density")
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(ionic_charges, dtype=np.float64).reshape(-1)
    if charges.size != positions.shape[0]:
        raise ValueError("one ionic charge per atom is needed")
    electronic_charge = -float(np.sum(density)) * float(volume_element)
    electronic_dipole = -(coordinates.T @ density) * float(volume_element)
    charge = electronic_charge + float(np.sum(charges))
    dipole = electronic_dipole + charges @ positions
    return DipoleMoment(charge=charge, dipole=np.asarray(dipole, dtype=np.float64))


def cluster_dipole_moment(
    density: np.ndarray,
    grid,
    atoms: Sequence,
    pseudopotentials: Mapping[str, object],
) -> DipoleMoment:
    """:func:`dipole_moment` of a density on an isolated grid.

    ``grid`` is a :class:`parsec_python.Grid.RealSpaceGrid` (its
    ``coordinates`` and ``volume_element``), ``atoms`` the atoms as the
    calculation placed them and ``pseudopotentials`` maps each symbol to an
    object with an ``ionic_charge``.
    """

    return dipole_moment(
        density,
        grid.coordinates,
        grid.volume_element,
        [atom.position for atom in atoms],
        [pseudopotentials[atom.symbol].ionic_charge for atom in atoms],
    )


__all__ = ["DEBYE_PER_E_BOHR", "DipoleMoment", "cluster_dipole_moment", "dipole_moment"]
