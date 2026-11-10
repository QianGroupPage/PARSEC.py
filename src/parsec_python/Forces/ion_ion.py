"""Ion-ion forces of an isolated system (``forceion.f90``).

For pseudopotential ionic charges ``Z_a`` the Rydberg pair energy is
``2*Z_a*Z_b/|R_a-R_b|`` and the force on atom ``a`` is

``F_a = sum_(b != a) 2*Z_a*Z_b*(R_a-R_b)/|R_a-R_b|**3``.

There is no Ewald or minimum-image sum: this is the nonperiodic molecular
expression that goes with :func:`parsec_python.V_ion.ion_ion_energy`.

Point charges ``q`` at ``R_q`` (``Add_Point_Charges``) push the atoms with
``F_a = sum_q 2*q*Z_a*(R_a-R_q)/|R_a-R_q|**3`` (``forcept.f90``), the
gradient of :func:`parsec_python.V_ion.point_charge_ion_energy`; PARSEC adds
it to the ion-ion forces.

:func:`ewald_ion_ion_forces` is the periodic counterpart (``ewald_sum.f90``): the exact derivative of
:func:`parsec_python.V_ion.ewald.ewald_ion_ion_energy`, from the same real-space translations and reciprocal vectors.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..models import Atom, PointCharge
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion.ewald import ION_ION_TOLERANCE, ewald_ion_ion_terms


def ion_ion_forces(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
) -> np.ndarray:
    """Return the ``(atoms, 3)`` ion-ion forces in Ry/bohr."""

    count = len(atoms)
    forces = np.zeros((count, 3), dtype=np.float64)
    if count < 2:
        return forces
    positions = np.array([atom.position for atom in atoms], dtype=np.float64)
    charges = np.array(
        [potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64
    )
    for index in range(count):
        delta = positions[index] - positions
        squared = np.einsum("ij,ij->i", delta, delta)
        squared[index] = 1.0
        if np.any(squared <= 0.0):
            raise ValueError("two nuclei occupy the same position")
        scale = 2.0 * charges[index] * charges / (squared * np.sqrt(squared))
        scale[index] = 0.0
        forces[index] = scale @ delta
    return forces


def point_charge_forces(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    point_charges: Sequence[PointCharge],
) -> np.ndarray:
    """Return the ``(atoms, 3)`` forces of the point charges in Ry/bohr."""

    forces = np.zeros((len(atoms), 3), dtype=np.float64)
    if not point_charges:
        return forces
    charges = np.array([item.charge for item in point_charges], dtype=np.float64)
    sites = np.array([item.position for item in point_charges], dtype=np.float64)
    for index, atom in enumerate(atoms):
        delta = np.asarray(atom.position, dtype=np.float64) - sites
        squared = np.einsum("ij,ij->i", delta, delta)
        if np.any(squared <= 0.0):
            raise ValueError("a point charge lies on a nucleus")
        ionic = float(potentials[atom.symbol].ionic_charge)
        scale = 2.0 * charges * ionic / (squared * np.sqrt(squared))
        forces[index] = scale @ delta
    return forces


def ewald_ion_ion_forces(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = ION_ION_TOLERANCE,
) -> np.ndarray:
    """Return the ``(atoms, 3)`` periodic ion-ion forces in Ry/bohr, ``-dE/dR`` of the Ewald energy.

    With ``d = |r_a - r_b + T|`` and ``w(G) = exp(-G^2/4 eta^2)/G^2`` the Hartree-unit force on atom ``a`` is

    ``F_a = sum_(b,T)' Z_a Z_b [erfc(eta d)/d^2 + 2 eta exp(-eta^2 d^2)/(sqrt(pi) d)] (r_a - r_b + T)/d
            + (4 pi/V) Z_a sum_(G != 0) w(G) G sum_b Z_b sin(G.(r_a - r_b))``,

    the prime excluding ``a = b`` at ``T = 0``; the self and neutralizing terms of the energy do not depend on the
    positions.  The result is doubled for Rydberg units, as the energy is.  The pairs and reciprocal vectors are
    those of :func:`parsec_python.V_ion.ewald.ewald_ion_ion_terms`, which evaluates both.
    """

    _, forces = ewald_ion_ion_terms(atoms, potentials, lattice_vectors, eta=eta, tolerance=tolerance, forces=True)
    return forces


__all__ = ["ewald_ion_ion_forces", "ion_ion_forces", "point_charge_forces"]
