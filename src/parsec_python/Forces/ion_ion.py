"""Ion-ion forces of an isolated system (``forceion.f90``).

For pseudopotential ionic charges ``Z_a`` the Rydberg pair energy is
``2*Z_a*Z_b/|R_a-R_b|`` and the force on atom ``a`` is

``F_a = sum_(b != a) 2*Z_a*Z_b*(R_a-R_b)/|R_a-R_b|**3``.

There is no Ewald or minimum-image sum: this is the nonperiodic molecular
expression that goes with :func:`parsec_python.V_ion.ion_ion_energy`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..models import Atom
from ..Pseudopotential import ParsecPseudopotential


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


__all__ = ["ion_ion_forces"]
