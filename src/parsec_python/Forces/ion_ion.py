"""Ion-ion forces of an isolated system (``forceion.f90``).

For pseudopotential ionic charges ``Z_a`` the Rydberg pair energy is
``2*Z_a*Z_b/|R_a-R_b|`` and the force on atom ``a`` is

``F_a = sum_(b != a) 2*Z_a*Z_b*(R_a-R_b)/|R_a-R_b|**3``.

There is no Ewald or minimum-image sum: this is the nonperiodic molecular
expression that goes with :func:`parsec_python.V_ion.ion_ion_energy`.

:func:`ewald_ion_ion_forces` is the periodic counterpart (``ewald_sum.f90``): the exact derivative of
:func:`parsec_python.V_ion.ewald.ewald_ion_ion_energy`, from the same real-space translations and reciprocal vectors.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from scipy.special import erfc

from ..models import Atom
from ..Pseudopotential import ParsecPseudopotential
from ..V_ion.ewald import ewald_image_sets


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


def ewald_ion_ion_forces(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Return the ``(atoms, 3)`` periodic ion-ion forces in Ry/bohr, ``-dE/dR`` of the Ewald energy.

    With ``d = |r_a - r_b + T|`` and ``w(G) = exp(-G^2/4 eta^2)/G^2`` the Hartree-unit force on atom ``a`` is

    ``F_a = sum_(b,T)' Z_a Z_b [erfc(eta d)/d^2 + 2 eta exp(-eta^2 d^2)/(sqrt(pi) d)] (r_a - r_b + T)/d
            + (4 pi/V) Z_a sum_(G != 0) w(G) G sum_b Z_b sin(G.(r_a - r_b))``,

    the prime excluding ``a = b`` at ``T = 0``; the self and neutralizing terms of the energy do not depend on the
    positions.  The result is doubled for Rydberg units, as the energy is.
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    volume = float(abs(np.linalg.det(lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")
    count = len(atoms)
    forces = np.zeros((count, 3), dtype=np.float64)
    if count == 0:
        return forces
    positions = np.stack([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    eta, real_cutoff, translations, reciprocal_points = ewald_image_sets(positions, lattice_vectors, eta, tolerance)
    charge_products = charges[:, None] * charges[None, :]
    extent = positions.max(axis=0) - positions.min(axis=0)

    for translation in translations:
        if np.linalg.norm(np.maximum(np.abs(translation) - extent, 0.0)) > real_cutoff:
            continue  # the image the energy skips
        displacement = positions[:, None, :] - positions[None, :, :] + translation[None, None, :]
        distance = np.linalg.norm(displacement, axis=2)
        if bool(np.all(translation == 0.0)):
            np.fill_diagonal(distance, np.inf)
        if float(np.min(distance)) > real_cutoff:
            continue
        # -d/dd [erfc(eta d)/d], over d for the unit vector
        radial = (
            erfc(eta * distance) / distance**2 + 2.0 * eta * np.exp(-((eta * distance) ** 2)) / (np.sqrt(np.pi) * distance)
        ) / distance
        forces += np.einsum("ab,abk->ak", charge_products * radial, displacement)

    if reciprocal_points.size:
        g_squared = np.sum(reciprocal_points**2, axis=1)
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
        phases = positions @ reciprocal_points.T  # (atoms, G)
        # sum_b Z_b sin(G.(r_a - r_b)) = Im[exp(i G.r_a) conj(S(G))]
        structure = np.sum(charges[:, None] * np.exp(1j * phases), axis=0)
        sine_sum = np.imag(np.exp(1j * phases) * np.conj(structure)[None, :])
        forces += (4.0 * np.pi / volume) * charges[:, None] * ((sine_sum * weight[None, :]) @ reciprocal_points)

    return 2.0 * forces  # Hartree -> Rydberg


__all__ = ["ewald_ion_ion_forces", "ion_ion_forces"]
