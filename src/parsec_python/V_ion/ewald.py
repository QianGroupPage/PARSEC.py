"""
Ewald summation for the periodic ion-ion
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf, erfc, erfcinv

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential


def _reciprocal_lattice(lattice_vectors: np.ndarray) -> np.ndarray:
    """Return reciprocal vectors ``b_i`` satisfying ``a_i . b_j = 2*pi*delta_ij``."""

    return 2.0 * np.pi * np.linalg.inv(lattice_vectors).T


def _lattice_points(
    basis: np.ndarray,
    reciprocal_basis: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Enumerate integer combinations of ``basis`` whose norm is at most ``cutoff``.

    The maximum index along direction ``i`` is bounded using the perpendicular
    spacing between lattice planes, ``2*pi/|reciprocal_basis[i]|``, which is
    always at least as generous as the true bound so no in-cutoff point is
    missed.
    """

    if cutoff <= 0.0:
        return np.zeros((1, 3), dtype=float)
    plane_spacing = 2.0 * np.pi / np.linalg.norm(reciprocal_basis, axis=1)
    max_index = np.ceil(cutoff / plane_spacing).astype(int) + 1
    ranges = [np.arange(-m, m + 1) for m in max_index]
    grid = np.meshgrid(*ranges, indexing="ij")
    integer_combinations = np.column_stack([axis.reshape(-1) for axis in grid])
    points = integer_combinations @ basis
    within_cutoff = np.linalg.norm(points, axis=1) <= cutoff
    return points[within_cutoff]


def _ewald_cutoffs(
    volume: float, eta: float | None, tolerance: float
) -> tuple[float, float, float]:
    """Return ``(eta, real_cutoff, reciprocal_cutoff)`` shared by both Ewald sums.

    Both :func:`ewald_ion_ion_energy` and :func:`ewald_local_ionic_potential`
    use this so that, called with the same ``eta``/``tolerance`` on the same
    cell (the normal way to use them together for one calculation), they
    enumerate exactly the same real- and reciprocal-space lattice points.
    """

    if eta is None:
        eta = float(np.sqrt(np.pi) / volume ** (1.0 / 3.0))
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be between 0 and 1")
    real_cutoff = float(erfcinv(tolerance) / eta)
    reciprocal_cutoff = float(2.0 * eta * np.sqrt(-np.log(tolerance)))
    return eta, real_cutoff, reciprocal_cutoff


def ewald_ion_ion_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> float:
    """Return the periodic pairwise ion-ion repulsion in Rydberg.

    Returns
    -------
    The Ewald ion-ion energy in Rydberg, for one unit cell (i.e. counting each
    ion once, not once per image).
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    volume = float(abs(np.linalg.det(lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")
    if not atoms:
        return 0.0

    positions = np.stack([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array(
        [potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64
    )

    eta, real_cutoff, reciprocal_cutoff = _ewald_cutoffs(volume, eta, tolerance)
    reciprocal_vectors = _reciprocal_lattice(lattice_vectors)

    translations = _lattice_points(lattice_vectors, reciprocal_vectors, real_cutoff)
    reciprocal_points = _lattice_points(
        reciprocal_vectors, lattice_vectors, reciprocal_cutoff
    )
    reciprocal_points = reciprocal_points[np.any(reciprocal_points != 0.0, axis=1)]

    # Real-space sum: (1/2) * sum_{a,b} sum'_R Z_a*Z_b*erfc(eta*d)/d, the
    # prime excluding the singular (a=b, R=0) term.  Each translation R
    # contributes an (n_atoms, n_atoms) block of pair separations.
    real_space_energy = 0.0
    charge_products = charges[:, None] * charges[None, :]
    for translation in translations:
        is_origin = bool(np.all(translation == 0.0))
        displacement = (
            positions[:, None, :] - positions[None, :, :] + translation[None, None, :]
        )
        distance = np.linalg.norm(displacement, axis=2)
        if is_origin:
            np.fill_diagonal(distance, np.inf)
        real_space_energy += 0.5 * np.sum(charge_products * erfc(eta * distance) / distance)

    # Reciprocal-space sum: (2*pi/V) * sum_{G!=0} exp(-G^2/(4*eta^2))/G^2 * |S(G)|^2.
    reciprocal_energy = 0.0
    if reciprocal_points.size:
        g_squared = np.sum(reciprocal_points**2, axis=1)
        phases = positions @ reciprocal_points.T
        structure_factor = np.sum(
            charges[:, None] * np.exp(1j * phases), axis=0
        )
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
        reciprocal_energy = (2.0 * np.pi / volume) * float(
            np.sum(weight * np.abs(structure_factor) ** 2)
        )

    self_energy = -(eta / np.sqrt(np.pi)) * float(np.sum(charges**2))
    neutralizing_energy = -(np.pi / (2.0 * eta**2 * volume)) * float(np.sum(charges)) ** 2

    hartree_energy = real_space_energy + reciprocal_energy + self_energy + neutralizing_energy
    # PARSEC's Rydberg convention, matching ion_ion_energy's factor of two.
    return 2.0 * hartree_energy


def _erf_over_r(charge: float, radius: np.ndarray, eta: float) -> np.ndarray:
    """Return ``2*charge*erf(eta*r)/r``, with the finite ``r=0`` limit."""

    with np.errstate(invalid="ignore", divide="ignore"):
        values = 2.0 * charge * erf(eta * radius) / radius
    at_origin = radius == 0.0
    if np.any(at_origin):
        values = np.where(at_origin, 4.0 * charge * eta / np.sqrt(np.pi), values)
    return values


def ewald_local_ionic_potential(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float | None = None,
    tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Return the periodic local ionic potential on ``grid``, in Rydberg.

    This applies the same real-space/reciprocal-space Ewald split as
    :func:`ewald_ion_ion_energy` to that tail.  For one atom of charge
    ``Z_a``, decompose

    ``V_a(r) = [V_a(r) + 2*Z_a*erf(eta*r)/r]  +  [-2*Z_a*erf(eta*r)/r]``.

    The first bracket equals ``V_a(r)`` near the atom (where ``V_a`` is the
    tabulated pseudopotential) and equals ``-2*Z_a*erfc(eta*r)/r`` beyond the
    table (where ``V_a(r) = -2*Z_a/r`` exactly) -- either way it decays like a
    Gaussian tail, so summing it over periodic images converges with just a
    handful of nearby image shells, the same real-space cutoff construction
    used by ``ewald_ion_ion_energy``.

    The second bracket, summed over every lattice translation, is the
    periodic Coulomb potential of a periodized Gaussian charge distribution.
    By the standard Ewald/Poisson-summation identity, its ``G != 0``
    reciprocal-space Fourier series is

    ``sum_R (-2*Z_a*erf(eta*|r-R|)/|r-R|)
        = -2*Z_a * (4*pi/V_cell) * sum_{G!=0} exp(-G**2/(4*eta**2))/G**2
              * cos(G.(r-r_a))  +  (a G=0 constant, per unit charge)``.

    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice_vectors.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    volume = float(abs(np.linalg.det(lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")

    total = np.zeros(grid.size, dtype=np.float64)
    if not atoms:
        return total

    eta, real_cutoff, reciprocal_cutoff = _ewald_cutoffs(volume, eta, tolerance)
    reciprocal_vectors = _reciprocal_lattice(lattice_vectors)
    translations = _lattice_points(lattice_vectors, reciprocal_vectors, real_cutoff)
    reciprocal_points = _lattice_points(
        reciprocal_vectors, lattice_vectors, reciprocal_cutoff
    )
    reciprocal_points = reciprocal_points[np.any(reciprocal_points != 0.0, axis=1)]
    g_squared = (
        np.sum(reciprocal_points**2, axis=1) if reciprocal_points.size else None
    )
    reciprocal_weight = (
        np.exp(-g_squared / (4.0 * eta**2)) / g_squared
        if g_squared is not None
        else None
    )

    for atom in atoms:
        specification = specifications[atom.symbol]
        potential = potentials[atom.symbol]
        charge = potential.ionic_charge
        position = np.asarray(atom.position, dtype=np.float64)

        # Short-range piece: sum over the few nearby periodic images.
        for translation in translations:
            displaced = position + translation
            radius = np.linalg.norm(grid.coordinates - displaced, axis=1)
            total += potential.local_potential(
                radius,
                specification.local_angular_momentum,
                use_spline=specification.use_spline,
                spline_padding_width=grid.settings.stencil_half_width,
            )
            total += _erf_over_r(charge, radius, eta)

        # Long-range piece: reciprocal-space sum, G=0 dropped (see docstring).
        if reciprocal_points.size:
            displacement = grid.coordinates - position
            phase = displacement @ reciprocal_points.T
            total += -2.0 * charge * (4.0 * np.pi / volume) * (
                np.cos(phase) @ reciprocal_weight
            )

    return total


__all__ = ["ewald_ion_ion_energy", "ewald_local_ionic_potential"]
