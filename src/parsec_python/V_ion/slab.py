"""
Ionic terms of a slab: ion-ion energy and local ionic potential for a cell periodic in the xy plane only.

Both use the open-z Coulomb kernel of the slab Hartree solver (:mod:`parsec_python.Hartree.slab`), whose planar
part is ``-2 pi sigma |z - z'|``, so the electrostatics of a neutral slab does not depend on the height of the box.

``slab_ewald_energy``
    the 2D Ewald sum (Parry; Ono and Hirose, as the Fortran ``ewald_slab``): real-space sum over the in-plane
    images, a reciprocal sum over in-plane ``G != 0`` with the z dependence ``exp(+-G z) erfc(G/2 eta +- eta z)``, and
    the ``G = 0`` sheet term ``-(pi/A)[z erf(eta z) + exp(-eta^2 z^2)/(eta sqrt(pi))]``.

``slab_local_ionic_potential``
    the Fortran ``ionpot_slab``: every atom (and its in-plane images) contributes ``V_a(r) + 2 Z erf(r/sigma)/r``,
    which is short ranged, and the potential of the compensating Gaussian charges (Z, width ``sigma = rcore/2``) is
    removed by solving the slab Poisson problem for them on the same grid.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf, erfc, erfcx

from ..Grid.pbc import SlabRealSpaceGrid
from ..models import Atom, HartreeSettings, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential

_SIGMA_FACTOR = 0.5  # Fortran ``sigmafac``: sigma = sigmafac * rcore
_GAUSSIAN_REACH = 6.0  # sigmas


def _in_plane_images(lattice: np.ndarray, reach: float, anchor: np.ndarray, radius: float) -> np.ndarray:
    """In-plane translations ``T`` such that a ball of radius ``reach`` around ``anchor_atom + T`` can touch the
    cell: the images with ``|anchor + T - center| <= reach + radius``, ``center``/``radius`` the circumscribed
    circle of the in-plane cell."""
    a1, a2 = lattice[0], lattice[1]
    plane_spacing = 1.0 / np.linalg.norm(np.linalg.inv(lattice[:2, :2]), axis=0)
    bound = np.ceil((reach + 2.0 * radius) / plane_spacing).astype(int) + 1
    i, j = np.meshgrid(np.arange(-bound[0], bound[0] + 1), np.arange(-bound[1], bound[1] + 1), indexing="ij")
    translations = np.zeros((i.size, 3))
    translations[:, :] = i.reshape(-1, 1) * a1 + j.reshape(-1, 1) * a2
    return translations


def _cell_center_radius(lattice: np.ndarray) -> tuple[np.ndarray, float]:
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float) @ lattice[:2, :2]
    center = 0.5 * (lattice[0, :2] + lattice[1, :2])
    return center, float(np.max(np.linalg.norm(corners - center, axis=1)))


def slab_ewald_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float = 0.5,
    tolerance: float = 1.0e-14,
) -> float:
    """Ion-ion energy (Rydberg) of point charges ``Z`` in a cell periodic in x and y."""
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    if not atoms:
        return 0.0
    positions = np.stack([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    in_plane = lattice[:2, :2]
    area = float(abs(np.linalg.det(in_plane)))
    reciprocal = 2.0 * np.pi * np.linalg.inv(in_plane).T  # rows b1, b2
    inverse_cell = np.linalg.inv(in_plane)
    # separations between atoms: in-plane part reduced to the cell, z as it is
    delta = positions[:, None, :] - positions[None, :, :]
    z_diff = delta[:, :, 2]
    q_products = charges[:, None] * charges[None, :]

    # real space: 1/2 sum_{ij} q_i q_j sum_R' erfc(eta r)/r
    cutoff = float(np.sqrt(-np.log(tolerance))) / eta
    plane_spacing = 1.0 / np.linalg.norm(inverse_cell, axis=0)
    pair_reach = float(np.max(np.linalg.norm(delta[:, :, :2], axis=2))) if len(atoms) > 1 else 0.0
    bound = np.ceil((cutoff + pair_reach) / plane_spacing).astype(int) + 1
    real_space = 0.0
    for n1 in range(-bound[0], bound[0] + 1):
        for n2 in range(-bound[1], bound[1] + 1):
            shift = n1 * in_plane[0] + n2 * in_plane[1]
            separation = delta.copy()
            separation[:, :, :2] += shift
            distance = np.linalg.norm(separation, axis=2)
            if n1 == 0 and n2 == 0:
                np.fill_diagonal(distance, np.inf)
            if float(np.min(distance)) > cutoff:
                continue
            real_space += 0.5 * float(np.sum(q_products * erfc(eta * distance) / distance))

    # reciprocal space (G != 0 of the plane), the z dependence in closed form
    g_max = 2.0 * eta * float(np.sqrt(-np.log(tolerance)))
    limits = np.ceil(g_max / np.linalg.norm(reciprocal, axis=1)).astype(int) + 1
    reciprocal_sum = 0.0
    for m1 in range(-limits[0], limits[0] + 1):
        for m2 in range(-limits[1], limits[1] + 1):
            if m1 == 0 and m2 == 0:
                continue
            g_vec = m1 * reciprocal[0] + m2 * reciprocal[1]
            g = float(np.linalg.norm(g_vec))
            if g > g_max:
                continue
            phase = delta[:, :, 0] * g_vec[0] + delta[:, :, 1] * g_vec[1]
            half = g / (2.0 * eta)
            damp = np.exp(-(half**2) - (eta * z_diff) ** 2)  # exp(+-G z) erfc(half +- eta z) = damp * erfcx(half +- eta z)
            f = damp * (erfcx(half + eta * z_diff) + erfcx(half - eta * z_diff))
            reciprocal_sum += float(np.sum(q_products * f * np.cos(phase))) / g
    reciprocal_sum *= np.pi / (2.0 * area)

    # G = 0 : the planar-average (sheet) term
    sheet = -(np.pi / area) * float(
        np.sum(q_products * (z_diff * erf(eta * z_diff) + np.exp(-((eta * z_diff) ** 2)) / (eta * np.sqrt(np.pi))))
    )
    self_energy = -(eta / np.sqrt(np.pi)) * float(np.sum(charges**2))
    return 2.0 * (real_space + reciprocal_sum + sheet + self_energy)  # Hartree -> Rydberg


def _rcore(potential: ParsecPseudopotential) -> float:
    if potential.core_radius is not None:
        return float(potential.core_radius)
    return float(max(potential.channel_cutoffs.values(), default=potential.radii[0]))


def slab_local_ionic_potential(
    grid: SlabRealSpaceGrid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    negative_laplacian,
    hartree_settings: HartreeSettings,
) -> np.ndarray:
    """Local ionic potential (Rydberg) of the atoms of a slab: short-ranged sums minus the Poisson solution of the
    compensating Gaussians."""
    from ..Hartree.slab import solve_slab_hartree
    from .ewald import _erf_over_r

    lattice = grid.cell.lattice_vectors
    center, radius = _cell_center_radius(lattice)
    total = np.zeros(grid.size, dtype=np.float64)
    gaussian_density = np.zeros(grid.size, dtype=np.float64)
    for atom in atoms:
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        charge = potential.ionic_charge
        sigma = _SIGMA_FACTOR * _rcore(potential)
        reach = _GAUSSIAN_REACH * sigma
        position = np.asarray(atom.position, dtype=np.float64)
        candidates = _in_plane_images(lattice, reach, position, radius)
        near = np.linalg.norm((position + candidates)[:, :2] - center, axis=1) <= reach + radius
        for translation in candidates[near]:
            displaced = position + translation
            distance = np.linalg.norm(grid.coordinates - displaced, axis=1)
            total += potential.local_potential(
                distance,
                specification.local_angular_momentum,
                use_spline=specification.use_spline,
                spline_padding_width=grid.settings.stencil_half_width,
            )
            total += _erf_over_r(charge, distance, 1.0 / sigma)
            gaussian_density += charge / (sigma**3 * np.pi**1.5) * np.exp(-((distance / sigma) ** 2))
    gaussian_potential = solve_slab_hartree(gaussian_density, grid, negative_laplacian, hartree_settings).potential
    return total - gaussian_potential


__all__ = ["slab_ewald_energy", "slab_local_ionic_potential"]
