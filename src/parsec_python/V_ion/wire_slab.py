"""Local ionic potential and ion-ion terms of a wire or slab system.

Local potential (``ionpot_wire.f90``, ``ionpot_slab.f90``).  Each atom of
charge ``Z`` carries a Gaussian charge of width ``sigma = 0.5*rcore``
(``sigmafac``, ``rcore`` the core radius of the pseudopotential),

``n_g(r) = Z/(pi^(3/2) sigma^3) exp(-d^2/sigma^2)``,

whose Rydberg potential ``2 Z erf(d/sigma)/d`` cancels the Coulomb tail of
the local pseudopotential.  The sum

``V_short(r) = sum_(a, images) [V_loc,a(d) + 2 Z_a erf(d/sigma_a)/d]``

is short ranged and is summed over a few cell images; the potential ``V_G``
of all Gaussian charges is the solution of the Poisson equation of the wire
or slab (:mod:`parsec_python.Hartree.confined`), and
``V_ion = V_short - V_G``.  The radial potential is interpolated as for a
cluster (:meth:`parsec_python.Pseudopotential.ParsecPseudopotential.local_potential`:
linear in ``r*V``, ``V(r_1)`` below the first sample, ``-2Z/r`` beyond the
table).  Deviation: ``ionpot_slab.f90`` ignores ``Cubic_Spline``; here a
species with it is interpolated by the spline in a slab as in a wire.

Ion-ion energy of a wire (``forceion_wire.f90``): the pairs ``j < j'`` of the
cell, each with

``E = (8 Z Z'/a) [ln(2a/rho)/2 - gamma/2 + sum_(n>=1) cos(n k x) K_0(n k rho)]``,

``k = 2 pi/a``, ``rho`` and ``x`` the perpendicular and axial separations.
The images of an atom itself contribute nothing in this convention (that of
the ``ln 2 - gamma`` constant of the wire Hartree boundary).  For two atoms
on one line parallel to the axis Fortran switches to a Gaussian-regularised
series (``eta = 1e4``); here the exact limit
``E = -(2 Z Z'/a) [psi(t) + psi(1-t) + 2 gamma]``, ``t = x/a mod 1``, is used.

Ion-ion energy of a slab (``ewald_slab.f90``): the 2-D Ewald sum of Parry
(Surf. Sci. 49, 433 (1975); Ono and Hirose, Appendix A) over ordered pairs,

``E = E_G + E_0 + E_R + E_self`` (Ry),

``E_G = (pi/A) sum_(G != 0) sum_ij Z_i Z_j f+(|G|, z_ij) cos(G.r_ij)/|G|``,
``f+-(G, z) = e^(-G z) erfc(G/(2 eta) - eta z) +- e^(G z) erfc(G/(2 eta) + eta z)``,
``E_0 = -(2 sqrt(pi)/A) sum_ij Z_i Z_j [exp(-eta^2 z_ij^2)/eta + sqrt(pi) z_ij erf(eta z_ij)]``,
``E_R = sum_ij Z_i Z_j sum'_P erfc(eta |P + r_ij|)/|P + r_ij|``,
``E_self = -sum_i Z_i^2 2 eta/sqrt(pi)``.

Fortran sums ``E_G`` over the first 1000 vectors of its FFT box and ``E_R``
over ``eta |.| < 6.5``; here both sums run to the same accuracy of the
terms (``erfc`` arguments up to 6.5), which the totals reach to ~1e-13 Ry.
There is no alpha (``G = 0``) energy for a slab (``inipbc.f90:206-209``)
nor for a wire (``totnrg.f90:151``).

Charged sheets (``ionpot_slab.f90:307-338``, ``force_charged_sh.f90``): a
sheet of charge ``q`` (per surface cell) at height ``z_s`` adds
``4 pi (q/A) |z - z_s|`` to the potential of the electrons and

``E = sum_s [ sum_a -4 pi Z_a q_s/A |z_a - z_s|
              + sum_(s' != s) -2 pi q_s q_s'/A |z_s - z_s'| ]``

to the ion-ion energy.  Its force on an ion is
``F_z = sum_s 4 pi Z q_s/A sign(z_a - z_s)``; Fortran has ``2 Z q_s/A``
there (``force_charged_sh.f90:79``), without the ``2 pi`` of the potential.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf, erfc, erfcx, k0, k1, polygamma, psi

from ..models import Atom, ChargedSheet, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential

#: ``sigmafac`` of ``const.f90``: Gaussian width per core radius.
GAUSSIAN_WIDTH_FACTOR = 0.5
#: The Euler-Mascheroni constant (``forceion_wire.f90``).
EULER_GAMMA = 0.57721566490153286
#: Fortran's on-axis threshold of ``forceion_wire.f90`` (bohr).
AXIS_TOLERANCE = 2.5e-5
#: Ewald parameter of ``ewald_slab.f90`` (1/bohr).
SLAB_EWALD_ETA = 0.5
#: Largest ``erfc`` argument kept by both sums of the slab Ewald energy.
_EWALD_ARGUMENT = 6.5


def fortran_core_radius(potential: ParsecPseudopotential) -> float:
    """``p_pot%rcore`` of ``pseudo.f90`` (bohr).

    The ``Core_Cutoff_Radius`` of the input for a file that takes it, else
    the first tabulated radius beyond the largest channel cutoff
    (``pseudo.f90:505-522``).
    """

    if potential.core_radius is not None:
        return float(potential.core_radius)
    requested = max(potential.channel_cutoffs.values(), default=potential.radii[0])
    index = int(np.searchsorted(potential.radii, requested, side="right"))
    index = min(index, potential.radii.size - 1)
    return float(potential.radii[index])


def gaussian_width(potential: ParsecPseudopotential) -> float:
    """``sigma = sigmafac * rcore`` of the compensating Gaussian (bohr)."""

    return GAUSSIAN_WIDTH_FACTOR * fortran_core_radius(potential)


@dataclass(frozen=True)
class ShortRangeLocalPotential:
    """``V_short`` (Ry) and the Gaussian charge density ``n_g`` (e/bohr**3).

    ``nominal_charge`` is ``sum_a Z_a`` and ``grid_charge`` the integral of
    ``n_g`` on the grid (Fortran's charge neutrality check).
    """

    potential: np.ndarray
    gaussian_density: np.ndarray
    nominal_charge: float
    grid_charge: float


def short_range_local_potential(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
) -> ShortRangeLocalPotential:
    """``V_short`` and ``n_g`` of a wire or slab grid (``ionpot_wire.f90:110-189``).

    The images of every atom are Fortran's ``|m_i| <= nint(6 sigma/|a_i|) + 1``
    (:meth:`parsec_python.Grid.confined.ConfinedPeriodicGrid.compensation_image_translations`).
    """

    total = np.zeros(grid.size, dtype=np.float64)
    gaussian = np.zeros(grid.size, dtype=np.float64)
    nominal = 0.0
    for atom in atoms:
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        charge = float(potential.ionic_charge)
        sigma = gaussian_width(potential)
        nominal += charge
        density_factor = charge / sigma**3 / np.sqrt(np.pi) / np.pi
        position = np.asarray(atom.position, dtype=np.float64)
        for translation in grid.compensation_image_translations(position, sigma):
            distance = np.linalg.norm(grid.coordinates - (position + translation), axis=1)
            total += potential.local_potential(
                distance,
                specification.local_angular_momentum,
                use_spline=specification.use_spline,
                spline_padding_width=grid.settings.stencil_half_width,
            )
            ratio = distance / sigma
            compensation = np.empty_like(distance)
            nonzero = distance != 0.0
            compensation[nonzero] = 2.0 * charge * erf(ratio[nonzero]) / distance[nonzero]
            compensation[~nonzero] = 2.0 / np.sqrt(np.pi) * 2.0 * charge / sigma
            total += compensation
            gaussian += density_factor * np.exp(-ratio * ratio)
    return ShortRangeLocalPotential(
        potential=total,
        gaussian_density=gaussian,
        nominal_charge=nominal,
        grid_charge=float(np.sum(gaussian) * grid.volume_element),
    )


def charged_sheet_potential(grid, sheets: Sequence[ChargedSheet]) -> np.ndarray:
    """``sum_s 4 pi (q_s/A) |z - z_s|`` on a slab grid (Ry, ``ionpot_slab.f90:307-338``)."""

    total = np.zeros(grid.size, dtype=np.float64)
    if not sheets:
        return total
    area = grid.surface_area
    z = grid.coordinates[:, 2]
    for sheet in sheets:
        total += 2.0 * np.pi * 2.0 * (sheet.charge / area) * np.abs(z - sheet.height)
    return total


def _charges(atoms, potentials) -> np.ndarray:
    return np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)


def _positions(atoms) -> np.ndarray:
    return np.array([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------
# Wire
# --------------------------------------------------------------------------


def _axial_pair(separation: float, period: float) -> tuple[float, float]:
    """Energy series and ``d/dx`` of it for two ions on one axial line.

    Returns ``S`` and ``dS/dx`` of ``S(x) = sum_m 1/|x + m a|`` in the
    convention of the wire (the ``rho -> 0`` limit of the ``K_0`` series):
    ``S = -(1/a) [psi(t) + psi(1-t) + 2 gamma]``, ``t = x/a mod 1``.
    """

    t = (separation / period) % 1.0
    if t == 0.0:
        raise ValueError("two wire ions occupy the same position up to a period")
    value = -(psi(t) + psi(1.0 - t) + 2.0 * EULER_GAMMA) / period
    slope = -(polygamma(1, t) - polygamma(1, 1.0 - t)) / period**2
    return float(value), float(slope)


def wire_pair_energy_and_force(
    displacement: np.ndarray, period: float, charge_product: float
) -> tuple[float, np.ndarray]:
    """Rydberg energy of one ion pair of a wire and the force on the first ion.

    ``displacement = R_j - R_j'`` (bohr); the force on ``j'`` is minus the
    returned one.  Translation of the pair loop of ``forceion_wire.f90``.
    """

    displacement = np.asarray(displacement, dtype=np.float64)
    prefactor = 8.0 * charge_product / period
    axial = float(displacement[0])
    perpendicular = float(np.hypot(displacement[1], displacement[2]))
    if perpendicular < AXIS_TOLERANCE:
        value, slope = _axial_pair(axial, period)
        # E = 2 Z Z' S(x): (8 Z Z'/a) * (a/4) S.
        energy = 2.0 * charge_product * value
        force = np.array([-2.0 * charge_product * slope, 0.0, 0.0])
        return energy, force
    wave = 2.0 * np.pi / period
    argument0 = wave * perpendicular
    # K_0(x) < 1e-20 beyond x ~ 44: the series of forceion_wire.f90 stops there.
    count = int(np.ceil(48.0 / argument0)) + 1
    n = np.arange(1, count + 1, dtype=np.float64)
    argument = n * argument0
    bessel0 = k0(argument)
    bessel1 = k1(argument)
    cosine = np.cos(n * wave * axial)
    sine = np.sin(n * wave * axial)
    series = np.log(2.0 * period / perpendicular) / 2.0 - EULER_GAMMA / 2.0 + np.sum(cosine * bessel0)
    energy = prefactor * float(series)
    axial_force = prefactor * float(np.sum(n * wave * sine * bessel0))
    radial_force = prefactor * float(0.5 / perpendicular + np.sum(n * wave * cosine * bessel1))
    direction = np.array([0.0, displacement[1], displacement[2]]) / perpendicular
    return energy, np.array([axial_force, 0.0, 0.0]) + radial_force * direction


def wire_ion_ion_energy_and_forces(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    period: float,
) -> tuple[float, np.ndarray]:
    """Ion-ion energy (Ry) and forces (Ry/bohr, ``(atoms, 3)``) of a wire."""

    positions = _positions(atoms)
    charges = _charges(atoms, potentials)
    count = len(atoms)
    forces = np.zeros((count, 3), dtype=np.float64)
    energy = 0.0
    for first in range(count):
        for second in range(first):
            pair, force = wire_pair_energy_and_force(
                positions[first] - positions[second],
                period,
                charges[first] * charges[second],
            )
            energy += pair
            forces[first] += force
            forces[second] -= force
    return energy, forces


# --------------------------------------------------------------------------
# Slab
# --------------------------------------------------------------------------


def _f_plus_minus(length: np.ndarray, height: np.ndarray, eta: float):
    """``f+`` and ``f-`` of ``ewald_slab.f90`` (``fplus``, ``fminus``), stable.

    ``f+`` is even and ``f-`` odd in ``height``; the growing exponential is
    combined with its ``erfc`` through ``erfcx``.
    """

    magnitude = np.abs(height)
    first = np.exp(-length * magnitude) * erfc(length / (2.0 * eta) - eta * magnitude)
    argument = length / (2.0 * eta) + eta * magnitude
    second = np.exp(-(length / (2.0 * eta)) ** 2 - (eta * magnitude) ** 2) * erfcx(argument)
    plus = first + second
    minus = np.sign(height) * (first - second)
    return plus, minus


def _slab_reciprocal_vectors(reciprocal: np.ndarray, cutoff: float) -> np.ndarray:
    """In-plane reciprocal vectors ``l b1 + m b2`` with ``0 < |G| <= cutoff``."""

    b1, b2 = reciprocal[0], reciprocal[1]
    area = abs(b1[0] * b2[1] - b1[1] * b2[0])
    # Distance between the lines of the reciprocal lattice.
    span1 = area / np.linalg.norm(b2)
    span2 = area / np.linalg.norm(b1)
    limit1 = int(np.ceil(cutoff / span1)) + 1
    limit2 = int(np.ceil(cutoff / span2)) + 1
    l, m = np.meshgrid(np.arange(-limit1, limit1 + 1), np.arange(-limit2, limit2 + 1), indexing="ij")
    vectors = l.reshape(-1, 1) * b1[None, :] + m.reshape(-1, 1) * b2[None, :]
    lengths = np.linalg.norm(vectors, axis=1)
    keep = (lengths > 0.0) & (lengths <= cutoff)
    return vectors[keep]


def _slab_real_translations(lattice: np.ndarray, cutoff: float, span: float) -> np.ndarray:
    a1, a2 = lattice[0], lattice[1]
    area = abs(a1[0] * a2[1] - a1[1] * a2[0])
    reach = cutoff + span
    limit1 = int(np.ceil(reach / (area / np.linalg.norm(a2)))) + 1
    limit2 = int(np.ceil(reach / (area / np.linalg.norm(a1)))) + 1
    l, m = np.meshgrid(np.arange(-limit1, limit1 + 1), np.arange(-limit2, limit2 + 1), indexing="ij")
    return l.reshape(-1, 1) * a1[None, :] + m.reshape(-1, 1) * a2[None, :]


@dataclass(frozen=True)
class SlabEwaldTerms:
    """The four parts of the slab Ewald energy (Ry) and the total force."""

    reciprocal: float
    zero: float
    real: float
    self_energy: float
    forces: np.ndarray

    @property
    def energy(self) -> float:
        return self.reciprocal + self.zero + self.real + self.self_energy


def slab_ewald(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    lattice_vectors: np.ndarray,
    *,
    eta: float = SLAB_EWALD_ETA,
) -> SlabEwaldTerms:
    """Parry's 2-D Ewald energy and forces of the ions of a slab.

    ``lattice_vectors`` is the ``latt_vec`` of the slab (rows; ``a1``, ``a2``
    in the ``xy`` plane).  The result does not depend on ``eta``.
    """

    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    a1, a2 = lattice[0], lattice[1]
    area = abs(a1[0] * a2[1] - a1[1] * a2[0])
    positions = _positions(atoms)
    charges = _charges(atoms, potentials)
    count = len(atoms)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    # erfc(G/(2 eta)) and erfc(eta r) are dropped below exp(-6.5**2).
    vectors = _slab_reciprocal_vectors(reciprocal, 2.0 * eta * _EWALD_ARGUMENT)
    lengths = np.linalg.norm(vectors, axis=1)

    products = charges[:, None] * charges[None, :]
    # Ordered pairs (i, j): separations r_j - r_i.
    separation = positions[None, :, :] - positions[:, None, :]
    heights = separation[:, :, 2]

    forces = np.zeros((count, 3), dtype=np.float64)
    reciprocal_energy = 0.0
    for first in range(count):
        for second in range(count):
            plus, minus = _f_plus_minus(lengths, heights[first, second], eta)
            phase = vectors[:, :2] @ separation[first, second, :2]
            weight = products[first, second] * np.pi / area
            reciprocal_energy += weight * float(np.sum(plus * np.cos(phase) / lengths))
            # d/dr_i of cos(G.(r_j - r_i)) = G sin(.); df+/dz = -G f-, z = z_j - z_i.
            in_plane = weight * (np.sin(phase) * plus / lengths) @ vectors[:, :2]
            vertical = weight * float(np.sum(np.cos(phase) * minus))
            # F_i = -dE/dr_i: the pair (i, j) and the pair (j, i) are equal.
            forces[first, :2] -= 2.0 * in_plane
            forces[first, 2] -= 2.0 * vertical

    zero_energy = 0.0
    for first in range(count):
        for second in range(count):
            height = heights[first, second]
            weight = products[first, second]
            zero_energy += weight * (
                np.exp(-(eta * height) ** 2) / eta + np.sqrt(np.pi) * height * erf(eta * height)
            )
            # d/dz_i of [e^(-eta^2 z^2)/eta + sqrt(pi) z erf(eta z)] = -sqrt(pi) erf(eta z).
            forces[first, 2] -= 2.0 * (-2.0 * np.sqrt(np.pi) / area) * weight * (
                -np.sqrt(np.pi) * erf(eta * height)
            )
    zero_energy *= -2.0 * np.sqrt(np.pi) / area

    cutoff = _EWALD_ARGUMENT / eta
    span = float(np.max(np.linalg.norm(separation[:, :, :2], axis=2), initial=0.0))
    translations = _slab_real_translations(lattice, cutoff, span)
    real_energy = 0.0
    for first in range(count):
        for second in range(count):
            vector = separation[first, second][None, :] + translations
            distance = np.linalg.norm(vector, axis=1)
            keep = (distance > 0.0) & (eta * distance < _EWALD_ARGUMENT)
            distance = distance[keep]
            vector = vector[keep]
            weight = products[first, second]
            term = erfc(eta * distance) / distance
            real_energy += weight * float(np.sum(term))
            # d/dr_i of erfc(eta d)/d with d = |r_j - r_i + P|.
            slope = (term + 2.0 * eta / np.sqrt(np.pi) * np.exp(-(eta * distance) ** 2)) / distance**2
            forces[first] -= 2.0 * weight * (slope @ vector)

    self_energy = -float(np.sum(charges**2)) * 2.0 * eta / np.sqrt(np.pi)
    return SlabEwaldTerms(
        reciprocal=float(reciprocal_energy),
        zero=float(zero_energy),
        real=float(real_energy),
        self_energy=self_energy,
        forces=forces,
    )


def charged_sheet_ion_energy_and_forces(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    sheets: Sequence[ChargedSheet],
    area: float,
    *,
    fortran_force: bool = False,
) -> tuple[float, np.ndarray]:
    """Ion-sheet and sheet-sheet energy (Ry) and the sheet forces on the ions.

    ``fortran_force=True`` reproduces ``force_charged_sh.f90``'s force
    ``2 Z q/A sign(z_a - z_s)`` (missing the ``2 pi`` of its own energy).
    """

    positions = _positions(atoms)
    charges = _charges(atoms, potentials)
    forces = np.zeros((len(atoms), 3), dtype=np.float64)
    energy = 0.0
    for index, sheet in enumerate(sheets):
        density = sheet.charge / area
        offset = positions[:, 2] - sheet.height
        energy -= float(np.sum(2.0 * np.pi * 2.0 * charges * density * np.abs(offset)))
        strength = 2.0 if fortran_force else 4.0 * np.pi
        forces[:, 2] += strength * charges * density * np.sign(offset)
        for other_index, other in enumerate(sheets):
            if other_index != index:
                energy -= 2.0 * np.pi * other.charge * density * abs(other.height - sheet.height)
    return energy, forces


__all__ = [
    "AXIS_TOLERANCE",
    "EULER_GAMMA",
    "GAUSSIAN_WIDTH_FACTOR",
    "SLAB_EWALD_ETA",
    "ShortRangeLocalPotential",
    "SlabEwaldTerms",
    "charged_sheet_ion_energy_and_forces",
    "charged_sheet_potential",
    "fortran_core_radius",
    "gaussian_width",
    "short_range_local_potential",
    "slab_ewald",
    "wire_ion_ion_energy_and_forces",
    "wire_pair_energy_and_force",
]
