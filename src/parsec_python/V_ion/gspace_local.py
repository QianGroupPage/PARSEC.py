"""Fortran PARSEC's reciprocal-space local pseudopotential of a 3-D periodic cell.

PARSEC builds the local ionic potential of a bulk cell in G space and
transforms it to the grid (``ionpbc.f90`` -> ``v_first.f90`` ->
``pot_local.f90``):

``V(r) = sum_(G != 0) V(G) exp(i G.r)``,
``V(G) = (1/Omega) sum_a v_a(|G|) exp(-i G.R_a)``,

``v(q) = 4 pi int_0^inf r (r V_loc(r) + 2 Z) j_0(q r) dr - 8 pi Z/q^2``
(Rydberg times bohr^3; ``pbc%vloc = v/2`` in Hartree, ``fourier_f.f90``
l.282-300), over the G vectors of the FFT box ``|m_i| <= kmax_i``,
``kmax = nint((n + 1/4)/2) = n/2`` (both ``+n/2`` and ``-n/2`` of an even
grid enter, on the same FFT point: ``pot_local.f90`` adds them), and only up
to the end of the q table, ``nint(q/dq + 2) < nq = int(2 pi/h_1/dq) + 4``
(``dq = 0.015``, ``v_first.f90`` l.62-70: "v_first contains information
about local potential only up to glmax").  ``G = 0`` is zero
(``ionpbc.f90``: ``vionz(1) = 0``): the average of the potential is the
alpha term, which enters the total energy as

``E_alpha = (N_e/Omega) sum_a alpha_a``, ``alpha_a = 4 pi int r (r V_loc + 2 Z) dr``

(``fourier_f.f90`` l.313-321, ``inipbc.f90`` l.204, ``totnrg.f90`` l.151).
Fortran takes ``alpha_a`` from the first two points of its q table,
``2 pi^2 (fvion(2) - fvion(1))/dq^2 = 4 pi int r (r V + 2Z) j_0(dq r) dr``,
which differs from the ``q -> 0`` limit by about ``-(4 pi dq^2/6) int r^3 (r V
+ 2Z) dr`` (1e-4 Ry for two Si atoms): :func:`alpha_coefficient` gives both.

Deviations from Fortran (the reasons are in the functions):

* ``v(|G|)`` is evaluated at every distinct ``|G|`` with the quadrature of
  ``fourier_1d`` (the five-point Newton-Cotes rule in the index of the
  logarithmic radial grid), not interpolated quadratically from the table
  on ``q = 0.015 (j - 1)`` (``v_first.f90`` l.73-83).  The validation of the
  port (its report) found the interpolation worth a few
  microrydberg per atom.
* An odd number of points along a direction (possible on this port's
  orthorhombic grid, never on Fortran's) takes ``|m| <= (n - 1)/2``, the
  distinct frequencies, instead of Fortran's ``nint((n + 1/4)/2) = (n + 1)/2``,
  which would add the alias of ``-(n - 1)/2``.
* ``N_e`` (the electron count) multiplies the alpha term where Fortran
  multiplies the total ionic charge (the same for a neutral cell).

The forces are the exact derivative of ``dV sum_r rho(r) V(r)`` with this
potential (:func:`gspace_local_forces`), the formula of ``force_loc_pb.f90``
without its star average.  Numerical routines only: nothing is printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..models import Atom, SpeciesPotential
from ..Pseudopotential import ParsecPseudopotential
from ..Pseudopotential.fourier_filter import DELTA_Q, spherical_bessel

#: The local-potential constructions of a 3-D periodic cell
#: (``PeriodicGridSettings.local_potential``): the real-space Ewald split of
#: :func:`parsec_python.V_ion.ewald_local_ionic_potential` (default) and
#: Fortran's G-space potential of this module.
LOCAL_POTENTIAL_METHODS = ("real_space", "g_space")

#: How the G-space potential's alpha term is computed
#: (``PeriodicGridSettings.alpha_term``): the ``q -> 0`` limit (default) or
#: Fortran's finite difference of its q table.
ALPHA_TERM_METHODS = ("analytic", "finite_difference")

# Number of |G| values whose radial transforms are evaluated together (the
# Bessel matrix has this many rows times the radial points).
_TRANSFORM_CHUNK = 2048


def _transform_weights(potential: ParsecPseudopotential, local_l: int) -> tuple[np.ndarray, np.ndarray]:
    """``(r, w)`` with ``int_0^inf r (r V + 2Z) f(r) dr ~= sum_i w_i f(r_i)``.

    The rule of ``fourier_1d`` (``fourier_f.f90`` l.375-390): Fortran's
    radial index 1 is the origin, panels ``(7, 32, 12, 32, 7) * 2/45`` start at
    every fourth index, ``dr/di = b (r + c)``, and the tail beyond the table
    (where ``r V = -2Z``) adds nothing.
    """

    radii = np.concatenate(([0.0], np.asarray(potential.radii, dtype=np.float64)))
    table = np.concatenate(
        ([0.0], np.asarray(potential.channel_potentials[local_l], dtype=np.float64))
    )
    charge = 2.0 * float(potential.ionic_charge)
    slope = (radii + potential.radial_offset) * potential.radial_parameter_b
    integrand = slope * radii * (radii * table + charge)
    integrand[0] = 0.0
    weights = np.zeros(radii.size + 4)
    for start in range(0, radii.size, 4):
        weights[start : start + 5] += (7.0, 32.0, 12.0, 32.0, 7.0)
    return radii, weights[: radii.size] * integrand * (2.0 / 45.0)


def local_form_factor(
    potential: ParsecPseudopotential, local_l: int, q: np.ndarray
) -> np.ndarray:
    """``v(q) = 4 pi int r (r V_loc + 2Z) j_0(q r) dr - 8 pi Z/q^2`` for ``q > 0``.

    Rydberg times bohr^3: ``2 pbc%vloc`` of ``fourier_f.f90`` (``vloc = fvion
    pi^2/q^2``, ``fvion = (2/pi)(q^2 int - 2Z)``), at the given ``q`` (1/bohr)
    instead of on the table.  ``V_loc`` is the channel ``local_l`` of the
    file, without Fourier filter (Fortran transforms the unfiltered
    ``vion``).
    """

    q = np.asarray(q, dtype=np.float64)
    if np.any(q <= 0.0):
        raise ValueError("the local form factor is defined for q > 0 (G = 0 is the alpha term)")
    radii, weights = _transform_weights(potential, local_l)
    charge = float(potential.ionic_charge)
    result = np.empty(q.shape, dtype=np.float64)
    flat_q = q.reshape(-1)
    flat = result.reshape(-1)
    for start in range(0, flat_q.size, _TRANSFORM_CHUNK):
        block = flat_q[start : start + _TRANSFORM_CHUNK]
        bessel = spherical_bessel(0, np.outer(block, radii))
        bessel[:, 0] = 0.0
        flat[start : start + block.size] = 4.0 * np.pi * (bessel @ weights) - (
            8.0 * np.pi * charge / (block * block)
        )
    return result


def alpha_coefficient(
    potential: ParsecPseudopotential, local_l: int, method: str = "analytic"
) -> float:
    """``alpha = lim_(q->0) [v(q) + 8 pi Z/q^2]`` of one atom, Ry bohr^3.

    ``"analytic"``: ``4 pi int r (r V_loc + 2Z) dr`` with the quadrature of
    :func:`local_form_factor`.  ``"finite_difference"``: Fortran's
    ``2 pi^2 (fvion(2) - fvion(1))/dq^2`` (``fourier_f.f90`` l.318-319) with
    ``fvion(1) = -4Z/pi`` and ``fvion(2)`` the transform at ``q = dq =
    0.015``, i.e. ``4 pi int r (r V + 2Z) j_0(dq r) dr``.
    """

    if method not in ALPHA_TERM_METHODS:
        raise ValueError("the alpha term is " + " or ".join(ALPHA_TERM_METHODS))
    radii, weights = _transform_weights(potential, local_l)
    if method == "analytic":
        return float(4.0 * np.pi * np.sum(weights[1:]))
    charge = float(potential.ionic_charge)
    bessel = spherical_bessel(0, DELTA_Q * radii)
    bessel[0] = 0.0
    # fourier_1d: vql = (2 q^2 sum/45 - vt) 2/pi, here with the 2/45 in w.
    second = (DELTA_Q * DELTA_Q * float(bessel @ weights) - 2.0 * charge) * 2.0 / np.pi
    first = -4.0 * charge / np.pi
    return float(2.0 * np.pi * np.pi * (second - first) / DELTA_Q**2)


def fortran_qtable_size(spacing: float) -> int:
    """``pbc%nq = int(2 pi/step/delq) + 4`` of ``fourier_f.f90`` l.283."""

    return int(2.0 * np.pi / float(spacing) / DELTA_Q) + 4


def _first_spacing(grid) -> float:
    """``grid%step(1)``: the step along the first lattice direction."""

    spacings = getattr(grid, "spacings", None)
    if spacings is not None:
        return float(np.asarray(spacings)[0])
    return float(grid.spacing)


@dataclass(frozen=True)
class GSpaceVectors:
    """The G vectors of the local potential on one grid.

    ``integers`` are ``m`` (``G = m @ B``, ``B`` the rows ``b_i`` of the
    reciprocal lattice), ``vectors`` the Cartesian ``G`` (1/bohr), ``norms``
    ``|G|``, ``phases`` ``exp(2 pi i m.s/n)`` of the grid shift ``s`` (grid
    points ``r = sum_i (k_i + s_i)/n_i a_i``), ``fft_index`` ``m mod n``.
    ``G = 0`` and the vectors beyond the q table are not in the set.
    """

    integers: np.ndarray
    vectors: np.ndarray
    norms: np.ndarray
    phases: np.ndarray
    fft_index: tuple[np.ndarray, np.ndarray, np.ndarray]
    shape: tuple[int, int, int]


def gspace_vectors(grid, lattice_vectors: np.ndarray) -> GSpaceVectors:
    """Fortran's G set of the local potential (box and q-table end) on ``grid``.

    ``grid`` is a 3-D periodic grid of this port (orthorhombic: points
    ``k h`` from the origin; general: ``A_n (k + shift) h``): in both the
    point ``k`` sits at the fractional coordinates ``(k + s)/n`` of
    ``lattice_vectors``, so ``G.r = 2 pi m.(k + s)/n``.
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    shape = tuple(int(value) for value in grid.points_per_axis)
    if len(shape) != 3:
        raise ValueError("the G-space local potential needs a 3-D periodic grid")
    shift = np.asarray(getattr(grid, "shift", np.zeros(3)), dtype=np.float64)
    half = [count // 2 for count in shape]
    mesh = np.meshgrid(*[np.arange(-m, m + 1) for m in half], indexing="ij")
    integers = np.column_stack([axis.reshape(-1) for axis in mesh]).astype(np.int64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice_vectors).T
    vectors = integers.astype(np.float64) @ reciprocal
    norms = np.sqrt(np.einsum("ij,ij->i", vectors, vectors))
    # v_first.f90: n = int(q/delq + 2 + 1/2) must stay below nq.
    table_index = np.floor(norms / DELTA_Q + 2.0 + 0.5)
    keep = (norms > 0.0) & (table_index < fortran_qtable_size(_first_spacing(grid)))
    integers = integers[keep]
    vectors = vectors[keep]
    norms = norms[keep]
    counts = np.asarray(shape, dtype=np.float64)
    phases = np.exp(2j * np.pi * (integers * (shift / counts)).sum(axis=1))
    wrapped = integers % np.asarray(shape, dtype=np.int64)
    return GSpaceVectors(
        integers=integers,
        vectors=vectors,
        norms=norms,
        phases=phases,
        fft_index=(wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]),
        shape=shape,
    )


def _species_form_factors(
    vectors: GSpaceVectors,
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    symbols: Sequence[str],
) -> dict[str, np.ndarray]:
    """``v_a(|G|)`` of every species, evaluated once per distinct ``|G|``."""

    distinct, inverse = np.unique(vectors.norms, return_inverse=True)
    return {
        symbol: local_form_factor(
            potentials[symbol], specifications[symbol].local_angular_momentum, distinct
        )[inverse]
        for symbol in symbols
    }


def _structure_factor(vectors: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """``sum_a exp(-i G.R_a)`` per G, in blocks of atoms."""

    total = np.zeros(vectors.shape[0], dtype=np.complex128)
    block = max(1, 4_000_000 // max(vectors.shape[0], 1))
    for start in range(0, positions.shape[0], block):
        phases = vectors @ positions[start : start + block].T
        total += np.exp(-1j * phases).sum(axis=1)
    return total


def gspace_local_ionic_potential(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
) -> np.ndarray:
    """Fortran's G-space local potential on ``grid`` (Rydberg), zero average.

    ``ionpbc.f90``: ``V(G)`` of :func:`gspace_vectors` (``v_first``), placed
    on the FFT box with the phase of the grid shift and transformed
    (``pot_local``), times 2 for Rydberg (here ``v`` is in Rydberg already).
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    volume = float(abs(np.linalg.det(lattice_vectors)))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")
    total = np.zeros(grid.size, dtype=np.float64)
    if not atoms:
        return total
    vectors = gspace_vectors(grid, lattice_vectors)
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    factors = _species_form_factors(vectors, potentials, specifications, symbols)
    coefficients = np.zeros(vectors.norms.size, dtype=np.complex128)
    for symbol in symbols:
        positions = np.array(
            [np.asarray(atom.position, dtype=np.float64) for atom in atoms if atom.symbol == symbol]
        )
        coefficients += factors[symbol] * _structure_factor(vectors.vectors, positions)
    coefficients *= vectors.phases / volume
    box = np.zeros(vectors.shape, dtype=np.complex128)
    np.add.at(box, vectors.fft_index, coefficients)
    values = np.fft.ifftn(box) * box.size
    return np.ascontiguousarray(values.real).reshape(-1)


def gspace_alpha_energy(
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    electron_count: float,
    *,
    method: str = "analytic",
) -> float:
    """``E_alpha = (N_e/Omega) sum_a alpha_a`` in Rydberg (``pbc%ealpha``)."""

    volume = float(abs(np.linalg.det(np.asarray(lattice_vectors, dtype=np.float64))))
    if volume <= 0.0:
        raise ValueError("lattice_vectors must span a nondegenerate cell")
    coefficients: dict[str, float] = {}
    total = 0.0
    for atom in atoms:
        if atom.symbol not in coefficients:
            coefficients[atom.symbol] = alpha_coefficient(
                potentials[atom.symbol],
                specifications[atom.symbol].local_angular_momentum,
                method,
            )
        total += coefficients[atom.symbol]
    return float(electron_count) * total / volume


def gspace_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """``-d/dR_a [dV sum_r rho(r) V(r)]`` for :func:`gspace_local_ionic_potential`.

    ``F_a = -(dV/Omega) sum_G v_a(|G|) G Im(P(G) exp(-i G.R_a))`` with
    ``P(G) = sum_r rho(r) exp(i G.r)`` (one FFT of the density; the G set and
    the form factors do not depend on the positions, so this is exact).
    ``force_loc_pb.f90`` evaluates the same sum over stars with the
    star-averaged density.  Ry/bohr, shape ``(atoms, 3)``.
    """

    lattice_vectors = np.asarray(lattice_vectors, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    forces = np.zeros((len(atoms), 3), dtype=np.float64)
    if not atoms:
        return forces
    if density.shape != (grid.size,):
        raise ValueError("density must hold one value per grid point")
    volume = float(abs(np.linalg.det(lattice_vectors)))
    vectors = gspace_vectors(grid, lattice_vectors)
    symbols = list(dict.fromkeys(atom.symbol for atom in atoms))
    factors = _species_form_factors(vectors, potentials, specifications, symbols)
    transformed = np.fft.ifftn(density.reshape(vectors.shape)) * density.size
    weighted = transformed[vectors.fft_index] * vectors.phases
    scale = grid.volume_element / volume
    for index, atom in enumerate(atoms):
        position = np.asarray(atom.position, dtype=np.float64)
        imaginary = np.imag(weighted * np.exp(-1j * (vectors.vectors @ position)))
        forces[index] = -scale * ((factors[atom.symbol] * imaginary) @ vectors.vectors)
    return forces


def periodic_local_ionic_potential(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
) -> np.ndarray:
    """The local potential of a 3-D cell as ``grid.settings.local_potential`` asks.

    ``"real_space"``: :func:`parsec_python.V_ion.ewald_local_ionic_potential`
    (the default, unchanged); ``"g_space"``: :func:`gspace_local_ionic_potential`.
    """

    if getattr(grid.settings, "local_potential", "real_space") == "g_space":
        return gspace_local_ionic_potential(
            grid, atoms, potentials, specifications, lattice_vectors
        )
    from .ewald import ewald_local_ionic_potential

    return ewald_local_ionic_potential(
        grid, atoms, potentials, specifications, lattice_vectors
    )


def periodic_alpha_energy(
    grid_settings,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    electron_count: float,
) -> float:
    """The alpha term that belongs to :func:`periodic_local_ionic_potential`.

    The real-space split: :func:`parsec_python.V_ion.ewald_alpha_z_energy`
    (its potential carries the rest of the average); G space:
    :func:`gspace_alpha_energy` with ``grid_settings.alpha_term``.
    """

    if getattr(grid_settings, "local_potential", "real_space") == "g_space":
        return gspace_alpha_energy(
            atoms,
            potentials,
            specifications,
            lattice_vectors,
            electron_count,
            method=getattr(grid_settings, "alpha_term", "analytic"),
        )
    from .ewald import ewald_alpha_z_energy

    return ewald_alpha_z_energy(atoms, potentials, lattice_vectors, electron_count)


__all__ = [
    "ALPHA_TERM_METHODS",
    "GSpaceVectors",
    "LOCAL_POTENTIAL_METHODS",
    "alpha_coefficient",
    "fortran_qtable_size",
    "gspace_alpha_energy",
    "gspace_local_forces",
    "gspace_local_ionic_potential",
    "gspace_vectors",
    "local_form_factor",
    "periodic_alpha_energy",
    "periodic_local_ionic_potential",
]
