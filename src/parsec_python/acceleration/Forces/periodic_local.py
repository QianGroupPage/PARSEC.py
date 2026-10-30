"""Local-pseudopotential and core-correction forces of a periodic cell, without a scan of the grid.

:func:`parsec_python.Forces.periodic_local.periodic_local_forces` measures the distance of every grid point to every image of
every atom and builds a grid-by-reciprocal-vector phase matrix for each atom, so a 864-atom cell is out of reach.  The same
integrals are formed here the way :func:`..V_ion.periodic_setup.fft_local_ionic_potential` forms the potential they
differentiate:

``F_ion,a = dv [ sum_T sum_(r in ball(R_a + T)) rho(r) s'(d) (r - R_a - T)/d
            + 2 Z_a (4 pi/V) sum_(G != 0) w(G) G Im( exp(-i G.R_a) rho~(G) ) ]``

with ``s(d) = V_a(d) + 2 Z erf(eta d)/d`` (short ranged: a ball around each image of the atom, as wide as the pseudopotential
table or the Gaussian tail), ``w(G) = exp(-G^2/4 eta^2)/G^2`` and ``rho~(G) = sum_r rho(r) exp(iG.r)``, the density's
discrete Fourier transform, one FFT, read at the reciprocal vectors of the series.  As in the potential, ``eta`` is the
largest one whose series ends below the Nyquist frequency of the grid, so ``rho~`` is read at frequencies the FFT holds
exactly.  The series is evaluated for all atoms at once with the separable phases of the structure factor.  The core
correction is the ball sum of the reference.

The sums are written against an array module ``xp`` (NumPy, or CuPy for a calculation on a GPU): the balls are generated
on the host from index ranges, their radial slopes, products and sums are formed with ``xp``, and only the ``(atoms, 3)``
result comes back.  The numbers are those of the reference to the accuracy of the splitting (about 1e-7 Ry/bohr).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.special import erf as _host_erf

from parsec_python.Forces import LocalForceTerms, species_force_tables
from parsec_python.Forces.periodic_local import dense_periodic_local_forces as reference_periodic_local_forces
from parsec_python.models import Atom, SpeciesPotential
from parsec_python.Pseudopotential import ParsecPseudopotential, ParsecRadialSpline
from parsec_python.Pseudopotential.radial_derivatives import tail_start_radius
from parsec_python.V_ion import ewald as _ewald
from parsec_python.V_ion import ionic_potential as _ionic

from ..V_ion.periodic_setup import _atom_balls, _cell_geometry, fast_splitting

#: Grid points whose radial slopes are evaluated together.
_POINTS_PER_CALL = 1_000_000
#: Complex numbers of one block of the reciprocal contraction.
_CONTRACTION_ELEMENTS = 4_000_000


class _Slope:
    """The radial slope of one tabulated function with the rules of ``forcloc.f90``, on the arrays of ``xp``.

    Zero closer than the first radial sample; between it and ``rs(npoint-6)`` the derivative of the padded cubic spline of
    the table, or the linear interpolation of its derivative table; from there outwards zero, or the Coulomb slope
    ``2 Z/r**2`` for the local potential (``charge``).  The same rules as
    :func:`parsec_python.Pseudopotential.radial_derivatives.evaluate_local_potential_derivative` and ``evaluate_density_derivative``.
    """

    def __init__(self, xp, potential: ParsecPseudopotential, values, *, spline_padding: int | None, table, charge=None):
        self.xp = xp
        self.first = float(potential.radii[0])
        self.tail = tail_start_radius(potential)
        self.charge = charge
        if spline_padding is not None:
            spline = ParsecRadialSpline.from_positive_grid(potential.radii, values, spline_padding)
            self.knots = xp.asarray(spline.knots)
            self.values = xp.asarray(spline.values)
            self.second = xp.asarray(spline.second_derivatives)
            self.radii = self.table = None
        else:
            self.knots = self.values = self.second = None
            self.radii = xp.asarray(np.asarray(potential.radii, dtype=float))
            self.table = xp.asarray(np.asarray(table, dtype=float))

    def __call__(self, distances):
        xp = self.xp
        result = xp.zeros_like(distances)
        inner = (distances >= self.first) & (distances < self.tail)
        sample = distances[inner]
        if self.knots is not None:
            lower = xp.clip(xp.searchsorted(self.knots, sample, side="right") - 1, 0, int(self.knots.size) - 2)
            upper = lower + 1
            step = self.knots[upper] - self.knots[lower]
            left = (self.knots[upper] - sample) / step
            right = (sample - self.knots[lower]) / step
            result[inner] = (
                (self.values[upper] - self.values[lower]) / step
                - (3.0 * left * left - 1.0) / 6.0 * step * self.second[lower]
                + (3.0 * right * right - 1.0) / 6.0 * step * self.second[upper]
            )
        else:
            result[inner] = xp.interp(sample, self.radii, self.table)
        if self.charge is not None:
            tail = distances >= self.tail
            result[tail] = 2.0 * self.charge / (distances[tail] * distances[tail])
        return result


def _erf_over_r_slope(xp, erf, charge: float, radius, eta: float):
    """``d/dr [2 Z erf(eta r)/r]``, smooth through ``r = 0`` (where it vanishes)."""

    far = radius > 1.0e-8
    r = xp.where(far, radius, 1.0)
    value = 2.0 * charge * (2.0 * eta * xp.exp(-((eta * r) ** 2)) / (math.sqrt(math.pi) * r) - erf(eta * r) / (r * r))
    return xp.where(far, value, 0.0)


def _inverse(xp, radius):
    positive = radius > 0.0
    return xp.where(positive, 1.0 / xp.where(positive, radius, 1.0), 0.0)


def _reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice_vectors):
    """``2 Z_a (4 pi/V) sum_G w(G) G Im(exp(-iG.R_a) rho~(G))`` for every atom, ``(atoms, 3)`` (no ``dv``)."""

    n, lattice, inverse = _cell_geometry(grid, lattice_vectors)
    volume = abs(float(np.linalg.det(lattice)))
    half = np.minimum(np.floor(g_cutoff * np.linalg.norm(lattice, axis=1) / (2.0 * np.pi)).astype(np.int64), (n - 1) // 2)
    frequencies = [np.arange(-m, m + 1, dtype=np.int64) for m in half]
    m0, m1, m2 = (frequencies[0][:, None, None], frequencies[1][None, :, None], frequencies[2][None, None, :])
    metric = (2.0 * np.pi) ** 2 * (inverse.T @ inverse)
    g_squared = (
        metric[0, 0] * m0**2 + metric[1, 1] * m1**2 + metric[2, 2] * m2**2
        + 2.0 * (metric[0, 1] * m0 * m1 + metric[0, 2] * m0 * m2 + metric[1, 2] * m1 * m2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.exp(-g_squared / (4.0 * eta**2)) / g_squared
    weight[g_squared == 0.0] = 0.0
    weight[g_squared > g_cutoff**2] = 0.0
    # the components of G = sum_j m_j b_j, b_j = 2 pi inverse[:, j]
    components = [
        2.0 * np.pi * (inverse[c, 0] * m0 + inverse[c, 1] * m1 + inverse[c, 2] * m2) * weight for c in range(3)
    ]

    # rho~(G) = sum_r rho(r) exp(iG.r), at the points (i + s)/n of the lattice vectors: one FFT
    shape = tuple(int(v) for v in n)
    spectrum = xp.fft.ifftn(xp.asarray(density).reshape(shape)) * int(np.prod(shape))
    index = [np.mod(frequencies[axis], n[axis]) for axis in range(3)]
    transform = spectrum[xp.asarray(index[0])[:, None, None], xp.asarray(index[1])[None, :, None], xp.asarray(index[2])[None, None, :]]
    shifts = np.asarray(grid.shift, dtype=float)
    if np.any(shifts != 0.0):
        transform = transform * xp.asarray(
            np.exp(2.0j * np.pi * (m0 * shifts[0] / n[0] + m1 * shifts[1] / n[1] + m2 * shifts[2] / n[2]))
        )
    sizes = tuple(int(2 * m + 1) for m in half)
    series = xp.stack([transform * xp.asarray(component) for component in components])  # (3, M0, M1, M2)
    series = series.reshape(3, sizes[0] * sizes[1], sizes[2])

    fractional = positions @ inverse
    forces = np.zeros((len(positions), 3))
    block = max(1, _CONTRACTION_ELEMENTS // (3 * sizes[0] * sizes[1]))
    for start in range(0, len(positions), block):
        stop = min(start + block, len(positions))
        phases = [
            xp.asarray(np.exp(-2.0j * np.pi * np.outer(frequencies[axis], fractional[start:stop, axis]))) for axis in range(3)
        ]
        along_z = xp.matmul(series, phases[2]).reshape(3, sizes[0], sizes[1], stop - start)
        along_y = xp.einsum("cija,ja->cia", along_z, phases[1])
        total = xp.einsum("cia,ia->ca", along_y, phases[0])
        forces[start:stop] = _to_host(xp, total.imag.T)
    return forces * (2.0 * charges * 4.0 * np.pi / volume)[:, None]


def _to_host(xp, array) -> np.ndarray:
    return np.asarray(xp.asnumpy(array) if hasattr(xp, "asnumpy") else array)


def fast_periodic_local_forces(
    grid,
    atoms: Sequence[Atom],
    potentials: Mapping[str, ParsecPseudopotential],
    specifications: Mapping[str, SpeciesPotential],
    lattice_vectors: np.ndarray,
    density: np.ndarray,
    xc_potential: np.ndarray,
    *,
    xp=np,
    erf=None,
    tolerance: float = 1.0e-14,
) -> LocalForceTerms:
    """The result of :func:`parsec_python.Forces.periodic_local.periodic_local_forces`, ``(atoms, 3)`` in Ry/bohr.

    ``atoms`` lie inside the cell.  ``xp`` is the array module that forms the sums (CuPy on a GPU, with
    ``erf`` from ``cupyx.scipy.special``); the periodic local potential must use the cubic spline of its table
    (:data:`parsec_python.V_ion.ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE`), otherwise the readable routine runs.
    """

    if not _ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE:
        return reference_periodic_local_forces(
            grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential, tolerance=tolerance
        )
    erf = _host_erf if erf is None and xp is np else erf
    if erf is None:
        raise ValueError("an array module other than NumPy needs its erf")
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    xc = np.asarray(xc_potential, dtype=np.float64)
    for name, field in (("density", density), ("xc_potential", xc)):
        if field.shape != (grid.size,):
            raise ValueError(f"{name} must hold one value per grid point")
    count = len(atoms)
    ionic = np.zeros((count, 3))
    core_correction = np.zeros((count, 3))
    if count == 0:
        return LocalForceTerms(ionic, np.zeros((0, 3)), core_correction)

    eta, erfc_cutoff, g_cutoff = fast_splitting(grid, lattice, tolerance)
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)
    half_width = grid.settings.stencil_half_width

    # the series of the long-range part: all atoms from one FFT
    ionic += _reciprocal_forces(xp, grid, positions, charges, density, eta, g_cutoff, lattice)

    density_x, xc_x = xp.asarray(density), xp.asarray(xc)
    tables: dict = {}
    slopes: dict = {}

    def sums(sample_rows, sample_relative, sample_distances, field, slope) -> np.ndarray:
        rows = xp.asarray(np.concatenate(sample_rows))
        relative = xp.asarray(np.concatenate(sample_relative))
        distances = xp.asarray(np.concatenate(sample_distances))
        coefficient = slope(distances) * field[rows] * _inverse(xp, distances)
        return _to_host(xp, xp.sum(coefficient[:, None] * relative, axis=0))

    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        charge = potential.ionic_charge
        position = positions[index]
        if atom.symbol not in tables:
            tables[atom.symbol] = species_force_tables(potential, specification)
            local = _Slope(
                xp, potential, potential.channel_potentials[specification.local_angular_momentum],
                spline_padding=half_width, table=None, charge=charge,
            )
            core = None
            if tables[atom.symbol].core_density_derivative is not None:
                core = _Slope(
                    xp, potential, potential.core_density,
                    spline_padding=half_width if specification.use_spline else None,
                    table=tables[atom.symbol].core_density_derivative,
                )
            slopes[atom.symbol] = (local, core)
        local_slope, core_slope = slopes[atom.symbol]

        # short-ranged part: V_a + 2 Z erf(eta r)/r over a ball around every image; beyond the Coulomb onset of the table
        # it is -2 Z erfc(eta r)/r, so the ball reaches the larger of the onset and the Gaussian tail
        reach = max(erfc_cutoff, _ionic.coulomb_onset_radius(potential, specification.local_angular_momentum))

        def short_range(distances, _slope=local_slope, _charge=charge):
            return _slope(distances) + _erf_over_r_slope(xp, erf, _charge, distances, eta)

        pending: tuple[list, list, list] = ([], [], [])
        points = 0
        for rows, distances, relative in _atom_balls(grid, position, reach, vectors=True):
            pending[0].append(rows), pending[1].append(relative), pending[2].append(distances)
            points += rows.size
            if points >= _POINTS_PER_CALL:
                ionic[index] += sums(*pending, density_x, short_range)
                pending, points = ([], [], []), 0
        if points:
            ionic[index] += sums(*pending, density_x, short_range)

        if core_slope is not None:
            support = _ionic.atomic_density_support_radius(potential, specification, core=True)
            pending, points = ([], [], []), 0
            for translation in _ionic._periodic_image_translations(lattice, support):
                rows, relative, radius = _ionic._support_finder(grid, position + translation, support)
                inside = radius < support
                if not np.any(inside):
                    continue
                pending[0].append(rows[inside]), pending[1].append(relative[inside]), pending[2].append(radius[inside])
                points += int(np.count_nonzero(inside))
                if points >= _POINTS_PER_CALL:
                    core_correction[index] += sums(*pending, xc_x, core_slope)
                    pending, points = ([], [], []), 0
            if points:
                core_correction[index] += sums(*pending, xc_x, core_slope)

    volume_element = grid.volume_element
    return LocalForceTerms(
        ionic=ionic * volume_element,
        self_consistency=np.zeros((count, 3)),
        core_correction=core_correction * volume_element,
    )


def _device_module():
    """``(xp, erf, device)`` for the forces of a calculation run through the reference drivers: CuPy on the first device of
    the k-point plan when that plan uses a GPU, else NumPy."""

    from ..SCF.kpoint_parallel import resolve_backend

    plan = resolve_backend()
    if plan is None or plan.name != "cupy":
        return np, None, None
    from cupyx.scipy.special import erf

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    return cp, erf, int(plan.device_ids[0])


def local_force_provider(
    grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential, *, tolerance: float = 1.0e-14
) -> LocalForceTerms:
    """What :func:`parsec_python.Forces.periodic_local.periodic_local_forces` hands its sums to once the acceleration
    package is imported: :func:`fast_periodic_local_forces` on the device the k-point solver uses, or on the host."""

    xp, erf, device = _device_module()
    arguments = (grid, atoms, potentials, specifications, lattice_vectors, density, xc_potential)
    if device is None:
        return fast_periodic_local_forces(*arguments, tolerance=tolerance)
    with xp.cuda.Device(device):
        return fast_periodic_local_forces(*arguments, xp=xp, erf=erf, tolerance=tolerance)


__all__ = ["fast_periodic_local_forces", "local_force_provider"]
