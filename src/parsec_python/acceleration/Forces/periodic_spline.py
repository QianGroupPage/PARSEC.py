"""Local forces of a periodic cell whose species use ``Cubic_Spline`` (Fortran-derived routines), on the host.

:func:`parsec_python.Forces.periodic.periodic_local_forces` differentiates the converged Ewald sum term by
term over every grid point of every image within ``erfcinv(1e-26)/eta`` of the reference ``eta``
(``eta = sqrt(pi)/V^(1/3)``: 43 bohr for an 8-atom silicon cube, 86 bohr for a 64-atom one), so its cost
is ``atoms x grid points x images``, minutes for 8 atoms and hours for 64.  The device kernel of
:mod:`.periodic_device` replaces it for linearly interpolated species only.  For the cells whose local
potential :func:`parsec_python.acceleration.V_ion.periodic_setup_fortran.fft_local_ionic_potential`
builds with the cubic spline of a table (the orthorhombic grid of the Fortran-derived routines), the
forces here are the derivatives of that potential and of the densities of the run, with its split and
its reach:

``F_ion,a = dV sum_T sum_(r in ball_T) rho(r) s'(d) (r - R_a - T)/d + F_long,a``,
``s(d) = V_a(d) + 2 Z_a erf(eta d)/d``

over the balls of :func:`~parsec_python.acceleration.V_ion.periodic_setup_fortran._atom_balls` of the
radius that the potential uses (``_local_reach``: the erfc tail of the fast ``eta`` or the Coulomb tail of
the table), ``s'`` the derivative of the spline that
:meth:`~parsec_python.Pseudopotential.ParsecPseudopotential.local_potential` evaluates
(:func:`parsec_python.Forces.periodic.local_potential_slope`), and ``F_long`` the force of the Fourier
series of the potential (one FFT of the density, the formula of the device forces evaluated with
NumPy).  The self-consistency term (``V_out - V_in`` times the slope of the atomic valence density) and
the core correction (``V_xc`` times the slope of the core density) are the terms of the readable
routine over the balls of the same radii (``_density_reach``: 1e-16 of the peak of the density) instead
of every grid point of every image that reaches them, summed by the CUDA kernel of :mod:`.periodic_device`
where a device is usable (the valence balls reach 30 bohr for silicon: most of the work), else here.

The general (oblique) grid, where the fast preparation builds its potential
(:func:`parsec_python.acceleration.V_ion.periodic_setup_fortran.oblique_fast`), takes the same terms over the
lattice balls of :func:`parsec_python.acceleration.V_ion.periodic_setup._atom_balls`, the long-ranged term of
:func:`parsec_python.acceleration.Forces.periodic_local._reciprocal_forces` (any lattice), for every species; its
density terms run in the lattice kernel of :func:`.periodic_device.device_lattice_density_forces` where a device is
usable (round-off equal), else over the lattice balls here.

The readable sum also adds, between the reach of the potential and its 43-86 bohr, the error of the
spline of ``V`` around ``-2 Z/r`` that the potential leaves out (Ry bohr^3 per atom near the end of a
table, see :func:`parsec_python.V_ion.ionic_potential.coulomb_tail_radius`); its gradient is what the two
forces differ by (2.6e-7 Ry/bohr for two Si atoms in a 6-bohr cube, the test of
:mod:`parsec_python.acceleration.tests.test_periodic_coulomb_tail`).  Numerical routine: nothing is printed.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import erfcinv

from parsec_python.Forces import LocalForceTerms
from parsec_python.Forces.periodic import (
    _density_reach,
    _density_slope,
    _erf_over_r_slope_over_r,
    _valence_density_table,
    local_potential_slope,
)
from parsec_python.Pseudopotential import ParsecRadialSpline

from ..V_ion import periodic_setup_fortran as _setup
from .periodic_device import _long_range_forces

#: Grid points whose radial slopes are evaluated together.
_POINTS_PER_CALL = 1_000_000


def takes(grid, lattice_vectors, specifications, atoms) -> bool:
    """Whether the cell's potential is the ball sum of the fast preparation that these forces differentiate: the
    orthorhombic grid with a ``Cubic_Spline`` species (the others take the device kernel or the readable sums), or
    the general grid where the fast preparation builds it (:func:`..V_ion.periodic_setup_fortran.oblique_fast`)."""

    symbols = dict.fromkeys(atom.symbol for atom in atoms)
    if getattr(grid.settings, "local_potential", "real_space") != "real_space":
        return False
    if _setup.oblique_fast(grid, specifications, atoms):
        return True
    return bool(
        _setup._fft_handles(grid, lattice_vectors)
        and any(getattr(specifications[symbol], "use_spline", False) for symbol in symbols)
    )


def _ball_sum(grid, position, radius, field, slope_over_distance, balls=None) -> np.ndarray:
    """``sum_T sum_(r in ball_T) field(r) s'(d)/d (r - R - T)`` (``d > 0``) over the balls of radius ``radius``;
    ``slope_over_distance(d)`` is ``s'(d)/d``; ``balls`` the generator of the balls (the orthorhombic one of
    :mod:`..V_ion.periodic_setup_fortran` by default, the lattice one of :mod:`..V_ion.periodic_setup` for the
    general grid)."""

    total = np.zeros(3)
    pending: tuple[list, list, list] = ([], [], [])
    points = 0

    def flush() -> None:
        nonlocal pending, points
        rows = np.concatenate(pending[0])
        distances = np.concatenate(pending[1])
        relative = np.concatenate(pending[2])
        nonzero = distances > 0.0
        weight = np.zeros_like(distances)
        weight[nonzero] = field[rows[nonzero]] * slope_over_distance(distances[nonzero])
        total[:] += weight @ relative
        pending, points = ([], [], []), 0

    if radius <= 0.0:
        return total
    for rows, distances, relative in (_setup._atom_balls if balls is None else balls)(
        grid, position, radius, vectors=True
    ):
        pending[0].append(rows), pending[1].append(distances), pending[2].append(relative)
        points += rows.size
        if points >= _POINTS_PER_CALL:
            flush()
    if points:
        flush()
    return total


def _device_density_terms(grid, atoms, potentials, specifications, lattice_vectors, fields, tolerance):
    """``(terms, host_core)``: the self-consistency and core terms summed by the CUDA kernel of
    :mod:`.periodic_device` (its linear slopes of the valence and core tables, the ones of the readable
    routine), the core term left to the host for the species whose core density is a spline (``host_core``);
    ``None`` without a usable device."""

    from ..V_ion import periodic_device as _fields
    from .periodic_device import _device_local_forces

    cp = _fields._cupy()
    if cp is None or not _fields.handles(grid, lattice_vectors):
        return None
    host_core = frozenset(
        symbol for symbol in dict.fromkeys(atom.symbol for atom in atoms) if specifications[symbol].use_spline
    )
    with cp.cuda.Device(_fields.device_id(cp)):
        terms = _device_local_forces(
            cp, grid, atoms, potentials, specifications, lattice_vectors, *fields, tolerance,
            ionic=False, host_core=host_core,
        )
    return terms, host_core


def _device_lattice_density_terms(grid, atoms, potentials, specifications, lattice_vectors, fields):
    """The density terms of an oblique cell from :func:`.periodic_device.device_lattice_density_forces`, as
    :func:`_device_density_terms` returns them, or ``None`` without a usable device."""

    from .periodic_device import device_lattice_density_forces

    host_core = frozenset(
        symbol for symbol in dict.fromkeys(atom.symbol for atom in atoms) if specifications[symbol].use_spline
    )
    found = device_lattice_density_forces(
        grid, atoms, potentials, specifications, lattice_vectors, fields[1], fields[2], host_core=host_core
    )
    if found is None:
        return None
    count = len(atoms)
    return LocalForceTerms(np.zeros((count, 3)), found[0], found[1]), host_core


def spline_periodic_local_forces(
    grid,
    atoms: Sequence,
    potentials: Mapping,
    specifications: Mapping,
    lattice_vectors,
    density,
    potential_residual,
    xc_potential,
    *,
    tolerance: float = 1.0e-14,
) -> LocalForceTerms | None:
    """The three local terms of :func:`parsec_python.Forces.periodic.periodic_local_forces` (Ry/bohr) for the
    potential of the fast preparation; ``None`` where that potential is not the spline ball sum (:func:`takes`).

    With a usable device the density terms run in the CUDA kernel of :mod:`.periodic_device` (their balls reach
    30 bohr for silicon: most of the work), else on the host.
    """

    if not takes(grid, lattice_vectors, specifications, atoms):
        return None
    fields = tuple(
        np.ascontiguousarray(values, dtype=np.float64) for values in (density, potential_residual, xc_potential)
    )
    if any(values.shape != (grid.size,) for values in fields):
        raise ValueError("the fields of the local forces need one value per grid point")
    rho, residual, xc = fields
    count = len(atoms)
    ionic = np.zeros((count, 3))
    self_consistency = np.zeros((count, 3))
    core_correction = np.zeros((count, 3))
    if count == 0:
        return LocalForceTerms(ionic, self_consistency, core_correction)

    # The split of fft_local_ionic_potential: its eta and the reach of every species.
    oblique = _setup.oblique_fast(grid, specifications, atoms)
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    if oblique:
        from ..V_ion import periodic_setup as _default

        eta, erfc_cutoff, g_cutoff = _default.fast_splitting(grid, lattice, tolerance)
        balls = _default._atom_balls
    else:
        n, sides, _ = _setup._axis_spacing(grid)
        nyquist = np.pi / float(np.max(sides / n))
        eta = _setup._NYQUIST_FRACTION * nyquist / (2.0 * np.sqrt(-np.log(tolerance)))
        erfc_cutoff = float(erfcinv(tolerance) / eta)
        balls = None
    rule = _setup.reach_rule()
    padding = grid.settings.stencil_half_width
    residual_present = bool(np.any(residual != 0.0))
    xc_present = bool(np.any(xc != 0.0))
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    charges = np.array([potentials[atom.symbol].ionic_charge for atom in atoms], dtype=np.float64)

    device = None
    if oblique:
        device = _device_lattice_density_terms(grid, atoms, potentials, specifications, lattice_vectors, fields)
    else:
        device = _device_density_terms(grid, atoms, potentials, specifications, lattice_vectors, fields, tolerance)
    host_core = None if device is None else device[1]
    species: dict = {}
    for index, atom in enumerate(atoms):
        potential = potentials[atom.symbol]
        specification = specifications[atom.symbol]
        if atom.symbol not in species:
            local_l = specification.local_angular_momentum
            valence = _valence_density_table(potential, specification)
            core_spline = None
            if potential.has_nonlinear_core_correction and specification.use_spline:
                core_spline = ParsecRadialSpline.from_positive_grid(potential.radii, potential.core_density, padding)

            def local_slope(distances, _p=potential, _l=local_l, _s=specification, _z=float(potential.ionic_charge)):
                # s'(d)/d of the terms V_a(d) + 2 Z erf(eta d)/d that fft_local_ionic_potential sums
                return local_potential_slope(
                    _p, distances, _l, use_spline=_s.use_spline, spline_padding_width=padding
                ) / distances + _erf_over_r_slope_over_r(_z, distances, eta)

            species[atom.symbol] = {
                "reach": _setup._local_reach(potential, specification, eta, erfc_cutoff, rule),
                "local": local_slope,
                "valence_reach": _density_reach(potential, valence),
                "valence": lambda distances, _p=potential, _t=valence: _density_slope(_p, distances, _t) / distances,
                "core_reach": _density_reach(potential, potential.core_density)
                if potential.has_nonlinear_core_correction
                else 0.0,
                "core": lambda distances, _p=potential, _c=core_spline: _density_slope(
                    _p, distances, _p.core_density, spline=_c
                )
                / distances,
            }
        terms = species[atom.symbol]
        position = positions[index]
        ionic[index] = _ball_sum(grid, position, terms["reach"], rho, terms["local"], balls)
        if residual_present and device is None:
            self_consistency[index] = _ball_sum(
                grid, position, terms["valence_reach"], residual, terms["valence"], balls
            )
        if (
            xc_present
            and potential.has_nonlinear_core_correction
            and (host_core is None or atom.symbol in host_core)
        ):
            core_correction[index] = _ball_sum(grid, position, terms["core_reach"], xc, terms["core"], balls)

    volume_element = grid.volume_element
    if oblique:
        from .periodic_local import _reciprocal_forces

        long_range = _reciprocal_forces(np, grid, positions, charges, rho, eta, g_cutoff, lattice) * volume_element
    else:
        long_range = _long_range_forces(np, grid, positions, charges, eta, tolerance, rho)
    ionic = ionic * volume_element + long_range
    self_consistency = self_consistency * volume_element
    core_correction = core_correction * volume_element
    if device is not None:
        self_consistency = device[0].self_consistency
        core_correction = core_correction + device[0].core_correction
    return LocalForceTerms(ionic=ionic, self_consistency=self_consistency, core_correction=core_correction)


__all__ = ["spline_periodic_local_forces", "takes"]
