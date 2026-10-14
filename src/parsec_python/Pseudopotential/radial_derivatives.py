"""Radial derivatives of PARSEC pseudopotential tables, as forces use them.

``pseudo.f90`` differentiates four radial tables once per species: the local
potential, the atomic valence density, the frozen core density and the
Kleinman-Bylander radial projector.  ``forcloc.f90`` and ``nonloc.F90`` then
interpolate those derivative tables at the atom-grid distances.  This module
translates both steps.

The radial grid of a ``MARTINS_NEW`` file is logarithmic,

``r_j = a*exp(b*(j-1)) - a``,  ``j = 2, ..., npoint``,

so a derivative with respect to the index ``j`` becomes a radial derivative
through ``dj/dr = 1/(b*(r_j + a))``.  PARSEC takes the index derivative with a
central stencil of nine points on each side (``mor = 9`` in ``pseudo.f90``)
whose weights ``fornberg.f90`` stores with ten digits; the same ten digits
are used here.  Fortran indices ``j <= 1`` are padding that continues the
first tabulated value, and file sample ``p`` (zero based) is Fortran index
``p + 2``.

Every table returned here is aligned with ``potential.radii``.
"""

from __future__ import annotations

import numpy as np

from .potre import ParsecPseudopotential
from .radial_quadrature import parsec_radial_integral
from .radial_spline import ParsecRadialSpline


#: ``mor`` of ``pseudo.f90``: stencil points on each side.
DERIVATIVE_HALF_WIDTH = 9

#: ``coe(1:9)`` of ``fornberg(1, 9, coe)``, with the digits PARSEC stores.
#: ``coe(-k) = -coe(k)`` and ``coe(0) = 0``.
FIRST_DERIVATIVE_WEIGHTS = np.array(
    [
        0.9000000000e00,
        -0.3272727273e00,
        0.1272727273e00,
        -0.4405594406e-01,
        0.1258741259e-01,
        -0.2797202797e-02,
        0.4495504496e-03,
        -0.4627725216e-04,
        0.2285296403e-05,
    ]
)

#: Local-potential derivatives at radii up to this one are set to zero.
_LOCAL_DERIVATIVE_NOISE_RADIUS = 3.0e-4

#: ``forcloc.f90`` switches to the analytic tail at ``rs(npoint-6)``, which is
#: this index of the positive radial samples.
_TAIL_START_INDEX = -7


def _stencil_range(count: int) -> tuple[int, int]:
    """First and one-past-last file sample with a complete PARSEC stencil.

    Fortran indices ``mor <= j <= npoint - mor`` with ``npoint = count + 1``
    and sample ``p = j - 2``.
    """

    return DERIVATIVE_HALF_WIDTH - 2, count - DERIVATIVE_HALF_WIDTH


def log_grid_derivative(
    potential: ParsecPseudopotential,
    values: np.ndarray,
    *,
    first: int | None = None,
) -> np.ndarray:
    """Differentiate one radial table as ``pseudo.f90`` does.

    The two padding entries below the first sample continue ``values[0]``.
    Samples without a complete stencil are returned as zero; every caller
    applies its own rule there.  ``first`` moves the first differentiated
    sample upwards (the projector tables start at Fortran index ``2*mor``).
    """

    table = np.asarray(values, dtype=float)
    radii = potential.radii
    if table.shape != radii.shape:
        raise ValueError("values must be aligned with the radial grid")
    start, stop = _stencil_range(table.size)
    if first is not None:
        start = max(start, int(first))
    result = np.zeros_like(table)
    if stop <= start:
        return result
    # Fortran indices 0 and 1 hold the continued first value.
    padded = np.concatenate((np.full(2, table[0]), table))
    centre = np.arange(start, stop) + 2
    accumulated = np.zeros(centre.size)
    for offset, weight in enumerate(FIRST_DERIVATIVE_WEIGHTS, start=1):
        accumulated += weight * (padded[centre + offset] - padded[centre - offset])
    result[start:stop] = accumulated / (
        potential.radial_parameter_b
        * (radii[start:stop] + potential.radial_parameter_a)
    )
    return result


def local_potential_derivative_table(
    potential: ParsecPseudopotential, local_l: int
) -> np.ndarray:
    """``dvion`` of ``pseudo.f90``: dV_local/dr in Ry/bohr on the radial grid.

    The last nine samples take the Coulomb value ``2*Z/r**2`` and samples at
    radii up to ``3e-4`` bohr are zero, where the numerical derivative is
    noise.
    """

    potential.validate_local_channel(local_l)
    radii = potential.radii
    table = log_grid_derivative(potential, potential.channel_potentials[local_l])
    _, stop = _stencil_range(radii.size)
    table[stop:] = 2.0 * potential.ionic_charge / (radii[stop:] * radii[stop:])
    table[radii <= _LOCAL_DERIVATIVE_NOISE_RADIUS] = 0.0
    return table


def density_derivative_table(
    potential: ParsecPseudopotential, density: np.ndarray
) -> np.ndarray:
    """``drhodr``/``ddenc`` of ``pseudo.f90`` for a volume-density table.

    The density is continued by its first value below the grid and its
    derivative is zero where the stencil is incomplete.
    """

    return log_grid_derivative(potential, density)


def valence_density_table(
    potential: ParsecPseudopotential, *, read_from_file: bool
) -> np.ndarray:
    """Spherical atomic valence density ``rho_r`` in electrons/bohr**3.

    Without ``Read_Vcd`` PARSEC rebuilds it from the reduced radial functions
    and their occupations, ``sum_l f_l*u_l(r)**2/(4*pi*r**2)``.
    """

    if read_from_file:
        return np.asarray(potential.valence_density, dtype=float)
    radii = potential.radii
    table = np.zeros_like(radii)
    for angular_momentum, wavefunction in potential.radial_wavefunctions.items():
        table += (
            potential.channel_occupations.get(angular_momentum, 0.0)
            * wavefunction
            * wavefunction
            / (4.0 * np.pi * radii * radii)
        )
    return table


def radial_projector_with_derivative(
    potential: ParsecPseudopotential, angular_momentum: int, local_l: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalized KB radial projector, its radial derivative and its sign.

    The first array and the sign are those of
    :meth:`ParsecPseudopotential.radial_projector`.  The derivative is
    ``dvw`` of ``pseudo.f90``: the stencil acts on the projector before its
    inner samples are replaced, the samples up to Fortran index ``2*mor``
    take the derivative at ``2*mor``, and the last nine are zero.
    """

    projector, sign = potential.radial_projector(angular_momentum, local_l)
    delta_v = (
        potential.channel_potentials[angular_momentum]
        - potential.channel_potentials[local_l]
    )
    radial_wave = potential.radial_wavefunctions[angular_momentum]
    denominator = parsec_radial_integral(
        potential.radii, radial_wave * radial_wave * delta_v
    )
    raw = delta_v * radial_wave / potential.radii / np.sqrt(abs(denominator))
    constant_through = 2 * DERIVATIVE_HALF_WIDTH - 2
    derivative = log_grid_derivative(potential, raw, first=constant_through)
    if derivative.size > constant_through:
        derivative[:constant_through] = derivative[constant_through]
    return projector, derivative, sign


def tail_start_radius(potential: ParsecPseudopotential) -> float:
    """Radius from which ``forcloc.f90`` uses the analytic Coulomb tail."""

    return float(potential.radii[_TAIL_START_INDEX])


def spline_derivative(spline: ParsecRadialSpline, points: np.ndarray) -> np.ndarray:
    """First derivative ``y1`` of PARSEC's ``splint`` at ``points``."""

    queries = np.asarray(points, dtype=float).reshape(-1)
    lower = np.searchsorted(spline.knots, queries, side="right") - 1
    lower = np.clip(lower, 0, spline.knots.size - 2)
    upper = lower + 1
    step = spline.knots[upper] - spline.knots[lower]
    left_weight = (spline.knots[upper] - queries) / step
    right_weight = (queries - spline.knots[lower]) / step
    return (
        (spline.values[upper] - spline.values[lower]) / step
        - (3.0 * left_weight * left_weight - 1.0)
        / 6.0
        * step
        * spline.second_derivatives[lower]
        + (3.0 * right_weight * right_weight - 1.0)
        / 6.0
        * step
        * spline.second_derivatives[upper]
    )


def evaluate_local_potential_derivative(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    local_l: int,
    *,
    table: np.ndarray | None = None,
    use_spline: bool = False,
    spline_padding_width: int | None = None,
) -> np.ndarray:
    """dV_local/dr at ``distances`` with the rules of ``forcloc.f90``.

    Points closer than the first radial sample return zero: ``forcloc.f90``
    leaves them out of every force term.  From ``rs(npoint-6)`` outwards the
    value is the Coulomb derivative ``2*Z/r**2``.  In between the default is
    PARSEC's linear interpolation of ``r**2 * dV/dr``; with ``use_spline`` it
    is the derivative of the padded cubic spline of the potential.
    """

    radius = np.asarray(distances, dtype=float)
    radii = potential.radii
    result = np.zeros_like(radius)
    tail = radius >= tail_start_radius(potential)
    result[tail] = 2.0 * potential.ionic_charge / (radius[tail] * radius[tail])
    inner = (radius >= radii[0]) & ~tail
    if not np.any(inner):
        return result
    sample = radius[inner]
    if use_spline:
        if spline_padding_width is None:
            raise ValueError("spline_padding_width is required when use_spline=True")
        spline = ParsecRadialSpline.from_positive_grid(
            radii, potential.channel_potentials[local_l], spline_padding_width
        )
        result[inner] = spline_derivative(spline, sample)
        return result
    if table is None:
        table = local_potential_derivative_table(potential, local_l)
    result[inner] = np.interp(sample, radii, table * radii * radii) / (sample * sample)
    return result


def evaluate_density_derivative(
    potential: ParsecPseudopotential,
    distances: np.ndarray,
    density: np.ndarray,
    *,
    table: np.ndarray | None = None,
    use_spline: bool = False,
    spline_padding_width: int | None = None,
) -> np.ndarray:
    """d(rho_atom)/dr at ``distances`` with the rules of ``forcloc.f90``.

    Zero closer than the first radial sample and from ``rs(npoint-6)``
    outwards; linear interpolation of the derivative table in between, or
    the derivative of the padded cubic spline of ``density``.
    """

    radius = np.asarray(distances, dtype=float)
    radii = potential.radii
    result = np.zeros_like(radius)
    inner = (radius >= radii[0]) & (radius < tail_start_radius(potential))
    if not np.any(inner):
        return result
    sample = radius[inner]
    if use_spline:
        if spline_padding_width is None:
            raise ValueError("spline_padding_width is required when use_spline=True")
        spline = ParsecRadialSpline.from_positive_grid(
            radii, density, spline_padding_width
        )
        result[inner] = spline_derivative(spline, sample)
        return result
    if table is None:
        table = density_derivative_table(potential, density)
    result[inner] = np.interp(sample, radii, table)
    return result


__all__ = [
    "DERIVATIVE_HALF_WIDTH",
    "FIRST_DERIVATIVE_WEIGHTS",
    "density_derivative_table",
    "evaluate_density_derivative",
    "evaluate_local_potential_derivative",
    "local_potential_derivative_table",
    "log_grid_derivative",
    "radial_projector_with_derivative",
    "spline_derivative",
    "tail_start_radius",
    "valence_density_table",
]
