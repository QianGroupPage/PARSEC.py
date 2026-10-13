"""Reader of PARSEC's old Martins pseudopotential format (``*_POTR.DAT``).

``Pseudopotential_Format: martins_old`` selects the ``MARTINS_OLD`` branches
of ``pseudo.f90``.  Such a file is read with list-directed statements and
holds::

    ns  lll  icore                   radial samples, channels, core flag
    nloc(1) ... nloc(lll)            angular momentum + 1 of every column
    r  V(nloc(1)) ... V(nloc(lll))   ns records: potentials in Rydberg
    r  u(nloc(1)) ... u(nloc(lll))   ns records: reduced radial functions
    r  rho_core                      ns records, only when icore > 0

Unlike a ``MARTINS_NEW`` file it stores the potentials themselves, not
``r*V``, and the core charge as a volume density.  It names no element and no
exchange-correlation functional, and it holds neither the parameters of its
logarithmic radial grid, nor an ionic charge, nor occupations, nor a cutoff
radius.  PARSEC takes the last three from the input of the species
(``Electron_Per_Orbital``, whose sum is the ionic charge, and
``Core_Cutoff_Radius``), checks ``Potential_Num`` against the file, and
derives the grid parameters from two radial samples.

The result is the :class:`ParsecPseudopotential` every builder reads, with
the same units and meaning as the tables of a ``MARTINS_NEW`` file.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .potre import _FLOAT_RE, ParsecPseudopotential, _numbers


#: ``i_0`` of ``pseudo.f90``: the Fortran index of the first of the two radial
#: samples the grid parameters are derived from.  Fortran index 1 is the
#: origin PARSEC inserts, so file sample ``p`` (zero based) has index ``p+2``.
GRID_FIT_INDEX = 101

_INTEGER_RE = re.compile(r"[-+]?\d+")


class _ListDirectedRecords:
    """Sequential ``read(unit,*)`` statements over the lines of a file.

    Every statement starts at a new record, takes as many records as its
    list needs and drops the rest of the last one.
    """

    def __init__(self, lines: list[str], source: Path) -> None:
        self._lines = lines
        self._index = 0
        self._source = source

    def skip(self) -> None:
        """``read(unit,*)`` without a list: pass one record."""
        self._index += 1

    def tokens(self, count: int, what: str, *, end_ok: bool = False) -> list[str] | None:
        """Return ``count`` items, or ``None`` at the end with ``end_ok``."""
        found: list[str] = []
        while len(found) < count:
            if self._index >= len(self._lines):
                if end_ok:
                    return None
                raise ValueError(
                    f"{self._source}: the file ends before {what} is complete"
                )
            found.extend(self._lines[self._index].replace(",", " ").split())
            self._index += 1
        return found[:count]

    def integers(self, count: int, what: str) -> list[int]:
        values = []
        for token in self.tokens(count, what):
            if _INTEGER_RE.fullmatch(token) is None:
                raise ValueError(
                    f"{self._source}: expected an integer in {what}, found {token!r}"
                )
            values.append(int(token))
        return values

    def reals(self, count: int, what: str, *, end_ok: bool = False) -> list[float] | None:
        tokens = self.tokens(count, what, end_ok=end_ok)
        if tokens is None:
            return None
        values = []
        for token in tokens:
            if _FLOAT_RE.fullmatch(token) is None:
                raise ValueError(
                    f"{self._source}: expected a number in {what}, found {token!r}"
                )
            values.append(_numbers(token)[0])
        return values


def logarithmic_grid_parameters(radii: np.ndarray) -> tuple[float, float]:
    """Parameters ``a`` and ``b`` of ``r(i) = a*exp(b*(i-1)) - a``.

    ``i`` is the Fortran index, ``i = 1`` being the origin.  ``pseudo.f90``
    solves the two equations at ``i_0 = 101`` and ``j0 = 2*(i_0-1)+1 = 201``,
    far from the origin for accuracy:

    ``a = r(i_0)**2 / (r(j0) - 2*r(i_0))``,
    ``b = log(r(j0)/a + 1) / (j0 - 1)``.
    """

    radii = np.asarray(radii, dtype=float)
    second_index = 2 * (GRID_FIT_INDEX - 1) + 1
    # ``j0 > p_pot%ns`` with ``ns`` the number of samples plus the origin.
    if second_index > radii.size + 1:
        raise ValueError(
            "an old Martins pseudopotential needs at least "
            f"{second_index - 1} radial samples to derive its grid parameters"
        )
    first = float(radii[GRID_FIT_INDEX - 2])
    second = float(radii[second_index - 2])
    parameter_a = first * first / (second - 2.0 * first)
    parameter_b = np.log(second / parameter_a + 1.0) / float(second_index - 1)
    return float(parameter_a), float(parameter_b)


def _radial_block(
    records: _ListDirectedRecords,
    count: int,
    columns: int,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    table = np.empty((count, columns + 1))
    for row in range(count):
        table[row] = records.reals(columns + 1, what)
    return table[:, 0].copy(), table[:, 1:].copy()


def read_martins_old_pseudopotential(
    path: str | Path,
    *,
    symbol: str,
    number_of_channels: int,
    electrons_per_orbital: Sequence[float],
    core_cutoff_radius: float,
) -> ParsecPseudopotential:
    """Read a PARSEC ``MARTINS_OLD``/``*_POTR.DAT`` pseudopotential.

    ``number_of_channels`` is ``Potential_Num``, ``electrons_per_orbital`` the
    line of the ``Electron_Per_Orbital`` block (s, p, d, f; its first
    ``number_of_channels`` values are used) and ``core_cutoff_radius`` the
    ``Core_Cutoff_Radius`` in bohr.  ``symbol`` names the element, which the
    file does not.

    The returned tables are aligned with the radial samples of the file:
    potentials in Rydberg, reduced radial functions ``u_l(r)``, and volume
    densities in electrons/bohr**3.  ``ionic_charge`` is the sum of the
    occupations, ``valence_density`` the density of the occupied
    pseudo-orbitals, ``sum_l f_l*u_l**2/(4*pi*r**2)``, which is what PARSEC
    uses for a species without ``Read_VCD``, and ``core_radius`` the cutoff
    radius of the input.
    """

    source = Path(path)
    channels = int(number_of_channels)
    if channels != number_of_channels or not 1 <= channels <= 4:
        raise ValueError(
            f"{source}: impossible number of potentials {number_of_channels!r}; "
            "Potential_Num must be 1 to 4"
        )
    occupations = [float(value) for value in electrons_per_orbital]
    if len(occupations) < channels or not np.all(np.isfinite(occupations)):
        raise ValueError(
            f"{source}: Electron_Per_Orbital must give a finite occupation for "
            f"each of the {channels} channels"
        )
    occupations = occupations[:channels]
    cutoff = float(core_cutoff_radius)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(f"{source}: Core_Cutoff_Radius must be positive")

    records = _ListDirectedRecords(
        source.read_text(encoding="utf-8").splitlines(), source
    )
    # First pass of pseudo.f90: ``read(20,*) ns, lll, icore``.
    count, file_channels, core_flag = records.integers(3, "the header")
    if file_channels != channels:
        raise ValueError(
            f"{source} holds {file_channels} pseudopotential components, but "
            f"Potential_Num is {channels}"
        )
    if count < 2:
        raise ValueError(f"{source}: too few radial samples")
    # ``nloc``: the angular momentum + 1 of every column.  PARSEC loops over
    # the components 1..lll afterwards, so that every other list loses a
    # column and leaves a component empty.
    column_slots = records.integers(channels, "the list of angular momenta")
    if sorted(column_slots) != list(range(1, channels + 1)):
        raise ValueError(
            f"{source}: the angular-momentum list {column_slots} must name each "
            f"of the components 1..{channels} once; PARSEC itself uses only "
            "those components"
        )

    potential_radii, potential_columns = _radial_block(
        records, count, channels, "the pseudopotentials"
    )
    radii, wave_columns = _radial_block(
        records, count, channels, "the pseudo-wavefunctions"
    )
    # The grid parameters are derived here, from the radii read last.
    parameter_a, parameter_b = logarithmic_grid_parameters(radii)

    core_density = np.zeros(count)
    core_radii = radii.copy()
    if core_flag > 0:
        # ``read(20,*,end=80)``: a file that ends early keeps zeros.
        for row in range(count):
            values = records.reals(2, "the core charge density", end_ok=True)
            if values is None:
                break
            core_radii[row], core_density[row] = values
    # Every block repeats the radial grid and PARSEC keeps the one read last.
    # Blocks on different grids would pair the tables with wrong radii.
    for name, other in (("pseudopotentials", potential_radii), ("core charge", core_radii)):
        if not np.allclose(other, radii, rtol=1.0e-6, atol=0.0):
            raise ValueError(
                f"{source}: the radial grid of the {name} differs from the "
                "grid of the pseudo-wavefunctions"
            )
    radii = core_radii
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radial grid in {source} must be positive and increasing")

    channel_potentials: dict[int, np.ndarray] = {}
    radial_wavefunctions: dict[int, np.ndarray] = {}
    for column, slot in enumerate(column_slots):
        channel_potentials[slot - 1] = potential_columns[:, column].copy()
        radial_wavefunctions[slot - 1] = wave_columns[:, column].copy()
    channel_occupations = {
        angular_momentum: occupations[angular_momentum]
        for angular_momentum in range(channels)
    }

    valence_density = np.zeros(count)
    for angular_momentum in range(channels):
        wavefunction = radial_wavefunctions[angular_momentum]
        valence_density += (
            channel_occupations[angular_momentum]
            * wavefunction
            * wavefunction
            / (4.0 * np.pi * radii * radii)
        )

    return ParsecPseudopotential(
        symbol=symbol,
        # The file names neither a functional nor a relativistic treatment.
        correlation="",
        relativity="",
        core_marker="pc" if core_flag > 0 else "nc",
        comment="",
        configuration="",
        number_of_channels=channels,
        number_of_spin_orbit_channels=0,
        radial_parameter_a=parameter_a,
        radial_parameter_b=parameter_b,
        # ``p_pot%zion = sum(p_pot%eleatm)``.
        ionic_charge=float(sum(occupations)),
        radii=radii,
        channel_potentials=channel_potentials,
        valence_density=valence_density,
        core_density=core_density,
        radial_wavefunctions=radial_wavefunctions,
        channel_occupations=channel_occupations,
        channel_cutoffs={
            angular_momentum: cutoff for angular_momentum in range(channels)
        },
        source=source,
        core_radius=cutoff,
        file_format="martins_old",
    )


__all__ = [
    "GRID_FIT_INDEX",
    "logarithmic_grid_parameters",
    "read_martins_old_pseudopotential",
]
