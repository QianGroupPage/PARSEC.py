"""Reader of PARSEC's Martins-Wang pseudopotential format (``*_POTRW.DAT``).

``Pseudopotential_Format: martins_wang`` selects the ``MARTINS_WANG``
branches of ``pseudo.f90`` (lines 195-202 and 524-609).  Such a file is read
with list-directed statements and holds::

    ns  icore  iatom  zion  loc  occ_s  occ_p  occ_d  iso
    iref_s  iref_p  iref_d
    r  V_s  V_p  V_d  R_s  R_p  R_d  [rho_core]  [V_so_p  V_so_d]   ns records

Everything but the projector radius comes from the file: the number of radial
samples, whether the core is corrected (``icore``), the ionic charge, the
local channel (``loc``, the position among the channels present, 1 = the
first), the occupations of s, p and d, and which of the three channels are
present (``iref`` nonzero).  The potentials are in Hartree and become
Rydberg, the radial functions are ``R_l(r)`` and become ``u_l = r*R_l``, and
the core charge is a volume density, used as it is.  The input of the species
adds only ``Core_Cutoff_Radius``, the radius of the nonlocal projectors; it
gives no ``Local_Component`` (``usrinputfile.F90:1806-1827``) and its
``Potential_Num`` is replaced by the number of channels of the file.  Like a
``MARTINS_OLD`` file it names neither an element nor a functional, and the
parameters of its logarithmic radial grid follow from the samples at Fortran
indices 101 and 201 (``c = a``).

Deviations from ``pseudo.f90``, all of them inputs Fortran PARSEC mishandles:

* ``pseudo.f90`` stores the potentials and radial functions of the present
  channels in consecutive slots but their occupations at the slot of the
  angular momentum, and later treats slot ``j`` as angular momentum ``j-1``.
  Only files whose present channels are s, s+p or s+p+d agree with that, and
  only those are read here.
* ``loc = 0`` passes the check of ``pseudo.f90`` and selects slot 0 of an
  array that has none; it is refused.
* The spin-orbit columns are read into the wrong array index
  (``vlj(1:2,isy,i)``); ``SO_PSP`` is refused for this format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .potr import _ListDirectedRecords, logarithmic_grid_parameters
from .potre import ParsecPseudopotential


@dataclass(frozen=True)
class MartinsWangHeader:
    """The two header records of a ``*_POTRW.DAT`` file.

    ``local_slot`` is ``loc`` of the file: the position (1 = first) of the
    local channel among the channels present.  ``present`` lists the
    angular momenta whose ``iref`` is nonzero, in ascending order.
    """

    sample_count: int
    core_flag: int
    ionic_charge: float
    local_slot: int
    occupations: tuple[float, float, float]
    spin_orbit_flag: int
    present: tuple[int, ...]

    @property
    def local_angular_momentum(self) -> int:
        return self.present[self.local_slot - 1]


def read_martins_wang_header(path: str | Path) -> MartinsWangHeader:
    """Read and check the header records of a Martins-Wang file.

    The input reader takes the local channel from here: PARSEC reads no
    ``Local_Component`` for this format.
    """

    source = Path(path)
    records = _ListDirectedRecords(
        source.read_text(encoding="utf-8").splitlines(), source
    )
    return _header(records, source)


def _header(records: _ListDirectedRecords, source: Path) -> MartinsWangHeader:
    first = records.tokens(9, "the first header record")
    try:
        count = int(first[0])
        core_flag = int(first[1])
        int(first[2])
        ionic_charge = records_number(first[3])
        local_slot = int(first[4])
        occupations = tuple(records_number(token) for token in first[5:8])
        spin_orbit_flag = int(first[8])
    except ValueError as error:
        raise ValueError(
            f"{source}: cannot read the first header record "
            "'ns, icore, iatom, zion, loc, occ_s, occ_p, occ_d, iso'"
        ) from error
    references = records.integers(3, "the channel record 'iref_s, iref_p, iref_d'")
    present = tuple(l for l, flag in enumerate(references) if flag != 0)
    if not present:
        raise ValueError(f"{source}: the file declares no pseudopotential channel")
    if present != tuple(range(len(present))):
        raise ValueError(
            f"{source}: the channels present (iref = {references}) must be s, "
            "s and p, or s, p and d.  pseudo.f90 stores the channels of the "
            "file in consecutive slots and treats slot j as angular momentum "
            "j-1, which mislabels any other combination"
        )
    if not 1 <= local_slot <= len(present):
        # pseudo.f90 stops with error 204 for loc beyond the channels; loc=0
        # passes that check and reads outside the potential array.
        raise ValueError(
            f"{source}: the local channel loc={local_slot} must name one of "
            f"the {len(present)} channels of the file (1 = s)"
        )
    if count + 1 < 201:
        # pseudo.f90 error 210: the grid parameters need Fortran index 201.
        raise ValueError(
            f"{source}: a Martins-Wang pseudopotential needs at least 200 "
            "radial samples to derive its grid parameters"
        )
    return MartinsWangHeader(
        sample_count=count,
        core_flag=core_flag,
        ionic_charge=ionic_charge,
        local_slot=local_slot,
        occupations=(occupations[0], occupations[1], occupations[2]),
        spin_orbit_flag=spin_orbit_flag,
        present=present,
    )


def records_number(token: str) -> float:
    """One list-directed real, with Fortran's ``d`` exponents."""

    return float(token.replace("D", "E").replace("d", "e"))


def read_martins_wang_pseudopotential(
    path: str | Path,
    *,
    symbol: str,
    core_cutoff_radius: float,
) -> ParsecPseudopotential:
    """Read a PARSEC ``MARTINS_WANG``/``*_POTRW.DAT`` pseudopotential.

    ``symbol`` names the element, which the file does not, and
    ``core_cutoff_radius`` is the ``Core_Cutoff_Radius`` of the species in
    bohr, the radius of its nonlocal projectors.

    The returned tables are aligned with the radial samples of the file:
    potentials in Rydberg, reduced radial functions ``u_l = r*R_l``, the
    valence density of the occupied pseudo-orbitals
    ``sum_l f_l*u_l**2/(4*pi*r**2)`` and the core density in
    electrons/bohr**3.  The ionic charge and the occupations are those of
    the file.
    """

    source = Path(path)
    cutoff = float(core_cutoff_radius)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(f"{source}: Core_Cutoff_Radius must be positive")
    records = _ListDirectedRecords(
        source.read_text(encoding="utf-8").splitlines(), source
    )
    header = _header(records, source)
    count = header.sample_count
    # ``read(20,*) rs, (rtmp(j), j=1,6) [, denc]``: the first eight numbers of
    # every record when the core is corrected, seven otherwise; the rest of
    # the record (spin-orbit columns) is not read.
    columns = 8 if header.core_flag != 0 else 7
    table = np.empty((count, columns))
    for row in range(count):
        table[row] = records.reals(columns, "the radial table")
    radii = table[:, 0].copy()
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radial grid in {source} must be positive and increasing")
    parameter_a, parameter_b = logarithmic_grid_parameters(radii)

    channel_potentials: dict[int, np.ndarray] = {}
    radial_wavefunctions: dict[int, np.ndarray] = {}
    for angular_momentum in header.present:
        # Hartree -> Rydberg; R_l -> u_l = r*R_l.
        channel_potentials[angular_momentum] = table[:, 1 + angular_momentum] * 2.0
        radial_wavefunctions[angular_momentum] = (
            table[:, 4 + angular_momentum] * radii
        )
    channel_occupations = {
        angular_momentum: float(header.occupations[angular_momentum])
        for angular_momentum in header.present
    }
    core_density = (
        table[:, 7].copy() if header.core_flag != 0 else np.zeros(count)
    )
    valence_density = np.zeros(count)
    for angular_momentum in header.present:
        wavefunction = radial_wavefunctions[angular_momentum]
        valence_density += (
            channel_occupations[angular_momentum]
            * wavefunction
            * wavefunction
            / (4.0 * np.pi * radii * radii)
        )

    return ParsecPseudopotential(
        symbol=symbol,
        correlation="",
        relativity="",
        core_marker="pc" if header.core_flag != 0 else "nc",
        comment="",
        configuration="",
        number_of_channels=len(header.present),
        number_of_spin_orbit_channels=0,
        radial_parameter_a=parameter_a,
        radial_parameter_b=parameter_b,
        ionic_charge=float(header.ionic_charge),
        radii=radii,
        channel_potentials=channel_potentials,
        valence_density=valence_density,
        core_density=core_density,
        radial_wavefunctions=radial_wavefunctions,
        channel_occupations=channel_occupations,
        channel_cutoffs={
            angular_momentum: cutoff for angular_momentum in header.present
        },
        source=source,
        core_radius=cutoff,
        file_format="martins_wang",
        local_channel=header.local_angular_momentum,
    )


__all__ = [
    "MartinsWangHeader",
    "read_martins_wang_header",
    "read_martins_wang_pseudopotential",
]
