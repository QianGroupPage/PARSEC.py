"""Reader of FHI98PP pseudopotentials in PARSEC's ``*_FHIPP.DAT`` layout.

``Pseudopotential_Format: fhipp`` selects the ``FHIPP`` branches of
``pseudo.f90`` (lines 204-220 and 611-655).  The file is the ``.cpi`` output
of the FHI98PP generator, read with list-directed statements::

    zion  lmax+1                          ionic charge, number of channels
    ten records                           not read (the abinit header block)
    mesh  amesh                           for each channel l = 0, 1, ...:
    i  r  u  V                            mesh records
    r  rho_core  ...                      mesh records, read only with NLCC

``u`` is the reduced radial function ``r*R_l`` and ``V`` the potential in
Hartree, which becomes Rydberg.  The logarithmic grid is
``r(i) = a*exp(b*(i-1))`` with Fortran index ``i`` (the inserted origin is
``i = 1``), so ``a = r_1/amesh``, ``b = log(amesh)`` and ``c = 0``.  The
core density column is ``4*pi`` times the volume density.

The input of the species adds what the file does not hold: the number of
channels (``Potential_Num``, which must equal the file's), the occupations
(``Electron_Per_Orbital``), the local channel (``Local_Component``), the
radius of the nonlocal projectors (``Core_Cutoff_Radius``) and whether the
core is corrected (``Nonlinear_Core_Correction``).  The ionic charge is the
one of the file.

``pseudo.f90:638`` compares the ``amesh`` of each channel with
``exp(log(amesh))`` of the same channel, which tests nothing; the check
meant, that every channel has the ``amesh`` of the first, is made here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .potr import _ListDirectedRecords
from .potre import ParsecPseudopotential

#: Records between the first one and the first channel (``pseudo.f90:217``).
SKIPPED_RECORDS = 10


def read_fhipp_pseudopotential(
    path: str | Path,
    *,
    symbol: str,
    number_of_channels: int,
    electrons_per_orbital: Sequence[float],
    core_cutoff_radius: float,
    nonlinear_core_correction: bool = False,
) -> ParsecPseudopotential:
    """Read a PARSEC ``FHIPP``/``*_FHIPP.DAT`` pseudopotential.

    ``number_of_channels`` is ``Potential_Num``, ``electrons_per_orbital``
    the line of the ``Electron_Per_Orbital`` block (s, p, d, f; its first
    ``number_of_channels`` values are used), ``core_cutoff_radius`` the
    ``Core_Cutoff_Radius`` in bohr and ``nonlinear_core_correction`` the
    ``Nonlinear_Core_Correction`` flag of the species, which makes PARSEC
    read the core density after the channels.  ``symbol`` names the element.

    The tables are aligned with the radial samples of the file: potentials
    in Rydberg, reduced radial functions, the valence density of the
    occupied pseudo-orbitals and the core density in electrons/bohr**3.
    ``radial_parameter_c`` is zero.
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
    ionic_charge, file_channels = records.reals(2, "the first record 'zion, lmax+1'")
    if int(file_channels) != file_channels or int(file_channels) != channels:
        # pseudo.f90 error 203.
        raise ValueError(
            f"{source} holds {file_channels:g} pseudopotential components, but "
            f"Potential_Num is {channels}"
        )
    for _ in range(SKIPPED_RECORDS):
        records.skip()

    count = None
    amesh = None
    radii = None
    channel_potentials: dict[int, np.ndarray] = {}
    radial_wavefunctions: dict[int, np.ndarray] = {}
    for angular_momentum in range(channels):
        mesh_value, mesh_ratio = records.reals(2, f"the mesh record of l={angular_momentum}")
        mesh = int(mesh_value)
        if mesh != mesh_value or mesh < 2:
            raise ValueError(f"{source}: invalid radial mesh size {mesh_value!r}")
        if count is None:
            count, amesh = mesh, mesh_ratio
        elif mesh != count:
            # pseudo.f90 error 211.
            raise ValueError(
                f"{source}: radial grid size for angular component "
                f"l = {angular_momentum} is {mesh}, not the {count} of the "
                "other components"
            )
        elif mesh_ratio != amesh:
            # The check pseudo.f90 error 212 means to make.
            raise ValueError(
                f"{source}: radial increment {mesh_ratio!r} of angular "
                f"component l = {angular_momentum} differs from the "
                f"increment {amesh!r} of the other components"
            )
        table = np.empty((mesh, 4))
        for row in range(mesh):
            table[row] = records.reals(4, f"the table of l={angular_momentum}")
        channel_radii = table[:, 1].copy()
        if radii is not None and not np.allclose(channel_radii, radii, rtol=1.0e-10, atol=0.0):
            raise ValueError(
                f"{source}: the radial grid of l={angular_momentum} differs "
                "from that of l=0"
            )
        radii = channel_radii
        radial_wavefunctions[angular_momentum] = table[:, 2].copy()
        channel_potentials[angular_momentum] = table[:, 3] * 2.0
    assert radii is not None and count is not None and amesh is not None
    if np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
        raise ValueError(f"radial grid in {source} must be positive and increasing")
    if not amesh > 1.0:
        raise ValueError(f"{source}: the radial increment amesh must exceed one")

    core_density = np.zeros(count)
    if nonlinear_core_correction:
        core_table = np.empty((count, 2))
        for row in range(count):
            values = records.reals(
                2, "the core charge density (Nonlinear_Core_Correction)", end_ok=True
            )
            if values is None:
                raise ValueError(
                    f"{source} ends before its core charge density: "
                    "Nonlinear_Core_Correction needs one record 'r, rho_core' "
                    "per radial sample after the channels"
                )
            core_table[row] = values
        if not np.allclose(core_table[:, 0], radii, rtol=1.0e-10, atol=0.0):
            raise ValueError(
                f"{source}: the radial grid of the core charge differs from "
                "that of the channels"
            )
        # PARSEC keeps the radii read last.
        radii = core_table[:, 0].copy()
        core_density = core_table[:, 1] / (4.0 * np.pi)

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
        correlation="",
        relativity="",
        core_marker="pc" if nonlinear_core_correction else "nc",
        comment="",
        configuration="",
        number_of_channels=channels,
        number_of_spin_orbit_channels=0,
        radial_parameter_a=float(radii[0] / amesh),
        radial_parameter_b=float(np.log(amesh)),
        ionic_charge=float(ionic_charge),
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
        file_format="fhipp",
        radial_parameter_c=0.0,
    )


__all__ = ["SKIPPED_RECORDS", "read_fhipp_pseudopotential"]
