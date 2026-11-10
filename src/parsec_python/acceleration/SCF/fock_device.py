"""The device pieces of the Fock-loop controls (``Exx_Outer_Mixing``).

:mod:`parsec_python.SCF.fock_diis` extrapolates the density matrix of each
spin channel and needs, from an engine, copies of the occupied orbitals of a
loop (one block per symmetry sector, kept while the eigensolver moves on) and
the channel of the natural orbitals of the extrapolated density matrix.  Here
they are formed where the orbitals are: the sector bases on their devices
(:class:`~parsec_python.acceleration.Hamiltonian.exchange_device.ChannelOrbitals`),
the natural orbitals of a sector on its device, the empty states of the last
loop orthonormalized against them there.  The exchange operator of the
result is built by the same pair solves as any other
(:func:`~parsec_python.acceleration.Hamiltonian.exchange_device.apply_device_exchange`).

Units Ry, bohr.  Nothing here prints or opens files.
"""

from __future__ import annotations

import numpy as np

from parsec_python.SCF.fock_diis import OccupiedBlock, orthonormal_complement
from parsec_python.V_xc.exact_exchange import OCCUPATION_THRESHOLD

from ..Hamiltonian.exchange_device import ChannelOrbitals, OrbitalSector


#: Share of each device's memory that the window of the outer DIIS leaves free.
WINDOW_MARGIN = 0.10
#: Columns of States_Num that the channel of an extrapolated density matrix holds at its peak
#: (:func:`natural_channels`: the natural orbitals and the copied empty states, their
#: orthonormalized copy, the joined block).
BUILD_COLUMNS = 3


def _sector_bytes(sets, occupied_only: bool) -> dict[int, int]:
    """Bytes of the occupied (or all) columns of ``sets`` on each device."""

    counts: dict[int, int] = {}
    for orbitals in sets:
        for sector in orbitals.sectors:
            weights = orbitals.occupations[sector.states]
            count = int(np.count_nonzero(weights > OCCUPATION_THRESHOLD)) if occupied_only else int(weights.size)
            counts[int(sector.device)] = counts.get(int(sector.device), 0) + 8 * int(sector.rows) * count
    return counts


def window_fits(sets) -> bool:
    """Whether one more copy of the occupied columns of ``sets`` fits where each sector is.

    The window is not in the memory plan of the run (the driver's sector
    memory rules see the exchange only): a copy is made only where the
    device driver has free (not the pool's free blocks, which the inner
    loops take again) the copy, the build of an extrapolated operator
    (``BUILD_COLUMNS`` sets of the States_Num columns) and ``WINDOW_MARGIN``
    of the device.  A spilled sector's copy goes to its device as well.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    copies = _sector_bytes(sets, True)
    states = _sector_bytes(sets, False)
    for device, count in copies.items():
        with cp.cuda.Device(device):
            free, total = cp.cuda.runtime.memGetInfo()
        if count + BUILD_COLUMNS * states.get(device, 0) + WINDOW_MARGIN * float(total) > float(free):
            return False
    return True


def build_fits(sets) -> bool:
    """Whether the channel of an extrapolated density matrix (:func:`natural_channels`) fits now.

    ``BUILD_COLUMNS`` sets of the States_Num columns of ``sets`` against the
    free bytes of each device (the driver's and the pool's: the inner loop
    is not running).
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    for device, count in _sector_bytes(sets, False).items():
        with cp.cuda.Device(device):
            free, _total = cp.cuda.runtime.memGetInfo()
            available = int(free) + int(cp.get_default_memory_pool().free_bytes())
        if BUILD_COLUMNS * count > available:
            return False
    return True


def occupied_blocks(sets) -> list | None:
    """Per channel ``{representation: OccupiedBlock}``: copies of the occupied columns on each sector's device.

    ``None`` when the copies do not fit the free memory of the devices
    (:func:`window_fits`): the outer DIIS then stops and the loop goes on
    with plain steps.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    if not window_fits(sets):
        return None
    channels = []
    for orbitals in sets:
        blocks = {}
        for sector in orbitals.sectors:
            weights = orbitals.occupations[sector.states]
            positions = np.flatnonzero(weights > OCCUPATION_THRESHOLD)
            if not positions.size:
                continue
            with cp.cuda.Device(sector.device):
                vectors = cp.array(sector.columns(cp, positions), dtype=cp.float64, order="F", copy=True)
            blocks[sector.representation] = OccupiedBlock(vectors, np.array(weights[positions], dtype=np.float64))
        channels.append(blocks)
    return channels


def natural_channels(sets, natural):
    """The :class:`ChannelOrbitals` of natural orbitals and the empty states of ``sets``, or ``None``.

    ``natural[s]`` maps a representation to ``(Phi, occupations)`` (``Phi``
    on the sector's device, orthonormal).  Each sector of the new channel
    holds its natural orbitals and the empty states of the same sector of
    ``sets``, orthonormalized against them; ``None`` when ``natural`` names a
    sector that ``sets`` does not have.
    """

    from ..backends.cupy import require_cupy

    cp, _ = require_cupy()
    result = []
    for orbitals, blocks in zip(sets, natural):
        if set(blocks) - {sector.representation for sector in orbitals.sectors}:
            return None
        sectors = []
        occupations = []
        start = 0
        for sector in orbitals.sectors:
            weights = orbitals.occupations[sector.states]
            empty = np.flatnonzero(weights <= OCCUPATION_THRESHOLD)
            phi, values = blocks.get(sector.representation, (None, np.zeros(0)))
            with cp.cuda.Device(sector.device):
                rest = sector.columns(cp, empty) if empty.size else None
                if phi is not None and rest is not None:
                    rest = orthonormal_complement(phi, rest)
                parts = [part for part in (phi, rest) if part is not None and part.shape[1]]
                if not parts:
                    continue
                vectors = cp.asfortranarray(cp.concatenate(parts, axis=1))
            count = int(vectors.shape[1])
            sectors.append(
                OrbitalSector(
                    representation=sector.representation,
                    device=int(sector.device),
                    vectors=vectors,
                    local=np.arange(count, dtype=np.int64),
                    states=np.arange(start, start + count, dtype=np.int64),
                    expansion=sector.expansion,
                )
            )
            occupations.append(np.concatenate([np.asarray(values, dtype=np.float64), np.zeros(count - len(values))]))
            start += count
        result.append(
            ChannelOrbitals(
                tuple(sectors),
                np.concatenate(occupations) if occupations else np.zeros(0),
                float(orbitals.spin_weight),
                int(orbitals.full_size),
                float(orbitals.volume_element),
            )
        )
    return tuple(result)


__all__ = ["build_fits", "natural_channels", "occupied_blocks", "window_fits"]
