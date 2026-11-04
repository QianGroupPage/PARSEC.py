"""The legacy restart file ``wfn.dat`` (``Restart_From_Wfndat``, ``restart.F90``).

Current PARSEC writes ``parsec.dat`` (:mod:`parsec_python.Output.parsec_dat`)
and reads ``wfn.dat`` only when ``Restart_Run`` and ``Restart_From_Wfndat``
are both true (``parsec.F90:917-933``, ``usrinputfile.F90:371-375``); no
routine of the current source writes it.  The record layout below is the one
``restart.F90`` reads (gfortran sequential unformatted, little endian,
4-byte markers):

1. ``datelabel`` (``character(26)``);
2. cluster: ``h, rmax`` (``real(dp)``), ``ndim, nwedge, nrep`` (integers);
   periodic: ``h_x, h_y, h_z, L_x, L_y, L_z`` (``real(dp)``), ``ndim,
   nwedge, nrep``;
3. five records that ``restart.F90`` skips (``do kk = 1, 5; read(4)``);
   :func:`write_wfn_dat` writes ``nwedge, ntrans``, ``rmtrx``, ``trans``,
   ``tnp`` and ``alatt, invlat`` of the identity there, the records
   ``wfnsave.F90`` writes at the same place of ``parsec.dat``;
4. ``((chi(k,j), k=1,nrep), j=1,nrep)`` (integers);
5. ``(kx(i), ky(i), kz(i), i=1,nwedge)`` (integers, the grid points);
6. for every spin channel: ``nstate``; ``irep(1:nstate)``;
   ``en(1:nstate)`` (Ry); ``occ(1:nstate)``; ``vold(1:nwedge)`` (Ry);
   ``rho(1:nwedge)`` (the density of the channel, e/bohr^3); then ``nstate``
   wavefunction records, ``real(dp)`` or ``complex(dpc)`` of length
   ``nwedge`` (Euclidean-normalized, as in ``parsec.dat``).

``restart.F90`` reads ``min(nstate, States_Num)`` levels and keeps those
wavefunctions; it has no k-points (``kpnum = 1``) and its checks carry the
errors 321-329.  Numerical modules do not open files; this one is the file
format and only reads or writes the file it is given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parsec_dat import _values, fortran_date_label, read_record, write_record


@dataclass(frozen=True)
class WfnDatChannel:
    """One spin channel of ``wfn.dat``: levels, fields and wavefunctions.

    ``wavefunctions`` has one column per state, on the points of the file.
    """

    representations: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    potential: np.ndarray
    density: np.ndarray
    wavefunctions: np.ndarray


@dataclass(frozen=True)
class WfnDat:
    """Everything ``restart.F90`` reads from ``wfn.dat``.

    ``step`` holds one spacing (cluster) or three (periodic, with
    ``box_size``); ``rmax`` is the sphere radius of a cluster (``None`` for a
    periodic file); ``points`` are the integer coordinates ``(kx, ky, kz)``;
    ``skipped`` are the raw five records ``restart.F90`` skips.
    """

    date_label: str
    step: np.ndarray
    rmax: float | None
    box_size: np.ndarray | None
    ndim: int
    representation_count: int
    characters: np.ndarray
    points: np.ndarray
    channels: tuple[WfnDatChannel, ...]
    skipped: tuple[bytes, ...] = ()

    @property
    def periodic(self) -> bool:
        return self.box_size is not None

    @property
    def nwedge(self) -> int:
        return int(self.points.shape[0])

    @property
    def spin_count(self) -> int:
        return len(self.channels)


def read_wfn_dat(path: str | Path, *, periodic: bool, spin_count: int, complex_values: bool) -> WfnDat:
    """Read ``wfn.dat`` as ``restart.F90`` does.

    The file does not say whether it belongs to a periodic run, how many spin
    channels it has or whether its wavefunctions are complex: PARSEC takes
    all three from the current run, and so does this reader.
    """

    with open(path, "rb") as stream:
        date_label = read_record(stream).decode("ascii", errors="replace")
        header = read_record(stream)
        doubles = 6 if periodic else 2
        if len(header) != 8 * doubles + 12:
            raise ValueError(
                f"{path}: the header record has {len(header)} bytes, "
                f"{8 * doubles + 12} expected for a "
                + ("periodic" if periodic else "cluster")
                + " run"
            )
        reals = np.frombuffer(header[: 8 * doubles], dtype="<f8").copy()
        ndim, nwedge, nrep = (int(value) for value in np.frombuffer(header[8 * doubles :], dtype="<i4"))
        skipped = tuple(read_record(stream) for _ in range(5))
        characters = _values(read_record(stream), "<i4", nrep * nrep, "chi").reshape(nrep, nrep, order="F")
        points = _values(read_record(stream), "<i4", 3 * nwedge, "kx,ky,kz").reshape(nwedge, 3)
        channels = []
        for _ in range(spin_count):
            count = int(_values(read_record(stream), "<i4", 1, "nstate")[0])
            representations = _values(read_record(stream), "<i4", count, "irep")
            energies = _values(read_record(stream), "<f8", count, "en")
            occupations = _values(read_record(stream), "<f8", count, "occ")
            potential = _values(read_record(stream), "<f8", nwedge, "vold")
            density = _values(read_record(stream), "<f8", nwedge, "rho")
            dtype = "<c16" if complex_values else "<f8"
            vectors = np.empty((nwedge, count), dtype=np.complex128 if complex_values else np.float64)
            for state in range(count):
                vectors[:, state] = _values(read_record(stream), dtype, nwedge, "wavefunction")
            channels.append(
                WfnDatChannel(representations, energies, occupations, potential, density, vectors)
            )
    return WfnDat(
        date_label=date_label,
        step=reals[:3] if periodic else reals[:1],
        rmax=None if periodic else float(reals[1]),
        box_size=reals[3:6] if periodic else None,
        ndim=ndim,
        representation_count=nrep,
        characters=characters,
        points=points,
        channels=tuple(channels),
        skipped=skipped,
    )


def write_wfn_dat(path: str | Path, data: WfnDat) -> Path:
    """Write ``data`` in the layout ``restart.F90`` reads.

    Not a PARSEC routine (no current PARSEC writes ``wfn.dat``); it exists
    so that a run of this port can restart PARSEC with
    ``Restart_From_Wfndat`` and for the tests of the reader.
    """

    target = Path(path)
    with open(target, "wb") as stream:
        label = data.date_label.encode("ascii", errors="replace")[:26].ljust(26)
        write_record(stream, label)
        reals = (
            np.concatenate([np.asarray(data.step, dtype="<f8"), np.asarray(data.box_size, dtype="<f8")])
            if data.periodic
            else np.array([float(data.step[0]), float(data.rmax)], dtype="<f8")
        )
        integers = np.array([data.ndim, data.nwedge, data.representation_count], dtype="<i4")
        write_record(stream, reals.tobytes() + integers.tobytes())
        skipped = data.skipped
        if len(skipped) != 5:
            identity = np.eye(3)
            alatt = (
                np.diag(np.asarray(data.box_size, dtype=float))
                if data.periodic
                else identity * 4.0 * float(data.rmax)
            )
            skipped = (
                np.array([data.nwedge, 1], dtype="<i4").tobytes(),
                np.eye(3, dtype="<i4").tobytes(order="F"),
                identity.astype("<f8").tobytes(order="F"),
                np.zeros(3, dtype="<f8").tobytes(),
                alatt.astype("<f8").tobytes(order="F")
                + np.linalg.inv(alatt.T).astype("<f8").tobytes(order="F"),
            )
        for record in skipped:
            write_record(stream, record)
        write_record(stream, np.asarray(data.characters, dtype="<i4").ravel(order="F"))
        write_record(stream, np.asarray(data.points, dtype="<i4").reshape(-1))
        for channel in data.channels:
            count = int(np.asarray(channel.energies).size)
            write_record(stream, np.array([count], dtype="<i4"))
            write_record(stream, np.asarray(channel.representations, dtype="<i4"))
            write_record(stream, np.asarray(channel.energies, dtype="<f8"))
            write_record(stream, np.asarray(channel.occupations, dtype="<f8"))
            write_record(stream, np.asarray(channel.potential, dtype="<f8"))
            write_record(stream, np.asarray(channel.density, dtype="<f8"))
            vectors = np.asarray(channel.wavefunctions)
            dtype = "<c16" if np.iscomplexobj(vectors) else "<f8"
            for state in range(count):
                write_record(stream, np.ascontiguousarray(vectors[:, state], dtype=dtype))
    return target


def wfn_dat_from_result(result, grid, *, date_label: str | None = None) -> WfnDat:
    """The ``wfn.dat`` contents of a finished Python run on ``grid``.

    Full grid (``nrep = 1``), the output potential of each channel as
    ``vold`` (``next_effective_potential``, PARSEC's ``vnew`` after mixing,
    where the result has it), the densities and every computed state.
    """

    from .parsec_dat import fortran_integer_coordinates, grid_rmax, grid_steps, result_channels

    if getattr(result, "k_points", None) is not None:
        raise ValueError("wfn.dat has no k-points (restart.F90: kpnum = 1)")
    channels = result_channels(result)
    periodic = hasattr(grid, "points_per_axis")
    stored = []
    for index in range(len(channels.potentials)):
        vectors = channels.wavefunctions[index]
        if vectors is None:
            raise ValueError("wfn.dat needs the orbitals of the result on the host")
        vectors = np.asarray(vectors)
        energies = np.asarray(channels.eigenvalues[index])[: vectors.shape[1]]
        occupations = np.asarray(channels.occupations[index])[: vectors.shape[1]]
        stored.append(
            WfnDatChannel(
                representations=np.ones(energies.size, dtype=np.int32),
                energies=energies,
                occupations=occupations,
                potential=np.asarray(channels.potentials[index]),
                density=np.asarray(channels.densities[index]),
                wavefunctions=vectors,
            )
        )
    return WfnDat(
        date_label=fortran_date_label() if date_label is None else date_label,
        step=grid_steps(grid) if periodic else grid_steps(grid)[:1],
        rmax=None if periodic else grid_rmax(grid),
        box_size=(
            np.linalg.norm(np.asarray(grid.cell.lattice_vectors, dtype=float), axis=1)
            if periodic
            else None
        ),
        ndim=int(grid.size),
        representation_count=1,
        characters=np.ones((1, 1), dtype=np.int32),
        points=fortran_integer_coordinates(grid),
        channels=tuple(stored),
    )


__all__ = [
    "WfnDat",
    "WfnDatChannel",
    "read_wfn_dat",
    "wfn_dat_from_result",
    "write_wfn_dat",
]
