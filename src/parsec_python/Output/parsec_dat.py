"""PARSEC's ``parsec.dat``: grid, potential, density, levels and wavefunctions.

``wfnsave.F90`` writes the file at the end of every run (``Output_Data``) and
``restart_run.F90`` reads it back for ``Restart_Run``.  It is a Fortran
sequential unformatted file: every ``write(iunit)`` is one record framed by
4-byte little-endian length markers (gfortran; a record longer than
2**31-9 bytes is split into subrecords whose markers carry a sign).  Integers
are 4 bytes, reals 8 and complex numbers 16.  The records, in order:

1. ``datelabel`` (26 characters, ``custom_date_time``);
2. ``stype(10)``: (1) ``nspin``; (2) 0 real, 1 complex, 2 spin-orbit,
   3 non-collinear, 4 both; (3) 0 cluster, 1 bulk, 2 wire, 3 slab;
3. periodic: ``step(3), box_size(3), rmax`` | ``latt_vec(3,3)`` |
   ``lap_dir_num`` | ``lap_dir(3)`` | ``lap_dir_step(3)`` | ``kpnum`` |
   ``kptmethod`` | ``mpgrid(3)`` | ``mpshift(3)`` | ``kpts(3,kpnum)`` |
   ``weight(kpnum)``; cluster: ``step(1), rmax``;
4. ``ndim`` | ``shift(3)`` | ``nwedge, ntrans`` | ``rmtrx(3,3,ntrans)`` |
   ``trans(3,3,ntrans)`` | ``tnp(3,ntrans)`` | ``alatt(3,3), invlat(3,3)`` |
   ``chi(ntrans,ntrans)`` | ``(kx,ky,kz)`` of the ``nwedge`` points;
5. per spin channel ``isp = 1..nspin/mxwd``: per k-point ``nstate`` |
   ``irep`` | ``en`` (Ry) | ``occ``; then ``vnew`` (Ry) | ``rho`` (e/bohr^3,
   the total density with one channel, that of the channel with two);
6. with ``mxwd = 2`` (spinors) ``vnew(:,2)`` | ``rho(:,3)``;
7. per spin channel and k-point: ``nwedge, ntrans`` | ``rmtrx`` | ``trans``
   | ``tnp`` | ``alatt, invlat`` | ``chi`` | ``(kx,ky,kz)`` | ``nsave`` |
   ``indxsave`` (when ``nsave > 0``) | ``nsave`` wavefunction records, real
   ``(nwedge)`` or complex ``(nwedge*mxwd)``, normalized to ``sum |q|^2 = 1``.

Grid points are integers ``(kx,ky,kz)`` with the position
``(shift + k) * step`` along each lattice direction.  For a cluster these are
the integer coordinates of :class:`parsec_python.Grid.RealSpaceGrid`
unchanged.  A periodic Python grid numbers its points ``i = 0..n-1`` with
position ``i * h``; the file takes the periodic image in Fortran's range
``k = -n/2..n-n/2-1`` (:func:`fortran_integer_coordinates`).  Python always
works on the full grid, so it writes one symmetry operation (the identity)
and ``nwedge = ndim``; a file of a symmetry-reduced Fortran run is unfolded
onto the full grid with its own operations (:func:`grid_rows_of_file`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np

#: Largest payload gfortran puts in one (sub)record.
_MAXIMUM_SUBRECORD = 2147483639
#: ``stype(3)`` of ``wfnsave.F90`` for the periodicity ``pbc%per`` 0..3.
BOUNDARY_CODES = {0: 0, 1: 2, 2: 3, 3: 1}
#: ``KP_MANUAL`` and ``MONKHORST_PACK`` of ``const.f90``.
KPOINT_MANUAL = 1
KPOINT_MONKHORST_PACK = 2


def fortran_date_label(now: datetime | None = None) -> str:
    """``custom_date_time`` of ``wfnsave.F90``: ``DD-MON-YYYY HH:MM:SS +zzzz``.

    26 characters, padded with blanks as a Fortran ``character(len=26)``.
    """

    moment = (now or datetime.now()).astimezone()
    month = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP",
             "OCT", "NOV", "DEC")[moment.month - 1]
    zone = moment.strftime("%z") or "+0000"
    text = f"{moment.day:02d}-{month}-{moment.year:04d} {moment:%H:%M:%S} {zone}"
    return text[:26].ljust(26)


# ----------------------------------------------------------------------------
# Sequential unformatted records
# ----------------------------------------------------------------------------


def write_record(stream: BinaryIO, payload: bytes | memoryview | np.ndarray) -> None:
    """Write one gfortran sequential record (subrecords above 2**31-9 bytes)."""

    if isinstance(payload, np.ndarray):
        payload = np.ascontiguousarray(payload)
        view = memoryview(payload).cast("B")
    else:
        view = memoryview(payload).cast("B")
    total = len(view)
    if total <= _MAXIMUM_SUBRECORD:
        marker = np.int32(total).tobytes()
        stream.write(marker)
        stream.write(view)
        stream.write(marker)
        return
    start = 0
    first = True
    while start < total:
        size = min(_MAXIMUM_SUBRECORD, total - start)
        last = start + size >= total
        # The head is negative where another subrecord follows, the tail
        # where this subrecord continues a previous one.
        head = size if last else -size
        tail = size if first else -size
        stream.write(np.int32(head).tobytes())
        stream.write(view[start : start + size])
        stream.write(np.int32(tail).tobytes())
        start += size
        first = False


def read_record(stream: BinaryIO) -> bytes:
    """Read one gfortran sequential record and check its markers."""

    pieces = []
    while True:
        head_bytes = stream.read(4)
        if len(head_bytes) != 4:
            raise EOFError("parsec.dat ended before the record that was expected")
        head = int(np.frombuffer(head_bytes, dtype="<i4")[0])
        size = abs(head)
        data = stream.read(size)
        tail_bytes = stream.read(4)
        if len(data) != size or len(tail_bytes) != 4:
            raise EOFError("parsec.dat ended inside a record")
        tail = abs(int(np.frombuffer(tail_bytes, dtype="<i4")[0]))
        if tail != size:
            raise ValueError(
                "parsec.dat is not a gfortran sequential unformatted file "
                f"(record markers {size} and {tail} disagree)"
            )
        pieces.append(data)
        if head >= 0:
            return b"".join(pieces) if len(pieces) > 1 else pieces[0]


def _values(record: bytes, dtype: str, count: int | None = None, label: str = "") -> np.ndarray:
    values = np.frombuffer(record, dtype=dtype)
    if count is not None and values.size != count:
        raise ValueError(
            f"parsec.dat record {label or dtype} holds {values.size} values, "
            f"{count} expected"
        )
    return values.copy()


# ----------------------------------------------------------------------------
# Contents
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsecDatSymmetry:
    """The symmetry records of ``rsymm`` (the Abelian subgroup of the run).

    ``rmtrx[t][i, j]`` is ``rmtrx(i,j,t)`` (integer, lattice coordinates),
    ``trans[t][i, j]`` is ``trans(i,j,t)`` (Cartesian), ``tnp[t]`` the
    fractional translation of operation ``t`` in lattice units, ``alatt``
    and ``invlat`` the matrices of ``symmetries.f90`` (columns are the cell
    vectors; ``invlat = inverse(transpose(alatt))``), ``chi[k, j]`` the
    character ``chi(k,j)``.  Operation ``t`` maps ``r`` to
    ``trans[t] @ (r - alatt @ tnp[t])`` (``symop``).
    """

    rmtrx: np.ndarray
    trans: np.ndarray
    tnp: np.ndarray
    alatt: np.ndarray
    invlat: np.ndarray
    chi: np.ndarray

    @property
    def ntrans(self) -> int:
        return int(self.rmtrx.shape[0])

    @classmethod
    def identity(cls, alatt: np.ndarray) -> "ParsecDatSymmetry":
        """The identity alone, as ``symmetries.f90`` sets it without symmetry."""

        alatt = np.asarray(alatt, dtype=np.float64)
        return cls(
            rmtrx=np.eye(3, dtype=np.int32)[None, :, :],
            trans=np.eye(3)[None, :, :],
            tnp=np.zeros((1, 3)),
            alatt=alatt,
            invlat=np.linalg.inv(alatt.T),
            chi=np.ones((1, 1), dtype=np.int32),
        )

    def apply(self, operation: int, positions: np.ndarray) -> np.ndarray:
        """``symop``: ``trans @ (r - alatt @ tnp)`` for rows ``r`` of ``positions``."""

        offset = self.alatt @ self.tnp[operation]
        return (np.asarray(positions, dtype=np.float64) - offset) @ self.trans[operation].T


@dataclass(frozen=True)
class ParsecDatPeriodic:
    """The periodic-cell records.  ``lattice_vectors`` rows are ``a1, a2, a3``
    (the columns of Fortran ``latt_vec``); ``k_points`` are Cartesian in
    1/bohr, one per row."""

    box_size: np.ndarray
    lattice_vectors: np.ndarray
    lap_dir_num: int = 0
    lap_dir: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))
    lap_dir_step: np.ndarray = field(default_factory=lambda: np.zeros(3))
    kpoint_method: int = 0
    monkhorst_pack_grid: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int32))
    monkhorst_pack_shift: np.ndarray = field(default_factory=lambda: np.zeros(3))
    k_points: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    k_weights: np.ndarray = field(default_factory=lambda: np.ones(1))


@dataclass(frozen=True)
class ParsecDatLevels:
    """``nstate | irep | en | occ`` of one spin channel at one k-point."""

    representations: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray


@dataclass(frozen=True)
class ParsecDatWavefunctions:
    """The wavefunction block of one spin channel at one k-point.

    ``state_numbers`` are the 1-based ``indxsave``; column ``n`` of
    ``vectors`` is the wavefunction of state ``state_numbers[n]`` on the
    ``points`` of the block (``nwedge * mxwd`` rows for spinors).
    """

    symmetry: ParsecDatSymmetry
    points: np.ndarray
    state_numbers: np.ndarray
    vectors: np.ndarray


@dataclass(frozen=True)
class ParsecDat:
    """Everything ``wfnsave.F90`` writes.

    ``step`` has three entries (all equal for a cluster, which writes the
    first only).  ``levels[s][k]``, ``potentials[s]``, ``densities[s]`` and
    ``wavefunctions[s][k]`` are indexed by spin channel ``s`` (one, or up and
    down) and k-point ``k``; ``potentials`` are ``vnew`` (Ry, after mixing)
    and ``densities`` the valence density of each channel in e/bohr^3 (the
    total density when there is one channel).  Fields are given on the
    ``points`` of the file (``nwedge`` of them).
    """

    date_label: str
    system_type: np.ndarray
    step: np.ndarray
    rmax: float
    ndim: int
    shift: np.ndarray
    symmetry: ParsecDatSymmetry
    points: np.ndarray
    levels: tuple[tuple[ParsecDatLevels, ...], ...]
    potentials: tuple[np.ndarray, ...]
    densities: tuple[np.ndarray, ...]
    wavefunctions: tuple[tuple[ParsecDatWavefunctions, ...], ...] = ()
    periodic: ParsecDatPeriodic | None = None

    @property
    def spin_count(self) -> int:
        return int(self.system_type[0])

    @property
    def wavefunction_kind(self) -> int:
        return int(self.system_type[1])

    @property
    def boundary_code(self) -> int:
        return int(self.system_type[2])

    @property
    def is_complex(self) -> bool:
        return self.wavefunction_kind != 0

    @property
    def spinor_width(self) -> int:
        """``mxwd``: 2 for spin-orbit or non-collinear spinors, else 1."""

        return 2 if self.wavefunction_kind >= 2 else 1

    @property
    def nwedge(self) -> int:
        return int(self.points.shape[0])

    @property
    def kpoint_count(self) -> int:
        return 1 if self.periodic is None else int(self.periodic.k_points.shape[0])


def system_type(spin_count: int, wavefunction_kind: int, boundary_code: int) -> np.ndarray:
    """``stype(10)`` of ``wfnsave.F90``."""

    values = np.zeros(10, dtype=np.int32)
    values[0] = int(spin_count)
    values[1] = int(wavefunction_kind)
    values[2] = int(boundary_code)
    return values


def _column_major(matrices: np.ndarray) -> np.ndarray:
    """Fortran order of ``(n, 3, 3)`` matrices ``[t][i, j]`` (i fastest)."""

    return np.ascontiguousarray(np.asarray(matrices).transpose(0, 2, 1)).ravel()


def _from_column_major(values: np.ndarray, count: int) -> np.ndarray:
    return values.reshape(count, 3, 3).transpose(0, 2, 1).copy()


def _write_symmetry(stream: BinaryIO, symmetry: ParsecDatSymmetry, points: np.ndarray) -> None:
    write_record(stream, np.asarray([points.shape[0], symmetry.ntrans], dtype="<i4"))
    write_record(stream, _column_major(symmetry.rmtrx).astype("<i4"))
    write_record(stream, _column_major(symmetry.trans).astype("<f8"))
    write_record(stream, np.asarray(symmetry.tnp, dtype="<f8").ravel())
    write_record(
        stream,
        np.concatenate(
            [np.asarray(symmetry.alatt).T.ravel(), np.asarray(symmetry.invlat).T.ravel()]
        ).astype("<f8"),
    )
    write_record(stream, np.asarray(symmetry.chi).T.astype("<i4").ravel())
    write_record(stream, np.asarray(points, dtype="<i4").ravel())


def _read_symmetry(stream: BinaryIO) -> tuple[ParsecDatSymmetry, np.ndarray]:
    nwedge, ntrans = (int(value) for value in _values(read_record(stream), "<i4", 2, "nwedge,ntrans"))
    rmtrx = _from_column_major(_values(read_record(stream), "<i4", 9 * ntrans, "rmtrx"), ntrans)
    trans = _from_column_major(_values(read_record(stream), "<f8", 9 * ntrans, "trans"), ntrans)
    tnp = _values(read_record(stream), "<f8", 3 * ntrans, "tnp").reshape(ntrans, 3)
    lattice = _values(read_record(stream), "<f8", 18, "alatt,invlat")
    chi = _values(read_record(stream), "<i4", ntrans * ntrans, "chi").reshape(ntrans, ntrans).T
    points = _values(read_record(stream), "<i4", 3 * nwedge, "kx,ky,kz").reshape(nwedge, 3)
    symmetry = ParsecDatSymmetry(
        rmtrx=rmtrx.astype(np.int32),
        trans=trans,
        tnp=tnp,
        alatt=lattice[:9].reshape(3, 3).T.copy(),
        invlat=lattice[9:].reshape(3, 3).T.copy(),
        chi=np.ascontiguousarray(chi).astype(np.int32),
    )
    return symmetry, points.astype(np.int64)


def write_parsec_dat(path: str | Path, data: ParsecDat) -> Path:
    """Write ``data`` in the record layout of ``wfnsave.F90``."""

    output = Path(path)
    spins = data.spin_count // data.spinor_width
    kpoints = data.kpoint_count
    if len(data.levels) != spins or any(len(block) != kpoints for block in data.levels):
        raise ValueError("parsec.dat needs the levels of every spin channel and k-point")
    expected_fields = data.spin_count
    if len(data.potentials) != expected_fields or len(data.densities) != expected_fields:
        raise ValueError("parsec.dat needs one potential and one density per spin component")
    nwedge = data.nwedge
    for values in (*data.potentials, *data.densities):
        if np.asarray(values).shape != (nwedge,):
            raise ValueError("a parsec.dat field must hold one value per grid point")
    wavefunctions = data.wavefunctions or tuple(
        tuple(
            ParsecDatWavefunctions(
                data.symmetry,
                data.points,
                np.zeros(0, dtype=np.int32),
                np.zeros((nwedge * data.spinor_width, 0)),
            )
            for _ in range(kpoints)
        )
        for _ in range(spins)
    )
    with output.open("wb") as stream:
        write_record(stream, data.date_label[:26].ljust(26).encode("ascii"))
        write_record(stream, np.asarray(data.system_type, dtype="<i4"))
        if data.periodic is not None:
            cell = data.periodic
            write_record(
                stream,
                np.concatenate(
                    [np.asarray(data.step, dtype=float), np.asarray(cell.box_size, dtype=float), [data.rmax]]
                ).astype("<f8"),
            )
            write_record(stream, np.asarray(cell.lattice_vectors, dtype="<f8").ravel())
            write_record(stream, np.asarray([cell.lap_dir_num], dtype="<i4"))
            write_record(stream, np.asarray(cell.lap_dir, dtype="<i4"))
            write_record(stream, np.asarray(cell.lap_dir_step, dtype="<f8"))
            write_record(stream, np.asarray([kpoints], dtype="<i4"))
            write_record(stream, np.asarray([cell.kpoint_method], dtype="<i4"))
            write_record(stream, np.asarray(cell.monkhorst_pack_grid, dtype="<i4"))
            write_record(stream, np.asarray(cell.monkhorst_pack_shift, dtype="<f8"))
            write_record(stream, np.asarray(cell.k_points, dtype="<f8").ravel())
            write_record(stream, np.asarray(cell.k_weights, dtype="<f8"))
        else:
            write_record(stream, np.asarray([data.step[0], data.rmax], dtype="<f8"))
        write_record(stream, np.asarray([data.ndim], dtype="<i4"))
        write_record(stream, np.asarray(data.shift, dtype="<f8"))
        _write_symmetry(stream, data.symmetry, data.points)
        for spin in range(spins):
            for block in data.levels[spin]:
                count = int(np.asarray(block.energies).size)
                write_record(stream, np.asarray([count], dtype="<i4"))
                write_record(stream, np.asarray(block.representations, dtype="<i4"))
                write_record(stream, np.asarray(block.energies, dtype="<f8"))
                write_record(stream, np.asarray(block.occupations, dtype="<f8"))
            write_record(stream, np.asarray(data.potentials[spin], dtype="<f8"))
            write_record(stream, np.asarray(data.densities[spin], dtype="<f8"))
        if data.spinor_width == 2:
            write_record(stream, np.asarray(data.potentials[1], dtype="<f8"))
            write_record(stream, np.asarray(data.densities[1], dtype="<f8"))
        dtype = "<c16" if data.is_complex else "<f8"
        rows = nwedge * data.spinor_width if data.is_complex else nwedge
        for spin in range(spins):
            for block in wavefunctions[spin]:
                _write_symmetry(stream, block.symmetry, block.points)
                numbers = np.asarray(block.state_numbers, dtype="<i4")
                write_record(stream, np.asarray([numbers.size], dtype="<i4"))
                if numbers.size:
                    write_record(stream, numbers)
                vectors = np.asarray(block.vectors)
                if vectors.shape != (rows, numbers.size):
                    raise ValueError(
                        f"parsec.dat wavefunctions need shape {(rows, numbers.size)}, "
                        f"got {vectors.shape}"
                    )
                for column in range(numbers.size):
                    write_record(stream, np.ascontiguousarray(vectors[:, column], dtype=dtype))
    return output


def read_parsec_dat(path: str | Path, *, wavefunctions: bool = True) -> ParsecDat:
    """Read a ``parsec.dat`` written by PARSEC or by :func:`write_parsec_dat`.

    ``wavefunctions=False`` stops after the potentials and densities, which
    is what ``restart_run.F90`` reads.
    """

    source = Path(path)
    with source.open("rb") as stream:
        label = read_record(stream).decode("ascii", errors="replace")
        stype = _values(read_record(stream), "<i4", 10, "stype").astype(np.int32)
        boundary = int(stype[2])
        periodic = None
        if boundary != 0:
            header = _values(read_record(stream), "<f8", 7, "step,box_size,rmax")
            step = header[:3].copy()
            rmax = float(header[6])
            lattice = _values(read_record(stream), "<f8", 9, "latt_vec").reshape(3, 3)
            lap_dir_num = int(_values(read_record(stream), "<i4", 1, "lap_dir_num")[0])
            lap_dir = _values(read_record(stream), "<i4", 3, "lap_dir")
            lap_dir_step = _values(read_record(stream), "<f8", 3, "lap_dir_step")
            kpnum = int(_values(read_record(stream), "<i4", 1, "kpnum")[0])
            method = int(_values(read_record(stream), "<i4", 1, "kptmethod")[0])
            mp_grid = _values(read_record(stream), "<i4", 3, "mpgrid")
            mp_shift = _values(read_record(stream), "<f8", 3, "mpshift")
            k_points = _values(read_record(stream), "<f8", 3 * kpnum, "kpts").reshape(kpnum, 3)
            k_weights = _values(read_record(stream), "<f8", kpnum, "weight")
            periodic = ParsecDatPeriodic(
                box_size=header[3:6].copy(),
                lattice_vectors=lattice.copy(),
                lap_dir_num=lap_dir_num,
                lap_dir=lap_dir,
                lap_dir_step=lap_dir_step,
                kpoint_method=method,
                monkhorst_pack_grid=mp_grid,
                monkhorst_pack_shift=mp_shift,
                k_points=k_points,
                k_weights=k_weights,
            )
        else:
            header = _values(read_record(stream), "<f8", 2, "step,rmax")
            step = np.full(3, header[0])
            rmax = float(header[1])
        ndim = int(_values(read_record(stream), "<i4", 1, "ndim")[0])
        shift = _values(read_record(stream), "<f8", 3, "shift")
        symmetry, points = _read_symmetry(stream)
        nwedge = points.shape[0]
        spin_count = int(stype[0])
        width = 2 if int(stype[1]) >= 2 else 1
        spins = max(spin_count // width, 1)
        kpoints = 1 if periodic is None else periodic.k_points.shape[0]
        levels = []
        potentials = []
        densities = []
        for _spin in range(spins):
            blocks = []
            for _k in range(kpoints):
                count = int(_values(read_record(stream), "<i4", 1, "nstate")[0])
                representations = _values(read_record(stream), "<i4", count, "irep")
                energies = _values(read_record(stream), "<f8", count, "en")
                occupations = _values(read_record(stream), "<f8", count, "occ")
                blocks.append(ParsecDatLevels(representations, energies, occupations))
            levels.append(tuple(blocks))
            potentials.append(_values(read_record(stream), "<f8", nwedge, "vnew"))
            densities.append(_values(read_record(stream), "<f8", nwedge, "rho"))
        if width == 2:
            potentials.append(_values(read_record(stream), "<f8", nwedge, "vnew(:,2)"))
            densities.append(_values(read_record(stream), "<f8", nwedge, "rho(:,3)"))
        blocks_by_spin: list[tuple[ParsecDatWavefunctions, ...]] = []
        if wavefunctions:
            dtype = "<c16" if int(stype[1]) != 0 else "<f8"
            for _spin in range(spins):
                blocks = []
                for _k in range(kpoints):
                    block_symmetry, block_points = _read_symmetry(stream)
                    rows = block_points.shape[0] * (width if dtype == "<c16" else 1)
                    count = int(_values(read_record(stream), "<i4", 1, "nsave")[0])
                    numbers = (
                        _values(read_record(stream), "<i4", count, "indxsave")
                        if count > 0
                        else np.zeros(0, dtype=np.int32)
                    )
                    vectors = np.empty((rows, max(count, 0)), dtype=np.complex128 if dtype == "<c16" else np.float64)
                    for column in range(max(count, 0)):
                        vectors[:, column] = _values(read_record(stream), dtype, rows, "wavefunction")
                    blocks.append(
                        ParsecDatWavefunctions(block_symmetry, block_points, numbers, vectors)
                    )
                blocks_by_spin.append(tuple(blocks))
    return ParsecDat(
        date_label=label,
        system_type=stype,
        step=np.asarray(step, dtype=float),
        rmax=rmax,
        ndim=ndim,
        shift=shift,
        symmetry=symmetry,
        points=points,
        levels=tuple(levels),
        potentials=tuple(potentials),
        densities=tuple(densities),
        wavefunctions=tuple(blocks_by_spin),
        periodic=periodic,
    )


# ----------------------------------------------------------------------------
# Grid conventions
# ----------------------------------------------------------------------------


def _is_periodic(grid) -> bool:
    return hasattr(grid, "points_per_axis")


def _is_general_periodic(grid) -> bool:
    return bool(getattr(grid, "is_general_periodic", False))


def _is_confined(grid) -> bool:
    """A wire or slab grid (:class:`parsec_python.Grid.confined.ConfinedPeriodicGrid`)."""

    return bool(getattr(grid, "is_confined_periodic", False))


def boundary_code(grid) -> int:
    """``stype(3)`` of ``wfnsave.F90`` for ``grid``: 0 cluster, 1 bulk, 2 wire, 3 slab."""

    if _is_confined(grid):
        return BOUNDARY_CODES[int(grid.periodicity)]
    return 1 if _is_periodic(grid) else 0


def grid_steps(grid) -> np.ndarray:
    """``grid%step``: the spacing along each lattice direction (bohr)."""

    if _is_general_periodic(grid):
        # GeneralPeriodicGrid: adjust_step's step per lattice direction.
        return np.asarray(grid.spacings, dtype=float).copy()
    if _is_periodic(grid):
        lengths = np.linalg.norm(np.asarray(grid.cell.lattice_vectors, dtype=float), axis=1)
        return lengths / np.asarray(grid.points_per_axis, dtype=float)
    return np.full(3, float(grid.spacing))


def grid_shift(grid) -> np.ndarray:
    """``grid%shift`` of the points of ``grid``.

    Every periodic grid carries its own (Fortran's rule ``usrinputfile.F90:786-794`` or PARSEC 1.5's
    ``Grid_Shift`` block; the orthorhombic grid of the Fortran-derived routines started at the origin before
    PARSEC 1.5's parity round, and its files of then hold zero), a cluster grid that of its settings.
    """

    if _is_general_periodic(grid):
        return np.asarray(grid.shift, dtype=float).copy()
    if _is_periodic(grid):
        if hasattr(grid, "shift"):
            # A bulk grid of either set of routines; a restart from a file written on another grid is refused
            # by the shift (SCF.restart).
            return np.asarray(grid.shift, dtype=float).copy()
        return np.zeros(3)
    return np.asarray(grid.settings.shift, dtype=float)


def fortran_integer_coordinates(grid) -> np.ndarray:
    """``(kx,ky,kz)`` of every point of ``grid`` in PARSEC's convention.

    A cluster grid already numbers its points as ``grid_partition.f90``
    (position ``(shift + k) h``).  A periodic Python grid numbers them
    ``i = 0..n-1`` (position ``i h``); the periodic image ``k = i`` or
    ``i - n`` in Fortran's range ``-n/2 .. n-n/2-1`` has the same position
    modulo the cell.
    """

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    if not _is_periodic(grid) or _is_confined(grid):
        # A cluster, wire or slab grid numbers its points as Fortran does.
        return points.copy()
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    low = counts // 2
    return (points + low) % counts - low


def grid_rmax(grid) -> float:
    """``grid%rmax``: the sphere radius of a cluster, the cylinder radius or
    half height of a wire or slab (``Boundary_Sphere_Radius``), one for a bulk
    cell."""

    if _is_confined(grid):
        return float(grid.confinement_radius)
    if _is_periodic(grid):
        return 1.0
    return float(grid.settings.enclosing_radius)


def cluster_alatt(rmax: float) -> np.ndarray:
    """``symm%alatt`` of a cluster: a cube of side ``4 rmax`` (``symmetries.f90``)."""

    return np.eye(3) * (4.0 * float(rmax))


@dataclass(frozen=True)
class GridMapping:
    """Where the points of a ``parsec.dat`` lie on a Python grid.

    ``point_of_row[r]`` is the index of the file point whose value grid row
    ``r`` takes (through a symmetry operation of the file), ``-1`` where the
    file has no point there.  ``dropped`` counts file points that are no
    point of the grid (outside its domain); it is exact for a file of the
    full grid and zero for a symmetry-reduced one.
    """

    point_of_row: np.ndarray
    dropped: int

    @property
    def missing(self) -> int:
        return int(np.count_nonzero(self.point_of_row < 0))

    def field(self, values: np.ndarray) -> np.ndarray:
        """The file field ``values`` on the grid, zero where the file has none."""

        values = np.asarray(values)
        result = np.zeros(self.point_of_row.shape[0], dtype=values.dtype)
        present = self.point_of_row >= 0
        result[present] = values[self.point_of_row[present]]
        return result


def grid_rows_of_file(data: ParsecDat, grid, *, tolerance: float = 1.0e-4) -> GridMapping:
    """Map the points of ``data`` onto ``grid``, unfolding its symmetry wedge.

    Every grid point ``r`` is moved by the operations of the file in turn
    (``symop``: ``trans (r - alatt tnp)``) until it lands on a point of the
    file, as ``grid_partition.f90`` built the wedge.  The integer of a
    position is ``nint(r/step - shift)`` with the shift of the file; a
    position more than ``tolerance`` bohr off its grid node is an error
    (``ierr = 271`` there).  A periodic grid wraps the moved points into the
    cell.  The grid and the file must have the same spacing and shift
    (checked by :func:`parsec_python.SCF.restart.check_restart_file`).
    """

    steps = np.asarray(data.step, dtype=float)
    shift = np.asarray(data.shift, dtype=float)
    file_points = np.asarray(data.points, dtype=np.int64)
    periodic = _is_periodic(grid)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64) if periodic else None
    # Only the periodic directions wrap; a wire or slab keeps the confined
    # indices of the file (a file of another radius has other boxes there).
    wrapped_axes = (
        np.arange(int(grid.periodicity))
        if _is_confined(grid)
        else (np.arange(3) if periodic else np.arange(0))
    )
    grid_points = fortran_integer_coordinates(grid)

    def wrap(values: np.ndarray) -> np.ndarray:
        values = values.copy()
        for axis in wrapped_axes:
            values[:, axis] = (values[:, axis] + counts[axis] // 2) % counts[axis] - counts[axis] // 2
        return values

    if periodic:
        file_points = wrap(file_points)
    low = file_points.min(axis=0) if file_points.size else np.zeros(3, dtype=np.int64)
    high = file_points.max(axis=0) if file_points.size else np.zeros(3, dtype=np.int64)
    lookup = np.full(tuple(int(v) for v in high - low + 1), -1, dtype=np.int64)
    local = file_points - low
    lookup[local[:, 0], local[:, 1], local[:, 2]] = np.arange(file_points.shape[0])

    def find(integers: np.ndarray) -> np.ndarray:
        if periodic:
            integers = wrap(integers)
        offsets = integers - low
        inside = np.all((offsets >= 0) & (offsets < np.asarray(lookup.shape)), axis=1)
        found = np.full(integers.shape[0], -1, dtype=np.int64)
        if np.any(inside):
            hit = offsets[inside]
            found[inside] = lookup[hit[:, 0], hit[:, 1], hit[:, 2]]
        return found

    # A general periodic grid steps along the unit lattice vectors (the
    # columns of A_n, Fortran avec_norm): r = A_n ((k + shift) step), and the
    # Cartesian operations of the file act on r (grid_partition.f90:146-151).
    normalized = (
        np.asarray(grid.normalized_lattice, dtype=float) if _is_general_periodic(grid) else None
    )
    if normalized is None:
        positions = (grid_points + shift) * steps
    else:
        positions = ((grid_points + shift) * steps) @ normalized.T
        to_lattice = np.linalg.inv(normalized).T
    point_of_row = np.full(grid_points.shape[0], -1, dtype=np.int64)
    remaining = np.arange(grid_points.shape[0])
    for operation in range(data.symmetry.ntrans):
        if remaining.size == 0:
            break
        moved = data.symmetry.apply(operation, positions[remaining])
        if normalized is None:
            integers = np.rint(moved / steps - shift).astype(np.int64)
            off = np.linalg.norm(moved - (integers + shift) * steps, axis=1)
        else:
            lattice = moved @ to_lattice
            integers = np.rint(lattice / steps - shift).astype(np.int64)
            off = np.linalg.norm(
                (lattice - (integers + shift) * steps) @ normalized.T, axis=1
            )
        if np.any(off > tolerance):
            raise ValueError(
                "a symmetry operation of parsec.dat moves grid points off the "
                f"grid (by up to {float(np.max(off)):.3g} bohr): the file belongs "
                "to another grid"
            )
        found = find(integers)
        hit = found >= 0
        point_of_row[remaining[hit]] = found[hit]
        remaining = remaining[~hit]
    dropped = 0
    if data.symmetry.ntrans == 1:
        used = np.zeros(file_points.shape[0], dtype=bool)
        used[point_of_row[point_of_row >= 0]] = True
        dropped = int(np.count_nonzero(~used))
    return GridMapping(point_of_row=point_of_row, dropped=dropped)


# ----------------------------------------------------------------------------
# From a Python result
# ----------------------------------------------------------------------------


def highest_occupied(occupations_by_k: Sequence[np.ndarray]) -> int:
    """``elec_st%ifmax`` of one spin channel (``flevel.f90``).

    The last state with a non-zero occupation, as the loop over k-points
    leaves it: the value of the last k-point that has one.
    """

    highest = 0
    for occupations in occupations_by_k:
        occupied = np.flatnonzero(np.asarray(occupations) > 0.0)
        if occupied.size:
            highest = int(occupied[-1]) + 1
    return highest


def selected_states(selection, occupations_by_k: Sequence[np.ndarray], state_count: int) -> np.ndarray:
    """``indxsave`` of one spin channel for a :class:`WavefunctionSelection`."""

    if selection.mode == "occupied":
        return np.arange(1, highest_occupied(occupations_by_k) + 1, dtype=np.int32)
    if selection.mode == "all":
        return np.arange(1, state_count + 1, dtype=np.int32)
    numbers = np.asarray(selection.indices, dtype=np.int32)
    if numbers.size and int(numbers.max()) > state_count:
        raise ValueError(
            f"Save_Wfn_Bands asks for state {int(numbers.max())}, but the "
            f"calculation has {state_count} states"
        )
    return numbers


@dataclass(frozen=True)
class ResultChannels:
    """What ``parsec.dat`` takes of a result, per spin channel.

    ``eigenvalues[s]``/``occupations[s]`` are k-major pooled arrays of
    channel ``s``; ``wavefunctions[s]`` the full-grid orbitals of the
    channel (columns follow the states of the first k-point), or ``None``
    where the result keeps none; ``potentials[s]`` the potential the next
    iteration would take (``vnew``) and ``densities[s]`` the density of the
    channel (the total density for one channel).
    """

    eigenvalues: tuple[np.ndarray, ...]
    occupations: tuple[np.ndarray, ...]
    wavefunctions: tuple[object | None, ...]
    potentials: tuple[np.ndarray, ...]
    densities: tuple[np.ndarray, ...]
    notes: tuple[str, ...] = ()
    # stype(2) of a spinor result (2 spin-orbit, 3 non-collinear, 4 both):
    # one set of levels and spinor wavefunctions, two potentials and densities.
    wavefunction_kind: int | None = None


def result_channels(result) -> ResultChannels:
    """The fields of ``parsec.dat`` from a Python SCF result.

    A :class:`parsec_python.models.SinglePointResult` gives its mixed
    potential (``next_effective_potential``, Fortran ``vnew``), with two
    channels where ``result.spin`` holds them.  A
    :class:`parsec_python.models.SpinPolarizedSinglePointResult` keeps no
    mixed potential: the output potentials of its last iteration
    (``scf_potentials``) take its place, which agree with ``vnew`` to the
    SCF criterion.
    """

    def require(owner, names) -> None:
        missing = [name for name in names if getattr(owner, name, None) is None]
        if missing:
            raise ValueError(
                f"this result keeps no {', '.join(missing)}, which parsec.dat needs"
            )

    def host_orbitals(wavefunctions):
        # Orbitals on a device or in symmetry sectors are not read here: the
        # caller brings them to the host first where it wants them.
        if (
            isinstance(wavefunctions, np.ndarray)
            and wavefunctions.ndim == 2
            and wavefunctions.shape[1] > 0
        ):
            return wavefunctions
        return None

    if hasattr(result, "spinors"):
        # A spin-orbit or non-collinear result (wfnsave.F90 with mxwd = 2).
        fields = getattr(result, "parsec_dat_fields", None)
        if not fields:
            raise ValueError(
                "this spin-orbit result keeps no potentials of its spin "
                "components, which parsec.dat needs"
            )
        has_k = getattr(result, "k_points", None) is not None
        spinors = getattr(result, "spinors", None)
        return ResultChannels(
            eigenvalues=(np.asarray(result.eigenvalues, dtype=float),),
            occupations=(np.asarray(result.occupations, dtype=float),),
            wavefunctions=(None if has_k else host_orbitals(spinors),),
            potentials=tuple(np.asarray(value, dtype=float) for value in fields["potentials"]),
            densities=tuple(np.asarray(value, dtype=float) for value in fields["densities"]),
            wavefunction_kind=int(fields["kind"]),
        )
    channels = getattr(result, "spin", None)
    if channels is not None:
        for channel in channels:
            require(channel, ("eigenvalues", "occupations", "next_effective_potential", "density"))
        return ResultChannels(
            eigenvalues=tuple(np.asarray(c.eigenvalues, dtype=float) for c in channels),
            occupations=tuple(np.asarray(c.occupations, dtype=float) for c in channels),
            wavefunctions=tuple(host_orbitals(c.wavefunctions) for c in channels),
            potentials=tuple(np.asarray(c.next_effective_potential, dtype=float) for c in channels),
            densities=tuple(np.asarray(c.density, dtype=float) for c in channels),
        )
    if hasattr(result, "eigenvalues_up"):
        potentials = getattr(result, "scf_potentials", None) or {}
        if "potential_up" not in potentials or "potential_down" not in potentials:
            raise ValueError(
                "this spin-polarized result keeps no potentials of its channels, "
                "which parsec.dat needs"
            )
        has_k = getattr(result, "k_points", None) is not None
        return ResultChannels(
            eigenvalues=(np.asarray(result.eigenvalues_up, dtype=float), np.asarray(result.eigenvalues_down, dtype=float)),
            occupations=(np.asarray(result.occupations_up, dtype=float), np.asarray(result.occupations_down, dtype=float)),
            wavefunctions=(
                None if has_k else host_orbitals(result.wavefunctions_up),
                None if has_k else host_orbitals(result.wavefunctions_down),
            ),
            potentials=(
                np.asarray(potentials["potential_up"], dtype=float),
                np.asarray(potentials["potential_down"], dtype=float),
            ),
            densities=(np.asarray(result.density_up, dtype=float), np.asarray(result.density_down, dtype=float)),
            notes=(
                "the potentials of parsec.dat are the output potentials of the last "
                "SCF iteration (this driver keeps no mixed potential)",
            ),
        )
    require(result, ("eigenvalues", "occupations", "next_effective_potential", "density"))
    has_k = getattr(result, "k_points", None) is not None
    return ResultChannels(
        eigenvalues=(np.asarray(result.eigenvalues, dtype=float),),
        occupations=(np.asarray(result.occupations, dtype=float),),
        wavefunctions=(None if has_k else host_orbitals(getattr(result, "wavefunctions", None)),),
        potentials=(np.asarray(result.next_effective_potential, dtype=float),),
        densities=(np.asarray(result.density, dtype=float),),
    )


def parsec_dat_from_result(
    result,
    grid,
    selection,
    *,
    periodic_cell=None,
    k_points: np.ndarray | None = None,
    k_weights: np.ndarray | None = None,
    monkhorst_pack_grid: Sequence[int] | None = None,
    date_label: str | None = None,
    monkhorst_pack_shift: Sequence[float] | None = None,
) -> tuple[ParsecDat, tuple[str, ...]]:
    """Build the contents of ``parsec.dat`` from a Python result on ``grid``.

    Returns the data and notes for the run log (wavefunctions that the
    result does not keep).  The file has one symmetry operation (the
    identity), ``nwedge = ndim`` and representation 1 for every state.
    """

    channels = result_channels(result)
    notes = list(channels.notes)
    spin_count = len(channels.potentials)
    k_points = getattr(result, "k_points", None) if k_points is None else k_points
    k_weights = getattr(result, "k_weights", None) if k_weights is None else k_weights
    kpoint_count = 1 if k_points is None else int(np.asarray(k_points).shape[0])
    complex_kind = (
        channels.wavefunction_kind
        if channels.wavefunction_kind is not None
        else (1 if k_points is not None else 0)
    )
    # One set of levels per spin channel, one for the spinors (nspin/mxwd).
    level_channels = len(channels.eigenvalues)
    spinor_rows = 2 if channels.wavefunction_kind is not None else 1
    periodic = _is_periodic(grid)
    boundary = boundary_code(grid)
    points = fortran_integer_coordinates(grid)
    rmax = grid_rmax(grid)
    steps = grid_steps(grid)
    if _is_confined(grid):
        # symm%alatt of a wire or slab: 4 rmax along the confined directions
        # (symmetries.f90:114-118).
        from ..Grid.space_group import confined_symmetry_lattice

        vectors = np.asarray((periodic_cell or grid.cell).lattice_vectors, dtype=float)
        alatt = confined_symmetry_lattice(vectors, grid.periodicity, rmax).T.copy()
    elif periodic:
        vectors = np.asarray(
            (periodic_cell or grid.cell).lattice_vectors, dtype=float
        )
        alatt = vectors.T.copy()
    else:
        alatt = cluster_alatt(rmax)
    symmetry = ParsecDatSymmetry.identity(alatt)
    levels = []
    wavefunctions = []
    for spin in range(level_channels):
        eigenvalues = channels.eigenvalues[spin]
        occupations = channels.occupations[spin]
        per_k = eigenvalues.size // kpoint_count
        blocks = []
        occupations_by_k = []
        for k in range(kpoint_count):
            part = slice(k * per_k, (k + 1) * per_k)
            order = np.argsort(eigenvalues[part], kind="stable")
            occupations_by_k.append(occupations[part][order])
            blocks.append(
                ParsecDatLevels(
                    representations=np.ones(per_k, dtype=np.int32),
                    energies=eigenvalues[part][order],
                    occupations=occupations[part][order],
                )
            )
        levels.append(tuple(blocks))
        numbers = selected_states(selection, occupations_by_k, per_k)
        orbitals = channels.wavefunctions[spin]
        spin_blocks = []
        for k in range(kpoint_count):
            chosen = numbers
            if chosen.size and orbitals is None:
                notes.append(
                    f"the wavefunctions of {chosen.size} state(s) are not in parsec.dat: "
                    "this calculation does not keep them"
                    + (" (k-point drivers keep no orbitals)" if k_points is not None else "")
                )
                chosen = np.zeros(0, dtype=np.int32)
            if chosen.size:
                order = np.argsort(eigenvalues[:per_k], kind="stable")
                columns = np.asarray(orbitals)[:, order[chosen - 1]]
                vectors = np.asarray(columns, dtype=np.complex128 if complex_kind else np.float64)
            else:
                vectors = np.zeros(
                    (points.shape[0] * spinor_rows, 0),
                    dtype=np.complex128 if complex_kind else np.float64,
                )
            spin_blocks.append(ParsecDatWavefunctions(symmetry, points, chosen.astype(np.int32), vectors))
        wavefunctions.append(tuple(spin_blocks))
    cell = None
    if periodic:
        lattice = np.asarray((periodic_cell or grid.cell).lattice_vectors, dtype=float)
        if k_points is None:
            cell_k, cell_w, method = np.zeros((1, 3)), np.ones(1), 0
        else:
            cell_k = np.asarray(k_points, dtype=float)
            cell_w = (
                np.full(kpoint_count, 1.0 / kpoint_count)
                if k_weights is None
                else np.asarray(k_weights, dtype=float)
            )
            method = KPOINT_MONKHORST_PACK if monkhorst_pack_grid is not None else KPOINT_MANUAL
        directions = {}
        if _is_general_periodic(grid):
            # grid%lap_dir_num, lap_dir, lap_dir_step of pbc_grid_coefs (the
            # auxiliary Laplacian directions); restart_run.F90 does not read them.
            from ..Laplacian.nonorthogonal import grid_laplacian_coefficients

            coefficients = grid_laplacian_coefficients(grid)
            directions = {
                "lap_dir_num": int(coefficients.number_of_directions),
                "lap_dir": np.asarray(coefficients.active, dtype=np.int32),
                "lap_dir_step": np.asarray(coefficients.steps, dtype=float),
            }
        box_size = np.linalg.norm(lattice, axis=1)
        if _is_confined(grid):
            # usrinputfile.F90:852-871: zero along the confined directions.
            box_size[int(grid.periodicity) :] = 0.0
        cell = ParsecDatPeriodic(
            box_size=box_size,
            lattice_vectors=lattice,
            **directions,
            kpoint_method=method,
            monkhorst_pack_grid=np.asarray(
                monkhorst_pack_grid if monkhorst_pack_grid is not None else (0, 0, 0),
                dtype=np.int32,
            ),
            # Fortran's mpshift where the input gives Monkhorst_Pack_Shift.
            **(
                {}
                if monkhorst_pack_shift is None
                else {"monkhorst_pack_shift": np.asarray(monkhorst_pack_shift, dtype=float)}
            ),
            k_points=cell_k,
            k_weights=cell_w,
        )
    data = ParsecDat(
        date_label=date_label or fortran_date_label(),
        system_type=system_type(spin_count, complex_kind, boundary),
        step=steps,
        rmax=rmax,
        ndim=int(grid.size),
        shift=grid_shift(grid),
        symmetry=symmetry,
        points=points,
        levels=tuple(levels),
        potentials=channels.potentials,
        densities=channels.densities,
        wavefunctions=tuple(wavefunctions),
        periodic=cell,
    )
    return data, tuple(dict.fromkeys(notes))


def wavefunction_bytes(grid_size: int, state_count: int, spin_count: int, complex_values: bool) -> int:
    """Bytes the wavefunction records of ``state_count`` states would take."""

    return int(grid_size) * int(state_count) * int(spin_count) * (16 if complex_values else 8)


def iter_records(path: str | Path) -> Iterator[bytes]:
    """Every record of a sequential unformatted file, in order (diagnostics)."""

    with Path(path).open("rb") as stream:
        while True:
            try:
                yield read_record(stream)
            except EOFError:
                return


__all__ = [
    "BOUNDARY_CODES",
    "GridMapping",
    "KPOINT_MANUAL",
    "KPOINT_MONKHORST_PACK",
    "ParsecDat",
    "ParsecDatLevels",
    "ParsecDatPeriodic",
    "ParsecDatSymmetry",
    "ParsecDatWavefunctions",
    "ResultChannels",
    "cluster_alatt",
    "fortran_date_label",
    "fortran_integer_coordinates",
    "grid_rmax",
    "grid_rows_of_file",
    "grid_shift",
    "grid_steps",
    "highest_occupied",
    "iter_records",
    "parsec_dat_from_result",
    "read_parsec_dat",
    "read_record",
    "result_channels",
    "selected_states",
    "system_type",
    "wavefunction_bytes",
    "write_parsec_dat",
    "write_record",
]
