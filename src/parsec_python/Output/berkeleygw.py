"""``Output_GW``: the WFN, RHO and VXC files of BerkeleyGW (Fortran ``gw_write.F90``).

PARSEC writes, after the SCF or the non-self-consistent run
(``parsec.F90`` l.2095-2147, ``#ifdef GW``), three Fortran unformatted
sequential files (4-byte record markers) in the layout of BerkeleyGW's
``write_binary_header`` (``wfn_rho_vxc_io_m``, complex flavour, ``iflag = 2``)
followed by records PARSEC writes itself:

* ``VXC`` and ``RHO``: header, the G-vectors of the potential sphere
  (``1`` / ``ng`` / ``gvec(3, ng)``), then ``1`` / ``ng`` / the coefficients
  (VXC: ``ng`` values per spin channel, Ry; RHO: the density times the cell
  volume, so ``rho(G = 0)`` is the electron count);
* ``WFN``: header with the k-points (crystal coordinates), weights, the
  lowest/highest band with occupation above 0.5, energies (Ry) and
  occupations (0..1), the potential G-vectors, then per k-point its sphere
  ``|k + G|^2 <= E_cut`` (``1`` / ``ngk`` / ``gvec``) and per band ``1`` /
  ``ngk`` / ``c(G)`` of every spin channel, normalized to one per channel.

How PARSEC fills them (kept here):

* ``E_cut = min_i (pi/h_i)^2 - 1e-4`` Ry, the potential sphere has radius
  ``2 sqrt(E_cut)`` and its FFT grid ``2 kmax + 1`` raised by ``adjustfft``
  (:mod:`parsec_python.Grid.gspace`);
* a field on the real-space grid is laid on the ``n1 x n2 x n3`` FFT box with
  the box corner (Fortran's first grid point ``-n_i/2``) as origin and
  transformed forward with ``1/N`` (``cfftw``); no phase for the position of
  the corner is applied, so every field of the files is translated by the
  corner (PARSEC's convention: the atom positions and the fractional
  translations in the header are those of the input frame); this port
  keeps it where the translation by the corner leaves every operation's
  translation unchanged modulo the lattice (fcc Si on its 10^3 grid: the
  corner is ``-(a1+a2+a3)/2``) and otherwise refers the coefficients to the
  cell origin (a deviation: PARSEC's files would contradict the symmetry
  operations of their own header, e.g. on the shift-1/2 grid of an
  orthogonal cell; the orthorhombic grid of this port starts at the origin);
* V_xc and the density are averaged over the symmetry stars of
  ``g_space`` (each coefficient ``(Re, conj Im)/m_star`` times the phase,
  summed per star, then times the phase of the vector again);
* a wavefunction coefficient is the FFT value times ``sqrt(N)`` for every G of
  the k-sphere inside the box ``|g_i| <= kmax_i`` of ``g_space`` (aliasing
  as PARSEC: ``+-n/2`` share one FFT index), zero outside, then the vector
  is scaled by ``1/norm`` per spin channel.

Deviations (both affect spin-polarized files only, where PARSEC's records do
not have BerkeleyGW's layout and BerkeleyGW cannot read them): the band
records give ``ngk`` (PARSEC: ``ngk * nspin``) as their G-vector count, and
RHO holds the density of each spin channel (PARSEC: the total density alone
under a two-channel header).  Spin-unpolarized files have PARSEC's records
exactly.

The numerical functions do not write; :func:`write_berkeleygw_file` and
:func:`read_berkeleygw_file` do the file I/O.  Units: bohr, Ry.
"""

from __future__ import annotations

import datetime as _datetime
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from ..Grid.gspace import (
    TWO_PI,
    GSphere,
    GVectorStars,
    LatticeMetrics,
    box_half_widths,
    fortran_lattice_metrics,
    g_space_stars,
    k_gspace,
    potential_gspace,
    wavefunction_cutoff,
)

#: BerkeleyGW's ``CFLAVOR``: the files are written in the complex flavour.
COMPLEX_FLAVOR = "Complex"
#: An occupation above this marks an occupied band for ``ifmin``/``ifmax``.
OCCUPIED_THRESHOLD = 0.5
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class BerkeleyGWHeader:
    """The header of a WFN/RHO/VXC file (``write_binary_header`` arguments).

    ``kind`` is ``"WFN"``, ``"RHO"`` or ``"VXC"``; ``date`` (11 characters)
    and ``time`` (14) as BerkeleyGW's ``date_time``.  ``lattice`` and
    ``reciprocal`` hold the vectors as columns (``avec``, ``bvec`` with
    ``alat = blat = 1``, bohr and bohr^-1), ``rotations`` ``(ntran, 3, 3)``
    the integer matrices acting on G (``mtrx`` = PARSEC's ``gmtrx``),
    ``translations`` ``(ntran, 3)`` the fractional translations times
    ``2 pi`` (``tnp``), ``atom_positions`` ``(nat, 3)`` Cartesian bohr and
    ``atom_types`` (1-based species).  The WFN fields (``None`` otherwise):
    ``k_points`` ``(nk, 3)`` crystal, ``k_weights``, ``k_gvector_counts``
    (``ngk``), ``lowest_occupied``/``highest_occupied`` ``(ns, nk)``
    (``ifmin``/``ifmax``, 1-based, 0 when none), ``energies`` and
    ``occupations`` ``(ns, nk, nbands)``.
    """

    kind: str
    spin_count: int
    gvector_count: int
    cell_symmetry: int
    ecutrho: float
    ecutwfc: float
    fft_grid: tuple[int, int, int]
    kgrid: tuple[int, int, int]
    kshift: tuple[float, float, float]
    cell_volume: float
    lattice: np.ndarray
    adot: np.ndarray
    reciprocal_volume: float
    reciprocal: np.ndarray
    bdot: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    atom_positions: np.ndarray
    atom_types: np.ndarray
    date: str = ""
    time: str = ""
    flavor: str = COMPLEX_FLAVOR
    alat: float = 1.0
    blat: float = 1.0
    band_count: int = 0
    max_k_gvectors: int = 0
    k_points: np.ndarray | None = None
    k_weights: np.ndarray | None = None
    k_gvector_counts: np.ndarray | None = None
    lowest_occupied: np.ndarray | None = None
    highest_occupied: np.ndarray | None = None
    energies: np.ndarray | None = None
    occupations: np.ndarray | None = None

    @property
    def is_wavefunction(self) -> bool:
        return self.kind == "WFN"


@dataclass(frozen=True)
class BerkeleyGWFile:
    """What :func:`read_berkeleygw_file` returns.

    ``gvectors`` ``(ng, 3)``; RHO/VXC: ``data`` ``(ns, ng)``; WFN:
    ``k_gvectors[k]`` ``(ngk, 3)`` and ``bands[k]`` ``(nbands, ns, ngk)``.
    ``record_counts`` are the G-vector counts written before every data
    record (for a check of the layout).
    """

    header: BerkeleyGWHeader
    gvectors: np.ndarray
    data: np.ndarray | None = None
    k_gvectors: tuple[np.ndarray, ...] = ()
    bands: tuple[np.ndarray, ...] = ()
    record_counts: tuple[int, ...] = ()


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


def crystal_k_points(k_points, lattice_vectors) -> np.ndarray:
    """``latt_vec^T k / 2 pi`` of ``gw_write.F90`` l.141-144 (rows ``a_i``, bohr)."""

    rows = np.asarray(lattice_vectors, dtype=np.float64)
    values = np.asarray(k_points, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(values)
    for axis in range(3):
        a = rows[axis]
        out[:, axis] = (
            ((0.0 + a[0] * values[:, 0]) + a[1] * values[:, 1]) + a[2] * values[:, 2]
        ) / TWO_PI
    return out


def occupied_band_range(occupations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``ifmin``/``ifmax`` (1-based, 0 if none) of ``(ns, nk, nbands)`` occupations."""

    occupations = np.asarray(occupations, dtype=np.float64)
    occupied = occupations > OCCUPIED_THRESHOLD
    any_occupied = occupied.any(axis=2)
    first = np.where(any_occupied, np.argmax(occupied, axis=2) + 1, 0)
    last = np.where(
        any_occupied,
        occupations.shape[2] - np.argmax(occupied[:, :, ::-1], axis=2),
        0,
    )
    return first.astype(np.int32), last.astype(np.int32)


def fft_box_origin(grid) -> tuple[int, int, int]:
    """The integer point of ``grid`` that PARSEC's FFT box starts at.

    Fortran's general grid numbers its points ``-n_i/2 .. n_i/2 - 1``
    (``pbc%mx = -n1/2``); this port's :class:`GeneralPeriodicGrid` numbers
    them ``0 .. n_i - 1`` modulo the lattice, so the box starts at
    ``-n_i/2``.  The orthorhombic grid of this port has its first point at
    the origin of the cell, which is the box corner.
    """

    counts = [int(value) for value in grid.points_per_axis]
    if getattr(grid, "is_general_periodic", False):
        return tuple(-(n // 2) for n in counts)
    return (0, 0, 0)


def fft_box(values: np.ndarray, grid, origin=None) -> np.ndarray:
    """``values`` on the ``(n1, n2, n3)`` FFT box: ``box[i, j, k]`` = point ``origin + (i, j, k)``.

    The first index runs along ``a_1`` (Fortran's fastest index of the box);
    ``values`` are ordered as the grid's points (row-major, ``a_1`` slowest).
    """

    counts = tuple(int(value) for value in grid.points_per_axis)
    origin = fft_box_origin(grid) if origin is None else tuple(int(v) for v in origin)
    array = np.asarray(values).reshape(counts)
    return np.roll(array, shift=tuple(-o for o in origin), axis=(0, 1, 2))


def forward_transform(box: np.ndarray) -> np.ndarray:
    """``cfftw(..., 1)``: ``F(g) = (1/N) sum_r f(r) exp(-2 pi i g.r/n)``."""

    return np.fft.fftn(np.asarray(box, dtype=np.complex128)) / float(box.size)


def _box_lookup(vectors: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT indices ``(g mod n)`` of integer vectors (``1 + g``, ``n + i`` if ``<= 0``)."""

    vectors = np.asarray(vectors, dtype=np.int64)
    return tuple(vectors[:, axis] % int(shape[axis]) for axis in range(3))


def box_corner(grid, origin=None) -> np.ndarray:
    """Lattice coordinates of the FFT box corner: ``(origin_i + shift_i)/n_i``."""

    counts = np.asarray(grid.points_per_axis, dtype=np.float64)
    origin = np.asarray(fft_box_origin(grid) if origin is None else origin, dtype=np.float64)
    shift = np.asarray(getattr(grid, "shift", (0.0, 0.0, 0.0)), dtype=np.float64)
    return (origin + shift) / counts


def corner_is_symmetric(corner, space_group, tolerance: float = 1.0e-8) -> bool:
    """Whether the fields translated by ``corner`` keep the header's translations.

    In the frame of the box corner ``d`` an operation ``T(f) = M (f - t)``
    becomes ``M (f' - t')`` with ``t' = t - d + M^-1 d``; the header (input
    frame) stays valid when ``(M^-1 - 1) d`` is a lattice vector for every
    operation.  True for fcc Si on its 10^3 grid (corner ``-(a1+a2+a3)/2``).
    """

    d = np.asarray(corner, dtype=np.float64)
    for matrix in space_group.rotation_matrices:
        change = np.linalg.solve(matrix.astype(np.float64), d) - d
        if np.any(np.abs(change - np.round(change)) > tolerance):
            return False
    return True


def corner_phases(vectors: np.ndarray, corner) -> np.ndarray:
    """``exp(-2 pi i g.d)``: refers coefficients of the box frame to the cell origin."""

    d = np.asarray(corner, dtype=np.float64)
    return np.exp(-2j * np.pi * (np.asarray(vectors, dtype=np.float64) @ d))


def star_averaged_coefficients(
    transform: np.ndarray, stars: GVectorStars, sphere: GSphere, corner=None
) -> np.ndarray:
    """The coefficients of one field on ``sphere`` after the star average.

    ``gw_write.F90`` l.326-371: for every vector of ``stars`` the FFT value
    with its imaginary part times ``conjugation``, over the star size, times
    the phase, summed per star (in Fortran's order); then every vector of the
    sphere that is also in the box gets its star's sum times its phase.  The
    rest of the sphere is zero.  With ``corner`` (lattice coordinates of the
    box corner) the FFT values are first referred to the cell origin
    (:func:`corner_phases`); PARSEC applies no such phase.
    """

    index = _box_lookup(stars.vectors, transform.shape)
    values = transform[index]
    if corner is not None:
        values = values * corner_phases(stars.vectors, corner)
    sizes = stars.star_sizes[stars.star_index].astype(np.float64)
    scaled = (values.real + 1j * (stars.conjugation * values.imag)) / sizes
    terms = scaled * stars.phases
    sums = np.zeros(stars.star_count, dtype=np.complex128)
    np.add.at(sums, stars.star_index, terms)
    lookup = _sphere_lookup(sphere)
    k = np.asarray(sphere.kmax)
    offsets = stars.vectors + k
    inside_box = np.all((offsets >= 0) & (offsets <= 2 * k), axis=1)
    out = np.zeros(sphere.count, dtype=np.complex128)
    rows = np.full(stars.count, -1, dtype=np.int64)
    rows[inside_box] = lookup[tuple(offsets[inside_box].T)]
    kept = rows >= 0
    out[rows[kept]] = sums[stars.star_index[kept]] * stars.phases[kept]
    return out


def _sphere_lookup(sphere: GSphere) -> np.ndarray:
    """``indexg``: a ``(2kmax+1)^3`` array of sphere rows (``-1`` outside)."""

    k = np.asarray(sphere.kmax)
    lookup = np.full(tuple(2 * k + 1), -1, dtype=np.int64)
    offsets = sphere.vectors + k
    lookup[tuple(offsets.T)] = np.arange(sphere.count)
    return lookup


def wavefunction_coefficients(
    transform: np.ndarray, sphere: GSphere, box_kmax, corner=None
) -> np.ndarray:
    """One band of one spin channel on the k-sphere (``gw_write.F90`` l.547-577).

    ``transform`` is the forward FFT (``1/N``) of ``u_nk`` on the box; the
    vectors of ``sphere`` inside ``|g_i| <= box_kmax_i`` take ``sqrt(N)``
    times their FFT value, the others zero; the result is multiplied by
    ``1/norm`` (``zscal(ONE/norm)``).  A vector of zero norm stays zero.
    ``corner``: as in :func:`star_averaged_coefficients`.
    """

    factor = math.sqrt(float(transform.size))
    vectors = sphere.vectors
    limit = np.asarray(box_kmax, dtype=np.int64)
    inside = np.all(np.abs(vectors) <= limit, axis=1)
    out = np.zeros(sphere.count, dtype=np.complex128)
    index = _box_lookup(vectors[inside], transform.shape)
    out[inside] = transform[index] * factor
    if corner is not None:
        out[inside] *= corner_phases(vectors[inside], corner)
    norm = float(np.sqrt(np.sum(out.real**2 + out.imag**2)))
    if norm > 0.0:
        out = out * (1.0 / norm)
    return out


@dataclass(frozen=True)
class BerkeleyGWSetup:
    """The reciprocal-space data of one ``gw_write`` call.

    ``cutoff`` is the wavefunction cutoff (Ry), ``metrics`` the lattice data,
    ``stars`` the box vectors of ``g_space`` in stars, ``box_kmax`` their
    half-widths, ``potential`` the potential sphere (with its FFT grid),
    ``k_spheres`` one sphere per k-point (crystal ``k_points``),
    ``box_origin`` the grid point at the FFT box corner and ``cell_volume``
    PARSEC's ``vcell``.  ``corner`` is the box corner in lattice
    coordinates when the coefficients are referred to the cell origin (a
    corner that does not keep the header's fractional translations, see
    :func:`corner_is_symmetric`), ``None`` for PARSEC's convention.
    """

    cutoff: float
    metrics: LatticeMetrics
    stars: GVectorStars
    box_kmax: tuple[int, int, int]
    potential: GSphere
    k_points: np.ndarray
    k_spheres: tuple[GSphere, ...]
    box_origin: tuple[int, int, int]
    cell_volume: float
    corner: np.ndarray | None = None


def grid_steps(grid) -> np.ndarray:
    """``grid%step``: the three steps along the lattice directions (bohr)."""

    if getattr(grid, "is_general_periodic", False):
        return np.asarray(grid.spacings, dtype=np.float64)
    lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
    return np.diag(lattice) / np.asarray(grid.points_per_axis, dtype=np.float64)


def berkeleygw_setup(
    grid,
    space_group,
    k_points,
    *,
    cell_volume: float,
) -> BerkeleyGWSetup:
    """G-spaces of ``gw_write`` for a periodic ``grid`` and Cartesian ``k_points``.

    ``space_group`` is the :class:`parsec_python.Grid.space_group.SpaceGroup`
    PARSEC passes (``symm``: the full group, or the identity with
    ``Ignore_Symmetry``); ``cell_volume`` is ``pbc%vcell``
    (:func:`parsec_python.Output.dos.fortran_cell_volume`).
    """

    lattice_vectors = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
    metrics = fortran_lattice_metrics(lattice_vectors)
    cutoff = wavefunction_cutoff(grid_steps(grid))
    box_kmax = box_half_widths(grid.points_per_axis)
    translations = space_group.translation_vectors
    stars = g_space_stars(box_kmax, metrics.bdot, space_group.reciprocal_rotations, translations)
    potential = potential_gspace(cutoff, metrics, translations)
    crystal = crystal_k_points(k_points, lattice_vectors)
    spheres = tuple(k_gspace(cutoff, k, metrics) for k in crystal)
    origin = fft_box_origin(grid)
    corner = box_corner(grid, origin)
    return BerkeleyGWSetup(
        cutoff=cutoff,
        metrics=metrics,
        stars=stars,
        box_kmax=stars.kmax,
        potential=potential,
        k_points=crystal,
        k_spheres=spheres,
        box_origin=origin,
        cell_volume=float(cell_volume),
        corner=None if corner_is_symmetric(corner, space_group) else corner,
    )


def field_coefficients(setup: BerkeleyGWSetup, grid, fields: Sequence[np.ndarray], *, scale: float = 1.0) -> np.ndarray:
    """``(len(fields), ng)``: each real field star-averaged on the potential sphere.

    ``scale`` multiplies the field first (``pbc%vcell`` for the density).
    """

    out = []
    for field in fields:
        values = np.asarray(field, dtype=np.float64)
        if scale != 1.0:
            values = values * scale
        transform = forward_transform(fft_box(values, grid, setup.box_origin))
        out.append(star_averaged_coefficients(transform, setup.stars, setup.potential, setup.corner))
    return np.asarray(out)


def band_coefficients(setup: BerkeleyGWSetup, grid, k_index: int, orbitals: Sequence[np.ndarray]) -> np.ndarray:
    """``(ns, ngk)``: one band's ``u_nk`` of every spin channel on the k-sphere."""

    sphere = setup.k_spheres[k_index]
    out = []
    for orbital in orbitals:
        transform = forward_transform(fft_box(orbital, grid, setup.box_origin))
        out.append(wavefunction_coefficients(transform, sphere, setup.box_kmax, setup.corner))
    return np.asarray(out)


def berkeleygw_header(
    kind: str,
    setup: BerkeleyGWSetup,
    space_group,
    atoms,
    *,
    spin_count: int,
    kgrid=(0, 0, 0),
    kshift=(0.0, 0.0, 0.0),
    k_weights=None,
    energies=None,
    occupations=None,
    stamp: tuple[str, str] | None = None,
) -> BerkeleyGWHeader:
    """The header PARSEC passes to ``write_binary_header`` (``gw_write.F90`` l.239-247).

    ``atoms`` in PARSEC's order (by species), positions Cartesian bohr;
    ``energies``/``occupations`` ``(ns, nk, nbands)`` for every file (they
    are written in WFN only).  ``stamp`` is ``(date, time)``; the current
    local time by default.
    """

    if kind not in ("WFN", "RHO", "VXC"):
        raise ValueError("kind must be WFN, RHO or VXC")
    date, time = stamp if stamp is not None else berkeleygw_date_time()
    species: list[str] = []
    for atom in atoms:
        if atom.symbol not in species:
            species.append(atom.symbol)
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float64).reshape(-1, 3)
    types = np.asarray([species.index(atom.symbol) + 1 for atom in atoms], dtype=np.int32)
    metrics = setup.metrics
    common = dict(
        kind=kind,
        spin_count=int(spin_count),
        gvector_count=setup.potential.count,
        cell_symmetry=space_group_cell_symmetry(space_group),
        ecutrho=4.0 * setup.cutoff,
        ecutwfc=setup.cutoff,
        fft_grid=tuple(int(v) for v in setup.potential.fft_size),
        kgrid=tuple(int(v) for v in kgrid),
        kshift=tuple(float(v) for v in kshift),
        cell_volume=setup.cell_volume,
        lattice=metrics.lattice,
        adot=metrics.adot,
        reciprocal_volume=TWO_PI**3 / setup.cell_volume,
        reciprocal=metrics.reciprocal,
        bdot=metrics.bdot,
        rotations=np.asarray(space_group.reciprocal_rotations, dtype=np.int32),
        translations=space_group.translation_vectors * TWO_PI,
        atom_positions=positions,
        atom_types=types,
        date=date,
        time=time,
    )
    if kind != "WFN":
        return BerkeleyGWHeader(**common)
    energies = np.asarray(energies, dtype=np.float64)
    occupations = np.asarray(occupations, dtype=np.float64)
    first, last = occupied_band_range(occupations)
    counts = np.asarray([sphere.count for sphere in setup.k_spheres], dtype=np.int32)
    return BerkeleyGWHeader(
        **common,
        band_count=int(energies.shape[2]),
        max_k_gvectors=int(counts.max()),
        k_points=setup.k_points,
        k_weights=np.asarray(k_weights, dtype=np.float64),
        k_gvector_counts=counts,
        lowest_occupied=first,
        highest_occupied=last,
        energies=energies,
        occupations=occupations,
    )


def space_group_cell_symmetry(space_group) -> int:
    """``symm%cell_symmetry`` (``symgen.f90`` l.155): 1 hexagonal candidates, 0 cubic.

    The identity group of ``Ignore_Symmetry`` (no ``symgen``) gives 0.
    """

    if getattr(space_group, "ignored", False):
        return 0
    return 1 if space_group.crystal_system in ("trigonal", "hexagonal") else 0


def berkeleygw_date_time(now: _datetime.datetime | None = None) -> tuple[str, str]:
    """BerkeleyGW's ``date_time``: ``'27-Oct-2026'`` and ``'19:02:10 -0700'``."""

    moment = now if now is not None else _datetime.datetime.now().astimezone()
    offset = moment.utcoffset() or _datetime.timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    sign = "-" if minutes < 0 else "+"
    minutes = abs(minutes)
    zone = f"{sign}{minutes // 60:02d}{minutes % 60:02d}"
    date = f"{moment.day:02d}-{_MONTHS[moment.month - 1]}-{moment.year:04d}"
    time = f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d} {zone}"
    return date, time


# --------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------


def _record(handle, payload: bytes) -> None:
    marker = struct.pack("<i", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def _fortran_order(matrix: np.ndarray) -> np.ndarray:
    """``((x(j, i), j = 1, 3), i = 1, 3)``: the columns one after the other."""

    return np.asarray(matrix, dtype=np.float64).flatten(order="F")


def _header_records(header: BerkeleyGWHeader) -> list[bytes]:
    title = f"{header.kind}-{header.flavor}"
    records = [
        title.ljust(32).encode("ascii")
        + header.date.ljust(32).encode("ascii")
        + header.time.ljust(32).encode("ascii")
    ]
    if header.is_wavefunction:
        records.append(
            struct.pack(
                "<5id3id",
                header.spin_count,
                header.gvector_count,
                header.rotations.shape[0],
                header.cell_symmetry,
                header.atom_positions.shape[0],
                header.ecutrho,
                header.k_points.shape[0],
                header.band_count,
                header.max_k_gvectors,
                header.ecutwfc,
            )
        )
        records.append(struct.pack("<6i3d", *header.fft_grid, *header.kgrid, *header.kshift))
    else:
        records.append(
            struct.pack(
                "<5id",
                header.spin_count,
                header.gvector_count,
                header.rotations.shape[0],
                header.cell_symmetry,
                header.atom_positions.shape[0],
                header.ecutrho,
            )
        )
        records.append(struct.pack("<3i", *header.fft_grid))
    records.append(
        np.concatenate(
            [[header.cell_volume, header.alat], _fortran_order(header.lattice), _fortran_order(header.adot)]
        ).astype("<f8").tobytes()
    )
    records.append(
        np.concatenate(
            [
                [header.reciprocal_volume, header.blat],
                _fortran_order(header.reciprocal),
                _fortran_order(header.bdot),
            ]
        ).astype("<f8").tobytes()
    )
    records.append(
        np.concatenate([np.asarray(m).flatten(order="F") for m in header.rotations])
        .astype("<i4")
        .tobytes()
    )
    records.append(np.asarray(header.translations, dtype="<f8").reshape(-1).tobytes())
    atoms = np.zeros(header.atom_positions.shape[0], dtype=[("pos", "<f8", 3), ("typ", "<i4")])
    atoms["pos"] = header.atom_positions
    atoms["typ"] = header.atom_types
    records.append(atoms.tobytes())
    if header.is_wavefunction:
        records.append(np.asarray(header.k_gvector_counts, dtype="<i4").tobytes())
        records.append(np.asarray(header.k_weights, dtype="<f8").tobytes())
        records.append(np.asarray(header.k_points, dtype="<f8").reshape(-1).tobytes())
        records.append(np.asarray(header.lowest_occupied, dtype="<i4").reshape(-1).tobytes())
        records.append(np.asarray(header.highest_occupied, dtype="<i4").reshape(-1).tobytes())
        records.append(np.asarray(header.energies, dtype="<f8").reshape(-1).tobytes())
        records.append(np.asarray(header.occupations, dtype="<f8").reshape(-1).tobytes())
    return records


def _gvector_records(handle, vectors: np.ndarray) -> None:
    _record(handle, struct.pack("<i", 1))
    _record(handle, struct.pack("<i", int(vectors.shape[0])))
    _record(handle, np.asarray(vectors, dtype="<i4").reshape(-1).tobytes())


def _data_records(handle, count: int, values: np.ndarray) -> None:
    _record(handle, struct.pack("<i", 1))
    _record(handle, struct.pack("<i", int(count)))
    _record(handle, np.asarray(values, dtype="<c16").reshape(-1).tobytes())


def write_berkeleygw_file(
    path,
    header: BerkeleyGWHeader,
    gvectors: np.ndarray,
    *,
    data: np.ndarray | None = None,
    k_blocks: Iterable[tuple[np.ndarray, Iterable[np.ndarray]]] | None = None,
) -> Path:
    """Write one file: header, G-vectors, then the data (RHO/VXC) or the bands (WFN).

    RHO/VXC: ``data`` ``(ns, ng)``.  WFN: ``k_blocks`` yields per k-point its
    sphere ``(ngk, 3)`` and an iterable of the bands, each ``(ns, ngk)``
    (consumed as written, so the bands need not all be in memory).
    """

    path = Path(path)
    with path.open("wb") as handle:
        for payload in _header_records(header):
            _record(handle, payload)
        _gvector_records(handle, gvectors)
        if header.is_wavefunction:
            if k_blocks is None:
                raise ValueError("a WFN file needs k_blocks")
            for vectors, bands in k_blocks:
                _gvector_records(handle, vectors)
                for band in bands:
                    band = np.asarray(band, dtype=np.complex128)
                    _data_records(handle, vectors.shape[0], band)
        else:
            if data is None:
                raise ValueError("an RHO or VXC file needs data")
            values = np.asarray(data, dtype=np.complex128)
            _data_records(handle, values.shape[-1], values)
    return path


def _read_records(path) -> Iterator[bytes]:
    data = Path(path).read_bytes()
    position = 0
    while position < len(data):
        (length,) = struct.unpack_from("<i", data, position)
        body = data[position + 4 : position + 4 + length]
        (closing,) = struct.unpack_from("<i", data, position + 4 + length)
        if closing != length:
            raise ValueError(f"{path}: record markers {length} and {closing} differ")
        yield body
        position += 8 + length


def read_berkeleygw_file(path) -> BerkeleyGWFile:
    """Read a WFN, RHO or VXC file of the binary complex (or real) flavour."""

    records = _read_records(path)
    first = next(records)
    title = first[:32].decode("ascii").rstrip()
    date = first[32:64].decode("ascii").rstrip()
    time = first[64:96].decode("ascii").rstrip()
    kind, _, flavor = title.partition("-")
    if kind not in ("WFN", "RHO", "VXC"):
        raise ValueError(f"{path}: unknown BerkeleyGW header {title!r}")
    dtype = "<c16" if flavor == COMPLEX_FLAVOR else "<f8"
    wavefunction = kind == "WFN"
    if wavefunction:
        (ns, ng, ntran, cell_symmetry, nat, ecutrho, nk, nbands, ngkmax, ecutwfc) = struct.unpack(
            "<5id3id", next(records)
        )
        values = struct.unpack("<6i3d", next(records))
        fft, kgrid, kshift = values[:3], values[3:6], values[6:9]
    else:
        ns, ng, ntran, cell_symmetry, nat, ecutrho = struct.unpack("<5id", next(records))
        fft = struct.unpack("<3i", next(records))
        kgrid, kshift, nk, nbands, ngkmax, ecutwfc = (0, 0, 0), (0.0, 0.0, 0.0), 0, 0, 0, 0.0
    cell = np.frombuffer(next(records), "<f8")
    reciprocal = np.frombuffer(next(records), "<f8")
    rotations = np.frombuffer(next(records), "<i4").reshape(ntran, 3, 3).transpose(0, 2, 1)
    translations = np.frombuffer(next(records), "<f8").reshape(ntran, 3)
    atoms = np.frombuffer(next(records), dtype=[("pos", "<f8", 3), ("typ", "<i4")])
    extra = {}
    if wavefunction:
        extra = dict(
            band_count=nbands,
            max_k_gvectors=ngkmax,
            k_gvector_counts=np.frombuffer(next(records), "<i4").copy(),
            k_weights=np.frombuffer(next(records), "<f8").copy(),
            k_points=np.frombuffer(next(records), "<f8").reshape(nk, 3).copy(),
            lowest_occupied=np.frombuffer(next(records), "<i4").reshape(ns, nk).copy(),
            highest_occupied=np.frombuffer(next(records), "<i4").reshape(ns, nk).copy(),
            energies=np.frombuffer(next(records), "<f8").reshape(ns, nk, nbands).copy(),
            occupations=np.frombuffer(next(records), "<f8").reshape(ns, nk, nbands).copy(),
        )
    header = BerkeleyGWHeader(
        kind=kind,
        flavor=flavor,
        date=date,
        time=time,
        spin_count=ns,
        gvector_count=ng,
        cell_symmetry=cell_symmetry,
        ecutrho=ecutrho,
        ecutwfc=ecutwfc,
        fft_grid=tuple(fft),
        kgrid=tuple(kgrid),
        kshift=tuple(kshift),
        cell_volume=float(cell[0]),
        alat=float(cell[1]),
        lattice=cell[2:11].reshape(3, 3).T.copy(),
        adot=cell[11:20].reshape(3, 3).T.copy(),
        reciprocal_volume=float(reciprocal[0]),
        blat=float(reciprocal[1]),
        reciprocal=reciprocal[2:11].reshape(3, 3).T.copy(),
        bdot=reciprocal[11:20].reshape(3, 3).T.copy(),
        rotations=rotations.copy(),
        translations=translations.copy(),
        atom_positions=atoms["pos"].copy(),
        atom_types=atoms["typ"].copy(),
        **extra,
    )

    def gvector_block() -> np.ndarray:
        (chunks,) = struct.unpack("<i", next(records))
        blocks = []
        for _ in range(chunks):
            (count,) = struct.unpack("<i", next(records))
            blocks.append(np.frombuffer(next(records), "<i4").reshape(count, 3))
        return np.concatenate(blocks) if blocks else np.zeros((0, 3), dtype=np.int32)

    counts: list[int] = []

    def data_block(size: int) -> np.ndarray:
        (chunks,) = struct.unpack("<i", next(records))
        pieces = []
        for _ in range(chunks):
            (count,) = struct.unpack("<i", next(records))
            counts.append(count)
            pieces.append(np.frombuffer(next(records), dtype))
        joined = np.concatenate(pieces)
        return joined.reshape(-1, size)

    gvectors = gvector_block()
    if not wavefunction:
        data = data_block(ng)
        return BerkeleyGWFile(header=header, gvectors=gvectors, data=data, record_counts=tuple(counts))
    k_gvectors = []
    bands = []
    for _ in range(nk):
        sphere = gvector_block()
        k_gvectors.append(sphere)
        bands.append(np.asarray([data_block(sphere.shape[0]) for _ in range(nbands)]))
    return BerkeleyGWFile(
        header=header,
        gvectors=gvectors,
        k_gvectors=tuple(k_gvectors),
        bands=tuple(bands),
        record_counts=tuple(counts),
    )


# --------------------------------------------------------------------------
# parsec.out lines
# --------------------------------------------------------------------------


def _list_directed_complex(value: complex) -> str:
    """A ``complex(dpc)`` item of a gfortran list-directed record, with its separator."""

    from .correction_output import list_directed_real

    parts = [list_directed_real(part).strip() for part in (value.real, value.imag)]
    text = f"({parts[0]},{parts[1]})"
    return " " + text.rjust(53)


def potential_gspace_line(sphere: GSphere) -> str:
    """``generate_potential_gspace`` format 10."""

    n1, n2, n3 = sphere.fft_size
    return f" SIZE OF POTENTIAL GSPACE = {sphere.count:8d} WITH FFT GRID {n1:6d}{n2:6d}{n3:6d}"


def k_gspace_lines(setup: BerkeleyGWSetup, weights) -> list[str]:
    """``generate_k_gspaces`` formats 10 and 20 for every k-point."""

    lines = [
        "",
        " HAMILTONIAN MATRIX DIMENSIONS:",
        " " + "-" * 27,
        " " * 12 + "WEIGHT" + " " * 4 + "K-POINT" + " " * 30 + "DIMENSION",
    ]
    for index, (weight, k, sphere) in enumerate(zip(weights, setup.k_points, setup.k_spheres), 1):
        lines.append(
            f"{index:4d}    {weight:10.4f}{k[0]:10.4f}{k[1]:10.4f}{k[2]:10.4f}{'':12s}{sphere.count:6d}"
        )
    return lines


def berkeleygw_report_lines(
    setup: BerkeleyGWSetup,
    weights,
    density_g0: complex,
    vxc_g0: complex,
) -> list[str]:
    """What ``gw_write`` and its caller print in ``parsec.out`` (``parsec.F90`` l.2137-2142).

    The two sphere passes of ``gw_write`` (counting, then the vectors) each
    print the potential sphere and the k-point table; between them the G=0
    coefficients of the density and of V_xc (first channel).
    """

    from .correction_output import list_directed_real

    lines = [
        " calling gw_write ",
        "",
        " \\nAssuming wave-function cutoff of " + list_directed_real(setup.cutoff) + "  Ry \\n",
        potential_gspace_line(setup.potential),
        *k_gspace_lines(setup, weights),
        potential_gspace_line(setup.potential),
        " rho" + _list_directed_complex(complex(density_g0)),
        " vxc" + _list_directed_complex(complex(vxc_g0)),
        *k_gspace_lines(setup, weights),
        " GW output is written",
        "",
    ]
    return lines


__all__ = [
    "BerkeleyGWFile",
    "BerkeleyGWHeader",
    "BerkeleyGWSetup",
    "COMPLEX_FLAVOR",
    "band_coefficients",
    "berkeleygw_date_time",
    "berkeleygw_header",
    "berkeleygw_report_lines",
    "berkeleygw_setup",
    "box_corner",
    "corner_is_symmetric",
    "corner_phases",
    "crystal_k_points",
    "fft_box",
    "fft_box_origin",
    "field_coefficients",
    "forward_transform",
    "grid_steps",
    "occupied_band_range",
    "read_berkeleygw_file",
    "space_group_cell_symmetry",
    "star_averaged_coefficients",
    "wavefunction_coefficients",
    "write_berkeleygw_file",
]
