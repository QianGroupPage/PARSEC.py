"""Density of states: tetrahedra, the Ylm-projected DOS and the ``DOS`` file.

A translation of ``dos.F90`` (``dos``, ``setup_tetrahedra``, ``mirrorijk``,
``ylm_eig``, ``mygauleg``, ``determinant``) and ``spec_tet.F90`` of PARSEC
(``Create_DOS``).  The routine runs after the SCF on the master process
(``parsec.F90`` l.2181-2189).

Energies of the DOS are ``(E_nk - E_F) * 13.6058`` eV.  The mesh has
``DOS_Pnum`` points ``E_i = e_1 + i*delta'`` (``i = 1..N``) on
``[min - 10 delta, max + 10 delta]``, ``delta = |max - min|/N``,
``delta' = width/N``, where ``min``/``max`` are the lowest and the highest
computed level.

Periodic systems: the Monkhorst-Pack sub-cell of every grid point is cut
into six tetrahedra along one of its four long diagonals; ``setup_tetrahedra``
takes the one whose six tetrahedra have the shortest longest edge, with the
edges measured as ``bvec`` times the integer offsets (the reciprocal lattice
vectors themselves, not divided by the grid size: the choice ignores an
anisotropic grid; any of the four splits is a valid decomposition).  The
corners map to computed k-points through ``kptgrid`` (:func:`monkhorst_pack_k_index`
for this port's unreduced grid; a symmetry-reduced grid passes its own map).
Each tetrahedron and band adds the linear-tetrahedron (Gilat-Bharatiya)
spectrum of ``spec_tet`` with the tetrahedron volume ``(2 pi)^3 / Omega / N_tet``;
the file holds ``DOS * Omega/(2 pi)^3 / nspin`` states/eV/cell, so a band
integrates to ``1/nspin``.  The Ylm-projected DOS (``YLM_DOS_L >= 0``) uses
the same tetrahedra with the projection weights of :func:`ylm_projection_weights`
at the corners.

Clusters (``dos.F90`` l.159-168): a histogram of the *occupations*: the
occupation of every level is added to the mesh point ``int((e - e_1)/delta' + 1)``
(the file then shows it at the upper edge of its bin), unscaled.  PARSEC then
deallocates an array it never allocated and stops with a runtime error after
writing the file; this port does not.

The cell volume of the tetrahedra and of the factor is PARSEC's rounding of
it (:func:`fortran_cell_volume`); with it the ``DOS`` file of the Fortran
binary is reproduced byte for byte from the same levels (fcc Si, 2x2x2 spin
and 3x3x3 grids, and the H2O histogram, with and without symmetry).

Deviations from ``dos.F90`` (index or physics defects of the Fortran; the
projected-DOS ones can be switched back with ``fortran_projection``):

* PARSEC reads ``nstate`` levels from every representation (each holds only
  its share), loops the tetrahedra over ``ntotal(irp)`` (a count per spin
  channel indexed by the representation) and takes the top of the mesh from
  level ``nec``.  This port takes the computed levels of every channel once
  (``States_Num`` per k-point).  In every run checked against the binary
  (including a cluster with four representations) both give the same file.
* ``ylm_eig``'s s channel has no radial factor (``ylm(1) = c0``): its
  "projection" is ``c0 sum psi`` over the sphere for every radial node.
  Here the s channel carries the radial Gaussian like the others.
* ``ylm_eig`` sums the Bloch function's periodic part ``u_k`` over the image
  cells without the phase ``exp(i k.(r+T))``; this port projects
  ``psi_k(r+T) = exp(i k.(r+T)) u_k(r)``.
* A grid point exactly on an atom (distance 0) gives ``1/0`` in PARSEC; it
  is skipped here (no direction, and the radial weight diverges).
* The image cells: PARSEC adds the 27 cells ``-1..1`` around its grid;
  here the image range covers the whole cutoff sphere (the same points for
  a cutoff smaller than the cell).
* ``ylm_eig`` indexes the wavefunctions of its process by the global grid
  index (correct on one process only) and reads an undefined spin index
  for spin-orbit spinors; neither arises here (spin-orbit is refused).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .correction_output import list_directed_real

#: ``rydberg`` of ``const.f90``: Ry to eV.
RYDBERG_TO_EV = 13.6058
#: ``tolerance`` and ``delta`` of ``spec_tet.F90``.
_MERGE_TOLERANCE = 1.0e-9
_NUDGE = 1.0e-8
#: ``ylmname`` of ``dos.F90``.
YLM_NAMES = (
    "s",
    "px",
    "py",
    "pz",
    "dxy",
    "dzy",
    "dxz",
    "dz2",
    "dx2-y2",
    "fz3",
    "fxz2",
    "fyz2",
    "fz(x2-y2)",
    "fxyz",
    "fx(3y2-x2)",
    "fy(3x2-y2)",
)
#: ``initialtetraoffset(3,4,6)`` of ``setup_tetrahedra``: the six
#: tetrahedra of a unit cube (corner offsets, in the order of the source).
_INITIAL_TETRAHEDRA = np.asarray(
    [
        0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
        0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
        1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
    ],
    dtype=np.int64,
).reshape(6, 4, 3)  # Fortran (3,4,6) column-major: [tetrahedron][vertex][axis]


def harmonic_count(l_max: int) -> int:
    """``nylm``: the real harmonics of ``l = 0..l_max`` (1, 4, 9, 16)."""

    return (int(l_max) + 1) ** 2


@dataclass(frozen=True)
class DOSMesh:
    """The energy mesh of ``dos.F90``: ``energies = lower + step * (1..points)`` (eV)."""

    lower: float
    upper: float
    step: float
    points: int

    @property
    def energies(self) -> np.ndarray:
        return self.lower + np.arange(1, self.points + 1, dtype=float) * self.step


def dos_energy_mesh(lowest: float, highest: float, points: int) -> DOSMesh:
    """``ebound`` and ``delta`` of ``dos.F90`` l.144-148 (energies in eV)."""

    delta = abs(highest - lowest) / float(points)
    lower = lowest - 1.0e1 * delta
    upper = highest + 1.0e1 * delta
    return DOSMesh(lower, upper, abs(upper - lower) / float(points), int(points))


def monkhorst_pack_k_index(dimensions: Sequence[int]) -> np.ndarray:
    """``kptgrid`` of this port's unreduced Monkhorst-Pack grid.

    :func:`parsec_python.Grid.kpoints.monkhorst_pack_grid` lists the k-points
    of the grid point ``(i, j, l)`` (``0 <= i < n1``, ...) at the index
    ``(i n2 + j) n3 + l``, increasing fractional coordinates along each
    axis; the map from the full grid to the computed k-points is therefore
    the identity in that order.  A symmetry-reduced grid (Fortran
    ``irrbz``) gives instead the index of the irreducible k-point of every
    grid point, 0-based (``abs(kptgrid) - 1``).
    """

    n1, n2, n3 = (int(value) for value in dimensions)
    return np.arange(n1 * n2 * n3, dtype=np.int64).reshape(n1, n2, n3)


def _mirror(offsets: np.ndarray, i: int, j: int, k: int) -> np.ndarray:
    """``mirrorijk``: reflect the offsets about 0.5 along the flagged axes."""

    out = offsets.copy()
    for axis, flag in enumerate((i, j, k)):
        if flag == 1:
            out[..., axis] = 1 - out[..., axis]
    return out


def tetrahedron_split(reciprocal_vectors: np.ndarray) -> np.ndarray:
    """The six tetrahedra (``tset``, shape ``(6, 4, 3)``) of ``setup_tetrahedra``.

    ``reciprocal_vectors`` holds ``b_1, b_2, b_3`` as rows (Fortran ``bvec``
    has them as columns).  The four mirrorings ``(i, j, 0)``, ``i, j = 0, 1``
    are tried in that order and the first with the smallest longest edge
    ``|bvec . (t_a - t_b)|`` is kept (``thisedgemax < edgemax``).
    """

    bvec = np.asarray(reciprocal_vectors, dtype=float).T
    best = None
    longest = 1.0e10
    for i in (0, 1):
        for j in (0, 1):
            trial = _mirror(_INITIAL_TETRAHEDRA, i, j, 0)
            edge_max = 0.0
            for tetrahedron in trial:
                cart = bvec @ tetrahedron.T.astype(float)
                for first in range(3):
                    for second in range(first + 1, 4):
                        difference = cart[:, first] - cart[:, second]
                        edge = np.sqrt(
                            difference[0] ** 2 + difference[1] ** 2 + difference[2] ** 2
                        )
                        edge_max = max(edge_max, edge)
            if edge_max < longest:
                best, longest = trial, edge_max
    return best


def tetrahedron_corners(
    dimensions: Sequence[int], k_index: np.ndarray, reciprocal_vectors: np.ndarray
) -> np.ndarray:
    """``itetra`` of ``setup_tetrahedra``: the k-point index of every corner.

    Returns shape ``(6 n1 n2 n3, 4)``, 0-based k indices; the tetrahedra
    of the sub-cell at ``(i, j, k)`` come in the order of the Fortran loops
    (``i`` slowest, then ``j``, ``k``, then the six tetrahedra), corners
    wrapped ``mod n``.
    """

    n = np.asarray([int(value) for value in dimensions], dtype=np.int64)
    k_index = np.abs(np.asarray(k_index, dtype=np.int64))
    if k_index.shape != tuple(n):
        raise ValueError("the k-point map must have the shape of the Monkhorst-Pack grid")
    tset = tetrahedron_split(reciprocal_vectors)
    cells = np.stack(
        np.meshgrid(*(np.arange(value) for value in n), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    corners = (cells[:, None, None, :] + tset[None, :, :, :]) % n
    return k_index[corners[..., 0], corners[..., 1], corners[..., 2]].reshape(-1, 4)


def heap_sort_order(values: Sequence[float]) -> list[int]:
    """``sort.F90`` (heap sort of Numerical Recipes): the 0-based ascending order.

    Ties keep the order this heap sort gives them, which decides which
    corner values ``spec_tet`` pairs with equal corner energies.
    """

    array = [float(value) for value in values]
    n = len(array)
    if n == 0:
        return []
    if n == 1:
        return [0]
    index = list(range(1, n + 1))
    left = n // 2 + 1
    right = n
    while True:
        if left > 1:
            left -= 1
            held = index[left - 1]
            value = array[held - 1]
        else:
            held = index[right - 1]
            value = array[held - 1]
            index[right - 1] = index[0]
            right -= 1
            if right == 1:
                index[0] = held
                return [entry - 1 for entry in index]
        i = left
        j = left + left
        while j <= right:
            if j < right and array[index[j - 1] - 1] < array[index[j] - 1]:
                j += 1
            if value < array[index[j - 1] - 1]:
                index[i - 1] = index[j - 1]
                i = j
                j = j + j
            else:
                j = right + 1
        index[i - 1] = held


def _sorted_corners(energies: np.ndarray, values: np.ndarray | None):
    """Sort the corners of every row as ``spec_tet`` does and merge near ties."""

    rows = energies.shape[0]
    order = np.empty((rows, 4), dtype=np.int64)
    for row in range(rows):
        order[row] = heap_sort_order(energies[row])
    es = np.take_along_axis(energies, order, axis=1).copy()
    fs = None if values is None else np.take_along_axis(values, order[:, :, None], axis=1)
    for j in (1, 2, 3):
        close = np.abs(es[:, j] - es[:, j - 1]) <= _MERGE_TOLERANCE
        es[close, j] = es[close, j - 1]
    return es, fs


@dataclass(frozen=True)
class SpectrumRows:
    """The kept rows of :func:`tetrahedron_spectrum` and their mesh windows.

    ``es`` are the sorted, merged corner energies (eV) of the rows with not
    all four corners equal, ``fs`` their corner values (``(rows, 4, P)`` or
    ``None``); ``e1``..``dn4`` the row quantities of ``spec_tet`` (``d21 =
    e2 - e1``, ..., ``dn1 = d41 d31 d21 / volume``, ...).  ``istart`` and
    ``counts`` are the 1-based first mesh point and the length of every
    row's window, ``first`` the position of its first (row, mesh point)
    pair; ``row`` and ``index`` give the row and the 1-based mesh point of
    every pair, rows slowest (the order of the Fortran loops).
    """

    es: np.ndarray
    fs: np.ndarray | None
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    e4: np.ndarray
    d21: np.ndarray
    d31: np.ndarray
    d32: np.ndarray
    d41: np.ndarray
    d42: np.ndarray
    d43: np.ndarray
    dn1: np.ndarray
    dn2: np.ndarray
    dn4: np.ndarray
    istart: np.ndarray
    counts: np.ndarray
    first: np.ndarray
    row: np.ndarray
    index: np.ndarray


def _spectrum_rows(volume: float, es: np.ndarray, fs, mesh: DOSMesh) -> SpectrumRows:
    """The row quantities and windows of the kept rows (host arrays)."""

    points = mesh.points
    e1, e2, e3, e4 = (es[:, j] for j in range(4))
    d21, d31, d32 = e2 - e1, e3 - e1, e3 - e2
    d41, d42, d43 = e4 - e1, e4 - e2, e4 - e3
    with np.errstate(divide="ignore", invalid="ignore"):
        dn1 = (d41 * d31 * d21) / volume
        dn2 = (d42 * d32 * d21) / volume
        dn4 = (d43 * d42 * d41) / volume
    denerg = abs(mesh.upper - mesh.lower) / float(points)
    istart = np.maximum(1, np.trunc((e1 - mesh.lower) / denerg).astype(np.int64) - 2)
    iend = np.minimum(
        points - np.trunc((mesh.upper - e4) / denerg).astype(np.int64) + 2, points
    )
    counts = np.maximum(iend - istart + 1, 0)
    row = np.repeat(np.arange(es.shape[0]), counts)
    first = np.cumsum(counts) - counts
    offsets = np.arange(row.size) - np.repeat(first, counts)
    index = istart[row] + offsets
    return SpectrumRows(
        es, fs, e1, e2, e3, e4, d21, d31, d32, d41, d42, d43, dn1, dn2, dn4,
        istart, counts, first, row, index,
    )


def _host_bin_sum(index, weights, points: int) -> np.ndarray:
    """The pair ``weights`` summed onto the mesh points (``bincount``, pair order).

    ``weights`` is ``(pairs,)`` (returns ``(N,)``) or ``(pairs, B)`` (returns
    ``(B, N)``, one ``bincount`` per column).
    """

    if weights.ndim == 1:
        return np.bincount(index - 1, weights=weights, minlength=points)
    return np.stack(
        [np.bincount(index - 1, weights=weights[:, p], minlength=points) for p in range(weights.shape[1])]
    )


def spectrum_sums(
    volume: float,
    rows: SpectrumRows,
    mesh: DOSMesh,
    *,
    xp=np,
    bin_sum: Callable | None = None,
    column_block: int | None = None,
):
    """The windowed ``spec_tet`` sums of :func:`tetrahedron_spectrum`.

    Returns ``(edos, fgrid)``: the sums over the rows of ``edos`` (``(N,)``)
    and of ``fgrid`` (``(P, N)``, ``None`` without corner values), as arrays
    of ``xp``.  ``xp`` is NumPy, or an array module with the same operations
    (CuPy: :mod:`parsec_python.acceleration.Output.device_dos`), for which the
    row arrays are moved to its memory; every pair-level quantity is then the
    same sequence of element-wise IEEE operations.  ``bin_sum(index,
    weights, points)`` sums the pair weights (``(pairs,)`` or ``(pairs, B)``)
    onto the mesh points; the default is ``bincount`` in pair order.  ``column_block``
    evaluates the ``P`` corner values that many at a time (the columns are
    independent), all at once by default.
    """

    bin_sum = _host_bin_sum if bin_sum is None else bin_sum
    on_host = xp is np
    errstate = (
        (lambda: np.errstate(divide="ignore", invalid="ignore"))
        if on_host
        else contextlib.nullcontext
    )
    move = (lambda array: array) if on_host else xp.asarray
    points = mesh.points
    denerg = abs(mesh.upper - mesh.lower) / float(points)
    start = mesh.lower
    row = move(rows.row)
    index = move(rows.index)
    es = move(rows.es)
    ei = start + denerg * index.astype(float)
    for j in range(4):
        corner = es[row, j]
        near = xp.abs(corner - ei) <= _NUDGE
        ei = xp.where(near, xp.where(ei < corner, ei + 2.02 * _NUDGE, ei + 1.01 * _NUDGE), ei)
    e1 = move(rows.e1)
    x = ei - e1[row]
    r_e1, r_e2, r_e3, r_e4 = e1[row], move(rows.e2)[row], move(rows.e3)[row], move(rows.e4)[row]
    r_d21, r_d31, r_d32 = move(rows.d21)[row], move(rows.d31)[row], move(rows.d32)[row]
    r_d41, r_d42, r_d43 = move(rows.d41)[row], move(rows.d42)[row], move(rows.d43)[row]
    r_dn1, r_dn2, r_dn4 = move(rows.dn1)[row], move(rows.dn2)[row], move(rows.dn4)[row]
    region2 = (ei > r_e1) & (ei <= r_e2)
    region3 = (ei > r_e2) & (ei <= r_e3)
    region4 = (ei > r_e3) & (ei <= r_e4)
    flat = region3 & (r_e1 == r_e2)
    steep = region3 & ~flat
    s = xp.zeros(row.size)
    with errstate():
        s = xp.where(region2, 3.0 * x * x / r_dn1, s)
        s = xp.where(
            steep,
            3.0 / r_dn1 * x * x - 3.0 / r_dn2 * ((x - r_d21) * (x - r_d21)),
            s,
        )
        s = xp.where(
            flat,
            3.0 * volume * x / (r_d41 * r_d31) * (2.0 - x * (1.0 / r_d31 + 1.0 / r_d41)),
            s,
        )
        s = xp.where(region4, 3.0 * ((x - r_d41) * (x - r_d41)) / r_dn4, s)
    edos = bin_sum(index, s, points)
    if rows.fs is None:
        return edos, None
    columns = rows.fs.shape[2]
    block = columns if column_block is None else max(1, int(column_block))
    fgrid = xp.zeros((columns, points))
    c = lambda array: array[:, None]  # noqa: E731 - broadcast a row quantity
    third = 1.0 / 3.0
    for begin in range(0, columns, block):
        f = move(np.ascontiguousarray(rows.fs[:, :, begin : begin + block]))[row]  # (pairs, 4, B)
        f1, f2, f3, f4 = (f[:, j, :] for j in range(4))
        f21, f31, f32 = f2 - f1, f3 - f1, f3 - f2
        f41, f42, f43 = f4 - f1, f4 - f2, f4 - f3
        fms = xp.zeros_like(f1)
        with errstate():
            term = (f21 / c(r_d21) + f31 / c(r_d31) + f41 / c(r_d41)) * c(x) * third
            fms = xp.where(c(region2), term * c(s), fms)
            x3 = c(x * x * x)
            shifted = c(x - r_d21)
            term = (
                (f21 / c(r_d21) + f31 / c(r_d31) + f41 / c(r_d41)) * x3 / c(r_dn1)
                - (f21 / c(r_d21) + f32 / c(r_d32) + f42 / c(r_d42))
                * (shifted * shifted * shifted)
                / c(r_dn2)
                - 3.0 * f21 * (shifted * shifted) / c(r_dn2)
            )
            fms = xp.where(c(steep), term, fms)
            prefactor = c(volume * x / (r_d41 * r_d31))
            term = prefactor * (
                3.0 * f21
                + 3.0 * c(x) * (f32 / c(r_d32) + f42 / c(r_d42))
                - c(x * x) / c(r_d41 * r_d31)
                * (
                    c(r_d41) * (f31 / c(r_d31) + f32 / c(r_d32) + f41 / c(r_d41))
                    + c(r_d31) * (f31 / c(r_d31) + f41 / c(r_d41) + f42 / c(r_d42))
                    - f21
                )
            )
            fms = xp.where(c(flat), term, fms)
            term = (
                (f41 / c(r_d41) + f42 / c(r_d42) + f43 / c(r_d43)) * c(x - r_d41) * third
                + f41
            )
            fms = xp.where(c(region4), term * c(s), fms)
        contribution = f1 * c(s) + fms
        fgrid[begin : begin + contribution.shape[1]] = bin_sum(index, contribution, points)
        del f, f1, f2, f3, f4, f21, f31, f32, f41, f42, f43, fms, term, contribution
    return edos, fgrid


#: Replaces :func:`spectrum_sums` (``provider(volume, rows, mesh)`` returns
#: the two host sums, or ``None`` to leave them to the host); see
#: :func:`set_spectrum_provider`.
_spectrum_provider: Callable | None = None


def set_spectrum_provider(provider: Callable | None) -> None:
    """Install (or, with ``None``, remove) the evaluator of the windowed sums.

    :mod:`parsec_python.acceleration.Output.device_dos` installs one that
    runs :func:`spectrum_sums` on a GPU with the pair order of ``bincount``
    kept, so the file is the same with and without it.
    """

    global _spectrum_provider
    _spectrum_provider = provider


def tetrahedron_spectrum(
    volume: float,
    energies: np.ndarray,
    mesh: DOSMesh,
    values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """``spec_tet`` for many tetrahedra at once.

    ``energies`` has one row of four corner energies (eV) per tetrahedron and
    band; ``values`` (optional, shape ``(rows, 4, P)``) the ``P`` spectral
    functions at the corners (the projection weights).  Returns the sum over
    rows of ``edos`` (shape ``(N,)``), of ``fgrid`` (``(P, N)``, or ``None``)
    and the number of rows whose four corners are equal (``neqtet``; they
    add nothing).  Every row is evaluated at the mesh points of its window
    ``istart..iend`` of the Fortran loop (it is zero elsewhere):
    :func:`spectrum_sums`, or the installed provider.
    """

    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 2 or energies.shape[1] != 4:
        raise ValueError("corner energies must have shape (rows, 4)")
    if values is not None:
        values = np.asarray(values, dtype=float)
        if values.ndim != 3 or values.shape[:2] != energies.shape:
            raise ValueError("corner values must have shape (rows, 4, P)")
    points = mesh.points
    dos = np.zeros(points)
    projected = None if values is None else np.zeros((values.shape[2], points))
    if energies.shape[0] == 0:
        return dos, projected, 0
    es, fs = _sorted_corners(energies, values)
    equal = (es[:, 0] == es[:, 1]) & (es[:, 1] == es[:, 2]) & (es[:, 2] == es[:, 3])
    equal_count = int(np.count_nonzero(equal))
    keep = ~equal
    es = es[keep]
    if fs is not None:
        fs = fs[keep]
    if es.shape[0] == 0:
        return dos, projected, equal_count
    rows = _spectrum_rows(volume, es, fs, mesh)
    if rows.row.size == 0:
        return dos, projected, equal_count
    sums = None if _spectrum_provider is None else _spectrum_provider(volume, rows, mesh)
    if sums is None:
        sums = spectrum_sums(volume, rows, mesh)
    edos, fgrid = sums
    dos += edos
    if fs is None:
        return dos, None, equal_count
    projected += fgrid
    return dos, projected, equal_count


@dataclass
class DOSResult:
    """The spectra of ``dos.F90`` before the file's scale factor.

    ``dos`` has shape ``(nspin, N)`` (``fulldos``), ``projected`` shape
    ``(nspin, atoms, nylm, N)`` or ``None``.  ``factor`` is ``dos_factor``
    (``Omega/(2 pi)^3/nspin`` for a cell, one for a cluster); ``tetrahedra``
    and ``equal_corners`` the two numbers ``dos.F90`` reports (``None`` for
    a cluster).  ``fermi_level`` is in Ry.
    """

    mesh: DOSMesh
    dos: np.ndarray
    factor: float
    fermi_level: float
    projected: np.ndarray | None = None
    tetrahedra: int | None = None
    equal_corners: int | None = None

    @property
    def spin_count(self) -> int:
        return int(self.dos.shape[0])


def _relative_ev(eigenvalues: Sequence[np.ndarray], fermi_level: float) -> list[np.ndarray]:
    return [
        (np.asarray(channel, dtype=float) - float(fermi_level)) * RYDBERG_TO_EV
        for channel in eigenvalues
    ]


def fortran_cell_volume(grid) -> float:
    """``pbc%vcell`` of ``inipbc.f90`` l.146-171 for a periodic grid.

    ``n1 n2 n3 hcub`` with ``hcub = |det(h_i a_i/|a_i|)|`` by the cofactor
    expansion of ``mtrxin``: the cell volume as PARSEC rounds it.  The
    tetrahedron volume and the DOS factor use it.
    """

    lattice = np.asarray(grid.cell.lattice_vectors, dtype=float)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    steps = (
        np.asarray(grid.spacings, dtype=float)
        if getattr(grid, "is_general_periodic", False)
        else np.diag(lattice) / counts
    )
    # Columns of tmpmat are step_i a_i / |a_i|.
    m = np.empty((3, 3))
    for axis in range(3):
        vector = lattice[axis]
        m[:, axis] = steps[axis] * vector / np.sqrt(np.sum(vector**2))
    a11 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    a21 = -m[1, 0] * m[2, 2] + m[1, 2] * m[2, 0]
    a31 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    determinant = m[0, 0] * a11 + m[0, 1] * a21 + m[0, 2] * a31
    return float(int(counts[0] * counts[1] * counts[2]) * abs(determinant))


def tetrahedron_dos(
    eigenvalues: Sequence[np.ndarray],
    fermi_level: float,
    *,
    dimensions: Sequence[int],
    k_index: np.ndarray,
    lattice_vectors: np.ndarray,
    points: int,
    projections: Sequence[np.ndarray] | None = None,
    cell_volume: float | None = None,
    chunk: int = 4096,
) -> DOSResult:
    """The tetrahedron DOS of a periodic system (``dos.F90`` l.169-209).

    ``eigenvalues[s]`` has shape ``(n_k, n_states)`` (Ry) for spin channel
    ``s``, with the computed k-points in the order ``k_index`` refers to;
    ``dimensions`` is the Monkhorst-Pack grid of the tetrahedra and
    ``k_index`` its map to the computed k-points (0-based,
    :func:`monkhorst_pack_k_index`).  ``projections[s]`` (optional) has
    shape ``(n_k, n_states, atoms, nylm)``: the weights of
    :func:`ylm_projection_weights`.  ``cell_volume`` (bohr^3) defaults to
    ``|det(lattice_vectors)|``; :func:`fortran_cell_volume` gives PARSEC's
    rounding of it.
    """

    from ..Grid.cell import reciprocal_lattice

    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    volume_cell = (
        float(abs(np.linalg.det(lattice_vectors)))
        if cell_volume is None
        else float(cell_volume)
    )
    channels = _relative_ev(eigenvalues, fermi_level)
    if any(channel.ndim != 2 for channel in channels):
        raise ValueError("eigenvalues must be (n_k, n_states) arrays, one per spin channel")
    lowest = min(float(channel[:, 0].min()) for channel in channels)
    highest = max(float(channel[:, -1].max()) for channel in channels)
    mesh = dos_energy_mesh(lowest, highest, points)
    corners = tetrahedron_corners(dimensions, k_index, reciprocal_lattice(lattice_vectors))
    count = corners.shape[0]
    tetra_volume = (2.0 * np.pi) ** 3 / volume_cell / count
    spin_count = len(channels)
    dos = np.zeros((spin_count, points))
    projected = None
    if projections is not None:
        atoms, nylm = np.asarray(projections[0]).shape[2:]
        projected = np.zeros((spin_count, atoms, nylm, points))
    equal = 0
    for spin, channel in enumerate(channels):
        bands = channel.shape[1]
        flat = None
        if projections is not None:
            weights = np.asarray(projections[spin], dtype=float)
            flat = weights.reshape(weights.shape[0], bands, -1)  # (k, bands, P)
        # Rows (tetrahedron, band), tetrahedra slowest, as the Fortran loops;
        # built a chunk at a time.
        rows = count * bands
        for begin in range(0, rows, chunk):
            row = np.arange(begin, min(begin + chunk, rows))
            tetrahedron, band = np.divmod(row, bands)
            corner_k = corners[tetrahedron]  # (m, 4)
            energies = channel[corner_k, band[:, None]]  # (m, 4)
            values = None if flat is None else flat[corner_k, band[:, None], :]
            spectrum, weighted, equal_part = tetrahedron_spectrum(
                tetra_volume, energies, mesh, values
            )
            dos[spin] += spectrum
            equal += equal_part
            if weighted is not None:
                projected[spin] += weighted.reshape(atoms, nylm, points)
    bands_total = channels[0].shape[1]
    return DOSResult(
        mesh=mesh,
        dos=dos,
        factor=volume_cell / (2.0 * np.pi) ** 3 / spin_count,
        fermi_level=float(fermi_level),
        projected=projected,
        tetrahedra=count,
        equal_corners=equal // (spin_count * bands_total),
    )


def occupation_histogram(
    eigenvalues: Sequence[np.ndarray],
    occupations: Sequence[np.ndarray],
    fermi_level: float,
    *,
    points: int,
) -> DOSResult:
    """The DOS of a cluster (``dos.F90`` l.149-168): binned occupations.

    ``eigenvalues[s]``/``occupations[s]`` are the levels (Ry) and their
    occupations (fraction of the state's capacity, as PARSEC stores them) of
    spin channel ``s``.  The occupation of a level adds to mesh point
    ``int((e - e_1)/delta' + 1)``; the factor is one.
    """

    channels = _relative_ev(eigenvalues, fermi_level)
    lowest = min(float(np.min(channel)) for channel in channels)
    highest = max(float(np.max(channel)) for channel in channels)
    mesh = dos_energy_mesh(lowest, highest, points)
    dos = np.zeros((len(channels), points))
    for spin, (channel, occupied) in enumerate(zip(channels, occupations)):
        bins = np.trunc((channel - mesh.lower) / mesh.step + 1.0).astype(np.int64)
        for position, occupation in zip(bins, np.asarray(occupied, dtype=float)):
            dos[spin, position - 1] += occupation
    return DOSResult(mesh=mesh, dos=dos, factor=1.0, fermi_level=float(fermi_level))


# ----------------------------------------------------------------------------
# Ylm projection (ylm_eig)
# ----------------------------------------------------------------------------

#: The prefactors of ``ylm_eig`` (they include ``sqrt((2l+1)/4 pi)``).
_C0 = 0.282094791773878143474039725779
_C1 = 0.488602511902919921586384622837
_C21 = 1.092548430592079070543385705802
_C22 = 0.315391565252520006030893690294
_C31 = 0.373176332590115391414395913197
_C32 = 0.457045799464465736158020696916
_C33 = 1.445305721320277027694690077198
_C34 = 2.890611442640554055389380154398
_C35 = 0.590043589926643510345610277541


def gauss_legendre(lower: float, upper: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """``mygauleg``: Gauss-Legendre nodes and weights on ``[lower, upper]``."""

    count = int(count)
    nodes = np.zeros(count)
    weights = np.zeros(count)
    middle = 0.5 * (upper + lower)
    half = 0.5 * (upper - lower)
    for i in range(1, (count + 1) // 2 + 1):
        z = np.cos(np.pi * (float(i) - 0.25) / (float(count) + 0.5))
        while True:
            p1, p2 = 1.0, 0.0
            for j in range(1, count + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            derivative = float(count) * (z * p1 - p2) / (z * z - 1.0)
            previous = z
            z = previous - p1 / derivative
            if abs(z - previous) < 3.0e-14:
                break
        nodes[i - 1] = middle - half * z
        nodes[count - i] = middle + half * z
        weights[i - 1] = 2.0 * half / ((1.0 - z * z) * derivative * derivative)
        weights[count - i] = weights[i - 1]
    return nodes, weights


def real_harmonics(unit: np.ndarray, count: int) -> np.ndarray:
    """``ylm(1..count)`` of ``ylm_eig`` without the radial factor, shape ``(count, m)``.

    ``unit`` holds the unit vectors ``(x, y, z)`` as rows.  Order: s; p_x,
    p_y, p_z; d_xy, d_zy, d_xz, d_z2, d_x2-y2; f_z3, f_xz2, f_yz2,
    f_z(x2-y2), f_xyz, f_x(x2-3y2), f_y(y2-3x2) (``ylmname``).
    """

    xx, yy, zz = unit[:, 0], unit[:, 1], unit[:, 2]
    values = [np.full(xx.shape, _C0)]
    if count >= 4:
        values += [_C1 * xx, _C1 * yy, _C1 * zz]
    if count >= 9:
        values += [
            _C21 * xx * yy,
            _C21 * zz * yy,
            _C21 * xx * zz,
            _C22 * (3.0 * zz * zz - 1.0),
            _C21 * 0.5 * (xx * xx - yy * yy),
        ]
    if count == 16:
        values += [
            _C31 * zz * (5.0 * zz * zz - 3.0),
            _C32 * xx * (5.0 * zz * zz - 1.0),
            _C32 * yy * (5.0 * zz * zz - 1.0),
            _C33 * zz * (xx * xx - yy * yy),
            _C34 * xx * yy * zz,
            _C35 * xx * (xx * xx - 3.0 * yy * yy),
            _C35 * yy * (yy * yy - 3.0 * xx * xx),
        ]
    return np.asarray(values[:count])


def radial_node_count(cutoff: float, sigma: float) -> int:
    """``ngl = max(10, nint(cutoff*5/sigma))`` of ``ylm_eig``."""

    return max(10, int(np.floor(float(cutoff) * 5.0 / float(sigma) + 0.5)))


def _image_translations(
    lattice_vectors: np.ndarray, fractional: np.ndarray, cutoff: float, periodicity: int = 3
) -> np.ndarray:
    """Lattice translations whose image of the cell can reach the cutoff sphere.

    Grid points have fractional coordinates in ``[0, 1)``; the sphere of
    ``cutoff`` around the atom at ``fractional`` spans ``cutoff/d_i`` along
    direction ``i`` (``d_i = 2 pi/|b_i|``, the distance of the lattice
    planes).  A wire or slab (``periodicity`` 1 or 2) has images along its
    periodic directions only.
    """

    from ..Grid.cell import lattice_plane_spacings

    reach = float(cutoff) / lattice_plane_spacings(lattice_vectors)
    ranges = [
        np.arange(int(np.floor(f - r)) - 1, int(np.floor(f + r)) + 2)
        if axis < periodicity
        else np.zeros(1, dtype=int)
        for axis, (f, r) in enumerate(zip(fractional, reach))
    ]
    mesh = np.meshgrid(*ranges, indexing="ij")
    integers = np.column_stack([axis.reshape(-1) for axis in mesh])
    return integers @ np.asarray(lattice_vectors, dtype=float)


def ylm_projection_weights(
    grid,
    atom_positions: np.ndarray,
    cutoffs: Sequence[float],
    wavefunctions: Sequence[Sequence[np.ndarray]],
    k_points: np.ndarray,
    *,
    l_max: int,
    fortran_projection: bool = False,
) -> list[np.ndarray]:
    """``ylm_eig``: the weight of every state in the real harmonics of every atom.

    ``wavefunctions[s][k]`` holds the states of spin channel ``s`` at
    k-point ``k`` as columns on ``grid`` (normalized as PARSEC's,
    ``sum |psi|^2 = 1``; for k-points the periodic part ``u_k``).  For atom
    ``a`` with radius ``r_c = cutoffs[a]`` and ``sigma = min(h)``, with the
    Gauss-Legendre nodes ``r_g`` (weights ``w_g``) of ``[0, r_c]``,

    ``P_lm(r_g) = sum_{r, T: |r+T-R_a| <= r_c} psi(r+T) Y_lm(d)
    exp(-(|d| - r_g)^2/2 sigma^2) / (sqrt(2 pi) sigma |d|^2) sqrt(h^3)``,
    ``d = r + T - R_a``, and the weight is ``sum_g w_g |r_g P_lm(r_g)|^2``:
    the charge of the state in the sphere carried by the harmonic, summing
    (for a complete set of harmonics) to the charge in the sphere.  Here
    ``psi(r+T) = exp(i k.(r+T)) u_k(r)``.  ``fortran_projection`` gives
    ``ylm_eig`` exactly: no Bloch phase and ``Y_00 = c0`` without the
    radial factor.  Returns, per spin channel, an array of shape
    ``(n_k, n_states, atoms, nylm)``.
    """

    nylm = harmonic_count(l_max)
    lattice = np.asarray(grid.cell.lattice_vectors, dtype=float)
    steps = (
        np.asarray(grid.spacings, dtype=float)
        if getattr(grid, "is_general_periodic", False)
        else np.full(3, float(grid.spacing))
    )
    sigma = float(np.min(steps))
    scale = np.sqrt(float(grid.volume_element))  # fac1: sqrt(h1 h2 h3 det(A_n))
    gaussian_norm = np.sqrt(2.0 * np.pi) * sigma  # fac2
    coordinates = np.asarray(grid.coordinates, dtype=float)
    inverse = np.linalg.inv(lattice)
    k_points = np.asarray(k_points, dtype=float).reshape(-1, 3)
    positions = np.asarray(atom_positions, dtype=float).reshape(-1, 3)
    results = [
        np.zeros(
            (len(channel), np.asarray(channel[0]).shape[1], positions.shape[0], nylm)
        )
        for channel in wavefunctions
    ]
    # A wire or slab: images along the periodic directions only.  Fortran's
    # ylm_eig adds icell*box_size along every direction, and box_size is zero
    # along a confined one: each point enters the projection three times per
    # confined direction, the weight 9 times, which fortran_projection
    # reproduces.
    periodicity = int(getattr(grid, "periodicity", 3)) if getattr(
        grid, "is_confined_periodic", False
    ) else 3
    multiplicity = float(9 ** (3 - periodicity)) if fortran_projection else 1.0
    for atom, (position, cutoff) in enumerate(zip(positions, cutoffs)):
        fractional = position @ inverse
        points, displacements, images = [], [], []
        for translation in _image_translations(lattice, fractional, cutoff, periodicity):
            difference = coordinates + translation - position
            distance = np.sqrt(np.sum(difference * difference, axis=1))
            inside = np.flatnonzero((distance <= cutoff) & (distance > 0.0))
            points.append(inside)
            displacements.append(difference[inside])
            images.append(coordinates[inside] + translation)
        point = np.concatenate(points)
        difference = np.concatenate(displacements)
        image = np.concatenate(images)
        if point.size == 0:
            continue
        distance = np.sqrt(np.sum(difference * difference, axis=1))
        unit = difference / distance[:, None]
        nodes, node_weights = gauss_legendre(0.0, float(cutoff), radial_node_count(cutoff, sigma))
        radial = (
            np.exp(-((distance[None, :] - nodes[:, None]) ** 2 / 2.0 / sigma**2))
            / (distance[None, :] ** 2)
            / gaussian_norm
        )  # (nodes, m)
        angular = real_harmonics(unit, nylm)  # (nylm, m)
        operator = radial[None, :, :] * angular[:, None, :]  # (nylm, nodes, m)
        if fortran_projection:
            operator[0] = angular[0][None, :]  # ylm(1) = c0
        operator = operator.reshape(nylm * nodes.size, point.size)
        for spin, channel in enumerate(wavefunctions):
            for k, vectors in enumerate(channel):
                values = np.asarray(vectors)[point]
                if not fortran_projection and np.iscomplexobj(values):
                    phase = np.exp(1j * (image @ k_points[k]))
                    values = values * phase[:, None]
                projection = (operator @ values) * scale  # (nylm*nodes, states)
                projection = projection.reshape(nylm, nodes.size, -1)
                weight = np.einsum(
                    "g,lgn->nl",
                    node_weights,
                    np.abs(nodes[None, :, None] * projection) ** 2,
                )
                results[spin][k, :, atom, :] = (
                    weight if multiplicity == 1.0 else weight * multiplicity
                )
    return results


# ----------------------------------------------------------------------------
# The DOS file and the lines of parsec.out
# ----------------------------------------------------------------------------


def _fortran_a(text: str, width: int) -> str:
    """An ``A`` edit descriptor of width ``width``: right-justified, or cut."""

    return text.rjust(width) if len(text) <= width else text[:width]


def _fortran_i(value: int, width: int) -> str:
    text = str(int(value))
    return "*" * width if len(text) > width else text.rjust(width)


def dos_file_lines(
    result: DOSResult,
    *,
    atom_labels: Sequence[str] = (),
) -> list[str]:
    """The lines of the file ``DOS`` (``dos.F90`` l.213-262).

    ``atom_labels`` names the species of every atom (projected DOS only).
    """

    lines = [
        " Density Of States - Units:  states/eV/unit cell vs. eV",
        _fortran_a("Fermi levels: ", 15)
        + f"{result.fermi_level:12.6f}{result.fermi_level:12.6f}",
        _fortran_a("Tot. DOS: " if result.spin_count == 1 else "Spin DOS: ", 17),
    ]
    energies = result.mesh.lower + np.arange(1, result.mesh.points + 1) * result.mesh.step
    for spin in range(result.spin_count):
        for energy, value in zip(energies, result.dos[spin]):
            lines.append(list_directed_real(energy) + list_directed_real(value * result.factor))
        lines.append(" &")
    if result.spin_count > 1:
        lines.append(_fortran_a("Sum of spin DOS: ", 100))
        for index, energy in enumerate(energies):
            lines.append(
                list_directed_real(energy)
                + list_directed_real(
                    result.dos[0, index] * result.factor + result.dos[1, index] * result.factor
                )
            )
        lines.append(" &")
    if result.projected is not None:
        atoms, nylm = result.projected.shape[1:3]
        for spin in range(result.spin_count):
            for atom in range(atoms):
                label = atom_labels[atom].strip()
                for harmonic in range(nylm):
                    lines.append(
                        "LDOS : Atom Type = "
                        + _fortran_a(label, 2)
                        + ", Atom Num ="
                        + _fortran_i(atom + 1, 2)
                        + ", Ylm Num = "
                        + _fortran_i(harmonic + 1, 2)
                        + ", Spin = "
                        + _fortran_i(spin + 1, 2)
                    )
                    lines.append(
                        "Graph Name : "
                        + label
                        + _fortran_i(atom + 1, 1)
                        + "_"
                        + YLM_NAMES[harmonic]
                        + "_sp"
                        + _fortran_i(spin + 1, 1)
                    )
                    for index, energy in enumerate(energies):
                        lines.append(
                            list_directed_real(energy)
                            + list_directed_real(
                                result.projected[spin, atom, harmonic, index] * result.factor
                            )
                        )
                    lines.append(" &")
    return lines


def dos_report_lines(result: DOSResult, seconds: float) -> list[str]:
    """What ``dos.F90`` writes to ``parsec.out`` (formats 110, 210, 200, 987, 988)."""

    lines = ["", " Generation of DOS : ", " " + "-" * 17]
    if result.tetrahedra is not None:
        lines += [
            f" Tot. number of tetrahedra :             {result.tetrahedra:10d}",
            f" Number of tetrahedra will eqaul corners:{result.equal_corners:10d}",
        ]
    lines += [
        "",
        ' The DOS data has been writted to the file "DOS"',
        "",
        "",
        f" TIME FOR DENSITY OF STATES: {seconds:14.4f}",
    ]
    return lines


__all__ = [
    "DOSMesh",
    "DOSResult",
    "RYDBERG_TO_EV",
    "SpectrumRows",
    "YLM_NAMES",
    "dos_energy_mesh",
    "dos_file_lines",
    "dos_report_lines",
    "fortran_cell_volume",
    "gauss_legendre",
    "harmonic_count",
    "heap_sort_order",
    "monkhorst_pack_k_index",
    "occupation_histogram",
    "radial_node_count",
    "real_harmonics",
    "set_spectrum_provider",
    "spectrum_sums",
    "tetrahedron_corners",
    "tetrahedron_dos",
    "tetrahedron_spectrum",
    "tetrahedron_split",
    "ylm_projection_weights",
]
