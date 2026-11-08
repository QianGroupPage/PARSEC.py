"""Brillouin-zone k-point sampling for a periodic cell.

Three sources are supported:

* this port's uniform, unreduced Monkhorst-Pack grid
  (:func:`monkhorst_pack_grid`, fractions ``(2i-n-1)/(2n)``, equal weights),
  used by a ``Kpoint_Method: mp`` input without a ``Monkhorst_Pack_Shift``
  block;
* Fortran PARSEC's grid ``k = ((i-1) + s_i)/n_i`` reduced to the irreducible
  wedge by the space group and time reversal
  (:func:`irreducible_monkhorst_pack`, ``irrbz.f90``), used by an input that
  gives ``Monkhorst_Pack_Shift``: unequal weights ``count/N``;
* a user-supplied explicit list (:func:`explicit_kpoints`,
  :func:`read_kpoint_file`, Fortran's ``Kpoint_Method: manual``), whose
  weights are normalized to one as ``init_var.F90`` does.

For odd ``n`` the first grid equals Fortran's with ``s = 0`` modulo
reciprocal lattice vectors, for even ``n`` Fortran's with ``s = 0.5``; Fortran
without a shift block uses ``s = 0`` (Gamma-centred) for every ``n``.  The
finite-difference Bloch Hamiltonian is not exactly periodic in ``k``, so
equivalent points give slightly different eigenvalues; the Fortran path keeps
Fortran's representatives.

Equal weights keep the scalar-``degeneracy`` Fermi-level pooling of the SCF
drivers (``degeneracy = 2*weight``); unequal weights use the per-state
weights of :func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations`
(Fortran ``flevel``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..models import PeriodicCell
from .cell import is_diagonal_lattice, reciprocal_lattice
from .space_group import fortran_anint, fortran_reciprocal_lattice


def monkhorst_pack_grid(
    cell: PeriodicCell,
    dimensions: tuple[int, int, int],
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(k_points, weights)`` for a uniform Monkhorst-Pack grid.

    ``dimensions=(n1,n2,n3)`` are the number of divisions along each
    reciprocal lattice vector.  For an orthorhombic cell with lattice
    vectors ``diag(Lx,Ly,Lz)``, the reciprocal lattice vectors are
    ``b_a = (2*pi/L_a)*e_a``, and the unshifted grid points are

    ``k_(i1,i2,i3) = sum_a [(2*i_a-n_a-1)/(2*n_a)] * b_a``,
    ``i_a = 1,...,n_a``,

    the standard Monkhorst-Pack construction, optionally displaced by
    ``shift`` (given in the same Cartesian reciprocal units as the returned
    k-points, e.g. a half-grid-step shift to avoid sampling exactly at
    Gamma).  Every returned weight is exactly ``1/(n1*n2*n3)``: this is the
    full uniform grid, not reduced to the irreducible wedge.

    ``k_points`` has shape ``(n1*n2*n3, 3)``, ``weights`` has shape
    ``(n1*n2*n3,)`` and sums to 1.

    A general (non-orthorhombic) cell uses the same fractions of its
    reciprocal lattice vectors, ``k = sum_a f_a b_a`` with
    ``b = 2*pi*inv(A)^T`` (rows), in Cartesian 1/bohr.  (Fortran's set is
    ``f = (i-1+s)/n`` in ``[0, 1)``, see report section 1.3: equal modulo
    ``G`` to this one for odd ``n`` without a shift.)
    """
    dims = tuple(int(value) for value in dimensions)
    if len(dims) != 3 or any(value < 1 for value in dims):
        raise ValueError("dimensions must be three positive integers")
    if not is_diagonal_lattice(cell.lattice_vectors):
        shift = np.asarray(shift, dtype=float)
        if shift.shape != (3,):
            raise ValueError("shift must have shape (3,)")
        fractional_axes = [
            (2.0 * np.arange(1, n + 1) - n - 1) / (2.0 * n) for n in dims
        ]
        grids = np.meshgrid(*fractional_axes, indexing="ij")
        fractional = np.column_stack([axis.reshape(-1) for axis in grids])
        k_points = fractional @ reciprocal_lattice(cell.lattice_vectors) + shift[None, :]
        n_total = int(np.prod(dims))
        return k_points, np.full(n_total, 1.0 / n_total, dtype=float)
    side_lengths = np.diag(cell.lattice_vectors)
    reciprocal = 2.0 * np.pi / side_lengths
    shift = np.asarray(shift, dtype=float)
    if shift.shape != (3,):
        raise ValueError("shift must have shape (3,)")

    fractional_axes = [
        (2.0 * np.arange(1, n + 1) - n - 1) / (2.0 * n) for n in dims
    ]
    grids = np.meshgrid(*fractional_axes, indexing="ij")
    fractional = np.column_stack([axis.reshape(-1) for axis in grids])
    if reciprocal.ndim == 2:
        k_points = fractional @ reciprocal + shift[None, :]
    else:
        k_points = fractional * reciprocal[None, :] + shift[None, :]
    n_total = int(np.prod(dims))
    weights = np.full(n_total, 1.0 / n_total, dtype=float)
    return k_points, weights


KPOINT_UNITS = ("cartesian_inverse_bohr", "reciprocal_lattice_vectors")


def explicit_kpoints(
    rows,
    *,
    unit: str = "cartesian_inverse_bohr",
    cell: PeriodicCell | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a user k-point list to ``(k_points, weights)``.

    ``rows`` is ``(n, 3)`` (equal weights are assumed) or ``(n, 4)`` with the
    sampling weight in the last column, the Fortran ``Kpoint_List`` row
    ``kx ky kz weight``.  ``unit`` is either ``"cartesian_inverse_bohr"``
    (the Fortran default: Cartesian components in 1/bohr) or
    ``"reciprocal_lattice_vectors"`` (fractional coordinates along the
    reciprocal lattice vectors ``b_i``, so ``k = sum_i f_i b_i`` with
    ``b = 2*pi*inv(A).T`` for lattice vectors ``A`` in the rows; this needs
    ``cell``).  The returned k-points are always Cartesian 1/bohr; weights
    are normalized to sum to 1 and must be non-negative.
    """

    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] not in (3, 4):
        raise ValueError(
            "k-points must be an (n, 3) array or an (n, 4) array with a weight column"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("k-points and weights must be finite")
    coordinates = array[:, :3]
    if array.shape[1] == 4:
        weights = array[:, 3].copy()
    else:
        weights = np.ones(array.shape[0])
    if np.any(weights < 0.0):
        raise ValueError("k-point sampling weights must be non-negative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("k-point sampling weights must not all be zero")
    weights = weights / total

    normalized_unit = unit.strip().lower()
    if normalized_unit not in KPOINT_UNITS:
        raise ValueError(
            f"Kpoint_Unit must be one of {', '.join(KPOINT_UNITS)}; got {unit!r}"
        )
    if normalized_unit == "reciprocal_lattice_vectors":
        if cell is None:
            raise ValueError("reciprocal_lattice_vectors units need the periodic cell")
        reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(cell.lattice_vectors)).T
        coordinates = coordinates @ reciprocal
    return np.ascontiguousarray(coordinates), weights


def read_kpoint_file(path) -> np.ndarray:
    """Read a k-point file: one ``kx ky kz [weight]`` row per line.

    Blank lines and anything after ``#`` or ``!`` are ignored.  All rows must
    have the same number of columns (three, or four with a weight).  Returns
    the raw ``(n, 3)`` or ``(n, 4)`` array; apply units with
    :func:`explicit_kpoints`.
    """

    from pathlib import Path

    source = Path(path)
    rows: list[list[float]] = []
    for number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw
        for marker in ("#", "!"):
            position = line.find(marker)
            if position >= 0:
                line = line[:position]
        fields = line.split()
        if not fields:
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"{source}: line {number}: non-numeric entry") from error
        if len(values) not in (3, 4):
            raise ValueError(
                f"{source}: line {number}: expected 'kx ky kz [weight]', got {len(values)} numbers"
            )
        rows.append(values)
    if not rows:
        raise ValueError(f"{source}: no k-points found")
    if len({len(row) for row in rows}) != 1:
        raise ValueError(f"{source}: every row must have the same number of columns")
    return np.asarray(rows, dtype=float)


def band_path_kpoints(
    starts: np.ndarray,
    ends: np.ndarray,
    points_in_shortest_segment: int,
    *,
    rounding: str = "truncate",
) -> list[np.ndarray]:
    """K-points of each band-path segment, as in Fortran ``calc_bands``.

    ``starts``/``ends`` are the Cartesian (1/bohr) end points of every
    segment.  The shortest segment gets ``points_in_shortest_segment`` points
    and the others proportionally more, so the point density along the path is
    constant.  ``rounding="truncate"`` (the default, this port's rule and that
    of the previous Fortran tree): ``nkpt = int(length / min_length * points)``,
    the Fortran integer assignment.  ``rounding="nint"`` (PARSEC 1.5,
    ``utilities/calc_bands.f90:152-158``, which the parser takes where the
    input writes ``Bandstruc_Points``): ``nint(length / min_length * points)``.
    A segment does not include its end point -- that is the first point of the
    next segment -- except the last segment, which gets one extra point to
    close the path (``int(...) + 1``, or 1.5's ``nint(... + 1)``).
    """

    if rounding not in ("truncate", "nint"):
        raise ValueError("rounding must be truncate or nint")
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("segment end points must be two (n, 3) arrays")
    if starts.shape[0] < 1:
        raise ValueError("a band path needs at least one segment")
    if points_in_shortest_segment < 1:
        raise ValueError("bandstruc_points must be at least 1")
    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("band-path segments must have non-zero length")
    shortest = float(lengths.min())
    segments = []
    last = starts.shape[0] - 1
    for index, (start, end, length) in enumerate(zip(starts, ends, lengths)):
        if rounding == "truncate":
            count = int(length / shortest * points_in_shortest_segment)
            if index == last:
                count += 1
        else:
            scaled = length / shortest * points_in_shortest_segment
            if index == last:
                scaled += 1.0
            # Fortran nint: halves away from zero (the values are positive).
            count = int(np.floor(scaled + 0.5))
        if index == last:
            fractions = np.arange(count) / (count - 1) if count > 1 else np.zeros(1)
        else:
            fractions = np.arange(count) / count
        segments.append(start + fractions[:, None] * (end - start))
    return segments


IBZ_TOLERANCE = 1.0e-8
"""``tol_bz`` of ``irrbz.f90`` (and of ``translate01``)."""


def translate01(vector) -> list[float]:
    """Fortran ``translate01`` (``irrbz.f90:213-254``), component by component.

    ``q = mod(k + 1/2, 1) - 1/2``, set to zero below ``1e-8`` in magnitude and
    raised by one when negative.  The result lies in ``[0, 1)`` for
    ``k > -1/2`` (every input of ``irrbz``); the operations are those of the
    Fortran, so the bits agree.
    """

    result = []
    for value in vector:
        shifted = float(value) + 0.5
        shifted = math.fmod(shifted, 1.0)
        shifted = shifted - 0.5
        if abs(shifted) < IBZ_TOLERANCE:
            shifted = 0.0
        if shifted < 0.0:
            shifted = shifted + 1.0
        result.append(shifted)
    return result


def fold_to_centered_zone(fractions) -> np.ndarray:
    """Reciprocal-lattice coordinates folded component by component into ``(-1/2, 1/2]``.

    ``f - ceil(f - 1/2)``: the representative of ``f`` modulo the reciprocal lattice with the smallest
    components (the shortest one of an orthorhombic cell), PARSEC 1.5's intent in ``irrbz.f90:132-157``, whose
    swap never runs (``norm2`` carries ``+ eight``).  Exactly ``1/2`` stays ``1/2`` (``-1/2`` becomes ``1/2``): a
    component in ``[0, 1/2]`` keeps its bits, so a grid whose points all lie there is unchanged, and folding a
    folded set changes nothing.  The finite-difference Bloch operator ``-nabla^2 - 2i k.grad + k^2`` is not
    invariant under ``k -> k + G``; with the folded points a run without symmetry takes the k-points a symmetric
    run reaches by rotating its representatives.
    """

    values = np.asarray(fractions, dtype=np.float64)
    return values - np.ceil(values - 0.5)


def _times(matrix, vector) -> list[float]:
    """``matvec3('N', matrix, vector)`` from ``+0.0``, in Fortran's order."""

    return [
        ((0.0 + matrix[i][0] * vector[0]) + matrix[i][1] * vector[1]) + matrix[i][2] * vector[2]
        for i in range(3)
    ]


def _dot(vector) -> float:
    return ((0.0 + vector[0] * vector[0]) + vector[1] * vector[1]) + vector[2] * vector[2]


def _translate01_rows(values: np.ndarray) -> np.ndarray:
    """:func:`translate01` of every row of ``values`` (elementwise the same)."""

    shifted = np.fmod(values + 0.5, 1.0) - 0.5
    shifted = np.where(np.abs(shifted) < IBZ_TOLERANCE, 0.0, shifted)
    return np.where(shifted < 0.0, shifted + 1.0, shifted)


def _first_equivalent(krot, representatives: np.ndarray, inversions, bvec) -> int | None:
    """The first representative ``k_IBZ`` with ``|B translate01(krot - inv k_IBZ)|^2 < tol``.

    In Fortran's order: representatives in turn, for each ``inv = +1`` then
    ``-1`` (``irrbz.f90:96-127``).
    """

    if representatives.shape[0] == 0:
        return None
    found = None
    for inv in inversions:
        difference = _translate01_rows(np.asarray(krot)[None, :] - inv * representatives)
        cartesian = [
            ((0.0 + bvec[i][0] * difference[:, 0]) + bvec[i][1] * difference[:, 1])
            + bvec[i][2] * difference[:, 2]
            for i in range(3)
        ]
        norm2 = ((0.0 + cartesian[0] * cartesian[0]) + cartesian[1] * cartesian[1]) + (
            cartesian[2] * cartesian[2]
        )
        hits = np.flatnonzero(norm2 < IBZ_TOLERANCE)
        if hits.size and (found is None or hits[0] < found):
            found = int(hits[0])
    return found


@dataclass(frozen=True)
class MonkhorstPackSampling:
    """Fortran PARSEC's Monkhorst-Pack sampling, reduced to the irreducible wedge.

    ``fractional`` are the representatives in reciprocal-lattice
    coordinates (each is the first grid point met of its star), folded into
    ``(-1/2, 1/2]`` (:func:`fold_to_centered_zone`; ``irrbz`` keeps them in
    ``[0, 1)``); ``k_points = B f`` are Cartesian (1/bohr, rows),
    the k-points of the Bloch Hamiltonians.  ``counts`` are the numbers of
    full-grid points each represents, ``weights`` the normalized weights
    ``counts/N`` (renormalized to sum one as ``init_var.F90:365-367``).
    ``kinetic_energies`` are irrbz's ``E_kin`` column (Ry; see
    :func:`irreducible_monkhorst_pack` for what it measures).

    ``grid_map`` is Fortran's ``kptgrid``: ``grid_map[i, j, l]`` is the
    0-based index into ``k_points`` of the full-grid point
    ``((i + s1)/n1, (j + s2)/n2, (l + s3)/n3)``, ``i = 0 .. n1-1`` (Fortran's
    ``kptgrid(i+1, j+1, l+1) - 1``); :meth:`full_grid_fractional` lists those
    points.  The tetrahedron DOS needs it.  ``time_reversal`` says whether
    ``k`` and ``-k`` were identified; ``operations`` is the order of the group
    used.
    """

    dimensions: tuple[int, int, int]
    shift: tuple[float, float, float]
    fractional: np.ndarray
    k_points: np.ndarray
    weights: np.ndarray
    counts: np.ndarray
    kinetic_energies: np.ndarray
    grid_map: np.ndarray
    time_reversal: bool = True
    operations: int = 1

    @property
    def count(self) -> int:
        return int(self.k_points.shape[0])

    @property
    def full_grid_size(self) -> int:
        return int(np.prod(self.dimensions))

    @property
    def printed_weights(self) -> np.ndarray:
        """``count/N`` as ``irrbz`` prints it (before the renormalization)."""

        return np.asarray(self.counts, dtype=np.float64) / float(self.full_grid_size)

    def full_grid_fractional(self) -> np.ndarray:
        """``(n1, n2, n3, 3)``: the fractional coordinates of every full-grid point."""

        n1, n2, n3 = self.dimensions
        s1, s2, s3 = self.shift
        axes = [
            (np.arange(n, dtype=np.float64) + s) / float(n)
            for n, s in ((n1, s1), (n2, s2), (n3, s3))
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack(grids, axis=-1)


def irreducible_monkhorst_pack(
    lattice_vectors,
    dimensions,
    shift,
    reciprocal_rotations=None,
    *,
    time_reversal: bool = True,
    fold: bool = True,
) -> MonkhorstPackSampling:
    """Translation of ``irrbz.f90`` (lines 12-213): the IBZ of a shifted MP grid.

    The full grid is ``k = ((i-1) + s_i)/n_i`` (reciprocal-lattice
    coordinates), visited with the last index fastest.  A point is rotated by
    every ``G`` of ``reciprocal_rotations`` (Fortran ``symm%gmtrx``, the
    identity first; ``None`` is the identity alone), folded by
    :func:`translate01`, and compared with every representative found so far,
    directly and (``time_reversal``) with the sign reversed:
    ``|B translate01(G k - inv k_IBZ)|^2 < 1e-8``.  The first match counts the
    point to that representative; a point without one becomes a new
    representative, stored unfolded.  Fortran's "replace by the shortest
    member" branch is kept as written, which makes it dead (``norm2`` has
    ``+ 8`` added): the representative is the first member met.  With
    ``fold`` (the default, a decision of this port) the representatives are
    then folded into ``(-1/2, 1/2]`` (:func:`fold_to_centered_zone`), which
    PARSEC 1.5 intends and does not do: its k-points stay in ``[0, 1)``, so a
    run without symmetry differs from the symmetric one through the
    finite-difference Bloch operator (Si8, MP 2x2x2 shifted, h 0.64: 15 mRy).
    The counts, weights, ``grid_map`` and ``E_kin`` do not depend on it;
    ``fold=False`` gives PARSEC 1.5's set.  ``E_kin``
    of a new representative is ``|B (k' - anint(k'))|^2`` with ``k'`` the
    point rotated by the *last* operation, as the Fortran computes it.

    ``lattice_vectors`` are rows in bohr; ``B`` is the reciprocal lattice of
    ``inipbc.f90`` (:func:`~parsec_python.Grid.space_group.fortran_reciprocal_lattice`).
    Raises ``ValueError`` where the counts do not add up to the grid (PARSEC
    error 241).
    """

    dims = tuple(int(value) for value in dimensions)
    if len(dims) != 3 or any(value < 1 for value in dims):
        raise ValueError("dimensions must be three positive integers")
    shifts = tuple(float(value) for value in shift)
    if len(shifts) != 3 or any(not math.isfinite(value) or value < 0.0 for value in shifts):
        raise ValueError("the Monkhorst-Pack shift must be three non-negative numbers")
    if reciprocal_rotations is None:
        rotations = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    else:
        rotations = [
            [[float(value) for value in row] for row in matrix]
            for matrix in np.asarray(reciprocal_rotations).reshape(-1, 3, 3)
        ]
    # bvec(i, j): component i of reciprocal vector j; rows of B are the vectors.
    b_rows = fortran_reciprocal_lattice(lattice_vectors)
    bvec = [[float(b_rows[j][i]) for j in range(3)] for i in range(3)]
    n1, n2, n3 = dims
    kibz: list[list[float]] = []
    counts: list[int] = []
    ekin: list[float] = []
    grid_map = np.zeros(dims, dtype=np.int64)
    inversions = (1.0, -1.0) if time_reversal else (1.0,)
    # The representatives so far, one row each, for the comparisons below
    # (elementwise the same arithmetic as the scalar Fortran loop).
    stored = np.zeros((n1 * n2 * n3 + 1, 3))
    for ii in range(n1):
        for jj in range(n2):
            for ll in range(n3):
                kprot = [
                    (float(ii) + shifts[0]) / float(n1),
                    (float(jj) + shifts[1]) / float(n2),
                    (float(ll) + shifts[2]) / float(n3),
                ]
                new_point = True
                krot = kprot
                for gmtrx in rotations:
                    krot = translate01(_times(gmtrx, kprot))
                    if not kibz:
                        kibz.append(list(krot))
                        stored[0] = krot
                        ktest = [value - fortran_anint(value) for value in krot]
                        ekin.append(_dot(_times(bvec, ktest)))
                        counts.append(0)
                        grid_map[ii, jj, ll] = 0
                    itest = _first_equivalent(krot, stored[: len(kibz)], inversions, bvec)
                    if itest is None:
                        continue
                    # The first match ends the search: Fortran goes on through
                    # the remaining operations with new_kpt false, which
                    # changes nothing (the E_kin of the last operation is
                    # used for new points only).
                    new_point = False
                    ktest = [value - fortran_anint(value) for value in kprot]
                    norm2 = _dot(_times(bvec, ktest)) + 8.0
                    grid_map[ii, jj, ll] = itest
                    norm1 = _dot(_times(bvec, kibz[itest]))
                    if norm2 < norm1:
                        kibz[itest] = translate01(kprot)
                        stored[itest] = kibz[itest]
                        ekin[itest] = norm2
                    counts[itest] += 1
                    break
                if new_point:
                    kibz.append(list(kprot))
                    stored[len(kibz) - 1] = kprot
                    ktest = [value - fortran_anint(value) for value in krot]
                    ekin.append(_dot(_times(bvec, ktest)))
                    counts.append(1)
                    grid_map[ii, jj, ll] = len(kibz) - 1
    total = n1 * n2 * n3
    if sum(counts) != total:
        raise ValueError(
            f"incorrect Brillouin zone: {sum(counts)} of {total} grid points "
            "counted (PARSEC error 241)"
        )
    fractional = np.array(kibz, dtype=np.float64)
    if fold:
        # the representatives themselves (the search above compares them modulo G, as irrbz does)
        fractional = fold_to_centered_zone(fractional)
    k_points = np.array([_times(bvec, [float(v) for v in point]) for point in fractional], dtype=np.float64)
    weights = np.array([float(count) / float(total) for count in counts], dtype=np.float64)
    norm = 0.0
    for value in weights:
        norm = norm + float(value)
    weights = weights / norm
    return MonkhorstPackSampling(
        dimensions=dims,
        shift=shifts,
        fractional=fractional,
        k_points=k_points,
        weights=weights,
        counts=np.asarray(counts, dtype=np.int64),
        kinetic_energies=np.asarray(ekin, dtype=np.float64),
        grid_map=grid_map,
        time_reversal=bool(time_reversal),
        operations=len(rotations),
    )


def kpoint_sampling_details(problem) -> MonkhorstPackSampling | None:
    """The Fortran-path sampling of a ``SinglePointInput``, or ``None``.

    Only an input with ``monkhorst_pack_shift`` (PARSEC's
    ``Monkhorst_Pack_Shift`` block) takes it: the grid of
    ``monkhorst_pack_dimensions`` with that shift, reduced by the input's
    ``space_group`` (the identity where it is ``None``, e.g. with
    ``Ignore_Symmetry``) and by time reversal unless the calculation is
    non-collinear (``irrbz.f90:96-99``).  A hybrid functional takes time
    reversal only (its exchange operator sums over the full mesh,
    :mod:`parsec_python.SCF.hybrid_periodic`).
    """

    shift = getattr(problem, "monkhorst_pack_shift", None)
    if shift is None or problem.monkhorst_pack_dimensions is None:
        return None
    group = None if _hybrid(problem) else getattr(problem, "space_group", None)
    lattice = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(problem.periodic_cell.lattice_vectors, dtype=float)
    )
    rotations = (
        None
        if group is None
        else tuple(tuple(tuple(int(v) for v in row) for row in matrix) for matrix in group.reciprocal_rotations)
    )
    return _cached_irreducible_monkhorst_pack(
        lattice,
        tuple(problem.monkhorst_pack_dimensions),
        tuple(float(value) for value in shift),
        rotations,
        not problem.scf.noncollinear,
    )


@lru_cache(maxsize=8)
def _cached_irreducible_monkhorst_pack(lattice, dimensions, shift, rotations, time_reversal):
    """One computation per input for the CLI, the report and ``qpoints.dat``.

    Callers must not modify the arrays of the returned (shared) object.
    """

    sampling = irreducible_monkhorst_pack(
        np.asarray(lattice, dtype=float),
        dimensions,
        shift,
        None if rotations is None else np.asarray(rotations, dtype=np.int64),
        time_reversal=time_reversal,
    )
    for array in (
        sampling.fractional,
        sampling.k_points,
        sampling.weights,
        sampling.counts,
        sampling.kinetic_energies,
        sampling.grid_map,
    ):
        array.flags.writeable = False
    return sampling


def _hybrid(problem) -> bool:
    return bool(getattr(getattr(problem, "scf", None), "is_hybrid", False))


def time_reversal_reduced(k_points, weights, lattice_vectors) -> tuple[np.ndarray, np.ndarray]:
    """The representatives of a uniform mesh of equal weights under time reversal (``k ~ -k``).

    The first point of each pair is kept with twice the weight; a point with
    ``-k = k`` modulo the reciprocal lattice of ``lattice_vectors`` keeps its
    own.  The order of the first members is kept.
    """

    k_points = np.atleast_2d(np.asarray(k_points, dtype=float))
    weights = np.asarray(weights, dtype=float)
    fractions = k_points @ np.linalg.inv(reciprocal_lattice(lattice_vectors))
    taken = np.zeros(k_points.shape[0], dtype=bool)
    kept, kept_weights = [], []
    for index in range(k_points.shape[0]):
        if taken[index]:
            continue
        taken[index] = True
        weight = float(weights[index])
        total = fractions + fractions[index]
        partners = np.flatnonzero(np.all(np.abs(total - np.round(total)) < 1.0e-8, axis=1) & ~taken)
        if partners.size:
            taken[partners[0]] = True
            weight += float(weights[partners[0]])
        kept.append(index)
        kept_weights.append(weight)
    return k_points[kept].copy(), np.asarray(kept_weights, dtype=float)


def kpoint_sampling(problem) -> tuple[np.ndarray, np.ndarray] | None:
    """``(k_points, weights)`` requested by a ``SinglePointInput``, or ``None``.

    An explicit list (``problem.kpoints``), Fortran's reduced grid
    (``problem.monkhorst_pack_shift`` given, :func:`kpoint_sampling_details`)
    or this port's unreduced Monkhorst-Pack grid
    (``problem.monkhorst_pack_dimensions``); ``None`` means Gamma only.  For a
    hybrid functional the port's grid and a list of equal weights are reduced
    by time reversal (:func:`time_reversal_reduced`) and Fortran's grid by
    time reversal only; the exchange checks that the set is a uniform mesh
    (:mod:`parsec_python.SCF.hybrid_periodic`).
    """

    if getattr(problem, "kpoints", None) is not None:
        k_points = np.asarray(problem.kpoints, dtype=float)
        weights = np.asarray(problem.kpoint_weights, dtype=float)
        if _hybrid(problem) and np.allclose(weights, weights[0], rtol=1.0e-12, atol=0.0):
            # a list of equal weights (a full mesh written out): its time-reversal representatives
            return time_reversal_reduced(k_points, weights, problem.periodic_cell.lattice_vectors)
        return k_points, weights
    details = kpoint_sampling_details(problem)
    if details is not None:
        return details.k_points.copy(), details.weights.copy()
    if problem.monkhorst_pack_dimensions is not None:
        k_points, weights = monkhorst_pack_grid(problem.periodic_cell, problem.monkhorst_pack_dimensions)
        if _hybrid(problem):
            return time_reversal_reduced(k_points, weights, problem.periodic_cell.lattice_vectors)
        return k_points, weights
    return None


__all__ = [
    "IBZ_TOLERANCE",
    "KPOINT_UNITS",
    "MonkhorstPackSampling",
    "explicit_kpoints",
    "band_path_kpoints",
    "fold_to_centered_zone",
    "irreducible_monkhorst_pack",
    "kpoint_sampling",
    "kpoint_sampling_details",
    "monkhorst_pack_grid",
    "read_kpoint_file",
    "time_reversal_reduced",
    "translate01",
]
