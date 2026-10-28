"""Brillouin-zone k-point sampling for an orthorhombic periodic cell.

Two sources are supported: a uniform, unreduced Monkhorst-Pack grid
(:func:`monkhorst_pack_grid`) and a user-supplied explicit list
(:func:`explicit_kpoints`, :func:`read_kpoint_file`, Fortran's
``Kpoint_Method: manual``).  The SCF drivers currently require every
k-point to carry the *same* weight (no symmetry-reduced weighted lists); the
Monkhorst-Pack grid satisfies that by construction and an explicit list must
too.  This deliberately keeps :mod:`~parsec_python.SCF.kpoints`'s Fermi-level
pooling simple: reusing
:func:`~parsec_python.Occupations.fermi_dirac.fermi_occupations`'s existing
scalar ``degeneracy`` parameter (``degeneracy = 2*weight``, spin-unpolarized)
requires every pooled eigenvalue to share one scale factor, which only holds
when every k-point's weight is identical.
"""

from __future__ import annotations

import numpy as np

from ..models import PeriodicCell


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
    """
    dims = tuple(int(value) for value in dimensions)
    if len(dims) != 3 or any(value < 1 for value in dims):
        raise ValueError("dimensions must be three positive integers")
    if np.allclose(cell.lattice_vectors, np.diag(np.diag(cell.lattice_vectors)), atol=1.0e-10):
        side_lengths = np.diag(cell.lattice_vectors)
        reciprocal = 2.0 * np.pi / side_lengths
    else:
        # rows b_i of 2 pi A^{-T}: k = sum_i f_i b_i
        reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(cell.lattice_vectors)).T
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
    starts: np.ndarray, ends: np.ndarray, points_in_shortest_segment: int
) -> list[np.ndarray]:
    """K-points of each band-path segment, as in Fortran ``calc_bands``.

    ``starts``/``ends`` are the Cartesian (1/bohr) end points of every
    segment.  The shortest segment gets ``points_in_shortest_segment`` points
    and the others proportionally more, so the point density along the path is
    constant: ``nkpt = int(length / min_length * points)`` (truncated, as in
    the Fortran integer assignment).  A segment does not include its end
    point -- that is the first point of the next segment -- except the last
    segment, which gets one extra point to close the path.
    """

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
        count = int(length / shortest * points_in_shortest_segment)
        if index == last:
            count += 1
            fractions = np.arange(count) / (count - 1) if count > 1 else np.zeros(1)
        else:
            fractions = np.arange(count) / count
        segments.append(start + fractions[:, None] * (end - start))
    return segments


def kpoint_sampling(problem) -> tuple[np.ndarray, np.ndarray] | None:
    """``(k_points, weights)`` requested by a ``SinglePointInput``, or ``None``.

    An explicit list (``problem.kpoints``) or a Monkhorst-Pack grid
    (``problem.monkhorst_pack_dimensions``); ``None`` means Gamma only.
    """

    if getattr(problem, "kpoints", None) is not None:
        return np.asarray(problem.kpoints, dtype=float), np.asarray(
            problem.kpoint_weights, dtype=float
        )
    if problem.monkhorst_pack_dimensions is not None:
        return monkhorst_pack_grid(problem.periodic_cell, problem.monkhorst_pack_dimensions)
    return None


__all__ = [
    "KPOINT_UNITS",
    "explicit_kpoints",
    "band_path_kpoints",
    "kpoint_sampling",
    "monkhorst_pack_grid",
    "read_kpoint_file",
]
