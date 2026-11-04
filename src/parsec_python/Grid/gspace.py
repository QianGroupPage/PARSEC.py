"""Reciprocal space of a periodic cell as Fortran PARSEC builds it for BerkeleyGW.

Translation of ``g_space.f90``: ``g_space`` (the G-vectors of the FFT box of
the real-space grid, sorted by length and collected into symmetry stars),
``generate_gspace`` with ``findbound`` (the G-spheres ``|k + G|^2 <= G_max^2``
of the potential and of every k-point, Bernd Pfrommer's code from PARATEC),
``adjustfft`` with ``fastnum`` (the FFT grid of the potential sphere), and the
lattice data of ``inipbc.f90`` l.50-75 (``adot``, ``bvec``, ``bdot``).  The
BerkeleyGW writer (:mod:`parsec_python.Output.berkeleygw`, ``gw_write.F90``)
uses them; nothing here reads or writes files.

Conventions (Fortran's, kept so that the G-vector lists match record by
record):

* G-vectors are integer triples ``(g1, g2, g3)`` in units of the reciprocal
  lattice vectors ``b_i`` (``G = g1 b1 + g2 b2 + g3 b3``); ``bdot = B^T B``
  is the metric, ``|G|^2 = g . bdot . g`` in bohr^-2 (= Ry for ``|G|^2``
  read as a kinetic energy).
* A symmetry operation acts on G with the integer matrix ``gmtrx`` (the
  transpose of the real-space ``rmtrx``,
  :attr:`parsec_python.Grid.space_group.SpaceGroup.reciprocal_rotations`) and
  carries the fractional translation ``tnp`` (lattice coordinates); its phase
  on the rotated vector ``G'`` is ``exp(i 2 pi tnp . G')``.
* k-points are in reciprocal-lattice (crystal) coordinates.

Floating-point expressions follow the Fortran statements term by term
(``matvec3``/``dot_product`` sums from ``+0.0`` in index order, the explicit
metric of ``generate_gspace``), so lengths, bins and sphere memberships are
decided on the same doubles.  Units: bohr, bohr^-1, Ry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .space_group import fortran_nint, fortran_reciprocal_lattice, lattice_columns

TWO_PI = 2.0 * math.pi
"""``twopi`` of ``const.f90``."""

STAR_TOLERANCE = 1.0e-8
"""``delta`` of ``g_space``: two ``|G|^2`` closer than this are one length."""

FFT_PRIMES = (2, 3, 5)
"""The prime factors ``fastnum`` allows in an FFT dimension."""


@dataclass(frozen=True)
class LatticeMetrics:
    """The lattice data ``inipbc.f90`` l.50-75 computes (``pbc%adot/bvec/bdot``).

    ``lattice`` holds the lattice vectors as columns (Fortran ``latt_vec``),
    ``adot = latt_vec^T latt_vec``; ``reciprocal`` holds ``b_i`` as columns
    (Fortran ``bvec = 2 pi inv(latt_vec^T)`` by ``mtrxin``) and
    ``bdot = bvec^T bvec``.  All ``(3, 3)``, bohr and bohr^-1.
    """

    lattice: np.ndarray
    adot: np.ndarray
    reciprocal: np.ndarray
    bdot: np.ndarray


def _fortran_matmul_transposed(matrix: np.ndarray) -> np.ndarray:
    """``matmul(transpose(m), m)`` with each sum from ``+0.0`` in index order."""

    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            total = 0.0
            for k in range(3):
                total = total + float(matrix[k, i]) * float(matrix[k, j])
            out[i, j] = total
    return out


def fortran_lattice_metrics(lattice_vectors) -> LatticeMetrics:
    """:class:`LatticeMetrics` of the cell with lattice vectors as rows (bohr)."""

    columns = np.asarray(lattice_columns(lattice_vectors), dtype=float)
    reciprocal = np.asarray(fortran_reciprocal_lattice(lattice_vectors), dtype=float).T.copy()
    return LatticeMetrics(
        lattice=columns,
        adot=_fortran_matmul_transposed(columns),
        reciprocal=reciprocal,
        bdot=_fortran_matmul_transposed(reciprocal),
    )


def g_squared_fortran(vectors: np.ndarray, bdot: np.ndarray) -> np.ndarray:
    """``gm`` of ``g_space``: ``dot_product(g, matvec3('N', bdot, g))`` per row."""

    d = np.asarray(vectors, dtype=np.float64)
    b = np.asarray(bdot, dtype=np.float64)
    t = [((0.0 + b[i, 0] * d[:, 0]) + b[i, 1] * d[:, 1]) + b[i, 2] * d[:, 2] for i in range(3)]
    return ((0.0 + d[:, 0] * t[0]) + d[:, 1] * t[1]) + d[:, 2] * t[2]


@dataclass(frozen=True)
class GVectorStars:
    """The G-vectors of the FFT box collected into stars (``pbc`` of ``g_space``).

    ``vectors`` ``(ng, 3)`` are Fortran's ``kgv`` in its final order: by
    length, then lexicographically within a length, then rearranged into
    stars (each star's members adjacent, ``G`` followed by ``-G``).
    ``star_index`` (0-based ``inds - 1``) gives the star of each vector,
    ``star_sizes`` (``mstar``) the members of each star, ``kinetic_energies``
    (``ek``) ``|G|^2/2`` of each star.  ``phases`` are the complex phase
    factors (the average of ``exp(i 2 pi tnp . G)`` over the operations that
    map the star's first vector onto this one; zero for every member of a
    star whose first phase is not one) and ``conjugation`` is ``+1``, or
    ``-1`` for a ``-G`` that no operation reaches from ``G``.  ``kmax`` is the
    box half-width after ``g_space`` (``maxval(kgv)``).
    """

    vectors: np.ndarray
    star_index: np.ndarray
    phases: np.ndarray
    conjugation: np.ndarray
    star_sizes: np.ndarray
    kinetic_energies: np.ndarray
    kmax: tuple[int, int, int]

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def star_count(self) -> int:
        return int(self.star_sizes.shape[0])


def box_half_widths(points_per_axis) -> tuple[int, int, int]:
    """``kmax_i = nint((n_i + 1/4)/2)`` of ``inipbc.f90`` l.163-169 (3-D periodic)."""

    return tuple(fortran_nint((float(n) + 0.25) / 2.0) for n in points_per_axis)


def _length_sorted_vectors(kmax, bdot) -> tuple[np.ndarray, np.ndarray]:
    """The box vectors in ``g_space``'s order before the stars (l.80-194).

    Fortran heap-sorts by ``gm`` and then sorts each run of lengths within
    ``delta`` of the run's first entry by ``kgv(1)``, ``kgv(2)``, ``kgv(3)``.
    Inside a run that order is total, so any sort by length followed by the
    same runs gives Fortran's list.
    """

    k1, k2, k3 = (int(value) for value in kmax)
    i, j, k = np.meshgrid(
        np.arange(-k1, k1 + 1), np.arange(-k2, k2 + 1), np.arange(-k3, k3 + 1), indexing="ij"
    )
    vectors = np.column_stack([i.reshape(-1), j.reshape(-1), k.reshape(-1)]).astype(np.int64)
    gm = g_squared_fortran(vectors, bdot)
    order = np.argsort(gm, kind="stable")
    vectors = vectors[order]
    gm = gm[order]
    ng = vectors.shape[0]
    # Runs (l.152-194): start at the second entry; a run ends at i when the
    # next length exceeds the run's first by more than delta.
    starts = [1]
    istar = 1
    for position in range(1, ng):
        diff = 1.0 if position == ng - 1 else gm[position + 1] - gm[istar]
        if diff > STAR_TOLERANCE:
            istar = position + 1
            starts.append(istar)
    for first, after in zip(starts[:-1], starts[1:]):
        if after - first > 1:
            block = vectors[first:after]
            local = np.lexsort((block[:, 2], block[:, 1], block[:, 0]))
            vectors[first:after] = block[local]
            gm[first:after] = gm[first:after][local]
    return vectors, gm


def g_space_stars(kmax, bdot, rotations, translations) -> GVectorStars:
    """``g_space``: every G of the box ``|g_i| <= kmax_i`` in symmetry stars.

    ``rotations`` ``(ntrans, 3, 3)`` are ``gmtrx`` (acting on G) and
    ``translations`` ``(ntrans, 3)`` the ``tnp`` of the space group.  The star
    collection (``g_space.f90`` l.199-383) is followed statement by statement,
    including its exchanges of vectors, so that every star has Fortran's
    first vector and every vector Fortran's phase.  Error 261 (no G-vector)
    is a :class:`ValueError`.
    """

    kmax = tuple(int(value) for value in kmax)
    if (2 * kmax[0] + 1) * (2 * kmax[1] + 1) * (2 * kmax[2] + 1) <= 1:
        raise ValueError("stopped in gspace: no G-vector besides G = 0 (PARSEC error 261)")
    rotations = np.asarray(rotations, dtype=np.int64).reshape(-1, 3, 3)
    translations = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    sorted_vectors, sorted_gm = _length_sorted_vectors(kmax, bdot)
    ng = sorted_vectors.shape[0]
    kgv = [tuple(int(value) for value in row) for row in sorted_vectors]
    gm = [float(value) for value in sorted_gm]
    where = {vector: position for position, vector in enumerate(kgv)}
    phase = np.zeros(ng, dtype=np.complex128)
    conj = np.zeros(ng)
    inds = np.zeros(ng, dtype=np.int64)
    sizes: list[int] = [1]
    energies: list[float] = [0.0]
    phase[0] = 1.0
    conj[0] = 1.0

    def swap(a: int, b: int) -> None:
        kgv[a], kgv[b] = kgv[b], kgv[a]
        gm[a], gm[b] = gm[b], gm[a]
        where[kgv[a]] = a
        where[kgv[b]] = b

    def image_phase(images, expt, target) -> tuple[int, complex]:
        """Count and sum of ``expt`` over the operations mapping onto ``target``."""
        count = 0
        total = 0.0 + 0.0j
        for index, image in enumerate(images):
            if image == target:
                total = total + expt[index]
                count += 1
        return count, total

    def place_inverse(position: int, images, expt) -> None:
        """Bring ``-G`` of ``position`` next to it with its phase (l.243-283)."""
        vector = kgv[position]
        inverse = (-vector[0], -vector[1], -vector[2])
        found = where.get(inverse)
        if found is None or found < position + 1:
            return
        if found != position + 1:
            swap(position + 1, found)
        count, total = image_phase(images, expt, kgv[position + 1])
        if count != 0:
            phase[position + 1] = total / float(count)
            conj[position + 1] = 1.0
        else:
            phase[position + 1] = phase[position]
            conj[position + 1] = -1.0
        sizes[-1] += 1
        inds[position + 1] = len(sizes) - 1

    images: list[tuple[int, int, int]] = []
    expt = np.zeros(0, dtype=np.complex128)
    iprot = 0
    iavail = 1
    itest = 1
    while True:
        diff = 1.0 if itest > ng - 1 else abs(gm[itest] - gm[iprot])
        if diff > STAR_TOLERANCE:
            # A new star, its first vector at iavail (l.217-288).
            iprot = iavail
            sizes.append(1)
            energies.append(gm[iprot] / 2.0)
            prototype = np.asarray(kgv[iprot], dtype=np.int64)
            ktran = rotations @ prototype
            rotated = ktran.astype(np.float64) * TWO_PI
            fi = (
                (0.0 + translations[:, 0] * rotated[:, 0]) + translations[:, 1] * rotated[:, 1]
            ) + translations[:, 2] * rotated[:, 2]
            expt = np.cos(fi) + 1j * np.sin(fi)
            images = [tuple(int(value) for value in row) for row in ktran]
            count, total = image_phase(images, expt, kgv[iprot])
            phase[iprot] = total / float(count)
            conj[iprot] = 1.0
            inds[iprot] = len(sizes) - 1
            place_inverse(iprot, images, expt)
            iavail = iprot + 2
            itest = iprot + 2
        else:
            count, total = image_phase(images, expt, kgv[itest])
            if count != 0:
                # A member of the current star (l.305-372).
                if itest != iavail:
                    swap(iavail, itest)
                phase[iavail] = total / float(count)
                conj[iavail] = 1.0
                sizes[-1] += 1
                inds[iavail] = len(sizes) - 1
                place_inverse(iavail, images, expt)
                itest = itest + 2 if itest == iavail else itest + 1
                iavail += 2
            else:
                itest += 1
        if iavail > ng - 1:
            break

    vectors = np.asarray(kgv, dtype=np.int64)
    star_sizes = np.asarray(sizes, dtype=np.int64)
    # Stars whose first phase is not one carry no component (l.391-404).
    first = 1
    for star in range(1, star_sizes.size):
        if abs(phase[first] - 1.0) > STAR_TOLERANCE:
            phase[first : first + star_sizes[star]] = 0.0
        first += int(star_sizes[star])
    return GVectorStars(
        vectors=vectors,
        star_index=inds,
        phases=phase,
        conjugation=conj,
        star_sizes=star_sizes,
        kinetic_energies=np.asarray(energies, dtype=np.float64),
        kmax=tuple(int(value) for value in vectors.max(axis=0)),
    )


def find_bound(radius: float, reciprocal: np.ndarray) -> tuple[int, int, int]:
    """``findbound``: box half-widths along ``b_1, b_2, b_3`` that hold a sphere.

    ``reciprocal`` holds the ``b_i`` as columns.  ``int(q / proj_i)`` with
    ``proj_i`` the distance between the lattice planes spanned by the other
    two vectors; parallel vectors stop PARSEC (here a :class:`ValueError`).
    """

    b = np.asarray(reciprocal, dtype=np.float64)
    bi, bj, bk = b[:, 0], b[:, 1], b[:, 2]

    def scp(u, v) -> float:
        return float(u[0] * v[0] + u[1] * v[1] + u[2] * v[2])

    volp = (
        bi[0] * (bj[1] * bk[2] - bj[2] * bk[1])
        + bi[1] * (bj[2] * bk[0] - bj[0] * bk[2])
        + bi[2] * (bj[0] * bk[1] - bj[1] * bk[0])
    )
    bounds = []
    for first, second, name in ((bj, bk, "bj,bk"), (bk, bi, "bk,bi"), (bi, bj, "bi,bj")):
        cross = math.sqrt(scp(first, first) * scp(second, second) - scp(first, second) ** 2)
        if cross <= 1.0e-6:
            raise ValueError(f"error in findbound: {name} parallel")
        projection = abs(volp / cross)
        bounds.append(int(float(radius) / projection))
    return tuple(bounds)


def fast_fft_size(size: int) -> int:
    """``fastnum``: the smallest ``n >= size`` of the form ``2^a 3^b 5^c``."""

    value = int(size)
    while True:
        remainder = value
        for prime in FFT_PRIMES:
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return value
        value += 1


def translation_fft_factors(translations) -> tuple[int, int, int]:
    """``fracfac`` of ``adjustfft``: what each FFT dimension must be a multiple of.

    For direction ``i`` a denominator ``j`` of 1..4 is present when some
    operation has ``|tnp_i| > 1e-5`` and ``|nint(tnp_i j)| = 1``; the factor is
    the product of the present ones, halved when 2 and 4 both are (Fortran's
    rule, kept as written: a translation of 1/4 gives 12).
    """

    tnp = np.asarray(translations, dtype=np.float64).reshape(-1, 3)
    factors = []
    for direction in range(3):
        present = [False, False, False, False]
        for value in tnp[:, direction]:
            if abs(float(value)) > 1.0e-5:
                for slot, denominator in enumerate((1, 2, 3, 4)):
                    if abs(abs(fortran_nint(float(value) * denominator)) - 1.0) < 1.0e-5:
                        present[slot] = True
        factor = 1
        for slot, denominator in enumerate((1, 2, 3, 4)):
            if present[slot]:
                factor *= denominator
        if present[1] and present[3]:
            factor //= 2
        factors.append(factor)
    return tuple(factors)


def adjust_fft(sizes, translations) -> tuple[int, int, int]:
    """``adjustfft``: each FFT dimension raised to ``fastnum`` and the translations."""

    factors = translation_fft_factors(translations)
    adjusted = []
    for size, factor in zip(sizes, factors):
        value = int(size)
        while True:
            value = fast_fft_size(value)
            if value % factor == 0:
                break
            value += 1
        adjusted.append(value)
    return tuple(adjusted)


@dataclass(frozen=True)
class GSphere:
    """The G-vectors of ``generate_gspace``: ``|k + G|^2 <= G_max^2``.

    ``vectors`` ``(n, 3)`` in Fortran's loop order (``g1`` slowest, ``g3``
    fastest, each from ``-kmax`` to ``kmax``), ``kmax`` the box of
    ``findbound``, ``k_point`` the crystal coordinates ``rk``, ``radius``
    ``G_max`` (bohr^-1), ``fft_size`` the FFT grid (potential sphere only,
    ``None`` for a k-point sphere) and ``zero_index`` the row of ``G = 0``
    (``ig0 - 1``, ``-1`` when the sphere does not hold it).
    """

    vectors: np.ndarray
    kmax: tuple[int, int, int]
    k_point: tuple[float, float, float]
    radius: float
    fft_size: tuple[int, int, int] | None
    zero_index: int

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])

    def index_map(self) -> dict[tuple[int, int, int], int]:
        """``indexg``: the row of every vector (0-based)."""

        return {tuple(int(v) for v in row): position for position, row in enumerate(self.vectors)}


def generate_gspace(
    radius: float,
    k_point,
    metrics: LatticeMetrics,
    *,
    translations=None,
) -> GSphere:
    """``generate_gspace``: the sphere ``|k + G| <= radius`` (``k`` crystal).

    The box comes from ``findbound(radius + |k|)``; ``|k + G|^2`` is the
    explicit metric expression of ``g_space.f90`` l.554-559.  With
    ``translations`` (the potential sphere) the FFT grid ``2 kmax + 1`` is
    adjusted by :func:`adjust_fft`.
    """

    rk = np.asarray(k_point, dtype=np.float64).reshape(3)
    bdot = metrics.bdot
    product = [((0.0 + bdot[i, 0] * rk[0]) + bdot[i, 1] * rk[1]) + bdot[i, 2] * rk[2] for i in range(3)]
    rkmod = math.sqrt(((0.0 + rk[0] * product[0]) + rk[1] * product[1]) + rk[2] * product[2])
    kmax = find_bound(float(radius) + rkmod, metrics.reciprocal)
    fft_size = None
    if translations is not None:
        fft_size = adjust_fft(tuple(2 * value + 1 for value in kmax), translations)
    gmax2 = float(radius) * float(radius)
    i, j, k = np.meshgrid(
        np.arange(-kmax[0], kmax[0] + 1),
        np.arange(-kmax[1], kmax[1] + 1),
        np.arange(-kmax[2], kmax[2] + 1),
        indexing="ij",
    )
    integers = np.column_stack([i.reshape(-1), j.reshape(-1), k.reshape(-1)]).astype(np.int64)
    q1 = integers[:, 0].astype(np.float64) + rk[0]
    q2 = integers[:, 1].astype(np.float64) + rk[1]
    q3 = integers[:, 2].astype(np.float64) + rk[2]
    b = bdot
    gmod = (
        ((q1 * b[0, 0] + q2 * b[1, 0]) + q3 * b[2, 0]) * q1
        + ((q1 * b[0, 1] + q2 * b[1, 1]) + q3 * b[2, 1]) * q2
    ) + ((q1 * b[0, 2] + q2 * b[1, 2]) + q3 * b[2, 2]) * q3
    inside = gmod <= gmax2
    vectors = integers[inside]
    zero = np.flatnonzero(np.all(vectors == 0, axis=1))
    return GSphere(
        vectors=vectors,
        kmax=tuple(int(value) for value in kmax),
        k_point=tuple(float(value) for value in rk),
        radius=float(radius),
        fft_size=fft_size,
        zero_index=int(zero[0]) if zero.size else -1,
    )


def wavefunction_cutoff(steps) -> float:
    """``gw_write.F90`` l.147-150: ``min_i (pi/h_i)^2 - 1e-4`` Ry, at most 1000."""

    cutoff = 1000.0
    for step in steps:
        ratio = math.pi / float(step)
        cutoff = min(ratio * ratio - 1.0e-4, cutoff)
    return cutoff


def potential_gspace(cutoff: float, metrics: LatticeMetrics, translations) -> GSphere:
    """``generate_potential_gspace``: radius ``2 sqrt(cutoff)``, ``k = 0``, its FFT grid."""

    return generate_gspace(2.0 * math.sqrt(cutoff), (0.0, 0.0, 0.0), metrics, translations=translations)


def k_gspace(cutoff: float, k_point, metrics: LatticeMetrics) -> GSphere:
    """``generate_k_gspaces``: radius ``sqrt(cutoff)`` around ``-k`` (``|k+G|^2 <= cutoff``)."""

    return generate_gspace(math.sqrt(cutoff), k_point, metrics)


__all__ = [
    "GSphere",
    "GVectorStars",
    "LatticeMetrics",
    "STAR_TOLERANCE",
    "adjust_fft",
    "box_half_widths",
    "fast_fft_size",
    "find_bound",
    "fortran_lattice_metrics",
    "g_space_stars",
    "g_squared_fortran",
    "generate_gspace",
    "k_gspace",
    "potential_gspace",
    "translation_fft_factors",
    "wavefunction_cutoff",
]
