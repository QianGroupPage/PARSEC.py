"""Space group of a periodic cell, as Fortran PARSEC finds it.

Translation of ``symmetries.f90`` (lines 52-238) and ``symgen.f90``
(``symgen``, ``atftmt``, ``pgl``, ``rot``, ``rlv``, ``symm_ident``).  The
search tests a *fixed* set of candidate rotations: the 24 operations of the
hexagonal holohedry with the six-fold axis along z (``rot`` with ``nr = 24``)
or, when fewer than twelve of those leave the lattice invariant, the 48
operations of the cubic holohedry with the axes along x, y and z
(``nr = 48``).  A candidate is a lattice operation when it maps every lattice
vector onto a lattice vector (``pgl``, tolerance ``10 * 1e-8``); it is a
crystal operation when, with one fractional translation, it maps every atom
onto an atom of the same type (``atftmt``).  Because the candidates are fixed
and axis aligned, only cells in PARSEC's conventional orientation reveal their
full symmetry; spglib would find more for a rotated cell.

A crystal operation ``T`` acts on lattice (fractional) coordinates ``f`` as

``T(f) = M (f - t)``

with the integer matrix ``M`` (Fortran ``symm%rmtrx``) and the fractional
translation ``t`` (``symm%tnp``); in Cartesian coordinates ``T(r) = U (r - A t)``
with the orthogonal ``U = A M inv(A)`` (``symm%trans``) and the lattice vectors
``A`` as columns.  ``T`` is the inverse of the operation ``atftmt`` tested
(``x -> R x - v`` with ``t = -v``), so the set is the same group.  In
reciprocal-lattice coordinates a k-point rotates with
``G = transpose(M)`` (``symm%gmtrx``; ``B G k = R B k`` in Cartesian
coordinates).

The arithmetic follows the Fortran statement by statement (sums in Fortran's
order, each started from ``+0.0`` as ``dot_product`` does, ``nint``/``anint``
rounding halves away from zero): the fractional translations then carry the
same signs, including those of zero, that PARSEC prints.

Not ported: ``symchk`` (Fortran calls it only with an Abelian subgroup, which
the k-point runs do not build), the Abelian subgroup itself (``abelian``: this
port keeps every field on the full grid) and the multiplication table that
``atftmt`` prints at ``Output_Level >= 2``.

Units: bohr.  This module does not print; :mod:`parsec_python.Output.kpoint_output`
formats the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .point_group import GridPointGroup

ROOT3 = 1.73205080756887729352744634151
"""``root3`` of ``const.f90``."""

SYMMETRY_TOLERANCE = 1.0e-8
"""``eps`` of ``pgl`` and ``atftmt``."""

CRYSTAL_SYSTEMS = (
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "cubic",
    "trigonal",
    "hexagonal",
)
"""``cst`` of ``atftmt``, indexed by the holohedral group number ``ihg - 1``."""

_AXES = {-2: "C2", -1: "C3", 0: "C4", 1: "C6", 2: "E "}


def fortran_anint(value: float) -> float:
    """Fortran ``anint``: nearest whole number, halves away from zero (exact)."""

    value = float(value)
    whole = math.trunc(value)
    fraction = value - whole
    if fraction >= 0.5:
        whole += 1
    elif fraction <= -0.5:
        whole -= 1
    # trunc keeps the sign of zero for values in (-1, 1): -0.3 -> -0.0.
    return math.copysign(float(whole), value) if whole == 0 else float(whole)


def fortran_nint(value: float) -> int:
    """Fortran ``nint``."""

    return int(fortran_anint(value))


def _matvec(matrix, vector) -> list[float]:
    """``matvec3('N', mat, vec)``: ``dot_product(mat(i,:), vec)`` from ``+0.0``."""

    return [
        ((0.0 + matrix[i][0] * vector[0]) + matrix[i][1] * vector[1])
        + matrix[i][2] * vector[2]
        for i in range(3)
    ]


def _matvec_transposed(matrix, vector) -> list[float]:
    """``matvec3('T', mat, vec)``: ``dot_product(mat(:,i), vec)`` from ``+0.0``."""

    return [
        ((0.0 + matrix[0][i] * vector[0]) + matrix[1][i] * vector[1])
        + matrix[2][i] * vector[2]
        for i in range(3)
    ]


def _mtrxin(m) -> list[list[float]]:
    """``mtrxin`` of ``symgen.f90``: inverse by cofactors (``stop 501`` if singular)."""

    a = [[0.0] * 3 for _ in range(3)]
    a[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1]
    a[1][0] = -m[1][0] * m[2][2] + m[1][2] * m[2][0]
    a[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0]
    a[0][1] = -m[0][1] * m[2][2] + m[0][2] * m[2][1]
    a[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0]
    a[2][1] = -m[0][0] * m[2][1] + m[0][1] * m[2][0]
    a[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1]
    a[1][2] = -m[0][0] * m[1][2] + m[0][2] * m[1][0]
    a[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    determinant = m[0][0] * a[0][0] + m[0][1] * a[1][0] + m[0][2] * a[2][0]
    if abs(determinant) < 1.0e-5:
        raise ValueError("the lattice vectors are (nearly) linearly dependent (PARSEC stop 501)")
    return [[a[i][j] / determinant for j in range(3)] for i in range(3)]


def lattice_columns(lattice_vectors) -> list[list[float]]:
    """``alatt`` (Fortran ``latt_vec``): the lattice vectors as columns."""

    rows = np.asarray(lattice_vectors, dtype=float)
    if rows.shape != (3, 3):
        raise ValueError("lattice_vectors must have shape (3, 3)")
    return [[float(rows[column][row]) for column in range(3)] for row in range(3)]


def fortran_reciprocal_lattice(lattice_vectors) -> np.ndarray:
    """Reciprocal vectors as rows, computed as ``inipbc.f90:61-71`` does.

    ``bvec = 2 pi inv(transpose(latt_vec))`` by ``mtrxin``; row ``i`` of the
    result is Fortran's column ``bvec(:,i)``.
    """

    columns = lattice_columns(lattice_vectors)
    transposed = [[columns[j][i] for j in range(3)] for i in range(3)]
    inverse = _mtrxin(transposed)
    twopi = 2.0 * math.pi
    bvec = [[twopi * inverse[i][j] for j in range(3)] for i in range(3)]
    return np.array([[bvec[row][column] for row in range(3)] for column in range(3)])


def candidate_rotations(count: int) -> list[list[list[float]]]:
    """``rot``: the 24 hexagonal (``count <= 24``) or 48 cubic rotations.

    Cartesian matrices, ``R[n][i][j]`` = Fortran ``r(n+1, i+1, j+1)``,
    generated with Fortran's products so that the hexagonal entries carry
    the same bits.
    """

    size = 24 if count <= 24 else 48
    r = [[[0.0] * 3 for _ in range(3)] for _ in range(size + 1)]

    def element(n, i, j):  # 1-based, as in the Fortran
        return r[n][i - 1][j - 1]

    def set_element(n, i, j, value):
        r[n][i - 1][j - 1] = value

    def accumulate(target, left, right, last):
        for i in range(1, last + 1):
            for j in range(1, last + 1):
                for k in range(1, last + 1):
                    set_element(
                        target, i, j,
                        element(target, i, j) + element(left, i, k) * element(right, k, j),
                    )

    if size == 24:
        f = ROOT3 * 0.5
        set_element(2, 1, 1, 0.5)
        set_element(2, 1, 2, -f)
        set_element(2, 2, 1, f)
        set_element(2, 2, 2, 0.5)
        set_element(7, 1, 1, -0.5)
        set_element(7, 1, 2, -f)
        set_element(7, 2, 1, -f)
        set_element(7, 2, 2, 0.5)
        for n in range(1, 7):
            set_element(n, 3, 3, 1.0)
            set_element(n + 18, 3, 3, 1.0)
            set_element(n + 6, 3, 3, -1.0)
            set_element(n + 12, 3, 3, -1.0)
        for i in range(1, 3):
            set_element(1, i, i, 1.0)
            for j in range(1, 3):
                set_element(6, i, j, element(2, j, i))
                for k in range(1, 3):
                    set_element(3, i, j, element(3, i, j) + element(2, i, k) * element(2, k, j))
                    set_element(8, i, j, element(8, i, j) + element(2, i, k) * element(7, k, j))
                    set_element(12, i, j, element(12, i, j) + element(7, i, k) * element(2, k, j))
        for i in range(1, 3):
            for j in range(1, 3):
                set_element(5, i, j, element(3, j, i))
                for k in range(1, 3):
                    set_element(4, i, j, element(4, i, j) + element(2, i, k) * element(3, k, j))
                    set_element(9, i, j, element(9, i, j) + element(2, i, k) * element(8, k, j))
                    set_element(10, i, j, element(10, i, j) + element(12, i, k) * element(3, k, j))
                    set_element(11, i, j, element(11, i, j) + element(12, i, k) * element(2, k, j))
        for n in range(1, 13):
            for i in range(1, 3):
                for j in range(1, 3):
                    set_element(n + 12, i, j, -element(n, i, j))
    else:
        set_element(9, 1, 3, 1.0)
        set_element(9, 2, 1, 1.0)
        set_element(9, 3, 2, 1.0)
        set_element(19, 1, 1, 1.0)
        set_element(19, 2, 3, -1.0)
        set_element(19, 3, 2, 1.0)
        for i in range(1, 4):
            set_element(1, i, i, 1.0)
            for j in range(1, 4):
                set_element(20, i, j, element(19, j, i))
                set_element(5, i, j, element(9, j, i))
                for k in range(1, 4):
                    set_element(2, i, j, element(2, i, j) + element(19, i, k) * element(19, k, j))
                    set_element(16, i, j, element(16, i, j) + element(9, i, k) * element(19, k, j))
                    set_element(23, i, j, element(23, i, j) + element(19, i, k) * element(9, k, j))
        for target, left, right in (
            (6, 2, 5), (7, 16, 23), (8, 5, 2), (10, 2, 9), (11, 9, 2),
            (12, 23, 16), (14, 16, 2), (15, 2, 16), (22, 23, 2), (24, 2, 23),
        ):
            accumulate(target, left, right, 3)
        for target, left, right in (
            (3, 5, 12), (4, 5, 10), (13, 23, 11), (17, 16, 12), (18, 16, 10), (21, 12, 15),
        ):
            accumulate(target, left, right, 3)
        for n in range(1, 25):
            for i in range(1, 4):
                for j in range(1, 4):
                    set_element(n + 24, i, j, -element(n, i, j))
    return [r[n] for n in range(1, size + 1)]


def _remove_lattice_vector(p, g) -> tuple[list[float], int]:
    """``rlv``: ``y = p g`` minus its nearest integer vector; ``l`` flags a removal."""

    y = _matvec(p, g)
    removed = sum(fortran_nint(abs(value)) for value in y)
    return [value - 1.0 * fortran_nint(value) for value in y], removed


def lattice_point_group(alatt, inverse) -> tuple[list, list[int], int, int]:
    """``pgl``: lattice operations among the candidate rotations.

    Returns ``(rotations, indices, ihg, ihc)``: the candidate set that was
    kept (hexagonal or cubic), the 1-based indices of the operations that map
    the lattice onto itself, the holohedral group number ``ihg`` (1..7, see
    :data:`CRYSTAL_SYSTEMS`) and ``ihc`` (0 hexagonal, 1 cubic candidates).
    """

    eps = SYMMETRY_TOLERANCE
    hexagonal_trial = True
    while True:
        rotations = candidate_rotations(24 if hexagonal_trial else 48)
        indices = []
        for number, rotation in enumerate(rotations, start=1):
            total = 0.0
            for column in range(3):
                xa = [
                    ((0.0 + rotation[i][0] * alatt[0][column]) + rotation[i][1] * alatt[1][column])
                    + rotation[i][2] * alatt[2][column]
                    for i in range(3)
                ]
                remainder, _ = _remove_lattice_vector(inverse, xa)
                for value in remainder:
                    total = total + abs(value)
            if total <= 10.0 * eps:
                indices.append(number)
        count = len(indices)
        if hexagonal_trial:
            if count == 12:
                return rotations, indices, 6, 0
            if count > 12:
                return rotations, indices, 7, 0
            hexagonal_trial = False
            continue
        if count == 16:
            return rotations, indices, 4, 1
        if count > 16:
            return rotations, indices, 5, 1
        if count == 4:
            return rotations, indices, 2, 1
        if count > 4:
            return rotations, indices, 3, 1
        return rotations, indices, 1, 1


def _fortran_mod(values: np.ndarray) -> np.ndarray:
    """Fortran ``mod(a, 1.0)`` (``fmod``: the sign of ``a``), elementwise."""

    return np.fmod(values, 1.0)


def _crystal_operations(inverse, positions, types, rotations, lattice_indices):
    """``atftmt``: the lattice operations that map the atoms onto themselves.

    ``positions`` is ``(3, atoms)`` Cartesian (Fortran ``x``), ``types`` the
    integer type of every atom.  Returns ``(indices, translations)``: the
    1-based candidate indices of the crystal operations, in the order tested,
    and for each the vector ``v`` (lattice coordinates) of the pair of atoms
    that matched first (Fortran ``v(:,l) = vr``; the fractional translation is
    ``tnp = -v``).
    """

    eps = SYMMETRY_TOLERANCE
    x = np.asarray(positions, dtype=np.float64)
    types = np.asarray(types)
    atoms = x.shape[1]
    same_type = types[:, None] == types[None, :]
    ai = [[float(value) for value in row] for row in inverse]
    found: list[int] = []
    translations: list[list[float]] = []
    for number in lattice_indices:
        rotation = rotations[number - 1]
        rx = np.empty_like(x)
        for i in range(3):
            rx[i] = ((0.0 + rotation[i][0] * x[0]) + rotation[i][1] * x[1]) + rotation[i][2] * x[2]
        # rdiff[k1, k2] = rlv(ai, R x_k1 - x_k2)  (Fortran rdiff(:, k2, k1)).
        xb = rx[:, :, None] - x[:, None, :]
        y = [
            ((0.0 + ai[i][0] * xb[0]) + ai[i][1] * xb[1]) + ai[i][2] * xb[2]
            for i in range(3)
        ]
        rdiff = np.stack(
            [component - 1.0 * _nint_array(component) for component in y], axis=-1
        )
        match_vector = None
        give_up = False
        for k1 in range(atoms):
            for k2 in range(atoms):
                if types[k1] != types[k2]:
                    continue
                vr = rdiff[k1, k2]
                dif = np.zeros((atoms, atoms))
                for i in range(3):
                    dif = dif + _fortran_mod(np.abs(vr[i] - rdiff[:, :, i]) + eps)
                matched = (dif <= 10.0 * eps) & same_type
                has_partner = np.any(matched, axis=1)
                if has_partner.all():
                    match_vector = [float(value) for value in vr]
                    break
                if not has_partner[0]:
                    # The first atom found no partner: ks == 0, the
                    # operation is given up (atftmt's shortcut).
                    give_up = True
                    break
            if match_vector is not None or give_up:
                break
        if match_vector is None:
            continue
        found.append(number)
        translations.append(match_vector)
    return found, translations


def _nint_array(values: np.ndarray) -> np.ndarray:
    """Elementwise Fortran ``nint`` as floats (halves away from zero)."""

    whole = np.trunc(values)
    fraction = values - whole
    return whole + (fraction >= 0.5) - (fraction <= -0.5)


def _labels(rotations) -> tuple[str, ...]:
    """``symm_ident``: E, C2, C3, C4, C6 and their products with the inversion."""

    labels = []
    for matrix in rotations:
        a = [[float(value) for value in row] for row in matrix]
        det = (
            a[0][0] * a[1][1] * a[2][2]
            + a[1][0] * a[2][1] * a[0][2]
            + a[0][1] * a[1][2] * a[2][0]
            - a[2][0] * a[1][1] * a[0][2]
            - a[1][0] * a[0][1] * a[2][2]
            - a[0][0] * a[2][1] * a[1][2]
        )
        trace = a[0][0] + a[1][1] + a[2][2]
        if fortran_nint(det) == 1:
            labels.append(_AXES[fortran_nint(trace - 1.0)].ljust(3))
        else:
            labels.append(("I" + _AXES[fortran_nint(-trace - 1.0)]).ljust(3)[:3])
    return tuple(labels)


@dataclass(frozen=True)
class SpaceGroup:
    """The symmetry operations of a periodic cell (Fortran ``symm``).

    Every field is a tuple so the object is hashable and compares by value
    (it may sit inside a frozen :class:`~parsec_python.models.SinglePointInput`).

    ``rotations`` are the integer matrices ``M`` (lattice coordinates,
    ``rmtrx``) and ``translations`` the fractional translations ``t``
    (``tnp``) of ``T(f) = M (f - t)``; ``cartesian_rotations`` are ``U``
    (``symm%trans``).  ``labels`` are ``symm_ident``'s names (three
    characters, e.g. ``"IC2"``).  ``crystal_system`` is the holohedry of the
    lattice, ``full_holohedry`` says whether the crystal has all of its
    operations; ``lattice_operations`` are the 1-based candidate numbers of
    the lattice operations (what PARSEC prints when the group is not the full
    holohedry), ``candidate_numbers`` those of the crystal operations.
    ``symmorphic`` is PARSEC's verdict, which looks at the translation of the
    *last* operation only (``atftmt``: ``vs`` is overwritten in its loop).
    ``ignored`` marks the identity group of ``Ignore_Symmetry``.
    """

    rotations: tuple[tuple[tuple[int, int, int], ...], ...]
    translations: tuple[tuple[float, float, float], ...]
    cartesian_rotations: tuple[tuple[tuple[float, float, float], ...], ...]
    labels: tuple[str, ...]
    crystal_system: str = "triclinic"
    full_holohedry: bool = False
    lattice_operations: tuple[int, ...] = (1,)
    candidate_numbers: tuple[int, ...] = (1,)
    symmorphic: bool = True
    ignored: bool = False

    @property
    def order(self) -> int:
        return len(self.rotations)

    @property
    def rotation_matrices(self) -> np.ndarray:
        """``(order, 3, 3)`` integer ``M`` (``rmtrx``)."""

        return np.asarray(self.rotations, dtype=np.int64).reshape(-1, 3, 3)

    @property
    def reciprocal_rotations(self) -> np.ndarray:
        """``(order, 3, 3)`` integer ``G = transpose(M)`` (``gmtrx``)."""

        return np.transpose(self.rotation_matrices, (0, 2, 1)).copy()

    @property
    def translation_vectors(self) -> np.ndarray:
        """``(order, 3)`` fractional translations ``t`` (``tnp``)."""

        return np.asarray(self.translations, dtype=np.float64).reshape(-1, 3)

    def commensurate_translations(self, directions: int = 3, *, tolerance: float = 1.0e-8):
        """The translations ``adjust_step`` makes the grid commensurate with.

        ``inipbc.f90:381-391``: for every operation the components of
        ``|M t|`` larger than ``tolerance``, per lattice direction, in the
        order of the operations.
        """

        lists: list[list[float]] = [[] for _ in range(directions)]
        for matrix, translation in zip(self.rotations, self.translations):
            mt = _matvec([[float(v) for v in row] for row in matrix], translation)
            for direction in range(directions):
                value = mt[direction]
                if value < 0.0:
                    value = -value
                if value > tolerance:
                    lists[direction].append(float(value))
        return tuple(tuple(values) for values in lists)


def identity_space_group(*, ignored: bool = True) -> SpaceGroup:
    """The group of the identity alone (``symmetries.f90`` with ``info = 2``)."""

    return SpaceGroup(
        rotations=(((1, 0, 0), (0, 1, 0), (0, 0, 1)),),
        translations=((0.0, 0.0, 0.0),),
        cartesian_rotations=(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),),
        labels=("E  ",),
        ignored=ignored,
    )


def skipped_space_group() -> SpaceGroup:
    """The identity of a run whose symmetry PARSEC does not search.

    ``symmetries.f90`` with ``info = 2`` without ``Ignore_Symmetry`` (a slab,
    molecular dynamics, Polarizability, manual movements: ``parsec.F90:470-486``):
    "skipping non-trivial symmetry operations".  Marked by an empty
    ``crystal_system`` (no lattice was examined).
    """

    return replace(identity_space_group(ignored=False), crystal_system="")


def find_space_group(lattice_vectors, atoms) -> SpaceGroup:
    """Space group of the atoms in a three-dimensional periodic cell.

    ``lattice_vectors`` are the rows ``a1, a2, a3`` in bohr and ``atoms`` the
    :class:`~parsec_python.models.Atom` objects at their input coordinates.
    Atom types are the species symbols in order of first appearance (PARSEC's
    ``Atom_Type`` blocks), atoms of one type in input order (``coorat``).
    Translation of ``symmetries.f90:120-170`` with ``nops = -1`` (fractional
    translations included) followed by ``symgen.f90:111-221``.
    """

    alatt = lattice_columns(lattice_vectors)
    # symmetries.f90: invlat = inv(transpose(alatt)); coorat = invlat^T r.
    invlat = _mtrxin([[alatt[j][i] for j in range(3)] for i in range(3)])
    order: list[str] = []
    grouped: dict[str, list] = {}
    for atom in atoms:
        if atom.symbol not in grouped:
            order.append(atom.symbol)
            grouped[atom.symbol] = []
        grouped[atom.symbol].append(atom)
    columns = []
    types = []
    for type_index, symbol in enumerate(order):
        for atom in grouped[symbol]:
            position = [float(value) for value in np.asarray(atom.position, dtype=float)]
            fractional = _matvec_transposed(invlat, position)
            # symgen: x = a * coorat, summed over the lattice vectors.
            cartesian = [
                ((0.0 + alatt[k][0] * fractional[0]) + alatt[k][1] * fractional[1])
                + alatt[k][2] * fractional[2]
                for k in range(3)
            ]
            columns.append(cartesian)
            types.append(type_index)
    positions = np.array(columns, dtype=np.float64).T
    inverse = _mtrxin(alatt)
    rotations, lattice_indices, holohedral, _hexagonal_flag = lattice_point_group(alatt, inverse)
    numbers, vectors = _crystal_operations(inverse, positions, types, rotations, lattice_indices)
    if not numbers or numbers[0] != 1:
        raise ValueError(
            "the symmetry search did not find the identity (overlapping atoms?); "
            "PARSEC stops with 'SYMMETRY GROUP DOES NOT HAVE IDENTITY'"
        )
    matrices = []
    cartesian = []
    translations = []
    for number, vector in zip(numbers, vectors):
        rotation = rotations[number - 1]
        # r1 = transpose(R) a ; rlat = b r1 ; rmtrx = nint(rlat).
        r1 = [
            [
                ((0.0 + rotation[0][j] * alatt[0][k]) + rotation[1][j] * alatt[1][k])
                + rotation[2][j] * alatt[2][k]
                for k in range(3)
            ]
            for j in range(3)
        ]
        rlat = [
            [
                ((0.0 + inverse[j][0] * r1[0][k]) + inverse[j][1] * r1[1][k])
                + inverse[j][2] * r1[2][k]
                for k in range(3)
            ]
            for j in range(3)
        ]
        matrices.append(tuple(tuple(fortran_nint(value) for value in row) for row in rlat))
        # symm%trans = transpose(trans), trans = R.
        cartesian.append(tuple(tuple(float(rotation[j][i]) for j in range(3)) for i in range(3)))
        translations.append(tuple(-value for value in vector))
    full = (holohedral == 7 and len(numbers) == 24) or (holohedral == 5 and len(numbers) == 48)
    last = vectors[-1]
    symmorphic = not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE)
    return SpaceGroup(
        rotations=tuple(matrices),
        translations=tuple(translations),
        cartesian_rotations=tuple(cartesian),
        labels=_labels(matrices),
        crystal_system=CRYSTAL_SYSTEMS[holohedral - 1],
        full_holohedry=full,
        lattice_operations=tuple(lattice_indices),
        candidate_numbers=tuple(numbers),
        symmorphic=symmorphic,
    )


ATOM_IMAGE_TOLERANCE = 1.0e-6
"""``tol`` of ``symm_forces.f90``: an atom and its image coincide within it
(a length in lattice coordinates)."""


def space_group_atom_images(
    group: SpaceGroup,
    lattice_vectors,
    atoms,
    *,
    tolerance: float = ATOM_IMAGE_TOLERANCE,
) -> np.ndarray:
    """``(order, atoms)`` index of the atom each operation maps every atom onto.

    ``symm_forces.f90:62-91``: operation ``T`` takes atom ``a`` to
    ``cro = U (R_a - A t)`` (``symop``) and its image is the first atom ``b``
    of the same type with ``inv(A) (cro - R_b)`` a lattice vector within
    ``tolerance`` (lattice coordinates, ``nint`` removed).  ``lattice_vectors``
    are the rows ``a1, a2, a3`` in bohr; positions may lie outside the cell.
    Raises ``ValueError`` (Fortran error 411) when an atom has no image, or
    when an operation does not permute the atoms (two atoms within
    ``tolerance`` of each other).
    """

    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    columns = vectors.T
    inverse = np.linalg.inv(columns)
    positions = np.array([np.asarray(atom.position, dtype=np.float64) for atom in atoms])
    symbols = [atom.symbol for atom in atoms]
    count = len(symbols)
    images = np.empty((group.order, count), dtype=np.int64)
    rotations = np.asarray(group.cartesian_rotations, dtype=np.float64).reshape(-1, 3, 3)
    for number, (rotation, translation) in enumerate(
        zip(rotations, group.translation_vectors)
    ):
        shift = columns @ translation
        for index in range(count):
            mapped = rotation @ (positions[index] - shift)
            found = -1
            for other in range(count):
                if symbols[other] != symbols[index]:
                    continue
                difference = inverse @ (mapped - positions[other])
                difference -= np.rint(difference)
                if np.sqrt(difference @ difference) <= tolerance:
                    found = other
                    break
            if found < 0:
                raise ValueError(
                    f"symmetry operation {number + 1} maps atom {index + 1} at "
                    f"{positions[index].tolist()} onto no atom (PARSEC error 411: "
                    "unable to find equivalent of atom)"
                )
            images[number, index] = found
        if np.unique(images[number]).size != count:
            raise ValueError(
                f"symmetry operation {number + 1} does not permute the atoms"
            )
    return images


@dataclass(frozen=True)
class PeriodicGridSymmetry(GridPointGroup):
    """Space-group operations as permutations of the points of a periodic grid.

    :func:`~parsec_python.Grid.point_group.symmetrize_scalar` averages a field
    over them (``symm_scalar`` of ``newrho.F90``).  ``reduced_order`` is one:
    the periodic grid of this port is never reduced to a wedge, as PARSEC's
    grid of a k-point run is not (its Abelian subgroup is the identity), so the
    printed deviation is relative to the sum over the whole cell.
    """

    @property
    def reduced_order(self) -> int:
        return 1


def grid_fractional_points(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(integer points, steps along each lattice direction, shift)`` of a grid.

    The fractional coordinate of point ``i`` along direction ``d`` is
    ``(i_d + shift_d) / n_d``, the ``rw`` of ``symm_scalar``, with the grid's
    own shift: Fortran's (0.5 unless ``Ignore_Symmetry``, or the ``Grid_Shift``
    block) on every grid of the Fortran-derived routines, the orthorhombic one
    included since PARSEC 1.5's parity round (it started at the origin before).
    """

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    shift = np.asarray(getattr(grid, "shift", (0.0, 0.0, 0.0)), dtype=np.float64)
    return points, counts, shift


def grid_commensurability_problem(group: SpaceGroup, counts, shift) -> str | None:
    """Why ``group`` does not map a periodic grid onto itself, or ``None``.

    ``counts`` are the points along the three lattice directions and
    ``shift`` the grid shift (units of the steps).  An operation maps the
    point ``f = (i + s)/n`` to ``M (f - t)``, which must again be of that form:
    with ``M`` mixing only directions of equal ``n`` and ``s``, this needs
    every ``(M t)_d n_d`` to be a whole number (``adjust_step``).
    """

    counts = np.asarray(counts, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    # Test on the eight corners of one grid cell: enough for an affine map.
    corners = np.array(
        [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.float64
    )
    fractional = (corners + shift) / counts
    for number, (matrix, translation) in enumerate(
        zip(group.rotation_matrices, group.translation_vectors), start=1
    ):
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = mapped * counts - shift
        if np.max(np.abs(target - np.rint(target))) > 1.0e-6:
            return (
                f"operation {number} ({group.labels[number - 1].strip()}, fractional "
                "translation "
                + " ".join(f"{value:.5f}" for value in translation)
                + ") does not map the grid "
                + "x".join(str(int(value)) for value in counts)
                + " onto itself"
            )
    return None


def periodic_grid_symmetry(grid, group: SpaceGroup) -> PeriodicGridSymmetry:
    """Map the operations of ``group`` onto the points of a periodic grid.

    ``symm_scalar`` (``newrho.F90:262-283``): the point of fractional
    coordinates ``rw = (i + s)/n`` goes to ``rrw = M (rw - t)`` and to the
    integer point ``nint(rrw n - s)``, folded into the cell.  Raises
    ``ValueError`` where an operation does not map the grid onto itself.
    """

    points, counts, shift = grid_fractional_points(grid)
    problem = grid_commensurability_problem(group, counts, shift)
    if problem is not None:
        raise ValueError(
            "Symmetrize_Charge_Density: " + problem + "; PARSEC's adjust_step makes "
            "the grid spacing commensurate with the fractional translations, which "
            "this cell's grid is not"
        )
    fractional = (points.astype(np.float64) + shift) / counts
    # One row of point indices per operation (int32: half the memory of the
    # table, operations x points, for any grid below 2**31 points).
    index_type = np.int32 if points.shape[0] < 2**31 - 1 else np.int64
    images = np.empty((group.order, points.shape[0]), dtype=index_type)
    for index, (matrix, translation) in enumerate(
        zip(group.rotation_matrices, group.translation_vectors)
    ):
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = _nint_array(mapped * counts - shift).astype(np.int64)
        rows = grid.rows_for_integer_coordinates(target % counts)
        if np.unique(rows).size != rows.size:
            raise ValueError(
                f"Symmetrize_Charge_Density: operation {index + 1} does not permute "
                "the grid points"
            )
        images[index] = rows
    return PeriodicGridSymmetry(
        matrices=group.rotation_matrices.astype(np.float64), images=images
    )


def confined_symmetry_lattice(lattice_vectors, periodicity: int, radius: float) -> np.ndarray:
    """``alatt`` of a wire or slab (``symmetries.f90:114-118``), as rows.

    ``symm%alatt = pbc%latt_vec`` with ``4*rmax`` on the diagonal of every
    confined direction (``latt_vec`` holds unit vectors there), as for a
    cluster's artificial cubic cell.
    """

    lattice = np.array(lattice_vectors, dtype=np.float64, copy=True)
    for axis in range(int(periodicity), 3):
        lattice[axis, axis] = 4.0 * float(radius)
    return lattice


def grid_preserving_operations(group: SpaceGroup, periodicity: int, *, tolerance: float = 1.0e-6):
    """The operations of a wire or slab group that map its grid onto itself.

    Returns ``(group, dropped)``.  Fortran's group of a wire or slab
    (:func:`find_wire_slab_space_group`) may hold operations that a
    confined grid does not have: a fractional translation along a confined
    direction (a mirror or inversion about a plane other than ``z = 0``, in
    units of ``4*rmax``) moves the cylinder or slab off itself, and a matrix
    that mixes a periodic with a confined direction (possible only when
    ``4*rmax`` equals a lattice length) maps the grid onto another one.
    ``symm_scalar`` then averages over points that are not images of each
    other (with ``bandstruc`` a slab turns Fortran's symmetry on: graphene
    with one atom lifted by 0.15 bohr ends 0.21 Ry off its own run).  Those
    operations are dropped here (a subgroup remains: block-diagonal matrices
    with no confined translation compose to the same form); every other
    field of ``group`` is kept, ``full_holohedry`` becomes false when an
    operation is dropped.
    """

    periodicity = int(periodicity)
    kept = []
    for index, (matrix, translation) in enumerate(zip(group.rotations, group.translations)):
        matrix = np.asarray(matrix, dtype=np.int64)
        mixes = np.any(matrix[:periodicity, periodicity:] != 0) or np.any(
            matrix[periodicity:, :periodicity] != 0
        )
        shifted = any(
            abs(float(translation[axis])) > tolerance for axis in range(periodicity, 3)
        )
        if not (mixes or shifted):
            kept.append(index)
    dropped = group.order - len(kept)
    if dropped == 0:
        return group, 0
    last = group.translations[kept[-1]]
    return (
        SpaceGroup(
            rotations=tuple(group.rotations[index] for index in kept),
            translations=tuple(group.translations[index] for index in kept),
            cartesian_rotations=tuple(group.cartesian_rotations[index] for index in kept),
            labels=tuple(group.labels[index] for index in kept),
            crystal_system=group.crystal_system,
            full_holohedry=False,
            lattice_operations=group.lattice_operations,
            candidate_numbers=tuple(group.candidate_numbers[index] for index in kept),
            symmorphic=not (abs(last[0]) + abs(last[1]) + abs(last[2]) > SYMMETRY_TOLERANCE),
            ignored=group.ignored,
        ),
        dropped,
    )


def find_wire_slab_space_group(lattice_vectors, periodicity: int, radius: float, atoms):
    """Space group of a wire or slab: ``symmetries.f90`` with ``pbc%per < 3``.

    ``lattice_vectors`` are the rows of ``latt_vec`` (unit vectors along the
    confined directions), ``radius`` is ``Boundary_Sphere_Radius``.  The
    search of :func:`find_space_group` runs on ``alatt``
    (:func:`confined_symmetry_lattice`); the operations that do not map the
    confined grid onto itself are dropped (:func:`grid_preserving_operations`).
    Returns ``(group, dropped)``.  ``M`` and ``t`` are relative to ``alatt``;
    along the periodic directions they are those of ``latt_vec``, so the
    reciprocal rotations act on Monkhorst-Pack fractions as in ``irrbz``.
    """

    alatt = confined_symmetry_lattice(lattice_vectors, periodicity, radius)
    group = find_space_group(alatt, atoms)
    return grid_preserving_operations(group, periodicity)


def confined_grid_symmetry(grid, group: SpaceGroup) -> PeriodicGridSymmetry:
    """The operations of ``group`` as permutations of a wire or slab grid.

    ``symm_scalar`` (``newrho.F90:262-283``) with ``is_pbc``: the point of
    fractional coordinates ``rw = (i + s)/n`` goes to ``rrw = M (rw - t)``
    and to the integer point ``nint(rrw n - s)``.  Fortran folds every
    direction into the index box; here only the periodic ones are folded,
    and a point mapped outside the cylinder or slab is an error (the
    operations of :func:`grid_preserving_operations` never do that).
    """

    points = np.asarray(grid.integer_coordinates, dtype=np.int64)
    counts = np.asarray(grid.points_per_axis, dtype=np.int64)
    shift = np.asarray(grid.shift, dtype=np.float64)
    problem = grid_commensurability_problem(group, counts, shift)
    if problem is not None:
        raise ValueError(
            "Symmetrize_Charge_Density: " + problem + "; PARSEC's adjust_step makes "
            "the grid spacing commensurate with the fractional translations, which "
            "this grid is not"
        )
    fractional = (points.astype(np.float64) + shift) / counts
    index_type = np.int32 if points.shape[0] < 2**31 - 1 else np.int64
    images = np.empty((group.order, points.shape[0]), dtype=index_type)
    for index, (matrix, translation) in enumerate(
        zip(group.rotation_matrices, group.translation_vectors)
    ):
        mapped = (fractional - translation) @ matrix.T.astype(np.float64)
        target = _nint_array(mapped * counts - shift).astype(np.int64)
        rows = grid.rows_for_integer_coordinates(target)
        if np.any(rows < 0) or np.unique(rows).size != rows.size:
            raise ValueError(
                f"Symmetrize_Charge_Density: operation {index + 1} does not permute "
                "the points of the wire or slab grid"
            )
        images[index] = rows
    return PeriodicGridSymmetry(
        matrices=group.rotation_matrices.astype(np.float64), images=images
    )


__all__ = [
    "ATOM_IMAGE_TOLERANCE",
    "CRYSTAL_SYSTEMS",
    "PeriodicGridSymmetry",
    "SpaceGroup",
    "candidate_rotations",
    "confined_grid_symmetry",
    "confined_symmetry_lattice",
    "find_space_group",
    "find_wire_slab_space_group",
    "skipped_space_group",
    "grid_preserving_operations",
    "fortran_anint",
    "fortran_nint",
    "fortran_reciprocal_lattice",
    "grid_commensurability_problem",
    "identity_space_group",
    "lattice_point_group",
    "periodic_grid_symmetry",
    "space_group_atom_images",
]
