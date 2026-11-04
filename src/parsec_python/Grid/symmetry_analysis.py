"""PARSEC's analysis of a symmetry group: multiplication tables, ``symchk``, Abelian subgroup.

Translation of the parts of ``symgen.f90`` that :mod:`parsec_python.Grid.space_group`
leaves out, which PARSEC uses for its report and for the irreducible wedge of
its real-space grid (this port keeps every field on the full grid, so only
the report uses them):

* ``atftmt`` l.471-520: the multiplication table of the crystal operations,
  ``mt(l, m) = n`` with ``R_l R_m = R_n`` (Cartesian candidate rotations,
  tolerance ``100 eps`` on the sum of absolute differences), printed at
  ``Output_Level >= 2`` (:func:`atftmt_multiplication_table`);
* ``symchk`` l.907-1053: duplicate operations, the table of products of the
  integer lattice matrices ``M_i M_j = M_k`` and PARSEC's test of the
  fractional translations, which marks a failing product ``-1``
  (:func:`symmetry_check`);
* ``abelian`` l.1132-1547: the Abelian subgroup PARSEC splits the
  eigenproblem by, its name and character table
  (:func:`abelian_subgroup`);
* ``symmetries.f90`` l.149-170 / ``parsec.F90`` l.464-492: which of these
  PARSEC runs for an input (``info``, ``norot``; :func:`symmetry_search_mode`).

:func:`report_space_group` finds the group of a cluster (the cube of edge
``4 rmax``, ``symmetries.f90`` l.114-126) or of a cell for the report, with
:func:`fast_crystal_operations`, an exact equivalent of ``atftmt``'s search
(:func:`parsec_python.Grid.space_group._crystal_operations`) that scales to
large systems: the same operations and the same translation vectors, bit for
bit.

Nothing here changes a number of a calculation.  Units: bohr; the vectors
of the operations are in lattice coordinates.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from .space_group import (
    CRYSTAL_SYSTEMS,
    SYMMETRY_TOLERANCE,
    SpaceGroup,
    _labels,
    _matvec_transposed,
    _mtrxin,
    _nint_array,
    fortran_nint,
    lattice_columns,
    lattice_point_group,
)

#: ``tol`` of ``abelian``: a translation longer than this excludes an operation.
ABELIAN_TOLERANCE = 1.0e-10


# ----------------------------------------------------------------------------
# Which part of the analysis PARSEC runs
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetrySearchMode:
    """``ierr`` handed to ``symmetries`` by ``parsec.F90`` l.464-492.

    ``info`` 0: full group and Abelian subgroup; 1: full group, no Abelian
    subgroup (``norot = -1``, no ``symchk``); 2: no search (identity only).
    ``suppressed`` is ``Ignore_Symmetry``; ``domain_warning`` the
    non-spherical cluster domain (its four lines are printed).
    """

    info: int
    suppressed: bool = False
    domain_warning: bool = False


def symmetry_search_mode(
    *,
    ignore_symmetry: bool,
    kpoints: bool,
    grid_shift,
    periodicity: int,
    spinors: bool,
    bands: bool,
    nscf: bool,
    molecular_dynamics: bool,
    polarizability: bool,
    manual_movement: bool,
    non_spherical_cluster: bool,
) -> SymmetrySearchMode:
    """``parsec.F90`` l.468-488: the ``ierr`` of the symmetry search.

    The assignments in PARSEC's order (later ones win): k-points, a grid
    shift other than 0.5, spin-orbit or non-collinear spinors, a band
    structure and a non-self-consistent run give 1; a slab, molecular
    dynamics, polarizability, manual movements and a non-spherical cluster
    domain give 2.  ``Ignore_Symmetry`` gives 2 before all of them.
    """

    if ignore_symmetry:
        return SymmetrySearchMode(info=2, suppressed=True)
    info = 0
    if kpoints:
        info = 1
    if not all(float(value) == 0.5 for value in grid_shift):
        info = 1
    if periodicity == 2:
        info = 2
    if spinors:
        info = 1
    if bands:
        info = 1
    if nscf:
        info = 1
    if molecular_dynamics:
        info = 2
    if polarizability:
        info = 2
    if manual_movement:
        info = 2
    warning = False
    if non_spherical_cluster:
        warning = True
        info = 2
    return SymmetrySearchMode(info=info, domain_warning=warning)


def abelian_rotation_flag(lattice_vectors, info: int) -> int:
    """``norot`` of ``symmetries.f90`` l.158-169.

    0 for a lattice along the Cartesian axes, 1 when ``alatt`` has an
    off-diagonal element above 1e-8 (the Abelian group then keeps only E
    and the inversion), -1 with ``info = 1`` (no Abelian subgroup).
    """

    if info == 1:
        return -1
    alatt = np.asarray(lattice_columns(lattice_vectors), dtype=float)
    off = alatt - np.diag(np.diag(alatt))
    return 1 if float(np.max(np.abs(off))) > 1.0e-8 else 0


# ----------------------------------------------------------------------------
# The crystal operations, fast
# ----------------------------------------------------------------------------


def fast_crystal_operations(inverse, positions, types, rotations, lattice_indices):
    """``atftmt``'s search with a neighbour tree: same operations, same vectors.

    Arguments and result as
    :func:`parsec_python.Grid.space_group._crystal_operations`.  For an
    operation ``R`` the Fortran loop takes the first atom pair ``(k1, k2)``
    whose vector ``v = rlv(R x_k1 - x_k2)`` maps every atom onto an atom of
    its type (within ``sum_i mod(|dv_i| + eps, 1) <= 10 eps``).  With
    ``k1 = 1`` some ``k2`` succeeds whenever the operation is a symmetry
    (``k2`` the image of the first atom), and a pair with ``k1 > 1`` maps the
    first atom as well, so the operation and its vector are those of the
    first successful ``k2`` of the first atom.  The test of a vector looks
    up every image in a periodic k-d tree of the fractional coordinates
    (L1 metric) and then applies Fortran's criterion to the hits.
    """

    from scipy.spatial import cKDTree

    eps = SYMMETRY_TOLERANCE
    x = np.asarray(positions, dtype=np.float64)
    types = np.asarray(types)
    atoms = x.shape[1]
    ai = [[float(value) for value in row] for row in inverse]

    def fractional(vectors):
        return np.stack(
            [((0.0 + ai[i][0] * vectors[0]) + ai[i][1] * vectors[1]) + ai[i][2] * vectors[2] for i in range(3)],
            axis=-1,
        )

    fx = fractional(x)
    folded = fx - np.floor(fx)
    folded[folded >= 1.0] = 0.0
    trees = {}
    members = {}
    for kind in np.unique(types):
        index = np.flatnonzero(types == kind)
        members[kind] = index
        trees[kind] = cKDTree(folded[index], boxsize=1.0)
    first_type = types[0]
    candidates_k2 = np.flatnonzero(types == first_type)
    # Look-up radius in L1: Fortran accepts sum |dv| <= 7 eps (plus wrap).
    radius = 1.0e-6

    found: list[int] = []
    translations: list[list[float]] = []
    for number in lattice_indices:
        rotation = rotations[number - 1]
        rx = np.empty_like(x)
        for i in range(3):
            rx[i] = ((0.0 + rotation[i][0] * x[0]) + rotation[i][1] * x[1]) + rotation[i][2] * x[2]
        frx = fractional(rx)
        match_vector = None
        # rdiff(:, k2, 1) = rlv(ai, R x_1 - x_k2) of every candidate k2, with
        # Fortran's operations in its order (elementwise, so the same bits).
        xb = rx[:, :1] - x[:, candidates_k2]
        y = [((0.0 + ai[i][0] * xb[0]) + ai[i][1] * xb[1]) + ai[i][2] * xb[2] for i in range(3)]
        vectors = np.stack([value - 1.0 * _nint_array(value) for value in y], axis=-1)
        # The first test of _maps_all_atoms (the image of the last atom) for
        # all candidates in one tree query: a large cluster has thousands of
        # candidates per operation that is not a symmetry, and one query per
        # candidate dominated the search.  The survivors are tested in full,
        # in Fortran's order, so the first match is the same.
        last = x.shape[1] - 1
        targets = frx[last][None, :] - vectors
        targets = targets - np.floor(targets)
        targets[targets >= 1.0] = 0.0
        probe, _ = trees[types[last]].query(targets, k=1, p=1, distance_upper_bound=radius)
        for candidate in np.flatnonzero(np.isfinite(probe)):
            vr = vectors[candidate]
            if _maps_all_atoms(vr, frx, fx, types, trees, members, radius, eps, rx, x, ai):
                match_vector = [float(value) for value in vr]
                break
        if match_vector is None:
            continue
        found.append(number)
        translations.append(match_vector)
    return found, translations


def _maps_all_atoms(vr, frx, fx, types, trees, members, radius, eps, rx, x, ai) -> bool:
    """Whether every atom ``k3`` has a partner ``k4`` with ``rdiff(k3, k4) ~ vr``.

    The nearest stored atom (periodic L1 distance of the fractional
    coordinates) of every image is the only candidate that can pass
    Fortran's criterion, which is then applied to it.  The last atom is
    tested first: a wrong vector almost always fails there.
    """

    targets = frx - vr[None, :]
    folded = targets - np.floor(targets)
    folded[folded >= 1.0] = 0.0
    last = folded.shape[0] - 1
    probe_kind = types[last]
    distance, _ = trees[probe_kind].query(folded[last], k=1, p=1, distance_upper_bound=radius)
    if not np.isfinite(distance):
        return False
    count = folded.shape[0]
    if count > _SAMPLE_ATOMS:
        # A spread sample first: a lattice translation of a large crystallite
        # maps the last atom and most inner atoms onto atoms, and the full
        # test of each such candidate would cost a query of every atom.  The
        # sample's test is part of the full one, so the answer is the same.
        sample = np.unique(np.linspace(0, count - 1, _SAMPLE_ATOMS).astype(np.int64))
        if not _atoms_map(sample, vr, folded, types, trees, members, radius, eps, rx, x, ai):
            return False
    return _atoms_map(np.arange(count), vr, folded, types, trees, members, radius, eps, rx, x, ai)


# Atoms of the sample test of _maps_all_atoms.
_SAMPLE_ATOMS = 64


def _atoms_map(selection, vr, folded, types, trees, members, radius, eps, rx, x, ai) -> bool:
    """Whether the atoms ``selection`` have partners ``k4`` with ``rdiff(k3, k4) ~ vr``.

    ``folded`` holds the folded fractional images of every atom.
    """

    for kind, tree in trees.items():
        own = selection[types[selection] == kind]
        if own.size == 0:
            continue
        distances, local = tree.query(folded[own], k=1, p=1, distance_upper_bound=radius)
        if not np.all(np.isfinite(distances)):
            return False
        partners = members[kind][local]
        # rdiff(:, k4, k3) and Fortran's comparison with vr, elementwise.
        xb = rx[:, own] - x[:, partners]
        y = [((0.0 + ai[i][0] * xb[0]) + ai[i][1] * xb[1]) + ai[i][2] * xb[2] for i in range(3)]
        dif = np.zeros(own.size)
        for i in range(3):
            vt = y[i] - 1.0 * _nint_array(y[i])
            da = np.abs(vr[i] - vt) + eps
            dif = dif + np.fmod(da, 1.0)
        if not np.all(dif <= 10.0 * eps):
            return False
    return True


def report_space_group(lattice_vectors, atoms) -> SpaceGroup:
    """The group ``symgen`` finds for ``atoms`` in the cell ``lattice_vectors``.

    As :func:`parsec_python.Grid.space_group.find_space_group`, with
    :func:`fast_crystal_operations` for the search; for a cluster pass the
    cube of edge ``4 rmax`` (``symmetries.f90`` l.123-126).
    """

    alatt = lattice_columns(lattice_vectors)
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
            cartesian = [
                ((0.0 + alatt[k][0] * fractional[0]) + alatt[k][1] * fractional[1])
                + alatt[k][2] * fractional[2]
                for k in range(3)
            ]
            columns.append(cartesian)
            types.append(type_index)
    positions = np.array(columns, dtype=np.float64).T
    inverse = _mtrxin(alatt)
    rotations, lattice_indices, holohedral, _ = lattice_point_group(alatt, inverse)
    numbers, vectors = fast_crystal_operations(
        inverse, positions, types, rotations, lattice_indices
    )
    if not numbers or numbers[0] != 1:
        raise ValueError("the symmetry search did not find the identity (overlapping atoms?)")
    matrices = []
    cartesian = []
    translations = []
    for number, vector in zip(numbers, vectors):
        rotation = rotations[number - 1]
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


# ----------------------------------------------------------------------------
# Multiplication tables and symchk
# ----------------------------------------------------------------------------


def atftmt_multiplication_table(group: SpaceGroup) -> np.ndarray:
    """``mt(ib(n1), ib(n2))`` of ``atftmt`` l.475-500, as candidate numbers.

    Row ``a`` and column ``b`` index the crystal operations in their order;
    the entry is the candidate number ``n`` of ``R_a R_b`` (``0`` where no
    operation matches: Fortran leaves ``mt`` unset there).
    """

    eps = SYMMETRY_TOLERANCE
    # The candidate rotation R is the transpose of cartesian_rotations.
    rotations = [np.asarray(matrix, dtype=float).T for matrix in group.cartesian_rotations]
    numbers = list(group.candidate_numbers)
    table = np.zeros((len(numbers), len(numbers)), dtype=np.int64)
    for a, ra in enumerate(rotations):
        for b, rb in enumerate(rotations):
            product = [
                [((0.0 + ra[i][0] * rb[0][j]) + ra[i][1] * rb[1][j]) + ra[i][2] * rb[2][j] for j in range(3)]
                for i in range(3)
            ]
            for c, rc in enumerate(rotations):
                ts = 0.0
                for i in range(3):
                    for j in range(3):
                        ts = ts + abs(product[i][j] - rc[i][j])
                if ts > 1.0e2 * eps:
                    continue
                table[a, b] = numbers[c]
                break
    return table


@dataclass(frozen=True)
class SymmetryCheck:
    """``symchk``: the multiplication table and its errors.

    ``table[i, j]`` is the 1-based index ``k`` of ``M_i M_j = M_k``, ``0``
    when no operation matches and ``-1`` when the fractional translations
    fail PARSEC's test.  ``failures`` are the ``(i, j, k, l, error)`` that
    ``symchk`` writes, ``duplicate`` the pair of equal operations (then no
    table is built), ``error`` its ``ierr`` (the operation with the most
    failures, ``0`` for none).
    """

    table: np.ndarray
    failures: tuple
    duplicate: tuple[int, int] | None
    error: int


def symmetry_check(group: SpaceGroup) -> SymmetryCheck:
    """Translation of ``symchk`` (``symgen.f90`` l.907-1053)."""

    mtrx = [np.asarray(matrix, dtype=np.int64) for matrix in group.rotations]
    tnp = [[float(value) for value in vector] for vector in group.translations]
    ntrans = len(mtrx)
    table = np.zeros((ntrans, ntrans), dtype=np.int64)
    if ntrans == 1:
        table[0, 0] = 1
        return SymmetryCheck(table=table, failures=(), duplicate=None, error=0)
    for i in range(1, ntrans):
        for j in range(i):
            same = all(abs(tnp[i][k] - tnp[j][k]) <= 1.0e-8 for k in range(3)) and bool(
                np.array_equal(mtrx[i], mtrx[j])
            )
            if same:
                return SymmetryCheck(table=table, failures=(), duplicate=(j + 1, i + 1), error=i + 1)
    for i in range(ntrans):
        for j in range(ntrans):
            product = mtrx[i] @ mtrx[j]
            for k in range(ntrans):
                if np.array_equal(product, mtrx[k]):
                    table[i, j] = k + 1
    failures = []
    for i in range(ntrans):
        for j in range(ntrans):
            k = int(table[i, j])
            if k == 0:
                continue
            for column in range(3):
                ttest = tnp[k - 1][column] - tnp[j][column]
                for m in range(3):
                    ttest = ttest - float(mtrx[j][m][column]) * tnp[i][m]
                itest = fortran_nint(ttest)
                if abs(ttest - float(itest)) < 1.0e-4:
                    continue
                failures.append((i + 1, j + 1, k, column + 1, abs(ttest - float(itest))))
                table[i, j] = -1
                break
    errors = np.zeros(ntrans, dtype=np.int64)
    for i in range(ntrans):
        for j in range(ntrans):
            if table[i, j] > 0:
                continue
            errors[i] += 1
            errors[j] += 1
    error = 0
    largest = 0
    for i in range(ntrans):
        if errors[i] <= largest:
            continue
        largest = int(errors[i])
        error = i + 1
    return SymmetryCheck(table=table, failures=tuple(failures), duplicate=None, error=error)


# ----------------------------------------------------------------------------
# The Abelian subgroup
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AbelianSubgroup:
    """``abelian``'s result: ``name`` (``'C_2v '`` etc., five characters),
    ``operations`` (1-based indices of the crystal operations, ``indx``),
    ``characters`` (``chi``, one row per representation) and ``labels``
    (three characters each)."""

    name: str
    operations: tuple[int, ...]
    characters: tuple[tuple[complex, ...], ...]
    labels: tuple[str, ...]

    @property
    def order(self) -> int:
        return len(self.operations)

    def character_exponents(self) -> tuple[tuple[int, ...], ...]:
        """``mod(nint(3 aimag(log(chi))/pi), 6)`` of the printed table (Fortran ``mod``)."""

        rows = []
        for row in self.characters:
            values = []
            for value in row:
                exponent = fortran_nint(3.0 * cmath.log(value).imag / math.pi)
                values.append(int(math.fmod(exponent, 6)))
            rows.append(tuple(values))
        return tuple(rows)


def abelian_subgroup(group: SpaceGroup, table: np.ndarray, norot: int) -> AbelianSubgroup:
    """Translation of ``abelian`` (``symgen.f90`` l.1132-1547).

    ``table`` is the ``symchk`` table (1-based indices); ``norot > 0``
    keeps only E and the inversion.  The operations considered have no
    fractional translation and a diagonal lattice matrix.  For ``C_3`` the
    Fortran fills the third row of the character table from itself (an
    unset array); the intended ``conjg`` of the second row is used.  The
    names ``C_3``, ``C_3h`` and ``C_6h`` need a three- or six-fold axis
    with a diagonal lattice matrix, which no lattice has.
    """

    ntrans = len(group.rotations)
    ids = [label.strip() for label in group.labels]
    tnp = [[float(value) for value in vector] for vector in group.translations]
    mult = np.asarray(table, dtype=np.int64)

    def m(a: int, b: int) -> int:
        return int(mult[a - 1, b - 1])

    pi = math.pi
    ph1 = cmath.exp(2.0 * pi * 1j)
    ph3 = cmath.exp(2.0 * pi * 1j / 3.0)
    phm3 = cmath.exp(-2.0 * pi * 1j / 3.0)
    ph6 = cmath.exp(pi * 1j / 3.0)
    phm6 = cmath.exp(-pi * 1j / 3.0)

    allowed = []
    for index in range(1, ntrans + 1):
        vector = tnp[index - 1]
        length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
        if length > ABELIAN_TOLERANCE:
            continue
        matrix = np.asarray(group.rotations[index - 1], dtype=np.int64)
        if np.any(matrix - np.diag(np.diag(matrix))):
            continue
        if norot > 0 and ids[index - 1] not in ("E", "IE"):
            continue
        allowed.append(index)

    nab = 0
    l_i = l_c2 = l_c22 = l_s2 = l_c3 = l_s3 = l_c6 = False
    i_e = i_i = i_c2 = i_c22 = i_s2 = i_c3 = i_s3 = i_c6 = 0
    for index in allowed:
        label = ids[index - 1]
        if label == "E":
            nab += 1
            i_e = index
        if label == "IE":
            l_i = True
            nab += 1
            i_i = index
        if not l_c2 and label == "C2":
            l_c2 = True
            nab += 1
            i_c2 = index
        if not l_c3 and label == "C3":
            l_c3 = True
            nab += 1
            i_c3 = index
        if not l_c6 and label == "C6":
            l_c6 = True
            nab += 1
            i_c6 = index
        if not l_s2 and label == "IC2":
            l_s2 = True
            nab += 1
            i_s2 = index
    # A second C2 that commutes with the first.
    done = False
    for ii in allowed:
        if ids[ii - 1] != "C2":
            continue
        for jj in allowed:
            if ids[jj - 1] != "C2":
                continue
            if m(ii, jj) == m(jj, ii) and ii != jj:
                l_c22 = True
                i_c2 = ii
                i_c22 = jj
                nab += 1
                done = True
                break
        if done:
            break
    # A mirror that commutes with a C2.
    l_s2 = any(ids[index - 1] == "IC2" for index in allowed)
    if l_s2 and l_c2:
        l_s2 = False
        nab -= 1
    done = False
    for ii in allowed:
        if ids[ii - 1] != "C2":
            continue
        for jj in allowed:
            vector = tnp[jj - 1]
            length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
            if length > ABELIAN_TOLERANCE:
                continue
            if ids[jj - 1] != "IC2":
                continue
            if m(ii, jj) == m(jj, ii):
                l_s2 = True
                i_c2 = ii
                i_s2 = jj
                nab += 1
                done = True
                break
        if done:
            break
    # A mirror that commutes with a C3.
    l_s3 = any(ids[index - 1] == "IC2" for index in allowed)
    if l_s3 and l_c3:
        l_s3 = False
        nab -= 1
    done = False
    for ii in allowed:
        if ids[ii - 1] != "C3":
            continue
        for jj in allowed:
            vector = tnp[jj - 1]
            length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
            if length > ABELIAN_TOLERANCE:
                continue
            if ids[jj - 1] != "IC2":
                continue
            if m(ii, jj) == m(jj, ii):
                l_s3 = True
                i_c3 = ii
                i_s3 = jj
                nab += 1
                done = True
                break
        if done:
            break

    if l_i:
        if l_c6:
            name = "C_6h"
        elif l_c2:
            name = "D_2h" if l_c22 else "C_2h"
        else:
            name = "C_i"
    else:
        if l_c3:
            name = "C_3h" if l_s3 else "C_3"
        elif l_c2:
            if l_c22:
                name = "D_2"
            elif l_s2:
                name = "C_2v"
            else:
                name = "C_2"
        else:
            name = "C_s" if l_s2 else "C_1"

    conj = lambda row: tuple(value.conjugate() for value in row)  # noqa: E731
    if name == "C_6h":
        indx = [i_e, i_c6, m(i_c6, i_c6)]
        indx.append(m(i_c6, indx[2]))
        indx.append(m(i_c6, indx[3]))
        indx.append(m(i_c6, indx[4]))
        indx.append(i_i)
        indx += [m(indx[k], i_i) for k in range(1, 6)]
        chi = [None] * 12
        chi[0] = (ph1,) * 12
        chi[1] = (ph1, -ph1, ph1, -ph1, ph1, -ph1, ph1, -ph1, ph1, -ph1, ph1, -ph1)
        chi[2] = (ph1, ph6, -phm6, -ph1, -ph6, phm6, ph1, ph6, -phm6, -ph1, -ph6, phm6)
        chi[3] = conj(chi[2])
        chi[4] = (ph1, -phm6, -ph6, ph1, -phm6, -ph6, ph1, -phm6, -ph6, ph1, -phm6, -ph6)
        chi[5] = conj(chi[4])
        chi[6] = (ph1,) * 6 + (-ph1,) * 6
        chi[7] = (ph1, -ph1, ph1, -ph1, ph1, -ph1, -ph1, ph1, -ph1, ph1, -ph1, ph1)
        chi[8] = (ph1, ph6, -phm6, -ph1, -ph6, phm6, -ph1, -ph6, phm6, ph1, ph6, -phm6)
        chi[9] = conj(chi[8])
        chi[10] = (ph1, -phm6, -ph6, ph1, -phm6, -ph6, -ph1, phm6, ph6, -ph1, phm6, ph6)
        chi[11] = conj(chi[9])
        labels = ["Au ", "Bg ", "E1g", "E1g", "E2g", "E2g", "Au ", "Bu ", "E1u", "E1u", "E2u", "E2u"]
    elif name == "C_3h":
        indx = [i_e, i_c3, m(i_c3, i_c3), i_s3]
        indx += [m(indx[1], i_s3), m(indx[2], i_s3)]
        chi = [
            (ph1,) * 6,
            (ph1, ph3, phm3, ph1, ph3, phm3),
        ]
        chi.append(conj(chi[1]))
        chi.append((ph1, ph1, ph1, -ph1, -ph1, -ph1))
        chi.append((ph1, ph3, phm3, -ph1, -ph3, -phm3))
        chi.append(conj(chi[4]))
        labels = ["Ap ", "Ep ", "Ep ", "App", "Epp", "Epp"]
    elif name == "C_3":
        indx = [i_e, i_c3, m(i_c3, i_c3)]
        chi = [(ph1, ph1, ph1), (ph1, ph3, phm3)]
        chi.append(conj(chi[1]))
        labels = ["A  ", "E  ", "E  "]
    elif name == "D_2h":
        indx = [i_e, i_c2, i_c22, m(i_c22, i_c2), i_i]
        indx += [m(indx[1], indx[4]), m(indx[2], indx[4]), m(indx[3], indx[4])]
        chi = [
            (ph1, ph1, ph1, ph1, ph1, ph1, ph1, ph1),
            (ph1, ph1, -ph1, -ph1, ph1, ph1, -ph1, -ph1),
            (ph1, -ph1, ph1, -ph1, ph1, -ph1, ph1, -ph1),
            (ph1, -ph1, -ph1, ph1, ph1, -ph1, -ph1, ph1),
            (ph1, ph1, ph1, ph1, -ph1, -ph1, -ph1, -ph1),
            (ph1, ph1, -ph1, -ph1, -ph1, -ph1, ph1, ph1),
            (ph1, -ph1, ph1, -ph1, -ph1, ph1, -ph1, ph1),
            (ph1, -ph1, -ph1, ph1, -ph1, ph1, ph1, -ph1),
        ]
        labels = ["Ag ", "B1g", "B2g", "B3g", "Au ", "B1u", "B2u", "B3u"]
    elif name == "C_2h":
        indx = [i_e, i_c2, i_i, m(i_c2, i_i)]
        chi = [
            (ph1, ph1, ph1, ph1),
            (ph1, -ph1, ph1, -ph1),
            (ph1, ph1, -ph1, -ph1),
            (ph1, -ph1, -ph1, ph1),
        ]
        labels = ["Ag ", "Bg ", "Au ", "Bu "]
    elif name == "D_2":
        indx = [i_e, i_c2, i_c22, m(i_c22, i_c2)]
        chi = [
            (ph1, ph1, ph1, ph1),
            (ph1, ph1, -ph1, -ph1),
            (ph1, -ph1, ph1, -ph1),
            (ph1, -ph1, -ph1, ph1),
        ]
        labels = ["A  ", "B1 ", "B2 ", "B3 "]
    elif name == "C_2v":
        indx = [i_e, i_c2, i_s2, m(i_c2, i_s2)]
        chi = [
            (ph1, ph1, ph1, ph1),
            (ph1, ph1, -ph1, -ph1),
            (ph1, -ph1, ph1, -ph1),
            (ph1, -ph1, -ph1, ph1),
        ]
        labels = ["A1 ", "A2 ", "B1 ", "B2 "]
    elif name == "C_2":
        indx = [i_e, i_c2]
        chi = [(ph1, ph1), (ph1, -ph1)]
        labels = ["A  ", "B  "]
    elif name == "C_s":
        indx = [i_e, i_s2]
        chi = [(ph1, ph1), (ph1, -ph1)]
        labels = ["A1 ", "A2 "]
    elif name == "C_i":
        indx = [i_e, i_i]
        chi = [(ph1, ph1), (ph1, -ph1)]
        labels = ["Ag ", "Au "]
    else:
        indx = [i_e]
        chi = [(ph1,)]
        labels = ["A  "]
    return AbelianSubgroup(
        name=name.ljust(5),
        operations=tuple(int(value) for value in indx),
        characters=tuple(tuple(complex(value) for value in row) for row in chi),
        labels=tuple(labels),
    )


__all__ = [
    "ABELIAN_TOLERANCE",
    "AbelianSubgroup",
    "SymmetryCheck",
    "SymmetrySearchMode",
    "abelian_rotation_flag",
    "abelian_subgroup",
    "atftmt_multiplication_table",
    "fast_crystal_operations",
    "report_space_group",
    "symmetry_check",
    "symmetry_search_mode",
]
