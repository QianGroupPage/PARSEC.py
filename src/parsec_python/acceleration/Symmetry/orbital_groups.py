"""Which part of a point group the orbitals use: one-dimensional sectors of a subgroup, with weights.

The orbitals of a symmetric structure are solved in sectors: the functions that transform under a group ``H``
by a real one-dimensional character ``lambda`` (``psi(h r) = lambda(h) psi(r)``), each sector a basis on the
orbits of ``H`` (:mod:`.representations`).  For an abelian ``H`` of order two per element (the reflection groups
of PARSEC, D2h and its subgroups) these characters are all of its irreducible representations and every sector
state is one state.  The full point group ``G`` of a structure is often larger and not abelian: a nanodiamond
cut around a carbon atom has Td (24 operations), whose irreducible representations E, T1, T2 are two- and
three-dimensional.  Its subgroup D2 gives four sectors, three of which (B1, B2, B3) hold the same levels: the
three partners of every T level, exchanged by the threefold rotations.

This module chooses a subgroup ``H`` of ``G`` and the sectors to solve so that every level of ``G`` is found
and counted once with its degeneracy, at the least cost:

* the sectors are the real one-dimensional characters ``lambda`` of ``H`` (a non-abelian ``H`` has fewer of them
  than elements; its higher-dimensional representations are not solved);
* a level of the ``G``-representation ``Gamma`` (dimension ``d``) appears in sector ``lambda`` as often as
  ``lambda`` occurs in the restriction of ``Gamma`` to ``H`` (``m(lambda, Gamma)``), so ``n1(Gamma) =
  sum_lambda m(lambda, Gamma)`` times in all.  Every ``Gamma`` must occur (``n1 >= 1``), and a sector ``lambda``
  counts each of its states ``w_lambda = d / n1(Gamma)`` times, which must be the same for every ``Gamma`` that
  it holds (otherwise the subgroup is not admitted: a state's weight would depend on its unknown ``Gamma``);
* sectors that an element of the normalizer of ``H`` in ``G`` maps onto each other hold the same levels: one of
  them is solved, with the weight of all;
* the cost of a sector is its rows times its states, about ``N / |H|`` times ``M sum_Gamma m(lambda, Gamma)
  d / |G|`` (``N`` grid points, ``M`` states); the admitted subgroup of the least total cost is chosen.

For Td this is D2d (the S4 axis along a coordinate axis): sectors A1, B1, A2, B2 of about ``N/8`` rows and
``M/8`` states each, with weights 1, 1, 3, 3 (A2 and B2 hold one partner of every T1 and T2 level; the two
others lie in the E of D2d, which is the threefold image of these), a quarter of the basis of D2 (four sectors
of ``N/4`` by ``M/4``).  For Oh it is D4h (eight sectors of ``N/16`` by ``3M/48``, weights 1, 3, 1, 3, ...).
For an abelian reflection group it is the group itself, with weight 1 everywhere: the former sectors.

Everything here acts on the operations as 3 x 3 integer matrices (signed permutations), by their
multiplication table; the characters of ``G`` come from its class multiplication coefficients (Burnside's
method), and a pair of complex conjugate representations, whose levels real orbitals join, counts as one real
representation of twice the dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _key(matrix: np.ndarray) -> bytes:
    return np.asarray(np.rint(matrix), dtype=np.int64).tobytes()


def _reduced(matrix: np.ndarray, modulus) -> np.ndarray:
    """An affine ``[[R, t], [0, 1]]`` with ``t`` taken modulo ``modulus`` (the translations of a periodic grid)."""

    if modulus is None:
        return matrix
    reduced = matrix.copy()
    reduced[:3, 3] %= np.asarray(modulus, dtype=np.int64)
    return reduced


def multiplication_table(operations, modulus=None) -> np.ndarray:
    """``table[i, j]`` = index of ``operations[i] @ operations[j]``; raises unless the set is a group.

    ``modulus`` (points per axis): the operations are affine ``4 x 4`` matrices whose translations (grid steps)
    count modulo it, those of a periodic cell.
    """

    matrices = [_reduced(np.asarray(np.rint(op), dtype=np.int64), modulus) for op in operations]
    index = {_key(m): i for i, m in enumerate(matrices)}
    if len(index) != len(matrices):
        raise ValueError("the operations are not distinct")
    order = len(matrices)
    table = np.empty((order, order), dtype=np.int64)
    for i, a in enumerate(matrices):
        for j, b in enumerate(matrices):
            k = index.get(_key(_reduced(a @ b, modulus)))
            if k is None:
                raise ValueError("the operations are not closed under multiplication")
            table[i, j] = k
    return table


def _identity_and_inverses(table: np.ndarray) -> tuple[int, np.ndarray]:
    order = table.shape[0]
    identity = int(np.flatnonzero(np.all(table == np.arange(order)[None, :], axis=1))[0])
    # the first (the only) j with g j = e in every row
    inverse = np.argmax(table == identity, axis=1).astype(np.int64)
    return identity, inverse


def conjugacy_classes(table: np.ndarray) -> list[tuple[int, ...]]:
    """The conjugacy classes (element indices), the class of the identity first."""

    identity, inverse = _identity_and_inverses(table)
    order = table.shape[0]
    seen = np.zeros(order, dtype=bool)
    classes = []
    for g in [identity] + [i for i in range(order) if i != identity]:
        if seen[g]:
            continue
        members = sorted({int(table[table[h, g], inverse[h]]) for h in range(order)})
        seen[members] = True
        classes.append(tuple(members))
    return classes


@dataclass(frozen=True)
class RealRepresentation:
    """A real irreducible representation of a group: its dimension and character per element."""

    dimension: int
    character: np.ndarray  # float, one value per element of the group
    # a pair of complex conjugate representations joined (a level of each and its conjugate are one real level)
    complex_pair: bool = False

    @property
    def levels(self) -> float:
        """Levels per ``M / |G|`` states: ``d`` partners each, or ``d/2`` real levels of a joined pair."""

        return self.dimension / 2.0 if self.complex_pair else float(self.dimension)


def real_irreducible_characters(table: np.ndarray) -> list[RealRepresentation]:
    """The real irreducible representations of the group of ``table``.

    Burnside's method: the class sums multiply as ``C_i C_j = sum_k a_ijk C_k``, and ``omega_k = |C_k|
    chi(C_k) / d`` of every irreducible character is a common eigenvector of the matrices ``(A_i)_jk =
    a_ijk`` (eigenvalue ``omega_i``).  A pair of complex conjugate characters becomes one real representation
    of twice the dimension (character ``2 Re chi``).  The characters are checked for orthogonality.
    """

    classes = conjugacy_classes(table)
    order = table.shape[0]
    count = len(classes)
    class_of = np.empty(order, dtype=np.int64)
    for c, members in enumerate(classes):
        class_of[list(members)] = c
    sizes = np.array([len(members) for members in classes], dtype=np.float64)
    coefficients = np.zeros((count, count, count))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            tally = np.zeros(count)
            for x in ci:
                for y in cj:
                    tally[class_of[table[x, y]]] += 1
            coefficients[i, j] = tally / sizes
    generator = np.random.default_rng(20261115)
    combined = np.tensordot(generator.standard_normal(count), coefficients, axes=(0, 0))
    _values, vectors = np.linalg.eig(combined)
    complex_characters = []
    for column in range(count):
        omega = vectors[:, column] / vectors[0, column]
        dimension = np.sqrt(order / np.sum(np.abs(omega) ** 2 / sizes).real)
        d = int(round(float(dimension)))
        if d < 1 or abs(dimension - d) > 1e-6:
            raise RuntimeError("the class sums gave no integer dimension")
        complex_characters.append((d, d * omega / sizes))
    # orthogonality of the class characters
    gram = np.array([[np.sum(sizes * a * np.conj(b)) for _, b in complex_characters] for _, a in complex_characters])
    if not np.allclose(gram, order * np.eye(count), atol=1e-6):
        raise RuntimeError("the computed characters are not orthogonal")
    representations: list[RealRepresentation] = []
    used = [False] * count
    for a, (d, chi) in enumerate(complex_characters):
        if used[a]:
            continue
        used[a] = True
        pair = False
        if np.allclose(chi.imag, 0.0, atol=1e-8):
            values = chi.real
            dimension = d
        else:
            pair = True
            partner = next(
                b for b in range(count)
                if not used[b] and np.allclose(complex_characters[b][1], np.conj(chi), atol=1e-8)
            )
            used[partner] = True
            values = 2.0 * chi.real
            dimension = 2 * d
        representations.append(RealRepresentation(dimension, np.asarray(values[class_of], dtype=np.float64), pair))
    representations.sort(key=lambda rep: (rep.dimension, -float(np.sum(rep.character))))
    return representations


def _closure(table: np.ndarray, generators) -> frozenset[int]:
    identity, _ = _identity_and_inverses(table)
    members = {identity}
    frontier = [identity]
    generators = [int(g) for g in generators]
    while frontier:
        nxt = []
        for x in frontier:
            for g in generators:
                y = int(table[x, g])
                if y not in members:
                    members.add(y)
                    nxt.append(y)
        frontier = nxt
    return frozenset(members)


def _extended(table: np.ndarray, members: np.ndarray, element: int) -> frozenset[int]:
    """The subgroup generated by the subgroup ``members`` (a boolean mask) and ``element``."""

    mask = members.copy()
    generators = np.append(np.flatnonzero(members), element).astype(np.int64)
    frontier = np.array([element], dtype=np.int64)
    mask[element] = True
    while frontier.size:
        images = table[frontier][:, generators].reshape(-1)
        images = np.unique(images[~mask[images]])
        mask[images] = True
        frontier = images
    return frozenset(np.flatnonzero(mask).tolist())


def subgroups(table: np.ndarray, start=()) -> list[frozenset[int]]:
    """Every subgroup of the group of ``table`` that contains ``start`` (default: every subgroup): the one that
    ``start`` generates, extended by one element at a time to a fixed point.

    Every subgroup is reached: a subgroup ``K`` contains a chain ``{e} < <g_1> < <g_1, g_2> < ... = K``.  The
    extensions of ``S`` by ``g`` and by ``g s`` (``s`` in ``S``) are the same group, so one element of each left
    coset ``g S`` is tried.  The order is that of before: by size, then by the sorted members.  (A subgroup
    ``K`` that contains ``start`` contains such a chain from the group ``start`` generates.)
    """

    order = table.shape[0]
    trivial = _closure(table, tuple(int(g) for g in start))
    found = {trivial}
    frontier = {trivial}
    while frontier:
        new = set()
        for group in frontier:
            members = np.zeros(order, dtype=bool)
            members[list(group)] = True
            tried = members.copy()
            group_members = np.flatnonzero(members)
            for g in range(order):
                if tried[g]:
                    continue
                tried[table[g, group_members]] = True
                extended = _extended(table, members, g)
                if extended not in found:
                    new.add(extended)
        found |= new
        frontier = new
    return sorted(found, key=lambda s: (len(s), sorted(s)))


def one_dimensional_characters(table: np.ndarray, members) -> list[dict[int, int]]:
    """The real one-dimensional characters (homomorphisms to {+1, -1}) of the subgroup ``members``."""

    members = sorted(int(m) for m in members)
    identity, _ = _identity_and_inverses(table)
    generators: list[int] = []
    span = _closure(table, ())
    for g in members:
        if g not in span:
            generators.append(g)
            span = _closure(table, generators)
    characters = []
    for signs in np.ndindex(*([2] * len(generators))):
        value = {identity: 1}
        frontier = [identity]
        consistent = True
        while frontier and consistent:
            nxt = []
            for x in frontier:
                for g, bit in zip(generators, signs):
                    y = int(table[x, g])
                    v = value[x] * (-1 if bit else 1)
                    if y in value:
                        if value[y] != v:
                            consistent = False
                            break
                    else:
                        value[y] = v
                        nxt.append(y)
                if not consistent:
                    break
            frontier = nxt
        if not consistent or len(value) != len(members):
            continue
        # a homomorphism on the whole multiplication table of the subgroup
        if all(value[int(table[a, b])] == value[a] * value[b] for a in members for b in members):
            characters.append(value)
    return characters


@dataclass(frozen=True)
class OrbitalGroupPlan:
    """The orbital subgroup, its solved sectors and their weights, and the right cosets of the full group.

    ``subgroup``: element indices of ``H`` in the full group; ``characters``: one dict per solved sector
    (element index -> +1/-1 on ``H``); ``weights``: the number of states each state of that sector counts
    for; ``cosets``: indices of right coset representatives ``c`` (``G`` = union of ``H c``), the identity
    first; ``cost``: the solved rows times states in units of ``N M``; ``abelian_cost``: the same for the
    largest abelian subgroup of order two per element (the former route).
    """

    subgroup: tuple[int, ...]
    characters: tuple[dict, ...]
    weights: tuple[int, ...]
    cosets: tuple[int, ...]
    cost: float
    abelian_cost: float


def _plan_for(table, representations, members, order):
    members = sorted(members)
    characters = one_dimensional_characters(table, members)
    if not characters:
        return None
    size = len(members)
    multiplicity = np.array(
        [[sum(chi[h] * rep.character[h] for h in members) / size for rep in representations] for chi in characters]
    )
    if not np.allclose(multiplicity, np.rint(multiplicity), atol=1e-6):
        raise RuntimeError("restriction multiplicities are not integers")
    multiplicity = np.rint(multiplicity).astype(np.int64)
    found = multiplicity.sum(axis=0)
    if np.any(found < 1):
        return None
    dimensions = np.array([rep.dimension for rep in representations])
    weights = []
    for row in multiplicity:
        held = np.flatnonzero(row)
        candidates = {float(dimensions[g]) / float(found[g]) for g in held}
        if len(candidates) != 1:
            return None
        weight = candidates.pop()
        if abs(weight - round(weight)) > 1e-9:
            return None
        weights.append(int(round(weight)))
    # sectors exchanged by the normalizer of H in G hold the same levels
    identity, inverse = _identity_and_inverses(table)
    member_set = set(members)
    normalizer = [g for g in range(order) if {int(table[table[g, h], inverse[g]]) for h in members} == member_set]
    keys = [tuple(chi[h] for h in members) for chi in characters]
    index_of = {key: i for i, key in enumerate(keys)}
    orbit_of = [-1] * len(characters)
    orbits = []
    for i, chi in enumerate(characters):
        if orbit_of[i] >= 0:
            continue
        orbit = set()
        for g in normalizer:
            image = tuple(chi[int(table[table[inverse[g], h], g])] for h in members)
            orbit.add(index_of[image])
        for j in orbit:
            orbit_of[j] = len(orbits)
        orbits.append(sorted(orbit))
    solved = [orbit[0] for orbit in orbits]
    levels = np.array([rep.levels for rep in representations])
    states = multiplicity @ levels / order
    cost = float(sum(states[i] for i in solved)) / size
    # right cosets G = union H c
    cosets = [identity]
    covered = set(int(table[h, identity]) for h in members)
    for g in range(order):
        if g not in covered:
            cosets.append(g)
            covered |= {int(table[h, g]) for h in members}
    return OrbitalGroupPlan(
        subgroup=tuple(members),
        characters=tuple(characters[i] for i in solved),
        weights=tuple(weights[i] * len(orbits[k]) for k, i in enumerate(solved)),
        cosets=tuple(cosets),
        cost=cost,
        abelian_cost=float("nan"),
    )


def _is_reflection_group(table: np.ndarray, members) -> bool:
    identity, _ = _identity_and_inverses(table)
    return all(int(table[m, m]) == identity for m in members)


def orbital_group_plan(operations, modulus=None, required=()) -> OrbitalGroupPlan:
    """The admitted subgroup of ``operations`` (a point group of signed permutations) of the least cost.

    Ties go to the larger subgroup and then to the earlier one in the enumeration order.  The abelian route
    of before (the largest subgroup whose elements are their own inverses, all of its characters, weight 1)
    is always admitted, so a plan always exists; its cost is reported as ``abelian_cost``.  ``modulus``: the
    operations are affine, with translations modulo it (:func:`multiplication_table`).  ``required``: indices of
    operations every candidate subgroup contains (the pure translations of a cell that repeats a smaller one:
    a normal subgroup of real characters, in the chosen subgroup of every cell measured; the search over the
    subgroups that contain it takes a tenth of a second instead of seconds for diamond's 192 operations).
    """

    table = multiplication_table(operations, modulus=modulus)
    order = table.shape[0]
    representations = real_irreducible_characters(table)
    best = None
    abelian = None
    for members in subgroups(table, required):
        plan = _plan_for(table, representations, members, order)
        if plan is None:
            continue
        if _is_reflection_group(table, members) and (abelian is None or len(members) > len(abelian.subgroup)):
            # the former route: every sector solved (no merging) at weight 1
            characters = one_dimensional_characters(table, members)
            abelian = OrbitalGroupPlan(tuple(sorted(members)), tuple(characters), (1,) * len(characters),
                                       plan.cosets, 1.0 / len(members), float("nan"))
        if (
            best is None
            or plan.cost < best.cost - 1e-12
            or (abs(plan.cost - best.cost) <= 1e-12 and len(plan.subgroup) > len(best.subgroup))
        ):
            best = plan
    if best is None:
        raise RuntimeError("no orbital subgroup was admitted")
    return OrbitalGroupPlan(best.subgroup, best.characters, best.weights, best.cosets, best.cost,
                            abelian.cost if abelian is not None else float("nan"))


__all__ = [
    "OrbitalGroupPlan",
    "RealRepresentation",
    "conjugacy_classes",
    "multiplication_table",
    "one_dimensional_characters",
    "orbital_group_plan",
    "real_irreducible_characters",
    "subgroups",
]
