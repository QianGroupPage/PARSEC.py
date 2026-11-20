"""Exactness of the forces of a periodic cell in the sectors of a point group whose operations carry translations.

A bulk cell at the Gamma point solves its orbitals in the weighted one-dimensional sectors of a subgroup ``H`` of the
group ``G`` of the cell (:class:`~parsec_python.acceleration.Symmetry.axis_reflection.PointGroupReduction`).  Where the
cell repeats a smaller one, or holds a non-symmorphic operation, the operations of ``G`` are ``r -> R r + t h`` with
``t`` in grid steps ``h`` along the axes.  The nonlocal force of the sector states is contracted with the projectors of
one atom per ``H``-orbit (columns scaled by the square root of the orbit size) and completed by the average over ``G``
(:mod:`parsec_python.acceleration.Forces.sector_symmetry`): ``S_G`` of the terms gathered on the representatives is the
force of the ``G``-averaged density matrix of the weighted sector states.

The host tests check the conventions of the operations of ``G`` on the atoms (images modulo the lattice, the
translation ``tau = -R^T t h`` that :func:`parsec_python.Forces.symmetrize_positions` reads) and that identity to
round-off: random sector states with random weights, contracted as the eigensolver does, against the all-atom
full-grid nonlocal forces of the density matrix averaged over explicit images ``P_g psi`` under the grid maps of every
operation of ``G``; and with the eigenstates of a ``G``-invariant model Hamiltonian, the weighted levels of the
sectors and their completed forces against the full grid (the partners of the degenerate levels).  The GPU test solves
a displaced silicon supercell whose group keeps translations, in the sectors and on the full grid
(``PARSEC_PERIODIC_SYMMETRY=off``).
"""

from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python.Forces.symmetry import symmetrize_forces, symmetrize_positions
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.acceleration.Forces.sector_symmetry import ATOM_TOLERANCE, sector_group_operations
from parsec_python.acceleration.Symmetry.axis_reflection import (
    PointGroupReduction,
    SignedPermutationReduction,
    atom_image_table,
    detect_point_group_reduction,
)
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.tests.test_periodic_point_group import displaced_silicon, silicon
from parsec_python.acceleration.tests.test_periodic_translations import _silicon_supercell, _supercell
from parsec_python.models import Atom


def _grid(problem):
    return build_periodic_grid(problem.periodic_cell, problem.grid)


def _reduction(grid, atoms, translations: str = "auto"):
    with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_TRANSLATIONS": translations}):
        return detect_point_group_reduction(grid, list(atoms), SignedPermutationReduction.detect(grid, list(atoms)))


def face_centred_displaced_supercell(spacing: float = 0.64, distance: float = 0.04):
    """Si 2 x 2 x 2 of :func:`silicon`, the neighbours of four atoms moved ``distance`` bohr outward along the bonds.

    The four atoms are the one at the origin and its images under the face-centring translations of the supercell,
    ``(0, a, a)``, ``(a, 0, a)`` and ``(a, a, 0)`` with ``a`` the side of the cube: Td about the origin and those three
    translations stay, the other half-cell translations of the supercell and the face centring of the cube go.  The
    group ``G`` has 96 operations, 72 of them with a translation.
    """

    problem = _silicon_supercell((2, 2, 2), spacing=spacing)
    side = np.diag(np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64))
    cube = 0.5 * side[0]
    centres = (np.zeros(3), np.array((0.0, cube, cube)), np.array((cube, 0.0, cube)), np.array((cube, cube, 0.0)))
    bond_length = cube * np.sqrt(3.0) / 4.0
    atoms = []
    for atom in problem.atoms:
        position = np.asarray(atom.position, dtype=np.float64)
        for centre in centres:
            bond = position - centre
            bond = bond - side * np.round(bond / side)  # nearest image
            length = float(np.linalg.norm(bond))
            if 0.0 < length < bond_length + 1.0e-6:
                position = position + distance * bond / length
                break
        atoms.append(Atom(atom.symbol, tuple(position)))
    return replace(problem, atoms=tuple(atoms))


def _lattice_residual(offsets: np.ndarray, lattice: np.ndarray) -> float:
    """The largest distance (bohr) of the rows of ``offsets`` from the nearest lattice vector."""

    steps = np.round(np.linalg.solve(lattice.T, offsets.T).T)
    return float(np.max(np.abs(offsets - steps @ lattice), initial=0.0))


def _conventions_cases():
    """``(label, grid, atoms, translations setting, |G|, operations with a translation)`` of the host checks."""

    cases = []
    for setting, order, translated in (("off", 24, 0), ("auto", 96, 72)):
        # Td about the origin; with the translations the face centring of the cube (half-cell translations)
        problem = silicon("pbe")
        cases.append((f"Si8 18 points {setting}", _grid(problem), problem.atoms, setting, order, translated))
    # 16 points per edge: a/4 is four steps, and the operations of Oh outside Td with the translation a/4 (1, 1, 1)
    # join (Fd-3m modulo the cube): the sign of tau shows there, not under translations by half the cell
    problem = silicon("pbe", spacing=10.26 / 16)
    cases.append(("Si8 16 points auto", _grid(problem), problem.atoms, "auto", 192, 168))
    grid, atoms = _supercell((2, 2, 2))
    cases.append(("Si 2x2x2 10 points auto", grid, atoms, "auto", 192, 168))
    problem = displaced_silicon()
    cases.append(("displaced Si8", _grid(problem), problem.atoms, "auto", 24, 0))
    problem = face_centred_displaced_supercell()
    cases.append(("displaced Si 2x2x2", _grid(problem), problem.atoms, "auto", 96, 72))
    return cases


class SectorGroupConventionTests(unittest.TestCase):
    """The operations of ``G`` of a reduction on the atoms of a cell: permutations, images modulo the lattice, tau."""

    @classmethod
    def setUpClass(cls) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.cases = _conventions_cases()

    def test_the_images_and_translations_of_the_operations(self) -> None:
        for label, grid, atoms, setting, order, translated in self.cases:
            reduction = _reduction(grid, atoms, setting)
            self.assertIsInstance(reduction, PointGroupReduction, label)
            matrices = np.asarray(reduction.full_operations, dtype=np.float64)
            self.assertEqual(matrices.shape[0], order, label)
            shifts = reduction.full_translations
            if translated:
                self.assertIsNotNone(shifts, label)
                self.assertEqual(int(np.count_nonzero(np.any(shifts != 0, axis=1))), translated, label)
            else:
                self.assertIsNone(shifts, label)
            lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
            steps = np.diag(lattice) / np.asarray(grid.points_per_axis, dtype=np.float64)
            positions = np.array([atom.position for atom in atoms], dtype=np.float64)
            count = positions.shape[0]
            # the table of the reduction, in the order of full_operations: r -> R r + t h carries atom a onto
            # images[g, a] modulo the lattice, and every row is a permutation
            images = atom_image_table(tuple(atoms), matrices, shifts, grid, ATOM_TOLERANCE)
            self.assertIsNotNone(images, label)
            for g, matrix in enumerate(matrices):
                np.testing.assert_array_equal(np.sort(images[g]), np.arange(count), err_msg=f"{label} {g}")
                moved = positions @ matrix.T
                if shifts is not None:
                    moved = moved + shifts[g] * steps
                self.assertLess(_lattice_residual(moved - positions[images[g]], lattice), 1.0e-12, f"{label} {g}")
            # the operations of the forces: the identity first, R (r - tau) = r[image] modulo the lattice with
            # tau = -R^T t h, and the average of the positions over G is the structure itself
            operations = sector_group_operations(reduction, atoms, grid, lattice)
            self.assertEqual(len(operations), order, label)
            np.testing.assert_array_equal(operations[0].matrix, np.eye(3), err_msg=label)
            np.testing.assert_array_equal(operations[0].image, np.arange(count), err_msg=label)
            self.assertIsNone(operations[0].translation, label)
            distinct = {
                (tuple(matrix.astype(int).ravel()), () if shifts is None else tuple(int(v) for v in shifts[g]))
                for g, matrix in enumerate(matrices)
            }
            self.assertEqual(len(distinct), order, label)
            with_translation = 0
            for operation in operations:
                np.testing.assert_array_equal(np.sort(operation.image), np.arange(count), err_msg=label)
                tau = np.zeros(3) if operation.translation is None else np.asarray(operation.translation)
                with_translation += operation.translation is not None
                moved = (positions - tau) @ operation.matrix.T
                self.assertLess(_lattice_residual(moved - positions[operation.image], lattice), 1.0e-12, label)
            self.assertEqual(with_translation, translated, label)
            np.testing.assert_allclose(
                symmetrize_positions(positions, operations), positions, rtol=0, atol=1.0e-12, err_msg=label
            )

    def test_the_checks_see_a_wrong_translation(self) -> None:
        # tau = +R^T t h (the sign) and tau = -R t h (the transpose) carry the atoms elsewhere: the positions test
        # above would fail with either, on the non-symmorphic operations and on the face centring respectively
        for label, grid, atoms, setting, _order, _translated in self.cases:
            if label not in {"Si8 16 points auto", "Si8 18 points auto"}:
                continue
            reduction = _reduction(grid, atoms, setting)
            lattice = np.asarray(grid.cell.lattice_vectors, dtype=np.float64)
            positions = np.array([atom.position for atom in atoms], dtype=np.float64)
            operations = sector_group_operations(reduction, atoms, grid, lattice)
            wrong = []
            for operation in operations:
                if operation.translation is None:
                    wrong.append(operation)
                    continue
                tau = np.asarray(operation.translation, dtype=np.float64)
                if label.startswith("Si8 16"):
                    wrong.append(replace(operation, translation=-tau))  # tau = +R^T t h
                else:
                    shift = -operation.matrix @ tau  # t h, from tau = -R^T t h
                    wrong.append(replace(operation, translation=-operation.matrix @ shift))  # tau = -R t h
            self.assertGreater(float(np.abs(symmetrize_positions(positions, wrong) - positions).max()), 1.0e-2, label)


def _species(problem):
    from parsec_python.V_ion.ionic_potential import load_pseudopotentials

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the file of examples/3d_Si was generated for ca
        return load_pseudopotentials(problem.pseudopotentials, xc_functional=problem.scf.xc_functional)


class _HostSectorForces:
    """The nonlocal forces of sector states formed on the host as the eigensolver and the forces form them.

    The sector operators take the projectors of :func:`compress_projectors_for_sectors`, the gradients go through
    ``CuPySymmetrySCFEigensolver._sector_force_gradients`` (compressed and reduced to every sector), the terms are those
    of :meth:`DeviceProjectorGradients.contract` and the forces those of ``compute._sector_atom_forces`` (gathered on
    the representatives, averaged over ``G``).  ``operator`` and ``gradients`` are the all-atom full-grid ones.
    """

    def __init__(self, problem, orbits: str, translations: str = "auto") -> None:
        from parsec_python.Forces.periodic import build_periodic_projector_gradients
        from parsec_python.V_ion.ionic_potential import build_nonlocal_projectors
        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
        from parsec_python.acceleration.Symmetry.axis_reflection import compress_projectors_for_sectors
        from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition

        self.problem = problem
        self.environment = {"PARSEC_PERIODIC_TRANSLATIONS": translations, "PARSEC_SECTOR_PROJECTOR_ORBITS": orbits}
        self.grid = grid = _grid(problem)
        self.atoms = atoms = tuple(problem.atoms)
        self.lattice = lattice = np.asarray(problem.periodic_cell.lattice_vectors, dtype=np.float64)
        potentials = _species(problem)
        self.operator = build_nonlocal_projectors(grid, atoms, potentials, problem.pseudopotentials, lattice)
        self.gradients = build_periodic_projector_gradients(grid, atoms, potentials, problem.pseudopotentials, lattice)
        with mock.patch.dict(os.environ, self.environment):
            self.reduction = reduction = _reduction(grid, atoms, translations)
            if not isinstance(reduction, PointGroupReduction):
                raise AssertionError("the cell has no point-group reduction")
            self.decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
            self.sectors = tuple(range(self.decomposition.representation_count))
            self.operators = self.decomposition.reduce_nonlocal_operators(
                compress_projectors_for_sectors(self.operator, reduction)
            )
            self.solver = SimpleNamespace(
                # The eigensolver asks for the representative atoms (atom_indices) where the sectors take one
                # atom per orbit; the gradients of every atom serve as well, since the compression matches the
                # columns by label.
                projector_gradient_provider=lambda atom_indices=None: self.gradients,
                decomposition=self.decomposition,
                _owned_representations=self.sectors,
            )
            self.columns, self.transposes, _ = CuPySymmetrySCFEigensolver._sector_force_gradients(self.solver)
        self.labels = self.solver.projector_force_labels
        self.group = sector_group_operations(reduction, atoms, grid, lattice)

    def terms(self, vectors: dict, occupations: dict) -> np.ndarray:
        """``t = 4 sum_n f_n s_q (B_s^T c_n)(G_s^T c_n)`` of the sector vectors ``c`` (occupations with the weights)."""

        terms = np.zeros((self.columns, 3), dtype=np.float64)
        for sector, block in vectors.items():
            operator = self.operators[sector]
            signed = operator.signs[:, None] * np.asarray(operator.projectors.T @ block)
            for axis in range(3):
                slopes = np.asarray(self.transposes[sector][axis] @ block)
                terms[:, axis] += 4.0 * np.sum(signed * slopes * occupations[sector][None, :], axis=1)
        return terms

    def forces(self, terms: np.ndarray) -> np.ndarray:
        from parsec_python.acceleration.Forces.compute import _sector_atom_forces

        system = SimpleNamespace(
            backend=SimpleNamespace(symmetry_eigensolver=self.solver), input=self.problem, grid=self.grid,
            atoms=self.atoms,
        )
        with mock.patch.dict(os.environ, self.environment):
            return _sector_atom_forces(system, self.solver, terms, self.operator, len(self.atoms))

    def full_grid_forces(self, states: np.ndarray, occupations: np.ndarray) -> np.ndarray:
        from parsec_python.Forces.nonlocal_force import nonlocal_forces

        return nonlocal_forces(self.operator, self.gradients, states, occupations, len(self.atoms))

    def row_maps(self):
        """The grid row map of every operation of ``G``, in the order of ``full_operations``."""

        from parsec_python.acceleration.Symmetry.axis_reflection import _grid_row_mapping_operation

        shifts = self.reduction.full_translations
        maps = []
        for g, matrix in enumerate(np.asarray(self.reduction.full_operations)):
            mapping = _grid_row_mapping_operation(
                self.grid, matrix, 1.0e-10, translation=None if shifts is None else shifts[g]
            )
            if mapping is None:
                raise AssertionError("an operation of G does not map the grid")
            maps.append(mapping)
        return maps


class SectorNonlocalForceIdentityTests(unittest.TestCase):
    """``S_G`` of the sector contraction on the representatives is the force of the ``G``-averaged density matrix.

    Random sector vectors with random occupations, each counted with the weight of its sector: the identity holds for
    any ``H``-invariant density matrix of sector states, not only for eigenstates.
    """

    def _identity(self, problem, orbits: str, *, states: int = 3, seed: int = 7):
        host = _HostSectorForces(problem, orbits)
        self.assertEqual(host.columns, len(host.labels))
        for sector in host.sectors:
            self.assertEqual(tuple(host.operators[sector].labels), host.labels)
            self.assertEqual(host.operators[sector].projectors.shape[1], host.columns)
        if orbits == "off":
            self.assertEqual(host.labels, tuple(host.operator.labels))
        else:
            self.assertLess(len(host.labels), len(host.operator.labels))
        rng = np.random.default_rng(seed)
        vectors, occupations, expanded = {}, {}, []
        for sector in host.sectors:
            vectors[sector] = rng.standard_normal((host.decomposition.sector_size(sector), states))
            occupations[sector] = rng.uniform(0.2, 1.0, states) * float(host.reduction.sector_weights[sector])
            expanded += [host.decomposition.expand_vector(vectors[sector][:, n], sector) for n in range(states)]
        terms = host.terms(vectors, occupations)
        forces = host.forces(terms)
        states_full = np.column_stack(expanded)
        weights = np.concatenate([occupations[sector] for sector in host.sectors])
        count = len(host.atoms)
        # the solved states alone: S_H of the terms gathered on the representatives (n_a T_a there)
        subgroup = SimpleNamespace(
            full_operations=host.reduction.operations, full_translations=host.reduction.translations
        )
        gathered = np.zeros((count, 3))
        owners = np.fromiter((label[0] for label in host.labels), dtype=np.int64, count=len(host.labels))
        for axis in range(3):
            gathered[:, axis] = np.bincount(owners, weights=terms[:, axis], minlength=count)
        within = symmetrize_forces(gathered, sector_group_operations(subgroup, host.atoms, host.grid, host.lattice))
        solved = host.full_grid_forces(states_full, weights)
        # the reference: the all-atom forces of the density matrix averaged over the images P_g psi under every
        # operation of G, on the full grid
        reference = np.zeros((count, 3))
        maps = host.row_maps()
        for mapping in maps:
            images = np.empty_like(states_full)
            images[mapping] = states_full
            reference += host.full_grid_forces(images, weights)
        reference /= float(len(maps))
        return SimpleNamespace(
            forces=forces, reference=reference, within=within, solved=solved, group=host.group,
            reduction=host.reduction,
        )

    def assert_identity(self, outcome, label: str) -> None:
        scale = float(np.abs(outcome.reference).max())
        self.assertGreater(scale, 1.0e-3, f"{label}: the reference has no force")
        # the grid maps and the atom images of G agree: the reference is G-symmetric
        np.testing.assert_allclose(
            symmetrize_forces(outcome.reference, outcome.group), outcome.reference, rtol=0, atol=1.0e-12 * scale,
            err_msg=label,
        )
        # the solved states alone: H-equivariant terms
        solved_scale = float(np.abs(outcome.solved).max())
        np.testing.assert_allclose(outcome.within, outcome.solved, rtol=0, atol=1.0e-12 * solved_scale, err_msg=label)
        # the completion over G
        np.testing.assert_allclose(outcome.forces, outcome.reference, rtol=0, atol=1.0e-12 * scale, err_msg=label)
        # which is needed: without it the weighted partners and the other orbits miss
        self.assertGreater(float(np.abs(outcome.solved - outcome.reference).max()), 1.0e-3 * scale, label)

    def test_a_displaced_cube_in_the_sectors_of_td(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = displaced_silicon()
        for orbits in ("auto", "off"):
            outcome = self._identity(problem, orbits)
            self.assertEqual(len(outcome.group), 24)
            self.assert_identity(outcome, f"displaced Si8 orbits {orbits}")

    def test_a_displaced_supercell_with_translations(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = face_centred_displaced_supercell()
        for orbits in ("auto", "off"):
            outcome = self._identity(problem, orbits)
            self.assertEqual(len(outcome.group), 96)
            self.assertIsNotNone(outcome.reduction.full_translations)
            self.assertIsNotNone(outcome.reduction.translations)
            self.assert_identity(outcome, f"displaced Si 2x2x2 orbits {orbits}")


class SectorEigenstateForceTests(unittest.TestCase):
    """The weighted sector eigenstates of a ``G``-invariant model Hamiltonian give the forces of the full grid.

    ``H = -1/2 Laplacian + V + B D B^T`` with the all-atom projectors and Gaussian wells on the atoms averaged over
    ``G``; the sectors take the projectors of one atom per orbit, as the eigensolver does.  The levels below a gap of
    the full spectrum, each sector state counted with its weight (its degeneracy in ``G`` over the solved sectors of
    the level), must be those of the full grid, and ``S_G`` of their sector forces the nonlocal forces of the full-grid
    eigenstates: the completion of the weighted partners (Schur) checked on the density matrix the forces read.
    """

    def _model(self, problem, orbits: str):
        import scipy.sparse as sp

        from parsec_python.Laplacian import build_negative_laplacian

        host = _HostSectorForces(problem, orbits)
        grid = host.grid
        side = np.diag(host.lattice)
        wells = np.zeros(grid.size)
        for atom in host.atoms:
            offset = np.asarray(grid.coordinates, dtype=np.float64) - np.asarray(atom.position, dtype=np.float64)
            offset -= side * np.round(offset / side)
            wells -= 0.8 * np.exp(-np.sum(offset**2, axis=1) / 2.0)
        maps = host.row_maps()
        averaged = np.zeros(grid.size)
        for mapping in maps:
            averaged[mapping] += wells
        local = 0.5 * build_negative_laplacian(grid) + sp.diags(averaged / float(len(maps)))
        operator = host.operator
        nonlocal_ = operator.projectors @ sp.diags(operator.signs) @ operator.projectors.T
        with mock.patch.dict(os.environ, host.environment):
            reduced = host.decomposition.reduce_operators(sp.csr_matrix(local))
        return host, sp.csr_matrix(local + nonlocal_), reduced

    def _check(self, problem, orbits: str, order: int, label: str) -> None:
        host, hamiltonian, reduced = self._model(problem, orbits)
        self.assertEqual(len(host.group), order, label)
        values, vectors = np.linalg.eigh(hamiltonian.toarray())
        # a gap of the full spectrum between 16 and 60 states
        gaps = np.diff(values[:61])
        cut = 16 + int(np.argmax(gaps[15:60]))
        self.assertGreater(float(gaps[cut - 1]), 1.0e-3, label)
        energy = 0.5 * (values[cut - 1] + values[cut])
        sector_vectors, sector_weights, levels = {}, {}, []
        for sector in host.sectors:
            projectors = host.operators[sector].projectors.toarray()
            matrix = reduced[sector].toarray() + (projectors * host.operators[sector].signs) @ projectors.T
            sector_values, sector_states = np.linalg.eigh(matrix)
            below = np.flatnonzero(sector_values < energy)
            weight = int(host.reduction.sector_weights[sector])
            if below.size:
                sector_vectors[sector] = sector_states[:, below]
                sector_weights[sector] = np.full(below.size, float(weight))
                levels += list(np.repeat(sector_values[below], weight))
        # the weighted levels of the sectors are the levels of the full grid below the gap
        self.assertEqual(len(levels), cut, label)
        np.testing.assert_allclose(np.sort(levels), values[:cut], rtol=0, atol=1.0e-10, err_msg=label)
        forces = host.forces(host.terms(sector_vectors, sector_weights))
        reference = host.full_grid_forces(vectors[:, :cut], np.ones(cut))
        scale = float(np.abs(reference).max())
        self.assertGreater(scale, 1.0e-4, f"{label}: the reference has no force")
        np.testing.assert_allclose(forces, reference, rtol=0, atol=1.0e-10 * scale, err_msg=label)
        # the partners of the weighted states (PointGroupReduction.partner_states) give the same density matrix
        expanded = [
            host.decomposition.expand_vector(sector_vectors[sector][:, n], sector)
            for sector in sector_vectors
            for n in range(sector_vectors[sector].shape[1])
        ]
        weights = np.concatenate([sector_weights[sector] for sector in sector_vectors]).astype(np.int64)
        partners = host.reduction.partner_states(np.column_stack(expanded), weights)
        self.assertEqual(partners.shape[1], cut, label)
        np.testing.assert_allclose(
            host.full_grid_forces(partners, np.ones(cut)), reference, rtol=0, atol=1.0e-10 * scale, err_msg=label
        )

    def test_a_displaced_cube(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = displaced_silicon()
        problem = replace(base, grid=replace(base.grid, spacing=10.26 / 12))
        for orbits in ("auto", "off"):
            self._check(problem, orbits, 24, f"displaced Si8 orbits {orbits}")

    def test_a_displaced_supercell_with_translations(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = face_centred_displaced_supercell(spacing=10.26 / 8)
        for orbits in ("auto", "off"):
            self._check(problem, orbits, 96, f"displaced Si 2x2x2 orbits {orbits}")




def replicated_displaced_supercell(spacing: float = 0.64):
    """Si 2 x 2 x 2 of eight copies of :func:`displaced_silicon`: Td about the origin with all seven half-cell
    translations of the supercell (``G`` of 192 operations, 168 with a translation), the face centring of the cube
    broken."""

    base = displaced_silicon()
    side = np.asarray(base.periodic_cell.lattice_vectors, dtype=np.float64)
    atoms = [
        Atom(atom.symbol, tuple(np.asarray(atom.position) + np.array((i, j, k)) @ side))
        for i in range(2)
        for j in range(2)
        for k in range(2)
        for atom in base.atoms
    ]
    return replace(_silicon_supercell((2, 2, 2), spacing=spacing), atoms=tuple(atoms))


@unittest.skipUnless(cupy_available() and native_available(), "CUDA runtime or native extension is unavailable")
class PeriodicSectorTranslationForceTests(unittest.TestCase):
    """Forces of supercells whose group keeps translations, in their sectors and on the full grid.

    About a minute on one A100 for both cells (three SCF loops each, 32^3 points, 176 states).  Measured on A100 at
    the commit of this test: sectors against the full grid within 7e-11 (nonlocal) and 7e-10 Ry/bohr (total) with
    forces up to 7.6e-3; the projectors of one atom per orbit against those of every atom within 2e-14; the same SCF
    against the host contraction of the full-grid partner orbitals within 2e-13.
    """

    def _forces(self, problem, order: int, label: str) -> None:
        from parsec_python.Forces.nonlocal_force import nonlocal_forces
        from parsec_python.acceleration import driver
        from parsec_python.acceleration.Forces.sector_symmetry import sector_force_operations, sector_reduction

        runs = (
            ("sectors", {"PARSEC_PERIODIC_SYMMETRY": "auto", "PARSEC_PERIODIC_TRANSLATIONS": "auto"}),
            (
                "every atom",
                {
                    "PARSEC_PERIODIC_SYMMETRY": "auto",
                    "PARSEC_PERIODIC_TRANSLATIONS": "auto",
                    "PARSEC_SECTOR_PROJECTOR_ORBITS": "off",
                },
            ),
            ("full grid", {"PARSEC_PERIODIC_SYMMETRY": "off"}),
        )
        tables, groups, partners = {}, {}, None
        for run, environment in runs:
            name = f"{label} {run}"
            with mock.patch.dict(os.environ, environment), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = driver.prepare_single_point(problem, backend="cupy")
                # the sector run keeps its full-grid orbitals, the partners of the weighted states included
                system.materialize_final_wavefunctions = run == "sectors"
                reduction = sector_reduction(system)
                if run == "full grid":
                    self.assertIsNone(reduction, name)
                else:
                    self.assertIsNotNone(reduction, f"{name}: the cell was not solved in the sectors of a point group")
                    self.assertIsNotNone(reduction.full_translations, name)
                    self.assertIsNotNone(reduction.translations, name)
                    self.assertIn(3, [int(value) for value in reduction.sector_weights], name)
                    groups[run] = sector_force_operations(system)
                    self.assertEqual(len(groups[run]), order, name)
                    self.assertTrue(any(operation.translation is not None for operation in groups[run]), name)
                result = driver.run_scf(system)
                self.assertTrue(result.converged, name)
                tables[run] = driver.compute_forces(system, result, symmetrize=False)
                if run == "sectors":
                    # the same SCF: the all-atom contraction of the full-grid partner orbitals on the host
                    wavefunctions = np.asarray(result.wavefunctions)
                    occupations = np.asarray(result.occupations, dtype=np.float64)
                    self.assertEqual(wavefunctions.shape[1], occupations.size, name)
                    partners = nonlocal_forces(
                        system.nonlocal_operator, system.projector_gradients(), wavefunctions, occupations,
                        len(system.atoms),
                    )
                    del wavefunctions
        sectors, every, full = tables["sectors"], tables["every atom"], tables["full grid"]
        self.assertGreater(float(np.abs(full.total).max()), 1.0e-3, label)
        np.testing.assert_allclose(
            sectors.nonlocal_pseudopotential, partners, rtol=0, atol=1.0e-9, err_msg=f"{label} same SCF"
        )
        for field in ("nonlocal_pseudopotential", "total"):
            np.testing.assert_allclose(
                getattr(sectors, field), getattr(full, field), rtol=0, atol=2.0e-6, err_msg=f"{label} sectors {field}"
            )
            np.testing.assert_allclose(
                getattr(sectors, field), getattr(every, field), rtol=0, atol=1.0e-8, err_msg=f"{label} orbits {field}"
            )
        np.testing.assert_allclose(
            symmetrize_forces(sectors.total, groups["sectors"]), sectors.total, rtol=0, atol=1.0e-12, err_msg=label
        )

    def test_the_face_centring_translations_of_a_supercell(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = face_centred_displaced_supercell()
        self._forces(problem, 96, "Si 2x2x2, face-centring translations")

    def test_every_half_cell_translation_of_a_supercell(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem = replicated_displaced_supercell()
        self._forces(problem, 192, "Si 2x2x2, half-cell translations")


if __name__ == "__main__":
    unittest.main()
