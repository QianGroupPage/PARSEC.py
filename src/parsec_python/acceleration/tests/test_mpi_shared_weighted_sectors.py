"""Sectors shared by groups of MPI ranks: the density and the force terms count every sector once.

With one rank per GPU and more ranks than sectors (``--rank-per-gpu``), or one sector of the identity on several
ranks, the ranks of a group share a sector (``experimental/mpi_scf.sector_rank_groups``).  A CHEBFF first solve
divides the columns of its basis among them (``Eigensolvers/mpi_shared_basis.SharedBasis``) and each rank adds its
own columns to the density and to the nonlocal force terms.  A CHEBDAV first solve (the default ``Eigensolver``)
leaves every rank of the group with the whole basis, which it solves alone.  Before 2026-11-18 every rank of the
group then added that whole basis, so that the density and the force terms of the sector counted as many times as
the group had ranks: on A100 the displaced Si8 cell in its weighted Td sectors had twice the density on 2 x 4 ranks
(-25.7 Ry against one process), and the sector weights played no part in it.  Now the group's first rank, which
alone reports the sector's Ritz values, adds such a sector, and the other ranks of the group add nothing for it
(``MPISectorContext.whole_basis_sectors``).

Here the real sector context and the real symmetry coordinator run on threads that stand for the ranks
(``thread_mpi``), with the toy sector solvers of ``test_mpi_scf``: the eigenvalues, the densities of two SCF steps
and the force terms of 8 and 11 ranks (groups of two and three) equal those of one and of four ranks, with the
sector weights 1, 1, 3, 3 of Td in D2d and without weights, whether every rank of a group holds the whole basis or
its share of the columns.  The GPU runs (Si8 and Si 2x2x2 cells, the Td Ge5 cluster, on 2 x 4 A100 ranks) are in
``item3_scripts`` of the branch.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.acceleration.Eigensolvers.distributed_state import ColumnLayout
from parsec_python.acceleration.Eigensolvers.mpi_shared_basis import SharedBasis
from parsec_python.acceleration.experimental.mpi_scf import MPISectorContext, MPISymmetryDensityBuilder
from parsec_python.acceleration.tests.test_mpi_scf import _ToySector, _make_solver
from parsec_python.acceleration.tests.thread_mpi import run_ranks

# The force rows of the toy projectors of each sector (three projectors), as projector_force_terms gives them
# for orbitals of norm one and occupation one.
_TABLE = {sector: np.random.default_rng(sector + 3).standard_normal((3, 3)) for sector in range(4)}
_WEIGHTS = (1, 1, 3, 3)


def _column_norms(vectors, weights):
    """``sum_n f_n |v_n|^2`` over the leading ``len(weights)`` columns that this rank holds of ``vectors``."""

    weights = np.asarray(weights, dtype=np.float64)
    if isinstance(vectors, SharedBasis):
        selected = vectors[:, : weights.size]
        return sum(float(np.dot(weights[column : column + count],
                                np.sum(np.asarray(selected.local[:, offset : offset + count]) ** 2, axis=0)))
                   for column, offset, count in selected.layout.spans(vectors.group.rank))
    return float(np.dot(weights, np.sum(np.asarray(vectors)[:, : weights.size] ** 2, axis=0)))


def _toy_force_terms(operator, vectors, weights, transposes):
    """What ``Forces.nonlocal_device.projector_force_terms`` adds for a sector: here ``transposes`` names it."""

    return _TABLE[transposes] * _column_norms(vectors, weights)


class _SavedSector(_ToySector):
    """A toy sector whose saved state holds its basis where the sector eigensolver keeps it (``subspace``).

    With ``share`` (the communicator of the sector's group) a rank keeps only its share of the columns, as a
    CHEBFF first solve on a basis that the ranks share leaves them (contiguous ranges, the last rank the rest).
    """

    def __init__(self, sector, dimension, share=None):
        self.share = share
        super().__init__(sector, dimension)

    def solve(self, count, settings):
        result = super().solve(count, settings)
        if self.share is not None and self.share.size > 1:
            size, rank = self.share.size, self.share.rank
            edges = [count * part // size for part in range(size + 1)]
            layout = ColumnLayout(tuple((edges[part], edges[part + 1]) for part in range(size)))
            group = SimpleNamespace(rank=rank, size=size, device=0, local_devices=(0,), devices=tuple(range(size)))
            block = np.asfortranarray(np.asarray(self.device_state.vectors)[:, edges[rank] : edges[rank + 1]])
            self.device_state.vectors = SharedBasis(group, block, layout)
            result = replace(result, vectors=self.device_state.vectors)
        self.device_state.subspace = SimpleNamespace(vectors=self.device_state.vectors, handover=None)
        return result

    def truncate_state(self, count):
        super().truncate_state(count)
        self.device_state.subspace.vectors = self.device_state.vectors


def _run(size, weights, *, divided=False, requested=14):
    """The eigenvalues, two densities and the force terms of a run of ``size`` ranks, one GPU each."""

    def rank_main(ctx):
        context = MPISectorContext(ctx.comm)
        context.gpu_per_rank = True
        solver, builder = _make_solver(context)
        if weights is not None:
            solver.decomposition.sector_weights = np.asarray(weights)
        # The device budget of the margin of weighted sectors: all of it fits.
        solver._margin_fraction_local = lambda *args: 1.0
        share = context.sector_comm if (divided and context.shared) else None
        solver._solvers = [None if item is None else _SavedSector(item.sector, item.dimension, share)
                           for item in solver._solvers]
        solver._operators = [None] * 4
        solver._force_gradients_now = lambda: (3, {sector: sector for sector in range(4)},
                                               dict(gradients=0.0, reduction=0.0))
        if ctx.rank:
            context.worker_loop(solver, builder)
            return context.shared
        density_of = MPISymmetryDensityBuilder(builder, context, solver)
        settings = EigvalSettings(safety_buffer=1)
        first = solver(solver.full_operator, requested, settings=settings)
        occupations = np.linspace(1.0, 0.25, first.eigenvalues.size)
        density = density_of(first.vectors, occupations, 0.25).values
        terms = solver.projector_force_terms(first.representations, first.representation_columns, occupations)
        second = solver(solver.full_operator, requested, settings=settings, state=first.state)
        again = density_of(second.vectors, occupations, 0.25).values
        context.stop_workers()
        return context.shared, first.eigenvalues, density, terms, second.eigenvalues, again

    module = "parsec_python.acceleration.Forces.nonlocal_device"
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda device: nullcontext()))
    with patch(f"{module}.projector_force_terms", _toy_force_terms), \
            patch("parsec_python.acceleration.Eigensolvers.mpi_shared_basis.require_cupy",
                  return_value=(fake_cupy, None)):
        return run_ranks(size, rank_main)


class SharedSectorSumTests(unittest.TestCase):
    def _compare(self, weights, divided):
        reference = _run(1, weights)[0]
        self.assertFalse(reference[0])
        # The density counts every selected state once: twice its occupation in all (two electrons a state).
        charge = 0.25 * float(np.dot(np.array([1, 2, 1, 3, 1, 2, 1, 2]), reference[2]))
        self.assertAlmostEqual(charge, 2 * float(np.sum(np.linspace(1.0, 0.25, reference[1].size))), places=12)
        for size in (4, 8, 11):
            with self.subTest(size=size, weights=weights, divided=divided):
                result = _run(size, weights, divided=divided)
                # Four ranks share no sector; 8 and 11 ranks share them in groups of two and three.
                self.assertEqual(result[0][0], size > 4)
                self.assertTrue(all(item == (size > 4) for item in result[1:]))
                for expected, found in zip(reference[1:], result[0][1:], strict=True):
                    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-13)

    def test_whole_bases_of_a_group_count_once_with_sector_weights(self):
        self._compare(_WEIGHTS, divided=False)

    def test_whole_bases_of_a_group_count_once_without_sector_weights(self):
        self._compare(None, divided=False)

    def test_the_shares_of_a_divided_basis_count_on_every_rank_of_its_group(self):
        self._compare(_WEIGHTS, divided=True)

    def test_one_sector_of_the_identity_counts_once_on_several_ranks(self):
        def rank_main(ctx):
            context = MPISectorContext(ctx.comm)
            context.configure(1, (0,))
            whole = np.eye(4)
            return context.shared, context.whole_basis_sectors(lambda sector: whole)

        self.assertEqual(run_ranks(3, rank_main),
                         [(True, frozenset()), (True, frozenset({0})), (True, frozenset({0}))])


class WholeBasisDecisionTests(unittest.TestCase):
    @staticmethod
    def _context(rank, *, shared=True, owned=(1,), owners=(0, 2, 4, 6)):
        context = MPISectorContext(SimpleNamespace(rank=rank, size=8))
        context.shared, context.owned_sectors, context.sector_owners = shared, tuple(owned), tuple(owners)
        context.representation_count = len(owners)
        return context

    def test_a_rank_leaves_out_only_whole_bases_of_a_group_that_another_rank_reports(self):
        whole = np.eye(5)
        divided = SharedBasis(SimpleNamespace(rank=1, size=2), np.asfortranarray(whole[:, 2:]), ((0, 2), (2, 5)))
        # The second rank of the group of sector 1 (ranks 2 and 3).
        self.assertEqual(self._context(3).whole_basis_sectors(lambda sector: whole), frozenset({1}))
        self.assertEqual(self._context(3).whole_basis_sectors(lambda sector: None), frozenset({1}))
        self.assertEqual(self._context(3).whole_basis_sectors(lambda sector: divided), frozenset())
        # The first rank reports the sector and adds it; ranks that share no sector add all of theirs.
        self.assertEqual(self._context(2).whole_basis_sectors(lambda sector: whole), frozenset())
        self.assertEqual(self._context(3, shared=False, owned=(3, 7)).whole_basis_sectors(lambda sector: whole),
                         frozenset())

    def test_the_density_command_selects_the_states_of_the_sectors_this_rank_adds(self):
        representations, columns = np.array([0, 1, 1, 2, 3]), np.array([0, 0, 1, 0, 0])
        occupations = np.array([1.0, 1.0, 0.5, 0.25, 0.0])
        for basis, expected in ((np.eye(4), []), (SharedBasis(SimpleNamespace(rank=1, size=2),
                                                              np.asfortranarray(np.eye(4)[:, 1:]),
                                                              ((0, 1), (1, 4))), [1, 1])):
            packed = {}

            def pack(results, chosen, chosen_columns):
                packed.update(representations=chosen.tolist(), columns=chosen_columns.tolist())
                return SimpleNamespace(chosen=chosen)

            solver = SimpleNamespace(
                _mpi_latest_results={1: SimpleNamespace(vectors=basis)},
                _pack_selected_wedge_vectors=pack,
                decomposition=SimpleNamespace(invariant_wedge_values=lambda values: np.asarray(values)))

            def builder(orbitals, weights, volume):
                packed["occupations"] = np.asarray(weights).tolist()
                return np.zeros(3)

            self._context(3)._apply(solver, "density", (representations, columns, occupations, 0.5), builder)
            self.assertEqual(packed["representations"], expected)
            self.assertEqual(packed["occupations"], occupations[representations == 1].tolist() if expected else [])

    def test_the_spin_forces_leave_out_whole_bases_channel_by_channel(self):
        def state(vectors):
            return SimpleNamespace(device_state=SimpleNamespace(subspace=SimpleNamespace(vectors=vectors)))

        divided = SharedBasis(SimpleNamespace(rank=1, size=2), np.asfortranarray(np.eye(3)[:, 1:]), ((0, 1), (1, 3)))
        # The active channel (down) holds a whole basis of sector 1, the stored one (up) its share of the columns.
        channels = {0: [None, state(divided), None, None], 1: [None, state(np.eye(3)), None, None]}
        received = []
        solver = SimpleNamespace(
            _spin_channels=(object(), object()), _solvers=channels[1],
            _channel_value=lambda index, name: channels[index],
            _spin_projector_force_terms_local=lambda selections: received.append(selections) or np.zeros((2, 3)))
        selection = (np.array([1, 0, 1]), np.array([0, 0, 1]), np.array([1.0, 1.0, 0.5]))
        self._context(3)._apply(solver, "spin_forces", (selection, selection), None)
        ((up, down),) = received
        np.testing.assert_array_equal(up[2], [1.0, 1.0, 0.5])
        np.testing.assert_array_equal(down[2], [0.0, 1.0, 0.0])
        # A single channel: the occupations of a whole basis that the group's first rank reports are left out.
        forces = []
        solver = SimpleNamespace(_solvers=channels[1],
                                 _projector_force_terms_local=lambda *payload: forces.append(payload) or np.zeros(3))
        self._context(3)._apply(solver, "forces", selection, None)
        np.testing.assert_array_equal(forces[0][2], [0.0, 1.0, 0.0])
        self._context(2)._apply(solver, "forces", selection, None)
        np.testing.assert_array_equal(forces[1][2], [1.0, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
