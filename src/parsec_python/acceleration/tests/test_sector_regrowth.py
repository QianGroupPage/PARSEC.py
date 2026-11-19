"""The SCF loop gives back every view of the last orbitals before the next eigensolve.

With weighted sectors (a subgroup of the point group, ``Symmetry/orbital_groups.py``) the record of an iteration
holds a copy of the orbitals that carries the weights of the states (``SCF/single_point._public_states``).  Both
views the bases of the sectors.  The loop dropped the views of the orbitals alone before the next solve, so the
copy kept the basis of a sector that grew from a new start in that solve (``_ensure_spectral_bracket`` of
``Eigensolvers/symmetry.py``) beside its new one: a 22^3 diamond cell with the default margin of its sectors ran
out of memory there on 16 A100.  These tests run on the host.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Eigensolvers import solve_eigval
from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Occupations.fermi_dirac import density_from_orbitals
from parsec_python.SCF import prepare_single_point, run_scf

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


@dataclass(frozen=True)
class _SectorViews:
    """Orbitals that hold views until they are released, as ``CuPySymmetryOrbitals`` do.

    Every instance, also each copy that ``dataclasses.replace`` makes, enters ``made``.
    """

    array: np.ndarray | None
    made: list = field(repr=False, compare=False, default_factory=list)
    state_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.made.append(self)

    def release_intermediate_storage(self) -> None:
        object.__setattr__(self, "array", None)


def _problem(max_iterations: int = 6) -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("H", [0.0, 0.0, 0.0])],
        pseudopotentials={"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
        grid=GridSettings(spacing=0.8, radius=4.0, expansion_order=2),
        scf=SCFSettings(max_iterations=max_iterations, number_of_states=2, convergence_criterion=1e-12),
        eigensolver=EigensolverSettings(method="chebdav", tolerance=1e-8),
    )


class RecordCopyReleaseTests(unittest.TestCase):
    def _run(self, weighted: bool):
        made: list = []
        held_at_solve: list[int] = []
        calls = []

        def solver(operator, states, *, settings, state):
            # Views that an earlier solve handed out and that are still held when this one begins.
            held_at_solve.append(sum(item.array is not None for item in made))
            result = solve_eigval(operator, states, settings=settings, state=state)
            calls.append(states)
            return SimpleNamespace(
                eigenvalues=result.eigenvalues,
                vectors=_SectorViews(np.asarray(result.vectors), made),
                state=result.state,
                residual_norms=result.residual_norms,
                # Every state counts once: the weights change no number.
                state_weights=np.ones(len(result.eigenvalues), dtype=np.int64) if weighted else None,
            )

        def density(orbitals, occupations, volume_element):
            return density_from_orbitals(orbitals.array, occupations, volume_element)

        result = run_scf(prepare_single_point(_problem()), eigenproblem_solver=solver,
                         orbital_density_builder=density)
        return result, made, held_at_solve, calls

    def test_the_weighted_copy_in_the_record_is_released_before_the_next_solve(self) -> None:
        result, made, held_at_solve, calls = self._run(weighted=True)
        self.assertGreater(len(calls), 2)
        # One orbitals object per solve and its copy with the weights in the record of the iteration.
        self.assertEqual(len(made), 2 * len(calls))
        self.assertIsNot(result.wavefunctions, made[-2])
        # No solve begins while a view of an earlier one is held.
        self.assertEqual(held_at_solve, [0] * len(calls))
        # Those of the last iteration are the result's: kept.
        self.assertIs(result.wavefunctions, made[-1])
        self.assertIsNotNone(result.wavefunctions.array)

    def test_weights_of_one_change_no_number(self) -> None:
        weighted, *_ = self._run(weighted=True)
        plain, made, held_at_solve, calls = self._run(weighted=False)
        self.assertEqual(len(made), len(calls))
        self.assertEqual(held_at_solve, [0] * len(calls))
        self.assertEqual(weighted.energies.total, plain.energies.total)
        np.testing.assert_array_equal(weighted.eigenvalues, plain.eigenvalues)
        np.testing.assert_array_equal(weighted.wavefunctions.array, plain.wavefunctions.array)


class _Grown(SimpleNamespace):
    """A sector result: weak references reach it, as they reach ``CuPyEigvalResult``."""


class BracketGrowthTests(unittest.TestCase):
    """``_ensure_spectral_bracket`` on three sectors whose spectra are given; the solves are recorded."""

    # The levels of each sector; a solve of ``count`` states returns the lowest ``count`` of them.
    LEVELS = (
        (0.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2),
        (0.5, 1.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1),
        (0.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    )

    def _solver(self, cut):
        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver

        solver = object.__new__(CuPySymmetrySCFEigensolver)
        solver.decomposition = SimpleNamespace(representation_count=3, sector_size=lambda representation: 8,
                                               sector_weights=None)
        solver._margin_cut = dict(cut)
        solver.calls = []

        def run(representations, counts, settings, *, reset):
            import weakref

            # What the bracket still holds of the results it replaces when the solves begin.
            alive = {index: solver.watched[index]() is not None for index in representations}
            solver.calls.append((tuple(representations), list(counts), reset, alive))
            grown = {index: _Grown(eigenvalues=np.array(self.LEVELS[index][:counts[index]]), solver_path="chebff")
                     for index in representations}
            for index, result in grown.items():
                solver.watched[index] = weakref.ref(result)
            return grown

        solver._run_sector_jobs = run
        return solver

    def _bracket(self, solver, counts, settings, environment):
        import weakref

        from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver  # noqa: F401

        results = [_Grown(eigenvalues=np.array(self.LEVELS[index][:count]), solver_path="chebff")
                   for index, count in enumerate(counts)]
        solver.watched = [weakref.ref(result) for result in results]
        with patch.dict("os.environ", environment):
            grown, restarted, reasons = solver._ensure_spectral_bracket(results, counts, 6, settings)
        return grown, restarted, reasons

    def test_cut_sectors_grow_alone_until_the_cutoff_is_theirs(self) -> None:
        # Sector 0 (cut) misses 1.2 and 1.4, which puts the cutoff at 2.6, the last value of sector 1.
        settings = EigvalSettings(safety_buffer=1)
        solver = self._solver({0: 0.0})
        counts = [2, 3, 4]
        grown, restarted, _ = self._bracket(solver, counts, settings, {})
        self.assertTrue(restarted)
        # Sector 0 alone, from new starts (2 -> 3 -> 5), until it brackets the cutoff of 1.4; sector 1 never grows.
        self.assertEqual([call[0] for call in solver.calls], [(0,), (0,)])
        self.assertTrue(all(call[2] for call in solver.calls))
        self.assertEqual(counts, [5, 3, 4])
        np.testing.assert_array_equal(grown[1].eigenvalues, self.LEVELS[1][:3])

    def test_off_grows_every_failing_sector_in_the_same_round_as_before(self) -> None:
        settings = EigvalSettings(safety_buffer=1)
        solver = self._solver({0: 0.0})
        counts = [2, 3, 4]
        self._bracket(solver, counts, settings, {"PARSEC_CUPY_CUT_SECTORS_FIRST": "off",
                                                 "PARSEC_CUPY_SECTOR_GROWTH_RELEASE": "off"})
        first, second = solver.calls[:2]
        self.assertEqual((first[0], first[2]), ((0,), True))
        # Sector 1 grows by half beside its former basis, which its result still holds.
        self.assertEqual((second[0], second[2], second[3]), ((1,), False, {1: True}))
        self.assertEqual(second[1][1], 5)

    def test_a_sector_that_grows_by_half_gives_its_basis_back_first(self) -> None:
        # No margin is cut: sector 2 misses states (its last value 3.0 is below the cutoff of 3.0 + 0).
        settings = EigvalSettings(safety_buffer=1)
        levels = (self.LEVELS[0], self.LEVELS[1], (0.2, 0.3, 0.4, 0.45, 0.6, 7.0, 8.0, 9.0))
        with patch.object(BracketGrowthTests, "LEVELS", levels):
            for environment, released in (({}, True), ({"PARSEC_CUPY_SECTOR_GROWTH_RELEASE": "off"}, False)):
                solver = self._solver({})
                counts = [2, 2, 2]
                self._bracket(solver, counts, settings, environment)
                growth = [call for call in solver.calls if 2 in call[0]]
                self.assertTrue(growth)
                for representations, _counts, reset, alive in growth:
                    self.assertEqual(reset, released)
                    # Released: nothing of the bracket holds the former result of the last sector (once a
                    # loop variable did); kept: the bracket's list still does.
                    self.assertEqual(alive[2], not released)

    def test_an_arpack_growth_keeps_the_state_whose_seed_it_continues(self) -> None:
        settings = EigvalSettings(safety_buffer=1, initial_method="arpack")
        solver = self._solver({})
        counts = [2, 2, 2]
        levels = (self.LEVELS[0], self.LEVELS[1], (0.2, 0.3, 0.4, 0.45, 0.6, 7.0, 8.0, 9.0))
        with patch.object(BracketGrowthTests, "LEVELS", levels):
            self._bracket(solver, counts, settings, {})
        self.assertTrue(solver.calls)
        self.assertTrue(all(call[2] is False for call in solver.calls))

    def test_the_trim_of_the_step_leaves_a_sector_grown_from_a_new_start_whole(self) -> None:
        settings = EigvalSettings(safety_buffer=1)
        solver = self._solver({0: 0.0})
        counts = [2, 3, 4]
        grown, _, _ = self._bracket(solver, counts, settings, {})
        self.assertEqual(solver._regrown_from_new_start, {0})
        solver._solvers = [SimpleNamespace(truncate_state=lambda count, index=index: solver.calls.append(
            ("trim", index, count))) for index in range(3)]
        solver.mpi_context = None
        trimmed = list(counts)
        solver._trim_sector_states_like_parsec(grown, trimmed, 6, 1, keep=frozenset({0}))
        # Sector 0 keeps its 5 states; the others are trimmed as PARSEC trims them (one beyond those selected).
        self.assertEqual(trimmed[0], 5)
        plain = list(counts)
        solver._trim_sector_states_like_parsec(grown, plain, 6, 1)
        self.assertLess(plain[0], 5)
        self.assertEqual(trimmed[1:], plain[1:])
        # A bracket without growth from a new start leaves nothing to keep.
        self._bracket(solver, list(trimmed), settings, {})
        self.assertEqual(solver._regrown_from_new_start, set())

    def test_unknown_values_are_refused(self) -> None:
        from parsec_python.acceleration.Eigensolvers.symmetry import (
            cut_sectors_first_setting,
            regrown_trim_setting,
            sector_growth_release_setting,
        )

        for name, read in (("PARSEC_CUPY_CUT_SECTORS_FIRST", cut_sectors_first_setting),
                           ("PARSEC_CUPY_SECTOR_GROWTH_RELEASE", sector_growth_release_setting),
                           ("PARSEC_CUPY_REGROWN_SECTOR_TRIM", regrown_trim_setting)):
            with patch.dict("os.environ", {name: "maybe"}):
                with self.assertRaisesRegex(ValueError, name):
                    read()
            with patch.dict("os.environ", {name: " OFF "}):
                self.assertEqual(read(), "off")
            with patch.dict("os.environ", {name: ""}):
                self.assertEqual(read(), "auto")


if __name__ == "__main__":
    unittest.main()
