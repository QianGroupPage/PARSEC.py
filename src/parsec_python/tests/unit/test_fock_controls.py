"""The controls of the Fock loop: carried Anderson history, adaptive inner criterion, outer DIIS.

Independent references: a linear fixed-point problem whose offset changes
from one loop to the next (the inner loops of a Fock loop in their linear
regime) for the carried mixer; direct dense algebra for the natural orbitals
and the Frobenius products of the outer DIIS; the reference Fock loop with
the fixed controls for the energies of the loop with the new ones.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Mixer import AndersonMixer
from parsec_python.Mixer.carried import CarriedAndersonMixer, RecordingMixer, ReducedVectors
from parsec_python.SCF.fock_controls import FIRST_RATIO, FORCING, InnerCriterion
from parsec_python.SCF.fock_diis import OccupiedBlock, OuterDIIS, OutputSet, orthonormal_complement
from parsec_python.SCF.single_point import SCFWarmStart, loop_criterion, warm_start_mixer
from parsec_python.models import ExactExchangeSettings, ExchangeLoopIteration, MixingSettings


def _linear_problem(size: int = 40, seed: int = 3):
    """``F(V) = b - A V`` with a symmetric positive A of spread spectrum (a dielectric-like map)."""

    generator = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(generator.standard_normal((size, size)))
    values = np.geomspace(0.6, 6.0, size)
    matrix = basis @ np.diag(values) @ basis.T
    return matrix, generator


def _iterations(mixer, matrix, offset, start, tolerance=1.0e-10, limit=200):
    potential = start.copy()
    for iteration in range(1, limit + 1):
        output = potential + (offset - matrix @ potential)
        if np.linalg.norm(output - potential) < tolerance:
            return iteration, potential
        potential = mixer.mix(potential, output, iteration=iteration)
    return limit, potential


class CarriedMixerTests(unittest.TestCase):
    def test_carried_pairs_shorten_the_next_loop_and_keep_its_fixed_point(self) -> None:
        matrix, generator = _linear_problem()
        settings = MixingSettings(parameter=0.3, memory=4, restart=200)
        offsets = [generator.standard_normal(matrix.shape[0])]
        for _ in range(3):  # each loop's offset moves by a tenth of the one before
            offsets.append(offsets[-1] + 0.1 ** len(offsets) * generator.standard_normal(matrix.shape[0]))
        carried = CarriedAndersonMixer(settings)
        start = np.zeros(matrix.shape[0])
        fresh_counts, carried_counts = [], []
        fresh_start = carried_start = start
        for offset in offsets:
            count, fresh_start = _iterations(AndersonMixer(settings), matrix, offset, fresh_start)
            fresh_counts.append(count)
            count, carried_start = _iterations(carried, matrix, offset, carried_start)
            carried_counts.append(count)
            np.testing.assert_allclose(carried_start, np.linalg.solve(matrix, offset), atol=1.0e-9)
        self.assertEqual(carried_counts[0], fresh_counts[0])  # nothing to carry into the first loop
        self.assertLess(sum(carried_counts[1:]), sum(fresh_counts[1:]))
        self.assertEqual(carried.seed_count, settings.memory)

    def test_first_call_is_linear_mixing_and_restart_clears_the_seeds(self) -> None:
        settings = MixingSettings(parameter=0.3, memory=2, restart=3)
        mixer = CarriedAndersonMixer(settings)
        x, y = np.array([1.0, 2.0]), np.array([2.0, 0.0])
        np.testing.assert_allclose(mixer.mix(x, y, iteration=1), x + 0.3 * (y - x))
        mixer.mix(x + 0.1, y, iteration=2)
        mixer.mix(x + 0.2, y, iteration=3)
        mixer.mix(x + 0.3, y, iteration=1)  # a new loop: the first differences become seeds
        self.assertEqual(mixer.seed_count, 2)
        mixer.mix(x + 0.4, y, iteration=4)  # Restart_Mixing: (4 - 1) % 3 == 0
        self.assertEqual(mixer.seed_count, 0)

    def test_unit_weights_of_a_wedge_give_the_plain_algebra(self) -> None:
        class Reducer:
            weights = np.ones(5)

            @staticmethod
            def wedge_values(field):
                return np.asarray(field)

            @staticmethod
            def expand_values(values):
                return np.asarray(values)

            field = expand_values

        matrix, generator = _linear_problem(5)
        offset = generator.standard_normal(5)
        plain = CarriedAndersonMixer(MixingSettings())
        wedge = CarriedAndersonMixer(MixingSettings(), ReducedVectors(Reducer()))
        x = np.zeros(5)
        for iteration in range(1, 6):
            output = x + offset - matrix @ x
            a = plain.mix(x, output, iteration=iteration)
            b = wedge.mix(x, output, iteration=iteration)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
            x = a

    def test_codec_of_each_loop(self) -> None:
        """Full grid: plain; unpolarized wedge: its multiplicities; spin sectors: those of both joined halves."""

        from types import SimpleNamespace

        from parsec_python.Mixer.carried import PlainVectors, ReducedVectors, WeightedVectors
        from parsec_python.SCF.fock_controls import fock_loop_mixer

        weights = np.array([1.0, 2.0, 4.0])

        def system(spin):
            return SimpleNamespace(
                input=SimpleNamespace(mixing=MixingSettings(), scf=SimpleNamespace(spin_polarized=spin))
            )

        settings = ExactExchangeSettings(mixer_history="carry")
        wedge = SimpleNamespace(weights=weights, wedge_values=lambda field: field)
        spin_sectors = SimpleNamespace(reducer=SimpleNamespace(weights=weights), join_channels=lambda fields: fields)
        self.assertIsInstance(fock_loop_mixer(system(False), settings, {}).codec, PlainVectors)
        self.assertIsInstance(fock_loop_mixer(system(True), settings, {}).codec, PlainVectors)
        self.assertIsInstance(
            fock_loop_mixer(system(False), settings, {"scalar_field_adapter": wedge}).codec, ReducedVectors
        )
        codec = fock_loop_mixer(system(True), settings, {"scalar_field_adapter": spin_sectors}).codec
        self.assertIsInstance(codec, WeightedVectors)
        np.testing.assert_array_equal(codec.weights, np.concatenate((weights, weights)))
        with self.assertRaises(ValueError):
            codec.values(np.zeros(3))
        with self.assertRaises(NotImplementedError):
            fock_loop_mixer(system(True), settings, {"scalar_field_adapter": SimpleNamespace()})
        self.assertIsNone(fock_loop_mixer(system(True), ExactExchangeSettings(), {}))

    def test_recording_mixer_changes_nothing(self) -> None:
        matrix, generator = _linear_problem(8)
        offset = generator.standard_normal(8)
        reference, inner = AndersonMixer(MixingSettings()), AndersonMixer(MixingSettings())
        from parsec_python.Mixer.carried import PlainVectors

        recorder = RecordingMixer(inner, PlainVectors(), 3)
        x = y = np.zeros(8)
        for iteration in range(1, 7):
            x = reference.mix(x, x + offset - matrix @ x, iteration=iteration)
            y = recorder.mix(y, y + offset - matrix @ y, iteration=iteration)
            np.testing.assert_array_equal(x, y)
        self.assertEqual(len(recorder.pairs), 3)

    def test_warm_start_hooks_of_the_loop(self) -> None:
        start = SCFWarmStart(input_potential=np.zeros(2), hartree_potential=np.zeros(2))
        self.assertEqual(loop_criterion(None, 1e-6, 1.0), 1e-6)
        self.assertEqual(loop_criterion(start, 1e-6, 1.0), 1e-6)
        relative = replace(start, relative_criterion=0.01)
        self.assertEqual(loop_criterion(relative, 1e-6, 1e-3), 1e-5)
        self.assertEqual(loop_criterion(relative, 1e-6, 1e-5), 1e-6)
        built = object()
        self.assertIs(warm_start_mixer(start, lambda: built), built)
        carried = object()
        self.assertIs(warm_start_mixer(replace(start, mixer=carried), lambda: built), carried)


class InnerCriterionTests(unittest.TestCase):
    @staticmethod
    def loop(first, last):
        from types import SimpleNamespace

        records = [
            SimpleNamespace(iteration=1, weighted_residual=first, plain_residual=first),
            SimpleNamespace(iteration=2, weighted_residual=last, plain_residual=last),
        ]
        return SimpleNamespace(history=records)

    def test_shares_and_the_last_loop(self) -> None:
        rule = InnerCriterion(adaptive=True, final=1.0e-6, tolerance=1.0e-10)
        self.assertAlmostEqual(rule.share(), FORCING * FIRST_RATIO)
        rule.observe(self.loop(5e-3, 4e-5), 1.0e-4)
        self.assertAlmostEqual(rule.predicted_error(), 1.0e-6)  # FIRST_DEXX_RATIO
        rule.observe(self.loop(5e-4, 4e-6), 2.0e-6)
        self.assertAlmostEqual(rule.share(), FORCING * 0.1)  # ratio of first residuals 0.1
        rule.observe(self.loop(5e-5, 4e-7), 4.0e-8)
        # predicted 4e-8 * 0.02 = 8e-10 < LAST_MARGIN * 1e-10: the next loop is exact
        self.assertIsNone(rule.share())
        self.assertFalse(rule.tight(self.loop(5e-5, 2e-6)))
        self.assertTrue(rule.tight(self.loop(5e-5, 5e-7)))
        # An iteration of level passes after the converged one (the same V_in, its
        # residual a little above the criterion) is not a step of the loop.
        from types import SimpleNamespace

        from parsec_python.SCF.fock_controls import loop_iterations

        refined = self.loop(5e-5, 9.2e-7)
        refined.history.append(
            SimpleNamespace(iteration=3, weighted_residual=1.7e-6, plain_residual=1.7e-6, level_refinement=True)
        )
        self.assertTrue(rule.tight(refined))
        self.assertEqual(loop_iterations(refined), 2)
        self.assertFalse(rule.tight(self.loop(5e-5, 1.7e-6)))
        self.assertAlmostEqual(rule.threshold(self.loop(5e-3, 1e-5), 0.01), 5e-5)
        fixed = InnerCriterion(adaptive=False, final=1.0e-6, tolerance=1.0e-8)
        self.assertIsNone(fixed.share())
        # The looser start only for spin-unpolarized starts (on any grid), never
        # below the input's criterion, and continued where its levels are partly filled.
        from types import SimpleNamespace

        from parsec_python.SCF.fock_controls import FRACTIONAL_START, START_CRITERION, fractional_start

        self.assertEqual(rule.start_criterion(False), START_CRITERION)
        self.assertIsNone(rule.start_criterion(True))
        self.assertIsNone(fixed.start_criterion(False))
        self.assertIsNone(InnerCriterion(adaptive=True, final=2.0e-4, tolerance=1e-6).start_criterion(False))
        self.assertFalse(fractional_start(SimpleNamespace(occupations=np.array([1.0, 1.0 - 1e-6, 1e-5, 0.0]))))
        self.assertTrue(fractional_start(SimpleNamespace(occupations=np.array([1.0, 0.5, 0.5, 0.0]))))
        self.assertTrue(fractional_start(SimpleNamespace(occupations=np.array([1.0, FRACTIONAL_START, 0.0]))))
        channels = (SimpleNamespace(occupations=np.ones(2)), SimpleNamespace(occupations=np.array([1.0, 0.3])))
        self.assertTrue(fractional_start(SimpleNamespace(occupations=None, spin=channels)))


def _orthonormal(rows, columns, generator):
    vectors, _ = np.linalg.qr(generator.standard_normal((rows, columns)))
    return vectors


class OuterDIISTests(unittest.TestCase):
    @staticmethod
    def extrapolation(scale: float):
        generator = np.random.default_rng(7)
        base = _orthonormal(30, 4, generator)
        steps = [generator.standard_normal(base.shape) for _ in range(3)]
        sets = []
        for step, factor in enumerate((0.0, 1.0, 1.3)):
            rotated, _ = np.linalg.qr(base + scale * factor * steps[step])
            sets.append(
                OutputSet(step, [{None: OccupiedBlock(rotated, np.ones(4))}], {} if step == 0 else {step - 1: 1.0})
            )
        diis = OuterDIIS(enabled=True, tolerance=1e-8)
        for entry in sets:
            diis.append(entry)
        coefficients = {1: -0.5, 2: 1.5}
        natural = diis.natural_orbitals(coefficients)
        projector = [entry.channels[0][None].vectors @ entry.channels[0][None].vectors.T for entry in sets]
        target = sum(c * projector[k] for k, c in coefficients.items())
        return diis, natural, target, projector, sets

    def test_natural_orbitals_reproduce_the_extrapolated_density_matrix(self) -> None:
        errors = []
        for scale in (0.004, 0.002):
            diis, natural, target, projector, sets = self.extrapolation(scale)
            self.assertIsNotNone(natural, diis.last_note)
            phi, occupations = natural[0][None]
            np.testing.assert_allclose(phi.T @ phi, np.eye(4), atol=1e-12)
            self.assertAlmostEqual(float(occupations.sum()), 4.0, places=12)
            error = np.linalg.norm(phi @ np.diag(occupations) @ phi.T - target)
            # The n_occ largest natural occupations: what is left is second order
            # in the changes, the extrapolation itself first order.
            self.assertLess(error, 0.1 * np.linalg.norm(target - projector[2]))
            errors.append(error)
        self.assertGreater(errors[0] / errors[1], 3.0)
        # The traces are the Frobenius products of the density matrices.
        self.assertAlmostEqual(diis.traces[(1, 2)], float(np.sum(projector[1] * projector[2])), places=12)

    def test_gate_fractional_states_rises_and_fast_loops(self) -> None:
        """No copies before the loop is judged slow; a fractional state, a rise, a fast loop or the end free the window."""

        from types import SimpleNamespace

        from parsec_python.SCF.fock_diis import WINDOW

        generator = np.random.default_rng(1)
        vectors = _orthonormal(20, 2, generator)
        block = {None: OccupiedBlock(vectors, np.ones(2))}
        ones = [np.ones(2)]
        criteria = InnerCriterion(adaptive=False, final=1e-6, tolerance=1e-10)
        diis = OuterDIIS(enabled=True, tolerance=1e-10)

        def output(step, source, error, energy, extrapolated=False):
            return OutputSet(step, [block], source, error, energy, extrapolated=extrapolated)

        self.assertFalse(diis.observe(OutputSet(0, [block], {}), ones))  # the start: nothing known
        self.assertFalse(diis.observe(output(1, {0: 1.0}, 1e-4, -1.0), ones))  # one dexx, no ratio
        self.assertFalse(diis.observe(output(2, {1: 1.0}, 1e-6, -1.1), ones))  # ratio 0.01: fast
        self.assertEqual(diis.window, [])
        self.assertTrue(diis.observe(output(3, {2: 1.0}, 3e-7, -1.2), ones))  # ratio 0.3: the window starts
        self.assertTrue(diis.observe(output(4, {3: 1.0}, 1e-7, -1.3), ones))
        self.assertIsNone(diis.proposal(criteria))
        self.assertIn("2 of the 3", diis.last_note)
        self.assertFalse(diis.observe(output(5, {4: 1.0}, 5e-8, -1.4), [np.array([0.5, 0.5])]))
        self.assertEqual(diis.window, [])  # a fractional state frees the window
        self.assertIn("fractional", diis.last_note)
        for step, error in ((6, 3e-8), (7, 2e-8), (8, 1.5e-8)):
            self.assertTrue(diis.observe(output(step, {step - 1: 1.0}, error, -1.4 - 0.1 * step), ones))
        self.assertEqual([entry.step for entry in diis.window], [6, 7, 8])  # three outputs: a proposal may follow
        self.assertFalse(diis.observe(output(9, {8: 1.0}, 2e-8, -2.4), ones))
        self.assertEqual(diis.window, [])  # dexx rose
        self.assertIn("dexx rose", diis.last_note)
        for step, error in ((10, 1.2e-8), (11, 1.0e-8), (12, 0.9e-8)):
            self.assertTrue(diis.observe(output(step, {step - 1: 1.0}, error, -2.4 - 0.1 * step), ones))
        self.assertFalse(diis.observe(output(13, {11: 0.5, 12: 0.5}, 0.95e-8, -3.7, True), ones))
        # A rejected extrapolation is taken back: the window is the output it started from.
        self.assertEqual(diis.steps_rejected, 1)
        self.assertEqual([entry.step for entry in diis.window], [12])
        self.assertEqual(diis.rollback, 12)
        self.assertIn("back to the output of loop 12", diis.last_note)
        natural = diis.natural_orbitals({12: 1.0})
        np.testing.assert_allclose(natural[0][None][0] @ natural[0][None][0].T, vectors @ vectors.T, atol=1e-12)
        # The loop after it is a plain step from loop 12.
        self.assertTrue(diis.observe(output(14, {12: 1.0}, 0.8e-8, -3.8), ones))
        self.assertAlmostEqual(diis.plain_ratio, 0.8 / 0.9)
        # At most WINDOW outputs, the oldest dropped before the next copy.
        for step in range(15, 15 + WINDOW + 2):
            diis.make_room()
            self.assertLess(len(diis.window), WINDOW)
            self.assertTrue(diis.observe(output(step, {step - 1: 1.0}, 0.9 ** step * 1e-8, -4.0 - step), ones))
        self.assertEqual(len(diis.window), WINDOW)
        # The end in sight: the next loop expected below the tolerance, or the last allowed one.
        last = 15 + WINDOW + 2
        soon = SimpleNamespace(predicted_error=lambda: 1e-12)
        self.assertFalse(diis.observe(output(last, {last - 1: 1.0}, 1e-9, -9.0), ones, soon))
        self.assertEqual(diis.window, [])
        self.assertIn("may end", diis.last_note)
        self.assertTrue(diis.observe(output(last + 1, {last: 1.0}, 0.9e-9, -9.1), ones))
        self.assertFalse(diis.observe(output(last + 2, {last + 1: 1.0}, 0.8e-9, -9.2), ones, final_next=True))
        self.assertEqual(diis.window, [])

    def test_window_that_does_not_fit_stops_diis(self) -> None:
        """A copy or a build that does not fit the devices stops DIIS with a note (plain steps)."""

        from types import SimpleNamespace

        from parsec_python.SCF.hybrid import _diis_operator, _diis_sets, _observe

        class Crowded:
            @staticmethod
            def occupied_blocks(sets):
                return None

        class Untouched:
            @staticmethod
            def occupied_blocks(sets):
                raise AssertionError("no copy before the gate is open")

        diis = OuterDIIS(enabled=True, tolerance=1e-8)
        sets = (SimpleNamespace(occupations=np.ones(2)),)
        _observe(diis, Untouched(), 0, sets, {})
        _observe(diis, Untouched(), 1, sets, {0: 1.0}, 1e-4, -1.0)
        _observe(diis, Untouched(), 2, sets, {1: 1.0}, 1e-6, -1.1)  # ratio 0.01: fast
        self.assertTrue(diis.enabled)
        _observe(diis, Crowded(), 3, sets, {2: 1.0}, 5e-7, -1.2)  # ratio 0.5: slow, the copy fails
        self.assertFalse(diis.enabled)
        self.assertIn("does not fit", diis.last_note)
        self.assertEqual(diis.window, [])

        class Full:
            @staticmethod
            def natural_sets(sets, natural):
                raise MemoryError("out of memory")

            @staticmethod
            def exchange(sets, columns="occupied", known=None):
                raise MemoryError("out of memory")

            @staticmethod
            def drop_orbitals(sets, result=None):
                pass

        generator = np.random.default_rng(3)
        block = {None: OccupiedBlock(_orthonormal(10, 2, generator), np.ones(2))}
        for build in (
            lambda state: _diis_sets(Full(), state, {0: 1.0}, sets),
            lambda state: _diis_operator(Full(), state, sets, 0.25),
        ):
            state = OuterDIIS(enabled=True, tolerance=1e-8)
            state.append(OutputSet(0, [block], {}))
            self.assertIsNone(build(state))
            self.assertFalse(state.enabled)
            self.assertIn("do not fit the device memory", state.last_note.replace("does not", "do not"))
            self.assertEqual(state.window, [])

    def test_complement_is_orthonormal_and_orthogonal(self) -> None:
        generator = np.random.default_rng(2)
        phi = _orthonormal(25, 3, generator)
        empty = generator.standard_normal((25, 4))
        rest = orthonormal_complement(phi, empty)
        np.testing.assert_allclose(phi.T @ rest, 0.0, atol=1e-12)
        np.testing.assert_allclose(rest.T @ rest, np.eye(4), atol=1e-12)


class FockLoopControlTests(unittest.TestCase):
    """The reference Fock loop of H4 (small grid) with every control against the fixed loop."""

    BASE = """Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 6
Solver_Lpole: 4
Hartree_Atomic_Tail: off
States_Num: 4
Fermi_Temp: 80
Max_Iter: 80
Convergence_Criterion: 1.0e-6 Ry
Eigensolver: chebff
Mixing_Method: Anderson
Mixing_Param: 0.3
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  -2.1 0.0 0.0
  -0.7 0.0 0.0
   0.7 0.0 0.0
   2.1 0.0 0.3
end Atom_Coord
Correlation_Type: hf
Exx_Tolerance: 1.0e-9 Ry
{extra}"""

    def run_case(self, extra: str):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.SCF import prepare_single_point, run_scf

        directory = Path(tempfile.mkdtemp(prefix="parsec_fock_controls_"))
        try:
            library = Path(__file__).resolve().parents[3] / "pp_generation" / "NCPPs" / "PBE" / "H" / "PARSEC"
            shutil.copy(library / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            (directory / "parsec.in").write_text(self.BASE.format(extra=extra))
            records = []
            result = run_scf(prepare_single_point(parse_parsec_input(directory / "parsec.in").problem), callback=records.append)
            return result, [record for record in records if isinstance(record, ExchangeLoopIteration)]
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_every_control_reaches_the_state_of_the_fixed_loop(self) -> None:
        fixed, fixed_outer = self.run_case(
            "Exx_Inner_Criterion: fixed\nExx_Mixer_History: restart\nExx_Outer_Mixing: none\n"
        )
        fast, fast_outer = self.run_case(
            "Exx_Inner_Criterion: adaptive\nExx_Mixer_History: carry\nExx_Outer_Mixing: diis\n"
        )
        self.assertTrue(fixed.converged and fast.converged)
        self.assertAlmostEqual(fast.energies.total, fixed.energies.total, delta=2.0e-9)
        self.assertLess(fast_outer[-1].estimated_error, 1.0e-9)
        self.assertIsNone(fast_outer[-1].inner_criterion)
        # No outer iteration after the first exact loop of a plain operator below the
        # tolerance (an iteration of level passes does not make its loop loose).
        for before, record in zip(fast_outer[1:-1], fast_outer[2:-1]):
            self.assertFalse(
                record.estimated_error < 1.0e-9 and record.inner_criterion is None and not before.extrapolated,
                record,
            )
        # The final loop ran at Convergence_Criterion and its operator was a plain one.
        self.assertFalse(fast_outer[-2].extrapolated)
        self.assertLess(fast.history[-1].weighted_residual, 1.0e-6)
        self.assertTrue(any(record.inner_criterion is not None for record in fast_outer[1:]))
        self.assertTrue(any(record.mixer_seeds for record in fast_outer[1:]))
        self.assertLess(
            sum(record.inner_iterations for record in fast_outer[1:]),
            sum(record.inner_iterations for record in fixed_outer[1:]),
        )

    def test_a_rejected_step_is_taken_back(self) -> None:
        """Every extrapolated step rejected (its energy made higher): the loop goes back and ends as the fixed one."""

        from unittest import mock

        from parsec_python.SCF import fock_diis

        original = fock_diis.OuterDIIS.judge

        def judge(diis, output, occupations, criteria=None, final_next=False):
            if output.extrapolated and output.energy is not None:
                output = replace(output, energy=output.energy + 1.0)
            return original(diis, output, occupations, criteria, final_next)

        fixed, _fixed_outer = self.run_case(
            "Exx_Inner_Criterion: fixed\nExx_Mixer_History: restart\nExx_Outer_Mixing: none\n"
        )
        with mock.patch.object(fock_diis.OuterDIIS, "judge", judge):
            result, outer = self.run_case("")
        notes = [record.outer_note for record in outer]
        self.assertTrue(any("back to the output of loop" in note for note in notes), notes)
        back = next(index for index, note in enumerate(notes) if "back to the output of loop" in note)
        # The operator after the rejected loop is not extrapolated, the loop after it is not the last one.
        self.assertFalse(outer[back].extrapolated)
        self.assertTrue(result.converged)
        self.assertGreater(outer[-1].iteration, back + 1)
        self.assertAlmostEqual(result.energies.total, fixed.energies.total, delta=2.0e-9)

    def test_the_last_allowed_loop_has_a_plain_operator(self) -> None:
        """No extrapolated operator for the loop at Exx_Max_Iter (it could not end the Fock loop)."""

        # The window starts at loop 2 (the first dexx ratio), the first extrapolation
        # follows loop 4: with Exx_Max_Iter 5 loop 5 is the last allowed one.
        saved = self.BASE
        try:
            self.BASE = saved.replace("Exx_Tolerance: 1.0e-9 Ry", "Exx_Tolerance: 5.2e-8 Ry")
            _free, free_outer = self.run_case("")
            _result, outer = self.run_case("Exx_Max_Iter: 5\n")
        finally:
            self.BASE = saved
        self.assertTrue(free_outer[4].extrapolated)
        self.assertEqual(outer[-1].iteration, 5)
        self.assertFalse(outer[4].extrapolated)
        # The exact loops end with their levels converged at their last Hamiltonian.
        self.assertGreater(outer[-1].level_passes, 0)

    def test_keywords(self) -> None:
        from parsec_python.Input import parse_parsec_input
        from parsec_python.Input.complete_input import COMPLETE_INPUT_NAME
        from parsec_python.Input.parsec_input import ParsecInputError

        directory = Path(tempfile.mkdtemp(prefix="parsec_fock_keywords_"))
        try:
            library = Path(__file__).resolve().parents[3] / "pp_generation" / "NCPPs" / "PBE" / "H" / "PARSEC"
            shutil.copy(library / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            for extra, expected in (
                ("", ExactExchangeSettings()),
                (
                    "Exx_Inner_Criterion: Fixed\nExx_Mixer_History: carry\nExx_Outer_Mixing: none\n",
                    ExactExchangeSettings(inner_criterion="fixed", mixer_history="carry", outer_mixing="none"),
                ),
                (
                    "Exx_Inner_Criterion: adaptive\nExx_Mixer_History: restart\nExx_Outer_Mixing: diis\n",
                    ExactExchangeSettings(inner_criterion="adaptive", mixer_history="restart", outer_mixing="diis"),
                ),
            ):
                with self.subTest(extra=extra):
                    path = directory / "parsec.in"
                    path.write_text(self.BASE.format(extra=extra + "Generate_Complete_Parsec_In: .true.\n"))
                    translation = parse_parsec_input(path)
                    self.assertEqual(translation.problem.scf.exact_exchange, replace(expected, tolerance=1.0e-9))
                    # bak_complete_parsec.in reads back to the same settings.
                    again = directory / COMPLETE_INPUT_NAME
                    again.write_text(translation.complete_input)
                    self.assertEqual(
                        parse_parsec_input(again).problem.scf.exact_exchange,
                        translation.problem.scf.exact_exchange,
                    )
            path.write_text(self.BASE.format(extra="Exx_Outer_Mixing: broyden\n"))
            with self.assertRaisesRegex(ParsecInputError, "Exx_Outer_Mixing"):
                parse_parsec_input(path)
            path.write_text(self.BASE.replace("Correlation_Type: hf", "Correlation_Type: pbe").replace(
                "Exx_Tolerance: 1.0e-9 Ry\n", "").format(extra="Exx_Inner_Criterion: adaptive\n"))
            with self.assertRaisesRegex(ParsecInputError, "hybrid Correlation_Type"):
                parse_parsec_input(path)
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        settings = ExactExchangeSettings(inner_criterion="fixed", mixer_history="restart", outer_mixing="none")
        self.assertIn("ExactExchangeSettings(", repr(settings))
        self.assertEqual(repr(ExactExchangeSettings()), "ExactExchangeSettings()")
        with self.assertRaisesRegex(ValueError, "Exx_Outer_Mixing"):
            ExactExchangeSettings(outer_mixing="broyden")
        with self.assertRaisesRegex(ValueError, "Exx_Inner_Criterion"):
            ExactExchangeSettings(inner_criterion="loose")
        with self.assertRaisesRegex(ValueError, "Exx_Mixer_History"):
            ExactExchangeSettings(mixer_history="keep")
        del ParsecInputError


if __name__ == "__main__":
    unittest.main()
