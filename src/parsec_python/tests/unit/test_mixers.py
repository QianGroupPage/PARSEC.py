"""The generalized Broyden and multisecant mixers of PARSEC.

``genbro.f90`` and ``msecant1-3.f90`` against dense implementations of the
equations they state (an explicit inverse Jacobian), the secant conditions of
those updates, ``qinv.f90`` against a line-by-line port and NumPy, the
inner-product weights of a symmetry wedge, the factory, the settings and the
parser keywords.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy.linalg import lapack

from parsec_python.Input.parsec_input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Mixer import (
    AndersonMixer,
    GeneralizedBroydenMixer,
    MultisecantMixer,
    make_mixer,
    quasi_inverse,
)
from parsec_python.Output.parsec_output import _mixer_lines
from parsec_python.models import MixingSettings

DATA = Path(__file__).resolve().parents[1] / "data"


def _fortran_qinv(matrix: np.ndarray) -> np.ndarray:
    """A loop-by-loop port of ``qinv.f90`` on LAPACK's ``dgeqp3``."""

    a = np.array(matrix, dtype=np.float64, order="F")
    n = a.shape[0]
    qr, jpvt, tau, _work, info = lapack.dgeqp3(a)
    assert info == 0
    tol = abs(qr[0, 0]) * float(np.float32(2.2204e-16))
    r = 0
    while r < n and abs(qr[r, r]) > tol:
        r += 1
    q = np.eye(n)
    for j in range(r - 1, -1, -1):
        u = np.zeros(n)
        u[j] = 1.0
        u[j + 1 :] = qr[j + 1 :, j]
        v = u[j:] @ q[j:, j:]
        u[j:] *= tau[j]
        q[j:, j:] -= np.outer(u[j:], v)
    rinv = np.zeros((n, n))
    for j in range(r - 1, -1, -1):
        rinv[j, j] = 1.0 / qr[j, j]
        for i in range(j - 1, -1, -1):
            rinv[i, j] = -sum(qr[i, k] * rinv[k, j] for k in range(i + 1, r)) / qr[i, i]
    inverse = np.zeros((n, n))
    for i in range(r):
        for k in range(i, r):
            inverse[jpvt[i] - 1, :] += rinv[i, k] * q[:, k]
    return inverse


def _map(size: int, seed: int = 3):
    """A smooth nonlinear map ``x -> x_out`` with a fixed point near zero."""

    generator = np.random.default_rng(seed)
    matrix = 0.4 * generator.standard_normal((size, size)) / np.sqrt(size)
    offset = 0.3 * generator.standard_normal(size)

    def output(x: np.ndarray) -> np.ndarray:
        return np.tanh(matrix @ x) + offset + 0.2 * np.sin(x)

    return output


def _trajectory(mixer, output, size: int, iterations: int):
    """Run ``mixer`` and return its inputs, outputs and mixed vectors."""

    inputs, outputs, mixed = [], [], []
    x = np.zeros(size)
    for iteration in range(1, iterations + 1):
        y = output(x)
        inputs.append(x)
        outputs.append(y)
        x = mixer.mix(x, y, iteration=iteration)
        mixed.append(x)
    return inputs, outputs, mixed


def _dense_broyden(settings: MixingSettings, inputs, outputs):
    """The equations of the ``genbro.f90`` header with an explicit ``B``.

    Returns the mixed vectors and, per step, ``B`` with the pairs
    ``(dx, df)`` of its window.
    """

    alpha, memory = settings.parameter, settings.memory
    size = inputs[0].size
    results, records = [], []
    for iteration, (x, y) in enumerate(zip(inputs, outputs), start=1):
        f = y - x
        if (iteration - 1) % settings.restart == 0:
            inverse_jacobian = -alpha * np.eye(size)
            window_dx, window_df = [], []
        else:
            dx = x - previous_x
            df = f - previous_f
            window_dx = (window_dx + [dx])[-memory:]
            window_df = (window_df + [df])[-memory:]
            gram = np.array([[a @ b for b in window_df] for a in window_df])
            v = np.linalg.inv(gram)[-1] @ np.array(window_df)
            u = dx - inverse_jacobian @ df
            inverse_jacobian = inverse_jacobian + np.outer(u, v)
        records.append((inverse_jacobian.copy(), list(window_dx), list(window_df)))
        results.append(x - inverse_jacobian @ f)
        previous_x, previous_f = x, f
    return results, records


def _dense_multisecant(settings: MixingSettings, inputs, outputs):
    """Fang and Saad's multisecant updates with an explicit ``G``.

    No restart (``restart_factor`` must be 0).  Returns the mixed vectors
    and, per step, ``G`` with the raw pairs of its open group.
    """

    assert settings.restart_factor == 0.0
    kind = {"msecant1": 1, "msecant2": 2, "msecant3": 3}[settings.method]
    group = settings.group_size
    if kind == 3 and group == 0:
        kind = 1 if settings.preferred_type == 1 else 2
    alpha = settings.parameter
    size = inputs[0].size
    completed = -alpha * np.eye(size)
    open_dx, open_df = [], []
    previous_group = None
    update_type = None
    stage = settings.en_like
    results, records = [], []
    for iteration, (x, y) in enumerate(zip(inputs, outputs), start=1):
        f = y - x
        if iteration == 1:
            base = (x, f)
            results.append(x + alpha * f)
            records.append((completed.copy(), [], []))
            stage = 2 if stage == 1 else stage
            continue
        if stage != 1:
            open_dx.append(x - base[0])
            open_df.append(f - base[1])
        if not open_dx:
            current = completed
        else:
            dx = np.array(open_dx).T
            df = np.array(open_df).T
            res = dx.shape[1]
            n = completed.T @ dx
            if kind == 3:
                if stage != 1 and previous_group is not None:
                    previous_dx, previous_df = previous_group
                    lhs = np.sum((df.T @ previous_df[:, -res:]) ** 2) * np.sum((n.T @ df) ** 2)
                    rhs = np.sum((df.T @ df) ** 2) * np.sum((dx.T @ previous_dx[:, -res:]) ** 2)
                    update_type = 1 if lhs > rhs else 2 if lhs < rhs else settings.preferred_type
                elif previous_group is None:
                    update_type = settings.preferred_type
                update = update_type
            else:
                update = kind
            e = dx - completed @ df
            if update == 1:
                current = completed + e @ np.linalg.inv(n.T @ df) @ n.T
            else:
                current = completed + e @ np.linalg.inv(df.T @ df) @ df.T
        records.append((current.copy(), list(open_dx), list(open_df)))
        results.append(x - current @ f)
        if open_dx and group and len(open_dx) == group:
            completed = current
            previous_group = (np.array(open_dx).T, np.array(open_df).T)
            open_dx, open_df = [], []
        if stage != 2:
            base = (x, f)
        stage = {1: 2, 2: 1}.get(stage, stage)
    return results, records


class QuasiInverseTests(unittest.TestCase):
    def test_full_rank_matrix_gives_the_inverse(self) -> None:
        generator = np.random.default_rng(1)
        for size in (1, 2, 5, 9):
            matrix = generator.standard_normal((size, size))
            np.testing.assert_allclose(
                quasi_inverse(matrix), np.linalg.inv(matrix), rtol=1e-10, atol=1e-11
            )
            np.testing.assert_allclose(
                quasi_inverse(matrix), _fortran_qinv(matrix), rtol=1e-12, atol=1e-13
            )

    def test_rank_deficient_matrix_gives_the_basic_solution(self) -> None:
        generator = np.random.default_rng(2)
        size, rank = 6, 3
        # A product of rank-3 factors is singular only to rounding: |R_44| is
        # about |R_11| * 1.5e-16, and whether that passes qinv.f90's
        # |R_11| * 2.2204e-16 depends on the LAPACK build (an A100 node's
        # finds rank 4).  The port follows Fortran's dgeqp3 either way.
        near = generator.standard_normal((size, rank)) @ generator.standard_normal((rank, size))
        np.testing.assert_allclose(
            quasi_inverse(near), _fortran_qinv(near), rtol=1e-10, atol=1e-10
        )
        # Exact rank 3 for the properties: the Householder steps keep the
        # zero rows exactly zero, so R_44 = R_55 = R_66 = 0 on any machine,
        # and the null space is not spanned by unit vectors.
        matrix = np.vstack(
            [generator.standard_normal((rank, size)), np.zeros((size - rank, size))]
        )
        inverse = quasi_inverse(matrix)
        np.testing.assert_allclose(inverse, _fortran_qinv(matrix), rtol=1e-10, atol=1e-10)
        # A X A = A, A X is the orthogonal projector on the range of A (the
        # least-squares residual of NumPy), and X b has only rank entries.
        np.testing.assert_allclose(matrix @ inverse @ matrix, matrix, atol=1e-10)
        rhs = generator.standard_normal(size)
        least_squares = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        np.testing.assert_allclose(matrix @ (inverse @ rhs), matrix @ least_squares, atol=1e-10)
        self.assertEqual(np.count_nonzero(np.abs(inverse @ rhs) > 1e-12), rank)
        self.assertFalse(np.allclose(inverse, np.linalg.pinv(matrix)))

    def test_zero_and_empty_matrices(self) -> None:
        np.testing.assert_array_equal(quasi_inverse(np.zeros((3, 3))), np.zeros((3, 3)))
        self.assertEqual(quasi_inverse(np.zeros((0, 0))).shape, (0, 0))
        with self.assertRaises(ValueError):
            quasi_inverse(np.zeros((2, 3)))


class GeneralizedBroydenTests(unittest.TestCase):
    def test_memory_zero_is_linear_mixing(self) -> None:
        settings = MixingSettings(method="broyden", memory=0, parameter=0.4)
        mixer = GeneralizedBroydenMixer(settings)
        x, y = np.array([1.0, -2.0, 0.5]), np.array([0.0, 1.0, 1.5])
        for iteration in (1, 2, 7):
            np.testing.assert_allclose(mixer.mix(x, y, iteration=iteration), 0.6 * x + 0.4 * y)

    def test_first_step_and_restart_are_linear_steps(self) -> None:
        settings = MixingSettings(method="broyden", restart=3)
        output = _map(7)
        mixer = GeneralizedBroydenMixer(settings)
        inputs, outputs, mixed = _trajectory(mixer, output, 7, 7)
        for index in (0, 3, 6):
            np.testing.assert_allclose(
                mixed[index], inputs[index] + 0.3 * (outputs[index] - inputs[index])
            )

    def test_matches_the_dense_equations_and_the_secant_conditions(self) -> None:
        size = 9
        output = _map(size)
        for memory, restart in ((1, 20), (2, 20), (4, 20), (3, 5)):
            settings = MixingSettings(method="broyden", memory=memory, restart=restart, parameter=0.35)
            mixer = GeneralizedBroydenMixer(settings)
            inputs, outputs, mixed = _trajectory(mixer, output, size, 11)
            expected, records = _dense_broyden(settings, inputs, outputs)
            for got, want in zip(mixed, expected):
                np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-11)
            # B_n df_j = dx_j for the last Memory_Param differences.
            for inverse_jacobian, window_dx, window_df in records:
                for dx, df in zip(window_dx, window_df):
                    np.testing.assert_allclose(inverse_jacobian @ df, dx, atol=1e-9)
            self.assertEqual(mixer.singular_warnings, 0)

    def test_converges_on_a_linear_problem(self) -> None:
        generator = np.random.default_rng(4)
        size = 12
        jacobian = np.eye(size) * 0.5 + 0.1 * generator.standard_normal((size, size))
        offset = generator.standard_normal(size)
        mixer = GeneralizedBroydenMixer(MixingSettings(method="broyden", memory=4))
        x = np.zeros(size)
        for iteration in range(1, 41):
            residual = jacobian @ x + offset - x
            if np.linalg.norm(residual) < 1e-11:
                break
            x = mixer.mix(x, x + residual, iteration=iteration)
        self.assertLess(np.linalg.norm(jacobian @ x + offset - x), 1e-11)

    def test_history_must_start_at_a_restart(self) -> None:
        mixer = GeneralizedBroydenMixer(MixingSettings(method="broyden"))
        with self.assertRaises(ValueError):
            mixer.mix(np.zeros(2), np.ones(2), iteration=2)


class MultisecantTests(unittest.TestCase):
    def _check(self, **options) -> list:
        # More unknowns than secant pairs: every T of the checks is regular.
        size = 14
        output = _map(size, seed=5)
        settings = MixingSettings(restart_factor=0.0, parameter=0.3, **options)
        mixer = MultisecantMixer(settings)
        inputs, outputs, mixed = _trajectory(mixer, output, size, 12)
        expected, records = _dense_multisecant(settings, inputs, outputs)
        for iteration, (got, want) in enumerate(zip(mixed, expected), start=1):
            np.testing.assert_allclose(
                got, want, rtol=1e-8, atol=1e-10, err_msg=f"{options} iteration {iteration}"
            )
        self.assertEqual(mixer.restarts, 0)
        return records

    def test_type_one_and_type_two_match_the_dense_updates(self) -> None:
        for method in ("msecant1", "msecant2"):
            for group_size in (1, 2, 3, 0):
                records = self._check(method=method, group_size=group_size)
                # G+ DF = DX for the pairs of the open group.
                for jacobian, open_dx, open_df in records:
                    for dx, df in zip(open_dx, open_df):
                        np.testing.assert_allclose(jacobian @ df, dx, atol=1e-8)

    def test_hybrid_matches_the_dense_choice_of_update(self) -> None:
        for group_size in (1, 2, 3):
            for preferred in (1, 2):
                self._check(method="msecant3", group_size=group_size, preferred_type=preferred)
        # Without groups msecant3 is the preferred method alone.
        for preferred, alone in ((1, "msecant1"), (2, "msecant2")):
            size = 6
            output = _map(size, seed=8)
            hybrid = MultisecantMixer(
                MixingSettings(method="msecant3", group_size=0, preferred_type=preferred)
            )
            single = MultisecantMixer(MixingSettings(method=alone, group_size=0))
            for got, want in zip(
                _trajectory(hybrid, output, size, 9)[2], _trajectory(single, output, size, 9)[2]
            ):
                np.testing.assert_array_equal(got, want)

    def test_en_like_stages_match_the_dense_updates(self) -> None:
        for method in ("msecant1", "msecant2", "msecant3"):
            for group_size in (1, 2):
                self._check(method=method, group_size=group_size, en_like=1)
        self._check(method="msecant1", group_size=0, en_like=1)

    def test_group_size_one_type_one_is_broyden_good_method(self) -> None:
        # One pair per group: G+ = G + (dx - G df) dx^T G / (dx^T G df).
        size = 5
        output = _map(size, seed=9)
        settings = MixingSettings(method="msecant1", restart_factor=0.0)
        inputs, outputs, mixed = _trajectory(MultisecantMixer(settings), output, size, 8)
        jacobian = -0.3 * np.eye(size)
        for index in range(1, 8):
            dx = inputs[index] - inputs[index - 1]
            df = (outputs[index] - inputs[index]) - (outputs[index - 1] - inputs[index - 1])
            jacobian = jacobian + np.outer(dx - jacobian @ df, dx @ jacobian) / (dx @ jacobian @ df)
            np.testing.assert_allclose(
                mixed[index], inputs[index] - jacobian @ (outputs[index] - inputs[index]), rtol=1e-8, atol=1e-10
            )

    def test_restart_on_residual_growth_returns_to_the_previous_iterate(self) -> None:
        settings = MixingSettings(method="msecant1", restart_factor=0.3)
        mixer = MultisecantMixer(settings)
        x1, x2, x3 = np.array([0.0, 0.0]), np.array([0.3, 0.1]), np.array([0.5, 0.2])
        mixer.mix(x1, x1 + np.array([1.0, 0.0]), iteration=1)
        mixer.mix(x2, x2 + np.array([0.5, 0.1]), iteration=2)
        f2 = np.array([0.5, 0.1])
        # ||f_2|| = 0.51 < 0.3 ||f_3||: restart from x_2 with a linear step.
        mixed = mixer.mix(x3, x3 + np.array([3.0, 0.0]), iteration=3)
        np.testing.assert_allclose(mixed, x2 + 0.3 * f2)
        self.assertEqual(mixer.restarts, 1)
        # A restart factor of zero never restarts.
        never = MultisecantMixer(MixingSettings(method="msecant1", restart_factor=0.0))
        never.mix(x1, x1 + np.array([1.0, 0.0]), iteration=1)
        never.mix(x2, x2 + f2, iteration=2)
        never.mix(x3, x3 + np.array([3.0, 0.0]), iteration=3)
        self.assertEqual(never.restarts, 0)

    def test_history_grows_past_memory_param(self) -> None:
        settings = MixingSettings(method="msecant2", group_size=0, memory=0, restart_factor=0.0)
        mixer = MultisecantMixer(settings)
        size = 12
        output = _map(size, seed=11)
        inputs, outputs, mixed = _trajectory(mixer, output, size, 9)
        self.assertGreaterEqual(mixer._capacity, 9)
        expected, _ = _dense_multisecant(settings, inputs, outputs)
        for got, want in zip(mixed, expected):
            np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)

    def test_iteration_one_starts_a_new_history(self) -> None:
        settings = MixingSettings(method="msecant1")
        output = _map(4, seed=12)
        mixer = MultisecantMixer(settings)
        first = _trajectory(mixer, output, 4, 5)[2]
        again = _trajectory(mixer, output, 4, 5)[2]
        for got, want in zip(again, first):
            np.testing.assert_array_equal(got, want)

    def test_converges_on_the_nonlinear_map(self) -> None:
        size = 15
        output = _map(size, seed=13)
        for options in (
            {"method": "msecant1"},
            {"method": "msecant2"},
            {"method": "msecant3", "group_size": 2},
            {"method": "msecant1", "group_size": 0},
            {"method": "msecant1", "en_like": 1},
        ):
            mixer = MultisecantMixer(MixingSettings(**options))
            x = np.zeros(size)
            for iteration in range(1, 61):
                y = output(x)
                if np.linalg.norm(y - x) < 1e-10:
                    break
                x = mixer.mix(x, y, iteration=iteration)
            self.assertLess(np.linalg.norm(output(x) - x), 1e-10, msg=str(options))


class WeightTests(unittest.TestCase):
    """A wedge with orbit multiplicities as weights is the full vector."""

    def test_weighted_wedge_equals_full_vector(self) -> None:
        generator = np.random.default_rng(21)
        wedge_size = 14
        multiplicities = generator.choice(np.array([1, 2, 4, 8]), size=wedge_size)
        expand = np.repeat(np.arange(wedge_size), multiplicities)
        output = _map(wedge_size, seed=22)
        for options in (
            {"method": "broyden"},
            {"method": "broyden", "memory": 2, "restart": 4},
            {"method": "msecant1"},
            {"method": "msecant2", "group_size": 0},
            {"method": "msecant3", "group_size": 2},
            {"method": "msecant1", "en_like": 1},
        ):
            settings = MixingSettings(**options)
            wedge = make_mixer(settings, weights=multiplicities.astype(float))
            inputs, outputs, mixed = _trajectory(wedge, output, wedge_size, 12)
            # The full mixer sees the same iterates, expanded: a trajectory
            # of its own would add the growth of roundoff along the path.
            full = make_mixer(settings)
            for iteration, (x, y, want) in enumerate(zip(inputs, outputs, mixed), start=1):
                got = full.mix(x[expand], y[expand], iteration=iteration)
                np.testing.assert_allclose(
                    got, want[expand], rtol=1e-10, atol=1e-12, err_msg=f"{options} {iteration}"
                )

    def test_uniform_weight_changes_nothing(self) -> None:
        output = _map(6, seed=23)
        for method in ("broyden", "msecant1", "msecant2", "msecant3"):
            settings = MixingSettings(method=method)
            plain = _trajectory(make_mixer(settings), output, 6, 10)[2]
            scaled = _trajectory(make_mixer(settings, weights=np.full(6, 4.0)), output, 6, 10)[2]
            for got, want in zip(scaled, plain):
                np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_weights_must_fit(self) -> None:
        with self.assertRaises(ValueError):
            make_mixer(MixingSettings(method="broyden"), weights=np.array([1.0, -1.0]))
        mixer = make_mixer(MixingSettings(method="msecant1"), weights=np.ones(3))
        with self.assertRaises(ValueError):
            mixer.mix(np.zeros(4), np.ones(4), iteration=1)


class FactoryAndSettingsTests(unittest.TestCase):
    def test_default_settings_give_the_anderson_mixer_unchanged(self) -> None:
        settings = MixingSettings()
        mixer = make_mixer(settings)
        self.assertIs(type(mixer), AndersonMixer)
        output = _map(10, seed=31)
        got = _trajectory(mixer, output, 10, 12)[2]
        want = _trajectory(AndersonMixer(settings), output, 10, 12)[2]
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)
        self.assertIs(type(make_mixer()), AndersonMixer)
        with self.assertRaises(ValueError):
            make_mixer(settings, weights=np.ones(3))

    def test_each_method_has_its_mixer(self) -> None:
        self.assertIsInstance(make_mixer(MixingSettings(method="broyden")), GeneralizedBroydenMixer)
        for method in ("msecant1", "msecant2", "msecant3"):
            self.assertIsInstance(make_mixer(MixingSettings(method=method)), MultisecantMixer)

    def test_repr_of_default_fields_is_unchanged(self) -> None:
        self.assertEqual(
            repr(MixingSettings()),
            "MixingSettings(parameter=0.3, memory=4, restart=20, regularization=0.0, "
            "safeguard=False, step_limit=2.0, growth_trigger=2.0, backoff=0.5)",
        )
        self.assertEqual(
            repr(MixingSettings(parameter=0.15, safeguard=True)),
            "MixingSettings(parameter=0.15, memory=4, restart=20, regularization=0.0, "
            "safeguard=True, step_limit=2.0, growth_trigger=2.0, backoff=0.5)",
        )
        self.assertTrue(
            repr(MixingSettings(method="msecant3", group_size=0, preferred_type=2)).endswith(
                "backoff=0.5, method='msecant3', group_size=0, preferred_type=2)"
            )
        )

    def test_validation_keeps_anderson_and_relaxes_the_others(self) -> None:
        for bad in ({"parameter": 1.5}, {"parameter": 0.0}, {"memory": 0}, {"restart": 0}):
            with self.assertRaises(ValueError):
                MixingSettings(**bad)
        # PARSEC checks none of these for the other mixers.
        MixingSettings(method="broyden", parameter=1.5, memory=0)
        MixingSettings(method="msecant1", parameter=-0.2, memory=0, restart=0, restart_factor=0.0)
        MixingSettings(method="msecant2", en_like=2, preferred_type=5, expand_factor=0.5)
        for bad in (
            {"method": "pulay"},
            {"method": "broyden", "restart": 0},
            {"method": "broyden", "memory": -1},
            {"method": "broyden", "safeguard": True},
            {"method": "msecant1", "regularization": 0.1},
            {"method": "msecant1", "group_size": -1},
            {"method": "msecant1", "parameter": float("nan")},
            {"method": "msecant1", "en_like": 0.5},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                MixingSettings(**bad)
        self.assertEqual(MixingSettings(method=" MSecant2 ").method, "msecant2")


class MixerInputAndOutputTests(unittest.TestCase):
    _INPUT = (
        "Boundary_Sphere_Radius: 4.0 bohr\nGrid_Spacing: 0.8 bohr\n"
        "Coordinate_Unit: Cartesian_Bohr\nExpansion_Order: 4\nStates_Num: 3\n"
        "Eigensolver: chebff\n{mixer}\nAtom_Types_Num: 1\nAtom_Type: H\n"
        "Local_Component: s\nbegin Atom_Coord\n 0.0 0.0 0.0\nend Atom_Coord\n"
    )

    def _parse(self, mixer: str):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
            path = directory / "parsec.in"
            path.write_text(self._INPUT.format(mixer=mixer), encoding="utf-8")
            return parse_parsec_input(path)

    def test_methods_and_keywords_with_parsec_defaults(self) -> None:
        mixing = self._parse("Mixing_Method: Anderson").problem.mixing
        self.assertEqual(mixing, MixingSettings())
        mixing = self._parse("Mixing_Method: broyden").problem.mixing
        self.assertEqual(mixing, MixingSettings(method="broyden"))
        translation = self._parse(
            "Mixing_Method: msecant3\nMixing_Group_Size: 0\nMixing_EN_Like: 1\n"
            "Mixing_Restart_Factor: 0.1\nMixing_Memory_Expand_Factor: 1.5\n"
            "Mixing_Preferred_Type: 2\nMemory_Param: 6\nMixing_Param: 0.2"
        )
        self.assertEqual(
            translation.problem.mixing,
            MixingSettings(
                method="msecant3", group_size=0, en_like=1, restart_factor=0.1,
                expand_factor=1.5, preferred_type=2, memory=6, parameter=0.2,
            ),
        )
        self.assertFalse(any("Mixing_" in warning for warning in translation.warnings))

    def test_inert_keywords_and_the_broyden_block_warn(self) -> None:
        warnings = self._parse("Mixing_Method: anderson\nMixing_Group_Size: 2").warnings
        self.assertTrue(any(w.startswith("Mixing_Group_Size has no effect") for w in warnings))
        warnings = self._parse("Mixing_Method: msecant1\nRestart_Mixing: 10").warnings
        self.assertTrue(any(w.startswith("Restart_Mixing has no effect") for w in warnings))
        warnings = self._parse("Mixing_Method: broyden\nRestart_Mixing: 30").warnings
        self.assertTrue(any("exceeds the Broyden block" in w for w in warnings))
        warnings = self._parse("Mixing_Method: broyden\nRestart_Mixing: 20").warnings
        self.assertFalse(any("Broyden block" in w for w in warnings))

    def test_unknown_mixer_and_safeguard_of_another_mixer_are_refused(self) -> None:
        with self.assertRaises(ParsecInputError):
            self._parse("Mixing_Method: pulay")
        with self.assertRaises(UnsupportedParsecOptionError):
            self._parse("Mixing_Method: msecant2\nMixing_Safeguard: true")
        with self.assertRaises(ParsecInputError):
            self._parse("Mixing_Method: broyden\nRestart_Mixing: 0")

    def test_header_lines_are_those_of_parsec(self) -> None:
        self.assertEqual(
            _mixer_lines(MixingSettings())[:3],
            [
                " Anderson mixer",
                " Initial Jacobian:  0.300  Mixing memory is  4",
                " Mixing restarted after           20 iterations",
            ],
        )
        self.assertEqual(
            _mixer_lines(MixingSettings(method="broyden")),
            [
                "  Generalized Broyden mixer",
                " Initial Jacobian:  0.300  Mixing memory is  4",
                " Mixing restarted after           20 iterations",
            ],
        )
        self.assertEqual(
            _mixer_lines(MixingSettings(method="msecant1")),
            [
                " Multisecant method, Broyden-like Type-I update",
                " Mixing parameter:  0.300  Group size of secant equations:  1",
                " Restarting factor is:  0.300",
            ],
        )
        self.assertEqual(
            _mixer_lines(MixingSettings(method="msecant2", group_size=0, en_like=1))[:2],
            [
                " Multisecant method, EN-like Type-II update",
                " Mixing parameter:  0.300  All secant equations taken into account in one group",
            ],
        )
        self.assertEqual(
            _mixer_lines(MixingSettings(method="msecant3", group_size=2)),
            [
                " Multisecant method, Broyden-like hybrid update",
                " Preferred type of update is Type-I",
                " Mixing parameter:  0.300  Group size of secant equations:  2",
                " Restarting factor is:  0.300",
            ],
        )


if __name__ == "__main__":
    unittest.main()
