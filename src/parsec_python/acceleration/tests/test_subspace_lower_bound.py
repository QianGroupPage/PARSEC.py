"""The lower bound of the later-SCF SUBSPACE filter (``Smart_Subspace``) on every filter.

:func:`parsec_python.Eigensolvers.subspace.subspace_filter_bounds` is the one rule of the reference filter, the
device filter of one GPU, of a distributed basis and of the complex k-point solver: at ``monotone`` (the default)
it is the earlier PARSEC's ``max(r_max, 0)`` / ``max(l_p, r_max)``, at ``smart``/``adaptive`` PARSEC 1.5's rules.
A spectrum far below 0 (the eigenvalues of a bulk cell, whose periodic local potential carries a large negative
mean) keeps the monotone bound at 0, so the filter damps nothing between the subspace and 0; the smart bound
follows the subspace.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers.spectral_bounds import lanczos_bound_parsec15
from parsec_python.Eigensolvers.subspace import (
    SubspaceSettings,
    SubspaceState,
    _next_filter_lower_bound,
    parsec15_lower_bound,
    subspace_filter_bounds,
)
from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy


def _matrix(offset: float, n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((n, n))
    return 0.05 * (a + a.T) + np.diag(np.linspace(0.0, 8.0, n)) + offset * np.eye(n)


class HelperTests(unittest.TestCase):
    def test_monotone_is_the_earlier_rule_and_the_lanczos_bound(self) -> None:
        bound = lanczos_bound_parsec15(_matrix(0.0), steps=8, rng=np.random.default_rng(0))
        for values in (np.array([-3.0, -2.0, -1.5]), np.array([-1.0, 0.5, 2.0])):
            for first, previous in ((True, None), (False, 0.7), (False, -2.0), (False, 3.0)):
                state = SubspaceState(operator_dimension=3, working_states=3, eigenvalues=values, vectors=np.eye(3),
                                      filter_lower_bound=previous, first_filter=first)
                with self.subTest(values=values.tolist(), first=first, previous=previous):
                    self.assertEqual(
                        subspace_filter_bounds("monotone", float(values.max()), previous, first, bound),
                        (_next_filter_lower_bound(state), float(bound.upper_bound)),
                    )

    def test_parsec15_rules_are_those_of_parsec15_lower_bound(self) -> None:
        bound = lanczos_bound_parsec15(_matrix(0.0), steps=8, rng=np.random.default_rng(0))
        for values in (np.array([-40.0, -38.0, -36.5]), np.array([-1.0, 0.5, 2.0]), np.array([9.0, 10.0, 99.0])):
            for first, previous in ((True, None), (True, 0.5), (False, 1.0), (False, -37.0), (False, 4.0)):
                state = SubspaceState(operator_dimension=3, working_states=3, eigenvalues=values, vectors=np.eye(3),
                                      filter_lower_bound=previous, first_filter=first)
                for rule in ("smart", "adaptive"):
                    with self.subTest(values=values.tolist(), first=first, previous=previous, rule=rule):
                        self.assertEqual(
                            subspace_filter_bounds(rule, float(values.max()), previous, first, bound),
                            parsec15_lower_bound(state, rule, bound),
                        )

    def test_smart_follows_a_negative_spectrum(self) -> None:
        bound = lanczos_bound_parsec15(_matrix(-40.0), steps=8, rng=np.random.default_rng(0))
        self.assertEqual(subspace_filter_bounds("monotone", -36.5, None, True, bound)[0], 0.0)
        self.assertEqual(subspace_filter_bounds("monotone", -36.4, 0.0, False, bound)[0], 0.0)
        self.assertEqual(subspace_filter_bounds("smart", -36.5, None, True, bound)[0], -36.5)
        self.assertEqual(subspace_filter_bounds("smart", -36.6, -36.5, False, bound)[0], -36.6)
        with self.assertRaises(ValueError):
            subspace_filter_bounds("zero", -36.5, None, True, bound)


@unittest.skipUnless(cupy_available(), "CUDA required")
class DeviceFilterTests(unittest.TestCase):
    """The device filter of one GPU on a 1-D chain whose spectrum spans [-38, 62] Ry (or [2, 102])."""

    def setUp(self) -> None:
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()

    def _operator(self, offset: float):
        n = 257
        kinetic = sp.diags((-np.ones(n - 1), 2.2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1))
        # A ramp: about 0.4 Ry between neighbouring levels over 100 Ry, from offset up.
        potential = offset + np.linspace(0.0, 100.0, n)
        projectors = (sp.csr_matrix(np.random.default_rng(8).normal(size=(n, 2)) * 0.002), np.array([1.0, -1.0]))
        operator = CuPyHamiltonian(kinetic, potential, projectors, retain_generic_laplacian=False)
        dense = kinetic.toarray() + np.diag(potential)
        p, signs = projectors
        dense = dense + (p.toarray() * signs) @ p.toarray().T
        return operator, np.linalg.eigvalsh(dense)

    def _errors(self, offset: float, rule: str, passes: int = 12):
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, run_subspace_filter

        cp = self.cp
        operator, exact = self._operator(offset)
        rng = np.random.default_rng(17)
        x = cp.array(np.linalg.qr(rng.normal(size=(257, 24)))[0], order="F")
        values = cp.asarray(np.sort(rng.uniform(exact[0], exact[30], 24)))
        state = DeviceSubspaceState(257, 24, values, x)
        settings = SubspaceSettings(polynomial_degree=8, degree_delta=2, lower_bound_rule=rule)
        lowers = []
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_FILTER="0", PARSEC_CUPY_MIXED_FILTER="off"):
            for _ in range(passes):
                result = run_subspace_filter(operator, state, settings=settings, compute_residuals=False)
                state = result.state
                lowers.append(float(state.filter_lower_bound))
        found = np.sort(cp.asnumpy(state.eigenvalues))[:16]
        return float(np.max(np.abs(found - exact[:16]))), lowers

    def test_the_smart_bound_converges_a_spectrum_far_below_zero(self) -> None:
        # 12 passes from a random subspace: 2e-5 Ry (monotone, l = 0 throughout) against 2e-13 Ry (smart)
        monotone, monotone_lowers = self._errors(-40.0, "monotone")
        smart, smart_lowers = self._errors(-40.0, "smart")
        self.assertEqual(set(monotone_lowers), {0.0})
        self.assertLess(smart_lowers[-1], -25.0)
        self.assertLess(smart, 1.0e-10)
        self.assertGreater(monotone, 1.0e4 * smart)

    def test_the_smart_bound_is_no_worse_above_zero(self) -> None:
        # the monotone bound cannot come down from a high first Ritz value (58 Ry here)
        monotone, monotone_lowers = self._errors(0.0, "monotone")
        smart, _ = self._errors(0.0, "smart")
        self.assertGreater(min(monotone_lowers), 0.0)
        self.assertLess(smart, 1.0e-10)
        self.assertLessEqual(smart, monotone)


if __name__ == "__main__":
    unittest.main()
