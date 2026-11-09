"""The short-Lanczos upper bound of the Chebyshev filters (``Bounds_Method``) on every filter.

:func:`parsec_python.Eigensolvers.spectral_bounds.lanczos_bound_from_ritz` is the one bound formula of the
reference and device filters: ``parsec`` (the default) the earlier PARSEC's radius ``max(theta_max, |theta_min|)``,
``lanczos`` PARSEC 1.5's ``theta_max``.  A spectrum far below 0 (a bulk cell with a large negative mean
potential) gets ``u ~ |theta_min|`` from the radius, an interval several times wider than the spectrum; 1.5's
bound moves with the spectrum.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.Eigensolvers.chebdav import ChebDavSettings
from parsec_python.Eigensolvers.chebff import ChebFFSettings
from parsec_python.Eigensolvers.spectral_bounds import (
    bound_request,
    lanczos_bound_from_ritz,
    lanczos_bound_parsec15,
    lanczos_bound_steps,
    lanczos_upper_bound,
)
from parsec_python.Eigensolvers.subspace import SubspaceSettings
from parsec_python.acceleration.Eigensolvers.chebff_spinor import _lanczos_upper_bound as spinor_bound
from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy


def _chain(offset: float, n: int = 400) -> tuple[sp.csr_matrix, np.ndarray]:
    """A 1-D chain whose spectrum spans about [offset, offset + 100] Ry (kinetic 0..4.4, a ramp of 100)."""

    kinetic = sp.diags((-np.ones(n - 1), 2.2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1)).tocsr()
    potential = offset + np.linspace(0.0, 100.0, n)
    return kinetic, potential


class _Dense:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __matmul__(self, other):
        return self.matrix @ other

    def apply(self, other):
        return self.matrix @ other


class HostTests(unittest.TestCase):
    def test_the_parsec_rule_is_the_earlier_formula(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            ritz = np.sort(rng.normal(scale=50.0, size=6))
            beta = float(10.0 ** rng.uniform(-3, 2))
            bound = lanczos_bound_from_ritz("parsec", ritz, beta, steps=6, requested_steps=5, breakdown=False)
            radius = max(ritz[-1], abs(ritz[0]))
            margin = beta * (10.0 if beta < 1e-2 else 5.0 if beta < 1e-1 else 1.0)
            self.assertEqual(bound.upper_bound, radius + margin)
            self.assertEqual(bound.middle, 0.5 * (radius + ritz[0]))
            self.assertEqual(bound.lower_bound, ritz[0])

    def test_the_lanczos_rule_is_theta_max_and_moves_with_the_spectrum(self) -> None:
        kinetic, potential = _chain(0.0)
        matrix = (kinetic + sp.diags(potential)).toarray()
        start = np.random.default_rng(4).random(matrix.shape[0])
        for offset in (0.0, -600.0):
            shifted = matrix + offset * np.eye(matrix.shape[0])
            reference = lanczos_bound_parsec15(shifted, start, steps=10)
            with self.subTest(offset=offset):
                self.assertEqual(reference.upper_bound,
                                 lanczos_bound_from_ritz("lanczos", reference.ritz_values, reference.raw_beta,
                                                         steps=10, requested_steps=10,
                                                         breakdown=False).upper_bound)
        plain = lanczos_bound_parsec15(matrix, start, steps=10)
        moved = lanczos_bound_parsec15(matrix - 600.0 * np.eye(matrix.shape[0]), start, steps=10)
        self.assertAlmostEqual(moved.upper_bound, plain.upper_bound - 600.0, delta=1e-8 * 600.0)
        radius = lanczos_upper_bound(matrix - 600.0 * np.eye(matrix.shape[0]), start, steps=8)
        self.assertGreater(radius.upper_bound, 500.0)  # |theta_min| ~ 600 sets the radius
        self.assertLess(moved.upper_bound, -400.0)  # the true top is about -496

    def test_steps_and_requests(self) -> None:
        self.assertEqual(lanczos_bound_steps("parsec", 5, 1000), 5)
        self.assertEqual(lanczos_bound_steps("parsec", 12, 1000), 8)
        self.assertEqual(lanczos_bound_steps("parsec", 2, 1000), 4)
        self.assertEqual(lanczos_bound_steps("lanczos", 12, 1000), 12)
        self.assertEqual(lanczos_bound_steps("lanczos", 12, 7), 7)
        with self.assertRaises(ValueError):
            lanczos_bound_steps("static", 5, 10)
        self.assertEqual(bound_request(SubspaceSettings()), ("parsec", 5))
        self.assertEqual(bound_request(SubspaceSettings(bounds="lanczos", bound_steps=10)), ("lanczos", 10))
        self.assertEqual(bound_request(ChebFFSettings(lanczos_steps=10, bounds="lanczos", bound_steps=12)), ("lanczos", 12))
        self.assertEqual(bound_request(ChebDavSettings(bounds="lanczos")), ("lanczos", ChebDavSettings().lanczos_steps))

    def test_the_spinor_bound_takes_the_rule(self) -> None:
        rng = np.random.default_rng(9)
        raw = rng.normal(size=(60, 60)) + 1j * rng.normal(size=(60, 60))
        matrix = 0.05 * (raw + raw.conj().T) + np.diag(np.linspace(0.0, 100.0, 60))
        parsec = spinor_bound(np, _Dense(matrix - 600.0 * np.eye(60)), 60, 5, np.random.default_rng(3))
        plain = spinor_bound(np, _Dense(matrix), 60, 10, np.random.default_rng(3), rule="lanczos")
        moved = spinor_bound(np, _Dense(matrix - 600.0 * np.eye(60)), 60, 10, np.random.default_rng(3), rule="lanczos")
        self.assertGreater(parsec[0], 500.0)
        self.assertAlmostEqual(moved[0], plain[0] - 600.0, delta=1e-7)
        self.assertGreaterEqual(moved[0], float(np.linalg.eigvalsh(matrix)[-1]) - 600.0)
        upper, lower, middle = spinor_bound(np, _Dense(matrix), 60, 10, np.random.default_rng(3), rule="lanczos",
                                            with_middle=True)
        self.assertEqual((upper, lower), plain)


@unittest.skipUnless(cupy_available(), "CUDA required")
class DeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cp, _ = require_cupy()
        self.cp.cuda.Device(0).use()

    def _operator(self, offset: float):
        kinetic, potential = _chain(offset)
        n = kinetic.shape[0]
        projectors = (sp.csr_matrix(np.random.default_rng(8).normal(size=(n, 2)) * 0.002), np.array([1.0, -1.0]))
        operator = CuPyHamiltonian(kinetic, potential, projectors, retain_generic_laplacian=False)
        p, signs = projectors
        dense = kinetic.toarray() + np.diag(potential) + (p.toarray() * signs) @ p.toarray().T
        return operator, dense

    def test_the_device_bound_is_the_reference_bound(self) -> None:
        from parsec_python.acceleration.Eigensolvers.spectral_bounds import lanczos_upper_bound as device_bound

        operator, dense = self._operator(-600.0)
        start = np.random.default_rng(5).random(dense.shape[0])
        for rule, reference in (
            ("parsec", lanczos_upper_bound(dense, start, steps=8)),
            ("lanczos", lanczos_bound_parsec15(dense, start, steps=10)),
        ):
            bound = device_bound(operator, self.cp.asarray(start), 8 if rule == "parsec" else 10, rule=rule)
            with self.subTest(rule=rule):
                self.assertAlmostEqual(bound.upper_bound, reference.upper_bound, delta=1e-8 * abs(reference.upper_bound))
                self.assertAlmostEqual(bound.middle, reference.middle, delta=1e-8 * abs(reference.middle))
                self.assertEqual(bound.steps, reference.steps)

    def _errors(self, offset: float, bounds: str, passes: int = 12):
        from parsec_python.acceleration.Eigensolvers.subspace import DeviceSubspaceState, run_subspace_filter

        cp = self.cp
        operator, dense = self._operator(offset)
        exact = np.linalg.eigvalsh(dense)
        rng = np.random.default_rng(17)
        x = cp.array(np.linalg.qr(rng.normal(size=(dense.shape[0], 24)))[0], order="F")
        state = DeviceSubspaceState(dense.shape[0], 24, cp.asarray(np.sort(rng.uniform(exact[0], exact[40], 24))), x)
        settings = SubspaceSettings(polynomial_degree=8, degree_delta=2, lower_bound_rule="smart", bounds=bounds,
                                    bound_steps=None if bounds == "parsec" else 10)
        with patch.dict(os.environ, PARSEC_CUPY_DISTRIBUTED_FILTER="0", PARSEC_CUPY_MIXED_FILTER="off"):
            for _ in range(passes):
                state = run_subspace_filter(operator, state, settings=settings, compute_residuals=False).state
        found = np.sort(cp.asnumpy(state.eigenvalues))[:16]
        return float(np.max(np.abs(found - exact[:16])))

    def test_the_lanczos_bound_filters_a_shifted_spectrum_as_an_unshifted_one(self) -> None:
        # The radius at -600 Ry spans [r_max, ~600]: 12 passes leave 1e-4 of the 1e-13 of the unshifted case.
        unshifted = self._errors(0.0, "lanczos")
        shifted = self._errors(-600.0, "lanczos")
        radius = self._errors(-600.0, "parsec")
        self.assertLess(unshifted, 1.0e-9)
        self.assertLess(shifted, 1.0e-9)
        self.assertGreater(radius, 1.0e3 * shifted)


if __name__ == "__main__":
    unittest.main()
