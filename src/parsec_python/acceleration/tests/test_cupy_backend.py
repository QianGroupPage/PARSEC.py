"""Parity and optional-runtime tests for the CuPy acceleration slice."""

from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from parsec_python.acceleration.backends.cupy import (
    CuPyHamiltonian,
    CuPyUnavailableError,
    cupy_available,
    require_cupy,
)
from parsec_python.acceleration.Eigensolvers import (
    CuPyEigvalSolver,
    CuPySymmetryOrbitals,
    chebyshev_filter as gpu_chebyshev_filter,
    GeneralizedRitzStabilityError,
    generalized_rayleigh_ritz,
    generalized_ritz_requested,
    lanczos_upper_bound as gpu_lanczos_upper_bound,
    orthonormalize as gpu_orthonormalize,
    rayleigh_ritz as gpu_rayleigh_ritz,
    run_chebdav,
)
from parsec_python.acceleration.Eigensolvers.orthogonalize import (
    _complete_subspace_policy,
    chebdav_block_orth_requested,
    orthonormalize_complete_appended_subspace,
    orthonormalize_complete_subspace,
    orthonormalize_appended_block,
)
from parsec_python.acceleration.Eigensolvers.chebyshev import (
    _mixed_filter_requested,
)
from parsec_python.Eigensolvers import (
    ChebFFSettings,
    EigvalSettings,
    SubspaceSettings,
    chebyshev_filter,
    lanczos_upper_bound,
    orthonormalize,
    rayleigh_ritz,
    solve_eigval,
)
from parsec_python.Hamiltonian import KohnShamHamiltonian
from parsec_python.V_ion import NonlocalProjectorOperator


GPU_AVAILABLE = cupy_available()


def _small_hamiltonian():
    dimension = 24
    kinetic = sp.diags(
        (
            -np.ones(dimension - 1),
            2.4 * np.ones(dimension),
            -np.ones(dimension - 1),
        ),
        (-1, 0, 1),
        format="csr",
    )
    potential = np.linspace(-0.75, 0.35, dimension)
    projectors = np.zeros((dimension, 2), dtype=np.float64)
    projectors[3:7, 0] = (0.2, -0.35, 0.15, 0.1)
    projectors[15:20, 1] = (-0.1, 0.25, 0.3, -0.2, 0.05)
    nonlocal_operator = NonlocalProjectorOperator(
        projectors=sp.csc_matrix(projectors),
        signs=np.array((1.0, -1.0)),
        labels=((0, 0, 0), (1, 0, 0)),
    )
    reference = KohnShamHamiltonian(kinetic, potential, nonlocal_operator)
    return kinetic, potential, nonlocal_operator, reference


class TestCuPyOptionalImport(unittest.TestCase):
    def test_chebdav_block_orth_policy_is_large_basis_only_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_CUPY_CHEBDAV_BLOCK_ORTH", None)
            os.environ.pop("PARSEC_CUPY_CHEBDAV_BLOCK_ORTH_MIN_ROWS", None)
            self.assertFalse(chebdav_block_orth_requested(99_999, 6))
            self.assertTrue(chebdav_block_orth_requested(100_000, 6))
            self.assertFalse(chebdav_block_orth_requested(360_000, 1))
            os.environ["PARSEC_CUPY_CHEBDAV_BLOCK_ORTH"] = "off"
            self.assertFalse(chebdav_block_orth_requested(360_000, 6))
            os.environ["PARSEC_CUPY_CHEBDAV_BLOCK_ORTH"] = "on"
            self.assertTrue(chebdav_block_orth_requested(24, 6))

    def test_mixed_filter_auto_policy_requires_enough_complete_basis_work(self):
        operator = SimpleNamespace(
            shape=(100, 100), mixed_precision_recurrence=object()
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ["PARSEC_CUPY_MIXED_FILTER"] = "auto"
            os.environ["PARSEC_CUPY_MIXED_FILTER_MIN_WORK"] = "1000"
            self.assertFalse(_mixed_filter_requested(operator, 3))
            self.assertTrue(_mixed_filter_requested(operator, 4))
            os.environ["PARSEC_CUPY_MIXED_FILTER"] = "on"
            self.assertTrue(_mixed_filter_requested(operator, 1))
            os.environ["PARSEC_CUPY_MIXED_FILTER"] = "off"
            self.assertFalse(_mixed_filter_requested(operator, 100))

    def test_complete_subspace_policy_is_size_adaptive_and_overridable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION", None)
            os.environ.pop("PARSEC_CUPY_SUBSPACE_QR", None)
            os.environ.pop("PARSEC_CUPY_SUBSPACE_QR_WORK_THRESHOLD", None)
            self.assertEqual(_complete_subspace_policy(20_000, 10), "mgs")
            self.assertEqual(_complete_subspace_policy(360_000, 80), "qr")
            self.assertEqual(_complete_subspace_policy(700_000, 600), "tsqr")
            os.environ["PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION"] = "mgs"
            self.assertEqual(_complete_subspace_policy(360_000, 80), "mgs")

    def test_generalized_ritz_policy_is_large_basis_only_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARSEC_CUPY_GENERALIZED_RITZ", None)
            os.environ.pop("PARSEC_CUPY_GENERALIZED_RITZ_WORK_THRESHOLD", None)
            os.environ.pop("PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION", None)
            self.assertFalse(generalized_ritz_requested(20_000, 10))
            self.assertTrue(generalized_ritz_requested(360_000, 80))
            os.environ["PARSEC_CUPY_GENERALIZED_RITZ"] = "off"
            self.assertFalse(generalized_ritz_requested(360_000, 80))

    def test_modules_import_without_eager_cupy_dependency(self):
        # Importing this test module already imported both public modules.  On
        # a CPU-only environment that must not raise ModuleNotFoundError.
        self.assertIsInstance(GPU_AVAILABLE, bool)

    def test_unavailable_runtime_has_specific_error(self):
        if GPU_AVAILABLE:
            self.skipTest("CuPy and CUDA are available")
        with self.assertRaises(CuPyUnavailableError):
            require_cupy()

    def test_chebdav_reports_an_unavailable_cuda_runtime(self):
        if GPU_AVAILABLE:
            self.skipTest("CuPy and CUDA are available")
        with self.assertRaises(CuPyUnavailableError):
            run_chebdav(np.eye(2), 1)
        chebdav_settings = EigvalSettings(
            safety_buffer=0,
            initial_method="chebdav",
        )
        with self.assertRaises(CuPyUnavailableError):
            CuPyEigvalSolver(
                sp.eye(2, format="csr"),
                np.zeros(2),
                settings=chebdav_settings,
            )


@unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA are not available")
class TestCuPyComponentParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp, _ = require_cupy()
        cls.kinetic, cls.potential, cls.nonlocal_operator, cls.reference = (
            _small_hamiltonian()
        )

    def setUp(self):
        # A local-potential update is intentionally mutable.  Give every test
        # a fresh tiny operator so test ordering cannot leak that update.
        self.gpu = CuPyHamiltonian(
            self.kinetic, self.potential, self.nonlocal_operator
        )

    def test_lazy_symmetry_orbitals_materialize_signed_full_vectors(self):
        wedge = np.asarray(
            ((0.1, -0.2), (0.3, 0.4), (-0.5, 0.6)), dtype=np.float64
        )
        mapping = np.asarray((0, 1, 2, 0, 1, 2), dtype=np.int64)
        phases = np.asarray(
            (
                (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
            ),
            dtype=np.float64,
        )
        orbitals = CuPySymmetryOrbitals(
            scaled_wedge_vectors=self.cp.asarray(wedge),
            representations=np.asarray((0, 1), dtype=np.int32),
            full_to_wedge=mapping,
            device_full_to_wedge=self.cp.asarray(mapping),
            phases=self.cp.asarray(phases),
            full_size=mapping.size,
        )

        actual = self.cp.asnumpy(orbitals.to_full_device())
        expected = wedge[mapping].copy()
        expected[:, 1] *= phases[1]
        np.testing.assert_array_equal(actual, expected)

    def test_hamiltonian_terms_vector_and_block(self):
        rng = np.random.default_rng(8)
        for vectors in (
            rng.standard_normal(self.kinetic.shape[0]),
            rng.standard_normal((self.kinetic.shape[0], 4)),
        ):
            for reference_action, gpu_action in (
                (self.reference.apply_kinetic, self.gpu.apply_kinetic),
                (self.reference.apply_local, self.gpu.apply_local),
                (self.reference.apply_nonlocal, self.gpu.apply_nonlocal),
                (self.reference.apply, self.gpu.apply),
            ):
                expected = reference_action(vectors)
                actual = self.cp.asnumpy(gpu_action(vectors))
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    def test_hamiltonian_apply_into_reuses_vector_and_block_outputs(self):
        rng = np.random.default_rng(81)
        for vectors in (
            rng.standard_normal(self.kinetic.shape[0]),
            rng.standard_normal((self.kinetic.shape[0], 4)),
        ):
            device_vectors = self.cp.asarray(vectors)
            output = self.cp.empty_like(device_vectors)
            returned = self.gpu.apply_into(device_vectors, output)
            self.assertIs(returned, output)
            np.testing.assert_allclose(
                self.cp.asnumpy(output),
                self.reference.apply(vectors),
                rtol=2e-12,
                atol=2e-12,
            )

    def test_generalized_ritz_matches_orthonormal_ritz_and_reuses_workspace(self):
        rng = np.random.default_rng(82)
        vectors = self.cp.asarray(
            rng.standard_normal((self.kinetic.shape[0], 5))
        )
        conventional_basis = gpu_orthonormalize(vectors).basis
        conventional = gpu_rayleigh_ritz(
            self.gpu, conventional_basis, compute_residuals=True
        )
        workspace = self.cp.empty(
            vectors.shape, dtype=self.cp.float64, order="F"
        )
        generalized = generalized_rayleigh_ritz(
            self.gpu,
            vectors,
            workspace=workspace,
            compute_residuals=True,
        )

        self.assertIs(generalized.workspace, workspace)
        np.testing.assert_allclose(
            self.cp.asnumpy(generalized.eigenvalues),
            self.cp.asnumpy(conventional.eigenvalues),
            rtol=2e-11,
            atol=2e-11,
        )
        overlap = self.cp.asnumpy(
            conventional.wavefunctions.T @ generalized.wavefunctions
        )
        np.testing.assert_allclose(
            np.abs(overlap), np.eye(5), rtol=2e-10, atol=2e-10
        )
        np.testing.assert_allclose(
            self.cp.asnumpy(generalized.residual_norms),
            self.cp.asnumpy(conventional.residual_norms),
            rtol=2e-9,
            atol=2e-11,
        )

    def test_tsqr_is_orthonormal_and_preserves_the_input_span(self):
        rng = np.random.default_rng(1207)
        host = rng.standard_normal((4096, 32))
        expected, _ = np.linalg.qr(host, mode="reduced")
        with patch.dict(
            os.environ,
            {
                "PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION": "tsqr",
                "PARSEC_CUPY_TSQR_TILE_BYTES": str(256 * 1024),
            },
            clear=False,
        ):
            result = orthonormalize_complete_subspace(
                self.cp.asarray(host, dtype=self.cp.float64)
            )
        actual = self.cp.asnumpy(result.basis)
        self.assertEqual(result.algorithm, "householder_tsqr")
        np.testing.assert_allclose(
            actual.T @ actual, np.eye(32), rtol=2e-12, atol=2e-12
        )
        # Bases may differ by a dense orthogonal rotation.  Equality of the
        # two projection operators is the sign/rotation-independent span test.
        probe = rng.standard_normal((4096, 5))
        np.testing.assert_allclose(
            actual @ (actual.T @ probe),
            expected @ (expected.T @ probe),
            rtol=2e-11,
            atol=2e-11,
        )

    def test_complete_appended_subspace_preserves_locked_prefix(self):
        rng = np.random.default_rng(1211)
        locked, _ = np.linalg.qr(rng.standard_normal((4096, 7)), mode="reduced")
        appended = rng.standard_normal((4096, 32))
        projected = appended - locked @ (locked.T @ appended)
        expected, _ = np.linalg.qr(projected, mode="reduced")
        device = self.cp.asarray(np.column_stack((locked, appended)))
        with patch.dict(
            os.environ,
            {
                "PARSEC_CUPY_SUBSPACE_ORTHOGONALIZATION": "tsqr",
                "PARSEC_CUPY_TSQR_TILE_BYTES": str(256 * 1024),
                "PARSEC_CUPY_PREFIX_PROJECTION_TILE_BYTES": str(256 * 1024),
            },
            clear=False,
        ):
            result = orthonormalize_complete_appended_subspace(
                device,
                existing_columns=7,
                active_columns=39,
            )
        actual = self.cp.asnumpy(result.basis)
        np.testing.assert_allclose(actual[:, :7], locked, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            actual[:, 7:].T @ actual[:, 7:],
            np.eye(32),
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            locked.T @ actual[:, 7:], 0.0, rtol=0.0, atol=2e-12
        )
        probe = rng.standard_normal((4096, 5))
        np.testing.assert_allclose(
            actual[:, 7:] @ (actual[:, 7:].T @ probe),
            expected @ (expected.T @ probe),
            rtol=2e-11,
            atol=2e-11,
        )

    def test_generalized_ritz_rejects_dependent_filtered_basis(self):
        vector = np.linspace(-1.0, 1.0, self.kinetic.shape[0])
        dependent = self.cp.asarray(np.column_stack((vector, vector)))
        with self.assertRaises(GeneralizedRitzStabilityError):
            generalized_rayleigh_ritz(self.gpu, dependent)

    def test_local_update_preserves_static_device_objects(self):
        laplacian = self.gpu.negative_laplacian
        projectors = self.gpu.projectors
        transpose = self.gpu.projectors_transpose
        updated = self.potential + 0.07 * np.sin(np.arange(self.potential.size))
        self.gpu.update_local_potential(updated)
        self.assertIs(self.gpu.negative_laplacian, laplacian)
        self.assertIs(self.gpu.projectors, projectors)
        self.assertIs(self.gpu.projectors_transpose, transpose)
        probe = np.linspace(-1.0, 1.0, self.potential.size)
        expected = KohnShamHamiltonian(
            self.kinetic, updated, self.nonlocal_operator
        ).apply(probe)
        np.testing.assert_allclose(
            self.cp.asnumpy(self.gpu @ probe), expected, rtol=2e-12, atol=2e-12
        )

    def test_zero_projector_operator(self):
        gpu = CuPyHamiltonian(self.kinetic, self.potential)
        probe = np.arange(self.kinetic.shape[0] * 3, dtype=float).reshape(-1, 3)
        expected = self.kinetic @ probe + self.potential[:, None] * probe
        np.testing.assert_allclose(
            self.cp.asnumpy(gpu @ probe), expected, rtol=2e-12, atol=2e-12
        )
        np.testing.assert_array_equal(
            self.cp.asnumpy(gpu.apply_nonlocal(probe)), np.zeros_like(probe)
        )

    def test_full_grid_long_projectors_select_parallel_reduction(self):
        """Full-grid operators must pass row support to the auto policy.

        Symmetry-sector operators already upload raw projector factors, whose
        transpose constructor measures every projector row.  The modular
        full-grid path keeps public cuSPARSE matrices too; this regression
        prevents that different storage path from silently treating an
        unknown row length as zero and selecting the serial CUDA kernel.
        """

        dimension = 320
        kinetic = sp.eye(dimension, format="csr")
        projectors = sp.csc_matrix(
            np.linspace(0.1, 1.0, dimension, dtype=np.float64)[:, None]
        )
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=projectors,
            signs=np.ones(1, dtype=np.float64),
            labels=((0, 0, 0),),
        )
        gpu = CuPyHamiltonian(
            kinetic,
            np.zeros(dimension, dtype=np.float64),
            nonlocal_operator,
        )

        projection = gpu.custom_projector_projection
        self.assertIsNotNone(projection)
        self.assertEqual(projection.max_row_entries, dimension)
        self.assertEqual(projection.reduction_mode, "parallel")

    def test_lanczos_filter_orthogonalize_and_rayleigh_ritz(self):
        rng = np.random.default_rng(15)
        initial = rng.random(self.kinetic.shape[0])
        reference_operator = self.reference.as_linear_operator()
        reference_bound = lanczos_upper_bound(
            reference_operator, initial_vector=initial, steps=5
        )
        gpu_bound = gpu_lanczos_upper_bound(
            self.gpu, initial_vector=initial, steps=5
        )
        self.assertAlmostEqual(gpu_bound.lower_bound, reference_bound.lower_bound, 10)
        self.assertAlmostEqual(gpu_bound.upper_bound, reference_bound.upper_bound, 10)

        vectors = rng.standard_normal((self.kinetic.shape[0], 4))
        expected_filter = chebyshev_filter(
            reference_operator,
            vectors,
            degree=5,
            lower_bound=reference_bound.middle,
            upper_bound=reference_bound.upper_bound,
            reference_eigenvalue=reference_bound.lower_bound,
        )
        actual_filter = gpu_chebyshev_filter(
            self.gpu,
            vectors,
            degree=5,
            lower_bound=reference_bound.middle,
            upper_bound=reference_bound.upper_bound,
            reference_eigenvalue=reference_bound.lower_bound,
        )
        np.testing.assert_allclose(
            self.cp.asnumpy(actual_filter), expected_filter, rtol=5e-11, atol=5e-11
        )

        expected_basis = orthonormalize(expected_filter).basis
        actual_basis = gpu_orthonormalize(actual_filter).basis
        np.testing.assert_allclose(
            np.abs(self.cp.asnumpy(actual_basis.T @ actual_basis)),
            np.eye(4),
            rtol=3e-11,
            atol=3e-11,
        )
        # QR-like bases may differ in roundoff/sign. Compare their projectors.
        actual_host = self.cp.asnumpy(actual_basis)
        np.testing.assert_allclose(
            actual_host @ actual_host.T,
            expected_basis @ expected_basis.T,
            rtol=2e-9,
            atol=2e-9,
        )

        expected_ritz = rayleigh_ritz(reference_operator, expected_basis)
        actual_ritz = gpu_rayleigh_ritz(self.gpu, actual_basis)
        np.testing.assert_allclose(
            self.cp.asnumpy(actual_ritz.eigenvalues),
            expected_ritz.eigenvalues,
            rtol=2e-9,
            atol=2e-9,
        )

    def test_device_lanczos_rejects_a_zero_initial_vector(self):
        with self.assertRaisesRegex(ValueError, "must be nonzero"):
            gpu_lanczos_upper_bound(
                self.gpu,
                initial_vector=np.zeros(self.kinetic.shape[0]),
                steps=5,
            )

    def test_speculative_mgs_falls_back_to_literal_replacement_path(self):
        rng = np.random.default_rng(207)
        vectors = rng.standard_normal((self.kinetic.shape[0], 4))
        vectors[:, 1] = vectors[:, 0]
        device = self.cp.asarray(vectors)
        with patch.dict(
            os.environ, {"PARSEC_CUPY_SPECULATIVE_MGS": "off"}
        ):
            literal = gpu_orthonormalize(
                device,
                rng=np.random.default_rng(991),
            )
        with patch.dict(
            os.environ, {"PARSEC_CUPY_SPECULATIVE_MGS": "complete"}
        ):
            audited = gpu_orthonormalize(
                device,
                rng=np.random.default_rng(991),
            )
        self.assertGreater(literal.random_replacements, 0)
        self.assertEqual(audited.random_replacements, literal.random_replacements)
        self.assertEqual(audited.zero_replacements, literal.zero_replacements)
        np.testing.assert_array_equal(
            self.cp.asnumpy(audited.basis),
            self.cp.asnumpy(literal.basis),
        )

    def test_appended_block_orthogonalization_passes_fp64_gram_audit(self):
        rng = self.cp.random.RandomState(911)
        source = rng.standard_normal((256, 11), dtype=self.cp.float64)
        prefix, _ = self.cp.linalg.qr(source[:, :5], mode="reduced")
        basis = self.cp.empty((256, 11), dtype=self.cp.float64)
        basis[:, :5] = prefix
        basis[:, 5:] = source[:, 5:]
        result = orthonormalize_appended_block(
            basis,
            existing_columns=5,
            rng=np.random.default_rng(17),
        )
        self.assertEqual(result.algorithm, "block_cgs2_device_mgs2")
        gram = result.basis.T @ result.basis
        np.testing.assert_allclose(
            self.cp.asnumpy(gram),
            np.eye(11),
            rtol=0.0,
            atol=5.0e-11,
        )

    def test_appended_block_fused_prefix_update_passes_fp64_audit(self):
        rng = self.cp.random.RandomState(917)
        source = rng.standard_normal((512, 15), dtype=self.cp.float64)
        prefix, _ = self.cp.linalg.qr(source[:, :5], mode="reduced")
        basis = self.cp.empty((512, 15), dtype=self.cp.float64)
        basis[:, :5] = prefix
        basis[:, 5:11] = source[:, 5:11]
        basis[:, 11:] = 0.0
        with patch.dict(
            os.environ,
            {
                "PARSEC_CUPY_CHEBDAV_FULL_WORKSPACE_CGS": "1",
                "PARSEC_CUPY_CHEBDAV_FUSED_PREFIX_UPDATE": "1",
            },
        ):
            result = orthonormalize_appended_block(
                basis,
                existing_columns=5,
                active_columns=11,
                rng=np.random.default_rng(19),
            )
        self.assertEqual(result.algorithm, "block_cgs2_device_mgs2")
        gram = result.basis[:, :11].T @ result.basis[:, :11]
        np.testing.assert_allclose(
            self.cp.asnumpy(gram),
            np.eye(11),
            rtol=0.0,
            atol=5.0e-11,
        )

@unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA are not available")
class TestCuPyEigvalPolicy(unittest.TestCase):
    def test_first_chebff_then_one_subspace_pass_keeps_device_state(self):
        cp, _ = require_cupy()
        kinetic, potential, nonlocal_operator, reference = _small_hamiltonian()
        settings = EigvalSettings(
            safety_buffer=2,
            initial_method="chebff",
            chebff=ChebFFSettings(
                polynomial_degree=6,
                filter_cycles=2,
                lanczos_steps=5,
                block_size=2,
                reset_recurrence_per_block=False,
                random_seed=21,
            ),
            subspace=SubspaceSettings(
                polynomial_degree=6,
                degree_delta=1,
                lanczos_steps=5,
                block_size=2,
                reset_recurrence_per_block=False,
                random_seed=21,
            ),
        )
        requested = 3
        reference_first = solve_eigval(
            reference.as_linear_operator(), requested, settings=settings
        )
        solver = CuPyEigvalSolver(
            kinetic,
            potential,
            nonlocal_operator,
            settings=settings,
        )
        gpu_first = solver.solve(requested)
        self.assertEqual(gpu_first.solver_path, "chebff")
        self.assertIsNone(gpu_first.residual_norms)
        self.assertIsInstance(solver.device_state.subspace.vectors, cp.ndarray)
        self.assertEqual(solver.device_state.subspace.vectors.shape, (24, 5))
        np.testing.assert_allclose(
            gpu_first.eigenvalues,
            reference_first.eigenvalues,
            rtol=2e-8,
            atol=2e-8,
        )

        updated = potential + 0.03 * np.cos(np.arange(potential.size))
        reference_updated = KohnShamHamiltonian(
            kinetic, updated, nonlocal_operator
        )
        reference_later = solve_eigval(
            reference_updated.as_linear_operator(),
            requested,
            settings=settings,
            state=reference_first.state,
        )
        buffered_before = solver.device_state.subspace.vectors
        solver.update_local_potential(updated)
        self.assertIs(solver.device_state.subspace.vectors, buffered_before)
        gpu_later = solver.solve(requested)
        self.assertEqual(gpu_later.solver_path, "subspace")
        self.assertIsNotNone(gpu_later.residual_norms)
        self.assertEqual(solver.device_state.solves_completed, 2)
        self.assertEqual(solver.device_state.subspace.filters_completed, 1)
        np.testing.assert_allclose(
            gpu_later.eigenvalues,
            reference_later.eigenvalues,
            rtol=5e-8,
            atol=5e-8,
        )
        self.assertEqual(solver.timing_stats.first_solve_calls, 1)
        self.assertEqual(solver.timing_stats.subspace_solve_calls, 1)
        self.assertEqual(solver.timing_stats.solve_calls, 2)


if __name__ == "__main__":
    unittest.main()
