"""Parity and optional-runtime tests for the CuPy acceleration slice."""

from __future__ import annotations

import gc
import hashlib
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
from parsec_python.acceleration.backends import cupy_projectors
from parsec_python.acceleration.backends.cupy_launch import kernel_parameters
from parsec_python.acceleration.backends.cupy_runtime import CuPyHamiltonianBackend
from parsec_python.acceleration.backends.cupy_projectors import (
    CuPySparseProjectorFactors,
    projection_kernels,
    projection_launches,
    projection_reduction,
    projector_block_columns,
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
from parsec_python.acceleration.Eigensolvers.symmetry import CuPySymmetrySCFEigensolver
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


def _long_row_hamiltonian(dimension=400):
    """Kinetic matrix, potential, projectors and signs with projector rows that take the tree reduction.

    Projectors of 400, 300 and 5 entries and one without any: the longest
    one decides the reduction of them all.
    """

    kinetic = sp.diags(
        (
            -0.25 * np.ones(dimension - 3),
            -np.ones(dimension - 1),
            2.4 * np.ones(dimension),
            -np.ones(dimension - 1),
        ),
        (-3, -1, 0, 1),
        format="csr",
    )
    potential = np.linspace(-0.75, 0.35, dimension)
    generator = np.random.default_rng(61)
    dense = np.zeros((dimension, 4), dtype=np.float64)
    dense[:, 0] = 0.05 * generator.standard_normal(dimension)
    dense[60:360, 1] = 0.05 * generator.standard_normal(300)
    dense[11:16, 3] = (0.2, -0.35, 0.15, 0.1, -0.3)
    return kinetic, potential, sp.csr_matrix(dense), np.array((1.0, -1.0, 1.0, -1.0))


def _bits(array):
    """The bits of a host array of FP64 values: a zero of the other sign differs too."""

    return np.ascontiguousarray(array).view(np.uint64)


def _block_columns(value=None):
    """The environment with ``PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS`` at ``value``, or without it."""

    name = "PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS"
    kept = {key: item for key, item in os.environ.items() if key != name}
    return patch.dict(
        os.environ, kept if value is None else {**kept, name: value}, clear=True
    )


class TestCuPyOptionalImport(unittest.TestCase):
    def test_projection_block_columns_are_six_unless_the_former_kernel_is_named(self):
        for value, columns in ((None, 6), ("6", 6), (" 6 ", 6), ("", 6), ("1", 1)):
            with self.subTest(value=value), _block_columns(value):
                self.assertEqual(projector_block_columns(), columns)
        for value in ("0", "2", "12", "auto", "on", "six"):
            with self.subTest(value=value), _block_columns(value):
                with self.assertRaisesRegex(
                    ValueError, "PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS must be 6 or 1"
                ):
                    projector_block_columns()
        # What an operator reports: the former names where its kernels run.
        self.assertEqual(
            [
                projection_reduction(parallel, columns)
                for parallel in (False, True)
                for columns in (1, 6)
            ],
            ["serial", "serial", "parallel", "parallel6"],
        )
        with self.assertRaisesRegex(ValueError, "6 columns or 1"):
            projection_reduction(True, 3)

    def test_projection_launches_take_every_projector_and_column_once(self):
        for projectors in (1, 5, 4601):
            for width in (1, 2, 5, 6, 7, 12, 13, 126, 128):
                with self.subTest(projectors=projectors, width=width):
                    pairs = projectors * width
                    # As before: one launch, with a block or a thread per
                    # projector and column.
                    self.assertEqual(
                        projection_launches("parallel", projectors, width),
                        (("parallel", (pairs,), 0, width),),
                    )
                    self.assertEqual(
                        projection_launches("serial", projectors, width),
                        (("serial", (-(-pairs // 128),), 0, width),),
                    )
                    # One column as the former kernel takes it, two to six
                    # in one launch of a block per projector, more in one
                    # launch of a block per projector and group of six.
                    if width == 1:
                        expected = (("parallel", (projectors,), 0, 1),)
                    elif width <= 6:
                        expected = (("parallel6", (projectors,), 0, width),)
                    else:
                        expected = (
                            ("parallel6_groups", (projectors, -(-width // 6)), 0, width),
                        )
                    self.assertEqual(
                        projection_launches("parallel6", projectors, width), expected
                    )
        # Every kernel that a reduction launches, and no other.
        for reduction in ("parallel", "serial", "parallel6"):
            with self.subTest(reduction=reduction):
                self.assertEqual(
                    {
                        kernel
                        for width in range(1, 20)
                        for kernel, *_ in projection_launches(reduction, 3, width)
                    },
                    set(projection_kernels(reduction)),
                )
        self.assertEqual(
            projection_kernels("parallel6"), ("parallel", "parallel6", "parallel6_groups")
        )
        # The y dimension of a grid holds 65,535 groups: wider blocks, far
        # beyond those of a calculation, take a launch for every 65,535.
        groups = 65535
        self.assertEqual(
            projection_launches("parallel6", 7, 6 * groups),
            (("parallel6_groups", (7, groups), 0, 6 * groups),),
        )
        self.assertEqual(
            projection_launches("parallel6", 7, 6 * groups + 1),
            (
                ("parallel6_groups", (7, groups), 0, 6 * groups),
                ("parallel6_groups", (7, 1), 6 * groups, 6 * groups + 1),
            ),
        )
        with patch.object(cupy_projectors, "_GRID_GROUPS", 2):
            for width in range(7, 40):
                with self.subTest(groups=2, width=width):
                    launches = projection_launches("parallel6", 5, width)
                    self.assertEqual(
                        [column for _kernel, _grid, start, stop in launches for column in range(start, stop)],
                        list(range(width)),
                    )
                    self.assertEqual(
                        [(kernel, grid) for kernel, grid, _start, _stop in launches],
                        [
                            ("parallel6_groups", (5, -(-(stop - start) // 6)))
                            for _kernel, _grid, start, stop in launches
                        ],
                    )
                    self.assertTrue(all(stop - start <= 12 for _kernel, _grid, start, stop in launches))
                    self.assertEqual(len(launches), -(-width // 12))
        self.assertEqual(projection_launches("parallel6", 7, 0), ())
        with self.assertRaisesRegex(ValueError, "unknown projection reduction"):
            projection_launches("tree", 1, 1)
        with self.assertRaisesRegex(ValueError, "unknown projection reduction"):
            projection_kernels("tree")

    def test_the_four_projection_kernels_declare_the_same_parameters(self):
        # One argument tuple serves them: only the blocks of a launch differ.
        former = kernel_parameters(cupy_projectors._SOURCE, "sparse_projector_projection")
        self.assertEqual(len(former), 12)
        for name in (
            "sparse_projector_projection6",
            "sparse_projector_projection6_groups",
            "sparse_projector_projection_serial",
        ):
            with self.subTest(kernel=name):
                self.assertEqual(kernel_parameters(cupy_projectors._SOURCE, name), former)
        source = cupy_projectors._SOURCE

        def text(name):
            start = source.index(f"void {name}(")
            return source[start : source.index("extern", start)]

        one, six, groups = (
            text(name)
            for name in (
                "sparse_projector_projection",
                "sparse_projector_projection6",
                "sparse_projector_projection6_groups",
            )
        )
        # The six-column kernel is the text that was first timed on an A100
        # (commit a1869453), to the character.
        self.assertEqual(
            hashlib.sha256(six.encode()).hexdigest(),
            "0b3c5513282dd6ef33470245761ae84e60578a3946588d1de3f70f3582a84677",
        )
        self.assertNotIn("blockIdx.y", six)
        self.assertEqual(six.count("if (column < width) {"), 2)
        # The six-column kernels sum a column as the former kernel does: a
        # partial sum per thread over every 128th entry, and the same tree.
        for many in (six, groups):
            for line in (
                "int position = row_offsets[projector] + threadIdx.x;",
                "position += blockDim.x",
                "for (int offset = 64; offset > 0; offset >>= 1) {",
                "if (threadIdx.x < offset) {",
            ):
                with self.subTest(text=line):
                    self.assertEqual((one.count(line), many.count(line)), (1, 1))
            self.assertIn(
                "sums[column][threadIdx.x] += sums[column][threadIdx.x + offset];", many
            )
            self.assertIn("] = signs[projector] * sums[column][0];", many)
            self.assertIn("partial[column] += value * vectors[", many)
        self.assertIn("sums[threadIdx.x] += sums[threadIdx.x + offset];", one)
        self.assertIn("] = signs[projector] * sums[0];", one)
        # A block of the group kernel reads and writes the columns of its
        # group, from six times its y index on: the offset of the group joins
        # the offset of an entry's row once per entry.
        self.assertIn("const int first = 6 * static_cast<int>(blockIdx.y);", groups)
        self.assertIn("const int columns = width - first;", groups)
        self.assertIn(
            "const long long group = static_cast<long long>(first) * vector_column_stride;",
            groups,
        )
        self.assertIn("* vector_row_stride + group;", groups)
        self.assertEqual(groups.count("static_cast<long long>(first + column)"), 1)
        self.assertEqual(groups.count("if (column < columns) {"), 2)
        self.assertNotIn("blockIdx.y", one)

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

    def tearDown(self):
        # Operators and solvers that a test left in a reference cycle are destroyed
        # here, between the tests, and not by a collection inside the next one.
        gc.collect()

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

    def test_symmetry_orbitals_expand_from_host_maps_like_device_maps(self):
        cp = self.cp
        generator = np.random.default_rng(77)
        # Two sectors on a five-orbit wedge; the second rejects orbit 1.
        sector_orbits = (np.arange(5), np.asarray((0, 2, 3, 4)))
        multiplicities = np.asarray((2, 1, 2, 2, 2))
        scales = tuple(1.0 / np.sqrt(multiplicities[item]) for item in sector_orbits)
        # The eigensolver shares its orbit lists and scales read-only.
        for item in (*sector_orbits, *scales):
            item.setflags(write=False)
        mapping = np.asarray((0, 1, 2, 3, 4, 0, 2, 3, 4), dtype=np.int64)
        phases = np.asarray(
            ((1,) * 9, (1, 0, 1, 1, 1, -1, -1, -1, -1)), dtype=np.int8
        )
        hosts = (
            generator.standard_normal((5, 3)),
            generator.standard_normal((4, 2)),
        )
        representations = np.asarray((1, 0, 0, 1, 0), dtype=np.int32)
        columns = np.asarray((0, 0, 1, 1, 2), dtype=np.int32)
        vectors = tuple(cp.asarray(values, order="F") for values in hosts)

        def orbitals(device_full_to_wedge, phase_table):
            return CuPySymmetryOrbitals(
                scaled_wedge_vectors=None,
                representations=representations,
                full_to_wedge=mapping,
                device_full_to_wedge=device_full_to_wedge,
                phases=phase_table,
                full_size=mapping.size,
                representation_columns=columns,
                sector_vectors=vectors,
                sector_orbits=sector_orbits,
                sector_scales=scales,
                wedge_size=5,
            )

        expected = np.empty((mapping.size, representations.size))
        for column, (representation, source) in enumerate(
            zip(representations, columns)
        ):
            wedge = np.zeros(5)
            wedge[sector_orbits[representation]] = (
                hosts[representation][:, source] * scales[representation]
            )
            expected[:, column] = wedge[mapping] * phases[representation]
        # Host maps, as the eigensolver now hands them out, and the device
        # maps it used to upload give the same signed full-grid states.
        host_maps = orbitals(None, phases)
        device_maps = orbitals(
            cp.asarray(mapping), cp.asarray(phases, dtype=cp.float64)
        )
        for item in (host_maps, device_maps):
            np.testing.assert_array_equal(
                cp.asnumpy(item.to_full_device()), expected
            )
            np.testing.assert_array_equal(
                item.to_full_host(block_states=2), expected
            )

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

    def test_cusolver_generalized_ritz_matches_orthonormal_ritz(self):
        # The eigensolver setting belongs to the host solve.
        with patch.dict(
            os.environ,
            {
                "PARSEC_CUPY_RITZ_DENSE_BACKEND": "host",
                "PARSEC_CUPY_RITZ_EIGH_BACKEND": "cupy",
            },
        ):
            self.test_generalized_ritz_matches_orthonormal_ritz_and_reuses_workspace()

    def test_consumed_ritz_rotation_swaps_disjoint_buffers(self):
        cp = self.cp
        rng = np.random.default_rng(831)
        basis = cp.asarray(rng.standard_normal((self.kinetic.shape[0], 5)), order="F")
        saved = basis.copy()
        workspace = cp.empty_like(basis, order="F")
        reference = generalized_rayleigh_ritz(self.gpu, basis)
        with patch.dict(os.environ, {"PARSEC_CUPY_RITZ_ROTATION": "reuse"}):
            actual = generalized_rayleigh_ritz(
                self.gpu, basis, workspace=workspace, consume_basis=True)
        self.assertIs(actual.wavefunctions, workspace)
        self.assertIs(actual.workspace, basis)
        self.assertFalse(cp.may_share_memory(actual.workspace, actual.wavefunctions))
        np.testing.assert_array_equal(cp.asnumpy(basis), cp.asnumpy(saved))
        np.testing.assert_allclose(cp.asnumpy(actual.wavefunctions),
                                   cp.asnumpy(reference.wavefunctions), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(cp.asnumpy(actual.wavefunctions.T @ actual.wavefunctions),
                                   np.eye(5), rtol=2e-11, atol=2e-11)
        # A second call may overwrite scratch but must preserve prior orbitals.
        old_vectors = actual.wavefunctions.copy()
        next_basis = cp.asarray(rng.standard_normal(basis.shape), order="F")
        with patch.dict(os.environ, {"PARSEC_CUPY_RITZ_ROTATION": "reuse"}):
            generalized_rayleigh_ritz(self.gpu, next_basis,
                                     workspace=actual.workspace, consume_basis=True)
        np.testing.assert_array_equal(cp.asnumpy(actual.wavefunctions), cp.asnumpy(old_vectors))

    def test_reuse_ritz_preserves_diagnostics_and_unconsumed_input(self):
        with patch.dict(os.environ, {"PARSEC_CUPY_RITZ_ROTATION": "reuse"}):
            self.test_generalized_ritz_matches_orthonormal_ritz_and_reuses_workspace()
        cp = self.cp
        basis = cp.asarray(np.random.default_rng(832).standard_normal(
            (self.kinetic.shape[0], 5)), order="F")
        saved = basis.copy()
        with patch.dict(os.environ, {"PARSEC_CUPY_RITZ_ROTATION": "reuse"}):
            actual = generalized_rayleigh_ritz(self.gpu, basis, workspace=basis,
                                               consume_basis=True, compute_residuals=True)
        self.assertFalse(cp.may_share_memory(actual.workspace, basis))
        np.testing.assert_array_equal(cp.asnumpy(basis), cp.asnumpy(saved))
        self.assertIsNotNone(actual.residual_norms)

    def test_cusolver_generalized_ritz_retains_dependence_guard(self):
        with patch.dict(
            os.environ,
            {
                "PARSEC_CUPY_RITZ_DENSE_BACKEND": "host",
                "PARSEC_CUPY_RITZ_EIGH_BACKEND": "cupy",
            },
        ):
            self.test_generalized_ritz_rejects_dependent_filtered_basis()

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
        # The tree reduction, of six columns per block unless the former
        # kernel is named.
        for value, reduction in ((None, "parallel6"), ("1", "parallel")):
            with self.subTest(block_columns=value), _block_columns(value):
                gpu = CuPyHamiltonian(
                    kinetic,
                    np.zeros(dimension, dtype=np.float64),
                    nonlocal_operator,
                )

                projection = gpu.custom_projector_projection
                self.assertIsNotNone(projection)
                self.assertEqual(projection.max_row_entries, dimension)
                self.assertEqual(projection.reduction_mode, reduction)

    def test_six_columns_per_block_give_the_coefficients_of_one_column_per_block(self):
        cp = self.cp
        generator = np.random.default_rng(71)
        rows = 3000
        # Rows around the 128 threads of a block and around the 256 entries
        # from which a row takes the tree, a full one and one without entries.
        lengths = (1, 127, 128, 129, 255, 256, 257, 700, 1500, rows, 0)
        dense = np.zeros((rows, len(lengths)))
        for projector, entries in enumerate(lengths):
            picked = generator.choice(rows, size=entries, replace=False)
            dense[picked, projector] = generator.standard_normal(entries)
        host = sp.csr_matrix(dense)
        host_signs = np.where(np.arange(len(lengths)) % 3 == 0, -1.0, 1.0)
        signs = cp.asarray(host_signs)
        former = CuPySparseProjectorFactors(cp, host, signs, block_columns=1).projection
        six = CuPySparseProjectorFactors(cp, host, signs, block_columns=6).projection
        kernels = cupy_projectors._kernels(cp)
        self.assertEqual(
            (former.reduction_mode, former.block_columns, six.reduction_mode, six.block_columns),
            ("parallel", 1, "parallel6", 6),
        )
        self.assertEqual(former.kernels, {"parallel": kernels[0]})
        self.assertEqual(
            six.kernels,
            {"parallel": kernels[0], "parallel6": kernels[3], "parallel6_groups": kernels[4]},
        )

        wide = generator.standard_normal((rows, 29))
        reference = host_signs[:, None] * (host.T @ wide)
        device = cp.asarray(wide, order="F")
        # One column takes the former kernel in both; two to six the
        # six-column kernel, seven to fourteen its group kernel in one launch
        # of two or three groups.
        for width in range(1, 15):
            # Columns as a filter block holds them, the same rows first,
            # and every other column of a wider block.
            views = (
                ("columns", device[:, :width], slice(0, width)),
                ("rows first", cp.ascontiguousarray(device[:, :width]), slice(0, width)),
                ("every other column", device[:, : 2 * width : 2], slice(0, 2 * width, 2)),
            )
            for name, block, columns in views:
                with self.subTest(width=width, block=name):
                    expected = cp.asnumpy(former(block))
                    actual = cp.asnumpy(six(block))
                    self.assertEqual(actual.shape, (len(lengths), width))
                    np.testing.assert_array_equal(_bits(actual), _bits(expected))
                    np.testing.assert_allclose(
                        actual, reference[:, columns], rtol=0.0, atol=1e-11
                    )
            # Into the buffer of a caller, column-major and row-major, as a
            # filter graph passes a view of its coefficient buffer.
            for order in ("F", "C"):
                with self.subTest(width=width, output=order):
                    buffers = [
                        cp.full((len(lengths), 15), np.nan, order=order) for _ in range(2)
                    ]
                    self.assertIs(
                        six(device[:, :width], output=buffers[1][:, :width]).base,
                        buffers[1],
                    )
                    former(device[:, :width], output=buffers[0][:, :width])
                    np.testing.assert_array_equal(
                        _bits(cp.asnumpy(buffers[1])), _bits(cp.asnumpy(buffers[0]))
                    )
                    self.assertTrue(bool(cp.isfinite(buffers[1][:, :width]).all()))
        # A single vector, as the Lanczos bound applies the operator.
        vector = cp.asnumpy(six(device[:, 3]))
        self.assertEqual(vector.shape, (len(lengths),))
        np.testing.assert_array_equal(
            _bits(vector), _bits(cp.asnumpy(former(device[:, 3])))
        )
        # A block of more groups than a launch takes, as only a block of
        # more than 393,210 columns has them: launches of two groups here.
        with patch.object(cupy_projectors, "_GRID_GROUPS", 2):
            for width in (2, 12, 13, 25, 29):
                with self.subTest(width=width, groups_per_launch=2):
                    block = device[:, :width]
                    np.testing.assert_array_equal(
                        _bits(cp.asnumpy(six(block))), _bits(cp.asnumpy(former(block)))
                    )

        # The launches: one of a block per projector and column, against one
        # of a block per projector and group of six columns, and one of the
        # former kernel for a single column.
        def launches(projection, widths):
            calls = []
            counting = {}
            for name, kernel in projection.kernels.items():

                def count(grid, block, arguments, name=name, kernel=kernel):
                    calls.append((name, grid, block, int(arguments[1])))
                    return kernel(grid, block, arguments)

                counting[name] = count
            with patch.dict(projection.kernels, counting):
                for width in widths:
                    projection(device[:, :width])
            return calls

        count = len(lengths)
        self.assertEqual(
            launches(former, (14, 6, 1)),
            [
                ("parallel", (count * 14,), (128,), 14),
                ("parallel", (count * 6,), (128,), 6),
                ("parallel", (count,), (128,), 1),
            ],
        )
        self.assertEqual(
            launches(six, (14, 6, 1, 2, 7)),
            [
                ("parallel6_groups", (count, 3), (128,), 14),
                ("parallel6", (count,), (128,), 6),
                ("parallel", (count,), (128,), 1),
                ("parallel6", (count,), (128,), 2),
                ("parallel6_groups", (count, 2), (128,), 7),
            ],
        )

    def test_projection_block_columns_switch_restores_the_former_kernel(self):
        cp = self.cp
        kinetic, potential, projectors, signs = _long_row_hamiltonian()
        kernels = cupy_projectors._kernels(cp)
        generator = np.random.default_rng(72)
        host = generator.standard_normal((kinetic.shape[0], 13))
        step = dict(center=0.5, scale=1.0 / 64.0, sigma=0.4, sigma_next=1.5)

        def results(operator):
            """What an operator gives with its projection: H X, into a buffer, and recurrence steps."""

            values = []
            for width in (1, 6, 13):
                block = cp.asarray(host[:, :width], order="F")
                former = cp.asarray(host[::-1, :width], order="F")
                values.append(cp.asnumpy(operator.apply(block)))
                values.append(cp.asnumpy(operator.apply_into(block, cp.empty_like(block))))
                values.append(cp.asnumpy(operator.chebyshev_recurrence(block, **step, previous=former)))
                values.append(
                    cp.asnumpy(operator.chebyshev_recurrence(block, center=0.5, scale=1.0 / 64.0))
                )
            values.append(cp.asnumpy(operator.apply(cp.asarray(host[:, 2]))))
            if operator.raw_projector_factors is not None:
                # The nonlocal part alone, which the full-grid operator
                # leaves to cuSPARSE and its own order of summation.
                values.append(cp.asnumpy(operator.apply_nonlocal(cp.asarray(host, order="F"))))
            return values

        # The operator of the full grid, with its cuSPARSE matrices, and the
        # operator of a symmetry sector, with raw factors alone.
        for sector in (False, True):
            built = {}
            for value in (None, "6", "1"):
                with _block_columns(value):
                    built[value] = CuPyHamiltonian(
                        kinetic,
                        potential,
                        projectors=projectors,
                        projector_signs=signs,
                        retain_generic_laplacian=not sector,
                    )
            with self.subTest(sector=sector):
                self.assertEqual(
                    [built[value].raw_projector_factors is not None for value in built],
                    [sector] * 3,
                )
                self.assertEqual(
                    [built[value].custom_projector_projection.reduction_mode for value in built],
                    ["parallel6", "parallel6", "parallel"],
                )
                self.assertEqual(
                    [built[value].custom_projector_projection.block_columns for value in built],
                    [6, 6, 1],
                )
                self.assertEqual(
                    built[None].custom_projector_projection.kernels,
                    {"parallel": kernels[0], "parallel6": kernels[3], "parallel6_groups": kernels[4]},
                )
                self.assertEqual(
                    built["1"].custom_projector_projection.kernels, {"parallel": kernels[0]}
                )
                # The former value launches what the code launched before
                # the switch: one launch, a block per projector and column.
                # The default launches thirteen columns as three groups of
                # a block per projector, six as a block per projector, and
                # one column as before.
                for value, expected in (
                    (
                        "1",
                        [
                            ("parallel", (4 * 13,), 4, 13),
                            ("parallel", (4 * 6,), 4, 6),
                            ("parallel", (4,), 4, 1),
                        ],
                    ),
                    (
                        None,
                        [
                            ("parallel6_groups", (4, 3), 4, 13),
                            ("parallel6", (4,), 4, 6),
                            ("parallel", (4,), 4, 1),
                        ],
                    ),
                ):
                    projection = built[value].custom_projector_projection
                    calls = []
                    counting = {}
                    for name, kernel in projection.kernels.items():

                        def count(grid, block, arguments, name=name, kernel=kernel):
                            self.assertEqual(block, (128,))
                            calls.append((name, grid, int(arguments[0]), int(arguments[1])))
                            return kernel(grid, block, arguments)

                        counting[name] = count
                    with patch.dict(projection.kernels, counting):
                        built[value].apply(cp.asarray(host, order="F"))
                        built[value].apply(cp.asarray(host[:, :6], order="F"))
                        built[value].apply(cp.asarray(host[:, 4]))
                    self.assertEqual(calls, expected)
                expected = results(built["1"])
                self.assertTrue(all(np.isfinite(value).all() and value.any() for value in expected))
                for value in (None, "6"):
                    for actual, former in zip(results(built[value]), expected, strict=True):
                        np.testing.assert_array_equal(_bits(actual), _bits(former))

        # Short projector rows keep their canonical-order kernel under either value.
        small_kinetic, small_potential, small_nonlocal, _ = _small_hamiltonian()
        for value in (None, "1"):
            with self.subTest(short_rows=value), _block_columns(value):
                short = CuPyHamiltonian(small_kinetic, small_potential, small_nonlocal)
                projection = short.custom_projector_projection
                self.assertEqual(
                    (projection.reduction_mode, projection.block_columns), ("serial", 1)
                )
                self.assertEqual(projection.kernels, {"serial": kernels[2]})

        # A value that the switch does not know is an error of the operator,
        # with projectors and without: it is not left to cuSPARSE.
        with _block_columns("2"):
            for arguments in (
                dict(projectors=projectors, projector_signs=signs),
                {},
            ):
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_PROJECTOR_BLOCK_COLUMNS"):
                    CuPyHamiltonian(kinetic, potential, **arguments)

        # The columns that a caller names outrank the switch, as a sector
        # group names those of its owner for its other devices: in the
        # operator of a sector, with raw factors alone, and in that of the
        # full grid, whose projection reads its cuSPARSE matrices.
        for value, named, reduction in (("6", 1, "parallel"), ("1", 6, "parallel6")):
            for sector in (True, False):
                with self.subTest(switch=value, named=named, sector=sector), _block_columns(value):
                    given = CuPyHamiltonian(
                        kinetic,
                        potential,
                        projectors=projectors,
                        projector_signs=signs,
                        retain_generic_laplacian=not sector,
                        projection_block_columns=named,
                    )
                    self.assertEqual(given.raw_projector_factors is not None, sector)
                    self.assertEqual(given.custom_projector_projection.reduction_mode, reduction)
        with self.assertRaisesRegex(ValueError, "6 columns or 1"):
            CuPyHamiltonian(
                kinetic,
                potential,
                projectors=projectors,
                projector_signs=signs,
                projection_block_columns=3,
            )

    def test_the_reduction_is_reported_where_the_projection_forms_the_coefficients(self):
        """With ``PARSEC_CUPY_FUSED_PROJECTORS=0`` cuSPARSE forms the coefficients.

        The operators hold a projection there as well, but launch none of its
        kernels, so the reports name its reduction on the fused route only.
        """

        cp = self.cp
        kinetic, potential, projectors, signs = _long_row_hamiltonian()
        nonlocal_operator = NonlocalProjectorOperator(
            projectors=sp.csc_matrix(projectors),
            signs=signs,
            labels=tuple((index, 0, 0) for index in range(signs.size)),
        )
        block = cp.asarray(
            np.random.default_rng(73).standard_normal((kinetic.shape[0], 7)), order="F"
        )
        for fused, full_grid, sectors in (("1", "parallel6", "parallel6"), ("0", "cuSPARSE", "none")):
            with (
                self.subTest(fused=fused),
                _block_columns(None),
                patch.dict(os.environ, PARSEC_CUPY_FUSED_PROJECTORS=fused),
            ):
                backend = CuPyHamiltonianBackend(kinetic, nonlocal_operator)
                sector = CuPyHamiltonian(
                    kinetic,
                    potential,
                    projectors=projectors,
                    projector_signs=signs,
                    retain_generic_laplacian=False,
                )
                launched = []
                for operator in (backend.device_operator, sector):
                    projection = operator.custom_projector_projection
                    self.assertEqual(projection.reduction_mode, "parallel6")
                    self.assertEqual(operator.fused_projector_scatter, fused == "1")
                    counting = {}
                    for name, kernel in projection.kernels.items():

                        def count(grid, block_shape, arguments, name=name, kernel=kernel):
                            launched.append(name)
                            return kernel(grid, block_shape, arguments)

                        counting[name] = count
                    with patch.dict(projection.kernels, counting):
                        operator.apply(block)
                        operator.apply(block[:, :4])
                        operator.apply(block[:, 0])
                        operator.chebyshev_recurrence(block, center=0.5, scale=1.0 / 64.0)
                self.assertEqual(
                    sorted(set(launched)),
                    ["parallel", "parallel6", "parallel6_groups"] if fused == "1" else [],
                )
                self.assertEqual(dict(backend.info.details)["gpu_projector_reduction"], full_grid)
                solver = object.__new__(CuPySymmetrySCFEigensolver)
                solver._operators = [sector, None]
                self.assertEqual(solver.projector_reduction_modes, sectors)

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
    def tearDown(self):
        # Operators and solvers that a test left in a reference cycle are destroyed
        # here, between the tests, and not by a collection inside the next one.
        gc.collect()

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
