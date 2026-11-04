"""Mock-CUDA integration coverage for the CuPy SCF execution hook.

The production CuPy tests use a real device when available.  This file uses a
small NumPy/SciPy API shim so the SCF handoff, persistent solver state, result
adapter, and timing propagation are also exercised in CPU-only CI.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

import parsec_python.acceleration.backends.cupy as cupy_kernels
from parsec_python.acceleration.SCF.single_point import (
    AcceleratedPreparedSinglePointSystem,
    run_scf,
)
from parsec_python.acceleration.backends.cupy_runtime import CuPyHamiltonianBackend
from parsec_python.acceleration.backends.scipy import ScipyHamiltonianBackend
from parsec_python.Hartree.poisson import HartreeResult
from parsec_python.V_ion import NonlocalProjectorOperator
from parsec_python.V_xc import XCResult
from parsec_python.models import (
    EigensolverSettings,
    MixingSettings,
    PreparationTimings,
    SCFSettings,
)


class _FakeStream:
    def synchronize(self) -> None:
        return None


class _FakeRuntime:
    @staticmethod
    def getDeviceCount() -> int:
        return 1

    @staticmethod
    def getDeviceProperties(device: int):
        return {"name": b"NumPy CUDA test shim", "device": int(device)}


class _FakeDevice:
    id = 0


def _fake_cupy():
    return SimpleNamespace(
        __version__="test-shim",
        asarray=np.asarray,
        asnumpy=np.asarray,
        array=np.array,
        # The reused prefix buffer of the device Gram-Schmidt (orthogonalize.py).
        empty=np.empty,
        empty_like=np.empty_like,
        zeros_like=np.zeros_like,
        tril=np.tril,
        vdot=np.vdot,
        float64=np.float64,
        dtype=np.dtype,
        linalg=np.linalg,
        ndarray=np.ndarray,
        cuda=SimpleNamespace(
            runtime=_FakeRuntime(),
            Device=_FakeDevice,
            get_current_stream=lambda: _FakeStream(),
        ),
    )


@contextmanager
def _mock_cuda_runtime():
    previous_cupy = cupy_kernels._CUPY
    previous_sparse = cupy_kernels._CUPYX_SPARSE
    cupy_kernels._CUPY = _fake_cupy()
    cupy_kernels._CUPYX_SPARSE = sp
    try:
        yield
    finally:
        cupy_kernels._CUPY = previous_cupy
        cupy_kernels._CUPYX_SPARSE = previous_sparse


class _SmallPhysicalSystem:
    """Minimal deterministic physics object accepted by the real SCF loop."""

    def __init__(self) -> None:
        dimension = 24
        self.grid = SimpleNamespace(size=dimension, volume_element=1.0)
        self.negative_laplacian = sp.diags(
            (
                -np.ones(dimension - 1),
                2.5 * np.ones(dimension),
                -np.ones(dimension - 1),
            ),
            (-1, 0, 1),
            format="csr",
        )
        projector_values = np.zeros((dimension, 2), dtype=np.float64)
        projector_values[3:7, 0] = (0.20, -0.35, 0.15, 0.10)
        projector_values[15:20, 1] = (-0.10, 0.25, 0.30, -0.20, 0.05)
        self.nonlocal_operator = NonlocalProjectorOperator(
            projectors=sp.csc_matrix(projector_values),
            signs=np.array((1.0, -1.0)),
            labels=((0, 0, 0), (1, 0, 0)),
        )
        self.ionic_potential = np.linspace(-0.8, 0.2, dimension)
        self.electron_count = 2.0
        self.initial_density = np.full(dimension, self.electron_count / dimension)
        self.core_density = np.zeros(dimension)
        self.ion_ion_energy = 0.0
        # Match PreparedSinglePoint's energy bookkeeping.  The synthetic
        # system has no isolated-atom reference shift, so the neutral value is
        # exactly zero.
        self.atomic_reference_correction = 0.0
        self.atoms = ()
        self.pseudopotentials = {}
        self.timings = PreparationTimings(total_seconds=0.01)
        self.input = SimpleNamespace(
            scf=SCFSettings(
                max_iterations=2,
                convergence_criterion=1.0e-30,
                fermi_temperature_kelvin=80.0,
                number_of_states=3,
            ),
            eigensolver=EigensolverSettings(
                method="chebff",
                first_filter_degree=10,
                first_filter_cycles=1,
                matvec_block_size=2,
                subspace_buffer=6,
                filter_degree=10,
                filter_degree_delta=0,
                lanczos_steps=5,
                random_seed=13,
            ),
            mixing=MixingSettings(parameter=0.2, memory=2, restart=20),
        )

    def solve_hartree(self, density, initial_potential=None, **kwargs):
        del initial_potential, kwargs
        density = np.asarray(density, dtype=np.float64)
        potential = 0.18 * density
        return HartreeResult(
            potential=potential,
            right_hand_side=density.copy(),
            boundary=SimpleNamespace(),
            converged=True,
            iterations=1,
            matrix_vector_products=1,
            residual_norm=0.0,
            initial_residual_norm=0.0,
        )

    def evaluate_xc(self, density):
        density = np.asarray(density, dtype=np.float64)
        potential = -0.12 * np.cbrt(np.maximum(density, 0.0))
        epsilon = 0.75 * potential
        energy_density = density * epsilon
        return XCResult(
            potential=potential,
            energy_per_electron=epsilon,
            energy_density=energy_density,
            total_energy=float(np.sum(energy_density)),
        )


def _accelerated_system(reference, backend):
    return AcceleratedPreparedSinglePointSystem(
        reference=reference,
        backend=backend,
        backend_info=backend.info,
        eigenproblem_solver=getattr(backend, "eigenproblem_solver", None),
    )


class _RecordingEigensolver:
    def __init__(self, delegate, expected_operator) -> None:
        self.delegate = delegate
        self.expected_operator = expected_operator
        self.paths: list[str] = []
        self.operators: list[object] = []

    def __call__(self, operator, *args, **kwargs):
        self.operators.append(operator)
        if operator is not self.expected_operator:
            raise AssertionError("SCF did not pass the persistent device operator")
        if isinstance(operator, LinearOperator):
            raise AssertionError("production CuPy SCF used a host LinearOperator")
        result = self.delegate(operator, *args, **kwargs)
        self.paths.append(result.solver_path)
        return result


class CuPySCFIntegrationTests(unittest.TestCase):
    def test_two_iteration_scf_uses_chebff_then_subspace_and_matches_reference(self):
        reference_cpu = _SmallPhysicalSystem()
        scipy_backend = ScipyHamiltonianBackend(
            reference_cpu.negative_laplacian,
            reference_cpu.nonlocal_operator,
        )
        expected = run_scf(_accelerated_system(reference_cpu, scipy_backend))

        with _mock_cuda_runtime():
            reference_gpu = _SmallPhysicalSystem()
            cupy_backend = CuPyHamiltonianBackend(
                reference_gpu.negative_laplacian,
                reference_gpu.nonlocal_operator,
            )
            system = _accelerated_system(reference_gpu, cupy_backend)
            recording = _RecordingEigensolver(
                system.eigenproblem_solver, cupy_backend.device_operator
            )
            system.eigenproblem_solver = recording
            actual = run_scf(system)

        self.assertEqual(recording.paths, ["chebff", "subspace"])
        self.assertTrue(
            all(operator is cupy_backend.device_operator for operator in recording.operators)
        )
        self.assertEqual(actual.iterations, 2)
        self.assertEqual(len(actual.history), 2)
        np.testing.assert_allclose(actual.eigenvalues, expected.eigenvalues, atol=1e-12)
        np.testing.assert_allclose(actual.density, expected.density, atol=1e-12)
        self.assertAlmostEqual(actual.energies.total, expected.energies.total, 12)
        self.assertGreater(actual.timings.diagonalization_seconds, 0.0)
        self.assertGreater(actual.backend_statistics.device_seconds, 0.0)
        self.assertGreater(actual.backend_statistics.device_to_host_seconds, 0.0)
        self.assertEqual(cupy_backend.eigenproblem_solver.solver.device_state.solves_completed, 2)

    def test_reusing_prepared_system_starts_a_fresh_chebff_sequence(self):
        """A second independent SCF run must not inherit the prior run state."""

        with _mock_cuda_runtime():
            reference = _SmallPhysicalSystem()
            backend = CuPyHamiltonianBackend(
                reference.negative_laplacian,
                reference.nonlocal_operator,
            )
            system = _accelerated_system(reference, backend)
            recording = _RecordingEigensolver(
                system.eigenproblem_solver, backend.device_operator
            )
            system.eigenproblem_solver = recording
            run_scf(system)
            first_run_paths = tuple(recording.paths)
            recording.paths.clear()
            run_scf(system)
            second_run_paths = tuple(recording.paths)

        self.assertEqual(first_run_paths, ("chebff", "subspace"))
        self.assertEqual(second_run_paths, ("chebff", "subspace"))


if __name__ == "__main__":
    unittest.main()
