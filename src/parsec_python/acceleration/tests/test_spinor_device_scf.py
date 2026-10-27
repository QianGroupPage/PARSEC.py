"""The device self-consistent SOC loop (run here with NumPy as the array backend).

``run_self_consistent_soc`` of ``acceleration/SCF/self_consistent_soc.py`` takes an array
backend, so the code the GPU runs executes on the host in these tests and is checked
against the reference solver and against exact diagonalization.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Eigensolvers.eigval import EigvalSettings
from parsec_python.Eigensolvers.chebff import ChebFFSettings
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Eigensolvers.subspace import SubspaceSettings
from parsec_python.SCF.self_consistent_soc import run_self_consistent_soc as run_reference
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.acceleration.Eigensolvers.chebff_spinor import (
    solve_spinor_eigval,
    spinor_density_on_device,
)
from parsec_python.acceleration.Hamiltonian.device_bloch import NumpyArrayBackend
from parsec_python.acceleration.SCF.self_consistent_soc import run_self_consistent_soc

_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"


def _problem() -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("Au", [0.0, 0.0, 0.0])],
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
        scf=SCFSettings(max_iterations=40, number_of_states=20, convergence_criterion=1.0e-4),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class _Dense:
    """A Hermitian matrix as a device operator (``shape`` and ``apply``)."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.shape = matrix.shape

    def apply(self, block):
        return self.matrix @ block


class DeviceSelfConsistentSOCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        problem = _problem()
        cls.system = prepare_single_point(problem)
        cls.projectors = tuple(
            build_spin_orbit_projectors(
                cls.system.grid, cls.system.atoms, cls.system.pseudopotentials, problem.pseudopotentials
            )
        )
        cls.reference = run_reference(cls.system, cls.projectors)
        cls.device = run_self_consistent_soc(cls.system, cls.projectors, backend=NumpyArrayBackend())

    def test_converges_to_the_reference_energy(self) -> None:
        self.assertTrue(self.reference.converged)
        self.assertTrue(self.device.converged)
        self.assertAlmostEqual(self.device.energies.total, self.reference.energies.total, delta=5.0e-5)
        np.testing.assert_allclose(
            self.device.eigenvalues[:16], self.reference.eigenvalues[:16], atol=5.0e-4
        )

    def test_spinors_come_back_to_the_host_normalized(self) -> None:
        self.assertIsInstance(self.device.spinors, np.ndarray)
        norms = np.sum(np.abs(self.device.spinors) ** 2, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-8)

    def test_first_iteration_is_chebff_and_later_ones_are_subspace_steps(self) -> None:
        timings = self.device.timings
        self.assertEqual(timings["eig_first_solves"], 1)
        self.assertEqual(timings["eig_subspace_steps"], self.device.iterations - 1)

    def test_stage_timings_are_recorded(self) -> None:
        for stage in ("hamiltonian", "diagonalization", "density", "hartree", "xc", "mixing_energy"):
            self.assertIn(stage, self.device.timings)
            self.assertGreaterEqual(self.device.timings[stage], 0.0)
        self.assertIsNotNone(self.reference.timings)

    def test_later_iterations_apply_far_fewer_vectors_than_a_first_solve(self) -> None:
        # a first solve is filter_cycles x degree applications of the whole subspace; a SUBSPACE
        # step is one filter of about half that degree: the total must stay well below
        # (iterations x first solve)
        timings = self.device.timings
        first = 4 * 30  # default filter cycles x first-filter degree, plus the Ritz steps
        self.assertLess(timings["eig_applications"], 0.5 * self.device.iterations * first)

    def test_device_density_matches_the_host_formula(self) -> None:
        rng = np.random.default_rng(3)
        dimension, states = 2 * 50, 7
        spinors = rng.normal(size=(dimension, states)) + 1j * rng.normal(size=(dimension, states))
        occupations = np.array([1.0, 1.0, 0.7, 0.2, 0.0, 0.0, 0.0])
        expected = (
            np.sum((np.abs(spinors[:50]) ** 2 + np.abs(spinors[50:]) ** 2) * occupations[None, :], axis=1)
            / 0.3
        )
        np.testing.assert_allclose(spinor_density_on_device(np, spinors, occupations, 0.3), expected, rtol=1e-12)


class SpinorEigvalTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        size = 160
        raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        self.matrix = (raw + raw.conj().T) / 2.0
        self.exact = np.linalg.eigvalsh(self.matrix)
        self.operator = _Dense(self.matrix)
        self.settings = EigvalSettings(
            safety_buffer=4,
            initial_method="chebff",
            chebff=ChebFFSettings(polynomial_degree=24, filter_cycles=4, lanczos_steps=10, block_size=4),
            subspace=SubspaceSettings(polynomial_degree=14, degree_delta=3, block_size=4),
        )

    def test_subspace_step_keeps_exact_ritz_vectors(self) -> None:
        values, vectors = np.linalg.eigh(self.matrix)
        from parsec_python.acceleration.Eigensolvers.chebff_spinor import SpinorEigvalState

        working = 10
        state = SpinorEigvalState(
            vectors=np.ascontiguousarray(vectors[:, :working]),
            eigenvalues=values[:working].copy(),
            working_states=working,
            operator_dimension=self.matrix.shape[0],
        )
        result = solve_spinor_eigval(self.operator, 6, settings=self.settings, state=state, xp=np)
        np.testing.assert_allclose(result.eigenvalues, values[:6], atol=1e-8)
        self.assertEqual(result.stats["subspace_steps"], 1)

    def test_warm_started_steps_converge_to_the_lowest_states(self) -> None:
        first = solve_spinor_eigval(self.operator, 6, settings=self.settings, xp=np)
        self.assertEqual(first.stats["first_solves"], 1)
        state = first.state
        for _ in range(6):
            step = solve_spinor_eigval(self.operator, 6, settings=self.settings, state=state, xp=np)
            state = step.state
        np.testing.assert_allclose(step.eigenvalues, self.exact[:6], atol=1e-5)

    def test_converged_first_solve_reaches_the_residual_tolerance(self) -> None:
        result = solve_spinor_eigval(
            self.operator, 6, settings=self.settings, xp=np, converge_first=True, tolerance=1.0e-6
        )
        scale = float(np.max(np.abs(result.eigenvalues)))
        self.assertLessEqual(result.residual_norms.max(), 1.0e-6 * scale)
        np.testing.assert_allclose(result.eigenvalues, self.exact[:6], atol=1e-5)
        self.assertGreaterEqual(result.stats.get("extra_cycles", 0), 0)
        # and the state continues as a subspace state
        step = solve_spinor_eigval(self.operator, 6, settings=self.settings, state=result.state, xp=np)
        self.assertEqual(step.stats["subspace_steps"], 1)
        np.testing.assert_allclose(step.eigenvalues, self.exact[:6], atol=1e-5)

    def test_extra_cycles_improve_an_unconverged_chebff_solve(self) -> None:
        loose = EigvalSettings(
            safety_buffer=4,
            initial_method="chebff",
            chebff=ChebFFSettings(polynomial_degree=6, filter_cycles=1, lanczos_steps=10, block_size=4),
            subspace=SubspaceSettings(polynomial_degree=14, degree_delta=3, block_size=4),
        )
        plain = solve_spinor_eigval(self.operator, 6, settings=loose, xp=np)
        refined = solve_spinor_eigval(
            self.operator, 6, settings=loose, xp=np, converge_first=True, tolerance=1.0e-5, max_extra_cycles=40
        )
        error = lambda r: float(np.max(np.abs(r.eigenvalues - self.exact[:6])))
        self.assertLess(error(refined), error(plain))
        self.assertGreater(refined.stats["extra_cycles"], 0)

    def test_states_are_orthonormal(self) -> None:
        first = solve_spinor_eigval(self.operator, 6, settings=self.settings, xp=np)
        basis = first.state.vectors
        np.testing.assert_allclose(basis.conj().T @ basis, np.eye(basis.shape[1]), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
