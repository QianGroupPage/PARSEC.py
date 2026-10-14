"""Tests for the correctness-first CuPy complex spinor Hamiltonian.

The ``apply_lzsz``/``apply_lsxy``/``_projector_overlaps`` helpers in
``acceleration.backends.cupy_spinor`` take the array module (``cp``) as an
explicit first argument specifically so they can be exercised with plain
``numpy`` in place of ``cupy`` on a machine with no GPU: every operation they
use (``asarray``, ``zeros_like``, ``where``, ``outer``, ``concatenate``,
fancy-indexed ``+=``) exists identically in both libraries, so calling them
with ``numpy`` runs the *exact* code a real GPU would run and lets it be
checked against the already-validated reference
(:mod:`parsec_python.Hamiltonian.spinor_operator`) without any CUDA runtime.
Only ``CuPySpinorHamiltonian`` itself (which calls ``require_cupy()``) needs
a real GPU and is skip-gated accordingly.
"""

from __future__ import annotations

from pathlib import Path
import unittest

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
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Hamiltonian import spinor_operator as reference_spinor_operator
from parsec_python.SCF.single_point import prepare_single_point

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.cupy_spinor import (
    _DeviceSOCProjector,
    apply_lsxy,
    apply_lzsz,
)

GPU_AVAILABLE = cupy_available()

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _au_soc_projectors():
    problem = SinglePointInput(
        atoms=[Atom("Au", [0.0, 0.0, 0.0])],
        pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
        scf=SCFSettings(max_iterations=1, number_of_states=10),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )
    system = prepare_single_point(problem)
    projectors = tuple(
        build_spin_orbit_projectors(
            system.grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials
        )
    )
    return system.grid.size, projectors


class NumpyAsCupyLzszLsxyTests(unittest.TestCase):
    """Runs cupy_spinor's apply_lzsz/apply_lsxy with numpy standing in for
    cupy, and checks the result against the independently-written reference
    implementation on the same random complex wavefunctions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.n_grid, cls.projectors = _au_soc_projectors()

    def _device_projectors(self):
        return tuple(_DeviceSOCProjector(np, atom) for atom in self.projectors)

    def test_apply_lzsz_matches_reference(self) -> None:
        rng = np.random.default_rng(13)
        wavefunctions = rng.normal(size=(self.n_grid, 4)) + 1j * rng.normal(
            size=(self.n_grid, 4)
        )
        device_projectors = self._device_projectors()
        for spin_sign in (1, -1):
            mine = apply_lzsz(np, device_projectors, wavefunctions, spin_sign)
            reference = reference_spinor_operator.apply_lzsz(
                self.projectors, wavefunctions, spin_sign
            )
            np.testing.assert_allclose(mine, reference, atol=1e-12)

    def test_apply_lsxy_matches_reference(self) -> None:
        rng = np.random.default_rng(29)
        wavefunctions = rng.normal(size=(self.n_grid, 4)) + 1j * rng.normal(
            size=(self.n_grid, 4)
        )
        device_projectors = self._device_projectors()
        for spin_sign in (1, -1):
            mine = apply_lsxy(np, device_projectors, wavefunctions, spin_sign)
            reference = reference_spinor_operator.apply_lsxy(
                self.projectors, wavefunctions, spin_sign
            )
            np.testing.assert_allclose(mine, reference, atol=1e-12)

    def test_empty_projectors_return_zero(self) -> None:
        wavefunctions = np.zeros((self.n_grid, 2), dtype=complex)
        self.assertTrue(
            np.all(apply_lzsz(np, (), wavefunctions, 1) == 0)
        )
        self.assertTrue(
            np.all(apply_lsxy(np, (), wavefunctions, 1) == 0)
        )

    def test_single_vector_1d_roundtrip(self) -> None:
        rng = np.random.default_rng(5)
        vector = rng.normal(size=self.n_grid) + 1j * rng.normal(size=self.n_grid)
        device_projectors = self._device_projectors()
        result = apply_lzsz(np, device_projectors, vector, 1)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape, (self.n_grid,))


@unittest.skipUnless(GPU_AVAILABLE, "CuPy/CUDA are not available")
class CuPySpinorHamiltonianGPUTests(unittest.TestCase):
    """Full end-to-end GPU checks: only run on a real CUDA device."""

    @classmethod
    def setUpClass(cls) -> None:
        problem = SinglePointInput(
            atoms=[Atom("Au", [0.0, 0.0, 0.0])],
            pseudopotentials={"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)},
            grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
            scf=SCFSettings(max_iterations=1, number_of_states=10),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        cls.system = prepare_single_point(problem)
        cls.soc_projectors = tuple(
            build_spin_orbit_projectors(
                cls.system.grid,
                cls.system.atoms,
                cls.system.pseudopotentials,
                cls.system.input.pseudopotentials,
            )
        )

    def _build_cpu_hamiltonian(self, veff):
        from parsec_python.Hamiltonian.operator import KohnShamHamiltonian

        scalar = self.system.hamiltonian(veff)
        return reference_spinor_operator.SpinorKohnShamHamiltonian(
            scalar.negative_laplacian,
            scalar.effective_potential,
            scalar.nonlocal_operator,
            self.soc_projectors,
        )

    def test_matches_reference_hamiltonian(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.backends.cupy_spinor import (
            CuPySpinorHamiltonian,
        )

        veff = self.system.ionic_potential
        cpu_hamiltonian = self._build_cpu_hamiltonian(veff)
        scalar = self.system.hamiltonian(veff)
        gpu_hamiltonian = CuPySpinorHamiltonian(
            scalar.negative_laplacian,
            scalar.effective_potential,
            scalar.nonlocal_operator,
            self.soc_projectors,
        )
        rng = np.random.default_rng(3)
        n = 2 * self.system.grid.size
        trial = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))

        expected = cpu_hamiltonian.apply(trial)
        actual = cp.asnumpy(gpu_hamiltonian.apply(cp.asarray(trial)))
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_hermitian(self) -> None:
        import cupy as cp

        from parsec_python.acceleration.backends.cupy_spinor import (
            CuPySpinorHamiltonian,
        )

        veff = self.system.ionic_potential
        scalar = self.system.hamiltonian(veff)
        gpu_hamiltonian = CuPySpinorHamiltonian(
            scalar.negative_laplacian,
            scalar.effective_potential,
            scalar.nonlocal_operator,
            self.soc_projectors,
        )
        rng = np.random.default_rng(17)
        n = gpu_hamiltonian.shape[0]
        u = cp.asarray(rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4)))
        v = cp.asarray(rng.normal(size=(n, 4)) + 1j * rng.normal(size=(n, 4)))
        lhs = cp.vdot(u, gpu_hamiltonian.apply(v))
        rhs = cp.vdot(gpu_hamiltonian.apply(u), v)
        self.assertLess(abs(complex(lhs) - complex(rhs)), 1e-8)


if __name__ == "__main__":
    unittest.main()
