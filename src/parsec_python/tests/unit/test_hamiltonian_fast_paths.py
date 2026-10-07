"""The optimized Hamiltonian applications must equal the straightforward ones.

``apply_spin_orbit`` (one GEMM overlap and one scatter per atom for both spin
halves) is checked against the per-harmonic ``apply_lzsz``/``apply_lsxy``
transcription of ``zmatvec_so.f90``; ``KPointKohnShamHamiltonian.apply`` (real
sparse operators on the real view of the complex block, pre-summed
``k.grad``) against the textbook
``-nabla^2 - 2i k.grad + |k|^2 + V + V_NL`` written with plain complex
products.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python import (
    Atom,
    GridSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SpeciesPotential,
)
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hamiltonian.kpoint_spinor_operator import KPointSpinorKohnShamHamiltonian
from parsec_python.Hamiltonian.spinor_operator import (
    SpinorKohnShamHamiltonian,
    apply_lsxy,
    apply_lzsz,
    apply_spin_orbit,
    apply_zeeman_field,
)
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _random_spinor(n_grid: int, n_vectors: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (2 * n_grid, n_vectors)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _reference_spin_orbit(projectors, spinor: np.ndarray) -> np.ndarray:
    n_grid = spinor.shape[0] // 2
    up, down = spinor[:n_grid], spinor[n_grid:]
    q_up = apply_lzsz(projectors, up, spin_sign=1) + apply_lsxy(
        projectors, down, spin_sign=-1
    )
    q_down = apply_lzsz(projectors, down, spin_sign=-1) + apply_lsxy(
        projectors, up, spin_sign=1
    )
    return np.concatenate([q_up, q_down], axis=0)


class _PeriodicFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        box = 9.0
        cls.cell = PeriodicCell(lattice_vectors=box * np.eye(3))
        cls.grid = build_periodic_grid(
            cls.cell, PeriodicGridSettings(spacing=0.75, expansion_order=6)
        )
        cls.atoms = [
            Atom("Au", [0.4 * box, 0.5 * box, 0.5 * box]),
            Atom("Au", [0.7 * box, 0.45 * box, 0.55 * box]),
        ]
        cls.specs = {"Au": SpeciesPotential(_AU_POTRE, 0, spin_orbit=True)}
        cls.pots = load_pseudopotentials(cls.specs, xc_functional="ca")
        cls.k_point = np.array([0.21, -0.13, 0.08])
        cls.soc = tuple(
            build_spin_orbit_projectors(
                cls.grid,
                cls.atoms,
                cls.pots,
                cls.specs,
                cls.cell.lattice_vectors,
                k_point=cls.k_point,
            )
        )
        cls.nonlocal_k = build_nonlocal_projectors(
            cls.grid,
            cls.atoms,
            cls.pots,
            cls.specs,
            cls.cell.lattice_vectors,
            k_point=cls.k_point,
        )
        cls.laplacian = build_negative_laplacian(cls.grid)
        cls.gradient = build_gradient(cls.grid)
        rng = np.random.default_rng(3)
        cls.potential = 0.3 * rng.normal(size=cls.grid.size)


class FusedSpinOrbitTests(_PeriodicFixture):
    def test_matches_per_harmonic_reference_periodic(self) -> None:
        self.assertTrue(self.soc)
        for n_vectors in (1, 5):
            spinor = _random_spinor(self.grid.size, n_vectors, seed=n_vectors)
            fused = apply_spin_orbit(self.soc, spinor)
            reference = _reference_spin_orbit(self.soc, spinor)
            self.assertGreater(np.abs(reference).max(), 1e-3)
            np.testing.assert_allclose(fused, reference, atol=1e-12)

    def test_matches_per_harmonic_reference_cluster(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.6, radius=3.0, expansion_order=4)
        )
        atoms = [Atom("Au", [0.0, 0.0, 0.0])]
        projectors = build_spin_orbit_projectors(grid, atoms, self.pots, self.specs)
        spinor = _random_spinor(grid.size, 4, seed=9)
        np.testing.assert_allclose(
            apply_spin_orbit(projectors, spinor),
            _reference_spin_orbit(projectors, spinor),
            atol=1e-12,
        )

    def test_input_is_not_modified(self) -> None:
        spinor = _random_spinor(self.grid.size, 3, seed=2)
        before = spinor.copy()
        apply_spin_orbit(self.soc, spinor)
        np.testing.assert_array_equal(spinor, before)

    def test_no_projectors_gives_zero(self) -> None:
        spinor = _random_spinor(self.grid.size, 2, seed=1)
        self.assertTrue(np.all(apply_spin_orbit((), spinor) == 0.0))


class KPointKineticTests(_PeriodicFixture):
    def _reference_scalar(self, k_point, vectors: np.ndarray) -> np.ndarray:
        result = self.laplacian.astype(complex) @ vectors
        for axis in range(3):
            result = result - 2j * k_point[axis] * (self.gradient[axis] @ vectors)
        result = result + float(np.dot(k_point, k_point)) * vectors
        result = result + self.potential[:, None] * vectors
        result = result + self.nonlocal_k.apply(vectors)
        return result

    def test_matches_textbook_form_at_nonzero_k(self) -> None:
        hamiltonian = KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.potential, self.nonlocal_k, self.k_point
        )
        block = _random_spinor(self.grid.size, 5, seed=4)[: self.grid.size]
        np.testing.assert_allclose(
            hamiltonian.apply(block),
            self._reference_scalar(self.k_point, block),
            atol=1e-11,
        )

    def test_single_vector_and_noncontiguous_input(self) -> None:
        hamiltonian = KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.potential, self.nonlocal_k, self.k_point
        )
        spinor = _random_spinor(self.grid.size, 6, seed=5)
        # A strided column and a Fortran-ordered block exercise the
        # contiguity handling of the real-view fast path.
        column = spinor[: self.grid.size, 2]
        np.testing.assert_allclose(
            hamiltonian.apply(column),
            self._reference_scalar(self.k_point, column[:, None])[:, 0],
            atol=1e-11,
        )
        fortran_block = np.asfortranarray(spinor[: self.grid.size, :3])
        np.testing.assert_allclose(
            hamiltonian.apply(fortran_block),
            self._reference_scalar(self.k_point, fortran_block),
            atol=1e-11,
        )

    def test_gamma_point_has_no_gradient_term(self) -> None:
        zero = np.zeros(3)
        hamiltonian = KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.potential, self.nonlocal_k, zero
        )
        block = _random_spinor(self.grid.size, 3, seed=6)[: self.grid.size]
        np.testing.assert_allclose(
            hamiltonian.apply(block), self._reference_scalar(zero, block), atol=1e-11
        )

    def test_full_noncollinear_spinor_hamiltonian(self) -> None:
        scalar = KPointKohnShamHamiltonian(
            self.laplacian, self.gradient, self.potential, self.nonlocal_k, self.k_point
        )
        rng = np.random.default_rng(8)
        field = 0.2 * rng.normal(size=(3, self.grid.size))
        hamiltonian = KPointSpinorKohnShamHamiltonian(scalar, self.soc, xc_field=field)
        spinor = _random_spinor(self.grid.size, 4, seed=7)
        n_grid = self.grid.size
        up, down = spinor[:n_grid], spinor[n_grid:]
        zeeman_up, zeeman_down = apply_zeeman_field(field, up, down)
        reference = np.concatenate(
            [
                self._reference_scalar(self.k_point, up) + zeeman_up,
                self._reference_scalar(self.k_point, down) + zeeman_down,
            ],
            axis=0,
        ) + _reference_spin_orbit(self.soc, spinor)
        np.testing.assert_allclose(hamiltonian.apply(spinor), reference, atol=1e-10)

    def test_cluster_spinor_hamiltonian_matches_reference(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.6, radius=3.0, expansion_order=4)
        )
        atoms = [Atom("Au", [0.0, 0.0, 0.0])]
        projectors = tuple(
            build_spin_orbit_projectors(grid, atoms, self.pots, self.specs)
        )
        nonlocal_operator = build_nonlocal_projectors(
            grid, atoms, self.pots, self.specs
        )
        laplacian = build_negative_laplacian(grid)
        potential = np.random.default_rng(1).normal(size=grid.size)
        hamiltonian = SpinorKohnShamHamiltonian(
            laplacian, potential, nonlocal_operator, projectors
        )
        spinor = _random_spinor(grid.size, 3, seed=11)
        up, down = spinor[: grid.size], spinor[grid.size :]

        def channel(vectors):
            return (
                laplacian @ vectors
                + potential[:, None] * vectors
                + nonlocal_operator.projectors
                @ (nonlocal_operator.signs[:, None] * (nonlocal_operator.projectors.T @ vectors))
            )

        reference = np.concatenate([channel(up), channel(down)]) + _reference_spin_orbit(
            projectors, spinor
        )
        np.testing.assert_allclose(hamiltonian.apply(spinor), reference, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
