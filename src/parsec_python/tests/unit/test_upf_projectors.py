"""The nonlocal operator of a UPF potential (several projectors per l): the gradient builders against finite differences.

For fixed orbitals the nonlocal energy is ``2 sum_k sign_k |<beta_k|psi>|^2``; its derivative with respect to an atom is the
nonlocal force of ``nonlocal_forces``.  Every builder that loops over the projectors is checked on a synthetic potential
(two projectors for s and p with a non-diagonal D, one for d): the isolated-cluster gradients, the double grid, the
accelerated double grid, the periodic gradients, and the Fourier-filtered potential.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Forces import build_nonlocal_projector_gradients, build_periodic_projector_gradients, nonlocal_forces
from parsec_python.Forces.wire_slab import wire_slab_projector_gradients
from parsec_python.Forces.nonlocal_force import double_grid_projector_gradients
from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.Grid.confined import build_confined_periodic_grid, confined_lattice_vectors
from parsec_python.Pseudopotential.radial_derivatives import radial_projector_with_derivative
from parsec_python.Pseudopotential.upf import read_upf_pseudopotential
from parsec_python.V_ion import apply_fourier_filters, build_nonlocal_projectors, load_pseudopotentials
from parsec_python.V_ion.double_grid import build_double_grid_projectors
from parsec_python.models import Atom, GridSettings, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.tests.unit.test_upf import _upf_text

_STEP = 1.0e-3


def _moved(atoms, index, axis, step):
    shift = np.zeros(3)
    shift[axis] = step
    return tuple(Atom(a.symbol, a.position + shift) if i == index else a for i, a in enumerate(atoms))


class UpfProjectorGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        path = Path(cls._directory.name) / "Si.upf"
        path.write_text(_upf_text(element="Si"), encoding="utf-8")
        cls.specifications = {"Si": SpeciesPotential(path, -1, pseudopotential_format="upf", read_valence_density=True, use_spline=True)}
        cls.potentials = load_pseudopotentials(cls.specifications, xc_functional="pbe")
        cls.grid = build_cluster_grid(GridSettings(spacing=0.4, radius=6.0))
        cls.atoms = (Atom("Si", np.array([-1.0, 0.3, 0.1])), Atom("Si", np.array([1.3, -0.4, 0.5])))
        rng = np.random.default_rng(5)
        r2 = np.einsum("ij,ij->i", cls.grid.coordinates, cls.grid.coordinates)
        envelope = np.exp(-0.15 * r2)
        orbitals = np.column_stack([envelope * (1.0 + 0.3 * cls.grid.coordinates @ rng.normal(size=3)) for _ in range(3)])
        cls.orbitals = orbitals / np.linalg.norm(orbitals, axis=0)
        cls.occupations = np.array([1.0, 0.6, 0.2])

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def energy(self, operator) -> float:
        overlaps = operator.projectors.T @ self.orbitals
        return 2.0 * float(np.sum(operator.signs[:, None] * overlaps**2 * self.occupations[None, :]))

    def check(self, build, gradients, potentials=None, *, rtol=2.0e-3, atol=5.0e-6) -> None:
        potentials = self.potentials if potentials is None else potentials
        operator = build(self.atoms)
        # five columns per l=0,1 pair of projectors... : every projector has its own columns and its own label
        self.assertEqual(len(operator.labels), len(set(operator.labels)))
        self.assertEqual(operator.projectors.shape[1], 2 * (2 * 1 + 2 * 3 + 5))
        grad = gradients(self.atoms)
        self.assertEqual(grad.labels, operator.labels)
        forces = nonlocal_forces(operator, grad, self.orbitals, self.occupations, len(self.atoms))
        self.assertGreater(np.abs(forces).max(), 1.0e-3)
        for index in range(len(self.atoms)):
            for axis in range(3):
                numerical = -(self.energy(build(_moved(self.atoms, index, axis, _STEP)))
                              - self.energy(build(_moved(self.atoms, index, axis, -_STEP)))) / (2.0 * _STEP)
                self.assertAlmostEqual(forces[index, axis], numerical, delta=atol + rtol * abs(numerical))

    def test_the_gradients_of_an_isolated_cluster(self) -> None:
        self.check(
            lambda atoms: build_nonlocal_projectors(self.grid, atoms, self.potentials, self.specifications),
            lambda atoms: build_nonlocal_projector_gradients(self.grid, atoms, self.potentials, self.specifications),
        )

    def test_the_gradients_of_the_double_grid(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=6.0, double_grid_order=2))
        self.check(
            lambda atoms: build_double_grid_projectors(grid, atoms, self.potentials, self.specifications, order=2),
            lambda atoms: double_grid_projector_gradients(grid, atoms, self.potentials, self.specifications, order=2),
        )

    def test_the_accelerated_double_grid_is_the_host_one(self) -> None:
        from parsec_python.acceleration.V_ion import double_grid as fast

        host = build_double_grid_projectors(self.grid, self.atoms, self.potentials, self.specifications, order=2)
        accelerated = fast.build_double_grid_projectors(self.grid, self.atoms, self.potentials, self.specifications, order=2)
        self.assertEqual(host.labels, accelerated.labels)
        np.testing.assert_array_equal(host.signs, accelerated.signs)
        self.assertLess(abs(host.projectors - accelerated.projectors).max(), 1.0e-12)
        host_gradients = double_grid_projector_gradients(self.grid, self.atoms, self.potentials, self.specifications, order=2)
        fast_gradients = fast.build_double_grid_projector_gradients(self.grid, self.atoms, self.potentials, self.specifications, order=2)
        self.assertEqual(host_gradients.labels, fast_gradients.labels)
        for left, right in zip(host_gradients.components, fast_gradients.components):
            self.assertLess(abs(left - right).max(), 1.0e-12)

    def test_the_fourier_filtered_projectors_and_their_slopes(self) -> None:
        specification = SpeciesPotential(
            self.specifications["Si"].path, -1, pseudopotential_format="upf", read_valence_density=True, use_spline=True,
            fourier_filter_alpha=0.8, fourier_filter_beta1=5.0,
        )
        specifications = {"Si": specification}
        filtered = apply_fourier_filters(self.potentials, specifications, 0.4)
        before, after = self.potentials["Si"], filtered["Si"]
        self.assertEqual(len(after.kb_channels), len(before.kb_channels))
        for old, new in zip(before.kb_channels, after.kb_channels):
            self.assertEqual((old.angular_momentum, old.index, old.sign), (new.angular_momentum, new.index, new.sign))
            self.assertIsNotNone(new.slopes)
            self.assertFalse(np.array_equal(old.values, new.values))
            # close to the original away from the core
            self.assertLess(np.max(np.abs(old.values - new.values)), 0.3 * np.max(np.abs(old.values)))
        for angular_momentum, index in after.nonlocal_channels(-1):
            table, slope, sign = radial_projector_with_derivative(after, angular_momentum, -1, index)
            np.testing.assert_array_equal(slope, after.projector_slopes(angular_momentum, index))
        self.check(
            lambda atoms: build_nonlocal_projectors(self.grid, atoms, filtered, specifications),
            lambda atoms: build_nonlocal_projector_gradients(self.grid, atoms, filtered, specifications),
            filtered,
        )

    def _periodic_energy_check(self, grid, lattice, atoms, build, gradients, k_point=None, rtol=2.0e-3, atol=5.0e-6) -> None:
        """The same check on a grid with orbitals of random (complex at a k-point) values."""

        rng = np.random.default_rng(11)
        shape = (grid.size, len(self.occupations))
        orbitals = rng.normal(size=shape)
        if k_point is not None:
            orbitals = orbitals + 1j * rng.normal(size=shape)
        orbitals = orbitals / np.linalg.norm(orbitals, axis=0)

        def energy(positions):
            operator = build(positions)
            overlaps = np.asarray(operator.projectors.conj().T @ orbitals)
            return 2.0 * float(np.sum(operator.signs[:, None] * np.abs(overlaps) ** 2 * self.occupations[None, :]))

        operator = build(atoms)
        self.assertEqual(len(operator.labels), len(set(operator.labels)))
        grad = gradients(atoms)
        self.assertEqual(grad.labels, operator.labels)
        forces = nonlocal_forces(operator, grad, orbitals, self.occupations, len(atoms))
        self.assertGreater(np.abs(forces).max(), 1.0e-3)
        for index in range(len(atoms)):
            for axis in range(3):
                numerical = -(energy(_moved(atoms, index, axis, _STEP)) - energy(_moved(atoms, index, axis, -_STEP))) / (2.0 * _STEP)
                self.assertAlmostEqual(forces[index, axis], numerical, delta=atol + rtol * abs(numerical))

    def test_the_gradients_of_a_periodic_cell_at_gamma_and_at_a_k_point(self) -> None:
        lattice = np.diag([7.0, 7.0, 7.0])
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.5, expansion_order=6))
        atoms = (Atom("Si", np.array([1.1, 2.0, 2.4])), Atom("Si", np.array([3.9, 4.6, 5.3])))
        for k_point in (None, np.array([0.11, 0.05, 0.2])):
            with self.subTest(k_point=k_point):
                self._periodic_energy_check(
                    grid, lattice, atoms,
                    lambda moved: build_nonlocal_projectors(grid, moved, self.potentials, self.specifications, lattice, k_point),
                    lambda moved: build_periodic_projector_gradients(grid, moved, self.potentials, self.specifications, lattice, k_point),
                    k_point,
                )

    def test_the_gradients_of_a_slab(self) -> None:
        vectors = np.array([[1.0, 0.0, 0.0], [0.5, 0.8660254037844386, 0.0]]) * 6.5
        grid = build_confined_periodic_grid(
            PeriodicCell(confined_lattice_vectors(vectors)),
            PeriodicGridSettings(spacing=0.5, shift=(0.0, 0.0, 0.5), expansion_order=6, periodicity=2, confinement_radius=5.0),
        )
        lattice = confined_lattice_vectors(vectors)
        atoms = (Atom("Si", np.array([0.4, 0.7, -0.6])), Atom("Si", np.array([2.9, 1.6, 0.9])))
        self._periodic_energy_check(
            grid, lattice, atoms,
            lambda moved: build_nonlocal_projectors(grid, moved, self.potentials, self.specifications, lattice),
            lambda moved: wire_slab_projector_gradients(grid, moved, self.potentials, self.specifications),
        )


if __name__ == "__main__":
    unittest.main()
