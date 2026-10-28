"""Periodic nonlocal forces: finite differences of ``E_NL = sum_k w_k 2 sum_n f_n sum_q s_q |<beta_q^k|u_n>|^2``.

The force formula is the exact derivative of E_NL for any orbitals (Hellmann-Feynman only needs the orbitals to be
eigenstates for the total energy), so random orbitals test it fully.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Forces import build_periodic_projector_gradients, nonlocal_forces
from parsec_python.Grid import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import build_nonlocal_projectors

_GE = Path(__file__).resolve().parents[1] / "data" / "Ge_POTRE.DAT"
_H = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "H_POTRE.DAT"
_CELLS = {
    "rectangular": np.diag([7.7, 8.4, 9.1]),
    "fcc": 6.6 * (np.ones((3, 3)) - np.eye(3)),
}
_FRACTIONAL = np.array([[0.13, 0.27, 0.31], [0.62, 0.45, 0.78]])
_OCCUPATIONS = np.array([1.0, 0.6, 0.2])


class _Case:
    def __init__(self, name: str, use_spline: bool, k_point=None, species: str = "Ge") -> None:
        self.lattice = _CELLS[name]
        self.grid = build_periodic_grid(PeriodicCell(self.lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6))
        path, local = (_GE, 2) if species == "Ge" else (_H, 0)  # Ge: s and p are nonlocal; H: p and d
        self.symbol = species
        self.potentials = {species: read_parsec_pseudopotential(path)}
        self.specifications = {species: SpeciesPotential(path, local, use_spline=use_spline)}
        self.positions = _FRACTIONAL @ self.lattice
        self.k_point = None if k_point is None else np.asarray(k_point, dtype=float)
        generator = np.random.default_rng(11)
        shape = (self.grid.size, len(_OCCUPATIONS))
        orbitals = generator.normal(size=shape)
        if k_point is not None:
            orbitals = orbitals + 1j * generator.normal(size=shape)
        self.orbitals = orbitals / np.linalg.norm(orbitals, axis=0)
        self.weight = 1.0 if k_point is None else 0.35  # a k-point weight

    def atoms(self, positions=None) -> list[Atom]:
        positions = self.positions if positions is None else positions
        return [Atom(self.symbol, np.array(position)) for position in positions]

    def operator(self, positions=None):
        return build_nonlocal_projectors(
            self.grid, self.atoms(positions), self.potentials, self.specifications, self.lattice, self.k_point
        )

    def forces(self) -> np.ndarray:
        gradients = build_periodic_projector_gradients(
            self.grid, self.atoms(), self.potentials, self.specifications, self.lattice, self.k_point
        )
        return nonlocal_forces(self.operator(), gradients, self.orbitals, self.weight * _OCCUPATIONS, len(self.positions))

    def energy(self, positions: np.ndarray) -> float:
        operator = self.operator(positions)
        overlaps = np.asarray(operator.projectors.conj().T @ self.orbitals)
        return float(2.0 * self.weight * np.sum(operator.signs[:, None] * np.abs(overlaps) ** 2 * _OCCUPATIONS[None, :]))

    def finite_difference(self, atom: int, axis: int, step: float) -> float:
        """Central difference, Richardson-extrapolated from ``step`` and ``2*step`` (error of order ``step**4``)."""

        def central(h: float) -> float:
            plus, minus = self.positions.copy(), self.positions.copy()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            return -(self.energy(plus) - self.energy(minus)) / (2.0 * h)

        return (4.0 * central(step) - central(2.0 * step)) / 3.0


class PeriodicNonlocalForceTests(unittest.TestCase):
    def _check(self, case: _Case, relative: float) -> None:
        forces = case.forces()
        scale = float(np.max(np.abs(forces)))
        self.assertGreater(scale, 1.0e-3)
        for atom, axis in ((0, 0), (0, 2), (1, 1), (1, 2)):
            numeric = case.finite_difference(atom, axis, 1.0e-3)
            self.assertAlmostEqual(float(forces[atom, axis]), numeric, delta=relative * scale)

    def test_gamma_force_is_the_derivative_of_the_nonlocal_energy(self) -> None:
        for name in _CELLS:
            # with the spline the force is the exact derivative; the linear mode uses PARSEC's derivative table
            for use_spline, relative in ((True, 1.0e-7), (False, 3.0e-2)):
                with self.subTest(cell=name, spline=use_spline):
                    self._check(_Case(name, use_spline), relative)

    def test_bloch_force_is_the_derivative_of_the_nonlocal_energy(self) -> None:
        for name in _CELLS:
            with self.subTest(cell=name):
                self._check(_Case(name, True, k_point=(0.21, -0.37, 0.11)), 1.0e-7)

    def test_a_zero_k_point_reproduces_the_gamma_force(self) -> None:
        case = _Case("fcc", True)
        gamma = case.forces()
        case.k_point = np.zeros(3)  # the complex code path
        np.testing.assert_allclose(case.forces(), gamma, atol=1.0e-12)

    def test_gradient_columns_of_an_atom_on_a_grid_point(self) -> None:
        # The first samples of a POTRE table are rounding noise: u_l(r)/r is constant there instead of growing as r**l,
        # so the gradient of a p projector at the grid point on top of its atom must not be taken at the first sample
        # (it was 800 times too large there).  The radial functions are smooth from 1e-3 bohr on.
        # The projector itself is not well defined at that point (the direction of a zero displacement), so the energy
        # has a small jump when an atom crosses a grid point and only the columns are compared with finite differences.
        for name in _CELLS:
            with self.subTest(cell=name):
                case = _Case(name, True, species="H")
                steps = case.grid.step_vectors
                case.positions = np.round(np.array([[0.2, 0.3, 0.5], [0.6, 0.8, 0.1]]) * case.grid.points_per_axis) @ steps
                self.assertLess(float(np.min(np.linalg.norm(case.grid.coordinates - case.positions[0], axis=1))), 1.0e-9)
                gradients = build_periodic_projector_gradients(
                    case.grid, case.atoms(), case.potentials, case.specifications, case.lattice
                )
                for axis in range(3):

                    def operator(shift: float, axis=axis):
                        positions = case.positions.copy()
                        positions[0, axis] += shift
                        return case.operator(positions).projectors.tocsc()

                    def central(step: float, axis=axis):
                        return (operator(step) - operator(-step)) / (2.0 * step)

                    derivative = ((4.0 * central(1.0e-3) - central(2.0e-3)) / 3.0).toarray()
                    analytic = -gradients.components[axis].toarray()
                    columns = [q for q, label in enumerate(gradients.labels) if label[0] == 0]
                    largest = float(np.max(np.abs(analytic[:, columns])))
                    self.assertGreater(largest, 1.0)
                    np.testing.assert_allclose(derivative[:, columns], analytic[:, columns], atol=1.0e-4 * largest)


    def test_projectors_that_cross_the_cell_boundary(self) -> None:
        # atoms at the faces of the cell: every projector is a sum over images
        for name, k_point in (("rectangular", None), ("fcc", (0.21, -0.37, 0.11))):
            with self.subTest(cell=name):
                case = _Case(name, True, k_point=k_point)
                case.positions = np.array([[0.02, 0.5, 0.97], [0.5, 0.98, 0.03]]) @ case.lattice
                self._check(case, 1.0e-7)


if __name__ == "__main__":
    unittest.main()
