"""Tests for PBE on periodic grids (wraparound gradient) and for periodic
self-consistent spin-orbit coupling run with PBE."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.cli import main as cli_main
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.V_xc import pbe, pbe_energy_partials
from parsec_python.V_xc.pbe import first_derivative_coefficients

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _grid(box: float = 7.0, spacing: float = 0.7, order: int = 6):
    cell = PeriodicCell(lattice_vectors=box * np.eye(3))
    return build_periodic_grid(
        cell, PeriodicGridSettings(spacing=spacing, expansion_order=order)
    )


def _gaussian_density(grid) -> np.ndarray:
    """Smooth periodic density: a Gaussian placed at the cell center."""
    center = 0.5 * np.diag(grid.cell.lattice_vectors)
    delta = grid.coordinates - center
    delta -= np.diag(grid.cell.lattice_vectors) * np.round(
        delta / np.diag(grid.cell.lattice_vectors)
    )
    return 0.2 * np.exp(-0.5 * np.einsum("ij,ij->i", delta, delta)) + 0.01


def _reference_pbe(density: np.ndarray, grid):
    """Independent implementation: np.roll stencil on the reshaped box."""
    weights = first_derivative_coefficients(grid.settings.expansion_order)
    weights = weights / grid.spacing
    width = weights.size // 2
    box = density.reshape(grid.shape)

    def derivative(field, axis):
        total = np.zeros_like(field)
        for offset in range(-width, width + 1):
            total += weights[width + offset] * np.roll(field, -offset, axis=axis)
        return total

    gradients = [derivative(box, axis) for axis in range(3)]
    sigma = sum(g * g for g in gradients).reshape(-1)
    energy, fn, fsigma = pbe_energy_partials(density, sigma)
    potential = fn.copy()
    for axis in range(3):
        flux = (2.0 * fsigma).reshape(grid.shape) * gradients[axis]
        # Transpose of the antisymmetric circulant stencil is its negative.
        potential -= derivative(flux, axis).reshape(-1)
    return 2.0 * potential, 2.0 * energy


class PeriodicPBEOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = _grid()
        self.density = _gaussian_density(self.grid)

    def test_matches_independent_roll_implementation(self) -> None:
        result = pbe(self.density, self.grid)
        potential, energy_density = _reference_pbe(self.density, self.grid)
        np.testing.assert_allclose(result.potential, potential, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            result.energy_density, energy_density, rtol=1e-10, atol=1e-12
        )

    def test_potential_is_variational_derivative_of_energy(self) -> None:
        direction = np.random.default_rng(3).normal(size=self.grid.size)
        direction /= np.linalg.norm(direction)
        step = 1.0e-5
        finite_difference = (
            pbe(self.density + step * direction, self.grid).total_energy
            - pbe(self.density - step * direction, self.grid).total_energy
        ) / (2.0 * step)
        analytic = self.grid.volume_element * np.dot(
            pbe(self.density, self.grid).potential, direction
        )
        self.assertAlmostEqual(finite_difference, analytic, delta=1.0e-8)

    def test_translation_invariance(self) -> None:
        """A rigid lattice-periodic shift cannot change the energy -- the
        zero-padded isolated convention would break this at the box faces."""
        box = self.density.reshape(self.grid.shape)
        shifted = np.roll(box, shift=(2, -3, 1), axis=(0, 1, 2)).reshape(-1)
        base = pbe(self.density, self.grid)
        moved = pbe(shifted, self.grid)
        self.assertAlmostEqual(base.total_energy, moved.total_energy, places=10)
        np.testing.assert_allclose(
            np.sort(base.potential), np.sort(moved.potential), rtol=1e-10, atol=1e-12
        )

    def test_uniform_density_has_no_gradient_correction(self) -> None:
        uniform = np.full(self.grid.size, 0.05)
        result = pbe(uniform, self.grid)
        energy, fn, _fsigma = pbe_energy_partials(uniform, np.zeros_like(uniform))
        np.testing.assert_allclose(result.potential, 2.0 * fn, rtol=1e-12)
        np.testing.assert_allclose(result.energy_density, 2.0 * energy, rtol=1e-12)

    def test_core_density_is_added_to_the_gradient_density(self) -> None:
        core = 0.5 * self.density
        with_core = pbe(self.density, self.grid, core)
        total = pbe(self.density + core, self.grid)
        np.testing.assert_allclose(with_core.potential, total.potential)
        self.assertAlmostEqual(with_core.total_energy, total.total_energy, places=12)


_PBE_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 5.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

SO_from_scratch: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: {correlation}
"""


class PeriodicSelfConsistentSOCWithPBETests(unittest.TestCase):
    """Periodic self-consistent SOC accepts Correlation_Type=PBE end to end.

    The POTRE header's correlation label is checked against the requested
    functional, so a copy of the example file is relabelled ``pb``.  (The
    example pseudopotential itself was generated with LDA; this tests the
    solver path, not pseudopotential/functional consistency.)
    """

    def _run(self, correlation: str, label: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            lines = _AU_POTRE.read_text(encoding="utf-8").splitlines()
            tokens = lines[0].split()
            tokens[1] = label
            lines[0] = " " + " ".join(tokens) + "  "
            (directory / "Au_POTRE.DAT").write_text("\n".join(lines) + "\n")
            (directory / "parsec.in").write_text(
                _PBE_INPUT.format(correlation=correlation), encoding="utf-8"
            )
            exit_code = cli_main([str(directory / "parsec.in"), "--quiet"])
            self.assertEqual(exit_code, 0)
            archive = np.load(directory / "parsec_python_results.npz")
            return {key: archive[key] for key in archive.files}

    def test_pbe_converges_with_kramers_pairs(self) -> None:
        archive = self._run("PBE", "pb")
        self.assertTrue(bool(archive["converged"]))
        eigenvalues = archive["eigenvalues_ry"]
        moments = archive["magnetic_moment_per_state"]
        for i in range(0, len(eigenvalues) - 3, 2):
            self.assertAlmostEqual(eigenvalues[i], eigenvalues[i + 1], delta=1.0e-4)
            self.assertAlmostEqual(moments[i] + moments[i + 1], 0.0, delta=0.05)

    def test_pbe_differs_from_lda(self) -> None:
        pbe_result = self._run("PBE", "pb")
        lda_result = self._run("CA", "ca")
        self.assertTrue(np.isfinite(pbe_result["energy_total_ry"]))
        self.assertGreater(
            abs(float(pbe_result["energy_total_ry"]) - float(lda_result["energy_total_ry"])),
            1.0e-3,
        )


if __name__ == "__main__":
    unittest.main()
