"""Tests for collinear spin-polarized PBE: the kernel, its exact discrete
variational derivative on cluster and periodic grids, and the SCF loops
(plain spin-polarized and self-consistent spin-orbit) that use it."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.SCF.self_consistent_soc_spin_polarized import (
    run_self_consistent_soc_spin_polarized,
)
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized
from parsec_python.V_xc import (
    pbe,
    pbe_energy_partials,
    pbe_spin_energy_partials,
    pbe_spin_polarized,
)
from parsec_python.V_xc.pbe import _pbe_spin_energy_density

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
_C_PBE_POTRE = _EXAMPLES / "0_CH4_CF4" / "python_pbe" / "pseudopotentials" / "C_POTRE.DAT"
_AU_POTRE = _EXAMPLES / "0d_AuH" / "Au_POTRE.DAT"


def _cluster_grid():
    return build_cluster_grid(
        GridSettings(
            spacing=0.7, radius=3.2, expansion_order=8, shift=(0.5, 0.5, 0.5)
        )
    )


def _periodic_grid():
    cell = PeriodicCell(lattice_vectors=7.0 * np.eye(3))
    return build_periodic_grid(
        cell, PeriodicGridSettings(spacing=0.7, expansion_order=6)
    )


def _spin_densities(grid):
    """Smooth, genuinely polarized, non-proportional spin densities."""
    coordinates = grid.coordinates
    if hasattr(grid, "cell"):
        side = np.diag(grid.cell.lattice_vectors)
        delta = coordinates - 0.5 * side
        delta -= side * np.round(delta / side)
    else:
        delta = coordinates
    r2 = np.einsum("ij,ij->i", delta, delta)
    up = 0.15 * np.exp(-0.5 * r2) + 0.01
    down = 0.08 * np.exp(-0.9 * r2) + 0.008
    return up, down


class SpinPolarizedPBEKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        n = 10 ** rng.uniform(-3, 0.3, 300)
        zeta = rng.uniform(-0.95, 0.95, 300)
        self.n_up = n * (1 + zeta) / 2
        self.n_down = n * (1 - zeta) / 2
        grad_up = rng.normal(size=(3, 300)) * self.n_up * rng.uniform(0.1, 2, 300)
        grad_down = rng.normal(size=(3, 300)) * self.n_down * rng.uniform(0.1, 2, 300)
        self.s_uu = (grad_up**2).sum(0)
        self.s_dd = (grad_down**2).sum(0)
        self.s_ud = (grad_up * grad_down).sum(0)

    def test_partials_match_finite_differences(self) -> None:
        args = [self.n_up, self.n_down, self.s_uu, self.s_ud, self.s_dd]
        partials = pbe_spin_energy_partials(*args)[1:]
        for index, analytic in enumerate(partials):
            step = 1.0e-4 * np.maximum(np.abs(args[index]), 1.0e-12)
            plus = [a.copy() for a in args]
            minus = [a.copy() for a in args]
            plus[index] = plus[index] + step
            minus[index] = minus[index] - step
            finite = (
                _pbe_spin_energy_density(*plus) - _pbe_spin_energy_density(*minus)
            ) / (2.0 * step)
            error = np.abs(finite - analytic) / np.maximum(np.abs(analytic), 1e-12)
            self.assertLess(error.max(), 1.0e-4, msg=f"partial {index}")

    def test_unpolarized_limit_matches_unpolarized_pbe(self) -> None:
        total = self.n_up + self.n_down
        gradient_sq = self.s_uu + 2.0 * self.s_ud + self.s_dd
        half = total / 2.0
        quarter = gradient_sq / 4.0
        f, f_up, f_down, f_uu, f_ud, f_dd = pbe_spin_energy_partials(
            half, half, quarter, quarter, quarter
        )
        energy, derivative_n, derivative_sigma = pbe_energy_partials(
            total, gradient_sq
        )
        np.testing.assert_allclose(f, energy, rtol=1e-12)
        np.testing.assert_allclose(0.5 * (f_up + f_down), derivative_n, rtol=1e-10)
        # With n_up = n_down and s_uu = s_ud = s_dd = sigma/4, the chain rule
        # gives df/dsigma = (f_uu + f_ud + f_dd)/4.
        np.testing.assert_allclose(
            (f_uu + f_ud + f_dd) / 4.0, derivative_sigma, rtol=1e-8
        )

    def test_spin_flip_symmetry(self) -> None:
        up_first = pbe_spin_energy_partials(
            self.n_up, self.n_down, self.s_uu, self.s_ud, self.s_dd
        )
        down_first = pbe_spin_energy_partials(
            self.n_down, self.n_up, self.s_dd, self.s_ud, self.s_uu
        )
        np.testing.assert_allclose(up_first[0], down_first[0], rtol=1e-12)
        np.testing.assert_allclose(up_first[1], down_first[2], rtol=1e-10)
        np.testing.assert_allclose(up_first[3], down_first[5], rtol=1e-10)
        np.testing.assert_allclose(up_first[4], down_first[4], rtol=1e-10)


class SpinPolarizedPBEGridTests(unittest.TestCase):
    def _check(self, grid) -> None:
        up, down = _spin_densities(grid)

        # zeta = 0 reproduces the unpolarized grid routine exactly.
        half = 0.5 * (up + down)
        unpolarized = pbe(up + down, grid)
        symmetric = pbe_spin_polarized(half, half, grid)
        np.testing.assert_allclose(
            symmetric.potential_up, unpolarized.potential, rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            symmetric.potential_down, unpolarized.potential, rtol=1e-9, atol=1e-12
        )
        self.assertAlmostEqual(
            symmetric.total_energy, unpolarized.total_energy, places=9
        )

        # Each spin potential is the exact derivative of the summed energy.
        result = pbe_spin_polarized(up, down, grid)
        rng = np.random.default_rng(5)
        for channel, potential in (("up", result.potential_up), ("down", result.potential_down)):
            direction = rng.normal(size=grid.size)
            direction /= np.linalg.norm(direction)
            step = 1.0e-5
            if channel == "up":
                plus = pbe_spin_polarized(up + step * direction, down, grid)
                minus = pbe_spin_polarized(up - step * direction, down, grid)
            else:
                plus = pbe_spin_polarized(up, down + step * direction, grid)
                minus = pbe_spin_polarized(up, down - step * direction, grid)
            finite = (plus.total_energy - minus.total_energy) / (2.0 * step)
            analytic = grid.volume_element * np.dot(potential, direction)
            self.assertAlmostEqual(finite, analytic, delta=1.0e-7, msg=channel)

        # Spin-flip symmetry of the full grid functional.
        flipped = pbe_spin_polarized(down, up, grid)
        np.testing.assert_allclose(result.potential_up, flipped.potential_down, rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(result.total_energy, flipped.total_energy, places=10)

        # A frozen core is split evenly between the channels.
        core = 0.3 * up
        with_core = pbe_spin_polarized(up, down, grid, core)
        manual = pbe_spin_polarized(up + 0.5 * core, down + 0.5 * core, grid)
        np.testing.assert_allclose(with_core.potential_up, manual.potential_up)
        self.assertAlmostEqual(with_core.total_energy, manual.total_energy, places=12)

    def test_cluster_grid(self) -> None:
        self._check(_cluster_grid())

    def test_periodic_grid(self) -> None:
        self._check(_periodic_grid())

    def test_periodic_translation_invariance(self) -> None:
        grid = _periodic_grid()
        up, down = _spin_densities(grid)
        shift = dict(shift=(2, -3, 1), axis=(0, 1, 2))
        up_s = np.roll(up.reshape(grid.shape), **shift).reshape(-1)
        down_s = np.roll(down.reshape(grid.shape), **shift).reshape(-1)
        self.assertAlmostEqual(
            pbe_spin_polarized(up, down, grid).total_energy,
            pbe_spin_polarized(up_s, down_s, grid).total_energy,
            places=10,
        )


class SpinPolarizedPBESCFTests(unittest.TestCase):
    def test_carbon_atom_triplet_moment(self) -> None:
        """An isolated C atom is a Hund's-rule triplet: N_up - N_down = 2."""
        problem = SinglePointInput(
            atoms=[Atom("C", [0.0, 0.0, 0.0])],
            pseudopotentials={
                "C": SpeciesPotential(_C_PBE_POTRE, 1, initial_spin_polarization=0.3)
            },
            grid=GridSettings(spacing=0.5, radius=5.0, expansion_order=6),
            scf=SCFSettings(
                max_iterations=80,
                number_of_states=6,
                spin_polarized=True,
                xc_functional="pbe",
                convergence_criterion=1.0e-3,
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_single_point(problem)
        result = run_scf_spin_polarized(system)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.magnetic_moment, 2.0, places=2)
        self.assertTrue(np.isfinite(result.energies.total))

    def test_self_consistent_soc_with_pbe_has_net_moment(self) -> None:
        """Isolated Au (6s^1) with PBE + spin-orbit + spin polarization.

        The example POTRE's correlation label is changed to ``pb`` so the
        loader accepts a PBE run; this exercises the solver path (the
        pseudopotential itself was generated with LDA).
        """
        with tempfile.TemporaryDirectory() as tmp:
            potre = Path(tmp) / "Au_POTRE.DAT"
            lines = _AU_POTRE.read_text(encoding="utf-8").splitlines()
            tokens = lines[0].split()
            tokens[1] = "pb"
            lines[0] = " " + " ".join(tokens) + "  "
            potre.write_text("\n".join(lines) + "\n", encoding="utf-8")
            problem = SinglePointInput(
                atoms=[Atom("Au", [0.0, 0.0, 0.0])],
                pseudopotentials={"Au": SpeciesPotential(potre, 0, spin_orbit=True)},
                grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
                scf=SCFSettings(
                    max_iterations=60,
                    number_of_states=20,
                    convergence_criterion=5.0e-3,
                    spin_polarized=True,
                    self_consistent_spin_orbit=True,
                    xc_functional="pbe",
                ),
                hartree=HartreeSettings(multipole_order=4),
                eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
            )
            system = prepare_single_point(problem)
            projectors = tuple(
                build_spin_orbit_projectors(
                    system.grid,
                    system.atoms,
                    system.pseudopotentials,
                    system.input.pseudopotentials,
                )
            )
            result = run_self_consistent_soc_spin_polarized(system, projectors)
        self.assertTrue(result.converged)
        moment = float(np.dot(result.occupations, result.magnetic_moment_per_state))
        self.assertGreater(abs(moment), 0.3)
        self.assertTrue(np.isfinite(result.energies.total))


if __name__ == "__main__":
    unittest.main()
