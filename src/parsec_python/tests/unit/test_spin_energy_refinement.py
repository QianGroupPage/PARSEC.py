"""B3LYP energy refinement of spin-polarized orbitals and densities.

* Pointwise energies per volume of the four density functionals for two
  different spin densities from libxc 7.0.0 (through PySCF 2.11.0,
  ``dft.libxc.eval_xc`` with ``spin=1``): ``LDA_X``, ``GGA_X_B88``,
  ``LDA_C_VWN_RPA``, ``LDA_C_VWN`` and ``GGA_C_LYP``, in Hartree at the
  densities and gradients of ``POINTS``.
* The unpolarized functionals and the unpolarized exact exchange, which two
  identical channels must give.
* An exact identity: one electron has no self-interaction, so its exchange
  energy is minus its Hartree energy.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Energy.refinement import refine_energy, refinement_report_lines
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import solve_hartree
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_xc import pbe_spin
from parsec_python.V_xc.b3lyp import (
    b3lyp_semilocal_energies,
    b3lyp_spin_semilocal_energies,
    b88_exchange_energy_density,
    b88_exchange_energy_density_spin,
    lyp_correlation_energy_density,
    lyp_correlation_energy_density_spin,
    slater_exchange_energy_density,
    slater_exchange_energy_density_spin,
    vwn_correlation_energy_density,
    vwn_correlation_energy_density_spin,
)
from parsec_python.V_xc.exact_exchange import (
    exact_exchange_energy,
    pair_hartree_settings,
    reference_pair_poisson,
    spin_exact_exchange_energy,
)
from parsec_python.models import (
    EnergyBreakdown,
    GridSettings,
    HartreeSettings,
    SpinChannelResult,
)


# (n_up, n_down, grad n_up, grad n_down) in atomic units.  The fifth point is
# unpolarized and the last one fully polarized.
POINTS = (
    (1.0e-4, 5.0e-5, (1.0e-4, 0.0, 0.0), (-2.0e-5, 3.0e-5, 0.0)),
    (0.01, 0.02, (0.02, 0.01, 0.0), (0.01, -0.03, 0.02)),
    (0.1, 0.03, (-0.15, 0.05, 0.1), (0.02, 0.04, -0.01)),
    (1.0, 0.4, (1.5, -0.5, 0.25), (-0.3, 0.6, 0.2)),
    (0.3, 0.3, (0.2, 0.1, -0.3), (0.2, 0.1, -0.3)),
    (0.25, 0.0025, (0.3, 0.0, 0.1), (0.001, 0.002, 0.0)),
    (0.5, 0.0, (0.4, -0.2, 0.1), (0.0, 0.0, 0.0)),
)
# libxc 7.0.0, energy per volume in Hartree.
LIBXC = {
    "slater": (-6.033160928730104e-06, -7.056427829239852e-03, -5.186525811910461e-02, -1.204772626189550e+00, -3.737547532869366e-01, -1.468643497683249e-01, -3.692793831910105e-01),
    "b88": (-1.006144800398574e-05, -8.343398887149012e-03, -5.521171543011448e-02, -1.221391353387216e+00, -3.792539791939869e-01, -1.493925301580918e-01, -3.714398535996367e-01),
    "vwn_rpa": (-4.204824377680728e-06, -1.817567859123737e-03, -8.814168065503879e-03, -1.250197671328464e-01, -5.236442559697765e-02, -1.354207913509899e-02, -2.767961140470537e-02),
    "lyp": (-8.200647773233662e-07, -6.453166575227206e-04, -3.851992449816107e-03, -6.155626249308178e-02, -2.628680475306733e-02, -5.756280892234340e-04, 0.0),
}


def _points():
    up = np.array([point[0] for point in POINTS])
    down = np.array([point[1] for point in POINTS])
    gradient_up = np.array([point[2] for point in POINTS]).T
    gradient_down = np.array([point[3] for point in POINTS]).T
    return (
        up,
        down,
        np.sum(gradient_up**2, axis=0),
        np.sum(gradient_down**2, axis=0),
        np.sum((gradient_up + gradient_down) ** 2, axis=0),
    )


class SpinDensityFunctionalTests(unittest.TestCase):
    def test_pointwise_values_match_libxc(self) -> None:
        up, down, sigma_up, sigma_down, sigma_total = _points()
        values = {
            "slater": slater_exchange_energy_density_spin(up, down),
            "b88": b88_exchange_energy_density_spin(up, down, sigma_up, sigma_down),
            "vwn_rpa": vwn_correlation_energy_density_spin(up, down, "rpa"),
            "lyp": lyp_correlation_energy_density_spin(
                up, down, sigma_up, sigma_down, sigma_total
            ),
        }
        for name, ours in values.items():
            with self.subTest(functional=name):
                np.testing.assert_allclose(
                    ours, np.array(LIBXC[name]), rtol=1.0e-12, atol=1.0e-18
                )
        # One electron of one spin has no opposite-spin partner to correlate
        # with in LYP.
        self.assertEqual(values["lyp"][-1], 0.0)

    def test_equal_spin_densities_give_the_unpolarized_functionals(self) -> None:
        density = np.array([2.0e-4, 0.03, 0.4, 2.5, 60.0])
        sigma = np.array([3.0e-7, 2.0e-3, 0.9, 11.0, 4.0e4])
        half, quarter = 0.5 * density, 0.25 * sigma
        pairs = {
            "slater": (
                slater_exchange_energy_density_spin(half, half),
                slater_exchange_energy_density(density),
            ),
            "b88": (
                b88_exchange_energy_density_spin(half, half, quarter, quarter),
                b88_exchange_energy_density(density, sigma),
            ),
            "vwn_rpa": (
                vwn_correlation_energy_density_spin(half, half, "rpa"),
                vwn_correlation_energy_density(density, "rpa"),
            ),
            "vwn5": (
                vwn_correlation_energy_density_spin(half, half, "vwn5"),
                vwn_correlation_energy_density(density, "vwn5"),
            ),
            "lyp": (
                lyp_correlation_energy_density_spin(half, half, quarter, quarter, sigma),
                lyp_correlation_energy_density(density, sigma),
            ),
        }
        for name, (spin, plain) in pairs.items():
            with self.subTest(functional=name):
                np.testing.assert_allclose(spin, plain, rtol=1.0e-11)

    def test_polarization_lowers_exchange_and_weakens_correlation(self) -> None:
        density = np.array([0.02, 0.3, 1.5])
        zero = np.zeros(3)
        self.assertTrue(
            np.all(
                slater_exchange_energy_density_spin(density, zero)
                < slater_exchange_energy_density(density)
            )
        )
        for name in ("rpa", "vwn5"):
            polarized = vwn_correlation_energy_density_spin(density, zero, name)
            unpolarized = vwn_correlation_energy_density(density, name)
            self.assertTrue(np.all(polarized > unpolarized), name)
            self.assertTrue(np.all(polarized < 0.0), name)
        with self.assertRaisesRegex(ValueError, "'rpa' or 'vwn5'"):
            vwn_correlation_energy_density_spin(density, zero, "pw92")
        with self.assertRaisesRegex(ValueError, "same shape"):
            slater_exchange_energy_density_spin(density, np.zeros(2))

    def test_grid_energies_of_equal_channels_with_a_core_density(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.7, radius=3.2, expansion_order=8)
        )
        radius_squared = np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)
        density = 0.2 * np.exp(-0.5 * radius_squared) + 0.01
        core = 0.05 * np.exp(-2.0 * radius_squared)
        spin = b3lyp_spin_semilocal_energies(0.5 * density, 0.5 * density, grid, core)
        plain = b3lyp_semilocal_energies(density, grid, core)
        for name in ("slater_exchange", "b88_exchange", "vwn_correlation", "lyp_correlation"):
            self.assertAlmostEqual(
                getattr(spin, name), getattr(plain, name), places=11, msg=name
            )
        polarized = b3lyp_spin_semilocal_energies(0.8 * density, 0.2 * density, grid, core)
        self.assertLess(polarized.slater_exchange, plain.slater_exchange)
        self.assertGreater(polarized.lyp_correlation, plain.lyp_correlation)


class _GaussianOrbitals:
    """Two Gaussian orbitals on a grid and a pair-density Poisson solver."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(spacing=0.5, radius=7.0, expansion_order=8)
        )
        cls.laplacian = build_negative_laplacian(cls.grid)
        cls.settings = HartreeSettings(
            multipole_order=4, relative_tolerance=1.0e-10, max_iterations=4000
        )
        cls.solver = staticmethod(
            reference_pair_poisson(cls.grid, cls.laplacian, cls.settings)
        )
        points = cls.grid.coordinates
        gaussian = np.exp(-0.45 * np.einsum("ij,ij->i", points, points))
        cls.s_orbital = gaussian / np.linalg.norm(gaussian)
        p_orbital = points[:, 0] * gaussian
        cls.p_orbital = p_orbital / np.linalg.norm(p_orbital)
        cls.orbitals = np.column_stack(
            (cls.s_orbital, cls.p_orbital, np.zeros(cls.grid.size))
        )


class SpinExactExchangeTests(_GaussianOrbitals, unittest.TestCase):
    def exchange(self, occupations_up, occupations_down, **options):
        return spin_exact_exchange_energy(
            [self.orbitals, self.orbitals],
            [np.asarray(occupations_up, float), np.asarray(occupations_down, float)],
            self.grid.volume_element,
            self.solver,
            **options,
        )

    def test_identical_channels_give_the_unpolarized_exchange(self) -> None:
        occupations = np.array([1.0, 0.4, 0.0])
        spin = self.exchange(occupations, occupations)
        plain = exact_exchange_energy(
            self.orbitals, occupations, self.grid.volume_element, self.solver
        )
        self.assertAlmostEqual(spin.energy, plain.energy, places=12)
        self.assertEqual(spin.pair_solves, 2 * plain.pair_solves)
        # Columns of the up channel, then those of the down channel.
        self.assertEqual(spin.orbitals, (0, 1, 3, 4))
        self.assertTrue(spin.fractional)
        self.assertAlmostEqual(spin.largest_fractional_deviation, 0.4)

    def test_one_electron_exchange_cancels_its_hartree_energy(self) -> None:
        result = self.exchange([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        density = self.s_orbital**2 / self.grid.volume_element
        hartree = solve_hartree(
            density, self.grid, self.laplacian, pair_hartree_settings(self.settings)
        )
        hartree_energy = 0.5 * self.grid.volume_element * float(
            density @ hartree.potential
        )
        self.assertAlmostEqual(result.energy / -hartree_energy, 1.0, delta=1.0e-9)
        self.assertEqual(result.orbitals, (0,))
        self.assertEqual(result.pair_solves, 1)
        self.assertFalse(result.fractional)

    def test_only_orbitals_of_the_same_spin_exchange(self) -> None:
        volume = self.grid.volume_element
        triplet = self.exchange([1.0, 1.0, 0.0], [0.0, 0.0, 0.0])
        singlet = self.exchange([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        only_s = exact_exchange_energy(self.orbitals, [1.0, 0.0, 0.0], volume, self.solver)
        only_p = exact_exchange_energy(self.orbitals, [0.0, 1.0, 0.0], volume, self.solver)
        both = exact_exchange_energy(self.orbitals, [1.0, 1.0, 0.0], volume, self.solver)
        # Opposite spins: the self-exchange of each electron and nothing else.
        self.assertAlmostEqual(
            singlet.energy, 0.5 * (only_s.energy + only_p.energy), places=12
        )
        self.assertEqual(singlet.orbitals, (0, 4))
        # Parallel spins add the cross term 2 (sp|sp), which lowers the energy.
        self.assertAlmostEqual(triplet.energy, 0.5 * both.energy, places=12)
        self.assertLess(triplet.energy, singlet.energy - 1.0e-3)

    def test_progress_counts_the_pairs_of_both_channels(self) -> None:
        calls: list[tuple[int, int]] = []
        result = self.exchange(
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            progress=lambda done, total, seconds: calls.append((done, total)),
        )
        self.assertEqual(result.pair_solves, 4)
        self.assertEqual(calls, [(1, 4), (2, 4), (3, 4), (4, 4)])
        with self.assertRaisesRegex(ValueError, "two spin channels"):
            spin_exact_exchange_energy(
                [self.orbitals], [np.ones(3)], self.grid.volume_element, self.solver
            )


class SpinRefinementAssemblyTests(_GaussianOrbitals, unittest.TestCase):
    """The refined energy from a stand-in for a spin-polarized result."""

    def result(self, occupations_up, occupations_down):
        volume = self.grid.volume_element
        channels = []
        for occupations in (occupations_up, occupations_down):
            occupations = np.asarray(occupations, dtype=float)
            channels.append(
                SpinChannelResult(
                    eigenvalues=np.zeros(3),
                    occupations=occupations,
                    wavefunctions=self.orbitals,
                    representations=np.ones(3, dtype=np.int32),
                    density=(self.orbitals**2 @ occupations) / volume,
                    xc_potential=np.zeros(self.grid.size),
                    input_effective_potential=np.zeros(self.grid.size),
                    output_effective_potential=np.zeros(self.grid.size),
                    next_effective_potential=np.zeros(self.grid.size),
                    electron_count=float(np.sum(occupations)),
                )
            )
        scf_xc = pbe_spin(channels[0].density, channels[1].density, self.grid).total_energy
        return SimpleNamespace(
            spin=tuple(channels),
            density=channels[0].density + channels[1].density,
            core_density=np.zeros(self.grid.size),
            wavefunctions=np.hstack((self.orbitals, self.orbitals)),
            occupations=np.concatenate([channel.occupations for channel in channels]),
            converged=True,
            atomic_reference_correction=0.0,
            energies=EnergyBreakdown(
                eigenvalue=-1.0,
                hartree=2.0,
                integral_vxc_rho=-1.5,
                exchange_correlation=scf_xc,
                electron_ion=-6.0,
                ion_ion=0.5,
                electronic=-3.5,
                total=-3.0,
            ),
        )

    @property
    def system(self):
        return SimpleNamespace(
            grid=self.grid,
            negative_laplacian=self.laplacian,
            input=SimpleNamespace(
                periodic_cell=None,
                hartree=self.settings,
                scf=SimpleNamespace(xc_functional="pbe"),
            ),
        )

    def test_refined_total_of_a_triplet(self) -> None:
        result = self.result([1.0, 1.0, 0.0], [0.0, 0.0, 0.0])
        refinement = refine_energy(self.system, result)
        self.assertTrue(refinement.spin_polarized)
        semilocal = b3lyp_spin_semilocal_energies(
            result.spin[0].density, result.spin[1].density, self.grid
        )
        exchange = spin_exact_exchange_energy(
            [self.orbitals, self.orbitals],
            [result.spin[0].occupations, result.spin[1].occupations],
            self.grid.volume_element,
            self.solver,
        )
        self.assertAlmostEqual(refinement.exact_exchange, exchange.energy, places=12)
        self.assertEqual(refinement.b88_exchange, semilocal.b88_exchange)
        # No opposite-spin electrons: no LYP correlation.
        self.assertAlmostEqual(refinement.lyp_correlation, 0.0, places=14)
        expected = (
            0.20 * exchange.energy
            + 0.08 * semilocal.slater_exchange
            + 0.72 * semilocal.b88_exchange
            + 0.19 * semilocal.vwn_correlation
            + 0.81 * semilocal.lyp_correlation
        )
        self.assertAlmostEqual(refinement.exchange_correlation, expected, places=13)
        scf_xc = result.energies.exchange_correlation
        self.assertAlmostEqual(refinement.total, -3.0 - scf_xc + expected, places=13)
        report = "\n".join(refinement_report_lines(refinement, 13.6058))
        self.assertIn("2 occupied orbitals of the two spin channels", report)
        self.assertIn("Spin-polarized: the orbitals of each spin channel", report)
        self.assertNotIn("fractional occupations", report)

    def test_closed_shell_channels_give_the_unpolarized_refinement(self) -> None:
        spin_result = self.result([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        plain_result = SimpleNamespace(
            density=spin_result.density,
            core_density=spin_result.core_density,
            wavefunctions=self.orbitals,
            occupations=np.array([1.0, 0.0, 0.0]),
            converged=True,
            atomic_reference_correction=0.0,
            energies=spin_result.energies,
        )
        spin = refine_energy(self.system, spin_result)
        plain = refine_energy(self.system, plain_result)
        self.assertFalse(plain.spin_polarized)
        for name in (
            "exact_exchange",
            "slater_exchange",
            "b88_exchange",
            "vwn_correlation",
            "lyp_correlation",
            "exchange_correlation",
            "total",
        ):
            self.assertAlmostEqual(
                getattr(spin, name), getattr(plain, name), places=11, msg=name
            )

    def test_fractional_occupations_and_foreign_orbitals(self) -> None:
        fractional = refine_energy(
            self.system, self.result([1.0, 0.5, 0.0], [1.0, 0.0, 0.0])
        )
        report = "\n".join(refinement_report_lines(fractional, 13.6058))
        self.assertIn("ensemble density matrix of each spin channel", report)
        self.assertNotIn("spin-unpolarized ensemble", report)
        # Orbitals that do not give the density of their channel.
        result = self.result([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        foreign = SimpleNamespace(**vars(result))
        foreign.spin = (
            replace(result.spin[0], density=result.spin[1].density),
            result.spin[1],
        )
        with self.assertRaisesRegex(ValueError, "do not give its density"):
            refine_energy(self.system, foreign)


if __name__ == "__main__":
    unittest.main()
