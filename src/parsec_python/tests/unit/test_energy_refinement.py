"""B3LYP energy refinement: functionals, exact exchange, keyword and report.

Three kinds of reference are used.

* Pointwise values of the four density functionals from libxc 7.0.0
  (evaluated through PySCF 2.14.0, ``dft.libxc.eval_xc`` with ``spin=0``):
  ``LDA_X``, ``GGA_X_B88``, ``LDA_C_VWN``, ``LDA_C_VWN_RPA``, ``GGA_C_LYP``
  and the semilocal part of ``B3LYP``.  They are energies per volume in
  Hartree at a total density ``n`` and ``sigma = |grad n|**2``.
* Formulas written independently of the implementation: the spin-resolved
  Lee-Yang-Parr energy of Miehlich, Savin, Stoll and Preuss evaluated at
  equal spin densities, and Coulomb integrals of Gaussians.
* Exact identities: the exchange energy of two electrons in one orbital is
  minus one half of the Hartree energy of their density.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python import cli as cli_module
from parsec_python.Energy.refinement import (
    apply_refinement_option,
    normalize_refinement,
    orbital_density_deviation,
    refine_energy,
    refinement_report_lines,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.Hartree import solve_hartree
from parsec_python.Input import (
    UnsupportedParsecOptionError,
    parse_parsec_input,
    summarize_translation,
)
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_xc import pbe
from parsec_python.V_xc.b3lyp import (
    B3LYP_B88_EXCHANGE,
    B3LYP_EXACT_EXCHANGE,
    B3LYP_LYP_CORRELATION,
    B3LYP_SLATER_EXCHANGE,
    B3LYP_VWN_CORRELATION,
    b3lyp_exchange_correlation_energy,
    b3lyp_semilocal_energies,
    b88_exchange_energy_density,
    density_gradient_squared,
    lyp_correlation_energy_density,
    slater_exchange_energy_density,
    vwn_correlation_energy_density,
)
from parsec_python.V_xc.exact_exchange import (
    exact_exchange_energy,
    occupied_columns,
    pair_hartree_settings,
    reference_pair_poisson,
)
from parsec_python.models import (
    EnergyBreakdown,
    GridSettings,
    HartreeSettings,
)


DATA = Path(__file__).resolve().parents[1] / "data"
EXAMPLES = Path(__file__).resolve().parents[4] / "examples"

# (n, sigma): Slater, B88, VWN5, VWN-RPA, LYP from libxc 7.0.0.
LIBXC_POINTS = {
    (3.4, 6.8): (
        -3.775927208361e00,
        -3.782897139115e00,
        -2.789781773675e-01,
        -3.499718043733e-01,
        -1.788747044394e-01,
    ),
    (0.176, 0.348): (
        -7.284536904138e-02,
        -8.516115450445e-02,
        -1.014837207802e-02,
        -1.350622424445e-02,
        -4.650241008028e-03,
    ),
    (3600.0, 2.2e7): (
        -4.074944753200e04,
        -4.075155690479e04,
        -5.327414770227e02,
        -6.164141801046e02,
        -2.376310505845e02,
    ),
    (0.106, 0.0384): (
        -3.705039801427e-02,
        -4.020290506321e-02,
        -5.705992725187e-03,
        -7.688851594052e-03,
        -3.188715438832e-03,
    ),
    # A density tail: the gradient term makes the LYP energy positive.
    (0.002, 4.0e-5): (
        -1.861051472698e-04,
        -3.812298985469e-04,
        -5.675309313628e-05,
        -8.678130208608e-05,
        6.224139756172e-05,
    ),
    (2.0e-6, 4.0e-12): (
        -1.861051472698e-08,
        -7.713966562082e-08,
        -1.155035519714e-08,
        -2.454102664142e-08,
        -3.427148927281e-09,
    ),
}
# Semilocal part of libxc's B3LYP (its exact-exchange fraction is 0.2).
LIBXC_B3LYP_SEMILOCAL = {
    (3.4, 6.8): -3.2371432703e00,
    (0.176, 0.348): -7.3476538589e-02,
    (0.106, 0.0384): -3.5953864795e-02,
}


def _lyp_spin_resolved(rho_a, rho_b, gamma_aa, gamma_ab, gamma_bb):
    """Miehlich et al., Chem. Phys. Lett. 157, 200 (1989), Eq. 2, per volume."""

    a, b, c, d = 0.04918, 0.132, 0.2533, 0.349
    cf = 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0)
    rho = rho_a + rho_b
    gamma = gamma_aa + 2.0 * gamma_ab + gamma_bb
    cube = rho ** (-1.0 / 3.0)
    omega = np.exp(-c * cube) / (1.0 + d * cube) * rho ** (-11.0 / 3.0)
    delta = c * cube + d * cube / (1.0 + d * cube)
    bracket = rho_a * rho_b * (
        2.0 ** (11.0 / 3.0) * cf * (rho_a ** (8.0 / 3.0) + rho_b ** (8.0 / 3.0))
        + (47.0 / 18.0 - 7.0 * delta / 18.0) * gamma
        - (2.5 - delta / 18.0) * (gamma_aa + gamma_bb)
        - (delta - 11.0) / 9.0 * (rho_a / rho * gamma_aa + rho_b / rho * gamma_bb)
    )
    bracket += (
        -2.0 / 3.0 * rho**2 * gamma
        + (2.0 / 3.0 * rho**2 - rho_a**2) * gamma_bb
        + (2.0 / 3.0 * rho**2 - rho_b**2) * gamma_aa
    )
    return -a * 4.0 / (1.0 + d * cube) * rho_a * rho_b / rho - a * b * omega * bracket


class DensityFunctionalTests(unittest.TestCase):
    def test_pointwise_values_match_libxc(self) -> None:
        for (density, sigma), reference in LIBXC_POINTS.items():
            n = np.array([density])
            s = np.array([sigma])
            values = (
                slater_exchange_energy_density(n)[0],
                b88_exchange_energy_density(n, s)[0],
                vwn_correlation_energy_density(n, "vwn5")[0],
                vwn_correlation_energy_density(n, "rpa")[0],
                lyp_correlation_energy_density(n, s)[0],
            )
            for name, value, expected in zip(
                ("slater", "b88", "vwn5", "vwn-rpa", "lyp"), values, reference
            ):
                with self.subTest(density=density, functional=name):
                    self.assertAlmostEqual(value / expected, 1.0, delta=2.0e-12)

    def test_b3lyp_is_the_vwn_rpa_combination_of_libxc(self) -> None:
        """libxc's B3LYP (Gaussian's) uses VWN-RPA; with VWN5 it is "B3LYP5"."""

        for (density, sigma), expected in LIBXC_B3LYP_SEMILOCAL.items():
            n = np.array([density])
            s = np.array([sigma])
            parts = {
                name: (
                    B3LYP_SLATER_EXCHANGE * slater_exchange_energy_density(n)[0]
                    + B3LYP_B88_EXCHANGE * b88_exchange_energy_density(n, s)[0]
                    + B3LYP_VWN_CORRELATION
                    * vwn_correlation_energy_density(n, name)[0]
                    + B3LYP_LYP_CORRELATION * lyp_correlation_energy_density(n, s)[0]
                )
                for name in ("rpa", "vwn5")
            }
            self.assertAlmostEqual(parts["rpa"] / expected, 1.0, delta=1.0e-10)
            self.assertGreater(abs(parts["vwn5"] / expected - 1.0), 1.0e-3)
        self.assertAlmostEqual(
            B3LYP_EXACT_EXCHANGE + B3LYP_SLATER_EXCHANGE + B3LYP_B88_EXCHANGE, 1.0
        )
        self.assertAlmostEqual(B3LYP_VWN_CORRELATION + B3LYP_LYP_CORRELATION, 1.0)

    def test_closed_shell_lyp_is_the_spin_resolved_formula_at_equal_spins(self) -> None:
        density = np.array([2.0e-4, 0.03, 0.4, 2.5, 60.0])
        sigma = np.array([3.0e-7, 2.0e-3, 0.9, 11.0, 4.0e4])
        general = _lyp_spin_resolved(
            0.5 * density, 0.5 * density, 0.25 * sigma, 0.25 * sigma, 0.25 * sigma
        )
        np.testing.assert_allclose(
            lyp_correlation_energy_density(density, sigma), general, rtol=1.0e-11
        )

    def test_limits_and_vacuum(self) -> None:
        density = np.array([1.0e-3, 0.2, 5.0])
        zero = np.zeros_like(density)
        # Without a gradient B88 is Slater exchange.
        np.testing.assert_allclose(
            b88_exchange_energy_density(density, zero),
            slater_exchange_energy_density(density),
            rtol=1.0e-14,
        )
        # The gradient correction lowers the exchange energy.
        self.assertTrue(
            np.all(
                b88_exchange_energy_density(density, density**2)
                < slater_exchange_energy_density(density)
            )
        )
        # RPA overbinds the uniform gas relative to the Monte Carlo fit.
        self.assertTrue(
            np.all(
                vwn_correlation_energy_density(density, "rpa")
                < vwn_correlation_energy_density(density, "vwn5")
            )
        )
        vacuum = np.zeros(4)
        for values in (
            slater_exchange_energy_density(vacuum),
            b88_exchange_energy_density(vacuum, vacuum),
            vwn_correlation_energy_density(vacuum),
            lyp_correlation_energy_density(vacuum, vacuum),
        ):
            np.testing.assert_array_equal(values, 0.0)
        with self.assertRaises(ValueError):
            vwn_correlation_energy_density(density, "pw92")
        with self.assertRaises(ValueError):
            slater_exchange_energy_density(np.array([-1.0]))
        with self.assertRaises(ValueError):
            b88_exchange_energy_density(density, zero[:2])


class GridFunctionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(spacing=0.4, radius=8.0, expansion_order=8)
        )
        cls.exponent = 0.4
        radius_squared = np.einsum("ij,ij->i", cls.grid.coordinates, cls.grid.coordinates)
        cls.radius_squared = radius_squared
        # Two electrons in the Gaussian orbital (2a/pi)**(3/4) exp(-a r**2).
        cls.density = (
            2.0
            * (2.0 * cls.exponent / np.pi) ** 1.5
            * np.exp(-2.0 * cls.exponent * radius_squared)
        )

    def test_gradient_is_the_one_of_the_self_consistent_pbe(self) -> None:
        # |grad n|**2 = (4 a r n)**2 for n ~ exp(-2 a r**2).  The order-8
        # stencil resolves the density of the orbital tests to a few 1e-3 of
        # the largest value and a smoother one ten times better.
        for exponent, tolerance in ((self.exponent, 5.0e-3), (0.15, 5.0e-4)):
            density = np.exp(-2.0 * exponent * self.radius_squared)
            sigma = density_gradient_squared(density, self.grid)
            analytic = (4.0 * exponent) ** 2 * self.radius_squared * density**2
            self.assertLess(
                np.max(np.abs(sigma - analytic)), tolerance * np.max(analytic)
            )
        with self.assertRaises(ValueError):
            density_gradient_squared(self.density[:-1], self.grid)

    def test_integrated_parts_of_a_gaussian_density(self) -> None:
        energies = b3lyp_semilocal_energies(self.density, self.grid)
        # Slater exchange of a Gaussian density, analytically (in Ry).
        prefactor = 2.0 * (2.0 * self.exponent / np.pi) ** 1.5
        slater = (
            -2.0
            * 0.75
            * (3.0 / np.pi) ** (1.0 / 3.0)
            * prefactor ** (4.0 / 3.0)
            * (np.pi / (8.0 * self.exponent / 3.0)) ** 1.5
        )
        self.assertAlmostEqual(energies.slater_exchange, slater, places=7)
        # The parts are the grid sums of the pointwise functions.
        sigma = density_gradient_squared(self.density, self.grid)
        scale = 2.0 * self.grid.volume_element
        self.assertAlmostEqual(
            energies.b88_exchange,
            scale * b88_exchange_energy_density(self.density, sigma).sum(),
            places=12,
        )
        self.assertAlmostEqual(
            energies.lyp_correlation,
            scale * lyp_correlation_energy_density(self.density, sigma).sum(),
            places=12,
        )
        self.assertAlmostEqual(
            energies.vwn_correlation,
            scale * vwn_correlation_energy_density(self.density, "rpa").sum(),
            places=12,
        )
        self.assertLess(energies.b88_exchange, energies.slater_exchange)
        self.assertEqual(energies.vwn_parametrization, "rpa")
        self.assertAlmostEqual(
            b3lyp_exchange_correlation_energy(-1.25, energies),
            0.20 * -1.25
            + 0.08 * energies.slater_exchange
            + 0.72 * energies.b88_exchange
            + 0.19 * energies.vwn_correlation
            + 0.81 * energies.lyp_correlation,
            places=13,
        )
        with self.assertRaises(ValueError):
            b3lyp_exchange_correlation_energy(
                -1.25,
                b3lyp_semilocal_energies(
                    self.density, self.grid, vwn_parametrization="vwn5"
                ),
            )

    def test_core_density_enters_as_in_the_self_consistent_functional(self) -> None:
        core = 0.3 * self.density
        with_core = b3lyp_semilocal_energies(self.density, self.grid, core)
        summed = b3lyp_semilocal_energies(self.density + core, self.grid)
        self.assertEqual(with_core, summed)
        self.assertNotEqual(with_core, b3lyp_semilocal_energies(self.density, self.grid))


class ExactExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(spacing=0.4, radius=8.0, expansion_order=8)
        )
        cls.laplacian = build_negative_laplacian(cls.grid)
        cls.settings = HartreeSettings(
            multipole_order=4, relative_tolerance=1.0e-10, max_iterations=4000
        )
        cls.solver = staticmethod(
            reference_pair_poisson(cls.grid, cls.laplacian, cls.settings)
        )
        cls.exponent = 0.4
        points = cls.grid.coordinates
        gaussian = np.exp(-cls.exponent * np.einsum("ij,ij->i", points, points))
        # Grid vectors q = sqrt(h**3) psi, normalized on the grid.
        cls.s_orbital = gaussian / np.linalg.norm(gaussian)
        p_orbital = points[:, 0] * gaussian
        cls.p_orbital = p_orbital / np.linalg.norm(p_orbital)
        cls.orbitals = np.column_stack(
            (cls.s_orbital, cls.p_orbital, np.zeros(cls.grid.size))
        )

    def exchange(self, occupations):
        return exact_exchange_energy(
            self.orbitals,
            np.asarray(occupations, dtype=float),
            self.grid.volume_element,
            self.solver,
        )

    def test_two_electrons_exchange_is_minus_half_the_hartree_energy(self) -> None:
        result = self.exchange([1.0, 0.0, 0.0])
        density = 2.0 * self.s_orbital**2 / self.grid.volume_element
        hartree = solve_hartree(
            density, self.grid, self.laplacian, pair_hartree_settings(self.settings)
        )
        hartree_energy = 0.5 * self.grid.volume_element * float(
            density @ hartree.potential
        )
        self.assertAlmostEqual(result.energy / (-0.5 * hartree_energy), 1.0, delta=1.0e-9)
        # (ss|ss) = 2 sqrt(a/pi) Hartree for a normalized Gaussian orbital.
        analytic = -2.0 * 2.0 * np.sqrt(self.exponent / np.pi)
        self.assertAlmostEqual(result.energy, analytic, delta=2.0e-5)
        self.assertEqual(result.orbitals, (0,))
        self.assertEqual(result.pair_solves, 1)
        self.assertGreater(result.poisson_iterations, 0)
        self.assertFalse(result.fractional)

    def test_cross_pair_matches_the_gaussian_exchange_integral(self) -> None:
        both = self.exchange([1.0, 1.0, 0.0])
        only_s = self.exchange([1.0, 0.0, 0.0])
        only_p = self.exchange([0.0, 1.0, 0.0])
        self.assertEqual(both.pair_solves, 3)
        self.assertEqual(both.orbitals, (0, 1))
        self.assertEqual(only_p.orbitals, (1,))
        # E = -[(ss|ss) + (pp|pp) + 2 (sp|sp)]; (sp|sp) = sqrt(a/pi)/3 Hartree.
        cross = -(both.energy - only_s.energy - only_p.energy) / 2.0
        analytic = 2.0 * np.sqrt(self.exponent / np.pi) / 3.0
        self.assertAlmostEqual(cross, analytic, delta=2.0e-5)

    def test_fractional_occupations_weight_the_density_matrix(self) -> None:
        calls: list[np.ndarray] = []

        def kernel(pair_density):
            calls.append(pair_density.copy())
            return 3.0 * pair_density

        occupations = np.array([1.0, 0.25, 0.0])
        result = exact_exchange_energy(
            self.orbitals, occupations, self.grid.volume_element, kernel
        )
        # An empty orbital starts no Poisson solve.
        self.assertEqual(len(calls), 3)
        volume = self.grid.volume_element
        expected = 0.0
        for i in range(2):
            for j in range(2):
                pair = self.orbitals[:, i] * self.orbitals[:, j] / volume
                expected -= occupations[i] * occupations[j] * volume * 3.0 * (pair @ pair)
        self.assertAlmostEqual(result.energy, expected, places=12)
        self.assertTrue(result.fractional)
        self.assertAlmostEqual(result.largest_fractional_deviation, 0.25)
        # A smeared closed shell is not reported as fractional.
        smeared = exact_exchange_energy(
            self.orbitals,
            np.array([1.0 - 1.0e-12, 1.0e-14, 0.0]),
            volume,
            kernel,
        )
        self.assertFalse(smeared.fractional)
        self.assertEqual(smeared.orbitals, (0,))

    def test_what_cannot_be_evaluated_is_refused(self) -> None:
        volume = self.grid.volume_element
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.exchange([2.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            occupied_columns(np.array([1.0, -0.5]))
        with self.assertRaisesRegex(ValueError, "one occupation per"):
            exact_exchange_energy(self.orbitals, np.ones(2), volume, self.solver)
        with self.assertRaisesRegex(ValueError, "not normalized"):
            exact_exchange_energy(
                2.0 * self.orbitals, np.array([1.0, 0.0, 0.0]), volume, self.solver
            )
        with self.assertRaises(NotImplementedError):
            exact_exchange_energy(
                self.orbitals.astype(complex), np.array([1.0, 0.0, 0.0]), volume, self.solver
            )
        with self.assertRaisesRegex(TypeError, "host array"):
            exact_exchange_energy(
                SimpleNamespace(ndim=2), np.array([1.0]), volume, self.solver
            )

    def test_pair_settings_drop_the_atomic_tail_only(self) -> None:
        settings = HartreeSettings(
            multipole_order=14, atomic_tail="on", minimum_multipole_order=9
        )
        pair = pair_hartree_settings(settings)
        self.assertEqual(pair.atomic_tail, "off")
        self.assertIsNone(pair.boundary_tolerance)
        self.assertEqual(pair.multipole_order, 14)
        self.assertEqual(pair.relative_tolerance, settings.relative_tolerance)
        self.assertEqual(pair.boundary_method, settings.boundary_method)


class RefinementAssemblyTests(unittest.TestCase):
    """The refined energy from a stand-in for a prepared system and its result."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_cluster_grid(
            GridSettings(spacing=0.5, radius=7.0, expansion_order=8)
        )
        points = cls.grid.coordinates
        gaussian = np.exp(-0.45 * np.einsum("ij,ij->i", points, points))
        orbital = gaussian / np.linalg.norm(gaussian)
        cls.wavefunctions = np.column_stack((orbital, np.roll(orbital, 7)))
        cls.density = 2.0 * orbital**2 / cls.grid.volume_element
        cls.system = SimpleNamespace(
            grid=cls.grid,
            negative_laplacian=build_negative_laplacian(cls.grid),
            input=SimpleNamespace(
                periodic_cell=None,
                hartree=HartreeSettings(multipole_order=2),
                scf=SimpleNamespace(xc_functional="pbe"),
            ),
        )

    def result(self, core=None, occupations=(1.0, 0.0), converged=True):
        core_density = np.zeros(self.grid.size) if core is None else core
        occupations = np.asarray(occupations, dtype=float)
        density = (
            2.0 * (self.wavefunctions**2 @ occupations) / self.grid.volume_element
        )
        scf_xc = pbe(density, self.grid, core_density).total_energy
        return SimpleNamespace(
            density=density,
            core_density=core_density,
            wavefunctions=self.wavefunctions,
            occupations=occupations,
            converged=converged,
            atomic_reference_correction=-12.5,
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

    def test_refined_total_replaces_the_exchange_correlation_energy(self) -> None:
        result = self.result()
        refinement = refine_energy(self.system, result)
        semilocal = b3lyp_semilocal_energies(self.density, self.grid)
        self.assertEqual(refinement.functional, "b3lyp")
        self.assertEqual(refinement.scf_functional, "pbe")
        self.assertEqual(refinement.slater_exchange, semilocal.slater_exchange)
        self.assertEqual(refinement.lyp_correlation, semilocal.lyp_correlation)
        expected_xc = (
            0.20 * refinement.exact_exchange
            + 0.08 * semilocal.slater_exchange
            + 0.72 * semilocal.b88_exchange
            + 0.19 * semilocal.vwn_correlation
            + 0.81 * semilocal.lyp_correlation
        )
        self.assertAlmostEqual(refinement.exchange_correlation, expected_xc, places=13)
        scf_xc = result.energies.exchange_correlation
        self.assertEqual(refinement.scf_exchange_correlation, scf_xc)
        self.assertEqual(refinement.scf_total, -3.0)
        self.assertAlmostEqual(refinement.total, -3.0 - scf_xc + expected_xc, places=13)
        self.assertAlmostEqual(refinement.correction, expected_xc - scf_xc, places=13)
        # The atomic reference correction is the one of the SCF functional.
        self.assertAlmostEqual(refinement.all_electron_total, refinement.total - 12.5)
        self.assertAlmostEqual(refinement.scf_all_electron_total, -15.5)
        self.assertFalse(refinement.core_correction)
        # Two electrons in one orbital: exact exchange = -E_Hartree / 2.
        hartree = solve_hartree(
            self.density, self.grid, self.system.negative_laplacian,
            self.system.input.hartree,
        )
        hartree_energy = 0.5 * self.grid.volume_element * float(
            self.density @ hartree.potential
        )
        self.assertAlmostEqual(
            refinement.exact_exchange / (-0.5 * hartree_energy), 1.0, delta=1.0e-6
        )

    def test_report_names_every_energy_and_condition(self) -> None:
        refinement = refine_energy(self.system, self.result())
        text = "\n".join(refinement_report_lines(refinement, 13.6058))
        for expected in (
            "Energy refinement: B3LYP exchange-correlation energy of the PBE orbitals",
            "Exact (Hartree-Fock) exchange",
            "VWN (RPA) correlation",
            "Exc (PBE, self-consistent)",
            "Refined Total Energy (B3LYP)",
            "B3LYP-refined all-electron total",
            "1 occupied orbitals, 1 pair-density Poisson solves",
            "Time for energy refinement [sec]",
        ):
            self.assertIn(expected, text)
        # The line of the SCF report is not repeated under its own label.
        self.assertNotIn("Reference-corrected all-electron total", text)
        value = re.search(
            r"B3LYP-refined all-electron total\s*:\s*([-0-9.]+) \[eV\]", text
        )
        self.assertAlmostEqual(
            float(value.group(1)), refinement.all_electron_total * 13.6058, places=6
        )
        for note in ("fractional occupations", "nonlinear core", "did not converge"):
            self.assertNotIn(note, text)

        core = 0.1 * self.density
        flagged = refine_energy(
            self.system,
            self.result(core=core, occupations=(1.0, 0.5), converged=False),
        )
        text = "\n".join(refinement_report_lines(flagged, 13.6058))
        self.assertTrue(flagged.core_correction)
        self.assertTrue(flagged.exchange.fractional)
        for note in ("fractional occupations", "nonlinear core", "did not converge"):
            self.assertIn(note, text)
        # The density functionals see valence plus core density.
        density = self.result(occupations=(1.0, 0.5)).density
        self.assertEqual(
            flagged.slater_exchange,
            b3lyp_semilocal_energies(density, self.grid, core).slater_exchange,
        )

    def test_requests_that_cannot_be_served(self) -> None:
        periodic = SimpleNamespace(
            grid=self.grid,
            negative_laplacian=self.system.negative_laplacian,
            input=SimpleNamespace(
                periodic_cell=object(),
                hartree=HartreeSettings(),
                scf=SimpleNamespace(xc_functional="ca"),
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "isolated systems"):
            refine_energy(periodic, self.result())
        with self.assertRaises(ValueError):
            refine_energy(self.system, self.result(), "pbe0")
        # Orbitals that are not those of the density of the result.
        stale = self.result()
        stale.wavefunctions = self.wavefunctions[:, ::-1].copy()
        with self.assertRaisesRegex(ValueError, "do not give its density"):
            refine_energy(self.system, stale)
        self.assertLess(
            orbital_density_deviation(
                self.wavefunctions,
                np.array([1.0, 0.0]),
                self.density,
                self.grid.volume_element,
            ),
            1.0e-14,
        )
        self.assertIsNone(normalize_refinement(None))
        for off in ("none", "off", "false", ".false.", "No"):
            self.assertIsNone(normalize_refinement(off))
        for on in ("b3lyp", "B3LYP", " b3-lyp "):
            self.assertEqual(normalize_refinement(on), "b3lyp")
        with self.assertRaisesRegex(ValueError, "supported choices"):
            normalize_refinement("hse06")


class RefinementInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        shutil.copy(DATA / "H_POTRE.DAT", self.directory / "H_POTRE.DAT")
        self.base = (DATA / "H_cli_smoke.in").read_text()

    def parse(self, extra: str = ""):
        path = self.directory / "parsec.in"
        path.write_text(self.base + extra)
        return parse_parsec_input(path)

    def test_keyword_is_off_by_default_and_changes_no_problem(self) -> None:
        plain = self.parse()
        self.assertIsNone(plain.energy_refinement)
        self.assertNotIn("refinement", summarize_translation(plain).lower())
        requested = self.parse("\nEnergy_Refinement: b3lyp\n")
        self.assertEqual(requested.energy_refinement, "b3lyp")
        self.assertIn("Energy refinement: b3lyp", summarize_translation(requested))
        # The request is not part of the problem the SCF solves.
        self.assertEqual(repr(requested.problem), repr(plain.problem))
        for off in ("none", "off", ".false."):
            self.assertIsNone(self.parse(f"\nEnergy_Refinement: {off}\n").energy_refinement)

    def test_keyword_is_refused_where_the_refinement_does_not_exist(self) -> None:
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "supported choices"):
            self.parse("\nEnergy_Refinement: pbe0\n")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Molecular_Dynamics"):
            self.parse("\nEnergy_Refinement: b3lyp\nMolecular_Dynamics: true\n")
        periodic = EXAMPLES / "3d_Si" / "parsec.in"
        if periodic.is_file():
            target = self.directory / "periodic"
            shutil.copytree(periodic.parent, target)
            path = target / "parsec.in"
            path.write_text(path.read_text() + "\nEnergy_Refinement: b3lyp\n")
            with self.assertRaisesRegex(UnsupportedParsecOptionError, "isolated systems"):
                parse_parsec_input(path)

    def test_command_line_option_overrules_the_input(self) -> None:
        plain = self.parse()
        self.assertIs(apply_refinement_option(plain, None), plain)
        self.assertEqual(
            apply_refinement_option(plain, "b3lyp").energy_refinement, "b3lyp"
        )
        requested = self.parse("\nEnergy_Refinement: b3lyp\n")
        self.assertIsNone(apply_refinement_option(requested, "none").energy_refinement)
        self.assertIs(apply_refinement_option(requested, None), requested)


class ReferenceCommandLineTests(unittest.TestCase):
    """H2, one doubly occupied orbital, through the reference command line.

    The bundled hydrogen pseudopotential is a CA-LDA one, so the refinement
    here follows an LDA SCF; the algebra is that of a PBE one.
    """

    INPUT = """Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 6.0 bohr
Grid_Spacing: 0.6 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 8
States_Num: 2
Fermi_Temp: 80
Max_Iter: 60
Convergence_Criterion: 1.0d-6 Ry
Eigensolver: chebff
Subspace_Buffer_Size: 6
Mixing_Method: Anderson
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
Read_VCD: false
Atomic_Energy_Correction: -0.25 Ry
begin Atom_Coord
    0.0  0.0  0.7
    0.0  0.0 -0.7
end Atom_Coord
Correlation_Type: ca
"""

    def run_cli(self, directory: Path, *options: str) -> str:
        directory.mkdir()
        shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
        path = directory / "parsec.in"
        path.write_text(self.INPUT)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = cli_module.main([str(path), "--quiet", *options])
        self.assertEqual(code, 0)
        return (directory / "parsec.out").read_text()

    def test_progress_lines_are_spaced_in_time(self) -> None:
        lines: list[str] = []
        announce, progress = cli_module.refinement_progress(
            lines.append, "b3lyp", interval=10.0
        )
        announce()
        for done, seconds in ((1, 3.0), (2, 12.0), (3, 15.0), (4, 23.0), (5, 30.0)):
            progress(done, 5, seconds)
        self.assertEqual(len(lines), 3)
        self.assertIn("evaluating the B3LYP exchange-correlation energy", lines[0])
        self.assertIn("2 of 5 pair-density Poisson solves done after 12.0 s", lines[1])
        self.assertIn("4 of 5", lines[2])

    def test_refinement_follows_the_scf_and_leaves_it_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            plain = self.run_cli(root / "plain", "--no-archive")
            refined = self.run_cli(root / "refined", "--energy-refinement", "b3lyp")
            archive = np.load(root / "refined" / "parsec_python_results.npz")
            archive = {name: archive[name] for name in archive.files}

        self.assertNotIn("refinement", plain.lower())
        self.assertIn("Post-SCF energy refinement: B3LYP", refined)
        self.assertIn(
            "Energy refinement: B3LYP exchange-correlation energy of the "
            "CA-LDA orbitals and density",
            refined,
        )

        def scf_lines(text: str) -> list[str]:
            start = text.index(" SCF iter #   1")
            end = text.index("Self-consistency convergence achieved.")
            return [
                line
                for line in text[start:end].splitlines()
                if "time [sec]" not in line and "tdiag_sum" not in line
            ]

        # Every energy, eigenvalue and residual of the SCF is the same text.
        self.assertEqual(scf_lines(plain), scf_lines(refined))

        def value(label: str) -> float:
            match = re.search(re.escape(label) + r"\s*[=:]\s*([-0-9.]+) \[Ry\]", refined)
            self.assertIsNotNone(match, label)
            return float(match.group(1))

        exchange = value("Exact (Hartree-Fock) exchange")
        hartree = float(archive["energy_hartree_ry"])
        # Two electrons in one orbital: E_x = -E_Hartree / 2.
        self.assertAlmostEqual(exchange / (-0.5 * hartree), 1.0, delta=2.0e-6)
        total = float(archive["energy_total_ry"])
        scf_xc = float(archive["energy_exchange_correlation_ry"])
        refined_xc = float(archive["refinement_exchange_correlation_ry"])
        self.assertAlmostEqual(
            float(archive["refinement_total_ry"]), total - scf_xc + refined_xc, places=12
        )
        self.assertAlmostEqual(value("Refined Total Energy (B3LYP)"), total - scf_xc + refined_xc, places=7)
        self.assertAlmostEqual(
            value("B3LYP-refined all-electron total"),
            total - scf_xc + refined_xc - 0.5,
            places=7,
        )
        self.assertEqual(str(archive["refinement_functional"]), "b3lyp")
        self.assertEqual(int(archive["refinement_pair_solves"]), 1)
        self.assertAlmostEqual(float(archive["refinement_exact_exchange_ry"]), exchange, places=7)


if __name__ == "__main__":
    unittest.main()
