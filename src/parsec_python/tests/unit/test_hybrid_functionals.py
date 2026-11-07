"""Hybrid functional definitions and their semilocal parts (``V_xc/hybrid.py``).

The registry (CAM coefficients, kernel decompositions, overrides), the
semilocal evaluator on the grid (the discrete variational potential is the
derivative of the discrete energy; equal spin channels give the unpolarized
result; PBEh(0) is PBE bit for bit), and the pointwise components against
libxc 7.1.2 reference values (``tests/hybrid_libxc_reference.py``, made with
pylibxc in WSL; libxc is a reference only, not a dependency).
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from importlib import import_module
import unittest
from unittest import mock

import numpy as np

from parsec_python import GridSettings
from parsec_python.Grid import build_cluster_grid
from parsec_python.V_xc import pbe, pbe_energy_partials
from parsec_python.V_xc.hybrid import (
    HYBRID_FUNCTIONALS,
    ExactExchangeTerm,
    canonical_hybrid_label,
    hybrid_functional,
    hybrid_semilocal_xc,
    hybrid_semilocal_xc_spin,
    is_hybrid_label,
)
from parsec_python.V_xc.pbe import _exchange_partials
from parsec_python.V_xc.pbe_spin import pbe_spin
from parsec_python.V_xc.semilocal import SemilocalXC
from parsec_python.V_xc.semilocal_components import ComponentSemilocal, SemilocalComponent
from parsec_python.tests.hybrid_libxc_reference import POLARIZED_INPUTS, REFERENCE, UNPOLARIZED_INPUTS


def _cluster_grid():
    return build_cluster_grid(GridSettings(spacing=0.6, radius=3.4, expansion_order=8, shift=(0.5, 0.5, 0.5)))


def _density(grid, centers=((0.4, -0.3, 0.2), (-0.9, 0.7, -0.4)), weights=(0.9, 0.5), width=1.3):
    coordinates = grid.coordinates
    values = np.zeros(grid.size)
    for center, weight in zip(centers, weights):
        difference = coordinates - np.asarray(center)
        values += weight * np.exp(-np.sum(difference * difference, axis=1) / width)
    return values


def _implemented(functional) -> bool:
    return getattr(functional.semilocal, "implemented", True)


class RegistryTests(unittest.TestCase):
    def test_cam_coefficients_and_kernels(self) -> None:
        expected = {
            "hf": ((1.0, 0.0, 0.0), (("coulomb", 1.0),)),
            "pbe0": ((0.25, 0.0, 0.0), (("coulomb", 0.25),)),
            "pbeh": ((0.25, 0.0, 0.0), (("coulomb", 0.25),)),
            "hse06": ((0.0, 0.25, 0.11), (("erfc", 0.25),)),
            "b3lyp": ((0.20, 0.0, 0.0), (("coulomb", 0.20),)),
            "b3pw91": ((0.20, 0.0, 0.0), (("coulomb", 0.20),)),
            "lc-wpbe": ((1.0, -1.0, 0.4), (("erf", 1.0),)),
            "cam-b3lyp": ((0.65, -0.46, 0.33), (("coulomb", 0.19), ("erf", 0.46))),
        }
        for label, (cam, terms) in expected.items():
            functional = HYBRID_FUNCTIONALS[label]
            np.testing.assert_allclose(functional.cam_coefficients, cam, rtol=0, atol=1e-15, err_msg=label)
            got = tuple((term.kernel, term.fraction) for term in functional.exact_exchange_terms)
            self.assertEqual(tuple(kernel for kernel, _ in got), tuple(kernel for kernel, _ in terms), label)
            np.testing.assert_allclose([f for _, f in got], [f for _, f in terms], rtol=0, atol=1e-15)
        hse03 = HYBRID_FUNCTIONALS["hse03"]
        self.assertAlmostEqual(hse03.omega, 0.15 / np.sqrt(2.0), places=15)
        omega_pbe = [c.omega for c in hse03.components if c.name == "wpbe_sr_x"]
        self.assertAlmostEqual(omega_pbe[0], 0.15 * 2.0 ** (1.0 / 3.0), places=15)

    def test_kernel_terms_reproduce_the_cam_kernel(self) -> None:
        # alpha/r + beta erfc(w r)/r == sum of the terms, at a few distances.
        from scipy.special import erf, erfc

        r = np.array([0.05, 0.3, 1.0, 2.5, 7.0])
        for functional in HYBRID_FUNCTIONALS.values():
            alpha, beta, omega = functional.cam_coefficients
            cam = alpha / r + (beta * erfc(omega * r) / r if beta else 0.0)
            total = np.zeros_like(r)
            for term in functional.exact_exchange_terms:
                kernel = {"coulomb": 1.0 / r}.get(term.kernel)
                if kernel is None:
                    kernel = (erfc if term.kernel == "erfc" else erf)(term.omega * r) / r
                total += term.fraction * kernel
            np.testing.assert_allclose(total, cam, rtol=1e-14, atol=0, err_msg=functional.label)
            spectrum = functional.exchange_kernel_spectrum(np.array([0.01, 1.0, 30.0]))
            k2 = np.array([0.01, 1.0, 30.0])
            screened = 0.0 if not beta else beta * (1.0 - np.exp(-k2 / (4.0 * omega * omega)))
            np.testing.assert_allclose(spectrum, 4 * np.pi / k2 * (alpha + screened), rtol=1e-14)

    def test_labels_and_overrides(self) -> None:
        self.assertEqual(canonical_hybrid_label("PBE0"), "pbe0")
        self.assertEqual(canonical_hybrid_label("LC_wPBE"), "lc-wpbe")
        self.assertEqual(canonical_hybrid_label("camb3lyp"), "cam-b3lyp")
        self.assertEqual(canonical_hybrid_label("HSE"), "hse06")
        self.assertTrue(is_hybrid_label("b3lyp"))
        self.assertFalse(is_hybrid_label("pbe"))
        pbeh = hybrid_functional("pbeh", exact_exchange_fraction=0.4)
        self.assertEqual(pbeh.alpha, 0.4)
        self.assertEqual([(c.name, c.weight) for c in pbeh.components], [("pbe_x", 0.6), ("pbe_c", 1.0)])
        hse = hybrid_functional("hse06", screening_parameter=0.106)
        self.assertEqual(hse.omega, 0.106)
        self.assertEqual([c.omega for c in hse.components if c.name == "wpbe_sr_x"], [0.106])
        hse = hybrid_functional("hse06", exact_exchange_fraction=0.3)
        self.assertEqual(hse.beta, 0.3)
        self.assertEqual([c.weight for c in hse.components if c.name == "wpbe_sr_x"], [-0.3])
        self.assertEqual(hybrid_functional("pbe0", exact_exchange_fraction=0.25).alpha, 0.25)
        for label, keywords in (
            ("pbe0", {"exact_exchange_fraction": 0.3}),
            ("b3lyp", {"exact_exchange_fraction": 0.25}),
            ("hf", {"exact_exchange_fraction": 0.5}),
            ("pbe0", {"screening_parameter": 0.2}),
            ("hse03", {"screening_parameter": 0.2}),
            ("pbeh", {"exact_exchange_fraction": 1.5}),
            ("hse06", {"screening_parameter": -0.1}),
        ):
            with self.assertRaises(ValueError, msg=(label, keywords)):
                hybrid_functional(label, **keywords)
        with self.assertRaises(ValueError):
            hybrid_functional("pbe")
        with self.assertRaises(ValueError):
            ExactExchangeTerm("coulomb", 1.0, 0.1)

    def test_every_entry_is_documented(self) -> None:
        for functional in HYBRID_FUNCTIONALS.values():
            self.assertIsInstance(functional.semilocal, SemilocalXC)
            self.assertTrue(functional.references and functional.libxc and functional.variants, functional.label)
            self.assertIn(functional.name, functional.describe())
            weights = sum(c.weight for c in functional.components if c.kind == "exchange")
            if functional.label in ("hse06", "hse03", "lc-wpbe", "cam-b3lyp"):
                continue
            # Global hybrids: semilocal plus exact exchange fractions add up to one.
            self.assertAlmostEqual(weights + functional.alpha, 1.0, places=14, msg=functional.label)


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = _cluster_grid()
        cls.density = _density(cls.grid)
        cls.core = 0.05 * _density(cls.grid, centers=((0.4, -0.3, 0.2),), weights=(1.0,), width=0.4)

    def test_pbeh_zero_is_pbe_bit_for_bit(self) -> None:
        functional = hybrid_functional("pbeh", exact_exchange_fraction=0.0)
        hybrid = hybrid_semilocal_xc(functional, self.density, self.grid, self.core)
        reference = pbe(self.density, self.grid, self.core)
        for name in ("potential", "energy_density", "energy_per_electron"):
            np.testing.assert_array_equal(getattr(hybrid, name), getattr(reference, name), err_msg=name)
        self.assertEqual(hybrid.total_energy, reference.total_energy)
        spin = hybrid_semilocal_xc_spin(functional, 0.6 * self.density, 0.4 * self.density, self.grid, self.core)
        spin_reference = pbe_spin(0.6 * self.density, 0.4 * self.density, self.grid, self.core)
        for name in ("potential_up", "potential_down", "energy_density"):
            np.testing.assert_allclose(
                getattr(spin, name), getattr(spin_reference, name), rtol=1e-13, atol=1e-13, err_msg=name
            )

    def test_pbe0_semilocal_part(self) -> None:
        functional = hybrid_functional("pbe0")
        density = self.density + self.core
        sigma = np.abs(np.random.default_rng(3).normal(size=density.size)) * density**2
        e, en, es = functional.semilocal.unpolarized(density, sigma)
        f, fn, fs = pbe_energy_partials(density, sigma)
        x, xn, xs = _exchange_partials(density, sigma)
        np.testing.assert_allclose(e, f - 0.25 * x, rtol=1e-13, atol=1e-300)
        np.testing.assert_allclose(en, fn - 0.25 * xn, rtol=1e-12)
        np.testing.assert_allclose(es, fs - 0.25 * xs, rtol=1e-12)

    def test_hf_has_no_semilocal_part(self) -> None:
        functional = hybrid_functional("hf")
        result = hybrid_semilocal_xc(functional, self.density, self.grid, self.core)
        self.assertEqual(result.total_energy, 0.0)
        self.assertFalse(np.any(result.potential))
        spin = hybrid_semilocal_xc_spin(functional, self.density, 0 * self.density, self.grid)
        self.assertEqual(spin.total_energy, 0.0)

    def _check_variational(self, functional) -> None:
        grid = self.grid
        rng = np.random.default_rng(11)
        direction = rng.normal(size=grid.size) * self.density
        result = hybrid_semilocal_xc(functional, self.density, grid, self.core)
        step = 1.0e-5
        plus = hybrid_semilocal_xc(functional, self.density + step * direction, grid, self.core).total_energy
        minus = hybrid_semilocal_xc(functional, self.density - step * direction, grid, self.core).total_energy
        derivative = (plus - minus) / (2.0 * step)
        predicted = grid.volume_element * float(result.potential @ direction)
        self.assertAlmostEqual(derivative / predicted, 1.0, delta=1e-7, msg=functional.label)
        up, down = 0.7 * self.density, 0.3 * self.density + 0.01 * self.core
        spin = hybrid_semilocal_xc_spin(functional, up, down, grid, self.core)
        for index, potential in enumerate((spin.potential_up, spin.potential_down)):
            shifted = [up, down]
            shifted[index] = shifted[index] + step * direction
            plus = hybrid_semilocal_xc_spin(functional, *shifted, grid, self.core).total_energy
            shifted[index] = shifted[index] - 2.0 * step * direction
            minus = hybrid_semilocal_xc_spin(functional, *shifted, grid, self.core).total_energy
            derivative = (plus - minus) / (2.0 * step)
            predicted = grid.volume_element * float(potential @ direction)
            self.assertAlmostEqual(derivative / predicted, 1.0, delta=1e-7, msg=(functional.label, index))
        # Equal channels: the unpolarized functional.
        half = 0.5 * self.density
        spin = hybrid_semilocal_xc_spin(functional, half, half, grid, self.core)
        np.testing.assert_allclose(spin.potential_up, result.potential, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(spin.potential_down, result.potential, rtol=1e-11, atol=1e-12)
        self.assertAlmostEqual(spin.total_energy, result.total_energy, delta=1e-12 * abs(result.total_energy))

    def test_potentials_are_derivatives_of_the_energy(self) -> None:
        for functional in HYBRID_FUNCTIONALS.values():
            if functional.label == "hf" or not _implemented(functional):
                continue
            with self.subTest(functional.label):
                self._check_variational(functional)


@contextmanager
def _libxc_digits():
    """The digits of libxc in the PBE, PW92 and PW91 of this code (Fortran PARSEC's and pbe.py's otherwise)."""

    pbe_module = import_module("parsec_python.V_xc.pbe")
    pbe_spin_module = import_module("parsec_python.V_xc.pbe_spin")
    pw91_module = import_module("parsec_python.V_xc.pw91")
    pw92_module = import_module("parsec_python.V_xc.pw92_lda")

    def pw92_unpolarized(rs):  # libxc PW_MOD: A = 0.0310907
        return pbe_module._pw92_g(rs, 0.0310907, 0.21370, (7.5957, 3.5876, 1.6382, 0.49294))

    nu = 16.0 / np.pi * (3.0 * np.pi**2) ** (1.0 / 3.0)
    with ExitStack() as stack:
        for module, name, value in (
            (pbe_module, "_pw92_unpolarized", pw92_unpolarized),
            (pbe_spin_module, "_pw92_unpolarized", pw92_unpolarized),
            (pbe_spin_module, "_SPIN_ETA", 0.0),
            (pw92_module, "_UNPOLARIZED", (0.031091, 0.21370, (7.5957, 3.5876, 1.6382, 0.49294))),
            (pw92_module, "_POLARIZED", (0.015545, 0.20548, (14.1189, 6.1977, 3.3662, 0.62517))),
            (pw92_module, "_STIFFNESS", (0.016887, 0.11125, (10.357, 3.6231, 0.88026, 0.49671))),
            (pw92_module, "_GAM", 2.0 ** (4.0 / 3.0) - 2.0),
            (pw91_module, "_XNU", nu),
            (pw91_module, "_CX", -0.001667),
            (pw91_module, "_C6", 0.0),
            (pw91_module, "_KS2_OVER_KF2", 4.0 / np.pi * (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)),
            (pw91_module, "_BET", nu * pw91_module._CC0),
            (pw91_module, "_DELT", 2.0 * pw91_module._ALF / (nu * pw91_module._CC0)),
            (pw91_module, "_A3", 0.1508),
            (pw91_module, "_SPIN_ETA", 0.0),
        ):
            stack.enter_context(mock.patch.object(module, name, value))
        yield


class _HendersonWPBE:
    """wPBE-SR with libxc's rescaling of s (for libxc 6.2.2, which has Heyd's constants)."""

    family = "gga"

    def __init__(self, omega):
        from parsec_python.V_xc.semilocal_components import positive_unpolarized, spin_scaled_exchange
        from parsec_python.V_xc.wpbe import wpbe_sr_exchange_partials

        core = lambda n, sigma, om: wpbe_sr_exchange_partials(n, sigma, om, rescaling="henderson")  # noqa: E731
        self._u, self._p, self.omega = positive_unpolarized(core), spin_scaled_exchange(core), omega

    def unpolarized(self, rho, sigma):
        return self._u(np.asarray(rho), np.asarray(sigma), self.omega)

    def polarized(self, *arrays):
        return self._p(*(np.asarray(a) for a in arrays), self.omega)


def _reference_evaluator(case: str):
    if case.startswith("hyb:"):
        return HYBRID_FUNCTIONALS[case[4:]].semilocal
    if case.startswith("wpbe_henderson("):
        return _HendersonWPBE(float(case[len("wpbe_henderson("):-1]))
    omega = None
    name = case
    if "(" in case:
        name, argument = case[:-1].split("(")
        omega = float(argument)
    return ComponentSemilocal((SemilocalComponent(name, 1.0, omega),))


def _worst(ours, reference) -> float:
    worst = 0.0
    for mine, ref in zip(ours, reference):
        mine, ref = np.asarray(mine), np.asarray(ref)
        scale = np.max(np.abs(ref))
        if scale == 0.0:
            worst = max(worst, float(np.max(np.abs(mine))))
            continue
        keep = np.abs(ref) > 1e-12 * scale
        worst = max(worst, float(np.max(np.abs(mine[keep] - ref[keep]) / np.abs(ref[keep]))))
    return worst


class LibxcReferenceTests(unittest.TestCase):
    """Agreement with libxc 7.1.2 (6.2.2 for Heyd's wPBE constants) at well-conditioned points."""

    #: Same definitions as libxc: agreement to round-off.
    SAME = ("slater_x", "pbe_x", "b88_x", "vwn_rpa_c", "vwn5_c", "lyp_c", "b88_sr_x(0.33)",
            "hyb:b3lyp", "hyb:b3lyp5", "hyb:cam-b3lyp")
    #: Same formulas, other digits (Fortran PARSEC's and pbe.py's): agreement with libxc's digits patched in,
    #: and the size of the energy difference without.
    DIGITS = {"pbe_c": 3e-6, "pw92_c": 3e-5, "pw91_c": 1e-3, "pw91_x": 3e-7, "hyb:pbe0": 1e-7, "hyb:b3pw91": 3e-4}

    def _check(self, case, tolerance, rows=slice(None)):
        reference = REFERENCE[case]
        evaluator = _reference_evaluator(case)
        unpolarized = evaluator.unpolarized(*map(np.asarray, UNPOLARIZED_INPUTS))
        polarized = evaluator.polarized(*map(np.asarray, POLARIZED_INPUTS))
        self.assertLess(_worst(unpolarized[rows], reference["unpolarized"][rows]), tolerance, (case, "unpolarized"))
        self.assertLess(_worst(polarized[rows], reference["polarized"][rows]), tolerance, (case, "polarized"))

    def test_same_definitions(self) -> None:
        for case in self.SAME:
            with self.subTest(case):
                self._check(case, 2e-12)

    def test_other_digits(self) -> None:
        for case, difference in self.DIGITS.items():
            with self.subTest(case):
                self._check(case, difference, rows=slice(0, 1))
                with _libxc_digits():
                    self._check(case, 2e-12)

    def test_heyd_wpbe_against_libxc_6(self) -> None:
        # Heyd's integration cancels digits as nu = omega/k_F grows (V_xc/wpbe.py): 1e-10 for nu < 3.
        for case in ("wpbe_henderson(0.11)", "wpbe_henderson(0.4)"):
            with self.subTest(case):
                self._check(case, 1e-10)


class ComponentTests(unittest.TestCase):
    def test_scaled_exp1(self) -> None:
        from scipy.special import exp1

        from parsec_python.V_xc.wpbe import scaled_exp1

        x = np.concatenate([10.0 ** np.linspace(-12, 0, 200), np.linspace(1.0, 700.0, 500)])
        np.testing.assert_allclose(scaled_exp1(x), np.exp(x) * exp1(x), rtol=2e-15, atol=0)

    def test_erf_attenuation_series(self) -> None:
        # The large-a series and the closed form meet at a = 1.35; the series is 1/(36 a**2) - 1/(960 a**4) + ...
        from parsec_python.V_xc import ityh

        self.assertAlmostEqual(ityh._SERIES[0], 1.0 / 36.0, places=17)
        self.assertAlmostEqual(ityh._SERIES[1], -1.0 / 960.0, places=17)
        below = np.nextafter(ityh._SERIES_CUTOFF, 0.0)
        exact, exact_slope = ityh.erf_attenuation(np.array([below]))
        series, series_slope = ityh.erf_attenuation(np.array([ityh._SERIES_CUTOFF]))
        self.assertAlmostEqual(exact[0] / series[0], 1.0, delta=1e-13)
        self.assertAlmostEqual(exact_slope[0] / series_slope[0], 1.0, delta=1e-12)

    def test_wpbe_branches_and_limits(self) -> None:
        from parsec_python.V_xc.wpbe import wpbe_enhancement

        # Continuity across s = 0.08 (EG polynomial) and s = 8.3 (rescaling), and the omega -> large limit.
        for s0 in (0.08, 8.3):
            s = np.array([s0, np.nextafter(s0, np.inf)])
            for nu in (0.05, 1.0):
                value = wpbe_enhancement(s, np.full(2, nu))[0]
                self.assertAlmostEqual(value[0], value[1], delta=2e-7 * abs(value[0]), msg=(s0, nu))
        value = wpbe_enhancement(np.array([0.5]), np.array([60.0]))[0]
        self.assertLess(abs(value[0]), 1e-3)
        # F(s, nu) -> PBE-like full-range hole for nu -> 0 (EP98 reproduces F_x^PBE to its fit).
        s = np.array([0.0, 0.5, 1.0, 2.0])
        from parsec_python.V_xc.pbe import _KAPPA, _MU

        pbe = 1.0 + _KAPPA - _KAPPA / (1.0 + _MU * s * s / _KAPPA)
        hole = wpbe_enhancement(s, np.full(s.size, 1e-6))[0]
        np.testing.assert_allclose(hole, pbe, rtol=6e-3)

    def test_wpbe_derivatives(self) -> None:
        # Forward-mode derivatives against central differences (well-conditioned nu).
        from parsec_python.V_xc.wpbe import wpbe_enhancement

        rng = np.random.default_rng(1)
        s = rng.uniform(0.1, 3.0, 50)
        nu = rng.uniform(0.02, 0.8, 50)
        value, d_s, d_nu, d_s_over_s = wpbe_enhancement(s, nu)
        h = 1e-4
        np.testing.assert_allclose(d_s, (wpbe_enhancement(s + h, nu)[0] - wpbe_enhancement(s - h, nu)[0]) / (2 * h),
                                   rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(d_nu, (wpbe_enhancement(s, nu + h)[0] - wpbe_enhancement(s, nu - h)[0]) / (2 * h),
                                   rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(d_s_over_s, d_s / s, rtol=1e-15)
        tiny = wpbe_enhancement(np.array([0.0, 1e-9, 1e-7]), np.full(3, 0.3))[3]
        self.assertAlmostEqual(tiny[0] / tiny[2], 1.0, delta=1e-6)

    def test_lyp_near_full_polarization(self) -> None:
        # The s_ab form does not cancel for an almost empty channel (libxc loses ~1e-9 there).
        from parsec_python.V_xc.b3lyp import lyp_correlation_energy_density_spin
        from parsec_python.V_xc.becke_lyp import lyp_spin_correlation_partials

        a, b = np.array([0.18313]), np.array([2.635e-4])
        s_aa, s_bb = np.array([0.2]), np.array([1.1e-6])
        s_ab = 0.3 * np.sqrt(s_aa * s_bb)
        values = lyp_spin_correlation_partials(a + b, a, b, s_aa, s_ab, s_bb)
        reference = lyp_correlation_energy_density_spin(a, b, s_aa, s_bb, s_aa + 2 * s_ab + s_bb)
        self.assertAlmostEqual(values[0][0] / reference[0], 1.0, delta=1e-10)

    def test_component_validation(self) -> None:
        with self.assertRaises(ValueError):
            SemilocalComponent("nonsense_x", 1.0)
        with self.assertRaises(ValueError):
            SemilocalComponent("pbe_x", 1.0, 0.1)
        with self.assertRaises(ValueError):
            SemilocalComponent("wpbe_sr_x", 1.0)
        self.assertEqual(ComponentSemilocal(()).family, "none")
        self.assertEqual(ComponentSemilocal((SemilocalComponent("pbe_x", 1.0),)).family, "gga")


if __name__ == "__main__":
    unittest.main()
