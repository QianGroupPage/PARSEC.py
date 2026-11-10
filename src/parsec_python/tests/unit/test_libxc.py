"""libxc as an optional dependency (``V_xc/libxc_library.py``, ``V_xc/libxc.py``).

Without libxc (``PARSEC_LIBXC_LIBRARY`` or a ``libxc.so``/``xc.dll`` on the
loader's path) only the spec syntax is tested.  With it:

* the host evaluator against the pylibxc 7.1.2 values of
  ``tests/hybrid_libxc_reference.py`` (round-off for libxc 7.1);
* the registry: families, tau, CAM coefficients, refusals;
* libxc's versions of this code's functionals against the code's own:
  round-off where the definitions match, round-off with libxc's digits
  patched into the code's PBE/PW92/PW91 (the patch of
  ``test_hybrid_functionals.py``), pointwise and as grid fields
  (unpolarized and spin, core density);
* libxc hybrids as ``HybridFunctional`` (CAM coefficients, the semilocal
  part, an exchange-only hybrid component beside a correlation).
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python import GridSettings
from parsec_python.Grid import build_cluster_grid
from parsec_python.V_xc.libxc import (
    LIBXC_PREFIX,
    is_libxc_functional,
    libxc_pseudopotential_functional,
    parse_libxc_spec,
)
from parsec_python.V_xc.libxc_library import libxc_available
from parsec_python.tests.hybrid_libxc_reference import POLARIZED_INPUTS, REFERENCE, UNPOLARIZED_INPUTS

HAVE_LIBXC = libxc_available()


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


def _cluster_grid():
    return build_cluster_grid(GridSettings(spacing=0.6, radius=3.4, expansion_order=8, shift=(0.5, 0.5, 0.5)))


def _density(grid, centers=((0.4, -0.3, 0.2), (-0.9, 0.7, -0.4)), weights=(0.9, 0.5), width=1.3):
    coordinates = grid.coordinates
    values = np.zeros(grid.size)
    for center, weight in zip(centers, weights):
        difference = coordinates - np.asarray(center)
        values += weight * np.exp(-np.sum(difference * difference, axis=1) / width)
    return values


def _relative_field(reference, value) -> float:
    reference, value = np.asarray(reference), np.asarray(value)
    return float(np.max(np.abs(reference - value)) / max(np.max(np.abs(reference)), 1e-300))


class SpecTests(unittest.TestCase):
    """The settings spelling, without the library."""

    def test_parse(self) -> None:
        spec = parse_libxc_spec("libxc:gga_x_pbe+gga_c_pbe")
        self.assertEqual(spec.names, ("gga_x_pbe", "gga_c_pbe"))
        self.assertEqual(spec.family, "gga")
        self.assertFalse(spec.is_hybrid)
        self.assertIsNone(spec.enforce_fhc)
        self.assertEqual(spec.text, "libxc:gga_x_pbe+gga_c_pbe")
        spec = parse_libxc_spec("libxc:mgga_x_scan+mgga_c_scan;fhc=off")
        self.assertEqual(spec.family, "mgga")
        self.assertIs(spec.enforce_fhc, False)
        self.assertEqual(spec.text, "libxc:mgga_x_scan+mgga_c_scan;fhc=off")
        spec = parse_libxc_spec("libxc:hyb_gga_x_s12h+gga_c_pbe")
        self.assertTrue(spec.is_hybrid)
        self.assertEqual(spec.hybrid_names, ("hyb_gga_x_s12h",))
        self.assertEqual(parse_libxc_spec("libxc:lda_x+gga_c_pbe").family, "gga")

    def test_refused_syntax(self) -> None:
        for text in (
            "gga_x_pbe",
            "libxc:",
            "libxc:GGA_X_PBE",
            "libxc:gga_x_pbe+gga_x_pbe",
            "libxc:gga_x_pbe;fhc=maybe",
            "libxc:k_tf",
            "libxc:gga_x_pbe gga_c_pbe",
        ):
            with self.subTest(text), self.assertRaises(ValueError):
                parse_libxc_spec(text)

    def test_helpers(self) -> None:
        self.assertTrue(is_libxc_functional(LIBXC_PREFIX + "lda_x"))
        self.assertFalse(is_libxc_functional("pbe"))
        self.assertFalse(is_libxc_functional(None))
        self.assertEqual(libxc_pseudopotential_functional("libxc:lda_x+lda_c_pw"), "ca")
        self.assertEqual(libxc_pseudopotential_functional("libxc:gga_x_pbe+gga_c_pbe"), "pbe")
        self.assertEqual(libxc_pseudopotential_functional("libxc:hyb_lda_xc_lda0"), "pbe")

    def test_settings_keep_their_text(self) -> None:
        from parsec_python.models import SCFSettings

        settings = SCFSettings(xc_functional="libxc:gga_x_pbe+gga_c_pbe")
        self.assertIn("xc_functional='libxc:gga_x_pbe+gga_c_pbe'", repr(settings))
        self.assertFalse(settings.is_hybrid)
        self.assertTrue(settings.is_libxc)
        self.assertFalse(SCFSettings(xc_functional="pbe").is_libxc)
        hybrid = SCFSettings(xc_functional="libxc:hyb_gga_xc_b3lyp")
        self.assertTrue(hybrid.is_hybrid)
        self.assertIsNotNone(hybrid.exact_exchange)
        with self.assertRaises(ValueError):
            SCFSettings(xc_functional="libxc:GGA_X_PBE")


@unittest.skipUnless(HAVE_LIBXC, "libxc is not available (PARSEC_LIBXC_LIBRARY)")
class LibraryTests(unittest.TestCase):
    def test_reference_points(self) -> None:
        from parsec_python.V_xc.libxc import LibxcSemilocal
        from parsec_python.V_xc.libxc_library import load_libxc

        if load_libxc().version[:2] != (7, 1):
            self.skipTest("the reference values are those of libxc 7.1.2")
        for case, entry in REFERENCE.items():
            name, version = entry["libxc"].split()
            if version != "7.1.2" or name == "gga_x_ityh":  # ityh: omega is an external parameter
                continue
            with self.subTest(case):
                evaluator = LibxcSemilocal(LIBXC_PREFIX + name)
                unpolarized = evaluator.unpolarized(*map(np.asarray, UNPOLARIZED_INPUTS))
                polarized = evaluator.polarized(*map(np.asarray, POLARIZED_INPUTS))
                self.assertLess(_worst(unpolarized, entry["unpolarized"]), 1e-12)
                self.assertLess(_worst(polarized, entry["polarized"]), 1e-12)

    def test_registry(self) -> None:
        from parsec_python.V_xc.libxc import canonical_libxc_functional, libxc_functional_info

        pbe = libxc_functional_info("GGA_X_PBE")
        self.assertEqual((pbe.number, pbe.name, pbe.family, pbe.kind), (101, "gga_x_pbe", "gga", "exchange"))
        self.assertIs(libxc_functional_info(101), pbe)
        self.assertIs(libxc_functional_info("xc_gga_x_pbe"), pbe)
        scan = libxc_functional_info("mgga_x_scan")
        self.assertTrue(scan.needs_tau)
        self.assertEqual(scan.family, "mgga")
        b3lyp = libxc_functional_info("hyb_gga_xc_b3lyp")
        self.assertTrue(b3lyp.hybrid)
        self.assertEqual((b3lyp.alpha, b3lyp.beta), (0.2, 0.0))
        hse = libxc_functional_info("hyb_gga_xc_hse06")
        self.assertEqual((hse.alpha, hse.beta, hse.omega), (0.0, 0.25, 0.11))
        cam = libxc_functional_info("hyb_gga_xc_cam_b3lyp")
        self.assertEqual((cam.alpha, cam.beta, cam.omega), (0.65, -0.46, 0.33))
        for name, reason in (
            ("hyb_mgga_xc_wb97m_v", "VV10"),
            ("hyb_gga_xc_camy_b3lyp", "Yukawa"),
            ("gga_x_lb", "no energy"),
            ("lda_k_tf", "kinetic"),
            ("hyb_gga_xc_wb97x_d", "dispersion"),
            ("hyb_gga_xc_wb97x_d3", "dispersion"),
            ("gga_xc_b97_d", "dispersion"),
            ("mgga_xc_otpss_d", "dispersion"),
        ):
            with self.subTest(name):
                self.assertIn(reason, libxc_functional_info(name).refusal())
                with self.assertRaisesRegex(ValueError, reason):
                    canonical_libxc_functional([name])
        self.assertEqual(canonical_libxc_functional(["GGA_X_PBE", 130]), "libxc:gga_x_pbe+gga_c_pbe")
        # "dispersionless" B97 holds no -D term.
        self.assertEqual(canonical_libxc_functional(["gga_xc_dlb97"]), "libxc:gga_xc_dlb97")
        self.assertEqual(
            canonical_libxc_functional(["mgga_x_scan", "mgga_c_scan"], enforce_fhc=False),
            "libxc:mgga_x_scan+mgga_c_scan;fhc=off",
        )
        with self.assertRaisesRegex(ValueError, "two libxc hybrids"):
            canonical_libxc_functional(["hyb_gga_xc_b3lyp", "hyb_gga_xc_pbeh"])
        with self.assertRaisesRegex(ValueError, "given twice"):
            canonical_libxc_functional(["gga_x_pbe", 101])
        with self.assertRaisesRegex(ValueError, "no functional"):
            canonical_libxc_functional(["gga_x_nonsense"])

    def test_own_functionals_pointwise(self) -> None:
        """libxc's versions of the code's components: round-off, with libxc's digits where they differ."""

        from parsec_python.V_xc.hybrid import HYBRID_FUNCTIONALS
        from parsec_python.V_xc.libxc import LibxcSemilocal
        from parsec_python.V_xc.semilocal_components import ComponentSemilocal, SemilocalComponent
        from parsec_python.tests.unit.test_hybrid_functionals import _libxc_digits

        def component(name):
            return ComponentSemilocal((SemilocalComponent(name, 1.0),))

        same = {
            "libxc:lda_x": component("slater_x"),
            "libxc:gga_x_pbe": component("pbe_x"),
            "libxc:gga_x_b88": component("b88_x"),
            "libxc:gga_c_lyp": component("lyp_c"),
            "libxc:lda_c_vwn_rpa": component("vwn_rpa_c"),
            "libxc:hyb_gga_xc_b3lyp": HYBRID_FUNCTIONALS["b3lyp"].semilocal,
            "libxc:hyb_gga_xc_cam_b3lyp": HYBRID_FUNCTIONALS["cam-b3lyp"].semilocal,
        }
        # (Exchange and correlation separately: their vsigma cancel to 1e-13 at sigma = 0.)
        digits = {
            "libxc:gga_c_pbe": component("pbe_c"),
            "libxc:hyb_gga_xc_pbeh": HYBRID_FUNCTIONALS["pbe0"].semilocal,
            "libxc:lda_c_pw": component("pw92_c"),
            "libxc:gga_x_pw91": component("pw91_x"),
            "libxc:gga_c_pw91": component("pw91_c"),
        }
        unpolarized = tuple(map(np.asarray, UNPOLARIZED_INPUTS))
        polarized = tuple(map(np.asarray, POLARIZED_INPUTS))
        for cases, patched in ((same, False), (digits, True)):
            for spec, own in cases.items():
                with self.subTest(spec):
                    libxc = LibxcSemilocal(spec)
                    if patched:
                        # Without the patch: the stated digit differences (PW92 A of PBE: 3e-7 of the
                        # local correlation; Fortran PARSEC's PW92/PW91 digits).
                        self.assertLess(_worst(own.unpolarized(*unpolarized)[:1], libxc.unpolarized(*unpolarized)[:1]), 1e-3)
                        with _libxc_digits():
                            self.assertLess(_worst(own.unpolarized(*unpolarized), libxc.unpolarized(*unpolarized)), 2e-12)
                            self.assertLess(_worst(own.polarized(*polarized), libxc.polarized(*polarized)), 2e-12)
                    else:
                        self.assertLess(_worst(own.unpolarized(*unpolarized), libxc.unpolarized(*unpolarized)), 2e-12)
                        self.assertLess(_worst(own.polarized(*polarized), libxc.polarized(*polarized)), 2e-12)

    def test_grid_fields_equal_the_code_pbe(self) -> None:
        """semilocal_xc with libxc's PBE = pbe()/pbe_spin() with libxc's digits, as grid fields."""

        from parsec_python.V_xc.libxc import libxc_semilocal
        from parsec_python.V_xc.pbe import pbe
        from parsec_python.V_xc.pbe_spin import pbe_spin
        from parsec_python.V_xc.semilocal import semilocal_xc, semilocal_xc_spin
        from parsec_python.tests.unit.test_hybrid_functionals import _libxc_digits

        grid = _cluster_grid()
        density = _density(grid)
        core = _density(grid, centers=((0.4, -0.3, 0.2),), weights=(0.05,), width=0.3)
        up = _density(grid, centers=((0.4, -0.3, 0.2),), weights=(0.6,))
        down = _density(grid, centers=((-0.9, 0.7, -0.4),), weights=(0.4,), width=1.6)
        evaluator = libxc_semilocal("libxc:gga_x_pbe+gga_c_pbe")
        libxc = semilocal_xc(evaluator, density, grid, core)
        libxc_spin = semilocal_xc_spin(evaluator, up, down, grid, core)
        plain = pbe(density, grid, core)
        # The digits of the PW92 A: 1e-7 of the correlation of this density.
        self.assertLess(abs(plain.total_energy - libxc.total_energy), 1e-6 * abs(plain.total_energy))
        with _libxc_digits():
            own = pbe(density, grid, core)
            own_spin = pbe_spin(up, down, grid, core)
        self.assertLess(_relative_field(own.potential, libxc.potential), 1e-12)
        self.assertLess(_relative_field(own.energy_density, libxc.energy_density), 1e-12)
        self.assertAlmostEqual(own.total_energy, libxc.total_energy, delta=1e-13 * abs(own.total_energy))
        for name in ("potential_up", "potential_down", "energy_density"):
            self.assertLess(_relative_field(getattr(own_spin, name), getattr(libxc_spin, name)), 1e-12, name)
        self.assertAlmostEqual(own_spin.total_energy, libxc_spin.total_energy, delta=1e-13 * abs(own_spin.total_energy))

    def test_meta_gga_points(self) -> None:
        """A meta-GGA: fixed arity with None placeholders, tau consistent between the spin forms."""

        from parsec_python.V_xc.libxc import LibxcSemilocal

        scan = LibxcSemilocal("libxc:mgga_x_scan+mgga_c_scan")
        self.assertEqual((scan.family, scan.needs_tau, scan.needs_laplacian), ("mgga", True, False))
        rho = np.array([0.01, 0.1, 1.0, 5.0])
        sigma = np.array([1e-4, 0.02, 0.5, 30.0])
        tau = np.array([0.003, 0.05, 1.2, 20.0])
        e, vrho, vsigma, vtau, vlapl = scan.unpolarized(rho, sigma, tau, None)
        self.assertIsNone(vlapl)
        self.assertTrue(np.all(np.isfinite(vtau)))
        # Equal channels: the polarized form at (rho/2, rho/2, sigma/4 each, tau/2 each) is the same energy.
        spin = scan.polarized(rho / 2, rho / 2, sigma / 4, sigma / 4, sigma / 4, tau / 2, tau / 2, None, None)
        self.assertEqual(len(spin), 10)
        np.testing.assert_allclose(spin[0], e, rtol=1e-12)
        np.testing.assert_allclose(spin[1], vrho, rtol=1e-12)
        np.testing.assert_allclose(spin[6], vtau, rtol=1e-12)
        with self.assertRaises(ValueError):
            scan.unpolarized(rho, sigma, None, None)
        with self.assertRaises(ValueError):
            LibxcSemilocal("libxc:gga_x_pbe").unpolarized(rho, sigma, tau, None)
        # The provider of V_xc/meta_gga.py: the evaluator of a libxc meta-GGA, None otherwise.
        from types import SimpleNamespace

        from parsec_python.V_xc.libxc import libxc_meta_gga_evaluator

        provided = libxc_meta_gga_evaluator(SimpleNamespace(xc_functional="libxc:mgga_x_scan+mgga_c_scan"))
        self.assertEqual(provided.spec, scan.spec)
        self.assertIsNone(libxc_meta_gga_evaluator(SimpleNamespace(xc_functional="libxc:gga_x_pbe")))
        self.assertIsNone(libxc_meta_gga_evaluator(SimpleNamespace(xc_functional="scan")))
        # The FHC switch: below the clamp (sigma <= 8 rho tau everywhere) nothing changes; fhc=on asks for the floor.
        # Without a suffix the clamp is off (as fhc=off); fhc=on keeps libxc's evaluation unclamped as well
        # (the grid layer floors tau first, tau_floor) and is a distinct spec.
        free = LibxcSemilocal("libxc:mgga_x_scan+mgga_c_scan;fhc=off")
        floored = LibxcSemilocal("libxc:mgga_x_scan+mgga_c_scan;fhc=on")
        np.testing.assert_array_equal(free.unpolarized(rho, sigma, tau, None)[0], e)
        self.assertIsNone(scan.tau_floor)
        self.assertIsNone(free.tau_floor)
        self.assertEqual(floored.tau_floor, "weizsaecker")
        self.assertIsNone(LibxcSemilocal("libxc:gga_x_pbe;fhc=on").tau_floor)
        violated = np.array([2.0])  # sigma > 8 rho tau at rho = 0.1, tau = 0.001
        unclamped = scan.unpolarized(rho[1:2], violated, np.array([0.001]), None)[0]
        np.testing.assert_array_equal(unclamped, free.unpolarized(rho[1:2], violated, np.array([0.001]), None)[0])
        # The floors of tau in the grid layer: the absolute floor tau_minimum (;tau= or 1e-12) for every
        # meta-GGA that needs tau, the tau_W floor for a functional with a kink at tau_W (TPSS-type
        # correlation clamps z = tau_W/tau at 1; SCAN, r2SCAN and TPSS exchange are smooth there).
        from parsec_python.V_xc.meta_gga import tau_floor_options, weizsaecker_chain, weizsaecker_floor

        tpss = LibxcSemilocal("libxc:mgga_x_tpss+mgga_c_tpss")
        self.assertEqual((tpss.tau_minimum, tpss.tau_floor), (1e-12, "weizsaecker"))
        self.assertTrue(tpss.kink_at_weizsaecker)
        self.assertFalse(scan.kink_at_weizsaecker)
        self.assertEqual(scan.tau_minimum, 1e-12)
        self.assertFalse(LibxcSemilocal("libxc:mgga_x_tpss").kink_at_weizsaecker)
        self.assertIsNone(LibxcSemilocal("libxc:gga_x_pbe").tau_minimum)
        self.assertEqual(LibxcSemilocal("libxc:mgga_x_tpss+mgga_c_tpss;tau=1e-10").tau_minimum, 1e-10)
        # A density maximum where every orbital has a zero gradient (Si bond centre at Gamma): tau and
        # sigma are round-off, z = tau_W/tau at libxc's own floor (1e-20) puts vtau at 1e9 (1e13 at sigma 1e-16).
        rho, sigma, tau = np.array([0.125, 0.125]), np.array([1e-30, 1e-20]), np.array([1e-16, 1e-16])
        raw = tpss.unpolarized(rho, sigma, tau, None)
        self.assertGreater(float(np.max(np.abs(raw[3]))), 1e8)
        # Through the floor: the point keeps the energy of tau = tau_minimum (Quantum ESPRESSO drops it),
        # z -> 0 with sigma, vtau = 0, and vsigma |grad n| stays small.
        given, below = weizsaecker_floor(np, rho, sigma, tau, **tau_floor_options(tpss))
        np.testing.assert_array_equal(below, [2, 2])
        e, vrho, vsigma, vtau, _ = tpss.unpolarized(rho, sigma, given, None)
        vrho, vsigma, vtau = weizsaecker_chain(np, rho, sigma, below, vrho, vsigma, vtau)
        limit = tpss.unpolarized(rho[:1], np.zeros(1), np.full(1, 1e-12), None)[0]
        np.testing.assert_allclose(e, limit[0], rtol=1e-8)
        self.assertLess(float(e[0]), -0.05)
        np.testing.assert_array_equal(vtau, 0.0)
        self.assertTrue(np.all(np.isfinite(vrho)))
        self.assertLess(float(np.max(np.abs(vsigma * np.sqrt(sigma)))), 1.0)
        self.assertEqual(
            parse_libxc_spec("libxc:mgga_x_tpss+mgga_c_tpss;tau=1e-20").text, "libxc:mgga_x_tpss+mgga_c_tpss;tau=1e-20"
        )
        # libxc's own clamp would change it (the default of libxc 7, not used here).
        from parsec_python.V_xc.libxc_library import load_libxc

        handle = load_libxc().functional(263, polarized=False, enforce_fhc=True)
        self.assertNotEqual(handle.flags & (1 << 17), 0)

    def test_tau_floors_on_the_grid(self) -> None:
        """libxc TPSS through the grid layer's floors (tau_minimum, tau_W): the potentials are the derivatives."""

        from parsec_python.V_xc.libxc import LibxcSemilocal
        from parsec_python.V_xc.meta_gga import meta_gga_xc, meta_gga_xc_spin
        from parsec_python.tests.unit.test_meta_gga import GridLayerTests, _cluster_grid, _fields, _relative_direction

        grid = _cluster_grid()
        _, _, density, tau = _fields(grid)
        minimum = float(np.quantile(tau, 0.3))
        checker = GridLayerTests()
        for fhc in (";fhc=off", ""):
            functional = LibxcSemilocal(f"libxc:mgga_x_tpss+mgga_c_tpss{fhc};tau={minimum!r}")
            self.assertEqual(functional.tau_floor, None if fhc else "weizsaecker")
            result = meta_gga_xc(functional, density, tau, grid)
            low = (tau < minimum) & (density > 1e-20)
            self.assertGreater(int(np.count_nonzero(low)), 20)
            self.assertTrue(np.all(result.kinetic_potential[low] == 0.0))
            self.assertTrue(np.all(result.energy_density[low] != 0.0))
            direction = _relative_direction(grid, density)
            checker.assert_derivative(
                lambda e: meta_gga_xc(functional, density + e * direction, tau, grid).total_energy,
                grid.volume_element * float(np.dot(result.potential, direction)),
                direction,
            )
            tau_direction = _relative_direction(grid, tau, 2)
            checker.assert_derivative(
                lambda e: meta_gga_xc(functional, density, tau + e * tau_direction, grid).total_energy,
                2.0 * grid.volume_element * float(np.dot(result.kinetic_potential, tau_direction)),
                tau_direction,
            )
            spin = meta_gga_xc_spin(functional, 0.6 * density, 0.4 * density, 0.53 * tau, 0.47 * tau, grid)
            checker.assert_derivative(
                lambda e: meta_gga_xc_spin(
                    functional, 0.6 * density + e * direction, 0.4 * density, 0.53 * tau, 0.47 * tau, grid
                ).total_energy,
                grid.volume_element * float(np.dot(spin.potential_up, direction)),
                direction,
            )

    def test_hybrids(self) -> None:
        from parsec_python.V_xc.hybrid import HYBRID_FUNCTIONALS
        from parsec_python.V_xc.libxc import libxc_hybrid_functional

        b3lyp = libxc_hybrid_functional("libxc:hyb_gga_xc_b3lyp")
        self.assertEqual(b3lyp.cam_coefficients, (0.2, 0.0, 0.0))
        self.assertEqual(b3lyp.exact_exchange_terms, HYBRID_FUNCTIONALS["b3lyp"].exact_exchange_terms)
        hse = libxc_hybrid_functional("libxc:hyb_gga_xc_hse06")
        self.assertEqual(hse.cam_coefficients, (0.0, 0.25, 0.11))
        cam = libxc_hybrid_functional("libxc:hyb_gga_xc_cam_b3lyp")
        self.assertEqual(cam.kernels, ("coulomb", "erf"))
        np.testing.assert_allclose([t.fraction for t in cam.exact_exchange_terms], [0.19, 0.46], rtol=1e-14)
        # An exchange-only hybrid beside a correlation: one fraction of exact exchange (libxc's
        # hyb_gga_x_s12h already returns its reduced semilocal exchange), the correlation added.
        from parsec_python.V_xc.libxc import LibxcSemilocal

        s12h = libxc_hybrid_functional("libxc:hyb_gga_x_s12h+gga_c_pbe")
        self.assertEqual(s12h.cam_coefficients, (0.25, 0.0, 0.0))
        rho, sigma = map(np.asarray, UNPOLARIZED_INPUTS)
        total = s12h.semilocal.unpolarized(rho, sigma)[0]
        parts = LibxcSemilocal("libxc:hyb_gga_x_s12h").unpolarized(rho, sigma)[0] + LibxcSemilocal(
            "libxc:gga_c_pbe"
        ).unpolarized(rho, sigma)[0]
        np.testing.assert_allclose(total, parts, rtol=1e-15)
        # The semilocal part on the grid: libxc's B3LYP = the code's B3LYP to round-off.
        from parsec_python.V_xc.hybrid import hybrid_semilocal_xc

        grid = _cluster_grid()
        density = _density(grid)
        own = hybrid_semilocal_xc(HYBRID_FUNCTIONALS["b3lyp"], density, grid)
        libxc = hybrid_semilocal_xc(b3lyp, density, grid)
        self.assertLess(_relative_field(own.potential, libxc.potential), 1e-12)
        self.assertAlmostEqual(own.total_energy, libxc.total_energy, delta=1e-13 * abs(own.total_energy))


@unittest.skipUnless(HAVE_LIBXC, "libxc is not available (PARSEC_LIBXC_LIBRARY)")
class InputAndSCFTests(unittest.TestCase):
    """The keywords, and SCF runs of H2 / the H atom: libxc's functional = the code's (reference path)."""

    def setUp(self) -> None:
        from parsec_python.tests.unit.test_hybrid_scf import _Inputs

        self.inputs = _Inputs()
        self.inputs.setUp()

    def tearDown(self) -> None:
        self.inputs.tearDown()

    def _translation(self, functional_lines: str, *, states=2, atoms=None, extra="", species=""):
        # The template of test_hybrid_scf.py with the Correlation_Type line replaced.
        from parsec_python.Input import parse_parsec_input
        from parsec_python.tests.unit.test_hybrid_scf import BASE, H2_ATOMS

        text = BASE.format(
            states=states, atoms=atoms or H2_ATOMS, functional="pbe", extra=extra, species=species
        ).replace("Correlation_Type: pbe\n", functional_lines)
        path = self.inputs.directory / "libxc.in"
        path.write_text(text)
        return parse_parsec_input(path)

    def _run(self, translation):
        from parsec_python.SCF import prepare_single_point, run_scf

        system = prepare_single_point(translation.problem)
        return run_scf(system)

    def test_keywords(self) -> None:
        from parsec_python.Input import UnsupportedParsecOptionError
        from parsec_python.Input.parsec_input import ParsecInputError

        translation = self._translation("LibXC_Functional: GGA_X_PBE, gga_c_pbe\n")
        self.assertEqual(translation.problem.scf.xc_functional, "libxc:gga_x_pbe+gga_c_pbe")
        translation = self._translation("Use_LibXC: .true.\nFunc_Id_X: 101\nFunc_Id_C: 130\n")
        self.assertEqual(translation.problem.scf.xc_functional, "libxc:gga_x_pbe+gga_c_pbe")
        self.assertTrue(any("first SCF iteration" in note for note in translation.warnings))
        # PARSEC 1.5's defaults: LDA_X + LDA_C_PZ (LDA_C_PW with spin).
        translation = self._translation("Use_LibXC: .true.\n")
        self.assertEqual(translation.problem.scf.xc_functional, "libxc:lda_x+lda_c_pz")
        translation = self._translation("Use_LibXC: .true.\nFunc_Id_C: 0\n")
        self.assertEqual(translation.problem.scf.xc_functional, "libxc:lda_x")
        # An exchange-correlation functional in Func_Id_X gets 1.5's default correlation, with a warning.
        translation = self._translation("Use_LibXC: .true.\nFunc_Id_X: 402\n")
        self.assertEqual(translation.problem.scf.xc_functional, "libxc:hyb_gga_xc_b3lyp+lda_c_pz")
        self.assertTrue(any("Func_Id_C: 0" in note for note in translation.warnings))
        translation = self._translation("Correlation_Type: pb\nLibXC_Functional: hyb_gga_xc_b3lyp\n")
        self.assertTrue(translation.problem.scf.is_hybrid)
        self.assertTrue(any("Correlation_Type does not select" in note for note in translation.warnings))
        translation = self._translation("LibXC_Functional: 402\nExx_Tolerance: 1e-8 Ry\n")
        self.assertEqual(translation.problem.scf.exact_exchange.tolerance, 1e-8)
        # bak_complete_parsec.in writes LibXC_Functional (and the options) back, not the 1.5 keywords.
        from parsec_python.Input import parse_parsec_input
        from parsec_python.Input.complete_input import COMPLETE_INPUT_NAME

        for lines in (
            "Use_LibXC: .true.\nFunc_Id_X: 101\nFunc_Id_C: 130\nLibXC_Density_Threshold: 1e-20\n",
            "LibXC_Functional: hyb_gga_xc_hse06\nExx_Tolerance: 1e-7 Ry\n",
        ):
            with self.subTest(lines):
                translation = self._translation(lines + "Generate_Complete_Parsec_In: .true.\n")
                path = self.inputs.directory / COMPLETE_INPUT_NAME
                path.write_text(translation.complete_input)
                again = parse_parsec_input(path).problem.scf
                self.assertEqual(again.xc_functional, translation.problem.scf.xc_functional)
                self.assertEqual(again.exact_exchange, translation.problem.scf.exact_exchange)
        self.assertEqual(again.xc_functional, "libxc:hyb_gga_xc_hse06")
        # --mpi with a libxc LDA/GGA is refused by name (not tested with the distributed solvers).
        from parsec_python.acceleration.cli import _mpi_unsupported_features

        features = _mpi_unsupported_features(self._translation("LibXC_Functional: gga_x_pbe gga_c_pbe\n"))
        self.assertTrue(any("libxc" in feature for feature in features))
        # parsec.out of a libxc meta-GGA (and meta-GGA hybrid): tau, the form of the density gradient
        # and the floors.
        from parsec_python.Output.parsec_output import _correlation_lines

        for names, form in (("mgga_x_r2scan mgga_c_r2scan", "orbital"), ("hyb_mgga_xc_tpssh", "density")):
            with self.subTest(names):
                scf = self._translation(f"LibXC_Functional: {names}\n").problem.scf
                lines = "\n".join(_correlation_lines(scf))
                self.assertIn(f"density gradient: {form} form", lines)
                self.assertIn("tau floor 1e-12", lines)
        lines = "\n".join(_correlation_lines(self._translation("LibXC_Functional: hyb_mgga_xc_tpssh\n").problem.scf))
        self.assertIn("tau >= tau_W floor", lines)
        for lines, error, pattern in (
            ("LibXC_Functional: gga_x_pbe\nUse_LibXC: .false.\n", ParsecInputError, "Use_LibXC"),
            ("LibXC_Functional: gga_x_pbe\nFunc_Id_X: 101\n", ParsecInputError, "not both"),
            ("LibXC_Functional: gga_x_nonsense\n", ParsecInputError, "no functional"),
            ("LibXC_Functional: hyb_mgga_xc_wb97m_v\n", UnsupportedParsecOptionError, "VV10"),
            ("LibXC_Functional: lda_k_tf\n", UnsupportedParsecOptionError, "kinetic"),
            ("LibXC_Enforce_FHC: .false.\n", ParsecInputError, "belong to a libxc"),
            ("LibXC_Functional: gga_x_pbe\nLibXC_Density_Threshold: -1\n", ParsecInputError, "positive"),
            (
                "LibXC_Functional: hyb_gga_xc_pbeh\nExact_Exchange_Fraction: 0.3\n",
                ParsecInputError,
                "libxc's coefficients",
            ),
            ("LibXC_Functional: gga_x_pbe\nExx_Tolerance: 1e-8 Ry\n", ParsecInputError, "hybrid"),
            (
                "LibXC_Functional: gga_x_pbe gga_c_pbe\nNon_Collinear_Magnetism: .true.\n",
                UnsupportedParsecOptionError,
                "collinear",
            ),
            ("LibXC_Functional: hyb_gga_xc_wb97x_d\n", UnsupportedParsecOptionError, "dispersion"),
            # usrinputfile.F90:4100: the TS correction needs the PBE of the code, not libxc (error 158).
            (
                "Correlation_Type: pb\nLibXC_Functional: gga_x_pbe gga_c_pbe\nVan_Der_Waals: .true.\n"
                "Ignore_Symmetry: .true.\n",
                ParsecInputError,
                "158",
            ),
            (
                "Correlation_Type: pbe\nLibXC_Functional: mgga_x_scan mgga_c_scan\nVan_Der_Waals: .true.\n"
                "Ignore_Symmetry: .true.\n",
                ParsecInputError,
                "158",
            ),
        ):
            with self.subTest(lines), self.assertRaisesRegex(error, pattern):
                self._translation(lines)

    def test_scf_pbe_equals_the_code_pbe(self) -> None:
        from parsec_python.tests.unit.test_hybrid_functionals import _libxc_digits

        libxc = self._run(self._translation("LibXC_Functional: gga_x_pbe gga_c_pbe\n"))
        plain = self._run(self.inputs.translation("pbe"))
        with _libxc_digits():
            own = self._run(self.inputs.translation("pbe"))
        self.assertTrue(libxc.converged and own.converged)
        self.assertAlmostEqual(libxc.energies.total, own.energies.total, delta=1.0e-9)
        np.testing.assert_allclose(libxc.eigenvalues, own.eigenvalues, atol=1.0e-8)
        # The PW92 A of the code's PBE: 1e-7 Ry of H2.
        self.assertLess(abs(libxc.energies.total - plain.energies.total), 1.0e-6)

    def test_scf_spin_pbe_equals_the_code_pbe(self) -> None:
        from parsec_python.tests.unit.test_hybrid_functionals import _libxc_digits

        keywords = dict(
            states=2,
            atoms="  0.0 0.0 0.0\n",
            species="Initial_Spin_Polarization: 1.0\n",
            extra="Spin_Polarization: .true.\n",
        )
        from importlib import import_module
        from unittest import mock

        # libxc floors each spin density at its density threshold: the empty channel of the H atom is
        # not empty (2.6e-9 Ry); with the threshold lowered the two agree to round-off.
        default = self._run(self._translation("LibXC_Functional: gga_x_pbe gga_c_pbe\n", **keywords))
        libxc = self._run(
            self._translation(
                "LibXC_Functional: gga_x_pbe gga_c_pbe\nLibXC_Density_Threshold: 1e-30\n", **keywords
            )
        )
        # libxc's digits, but the code's regularization of d phi/d zeta at |zeta| = 1 (the H atom is
        # fully polarized; libxc clamps zeta instead): it changes the potential of the empty channel only.
        with _libxc_digits(), mock.patch.object(import_module("parsec_python.V_xc.pbe_spin"), "_SPIN_ETA", 1.0e-12):
            own = self._run(self.inputs.translation("pbe", **keywords))
        self.assertTrue(libxc.converged and own.converged and default.converged)
        self.assertAlmostEqual(libxc.energies.total, own.energies.total, delta=1.0e-9)
        self.assertAlmostEqual(default.energies.total, own.energies.total, delta=5.0e-9)

    def test_periodic_cell(self) -> None:
        """A periodic cell (the wrapped stencil): libxc's PBE = the code's PBE with libxc's digits."""

        import warnings
        from dataclasses import replace

        from parsec_python.SCF import prepare_periodic_single_point, run_scf
        from parsec_python.tests.unit.test_hybrid_functionals import _libxc_digits
        from parsec_python.tests.unit.test_periodic_scf import _hydrogen_box

        base = _hydrogen_box(6.0, 0.6)

        def run(functional):
            problem = replace(base, scf=replace(base.scf, xc_functional=functional, convergence_criterion=1e-8))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the CA file of the test data under a GGA
                return run_scf(prepare_periodic_single_point(problem))

        libxc = run("libxc:gga_x_pbe+gga_c_pbe")
        with _libxc_digits():
            own = run("pbe")
        self.assertTrue(libxc.converged and own.converged)
        self.assertAlmostEqual(libxc.energies.total, own.energies.total, delta=1.0e-9)

    def test_periodic_hybrid(self) -> None:
        """libxc's PBE0 (hyb_gga_xc_pbeh) of a periodic cell = the code's PBE0, at the Gamma point (real orbitals)
        and on a 2x1x1 mesh (Bloch factors): round-off with libxc's digits in the code's PBE, the PW92 offset of the
        two PBEs (1e-8 Ry of H2 here) without them."""

        import warnings

        from parsec_python.SCF import prepare_periodic_single_point, run_scf
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf
        from parsec_python.tests.unit.test_hybrid_functionals import _libxc_digits
        from parsec_python.tests.unit.test_periodic_hybrid import _h2_problem

        b1 = 2.0 * np.pi / 6.0
        mesh = (np.array([[0.0, 0.0, 0.0], [0.5 * b1, 0.0, 0.0]]), np.array([0.5, 0.5]))

        def run(functional, k_points):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the CA file of the test data under a GGA
                system = prepare_periodic_single_point(_h2_problem(functional))
                if k_points is None:
                    return run_scf(system)
                return run_periodic_hybrid_scf(system, *k_points)

        for label, k_points in (("gamma", None), ("2x1x1", mesh)):
            with self.subTest(label):
                libxc = run("libxc:hyb_gga_xc_pbeh", k_points)
                plain = run("pbe0", k_points)
                with _libxc_digits():
                    own = run("pbe0", k_points)
                self.assertTrue(libxc.converged and plain.converged and own.converged)
                self.assertIsNotNone(libxc.exact_exchange)
                # measured: 3e-15 (Gamma), 9e-16 (2x1x1); E_x 1e-12
                self.assertAlmostEqual(libxc.energies.total, own.energies.total, delta=1.0e-9)
                self.assertAlmostEqual(
                    libxc.exact_exchange.exchange_energy, own.exact_exchange.exchange_energy, delta=1.0e-9
                )
                # measured: -1.1e-8
                self.assertLess(abs(libxc.energies.total - plain.energies.total), 1.0e-6)

    def test_mpi_routes_are_refused(self) -> None:
        """No libxc functional on the MPI SCF of several ranks (the runner of scripts/run_multi_gpu_scf.sh): the
        accelerated preparation refuses it by name before it prepares anything, a hybrid as an LDA/GGA."""

        from parsec_python.acceleration.driver import prepare_single_point

        for lines in ("LibXC_Functional: hyb_gga_xc_pbeh\n", "LibXC_Functional: gga_x_pbe gga_c_pbe\n"):
            problem = self._translation(lines).problem
            with self.subTest(lines), self.assertRaisesRegex(NotImplementedError, "libxc functional .* MPI SCF"):
                prepare_single_point(problem, backend="cupy", mpi_context=object())

    def test_scf_libxc_b3lyp_equals_the_code_b3lyp(self) -> None:
        extra = "Exx_Tolerance: 1.0e-9 Ry\n"
        libxc = self._run(self._translation("LibXC_Functional: hyb_gga_xc_b3lyp\n", extra=extra))
        own = self._run(self.inputs.translation("b3lyp", extra=extra))
        self.assertTrue(libxc.converged and own.converged)
        self.assertAlmostEqual(libxc.energies.total, own.energies.total, delta=1.0e-9)
        self.assertAlmostEqual(
            libxc.exact_exchange.exchange_energy, own.exact_exchange.exchange_energy, delta=1.0e-9
        )


if __name__ == "__main__":
    unittest.main()
