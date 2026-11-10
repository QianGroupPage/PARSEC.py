"""Meta-GGA hybrids (libxc's ``hyb_mgga_``): the Fock loop with meta-GGA inner loops (reference path).

The inner loops of :func:`parsec_python.SCF.hybrid.run_hybrid_scf` are the
meta-GGA loops of :mod:`parsec_python.SCF.meta_gga` with the exchange
operator in every Hamiltonian.  Checked on H2 (R = 5 bohr, h = 0.5 bohr):

* the converged total energy is the functional evaluated directly from its
  orbitals: kinetic, nonlocal, ionic, Hartree, the semilocal part at the
  density and ``tau`` of the orbitals, ``a_x E_x`` and the ion-ion term; TPSSh
  with the stencil gradient of the density, r2SCAN0 with the orbital form of
  the gradient (``PARSEC_META_GGA_GRADIENT=orbital``).  TPSSh does not
  converge in the orbital form on H2: with one occupied orbital the orbital
  form gives ``tau_W = tau`` (``z = 1``) at every point up to round-off, where
  libxc's TPSS correlation, which takes ``min(z, 1)``, has a jump of its
  derivatives (``vtau`` 0 above, 1e-4 below at a typical point), and the
  residual stays at 1e-2 while the energy settles;
* two equal spin channels give the unpolarized loop;
* an exchange-only hybrid beside a correlation (SCAN0 = ``hyb_mgga_x_scan0``
  + ``mgga_c_scan``) takes the exact exchange of that component once.

Needs libxc (``PARSEC_LIBXC_LIBRARY``).
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.V_xc.libxc_library import libxc_available

HAVE_LIBXC = libxc_available()
TIGHT = "Exx_Tolerance: 1.0e-10 Ry\n"


@unittest.skipUnless(HAVE_LIBXC, "libxc is not available (PARSEC_LIBXC_LIBRARY)")
class MetaGGAHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.SCF import prepare_single_point, run_scf
        from parsec_python.tests.unit.test_hybrid_scf import _Inputs

        cls.inputs = _Inputs()
        cls.inputs.setUp()
        cls.system = prepare_single_point(cls.translation("hyb_mgga_xc_tpssh", extra=TIGHT).problem)
        cls.records = []
        cls.result = run_scf(cls.system, callback=cls.records.append)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.inputs.tearDown()

    @classmethod
    def translation(cls, names, *, states=2, atoms=None, extra="", species=""):
        from parsec_python.Input import parse_parsec_input
        from parsec_python.tests.unit.test_hybrid_scf import BASE, H2_ATOMS

        text = BASE.format(
            states=states, atoms=atoms or H2_ATOMS, functional="pbe", extra=extra, species=species
        ).replace("Convergence_Criterion: 1.0e-7 Ry", "Convergence_Criterion: 1.0e-8 Ry")
        text = text.replace("Correlation_Type: pbe\n", f"LibXC_Functional: {names}\n")
        path = cls.inputs.directory / "meta_hybrid.in"
        path.write_text(text)
        return parse_parsec_input(path)

    def test_definition(self) -> None:
        from parsec_python.SCF.hybrid import hybrid_definition

        scf = self.system.input.scf
        self.assertTrue(scf.is_hybrid and scf.is_meta_gga)
        definition = hybrid_definition(scf)
        self.assertEqual(definition.terms, (("coulomb", 0.1, None),))
        self.assertEqual(definition.start_functional, "pbe")

    def test_total_energy_is_the_functional_of_the_orbitals(self) -> None:
        self._check_direct_energy(self.system, self.result)

    def test_orbital_form_of_the_gradient(self) -> None:
        # r2SCAN0 is smooth across tau = tau_W: the default (auto) takes the orbital form.
        from unittest import mock

        from parsec_python.SCF import prepare_single_point, run_scf

        with mock.patch.dict("os.environ", {}, clear=False):
            __import__("os").environ.pop("PARSEC_META_GGA_GRADIENT", None)
            system = prepare_single_point(self.translation("hyb_mgga_xc_r2scan0", extra=TIGHT).problem)
            result = run_scf(system)
            self.assertEqual(result.meta_gga.gradient, "orbital")
            self._check_direct_energy(system, result)

    def test_start_energy_takes_the_form_of_the_run(self) -> None:
        # The semilocal energy of the start orbitals (outer iteration 0) in the form of the inner
        # loops: the orbital form for r2SCAN0 and the density form for TPSSh under the default.
        from types import SimpleNamespace
        from unittest import mock

        from parsec_python.SCF import prepare_single_point
        from parsec_python.SCF.hybrid import meta_gga_start_semilocal
        from parsec_python.V_xc.meta_gga import grid_gradient, orbital_density_gradient

        start = SimpleNamespace(
            wavefunctions=self.result.wavefunctions,
            occupations=self.result.occupations,
            density=self.result.density,
            spin=None,
        )
        orbitals, occupations = self.result.wavefunctions, np.asarray(self.result.occupations)
        with mock.patch.dict("os.environ", {}, clear=False):
            __import__("os").environ.pop("PARSEC_META_GGA_GRADIENT", None)
            for names, form in (("hyb_mgga_xc_r2scan0", "orbital"), ("hyb_mgga_xc_tpssh", "density")):
                system = prepare_single_point(self.translation(names, extra=TIGHT).problem)
                tau = system.kinetic_energy_density(orbitals, occupations, 2.0)
                gradient = orbital_density_gradient(
                    orbitals, occupations, grid_gradient(system.grid), system.grid.volume_element
                )
                orbital = system.evaluate_xc(self.result.density, tau=tau, density_gradient=gradient).total_energy
                density = system.evaluate_xc(self.result.density, tau=tau).total_energy
                self.assertGreater(abs(orbital - density), 1e-6, names)
                expected = orbital if form == "orbital" else density
                self.assertAlmostEqual(meta_gga_start_semilocal(system, start), expected, delta=1e-12, msg=names)

    def _check_direct_energy(self, system, result) -> None:
        from parsec_python.SCF.hybrid import default_pair_solver, hybrid_definition
        from parsec_python.V_xc.exchange_operator import apply_exchange_to_orbitals
        from parsec_python.V_xc.meta_gga import grid_gradient, orbital_density_gradient

        self.assertTrue(result.converged)
        volume = system.grid.volume_element
        orbitals, occupations = result.wavefunctions, result.occupations
        one_body = 2.0 * sum(
            occupations[i]
            * float(
                orbitals[:, i] @ (system.negative_laplacian @ orbitals[:, i])
                + orbitals[:, i] @ system.nonlocal_operator.apply(orbitals[:, i])
            )
            for i in range(orbitals.shape[1])
        )
        density = result.density
        hartree = system.solve_hartree(density).potential
        tau = system.kinetic_energy_density(orbitals, occupations, 2.0)
        gradient = None
        if result.meta_gga.gradient == "orbital":
            gradient = orbital_density_gradient(orbitals, occupations, grid_gradient(system.grid), volume)
        semilocal = system.evaluate_xc(density, tau=tau, density_gradient=gradient).total_energy
        definition = hybrid_definition(system.input.scf)
        ((_, fraction, _),) = definition.terms
        solver = default_pair_solver(system, definition)
        exchange = apply_exchange_to_orbitals(
            orbitals, occupations, volume, solver, columns=np.flatnonzero(occupations > 1.0e-12)
        ).energy
        direct = (
            one_body
            + volume * float(density @ system.ionic_potential)
            + 0.5 * volume * float(density @ hartree)
            + semilocal
            + fraction * exchange
            + system.ion_ion_energy
        )
        self.assertAlmostEqual(result.energies.total, direct, delta=1.0e-7)
        self.assertAlmostEqual(result.energies.semilocal_exchange_correlation, semilocal, delta=1.0e-7)
        self.assertAlmostEqual(result.exact_exchange.exchange_energy, exchange, delta=1.0e-8)
        self.assertIsNotNone(result.meta_gga)

    def test_outer_iterations(self) -> None:
        from parsec_python.models import ExchangeLoopIteration

        outer = [record for record in self.records if isinstance(record, ExchangeLoopIteration)]
        self.assertEqual(outer[0].start_functional, "pbe")
        self.assertIsNotNone(outer[0].total_energy)
        self.assertTrue(outer[-1].converged)

    def test_two_equal_spin_channels_give_the_unpolarized_loop(self) -> None:
        from parsec_python.SCF import prepare_single_point, run_scf

        translation = self.translation(
            "hyb_mgga_xc_tpssh",
            species="Initial_Spin_Polarization: 0.0\n",
            extra="Spin_Polarization: .true.\n" + TIGHT,
        )
        spin = run_scf(prepare_single_point(translation.problem))
        self.assertTrue(spin.converged)
        self.assertAlmostEqual(spin.energies.total, self.result.energies.total, delta=2.0e-7)
        self.assertAlmostEqual(
            spin.exact_exchange.exchange_energy, self.result.exact_exchange.exchange_energy, delta=1.0e-6
        )

    def test_exchange_only_hybrid_beside_a_correlation(self) -> None:
        """SCAN0: libxc's hyb_mgga_x_scan0 holds 0.75 E_x^SCAN; the exact exchange 0.25 is added once."""

        from parsec_python.SCF.hybrid import hybrid_definition
        from parsec_python.V_xc.libxc import LibxcSemilocal, libxc_hybrid_functional

        scan0 = libxc_hybrid_functional("libxc:hyb_mgga_x_scan0+mgga_c_scan")
        self.assertEqual(scan0.cam_coefficients, (0.25, 0.0, 0.0))
        self.assertEqual(scan0.semilocal.family, "mgga")
        rho = np.array([0.01, 0.1, 1.0])
        sigma = np.array([1e-4, 0.02, 0.5])
        tau = sigma / (8.0 * rho) + 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
        x0 = LibxcSemilocal("libxc:hyb_mgga_x_scan0").unpolarized(rho, sigma, tau, None)[0]
        x = LibxcSemilocal("libxc:mgga_x_scan").unpolarized(rho, sigma, tau, None)[0]
        np.testing.assert_allclose(x0, 0.75 * x, rtol=1e-12)
        total = scan0.semilocal.unpolarized(rho, sigma, tau, None)[0]
        c = LibxcSemilocal("libxc:mgga_c_scan").unpolarized(rho, sigma, tau, None)[0]
        np.testing.assert_allclose(total, x0 + c, rtol=1e-14)
        translation = self.translation("hyb_mgga_x_scan0 mgga_c_scan")
        self.assertEqual(hybrid_definition(translation.problem.scf).terms, (("coulomb", 0.25, None),))


if __name__ == "__main__":
    unittest.main()
