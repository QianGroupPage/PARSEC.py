"""Meta-GGAs: the native SCAN, the discrete energy and its exact derivatives, the orbital term.

Independent references:

* libxc 7.1.2 values of SCAN exchange and correlation (``tests/meta_gga_libxc_reference.py``);
* the potentials are the derivatives of the discrete energy (central differences
  along a direction, one Richardson step), for SCAN and for a test functional
  with a Laplacian dependence;
* the orbital term ``sum_c G_c^T diag(vtau) G_c`` is the derivative of the
  energy with respect to an orbital (at fixed density) and is symmetric;
  with a wave vector it is the derivative for a Bloch factor and Hermitian;
* the sparse gradient operators are the density gradient of the GGAs;
* two equal spin channels give the unpolarized functional.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.V_xc.meta_gga import (
    KineticPotentialOperator,
    grid_gradient,
    kinetic_energy_density,
    meta_gga_xc,
    meta_gga_xc_spin,
)
from parsec_python.V_xc.pbe import density_gradient
from parsec_python.V_xc.scan import NativeScan
from parsec_python.models import GridSettings
from parsec_python.tests import meta_gga_libxc_reference as reference

_STEP = 1.0e-30


class _LaplacianTestFunctional:
    """A smooth test meta-GGA with ``tau`` and Laplacian dependence (not a physical functional).

    ``e = -0.7 n^(4/3) + 0.01 sigma/(n^(4/3) + 0.1) + 0.05 tau n^(1/3)/(1 + n) + 0.002 lapl^2 n^(1/3)/(1 + n)``
    per channel pair; spin: the sum of the unpolarized form of ``2 n_a`` halves.  Derivatives by complex step.
    """

    family = "mgga"
    needs_tau = True
    needs_laplacian = True

    @staticmethod
    def _energy(n, sigma, tau, lapl):
        n13 = n ** (1.0 / 3.0)
        return (
            -0.7 * n * n13
            + 0.01 * sigma / (n * n13 + 0.1)
            + 0.05 * tau * n13 / (1.0 + n)
            + 0.002 * lapl * lapl * n13 / (1.0 + n)
        )

    def _partials(self, function, values):
        energy = function(*values)
        partials = []
        for index in range(len(values)):
            shifted = [np.asarray(value, dtype=np.complex128) for value in values]
            shifted[index] = shifted[index] + 1j * _STEP
            partials.append(function(*shifted).imag / _STEP)
        return energy, partials

    def unpolarized(self, rho, sigma, tau, lapl):
        energy, (vrho, vsigma, vtau, vlapl) = self._partials(self._energy, [rho, sigma, tau, lapl])
        return energy, vrho, vsigma, vtau, vlapl

    def polarized(self, n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d, l_u, l_d):
        def total(n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d, l_u, l_d):
            return 0.5 * (
                self._energy(2.0 * n_u, 4.0 * s_uu, 2.0 * t_u, 2.0 * l_u)
                + self._energy(2.0 * n_d, 4.0 * s_dd, 2.0 * t_d, 2.0 * l_d)
            ) + 0.0 * s_ud

        energy, partials = self._partials(total, [n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d, l_u, l_d])
        return (energy, *partials)


def _cluster_grid():
    return build_cluster_grid(GridSettings(spacing=0.6, radius=3.6, expansion_order=8, shift=(0.5, 0.5, 0.5)))


def _periodic_grid():
    return build_periodic_grid(
        PeriodicCell(lattice_vectors=np.diag([6.0, 6.6, 7.2])),
        PeriodicGridSettings(spacing=0.6, expansion_order=6),
    )


def _orbitals(grid, count=4):
    """Smooth, orthonormal (Euclidean) grid vectors."""

    center = grid.coordinates.mean(axis=0)
    x = grid.coordinates - center
    r2 = np.einsum("ij,ij->i", x, x)
    columns = [
        np.exp(-0.35 * r2),
        x[:, 0] * np.exp(-0.3 * r2),
        (x[:, 1] + 0.3 * x[:, 2]) * np.exp(-0.25 * r2),
        (x[:, 0] * x[:, 2] - 0.2) * np.exp(-0.2 * r2),
    ][:count]
    q, _ = np.linalg.qr(np.column_stack(columns))
    return q


def _relative_direction(grid, field, seed=0):
    """A direction ``field * (smooth function of no symmetry)``: the same relative change at every point.

    The meta-GGA varies fast with ``alpha`` where the density is small; an
    absolute direction would move ``alpha`` of the tail points far within the
    steps of the differences.
    """

    x = grid.coordinates - grid.coordinates.mean(axis=0)
    shape = 0.6 + 0.25 * np.sin(0.9 * x[:, 0] + 0.4 + seed) + 0.1 * np.cos(0.7 * x[:, 1] - 0.3 * seed) + 0.05 * x[:, 2] / (
        1.0 + np.abs(x[:, 2])
    )
    return field * shape


def _fields(grid, occupations=(1.0, 1.0, 0.7, 0.3), degeneracy=2.0):
    q = _orbitals(grid, len(occupations))
    weights = np.asarray(occupations)
    density = degeneracy / grid.volume_element * np.sum(q * q * weights[None, :], axis=1)
    tau = kinetic_energy_density(q, weights, grid_gradient(grid), grid.volume_element, degeneracy=degeneracy)
    return q, weights, density, tau


class ScanPointwiseTests(unittest.TestCase):
    def _compare(self, mine, theirs, *, skip=()):
        for index, (value, expected) in enumerate(zip(mine, theirs)):
            if index in skip:
                continue
            expected = np.asarray(expected)
            scale = np.abs(expected) + 1e-12 * np.max(np.abs(expected), initial=1.0)
            np.testing.assert_array_less(np.abs(np.asarray(value) - expected) / scale, 5.0e-10)

    def test_scan_is_libxc_scan(self) -> None:
        inputs = [np.asarray(values) for values in reference.UNPOLARIZED_INPUTS]
        spin_inputs = [np.asarray(values) for values in reference.POLARIZED_INPUTS]
        for weights, name in (((1.0, 0.0), "mgga_x_scan"), ((0.0, 1.0), "mgga_c_scan")):
            functional = NativeScan(*weights, enforce_fhc=True, derivatives="libxc")
            self._compare(functional.unpolarized(*inputs, None)[:4], reference.REFERENCE[name]["unpolarized"])
            # libxc's exchange has no sigma_ud term (an exact zero).
            self._compare(
                functional.polarized(*spin_inputs)[:8],
                reference.REFERENCE[name]["polarized"],
                skip=(4,) if name == "mgga_x_scan" else (),
            )

    def test_energies_do_not_depend_on_the_derivative_convention(self) -> None:
        inputs = [np.asarray(values) for values in reference.UNPOLARIZED_INPUTS]
        exact = NativeScan(enforce_fhc=True).unpolarized(*inputs, None)
        partial = NativeScan(enforce_fhc=True, derivatives="libxc").unpolarized(*inputs, None)
        np.testing.assert_array_equal(exact[0], partial[0])
        rho, sigma, tau = inputs
        free = sigma < 8.0 * rho * tau
        for index in (1, 2, 3):
            np.testing.assert_allclose(exact[index][free], partial[index][free], rtol=1e-13, atol=0.0)

    def test_without_the_curvature_clamp_alpha_may_be_negative(self) -> None:
        rho = np.array([0.1, 0.1])
        sigma = np.array([0.02, 0.02])
        tau = np.array([0.02 / 0.8 * 0.95, 0.02 / 0.8 * 1.05])  # tau_W = sigma/(8 rho) = 0.025
        off = NativeScan().unpolarized(rho, sigma, tau, None)
        on = NativeScan(enforce_fhc=True).unpolarized(rho, sigma, tau, None)
        self.assertNotEqual(off[0][0], on[0][0])
        self.assertEqual(off[0][1], on[0][1])
        self.assertEqual(on[2][0], 0.0)

    def test_two_equal_channels_are_the_unpolarized_functional(self) -> None:
        rho, sigma, tau = (np.asarray(values) for values in reference.UNPOLARIZED_INPUTS)
        functional = NativeScan()
        unpolarized = functional.unpolarized(rho, sigma, tau, None)
        spin = functional.polarized(0.5 * rho, 0.5 * rho, 0.25 * sigma, 0.25 * sigma, 0.25 * sigma, 0.5 * tau, 0.5 * tau)
        np.testing.assert_allclose(spin[0], unpolarized[0], rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(spin[1], unpolarized[1], rtol=1e-10, atol=1e-14)
        # d/dsigma_uu + d/dsigma_ud + d/dsigma_dd of the total sigma = 4 vsigma / ... : sigma = s_uu + 2 s_ud + s_dd.
        np.testing.assert_allclose(spin[3] + spin[4] + spin[5], 4.0 * unpolarized[2], rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(spin[6], unpolarized[3], rtol=1e-10, atol=1e-14)


def _libxc_available() -> bool:
    try:
        from parsec_python.V_xc.libxc_library import libxc_available

        return bool(libxc_available())
    except Exception:
        return False


@unittest.skipUnless(_libxc_available(), "libxc is not available (PARSEC_LIBXC_LIBRARY)")
class LibxcMetaGGATests(unittest.TestCase):
    def test_libxc_scan_is_the_native_scan_on_the_grid(self) -> None:
        # libxc's evaluator through interface 2 (V_xc/libxc.py), FHC off on both sides.
        from parsec_python.V_xc.libxc import libxc_semilocal
        from parsec_python.V_xc.meta_gga import is_meta_gga, meta_gga_evaluator
        from parsec_python.models import SCFSettings

        settings = SCFSettings(xc_functional="libxc:mgga_x_scan+mgga_c_scan")
        self.assertTrue(is_meta_gga(settings))
        self.assertTrue(settings.is_meta_gga)
        evaluator = meta_gga_evaluator(settings)
        self.assertIs(evaluator, meta_gga_evaluator(settings))
        grid = _cluster_grid()
        _, _, density, tau = _fields(grid)
        for functional in (evaluator, libxc_semilocal("libxc:mgga_x_scan+mgga_c_scan")):
            mine = meta_gga_xc(NativeScan(), density, tau, grid)
            theirs = meta_gga_xc(functional, density, tau, grid)
            self.assertAlmostEqual(theirs.total_energy, mine.total_energy, delta=1e-12 * abs(mine.total_energy))
            np.testing.assert_allclose(theirs.potential, mine.potential, rtol=0, atol=1e-12)
            np.testing.assert_allclose(theirs.kinetic_potential, mine.kinetic_potential, rtol=0, atol=1e-12)
        spin_mine = meta_gga_xc_spin(NativeScan(), 0.6 * density, 0.4 * density, 0.6 * tau, 0.4 * tau, grid)
        spin_theirs = meta_gga_xc_spin(evaluator, 0.6 * density, 0.4 * density, 0.6 * tau, 0.4 * tau, grid)
        self.assertAlmostEqual(spin_theirs.total_energy, spin_mine.total_energy, delta=1e-12 * abs(spin_mine.total_energy))
        np.testing.assert_allclose(spin_theirs.potential_up, spin_mine.potential_up, rtol=0, atol=1e-12)
        np.testing.assert_allclose(spin_theirs.kinetic_potential_down, spin_mine.kinetic_potential_down, rtol=0, atol=1e-12)


class GridLayerTests(unittest.TestCase):
    def assert_derivative(self, energy, derivative, direction, tolerance=1.0e-8):
        def central(epsilon):
            return (energy(epsilon) - energy(-epsilon)) / (2.0 * epsilon)

        change = (4.0 * central(2.5e-6) - central(5.0e-6)) / 3.0
        self.assertAlmostEqual(change, derivative, delta=tolerance * max(1.0, abs(change)))

    def test_sparse_gradient_is_the_density_gradient(self) -> None:
        for grid in (_cluster_grid(), _periodic_grid()):
            field = np.exp(-0.2 * np.sum((grid.coordinates - grid.coordinates.mean(axis=0)) ** 2, axis=1))
            np.testing.assert_allclose(grid_gradient(grid).apply(field), density_gradient(field, grid), rtol=0, atol=1e-14)

    def test_potentials_are_the_derivatives_of_the_discrete_energy(self) -> None:
        for functional in (NativeScan(), NativeScan(enforce_fhc=True), _LaplacianTestFunctional()):
            for grid in (_cluster_grid(), _periodic_grid()):
                laplacian = build_negative_laplacian(grid)
                _, _, density, tau = _fields(grid)
                core = 0.02 * np.exp(-2.0 * np.sum((grid.coordinates - grid.coordinates.mean(axis=0)) ** 2, axis=1))
                result = meta_gga_xc(functional, density, tau, grid, core, negative_laplacian=laplacian)
                direction = _relative_direction(grid, density)
                self.assert_derivative(
                    lambda e: meta_gga_xc(functional, density + e * direction, tau, grid, core, negative_laplacian=laplacian).total_energy,
                    grid.volume_element * float(np.dot(result.potential, direction)),
                    direction,
                )
                # vtau: 2 h^3 sum vtau dtau (Ry) along a tau direction.
                tau_direction = _relative_direction(grid, tau, 2)
                self.assert_derivative(
                    lambda e: meta_gga_xc(functional, density, tau + e * tau_direction, grid, core, negative_laplacian=laplacian).total_energy,
                    2.0 * grid.volume_element * float(np.dot(result.kinetic_potential, tau_direction)),
                    tau_direction,
                )

    def test_spin_potentials_are_the_derivatives_of_the_discrete_energy(self) -> None:
        for functional in (NativeScan(), _LaplacianTestFunctional()):
            for grid in (_cluster_grid(), _periodic_grid()):
                laplacian = build_negative_laplacian(grid)
                _, _, up, tau_up = _fields(grid, (1.0, 1.0, 0.5), 1.0)
                _, _, down, tau_down = _fields(grid, (1.0, 0.2), 1.0)
                result = meta_gga_xc_spin(functional, up, down, tau_up, tau_down, grid, None, negative_laplacian=laplacian)
                for channel in (0, 1):
                    direction = _relative_direction(grid, (up, down)[channel], channel)
                    tau_direction = _relative_direction(grid, (tau_up, tau_down)[channel], channel + 1)

                    def energy(e, channel=channel, direction=direction):
                        fields = [up, down]
                        fields[channel] = fields[channel] + e * direction
                        return meta_gga_xc_spin(functional, *fields, tau_up, tau_down, grid, None, negative_laplacian=laplacian).total_energy

                    self.assert_derivative(
                        energy, grid.volume_element * float(np.dot(result.potentials[channel], direction)), direction
                    )

                    def tau_energy(e, channel=channel, tau_direction=tau_direction):
                        taus = [tau_up, tau_down]
                        taus[channel] = taus[channel] + e * tau_direction
                        return meta_gga_xc_spin(functional, up, down, *taus, grid, None, negative_laplacian=laplacian).total_energy

                    self.assert_derivative(
                        tau_energy,
                        2.0 * grid.volume_element * float(np.dot(result.kinetic_potentials[channel], tau_direction)),
                        tau_direction,
                    )

    def test_weizsaecker_floor_is_consistent(self) -> None:
        # An evaluator undefined below tau_W (as libxc's polarized SCAN, NaN
        # there) with tau_floor: the grid layer passes max(tau, tau_W) and the
        # potentials are the derivatives of that energy.
        class Floored(NativeScan):
            tau_floor = "weizsaecker"

            def unpolarized(self, rho, sigma, tau, lapl=None):
                if np.any((rho > 1e-20) & (tau < sigma / (8.0 * np.maximum(rho, 1e-300)) * (1 - 1e-12))):
                    raise AssertionError("the evaluator saw tau below tau_W")
                return super().unpolarized(rho, sigma, tau, lapl)

            def polarized(self, n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d, l_u=None, l_d=None):
                for n, s, t in ((n_u, s_uu, t_u), (n_d, s_dd, t_d)):
                    if np.any((n > 1e-20) & (t < s / (8.0 * np.maximum(n, 1e-300)) * (1 - 1e-12))):
                        raise AssertionError("the evaluator saw tau below tau_W")
                return super().polarized(n_u, n_d, s_uu, s_ud, s_dd, t_u, t_d, l_u, l_d)

        functional = Floored()
        grid = _cluster_grid()
        _, _, density, tau = _fields(grid)
        tau = 0.6 * tau  # below tau_W at many points
        core = 0.02 * np.exp(-2.0 * np.sum((grid.coordinates - grid.coordinates.mean(axis=0)) ** 2, axis=1))
        result = meta_gga_xc(functional, density, tau, grid, core)
        self.assertGreater(result.fermi_hole_clamped, 10)
        self.assertTrue(np.all(np.isfinite(result.potential)))
        direction = _relative_direction(grid, density)
        self.assert_derivative(
            lambda e: meta_gga_xc(functional, density + e * direction, tau, grid, core).total_energy,
            grid.volume_element * float(np.dot(result.potential, direction)),
            direction,
        )
        tau_direction = _relative_direction(grid, tau, 2)
        self.assert_derivative(
            lambda e: meta_gga_xc(functional, density, tau + e * tau_direction, grid, core).total_energy,
            2.0 * grid.volume_element * float(np.dot(result.kinetic_potential, tau_direction)),
            tau_direction,
        )
        # (tau ratios that put no point exactly on the kink tau = tau_W)
        spin = meta_gga_xc_spin(functional, 0.6 * density, 0.4 * density, 0.53 * tau, 0.37 * tau, grid, core)
        self.assertTrue(np.all(np.isfinite(spin.potential_up)) and np.all(np.isfinite(spin.potential_down)))
        self.assert_derivative(
            lambda e: meta_gga_xc_spin(functional, 0.6 * density + e * direction, 0.4 * density, 0.53 * tau, 0.37 * tau, grid, core).total_energy,
            grid.volume_element * float(np.dot(spin.potential_up, direction)),
            direction,
        )

    def test_absolute_tau_floor_keeps_the_energy_and_is_consistent(self) -> None:
        # tau_minimum: points below it keep the energy of the floored input
        # (never dropped), vtau = 0 there, and the potentials are the
        # derivatives of that energy, unpolarized and spin, with and without
        # the tau_W floor.
        grid = _cluster_grid()
        _, _, density, tau = _fields(grid)
        minimum = float(np.quantile(tau, 0.3))
        for weizsaecker in (False, True):
            functional = NativeScan()
            functional.tau_minimum = minimum
            if weizsaecker:
                functional.tau_floor = "weizsaecker"
            result = meta_gga_xc(functional, density, tau, grid)
            low = (tau < minimum) & (density > 1e-20)
            self.assertGreater(int(np.count_nonzero(low)), 20)
            self.assertTrue(np.all(result.kinetic_potential[low] == 0.0))
            if not weizsaecker:
                gradients = grid_gradient(grid).apply(density)
                sigma = np.einsum("ij,ij->j", gradients, gradients)
                expected = NativeScan().unpolarized(density, sigma, np.maximum(tau, minimum))[0]
                np.testing.assert_allclose(result.energy_density, 2.0 * expected, rtol=1e-13, atol=0)
            direction = _relative_direction(grid, density)
            self.assert_derivative(
                lambda e: meta_gga_xc(functional, density + e * direction, tau, grid).total_energy,
                grid.volume_element * float(np.dot(result.potential, direction)),
                direction,
            )
            tau_direction = _relative_direction(grid, tau, 2)
            self.assert_derivative(
                lambda e: meta_gga_xc(functional, density, tau + e * tau_direction, grid).total_energy,
                2.0 * grid.volume_element * float(np.dot(result.kinetic_potential, tau_direction)),
                tau_direction,
            )
            spin = meta_gga_xc_spin(functional, 0.6 * density, 0.4 * density, 0.53 * tau, 0.47 * tau, grid)
            self.assert_derivative(
                lambda e: meta_gga_xc_spin(
                    functional, 0.6 * density + e * direction, 0.4 * density, 0.53 * tau, 0.47 * tau, grid
                ).total_energy,
                grid.volume_element * float(np.dot(spin.potential_up, direction)),
                direction,
            )

    def test_gradient_form_follows_the_smoothness_at_tau_w(self) -> None:
        # The orbital form puts one-orbital regions at tau = tau_W: auto takes
        # it for functionals smooth there and the density form for a kink
        # (an evaluator floored at tau_W, TPSS-type correlation).
        from types import SimpleNamespace
        from unittest import mock

        from parsec_python.V_xc import meta_gga as module

        self.assertTrue(module.smooth_at_weizsaecker(NativeScan()))
        floored = NativeScan()
        floored.tau_floor = "weizsaecker"
        self.assertFalse(module.smooth_at_weizsaecker(floored))

        class Clamped(NativeScan):
            # tau_W/tau clamped at 1 in the correlation only, as TPSS's z.
            def unpolarized(self, rho, sigma, tau, lapl=None):
                exchange = NativeScan(1.0, 0.0).unpolarized(rho, sigma, tau, lapl)
                weizsaecker = sigma / (8.0 * rho)
                below = tau < weizsaecker
                correlation = list(NativeScan(0.0, 1.0).unpolarized(rho, sigma, np.maximum(tau, weizsaecker), lapl))
                correlation[3] = np.where(below, 0.0, correlation[3])
                return tuple(
                    None if a is None else a + b for a, b in zip(exchange, correlation)
                )

        self.assertFalse(module.smooth_at_weizsaecker(Clamped()))
        settings = SimpleNamespace(xc_functional="scan")
        with mock.patch.dict("os.environ", {"PARSEC_META_GGA_GRADIENT": "auto"}):
            self.assertEqual(module.meta_gga_gradient_form(settings)[0], "orbital")
            with mock.patch.object(module, "meta_gga_evaluator", lambda _: floored):
                module._GRADIENT_FORMS.clear()
                self.assertEqual(module.meta_gga_gradient_form(settings)[0], "density")
            module._GRADIENT_FORMS.clear()
        for request in ("orbital", "density"):
            with mock.patch.dict("os.environ", {"PARSEC_META_GGA_GRADIENT": request}):
                self.assertEqual(module.meta_gga_gradient_form(settings), (request, "PARSEC_META_GGA_GRADIENT=" + request))

    def test_density_switch_is_consistent(self) -> None:
        # The orbital terms switched off below a density: the potentials stay
        # the derivatives of the switched energy, vtau vanishes below the
        # switch and the functional is unchanged above it.
        from parsec_python.V_xc.meta_gga import DensitySwitchedMetaGGA

        grid = _cluster_grid()
        _, _, density, tau = _fields(grid)
        top = float(density.max())
        inner = NativeScan()
        functional = DensitySwitchedMetaGGA(inner, 1e-4 * top, 3e-2 * top)
        switching = (density > 1e-4 * top) & (density < 3e-2 * top)
        self.assertGreater(int(np.count_nonzero(switching)), 50)
        result = meta_gga_xc(functional, density, tau, grid)
        plain = meta_gga_xc(inner, density, tau, grid)
        below = density < 1e-4 * top
        above = density > 3e-2 * top
        self.assertTrue(np.all(result.kinetic_potential[below] == 0.0))
        np.testing.assert_allclose(result.kinetic_potential[above], plain.kinetic_potential[above], rtol=1e-14, atol=0)
        np.testing.assert_allclose(result.energy_density[above], plain.energy_density[above], rtol=1e-14, atol=0)
        direction = _relative_direction(grid, density)
        self.assert_derivative(
            lambda e: meta_gga_xc(functional, density + e * direction, tau, grid).total_energy,
            grid.volume_element * float(np.dot(result.potential, direction)),
            direction,
        )
        tau_direction = _relative_direction(grid, tau, 2)
        self.assert_derivative(
            lambda e: meta_gga_xc(functional, density, tau + e * tau_direction, grid).total_energy,
            2.0 * grid.volume_element * float(np.dot(result.kinetic_potential, tau_direction)),
            tau_direction,
        )
        up, down, tau_up, tau_down = 0.6 * density, 0.4 * density, 0.53 * tau, 0.47 * tau
        spin = meta_gga_xc_spin(functional, up, down, tau_up, tau_down, grid)
        self.assertTrue(np.all(spin.kinetic_potential_down[below] == 0.0))
        for channel in (0, 1):
            direction = _relative_direction(grid, (up, down)[channel], channel)

            def energy(e, channel=channel, direction=direction):
                fields = [up, down]
                fields[channel] = fields[channel] + e * direction
                return meta_gga_xc_spin(functional, *fields, tau_up, tau_down, grid).total_energy

            self.assert_derivative(energy, grid.volume_element * float(np.dot(spin.potentials[channel], direction)), direction)

            tau_direction = _relative_direction(grid, (tau_up, tau_down)[channel], channel + 1)

            def tau_energy(e, channel=channel, tau_direction=tau_direction):
                taus = [tau_up, tau_down]
                taus[channel] = taus[channel] + e * tau_direction
                return meta_gga_xc_spin(functional, up, down, *taus, grid).total_energy

            self.assert_derivative(
                tau_energy,
                2.0 * grid.volume_element * float(np.dot(spin.kinetic_potentials[channel], tau_direction)),
                tau_direction,
            )

    def test_two_equal_channels_are_the_unpolarized_grid_functional(self) -> None:
        grid = _cluster_grid()
        _, _, density, tau = _fields(grid)
        functional = NativeScan()
        unpolarized = meta_gga_xc(functional, density, tau, grid)
        spin = meta_gga_xc_spin(functional, 0.5 * density, 0.5 * density, 0.5 * tau, 0.5 * tau, grid)
        self.assertAlmostEqual(spin.total_energy, unpolarized.total_energy, delta=1e-12 * abs(unpolarized.total_energy))
        np.testing.assert_allclose(spin.potential_up, unpolarized.potential, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(spin.kinetic_potential_up, unpolarized.kinetic_potential, rtol=1e-9, atol=1e-11)

    def test_orbital_term_is_the_orbital_derivative_and_symmetric(self) -> None:
        for grid in (_cluster_grid(), _periodic_grid()):
            q, weights, density, tau = _fields(grid)
            functional = NativeScan()
            result = meta_gga_xc(functional, density, tau, grid)
            operator = KineticPotentialOperator(grid_gradient(grid), result.kinetic_potential)
            gradient = grid_gradient(grid)
            direction = np.cos(0.7 * grid.coordinates[:, 2]) * np.exp(-0.15 * np.sum(grid.coordinates**2, axis=1))

            def energy(e):
                moved = q.copy()
                moved[:, 1] = moved[:, 1] + e * direction
                moved_tau = kinetic_energy_density(moved, weights, gradient, grid.volume_element)
                return meta_gga_xc(functional, density, moved_tau, grid).total_energy

            # dE/de = 2 (degeneracy) f_1 <direction|H_tau|q_1> (Ry).
            self.assert_derivative(
                energy, 2.0 * 2.0 * weights[1] * float(direction @ operator.apply(q[:, 1])), direction
            )
            block = operator.apply(q)
            np.testing.assert_allclose(q.T @ block, (q.T @ block).T, rtol=0, atol=1e-13)
            matrix = operator.as_sparse()
            self.assertLess(abs(matrix - matrix.T).max(), 1e-13)
            np.testing.assert_allclose(matrix @ q, block, rtol=0, atol=1e-12)
            # tau of the orbitals = 1/2 sum w <q|G^T G|q>/h^3 summed: the band term is 2 h^3 sum vtau tau.
            band = 2.0 * float(np.sum(weights * np.einsum("ij,ij->j", q, block)))
            self.assertAlmostEqual(
                band, 2.0 * grid.volume_element * float(np.dot(result.kinetic_potential, tau)), delta=1e-12 * abs(band)
            )

    def test_orbital_gradient_form_is_the_orbital_derivative(self) -> None:
        # sigma = |g|^2 with g = (d/h^3) sum f 2 q G q: the energy depends on the
        # orbitals through n, g and tau; its orbital derivative is
        # 2 d f [vrho + G^T vtau G + 2 sum_c (W_c G_c + G_c^T W_c)] q with W = de/dg.
        from parsec_python.V_xc.meta_gga import orbital_density_gradient

        for grid in (_cluster_grid(), _periodic_grid()):
            q, weights, density, tau = _fields(grid)
            gradient = grid_gradient(grid)
            core = 0.02 * np.exp(-2.0 * np.sum((grid.coordinates - grid.coordinates.mean(axis=0)) ** 2, axis=1))
            functional = NativeScan()

            def energy_of(orbitals):
                n = 2.0 / grid.volume_element * np.sum(orbitals * orbitals * weights[None, :], axis=1)
                t = kinetic_energy_density(orbitals, weights, gradient, grid.volume_element)
                g = orbital_density_gradient(orbitals, weights, gradient, grid.volume_element)
                return meta_gga_xc(functional, n, t, grid, core, density_gradient=g)

            result = energy_of(q)
            # tau >= |g|^2/(8 n) of the valence part: no point below tau_W without the core.
            g = orbital_density_gradient(q, weights, gradient, grid.volume_element)
            self.assertTrue(np.all(tau >= np.einsum("ij,ij->j", g, g) / (8.0 * density) * (1 - 1e-12) - 1e-30))
            operator = KineticPotentialOperator(gradient, result.kinetic_potential, gradient_flux=result.gradient_flux)
            direction = np.cos(0.7 * grid.coordinates[:, 2]) * np.exp(-0.15 * np.sum(grid.coordinates**2, axis=1))

            def energy(e):
                moved = q.copy()
                moved[:, 1] = moved[:, 1] + e * direction
                return energy_of(moved).total_energy

            applied = result.potential * q[:, 1] + operator.apply(q[:, 1])
            # E = sum_m d f_m <q_m|H|q_m>: dE/dq_1 = 2 d f_1 H q_1 (d = 2).
            self.assert_derivative(energy, 2.0 * 2.0 * weights[1] * float(direction @ applied), direction)
            block = operator.apply(q)
            np.testing.assert_allclose(q.T @ block, (q.T @ block).T, rtol=0, atol=1e-12)
            # The core force sees dE/dn_core of the whole (density + core) gradient.
            self.assertIsNotNone(result.core_potential)

    def test_complex_orbital_gradient_term_is_the_orbital_derivative(self) -> None:
        # Complex orbitals (Bloch factors, with and without k): g = (d/h^3) sum f 2 Re(u^* G u)
        # and dE/du^* of E = h^3 sum W . g is d f [2 (W G + G^T W)] u, the k parts cancelling.
        from parsec_python.V_xc.meta_gga import orbital_density_gradient

        for grid in (_cluster_grid(), _periodic_grid()):
            q, weights, _, _ = _fields(grid)
            rng = np.random.default_rng(3)
            phases = np.exp(1j * rng.uniform(0, 2 * np.pi, q.shape))
            u = q * phases
            gradient = grid_gradient(grid)
            flux = rng.standard_normal((3, grid.size))
            volume = grid.volume_element

            def energy_of(orbitals):
                g = orbital_density_gradient(orbitals, weights, gradient, volume)
                return volume * float(np.sum(flux * g))

            direction = (rng.standard_normal(grid.size) + 1j * rng.standard_normal(grid.size)) * np.exp(
                -0.1 * np.sum((grid.coordinates - grid.coordinates.mean(axis=0)) ** 2, axis=1)
            )
            for k in (None, np.array([0.3, -0.2, 0.1])):
                operator = KineticPotentialOperator(gradient, np.zeros(grid.size), k_point=k, gradient_flux=flux)

                def energy(e):
                    moved = u.copy()
                    moved[:, 1] = moved[:, 1] + e * direction
                    return energy_of(moved)

                applied = operator.apply(u[:, 1])
                self.assert_derivative(
                    energy, 2.0 * weights[1] * float(np.real(np.vdot(direction, applied))), direction
                )
                block = operator.apply(u)
                np.testing.assert_allclose(u.conj().T @ block, (u.conj().T @ block).conj().T, rtol=0, atol=1e-11)
            # Real orbitals through the complex path: the real term.
            real = KineticPotentialOperator(gradient, np.zeros(grid.size), gradient_flux=flux)
            np.testing.assert_allclose(real.apply(q.astype(np.complex128)).real, real.apply(q), rtol=0, atol=1e-12)
            # A constant phase per orbital leaves g unchanged.
            turned = q * np.exp(1j * rng.uniform(0, 2 * np.pi, q.shape[1]))[None, :]
            np.testing.assert_allclose(
                orbital_density_gradient(turned, weights, gradient, volume),
                orbital_density_gradient(q, weights, gradient, volume),
                rtol=0,
                atol=1e-12,
            )

    def test_bloch_orbital_term_is_hermitian_and_the_derivative_of_tau(self) -> None:
        grid = _periodic_grid()
        gradient = grid_gradient(grid)
        k_point = np.array([0.31, -0.12, 0.2])
        rng = np.random.default_rng(3)
        u = rng.standard_normal((grid.size, 3)) + 1j * rng.standard_normal((grid.size, 3))
        u, _ = np.linalg.qr(u)
        weights = np.array([1.0, 0.8, 0.4])
        vtau = 0.3 + 0.1 * np.sin(grid.coordinates[:, 0])
        operator = KineticPotentialOperator(gradient, vtau, k_point)
        block = operator.apply(u)
        hermitian = u.conj().T @ block
        np.testing.assert_allclose(hermitian, hermitian.conj().T, rtol=0, atol=1e-11)
        # sum_m w_m <u_m|H_tau|u_m> = 2 h^3 sum vtau tau (degeneracy 2, k-point tau).
        tau = kinetic_energy_density(u, weights, gradient, grid.volume_element, k_point=k_point)
        band = 2.0 * float(np.real(np.sum(weights * np.einsum("ij,ij->j", u.conj(), block))))
        self.assertAlmostEqual(band, 2.0 * grid.volume_element * float(np.dot(vtau, tau)), delta=1e-11 * abs(band))
        # A plane wave e^{iGr} u with the stencil's own G: tau of the phase-twisted factor is that of e^{i(k+G)r}.
        constant = np.full((grid.size, 1), 1.0 / np.sqrt(grid.size), dtype=np.complex128)
        tau_constant = kinetic_energy_density(constant, np.array([1.0]), gradient, grid.volume_element, k_point=k_point)
        np.testing.assert_allclose(tau_constant, float(k_point @ k_point) / grid.volume_element / grid.size, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
