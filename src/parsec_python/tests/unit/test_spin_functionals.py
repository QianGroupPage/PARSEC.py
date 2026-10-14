"""Spin-polarized exchange-correlation functionals.

Three kinds of evidence:

* Pointwise values of the local spin density functional (Slater exchange
  with Perdew-Zunger correlation) and of spin-polarized PBE from libxc 7.0.0,
  evaluated through PySCF 2.11.0 (``dft.libxc.eval_xc`` with ``spin=1``) for
  the densities and gradients of ``POINTS``.
* The unpolarized functionals of this package, which equal spin densities
  must reproduce.
* Finite differences of the discrete energy, whose derivatives the
  potentials are.

PBE agrees with libxc to 1e-8 only: the unpolarized Perdew-Wang parameter
``A`` of this package is ``(1 - ln 2)/pi**2`` and that of libxc (and of
PARSEC) 0.0310907.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.V_xc import (
    ca_lda,
    ca_lda_spin,
    pbe,
    pbe_energy_partials,
    pbe_spin,
    pbe_spin_energy_partials,
    pw92_spin_correlation,
)
from parsec_python.models import GridSettings


# (n_up, n_down, grad n_up, grad n_down) in atomic units.  The fifth point is
# unpolarized, the sixth almost and the last one fully polarized.
POINTS = (
    (1.0e-4, 5.0e-5, (1.0e-4, 0.0, 0.0), (-2.0e-5, 3.0e-5, 0.0)),
    (0.01, 0.02, (0.02, 0.01, 0.0), (0.01, -0.03, 0.02)),
    (0.1, 0.03, (-0.15, 0.05, 0.1), (0.02, 0.04, -0.01)),
    (1.0, 0.4, (1.5, -0.5, 0.25), (-0.3, 0.6, 0.2)),
    (0.3, 0.3, (0.2, 0.1, -0.3), (0.2, 0.1, -0.3)),
    (0.25, 0.0025, (0.3, 0.0, 0.1), (0.001, 0.002, 0.0)),
    (0.5, 0.0, (0.4, -0.2, 0.1), (0.0, 0.0, 0.0)),
)
# libxc, Hartree: energy per volume, d/dn_up, d/dn_down and, for the
# gradient functionals, d/dsigma_uu, d/dsigma_ud, d/dsigma_dd.
LIBXC = {
    "lda_x": (
        (-6.033160928730104e-06, -7.056427829239852e-03, -5.186525811910461e-02, -1.204772626189550e+00, -3.737547532869366e-01, -1.468643497683249e-01, -3.692793831910105e-01),
        (-5.758823822969722e-02, -2.673009235143952e-01, -5.758823822969721e-01, -1.240700981798800e+00, -8.305661184154147e-01, -7.815926417967721e-01, -9.847450218426967e-01),
        (-4.570781497340831e-02, -3.367780601921260e-01, -3.855146420814097e-01, -9.141562994681662e-01, -8.305661184154147e-01, -1.683890300960633e-01, -2.220446049250313e-16),
    ),
    "lda_c_pz": (
        (-2.410234536409973e-06, -1.293264098490181e-03, -6.329247942795672e-03, -9.507928120298215e-02, -3.992252779779538e-02, -8.438438319536834e-03, -1.730916067886439e-02),
        (-1.684324488266743e-02, -6.584117811693747e-02, -4.322962072427725e-02, -6.067132764323707e-02, -7.439996155522585e-02, -3.603262851048618e-02, -3.868853777644667e-02),
        (-2.523970203353842e-02, -4.135915432331193e-02, -9.391374617888915e-02, -1.130772107759236e-01, -7.439996155522585e-02, -1.779886577669352e-01, -2.357915336830822e-01),
    ),
    "lda_c_pw_mod": (
        (-2.417107889484498e-06, -1.291368494828011e-03, -6.433955977701277e-03, -9.716998903387246e-02, -4.021904401105750e-02, -8.502603690383897e-03, -1.727710195557495e-02),
    ),
    "gga_x_pbe": (
        (-9.251924033292811e-06, -8.296442555562769e-03, -5.491045217040474e-02, -1.218737940376548e+00, -3.783690730310938e-01, -1.490020726589567e-01, -3.710474439709642e-01),
        (-6.859685091532765e-02, -2.429284142710947e-01, -5.486719866465714e-01, -1.229605924581312e+00, -8.206269871694850e-01, -7.708103133757128e-01, -9.800863472643477e-01),
        (-5.186945501217163e-02, -3.059068833007901e-01, -3.611033559108274e-01, -8.968871244272861e-01, -8.206269871694850e-01, -1.535028384416371e-01, -1.110223024625157e-16),
        (-7.607124882697417e+01, -7.084604664910688e-01, -6.289882197493721e-02, -3.285154891887693e-03, -1.622665858004571e-02, -2.058982389766512e-02, -8.369199397225231e-03),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-2.463964135181986e+02, -4.205056187595135e-01, -2.934071727792069e-01, -1.084789876678942e-02, -1.622665858004571e-02, -6.935629939948870e+00, 0.0),
    ),
    "gga_c_pbe": (
        (-5.475198056782733e-07, -7.612706012120165e-04, -4.821732162559797e-03, -9.301695614125614e-02, -3.604221972272486e-02, -6.725932820932776e-03, -1.567650667117305e-02),
        (-1.457069861291092e-02, -7.084387655114749e-02, -5.743213573749794e-02, -6.846870681610583e-02, -8.265123640570737e-02, -4.266820579730250e-02, -4.217264085364213e-02),
        (-1.788118290413652e-02, -5.417771538722350e-02, -9.711968105789094e-02, -1.121065747295748e-01, -8.265123640570737e-02, -1.952106120175764e-01, 1.978660206690520e+01),
        (8.922947182185295e+01, 2.134124976211506e-01, 3.844602175920322e-02, 2.388894358736370e-03, 6.668820665895032e-03, 1.435043571004095e-02, 6.878108798793308e-03),
        (1.784589436437059e+02, 4.268249952423011e-01, 7.689204351840644e-02, 4.777788717472740e-03, 1.333764133179006e-02, 2.870087142008191e-02, 1.375621759758662e-02),
        (8.922947182185295e+01, 2.134124976211506e-01, 3.844602175920322e-02, 2.388894358736370e-03, 6.668820665895032e-03, 1.435043571004095e-02, 6.878108798793308e-03),
    ),
}
# Every point but the fully polarized last one: there the derivative with
# respect to the empty channel is singular for PBE and each code regularizes
# it in its own way.
REGULAR = slice(0, 6)


def _points():
    up = np.array([point[0] for point in POINTS])
    down = np.array([point[1] for point in POINTS])
    gradient_up = np.array([point[2] for point in POINTS]).T
    gradient_down = np.array([point[3] for point in POINTS]).T
    return up, down, gradient_up, gradient_down


def _libxc(*names: str) -> list[np.ndarray]:
    """Sum the tables of several libxc functionals, column by column."""

    return [
        np.sum([np.array(LIBXC[name][column]) for name in names], axis=0)
        for column in range(len(LIBXC[names[0]]))
    ]


def _test_grid():
    return build_cluster_grid(
        GridSettings(spacing=0.7, radius=3.2, expansion_order=8, shift=(0.5, 0.5, 0.5))
    )


def _test_densities(grid):
    """Two different smooth spin densities and a core density on ``grid``."""

    radius_squared = np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)
    up = 0.15 * np.exp(-0.5 * radius_squared) + 0.006
    down = (
        0.05 * np.exp(-0.4 * ((grid.coordinates[:, 0] - 0.4) ** 2 + radius_squared))
        + 0.004
    )
    core = 0.05 * np.exp(-2.0 * radius_squared)
    return up, down, core


class LibxcPointwiseTests(unittest.TestCase):
    def test_local_spin_density_functional(self) -> None:
        up, down, _, _ = _points()
        energy, potential_up, potential_down = _libxc("lda_x", "lda_c_pz")
        result = ca_lda_spin(up, down, 1.0)
        # Rydberg here, Hartree in libxc.
        np.testing.assert_allclose(result.energy_density, 2.0 * energy, rtol=1.0e-10)
        np.testing.assert_allclose(result.potential_up, 2.0 * potential_up, rtol=1.0e-10)
        np.testing.assert_allclose(
            result.potential_down[REGULAR], 2.0 * potential_down[REGULAR], rtol=1.0e-9
        )
        self.assertAlmostEqual(
            result.total_energy, float(np.sum(2.0 * energy)), delta=1.0e-9
        )
        # The correlation potential of an empty channel is finite; libxc
        # stops just short of zeta = 1.
        self.assertAlmostEqual(
            result.potential_down[-1], 2.0 * potential_down[-1], delta=1.0e-4
        )

    def test_perdew_wang_spin_interpolation(self) -> None:
        up, down, _, _ = _points()
        density = up + down
        energy, _, _ = pw92_spin_correlation(
            np.cbrt(3.0 / (4.0 * np.pi * density)), (up - down) / density
        )
        np.testing.assert_allclose(
            density * energy, np.array(LIBXC["lda_c_pw_mod"][0]), rtol=2.0e-7
        )

    def test_spin_polarized_pbe(self) -> None:
        up, down, gradient_up, gradient_down = _points()
        energy, potential_up, potential_down, sigma_uu, sigma_ud, sigma_dd = _libxc(
            "gga_x_pbe", "gga_c_pbe"
        )
        total = gradient_up + gradient_down
        f, f_up, f_down, f_sigma_up, f_sigma_down, f_sigma = pbe_spin_energy_partials(
            up,
            down,
            np.sum(gradient_up**2, axis=0),
            np.sum(gradient_down**2, axis=0),
            np.sum(total**2, axis=0),
        )
        np.testing.assert_allclose(f, energy, rtol=3.0e-8)
        np.testing.assert_allclose(f_up, potential_up, rtol=3.0e-8)
        np.testing.assert_allclose(f_down[REGULAR], potential_down[REGULAR], rtol=3.0e-7)
        # libxc differentiates with respect to sigma_uu, sigma_ud and
        # sigma_dd; here the total gradient sigma = uu + 2 ud + dd is a
        # variable of its own.
        np.testing.assert_allclose(f_sigma_up + f_sigma, sigma_uu, rtol=3.0e-7)
        np.testing.assert_allclose(2.0 * f_sigma, sigma_ud, rtol=3.0e-7)
        np.testing.assert_allclose(
            (f_sigma_down + f_sigma)[REGULAR], sigma_dd[REGULAR], rtol=3.0e-7
        )
        # An empty channel has no exchange of its own and a finite potential.
        self.assertEqual(f_sigma_down[-1], 0.0)
        self.assertTrue(np.isfinite(f_down[-1]))


class ClosedShellTests(unittest.TestCase):
    """Equal spin densities give the unpolarized functionals."""

    def test_pbe_partials(self) -> None:
        density = np.array([1.0e-5, 0.01, 0.1, 1.0, 3.0])
        sigma = np.array([1.0e-8, 1.0e-4, 0.03, 0.7, 20.0])
        f, f_up, f_down, f_sigma_up, f_sigma_down, f_sigma = pbe_spin_energy_partials(
            0.5 * density, 0.5 * density, 0.25 * sigma, 0.25 * sigma, sigma
        )
        energy, derivative_density, derivative_sigma = pbe_energy_partials(density, sigma)
        np.testing.assert_allclose(f, energy, rtol=1.0e-14)
        np.testing.assert_allclose(f_up, derivative_density, rtol=1.0e-13)
        np.testing.assert_array_equal(f_up, f_down)
        np.testing.assert_array_equal(f_sigma_up, f_sigma_down)
        # d/dsigma of the unpolarized f at fixed spin ratio:
        # sigma_uu = sigma_dd = sigma/4.
        np.testing.assert_allclose(
            0.5 * f_sigma_up + f_sigma, derivative_sigma, rtol=1.0e-13
        )

    def test_grid_potentials_and_energies(self) -> None:
        grid = _test_grid()
        up, down, core = _test_densities(grid)
        density = up + down
        for name, spin, plain in (
            (
                "ca",
                ca_lda_spin(0.5 * density, 0.5 * density, grid.volume_element, core),
                ca_lda(density, grid.volume_element, core),
            ),
            ("pbe", pbe_spin(0.5 * density, 0.5 * density, grid, core), pbe(density, grid, core)),
        ):
            with self.subTest(functional=name):
                np.testing.assert_allclose(
                    spin.potential_up, plain.potential, rtol=0, atol=1.0e-13
                )
                np.testing.assert_allclose(
                    spin.potential_down, plain.potential, rtol=0, atol=1.0e-13
                )
                np.testing.assert_allclose(
                    spin.energy_density, plain.energy_density, rtol=0, atol=1.0e-14
                )
                self.assertAlmostEqual(spin.total_energy, plain.total_energy, places=12)


class DiscreteDerivativeTests(unittest.TestCase):
    """Each potential is the derivative of the summed energy."""

    def test_potentials_are_energy_derivatives(self) -> None:
        grid = _test_grid()
        up, down, core = _test_densities(grid)
        generator = np.random.default_rng(11)
        functionals = {
            "ca": lambda a, b: ca_lda_spin(a, b, grid.volume_element, core),
            "pbe": lambda a, b: pbe_spin(a, b, grid, core),
        }
        step = 1.0e-5
        for name, functional in functionals.items():
            result = functional(up, down)
            for channel in range(2):
                with self.subTest(functional=name, channel=channel):
                    direction = generator.normal(size=grid.size)
                    direction /= np.linalg.norm(direction)
                    shift = [np.zeros(grid.size), np.zeros(grid.size)]
                    shift[channel] = step * direction
                    finite_difference = (
                        functional(up + shift[0], down + shift[1]).total_energy
                        - functional(up - shift[0], down - shift[1]).total_energy
                    ) / (2.0 * step)
                    self.assertAlmostEqual(
                        finite_difference,
                        grid.volume_element
                        * float(np.dot(result.potentials[channel], direction)),
                        delta=2.0e-9,
                    )

    def test_half_of_the_core_density_enters_each_channel(self) -> None:
        grid = _test_grid()
        up, down, core = _test_densities(grid)
        for name, with_core, added in (
            (
                "ca",
                ca_lda_spin(up, down, grid.volume_element, core),
                ca_lda_spin(up + 0.5 * core, down + 0.5 * core, grid.volume_element),
            ),
            (
                "pbe",
                pbe_spin(up, down, grid, core),
                pbe_spin(up + 0.5 * core, down + 0.5 * core, grid),
            ),
        ):
            with self.subTest(functional=name):
                np.testing.assert_array_equal(with_core.potential_up, added.potential_up)
                np.testing.assert_array_equal(
                    with_core.potential_down, added.potential_down
                )
                self.assertEqual(with_core.total_energy, added.total_energy)


class EmptyChannelTests(unittest.TestCase):
    """A channel without electrons: the hydrogen atom.

    ``exc_spn.f90`` skips the local functional where a spin density is zero
    and ``spn_pbe.f90`` drops the exchange energy there.  The functionals
    here are the limits of their formulas.
    """

    def test_fully_polarized_local_density_exchange(self) -> None:
        density = np.array([0.02, 0.3, 1.5])
        polarized = ca_lda_spin(density, np.zeros(3), 1.0)
        unpolarized = ca_lda_spin(0.5 * density, 0.5 * density, 1.0)
        # Exchange alone: E_x[n, 0] = 2**(1/3) E_x[n/2, n/2].  Correlation
        # is weaker in the polarized gas.
        exchange = -2.0 * 0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * density ** (4.0 / 3.0)
        self.assertTrue(
            np.all(polarized.energy_density < 2.0 ** (1.0 / 3.0) * exchange)
        )
        self.assertTrue(
            np.all(
                polarized.energy_density - 2.0 ** (1.0 / 3.0) * exchange
                > unpolarized.energy_density - exchange
            )
        )
        self.assertTrue(np.all(np.isfinite(polarized.potential_down)))

    def test_energy_is_continuous_at_an_empty_channel(self) -> None:
        grid = _test_grid()
        up, _, _ = _test_densities(grid)
        for name, functional in (
            ("ca", lambda a, b: ca_lda_spin(a, b, grid.volume_element)),
            ("pbe", lambda a, b: pbe_spin(a, b, grid)),
        ):
            with self.subTest(functional=name):
                empty = functional(up, np.zeros(grid.size))
                nearly = functional(up, 1.0e-12 * up)
                self.assertTrue(np.all(np.isfinite(empty.potential_up)))
                self.assertTrue(np.all(np.isfinite(empty.potential_down)))
                self.assertAlmostEqual(
                    empty.total_energy, nearly.total_energy, delta=1.0e-9
                )
                np.testing.assert_allclose(
                    empty.potential_up, nearly.potential_up, rtol=0, atol=1.0e-7
                )
                # Exchanging the channels exchanges the potentials.
                swapped = functional(np.zeros(grid.size), up)
                np.testing.assert_array_equal(swapped.potential_down, empty.potential_up)
                self.assertEqual(swapped.total_energy, empty.total_energy)

    def test_vacuum_and_invalid_densities(self) -> None:
        zero = np.zeros(4)
        result = ca_lda_spin(zero, zero, 0.3)
        np.testing.assert_array_equal(result.potential_up, zero)
        self.assertEqual(result.total_energy, 0.0)
        partials = pbe_spin_energy_partials(zero, zero, zero, zero, zero)
        for values in partials:
            np.testing.assert_array_equal(values, zero)
        with self.assertRaisesRegex(ValueError, "nonnegative spin densities"):
            ca_lda_spin(np.array([0.1, -0.1]), np.array([0.1, 0.1]), 1.0)
        with self.assertRaisesRegex(ValueError, "same shape"):
            ca_lda_spin(np.ones(3), np.ones(4), 1.0)
        with self.assertRaisesRegex(ValueError, "nonnegative spin densities"):
            pbe_spin_energy_partials(-np.ones(2), np.ones(2), zero[:2], zero[:2], zero[:2])
        with self.assertRaisesRegex(ValueError, "active real-space grid"):
            pbe_spin(np.ones(3), np.ones(3), _test_grid())


if __name__ == "__main__":
    unittest.main()
