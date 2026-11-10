"""Non-additive kinetic functionals of PARSEC 1.5's FDET (``potential/kinetic.f90``).

* The pointwise functions equal ``pw91k``, ``pbek`` and ``tfwk`` of kinetic.f90:180-283,
  compiled verbatim with gfortran 13 in a small driver program: the table below was
  printed by that driver (``es25.17``).
* The potential of the default form is the derivative of its energy; the ``parsec1.5``
  form of ``pw`` and ``pb`` is not (``u`` twice Burke's), and its ``tw`` is
  ``T_TF + T_vW/18`` instead of the gradient expansion ``T_TF + T_vW/9``.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Embedding.kinetic import (
    KineticFunctional,
    THOMAS_FERMI_CONSTANT,
    _pbek,
    _pw91k,
    _tfwk,
    kinetic_functional,
    nonadditive_kinetic,
)
from parsec_python.Grid import build_cluster_grid
from parsec_python.models import GridSettings

# rho, s, u, v, |grad rho|, lap rho
_INPUTS = np.array([
    [9.9999999999999995e-07, 3.0000000000000000e+00, 8.6110735023780193e-04, 2.6887098375762292e-01, 8.2241356608665256e-07, -4.4529591937863711e-06],
    [1.0000000000000000e-03, 5.0000000000000000e-01, -3.1826954962020576e-01, -8.9248189949681611e-01, 1.8043080779231544e-04, 6.7010762277726682e-03],
    [1.0000000000000000e-02, 1.2000000000000000e+00, -3.4454456298593072e-01, -5.5842740983794636e-01, 1.4695261505555947e-02, 1.7844350408003036e-02],
    [1.0000000000000001e-01, 2.9999999999999999e-01, 7.3789974298528987e-02, -8.3742124023738418e-01, 8.7755467389820478e-03, 3.4765159722914390e-01],
    [5.0000000000000000e-01, 5.0000000000000003e-02, -9.4095018309955725e-01, -4.1185418493619635e-01, 2.8518341097012661e+00, -3.2238443494624400e+00],
    [2.0000000000000000e+00, 8.0000000000000004e-01, -1.2892145264542125e+00, -2.1158201796721315e-01, 7.6046788886622192e+00, 2.7126435882170155e+00],
    [2.9999999999999999e-02, 7.5000000000000000e+00, 1.0972576063695760e-01, -1.6823785016695894e-01, 2.2650837397384616e-01, -8.0803934376995484e-02],
    [1.0000000000000000e-10, 4.0000000000000000e+01, -3.3950661780750388e-02, 1.0197808740297681e-01, 4.5904072965161804e-10, -2.3887663801696536e-10],
    [2.0000000000000001e-01, 0.0000000000000000e+00, -6.8496335463964764e-01, -7.2795351548303933e-01, 6.3653917403164728e-01, -8.0753467533189660e-01],
    [7.0000000000000007e-02, 2.0000000000000000e-02, -2.2765193461864418e-02, 7.9595088064485653e-01, 1.2255609087609343e-01, -3.9095682354455881e-02],
])
# pw91k (ek, vk), pbek (ek, vk) with mu 0.2319 and kappa 0.8438, tfwk (ek, vk) with 1/72
_FORTRAN = np.array([
    [1.8752487544899680e-04, 1.2581942969092967e-04, 1.6008717390562634e-04, 1.2285008958927657e-04, 1.7358724971926352e-03, 2.3342634978133405e-02],
    [1.0552667332072876e-02, 1.9555217724933658e-02, 1.0542477946272629e-02, 1.9843784666624158e-02, 1.0078739074670542e-02, -1.5669403764752884e-02],
    [5.7616799768082538e-02, 7.9670774617836601e-02, 5.7520962877925984e-02, 6.9073039377596776e-02, 5.1638930809196674e-02, 7.3951090370012046e-02],
    [2.2010064514439578e-01, 4.4773018204523146e-01, 2.1983145485833214e-01, 4.4437495381762121e-01, 2.1546209490405033e-01, 3.4227428255466624e-01],
    [6.3027722535879249e-01, 2.3366263047890641e+00, 6.3032549380247915e-01, 1.1553850701388457e+00, 7.0864289431586214e-01, 1.1598056872177740e+00],
    [1.7859920541011116e+00, 2.3543910304784541e+00, 1.7877562809904042e+00, 2.1934089538946471e+00, 1.6223690435882383e+00, 2.6740755381952694e+00],
    [3.2890465384045742e-01, 1.9719797614267873e-02, 1.7306720438068315e-01, 2.3880646410418710e-01, 2.3442692579967817e-01, 3.1182185729035572e-01],
    [5.4501903782186187e-07, 4.1965511367753777e-06, 3.9682218603985477e-07, 6.5671602616079407e-07, 5.0965137248982710e-02, 6.2520351218280909e-02],
    [3.4199518933533946e-01, 6.6216492707118058e-01, 3.4199518933533940e-01, 6.8545805347019062e-01, 3.6649480102296023e-01, 6.1402284664088924e-01],
    [1.6986268885685979e-01, 2.3576524554708805e-01, 1.6986567877143749e-01, 2.2037117102743786e-01, 1.7726375550316228e-01, 2.9319869264139015e-01],
])


def _grid(spacing=0.3, radius=6.0):
    return build_cluster_grid(GridSettings(spacing=spacing, radius=radius, expansion_order=12, shift=(0.0, 0.0, 0.0)))


def _density(grid):
    r = grid.coordinates

    def atom(center, alpha, weight):
        return weight * np.exp(-alpha * np.linalg.norm(r - np.asarray(center), axis=1))

    return (
        atom((0.71, 0.13, -0.07), 2.0, 1.0 / np.pi)
        + atom((-0.69, -0.05, 0.11), 2.0, 1.0 / np.pi)
        + atom((0.2, 0.3, 1.9), 1.2, 0.05)
    )


class PointwiseFormulasTest(unittest.TestCase):
    def test_equal_to_the_fortran_routines(self):
        rho, s, u, v, grad, lap = _INPUTS.T
        python = np.column_stack(
            [*_pw91k(rho, s, u, v), *_pbek(rho, s, u, v, 0.2319, 0.8438), *_tfwk(rho, grad, lap, 1.0 / 72.0)]
        )
        np.testing.assert_allclose(python, _FORTRAN, rtol=1.0e-15, atol=0.0)

    def test_thomas_fermi_constant_in_rydberg(self):
        self.assertAlmostEqual(THOMAS_FERMI_CONSTANT, 2.0 * 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0), places=14)


class DerivativeTest(unittest.TestCase):
    """``h^3 sum v d`` against the central difference of ``T[rho + e d]``."""

    @classmethod
    def setUpClass(cls):
        cls.grid = _grid()
        cls.rho = _density(cls.grid)
        r = cls.grid.coordinates
        cls.perturbation = np.exp(-((r - np.array([0.9, 0.4, 0.2])) ** 2).sum(1) / 0.5)

    def _relative_error(self, functional):
        rho, d = self.rho, self.perturbation
        base = kinetic_functional(rho, self.grid, functional)
        scale = 1.0e-4 * float(np.max(rho))
        numeric = (
            kinetic_functional(rho + scale * d, self.grid, functional).energy
            - kinetic_functional(rho - scale * d, self.grid, functional).energy
        ) / (2.0 * scale)
        analytic = self.grid.volume_element * float(np.dot(base.potential, d))
        return (analytic - numeric) / numeric

    def test_corrected_potentials_are_the_derivatives(self):
        for name in ("tf", "tw", "pw", "pb"):
            with self.subTest(functional=name):
                self.assertLess(abs(self._relative_error(KineticFunctional(name))), 1.0e-3)

    def test_parsec15_gga_potentials_are_not(self):
        # kinetic.f90:150 doubles u: a few per cent off for this density.
        for name in ("pw", "pb"):
            with self.subTest(functional=name):
                self.assertGreater(abs(self._relative_error(KineticFunctional(name, form="parsec1.5"))), 1.0e-2)

    def test_energies_do_not_depend_on_the_form_but_for_tw(self):
        for name in ("tf", "pw", "pb"):
            with self.subTest(functional=name):
                corrected = kinetic_functional(self.rho, self.grid, KineticFunctional(name)).energy
                fortran = kinetic_functional(self.rho, self.grid, KineticFunctional(name, form="parsec1.5")).energy
                self.assertEqual(corrected, fortran)


class GradientExpansionTest(unittest.TestCase):
    def test_tw_coefficients_on_the_hydrogen_atom(self):
        # rho = exp(-2r)/pi: T_vW = 1 Ry exactly.  The gradient expansion is T_TF + T_vW/9;
        # PARSEC 1.5's tfwk gives T_TF + T_vW/18 (the Hartree coefficient 1/72 in Ry).
        grid = _grid(spacing=0.2, radius=8.0)
        rho = np.exp(-2.0 * np.linalg.norm(grid.coordinates - np.array([0.03, -0.02, 0.01]), axis=1)) / np.pi
        tf = kinetic_functional(rho, grid, KineticFunctional("tf")).energy
        corrected = kinetic_functional(rho, grid, KineticFunctional("tw")).energy - tf
        fortran = kinetic_functional(rho, grid, KineticFunctional("tw", form="parsec1.5")).energy - tf
        self.assertAlmostEqual(corrected, 1.0 / 9.0, delta=2.0e-3)
        self.assertAlmostEqual(fortran, 1.0 / 18.0, delta=1.0e-3)
        self.assertAlmostEqual(corrected, 2.0 * fortran, places=12)
        # TF of the H atom: 2 x 0.2891 Ha.
        self.assertAlmostEqual(tf, 0.5782, delta=2.0e-3)


class NonadditiveTest(unittest.TestCase):
    def test_definition(self):
        grid = _grid()
        rho = _density(grid)
        frozen = 0.4 * np.exp(-np.linalg.norm(grid.coordinates - np.array([0.0, 0.0, -1.5]), axis=1))
        total = rho + frozen
        for name in ("tf", "tw", "pw", "pb", "none"):
            functional = KineticFunctional(name)
            terms = nonadditive_kinetic(total, frozen, grid, functional)
            whole = kinetic_functional(total, grid, functional)
            sample = kinetic_functional(frozen, grid, functional)
            active = kinetic_functional(total - frozen, grid, functional)
            with self.subTest(functional=name):
                self.assertEqual(terms.energy, whole.energy - sample.energy - active.energy)
                np.testing.assert_array_equal(terms.potential, whole.potential - active.potential)

    def test_low_density_points_are_skipped(self):
        grid = _grid()
        rho = np.zeros(grid.size)
        rho[:10] = 1.0e-19
        terms = kinetic_functional(rho, grid, KineticFunctional("pw"))
        self.assertEqual(terms.energy, 0.0)
        self.assertFalse(np.any(terms.potential))

    def test_refuses_unknown_names(self):
        with self.assertRaises(ValueError):
            KineticFunctional("pw91k")
        with self.assertRaises(ValueError):
            KineticFunctional("tf", form="fortran")


if __name__ == "__main__":
    unittest.main()
