"""The exchange-correlation functionals added for parity with Fortran PARSEC.

``Correlation_Type`` values ``pl`` (PW92), ``xa``, ``wi``, ``hl``, ``lb``,
``cs`` and ``pw`` (PW91).  Evidence:

* pointwise values from libxc 7.0.0 through PySCF 2.11.0
  (``dft.libxc.eval_xc``) for the local density functionals, with the same
  parameters where libxc has them;
* the energy densities of the compiled Fortran routines ``pw91`` (pbe.f90)
  and ``spn_pbe`` (spn_pbe.f90) at the points of ``PW91_POINTS``, called by
  a small driver built with gfortran 13.3 from the PARSEC sources;
* finite differences: every potential and partial derivative is the
  derivative of its energy;
* the unpolarized functional equals the spin-polarized one at equal spin
  densities;
* complete calculations against ``parsec.out`` of Fortran PARSEC
  (``data/xc_parsec_reference.json``), here for PW92 and Casida-Salahub on
  the readable solver; the accelerated backends run more of the fixture
  (``acceleration/tests/test_xc_functionals.py``).
"""

from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from parsec_python import PeriodicCell, PeriodicGridSettings
from parsec_python.Grid import build_cluster_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Output import ParsecTextReporter
from parsec_python.SCF import prepare_periodic_single_point, prepare_single_point, run_scf
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.V_xc import (
    ca_lda,
    ca_lda_spin,
    density_gradient,
    evaluate_spin_xc,
    evaluate_unpolarized_xc,
    first_derivative_coefficients,
    hedin_lundqvist_lda,
    lb94,
    lb94_potential_correction,
    lb94_spin,
    pw91,
    pw91_energy_partials,
    pw91_spin,
    pw91_spin_energy_partials,
    pw92_correlation,
    pw92_lda,
    pw92_lda_spin,
    pw92_lsd_correlation,
    wigner_lda,
    xalpha_lda,
)
from parsec_python.V_xc.selection import UNPOLARIZED_ONLY_FUNCTIONALS
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.tests.spin_reference import load_cases, pseudopotential_file
from parsec_python.tests.xc_reference import (
    assert_matches_fortran_case,
    load_xc_cases,
    write_xc_case,
)

_pw91_module = importlib.import_module("parsec_python.V_xc.pw91")
_asymptotic_module = importlib.import_module("parsec_python.V_xc.asymptotic_correction")

# Unpolarized points: density and squared gradient (atomic units).
RHO = np.array([1.0e-4, 0.003, 0.02, 0.1, 0.4, 1.5])
SIGMA = np.array([2.0e-9, 1.0e-5, 4.0e-4, 0.02, 0.3, 6.0])
# Spin points: n_up, n_down and their gradients.
UP = np.array([1.0e-4, 0.01, 0.1, 1.0, 0.25])
DOWN = np.array([5.0e-5, 0.02, 0.03, 0.4, 0.0025])
GRAD_UP = np.array(
    [[1.0e-4, 0.0, 0.0], [0.02, 0.01, 0.0], [-0.15, 0.05, 0.1], [1.5, -0.5, 0.25], [0.3, 0.0, 0.1]]
).T
GRAD_DOWN = np.array(
    [[-2.0e-5, 3.0e-5, 0.0], [0.01, -0.03, 0.02], [0.02, 0.04, -0.01], [-0.3, 0.6, 0.2], [0.001, 0.002, 0.0]]
).T

# libxc (Hartree): energy per volume and d/dn at RHO (lda), plus d/dsigma
# at SIGMA (gga).
LIBXC = {
    "lda_x": (
        (-3.4280861230056235e-06, -0.0003195558190383724, -0.004009513852715926, -0.03428086123005624, -0.21766990071586856, -1.2681581082601796),
        (-0.04570781497340832, -0.14202480846149884, -0.2673009235143951, -0.4570781497340832, -0.7255663357195619, -1.1272516517868265),
    ),
    "lda_c_pw_mod": (
        (-1.5316224758011773e-06, -9.195135721266533e-05, -0.0008420918053118427, -0.005325090691547259, -0.025518302346181895, -0.11186719486628313),
        (-0.018796901133371857, -0.03617022000641343, -0.048639331046144596, -0.06055395856471968, -0.07169517392490173, -0.08298289347154153),
    ),
    "lda_c_hl": (
        (-1.727289380370075e-06, -0.00010221496495646262, -0.0009150832128892589, -0.005650488739991214, -0.026470692065388348, -0.11345379011392978),
        (-0.02124893573322497, -0.039868745643685824, -0.05222900568863378, -0.06336370365479022, -0.07325713393619537, -0.08285910243371543),
    ),
    # libxc's Wigner correlation has a = -0.44 Ha, PARSEC's -0.875529 Ry.
    "lda_c_wigner": (
        (-2.0788993151584132e-06, -0.00010907945963177351, -0.0008725492367422777, -0.004815846085753196, -0.020365785356809173, -0.0791184187501412),
        (-0.02516485341211098, -0.04066772596380458, -0.04692284886954513, -0.05050669678966517, -0.05256791774733238, -0.05388780365430833),
    ),
    # Slater's X-alpha beyond Dirac exchange, alpha = 1.
    "lda_c_xalpha": (
        (-1.7140430615028117e-06, -0.0001597779095191862, -0.002004756926357963, -0.01714043061502812, -0.10883495035793428, -0.6340790541300898),
        (-0.02285390748670416, -0.07101240423074942, -0.13365046175719755, -0.2285390748670416, -0.36278316785978093, -0.5636258258934133),
    ),
    "gga_x_pw91": (
        (-4.52798003120264e-06, -0.0003906842473073055, -0.004316759049717438, -0.03617583397100734, -0.22258836638997184, -1.285523711260711),
        (-0.042202802632734515, -0.12933906752056462, -0.25316892185433126, -0.4381725088876717, -0.7114139192221766, -1.1128600929284729),
        (-340.69245843688793, -4.983567269301753, -0.6490315273780874, -0.08282239511079927, -0.015273651038864806, -0.0027963422263482616),
    ),
    "gga_c_pw91": (
        (-4.833619509998551e-07, -3.653505944116533e-05, -0.000603431621378293, -0.003967211964000235, -0.02194378385222688, -0.0994057931348311),
        (-0.018386197708217762, -0.03906245702537346, -0.057991247159297966, -0.07090487513509817, -0.08020579125609663, -0.09101150642553232),
        (240.25292991679154, 2.884156935948746, 0.4582004500459983, 0.05139933524309162, 0.00997359906298753, 0.0017521422842544413),
    ),
}
# libxc spin (Hartree): energy per volume, d/dn_up, d/dn_down and, for the
# gradient functionals, d/dsigma_uu, d/dsigma_ud, d/dsigma_dd.
LIBXC_SPIN = {
    "lda_x": (
        (-6.033160928730104e-06, -0.007056427829239852, -0.05186525811910461, -1.20477262618955, -0.14686434976832488),
        (-0.05758823822969722, -0.2673009235143952, -0.5758823822969721, -1.2407009817988, -0.7815926417967721),
        (-0.04570781497340831, -0.33677806019212597, -0.38551464208140973, -0.9141562994681662, -0.16838903009606326),
    ),
    "lda_c_pw_mod": (
        (-2.417107889484498e-06, -0.0012913684948280108, -0.006433955977701277, -0.09716998903387246, -0.008502603690383897),
        (-0.016998190037488616, -0.06374191453680067, -0.04585401610146817, -0.06465468821938497, -0.036162563839730814),
        (-0.025045179961642662, -0.042440675672542796, -0.09076087240107143, -0.10899694845409669, -0.21106758302390044),
    ),
    "gga_x_pw91": (
        (-9.393884105053735e-06, -0.008315988927356192, -0.05519637253744903, -1.22129316025866, -0.14938783826130622),
        (-0.06821042585887951, -0.24378584159333683, -0.5504424488746935, -1.2273323946437866, -0.7697227739951135),
        (-0.050356892441390494, -0.3103820077523811, -0.36369761996088723, -0.8959056641053967, -0.15563950825241396),
        (-82.86240493694106, -0.7020168679443478, -0.06438664508166332, -0.003938212927424424, -0.023531476353631217),
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-281.71946167184825, -0.40351718970575035, -0.2911731588120552, -0.012080432439949974, -6.669878852293988),
    ),
    "gga_c_pw91": (
        (-5.694836755759881e-07, -0.0007856030313703167, -0.004890511588481155, -0.09292833686173184, -0.006797626341391347),
        (-0.014898563094944644, -0.07002671905119721, -0.05631690161779979, -0.06801498596468596, -0.04199008303586553),
        (-0.01830468631240519, -0.053211548979987824, -0.09605770689926131, -0.11169879879400275, -0.1945460667684138),
        (90.357749175542, 0.1995597852061658, 0.03565684602645455, 0.00227326460477672, 0.013309684922121378),
        (180.71549835108394, 0.39911957041233154, 0.0713136920529091, 0.00454652920955344, 0.026619369844242756),
        (90.357749175542, 0.1995597852061658, 0.03565684602645455, 0.00227326460477672, 0.013309684922121378),
    ),
}
# Fortran PARSEC's pw91 (RHO, |grad n| = sqrt(SIGMA)): rho (e_x + e_c + H)
# and rho e_x; spn_pbe's PW91 at the spin points: rho (e_x + e_c + H).
# Hartree per volume.
FORTRAN_PW91 = (
    (-5.01134168501923877e-06, -4.27219223271038217e-04, -4.92018910377385772e-03, -4.01430322075748414e-02, -2.44532070115820455e-01, -1.38492915804239947e00),
    (-4.52798003120264054e-06, -3.90684247307305523e-04, -4.31675904971744011e-03, -3.61758339710073484e-02, -2.22588366303235891e-01, -1.28552369877477002e00),
)
FORTRAN_SPIN_PW91 = (
    -9.96336731862496844e-06, -9.10158938859808794e-03, -6.00868671516880046e-02, -1.31422209480158947e00, -1.56185518907305532e-01,
)


def _libxc(table, *names):
    return [
        np.sum([np.array(table[name][column]) for name in names], axis=0)
        for column in range(len(table[names[0]]))
    ]


def _spin_sigmas():
    total = GRAD_UP + GRAD_DOWN
    return (
        np.sum(GRAD_UP**2, axis=0),
        np.sum(GRAD_DOWN**2, axis=0),
        np.sum(total**2, axis=0),
    )


def _richardson(function, value, step=1.0e-3):
    """Central difference of ``function`` at ``value`` with one Richardson step."""

    def central(h):
        return (function(value * (1.0 + h)) - function(value * (1.0 - h))) / (2.0 * h * value)

    return (4.0 * central(0.5 * step) - central(step)) / 3.0


def _cluster_grid():
    return build_cluster_grid(
        GridSettings(spacing=0.6, radius=3.6, expansion_order=8, shift=(0.5, 0.5, 0.5))
    )


def _periodic_grid():
    return build_periodic_grid(
        PeriodicCell(lattice_vectors=np.diag([6.0, 6.6, 7.2])),
        PeriodicGridSettings(spacing=0.6, expansion_order=6),
    )


def _densities(grid):
    """Two smooth, different spin densities and a core density on ``grid``."""

    center = grid.coordinates.mean(axis=0)
    x = grid.coordinates - center
    r2 = np.einsum("ij,ij->i", x, x)
    up = 0.15 * np.exp(-0.5 * r2) + 0.004 * np.exp(-0.05 * r2)
    down = 0.05 * np.exp(-0.4 * ((x[:, 0] - 0.4) ** 2 + r2)) + 0.003 * np.exp(-0.04 * r2)
    core = 0.05 * np.exp(-2.0 * r2)
    return up, down, core


class LocalDensityFunctionalTests(unittest.TestCase):
    def test_perdew_wang_lda_is_libxc_pw_mod(self) -> None:
        # libxc's PW_MOD has the Fortran's A = 0.0310907.
        energy, potential = _libxc(LIBXC, "lda_x", "lda_c_pw_mod")
        result = pw92_lda(RHO, 1.0)
        np.testing.assert_allclose(result.energy_density, 2.0 * energy, rtol=1.0e-13)
        np.testing.assert_allclose(result.potential, 2.0 * potential, rtol=1.0e-13)
        np.testing.assert_allclose(result.energy_per_electron, 2.0 * energy / RHO, rtol=1.0e-13)

    def test_perdew_wang_lsda_is_libxc_pw_mod(self) -> None:
        # The Fortran truncates gamma = 0.5198421 and f''(0) = 1.709921;
        # libxc has all digits of both.
        energy, potential_up, potential_down = _libxc(LIBXC_SPIN, "lda_x", "lda_c_pw_mod")
        result = pw92_lda_spin(UP, DOWN, 1.0)
        np.testing.assert_allclose(result.energy_density, 2.0 * energy, rtol=1.0e-9)
        np.testing.assert_allclose(result.potential_up, 2.0 * potential_up, rtol=1.0e-8)
        np.testing.assert_allclose(result.potential_down, 2.0 * potential_down, rtol=5.0e-8)

    def test_perdew_wang_spin_limit(self) -> None:
        rs = np.array([0.1, 0.7, 1.0, 3.0, 12.0])
        energy, energy_rs = pw92_correlation(rs)
        spin_energy, spin_energy_rs, spin_energy_zeta = pw92_lsd_correlation(rs, np.zeros_like(rs))
        np.testing.assert_array_equal(spin_energy, energy)
        np.testing.assert_array_equal(spin_energy_rs, energy_rs)
        np.testing.assert_array_equal(spin_energy_zeta, 0.0)
        density = np.array([1.0e-5, 0.01, 0.2, 2.0])
        unpolarized = pw92_lda(density, 0.5)
        spin = pw92_lda_spin(0.5 * density, 0.5 * density, 0.5)
        np.testing.assert_allclose(spin.energy_density, unpolarized.energy_density, rtol=1.0e-14)
        np.testing.assert_allclose(spin.potential_up, unpolarized.potential, rtol=1.0e-14)
        np.testing.assert_array_equal(spin.potential_up, spin.potential_down)
        self.assertAlmostEqual(spin.total_energy, unpolarized.total_energy, delta=1.0e-14)

    def test_x_alpha_and_hedin_lundqvist_are_libxc(self) -> None:
        for function, name in ((xalpha_lda, "lda_c_xalpha"), (hedin_lundqvist_lda, "lda_c_hl")):
            energy, potential = _libxc(LIBXC, "lda_x", name)
            result = function(RHO, 1.0)
            np.testing.assert_allclose(result.energy_density, 2.0 * energy, rtol=1.0e-13, err_msg=name)
            np.testing.assert_allclose(result.potential, 2.0 * potential, rtol=1.0e-13, err_msg=name)

    def test_wigner_is_libxc_with_the_parsec_parameter(self) -> None:
        result = wigner_lda(RHO, 1.0)
        exchange_energy, exchange_potential = _libxc(LIBXC, "lda_x")
        correlation_energy, correlation_potential = _libxc(LIBXC, "lda_c_wigner")
        # Both energy and potential are proportional to a_w.
        scale = -0.875529 / (2.0 * -0.44)
        np.testing.assert_allclose(
            result.energy_density - 2.0 * exchange_energy,
            2.0 * scale * correlation_energy,
            rtol=1.0e-13,
        )
        np.testing.assert_allclose(
            result.potential - 2.0 * exchange_potential,
            2.0 * scale * correlation_potential,
            rtol=1.0e-13,
        )

    def test_potentials_are_derivatives_of_the_energy(self) -> None:
        density = np.array([1.0e-5, 3.0e-3, 0.05, 0.5, 3.0])
        for function in (pw92_lda, xalpha_lda, wigner_lda, hedin_lundqvist_lda):
            result = function(density, 1.0)
            derivative = _richardson(lambda n: function(n, 1.0).energy_density, density)
            np.testing.assert_allclose(result.potential, derivative, rtol=1.0e-9, err_msg=function.__name__)
        up = np.array([1.0e-5, 3.0e-3, 0.05, 0.5, 0.2])
        down = np.array([2.0e-5, 1.0e-3, 0.06, 0.1, 0.002])
        result = pw92_lda_spin(up, down, 1.0)
        np.testing.assert_allclose(
            result.potential_up,
            _richardson(lambda n: pw92_lda_spin(n, down, 1.0).energy_density, up),
            rtol=1.0e-9,
        )
        np.testing.assert_allclose(
            result.potential_down,
            _richardson(lambda n: pw92_lda_spin(up, n, 1.0).energy_density, down),
            rtol=1.0e-9,
        )

    def test_core_density_and_vacuum(self) -> None:
        valence = np.array([0.0, 0.0, 0.02, 0.3])
        core = np.array([0.0, 0.01, 0.01, 0.1])
        for function in (pw92_lda, xalpha_lda, wigner_lda, hedin_lundqvist_lda):
            with_core = function(valence, 0.7, core)
            total = function(valence + core, 0.7)
            np.testing.assert_array_equal(with_core.potential, total.potential)
            np.testing.assert_array_equal(with_core.energy_density, total.energy_density)
            # No density, no potential and no energy (exc_nspn.f90).
            self.assertEqual(with_core.potential[0], 0.0)
            self.assertEqual(with_core.energy_density[0], 0.0)
            with self.assertRaises(ValueError):
                function(np.array([-1.0e-3, 0.1]), 1.0)
        spin = pw92_lda_spin(0.5 * valence, 0.5 * valence, 0.7, core)
        np.testing.assert_allclose(spin.potential_up, pw92_lda(valence, 0.7, core).potential, rtol=1e-14)


class PW91KernelTests(unittest.TestCase):
    def test_energy_is_that_of_the_fortran_routines(self) -> None:
        energy, _, _ = pw91_energy_partials(RHO, SIGMA)
        np.testing.assert_allclose(energy, FORTRAN_PW91[0], rtol=1.0e-14)
        exchange, _, _ = _pw91_module._exchange_partials(RHO, SIGMA)
        np.testing.assert_allclose(exchange, FORTRAN_PW91[1], rtol=1.0e-14)
        energy = pw91_spin_energy_partials(UP, DOWN, *_spin_sigmas())[0]
        np.testing.assert_allclose(energy, FORTRAN_SPIN_PW91, rtol=1.0e-14)

    def test_exchange_is_libxc_with_its_constant(self) -> None:
        # libxc rounds a3 = 0.15084 of exchpw91 to 0.1508.
        with mock.patch.object(_pw91_module, "_A3", 0.1508):
            values = _pw91_module._exchange_partials(RHO, SIGMA)
        for column, (value, reference) in enumerate(zip(values, LIBXC["gga_x_pw91"])):
            np.testing.assert_allclose(value, reference, rtol=1.0e-12, err_msg=str(column))

    def test_close_to_libxc(self) -> None:
        # libxc's PW91 correlation uses C_x = -0.001667, nu = 16/pi (3 pi^2)^(1/3)
        # and the PW92 LDA with A = 0.031091; the Fortran -0.001667212,
        # 15.75592 and 0.0310907.  The gradient derivative of H1 is the most
        # sensitive to C_x.
        f, f_n, f_sigma = pw91_energy_partials(RHO, SIGMA)
        energy, potential, potential_sigma = _libxc(LIBXC, "gga_x_pw91", "gga_c_pw91")
        np.testing.assert_allclose(f, energy, rtol=1.0e-6)
        np.testing.assert_allclose(f_n, potential, rtol=1.0e-6)
        np.testing.assert_allclose(f_sigma, potential_sigma, rtol=1.0e-4)
        f, f_up, f_down, f_sigma_up, f_sigma_down, f_sigma = pw91_spin_energy_partials(
            UP, DOWN, *_spin_sigmas()
        )
        reference = _libxc(LIBXC_SPIN, "gga_x_pw91", "gga_c_pw91")
        np.testing.assert_allclose(f, reference[0], rtol=1.0e-6)
        np.testing.assert_allclose(f_up, reference[1], rtol=3.0e-5)
        np.testing.assert_allclose(f_down, reference[2], rtol=3.0e-5)
        np.testing.assert_allclose(f_sigma_up + f_sigma, reference[3], rtol=5.0e-4)
        np.testing.assert_allclose(2.0 * f_sigma, reference[4], rtol=5.0e-4)
        np.testing.assert_allclose(f_sigma_down + f_sigma, reference[5], rtol=5.0e-4)

    def test_partials_are_derivatives(self) -> None:
        f, f_n, f_sigma = pw91_energy_partials(RHO, SIGMA)
        np.testing.assert_allclose(
            f_n, _richardson(lambda n: pw91_energy_partials(n, SIGMA)[0], RHO), rtol=1.0e-9
        )
        np.testing.assert_allclose(
            f_sigma,
            _richardson(lambda s: pw91_energy_partials(RHO, s)[0], SIGMA),
            rtol=1.0e-8,
        )
        arguments = [UP, DOWN, *_spin_sigmas()]
        values = pw91_spin_energy_partials(*arguments)
        for index in range(5):

            def energy(x, index=index):
                changed = list(arguments)
                changed[index] = x
                return pw91_spin_energy_partials(*changed)[0]

            np.testing.assert_allclose(
                values[index + 1],
                _richardson(energy, arguments[index]),
                rtol=1.0e-8,
                err_msg=f"partial {index}",
            )

    def test_spin_limit(self) -> None:
        f, f_n, f_sigma = pw91_energy_partials(RHO, SIGMA)
        values = pw91_spin_energy_partials(0.5 * RHO, 0.5 * RHO, 0.25 * SIGMA, 0.25 * SIGMA, SIGMA)
        np.testing.assert_allclose(values[0], f, rtol=1.0e-14)
        np.testing.assert_allclose(values[1], f_n, rtol=1.0e-13)
        np.testing.assert_array_equal(values[1], values[2])
        # sigma_uu = sigma_dd = sigma/4 at a fixed spin ratio.
        np.testing.assert_allclose(0.5 * values[3] + values[5], f_sigma, rtol=1.0e-13)

    def test_vacuum_and_an_empty_channel(self) -> None:
        f, f_n, f_sigma = pw91_energy_partials(np.array([0.0, 1.0e-22]), np.array([0.0, 1.0e-40]))
        np.testing.assert_array_equal(f, 0.0)
        np.testing.assert_array_equal(f_n, 0.0)
        values = pw91_spin_energy_partials(
            np.array([0.3]), np.array([0.0]), np.array([0.1]), np.array([0.0]), np.array([0.1])
        )
        self.assertTrue(all(np.all(np.isfinite(value)) for value in values))
        self.assertEqual(values[4][0], 0.0)


class DiscreteGradientFunctionalTests(unittest.TestCase):
    """The PW91 potentials are the derivatives of the discrete energy."""

    def assert_variational(self, grid, energy, potential, density, direction):
        def central(epsilon):
            return (
                energy(density + epsilon * direction) - energy(density - epsilon * direction)
            ) / (2.0 * epsilon)

        # One Richardson step: the curvature along ``direction`` is large
        # next to the small first derivative of the periodic case.
        change = (4.0 * central(2.5e-6) - central(5.0e-6)) / 3.0
        self.assertAlmostEqual(
            change,
            grid.volume_element * float(np.dot(potential, direction)),
            delta=1.0e-8 * max(1.0, abs(change)),
        )

    def test_unpolarized_potential_on_cluster_and_periodic_grids(self) -> None:
        for grid in (_cluster_grid(), _periodic_grid()):
            up, down, core = _densities(grid)
            density = up + down
            direction = np.sin(grid.coordinates[:, 0]) * np.exp(-0.1 * np.sum(grid.coordinates**2, axis=1))
            result = pw91(density, grid, core)
            self.assert_variational(
                grid, lambda n: pw91(n, grid, core).total_energy, result.potential, density, direction
            )

    def test_spin_potentials_and_the_spin_limit(self) -> None:
        for grid in (_cluster_grid(), _periodic_grid()):
            up, down, core = _densities(grid)
            direction = np.cos(grid.coordinates[:, 1]) * np.exp(-0.1 * np.sum(grid.coordinates**2, axis=1))
            result = pw91_spin(up, down, grid, core)
            self.assert_variational(
                grid, lambda n: pw91_spin(n, down, grid, core).total_energy, result.potential_up, up, direction
            )
            self.assert_variational(
                grid, lambda n: pw91_spin(up, n, grid, core).total_energy, result.potential_down, down, direction
            )
            half = 0.5 * (up + down)
            spin = pw91_spin(half, half, grid, core)
            unpolarized = pw91(up + down, grid, core)
            np.testing.assert_allclose(spin.potential_up, unpolarized.potential, rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(spin.energy_density, unpolarized.energy_density, rtol=1e-12, atol=1e-16)


class AsymptoticCorrectionTests(unittest.TestCase):
    def test_gradient_is_the_fortran_stencil(self) -> None:
        """``density_gradient`` is gradmvs: centered weights, zero padding."""

        grid = _cluster_grid()
        up, _, _ = _densities(grid)
        width = grid.settings.expansion_order // 2
        # fornberg.f90, norder = 4.
        fortran = np.array([4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0])
        np.testing.assert_allclose(first_derivative_coefficients(8)[width + 1 :], fortran, rtol=1e-15)
        box = np.zeros(grid.shape)
        local = grid.integer_coordinates - grid.index_min
        box[tuple(local.T)] = up
        padded = np.pad(box, width)
        expected = np.zeros((3, grid.size))
        for axis in range(3):
            total = np.zeros(grid.shape)
            for shift, weight in enumerate(fortran, start=1):
                forward = np.roll(padded, -shift, axis=axis)
                backward = np.roll(padded, shift, axis=axis)
                inner = tuple(slice(width, -width) for _ in range(3))
                total += weight * (forward[inner] - backward[inner])
            expected[axis] = total[tuple(local.T)] / grid.spacing
        np.testing.assert_allclose(density_gradient(up, grid), expected, rtol=1e-12, atol=1e-15)

    def test_energy_is_ca_and_the_correction_follows_the_fortran_formula(self) -> None:
        grid = _cluster_grid()
        up, down, core = _densities(grid)
        density = up + down
        local = ca_lda(density, grid.volume_element, core)
        result = lb94(density, grid, core)
        np.testing.assert_array_equal(result.energy_density, local.energy_density)
        self.assertEqual(result.total_energy, local.total_energy)
        # exc_nspn.f90:296-338 on the total density with the core.
        rho = density + core
        gradient = np.sqrt(np.sum(density_gradient(rho, grid) ** 2, axis=0))
        ac1 = 2.0 ** (1.0 / 3.0)
        x = ac1 * gradient / rho ** (4.0 / 3.0)
        asinh = np.log(x + np.sqrt(x * x + 1.0))
        correction = 2.0 * 0.05 * rho ** (1.0 / 3.0) / ac1 * x * x / (1.0 + 3.0 * 0.05 * x * asinh)
        np.testing.assert_allclose(local.potential - result.potential, correction, rtol=1e-12)
        spin = lb94_spin(up, down, grid, core)
        spin_local = ca_lda_spin(up, down, grid.volume_element, core)
        np.testing.assert_array_equal(spin.energy_density, spin_local.energy_density)
        for channel, values, potential, potential_local in (
            ("up", up, spin.potential_up, spin_local.potential_up),
            ("down", down, spin.potential_down, spin_local.potential_down),
        ):
            rho_s = values + 0.5 * core
            g = np.sqrt(np.sum(density_gradient(rho_s, grid) ** 2, axis=0))
            x = g / rho_s ** (4.0 / 3.0)
            asinh = np.log(x + np.sqrt(x * x + 1.0))
            expected = 2.0 * 0.05 * rho_s ** (1.0 / 3.0) * x * x / (1.0 + 3.0 * 0.05 * x * asinh)
            np.testing.assert_allclose(potential_local - potential, expected, rtol=1e-12, err_msg=channel)
        # Equal channels: the unpolarized correction.
        half = 0.5 * density
        equal = lb94_spin(half, half, grid, core)
        np.testing.assert_allclose(equal.potential_up, result.potential, rtol=1e-13)

    def test_tail_of_an_exponential_density(self) -> None:
        """For rho = exp(-a r) the correction tends to 2/r Ry (-1/r Ha).

        ``x = a rho**(-1/3)`` grows exponentially and ``asinh x`` linearly,
        so ``c r = (2/3) a r / asinh(x)`` approaches 2 like ``1/r``.
        """

        r = np.array([40.0, 80.0, 160.0])
        a = 2.0
        rho = np.exp(-a * r)
        gradient = np.vstack([a * rho, np.zeros_like(r), np.zeros_like(r)])
        correction = _asymptotic_module._channel_correction(rho, gradient, 1.0, None)
        x = a * rho ** (-1.0 / 3.0)
        np.testing.assert_allclose(correction * r, (2.0 / 3.0) * a * r / np.arcsinh(x), rtol=1e-12)
        np.testing.assert_allclose(correction * r, 2.0, rtol=0.06)
        self.assertTrue(np.all(np.diff(np.abs(correction * r - 2.0)) < 0.0))

    def test_casida_salahub_cap(self) -> None:
        grid = _cluster_grid()
        up, down, core = _densities(grid)
        density = up + down
        local = ca_lda(density, grid.volume_element, core)
        lb = lb94(density, grid, core)
        correction = lb94_potential_correction(density + core, grid)
        cap = float(np.median(correction))
        capped = evaluate_unpolarized_xc(
            "casida_salahub", density, grid, core, ionization_energy_difference=cap
        )
        np.testing.assert_allclose(
            capped.potential, np.maximum(lb.potential, local.potential - cap), rtol=1e-14
        )
        self.assertEqual(capped.total_energy, local.total_energy)
        # Ion_Energy_Diff = 0 (the default) is the CA potential.
        zero = evaluate_unpolarized_xc("casida_salahub", density, grid, core)
        np.testing.assert_array_equal(zero.potential, local.potential)
        large = evaluate_unpolarized_xc(
            "casida_salahub", density, grid, core, ionization_energy_difference=10.0
        )
        np.testing.assert_array_equal(large.potential, lb.potential)
        spin = evaluate_spin_xc("casida_salahub", up, down, grid, core, ionization_energy_difference=cap)
        spin_lb = lb94_spin(up, down, grid, core)
        spin_local = ca_lda_spin(up, down, grid.volume_element, core)
        np.testing.assert_allclose(
            spin.potential_down,
            np.maximum(spin_lb.potential_down, spin_local.potential_down - cap),
            rtol=1e-14,
        )


class SelectionAndSettingsTests(unittest.TestCase):
    def test_dispatch(self) -> None:
        grid = _cluster_grid()
        up, down, core = _densities(grid)
        density = up + down
        for name, function in (
            ("pw92", lambda: pw92_lda(density, grid.volume_element, core)),
            ("xalpha", lambda: xalpha_lda(density, grid.volume_element, core)),
            ("wigner", lambda: wigner_lda(density, grid.volume_element, core)),
            ("hedin_lundqvist", lambda: hedin_lundqvist_lda(density, grid.volume_element, core)),
            ("pw91", lambda: pw91(density, grid, core)),
            ("lb94", lambda: lb94(density, grid, core)),
        ):
            np.testing.assert_array_equal(
                evaluate_unpolarized_xc(name, density, grid, core).potential, function().potential
            )
        for name in ("pw92", "pw91", "lb94", "casida_salahub"):
            result = evaluate_spin_xc(name, up, down, grid, core)
            self.assertTrue(np.all(np.isfinite(result.potential_up)))
        for name in ("xalpha", "wigner", "hedin_lundqvist"):
            with self.assertRaisesRegex(ValueError, "no spin-polarized form"):
                evaluate_spin_xc(name, up, down, grid, core)

    def test_settings(self) -> None:
        default = SCFSettings()
        self.assertNotIn("ionization_energy_difference", repr(default))
        self.assertEqual(
            repr(default),
            "SCFSettings(max_iterations=50, convergence_criterion=0.0002, "
            "fermi_temperature_kelvin=80.0, number_of_states=None, net_charge=0.0, "
            "use_plain_residual=False, normalize_initial_density=True, xc_functional='ca')",
        )
        settings = SCFSettings(xc_functional="casida_salahub", ionization_energy_difference=0.3)
        self.assertIn("ionization_energy_difference=0.3", repr(settings))
        with self.assertRaisesRegex(ValueError, "casida_salahub functional only"):
            SCFSettings(xc_functional="lb94", ionization_energy_difference=0.3)
        with self.assertRaisesRegex(ValueError, "no spin-polarized form"):
            SCFSettings(xc_functional="wigner", spin_polarized=True)
        with self.assertRaisesRegex(ValueError, "no spin-polarized form"):
            SCFSettings(xc_functional="xalpha", noncollinear=True)
        with self.assertRaisesRegex(ValueError, "xc_functional must be one of"):
            SCFSettings(xc_functional="pw")

    def test_pseudopotential_label(self) -> None:
        specification = {
            "O": SpeciesPotential(
                path=pseudopotential_file("tests/data/spin_ca_O_POTRE.DAT"),
                local_angular_momentum=1,
            )
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_pseudopotentials(specification, xc_functional="ca")
            self.assertEqual(caught, [])
            for functional in ("pw92", "pw91", "lb94", "xalpha"):
                load_pseudopotentials(specification, xc_functional=functional)
        self.assertEqual(len(caught), 4)
        self.assertTrue(all("does not match" in str(item.message) for item in caught))
        with self.assertRaisesRegex(ValueError, "not the requested PBE functional"):
            load_pseudopotentials(specification, xc_functional="pbe")


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.cases = load_xc_cases()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def parse(self, name: str, *extra: str, drop: tuple[str, ...] = ()):
        case = dict(self.cases[name])
        case["input"] = [line for line in case["input"] if not line.startswith(drop or ("\0",))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parse_parsec_input(write_xc_case(case, self.directory, extra=extra))

    def test_labels(self) -> None:
        expected = {
            "xa": "xalpha", "wi": "wigner", "hl": "hedin_lundqvist", "ca": "ca", "pz": "ca",
            "lda": "ca", "pl": "pw92", "pw92": "pw92", "PW-LDA": "pw92", "pw": "pw91",
            "PW.91": "pw91", "pwgga": "pw91", "pb": "pbe", "pbe": "pbe", "lb": "lb94",
            "cs": "casida_salahub",
        }
        for label, functional in expected.items():
            translation = self.parse(
                "h2o_ca", f"Correlation_Type: {label}", drop=("Correlation_Type",)
            )
            self.assertEqual(translation.problem.scf.xc_functional, functional, label)
        with self.assertRaisesRegex(ParsecInputError, "unknown"):
            self.parse("h2o_ca", "Correlation_Type: b3lyp", drop=("Correlation_Type",))

    def test_spin_polarization(self) -> None:
        self.assertEqual(self.parse("ch3_default").problem.scf.xc_functional, "pw92")
        for label in ("xa", "wi", "hl"):
            with self.assertRaisesRegex(UnsupportedParsecOptionError, "no spin-polarized form"):
                self.parse("o2_ca", f"Correlation_Type: {label}", drop=("Correlation_Type",))
            with self.assertRaisesRegex(UnsupportedParsecOptionError, "no spin-polarized form"):
                self.parse(
                    "h2o_ca",
                    f"Correlation_Type: {label}",
                    "Non_Collinear_magnetism: .true.",
                    drop=("Correlation_Type",),
                )

    def test_ion_energy_diff(self) -> None:
        scf = self.parse("h2o_cs3").problem.scf
        self.assertEqual(scf.xc_functional, "casida_salahub")
        self.assertEqual(scf.ionization_energy_difference, 0.3)
        scf = self.parse("h2o_cs3", "Ion_Energy_Diff: 1.0 eV", drop=("Ion_Energy_Diff",)).problem.scf
        self.assertAlmostEqual(scf.ionization_energy_difference, 1.0 / 13.605693122994, delta=1e-15)
        self.assertEqual(self.parse("h2o_cs0").problem.scf.ionization_energy_difference, 0.0)
        # Read only for Casida-Salahub, ignored otherwise as in Fortran, and
        # not an ionic-relaxation control.
        translation = self.parse("h2o_lb", "Ion_Energy_Diff: 0.3 Ry")
        self.assertEqual(translation.problem.scf.ionization_energy_difference, 0.0)
        self.assertFalse(any("fixed single point" in item for item in translation.warnings))
        with self.assertRaises(ParsecInputError):
            self.parse("h2o_lb", "Ion_Energy_Diff: 0.3 furlongs")

    def test_report(self) -> None:
        for name, expected in (
            ("h2o_cs3", (
                " Exchange-Correlation functional is cs",
                " ACLDA, Casida-Salahub correction",
                " Ionization energy difference is:  0.3000 Ry",
            )),
            ("h2o_pl", (" Exchange-Correlation functional is pl", " LDA, Perdew-Wang parametrization")),
            ("h2o_xa", (" LDA, Slater x-alpha", " performed x-alpha correlation with alpha =  1.000")),
            ("h2o_pw", (" Exchange-Correlation functional is pw", " GGA, Perdew-Wang parametrization")),
            ("h2o_ca", (
                " Exchange-Correlation functional is ca",
                " LDA, Ceperley-Alder, Perdew-Zunger parametrization",
            )),
        ):
            lines: list[str] = []
            ParsecTextReporter(lines.append, self.parse(name)).header()
            text = "\n".join(lines).splitlines()
            for line in expected:
                self.assertIn(line, text, name)


class PeriodicAndNoncollinearTests(unittest.TestCase):
    """The periodic SCF and the non-collinear wrapper take every functional."""

    def test_hydrogen_molecule_in_a_box(self) -> None:
        hydrogen = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"
        totals = {}
        for functional in ("ca", "pw92", "pw91", "lb94", "casida_salahub", "xalpha"):
            extra = {"ionization_energy_difference": 0.2} if functional == "casida_salahub" else {}
            problem = SinglePointInput(
                atoms=[Atom("H", [3.0, 3.0, 3.0]), Atom("H", [3.0, 3.0, 4.4])],
                pseudopotentials={"H": SpeciesPotential(hydrogen, 0)},
                grid=PeriodicGridSettings(spacing=0.6, expansion_order=4),
                periodic_cell=PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
                scf=SCFSettings(
                    max_iterations=40,
                    number_of_states=3,
                    convergence_criterion=1.0e-5,
                    xc_functional=functional,
                    **extra,
                ),
                hartree=HartreeSettings(
                    relative_tolerance=1.0e-8, absolute_tolerance=1.0e-12, max_iterations=2000
                ),
                eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                system = prepare_periodic_single_point(problem)
            result = run_scf(system)
            self.assertTrue(result.converged, functional)
            totals[functional] = result.energies.total
            if functional in UNPOLARIZED_ONLY_FUNCTIONALS:
                continue
            half = 0.5 * result.density
            spin = system.evaluate_xc_spin_polarized(half, half)
            unpolarized = system.evaluate_xc(result.density)
            np.testing.assert_allclose(spin.potential_up, unpolarized.potential, rtol=1e-12, atol=1e-14)
            noncollinear = system.evaluate_xc_noncollinear(
                result.density, np.zeros((3, result.density.size))
            )
            np.testing.assert_allclose(noncollinear.potential, unpolarized.potential, rtol=1e-12, atol=1e-14)
        # Two LDAs, and CA with two potentials of its own.
        self.assertAlmostEqual(totals["pw92"], totals["ca"], delta=5.0e-3)
        self.assertAlmostEqual(totals["lb94"], totals["ca"], delta=1.0e-2)


class FortranComparisonTests(unittest.TestCase):
    """Complete calculations on the readable solver against Fortran PARSEC."""

    def solve(self, name: str):
        case = load_xc_cases()[name]
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            translation = parse_parsec_input(write_xc_case(case, Path(directory)))
            system = prepare_single_point(translation.problem)
        return case, run_scf(system)

    def test_perdew_wang_lda(self) -> None:
        case, result = self.solve("h2o_pl")
        assert_matches_fortran_case(self, result, case)

    def test_casida_salahub(self) -> None:
        case, result = self.solve("h2o_cs3")
        assert_matches_fortran_case(self, result, case)
        # The same energy expression as CA, with another potential.
        self.assertNotAlmostEqual(result.energies.total, load_cases()["h2o_nospin_ca"]["energy"], delta=1e-4)


if __name__ == "__main__":
    unittest.main()
