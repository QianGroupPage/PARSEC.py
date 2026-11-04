"""The functionals added for parity with Fortran PARSEC on the accelerated backends.

The accelerated driver evaluates every functional but the native CA through
the readable :mod:`parsec_python.V_xc` package (on the full grid, expanding
a symmetry-reduced density first); the spin-polarized SCF loop is that of
:mod:`parsec_python.SCF.spin_polarized`.  Complete calculations of
``tests/data/xc_parsec_reference.json`` are compared with Fortran PARSEC:
the local functionals and the asymptotic corrections to the digits Fortran
prints, PW91 to the agreement of PBE (the potential here is the derivative
of the discrete energy, the Fortran one the pointwise EASYPBE formula).
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.tests.xc_reference import (
    assert_matches_fortran_case,
    load_xc_cases,
    result_levels,
    write_xc_case,
)
from parsec_python.V_xc import wigner_lda, xalpha_lda


def _solve(name: str, backend: str):
    case = load_xc_cases()[name]
    with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
        # The CA files of the fixture warn with every other functional.
        warnings.simplefilter("ignore")
        translation = parse_parsec_input(write_xc_case(case, Path(directory)))
        system = driver.prepare_single_point(translation.problem, backend=backend)
    return case, system, driver.run_scf(system)


class _BackendCases:
    backend: str

    def assert_case(self, name: str, **tolerances):
        case, system, result = _solve(name, self.backend)
        self.assertEqual(result.backend.selected, self.backend)
        assert_matches_fortran_case(self, result, case, **tolerances)
        return case, system, result

    def test_x_alpha_and_hedin_lundqvist(self) -> None:
        self.assert_case("h2o_xa")
        self.assert_case("h2o_hl")

    def test_wigner(self) -> None:
        # Without symmetry the Fortran energy is that of the functional.
        self.assert_case("h2o_wi_nosym")
        # With symmetry (order 4) Fortran counts a quarter of the Wigner
        # correlation energy (exc_nspn.f90:225); the potential is right.
        case, system, result = _solve("h2o_wi", self.backend)
        assert_matches_fortran_case(
            self,
            result,
            case,
            energy_tolerance=np.inf,
            terms=("eigenvalue", "hartree", "integral_vxc_rho", "electron_ion"),
        )
        density = np.asarray(result.density)
        grid = system.grid
        exchange = (2.0 / 3.0) * xalpha_lda(density, grid.volume_element, system.core_density).total_energy
        correlation = wigner_lda(density, grid.volume_element, system.core_density).total_energy - exchange
        self.assertAlmostEqual(
            result.energies.exchange_correlation - 0.75 * correlation,
            case["energies"]["exchange_correlation"],
            delta=2.0e-7,
        )

    def test_casida_salahub(self) -> None:
        self.assert_case("h2o_cs3")

    def test_spin_polarized_local_functionals(self) -> None:
        self.assert_case("ch3_default")
        self.assert_case("o2_lb")
        self.assert_case("o2_cs3")

    def test_pw91(self) -> None:
        # PBE of this package differs from Fortran's PBE on these inputs by
        # 3.5e-4 Ry (water) and 1.1e-3 Ry (O2), its levels by 3e-3 and
        # 6e-3 Ry; PW91 by as much.  The closed shell is compared with the
        # spin-polarized Fortran run: the unpolarized Fortran potential has
        # a factor two too many in its reduced gradient u (pbe.f90:413).
        for name, energy, level in (("h2o_spin_pw", 5.0e-4, 4.0e-3), ("o2_pw", 1.5e-3, 1.0e-2)):
            case, _, result = _solve(name, self.backend)
            self.assertTrue(result.converged)
            self.assertLess(result.energies.total, case["energy"])
            self.assertAlmostEqual(result.energies.total, case["energy"], delta=energy)
            levels = result_levels(result)
            for channel, reference in case["eigenvalues"].items():
                occupied = np.asarray(case["occupations"][channel]) > 1.0e-3
                np.testing.assert_allclose(
                    levels[channel][occupied], np.asarray(reference)[occupied], rtol=0, atol=level
                )
        unpolarized = load_xc_cases()["h2o_pw"]
        _, _, result = _solve("h2o_pw", self.backend)
        self.assertAlmostEqual(
            result.energies.total, load_xc_cases()["h2o_spin_pw"]["energy"], delta=5.0e-4
        )
        self.assertGreater(abs(result.energies.total - unpolarized["energy"]), 5.0e-3)


@unittest.skipUnless(native_available(), "native extension is unavailable")
class NativeBackendTests(_BackendCases, unittest.TestCase):
    backend = "native"


@unittest.skipUnless(
    cupy_available() and native_available(),
    "CUDA runtime or native extension is unavailable",
)
class CuPyBackendTests(_BackendCases, unittest.TestCase):
    backend = "cupy"


if __name__ == "__main__":
    unittest.main()
