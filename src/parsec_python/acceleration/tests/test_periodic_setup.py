"""The fast periodic preparation reproduces the readable definitions.

``grid_ball`` must return exactly what the dense scan of every grid point returns, so projectors and
atomic densities built through it are identical; the FFT local ionic potential must equal the direct
Ewald sum once that sum is converged (the sum at the default tolerance stops at the radius where
the images it leaves out are still of order 1e-6 Ry).
"""

from __future__ import annotations

import functools
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.acceleration.V_ion import periodic_setup
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion import (
    build_nonlocal_projectors,
    superpose_atomic_density,
)
from parsec_python.V_ion.ewald import ewald_local_ionic_potential
from parsec_python.V_ion.ionic_potential import dense_support

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
_SIDES = (6.0, 7.2, 8.4)  # spacing 0.6 on all three axes
_FCC = 5.4 * (np.ones((3, 3)) - np.eye(3))  # oblique: 60 degrees between the lattice vectors, 12 points per axis
_FRACTIONAL = np.array([[0.1, 0.2, 0.3], [0.55, 0.4, 0.8], [0.3, 0.9, 0.5]])


def _lattice(oblique: bool) -> np.ndarray:
    return _FCC if oblique else np.diag(_SIDES)


_SHIFT = (0.5, 0.3, 0.8)  # grid points off the origin by a fraction of a step along each lattice vector


@functools.lru_cache(maxsize=None)
def _system(oblique: bool = False, shifted: bool = False):
    if oblique:
        positions = _FRACTIONAL @ _FCC
    else:
        positions = np.array([[1.3, 5.9, 0.2], [5.7, 0.4, 7.9], [3.1, 3.3, 4.6]])
    problem = SinglePointInput(
        atoms=[
            Atom("Au", positions[0].tolist(), [0.0, 0.0, 0.0]),
            Atom("H", positions[1].tolist(), [0.0, 0.0, 0.0]),
            Atom("H", positions[2].tolist(), [0.0, 0.0, 0.0]),
        ],
        pseudopotentials={
            "Au": SpeciesPotential(_EXAMPLES / "Au_POTRE.DAT", 0),
            "H": SpeciesPotential(_EXAMPLES / "H_POTRE.DAT", 0),
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6, shift=_SHIFT if shifted else (0.0, 0.0, 0.0)),
        periodic_cell=PeriodicCell(lattice_vectors=_lattice(oblique)),
        scf=SCFSettings(max_iterations=5, number_of_states=8),
        hartree=HartreeSettings(multipole_order=4),
    )
    periodic_setup.uninstall()
    try:
        return problem, prepare_periodic_single_point(problem)
    finally:
        periodic_setup.install()


class GridBallTests(unittest.TestCase):
    def test_equals_the_dense_scan_inside_and_outside_the_cell(self) -> None:
        for oblique, shifted in ((False, False), (True, False), (False, True), (True, True)):
            _, system = _system(oblique, shifted)
            generator = np.random.default_rng(3)
            for _ in range(60):
                centre = generator.uniform(-9.0, 17.0, size=3)
                radius = float(generator.uniform(0.2, 9.0))
                with self.subTest(oblique=oblique, shifted=shifted):
                    rows, relative, distances = periodic_setup.grid_ball(system.grid, centre, radius)
                    expected = dense_support(system.grid, centre, radius)
                    np.testing.assert_array_equal(rows, expected[0])
                    np.testing.assert_array_equal(relative, expected[1])
                    np.testing.assert_array_equal(distances, expected[2])

    def test_other_grids_use_the_dense_scan(self) -> None:
        class Plain:
            coordinates = np.random.default_rng(1).uniform(0, 5, size=(40, 3))

        rows, _, _ = periodic_setup.grid_ball(Plain, np.array([2.0, 2.0, 2.0]), 2.0)
        np.testing.assert_array_equal(rows, dense_support(Plain, np.array([2.0, 2.0, 2.0]), 2.0)[0])


class SupportBuildersTests(unittest.TestCase):
    oblique = False
    shifted = False

    def setUp(self) -> None:
        self.problem, self.system = _system(self.oblique, self.shifted)
        self.addCleanup(periodic_setup.install)

    def _build(self, fast: bool, k_point=None):
        (periodic_setup.install if fast else periodic_setup.uninstall)()
        return build_nonlocal_projectors(
            self.system.grid,
            self.system.atoms,
            self.system.pseudopotentials,
            self.problem.pseudopotentials,
            _lattice(self.oblique),
            k_point=k_point,
        )

    def test_projectors_are_identical(self) -> None:
        for k_point in (None, np.array([0.21, -0.37, 0.11])):
            with self.subTest(k_point=k_point):
                dense, fast = self._build(False, k_point), self._build(True, k_point)
                difference = abs(dense.projectors - fast.projectors)
                self.assertEqual(dense.projectors.nnz, fast.projectors.nnz)
                self.assertEqual(difference.max() if difference.nnz else 0.0, 0.0)

    def test_atomic_densities_are_identical(self) -> None:
        for core in (False, True):
            with self.subTest(core=core):
                results = []
                for fast in (False, True):
                    (periodic_setup.install if fast else periodic_setup.uninstall)()
                    results.append(
                        superpose_atomic_density(
                            self.system.grid,
                            self.system.atoms,
                            self.system.pseudopotentials,
                            self.problem.pseudopotentials,
                            _lattice(self.oblique),
                            core=core,
                        )
                    )
                np.testing.assert_array_equal(results[0], results[1])
                if not core:
                    self.assertGreater(results[0].max(), 0.0)


class ObliqueSupportBuildersTests(SupportBuildersTests):
    """The same builders in an oblique cell (fcc lattice vectors)."""

    oblique = True


class ShiftedSupportBuildersTests(SupportBuildersTests):
    """The same builders on a grid whose points are off the origin by a fraction of a step."""

    shifted = True


class ShiftedObliqueSupportBuildersTests(SupportBuildersTests):
    oblique = True
    shifted = True


_CONVERGED: dict[tuple[bool, bool], np.ndarray] = {}


def _converged_direct_sum(arguments, oblique: bool = False, shifted: bool = False) -> np.ndarray:
    if (oblique, shifted) not in _CONVERGED:
        _CONVERGED[(oblique, shifted)] = ewald_local_ionic_potential(*arguments, tolerance=1.0e-26)
    return _CONVERGED[(oblique, shifted)]


class FFTLocalPotentialTests(unittest.TestCase):
    oblique = False
    shifted = False

    def setUp(self) -> None:
        self.problem, self.system = _system(self.oblique, self.shifted)
        self.addCleanup(periodic_setup.install)
        self.arguments = (
            self.system.grid,
            self.system.atoms,
            self.system.pseudopotentials,
            self.problem.pseudopotentials,
            _lattice(self.oblique),
        )

    def test_matches_the_converged_direct_sum(self) -> None:
        periodic_setup.uninstall()
        converged = _converged_direct_sum(self.arguments, self.oblique, self.shifted)
        fast = periodic_setup.fft_local_ionic_potential(
            *self.arguments, None, 1.0e-14
        )
        np.testing.assert_allclose(fast, converged, atol=1.0e-9)

    def test_stays_within_the_accuracy_of_the_default_direct_sum(self) -> None:
        periodic_setup.uninstall()
        direct = ewald_local_ionic_potential(*self.arguments)
        periodic_setup.install()
        provided = ewald_local_ionic_potential(*self.arguments)
        self.assertLess(np.max(np.abs(provided - direct)), 1.0e-4)
        # the constant of the G = 0 term is the reference's: the means agree
        self.assertAlmostEqual(float(provided.mean()), float(direct.mean()), delta=1.0e-5)


class ObliqueFFTLocalPotentialTests(FFTLocalPotentialTests):
    """The Ewald split with the reciprocal series on the oblique reciprocal lattice."""

    oblique = True


class ShiftedFFTLocalPotentialTests(FFTLocalPotentialTests):
    """The reciprocal series on points ``(i + s)/n`` of the lattice vectors."""

    shifted = True


class ShiftedObliqueFFTLocalPotentialTests(FFTLocalPotentialTests):
    oblique = True
    shifted = True


if __name__ == "__main__":
    unittest.main()
