"""Periodic local-pseudopotential forces: finite differences of the energies they differentiate."""

from __future__ import annotations

import dataclasses
import unittest
from pathlib import Path

import numpy as np

from parsec_python import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.Forces.periodic_local import periodic_local_forces
from parsec_python.Grid import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import ewald, superpose_atomic_density
from parsec_python.V_ion.ewald import ewald_local_ionic_potential

_DATA = Path(__file__).resolve().parents[1] / "data"
_GE = _DATA / "Ge_POTRE.DAT"  # has a nonlinear core correction
_NA = _DATA / "Na_POTRE.DAT"  # none: the full table (119 bohr) is cheap for the local potential alone
_CELLS = {
    "rectangular": np.diag([7.7, 8.4, 9.1]),
    "fcc": 6.6 * (np.ones((3, 3)) - np.eye(3)),
}
_FRACTIONAL = np.array([[0.13, 0.27, 0.31], [0.62, 0.45, 0.78]])


def _table(path: Path, cut: float | None = None):
    """The Ge table, optionally cut at ``cut`` bohr (its core density is 1e-14 at 10 bohr).

    A cut table ends in a spline that leaves the Coulomb tail over its last knots, where the force has already switched
    to the Coulomb slope.  That is harmless for a full table (119 bohr, far beyond the images of the local potential)
    but not for the local potential of a cut one, so only the core-correction test, which needs no local potential,
    cuts the table (to keep its image sum small).
    """
    full = read_parsec_pseudopotential(path)
    if cut is None:
        return full
    keep = int(np.searchsorted(full.radii, cut, side="right"))
    take = lambda values: values[:keep]  # noqa: E731
    return dataclasses.replace(
        full,
        radii=take(full.radii),
        channel_potentials={l: take(v) for l, v in full.channel_potentials.items()},
        valence_density=take(full.valence_density),
        core_density=take(full.core_density),
        radial_wavefunctions={l: take(v) for l, v in full.radial_wavefunctions.items()},
        spin_orbit_channel_potentials={l: take(v) for l, v in full.spin_orbit_channel_potentials.items()},
    )


class _Case:
    """A two-atom cell with smooth, arbitrary density and XC fields (the forces are linear in them)."""

    def __init__(self, name: str, use_spline: bool, *, core: bool = False) -> None:
        # the core-correction terms need a table cut short (their image sum reaches the table's end)
        symbol, path, cut = ("Ge", _GE, 14.0) if core else ("Na", _NA, None)
        self.symbol = symbol
        self.lattice = _CELLS[name]
        self.grid = build_periodic_grid(
            PeriodicCell(self.lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6)
        )
        self.potentials = {symbol: _table(path, cut)}
        self.specifications = {symbol: SpeciesPotential(path, 0, use_spline=use_spline)}
        self.positions = _FRACTIONAL @ self.lattice
        fractional = self.grid.coordinates @ np.linalg.inv(self.lattice)
        phase = 2.0 * np.pi * fractional
        self.density = 0.3 + 0.2 * np.cos(phase[:, 0] + 0.4) * np.sin(phase[:, 1]) + 0.1 * np.sin(phase[:, 2] - phase[:, 0])
        self.xc = -0.5 + 0.3 * np.sin(phase[:, 0]) * np.cos(phase[:, 2] + 0.3) + 0.2 * np.cos(phase[:, 1] - 2 * phase[:, 2])

    def atoms(self, positions: np.ndarray | None = None) -> list[Atom]:
        positions = self.positions if positions is None else positions
        return [Atom(self.symbol, np.array(position)) for position in positions]

    def forces(self, positions: np.ndarray | None = None, **kwargs):
        return periodic_local_forces(
            self.grid, self.atoms(positions), self.potentials, self.specifications, self.lattice,
            self.density, self.xc, **kwargs,
        )

    def ionic_energy(self, positions: np.ndarray) -> float:
        potential = ewald_local_ionic_potential(
            self.grid, self.atoms(positions), self.potentials, self.specifications, self.lattice
        )
        return float(self.density @ potential) * self.grid.volume_element

    def core_energy(self, positions: np.ndarray) -> float:
        core = superpose_atomic_density(
            self.grid, self.atoms(positions), self.potentials, self.specifications, self.lattice, core=True
        )
        return float(self.xc @ core) * self.grid.volume_element

    def finite_difference(self, energy, atom: int, axis: int, step: float) -> float:
        plus, minus = self.positions.copy(), self.positions.copy()
        plus[atom, axis] += step
        minus[atom, axis] -= step
        return -(energy(plus) - energy(minus)) / (2.0 * step)


class PeriodicLocalForceTests(unittest.TestCase):
    def setUp(self) -> None:
        # the dense reference potential: another test may have installed the FFT provider of the acceleration package
        self.addCleanup(setattr, ewald, "_local_potential_provider", ewald._local_potential_provider)
        ewald.set_local_potential_provider(None)
        # the periodic local potential always uses the cubic spline of its table (the default); the tests below that
        # compare the linear mode switch it off themselves
        self.addCleanup(setattr, ewald, "LOCAL_POTENTIAL_USES_CUBIC_SPLINE", ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE)
        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = True

    def test_ionic_force_is_the_derivative_of_the_electron_ion_energy(self) -> None:
        for name in _CELLS:
            # with the spline the force is the exact derivative; PARSEC's linear mode differentiates a log-grid table
            # instead of the piecewise-linear interpolant (a few percent of the largest component)
            for use_spline, relative in ((True, 1.0e-5), (False, 3.0e-2)):
                with self.subTest(cell=name, spline=use_spline):
                    ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = use_spline
                    case = _Case(name, use_spline)
                    ionic = case.forces().ionic
                    scale = float(np.max(np.abs(ionic)))
                    self.assertGreater(scale, 1.0e-2)
                    for atom, axis in ((0, 0), (0, 2), (1, 1), (1, 2)):
                        numeric = case.finite_difference(case.ionic_energy, atom, axis, 2.0e-3)
                        self.assertAlmostEqual(float(ionic[atom, axis]), numeric, delta=relative * scale)

    def test_core_correction_force_is_the_derivative_of_the_core_energy(self) -> None:
        for name in _CELLS:
            # with the spline the force is the exact derivative; PARSEC's linear mode differentiates a log-grid table
            # instead of the piecewise-linear interpolant, which differs by about 1 % of the force
            for use_spline, relative in ((True, 1.0e-5), (False, 3.0e-2)):
                with self.subTest(cell=name, spline=use_spline):
                    case = _Case(name, use_spline, core=True)
                    core = case.forces().core_correction
                    self.assertGreater(float(np.max(np.abs(core))), 1.0e-3)
                    for atom, axis in ((0, 1), (0, 2), (1, 0), (1, 2)):
                        numeric = case.finite_difference(case.core_energy, atom, axis, 2.0e-3)
                        self.assertAlmostEqual(float(core[atom, axis]), numeric, delta=relative * float(np.max(np.abs(core))))

    def test_ionic_force_does_not_depend_on_the_splitting_parameter(self) -> None:
        for name in _CELLS:
            with self.subTest(cell=name):
                case = _Case(name, True)
                default = case.forces().ionic
                for eta in (0.3, 0.55):
                    np.testing.assert_allclose(case.forces(eta=eta).ionic, default, atol=2.0e-7)

    def test_an_atom_moved_by_a_lattice_vector_keeps_its_forces(self) -> None:
        case = _Case("fcc", True, core=True)
        moved = case.positions.copy()
        moved[1] += case.lattice[0] - case.lattice[2]
        # the cell is no longer the one of the grid for the atom, but periodic images make it the same crystal
        reference, shifted = case.forces(), case.forces(moved)
        np.testing.assert_allclose(shifted.ionic, reference.ionic, atol=1.0e-8)
        np.testing.assert_allclose(shifted.core_correction, reference.core_correction, atol=1.0e-8)

    def test_cubic_spline_does_not_change_the_periodic_local_potential(self) -> None:
        # Fortran's periodic local potential (ionpbc.f90, from the form factors of fourier_f.f90) never reads Cubic_Spline
        from parsec_python.acceleration.V_ion import periodic_setup

        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = True  # the default
        linear, spline = _Case("fcc", False), _Case("fcc", True)
        args = lambda case: (case.grid, case.atoms(), case.potentials, case.specifications, case.lattice)  # noqa: E731
        reference = ewald_local_ionic_potential(*args(spline))
        np.testing.assert_array_equal(ewald_local_ionic_potential(*args(linear)), reference)
        # the FFT provider of the acceleration package is the same potential in either mode, to the 1e-8 Ry by which the
        # spline of the table differs from the exact Coulomb tail that the provider adds beyond the Coulomb onset
        for case in (linear, spline):
            fast = periodic_setup.fft_local_ionic_potential(*args(case), None, 1.0e-14)
            np.testing.assert_allclose(fast, reference, atol=1.0e-6)

    def test_the_linear_table_interpolation_is_not_what_the_periodic_potential_uses(self) -> None:
        # linear interpolation of the logarithmic table (1 % steps) is off by about 2e-5 of the Coulomb-like potential,
        # +0.7 mRy in the Ge3 total energy against Fortran's form factors, which integrate the table to high order
        spline, linear = _Case("fcc", True), _Case("fcc", True)
        args = lambda case: (case.grid, case.atoms(), case.potentials, case.specifications, case.lattice)  # noqa: E731
        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = False
        biased = ewald_local_ionic_potential(*args(linear))
        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = True
        accurate = ewald_local_ionic_potential(*args(spline))
        # 2e-4 Ry at the largest (near an atom, where the table is stiffest) on this grid, 8e-6 on average
        largest = float(np.max(np.abs(biased - accurate)))
        self.assertGreater(largest, 1.0e-5)
        self.assertLess(largest, 1.0e-2)

    def test_the_self_consistency_term_is_absent(self) -> None:
        terms = _Case("rectangular", False).forces()
        np.testing.assert_array_equal(terms.self_consistency, 0.0)
        np.testing.assert_allclose(terms.total, terms.ionic + terms.core_correction)


if __name__ == "__main__":
    unittest.main()
