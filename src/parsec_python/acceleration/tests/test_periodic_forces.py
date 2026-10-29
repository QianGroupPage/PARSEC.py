"""Forces of a periodic cell on the accelerated path (CPU backend; the sums are written against an array module).

The fast local forces (ball sums and one FFT) must equal the dense reference
:func:`parsec_python.Forces.periodic_local.periodic_local_forces`, on rectangular and oblique cells and for grids off the
origin (the grid shift is a phase of the Fourier transform of the density).  The accelerated system must give the forces of
the reference system, keep them when its atoms move, and refuse symmetrization, which a crystal does not have.
"""

from __future__ import annotations

import dataclasses
import types
import unittest
from pathlib import Path

import numpy as np
from scipy.special import erf

from parsec_python import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.acceleration import driver
from parsec_python.acceleration.Forces.periodic_local import fast_periodic_local_forces
from parsec_python.Forces import compute_periodic_forces
from parsec_python.Forces.periodic_local import periodic_local_forces
from parsec_python.Grid import build_periodic_grid
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.Relaxation import run_relaxation
from parsec_python.SCF import prepare_periodic_single_point, run_scf
from parsec_python.V_ion import ewald
from parsec_python.models import RelaxationSettings

_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_NA, _GE = _DATA / "Na_POTRE.DAT", _DATA / "Ge_POTRE.DAT"
_CELLS = {"rectangular": np.diag([7.7, 8.4, 9.1]), "fcc": 6.6 * (np.ones((3, 3)) - np.eye(3))}
_SHIFTS = {"origin": (0.0, 0.0, 0.0), "half": (0.5, 0.5, 0.5), "odd": (0.5, 0.3, 0.8)}
_FRACTIONAL = np.array([[0.13, 0.27, 0.31], [0.62, 0.45, 0.78]])


def _fields(grid, lattice):
    """Smooth, arbitrary density and XC fields (the forces are linear in them)."""
    phase = 2.0 * np.pi * (grid.coordinates @ np.linalg.inv(lattice))
    density = 0.3 + 0.2 * np.cos(phase[:, 0] + 0.4) * np.sin(phase[:, 1]) + 0.1 * np.sin(phase[:, 2] - phase[:, 0])
    xc = -0.5 + 0.3 * np.sin(phase[:, 0]) * np.cos(phase[:, 2] + 0.3) + 0.2 * np.cos(phase[:, 1] - 2 * phase[:, 2])
    return density, xc


class FastPeriodicLocalForceTests(unittest.TestCase):
    def setUp(self) -> None:
        # the dense reference potential: the fast one is installed when the acceleration package is imported
        self.addCleanup(setattr, ewald, "_local_potential_provider", ewald._local_potential_provider)
        ewald.set_local_potential_provider(None)

    def _compare(self, cell: str, shift: str, symbol: str, path: Path, core: bool, use_spline: bool = False) -> None:
        potential = read_parsec_pseudopotential(path)
        if core:  # a table cut at 14 bohr keeps the image sum of the dense core term small
            keep = int(np.searchsorted(potential.radii, 14.0, side="right"))
            cut = lambda values: values[:keep]  # noqa: E731
            potential = dataclasses.replace(
                potential,
                radii=cut(potential.radii),
                channel_potentials={l: cut(v) for l, v in potential.channel_potentials.items()},
                valence_density=cut(potential.valence_density),
                core_density=cut(potential.core_density),
                radial_wavefunctions={l: cut(v) for l, v in potential.radial_wavefunctions.items()},
                spin_orbit_channel_potentials={l: cut(v) for l, v in potential.spin_orbit_channel_potentials.items()},
            )
        lattice = _CELLS[cell]
        grid = build_periodic_grid(
            PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6, shift=_SHIFTS[shift])
        )
        density, xc = _fields(grid, lattice)
        atoms = [Atom(symbol, position) for position in _FRACTIONAL @ lattice]
        arguments = (
            grid, atoms, {symbol: potential}, {symbol: SpeciesPotential(path, 0 if symbol == "Na" else 1, use_spline=use_spline)},
            lattice, density, xc,
        )
        reference, fast = periodic_local_forces(*arguments), fast_periodic_local_forces(*arguments)
        scale = max(float(np.max(np.abs(reference.ionic))), 1.0e-3)
        # Na: the tail of its 119-bohr table is a spline of 1/r, off by about 1e-8 per image
        np.testing.assert_allclose(fast.ionic, reference.ionic, atol=2.0e-7)
        np.testing.assert_allclose(fast.core_correction, reference.core_correction, atol=1.0e-9)
        np.testing.assert_array_equal(fast.self_consistency, 0.0)
        self.assertGreater(scale, 1.0e-2)
        if core:
            self.assertGreater(float(np.max(np.abs(reference.core_correction))), 1.0e-2)

    def test_ionic_force_equals_the_dense_reference(self) -> None:
        for cell in _CELLS:
            for shift in _SHIFTS:
                with self.subTest(cell=cell, shift=shift):
                    self._compare(cell, shift, "Na", _NA, core=False)

    def test_core_correction_and_ionic_force_of_a_species_with_a_core(self) -> None:
        for cell, shift in (("rectangular", "half"), ("fcc", "odd"), ("fcc", "origin")):
            for use_spline in (False, True):  # the core density follows Cubic_Spline
                with self.subTest(cell=cell, shift=shift, spline=use_spline):
                    self._compare(cell, shift, "Ge", _GE, core=True, use_spline=use_spline)

    def test_the_linear_mode_of_the_local_potential_runs_the_reference(self) -> None:
        self.addCleanup(setattr, ewald, "LOCAL_POTENTIAL_USES_CUBIC_SPLINE", ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE)
        ewald.LOCAL_POTENTIAL_USES_CUBIC_SPLINE = False
        potential = read_parsec_pseudopotential(_NA)
        lattice = _CELLS["rectangular"]
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6))
        density, xc = _fields(grid, lattice)
        atoms = [Atom("Na", position) for position in _FRACTIONAL @ lattice]
        arguments = (grid, atoms, {"Na": potential}, {"Na": SpeciesPotential(_NA, 0)}, lattice, density, xc)
        np.testing.assert_array_equal(
            fast_periodic_local_forces(*arguments).ionic, periodic_local_forces(*arguments).ionic
        )

    def test_the_sums_use_only_operations_that_cupy_has(self) -> None:
        # CuPy cannot be run here: the same routine runs on a module that offers nothing but the array functions of the
        # sums (all of them in CuPy, with the same meaning) and fails on any other
        allowed = {
            "asarray", "zeros_like", "searchsorted", "clip", "interp", "where", "exp", "sum", "stack", "matmul",
            "einsum", "float64", "complex128",
        }

        class Restricted:
            fft = types.SimpleNamespace(ifftn=np.fft.ifftn)

            def __getattr__(self, name):
                if name not in allowed:
                    raise AttributeError(f"{name} is not on the list of CuPy operations of the force sums")
                return getattr(np, name)

        for core, symbol, path, spline in ((False, "Na", _NA, False), (True, "Ge", _GE, True), (True, "Ge", _GE, False)):
            potential = read_parsec_pseudopotential(path)
            lattice = _CELLS["fcc"]
            grid = build_periodic_grid(
                PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6, shift=_SHIFTS["odd"])
            )
            density, xc = _fields(grid, lattice)
            atoms = [Atom(symbol, position) for position in _FRACTIONAL @ lattice]
            specification = SpeciesPotential(path, 0 if symbol == "Na" else 1, use_spline=spline)
            arguments = (grid, atoms, {symbol: potential}, {symbol: specification}, lattice, density, xc)
            with self.subTest(species=symbol, spline=spline):
                restricted = fast_periodic_local_forces(*arguments, xp=Restricted(), erf=erf)
                plain = fast_periodic_local_forces(*arguments)
                np.testing.assert_allclose(restricted.ionic, plain.ionic, atol=1.0e-12)
                np.testing.assert_allclose(restricted.core_correction, plain.core_correction, atol=1.0e-12)

    def test_an_array_module_other_than_numpy_needs_its_erf(self) -> None:
        potential = read_parsec_pseudopotential(_NA)
        lattice = _CELLS["rectangular"]
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.7, expansion_order=6))
        density, xc = _fields(grid, lattice)
        atoms = [Atom("Na", position) for position in _FRACTIONAL @ lattice]
        with self.assertRaises(ValueError):
            fast_periodic_local_forces(
                grid, atoms, {"Na": potential}, {"Na": SpeciesPotential(_NA, 0)}, lattice, density, xc, xp=object()
            )


def _problem() -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("Ge", np.array([2.4, 3.1, 3.9])), Atom("Ge", np.array([5.0, 4.4, 3.3]))],
        pseudopotentials={"Ge": SpeciesPotential(_GE, 1, use_spline=True)},
        grid=PeriodicGridSettings(spacing=0.7, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=7.7 * np.eye(3)),
        scf=SCFSettings(max_iterations=80, number_of_states=6, convergence_criterion=1.0e-9, fermi_temperature_kelvin=300.0),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-9),
    )


class AcceleratedPeriodicSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = _problem()
        cls.reference_system = prepare_periodic_single_point(cls.problem)
        cls.reference = run_scf(cls.reference_system)
        cls.system = driver.prepare_single_point(cls.problem, backend="auto")
        cls.result = driver.run_scf(cls.system)

    def test_forces_equal_the_reference(self) -> None:
        reference = compute_periodic_forces(self.reference_system, self.reference)
        forces = driver.compute_forces(self.system, self.result, symmetrize=False)
        self.assertAlmostEqual(self.result.energies.total, self.reference.energies.total, delta=1.0e-8)
        for name in ("ion_ion", "nonlocal_pseudopotential", "unsymmetrized", "total"):
            np.testing.assert_allclose(getattr(forces, name), getattr(reference, name), atol=1.0e-7, err_msg=name)
        np.testing.assert_allclose(forces.local.ionic, reference.local.ionic, atol=1.0e-7)
        np.testing.assert_allclose(forces.local.core_correction, reference.local.core_correction, atol=1.0e-8)
        np.testing.assert_array_equal(forces.torque, 0.0)  # a crystal has none
        np.testing.assert_allclose(forces.total.sum(axis=0), 0.0, atol=1.0e-12)
        self.assertGreater(float(np.max(np.abs(forces.total))), 0.1)
        self.assertEqual(set(forces.phase_seconds), {"ion_ion", "local", "nonlocal"})

    def test_a_crystal_has_no_symmetrized_forces(self) -> None:
        with self.assertRaises(ValueError):
            driver.compute_forces(self.system, self.result, symmetrize=True)

    def test_moved_atoms_give_what_a_fresh_preparation_gives(self) -> None:
        shifts = ([0.05, -0.03, 0.02], [-0.02, 0.04, 0.01])
        # the second atom is moved a lattice vector out of the cell: the coordinates stay continuous, the sums fold them
        moved = [
            Atom(atom.symbol, np.array(atom.position) + np.array(shift) + (7.7 * np.eye(3)[0] if index else 0.0))
            for index, (atom, shift) in enumerate(zip(self.problem.atoms, shifts))
        ]
        new = driver.update_geometry(self.system, moved)
        result = driver.run_scf(new, warm_start=driver.warm_start_for_moved_atoms(self.system, self.result, new))
        np.testing.assert_allclose(new.atoms[1].position, moved[1].position)  # unwrapped, as an optimizer needs
        folded = [Atom(a.symbol, np.array(a.position) - (7.7 * np.eye(3)[0] if i else 0.0)) for i, a in enumerate(moved)]
        fresh_system = prepare_periodic_single_point(dataclasses.replace(self.problem, atoms=folded))
        fresh = run_scf(fresh_system)
        self.assertAlmostEqual(result.energies.total, fresh.energies.total, delta=1.0e-8)
        np.testing.assert_allclose(
            driver.compute_forces(new, result, symmetrize=False).total,
            compute_periodic_forces(fresh_system, fresh).total,
            atol=1.0e-6,
        )

    def test_a_relaxation_with_the_accelerated_backend_lowers_the_energy(self) -> None:
        system = driver.prepare_single_point(self.problem, backend="auto")
        outcome = run_relaxation(
            system,
            RelaxationSettings(method="bfgs", max_movements=3, minimum_force=1.0e-3),
            driver.ionic_backend(),
            symmetrize=False,
        )
        energies = [record.energy for record in outcome.records]
        self.assertGreaterEqual(len(energies), 3)
        self.assertLess(energies[-1], energies[0])
        self.assertTrue(all(record.converged for record in outcome.records))


if __name__ == "__main__":
    unittest.main()
