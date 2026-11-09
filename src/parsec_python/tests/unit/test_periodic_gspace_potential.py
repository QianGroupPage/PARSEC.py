"""Fortran's G-space local potential of a 3-D cell, its alpha term and potshift.

``parsec_python.V_ion.gspace_local`` (``v_first.f90``, ``pot_local.f90``,
``fourier_f.f90``, ``force_loc_pb.f90``) and the zero-average XC potential of
``SCF.pbc`` (``potshift.f90``).  Exact properties: the form factor is the q
table of ``fourier_1d``, the FFT equals the direct G sum at the grid points
(orthorhombic and general grids, with a grid shift), a rigid translation by a
grid step moves the potential by one point, the average and alpha term
agree with the real-space Ewald split, the forces are the derivative of the
energy (components and SCF), potshift changes no total energy and shifts the
eigenvalues by the removed average; the input rules and the defaults.  The
comparisons with the Fortran binary and the convergence study are in the
report of the parity round.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from parsec_python.Forces.periodic import periodic_local_forces
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import UnsupportedParsecOptionError
from parsec_python.models import (
    Atom,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SpeciesPotential,
)
from parsec_python.Pseudopotential.fourier_filter import DELTA_Q, fourier_transform
from parsec_python.SCF import prepare_periodic_single_point, run_scf
from parsec_python.SCF.pbc import update_periodic_geometry
from parsec_python.V_ion import (
    ewald_alpha_z_energy,
    ewald_local_ionic_potential,
    load_pseudopotentials,
)
from parsec_python.V_ion import gspace_local as G
from parsec_python.tests.unit.test_periodic_forces import (
    _H2,
    _free_energy,
    _hydrogen_cell,
    _moved,
)

DATA = Path(__file__).resolve().parents[1] / "data"
_FCC = 0.5 * 10.26 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def _silicon():
    specifications = {"Si": SpeciesPotential(DATA / "Si_POTRE.DAT", 2)}
    return specifications, load_pseudopotentials(specifications)


def _grids():
    cube = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
    yield "orthorhombic", cube, build_periodic_grid(
        cube, PeriodicGridSettings(fortran_routines=True, spacing=0.7, expansion_order=4, local_potential="g_space")
    )
    fcc = PeriodicCell(lattice_vectors=_FCC)
    yield "general, shifted", fcc, build_periodic_grid(
        fcc,
        PeriodicGridSettings(
            fortran_routines=True,
            spacing=0.9, expansion_order=4, shift=(0.5, 0.0, 0.5), local_potential="g_space"
        ),
    )


_SI_ATOMS = (
    Atom("Si", np.array([0.12, 0.03, 0.10])),
    Atom("Si", 0.25 * 10.26 * np.ones(3) + np.array([0.12, 0.0, -0.07])),
)


class FormFactorAndAlphaTests(unittest.TestCase):
    def test_form_factor_is_the_q_table_of_fourier_1d(self) -> None:
        # pbc%vloc(j+1) = fvion(j) pi^2/q^2 (Hartree) at q = 0.015 (j - 1).
        _, potentials = _silicon()
        potential = potentials["Si"]
        count = 600
        table = fourier_transform(
            potential, potential.channel_potentials[2], 0, count, coulomb=2.0 * potential.ionic_charge
        )
        q = DELTA_Q * np.arange(1, count)
        np.testing.assert_allclose(
            G.local_form_factor(potential, 2, q),
            2.0 * table[1:] * np.pi**2 / q**2,
            rtol=1e-12,
            atol=1e-9,
        )
        with self.assertRaises(ValueError):
            G.local_form_factor(potential, 2, np.array([0.0]))

    def test_alpha_limit_and_fortran_difference(self) -> None:
        _, potentials = _silicon()
        potential = potentials["Si"]
        radii, weights = G._transform_weights(potential, 2)
        analytic = G.alpha_coefficient(potential, 2, "analytic")
        self.assertAlmostEqual(analytic, 4.0 * np.pi * float(np.sum(weights)), places=10)
        difference = G.alpha_coefficient(potential, 2, "finite_difference")
        # 4 pi int r (rV + 2Z) j0(dq r) dr: the q -> 0 limit minus dq^2/6 of the r^2 moment.
        moment = 4.0 * np.pi * float(np.sum(weights * radii * radii))
        self.assertAlmostEqual(difference - analytic, -DELTA_Q**2 / 6.0 * moment, delta=1e-3 * abs(DELTA_Q**2 / 6.0 * moment))
        self.assertNotEqual(difference, analytic)
        with self.assertRaises(ValueError):
            G.alpha_coefficient(potential, 2, "table")

    def test_qtable_end(self) -> None:
        self.assertEqual(G.fortran_qtable_size(0.6045762979144981), int(2 * np.pi / 0.6045762979144981 / 0.015) + 4)


class GSpacePotentialTests(unittest.TestCase):
    def test_fft_is_the_direct_sum_with_zero_average(self) -> None:
        specifications, potentials = _silicon()
        for name, cell, grid in _grids():
            with self.subTest(grid=name):
                lattice = cell.lattice_vectors
                values = G.gspace_local_ionic_potential(
                    grid, _SI_ATOMS, potentials, specifications, lattice
                )
                vectors = G.gspace_vectors(grid, lattice)
                self.assertFalse(np.any(vectors.norms == 0.0))
                # Both Nyquist planes of an even grid are in the box.
                self.assertEqual(int(np.max(vectors.integers[:, 0])), grid.points_per_axis[0] // 2)
                self.assertEqual(int(np.min(vectors.integers[:, 0])), -(grid.points_per_axis[0] // 2))
                factors = G.local_form_factor(potentials["Si"], 2, vectors.norms)
                positions = np.array([atom.position for atom in _SI_ATOMS])
                coefficients = factors * np.exp(-1j * vectors.vectors @ positions.T).sum(axis=1)
                coefficients /= abs(np.linalg.det(lattice))
                rows = np.arange(0, grid.size, 7)
                direct = (np.exp(1j * grid.coordinates[rows] @ vectors.vectors.T) @ coefficients).real
                np.testing.assert_allclose(values[rows], direct, rtol=0.0, atol=1e-11)
                self.assertAlmostEqual(float(np.mean(values)), 0.0, delta=1e-13)

    def test_translation_by_a_grid_step(self) -> None:
        specifications, potentials = _silicon()
        for name, cell, grid in _grids():
            with self.subTest(grid=name):
                lattice = cell.lattice_vectors
                n = tuple(int(value) for value in grid.points_per_axis)
                step = lattice[0] / n[0]
                moved = tuple(Atom("Si", atom.position + step) for atom in _SI_ATOMS)
                first = G.gspace_local_ionic_potential(grid, _SI_ATOMS, potentials, specifications, lattice)
                second = G.gspace_local_ionic_potential(grid, moved, potentials, specifications, lattice)
                np.testing.assert_allclose(
                    second.reshape(n), np.roll(first.reshape(n), 1, axis=0), rtol=0.0, atol=1e-11
                )

    def test_shape_and_average_agree_with_the_ewald_split(self) -> None:
        # Both constructions are the same potential up to the grid: the shape
        # to the aliasing of the Si potential on a fine grid, the average plus
        # alpha term to the sampling of the short-range sum.
        specifications, potentials = _silicon()
        cell = PeriodicCell(lattice_vectors=5.6 * np.eye(3))
        atoms = (Atom("Si", np.array([1.0, 2.0, 2.5])), Atom("Si", np.array([4.2, 3.1, 5.3])))
        grid = build_periodic_grid(cell, PeriodicGridSettings(fortran_routines=True, spacing=0.25, expansion_order=4))
        real = ewald_local_ionic_potential(grid, atoms, potentials, specifications, cell.lattice_vectors)
        reciprocal = G.gspace_local_ionic_potential(grid, atoms, potentials, specifications, cell.lattice_vectors)
        np.testing.assert_allclose(reciprocal, real - real.mean(), rtol=0.0, atol=5e-3)
        electrons = 8.0
        ewald_average = real.mean() + ewald_alpha_z_energy(atoms, potentials, cell.lattice_vectors, electrons) / electrons
        alpha = G.gspace_alpha_energy(atoms, potentials, specifications, cell.lattice_vectors, electrons) / electrons
        self.assertAlmostEqual(ewald_average, alpha, delta=1e-5)
        # the preparation's convention (SCF.pbc.zero_average_local_potential, PARSEC 1.5's): the split without the
        # average of its split, Fortran's alpha term; the G-space potential has a zero average
        from parsec_python.SCF.pbc import zero_average_local_potential

        shifted, energy = zero_average_local_potential(
            real, atoms, potentials, specifications, cell.lattice_vectors, electrons
        )
        self.assertAlmostEqual(float(shifted.mean()), 0.0, delta=1e-5)  # the sampling of the short-range sum
        self.assertEqual(energy, alpha * electrons)
        # the total is that of the earlier convention: the band energy loses N_e times the removed constant
        removed = float(real.mean() - shifted.mean())
        earlier = ewald_alpha_z_energy(atoms, potentials, cell.lattice_vectors, electrons)
        self.assertAlmostEqual(energy - electrons * removed, earlier, delta=1e-9)
        self.assertAlmostEqual(float(reciprocal.mean()), 0.0, delta=1e-12)


class GSpaceForceTests(unittest.TestCase):
    def test_local_force_is_the_gradient_of_the_gspace_energy(self) -> None:
        specifications, potentials = _silicon()
        step = 1e-6
        for name, cell, grid in _grids():
            with self.subTest(grid=name):
                lattice = cell.lattice_vectors
                x = grid.coordinates
                b = 2.0 * np.pi * np.linalg.inv(lattice).T
                density = 0.05 + 0.02 * np.cos(x @ b[0]) + 0.015 * np.sin(x @ (b[1] + b[2]))
                zeros = np.zeros(grid.size)
                terms = periodic_local_forces(
                    grid, _SI_ATOMS, potentials, specifications, lattice, density, zeros, zeros
                )
                np.testing.assert_allclose(
                    terms.ionic,
                    G.gspace_local_forces(grid, _SI_ATOMS, potentials, specifications, lattice, density),
                    rtol=0.0,
                    atol=0.0,
                )

                def energy(moved):
                    field = G.gspace_local_ionic_potential(grid, moved, potentials, specifications, lattice)
                    return grid.integrate(density * field)

                for index in range(2):
                    for axis in range(3):
                        numeric = -(
                            energy(_moved(_SI_ATOMS, index, axis, step))
                            - energy(_moved(_SI_ATOMS, index, axis, -step))
                        ) / (2.0 * step)
                        self.assertAlmostEqual(terms.ionic[index, axis], numeric, delta=2e-8)

    def test_scf_forces_are_the_derivative_of_the_free_energy(self) -> None:
        from parsec_python.Relaxation.backend import fortran_periodic_backend as periodic_backend

        for name, lattice, components in (
            ("orthorhombic", 5.0 * np.eye(3), ((1, 0), (0, 2))),
            ("monoclinic", [[5.0, 0.0, 0.0], [1.5, 5.0, 0.0], [0.0, 0.0, 5.0]], ((1, 0),)),
        ):
            with self.subTest(cell=name):
                problem = _hydrogen_cell(lattice, _H2)
                problem = replace(
                    problem,
                    grid=replace(problem.grid, local_potential="g_space", alpha_term="finite_difference"),
                )
                backend = periodic_backend(None)
                system = prepare_periodic_single_point(problem)
                self.assertAlmostEqual(float(np.mean(system.ionic_potential)), 0.0, delta=1e-13)
                result = backend.run_scf(system)
                self.assertTrue(result.converged)
                forces = backend.compute_forces(system, result, symmetrize=False)
                for index, axis in components:
                    energies = []
                    for sign in (1.0, -1.0):
                        moved = update_periodic_geometry(
                            system, _moved(problem.atoms, index, axis, sign * 1e-5)
                        )
                        np.testing.assert_allclose(
                            moved.ionic_potential,
                            prepare_periodic_single_point(replace(problem, atoms=moved.input.atoms)).ionic_potential,
                            rtol=0.0,
                            atol=0.0,
                        )
                        energies.append(_free_energy(backend.run_scf(moved)))
                    numeric = -(energies[0] - energies[1]) / 2e-5
                    self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=2e-6)


class PotentialShiftTests(unittest.TestCase):
    def test_total_energy_unchanged_and_eigenvalues_shifted(self) -> None:
        problem = _hydrogen_cell(5.0 * np.eye(3), _H2)
        plain = run_scf(prepare_periodic_single_point(problem))
        shifted_problem = replace(problem, scf=replace(problem.scf, zero_average_xc_potential=True))
        system = prepare_periodic_single_point(shifted_problem)
        shifted = run_scf(system)
        self.assertTrue(plain.converged and shifted.converged)
        self.assertAlmostEqual(shifted.energies.total, plain.energies.total, delta=2e-9)
        self.assertAlmostEqual(float(np.mean(shifted.xc_potential)), 0.0, delta=1e-12)
        average = float(np.mean(plain.xc_potential))
        np.testing.assert_allclose(
            np.asarray(shifted.eigenvalues), np.asarray(plain.eigenvalues) - average, atol=1e-7
        )
        self.assertAlmostEqual(shifted.fermi_level, plain.fermi_level - average, delta=1e-6)
        # The energy of the functional and the unshifted potential on request.
        density = np.asarray(shifted.density)
        both = system.evaluate_xc(density), system.evaluate_xc(density, zero_average=False)
        self.assertEqual(both[0].total_energy, both[1].total_energy)
        np.testing.assert_allclose(both[0].potential, both[1].potential - both[1].potential.mean(), atol=1e-14)

    def test_spin_channels_lose_their_own_average(self) -> None:
        problem = _hydrogen_cell(5.0 * np.eye(3), _H2, spin=True)
        problem = replace(problem, scf=replace(problem.scf, zero_average_xc_potential=True))
        system = prepare_periodic_single_point(problem)
        up = np.asarray(system.initial_density) * 0.6
        down = np.asarray(system.initial_density) * 0.4
        for result in (system.evaluate_xc_spin_polarized(up, down), system.evaluate_spin_xc(up, down)):
            self.assertAlmostEqual(float(np.mean(result.potential_up)), 0.0, delta=1e-13)
            self.assertAlmostEqual(float(np.mean(result.potential_down)), 0.0, delta=1e-13)
        reference = system.evaluate_spin_xc(up, down, zero_average=False)
        self.assertLess(float(np.mean(reference.potential_up)), float(np.mean(reference.potential_down)))


_CELL = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  5.0 5.0 5.0
end Cell_Shape
Grid_Spacing: 0.6
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-5
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  1.6 2.4 2.2
  3.05 2.55 2.45
end Atom_Coord
"""

_CLUSTER = """\
Boundary_Conditions: cluster
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.6 bohr
Coordinate_Unit: Cartesian_Bohr
Hartree_Boundary_Tolerance: off
States_Num: 2
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 0.7
  0.0 0.0 -0.7
end Atom_Coord
"""

_SLAB = """\
Boundary_Conditions: slab
begin Cell_Shape
 3.0 0.0
 0.0 3.2
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.6
States_Num: 2
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.0 0.0 0.7
  0.0 0.0 -0.7
end Atom_Coord
"""


class InputTests(unittest.TestCase):
    def parse(self, template: str, extra: str = ""):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(DATA / "H_POTRE.DAT", Path(folder) / "H_POTRE.DAT")
            path = Path(folder) / "parsec.in"
            path.write_text(template.format(extra=extra), encoding="utf-8")
            return parse_parsec_input(path)

    def test_defaults_print_nothing_new(self) -> None:
        translation = self.parse(_CELL)
        self.assertEqual(translation.problem.grid.local_potential, "real_space")
        # PARSEC 1.5's potshift is the default of an unpolarized 3-D cell since 2026-11-08; .false. restores the
        # earlier potential and its settings
        self.assertTrue(translation.problem.scf.zero_average_xc_potential)
        self.assertIn("zero_average_xc_potential=True", repr(translation.problem.scf))
        earlier = self.parse(_CELL, "Zero_Average_Vxc: .false.\n")
        self.assertFalse(earlier.problem.scf.zero_average_xc_potential)
        for text in (repr(earlier.problem.grid), repr(earlier.problem.scf), repr(earlier.problem)):
            self.assertNotIn("local_potential", text)
            self.assertNotIn("alpha_term", text)
            self.assertNotIn("zero_average", text)
        self.assertNotIn("local_potential", repr(PeriodicGridSettings(spacing=0.5)))
        self.assertNotIn("zero_average", repr(SCFSettings()))

    def test_the_zero_average_default_is_that_of_an_unpolarized_bulk_cell(self) -> None:
        # a spin-polarized cell keeps the potential of each channel (each channel losing its own average would move
        # them against each other), a slab of either routines and a cluster never take the shift
        for template, extra in (
            (_CELL, "Spin_Polarization: .true.\n"),
            (_SLAB, ""),
            (_SLAB, "Periodic_System: .true.\nUse_Fortran_Routines: .false.\n"),
            (_CLUSTER, ""),
        ):
            with self.subTest(extra=extra, template=template.splitlines()[0]):
                self.assertFalse(self.parse(template, extra).problem.scf.zero_average_xc_potential)
        self.assertTrue(
            self.parse(_CELL, "Spin_Polarization: .true.\nZero_Average_Vxc: .true.\n").problem.scf.zero_average_xc_potential
        )

    def test_keywords(self) -> None:
        translation = self.parse(
            _CELL,
            "Periodic_Local_Potential: g_space\nPeriodic_Alpha_Term: fortran\nZero_Average_Vxc: .true.\n",
        )
        problem = translation.problem
        self.assertEqual(problem.grid.local_potential, "g_space")
        self.assertEqual(problem.grid.alpha_term, "finite_difference")
        self.assertTrue(problem.scf.zero_average_xc_potential)
        self.assertIn("local_potential='g_space'", repr(problem.grid))
        self.assertIn("zero_average_xc_potential=True", repr(problem.scf))
        self.assertEqual(
            self.parse(_CELL, "Periodic_Local_Potential: g_space\n").problem.grid.alpha_term, "analytic"
        )

    def test_refusals_and_warnings(self) -> None:
        for template, extra in (
            (_CELL, "Periodic_Alpha_Term: fortran\n"),
            (_CELL, "Zero_Average_Vxc: .true.\nNon_Collinear_Magnetism: .true.\nSpin_Polarization: .true.\n"),
            (_SLAB, "Periodic_Local_Potential: g_space\n"),
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(UnsupportedParsecOptionError):
                    self.parse(template, extra)
        warned = self.parse(_CLUSTER, "Zero_Average_Vxc: .true.\nPeriodic_Local_Potential: g_space\n")
        self.assertTrue(any("periodic cell only" in text for text in warned.warnings))
        self.assertFalse(warned.problem.scf.zero_average_xc_potential)
        slab = self.parse(_SLAB, "Zero_Average_Vxc: .true.\n")
        self.assertTrue(any("3-D periodic cells only" in text for text in slab.warnings))
        self.assertFalse(slab.problem.scf.zero_average_xc_potential)
        with self.assertRaises(ValueError):
            PeriodicGridSettings(spacing=0.5, alpha_term="finite_difference")
        with self.assertRaises(ValueError):
            PeriodicGridSettings(spacing=0.5, local_potential="fourier")

    def test_accelerated_cli_runs_both_options_at_gamma(self) -> None:
        # A Gamma-point cell with either option runs on the accelerated driver
        # (acceleration/SCF/periodic_routes.py); with spin it is routed by name.
        from parsec_python.acceleration.cli import _reference_only_features

        for extra, name in (
            ("Periodic_Local_Potential: g_space\n", "Periodic_Local_Potential"),
            ("Zero_Average_Vxc: .true.\n", "Zero_Average_Vxc"),
        ):
            with self.subTest(name=name):
                features = _reference_only_features(
                    self.parse(_CELL, extra), accelerated_backend_selected=False
                )
                self.assertEqual(features, [])
                spin = _reference_only_features(
                    self.parse(_CELL, extra + "Spin_Polarization: .true.\n"),
                    accelerated_backend_selected=False,
                )
                self.assertTrue(any("spin polarization" in item for item in spin), spin)
        self.assertEqual(
            _reference_only_features(self.parse(_CELL), accelerated_backend_selected=False), []
        )


if __name__ == "__main__":
    unittest.main()
