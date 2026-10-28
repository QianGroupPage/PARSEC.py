"""Slab boundary conditions: periodic in the xy plane, a finite box along z.

The Hartree solver is checked against the analytic potential of in-plane Fourier modes and of uniform sheets, the 2D
Ewald energy against the 3D one for a neutral cell with a large vacuum and against its own eta independence, and the
local ionic potential against itself for two Gaussian widths.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from parsec_python.Grid import build_periodic_grid
from parsec_python.Hartree.slab import solve_slab_hartree
from parsec_python.Input.parsec_input import parse_parsec_input
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.models import Atom, HartreeSettings, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.V_ion import ewald, load_pseudopotentials
from parsec_python.V_ion import slab as slab_ions
from parsec_python.V_xc.pbe import pbe

_H_POTRE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"
_TIGHT = HartreeSettings(relative_tolerance=1.0e-12, absolute_tolerance=1.0e-14, max_iterations=6000)

LATTICE = np.array([[8.0, 0.0, 0.0], [2.0, 7.0, 0.0], [0.0, 0.0, 30.0]])


def _slab_grid(lattice=LATTICE, spacing=0.5, order=12):
    return build_periodic_grid(
        PeriodicCell(lattice, dimensions=2), PeriodicGridSettings(spacing=spacing, expansion_order=order)
    )


class SlabCellTests(unittest.TestCase):
    def test_cell_needs_in_plane_vectors_and_a_z_height(self) -> None:
        with self.assertRaises(ValueError):
            PeriodicCell(np.array([[8.0, 0, 1.0], [2.0, 7.0, 0], [0, 0, 30.0]]), dimensions=2)
        with self.assertRaises(ValueError):
            PeriodicCell(np.array([[8.0, 0, 0], [2.0, 7.0, 0], [1.0, 0, 30.0]]), dimensions=2)
        with self.assertRaises(ValueError):
            PeriodicCell(np.eye(3), dimensions=1)

    def test_grid_layout(self) -> None:
        grid = _slab_grid()
        n1, n2, n3 = grid.shape
        self.assertEqual((n1, n2), (16, 14))
        self.assertEqual(n3, int(30.0 / 0.5) + 2)  # the Fortran count of planes
        z = grid.coordinates[:, 2]
        self.assertAlmostEqual(float(z.min()), -float(z.max()))  # planes symmetric about z = 0, half-step shifted
        self.assertNotIn(0.0, np.round(np.unique(z), 12))
        self.assertAlmostEqual(grid.volume_element * grid.size, 8.0 * 7.0 * 0.5 * n3)
        # neighbors: wrapped in plane, missing beyond the box
        rows = grid.rows_for_integer_coordinates(np.array([[n1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, n3], [1, 2, 3]]))
        self.assertGreaterEqual(rows[0], 0)
        self.assertGreaterEqual(rows[1], 0)
        self.assertEqual(rows[2], -1)
        self.assertEqual(rows[3], -1)
        self.assertEqual(rows[4], int(grid.lookup[1, 2, 3]))

    def test_laplacian_is_exact_for_smooth_fields_and_zero_padded_in_z(self) -> None:
        grid = _slab_grid()
        laplacian = build_negative_laplacian(grid)
        self.assertEqual(abs(laplacian - laplacian.T).max(), 0.0)
        reciprocal = 2.0 * np.pi * np.linalg.inv(LATTICE[:2, :2]).T
        k = reciprocal[0] + reciprocal[1]
        z = grid.coordinates[:, 2]
        profile = np.exp(-((z / 5.0) ** 2))
        values = np.cos(grid.coordinates[:, :2] @ k) * profile
        second = (4.0 * z**2 / 5.0**4 - 2.0 / 5.0**2) * profile
        exact = np.cos(grid.coordinates[:, :2] @ k) * ((k @ k) * profile - second)
        interior = np.abs(z) < 10.0
        np.testing.assert_allclose((laplacian @ values)[interior], exact[interior], atol=1.0e-8)
        # the gradient matrices differentiate the periodic directions and the field inside the box
        gradient = build_gradient(grid)
        np.testing.assert_allclose(
            (gradient[2] @ values)[interior],
            (np.cos(grid.coordinates[:, :2] @ k) * (-2.0 * z / 25.0) * profile)[interior],
            atol=1.0e-8,
        )


class SlabHartreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = _slab_grid()
        self.laplacian = build_negative_laplacian(self.grid)
        self.z = self.grid.coordinates[:, 2]
        self.xy = self.grid.coordinates[:, :2]
        self.reciprocal = 2.0 * np.pi * np.linalg.inv(LATTICE[:2, :2]).T

    def _profile(self, z):
        return (1.0 + 0.3 * z / 5.0) * np.exp(-((z / 3.0) ** 2))  # asymmetric: a dipole along z

    def test_matches_the_analytic_potential_of_sheets_and_fourier_modes(self) -> None:
        zz = np.linspace(-14.0, 14.0, 5601)
        dz = zz[1] - zz[0]
        planes = np.unique(self.z)
        for name, kvec in (
            ("uniform", np.zeros(2)),
            ("b1", self.reciprocal[0]),
            ("b1+b2", self.reciprocal[0] + self.reciprocal[1]),
        ):
            with self.subTest(name):
                density = self._profile(self.z) * np.cos(self.xy @ kvec)
                result = solve_slab_hartree(density, self.grid, self.laplacian, _TIGHT)
                g = float(np.linalg.norm(kvec))
                errors = []
                for z_target in planes[3:-3:5]:
                    if g == 0.0:
                        exact = -2.0 * np.pi * np.sum(self._profile(zz) * np.abs(z_target - zz)) * dz
                    else:
                        exact = (2.0 * np.pi / g) * np.sum(self._profile(zz) * np.exp(-g * np.abs(z_target - zz))) * dz
                    row = int(np.argmin(np.abs(self.z - z_target) + 10.0 * np.linalg.norm(self.xy, axis=1)))
                    errors.append(result.potential[row] / 2.0 - exact)  # Hartree
                self.assertLess(float(np.max(np.abs(errors))), 1.0e-4)

    def test_potential_far_from_the_charge_is_a_charged_sheet(self) -> None:
        # a net charge per area gives the slope -4 pi sigma (Hartree) outside the charge, on both sides
        density = np.exp(-((self.z / 2.0) ** 2))
        result = solve_slab_hartree(density, self.grid, self.laplacian, _TIGHT)
        sigma = float(np.sum(density) * self.grid.volume_element / (8.0 * 7.0))
        potential = result.potential.reshape(self.grid.shape)[0, 0] / 2.0
        planes = np.unique(self.z)
        slope_top = (potential[-1] - potential[-9]) / (planes[-1] - planes[-9])
        slope_bottom = (potential[8] - potential[0]) / (planes[8] - planes[0])
        self.assertAlmostEqual(slope_top, -2.0 * np.pi * sigma, delta=1.0e-6 * abs(sigma))
        self.assertAlmostEqual(slope_bottom, 2.0 * np.pi * sigma, delta=1.0e-6 * abs(sigma))


class SlabEwaldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.potentials = {"A": SimpleNamespace(ionic_charge=1.0), "B": SimpleNamespace(ionic_charge=-1.0)}
        self.atoms = [Atom("A", [0.7, 0.4, -1.9]), Atom("B", [3.3, 2.2, 0.8]), Atom("A", [5.1, 1.0, 2.5]), Atom("B", [1.9, 3.8, -0.6])]
        self.lattice = np.array([[6.5, 0.0, 0.0], [1.5, 5.8, 0.0], [0.0, 0.0, 40.0]])

    def test_independent_of_the_splitting_parameter(self) -> None:
        energies = [
            slab_ions.slab_ewald_energy(self.atoms, self.potentials, self.lattice, eta=eta) for eta in (0.35, 0.5, 0.8)
        ]
        np.testing.assert_allclose(energies, energies[0], rtol=0.0, atol=1.0e-9)

    def test_neutral_cell_agrees_with_the_bulk_sum_in_a_tall_box(self) -> None:
        # for a neutral cell the 3D sum with a tall vacuum has the dipole energy 2 pi p^2 / V -> 0
        slab = slab_ions.slab_ewald_energy(self.atoms, self.potentials, self.lattice)
        tall = np.array(self.lattice)
        tall[2, 2] = 400.0
        bulk = ewald.ewald_ion_ion_energy(self.atoms, self.potentials, tall)
        p_z = float(sum(self.potentials[a.symbol].ionic_charge * a.position[2] for a in self.atoms))
        dipole = 2.0 * (2.0 * np.pi / (abs(np.linalg.det(tall)))) * p_z**2 / 2.0  # Hartree 2 pi p^2/(2V), Rydberg x 2
        self.assertAlmostEqual(slab, bulk - 0.0, delta=max(5.0 * abs(dipole), 1.0e-4))

    def test_translation_along_z_and_in_plane_lattice_vectors(self) -> None:
        reference = slab_ions.slab_ewald_energy(self.atoms, self.potentials, self.lattice)
        moved = [Atom(a.symbol, np.asarray(a.position) + np.array([0.0, 0.0, 2.3])) for a in self.atoms]
        self.assertAlmostEqual(slab_ions.slab_ewald_energy(moved, self.potentials, self.lattice), reference, places=8)
        wrapped = [
            Atom(a.symbol, np.asarray(a.position) + (self.lattice[0] if i == 1 else 0.0)) for i, a in enumerate(self.atoms)
        ]
        self.assertAlmostEqual(slab_ions.slab_ewald_energy(wrapped, self.potentials, self.lattice), reference, places=8)


class SlabLocalPotentialTests(unittest.TestCase):
    def test_does_not_depend_on_the_width_of_the_compensating_gaussians(self) -> None:
        specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
        potentials = load_pseudopotentials(specifications, xc_functional="ca")
        lattice = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 14.0]])
        grid = build_periodic_grid(
            PeriodicCell(lattice, dimensions=2), PeriodicGridSettings(spacing=0.35, expansion_order=8)
        )
        laplacian = build_negative_laplacian(grid)
        atoms = [Atom("H", [1.2, 3.0, 0.4]), Atom("H", [4.1, 2.2, -0.9])]
        results = []
        original = slab_ions._SIGMA_FACTOR
        try:
            for factor in (0.8, 1.2):
                slab_ions._SIGMA_FACTOR = factor
                results.append(
                    slab_ions.slab_local_ionic_potential(grid, atoms, potentials, specifications, laplacian, _TIGHT)
                )
        finally:
            slab_ions._SIGMA_FACTOR = original
        inside = np.abs(grid.coordinates[:, 2]) < 4.0
        np.testing.assert_allclose(results[0][inside], results[1][inside], atol=5.0e-4)


class SlabXCTests(unittest.TestCase):
    def test_pbe_potential_is_the_derivative_of_the_energy_with_zero_padding_in_z(self) -> None:
        grid = _slab_grid(spacing=0.6, order=8)
        z = grid.coordinates[:, 2]
        reciprocal = 2.0 * np.pi * np.linalg.inv(LATTICE[:2, :2]).T
        density = 0.02 * (1.2 + 0.4 * np.cos(grid.coordinates[:, :2] @ reciprocal[0])) * np.exp(-((z / 4.0) ** 2))
        direction = np.random.default_rng(1).standard_normal(grid.size) * 1.0e-5 * np.exp(-((z / 6.0) ** 2))
        step = 1.0e-3
        analytic = float(np.sum(pbe(density, grid).potential * direction) * grid.volume_element)
        numerical = (pbe(density + step * direction, grid).total_energy - pbe(density - step * direction, grid).total_energy) / (
            2.0 * step
        )
        self.assertAlmostEqual(numerical / analytic, 1.0, places=5)


_SLAB_INPUT = """
Periodic_System: .true.
Boundary_Conditions: slab
begin Cell_Shape
  7.0  0.0  0
  3.0  6.0  0
end Cell_Shape
Lattice_Vector_Scale: 1
Boundary_Sphere_Radius: 8
Grid_Spacing: 0.5
Expansion_Order: 8
Atom_Types_Num: 1
Atom_Type: H
Element_Symbol: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  1.0 1.0 3.5
  9.0 2.0 4.5
end Atom_Coord
Correlation_Type: ca
States_Num: 3
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
3 3
end Monkhorst_Pack_Grid
"""


class SlabInputTests(unittest.TestCase):
    def test_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "parsec.in").write_text(_SLAB_INPUT)
            (folder / "H_POTRE.DAT").write_bytes(_H_POTRE.read_bytes())
            translation = parse_parsec_input(folder / "parsec.in")
        problem = translation.problem
        cell = problem.periodic_cell
        self.assertEqual(cell.dimensions, 2)
        np.testing.assert_allclose(cell.lattice_vectors, [[7, 0, 0], [3, 6, 0], [0, 0, 16]])
        self.assertEqual(problem.monkhorst_pack_dimensions, (3, 3, 1))
        positions = np.array([atom.position for atom in problem.atoms])
        # the slab is centred along z (mean of the extremes was 4.0) and in-plane positions are folded into the cell
        np.testing.assert_allclose(positions[:, 2], [-0.5, 0.5])
        fractions = positions[:, :2] @ np.linalg.inv(cell.lattice_vectors[:2, :2])
        self.assertTrue(np.all((fractions >= 0.0) & (fractions < 1.0)))
        self.assertTrue(any("shifted" in warning for warning in translation.warnings))

    def test_spin_orbit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "parsec.in").write_text(_SLAB_INPUT + "\nSO_from_scratch: true\nSpin_Polarization: true\n")
            (folder / "H_POTRE.DAT").write_bytes(_H_POTRE.read_bytes())
            with self.assertRaises(Exception):
                parse_parsec_input(folder / "parsec.in")


if __name__ == "__main__":
    unittest.main()
