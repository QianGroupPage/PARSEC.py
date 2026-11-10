"""Non-orthogonal periodic cells: Fortran's grid, Laplacian, gradient and image sums.

Covers :mod:`parsec_python.Laplacian.nonorthogonal` (``pbc_grid_coefs``),
:class:`parsec_python.Grid.pbc.GeneralPeriodicGrid` (``adjust_step``, shift),
the FFT Hartree solve, PBE and the k-point kinetic term on such a grid, the
general-lattice Ewald, projector and k-point code, ``Super_Cell`` /
``Coordinate_Unit: lattice_vectors`` input and atoms outside the cell.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from parsec_python.Grid.cell import (
    back_to_cell,
    cell_diagonal_length,
    lattice_plane_spacings,
    replicate_positions,
    wrap_into_cell,
)
from parsec_python.Grid.kpoints import monkhorst_pack_grid
from parsec_python.Grid.pbc import (
    GeneralPeriodicGrid,
    adjust_step,
    build_general_periodic_grid,
    build_periodic_grid,
    periodic_grid_shift,
    uses_general_periodic_grid,
)
from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
from parsec_python.Hartree import solve_periodic_hartree
from parsec_python.Input.parsec_input import ParsecInputError, parse_parsec_input
from parsec_python.Laplacian import build_gradient, build_negative_laplacian
from parsec_python.Laplacian.nonorthogonal import pbc_grid_coefficients
from parsec_python.models import (
    Atom,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.V_ion import (
    build_nonlocal_projectors,
    ewald_ion_ion_energy,
    load_pseudopotentials,
)
from parsec_python.V_ion.ionic_potential import (
    _periodic_image_translations,
    _projector_support_radius,
    real_spherical_harmonics,
)
from parsec_python.V_xc import pbe, pbe_spin_polarized

_DATA = Path(__file__).resolve().parents[1] / "data"
_H_POTRE = _DATA / "H_POTRE.DAT"
_ANGSTROM = 1.0 / 0.529177

# Lattices of the Fortran examples (rows = lattice vectors, bohr).
_FCC = 0.5 * 10.26 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
_BCC = 0.5 * 5.71 * _ANGSTROM * np.array(
    [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]
)
_HEXAGONAL = 3.21 * _ANGSTROM * np.array(
    [[0.5, -0.866025403785, 0.0], [0.5, 0.866025403785, 0.0], [0.0, 0.0, 1.632993]]
)
_HEXAGONAL_ACUTE = np.array([[5.0, 0.0, 0.0], [2.5, 4.330127018922193, 0.0], [0.0, 0.0, 7.0]])
_TRICLINIC = 3.21 * _ANGSTROM * np.array(
    [[0.6, -0.866025403785, 0.0], [0.4, 0.866025403785, 0.0], [0.0, 0.1, 1.632993]]
)


def _spacings(lattice: np.ndarray, spacing: float) -> np.ndarray:
    return np.array([adjust_step(length, spacing)[1] for length in np.linalg.norm(lattice, axis=1)])


def _general_grid(lattice: np.ndarray, spacing: float, order: int = 12, shift=(0.0, 0.0, 0.0)):
    return build_general_periodic_grid(
        PeriodicCell(lattice_vectors=lattice),
        PeriodicGridSettings(fortran_routines=True, spacing=spacing, expansion_order=order, shift=shift),
    )


def _plane_wave(grid, integers) -> tuple[np.ndarray, np.ndarray]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(grid.cell.lattice_vectors).T
    k = np.asarray(integers, dtype=float) @ reciprocal
    return k, np.exp(1j * (grid.coordinates @ k))


class AdjustStepTests(unittest.TestCase):
    def test_largest_even_count_with_step_not_below_the_target(self) -> None:
        # The example of the report: 10 bohr at h = 0.6 gives n = 16.
        self.assertEqual(adjust_step(10.0, 0.6), (16, 10.0 / 16))
        count, step = adjust_step(7.254540, 0.6)
        self.assertEqual(count, 12)
        self.assertAlmostEqual(step, 7.254540 / 12, places=14)
        # An exact even ratio is kept.
        self.assertEqual(adjust_step(14.0, 0.5)[0], 28)
        with self.assertRaises(ValueError):
            adjust_step(1.0, 0.6)

    def test_shift_rule(self) -> None:
        self.assertEqual(periodic_grid_shift(_FCC, ignore_symmetry=False), (0.0, 0.0, 0.0))
        # Hexagonal: the c axis is orthogonal to both others.
        self.assertEqual(periodic_grid_shift(_HEXAGONAL, ignore_symmetry=False), (0.0, 0.0, 0.5))
        self.assertEqual(periodic_grid_shift(_HEXAGONAL, ignore_symmetry=True), (0.0, 0.0, 0.0))

    def test_which_cells_get_the_general_grid(self) -> None:
        settings = PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4)
        cubic = PeriodicCell(lattice_vectors=6.0 * np.eye(3))
        self.assertFalse(uses_general_periodic_grid(cubic, settings))
        self.assertIs(type(build_periodic_grid(cubic, settings)).__name__, "PeriodicRealSpaceGrid")
        for vectors in (_FCC, np.diag([6.0, 6.0, 6.3]), np.diag([-6.0, 6.0, 6.0])):
            cell = PeriodicCell(lattice_vectors=vectors)
            self.assertTrue(uses_general_periodic_grid(cell, settings))
            self.assertIsInstance(build_periodic_grid(cell, settings), GeneralPeriodicGrid)

    def test_grid_points_lie_in_the_cell(self) -> None:
        grid = _general_grid(_HEXAGONAL, 0.6, shift=(0.0, 0.0, 0.5))
        fractions = np.linalg.solve(_HEXAGONAL.T, grid.coordinates.T).T
        self.assertTrue(np.all(fractions >= -1.0e-12) and np.all(fractions < 1.0))
        np.testing.assert_allclose(
            grid.physical_coordinates(grid.integer_coordinates), grid.coordinates, atol=1e-12
        )
        with self.assertRaises(AttributeError):
            grid.spacing
        self.assertAlmostEqual(
            grid.volume_element * grid.size, abs(np.linalg.det(_HEXAGONAL)), places=10
        )


class FortranCoefficientTests(unittest.TestCase):
    """Numbers printed by Fortran PARSEC (examples' ``ref.out.0000`` and runs)."""

    def test_triclinic_example_with_fortran_step_ratio(self) -> None:
        coefficients = pbc_grid_coefficients(
            _TRICLINIC, _spacings(_TRICLINIC, 0.6), step_ratio_quirk=True
        )
        np.testing.assert_allclose(
            coefficients.original,
            [1.34768800017690, 1.34843800032558, 1.00375000074337,
             1.36443865530008, 5.971030655923869e-2, -8.109606222593960e-2],
            rtol=1e-12,
        )
        self.assertEqual(coefficients.directions.tolist(), [[-1, -1, 0], [1, 0, 1], [1, 1, -1]])
        np.testing.assert_allclose(
            coefficients.coefficients[3:],
            [0.635226634830649, 0.126317483662964, 7.712858627828446e-2],
            rtol=1e-12,
        )
        np.testing.assert_array_equal(
            np.round(coefficients.coefficients, 2), [0.67, 0.58, 0.92, 0.64, 0.13, 0.08]
        )

    def test_fcc_bcc_and_hexagonal(self) -> None:
        fcc = pbc_grid_coefficients(_FCC, _spacings(_FCC, 0.6))
        self.assertEqual(fcc.directions.tolist(), [[-1, 0, 1], [-1, 1, 0], [0, -1, 1]])
        np.testing.assert_allclose(fcc.coefficients, 0.5, rtol=1e-12)
        self.assertEqual(fcc.number_of_directions, 3)
        bcc = pbc_grid_coefficients(_BCC, _spacings(_BCC, 0.7))
        self.assertEqual(bcc.directions.tolist(), [[-1, -1, -1], [-1, -1, 0], [0, 1, 1]])
        self.assertEqual(bcc.active, (1, 0, 0))
        np.testing.assert_allclose(bcc.coefficients, [0.75, 0.75, 0.75, 0.75, 0.0, 0.0], atol=1e-12)
        acute = pbc_grid_coefficients(_HEXAGONAL_ACUTE, _spacings(_HEXAGONAL_ACUTE, 0.55))
        self.assertEqual(acute.active, (-1, 0, 0))
        self.assertEqual(acute.directions[0].tolist(), [1, -1, 0])
        hexagonal = pbc_grid_coefficients(_HEXAGONAL, _spacings(_HEXAGONAL, 0.6))
        self.assertEqual(hexagonal.number_of_directions, 1)
        self.assertEqual(hexagonal.directions[0].tolist(), [1, 1, 0])
        np.testing.assert_array_equal(
            np.round(hexagonal.coefficients, 2), [0.67, 0.67, 1.0, 0.67, 0.0, 0.0]
        )

    def test_orthogonal_cell_has_no_auxiliary_direction(self) -> None:
        rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        coefficients = pbc_grid_coefficients(np.diag([5.0, 6.0, 7.0]) @ rotation, [0.5, 0.5, 0.5])
        self.assertEqual(coefficients.number_of_directions, 0)
        np.testing.assert_array_equal(coefficients.coefficients, [1, 1, 1, 0, 0, 0])


class LaplacianTests(unittest.TestCase):
    def assert_plane_waves(self, grid, operator, waves, tolerance) -> None:
        for integers in waves:
            k, wave = _plane_wave(grid, integers)
            ratio = (operator @ wave) / wave
            # A plane wave is an eigenvector of the wrapped stencil ...
            np.testing.assert_allclose(ratio, ratio[0], atol=1e-9)
            # ... with eigenvalue |k|^2 to finite-difference accuracy.
            self.assertLess(abs(ratio[0].real - k @ k) / (k @ k), tolerance, msg=str(integers))

    def test_plane_waves_on_non_orthogonal_grids(self) -> None:
        cases = (
            (_FCC, 0.6, [(1, 0, 0), (1, 1, 0), (1, -1, 2)]),
            (_BCC, 0.6, [(1, 0, 0), (0, 1, 1), (1, 2, -1)]),
            (_HEXAGONAL, 0.6, [(1, 0, 0), (1, 1, 0), (0, 1, 1)]),
            # 60 degrees between a1 and a2: the u - v branch of pbc_grid_coefs_2d.
            (_HEXAGONAL_ACUTE, 0.55, [(1, 0, 0), (1, -1, 0), (0, 1, 1)]),
            (_TRICLINIC, 0.6, [(1, 0, 0), (1, -1, 1), (0, 1, 2)]),
        )
        for lattice, spacing, waves in cases:
            with self.subTest(lattice=lattice.tolist()):
                grid = _general_grid(lattice, spacing)
                self.assert_plane_waves(grid, build_negative_laplacian(grid), waves, 1e-3)

    def test_step_ratio_quirk_is_not_a_laplacian(self) -> None:
        grid = _general_grid(_TRICLINIC, 0.6)
        from parsec_python.Laplacian.nonorthogonal import build_general_negative_laplacian

        fixed = build_general_negative_laplacian(grid)
        quirk = build_general_negative_laplacian(
            grid,
            pbc_grid_coefficients(_TRICLINIC, grid.spacings, step_ratio_quirk=True),
        )
        k, wave = _plane_wave(grid, (0, 0, 1))
        error_fixed = abs(((fixed @ wave) / wave)[0].real - k @ k)
        error_quirk = abs(((quirk @ wave) / wave)[0].real - k @ k)
        self.assertLess(error_fixed, 1e-6)
        self.assertGreater(error_quirk, 1e3 * error_fixed)

    def test_orthorhombic_limit_equals_the_original_operator(self) -> None:
        cell = PeriodicCell(lattice_vectors=6.0 * np.eye(3))
        settings = PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4)
        original = build_periodic_grid(cell, settings)
        general = build_general_periodic_grid(cell, settings)
        np.testing.assert_array_equal(general.coordinates, original.coordinates)
        old_laplacian = build_negative_laplacian(original)
        new_laplacian = build_negative_laplacian(general)
        self.assertEqual((old_laplacian != 0).nnz, (new_laplacian != 0).nnz)
        np.testing.assert_allclose(new_laplacian.toarray(), old_laplacian.toarray(), rtol=1e-14, atol=1e-13)
        for old, new in zip(build_gradient(original), build_gradient(general)):
            np.testing.assert_allclose(new.toarray(), old.toarray(), rtol=1e-14, atol=1e-13)
        density = 0.2 * np.exp(-0.5 * np.sum((original.coordinates - 3.0) ** 2, axis=1)) + 0.01
        old_pbe, new_pbe = pbe(density, original), pbe(density, general)
        np.testing.assert_allclose(new_pbe.potential, old_pbe.potential, rtol=1e-12, atol=1e-13)
        self.assertAlmostEqual(new_pbe.total_energy, old_pbe.total_energy, places=12)
        old_spin = pbe_spin_polarized(0.6 * density, 0.4 * density, original)
        new_spin = pbe_spin_polarized(0.6 * density, 0.4 * density, general)
        np.testing.assert_allclose(new_spin.potential_up, old_spin.potential_up, rtol=1e-12, atol=1e-13)
        # The FFT Hartree solve of the general grid and the CG of the original.
        settings_tight = HartreeSettings(relative_tolerance=1e-12, absolute_tolerance=1e-14, max_iterations=5000)
        cg = solve_periodic_hartree(density, original, old_laplacian, settings_tight)
        fft = solve_periodic_hartree(density, general, new_laplacian, settings_tight)
        np.testing.assert_allclose(fft.potential, cg.potential, atol=1e-9)

    def test_gradient_of_plane_waves(self) -> None:
        grid = _general_grid(_TRICLINIC, 0.6)
        gradient = build_gradient(grid)
        k, wave = _plane_wave(grid, (1, -1, 1))
        derivative = np.array([((matrix @ wave) / wave).mean() for matrix in gradient])
        np.testing.assert_allclose(derivative.imag, k, rtol=2e-4, atol=1e-5)
        self.assertLess(np.max(np.abs(derivative.real)), 1e-10)


class HartreeAndXCTests(unittest.TestCase):
    def test_fft_solution_of_the_finite_difference_equation(self) -> None:
        grid = _general_grid(_FCC, 0.6)
        operator = build_negative_laplacian(grid)
        center = 0.5 * np.sum(_FCC, axis=0)
        delta = grid.coordinates - center
        density = np.exp(-np.sum(delta**2, axis=1))
        result = solve_periodic_hartree(density, grid, operator)
        self.assertTrue(result.converged)
        rhs = 8.0 * np.pi * (density - density.mean())
        np.testing.assert_allclose(operator @ result.potential, rhs, atol=1e-10 * np.abs(rhs).max())
        self.assertAlmostEqual(float(result.potential.mean()), 0.0, places=12)

    def test_pbe_potential_is_the_derivative_of_the_energy(self) -> None:
        grid = _general_grid(_HEXAGONAL, 0.7, order=6)
        center = 0.5 * np.sum(_HEXAGONAL, axis=0)
        delta = grid.coordinates - center
        density = 0.3 * np.exp(-0.4 * np.sum(delta**2, axis=1)) + 0.02
        probe = np.cos(grid.coordinates[:, 0]) * np.exp(-0.1 * np.sum(delta**2, axis=1))
        epsilon = 1e-6
        derivative = (
            pbe(density + epsilon * probe, grid).total_energy
            - pbe(density - epsilon * probe, grid).total_energy
        ) / (2 * epsilon)
        predicted = grid.volume_element * float(pbe(density, grid).potential @ probe)
        self.assertAlmostEqual(derivative, predicted, delta=1e-6 * abs(predicted))
        spin = pbe_spin_polarized(0.5 * density, 0.5 * density, grid)
        np.testing.assert_allclose(spin.potential_up, pbe(density, grid).potential, rtol=1e-8, atol=1e-10)


class KPointKineticTests(unittest.TestCase):
    def test_bloch_kinetic_energy_of_plane_waves(self) -> None:
        grid = _general_grid(_FCC, 0.6)
        laplacian = build_negative_laplacian(grid)
        gradient = build_gradient(grid)
        empty = SimpleNamespace(shape=(grid.size, grid.size), apply=lambda vectors: 0.0 * vectors)
        reciprocal = 2.0 * np.pi * np.linalg.inv(_FCC).T
        k = np.array([0.25, 0.5, 0.0]) @ reciprocal
        hamiltonian = KPointKohnShamHamiltonian(laplacian, gradient, np.zeros(grid.size), empty, k)
        for integers in ((0, 0, 0), (1, 0, 0), (0, -1, 1)):
            g, wave = _plane_wave(grid, integers)
            ratio = hamiltonian.apply(wave) / wave
            np.testing.assert_allclose(ratio, ratio[0], atol=1e-9)
            total = k + g
            self.assertLess(abs(ratio[0].real - total @ total), 2e-4 * (total @ total) + 1e-12)


class LatticeSumTests(unittest.TestCase):
    def test_madelung_constants_of_fcc_and_bcc(self) -> None:
        # Point charges in a neutralizing background: E = -M Z^2/r_s Hartree
        # per ion (Wigner-Seitz radius r_s; M = 0.895873615195 fcc,
        # 0.895929255682 bcc), returned in Rydberg.
        potentials = {"X": SimpleNamespace(ionic_charge=2.0)}
        for lattice, constant in ((_FCC, 0.895873615195), (_BCC, 0.895929255682)):
            volume = abs(np.linalg.det(lattice))
            radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            energy = ewald_ion_ion_energy([Atom("X", [1.0, 2.0, 0.5])], potentials, lattice)
            self.assertAlmostEqual(energy, -2.0 * constant * 4.0 / radius, places=9)

    def test_rocksalt_primitive_equals_conventional_cell(self) -> None:
        a = 10.0
        potentials = {"Na": SimpleNamespace(ionic_charge=1.0), "Cl": SimpleNamespace(ionic_charge=-1.0)}
        primitive = [Atom("Na", [0.0, 0.0, 0.0]), Atom("Cl", [a / 2, 0.0, 0.0])]
        conventional = []
        for corner in ([0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]):
            corner = a * np.asarray(corner)
            conventional.append(Atom("Na", corner))
            conventional.append(Atom("Cl", (corner + [a / 2, 0, 0]) % a))
        e_primitive = ewald_ion_ion_energy(primitive, potentials, 0.5 * a * (np.ones((3, 3)) - np.eye(3)))
        e_conventional = ewald_ion_ion_energy(conventional, potentials, a * np.eye(3))
        self.assertAlmostEqual(4.0 * e_primitive, e_conventional, places=9)

    def test_skewed_cell_energy_does_not_depend_on_eta(self) -> None:
        lattice = np.array([[6.0, 0.0, 0.0], [5.0, 3.0, 0.0], [4.5, 2.5, 2.0]])
        potentials = {"X": SimpleNamespace(ionic_charge=1.0), "Y": SimpleNamespace(ionic_charge=-1.0)}
        atoms = [Atom("X", [0.01, 0.01, 0.01]), Atom("Y", 0.98 * np.sum(lattice, axis=0))]
        energies = [
            ewald_ion_ion_energy(atoms, potentials, lattice, eta=eta) for eta in (0.3, 0.6, 1.2)
        ]
        self.assertLess(max(energies) - min(energies), 1e-10)

    def test_image_translations_reach_every_projector_image(self) -> None:
        lattice = _HEXAGONAL
        spacing = lattice_plane_spacings(lattice)
        self.assertTrue(np.all(spacing > 0))
        self.assertGreater(cell_diagonal_length(lattice), np.max(np.linalg.norm(lattice, axis=1)))
        radius = 3.0
        translations = _periodic_image_translations(lattice, radius)
        grid = _general_grid(lattice, 0.9, order=4)
        atom = np.array([0.1, 0.2, 0.3]) @ lattice
        brute = [
            np.array([i, j, k]) @ lattice
            for i in range(-6, 7) for j in range(-6, 7) for k in range(-6, 7)
        ]
        reaching = [
            t for t in brute if np.min(np.linalg.norm(grid.coordinates - atom - t, axis=1)) <= radius
        ]
        listed = {tuple(np.round(t, 9)) for t in translations}
        for t in reaching:
            self.assertIn(tuple(np.round(t, 9)), listed)

    def test_projectors_of_an_oblique_cell_match_a_brute_force_image_sum(self) -> None:
        specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
        potentials = load_pseudopotentials(specifications, xc_functional="ca")
        lattice = np.array([[3.0, 0.0, 0.0], [1.5, 2.6, 0.0], [0.4, 0.3, 3.1]])
        grid = _general_grid(lattice, 0.3, order=4)
        atoms = [Atom("H", np.array([0.2, 0.7, 0.4]) @ lattice)]
        operator = build_nonlocal_projectors(grid, atoms, potentials, specifications, lattice)
        potential = potentials["H"]
        local_l = specifications["H"].local_angular_momentum
        channels = sorted(set(potential.radial_wavefunctions) - {local_l})
        if not channels:
            self.skipTest("the test pseudopotential has no nonlocal channel")
        angular_momentum = channels[0]
        radial_grid, _ = potential.radial_projector(angular_momentum, local_l)
        support_radius = _projector_support_radius(potential)
        column = next(
            index for index, (_a, l, m) in enumerate(operator.labels) if l == angular_momentum and m == 0
        )
        computed = np.asarray(operator.projectors[:, column].todense()).ravel()
        brute = np.zeros(grid.size)
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    relative = grid.coordinates - atoms[0].position - np.array([i, j, k]) @ lattice
                    radius = np.linalg.norm(relative, axis=1)
                    support = radius <= support_radius
                    if not np.any(support):
                        continue
                    radial = np.interp(radius[support], potential.radii, radial_grid,
                                       left=radial_grid[0], right=0.0)
                    harmonics = real_spherical_harmonics(angular_momentum, relative[support])
                    brute[support] += np.sqrt(grid.volume_element) * radial * harmonics[:, 0]
        np.testing.assert_allclose(computed, brute, atol=1e-14)
        self.assertGreater(np.max(np.abs(computed)), 0.0)

    def test_monkhorst_pack_of_a_general_cell(self) -> None:
        cell = PeriodicCell(lattice_vectors=_FCC)
        points, weights = monkhorst_pack_grid(cell, (2, 1, 3))
        reciprocal = 2.0 * np.pi * np.linalg.inv(_FCC).T
        fractions = points @ _FCC.T / (2.0 * np.pi)
        np.testing.assert_allclose(np.sort(np.unique(np.round(fractions[:, 0], 12))), [-0.25, 0.25])
        np.testing.assert_allclose(np.unique(np.round(fractions[:, 1], 12)), [0.0])
        np.testing.assert_allclose(np.sort(np.unique(np.round(fractions[:, 2], 12))), [-1 / 3, 0.0, 1 / 3])
        np.testing.assert_allclose(points, fractions @ reciprocal, atol=1e-12)
        np.testing.assert_allclose(weights, 1.0 / 6.0)


class AtomsOutsideTheCellTests(unittest.TestCase):
    def test_wrap_keeps_atoms_inside_and_translates_the_others(self) -> None:
        cubic = 6.0 * np.eye(3)
        inside = np.array([0.0, 5.999, 3.0])
        self.assertIs(wrap_into_cell(inside, cubic), inside)
        np.testing.assert_allclose(wrap_into_cell(np.array([-0.5, 6.0, 15.0]), cubic), [5.5, 0.0, 3.0])
        position = np.array([-1.0, -2.0, 7.0])
        wrapped = wrap_into_cell(position, _FCC)
        fractions = np.linalg.solve(_FCC.T, wrapped)
        self.assertTrue(np.all(fractions >= 0.0) and np.all(fractions < 1.0))
        np.testing.assert_allclose(
            np.round(np.linalg.solve(_FCC.T, position - wrapped), 12) % 1.0, 0.0, atol=1e-12
        )

    def test_prepared_system_equals_the_one_with_the_translated_atom(self) -> None:
        def problem(position):
            return SinglePointInput(
                atoms=[Atom("H", position)],
                pseudopotentials={"H": SpeciesPotential(_H_POTRE, 0)},
                grid=PeriodicGridSettings(fortran_routines=True, spacing=0.6, expansion_order=4),
                periodic_cell=PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
            )

        outside = prepare_periodic_single_point(problem([-0.5, 3.0, 9.0]))
        inside = prepare_periodic_single_point(problem([5.5, 3.0, 3.0]))
        np.testing.assert_array_equal(outside.atoms[0].position, [5.5, 3.0, 3.0])
        np.testing.assert_array_equal(outside.ionic_potential, inside.ionic_potential)
        np.testing.assert_array_equal(outside.initial_density, inside.initial_density)
        self.assertEqual(outside.ion_ion_energy, inside.ion_ion_energy)


_SUPER_CELL_INPUT = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  3.0  3.0  3.0
end Cell_Shape
{super}
Coordinate_Unit: {unit}
Grid_Spacing: 0.6 bohr
Expansion_Order: 4
States_Num: 4
Ignore_Symmetry: .true.
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
Move_Flag: 1
Minimization: none
begin Atom_Coord
{coordinates}
end Atom_Coord
Correlation_Type: ca
"""


class SuperCellInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        shutil.copy(_H_POTRE, Path(self._directory.name) / "H_POTRE.DAT")

    def parse(self, text: str):
        path = Path(self._directory.name) / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return parse_parsec_input(path)

    def test_replication_and_lattice_vector_coordinates(self) -> None:
        translation = self.parse(
            _SUPER_CELL_INPUT.format(
                super="begin Super_Cell\n  2 1 3\nend Super_Cell\n"
                "begin Super_Cell_Vac\n  0 1 0\nend Super_Cell_Vac",
                unit="lattice_vectors",
                coordinates="  0.5 0.5 0.5\n  0.0 0.25 0.0",
            )
        )
        problem = translation.problem
        np.testing.assert_array_equal(problem.periodic_cell.lattice_vectors, np.diag([6.0, 6.0, 9.0]))
        self.assertEqual(len(problem.atoms), 12)
        expected = []
        for base in ([1.5, 1.5, 1.5], [0.0, 0.75, 0.0]):
            expected.extend(replicate_positions(np.array(base), 3.0 * np.eye(3), (2, 1, 3)))
        # translated about the cell centre at start, as PARSEC 1.5 does (back_to_cell; Fortran-derived routines)
        expected = [back_to_cell(position, np.diag([6.0, 6.0, 9.0])) for position in expected]
        np.testing.assert_array_equal([atom.position for atom in problem.atoms], expected)
        # i3 runs fastest, as in usrinputfile.F90 (z = 4.5, half the cell: -4.5 after back_to_cell).
        np.testing.assert_array_equal(problem.atoms[1].position, [1.5, 1.5, -4.5])
        self.assertEqual(translation.super_cell, (2, 1, 3))
        self.assertEqual(translation.super_cell_vacuum, (0, 1, 0))
        self.assertEqual(len(translation.movable), 12)

    def test_super_cell_input_equals_the_explicit_input(self) -> None:
        replicated = self.parse(
            _SUPER_CELL_INPUT.format(
                super="begin Super_Cell\n  2 2 1\nend Super_Cell",
                unit="lattice_vectors",
                coordinates="  0.5 0.5 0.5",
            )
        ).problem
        explicit = self.parse(
            _SUPER_CELL_INPUT.replace("3.0  3.0  3.0", "6.0  6.0  3.0").format(
                super="",
                unit="Cartesian_Bohr",
                coordinates="\n".join(
                    f"  {x!r} {y!r} {z!r}"
                    for x, y, z in ([1.5, 1.5, 1.5], [1.5, 4.5, 1.5], [4.5, 1.5, 1.5], [4.5, 4.5, 1.5])
                ),
            )
        ).problem
        np.testing.assert_array_equal(
            replicated.periodic_cell.lattice_vectors, explicit.periodic_cell.lattice_vectors
        )
        np.testing.assert_array_equal(
            [atom.position for atom in replicated.atoms], [atom.position for atom in explicit.atoms]
        )
        self.assertEqual(replicated.grid, explicit.grid)
        first = prepare_periodic_single_point(replicated)
        second = prepare_periodic_single_point(explicit)
        np.testing.assert_array_equal(first.ionic_potential, second.ionic_potential)
        np.testing.assert_array_equal(first.initial_density, second.initial_density)

    def test_input_errors_and_cluster_warning(self) -> None:
        for block in (
            "begin Super_Cell\n  2 2\nend Super_Cell",
            "begin Super_Cell\n  2 2 0\nend Super_Cell",
            "begin Super_Cell\n  2 2 2\n  1 1 1\nend Super_Cell",
            "begin Super_Cell\n  2.5 2 2\nend Super_Cell",
            "begin Super_Cell_Vac\n  0 -1 0\nend Super_Cell_Vac",
        ):
            with self.subTest(block=block), self.assertRaises(ParsecInputError):
                self.parse(_SUPER_CELL_INPUT.format(super=block, unit="lattice_vectors", coordinates="  0 0 0"))
        cluster = (
            _SUPER_CELL_INPUT.replace("Periodic_System: .true.\nUse_Fortran_Routines: .true.\nBoundary_Conditions: bulk\n", "Boundary_Sphere_Radius: 4.0\n")
            .replace("begin Cell_Shape\n  3.0  3.0  3.0\nend Cell_Shape\n", "")
        )
        translation = self.parse(
            cluster.format(super="begin Super_Cell\n  2 2 2\nend Super_Cell", unit="Cartesian_Bohr", coordinates="  0 0 0")
        )
        self.assertEqual(len(translation.problem.atoms), 1)
        self.assertTrue(any("Super_Cell" in warning for warning in translation.warnings))
        with self.assertRaisesRegex(ParsecInputError, "lattice_vectors"):
            self.parse(cluster.format(super="", unit="lattice_vectors", coordinates="  0 0 0"))

    def test_non_orthogonal_input_and_band_path(self) -> None:
        text = (
            _SUPER_CELL_INPUT.replace("  3.0  3.0  3.0\n", "  0.0 2.0 2.0\n  2.0 0.0 2.0\n  2.0 2.0 0.0\n")
            .format(super="", unit="lattice_vectors", coordinates="  0 0 0\n  0.25 0.25 0.25")
            + "begin bandstruc\n1 0 0 0 0.5 0 0.5 G-X\nend bandstruc\nbandstruc_points 4\n"
        )
        translation = self.parse(text)
        problem = translation.problem
        np.testing.assert_array_equal(problem.atoms[1].position, [1.0, 1.0, 1.0])
        self.assertEqual(problem.grid.shift, (0.0, 0.0, 0.0))
        self.assertIsNotNone(problem.bands)
        from parsec_python.acceleration.cli import _reference_only_features

        features = _reference_only_features(translation, accelerated_backend_selected=False)
        self.assertTrue(any("general grid" in feature for feature in features))


if __name__ == "__main__":
    unittest.main()
