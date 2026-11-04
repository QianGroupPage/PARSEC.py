"""Wire (1-D periodic) and slab (2-D periodic) boundary conditions.

Covers :mod:`parsec_python.Grid.confined`, the Hartree boundary values of
:mod:`parsec_python.Hartree.wire` / :mod:`parsec_python.Hartree.slab` and the
CG solve of :mod:`parsec_python.Hartree.confined`, the local potential,
ion-ion and charged-sheet terms of :mod:`parsec_python.V_ion.wire_slab`, the
forces of :mod:`parsec_python.Forces.wire_slab`, the input of
``Boundary_Conditions: wire/slab`` and small SCF runs.  Fortran numbers are
from PARSEC runs of the inputs named.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import exp1, k0

from parsec_python.Grid.confined import (
    ConfinedPeriodicGrid,
    build_confined_periodic_grid,
    confined_lattice_vectors,
)
from parsec_python.Hartree.confined import (
    exterior_stencil,
    plan_confined_hartree,
    solve_confined_hartree,
)
from parsec_python.Hartree.wire import LN2_MINUS_EULER
from parsec_python.Input.parsec_input import (
    ParsecInputError,
    UnsupportedParsecOptionError,
    parse_parsec_input,
)
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.models import (
    Atom,
    ChargedSheet,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.V_ion.wire_slab import (
    EULER_GAMMA,
    charged_sheet_ion_energy_and_forces,
    slab_ewald,
    wire_ion_ion_energy_and_forces,
    wire_pair_energy_and_force,
)

DATA = Path(__file__).resolve().parents[1] / "data"
HYDROGEN = DATA / "water_H_POTRE.DAT"
CHARGES = {"Na": SimpleNamespace(ionic_charge=1.0), "C": SimpleNamespace(ionic_charge=4.0), "H": SimpleNamespace(ionic_charge=1.0)}
GRAPHENE_A = 4.6487
HEXAGONAL = np.array([[1.0, 0.0, 0.0], [0.5, 0.8660254037844386, 0.0]]) * GRAPHENE_A


def wire_grid(length=10.0, radius=12.0, spacing=0.7, shift=(0.5, 0.5, 0.5), order=12):
    cell = PeriodicCell(confined_lattice_vectors([[length, 0.0, 0.0]]))
    settings = PeriodicGridSettings(
        spacing=spacing, shift=shift, expansion_order=order, periodicity=1, confinement_radius=radius
    )
    return build_confined_periodic_grid(cell, settings)


def slab_grid(vectors=HEXAGONAL, radius=6.0, spacing=0.5, shift=(0.0, 0.0, 0.5), order=12):
    cell = PeriodicCell(confined_lattice_vectors(vectors))
    settings = PeriodicGridSettings(
        spacing=spacing, shift=shift, expansion_order=order, periodicity=2, confinement_radius=radius
    )
    return build_confined_periodic_grid(cell, settings)


def finite_difference_forces(energy, atoms, step=1.0e-5):
    forces = np.zeros((len(atoms), 3))
    for index, atom in enumerate(atoms):
        for axis in range(3):
            moved = []
            for sign in (1.0, -1.0):
                position = np.array(atom.position, dtype=float)
                position[axis] += sign * step
                shifted = list(atoms)
                shifted[index] = Atom(atom.symbol, position)
                moved.append(energy(shifted))
            forces[index, axis] = -(moved[0] - moved[1]) / (2.0 * step)
    return forces


class GridTests(unittest.TestCase):
    def test_wire_grid_is_fortrans(self) -> None:
        # examples/tests/chain_Na with R = 12: Fortran prints "Final grid
        # spacings: 0.714 0.700 0.700", 14 36 36 points, matrix size 12992.
        grid = wire_grid()
        self.assertIsInstance(grid, ConfinedPeriodicGrid)
        self.assertEqual(grid.box_shape, (14, 36, 36))
        self.assertEqual(grid.size, 12992)
        np.testing.assert_allclose(grid.spacings, [10.0 / 14, 0.7, 0.7])
        np.testing.assert_array_equal(grid.index_min, [-7, -18, -18])
        self.assertAlmostEqual(grid.volume_element, 10.0 / 14 * 0.49)
        radial = np.hypot(grid.coordinates[:, 1], grid.coordinates[:, 2])
        self.assertLessEqual(radial.max(), 12.0)
        # Fortran's order: i, then j, then k from their largest values.
        self.assertEqual(grid.integer_coordinates[0, 0], 6)
        self.assertTrue(np.all(np.diff(grid.integer_coordinates[:, 0]) <= 0))
        first = grid.integer_coordinates[grid.integer_coordinates[:, 0] == 6]
        self.assertTrue(np.all(np.diff(first[:, 1]) <= 0))
        with self.assertRaises(AttributeError):
            _ = grid.spacing

    def test_lookup_wraps_periodic_and_misses_confined(self) -> None:
        grid = wire_grid()
        point = grid.integer_coordinates[100]
        moved = np.array([point + [14, 0, 0], point - [28, 0, 0]])
        np.testing.assert_array_equal(grid.rows_for_integer_coordinates(moved), [100, 100])
        outside = np.array([[0, 40, 0], [0, 0, -30]])
        np.testing.assert_array_equal(grid.rows_for_integer_coordinates(outside), [-1, -1])
        np.testing.assert_allclose(
            grid.physical_coordinates(point[None, :])[0], grid.coordinates[100]
        )

    def test_slab_grid_is_fortrans(self) -> None:
        # Graphene-like slab: Fortran prints spacings 0.581 0.581 0.500,
        # 8 8 26 points, matrix size 1536 and one Laplacian direction.
        grid = slab_grid()
        self.assertEqual(grid.box_shape, (8, 8, 26))
        self.assertEqual(grid.size, 1536)
        np.testing.assert_allclose(grid.spacings[:2], GRAPHENE_A / 8)
        self.assertAlmostEqual(grid.surface_area, GRAPHENE_A**2 * 0.8660254037844386)
        self.assertAlmostEqual(grid.volume_element * 8 * 8 / grid.spacings[2], grid.surface_area)
        self.assertTrue(np.all(np.abs(grid.coordinates[:, 2]) <= 6.0))

    def test_laplacian_has_no_entry_outside_and_is_exact_for_quadratics(self) -> None:
        grid = slab_grid()
        laplacian = build_negative_laplacian(grid)
        self.assertEqual(laplacian.shape, (grid.size, grid.size))
        np.testing.assert_allclose((laplacian - laplacian.T).toarray(), 0.0, atol=1e-12)
        # A periodic plane wave times z^2 away from the boundary planes.
        interior = np.abs(grid.coordinates[:, 2]) < 2.0
        b = 2.0 * np.pi * np.linalg.inv(grid.cell.lattice_vectors).T
        field = np.cos(grid.coordinates @ b[0]) * grid.coordinates[:, 2] ** 2
        expected = (b[0] @ b[0]) * field - 2.0 * np.cos(grid.coordinates @ b[0])
        # Order 12 at h = 0.58 bohr: the stencil error of this plane wave is ~1e-6.
        np.testing.assert_allclose((laplacian @ field)[interior], expected[interior], atol=2e-5)

    def test_image_translations_reach_every_image(self) -> None:
        grid = slab_grid()
        position = np.array([10.0, -7.0, 0.3])
        radius = 3.5
        translations = grid.image_translations(position, radius)
        brute = []
        for m1 in range(-10, 11):
            for m2 in range(-10, 11):
                shift = m1 * HEXAGONAL[0] + m2 * HEXAGONAL[1]
                distance = np.linalg.norm(grid.coordinates - (position + shift), axis=1)
                if np.any(distance <= radius):
                    brute.append(shift)
        listed = {tuple(np.round(t, 8)) for t in translations}
        for shift in brute:
            self.assertIn(tuple(np.round(shift, 8)), listed)


class ExchangeCorrelationTests(unittest.TestCase):
    def test_every_functional_on_a_wire_grid_is_that_of_a_cluster_grid(self) -> None:
        # A localized density: the wire grid (period 12 = 24 steps of 0.5)
        # holds the points of a cluster grid of the same spacing and shift,
        # and every functional, with its gradients, must agree there.
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings
        from parsec_python.V_xc import ca_lda_spin_polarized, pbe_spin_polarized
        from parsec_python.V_xc.selection import (
            UNPOLARIZED_ONLY_FUNCTIONALS,
            XC_FUNCTIONALS,
            evaluate_spin_xc,
            evaluate_unpolarized_xc,
        )

        wire = wire_grid(length=12.0, radius=6.0, spacing=0.5)
        cluster = build_cluster_grid(GridSettings(spacing=0.5, radius=6.0, shift=(0.5, 0.5, 0.5)))

        def density(points):
            return 0.2 * np.exp(-np.sum(points**2, axis=1)) + 0.05 * np.exp(
                -np.sum((points - [0.7, 0.2, 0.0]) ** 2, axis=1) / 0.5
            )

        index = {tuple(np.round(point, 6)): row for row, point in enumerate(cluster.coordinates)}
        rows = np.array([index.get(tuple(np.round(point, 6)), -1) for point in wire.coordinates])
        central = (rows >= 0) & (np.linalg.norm(wire.coordinates, axis=1) < 3.0)
        on_wire, on_cluster = density(wire.coordinates), density(cluster.coordinates)
        for functional in XC_FUNCTIONALS:
            with self.subTest(functional=functional):
                a = evaluate_unpolarized_xc(functional, on_wire, wire)
                b = evaluate_unpolarized_xc(functional, on_cluster, cluster)
                self.assertAlmostEqual(a.total_energy, b.total_energy, places=12)
                np.testing.assert_allclose(a.potential[central], b.potential[rows[central]], atol=1e-12)
            if functional in UNPOLARIZED_ONLY_FUNCTIONALS:
                continue
            with self.subTest(functional=functional, spin=True):
                if functional in ("ca", "pbe"):
                    evaluate = ca_lda_spin_polarized if functional == "ca" else pbe_spin_polarized
                    if functional == "ca":
                        a = evaluate(0.6 * on_wire, 0.4 * on_wire, wire.volume_element)
                        b = evaluate(0.6 * on_cluster, 0.4 * on_cluster, cluster.volume_element)
                    else:
                        a = evaluate(0.6 * on_wire, 0.4 * on_wire, wire)
                        b = evaluate(0.6 * on_cluster, 0.4 * on_cluster, cluster)
                    potentials = (a.potential_up, b.potential_up)
                else:
                    a = evaluate_spin_xc(functional, 0.6 * on_wire, 0.4 * on_wire, wire)
                    b = evaluate_spin_xc(functional, 0.6 * on_cluster, 0.4 * on_cluster, cluster)
                    potentials = (a.potential_up, b.potential_up)
                self.assertAlmostEqual(a.total_energy, b.total_energy, places=12)
                np.testing.assert_allclose(
                    potentials[0][central], potentials[1][rows[central]], atol=1e-12
                )


class HartreeTests(unittest.TestCase):
    def test_wire_boundary_is_the_periodic_line_green_function(self) -> None:
        grid = wire_grid()
        plan = plan_confined_hartree(grid, 9)
        # A unit charge at one grid point near the axis.
        source = int(np.argmin(np.hypot(grid.coordinates[:, 1] - 0.3, grid.coordinates[:, 2] - 0.2)))
        density = np.zeros(grid.size)
        density[source] = 1.0 / grid.volume_element
        values = plan.boundary_values(density)
        points = plan.stencil.coordinates
        r = grid.coordinates[source]
        rho = np.hypot(points[:, 1] - r[1], points[:, 2] - r[2])
        dx = points[:, 0] - r[0]
        a = 10.0
        exact = (2.0 / a) * (np.log(2.0 * a / rho) - EULER_GAMMA)
        for n in range(1, 40):
            exact += (4.0 / a) * k0(n * 2.0 * np.pi / a * rho) * np.cos(n * 2.0 * np.pi / a * dx)
        np.testing.assert_allclose(values, 2.0 * exact, atol=1e-7)

    def test_wire_hartree_of_a_gaussian_line(self) -> None:
        grid = wire_grid(radius=8.0, spacing=0.5)
        plan = plan_confined_hartree(grid, 9)
        width = 1.2
        rho2 = grid.coordinates[:, 1] ** 2 + grid.coordinates[:, 2] ** 2
        density = np.exp(-rho2 / width**2) / (np.pi * width**2)  # one electron per bohr
        line = np.sum(density) * grid.volume_element / grid.period
        solution = solve_confined_hartree(
            density, plan, build_negative_laplacian(grid), HartreeSettings(relative_tolerance=1e-12)
        )
        # V = 4 lambda (ln2 - gamma - ln(rho/a)) - 2 lambda E1(rho^2/w^2) (Ry).
        exact = 4.0 * line * (LN2_MINUS_EULER - 0.5 * np.log(rho2 / grid.period**2)) - 2.0 * line * exp1(
            rho2 / width**2
        )
        # The finite-difference error of the Gaussian is ~1e-6 relative.
        np.testing.assert_allclose(solution.potential, exact, rtol=1e-5, atol=1e-5)

    def test_slab_hartree_of_a_gaussian_sheet(self) -> None:
        grid = slab_grid(radius=7.0)
        plan = plan_confined_hartree(grid, 9)
        width = 1.0
        z = grid.coordinates[:, 2] - 0.4
        density = np.exp(-(z**2) / width**2) / (np.sqrt(np.pi) * width)  # one electron per bohr^2
        sheet = np.sum(density) * grid.volume_element / grid.surface_area
        solution = solve_confined_hartree(
            density, plan, build_negative_laplacian(grid), HartreeSettings(relative_tolerance=1e-12)
        )
        # V = -4 pi sigma [z erf(z/w) + w exp(-z^2/w^2)/sqrt(pi)] (Ry).
        from scipy.special import erf

        exact = -4.0 * np.pi * sheet * (z * erf(z / width) + width * np.exp(-(z**2) / width**2) / np.sqrt(np.pi))
        # The finite-difference error of the Gaussian is ~1e-5 relative.
        np.testing.assert_allclose(solution.potential, exact, rtol=5e-5, atol=5e-5)

    def test_slab_boundary_matches_fortrans_sums(self) -> None:
        # A literal transcription of hartset_slab.F90's loops.
        grid = slab_grid(radius=3.0, spacing=0.6)
        order = 3
        plan = plan_confined_hartree(grid, order)
        rng = np.random.default_rng(3)
        density = rng.random(grid.size)
        values = plan.boundary_values(density)
        lattice = grid.cell.lattice_vectors
        bvec = 2.0 * np.pi * np.linalg.inv(lattice).T
        area = grid.surface_area
        pairs = [(l, m) for l in range(-order, order + 1) for m in range(1, order + 1)] + [
            (l, 0) for l in range(1, order + 1)
        ]
        wave = 2.0 * np.pi / np.linalg.norm(lattice[:2], axis=1)
        u = (grid.integer_coordinates[:, :2] + grid.shift[:2]) * grid.spacings[:2]
        kz = grid.integer_coordinates[:, 2]
        planes = np.unique(kz)
        expected = []
        for row, point in zip(plan.stencil.rows, plan.stencil.integer_points):
            ub = (grid.wrap_periodic(point[None, :])[0, :2] + grid.shift[:2]) * grid.spacings[:2]
            total = 0.0
            for plane in planes:
                inside = kz == plane
                separation = abs(point[2] - plane) * grid.spacings[2]
                total += -2.0 * np.pi * grid.spacings[2] * np.mean(density[inside]) * separation
                for l, m in pairs:
                    length = np.linalg.norm(l * bvec[0] + m * bvec[1])
                    phase = l * wave[0] * u[inside, 0] + m * wave[1] * u[inside, 1]
                    phase_b = l * wave[0] * ub[0] + m * wave[1] * ub[1]
                    qc = 2.0 * np.pi / (area * length) * np.sum(density[inside] * np.cos(phase))
                    qs = 2.0 * np.pi / (area * length) * np.sum(density[inside] * np.sin(phase))
                    total += 2.0 * grid.volume_element * np.exp(-length * separation) * (
                        np.cos(phase_b) * qc + np.sin(phase_b) * qs
                    )
            expected.append(2.0 * total)
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)

    def test_slab_lowest_plane_quirk(self) -> None:
        # int(2*6.3/0.5) = 25 is odd: the lowest plane of the box holds points.
        grid = slab_grid(radius=6.3)
        local = grid.integer_coordinates - grid.index_min
        self.assertTrue(np.any(local[:, 2] == 0))
        density = np.where(local[:, 2] == 0, 1.0, 0.0)
        fixed = plan_confined_hartree(grid, 4).boundary_values(density)
        fortran = plan_confined_hartree(grid, 4, fortran_quirks={"slab_lowest_plane"}).boundary_values(density)
        self.assertGreater(np.max(np.abs(fixed)), 1e-3)
        np.testing.assert_allclose(fortran, 0.0, atol=1e-14)

    def test_only_confined_neighbours_are_exterior(self) -> None:
        wire = exterior_stencil(wire_grid())
        radial = np.hypot(wire.coordinates[:, 1], wire.coordinates[:, 2])
        self.assertTrue(np.all(radial > 12.0))
        slab = exterior_stencil(slab_grid())
        self.assertTrue(np.all(np.abs(slab.coordinates[:, 2]) > 6.0))


class IonIonTests(unittest.TestCase):
    def test_wire_ion_ion_matches_fortran(self) -> None:
        # examples/tests/chain_Na: Fortran "Ion-Ion Energy = 1.14019152 [Ry]".
        atoms = [
            Atom("Na", (0.0, 2.931429, 0.0)),
            Atom("Na", (0.0, -2.931429, 0.0)),
            Atom("Na", (0.0, 0.0, 5.890929)),
            Atom("Na", (0.0, 0.0, -5.890929)),
        ]
        energy, forces = wire_ion_ion_energy_and_forces(atoms, CHARGES, 10.0)
        self.assertAlmostEqual(energy, 1.14019152, places=8)
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)

    def test_wire_forces_are_the_derivative(self) -> None:
        atoms = [
            Atom("Na", (0.3, 2.9, 0.2)),
            Atom("Na", (-0.4, -2.7, 0.0)),
            Atom("Na", (4.8, 0.0, -5.9)),
            Atom("Na", (1.1, 0.0, -5.9)),  # on one axial line with the third
        ]
        _, forces = wire_ion_ion_energy_and_forces(atoms, CHARGES, 10.0)
        numeric = finite_difference_forces(
            lambda moved: wire_ion_ion_energy_and_forces(moved, CHARGES, 10.0)[0], atoms
        )
        np.testing.assert_allclose(forces, numeric, atol=1e-7)

    def test_on_axis_pair_is_the_limit(self) -> None:
        on_axis, _ = wire_pair_energy_and_force(np.array([2.3, 0.0, 0.0]), 10.0, 1.0)
        near, _ = wire_pair_energy_and_force(np.array([2.3, 1.0e-3, 0.0]), 10.0, 1.0)
        self.assertAlmostEqual(on_axis, near, places=6)
        # 2 Z Z' S with S = lim [sum_(|m|<=M) 1/|x + m a| - (2/a) ln M] - 2 gamma/a,
        # the convention of the K_0 series (and of the wire Hartree boundary).
        count = 400000
        m = np.arange(-count, count + 1, dtype=np.float64)
        direct = np.sum(1.0 / np.abs(2.3 + 10.0 * m)) - 0.2 * np.log(count)
        self.assertAlmostEqual(on_axis, 2.0 * (direct - 2.0 * EULER_GAMMA / 10.0), places=5)

    def test_slab_ewald_matches_fortran_and_does_not_depend_on_eta(self) -> None:
        lattice = confined_lattice_vectors(HEXAGONAL)
        atoms = [
            Atom("C", (HEXAGONAL[0] + HEXAGONAL[1]) / 3.0),
            Atom("C", 2.0 * (HEXAGONAL[0] + HEXAGONAL[1]) / 3.0 + np.array([0.0, 0.0, 0.15])),
        ]
        terms = slab_ewald(atoms, CHARGES, lattice)
        # Fortran: "Ion-Ion Energy = -39.70262050 [Ry]", Fz = -+1.102467.
        self.assertAlmostEqual(terms.energy, -39.70262050, places=7)
        np.testing.assert_allclose(terms.forces[:, 2], [-1.102467, 1.102467], atol=1e-6)
        for eta in (0.3, 1.0):
            self.assertAlmostEqual(slab_ewald(atoms, CHARGES, lattice, eta=eta).energy, terms.energy, places=10)
        numeric = finite_difference_forces(lambda moved: slab_ewald(moved, CHARGES, lattice).energy, atoms)
        np.testing.assert_allclose(terms.forces, numeric, atol=1e-7)

    def test_slab_ewald_far_layers_are_charged_sheets(self) -> None:
        lattice = confined_lattice_vectors([[3.0, 0.0, 0.0], [0.0, 3.5, 0.0]])
        near = slab_ewald([Atom("H", (0, 0, 0)), Atom("H", (0.5, 1.0, 10.0))], CHARGES, lattice).energy
        far = slab_ewald([Atom("H", (0, 0, 0)), Atom("H", (0.5, 1.0, 12.0))], CHARGES, lattice).energy
        # Two sheets of one ion per cell: E changes by -4 pi/A per bohr.
        self.assertAlmostEqual(far - near, -4.0 * np.pi / 10.5 * 2.0, places=6)

    def test_charged_sheets(self) -> None:
        sheets = (ChargedSheet(0.05, -4.5), ChargedSheet(-0.05, 4.0))
        atoms = [Atom("C", (0.0, 0.0, 0.0)), Atom("C", (1.0, 0.5, 0.15))]
        area = 18.7
        energy, forces = charged_sheet_ion_energy_and_forces(atoms, CHARGES, sheets, area)
        expected = -4.0 * np.pi * 4.0 / area * (0.05 * (4.5 + 4.65) - 0.05 * (4.0 + 3.85))
        expected -= 2.0 * 2.0 * np.pi * 0.05 * (-0.05) / area * 8.5
        self.assertAlmostEqual(energy, expected, places=12)
        numeric = finite_difference_forces(
            lambda moved: charged_sheet_ion_energy_and_forces(moved, CHARGES, sheets, area)[0], atoms
        )
        np.testing.assert_allclose(forces, numeric, atol=1e-7)
        _, fortran = charged_sheet_ion_energy_and_forces(atoms, CHARGES, sheets, area, fortran_force=True)
        np.testing.assert_allclose(fortran * 2.0 * np.pi, forces, atol=1e-12)


INPUT_HEAD = """
Boundary_Conditions: {boundary}
begin Cell_Shape
{cell}
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
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def parse(self, boundary="wire", cell=" 3.0", extra=""):
        path = self.directory / "parsec.in"
        path.write_text(INPUT_HEAD.format(boundary=boundary, cell=cell, extra=extra), encoding="utf-8")
        return parse_parsec_input(path)

    def test_wire_and_slab_inputs(self) -> None:
        for label in ("wire", "tube", "1d"):
            problem = self.parse(label).problem
            self.assertEqual(problem.grid.periodicity, 1)
            self.assertEqual(problem.grid.confinement_radius, 5.0)
            np.testing.assert_allclose(problem.periodic_cell.lattice_vectors, np.diag([3.0, 1.0, 1.0]))
        for cell in (" 3.0 0.0\n 0.0 3.5", " 3.0 0.0 0.0\n 0.0 3.5 0.0"):
            problem = self.parse("2d", cell).problem
            self.assertEqual(problem.grid.periodicity, 2)
            np.testing.assert_allclose(problem.periodic_cell.lattice_vectors, np.diag([3.0, 3.5, 1.0]))
            self.assertEqual(problem.grid.shift, (0.5, 0.5, 0.5))
        hexagonal = self.parse("slab", " 3.0 0.0\n 1.5 2.598").problem
        self.assertEqual(hexagonal.grid.shift, (0.0, 0.0, 0.5))
        scaled = self.parse("wire", " 3.0", "Lattice_Vector_Scale: 2.0").problem
        self.assertEqual(scaled.periodic_cell.lattice_vectors[0, 0], 6.0)
        ignored = self.parse("wire", " 3.0", "Ignore_Symmetry: .true.").problem
        self.assertEqual(ignored.grid.shift, (0.0, 0.0, 0.0))

    def test_bulk_settings_text_is_unchanged(self) -> None:
        settings = PeriodicGridSettings(spacing=0.5)
        self.assertNotIn("periodicity", repr(settings))
        self.assertNotIn("confinement_radius", repr(settings))
        wire = replace(settings, periodicity=1, confinement_radius=4.0)
        self.assertIn("periodicity=1", repr(wire))

    def test_cell_and_radius_errors(self) -> None:
        for boundary, cell, extra, message in (
            ("slab", " 3.0 0.0 0.0", "", "has 2 line"),
            ("slab", " 3.0 0.0 1.0\n 0.0 3.5 0.0", "", "third component"),
            ("wire", " 3.0 1.0", "", "one number"),
            ("wire", " -3.0", "", "positive"),
            ("pencil", " 3.0", "", "Boundary_Conditions='pencil' is unknown"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ParsecInputError, message):
                    self.parse(boundary, cell, extra)
        path = self.directory / "parsec.in"
        path.write_text(
            INPUT_HEAD.format(boundary="wire", cell=" 3.0", extra="").replace(
                "Boundary_Sphere_Radius: 5.0\n", ""
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ParsecInputError, "Boundary_Sphere_Radius is required"):
            parse_parsec_input(path)

    def test_kpoints_along_the_periodic_directions(self) -> None:
        mp = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 4\nend Monkhorst_Pack_Grid"
        self.assertEqual(self.parse("wire", " 3.0", mp).problem.monkhorst_pack_dimensions, (4, 1, 1))
        mp2 = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n 2 3 5\nend Monkhorst_Pack_Grid"
        translation = self.parse("slab", " 3.0 0.0\n 0.0 3.5", mp2)
        self.assertEqual(translation.problem.monkhorst_pack_dimensions, (2, 3, 1))
        self.assertTrue(any("ignored, as in PARSEC" in note for note in translation.warnings))
        manual = "Kpoint_Method: manual\nbegin Kpoint_List\n 0.1 0.2 0.0 1.0\nend Kpoint_List"
        with self.assertRaisesRegex(ParsecInputError, "confined"):
            self.parse("wire", " 3.0", manual)

    def test_charged_sheets_input(self) -> None:
        sheets = (
            "add_charged_sheets: .true.\nnum_charged_sheets: 2\n"
            "sheet_charge: 0.05\nbegin Sheet_Coord\n 0.0 0.0 -4.5\n 1.0 1.0 -4.0\nend Sheet_Coord\n"
            "sheet_charge: -0.1\nbegin Sheet_Coord\n 0.0 0.0 4.0\nend Sheet_Coord"
        )
        problem = self.parse("slab", " 3.0 0.0\n 0.0 3.5", sheets).problem
        self.assertEqual(
            problem.charged_sheets,
            (ChargedSheet(0.05, -4.5), ChargedSheet(0.05, -4.0), ChargedSheet(-0.1, 4.0)),
        )
        self.assertIn("charged_sheets", repr(problem))
        self.assertNotIn("charged_sheets", repr(self.parse("slab", " 3.0 0.0\n 0.0 3.5").problem))
        with self.assertRaisesRegex(ParsecInputError, "error 145"):
            self.parse("wire", " 3.0", sheets)
        with self.assertRaisesRegex(ParsecInputError, "error 146"):
            self.parse("slab", " 3.0 0.0\n 0.0 3.5", "add_charged_sheets: .true.")

    def test_refused_combinations(self) -> None:
        # Minimization, Molecular_Dynamics, Monkhorst_Pack_Shift,
        # Symmetrize_Charge_Density, Polarizability, Super_Cell, Create_DOS
        # and LDA+U of a wire or slab: test_wire_slab_ionic.
        for extra, message in (
            ("Double_Grid_Order: 2", "Double_Grid_Order"),
            ("Non_Collinear_magnetism: .true.", "non-collinear"),
            (
                "Van_Der_Waals: .true.\nCorrelation_Type: pb\nIgnore_Symmetry: .true.",
                "Van_Der_Waals",
            ),
        ):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(UnsupportedParsecOptionError, message):
                    self.parse("wire", " 3.0", extra)

    def test_calc_hartree_g_is_inert(self) -> None:
        # usrinputfile.F90 reads Calc_Hartree_G for a bulk cell only.
        translation = self.parse("slab", " 3.0 0.0\n 0.0 3.5", "Calc_Hartree_G: .true.")
        self.assertFalse(translation.problem.hartree.periodic_g_space)
        self.assertTrue(any("bulk cell only" in note for note in translation.warnings))

    def test_accelerated_cli_runs_the_gamma_point_and_routes_the_rest(self) -> None:
        # A Gamma-point single point runs on the accelerated driver
        # (acceleration/SCF/periodic_routes.py); spin and a restart go to the
        # reference solver by name.
        from parsec_python.acceleration.cli import _reference_only_features

        translation = self.parse("slab", " 3.0 0.0\n 0.0 3.5")
        self.assertEqual(
            _reference_only_features(translation, accelerated_backend_selected=False), []
        )
        for extra, name in (
            ("Spin_Polarization: .true.", "spin polarization"),
            ("Restart_Run: .true.", "Restart_Run of a wire or slab"),
        ):
            with self.subTest(name=name):
                features = _reference_only_features(
                    self.parse("slab", " 3.0 0.0\n 0.0 3.5", extra),
                    accelerated_backend_selected=False,
                )
                self.assertTrue(any(name in feature for feature in features), features)


#: A tilted H2 lifted by +0.95 bohr along z on a 5 x 5.5 bohr slab.  PARSEC
#: re-centres it (initial.f90:201-228) and gives -2.45542804 Ry, as for the centred input
#: (z = +-0.7); this port gave -2.4529447349 Ry before it re-centred slabs (eigenvalues 8.6e-2
#: Ry and forces 5.5e-3 Ry/bohr off).
LIFTED_SLAB = """
Boundary_Conditions: {boundary}
begin Cell_Shape
{cell}
end Cell_Shape
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.5
Expansion_Order: 8
States_Num: 4
Convergence_Criterion: 1.0e-8
Skip_Force: .false.
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  0.3 0.1 {first}
  -0.2 0.0 {second}
end Atom_Coord
"""
SLAB_CELL = " 5.0 0.0 0.0\n 0.0 5.5 0.0"


class SlabRecentringTests(unittest.TestCase):
    """PARSEC moves a slab so that its highest and lowest atoms are as far from z = 0."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def parse(self, first="1.65", second="0.25", boundary="slab", cell=SLAB_CELL, extra=""):
        path = self.directory / "parsec.in"
        path.write_text(
            LIFTED_SLAB.format(boundary=boundary, cell=cell, extra=extra, first=first, second=second),
            encoding="utf-8",
        )
        return parse_parsec_input(path)

    def test_the_input_is_moved_along_z_only_beyond_a_tenth_of_a_bohr(self) -> None:
        from parsec_python.Output.parsec_output import slab_recentring_lines

        translation = self.parse()
        self.assertAlmostEqual(translation.slab_shift, 0.95, places=14)
        positions = np.array([atom.position for atom in translation.problem.atoms])
        np.testing.assert_allclose(positions, [[0.3, 0.1, 0.7], [-0.2, 0.0, -0.7]], rtol=0, atol=1e-15)
        self.assertEqual(
            slab_recentring_lines(translation),
            [
                "",
                " WARNING: atom coordinates were shifted so",
                " that the slab is centered at the origin",
                " The new atom coordinates are:",
                "    0.300000000       0.100000000       0.700000000",
                "   -0.200000000       0.000000000      -0.700000000",
                "",
            ],
        )
        # |z_mov| = 0.075 is not above the threshold; a wire, a restart and manual
        # movements keep the coordinates of the input (Fortran: per == 2,
        # istart == 0, not MANUAL).
        for first, second, boundary, cell, extra in (
            ("0.75", "-0.6", "slab", SLAB_CELL, ""),
            ("1.65", "0.25", "wire", " 5.0", ""),
            ("1.65", "0.25", "slab", SLAB_CELL, "Restart_Run: .true."),
            ("1.65", "0.25", "slab", SLAB_CELL, "Minimization: manual"),
        ):
            with self.subTest(boundary=boundary, extra=extra, z=(first, second)):
                kept = self.parse(first, second, boundary, cell, extra)
                self.assertEqual(kept.slab_shift, 0.0)
                self.assertEqual(slab_recentring_lines(kept), [])
                np.testing.assert_array_equal(
                    [atom.position[2] for atom in kept.problem.atoms], [float(first), float(second)]
                )
        restart = self.parse(extra="Restart_Run: .true.")
        self.assertTrue(any("re-centres a slab" in note and "-0.9500" in note for note in restart.warnings))

    def test_restart_structures_of_a_slab(self) -> None:
        from parsec_python.Relaxation.files import BOHR_TO_ANGSTROM
        from parsec_python.Relaxation.run import restart_structure

        translation = self.parse(first="0.7", second="-0.7", extra="Minimization: bfgs")
        # An XYZ file stands for the Atom_Coord block: re-centred as the input is.
        xyz = self.directory / "start.xyz"
        rows = [(0.3, 0.1, 2.0), (-0.2, 0.0, 0.6)]
        xyz.write_text(
            "2\nlifted\n"
            + "".join(f"H {x * BOHR_TO_ANGSTROM:.12f} {y * BOHR_TO_ANGSTROM:.12f} {z * BOHR_TO_ANGSTROM:.12f}\n" for x, y, z in rows),
            encoding="utf-8",
        )
        moved, _, _ = restart_structure(translation, self.directory, source=xyz)
        self.assertAlmostEqual(moved.slab_shift, 1.3, places=10)
        np.testing.assert_allclose(
            [atom.position for atom in moved.problem.atoms], [[0.3, 0.1, 0.7], [-0.2, 0.0, -0.7]], atol=1e-10
        )

    def test_the_lifted_slab_gives_parsecs_energy_and_forces(self) -> None:
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF.single_point import run_scf
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        results = []
        for first, second in (("1.65", "0.25"), ("0.7", "-0.7")):
            system = prepare_wire_slab_single_point(self.parse(first, second).problem)
            result = run_scf(system)
            self.assertTrue(result.converged)
            results.append((result.energies.total, compute_forces(system, result).total))
        # Fortran: -2.45542804 Ry for both inputs, forces of atom 1 printed as
        # (0.041473, 0.005703, 0.080488) Ry/bohr after the net force is removed.
        self.assertAlmostEqual(results[0][0], -2.45542804, delta=2.0e-8)
        self.assertAlmostEqual(results[0][0], results[1][0], delta=1.0e-11)
        np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=1.0e-8)
        np.testing.assert_allclose(results[0][1][0], [0.041473, 0.005703, 0.080488], rtol=0, atol=2.0e-6)


def hydrogen_problem(
    periodicity: int, *, atoms, kpoints=None, sheets=(), xc="ca", radius=5.0, spacing=0.5, order=8, path=HYDROGEN
):
    if periodicity == 1:
        cell = PeriodicCell(confined_lattice_vectors([[3.0, 0.0, 0.0]]))
    else:
        cell = PeriodicCell(confined_lattice_vectors([[3.0, 0.0, 0.0], [0.0, 3.2, 0.0]]))
    return SinglePointInput(
        atoms=atoms,
        pseudopotentials={"H": SpeciesPotential(path=path, local_angular_momentum=0)},
        grid=PeriodicGridSettings(
            spacing=spacing,
            expansion_order=order,
            shift=(0.5, 0.5, 0.5),
            periodicity=periodicity,
            confinement_radius=radius,
        ),
        periodic_cell=cell,
        scf=SCFSettings(number_of_states=2, xc_functional=xc, convergence_criterion=1e-9, max_iterations=80),
        eigensolver=EigensolverSettings(method="chebdav", tolerance=1e-9),
        hartree=HartreeSettings(relative_tolerance=1e-12),
        kpoints=kpoints,
        charged_sheets=sheets,
    )


class SCFTests(unittest.TestCase):
    def test_wire_local_potential_is_the_image_sum(self) -> None:
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point
        from parsec_python.V_ion import load_pseudopotentials

        atoms = [Atom("H", (0.2, 0.3, 0.7)), Atom("H", (-0.4, 0.0, -0.6))]
        system = prepare_wire_slab_single_point(
            hydrogen_problem(1, atoms=atoms, spacing=0.25, order=12, radius=4.0)
        )
        potentials = load_pseudopotentials(system.input.pseudopotentials)
        grid = system.grid
        # Differences between two points converge in a direct image sum.
        first = int(np.argmin(np.linalg.norm(grid.coordinates - [0.0, 1.5, 0.0], axis=1)))
        second = int(np.argmin(np.linalg.norm(grid.coordinates - [1.0, -2.0, 1.0], axis=1)))
        direct = 0.0
        for atom in atoms:
            for m in range(-4000, 4001):
                image = np.asarray(atom.position) + [3.0 * m, 0.0, 0.0]
                distances = np.linalg.norm(grid.coordinates[[first, second]] - image, axis=1)
                values = potentials["H"].local_potential(distances, 0)
                direct += values[0] - values[1]
        self.assertAlmostEqual(
            system.ionic_potential[first] - system.ionic_potential[second], direct, places=5
        )

    def test_slab_scf_forces_are_the_energy_derivative(self) -> None:
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF.single_point import run_scf
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        sheets = (ChargedSheet(0.1, -3.5),)

        def energy_and_forces(atoms, forces=False):
            system = prepare_wire_slab_single_point(hydrogen_problem(2, atoms=atoms, sheets=sheets))
            result = run_scf(system)
            self.assertTrue(result.converged)
            return result.energies.total, (compute_forces(system, result) if forces else None)

        atoms = [Atom("H", (0.3, 0.1, 0.65)), Atom("H", (-0.2, 0.0, -0.75))]
        _, forces = energy_and_forces(atoms, forces=True)
        step = 0.01
        for index, axis in ((0, 0), (1, 2)):
            moved = []
            for sign in (1.0, -1.0):
                position = np.array(atoms[index].position, dtype=float)
                position[axis] += sign * step
                shifted = list(atoms)
                shifted[index] = Atom("H", position)
                moved.append(energy_and_forces(shifted)[0])
            numeric = -(moved[0] - moved[1]) / (2.0 * step)
            self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=2e-3)

    def test_wire_kpoint_scf_runs_and_has_forces(self) -> None:
        from parsec_python.Forces import compute_forces
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.wire_slab import prepare_wire_slab_single_point

        atoms = [Atom("H", (0.0, 0.0, 0.7)), Atom("H", (0.3, 0.0, -0.7))]
        kpoints = np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]) * 2.0 * np.pi / 3.0
        problem = hydrogen_problem(
            1, atoms=atoms, kpoints=kpoints, xc="pbe", radius=4.5, spacing=0.6, path=DATA / "spin_pbe_H_POTRE.DAT"
        )
        system = prepare_wire_slab_single_point(problem)
        result = run_scf_kpoints(system, kpoints, np.full(2, 0.5))
        self.assertTrue(result.converged)
        forces = compute_forces(system, result)
        self.assertEqual(forces.total.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(forces.total)))
        np.testing.assert_allclose(forces.net_force[1], 0.0, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
