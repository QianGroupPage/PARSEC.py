"""The tip sphere, the warm starts and the AFM scan of PARSEC 1.5's FDET (synthetic fields)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from parsec_python.Embedding.afm import AFMScanPoint, AFMScanSettings, RYDBERG_PER_BOHR2_IN_N_PER_M, force_gradients
from parsec_python.Embedding.tip_sphere import (
    check_spot_file,
    outside_energy,
    place_tip,
    spot_field_on_tip_grid,
    tip_offset,
)
from parsec_python.Embedding.warm_start import (
    rigid_grid_shift,
    translate_field,
    translate_subspace,
    translation_rows,
)
from parsec_python.Eigensolvers import EigvalState
from parsec_python.Eigensolvers.subspace import SubspaceState
from parsec_python.Grid import build_cluster_grid
from parsec_python.models import Atom, GridSettings, PointCharge
from parsec_python.Output.parsec_dat import ParsecDatSymmetry, cluster_alatt
from parsec_python.Output.pot_dat import PotentialFieldError, PotentialFieldFile


def _spot_file(grid, rng):
    fields = [rng.normal(size=grid.size) for _ in range(5)]
    fields[3] = np.abs(fields[3])
    return PotentialFieldFile(
        kinetic_energy=0.75,
        atom_positions=np.zeros((1, 3)),
        system_type=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        steps=np.full(3, grid.spacing),
        ndim=grid.size,
        shift=np.zeros(3),
        symmetry=ParsecDatSymmetry.identity(cluster_alatt(grid.settings.radius)),
        points=grid.integer_coordinates.copy(),
        ionic_potential=fields[0],
        hartree_potential=fields[1],
        xc_potential=fields[2],
        density=fields[3],
        core_density=fields[4],
        rmax=grid.settings.radius,
    )


class TipSphereTest(unittest.TestCase):
    def setUp(self):
        self.spot_grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, shift=(0.0, 0.0, 0.0)))
        self.tip_grid = build_cluster_grid(GridSettings(spacing=0.5, radius=2.0, shift=(0.0, 0.0, 0.0)))
        self.spot = _spot_file(self.spot_grid, np.random.default_rng(5))

    def test_recentring(self):
        atoms = (Atom("C", (0.0, 0.1, 7.80)), Atom("O", (0.0, 0.0, 5.67)))
        charges = (PointCharge(1.0, (-0.7, 0.0, 0.0)),)
        # fdet_tipsphere.F90: nint(centre/h), the unweighted centre; 0 0 17 for this CO tip in PARSEC 1.5.
        np.testing.assert_array_equal(tip_offset([atom.position for atom in atoms], 0.4), [0, 0, 17])
        moved, moved_charges, offset, spot = place_tip(atoms, charges, 0.4)
        np.testing.assert_allclose(moved[1].position, [0.0, 0.0, 5.67 - 17 * 0.4])
        np.testing.assert_allclose(moved_charges[0].position, [-0.7, 0.0, -17 * 0.4])
        self.assertEqual(spot, charges)

    def test_field_is_a_copy_of_the_spot_points(self):
        offset = np.array([0, 1, 5])
        density, ionic, hartree, missing, on_tip = spot_field_on_tip_grid(self.spot, self.tip_grid, offset)
        expected_missing = 0
        for row, point in enumerate(self.tip_grid.integer_coordinates):
            source = self.spot_grid.rows_for_integer_coordinates((point + offset)[None, :])[0]
            if source < 0:
                expected_missing += 1
                self.assertEqual(density[row], 0.0)
            else:
                self.assertEqual(density[row], self.spot.density[source])
                self.assertEqual(ionic[row], self.spot.ionic_potential[source])
                self.assertEqual(hartree[row], self.spot.hartree_potential[source])
        self.assertEqual(missing, expected_missing)
        self.assertGreater(missing, 0)
        self.assertEqual(int(np.count_nonzero(on_tip)), self.tip_grid.size - missing)

    def test_outside_energy(self):
        offset = np.array([0, 0, 3])
        *_, on_tip = spot_field_on_tip_grid(self.spot, self.tip_grid, offset)
        atoms = (Atom("O", (0.1, 0.0, 0.2)),)
        potentials = {"O": SimpleNamespace(ionic_charge=6.0)}
        energy, count = outside_energy(self.spot, on_tip, atoms, potentials, offset, 0.5)
        spot_ion = np.array([0.1, 0.0, 0.2 + 1.5])
        brute = 0.0
        points = 0
        for j, point in enumerate(self.spot.points):
            if on_tip[j] or self.spot.density[j] == 0.0:
                continue
            points += 1
            brute += self.spot.density[j] * (-2.0 * 6.0 / np.linalg.norm(point * 0.5 - spot_ion))
        self.assertEqual(count, points)
        self.assertLess(abs(energy - brute * 0.125), 1.0e-12 * abs(energy))

    def test_checks(self):
        check_spot_file(self.spot, self.tip_grid, "pot.dat")
        shifted = PotentialFieldFile(**{**self.spot.__dict__, "shift": np.full(3, 0.5)})
        with self.assertRaises(PotentialFieldError) as caught:
            check_spot_file(shifted, self.tip_grid, "pot.dat")
        self.assertEqual(caught.exception.code, 625)
        other = build_cluster_grid(GridSettings(spacing=0.4, radius=2.0, shift=(0.0, 0.0, 0.0)))
        with self.assertRaises(PotentialFieldError) as caught:
            check_spot_file(self.spot, other, "pot.dat")
        self.assertEqual(caught.exception.code, 621)
        unshifted = build_cluster_grid(GridSettings(spacing=0.5, radius=2.0))
        with self.assertRaises(PotentialFieldError) as caught:
            check_spot_file(self.spot, unshifted, "pot.dat")
        self.assertEqual(caught.exception.code, 622)


class TranslationTest(unittest.TestCase):
    def test_rigid_shift(self):
        old = [[0.1, 0.0, 1.9], [0.1, 0.0, 3.3]]
        self.assertIsNone(rigid_grid_shift(old, old, 0.5))
        np.testing.assert_array_equal(rigid_grid_shift(old, [[0.1, 0.0, 1.4], [0.1, 0.0, 2.8]], 0.5), [0, 0, -1])
        self.assertIsNone(rigid_grid_shift(old, [[0.1, 0.0, 1.4], [0.1, 0.0, 2.9]], 0.5))
        self.assertIsNone(rigid_grid_shift(old, [[0.1, 0.0, 1.6], [0.1, 0.0, 3.0]], 0.5))

    def test_translate_fields_and_subspace(self):
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=2.0, shift=(0.0, 0.0, 0.0)))
        values = np.arange(grid.size, dtype=float) + 1.0
        rows = translation_rows(grid, np.array([0, 0, 1]))
        moved = translate_field(values, rows)
        for row, point in enumerate(grid.integer_coordinates):
            source = grid.rows_for_integer_coordinates((point - [0, 0, 1])[None, :])[0]
            self.assertEqual(moved[row], 0.0 if source < 0 else values[source])
        kept = translate_field(values, rows, fill=values)
        self.assertTrue(np.all(kept[rows < 0] == values[rows < 0]))
        vectors = np.column_stack([values, 2.0 * values])
        state = EigvalState(
            operator_dimension=grid.size,
            requested_states=1,
            working_states=2,
            initial_method="chebff",
            subspace=SubspaceState(operator_dimension=grid.size, working_states=2, eigenvalues=np.zeros(2), vectors=vectors),
            solves_completed=1,
        )
        translated = translate_subspace(state, rows)
        np.testing.assert_array_equal(translated.subspace.vectors[:, 1], 2.0 * moved)
        self.assertIsNone(translate_subspace(object(), rows))


class AFMScanTest(unittest.TestCase):
    def test_order_and_gradients(self):
        settings = AFMScanSettings(heights=5, height_step=0.4, lateral_points=((0.0, 0.0), (0.8, 0.0)))
        offsets = settings.offsets()
        self.assertEqual([(line, k) for line, k, _ in offsets],
                         [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0)])
        np.testing.assert_allclose(offsets[0][2], [0.0, 0.0, -0.8])
        np.testing.assert_allclose(offsets[5][2], [0.8, 0.0, 0.8])
        structures = settings.structures([[0.0, 0.0, 5.67], [0.0, 0.0, 7.80]])
        np.testing.assert_allclose(structures[1], [[0.0, 0.0, 5.27], [0.0, 0.0, 7.40]])
        # E = a z^2 + b z + c: k_ts = 2a at every interior height
        points = [
            AFMScanPoint(line, k, d, 0.3 * d[2] ** 2 - 0.1 * d[2] - 56.0 + line, True) for line, k, d in offsets
        ]
        gradients = force_gradients(settings, points)
        self.assertEqual(sorted(gradients), [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)])
        for value in gradients.values():
            self.assertAlmostEqual(value, 0.6, places=10)
        # 1 Ry/bohr^2 = 48.587 eV/A^2 = 778.4 N/m
        self.assertAlmostEqual(RYDBERG_PER_BOHR2_IN_N_PER_M, 778.4, delta=0.1)

    def test_line_range_and_validation(self):
        settings = AFMScanSettings(heights=3, height_step=0.4, lateral_points=((0, 0), (1, 0), (2, 0)), line_range=(2, 3))
        self.assertEqual([line for line, _, _ in settings.offsets()], [1, 1, 1, 2, 2, 2])
        with self.assertRaises(ValueError):
            AFMScanSettings(heights=2, height_step=0.4)
        with self.assertRaises(ValueError):
            AFMScanSettings(heights=3, height_step=0.4, line_range=(1, 2))


if __name__ == "__main__":
    unittest.main()
