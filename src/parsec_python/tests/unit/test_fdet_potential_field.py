"""PARSEC 1.5's ``pot.dat`` (``potsave.F90``) and its map onto the tip grid (``potfield.F90``)."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Embedding.frozen_density import map_classical_field
from parsec_python.Grid import build_cluster_grid
from parsec_python.models import GridSettings
from parsec_python.Output.parsec_dat import ParsecDatSymmetry, cluster_alatt
from parsec_python.Output.pot_dat import (
    PotentialFieldError,
    PotentialFieldFile,
    read_potential_field,
    write_potential_field,
)


def _record(payload: bytes) -> bytes:
    return struct.pack("<i", len(payload)) + payload + struct.pack("<i", len(payload))


def _fortran_style_file(points: np.ndarray, fields, *, step=0.5, rmax=3.0, newk=1.25) -> bytes:
    """The records of potsave.F90 for a cluster with the identity alone, written by hand."""

    n = points.shape[0]
    out = _record(struct.pack("<d", newk))
    out += _record(struct.pack("<i", 2))
    out += _record(struct.pack("<3d", -0.7, 0.0, 0.0)) + _record(struct.pack("<3d", 0.7, 0.0, 0.0))
    out += _record(struct.pack("<10i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    out += _record(struct.pack("<2d", step, rmax))
    out += _record(struct.pack("<i", n))
    out += _record(struct.pack("<3d", 0.0, 0.0, 0.0))
    out += _record(struct.pack("<2i", n, 1))
    out += _record(struct.pack("<9i", 1, 0, 0, 0, 1, 0, 0, 0, 1))
    out += _record(struct.pack("<9d", 1, 0, 0, 0, 1, 0, 0, 0, 1))
    out += _record(struct.pack("<3d", 0, 0, 0))
    alatt = 4.0 * rmax
    out += _record(struct.pack("<18d", alatt, 0, 0, 0, alatt, 0, 0, 0, alatt, 1 / alatt, 0, 0, 0, 1 / alatt, 0, 0, 0, 1 / alatt))
    out += _record(struct.pack("<i", 1))
    out += _record(np.asarray(points, dtype="<i4").ravel().tobytes())
    for values in fields:
        out += _record(np.asarray(values, dtype="<f8").tobytes())
    return out


class PotDatTest(unittest.TestCase):
    def setUp(self):
        self.grid = build_cluster_grid(GridSettings(spacing=0.5, radius=3.0, shift=(0.0, 0.0, 0.0)))
        rng = np.random.default_rng(3)
        self.order = rng.permutation(self.grid.size)
        self.points = self.grid.integer_coordinates[self.order]
        self.fields = [rng.normal(size=self.grid.size) for _ in range(5)]
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "pot.dat"

    def tearDown(self):
        self.directory.cleanup()

    def test_reads_the_fortran_layout_and_writes_it_back_byte_for_byte(self):
        data = _fortran_style_file(self.points, self.fields)
        self.path.write_bytes(data)
        field = read_potential_field(self.path)
        self.assertEqual(field.kinetic_energy, 1.25)
        np.testing.assert_array_equal(field.atom_positions, [[-0.7, 0, 0], [0.7, 0, 0]])
        self.assertTrue(field.is_cluster)
        self.assertEqual((field.ndim, field.nwedge, field.ntrans, field.rmax), (self.grid.size,) * 2 + (1, 3.0))
        np.testing.assert_array_equal(field.points, self.points)
        np.testing.assert_array_equal(field.density, self.fields[3])
        np.testing.assert_array_equal(field.symmetry.alatt, cluster_alatt(3.0))
        copy = Path(self.directory.name) / "copy.dat"
        write_potential_field(copy, field)
        self.assertEqual(copy.read_bytes(), data)

    def test_maps_by_integer_coordinates(self):
        self.path.write_bytes(_fortran_style_file(self.points, self.fields))
        density, ionic, hartree = map_classical_field(read_potential_field(self.path), self.grid)
        np.testing.assert_array_equal(density[self.order], self.fields[3])
        np.testing.assert_array_equal(ionic[self.order], self.fields[0])
        np.testing.assert_array_equal(hartree[self.order], self.fields[1])

    def test_fortran_error_codes(self):
        self.path.write_bytes(_fortran_style_file(self.points, self.fields))
        field = read_potential_field(self.path)
        other_step = build_cluster_grid(GridSettings(spacing=0.45, radius=3.0, shift=(0.0, 0.0, 0.0)))
        with self.assertRaises(PotentialFieldError) as caught:
            map_classical_field(field, other_step)
        self.assertEqual(caught.exception.code, 601)
        bigger = build_cluster_grid(GridSettings(spacing=0.5, radius=3.5, shift=(0.0, 0.0, 0.0)))
        with self.assertRaises(PotentialFieldError) as caught:
            map_classical_field(field, bigger)
        self.assertEqual(caught.exception.code, 602)
        shifted = build_cluster_grid(GridSettings(spacing=0.5, radius=3.0, shift=(0.5, 0.5, 0.5)))
        with self.assertRaises(PotentialFieldError):
            map_classical_field(field, shifted)
        moved = field.points.copy()
        moved[0] = [0, 0, 0] if not np.all(moved[0] == 0) else [1, 0, 0]
        duplicate = PotentialFieldFile(**{**field.__dict__, "points": moved})
        with self.assertRaises(PotentialFieldError) as caught:
            map_classical_field(duplicate, self.grid)
        self.assertEqual(caught.exception.code, 604)
        far = field.points.copy()
        far[0] = [40, 0, 0]
        with self.assertRaises(PotentialFieldError) as caught:
            map_classical_field(PotentialFieldFile(**{**field.__dict__, "points": far}), self.grid)
        self.assertEqual(caught.exception.code, 603)

    def test_shifted_and_folded_grids(self):
        # Grid_Shift and Foldblock_Side_Length 2: the sample file of such a grid maps onto the same grid
        # by its indices; a tip grid of another shift is refused (PARSEC 1.5 does not compare the shifts), one with
        # or without the fold blocks has another number of points (602).
        for settings in (
            GridSettings(spacing=0.5, radius=3.0, shift=(0.25, 0.0, 0.5)),
            GridSettings(spacing=0.5, radius=3.0, shift=(0.0, 0.0, 0.0), foldblock_side_length=2),
        ):
            grid = build_cluster_grid(settings)
            rng = np.random.default_rng(11)
            order = rng.permutation(grid.size)
            fields = [rng.normal(size=grid.size) for _ in range(5)]
            field = PotentialFieldFile(
                kinetic_energy=0.5,
                atom_positions=np.zeros((1, 3)),
                system_type=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
                steps=np.full(3, 0.5),
                ndim=grid.size,
                shift=np.asarray(settings.shift),
                symmetry=ParsecDatSymmetry.identity(cluster_alatt(3.0)),
                points=grid.integer_coordinates[order],
                ionic_potential=fields[0],
                hartree_potential=fields[1],
                xc_potential=fields[2],
                density=fields[3],
                core_density=fields[4],
                rmax=3.0,
            )
            write_potential_field(self.path, field)
            density, _, _ = map_classical_field(read_potential_field(self.path), grid)
            np.testing.assert_array_equal(density[order], fields[3])
            other = build_cluster_grid(GridSettings(spacing=0.5, radius=3.0, shift=(0.0, 0.0, 0.0)))
            with self.assertRaises(PotentialFieldError):
                map_classical_field(field, other)
        # The same points with another shift in the file: only this port's shift check stops it.
        moved = PotentialFieldFile(**{**field.__dict__, "shift": np.array([0.0, 0.0, 0.5])})
        with self.assertRaises(PotentialFieldError) as caught:
            map_classical_field(moved, grid)
        self.assertEqual(caught.exception.code, 0)

    def test_unreadable_files(self):
        with self.assertRaises(PotentialFieldError) as caught:
            read_potential_field(Path(self.directory.name) / "missing.dat")
        self.assertEqual(caught.exception.code, 611)
        self.path.write_bytes(_fortran_style_file(self.points, self.fields)[:-100])
        with self.assertRaises(PotentialFieldError) as caught:
            read_potential_field(self.path)
        self.assertEqual(caught.exception.code, 612)

    def test_identity_symmetry_of_a_python_file(self):
        field = PotentialFieldFile(
            kinetic_energy=0.5,
            atom_positions=np.zeros((1, 3)),
            system_type=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
            steps=np.full(3, 0.5),
            ndim=self.grid.size,
            shift=np.zeros(3),
            symmetry=ParsecDatSymmetry.identity(cluster_alatt(3.0)),
            points=self.grid.integer_coordinates,
            ionic_potential=self.fields[0],
            hartree_potential=self.fields[1],
            xc_potential=self.fields[2],
            density=self.fields[3],
            core_density=self.fields[4],
            rmax=3.0,
        )
        write_potential_field(self.path, field)
        again = read_potential_field(self.path)
        for name in ("ionic_potential", "hartree_potential", "xc_potential", "density", "core_density", "points"):
            np.testing.assert_array_equal(getattr(again, name), getattr(field, name))
        # 19 records for one atom (20 for the two atoms of an H2 sample of PARSEC 1.5), 8 marker bytes each.
        payload = 8 + 4 + 24 + 40 + 16 + 4 + 24 + 8 + 36 + 72 + 24 + 144 + 4 + (12 + 5 * 8) * self.grid.size
        self.assertEqual(self.path.stat().st_size, payload + 8 * 19)


if __name__ == "__main__":
    unittest.main()
