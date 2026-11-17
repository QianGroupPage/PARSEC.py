"""The projectors of a periodic cell, all atoms at once (``acceleration/V_ion/periodic_projectors.py``).

:func:`batched_nonlocal_projectors` must give the operator of the reference loop of
:func:`parsec_python.V_ion.ionic_potential.build_nonlocal_projectors` array for array (the CSC arrays and their
types, the signs, the labels) and decline what it does not take, so that the reference runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

import parsec_python.acceleration  # noqa: F401  (installs the builder)
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.V_ion import ionic_potential
from parsec_python.acceleration.V_ion.periodic_projectors import batched_nonlocal_projectors
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential

_ROOT = Path(__file__).resolve().parents[4]
_DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
_CARBON = _ROOT / "examples" / "3d_diamond_benchmark" / "C_POTRE.DAT"
_SILICON = _ROOT / "examples" / "3d_Si" / "Si_POTRE.DAT"
_HYDROGEN = _DATA / "H_POTRE.DAT"
_SIDE = 3.567 / 0.529177210903
_FCC = np.array([(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)])


def _diamond(cells: int, offset=(0.0, 0.0, 0.0)):
    shifts = np.array([(i, j, k) for i in range(cells) for j in range(cells) for k in range(cells)], dtype=float)
    basis = np.vstack([_FCC, _FCC + 0.25])
    sites = (shifts[:, None, :] + basis[None]).reshape(-1, 3) * _SIDE + np.asarray(offset)
    return [Atom("C", tuple(site)) for site in sites]


def _reference(grid, atoms, potentials, specifications, lattice, **keywords):
    with mock.patch.object(ionic_potential, "_projector_provider", None):
        return ionic_potential.build_nonlocal_projectors(grid, atoms, potentials, specifications, lattice, **keywords)


class BatchedProjectorTests(unittest.TestCase):
    def assert_same(self, grid, atoms, specifications, lattice, label) -> None:
        potentials = ionic_potential.load_pseudopotentials(specifications, xc_functional="pbe")
        batched = batched_nonlocal_projectors(grid, atoms, potentials, specifications, lattice)
        self.assertIsNotNone(batched, label)
        reference = _reference(grid, atoms, potentials, specifications, lattice)
        mine, theirs = batched.projectors, reference.projectors
        self.assertEqual(mine.shape, theirs.shape, label)
        for name in ("indptr", "indices", "data"):
            self.assertEqual(getattr(mine, name).dtype, getattr(theirs, name).dtype, f"{label}: {name}")
            np.testing.assert_array_equal(getattr(mine, name), getattr(theirs, name), err_msg=f"{label}: {name}")
        self.assertTrue(mine.has_sorted_indices, label)
        np.testing.assert_array_equal(batched.signs, reference.signs, err_msg=label)
        self.assertEqual(batched.labels, reference.labels, label)
        # what the drivers call: the installed builder
        installed = ionic_potential.build_nonlocal_projectors(grid, atoms, potentials, specifications, lattice)
        np.testing.assert_array_equal(installed.projectors.data, theirs.data, err_msg=label)

    def test_diamond_cells(self) -> None:
        carbon = {"C": SpeciesPotential(_CARBON, 2, use_spline=True)}
        for cells, shift, offset in ((2, 0.0, (0.0, 0.0, 0.0)), (2, 0.5, (0.11, -0.07, 0.23))):
            lattice = cells * _SIDE * np.eye(3)
            grid = build_periodic_grid(
                PeriodicCell(lattice), PeriodicGridSettings(spacing=_SIDE / 18, expansion_order=8, shift=(shift,) * 3)
            )
            self.assert_same(grid, _diamond(cells, offset), carbon, lattice, f"diamond {cells} {shift} {offset}")

    def test_linear_tables_and_threads(self) -> None:
        lattice = 2 * _SIDE * np.eye(3)
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=_SIDE / 16, expansion_order=8))
        carbon = {"C": SpeciesPotential(_CARBON, 2, use_spline=False)}
        for threads in ("1", "3"):
            with mock.patch.dict(os.environ, {"PARSEC_PERIODIC_SETUP_THREADS": threads}):
                self.assert_same(grid, _diamond(2), carbon, lattice, f"linear, {threads} threads")

    def test_two_species_near_the_faces(self) -> None:
        lattice = np.diag([10.0, 10.0, 12.0])
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=0.4, expansion_order=6))
        atoms = [
            Atom("Si", (0.4, 0.5, 0.6)),
            Atom("H", (2.9, 3.1, 1.2)),
            Atom("Si", (5.1, 0.2, 11.6)),
            Atom("H", (10.1, 9.9, 0.05)),
        ]
        specifications = {
            "Si": SpeciesPotential(_SILICON, 2, use_spline=True),
            "H": SpeciesPotential(_HYDROGEN, 0, use_spline=True),
        }
        self.assert_same(grid, atoms, specifications, lattice, "two species")

    def test_what_it_declines(self) -> None:
        carbon = {"C": SpeciesPotential(_CARBON, 2, use_spline=True)}
        potentials = ionic_potential.load_pseudopotentials(carbon, xc_functional="pbe")
        lattice = 2 * _SIDE * np.eye(3)
        grid = build_periodic_grid(PeriodicCell(lattice), PeriodicGridSettings(spacing=_SIDE / 12, expansion_order=4))
        atoms = _diamond(2)
        # a k-point (Bloch phases) and an isolated grid keep the reference
        self.assertIsNone(
            batched_nonlocal_projectors(grid, atoms, potentials, carbon, lattice, k_point=np.array([0.1, 0.0, 0.0]))
        )
        self.assertIsNone(batched_nonlocal_projectors(grid, atoms, potentials, carbon, None))
        # a cell less than two support radii wide: two images of an atom can meet on a grid point
        small = np.diag([3.0, 3.0, 3.0])
        small_grid = build_periodic_grid(PeriodicCell(small), PeriodicGridSettings(spacing=0.3, expansion_order=4))
        self.assertIsNone(batched_nonlocal_projectors(small_grid, [Atom("C", (0.1, 0.2, 0.3))], potentials, carbon, small))

    def test_installed(self) -> None:
        self.assertIs(ionic_potential._projector_provider, batched_nonlocal_projectors)


if __name__ == "__main__":
    unittest.main()
