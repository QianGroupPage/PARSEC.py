"""Projector harmonics at an atom's own grid point (2026-11-08).

A grid point on an atom up to round-off takes the zero-limit direction of the center, the rule of the nonlocal
forces (``Forces.periodic.ROUND_OFF_DISTANCE``), in the Python projectors and the native ones.
"""

from __future__ import annotations

import unittest

import numpy as np

from parsec_python.Grid.space_group import find_grid_space_group, periodic_grid_symmetry
from parsec_python.models import Atom
from parsec_python.tests.unit.test_default_routines_kpoint_wedge import _C_POTRE, _diamond, _grid


class ProjectorCenterTests(unittest.TestCase):
    """A grid point on an atom up to round-off takes the zero-limit direction of the center (2026-11-08).

    Before, only an exact zero displacement did: a displacement of 1e-12 bohr gave its random direction to the p
    projectors, which broke the symmetry of V_NL (diamond with half its atoms on the points of its grid: 2e-4 of
    the operator under a two-fold axis) and moved the reduced k-point mesh 3e-5 Ry off the full one.
    """

    def setUp(self) -> None:
        from parsec_python.Pseudopotential import read_parsec_pseudopotential

        self.potential = read_parsec_pseudopotential(_C_POTRE)

    def test_python_projectors_commute_with_the_operations_of_the_grid(self) -> None:
        from parsec_python.V_ion import build_nonlocal_projectors
        from parsec_python.models import SpeciesPotential

        lattice, atoms = _diamond(1, decimals=16)
        # atoms at a/4 lie on points of the half-step grid of 18 points per edge: displace one by 1e-12 bohr
        atoms[4] = Atom("C", np.asarray(atoms[4].position) + np.array([1.0e-12, -0.7e-12, 0.4e-12]))
        grid = _grid(lattice, 18)
        group = find_grid_space_group(lattice, atoms, grid.points_per_axis, grid.shift)
        symmetry = periodic_grid_symmetry(grid, group)
        specification = SpeciesPotential(path=_C_POTRE, local_angular_momentum=2, use_spline=True)
        operator = build_nonlocal_projectors(
            grid, atoms, {"C": self.potential}, {"C": specification}, lattice,
            k_point=np.array([0.0, 0.0, np.pi / lattice[2, 2]]),
        )
        vector = np.random.default_rng(5).standard_normal(grid.size) + 0.0j
        reference = operator.apply(vector[:, None])[:, 0]
        for index, matrix in enumerate(group.rotation_matrices):
            if abs(int(matrix[2, 2])) != 1 or int(matrix[2, 2]) != 1:
                continue  # the operations that fix k = (0, 0, 1/2)
            with self.subTest(operation=index):
                images = symmetry.images[index]
                rotated = operator.apply(vector[images][:, None])[:, 0]
                # 2e-4 with the direction of the round-off; ~1e-8 left from the displacement itself
                self.assertLess(
                    np.linalg.norm(rotated - reference[images]) / np.linalg.norm(reference), 1.0e-7
                )

    def test_harmonics_go_to_the_center_continuously(self) -> None:
        from parsec_python.V_ion.ionic_potential import PROJECTOR_CENTER_ROUND_OFF, real_spherical_harmonics

        center = min(float(self.potential.radii[0]), PROJECTOR_CENTER_ROUND_OFF)
        points = np.array([[1.0e-12, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5 * center, 0.0, 0.0], [1.0, 0.0, 0.0]])
        old = real_spherical_harmonics(1, points)
        new = real_spherical_harmonics(1, points, center_radius=center)
        np.testing.assert_array_equal(new[1], 0.0)
        np.testing.assert_array_equal(new[3], old[3])  # beyond the center radius: the same bits
        self.assertAlmostEqual(float(new[0, 0]), 0.4886025119029199 * 1.0e-12 / center, places=18)
        self.assertAlmostEqual(float(new[2, 0]), 0.5 * 0.4886025119029199, places=15)

    def test_native_projectors_take_the_same_rule(self) -> None:
        import importlib

        try:
            module = importlib.import_module("parsec_accelerated_native")
        except (ImportError, OSError):
            self.skipTest("the native extension is not built")
        from parsec_python.V_ion.ionic_potential import PROJECTOR_CENTER_ROUND_OFF

        # a module built before the rule does not report it (build_info, 2026-11-08): rebuild it
        reported = module.build_info().get("projector_center_round_off")
        if reported is None:
            self.skipTest("the native extension predates the projector-center rule (rebuild it)")
        self.assertEqual(float(reported), PROJECTOR_CENTER_ROUND_OFF)

        center = min(float(self.potential.radii[0]), PROJECTOR_CENTER_ROUND_OFF)
        atom = np.array([1.0, 2.0, 3.0])
        near = np.array([1.0e-12, -0.7e-12, 0.4e-12])
        half = np.array([0.5 * center, 0.0, 0.0])
        evaluator = module.RadialGridEvaluator(np.ascontiguousarray(np.vstack([atom + near, atom + half])))
        radial, _ = self.potential.radial_projector(1, 2)
        empty = np.zeros(0)
        payload = evaluator.projector_channel(
            atom, np.ascontiguousarray(self.potential.radii, dtype=np.float64),
            np.ascontiguousarray(radial, dtype=np.float64), 2.0, 1, 1.0, empty, empty, empty,
        )
        values = np.asarray(payload["values"])
        # both points interpolate the radial table at its first knot: the rows differ by |u| alone
        ratio = np.linalg.norm(values[0]) / np.linalg.norm(values[1])
        self.assertAlmostEqual(ratio, np.linalg.norm(near) / (0.5 * center), delta=1.0e-6)


if __name__ == "__main__":
    unittest.main()
