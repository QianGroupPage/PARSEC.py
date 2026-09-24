from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python.Grid import build_periodic_grid
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings, SpeciesPotential
from parsec_python.V_ion import build_nonlocal_projectors, load_pseudopotentials
from parsec_python.V_ion.ionic_potential import (
    _projector_support_radius,
    real_spherical_harmonics,
)

_H_POTRE = Path(__file__).resolve().parents[1] / "data" / "H_POTRE.DAT"


def _setup(side_length: float, spacing: float = 0.5):
    specifications = {"H": SpeciesPotential(_H_POTRE, 0)}
    potentials = load_pseudopotentials(specifications, xc_functional="ca")
    cell = PeriodicCell(lattice_vectors=side_length * np.eye(3))
    grid = build_periodic_grid(
        cell, PeriodicGridSettings(spacing=spacing, expansion_order=4)
    )
    center = side_length / 2.0
    atoms = [Atom("H", [center, center, center])]
    return grid, atoms, potentials, specifications, cell


class PeriodicNonlocalProjectorTests(unittest.TestCase):
    def test_large_cell_exactly_matches_isolated_single_copy(self) -> None:
        # Support radius (1.6 bohr for this H pseudopotential) times two is
        # well under 20 bohr, so only one image can ever be in range: the
        # periodic and isolated (lattice_vectors=None) operators must be
        # bit-for-bit identical, not just approximately close.
        grid, atoms, potentials, specifications, cell = _setup(20.0)
        periodic = build_nonlocal_projectors(
            grid, atoms, potentials, specifications, cell.lattice_vectors
        )
        isolated = build_nonlocal_projectors(grid, atoms, potentials, specifications)

        self.assertEqual(periodic.labels, isolated.labels)
        difference = (periodic.as_sparse() - isolated.as_sparse()).tocoo()
        self.assertEqual(difference.nnz, 0)

    def test_small_cell_matches_independent_brute_force_image_sum(self) -> None:
        # A cell smaller than twice the support radius forces multiple
        # periodic images to genuinely overlap the same grid points.
        grid, atoms, potentials, specifications, cell = _setup(3.0, spacing=0.25)
        side_length = 3.0
        support_radius = _projector_support_radius(potentials["H"])
        self.assertLess(side_length, 2.0 * support_radius)  # sanity on the setup

        operator = build_nonlocal_projectors(
            grid, atoms, potentials, specifications, cell.lattice_vectors
        )

        potential = potentials["H"]
        local_l = specifications["H"].local_angular_momentum
        angular_momentum = sorted(
            set(potential.radial_wavefunctions) - {local_l}
        )[0]
        radial_grid, _sign = potential.radial_projector(angular_momentum, local_l)
        sqrt_dv = np.sqrt(grid.volume_element)
        column = next(
            index
            for index, (_atom, l, m) in enumerate(operator.labels)
            if l == angular_momentum and m == 0
        )
        computed = np.asarray(operator.projectors[:, column].todense()).ravel()

        base_position = np.asarray(atoms[0].position)
        brute_force = np.zeros(grid.size)
        shells = range(-3, 4)
        for i in shells:
            for j in shells:
                for k in shells:
                    position = base_position + np.array([i, j, k]) * side_length
                    relative = grid.coordinates - position
                    radius = np.linalg.norm(relative, axis=1)
                    support = radius <= support_radius
                    if not np.any(support):
                        continue
                    radial = np.interp(
                        radius[support],
                        potential.radii,
                        radial_grid,
                        left=radial_grid[0],
                        right=0.0,
                    )
                    harmonics = real_spherical_harmonics(
                        angular_momentum, relative[support]
                    )
                    brute_force[support] += sqrt_dv * radial * harmonics[:, 0]

        np.testing.assert_array_equal(computed, brute_force)
        self.assertGreater(np.max(np.abs(computed)), 0.0)

    def test_apply_matches_as_sparse(self) -> None:
        grid, atoms, potentials, specifications, cell = _setup(3.0, spacing=0.25)
        operator = build_nonlocal_projectors(
            grid, atoms, potentials, specifications, cell.lattice_vectors
        )
        probe = np.linspace(-1.0, 1.0, grid.size)
        np.testing.assert_allclose(
            operator.apply(probe), operator.as_sparse() @ probe, atol=1.0e-12
        )

    def test_rejects_oblique_cell(self) -> None:
        grid, atoms, potentials, specifications, _cell = _setup(20.0)
        oblique = np.array([[20.0, 0.0, 0.0], [1.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        with self.assertRaises(ValueError):
            build_nonlocal_projectors(grid, atoms, potentials, specifications, oblique)


if __name__ == "__main__":
    unittest.main()
