"""The Bloch block of the finite-difference Laplacian as the kinetic term of a k-point (``PARSEC_KPOINT_KINETIC``).

``build_bloch_negative_laplacian`` gives the exact Bloch block of the supercell's matrix: the two gauges are
unitarily related, the spectra of a mesh of ``m`` points along each axis are together the spectrum of the
supercell of ``m`` cells, and the SCF of a cell with that mesh is the Gamma-point SCF of the supercell.  On the
device the sparse route and the stencil kernel apply the Hamiltonian of the host.
"""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.sparse as sp

from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Laplacian import build_negative_laplacian
from parsec_python.Laplacian.finite_difference import build_bloch_negative_laplacian, kpoint_kinetic_setting
from parsec_python.models import Atom, PeriodicCell, PeriodicGridSettings

try:
    from parsec_python.acceleration.backends.cupy import cupy_available
except ImportError:  # pragma: no cover
    def cupy_available() -> bool:
        return False


def _box(side: float = 6.0, points: int = 9, order: int = 4):
    return build_periodic_grid(
        PeriodicCell(side * np.eye(3)), PeriodicGridSettings(spacing=side / points, expansion_order=order, shift=(0.5,) * 3)
    )


class BlochLaplacianTests(unittest.TestCase):
    def test_the_two_gauges_and_gamma(self) -> None:
        grid = _box()
        k = np.array([0.3, -0.7, 1.1])
        u_gauge = build_bloch_negative_laplacian(grid, k, "u")
        psi_gauge = build_bloch_negative_laplacian(grid, k, "psi")
        phases = sp.diags(np.exp(1j * np.asarray(grid.coordinates) @ k))
        self.assertLess(abs(phases.conj() @ psi_gauge @ phases - u_gauge).max(), 1.0e-13)
        self.assertEqual(abs(u_gauge - u_gauge.conj().T).max(), 0.0)
        self.assertEqual(abs(build_bloch_negative_laplacian(grid, np.zeros(3)) - build_negative_laplacian(grid)).max(), 0.0)
        # a time-reversal invariant k: the twisted Laplacian is real
        self.assertEqual(build_bloch_negative_laplacian(grid, np.pi / 6.0 * np.array([1.0, 0.0, 1.0]), "psi").dtype, np.float64)

    def test_the_mesh_spectra_are_the_spectrum_of_the_supercell(self) -> None:
        side, points = 6.0, 9
        cell = _box(side, points)
        supercell = build_periodic_grid(
            PeriodicCell(2 * side * np.eye(3)),
            PeriodicGridSettings(spacing=side / points, expansion_order=4, shift=(0.5,) * 3),
        )
        full = np.sort(np.linalg.eigvalsh(build_negative_laplacian(supercell).toarray()))
        mesh = np.sort(np.concatenate([
            np.linalg.eigvalsh(build_bloch_negative_laplacian(cell, np.pi / side * np.array(bits, float)).toarray())
            for bits in np.ndindex(2, 2, 2)
        ]))
        np.testing.assert_allclose(mesh, full, rtol=0, atol=1.0e-11)

    def test_the_switch(self) -> None:
        for value, expected in (("", "gradient"), ("Bloch", "bloch"), ("AUTO", "auto")):
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": value}):
                self.assertEqual(kpoint_kinetic_setting(), expected)
        with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "u"}):
            with self.assertRaises(ValueError):
                kpoint_kinetic_setting()


class BlochSCFTests(unittest.TestCase):
    def test_a_mesh_is_the_gamma_point_of_the_supercell(self) -> None:
        # the conventional cube of silicon with the 2 x 2 x 2 mesh of the time-reversal invariant k (Bloch
        # kinetic term) against the cube of eight cubes at Gamma, SCF to 1e-9 Ry; on the device where the
        # acceleration package runs the k-points (the pooled solver)
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.SCF.pbc import prepare_periodic_single_point
        from parsec_python.SCF.single_point import run_scf as run_scf_gamma
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
            base = replace(base, scf=replace(base.scf, convergence_criterion=1.0e-9, max_iterations=120),
                           eigensolver=replace(base.eigensolver, tolerance=1.0e-10))
            side = float(np.asarray(base.periodic_cell.lattice_vectors)[0, 0])
            k_points = np.array([np.array(b, dtype=float) * np.pi / side for b in np.ndindex(2, 2, 2)])
            weights = np.full(8, 1.0 / 8.0)
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "bloch"}):
                mesh = run_scf_kpoints(prepare_periodic_single_point(base), k_points, weights)
            lattice = np.asarray(base.periodic_cell.lattice_vectors, dtype=float)
            atoms = [
                Atom(atom.symbol, tuple(np.asarray(atom.position) + np.array(shift, float) @ lattice))
                for shift in np.ndindex(2, 2, 2) for atom in base.atoms
            ]
            supercell = replace(
                base, atoms=atoms, periodic_cell=replace(base.periodic_cell, lattice_vectors=2.0 * lattice),
                scf=replace(base.scf, number_of_states=8 * base.scf.number_of_states),
            )
            gamma = run_scf_gamma(prepare_periodic_single_point(supercell))
        self.assertTrue(mesh.converged and gamma.converged)
        self.assertAlmostEqual(mesh.energies.total, gamma.energies.total / 8.0, delta=1.0e-8)


class MeshAndSupercellTests(unittest.TestCase):
    """``SCF/kpoint_supercell.py``: a mesh of the time-reversal invariant k-points as the supercell, and back."""

    def test_the_two_transforms(self) -> None:
        from parsec_python.SCF import kpoint_supercell as transforms
        from parsec_python.acceleration.tests.test_periodic_point_group import silicon

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = silicon("pbe", spacing=0.855)
        mesh = replace(base, monkhorst_pack_dimensions=(2, 2, 2), monkhorst_pack_shift=(0.0, 0.0, 0.0))
        switches = {"PARSEC_KPOINT_SUPERCELL": "auto", "PARSEC_SUPERCELL_KPOINTS": "auto"}
        with mock.patch.dict(os.environ, switches):
            supercell, repeats = transforms.supercell_of_trim_mesh(mesh)
            self.assertEqual(repeats, (2, 2, 2))
            self.assertEqual(len(supercell.atoms), 8 * len(base.atoms))
            self.assertEqual(supercell.scf.number_of_states, 8 * base.scf.number_of_states)
            np.testing.assert_array_equal(transforms._grid_counts(supercell), 2 * transforms._grid_counts(base))
            self.assertFalse(supercell.has_kpoint_sampling)
            # a mesh of k = +-1/4 (no shift) is not time-reversal invariant
            self.assertIsNone(transforms.supercell_of_trim_mesh(replace(base, monkhorst_pack_dimensions=(2, 2, 2))))
            cell, back = transforms.kpoints_of_supercell(supercell)
            self.assertEqual(back, (2, 2, 2))
            self.assertEqual(len(cell.atoms), len(base.atoms))
            np.testing.assert_allclose(cell.periodic_cell.lattice_vectors, base.periodic_cell.lattice_vectors)
            self.assertEqual((cell.monkhorst_pack_dimensions, cell.monkhorst_pack_shift), ((2, 2, 2), (0.0, 0.0, 0.0)))
            # the face-centred cube itself repeats no smaller orthorhombic cell
            self.assertIsNone(transforms.kpoints_of_supercell(base))
            # the supercell of the mesh needs no kinetic setting; the mesh of the supercell is refused under an
            # explicit gradient form (it reproduces the supercell with the Bloch form only)
            with mock.patch.dict(os.environ, {"PARSEC_KPOINT_KINETIC": "gradient"}):
                self.assertIsNotNone(transforms.supercell_of_trim_mesh(mesh))
                self.assertIsNone(transforms.kpoints_of_supercell(supercell))
        self.assertIsNone(transforms.supercell_of_trim_mesh(mesh))  # off by default


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceBlochKineticTests(unittest.TestCase):
    def test_the_device_operators_apply_the_host_hamiltonian(self) -> None:
        import cupy as cp

        from parsec_python.Hamiltonian.kpoint_operator import KPointKohnShamHamiltonian
        from parsec_python.acceleration.Hamiltonian.device_bloch import CupyArrayBackend, DeviceBlochConverter
        from parsec_python.tests.unit.test_hamiltonian_fast_paths import _PeriodicFixture

        _PeriodicFixture.setUpClass()
        fx = _PeriodicFixture
        side = float(np.asarray(fx.cell.lattice_vectors)[0, 0])
        rng = np.random.default_rng(7)
        block = rng.normal(size=(fx.grid.size, 5)) + 1j * rng.normal(size=(fx.grid.size, 5))
        for label, k in (("general", fx.k_point), ("time-reversal invariant", np.pi / side * np.array([1.0, 0.0, 1.0]))):
            host = KPointKohnShamHamiltonian(
                fx.laplacian, fx.gradient, fx.potential, fx.nonlocal_k, k,
                bloch_laplacian=build_bloch_negative_laplacian(fx.grid, k, "u"),
            )
            expected = host.apply(block)
            scale = float(np.max(np.abs(expected)))
            for fused in (False, True):
                converter = DeviceBlochConverter(CupyArrayBackend(0))
                converter.fused_stencil = fused
                operator = converter.operator(host)
                if fused:
                    self.assertIsNotNone(operator.stencil, label)
                result = cp.asnumpy(operator.apply(cp.asarray(block)))
                self.assertLess(float(np.max(np.abs(result - expected))), 1.0e-12 * scale, f"{label} fused={fused}")


if __name__ == "__main__":
    unittest.main()
