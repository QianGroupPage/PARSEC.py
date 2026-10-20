"""Forces and moved atoms on the accelerated paths (real CUDA device).

Three things are checked on a GPU: that complete calculations reproduce the
force tables of Fortran PARSEC, that the sums formed on the device (in
symmetry sectors too) are those of the readable full-grid routines, and that
a prepared system whose atoms were moved in place, continued from the
previous SCF loop, gives what a fresh preparation of the moved structure
gives.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from parsec_python.Forces import compute_forces as readable_forces
from parsec_python.Forces import local_forces
from parsec_python.Grid import build_cluster_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.acceleration import driver
from parsec_python.acceleration.Forces import sector_columns
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, GridSettings, SpeciesPotential
from parsec_python.tests.force_reference import (
    DATA,
    assert_matches_fortran,
    load_cases,
    write_case,
)


def _hybrid_available() -> bool:
    return cupy_available() and native_available()


def _translation(name: str, directory: Path):
    return parse_parsec_input(write_case(load_cases()[name], directory))


SYMMETRIC_WATER = (
    Atom("O", (0.0, -0.2, 0.0)),
    Atom("H", (1.6, 0.2, 0.0)),
    Atom("H", (-1.6, 0.2, 0.0)),
)


class SectorColumnTests(unittest.TestCase):
    def test_columns_count_the_states_of_each_sector(self) -> None:
        columns = sector_columns(np.array([2, 0, 2, 1, 0, 2]))
        np.testing.assert_array_equal(columns, [0, 0, 1, 0, 1, 2])


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceLocalForceTests(unittest.TestCase):
    def test_device_sums_equal_the_readable_routine(self) -> None:
        from parsec_python.acceleration.Forces.local_device import cupy_local_forces

        for use_spline in (False, True):
            specifications = {
                "O": SpeciesPotential(DATA / "water_O_POTRE.DAT", 1),
                "H": SpeciesPotential(DATA / "water_H_POTRE.DAT", 0),
                "Ge": SpeciesPotential(DATA / "Ge_POTRE.DAT", 1, use_spline=use_spline),
            }
            potentials = load_pseudopotentials(specifications)
            grid = build_cluster_grid(GridSettings(spacing=0.35, radius=6.5))
            atoms = (
                Atom("O", (0.05, -0.12, 0.08)),
                Atom("H", (1.55, 0.45, -0.20)),
                Atom("Ge", (2.0, -2.5, 1.0)),
                Atom("Ge", (-2.2, 2.4, -1.5)),
                Atom("H", (-1.35, 0.75, 0.30)),
            )
            coordinates = grid.coordinates
            rng = np.random.default_rng(7)
            density = 0.2 * np.exp(-0.1 * np.einsum("ij,ij->i", coordinates, coordinates))
            residual = 0.01 * rng.normal(size=grid.size)
            xc = -0.5 * np.cbrt(density)
            arguments = (grid, atoms, potentials, specifications, density, residual, xc)
            host = local_forces(*arguments)
            device = cupy_local_forces(*arguments)
            for name in ("ionic", "self_consistency", "core_correction"):
                np.testing.assert_allclose(
                    getattr(device, name),
                    getattr(host, name),
                    rtol=0,
                    atol=2e-12,
                    err_msg=f"{name}, spline={use_spline}",
                )
            self.assertGreater(np.abs(host.core_correction).max(), 1e-3)
            # Devices that share the atoms give the bits of one device: with
            # one CUDA device its threads stand for three.
            import cupy as cp

            present = tuple(range(cp.cuda.runtime.getDeviceCount()))
            for devices in (present if len(present) > 1 else (0, 0, 0), (0,) * 7):
                shared = cupy_local_forces(*arguments, device_ids=devices)
                for name in ("ionic", "self_consistency", "core_correction"):
                    np.testing.assert_array_equal(
                        getattr(shared, name),
                        getattr(device, name),
                        err_msg=f"{name}, spline={use_spline}, devices={devices}",
                    )

    def test_wedge_sums_equal_the_sums_over_the_grid(self) -> None:
        from parsec_python.acceleration.Forces.local_device import (
            cupy_local_forces,
            cupy_wedge_local_forces,
        )
        from parsec_python.acceleration.SCF.symmetry_fields import SymmetryScalarField
        from parsec_python.acceleration.Symmetry.axis_reflection import (
            SignedPermutationReduction,
        )

        specifications = {
            "O": SpeciesPotential(DATA / "water_O_POTRE.DAT", 1),
            "H": SpeciesPotential(DATA / "water_H_POTRE.DAT", 0),
            "Ge": SpeciesPotential(DATA / "Ge_POTRE.DAT", 1, use_spline=True),
        }
        potentials = load_pseudopotentials(specifications)
        grid = build_cluster_grid(GridSettings(spacing=0.35, radius=6.5))
        structures = {
            # Two mirror planes along the axes and their twofold axis.
            4: (
                Atom("O", (0.0, 0.0, 0.3)),
                Atom("H", (1.5, 0.0, -0.4)),
                Atom("H", (-1.5, 0.0, -0.4)),
                Atom("Ge", (0.0, 2.2, 1.0)),
                Atom("Ge", (0.0, -2.2, 1.0)),
            ),
            # One mirror plane that exchanges x and y.
            2: (
                Atom("O", (0.4, 0.4, 0.3)),
                Atom("H", (1.5, -0.7, -0.4)),
                Atom("H", (-0.7, 1.5, -0.4)),
                Atom("Ge", (2.2, 0.9, 1.0)),
                Atom("Ge", (0.9, 2.2, 1.0)),
            ),
        }
        coordinates = grid.coordinates
        rng = np.random.default_rng(11)
        raw = (
            0.2 * np.exp(-0.1 * np.einsum("ij,ij->i", coordinates, coordinates))
            * (1.0 + 0.3 * rng.random(grid.size)),
            0.01 * rng.normal(size=grid.size),
            -0.5 * rng.random(grid.size),
        )
        for order, atoms in structures.items():
            reduction = SignedPermutationReduction.detect(grid, atoms)
            self.assertEqual(reduction.group_order, order)
            self.assertLess(reduction.wedge_size, 0.6 * grid.size)
            # Any field becomes invariant by its average over the orbits.
            wedge = [
                np.bincount(reduction.full_to_wedge, weights=field) / reduction.multiplicities
                for field in raw
            ]
            fields = [values[reduction.full_to_wedge] for values in wedge]
            full = cupy_local_forces(grid, atoms, potentials, specifications, *fields)
            self.assertGreater(np.abs(full.core_correction).max(), 1e-3)
            for given in (
                fields,
                [SymmetryScalarField(reduction, values) for values in wedge],
            ):
                sums = cupy_wedge_local_forces(
                    grid, atoms, potentials, specifications, *given, reduction,
                    device_ids=(0, 0),
                )
                for name in ("ionic", "self_consistency", "core_correction"):
                    np.testing.assert_allclose(
                        getattr(sums, name),
                        getattr(full, name),
                        rtol=0,
                        atol=5e-12,
                        err_msg=f"{name}, group order {order}",
                    )
            moved = (Atom("O", (0.1, 0.2, 0.3)), *atoms[1:])
            with self.assertRaisesRegex(ValueError, "symmetry of the sectors"):
                cupy_wedge_local_forces(
                    grid, moved, potentials, specifications, *fields, reduction
                )

    def test_the_devices_of_the_sectors_share_the_local_forces(self) -> None:
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        import cupy as cp

        from parsec_python.acceleration.Forces.local_device import local_force_device_ids

        sectors = SimpleNamespace(backend=SimpleNamespace(
            symmetry_eigensolver=SimpleNamespace(device_ids=(2, 3, 2, 1))))
        plain = SimpleNamespace(backend=SimpleNamespace())
        with patch.dict(os.environ):
            os.environ.pop("PARSEC_IONIC_GPU_COUNT", None)
            self.assertEqual(local_force_device_ids(sectors), (2, 3, 1))
            self.assertEqual(local_force_device_ids(plain), (int(cp.cuda.Device().id),))
            os.environ["PARSEC_IONIC_GPU_COUNT"] = "2"
            self.assertEqual(local_force_device_ids(sectors), (2, 3))
            os.environ["PARSEC_IONIC_GPU_COUNT"] = "0"
            with self.assertRaisesRegex(ValueError, "must be positive"):
                local_force_device_ids(sectors)


@unittest.skipUnless(_hybrid_available(), "CUDA runtime or native extension is unavailable")
class FortranReferenceTests(unittest.TestCase):
    """Every force table of PARSEC for low-symmetry structures."""

    def _run(self, name: str) -> None:
        case = load_cases()[name]
        with tempfile.TemporaryDirectory() as directory:
            translation = _translation(name, Path(directory))
            system = driver.prepare_single_point(translation.problem)
        result = driver.run_scf(system)
        self.assertTrue(result.converged)
        forces = driver.compute_forces(system, result)
        assert_matches_fortran(self, case, result.energies.total, forces)
        # The readable routines see the same result object.
        again = readable_forces(system, result)
        np.testing.assert_allclose(forces.total, again.total, rtol=0, atol=1e-11)

    def test_water(self) -> None:
        self._run("water_lda_tiny")

    def test_water_with_cubic_splines(self) -> None:
        self._run("water_spline_tiny")

    def test_core_correction_and_d_projectors(self) -> None:
        self._run("ge2h2_nlcc_tiny")

    def test_p_and_d_projectors(self) -> None:
        self._run("ag2h2_tiny")


@unittest.skipUnless(_hybrid_available(), "CUDA runtime or native extension is unavailable")
class MovedAtomTests(unittest.TestCase):
    """In-place geometry update and continued SCF, with and without sectors."""

    @classmethod
    def setUpClass(cls) -> None:
        # The problem names its pseudopotential files: they stay for the class.
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        translation = _translation("water_lda_tiny", Path(directory.name))
        problem = translation.problem
        cls.problem = replace(
            problem,
            atoms=SYMMETRIC_WATER,
            recenter_geometry=False,
            grid=replace(problem.grid, radius=8.0),
        )

    def _prepare(self, problem, symmetry):
        system = driver.prepare_single_point(problem, symmetry=symmetry)
        # The lazy sector orbitals stay on the device; forces read them there.
        system.materialize_final_wavefunctions = False
        return system

    def _check(self, symmetry: str) -> None:
        system = self._prepare(self.problem, symmetry)
        sectors = getattr(system.backend, "symmetry_eigensolver", None)
        self.assertEqual(sectors is not None, symmetry == "on")
        result = driver.run_scf(system)
        forces = driver.compute_forces(system, result)
        self.assertEqual(forces.symmetry_operations, 4)
        np.testing.assert_allclose(forces.total[:, 2], 0.0, atol=1e-12)
        self.assertAlmostEqual(forces.total[1, 0], -forces.total[2, 0], places=12)

        previous_system, previous = system, result
        energies = []
        for _ in range(2):
            step = 0.04 * forces.total / np.abs(forces.total).max()
            atoms = tuple(
                Atom(atom.symbol, atom.position + delta)
                for atom, delta in zip(previous_system.atoms, step)
            )
            moved = driver.update_geometry(previous_system, atoms)
            self.assertIs(moved.backend, previous_system.backend)
            self.assertIs(moved.grid, previous_system.grid)
            warm = driver.run_scf(
                moved,
                warm_start=driver.warm_start_for_moved_atoms(
                    previous_system, previous, moved
                ),
            )
            warm_forces = driver.compute_forces(moved, warm)
            cold_system = self._prepare(
                replace(self.problem, atoms=atoms), symmetry
            )
            cold = driver.run_scf(cold_system)
            cold_forces = driver.compute_forces(cold_system, cold)
            self.assertTrue(warm.converged and cold.converged)
            self.assertAlmostEqual(warm.energies.total, cold.energies.total, delta=5e-8)
            np.testing.assert_allclose(
                warm_forces.total, cold_forces.total, rtol=0, atol=2e-6
            )
            energies.append(warm.energies.total)
            previous_system, previous, forces = moved, warm, warm_forces
        # Moving along the forces lowers the energy.
        self.assertLess(energies[0], result.energies.total)
        self.assertLess(energies[1], energies[0])
        return result.energies.total, energies

    def test_full_grid(self) -> None:
        self._check("off")

    def test_symmetry_sectors(self) -> None:
        sector_energies = self._check("on")
        full_energies = self._check("off")
        self.assertAlmostEqual(sector_energies[0], full_energies[0], delta=5e-8)
        np.testing.assert_allclose(
            sector_energies[1], full_energies[1], rtol=0, atol=5e-8
        )

    def test_sector_forces_equal_full_grid_forces(self) -> None:
        tables = {}
        for symmetry in ("on", "off"):
            system = self._prepare(self.problem, symmetry)
            result = driver.run_scf(system)
            tables[symmetry] = driver.compute_forces(system, result)
        np.testing.assert_allclose(
            tables["on"].nonlocal_pseudopotential,
            tables["off"].nonlocal_pseudopotential,
            rtol=0,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            tables["on"].total, tables["off"].total, rtol=0, atol=2e-6
        )

    def test_one_sector_of_the_identity_is_the_full_grid(self) -> None:
        # PARSEC_CUPY_SINGLE_SECTOR: a run without point group solved as the
        # single sector of the identity (what the full-SCF runner does for
        # molecular dynamics), its atoms moved once.
        lopsided = (
            SYMMETRIC_WATER[0],
            Atom(SYMMETRIC_WATER[1].symbol, SYMMETRIC_WATER[1].position + [0.0, 0.05, 0.1]),
            SYMMETRIC_WATER[2],
        )
        problem = replace(self.problem, atoms=lopsided)
        outcomes = {}
        for setting in ("on", "auto"):
            with mock.patch.dict(os.environ, {"PARSEC_CUPY_SINGLE_SECTOR": setting}):
                system = self._prepare(problem, "off")
                sectors = getattr(system.backend, "symmetry_eigensolver", None)
                # Without the sector context of the runner, auto keeps the full grid.
                self.assertEqual(sectors is not None, setting == "on")
                result = driver.run_scf(system)
                forces = driver.compute_forces(system, result, symmetrize=False)
                atoms = tuple(
                    Atom(atom.symbol, atom.position + 0.02 * force / np.abs(forces.total).max())
                    for atom, force in zip(system.atoms, forces.total)
                )
                moved = driver.update_geometry(system, atoms)
                warm = driver.run_scf(
                    moved,
                    warm_start=driver.warm_start_for_moved_atoms(system, result, moved),
                )
                outcomes[setting] = (
                    result.energies.total,
                    forces.total,
                    warm.energies.total,
                    driver.compute_forces(moved, warm, symmetrize=False).total,
                )
        sector, grid = outcomes["on"], outcomes["auto"]
        self.assertAlmostEqual(sector[0], grid[0], delta=1e-9)
        np.testing.assert_allclose(sector[1], grid[1], rtol=0, atol=1e-7)
        self.assertAlmostEqual(sector[2], grid[2], delta=1e-8)
        np.testing.assert_allclose(sector[3], grid[3], rtol=0, atol=1e-6)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_SINGLE_SECTOR": "sometimes"}):
            with self.assertRaisesRegex(ValueError, "auto, on or off"):
                driver.single_sector_setting()

    def test_gradients_built_beside_the_scf_loop_give_the_same_forces(self) -> None:
        # PARSEC_CUPY_FORCE_PREFETCH: the gradients of a moved structure are
        # sampled on a host thread during its SCF loop.
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_FORCE_PREFETCH": "on"}):
            system = self._prepare(self.problem, "on")
            result = driver.run_scf(system)
            forces = driver.compute_forces(system, result)
            step = 0.04 * forces.total / np.abs(forces.total).max()
            atoms = tuple(
                Atom(atom.symbol, atom.position + delta)
                for atom, delta in zip(system.atoms, step)
            )
            moved = driver.update_geometry(system, atoms)
            sectors = moved.backend.symmetry_eigensolver
            self.assertIsNotNone(sectors._prepared_force_gradients)
            warm = driver.run_scf(
                moved, warm_start=driver.warm_start_for_moved_atoms(system, result, moved)
            )
            prefetched = driver.compute_forces(moved, warm)
            self.assertIn("background", sectors.force_phase_seconds)
            self.assertIsNone(sectors._prepared_force_gradients)
            built = driver.compute_forces(moved, warm)
            self.assertNotIn("background", sectors.force_phase_seconds)
        np.testing.assert_array_equal(prefetched.total, built.total)
        with mock.patch.dict(os.environ, {"PARSEC_CUPY_FORCE_PREFETCH": "off"}):
            again = driver.update_geometry(moved, system.atoms)
            self.assertIsNone(again.backend.symmetry_eigensolver._prepared_force_gradients)

    def test_atoms_that_leave_the_point_group_of_the_sectors_are_refused(self) -> None:
        # Sectors, wedge fields and wedge sums hold for the group of the
        # prepared structure; without symmetry any structure is taken.
        lopsided = (
            SYMMETRIC_WATER[0],
            Atom(SYMMETRIC_WATER[1].symbol, SYMMETRIC_WATER[1].position + [0.0, 0.0, 0.2]),
            SYMMETRIC_WATER[2],
        )
        symmetric = tuple(
            Atom(atom.symbol, atom.position * [1.02, 1.0, 1.0]) for atom in SYMMETRIC_WATER
        )
        sectors = self._prepare(self.problem, "on")
        with self.assertRaisesRegex(ValueError, "symmetry of the sectors"):
            driver.update_geometry(sectors, lopsided)
        self.assertIsNot(driver.update_geometry(sectors, symmetric), sectors)
        plain = self._prepare(self.problem, "off")
        self.assertIsNot(driver.update_geometry(plain, lopsided), plain)

    def test_a_system_without_updater_refuses(self) -> None:
        system = self._prepare(self.problem, "off")
        bare = replace(system, geometry_updater=None)
        with self.assertRaisesRegex(NotImplementedError, "moving its atoms"):
            driver.update_geometry(bare, SYMMETRIC_WATER)
        with self.assertRaisesRegex(ValueError, "number, order and species"):
            driver.update_geometry(system, SYMMETRIC_WATER[:2])


if __name__ == "__main__":
    unittest.main()
