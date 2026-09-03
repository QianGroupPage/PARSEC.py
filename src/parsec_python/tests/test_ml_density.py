from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python import (
    Atom,
    GridSettings,
    InitialDensitySettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    build_cluster_grid,
    build_initial_density,
    load_density_for_grid,
    prepare_reference_single_point,
    save_point_density,
)
from parsec_python.MLDensity.field import BOHR_TO_ANGSTROM
from parsec_python.Input import parse_parsec_input


DATA = Path(__file__).parent / "data"


class DensityFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = build_cluster_grid(
            GridSettings(spacing=1.0, radius=2.1, expansion_order=4)
        )

    def test_legacy_npy_uses_exact_cartesian_indices_and_angstrom_units(self) -> None:
        dense = np.arange(np.prod(self.grid.lookup.shape), dtype=float).reshape(
            self.grid.lookup.shape
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "density.npy"
            np.save(path, dense)
            result = load_density_for_grid(path, self.grid)

        local = self.grid.integer_coordinates - self.grid.index_min
        expected = dense[local[:, 0], local[:, 1], local[:, 2]]
        np.testing.assert_allclose(
            result.density, expected * BOHR_TO_ANGSTROM**3
        )
        self.assertEqual(result.units, "e_per_angstrom3")

    def test_portable_point_density_validates_coordinates_and_clips(self) -> None:
        density = np.ones(self.grid.size)
        density[3] = -0.25
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = save_point_density(
                Path(directory) / "density.npz",
                density,
                self.grid.coordinates,
                units="e_per_bohr3",
                provider="test",
            )
            result = load_density_for_grid(path, self.grid)
            self.assertEqual(result.negative_values_clipped, 1)
            self.assertEqual(result.density[3], 0.0)

            bad = Path(directory) / "bad.npz"
            save_point_density(
                bad,
                density,
                self.grid.coordinates[::-1],
                units="e_per_bohr3",
                provider="test",
            )
            with self.assertRaisesRegex(ValueError, "coordinates do not match"):
                load_density_for_grid(bad, self.grid)

    def test_affine_dense_field_is_interpolated_at_active_dft_points(self) -> None:
        shape = self.grid.lookup.shape
        indices = np.indices(shape, dtype=float)
        dense = indices[0] + 2.0 * indices[1] + 3.0 * indices[2] + 1.0
        origin = self.grid.physical_coordinates(self.grid.index_min[None, :])[0]
        vectors = np.eye(3) * self.grid.spacing
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "affine.npz"
            np.savez_compressed(
                path,
                schema=np.asarray("parsec_python.ml_density.v1"),
                density=dense,
                units=np.asarray("e_per_bohr3"),
                origin_bohr=origin,
                voxel_vectors_bohr=vectors,
            )
            result = load_density_for_grid(path, self.grid)
        local = self.grid.integer_coordinates - self.grid.index_min
        expected = local[:, 0] + 2.0 * local[:, 1] + 3.0 * local[:, 2] + 1.0
        np.testing.assert_allclose(result.density, expected, atol=1.0e-12)


class InitialDensityIntegrationTests(unittest.TestCase):
    def test_file_density_enters_prepare_and_is_normalized_without_ml_imports(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=1.0, radius=2.0, expansion_order=4)
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            density_path = save_point_density(
                Path(directory) / "guess.npz",
                np.linspace(1.0, 2.0, grid.size),
                grid.coordinates,
                units="e_per_bohr3",
                provider="test",
            )
            problem = SinglePointInput(
                atoms=[Atom("H", [0.0, 0.0, 0.0])],
                pseudopotentials={
                    "H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)
                },
                grid=grid.settings,
                scf=SCFSettings(max_iterations=1, number_of_states=2),
                initial_density_settings=InitialDensitySettings(
                    method="file", file=density_path, units="auto"
                ),
            )

            system = prepare_reference_single_point(problem)
        self.assertAlmostEqual(
            system.grid.integrate(system.initial_density),
            system.electron_count,
            places=12,
        )

    def test_external_provider_request_is_cached_and_grid_exact(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=1.0, radius=2.0, expansion_order=4)
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            repository = root / "charge3net"
            repository.mkdir()
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            cache = root / "cache"
            settings = InitialDensitySettings(
                method="charge3net",
                repository=repository,
                checkpoint=checkpoint,
                python_executable=Path(__import__("sys").executable),
                cache_directory=cache,
            )

            def fake_run(command, **kwargs):
                request_path = Path(command[command.index("--request") + 1])
                output_path = Path(command[command.index("--output") + 1])
                with np.load(request_path, allow_pickle=False) as request:
                    coordinates = request["target_coordinates_bohr"]
                save_point_density(
                    output_path,
                    np.ones(coordinates.shape[0]),
                    coordinates,
                    units="e_per_angstrom3",
                    provider="charge3net",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "parsec_python.MLDensity.providers.subprocess.run",
                side_effect=fake_run,
            ) as run:
                first = build_initial_density(
                    settings,
                    grid,
                    [Atom("C-1s", [0.0, 0.0, 0.0])],
                    {
                        "C-1s": SpeciesPotential(
                            DATA / "H_POTRE.DAT", 0, element_symbol="C"
                        )
                    },
                )
                second = build_initial_density(
                    settings,
                    grid,
                    [Atom("C-1s", [0.0, 0.0, 0.0])],
                    {
                        "C-1s": SpeciesPotential(
                            DATA / "H_POTRE.DAT", 0, element_symbol="C"
                        )
                    },
                )
            self.assertEqual(run.call_count, 1)
            np.testing.assert_array_equal(first.density, second.density)
            np.testing.assert_allclose(first.density, BOHR_TO_ANGSTROM**3)

    def test_named_provider_can_reuse_precomputed_portable_density(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=1.0, radius=2.0, expansion_order=4)
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            density_path = save_point_density(
                Path(directory) / "charge3net-density.npz",
                np.linspace(1.0, 2.0, grid.size),
                grid.coordinates,
                units="e_per_bohr3",
                provider="charge3net",
            )
            settings = InitialDensitySettings(
                method="charge3net",
                file=density_path,
                # These deliberately do not exist.  A precomputed prediction
                # must not resolve or import an external provider environment.
                repository=Path(directory) / "missing-repository",
                checkpoint=Path(directory) / "missing-checkpoint.pt",
            )
            result = build_initial_density(
                settings,
                grid,
                [Atom("H", [0.0, 0.0, 0.0])],
                {"H": SpeciesPotential(DATA / "H_POTRE.DAT", 0)},
            )

        np.testing.assert_allclose(
            result.density, np.linspace(1.0, 2.0, grid.size)
        )
        self.assertEqual(settings.method, "charge3net")


class InitialDensityInputTests(unittest.TestCase):
    def test_parsec_extensions_are_relative_and_default_remains_sad(self) -> None:
        source = DATA / "H2_parsec.in"
        default = parse_parsec_input(source)
        self.assertEqual(default.problem.initial_density_settings.method, "sad")

        text = source.read_text(encoding="utf-8") + """
Initial_Density: file
ML_Density_File: guesses/rho.npz
ML_Density_Units: e_per_bohr3
ML_Density_Negative_Policy: error
Normalize_Initial_Density: true
"""
        with patch.object(Path, "read_text", return_value=text):
            translated = parse_parsec_input(source)
        settings = translated.problem.initial_density_settings
        self.assertEqual(settings.method, "file")
        self.assertEqual(settings.file, (DATA / "guesses" / "rho.npz").resolve())
        self.assertEqual(settings.units, "e_per_bohr3")
        self.assertEqual(settings.negative_policy, "error")


if __name__ == "__main__":
    unittest.main()
