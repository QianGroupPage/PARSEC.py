from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

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
from parsec_python.MLDensity import (
    BoxGrid,
    box_around_atoms,
    predict_density,
    provider_names,
    read_xyz,
    save_box_density,
    write_cube,
)
from parsec_python.MLDensity import providers, registry
from parsec_python.MLDensity.cli import main as ml_density_main
from parsec_python.MLDensity.field import BOHR_TO_ANGSTROM
from parsec_python.Input import parse_parsec_input


DATA = Path(__file__).parent / "data"
TEMPLATE = registry.BRIDGES / "template.py"


def template_density(points_bohr, atoms_bohr, valences) -> np.ndarray:
    """The toy model of ``bridges/template.py``, in electrons per bohr^3."""

    density = np.zeros(points_bohr.shape[0])
    for atom, valence in zip(np.asarray(atoms_bohr, dtype=float), valences):
        distance = np.linalg.norm(points_bohr - atom, axis=1)
        density += valence / np.pi * np.exp(-2.0 * distance)
    return density


def small_input(directory: Path, extra: str = "") -> Path:
    """H2 on a coarse sphere, written beside the test pseudopotential."""

    text = (DATA / "H2_parsec.in").read_text(encoding="utf-8")
    text = text.replace("Boundary_Sphere_Radius: 7.0 ang", "Boundary_Sphere_Radius: 3.0 ang")
    text = text.replace("Grid_Spacing: 0.20 ang", "Grid_Spacing: 0.40 ang")
    path = directory / "parsec.in"
    path.write_text(text + extra, encoding="utf-8")
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    return path


def quietly(function, *args):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        result = function(*args)
    return result, stream.getvalue()


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

    def test_recorded_atoms_must_be_in_the_frame_of_the_calculation(self) -> None:
        atoms = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = save_point_density(
                Path(directory) / "density.npz",
                np.ones(self.grid.size),
                self.grid.coordinates,
                units="e_per_bohr3",
                provider="test",
                symbols=["H", "H"],
                atom_positions_bohr=atoms,
            )
            # The same frame, a slightly different geometry, or other atoms
            # altogether: a starting guess, accepted.
            for current in (atoms, atoms * 1.05, np.zeros((3, 3))):
                load_density_for_grid(path, self.grid, atom_positions_bohr=current)
            with self.assertRaisesRegex(ValueError, "translated by"):
                load_density_for_grid(
                    path, self.grid, atom_positions_bohr=atoms + [1.0, 0.0, 0.0]
                )

    def test_cube_file_holds_the_box_and_the_values_in_order(self) -> None:
        values = np.arange(2 * 3 * 7, dtype=float).reshape(2, 3, 7) * 1.0e-3
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = write_cube(
                Path(directory) / "rho.cube",
                values,
                [-1.0, -2.0, -3.0],
                np.diag([0.5, 0.25, 0.125]),
                [8, 1],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            )
            lines = path.read_text(encoding="ascii").splitlines()
        self.assertEqual(lines[2].split(), ["2", "-1.000000", "-2.000000", "-3.000000"])
        self.assertEqual([line.split()[0] for line in lines[3:6]], ["2", "3", "7"])
        self.assertEqual(lines[6].split()[0], "8")
        data = np.asarray(" ".join(lines[8:]).split(), dtype=float)
        np.testing.assert_allclose(data, values.reshape(-1), rtol=1.0e-5)
        # z runs fastest, and a new line starts with every (x, y) pair.
        self.assertEqual(len(lines[8:]), 2 * 3 * 2)


class BoxTests(unittest.TestCase):
    atoms = np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])

    def test_two_of_size_points_and_spacing_fix_the_box(self) -> None:
        by_spacing = box_around_atoms(self.atoms, vacuum_bohr=3.0, spacing_bohr=0.5)
        np.testing.assert_array_equal(by_spacing.shape, (16, 14, 14))
        np.testing.assert_allclose(by_spacing.lengths_bohr, [8.0, 7.0, 7.0])
        points = by_spacing.points()
        np.testing.assert_allclose(points.mean(axis=0), [1.0, 0.5, -0.5])
        np.testing.assert_allclose(points[1] - points[0], [0.0, 0.0, 0.5])

        by_points = box_around_atoms(self.atoms, box_bohr=[10.0, 8.0, 6.0], points=[20, 16, 30])
        np.testing.assert_allclose(by_points.spacing_bohr, [0.5, 0.5, 0.2])
        fixed = box_around_atoms(self.atoms, points=11, spacing_bohr=1.0)
        np.testing.assert_allclose(fixed.lengths_bohr, 11.0)
        self.assertEqual(fixed.size, 11**3)

        for options in (
            {"vacuum_bohr": 3.0},
            {"box_bohr": 9.0, "points": 9, "spacing_bohr": 1.0},
            {"box_bohr": 9.0, "vacuum_bohr": 1.0, "spacing_bohr": 1.0},
        ):
            with self.assertRaises(ValueError):
                box_around_atoms(self.atoms, **options)
        with self.assertRaisesRegex(ValueError, "smaller than the molecule"):
            box_around_atoms(self.atoms, box_bohr=1.5, spacing_bohr=0.5)

    def test_xyz_symbols_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "mol.xyz"
            path.write_text("3\ncomment\nc 0 0 0\nCL1 1 0 0\n8 0 1 0\n", encoding="utf-8")
            symbols, positions = read_xyz(path)
        self.assertEqual(symbols, ["C", "Cl", "O"])
        np.testing.assert_allclose(positions[2], [0.0, 1.0, 0.0])


class ProviderTests(unittest.TestCase):
    def test_registry_lists_the_models_and_validates_their_names(self) -> None:
        self.assertEqual(provider_names()[:3], ("charge3net", "scdp", "custom"))
        with self.assertRaisesRegex(ValueError, "sad, file, charge3net, scdp, custom"):
            InitialDensitySettings(method="unknown")
        with self.assertRaisesRegex(ValueError, "ChargE3Net model must be one of"):
            InitialDensitySettings(method="charge3net", model="fast")
        with self.assertRaisesRegex(ValueError, "needs ML_Density_Bridge"):
            InitialDensitySettings(method="custom")
        with self.assertRaisesRegex(ValueError, "needs a model"):
            InitialDensitySettings(bridge=TEMPLATE)
        self.assertEqual(
            InitialDensitySettings(method="scdp", model="QM9-Fast").model, "qm9-fast"
        )

        registry.register_provider(
            registry.DensityProvider(
                name="toy", title="Toy", bridge=TEMPLATE, models=("v1",),
                default_model="v1", needs_repository=False, needs_checkpoint=False,
            )
        )
        try:
            self.assertEqual(InitialDensitySettings(method="toy", model="v1").method, "toy")
            with self.assertRaises(ValueError):
                registry.register_provider(registry.get_provider("toy"))
        finally:
            registry._PROVIDERS.pop("toy")

    def test_custom_bridge_runs_the_model_once_and_is_cached(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=1.0, radius=2.0, expansion_order=4))
        atoms = [Atom("C-1s", [0.0, 0.0, 0.0])]
        specifications = {
            "C-1s": SpeciesPotential(DATA / "H_POTRE.DAT", 0, element_symbol="C")
        }
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bridge = root / "bridge.py"
            shutil.copy(TEMPLATE, bridge)
            settings = InitialDensitySettings(
                method="custom",
                model="toy-v1",
                bridge=bridge,
                python_executable=Path(sys.executable),
                cache_directory=root / "cache",
            )
            first = build_initial_density(settings, grid, atoms, specifications)
            entries = sorted((root / "cache").glob("custom-*"))
            self.assertEqual(len(entries), 1)
            self.assertEqual(len(entries[0].name), len("custom-") + 16)
            manifest = json.loads((entries[0] / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["key"][:16], entries[0].name[-16:])
            self.assertEqual(manifest["model"], "toy-v1")
            self.assertFalse(list(entries[0].glob("*.partial.npz")))
            np.testing.assert_allclose(
                first.density,
                template_density(grid.coordinates, [[0.0, 0.0, 0.0]], [4]),
                rtol=1.0e-12,
            )

            with patch.object(providers, "run_bridge", side_effect=AssertionError("ran")):
                second = build_initial_density(settings, grid, atoms, specifications)
            np.testing.assert_array_equal(first.density, second.density)

            # A changed bridge is a changed model: it does not reuse the entry.
            bridge.write_text(bridge.read_text(encoding="utf-8") + "\n# v2\n", encoding="utf-8")
            build_initial_density(settings, grid, atoms, specifications)
            self.assertEqual(len(list((root / "cache").glob("custom-*"))), 2)

    def test_named_provider_request_is_cached_and_converted_from_angstrom(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=1.0, radius=2.0, expansion_order=4))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            repository = root / "charge3net"
            repository.mkdir()
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            settings = InitialDensitySettings(
                method="charge3net",
                repository=repository,
                checkpoint=checkpoint,
                python_executable=Path(sys.executable),
                cache_directory=root / "cache",
            )

            def fake_bridge(resolved, request, output, **options):
                self.assertEqual(resolved.name, "charge3net")
                self.assertEqual(resolved.checkpoint, checkpoint.resolve())
                with np.load(request, allow_pickle=False) as data:
                    coordinates = data["target_coordinates_bohr"]
                    self.assertEqual(list(data["symbols"]), ["C"])
                    self.assertTrue(np.all(data["probe_positions_angstrom"] > 0.0))
                save_point_density(
                    output,
                    np.ones(coordinates.shape[0]),
                    coordinates,
                    units="e_per_angstrom3",
                    provider="charge3net",
                )

            arguments = (
                grid,
                [Atom("C-1s", [0.0, 0.0, 0.0])],
                {"C-1s": SpeciesPotential(DATA / "H_POTRE.DAT", 0, element_symbol="C")},
            )
            with patch.object(providers, "run_bridge", side_effect=fake_bridge) as run:
                first = build_initial_density(settings, *arguments)
                second = build_initial_density(settings, *arguments)
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

    def test_box_prediction_on_the_lattice_of_a_grid_imports_exactly(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=2.0, expansion_order=4))
        atoms = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
        box = BoxGrid(
            origin_bohr=grid.physical_coordinates(grid.index_min[None, :])[0],
            spacing_bohr=np.full(3, grid.spacing),
            shape=tuple(grid.lookup.shape),
        )
        prediction = predict_density(
            "custom", ["H", "H"], atoms, box.points(),
            bridge=TEMPLATE, python=sys.executable,
        )
        self.assertEqual(prediction.metadata["model"], "custom")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = save_box_density(
                Path(directory) / "box.npz",
                prediction.density.reshape(box.shape),
                box.origin_bohr,
                box.voxel_vectors_bohr,
                units="e_per_bohr3",
                provider="custom",
                symbols=["H", "H"],
                atom_positions_bohr=atoms,
            )
            imported = load_density_for_grid(path, grid, atom_positions_bohr=atoms)
        np.testing.assert_allclose(
            imported.density, template_density(grid.coordinates, atoms, [1, 1]),
            rtol=1.0e-10, atol=1.0e-14,
        )

    def test_a_linked_venv_interpreter_is_run_under_its_own_name(self) -> None:
        # The bin/python of a Linux venv links to the base interpreter;
        # resolving the link would run the bridge outside the venv.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            link = root / "venv" / "bin" / "python"
            link.parent.mkdir(parents=True)
            try:
                link.symlink_to(sys.executable)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic links are not available")
            resolved = providers.resolve_provider("custom", bridge=TEMPLATE, python=link)
            self.assertEqual(resolved.python, link)
            source = small_input(root, f"\nInitial_Density: custom\nML_Density_Bridge: {TEMPLATE}\n"
                                 "ML_Density_Python: venv/bin/python\n")
            settings = parse_parsec_input(source).problem.initial_density_settings
            self.assertEqual(settings.python_executable, link)

    def test_failed_bridge_reports_its_output_and_leaves_no_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            bridge = Path(directory) / "broken.py"
            bridge.write_text(
                "import sys\nprint('progress: starting')\nsys.exit('model exploded')\n",
                encoding="utf-8",
            )
            messages = []
            with self.assertRaisesRegex(RuntimeError, "model exploded"):
                predict_density(
                    "custom", ["H"], np.zeros((1, 3)), np.ones((4, 3)),
                    bridge=bridge, python=sys.executable, progress=messages.append,
                )
        self.assertEqual(messages, ["starting"])


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

            # About fifty electrons for one: a start that far off is reported.
            with self.assertWarnsRegex(RuntimeWarning, "outside its training domain"):
                system = prepare_reference_single_point(problem)
            self.assertAlmostEqual(
                system.grid.integrate(system.initial_density),
                system.electron_count,
                places=12,
            )
            save_point_density(
                density_path,
                system.initial_density * 1.05,
                grid.coordinates,
                units="e_per_bohr3",
                provider="test",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                prepare_reference_single_point(problem)


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

    def test_custom_bridge_and_model_defaults_are_read_from_the_input(self) -> None:
        source = DATA / "H2_parsec.in"
        base = source.read_text(encoding="utf-8")
        with patch.object(
            Path, "read_text",
            return_value=base + "Initial_Density: custom\nML_Density_Bridge: models/my_bridge.py\n",
        ):
            settings = parse_parsec_input(source).problem.initial_density_settings
        self.assertEqual(settings.method, "custom")
        self.assertEqual(settings.bridge, (DATA / "models" / "my_bridge.py").resolve())
        self.assertEqual(settings.model, "custom")
        self.assertEqual(settings.cache_directory, DATA / ".parsec_ml_density_cache")
        self.assertNotIn("bridge", repr(settings))
        with patch.object(Path, "read_text", return_value=base + "Initial_Density: scdp\n"):
            self.assertEqual(
                parse_parsec_input(source).problem.initial_density_settings.model, "fast"
            )
        with patch.object(Path, "read_text", return_value=base + "Initial_Density: dftb\n"):
            with self.assertRaisesRegex(Exception, "unknown Initial_Density 'dftb'"):
                parse_parsec_input(source)


class CommandLineTests(unittest.TestCase):
    def test_predict_on_the_grid_of_an_input_then_start_the_scf_from_it(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            source = small_input(
                root,
                f"\nInitial_Density: custom\nML_Density_Bridge: {TEMPLATE}\n"
                f"ML_Density_Python: {sys.executable}\n",
            )
            code, printed = quietly(
                ml_density_main,
                ["predict", "--input", str(source), "--cube", str(root / "h2.cube"), "--quiet"],
            )
            self.assertEqual(code, 0, printed)
            output = root / "custom_custom_density.npz"
            self.assertTrue(output.is_file())
            self.assertIn("ML_Density_File: custom_custom_density.npz", printed)
            self.assertIn("valence electrons of the calculation: 2", printed)
            self.assertTrue((root / "h2.cube").is_file())

            source.write_text(
                source.read_text(encoding="utf-8")
                + "ML_Density_File: custom_custom_density.npz\n",
                encoding="utf-8",
            )
            problem = parse_parsec_input(source).problem
            system = prepare_reference_single_point(problem)
            self.assertAlmostEqual(
                system.grid.integrate(system.initial_density), 2.0, places=10
            )

    def test_predict_on_a_box_from_xyz_and_import_it(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            xyz = root / "h2.xyz"
            # Off-centre on purpose: the default recentres as PARSEC.py does.
            xyz.write_text("2\nH2\nH 1.0 2.0 2.625\nH 1.0 2.0 3.375\n", encoding="utf-8")
            code, printed = quietly(
                ml_density_main,
                [
                    "predict", "--xyz", str(xyz), "--provider", "custom",
                    "--bridge", str(TEMPLATE), "--python", sys.executable,
                    "--vacuum", "3", "--points", "24", "--quiet",
                ],
            )
            self.assertEqual(code, 0, printed)
            self.assertIn("24 x 24 x 24 = 13,824 points", printed)
            output = root / "h2_custom_custom_box.npz"
            with np.load(output) as archive:
                self.assertEqual(archive["density"].shape, (24, 24, 24))
                np.testing.assert_allclose(archive["atom_positions_bohr"].mean(axis=0), 0.0, atol=1e-12)

            source = small_input(root, f"\nInitial_Density: custom\nML_Density_File: {output}\n")
            system = prepare_reference_single_point(parse_parsec_input(source).problem)
            self.assertAlmostEqual(system.grid.integrate(system.initial_density), 2.0, places=10)

            code, printed = quietly(
                ml_density_main,
                [
                    "predict", "--xyz", str(xyz), "--provider", "custom", "--no-center",
                    "--bridge", str(TEMPLATE), "--python", sys.executable,
                    "--box", "8", "--spacing", "0.5", "-o", str(root / "raw.npz"), "--quiet",
                ],
            )
            self.assertEqual(code, 0, printed)
            source.write_text(
                source.read_text(encoding="utf-8").replace(str(output), str(root / "raw.npz")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "translated by"):
                prepare_reference_single_point(parse_parsec_input(source).problem)

    def test_check_runs_a_model_on_water(self) -> None:
        code, printed = quietly(
            ml_density_main,
            ["check", "--provider", "custom", "--bridge", str(TEMPLATE), "--python", sys.executable],
        )
        self.assertEqual(code, 0, printed)
        self.assertIn("electrons in the prediction:", printed)
        code, printed = quietly(
            ml_density_main,
            ["check", "--provider", "custom", "--bridge", str(Path("missing.py")), "--python", sys.executable],
        )
        self.assertEqual(code, 1)
        self.assertIn("FAILED", printed)


if __name__ == "__main__":
    unittest.main()
