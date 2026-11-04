"""The cluster features of the Fortran-parity round on the accelerated path.

Polarizability (seven field loops with warm starts), point charges,
Symmetrize_Charge_Density on sector orbits, parsec.dat after every SCF
iteration, ellipsoidal and cylindrical domains (grid and direct Hartree
boundary on the GPU) and Restart_From_Wfndat: what the accelerated CLI no
longer hands to the reference solver, what it still does by name, and the
pieces that the accelerated driver adds, against the reference.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.cli import _reference_only_features
from parsec_python.acceleration.cli import main as accelerated_main
from parsec_python.acceleration.Grid.cluster import build_cluster_grid_by_slabs
from parsec_python.acceleration.SCF.density_symmetry import (
    build_wedge_density_symmetry,
)
from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
from parsec_python.acceleration.Symmetry.axis_reflection import (
    AxisReflectionReduction,
    SignedPermutationReduction,
)
from parsec_python.Grid import build_cluster_grid, symmetrize_density
from parsec_python.Input import parse_parsec_input
from parsec_python.SCF.single_point import density_point_group
from parsec_python.models import Atom, GridSettings, HartreeSettings

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"
CUPY = cupy_available()


def _write_input(directory: Path, extra: str = "", atoms: str | None = None, head: str = "") -> Path:
    """A two-atom H2 input on a coarse grid, with ``extra`` lines."""

    text = f"""\
{head or "Boundary_Sphere_Radius: 4.0 bohr"}
Grid_Spacing: 0.8 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 4
States_Num: 3
Max_Iter: 30
Convergence_Criterion: 1.0e-6
Eigensolver: chebff
Hartree_Boundary_Tolerance: off
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
{atoms or "  0.0  0.0  0.7\n  0.0  0.0 -0.7"}
end Atom_Coord
{extra}
"""
    path = directory / "parsec.in"
    path.write_text(text, encoding="utf-8")
    (directory / "H_POTRE.DAT").write_bytes((DATA / "water_H_POTRE.DAT").read_bytes())
    return path


POINT_CHARGES = (
    "Add_Point_Charges: .true.\nPoint_Typ_Num: 1\nPt_Chg: 0.3\n"
    "begin Point_Coord\n 0.4 2.0 0.3\nend Point_Coord"
)
ELLIPSOID = "Cluster_Domain_Shape: ellipsoid\nbegin Domain_Shape_Parameters\n 3.6 4.0 4.4\nend Domain_Shape_Parameters"
CYLINDER = "Cluster_Domain_Shape: cylinder\nbegin Domain_Shape_Parameters\n 3.6 8.4 3\nend Domain_Shape_Parameters"


class RoutingTests(unittest.TestCase):
    """What the accelerated CLI runs itself and what it hands on by name."""

    def features(self, extra: str = "", head: str = "", **environment) -> list[str]:
        with tempfile.TemporaryDirectory() as folder:
            translation = parse_parsec_input(_write_input(Path(folder), extra, head=head))
            with patch.dict(os.environ, environment):
                return _reference_only_features(translation, accelerated_backend_selected=False)

    def test_cluster_features_run_on_the_accelerated_path(self) -> None:
        for extra, head in (
            ("Polarizability: .true.", ""),
            ("Polarizability: .true.\nSkip_Force: .false.", ""),
            (POINT_CHARGES, ""),
            ("Symmetrize_Charge_Density: .true.", ""),
            ("Save_Intermediate_Charge_Density: .true.", ""),
            ("Save_Intermediate_Eigenvalues: .true.", ""),
            ("", ELLIPSOID),
            ("", CYLINDER),
            ("Restart_Run: .true.\nRestart_From_Wfndat: .true.", ""),
        ):
            with self.subTest(extra=extra, head=head):
                self.assertEqual(self.features(extra, head), [])

    def test_uncovered_combinations_stay_routed_by_name(self) -> None:
        features = self.features(
            "Restart_Run: .true.\nRestart_From_Wfndat: .true.",
            PARSEC_CUPY_SINGLE_SECTOR="on",
        )
        self.assertTrue(any("PARSEC_CUPY_SINGLE_SECTOR" in item for item in features), features)


class WedgeDensitySymmetryTests(unittest.TestCase):
    """The orbit average equals the reference average of the expanded field."""

    def setUp(self) -> None:
        # Methane-like Td cluster: point group of 24, sectors of D2.
        a = 1.2
        self.atoms = (
            Atom("C", [0.0, 0.0, 0.0]),
            Atom("H", [a, a, a]),
            Atom("H", [-a, -a, a]),
            Atom("H", [-a, a, -a]),
            Atom("H", [a, -a, -a]),
        )
        self.grid = build_cluster_grid(GridSettings(spacing=0.7, radius=4.0, expansion_order=4))

    def check(self, reduction) -> None:
        group = density_point_group(self.grid, self.atoms)
        self.assertEqual(group.order, 24)
        self.assertLess(reduction.group_order, group.order)
        reducer = SymmetrySCFReducer(reduction)
        rng = np.random.default_rng(5)
        density = reducer.from_full(rng.random(self.grid.size))
        wedge = build_wedge_density_symmetry(group, reduction)
        symmetric, report = wedge.symmetrize_density(density, self.grid.volume_element)
        expected, expected_report = symmetrize_density(
            reducer.to_full(density), group, self.grid.volume_element
        )
        # The same sums at the representative rows, term for term.
        self.assertTrue(
            np.array_equal(symmetric.values, expected[reduction.representative_rows])
        )
        # The average is constant on the orbits of the sectors.
        self.assertTrue(np.array_equal(reducer.to_full(symmetric), expected))
        self.assertAlmostEqual(
            report.maximum_deviation_percent, expected_report.maximum_deviation_percent, places=10
        )
        self.assertAlmostEqual(report.charge_before, expected_report.charge_before, places=10)
        self.assertAlmostEqual(report.charge_after, expected_report.charge_after, places=10)

    def test_diagonal_sectors(self) -> None:
        self.check(AxisReflectionReduction.detect(self.grid, self.atoms))

    def test_signed_permutation_sectors(self) -> None:
        self.check(SignedPermutationReduction.detect(self.grid, self.atoms))

    def test_operations_outside_the_group_are_refused(self) -> None:
        reduction = AxisReflectionReduction.detect(self.grid, self.atoms)
        group = density_point_group(self.grid, self.atoms)
        from parsec_python.Grid.point_group import GridPointGroup

        identity_only = GridPointGroup(matrices=group.matrices[:1], images=group.images[:1])
        with self.assertRaises(ValueError):
            build_wedge_density_symmetry(identity_only, reduction)


class DomainGridTests(unittest.TestCase):
    """The slab-wise grid of an ellipsoid and a cylinder is the reference grid."""

    def test_same_arrays_as_the_reference_builder(self) -> None:
        for settings in (
            GridSettings(spacing=0.45, radius=4.4, expansion_order=4, domain_shape="ellipsoid",
                         domain_parameters=(3.1, 4.4, 2.7)),
            GridSettings(spacing=0.45, radius=4.4, expansion_order=4, shift=(0.0, 0.0, 0.0),
                         domain_shape="ellipsoid", domain_parameters=(3.1, 4.4, 2.7)),
            GridSettings(spacing=0.4, radius=float(np.hypot(3.0, 2.5)), expansion_order=4,
                         domain_shape="cylinder", domain_parameters=(2.5, 6.0, 2)),
            GridSettings(spacing=0.4, radius=float(np.hypot(3.0, 2.5)), expansion_order=4,
                         domain_shape="cylinder", domain_parameters=(2.5, 6.0, 1)),
        ):
            with self.subTest(settings=settings):
                fast = build_cluster_grid_by_slabs(settings)
                reference = build_cluster_grid(settings)
                for name in ("integer_coordinates", "coordinates", "index_min", "index_max", "lookup"):
                    left, right = getattr(fast, name), getattr(reference, name)
                    self.assertEqual(left.dtype, right.dtype, name)
                    self.assertTrue(np.array_equal(left, right), name)


class DirectBoundarySwitchTests(unittest.TestCase):
    """Which shapes take the GPU direct sum (PARSEC_HARTREE_DIRECT_DEVICE)."""

    def test_switch(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.Hartree import cupy_direct_boundary as module

        def grid(shape):
            return SimpleNamespace(settings=SimpleNamespace(domain_shape=shape))

        hartree = HartreeSettings(boundary_method="auto")
        made = []
        with patch.object(module, "CuPyDirectCoulombBoundary", side_effect=lambda g, **_: made.append(g) or g):
            for value, shape, selected, expected in (
                ("auto", "ellipsoid", "cupy", True),
                ("auto", "cylinder", "cupy", True),
                ("auto", "box", "cupy", False),
                ("auto", "sphere", "cupy", False),
                ("auto", "ellipsoid", "native", False),
                ("on", "box", "cupy", True),
                ("off", "ellipsoid", "cupy", False),
            ):
                with self.subTest(value=value, shape=shape, selected=selected), patch.dict(
                    os.environ, {"PARSEC_HARTREE_DIRECT_DEVICE": value}
                ):
                    built = module.direct_boundary_for(grid(shape), hartree, selected=selected)
                    self.assertEqual(built is not None, expected)
            with patch.dict(os.environ, {"PARSEC_HARTREE_DIRECT_DEVICE": "sometimes"}):
                with self.assertRaises(ValueError):
                    module.direct_boundary_for(grid("ellipsoid"), hartree, selected="cupy")


@unittest.skipUnless(CUPY, "CuPy and a usable GPU are required")
class DeviceDirectBoundaryTests(unittest.TestCase):
    """The GPU direct boundary gives the right-hand side of the reference."""

    def test_right_hand_side(self) -> None:
        from parsec_python.acceleration.Hartree.cupy_direct_boundary import (
            CuPyDirectCoulombBoundary,
        )
        from parsec_python.acceleration.Hartree.poisson import build_hartree_problem

        for settings in (
            GridSettings(spacing=0.5, radius=4.0, expansion_order=6, domain_shape="ellipsoid",
                         domain_parameters=(3.0, 4.0, 3.5)),
            GridSettings(spacing=0.5, radius=float(np.hypot(3.0, 2.5)), expansion_order=4,
                         domain_shape="cylinder", domain_parameters=(2.5, 6.0, 3)),
        ):
            with self.subTest(shape=settings.domain_shape):
                grid = build_cluster_grid(settings)
                rng = np.random.default_rng(2)
                density = np.exp(-np.sum(grid.coordinates**2, axis=1)) + 0.01 * rng.random(grid.size)
                hartree = HartreeSettings(boundary_method="auto")
                expected, _ = build_hartree_problem(density, grid, hartree)
                device = CuPyDirectCoulombBoundary(grid)
                rhs, boundary = device.build(density, hartree)
                scale = np.max(np.abs(expected))
                self.assertLess(np.max(np.abs(rhs - expected)) / scale, 1.0e-13)
                # The boundary object is the reference's direct sum.
                points = device.exterior_points[:5]
                self.assertTrue(
                    np.allclose(boundary.potential(points), device.boundary_values(density * grid.volume_element)[:5],
                                rtol=1.0e-13, atol=0.0)
                )


class PointChargeForceTests(unittest.TestCase):
    """The accelerated forces add those of the point charges as forcept.f90."""

    def test_forces_hold_the_point_charge_term(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.Forces import compute as accelerated
        from parsec_python.Forces import point_charge_forces
        from parsec_python.models import PointCharge

        atoms = (Atom("H", [0.0, 0.0, 0.7]), Atom("H", [0.0, 0.0, -0.7]))
        potentials = {"H": SimpleNamespace(ionic_charge=1.0)}
        charges = (PointCharge(0.3, (0.4, 2.0, 0.3)),)
        captured = {}

        def assemble(positions, repulsion, local, separable, operations, **keywords):
            captured.update(repulsion=repulsion, **keywords)
            return SimpleNamespace()

        system = SimpleNamespace(
            grid=None,
            atoms=atoms,
            pseudopotentials=potentials,
            point_charges=charges,
            input=SimpleNamespace(pseudopotentials={}),
            backend_info=SimpleNamespace(selected="scipy"),
        )
        result = SimpleNamespace(
            output_effective_potential=np.zeros(2),
            input_effective_potential=np.zeros(2),
            density=None,
            xc_potential=None,
        )
        with patch.object(accelerated, "local_forces", return_value=None), patch.object(
            accelerated, "_nonlocal_forces", return_value=np.zeros((2, 3))
        ), patch.object(accelerated, "assemble_forces", side_effect=assemble), patch.object(
            accelerated, "replace", side_effect=lambda value, **_: value
        ), patch.object(accelerated, "ion_ion_forces", return_value=np.zeros((2, 3))):
            accelerated.compute_forces(system, result, symmetrize=False)
        expected = point_charge_forces(atoms, potentials, charges)
        self.assertTrue(np.array_equal(captured["point_charge"], expected))
        self.assertTrue(np.array_equal(captured["repulsion"], expected))


class HostFieldsAndRestartTests(unittest.TestCase):
    """The host fields of an iteration and the warm start of a wfn.dat restart."""

    def test_iteration_fields_drop_or_keep_the_orbitals(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.Output.iteration_fields import host_iteration_fields
        from parsec_python.SCF.single_point import IterationChannelFields, SCFIterationFields

        system = SimpleNamespace(backend_info=SimpleNamespace(selected="scipy"))
        orbitals = np.arange(6.0).reshape(3, 2)
        channel = IterationChannelFields(
            np.zeros(2), np.zeros(2), np.zeros(3), np.zeros(3), wavefunctions=orbitals
        )
        state = SCFIterationFields(
            iteration=1, eigenvalues=np.zeros(4), occupations=np.zeros(4),
            representations=np.ones(4, dtype=np.int32), wavefunctions=None,
            density=np.zeros(3), next_effective_potential=np.zeros(3), spin=(channel, channel),
        )
        dropped = host_iteration_fields(system, state, wavefunctions=False)
        self.assertTrue(all(item.wavefunctions is None for item in dropped.spin))
        kept = host_iteration_fields(system, state, wavefunctions=True)
        self.assertIs(kept.spin[0].wavefunctions, orbitals)

    def test_restart_start_of_the_host_backends_is_the_reference_state(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration.SCF.restart import accelerated_restart_start
        from parsec_python.SCF import SCFWarmStart

        start = SCFWarmStart(np.zeros(3), np.zeros(3), eigensolver_state=object())
        host = SimpleNamespace(eigenproblem_solver=None, backend=SimpleNamespace())
        self.assertIs(accelerated_restart_start(host, start), start)
        sectors = SimpleNamespace(
            eigenproblem_solver=object(), backend=SimpleNamespace(symmetry_eigensolver=object())
        )
        with self.assertRaises(NotImplementedError):
            accelerated_restart_start(sectors, start)
        # parsec.dat holds no orbitals: nothing to seed.
        plain = SCFWarmStart(np.zeros(3), np.zeros(3))
        self.assertIs(accelerated_restart_start(sectors, plain), plain)

    def test_reference_loop_takes_a_group_with_its_own_method(self) -> None:
        from types import SimpleNamespace

        from parsec_python.Grid.point_group import ScalarSymmetrization
        from parsec_python.SCF import prepare_single_point, run_scf

        with tempfile.TemporaryDirectory() as folder:
            translation = parse_parsec_input(
                _write_input(Path(folder), "Symmetrize_Charge_Density: .true.")
            )
            system = prepare_single_point(translation.problem)
        calls = []

        def symmetrize(density, volume_element):
            calls.append(volume_element)
            return density, ScalarSymmetrization(0.0, 2.0, 2.0)

        from dataclasses import replace as dataclass_replace

        group = SimpleNamespace(symmetrize_density=symmetrize)
        result = run_scf(dataclass_replace(system, density_symmetry=group))
        self.assertEqual(len(calls), result.iterations)
        self.assertEqual(result.history[0].density_symmetrization[0].charge_before, 2.0)


def _run(path: Path, *arguments: str) -> tuple[int, str]:
    stream = StringIO()
    with redirect_stdout(stream), redirect_stderr(stream):
        status = accelerated_main([str(path), "--quiet", *arguments])
    return status, stream.getvalue()


@unittest.skipUnless(CUPY and native_available(), "CuPy, a usable GPU and the native extension are required")
class AcceleratedRunTests(unittest.TestCase):
    """Complete runs of the accelerated CLI on a coarse H2, against the reference."""

    def reference(self, directory: Path):
        from parsec_python.cli import main as reference_main

        stream = StringIO()
        with redirect_stdout(stream), redirect_stderr(stream):
            status = reference_main([str(directory / "parsec.in"), "--quiet"])
        self.assertIn(status, (0, 3))
        with np.load(directory / "parsec_python_results.npz") as archive:
            return {name: archive[name] for name in archive.files}

    def compare(self, extra: str, head: str = "", *, forces: bool = False, polar: bool = False):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            accelerated_dir, reference_dir = Path(first), Path(second)
            path = _write_input(accelerated_dir, extra, head=head)
            _write_input(reference_dir, extra, head=head)
            status, output = _run(path)
            self.assertIn(status, (0, 3), output)
            self.assertNotIn("running the reference parsec_python.cli solver", output)
            log = (accelerated_dir / "parsec.out").read_text(encoding="utf-8")
            self.assertIn("Acceleration backend:", log)
            with np.load(accelerated_dir / "parsec_python_results.npz") as archive:
                accelerated = {name: archive[name] for name in archive.files}
            reference = self.reference(reference_dir)
            self.assertAlmostEqual(
                float(accelerated["energy_total_ry"]), float(reference["energy_total_ry"]), delta=1.0e-7
            )
            occupied = np.asarray(reference["occupations"]) > 1.0e-6
            self.assertLess(
                np.max(np.abs(np.sort(accelerated["eigenvalues_ry"])[occupied]
                              - np.sort(reference["eigenvalues_ry"])[occupied])),
                1.0e-6,
            )
            if forces:
                self.assertLess(
                    np.max(np.abs(accelerated["forces_total"] - reference["forces_total"])), 1.0e-6
                )
            if polar:
                self.assertTrue(
                    np.allclose(accelerated["polarizability_energies_ry"],
                                reference["polarizability_energies_ry"], rtol=0.0, atol=1.0e-7)
                )
                lines = (accelerated_dir / "polar.dat").read_text().splitlines()
                expected = (reference_dir / "polar.dat").read_text().splitlines()
                self.assertEqual(len(lines), len(expected))
            return accelerated_dir, log

    def test_polarizability_with_forces(self) -> None:
        self.compare("Polarizability: .true.\nElectric_Field: 0.01\nSkip_Force: .false.", forces=True, polar=True)

    def test_point_charges_with_forces(self) -> None:
        _, log = self.compare(POINT_CHARGES + "\nSkip_Force: .false.", forces=True)
        self.assertIn("point charges", log)

    def test_symmetrized_density(self) -> None:
        _, log = self.compare("Symmetrize_Charge_Density: .true.")
        self.assertIn("Symmetrizing", log)

    def test_ellipsoid(self) -> None:
        self.compare("", ELLIPSOID)

    def test_parsec_dat_after_every_iteration(self) -> None:
        from parsec_python.Output.parsec_dat import read_parsec_dat

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write_input(directory, "Save_Intermediate_Charge_Density: .true.")
            written = []
            from parsec_python.Output import output_files

            original = output_files.RunOutputFiles._write_parsec_dat

            def capturing(files, result, grid, problem, *, report):
                target = original(files, result, grid, problem, report=report)
                written.append(read_parsec_dat(target))
                return target

            with patch.object(output_files.RunOutputFiles, "_write_parsec_dat", capturing):
                status, output = _run(path)
            self.assertIn(status, (0, 3), output)
            with np.load(directory / "parsec_python_results.npz") as archive:
                iterations = int(archive["iterations"])
                density = archive["density_e_per_bohr3"]
            # One file per iteration and the final one, which repeats the last.
            self.assertEqual(len(written), iterations + 1)
            for data in written:
                self.assertEqual(sum(block.state_numbers.size for spin in data.wavefunctions for block in spin), 1)
            self.assertTrue(np.array_equal(written[-2].densities[0], written[-1].densities[0]))
            self.assertTrue(np.array_equal(written[-2].potentials[0], written[-1].potentials[0]))
            self.assertLess(np.max(np.abs(written[-1].densities[0] - density)), 1.0e-15)

    def test_restart_from_wfndat(self) -> None:
        from parsec_python.Output.wfn_dat import wfn_dat_from_result, write_wfn_dat
        from parsec_python.SCF import prepare_single_point, run_scf

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write_input(directory)
            translation = parse_parsec_input(path)
            first = run_scf(prepare_single_point(translation.problem))
            write_wfn_dat(directory / "wfn.dat", wfn_dat_from_result(first, first.grid))
            _write_input(directory, "Restart_Run: .true.\nRestart_From_Wfndat: .true.")
            status, output = _run(path)
            self.assertIn(status, (0, 3), output)
            with np.load(directory / "parsec_python_results.npz") as archive:
                self.assertLessEqual(int(archive["iterations"]), 3)
                self.assertAlmostEqual(float(archive["energy_total_ry"]), first.energies.total, delta=1.0e-6)


if __name__ == "__main__":
    unittest.main()
