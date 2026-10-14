"""Band structure after the SCF (Fortran ``bandstruc`` / ``calc_bands``).

The k-point generator is checked against the Fortran rules, the input block
against Fortran's line format, and the end-to-end runs against physics that
must hold exactly: bands at the SCF k-points reproduce the SCF eigenvalues,
time reversal gives ``E(k) = E(-k)`` for a non-magnetic system, spin-orbit
levels come in Kramers pairs, and spin-polarized runs return two channels.
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Grid import band_path_kpoints
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import ParsecInputError
from parsec_python.Output.bands import (
    load_band_structure_archive,
    read_bands_dat,
    write_bands_files,
)
from parsec_python.cli import main as cli_main

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)
_BOX = 8.0
_K = 2.0 * np.pi / _BOX * 0.25  # |k| of the +/-0.25 reciprocal-lattice points

_INPUT = """\
Periodic_System: .true.
Boundary_Conditions: bulk

begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape

Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: {states}
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 1.0e-5

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebdav
{extra}
Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: {so_psp}
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   4.0 4.0 4.0
End atom_coord

Correlation_Type: CA
{kpoints}
{bands}
"""

_MP = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"
# Three points -0.25, 0, +0.25 along x: both SCF k-points and Gamma.
_PATH = (
    "begin bandstruc\n1 -0.25 0.0 0.0  0.25 0.0 0.0  left-right\nend bandstruc\n"
    "bandstruc_points 2\n"
)


class BandPathTests(unittest.TestCase):
    def test_fortran_point_counts_and_closure(self) -> None:
        # Segment lengths 1 : 2 : 1.5 with 4 points on the shortest.
        starts = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0.0]])
        ends = np.array([[1, 0, 0], [1, 2, 0], [1, 2, 1.5]])
        segments = band_path_kpoints(starts, ends, 4)
        self.assertEqual([len(s) for s in segments], [4, 8, 6 + 1])
        # Interior segments omit their end point (the next one starts there);
        # the last segment ends exactly on its end point.
        np.testing.assert_allclose(segments[0][-1], [0.75, 0, 0])
        np.testing.assert_allclose(segments[1][0], starts[1])
        np.testing.assert_allclose(segments[-1][-1], ends[-1])
        np.testing.assert_allclose(np.diff(segments[1][:, 1]), 0.25)

    def test_count_is_truncated_like_the_fortran_integer_assignment(self) -> None:
        starts = np.array([[0, 0, 0], [1, 0, 0.0]])
        ends = np.array([[1, 0, 0], [1, 1.7, 0.0]])
        segments = band_path_kpoints(starts, ends, 3)
        self.assertEqual(len(segments[1]), int(1.7 * 3) + 1)

    def test_invalid_paths_are_rejected(self) -> None:
        zero = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            band_path_kpoints(zero, zero, 5)
        with self.assertRaises(ValueError):
            band_path_kpoints(zero, np.ones((1, 3)), 0)
        with self.assertRaises(ValueError):
            band_path_kpoints(np.zeros((1, 3)), np.ones((2, 3)), 3)


class BandInputTests(unittest.TestCase):
    def _parse(self, bands: str, *, periodic: bool = True):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            text = _INPUT.format(states=20, so_psp="false", extra="", kpoints="", bands=bands)
            if not periodic:
                text = (
                    text.replace("Periodic_System: .true.", "Periodic_System: .false.")
                    .replace("Boundary_Conditions: bulk", "Boundary_Sphere_Radius: 6.0")
                    .replace("begin Cell_Shape\n   8.0  8.0  8.0\nend Cell_Shape\n", "")
                )
            (directory / "parsec.in").write_text(text, encoding="utf-8")
            return parse_parsec_input(directory / "parsec.in")

    def test_fortran_block_with_d_exponents_and_labels(self) -> None:
        problem = self._parse(
            "begin bandstruc\n"
            "1 0.00d0 0.00d0 0.00d0   0.50d0 0.00d0 0.50d0   gamma-X\n"
            "2 0.5 0.0 0.5  0.5 0.25 0.75  X-W\n"
            "end bandstruc\nbandstruc_points 7\n"
        ).problem
        bands = problem.bands
        self.assertEqual(bands.points, 7)
        self.assertEqual([s.name for s in bands.segments], ["gamma-X", "X-W"])
        self.assertEqual(bands.segments[0].end, (0.5, 0.0, 0.5))

    def test_default_points_and_missing_label(self) -> None:
        bands = self._parse("begin bandstruc\n1 0 0 0 0.5 0 0\nend bandstruc\n").problem.bands
        self.assertEqual(bands.points, 20)
        self.assertEqual(bands.segments[0].name, "")

    def test_no_bands_by_default(self) -> None:
        self.assertIsNone(self._parse("").problem.bands)

    def test_input_errors(self) -> None:
        cases = {
            "short line": "begin bandstruc\n1 0 0 0 0.5 0\nend bandstruc\n",
            "zero length segment": "begin bandstruc\n1 0 0 0 0 0 0 a-a\nend bandstruc\n",
            "points without a block": "bandstruc_points 10\n",
            "non-numeric": "begin bandstruc\n1 0 0 0 x 0 0 a-b\nend bandstruc\n",
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(ParsecInputError):
                self._parse(text)

    def test_cluster_input_is_rejected(self) -> None:
        with self.assertRaises(ParsecInputError):
            self._parse(_PATH, periodic=False)


class BandStructureRunTests(unittest.TestCase):
    def _run(self, *, so_psp: str, extra: str = "", states: int = 20, bands: str = _PATH):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            (directory / "parsec.in").write_text(
                _INPUT.format(
                    states=states, so_psp=so_psp, extra=extra, kpoints=_MP, bands=bands
                ),
                encoding="utf-8",
            )
            self.assertEqual(cli_main([str(directory / "parsec.in"), "--quiet"]), 0)
            log = (directory / "parsec.out").read_text(encoding="utf-8")
            scf = np.load(directory / "parsec_python_results.npz")
            scf = {key: scf[key] for key in scf.files}
            bands_dat = read_bands_dat(directory / "bands.dat")
            archive = load_band_structure_archive(directory / "parsec_python_bands.npz")
            plot_text = (directory / "bands_plot.dat").read_text(encoding="utf-8")
            return log, scf, bands_dat, archive, plot_text

    def test_spin_orbit_bands(self) -> None:
        log, scf, bands, archive, plot_text = self._run(
            so_psp="true", extra="\nSO_from_scratch: .true.\n"
        )
        self.assertIn("Band structure (non-self-consistent", log)
        self.assertEqual(archive.mode, "soc")
        self.assertEqual(archive.channels, ("all",))
        self.assertEqual(archive.eigenvalues.shape, (1, 3, 20))
        np.testing.assert_allclose(archive.k_points[:, 0], [-_K, 0.0, _K], atol=1e-12)
        self.assertEqual(archive.spin_expectation.shape, (1, 3, 20, 3))

        # bands.dat round-trips the archive (Ry, not shifted by E_F).
        np.testing.assert_allclose(bands.eigenvalues, archive.eigenvalues, rtol=1e-14)
        np.testing.assert_allclose(bands.k_points, archive.k_points, atol=1e-7)
        self.assertAlmostEqual(bands.fermi_level, float(scf["fermi_level_ry"]), places=10)
        self.assertEqual(bands.segment_counts.tolist(), [3])
        self.assertEqual(plot_text.count("\n \n"), 20)  # one block per state

        e = archive.eigenvalues[0]
        # Time reversal of a non-magnetic system: E(-k) = E(+k).
        np.testing.assert_allclose(e[0], e[2], atol=5e-3)
        # Inversion + time reversal: Kramers pairs at every k-point.
        for point in range(3):
            for state in range(0, 14, 2):
                self.assertAlmostEqual(e[point, state], e[point, state + 1], delta=5e-3)
        # The bands at the SCF k-points are the SCF eigenvalues (k-major).
        pooled = scf["eigenvalues_ry"].reshape(2, 20)
        np.testing.assert_allclose(e[0, :14], pooled[0, :14], atol=5e-3)
        np.testing.assert_allclose(e[2, :14], pooled[1, :14], atol=5e-3)
        # |<sigma>| <= 1 for a normalized spinor.
        self.assertLess(np.abs(archive.spin_expectation).max(), 1.0 + 1e-6)

    def test_scalar_bands_match_the_scf_levels(self) -> None:
        log, scf, bands, archive, _ = self._run(so_psp="false", states=12)
        self.assertEqual(archive.mode, "scalar")
        self.assertIsNone(archive.spin_expectation)
        pooled = scf["eigenvalues_ry"].reshape(2, 12)
        np.testing.assert_allclose(archive.eigenvalues[0, 0, :8], pooled[0, :8], atol=5e-3)
        np.testing.assert_allclose(archive.eigenvalues[0, 2, :8], pooled[1, :8], atol=5e-3)
        # Bands rise from the Gamma-point bottom; Gamma (middle point) is below both ends.
        self.assertLess(archive.eigenvalues[0, 1, 0], archive.eigenvalues[0, 0, 0] + 1e-6)

    def test_spin_polarized_bands_have_two_channels(self) -> None:
        _, scf, bands, archive, _ = self._run(
            so_psp="false", extra="\nSpin_Polarization: .true.\n", states=12
        )
        self.assertEqual(archive.mode, "spin_polarized")
        self.assertEqual(archive.channels, ("up", "down"))
        self.assertEqual(archive.eigenvalues.shape, (2, 3, 12))
        self.assertEqual(bands.eigenvalues.shape, (2, 3, 12))
        # Au has a net moment: the channels are not identical.
        self.assertGreater(np.abs(archive.eigenvalues[0] - archive.eigenvalues[1]).max(), 1e-3)
        up = scf["eigenvalues_up_ry"].reshape(2, 12)
        np.testing.assert_allclose(archive.eigenvalues[0, 0, :8], up[0, :8], atol=5e-3)

    def test_noncollinear_bands(self) -> None:
        _, scf, _, archive, _ = self._run(
            so_psp="false",
            states=20,
            extra=(
                "\nNon_Collinear_magnetism: .true.\n"
                "begin Initial_NCL_Moment\n0 0 1\nend Initial_NCL_Moment\n"
            ),
        )
        self.assertEqual(archive.mode, "noncollinear")
        self.assertEqual(archive.eigenvalues.shape, (1, 3, 20))
        pooled = scf["eigenvalues_ry"].reshape(2, 20)
        np.testing.assert_allclose(archive.eigenvalues[0, 0, :10], pooled[0, :10], atol=5e-3)

    def test_perturbative_soc_route_is_rejected_before_the_scf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            (directory / "parsec.in").write_text(
                _INPUT.format(
                    states=20,
                    so_psp="true",
                    extra="\nSpin_Polarization: .true.\n",
                    kpoints=_MP,
                    bands=_PATH,
                ),
                encoding="utf-8",
            )
            self.assertEqual(cli_main([str(directory / "parsec.in"), "--quiet"]), 2)
            self.assertFalse((directory / "bands.dat").exists())


class BandsFromArchiveTests(unittest.TestCase):
    """``--bands-from``: bands from the converged potential of a previous run."""

    def _setup(
        self, directory: Path, extra: str = "", states: int = 12, bands: str = _PATH, so_psp: str = "false"
    ) -> Path:
        shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
        (directory / "parsec.in").write_text(
            _INPUT.format(states=states, so_psp=so_psp, extra=extra, kpoints=_MP, bands=bands),
            encoding="utf-8",
        )
        return directory / "parsec.in"

    def test_reproduces_the_bands_of_the_scf_run_without_touching_its_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            parsec_in = self._setup(directory)
            self.assertEqual(cli_main([str(parsec_in), "--quiet"]), 0)
            first = load_band_structure_archive(directory / "parsec_python_bands.npz")
            scf_log = (directory / "parsec.out").read_text(encoding="utf-8")
            archive = np.load(directory / "parsec_python_results.npz")
            self.assertIn("output_effective_potential_ry", archive.files)

            (directory / "parsec_python_bands.npz").unlink()
            code = cli_main(
                [str(parsec_in), "--bands-from", str(directory / "parsec_python_results.npz"), "--quiet"]
            )
            self.assertEqual(code, 0)
            second = load_band_structure_archive(directory / "parsec_python_bands.npz")
            np.testing.assert_allclose(second.eigenvalues, first.eigenvalues, atol=1e-8)
            self.assertAlmostEqual(second.fermi_level, first.fermi_level, places=10)
            self.assertEqual((directory / "parsec.out").read_text(encoding="utf-8"), scf_log)
            self.assertIn(
                "no SCF", (directory / "parsec_bands.out").read_text(encoding="utf-8")
            )

    def test_spin_orbit_archive_stores_the_potentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            parsec_in = self._setup(
                directory, extra="\nSO_from_scratch: .true.\n", states=20, so_psp="true"
            )
            # Write the SCF archive without computing bands first (bandstruc removed).
            no_bands = directory / "scf_only.in"
            no_bands.write_text(
                parsec_in.read_text(encoding="utf-8").replace(_PATH, ""), encoding="utf-8"
            )
            self.assertEqual(cli_main([str(no_bands), "--quiet", "--output", "scf.npz"]), 0)
            archive = np.load(directory / "scf.npz")
            self.assertIn("scf_potential_effective_potential", archive.files)
            self.assertEqual(
                archive["scf_potential_effective_potential"].shape[0], archive["density_e_per_bohr3"].shape[0]
            )
            code = cli_main([str(parsec_in), "--bands-from", str(directory / "scf.npz"), "--quiet"])
            self.assertEqual(code, 0)
            bands = load_band_structure_archive(directory / "parsec_python_bands.npz")
            self.assertEqual(bands.mode, "soc")
            e = bands.eigenvalues[0]
            np.testing.assert_allclose(e[0], e[2], atol=5e-3)  # time reversal

    def test_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            parsec_in = self._setup(directory)
            self.assertEqual(
                cli_main([str(parsec_in), "--bands-from", str(directory / "missing.npz"), "--quiet"]), 2
            )
            no_bands = self._setup(directory, bands="")
            self.assertEqual(
                cli_main([str(no_bands), "--bands-from", str(directory / "missing.npz"), "--quiet"]), 2
            )


class BandRetryTests(unittest.TestCase):
    """A CHEBDAV result with a large residual is re-solved with a larger block."""

    def test_scrambled_first_attempt_is_retried_and_recorded(self) -> None:
        from unittest import mock

        from parsec_python.Eigensolvers import solve_eigval as real_solve
        from parsec_python.driver import prepare_periodic_single_point
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.SCF import bands as bands_module
        from parsec_python.SCF.kpoints import run_scf_kpoints

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            (directory / "parsec.in").write_text(
                _INPUT.format(states=12, so_psp="false", extra="", kpoints=_MP, bands=_PATH),
                encoding="utf-8",
            )
            problem = parse_parsec_input(directory / "parsec.in").problem
            system = prepare_periodic_single_point(problem)
            k_points, weights = kpoint_sampling(problem)
            scf = run_scf_kpoints(system, k_points, weights)
            potentials = bands_module.band_potentials(scf, "scalar")

            blocks_used: list[int] = []

            def flaky(operator, requested, *, settings, state):
                blocks_used.append(settings.chebdav.block_size)
                solution = real_solve(operator, requested, settings=settings, state=state)
                if len(blocks_used) % 2 == 1:  # every first attempt returns garbage vectors
                    rng = np.random.default_rng(len(blocks_used))
                    garbage = rng.normal(size=solution.vectors.shape) + 0j
                    garbage /= np.linalg.norm(garbage, axis=0)
                    solution = dataclasses.replace(solution, vectors=garbage)
                return solution

            with mock.patch.object(bands_module, "solve_eigval", flaky):
                result = bands_module.run_band_structure(
                    system,
                    problem.bands,
                    mode="scalar",
                    potentials=potentials,
                    fermi_level=scf.fermi_level,
                    number_of_states=12,
                )
            self.assertTrue(np.all(result.attempts == 2))
            self.assertEqual(blocks_used[:2], [6, 8])  # first the configured block, then block + 2
            self.assertLess(result.residual_norms.max(), 2e-2)
            reference = bands_module.run_band_structure(
                system, problem.bands, mode="scalar", potentials=potentials,
                fermi_level=scf.fermi_level, number_of_states=12,
            )
            self.assertTrue(np.all(reference.attempts == 1))


class ParallelBandTests(unittest.TestCase):
    """Band k-points are independent: workers must not change the bands."""

    def test_workers_reproduce_serial_bands_and_report_every_k_point(self) -> None:
        from parsec_python.driver import prepare_periodic_single_point
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.SCF import bands as bands_module
        from parsec_python.SCF.kpoints import run_scf_kpoints

        longer_path = (
            "begin bandstruc\n1 -0.25 0.0 0.0  0.25 0.0 0.0  left-right\nend bandstruc\n"
            "bandstruc_points 6\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            (directory / "parsec.in").write_text(
                _INPUT.format(
                    states=12, so_psp="false", extra="", kpoints=_MP, bands=longer_path
                ),
                encoding="utf-8",
            )
            problem = parse_parsec_input(directory / "parsec.in").problem
            system = prepare_periodic_single_point(problem)
            k_points, weights = kpoint_sampling(problem)
            scf = run_scf_kpoints(system, k_points, weights)
            potentials = bands_module.band_potentials(scf, "scalar")

            def bands(**options):
                seen = []
                result = bands_module.run_band_structure(
                    system,
                    problem.bands,
                    mode="scalar",
                    potentials=potentials,
                    fermi_level=scf.fermi_level,
                    number_of_states=12,
                    callback=lambda index, count, values, residual, attempts: seen.append(
                        (index, count, attempts)
                    ),
                    **options,
                )
                return result, seen

            serial, seen_serial = bands(workers=1)
            self.assertEqual(serial.k_points.shape[0], 7)
            for options in (
                dict(workers=3),
                dict(workers=2),
                dict(backend="numpy-device"),  # the GPU code path, on the host
            ):
                with self.subTest(**options):
                    parallel, seen = bands(**options)
                    np.testing.assert_allclose(
                        parallel.eigenvalues, serial.eigenvalues, atol=1e-9
                    )
                    np.testing.assert_allclose(parallel.k_points, serial.k_points)
                    np.testing.assert_allclose(
                        parallel.residual_norms, serial.residual_norms, atol=1e-9
                    )
                    self.assertEqual(sorted(index for index, _, _ in seen), list(range(7)))
                    self.assertTrue(all(count == 7 for _, count, _ in seen))
            self.assertEqual([index for index, _, _ in seen_serial], list(range(7)))


class BandFileTests(unittest.TestCase):
    def test_write_read_roundtrip_with_two_segments(self) -> None:
        from parsec_python.models import BandStructureResult

        rng = np.random.default_rng(0)
        result = BandStructureResult(
            mode="spin_polarized",
            channels=("up", "down"),
            segment_names=("a-b", "b-c"),
            segment_counts=np.array([3, 4]),
            k_points=rng.normal(size=(7, 3)),
            eigenvalues=rng.normal(size=(2, 7, 5)),
            residual_norms=np.zeros((2, 7)),
            fermi_level=-0.123456789012,
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_bands_files(tmp, result)
            read = read_bands_dat(Path(tmp) / "bands.dat")
        np.testing.assert_allclose(read.eigenvalues, result.eigenvalues, rtol=1e-14)
        np.testing.assert_allclose(read.k_points, result.k_points, atol=1e-7)
        self.assertEqual(read.segment_counts.tolist(), [3, 4])
        self.assertEqual(read.channels, ("up", "down"))
        self.assertAlmostEqual(read.fermi_level, result.fermi_level, places=12)
        self.assertEqual(result.segment_index.tolist(), [0, 0, 0, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
