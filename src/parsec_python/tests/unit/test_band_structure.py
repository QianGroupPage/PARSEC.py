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
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
from parsec_python.tests.archives import read_archive

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


def _cupy_visible() -> bool:
    """Whether CuPy sees a CUDA device.  Then a band run left at its default backend
    (``PARSEC_DEVICE=auto``) solves its k-points on the GPU, with the device-resident
    eigensolver and one worker per device whatever ``workers`` says."""
    try:
        from parsec_python.acceleration.Hamiltonian.device_bloch import resolve_kpoint_backend
    except ImportError:
        return False
    return resolve_kpoint_backend("auto").name == "cupy"


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
        # The default rule (this port's, and the previous tree's): truncation.
        starts = np.array([[0, 0, 0], [1, 0, 0.0]])
        ends = np.array([[1, 0, 0], [1, 1.7, 0.0]])
        segments = band_path_kpoints(starts, ends, 3)
        self.assertEqual(len(segments[1]), int(1.7 * 3) + 1)
        starts = np.array([[0, 0, 0], [1, 0, 0.0], [1, 1.9, 0.0]])
        ends = np.array([[1, 0, 0], [1, 1.9, 0.0], [1, 1.9, 1.0]])
        self.assertEqual([len(s) for s in band_path_kpoints(starts, ends, 3)], [3, 5, 3 + 1])

    def test_count_is_rounded_like_parsec15_nint(self) -> None:
        # PARSEC 1.5 calc_bands.f90:152-158 rounds (nint), the rule of a
        # written Bandstruc_Points: 1.7*3 = 5.1 -> 5 (+1 closing point) either
        # way, but 1.9*3 = 5.7 -> 6, and an interior 2.5*2 = 5.0 stays 5.
        starts = np.array([[0, 0, 0], [1, 0, 0.0]])
        ends = np.array([[1, 0, 0], [1, 1.7, 0.0]])
        segments = band_path_kpoints(starts, ends, 3, rounding="nint")
        self.assertEqual(len(segments[1]), 5 + 1)
        ends = np.array([[1, 0, 0], [1, 1.9, 0.0]])
        segments = band_path_kpoints(starts, ends, 3, rounding="nint")
        self.assertEqual(len(segments[1]), 6 + 1)
        starts = np.array([[0, 0, 0], [1, 0, 0.0], [1, 1.9, 0.0]])
        ends = np.array([[1, 0, 0], [1, 1.9, 0.0], [1, 1.9, 1.0]])
        segments = band_path_kpoints(starts, ends, 3, rounding="nint")
        self.assertEqual([len(s) for s in segments], [3, 6, 3 + 1])
        # nint(x + 1) of the last line: 1.5*3 + 1 = 5.5 -> 6.
        starts = np.array([[0, 0, 0], [1, 0, 0.0]])
        ends = np.array([[1, 0, 0], [1, 1.5, 0.0]])
        self.assertEqual(len(band_path_kpoints(starts, ends, 3, rounding="nint")[1]), 6)
        self.assertEqual(len(band_path_kpoints(starts, ends, 3)[1]), 5)
        with self.assertRaises(ValueError):
            band_path_kpoints(starts, ends, 3, rounding="round")

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
        # This port's default, 20 with truncation (PARSEC 1.5's is 100, with
        # nint): this port keeps its defaults where 1.5 changed them.
        self.assertEqual(bands.points, 20)
        self.assertEqual(bands.rounding, "truncate")
        self.assertIsNone(bands.ff_maxiter)
        self.assertNotIn("ff_maxiter", repr(bands))
        self.assertNotIn("rounding", repr(bands))
        self.assertEqual(bands.segments[0].name, "")
        # a written Bandstruc_Points takes 1.5's rounding, also at 20
        written = self._parse(
            "begin bandstruc\n1 0 0 0 0.5 0 0\nend bandstruc\nbandstruc_points 20\n"
        ).problem.bands
        self.assertEqual((written.points, written.rounding), (20, "nint"))
        self.assertIn("rounding='nint'", repr(written))

    def test_bands_ff_maxiter(self) -> None:
        translation = self._parse(
            "begin bandstruc\n1 0 0 0 0.5 0 0\nend bandstruc\nBands_FF_MaxIter 4\n"
        )
        self.assertEqual(translation.problem.bands.ff_maxiter, 4)
        self.assertIn("ff_maxiter=4", repr(translation.problem.bands))
        # The input's eigensolver is chebdav: the cycles of CHEBFF do nothing.
        self.assertTrue(any("Bands_FF_MaxIter" in note for note in translation.warnings))
        with self.assertRaises(ParsecInputError):
            self._parse("begin bandstruc\n1 0 0 0 0.5 0 0\nend bandstruc\nBands_FF_MaxIter 0\n")

    def test_points_without_a_block_are_read_and_ignored(self) -> None:
        # PARSEC 1.5 reads Bandstruc_Points with or without the block.
        translation = self._parse("bandstruc_points 10\n")
        self.assertIsNone(translation.problem.bands)
        self.assertTrue(any("Bandstruc_Points" in note for note in translation.warnings))
        with self.assertRaises(ParsecInputError):
            self._parse("bandstruc_points 0\n")

    def test_no_bands_by_default(self) -> None:
        self.assertIsNone(self._parse("").problem.bands)

    def test_input_errors(self) -> None:
        cases = {
            "short line": "begin bandstruc\n1 0 0 0 0.5 0\nend bandstruc\n",
            "zero length segment": "begin bandstruc\n1 0 0 0 0 0 0 a-a\nend bandstruc\n",
            "non-numeric":"begin bandstruc\n1 0 0 0 x 0 0 a-b\nend bandstruc\n",
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
            scf = read_archive(directory / "parsec_python_results.npz")
            scf = {key: scf[key] for key in scf.files}
            bands_dat = read_bands_dat(directory / "bands.dat")
            archive = load_band_structure_archive(directory / "parsec_python_bands.npz")
            plot_text = (directory / "bands_plot.dat").read_text(encoding="utf-8")
            spin_path = directory / "bands_spin.dat"
            self.spin_text = spin_path.read_text(encoding="utf-8") if spin_path.exists() else None
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

        # bands_spin.dat: a block of k-points per state, columns k, path, E[eV], Sx, Sy, Sz, level size
        self.assertIsNotNone(self.spin_text)
        rows = [line.split() for line in self.spin_text.splitlines() if line.strip() and not line.startswith("#")]
        self.assertEqual(len(rows), 3 * 20)
        table = np.array(rows, dtype=float)
        np.testing.assert_allclose(table[:3, 2], archive.eigenvalues[0, :, 0] * 13.605693122994, rtol=1e-6)
        from parsec_python.Output.bands import level_averaged_spin

        averaged, sizes = level_averaged_spin(archive)  # what the file holds, at the default tolerance
        np.testing.assert_allclose(table[:, 3:6].reshape(20, 3, 3).transpose(1, 0, 2), averaged, atol=1e-5)
        # a crystal with inversion symmetry has Kramers-degenerate levels: the mean spin of a level is zero
        # (only the sum over the level is determined, and it cancels) whatever the spins of its states.  The
        # pairs of this loosely converged run agree to 5e-3 Ry, so group them at 1e-2 Ry (a group may hold two
        # pairs: still a sum of complete Kramers pairs)
        averaged, sizes = level_averaged_spin(archive, 1.0e-2)
        for point in range(3):
            for state in range(0, 14, 2):
                self.assertGreaterEqual(int(sizes[point, state]), 2)
                self.assertLess(np.abs(averaged[point, state]).max(), 0.1)

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
            archive = read_archive(directory / "parsec_python_results.npz")
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
            archive = read_archive(directory / "scf.npz")
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
    """A k-point whose first solve leaves a large residual is solved again, and the attempts recorded.

    Both eigensolver routes of the band workers go through the re-solve of
    ``solve_kpoint_bands``: the host CHEBDAV (the CPU backend) again with a block larger by
    two, and the device-resident first solve of a device worker (``DeviceBlochConverter.eigensolve``,
    the route of a GPU) again from another random start.  Every run pins its backend: left at
    the default, a visible GPU takes the k-points on the device-resident route, which never calls
    the host ``solve_eigval`` that the host case scrambles.
    """

    def test_scrambled_first_attempt_is_retried_and_recorded(self) -> None:
        from unittest import mock

        from parsec_python.Eigensolvers import solve_eigval as real_solve
        from parsec_python.driver import prepare_periodic_single_point
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.SCF import bands as bands_module
        from parsec_python.SCF.kpoints import run_scf_kpoints

        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        directory = Path(scratch.name)
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

        def run(**options):
            return bands_module.run_band_structure(
                system,
                problem.bands,
                mode="scalar",
                potentials=potentials,
                fermi_level=scf.fermi_level,
                number_of_states=12,
                **options,
            )

        # The host route, with one worker: the odd/even count below marks the first attempt of
        # every k-point only when the k-points are solved one after the other.
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

        with self.subTest(route="host CHEBDAV", backend="cpu"):
            with mock.patch.object(bands_module, "solve_eigval", flaky):
                result = run(workers=1, backend="cpu")
            self.assertTrue(np.all(result.attempts == 2))
            self.assertEqual(blocks_used[:2], [6, 8])  # first the configured block, then block + 2
            self.assertLess(result.residual_norms.max(), 2e-2)
            reference = run(workers=1, backend="cpu")
            self.assertTrue(np.all(reference.attempts == 1))

        # The device-resident route: the device code on the host everywhere, CuPy on every
        # visible GPU.  The first attempt of every k-point (attempt 0) returns garbage vectors
        # with their true residual.
        try:
            from parsec_python.acceleration.Hamiltonian.device_bloch import DeviceBlochConverter
        except ImportError:
            return
        real_eigensolve = DeviceBlochConverter.eigensolve
        attempts_seen: list[int] = []

        def scrambled(converter, hamiltonian, number_of_states, settings, attempt=0):
            attempts_seen.append(attempt)
            values, vectors, residual = real_eigensolve(
                converter, hamiltonian, number_of_states, settings, attempt
            )
            if attempt == 0:
                rng = np.random.default_rng(len(attempts_seen))
                garbage = rng.normal(size=vectors.shape) + 0j
                garbage /= np.linalg.norm(garbage, axis=0)
                applied = hamiltonian.as_linear_operator() @ garbage
                residual = float(np.max(np.linalg.norm(applied - garbage * values[None, :], axis=0)))
                vectors = garbage
            return values, vectors, residual

        routes = [("numpy-device", {"PARSEC_KPOINT_WORKERS": "1"})]
        if _cupy_visible():
            routes.append(("cupy", {}))
        for backend, environment in routes:
            environment = {"PARSEC_KPOINT_SOLVER": "device", **environment}
            with self.subTest(route="device-resident", backend=backend), mock.patch.dict(
                os.environ, environment
            ):
                reference = run(backend=backend)
                self.assertTrue(np.all(reference.attempts == 1))
                n_k = len(reference.k_points)
                attempts_seen.clear()
                with mock.patch.object(DeviceBlochConverter, "eigensolve", scrambled):
                    retried = run(backend=backend)
                self.assertTrue(np.all(retried.attempts == 2))
                # one scrambled first attempt and one fresh start (seed + 1) per k-point
                self.assertEqual(sorted(attempts_seen), [0] * n_k + [1] * n_k)
                self.assertLess(retried.residual_norms.max(), 2e-2)

    def test_bands_ff_maxiter_sets_the_chebff_cycles_of_the_band_kpoints(self) -> None:
        # PARSEC 1.5 calc_bands.f90:336-338: solver%ff_maxiter = bands_ff_maxiter
        # around the band eigval call, FF_MaxIter for the SCF.
        from unittest import mock

        from parsec_python.Eigensolvers import solve_eigval as real_solve
        from parsec_python.driver import prepare_periodic_single_point
        from parsec_python.Grid import kpoint_sampling
        from parsec_python.SCF import bands as bands_module
        from parsec_python.SCF.kpoints import run_scf_kpoints

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            shutil.copy(_AU_POTRE, directory / "Au_POTRE.DAT")
            text = _INPUT.format(states=12, so_psp="false", extra="", kpoints=_MP, bands=_PATH)
            text = text.replace("Eigensolver: chebdav", "Eigensolver: chebff")
            (directory / "parsec.in").write_text(text + "Bands_FF_MaxIter 4\n", encoding="utf-8")
            problem = parse_parsec_input(directory / "parsec.in").problem
            self.assertEqual(problem.bands.ff_maxiter, 4)
            system = prepare_periodic_single_point(problem)
            k_points, weights = kpoint_sampling(problem)
            scf = run_scf_kpoints(system, k_points, weights)
            potentials = bands_module.band_potentials(scf, "scalar")

            cycles: list[int] = []

            def recording(operator, requested, *, settings, state):
                cycles.append(settings.chebff.filter_cycles)
                return real_solve(operator, requested, settings=settings, state=state)

            for bands, expected in (
                (problem.bands, 4),
                (dataclasses.replace(problem.bands, ff_maxiter=None), problem.eigensolver.first_filter_cycles),
            ):
                cycles.clear()
                with mock.patch.object(bands_module, "solve_eigval", recording):
                    bands_module.run_band_structure(
                        system,
                        bands,
                        mode="scalar",
                        potentials=potentials,
                        fermi_level=scf.fermi_level,
                        number_of_states=12,
                        workers=1,
                        backend="cpu",
                    )
                self.assertEqual(len(cycles), 3)
                self.assertEqual(set(cycles), {expected})


class ParallelBandTests(unittest.TestCase):
    """Band k-points are independent: workers must not change the bands.

    The reference is the readable definition, pinned to the CPU backend with one worker: the
    host eigensolver, one k-point after the other.  Left at the default backend
    (``PARSEC_DEVICE=auto``) a visible GPU would take it on the device-resident route, with one
    worker per GPU whatever ``workers`` says: neither serial nor the route of the host-solver
    runs it is compared with.  A run on the same eigensolver route reproduces the reference to
    round-off whatever its workers and device; the device-resident first solve is another
    algorithm and reaches the same bands to the eigensolver tolerance.
    """

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

            serial, seen_serial = bands(workers=1, backend="cpu")
            self.assertEqual(serial.k_points.shape[0], 7)
            self.assertEqual([index for index, _, _ in seen_serial], list(range(7)))
            gpu = _cupy_visible()

            def reported_every_k_point(seen):
                # several workers report in completion order
                self.assertEqual(sorted(index for index, _, _ in seen), list(range(7)))
                self.assertTrue(all(count == 7 for _, count, _ in seen))

            # The host eigensolver: the same computation as the serial run, identical to
            # round-off.  "host" on a device is the route with the host eigensolver and only the
            # filter on the device; CuPy runs it on every visible GPU.
            host_routes = [
                (dict(workers=3, backend="cpu"), {}),
                (dict(workers=2, backend="cpu"), {}),
                (
                    dict(backend="numpy-device"),
                    {"PARSEC_KPOINT_SOLVER": "host", "PARSEC_KPOINT_WORKERS": "2"},
                ),
            ]
            if gpu:
                host_routes.append((dict(backend="cupy"), {"PARSEC_KPOINT_SOLVER": "host"}))
            for options, environment in host_routes:
                with self.subTest(**options, **environment), mock.patch.dict(os.environ, environment):
                    parallel, seen = bands(**options)
                    np.testing.assert_allclose(
                        parallel.eigenvalues, serial.eigenvalues, atol=1e-9
                    )
                    np.testing.assert_allclose(parallel.k_points, serial.k_points)
                    np.testing.assert_allclose(
                        parallel.residual_norms, serial.residual_norms, atol=1e-9
                    )
                    reported_every_k_point(seen)
            # The device-resident first solve (CHEBFF, with more CHEBFF cycles to CHEBDAV's
            # residual test for Eigensolver: chebdav) reaches the same bands to the eigensolver
            # tolerance: both routes stop at ||H x - e x|| <= Diag_Tolerance * max|e| (3.4e-4 Ry
            # here; CHEBDAV allows up to twice that for its last states), a test that leaves the
            # eigenvalues uncertain far above round-off (the two routes differ by 3e-7 Ry on the
            # top band, the host one 2.9e-7 Ry above the exact level).  Its own runs agree to
            # round-off whatever the worker count and device.  The host double and CuPy draw
            # their random start blocks from different generators: two different runs.
            device_routes = [
                ("numpy-device", ({"PARSEC_KPOINT_WORKERS": "1"}, {"PARSEC_KPOINT_WORKERS": "3"})),
            ]
            if gpu:
                device_routes.append(
                    ("cupy", ({"PARSEC_CUPY_DEVICES": "0"}, {"PARSEC_CUPY_DEVICES": "auto"}))
                )
            for backend, environments in device_routes:
                first = None
                for environment in environments:
                    environment = {"PARSEC_KPOINT_SOLVER": "device", **environment}
                    with self.subTest(route="device-resident", backend=backend, **environment), (
                        mock.patch.dict(os.environ, environment)
                    ):
                        device, seen = bands(backend=backend)
                        np.testing.assert_allclose(device.eigenvalues, serial.eigenvalues, atol=5e-5)
                        self.assertLess(float(device.residual_norms.max()), 5e-2)
                        reported_every_k_point(seen)
                        if first is None:
                            first = device
                        else:
                            np.testing.assert_allclose(device.eigenvalues, first.eigenvalues, atol=1e-9)
                            np.testing.assert_allclose(
                                device.residual_norms, first.residual_norms, atol=1e-9
                            )


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
