"""Non-self-consistent runs, the density of states and Calc_Hartree_G.

Fortran sources translated: ``calc_nscf.F90`` and the NSCF branch of
``flevel.f90`` (:mod:`parsec_python.SCF.nscf`), ``dos.F90`` and
``spec_tet.F90`` (:mod:`parsec_python.Output.dos`), ``hpot_g.F90``
(:mod:`parsec_python.Hartree.gspace`), the keywords of ``usrinputfile.F90``,
and ``parsec.dat`` on Fortran's general periodic grid.  The tests check the
input rules, exact properties (tetrahedron integrals, plane-wave Poisson
solutions, projections of model orbitals, the identity of a NSCF run with the
band machinery) and small end-to-end runs of the reference command line.
The comparisons with the Fortran binary are in the report of this stream.
"""


# With both sets of routines: the periodic inputs and grids of this module are those of the Fortran-derived
# routines (Use_Fortran_Routines: .true., PeriodicGridSettings(fortran_routines=True)); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from parsec_python.Grid.cell import reciprocal_lattice
from parsec_python.Grid.kpoints import monkhorst_pack_grid
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import ParsecInputError, UnsupportedParsecOptionError
from parsec_python.models import HartreeSettings, PeriodicCell, PeriodicGridSettings
from parsec_python.Output import dos as D
from parsec_python.Output.correction_output import list_directed_real
from parsec_python.SCF import nscf as N

DATA = Path(__file__).resolve().parents[1] / "data"

PERIODIC = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  6.0 6.0 6.0
end Cell_Shape
Grid_Spacing: 0.75
Expansion_Order: 4
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-5
Eigensolver: chebdav
Mixing_Method: Anderson
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
  3.1 2.9 3.05
end Atom_Coord
Correlation_Type: ca
{extra}
"""

CLUSTER = """\
Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.6 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 4
Hartree_Boundary_Tolerance: off
States_Num: 3
Fermi_Temp: 500
Max_Iter: 40
Convergence_Criterion: 1.0e-5
Eigensolver: chebff
FF_Maxiter: 9
Chebdav_Degree: 40
Mixing_Method: Anderson
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
    0.0  0.0  -0.7
    0.0  0.0   0.7
end Atom_Coord
Correlation_Type: ca
{extra}
"""

NSCF_BLOCK = """\
Restart_Run: .true.
nscf_kpoint_unit: reciprocal_lattice_vectors
begin nscf_kpoints
  0.0 0.0 0.0 1.0
  0.5 0.0 0.0 3.0
  0.25 0.25 0.0 4.0
end nscf_kpoints
"""


def _write(directory: Path, template: str, extra: str = "") -> Path:
    shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
    path = directory / "parsec.in"
    path.write_text(template.format(extra=extra), encoding="utf-8")
    return path


def _parse(template: str, extra: str = ""):
    with tempfile.TemporaryDirectory() as directory:
        return parse_parsec_input(_write(Path(directory), template, extra))


def _run_cli(path: Path, *extra: str) -> tuple[int, str]:
    from parsec_python.cli import main

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as error:
        code = main([str(path), "--quiet", *extra])
    return code, error.getvalue()


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------


class NSCFInputTests(unittest.TestCase):
    def test_keywords_follow_usrinputfile(self) -> None:
        translation = _parse(PERIODIC, NSCF_BLOCK)
        settings = translation.nscf
        reciprocal = reciprocal_lattice(np.diag([6.0, 6.0, 6.0]))
        expected = np.array([[0, 0, 0], [0.5, 0, 0], [0.25, 0.25, 0]]) @ reciprocal
        np.testing.assert_allclose(settings.k_points, expected)
        np.testing.assert_allclose(settings.weights, np.array([1, 3, 4]) / 8.0)
        self.assertEqual(settings.number_of_states, 4)  # States_Num
        more = _parse(PERIODIC, NSCF_BLOCK + "nscf_States_Num: 7\n")
        self.assertEqual(more.nscf.number_of_states, 7)
        cartesian = _parse(PERIODIC, NSCF_BLOCK.replace("nscf_kpoint_unit: reciprocal_lattice_vectors\n", ""))
        np.testing.assert_allclose(cartesian.nscf.k_points[1], [0.5, 0.0, 0.0])

    def test_errors_of_usrinputfile(self) -> None:
        with self.assertRaisesRegex(ParsecInputError, "error 119"):
            _parse(PERIODIC, NSCF_BLOCK.replace("3.0", "-3.0"))
        with self.assertRaisesRegex(ParsecInputError, "error 114"):
            _parse(PERIODIC, NSCF_BLOCK.replace("Restart_Run: .true.\n", ""))
        with self.assertRaisesRegex(ParsecInputError, "four numbers"):
            _parse(PERIODIC, NSCF_BLOCK.replace("0.5 0.0 0.0 3.0", "0.5 0.0 0.0"))
        with self.assertRaises(ParsecInputError):
            _parse(PERIODIC, NSCF_BLOCK.replace("reciprocal_lattice_vectors", "crystal"))
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "bandstruc"):
            _parse(
                PERIODIC,
                NSCF_BLOCK
                + "begin bandstruc\n1 0 0 0 0.5 0 0 GX\nend bandstruc\n",
            )

    def test_inert_keywords_are_named(self) -> None:
        warnings = _parse(
            PERIODIC,
            NSCF_BLOCK + "nscf_fermi_level 0.5 Ry\nbegin nscf_kgrid\n2 2 2\nend nscf_kgrid\n",
        ).warnings
        self.assertTrue(any("Output_GW" in item and "nscf_fermi_level" in item for item in warnings))
        warnings = _parse(PERIODIC, "nscf_States_Num: 5\n").warnings
        self.assertTrue(any("without a nscf_kpoints block" in item for item in warnings))
        translation = _parse(CLUSTER, "begin nscf_kpoints\n0 0 0 1\nend nscf_kpoints\n")
        self.assertIsNone(translation.nscf)
        self.assertTrue(any("periodic systems only" in item for item in translation.warnings))

    def test_spin_restart_is_allowed_for_nscf_only(self) -> None:
        spin = "Spin_Polarization: .true.\n"
        self.assertIsNotNone(_parse(PERIODIC, spin + NSCF_BLOCK).nscf)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Restart_Run"):
            _parse(PERIODIC, spin + "Restart_Run: .true.\n")


class DOSInputTests(unittest.TestCase):
    MP = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"

    def test_cluster_histogram_settings(self) -> None:
        translation = _parse(CLUSTER, "Create_DOS: .true.\nDOS_Pnum: 50\nYLM_DOS_L: 1\n")
        self.assertEqual(translation.dos.points, 50)
        self.assertEqual(translation.dos.ylm_l, -1)
        self.assertTrue(any("periodic systems only" in item for item in translation.warnings))
        self.assertIsNone(_parse(CLUSTER, "DOS_Pnum: 50\n").dos)

    def test_periodic_grid_rules(self) -> None:
        translation = _parse(PERIODIC, self.MP + "Create_DOS: .true.\nYLM_DOS_L: 2\nylm_dos_cutoff: 1.5 bohr\n")
        dos = translation.dos
        self.assertEqual(dos.monkhorst_pack_grid, (2, 1, 1))
        self.assertEqual(dos.points, 1000)
        self.assertEqual(dos.ylm_l, 2)
        self.assertEqual(dos.ylm_cutoffs, (("H", 1.5),))
        self.assertFalse(dos.fortran_projection)
        other = _parse(
            PERIODIC,
            self.MP + "Create_DOS: .true.\nbegin DOS_MP_Grid\n4 4 4\nend DOS_MP_Grid\n",
        )
        self.assertEqual(other.dos.monkhorst_pack_grid, (2, 1, 1))
        self.assertTrue(any("original MP Grid" in item for item in other.warnings))
        restarted = _parse(
            PERIODIC,
            self.MP
            + "Create_DOS: .true.\nRestart_Run: .true.\nbegin DOS_MP_Grid\n3 3 1\nend DOS_MP_Grid\n",
        )
        self.assertEqual(restarted.problem.monkhorst_pack_dimensions, (3, 3, 1))
        self.assertEqual(restarted.dos.monkhorst_pack_grid, (3, 3, 1))

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "kpoint map"):
            _parse(PERIODIC, "Create_DOS: .true.\n")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "YLM_DOS_L"):
            _parse(PERIODIC, self.MP + "Create_DOS: .true.\nYLM_DOS_L: 4\n")
        # DOS_MP_Shift with Restart_Run sets the shift of the restarted SCF
        # (usrinputfile.F90 l.1188-1193): Fortran's Monkhorst-Pack path.  The
        # test cell's translations do not fit its grid: no symmetry here.
        mp = "Ignore_Symmetry: .true.\n" + self.MP
        shifted = _parse(
            PERIODIC,
            mp
            + "Create_DOS: .true.\nRestart_Run: .true.\nbegin DOS_MP_Grid\n2 2 2\nend DOS_MP_Grid\n"
            "begin DOS_MP_Shift\n0.5 0.5 0.5\nend DOS_MP_Shift\n",
        )
        self.assertEqual(shifted.problem.monkhorst_pack_shift, (0.5, 0.5, 0.5))
        self.assertEqual(shifted.problem.monkhorst_pack_dimensions, (2, 2, 2))
        self.assertIsNotNone(shifted.problem.space_group)
        self.assertEqual(shifted.dos.monkhorst_pack_shift, (0.5, 0.5, 0.5))
        # The SCF's shift block with DOS_MP_Grid but no DOS_MP_Shift: zero.
        zero = _parse(
            PERIODIC,
            mp
            + "begin Monkhorst_Pack_Shift\n0.5 0.5 0.5\nend Monkhorst_Pack_Shift\n"
            "Create_DOS: .true.\nRestart_Run: .true.\nbegin DOS_MP_Grid\n2 2 2\nend DOS_MP_Grid\n",
        )
        self.assertEqual(zero.problem.monkhorst_pack_shift, (0.0, 0.0, 0.0))
        # Without DOS_MP_Grid the SCF keeps its own grid and shift.
        kept = _parse(
            PERIODIC,
            mp
            + "begin Monkhorst_Pack_Shift\n0.5 0.5 0.5\nend Monkhorst_Pack_Shift\n"
            "Create_DOS: .true.\nRestart_Run: .true.\n",
        )
        self.assertEqual(kept.problem.monkhorst_pack_shift, (0.5, 0.5, 0.5))
        self.assertEqual(kept.problem.monkhorst_pack_dimensions, (2, 1, 1))
        manual = (
            "Kpoint_Method: manual\nbegin Kpoint_List\n0 0 0 1\n0.1 0 0 1\nend Kpoint_List\n"
            "Create_DOS: .true.\n"
        )
        translation = _parse(PERIODIC, manual)
        self.assertIsNone(translation.dos)
        self.assertTrue(any("DOS creation aborted" in item for item in translation.warnings))


class HartreeKeywordTests(unittest.TestCase):
    def test_repr_hides_the_default(self) -> None:
        text = repr(HartreeSettings())
        self.assertNotIn("periodic_g_space", text)
        self.assertEqual(
            text,
            "HartreeSettings(multipole_order=9, relative_tolerance=1e-07, "
            "absolute_tolerance=1e-13, max_iterations=1600, boundary_method='auto', "
            "direct_chunk_size=16, boundary_tolerance=0.001, atomic_tail='auto', "
            "minimum_multipole_order=None)",
        )
        self.assertIn("periodic_g_space=True", repr(HartreeSettings(periodic_g_space=True)))

    def test_keyword(self) -> None:
        translation = _parse(PERIODIC, "Calc_Hartree_G: .true.\n")
        self.assertTrue(translation.problem.hartree.periodic_g_space)
        self.assertTrue(any("IT WILL NOT WORK" in item for item in translation.warnings))
        cluster = _parse(CLUSTER, "Calc_Hartree_G: .true.\n")
        self.assertFalse(cluster.problem.hartree.periodic_g_space)
        self.assertFalse(_parse(PERIODIC).problem.hartree.periodic_g_space)


# ----------------------------------------------------------------------------
# Calc_Hartree_G
# ----------------------------------------------------------------------------


def _grid(lattice, spacing):
    cell = PeriodicCell(lattice_vectors=np.asarray(lattice, dtype=float))
    return build_periodic_grid(cell, PeriodicGridSettings(fortran_routines=True, spacing=spacing, expansion_order=4))


class GSpaceHartreeTests(unittest.TestCase):
    def check_plane_wave(self, grid) -> None:
        from parsec_python.Hartree.gspace import solve_hartree_g_space

        reciprocal = reciprocal_lattice(grid.cell.lattice_vectors)
        g = reciprocal[0] + 2 * reciprocal[2]
        density = 0.3 + np.cos(grid.coordinates @ g)
        potential = solve_hartree_g_space(density, grid)
        expected = 8.0 * np.pi / float(g @ g) * np.cos(grid.coordinates @ g)
        np.testing.assert_allclose(potential, expected, atol=1e-12)
        self.assertAlmostEqual(float(np.mean(potential)), 0.0, places=13)

    def test_plane_waves_orthorhombic_and_general(self) -> None:
        self.check_plane_wave(_grid(np.diag([5.0, 6.0, 7.0]) * 1.0, 0.5))
        fcc = 10.26 * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        grid = _grid(fcc, 0.6)
        self.assertTrue(getattr(grid, "is_general_periodic", False))
        self.check_plane_wave(grid)

    def test_dispatch_and_closeness_to_the_finite_difference_solve(self) -> None:
        from parsec_python.Hartree.pbc import solve_periodic_hartree
        from parsec_python.Laplacian import build_negative_laplacian

        grid = _grid(np.diag([6.0, 6.0, 6.0]), 0.4)
        centre = np.array([3.0, 3.1, 2.9])
        distance = np.linalg.norm(grid.coordinates - centre, axis=1)
        density = np.exp(-(distance**2))
        laplacian = build_negative_laplacian(grid)
        default = solve_periodic_hartree(density, grid, laplacian, HartreeSettings())
        spectral = solve_periodic_hartree(
            density, grid, laplacian, HartreeSettings(periodic_g_space=True)
        )
        self.assertTrue(spectral.converged)
        self.assertEqual(spectral.residual_norm, 0.0)
        scale = float(np.max(np.abs(default.potential)))
        self.assertLess(float(np.max(np.abs(spectral.potential - default.potential))), 1e-2 * scale)
        self.assertGreater(float(np.max(np.abs(spectral.potential - default.potential))), 0.0)


# ----------------------------------------------------------------------------
# Fermi level and occupations (flevel.f90)
# ----------------------------------------------------------------------------


class FermiLevelTests(unittest.TestCase):
    def test_bisection_counts_the_electrons(self) -> None:
        levels = [[np.array([-0.5, -0.1, 0.2, 0.4]), np.array([-0.4, 0.0, 0.3, 0.5])]]
        weights = [0.25, 0.75]
        fermi = N.fermi_level_from_levels(levels, weights, 3.0, 2000.0)
        count = sum(
            w * N.fixed_fermi_occupations(values, fermi, 2000.0).sum()
            for values, w in zip(levels[0], weights)
        )
        self.assertAlmostEqual(count, 1.5, places=11)  # pairs: xele/2
        spin = N.fermi_level_from_levels([levels[0], levels[0]], weights, 3.0, 2000.0)
        self.assertAlmostEqual(spin, fermi, places=9)

    def test_fixed_occupations(self) -> None:
        beta = 1.0 / N.KELVIN_TO_RYDBERG / 300.0
        values = np.array([-1.0, 0.0, 1.0, 35.5 / beta, -35.5 / beta])
        occupations = N.fixed_fermi_occupations(values, 0.0, 300.0)
        np.testing.assert_array_equal(occupations, [1.0, 0.5, 0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------
# Tetrahedra (dos.F90, spec_tet.F90)
# ----------------------------------------------------------------------------


class TetrahedronTests(unittest.TestCase):
    def test_heap_sort(self) -> None:
        generator = np.random.default_rng(3)
        for _ in range(200):
            values = generator.integers(0, 4, size=4).astype(float)
            order = D.heap_sort_order(values)
            self.assertEqual(sorted(order), [0, 1, 2, 3])
            self.assertTrue(np.all(np.diff(values[order]) >= 0))
        # Ties keep the order of the Numerical Recipes heap sort.
        self.assertEqual(D.heap_sort_order([1.0, 1.0, 1.0, 1.0]), [1, 2, 3, 0])
        self.assertEqual(D.heap_sort_order([2.0, 1.0]), [1, 0])

    def test_split_fills_the_cube(self) -> None:
        for reciprocal in (np.eye(3), reciprocal_lattice(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.0]]))):
            tset = D.tetrahedron_split(reciprocal)
            volumes = [abs(np.linalg.det((t[1:] - t[0]).astype(float))) / 6.0 for t in tset]
            self.assertAlmostEqual(sum(volumes), 1.0)
            # The six tetrahedra share one long diagonal of the cube.
            common = set(map(tuple, tset[0]))
            for t in tset[1:]:
                common &= set(map(tuple, t))
            self.assertEqual(len(common), 2)

    def test_one_tetrahedron(self) -> None:
        energies = np.array([[0.3, 0.0, 1.0, 2.0]])
        mesh = D.dos_energy_mesh(-0.5, 2.5, 3000)
        volume = 0.7
        dos, projected, equal = D.tetrahedron_spectrum(
            volume, energies, mesh, values=np.full((1, 4, 1), 2.5)
        )
        self.assertEqual(equal, 0)
        self.assertAlmostEqual(float(dos.sum() * mesh.step), volume, places=3)
        np.testing.assert_allclose(projected[0], 2.5 * dos, rtol=1e-12, atol=1e-14)
        # The linear-tetrahedron density between e1 and e2: 3 vol (E-e1)^2 / prod(e_j - e1).
        e = mesh.energies
        inside = (e > 0.0) & (e < 0.29)
        expected = 3.0 * volume * e[inside] ** 2 / (0.3 * 1.0 * 2.0)
        np.testing.assert_allclose(dos[inside], expected, rtol=1e-9)
        flat, _, count = D.tetrahedron_spectrum(volume, np.full((2, 4), 0.4), mesh)
        self.assertEqual(count, 2)
        self.assertEqual(float(np.abs(flat).sum()), 0.0)

    def test_band_integrates_to_one_over_nspin(self) -> None:
        # Bands without four equal corners in any tetrahedron (those add
        # nothing, as in spec_tet: |k|^2 itself has such cells on this grid).
        lattice = np.diag([4.0, 5.0, 6.0])
        dims = (4, 3, 5)
        k_points, _ = monkhorst_pack_grid(PeriodicCell(lattice_vectors=lattice), dims)
        base = np.sum(k_points**2, axis=1) + k_points @ np.array([0.31, 0.17, 0.23])
        band = base[:, None] * np.array([[1.0, 1.3]]) + np.array([[0.0, 0.9]])
        result = D.tetrahedron_dos(
            [band, band + 0.05],
            0.2,
            dimensions=dims,
            k_index=D.monkhorst_pack_k_index(dims),
            lattice_vectors=lattice,
            points=4000,
        )
        self.assertEqual(result.tetrahedra, 6 * 60)
        self.assertEqual(result.equal_corners, 0)
        for spin in range(2):
            integral = float(result.dos[spin].sum() * result.factor * result.mesh.step)
            self.assertAlmostEqual(integral, 2 * 0.5, places=2)

    def test_k_index_follows_the_python_grid(self) -> None:
        dims = (3, 2, 4)
        cell = PeriodicCell(lattice_vectors=np.diag([3.0, 4.0, 5.0]))
        k_points, _ = monkhorst_pack_grid(cell, dims)
        index = D.monkhorst_pack_k_index(dims)
        fractional = k_points @ cell.lattice_vectors.T / (2 * np.pi)
        # Increasing fractions along every axis of the index grid.
        self.assertTrue(np.all(np.diff(fractional[index[:, 0, 0], 0]) > 0))
        self.assertTrue(np.all(np.diff(fractional[index[0, 0, :], 2]) > 0))

    def test_histogram_and_file_lines(self) -> None:
        result = D.occupation_histogram(
            [np.array([-0.5, -0.2, 0.1])], [np.array([1.0, 0.5, 0.0])], -0.2, points=20
        )
        self.assertAlmostEqual(float(result.dos.sum()), 1.5)
        # The mesh starts 10 delta below the lowest level and its step is
        # (width + 20 delta)/N = 2 delta here: the lowest level is in bin 6.
        self.assertEqual(int(np.flatnonzero(result.dos[0])[0]), 5)
        lines = D.dos_file_lines(result)
        self.assertEqual(lines[0], " Density Of States - Units:  states/eV/unit cell vs. eV")
        self.assertEqual(lines[1], " Fermi levels:    -0.200000   -0.200000")
        self.assertEqual(lines[2], "       Tot. DOS: ")
        energy = result.mesh.lower + result.mesh.step
        self.assertEqual(lines[3], list_directed_real(energy) + list_directed_real(0.0))
        self.assertEqual(lines[-1], " &")
        self.assertEqual(len(lines), 3 + 20 + 1)
        spin = D.DOSResult(
            mesh=result.mesh,
            dos=np.vstack([result.dos, result.dos]),
            factor=1.0,
            fermi_level=-0.2,
            projected=np.zeros((2, 1, 4, 20)),
            tetrahedra=12,
            equal_corners=0,
        )
        lines = D.dos_file_lines(spin, atom_labels=["H"])
        self.assertEqual(lines[2], "       Spin DOS: ")
        sum_line = lines[3 + 2 * 21]
        self.assertEqual(len(sum_line), 100)
        self.assertEqual(sum_line.strip(), "Sum of spin DOS:")
        header = lines.index("LDOS : Atom Type =  H, Atom Num = 1, Ylm Num =  2, Spin =  1")
        self.assertEqual(lines[header + 1], "Graph Name : H1_px_sp1")
        report = D.dos_report_lines(spin, 0.25)
        self.assertEqual(
            report,
            [
                "",
                " Generation of DOS : ",
                " -----------------",
                " Tot. number of tetrahedra :                     12",
                " Number of tetrahedra will eqaul corners:         0",
                "",
                ' The DOS data has been writted to the file "DOS"',
                "",
                "",
                " TIME FOR DENSITY OF STATES:         0.2500",
            ],
        )


# ----------------------------------------------------------------------------
# Ylm projection (ylm_eig)
# ----------------------------------------------------------------------------


class YlmProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = _grid(np.diag([8.0, 8.0, 8.0]), 0.25)
        self.atom = np.array([4.02, 3.97, 4.05])
        self.difference = self.grid.coordinates - self.atom
        self.distance = np.linalg.norm(self.difference, axis=1)

    def normalized(self, values):
        values = np.asarray(values, dtype=complex)
        return values / np.linalg.norm(values)

    def weights(self, function, *, fortran=False, k=(0.0, 0.0, 0.0), l_max=2):
        return D.ylm_projection_weights(
            self.grid,
            self.atom[None, :],
            [1.6],
            [[function[:, None]]],
            np.asarray(k)[None, :],
            l_max=l_max,
            fortran_projection=fortran,
        )[0][0, 0, 0]

    def test_s_and_p_orbitals(self) -> None:
        # The Gaussian-smeared radial shells of ylm_eig (sigma = h) estimate
        # the charge in the sphere carried by each harmonic, to the accuracy
        # of the grid: a smooth s orbital is mostly s, a p_z orbital mostly p_z.
        s = self.normalized(np.exp(-0.3 * self.distance**2))
        inside = float(np.sum(np.abs(s[self.distance <= 1.6]) ** 2))
        weights = self.weights(s)
        self.assertAlmostEqual(weights[0], inside, delta=0.2 * inside)
        self.assertLess(float(np.max(weights[1:])), 2e-2 * weights[0])
        pz = self.normalized(self.difference[:, 2] * np.exp(-0.3 * self.distance**2))
        weights = self.weights(pz)
        self.assertEqual(int(np.argmax(weights)), 3)  # p_z
        others = np.delete(weights, 3)
        self.assertLess(float(np.max(others)), 2e-2 * weights[3])
        # PARSEC's s channel has no radial factor: another number; the
        # other channels are the same (no Bloch phase at k = 0).
        fixed = self.weights(s)
        fortran = self.weights(s, fortran=True)
        self.assertGreater(abs(fortran[0] - fixed[0]), 1e-2 * fixed[0])
        np.testing.assert_allclose(fortran[1:], fixed[1:], rtol=1e-12, atol=1e-18)

    def test_bloch_phase(self) -> None:
        constant = self.normalized(np.ones(self.grid.size))
        k = (0.0, 0.0, 2 * np.pi / 8.0)
        with_phase = self.weights(constant, k=k)
        without = self.weights(constant, k=k, fortran=True)
        # u_k = const is psi = exp(i k z): a p_z part around the atom, which
        # PARSEC's projection of u_k (no phase) misses.
        self.assertGreater(with_phase[3], 10.0 * without[3])
        self.assertGreater(with_phase[3], 1e-3)
        np.testing.assert_allclose(with_phase[[1, 2]], without[[1, 2]], rtol=0.05)

    def test_gauss_legendre(self) -> None:
        nodes, weights = D.gauss_legendre(0.0, 2.0, 12)
        self.assertAlmostEqual(float(np.sum(weights * nodes**5)), 2.0**6 / 6.0, places=12)
        self.assertEqual(D.radial_node_count(1.0, 0.6), 10)
        self.assertEqual(D.radial_node_count(2.0, 0.6045763), 17)


# ----------------------------------------------------------------------------
# parsec.dat on Fortran's general periodic grid
# ----------------------------------------------------------------------------


class GeneralGridParsecDatTests(unittest.TestCase):
    def test_round_trip_unfolding_and_restart_checks(self) -> None:
        from parsec_python.Output.parsec_dat import (
            ParsecDat,
            ParsecDatLevels,
            ParsecDatPeriodic,
            ParsecDatSymmetry,
            fortran_integer_coordinates,
            grid_rows_of_file,
            grid_shift,
            grid_steps,
            read_parsec_dat,
            system_type,
            write_parsec_dat,
        )
        from parsec_python.SCF.restart import RestartError, check_restart_file

        fcc = 10.26 * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        grid = _grid(fcc, 0.9)
        self.assertTrue(grid.is_general_periodic)
        np.testing.assert_array_equal(grid_steps(grid), grid.spacings)
        np.testing.assert_array_equal(grid_shift(grid), grid.shift)
        points = fortran_integer_coordinates(grid)
        self.assertEqual(int(points.min()), -int(grid.points_per_axis[0]) // 2)
        # An inversion-symmetric field, stored on half of the points with
        # the inversion as the second operation (r -> -r).
        reciprocal = reciprocal_lattice(fcc)
        field = np.cos(grid.coordinates @ (reciprocal[0] + reciprocal[1]))
        key = [tuple(row) for row in points]
        keep = []
        seen = set()
        for row, item in enumerate(key):
            image = tuple(int(v) for v in ((-np.asarray(item) + grid.points_per_axis // 2) % grid.points_per_axis - grid.points_per_axis // 2))
            if image in seen:
                continue
            seen.add(item)
            keep.append(row)
        keep = np.asarray(keep)
        symmetry = ParsecDatSymmetry(
            rmtrx=np.stack([np.eye(3, dtype=np.int32), -np.eye(3, dtype=np.int32)]),
            trans=np.stack([np.eye(3), -np.eye(3)]),
            tnp=np.zeros((2, 3)),
            alatt=fcc.T.copy(),
            invlat=np.linalg.inv(fcc),
            chi=np.ones((2, 2), dtype=np.int32),
        )
        data = ParsecDat(
            date_label="27-OCT-2026 12:00:00 +0000",
            system_type=system_type(1, 0, 1),
            step=grid_steps(grid),
            rmax=1.0,
            ndim=grid.size,
            shift=grid_shift(grid),
            symmetry=symmetry,
            points=points[keep],
            levels=((ParsecDatLevels(np.ones(2, dtype=np.int32), np.array([-0.3, 0.1]), np.array([1.0, 0.0])),),),
            potentials=(field[keep],),
            densities=(field[keep] ** 2,),
            periodic=ParsecDatPeriodic(box_size=np.linalg.norm(fcc, axis=1), lattice_vectors=fcc),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = write_parsec_dat(Path(folder) / "parsec.dat", data)
            back = read_parsec_dat(path)
        self.assertLess(keep.size, grid.size)
        mapping = grid_rows_of_file(back, grid)
        self.assertEqual(mapping.missing, 0)
        np.testing.assert_allclose(mapping.field(back.potentials[0]), field, atol=1e-12)
        self.assertFalse(check_restart_file(back, grid, spin_count=1, complex_wavefunctions=False))
        shifted = ParsecDat(**{**data.__dict__, "shift": np.array([0.5, 0.5, 0.5])})
        with self.assertRaises(RestartError):
            check_restart_file(shifted, grid, spin_count=1, complex_wavefunctions=False)
        # The Fermi level of the stored levels (flevel bisection).
        self.assertAlmostEqual(N.restart_fermi_level(back, 2.0, 300.0), -0.1, delta=0.2)


# ----------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------


class NSCFRunTests(unittest.TestCase):
    def test_nscf_equals_the_band_machinery_and_spin_channels(self) -> None:
        from parsec_python.SCF.bands import solve_kpoint_bands
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        translation = parse_parsec_input(_write(Path(folder.name), PERIODIC, NSCF_BLOCK))
        system = prepare_periodic_single_point(translation.problem)
        potential = np.asarray(system.ionic_potential) + 0.1
        settings = translation.nscf
        result = N.run_nscf(
            system,
            settings.k_points,
            settings.weights,
            potentials=[potential],
            number_of_states=3,
            fermi_level=-0.2,
            workers=1,
        )
        bands = solve_kpoint_bands(
            system,
            settings.k_points,
            mode="scalar",
            potentials={"effective_potential": potential},
            number_of_states=3,
            workers=1,
        )
        np.testing.assert_array_equal(result.eigenvalues, bands.eigenvalues)
        np.testing.assert_array_equal(
            result.occupations,
            N.fixed_fermi_occupations(bands.eigenvalues, -0.2, 500.0),
        )
        self.assertEqual(result.channels, ("all",))
        spin = N.run_nscf(
            system,
            settings.k_points,
            settings.weights,
            potentials=[potential, potential + 0.05],
            number_of_states=3,
            fermi_level=-0.2,
            workers=1,
        )
        self.assertEqual(spin.channels, ("up", "down"))
        np.testing.assert_allclose(spin.eigenvalues[1], spin.eigenvalues[0] + 0.05, atol=1e-6)

    def test_command_line_scf_then_nscf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, PERIODIC)
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            stored = (directory / "parsec.dat").read_bytes()
            scf_log = (directory / "parsec.out").read_text(encoding="utf-8")
            scf_fermi = float(re.findall(r"Fermi level at\s+(-?\d+\.\d+)", scf_log)[-1])
            with np.load(directory / "parsec_python_results.npz") as archive:
                scf_levels = np.asarray(archive["eigenvalues_ry"])
            path.write_text(PERIODIC.format(extra=NSCF_BLOCK), encoding="utf-8")
            code, error = _run_cli(path, "--log", "nscf.out")
            self.assertEqual(code, 0, error)
            text = (directory / "nscf.out").read_text(encoding="utf-8")
            self.assertIn("  Calculating non self consistent electronic structure", text)
            self.assertIn("    0.52360  0.00000  0.00000  0.37500", text)
            self.assertIn(" calling calc_nscf ", text)
            self.assertIn(" ...NSCF calculation done", text)
            self.assertNotIn("SCF iter #", text)
            self.assertEqual(text.count("   State   Eigenvalue [Ry]"), 3)
            self.assertIn(" k-point =    0.5000   0.0000   0.0000 (recip. latt. vec.) ", text)
            fermi = float(re.findall(r"Fermi level at\s+(-?\d+\.\d+)", text)[-1])
            self.assertAlmostEqual(fermi, scf_fermi, places=3)
            # NSCF writes no parsec.dat.
            self.assertEqual((directory / "parsec.dat").read_bytes(), stored)
            # The SCF archive is kept; the NSCF levels have their own.
            with np.load(directory / "parsec_python_results.npz") as archive:
                self.assertNotIn("nscf_eigenvalues_ry", archive.files)
            with np.load(directory / "parsec_python_nscf.npz") as archive:
                values = archive["nscf_eigenvalues_ry"]
                self.assertEqual(values.shape, (1, 3, 4))
            # At Gamma the NSCF levels are those of the SCF in the potential
            # the SCF converged to (vnew of the last iteration).
            np.testing.assert_allclose(values[0, 0, :2], scf_levels[:2], atol=2e-4)

    def test_command_line_dos_of_a_cluster_and_of_a_cell(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, CLUSTER, "Create_DOS: .true.\nDOS_Pnum: 40\n")
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            lines = (directory / "DOS").read_text().splitlines()
            self.assertEqual(len(lines), 3 + 40 + 1)
            total = sum(float(line.split()[1]) for line in lines[3:-1])
            self.assertAlmostEqual(total, 1.0, places=6)  # two electrons = one pair
            self.assertIn(" Generation of DOS : ", (directory / "parsec.out").read_text())
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(
                directory,
                PERIODIC,
                "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 1\nend Monkhorst_Pack_Grid\n"
                "Create_DOS: .true.\nDOS_Pnum: 60\nYLM_DOS_L: 1\nylm_dos_cutoff: 1.2\n",
            )
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            lines = (directory / "DOS").read_text().splitlines()
            self.assertEqual(len(lines), 3 + 61 + 4 * 63)
            self.assertIn("Graph Name : H1_pz_sp1", lines)
            text = (directory / "parsec.out").read_text()
            self.assertIn(" Tot. number of tetrahedra :                     24", text)


class AcceleratedRoutingTests(unittest.TestCase):
    def test_features_go_to_the_reference_solver(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_features

        for template, extra, name in (
            (PERIODIC, NSCF_BLOCK, "nscf_kpoints"),
            (
                PERIODIC,
                "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 1\nend Monkhorst_Pack_Grid\n"
                "Create_DOS: .true.\n",
                "Create_DOS",
            ),
        ):
            with self.subTest(name=name):
                features = _reference_only_features(
                    _parse(template, extra), accelerated_backend_selected=False
                )
                self.assertTrue(any(name in item for item in features), features)
        self.assertEqual(
            _reference_only_features(_parse(CLUSTER), accelerated_backend_selected=False), []
        )
        # The DOS of a cluster follows the accelerated SCF
        # (acceleration/Output/post_scf.py).
        self.assertEqual(
            _reference_only_features(
                _parse(CLUSTER, "Create_DOS: .true.\n"), accelerated_backend_selected=False
            ),
            [],
        )
        # Calc_Hartree_G of a Gamma-point cell runs on the accelerated driver
        # (acceleration/SCF/periodic_routes.py; its solve on the device).
        self.assertEqual(
            _reference_only_features(
                _parse(PERIODIC, "Calc_Hartree_G: .true.\n"), accelerated_backend_selected=False
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
