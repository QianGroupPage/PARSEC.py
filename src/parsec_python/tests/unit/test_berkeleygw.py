"""``Output_GW``: BerkeleyGW's WFN, RHO and VXC files (``gw_write.F90``, ``g_space.f90``).

The tests check the G-space machinery (sphere counts and order, ``findbound``,
``adjustfft``, the symmetry stars of ``g_space`` and the star average), the
binary layout through a write/read round trip, numbers of Fortran's reference
run of ``tests/BGW_interface`` (fcc Si, its potential sphere, FFT grid and
k-sphere sizes; record-by-record comparisons with the Fortran files are in
the report of this stream), the input rules of ``usrinputfile.F90``, and
small end-to-end runs of the reference command line on the orthorhombic grid
(SCF with k-points, spin) and on Fortran's general grid (SCF, then NSCF).
"""


# With both sets of routines: the periodic inputs of this module set Use_Fortran_Routines: .true. (the
# Fortran-derived routines); the numbers checked here are those of the Fortran parity round.
from __future__ import annotations

import math
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from parsec_python.Grid.gspace import (
    adjust_fft,
    box_half_widths,
    fast_fft_size,
    find_bound,
    fortran_lattice_metrics,
    g_space_stars,
    generate_gspace,
    k_gspace,
    potential_gspace,
    translation_fft_factors,
    wavefunction_cutoff,
)
from parsec_python.Grid.pbc import adjust_step
from parsec_python.Grid.space_group import find_space_group, identity_space_group
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.parsec_input import ParsecInputError, UnsupportedParsecOptionError
from parsec_python.models import Atom
from parsec_python.Output import berkeleygw as B

DATA = Path(__file__).resolve().parents[1] / "data"

_FCC = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]) * 10.2612
_SI_ATOMS = [Atom("Si", np.array(f) @ _FCC) for f in ([-0.125] * 3, [0.125] * 3)]

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
  3.1 2.9 2.35
  3.1 2.9 3.75
end Atom_Coord
Correlation_Type: ca
{extra}
"""

FCC_CELL = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
  0.0 3.2 3.2
  3.2 0.0 3.2
  3.2 3.2 0.0
end Cell_Shape
Grid_Spacing: 0.8
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
  0.0 0.0 0.0
end Atom_Coord
Correlation_Type: ca
{extra}
"""

MP_SCF = """\
Kpoint_Method: mp
begin Monkhorst_Pack_Grid
  2 2 2
end Monkhorst_Pack_Grid
Output_GW: .true.
"""

NSCF_GW = """\
Restart_Run: .true.
nscf_kpoint_unit: reciprocal_lattice_vectors
nscf_states_num: 5
begin nscf_kpoints
  0.0 0.0 0.0 1.0
  0.5 0.0 0.0 3.0
  0.25 0.25 0.0 4.0
end nscf_kpoints
Output_GW: .true.
begin nscf_kgrid
  2 2 2
end nscf_kgrid
begin nscf_kgrid_shift
  0.0 0.0 0.0
end nscf_kgrid_shift
nscf_fermi_level: {fermi} Ry
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


def _cubic_metrics(side: float = 2.0 * math.pi):
    return fortran_lattice_metrics(np.eye(3) * side)


# G-space
# ----------------------------------------------------------------------------


class GSpaceTests(unittest.TestCase):
    def test_sphere_counts_and_order_of_a_simple_cubic_lattice(self) -> None:
        # a = 2 pi: b_i are unit vectors, |G|^2 = g.g; the lattice points of
        # the spheres |g|^2 <= 1, 2, 3, 4 number 7, 19, 27, 33.
        metrics = _cubic_metrics()
        for radius2, count in ((1, 7), (2, 19), (3, 27), (4, 33), (5, 57)):
            sphere = generate_gspace(math.sqrt(radius2) + 1e-12, (0.0, 0.0, 0.0), metrics)
            self.assertEqual(sphere.count, count)
        sphere = generate_gspace(1.0 + 1e-12, (0.0, 0.0, 0.0), metrics)
        # Fortran's loop order: g1 slowest, g3 fastest, each from -kmax.
        np.testing.assert_array_equal(
            sphere.vectors,
            [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
        )
        self.assertEqual(sphere.zero_index, 3)
        # |k + G| <= 0.6 around k = (0.5, 0, 0): G = 0 and G = (-1, 0, 0).
        shifted = generate_gspace(0.6, (0.5, 0.0, 0.0), metrics)
        np.testing.assert_array_equal(shifted.vectors, [[-1, 0, 0], [0, 0, 0]])

    def test_find_bound_and_fft_sizes(self) -> None:
        metrics = _cubic_metrics()
        self.assertEqual(find_bound(2.5, metrics.reciprocal), (2, 2, 2))
        self.assertEqual(fast_fft_size(19), 20)
        self.assertEqual(fast_fft_size(7), 8)
        self.assertEqual(fast_fft_size(31), 32)
        self.assertEqual(fast_fft_size(14), 15)
        # adjustfft: a translation of 1/2 asks for even sizes, and Fortran's
        # rule turns 1/4 into a factor 12 (2*3*4/2), kept as written.
        self.assertEqual(translation_fft_factors([[0.5, 0.0, 0.25]]), (2, 1, 12))
        self.assertEqual(adjust_fft((15, 15, 15), [[0.5, 0.0, 0.25]]), (16, 15, 24))
        self.assertEqual(adjust_fft((19, 19, 19), [[0.0, 0.0, 0.0]]), (20, 20, 20))

    def test_fortran_reference_sizes_of_the_bgw_example(self) -> None:
        # tests/BGW_interface (fcc Si, a = 10.2612 bohr, Grid_Spacing 0.70):
        # Fortran prints 1331 g-vectors in 91 stars (kmax 5), a potential
        # sphere of 2891 vectors on a 20^3 FFT grid, a cutoff of
        # 18.746971514246749 Ry and the k-sphere sizes of its WFN header.
        group = find_space_group(_FCC, _SI_ATOMS)
        self.assertEqual(group.order, 48)
        length = float(np.linalg.norm(_FCC[0]))
        count, step = adjust_step(length, 0.70, translations=group.commensurate_translations()[0])
        self.assertEqual(count, 10)
        metrics = fortran_lattice_metrics(_FCC)
        stars = g_space_stars(
            box_half_widths((count,) * 3),
            metrics.bdot,
            group.reciprocal_rotations,
            group.translation_vectors,
        )
        self.assertEqual((stars.count, stars.star_count, stars.kmax), (1331, 91, (5, 5, 5)))
        cutoff = wavefunction_cutoff([step] * 3)
        self.assertEqual(cutoff, 18.74697151424675)
        potential = potential_gspace(cutoff, metrics, group.translation_vectors)
        self.assertEqual((potential.count, potential.fft_size), (2891, (20, 20, 20)))
        sizes = [k_gspace(cutoff, k, metrics).count for k in ((0.1, 0.1, 0.1), (0.1, 0.3, 0.7), (0.5, 0.5, 0.5))]
        self.assertEqual(sizes, [353, 377, 368])

    def test_stars_of_the_identity_group_are_inverse_pairs(self) -> None:
        metrics = _cubic_metrics(5.0)
        stars = g_space_stars((2, 2, 2), metrics.bdot, identity_space_group().reciprocal_rotations, np.zeros((1, 3)))
        self.assertEqual(stars.count, 125)
        np.testing.assert_array_equal(stars.vectors[0], [0, 0, 0])
        self.assertEqual(stars.star_count, 1 + 62)
        np.testing.assert_array_equal(stars.star_sizes[1:], 2)
        # -G follows G, reached by no operation: conj = -1, same phase.
        np.testing.assert_array_equal(stars.vectors[2::2], -stars.vectors[1::2])
        np.testing.assert_array_equal(stars.conjugation[1::2], 1.0)
        np.testing.assert_array_equal(stars.conjugation[2::2], -1.0)
        np.testing.assert_allclose(stars.phases, 1.0)
        # Sorted by length.
        lengths = np.sum(stars.vectors**2, axis=1)
        self.assertTrue(np.all(np.diff(lengths) >= 0))

    def test_cubic_stars_and_the_star_average(self) -> None:
        side = 6.0
        group = find_space_group(np.eye(3) * side, [Atom("H", np.zeros(3))])
        self.assertEqual(group.order, 48)
        metrics = fortran_lattice_metrics(np.eye(3) * side)
        stars = g_space_stars((3, 3, 3), metrics.bdot, group.reciprocal_rotations, group.translation_vectors)
        self.assertEqual(int(np.sum(stars.star_sizes)), stars.count)
        # Every star holds vectors of one length; (1,0,0) has six members.
        for star in range(stars.star_count):
            members = stars.vectors[stars.star_index == star]
            self.assertEqual(len(set(np.sum(members**2, axis=1))), 1)
        first = stars.star_index[np.flatnonzero(np.all(stars.vectors == [1, 0, 0], axis=1))[0]]
        self.assertEqual(stars.star_sizes[first], 6)
        np.testing.assert_allclose(stars.phases, 1.0)  # symmorphic: every phase is one
        # A field with the cubic symmetry keeps its coefficients; a field without
        # it is replaced by its star average.
        n = 6
        sphere = generate_gspace(3.0 * 2.0 * math.pi / side, (0.0, 0.0, 0.0), metrics)
        x = np.arange(n) / n
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        symmetric = np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y) + np.cos(2 * np.pi * Z)
        transform = B.forward_transform(symmetric)
        values = B.star_averaged_coefficients(transform, stars, sphere)
        lookup = sphere.index_map()
        self.assertAlmostEqual(values[lookup[(1, 0, 0)]], 0.5, places=14)
        self.assertAlmostEqual(values[lookup[(0, 0, -1)]], 0.5, places=14)
        self.assertAlmostEqual(abs(values[lookup[(1, 1, 0)]]), 0.0, places=14)
        lopsided = np.cos(2 * np.pi * X)
        values = B.star_averaged_coefficients(B.forward_transform(lopsided), stars, sphere)
        for vector in ((1, 0, 0), (0, -1, 0), (0, 0, 1)):
            self.assertAlmostEqual(values[lookup[vector]], 1.0 / 6.0, places=14)

    def test_wavefunction_coefficients_of_a_plane_wave_sum(self) -> None:
        # u(r) = 2 exp(i G1.r) - 1j exp(i G2.r) on a 6^3 box: the sphere holds
        # both, the box limit drops what lies beyond kmax, the norm is one.
        metrics = _cubic_metrics(6.0)
        sphere = k_gspace(30.0, (0.1, 0.0, 0.0), metrics)
        n = 6
        x = np.arange(n) / n
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        u = 2.0 * np.exp(2j * np.pi * X) - 1j * np.exp(2j * np.pi * (Y - Z))
        values = B.wavefunction_coefficients(B.forward_transform(u), sphere, (3, 3, 3))
        lookup = sphere.index_map()
        expected = np.array([2.0, -1j]) / math.sqrt(5.0)
        self.assertAlmostEqual(values[lookup[(1, 0, 0)]], expected[0], places=14)
        self.assertAlmostEqual(values[lookup[(0, 1, -1)]], expected[1], places=14)
        self.assertAlmostEqual(float(np.linalg.norm(values)), 1.0, places=14)
        # Aliasing as Fortran: +-n/2 share one FFT index.
        nyquist = np.exp(2j * np.pi * 3 * X)
        values = B.wavefunction_coefficients(B.forward_transform(nyquist), sphere, (3, 3, 3))
        self.assertAlmostEqual(values[lookup[(3, 0, 0)]], values[lookup[(-3, 0, 0)]], places=14)

    def test_fft_box_origin_follows_fortrans_first_point(self) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        translation = parse_parsec_input(_write(Path(folder.name), FCC_CELL))
        grid = prepare_periodic_single_point(translation.problem).grid
        n = tuple(int(v) for v in grid.points_per_axis)
        self.assertTrue(getattr(grid, "is_general_periodic", False))
        self.assertEqual(B.fft_box_origin(grid), tuple(-(v // 2) for v in n))
        values = np.arange(grid.size, dtype=float)
        box = B.fft_box(values, grid)
        # box[0, 0, 0] is the point -n/2 (Python's n/2 modulo the lattice).
        self.assertEqual(box[0, 0, 0], grid.lookup[n[0] // 2, n[1] // 2, n[2] // 2])
        translation = parse_parsec_input(_write(Path(folder.name), PERIODIC))
        orthorhombic = prepare_periodic_single_point(translation.problem).grid
        self.assertEqual(B.fft_box_origin(orthorhombic), (0, 0, 0))

    def test_box_corner_that_breaks_the_symmetry_is_referred_to_the_origin(self) -> None:
        from parsec_python.SCF.pbc import prepare_periodic_single_point

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        # Sides 6 and 7 at h = 0.75 have no common spacing: Fortran's general
        # grid with the shift 1/2 of an orthogonal cell; H2 at the centre.
        text = PERIODIC.replace("6.0 6.0 6.0", "6.0 6.0 7.0")
        text = text.replace("3.1 2.9 2.35", "3.0 3.0 2.8").replace("3.1 2.9 3.75", "3.0 3.0 4.2")
        translation = parse_parsec_input(_write(Path(folder.name), text, MP_SCF))
        grid = prepare_periodic_single_point(translation.problem).grid
        group = translation.gw.space_group
        self.assertTrue(getattr(grid, "is_general_periodic", False))
        np.testing.assert_array_equal(grid.shift, 0.5)
        self.assertEqual(group.order, 16)
        corner = B.box_corner(grid)
        self.assertFalse(B.corner_is_symmetric(corner, group))
        # The fcc example's corner -(a1+a2+a3)/2 keeps every translation.
        fcc = find_space_group(_FCC, _SI_ATOMS)
        self.assertTrue(B.corner_is_symmetric(np.full(3, -0.5), fcc))
        setup = B.berkeleygw_setup(grid, group, np.zeros((1, 3)), cell_volume=252.0)
        np.testing.assert_allclose(setup.corner, corner)
        # A field with the symmetry of the molecule: cos(2 pi (x-3)/6) + ...
        r = grid.coordinates
        field = (
            np.cos(2 * np.pi * (r[:, 0] - 3.0) / 6.0)
            + np.cos(2 * np.pi * (r[:, 1] - 3.0) / 6.0)
            + 2.0 * np.cos(2 * np.pi * (r[:, 2] - 3.5) / 7.0)
        )
        values = B.field_coefficients(setup, grid, [field])[0]
        lookup = setup.potential.index_map()
        for vector, expected in (((1, 0, 0), -0.5), ((0, -1, 0), -0.5), ((0, 0, 1), -1.0), ((1, 1, 0), 0.0)):
            self.assertAlmostEqual(values[lookup[vector]], expected, places=12)
        # PARSEC's convention (no corner phase) would average phases that do not agree.
        transform = B.forward_transform(B.fft_box(field, grid))
        plain = B.star_averaged_coefficients(transform, setup.stars, setup.potential)
        self.assertGreater(abs(abs(plain[lookup[(1, 0, 0)]]) - 0.5), 1e-3)

    def test_crystal_k_points_and_occupied_range(self) -> None:
        reciprocal = 2 * np.pi * np.linalg.inv(_FCC).T
        k = np.array([[0.1, 0.2, 0.3]]) @ reciprocal
        np.testing.assert_allclose(B.crystal_k_points(k, _FCC), [[0.1, 0.2, 0.3]], atol=1e-14)
        occupations = np.array([[[1.0, 1.0, 0.6, 0.4], [0.0, 0.0, 0.0, 0.0]]])
        first, last = B.occupied_band_range(occupations)
        np.testing.assert_array_equal(first, [[1, 0]])
        np.testing.assert_array_equal(last, [[3, 0]])


# Files
# ----------------------------------------------------------------------------


def _header(kind: str, spin_count: int = 2) -> B.BerkeleyGWHeader:
    rng = np.random.default_rng(7)
    group = find_space_group(_FCC, _SI_ATOMS)
    metrics = fortran_lattice_metrics(_FCC)
    nk, nb = 3, 5
    extra = {}
    if kind == "WFN":
        occupations = rng.random((spin_count, nk, nb))
        first, last = B.occupied_band_range(occupations)
        extra = dict(
            band_count=nb,
            max_k_gvectors=9,
            k_points=rng.random((nk, 3)),
            k_weights=np.array([0.25, 0.25, 0.5]),
            k_gvector_counts=np.array([7, 9, 8], dtype=np.int32),
            lowest_occupied=first,
            highest_occupied=last,
            energies=rng.random((spin_count, nk, nb)),
            occupations=occupations,
            kgrid=(4, 4, 4),
            kshift=(0.5, 0.5, 0.5),
        )
    return B.BerkeleyGWHeader(
        kind=kind,
        spin_count=spin_count,
        gvector_count=11,
        cell_symmetry=0,
        ecutrho=75.0,
        # RHO and VXC store no wavefunction cutoff.
        ecutwfc=18.75 if kind == "WFN" else 0.0,
        fft_grid=(20, 20, 20),
        cell_volume=270.1,
        lattice=metrics.lattice,
        adot=metrics.adot,
        reciprocal_volume=(2 * np.pi) ** 3 / 270.1,
        reciprocal=metrics.reciprocal,
        bdot=metrics.bdot,
        rotations=group.reciprocal_rotations.astype(np.int32),
        translations=group.translation_vectors * 2 * np.pi,
        atom_positions=np.array([atom.position for atom in _SI_ATOMS]),
        atom_types=np.array([1, 1], dtype=np.int32),
        date="27-Oct-2026",
        time="19:02:10 -0700",
        **({"kgrid": (0, 0, 0), "kshift": (0.0, 0.0, 0.0)} if kind != "WFN" else {}),
        **extra,
    )


class FileTests(unittest.TestCase):
    def _assert_headers_equal(self, a: B.BerkeleyGWHeader, b: B.BerkeleyGWHeader) -> None:
        for name in a.__dataclass_fields__:
            first, second = getattr(a, name), getattr(b, name)
            if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
                np.testing.assert_array_equal(np.asarray(first), np.asarray(second), err_msg=name)
            else:
                self.assertEqual(first, second, name)

    def test_header_and_data_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        gvectors = rng.integers(-5, 6, size=(11, 3)).astype(np.int32)
        with tempfile.TemporaryDirectory() as folder:
            for kind in ("RHO", "VXC"):
                header = _header(kind)
                data = rng.random((2, 11)) + 1j * rng.random((2, 11))
                path = B.write_berkeleygw_file(Path(folder) / kind, header, gvectors, data=data)
                read = B.read_berkeleygw_file(path)
                self._assert_headers_equal(header, read.header)
                np.testing.assert_array_equal(read.gvectors, gvectors)
                np.testing.assert_array_equal(read.data, data)
                self.assertEqual(read.record_counts, (11,))
            header = _header("WFN")
            spheres = [rng.integers(-3, 4, size=(count, 3)) for count in header.k_gvector_counts]
            bands = [rng.random((5, 2, len(s))) + 1j * rng.random((5, 2, len(s))) for s in spheres]
            path = B.write_berkeleygw_file(
                Path(folder) / "WFN",
                header,
                gvectors,
                k_blocks=((s, iter(b)) for s, b in zip(spheres, bands)),
            )
            read = B.read_berkeleygw_file(path)
            self._assert_headers_equal(header, read.header)
            for k in range(3):
                np.testing.assert_array_equal(read.k_gvectors[k], spheres[k])
                np.testing.assert_array_equal(read.bands[k], bands[k])
            # BerkeleyGW's layout: one G-count of ngk per band record (not ngk*nspin).
            self.assertEqual(read.record_counts, tuple(int(c) for c in header.k_gvector_counts for _ in range(5)))

    def test_record_lengths_are_berkeleygw_4(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            header = _header("WFN", spin_count=1)
            spheres = [np.zeros((count, 3), dtype=np.int32) for count in header.k_gvector_counts]
            path = B.write_berkeleygw_file(
                Path(folder) / "WFN",
                header,
                np.zeros((11, 3), dtype=np.int32),
                k_blocks=((s, [np.zeros((1, len(s)))] * 5) for s in spheres),
            )
            raw = path.read_bytes()
        lengths = []
        position = 0
        while position < len(raw):
            length = int.from_bytes(raw[position : position + 4], "little", signed=True)
            lengths.append(length)
            position += 8 + length
        # title/date/time, scalars, grids, cell, reciprocal cell, 48 3x3
        # matrices, 48 translations, 2 atoms, ngk, kw, kpt, ifmin, ifmax,
        # energies, occupations, then the G-vector records.
        self.assertEqual(
            lengths[:15],
            [96, 48, 48, 160, 160, 48 * 36, 48 * 24, 2 * 28, 12, 24, 72, 12, 12, 120, 120],
        )
        self.assertEqual(lengths[15:18], [4, 4, 11 * 12])

    def test_date_time_and_report_formats(self) -> None:
        import datetime

        moment = datetime.datetime(
            2026, 10, 27, 19, 2, 10, tzinfo=datetime.timezone(datetime.timedelta(hours=-7))
        )
        self.assertEqual(B.berkeleygw_date_time(moment), ("27-Oct-2026", "19:02:10 -0700"))
        # gfortran's list-directed complex: (re,im) right-justified in 53 columns.
        self.assertEqual(
            " rho" + B._list_directed_complex(complex(7.9999999999999947, 0.0)),
            " rho               (7.9999999999999947,0.0000000000000000)",
        )
        self.assertEqual(
            " vxc" + B._list_directed_complex(complex(-0.672795295560767, 0.0)),
            " vxc             (-0.67279529556076700,0.0000000000000000)",
        )


# Input
# ----------------------------------------------------------------------------


class InputTests(unittest.TestCase):
    def test_scf_output_gw_settings(self) -> None:
        translation = _parse(PERIODIC, MP_SCF)
        gw = translation.gw
        self.assertIsNotNone(gw)
        self.assertEqual(gw.kgrid, (2, 2, 2))
        # This port's 2x2x2 grid is (+-1/4): k = (i + 1/2)/2 in BerkeleyGW's convention.
        self.assertEqual(gw.kshift, (0.5, 0.5, 0.5))
        self.assertIsNone(gw.fermi_level)
        self.assertEqual(gw.space_group.order, 16)  # H2 along z in a cube: D4h
        self.assertTrue(any("PARSEC writes 0 0 0" in w for w in translation.warnings))
        # Not in the problem: the SCF is the same as without the keyword.
        with tempfile.TemporaryDirectory() as directory:
            with_gw = parse_parsec_input(_write(Path(directory), PERIODIC, MP_SCF))
            without = parse_parsec_input(
                _write(Path(directory), PERIODIC, MP_SCF.replace("Output_GW: .true.", ""))
            )
            self.assertEqual(repr(with_gw.problem), repr(without.problem))
        ignored = _parse(PERIODIC, MP_SCF + "Ignore_Symmetry: .true.\n")
        self.assertEqual(ignored.gw.space_group.order, 1)

    def test_nscf_keys_are_required_with_output_gw(self) -> None:
        translation = _parse(PERIODIC, NSCF_GW.format(fermi="-0.25"))
        self.assertEqual(translation.gw.kgrid, (2, 2, 2))
        self.assertEqual(translation.gw.kshift, (0.0, 0.0, 0.0))
        self.assertEqual(translation.gw.fermi_level, -0.25)
        base = NSCF_GW.format(fermi="-0.25")
        for missing, code in (
            ("begin nscf_kgrid\n  2 2 2\nend nscf_kgrid\n", "114"),
            ("begin nscf_kgrid_shift\n  0.0 0.0 0.0\nend nscf_kgrid_shift\n", "114"),
            ("nscf_fermi_level: -0.25 Ry\n", "114"),
        ):
            with self.assertRaisesRegex(ParsecInputError, code):
                _parse(PERIODIC, base.replace(missing, ""))
        with self.assertRaisesRegex(ParsecInputError, "117"):
            _parse(PERIODIC, base.replace("0.0 0.0 0.0\nend nscf_kgrid_shift", "0.0 -0.5 0.0\nend nscf_kgrid_shift"))
        with self.assertRaisesRegex(ParsecInputError, "116"):
            _parse(PERIODIC, base.replace("0.0 0.0 0.0\nend nscf_kgrid_shift", "0 0 0\n0 0 0\nend nscf_kgrid_shift"))
        with self.assertRaisesRegex(ParsecInputError, "at least 1"):
            _parse(PERIODIC, base.replace("2 2 2\nend nscf_kgrid", "0 2 2\nend nscf_kgrid"))
        # An ESDF energy unit.
        translation = _parse(PERIODIC, NSCF_GW.format(fermi="-0.25").replace("-0.25 Ry", "-1.0 eV"))
        self.assertAlmostEqual(translation.gw.fermi_level, -1.0 / 13.60563, places=5)
        # Without Output_GW the keys are inert, with a warning.
        quiet = _parse(PERIODIC, base.replace("Output_GW: .true.", ""))
        self.assertIsNone(quiet.gw)
        self.assertTrue(any("only with Output_GW" in w for w in quiet.warnings))

    def test_refusals_by_name(self) -> None:
        from parsec_python.tests.unit.test_nscf_dos import CLUSTER

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parsec.in"
            shutil.copy(DATA / "H_POTRE.DAT", Path(directory) / "H_POTRE.DAT")
            path.write_text(CLUSTER.format(extra="Output_GW: .true.\n"), encoding="utf-8")
            with self.assertRaisesRegex(UnsupportedParsecOptionError, "bulk periodic cell only"):
                parse_parsec_input(path)
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Gamma-point SCF"):
            _parse(PERIODIC, "Output_GW: .true.\n")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "bandstruc"):
            _parse(
                PERIODIC,
                MP_SCF + "begin bandstruc\n1 0 0 0 0.5 0 0 GX\nend bandstruc\n",
            )
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "forces"):
            _parse(PERIODIC, MP_SCF + "Skip_Force: .false.\n")

    def test_accelerated_cli_routes_output_gw_to_the_reference_solver(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_features

        features = _reference_only_features(
            _parse(PERIODIC, MP_SCF), accelerated_backend_selected=False
        )
        self.assertTrue(any("Output_GW" in feature for feature in features))
        plain = _parse(PERIODIC, MP_SCF.replace("Output_GW: .true.", ""))
        self.assertFalse(
            any(
                "Output_GW" in feature
                for feature in _reference_only_features(plain, accelerated_backend_selected=False)
            )
        )


# Runs
# ----------------------------------------------------------------------------


class RunTests(unittest.TestCase):
    def _check_files(self, directory: Path, *, spin_count: int, nk: int, nbands: int, electrons: float):
        rho = B.read_berkeleygw_file(directory / "RHO")
        vxc = B.read_berkeleygw_file(directory / "VXC")
        wfn = B.read_berkeleygw_file(directory / "WFN")
        for item in (rho, vxc, wfn):
            self.assertEqual(item.header.spin_count, spin_count)
            np.testing.assert_array_equal(item.gvectors, rho.gvectors)
        zero = np.flatnonzero(np.all(rho.gvectors == 0, axis=1))[0]
        self.assertAlmostEqual(float(np.sum(rho.data[:, zero]).real), electrons, places=8)
        self.assertEqual(rho.data.shape, (spin_count, rho.header.gvector_count))
        self.assertEqual(vxc.data.shape, (spin_count, rho.header.gvector_count))
        self.assertLess(float(vxc.data[0, zero].real), 0.0)
        # Real fields: c(-G) = conj(c(G)).
        lookup = {tuple(g): i for i, g in enumerate(rho.gvectors)}
        for g, i in list(lookup.items())[:200]:
            j = lookup.get(tuple(-v for v in g))
            if j is not None:
                self.assertAlmostEqual(rho.data[0, i], np.conj(rho.data[0, j]), places=10)
        header = wfn.header
        self.assertEqual((header.k_points.shape[0], header.band_count), (nk, nbands))
        self.assertAlmostEqual(float(np.sum(header.k_weights)), 1.0, places=12)
        for k in range(nk):
            self.assertEqual(wfn.bands[k].shape, (nbands, spin_count, header.k_gvector_counts[k]))
            np.testing.assert_allclose(np.linalg.norm(wfn.bands[k], axis=2), 1.0, atol=1e-12)
            # The bands of one k-point are (nearly) orthogonal on the sphere.
            overlap = wfn.bands[k][:, 0, :].conj() @ wfn.bands[k][:, 0, :].T
            self.assertLess(float(np.max(np.abs(overlap - np.eye(nbands)))), 0.05)
        return rho, vxc, wfn

    def test_scf_with_kpoints_on_the_orthorhombic_grid(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, PERIODIC, MP_SCF)
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            rho, vxc, wfn = self._check_files(directory, spin_count=1, nk=8, nbands=4, electrons=2.0)
            header = wfn.header
            self.assertEqual(header.kgrid, (2, 2, 2))
            np.testing.assert_array_equal(np.abs(header.k_points), 0.25)
            np.testing.assert_array_equal(header.lowest_occupied, 1)
            np.testing.assert_array_equal(header.highest_occupied, 1)
            with np.load(directory / "parsec_python_results.npz") as archive:
                levels = np.asarray(archive["eigenvalues_ry"]).reshape(8, -1)
            np.testing.assert_array_equal(header.energies[0], levels[:, :4])
            text = (directory / "parsec.out").read_text(encoding="utf-8")
            self.assertIn(" Writing files for GW calculation using BerkeleyGW", text)
            self.assertIn(" calling gw_write ", text)
            self.assertIn(" GW output is written", text)
            self.assertEqual(text.count(" HAMILTONIAN MATRIX DIMENSIONS:"), 2)

    def test_spin_polarized_scf_with_kpoints(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            extra = MP_SCF.replace("2 2 2", "1 1 2") + "Spin_Polarization: .true.\n"
            path = _write(directory, PERIODIC, extra)
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            rho, vxc, wfn = self._check_files(directory, spin_count=2, nk=2, nbands=4, electrons=2.0)
            # H2: one electron per channel, band 1 of each channel occupied.
            zero = np.flatnonzero(np.all(rho.gvectors == 0, axis=1))[0]
            np.testing.assert_allclose(rho.data[:, zero].real, [1.0, 1.0], atol=1e-6)
            np.testing.assert_array_equal(wfn.header.highest_occupied, 1)
            self.assertEqual(vxc.data.shape[0], 2)

    def test_spin_polarized_scf_then_nscf(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            kpoints = MP_SCF.replace("2 2 2", "1 1 2").replace("Output_GW: .true.\n", "")
            kpoints += "Spin_Polarization: .true.\n"
            path = _write(directory, PERIODIC, kpoints)
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            scf_log = (directory / "parsec.out").read_text(encoding="utf-8")
            fermi = float(re.findall(r"Fermi level at\s+(-?\d+\.\d+)", scf_log)[-1])
            path.write_text(
                PERIODIC.format(extra=kpoints + NSCF_GW.format(fermi=fermi)), encoding="utf-8"
            )
            code, error = _run_cli(path, "--log", "nscf.out")
            self.assertEqual(code, 0, error)
            rho, vxc, wfn = self._check_files(directory, spin_count=2, nk=3, nbands=5, electrons=2.0)
            zero = np.flatnonzero(np.all(rho.gvectors == 0, axis=1))[0]
            np.testing.assert_allclose(rho.data[:, zero].real, [1.0, 1.0], atol=1e-6)
            np.testing.assert_array_equal(wfn.header.highest_occupied, 1)
            with np.load(directory / "parsec_python_nscf.npz") as archive:
                np.testing.assert_array_equal(wfn.header.energies, archive["nscf_eigenvalues_ry"])

    def test_scf_then_nscf_on_the_general_grid(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = _write(directory, FCC_CELL, MP_SCF.replace("Output_GW: .true.\n", ""))
            code, error = _run_cli(path)
            self.assertEqual(code, 0, error)
            self.assertFalse((directory / "WFN").exists())
            scf_log = (directory / "parsec.out").read_text(encoding="utf-8")
            fermi = float(re.findall(r"Fermi level at\s+(-?\d+\.\d+)", scf_log)[-1])
            scf_kpoints = MP_SCF.replace("Output_GW: .true.\n", "")
            path.write_text(
                FCC_CELL.format(extra=scf_kpoints + NSCF_GW.format(fermi=fermi)), encoding="utf-8"
            )
            code, error = _run_cli(path, "--log", "nscf.out")
            self.assertEqual(code, 0, error)
            rho, vxc, wfn = self._check_files(directory, spin_count=1, nk=3, nbands=5, electrons=1.0)
            header = wfn.header
            np.testing.assert_allclose(header.k_points, [[0, 0, 0], [0.5, 0, 0], [0.25, 0.25, 0]], atol=1e-14)
            np.testing.assert_allclose(header.k_weights, [0.125, 0.375, 0.5])
            self.assertEqual((header.kgrid, header.kshift), ((2, 2, 2), (0.0, 0.0, 0.0)))
            with np.load(directory / "parsec_python_nscf.npz") as archive:
                np.testing.assert_array_equal(header.energies, archive["nscf_eigenvalues_ry"])
                np.testing.assert_array_equal(header.occupations, archive["nscf_occupations"])
            text = (directory / "nscf.out").read_text(encoding="utf-8")
            self.assertIn(f" Fermi level at {fermi:10.4f} [Ry]", text)
            self.assertIn("  Non-self consistent kgrid dimensions", text)
            self.assertIn("   2 X  2 X  2", text)


if __name__ == "__main__":
    unittest.main()
