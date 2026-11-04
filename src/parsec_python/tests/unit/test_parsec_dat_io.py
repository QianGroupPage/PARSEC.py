"""parsec.dat, eigen.dat and cube files in the layouts of PARSEC.

The byte layout of ``parsec.dat`` is checked against ``scipy.io.FortranFile``
(gfortran sequential records); the text layouts against lines that the
PARSEC executable wrote for an H2O cluster (``eigensave.f90`` and
``export_grid_data.f90``, Ignore_Symmetry, 0.57 bohr, 10.4 bohr sphere).
"""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import FortranFile

from parsec_python.Grid import build_cluster_grid, build_periodic_grid
from parsec_python.models import (
    GridSettings,
    OutputFileSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SpinChannelResult,
    WavefunctionSelection,
)
from parsec_python.Output import parsec_dat as dat
from parsec_python.Output.cube import cube_text, dense_box, fortran_exponential
from parsec_python.Output.eigen_dat import (
    EigenChannel,
    eigen_dat_block,
    eigen_dat_header,
    eigen_dat_line,
)
from parsec_python.Output.output_files import RunOutputFiles, output_header_lines
from parsec_python.SCF.restart import (
    RestartError,
    restart_fields,
    restart_warm_start,
    with_first_warm_start,
)


def _cluster_grid(spacing=0.8, radius=3.0, shift=(0.5, 0.5, 0.5)):
    return build_cluster_grid(GridSettings(spacing=spacing, radius=radius, expansion_order=4, shift=shift))


class _Result:
    """The fields of a scalar SinglePointResult that parsec.dat reads."""

    def __init__(self, grid, states=4, occupied=2, rng=None, k_points=None):
        rng = np.random.default_rng(1) if rng is None else rng
        self.eigenvalues = np.sort(rng.normal(size=states))
        self.occupations = np.r_[np.ones(occupied), np.zeros(states - occupied)]
        vectors = rng.normal(size=(grid.size, states))
        self.wavefunctions = vectors / np.linalg.norm(vectors, axis=0)
        self.next_effective_potential = rng.normal(size=grid.size)
        self.density = np.abs(rng.normal(size=grid.size))
        self.k_points = k_points
        self.k_weights = None if k_points is None else np.full(len(k_points), 1.0 / len(k_points))
        if k_points is not None:
            count = len(k_points)
            self.eigenvalues = np.tile(self.eigenvalues, count)
            self.occupations = np.tile(self.occupations, count)


class RecordLayoutTests(unittest.TestCase):
    def test_cluster_file_has_the_records_of_wfnsave(self) -> None:
        grid = _cluster_grid()
        result = _Result(grid)
        data, notes = dat.parsec_dat_from_result(
            result, grid, WavefunctionSelection(), date_label="27-OCT-2026 14:01:07 -0700"
        )
        self.assertEqual(notes, ())
        with tempfile.TemporaryDirectory() as directory:
            path = dat.write_parsec_dat(Path(directory) / "parsec.dat", data)
            with FortranFile(path, "r", header_dtype=np.uint32) as stream:
                self.assertEqual(stream.read_record("S26")[0], b"27-OCT-2026 14:01:07 -0700")
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                np.testing.assert_allclose(stream.read_reals(np.float64), [0.8, 3.0])
                self.assertEqual(stream.read_ints(np.int32)[0], grid.size)
                np.testing.assert_allclose(stream.read_reals(np.float64), [0.5, 0.5, 0.5])
                np.testing.assert_array_equal(stream.read_ints(np.int32), [grid.size, 1])
                np.testing.assert_array_equal(stream.read_ints(np.int32), np.eye(3).ravel())
                np.testing.assert_allclose(stream.read_reals(np.float64), np.eye(3).ravel())
                np.testing.assert_allclose(stream.read_reals(np.float64), [0, 0, 0])
                lattice = stream.read_reals(np.float64)
                np.testing.assert_allclose(lattice[:9], (12.0 * np.eye(3)).ravel())
                np.testing.assert_allclose(lattice[9:], (np.eye(3) / 12.0).ravel())
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1])
                points = stream.read_ints(np.int32).reshape(-1, 3)
                np.testing.assert_array_equal(points, grid.integer_coordinates)
                self.assertEqual(stream.read_ints(np.int32)[0], 4)
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1, 1, 1, 1])
                np.testing.assert_allclose(stream.read_reals(np.float64), result.eigenvalues)
                np.testing.assert_allclose(stream.read_reals(np.float64), result.occupations)
                np.testing.assert_allclose(stream.read_reals(np.float64), result.next_effective_potential)
                np.testing.assert_allclose(stream.read_reals(np.float64), result.density)
                for _ in range(7):
                    stream.read_record(np.uint8)
                self.assertEqual(stream.read_ints(np.int32)[0], 2)
                np.testing.assert_array_equal(stream.read_ints(np.int32), [1, 2])
                np.testing.assert_allclose(stream.read_reals(np.float64), result.wavefunctions[:, 0])
                np.testing.assert_allclose(stream.read_reals(np.float64), result.wavefunctions[:, 1])
                with self.assertRaises(Exception):
                    stream.read_record(np.uint8)

    def test_round_trip_of_spin_channels_and_k_points(self) -> None:
        grid = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.8])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        rng = np.random.default_rng(3)
        channels = tuple(
            SpinChannelResult(
                eigenvalues=np.sort(rng.normal(size=3)),
                occupations=np.array([1.0, 0.5, 0.0]),
                wavefunctions=np.linalg.qr(rng.normal(size=(grid.size, 3)))[0],
                representations=np.ones(3, dtype=np.int32),
                density=np.abs(rng.normal(size=grid.size)),
                xc_potential=np.zeros(grid.size),
                input_effective_potential=np.zeros(grid.size),
                output_effective_potential=np.zeros(grid.size),
                next_effective_potential=rng.normal(size=grid.size),
                electron_count=1.5,
            )
            for _ in range(2)
        )

        class Spin:
            spin = channels
            k_points = None

        data, _ = dat.parsec_dat_from_result(Spin(), grid, WavefunctionSelection(mode="all"))
        with tempfile.TemporaryDirectory() as directory:
            path = dat.write_parsec_dat(Path(directory) / "parsec.dat", data)
            back = dat.read_parsec_dat(path)
        self.assertEqual(back.spin_count, 2)
        self.assertEqual(back.boundary_code, 1)
        np.testing.assert_allclose(back.periodic.lattice_vectors, np.diag([4.0, 4.0, 4.8]))
        np.testing.assert_allclose(back.step, [0.8, 0.8, 0.8])
        self.assertEqual(back.rmax, 1.0)
        for spin in range(2):
            np.testing.assert_array_equal(back.potentials[spin], channels[spin].next_effective_potential)
            np.testing.assert_array_equal(back.densities[spin], channels[spin].density)
            np.testing.assert_array_equal(back.wavefunctions[spin][0].state_numbers, [1, 2, 3])
            np.testing.assert_allclose(back.wavefunctions[spin][0].vectors, channels[spin].wavefunctions)
        # PARSEC's periodic integers run from -n/2 to n-n/2-1 with the same
        # positions as the Python integers 0..n-1, modulo the cell.
        counts = grid.points_per_axis
        self.assertTrue(np.all(back.points >= -(counts // 2)))
        self.assertTrue(np.all(back.points < counts - counts // 2))
        np.testing.assert_array_equal(back.points % counts, grid.integer_coordinates)

        k_points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _Result(grid, k_points=k_points)
        data, notes = dat.parsec_dat_from_result(
            result, grid, WavefunctionSelection(), monkhorst_pack_grid=(2, 1, 1)
        )
        self.assertTrue(any("k-point drivers keep no orbitals" in note for note in notes))
        with tempfile.TemporaryDirectory() as directory:
            back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
        self.assertEqual(back.wavefunction_kind, 1)
        self.assertEqual(back.periodic.kpoint_method, dat.KPOINT_MONKHORST_PACK)
        np.testing.assert_allclose(back.periodic.k_points, k_points)
        np.testing.assert_allclose(back.periodic.k_weights, [0.5, 0.5])
        self.assertEqual(len(back.levels[0]), 2)
        self.assertEqual([block.state_numbers.size for block in back.wavefunctions[0]], [0, 0])

    def test_long_records_are_split_into_gfortran_subrecords(self) -> None:
        values = np.arange(40, dtype="<f8")
        stream = io.BytesIO()
        with patch.object(dat, "_MAXIMUM_SUBRECORD", 96):
            dat.write_record(stream, values)
        raw = stream.getvalue()
        markers = []
        offset = 0
        while offset < len(raw):
            head = int(np.frombuffer(raw[offset : offset + 4], "<i4")[0])
            size = abs(head)
            tail = int(np.frombuffer(raw[offset + 4 + size : offset + 8 + size], "<i4")[0])
            markers.append((head, tail))
            offset += size + 8
        self.assertEqual(markers, [(-96, 96), (-96, -96), (-96, -96), (32, -32)])
        stream.seek(0)
        np.testing.assert_array_equal(np.frombuffer(dat.read_record(stream), "<f8"), values)

    def test_occupied_states_follow_ifmax_of_the_last_k_point(self) -> None:
        self.assertEqual(dat.highest_occupied([np.array([1.0, 1e-14, 0.0])]), 2)
        self.assertEqual(dat.highest_occupied([np.array([1, 1, 1.0]), np.array([1.0, 0, 0])]), 1)
        self.assertEqual(dat.highest_occupied([np.zeros(3)]), 0)
        with self.assertRaisesRegex(ValueError, "Save_Wfn_Bands asks for state 5"):
            dat.selected_states(WavefunctionSelection(mode="list", indices=(5,)), [np.ones(4)], 4)


def _symmetric_field(positions):
    x, y, z = positions.T
    return np.exp(-((x - 0.0) ** 2) - 2.0 * (y - 0.3) ** 2 - 0.5 * z**2) + 0.1 * x**2 * z**2


def _fortran_wedge(grid, symmetry):
    """The irreducible wedge as grid_partition.f90 builds it."""

    steps = dat.grid_steps(grid)
    shift = np.asarray(grid.settings.shift)
    wedge = {}
    order = []
    for point in grid.integer_coordinates:
        position = (point + shift) * steps
        found = None
        for operation in range(symmetry.ntrans):
            image = symmetry.apply(operation, position[None, :])[0]
            integers = tuple(np.rint(image / steps - shift).astype(int))
            if integers in wedge:
                found = integers
                break
        if found is None:
            wedge[tuple(point)] = len(order)
            order.append(tuple(point))
    return np.asarray(order, dtype=np.int64)


class GridMappingTests(unittest.TestCase):
    def test_a_symmetry_reduced_file_unfolds_onto_the_full_grid(self) -> None:
        grid = _cluster_grid(spacing=0.6, radius=3.0)
        alatt = dat.cluster_alatt(3.0)
        mirrors = [np.diag(d) for d in ((1, 1, 1), (-1, 1, -1), (-1, 1, 1), (1, 1, -1))]
        symmetry = dat.ParsecDatSymmetry(
            rmtrx=np.asarray(mirrors, dtype=np.int32),
            trans=np.asarray(mirrors, dtype=float),
            tnp=np.zeros((4, 3)),
            alatt=alatt,
            invlat=np.linalg.inv(alatt.T),
            chi=np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]]),
        )
        wedge = _fortran_wedge(grid, symmetry)
        self.assertLess(len(wedge), grid.size / 3)
        positions = (wedge + 0.5) * 0.6
        values = _symmetric_field(positions)
        data = dat.ParsecDat(
            date_label=dat.fortran_date_label(),
            system_type=dat.system_type(1, 0, 0),
            step=np.full(3, 0.6),
            rmax=3.0,
            ndim=grid.size,
            shift=np.full(3, 0.5),
            symmetry=symmetry,
            points=wedge,
            levels=((dat.ParsecDatLevels(np.ones(1, np.int32), np.zeros(1), np.ones(1)),),),
            potentials=(values,),
            densities=(2.0 * values,),
        )
        with tempfile.TemporaryDirectory() as directory:
            back = dat.read_parsec_dat(dat.write_parsec_dat(Path(directory) / "parsec.dat", data))
        np.testing.assert_array_equal(back.symmetry.chi, symmetry.chi)
        mapping = dat.grid_rows_of_file(back, grid)
        self.assertEqual(mapping.missing, 0)
        np.testing.assert_allclose(mapping.field(back.potentials[0]), _symmetric_field(grid.coordinates), atol=1e-12)

    def test_restart_checks_and_a_smaller_stored_sphere(self) -> None:
        small = _cluster_grid(radius=2.4)
        large = _cluster_grid(radius=3.0)
        result = _Result(small)
        data, _ = dat.parsec_dat_from_result(result, small, WavefunctionSelection())
        fields = restart_fields(data, small, spin_count=1)
        self.assertFalse(fields.extrapolate)
        np.testing.assert_array_equal(fields.potentials[0], result.next_effective_potential)
        grown = restart_fields(data, large, spin_count=1)
        self.assertTrue(grown.extrapolate)
        self.assertEqual(grown.missing_points, large.size - small.size)
        rows = large.rows_for_integer_coordinates(small.integer_coordinates)
        np.testing.assert_array_equal(grown.densities[0][rows], result.density)
        self.assertEqual(float(np.sum(np.delete(grown.densities[0], rows))), 0.0)
        shrunk = restart_fields(
            dat.parsec_dat_from_result(_Result(large), large, WavefunctionSelection())[0],
            small,
            spin_count=1,
        )
        self.assertEqual(shrunk.dropped_points, large.size - small.size)
        for kwargs, code in (
            ({"spin_count": 2}, 361),
            ({"spin_count": 1, "complex_wavefunctions": True}, 362),
        ):
            with self.assertRaises(RestartError) as caught:
                restart_fields(data, small, **kwargs)
            self.assertEqual(caught.exception.code, code)
        with self.assertRaises(RestartError) as caught:
            restart_fields(data, _cluster_grid(spacing=0.7, radius=2.4), spin_count=1)
        self.assertEqual(caught.exception.code, 366)
        with self.assertRaisesRegex(RestartError, "Ignore_Symmetry"):
            restart_fields(data, _cluster_grid(radius=2.4, shift=(0.0, 0.0, 0.0)), spin_count=1)
        periodic = build_periodic_grid(
            PeriodicCell(np.diag([4.0, 4.0, 4.0])), PeriodicGridSettings(spacing=0.8, expansion_order=4)
        )
        with self.assertRaises(RestartError) as caught:
            restart_fields(data, periodic, spin_count=1)
        self.assertEqual(caught.exception.code, 363)

    def test_warm_start_takes_the_stored_potential_or_rebuilds_it(self) -> None:
        grid = _cluster_grid()
        result = _Result(grid)
        data, _ = dat.parsec_dat_from_result(result, grid, WavefunctionSelection())
        fields = restart_fields(data, grid, spin_count=1)

        class System:
            ionic_potential = -np.ones(grid.size)

        start = restart_warm_start(System(), fields)
        np.testing.assert_array_equal(start.input_potential, result.next_effective_potential)
        np.testing.assert_array_equal(start.hartree_potential, np.ones(grid.size))
        self.assertIsNone(start.eigensolver_state)
        self.assertIsNone(start.density_guess)
        rebuilt = restart_warm_start(System(), type(fields)(**{**fields.__dict__, "extrapolate": True}))
        np.testing.assert_array_equal(rebuilt.input_potential, System.ionic_potential)
        np.testing.assert_array_equal(rebuilt.density_guess, result.density)
        np.testing.assert_array_equal(rebuilt.hxc_reference, np.zeros(grid.size))

    def test_only_the_first_loop_of_an_ionic_run_takes_the_restart(self) -> None:
        from parsec_python.Relaxation.backend import IonicBackend

        seen = []
        backend = IonicBackend(
            run_scf=lambda system, callback=None, warm_start=None: seen.append(warm_start),
            compute_forces=None,
            update_geometry=None,
            warm_start=None,
        )
        restarted = with_first_warm_start(backend, "restart")
        restarted.run_scf("system", callback=None, warm_start=None)
        restarted.run_scf("system", callback=None, warm_start=None)
        restarted.run_scf("system", callback=None, warm_start="moved")
        self.assertEqual(seen, ["restart", None, "moved"])


class EigenDatTests(unittest.TestCase):
    # Lines of eigen.dat that PARSEC wrote for H2O (Ignore_Symmetry).
    FORTRAN = [
        "  State     Eigenvalue [Ry]    Eigenvalue [eV]    Occupation   Repr.   Movement   spin",
        "    1       -2.2076030485      -30.0362        1.0000   1      0    ",
        "    5       -0.5433746657       -7.3930        0.0000   1      0    ",
    ]

    def test_header_and_rows_are_those_of_eigensave(self) -> None:
        self.assertEqual(eigen_dat_header(False), self.FORTRAN[0])
        self.assertEqual(
            eigen_dat_header(True),
            "  State     Eigenvalue [Ry]    Eigenvalue [eV]    Occupation   Repr.   k-point  Movement   spin",
        )
        self.assertEqual(eigen_dat_line(1, -2.2076030485, 1.0, 1, 0), self.FORTRAN[1])
        self.assertEqual(eigen_dat_line(5, -0.5433746657, 8.79e-13, 1, 0), self.FORTRAN[2])
        self.assertEqual(
            eigen_dat_line(3, -0.25, 0.5, 2, 4, "dn", kpoint=7),
            "    3       -0.2500000000       -3.4015        0.5000   2      7      4  dn",
        )

    def test_blocks_hold_the_occupied_or_all_states_of_each_channel(self) -> None:
        up = EigenChannel(np.array([-1.0, -0.5, 0.2]), np.array([1.0, 1.0, 0.0]), np.ones(3))
        down = EigenChannel(np.array([-0.9, -0.1, 0.3]), np.array([1.0, 0.0, 0.0]), np.ones(3))
        lines = eigen_dat_block((up, down), move=2, all_states=False)
        self.assertEqual([line[-2:] for line in lines], ["up", "up", "dn"])
        self.assertTrue(all(line[-9:-4] == "    2" for line in lines))
        self.assertEqual(len(eigen_dat_block((up,), move=0, all_states=True)), 3)
        k_lines = eigen_dat_block(
            (EigenChannel(np.r_[up.eigenvalues, up.eigenvalues], np.r_[up.occupations, up.occupations], ()),),
            move=0,
            all_states=False,
            kpoint_count=2,
        )
        self.assertEqual([int(line.split()[5]) for line in k_lines], [1, 1, 2, 2])


class CubeTests(unittest.TestCase):
    def test_fortran_exponential_format(self) -> None:
        self.assertEqual(fortran_exponential(0.0), "  0.00000E+00")
        self.assertEqual(fortran_exponential(6.552e-08), "  0.65520E-07")
        self.assertEqual(fortran_exponential(-1.52234), " -0.15223E+01")
        self.assertEqual(fortran_exponential(2.528e-319), "  0.25280-318")
        self.assertEqual(fortran_exponential(0.999996), "  0.10000E+01")

    def test_cube_header_and_box_of_a_cluster(self) -> None:
        grid = build_cluster_grid(GridSettings(spacing=0.57, radius=10.4, expansion_order=8))
        dense, origin, voxels = dense_box(np.ones(grid.size), grid)
        self.assertEqual(dense.shape, (38, 38, 38))
        lines = cube_text("charge_den", dense, origin, voxels).splitlines()
        # The header PARSEC wrote for this grid.
        self.assertEqual(
            lines[:7],
            [
                " Cubefile created from PARSEC calculation",
                " Volumetric charge_den data",
                "    1  -10.545000  -10.545000  -10.545000",
                "   38    0.570000    0.000000    0.000000",
                "   38    0.000000    0.570000    0.000000",
                "   38    0.000000    0.000000    0.570000",
                "    1    1.000000    0.000000    0.000000    0.000000",
            ],
        )
        self.assertEqual(len(lines), 7 + 38 * 38 * 7)
        self.assertEqual(lines[7], "  0.00000E+00" * 6)
        self.assertEqual(float(np.sum(dense)), float(grid.size))


class OutputFileTests(unittest.TestCase):
    def test_default_wavefunctions_give_way_above_the_limit(self) -> None:
        files = RunOutputFiles(OutputFileSettings(), ".", write=lambda line: None)
        self.assertTrue(files.plan_wavefunctions(1000, 10, 1))
        self.assertTrue(files.selection_has_states)
        small = RunOutputFiles(
            OutputFileSettings(), ".", write=lambda line: None, wavefunction_limit_bytes=1000
        )
        self.assertFalse(small.plan_wavefunctions(1000, 10, 1))
        self.assertEqual(small.selection.indices, ())
        explicit = RunOutputFiles(
            OutputFileSettings(wavefunctions=WavefunctionSelection(mode="all", explicit=True)),
            ".",
            write=lambda line: None,
            wavefunction_limit_bytes=1000,
        )
        self.assertTrue(explicit.plan_wavefunctions(1000, 10, 1))
        off = RunOutputFiles(OutputFileSettings(output_data=False), ".", write=lambda line: None)
        self.assertFalse(off.plan_wavefunctions(1000, 10, 1))
        self.assertEqual(off.paths(), [])

    def test_header_lines_only_for_keywords_of_the_input(self) -> None:
        self.assertEqual(output_header_lines(OutputFileSettings()), [])
        settings = OutputFileSettings(
            wavefunctions=WavefunctionSelection(mode="list", indices=tuple(range(1, 12)), explicit=True),
            eigenvalues=True,
            all_states=False,
            keywords=frozenset({"output_eigenvalues", "save_wfn_bands"}),
        )
        lines = output_header_lines(settings)
        self.assertEqual(lines[0], " Writing           11 selected wave functions to parsec.dat file:")
        self.assertEqual(lines[1], "".join(f"{n:6d}" for n in range(1, 11)))
        self.assertEqual(lines[2], "    11")
        self.assertIn(" outevflag =            1", lines)


if __name__ == "__main__":
    unittest.main()
