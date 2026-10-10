from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import gc
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
from parsec_python.acceleration.Eigensolvers import distributed_filter, filter_graph
from parsec_python.acceleration.Eigensolvers.distributed_filter import DistributedFilter, replica_stencil
from parsec_python.acceleration.Eigensolvers.distributed_state import SectorDeviceGroup
from parsec_python.acceleration.backends import (
    cupy as cupy_backend, cupy_compact, cupy_compile, cupy_projectors, cupy_stencil_major, implicit_stencil)
from parsec_python.acceleration.backends.cupy import CuPyHamiltonian, cupy_available, require_cupy
from parsec_python.acceleration.backends.cupy_compact import CuPyCompactFiniteDifference
from parsec_python.acceleration.backends.cupy_stencil_major import (
    _CUDA_SOURCE, CuPyStencilMajorFiniteDifference, build_stencil_major_metadata)
from parsec_python.acceleration.backends.implicit_stencil import (
    PackedTileHostMetadata, _pack_affine_tiles_reference, implicit_source, implicit_tile_for_group,
    implicit_tile_for_sector, implicit_tile_setting, pack_affine_tiles, pack_worker_setting, unpack_affine_tiles)
from parsec_python.acceleration.tests.test_mixed_precision import _HostRawKernel, host_cupy


def _irregular_operator(n=1031):
    """Rows that end at a boundary, an empty row, a missing diagonal and an odd coefficient that FP32 cannot hold."""
    a = sp.diags((-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)), (-1, 0, 1), format="lil")
    a[95, 94] = .1
    a[521, :] = 0
    a[775, 775] = 0
    return a.tocsr()


def _settings(**values):
    """The environment without the tile settings and those that choose another stencil, then ``values``."""
    names = ("PARSEC_CUPY_IMPLICIT_TILE", "PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS", "PARSEC_CUPY_IMPLICIT_PACK_WORKERS",
             "PARSEC_CUPY_STENCIL_MAJOR", "PARSEC_CUPY_MIXED_FILTER", "PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS")
    kept = {name: value for name, value in os.environ.items() if name not in names}
    return patch.dict(os.environ, {**kept, **values}, clear=True)


class ImplicitPackingTests(unittest.TestCase):
    def test_exact_operator_for_boundary_missing_rows_and_irregular_coefficients(self):
        n = 1031
        m = build_stencil_major_metadata(_irregular_operator(n))
        for tile in (16, 32, 64, 128, 256):
            with self.subTest(tile=tile):
                packed, codes, stats = pack_affine_tiles(m, tile)
                nn, cc = unpack_affine_tiles(packed, codes, n, m.neighbors.shape[0], tile)
                np.testing.assert_array_equal(nn, m.neighbors)
                valid = nn >= 0
                np.testing.assert_array_equal(cc[valid], m.coefficient_codes[valid])
                self.assertGreater(stats['regular_rows'], 0)

    def test_partial_tile_and_invalid_tile(self):
        m = build_stencil_major_metadata(sp.eye(7, format="csr"))
        nn, cc, stats = pack_affine_tiles(m, 32)
        n, c = unpack_affine_tiles(nn, cc, 7, 1, 32)
        np.testing.assert_array_equal(n, m.neighbors)
        self.assertEqual(stats['regular_rows'], 0)
        with self.assertRaises(ValueError):
            pack_affine_tiles(m, 0)

    def test_chunks_packed_side_by_side_give_the_arrays_of_the_serial_and_reference_packers(self):
        # 2,500 tiles of 16 rows are three chunks of the packer; every chunk has irregular tiles.
        n = 40_000
        a = _irregular_operator(n).tolil()
        a[20_000, 19_999] = .25
        a[39_990, :] = 0
        m = build_stencil_major_metadata(a.tocsr())
        reference = _pack_affine_tiles_reference(m, 16)
        pools = []

        class Pool(ThreadPoolExecutor):
            def __init__(self, *arguments, **options):
                pools.append(options["max_workers"])
                super().__init__(*arguments, **options)

        def pack(tile=16, metadata=m, **settings):
            with patch.object(implicit_stencil, "ThreadPoolExecutor", Pool), _settings(**settings):
                return pack_affine_tiles(metadata, tile)

        def assert_same(packed, expected):
            np.testing.assert_array_equal(packed[0], expected[0])
            np.testing.assert_array_equal(packed[1], expected[1])
            self.assertEqual(packed[2], expected[2])
            self.assertEqual((packed[0].dtype, packed[1].dtype), (np.int32, np.uint8))

        # No more threads than chunks by default, the named number otherwise, none for 1.
        assert_same(pack(), reference)
        assert_same(pack(PARSEC_CUPY_IMPLICIT_PACK_WORKERS="2"), reference)
        assert_same(pack(PARSEC_CUPY_IMPLICIT_PACK_WORKERS="16"), reference)
        self.assertEqual(pools, [3, 2, 3])
        serial = pack(PARSEC_CUPY_IMPLICIT_PACK_WORKERS="1")
        assert_same(serial, reference)
        self.assertEqual(len(pools), 3)
        # One chunk needs no pool; other tile sizes follow the serial packer.
        pack(metadata=build_stencil_major_metadata(sp.eye(7, format="csr")))
        self.assertEqual(len(pools), 3)
        for tile in (32, 256):
            assert_same(pack(tile), pack(tile, PARSEC_CUPY_IMPLICIT_PACK_WORKERS="1"))
        for value in ("0", "-1", "many"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_PACK_WORKERS"):
                pack(PARSEC_CUPY_IMPLICIT_PACK_WORKERS=value)
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_PACK_WORKERS=value), \
                 self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_PACK_WORKERS"):
                pack_worker_setting()
        for value, workers in ((None, 4), ("1", 1), (" 16 ", 16)):
            with _settings(**({} if value is None else {"PARSEC_CUPY_IMPLICIT_PACK_WORKERS": value})):
                self.assertEqual(pack_worker_setting(), workers)

    def test_a_failing_chunk_raises_in_the_caller(self):
        m = build_stencil_major_metadata(_irregular_operator(40_000))

        class Failing:
            neighbors = m.neighbors
            @property
            def coefficient_codes(self):
                return np.zeros((1, 1), np.uint8)

        with _settings(), self.assertRaises(ValueError):
            pack_affine_tiles(Failing(), 16)


class ImplicitTilePolicyTests(unittest.TestCase):
    def test_auto_packs_a_sector_on_one_device_from_the_size_that_gained(self):
        with _settings():
            self.assertIsNone(implicit_tile_setting())
            # Sector sizes of the measured cases C795H300 to C5659H1132, which all gained; nothing smaller was run.
            for rows, tile in ((546_152, 0), (649_999, 0), (650_000, 16), (677_542, 16), (1_031_322, 16),
                               (1_194_720, 16), (1_924_792, 16), (2_604_846, 16), (2_873_400, 16), (3_786_832, 16)):
                with self.subTest(rows=rows):
                    self.assertEqual(implicit_tile_for_sector(rows, 1), tile)
            # A sector that is to hold the FP32 recurrence reads the slot-major stencil.
            self.assertEqual(implicit_tile_for_sector(3_786_832, 1, float32_filter=True), 0)
        for value in ("auto", "", " AUTO "):
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_TILE=value):
                self.assertIsNone(implicit_tile_setting())
                self.assertEqual(implicit_tile_for_sector(677_542, 1), 16)
                self.assertEqual(implicit_tile_for_sector(546_152, 1), 0)

    def test_a_device_that_filters_several_sectors_packs_smaller_ones(self):
        with _settings():
            # C795H300 and C1213H412 gained with four, two and one sector on a device (1, 2 and 4 GPUs); the
            # smaller sectors were not run, and pack where a device filters enough of them: C459H204 (577,426
            # rows) with four and with two, C185H124 (280,790) nowhere.
            for rows, tiles in ((274_784, (0, 0, 0)), (280_790, (0, 0, 0)), (407_478, (16, 0, 0)),
                                (546_152, (16, 16, 0)), (577_426, (16, 16, 0)), (677_542, (16, 16, 16)),
                                (1_031_322, (16, 16, 16))):
                for sectors, tile in zip((4, 2, 1), tiles):
                    with self.subTest(rows=rows, sectors=sectors):
                        self.assertEqual(implicit_tile_for_sector(rows, 1, sectors_on_device=sectors), tile)
            # The limit over the square root of the sectors, rounded up.  Beyond four sectors it falls no further:
            # a process that solves eight packs eight stencils, and no more than four on a device were run.
            for sectors, limit in ((1, 650_000), (2, 459_620), (3, 375_278), (4, 325_000), (5, 325_000),
                                   (8, 325_000), (16, 325_000)):
                with self.subTest(sectors=sectors):
                    self.assertEqual(implicit_tile_for_sector(limit, 1, False, sectors), 16)
                    self.assertEqual(implicit_tile_for_sector(limit - 1, 1, False, sectors), 0)
            # The two sizes gained with one device for the basis of a sector.  With a basis that the devices of
            # a group share (8 and 16 GPUs) they were run slot-major only: the lower limit is not theirs, and
            # the sectors of a device do not make up for it.
            for devices in (2, 4):
                for rows in (677_542, 1_031_322):
                    with self.subTest(devices=devices, rows=rows):
                        self.assertEqual(implicit_tile_for_sector(rows, devices), 0)
                        self.assertEqual(implicit_tile_for_sector(rows, devices, sectors_on_device=4), 0)
            # No sector count below one, and none makes up for a recurrence that reads no tiles.
            self.assertEqual(implicit_tile_for_sector(649_999, 1, sectors_on_device=0), 0)
            self.assertEqual(implicit_tile_for_sector(650_000, 1, sectors_on_device=0), 16)
            self.assertEqual(implicit_tile_for_sector(3_786_832, 1, True, sectors_on_device=4), 0)
        with _settings(PARSEC_CUPY_IMPLICIT_TILE="0"):
            self.assertEqual(implicit_tile_for_sector(3_786_832, 1, sectors_on_device=4), 0)
        with _settings(PARSEC_CUPY_IMPLICIT_TILE="32"):
            self.assertEqual(implicit_tile_for_sector(8, 1, sectors_on_device=4), 32)

    def test_threshold_setting(self):
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="600000"):
            self.assertEqual(implicit_tile_for_sector(677_542, 1), 16)
            self.assertEqual(implicit_tile_for_sector(599_999, 1), 0)
            self.assertEqual(implicit_tile_for_sector(677_542, 2), 0)
            # The setting is the limit of a sector alone on its device and scales the others.
            self.assertEqual(implicit_tile_for_sector(300_000, 1, sectors_on_device=4), 16)
            self.assertEqual(implicit_tile_for_sector(299_999, 1, sectors_on_device=4), 0)
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="1"):
            self.assertEqual(implicit_tile_for_sector(1, 1, sectors_on_device=4), 16)
        # The settings that give four, two and one sector on a device the former limit of 1,100,000 rows.
        for value, sectors in (("2200000", 4), ("1555634", 2), ("1100000", 1)):
            with self.subTest(sectors=sectors), _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS=value):
                for rows, tile in ((1_031_322, 0), (1_099_999, 0), (1_100_000, 16), (1_194_720, 16)):
                    self.assertEqual(implicit_tile_for_sector(rows, 1, sectors_on_device=sectors), tile)
        # Each holds for its own number of sectors only: devices with unequal numbers have no common setting.
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="1555634"):
            self.assertEqual(implicit_tile_for_sector(1_194_720, 1, sectors_on_device=1), 0)
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="1100000"):
            self.assertEqual(implicit_tile_for_sector(1_031_322, 1, sectors_on_device=2), 16)
        for value in ("0", "-5", "large"):
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS=value), \
                 self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS"):
                implicit_tile_for_sector(2_000_000, 1)

    def test_explicit_value_wins(self):
        for value in (0, 16, 32, 64, 128, 256):
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_TILE=str(value)):
                self.assertEqual(implicit_tile_setting(), value)
                for rows, devices, float32 in ((8, 1, False), (3_786_832, 1, False), (3_786_832, 4, False),
                                               (3_786_832, 1, True)):
                    self.assertEqual(implicit_tile_for_sector(rows, devices, float32), value)
        for value in ("1", "8", "-16", "17", "tiles"):
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_TILE=value):
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_TILE"):
                    implicit_tile_setting()
                with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_TILE"):
                    implicit_tile_for_sector(3_786_832, 1)

    def test_auto_packs_a_sector_that_several_devices_filter_from_a_row_count_of_its_own(self):
        with _settings():
            # Sector sizes of C795H300 to C7067H1308; no group has run with tiles, the row count is assumed.
            for devices in (2, 4):
                for rows, tile in ((677_542, 0), (1_031_322, 0), (1_099_999, 0), (1_100_000, 16), (1_194_720, 16),
                                   (3_786_832, 16), (4_130_510, 16)):
                    with self.subTest(devices=devices, rows=rows):
                        self.assertEqual(implicit_tile_for_sector(rows, devices), tile)
                        self.assertEqual(implicit_tile_for_group(rows), tile)
                self.assertEqual(implicit_tile_for_sector(3_786_832, devices, float32_filter=True), 0)
            self.assertEqual(implicit_tile_for_group(3_786_832, float32_filter=True), 0)
        # The limit of one device does not move that of a group, nor the other way round.
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="600000"):
            self.assertEqual([implicit_tile_for_sector(677_542, devices) for devices in (1, 2, 4)], [16, 0, 0])
            self.assertEqual([implicit_tile_for_sector(1_194_720, devices) for devices in (1, 2, 4)], [16, 16, 16])
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="2000000"):
            self.assertEqual([implicit_tile_for_sector(1_194_720, devices) for devices in (1, 2, 4)], [0, 16, 16])
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS="500000"):
            self.assertEqual([implicit_tile_for_sector(546_152, devices) for devices in (1, 2, 4)], [0, 16, 16])
            self.assertEqual(implicit_tile_for_sector(499_999, 2), 0)
            self.assertEqual(implicit_tile_for_sector(546_152, 2, float32_filter=True), 0)
        with _settings(PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS=" 2000000 "):
            self.assertEqual([implicit_tile_for_sector(1_924_792, devices) for devices in (1, 2, 4)], [16, 0, 0])
            self.assertEqual(implicit_tile_for_sector(2_604_846, 4), 16)
        # An explicit tile size is taken as it is by a group as well.
        for value in (0, 16, 64):
            with _settings(PARSEC_CUPY_IMPLICIT_TILE=str(value), PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS="2000000"):
                self.assertEqual([implicit_tile_for_group(rows, float32) for rows, float32 in
                                  ((8, False), (3_786_832, False), (3_786_832, True))], [value]*3)
        # A value that is no row count is an error of the route that reads it.
        for value in ("0", "-5", "large"):
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS=value):
                for devices in (2, 4):
                    with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS"):
                        implicit_tile_for_sector(2_000_000, devices)
                self.assertEqual(implicit_tile_for_sector(2_000_000, 1), 16)
            with self.subTest(value=value), _settings(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS=value):
                self.assertEqual(implicit_tile_for_sector(2_000_000, 2), 16)

    def test_tile_lookup_replaces_the_slot_major_read_in_both_kernels(self):
        slot_major = "static_cast<long long>(slot) * row_count + row"
        self.assertEqual(_CUDA_SOURCE.count(slot_major), 2)
        for tile in (16, 64):
            source = implicit_source(tile)
            self.assertNotIn(slot_major, source)
            self.assertEqual(source.count(f"neighbors[row / {tile}]"), 2)
            self.assertEqual(source.count(f"slot * {tile} + row % {tile}"), 2)
            # Nothing else of the kernels moves: same declarations, same arithmetic.
            self.assertEqual(source.count("void stencil_major_"), _CUDA_SOURCE.count("void stencil_major_"))
            self.assertEqual(source.count("accumulator[column] += coefficient"), 2)

    def test_stencil_outside_a_sector_is_packed_by_an_explicit_size_only(self):
        metadata = build_stencil_major_metadata(sp.eye(64, format="csr"))
        packed = []

        def record(instance, cp, given, tile):
            packed.append(tile)

        def build(**arguments):
            packed.clear()
            # NumPy stands in for CuPy: the constructor uploads three arrays.
            with patch.object(cupy_stencil_major, "_compiled_kernels", return_value=(None, None)), \
                 patch.object(implicit_stencil, "initialize_implicit", record):
                CuPyStencilMajorFiniteDifference(np, metadata=metadata, **arguments)
            return list(packed)

        with _settings():
            self.assertEqual(build(), [])
            self.assertEqual(build(implicit_tile=16), [16])
            self.assertEqual(build(implicit_tile=0), [])
        with _settings(PARSEC_CUPY_IMPLICIT_TILE="auto"):
            self.assertEqual(build(), [])
        with _settings(PARSEC_CUPY_IMPLICIT_TILE="32"):
            self.assertEqual(build(), [32])
            # What the symmetry eigensolver names for a sector is not read again.
            self.assertEqual(build(implicit_tile=0), [])
            self.assertEqual(build(implicit_tile=16), [16])
        with _settings(PARSEC_CUPY_IMPLICIT_TILE="0"):
            self.assertEqual(build(), [])


class OperatorTileTests(unittest.TestCase):
    """The stencil an operator builds: ``CuPyHamiltonian`` with NumPy in place of CuPy."""
    slot_major = "stencil_major_int32_neighbors_uint8_coefficient_palette"

    def operator(self, *tile, rows=1031, **settings):
        """The operator of a sector that was named ``tile``, or of a stencil outside sectors."""
        metadata = build_stencil_major_metadata(_irregular_operator(rows))
        with _settings(**settings), host_cupy():
            return CuPyHamiltonian(None, np.zeros(rows), None, retain_generic_laplacian=False,
                                   finite_difference_metadata=metadata,
                                   **(dict(implicit_tile=tile[0]) if tile else {}))

    def test_operator_builds_the_layout_named_for_its_sector(self):
        packed = self.operator(16)
        stencil = packed.compact_finite_difference
        self.assertIsInstance(stencil, CuPyStencilMajorFiniteDifference)
        self.assertEqual(stencil.storage_mode, "implicit_affine_tile_16")
        self.assertEqual(stencil.implicit_statistics["rows"], 1031)
        self.assertIsNone(packed.compact_finite_difference_reason)
        plain = self.operator(0)
        self.assertEqual(plain.compact_finite_difference.storage_mode, self.slot_major)
        self.assertEqual(plain.compact_finite_difference.neighbors.shape[1], 1031)
        self.assertIsNone(plain.compact_finite_difference_reason)
        # What the symmetry eigensolver names is not read again from the environment.
        self.assertEqual(self.operator(0, PARSEC_CUPY_IMPLICIT_TILE="32").compact_finite_difference.storage_mode,
                         self.slot_major)
        # An operator outside symmetry sectors is packed by an explicit size only.
        self.assertEqual(self.operator().compact_finite_difference.storage_mode, self.slot_major)
        self.assertEqual(self.operator(PARSEC_CUPY_IMPLICIT_TILE="auto").compact_finite_difference.storage_mode,
                         self.slot_major)
        self.assertEqual(self.operator(PARSEC_CUPY_IMPLICIT_TILE="32").compact_finite_difference.storage_mode,
                         "implicit_affine_tile_32")

    def test_tiles_that_cannot_be_built_give_way_to_slot_major_or_raise_if_named(self):
        failure = ValueError("packed stencil exceeds int32 offsets")
        with patch.object(implicit_stencil, "pack_affine_tiles", side_effect=failure):
            # The default chose the tiles: the sector keeps the stencil-major kernels and their fused recurrence.
            for settings in ({}, {"PARSEC_CUPY_IMPLICIT_TILE": "auto"}):
                operator = self.operator(16, **settings)
                stencil = operator.compact_finite_difference
                self.assertIsInstance(stencil, CuPyStencilMajorFiniteDifference)
                self.assertEqual(stencil.storage_mode, self.slot_major)
                self.assertTrue(callable(stencil.chebyshev_recurrence))
                self.assertEqual(operator.compact_finite_difference_reason,
                                 "affine tiles of 16 ValueError: packed stencil exceeds int32 offsets")
            # The environment named them: no other layout stands in.
            for tile in ((16,), ()):
                with self.assertRaisesRegex(ValueError, "int32 offsets"):
                    self.operator(*tile, PARSEC_CUPY_IMPLICIT_TILE="16")
        # A value that is no tile size is an error for an operator outside sectors too.
        for tile in ((), (16,)):
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_TILE must be"):
                self.operator(*tile, PARSEC_CUPY_IMPLICIT_TILE="1")
        # Where the stencil-major kernels themselves fail, the CSR-order kernel is the fallback it always was.
        with patch.object(implicit_stencil, "pack_affine_tiles", side_effect=failure), \
             patch.object(cupy_stencil_major, "_compiled_kernels", side_effect=RuntimeError("no compiler")):
            operator = self.operator(16)
        self.assertIsInstance(operator.compact_finite_difference, CuPyCompactFiniteDifference)
        self.assertEqual(operator.compact_finite_difference_reason,
                         "affine tiles of 16 ValueError: packed stencil exceeds int32 offsets; "
                         "stencil-major RuntimeError: no compiler")
        # So it is with no tiles in play, whether by default or because the environment names 0.
        for settings in ({}, {"PARSEC_CUPY_IMPLICIT_TILE": "0"}):
            with patch.object(cupy_stencil_major, "_compiled_kernels", side_effect=RuntimeError("no compiler")):
                operator = self.operator(**settings)
            self.assertIsInstance(operator.compact_finite_difference, CuPyCompactFiniteDifference)
            self.assertEqual(operator.compact_finite_difference_reason, "stencil-major RuntimeError: no compiler")


    def test_a_pack_worker_setting_that_is_no_thread_count_is_an_error_of_the_operator(self):
        named = {"PARSEC_CUPY_IMPLICIT_TILE": "16"}
        for value in ("0", "-2", "auto"):
            with self.subTest(value=value):
                workers = {"PARSEC_CUPY_IMPLICIT_PACK_WORKERS": value}
                # Whoever chose the tiles: the default for a sector, or the environment for any operator.
                for tile, settings in (((16,), {}), ((16,), named), ((), named)):
                    with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_IMPLICIT_PACK_WORKERS"):
                        self.operator(*tile, **workers, **settings)
                # An operator that packs no tiles does not read it.
                for tile in ((0,), ()):
                    self.assertEqual(self.operator(*tile, **workers).compact_finite_difference.storage_mode,
                                     self.slot_major)
        # 2,500 tiles are three chunks; any thread count gives the operator the same packed arrays.
        serial = self.operator(16, rows=40_000, PARSEC_CUPY_IMPLICIT_PACK_WORKERS="1").compact_finite_difference
        for value in ("2", "8"):
            threaded = self.operator(16, rows=40_000, PARSEC_CUPY_IMPLICIT_PACK_WORKERS=value).compact_finite_difference
            np.testing.assert_array_equal(threaded.neighbors, serial.neighbors)
            np.testing.assert_array_equal(threaded.coefficient_codes, serial.coefficient_codes)
            self.assertEqual(threaded.implicit_statistics, serial.implicit_statistics)

    def test_a_pool_that_cannot_start_is_a_failure_of_the_tiles_like_any_other(self):
        class Pool:
            def __init__(self, **options):
                pass
            def map(self, work, items):
                raise RuntimeError("can't start new thread")
            def shutdown(self):
                pass

        with patch.object(implicit_stencil, "ThreadPoolExecutor", Pool):
            operator = self.operator(16, rows=40_000)
            self.assertEqual(operator.compact_finite_difference.storage_mode, self.slot_major)
            self.assertEqual(operator.compact_finite_difference_reason,
                             "affine tiles of 16 RuntimeError: can't start new thread")
            # One thread needs no pool.
            alone = self.operator(16, rows=40_000, PARSEC_CUPY_IMPLICIT_PACK_WORKERS="1")
            self.assertEqual(alone.compact_finite_difference.storage_mode, "implicit_affine_tile_16")
            with self.assertRaisesRegex(RuntimeError, "new thread"):
                self.operator(16, rows=40_000, PARSEC_CUPY_IMPLICIT_TILE="16")


class _DeviceArray(np.ndarray):
    """An array that knows the device it was uploaded to."""
    device = None

    def __array_finalize__(self, parent):
        self.device = getattr(parent, "device", None)


class _DevicesCuPy:
    """NumPy in place of CuPy on several devices: an upload lands on the current device, a download leaves it."""
    RawKernel = _HostRawKernel

    def __init__(self):
        # The devices that were made current, the innermost last, and (device, bytes) of every upload.
        self.current, self.uploads = [0], []
        cp = self

        class Device:
            def __init__(self, index=None):
                self.id = cp.current[-1] if index is None else int(index)

            def __enter__(self):
                cp.current.append(self.id)
                return self

            def __exit__(self, *_exc):
                cp.current.pop()
                return False

        self.cuda = SimpleNamespace(Device=Device, Stream=lambda non_blocking=False: object(),
                                    get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None))

    def asarray(self, values, dtype=None):
        if isinstance(values, _DeviceArray):
            if values.device.id != self.current[-1]:
                raise AssertionError("an array of another device was used")
            return values if dtype is None else values.astype(dtype, copy=False)
        array = np.array(values, dtype=dtype).view(_DeviceArray)
        array.device = SimpleNamespace(id=self.current[-1])
        self.uploads.append((self.current[-1], array.nbytes))
        return array

    def asnumpy(self, array):
        if array.device.id != self.current[-1]:
            raise AssertionError("a download needs the device of its array to be current")
        return np.array(array)

    def __getattr__(self, name):
        return getattr(np, name)


@contextmanager
def _devices(cp):
    """Let operators, their replicas and filter workspaces be built on the devices of ``cp`` without CUDA."""
    def runtime():
        return cp, sp

    with patch.object(cupy_backend, "require_cupy", runtime), patch.object(distributed_filter, "require_cupy", runtime), \
         patch.object(filter_graph, "require_cupy", runtime), \
         patch.dict(cupy_stencil_major._KERNEL_CACHE, clear=True), patch.dict(cupy_compact._KERNEL_CACHE, clear=True), \
         patch.dict(cupy_projectors._KERNEL_CACHE, clear=True), patch.dict(implicit_stencil._CACHE, clear=True):
        yield


class PackedStencilTests(unittest.TestCase):
    """A stencil that arrives packed: the arrays of ``pack_affine_tiles`` in place of the slot-major metadata."""

    def packed(self, rows=1031, tile=16):
        metadata = build_stencil_major_metadata(_irregular_operator(rows))
        with _settings():
            neighbors, codes, statistics = pack_affine_tiles(metadata, tile)
        return metadata, PackedTileHostMetadata(metadata.shape, metadata.neighbors.shape[0], neighbors, codes,
                                                metadata.coefficient_palette, statistics)

    def test_packed_arrays_are_uploaded_as_they_are(self):
        metadata, packed = self.packed()
        self.assertEqual((packed.tile, packed.shape, packed.slot_count), (16, (1031, 1031), 3))
        with patch.object(implicit_stencil, "pack_affine_tiles", side_effect=AssertionError("packed again")), \
             _settings(), host_cupy():
            operator = CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                       finite_difference_metadata=packed, implicit_tile=16)
        stencil = operator.compact_finite_difference
        self.assertIsInstance(stencil, CuPyStencilMajorFiniteDifference)
        self.assertEqual(stencil.storage_mode, "implicit_affine_tile_16")
        self.assertIsNone(operator.compact_finite_difference_reason)
        np.testing.assert_array_equal(stencil.neighbors, packed.neighbors)
        np.testing.assert_array_equal(stencil.coefficient_codes, packed.coefficient_codes)
        np.testing.assert_array_equal(stencil.coefficient_palette, metadata.coefficient_palette)
        # The slot count is that of the stencil, not the length of the flat packed array.
        self.assertEqual((stencil.slot_count, stencil.shape), (3, (1031, 1031)))
        self.assertEqual(stencil.implicit_statistics, packed.statistics)
        self.assertIsNot(stencil.implicit_statistics, packed.statistics)
        # The tiles are those of the slot-major stencil.
        neighbors, codes = unpack_affine_tiles(stencil.neighbors, stencil.coefficient_codes, 1031, 3, 16)
        np.testing.assert_array_equal(neighbors, metadata.neighbors)
        np.testing.assert_array_equal(codes[neighbors >= 0], metadata.coefficient_codes[neighbors >= 0])

    def test_a_packed_stencil_has_no_other_layout(self):
        _metadata, packed = self.packed()

        def operator(*tile, **settings):
            with _settings(**settings), host_cupy():
                return CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                       finite_difference_metadata=packed, **(dict(implicit_tile=tile[0]) if tile else {}))

        # Another tile size, the slot-major layout, and the layout that an environment without tiles names.
        with self.assertRaisesRegex(ValueError, "tiles of 16, not 32"):
            operator(32)
        for tile, settings in (((0,), {}), ((), {}), ((), {"PARSEC_CUPY_IMPLICIT_TILE": "0"})):
            with self.subTest(tile=tile, settings=settings), self.assertRaisesRegex(ValueError, "no slot-major layout"):
                operator(*tile, **settings)
        with self.assertRaisesRegex(ValueError, "no slot-major layout"):
            CuPyStencilMajorFiniteDifference(np, metadata=packed, implicit_tile=0)
        # Tiles that the default chose for a sector give way to the slot-major stencil where they cannot
        # be built. These cannot: the failure is the error, whatever the environment names.
        for settings in ({}, {"PARSEC_CUPY_IMPLICIT_TILE": "auto"}, {"PARSEC_CUPY_IMPLICIT_TILE": "16"}):
            with self.subTest(settings=settings), \
                 patch.object(cupy_compile, "compile_cupy_raw", side_effect=RuntimeError("no compiler")), \
                 self.assertRaisesRegex(RuntimeError, "no compiler"):
                operator(16, **settings)

    def test_arrays_that_are_not_those_of_the_statistics_are_refused(self):
        _metadata, packed = self.packed()
        self.assertEqual(replace(packed).statistics, packed.statistics)
        for changes in (dict(neighbors=packed.neighbors[:-1]), dict(coefficient_codes=packed.coefficient_codes[:-1]),
                        dict(neighbors=packed.neighbors.astype(np.int64)),
                        dict(coefficient_codes=packed.coefficient_codes.astype(np.int32)),
                        dict(neighbors=packed.neighbors[:-1], coefficient_codes=packed.coefficient_codes[:-1]),
                        dict(shape=(1030, 1030)), dict(slot_count=0),
                        dict(statistics={**packed.statistics, "tile": 8})):
            with self.subTest(changes=sorted(changes)), self.assertRaisesRegex(ValueError, "packed stencil"):
                replace(packed, **changes)
        # The kernels read FP64 coefficients, of which a code can name 256.
        palette = packed.coefficient_palette
        for name, given in (("float32", palette.astype(np.float32)), ("two rows", palette.reshape(1, -1)),
                            ("empty", palette[:0]), ("257", np.arange(257.))):
            with self.subTest(palette=name), self.assertRaisesRegex(ValueError, "packed stencil palette"):
                replace(packed, coefficient_palette=given)
        self.assertEqual(replace(packed, coefficient_palette=np.arange(256.)).coefficient_palette.size, 256)

    def test_the_palette_is_uploaded_in_the_precision_of_the_kernels(self):
        # As the slot-major upload names it; here for a palette that was put past the checks of the metadata.
        _metadata, packed = self.packed()
        object.__setattr__(packed, "coefficient_palette", packed.coefficient_palette.astype(np.float32))
        with _settings(), host_cupy():
            operator = CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                       finite_difference_metadata=packed, implicit_tile=16)
        self.assertEqual(operator.compact_finite_difference.coefficient_palette.dtype, np.float64)


class SectorGroupTileTests(unittest.TestCase):
    """The stencil of the devices that filter for a sector beside its owner: ``DistributedFilter`` without CUDA."""
    slot_major = "stencil_major_int32_neighbors_uint8_coefficient_palette"
    rows = 40_000

    def group(self, devices=(0, 1, 2, 3), failing=(), **settings):
        """The owner of a sector of ``rows`` rows under ``settings`` and the operators of ``devices``.

        Returns the stand-in, the slot-major metadata, the owner, the worker, the tile sizes that were packed
        and the devices whose kernels were compiled, per layout. A device of ``failing`` compiles none.
        """
        cp = _DevicesCuPy()
        metadata = build_stencil_major_metadata(_irregular_operator(self.rows))
        packs = []
        pack, compile_raw = implicit_stencil.pack_affine_tiles, cupy_compile.compile_cupy_raw

        def counted(*arguments):
            packs.append(arguments[1])
            return pack(*arguments)

        def compiled(kernel):
            if cp.current[-1] in failing:
                raise RuntimeError("no compiler")
            compile_raw(kernel)

        with _settings(**settings), _devices(cp), patch.object(implicit_stencil, "pack_affine_tiles", counted), \
             patch.object(cupy_compile, "compile_cupy_raw", compiled):
            tile = implicit_tile_for_sector(self.rows, len(devices))
            owner = CuPyHamiltonian(None, np.zeros(self.rows), None, retain_generic_laplacian=False,
                                    finite_difference_metadata=metadata, implicit_tile=tile)
            # As the routes that use the worker leave it, for the group of the sector to find.
            worker = owner._distributed_filter = DistributedFilter(owner, devices)
            kernels = sorted(implicit_stencil._CACHE), sorted(cupy_stencil_major._KERNEL_CACHE)
        return cp, metadata, owner, worker, packs, kernels

    def test_replicas_upload_the_tiles_that_the_owner_packed(self):
        large = dict(PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS="1000")
        for settings in (large, dict(PARSEC_CUPY_IMPLICIT_TILE="16"), dict(large, PARSEC_CUPY_IMPLICIT_PACK_WORKERS="1")):
            with self.subTest(settings=settings):
                cp, metadata, owner, worker, packs, kernels = self.group(**settings)
                stencil = owner.compact_finite_difference
                self.assertEqual(stencil.storage_mode, "implicit_affine_tile_16")
                # Packed once, by the owner.
                self.assertEqual(packs, [16])
                self.assertEqual(sorted(worker.replicas), [1, 2, 3])
                for device, replica in worker.replicas.items():
                    held = replica.compact_finite_difference
                    self.assertIsInstance(held, CuPyStencilMajorFiniteDifference)
                    self.assertEqual(held.storage_mode, "implicit_affine_tile_16")
                    self.assertIsNone(replica.compact_finite_difference_reason)
                    for name in ("neighbors", "coefficient_codes", "coefficient_palette"):
                        mine, theirs = getattr(held, name), getattr(stencil, name)
                        self.assertEqual((mine.device.id, mine.dtype, mine.shape), (device, theirs.dtype, theirs.shape))
                        np.testing.assert_array_equal(mine, theirs)
                    # The coefficients arrive in FP64 to the last bit: one of them has no FP32 value.
                    self.assertIn(.1, held.coefficient_palette.tolist())
                    self.assertEqual((held.slot_count, held.shape, held.implicit_statistics),
                                     (stencil.slot_count, stencil.shape, stencil.implicit_statistics))
                    self.assertEqual(replica.effective_potential.device.id, device)
                    # Nothing of the slot-major layout reaches the device: its uploads are the packed arrays,
                    # the palette, the potential and the empty projector factors.
                    uploaded = sum(size for target, size in cp.uploads if target == device)
                    self.assertEqual(uploaded, held.neighbors.nbytes + held.coefficient_codes.nbytes
                                     + held.coefficient_palette.nbytes + 8 * self.rows + 4 * (self.rows + 1))
                self.assertLess(stencil.neighbors.nbytes + stencil.coefficient_codes.nbytes,
                                (metadata.neighbors.nbytes + metadata.coefficient_codes.nbytes) // 2)
                # Every device has the tile kernels and a filter workspace of its own operator; no device
                # compiled the slot-major kernels.
                self.assertEqual(kernels, ([(device, 16) for device in range(4)], []))
                self.assertEqual(sorted(worker.graphs), [0, 1, 2, 3])
                self.assertIs(worker.graphs[0].operator, owner)
                for device, replica in worker.replicas.items():
                    self.assertIs(worker.graphs[device].operator, replica)
                    self.assertEqual(worker.graphs[device].device, device)
                group = SectorDeviceGroup(owner, (0, 1, 2, 3))
                self.assertIs(group._worker, worker)
                self.assertEqual(group.stencil_storage, "implicit_affine_tile_16")
                self.assertIsInstance(group.replica_seconds, float)
                self.assertGreater(group.replica_seconds, 0.0)
                # The tiles of every device are those of the slot-major stencil.
                slots = metadata.neighbors.shape[0]
                held = worker.replicas[3].compact_finite_difference
                neighbors, codes = unpack_affine_tiles(np.asarray(held.neighbors), np.asarray(held.coefficient_codes),
                                                       self.rows, slots, 16)
                np.testing.assert_array_equal(neighbors, metadata.neighbors)
                np.testing.assert_array_equal(codes[neighbors >= 0], metadata.coefficient_codes[neighbors >= 0])

    def test_without_tiles_the_replicas_hold_the_slot_major_stencil_as_before(self):
        # Below the row count of a group, which the limit of one device does not move, and the setting that
        # keeps the former route for every size.
        for settings in ({}, dict(PARSEC_CUPY_IMPLICIT_TILE_MIN_ROWS="1000"),
                         dict(PARSEC_CUPY_IMPLICIT_TILE="0", PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS="1000")):
            with self.subTest(settings=settings):
                cp, metadata, owner, worker, packs, kernels = self.group(devices=(0, 1), **settings)
                self.assertEqual(packs, [])
                self.assertEqual(kernels, ([], [0, 1]))
                held = worker.replicas[1].compact_finite_difference
                self.assertEqual((owner.compact_finite_difference.storage_mode, held.storage_mode), (self.slot_major,)*2)
                self.assertFalse(hasattr(held, "implicit_statistics"))
                np.testing.assert_array_equal(held.neighbors, metadata.neighbors)
                np.testing.assert_array_equal(held.coefficient_codes, metadata.coefficient_codes)
                self.assertEqual(held.neighbors.device.id, 1)
                self.assertEqual(SectorDeviceGroup(owner, (0, 1)).stencil_storage, self.slot_major)

    def test_a_device_that_cannot_build_the_tiles_of_its_owner_is_an_error(self):
        # The owner of a sector gives way to the slot-major stencil under the default. A replica does not:
        # it would filter with another layout than the report of its sector names.
        large = dict(PARSEC_CUPY_IMPLICIT_TILE_GROUP_MIN_ROWS="1000")
        with self.assertRaisesRegex(RuntimeError, "no compiler"):
            self.group(failing=(2,), **large)
        # Where the owner itself gave way, the others hold what it holds.
        _cp, metadata, owner, worker, packs, kernels = self.group(failing=(0,), devices=(0, 1), **large)
        self.assertEqual(owner.compact_finite_difference_reason, "affine tiles of 16 RuntimeError: no compiler")
        self.assertEqual((packs, kernels), ([16], ([], [0, 1])))
        held = worker.replicas[1].compact_finite_difference
        self.assertEqual((owner.compact_finite_difference.storage_mode, held.storage_mode), (self.slot_major,)*2)
        np.testing.assert_array_equal(held.neighbors, metadata.neighbors)
        # A size that the environment names is an error of the owner already.
        with self.assertRaisesRegex(RuntimeError, "no compiler"):
            self.group(failing=(0,), devices=(0, 1), PARSEC_CUPY_IMPLICIT_TILE="16")

    def test_a_group_whose_devices_hold_different_layouts_reports_mixed(self):
        # No route builds such a group; the report must not name the layout of the owner for all of them.
        cp, metadata, owner, worker, _packs, _kernels = self.group(devices=(0, 1, 2), PARSEC_CUPY_IMPLICIT_TILE="16")
        group = SectorDeviceGroup(owner, (0, 1, 2))
        self.assertEqual(group.stencil_storage, "implicit_affine_tile_16")
        with _settings(), _devices(cp), cp.cuda.Device(2):
            worker.replicas[2] = CuPyHamiltonian(None, np.zeros(self.rows), None, retain_generic_laplacian=False,
                                                 finite_difference_metadata=metadata, implicit_tile=0)
        self.assertEqual(worker.replicas[2].compact_finite_difference.storage_mode, self.slot_major)
        self.assertEqual(group.stencil_storage, "mixed")

    def test_the_stencil_for_the_replicas_is_downloaded_in_the_layout_of_the_owner(self):
        cp = _DevicesCuPy()
        metadata = build_stencil_major_metadata(_irregular_operator(1031))
        for tile in (0, 16, 64):
            with self.subTest(tile=tile), _settings(), _devices(cp):
                with cp.cuda.Device(2):
                    owner = CuPyHamiltonian(None, np.zeros(1031), None, retain_generic_laplacian=False,
                                            finite_difference_metadata=metadata, implicit_tile=tile)
                # Read on the device of the owner, whichever is current.
                given, named = replica_stencil(owner)
                self.assertEqual(named, tile)
                stencil = owner.compact_finite_difference
                np.testing.assert_array_equal(given.neighbors, stencil.neighbors)
                np.testing.assert_array_equal(given.coefficient_codes, stencil.coefficient_codes)
                np.testing.assert_array_equal(given.coefficient_palette, metadata.coefficient_palette)
                self.assertEqual(given.coefficient_palette.dtype, np.float64)
                self.assertNotIsInstance(given.neighbors, _DeviceArray)
                self.assertEqual(isinstance(given, PackedTileHostMetadata), bool(tile))
                if tile:
                    self.assertEqual((given.tile, given.slot_count, given.statistics),
                                     (tile, 3, stencil.implicit_statistics))


@unittest.skipUnless(cupy_available(), "CuPy/CUDA are not available")
class ImplicitKernelDeviceTests(unittest.TestCase):
    def tearDown(self):
        # Stencils that a test left in a reference cycle are destroyed here, between the tests.
        gc.collect()

    def test_packed_stencil_applies_bit_for_bit_like_the_slot_major_one(self):
        cp, _ = require_cupy()
        n = 1031
        metadata = build_stencil_major_metadata(_irregular_operator(n))
        generator = np.random.default_rng(41)
        vectors = cp.asarray(generator.standard_normal((n, 8)), order="F")
        previous = cp.asarray(generator.standard_normal((n, 8)), order="F")
        potential = cp.asarray(generator.standard_normal(n))
        dense = np.zeros((n, 2))
        dense[90:110, 0] = generator.standard_normal(20)
        dense[500:540, 1] = generator.standard_normal(40)
        projectors = sp.csr_matrix(dense)
        projector_data = (cp.asarray(projectors.indptr.astype(np.int32)), cp.asarray(projectors.indices.astype(np.int32)),
                          cp.asarray(projectors.data))
        coefficients = cp.asarray(generator.standard_normal((2, 8)), order="F")
        with _settings():
            plain = CuPyStencilMajorFiniteDifference(cp, metadata=metadata, implicit_tile=0)
            self.assertEqual(CuPyStencilMajorFiniteDifference(cp, metadata=metadata).storage_mode, plain.storage_mode)
        self.assertFalse(hasattr(plain, "implicit_statistics"))
        step = dict(center=0.5, scale=1/64, sigma=0.4, sigma_next=1.5)
        for tile in (16, 64):
            with self.subTest(tile=tile):
                packed = CuPyStencilMajorFiniteDifference(cp, metadata=metadata, implicit_tile=tile)
                self.assertEqual(packed.storage_mode, f"implicit_affine_tile_{tile}")
                self.assertGreater(packed.implicit_statistics["regular_rows"], 0)
                self.assertLess(packed.implicit_statistics["regular_rows"], n)
                for scatter in (dict(), dict(projector_data=projector_data, projector_coefficients=coefficients)):
                    np.testing.assert_array_equal(cp.asnumpy(packed.apply(vectors, potential, **scatter)),
                                                  cp.asnumpy(plain.apply(vectors, potential, **scatter)))
                    for former in (None, previous):
                        np.testing.assert_array_equal(
                            cp.asnumpy(packed.chebyshev_recurrence(vectors, potential, previous=former, **step, **scatter)),
                            cp.asnumpy(plain.chebyshev_recurrence(vectors, potential, previous=former, **step, **scatter)))
                np.testing.assert_array_equal(cp.asnumpy(packed.apply(vectors[:, 0])), cp.asnumpy(plain.apply(vectors[:, 0])))

    def test_an_operator_built_from_the_tiles_of_another_filters_bit_for_bit_like_it(self):
        # What a device of a sector group is given, here on the device of the owner: the packed arrays that
        # the owner holds, read back from it.
        from parsec_python.acceleration.Eigensolvers.chebyshev import FilterBlock
        from parsec_python.acceleration.Eigensolvers.filter_graph import BlockFilterGraphs
        cp, _ = require_cupy()
        n = 1031
        metadata = build_stencil_major_metadata(_irregular_operator(n))
        generator = np.random.default_rng(43)
        dense = np.zeros((n, 2))
        dense[90:110, 0] = .05 * generator.standard_normal(20)
        dense[500:540, 1] = .05 * generator.standard_normal(40)
        potential = .1 * generator.standard_normal(n)

        def operator(given, tile):
            with _settings():
                return CuPyHamiltonian(None, potential, (sp.csr_matrix(dense), np.array([1., -1.])),
                                       retain_generic_laplacian=False, finite_difference_metadata=given, implicit_tile=tile)

        owner, plain = operator(metadata, 16), operator(metadata, 0)
        given, tile = replica_stencil(owner)
        self.assertEqual((type(given), tile), (PackedTileHostMetadata, 16))
        with patch.object(implicit_stencil, "pack_affine_tiles", side_effect=AssertionError("packed again")):
            replica = operator(given, tile)
        held, packed = replica.compact_finite_difference, owner.compact_finite_difference
        self.assertEqual((held.storage_mode, held.slot_count, held.implicit_statistics),
                         ("implicit_affine_tile_16", packed.slot_count, packed.implicit_statistics))
        self.assertNotEqual(int(held.neighbors.data.ptr), int(packed.neighbors.data.ptr))
        for name in ("neighbors", "coefficient_codes", "coefficient_palette"):
            np.testing.assert_array_equal(cp.asnumpy(getattr(held, name)), cp.asnumpy(getattr(packed, name)))
        self.assertTrue(replica.fused_projector_scatter and replica.custom_projector_projection is not None)
        operators = (owner, replica, plain)
        host = generator.standard_normal((n, 14))
        # H times a block, as the Ritz step of a shared basis applies it on every device.
        vectors = cp.asarray(host, order="F")
        applied = [cp.asnumpy(item.apply_into(vectors, cp.empty_like(vectors))) for item in operators]
        self.assertTrue(np.isfinite(applied[0]).all() and applied[0].any())
        # The filter graphs of a device: two full blocks and a narrow one, the sigma carried between them.
        blocks = (FilterBlock(0, 6, 7), FilterBlock(6, 12, 7), FilterBlock(12, 14, 5))
        filtered = [cp.asnumpy(BlockFilterGraphs(item).apply(cp.asarray(host, order="F"), blocks, 1.2, 6., 1.2, False))
                    for item in operators]
        self.assertTrue(np.isfinite(filtered[0]).all() and filtered[0].any())
        for results in (applied, filtered):
            np.testing.assert_array_equal(results[1], results[0])
            np.testing.assert_array_equal(results[2], results[0])
