"""Host gather must preserve input values/order and avoid full-row copies."""
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np

from parsec_python.acceleration.experimental.host_gather import gather_rows_columns


class HostGatherTests(unittest.TestCase):
    def test_full_f_rows_use_a_view(self):
        source = np.arange(256*25, dtype=np.float64).reshape((256, 25), order='F')
        selected = gather_rows_columns(source, np.arange(256), 3, 21)
        self.assertTrue(np.shares_memory(source, selected))
        self.assertTrue(selected.flags.f_contiguous)
        np.testing.assert_array_equal(selected, source[:, 3:21])

    def test_partial_contiguous_and_arbitrary_rows_preserve_order(self):
        for order in ('C', 'F'):
            source = np.array(np.arange(256*25).reshape(256, 25), dtype=np.float64, order=order)
            for rows in (np.arange(13, 201), np.array([200, 0, 99, 99, 7]),
                         np.array([31]), np.array([], dtype=np.int64)):
                selected = gather_rows_columns(source, rows, 4, 19)
                self.assertTrue(selected.flags.f_contiguous)
                np.testing.assert_array_equal(selected, source[rows, 4:19])

    def test_f_order_memmap(self):
        # A direct temporary file avoids platform-specific directory ACLs
        # imposed by tempfile's private-directory mode on Windows runners.
        # Some file systems map no file into memory (ENOSYS on a DVS mount
        # of a compute node): the temporary directory of the system is tried
        # after the working directory, and the test is skipped where neither
        # maps one.
        name = '.host_gather_'+uuid.uuid4().hex+'.npy'
        for folder in (Path.cwd(), Path(tempfile.gettempdir())):
            path = folder/name
            try:
                try:
                    writer = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                                       shape=(257, 31), fortran_order=True)
                    try:
                        writer[:] = np.arange(257*31).reshape(257, 31)
                        writer.flush()
                    finally:
                        writer._mmap.close()
                    source = np.load(path, mmap_mode='r', allow_pickle=False)
                except OSError:
                    continue
                try:
                    for rows in (np.arange(257), np.arange(10, 220), np.array([250, 1, 110, 2, 110])):
                        result = gather_rows_columns(source, rows, 2, 30)
                        np.testing.assert_array_equal(result, source[rows, 2:30])
                        self.assertTrue(result.flags.f_contiguous)
                        if len(rows) == 257:
                            self.assertTrue(np.shares_memory(result, source))
                finally:
                    source._mmap.close()
                return
            finally:
                path.unlink(missing_ok=True)
        self.skipTest('no directory here maps a file into memory')

    def test_invalid_indices_or_columns_are_rejected(self):
        source = np.zeros((8, 5))
        for rows, start, stop in ((np.array([-1]), 0, 2), (np.array([8]), 0, 2),
                                  (np.array([1.0]), 0, 2), (np.array([1]), 3, 2),
                                  (np.array([1]), 0, 6)):
            with self.assertRaises(ValueError):
                gather_rows_columns(source, rows, start, stop)


if __name__ == '__main__':
    unittest.main()
