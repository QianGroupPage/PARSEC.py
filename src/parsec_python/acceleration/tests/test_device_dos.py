"""The tetrahedron sums of Create_DOS on the GPU equal those of the host bit for bit.

:mod:`parsec_python.acceleration.Output.device_dos` runs the readable
:func:`parsec_python.Output.dos.spectrum_sums` with CuPy arrays and sums the pairs onto the mesh in
the order of ``bincount``.  The tests compare it with the host on rows that reach every branch of
``spec_tet`` (the four regions, ``e1 = e2``, corners merged within 1e-9 eV, all-equal rows, mesh
points nudged off a corner) and on the DOS file of a k-point run, and check when the provider
declines.  They need CuPy and a GPU (skipped otherwise), apart from the rule tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from parsec_python.Output import dos as D
from parsec_python.acceleration.Output import device_dos

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def _cupy_device() -> bool:
    try:
        return device_dos._device_id() is not None
    except Exception:  # noqa: BLE001
        return False


def _rows(seed: int = 7, count: int = 3000, columns: int = 5):
    """Corner energies (eV) and values that reach every branch of spec_tet."""

    rng = np.random.default_rng(seed)
    energies = rng.uniform(-8.0, 4.0, (count, 1)) + rng.uniform(0.0, 1.5, (count, 4))
    energies[::7, 1] = energies[::7, 0]  # e1 = e2 (the flat branch of region 3)
    energies[::11, 2] = energies[::11, 3] + 4e-10  # merged within 1e-9 eV
    energies[::13] = energies[::13, :1]  # all four corners equal
    energies[::17, 0] = energies[::17, 2]  # ties for the heap sort
    energies[::19, 1] = np.round(energies[::19, 1], 2)  # on the points of a 0.01 eV mesh
    values = rng.uniform(0.0, 0.2, (count, 4, columns))
    return energies, values


def _spectrum(energies, values, mesh, mode: str):
    with mock.patch.dict(os.environ, {device_dos.DEVICE_ENV: mode}):
        return D.tetrahedron_spectrum(2.5, energies, mesh, values)


@unittest.skipUnless(_cupy_device(), "needs CuPy and a CUDA device")
class DeviceSpectrumTests(unittest.TestCase):
    def test_sums_equal_the_host_bit_for_bit(self) -> None:
        energies, values = _rows()
        for mesh in (
            D.dos_energy_mesh(-8.0, 6.0, 700),
            # a mesh whose points fall on corners: the 1e-8 nudges
            D.DOSMesh(lower=-8.0, upper=6.0, points=1400, step=0.01),
        ):
            for corner_values in (values, None):
                with self.subTest(points=mesh.points, projected=corner_values is not None):
                    host = _spectrum(energies, corner_values, mesh, "off")
                    device = _spectrum(energies, corner_values, mesh, "on")
                    np.testing.assert_array_equal(device[0], host[0])
                    self.assertEqual(device[2], host[2])
                    if corner_values is None:
                        self.assertIsNone(device[1])
                    else:
                        np.testing.assert_array_equal(device[1], host[1])
                    self.assertGreater(host[2], 0)  # the all-equal rows were met

    def test_column_blocks_do_not_change_the_numbers(self) -> None:
        energies, values = _rows(seed=3, count=500, columns=9)
        mesh = D.dos_energy_mesh(-8.0, 6.0, 300)
        host = _spectrum(energies, values, mesh, "off")
        with mock.patch.object(device_dos, "_MEMORY_FRACTION", 1e-12):  # one column per block
            device = _spectrum(energies, values, mesh, "on")
        np.testing.assert_array_equal(device[1], host[1])

    def test_dos_file_of_a_kpoint_run(self) -> None:
        from parsec_python.acceleration.tests.test_kpoint_post_processing_device import CELL, MP
        from parsec_python.cli import main

        texts = {}
        for mode in ("off", "on"):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                shutil.copy(DATA / "H_POTRE.DAT", directory / "H_POTRE.DAT")
                path = directory / "parsec.in"
                extra = MP.format(nz=1) + "Create_DOS: .true.\nDOS_Pnum: 300\nYLM_DOS_L: 2\nylm_dos_cutoff: 1.4\n"
                path.write_text(CELL.format(extra=extra), encoding="utf-8")
                # the k-point SCF on the host in both runs (deterministic levels and orbitals);
                # the tetrahedra on the GPU in one of them
                # (PARSEC_KPOINT_BACKEND concerns the k-point eigensolves, not the DOS)
                environment = {device_dos.DEVICE_ENV: mode, "PARSEC_KPOINT_BACKEND": "cpu"}
                with mock.patch.dict(os.environ, environment):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        self.assertEqual(main([str(path), "--quiet"]), 0)
                texts[mode] = (directory / "DOS").read_text()
        self.assertEqual(texts["on"], texts["off"])
        self.assertIn("Graph Name : H1_dz2_sp1", texts["on"])


class ProviderRuleTests(unittest.TestCase):
    def test_provider_is_installed(self) -> None:
        self.assertIs(D._spectrum_provider, device_dos.device_spectrum_sums)

    def test_device_follows_the_device_request_not_the_kpoint_override(self) -> None:
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "cpu"}):
            self.assertIsNone(device_dos._device_id())
        with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "anything"}):
            with self.assertRaises(ValueError):
                device_dos._device_id()
        if _cupy_device():
            with mock.patch.dict(os.environ, {"PARSEC_DEVICE": "auto", "PARSEC_KPOINT_BACKEND": "cpu"}):
                self.assertIsNotNone(device_dos._device_id())

    def test_declines_when_off_small_or_without_a_device(self) -> None:
        energies, values = _rows(count=40, columns=2)
        mesh = D.dos_energy_mesh(-8.0, 6.0, 100)
        es, fs = D._sorted_corners(energies, values)
        rows = D._spectrum_rows(2.5, es, fs, mesh)
        with mock.patch.dict(os.environ, {device_dos.DEVICE_ENV: "off"}):
            self.assertIsNone(device_dos.device_spectrum_sums(2.5, rows, mesh))
        with mock.patch.dict(os.environ, {device_dos.DEVICE_ENV: "auto"}):
            # far fewer pair values than AUTO_VALUES: the host is faster
            self.assertIsNone(device_dos.device_spectrum_sums(2.5, rows, mesh))
        with mock.patch.dict(os.environ, {device_dos.DEVICE_ENV: "on"}):
            with mock.patch.object(device_dos, "_device_id", return_value=None):
                self.assertIsNone(device_dos.device_spectrum_sums(2.5, rows, mesh))
        with mock.patch.dict(os.environ, {device_dos.DEVICE_ENV: "sometimes"}):
            with self.assertRaises(ValueError):
                device_dos.device_spectrum_sums(2.5, rows, mesh)

    def test_host_sums_without_the_provider_are_the_readable_definition(self) -> None:
        energies, values = _rows(count=200, columns=3)
        mesh = D.dos_energy_mesh(-8.0, 6.0, 150)
        with mock.patch.object(D, "_spectrum_provider", None):
            readable = D.tetrahedron_spectrum(2.5, energies, mesh, values)
        installed = _spectrum(energies, values, mesh, "off")
        for left, right in zip(readable[:2], installed[:2]):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
