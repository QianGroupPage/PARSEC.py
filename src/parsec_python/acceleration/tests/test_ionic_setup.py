"""Check support selection, tails, interpolation, atom order and GPU slabs."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import os
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from parsec_python.Grid import build_cluster_grid
from parsec_python.models import Atom, GridSettings, SpeciesPotential
from parsec_python.V_ion import load_pseudopotentials
from parsec_python.acceleration.V_ion.native_ionic import (
    NativeIonicBuilders, _projector_box_rows, _projector_support_radius,
)
from parsec_python.acceleration.V_ion.cupy_ionic import (
    CupyIonicBuilders, ionic_device_ids, ionic_gpu_count_setting)
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.backends.cupy import cupy_available, require_cupy


_POTRE = Path(__file__).resolve().parents[4]/"examples/0d_naphthalene/C_POTRE.DAT"
_ATOMS = (Atom("C", (0.,0.,0.)), Atom("C", (.1,-.2,.05)), Atom("C", (4.,0.,0.)), Atom("C", (30.,0.,0.)))


def _settings(**values):
    """The environment without the device settings of the ionic sums, then ``values``."""
    names = ("PARSEC_IONIC_GPU_COUNT", "PARSEC_CUPY_DEVICES", "PARSEC_CUPY_TEMP_DIR")
    kept = {name: value for name, value in os.environ.items() if name not in names}
    return patch.dict(os.environ, {**kept, **values}, clear=True)


class _HostResult(np.ndarray):
    def get(self, out):
        out[...] = self


class _HostKernel:
    """Stands in for ``ionic_sum``: one value per grid point from its own coordinates."""
    def __init__(self, current):
        self.current, self.compiled, self.launches = current, [], []

    def compile(self):
        self.compiled.append((threading.get_ident(), self.current.id))

    def __call__(self, grid, block, arguments):
        xyz, count, positions, _types, atoms, _tables, _descriptors, _charges, local, output = arguments
        self.launches.append((threading.get_ident(), self.current.id, int(count), bool(self.compiled)))
        distances = np.sqrt(((xyz[:, None, :]-positions[None, :int(atoms), :])**2).sum(axis=2))
        output[...] = (distances*(2+int(local))).sum(axis=1)


def _host_cupy():
    """NumPy in place of CuPy, with a current device per thread as CuPy keeps one."""
    current = threading.local()
    current.id = 0

    class Device:
        def __init__(self, index=None):
            self.id = getattr(current, "id", 0) if index is None else int(index)
        def __enter__(self):
            self.former, current.id = getattr(current, "id", 0), self.id
            return self
        def __exit__(self, *error):
            current.id = self.former
            return False

    @contextmanager
    def using_allocator(allocator):
        yield

    cuda = SimpleNamespace(Device=Device, using_allocator=using_allocator)
    cp = SimpleNamespace(cuda=cuda, float64=np.float64, asarray=lambda values, order=None: np.asarray(values),
                         empty=lambda count, dtype: np.empty(count, dtype=dtype).view(_HostResult))
    return cp, current


class IonicDevicePolicyTests(unittest.TestCase):
    def test_default_takes_every_device_of_a_sector_rank_and_one_of_any_other_process(self):
        with _settings():
            self.assertEqual(ionic_gpu_count_setting(), "auto")
            self.assertEqual(ionic_device_ids((0, 1, 2, 3), 1000, sector_rank=True), (0, 1, 2, 3))
            self.assertEqual(ionic_device_ids((3, 1), 1000, sector_rank=True), (3, 1))
            self.assertEqual(ionic_device_ids((2,), 1000, sector_rank=True), (2,))
            # No device is left with an empty slab.
            self.assertEqual(ionic_device_ids((0, 1, 2, 3), 2, sector_rank=True), (0, 1))
            self.assertEqual(ionic_device_ids((0, 1, 2, 3), 0, sector_rank=True), (0,))
            # A process that may yet solve on one device touches no other: the former default.
            self.assertEqual(ionic_device_ids((0, 1, 2, 3), 1000), (0,))
            self.assertEqual(ionic_device_ids((3, 1), 1000, sector_rank=False), (3,))

    def test_explicit_count_wins_and_one_is_the_former_default(self):
        for sector_rank in (False, True):
            every = (3, 1, 2) if sector_rank else (3,)
            for value, expected in (("auto", every), ("", every), (" AUTO ", every), ("1", (3,)),
                                    ("2", (3, 1)), ("3", (3, 1, 2)), ("8", (3, 1, 2))):
                with self.subTest(value=value, sector_rank=sector_rank), _settings(PARSEC_IONIC_GPU_COUNT=value):
                    self.assertEqual(ionic_device_ids((3, 1, 2), 1000, sector_rank=sector_rank), expected)
                    self.assertEqual(ionic_gpu_count_setting(), value.strip().lower() or "auto")
        for value in ("0", "-2", "all", "1.5"):
            with self.subTest(value=value), _settings(PARSEC_IONIC_GPU_COUNT=value), \
                 self.assertRaisesRegex(ValueError, "PARSEC_IONIC_GPU_COUNT"):
                ionic_device_ids((0, 1), 1000)
        with _settings(), self.assertRaisesRegex(ValueError, "at least one"):
            ionic_device_ids((), 1000, sector_rank=True)


class IonicSlabTests(unittest.TestCase):
    """The slabs of the sums with NumPy in place of CuPy; four devices are visible."""
    @classmethod
    def setUpClass(cls):
        cls.spec = {"C": SpeciesPotential(_POTRE, 1, read_valence_density=False)}
        cls.pots = load_pseudopotentials(cls.spec)
        cls.grid = build_cluster_grid(GridSettings(spacing=.5, radius=4., expansion_order=8))

    def sums(self, sector_rank=True, **settings):
        """Both fields as an MPI rank that solves symmetry sectors sums them, or as another process does."""
        cp, current = _host_cupy()
        kernel = _HostKernel(current)
        module = "parsec_python.acceleration.V_ion.cupy_ionic."
        with patch(module+"require_cupy", return_value=(cp, None)), patch(module+"_kernel", return_value=kernel), \
             patch("parsec_python.acceleration.Eigensolvers.symmetry.cupy_device_count", return_value=4), \
             _settings(**settings):
            builders = CupyIonicBuilders(sector_rank=sector_rank)
            self.assertEqual(builders.device_ids, ())
            fields = [builders.build_local_ionic_potential(self.grid, _ATOMS, self.pots, self.spec),
                      builders.superpose_atomic_density(self.grid, _ATOMS, self.pots, self.spec)]
        # What the builders say they used is what launched.
        self.assertEqual(sorted(builders.device_ids), sorted({launch[1] for launch in kernel.launches}))
        return fields, kernel

    def test_default_uses_every_device_of_a_sector_rank_and_one_restores_the_single_slab(self):
        single, one = self.sums(PARSEC_IONIC_GPU_COUNT="1")
        self.assertEqual([launch[1:3] for launch in one.launches], [(0, self.grid.size)]*2)
        # One device sums in the calling thread and compiles when it launches, as before.
        self.assertEqual({launch[0] for launch in one.launches}, {threading.get_ident()})
        self.assertEqual(one.compiled, [])
        self.assertFalse(np.array_equal(single[0], single[1]))

        # A process without a sector context does exactly that by default: it may solve on one device.
        alone, other = self.sums(sector_rank=False)
        self.assertEqual(other.launches, one.launches)
        self.assertEqual(other.compiled, [])
        for one_device, default in zip(single, alone):
            np.testing.assert_array_equal(default, one_device)
        # It still takes the devices a count names.
        self.assertEqual(sorted(launch[1] for launch in self.sums(False, PARSEC_IONIC_GPU_COUNT="3")[1].launches[:3]),
                         [0, 1, 2])

        fields, every = self.sums()
        for launches in (every.launches[:4], every.launches[4:]):
            self.assertEqual(sorted(launch[1] for launch in launches), [0, 1, 2, 3])
            self.assertEqual(sum(launch[2] for launch in launches), self.grid.size)
        # The kernel was compiled by the calling thread before any device thread launched it.
        self.assertEqual(every.compiled, [(threading.get_ident(), 0)]*2)
        self.assertTrue(all(launch[3] for launch in every.launches))
        self.assertNotIn(threading.get_ident(), {launch[0] for launch in every.launches})
        for one_device, all_devices in zip(single, fields):
            np.testing.assert_array_equal(all_devices, one_device)

    def test_devices_follow_the_process_setting_and_an_explicit_count(self):
        single, _ = self.sums(PARSEC_IONIC_GPU_COUNT="1")
        for settings, expected in ((dict(PARSEC_CUPY_DEVICES="1,3"), [1, 3]),
                                   (dict(PARSEC_CUPY_DEVICES="auto", PARSEC_IONIC_GPU_COUNT="3"), [0, 1, 2]),
                                   (dict(PARSEC_CUPY_DEVICES="3,0,2", PARSEC_IONIC_GPU_COUNT="2"), [0, 3]),
                                   (dict(PARSEC_CUPY_DEVICES="current"), [0]),
                                   (dict(PARSEC_CUPY_DEVICES="2", PARSEC_IONIC_GPU_COUNT="4"), [2])):
            with self.subTest(settings=settings):
                fields, kernel = self.sums(**settings)
                count = len(expected)
                self.assertEqual(sorted(launch[1] for launch in kernel.launches[:count]), expected)
                self.assertEqual(len(kernel.launches), 2*count)
                for one_device, several in zip(single, fields):
                    np.testing.assert_array_equal(several, one_device)

    def test_sums_on_another_thread_take_the_device_their_caller_kept(self):
        # A new thread starts with device 0 current. Under the process setting "current" the sums follow the
        # device of the thread that sums, unless the caller that hands them over has kept its own.
        cp, current = _host_cupy()
        kernel = _HostKernel(current)
        module = "parsec_python.acceleration.V_ion.cupy_ionic."
        arguments = (self.grid, _ATOMS, self.pots, self.spec)
        with patch(module+"require_cupy", return_value=(cp, None)), patch(module+"_kernel", return_value=kernel), \
             patch("parsec_python.acceleration.Eigensolvers.symmetry.cupy_device_count", return_value=4):
            for setting in ("current", "off"):
                for kept, expected in ((False, 0), (True, 2)):
                    with self.subTest(setting=setting, kept=kept), _settings(PARSEC_CUPY_DEVICES=setting), \
                         cp.cuda.Device(2):
                        builders = CupyIonicBuilders(sector_rank=True)
                        self.assertIsNone(builders.current_device)
                        # In line the sums are on the caller's device either way.
                        inline = builders.build_local_ionic_potential(*arguments)
                        self.assertEqual(builders.device_ids, (2,))
                        if kept:
                            builders.keep_current_device()
                            self.assertEqual(builders.current_device, 2)
                        del kernel.launches[:]
                        fields = []
                        thread = threading.Thread(
                            target=lambda: fields.append(builders.build_local_ionic_potential(*arguments)))
                        thread.start()
                        thread.join()
                        self.assertEqual(builders.device_ids, (expected,))
                        self.assertEqual([launch[1] for launch in kernel.launches], [expected])
                        self.assertNotEqual(kernel.launches[0][0], threading.get_ident())
                        np.testing.assert_array_equal(fields[0], inline)
            # A list of devices does not depend on the current one.
            with _settings(PARSEC_CUPY_DEVICES="1,3"), cp.cuda.Device(2):
                builders = CupyIonicBuilders(sector_rank=True)
                builders.keep_current_device()
                builders.build_local_ionic_potential(*arguments)
                self.assertEqual(builders.device_ids, (1, 3))


class SupportBoxTests(unittest.TestCase):
    def test_shifted_grid_boundary_and_outside_atoms(self):
        grid = build_cluster_grid(GridSettings(spacing=.5, radius=3., expansion_order=8,
                                               shift=(.25, -.25, .5)))
        for position in ((0.,0.,0.), (2.875,0.,0.), (20.,0.,0.)):
            for radius in (0., .5, 1., 5.):
                with self.subTest(position=position, radius=radius):
                    rows = _projector_box_rows(grid, position, radius)
                    delta = grid.coordinates - np.asarray(position)
                    exact = np.flatnonzero(np.sum(delta*delta, axis=1) <= radius*radius)
                    self.assertTrue(np.all(np.isin(exact, rows)))
                    self.assertTrue(np.all(np.diff(rows) > 0))


@unittest.skipUnless(native_available(), "native extension required")
class IonicSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[4]
        cls.spec = {"C": SpeciesPotential(root/"examples/0d_naphthalene/C_POTRE.DAT", 1,
                                           read_valence_density=False)}
        cls.pots = load_pseudopotentials(cls.spec)
        cls.grid = build_cluster_grid(GridSettings(spacing=.5, radius=4., expansion_order=8))
        cls.atoms = (Atom("C", (0.,0.,0.)), Atom("C", (.1,-.2,.05)),
                     Atom("C", (4.,0.,0.)), Atom("C", (30.,0.,0.)))

    def test_lookup_projectors_bitwise_equal_full_scan(self):
        for spline in (False, True):
            spec = {"C": replace(self.spec["C"], use_spline=spline)}
            with patch.dict(os.environ, PARSEC_NATIVE_PROJECTOR_LOOKUP="0"):
                full = NativeIonicBuilders().build_nonlocal_projectors(self.grid, self.atoms, self.pots, spec)
            with patch.dict(os.environ, PARSEC_NATIVE_PROJECTOR_LOOKUP="1"):
                box = NativeIonicBuilders().build_nonlocal_projectors(self.grid, self.atoms, self.pots, spec)
            self.assertEqual(full.labels, box.labels)
            for name in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(full.projectors,name), getattr(box.projectors,name))
            np.testing.assert_array_equal(full.signs, box.signs)

    @unittest.skipUnless(cupy_available(), "CUDA device required")
    def test_gpu_fields_and_multidevice_partition(self):
        cp, _ = require_cupy()
        gpu = CupyIonicBuilders()
        native = NativeIonicBuilders()
        for spline in (False, True):
            for vcd in (False, True):
                spec = {"C": replace(self.spec["C"], use_spline=spline, read_valence_density=vcd)}
                # Synthetic NLCC exercises a path absent from C/H benchmark inputs.
                pots = {"C": replace(self.pots["C"], core_marker="pcec",
                                       core_density=np.exp(-self.pots["C"].radii**2))}
                expected = [native.build_local_ionic_potential(self.grid,self.atoms,pots,spec)]
                expected += [native.superpose_atomic_density(self.grid,self.atoms,pots,spec,core=core)
                             for core in (False,True)]
                for count in (1, min(4,cp.cuda.runtime.getDeviceCount())):
                    with patch.dict(os.environ, PARSEC_CUPY_DEVICES="auto", PARSEC_IONIC_GPU_COUNT=str(count)):
                        actual = [gpu.build_local_ionic_potential(self.grid,self.atoms,pots,spec)]
                        actual += [gpu.superpose_atomic_density(self.grid,self.atoms,pots,spec,core=core)
                                   for core in (False,True)]
                    for a,b in zip(actual,expected):
                        np.testing.assert_allclose(a,b,rtol=2e-14,atol=2e-14)
        np.testing.assert_array_equal(gpu.superpose_atomic_density(self.grid,self.atoms,self.pots,self.spec,core=True),
                                      np.zeros(self.grid.size))

    @unittest.skipUnless(cupy_available(), "CUDA device required")
    def test_radial_cutoff_and_origin(self):
        radius = self.pots["C"].radii[-2]
        r0 = self.pots["C"].radii[0]
        distances = np.array([0., r0, np.nextafter(radius,0.), radius, np.nextafter(radius,np.inf), 2*radius])
        coords = np.column_stack((distances,np.zeros((len(distances),2))))
        grid = replace(self.grid, coordinates=coords)
        atoms = self.atoms[:1]
        for spline in (False,True):
            spec = {"C": replace(self.spec["C"], use_spline=spline)}
            for method in ("build_local_ionic_potential", "superpose_atomic_density"):
                with patch.dict(os.environ, PARSEC_CUPY_DEVICES="0", PARSEC_IONIC_GPU_COUNT="1"):
                    a = getattr(CupyIonicBuilders(),method)(grid,atoms,self.pots,spec)
                b = getattr(NativeIonicBuilders(),method)(grid,atoms,self.pots,spec)
                np.testing.assert_allclose(a,b,rtol=2e-14,atol=2e-14)


@unittest.skipUnless(cupy_available(), "CUDA device required")
class IonicDeviceCountTests(unittest.TestCase):
    def test_gpu_fields_are_bitwise_the_same_on_one_and_on_every_device(self):
        cp, _ = require_cupy()
        if cp.cuda.runtime.getDeviceCount() < 2:
            self.skipTest("two CUDA devices required")
        base = SpeciesPotential(_POTRE, 1, read_valence_density=False)
        loaded = load_pseudopotentials({"C": base})
        # Synthetic NLCC, as in the partition test above.
        pots = {"C": replace(loaded["C"], core_marker="pcec", core_density=np.exp(-loaded["C"].radii**2))}
        grid = build_cluster_grid(GridSettings(spacing=.5, radius=4., expansion_order=8, shift=(.25, -.25, .5)))

        def fields(spec, devices, **settings):
            with _settings(PARSEC_CUPY_DEVICES="auto", **settings):
                # As an MPI rank that solves symmetry sectors: the default takes every device.
                gpu = CupyIonicBuilders(sector_rank=True)
                sums = [gpu.build_local_ionic_potential(grid, _ATOMS, pots, spec),
                        *(gpu.superpose_atomic_density(grid, _ATOMS, pots, spec, core=core) for core in (False, True))]
                self.assertEqual(len(gpu.device_ids), devices)
                return sums

        every_device = cp.cuda.runtime.getDeviceCount()
        for spline in (False, True):
            for vcd in (False, True):
                with self.subTest(spline=spline, vcd=vcd):
                    spec = {"C": replace(base, use_spline=spline, read_valence_density=vcd)}
                    for one, every in zip(fields(spec, 1, PARSEC_IONIC_GPU_COUNT="1"), fields(spec, every_device)):
                        np.testing.assert_array_equal(every, one)

    def test_gpu_fields_summed_on_a_thread_beside_another_user_of_the_devices_are_bitwise_the_same(self):
        # As the root rank of an MPI run sums them by default: on a thread of their own, on every device,
        # while the thread that prepares uploads to the same devices. One device is enough to run this.
        cp, _ = require_cupy()
        count = cp.cuda.runtime.getDeviceCount()
        base = SpeciesPotential(_POTRE, 1, read_valence_density=False)
        pots = load_pseudopotentials({"C": base})
        spec = {"C": base}
        grid = build_cluster_grid(GridSettings(spacing=.5, radius=4., expansion_order=8, shift=(.25, -.25, .5)))
        arguments = (grid, _ATOMS, pots, spec)
        with _settings(PARSEC_CUPY_DEVICES="auto", PARSEC_IONIC_GPU_COUNT="1"):
            inline = CupyIonicBuilders(sector_rank=True)
            expected = [inline.build_local_ionic_potential(*arguments), inline.superpose_atomic_density(*arguments)]
        host = np.arange(4096, dtype=np.float64)
        with _settings(PARSEC_CUPY_DEVICES="auto"):
            gpu = CupyIonicBuilders(sector_rank=True)
            gpu.keep_current_device()
            fields, failures = [], []

            def sums():
                try:
                    fields.extend((gpu.build_local_ionic_potential(*arguments),
                                   gpu.superpose_atomic_density(*arguments)))
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=sums, name="parsec-ionic-setup-test")
            thread.start()
            rounds = 0
            while rounds == 0 or thread.is_alive():
                for device in range(count):
                    with cp.cuda.Device(device):
                        np.testing.assert_array_equal(cp.asnumpy(cp.asarray(host)*2.0), host*2.0)
                rounds += 1
            thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(gpu.device_ids, tuple(range(count)))
        for actual, wanted in zip(fields, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)


if __name__ == "__main__":
    unittest.main()
