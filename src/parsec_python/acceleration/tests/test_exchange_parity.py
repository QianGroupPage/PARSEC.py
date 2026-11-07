"""Pair densities of sector orbitals in reflection-parity classes (``exchange_device.py``).

``W = V_x Psi`` of the same sector orbitals through the parity path (coset
form on the octant, class solves; the default for sector orbitals) and the
full-grid path (``PARSEC_CUPY_EXCHANGE_PARITY=off``: the sectors unfolded)
must agree to round-off, for every group the detection gives (D2h, D2, C2v,
Cs, Ci and the signed-permutation groups), grids with an even number of
points per axis (shift 1/2) and with points on the mirror planes (shift 0,
where orbits with a stabilizer are rejected by some sectors), the Coulomb,
screened and weighted kernels, an ACE build in two steps (occupied columns,
then the empty ones), and on one device or all.  Random orbitals under a
Gaussian envelope on a sphere of 4 bohr keep every case to seconds.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available

# Atoms (symbol, position in bohr) whose symmetry gives each group.
MOLECULES = {
    "D2h": [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    "D2": [("C", (0.0, 0.0, 0.0))]
    + [("H", (1.18 * sx, 1.18 * sy, 1.18 * sx * sy)) for sx in (1, -1) for sy in (1, -1)],
    "C2v": [("O", (0.0, 0.0, 0.0)), ("H", (1.43, 0.0, 1.1)), ("H", (-1.43, 0.0, 1.1))],
    "Cs": [("O", (0.0, 0.0, 0.0)), ("H", (1.4, 1.1, 0.0)), ("N", (-0.7, 0.35, 0.0))],
    "Ci": [("O", (1.0, 0.3, 0.2)), ("O", (-1.0, -0.3, -0.2)), ("H", (0.2, 1.1, -0.4)), ("H", (-0.2, -1.1, 0.4))],
    # Mirror planes x = y and x = -y and the z axis: a C2v of signed permutations, larger than
    # its diagonal subgroup {E, C2z}.
    "signed": [("C", (0.0, 0.0, 0.0)), ("H", (1.1, 1.1, 0.7)), ("H", (-1.1, -1.1, 0.7))],
}


def _device_count() -> int:
    if not cupy_available():
        return 0
    import cupy as cp

    return int(cp.cuda.runtime.getDeviceCount())


def sector_channel(grid, molecule, *, states=14, occupied=9, devices=(0,), seed=3):
    """:class:`ChannelOrbitals` of random sector orbitals of ``molecule`` on ``grid``."""

    import cupy as cp

    from parsec_python.acceleration.Hamiltonian.exchange_device import (
        ChannelOrbitals,
        OrbitalSector,
        SectorExpansion,
    )
    from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction
    from parsec_python.acceleration.Symmetry.representations import ReflectionRepresentationDecomposition
    from parsec_python.models import Atom

    atoms = [Atom(symbol, position) for symbol, position in MOLECULES[molecule]]
    reduction = SignedPermutationReduction.detect(grid, atoms)
    decomposition = ReflectionRepresentationDecomposition.build(grid, reduction)
    count = decomposition.representation_count
    rng = np.random.default_rng(seed)
    envelope = np.exp(-0.2 * np.einsum("ij,ij->i", grid.coordinates, grid.coordinates))
    # Occupied and empty states spread over the sectors, the first sector heaviest.
    per_sector = [states // count + (1 if r < states % count else 0) for r in range(count)]
    occupations = np.zeros(sum(per_sector))
    sectors, next_state = [], 0
    remaining = occupied
    for representation in range(count):
        columns = per_sector[representation]
        if not columns:
            continue
        orbits = decomposition.sector_orbit_indices(representation)
        scales = 1.0 / np.sqrt(decomposition.reduction.multiplicities[orbits])
        expansion = SectorExpansion(
            decomposition.reduction.full_to_wedge, decomposition.phases[representation], orbits, scales,
            decomposition.wedge_size,
        )
        # Smooth random sector functions: U^T of enveloped noise.
        full = rng.standard_normal((grid.size, columns)) * envelope[:, None]
        host = np.column_stack(
            [decomposition.reduce_vector(full[:, column], representation) for column in range(columns)]
        )
        host /= np.linalg.norm(host, axis=0)[None, :]
        taken = min(remaining, max(1, columns - 1))
        remaining -= taken
        occupations[next_state : next_state + taken] = np.linspace(1.0, 0.6, taken)
        device = int(devices[representation % len(devices)])
        with cp.cuda.Device(device):
            vectors = cp.ascontiguousarray(cp.asarray(host))
        sectors.append(OrbitalSector(representation, device, vectors, np.arange(columns),
                                     np.arange(next_state, next_state + columns), expansion))
        next_state += columns
    return ChannelOrbitals(tuple(sectors), occupations, 2.0, int(grid.size), float(grid.volume_element)), reduction


class _Setting:
    """Run a block with ``PARSEC_CUPY_EXCHANGE_PARITY`` set, restoring it after."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __enter__(self):
        from parsec_python.acceleration.Hamiltonian.exchange_device import PARITY_SETTING

        self.name = PARITY_SETTING
        self.saved = os.environ.get(PARITY_SETTING)
        os.environ[PARITY_SETTING] = self.value

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.saved


def _applied(cp, orbitals, application):
    blocks = []
    for index, sector in enumerate(orbitals.sectors):
        with cp.cuda.Device(sector.device):
            blocks.append(cp.asnumpy(application.applied[index]))
    return blocks


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class ParityExchangeTests(unittest.TestCase):
    def compare(self, grid, molecule, terms, *, batch=4, devices=(0,), expect_parity=True, tolerance=1e-12):
        import cupy as cp

        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            apply_device_exchange,
            parity_context,
        )

        orbitals, reduction = sector_channel(grid, molecule, devices=devices)
        solvers = DevicePairSolvers(grid, terms, 6, batch)
        context = parity_context(orbitals, solvers)
        self.assertEqual(context is not None, expect_parity, molecule)
        results = {}
        for setting in ("off", "auto"):
            with _Setting(setting):
                occupied = apply_device_exchange(orbitals, solvers, devices, columns="occupied")
                application = apply_device_exchange(orbitals, solvers, devices, columns="all", known=occupied)
            results[setting] = (application.energy, _applied(cp, orbitals, application), application.pair_solves)
        (e0, w0, n0), (e1, w1, n1) = results["off"], results["auto"]
        self.assertEqual(n0, n1)
        self.assertAlmostEqual(e1, e0, delta=tolerance * abs(e0))
        scale = max(float(np.abs(block).max()) for block in w0)
        for a, b in zip(w0, w1):
            self.assertLess(float(np.abs(a - b).max()), 10 * tolerance * scale, molecule)
        return context, reduction

    def test_groups_and_grid_shifts(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings

        coulomb = (("coulomb", 1.0, None),)
        for shift in ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)):
            grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6, shift=shift))
            for molecule, cosets in (("D2h", 1), ("D2", 2), ("C2v", 2), ("Cs", 4), ("Ci", 4)):
                with self.subTest(shift=shift, molecule=molecule):
                    context, reduction = self.compare(grid, molecule, coulomb)
                    self.assertEqual(len(context.group.cosets), cosets)
                    self.assertEqual(len(context.group.operations) * cosets, 8)
            with self.subTest(shift=shift, molecule="signed"):
                from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction

                context, reduction = self.compare(grid, "signed", coulomb)
                # The signed permutations give the sectors; their diagonal subgroup the classes.
                self.assertIsInstance(reduction, SignedPermutationReduction)
                self.assertEqual(len(context.group.operations), 2)

    def test_screened_and_weighted_kernels(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        for terms in ((("erfc", 1.0, 0.11),), (("coulomb", 0.19, None), ("erf", 0.46, 0.33))):
            with self.subTest(terms=terms):
                self.compare(grid, "D2", terms, tolerance=1e-11)

    def test_densities_on_the_box_and_on_the_octant_points(self) -> None:
        # The class densities formed on the octant box (the default) or on R
        # and scattered by the solve: the same operator to round-off.
        import cupy as cp

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import DevicePairSolvers, apply_device_exchange
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6, shift=(0.0, 0.0, 0.0)))
        orbitals, _ = sector_channel(grid, "C2v")
        solvers = DevicePairSolvers(grid, (("erfc", 1.0, 0.11),), 6, 3)
        results = []
        for value in ("1", "0"):
            saved = os.environ.get("PARSEC_CUPY_EXCHANGE_PARITY_BOX")
            os.environ["PARSEC_CUPY_EXCHANGE_PARITY_BOX"] = value
            try:
                application = apply_device_exchange(orbitals, solvers, (0,), columns="all")
            finally:
                if saved is None:
                    os.environ.pop("PARSEC_CUPY_EXCHANGE_PARITY_BOX", None)
                else:
                    os.environ["PARSEC_CUPY_EXCHANGE_PARITY_BOX"] = saved
            results.append((application.energy, _applied(cp, orbitals, application)))
        (e1, w1), (e0, w0) = results
        self.assertAlmostEqual(e1, e0, delta=1e-12 * abs(e0))
        scale = max(float(np.abs(block).max()) for block in w0)
        for a, b in zip(w0, w1):
            self.assertLess(float(np.abs(a - b).max()), 1e-11 * scale)

    def test_plans_and_description_follow_the_path(self) -> None:
        # Class plans are prebuilt only for a run that will use them, and the
        # description names the path the builds took (review F3).
        import cupy as cp

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            apply_device_exchange,
            channel_orbitals,
        )
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        coulomb = (("coulomb", 1.0, None),)
        plain = DevicePairSolvers(grid, coulomb, 6, 4)
        self.assertFalse(plain.hosts[0].__dict__.get("_parity_plans"))
        self.assertNotIn("parity", plain.description)
        full = cp.asarray(np.random.default_rng(1).standard_normal((grid.size, 4)))
        apply_device_exchange(channel_orbitals(full, np.array([1.0, 1.0, 0.0, 0.0]), 2.0, grid.volume_element),
                              plain, (0,), columns="all")
        self.assertNotIn("parity", plain.description)
        orbitals, _ = sector_channel(grid, "D2")
        apply_device_exchange(orbitals, plain, (0,), columns="all")
        self.assertIn("reflection-parity classes", plain.description)
        self.assertIn("the others on the full grid", plain.description)
        planned = DevicePairSolvers(grid, coulomb, 6, 4, parity=True)
        self.assertEqual(len(planned.hosts[0].__dict__.get("_parity_plans", {})), 8)
        self.assertNotIn("parity", planned.description)  # no build yet
        with _Setting("off"):
            self.assertFalse(DevicePairSolvers(grid, coulomb, 6, 4, parity=True).parity)
        saved = os.environ.get("PARSEC_CUPY_EXCHANGE_PARITY_BOX")
        os.environ["PARSEC_CUPY_EXCHANGE_PARITY_BOX"] = "2"
        try:
            with self.assertRaisesRegex(ValueError, "PARSEC_CUPY_EXCHANGE_PARITY_BOX"):
                DevicePairSolvers(grid, coulomb, 6, 4)
        finally:
            if saved is None:
                os.environ.pop("PARSEC_CUPY_EXCHANGE_PARITY_BOX", None)
            else:
                os.environ["PARSEC_CUPY_EXCHANGE_PARITY_BOX"] = saved

    def test_paths_without_classes(self) -> None:
        import cupy as cp

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            channel_orbitals,
            parity_context,
        )
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))
        solvers = DevicePairSolvers(grid, (("coulomb", 1.0, None),), 6, 4)
        orbitals, _ = sector_channel(grid, "D2")
        with _Setting("off"):
            self.assertIsNone(parity_context(orbitals, solvers))
        with _Setting("sometimes"):
            with self.assertRaisesRegex(ValueError, "auto or off"):
                parity_context(orbitals, solvers)
        # Orbitals on the full grid keep the full-grid path.
        full = cp.asarray(np.random.default_rng(1).standard_normal((grid.size, 4)))
        self.assertIsNone(parity_context(channel_orbitals(full, np.ones(4), 2.0, grid.volume_element), solvers))
        # A box that is not symmetric about the origin has no classes.
        shifted = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6, shift=(0.5, 0.5, 0.25)))
        self.assertIsNone(DevicePairSolvers(shifted, (("coulomb", 1.0, None),), 6, 4).parity_layout())


class ParityMemoryPlanTests(unittest.TestCase):
    """The memory rules plan the parity path where it is taken, and it holds less than the full grid."""

    def test_planned_parity_and_bytes(self) -> None:
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            estimated_box_points,
            planned_exchange_bytes,
            planned_parity,
        )
        from parsec_python.acceleration.Symmetry.axis_reflection import SignedPermutationReduction
        from parsec_python.models import Atom, GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6))

        def reduction(molecule):
            atoms = [Atom(symbol, position) for symbol, position in MOLECULES[molecule]]
            return SignedPermutationReduction.detect(grid, atoms)

        plan = planned_parity(grid, ["coulomb"], 6, reduction("D2"))
        self.assertEqual(plan["cosets"], 2)
        self.assertEqual(plan["octant_points"] * 8, grid.size)  # shift 1/2: eight distinct images
        self.assertEqual(plan["box_points"], 8 * plan["octant_box"])
        self.assertEqual(planned_parity(grid, ["coulomb"], 6, reduction("signed"))["cosets"], 4)
        self.assertIsNone(planned_parity(grid, ["coulomb"], 6, None))
        with _Setting("off"):
            self.assertIsNone(planned_parity(grid, ["coulomb"], 6, reduction("D2")))
        shifted = build_cluster_grid(GridSettings(spacing=0.5, radius=4.0, expansion_order=6, shift=(0.5, 0.5, 0.25)))
        self.assertIsNone(planned_parity(shifted, ["coulomb"], 6, reduction("D2")))
        # Four sectors of a k04-like grid on four devices: parity holds less on every device.
        sizes = dict(full_size=1_099_136, box_points=128**3, kernels=["coulomb"], states=605, channels=1,
                     batch_size=32, devices=(0, 1, 2, 3), sector_rows={r: 274_784 for r in range(4)},
                     sector_devices={r: r for r in range(4)})
        full = planned_exchange_bytes(**sizes)
        parity = planned_exchange_bytes(
            **sizes, parity={"cosets": 2, "octant_points": 137_392, "octant_box": 64**3, "box_points": 128**3}
        )
        for device in range(4):
            self.assertLess(parity[device], full[device])
        self.assertGreater(estimated_box_points(grid, ["coulomb"], 6), 0)

    def test_four_classes_hold_no_more_than_two(self) -> None:
        # Groups of order 2 (Cs, Ci, C2) hold four classes per chunk: their
        # chunks are halved so that the parity path never needs more than the
        # full-grid path (review F2), in the plan as in the run.
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            parity_densities_per_solve,
            planned_exchange_bytes,
        )

        box, octant = 100**3, 50**3
        self.assertEqual(parity_densities_per_solve(32, box, octant, 1), 256)
        self.assertEqual(parity_densities_per_solve(32, box, octant, 2), 256)
        self.assertEqual(parity_densities_per_solve(32, box, octant, 4), 128)
        self.assertEqual(parity_densities_per_solve(1, 8, 8, 4), 1)
        # A Cs molecule: N 523,984, two sectors of 96 states.
        sizes = dict(full_size=523_984, box_points=box, kernels=["coulomb"], states=96, channels=1,
                     batch_size=32, devices=(0,), sector_rows={0: 261_992, 1: 261_992},
                     sector_devices={0: 0, 1: 0})
        full = planned_exchange_bytes(**sizes)[0]
        for cosets in (1, 2, 4):
            plan = {"cosets": cosets, "octant_points": 65_498, "octant_box": octant, "box_points": box}
            self.assertLess(planned_exchange_bytes(**sizes, parity=plan)[0], full, cosets)


@unittest.skipUnless(_device_count() > 1, "two or more CUDA devices are required")
class MultiDeviceParityTests(unittest.TestCase):
    def test_all_devices_give_the_one_device_operator(self) -> None:
        import cupy as cp

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import DevicePairSolvers, apply_device_exchange
        from parsec_python.models import GridSettings

        grid = build_cluster_grid(GridSettings(spacing=0.4, radius=6.0, expansion_order=8))
        devices = tuple(range(_device_count()))
        orbitals, _ = sector_channel(grid, "D2", states=60, occupied=40, devices=devices)
        solvers = DevicePairSolvers(grid, (("erfc", 1.0, 0.11),), 6, 8)
        one = apply_device_exchange(orbitals, solvers, (0,), columns="all")
        reference = _applied(cp, orbitals, one)
        scale = max(float(np.abs(block).max()) for block in reference)
        for trial in range(2):
            many = apply_device_exchange(orbitals, solvers, devices, columns="all")
            for a, b in zip(reference, _applied(cp, orbitals, many)):
                self.assertLess(float(np.abs(a - b).max()), 1e-12 * scale, trial)
            self.assertAlmostEqual(many.energy, one.energy, delta=1e-12 * abs(one.energy))


if __name__ == "__main__":
    unittest.main()
