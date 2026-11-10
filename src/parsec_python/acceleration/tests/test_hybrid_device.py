"""Self-consistent hybrid functionals on the GPU (``acceleration/SCF/hybrid.py``).

The device path must give what the reference Fock loop gives on the same
input: the exchange term in the device Hamiltonians (full grid and symmetry
sectors), ``W = V_x Psi`` and ACE formed on the devices, and the loop of
:func:`parsec_python.SCF.hybrid.run_hybrid_scf` with the device pieces.
H2 on a small sphere (R = 5 bohr, h = 0.5 bohr, order 6) keeps every case to
seconds.  The SCF of the sectors converges along another path than the
full-grid reference: total energies agree to 1e-9 Ry (second order in the
SCF residual), exchange energies and levels to 1e-6.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration.backends.cupy import cupy_available

LIBRARY = Path(__file__).resolve().parents[3] / "pp_generation"
HYDROGEN = LIBRARY / "NCPPs" / "PBE" / "H" / "PARSEC" / "H_POTRE.DAT"

BASE = """Boundary_Conditions: cluster
Cluster_Domain_Shape: sphere
Boundary_Sphere_Radius: 5.0 bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 6
Solver_Lpole: 4
Hartree_Atomic_Tail: off

States_Num: {states}
Fermi_Temp: 80
Max_Iter: 80
Convergence_Criterion: 1.0e-7 Ry
Eigensolver: chebff
Mixing_Method: Anderson
Mixing_Param: 0.3

Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
{species}begin Atom_Coord
{atoms}end Atom_Coord

Correlation_Type: {functional}
{extra}"""

H2 = "  0.0 0.0 -0.70\n  0.0 0.0  0.70\n"
TIGHT = "Exx_Tolerance: 1.0e-10 Ry\n"


class _Inputs:
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_hybrid_device_"))
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def problem(self, functional, *, states=3, atoms=H2, extra=TIGHT, species="", name="parsec.in"):
        path = self.directory / name
        path.write_text(
            BASE.format(states=states, atoms=atoms, functional=functional, extra=extra, species=species)
        )
        return parse_parsec_input(path).problem

    def both(self, problem, *, symmetry="auto"):
        from parsec_python.SCF import prepare_single_point as prepare_reference
        from parsec_python.SCF import run_scf as run_reference
        from parsec_python.acceleration.driver import prepare_single_point, run_scf

        reference = run_reference(prepare_reference(problem))
        system = prepare_single_point(problem, backend="cupy", symmetry=symmetry)
        records = []
        accelerated = run_scf(system, callback=records.append)
        return reference, system, accelerated, records


def _levels(result):
    channels = getattr(result, "spin", None)
    if channels is None:
        return np.asarray(result.eigenvalues)
    return np.concatenate([np.asarray(channel.eigenvalues) for channel in channels])


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceOperatorTermTests(_Inputs, unittest.TestCase):
    """The exchange term in the CuPy Hamiltonian and the grouped filter."""

    def setUp(self) -> None:
        super().setUp()
        import cupy as cp

        from parsec_python.acceleration.driver import prepare_single_point

        self.cp = cp
        system = prepare_single_point(self.problem("pbe", extra=""), backend="cupy", symmetry="off")
        self.operator = system.backend.ensure_device_operator()
        self.operator.update_local_potential(system.ionic_potential)
        rng = np.random.default_rng(7)
        rows = self.operator.shape[0]
        self.xi = cp.asarray(np.asfortranarray(rng.standard_normal((rows, 5)) / np.sqrt(rows)))
        self.block = cp.asarray(np.asfortranarray(rng.standard_normal((rows, 9))))
        self.previous = cp.asarray(np.asfortranarray(rng.standard_normal((rows, 9))))

    def tearDown(self) -> None:
        self.operator.bind_exchange(None)
        super().tearDown()

    def test_every_application_holds_the_term(self) -> None:
        cp = self.cp
        operator, xi, block = self.operator, self.xi, self.block
        fraction = 0.25
        term = -fraction * (xi @ (xi.T @ block))
        plain = operator.apply(block)
        recurrence = dict(center=0.3, scale=0.7, sigma_next=1.3, sigma=0.4)
        plain_step = operator.chebyshev_recurrence(block, previous=self.previous, **recurrence)
        operator.bind_exchange(xi, -fraction)
        try:
            scale = float(cp.abs(plain).max())
            self.assertLess(float(cp.abs(operator.apply(block) - plain - term).max()), 1e-12 * scale)
            target = cp.empty_like(block, order="F")
            operator.apply_into(block, target)
            self.assertLess(float(cp.abs(target - plain - term).max()), 1e-12 * scale)
            step = operator.chebyshev_recurrence(block, previous=self.previous, **recurrence)
            expected = plain_step + recurrence["sigma_next"] * recurrence["scale"] * term
            self.assertLess(float(cp.abs(step - expected).max()), 1e-12 * float(cp.abs(expected).max()))
            vector = operator.apply(block[:, 0])
            self.assertLess(float(cp.abs(vector - plain[:, 0] - term[:, 0]).max()), 1e-12 * scale)
            with self.assertRaisesRegex(RuntimeError, "mixed-precision|FP32"):
                operator.chebyshev_recurrence_float32(block.astype(cp.float32), center=0.0, scale=1.0)
        finally:
            operator.bind_exchange(None)
        self.assertEqual(float(cp.abs(operator.apply(block) - plain).max()), 0.0)

    def test_grouped_filter_is_the_block_filter(self) -> None:
        from parsec_python.acceleration.Eigensolvers.chebyshev import (
            chebff_filter,
            chebyshev_filter,
            subspace_filter,
            subspace_filter_blocks,
            uniform_filter_blocks,
        )

        cp = self.cp
        operator = self.operator
        operator.bind_exchange(self.xi, -0.25)
        block = self.block[:, :8]
        lower, upper, reference = -1.0, 40.0, -2.0

        def blockwise(blocks):
            out = cp.empty_like(block)
            carried = None
            for item in blocks:
                result, carried = chebyshev_filter(
                    operator,
                    block[:, item.start : item.stop],
                    item.degree,
                    lower,
                    upper,
                    reference,
                    initial_sigma=carried,
                    return_final_sigma=True,
                )
                out[:, item.start : item.stop] = result
            return out

        grouped = chebff_filter(operator, block, 5, lower, upper, reference, block_size=3)
        expected = blockwise(uniform_filter_blocks(8, 3, 5))
        self.assertLess(float(cp.abs(grouped - expected).max()), 1e-11 * float(cp.abs(expected).max()))
        grouped = subspace_filter(operator, block, 6, 2, lower, upper, block_size=2)
        reference = lower
        expected = blockwise(subspace_filter_blocks(8, 2, 6, 2))
        self.assertLess(float(cp.abs(grouped - expected).max()), 1e-11 * float(cp.abs(expected).max()))


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceExchangeTests(_Inputs, unittest.TestCase):
    """``W = V_x Psi`` and ACE on the devices against the host operator."""

    def test_full_grid_and_sector_operators_match_the_host(self) -> None:
        import cupy as cp

        from parsec_python.Hartree.kronecker import KroneckerPoisson
        from parsec_python.V_xc.exchange_operator import apply_exchange_to_orbitals, compress_exchange
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            apply_device_exchange,
            channel_orbitals,
            compress_device_exchange,
        )
        from parsec_python.acceleration.driver import prepare_single_point, run_scf

        for symmetry in ("off", "auto"):
            system = prepare_single_point(
                self.problem("pbe", states=6, extra=""), backend="cupy", symmetry=symmetry
            )
            # The orbitals stay where the eigensolver keeps them.
            system.materialize_final_wavefunctions = False
            result = run_scf(system).result
            # Fractional occupations: the operator of the ensemble.
            occupations = np.array([1.0, 0.7, 0.3, 0.0, 0.0, 0.0])
            orbitals = channel_orbitals(
                result.wavefunctions, occupations, 2.0, system.grid.volume_element
            )
            if symmetry == "auto":
                self.assertGreater(len(orbitals.sectors), 1)
            full = (
                result.wavefunctions.to_full_host()
                if hasattr(result.wavefunctions, "to_full_host")
                else cp.asnumpy(result.wavefunctions)
            )
            host_solver = KroneckerPoisson(system.grid, multipole_order=system.input.hartree.multipole_order)
            host = apply_exchange_to_orbitals(
                full, occupations, system.grid.volume_element, host_solver, batch_size=4
            )
            solvers = DevicePairSolvers(
                system.grid, (("coulomb", 1.0, None),), system.input.hartree.multipole_order, 2
            )
            occupied = apply_device_exchange(orbitals, solvers, (0,), columns="occupied")
            self.assertAlmostEqual(occupied.energy, host.energy, delta=1e-12 * abs(host.energy))
            application = apply_device_exchange(orbitals, solvers, (0,), columns="all", known=occupied)
            scale = float(np.max(np.abs(host.applied)))
            for index, sector in enumerate(orbitals.sectors):
                with cp.cuda.Device(sector.device):
                    applied = application.applied[index]
                    full_applied = (
                        applied if sector.expansion is None else sector.expansion.unfold(cp, applied)
                    )
                    difference = cp.asnumpy(full_applied) - host.applied[:, sector.states]
                self.assertLess(float(np.max(np.abs(difference))), 1e-11 * scale, symmetry)
            compressed = compress_device_exchange(application, fraction=0.25)
            reference = compress_exchange(full, host, fraction=0.25)
            self.assertEqual(compressed.rank, reference.rank)
            # V_ACE on the orbitals of the block: the same band energy.
            self.assertAlmostEqual(
                compressed.band_energy(orbitals),
                float(2.0 * np.dot(occupations, reference.expectation_values(full))),
                delta=1e-11,
            )


def _device_count() -> int:
    if not cupy_available():
        return 0
    import cupy as cp

    return int(cp.cuda.runtime.getDeviceCount())


@unittest.skipUnless(_device_count() > 1, "two or more CUDA devices are required")
class MultiDeviceExchangeTests(unittest.TestCase):
    """``W`` with the tiles on every device equals ``W`` on one device, for every basis layout.

    The eigensolvers hand the exchange C-ordered sector bases and strided
    full-grid blocks; a device that reads columns of a basis on another
    device gathers them there first (``OrbitalSector.columns``).  Several
    trials, so that a temporary reused before its copy finished would show.
    """

    def test_every_layout_gives_the_one_device_operator(self) -> None:
        import cupy as cp

        from parsec_python.Grid import build_cluster_grid
        from parsec_python.acceleration.Hamiltonian.exchange_device import (
            DevicePairSolvers,
            apply_device_exchange,
            channel_orbitals,
        )
        from parsec_python.models import GridSettings

        # Large enough that the copies of one device queue behind its solves
        # while another device gathers again (the race of a reused temporary).
        grid = build_cluster_grid(GridSettings(spacing=0.25, radius=12.0, expansion_order=8))
        count, occupied = 96, 72
        rng = np.random.default_rng(11)
        points = np.asarray(grid.coordinates)
        envelope = np.exp(-0.08 * np.sum(points**2, axis=1))
        host = rng.standard_normal((grid.size, 2 * count)) * envelope[:, None]
        host /= np.linalg.norm(host, axis=0)[None, :]
        occupations = np.concatenate([np.ones(occupied), np.zeros(count - occupied)])
        with cp.cuda.Device(0):
            wide = cp.ascontiguousarray(cp.asarray(host))
            layouts = {
                "C-ordered": cp.ascontiguousarray(wide[:, :count]),
                "strided": wide[:, ::2],
                "F-ordered": cp.asfortranarray(wide[:, :count]),
            }
        solvers = DevicePairSolvers(grid, (("coulomb", 1.0, None),), 4, 32)
        devices = tuple(range(_device_count()))
        for name, block in layouts.items():
            orbitals = channel_orbitals(block, occupations, 2.0, grid.volume_element)
            one = apply_device_exchange(orbitals, solvers, (0,), columns="all")
            with cp.cuda.Device(0):
                reference = cp.asnumpy(one.applied[0])
            scale = float(np.max(np.abs(reference)))
            for trial in range(3):
                many = apply_device_exchange(orbitals, solvers, devices, columns="all")
                with cp.cuda.Device(0):
                    values = cp.asnumpy(many.applied[0])
                self.assertLess(float(np.max(np.abs(values - reference))), 1e-12 * scale, (name, trial))
                self.assertAlmostEqual(many.energy, one.energy, delta=1e-12 * abs(one.energy))


@unittest.skipUnless(cupy_available(), "CuPy with a usable CUDA device is required")
class DeviceFockLoopTests(_Inputs, unittest.TestCase):
    """The accelerated Fock loop against the reference loop on the same input."""

    def check(self, reference, accelerated, *, energy=1e-9, levels=1e-6):
        self.assertTrue(reference.converged and accelerated.converged)
        self.assertAlmostEqual(accelerated.energies.total, reference.energies.total, delta=energy)
        self.assertAlmostEqual(
            accelerated.exact_exchange.exchange_energy,
            reference.exact_exchange.exchange_energy,
            delta=levels,
        )
        np.testing.assert_allclose(_levels(accelerated), _levels(reference), atol=levels)
        self.assertEqual(
            len(accelerated.exact_exchange.iterations), len(reference.exact_exchange.iterations)
        )

    def test_pbe0_full_grid_and_sectors(self) -> None:
        from parsec_python.models import ExchangeLoopIteration, HybridEnergyBreakdown

        problem = self.problem("pbe0")
        reference, _system, full, _ = self.both(problem, symmetry="off")
        # The full grid takes the reference path's arithmetic to round-off.
        self.check(reference, full, energy=1e-12, levels=1e-12)
        _, system, sectors, records = self.both(problem)
        self.assertIsNotNone(system.exchange_device.sectors)
        # The memory rules of the sectors count the exchange's device bytes.
        resident = system.exchange_device.sectors.resident_device_bytes
        for device, count in system.exchange_device.planned_bytes.items():
            self.assertGreater(count, 0)
            self.assertGreaterEqual(resident[device], count)
        from types import SimpleNamespace

        from parsec_python.acceleration.Hamiltonian.exchange_device import require_exchange_memory

        huge = SimpleNamespace(
            sectors=(SimpleNamespace(device=0, rows=10**12, local=np.arange(8)),), full_size=10
        )
        with self.assertRaisesRegex(MemoryError, "needs .* GiB on device 0"):
            require_exchange_memory(huge, SimpleNamespace(batch_size=8), (0,))
        self.check(reference, sectors)
        outer = [record for record in records if isinstance(record, ExchangeLoopIteration)]
        self.assertEqual(outer[0].start_functional, "pbe")
        self.assertTrue(outer[-1].converged)
        self.assertIsInstance(sectors.energies, HybridEnergyBreakdown)
        self.assertFalse(sectors.energies.linearized)
        self.assertIn("FP64 CUDA Kronecker", sectors.exact_exchange.pair_solver)
        # No exchange term is left in the device Hamiltonians after the loop.
        for operator in system.exchange_device.operators().values():
            self.assertIsNone(operator.exchange_projectors)

    def test_spin_polarized_atom(self) -> None:
        # Both channels on the full grid take the reference path's arithmetic
        # to round-off; in the symmetry sectors (the default of a hybrid,
        # Spin_Symmetry_Sectors: auto) the loop takes another SCF path, and
        # E_x and the levels agree to first order in the SCF residual.
        for sectors, tolerances in (("off", dict(energy=1e-12, levels=1e-12)), ("auto", {})):
            problem = self.problem(
                "pbe0",
                states=2,
                atoms="  0.0 0.0 0.0\n",
                species="Initial_Spin_Polarization: 1.0\n",
                extra="Spin_Polarization: .true.\n" + TIGHT + f"Spin_Symmetry_Sectors: {sectors}\n",
            )
            reference, system, accelerated, _ = self.both(problem)
            self.assertEqual(getattr(system.backend, "symmetry_eigensolver", None) is not None, sectors == "auto")
            self.check(reference, accelerated, **tolerances)
            up, down = accelerated.spin
            self.assertAlmostEqual(float(np.sum(up.occupations)), 1.0, places=8)

    def test_screened_and_two_kernel_hybrids(self) -> None:
        for functional in ("hse06", "cam-b3lyp"):
            reference, _, accelerated, _ = self.both(self.problem(functional, name=f"{functional}.in"))
            self.check(reference, accelerated)

    def test_forces(self) -> None:
        from parsec_python.Forces import compute_forces as reference_forces
        from parsec_python.SCF import prepare_single_point as prepare_reference
        from parsec_python.SCF import run_scf as run_reference
        from parsec_python.acceleration.driver import compute_forces, prepare_single_point, run_scf

        problem = self.problem("pbe0", atoms="  0.0 0.0 -0.60\n  0.0 0.0  0.80\n")
        system = prepare_reference(problem)
        expected = reference_forces(system, run_reference(system)).total
        accelerated_system = prepare_single_point(problem, backend="cupy")
        forces = compute_forces(accelerated_system, run_scf(accelerated_system)).total
        self.assertGreater(float(np.max(np.abs(expected))), 1e-2)
        np.testing.assert_allclose(forces, expected, atol=2e-6)


class DeviceRoutingTests(_Inputs, unittest.TestCase):
    def test_routes_and_refusals(self) -> None:
        from types import SimpleNamespace

        from parsec_python.acceleration import cli as accelerated_cli
        from parsec_python.acceleration import driver as accelerated_driver

        problem = self.problem("pbe0")
        with self.assertRaisesRegex(NotImplementedError, "MPI"):
            accelerated_driver.prepare_single_point(problem, mpi_context=object())
        with self.assertRaisesRegex(NotImplementedError, "CuPy backend"):
            accelerated_driver.prepare_single_point(problem, backend="scipy")
        with self.assertRaisesRegex(NotImplementedError, "Kronecker"):
            accelerated_driver.prepare_single_point(
                replace(
                    problem,
                    scf=replace(
                        problem.scf, exact_exchange=replace(problem.scf.exact_exchange, pair_solver="cg")
                    ),
                )
            )
        translation = SimpleNamespace(problem=problem)
        plain = SimpleNamespace(problem=self.problem("pbe", extra="", name="plain.in"))
        arguments = SimpleNamespace(backend="scipy", mpi=False)
        self.assertFalse(accelerated_cli._hybrid_on_device(arguments, translation))
        self.assertFalse(accelerated_cli._hybrid_on_device(arguments, plain))
        if cupy_available():
            arguments = SimpleNamespace(backend="cupy", mpi=False)
            self.assertTrue(accelerated_cli._hybrid_on_device(arguments, translation))
            self.assertFalse(
                accelerated_cli._hybrid_on_device(SimpleNamespace(backend="cupy", mpi=True), translation)
            )
            # Exx_Pair_Solver: cg goes to the reference solver.
            cg = SimpleNamespace(
                problem=self.problem("pbe0", extra=TIGHT + "Exx_Pair_Solver: cg\n", name="cg.in")
            )
            self.assertFalse(accelerated_cli._hybrid_on_device(arguments, cg))
            self.assertTrue(
                any("hybrid" in item for item in accelerated_cli._reference_only_features(
                    cg, accelerated_backend_selected=False,
                    hybrid_on_device=accelerated_cli._hybrid_on_device(arguments, cg),
                ))
            )

    def test_host_start_functional_reads_the_present_structure(self) -> None:
        """After an ionic move the start functional reads the moved structure (its core density)."""

        from dataclasses import dataclass
        from types import SimpleNamespace

        from parsec_python.V_xc import XCResult
        from parsec_python.acceleration.driver import _start_xc_with_reducer

        problem = self.problem("pbe0")

        @dataclass(frozen=True)
        class Structure:
            input: object
            core: float

            def evaluate_xc(self, density):
                assert self.input.scf.xc_functional == "pbe"
                return XCResult(
                    potential=density * 0.0 + self.core,
                    energy_per_electron=density * 0.0,
                    energy_density=density * 0.0,
                    total_energy=self.core,
                )

        identity = SimpleNamespace(to_full=lambda values: values, from_full=lambda values: values)
        present = [Structure(problem, 1.0)]
        evaluator, spin = _start_xc_with_reducer((None, None), lambda: present[0], identity)
        self.assertIsNone(spin)
        self.assertEqual(evaluator(np.ones(3)).total_energy, 1.0)
        present[0] = Structure(problem, 2.0)
        self.assertEqual(evaluator(np.ones(3)).total_energy, 2.0)


if __name__ == "__main__":
    unittest.main()
