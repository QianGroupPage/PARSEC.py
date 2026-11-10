"""Self-consistent hybrids of periodic cells on the GPU (``acceleration/SCF/hybrid_periodic.py``).

The device engine must give the host engine's Fock loop: H2 in a 6-bohr box
(10**3 points) with PBE0, HSE and B3LYP on time-reversal-reduced meshes,
unpolarized and spin-polarized.  To compare the exchange alone, the host
engine runs here with the same device k-point workers (its host ``xi``
uploaded into the device Bloch operators): both loops then take the same
path, and their energies agree to 1e-9 Ry (round-off of the pair solves,
amplified by the SCF).  The host engine with the host eigensolver (the
reference path) converges along another path: 1e-7 Ry.  The device loop of
real orbitals at the Gamma point (the accelerated driver) is checked against
the reference Gamma loop.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from parsec_python.acceleration.backends.cupy import cupy_available

LIBRARY = Path(__file__).resolve().parents[3] / "pp_generation"
HYDROGEN = LIBRARY / "NCPPs" / "PBE" / "H" / "PARSEC" / "H_POTRE.DAT"


def _h2_problem(functional="pbe0", spin=False, mesh=None, outer_mixing="diis"):
    from parsec_python.models import (
        Atom,
        EigensolverSettings,
        ExactExchangeSettings,
        HartreeSettings,
        PeriodicCell,
        PeriodicGridSettings,
        SCFSettings,
        SinglePointInput,
        SpeciesPotential,
    )

    return SinglePointInput(
        atoms=[Atom("H", [3.0, 3.0, 2.3]), Atom("H", [3.0, 3.0, 3.7])],
        pseudopotentials={"H": SpeciesPotential(HYDROGEN, 0)},
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=6.0 * np.eye(3)),
        scf=SCFSettings(
            xc_functional=functional,
            exact_exchange=ExactExchangeSettings(tolerance=1.0e-9, outer_mixing=outer_mixing),
            max_iterations=80,
            number_of_states=4,
            convergence_criterion=1.0e-8,
            spin_polarized=spin,
        ),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-10),
        **({} if mesh is None else {"monkhorst_pack_dimensions": tuple(mesh)}),
    )


_SLAB = """Boundary_Conditions: slab
Periodic_System: .true.
begin Cell_Shape
  4.0 0.0 0.0
  0.0 4.2 0.0
end Cell_Shape
Boundary_Sphere_Radius: 4.0
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.5
Expansion_Order: 6
States_Num: 3
Convergence_Criterion: 1.0e-8 Ry
Diag_Tolerance: 1.0e-10
Max_Iter: 80
Exx_Tolerance: 1.0e-9 Ry
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  0.3 0.1 0.70
 -0.2 0.0 -0.70
end Atom_Coord
Correlation_Type: pbe0
{extra}"""


def _slab_translation(extra="", text=None):
    """The translation of H2 in a slab (the Fortran-derived confined grid, the kernel on the box padded along z)."""

    import shutil
    import tempfile

    from parsec_python.Input import parse_parsec_input

    # The problem reads its potential from the folder later: kept until the module's tests end.
    folder = tempfile.mkdtemp(prefix="parsec_slab_hybrid_")
    _FOLDERS.append(folder)
    shutil.copy(HYDROGEN, Path(folder) / "H_POTRE.DAT")
    path = Path(folder) / "parsec.in"
    path.write_text((text or _SLAB).format(extra=extra), encoding="utf-8")
    return parse_parsec_input(path)


def _slab_problem(extra=""):
    return _slab_translation(extra).problem


_FOLDERS: list = []


def tearDownModule() -> None:
    import shutil

    for folder in _FOLDERS:
        shutil.rmtree(folder, ignore_errors=True)


def _mesh(problem):
    from parsec_python.Grid.kpoints import kpoint_sampling

    sampling = kpoint_sampling(problem)
    if sampling is None:
        return np.zeros((1, 3)), np.ones(1)
    return sampling


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class DeviceBlochEngineTests(unittest.TestCase):
    """The Fock loop with k-points: device engine against the host engine on the same device workers."""

    def _loops(self, problem):
        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.SCF import hybrid_periodic
        from parsec_python.SCF.hybrid import hybrid_definition, run_hybrid_scf
        from parsec_python.acceleration.SCF.hybrid_periodic import DeviceBlochExchange, device_bloch_engine

        system = prepare_periodic_single_point(problem)
        self.system = system
        k_points, k_weights = _mesh(problem)
        definition = hybrid_definition(system.input.scf)
        engine = device_bloch_engine(system, definition, k_points, k_weights)
        self.assertIsInstance(engine, DeviceBlochExchange)
        device = engine.finish(run_hybrid_scf(system, engine=engine))

        class HostOnDeviceWorkers(hybrid_periodic.BlochExchange):
            kpoint_backend = None

        host = run_hybrid_scf(system, engine=HostOnDeviceWorkers(system, definition, k_points, k_weights))
        return device, host

    def _compare(self, device, host, *, energy=1.0e-9):
        self.assertTrue(device.converged and host.converged)
        self.assertAlmostEqual(device.energies.total, host.energies.total, delta=energy)
        self.assertAlmostEqual(
            device.exact_exchange.exchange_energy, host.exact_exchange.exchange_energy, delta=energy
        )
        self.assertEqual(len(device.exact_exchange.iterations), len(host.exact_exchange.iterations))
        for ours, theirs in zip(device.exact_exchange.iterations, host.exact_exchange.iterations):
            # the device solves each pair of two occupied states once for both
            self.assertLessEqual(ours.pair_solves, theirs.pair_solves)
        # the result holds host orbitals (forces and output read them)
        blocks = device.k_wavefunctions
        if hasattr(device, "occupations_up"):
            blocks = [block for channel in blocks for block in channel]
        self.assertTrue(all(isinstance(block, np.ndarray) for block in blocks))
        print(
            f"\n  device - host: total {device.energies.total - host.energies.total:.2e} Ry, "
            f"E_x {device.exact_exchange.exchange_energy - host.exact_exchange.exchange_energy:.2e} Ry, "
            f"{len(device.exact_exchange.iterations)} outer iterations"
        )

    def test_unpolarized_mesh_with_partners(self) -> None:
        device, host = self._loops(_h2_problem(mesh=(3, 1, 1)))
        self._compare(device, host)
        np.testing.assert_allclose(device.eigenvalues, host.eigenvalues, rtol=0, atol=1e-8)

    def test_screened_kernel_without_outer_mixing(self) -> None:
        device, host = self._loops(_h2_problem("hse06", mesh=(2, 1, 1), outer_mixing="none"))
        self._compare(device, host)

    def test_spin_polarized_gamma_on_bloch_factors(self) -> None:
        device, host = self._loops(_h2_problem("b3lyp", spin=True))
        self._compare(device, host)
        self.assertAlmostEqual(device.magnetic_moment, 0.0, delta=1e-6)

    def test_spin_polarized_mesh(self) -> None:
        """Tasks (spin, k) dealt to the workers of the GPUs: each channel's operators where its k-points are."""

        device, host = self._loops(_h2_problem("pbe0", spin=True, mesh=(3, 1, 1), outer_mixing="none"))
        self._compare(device, host)

    def test_slab_with_in_plane_mesh(self) -> None:
        mesh = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 2 1\nend Monkhorst_Pack_Grid\n"
        device, host = self._loops(_slab_problem(mesh))
        self._compare(device, host)
        # The forces of the Fortran-derived slab read the Bloch factors each Fock loop keeps (Forces.wire_slab).
        from parsec_python.Forces import compute_forces

        ours = np.asarray(compute_forces(self.system, device).total)
        theirs = np.asarray(compute_forces(self.system, host).total)
        print(f"  slab forces device - host: {float(np.max(np.abs(ours - theirs))):.2e} Ry/bohr")
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1.0e-8)

    def test_next_structure_starts_from_the_carried_orbitals(self) -> None:
        """An ionic step: the host orbitals of the structure before build the first operator on the workers' GPUs."""

        from dataclasses import replace

        from parsec_python.Relaxation.backend import periodic_backend

        problem = _h2_problem(mesh=(3, 1, 1))
        k_points, k_weights = _mesh(problem)
        backend = periodic_backend(k_points, k_weights)
        from parsec_python.SCF import prepare_periodic_single_point

        system = prepare_periodic_single_point(problem)
        result = backend.run_scf(system)
        self.assertIn("cuFFT", result.exact_exchange.pair_solver)
        atoms = list(system.atoms)
        atoms[1] = replace(atoms[1], position=np.asarray(atoms[1].position) + np.array([0.0, 0.0, 0.05]))
        moved = backend.update_geometry(system, atoms)
        start = backend.warm_start(system, result, moved)
        self.assertIsNotNone(start.exchange_orbitals)
        records = []
        again = backend.run_scf(moved, warm_start=start, callback=records.append)
        cold = backend.run_scf(moved)
        self.assertTrue(again.converged and cold.converged)
        outer = [record for record in records if type(record).__name__ == "ExchangeLoopIteration"]
        self.assertIsNone(outer[0].start_functional)  # no semilocal start
        self.assertAlmostEqual(again.energies.total, cold.energies.total, delta=1.0e-7)

    def test_reference_path_agrees(self) -> None:
        """The host engine on the host eigensolver (the reference solver) reaches the same energy."""

        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf

        problem = _h2_problem(mesh=(3, 1, 1))
        system = prepare_periodic_single_point(problem)
        k_points, k_weights = _mesh(problem)
        device = run_periodic_hybrid_scf(system, k_points, k_weights)  # the installed device engine
        self.assertIn("cuFFT", device.exact_exchange.pair_solver)
        from parsec_python.SCF import hybrid_periodic

        installed = hybrid_periodic._device_engine
        hybrid_periodic.set_device_engine(None)
        try:
            host = run_periodic_hybrid_scf(system, k_points, k_weights)
        finally:
            hybrid_periodic.set_device_engine(installed)
        self.assertNotIn("cuFFT", host.exact_exchange.pair_solver)
        self.assertTrue(device.converged and host.converged)
        self.assertAlmostEqual(device.energies.total, host.energies.total, delta=1.0e-7)
        self.assertAlmostEqual(
            device.exact_exchange.exchange_energy, host.exact_exchange.exchange_energy, delta=1.0e-7
        )


def _antiferromagnetic_problem():
    """Stretched H2 in a box, the two atoms as species of opposite Initial_Spin_Polarization (a broken-symmetry
    state, which a start without local moments does not reach)."""

    from parsec_python.models import (
        Atom,
        EigensolverSettings,
        ExactExchangeSettings,
        HartreeSettings,
        PeriodicCell,
        PeriodicGridSettings,
        SCFSettings,
        SinglePointInput,
        SpeciesPotential,
    )

    return SinglePointInput(
        atoms=[Atom("H-up", [3.0, 3.0, 3.0]), Atom("H-dn", [9.0, 3.0, 3.0])],
        pseudopotentials={
            "H-up": SpeciesPotential(HYDROGEN, 0, element_symbol="H", initial_spin_polarization=0.8),
            "H-dn": SpeciesPotential(HYDROGEN, 0, element_symbol="H", initial_spin_polarization=-0.8),
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=np.diag([12.0, 6.0, 6.0])),
        scf=SCFSettings(
            xc_functional="pbe0",
            exact_exchange=ExactExchangeSettings(tolerance=1.0e-8),
            max_iterations=120,
            number_of_states=4,
            convergence_criterion=1.0e-7,
            spin_polarized=True,
        ),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-10),
    )


def _left_moment(system, result) -> float:
    """``integral (n_up - n_down)`` over the left half of the cell (around the first atom)."""

    channels = getattr(result, "spin", None)
    if channels is not None:
        up, down = (np.asarray(channel.density, dtype=float) for channel in channels)
    else:
        up, down = np.asarray(result.density_up, dtype=float), np.asarray(result.density_down, dtype=float)
    left = np.asarray(system.grid.coordinates)[:, 0] % 12.0 < 6.0
    return float(np.sum((up - down)[left]) * system.grid.volume_element)


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class GammaSpinStartTests(unittest.TestCase):
    """A spin-polarized hybrid at the Gamma point: the 1x1x1 Bloch loop of the GPUs starts where the real loop of
    the host does (the per-species spin densities of initchrg.f90), so both reach the same state."""

    def test_bloch_loop_and_real_loop_reach_one_broken_symmetry_state(self) -> None:
        from dataclasses import replace

        from parsec_python.SCF import prepare_periodic_single_point, run_scf
        from parsec_python.SCF.hybrid_periodic import run_periodic_hybrid_scf
        from parsec_python.SCF.kpoints import run_scf_kpoints_spin_polarized
        from parsec_python.SCF.pbc import with_initial_spin_densities

        prepared = prepare_periodic_single_point(_antiferromagnetic_problem())
        system = with_initial_spin_densities(prepared)
        real = run_scf(system)  # the real loop of the host (CPU route)
        # the Bloch loop of the GPU route, from a system as the k-point drivers prepare it
        bloch = run_periodic_hybrid_scf(prepared, np.zeros((1, 3)), np.ones(1))
        self.assertTrue(real.converged and bloch.converged)
        self.assertIn("cuFFT", bloch.exact_exchange.pair_solver)
        moments = (_left_moment(system, real), _left_moment(system, bloch))
        print(f"\n  AFM H2: real - Bloch {real.energies.total - bloch.energies.total:.2e} Ry, left moments {moments}")
        self.assertGreater(moments[0], 0.3)
        self.assertAlmostEqual(moments[1], moments[0], delta=1e-4)
        self.assertAlmostEqual(bloch.energies.total, real.energies.total, delta=1.0e-6)
        # Why the Fock loop passes that start: the k-point driver's own start (one polarization fraction of the
        # whole cell, zero here) has no local moments, and its loop stays in the symmetric state.
        plain = run_scf_kpoints_spin_polarized(
            replace(prepared, initial_spin_densities=None), np.zeros((1, 3)), np.ones(1)
        )
        self.assertLess(abs(_left_moment(system, plain)), 1e-3)


@unittest.skipUnless(cupy_available(), "CuPy and a CUDA device are required")
class DeviceGammaLoopTests(unittest.TestCase):
    """The Gamma point (real orbitals): the accelerated driver's device loop with the cuFFT pair solver."""

    def test_accelerated_gamma_loop_matches_the_reference_loop(self) -> None:
        from parsec_python.SCF import prepare_periodic_single_point, run_scf as reference_run_scf
        from parsec_python.acceleration.driver import prepare_single_point, run_scf

        from parsec_python.Forces.total import compute_periodic_forces
        from parsec_python.acceleration.driver import compute_forces

        problem = _h2_problem()
        system = prepare_single_point(problem, backend="cupy")
        self.assertIsNotNone(getattr(system, "exchange_device", None))
        device = run_scf(system)
        reference_system = prepare_periodic_single_point(problem)
        reference = reference_run_scf(reference_system)
        self.assertTrue(device.converged and reference.converged)
        self.assertIn("cuFFT", device.exact_exchange.pair_solver)
        # Forces of the hybrid orbitals and density (no exchange term with norm-conserving potentials).
        device_forces = np.asarray(compute_forces(system, device, symmetrize=False).total, dtype=float)
        reference_forces = np.asarray(compute_periodic_forces(reference_system, reference).total, dtype=float)
        print(
            f"\n  Gamma device - reference: total {device.energies.total - reference.energies.total:.2e} Ry, "
            f"E_x {device.exact_exchange.exchange_energy - reference.exact_exchange.exchange_energy:.2e} Ry, "
            f"forces {float(np.abs(device_forces - reference_forces).max()):.2e} Ry/bohr"
        )
        # Different eigensolvers (device full-grid CHEBFF and the readable one) reach the same fixed point.
        self.assertAlmostEqual(device.energies.total, reference.energies.total, delta=1.0e-7)
        self.assertAlmostEqual(
            device.exact_exchange.exchange_energy, reference.exact_exchange.exchange_energy, delta=1.0e-7
        )
        np.testing.assert_allclose(device_forces, reference_forces, rtol=0, atol=1.0e-6)
        self.assertGreater(float(np.abs(reference_forces).max()), 1.0e-3)  # the bond is not at equilibrium

    def test_accelerated_gamma_slab_matches_the_reference_loop(self) -> None:
        from parsec_python.SCF import prepare_periodic_single_point, run_scf as reference_run_scf
        from parsec_python.acceleration.driver import prepare_single_point, run_scf

        problem = _slab_problem()
        device = run_scf(prepare_single_point(problem, backend="cupy"))
        reference = reference_run_scf(prepare_periodic_single_point(problem))
        self.assertTrue(device.converged and reference.converged)
        self.assertIn("cuFFT", device.exact_exchange.pair_solver)
        self.assertIn("zero-padded", device.exact_exchange.pair_solver)
        print(
            f"\n  slab Gamma device - reference: total {device.energies.total - reference.energies.total:.2e} Ry, "
            f"E_x {device.exact_exchange.exchange_energy - reference.exact_exchange.exchange_energy:.2e} Ry"
        )
        self.assertAlmostEqual(device.energies.total, reference.energies.total, delta=1.0e-7)
        self.assertAlmostEqual(
            device.exact_exchange.exchange_energy, reference.exact_exchange.exchange_energy, delta=1.0e-7
        )


_BULK = """Boundary_Conditions: bulk
Periodic_System: .true.
begin Cell_Shape
  6.0 0.0 0.0
  0.0 6.0 0.0
  0.0 0.0 6.0
end Cell_Shape
Lattice_Vector_Scale: 1.0 bohr
Coordinate_Unit: Cartesian_Bohr
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 3
Atom_Types_Num: 1
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  3.0 3.0 2.3
  3.0 3.0 3.7
end Atom_Coord
Correlation_Type: pbe0
{extra}"""


class PeriodicHybridRoutingTests(unittest.TestCase):
    """Which device route ``main.py`` gives a periodic hybrid (``acceleration/cli.py::_hybrid_on_device``)."""

    def test_routes(self) -> None:
        from argparse import Namespace

        from parsec_python.acceleration import cli

        arguments = Namespace(backend="cupy", mpi=False)
        mesh = "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n"
        on_device = cupy_available()
        # the Gamma point without spin: the accelerated driver's device loop (bulk and slab)
        self.assertEqual(cli._hybrid_on_device(arguments, _slab_translation(text=_BULK)), on_device)
        self.assertEqual(cli._hybrid_on_device(arguments, _slab_translation()), on_device)
        # k-points, and spin: the reference solver's Bloch loop (its k-point workers and exchange on the GPUs)
        self.assertFalse(cli._hybrid_on_device(arguments, _slab_translation(mesh, text=_BULK)))
        self.assertFalse(cli._hybrid_on_device(arguments, _slab_translation("Spin_Polarization: .true.\n", text=_BULK)))
        self.assertFalse(cli._hybrid_on_device(Namespace(backend="cupy", mpi=True), _slab_translation(text=_BULK)))


if __name__ == "__main__":
    unittest.main()
