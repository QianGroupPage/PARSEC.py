"""Spin-polarized calculations on the accelerated backends.

The accelerated driver runs the SCF loop of
:mod:`parsec_python.SCF.spin_polarized` with its own Hamiltonian
application, Hartree solver and, on a GPU, eigensolver and density
construction.  Complete calculations are compared with the ``parsec.out`` of
Fortran PARSEC for the cases of ``tests/data/spin_parsec_reference.json``:
energies, the levels and occupations of each spin channel and the force
tables.  The two spin channels are solved on the full grid; what cannot do
that refuses.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Input import parse_parsec_input
from parsec_python.acceleration import driver
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.acceleration.cli import main as cli_main
from parsec_python.acceleration.refinement import refine_energy
from parsec_python.models import Atom
from parsec_python.tests.spin_reference import (
    assert_forces_match,
    assert_matches_fortran,
    load_cases,
    write_case,
)


def _solve(name: str, backend: str, *, symmetry: str = "auto", extra=()):
    """Prepare and solve one case of the fixture on ``backend``."""

    case = load_cases()[name]
    with tempfile.TemporaryDirectory() as directory:
        translation = parse_parsec_input(
            write_case(case, Path(directory), extra=tuple(extra))
        )
        system = driver.prepare_single_point(
            translation.problem, backend=backend, symmetry=symmetry
        )
    system.materialize_final_wavefunctions = True
    return case, system, driver.run_scf(system)


def _solve_hydrogen(lines: list[str], spin: bool, backend: str):
    """Solve a hydrogen system of the PBE library on the full grid."""

    case = dict(
        input=lines + [f"Spin_Polarization: {'true' if spin else 'false'}"],
        pseudopotentials={"H_POTRE.DAT": "NCPPs/PBE/H/PARSEC/H_POTRE.DAT"},
    )
    with tempfile.TemporaryDirectory() as directory:
        translation = parse_parsec_input(write_case(case, Path(directory)))
        system = driver.prepare_single_point(
            translation.problem, backend=backend, symmetry="off"
        )
    system.materialize_final_wavefunctions = True
    return system, driver.run_scf(system)


class _BackendCases:
    """Fortran comparisons every backend must pass."""

    backend: str

    def assert_case(self, name: str, *, forces: bool = False, **tolerances):
        case, system, result = _solve(name, self.backend)
        self.assertEqual(result.backend.selected, self.backend)
        assert_matches_fortran(self, result, case, **tolerances)
        if forces:
            assert_forces_match(self, driver.compute_forces(system, result), case)
        return case, system, result

    def test_methyl_radical_with_forces(self) -> None:
        _, _, result = self.assert_case("ch3_ca", forces=True)
        self.assertAlmostEqual(abs(result.net_magnetic_moment), 1.0, places=10)
        states = result.spin[0].eigenvalues.size
        self.assertIsInstance(result.wavefunctions, np.ndarray)
        self.assertEqual(result.wavefunctions.shape[1], 2 * states)
        self.assertTrue(
            np.shares_memory(result.spin[1].wavefunctions, result.wavefunctions)
        )

    def test_nitrogen_atom(self) -> None:
        _, _, result = self.assert_case("n_ca")
        self.assertAlmostEqual(abs(result.net_magnetic_moment), 3.0, places=10)

    def test_density_that_breaks_the_symmetry_of_the_structure(self) -> None:
        """The oxygen atom at 80 K: the fourth p electron takes one orbital.

        Its density is not invariant under the mirror planes of the atom, so
        the Hartree potential must be that of the full grid: the symmetry
        wedge of the Hartree solver would average the density over the orbits.
        """

        case = dict(load_cases()["o_spherical_ca"])
        case["input"] = [
            line for line in case["input"] if not line.startswith("Fermi_Temp")
        ]
        del case["occup"]
        totals = {}
        for symmetry in ("auto", "off"):
            with tempfile.TemporaryDirectory() as directory:
                translation = parse_parsec_input(write_case(case, Path(directory)))
                system = driver.prepare_single_point(
                    translation.problem, backend=self.backend, symmetry=symmetry
                )
            result = driver.run_scf(system)
            self.assertTrue(result.converged)
            self.assertAlmostEqual(abs(result.net_magnetic_moment), 2.0, places=6)
            # One of the three down p levels is filled and lies below the others.
            down = result.spin[1] if result.net_magnetic_moment > 0 else result.spin[0]
            self.assertGreater(down.eigenvalues[2] - down.eigenvalues[1], 1.0e-2)
            self.assertTrue(
                dict(result.backend.details)
                .get("hartree_symmetry", "full grid")
                .startswith("full grid")
            )
            totals[symmetry] = result.energies.total
        self.assertAlmostEqual(totals["auto"], totals["off"], delta=1.0e-7)


class SciPyBackendTests(_BackendCases, unittest.TestCase):
    backend = "scipy"

    def test_oxygen_molecule_with_forces(self) -> None:
        _, _, result = self.assert_case("o2_ca", forces=True)
        self.assertAlmostEqual(abs(result.net_magnetic_moment), 2.0, places=10)

    def test_core_corrected_radical_with_forces(self) -> None:
        """GeH3: half of the core density of Ge enters each spin channel."""

        _, system, result = self.assert_case("geh3_nlcc_ca", forces=True)
        self.assertGreater(float(np.max(system.core_density)), 0.1)
        self.assertAlmostEqual(abs(result.net_magnetic_moment), 1.0, places=10)
        forces = driver.compute_forces(system, result)
        self.assertGreater(float(np.max(np.abs(forces.local.core_correction))), 1.0e-2)

    def test_oxygen_atom_with_fractional_fixed_occupations(self) -> None:
        _, _, result = self.assert_case("o_spherical_ca")
        np.testing.assert_allclose(
            result.spin[1].occupations[1:4], [1.0 / 3.0] * 3, atol=1.0e-12
        )
        # The spread electron keeps the three p levels of each channel
        # degenerate, as the grid does.
        for channel in result.spin:
            self.assertLess(channel.eigenvalues[3] - channel.eigenvalues[1], 1.0e-7)

    def test_closed_shell_water_equals_the_unpolarized_run(self) -> None:
        case, _, result = self.assert_case("h2o_spin_ca", forces=True)
        unpolarized = load_cases()["h2o_nospin_ca"]
        self.assertEqual(unpolarized["energy"], case["energy"])
        self.assertAlmostEqual(result.net_magnetic_moment, 0.0, delta=1.0e-6)
        for channel in result.spin:
            np.testing.assert_allclose(
                channel.eigenvalues, unpolarized["eigenvalues"], rtol=0, atol=2.0e-7
            )
        # This package without spin polarization gives the same state.
        _, _, plain = _solve("h2o_nospin_ca", self.backend)
        self.assertIsNone(plain.spin)
        self.assertAlmostEqual(result.energies.total, plain.energies.total, delta=2.0e-8)
        np.testing.assert_allclose(
            2.0 * result.spin[0].density, plain.density, rtol=0, atol=1.0e-6
        )

    def test_hydrogen_atom_pbe_within_the_pbe_difference(self) -> None:
        """PBE: the discrete potential of this package against Fortran's.

        On this grid the unpolarized PBE of this package differs from
        Fortran by as much (``V_xc/pbe_spin.py``); the bounds are several
        times what was measured for this atom (9.4e-5 Ry and 7.4e-3 Ry).
        """

        case, _, result = self.assert_case(
            "h_fixed_pbe",
            energy_tolerance=3.0e-4,
            term_tolerance=3.0e-2,
            level_tolerance=2.0e-2,
        )
        self.assertGreater(abs(result.energies.total - case["energy"]), 1.0e-6)

    def test_closed_shell_pbe_equals_the_unpolarized_run(self) -> None:
        """H2 with PBE: this package with and without spin polarization."""

        lines = [
            "Boundary_Sphere_Radius: 4.5",
            "Grid_Spacing: 0.6",
            "States_Num: 4",
            "Max_Iter: 200",
            "Convergence_Criterion: 1.0e-9",
            "Eigensolver: chebff",
            "Correlation_Type: pbe",
            "Atom_Types_Num: 1",
            "Atom_Type: H",
            "Local_Component: s",
            "begin Atom_Coord",
            "0.0 0.0 0.7",
            "0.0 0.0 -0.7",
            "end Atom_Coord",
        ]
        results = {}
        refinements = {}
        for spin in (False, True):
            system, results[spin] = _solve_hydrogen(lines, spin, self.backend)
            refinements[spin] = refine_energy(system, results[spin])
        unpolarized, polarized = results[False], results[True]
        self.assertTrue(unpolarized.converged and polarized.converged)
        self.assertIsNone(unpolarized.spin)
        self.assertEqual(unpolarized.spin_degeneracy, 2)
        self.assertAlmostEqual(
            polarized.energies.total, unpolarized.energies.total, delta=1.0e-8
        )
        for name in ("eigenvalue", "hartree", "integral_vxc_rho", "exchange_correlation"):
            self.assertAlmostEqual(
                getattr(polarized.energies, name),
                getattr(unpolarized.energies, name),
                delta=2.0e-7,
                msg=name,
            )
        self.assertAlmostEqual(polarized.net_magnetic_moment, 0.0, delta=1.0e-7)
        for channel in polarized.spin:
            # The occupied level; the empty ones are not converged as far.
            self.assertAlmostEqual(
                channel.eigenvalues[0], unpolarized.eigenvalues[0], delta=1.0e-7
            )
            np.testing.assert_allclose(
                channel.eigenvalues, unpolarized.eigenvalues, rtol=0, atol=1.0e-5
            )
            np.testing.assert_allclose(
                2.0 * channel.density, unpolarized.density, rtol=0, atol=1.0e-6
            )
            np.testing.assert_allclose(
                channel.xc_potential, unpolarized.xc_potential, rtol=0, atol=1.0e-5
            )
        # The B3LYP refinement of the two results is the same energy.
        self.assertTrue(refinements[True].spin_polarized)
        self.assertFalse(refinements[False].spin_polarized)
        for name in (
            "exact_exchange",
            "slater_exchange",
            "b88_exchange",
            "vwn_correlation",
            "lyp_correlation",
            "total",
        ):
            self.assertAlmostEqual(
                getattr(refinements[True], name),
                getattr(refinements[False], name),
                delta=2.0e-8,
                msg=name,
            )

    def test_one_electron_has_no_self_interaction_in_the_refinement(self) -> None:
        """H2+: one channel is empty, and its exchange cancels the Hartree energy."""

        lines = [
            "Boundary_Sphere_Radius: 4.5",
            "Grid_Spacing: 0.6",
            "States_Num: 4",
            "Net_Charges: 1",
            "Max_Iter: 200",
            "Convergence_Criterion: 1.0e-9",
            "Eigensolver: chebff",
            "Correlation_Type: pbe",
            "Atom_Types_Num: 1",
            "Atom_Type: H",
            "Local_Component: s",
            "begin Atom_Coord",
            "0.0 0.0 1.0",
            "0.0 0.0 -1.0",
            "end Atom_Coord",
        ]
        system, result = _solve_hydrogen(lines, True, self.backend)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(abs(result.net_magnetic_moment), 1.0, places=12)
        refinement = refine_energy(system, result)
        self.assertTrue(refinement.spin_polarized)
        self.assertEqual(refinement.exchange.pair_solves, 1)
        # Both energies in Ry, from the same Poisson solver and boundary.
        self.assertAlmostEqual(
            refinement.exact_exchange, -result.energies.hartree, delta=5.0e-8
        )
        self.assertAlmostEqual(refinement.lyp_correlation, 0.0, places=12)
        self.assertLess(refinement.vwn_correlation, -0.01)

    def test_moved_atoms_continue_from_both_channels(self) -> None:
        """An ionic step hands over the potential and subspace of each channel."""

        case, system, first = _solve("ch3_ca", self.backend)
        moved = tuple(
            Atom(atom.symbol, atom.position + shift)
            for atom, shift in zip(
                system.atoms,
                ([0.02, 0.0, -0.01], [-0.03, 0.01, 0.0], [0.0, 0.02, 0.01], [0.01, 0.0, 0.0]),
            )
        )
        next_system = driver.update_geometry(system, moved)
        warm = driver.warm_start_for_moved_atoms(system, first, next_system)
        self.assertEqual(len(warm.input_potential), 2)
        self.assertEqual(len(warm.eigensolver_state), 2)
        continued = driver.run_scf(next_system, warm_start=warm)
        self.assertTrue(continued.converged)
        # The same structure solved from its own atomic densities.
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(case, Path(directory)))
            cold_system = driver.prepare_single_point(
                replace(translation.problem, atoms=moved, recenter_geometry=False),
                backend=self.backend,
            )
        cold = driver.run_scf(cold_system)
        self.assertAlmostEqual(
            continued.energies.total, cold.energies.total, delta=5.0e-8
        )
        self.assertAlmostEqual(
            abs(continued.net_magnetic_moment), abs(cold.net_magnetic_moment), delta=1e-8
        )
        self.assertNotAlmostEqual(continued.energies.total, first.energies.total, places=4)
        self.assertLess(continued.iterations, cold.iterations)

    def test_relaxation_follows_the_fortran_trajectory(self) -> None:
        """BFGS on the O2 triplet: every step of Fortran's ``atom.cor``.

        The second structure is far too short a bond, so the loop also
        passes through an SCF far from the one it started with.
        """

        case = load_cases()["o2_bfgs_ca"]
        with tempfile.TemporaryDirectory() as directory:
            path = write_case(case, Path(directory))
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = cli_main(
                    [str(path), "--backend", self.backend, "--quiet", "--no-archive"]
                )
            record = (Path(directory) / "atom.cor").read_text()
            report = (Path(directory) / "parsec.out").read_text()
        self.assertEqual(code, 0)
        steps = re.findall(
            r"Step #\s+(\d+)\n.*?\n(.*?)\n(.*?)\nTotal Energy =\s*(\S+) \[eV/at\], "
            r"Max force =\s*(\S+)",
            record,
        )
        self.assertEqual(len(steps), len(case["structures"]))
        for (_, first, second, energy, force), structure in zip(steps, case["structures"]):
            positions = [[float(value) for value in line.split()] for line in (first, second)]
            np.testing.assert_allclose(
                positions, structure["positions"], rtol=0, atol=1.5e-6
            )
            self.assertAlmostEqual(
                float(energy), structure["energy_per_atom_ev"], delta=1.5e-5
            )
            self.assertAlmostEqual(float(force), structure["maximum_force"], delta=1.5e-5)
        self.assertIn(" Local minimum of total energy was found", record)
        # The report shows the triplet.
        moments = set(re.findall(r"Net magnetic moment \(in Bohr magnetons\):\s+(\S+)", report))
        self.assertIn("2.00", moments)

    def test_mpi_sectors_are_refused(self) -> None:
        case = load_cases()["n_ca"]
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(write_case(case, Path(directory)))
        with self.assertRaisesRegex(NotImplementedError, "cannot run as an MPI SCF"):
            driver.prepare_single_point(
                translation.problem,
                backend=self.backend,
                mpi_context=SimpleNamespace(rank=0, root=0),
            )


@unittest.skipUnless(native_available(), "native extension is unavailable")
class NativeBackendTests(_BackendCases, unittest.TestCase):
    backend = "native"


@unittest.skipUnless(
    cupy_available() and native_available(),
    "CUDA runtime or native extension is unavailable",
)
class CuPyBackendTests(_BackendCases, unittest.TestCase):
    """Both channels on one GPU, each with a subspace of its own."""

    backend = "cupy"

    def test_symmetric_molecule_is_solved_on_the_full_grid(self) -> None:
        # Without spin polarization this molecule is solved in the symmetry
        # representations of its two mirror planes.
        case, system, result = self.assert_case("h2o_spin_ca", forces=True)
        details = dict(result.backend.details)
        self.assertTrue(details["orbital_symmetry"].startswith("full grid (spin-polarized"))
        self.assertIn("local spin density", details["xc_implementation"])
        self.assertIsNone(system.scalar_field_adapter)
        _, plain_system, plain = _solve("h2o_nospin_ca", self.backend)
        self.assertTrue(
            dict(plain.backend.details)["orbital_symmetry"].startswith("CuPy real")
        )
        self.assertAlmostEqual(result.energies.total, plain.energies.total, delta=2.0e-8)

    def test_each_channel_has_its_own_device_subspace(self) -> None:
        _, system, result = _solve("n_ca", self.backend)
        solver = system.eigenproblem_solver
        first, second = solver.for_spin_channel(0), solver.for_spin_channel(1)
        self.assertIsNot(first, second)
        self.assertIs(first.operator, second.operator)
        self.assertIsNot(first.solver.device_state, second.solver.device_state)
        self.assertIs(result.spin[0].eigensolver_state, first.solver.device_state)
        self.assertIs(result.spin[1].eigensolver_state, second.solver.device_state)
        # A second loop on the same system starts both channels afresh.
        again = driver.run_scf(system)
        self.assertAlmostEqual(again.energies.total, result.energies.total, delta=2.0e-8)


if __name__ == "__main__":
    unittest.main()
