"""The Fock loop of the self-consistent hybrid functionals (reference path).

Independent references:

* Hartree-Fock of two electrons in one orbital is a local problem: the
  exchange operator on the occupied orbital is ``-V_H/2``, so HF H2 is the
  SCF of ``T + V_ion + V_NL + V_H/2`` with ``E = T + E_NL + E_ion + E_H/2``
  (solved here with SciPy's ``eigsh``, no exchange code involved);
* Hartree-Fock of one electron has no self-interaction: the HF energy of the
  H atom is the lowest eigenvalue of ``T + V_ion + V_NL`` (spin-polarized,
  with an empty down channel);
* the total energy of a converged PBE0 loop is the functional evaluated
  directly from its orbitals (kinetic, nonlocal, ionic, Hartree, semilocal
  exchange-correlation and ``a_x E_x``): the double counting is right;
* two equal spin channels give the unpolarized result.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy.sparse.linalg import eigsh

from parsec_python.Hamiltonian import KohnShamHamiltonian
from parsec_python.Input import UnsupportedParsecOptionError, parse_parsec_input
from parsec_python.Input.parsec_input import ParsecInputError
from parsec_python.SCF import prepare_single_point, run_scf
from parsec_python.SCF.hybrid import FockLoopSummary, default_pair_solver, hybrid_definition
from parsec_python.V_xc.exchange_operator import MultipolePairPoisson, apply_exchange_to_orbitals
from parsec_python.models import (
    ExactExchangeSettings,
    ExchangeLoopIteration,
    HybridEnergyBreakdown,
    SCFIteration,
    SCFSettings,
)


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

H2_ATOMS = "  0.0 0.0 -0.70\n  0.0 0.0  0.70\n"


class _Inputs:
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_hybrid_"))
        shutil.copy(HYDROGEN, self.directory / "H_POTRE.DAT")

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def translation(self, functional, *, states=2, atoms=H2_ATOMS, extra="", species="", name="parsec.in"):
        path = self.directory / name
        path.write_text(
            BASE.format(states=states, atoms=atoms, functional=functional, extra=extra, species=species)
        )
        return parse_parsec_input(path)

    def run_case(self, functional, **keywords):
        records = []
        system = prepare_single_point(self.translation(functional, **keywords).problem)
        result = run_scf(system, callback=records.append)
        return system, result, records


class HartreeFockReferenceTests(_Inputs, unittest.TestCase):
    def test_two_electrons_are_the_half_hartree_problem(self) -> None:
        system, result, records = self.run_case(
            "hf", extra="Exx_Tolerance: 1.0e-11 Ry\nExx_Pair_Solver: cg\n"
        )
        self.assertTrue(result.converged)
        grid = system.grid
        volume = grid.volume_element
        solver = MultipolePairPoisson(grid, system.negative_laplacian, system.input.hartree)

        def hamiltonian(potential):
            return KohnShamHamiltonian(
                system.negative_laplacian, potential, system.nonlocal_operator
            ).as_linear_operator()

        _, vectors = eigsh(hamiltonian(system.ionic_potential), k=1, which="SA", tol=1.0e-12)
        orbital = vectors[:, 0]
        hartree = np.zeros(grid.size)
        for _ in range(200):
            density = 2.0 * orbital**2 / volume
            new = solver.solve_batch(density[:, None])[:, 0]
            change = float(np.max(np.abs(new - hartree)))
            hartree = 0.5 * hartree + 0.5 * new
            values, vectors = eigsh(
                hamiltonian(system.ionic_potential + 0.5 * hartree), k=1, which="SA", v0=orbital, tol=1.0e-13
            )
            orbital = vectors[:, 0]
            if change < 1.0e-11:
                break
        density = 2.0 * orbital**2 / volume
        hartree = solver.solve_batch(density[:, None])[:, 0]
        one_body = 2.0 * float(
            orbital @ (system.negative_laplacian @ orbital) + orbital @ system.nonlocal_operator.apply(orbital)
        )
        hartree_energy = 0.5 * volume * float(density @ hartree)
        expected = (
            one_body
            + volume * float(density @ system.ionic_potential)
            + 0.5 * hartree_energy
            + system.ion_ion_energy
        )
        self.assertAlmostEqual(result.energies.total, expected, delta=2.0e-7)
        self.assertAlmostEqual(float(result.eigenvalues[0]), float(values[0]), delta=2.0e-6)
        # E_x = -E_H / 2 for two electrons in one orbital.  E_x is first order in
        # the orbitals, so the inner SCF criterion (1e-7 Ry here) bounds it to a
        # few 1e-6 relative while the total energy above, second order, agrees
        # to 2e-7: measured 2.0e-6 with the adaptive inner criterion and DIIS
        # (5 outer iterations; total energy equal to 1e-10 to the fixed loop's
        # 8), 5e-7 with the fixed loop.
        self.assertAlmostEqual(result.exact_exchange.exchange_energy / (-0.5 * hartree_energy), 1.0, delta=3.0e-6)
        self.assertIsInstance(result.energies, HybridEnergyBreakdown)
        self.assertEqual(result.energies.semilocal_exchange_correlation, 0.0)
        self.assertAlmostEqual(result.energies.exact_exchange, result.exact_exchange.exchange_energy)

    def test_one_electron_has_no_self_interaction(self) -> None:
        system, result, _ = self.run_case(
            "hf",
            states=2,
            atoms="  0.0 0.0 0.0\n",
            species="Initial_Spin_Polarization: 1.0\n",
            extra="Spin_Polarization: .true.\nExx_Tolerance: 1.0e-11 Ry\nExx_Pair_Solver: cg\n",
        )
        self.assertTrue(result.converged)
        values, _ = eigsh(
            KohnShamHamiltonian(
                system.negative_laplacian, system.ionic_potential, system.nonlocal_operator
            ).as_linear_operator(),
            k=1,
            which="SA",
            tol=1.0e-12,
        )
        up, down = result.spin
        self.assertAlmostEqual(float(np.sum(up.occupations)), 1.0, places=8)
        self.assertAlmostEqual(float(np.sum(down.occupations)), 0.0, places=8)
        self.assertAlmostEqual(result.energies.total - system.ion_ion_energy, float(values[0]), delta=2.0e-7)
        # A level is first order in the eigensolver residual, the energy second.
        self.assertAlmostEqual(float(up.eigenvalues[0]), float(values[0]), delta=1.0e-5)
        # E_x = -E_H of one electron.
        hartree = system.solve_hartree(result.density).potential
        hartree_energy = 0.5 * system.grid.volume_element * float(result.density @ hartree)
        self.assertAlmostEqual(result.exact_exchange.exchange_energy / -hartree_energy, 1.0, delta=1.0e-6)


class PBE0LoopTests(_Inputs, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = _Inputs()
        cls.inputs.setUp()
        cls.system, cls.result, cls.records = cls.inputs.run_case(
            "pbe0", states=3, extra="Exx_Tolerance: 1.0e-10 Ry\n"
        )
        cls.pbe_system, cls.pbe_result, _ = cls.inputs.run_case("pbe", states=3, name="pbe.in")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.inputs.tearDown()

    def direct_energy(self, system, result, fraction):
        """The hybrid functional evaluated from the orbitals and density of ``result``."""

        volume = system.grid.volume_element
        orbitals = result.wavefunctions
        occupations = result.occupations
        one_body = 2.0 * sum(
            occupations[i]
            * float(
                orbitals[:, i] @ (system.negative_laplacian @ orbitals[:, i])
                + orbitals[:, i] @ system.nonlocal_operator.apply(orbitals[:, i])
            )
            for i in range(orbitals.shape[1])
        )
        density = result.density
        hartree = system.solve_hartree(density).potential
        semilocal = system.evaluate_xc(density).total_energy
        # The pair solver of the run (the Kronecker solver on the bounding box).
        solver = default_pair_solver(system, hybrid_definition(system.input.scf))
        exchange = apply_exchange_to_orbitals(
            orbitals, occupations, volume, solver, columns=np.flatnonzero(occupations > 1.0e-12)
        ).energy
        total = (
            one_body
            + volume * float(density @ system.ionic_potential)
            + 0.5 * volume * float(density @ hartree)
            + semilocal
            + fraction * exchange
            + system.ion_ion_energy
        )
        return total, semilocal, exchange

    def test_total_energy_is_the_functional_of_the_orbitals(self) -> None:
        system, result = self.system, self.result
        self.assertTrue(result.converged)
        direct, semilocal, exchange = self.direct_energy(system, result, 0.25)
        self.assertAlmostEqual(result.energies.total, direct, delta=1.0e-7)
        self.assertAlmostEqual(result.energies.semilocal_exchange_correlation, semilocal, delta=1.0e-7)
        self.assertAlmostEqual(result.exact_exchange.exchange_energy, exchange, delta=1.0e-8)
        self.assertFalse(result.energies.linearized)
        # The band energy holds 2 a_x E_x of exchange.
        # The operator of the band energy is that of the orbitals before the last
        # inner loop: first order in their change (dexx is second order).
        self.assertAlmostEqual(result.energies.exact_exchange_band, 2.0 * 0.25 * exchange, delta=1.0e-5)

    def test_fractional_occupations(self) -> None:
        """A hot electron gas (40000 K): the exchange of the ensemble density matrix."""

        translation = self.inputs.translation(
            "pbeh",
            states=4,
            extra="Exact_Exchange_Fraction: 0.4\nExx_Tolerance: 1.0e-10 Ry\n",
            name="hot.in",
        )
        problem = translation.problem
        problem = replace(problem, scf=replace(problem.scf, fermi_temperature_kelvin=40000.0))
        system = prepare_single_point(problem)
        result = run_scf(system)
        self.assertTrue(result.converged)
        self.assertGreater(float(result.occupations[1]), 1.0e-3)
        self.assertLess(float(result.occupations[0]), 1.0 - 1.0e-3)
        direct, _, exchange = self.direct_energy(system, result, 0.4)
        self.assertAlmostEqual(result.energies.total, direct, delta=1.0e-7)
        self.assertAlmostEqual(result.exact_exchange.exchange_energy, exchange, delta=1.0e-8)

    def test_outer_iterations_are_variational_and_reported(self) -> None:
        outer = [record for record in self.records if isinstance(record, ExchangeLoopIteration)]
        inner = [record for record in self.records if isinstance(record, SCFIteration)]
        self.assertEqual(outer[0].iteration, 0)
        self.assertEqual(outer[0].start_functional, "pbe")
        self.assertTrue(outer[-1].converged)
        self.assertEqual(len(inner), self.result.iterations)
        self.assertEqual(self.result.iterations, len(self.result.history))
        totals = [record.total_energy for record in outer]
        final = self.result.energies.total
        self.assertAlmostEqual(totals[-1], final, places=12)
        for earlier, later in zip(totals, totals[1:]):
            self.assertLessEqual(later, earlier + 1.0e-9)
        for record in outer[1:]:
            self.assertGreaterEqual(record.estimated_error, -1.0e-12)
            # dexx bounds the energy still to be gained.
            self.assertGreaterEqual(record.estimated_error + 1.0e-9, record.total_energy - final)
        errors = [record.estimated_error for record in outer[1:]]
        self.assertLess(errors[-1], 1.0e-10)
        summary = self.result.exact_exchange
        self.assertIsInstance(summary, FockLoopSummary)
        self.assertEqual(summary.iterations, tuple(outer))
        # Inner iterations report the linearized energy E_formula - a_x E_x(P0).
        self.assertTrue(all(record.energies.linearized for record in inner if isinstance(record.energies, HybridEnergyBreakdown)))
        self.assertGreater(summary.start_iterations, 0)

    def test_levels_move_as_exact_exchange_moves_them(self) -> None:
        pbe = self.pbe_result.eigenvalues
        hybrid = self.result.eigenvalues
        self.assertLess(hybrid[0], pbe[0])  # the HOMO goes down
        self.assertGreater(hybrid[1] - hybrid[0], pbe[1] - pbe[0])  # the gap opens
        self.assertLess(self.result.energies.total, self.pbe_result.energies.total + 0.05)

    def test_two_equal_spin_channels_give_the_unpolarized_loop(self) -> None:
        _, spin, _ = self.inputs.run_case(
            "pbe0",
            states=3,
            species="Initial_Spin_Polarization: 0.0\n",
            extra="Spin_Polarization: .true.\nExx_Tolerance: 1.0e-10 Ry\n",
            name="spin.in",
        )
        self.assertTrue(spin.converged)
        self.assertAlmostEqual(spin.energies.total, self.result.energies.total, delta=2.0e-7)
        self.assertAlmostEqual(
            spin.exact_exchange.exchange_energy, self.result.exact_exchange.exchange_energy, delta=1.0e-6
        )
        up, down = spin.spin
        np.testing.assert_allclose(up.eigenvalues, self.result.eigenvalues, atol=2.0e-6)
        np.testing.assert_allclose(down.eigenvalues, self.result.eigenvalues, atol=2.0e-6)


class ScreenedHybridTests(_Inputs, unittest.TestCase):
    """HSE06 (erfc kernel) and CAM-B3LYP (1/r and erf kernels) through the Fock loop."""

    def exchange_energy(self, system, result, terms):
        """``sum_t a_t E_x[kernel_t]`` of the orbitals of ``result``, one Kronecker solver per kernel."""

        from parsec_python.Hartree.kronecker import KroneckerPoisson

        occupied = np.flatnonzero(result.occupations > 1.0e-12)
        total = 0.0
        for kernel, fraction, omega in terms:
            solver = KroneckerPoisson(
                system.grid,
                kernel=kernel,
                omega=omega,
                multipole_order=system.input.hartree.multipole_order,
            )
            total += fraction * apply_exchange_to_orbitals(
                result.wavefunctions, result.occupations, system.grid.volume_element, solver, columns=occupied
            ).energy
        return total

    def test_screened_and_two_kernel_hybrids(self) -> None:
        extra = "Exx_Tolerance: 1.0e-10 Ry\n"
        levels = {}
        for functional in ("pbe", "pbe0", "hse06", "cam-b3lyp"):
            system, result, _ = self.run_case(
                functional, states=3, extra="" if functional == "pbe" else extra, name=f"{functional}.in"
            )
            self.assertTrue(result.converged, functional)
            levels[functional] = result.eigenvalues
            if functional in ("hse06", "cam-b3lyp"):
                definition = hybrid_definition(system.input.scf)
                expected = self.exchange_energy(system, result, definition.terms)
                weighted = definition.fraction * result.exact_exchange.exchange_energy
                # CAM-B3LYP's one solve of 0.19/r + 0.46 erf(w r)/r works on the box
                # of its erf part, which this 5-bohr sphere at h = 0.5 pads for the
                # smearing cap of the split screening; the separate 1/r solve of
                # exchange_energy is not padded.  Their multipole boundaries at
                # Solver_Lpole 4 differ by 2e-8 Ry in E_x.
                tolerance = 1.0e-8 if functional == "hse06" else 1.0e-7
                self.assertAlmostEqual(weighted, expected, delta=tolerance)
                self.assertAlmostEqual(result.energies.exact_exchange, expected, delta=tolerance)
                self.assertTrue(result.exact_exchange.converged)
        # Screening keeps only the short-range part of PBE0's exact exchange:
        # the HOMO of HSE06 lies between those of PBE and PBE0.
        self.assertLess(levels["pbe0"][0], levels["hse06"][0])
        self.assertLess(levels["hse06"][0], levels["pbe"][0])
        # Long-range exact exchange lowers the HOMO of CAM-B3LYP below PBE's.
        self.assertLess(levels["cam-b3lyp"][0], levels["pbe"][0])


class HybridReportTests(_Inputs, unittest.TestCase):
    def test_reference_command_line_reports_the_fock_loop(self) -> None:
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from parsec_python import cli

        self.translation("pbe0", extra="Exx_Tolerance: 1.0e-6 Ry\n")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            status = cli.main([str(self.directory / "parsec.in"), "--quiet"])
        self.assertEqual(status, 0)
        # The archive: energies only under energy_*_ry, the
        # loop of every SCF history row.
        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertNotIn("energy_exact_exchange_fraction_ry", archive.files)
        self.assertNotIn("energy_linearized_ry", archive.files)
        self.assertEqual(float(archive["exx_fraction"]), 0.25)
        self.assertFalse(bool(archive["exx_energies_linearized"]))
        loops = archive["scf_history_loop"]
        self.assertEqual(loops.shape[0], archive["scf_history"].shape[0])
        self.assertEqual(int(loops[0]), 0)
        self.assertEqual(int(loops[-1]), int(archive["exx_history"][-1, 0]))
        self.assertTrue(np.all(np.diff(loops) >= 0))
        self.assertAlmostEqual(
            float(archive["energy_total_ry"]), float(archive["exx_history"][-1, 2]), places=10
        )
        text = (self.directory / "parsec.out").read_text()
        for expected in (
            " Exchange-Correlation functional is pbe0",
            "Fock loop: start from a PBE SCF",
            "Fock loop (PBE0): exchange operator of the PBE start orbitals",
            "Fock loop iteration   1 (PBE0)",
            "Exact exchange a_x*E_x",
            "(linearized: fixed operator)",
            "Hybrid functional PBE0: generalized Kohn-Sham energies",
            "Fock loop converged after",
            "Pair-density Poisson solver: Kronecker",
            "Self-consistency convergence achieved.",
        ):
            self.assertIn(expected, text)


class ReviewFixTests(_Inputs, unittest.TestCase):
    """Defects of the hybrids' integration with the rest of the code, fixed on 2026-11-06."""

    def test_eigen_dat_keeps_spin_records_and_skips_fock_records(self) -> None:
        """eigen.dat of a spin-polarized run (eigenvalues_up/_down only)."""

        from parsec_python.Output.output_files import RunOutputFiles
        from parsec_python.models import (
            EnergyBreakdown,
            OutputFileSettings,
            SpinPolarizedSCFIteration,
        )

        settings = OutputFileSettings(eigenvalues=True, intermediate_eigenvalues=True)
        files = RunOutputFiles(settings, self.directory, write=lambda _line: None)
        files.start()
        energies = EnergyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        files.iteration(
            SpinPolarizedSCFIteration(
                iteration=1,
                weighted_residual=1.0,
                plain_residual=1.0,
                energies=energies,
                eigenvalues_up=(-0.75, 0.25),
                eigenvalues_down=(-0.5, 0.5),
                occupations_up=(1.0, 0.0),
                occupations_down=(1.0, 0.0),
            )
        )
        files.iteration(ExchangeLoopIteration(iteration=1, functional="PBE0", fraction=0.25, exchange_energy=-1.0))
        text = (self.directory / "eigen.dat").read_text().splitlines()
        self.assertGreater(len(text), 1)  # the header and the levels of both channels
        self.assertTrue(any("-0.75" in line or "-0.750" in line for line in text[1:]), text)

    def test_complete_input_reads_back_with_every_hybrid_keyword(self) -> None:
        """The Fock-loop keys are rebuilt, not also copied."""

        from parsec_python.Input.complete_input import COMPLETE_INPUT_NAME, _RECONSTRUCTED
        from parsec_python.Input.parsec_input import _HYBRID_KEYS

        self.assertLessEqual(set(_HYBRID_KEYS), _RECONSTRUCTED)
        for functional, extra in (
            ("pbeh", "Exact_Exchange_Fraction: 0.4\nExx_Tolerance: 1.0e-9 Ry\nExx_Batch_Size: 4\n"),
            ("hse06", "Screening_Parameter: 0.106\nExx_Max_Iter: 9\nExx_Tolerance: 1.0e-5 eV\n"),
            ("hf", "Exx_Pair_Solver: cg\n"),
        ):
            with self.subTest(functional=functional):
                translation = self.translation(
                    functional, extra=extra + "Generate_Complete_Parsec_In: .true.\n"
                )
                # The text the command lines write to bak_complete_parsec.in.
                text = translation.complete_input
                self.assertIsNotNone(text)
                path = self.directory / COMPLETE_INPUT_NAME
                path.write_text(text)
                again = parse_parsec_input(path).problem.scf
                self.assertEqual(again.xc_functional, translation.problem.scf.xc_functional)
                self.assertEqual(again.exact_exchange, translation.problem.scf.exact_exchange)


    def use_hydrogen_label(self, label: str) -> None:
        """The library's H potential with another correlation label in its header."""

        lines = HYDROGEN.read_text().splitlines(keepends=True)
        lines[0] = lines[0].replace(" pb ", f" {label} ", 1)
        (self.directory / "H_POTRE.DAT").write_text("".join(lines))

    def test_pseudopotential_rule_of_the_hybrids(self) -> None:
        """One rule for every hybrid, applied when the input is read."""

        from parsec_python.Output.hybrid_output import hybrid_correlation_lines
        from parsec_python.V_ion import load_pseudopotentials

        # PBE files: no note.
        self.assertFalse(any("was generated for" in w for w in self.translation("pbe0").warnings))
        # B3PW91 also reads PW91 files; PBE0 refuses them by name before the run.
        self.use_hydrogen_label("pw")
        translation = self.translation("b3pw91")
        self.assertFalse(any("was generated for" in w for w in translation.warnings))
        load_pseudopotentials(translation.problem.pseudopotentials, xc_functional="b3pw91")
        self.assertIn("files of PBE or PW91", "\n".join(hybrid_correlation_lines(translation.problem.scf)))
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "generated for 'pw'; a pbe0 run reads PBE"):
            self.translation("pbe0")
        # A CA file: used with a named note, as in a PBE run.
        self.use_hydrogen_label("ca")
        for functional in ("pbe0", "b3lyp", "hf"):
            with self.subTest(functional=functional):
                notes = [w for w in self.translation(functional).warnings if "generated for 'ca'" in w]
                self.assertEqual(len(notes), 1, functional)
                self.assertIn(f"the hybrid {functional}", notes[0])

    def test_fock_loop_diagnostics(self) -> None:
        """The ring of 12 hydrogen atoms: an unstable symmetric start and a slow soft mode are reported."""

        from parsec_python.Output.hybrid_output import fock_loop_warning_lines
        from parsec_python.SCF.hybrid import fock_loop_diagnostics

        def record(n, energy, error):
            return ExchangeLoopIteration(
                iteration=n, functional="PBE0", fraction=0.25, exchange_energy=-1.0,
                total_energy=energy, estimated_error=error,
            )

        # dexx of the coarse device run of the ring: falls, rises (symmetry
        # breaking), then energy steps shrink by 0.9 per iteration.
        errors = [None, 5.4e-5, 2.6e-7, 2.3e-7, 1.0e-5, 5.5e-4, 3.2e-2, 4.9e-3]
        energies = [-12.6309, -12.6319, -12.63187, -12.631868, -12.63188, -12.6325, -12.6689, -12.6973]
        records = [record(n, e, d) for n, (e, d) in enumerate(zip(energies, errors))]
        step = 1.0e-5
        for n in range(8, 14):
            records.append(record(n, records[-1].total_energy - step, 1.0e-6))
            step *= 0.9
        found = fock_loop_diagnostics(records, [np.array([1.0, 1.0, 0.0])])
        self.assertEqual(found["instability"], 4)
        self.assertAlmostEqual(found["convergence_ratio"], 0.9, places=6)
        self.assertAlmostEqual(found["remaining_energy"], step / 0.9 * 0.9 / 0.1, places=12)
        self.assertEqual(found["fractional_occupation"], 0.0)
        half = fock_loop_diagnostics(records[:3], [np.array([1.0, 0.5, 0.5, 0.0])])
        self.assertIsNone(half["instability"])
        self.assertIsNone(half["convergence_ratio"])
        self.assertEqual(half["fractional_occupation"], 0.5)
        summary = FockLoopSummary(
            functional="pbe0", display_name="PBE0", fraction=0.25, kernel="coulomb", omega=None,
            iterations=tuple(records), converged=False, exchange_energy=-1.0, estimated_error=1e-6,
            tolerance=1e-8, start_functional="pbe", start_iterations=10, pair_solver="x",
            pair_solves=0, poisson_seconds=0.0, exchange_seconds=0.0, seconds=0.0,
            **dict(found, fractional_occupation=0.5),
        )
        text = "\n".join(fock_loop_warning_lines(summary))
        self.assertIn("rose at outer iteration 4", text)
        self.assertIn("converges slowly", text)
        self.assertIn("fractional occupations", text)
        from parsec_python.models import DEFAULT_EXX_MAX_ITERATIONS

        self.assertEqual(DEFAULT_EXX_MAX_ITERATIONS, 100)

    def test_energy_refinement_option_is_refused_with_a_hybrid(self) -> None:
        """The command-line option takes the input's rule."""

        from parsec_python.Energy.refinement import apply_refinement_option

        translation = self.translation("pbe0")
        with self.assertRaisesRegex(ValueError, "already self-consistent"):
            apply_refinement_option(translation, "b3lyp")
        # "none" stays possible, and so does a refinement of a semilocal run.
        self.assertIsNone(apply_refinement_option(translation, "none").energy_refinement)
        plain = self.translation("pbe", name="plain.in")
        self.assertEqual(apply_refinement_option(plain, "b3lyp").energy_refinement, "b3lyp")


class HybridInputTests(_Inputs, unittest.TestCase):
    def test_labels_keywords_and_defaults(self) -> None:
        translation = self.translation("PBE0")
        scf = translation.problem.scf
        self.assertEqual(scf.xc_functional, "pbe0")
        self.assertTrue(scf.is_hybrid)
        self.assertEqual(scf.exact_exchange, ExactExchangeSettings())
        self.assertNotIn("exact_exchange", repr(replace(scf, xc_functional="pbe", exact_exchange=None)))
        self.assertIn("ExactExchangeSettings()", repr(scf))
        translation = self.translation(
            "pbeh",
            extra=(
                "Exact_Exchange_Fraction: 0.4\nExx_Max_Iter: 7\nExx_Tolerance: 1.0e-5 Ry\n"
                "Exx_Batch_Size: 4\n"
            ),
        )
        settings = translation.problem.scf.exact_exchange
        self.assertEqual(
            settings,
            ExactExchangeSettings(fraction=0.4, max_iterations=7, tolerance=1.0e-5, batch_size=4),
        )
        self.assertEqual(hybrid_definition(translation.problem.scf).fraction, 0.4)
        self.assertEqual(self.translation("hartree_fock").problem.scf.xc_functional, "hf")
        self.assertEqual(hybrid_definition(self.translation("pbe0").problem.scf).fraction, 0.25)
        # A functional without exact exchange keeps its settings and their text.
        plain = self.translation("pbe").problem.scf
        self.assertIsNone(plain.exact_exchange)
        self.assertNotIn("exact_exchange", repr(plain))

    def test_complete_input_keeps_the_fock_loop_keywords(self) -> None:
        from parsec_python.Input.complete_input import complete_input_text

        translation = self.translation(
            "pbeh", extra="Exact_Exchange_Fraction: 0.3\nExx_Tolerance: 1.0e-8 Ry\nExx_Pair_Solver: cg\n"
        )
        path = self.directory / "complete.in"
        path.write_text(complete_input_text(translation, {}, {}))
        again = parse_parsec_input(path).problem.scf
        self.assertEqual(again.xc_functional, "pbeh")
        self.assertEqual(again.exact_exchange, translation.problem.scf.exact_exchange)

    def test_labels_are_those_of_the_registry(self) -> None:
        from parsec_python.V_xc.hybrid import HYBRID_FUNCTIONALS
        from parsec_python.models import HYBRID_XC_FUNCTIONALS, XC_FUNCTIONAL_NAMES

        self.assertEqual(HYBRID_XC_FUNCTIONALS, frozenset(HYBRID_FUNCTIONALS))
        self.assertLessEqual(HYBRID_XC_FUNCTIONALS, XC_FUNCTIONAL_NAMES)
        for text, label in (("LC-wPBE", "lc-wpbe"), ("cam_b3lyp", "cam-b3lyp"), ("B3LYP", "b3lyp")):
            from parsec_python.Input.parsec_input import _hybrid_label

            self.assertEqual(_hybrid_label(text), label)
        self.assertIsNone(_hybrid_label("pbe"))
        self.assertEqual(self.translation("b3lyp").problem.scf.xc_functional, "b3lyp")

    def test_accelerated_path_routes_or_refuses_by_name(self) -> None:
        from parsec_python.acceleration import cli as accelerated_cli
        from parsec_python.acceleration import driver as accelerated_driver

        translation = self.translation("pbe0")
        features = accelerated_cli._reference_only_features(
            translation, accelerated_backend_selected=False
        )
        self.assertTrue(any("hybrid functional" in feature for feature in features))
        self.assertTrue(
            any("hybrid functional" in item for item in accelerated_cli._mpi_unsupported_features(translation))
        )
        plain = self.translation("pbe", name="plain.in")
        self.assertFalse(
            any(
                "hybrid" in feature
                for feature in accelerated_cli._reference_only_features(plain, accelerated_backend_selected=False)
            )
        )
        # The device Fock loop needs the CuPy backend in one process
        # (acceleration/tests/test_hybrid_device.py runs it).
        with self.assertRaisesRegex(NotImplementedError, "CuPy backend"):
            accelerated_driver.prepare_single_point(translation.problem, backend="scipy")
        with self.assertRaisesRegex(NotImplementedError, "MPI"):
            accelerated_driver.prepare_single_point(translation.problem, mpi_context=object())

    def test_requests_that_cannot_be_served_are_refused(self) -> None:
        with self.assertRaisesRegex(ParsecInputError, "hybrid Correlation_Type"):
            self.translation("pbe", extra="Exact_Exchange_Fraction: 0.3\n")
        with self.assertRaisesRegex(ParsecInputError, "fraction"):
            self.translation("pbe0", extra="Exact_Exchange_Fraction: 1.5\n")
        with self.assertRaisesRegex(ParsecInputError, "cannot be changed"):
            self.translation("hf", extra="Exact_Exchange_Fraction: 0.5\n")
        with self.assertRaisesRegex(ParsecInputError, "use pbeh"):
            self.translation("pbe0", extra="Exact_Exchange_Fraction: 0.3\n")
        with self.assertRaisesRegex(ParsecInputError, "no screening parameter"):
            self.translation("pbe0", extra="Screening_Parameter: 0.11\n")
        # Screened kernels need the Kronecker pair solver.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Exx_Pair_Solver: kronecker"):
            self.translation("hse06", extra="Exx_Pair_Solver: cg\n")
        with self.assertRaisesRegex(ParsecInputError, "HSE03 is defined"):
            self.translation("hse03", extra="Screening_Parameter: 0.106\n")
        settings = self.translation("hse06", extra="Screening_Parameter: 0.106\n").problem.scf
        definition = hybrid_definition(settings)
        self.assertEqual((definition.kernel, definition.omega, definition.fraction), ("erfc", 0.106, 0.25))
        definition = hybrid_definition(self.translation("cam-b3lyp").problem.scf)
        self.assertEqual(definition.fraction, 1.0)
        self.assertEqual([term[0] for term in definition.terms], ["coulomb", "erf"])
        with self.assertRaisesRegex(ParsecInputError, "pair solver"):
            self.translation("pbe0", extra="Exx_Pair_Solver: fft\n")
        with self.assertRaisesRegex(ParsecInputError, "Exx_Max_Iter"):
            self.translation("pbe0", extra="Exx_Max_Iter: 2.5\n")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Energy_Refinement"):
            self.translation("pbe0", extra="Energy_Refinement: b3lyp\n")
        # PARSEC's rule (error 158: TS needs PBE) already stops it.
        with self.assertRaisesRegex(ParsecInputError, "Van_Der_Waals"):
            self.translation("pbe0", extra="Van_Der_Waals: .true.\n")
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Polarizability"):
            self.translation("pbe0", extra="Polarizability: .true.\n")
        with self.assertRaises(ValueError):
            SCFSettings(xc_functional="pbe", exact_exchange=ExactExchangeSettings())
        with self.assertRaises(ValueError):
            ExactExchangeSettings(batch_size=0)


if __name__ == "__main__":
    unittest.main()
