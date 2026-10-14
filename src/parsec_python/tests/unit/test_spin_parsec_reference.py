"""Spin-polarized calculations on the readable path against Fortran PARSEC.

Two cases of ``data/spin_parsec_reference.json`` are solved to
self-consistency and compared with the ``parsec.out`` of Fortran PARSEC for
the same input: total energy, energy terms, the levels, occupations and
electrons of each spin channel, the Fermi level and, for the methyl radical,
every force table.  With the CA functional the two programs agree to the
digits Fortran prints.  The other cases of the fixture (the nitrogen and
oxygen atoms, O2, water, PBE) run on the accelerated backends, which use the
same SCF loop and take a tenth of the time
(``acceleration/tests/test_spin_polarized.py``).

One thing Fortran cannot be asked: a channel without electrons, here H2+
with PBE, must be a calculation like any other.  It does not converge in
Fortran.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parsec_python.Forces import compute_forces
from parsec_python.Input import parse_parsec_input
from parsec_python.Output import ParsecTextReporter
from parsec_python.SCF import (
    prepare_single_point,
    run_scf,
    run_spin_scf,
    update_geometry,
    warm_start_for_moved_atoms,
)
from parsec_python.SCF.spin_polarized import SpinOrbitalBlocks
from parsec_python.cli import save_result_archive
from parsec_python.models import Atom
from parsec_python.tests.spin_reference import (
    assert_forces_match,
    assert_matches_fortran,
    load_cases,
    pseudopotential_file,
    write_case,
)


class _FortranCase:
    """Solve one case of the fixture once for the tests of a class."""

    case_name: str
    with_forces = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.case = load_cases()[cls.case_name]
        with tempfile.TemporaryDirectory() as directory:
            cls.translation = parse_parsec_input(write_case(cls.case, Path(directory)))
            cls.system = prepare_single_point(cls.translation.problem)
        cls.report: list[str] = []
        reporter = ParsecTextReporter(cls.report.append, cls.translation)
        reporter.header()
        reporter.setup(cls.system)
        cls.result = run_scf(cls.system, callback=reporter.iteration)
        cls.forces = (
            compute_forces(cls.system, cls.result) if cls.with_forces else None
        )

    def assert_spin_result_is_consistent(self) -> None:
        """The fields of a spin-polarized result describe its two channels."""

        result = self.result
        grid = self.system.grid
        up, down = result.spin
        self.assertTrue(result.spin_polarized)
        self.assertEqual(result.spin_degeneracy, 1)
        np.testing.assert_array_equal(result.density, up.density + down.density)
        for channel in (up, down):
            self.assertAlmostEqual(
                grid.integrate(channel.density), channel.electron_count, delta=1.0e-10
            )
            self.assertAlmostEqual(
                float(np.sum(channel.occupations)), channel.electron_count, places=12
            )
            np.testing.assert_array_equal(
                channel.output_effective_potential,
                result.ionic_potential + result.hartree_potential + channel.xc_potential,
            )
        self.assertAlmostEqual(
            up.electron_count + down.electron_count, result.electron_count, delta=1.0e-10
        )
        states = up.eigenvalues.size
        np.testing.assert_array_equal(result.eigenvalues[:states], up.eigenvalues)
        np.testing.assert_array_equal(result.eigenvalues[states:], down.eigenvalues)
        np.testing.assert_array_equal(result.occupations[states:], down.occupations)
        self.assertEqual(result.wavefunctions.shape, (grid.size, 2 * states))
        self.assertTrue(np.shares_memory(up.wavefunctions, result.wavefunctions))
        np.testing.assert_array_equal(
            result.wavefunctions[:, states:], down.wavefunctions
        )
        for name in (
            "xc_potential",
            "input_effective_potential",
            "output_effective_potential",
            "next_effective_potential",
        ):
            np.testing.assert_array_equal(
                getattr(result, name),
                0.5 * (getattr(up, name) + getattr(down, name)),
                err_msg=name,
            )
        last = result.history[-1]
        self.assertTrue(last.spin_polarized)
        self.assertEqual(last.spin_electrons, (up.electron_count, down.electron_count))
        self.assertEqual(last.weighted_residual, max(last.spin_weighted_residuals))
        self.assertEqual(last.plain_residual, max(last.spin_plain_residuals))
        self.assertEqual(len(last.eigenvalues), 2 * states)


class HydrogenFixedOccupationTests(_FortranCase, unittest.TestCase):
    case_name = "h_fixed_ca"

    def test_energy_terms_levels_and_occupations(self) -> None:
        assert_matches_fortran(self, self.result, self.case)
        up, down = self.result.spin
        np.testing.assert_array_equal(up.occupations, [0.7, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(down.occupations, [0.3, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(self.result.net_magnetic_moment, 0.4, places=12)
        # The majority channel is bound more strongly.
        self.assertLess(up.eigenvalues[0], down.eigenvalues[0] - 0.05)

    def test_result_describes_both_channels(self) -> None:
        self.assert_spin_result_is_consistent()

    def test_starting_densities_of_initchrg(self) -> None:
        """Both channels are scaled like the total: they add up to twice it."""

        system = self.system
        up, down = system.initial_spin_densities
        np.testing.assert_allclose(up, 1.1 * system.initial_density, rtol=1.0e-15)
        np.testing.assert_allclose(down, 0.9 * system.initial_density, rtol=1.0e-15)
        moment = system.grid.integrate(up) - system.grid.integrate(down)
        # "Initial magnetic moment [Bohr magnetons]" of the Fortran run.
        self.assertAlmostEqual(moment, self.case["initial_magnetic_moment"], delta=5.1e-6)
        self.assertIn(
            f" Initial magnetic moment [Bohr magnetons] = {moment:12.5f}",
            "\n".join(self.report),
        )


class MethylRadicalTests(_FortranCase, unittest.TestCase):
    case_name = "ch3_ca"
    with_forces = True

    def test_energy_terms_levels_and_occupations(self) -> None:
        assert_matches_fortran(self, self.result, self.case)
        self.assertAlmostEqual(abs(self.result.net_magnetic_moment), 1.0, places=10)

    def test_forces_match_fortran_parsec(self) -> None:
        assert_forces_match(self, self.forces, self.case)
        self.assertEqual(self.forces.symmetry_operations, 1)

    def test_report_has_the_layout_of_a_spin_polarized_parsec_run(self) -> None:
        text = "\n".join(self.report)
        self.assertIn(" Spin-polarized computation!", text)
        self.assertNotIn("No spin effects", text)
        self.assertIn(" Initial spin polarization: \n  C           0.1000\n  H           0.1000", text)
        last = self.report[-1].splitlines()
        up, down = self.result.spin
        self.assertIn(
            f" Electrons in up, down spins: {up.electron_count:7.2f} {down.electron_count:7.2f}",
            last,
        )
        self.assertIn(
            " Net magnetic moment (in Bohr magnetons): "
            f"{self.result.net_magnetic_moment:7.2f}",
            last,
        )
        self.assertIn(f" Fermi level at {self.result.fermi_level:10.4f} [Ry]", last)
        for label, channel in (("up", up), ("dn", down)):
            header = last.index(
                f"{label} State   Eigenvalue [Ry]      Eigenvalue [eV]    Occup.     Repr."
            )
            first = last[header + 2]
            self.assertTrue(first.startswith(f"    1   {channel.eigenvalues[0]:18.10f}"))
            self.assertTrue(first.endswith(f" {channel.occupations[0]:9.4f}        1"))
        step = self.result.history[-1]
        for label, plain, weighted in zip(
            ("up", "dn"), step.spin_plain_residuals, step.spin_weighted_residuals
        ):
            self.assertIn(
                f"  0-{step.iteration:3d} {label} SRE of pot. & charge weighted pot = "
                f"{plain:14.10f} {weighted:14.10f}",
                last,
            )
        self.assertEqual(
            sum(line.endswith("   up") or line.endswith("   dn") for line in last), 2
        )

    def test_archive_keeps_both_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_result_archive(Path(directory) / "result", self.result)
            with np.load(path) as archive:
                self.assertTrue(bool(archive["spin_polarized"]))
                states = self.result.spin[0].eigenvalues.size
                self.assertEqual(archive["spin_eigenvalues_ry"].shape, (2, states))
                np.testing.assert_array_equal(
                    archive["spin_eigenvalues_ry"][1], self.result.spin[1].eigenvalues
                )
                np.testing.assert_array_equal(
                    archive["spin_density_e_per_bohr3"].sum(axis=0),
                    archive["density_e_per_bohr3"],
                )
                self.assertEqual(archive["eigenvalues_ry"].shape, (2 * states,))
                self.assertAlmostEqual(
                    float(archive["net_magnetic_moment"]),
                    self.result.net_magnetic_moment,
                )
                self.assertEqual(
                    archive["scf_spin_history"].shape, (self.result.iterations, 7)
                )


class EmptyChannelTests(unittest.TestCase):
    """H2+ with PBE: one electron, hence a channel without any.

    ``spn_pbe.f90`` drops the exchange energy where a spin density is zero
    and the Fortran loop does not converge for such a state; here it is an
    ordinary calculation.  Moved atoms continue from the potentials and
    subspaces of both channels.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.case = dict(
            input=[
                "Boundary_Sphere_Radius: 4.5",
                "Grid_Spacing: 0.6",
                "States_Num: 4",
                "Net_Charges: 1",
                "Max_Iter: 200",
                "Convergence_Criterion: 1.0e-8",
                "Eigensolver: chebff",
                "Correlation_Type: pbe",
                "Spin_Polarization: true",
                "Hartree_Boundary_Tolerance: off",
                "Atom_Types_Num: 1",
                "Atom_Type: H",
                "Local_Component: s",
                "begin Atom_Coord",
                "0.0 0.0 1.0",
                "0.0 0.0 -1.0",
                "end Atom_Coord",
            ],
            pseudopotentials={"H_POTRE.DAT": "NCPPs/PBE/H/PARSEC/H_POTRE.DAT"},
        )
        cls._directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._directory.cleanup)
        cls.translation = parse_parsec_input(
            write_case(cls.case, Path(cls._directory.name), parsec_boundary=False)
        )
        cls.system = prepare_single_point(cls.translation.problem)
        cls.result = run_scf(cls.system)

    def test_one_channel_holds_the_electron(self) -> None:
        result = self.result
        self.assertTrue(result.converged)
        self.assertEqual(result.electron_count, 1.0)
        self.assertAlmostEqual(abs(result.net_magnetic_moment), 1.0, places=12)
        occupied, empty = sorted(
            result.spin, key=lambda channel: -channel.electron_count
        )
        np.testing.assert_array_equal(occupied.occupations, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(empty.occupations, np.zeros(4))
        np.testing.assert_array_equal(empty.density, np.zeros(self.system.grid.size))
        self.assertTrue(np.all(np.isfinite(empty.xc_potential)))
        # No exchange binds the empty channel: its lowest level lies above.
        self.assertGreater(empty.eigenvalues[0], occupied.eigenvalues[0] + 0.1)
        # One electron: the exchange-correlation energy is far from zero,
        # where Fortran's exchange energy of this state would be.
        self.assertLess(result.energies.exchange_correlation, -0.4)
        last = result.history[-1]
        self.assertEqual(min(last.spin_weighted_residuals), 0.0)
        self.assertLess(last.weighted_residual, 1.0e-8)

    def test_moved_atoms_continue_from_both_channels(self) -> None:
        moved = (
            Atom("H", self.system.atoms[0].position + [0.0, 0.0, 0.05]),
            Atom("H", self.system.atoms[1].position - [0.0, 0.0, 0.05]),
        )
        next_system = update_geometry(self.system, moved)
        warm = warm_start_for_moved_atoms(self.system, self.result, next_system)
        self.assertEqual(len(warm.input_potential), 2)
        self.assertEqual(len(warm.eigensolver_state), 2)
        continued = run_scf(next_system, warm_start=warm)
        cold = run_scf(
            prepare_single_point(
                replace(self.translation.problem, atoms=moved, recenter_geometry=False)
            )
        )
        self.assertTrue(continued.converged and cold.converged)
        self.assertAlmostEqual(
            continued.energies.total, cold.energies.total, delta=5.0e-8
        )
        self.assertLess(continued.iterations, cold.iterations)
        # The first eigenproblem of the continued loop is no first solve.
        self.assertEqual(continued.spin[0].eigensolver_state.solves_completed,
                         self.result.spin[0].eigensolver_state.solves_completed
                         + continued.iterations)

    def test_hooks_without_a_spin_counterpart_are_refused(self) -> None:
        class OneSubspaceSolver:
            def __call__(self, *arguments, **options):
                raise AssertionError("must not be called")

        with self.assertRaisesRegex(NotImplementedError, "keeps one saved subspace"):
            run_scf(self.system, eigenproblem_solver=OneSubspaceSolver())
        with self.assertRaisesRegex(NotImplementedError, "symmetry-reduced SCF hooks"):
            run_scf(self.system, scalar_field_adapter=object())
        with self.assertRaisesRegex(NotImplementedError, "mixer_factory"):
            run_spin_scf(self.system, mixer_factory=lambda settings: None)
        unprepared = replace(self.system, initial_spin_densities=None)
        with self.assertRaisesRegex(ValueError, "no starting spin densities"):
            run_spin_scf(unprepared)


class OrbitalBlockTests(unittest.TestCase):
    """Orbitals a backend keeps on a device are joined on the host only."""

    def test_device_blocks_are_downloaded_into_one_host_array(self) -> None:
        class DeviceArray:
            def __init__(self, values):
                self.values = values
                self.shape = values.shape
                self.downloads = 0

            def get(self):
                self.downloads += 1
                return self.values

        generator = np.random.default_rng(3)
        up = DeviceArray(generator.normal(size=(7, 3)))
        down = DeviceArray(generator.normal(size=(7, 2)))
        blocks = SpinOrbitalBlocks(up, down)
        self.assertEqual(blocks.shape, (7, 5))
        self.assertEqual(blocks.ndim, 2)
        host = blocks.to_full_host()
        np.testing.assert_array_equal(host[:, :3], up.values)
        np.testing.assert_array_equal(host[:, 3:], down.values)
        self.assertEqual((up.downloads, down.downloads), (1, 1))
        # NumPy blocks are joined by NumPy itself.
        joined = SpinOrbitalBlocks(up.values, down.values).to_full_device()
        np.testing.assert_array_equal(joined, host)
        with self.assertRaisesRegex(TypeError, "cannot join orbital blocks"):
            blocks.to_full_device()


class SpinInputTests(unittest.TestCase):
    """What ``parsec.in`` says about spin, and what is refused."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.cases = load_cases()

    def parse(self, case, *extra: str, drop: tuple[str, ...] = ()):
        case = dict(case)
        case["input"] = [
            line for line in case["input"] if not line.startswith(drop or ("\0",))
        ]
        return parse_parsec_input(write_case(case, self.directory, extra=extra))

    def test_spin_polarization_and_default_starting_polarization(self) -> None:
        translation = self.parse(self.cases["ch3_ca"])
        problem = translation.problem
        self.assertTrue(problem.scf.spin_polarized)
        self.assertEqual(
            {
                symbol: potential.initial_spin_polarization
                for symbol, potential in problem.pseudopotentials.items()
            },
            {"C": 0.1, "H": 0.1},
        )
        unpolarized = self.parse(self.cases["h2o_nospin_ca"])
        self.assertFalse(unpolarized.problem.scf.spin_polarized)
        self.assertNotIn("spin_polarized", repr(unpolarized.problem.scf))
        self.assertNotIn(
            "initial_spin_polarization",
            repr(unpolarized.problem.pseudopotentials["O"]),
        )

    def test_initial_spin_polarization_beyond_one_is_refused(self) -> None:
        """Such a value makes one starting spin density negative."""

        case = dict(self.cases["ch3_ca"])
        lines = []
        for line in case["input"]:
            lines.append(line)
            if line == "Atom_Type: C":
                lines.append("Initial_Spin_Polarization: 1.5")
        case["input"] = lines
        with self.assertRaisesRegex(ValueError, "between -1 and 1"):
            parse_parsec_input(write_case(case, self.directory))

    def test_initial_spin_polarization_of_each_species(self) -> None:
        """The n-th line goes to the n-th ``Atom_Type``, as ESDF hands them out."""

        case = dict(self.cases["ch3_ca"])
        lines = []
        for line in case["input"]:
            lines.append(line)
            if line == "Atom_Type: C":
                lines.append("Initial_Spin_Polarization: 0.5")
            if line == "Atom_Type: H":
                lines.append("Initial_Spin_Polarization: -0.25")
        case["input"] = lines
        translation = parse_parsec_input(write_case(case, self.directory))
        potentials = translation.problem.pseudopotentials
        self.assertEqual(potentials["C"].initial_spin_polarization, 0.5)
        self.assertEqual(potentials["H"].initial_spin_polarization, -0.25)
        system = prepare_single_point(translation.problem)
        up, down = system.initial_spin_densities
        np.testing.assert_allclose(
            up + down, 2.0 * system.initial_density, rtol=0, atol=1.0e-14
        )
        # C holds four of the seven valence electrons, the three H one each:
        # the moment is 2 * (0.5 * 4 - 0.25 * 3) for the free atoms, scaled
        # with the density.  It is between the two one-species values.
        moment = system.grid.integrate(up) - system.grid.integrate(down)
        self.assertGreater(moment, 2.0 * -0.25 * 7.0)
        self.assertLess(moment, 2.0 * 0.5 * 7.0)
        self.assertAlmostEqual(moment, 2.5, delta=0.35)

    def test_default_functional_of_a_spin_polarized_run_is_refused(self) -> None:
        from parsec_python.Input import UnsupportedParsecOptionError

        # usrinputfile.F90 then takes the Perdew-Wang LSDA.
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Perdew-Wang LSDA"):
            self.parse(self.cases["ch3_ca"], drop=("Correlation_Type",))

    def test_imported_starting_density_is_refused(self) -> None:
        translation = self.parse(
            self.cases["h_fixed_ca"],
            "Initial_Density: file",
            "ML_Density_File: density.npz",
        )
        with self.assertRaisesRegex(NotImplementedError, "holds no spin densities"):
            prepare_single_point(translation.problem)

    def test_pseudopotentials_of_the_fixture_exist(self) -> None:
        for case in self.cases.values():
            for source in case["pseudopotentials"].values():
                self.assertTrue(pseudopotential_file(source).is_file(), source)


if __name__ == "__main__":
    unittest.main()
