"""Spin-polarized loops in the symmetry sectors (``Spin_Symmetry_Sectors``).

The two channels of a spin-polarized CuPy run are solved in the representation
sectors of the point group, with every scalar field on one value per orbit
(:mod:`parsec_python.acceleration.SCF.spin_sectors`).  The orbit algebra must
be that of the full grid; a complete run must reach the full-grid result
(energies, levels, forces); a run whose occupations end fractional (the oxygen
atom at 80 K) must leave the sectors and give the full-grid run; the keyword is
read, defaulted, completed and refused as documented.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from parsec_python.Energy import spin_total_energy
from parsec_python.Grid.cluster import build_cluster_grid
from parsec_python.Input import parse_parsec_input
from parsec_python.Input.complete_input import complete_input_text
from parsec_python.Mixer import make_mixer, spin_potential_residual_metrics
from parsec_python.acceleration.SCF.spin_sectors import (
    SPIN_SECTOR_OCCUPATION_TOLERANCE,
    SpinSectorReducer,
    SpinSectorsAbandoned,
    fractional_occupations,
)
from parsec_python.acceleration.SCF.symmetry_fields import SymmetrySCFReducer
from parsec_python.acceleration.backends.cupy import cupy_available
from parsec_python.acceleration.backends.native import native_available
from parsec_python.models import Atom, GridSettings, MixingSettings, SCFSettings
from parsec_python.tests.spin_reference import load_cases, write_case


def _reducer(size: int, seed: int = 3) -> SymmetrySCFReducer:
    """A reducer on ``size`` orbits with the orbit sizes of a group of order eight."""

    rng = np.random.default_rng(seed)
    multiplicities = rng.choice(np.array([1, 2, 4, 8]), size=size)
    return SymmetrySCFReducer(
        SimpleNamespace(
            multiplicities=multiplicities,
            full_to_wedge=np.repeat(np.arange(size), multiplicities),
            wedge_size=size,
            full_size=int(multiplicities.sum()),
        )
    )


def _channel(values, occupations, representations=None):
    values = np.asarray(values, dtype=float)
    return SimpleNamespace(
        eigenvalues=values,
        occupations=np.asarray(occupations, dtype=float),
        representations=np.ones(values.size, dtype=np.int32)
        if representations is None
        else np.asarray(representations, dtype=np.int32),
    )


class OrbitAlgebraTests(unittest.TestCase):
    """The hooks of the spin loop on the orbits give the full-grid numbers."""

    def setUp(self) -> None:
        self.size = 17
        self.base = _reducer(self.size)
        self.adapter = SpinSectorReducer(self.base, hybrid=False)
        rng = np.random.default_rng(11)
        self.fields = {
            name: [self.base.field(rng.standard_normal(self.size)) for _ in range(2)]
            for name in ("input", "output", "xc")
        }
        self.densities = [self.base.field(np.abs(rng.standard_normal(self.size))) for _ in range(2)]
        self.ionic = self.base.field(rng.standard_normal(self.size))
        self.hartree = self.base.field(rng.standard_normal(self.size))

    def full(self, field):
        return self.base.to_full(field)

    def test_residual_norms_per_channel(self) -> None:
        for electrons in ((3.0, 2.0), (3.0, 0.0)):
            wedge = self.adapter.spin_potential_residual_metrics(
                self.fields["input"], self.fields["output"], self.densities, 0.3, electrons
            )
            full = spin_potential_residual_metrics(
                [self.full(item) for item in self.fields["input"]],
                [self.full(item) for item in self.fields["output"]],
                [self.full(item) for item in self.densities],
                0.3,
                electrons,
            )
            np.testing.assert_allclose(wedge.weighted_by_spin, full.weighted_by_spin, rtol=1e-13, atol=0)
            np.testing.assert_allclose(wedge.plain_by_spin, full.plain_by_spin, rtol=1e-13, atol=0)

    def test_spin_energy(self) -> None:
        eigenvalues = (np.array([-1.0, -0.5, 0.2]), np.array([-0.9, 0.1, 0.3]))
        occupations = (np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        for extra in ({}, {"hubbard_energy": 0.25}, {"alpha_z_energy": 0.125}):
            wedge = self.adapter.spin_total_energy(
                eigenvalues, occupations, self.densities, self.fields["input"], self.ionic, self.hartree,
                self.fields["xc"], -3.0, 2.0, 0.3, **extra,
            )
            full = spin_total_energy(
                eigenvalues,
                occupations,
                [self.full(item) for item in self.densities],
                [self.full(item) for item in self.fields["input"]],
                self.full(self.ionic),
                self.full(self.hartree),
                [self.full(item) for item in self.fields["xc"]],
                -3.0,
                2.0,
                0.3,
                **extra,
            )
            self.assertIs(type(wedge), type(full))
            for name in ("eigenvalue", "hartree", "integral_vxc_rho", "electron_ion", "electronic", "total"):
                self.assertAlmostEqual(getattr(wedge, name), getattr(full, name), delta=1e-12, msg=name)

    def test_mixers_of_joined_channels_are_the_full_grid_mixers(self) -> None:
        rng = np.random.default_rng(5)
        size = self.size
        matrix = 0.4 * rng.standard_normal((2 * size, 2 * size)) / np.sqrt(size)
        expand = self.base.reduction.full_to_wedge
        for options in ({"method": "anderson"}, {"method": "broyden"}, {"method": "msecant1"}):
            settings = MixingSettings(**options)
            wedge = self.adapter.spin_mixer(settings)
            full = make_mixer(settings)
            x = [self.base.field(np.zeros(size)), self.base.field(np.zeros(size))]
            x_full = np.zeros(2 * expand.size)
            for iteration in range(1, 10):
                joined = self.adapter.join_channels(x)
                output = np.tanh(matrix @ joined) + 0.1
                output_fields = self.adapter.split_channels(output)
                mixed = self.adapter.split_channels(
                    wedge.mix(joined, self.adapter.join_channels(output_fields), iteration=iteration)
                )
                output_full = np.concatenate([self.full(item) for item in output_fields])
                x_full = full.mix(x_full, output_full, iteration=iteration)
                # (The least-squares step of a short history rounds differently
                # in the two orders of summation.)
                np.testing.assert_allclose(
                    np.concatenate([self.full(item) for item in mixed]), x_full, rtol=0, atol=1e-9,
                    err_msg=str(options),
                )
                x = mixed
                x_full = np.concatenate([self.full(item) for item in mixed])

    def test_hooks_are_those_the_spin_loop_takes(self) -> None:
        hooks = self.adapter.spin_hooks()
        self.assertIs(hooks["scalar_field_adapter"], self.adapter)
        self.assertEqual(hooks["mixer_factory"], self.adapter.spin_mixer)
        from parsec_python.SCF.spin_polarized import run_spin_scf

        # The hooks of one channel are refused (the reducer of an unpolarized loop).
        with self.assertRaisesRegex(NotImplementedError, "symmetry-reduced SCF hooks"):
            run_spin_scf(
                SimpleNamespace(initial_spin_densities=None),
                mixer_factory=self.base.mixer,
                scalar_field_adapter=self.base,
            )


class OccupationGuardTests(unittest.TestCase):
    def test_integer_occupations_pass(self) -> None:
        channels = (
            _channel([-1.0, -0.5, -0.5, 0.2], [1.0, 1.0, 1.0, 1.0e-6]),
            _channel([-0.9, -0.4, 0.1], [1.0, 1.0 - 5.0e-4, 0.0]),
        )
        self.assertIsNone(fractional_occupations(channels))
        SpinSectorReducer(_reducer(4), hybrid=True).check_spin_result(SimpleNamespace(spin=channels))

    def test_a_partly_filled_degenerate_shell_is_reported(self) -> None:
        third = 1.0 / 3.0
        channels = (
            _channel([-1.6, -0.7, -0.7, -0.7], [1.0, 1.0, 1.0, 1.0], [1, 6, 7, 8]),
            _channel([-1.5, -0.53, -0.53, -0.53, 0.2], [1.0, third, third, third, 0.0], [1, 7, 8, 6, 1]),
        )
        report = fractional_occupations(channels)
        self.assertAlmostEqual(report.largest, third)
        self.assertEqual(report.tolerance, SPIN_SECTOR_OCCUPATION_TOLERANCE)
        self.assertEqual(len(report.lines), 1)
        self.assertIn("down channel: 3 level(s)", report.lines[0])
        self.assertIn("1.0000 electrons", report.lines[0])
        self.assertIn("sectors 7 8 6", report.lines[0])
        for hybrid, final, stage in (
            (False, False, "the SCF loop"),
            (False, True, "the SCF loop"),
            (True, False, "the semilocal start of the Fock loop"),
            (True, True, "the end of the Fock loop"),
        ):
            adapter = SpinSectorReducer(_reducer(4), hybrid=hybrid)
            with self.assertRaises(SpinSectorsAbandoned) as caught:
                # An accelerated result wraps the SCF result.
                adapter.check_spin_result(SimpleNamespace(result=SimpleNamespace(spin=channels)), final=final)
            self.assertEqual(caught.exception.stage, stage)
            self.assertIn("fractional occupations", str(caught.exception))

    def test_partly_filled_multiplets_of_the_start(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import partly_filled_multiplets

        # The O atom: the minority p shell holds one electron (Fermi-Dirac thirds or an integer aufbau).
        values = ([-1.6, -0.7, -0.7, -0.7, 0.2], [-1.5, -0.53, -0.53, -0.53, 0.2])
        sectors = ([1, 6, 7, 8, 1], [1, 6, 7, 8, 1])
        for down in ([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]):
            report = partly_filled_multiplets(values, ([1.0, 1.0, 1.0, 1.0, 0.0], down), sectors, fixed=False)
            self.assertEqual(len(report.lines), 1)
            self.assertIn("down channel: 3 levels", report.lines[0])
        # Fixed equal thirds keep an invariant density: kept; fixed integer filling is not.
        thirds = ([1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
        self.assertIsNone(partly_filled_multiplets(values, thirds, sectors, fixed=True))
        integer = ([1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertIsNotNone(partly_filled_multiplets(values, integer, sectors, fixed=True))
        # A d shell split by the grid into e_g and t_2g (Fe at h 0.30: 18 mRy) is one multiplet;
        # full and empty multiplets, and a filled shell with a gap, pass.
        d = [-1.1088, -1.1088, -1.0906, -1.0906, -1.0906, -0.5]
        report = partly_filled_multiplets(
            (d, d), ([1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]), ([1, 1, 2, 3, 4, 1],) * 2, fixed=False, width=0.02
        )
        self.assertIn("down channel: 5 levels", report.lines[0])
        self.assertIsNone(
            partly_filled_multiplets((d, d), ([1, 1, 1, 1, 1, 0], [0] * 6), ([1, 1, 2, 3, 4, 1],) * 2, fixed=False)
        )
        # The Sc atom at h 0.30 (first iteration): t_2g filled, 4s and e_g within 11.5 mRy above, in four
        # sectors: three electrons in six levels.
        sc = [-0.42423461, -0.42423461, -0.42423460, -0.41688434, -0.41268802, -0.41268802, -0.1795]
        report = partly_filled_multiplets(
            (sc, sc), ([0.9996, 0.9996, 0.9996, 0.0012, 0, 0, 0], [0] * 7), ([2, 3, 4, 1, 1, 1, 8],) * 2, fixed=False
        )
        self.assertIn("up channel: 6 levels", report.lines[0])

    def test_a_start_report_counts_once_its_levels_are_converged(self) -> None:
        """The spin loop asks for the report, solves the start again while it moves, then abandons (run_spin_scf)."""

        adapter = SpinSectorReducer(_reducer(4), hybrid=False)
        values = ([-1.6, -0.7, -0.7, -0.7, 0.2], [-1.5, -0.53, -0.53, -0.53, 0.2])
        filled = ([1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0])
        sectors = ([1, 6, 7, 8, 1],) * 2
        report = adapter.start_spectrum_report(values, filled, sectors)
        self.assertIsNotNone(report)
        self.assertGreaterEqual(adapter.start_spectrum_passes, 1)
        self.assertLess(adapter.start_level_change, 1e-3)
        with self.assertRaises(SpinSectorsAbandoned) as caught:
            adapter.abandon_start(report, 2)
        self.assertEqual(caught.exception.stage, "the start Hamiltonian")
        self.assertIn("after 2 further solve(s)", caught.exception.report.lines[-1])
        self.assertIsNone(adapter.start_spectrum_report(values, ([1.0] * 4 + [0.0], [1.0] * 4 + [0.0]), sectors))

    def test_a_fermi_dirac_tail_is_no_partial_filling(self) -> None:
        """Nine empty levels in four sectors holding 3.4e-3 electrons (C459H204+ at 500 K) are not partly filled."""

        from parsec_python.acceleration.SCF.spin_sectors import partly_filled_multiplets

        levels = np.concatenate(([-0.5, -0.45], -0.33 + 0.003 * np.arange(9)))
        sectors = np.array([1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1])
        tail = np.concatenate(([1.0, 1.0], np.full(9, 0.0034 / 9)))
        self.assertIsNone(partly_filled_multiplets((levels, levels), (tail, tail), (sectors, sectors), fixed=False))
        channels = (_channel(levels, tail, sectors), _channel(levels, tail, sectors))
        self.assertIsNone(fractional_occupations(channels))
        # A third of an electron per level is a partly filled shell.
        third = np.concatenate(([1.0, 1.0], np.full(3, 1 / 3), np.zeros(6)))
        self.assertIsNotNone(partly_filled_multiplets((levels, levels), (third, third), (sectors, sectors), fixed=False))

    def test_what_is_not_a_multiplet_of_the_sectors(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import partly_filled_multiplets

        # Partners in one sector (an e_g pair of Mn or Co, both in the sector of x^2, y^2, z^2) mix there freely.
        pair = [-0.8, -0.5, -0.5, 0.1]
        self.assertIsNone(
            partly_filled_multiplets((pair, pair), ([1, 0.5, 0.5, 0], [1, 0, 0, 0]), ([1, 1, 1, 2],) * 2, fixed=False)
        )
        # A band: levels 5 mRy apart over 0.25 Ry in all sectors, the Fermi level inside (the first filter of a
        # large cluster) is not a multiplet; the same levels around a degenerate pair 0.05 Ry away from them
        # leave the pair one.
        band = -0.3 + 0.005 * np.arange(50)
        sectors = np.arange(50) % 8 + 1
        filled = 1.0 / (1.0 + np.exp((band - band[25]) / 0.003))
        self.assertIsNone(partly_filled_multiplets((band, band), (filled, filled), (sectors, sectors), fixed=False))
        levels = np.concatenate((band, [band[-1] + 0.05] * 2))
        occupied = np.concatenate((np.ones(50), [0.5, 0.5]))
        labels = np.concatenate((sectors, [6, 7]))
        report = partly_filled_multiplets((levels, levels), (occupied, occupied), (labels, labels), fixed=False)
        self.assertEqual(len(report.lines), 2)
        self.assertIn("2 levels", report.lines[0])

    def test_fractional_occupations_the_sectors_do_not_constrain(self) -> None:
        third = 1.0 / 3.0
        # A fractional single level (a small gap at the Fermi level) and a fractional pair within one sector pass.
        single = (_channel([-1.0, -0.5, -0.49], [1.0, 0.6, 0.4], [1, 2, 3]), _channel([-0.9], [1.0], [1]))
        self.assertIsNone(fractional_occupations(single))
        within = (_channel([-1.0, -0.5, -0.5], [1.0, 0.5, 0.5], [1, 2, 2]), _channel([-0.9], [1.0], [1]))
        self.assertIsNone(fractional_occupations(within))
        across = (_channel([-1.0, -0.5, -0.5, -0.5], [1.0, third, third, third], [1, 2, 3, 4]), _channel([-0.9], [1.0], [1]))
        self.assertAlmostEqual(fractional_occupations(across).largest, third)

    def test_the_operations_of_the_atoms_on_grids_with_less_symmetry(self) -> None:
        """Test 2 tests generators of the group the grid keeps (a domain or grid shift that singles out y)."""

        from parsec_python.Forces.symmetry import atom_symmetry_operations
        from parsec_python.acceleration.SCF.spin_sectors import (
            _grid_images,
            _key,
            atom_group_images,
            density_symmetry_breaking,
        )

        def closure(matrices):
            elements = {_key(np.eye(3)): np.eye(3)}
            grown = True
            while grown:
                grown = False
                for element in list(elements.values()):
                    for matrix in matrices:
                        product = element @ matrix
                        if _key(product) not in elements:
                            elements[_key(product)] = product
                            grown = True
            return set(elements)

        domains = {
            "sphere": dict(radius=4.0),
            "cylinder along y": dict(radius=4.0, domain_shape="cylinder", domain_parameters=(4.0, 6.0, 2.0)),
            "cylinder along z": dict(radius=4.0, domain_shape="cylinder", domain_parameters=(4.0, 6.0, 3.0)),
            "box 8 x 6 x 8": dict(radius=4.0, domain_shape="box", box_lengths=(8.0, 6.0, 8.0)),
            "ellipsoid 4 3.5 4": dict(radius=4.0, domain_shape="ellipsoid", domain_parameters=(4.0, 3.5, 4.0)),
            "Grid_Shift 0.5 0 0.5": dict(radius=4.0, shift=(0.5, 0.0, 0.5)),
        }
        methane = (
            Atom("C", (0.0, 0.0, 0.0)),
            Atom("H", (1.2, 1.2, 1.2)),
            Atom("H", (-1.2, -1.2, 1.2)),
            Atom("H", (-1.2, 1.2, -1.2)),
            Atom("H", (1.2, -1.2, -1.2)),
        )
        for atoms in ((Atom("V", (0.0, 0.0, 0.0)),), methane):
            for name, settings in domains.items():
                grid = build_cluster_grid(GridSettings(spacing=0.5, expansion_order=4, **settings))
                kept = {
                    _key(np.rint(operation.matrix))
                    for operation in atom_symmetry_operations(atoms)
                    if _grid_images(grid, np.rint(operation.matrix)) is not None
                }
                tested = atom_group_images(grid, atoms)
                self.assertEqual(closure([matrix for matrix, _ in tested]), kept, name)
        # The V atom of the review in a cylinder along y: one axis-aligned d or p orbital in the x-z plane
        # breaks the swap of x and z, which the grid keeps (no operation was tested before).
        grid = build_cluster_grid(GridSettings(spacing=0.5, expansion_order=4, **domains["cylinder along y"]))
        x = grid.coordinates
        radial = np.exp(-np.sum(x * x, axis=1))
        p_x = x[:, 0] ** 2 * radial
        report = density_symmetry_breaking([p_x / (grid.volume_element * np.sum(p_x)), radial], grid, (Atom("V", (0, 0, 0)),))
        self.assertGreater(report.largest, 0.5)
        pair = (x[:, 0] ** 2 + x[:, 2] ** 2) * radial
        self.assertIsNone(density_symmetry_breaking([pair, radial], grid, (Atom("V", (0, 0, 0)),)))

    def test_density_symmetry_of_the_atoms(self) -> None:
        from parsec_python.acceleration.SCF.spin_sectors import density_symmetry_breaking

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=4, shift=(0.5, 0.5, 0.5)))
        x = grid.coordinates
        radial = np.exp(-np.sum(x * x, axis=1))
        atom = (Atom("O", (0.0, 0.0, 0.0)),)
        # Spherical channels pass; one axis-aligned p orbital breaks the 90-degree rotations of the atom.
        self.assertIsNone(density_symmetry_breaking([radial, 0.5 * radial], grid, atom))
        p_x = x[:, 0] ** 2 * radial
        report = density_symmetry_breaking([radial, p_x / (grid.volume_element * np.sum(p_x))], grid, atom)
        self.assertGreater(report.largest, 0.5)
        self.assertIn("down channel", report.lines[0])
        # A molecule along z: p_x + p_y is invariant under its fourfold axis, p_x alone is not.
        molecule = (Atom("O", (0.0, 0.0, 1.0)), Atom("O", (0.0, 0.0, -1.0)))
        pair = (x[:, 0] ** 2 + x[:, 1] ** 2) * radial
        self.assertIsNone(density_symmetry_breaking([pair, radial], grid, molecule))
        self.assertIsNotNone(density_symmetry_breaking([p_x, radial], grid, molecule))
        # Without a symmetry beyond the reflections there is nothing to break.
        bent = (Atom("O", (0.0, 0.0, 0.0)), Atom("H", (1.4, 0.0, 1.1)), Atom("H", (-1.4, 0.0, 1.1)))
        self.assertIsNone(density_symmetry_breaking([p_x, radial], grid, bent))

    def test_unpolarized_results_are_not_checked(self) -> None:
        SpinSectorReducer(_reducer(4), hybrid=False).check_spin_result(SimpleNamespace(spin=None))


class KeywordTests(unittest.TestCase):
    """``Spin_Symmetry_Sectors``: defaults, stored values, the completed input, refusals."""

    def translate(self, lines, *, spin=True, functional="pbe"):
        case = dict(load_cases()["n_ca"])
        text = [
            line
            for line in case["input"]
            if not line.startswith(("Correlation_Type", "Spin_Polarization"))
        ] + [f"Correlation_Type: {functional}", f"Spin_Polarization: {'.true.' if spin else '.false.'}", *lines]
        case["input"] = text
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return parse_parsec_input(write_case(case, directory)), directory

    def test_defaults_and_stored_values(self) -> None:
        for functional, given, stored, mode in (
            ("pbe", None, None, "auto"),
            ("pbe", "auto", None, "auto"),
            ("pbe", "off", "off", "off"),
            ("pbe0", None, None, "auto"),
            ("pbe0", "AUTO", None, "auto"),
            ("pbe0", "off", "off", "off"),
        ):
            lines = [] if given is None else [f"Spin_Symmetry_Sectors: {given}"]
            translation, _ = self.translate(lines, functional=functional)
            scf = translation.problem.scf
            self.assertEqual(scf.spin_symmetry_sectors, stored, (functional, given))
            self.assertEqual(scf.spin_symmetry_sector_mode, mode, (functional, given))
            # Nothing in the text of the settings (cache keys) at the default.
            self.assertEqual("spin_symmetry_sectors" in repr(scf), stored is not None)

    def test_completed_input_reads_back_as_the_same_problem(self) -> None:
        for functional, lines, written in (
            ("pbe0", [], "Spin_Symmetry_Sectors  auto"),
            ("pbe0", ["Spin_Symmetry_Sectors: off"], "Spin_Symmetry_Sectors  off"),
            ("pbe", ["Spin_Symmetry_Sectors: off"], "Spin_Symmetry_Sectors  off"),
            ("pbe", ["Spin_Symmetry_Sectors: auto"], None),
            ("pbe", [], None),
        ):
            translation, directory = self.translate(lines, functional=functional)
            text = complete_input_text(translation, {}, {})
            if written is None:
                self.assertNotIn("Spin_Symmetry_Sectors", text)
            else:
                self.assertIn(written, text)
            (directory / "complete.in").write_text(text)
            again = parse_parsec_input(directory / "complete.in")
            self.assertEqual(again.problem.scf, translation.problem.scf)

    def test_the_full_grid_input_of_a_fallback(self) -> None:
        """The input a run that leaves the sectors is prepared again from: as given, or at moved atoms."""

        from parsec_python.acceleration.driver import _SpinSectorFallback

        translation, _ = self.translate([], functional="pbe0")
        problem = translation.problem
        prepared_atoms = tuple(replace(atom, position=np.asarray(atom.position) + 0.25) for atom in problem.atoms)
        fallback = _SpinSectorFallback(
            problem, prepared_atoms, backend="auto", symmetry="auto", symmetry_cache_directory=None
        )
        same = fallback._problem_of(SimpleNamespace(reference=SimpleNamespace(atoms=prepared_atoms)))
        self.assertEqual(same.scf.spin_symmetry_sectors, "off")
        self.assertEqual(same.scf.spin_symmetry_sector_mode, "off")
        def positions(atoms):
            return np.array([np.asarray(atom.position, dtype=float) for atom in atoms])

        np.testing.assert_array_equal(positions(same.atoms), positions(problem.atoms))
        self.assertEqual(same.recenter_geometry, problem.recenter_geometry)
        moved_atoms = tuple(replace(atom, position=np.asarray(atom.position) + 0.5) for atom in problem.atoms)
        moved = fallback._problem_of(SimpleNamespace(reference=SimpleNamespace(atoms=moved_atoms)))
        np.testing.assert_array_equal(positions(moved.atoms), positions(moved_atoms))
        self.assertFalse(moved.recenter_geometry)
        self.assertEqual(moved.scf.spin_symmetry_sectors, "off")

    def test_refusals(self) -> None:
        from parsec_python.Input.parsec_input import ParsecInputError

        with self.assertRaisesRegex(ParsecInputError, "auto or off"):
            self.translate(["Spin_Symmetry_Sectors: on"])
        translation, _ = self.translate(["Spin_Symmetry_Sectors: auto"], spin=False)
        self.assertIsNone(translation.problem.scf.spin_symmetry_sectors)
        self.assertTrue(any("Spin_Symmetry_Sectors" in warning for warning in translation.warnings))
        with self.assertRaises(ValueError):
            SCFSettings(spin_symmetry_sectors="on")


@unittest.skipUnless(cupy_available(), "CUDA runtime is unavailable")
class DeviceSpinXCOnOrbitsTests(unittest.TestCase):
    """The device spin functionals on the orbit fields of the sectors."""

    def test_orbit_fields_give_the_full_grid_functional(self) -> None:
        from parsec_python.V_xc.hybrid import HYBRID_FUNCTIONALS, hybrid_semilocal_xc_spin
        from parsec_python.V_xc.pbe_spin import pbe_spin
        from parsec_python.V_xc.selection import evaluate_spin_xc
        from parsec_python.acceleration.Symmetry.axis_reflection import AxisReflectionReduction
        from parsec_python.acceleration.V_xc.device_hybrid_xc import DeviceSpinHybridXCEvaluator
        from parsec_python.acceleration.V_xc.device_xc import DeviceSpinXCEvaluator

        grid = build_cluster_grid(GridSettings(spacing=0.5, radius=4.5, expansion_order=8, shift=(0.5, 0.5, 0.5)))
        atoms = (Atom("O", (0.0, 0.0, 1.1)), Atom("O", (0.0, 0.0, -1.1)))
        reducer = SymmetrySCFReducer(AxisReflectionReduction.detect(grid, atoms))
        self.assertEqual(reducer.reduction.group_order, 8)
        x = grid.coordinates
        bumps = [np.exp(-np.sum((x - np.asarray(atom.position)) ** 2, axis=1) / width) for atom in atoms for width in (0.6, 1.4)]
        up = reducer.from_full(0.6 * bumps[0] + 0.6 * bumps[2] + 0.3 * bumps[1] + 0.3 * bumps[3])
        down = reducer.from_full(0.4 * bumps[0] + 0.4 * bumps[2] + 0.1 * bumps[1] + 0.1 * bumps[3])
        core = reducer.to_full(reducer.from_full(0.2 * bumps[0] + 0.2 * bumps[2]))
        cases = [
            (
                f"spin {functional}",
                (
                    pbe_spin(reducer.to_full(up), reducer.to_full(down), grid, core)
                    if functional == "pbe"
                    else evaluate_spin_xc(functional, reducer.to_full(up), reducer.to_full(down), grid, core)
                ),
                DeviceSpinXCEvaluator(functional, grid, core_density=core, reducer=reducer)(up, down),
            )
            for functional in ("pbe", "pw91", "lb94", "pw92")
        ]
        for label in ("pbe0", "hse06", "b3lyp"):
            functional = HYBRID_FUNCTIONALS[label]
            cases.append(
                (
                    f"hybrid {label}",
                    hybrid_semilocal_xc_spin(functional, reducer.to_full(up), reducer.to_full(down), grid, core),
                    DeviceSpinHybridXCEvaluator(functional, grid, core_density=core, reducer=reducer)(up, down),
                )
            )
        for label, reference, value in cases:
            for name in ("potential_up", "potential_down", "energy_density"):
                expected = reducer.wedge_values(getattr(reference, name))
                got = getattr(value, name).values
                scale = max(float(np.max(np.abs(expected))), 1e-300)
                self.assertLess(float(np.max(np.abs(expected - got))) / scale, 1e-11, (label, name))
            self.assertAlmostEqual(
                value.total_energy, reference.total_energy, delta=1e-12 * abs(reference.total_energy), msg=label
            )


def _solve(name, mode, *, extra=(), drop=(), records=None, symmetry="auto"):
    """Prepare and solve one fixture case with ``Spin_Symmetry_Sectors: mode`` on CuPy."""

    from parsec_python.acceleration import driver

    case = dict(load_cases()[name])
    case["input"] = [line for line in case["input"] if not line.startswith(tuple(drop))]
    if any(prefix.startswith("Fermi_Temp") for prefix in drop):
        case.pop("occup", None)
    # The directory stays until the end of the test: a run that leaves the
    # sectors prepares again from the input files.
    directory = Path(tempfile.mkdtemp())
    translation = parse_parsec_input(
        write_case(case, directory, extra=tuple(extra) + (f"Spin_Symmetry_Sectors: {mode}",))
    )
    # The route of main.py: CuPy orbitals, the native (wedge) Hartree solver.
    system = driver.prepare_single_point(translation.problem, backend="auto", symmetry=symmetry)
    system.materialize_final_wavefunctions = True
    result = driver.run_scf(system, callback=None if records is None else records.append)
    return directory, system, result


@unittest.skipUnless(
    cupy_available() and native_available(), "CUDA runtime or native extension is unavailable"
)
class SectorRunTests(unittest.TestCase):
    """Complete runs in the sectors against the full grid."""

    def solve(self, *arguments, **options):
        directory, system, result = _solve(*arguments, **options)
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        return system, result

    def assert_same_run(self, full, sectors, *, levels=2e-8, forces=None) -> None:
        self.assertTrue(full.converged and sectors.converged)
        self.assertAlmostEqual(sectors.energies.total, full.energies.total, delta=1e-9)
        for index in range(2):
            np.testing.assert_allclose(sectors.spin[index].eigenvalues, full.spin[index].eigenvalues, atol=levels)
            np.testing.assert_array_equal(sectors.spin[index].occupations, full.spin[index].occupations)
            np.testing.assert_allclose(sectors.spin[index].density, full.spin[index].density, atol=1e-7)
        if forces is not None:
            np.testing.assert_allclose(forces[1], forces[0], atol=1e-7)

    def test_oxygen_molecule(self) -> None:
        from parsec_python.acceleration import driver

        full_system, full = self.solve("o2_ca", "off")
        system, sectors = self.solve("o2_ca", "auto")
        details = dict(sectors.backend.details)
        self.assertTrue(details["orbital_symmetry"].startswith("CuPy real"))
        self.assertIn("8 sectors", details["spin_symmetry_sectors"])
        self.assertEqual(sectors.backend.selected, "cupy")
        self.assertEqual(details["hartree_symmetry"], "normalized Cartesian axis-reflection wedge")
        self.assertIn("up:", details["orbital_sector_final_state_counts"])
        forces = (
            driver.compute_forces(full_system, full).total,
            driver.compute_forces(system, sectors).total,
        )
        self.assert_same_run(full, sectors, forces=forces)
        # The channels have subspaces of their own on the shared sector operators.
        solver = system.eigenproblem_solver
        up, down = solver.for_spin_channel(0), solver.for_spin_channel(1)
        self.assertIsNot(up.state, down.state)
        self.assertIs(sectors.spin[0].eigensolver_state, up.state)
        self.assertIs(sectors.spin[1].eigensolver_state, down.state)
        # The orbitals as one host matrix, up first, normalized.
        self.assertEqual(sectors.wavefunctions.shape[1], 2 * sectors.spin[0].eigenvalues.size)
        norms = np.sum(sectors.wavefunctions * sectors.wavefunctions, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
        # A second loop of the same system starts both channels afresh.
        again = driver.run_scf(system)
        self.assertAlmostEqual(again.energies.total, sectors.energies.total, delta=1e-9)

    def test_a_fallback_after_the_input_files_are_gone(self) -> None:
        """The input directory is removed after the preparation (a temporary one): the full grid still comes."""

        from parsec_python.acceleration import driver

        case = dict(load_cases()["o_spherical_ca"])
        case["input"] = [line for line in case["input"] if not line.startswith("Fermi_Temp")]
        case.pop("occup", None)
        with tempfile.TemporaryDirectory() as directory:
            translation = parse_parsec_input(
                write_case(case, Path(directory), extra=("Spin_Symmetry_Sectors: auto",))
            )
            system = driver.prepare_single_point(translation.problem, backend="auto", symmetry="auto")
        result = driver.run_scf(system)
        _, full = self.solve("o_spherical_ca", "off", drop=("Fermi_Temp",))
        self.assertIn("left the sectors", dict(result.backend.details)["spin_symmetry_sectors"])
        self.assertEqual(result.energies.total, full.energies.total)

    def test_a_fallback_after_an_ionic_move_keeps_the_warm_start(self) -> None:
        """A sector loop from a warm start (an ionic step) that leaves the sectors continues from its potentials."""

        from parsec_python.SCF import SCFWarmStart
        from parsec_python.acceleration import driver
        from parsec_python.models import SpinSectorFallbackRecord

        _, full = self.solve("o_spherical_ca", "off", drop=("Fermi_Temp",))
        case = dict(load_cases()["o_spherical_ca"])
        case["input"] = [line for line in case["input"] if not line.startswith("Fermi_Temp")]
        case.pop("occup", None)
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        translation = parse_parsec_input(write_case(case, directory, extra=("Spin_Symmetry_Sectors: auto",)))
        system = driver.prepare_single_point(translation.problem, backend="auto", symmetry="auto")
        # The potentials of a converged full-grid run of the same structure (as after a step of zero length).
        warm = SCFWarmStart(
            input_potential=tuple(np.asarray(channel.next_effective_potential) for channel in full.spin),
            hartree_potential=np.asarray(full.hartree_potential),
            eigensolver_state=None,
        )
        records = []
        result = driver.run_scf(system, warm_start=warm, callback=records.append)
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        self.assertNotEqual(notes[0].stage, "the start Hamiltonian")
        self.assertIn("left the sectors", dict(result.backend.details)["spin_symmetry_sectors"])
        # The full grid continues from the potentials: the off state, in fewer iterations than from the atoms.
        self.assertAlmostEqual(result.energies.total, full.energies.total, delta=1e-7)
        self.assertLess(result.iterations, full.iterations)
        self.assertIsNone(driver._SpinSectorFallback.full_grid_warm_start(None))
        kept = driver._SpinSectorFallback.full_grid_warm_start(replace(warm, mixer=object(), first_iteration_degree=7))
        self.assertIsNone(kept.mixer)
        self.assertIsNone(kept.first_iteration_degree)
        self.assertIs(kept.hartree_potential, warm.hartree_potential)

    def test_an_asymmetry_that_shrinks_with_the_residual_is_noise(self) -> None:
        """Test 2 at the convergence of a loop: a tighter criterion while the asymmetry shrinks (noise of a loose SCF)."""

        from parsec_python.acceleration.SCF import spin_sectors

        _, plain = self.solve("o2_ca", "auto")
        for values, measured in (([0.05, 0.02, 0.004], 3), ([0.05, 0.04], 2)):
            sequence = iter(values)
            calls = []

            def measure(adapter, densities, sequence=sequence, calls=calls):
                calls.append(len(densities))
                return next(sequence, 0.0)

            with mock.patch.object(spin_sectors.SpinSectorReducer, "density_symmetry_measure", measure):
                _, tightened = self.solve("o2_ca", "auto")
            # Shrinking: two tightenings, then accepted; not shrinking by half: one, then the end check decides
            # (the real density of O2 is symmetric, so it stays in the sectors here).
            self.assertEqual(len(calls), measured)
            self.assertTrue(tightened.converged)
            self.assertGreater(tightened.iterations, plain.iterations)
            self.assertIn("8 sectors", dict(tightened.backend.details)["spin_symmetry_sectors"])
            self.assertAlmostEqual(tightened.energies.total, plain.energies.total, delta=1e-8)

    def test_oxygen_atom_at_80_kelvin_leaves_the_sectors(self) -> None:
        from parsec_python.models import SpinSectorFallbackRecord

        _, full = self.solve("o_spherical_ca", "off", drop=("Fermi_Temp",))
        records = []
        system, result = self.solve("o_spherical_ca", "auto", drop=("Fermi_Temp",), records=records)
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual(len(notes), 1)
        # The start Hamiltonian already has the partly filled p shell (one third in each level).
        self.assertEqual(notes[0].stage, "the start Hamiltonian")
        self.assertEqual(notes[0].reason, "a partly filled multiplet")
        self.assertAlmostEqual(notes[0].largest_fractional_occupation, 1.0, delta=1e-6)
        self.assertIn("down channel: 3 levels", notes[0].multiplets[0])
        # The full-grid run itself, bit for bit, and the system holds its objects now.
        self.assertEqual(result.energies.total, full.energies.total)
        for index in range(2):
            np.testing.assert_array_equal(result.spin[index].eigenvalues, full.spin[index].eigenvalues)
        details = dict(result.backend.details)
        self.assertTrue(details["orbital_symmetry"].startswith("full grid (spin-polarized"))
        self.assertIn("left the sectors after the start Hamiltonian", details["spin_symmetry_sectors"])
        self.assertIsNone(system.spin_sector_fallback)
        self.assertIsNone(getattr(system.backend, "symmetry_eigensolver", None))

    def test_integer_filling_of_a_partly_filled_shell_leaves_the_sectors(self) -> None:
        """The O atom with integer occupations: one minority p electron in three levels.

        At ``Fermi_Temp: 0`` the occupations are integer from the start, and the
        sector state (one axis-aligned p orbital) lies above the full grid's;
        the start Hamiltonian has the three p levels as one multiplet holding one
        electron, and the run goes to the full grid there.
        """

        from parsec_python.models import SpinSectorFallbackRecord

        options = dict(drop=("Fermi_Temp",), extra=("Fermi_Temp: 0",))
        _, full = self.solve("o_spherical_ca", "off", **options)
        records = []
        _, result = self.solve("o_spherical_ca", "auto", records=records, **options)
        notes = [record for record in records if isinstance(record, SpinSectorFallbackRecord)]
        self.assertEqual([note.stage for note in notes], ["the start Hamiltonian"])
        self.assertEqual(result.energies.total, full.energies.total)

    def test_forced_symmetry_leaves_the_sectors_too(self) -> None:
        """``--symmetry on``: the sectors are taken, and a run that leaves them drops the forced symmetry."""

        from parsec_python.models import SpinSectorFallbackRecord

        _, full = self.solve("o_spherical_ca", "off", drop=("Fermi_Temp",))
        records = []
        _, result = self.solve("o_spherical_ca", "auto", drop=("Fermi_Temp",), records=records, symmetry="on")
        self.assertEqual(sum(isinstance(record, SpinSectorFallbackRecord) for record in records), 1)
        self.assertEqual(result.energies.total, full.energies.total)

    def test_hybrid_in_the_sectors(self) -> None:
        drop = ("Correlation_Type",)
        # The same Fock-loop path on both sides: with Exx_Inner_Criterion adaptive the PBE
        # start is looser in the sectors only (SCF.fock_controls), and E_x, first order in
        # the orbitals, then differs at the inner criterion (2.8e-8 Ry measured), not at round-off.
        extra = ("Correlation_Type: pbe0", "Exx_Tolerance: 1e-9 Ry", "Exx_Inner_Criterion: fixed")
        _, full = self.solve("o2_ca", "off", drop=drop, extra=extra)
        _, sectors = self.solve("o2_ca", "auto", drop=drop, extra=extra)
        self.assert_same_run(full, sectors)
        self.assertIsNotNone(sectors.exact_exchange)
        self.assertAlmostEqual(
            sectors.exact_exchange.exchange_energy, full.exact_exchange.exchange_energy, delta=1e-8
        )


if __name__ == "__main__":
    unittest.main()
