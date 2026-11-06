"""Tests of the remaining Fortran-parity items (stream ``misc`` of the 2026-10-27 round).

LDA+U and the TS correction in the k-point, periodic spin and band drivers
(the Bloch form of the LDA+U term, its time-reversal and space-group
completion), and the parser rules that come with them.
"""


# With both sets of routines: the periodic inputs of this module set Use_Fortran_Routines: .true. (the
# Fortran-derived routines); the numbers checked here are those of the Fortran parity round.
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from parsec_python.Hamiltonian.hubbard import (
    HubbardOperator,
    build_hubbard_projectors,
    d_harmonic_rotation,
    hubbard_atom_permutations,
    kpoint_occupation_matrix,
    occupation_matrices,
    symmetrize_occupations,
    time_reversal_occupations,
)
from parsec_python.Pseudopotential import read_parsec_pseudopotential
from parsec_python.V_ion import real_spherical_harmonics
from parsec_python.models import Atom, SpeciesPotential
from parsec_python.tests.environment import pin_kpoints_on_the_cpu
from parsec_python.tests.archives import read_archive


DATA = Path(__file__).resolve().parents[1] / "data"
AG = DATA / "Ag_POTRE.DAT"


def _box_grid(sides, spacing):
    axes = [np.arange(0.0, side - 1e-9, spacing) for side in sides]
    coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return SimpleNamespace(
        coordinates=coordinates,
        volume_element=spacing**3,
        size=coordinates.shape[0],
        spacing=spacing,
        settings=SimpleNamespace(double_grid_order=1),
        shape=tuple(len(axis) for axis in axes),
    )


class BlochHubbardProjectorTest(unittest.TestCase):
    """The k-point term equals the Gamma term of a supercell on Bloch functions."""

    def setUp(self):
        self.potential = {"Ag": read_parsec_pseudopotential(AG)}
        self.specifications = {
            "Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=0.3)
        }
        self.side = 4.2
        self.spacing = 0.6
        self.position = np.array([0.7, 1.1, 2.0])

    def test_supercell_equivalence(self):
        side, spacing = self.side, self.spacing
        primitive = _box_grid((side, side, side), spacing)
        supercell = _box_grid((2 * side, side, side), spacing)
        k_point = np.array([np.pi / side, 0.0, 0.0])  # folds onto the supercell Gamma
        primitive_projectors = build_hubbard_projectors(
            primitive,
            (Atom("Ag", self.position),),
            self.potential,
            self.specifications,
            np.diag([side] * 3),
            k_point=k_point,
        )
        supercell_projectors = build_hubbard_projectors(
            supercell,
            (Atom("Ag", self.position), Atom("Ag", self.position + [side, 0, 0])),
            self.potential,
            self.specifications,
            np.diag([2 * side, side, side]),
        )
        rng = np.random.default_rng(3)
        states = 3
        u = rng.normal(size=(primitive.size, states)) + 1j * rng.normal(size=(primitive.size, states))
        # psi(r) = exp(ik.r) u(r mod a) on the supercell grid.
        n = primitive.shape[0]
        index = np.arange(supercell.size)
        ix, rest = np.divmod(index, primitive.shape[1] * primitive.shape[2])
        primitive_rows = (ix % n) * primitive.shape[1] * primitive.shape[2] + rest
        phase = np.exp(1j * supercell.coordinates @ k_point)
        psi = phase[:, None] * u[primitive_rows]

        overlap_k = primitive_projectors.overlaps(u)
        overlap_sc = supercell_projectors.overlaps(psi)
        np.testing.assert_allclose(overlap_sc[0], overlap_k[0], atol=1e-12)
        # The image atom sees the same Bloch function times exp(ik.a) = -1.
        np.testing.assert_allclose(overlap_sc[1], -overlap_k[0], atol=1e-12)

        occupations = np.array([1.0, 0.5, 0.2])
        n_k = kpoint_occupation_matrix(primitive_projectors, u, occupations, 1.0)
        n_sc = occupation_matrices(supercell_projectors, psi, occupations, keep_complex=True)
        np.testing.assert_allclose(n_sc[0], n_k[0], atol=1e-11)
        np.testing.assert_allclose(n_sc[1], n_k[0], atol=1e-11)

        matrix = rng.normal(size=(5, 5))
        matrix = matrix + matrix.T
        applied_k = HubbardOperator(primitive_projectors, matrix[None]).apply(u)
        applied_sc = HubbardOperator(
            supercell_projectors, np.stack([matrix, matrix])
        ).apply(psi)
        np.testing.assert_allclose(applied_sc, phase[:, None] * applied_k[primitive_rows], atol=1e-11)

    def test_gamma_projectors_unchanged_and_k_zero_is_gamma(self):
        grid = _box_grid((self.side,) * 3, self.spacing)
        atoms = (Atom("Ag", self.position),)
        lattice = np.diag([self.side] * 3)
        gamma = build_hubbard_projectors(grid, atoms, self.potential, self.specifications, lattice)
        zero = build_hubbard_projectors(
            grid, atoms, self.potential, self.specifications, lattice, k_point=np.zeros(3)
        )
        self.assertFalse(np.iscomplexobj(gamma.projectors.data))
        np.testing.assert_allclose(zero.projectors.toarray(), gamma.projectors.toarray(), atol=1e-15)

    def test_time_reversal_partner_is_the_conjugate(self):
        grid = _box_grid((self.side,) * 3, self.spacing)
        atoms = (Atom("Ag", self.position),)
        lattice = np.diag([self.side] * 3)
        k_point = np.array([0.3, -0.2, 0.5])
        plus = build_hubbard_projectors(grid, atoms, self.potential, self.specifications, lattice, k_point=k_point)
        minus = build_hubbard_projectors(grid, atoms, self.potential, self.specifications, lattice, k_point=-k_point)
        rng = np.random.default_rng(5)
        u = rng.normal(size=(grid.size, 2)) + 1j * rng.normal(size=(grid.size, 2))
        occupations = np.array([1.0, 0.3])
        n_plus = kpoint_occupation_matrix(plus, u, occupations, 0.5)
        n_minus = kpoint_occupation_matrix(minus, u.conj(), occupations, 0.5)
        np.testing.assert_allclose(n_minus, n_plus.conj(), atol=1e-12)
        np.testing.assert_allclose(
            time_reversal_occupations(n_plus + n_minus), 2.0 * n_plus.real, atol=1e-12
        )


class OccupationSymmetrizationTest(unittest.TestCase):
    def test_d_rotation_is_orthogonal_and_represents_the_rotation(self):
        rng = np.random.default_rng(1)
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        matrix = d_harmonic_rotation(rotation)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(5), atol=1e-12)
        vectors = rng.normal(size=(7, 3))
        np.testing.assert_allclose(
            real_spherical_harmonics(2, vectors @ np.linalg.inv(rotation).T),
            real_spherical_harmonics(2, vectors) @ matrix,
            atol=1e-12,
        )
        product = d_harmonic_rotation(rotation @ rotation.T)
        np.testing.assert_allclose(product, np.eye(5), atol=1e-12)

    def test_group_average_of_rotated_orbitals(self):
        """Orbitals rotated by every operation give the average of one set."""

        from parsec_python.Grid.space_group import find_space_group

        side = 4.2
        spacing = 0.6
        grid = _box_grid((side,) * 3, spacing)
        potential = {"Ag": read_parsec_pseudopotential(AG)}
        specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=0.3)}
        atoms = (Atom("Ag", [0.0, 0.0, 0.0]),)
        lattice = np.diag([side] * 3)
        group = find_space_group(lattice, atoms)
        self.assertEqual(group.order, 48)
        projectors = build_hubbard_projectors(grid, atoms, potential, specifications, lattice)
        operations = hubbard_atom_permutations(atoms, projectors.atom_indices, lattice, group)
        rng = np.random.default_rng(2)
        orbitals = rng.normal(size=(grid.size, 2))
        occupations = np.array([1.0, 0.4])
        # The rotated copies f(g^-1 r) on the grid (a cubic grid through the atom).
        n = grid.shape[0]
        integer = np.rint(grid.coordinates / spacing).astype(int)
        rotated_sets = []
        for matrix, translation in zip(group.rotation_matrices, group.translation_vectors):
            self.assertTrue(np.allclose(translation, 0.0))
            inverse = np.rint(np.linalg.inv(matrix)).astype(int)
            source = (integer @ inverse.T) % n
            rows = (source[:, 0] * n + source[:, 1]) * n + source[:, 2]
            rotated_sets.append(orbitals[rows])
        full = sum(
            occupation_matrices(projectors, block, occupations) for block in rotated_sets
        ) / len(rotated_sets)
        single = occupation_matrices(projectors, orbitals, occupations)
        averaged = symmetrize_occupations(single, operations)
        np.testing.assert_allclose(averaged, full, atol=1e-12)
        # The average is invariant under the group.
        np.testing.assert_allclose(symmetrize_occupations(averaged, operations), averaged, atol=1e-12)


_INPUT = """Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   6.0  6.0  6.0
end Cell_Shape
{kpoints}
Grid_Spacing: 0.6
Ignore_Symmetry: .true.
States_Num: 10
Fermi_Temp: 1000.0
Max_Iter: {iterations}
Convergence_Criterion: 1.0e-4
Mixing_Param: 0.2
Eigensolver: chebff
Correlation_Type: ca
{extra}
Atom_Types_Num: 1
Atom_Type: Ag
Local_Component: s
{ag_line}
begin Atom_Coord
   3.0 3.0 3.0
end Atom_Coord
"""

_KPOINTS = """Kpoint_Method: manual
Kpoint_Unit: reciprocal_lattice_vectors
begin Kpoint_List
 0.0 0.0 0.0 0.5
 0.5 0.0 0.0 0.5
end Kpoint_List"""


def _parse(**fields):
    from parsec_python.Input import parse_parsec_input

    values = dict(kpoints=_KPOINTS, iterations=3, extra="", ag_line="ldaplusu_u 4.0 eV")
    values.update(fields)
    directory = tempfile.mkdtemp()
    path = Path(directory) / "parsec.in"
    path.write_text(_INPUT.format(**values))
    (Path(directory) / "Ag_POTRE.DAT").write_bytes(AG.read_bytes())
    return parse_parsec_input(path)


class KPointHubbardDriverTest(unittest.TestCase):
    def test_parser_accepts_kpoints_spin_and_bands_and_refuses_forces(self):
        from parsec_python.Input import UnsupportedParsecOptionError

        self.assertTrue(_parse().problem.pseudopotentials["Ag"].has_hubbard)
        _parse(extra="Spin_Polarization: .true.")
        _parse(kpoints="", extra="Spin_Polarization: .true.")
        with self.assertRaises(UnsupportedParsecOptionError):
            _parse(extra="Skip_Force: .false.")
        with self.assertRaises(UnsupportedParsecOptionError):
            _parse(extra="SO_from_scratch: .true.")

    def test_kpoint_loop_carries_the_term_from_the_second_iteration(self):
        import os

        from parsec_python.SCF import prepare_periodic_single_point
        from parsec_python.SCF.kpoints import run_scf_kpoints
        from parsec_python.Grid.kpoints import kpoint_sampling

        pin_kpoints_on_the_cpu(self)
        translation = _parse(iterations=3)
        system = prepare_periodic_single_point(translation.problem)
        k_points, weights = kpoint_sampling(translation.problem)
        records = []
        result = run_scf_kpoints(system, k_points, weights, callback=records.append)
        self.assertIsNone(records[0].hubbard_energy)
        self.assertIsNotNone(records[1].hubbard_energy)
        self.assertEqual(result.energies.hubbard, records[-1].hubbard_energy)
        self.assertGreater(result.energies.hubbard, 0.0)
        self.assertEqual(result.hubbard.occupations.dtype, np.float64)
        # The occupation matrix of the 4d shell holds about ten electrons over two spins.
        self.assertGreater(np.trace(result.hubbard.occupations[0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()


_AU_POTRE = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"

_SOC_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 20
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 5.0e-3
Mixing_Param: 0.25
Eigensolver: chebff
{mode}
Atom_Types_Num: 1
Atom_Type: Au
SO_PSP: {so}
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   0 0 0
End atom_coord
Correlation_Type: CA
{extra}
"""


class SpinorSymmetrizationAndPolarizabilityTest(unittest.TestCase):
    def _run(self, mode, extra, so=".true."):
        import os

        from parsec_python.cli import main as cli_main

        pin_kpoints_on_the_cpu(self)
        directory = Path(tempfile.mkdtemp())
        (directory / "Au_POTRE.DAT").write_bytes(_AU_POTRE.read_bytes())
        path = directory / "parsec.in"
        path.write_text(_SOC_INPUT.format(mode=mode, extra=extra, so=so), encoding="utf-8")
        code = cli_main([str(path), "--quiet"])
        return code, directory

    def test_spin_orbit_density_is_symmetrized(self):
        code, directory = self._run("SO_from_scratch: .true.", "Symmetrize_Charge_Density: .true.")
        self.assertIn(code, (0, 3))
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("maximum deviation after symmetrization", log)

    def test_noncollinear_magnetization_keeps_its_direction(self):
        from parsec_python.Grid import symmetrize_density
        from parsec_python.SCF.noncollinear import symmetrize_noncollinear

        rng = np.random.default_rng(4)
        group = SimpleNamespace()
        # A two-element "group" of grid permutations: identity and a reversal.
        n = 6
        permutation = np.arange(n)[::-1].copy()

        class Pair:
            pass

        from parsec_python.Grid.point_group import ScalarSymmetrization  # noqa: F401

        density = rng.uniform(0.1, 1.0, n)
        magnetization = rng.normal(size=(3, n))
        averaged = lambda f: 0.5 * (f + f[permutation])  # noqa: E731

        def fake_symmetrize(values, _group, _volume):
            return averaged(np.asarray(values)), None

        import parsec_python.Grid.point_group as point_group

        original = point_group.symmetrize_density
        point_group.symmetrize_density = fake_symmetrize
        try:
            new_density, new_magnetization, reports = symmetrize_noncollinear(
                density, magnetization, group, 1.0
            )
        finally:
            point_group.symmetrize_density = original
        np.testing.assert_allclose(new_density, averaged(density))
        length = np.linalg.norm(magnetization, axis=0)
        np.testing.assert_allclose(np.linalg.norm(new_magnetization, axis=0), averaged(length))
        np.testing.assert_allclose(
            new_magnetization / np.linalg.norm(new_magnetization, axis=0),
            magnetization / length,
        )
        self.assertEqual(len(reports), 3)
        del symmetrize_density

    def test_spin_orbit_polarizability_writes_polar_dat(self):
        code, directory = self._run(
            "SO_from_scratch: .true.", "Polarizability: .true.\nElectric_Field: 0.005"
        )
        self.assertIn(code, (0, 3))
        text = (directory / "polar.dat").read_text(encoding="utf-8")
        self.assertIn("Polarizability of the cluster", text)
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        self.assertEqual(log.count("Converged spin-orbit-split spinor states"), 7)


_H2_INPUT = """\
Boundary_Sphere_Radius: 5.0
Grid_Spacing: 0.5
Ignore_Symmetry: .true.
Hartree_Boundary_Tolerance: off
States_Num: 4
Fermi_Temp: 500.0
Max_Iter: 40
Convergence_Criterion: 1.0e-6
Eigensolver: chebff
Mixing_Param: 0.3
{extra}
Atom_Types_Num: 1
Atom_Type: H
Local_Component: s
begin Atom_Coord
   0.0 0.0 -0.7
   0.0 0.0 0.7
end Atom_Coord
"""


class WfnDatRestartTest(unittest.TestCase):
    def _directory(self, extra=""):
        directory = Path(tempfile.mkdtemp())
        (directory / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
        (directory / "parsec.in").write_text(_H2_INPUT.format(extra=extra), encoding="utf-8")
        return directory

    def test_round_trip_of_the_layout(self):
        from parsec_python.Output.wfn_dat import WfnDat, WfnDatChannel, read_wfn_dat, write_wfn_dat

        rng = np.random.default_rng(7)
        channels = tuple(
            WfnDatChannel(
                representations=np.ones(3, dtype=np.int32),
                energies=rng.normal(size=3),
                occupations=np.array([1.0, 1.0, 0.0]),
                potential=rng.normal(size=10),
                density=rng.uniform(size=10),
                wavefunctions=rng.normal(size=(10, 3)),
            )
            for _ in range(2)
        )
        data = WfnDat(
            date_label="27-OCT-2026 12:00:00 -0700",
            step=np.array([0.5]),
            rmax=5.0,
            box_size=None,
            ndim=10,
            representation_count=1,
            characters=np.ones((1, 1), dtype=np.int32),
            points=rng.integers(-5, 5, size=(10, 3)),
            channels=channels,
        )
        path = Path(tempfile.mkdtemp()) / "wfn.dat"
        write_wfn_dat(path, data)
        back = read_wfn_dat(path, periodic=False, spin_count=2, complex_values=False)
        self.assertEqual(back.date_label, data.date_label)
        self.assertEqual(back.rmax, 5.0)
        np.testing.assert_array_equal(back.points, data.points)
        for original, read in zip(channels, back.channels):
            np.testing.assert_array_equal(read.wavefunctions, original.wavefunctions)
            np.testing.assert_array_equal(read.potential, original.potential)
            np.testing.assert_array_equal(read.energies, original.energies)
        # The header record: two doubles and three integers (restart.F90).
        from parsec_python.Output.parsec_dat import iter_records

        records = list(iter_records(path))
        self.assertEqual(len(records[1]), 2 * 8 + 3 * 4)
        self.assertEqual(len(records), 1 + 1 + 5 + 1 + 1 + 2 * (6 + 3))

    def test_restart_converges_at_once_and_checks_the_grid(self):
        import os

        from parsec_python.cli import main as cli_main
        from parsec_python.Input import parse_parsec_input
        from parsec_python.Output.wfn_dat import wfn_dat_from_result, write_wfn_dat
        from parsec_python.SCF import prepare_single_point, run_scf
        from parsec_python.SCF.restart import RestartError, load_wfndat_restart

        pin_kpoints_on_the_cpu(self)
        first = self._directory()
        system = prepare_single_point(parse_parsec_input(first / "parsec.in").problem)
        result = run_scf(system)
        restart = self._directory("Restart_Run: .true.\nRestart_From_Wfndat: .true.")
        write_wfn_dat(restart / "wfn.dat", wfn_dat_from_result(result, system.grid))
        self.assertEqual(cli_main([str(restart / "parsec.in"), "--quiet"]), 0)
        archive = read_archive(restart / "parsec_python_results.npz")
        self.assertLessEqual(int(archive["iterations"]), 2)
        self.assertAlmostEqual(float(archive["energy_total_ry"]), result.energies.total, places=6)
        log = (restart / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("wfn.dat file created on", log)
        self.assertIn("You should think seriously about upgrading source code!", log)
        # Another spacing: PARSEC error 323.
        other = self._directory("Restart_Run: .true.\nRestart_From_Wfndat: .true.").joinpath("parsec.in")
        other.write_text(other.read_text().replace("Grid_Spacing: 0.5", "Grid_Spacing: 0.55"))
        system_other = prepare_single_point(parse_parsec_input(other).problem)
        with self.assertRaises(RestartError) as caught:
            load_wfndat_restart(restart / "wfn.dat", system_other)
        self.assertEqual(caught.exception.code, 323)

    def test_flag_alone_is_inert_and_kpoints_are_refused(self):
        from parsec_python.Input import UnsupportedParsecOptionError, parse_parsec_input

        directory = self._directory("Restart_From_Wfndat: .true.")
        translation = parse_parsec_input(directory / "parsec.in")
        self.assertFalse(translation.restart_from_wfndat)
        self.assertTrue(any("Restart_From_Wfndat" in w for w in translation.warnings))
        with self.assertRaises(UnsupportedParsecOptionError):
            _parse(extra="Restart_Run: .true.\nRestart_From_Wfndat: .true.")


class DoubleGridProjectorTest(unittest.TestCase):
    """LDA+U and spin-orbit projectors averaged over the double grid."""

    def _grid(self, order):
        from parsec_python.Grid import build_cluster_grid
        from parsec_python.models import GridSettings

        return build_cluster_grid(GridSettings(radius=4.5, spacing=0.6, double_grid_order=order))

    def test_hubbard_projector_is_the_weighted_fine_point_sum(self):
        from parsec_python.Hamiltonian.hubbard import interpolate_hubbard_radial
        from parsec_python.V_ion.double_grid import double_grid_points

        potential = {"Ag": read_parsec_pseudopotential(AG)}
        specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=0.3)}
        atoms = (Atom("Ag", [0.13, -0.07, 0.21]),)
        plain = build_hubbard_projectors(self._grid(1), atoms, potential, specifications)
        grid = self._grid(2)
        double = build_hubbard_projectors(grid, atoms, potential, specifications)
        self.assertGreaterEqual(double.support_sizes[0], plain.support_sizes[0])
        rc = double.cutoff_radii["Ag"]
        displacements, weights = double_grid_points(2, 0.6)
        dense = double.projectors.toarray()
        rng = np.random.default_rng(0)
        for row in rng.choice(np.flatnonzero(np.any(dense != 0.0, axis=1)), size=12, replace=False):
            fine = grid.coordinates[row] + displacements - atoms[0].position
            distance = np.linalg.norm(fine, axis=1)
            inside = distance <= rc
            radial, _ = interpolate_hubbard_radial(potential["Ag"], distance[inside])
            expected = np.sqrt(grid.volume_element) * np.sum(
                (weights[inside] * radial)[:, None] * real_spherical_harmonics(2, fine[inside]),
                axis=0,
            )
            np.testing.assert_allclose(dense[row], expected, atol=1e-14)

    def test_hubbard_gradients_are_the_derivative_of_the_projectors(self):
        from parsec_python.Forces.hubbard_force import build_hubbard_projector_gradients

        potential = {"Ag": read_parsec_pseudopotential(AG)}
        specifications = {"Ag": SpeciesPotential(AG, 0, element_symbol="Ag", hubbard_u=0.3)}
        grid = self._grid(2)
        position = np.array([0.13, -0.07, 0.21])
        step = 1.0e-5
        projectors = build_hubbard_projectors(grid, (Atom("Ag", position),), potential, specifications)
        gradients = build_hubbard_projector_gradients(grid, projectors, (Atom("Ag", position),), potential)
        # d phi(r - R)/dR_x = -d phi/dx: central difference in the atom position.
        plus = build_hubbard_projectors(grid, (Atom("Ag", position + [step, 0, 0]),), potential, specifications)
        minus = build_hubbard_projectors(grid, (Atom("Ag", position - [step, 0, 0]),), potential, specifications)
        difference = -(plus.projectors.toarray() - minus.projectors.toarray()) / (2 * step)
        analytic = gradients[0].toarray()
        # Points whose fine points cross r_c between the two positions are not differentiable.
        smooth = np.all(np.abs(difference - analytic) < 1e-3, axis=1)
        self.assertGreater(np.count_nonzero(smooth), 0.95 * grid.size)
        np.testing.assert_allclose(analytic[smooth], difference[smooth], atol=2e-6)

    def test_spin_orbit_projectors_on_the_double_grid(self):
        from parsec_python.Eigensolvers.perturbative_soc import (
            build_spin_orbit_projectors,
            complex_spherical_harmonics,
        )
        from parsec_python.V_ion.double_grid import double_grid_points

        au = _AU_POTRE
        potential = {"Au": read_parsec_pseudopotential(au)}
        specifications = {"Au": SpeciesPotential(au, 0, element_symbol="Au", spin_orbit=True)}
        atoms = (Atom("Au", [0.11, 0.05, -0.17]),)
        grid = self._grid(2)
        (pair,) = build_spin_orbit_projectors(grid, atoms, potential, specifications)
        (plain,) = build_spin_orbit_projectors(self._grid(1), atoms, potential, specifications)
        self.assertGreaterEqual(pair.support_rows.size, plain.support_rows.size)
        displacements, weights = double_grid_points(2, 0.6)
        radial_ion, radial_so, _ = potential["Au"].spin_orbit_projectors(1, 0)
        from parsec_python.V_ion.ionic_potential import _projector_support_radius

        rc = _projector_support_radius(potential["Au"])
        for local in (0, pair.support_rows.size // 2, pair.support_rows.size - 1):
            row = pair.support_rows[local]
            fine = grid.coordinates[row] + displacements - atoms[0].position
            distance = np.linalg.norm(fine, axis=1)
            inside = distance <= rc
            values = np.interp(distance[inside], potential["Au"].radii, radial_so, left=radial_so[0], right=0.0)
            expected = np.sqrt(grid.volume_element) * np.sum(
                (weights[inside] * values)[:, None] * complex_spherical_harmonics(1, fine[inside]),
                axis=0,
            )
            np.testing.assert_allclose(pair.v_so[local, 0:3], expected, atol=1e-13)


class ClusterDomainShapeTest(unittest.TestCase):
    """Cluster_Domain_Shape ellipsoid and cylinder (grid_partition.f90 outside_domain)."""

    def _parse(self, shape, parameters):
        from parsec_python.Input import parse_parsec_input

        directory = Path(tempfile.mkdtemp())
        (directory / "H_POTRE.DAT").write_bytes((DATA / "H_POTRE.DAT").read_bytes())
        text = _H2_INPUT.format(extra="MPI_Groups: 2").replace(
            "Boundary_Sphere_Radius: 5.0",
            f"Cluster_Domain_Shape: {shape}\nbegin Domain_Shape_Parameters\n{parameters}\nend Domain_Shape_Parameters",
        )
        (directory / "parsec.in").write_text(text, encoding="utf-8")
        return parse_parsec_input(directory / "parsec.in")

    def test_masks_and_enclosing_radius(self):
        from parsec_python.Grid import build_cluster_grid

        translation = self._parse("ellipsoid", "4.0 5.0 3.0")
        settings = translation.problem.grid
        self.assertEqual(settings.domain_shape, "ellipsoid")
        self.assertEqual(settings.enclosing_radius, 5.0)
        self.assertIn("domain_parameters=(4.0, 5.0, 3.0)", repr(settings))
        self.assertTrue(any("MPI_Groups" in w for w in translation.warnings))
        grid = build_cluster_grid(settings)
        r = grid.coordinates
        np.testing.assert_array_less(np.sum((r / [4.0, 5.0, 3.0]) ** 2, axis=1), 1.0 + 1e-12)
        cylinder = self._parse("cylinder", "3.0 8.0 1").problem.grid
        self.assertAlmostEqual(cylinder.enclosing_radius, np.sqrt(16.0 + 9.0))
        points = build_cluster_grid(cylinder).coordinates
        self.assertLessEqual(np.max(np.abs(points[:, 0])), 4.0)
        self.assertLessEqual(np.max(points[:, 1] ** 2 + points[:, 2] ** 2), 9.0 + 1e-12)
        # A sphere's settings keep their former text (cache keys).
        from parsec_python.models import GridSettings

        self.assertNotIn("domain_parameters", repr(GridSettings(spacing=0.5, radius=5.0)))

    def test_parameter_errors(self):
        from parsec_python.Input import ParsecInputError

        with self.assertRaises(ParsecInputError):
            self._parse("cylinder", "3.0 8.0 4")
        with self.assertRaises(ParsecInputError):
            self._parse("ellipsoid", "4.0 -1.0 3.0")

    def test_scf_runs_on_the_reference_solver(self):
        from parsec_python.acceleration.cli import _reference_only_features
        from parsec_python.SCF import prepare_single_point, run_scf

        translation = self._parse("ellipsoid", "5.0 5.0 6.0")
        # The accelerated grid and Hartree builders know these shapes too
        # (acceleration/tests/test_gpu_fields.py).
        self.assertEqual(
            _reference_only_features(translation, accelerated_backend_selected=False), []
        )
        result = run_scf(prepare_single_point(translation.problem))
        self.assertTrue(result.converged)


class SpinorParsecDatTest(unittest.TestCase):
    """parsec.dat of spin-orbit and non-collinear runs (wfnsave.F90, mxwd = 2)."""

    def _run(self, mode, extra=""):
        import os

        from parsec_python.cli import main as cli_main

        pin_kpoints_on_the_cpu(self)
        directory = Path(tempfile.mkdtemp())
        (directory / "Au_POTRE.DAT").write_bytes(_AU_POTRE.read_bytes())
        text = _SOC_INPUT.format(mode=mode, extra=extra, so=".true." if "SO_" in mode else ".false.")
        text = text.replace("Convergence_Criterion: 5.0e-3", "Convergence_Criterion: 1.0e-5\nFF_MaxIter: 9\nChebdav_Degree: 30")
        (directory / "parsec.in").write_text(text, encoding="utf-8")
        return cli_main([str(directory / "parsec.in"), "--quiet"]), directory

    def test_spin_orbit_file_restarts_the_spin_orbit_loop(self):
        from parsec_python.Output.parsec_dat import read_parsec_dat

        code, directory = self._run("SO_from_scratch: .true.")
        self.assertEqual(code, 0)
        data = read_parsec_dat(directory / "parsec.dat")
        self.assertEqual(int(data.system_type[0]), 2)
        self.assertEqual(data.wavefunction_kind, 2)
        self.assertEqual(len(data.potentials), 2)
        self.assertEqual(len(data.levels), 1)
        block = data.wavefunctions[0][0]
        self.assertEqual(block.vectors.shape[0], 2 * data.nwedge)
        norms = np.sum(np.abs(block.vectors) ** 2, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)
        first = read_archive(directory / "parsec_python_results.npz")
        energy = float(first["energy_total_ry"])
        path = directory / "parsec.in"
        path.write_text(
            path.read_text().replace("SO_from_scratch: .true.", "SO_from_scratch: .true.\nRestart_Run: .true."),
            encoding="utf-8",
        )
        from parsec_python.cli import main as cli_main

        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        archive = read_archive(directory / "parsec_python_results.npz")
        self.assertAlmostEqual(float(archive["energy_total_ry"]), energy, places=6)
        self.assertLessEqual(int(archive["iterations"]), 6)

    def test_noncollinear_file_is_written_and_refused_for_a_restart(self):
        from parsec_python.Output.parsec_dat import read_parsec_dat
        from parsec_python.SCF.restart import RestartError, check_restart_file

        code, directory = self._run("Non_Collinear_magnetism: .true.")
        self.assertIn(code, (0, 3))
        data = read_parsec_dat(directory / "parsec.dat")
        self.assertEqual(data.wavefunction_kind, 3)
        up, down = data.densities
        self.assertTrue(np.all(up >= down - 1e-14))
        with self.assertRaises(RestartError):
            check_restart_file(data, None, spin_count=2, complex_wavefunctions=True, spinors=True)


_PERIODIC_SOC_INPUT = """\
Periodic_System: .true.
Use_Fortran_Routines: .true.
Boundary_Conditions: bulk
begin Cell_Shape
   8.0  8.0  8.0
end Cell_Shape
Grid_Spacing: 0.6
Expansion_Order: 6
States_Num: 20
Fermi_Temp: 500.0
Max_Iter: 60
Convergence_Criterion: 1.0e-5
FF_MaxIter: 9
Chebdav_Degree: 30
Mixing_Param: 0.25
Eigensolver: chebff
SO_from_scratch: .true.
{extra}
Atom_Types_Num: 1
Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Begin atom_coord
   4.0 4.0 4.0
End atom_coord
Correlation_Type: CA
"""


class PeriodicSpinOrbitRestartTest(unittest.TestCase):
    def test_gamma_cell_restarts_from_its_spin_orbit_file(self):
        import os

        from parsec_python.cli import main as cli_main
        from parsec_python.Output.parsec_dat import read_parsec_dat

        pin_kpoints_on_the_cpu(self)
        directory = Path(tempfile.mkdtemp())
        (directory / "Au_POTRE.DAT").write_bytes(_AU_POTRE.read_bytes())
        path = directory / "parsec.in"
        path.write_text(_PERIODIC_SOC_INPUT.format(extra=""), encoding="utf-8")
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        data = read_parsec_dat(directory / "parsec.dat", wavefunctions=False)
        self.assertEqual(data.wavefunction_kind, 2)
        self.assertEqual(data.boundary_code, 1)
        first = read_archive(directory / "parsec_python_results.npz")
        energy = float(first["energy_total_ry"])
        path.write_text(_PERIODIC_SOC_INPUT.format(extra="Restart_Run: .true."), encoding="utf-8")
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        archive = read_archive(directory / "parsec_python_results.npz")
        self.assertAlmostEqual(float(archive["energy_total_ry"]), energy, places=6)
        self.assertLessEqual(int(archive["iterations"]), 4)
