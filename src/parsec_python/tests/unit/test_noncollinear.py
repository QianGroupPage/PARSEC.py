"""Tests for non-collinear magnetism: XC, Hamiltonian, energy, SCF, input."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python import (
    Atom,
    EigensolverSettings,
    GridSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
    ca_lda_spin_polarized,
)
from parsec_python.cli import main as cli_main
from parsec_python.Eigensolvers.perturbative_soc import build_spin_orbit_projectors
from parsec_python.Energy import (
    total_energy_no_degeneracy_spin_polarized,
    total_energy_noncollinear,
)
from parsec_python.Grid import build_cluster_grid, monkhorst_pack_grid
from parsec_python.Hamiltonian.spinor_operator import (
    SpinorKohnShamHamiltonian,
    apply_zeeman_field,
)
from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints_spin_polarized
from parsec_python.SCF.noncollinear import (
    run_self_consistent_noncollinear,
    run_self_consistent_noncollinear_kpoints,
    spinor_density_and_magnetization,
)
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.single_point import prepare_single_point
from parsec_python.SCF.spin_polarized import run_scf_spin_polarized
from parsec_python.V_xc import noncollinear_xc, pbe_spin_polarized

_AU_POTRE = (
    Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
)


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class NoncollinearXCTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.n = 10 ** rng.uniform(-2, 0, 200)
        direction = rng.normal(size=(3, 200))
        direction /= np.linalg.norm(direction, axis=0)
        self.m = direction * self.n * rng.uniform(0.05, 0.95, 200)

    def _lsda(self, up, down):
        return ca_lda_spin_polarized(up, down, 1.0)

    def test_collinear_limit_matches_lsda(self) -> None:
        m = np.zeros((3, self.n.size))
        m[2] = 0.6 * self.n
        result = noncollinear_xc(self.n, m, self._lsda)
        direct = ca_lda_spin_polarized(0.8 * self.n, 0.2 * self.n, 1.0)
        np.testing.assert_allclose(
            result.potential, 0.5 * (direct.potential_up + direct.potential_down)
        )
        np.testing.assert_allclose(result.field[2], 0.5 * (direct.potential_up - direct.potential_down))
        np.testing.assert_allclose(result.field[:2], 0.0, atol=1e-14)
        self.assertAlmostEqual(result.total_energy, direct.total_energy, places=12)

    def test_rotation_covariance(self) -> None:
        rotation = _rotation([1.0, 2.0, -0.5], 0.9)
        base = noncollinear_xc(self.n, self.m, self._lsda)
        rotated = noncollinear_xc(self.n, rotation @ self.m, self._lsda)
        np.testing.assert_allclose(rotated.potential, base.potential, rtol=1e-12)
        np.testing.assert_allclose(rotated.field, rotation @ base.field, rtol=1e-10, atol=1e-13)
        self.assertAlmostEqual(rotated.total_energy, base.total_energy, places=10)

    def test_field_is_parallel_to_magnetization(self) -> None:
        result = noncollinear_xc(self.n, self.m, self._lsda)
        cross = np.cross(result.field.T, self.m.T)
        self.assertLess(np.max(np.abs(cross)), 1e-12)

    def test_zero_magnetization_gives_zero_field(self) -> None:
        zero = np.zeros((3, self.n.size))
        result = noncollinear_xc(self.n, zero, self._lsda)
        np.testing.assert_array_equal(result.field, 0.0)
        self.assertTrue(np.all(np.isfinite(result.potential)))

    def test_magnetization_above_density_is_clipped(self) -> None:
        m = np.zeros((3, self.n.size))
        m[0] = self.n * (1.0 + 1e-13)
        result = noncollinear_xc(self.n, m, self._lsda)
        self.assertTrue(np.all(np.isfinite(result.potential)))

    def test_pbe_rotation_covariance_on_a_grid(self) -> None:
        grid = build_cluster_grid(
            GridSettings(spacing=0.7, radius=3.2, expansion_order=8, shift=(0.5, 0.5, 0.5))
        )
        r2 = np.einsum("ij,ij->i", grid.coordinates, grid.coordinates)
        n = 0.2 * np.exp(-0.5 * r2) + 0.01
        direction = np.stack([np.cos(0.3 * grid.coordinates[:, 0]), np.sin(0.3 * grid.coordinates[:, 0]), 0 * r2])
        m = 0.5 * n * direction
        rotation = _rotation([0.3, 1.0, 0.2], 1.1)

        def functional(up, down):
            return pbe_spin_polarized(up, down, grid)

        base = noncollinear_xc(n, m, functional)
        rotated = noncollinear_xc(n, rotation @ m, functional)
        self.assertAlmostEqual(rotated.total_energy, base.total_energy, places=9)
        np.testing.assert_allclose(rotated.field, rotation @ base.field, rtol=1e-8, atol=1e-10)


class ZeemanFieldTests(unittest.TestCase):
    def test_hermitian_and_pauli_structure(self) -> None:
        rng = np.random.default_rng(1)
        n = 30
        field = rng.normal(size=(3, n))
        up = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        down = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        a_up, a_down = apply_zeeman_field(field, up, down)
        up2 = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        down2 = rng.normal(size=(n, 1)) + 1j * rng.normal(size=(n, 1))
        b_up, b_down = apply_zeeman_field(field, up2, down2)
        lhs = np.vdot(up2, a_up) + np.vdot(down2, a_down)
        rhs = np.vdot(b_up, up) + np.vdot(b_down, down)
        self.assertAlmostEqual(lhs.real, rhs.real, places=10)
        self.assertAlmostEqual(lhs.imag, rhs.imag, places=10)
        # B along z -> diagonal +/-Bz; B along x -> pure swap.
        z_only = np.zeros((3, n))
        z_only[2] = field[2]
        zu, zd = apply_zeeman_field(z_only, up, down)
        np.testing.assert_allclose(zu, field[2][:, None] * up)
        np.testing.assert_allclose(zd, -field[2][:, None] * down)
        x_only = np.zeros((3, n))
        x_only[0] = field[0]
        xu, xd = apply_zeeman_field(x_only, up, down)
        np.testing.assert_allclose(xu, field[0][:, None] * down)
        np.testing.assert_allclose(xd, field[0][:, None] * up)


class SpinorMagnetizationTests(unittest.TestCase):
    def _single(self, up, down):
        spinor = np.array([[up], [down]], dtype=complex)  # one grid point, one state
        return spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0)

    def test_known_spin_directions(self) -> None:
        s = 1.0 / np.sqrt(2.0)
        for up, down, expected in (
            (1.0, 0.0, (0, 0, 1)),
            (0.0, 1.0, (0, 0, -1)),
            (s, s, (1, 0, 0)),
            (s, -s, (-1, 0, 0)),
            (s, 1j * s, (0, 1, 0)),
            (s, -1j * s, (0, -1, 0)),
        ):
            n, m = self._single(up, down)
            self.assertAlmostEqual(float(n[0]), 1.0)
            np.testing.assert_allclose(m[:, 0], expected, atol=1e-12)

    def test_pure_state_has_unit_polarization(self) -> None:
        rng = np.random.default_rng(2)
        spinor = rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1))
        spinor /= np.linalg.norm(spinor)
        n, m = spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(m[:, 0])), float(n[0]), places=12)

    def test_k_weight_scales_linearly(self) -> None:
        spinor = np.array([[0.6], [0.8j]], dtype=complex)
        n1, m1 = spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0, 1.0)
        n2, m2 = spinor_density_and_magnetization(spinor, np.array([1.0]), 1.0, 0.25)
        np.testing.assert_allclose(n2, 0.25 * n1)
        np.testing.assert_allclose(m2, 0.25 * m1)


class NoncollinearEnergyTests(unittest.TestCase):
    def test_z_aligned_matches_collinear_energy(self) -> None:
        rng = np.random.default_rng(3)
        n_grid = 40
        up = rng.uniform(0.01, 0.2, n_grid)
        down = rng.uniform(0.01, 0.2, n_grid)
        v_in_up, v_in_down = rng.normal(size=(2, n_grid))
        ionic, hartree = rng.normal(size=(2, n_grid))
        vxc_up, vxc_down = rng.normal(size=(2, n_grid))
        eig = rng.normal(size=12)
        occ = rng.uniform(0, 1, 12)
        collinear = total_energy_no_degeneracy_spin_polarized(
            eig, occ, up, down, v_in_up, v_in_down, ionic, hartree, vxc_up, vxc_down,
            -1.3, 0.7, 0.05, alpha_z_energy=0.1, band_energy_weight=0.5,
        )
        magnetization = np.zeros((3, n_grid))
        magnetization[2] = up - down
        input_field = np.zeros((3, n_grid))
        input_field[2] = 0.5 * (v_in_up - v_in_down)
        xc_field = np.zeros((3, n_grid))
        xc_field[2] = 0.5 * (vxc_up - vxc_down)
        general = total_energy_noncollinear(
            eig, occ, up + down, magnetization,
            0.5 * (v_in_up + v_in_down), input_field, ionic, hartree,
            0.5 * (vxc_up + vxc_down), xc_field, -1.3, 0.7, 0.05,
            alpha_z_energy=0.1, band_energy_weight=0.5,
        )
        for name in ("eigenvalue", "hartree", "integral_vxc_rho", "electron_ion", "electronic", "total"):
            self.assertAlmostEqual(getattr(general, name), getattr(collinear, name), places=10, msg=name)


def _au_cluster(direction, *, soc=False, nc=True, polarization=0.3, convergence=1e-4):
    return SinglePointInput(
        atoms=[Atom("Au", [0.0, 0.0, 0.0], direction)],
        pseudopotentials={
            "Au": SpeciesPotential(
                _AU_POTRE, 0, spin_orbit=soc, initial_spin_polarization=polarization
            )
        },
        grid=GridSettings(spacing=0.6, radius=6.0, expansion_order=6),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=20,
            convergence_criterion=convergence,
            noncollinear=nc,
            spin_polarized=not nc,
            self_consistent_spin_orbit=soc,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


def _cos(vector, direction):
    direction = np.asarray(direction, dtype=float)
    return float(np.dot(vector, direction) / np.linalg.norm(vector) / np.linalg.norm(direction))


class NoncollinearClusterSCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference = run_scf_spin_polarized(
            prepare_single_point(_au_cluster(None, nc=False))
        )

    def test_z_aligned_reproduces_the_collinear_driver(self) -> None:
        result = run_self_consistent_noncollinear(
            prepare_single_point(_au_cluster([0, 0, 1]))
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energies.total, self.reference.energies.total, delta=1e-4)
        self.assertAlmostEqual(result.magnetic_moment, self.reference.magnetic_moment, delta=1e-2)

    def test_energy_is_invariant_to_the_moment_direction_without_soc(self) -> None:
        """No spin-orbit coupling: a global spin rotation cannot change the
        energy, and the converged moment must follow the seed direction."""
        for direction in ([1, 0, 0], [1, 2, 2], [0, -1, 0]):
            with self.subTest(direction=direction):
                result = run_self_consistent_noncollinear(
                    prepare_single_point(_au_cluster(direction))
                )
                self.assertTrue(result.converged)
                self.assertAlmostEqual(
                    result.energies.total, self.reference.energies.total, delta=1e-4
                )
                self.assertAlmostEqual(result.magnetic_moment, 1.0, delta=1e-2)
                self.assertGreater(_cos(result.magnetic_moment_vector, direction), 0.999)

    def test_with_soc_a_spherical_atom_is_still_rotation_invariant(self) -> None:
        energies = []
        for direction in ([0, 0, 1], [1, 0, 0], [1, 1, 1]):
            system = prepare_single_point(_au_cluster(direction, soc=True))
            projectors = tuple(
                build_spin_orbit_projectors(
                    system.grid,
                    system.atoms,
                    system.pseudopotentials,
                    system.input.pseudopotentials,
                )
            )
            result = run_self_consistent_noncollinear(system, projectors)
            self.assertTrue(result.converged)
            self.assertGreater(result.magnetic_moment, 0.9)
            energies.append(result.energies.total)
        # A cubic grid breaks continuous rotation symmetry slightly.
        self.assertLess(max(energies) - min(energies), 5e-3)

    def test_requires_the_noncollinear_setting(self) -> None:
        system = prepare_single_point(_au_cluster(None, nc=False))
        with self.assertRaises(ValueError):
            run_self_consistent_noncollinear(system)


class NoncollinearTwoAtomTests(unittest.TestCase):
    def test_perpendicular_moments_are_preserved(self) -> None:
        problem = SinglePointInput(
            atoms=[
                Atom("Au", [-4.0, 0.0, 0.0], [1, 0, 0]),
                Atom("Au", [4.0, 0.0, 0.0], [0, 1, 0]),
            ],
            pseudopotentials={
                "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.4)
            },
            grid=GridSettings(spacing=0.7, radius=9.0, expansion_order=6),
            scf=SCFSettings(
                max_iterations=100,
                number_of_states=30,
                convergence_criterion=1e-3,
                noncollinear=True,
            ),
            hartree=HartreeSettings(multipole_order=4),
            eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
        )
        system = prepare_single_point(problem)
        result = run_self_consistent_noncollinear(system)
        self.assertTrue(result.converged)
        m = result.magnetization
        dv = system.grid.volume_element
        left = system.grid.coordinates[:, 0] < 0
        left_moment = dv * m[:, left].sum(axis=1)
        right_moment = dv * m[:, ~left].sum(axis=1)
        self.assertGreater(_cos(left_moment, [1, 0, 0]), 0.99)
        self.assertGreater(_cos(right_moment, [0, 1, 0]), 0.99)
        self.assertAlmostEqual(float(np.linalg.norm(left_moment)), 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.linalg.norm(right_moment)), 1.0, delta=0.05)
        # not collinear: the two atomic moments are orthogonal
        self.assertLess(abs(float(np.dot(left_moment, right_moment))), 0.05)


def _au_periodic(direction, *, convergence=1e-4):
    box = 8.0
    return SinglePointInput(
        atoms=[Atom("Au", [box / 2] * 3, direction)],
        pseudopotentials={
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.3)
        },
        grid=PeriodicGridSettings(spacing=0.6, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=box * np.eye(3)),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=20,
            convergence_criterion=convergence,
            noncollinear=True,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-6),
    )


class NoncollinearPeriodicTests(unittest.TestCase):
    def test_rotation_invariance_with_kpoints(self) -> None:
        energies = []
        for direction in ([0, 0, 1], [1, 1, 0]):
            system = prepare_periodic_single_point(_au_periodic(direction))
            k_points, weights = monkhorst_pack_grid(system.input.periodic_cell, (2, 1, 1))
            result = run_self_consistent_noncollinear_kpoints(system, k_points, weights)
            self.assertTrue(result.converged)
            # Without SOC the global moment direction is a zero-energy soft
            # mode the SCF residual does not pin, so only require that it
            # stays near the seed; energy and |m| are the sharp checks.
            self.assertGreater(_cos(result.magnetic_moment_vector, direction), 0.99)
            self.assertAlmostEqual(result.magnetic_moment, 1.0, delta=0.05)
            energies.append(result.energies.total)
        self.assertAlmostEqual(energies[0], energies[1], delta=2e-4)

    def test_z_aligned_matches_the_collinear_kpoint_driver_without_soc(self) -> None:
        """With no spin-orbit coupling a z-aligned non-collinear run is exactly
        the collinear problem.  (With SOC they legitimately differ: the spinors
        then carry local transverse magnetization that the collinear driver's
        z-projected spin densities cannot see, so the non-collinear XC acts on
        a larger local |m| -- measured as -1.5 mRy and |m_z| 0.895 vs 0.953
        for Au at Gamma, independent of SCF tolerance.)"""
        species = {
            "Au": SpeciesPotential(_AU_POTRE, 0, initial_spin_polarization=0.3)
        }
        base = _au_periodic([0, 0, 1])

        def build(**scf_flags):
            return SinglePointInput(
                atoms=base.atoms,
                pseudopotentials=species,
                grid=base.grid,
                periodic_cell=base.periodic_cell,
                scf=SCFSettings(
                    max_iterations=80,
                    number_of_states=20,
                    convergence_criterion=1e-4,
                    **scf_flags,
                ),
                hartree=base.hartree,
                eigensolver=base.eigensolver,
            )

        gamma = (np.zeros((1, 3)), np.ones(1))
        collinear = run_self_consistent_soc_kpoints_spin_polarized(
            prepare_periodic_single_point(
                build(spin_polarized=True, self_consistent_spin_orbit=True)
            ),
            *gamma,
        )
        general = run_self_consistent_noncollinear_kpoints(
            prepare_periodic_single_point(build(noncollinear=True)), *gamma
        )
        self.assertTrue(collinear.converged and general.converged)
        self.assertAlmostEqual(general.energies.total, collinear.energies.total, delta=2e-4)
        self.assertAlmostEqual(general.magnetic_moment, abs(collinear.magnetic_moment), delta=1e-2)


_CLUSTER_INPUT = """\
Boundary_Sphere_Radius: 6.0
Grid_Spacing: 0.6
Expansion_Order: 6

States_Num: 20
Net_Charges: 0
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 1.0e-3

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Non_Collinear_magnetism: .true.

Atom_Types_Num: 1
Total_Atom_Num: 1

Atom_Type: Au
SO_PSP: .false.
Local_Component: s
Initial_Spin_Polarization: 0.3
Begin atom_coord
   0.0 0.0 0.0
End atom_coord
{extra}
Correlation_Type: CA
"""


class NoncollinearInputAndCliTests(unittest.TestCase):
    def _directory(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        lines = _AU_POTRE.read_text(encoding="utf-8").splitlines()
        (directory / "Au_POTRE.DAT").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return directory

    def _write(self, directory, extra=""):
        path = directory / "parsec.in"
        path.write_text(_CLUSTER_INPUT.format(extra=extra), encoding="utf-8")
        return path

    def test_cluster_run_reports_the_moment_vector(self) -> None:
        directory = self._directory()
        path = self._write(
            directory, "begin Initial_NCL_Moment\n0 3 4\nend Initial_NCL_Moment\n"
        )
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        log = (directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn("Converged magnetic moment vector", log)
        archive = np.load(directory / "parsec_python_results.npz")
        vector = archive["magnetic_moment_vector"]
        self.assertGreater(_cos(vector, [0, 3, 4]), 0.999)
        self.assertEqual(archive["magnetization_mu_b_per_bohr3"].shape[0], 3)

    def test_periodic_run_with_kpoints(self) -> None:
        directory = self._directory()
        text = _CLUSTER_INPUT.format(
            extra=(
                "begin Initial_NCL_Moment\n1 0 0\nend Initial_NCL_Moment\n"
                "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\n"
                "end Monkhorst_Pack_Grid\n"
            )
        )
        text = text.replace("Boundary_Sphere_Radius: 6.0\n", "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n8.0 8.0 8.0\nend Cell_Shape\n")
        text = text.replace("0.0 0.0 0.0\nEnd atom_coord", "4.0 4.0 4.0\nEnd atom_coord")
        path = directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(cli_main([str(path), "--quiet"]), 0)
        archive = np.load(directory / "parsec_python_results.npz")
        self.assertGreater(_cos(archive["magnetic_moment_vector"], [1, 0, 0]), 0.999)

    def test_moment_block_without_the_flag_is_rejected(self) -> None:
        directory = self._directory()
        text = _CLUSTER_INPUT.format(
            extra="begin Initial_NCL_Moment\n0 0 1\nend Initial_NCL_Moment\n"
        ).replace("Non_Collinear_magnetism: .true.\n", "")
        path = directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(cli_main([str(path), "--dry-run"]), 2)

    def test_wrong_number_of_moment_rows_is_rejected(self) -> None:
        directory = self._directory()
        path = self._write(
            directory, "begin Initial_NCL_Moment\n0 0 1\n1 0 0\nend Initial_NCL_Moment\n"
        )
        self.assertEqual(cli_main([str(path), "--dry-run"]), 2)

    def test_atom_normalizes_its_moment_direction(self) -> None:
        atom = Atom("Au", [0, 0, 0], [0, 0, 5])
        np.testing.assert_allclose(atom.initial_moment, [0, 0, 1])
        self.assertIsNone(Atom("Au", [0, 0, 0], [0, 0, 0]).initial_moment)
        with self.assertRaises(ValueError):
            Atom("Au", [0, 0, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
