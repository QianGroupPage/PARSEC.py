"""Forces of spinor orbitals: self-consistent spin-orbit coupling and non-collinear magnetism.

Two Au atoms (relativistic pseudopotential, no core correction), one electron short so that the spinors carry a moment.
There is no Fortran reference for these forces (``forcso`` of ``soforc.F90`` warns that its spin-orbit forces "may be
inaccurate" and pairs the harmonics of the spin-flip term differently from the Hamiltonian), so they are checked against

* finite differences of the matching energy term at fixed orbitals, term by term (exact, with the cubic spline);
* finite differences of the total energy of the converged SCF loop, which is the force a relaxation follows;
* the unpolarized spin-orbit driver, for a non-magnetic run of the non-collinear one.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

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
)
from parsec_python.cli import main as cli_main
from parsec_python.Eigensolvers.perturbative_soc import (
    build_spin_orbit_projectors,
    complex_spherical_harmonics,
)
from parsec_python.Forces import (
    compute_periodic_forces,
    compute_spinor_forces,
    spinor_nonlocal_forces,
)
from parsec_python.Forces.spin_orbit import (
    build_spin_orbit_force_operator,
    build_spin_orbit_projector_gradients,
    complex_harmonics_with_gradients,
    spin_orbit_coefficients,
    spin_orbit_forces,
)
from parsec_python.Grid import monkhorst_pack_grid
from parsec_python.Hamiltonian.spinor_operator import _apply_nonlocal_complex, apply_spin_orbit
from parsec_python.Relaxation.backend import spinor_backend
from parsec_python.SCF import update_geometry, update_periodic_geometry
from parsec_python.SCF.pbc import prepare_periodic_single_point
from parsec_python.SCF.single_point import prepare_single_point

_AU = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH" / "Au_POTRE.DAT"
_POSITIONS = np.array([[-2.31, 0.17, 0.22], [2.43, -0.11, 0.13]])


def _cluster(*, mode: str, net_charge: float = 1.0, polarization: float = 0.5) -> SinglePointInput:
    """Au2 in a sphere; ``mode`` is ``"ncl"`` (non-collinear), ``"soc"`` (unpolarized) or ``"spin"`` (collinear)."""

    return SinglePointInput(
        atoms=[Atom("Au", _POSITIONS[0], [1, 0, 0]), Atom("Au", _POSITIONS[1], [0, 1, 0])],
        pseudopotentials={
            "Au": SpeciesPotential(_AU, 0, use_spline=True, spin_orbit=True, initial_spin_polarization=polarization)
        },
        grid=GridSettings(spacing=0.7, radius=8.0, expansion_order=6),
        scf=SCFSettings(
            max_iterations=80,
            number_of_states=24,
            convergence_criterion=5.0e-5,
            net_charge=net_charge,
            fermi_temperature_kelvin=500.0,
            noncollinear=mode == "ncl",
            spin_polarized=mode == "spin",
            self_consistent_spin_orbit=True,
        ),
        hartree=HartreeSettings(multipole_order=4),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-7),
    )


def _smooth_spinors(system, centers: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Localized complex spinors (smooth in the atom positions, unlike white noise)."""

    generator = np.random.default_rng(seed)
    coordinates = system.grid.coordinates

    def bump(center: np.ndarray, width: float = 1.6) -> np.ndarray:
        d = coordinates - center
        r2 = np.sum(d * d, axis=1)
        poly = (
            1.0
            + generator.normal() * d[:, 0]
            + 0.5 * generator.normal() * d[:, 1]
            + 0.7 * generator.normal() * d[:, 2]
            + 0.3 * generator.normal() * d[:, 0] * d[:, 1]
            + 0.2 * generator.normal() * (d[:, 2] ** 2 - r2 / 3.0)
        )
        return np.exp(-r2 / (2.0 * width * width)) * poly

    columns = []
    for _ in range(count):
        up = (bump(centers[0]) + 0.6 * bump(centers[1])) * complex(generator.normal(), generator.normal())
        down = complex(generator.normal(), generator.normal()) * bump(centers[0]) + bump(centers[1]) * complex(
            generator.normal(), generator.normal()
        )
        vector = np.concatenate([up, down])
        columns.append(vector / np.linalg.norm(vector))
    return np.array(columns).T


class SpinOrbitProjectorGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = _cluster(mode="soc")
        cls.system = prepare_single_point(cls.problem)
        cls.positions = np.array([atom.position for atom in cls.system.atoms])  # the cluster is recentered

    def _projectors(self, positions: np.ndarray):
        system = update_geometry(self.system, [Atom("Au", position) for position in positions])
        return system, build_spin_orbit_projectors(
            system.grid, system.atoms, system.pseudopotentials, self.problem.pseudopotentials
        )

    def test_harmonic_values_and_gradients(self) -> None:
        generator = np.random.default_rng(1)
        directions = generator.normal(size=(40, 3))
        radii = generator.uniform(0.5, 2.0, size=40)
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        for l in (1, 2):
            values, gradients = complex_harmonics_with_gradients(l, directions, 1.0 / radii)
            np.testing.assert_allclose(values, complex_spherical_harmonics(l, directions), atol=1.0e-13)
            vectors = directions * radii[:, None]  # Y depends on the direction only: gradient = tangential, over |d|
            step = 1.0e-6
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = step
                numeric = (
                    complex_spherical_harmonics(l, vectors + shift) - complex_spherical_harmonics(l, vectors - shift)
                ) / (2.0 * step)
                np.testing.assert_allclose(gradients[:, :, axis], numeric, atol=2.0e-7)

    def test_the_gradients_are_the_derivatives_of_the_projectors(self) -> None:
        system, projectors = self._projectors(self.positions)
        gradients = build_spin_orbit_projector_gradients(
            system.grid, system.atoms, system.pseudopotentials, self.problem.pseudopotentials, projectors
        )
        step = 1.0e-5
        for atom_index in (0, 1):
            for axis in range(3):
                plus, minus = self.positions.copy(), self.positions.copy()
                plus[atom_index, axis] += step
                minus[atom_index, axis] -= step
                dense = []
                for shifted in (plus, minus):
                    _, moved = self._projectors(shifted)
                    full = np.zeros((system.grid.size, 16), dtype=complex)
                    full[moved[atom_index].support_rows] = moved[atom_index].stacked
                    dense.append(full)
                # dP/dR = -grad_r P
                numeric = -(dense[0] - dense[1]) / (2.0 * step)
                analytic = np.zeros((system.grid.size, 16), dtype=complex)
                analytic[gradients[atom_index].support_rows] = gradients[atom_index].components[axis]
                scale = float(np.max(np.abs(analytic)))
                self.assertGreater(scale, 0.5)
                # a point that enters or leaves the support within the step (a jump) or a knot of the table in between
                # (a kink) spoils a few entries, not the bulk
                close = np.abs(analytic - numeric) < 1.0e-3 * scale
                self.assertGreater(float(np.mean(close[np.abs(analytic) > 0])), 0.995)

    def test_the_coefficient_map_is_hermitian_and_is_the_hamiltonian(self) -> None:
        for signs in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            row = np.array([[signs[0]] * 3 + [signs[1]] * 5])
            matrix = np.zeros((32, 32), dtype=complex)
            for column in range(32):
                unit = np.zeros((2, 1, 16, 1), dtype=complex)
                unit.flat[column] = 1.0
                matrix[:, column] = spin_orbit_coefficients(np, unit, row).reshape(-1)
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=1.0e-14)
        # the energy of the bilinear form is that of the operator the eigensolver applies
        system, projectors = self._projectors(self.positions)
        operator = build_spin_orbit_force_operator(
            system.grid, system.atoms, system.pseudopotentials, self.problem.pseudopotentials, projectors
        )
        psi = _smooth_spinors(system, self.positions, 3, seed=2)
        size = system.grid.size
        overlaps = np.stack((operator.overlap @ psi[:size], operator.overlap @ psi[size:])).reshape(2, 2, 16, 3)
        coefficients = spin_orbit_coefficients(np, overlaps, operator.signs)
        bilinear = np.einsum("sajn,sajn->n", overlaps.conj(), coefficients)
        direct = np.einsum("in,in->n", psi.conj(), apply_spin_orbit(projectors, psi))
        np.testing.assert_allclose(bilinear, direct, atol=1.0e-12)
        self.assertGreater(float(np.max(np.abs(direct))), 1.0e-3)

    def test_the_force_is_the_derivative_of_the_spin_orbit_energy(self) -> None:
        system, projectors = self._projectors(self.positions)
        psi = _smooth_spinors(system, self.positions, 4, seed=3)
        weights = np.array([1.0, 0.7, 0.4, 0.2])
        operator = build_spin_orbit_force_operator(
            system.grid, system.atoms, system.pseudopotentials, self.problem.pseudopotentials, projectors
        )
        forces = spin_orbit_forces(operator, psi, weights, 2)

        def energy(positions: np.ndarray) -> float:
            _, moved = self._projectors(positions)
            return float(np.real(np.einsum("in,in,n->", psi.conj(), apply_spin_orbit(moved, psi), weights)))

        self.assertGreater(float(np.max(np.abs(forces))), 5.0e-3)
        step = 1.0e-4  # the radial tables are piecewise linear: the finite difference needs a short step
        for atom, axis in ((0, 0), (0, 2), (1, 1), (1, 2)):
            plus, minus = self.positions.copy(), self.positions.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numeric = -(energy(plus) - energy(minus)) / (2.0 * step)
            self.assertAlmostEqual(float(forces[atom, axis]), numeric, delta=3.0e-5)

    def test_the_contraction_uses_only_array_module_operations(self) -> None:
        # CuPy cannot be run here: the contraction runs on a module with only CuPy's operations, over several blocks
        system, projectors = self._projectors(self.positions)
        psi = _smooth_spinors(system, self.positions, 5, seed=4)
        weights = np.array([1.0, 0.0, 0.6, 0.4, 0.2])
        operator = build_spin_orbit_force_operator(
            system.grid, system.atoms, system.pseudopotentials, self.problem.pseudopotentials, projectors
        )
        allowed = {"asarray", "ascontiguousarray", "complex128", "stack", "conj", "einsum", "empty_like", "matmul"}

        class Restricted:
            def __getattr__(self, name):
                if name not in allowed:
                    raise AttributeError(f"{name} is not on the list of CuPy operations of the spin-orbit force")
                return getattr(np, name)

        host = spin_orbit_forces(operator, psi, weights, 2)
        block = 16 * (2 * system.grid.size + 2 * 16 * 2 * 5) * 2  # two states per block
        device = spin_orbit_forces(operator, psi, weights, 2, xp=Restricted(), sparse=sp, block_bytes=block)
        self.assertGreater(float(np.max(np.abs(host))), 1.0e-3)
        np.testing.assert_allclose(device, host, atol=1.0e-12)


class IsolatedSpinorForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = _cluster(mode="ncl")
        cls.system = prepare_single_point(cls.problem)
        cls.positions = np.array([atom.position for atom in cls.system.atoms])
        cls.backend = spinor_backend()
        cls.result = cls.backend.run_scf(cls.system)
        cls.forces = cls.backend.compute_forces(cls.system, cls.result)

    def _moved(self, positions: np.ndarray):
        return update_geometry(self.system, [replace(a, position=p) for a, p in zip(self.system.atoms, positions)])

    def test_the_run_is_magnetic_and_converged(self) -> None:
        self.assertTrue(self.result.converged)
        self.assertGreater(self.result.magnetic_moment, 0.5)
        self.assertGreater(float(np.max(np.abs(self.forces.total))), 1.0)

    def test_each_force_term_is_the_derivative_of_its_energy(self) -> None:
        system, result, forces = self.system, self.result, self.forces
        size = system.grid.size
        weights = np.asarray(result.occupations)
        spinors = np.asarray(result.spinors)
        volume = system.grid.volume_element

        def terms(positions: np.ndarray) -> dict[str, float]:
            moved = self._moved(positions)
            nonlocal_energy = sum(
                float(weights @ np.einsum("ij,ij->j", part.conj(), _apply_nonlocal_complex(moved.nonlocal_operator, part)).real)
                for part in (spinors[:size], spinors[size:])
            )
            projectors = build_spin_orbit_projectors(
                moved.grid, moved.atoms, moved.pseudopotentials, self.problem.pseudopotentials
            )
            return {
                "ion_ion": moved.ion_ion_energy,
                "ionic": volume * float(result.density @ moved.ionic_potential),
                "nonlocal": nonlocal_energy
                + float(weights @ np.einsum("ij,ij->j", spinors.conj(), apply_spin_orbit(projectors, spinors)).real),
            }

        reference = {
            "ion_ion": forces.ion_ion,
            "ionic": forces.local.ionic,
            "nonlocal": forces.nonlocal_pseudopotential,
        }
        tolerance = {"ion_ion": 1.0e-8, "ionic": 1.0e-5, "nonlocal": 3.0e-4}
        for atom, axis in ((0, 0), (1, 2), (0, 1)):

            def central(step: float) -> dict[str, float]:
                plus, minus = self.positions.copy(), self.positions.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                a, b = terms(plus), terms(minus)
                return {key: -(a[key] - b[key]) / (2.0 * step) for key in a}

            fine, coarse = central(0.0025), central(0.005)
            for key, expected in reference.items():
                numeric = (4.0 * fine[key] - coarse[key]) / 3.0
                with self.subTest(term=key, atom=atom, axis=axis):
                    self.assertAlmostEqual(float(expected[atom, axis]), numeric, delta=tolerance[key])
        for key in reference:
            self.assertGreater(float(np.max(np.abs(reference[key]))), 0.1)

    def test_the_forces_are_the_derivative_of_the_converged_energy(self) -> None:
        # what a relaxation follows: the total energy of the SCF loop at displaced atoms
        atom, axis = 0, 1

        def energy(shift: float) -> float:
            positions = self.positions.copy()
            positions[atom, axis] += shift
            result = self.backend.run_scf(self._moved(positions))
            self.assertTrue(result.converged)
            return result.energies.total

        central = [-(energy(h) - energy(-h)) / (2.0 * h) for h in (0.01, 0.005)]
        extrapolated = (4.0 * central[1] - central[0]) / 3.0
        self.assertAlmostEqual(float(self.forces.unsymmetrized[atom, axis]), extrapolated, delta=5.0e-3)

    def test_the_net_force_is_removed(self) -> None:
        np.testing.assert_allclose(self.forces.total.sum(axis=0), 0.0, atol=1.0e-12)

    def test_a_looser_scf_criterion_is_honoured(self) -> None:
        # Relaxation_SCF_Upscale hands the loop of a structure a looser criterion; the system itself is not changed
        loose = self.backend.run_scf(self.system, convergence_criterion=1.0e-2)
        self.assertTrue(loose.converged)
        self.assertLess(loose.iterations, self.result.iterations)
        self.assertEqual(self.system.input.scf.convergence_criterion, 5.0e-5)

    def test_a_result_without_the_potentials_is_refused(self) -> None:
        bare = replace(self.result, xc_potential=None)
        with self.assertRaises(ValueError):
            compute_spinor_forces(self.system, bare)


class CollinearLimitTests(unittest.TestCase):
    def test_without_spin_orbit_the_noncollinear_forces_are_the_collinear_spin_forces(self) -> None:
        # the spin-polarized forces of the scalar driver were compared with Fortran PARSEC; a ground state of one unpaired
        # electron has the same energy and forces whichever way its moment points
        from parsec_python.Relaxation.backend import reference_backend

        base = _cluster(mode="ncl")
        general = replace(
            base,
            pseudopotentials={"Au": replace(base.pseudopotentials["Au"], spin_orbit=False)},
            scf=replace(base.scf, self_consistent_spin_orbit=False),
        )
        collinear = replace(
            general,
            atoms=[replace(atom, initial_moment=None) for atom in general.atoms],
            scf=replace(general.scf, noncollinear=False, spin_polarized=True),
        )
        spinor_system = prepare_single_point(general)
        spinor = spinor_backend()
        spinor_result = spinor.run_scf(spinor_system)
        spinor_forces = spinor.compute_forces(spinor_system, spinor_result)
        scalar_system = prepare_single_point(collinear)
        scalar = reference_backend()
        scalar_result = scalar.run_scf(scalar_system)
        scalar_forces = scalar.compute_forces(scalar_system, scalar_result, symmetrize=False)
        self.assertTrue(spinor_result.converged and scalar_result.converged)
        self.assertAlmostEqual(spinor_result.magnetic_moment, 1.0, delta=0.01)
        self.assertAlmostEqual(spinor_result.energies.total, scalar_result.energies.total, delta=1.0e-4)
        self.assertGreater(float(np.max(np.abs(scalar_forces.unsymmetrized))), 1.0)
        np.testing.assert_allclose(spinor_forces.unsymmetrized, scalar_forces.unsymmetrized, atol=1.0e-3)
        np.testing.assert_allclose(spinor_forces.nonlocal_pseudopotential, scalar_forces.nonlocal_pseudopotential, atol=1.0e-3)
        np.testing.assert_allclose(spinor_forces.local.ionic, scalar_forces.local.ionic, atol=1.0e-3)


def _periodic(mode: str = "ncl") -> SinglePointInput:
    return SinglePointInput(
        atoms=[Atom("Au", [3.1, 3.4, 3.9], [1, 0, 0]), Atom("Au", [6.0, 5.2, 4.7], [0, 1, 0])],
        pseudopotentials={
            "Au": SpeciesPotential(_AU, 0, use_spline=True, spin_orbit=True, initial_spin_polarization=0.5)
        },
        grid=PeriodicGridSettings(spacing=0.7, expansion_order=6),
        periodic_cell=PeriodicCell(lattice_vectors=9.0 * np.eye(3)),
        scf=SCFSettings(
            max_iterations=100,
            number_of_states=24,
            convergence_criterion=1.0e-4,
            net_charge=1.0,
            fermi_temperature_kelvin=500.0,
            noncollinear=mode == "ncl",
            spin_polarized=mode == "spin",
            self_consistent_spin_orbit=True,
        ),
        hartree=HartreeSettings(relative_tolerance=1e-10, absolute_tolerance=1e-13, max_iterations=4000),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1.0e-7),
    )


class PeriodicSpinorForceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from parsec_python.acceleration.V_ion import periodic_setup
        from parsec_python.V_ion import ewald
        from parsec_python.V_ion import ionic_potential as _ionic

        # the fast periodic setup gives the same potentials as the dense one in a second instead of minutes
        cls._saved = (ewald._local_potential_provider, _ionic._support_finder)
        periodic_setup.install()
        cls.system = prepare_periodic_single_point(_periodic())
        cls.positions = np.array([atom.position for atom in cls.system.atoms])

    @classmethod
    def tearDownClass(cls) -> None:
        from parsec_python.V_ion import ewald
        from parsec_python.V_ion import ionic_potential as _ionic

        ewald._local_potential_provider, _ionic._support_finder = cls._saved

    def _moved(self, shift: float, atom: int, axis: int):
        positions = self.positions.copy()
        positions[atom, axis] += shift
        return update_periodic_geometry(
            self.system, [replace(a, position=p) for a, p in zip(self.system.atoms, positions)]
        )

    def test_the_gamma_point_force_pass_equals_the_force_of_the_kept_orbitals(self) -> None:
        backend = spinor_backend()
        result = backend.run_scf(self.system)
        self.assertEqual(result.k_points.shape, (1, 3))
        system = self.system
        specifications = system.input.pseudopotentials
        atoms = system.cell_atoms
        lattice = system.input.periodic_cell.lattice_vectors
        from parsec_python.Forces import build_periodic_projector_gradients
        from parsec_python.V_ion import build_nonlocal_projectors

        operator = build_nonlocal_projectors(
            system.grid, atoms, system.pseudopotentials, specifications, lattice, k_point=np.zeros(3)
        )
        gradients = build_periodic_projector_gradients(
            system.grid, atoms, system.pseudopotentials, specifications, lattice, np.zeros(3)
        )
        projectors = build_spin_orbit_projectors(
            system.grid, atoms, system.pseudopotentials, specifications, lattice, k_point=np.zeros(3)
        )
        spin_orbit = build_spin_orbit_force_operator(
            system.grid, atoms, system.pseudopotentials, specifications, projectors, lattice, np.zeros(3)
        )
        expected = spinor_nonlocal_forces(
            operator, gradients, spin_orbit, np.asarray(result.spinors), np.asarray(result.occupations), len(atoms)
        )
        self.assertGreater(float(np.max(np.abs(expected))), 1.0)
        np.testing.assert_allclose(result.nonlocal_forces, expected, atol=1.0e-10)

    def test_the_k_point_forces_are_the_derivative_of_the_converged_energy(self) -> None:
        k_points, weights = monkhorst_pack_grid(self.system.input.periodic_cell, (2, 1, 1))
        backend = spinor_backend(k_points, weights)
        result = backend.run_scf(self.system)
        self.assertTrue(result.converged)
        forces = backend.compute_forces(self.system, result)
        atom, axis = 0, 1

        def energy(shift: float) -> float:
            run = backend.run_scf(self._moved(shift, atom, axis))
            self.assertTrue(run.converged)
            return run.energies.total

        central = [-(energy(h) - energy(-h)) / (2.0 * h) for h in (0.01, 0.005)]
        extrapolated = (4.0 * central[1] - central[0]) / 3.0
        self.assertGreater(abs(extrapolated), 1.0)
        # before the net force is removed: only the derivative of the energy of this atom
        self.assertAlmostEqual(float(forces.unsymmetrized[atom, axis]), extrapolated, delta=5.0e-3)

    def test_a_periodic_cell_is_not_symmetrized(self) -> None:
        backend = spinor_backend()
        result = backend.run_scf(self.system)
        with self.assertRaises(ValueError):
            backend.compute_forces(self.system, result, symmetrize=True)
        compute_periodic_forces(self.system, result)


_CLUSTER_INPUT = """\
Boundary_Sphere_Radius: 8.0
Grid_Spacing: 0.7
Expansion_Order: 6

States_Num: 24
Net_Charges: 1
Fermi_Temp: 500.0
Max_Iter: 80
Convergence_Criterion: 1.0e-4

Mixing_Method: Anderson
Mixing_Param: 0.25
Eigensolver: chebff

Non_Collinear_magnetism: .true.
SO_from_scratch: .true.
Cubic_Spline: .true.
{ionic}
Atom_Types_Num: 1
Total_Atom_Num: 2

Atom_Type: Au
SO_PSP: .true.
Local_Component: s
Initial_Spin_Polarization: 0.5
Begin atom_coord
  -2.31 0.17 0.22
   2.43 -0.11 0.13
End atom_coord
begin Initial_NCL_Moment
1 0 0
0 1 0
end Initial_NCL_Moment
{extra}
Correlation_Type: CA
"""


class SpinorIonicCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        shutil.copy(_AU, self.directory / "Au_POTRE.DAT")

    def _write(self, ionic: str, extra: str = "", *, periodic: bool = False) -> Path:
        text = _CLUSTER_INPUT.format(ionic=ionic, extra=extra)
        if periodic:
            text = text.replace(
                "Boundary_Sphere_Radius: 8.0\n",
                "Periodic_System: .true.\nBoundary_Conditions: bulk\nbegin Cell_Shape\n9.0 9.0 9.0\nend Cell_Shape\n",
            )
            text = text.replace("-2.31 0.17 0.22\n   2.43 -0.11 0.13", "3.1 3.4 3.9\n   6.0 5.2 4.7")
        path = self.directory / "parsec.in"
        path.write_text(text, encoding="utf-8")
        return path

    def _run(self, ionic: str, extra: str = "", *, periodic: bool = False) -> tuple[int, str]:
        code = cli_main([str(self._write(ionic, extra, periodic=periodic)), "--quiet"])
        return code, (self.directory / "parsec.out").read_text(encoding="utf-8")

    def test_a_single_point_with_forces_reports_and_archives_them(self) -> None:
        code, log = self._run("Skip_Force: .false.\n")
        self.assertEqual(code, 0)
        self.assertIn("Reporting non-local force components", log)
        self.assertIn("Converged magnetic moment vector", log)
        self.assertIn("The forces of a spin-polarized or non-collinear spin-orbit calculation are not symmetrized", log)
        archive = np.load(self.directory / "parsec_python_results.npz")
        forces = archive["forces_total"]
        self.assertEqual(forces.shape, (2, 3))
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1.0e-9)
        self.assertGreater(float(np.max(np.abs(archive["forces_nonlocal"]))), 1.0)

    def test_a_relaxation_moves_the_atoms_and_keeps_their_moments(self) -> None:
        code, log = self._run("Skip_Force: .false.\nMinimization: bfgs\nMovement_Num: 1\nForce_Min: 0.001\n")
        self.assertEqual(code, 4)  # one movement is not enough to relax it
        self.assertIn("the SCF loop starts afresh", log)
        self.assertNotIn("the SCF loop continues from", log)
        # the second structure is magnetic too: the moved atoms keep the direction of their initial moments
        self.assertEqual(log.count("Converged magnetic moment vector"), 2)
        archive = np.load(self.directory / "parsec_python_results.npz")
        positions = archive["relaxation_positions"]
        self.assertEqual(positions.shape, (2, 2, 3))
        self.assertGreater(float(np.max(np.abs(positions[1] - positions[0]))), 1.0e-3)
        for line in log.splitlines():
            if line.startswith("Converged magnetic moment vector"):
                self.assertGreater(float(line.split("|m| =")[1]), 0.5)

    def test_molecular_dynamics_runs_on_the_spinors(self) -> None:
        code, log = self._run(
            "Molecular_Dynamics: .true.\nTinit: 300\nT_Final: 200\nT_Step: 10\nTime_Step: 20\nStep_Num: 2\nRng_Seed: 7\n"
        )
        self.assertEqual(code, 0, log[-1500:])
        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertEqual(archive["dynamics_positions"].shape[1:], (2, 3))
        self.assertGreaterEqual(archive["dynamics_positions"].shape[0], 3)
        energies = archive["dynamics_energies"]
        self.assertTrue(np.all(np.isfinite(energies)))
        self.assertEqual(log.count("Converged magnetic moment vector"), archive["dynamics_positions"].shape[0])

    def test_a_periodic_cell_with_k_points_has_forces(self) -> None:
        code, log = self._run(
            "Skip_Force: .false.\n",
            "Kpoint_Method: mp\nbegin Monkhorst_Pack_Grid\n2 1 1\nend Monkhorst_Pack_Grid\n",
            periodic=True,
        )
        self.assertEqual(code, 0, log[-2000:])
        archive = np.load(self.directory / "parsec_python_results.npz")
        self.assertEqual(archive["forces_total"].shape, (2, 3))
        self.assertGreater(float(np.max(np.abs(archive["forces_total"]))), 0.1)

    def test_energy_refinement_is_still_refused(self) -> None:
        path = self._write("Skip_Force: .false.\nEnergy_Refinement: b3lyp\n")
        self.assertEqual(cli_main([str(path), "--quiet"]), 2)  # refused before the run starts


if __name__ == "__main__":
    unittest.main()
