"""Forces of the spin-orbit and non-collinear drivers (``soforc.F90``).

The spin-orbit term is checked against central differences of the spin-orbit
energy of fixed spinors (isolated, periodic images, a k-point), the complete
forces of periodic spin-orbit and non-collinear SCF loops against central
differences of their free energy, and the command line (single point,
relaxation, refusal of the perturbative correction, accelerated routing).
The Fortran comparison (whose spin-orbit term is not the derivative of its
energy) is in the report of the parity round.
"""


# With both sets of routines: the periodic grids of this module are those of the Fortran-derived routines
# (PeriodicGridSettings(fortran_routines=True), as Use_Fortran_Routines: .true. gives); the numbers checked
# here are those of the Fortran parity round.
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parsec_python.Eigensolvers.perturbative_soc import (
    build_spin_orbit_projectors,
    complex_spherical_harmonics,
)
from parsec_python.Forces import compute_forces
from parsec_python.Forces.spin_orbit_fortran import (
    build_spin_orbit_projector_gradients,
    complex_harmonics_with_gradients,
    spin_orbit_energy,
    spin_orbit_forces,
)
from parsec_python.Grid.pbc import build_periodic_grid
from parsec_python.models import (
    Atom,
    EigensolverSettings,
    HartreeSettings,
    PeriodicCell,
    PeriodicGridSettings,
    SCFSettings,
    SinglePointInput,
    SpeciesPotential,
)
from parsec_python.SCF import prepare_periodic_single_point
from parsec_python.SCF.kpoints_soc import run_self_consistent_soc_kpoints
from parsec_python.SCF.noncollinear import run_self_consistent_noncollinear_kpoints
from parsec_python.SCF.pbc import update_periodic_geometry
from parsec_python.V_ion import load_pseudopotentials

AUH = Path(__file__).resolve().parents[4] / "examples" / "0d_AuH"
SPECIES = {
    "Au": SpeciesPotential(AUH / "Au_POTRE.DAT", 0, spin_orbit=True),
    "H": SpeciesPotential(AUH / "H_POTRE.DAT", 0),
}
ATOMS = (Atom("Au", np.array([3.4, 3.1, 3.3])), Atom("H", np.array([0.6, 3.5, 2.9])))


def _moved(atoms, index, axis, step):
    out = list(atoms)
    position = np.array(out[index].position, dtype=float)
    position[axis] += step
    out[index] = Atom(out[index].symbol, position)
    return tuple(out)


class HarmonicTests(unittest.TestCase):
    def test_gradients_of_the_complex_harmonics(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.normal(size=(6, 3))
        step = 1e-6
        for l in (1, 2):
            radius = np.linalg.norm(points, axis=1)
            values, gradients = complex_harmonics_with_gradients(
                l, points / radius[:, None], 1.0 / radius
            )
            np.testing.assert_allclose(values, complex_spherical_harmonics(l, points), atol=1e-14)
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = step
                numeric = (
                    complex_spherical_harmonics(l, points + shift)
                    - complex_spherical_harmonics(l, points - shift)
                ) / (2 * step)
                np.testing.assert_allclose(gradients[:, :, axis], numeric, atol=1e-8)


class SpinOrbitTermTests(unittest.TestCase):
    """The spin-orbit force of fixed spinors is the derivative of their energy."""

    def test_isolated_periodic_and_kpoint(self) -> None:
        cell = PeriodicCell(lattice_vectors=6.5 * np.eye(3))
        grid = build_periodic_grid(cell, PeriodicGridSettings(fortran_routines=True, spacing=0.5, expansion_order=4))
        potentials = load_pseudopotentials(SPECIES)
        rng = np.random.default_rng(1)
        spinors = rng.normal(size=(2 * grid.size, 4)) + 1j * rng.normal(size=(2 * grid.size, 4))
        spinors /= np.linalg.norm(spinors, axis=0)
        occupations = np.array([1.0, 0.7, 0.3, 0.05])
        step = 1e-6
        for lattice, k_point in (
            (None, None),
            (cell.lattice_vectors, None),
            (cell.lattice_vectors, np.array([0.2, -0.1, 0.3])),
        ):
            projectors = build_spin_orbit_projectors(
                grid, ATOMS, potentials, SPECIES, lattice, k_point
            )
            gradients = build_spin_orbit_projector_gradients(
                grid, ATOMS, potentials, SPECIES, lattice, k_point
            )
            forces = spin_orbit_forces(projectors, gradients, spinors, occupations, 2)
            np.testing.assert_array_equal(forces[1], 0.0)
            for axis in range(3):
                energies = [
                    spin_orbit_energy(
                        build_spin_orbit_projectors(
                            grid, _moved(ATOMS, 0, axis, sign * step), potentials, SPECIES,
                            lattice, k_point,
                        ),
                        spinors,
                        occupations,
                    )
                    for sign in (1.0, -1.0)
                ]
                numeric = -(energies[0] - energies[1]) / (2 * step)
                self.assertAlmostEqual(forces[0, axis], numeric, delta=1e-11)


def _periodic_problem(noncollinear=False):
    return SinglePointInput(
        atoms=list(ATOMS),
        pseudopotentials=SPECIES,
        grid=PeriodicGridSettings(fortran_routines=True, spacing=0.55, expansion_order=4),
        periodic_cell=PeriodicCell(lattice_vectors=6.5 * np.eye(3)),
        scf=SCFSettings(
            max_iterations=100,
            number_of_states=16,
            convergence_criterion=1e-10,
            fermi_temperature_kelvin=300.0,
            self_consistent_spin_orbit=True,
            noncollinear=noncollinear,
        ),
        hartree=HartreeSettings(
            relative_tolerance=1e-12, absolute_tolerance=1e-14, max_iterations=4000
        ),
        eigensolver=EigensolverSettings(method="chebff", tolerance=1e-10),
    )


def _free_energy(result, temperature=300.0) -> float:
    f = np.clip(np.asarray(result.occupations), 1e-300, 1 - 1e-16)
    weights = np.repeat(result.k_weights, f.size // len(result.k_weights))
    entropy = np.sum(weights * (f * np.log(f) + (1 - f) * np.log(1 - f)))
    return float(result.energies.total + 6.33327186e-6 * temperature * entropy)


class SelfConsistentForceTests(unittest.TestCase):
    """Periodic spinor SCF forces against the free energy (step 2e-6 bohr)."""

    K_POINTS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 6.5]])
    WEIGHTS = np.array([0.5, 0.5])

    def check(self, driver, problem, components) -> None:
        system = prepare_periodic_single_point(problem)
        result = driver(system, self.K_POINTS, self.WEIGHTS, keep_wavefunctions=True)
        self.assertTrue(result.converged)
        self.assertEqual(len(result.k_spinors), 2)
        forces = compute_forces(system, result, symmetrize=False)
        self.assertTrue(forces.periodic)
        self.assertGreater(np.max(np.abs(forces.spin_orbit[0])), 1e-3)
        np.testing.assert_array_equal(forces.spin_orbit[1], 0.0)
        step = 2e-6
        for index, axis in components:
            energies = [
                _free_energy(
                    driver(
                        update_periodic_geometry(system, _moved(ATOMS, index, axis, sign * step)),
                        self.K_POINTS,
                        self.WEIGHTS,
                    )
                )
                for sign in (1.0, -1.0)
            ]
            numeric = -(energies[0] - energies[1]) / (2 * step)
            self.assertAlmostEqual(forces.unsymmetrized[index, axis], numeric, delta=2e-8)

    def test_spin_orbit_kpoints(self) -> None:
        self.check(run_self_consistent_soc_kpoints, _periodic_problem(), ((0, 0),))

    def test_noncollinear_kpoints(self) -> None:
        self.check(
            run_self_consistent_noncollinear_kpoints, _periodic_problem(True), ((1, 1),)
        )


class CommandLineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        source = AUH / "reduced_self_consistent_soc"
        for name in ("Au_POTRE.DAT", "H_POTRE.DAT"):
            shutil.copy(source / name, self.directory / name)
        # With both sets of routines: the spinor forces checked here are those of the Fortran-derived routines.
        self.base = (source / "parsec.in").read_text(encoding="utf-8") + "\nUse_Fortran_Routines: .true.\n"

    def write(self, extra: str, base: str | None = None) -> Path:
        path = self.directory / "parsec.in"
        path.write_text((self.base if base is None else base) + extra, encoding="utf-8")
        return path

    def test_forces_and_relaxation(self) -> None:
        from parsec_python.cli import main

        self.assertEqual(main([str(self.write("Skip_force: .false.\n")), "--quiet"]), 0)
        text = (self.directory / "parsec.out").read_text(encoding="utf-8")
        self.assertIn(" Adding spin-orbit contribution to forces", text)
        self.assertIn(" Reporting spin-orbit force components", text)
        self.assertIn("tot. torque", text)
        with np.load(self.directory / "parsec_python_results.npz") as archive:
            self.assertEqual(archive["forces_spin_orbit"].shape, (2, 3))
            self.assertIn("eigenvalues_ry", archive.files)
        code = main(
            [
                str(self.write("Minimization: bfgs\nMovement_Num: 1\nForce_Min: 0.001\n")),
                "--quiet",
                "--no-archive",
            ]
        )
        self.assertIn(code, (0, 4))
        self.assertIn("Step #   1", (self.directory / "atom.cor").read_text(encoding="utf-8"))

    def test_perturbative_correction_is_refused_and_routing(self) -> None:
        from parsec_python.acceleration.cli import _reference_only_features
        from parsec_python.cli import main
        from parsec_python.Input.parsec_input import parse_parsec_input

        perturbative = self.base.replace("SO_from_scratch: .true.\n", "Spin_Polarization: .true.\n")
        self.assertEqual(main([str(self.write("Skip_force: .false.\n", perturbative)), "--quiet"]), 2)
        translation = parse_parsec_input(self.write("Molecular_Dynamics: .true.\n"))
        features = _reference_only_features(translation, accelerated_backend_selected=True)
        self.assertTrue(any("spin-orbit coupling or non-collinear" in item for item in features))


if __name__ == "__main__":
    unittest.main()
