"""The SCF loop of a meta-GGA (reference path): energy, spin, forces.

Independent references:

* the total energy of a converged loop is the functional evaluated directly
  from its orbitals (kinetic, nonlocal, ionic, Hartree, ``E_xc[n + n_core, tau]``):
  the double counting of the orbital term ``-div(vtau grad)`` is right;
* two equal spin channels give the unpolarized loop;
* the forces (with a nonlinear core correction) are the derivatives of the
  converged total energy (central differences of two displaced structures);
* the first iteration is PBE's, as PARSEC 1.5 starts its meta-GGAs.

The loops run the smooth test meta-GGA of ``test_meta_gga`` (it converges
like a GGA) and SCAN.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import parsec_python.V_xc.meta_gga as meta_gga_module
from parsec_python.Input import UnsupportedParsecOptionError, parse_parsec_input
from parsec_python.SCF import prepare_single_point, run_scf
from parsec_python.V_xc.meta_gga import (
    grid_gradient,
    kinetic_energy_density,
    meta_gga_evaluator,
    meta_gga_xc,
    orbital_density_gradient,
)
from parsec_python.models import MetaGGAEnergyBreakdown
from parsec_python.tests.unit.test_meta_gga import _LaplacianTestFunctional


# The pseudopotentials of the NCPPs/PBE library that the tests read, as examples/0d_meta_gga keeps them
# (src/pp_generation does not travel with the A100 requests).
FILES = Path(__file__).resolve().parents[4] / "examples" / "0d_meta_gga"

BASE = """Boundary_Conditions: cluster
Boundary_Sphere_Radius: {radius} bohr
Grid_Spacing: 0.5 bohr
Coordinate_Unit: Cartesian_Bohr
Expansion_Order: 8
Solver_Lpole: 4
Hartree_Atomic_Tail: off
States_Num: {states}
Fermi_Temp: 80
Max_Iter: 80
Convergence_Criterion: {criterion} Ry
Eigensolver: chebff
Mixing_Method: Anderson
Mixing_Param: 0.3
Correlation_Type: scan
Atom_Types_Num: {types}
{species}{extra}"""

H2 = """Atom_Type: H
Pseudopotential_Format: fhipp
Potential_Num: 1
Local_Component: s
Core_Cutoff_Radius: 1.3
begin Electron_Per_Orbital
1
end Electron_Per_Orbital
{initial}begin Atom_Coord
  {x} 0.0 -0.70
  0.0 0.0  0.70
end Atom_Coord
"""

LIH = """Atom_Type: Li
Pseudopotential_Format: martins_new
Local_Component: d
begin Atom_Coord
  {x} 0.0 -1.50
end Atom_Coord
Atom_Type: H
Pseudopotential_Format: martins_new
Local_Component: s
begin Atom_Coord
  0.0 0.0 1.50
end Atom_Coord
"""


class _SmoothTestFunctional(_LaplacianTestFunctional):
    """The test meta-GGA without its Laplacian term."""

    needs_laplacian = False

    def unpolarized(self, rho, sigma, tau, lapl):
        energy, vrho, vsigma, vtau, _ = super().unpolarized(rho, sigma, tau, np.zeros_like(rho))
        return energy, vrho, vsigma, vtau, None

    def polarized(self, *values):
        values = list(values)
        values[7] = np.zeros_like(values[0])
        values[8] = np.zeros_like(values[0])
        outputs = list(super().polarized(*values))
        outputs[8] = outputs[9] = None
        return tuple(outputs)


class _Inputs:
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="parsec_meta_gga_"))
        for name in ("H_FHIPP.DAT", "H_POTRE.DAT", "Li_POTRE.DAT"):
            shutil.copy(FILES / name, self.directory / name)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def system(self, *, molecule="H2", x=0.0, states=2, criterion=1.0e-8, extra="", initial="", name="parsec.in"):
        if molecule == "H2":
            species, types, radius = H2.format(x=x, initial=initial), 1, 5.0
        else:
            species, types, radius = LIH.format(x=x), 2, 6.0
        path = self.directory / name
        path.write_text(
            BASE.format(radius=radius, states=states, criterion=criterion, types=types, species=species, extra=extra)
        )
        return prepare_single_point(parse_parsec_input(path).problem)


def _smooth():
    return mock.patch.dict(meta_gga_module.NATIVE_META_GGA_FUNCTIONALS, {"scan": _SmoothTestFunctional})


def _direct_energy(system, result):
    """The meta-GGA total energy evaluated from the orbitals and density of ``result``."""

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
    tau = kinetic_energy_density(orbitals, occupations, grid_gradient(system.grid), volume)
    hartree = system.solve_hartree(density).potential
    gradient = None
    if result.meta_gga.gradient == "orbital":
        gradient = orbital_density_gradient(orbitals, occupations, grid_gradient(system.grid), volume)
    xc = meta_gga_xc(
        meta_gga_evaluator(system.input.scf), density, tau, system.grid, system.core_density, density_gradient=gradient
    )
    return (
        one_body
        + volume * float(density @ system.ionic_potential)
        + 0.5 * volume * float(density @ hartree)
        + xc.total_energy
        + system.ion_ion_energy
    )


class SmoothMetaGGALoopTests(_Inputs, unittest.TestCase):
    def test_total_energy_is_the_functional_of_the_orbitals(self) -> None:
        with _smooth():
            system = self.system()
            records = []
            result = run_scf(system, callback=records.append)
            self.assertTrue(result.converged)
            self.assertIsInstance(result.energies, MetaGGAEnergyBreakdown)
            # The first iteration has no orbital term (PBE, as PARSEC 1.5).
            self.assertEqual(records[0].energies.integral_vtau_tau, 0.0)
            self.assertGreater(result.energies.integral_vtau_tau, 0.0)
            self.assertAlmostEqual(result.energies.total, _direct_energy(system, result), delta=2.0e-7)
            record = result.meta_gga
            np.testing.assert_allclose(record.next_kinetic_potentials[0], record.kinetic_potentials[0], atol=1e-4)

    def test_two_equal_spin_channels_give_the_unpolarized_loop(self) -> None:
        with _smooth():
            unpolarized = run_scf(self.system())
            spin = run_scf(
                self.system(
                    extra="Spin_Polarization: .true.\n", initial="Initial_Spin_Polarization: 0.0\n", name="spin.in"
                )
            )
        self.assertTrue(spin.converged)
        self.assertAlmostEqual(spin.energies.total, unpolarized.energies.total, delta=2.0e-7)
        up, down = spin.spin
        np.testing.assert_allclose(up.eigenvalues, unpolarized.eigenvalues, atol=2.0e-6)
        np.testing.assert_allclose(down.eigenvalues, unpolarized.eigenvalues, atol=2.0e-6)
        np.testing.assert_allclose(
            spin.meta_gga.kinetic_potentials[0], unpolarized.meta_gga.kinetic_potentials[0], atol=1e-5
        )

    def test_core_correction_force_is_the_derivative_through_the_core_density(self) -> None:
        # E_xc[n_v + n_core(R), tau] at fixed valence density and tau: the core
        # term of the local forces with the meta-GGA's dE/dn is -dE/dR.  Li of
        # LiH has a nonlinear core correction in its POTRE file.
        from parsec_python.Forces import local_forces
        from parsec_python.V_ion import check_core_density, superpose_atomic_density as build_atomic_density
        from parsec_python.V_xc.scan import NativeScan

        system = self.system(molecule="LiH", x=0.15, states=3)
        grid = system.grid
        self.assertGreater(float(np.max(system.core_density)), 0.0)
        x = grid.coordinates - np.array([0.15, 0.0, 0.0])
        r2 = np.einsum("ij,ij->i", x, x)
        valence = 0.05 * np.exp(-0.3 * r2) + 0.02 * np.exp(-0.5 * np.sum((grid.coordinates - [0, 0, 1.5]) ** 2, axis=1))
        tau = 0.3 * valence ** (5.0 / 3.0) + 0.01 * valence
        functional = NativeScan()

        def core_of(position):
            atoms = list(system.atoms)
            atoms[0] = replace(atoms[0], position=tuple(position))
            density = build_atomic_density(grid, atoms, system.pseudopotentials, system.input.pseudopotentials, core=True)
            return check_core_density(density)

        # The orbital form: the gradient of the valence orbitals, fixed, plus
        # that of the core density; the force reads dE/dn_core (core_potential).
        from parsec_python.V_xc.meta_gga import grid_gradient

        orbital_gradient = 0.9 * grid_gradient(grid).apply(valence)
        position = np.asarray(system.atoms[0].position, dtype=float)
        for gradient in (None, orbital_gradient):

            def energy(position, gradient=gradient):
                return meta_gga_xc(
                    functional, valence, tau, grid, core_of(position), density_gradient=gradient
                ).total_energy

            xc = meta_gga_xc(functional, valence, tau, grid, system.core_density, density_gradient=gradient)
            potential = xc.potential if gradient is None else xc.core_potential
            terms = local_forces(
                grid, system.atoms, system.pseudopotentials, system.input.pseudopotentials,
                valence, np.zeros(grid.size), potential,
            )
            # Along the bond, where the term is large: the grid sampling of the
            # moved core density limits both sides to ~2e-4 relative (PBE alike:
            # 0.029876 against 0.029881 Ry/bohr in this geometry).
            shift = np.array([0.0, 0.0, 1.0e-3])
            derivative = (energy(position + shift) - energy(position - shift)) / (2.0 * shift[2])
            self.assertAlmostEqual(float(terms.core_correction[0, 2]), -derivative, delta=5.0e-4 * abs(derivative))


def _orbital_form():
    return mock.patch.dict("os.environ", {"PARSEC_META_GGA_GRADIENT": "orbital"})


def _density_form():
    return mock.patch.dict("os.environ", {"PARSEC_META_GGA_GRADIENT": "density"})


class OrbitalGradientLoopTests(_Inputs, unittest.TestCase):
    """SCAN with the density gradient of the orbitals: GGA-like convergence, energy, spin."""

    def test_scan_converges_and_is_the_functional_of_its_orbitals(self) -> None:
        with _orbital_form():
            system = self.system(criterion=1.0e-8)
            records = []
            result = run_scf(system, callback=records.append)
        self.assertTrue(result.converged)
        # H2 with the density stencil does not converge at this grid (see the module of SCF/meta_gga.py).
        self.assertLess(len(records), 30)
        self.assertEqual(result.meta_gga.gradient, "orbital")
        self.assertAlmostEqual(result.energies.total, _direct_energy(system, result), delta=2.0e-8)

    def test_two_equal_spin_channels_give_the_unpolarized_loop(self) -> None:
        with _orbital_form():
            unpolarized = run_scf(self.system(criterion=1.0e-8))
            spin = run_scf(
                self.system(
                    criterion=1.0e-8,
                    extra="Spin_Polarization: .true.\n",
                    initial="Initial_Spin_Polarization: 0.0\n",
                    name="spin.in",
                )
            )
        self.assertTrue(spin.converged)
        self.assertAlmostEqual(spin.energies.total, unpolarized.energies.total, delta=1.0e-10)
        np.testing.assert_allclose(spin.spin[0].eigenvalues, unpolarized.eigenvalues, atol=1e-8)


class ScanLoopTests(_Inputs, unittest.TestCase):
    def test_density_form_is_variational(self) -> None:
        # The density stencil form converges slowly on this coarse grid: a loose criterion.
        with _density_form():
            system = self.system(criterion=5.0e-2)
            result = run_scf(system)
        self.assertTrue(result.converged)
        self.assertEqual(result.meta_gga.gradient, "density")
        self.assertAlmostEqual(result.energies.total, _direct_energy(system, result), delta=1.0e-4)

    def test_refused_combinations(self) -> None:
        path = self.directory / "refused.in"
        path.write_text(
            BASE.format(radius=5.0, states=2, criterion=1e-6, types=1, species=H2.format(x=0.0, initial=""), extra="")
            .replace("Boundary_Conditions: cluster", "Boundary_Conditions: cluster\nPolarizability: .true.")
        )
        with self.assertRaises(UnsupportedParsecOptionError):
            parse_parsec_input(path)
        # A restart file holds V and no vtau or W.
        path.write_text(
            BASE.format(radius=5.0, states=2, criterion=1e-6, types=1, species=H2.format(x=0.0, initial=""), extra="")
            .replace("Boundary_Conditions: cluster", "Boundary_Conditions: cluster\nRestart_Run: .true.")
        )
        with self.assertRaisesRegex(UnsupportedParsecOptionError, "Restart_Run"):
            parse_parsec_input(path)


if __name__ == "__main__":
    unittest.main()
